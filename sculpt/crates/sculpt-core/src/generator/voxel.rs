//! SPIKE B (measurement only): sparse brick SDF around instances + naive surface nets.
//!
//! This is *not* the feature. It implements just enough of each stage to time it:
//!
//! 1. brick enumeration (bricks of 8^3 voxels overlapping instance AABBs + band),
//! 2. exact mesh signed distance per lattice sample (BVH closest point; sign from
//!    angle-weighted vertex pseudo-normals interpolated at the closest point;
//!    exact for smooth closed meshes, an approximation at sharp creases),
//! 3. combine with instance SDFs (`max(mesh, -min inst)` for subtract),
//! 4. naive surface nets inside each brick (cross-brick quads are not emitted,
//!    which undercounts quads by roughly 1/8 at 8^3 bricks).
//!
//! Because the full run at 100k instances is minutes of CPU, [`measure`] times
//! a deterministic, evenly strided sample of bricks and extrapolates linearly
//! by brick count whenever the brick count exceeds `sample_bricks`
//! (`Report::extrapolated`).

use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::instances::InstanceSet;
use crate::bvh::{Bvh, NodeKind};
use crate::geom::Aabb;
use crate::mesh::{Face, NO_VERT, Topology, face_len, face_verts};

const B: usize = 8; // voxels per brick edge
const S: usize = B + 1; // lattice samples per brick edge
const SAMPLES: usize = S * S * S;

#[derive(Clone, Debug)]
struct FNode {
    bounds: Aabb,
    start: u32,
    count: u32,
    right: u32,
}

#[derive(Clone, Debug, Default)]
struct FineTree {
    nodes: Vec<FNode>,
    faces: Vec<u32>,
}

/// Flattened closest-point tree for the GPU, see [`MeshSdf::flatten`].
pub struct FlatBvh {
    pub nodes: Vec<[u32; 8]>,
    /// Triangle vertex indices, 3 per triangle.
    pub tris: Vec<u32>,
}

impl FlatBvh {
    pub const LEAF: u32 = 0x8000_0000;
}

/// Closest-point / signed distance acceleration structure over a [`Bvh`]'s leaves.
pub struct MeshSdf<'a> {
    bvh: std::borrow::Cow<'a, Bvh>,
    positions: std::borrow::Cow<'a, [Vec3]>,
    faces: std::borrow::Cow<'a, [Face]>,
    trees: Vec<FineTree>,
    pseudo: Vec<Vec3>,
}

fn build_fine(t: &mut Vec<FNode>, order: &mut [u32], base: u32, c: &[Vec3]) {
    let id = t.len();
    t.push(FNode { bounds: Aabb::EMPTY, start: base, count: order.len() as u32, right: 0 });
    if order.len() <= 4 {
        return;
    }
    let mut cb = Aabb::EMPTY;
    for &f in order.iter() {
        cb.grow(c[f as usize]);
    }
    let axis = cb.extent().max_position();
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| c[*a as usize][axis].total_cmp(&c[*b as usize][axis]));
    let (lo, hi) = order.split_at_mut(mid);
    build_fine(t, lo, base, c);
    let right = t.len() as u32;
    build_fine(t, hi, base + mid as u32, c);
    t[id].count = 0;
    t[id].right = right;
}

#[inline]
fn aabb_dist_sq(b: &Aabb, p: Vec3) -> f32 {
    let d = (b.min - p).max(p - b.max).max(Vec3::ZERO);
    d.length_squared()
}

/// Closest point on triangle (Ericson) with barycentric weights.
#[inline]
fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let inv = 1.0 / (va + vb + vc);
    let (v, w) = (vb * inv, vc * inv);
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

impl<'a> MeshSdf<'a> {
    /// Build per-leaf fine trees (parallel) and angle-weighted pseudo-normals.
    pub fn build(bvh: &'a Bvh, positions: &'a [Vec3], faces: &'a [Face], topo: &Topology) -> MeshSdf<'a> {
        let trees: Vec<FineTree> = bvh
            .leaves
            .par_iter()
            .map(|l| {
                let c: Vec<Vec3> = l
                    .faces
                    .iter()
                    .map(|&fi| {
                        let vs = face_verts(&faces[fi as usize]);
                        vs.iter().map(|&v| positions[v as usize]).sum::<Vec3>() / vs.len() as f32
                    })
                    .collect();
                let mut order: Vec<u32> = (0..l.faces.len() as u32).collect();
                let mut nodes = Vec::new();
                build_fine(&mut nodes, &mut order, 0, &c);
                let fl: Vec<u32> = order.iter().map(|&i| l.faces[i as usize]).collect();
                // bounds, bottom-up (preorder => reverse sweep)
                for i in (0..nodes.len()).rev() {
                    let n = nodes[i].clone();
                    nodes[i].bounds = if n.count == 0 {
                        nodes[i + 1].bounds.union(&nodes[n.right as usize].bounds)
                    } else {
                        let mut b = Aabb::EMPTY;
                        for &fi in &fl[n.start as usize..(n.start + n.count) as usize] {
                            for &v in face_verts(&faces[fi as usize]) {
                                b.grow(positions[v as usize]);
                            }
                        }
                        b
                    };
                }
                FineTree { nodes, faces: fl }
            })
            .collect();
        let pseudo: Vec<Vec3> = (0..positions.len())
            .into_par_iter()
            .map(|v| {
                let mut n = Vec3::ZERO;
                for &fi in topo.vert_faces.row(v) {
                    let f = &faces[fi as usize];
                    let len = face_len(f);
                    let k = f[..len].iter().position(|&x| x as usize == v).unwrap();
                    let p = positions[v];
                    let e0 = (positions[f[(k + 1) % len] as usize] - p).normalize_or_zero();
                    let e1 = (positions[f[(k + len - 1) % len] as usize] - p).normalize_or_zero();
                    let fnrm = crate::mesh::face_normal(positions, f).normalize_or_zero();
                    n += fnrm * e0.dot(e1).clamp(-1.0, 1.0).acos();
                }
                n.normalize_or(Vec3::Y)
            })
            .collect();
        MeshSdf { bvh: std::borrow::Cow::Borrowed(bvh), positions: std::borrow::Cow::Borrowed(positions), faces: std::borrow::Cow::Borrowed(faces), trees, pseudo }
    }

    /// Like [`Self::build`], but the structure keeps its own copy of the mesh, so it can be cached past the mesh's owner.
    pub fn build_owned(bvh: Bvh, positions: Vec<Vec3>, faces: Vec<Face>, topo: &Topology) -> MeshSdf<'static> {
        let (trees, pseudo) = {
            let built = MeshSdf::build(&bvh, &positions, &faces, topo);
            (built.trees, built.pseudo)
        };
        MeshSdf { bvh: std::borrow::Cow::Owned(bvh), positions: std::borrow::Cow::Owned(positions), faces: std::borrow::Cow::Owned(faces), trees, pseudo }
    }

    /// Per-vertex angle-weighted pseudo-normals (the sign oracle of [`Self::signed_distance`]).
    pub fn pseudo_normals(&self) -> &[Vec3] {
        &self.pseudo
    }

    /// One flat binary tree for GPU upload: the engine BVH's inner nodes followed
    /// by every leaf's fine tree, with leaves holding ranges of a triangle list
    /// (quads split `[0,1,2]`, `[0,2,3]`, as in the CPU query).
    ///
    /// Node layout (8 words): `min.xyz` (f32 bits), `a`, `max.xyz` (f32 bits), `b`.
    /// Inner node: `a` = left child, `b` = right child. Leaf: `a` = first triangle,
    /// `b` = [`FlatBvh::LEAF`] | triangle count. Root is node 0.
    pub fn flatten(&self) -> FlatBvh {
        let core = &self.bvh.nodes;
        let mut nodes: Vec<[u32; 8]> = vec![[0; 8]; core.len()];
        let mut tris: Vec<u32> = Vec::new();
        let put = |nodes: &mut Vec<[u32; 8]>, i: usize, b: &Aabb, a: u32, bb: u32| {
            nodes[i] = [b.min.x.to_bits(), b.min.y.to_bits(), b.min.z.to_bits(), a, b.max.x.to_bits(), b.max.y.to_bits(), b.max.z.to_bits(), bb];
        };
        for (n, node) in core.iter().enumerate() {
            match node.kind {
                NodeKind::Inner { left, right } => put(&mut nodes, n, &node.bounds, left, right),
                NodeKind::Leaf { leaf } => {
                    let t = &self.trees[leaf as usize];
                    let base = nodes.len();
                    let map = |j: u32| if j == 0 { n } else { base + j as usize - 1 };
                    nodes.resize(base + t.nodes.len() - 1, [0; 8]);
                    for (j, fnode) in t.nodes.iter().enumerate() {
                        let id = map(j as u32);
                        if fnode.count == 0 {
                            put(&mut nodes, id, &fnode.bounds, map(j as u32 + 1) as u32, map(fnode.right) as u32);
                        } else {
                            let start = tris.len() as u32 / 3;
                            for &fi in &t.faces[fnode.start as usize..(fnode.start + fnode.count) as usize] {
                                let f = &self.faces[fi as usize];
                                tris.extend_from_slice(&[f[0], f[1], f[2]]);
                                if f[3] != NO_VERT {
                                    tris.extend_from_slice(&[f[0], f[2], f[3]]);
                                }
                            }
                            put(&mut nodes, id, &fnode.bounds, start, FlatBvh::LEAF | (tris.len() as u32 / 3 - start));
                        }
                    }
                }
            }
        }
        FlatBvh { nodes, tris }
    }

    pub fn memory_bytes(&self) -> usize {
        self.trees.iter().map(|t| t.nodes.len() * std::mem::size_of::<FNode>() + t.faces.len() * 4).sum::<usize>() + self.pseudo.len() * 12
    }

    /// Signed distance (negative inside) to the mesh, searching only within
    /// `bound` of `p`; returns `bound` (positive) when nothing is closer.
    pub fn signed_distance(&self, p: Vec3, bound: f32) -> f32 {
        let mut best = bound * bound;
        let mut hit: Option<(Vec3, [u32; 3], [f32; 3])> = None;
        let mut stack: smallvec::SmallVec<[(f32, u32); 64]> = smallvec::smallvec![(0.0, 0)];
        let nodes = &self.bvh.nodes;
        while let Some((d, n)) = stack.pop() {
            if d >= best {
                continue;
            }
            let node = &nodes[n as usize];
            match node.kind {
                NodeKind::Inner { left, right } => {
                    let dl = aabb_dist_sq(&nodes[left as usize].bounds, p);
                    let dr = aabb_dist_sq(&nodes[right as usize].bounds, p);
                    let (near, far) = if dl <= dr { ((dl, left), (dr, right)) } else { ((dr, right), (dl, left)) };
                    if far.0 < best {
                        stack.push(far);
                    }
                    if near.0 < best {
                        stack.push(near);
                    }
                }
                NodeKind::Leaf { leaf } => self.leaf_closest(leaf as usize, p, &mut best, &mut hit),
            }
        }
        match hit {
            None => bound,
            Some((cp, vs, w)) => {
                let n = self.pseudo[vs[0] as usize] * w[0] + self.pseudo[vs[1] as usize] * w[1] + self.pseudo[vs[2] as usize] * w[2];
                let d = best.sqrt();
                if (p - cp).dot(n) < 0.0 { -d } else { d }
            }
        }
    }

    fn leaf_closest(&self, leaf: usize, p: Vec3, best: &mut f32, hit: &mut Option<(Vec3, [u32; 3], [f32; 3])>) {
        let t = &self.trees[leaf];
        let mut stack: smallvec::SmallVec<[(f32, u32); 32]> = smallvec::smallvec![(0.0, 0)];
        while let Some((d, i)) = stack.pop() {
            if d >= *best {
                continue;
            }
            let node = &t.nodes[i as usize];
            if node.count == 0 {
                let dl = aabb_dist_sq(&t.nodes[i as usize + 1].bounds, p);
                let dr = aabb_dist_sq(&t.nodes[node.right as usize].bounds, p);
                let (near, far) = if dl <= dr { ((dl, i + 1), (dr, node.right)) } else { ((dr, node.right), (dl, i + 1)) };
                if far.0 < *best {
                    stack.push(far);
                }
                if near.0 < *best {
                    stack.push(near);
                }
                continue;
            }
            for &fi in &t.faces[node.start as usize..(node.start + node.count) as usize] {
                let f = &self.faces[fi as usize];
                let mut tris = [[f[0], f[1], f[2]], [0; 3]];
                let nt = if f[3] != NO_VERT {
                    tris[1] = [f[0], f[2], f[3]];
                    2
                } else {
                    1
                };
                for tri in &tris[..nt] {
                    let (cp, w) = closest_on_tri(p, self.positions[tri[0] as usize], self.positions[tri[1] as usize], self.positions[tri[2] as usize]);
                    let d2 = cp.distance_squared(p);
                    if d2 < *best {
                        *best = d2;
                        *hit = Some((cp, *tri, w));
                    }
                }
            }
        }
    }
}

#[inline]
fn pack(x: i32, y: i32, z: i32) -> u64 {
    ((x as u64) & 0x1F_FFFF) | ((y as u64) & 0x1F_FFFF) << 21 | ((z as u64) & 0x1F_FFFF) << 42
}
#[inline]
fn unpack(k: u64) -> [i32; 3] {
    let f = |v: u64| ((v & 0x1F_FFFF) as i32) << 11 >> 11; // sign-extend 21 bits
    [f(k), f(k >> 21), f(k >> 42)]
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub voxel_size: f32,
    pub bricks_with_duplicates: usize,
    pub bricks: usize,
    pub bricks_near: usize,
    pub bricks_far: usize,
    pub voxels: usize,
    pub sample_bricks: usize,
    pub extrapolated: bool,
    pub ms_bricks: f64,
    pub ms_classify: f64,
    pub ms_mesh_sdf: f64,
    pub ms_combine: f64,
    pub ms_nets: f64,
    pub net_vertices: usize,
    pub net_quads: usize,
    pub brick_memory_bytes: usize,
}

impl Report {
    pub fn ms_total(&self) -> f64 {
        self.ms_bricks + self.ms_classify + self.ms_mesh_sdf + self.ms_combine + self.ms_nets
    }
}

/// Time the voxel-boolean pipeline (subtract) around `set`'s instances.
/// `set` should have been built with margin >= the band (2 voxels).
pub fn measure(mesh: &MeshSdf, set: &InstanceSet, voxel_size: f32, sample_bricks: usize) -> Report {
    let vs = voxel_size;
    let band = 2.0 * vs;
    let bsz = B as f32 * vs;
    let mesh_bounds = mesh.bvh.bounds();
    let origin = mesh_bounds.min - Vec3::splat(4.0 * bsz);
    let mut r = Report { voxel_size: vs, ..Default::default() };

    // 1. Brick enumeration.
    let t = Instant::now();
    let inv = 1.0 / bsz;
    let mut keys: Vec<u64> = set
        .instances()
        .par_iter()
        .flat_map_iter(|inst| {
            let a = inst.aabb();
            let lo = ((a.min - Vec3::splat(band) - origin) * inv).floor();
            let hi = ((a.max + Vec3::splat(band) - origin) * inv).floor();
            let (lo, hi) = (lo.as_ivec3(), hi.as_ivec3());
            (lo.z..=hi.z).flat_map(move |z| (lo.y..=hi.y).flat_map(move |y| (lo.x..=hi.x).map(move |x| pack(x, y, z))))
        })
        .collect();
    r.bricks_with_duplicates = keys.len();
    keys.par_sort_unstable();
    keys.dedup();
    r.ms_bricks = t.elapsed().as_secs_f64() * 1e3;
    r.bricks = keys.len();
    r.voxels = keys.len() * B * B * B;
    r.brick_memory_bytes = keys.len() * (SAMPLES * 4 + 8);

    let k = sample_bricks.min(keys.len());
    r.sample_bricks = k;
    r.extrapolated = k < keys.len();
    if k == 0 {
        return r;
    }
    let scale = keys.len() as f64 / k as f64;
    let sample: Vec<[i32; 3]> = (0..k).map(|j| unpack(keys[j * keys.len() / k])).collect();
    let half_diag = 0.5 * 3.0f32.sqrt() * bsz;
    let brick_origin = |c: &[i32; 3]| origin + Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) * bsz;

    // 2a. Classification: one mesh query at the brick centre.
    let t = Instant::now();
    let class: Vec<f32> = sample
        .par_iter()
        .map(|c| mesh.signed_distance(brick_origin(c) + Vec3::splat(0.5 * bsz), 1e30_f32.min(mesh_bounds.diagonal())))
        .collect();
    r.ms_classify = t.elapsed().as_secs_f64() * 1e3 * scale;
    let near: Vec<bool> = class.iter().map(|d| d.abs() <= half_diag + band).collect();
    r.bricks_near = (near.iter().filter(|&&n| n).count() as f64 * scale).round() as usize;
    r.bricks_far = r.bricks - r.bricks_near;

    // 2b. Mesh SDF at every lattice sample of near bricks (far bricks: constant).
    // Bound per query comes from the previous sample (Lipschitz-1).
    let t = Instant::now();
    let mut fields: Vec<Vec<f32>> = sample
        .par_iter()
        .zip(class.par_iter().zip(near.par_iter()))
        .map(|(c, (&dc, &is_near))| {
            if !is_near {
                return vec![dc; SAMPLES];
            }
            let o = brick_origin(c);
            let mut f = vec![0.0f32; SAMPLES];
            let mut prev = dc.abs() + half_diag;
            let mut idx = 0;
            for z in 0..S {
                for y in 0..S {
                    for x in 0..S {
                        let p = o + Vec3::new(x as f32, y as f32, z as f32) * vs;
                        let d = mesh.signed_distance(p, prev + vs * 1.01);
                        f[idx] = d;
                        prev = d.abs();
                        idx += 1;
                    }
                }
            }
            f
        })
        .collect();
    r.ms_mesh_sdf = t.elapsed().as_secs_f64() * 1e3 * scale;

    // 3. Combine with instance SDFs: subtract = max(mesh, -min inst).
    let t = Instant::now();
    fields.par_iter_mut().zip(sample.par_iter()).for_each(|(f, c)| {
        let o = brick_origin(c);
        let mut idx = 0;
        for z in 0..S {
            for y in 0..S {
                for x in 0..S {
                    let p = o + Vec3::new(x as f32, y as f32, z as f32) * vs;
                    let mut m = f32::INFINITY;
                    for &i in set.candidates(p) {
                        if set.inflated_aabb_contains(i, p) {
                            m = m.min(set.sdf(i, p));
                        }
                    }
                    if m.is_finite() {
                        f[idx] = f[idx].max(-m);
                    }
                    idx += 1;
                }
            }
        }
    });
    r.ms_combine = t.elapsed().as_secs_f64() * 1e3 * scale;

    // 4. Naive surface nets within each brick.
    let t = Instant::now();
    let (nv, nq) = fields
        .par_iter()
        .zip(sample.par_iter())
        .map(|(f, c)| surface_nets(f, brick_origin(c), vs))
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    r.ms_nets = t.elapsed().as_secs_f64() * 1e3 * scale;
    r.net_vertices = (nv as f64 * scale).round() as usize;
    r.net_quads = (nq as f64 * scale).round() as usize;
    r
}

/// Naive surface nets on one brick's 9^3 lattice. Returns (vertices, quads).
fn surface_nets(f: &[f32], origin: Vec3, vs: f32) -> (usize, usize) {
    const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    let at = |x: usize, y: usize, z: usize| (z * S + y) * S + x;
    let mut cell_vertex = [u32::MAX; B * B * B];
    let mut verts: Vec<Vec3> = Vec::new();
    for z in 0..B {
        for y in 0..B {
            for x in 0..B {
                let mut d = [0.0f32; 8];
                let mut mask = 0u8;
                for (i, c) in CORNERS.iter().enumerate() {
                    d[i] = f[at(x + c[0], y + c[1], z + c[2])];
                    mask |= ((d[i] < 0.0) as u8) << i;
                }
                if mask == 0 || mask == 255 {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                let mut n = 0.0;
                for e in EDGES {
                    let (a, b) = (d[e[0]], d[e[1]]);
                    if (a < 0.0) != (b < 0.0) {
                        let t = a / (a - b);
                        let pa = Vec3::new(CORNERS[e[0]][0] as f32, CORNERS[e[0]][1] as f32, CORNERS[e[0]][2] as f32);
                        let pb = Vec3::new(CORNERS[e[1]][0] as f32, CORNERS[e[1]][1] as f32, CORNERS[e[1]][2] as f32);
                        sum += pa.lerp(pb, t);
                        n += 1.0;
                    }
                }
                cell_vertex[(z * B + y) * B + x] = verts.len() as u32;
                verts.push(origin + (Vec3::new(x as f32, y as f32, z as f32) + sum / n) * vs);
            }
        }
    }
    // One quad per sign-changing lattice edge whose four surrounding cells exist in this brick.
    let mut quads = 0;
    let cv = |x: i32, y: i32, z: i32| {
        (0..B as i32).contains(&x) && (0..B as i32).contains(&y) && (0..B as i32).contains(&z) && cell_vertex[((z as usize * B) + y as usize) * B + x as usize] != u32::MAX
    };
    for z in 0..S as i32 {
        for y in 0..S as i32 {
            for x in 0..S as i32 {
                let s0 = f[at(x as usize, y as usize, z as usize)] < 0.0;
                for axis in 0..3 {
                    let (nx, ny, nz) = (x + (axis == 0) as i32, y + (axis == 1) as i32, z + (axis == 2) as i32);
                    if nx >= S as i32 || ny >= S as i32 || nz >= S as i32 {
                        continue;
                    }
                    if (f[at(nx as usize, ny as usize, nz as usize)] < 0.0) == s0 {
                        continue;
                    }
                    let ok = match axis {
                        0 => cv(x, y, z) && cv(x, y - 1, z) && cv(x, y, z - 1) && cv(x, y - 1, z - 1),
                        1 => cv(x, y, z) && cv(x - 1, y, z) && cv(x, y, z - 1) && cv(x - 1, y, z - 1),
                        _ => cv(x, y, z) && cv(x - 1, y, z) && cv(x, y - 1, z) && cv(x - 1, y - 1, z),
                    };
                    quads += ok as usize;
                }
            }
        }
    }
    (verts.len(), quads)
}

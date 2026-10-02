//! Chunked spatial hierarchy (a PBVH in Blender terms).
//!
//! Faces are partitioned into leaves of ~`LEAF_FACES` faces. Vertices are then
//! *renumbered* so every leaf owns a contiguous vertex range. That one decision
//! carries most of the performance design:
//!
//! * brush kernels run one rayon task per leaf and write disjoint slices, so
//!   there is no locking and no scattered writes;
//! * sculpt-layer deltas are stored sparsely per leaf (untouched regions cost
//!   nothing);
//! * undo snapshots, normal updates and GPU uploads all operate on dirty leaves.

use std::ops::Range;

use glam::Vec3;
use rayon::prelude::*;

use crate::geom::{Aabb, Ray, ray_triangle};
use crate::mesh::{Face, NO_VERT, face_triangles, face_verts};

pub const LEAF_FACES: usize = 2048;
/// Faces per node of the per-leaf ray tree.
const RAY_LEAF_FACES: usize = 4;

#[derive(Clone, Copy, Debug)]
pub enum NodeKind {
    Inner { left: u32, right: u32 },
    Leaf { leaf: u32 },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub bounds: Aabb,
    pub kind: NodeKind,
}

#[derive(Clone, Debug)]
pub struct Leaf {
    pub faces: Vec<u32>,
    /// Vertices this leaf owns (and is the only writer of).
    pub owned: Range<u32>,
    /// Every vertex referenced by this leaf's faces (owned + borrowed), sorted.
    pub verts: Vec<u32>,
    pub bounds: Aabb,
    /// Fine BVH over `faces` (which are stored in tree order) for ray queries.
    /// The coarse leaves are sized for brush chunking, far too big to
    /// brute-force per ray.
    tree: Vec<RayNode>,
}

/// Preorder node: left child is `i + 1`, right child is `right`.
/// Leaf nodes cover `faces[start..start + count]`.
#[derive(Clone, Debug)]
struct RayNode {
    bounds: Aabb,
    start: u32,
    count: u32,
    right: u32,
}

impl Leaf {
    #[inline]
    pub fn owned_range(&self) -> Range<usize> {
        self.owned.start as usize..self.owned.end as usize
    }
    #[inline]
    pub fn owned_len(&self) -> usize {
        (self.owned.end - self.owned.start) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub face: u32,
    pub point: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    pub leaves: Vec<Leaf>,
    pub face_leaf: Vec<u32>,
    pub vert_leaf: Vec<u32>,
    /// Vertex is referenced by faces from more than one leaf.
    pub vert_shared: Vec<bool>,
    /// Leaves sharing at least one vertex with each leaf (excluding itself).
    pub leaf_neighbors: Vec<Vec<u32>>,
}

impl Bvh {
    /// Builds the hierarchy and renumbers vertices in place.
    /// Returns the `old -> new` vertex permutation so callers can permute
    /// their other per-vertex arrays with [`crate::mesh::permute`].
    pub fn build(positions: &mut Vec<Vec3>, faces: &mut [Face]) -> (Bvh, Vec<u32>) {
        let centroids: Vec<Vec3> = faces
            .par_iter()
            .map(|f| {
                let vs = face_verts(f);
                vs.iter().map(|&v| positions[v as usize]).sum::<Vec3>() / vs.len() as f32
            })
            .collect();

        let mut order: Vec<u32> = (0..faces.len() as u32).collect();
        let mut nodes = Vec::new();
        let mut leaf_ranges = Vec::new();
        split(&mut nodes, &mut leaf_ranges, &mut order, 0, &centroids);

        // Renumber vertices leaf by leaf.
        let vcount = positions.len();
        let mut old_to_new = vec![NO_VERT; vcount];
        let mut next = 0u32;
        let mut leaves: Vec<Leaf> = Vec::with_capacity(leaf_ranges.len());
        for r in &leaf_ranges {
            let start = next;
            let leaf_faces: Vec<u32> = order[r.clone()].to_vec();
            for &fi in &leaf_faces {
                for &v in face_verts(&faces[fi as usize]) {
                    let slot = &mut old_to_new[v as usize];
                    if *slot == NO_VERT {
                        *slot = next;
                        next += 1;
                    }
                }
            }
            leaves.push(Leaf { faces: leaf_faces, owned: start..next, verts: Vec::new(), bounds: Aabb::EMPTY, tree: Vec::new() });
        }
        // Loose vertices (no faces) go to the last leaf.
        for slot in old_to_new.iter_mut().filter(|s| **s == NO_VERT) {
            *slot = next;
            next += 1;
        }
        if let Some(last) = leaves.last_mut() {
            last.owned.end = next;
        }

        *positions = crate::mesh::permute(&old_to_new, positions);
        faces.par_iter_mut().for_each(|f| {
            for v in f.iter_mut().filter(|v| **v != NO_VERT) {
                *v = old_to_new[*v as usize];
            }
        });

        let mut face_leaf = vec![0u32; faces.len()];
        let mut vert_leaf = vec![0u32; vcount];
        let mut ref_count = vec![0u8; vcount];
        for (li, leaf) in leaves.iter_mut().enumerate() {
            for &fi in &leaf.faces {
                face_leaf[fi as usize] = li as u32;
            }
            for v in leaf.owned_range() {
                vert_leaf[v] = li as u32;
            }
            let mut vs: Vec<u32> = leaf.faces.iter().flat_map(|&fi| face_verts(&faces[fi as usize]).to_vec()).collect();
            vs.extend(leaf.owned.clone());
            vs.sort_unstable();
            vs.dedup();
            for &v in &vs {
                ref_count[v as usize] = ref_count[v as usize].saturating_add(1);
            }
            leaf.verts = vs;
        }
        let vert_shared: Vec<bool> = ref_count.iter().map(|&c| c > 1).collect();

        let vert_refs = crate::mesh::Csr::from_pairs(
            vcount,
            leaves.iter().enumerate().flat_map(|(li, l)| {
                l.verts.iter().filter(|&&v| vert_shared[v as usize]).map(move |&v| (v, li as u32))
            }),
        );
        let leaf_neighbors = leaves
            .par_iter()
            .enumerate()
            .map(|(li, l)| {
                let mut n: Vec<u32> = l
                    .verts
                    .iter()
                    .filter(|&&v| vert_shared[v as usize])
                    .flat_map(|&v| vert_refs.row(v as usize).iter().copied())
                    .filter(|&o| o != li as u32)
                    .collect();
                n.sort_unstable();
                n.dedup();
                n
            })
            .collect();

        let faces_ro: &[Face] = faces;
        leaves.par_iter_mut().for_each(|l| {
            let centroids: Vec<Vec3> = l
                .faces
                .iter()
                .map(|&fi| {
                    let vs = face_verts(&faces_ro[fi as usize]);
                    vs.iter().map(|&v| positions[v as usize]).sum::<Vec3>() / vs.len() as f32
                })
                .collect();
            let mut order: Vec<u32> = (0..l.faces.len() as u32).collect();
            build_ray_tree(&mut l.tree, &mut order, 0, &centroids);
            l.faces = order.iter().map(|&i| l.faces[i as usize]).collect();
        });

        let mut bvh = Bvh { nodes, leaves, face_leaf, vert_leaf, vert_shared, leaf_neighbors };
        bvh.refit_all(positions, faces);
        (bvh, old_to_new)
    }

    pub fn refit_all(&mut self, positions: &[Vec3], faces: &[Face]) {
        self.leaves.par_iter_mut().for_each(|l| refit_leaf(l, positions, faces));
        self.refit_nodes();
    }

    pub fn refit_leaves(&mut self, positions: &[Vec3], faces: &[Face], leaves: &[u32]) {
        // `leaves` is sorted and unique, so we can hand out disjoint &mut.
        let mut wanted = leaves.iter().peekable();
        let selected: Vec<&mut Leaf> = self
            .leaves
            .iter_mut()
            .enumerate()
            .filter_map(|(i, l)| {
                (wanted.peek().is_some_and(|&&w| w as usize == i)).then(|| {
                    wanted.next();
                    l
                })
            })
            .collect();
        selected.into_par_iter().for_each(|l| refit_leaf(l, positions, faces));
        self.refit_nodes();
    }

    /// Nodes are stored parent-before-children, so a reverse sweep is bottom-up.
    fn refit_nodes(&mut self) {
        for i in (0..self.nodes.len()).rev() {
            self.nodes[i].bounds = match self.nodes[i].kind {
                NodeKind::Leaf { leaf } => self.leaves[leaf as usize].bounds,
                NodeKind::Inner { left, right } => {
                    self.nodes[left as usize].bounds.union(&self.nodes[right as usize].bounds)
                }
            };
        }
    }

    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map(|n| n.bounds).unwrap_or(Aabb::EMPTY)
    }

    /// Leaves whose bounds intersect the sphere.
    pub fn query_sphere(&self, center: Vec3, radius: f32, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if !node.bounds.intersects_sphere(center, radius) {
                continue;
            }
            match node.kind {
                NodeKind::Leaf { leaf } => out.push(leaf),
                NodeKind::Inner { left, right } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        out.sort_unstable();
    }

    /// Closest hit within `tmax`.
    pub fn raycast(&self, positions: &[Vec3], faces: &[Face], ray: &Ray, tmax: f32) -> Option<Hit> {
        self.trace(positions, faces, ray, tmax, false)
    }

    /// Any hit within `tmax` (shadow/occlusion rays).
    pub fn occluded(&self, positions: &[Vec3], faces: &[Face], ray: &Ray, tmax: f32) -> bool {
        self.trace(positions, faces, ray, tmax, true).is_some()
    }

    fn trace(&self, positions: &[Vec3], faces: &[Face], ray: &Ray, tmax: f32, any: bool) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        // Avoid 0 * inf = NaN in the slab test for axis-aligned rays.
        let safe = |d: f32| if d.abs() < 1e-12 { 1e-12f32.copysign(d) } else { d };
        let inv = Vec3::new(safe(ray.dir.x), safe(ray.dir.y), safe(ray.dir.z)).recip();
        let mut best: Option<Hit> = None;
        let mut limit = tmax;
        let mut stack: smallvec::SmallVec<[u32; 64]> = smallvec::smallvec![0];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if node.bounds.ray_entry(ray.origin, inv, limit).is_none() {
                continue;
            }
            match node.kind {
                NodeKind::Inner { left, right } => {
                    stack.push(right);
                    stack.push(left);
                }
                NodeKind::Leaf { leaf } => {
                    let l = &self.leaves[leaf as usize];
                    let mut sub: smallvec::SmallVec<[u32; 32]> = smallvec::smallvec![0];
                    while let Some(i) = sub.pop() {
                        let node = &l.tree[i as usize];
                        if node.bounds.ray_entry(ray.origin, inv, limit).is_none() {
                            continue;
                        }
                        if node.count == 0 {
                            sub.push(node.right);
                            sub.push(i + 1);
                            continue;
                        }
                        for &fi in &l.faces[node.start as usize..(node.start + node.count) as usize] {
                            for [a, b, c] in face_triangles(&faces[fi as usize]) {
                                let hit = ray_triangle(
                                    ray.origin,
                                    ray.dir,
                                    positions[a as usize],
                                    positions[b as usize],
                                    positions[c as usize],
                                );
                                if let Some(t) = hit.filter(|&t| t < limit) {
                                    limit = t;
                                    best = Some(Hit { t, face: fi, point: ray.origin + ray.dir * t });
                                    if any {
                                        return best;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        best
    }
}

fn refit_leaf(l: &mut Leaf, positions: &[Vec3], faces: &[Face]) {
    let mut b = Aabb::EMPTY;
    for &v in &l.verts {
        b.grow(positions[v as usize]);
    }
    l.bounds = b;
    for i in (0..l.tree.len()).rev() {
        let node = &l.tree[i];
        l.tree[i].bounds = if node.count == 0 {
            l.tree[i + 1].bounds.union(&l.tree[node.right as usize].bounds)
        } else {
            let mut nb = Aabb::EMPTY;
            for &fi in &l.faces[node.start as usize..(node.start + node.count) as usize] {
                for &v in face_verts(&faces[fi as usize]) {
                    nb.grow(positions[v as usize]);
                }
            }
            nb
        };
    }
}

fn build_ray_tree(tree: &mut Vec<RayNode>, order: &mut [u32], base: u32, centroids: &[Vec3]) {
    let id = tree.len();
    tree.push(RayNode { bounds: Aabb::EMPTY, start: base, count: order.len() as u32, right: 0 });
    if order.len() <= RAY_LEAF_FACES {
        return;
    }
    let mut cb = Aabb::EMPTY;
    for &f in order.iter() {
        cb.grow(centroids[f as usize]);
    }
    let axis = cb.extent().max_position();
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis]));
    let (lo, hi) = order.split_at_mut(mid);
    build_ray_tree(tree, lo, base, centroids);
    let right = tree.len() as u32;
    build_ray_tree(tree, hi, base + mid as u32, centroids);
    tree[id].count = 0;
    tree[id].right = right;
}

fn split(nodes: &mut Vec<Node>, leaves: &mut Vec<Range<usize>>, order: &mut [u32], base: usize, centroids: &[Vec3]) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(Node { bounds: Aabb::EMPTY, kind: NodeKind::Leaf { leaf: 0 } });
    if order.len() <= LEAF_FACES {
        nodes[id as usize].kind = NodeKind::Leaf { leaf: leaves.len() as u32 };
        leaves.push(base..base + order.len());
        return id;
    }
    let mut cb = Aabb::EMPTY;
    for &f in order.iter() {
        cb.grow(centroids[f as usize]);
    }
    let axis = cb.extent().max_position();
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |a, b| {
        centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis])
    });
    let (lo, hi) = order.split_at_mut(mid);
    let left = split(nodes, leaves, lo, base, centroids);
    let right = split(nodes, leaves, hi, base + mid, centroids);
    nodes[id as usize].kind = NodeKind::Inner { left, right };
    id
}

/// Split `data` into the mutable sub-slices for the given ascending,
/// non-overlapping ranges.
pub fn split_ranges_mut<T>(mut data: &mut [T], ranges: impl IntoIterator<Item = Range<usize>>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    let mut consumed = 0;
    for r in ranges {
        let (_, rest) = std::mem::take(&mut data).split_at_mut(r.start - consumed);
        let (mid, rest) = rest.split_at_mut(r.end - r.start);
        out.push(mid);
        data = rest;
        consumed = r.end;
    }
    out
}

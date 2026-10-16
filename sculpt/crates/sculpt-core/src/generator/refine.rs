//! SPIKE 3: adaptive boundary refinement for the displacement boolean (Subtract).
//!
//! The displacement boolean ([`super::eval_vertex`]) projects every vertex that
//! is inside a cutter onto the cutter surface. Where the cutter boundary crosses
//! the mesh, the first inside vertex and the last outside vertex are one edge
//! apart, so the rim of a crater is a ramp one edge wide and a cutter smaller
//! than ~3 edges is a bump or nothing at all.
//!
//! This module builds, per touched BVH leaf, a **temporary patch** that replaces
//! the faces around the rim:
//!
//! 1. *Find the faces* (instance driven, cost ~ footprint of the instances, not
//!    mesh size): the fine ray tree of each leaf is queried with the instance
//!    AABBs. A face enters the patch if it has a vertex inside a cutter, or one
//!    of its (original mesh) edges crosses the zero set, or dips into it.
//! 2. *Refine in original space* by edge splitting with 1/2/3-edge templates
//!    (bisection / red). A split vertex of an edge whose ends are on different
//!    sides lies **exactly on the zero crossing** (secant/bisection on the SDF).
//!    Edges *inside* a source triangle that separate "surface" triangles from
//!    "cutter wall" triangles (the rim) are split at the midpoint and **snapped
//!    onto the intersection curve**: Newton on the SDF restricted to the local
//!    tangent plane. Snapped vertices are `Crease` vertices and keep both the
//!    surface normal and the wall normal (duplicated in the output).
//! 3. *Displace*: `Out` and `Crease` vertices stay, `In` vertices are projected
//!    with the same rule as [`super::eval_vertex`] (so patch interior == A). Faces whose vertices
//!    are all inside a cutter are not in the patch: the displacement boolean already draws them
//!    exactly, so the patch is only the rim ring, and cost ~ rim faces.
//!
//! **Crack-freeness** comes from construction, not from a stitching pass: a split
//! on an original mesh edge ("fixed" edge) is a pure function of the two end
//! positions (canonical end order, no dependence on the leaf), and a face is
//! in the patch iff a pure function of its own vertices and edges says so. Two
//! leaves that share an edge therefore produce bit-identical vertices on it and
//! faces next to an unrefined edge are never split on the other side.
//! Patch border vertices coincide with base mesh vertices. See
//! `docs/generator/spike3_refine.md`.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use glam::Vec3;
use rayon::prelude::*;

use super::{BoolOp, EvalParams, InstanceSet, affected_leaves, eval_vertex};
use crate::bvh::Bvh;
use crate::geom::Aabb;
use crate::mesh::{Face, face_len, face_triangles, face_verts};

const NONE: u32 = u32::MAX;

// ------------------------------------------------------------------ hashing

#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
        self.0 ^= self.0 >> 29;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

#[inline]
fn key(a: u32, b: u32) -> u64 {
    (a.min(b) as u64) << 32 | a.max(b) as u64
}

// ----------------------------------------------------------------- public API

#[derive(Clone, Copy, Debug)]
pub struct RefineParams {
    /// Projection settings for vertices inside a cutter (`op` must be Subtract). Keep equal to the
    /// settings of the displacement boolean drawn underneath so patch and base meet exactly.
    pub eval: EvalParams,
    /// Target length of rim (crease) edges as a fraction of the nearest cutter's
    /// smallest half size. 0.3 gives ~21 segments on a sphere rim.
    pub crease_frac: f32,
    /// Same for edges inside the cutter wall (cap interior); larger = coarser caps.
    pub wall_frac: f32,
    /// Maximum template passes per leaf.
    pub max_levels: u32,
    /// Hard cap on triangles per leaf patch.
    pub max_tris: usize,
    /// Debug only: leave every face that shares a vertex with another leaf to the base mesh, i.e. a
    /// naive "each leaf refines only its own interior" policy. Produces T-junction cracks at leaf
    /// borders; it exists to demonstrate and test the crack check.
    pub debug_skip_leaf_borders: bool,
}

impl Default for RefineParams {
    fn default() -> Self {
        RefineParams {
            eval: EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 3 },
            crease_frac: 0.3,
            wall_frac: 0.5,
            max_levels: 6,
            max_tris: 400_000,
            debug_skip_leaf_borders: false,
        }
    }
}

/// Read-only view of the mesh a patch is built from (undisplaced positions).
#[derive(Clone, Copy)]
pub struct MeshView<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub faces: &'a [Face],
    pub bvh: &'a Bvh,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PatchStats {
    /// Faces of the base mesh the patch replaces.
    pub faces_replaced: u32,
    /// Triangles those faces had (quad = 2).
    pub tris_before: u32,
    pub tris_after: u32,
    /// Patch vertices that are not base vertices.
    pub new_vertices: u32,
    /// Vertices snapped onto the intersection curve / zero crossings.
    pub crease_vertices: u32,
    pub levels: u32,
}

/// Temporary refined replacement of part of one leaf. Not part of the editable mesh.
#[derive(Clone, Debug, Default)]
pub struct RefinedPatch {
    pub leaf: u32,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Triangle list into `positions`.
    pub indices: Vec<u32>,
    /// Base vertex id for vertices that exist in the base mesh, `u32::MAX` otherwise.
    pub base_vertex: Vec<u32>,
    /// Topological id: output vertices with the same id share a position (a crease
    /// vertex is duplicated for its surface and wall normals).
    pub topo: Vec<u32>,
    /// Edges with at least one rim vertex (output vertex indices) shared by a surface and a wall triangle.
    /// Rim edges that run along a leaf border are listed by the leaf that owns the wall side only
    /// when the other side is also in this patch; use `crease_vertices` for point metrics.
    pub crease_edges: Vec<[u32; 2]>,
    /// Output indices (surface copy) of all vertices lying on the intersection curve.
    pub crease_vertices: Vec<u32>,
    /// Base faces hidden by this patch.
    pub replaced_faces: Vec<u32>,
    pub stats: PatchStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    pub leaves: usize,
    pub patches: usize,
    pub faces_replaced: usize,
    pub tris_before: usize,
    pub tris_after: usize,
    pub new_vertices: usize,
}

impl RefineStats {
    pub fn of(patches: &[RefinedPatch], leaves: usize) -> RefineStats {
        let mut s = RefineStats { leaves, patches: patches.len(), ..Default::default() };
        for p in patches {
            s.faces_replaced += p.stats.faces_replaced as usize;
            s.tris_before += p.stats.tris_before as usize;
            s.tris_after += p.stats.tris_after as usize;
            s.new_vertices += p.stats.new_vertices as usize;
        }
        s
    }
}

/// Patches for every leaf the instances overlap.
pub fn refine_all(mesh: &MeshView, set: &InstanceSet, params: &RefineParams) -> (Vec<RefinedPatch>, usize) {
    let boxes: Vec<Aabb> = (0..set.len()).map(|i| set.aabb(i)).collect();
    let leaves = affected_leaves(mesh.bvh, &boxes);
    (refine_leaves(mesh, set, params, &leaves), leaves.len())
}

/// Patches for the given leaves (e.g. [`affected_leaves`] of the changed instances). Leaves with
/// nothing to refine produce no patch.
pub fn refine_leaves(mesh: &MeshView, set: &InstanceSet, params: &RefineParams, leaves: &[u32]) -> Vec<RefinedPatch> {
    leaves.par_iter().filter_map(|&l| refine_leaf(mesh, set, params, l)).collect()
}

// ------------------------------------------------------------------ field

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Out,
    Crease,
    In,
}

#[derive(Clone, Copy)]
struct Probe {
    s: f32,
    inst: u32,
    scale: f32,
}

#[derive(Clone, Copy)]
struct PVert {
    orig: Vec3,
    ns: Vec3,
    kind: Kind,
    base: u32,
}

#[derive(Clone, Copy)]
struct Tri {
    v: [u32; 3],
    /// `fixed[i]`: edge `v[i] -> v[i+1]` descends from an original mesh edge.
    fixed: [bool; 3],
}

struct Field<'a> {
    set: &'a InstanceSet,
    p: &'a RefineParams,
}

impl Field<'_> {
    /// Union SDF (min over instances) with the nearest instance.
    #[inline]
    fn probe(&self, x: Vec3) -> Probe {
        let mut best = Probe { s: f32::INFINITY, inst: NONE, scale: 0.0 };
        for &i in self.set.candidates(x) {
            if !self.set.inflated_aabb_contains(i, x) {
                continue;
            }
            let s = self.set.sdf(i, x);
            if s < best.s {
                best = Probe { s, inst: i, scale: self.set.min_scale(i) };
            }
        }
        best
    }

    /// Classification tolerance (a few ulp of the coordinates, a small fraction of the cutter).
    #[inline]
    fn tol(x: Vec3, scale: f32) -> f32 {
        (2e-5 * scale).max(1e-6 * (1.0 + x.abs().max_element()))
    }

    #[inline]
    fn kind(&self, x: Vec3) -> Kind {
        let pr = self.probe(x);
        Self::kind_of(x, &pr)
    }

    #[inline]
    fn kind_of(x: Vec3, pr: &Probe) -> Kind {
        if pr.inst == NONE {
            return Kind::Out;
        }
        let t = Self::tol(x, pr.scale);
        if pr.s < -t {
            Kind::In
        } else if pr.s > t {
            Kind::Out
        } else {
            Kind::Crease
        }
    }

    /// Point on the segment `a -> b` (a, b on different sides) where the union SDF is zero.
    fn crossing(&self, a: Vec3, b: Vec3) -> Vec3 {
        let pa = self.probe(a);
        let pb = self.probe(b);
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        let (mut flo, mut fhi) = (pa.s.clamp(-1e9, 1e9), pb.s.clamp(-1e9, 1e9));
        let mut best = a;
        for it in 0..48 {
            let w = hi - lo;
            let t = if it % 2 == 0 && flo.abs() < 1e8 && fhi.abs() < 1e8 && flo != fhi {
                (lo + w * flo / (flo - fhi)).clamp(lo + 0.02 * w, hi - 0.02 * w)
            } else {
                0.5 * (lo + hi)
            };
            let x = a + (b - a) * t;
            best = x;
            let pr = self.probe(x);
            let f = pr.s.clamp(-1e9, 1e9);
            if pr.inst != NONE && f.abs() <= 0.25 * Self::tol(x, pr.scale) {
                return x;
            }
            if (f < 0.0) == (flo < 0.0) {
                lo = t;
                flo = f;
            } else {
                hi = t;
                fhi = f;
            }
        }
        best
    }

    /// Newton on the SDF inside the plane through `x` with normal `n`.
    fn snap(&self, mut x: Vec3, n: Vec3) -> Option<Vec3> {
        for _ in 0..10 {
            let pr = self.probe(x);
            if pr.inst == NONE {
                return None;
            }
            if pr.s.abs() <= 0.25 * Self::tol(x, pr.scale) {
                return Some(x);
            }
            let (_, g) = self.set.sdf_grad(pr.inst, x);
            let gt = g - n * g.dot(n);
            let l2 = gt.length_squared();
            if l2 < 1e-6 {
                return None;
            }
            x -= gt * (pr.s / l2);
        }
        let pr = self.probe(x);
        (pr.inst != NONE && pr.s.abs() <= Self::tol(x, pr.scale)).then_some(x)
    }

    /// Is the (non-In, non-In) edge segment dipping into a cutter at its midpoint?
    #[inline]
    fn dips(&self, a: &PVert, b: &PVert) -> bool {
        if a.kind == Kind::In || b.kind == Kind::In {
            return false;
        }
        let m = (a.orig + b.orig) * 0.5;
        let pr = self.probe(m);
        pr.inst != NONE && pr.s < -Self::tol(m, pr.scale)
    }
}

#[inline]
fn straddles(a: Kind, b: Kind) -> bool {
    (a == Kind::Out && b == Kind::In) || (a == Kind::In && b == Kind::Out)
}

#[inline]
fn bits(p: Vec3) -> (u32, u32, u32) {
    (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
}

/// Create the vertex splitting edge `a-b`. Pure in the two end vertices (order independent).
fn split_vertex(f: &Field, a: &PVert, b: &PVert, snap: bool) -> PVert {
    let (a, b) = if bits(a.orig) <= bits(b.orig) { (a, b) } else { (b, a) };
    let ns = (a.ns + b.ns).normalize_or(a.ns);
    if straddles(a.kind, b.kind) {
        return PVert { orig: f.crossing(a.orig, b.orig), ns, kind: Kind::Crease, base: NONE };
    }
    let m = (a.orig + b.orig) * 0.5;
    let pr = f.probe(m);
    let k = Field::kind_of(m, &pr);
    if snap {
        if k == Kind::Crease {
            return PVert { orig: m, ns, kind: Kind::Crease, base: NONE };
        }
        if let Some(x) = f.snap(m, ns) {
            return PVert { orig: x, ns, kind: Kind::Crease, base: NONE };
        }
    }
    PVert { orig: m, ns, kind: k, base: NONE }
}

// ------------------------------------------------------------------ refine

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Top,
    Wall,
}

struct EdgeRec {
    a: u32,
    b: u32,
    tris: [u32; 2],
    n: u8,
    fixed: bool,
    mid: u32,
}

/// Build the patch for one leaf, or `None` if no face of it needs one.
pub fn refine_leaf(mesh: &MeshView, set: &InstanceSet, params: &RefineParams, leaf_id: u32) -> Option<RefinedPatch> {
    let leaf = &mesh.bvh.leaves[leaf_id as usize];
    if !set.touches(&leaf.bounds) {
        return None;
    }
    let mut inst = Vec::new();
    set.candidates_in_box(&leaf.bounds, &mut inst);
    if inst.is_empty() {
        return None;
    }
    let mut f0: Vec<u32> = Vec::new();
    for &i in &inst {
        leaf.faces_overlapping(&set.aabb(i as usize), &mut f0);
    }
    f0.sort_unstable();
    f0.dedup();
    let fld = Field { set, p: params };

    // Vertex table of the candidate faces.
    let mut verts: Vec<PVert> = Vec::new();
    let mut vmap: FxMap<u32, u32> = FxMap::default();
    for &fi in &f0 {
        for &v in face_verts(&mesh.faces[fi as usize]) {
            vmap.entry(v).or_insert_with(|| {
                let p = mesh.positions[v as usize];
                verts.push(PVert { orig: p, ns: mesh.normals[v as usize], kind: fld.kind(p), base: v });
                verts.len() as u32 - 1
            });
        }
    }

    // Pure face selection + initial triangles. Faces entirely inside a cutter are left to the
    // displacement boolean (their vertices are moved by it); the patch holds the rest.
    let mut work: Vec<Tri> = Vec::new();
    let mut replaced: Vec<u32> = Vec::new();
    for &fi in &f0 {
        let f = &mesh.faces[fi as usize];
        let n = face_len(f);
        let id: [u32; 4] = std::array::from_fn(|k| if k < n { vmap[&f[k]] } else { NONE });
        let inside = (0..n).filter(|&k| verts[id[k] as usize].kind == Kind::In).count();
        if inside == n || (params.debug_skip_leaf_borders && face_verts(f).iter().any(|&v| mesh.bvh.vert_shared[v as usize])) {
            continue;
        }
        let hit = inside > 0
            || (0..n).any(|k| {
                let (a, b) = (&verts[id[k] as usize], &verts[id[(k + 1) % n] as usize]);
                fld.dips(a, b)
            });
        if !hit {
            continue;
        }
        replaced.push(fi);
        if n == 3 {
            work.push(Tri { v: [id[0], id[1], id[2]], fixed: [true; 3] });
        } else {
            work.push(Tri { v: [id[0], id[1], id[2]], fixed: [true, true, false] });
            work.push(Tri { v: [id[0], id[2], id[3]], fixed: [false, true, true] });
        }
    }
    if work.is_empty() {
        return None;
    }
    let tris_before = work.len();
    let base_verts = verts.len();

    // Template refinement. `done` holds triangles that can no longer change: no marked edge and no
    // interior neighbour that was split in the same pass.
    let mut levels = 0;
    let mut done: Vec<Tri> = Vec::new();
    let mut edges: Vec<EdgeRec> = Vec::new();
    let mut emap: FxMap<u64, u32> = FxMap::default();
    let mut sides: Vec<Side> = Vec::new();
    while levels < params.max_levels && work.len() + done.len() < params.max_tris {
        sides.clear();
        sides.extend(work.iter().map(|t| tri_side(&fld, &verts, t)));
        edges.clear();
        emap.clear();
        for (ti, t) in work.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t.v[k], t.v[(k + 1) % 3]);
                let e = *emap.entry(key(a, b)).or_insert_with(|| {
                    edges.push(EdgeRec { a, b, tris: [NONE; 2], n: 0, fixed: t.fixed[k], mid: NONE });
                    edges.len() as u32 - 1
                });
                let r = &mut edges[e as usize];
                if r.n < 2 {
                    r.tris[r.n as usize] = ti as u32;
                }
                r.n += 1;
            }
        }
        let mut marked_tri = vec![false; work.len()];
        let mut any = false;
        for r in edges.iter_mut() {
            let (a, b, fixed, n, t0, t1) = (r.a, r.b, r.fixed, r.n, r.tris[0], r.tris[1]);
            let (va, vb) = (verts[a as usize], verts[b as usize]);
            let snap = if straddles(va.kind, vb.kind) {
                Some(false)
            } else if fixed {
                // Both ends outside but the middle inside: split in the (inside) middle; the
                // two halves then straddle and the next pass puts exact crossings on them.
                fld.dips(&va, &vb).then_some(false)
            } else if n == 2 {
                interior_mark(&fld, &va, &vb, sides[t0 as usize], sides[t1 as usize])
            } else {
                None
            };
            if let Some(snap) = snap {
                let nv = split_vertex(&fld, &va, &vb, snap);
                verts.push(nv);
                r.mid = verts.len() as u32 - 1;
                marked_tri[t0 as usize] = true;
                if n == 2 {
                    marked_tri[t1 as usize] = true;
                }
                any = true;
            }
        }
        if !any {
            break;
        }
        levels += 1;
        let mut next: Vec<Tri> = Vec::with_capacity(work.len() * 2);
        for (ti, t) in work.iter().enumerate() {
            let rec = |k: usize| &edges[emap[&key(t.v[k], t.v[(k + 1) % 3])] as usize];
            if marked_tri[ti] {
                let mid: [u32; 3] = std::array::from_fn(|k| rec(k).mid);
                apply_template(t, mid, &verts, &mut next);
            } else {
                // Unmarked: settle unless an interior neighbour was split (its children may change
                // which of our edges are rim edges).
                let neighbour_split = (0..3).any(|k| {
                    let r = rec(k);
                    !r.fixed && r.n == 2 && marked_tri[(if r.tris[0] == ti as u32 { r.tris[1] } else { r.tris[0] }) as usize]
                });
                if neighbour_split { next.push(*t) } else { done.push(*t) }
            }
        }
        work = next;
    }
    done.append(&mut work);
    let tris = done;

    // Output.
    sides.clear();
    sides.extend(tris.iter().map(|t| tri_side(&fld, &verts, t)));
    let mut out = RefinedPatch { leaf: leaf_id, replaced_faces: replaced, ..Default::default() };
    let mut top_idx = vec![NONE; verts.len()];
    let mut wall_idx = vec![NONE; verts.len()];
    let final_pos = |v: &PVert| -> Vec3 {
        if v.kind == Kind::Out {
            v.orig
        } else if v.base != NONE {
            // A base vertex goes exactly where the displacement boolean puts it (same function,
            // same arithmetic as `position + delta`), whatever tolerance class it is in here, so
            // the patch and the base mesh drawn with that delta meet bit for bit.
            let q = eval_vertex(v.orig, v.ns, set, &params.eval);
            v.orig + (q - v.orig)
        } else if v.kind == Kind::In {
            eval_vertex(v.orig, v.ns, set, &params.eval)
        } else {
            v.orig
        }
    };
    let mut topo_of = vec![NONE; verts.len()];
    let mut ntopo = 0u32;
    for (ti, t) in tris.iter().enumerate() {
        for &vi in &t.v {
            let v = &verts[vi as usize];
            let wall = sides[ti] == Side::Wall || v.kind == Kind::In;
            let slot = if wall { &mut wall_idx[vi as usize] } else { &mut top_idx[vi as usize] };
            if *slot == NONE {
                let p = final_pos(v);
                let nrm = if wall && v.kind != Kind::Out {
                    let pr = fld.probe(p);
                    if pr.inst != NONE { -set.sdf_grad(pr.inst, p).1 } else { v.ns }
                } else {
                    v.ns
                };
                *slot = out.positions.len() as u32;
                out.positions.push(p);
                out.normals.push(nrm);
                out.base_vertex.push(v.base);
                if topo_of[vi as usize] == NONE {
                    topo_of[vi as usize] = ntopo;
                    ntopo += 1;
                }
                out.topo.push(topo_of[vi as usize]);
            }
            out.indices.push(*slot);
        }
    }
    // Rim edges: shared by a surface and a wall triangle.
    let mut emap2: FxMap<u64, (u32, Side)> = FxMap::default();
    let mut seen_crease = vec![false; verts.len()];
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t.v[k], t.v[(k + 1) % 3]);
            if verts[a as usize].kind != Kind::Crease && verts[b as usize].kind != Kind::Crease {
                continue;
            }
            match emap2.get(&key(a, b)) {
                None => {
                    emap2.insert(key(a, b), (ti as u32, sides[ti]));
                }
                Some(&(_, s0)) => {
                    if s0 != sides[ti] {
                        let (ia, ib) = (top_idx[a as usize], top_idx[b as usize]);
                        if ia != NONE && ib != NONE {
                            out.crease_edges.push([ia, ib]);
                        }
                    }
                }
            }
        }
    }
    for (vi, v) in verts.iter().enumerate() {
        if v.kind == Kind::Crease && top_idx[vi] != NONE && !seen_crease[vi] {
            seen_crease[vi] = true;
            out.crease_vertices.push(top_idx[vi]);
        }
    }
    out.stats = PatchStats {
        faces_replaced: out.replaced_faces.len() as u32,
        tris_before: tris_before as u32,
        tris_after: tris.len() as u32,
        new_vertices: (verts.len() - base_verts) as u32,
        crease_vertices: out.crease_vertices.len() as u32,
        levels,
    };
    out.indices.shrink_to_fit();
    Some(out)
}

fn tri_side(f: &Field, verts: &[PVert], t: &Tri) -> Side {
    let (a, b, c) = (&verts[t.v[0] as usize], &verts[t.v[1] as usize], &verts[t.v[2] as usize]);
    if a.kind == Kind::In || b.kind == Kind::In || c.kind == Kind::In {
        return Side::Wall;
    }
    let x = (a.orig + b.orig + c.orig) / 3.0;
    let pr = f.probe(x);
    if pr.inst != NONE && pr.s < -Field::tol(x, pr.scale) { Side::Wall } else { Side::Top }
}

/// Mark rule for an edge interior to a source triangle. `Some(snap)` = split it.
fn interior_mark(f: &Field, a: &PVert, b: &PVert, sa: Side, sb: Side) -> Option<bool> {
    let crease = sa != sb;
    if !crease && sa == Side::Top {
        return None;
    }
    let m = (a.orig + b.orig) * 0.5;
    let pr = f.probe(m);
    if pr.inst == NONE {
        return None;
    }
    let frac = if crease { f.p.crease_frac } else { f.p.wall_frac };
    ((a.orig - b.orig).length() > frac * pr.scale).then_some(crease)
}

fn apply_template(t: &Tri, mid: [u32; 3], verts: &[PVert], out: &mut Vec<Tri>) {
    let marked: [bool; 3] = std::array::from_fn(|k| mid[k] != NONE);
    let count = marked.iter().filter(|&&m| m).count();
    let (v, f) = (t.v, t.fixed);
    match count {
        0 => out.push(*t),
        3 => {
            let (a, b, c) = (mid[0], mid[1], mid[2]);
            out.push(Tri { v: [v[0], a, c], fixed: [f[0], false, f[2]] });
            out.push(Tri { v: [a, v[1], b], fixed: [f[0], f[1], false] });
            out.push(Tri { v: [c, b, v[2]], fixed: [false, f[1], f[2]] });
            out.push(Tri { v: [a, b, c], fixed: [false; 3] });
        }
        1 => {
            let i = marked.iter().position(|&m| m).unwrap();
            let (v0, v1, v2) = (v[i], v[(i + 1) % 3], v[(i + 2) % 3]);
            let (f0, f1, f2) = (f[i], f[(i + 1) % 3], f[(i + 2) % 3]);
            let a = mid[i];
            out.push(Tri { v: [v0, a, v2], fixed: [f0, false, f2] });
            out.push(Tri { v: [a, v1, v2], fixed: [f0, f1, false] });
        }
        _ => {
            let u = marked.iter().position(|&m| !m).unwrap();
            let s = (u + 1) % 3;
            let (v0, v1, v2) = (v[s], v[(s + 1) % 3], v[(s + 2) % 3]);
            let (f0, f1, f2) = (f[s], f[(s + 1) % 3], f[(s + 2) % 3]);
            let (a, b) = (mid[s], mid[(s + 1) % 3]);
            out.push(Tri { v: [a, v1, b], fixed: [f0, f1, false] });
            let o = |i: u32| verts[i as usize].orig;
            if (o(v0) - o(b)).length_squared() <= (o(a) - o(v2)).length_squared() {
                out.push(Tri { v: [v0, a, b], fixed: [f0, false, false] });
                out.push(Tri { v: [v0, b, v2], fixed: [false, f1, f2] });
            } else {
                out.push(Tri { v: [v0, a, v2], fixed: [f0, false, f2] });
                out.push(Tri { v: [a, b, v2], fixed: [false, f1, false] });
            }
        }
    }
}

// ---------------------------------------------------------------- compose

/// Base mesh with the patched faces removed and the patches added (for drawing, OBJ, checks).
#[derive(Clone, Debug, Default)]
pub struct Composed {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
}

/// Debug/tool helper, O(mesh): not part of the timed refinement path.
///
/// `positions` / `normals` are the base mesh **with the displacement boolean applied** (base position
/// plus [`super::evaluate`] delta): faces entirely inside a cutter are not in any patch and are drawn
/// from there.
pub fn compose(positions: &[Vec3], normals: &[Vec3], faces: &[Face], patches: &[RefinedPatch]) -> Composed {
    let mut hidden = vec![false; faces.len()];
    for p in patches {
        for &f in &p.replaced_faces {
            hidden[f as usize] = true;
        }
    }
    let mut c = Composed { positions: positions.to_vec(), normals: normals.to_vec(), tris: Vec::new() };
    for (fi, f) in faces.iter().enumerate() {
        if !hidden[fi] {
            c.tris.extend(face_triangles(f));
        }
    }
    for p in patches {
        let off = c.positions.len() as u32;
        c.positions.extend_from_slice(&p.positions);
        c.normals.extend_from_slice(&p.normals);
        c.tris.extend(p.indices.as_chunks::<3>().0.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    }
    c
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClosedReport {
    pub vertices: usize,
    pub edges: usize,
    /// Undirected edges used by exactly one triangle (cracks or open borders).
    pub boundary_edges: usize,
    /// Edges used by more than two triangles.
    pub non_manifold_edges: usize,
    /// Directed edges used twice (inconsistent winding).
    pub flipped_edges: usize,
    pub degenerate_tris: usize,
}

impl ClosedReport {
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0 && self.flipped_edges == 0
    }
}

/// Weld vertices by exact position and check the triangle soup is a closed, consistently oriented manifold.
pub fn check_closed(c: &Composed) -> ClosedReport {
    let mut weld: HashMap<(u32, u32, u32), u32> = HashMap::with_capacity(c.positions.len());
    let id: Vec<u32> = c
        .positions
        .iter()
        .map(|&p| {
            let n = weld.len() as u32;
            *weld.entry(bits(p)).or_insert(n)
        })
        .collect();
    let mut directed: FxMap<u64, u8> = FxMap::default();
    let mut rep = ClosedReport { vertices: weld.len(), ..Default::default() };
    for t in &c.tris {
        let w = [id[t[0] as usize], id[t[1] as usize], id[t[2] as usize]];
        if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] {
            rep.degenerate_tris += 1;
            continue;
        }
        for k in 0..3 {
            let (a, b) = (w[k], w[(k + 1) % 3]);
            *directed.entry((a as u64) << 32 | b as u64).or_insert(0) += 1;
        }
    }
    let mut undirected: FxMap<u64, u8> = FxMap::default();
    for (&k, &n) in &directed {
        let (a, b) = ((k >> 32) as u32, k as u32);
        if n > 1 {
            rep.flipped_edges += 1;
        }
        *undirected.entry(key(a, b)).or_insert(0) += n;
    }
    rep.edges = undirected.len();
    for &n in undirected.values() {
        match n {
            1 => rep.boundary_edges += 1,
            2 => {}
            _ => rep.non_manifold_edges += 1,
        }
    }
    rep
}

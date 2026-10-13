//! SPIKE 6: Adaptively Sampled Distance Field (ADF) boolean.
//!
//! Compared with the uniform bricks of `voxel.rs`, the distance samples live at the corners of an
//! octree that is subdivided top-down only where trilinear interpolation of the corner values
//! misses the true field by more than a tolerance (Frisken, Perry et al. 2000; written from memory
//! of the method, the papers could not be fetched in the build container). Cost follows surface
//! complexity instead of volume.
//!
//! Layout
//! ------
//! * One root cube ([`Frame`]); cells are addressed by `(depth, x, y, z)`.
//! * The tree is cut into **blocks**: the cells at depth `block_depth`, about two instance
//!   diameters wide. A block exists only where an instance AABB touches it. Blocks are the unit
//!   of caching, parallelism and incremental update.
//! * [`BaseAdf`]: per block, an error-driven tree of the *mesh* signed distance (the expensive
//!   closest-point queries). Built lazily, cached across parameter changes, rebuilt only when the
//!   mesh below changes ([`BaseAdf::clear`]).
//! * [`BooleanAdf`]: per block, a second tree of `f = combine(base ADF, instance SDFs)` with
//!   `max(b, -min d)` (subtract), `min(b, min d)` (add), `max(b, min d)` (intersect). The
//!   combined field is re-adapted only in blocks whose instance list changed.
//! * Extraction: dual contouring on the adaptive octree (Ju et al. 2002, cellProc/faceProc/edgeProc),
//!   one vertex per leaf from a QEF over Hermite data (edge crossing refined on the real combined
//!   field, normal from the active term: analytic for instances, trilinear gradient for the
//!   base). Quads that straddle octree levels are crack free by construction; the patch has an
//!   open boundary where blocks end. Only the quads touching dirty blocks are rebuilt on edits.
//!
//! Exactness of per-cell instance culling: an instance whose AABB is farther than
//! `R = cell diagonal + tol` from a cell has `d >= R` there. Dropping it changes `f` only where
//! `f < -R` (subtract) or `f > R` (add/intersect), never the sign and never within `R` of the
//! surface. Non-uniform scale uses the lower-bound SDF of the spike, which weakens this slightly
//! (a kink can appear deep inside the solid, never on the surface).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::BoolOp;
use super::instances::{Instance, InstanceSet};
use crate::geom::Aabb;

// ---------------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        let mut x = (self.0.rotate_left(5) ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 32;
        self.0 = x;
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
type FxSet<K> = HashSet<K, BuildHasherDefault<Fx>>;

/// `(depth, x, y, z)` packed; coordinates < 2^20, depth < 16.
#[inline]
fn pack(d: u32, c: [u32; 3]) -> u64 {
    (d as u64) << 60 | (c[2] as u64) << 40 | (c[1] as u64) << 20 | c[0] as u64
}

/// Lattice index of a 3x3x3 node grid: `x + 3y + 9z`, each in 0..3.
const fn lat(x: usize, y: usize, z: usize) -> usize {
    x + 3 * y + 9 * z
}

/// Which lattice points (besides the 8 corners) are compared with the trilinear interpolant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestPoints {
    /// Cell centre only (cheapest).
    Center,
    /// Centre and the 6 face centres.
    Face7,
    /// All 19 non-corner lattice points (the ADF paper's test set).
    Full19,
}

fn test_indices(t: TestPoints, out: &mut [usize; 19]) -> usize {
    match t {
        TestPoints::Center => {
            out[0] = lat(1, 1, 1);
            1
        }
        TestPoints::Face7 => {
            out[..7].copy_from_slice(&[lat(1, 1, 1), lat(0, 1, 1), lat(2, 1, 1), lat(1, 0, 1), lat(1, 2, 1), lat(1, 1, 0), lat(1, 1, 2)]);
            7
        }
        TestPoints::Full19 => {
            let mut n = 0;
            for z in 0..3 {
                for y in 0..3 {
                    for x in 0..3 {
                        if x == 1 || y == 1 || z == 1 {
                            out[n] = lat(x, y, z);
                            n += 1;
                        }
                    }
                }
            }
            n
        }
    }
}

#[inline]
fn corner_bits(i: usize) -> [usize; 3] {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1]
}

#[inline]
fn trilerp(c: &[f32; 8], u: Vec3) -> f32 {
    let x00 = c[0] + (c[1] - c[0]) * u.x;
    let x10 = c[2] + (c[3] - c[2]) * u.x;
    let x01 = c[4] + (c[5] - c[4]) * u.x;
    let x11 = c[6] + (c[7] - c[6]) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

/// Gradient of the trilinear interpolant with respect to `u` (divide by cell size for world units).
#[inline]
fn trilerp_grad(c: &[f32; 8], u: Vec3) -> Vec3 {
    let gx = {
        let a = (c[1] - c[0]) * (1.0 - u.y) + (c[3] - c[2]) * u.y;
        let b = (c[5] - c[4]) * (1.0 - u.y) + (c[7] - c[6]) * u.y;
        a * (1.0 - u.z) + b * u.z
    };
    let gy = {
        let a = (c[2] - c[0]) * (1.0 - u.x) + (c[3] - c[1]) * u.x;
        let b = (c[6] - c[4]) * (1.0 - u.x) + (c[7] - c[5]) * u.x;
        a * (1.0 - u.z) + b * u.z
    };
    let gz = {
        let a = (c[4] - c[0]) * (1.0 - u.x) + (c[5] - c[1]) * u.x;
        let b = (c[6] - c[2]) * (1.0 - u.x) + (c[7] - c[3]) * u.x;
        a * (1.0 - u.y) + b * u.y
    };
    Vec3::new(gx, gy, gz)
}

#[inline]
fn mixed(c: &[f32; 8]) -> bool {
    let neg = c.iter().filter(|&&v| v < 0.0).count();
    neg != 0 && neg != 8
}

// ---------------------------------------------------------------------------------------------
// Frame and parameters
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct AdfParams {
    /// Mean mesh edge length (the unit of every tolerance below).
    pub edge: f32,
    /// Interpolation error tolerance in mesh edge lengths.
    pub tol_edges: f32,
    /// Smallest cell edge in mesh edge lengths (sets the max depth).
    pub min_cell_edges: f32,
    /// Block width in mean instance extents.
    pub block_scale: f32,
    pub op: BoolOp,
    pub test: TestPoints,
}

impl AdfParams {
    pub fn new(edge: f32, tol_edges: f32, op: BoolOp) -> AdfParams {
        AdfParams { edge, tol_edges, min_cell_edges: 0.5, block_scale: 2.0, op, test: TestPoints::Face7 }
    }
    pub fn tol(&self) -> f32 {
        self.tol_edges * self.edge
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub origin: Vec3,
    pub size: f32,
    pub max_depth: u32,
    pub block_depth: u32,
}

impl Frame {
    /// Root cube around `bounds` (which must contain the mesh patch and every instance that may
    /// ever be placed), with depths derived from `params` and the mean instance extent.
    pub fn new(bounds: Aabb, params: &AdfParams, mean_instance_extent: f32) -> Frame {
        let ext = bounds.extent().max_element() * 1.02 + 2.0 * params.edge;
        let center = bounds.center();
        let min_cell = (params.min_cell_edges * params.edge).max(ext * 1e-6);
        let max_depth = ((ext / min_cell).log2().ceil() as u32).clamp(2, 22);
        let want = (ext / (params.block_scale * mean_instance_extent).max(min_cell)).log2().round();
        let block_depth = (want.max(1.0) as u32).min(max_depth - 1).min(14);
        Frame { origin: center - Vec3::splat(ext * 0.5), size: ext, max_depth, block_depth }
    }
    #[inline]
    pub fn cell_size(&self, depth: u32) -> f32 {
        self.size / (1u64 << depth) as f32
    }
    pub fn block_size(&self) -> f32 {
        self.cell_size(self.block_depth)
    }
    #[inline]
    fn block_min(&self, c: [u32; 3]) -> Vec3 {
        self.origin + Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) * self.block_size()
    }
    /// Block coordinate range overlapping a box (clamped to the root).
    fn block_range(&self, b: &Aabb) -> ([u32; 3], [u32; 3]) {
        let inv = 1.0 / self.block_size();
        let lim = ((1u32 << self.block_depth) - 1) as f32;
        let f = |v: Vec3| {
            let c = ((v - self.origin) * inv).floor().clamp(Vec3::ZERO, Vec3::splat(lim));
            [c.x as u32, c.y as u32, c.z as u32]
        };
        (f(b.min), f(b.max))
    }
    fn blocks_of(&self, b: &Aabb, mut f: impl FnMut(u64)) {
        let (lo, hi) = self.block_range(b);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    f(pack(self.block_depth, [x, y, z]));
                }
            }
        }
    }
    fn block_coords(&self, key: u64) -> [u32; 3] {
        let m = (1u64 << 20) - 1;
        [(key & m) as u32, ((key >> 20) & m) as u32, ((key >> 40) & m) as u32]
    }
}

// ---------------------------------------------------------------------------------------------
// Base ADF (mesh signed distance), cached
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct BNode {
    c: [f32; 8],
    child: u32,
}

#[derive(Clone, Debug, Default)]
pub struct BaseBlock {
    nodes: Vec<BNode>,
}

impl BaseBlock {
    #[inline]
    fn locate(&self, min: Vec3, size: f32, p: Vec3) -> (&BNode, Vec3, f32) {
        let (mut min, mut size, mut idx) = (min, size, 0usize);
        loop {
            let n = &self.nodes[idx];
            if n.child == 0 {
                return (n, min, size);
            }
            let half = size * 0.5;
            let mid = min + Vec3::splat(half);
            let (bx, by, bz) = ((p.x >= mid.x) as usize, (p.y >= mid.y) as usize, (p.z >= mid.z) as usize);
            idx = n.child as usize + bx + 2 * by + 4 * bz;
            min += Vec3::new(bx as f32, by as f32, bz as f32) * half;
            size = half;
        }
    }
    #[inline]
    fn eval(&self, min: Vec3, size: f32, p: Vec3) -> f32 {
        let (n, m, s) = self.locate(min, size, p);
        trilerp(&n.c, ((p - m) / s).clamp(Vec3::ZERO, Vec3::ONE))
    }
    #[inline]
    fn eval_grad(&self, min: Vec3, size: f32, p: Vec3) -> (f32, Vec3) {
        let (n, m, s) = self.locate(min, size, p);
        let u = ((p - m) / s).clamp(Vec3::ZERO, Vec3::ONE);
        (trilerp(&n.c, u), trilerp_grad(&n.c, u) / s)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BaseStats {
    pub blocks: usize,
    pub nodes: usize,
    pub leaves: usize,
    pub mesh_queries: u64,
    pub cache_hits: u64,
    /// Thread-summed time inside the mesh closure.
    pub query_cpu_ms: f64,
    pub bytes: usize,
}

/// Memo of mesh distances at lattice points of the finest grid, shared by all blocks (adjacent
/// blocks share corner and edge samples) and across `ensure` calls. Points are always evaluated at
/// their canonical lattice position, so the stored value does not depend on which cell asked.
struct LatticeCache {
    shards: Vec<std::sync::Mutex<FxMap<u64, f32>>>,
    queries: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
    ns: std::sync::atomic::AtomicU64,
}

impl LatticeCache {
    fn new() -> LatticeCache {
        LatticeCache {
            shards: (0..128).map(|_| std::sync::Mutex::new(FxMap::default())).collect(),
            queries: Default::default(),
            hits: Default::default(),
            ns: Default::default(),
        }
    }
    fn clear(&self) {
        for s in &self.shards {
            s.lock().unwrap().clear();
        }
    }
}

/// Lazily built, cached error-driven ADF of the mesh signed distance.
pub struct BaseAdf {
    frame: Frame,
    tol: f32,
    test: TestPoints,
    blocks: FxMap<u64, BaseBlock>,
    cache: LatticeCache,
}

struct BaseCtx<'a, F> {
    frame: Frame,
    tol: f32,
    test: TestPoints,
    field: &'a F,
    cache: &'a LatticeCache,
    unit: f32,
}

impl<F: Fn(Vec3, f32) -> f32> BaseCtx<'_, F> {
    fn at_unbounded(&self, ip: [u32; 3]) -> f32 {
        let p = self.frame.origin + Vec3::new(ip[0] as f32, ip[1] as f32, ip[2] as f32) * self.unit;
        let v = (self.field)(p, f32::MAX);
        let key = (ip[2] as u64) << 42 | (ip[1] as u64) << 21 | ip[0] as u64;
        let mut h = Fx::default();
        h.write_u64(key);
        self.cache.shards[(h.finish() >> 57) as usize & 127].lock().unwrap().insert(key, v);
        v
    }

    /// Mesh distance at integer lattice point `ip` (units of the finest cell).
    fn at(&self, ip: [u32; 3], bound: f32) -> f32 {
        use std::sync::atomic::Ordering::Relaxed;
        let key = (ip[2] as u64) << 42 | (ip[1] as u64) << 21 | ip[0] as u64;
        let mut h = Fx::default();
        h.write_u64(key);
        let shard = &self.cache.shards[(h.finish() >> 57) as usize & 127];
        if let Some(&v) = shard.lock().unwrap().get(&key) {
            self.cache.hits.fetch_add(1, Relaxed);
            return v;
        }
        let p = self.frame.origin + Vec3::new(ip[0] as f32, ip[1] as f32, ip[2] as f32) * self.unit;
        let t = std::time::Instant::now();
        let v = (self.field)(p, bound);
        self.cache.ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        self.cache.queries.fetch_add(1, Relaxed);
        shard.lock().unwrap().insert(key, v);
        v
    }
}

impl BaseAdf {
    pub fn new(frame: Frame, params: &AdfParams) -> BaseAdf {
        BaseAdf { frame, tol: params.tol(), test: params.test, blocks: FxMap::default(), cache: LatticeCache::new() }
    }

    /// Drop everything (the mesh below changed).
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.cache.clear();
    }

    pub fn contains(&self, key: u64) -> bool {
        self.blocks.contains_key(&key)
    }

    /// Build the blocks in `keys` that do not exist yet. `field(p, bound)` is the signed distance of
    /// the mesh; `bound` is an upper bound on `|d|` that the closure may use to prune its search
    /// (`f32::MAX` when unknown). Returns the number of blocks built.
    pub fn ensure<F>(&mut self, keys: &[u64], field: &F) -> usize
    where
        F: Fn(Vec3, f32) -> f32 + Sync,
    {
        let todo: Vec<u64> = keys.iter().copied().filter(|k| !self.blocks.contains_key(k)).collect();
        let cx = BaseCtx { frame: self.frame, tol: self.tol, test: self.test, field, cache: &self.cache, unit: self.frame.cell_size(self.frame.max_depth) };
        let built: Vec<(u64, BaseBlock)> = todo.par_iter().map(|&k| (k, build_base_block(&cx, k))).collect();
        let n = built.len();
        self.blocks.extend(built);
        n
    }

    fn block_ref(&self, key: u64) -> Option<(&BaseBlock, Vec3, f32)> {
        let b = self.blocks.get(&key)?;
        Some((b, self.frame.block_min(self.frame.block_coords(key)), self.frame.block_size()))
    }

    /// Cached distance at `p` (the block containing `p` must exist).
    pub fn eval(&self, p: Vec3) -> Option<f32> {
        let f = &self.frame;
        let c = ((p - f.origin) / f.block_size()).floor();
        if c.min_element() < 0.0 || c.max_element() >= (1u32 << f.block_depth) as f32 {
            return None;
        }
        let key = pack(f.block_depth, [c.x as u32, c.y as u32, c.z as u32]);
        let (b, m, s) = self.block_ref(key)?;
        Some(b.eval(m, s, p))
    }

    pub fn stats(&self) -> BaseStats {
        use std::sync::atomic::Ordering::Relaxed;
        let mut s = BaseStats { blocks: self.blocks.len(), mesh_queries: self.cache.queries.load(Relaxed), cache_hits: self.cache.hits.load(Relaxed), query_cpu_ms: self.cache.ns.load(Relaxed) as f64 * 1e-6, ..Default::default() };
        for b in self.blocks.values() {
            s.nodes += b.nodes.len();
            s.leaves += b.nodes.iter().filter(|n| n.child == 0).count();
            s.bytes += b.nodes.len() * std::mem::size_of::<BNode>();
        }
        s
    }
}

fn build_base_block<F>(cx: &BaseCtx<F>, key: u64) -> BaseBlock
where
    F: Fn(Vec3, f32) -> f32,
{
    let frame = &cx.frame;
    let shift = frame.max_depth - frame.block_depth;
    let bc = frame.block_coords(key);
    let imin = [bc[0] << shift, bc[1] << shift, bc[2] << shift];
    let isize = 1u32 << shift;
    let size = frame.block_size();
    let h = isize / 2;
    // Block centre: search a few block widths first (the surface is nearly always that close);
    // only if that finds nothing is the unbounded query paid for.
    let guess = 3.0 * size;
    let mut dc = cx.at([imin[0] + h, imin[1] + h, imin[2] + h], guess);
    if dc.abs() >= guess {
        dc = cx.at_unbounded([imin[0] + h, imin[1] + h, imin[2] + h]);
    }
    let bound = dc.abs() + size * 0.8660254 * 1.01;
    let mut c = [0.0f32; 8];
    for (i, v) in c.iter_mut().enumerate() {
        let b = corner_bits(i);
        *v = cx.at([imin[0] + b[0] as u32 * isize, imin[1] + b[1] as u32 * isize, imin[2] + b[2] as u32 * isize], bound);
    }
    let mut nodes = vec![BNode { c, child: 0 }];
    base_rec(cx, &mut nodes, 0, imin, isize, frame.block_depth, Some(dc), bound);
    BaseBlock { nodes }
}

#[allow(clippy::too_many_arguments)]
fn base_rec<F>(cx: &BaseCtx<F>, nodes: &mut Vec<BNode>, idx: usize, imin: [u32; 3], isize: u32, depth: u32, centre_value: Option<f32>, bound: f32)
where
    F: Fn(Vec3, f32) -> f32,
{
    let c = nodes[idx].c;
    let size = cx.frame.cell_size(depth);
    let half_diag = size * 0.8660254;
    let mut lat_v = [f32::NAN; 27];
    for (i, &v) in c.iter().enumerate() {
        let b = corner_bits(i);
        lat_v[lat(2 * b[0], 2 * b[1], 2 * b[2])] = v;
    }
    let ipos = |li: usize| -> [u32; 3] {
        let h = isize / 2;
        [imin[0] + (li % 3) as u32 * h, imin[1] + ((li / 3) % 3) as u32 * h, imin[2] + (li / 9) as u32 * h]
    };
    let fc = match centre_value {
        Some(v) => v,
        None => cx.at(ipos(lat(1, 1, 1)), bound),
    };
    lat_v[lat(1, 1, 1)] = fc;
    let near = fc.abs() <= half_diag + cx.tol || mixed(&c);
    if !near || depth >= cx.frame.max_depth {
        return;
    }
    let b2 = fc.abs() + half_diag * 1.01;
    let mut ti = [0usize; 19];
    let nt = test_indices(cx.test, &mut ti);
    let mut err = 0.0f32;
    for &li in &ti[..nt] {
        if li != lat(1, 1, 1) {
            lat_v[li] = cx.at(ipos(li), b2);
        }
        let u = Vec3::new((li % 3) as f32 * 0.5, ((li / 3) % 3) as f32 * 0.5, (li / 9) as f32 * 0.5);
        err = err.max((trilerp(&c, u) - lat_v[li]).abs());
    }
    if err <= cx.tol {
        return;
    }
    for (li, v) in lat_v.iter_mut().enumerate() {
        if v.is_nan() {
            *v = cx.at(ipos(li), b2);
        }
    }
    let first = nodes.len();
    nodes[idx].child = first as u32;
    for j in 0..8 {
        let jb = corner_bits(j);
        let mut cc = [0.0f32; 8];
        for (i, v) in cc.iter_mut().enumerate() {
            let ib = corner_bits(i);
            *v = lat_v[lat(jb[0] + ib[0], jb[1] + ib[1], jb[2] + ib[2])];
        }
        nodes.push(BNode { c: cc, child: 0 });
    }
    let h = isize / 2;
    for j in 0..8 {
        let jb = corner_bits(j);
        let cmin = [imin[0] + jb[0] as u32 * h, imin[1] + jb[1] as u32 * h, imin[2] + jb[2] as u32 * h];
        base_rec(cx, nodes, first + j, cmin, h, depth + 1, None, b2);
    }
}

// ---------------------------------------------------------------------------------------------
// Combined (boolean) ADF
// ---------------------------------------------------------------------------------------------

const NO_VERT: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
struct CNode {
    c: [f32; 8],
    child: u32,
    vert: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CBlock {
    id: u32,
    key: u64,
    nodes: Vec<CNode>,
    verts: Vec<Vec3>,
}

struct CCtx<'a> {
    frame: Frame,
    tol: f32,
    test: TestPoints,
    op: BoolOp,
    set: &'a InstanceSet,
    base: &'a BaseBlock,
    bmin: Vec3,
    bsize: f32,
}

impl CCtx<'_> {
    #[inline]
    fn f(&self, p: Vec3, list: &[u32], r: f32) -> f32 {
        let b = self.base.eval(self.bmin, self.bsize, p);
        match self.op {
            BoolOp::Subtract => {
                let mut m = f32::INFINITY;
                for &i in list {
                    m = m.min(self.set.sdf(i, p));
                }
                if m.is_finite() { b.max(-m) } else { b }
            }
            BoolOp::Add => {
                let mut m = f32::INFINITY;
                for &i in list {
                    m = m.min(self.set.sdf(i, p));
                }
                b.min(m)
            }
            BoolOp::Intersect => {
                let mut m = r;
                for &i in list {
                    m = m.min(self.set.sdf(i, p));
                }
                b.max(m)
            }
        }
    }

    /// Value and gradient of the *active* term.
    fn fg(&self, p: Vec3, list: &[u32], r: f32) -> (f32, Vec3) {
        let (b, bg) = self.base.eval_grad(self.bmin, self.bsize, p);
        let bg = bg.normalize_or(Vec3::Y);
        let mut m = if self.op == BoolOp::Intersect { r } else { f32::INFINITY };
        let mut arg = u32::MAX;
        for &i in list {
            let d = self.set.sdf(i, p);
            if d < m {
                m = d;
                arg = i;
            }
        }
        if arg == u32::MAX {
            return (if self.op == BoolOp::Intersect { b.max(m) } else { b }, bg);
        }
        let ig = self.set.sdf_grad(arg, p).1;
        match self.op {
            BoolOp::Subtract => {
                if b >= -m { (b, bg) } else { (-m, -ig) }
            }
            BoolOp::Add => {
                if b <= m { (b, bg) } else { (m, ig) }
            }
            BoolOp::Intersect => {
                if b >= m { (b, bg) } else { (m, ig) }
            }
        }
    }
}

fn aabb_near(a: &Aabb, min: Vec3, size: f32, r: f32) -> bool {
    let (lo, hi) = (min - Vec3::splat(r), min + Vec3::splat(size + r));
    !(a.min.cmpgt(hi).any() || a.max.cmplt(lo).any())
}

fn build_c_block(cx: &CCtx, id: u32, key: u64, members: &[u32]) -> CBlock {
    let size = cx.bsize;
    let min = cx.bmin;
    let mut scratch: Vec<u32> = members.to_vec();
    let r = size * 1.7320508 + cx.tol;
    let mut c = [0.0f32; 8];
    for (i, v) in c.iter_mut().enumerate() {
        let b = corner_bits(i);
        *v = cx.f(min + Vec3::new(b[0] as f32, b[1] as f32, b[2] as f32) * size, members, r);
    }
    let mut blk = CBlock { id, key, nodes: vec![CNode { c, child: 0, vert: NO_VERT }], verts: Vec::new() };
    c_rec(cx, &mut blk, 0, min, size, cx.frame.block_depth, 0, members.len(), &mut scratch);
    blk
}

#[allow(clippy::too_many_arguments)]
fn c_rec(cx: &CCtx, blk: &mut CBlock, idx: usize, min: Vec3, size: f32, depth: u32, ls: usize, le: usize, scratch: &mut Vec<u32>) {
    let c = blk.nodes[idx].c;
    let half_diag = size * 0.8660254;
    let r = size * 1.7320508 + cx.tol;
    // copy: `scratch` grows below while this node's list is still in use
    let list_buf: smallvec::SmallVec<[u32; 32]> = scratch[ls..le].iter().copied().collect();
    let list: &[u32] = &list_buf;
    let mut lat_v = [f32::NAN; 27];
    for (i, &v) in c.iter().enumerate() {
        let b = corner_bits(i);
        lat_v[lat(2 * b[0], 2 * b[1], 2 * b[2])] = v;
    }
    let pos = |li: usize| -> Vec3 {
        let (x, y, z) = (li % 3, (li / 3) % 3, li / 9);
        min + Vec3::new(x as f32, y as f32, z as f32) * (size * 0.5)
    };
    let mx = mixed(&c);
    let fc = cx.f(pos(lat(1, 1, 1)), list, r);
    lat_v[lat(1, 1, 1)] = fc;
    let near = fc.abs() <= half_diag + cx.tol || mx;
    let mut split = false;
    if near && depth < cx.frame.max_depth {
        let mut ti = [0usize; 19];
        let nt = test_indices(cx.test, &mut ti);
        let mut err = 0.0f32;
        for &li in &ti[..nt] {
            if li != lat(1, 1, 1) {
                lat_v[li] = cx.f(pos(li), list, r);
            }
            let u = Vec3::new((li % 3) as f32 * 0.5, ((li / 3) % 3) as f32 * 0.5, (li / 9) as f32 * 0.5);
            err = err.max((trilerp(&c, u) - lat_v[li]).abs());
        }
        // Anti-aliasing guard: a surface cell larger than a nearby instance can sample a cutter
        // at no test point at all, so such cells are always split.
        let small_instance = list.iter().any(|&i| cx.set.aabb(i as usize).extent().max_element() < size);
        split = err > cx.tol || small_instance;
    }
    if !split {
        if mx {
            let v = solve_vertex(cx, &c, min, size, list, r);
            blk.nodes[idx].vert = blk.verts.len() as u32;
            blk.verts.push(v);
        }
        return;
    }
    for (li, v) in lat_v.iter_mut().enumerate() {
        if v.is_nan() {
            *v = cx.f(pos(li), list, r);
        }
    }
    let first = blk.nodes.len();
    blk.nodes[idx].child = first as u32;
    for j in 0..8 {
        let jb = corner_bits(j);
        let mut cc = [0.0f32; 8];
        for (i, v) in cc.iter_mut().enumerate() {
            let ib = corner_bits(i);
            *v = lat_v[lat(jb[0] + ib[0], jb[1] + ib[1], jb[2] + ib[2])];
        }
        blk.nodes.push(CNode { c: cc, child: 0, vert: NO_VERT });
    }
    let half = size * 0.5;
    let rc = half * 1.7320508 + cx.tol;
    for j in 0..8 {
        let jb = corner_bits(j);
        let cmin = min + Vec3::new(jb[0] as f32, jb[1] as f32, jb[2] as f32) * half;
        let s = scratch.len();
        for &i in list {
            if aabb_near(&cx.set.aabb(i as usize), cmin, half, rc) {
                scratch.push(i);
            }
        }
        let e = scratch.len();
        c_rec(cx, blk, first + j, cmin, half, depth + 1, s, e, scratch);
        scratch.truncate(s);
    }
}

const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

/// Hermite data on the sign-changing edges of one leaf, then a QEF (mass-point biased, truncated
/// pseudo-inverse) solved in the cell and clamped to it.
fn solve_vertex(cx: &CCtx, c: &[f32; 8], min: Vec3, size: f32, list: &[u32], r: f32) -> Vec3 {
    let corner = |i: usize| {
        let b = corner_bits(i);
        min + Vec3::new(b[0] as f32, b[1] as f32, b[2] as f32) * size
    };
    let mut pts: [(Vec3, Vec3); 12] = [(Vec3::ZERO, Vec3::ZERO); 12];
    let mut n = 0;
    for e in EDGES {
        let (fa0, fb0) = (c[e[0]], c[e[1]]);
        if (fa0 < 0.0) == (fb0 < 0.0) {
            continue;
        }
        let (pa, pb) = (corner(e[0]), corner(e[1]));
        let (mut ta, mut tb, mut fa, mut fb) = (0.0f32, 1.0f32, fa0, fb0);
        let mut t = fa / (fa - fb);
        for _ in 0..3 {
            let p = pa + (pb - pa) * t;
            let fp = cx.f(p, list, r);
            if (fp < 0.0) == (fa < 0.0) {
                ta = t;
                fa = fp;
            } else {
                tb = t;
                fb = fp;
            }
            let d = fa - fb;
            t = if d.abs() > 1e-20 { (ta + fa / d * (tb - ta)).clamp(ta, tb) } else { 0.5 * (ta + tb) };
        }
        let p = pa + (pb - pa) * t;
        let (_, g) = cx.fg(p, list, r);
        pts[n] = (p, g.normalize_or(Vec3::Y));
        n += 1;
    }
    let mass = pts[..n].iter().map(|x| x.0).sum::<Vec3>() / n as f32;
    let (mut ata, mut atb) = ([[0.0f32; 3]; 3], Vec3::ZERO);
    for &(p, g) in &pts[..n] {
        let d = g.dot(p - mass);
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += g[i] * g[j];
            }
        }
        atb += g * d;
    }
    let x = mass + solve_sym3(ata, atb);
    x.clamp(min, min + Vec3::splat(size))
}

/// Minimum-norm least squares `A x = b` for symmetric PSD `A` via Jacobi eigen decomposition,
/// discarding eigenvalues below 5% of the largest.
#[allow(clippy::needless_range_loop)]
fn solve_sym3(mut a: [[f32; 3]; 3], b: Vec3) -> Vec3 {
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..8 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-12 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let cs = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * cs;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = cs * akp - sn * akq;
                a[k][q] = sn * akp + cs * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = cs * apk - sn * aqk;
                a[q][k] = sn * apk + cs * aqk;
            }
            for row in v.iter_mut() {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = cs * vkp - sn * vkq;
                row[q] = sn * vkp + cs * vkq;
            }
        }
    }
    let lam = [a[0][0], a[1][1], a[2][2]];
    let lmax = lam.iter().cloned().fold(0.0f32, f32::max);
    let mut x = Vec3::ZERO;
    for k in 0..3 {
        if lam[k] > 0.05 * lmax && lam[k] > 1e-9 {
            let vk = Vec3::new(v[0][k], v[1][k], v[2][k]);
            x += vk * (vk.dot(b) / lam[k]);
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------
// BooleanAdf: membership, build, incremental update
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct BuildTimes {
    pub bake_ms: f64,
    pub members_ms: f64,
    pub base_ms: f64,
    pub adapt_ms: f64,
    pub base_blocks_built: usize,
    pub blocks_adapted: usize,
}

impl BuildTimes {
    pub fn total_ms(&self) -> f64 {
        self.bake_ms + self.members_ms + self.base_ms + self.adapt_ms
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdfStats {
    pub blocks: usize,
    pub cells: usize,
    pub leaves: usize,
    pub surface_leaves: usize,
    pub max_depth_reached: u32,
    pub bytes: usize,
    pub base: BaseStats,
}

pub struct BooleanAdf {
    pub frame: Frame,
    pub params: AdfParams,
    pub base: BaseAdf,
    set: InstanceSet,
    members: FxMap<u64, Vec<u32>>,
    blocks: FxMap<u64, CBlock>,
    top_occ: FxSet<u64>,
    next_id: u32,
}

impl BooleanAdf {
    /// `frame` must contain every instance AABB for the lifetime of this object.
    pub fn new(frame: Frame, params: AdfParams, instances: Vec<Instance>) -> (BooleanAdf, f64) {
        let t = Instant::now();
        let set = InstanceSet::bake_only(instances, 0.0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let me = BooleanAdf {
            frame,
            params,
            base: BaseAdf::new(frame, &params),
            set,
            members: FxMap::default(),
            blocks: FxMap::default(),
            top_occ: FxSet::default(),
            next_id: 1,
        };
        (me, ms)
    }

    pub fn instance_set(&self) -> &InstanceSet {
        &self.set
    }

    fn pad_box(&self, a: &Aabb) -> Aabb {
        let p = Vec3::splat(self.params.tol() * 2.0);
        Aabb { min: a.min - p, max: a.max + p }
    }

    /// Full build: membership, base ADF (cached on later calls), combined ADF in every block.
    pub fn build<F>(&mut self, field: &F) -> BuildTimes
    where
        F: Fn(Vec3, f32) -> f32 + Sync,
    {
        let mut bt = BuildTimes::default();
        let t = Instant::now();
        self.rebuild_members();
        bt.members_ms = t.elapsed().as_secs_f64() * 1e3;
        let keys: Vec<u64> = self.members.keys().copied().collect();
        self.ensure_base(&keys, field, &mut bt);
        self.blocks.clear();
        self.adapt(&keys, &mut bt);
        self.rebuild_top();
        bt
    }

    /// Membership lists from scratch: parallel enumerate (block, instance) pairs, sort, group.
    fn rebuild_members(&mut self) {
        let frame = self.frame;
        let n = self.set.len();
        let mut pairs: Vec<(u64, u32)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let a = self.pad_box(&self.set.aabb(i));
                let mut v = Vec::new();
                frame.blocks_of(&a, |k| v.push((k, i as u32)));
                v
            })
            .collect();
        pairs.par_sort_unstable();
        self.members.clear();
        let mut k = 0;
        while k < pairs.len() {
            let key = pairs[k].0;
            let mut e = k;
            while e < pairs.len() && pairs[e].0 == key {
                e += 1;
            }
            self.members.insert(key, pairs[k..e].iter().map(|p| p.1).collect());
            k = e;
        }
    }

    fn ensure_base<F>(&mut self, keys: &[u64], field: &F, bt: &mut BuildTimes)
    where
        F: Fn(Vec3, f32) -> f32 + Sync,
    {
        let t = Instant::now();
        bt.base_blocks_built += self.base.ensure(keys, field);
        bt.base_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    /// Rebuild the combined tree of `keys` (all must have members and a base block).
    fn adapt(&mut self, keys: &[u64], bt: &mut BuildTimes) {
        let t = Instant::now();
        let ids: Vec<u32> = keys
            .iter()
            .map(|k| match self.blocks.get(k) {
                Some(b) => b.id,
                None => {
                    self.next_id += 1;
                    self.next_id
                }
            })
            .collect();
        let frame = self.frame;
        let (tol, test, op) = (self.params.tol(), self.params.test, self.params.op);
        let (set, base, members) = (&self.set, &self.base, &self.members);
        let built: Vec<(u64, Option<CBlock>)> = keys
            .par_iter()
            .zip(ids.par_iter())
            .map(|(&key, &id)| {
                let Some((bb, bmin, bsize)) = base.block_ref(key) else { return (key, None) };
                let cx = CCtx { frame, tol, test, op, set, base: bb, bmin, bsize };
                let blk = build_c_block(&cx, id, key, &members[&key]);
                // Blocks without any surface leaf contribute no quads and are not stored.
                (key, if blk.verts.is_empty() { None } else { Some(blk) })
            })
            .collect();
        bt.blocks_adapted += built.len();
        for (key, b) in built {
            match b {
                Some(b) => {
                    self.blocks.insert(key, b);
                }
                None => {
                    self.blocks.remove(&key);
                }
            }
        }
        bt.adapt_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    fn rebuild_top(&mut self) {
        self.top_occ.clear();
        for &k in self.blocks.keys() {
            let c = self.frame.block_coords(k);
            for d in 0..self.frame.block_depth {
                let s = self.frame.block_depth - d;
                self.top_occ.insert(pack(d, [c[0] >> s, c[1] >> s, c[2] >> s]));
            }
        }
    }

    /// Replace some instances and re-adapt only the blocks they touched (before or after).
    /// Returns the dirty block keys (sorted) for [`BooleanAdf::extract`].
    pub fn update<F>(&mut self, field: &F, changes: &[(usize, Instance)]) -> (Vec<u64>, BuildTimes)
    where
        F: Fn(Vec3, f32) -> f32 + Sync,
    {
        let mut bt = BuildTimes::default();
        let t = Instant::now();
        let old_new = self.set.update_bake_only(changes);
        if changes.len() * 4 > self.set.len() {
            // Most instances changed: regrouping from scratch beats per-instance list edits.
            let old_keys: Vec<u64> = self.members.keys().copied().collect();
            self.rebuild_members();
            let keys: Vec<u64> = self.members.keys().copied().collect();
            for k in old_keys {
                if !self.members.contains_key(&k) {
                    self.blocks.remove(&k);
                }
            }
            bt.members_ms = t.elapsed().as_secs_f64() * 1e3;
            self.ensure_base(&keys, field, &mut bt);
            self.adapt(&keys, &mut bt);
            self.rebuild_top();
            let mut all = keys;
            all.sort_unstable();
            return (all, bt);
        }
        let mut dirty: FxSet<u64> = FxSet::default();
        for (k, &(i, _)) in changes.iter().enumerate() {
            let (old, new) = (self.pad_box(&old_new[2 * k]), self.pad_box(&old_new[2 * k + 1]));
            let frame = self.frame;
            let mut olds = Vec::new();
            frame.blocks_of(&old, |key| olds.push(key));
            let mut news = Vec::new();
            frame.blocks_of(&new, |key| news.push(key));
            for &key in &olds {
                if let Some(l) = self.members.get_mut(&key)
                    && let Ok(p) = l.binary_search(&(i as u32))
                {
                    l.remove(p);
                }
                dirty.insert(key);
            }
            for &key in &news {
                let l = self.members.entry(key).or_default();
                if let Err(p) = l.binary_search(&(i as u32)) {
                    l.insert(p, i as u32);
                }
                dirty.insert(key);
            }
        }
        let mut dirty: Vec<u64> = dirty.into_iter().collect();
        dirty.sort_unstable();
        let empty: Vec<u64> = dirty.iter().copied().filter(|k| self.members.get(k).is_none_or(|l| l.is_empty())).collect();
        for k in &empty {
            self.members.remove(k);
            self.blocks.remove(k);
        }
        dirty.retain(|k| self.members.contains_key(k));
        bt.members_ms = t.elapsed().as_secs_f64() * 1e3;
        self.ensure_base(&dirty, field, &mut bt);
        let before = self.blocks.len();
        self.adapt(&dirty, &mut bt);
        if self.blocks.len() != before || !empty.is_empty() {
            self.rebuild_top();
        }
        let mut out = dirty;
        out.extend(empty);
        out.sort_unstable();
        (out, bt)
    }

    pub fn stats(&self) -> AdfStats {
        let mut s = AdfStats { blocks: self.blocks.len(), base: self.base.stats(), ..Default::default() };
        for b in self.blocks.values() {
            s.cells += b.nodes.len();
            s.leaves += b.nodes.iter().filter(|n| n.child == 0).count();
            s.surface_leaves += b.verts.len();
            s.bytes += b.nodes.len() * std::mem::size_of::<CNode>() + b.verts.len() * 12;
            let mut depth = vec![0u32; b.nodes.len()];
            for (i, n) in b.nodes.iter().enumerate() {
                if n.child != 0 {
                    for j in 0..8 {
                        depth[n.child as usize + j] = depth[i] + 1;
                    }
                }
                s.max_depth_reached = s.max_depth_reached.max(self.frame.block_depth + depth[i]);
            }
        }
        s.bytes += s.base.bytes + self.set.len() * 160 + self.members.values().map(|v| v.len() * 4 + 24).sum::<usize>();
        s
    }

    /// Every leaf cell as `(min corner, size, depth, has_vertex)`; for visualisation.
    pub fn leaf_boxes(&self) -> Vec<(Vec3, f32, u32, bool)> {
        let mut out = Vec::new();
        for b in self.blocks.values() {
            let mut stack = vec![(0usize, self.frame.block_min(self.frame.block_coords(b.key)), self.frame.block_size(), self.frame.block_depth)];
            while let Some((i, min, size, d)) = stack.pop() {
                let n = &b.nodes[i];
                if n.child == 0 {
                    out.push((min, size, d, n.vert != NO_VERT));
                } else {
                    let h = size * 0.5;
                    for j in 0..8 {
                        let jb = corner_bits(j);
                        stack.push((n.child as usize + j, min + Vec3::new(jb[0] as f32, jb[1] as f32, jb[2] as f32) * h, h, d + 1));
                    }
                }
            }
        }
        out
    }

    /// Structural fingerprint used by tests: every block's nodes and vertices, sorted by key.
    pub fn fingerprint(&self) -> Vec<Fingerprint> {
        let mut v: Vec<_> = self
            .blocks
            .values()
            .map(|b| (b.key, b.nodes.iter().map(|n| n.c).collect(), b.nodes.iter().map(|n| n.child).collect(), b.verts.iter().map(|p| p.to_array()).collect()))
            .collect();
        v.sort_by_key(|x| x.0);
        v
    }
}

// ---------------------------------------------------------------------------------------------
// Extraction: dual contouring on the adaptive octree
// ---------------------------------------------------------------------------------------------

/// `(block key, node corner values, node child links, vertex positions)`.
pub type Fingerprint = (u64, Vec<[f32; 8]>, Vec<u32>, Vec<[f32; 3]>);

#[derive(Clone, Debug, Default)]
pub struct AdfMesh {
    pub positions: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
    /// Quads skipped because a cell around a sign-changing edge had no vertex (level crossings
    /// where the coarser neighbour misses the crossing). Small holes; counted, not patched.
    pub missing: usize,
    pub quads: usize,
}

#[derive(Clone, Copy)]
enum H<'a> {
    None,
    Top(u8, [u32; 3]),
    Blk(&'a CBlock, u32, u8),
}

struct Tree<'a> {
    blocks: &'a FxMap<u64, CBlock>,
    top_occ: &'a FxSet<u64>,
    lr: u32,
    dirty_keys: Option<&'a FxSet<u64>>,
    dirty_top: Option<&'a FxSet<u64>>,
}

type Quad<'a> = [(&'a CBlock, u32); 4];

#[derive(Clone, Copy)]
enum Task<'a> {
    Cell(H<'a>),
    Face(usize, H<'a>, H<'a>),
    Edge(usize, [H<'a>; 4]),
}

struct Walker<'a> {
    t: &'a Tree<'a>,
    collect: bool,
    tasks: Vec<Task<'a>>,
    quads: Vec<Quad<'a>>,
    missing: usize,
}

impl<'a> Walker<'a> {
    fn is_leaf(&self, h: H<'a>) -> bool {
        match h {
            H::None => true,
            H::Top(..) => false,
            H::Blk(b, n, _) => b.nodes[n as usize].child == 0,
        }
    }
    fn is_top(h: H<'a>) -> bool {
        matches!(h, H::Top(..))
    }
    fn dirty(&self, h: H<'a>) -> bool {
        match (h, self.t.dirty_keys) {
            (H::None, _) => false,
            (_, None) => true,
            (H::Top(d, c), Some(_)) => self.t.dirty_top.is_some_and(|s| s.contains(&pack(d as u32, c))),
            (H::Blk(b, _, _), Some(k)) => k.contains(&b.key),
        }
    }
    fn sub(&self, h: H<'a>, i: usize) -> H<'a> {
        match h {
            H::None => H::None,
            H::Top(d, c) => {
                let nc = [c[0] * 2 + (i & 1) as u32, c[1] * 2 + ((i >> 1) & 1) as u32, c[2] * 2 + (i >> 2) as u32];
                if d as u32 + 1 == self.t.lr {
                    match self.t.blocks.get(&pack(self.t.lr, nc)) {
                        Some(b) => H::Blk(b, 0, self.t.lr as u8),
                        None => H::None,
                    }
                } else if self.t.top_occ.contains(&pack(d as u32 + 1, nc)) {
                    H::Top(d + 1, nc)
                } else {
                    H::None
                }
            }
            H::Blk(b, n, d) => {
                let ch = b.nodes[n as usize].child;
                if ch == 0 { h } else { H::Blk(b, ch + i as u32, d + 1) }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn cell(&mut self, h: H<'a>) {
        if self.is_leaf(h) || !self.dirty(h) {
            return;
        }
        if self.collect && !Self::is_top(h) {
            self.tasks.push(Task::Cell(h));
            return;
        }
        for i in 0..8 {
            let c = self.sub(h, i);
            self.cell(c);
        }
        for a in 0..3 {
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            for q in 0..4 {
                let base = (q & 1) << u | (q >> 1) << v;
                let (lo, hi) = (self.sub(h, base), self.sub(h, base | 1 << a));
                self.face(a, lo, hi);
            }
            // interior edges along axis a: two halves
            for half in 0..2 {
                let mut nodes = [H::None; 4];
                for q in 0..4 {
                    let idx = half << a | (q & 1) << u | (q >> 1) << v;
                    nodes[q] = self.sub(h, idx);
                }
                self.edge(a, nodes);
            }
        }
    }

    fn face(&mut self, a: usize, n0: H<'a>, n1: H<'a>) {
        if matches!(n0, H::None) || matches!(n1, H::None) || (self.is_leaf(n0) && self.is_leaf(n1)) {
            return;
        }
        if !(self.dirty(n0) || self.dirty(n1)) {
            return;
        }
        if self.collect && !Self::is_top(n0) && !Self::is_top(n1) {
            self.tasks.push(Task::Face(a, n0, n1));
            return;
        }
        let (u, v) = ((a + 1) % 3, (a + 2) % 3);
        for q in 0..4 {
            let o = (q & 1) << u | (q >> 1) << v;
            let (c0, c1) = (self.sub(n0, o | 1 << a), self.sub(n1, o));
            self.face(a, c0, c1);
        }
        // edges lying in the face: along u (varying v side) and along v
        for (b, o_axis) in [(u, v), (v, u)] {
            let (c1ax, c2ax) = ((b + 1) % 3, (b + 2) % 3);
            for hb in 0..2 {
                let mut nodes = [H::None; 4];
                for qa in 0..2 {
                    for qo in 0..2 {
                        let side = if qa == 0 { n0 } else { n1 };
                        let abit = if qa == 0 { 1 } else { 0 };
                        let idx = hb << b | abit << a | qo << o_axis;
                        let node = self.sub(side, idx);
                        let (mut q1, mut q2) = (0, 0);
                        for (ax, qv) in [(a, qa), (o_axis, qo)] {
                            if ax == c1ax {
                                q1 = qv;
                            } else if ax == c2ax {
                                q2 = qv;
                            }
                        }
                        nodes[q1 + 2 * q2] = node;
                    }
                }
                self.edge(b, nodes);
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn edge(&mut self, b: usize, nodes: [H<'a>; 4]) {
        if nodes.iter().any(|n| matches!(n, H::None)) {
            return;
        }
        if !nodes.iter().any(|&n| self.dirty(n)) {
            return;
        }
        if nodes.iter().all(|&n| self.is_leaf(n)) {
            self.process_edge(b, nodes);
            return;
        }
        if self.collect && !nodes.iter().any(|&n| Self::is_top(n)) {
            self.tasks.push(Task::Edge(b, nodes));
            return;
        }
        let (c1, c2) = ((b + 1) % 3, (b + 2) % 3);
        for hb in 0..2 {
            let mut sn = [H::None; 4];
            for q in 0..4 {
                let (q1, q2) = (q & 1, q >> 1);
                let idx = hb << b | (1 - q1) << c1 | (1 - q2) << c2;
                sn[q] = self.sub(nodes[q], idx);
            }
            self.edge(b, sn);
        }
    }

    fn process_edge(&mut self, b: usize, nodes: [H<'a>; 4]) {
        let (c1, c2) = ((b + 1) % 3, (b + 2) % 3);
        let mut best = 0;
        let mut best_d = 0;
        for (q, n) in nodes.iter().enumerate() {
            if let H::Blk(_, _, d) = n
                && *d > best_d
            {
                best_d = *d;
                best = q;
            }
        }
        let H::Blk(sb, sn, _) = nodes[best] else { return };
        let (q1, q2) = (best & 1, best >> 1);
        let lo = (1 - q1) << c1 | (1 - q2) << c2;
        let (f0, f1) = (sb.nodes[sn as usize].c[lo], sb.nodes[sn as usize].c[lo | 1 << b]);
        if (f0 < 0.0) == (f1 < 0.0) {
            return;
        }
        let order: [usize; 4] = if f0 < 0.0 { [0, 1, 3, 2] } else { [0, 2, 3, 1] };
        let mut slots: [Option<(&CBlock, u32)>; 4] = [None; 4];
        for (k, &q) in order.iter().enumerate() {
            let H::Blk(bk, n, _) = nodes[q] else { return };
            let v = bk.nodes[n as usize].vert;
            if v != NO_VERT {
                slots[k] = Some((bk, v));
            }
        }
        let have = slots.iter().flatten().count();
        if have < 4 {
            // A coarser neighbour whose corners all share a sign has no vertex although the finer
            // cell's edge crosses. One missing vertex is replaced by its predecessor (the quad
            // becomes a triangle, no hole); two or more missing drops the quad.
            self.missing += 1;
            if have < 3 {
                return;
            }
            for k in 0..4 {
                if slots[k].is_none() {
                    slots[k] = slots[(k + 3) % 4].or(slots[(k + 1) % 4]);
                }
            }
        }
        self.quads.push([slots[0].unwrap(), slots[1].unwrap(), slots[2].unwrap(), slots[3].unwrap()]);
    }
}

impl BooleanAdf {
    /// Dual-contour the combined field. With `dirty = Some(keys)` only quads that touch one of
    /// those blocks are produced (the replacement patch for an incremental edit).
    pub fn extract(&self, dirty: Option<&[u64]>) -> AdfMesh {
        if self.blocks.is_empty() {
            return AdfMesh::default();
        }
        let dk: Option<FxSet<u64>> = dirty.map(|d| d.iter().copied().collect());
        let dt: Option<FxSet<u64>> = dirty.map(|d| {
            let mut s = FxSet::default();
            for &k in d {
                let c = self.frame.block_coords(k);
                for dd in 0..self.frame.block_depth {
                    let sh = self.frame.block_depth - dd;
                    s.insert(pack(dd, [c[0] >> sh, c[1] >> sh, c[2] >> sh]));
                }
            }
            s
        });
        let tree = Tree { blocks: &self.blocks, top_occ: &self.top_occ, lr: self.frame.block_depth, dirty_keys: dk.as_ref(), dirty_top: dt.as_ref() };
        let mut top = Walker { t: &tree, collect: true, tasks: Vec::new(), quads: Vec::new(), missing: 0 };
        top.cell(H::Top(0, [0, 0, 0]));
        let tasks = std::mem::take(&mut top.tasks);
        let results: Vec<(Vec<Quad>, usize)> = tasks
            .par_chunks(64)
            .map(|chunk| {
                let mut w = Walker { t: &tree, collect: false, tasks: Vec::new(), quads: Vec::new(), missing: 0 };
                for t in chunk {
                    match *t {
                        Task::Cell(h) => w.cell(h),
                        Task::Face(a, x, y) => w.face(a, x, y),
                        Task::Edge(b, n) => w.edge(b, n),
                    }
                }
                (w.quads, w.missing)
            })
            .collect();
        let mut quads: Vec<Quad> = top.quads;
        let mut missing = top.missing;
        for (q, m) in results {
            quads.extend(q);
            missing += m;
        }
        // Assign compact vertex ids.
        let mut keys: Vec<u64> = quads.iter().flat_map(|q| q.iter().map(|(b, v)| (b.id as u64) << 32 | *v as u64)).collect();
        keys.par_sort_unstable();
        keys.dedup();
        let by_id: FxMap<u32, &CBlock> = self.blocks.values().map(|b| (b.id, b)).collect();
        let positions: Vec<Vec3> = keys.iter().map(|k| by_id[&((k >> 32) as u32)].verts[(k & 0xFFFF_FFFF) as usize]).collect();
        let id_of = |b: &CBlock, v: u32| keys.binary_search(&((b.id as u64) << 32 | v as u64)).unwrap() as u32;
        let mut tris = Vec::with_capacity(quads.len() * 2);
        for q in &quads {
            let ids = [id_of(q[0].0, q[0].1), id_of(q[1].0, q[1].1), id_of(q[2].0, q[2].1), id_of(q[3].0, q[3].1)];
            let (p0, p1, p2, p3) = (positions[ids[0] as usize], positions[ids[1] as usize], positions[ids[2] as usize], positions[ids[3] as usize]);
            // split along the shorter diagonal
            let quad = if p0.distance_squared(p2) <= p1.distance_squared(p3) {
                [[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]]
            } else {
                [[ids[0], ids[1], ids[3]], [ids[1], ids[2], ids[3]]]
            };
            for t in quad {
                if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                    tris.push(t);
                }
            }
        }
        AdfMesh { positions, tris, missing, quads: quads.len() }
    }
}

// ---------------------------------------------------------------------------------------------
// Reference: uniform-grid naive surface nets (the uniform-brick approach of voxel.rs, on a dense
// grid, with the real triangle output). Used by the quality bench and tests only.
// ---------------------------------------------------------------------------------------------

/// Naive surface nets of `f` on a dense lattice of spacing `vs` over `bounds`.
pub fn uniform_surface_nets<F: Fn(Vec3) -> f32 + Sync>(f: &F, bounds: Aabb, vs: f32) -> AdfMesh {
    let n = ((bounds.extent() / vs).ceil().as_uvec3() + glam::UVec3::ONE).to_array().map(|v| v as usize);
    let at = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let vals: Vec<f32> = (0..n[0] * n[1] * n[2])
        .into_par_iter()
        .map(|i| {
            let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
            f(bounds.min + Vec3::new(x as f32, y as f32, z as f32) * vs)
        })
        .collect();
    const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    let (cx, cy, cz) = (n[0] - 1, n[1] - 1, n[2] - 1);
    let cidx = |x: usize, y: usize, z: usize| (z * cy + y) * cx + x;
    let mut cell_v = vec![u32::MAX; cx * cy * cz];
    let mut positions = Vec::new();
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                let d: Vec<f32> = CORNERS.iter().map(|c| vals[at(x + c[0], y + c[1], z + c[2])]).collect();
                let neg = d.iter().filter(|&&v| v < 0.0).count();
                if neg == 0 || neg == 8 {
                    continue;
                }
                let (mut sum, mut cnt) = (Vec3::ZERO, 0.0);
                for e in EDGES {
                    let (a, b) = (d[e[0]], d[e[1]]);
                    if (a < 0.0) != (b < 0.0) {
                        let t = a / (a - b);
                        let pa = Vec3::new(CORNERS[e[0]][0] as f32, CORNERS[e[0]][1] as f32, CORNERS[e[0]][2] as f32);
                        let pb = Vec3::new(CORNERS[e[1]][0] as f32, CORNERS[e[1]][1] as f32, CORNERS[e[1]][2] as f32);
                        sum += pa.lerp(pb, t);
                        cnt += 1.0;
                    }
                }
                cell_v[cidx(x, y, z)] = positions.len() as u32;
                positions.push(bounds.min + (Vec3::new(x as f32, y as f32, z as f32) + sum / cnt) * vs);
            }
        }
    }
    let mut tris = Vec::new();
    let mut quads = 0;
    for z in 1..cz {
        for y in 1..cy {
            for x in 1..cx {
                for axis in 0..3 {
                    let (nx, ny, nz) = (x + (axis == 0) as usize, y + (axis == 1) as usize, z + (axis == 2) as usize);
                    if nx >= n[0] || ny >= n[1] || nz >= n[2] {
                        continue;
                    }
                    let (f0, f1) = (vals[at(x, y, z)], vals[at(nx, ny, nz)]);
                    if (f0 < 0.0) == (f1 < 0.0) {
                        continue;
                    }
                    let (c1, c2) = ((axis + 1) % 3, (axis + 2) % 3);
                    let cell = |q1: usize, q2: usize| {
                        let mut p = [x, y, z];
                        p[c1] -= 1 - q1;
                        p[c2] -= 1 - q2;
                        cell_v[cidx(p[0], p[1], p[2])]
                    };
                    let q = [cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1)];
                    if q.contains(&u32::MAX) {
                        continue;
                    }
                    let q = if f0 < 0.0 { q } else { [q[0], q[3], q[2], q[1]] };
                    tris.push([q[0], q[1], q[2]]);
                    tris.push([q[0], q[2], q[3]]);
                    quads += 1;
                }
            }
        }
    }
    AdfMesh { positions, tris, missing: 0, quads }
}

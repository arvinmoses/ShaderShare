//! A sculptable solid held as a distance field at several resolutions: detail exists only where it was sculpted, at the
//! resolution it was sculpted at, and a view can show as much of it as its distance calls for.
//!
//! The layout spike 4 (`levels.rs`) settled on:
//!
//! * **Levels.** Level 0 is the coarse field; each level above halves the lattice step. Every level is cut into 8x8x8
//!   blocks on one shared lattice (level `l + 1`'s point `2i` is level `l`'s point `i`), so a coarse cell splits exactly
//!   into eight finer cells and a block at level `l + 1` covers 4x4x4 cells of level `l`. A block exists at a level only
//!   where something was sculpted there or finer; elsewhere that level's values are the coarser level's, interpolated.
//!   Levels nest: a block's parent at the level below always exists.
//! * **Each level holds its own work.** A brush writes at one level, the one its size on screen calls for. After an
//!   edit at a coarse level, every finer level's own detail (its values less the level below) moves with the surface,
//!   along the normal by as far as the edit moved it, so fine detail rides on large edits. Fine work that builds real
//!   form (more than a small fraction of its level's band away from the level below) is moved down a level when the
//!   stroke ends, so a later coarse edit sees it as form and no level's detail outgrows its band.
//! * **Surface.** Dual contouring over the tree of cells, the same rules the GPU extractor follows (see
//!   [`LevelVolume::extract`]): a vertex per surface piece of each smallest cell, a quad per smallest crossing lattice
//!   edge, and a lattice point shared by several levels takes the finest one's value, so cells of different levels agree
//!   on every crossing and the surface closes. A level cap gives what a distant view draws.
//!
//! Plane brushes (clay and trim), the stroke-start surface brushes are placed on, checkerboard cleanup, speck removal
//! at the end of a stroke, undo and saving work as in `volume.rs`, per level.

use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::remesh::{EDGES, EDGE_OF, FACES, corner, find, union};
use crate::mesh::{NO_VERT, PolyMesh};

/// Hash maps keyed by block coordinates, looked up in every inner loop here: hashed the fast way (as rustc's own maps
/// are) rather than the default, which guards against crafted keys at several times the cost.
pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FastHasher>>;

/// Shards a [`Shared`] map splits its keys over, by place: neighbouring blocks share one.
const SHARDS: usize = 1024;

/// A key a [`Shared`] map can shard by place.
pub trait Sharded: std::hash::Hash + Eq + Copy {
    fn shard(&self) -> usize;
}

impl Sharded for IVec3 {
    #[inline]
    fn shard(&self) -> usize {
        // Blocks 8 apart along each axis to a shard: a stroke writes a few shards, not every one.
        let c = IVec3::new(self.x >> 3, self.y >> 3, self.z >> 3);
        ((c.x as u32).wrapping_mul(73856093) ^ (c.y as u32).wrapping_mul(19349663) ^ (c.z as u32).wrapping_mul(83492791)) as usize & (SHARDS - 1)
    }
}

impl Sharded for Key {
    #[inline]
    fn shard(&self) -> usize {
        (self.1.shard() + 389 * self.0 as usize) & (SHARDS - 1)
    }
}

/// A map that copies in a moment: its shards are shared between copies and a shard is copied only when one copy
/// writes to it while another still holds it. Carrying detail on a worker reads such a copy of the volume while the
/// next dabs change the original.
pub struct Shared<K, V> {
    shards: Vec<Arc<HashMap<K, V>>>,
    len: usize,
}

impl<K, V> Clone for Shared<K, V> {
    fn clone(&self) -> Self {
        Shared { shards: self.shards.clone(), len: self.len }
    }
}

impl<K: Sharded, V: Clone> Default for Shared<K, V> {
    fn default() -> Self {
        let empty = Arc::new(HashMap::default());
        Shared { shards: vec![empty; SHARDS], len: 0 }
    }
}

impl<K: Sharded, V: Clone> Shared<K, V> {
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.shards[k.shard()].get(k)
    }

    #[inline]
    pub fn contains_key(&self, k: &K) -> bool {
        self.shards[k.shard()].contains_key(k)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let sh = &mut self.shards[k.shard()];
        if !sh.contains_key(k) {
            return None;
        }
        Arc::make_mut(sh).get_mut(k)
    }

    pub fn get_or_insert_with(&mut self, k: K, f: impl FnOnce() -> V) -> &mut V {
        let sh = Arc::make_mut(&mut self.shards[k.shard()]);
        if !sh.contains_key(&k) {
            self.len += 1;
        }
        sh.entry(k).or_insert_with(f)
    }

    pub fn insert(&mut self, k: K, v: impl Into<V>) -> Option<V> {
        let old = Arc::make_mut(&mut self.shards[k.shard()]).insert(k, v.into());
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let sh = &mut self.shards[k.shard()];
        if !sh.contains_key(k) {
            return None;
        }
        self.len -= 1;
        Arc::make_mut(sh).remove(k)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this map is still `other`, a copy of it, shard for shard: neither written since (a write copies the
    /// shard it writes while the other still holds it).
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        self.len == other.len && self.shards.len() == other.shards.len() && self.shards.iter().zip(&other.shards).all(|(a, b)| Arc::ptr_eq(a, b))
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.shards.iter().flat_map(|s| s.keys())
    }

    /// The shards, for going over the map on several threads.
    pub(crate) fn shards(&self) -> &[Arc<HashMap<K, V>>] {
        &self.shards
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.shards.iter().flat_map(|s| s.values())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.shards.iter().flat_map(|s| s.iter())
    }
}

impl<K: Sharded, V: Clone> std::ops::Index<&K> for Shared<K, V> {
    type Output = V;
    fn index(&self, k: &K) -> &V {
        self.get(k).expect("no such key")
    }
}

/// A [`Shared`] set.
#[derive(Clone)]
pub struct SharedSet<K>(Shared<K, ()>);

impl<K: Sharded> Default for SharedSet<K> {
    fn default() -> Self {
        SharedSet(Shared::default())
    }
}

impl<K: Sharded> SharedSet<K> {
    #[inline]
    pub fn contains(&self, k: &K) -> bool {
        self.0.contains_key(k)
    }

    pub fn insert(&mut self, k: K) -> bool {
        self.0.insert(k, ()).is_none()
    }

    pub fn remove(&mut self, k: &K) -> bool {
        self.0.remove(k).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.0.keys()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // Mixed down so the low bits a table indexes by depend on every input bit.
        let h = self.0;
        (h ^ (h >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (h >> 29)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u32 as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

const B: i32 = 8;
const N: usize = 512;
const B3: IVec3 = IVec3::splat(B);

/// Most levels a volume may have.
pub const MAX_LEVELS: usize = 8;
/// Vertices each block may hold in the extracted surface (the GPU gives every block this many places).
pub const VERTS_PER_BLOCK: usize = 768;
/// Surface pieces with at most this many vertices, closed and inside a stroke's reach, are specks: removed when the
/// stroke ends.
pub const SPECK_VERTS: usize = 64;
/// How much of a clay dab's height each dab adds once the first layer is down.
pub const BUILD_STEP: f32 = 0.05;
/// Bytes of stroke history kept before the oldest strokes are forgotten.
const HISTORY_BYTES: usize = 1 << 30;
/// Detail at a level up to this fraction of its band stays there; beyond it, form is moved down a level.
const REBASE_KEEP: f32 = 0.4;

/// A block's samples while they are worked on.
pub type Block = Box<[f32; N]>;
/// A block's samples as the volume keeps them: shared by the copies of the volume that hold them.
pub type Stored = Arc<[f32; N]>;
/// A block's level and coordinates (in blocks of that level).
pub type Key = (u8, IVec3);
/// One bit per sample.
type SignMask = [u64; 8];

/// A plane brush's dab as a push on the surface: its centre on the surface, its plane's normal, its radius and
/// hardness (the fraction of the radius its falloff starts at).
#[derive(Clone, Copy)]
struct Push {
    on: Vec3,
    n: Vec3,
    r: f32,
    hard: f32,
}

/// The surface under a plane brush as it is now, as heights along the brush's normal `n` over its disc: a column every
/// `step` across, each the first crossing of the dab's level's field along `n` from above the dab (NaN where the
/// column meets no surface in reach). A sample that holds only a bound on its distance to the surface (see
/// [`LevelVolume::plateau`]) reads its distance from here: its height over the column's surface, to the tangent plane
/// there. Blended from a bound instead, a dab would put the new surface where the bound ran out and leave the field
/// around it with the dab's weight for a slope, a lattice of facets along a big brush's rim.
#[derive(Clone)]
struct Ground {
    on: Vec3,
    n: Vec3,
    t1: Vec3,
    t2: Vec3,
    step: f32,
    /// Columns `-half..=half` along `t1` and `t2`.
    half: i32,
    h: Vec<f32>,
}

impl Ground {
    fn size(&self) -> i32 {
        2 * self.half + 1
    }

    /// The signed distance from `p` to the tangent plane of the surface under it, the surface interpolated smoothly
    /// (Catmull-Rom over the four columns around it each way, so its slope is continuous across columns: clay laid on
    /// it shows no seam at the columns), when all sixteen meet the surface.
    fn distance_smooth(&self, p: Vec3) -> Option<f32> {
        let q = p - self.on;
        let s = q.dot(self.t1) / self.step + self.half as f32;
        let t = q.dot(self.t2) / self.step + self.half as f32;
        let (i, j) = (s.floor() as i32, t.floor() as i32);
        if i < 1 || j < 1 || i + 2 >= self.size() || j + 2 >= self.size() {
            return None;
        }
        let (fs, ft) = (s - i as f32, t - j as f32);
        // A Catmull-Rom segment's value and slope at `x`.
        let spline = |p0: f32, p1: f32, p2: f32, p3: f32, x: f32| -> (f32, f32) {
            let (a, b, c) = (-p0 + 3.0 * p1 - 3.0 * p2 + p3, 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3, -p0 + p2);
            (0.5 * (((a * x + b) * x + c) * x + 2.0 * p1), 0.5 * ((3.0 * a * x + 2.0 * b) * x + c))
        };
        let mut rows = [(0.0f32, 0.0f32); 4];
        for (k, row) in rows.iter_mut().enumerate() {
            let at = |a: i32| self.h[((j - 1 + k as i32) * self.size() + a) as usize];
            let (h0, h1, h2, h3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            if !(h0.is_finite() && h1.is_finite() && h2.is_finite() && h3.is_finite()) {
                return None;
            }
            *row = spline(h0, h1, h2, h3, fs);
        }
        let (h, ht) = spline(rows[0].0, rows[1].0, rows[2].0, rows[3].0, ft);
        let (hs, _) = spline(rows[0].1, rows[1].1, rows[2].1, rows[3].1, ft);
        let (hs, ht) = (hs / self.step, ht / self.step);
        Some((q.dot(self.n) - h) / (1.0 + hs * hs + ht * ht).sqrt())
    }

    /// The heights smoothed over `radius` columns either way (a box three times over, so near a Gaussian), leaving
    /// only the form at the brush's scale: what clay laid on the surface follows. A column with no surface takes its
    /// neighbours' mean, when any has one.
    fn smooth(&mut self, radius: i32) {
        let size = self.size();
        for _ in 0..3 {
            for axis in 0..2 {
                let prev = self.h.clone();
                // A row at a time over the threads: a wide brush's are a few hundred columns each way.
                self.h.par_chunks_mut(size as usize).enumerate().for_each(|(b, row)| {
                    let b = b as i32;
                    for a in 0..size {
                        let (mut sum, mut count) = (0.0f32, 0usize);
                        for d in -radius..=radius {
                            let (x, y) = if axis == 0 { (a + d, b) } else { (a, b + d) };
                            if x < 0 || y < 0 || x >= size || y >= size {
                                continue;
                            }
                            let v = prev[(y * size + x) as usize];
                            if v.is_finite() {
                                sum += v;
                                count += 1;
                            }
                        }
                        row[a as usize] = if count > 0 { sum / count as f32 } else { f32::NAN };
                    }
                });
            }
        }
    }

    /// The signed distance from `p` to the surface's tangent plane under it, when its column and the three around it
    /// meet the surface.
    fn distance(&self, p: Vec3) -> Option<f32> {
        let q = p - self.on;
        let s = q.dot(self.t1) / self.step + self.half as f32;
        let t = q.dot(self.t2) / self.step + self.half as f32;
        let (i, j) = (s.floor() as i32, t.floor() as i32);
        if i < 0 || j < 0 || i + 1 >= self.size() || j + 1 >= self.size() {
            return None;
        }
        let at = |a: i32, b: i32| self.h[(b * self.size() + a) as usize];
        let (h00, h10, h01, h11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
        if h00.is_nan() || h10.is_nan() || h01.is_nan() || h11.is_nan() {
            return None;
        }
        let (fs, ft) = (s - i as f32, t - j as f32);
        let lerp = |a: f32, b: f32, x: f32| a + (b - a) * x;
        let h = lerp(lerp(h00, h10, fs), lerp(h01, h11, fs), ft);
        let hs = lerp(h10 - h00, h11 - h01, ft) / self.step;
        let ht = lerp(h01 - h00, h11 - h10, fs) / self.step;
        Some((q.dot(self.n) - h) / (1.0 + hs * hs + ht * ht).sqrt())
    }
}

/// A block as it was before a stroke changed it.
#[derive(Clone)]
enum Was {
    Missing,
    /// A level 0 block wholly inside the solid.
    Inside,
    Stored(Stored),
}

/// The terms of a polynomial in `a`, `b` up to the fourth degree.
fn quartic_terms(a: f64, b: f64) -> [f64; 15] {
    let (a2, b2) = (a * a, b * b);
    [1.0, a, b, a2, a * b, b2, a2 * a, a2 * b, a * b2, b2 * b, a2 * a2, a2 * a * b, a2 * b2, a * b2 * b, b2 * b2]
}

/// `m x = v` by elimination with partial pivoting; `None` when `m` is (near) singular.
fn solve<const K: usize>(mut m: [[f64; K]; K], mut v: [f64; K]) -> Option<[f64; K]> {
    let scale = (0..K).map(|i| m[i][i].abs()).fold(0.0, f64::max).max(1e-300);
    for c in 0..K {
        let p = (c..K).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        if m[p][c].abs() < 1e-12 * scale {
            return None;
        }
        m.swap(c, p);
        v.swap(c, p);
        for r in c + 1..K {
            let f = m[r][c] / m[c][c];
            for k in c..K {
                m[r][k] -= f * m[c][k];
            }
            v[r] -= f * v[c];
        }
    }
    let mut x = [0.0f64; K];
    for c in (0..K).rev() {
        let s: f64 = (c + 1..K).map(|k| m[c][k] * x[k]).sum();
        x[c] = (v[c] - s) / m[c][c];
    }
    Some(x)
}

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(B3);
    (l.z * 64 + l.y * 8 + l.x) as usize
}

fn cell_of(i: usize) -> IVec3 {
    IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32)
}

fn trilinear(f: Vec3, c: [f32; 8]) -> f32 {
    let x00 = c[0] + (c[1] - c[0]) * f.x;
    let x10 = c[2] + (c[3] - c[2]) * f.x;
    let x01 = c[4] + (c[5] - c[4]) * f.x;
    let x11 = c[6] + (c[7] - c[6]) * f.x;
    let y0 = x00 + (x10 - x00) * f.y;
    let y1 = x01 + (x11 - x01) * f.y;
    y0 + (y1 - y0) * f.z
}

/// Gradient of the trilinear interpolation of a cell's corner values `c` at `f` (in cell units).
fn trilinear_gradient(f: Vec3, c: [f32; 8]) -> Vec3 {
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let dx = lerp(lerp(c[1] - c[0], c[3] - c[2], f.y), lerp(c[5] - c[4], c[7] - c[6], f.y), f.z);
    let dy = lerp(lerp(c[2] - c[0], c[3] - c[1], f.x), lerp(c[6] - c[4], c[7] - c[5], f.x), f.z);
    let dz = lerp(lerp(c[4] - c[0], c[5] - c[1], f.x), lerp(c[6] - c[2], c[7] - c[3], f.x), f.y);
    Vec3::new(dx, dy, dz)
}

/// A sample kept at 16 bits, on the same side of zero: the surface must not move by a voxel through rounding.
/// How far apart (in steps of the stroke's level) the surface's move is measured while detail is carried.
const MOVE_EVERY: i32 = 4;

/// `v` as the GPU reads it back from [`quantize`].
fn as_held(v: f32, band: f32) -> f32 {
    quantize(v, band) as f32 / 32767.0 * band
}

pub fn quantize(v: f32, band: f32) -> i16 {
    let x = v / band * 32767.0;
    // Rounded half away from zero as `f32::round` does, without its call (a library routine without SSE4): adding a
    // half is exact at these sizes but for the one value just under a half, which rounds to 0 on its own.
    let q = if x.abs() < 0.5 { 0 } else { ((x + 0.5f32.copysign(x)) as i32).clamp(-32767, 32767) as i16 };
    if v < 0.0 && q == 0 { -1 } else { q }
}

/// For every pattern of inside corners (bit `k` set when corner `k` is inside), how a cell's crossing edges group into
/// surface pieces: two bits per cube edge (its piece) in bits 0-23, the number of pieces in bits 24-26. Crossing edges
/// are linked across each face the way `remesh.rs` links them, a checkerboard face cutting off each inside corner, so
/// the two cells on either side of a face always agree.
pub fn piece_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (config, entry) in table.iter_mut().enumerate() {
        let inside = |k: usize| config >> k & 1 == 1;
        let crossing: [bool; 12] = std::array::from_fn(|e| inside(EDGES[e][0]) != inside(EDGES[e][1]));
        let mut parent: [u8; 12] = std::array::from_fn(|i| i as u8);
        for face in FACES {
            let edge = |k: usize| EDGE_OF[face[k]][face[(k + 1) % 4]] as usize;
            let on: Vec<usize> = (0..4).filter(|&k| crossing[edge(k)]).collect();
            match on.len() {
                2 => union(&mut parent, edge(on[0]), edge(on[1])),
                4 => {
                    for (k, &c) in face.iter().enumerate() {
                        if inside(c) {
                            union(&mut parent, edge((k + 3) % 4), edge(k));
                        }
                    }
                }
                _ => {}
            }
        }
        let mut piece_of_root = [u8::MAX; 12];
        let mut pieces = 0u32;
        let mut packed = 0u32;
        for (e, &crosses) in crossing.iter().enumerate() {
            if !crosses {
                continue;
            }
            let root = find(&mut parent, e);
            if piece_of_root[root] == u8::MAX {
                piece_of_root[root] = pieces.min(3) as u8;
                pieces += 1;
            }
            packed |= (piece_of_root[root] as u32) << (2 * e);
        }
        *entry = packed | pieces.min(4) << 24;
    }
    table
}

/// A block a [`Near`] read: stored, all one value (level 0's inside or outside), or made from the coarser level.
enum Src<'a> {
    Stored(&'a [f32; N]),
    Fill(f32),
    Made(Block),
}

impl Src<'_> {
    fn values(&self) -> Option<&[f32; N]> {
        match self {
            Src::Stored(b) => Some(b),
            Src::Made(b) => Some(b),
            Src::Fill(_) => None,
        }
    }

    fn at(&self, i: IVec3) -> f32 {
        match self {
            Src::Stored(b) => b[local(i)],
            Src::Fill(v) => *v,
            Src::Made(b) => b[local(i)],
        }
    }
}

/// Level `l`'s field over a box of blocks, read as [`LevelVolume::value_at`] reads it, bit for bit, for a lookup a
/// read: where no block is stored at `l`, a read interpolates through every coarser level down to a stored one (eight
/// corners a level, hundreds of lookups a read). Here each block in the box (and each coarser one under it) is made
/// once, the first time it is read, from the level below's as [`LevelVolume::sample`] makes each sample. Outside the
/// box, reads go to the volume.
struct Near<'a> {
    vol: &'a LevelVolume,
    l: usize,
    base: bool,
    /// Per level up to `l`: the first block of its box, the box's size, and a slot a block.
    boxes: Vec<(IVec3, IVec3, Vec<std::sync::OnceLock<Src<'a>>>)>,
}

impl<'a> Near<'a> {
    /// Over level `l`'s blocks `lo..=hi` (as the volume was when the stroke began, with `base`); with `on` false, no
    /// box: every read goes to the volume.
    fn new(vol: &'a LevelVolume, l: usize, lo: IVec3, hi: IVec3, base: bool, on: bool) -> Near<'a> {
        let mut boxes = Vec::new();
        if on {
            let (mut lo, mut hi) = (lo, hi);
            for _ in 0..=l {
                let n = hi - lo + IVec3::ONE;
                boxes.push((lo, n, (0..n.x * n.y * n.z).map(|_| std::sync::OnceLock::new()).collect()));
                // The blocks below read level `m - 1`'s points `4 bc ..= 4 bc + 4`.
                (lo, hi) = ((lo * 4).div_euclid(B3), (hi * 4 + IVec3::splat(4)).div_euclid(B3));
            }
            boxes.reverse();
        }
        Near { vol, l, base, boxes }
    }

    /// Level `m`'s block `bc`, when it is in the box.
    fn block(&self, m: usize, bc: IVec3) -> Option<&Src<'a>> {
        let (lo, n, slots) = self.boxes.get(m)?;
        let d = bc - *lo;
        if d.cmplt(IVec3::ZERO).any() || d.cmpge(*n).any() {
            return None;
        }
        Some(slots[((d.z * n.y + d.y) * n.x + d.x) as usize].get_or_init(|| self.make(m, bc)))
    }

    fn sample_at(&self, m: usize, i: IVec3) -> f32 {
        match self.block(m, i.div_euclid(B3)) {
            Some(b) => b.at(i),
            None => self.vol.sample(m, i, self.base),
        }
    }

    fn make(&self, m: usize, bc: IVec3) -> Src<'a> {
        if let Some(b) = self.vol.stored(m, bc, self.base) {
            return Src::Stored(b);
        }
        if m == 0 {
            let band = self.vol.band(0);
            return Src::Fill(if self.vol.inside0_at(bc, self.base) { -band } else { band });
        }
        // Level `m - 1`'s 5x5x5 points under the block, from the (up to) eight blocks holding them.
        let o = bc * 4;
        let lo = o.div_euclid(B3);
        let src: [Option<&Src>; 8] = std::array::from_fn(|q| self.block(m - 1, lo + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32)));
        let mut c = [0.0f32; 125];
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let p = o + IVec3::new(x, y, z);
                    let d = p.div_euclid(B3) - lo;
                    c[(25 * z + 5 * y + x) as usize] = match src[(d.x + 2 * d.y + 4 * d.z) as usize] {
                        Some(b) => b.at(p),
                        None => self.vol.sample(m - 1, p, self.base),
                    };
                }
            }
        }
        // As `sample` makes them: a point on the coarser lattice takes its value, any other the trilinear one.
        let band = self.vol.band(m);
        let mut out = Box::new([0.0f32; N]);
        for z in 0..B {
            for y in 0..B {
                for x in 0..B {
                    let i = bc * B + IVec3::new(x, y, z);
                    let v = if i.x & 1 == 0 && i.y & 1 == 0 && i.z & 1 == 0 {
                        c[(25 * (z / 2) + 5 * (y / 2) + x / 2) as usize]
                    } else {
                        let g = i.as_vec3() * 0.5;
                        let gi = g.floor().as_ivec3();
                        let f = g - gi.as_vec3();
                        let d = gi - o;
                        trilinear(f, std::array::from_fn(|k| c[(25 * (d.z + ((k >> 2) & 1) as i32) + 5 * (d.y + ((k >> 1) & 1) as i32) + d.x + (k & 1) as i32) as usize]))
                    };
                    out[local(i)] = v.clamp(-band, band);
                }
            }
        }
        Src::Made(out)
    }

    /// Level `l`'s field at lattice coordinates `g`.
    fn value_at(&self, g: Vec3) -> f32 {
        if self.boxes.is_empty() {
            return self.vol.value_at(self.l, g, self.base);
        }
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        // The cell within one block (most cells): look it up once.
        if i.rem_euclid(B3).cmplt(IVec3::splat(B - 1)).all()
            && let Some(b) = self.block(self.l, i.div_euclid(B3)).and_then(Src::values)
        {
            let at = local(i);
            return trilinear(f, std::array::from_fn(|k| b[at + (k & 1) + 8 * ((k >> 1) & 1) + 64 * ((k >> 2) & 1)]));
        }
        trilinear(f, std::array::from_fn(|k| self.sample_at(self.l, i + IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32))))
    }
}

/// Samples of one level over a box of lattice points, read once so that hot loops index an array instead of hashing.
struct Grid {
    lo: IVec3,
    n: IVec3,
    v: Vec<f32>,
}

impl Grid {
    fn idx(&self, i: IVec3) -> usize {
        let d = i - self.lo;
        ((d.z * self.n.y + d.y) * self.n.x + d.x) as usize
    }

    fn at(&self, i: IVec3) -> f32 {
        self.v[self.idx(i)]
    }

    /// Interpolated at `g` (lattice coordinates; it and the point past it must lie in the box).
    fn trilinear(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3().clamp(self.lo, self.lo + self.n - IVec3::splat(2));
        let f = (g - i.as_vec3()).clamp(Vec3::ZERO, Vec3::ONE);
        trilinear(f, std::array::from_fn(|k| self.at(i + IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32))))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LevelDab {
    pub ms: f64,
    pub level: usize,
    /// Blocks changed at the dab's level, made there, and changed at finer levels by carrying detail.
    pub changed: usize,
    pub created: usize,
    pub carried: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StrokeEnd {
    /// Specks removed, and blocks of coarser levels that took form from finer ones.
    pub specks: usize,
    pub rebased: usize,
    pub ms: f64,
}

/// One block's part of the extracted surface: what the GPU extractor writes for it.
#[derive(Clone, Default)]
pub struct BlockSurface {
    /// Positions in world units, and normals (the field's gradient, not normalised).
    pub verts: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Per cell: first vertex (bits 0-9), pieces (10-12), inside corners (13-20), and whether a finer cell touches it
    /// (bit 21, then it has one vertex whatever its pieces).
    pub cells: Vec<u32>,
    /// Quads whose lattice edge this block owns, each corner as (block, cell, piece): the vertex of that surface piece
    /// of that cell, which [`BlockSurface::vertex`] finds. Naming the cell rather than a vertex number keeps a quad
    /// valid when the block it points into is rebuilt and its vertices renumbered. Where two corners are one coarser
    /// cell the quad is a triangle with a repeated corner.
    pub quads: Vec<[Corner; 4]>,
}

/// A quad corner: (block, cell, piece).
pub type Corner = (Key, u16, u8);

impl BlockSurface {
    /// The vertex of surface piece `piece` of cell `cell` (the last piece if it has fewer), if the cell has one.
    pub fn vertex(&self, cell: u16, piece: u8) -> Option<u16> {
        let info = *self.cells.get(cell as usize)?;
        let count = (info >> 10) & 7;
        (count > 0).then(|| ((info & 1023) + (piece as u32).min(count - 1)) as u16)
    }
}

/// What [`LevelVolume::extract`] found.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtractStats {
    pub ms: f64,
    pub blocks: usize,
    pub verts: usize,
    pub quads: usize,
    pub triangles: usize,
    /// Quads dropped because a corner's cell had no vertex (a level 0 cell with no block, or a block over its vertex
    /// budget): holes, which should never happen.
    pub dropped: usize,
}

/// Which blocks of a level a list names: as bits over their box when that is small (a lookup is then a shift), else
/// as a set.
enum Named {
    None,
    Bits { lo: IVec3, n: IVec3, bits: Vec<u64> },
    Set(HashSet<IVec3>),
}

impl Named {
    /// Bits over a box of up to this many blocks (8 MB).
    const MOST_BITS: i64 = 1 << 26;

    fn of(keys: impl Iterator<Item = IVec3> + Clone) -> Named {
        let (mut lo, mut hi, mut count) = (IVec3::MAX, IVec3::MIN, 0usize);
        for c in keys.clone() {
            lo = lo.min(c);
            hi = hi.max(c);
            count += 1;
        }
        if count == 0 {
            return Named::None;
        }
        let n = hi - lo + IVec3::ONE;
        if n.x as i64 * n.y as i64 * n.z as i64 <= Self::MOST_BITS {
            let mut bits = vec![0u64; ((n.x as i64 * n.y as i64 * n.z as i64 + 63) / 64) as usize];
            for c in keys {
                let d = c - lo;
                let i = ((d.z * n.y + d.y) * n.x + d.x) as usize;
                bits[i >> 6] |= 1 << (i & 63);
            }
            Named::Bits { lo, n, bits }
        } else {
            let mut set: HashSet<IVec3> = HashSet::default();
            set.reserve(count);
            set.extend(keys);
            Named::Set(set)
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Named::None)
    }

    fn remove(&mut self, c: IVec3) {
        match self {
            Named::None => {}
            Named::Bits { lo, n, bits } => {
                let d = c - *lo;
                if !(d.cmplt(IVec3::ZERO).any() || d.cmpge(*n).any()) {
                    let i = ((d.z * n.y + d.y) * n.x + d.x) as usize;
                    bits[i >> 6] &= !(1 << (i & 63));
                }
            }
            Named::Set(s) => {
                s.remove(&c);
            }
        }
    }

    #[inline]
    fn contains(&self, c: IVec3) -> bool {
        match self {
            Named::None => false,
            Named::Bits { lo, n, bits } => {
                let d = c - *lo;
                if d.cmplt(IVec3::ZERO).any() || d.cmpge(*n).any() {
                    return false;
                }
                let i = ((d.z * n.y + d.y) * n.x + d.x) as usize;
                bits[i >> 6] & (1 << (i & 63)) != 0
            }
            Named::Set(s) => s.contains(&c),
        }
    }
}

/// What a view draws: the stored blocks of levels up to `cap`, less those in `hidden`. A hidden block's stored finer
/// blocks must be hidden too, so that what is shown nests as the stored blocks do (a shown block's parent is shown).
#[derive(Clone, Copy)]
pub struct Show<'a> {
    pub cap: usize,
    pub hidden: Option<&'a HashSet<Key>>,
}

impl Show<'static> {
    /// Every stored block of levels up to `cap`.
    pub fn upto(cap: usize) -> Show<'static> {
        Show { cap, hidden: None }
    }
}

impl Show<'_> {
    pub fn hides(&self, k: Key) -> bool {
        k.0 as usize > self.cap || self.hidden.is_some_and(|h| h.contains(&k))
    }
}

pub struct LevelVolume {
    pub voxel0: f32,
    origin: Vec3,
    blocks: Vec<Shared<IVec3, Stored>>,
    /// Level 0 blocks wholly inside the solid (any other level 0 block not stored is wholly outside).
    inside0: SharedSet<IVec3>,
    /// While a stroke is under way: each block as it was before the stroke first changed it. Dabs are placed on the
    /// surface as it was when the stroke began, and the stroke is undone from it.
    stroke: Option<Shared<Key, Was>>,
    history: Vec<Shared<Key, Was>>,
    redo: Vec<Shared<Key, Was>>,
    /// Blocks whose samples or existence changed since [`Self::drain_dirty`] last ran.
    dirty: HashSet<Key>,
    /// During a stroke: its change so far at level `carry_level` (the level its dabs write), per block, which finer
    /// levels' detail is carried by.
    carry: Shared<IVec3, Stored>,
    /// During a stroke: its dabs so far at `carry_level`, each the direction it pushed the surface in (its plane's
    /// normal) with its centre, radius and hardness, so that direction can be blended by the dabs' falloff at any
    /// point ([`Self::push_at`]). Detail is carried along it: along the surface's normal instead, it slides sideways
    /// wherever the stroke leaves a slope.
    carry_push: Vec<Push>,
    /// Blocks at `carry_level` dabs changed since the last [`Self::carry_take`]: a job's result lands after them.
    recent: HashSet<IVec3>,
    carry_level: usize,
    /// Finer blocks (per level) whose detail still has to be carried this stroke.
    pending: Vec<HashSet<IVec3>>,
    /// How far the stroke moved the surface ([`Self::moved_at`]) at level `carry_level` points, measured since its last
    /// dab: measuring takes a march, and many finer blocks read the same points.
    moved: HashMap<IVec3, f32>,
    /// Finer blocks (per level) this stroke carried detail into: each dab carries them again, since detail moved far
    /// enough can sit outside the dab's own reach, and would otherwise keep where an earlier dab put it.
    carried: Vec<HashSet<IVec3>>,
    /// How long a dab may spend carrying detail, in milliseconds; the rest waits for [`Self::carry_pending`]. `None`
    /// (the default) carries it all at once.
    pub carry_budget: Option<f64>,
    /// Whether a coarse stroke keeps the finer detail under it, carried along on the new surface (as sculpting at a
    /// lower subdivision level does). Off by default: the stroke works on the finer levels under it as it does on its
    /// own, as clay on one surface does, filling grooves and covering bumps wherever it builds over them.
    pub keep_detail: bool,
    /// Plane and smooth brushes read the surface under them point by point through the coarser levels, as before
    /// their blocks were made once a dab (see [`Near`]): the same values, many times slower. For tests that compare.
    #[doc(hidden)]
    pub uncached_reads: bool,
}

/// Queued detail to carry on another thread: a copy of the volume (see [`LevelVolume::carry_take`]) and, per level,
/// the blocks to carry.
pub struct CarryJob {
    vol: LevelVolume,
    plan: Vec<(usize, Vec<IVec3>)>,
    level: usize,
}

/// A stroke's end to work out on another thread (see [`LevelVolume::end_stroke_take`]).
pub struct EndJob {
    vol: LevelVolume,
}

/// What an [`EndJob`] did, for [`LevelVolume::end_stroke_apply`].
pub struct EndDone {
    vol: LevelVolume,
    end: StrokeEnd,
}

impl EndJob {
    pub fn run(mut self) -> EndDone {
        let t = Instant::now();
        let mut end = self.vol.stroke_end_steps();
        end.ms = t.elapsed().as_secs_f64() * 1e3;
        EndDone { vol: self.vol, end }
    }
}

/// What a [`CarryJob`] carried, for [`LevelVolume::carry_apply`].
pub struct CarryDone {
    /// Blocks it changed: their samples, or `None` where it dropped one.
    changes: Vec<(u8, IVec3, Option<Stored>)>,
    carried: Vec<(usize, IVec3)>,
    level: usize,
    /// Blocks rebuilt, and how long it took.
    pub blocks: usize,
    pub ms: f64,
}

impl CarryJob {
    /// Blocks it will carry.
    pub fn len(&self) -> usize {
        self.plan.iter().map(|p| p.1.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Carry it, coarser levels first: each level reads the one below as this job left it.
    pub fn run(mut self) -> CarryDone {
        let t = Instant::now();
        let mut blocks = 0;
        for (m, batch) in std::mem::take(&mut self.plan) {
            blocks += self.vol.carry_batch(self.level, m, &batch);
        }
        let v = &self.vol;
        let changes = v.dirty.iter().map(|&(m, bc)| (m, bc, v.blocks[m as usize].get(&bc).cloned())).collect();
        let carried = v.carried.iter().enumerate().flat_map(|(m, c)| c.iter().map(move |&bc| (m, bc))).collect();
        CarryDone { changes, carried, level: self.level, blocks, ms: t.elapsed().as_secs_f64() * 1e3 }
    }
}

impl LevelVolume {
    /// `field` (negative inside) over `[-half, half]^3` at level 0's step `voxel0`, with `levels` levels in all.
    pub fn from_field<F: Fn(Vec3) -> f32 + Sync>(field: &F, half: f32, voxel0: f32, levels: usize) -> LevelVolume {
        let levels = levels.clamp(1, MAX_LEVELS);
        let origin = Vec3::ZERO;
        let mut v = LevelVolume { voxel0, origin, blocks: vec![Shared::default(); levels], inside0: SharedSet::default(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::default(), carry: Shared::default(), carry_push: Vec::new(), recent: HashSet::default(), carry_level: 0, pending: vec![HashSet::default(); MAX_LEVELS], moved: HashMap::default(), carried: vec![HashSet::default(); MAX_LEVELS], carry_budget: None, keep_detail: false, uncached_reads: false };
        let band = v.band(0);
        let step = voxel0 * B as f32;
        let lo = ((Vec3::splat(-half) - origin) / step).floor().as_ivec3() - IVec3::ONE;
        let hi = ((Vec3::splat(half) - origin) / step).floor().as_ivec3() + IVec3::ONE;
        let n = hi - lo + IVec3::ONE;
        let reach = 0.5 * B as f32 * voxel0 * 3f32.sqrt() + band;
        let built: Vec<(IVec3, Option<Block>, bool)> = (0..n.x * n.y * n.z)
            .into_par_iter()
            .map(|k| {
                let bc = lo + IVec3::new(k % n.x, (k / n.x) % n.y, k / (n.x * n.y));
                let corner = origin + (bc * B).as_vec3() * voxel0;
                let c = field(corner + Vec3::splat(3.5 * voxel0));
                if c.abs() > reach {
                    return (bc, None, c < 0.0);
                }
                let mut s = Box::new([0.0f32; N]);
                for (i, x) in s.iter_mut().enumerate() {
                    *x = field(corner + cell_of(i).as_vec3() * voxel0).clamp(-band, band);
                }
                (bc, Some(s), false)
            })
            .collect();
        for (bc, s, inside) in built {
            match s {
                Some(s) => {
                    v.blocks[0].insert(bc, s);
                    v.dirty.insert((0, bc));
                }
                None if inside => {
                    v.inside0.insert(bc);
                }
                None => {}
            }
        }
        v
    }

    pub fn levels(&self) -> usize {
        self.blocks.len()
    }

    pub fn voxel(&self, l: usize) -> f32 {
        self.voxel0 / (1u32 << l) as f32
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Distances kept at level `l`: 2.5 lattice steps, but 16 at level 0, so an edit there can carry finer detail as far
    /// as it moves the surface.
    pub fn band(&self, l: usize) -> f32 {
        if l == 0 { 16.0 * self.voxel0 } else { 2.5 * self.voxel(l) }
    }

    /// How many steps from the surface [`Self::redistance`] measures at a level with step `vx` and band `band`: the
    /// whole band.
    fn redistance_reach(vx: f32, band: f32) -> i32 {
        (band / vx).ceil() as i32 + 1
    }

    /// How far from the surface level `l` holds distances: its band. A sample at the band holds only a bound on its
    /// distance. (Level 0 once held them only six steps out, and a plane dab read the rest from the ground under the
    /// brush, see [`Ground`]; where that estimate and its neighbours' own distances disagreed, a wall's foot jumped
    /// from sample to sample, and a later stroke raising the surface six steps over an earlier one showed where the two
    /// met as a crease.)
    pub fn plateau(&self, l: usize) -> f32 {
        let (vx, band) = (self.voxel(l), self.band(l));
        band.min(Self::redistance_reach(vx, band) as f32 * vx)
    }

    /// Steps from the surface level 0 is drawn from (see [`Self::sent_reach`]).
    pub const SENT0_STEPS: f32 = 6.0;

    /// How far from the surface level `l`'s samples are drawn from: its band, but at level 0 six steps. The GPU is
    /// sent each sample clamped to it, and the reference extraction reads them so. Nothing drawn reads one farther out
    /// (a cell the surface crosses has every corner within about two steps of it), and a wide stroke at level 0 changes
    /// a slab of samples sixteen steps either side of the surface across its whole disc: the blocks it changes only out
    /// there come out as the GPU holds them already, and are neither sent nor extracted again.
    pub fn sent_reach(&self, l: usize) -> f32 {
        if l == 0 { Self::SENT0_STEPS * self.voxel0 } else { self.band(l) }
    }

    /// Stored blocks per level, and the bytes their samples take at 16 bits.
    pub fn memory(&self) -> (Vec<usize>, usize) {
        let counts: Vec<usize> = self.blocks.iter().map(|m| m.len()).collect();
        let bytes = counts.iter().sum::<usize>() * N * 2;
        (counts, bytes)
    }

    pub fn block(&self, k: Key) -> Option<&[f32; N]> {
        self.blocks.get(k.0 as usize)?.get(&k.1).map(|b| &**b)
    }

    pub fn exists(&self, l: usize, bc: IVec3) -> bool {
        self.blocks[l].contains_key(&bc)
    }

    /// The stored blocks of level `l`, in no order.
    pub fn level_keys(&self, l: usize) -> Vec<IVec3> {
        self.blocks.get(l).map_or(Vec::new(), |m| m.keys().copied().collect())
    }

    /// How many blocks level `l` stores.
    pub fn level_len(&self, l: usize) -> usize {
        self.blocks.get(l).map_or(0, |m| m.len())
    }

    /// Every stored block.
    pub fn keys(&self) -> Vec<Key> {
        let mut out: Vec<Key> = self.blocks.iter().enumerate().flat_map(|(l, m)| m.keys().map(move |bc| (l as u8, *bc))).collect();
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out
    }

    /// Blocks whose samples or existence changed since the last call.
    /// Whether block `k` changed since [`Self::drain_dirty`] last ran (for checks).
    pub fn dirty_contains(&self, k: Key) -> bool {
        self.dirty.contains(&k)
    }

    /// The blocks changed since [`Self::drain_dirty`] last ran, without taking them.
    pub fn dirty_keys(&self) -> Vec<Key> {
        self.dirty.iter().copied().collect()
    }

    /// Count blocks `keys` as changed (for whoever drains them next).
    pub fn mark_dirty(&mut self, keys: impl IntoIterator<Item = Key>) {
        self.dirty.extend(keys);
    }

    /// A copy to draw a stroke on for a while, before it is done for real (see `Layered::preview_copy`): made in a
    /// moment, sharing its maps' shards with this one; no history, nothing changed yet.
    pub fn preview_copy(&self) -> LevelVolume {
        let mut out = self.snapshot();
        out.carry_budget = self.carry_budget;
        out.keep_detail = self.keep_detail;
        out.uncached_reads = self.uncached_reads;
        out
    }

    /// Whether the blocks (and level 0's inside blocks) are still those of `other`, a snapshot of this volume: nothing
    /// written to either since, in a moment whatever their number.
    pub(crate) fn same_blocks(&self, other: &LevelVolume) -> bool {
        self.blocks.len() == other.blocks.len() && self.blocks.iter().zip(&other.blocks).all(|(a, b)| a.same_as(b)) && self.inside0.0.same_as(&other.inside0.0)
    }

    /// Take `other`'s blocks (and level 0's inside blocks) in place of these, and count its changed blocks as changed
    /// here too; the rest (history, a stroke under way, the blocks still to send) stays. Returns `other`, holding
    /// what this volume held.
    pub(crate) fn take_blocks(&mut self, mut other: LevelVolume) -> LevelVolume {
        std::mem::swap(&mut self.blocks, &mut other.blocks);
        std::mem::swap(&mut self.inside0, &mut other.inside0);
        self.dirty.extend(other.dirty.drain());
        other
    }

    pub fn drain_dirty(&mut self) -> Vec<Key> {
        let mut out: Vec<Key> = self.dirty.drain().collect();
        out.par_sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out
    }

    /// The box the stored blocks cover.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let step = self.voxel0 * B as f32;
        let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        for bc in self.blocks[0].keys() {
            lo = lo.min(*bc);
            hi = hi.max(*bc);
        }
        if lo.x > hi.x {
            return (Vec3::splat(-1.0), Vec3::splat(1.0));
        }
        (self.origin + lo.as_vec3() * step, self.origin + (hi + IVec3::ONE).as_vec3() * step)
    }

    // ------------------------------------------------------------------ reading the field

    /// Block `bc` of level `l`: as it is, or with `base` during a stroke, as it was when the stroke began.
    fn stored(&self, l: usize, bc: IVec3, base: bool) -> Option<&[f32; N]> {
        if base && let Some(was) = self.stroke.as_ref().and_then(|s| s.get(&(l as u8, bc))) {
            return match was {
                Was::Stored(b) => Some(b),
                _ => None,
            };
        }
        self.blocks[l].get(&bc).map(|b| &**b)
    }

    fn inside0_at(&self, bc: IVec3, base: bool) -> bool {
        if base && let Some(was) = self.stroke.as_ref().and_then(|s| s.get(&(0, bc))) {
            return matches!(was, Was::Inside);
        }
        self.inside0.contains(&bc)
    }

    fn exists_at(&self, l: usize, bc: IVec3, base: bool) -> bool {
        self.stored(l, bc, base).is_some()
    }

    /// The sample at level `l`'s lattice point `i`: stored, or else the coarser level's, interpolated.
    pub fn sample(&self, l: usize, i: IVec3, base: bool) -> f32 {
        let bc = i.div_euclid(B3);
        if let Some(b) = self.stored(l, bc, base) {
            return b[local(i)];
        }
        if l == 0 {
            return if self.inside0_at(bc, base) { -self.band(0) } else { self.band(0) };
        }
        let band = self.band(l);
        if i.x & 1 == 0 && i.y & 1 == 0 && i.z & 1 == 0 {
            return self.sample(l - 1, i.div_euclid(IVec3::splat(2)), base).clamp(-band, band);
        }
        self.value_at(l - 1, i.as_vec3() * 0.5, base).clamp(-band, band)
    }

    /// Level `l`'s field at lattice coordinates `g` (in level `l` steps), interpolated.
    fn value_at(&self, l: usize, g: Vec3, base: bool) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        // The cell within one stored block (most cells): look it up once.
        let li = i.rem_euclid(B3);
        if li.cmplt(IVec3::splat(B - 1)).all()
            && let Some(b) = self.stored(l, i.div_euclid(B3), base)
        {
            let at = local(i);
            return trilinear(f, std::array::from_fn(|k| b[at + (k & 1) + 8 * ((k >> 1) & 1) + 64 * ((k >> 2) & 1)]));
        }
        trilinear(f, std::array::from_fn(|k| self.sample(l, i + IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32), base)))
    }

    /// The finest level up to `max` holding a block at world point `p`.
    fn level_at(&self, p: Vec3, max: usize, base: bool) -> usize {
        for l in (1..=max.min(self.levels() - 1)).rev() {
            let g = (p - self.origin) / self.voxel(l);
            if self.exists_at(l, g.floor().as_ivec3().div_euclid(B3), base) {
                return l;
            }
        }
        0
    }

    /// The field at world point `p`, from the finest level up to `max` that holds it.
    pub fn value_upto(&self, p: Vec3, max: usize, base: bool) -> f32 {
        let l = self.level_at(p, max, base);
        self.value_at(l, (p - self.origin) / self.voxel(l), base)
    }

    /// The finest level holding a block at world point `p`.
    pub fn level_of(&self, p: Vec3) -> usize {
        self.level_at(p, self.levels() - 1, false)
    }

    /// The field with every level, as it is now.
    pub fn value(&self, p: Vec3) -> f32 {
        self.value_upto(p, self.levels() - 1, false)
    }

    /// Outward normal at `p` of the surface of levels up to `max`.
    pub fn normal_upto(&self, p: Vec3, max: usize, base: bool) -> Vec3 {
        let h = self.voxel(self.level_at(p, max, base));
        let d = |o: Vec3| self.value_upto(p + o, max, base) - self.value_upto(p - o, max, base);
        Vec3::new(d(Vec3::X * h), d(Vec3::Y * h), d(Vec3::Z * h)).normalize_or_zero()
    }

    /// Where a ray from `origin` along `dir` (normalised) first enters the solid of levels up to `max` (as it was when
    /// the stroke began, with `base`), with the normal there.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max: usize, base: bool) -> Option<(Vec3, Vec3)> {
        let value = |p: Vec3| self.value_upto(p, max, base);
        let fine = self.voxel(max.min(self.levels() - 1));
        let (lo, hi) = self.bounds();
        let (lo, hi) = (lo - Vec3::splat(self.band(0)), hi + Vec3::splat(self.band(0)));
        let inv = dir.recip();
        let (t0, t1) = ((lo - origin) * inv, (hi - origin) * inv);
        let (near, far) = (t0.min(t1).max_element().max(0.0), t0.max(t1).min_element());
        if near > far {
            return None;
        }
        let (mut t, mut prev) = (near, near);
        let mut hit = false;
        let max = max.min(self.levels() - 1);
        while t < far {
            // March by the field of the finest level at the point, less what the levels finer than it could raise the
            // surface ahead (at most their bands added up, under 2.5 of this level's steps): a coarser level knows
            // nothing of the detail in the finer blocks just ahead.
            let q = origin + dir * t;
            let lp = self.level_at(q, max, base);
            let v = self.value_at(lp, (q - self.origin) / self.voxel(lp), base);
            if v < 0.0 {
                hit = true;
                break;
            }
            prev = t;
            let finer = if lp < max { 2.5 * self.voxel(lp) } else { 0.0 };
            t += (v - finer).max(0.5 * fine);
        }
        if !hit {
            return None;
        }
        let (mut a, mut b) = (prev, t);
        for _ in 0..18 {
            let m = 0.5 * (a + b);
            if value(origin + dir * m) < 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        let p = origin + dir * b;
        Some((p, self.normal_upto(p, max, base)))
    }

    /// Level `l`'s samples over the lattice box `lo..=hi`, stored or interpolated from coarser levels.
    fn grid(&self, l: usize, lo: IVec3, hi: IVec3, base: bool) -> Grid {
        self.grid_hiding(l, lo, hi, base, None)
    }

    /// [`Self::grid`] as if the blocks in `hidden` were not stored: what a view that hides them reads.
    fn grid_hiding(&self, l: usize, lo: IVec3, hi: IVec3, base: bool, hidden: Option<&HashSet<Key>>) -> Grid {
        let n = hi - lo + IVec3::ONE;
        let mut g = Grid { lo, n, v: vec![f32::NAN; (n.x * n.y * n.z) as usize] };
        let (blo, bhi) = (lo.div_euclid(B3), hi.div_euclid(B3));
        let mut missing = false;
        let band = self.band(l);
        for bz in blo.z..=bhi.z {
            for by in blo.y..=bhi.y {
                for bx in blo.x..=bhi.x {
                    let bc = IVec3::new(bx, by, bz);
                    let a = (bc * B).max(lo);
                    let b = (bc * B + IVec3::splat(B - 1)).min(hi);
                    let stored = self.stored(l, bc, base).filter(|_| !hidden.is_some_and(|h| h.contains(&(l as u8, bc))));
                    let fill = if l == 0 && stored.is_none() { Some(if self.inside0_at(bc, base) { -band } else { band }) } else { None };
                    if stored.is_none() && fill.is_none() {
                        missing = true;
                        continue;
                    }
                    for z in a.z..=b.z {
                        for y in a.y..=b.y {
                            for x in a.x..=b.x {
                                let i = IVec3::new(x, y, z);
                                let at = g.idx(i);
                                g.v[at] = match stored {
                                    Some(s) => s[local(i)],
                                    None => fill.unwrap(),
                                };
                            }
                        }
                    }
                }
            }
        }
        if missing {
            let coarse = self.grid_hiding(l - 1, lo.div_euclid(IVec3::splat(2)), (hi + IVec3::ONE).div_euclid(IVec3::splat(2)), base, hidden);
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let i = IVec3::new(x, y, z);
                        let at = g.idx(i);
                        if g.v[at].is_nan() {
                            g.v[at] = coarse.trilinear(i.as_vec3() * 0.5).clamp(-band, band);
                        }
                    }
                }
            }
        }
        g
    }

    /// Level `l`'s samples over the lattice box `lo..=hi` (x fastest), stored or interpolated from coarser levels.
    pub(crate) fn samples(&self, l: usize, lo: IVec3, hi: IVec3) -> Vec<f32> {
        self.grid(l, lo, hi, false).v
    }

    /// The direction the stroke under way (or the last one) pushed the surface at world point `p` where its normal is
    /// `n` (see [`Self::carry_dir`]).
    pub(crate) fn push_dir(&self, p: Vec3, n: Vec3) -> Vec3 {
        self.carry_dir((p - self.origin) / self.voxel(self.carry_level), n)
    }

    /// Make block `k` here what it is in `src` (stored, missing, or wholly inside), outside any stroke. A block finer
    /// levels here still hang from is kept, as the coarser level shows it.
    pub(crate) fn mirror(&mut self, src: &LevelVolume, k: Key) {
        let (l, bc) = (k.0 as usize, k.1);
        if let Some(b) = src.blocks[l].get(&bc) {
            self.put_stored(l, bc, b.clone());
            return;
        }
        let had = self.blocks[l].remove(&bc).is_some();
        if l == 0 {
            if src.inside0.contains(&bc) {
                self.inside0.insert(bc);
            } else {
                self.inside0.remove(&bc);
            }
        }
        self.dirty.insert(k);
        let children = l + 1 < self.levels() && (0..8).any(|q| self.blocks[l + 1].contains_key(&(bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1))));
        if had && children {
            self.ensure(l, bc);
        }
    }

    /// Make block `bc` of level `l` again from the coarser level, outside any stroke (a block held only as a parent,
    /// whose coarser level changed).
    pub(crate) fn refill(&mut self, l: usize, bc: IVec3) {
        if self.blocks[l].remove(&bc).is_some() {
            let s = self.block_values(l, bc, false);
            self.blocks[l].insert(bc, Stored::from(s));
            self.dirty.insert((l as u8, bc));
        }
    }

    /// Remove every block of level `l` (and finer levels' blocks under none), outside any stroke.
    pub(crate) fn remove_level(&mut self, l: usize) {
        let keys: Vec<IVec3> = self.blocks[l].keys().copied().collect();
        for bc in keys {
            self.blocks[l].remove(&bc);
            self.dirty.insert((l as u8, bc));
        }
    }

    /// The blocks the last stroke kept for undo changed.
    pub fn last_stroke_keys(&self) -> Vec<Key> {
        self.history.last().map(|h| h.keys().copied().collect()).unwrap_or_default()
    }

    /// Drop the last stroke from the history (something else undoes it).
    pub fn forget_last(&mut self) {
        self.history.pop();
    }

    /// Drop all the history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.redo.clear();
    }


    /// Block `bc` of level `l`'s 512 samples, stored or interpolated from coarser levels.
    pub(crate) fn block_values(&self, l: usize, bc: IVec3, base: bool) -> Block {
        if let Some(s) = self.stored(l, bc, base) {
            return Box::new(*s);
        }
        let g = self.grid(l, bc * B, bc * B + IVec3::splat(B - 1), base);
        let mut s = Box::new([0.0f32; N]);
        s.copy_from_slice(&g.v);
        s
    }

    /// The 9x9x9 samples block `bc` of level `l`'s cells read: its own and the first layer of the blocks above it, at
    /// this level (stored or interpolated). What the GPU holds per block.
    /// [`Self::gather9`] into `out`, without allocating where the block and the seven past it are all stored at its
    /// level (most blocks): their samples are copied straight across.
    pub fn gather9_into(&self, l: usize, bc: IVec3, out: &mut [f32; 729]) {
        self.gather9_hiding(l, bc, out, None)
    }

    /// [`Self::gather9_into`] as a view hiding the blocks in `hidden` reads them: the first layer past the block,
    /// where it falls in a hidden block, is interpolated from the coarser levels shown, as it is where no block is
    /// stored. So both sides of a face between a shown block and a hidden one read the same values on it.
    pub fn gather9_hiding(&self, l: usize, bc: IVec3, out: &mut [f32; 729], hidden: Option<&HashSet<Key>>) {
        let nb: [Option<&Stored>; 8] = std::array::from_fn(|q| {
            let c = bc + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32);
            self.blocks[l].get(&c).filter(|_| !hidden.is_some_and(|h| h.contains(&(l as u8, c))))
        });
        if nb.iter().all(|b| b.is_some()) {
            let b: [&[f32; N]; 8] = std::array::from_fn(|q| &**nb[q].unwrap());
            for z in 0..9usize {
                for y in 0..9usize {
                    // A row of eight from the block holding it, and the one past it from the block along x.
                    let q = ((y >> 3) << 1) | ((z >> 3) << 2);
                    let row = 8 * (y & 7) + 64 * (z & 7);
                    let o = 9 * y + 81 * z;
                    out[o..o + 8].copy_from_slice(&b[q][row..row + 8]);
                    out[o + 8] = b[q | 1][row];
                }
            }
            return;
        }
        out.copy_from_slice(&self.grid_hiding(l, bc * B, bc * B + IVec3::splat(B), false, hidden).v);
    }

    fn gather9_in(&self, l: usize, bc: IVec3, v: &Show) -> Box<[f32; 729]> {
        let mut out = Box::new([0.0f32; 729]);
        self.gather9_hiding(l, bc, &mut out, v.hidden);
        out
    }

    pub fn gather9(&self, l: usize, bc: IVec3) -> Box<[f32; 729]> {
        let g = self.grid(l, bc * B, bc * B + IVec3::splat(B), false);
        let mut out = Box::new([0.0f32; 729]);
        out.copy_from_slice(&g.v);
        out
    }

    /// About how many quads block `bc` of level `l` gives the surface when no finer block covers it: the lattice edges
    /// its surface crosses, counted among its own samples and scaled for the edges into the blocks past it.
    pub fn surface_quads(&self, l: usize, bc: IVec3) -> u32 {
        let Some(s) = self.blocks.get(l).and_then(|m| m.get(&bc)) else { return 0 };
        let mut n = 0u32;
        for z in 0..8usize {
            for y in 0..8usize {
                for x in 0..8usize {
                    let i = x + 8 * y + 64 * z;
                    let neg = s[i] < 0.0;
                    n += (x < 7 && (s[i + 1] < 0.0) != neg) as u32 + (y < 7 && (s[i + 8] < 0.0) != neg) as u32 + (z < 7 && (s[i + 64] < 0.0) != neg) as u32;
                }
            }
        }
        n * 8 / 7
    }

    /// How far the surface of block `bc` at level `l` lies from the level below's (the most a view leaving this level
    /// out would be off by, near the surface).
    pub fn detail_of(&self, l: usize, bc: IVec3) -> f32 {
        if l == 0 {
            return 0.0;
        }
        let Some(s) = self.blocks[l].get(&bc) else { return 0.0 };
        let band = self.band(l);
        let coarse = self.grid(l - 1, (bc * B).div_euclid(IVec3::splat(2)), (bc * B + IVec3::splat(B)).div_euclid(IVec3::splat(2)), false);
        let mut most = 0.0f32;
        for (i, v) in s.iter().enumerate() {
            if v.abs() < 0.8 * band {
                let c = coarse.trilinear((bc * B + cell_of(i)).as_vec3() * 0.5).clamp(-band, band);
                most = most.max((v - c).abs());
            }
        }
        most
    }

    // ------------------------------------------------------------------ changing the field

    /// During a stroke, remember block `k` as it was before the stroke first changes it.
    fn keep(&mut self, k: Key) {
        let Some(stroke) = self.stroke.as_mut() else { return };
        if stroke.contains_key(&k) {
            return;
        }
        let was = match self.blocks[k.0 as usize].get(&k.1) {
            Some(b) => Was::Stored(b.clone()),
            None if k.0 == 0 && self.inside0.contains(&k.1) => Was::Inside,
            None => Was::Missing,
        };
        stroke.insert(k, was);
    }

    /// Store `s` as block `bc` of level `l`, making its parents first if they do not exist.
    fn put(&mut self, l: usize, bc: IVec3, s: Block) {
        self.put_stored(l, bc, Stored::from(s));
    }

    pub(crate) fn put_stored(&mut self, l: usize, bc: IVec3, s: Stored) {
        if !self.blocks[l].contains_key(&bc) && l > 0 {
            self.ensure(l - 1, bc.div_euclid(IVec3::splat(2)));
        }
        self.keep((l as u8, bc));
        if l == 0 {
            self.inside0.remove(&bc);
        }
        self.blocks[l].insert(bc, s);
        self.dirty.insert((l as u8, bc));
    }

    /// Make block `bc` of level `l` (from the coarser level) and its parents, if they do not exist.
    pub(crate) fn ensure(&mut self, l: usize, bc: IVec3) {
        if self.blocks[l].contains_key(&bc) {
            return;
        }
        let s = self.block_values(l, bc, false);
        self.put(l, bc, s);
    }

    /// Combine a shape into level `l`'s field: `shape` (a distance, negative inside) lies within `reach` of `c` and in
    /// the slab `lo..hi` along `n` from `c`. Each sample moves `weight` of the way to the field combined with it, so
    /// overlapping dabs blend into one smooth change. Then the finer levels' detail moves with the surface.
    #[allow(clippy::too_many_arguments)]
    /// `push`: the dab as a direction the surface is moved in (a plane brush's), which finer detail is carried along;
    /// `None` carries it along the surface's normal. `set`: instead of combining with the shape, each sample moves
    /// toward the shape's distance itself (the smooth brush's target surface; infinite where it has none), but only
    /// down to `set` under that surface (fading over its last 0.3), so the brush leaves alone what lies deeper.
    #[allow(clippy::too_many_arguments)]
    fn modify(&mut self, l: usize, c: Vec3, reach: f32, k: f32, carve: bool, slab: (Vec3, f32, f32), push: Option<Push>, set: Option<f32>, weight: &(dyn Fn(Vec3) -> f32 + Sync), shape: &(dyn Fn(Vec3) -> f32 + Sync)) -> LevelDab {
        let t = Instant::now();
        let mut stats = LevelDab { level: l, ..Default::default() };
        let (vx, band) = (self.voxel(l), self.band(l));
        let step = vx * B as f32;
        let half = 3.5 * vx;
        let diag = half * 3f32.sqrt();
        let margin = diag + band + k;
        let lo = ((c - Vec3::splat(reach + margin) - self.origin) / step).floor().as_ivec3();
        let hi = ((c + Vec3::splat(reach + margin) - self.origin) / step).floor().as_ivec3();
        let (n, s_lo, s_hi) = slab;
        let mut cand = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let bc = IVec3::new(x, y, z);
                    let centre = self.origin + (bc * B).as_vec3() * vx + Vec3::splat(half);
                    if (centre - c).length() > reach + margin {
                        continue;
                    }
                    let u = (centre - c).dot(n);
                    if u < s_lo - margin || u > s_hi + margin {
                        continue;
                    }
                    // Beyond a plane brush's radius from its axis its weight is nothing: a block whose every sample
                    // lies there, and every finer level's sample within its cells, changes nothing (the distance from
                    // the axis moves no faster than the point, and those lie within 4.5 steps of the centre per axis).
                    if let Some(p) = push {
                        let q = centre - p.on;
                        if (q - p.n * q.dot(p.n)).length() >= p.r + 4.5 * vx * 3f32.sqrt() {
                            continue;
                        }
                    }
                    // Far outside the shape, combining changes nothing. (Setting is limited by the slab and the
                    // brush's disc alone.)
                    let sd = if set.is_some() { 0.0 } else { shape(centre) };
                    if sd > diag + band {
                        continue;
                    }
                    // Deep inside it, nothing changes where the field is already all `fill` (inside for clay).
                    if sd < -(diag + band) {
                        let f = self.value_upto(centre, l, false);
                        if (!carve && f < -(diag + band)) || (carve && f > diag + band) {
                            continue;
                        }
                    }
                    cand.push(bc);
                }
            }
        }
        // A plane brush's dab blends each sample part of the way, so it needs the samples' distances; those at its
        // band (only a bound) read theirs from the surface under the brush. (A finer level's sample at its band reads
        // the levels below.) Above level 0 too: there the band is two and a half steps, less than a close trim cuts
        // (three), and blended from the bound, a sample held there moved less than its neighbours just inside it, so
        // the new surface rippled along the line between them, where the lattice's planes crossed the surface the
        // stroke began on: stripes every few steps, a tenth of a step deep, across a trim on a slope.
        let plateaus: Vec<f32> = (0..self.levels()).map(|k| self.plateau(k)).collect();
        let plateau = plateaus[l];
        let ground = if set.is_some() { None } else { push.map(|p| self.ground(l, c, n, p.r, s_hi + band, s_lo - 2.0 * band, false)) };
        let ground = ground.as_ref();
        // A sample's distance to the surface for the dab to blend from: its value, unless that is only a bound and the
        // ground under the brush says how far the surface is (on the same side, and farther than the bound: a bound
        // says the surface is at least that far, and the ground's tangent plane comes short of a surface curving away
        // under it). Only the bound itself (within what holding it at 16 bits moves it) reads the ground: a sample a
        // dab already blended holds a distance of its own. Set back to the surface's distance each dab, a sample the
        // dabs kept past the bound stood still while its neighbours just under the bound accumulated the dabs', and
        // the mismatch along that line rippled the wall the dabs raise. And the choice is continuous in the ground's
        // distance: taking it only when it came within 0.9 of the bound, and the bound otherwise, jumped by a tenth of
        // the bound from one sample to the next where the ground crossed that line, out on a wide brush's falloff where
        // the form curves away from its plane, and every dab there left a row of pits along its rim.
        let tolerance = 0.01 * vx;
        let from_ground = |p: Vec3, v: f32| -> f32 {
            if let Some(g) = ground
                && (v.abs() - plateau).abs() <= tolerance
                && let Some(d) = g.distance(p)
                && (d < 0.0) == (v < 0.0)
                && d.abs() > v.abs()
            {
                return d;
            }
            v
        };
        // Setting: the weight a sample `blob` from the target surface moves toward it with (none deep under it).
        let set_weight = move |blob: f32, w: f32| -> f32 {
            let Some(deep) = set else { return w };
            if !blob.is_finite() {
                return 0.0;
            }
            let x = ((-blob / deep - 0.7) / 0.3).clamp(0.0, 1.0);
            w * (1.0 - x * x * (3.0 - 2.0 * x))
        };
        // New samples for each candidate, worked out in parallel from the field as it is.
        let origin = self.origin;
        let fill = if carve { band } else { -band };
        let me = &*self;
        let built: Vec<(IVec3, Stored, Stored, SignMask, Block)> = cand
            .par_iter()
            .filter_map(|&bc| {
                // A sample at the band on the side the dab pushes toward (inside for clay) stays there: the dab moves
                // it no farther than its own value, and the clamp brings it back. A block of only such samples (the
                // slab of a wide dab reaches sixteen steps into the solid across its disc, all of it at the band where
                // no stroke has been yet) changes nothing, and is passed over.
                let at_fill = set.is_none()
                    && match me.blocks[l].get(&bc) {
                        Some(b) => b.iter().all(|&v| v == fill),
                        None => l == 0 && me.inside0.contains(&bc) != carve,
                    };
                if at_fill {
                    return None;
                }
                let old = me.block_values(l, bc, false);
                // The block as the stroke found it. Only a sample still holding what it held then can hold the bound:
                // one a dab of this stroke blended holds a distance of its own, even where it came out within a
                // hundredth of a step of the bound (a little blending out on a wide brush's falloff does that). Read as
                // the bound, the next dab took the ground for it while its neighbours went on from their own values,
                // and it stood out from them by up to half a step: a pit, once the clay reached it.
                let found = ground.map(|_| me.block_values(l, bc, true));
                let mut s = old.clone();
                let mut d = Box::new([0.0f32; N]);
                let mut flips = [0u64; 8];
                let mut any = false;
                let corner = origin + (bc * B).as_vec3() * vx;
                for (i, v) in s.iter_mut().enumerate() {
                    if *v == fill && set.is_none() {
                        continue;
                    }
                    let p = corner + cell_of(i).as_vec3() * vx;
                    let w = weight(p);
                    if w <= 0.0 {
                        continue;
                    }
                    let blob = shape(p);
                    if set.is_some() {
                        let w = set_weight(blob, w);
                        if w <= 0.0 {
                            continue;
                        }
                        let next = (*v + w * (blob - *v)).clamp(-band, band);
                        if next != *v {
                            if (next < 0.0) != (*v < 0.0) {
                                flips[i / 64] |= 1 << (i % 64);
                            }
                            d[i] = next - *v;
                            *v = next;
                            any = true;
                        }
                        continue;
                    }
                    // Only a sample the dab changes, and blends (a full-weight dab takes the shape outright), needs
                    // its distance.
                    let changes = if carve { -blob > *v } else { blob < *v };
                    let held = found.as_ref().is_some_and(|f| f[i].to_bits() == v.to_bits());
                    let was = if changes && w < 1.0 && held { from_ground(p, *v) } else { *v };
                    let full = if carve { was.max(-blob) } else { was.min(blob) };
                    let next = (was + w * (full - was)).clamp(-band, band);
                    if next != *v {
                        if (next < 0.0) != (*v < 0.0) {
                            flips[i / 64] |= 1 << (i % 64);
                        }
                        // The stroke's change of the distance here (what carrying detail reads as the surface's move),
                        // not of the bound the sample held.
                        d[i] = next - was;
                        *v = next;
                        any = true;
                    }
                }
                // Made shared here, in parallel: a wide dab at level 0 changes thousands of blocks.
                any.then(|| {
                    let moved: Block = Box::new(std::array::from_fn(|i| s[i] - old[i]));
                    (bc, Stored::from(s), Stored::from(d), flips, moved)
                })
            })
            .collect();
        // Unless the finer detail is kept (carried), the finer levels under the dab take it as their own: each of their
        // stored blocks under it, from the field as it is.
        let finer: Vec<(usize, IVec3, Block)> = if self.keep_detail || l + 1 >= self.levels() {
            Vec::new()
        } else {
            // How far the dab moves level `l`'s field at lattice coordinates `g` (its samples' moves, interpolated).
            let moved: HashMap<IVec3, &Block> = built.iter().map(|(bc, _, _, _, m)| (*bc, m)).collect();
            let moved = &moved;
            let moved_at = move |g: Vec3| -> f32 {
                let i = g.floor().as_ivec3();
                let f = g - i.as_vec3();
                trilinear(f, std::array::from_fn(|k| {
                    let c = i + IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32);
                    moved.get(&c.div_euclid(B3)).map_or(0.0, |b| b[local(c)])
                }))
            };
            let under: HashSet<IVec3> = cand.iter().copied().collect();
            let under = &under;
            let blocks: Vec<(usize, IVec3)> = (l + 1..self.levels())
                .flat_map(|m| {
                    let s = (m - l) as i32;
                    self.blocks[m].keys().filter(|c| under.contains(&IVec3::new(c.x >> s, c.y >> s, c.z >> s))).map(move |&c| (m, c)).collect::<Vec<_>>()
                })
                .collect();
            let me = &*self;
            blocks
                .par_iter()
                .filter_map(|&(m, bc)| {
                    let (vm, fband) = (me.voxel(m), me.band(m));
                    let old = me.blocks[m].get(&bc)?;
                    let corner = origin + (bc * B).as_vec3() * vm;
                    // A sample already at the band on the side the dab pushes toward (inside for clay) stays there,
                    // whatever it reads below: the dab moves it no farther than that side's own reading, which the
                    // clamp brings back to the band. A block with only such samples under the brush (finer work a
                    // big stroke has buried, under each of its dabs) is left as it is without reading the levels below.
                    let fill = if carve { fband } else { -fband };
                    if old.iter().enumerate().all(|(i, &v)| (v == fill && set.is_none()) || weight(corner + cell_of(i).as_vec3() * vm) <= 0.0) {
                        return None;
                    }
                    // The levels below over the block, for samples this level holds only clamped.
                    let below: Vec<Grid> = (0..m)
                        .map(|k| {
                            let d = IVec3::splat(1 << (m - k));
                            me.grid(k, (bc * B).div_euclid(d) - IVec3::ONE, (bc * B + B3).div_euclid(d) + IVec3::ONE, false)
                        })
                        .collect();
                    let mut s: Block = Box::new(**old);
                    let mut any = false;
                    for (i, v) in s.iter_mut().enumerate() {
                        let p = corner + cell_of(i).as_vec3() * vm;
                        let w = weight(p);
                        if w <= 0.0 {
                            continue;
                        }
                        // Smoothing: what a finer level holds past the dab's level is the finest detail under the
                        // brush, which a smoothing takes away first. Moved only as far as the dab's level moves, it
                        // lingered at a few tenths of a finer step after a stroke, enough for the view by distance to
                        // show some of its blocks and leave out their neighbours, a lattice of seams; it goes twice as
                        // fast (a dab's weight `w` taken twice over).
                        let w = if set.is_some() {
                            let w = set_weight(shape(p), w);
                            w * (2.0 - w)
                        } else {
                            w
                        };
                        if w <= 0.0 {
                            continue;
                        }
                        // A sample at its band holds only a bound; the levels below, on the same side and farther, say
                        // how far: down to the first that holds the distance itself (within its own plateau), since
                        // the next level down is often clamped too, and when even level 0 holds only its bound, the
                        // ground under the brush. Blended from a bound, the dab would reach it sooner here than on the
                        // levels below, and clay would show at this level's blocks (a block at a time, as a step)
                        // before it shows around them. A level's reading near its plateau mixes in the levels below
                        // it, the more the nearer it lies, from 0.7 of the plateau on: taken whole up to the plateau and the
                        // deeper reading past it, the choice jumped where the reading crossed that line (as levels
                        // clamped there interpolate, in a lattice at their spacing), by up to the next level's band
                        // less this one's; a wide trim cutting down past a finer level's band through earlier fine work
                        // showed a lattice of dents and steps there, one a block, while the stroke was drawn and after.
                        let blob = shape(p);
                        let mut was = *v;
                        if was.abs() >= 0.999 * fband {
                            let farther = |a: f32, b: f32| if (b < 0.0) == (a < 0.0) && b.abs() > a.abs() { b } else { a };
                            let at = |k: usize| below[k].trilinear((bc * B + cell_of(i)).as_vec3() / (1 << (m - k)) as f32);
                            let mut deep = at(0);
                            if l == 0 && deep.abs() >= 0.999 * plateaus[0] && w < 1.0 {
                                deep = from_ground(p, deep);
                            }
                            for k in 1..m {
                                let b = at(k);
                                let x = ((b.abs() / plateaus[k] - 0.7) / 0.3).clamp(0.0, 1.0);
                                deep = b + x * x * (3.0 - 2.0 * x) * (farther(b, deep) - b);
                            }
                            was = farther(was, deep);
                        }
                        let full = if set.is_some() {
                            blob
                        } else if carve {
                            was.max(-blob)
                        } else {
                            was.min(blob)
                        };
                        let mut next = (was + w * (full - was)).clamp(-fband, fband);
                        // Where the dab reshapes the surface here, the finer level also moves toward level `l`'s new
                        // field: the dab's own surface, at the level it works at. Each dab of a plane brush moves part of
                        // the way, so finer work it passes over fades only by that part a dab, and a stroke left some
                        // of it (a twentieth of a clay ridge it trimmed); where the dab moved the surface out of the
                        // finer level's blocks, level `l` showed instead, without it, and the rest stood as a step at
                        // every block edge, a stair along the trim. Taken twice over, it fades to nothing under the
                        // stroke, and the blocks holding nothing more than level `l` are dropped as it ends. (Not
                        // where level `l` holds only its bound: finer work standing that far off it.)
                        if set.is_none() && full != was {
                            let g = (bc * B + cell_of(i)).as_vec3() / (1 << (m - l)) as f32;
                            let coarse = below[l].trilinear(g) + moved_at(g);
                            if coarse.abs() < 0.999 * plateaus[l] {
                                next = (next + w * (coarse - next)).clamp(-fband, fband);
                            }
                        }
                        if next != *v {
                            *v = next;
                            any = true;
                        }
                    }
                    any.then_some((m, bc, s))
                })
                .collect()
        };
        // Outside a stroke, the dab is a stroke of its own: detail is carried from the field as it was before it.
        let own = self.stroke.is_none();
        if own {
            self.begin_stroke();
        }
        if self.carry_level != l {
            self.carry.clear();
            self.carry_push.clear();
            self.carry_level = l;
        }
        if let Some(push) = push {
            self.carry_push.push(push);
        }
        let mut flipped: HashMap<IVec3, SignMask> = HashMap::default();
        let mut changed = Vec::with_capacity(built.len());
        for (bc, s, d, flips, _) in built {
            if !self.blocks[l].contains_key(&bc) {
                stats.created += 1;
            }
            self.put_stored(l, bc, s);
            match self.carry.get_mut(&bc) {
                Some(total) => {
                    for (t, x) in Arc::make_mut(total).iter_mut().zip(d.iter()) {
                        *t += x;
                    }
                }
                None => {
                    self.carry.insert(bc, d);
                }
            }
            self.recent.insert(bc);
            changed.push(bc);
            if flips.iter().any(|&m| m != 0) {
                flipped.insert(bc, flips);
            }
        }
        for (m, bc, s) in finer {
            self.put(m, bc, s);
            stats.carried += 1;
        }
        stats.changed = changed.len();
        // Moves measured near what changed are measured again: a measurement marches through the field around it.
        if !self.moved.is_empty() {
            let near = Self::grown(&changed, 2);
            self.moved.retain(|p, _| !near.contains(&p.div_euclid(B3)));
        }
        self.clear_checkerboards(l, flipped);
        if self.keep_detail {
            stats.carried = self.carry_detail(l, &changed);
        }
        if own {
            stats.carried += self.carry_pending(None);
            self.stroke = None;
            self.carry.clear();
        }
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        stats
    }

    /// After a dab at level `l` changed `changed` (blocks of that level), every finer level's own detail (its values
    /// less the level below, both within its band) moves with the surface: along the stroke's push, as far as the
    /// stroke has moved it so far. The detail is always taken from the surface as it was when the stroke began and moved by the
    /// stroke's whole change, so however many dabs a stroke has, its detail is resampled once, and the work can be done
    /// late: the finer blocks under the dab are queued, and as many are rebuilt now as [`Self::carry_budget`] allows
    /// (all of them when it is `None`); the rest follow in [`Self::carry_pending`], and all by the stroke's end.
    fn carry_detail(&mut self, l: usize, changed: &[IVec3]) -> usize {
        if changed.is_empty() || l + 1 >= self.levels() {
            return 0;
        }
        // The dab's footprint at level `l`, with a ring of blocks for detail moved across block edges, and each finer
        // level's blocks under it, now or when the stroke began. Found from the finer levels' side: they hold far fewer
        // blocks than a big dab touches at a coarse level.
        let ring = Self::grown(changed, 1);
        for m in l + 1..self.levels() {
            let s = m - l;
            let under = |c: &IVec3| ring.contains(&IVec3::new(c.x >> s, c.y >> s, c.z >> s));
            let mut region: Vec<IVec3> = self.blocks[m].keys().copied().filter(under).collect();
            if let Some(stroke) = &self.stroke {
                region.extend(stroke.iter().filter(|(k, w)| k.0 as usize == m && matches!(w, Was::Stored(_))).map(|(k, _)| k.1).filter(under));
            }
            if region.is_empty() {
                break;
            }
            self.pending[m].extend(region);
        }
        // Blocks this stroke carried detail into, near what changed: detail moved far can sit outside the dab's own
        // reach, and the move they read was measured again. Farther ones would rebuild as they are.
        let near = if self.carried.iter().any(|c| !c.is_empty()) { Self::grown(changed, 2) } else { HashSet::default() };
        for m in l + 1..self.levels() {
            let s = m - l;
            let again: Vec<IVec3> = self.carried[m].iter().copied().filter(|c| near.contains(&IVec3::new(c.x >> s, c.y >> s, c.z >> s))).collect();
            self.pending[m].extend(again);
        }
        self.carry_pending(self.carry_budget)
    }

    /// Blocks within `r` of `changed` (blocks of one level), along each axis. Two reach as far as a level 0 band, and a
    /// little more.
    fn grown(changed: &[IVec3], r: i32) -> HashSet<IVec3> {
        // One axis at a time: 2r + 1 inserts a block per axis rather than (2r + 1) cubed.
        let mut set: HashSet<IVec3> = changed.iter().copied().collect();
        for axis in [IVec3::X, IVec3::Y, IVec3::Z] {
            let grown: HashSet<IVec3> = set.iter().flat_map(|&b| (-r..=r).map(move |o| b + axis * o)).collect();
            set = grown;
        }
        set
    }

    /// A copy of the volume to carry detail on, made in a moment (its maps share their shards with this one's).
    pub(crate) fn snapshot(&self) -> LevelVolume {
        LevelVolume {
            voxel0: self.voxel0,
            origin: self.origin,
            blocks: self.blocks.clone(),
            inside0: self.inside0.clone(),
            stroke: self.stroke.clone(),
            history: Vec::new(),
            redo: Vec::new(),
            dirty: HashSet::default(),
            carry: self.carry.clone(),
            carry_push: self.carry_push.clone(),
            recent: HashSet::default(),
            carry_level: self.carry_level,
            pending: vec![HashSet::default(); MAX_LEVELS],
            moved: HashMap::default(),
            carried: vec![HashSet::default(); MAX_LEVELS],
            carry_budget: None,
            keep_detail: self.keep_detail,
            uncached_reads: self.uncached_reads,
        }
    }

    /// Take up to `most` queued blocks off the carrying queue, to carry on another thread ([`CarryJob::run`]) while
    /// the stroke goes on; [`Self::carry_apply`] then brings the result in. A level's blocks are taken only once the
    /// coarser ones queued have all been, since carrying reads the level below as carried. A block a later dab
    /// changes the ground of is queued again, so a result that comes in late is replaced.
    pub fn carry_take(&mut self, most: usize) -> Option<CarryJob> {
        self.stroke.as_ref()?;
        let l = self.carry_level;
        let mut plan = Vec::new();
        let mut left = most;
        for m in l + 1..self.levels() {
            if self.pending[m].is_empty() {
                continue;
            }
            let mut batch: Vec<IVec3> = self.pending[m].iter().copied().take(left).collect();
            for c in &batch {
                self.pending[m].remove(c);
            }
            batch.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            left -= batch.len();
            plan.push((m, batch));
            if !self.pending[m].is_empty() || left == 0 {
                break;
            }
        }
        if plan.is_empty() {
            return None;
        }
        self.recent.clear();
        Some(CarryJob { vol: self.snapshot(), plan, level: l })
    }

    /// Bring in what a [`CarryJob`] carried: its blocks, those it dropped, and the blocks it carried detail into.
    pub fn carry_apply(&mut self, done: CarryDone) {
        if self.stroke.is_none() || done.level != self.carry_level {
            return;
        }
        // Blocks made, coarser levels first (a block's parents before it); then blocks dropped, finer levels first (a
        // block's children before it, which may be dropped too).
        let mut changes = done.changes;
        changes.sort_unstable_by_key(|c| (c.2.is_none(), if c.2.is_none() { u8::MAX - c.0 } else { c.0 }, c.1.z, c.1.y, c.1.x));
        for (m, bc, s) in changes {
            let m = m as usize;
            match s {
                Some(s) => self.put_stored(m, bc, s),
                None => {
                    let has_children = m + 1 < self.levels() && (0..8).any(|q| self.blocks[m + 1].contains_key(&(bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1))));
                    if !has_children && self.blocks[m].contains_key(&bc) {
                        self.keep((m as u8, bc));
                        self.blocks[m].remove(&bc);
                        self.dirty.insert((m as u8, bc));
                    }
                }
            }
        }
        // Blocks it carried detail into under dabs made while it ran: carried again, as those dabs would have had
        // the blocks existed then.
        let near = if self.recent.is_empty() { HashSet::default() } else { Self::grown(&self.recent.iter().copied().collect::<Vec<_>>(), 2) };
        let l = self.carry_level;
        for (m, bc) in done.carried {
            self.carried[m].insert(bc);
            let s = m - l;
            if near.contains(&IVec3::new(bc.x >> s, bc.y >> s, bc.z >> s)) {
                self.pending[m].insert(bc);
            }
        }
    }

    /// Whether carrying detail has blocks still queued.
    pub fn carry_busy(&self) -> bool {
        self.pending.iter().any(|p| !p.is_empty())
    }

    /// Rebuild queued finer blocks (see [`Self::carry_detail`]), coarsest level first, for at most `budget_ms` (all of
    /// them with `None`). Returns the blocks rebuilt.
    pub fn carry_pending(&mut self, budget_ms: Option<f64>) -> usize {
        let t = Instant::now();
        let l = self.carry_level;
        let mut carried = 0;
        for m in l + 1..self.levels() {
            while !self.pending[m].is_empty() {
                if budget_ms.is_some_and(|b| b <= 0.0 || t.elapsed().as_secs_f64() * 1e3 > b) {
                    return carried;
                }
                // A batch, so a budget is checked often enough.
                let take = if budget_ms.is_some() { 48 } else { usize::MAX };
                let mut batch: Vec<IVec3> = self.pending[m].iter().copied().take(take).collect();
                for c in &batch {
                    self.pending[m].remove(c);
                }
                batch.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                carried += self.carry_batch(l, m, &batch);
            }
        }
        carried
    }

    /// Carry the detail of level `m`'s blocks `region` (see [`Self::carry_detail`]).
    fn carry_batch(&mut self, l: usize, m: usize, region: &[IVec3]) -> usize {
        let mut carried = 0;
        let k = 1i32 << (m - l);
        let vm = self.voxel(m);
        // Detail comes only from blocks that held it when the stroke began: blocks an earlier dab of the stroke made
        // hold detail moved into them, and moving it on from there would spawn blocks without end.
        let sources: Vec<IVec3> = region.iter().copied().filter(|c| self.exists_at(m, *c, true)).collect();
        // Where the detail goes: the blocks themselves, and those the sources' boxes cover once moved.
        let dest: Vec<IVec3> = {
            let me = &*self;
            let mut dest: Vec<IVec3> = region.to_vec();
            dest.extend(
                sources
                    .par_iter()
                    .flat_map_iter(|&bc| {
                        let mut out = Vec::new();
                        // Where this block's detail lands: each of its corners and its centre, from the surface the
                        // stroke began on, along the direction it is carried in (the one the blocks it lands in read
                        // it back along, so the two agree), on to the new surface.
                        let vl = me.voxel(l);
                        for c in 0..9 {
                            let corner = if c == 8 { Vec3::splat(0.5) } else { Vec3::new((c & 1) as f32, ((c >> 1) & 1) as f32, ((c >> 2) & 1) as f32) };
                            let at = (bc * B).as_vec3() + corner * (B - 1) as f32;
                            let gl = at / k as f32;
                            let i = gl.round().as_ivec3();
                            if !me.carry.get(&i.div_euclid(B3)).is_some_and(|d| d[local(i)] != 0.0) {
                                continue;
                            }
                            let dn = |o: Vec3| me.value_at(l, gl + o, true) - me.value_at(l, gl - o, true);
                            let n_old = Vec3::new(dn(Vec3::X), dn(Vec3::Y), dn(Vec3::Z)).normalize_or_zero();
                            if n_old == Vec3::ZERO {
                                continue;
                            }
                            let u = me.carry_dir(gl, n_old);
                            let Some(to_old) = me.cross_along(l, gl, u, true) else { continue };
                            let on_old = gl + u * (to_old / vl);
                            let Some(rise) = me.cross_along(l, on_old, u, false) else { continue };
                            out.push((at + u * (rise / vm)).floor().as_ivec3().div_euclid(B3));
                        }
                        out
                    })
                    .collect::<Vec<IVec3>>(),
            );
            dest.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            dest.dedup();
            dest
        };
        // The surface's move at every level `l` point the blocks read, measured once: the cells of the `MOVE_EVERY`
        // lattice over the footprints `carry_block` covers (every one, since a point missed here is measured again by
        // each block that reads it).
        let mut cells: Vec<IVec3> = dest
            .iter()
            .flat_map(|&bc| {
                let lo = bc * B;
                let (flo, fhi) = (lo.div_euclid(IVec3::splat(k)) - IVec3::ONE, (lo + B3).div_euclid(IVec3::splat(k)) + IVec3::splat(2));
                let (llo, lhi) = (flo.div_euclid(IVec3::splat(MOVE_EVERY)), fhi.div_euclid(IVec3::splat(MOVE_EVERY)));
                (llo.z..=lhi.z).flat_map(move |z| (llo.y..=lhi.y).flat_map(move |y| (llo.x..=lhi.x).map(move |x| IVec3::new(x, y, z))))
            })
            .collect();
        cells.sort_unstable_by_key(|v| (v.z, v.y, v.x));
        cells.dedup();
        let corner = |q: usize| IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32);
        self.measure_all(cells.iter().flat_map(|&c| (0..8).map(move |q| (c + corner(q)) * MOVE_EVERY)));
        // Where that lattice is too coarse for the move, every point of the cell: the surface the stroke began on
        // may hold form at a finer scale than the lattice (clay built at a finer level and moved down, say), which the
        // stroke buries unevenly. Found by measuring each cell's centre against what its corners give for it; a
        // groove on such form read an interpolated move off by a few of its steps and was lost.
        let rough: Vec<IVec3> = {
            let centres: Vec<IVec3> = cells.iter().map(|&c| c * MOVE_EVERY + IVec3::splat(MOVE_EVERY / 2)).collect();
            self.measure_all(centres.iter().copied());
            cells
                .iter()
                .filter(|&&c| {
                    let at = |p: IVec3| self.moved.get(&p).copied().unwrap_or(0.0);
                    let from_corners = (0..8).map(|q| at((c + corner(q)) * MOVE_EVERY)).sum::<f32>() / 8.0;
                    (at(c * MOVE_EVERY + IVec3::splat(MOVE_EVERY / 2)) - from_corners).abs() > 0.5 * vm
                })
                .copied()
                .collect()
        };
        self.measure_all(rough.iter().flat_map(|&c| {
            (0..MOVE_EVERY).flat_map(move |z| (0..MOVE_EVERY).flat_map(move |y| (0..MOVE_EVERY).map(move |x| c * MOVE_EVERY + IVec3::new(x, y, z))))
        }));
        let me = &*self;
        let built: Vec<(IVec3, Block)> = dest.par_iter().filter_map(|&bc| me.carry_block(l, m, bc).map(|s| (bc, s))).collect();
        for (bc, s) in built {
            self.put(m, bc, s);
            self.carried[m].insert(bc);
            carried += 1;
        }
        // Blocks the detail has moved out of hold only the level below: drop them (kept for undo), so moving detail
        // leaves no trail of empty blocks behind. Those with finer blocks inside stay, so the levels stay nested.
        let tol = 0.02 * vm;
        for &bc in &dest {
            if !self.blocks[m].contains_key(&bc) {
                continue;
            }
            let has_children = m + 1 < self.levels() && (0..8).any(|q| self.blocks[m + 1].contains_key(&(bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1))));
            if has_children {
                continue;
            }
            self.keep((m as u8, bc));
            let Some(s) = self.blocks[m].remove(&bc) else { continue };
            let fallback = self.block_values(m, bc, false);
            if s.iter().zip(fallback.iter()).any(|(a, b)| (a - b).abs() > tol) {
                self.blocks[m].insert(bc, s);
            } else {
                self.dirty.insert((m as u8, bc));
                // Its parent may have held it alone; it is checked when its own detail is next carried.
            }
        }
        carried
    }

    /// Measure the stroke's move ([`Self::measure_moved`]) at the level `carry_level` points of `points` not measured
    /// yet, in parallel, into `moved`.
    fn measure_all(&mut self, points: impl Iterator<Item = IVec3>) {
        let mut points: Vec<IVec3> = points.filter(|p| !self.moved.contains_key(p) && self.carry.contains_key(&p.div_euclid(B3))).collect();
        if points.is_empty() {
            return;
        }
        points.sort_unstable_by_key(|v| (v.z, v.y, v.x));
        points.dedup();
        let me = &*self;
        let measured: Vec<(IVec3, f32)> = points.par_iter().map(|&p| (p, me.measure_moved(p))).collect();
        self.moved.extend(measured);
    }

    /// The direction the stroke pushed the surface in at level `carry_level`'s lattice coordinates `g`: its dabs'
    /// directions blended by each dab's falloff at the point, so it turns as smoothly as the brush itself. Blended
    /// between blocks instead, it turned in kinks a block apart, and a groove carried along it wandered sideways by a
    /// step or two within a block and smeared wide and shallow where the kinks met. Weighted by the falloff, not by
    /// how much each dab changed: the first dab over a place changes it most, which tilted the push toward where the
    /// stroke began. `None` where no dab reaches.
    fn push_at(&self, g: Vec3) -> Option<Vec3> {
        if self.carry_push.is_empty() {
            return None;
        }
        let p = self.origin + g * self.voxel(self.carry_level);
        let mut sum = Vec3::ZERO;
        for dab in &self.carry_push {
            let q = p - dab.on;
            let t = (q - dab.n * q.dot(dab.n)).length();
            let x = ((t / dab.r - dab.hard) / (1.0 - dab.hard)).clamp(0.0, 1.0);
            sum += dab.n * (1.0 - x * x * (3.0 - 2.0 * x));
        }
        sum.try_normalize()
    }

    /// The direction detail is carried along at `g` (level `carry_level` lattice coordinates) where the surface's
    /// normal is `n`: the stroke's push, unless it runs nearly along the surface (the wall of a hard brush), where
    /// marching along it could miss the surface; there, and where there is none, the normal. Along the new surface's
    /// normal instead, detail slid sideways wherever the stroke left a slope (bumps under six big strokes ended up
    /// 0.027 off, against 0.004), a groove on the rim of a second pass was lost, and a groove over clay built at a
    /// finer level stepped sideways by five of its steps where the normal turned; the stroke-start normal did the
    /// last two as well.
    fn carry_dir(&self, g: Vec3, n: Vec3) -> Vec3 {
        match self.push_at(g) {
            Some(u) if u.dot(n).abs() >= 0.2 => u,
            _ => n,
        }
    }

    /// How far along `n` (world units, signed) from level `l`'s lattice point `from` the field (as it is, or as the
    /// stroke began with `base`) next crosses zero, toward the surface. By stepping until the sign changes, then
    /// halving: inside fresh clay the field is not a distance yet (it is made one when the stroke ends), so its value
    /// cannot be trusted as a step.
    fn cross_along(&self, l: usize, from: Vec3, n: Vec3, base: bool) -> Option<f32> {
        let (band, vl) = (self.band(l), self.voxel(l));
        let f = |t: f32| self.value_at(l, from + n * (t / vl), base);
        let v0 = f(0.0);
        if v0 == 0.0 {
            return Some(0.0);
        }
        // Toward the surface: inward from outside, outward from inside.
        let dir = if v0 > 0.0 { -1.0 } else { 1.0 };
        let mut t = 0.0f32;
        for _ in 0..256 {
            let next = t + f(t).abs().clamp(0.5 * vl, band) * dir;
            if (f(next) > 0.0) != (v0 > 0.0) {
                let (mut lo, mut hi) = (t, next);
                for _ in 0..12 {
                    let m = 0.5 * (lo + hi);
                    if (f(m) > 0.0) == (v0 > 0.0) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
            t = next;
            if t.abs() > 64.0 * band {
                break;
            }
        }
        None
    }

    /// How far the stroke moved the surface at level `carry_level`'s lattice point `p`, as a change of the field there
    /// (negative where it moved out; zero where it changed nothing). Near the new surface the move is measured,
    /// from there back to the surface the stroke began on: the stroke's change of the samples says the same only while
    /// both old and new values lie within the band, and a stroke can move the surface farther than that (the samples
    /// are clamped, so their change stops growing). Farther out no detail is carried, and the change itself will do.
    fn moved_at(&self, p: IVec3) -> f32 {
        if !self.carry.contains_key(&p.div_euclid(B3)) {
            return 0.0;
        }
        // Measured on every `MOVE_EVERY`th lattice point and interpolated (the move varies smoothly over a brush, and
        // measuring is a march), except where the batch found that lattice too coarse and measured every point.
        if let Some(&d) = self.moved.get(&p) {
            return d;
        }
        let base = p.div_euclid(IVec3::splat(MOVE_EVERY)) * MOVE_EVERY;
        let f = (p - base).as_vec3() / MOVE_EVERY as f32;
        trilinear(f, std::array::from_fn(|q| {
            let c = base + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32) * MOVE_EVERY;
            self.moved.get(&c).copied().unwrap_or_else(|| self.measure_moved(c))
        }))
    }

    fn measure_moved(&self, p: IVec3) -> f32 {
        let Some(d) = self.carry.get(&p.div_euclid(B3)).map(|d| d[local(p)]) else { return 0.0 };
        if d == 0.0 {
            return 0.0;
        }
        let l = self.carry_level;
        let (band, vl) = (self.band(l), self.voxel(l));
        let g = p.as_vec3();
        let now = self.sample(l, p, false);
        if now.abs() >= 0.9 * band {
            // Far from the new surface: no detail is carried here.
            return d;
        }
        let dn = |o: Vec3| self.value_at(l, g + o, false) - self.value_at(l, g - o, false);
        let n = Vec3::new(dn(Vec3::X), dn(Vec3::Y), dn(Vec3::Z)).normalize_or_zero();
        if n == Vec3::ZERO {
            return d;
        }
        let n = self.carry_dir(g, n);
        // Along the carrying direction to the new surface, then on to where the field the stroke began with is zero.
        let cross = |from: Vec3, base: bool| self.cross_along(l, from, n, base);
        let Some(to_new) = cross(g, false) else { return d };
        let on_new = g + n * (to_new / vl);
        let Some(to_old) = cross(on_new, true) else { return d };
        let moved = -to_old;
        // The change of the field at `p` that moving the surface out by `moved` makes.
        -moved
    }

    /// Block `bc` of level `m` (finer than the stroke's level `l`) rebuilt: the new level below plus this level's detail
    /// from when the stroke began, moved with the surface. `None` if that is what the block holds already.
    fn carry_block(&self, l: usize, m: usize, bc: IVec3) -> Option<Block> {
        let k = 1i32 << (m - l);
        let (vm, fband) = (self.voxel(m), self.band(m));
        let lo = bc * B;
        // The stroke's change and the level `l` field over this block's footprint.
        // With a margin of three steps of the stroke's level, which the march below may read.
        let (llo, lhi) = (lo.div_euclid(IVec3::splat(k)) - IVec3::splat(3), (lo + B3).div_euclid(IVec3::splat(k)) + IVec3::splat(4));
        let ln = lhi - llo + IVec3::ONE;
        let mut dgrid = Grid { lo: llo, n: ln, v: vec![0.0; (ln.x * ln.y * ln.z) as usize] };
        let mut most = 0.0f32;
        for z in llo.z..=lhi.z {
            for y in llo.y..=lhi.y {
                for x in llo.x..=lhi.x {
                    let p = IVec3::new(x, y, z);
                    if self.carry.contains_key(&p.div_euclid(B3)) {
                        let d = self.moved_at(p);
                        let at = dgrid.idx(p);
                        dgrid.v[at] = d;
                        most = most.max(d.abs());
                    }
                }
            }
        }
        let current = self.blocks[m].get(&bc);
        // A block that is not there yet is only worth making if it holds detail: something other than the level below.
        let fallback = current.is_none().then(|| self.block_values(m, bc, false));
        let tol = 0.02 * vm;
        let differs = |s: &Block| match (current, &fallback) {
            (Some(c), _) => **c != **s,
            (None, Some(f)) => s.iter().zip(f.iter()).any(|(a, b)| (a - b).abs() > tol),
            (None, None) => true,
        };
        if most == 0.0 {
            // Nothing moved here: the detail is as it was.
            let base = self.block_values(m, bc, true);
            return differs(&base).then_some(base);
        }
        let lgrid = self.grid(l, llo, lhi, false);
        let kf = k as f32;
        // Where each sample's detail comes from: `-d` farther in along the carrying direction, where the surface moved
        // out by `-d`.
        let below = self.grid(m - 1, lo.div_euclid(IVec3::splat(2)) - IVec3::splat(2), (lo + B3).div_euclid(IVec3::splat(2)) + IVec3::splat(2), false);
        // Samples more than two bands from the new surface come out clamped whatever detail they carry.
        let mut far = vec![false; N];
        let mut from: Vec<Option<Vec3>> = vec![None; N];
        // The carrying direction at each sample.
        let mut dir = vec![Vec3::ZERO; N];
        let (mut qlo, mut qhi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        for (i, q) in from.iter_mut().enumerate() {
            let p = lo + cell_of(i);
            let g = p.as_vec3() / kf;
            let d = dgrid.trilinear(g);
            if d == 0.0 {
                continue;
            }
            if below.trilinear(p.as_vec3() * 0.5).abs() > 2.0 * fband {
                far[i] = true;
                continue;
            }
            let dn = |o: Vec3| lgrid.trilinear(g + o) - lgrid.trilinear(g - o);
            let grad = Vec3::new(dn(Vec3::X * 0.5), dn(Vec3::Y * 0.5), dn(Vec3::Z * 0.5));
            let n = grad.normalize_or_zero();
            // Along the direction the move was measured in (as `measure_moved` takes it).
            let n = if n == Vec3::ZERO { n } else { self.carry_dir(g, n) };
            dir[i] = n;
            // The surface moved out by `-d` here: the detail now here was `-d` farther in.
            let at = p.as_vec3() + n * (d / vm);
            qlo = qlo.min(at.floor().as_ivec3());
            qhi = qhi.max(at.floor().as_ivec3() + IVec3::ONE);
            *q = Some(at);
        }
        // Detail as it was, over the box it comes from: the block moved as a whole, so about a block across however far
        // the stroke moved it. Kept to a bounded box; the odd sample whose source lies beyond it (where the stroke bent
        // the surface hard) takes no detail, rather than reading every level point by point.
        if qlo.x > qhi.x {
            (qlo, qhi) = (lo, lo + IVec3::ONE);
        }
        // With the point below and the two above every source, which the cubic interpolation reads: with the box
        // cut at the sources themselves those read as no detail, and every block's edge samples came out short,
        // a lattice of block-sized dips over everything a big stroke carried.
        let (blo, bhi) = (qlo - IVec3::ONE, qhi.min(qlo + IVec3::splat(4 * B)) + IVec3::splat(2));
        let unmoved = self.block_values(m, bc, true);
        let base_m = self.grid(m, blo, bhi, true);
        let base_below = self.grid(m - 1, blo.div_euclid(IVec3::splat(2)) - IVec3::ONE, (bhi + IVec3::ONE).div_euclid(IVec3::splat(2)) + IVec3::ONE, true);
        // The detail over the box: the value less the level below's as this level sees it, clamped to this level's
        // band, the way a level is built from the one below where it has no block of its own. It is put back on the
        // new level below clamped the same way; put on it unclamped, a bump taller than about two of this level's
        // steps read shorter by a tenth with every stroke that carried it (the sample just above its top is within
        // this level's band but beyond it on the level below). Taken unclamped instead, the detail ran to several
        // bands wherever this level's surface sits far from the level below's (clay built at this level and moved
        // down only in part), and a big stroke over such clay exaggerated it and left the surface rough.
        let bn = bhi - blo + IVec3::ONE;
        let detail_box: Vec<f32> = (0..bn.x * bn.y * bn.z)
            .map(|idx| {
                let j = blo + IVec3::new(idx % bn.x, (idx / bn.x) % bn.y, idx / (bn.x * bn.y));
                base_m.at(j) - base_below.trilinear(j.as_vec3() * 0.5).clamp(-fband, fband)
            })
            .collect();
        // Outside the box (the odd sample whose source lies beyond it, where the stroke bent the surface hard): none.
        let detail_at = |j: IVec3| -> f32 {
            if j.cmpge(blo).all() && j.cmple(bhi).all() {
                let d = j - blo;
                detail_box[((d.z * bn.y + d.y) * bn.x + d.x) as usize]
            } else {
                0.0
            }
        };
        // Catmull-Rom weights for the four points around a fraction `t`.
        let cubic = |t: f32| {
            let (t2, t3) = (t * t, t * t * t);
            [-0.5 * t + t2 - 0.5 * t3, 1.0 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2.0 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3]
        };
        let mut s = Box::new([0.0f32; N]);
        for (i, v) in s.iter_mut().enumerate() {
            let p = lo + cell_of(i);
            if far[i] {
                *v = below.trilinear(p.as_vec3() * 0.5).clamp(-fband, fband);
                continue;
            }
            let Some(q) = from[i] else {
                *v = unmoved[i];
                continue;
            };
            let j = q.floor().as_ivec3();
            let f = q - j.as_vec3();
            // The detail read with cubic interpolation over the 4x4x4 points around the source: the finest level's
            // bumps and pits are a few of its steps across, and trilinear interpolation at a fraction of a step read
            // them lower by a tenth every stroke that carried them (six big strokes left 71% of a bump, against 92%).
            // Every level is read the same way: the finer levels' detail sits within their bands of the level below
            // as carried, and a level carried more smoothly than the one above it put the finest detail at the edge
            // of its band, where the clamp cut into it. It does resolve the clay brush's own texture at the level
            // right above the stroke's more sharply (the surface's second difference there is 1.4 times what
            // trilinear interpolation leaves).
            let (wx, wy, wz) = (cubic(f.x), cubic(f.y), cubic(f.z));
            let mut detail = 0.0;
            for (dz, wz) in wz.iter().enumerate() {
                for (dy, wy) in wy.iter().enumerate() {
                    for (dx, wx) in wx.iter().enumerate() {
                        detail += detail_at(j + IVec3::new(dx as i32 - 1, dy as i32 - 1, dz as i32 - 1)) * wx * wy * wz;
                    }
                }
            }
            // The level below as a distance. The stroke's level is no distance field while the stroke lasts: a
            // plane brush leaves a blend of the stroke's start with its plane, and once a big stroke has moved the
            // surface nearly a band, the start is at its band above the new surface and the blend's slope there is
            // the dab's weight, a third or less on the brush's falloff. Detail placed against it by value then landed
            // at its height over the weight: form held at finer levels came out half again as tall near the
            // brush's line, and a plateau that swallowed the finest grooves farther out. So the level right above
            // the stroke's reads the distance to the stroke's level's surface, marched along the carrying
            // direction; the levels above that read their carried level below, which is a distance again by this.
            let mut under = below.trilinear(p.as_vec3() * 0.5);
            if m == l + 1 && dir[i] != Vec3::ZERO && under.abs() < fband {
                // Its signed distance to the stroke's level's surface along the carrying direction, marched on the
                // footprint's grid: a quarter step at a time to the far side, then halved in. Beyond two bands with
                // no crossing, the band.
                let g = p.as_vec3() / kf;
                let vl = self.voxel(l);
                let sign = under.signum();
                let f = |t: f32| lgrid.trilinear(g - dir[i] * (sign * t / vl));
                let limit = 2.0 * fband;
                let mut t = 0.0;
                let mut found = None;
                while t < limit {
                    let next = (t + 0.25 * vl).min(limit);
                    if f(next) * sign <= 0.0 {
                        let (mut a, mut b) = (t, next);
                        for _ in 0..8 {
                            let mid = 0.5 * (a + b);
                            if f(mid) * sign > 0.0 {
                                a = mid;
                            } else {
                                b = mid;
                            }
                        }
                        found = Some(0.5 * (a + b));
                        break;
                    }
                    t = next;
                }
                under = match found {
                    Some(dist) => sign * dist,
                    None => sign * fband,
                };
            }
            *v = (under.clamp(-fband, fband) + detail).clamp(-fband, fband);
        }
        differs(&s).then_some(s)
    }

    /// [`Self::clear_checkerboards`] around blocks `blocks` of level `l`, whose samples were all written anew (a layer
    /// stack composed there).
    pub(crate) fn clear_checkerboards_at(&mut self, l: usize, blocks: &[IVec3]) {
        if blocks.is_empty() {
            return;
        }
        // With the blocks past them on each axis: a face on a block's far side is looked at from the block past it,
        // and the cleanup looks at the blocks given and the ones below them.
        let flipped: HashMap<IVec3, SignMask> = blocks.iter().flat_map(|bc| (0..8).map(move |k| *bc + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).map(|bc| (bc, [u64::MAX; 8])).collect();
        self.clear_checkerboards_in(l, flipped, true);
    }

    /// Remove checkerboard faces (a cell face whose corners alternate in sign, which pieces cannot represent cleanly)
    /// from level `l`'s stored field, by filling one outside corner of each, repeatedly, as `volume.rs` does. Only the
    /// blocks with a sample whose sign changed (`flipped`) and the ones below them can have gained one.
    fn clear_checkerboards(&mut self, l: usize, flipped: HashMap<IVec3, SignMask>) {
        self.clear_checkerboards_in(l, flipped, false);
    }

    /// [`Self::clear_checkerboards`]; with `unstored`, the blocks looked at include those not stored (a face between a
    /// stored block and the empty one before it along an axis is looked at from the empty one; a composed region's
    /// edge has such faces, a stroke's region hardly ever).
    fn clear_checkerboards_in(&mut self, l: usize, mut flipped: HashMap<IVec3, SignMask>, unstored: bool) {
        let fill = -1e-3 * self.voxel(l);
        for _ in 0..32 {
            if flipped.is_empty() {
                break;
            }
            let mut look: Vec<IVec3> = flipped.keys().flat_map(|&b| (0..8).map(move |k| b - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).filter(|b| unstored || self.blocks[l].contains_key(b)).collect();
            look.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            look.dedup();
            let me = &*self;
            let mut flips: Vec<(IVec3, bool)> = look
                .par_iter()
                .flat_map_iter(|&bc| {
                    let base = bc * B;
                    let g = me.gather9(l, bc);
                    let mut rows = [[0u16; 9]; 9];
                    for (z, plane) in rows.iter_mut().enumerate() {
                        for (y, row) in plane.iter_mut().enumerate() {
                            for x in 0..9 {
                                if g[z * 81 + y * 9 + x] < 0.0 {
                                    *row |= 1 << x;
                                }
                            }
                        }
                    }
                    let idx = |p: IVec3| (p.z * 81 + p.y * 9 + p.x) as usize;
                    let mut out = Vec::new();
                    let mut take = |p: IVec3, u: usize, v: usize| {
                        let mut c = [p; 4];
                        c[1][u] += 1;
                        c[2][u] += 1;
                        c[2][v] += 1;
                        c[3][v] += 1;
                        let stored = |q: &&IVec3| me.blocks[l].contains_key(&(base + **q).div_euclid(B3));
                        if let Some(q) = c.iter().filter(stored).find(|q| g[idx(**q)] >= 0.0) {
                            out.push((base + *q, true));
                        } else if let Some(q) = c.iter().filter(stored).find(|q| g[idx(**q)] < 0.0) {
                            // Every outside corner lies in a block that is not stored (the edge of what a layer
                            // composes): an inside one is carved instead.
                            out.push((base + *q, false));
                        }
                    };
                    for z in 0..8 {
                        for y in 0..8 {
                            let a = rows[z][y];
                            let (ny, nz, nyz) = (rows[z][y + 1], rows[z + 1][y], rows[z + 1][y + 1]);
                            let xy = (a ^ (a >> 1)) & !((a >> 1) ^ ny) & !(a ^ (ny >> 1)) & 0xFF;
                            let zx = (a ^ nz) & !(a ^ (nz >> 1)) & !(nz ^ (a >> 1)) & 0xFF;
                            let yz = (a ^ ny) & !(a ^ nyz) & !(ny ^ nz) & 0xFF;
                            for (mask, u, v) in [(yz, 1, 2), (zx, 2, 0), (xy, 0, 1)] {
                                let mut m = mask;
                                while m != 0 {
                                    let x = m.trailing_zeros() as i32;
                                    m &= m - 1;
                                    take(IVec3::new(x, y as i32, z as i32), u, v);
                                }
                            }
                        }
                    }
                    out
                })
                .collect();
            flips.sort_unstable_by_key(|p| (p.0.z, p.0.y, p.0.x, !p.1));
            flips.dedup_by_key(|p| p.0);
            flipped.clear();
            for (p, inward) in flips {
                let bc = p.div_euclid(B3);
                self.keep((l as u8, bc));
                let i = local(p);
                if let Some(s) = self.blocks[l].get_mut(&bc) {
                    Arc::make_mut(s)[i] = if inward { fill } else { -fill };
                }
                self.dirty.insert((l as u8, bc));
                flipped.entry(bc).or_insert([0; 8])[i / 64] |= 1 << (i % 64);
            }
        }
    }

    // ------------------------------------------------------------------ plane brushes

    /// The surface of level `l` as it is now under a plane brush at `on` with normal `n` and radius `r` (see
    /// [`Ground`]): each column marched from `top` along `-n` (or up, where `top` is inside) to the field's first
    /// crossing, no farther down than `bottom` (heights along `n` from `on`).
    fn ground(&self, l: usize, on: Vec3, n: Vec3, r: f32, top: f32, bottom: f32, base: bool) -> Ground {
        let (vx, band) = (self.voxel(l), self.band(l));
        let step = 2.0 * vx;
        let half = (r / step).ceil() as i32 + 1;
        let size = 2 * half + 1;
        let (t1, t2) = n.any_orthonormal_pair();
        // Where no block is stored at `l`, a read interpolates through every coarser level down to a stored one (eight
        // corners a level: hundreds of lookups a read), and a column takes some twenty reads. The blocks the columns
        // cross are made once instead, each the first time one reads it, exactly as those reads make each sample.
        let near = {
            let w = half as f32 * step + vx;
            let (ulo, uhi) = (bottom - vx, top + 2.0 * band + vx);
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for k in 0..8 {
                let p = on + t1 * if k & 1 == 0 { -w } else { w } + t2 * if k & 2 == 0 { -w } else { w } + n * if k & 4 == 0 { ulo } else { uhi };
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let bs = vx * B as f32;
            let cached = l > 0 && !self.uncached_reads;
            Near::new(self, l, ((lo - self.origin) / bs).floor().as_ivec3(), ((hi - self.origin) / bs).floor().as_ivec3(), base, cached)
        };
        let f = |a: i32, b: i32, u: f32| near.value_at((on + t1 * ((a - half) as f32 * step) + t2 * ((b - half) as f32 * step) + n * u - self.origin) / vx);
        let h: Vec<f32> = (0..size * size)
            .into_par_iter()
            .map(|k| {
                let (a, b) = (k % size, k / size);
                let (mut u, mut v) = (top, f(a, b, top));
                if v == 0.0 {
                    return u;
                }
                let dir = if v > 0.0 { -1.0 } else { 1.0 };
                for _ in 0..64 {
                    let next = u + v.abs().clamp(0.5 * vx, band) * dir;
                    if next < bottom || next > top + 2.0 * band {
                        return f32::NAN;
                    }
                    let nv = f(a, b, next);
                    if (nv < 0.0) != (v < 0.0) {
                        let (mut lo, mut hi) = (u, next);
                        for _ in 0..10 {
                            let m = 0.5 * (lo + hi);
                            if (f(a, b, m) < 0.0) == (v < 0.0) {
                                lo = m;
                            } else {
                                hi = m;
                            }
                        }
                        return 0.5 * (lo + hi);
                    }
                    u = next;
                    v = nv;
                }
                f32::NAN
            })
            .collect();
        Ground { on, n, t1, t2, step, half, h }
    }

    /// Where the line through `p` along `n` meets the surface of levels up to `l` (as it was when the stroke began,
    /// with `base`), within `limit` of `p`.
    fn along_onto(&self, l: usize, mut p: Vec3, n: Vec3, limit: f32, base: bool) -> Option<Vec3> {
        let start = p;
        let vx = self.voxel(l);
        for _ in 0..48 {
            let v = self.value_upto(p, l, base);
            if v.abs() < 0.01 * vx {
                break;
            }
            p -= n * v;
            if (p - start).length() > limit {
                return None;
            }
        }
        Some(p)
    }

    /// The surface's overall normal under a brush of radius `r` at `c` (as it was when the stroke began): the plane
    /// through rings of surface points around `c`, found along `n`.
    /// With `wide`, when no ring meets the surface along `n`, they are looked for again along the brush's own scale's
    /// normal (see below).
    fn area_normal(&self, l: usize, c: Vec3, n: Vec3, r: f32, wide: bool) -> Vec3 {
        let (t1, t2) = n.any_orthonormal_pair();
        let mut sum = Vec3::ZERO;
        for (ring, weight) in [(0.5, 0.6), (0.8, 1.0), (1.0, 0.8)] {
            let pts: Option<Vec<Vec3>> = (0..12)
                .map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 12.0;
                    self.along_onto(l, c + (t1 * a.cos() + t2 * a.sin()) * ring * r, n, r, true)
                })
                .collect();
            let Some(pts) = pts else { continue };
            let mid = pts.iter().sum::<Vec3>() / pts.len() as f32;
            let mut nn = Vec3::ZERO;
            for k in 0..pts.len() {
                nn += (pts[k] - mid).cross(pts[(k + 1) % pts.len()] - mid);
            }
            if let Some(u) = nn.try_normalize() {
                sum += u * weight;
            }
        }
        match sum.try_normalize().filter(|u| u.dot(n) > 0.3) {
            Some(u) => u,
            // No ring met the surface along `n`: tilted that far from the area's (the normal at a point on a bump, some
            // 50 degrees off, as a stroke's first dab takes it), no ring point's line along it meets the surface near.
            // Given back as it came, the brush kept that tilt from dab to dab until a ring met the surface, and then
            // turned some 20 degrees in one. On a stroke's first dab they are looked for again along the normal at the
            // brush's own scale: level 0's field differenced half the brush either way (within its band), which bumps
            // smaller than the brush hardly turn. (Not later in a stroke, whose brush already lies along the area.)
            None if wide && sum == Vec3::ZERO => {
                let h = (0.5 * r).min(0.8 * self.band(0));
                let d = |o: Vec3| self.value_upto(c + o, 0, true) - self.value_upto(c - o, 0, true);
                match Vec3::new(d(Vec3::X * h), d(Vec3::Y * h), d(Vec3::Z * h)).try_normalize() {
                    Some(w) if w.dot(n) < 10f32.to_radians().cos() => self.area_normal(l, c, w, r, false),
                    _ => n,
                }
            }
            None => n,
        }
    }

    /// `p` moved onto the surface of levels up to `l` as it was when the stroke began, along that surface's normal.
    fn onto_base(&self, l: usize, mut p: Vec3, dir: Vec3) -> Vec3 {
        let band = self.band(self.level_at(p, l, true));
        for _ in 0..32 {
            let v = self.value_upto(p, l, true);
            if v.abs() < band * 0.95 {
                break;
            }
            p -= dir * v;
        }
        for _ in 0..3 {
            let v = self.value_upto(p, l, true);
            p -= self.normal_upto(p, l, true) * v;
        }
        p
    }

    /// One dab of a plane brush at level `l`, the way the clay and trim brushes of mesh sculpting work (see
    /// `Volume::plane_dab`): the plane takes its tilt from the surface as it was when the stroke began and its height
    /// from the surface as it is now under the brush. Clay fills below a plane `h` above the surface and keeps building
    /// as you scrub; trim cuts above a plane `h` below it. Each dab moves the surface `alpha` of the way to the plane,
    /// fully out to `hardness` of the radius and easing to nothing at the rim. `along` is the rough normal near `at`.
    /// Returns the plane's normal and what the dab did.
    #[allow(clippy::too_many_arguments)]
    pub fn plane_dab(&mut self, l: usize, at: Vec3, along: Vec3, r: f32, h: f32, hardness: f32, alpha: f32, trim: bool) -> Option<(Vec3, LevelDab)> {
        let l = l.min(self.levels() - 1);
        let on = self.onto_base(l, at, along);
        if self.normal_upto(on, l, true) == Vec3::ZERO {
            return None;
        }
        // (On a stroke's first dab, its rings looked for farther when none meets the surface: see `area_normal`.)
        let first = self.stroke.as_ref().is_none_or(|s| s.is_empty());
        let rough = self.area_normal(l, on, along, r, first);
        let n = self.area_normal(l, on, rough, r, first);
        let (t1, t2) = n.any_orthonormal_pair();
        let (mut base_sum, mut added_sum, mut weight) = (0.0, 0.0, 0.0);
        for (ring, w, count) in [(0.0, 1.0, 1), (0.5, 0.8, 8), (0.9, 0.5, 12)] {
            for k in 0..count {
                let a = k as f32 * std::f32::consts::TAU / count as f32;
                let p = on + (t1 * a.cos() + t2 * a.sin()) * ring * r;
                if let (Some(b), Some(c)) = (self.along_onto(l, p, n, r, true), self.along_onto(l, p, n, r, false)) {
                    base_sum += (b - on).dot(n) * w;
                    added_sum += (c - b).dot(n) * w;
                    weight += w;
                }
            }
        }
        let (base, added) = if weight > 0.0 { (base_sum / weight, added_sum / weight) } else { (0.0, 0.0) };
        let ground = if trim { base.min(0.0) } else { base.max(0.0) };
        let level = if trim { ground } else { ground + added };
        let plane = if trim { ground - h } else { (ground + h).max(level + BUILD_STEP * h) };
        let hard = hardness.clamp(0.0, 0.95);
        let depth = 0.35 * r;
        let weight = move |p: Vec3| {
            let q = p - on;
            let u = q.dot(n);
            let t = (q - n * u).length();
            let x = ((t / r - hard) / (1.0 - hard)).clamp(0.0, 1.0);
            alpha * (1.0 - x * x * (3.0 - 2.0 * x))
        };
        let k = (0.5 * h).max(self.voxel(l));
        // The surface as the stroke found it, seen from above the brush (see [`Ground`]): a dab's slab reaches `depth`
        // past the plane's level, and also down to that surface (up to it, trimming) and `margin` past it wherever it
        // lies farther. Held to the slab alone, a dab whose surface fell away under part of it (the end of a stroke
        // where the form curves off, a valley between earlier strokes) laid a slab floating over it: one row of
        // samples left in the air between, a hidden fold that later strokes broke into as dents and rows of dots (and
        // a trim, over a ridge higher than its slab, left a shell). The surface seen from above, not the field:
        // whatever lies behind a thin part (the back of a fin) is as far as before.
        let (vx, band) = (self.voxel(l), self.band(l));
        let margin = 5.0 * vx;
        // Looked for only as far as half the radius past where the slab ends (farther, the slab stops where it did):
        // the columns' blocks are made for the box they span.
        // (A clay dab wide enough to follow the surface, below, lays its clay on that surface already.)
        let follows = !trim && -base / r > 0.14;
        let (top, bottom) = if trim { (level + depth + 0.5 * r, plane - band) } else { (plane + band, level - depth - 0.5 * r) };
        let front = if follows { Ground { on, n, t1: n, t2: n, step: 1.0, half: 0, h: vec![f32::NAN] } } else { self.ground(l, on, n, r + 4.0 * vx, top, bottom, true) };
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for (i, &h) in front.h.iter().enumerate() {
            let (a, b) = ((i as i32 % front.size() - front.half) as f32, (i as i32 / front.size() - front.half) as f32);
            if h.is_finite() && (a * a + b * b).sqrt() * front.step <= r + 2.0 * front.step {
                low = low.min(h);
                high = high.max(h);
            }
        }
        let deepest = if low.is_finite() { depth.max(level - low + margin).max(high + margin - level) } else { depth };
        let reach = 1.1 * r + deepest.min(2.0 * r);
        let dab = if trim {
            let top = (level + depth).max(high + margin);
            let shape = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                let ceiling = match front.distance(p) {
                    Some(d) => (u - (level + depth)).min(d - margin),
                    None => u - (level + depth),
                };
                (plane - u).max(ceiling).max((q - n * u).length() - r)
            };
            self.modify(l, on, reach, k, true, (n, plane, top), Some(Push { on, n, r, hard }), None, &weight, &shape)
        } else {
            let bottom = (level - depth).min(low - margin);
            let flat = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                let floor = match front.distance(p) {
                    Some(d) => ((level - depth) - u).min(-d - margin),
                    None => (level - depth) - u,
                };
                (u - plane).max(floor).max((q - n * u).length() - r)
            };
            // Under a brush wide enough for the form to curve away from the plane under it (by more than a band, and
            // the brush's rings lie more than 0.14 of its radius below its centre on average, where the dab's slab,
            // 0.35 of the radius deep, begins to leave the rim floating: a stroke seen from far, as wide as the ball,
            // not a small brush on a bump of its own size), clay lies on the surface as it was when the stroke began,
            // not on the plane: heights are measured from that surface under each point (see [`Ground`], smoothed to
            // the brush's scale), so the stroke lays a shell that follows the form instead of a stack of flat discs,
            // one per dab, each tilted with its own normal, that terrace where they meet; and nothing is laid where
            // no surface lies under the brush. The two mix by how far the form curves away, so a brush just that
            // wide shows no seam; under a smaller one the plane is used as it is.
            let x = ((-base / r - 0.14) / 0.1).clamp(0.0, 1.0);
            let floor = if x > 0.0 { Some(self.ground(l, on, n, r + 4.0 * self.voxel(l), plane + band, -(2.0 * r + band), true)) } else { None };
            let (mut low, mut high) = (0.0f32, 0.0f32);
            if let Some(floor) = &floor {
                for (i, &h) in floor.h.iter().enumerate() {
                    let (a, b) = ((i as i32 % floor.size() - floor.half) as f32, (i as i32 / floor.size() - floor.half) as f32);
                    if h.is_finite() && (a * a + b * b).sqrt() * floor.step <= r {
                        low = low.min(h);
                        high = high.max(h);
                    }
                }
            }
            let x = x.min(((high - low - band) / band).clamp(0.0, 1.0));
            let follow = x * x * (3.0 - 2.0 * x);
            if follow <= 0.0 {
                self.modify(l, on, reach, k, false, (n, bottom, plane), Some(Push { on, n, r, hard }), None, &weight, &flat)
            } else {
                let mut floor = floor.unwrap();
                floor.smooth((floor.half / 8).max(1));
                let shape = move |p: Vec3| {
                    let q = p - on;
                    let u = q.dot(n);
                    let Some(d) = floor.distance_smooth(p) else { return f32::INFINITY };
                    let d = u + follow * (d - u);
                    (d - plane).max((level - depth) - d).max((q - n * u).length() - r)
                };
                self.modify(l, on, reach, k, false, (n, level - depth + follow * low, plane + follow * high), Some(Push { on, n, r, hard }), None, &weight, &shape)
            }
        };
        Some((n, dab))
    }

    /// Add (or with `carve`, remove) a soft ball of radius `r` at `c`, blended over `k`, at level `l`.
    pub fn ball_dab(&mut self, l: usize, c: Vec3, r: f32, k: f32, carve: bool) -> LevelDab {
        let l = l.min(self.levels() - 1);
        let shape = move |p: Vec3| (p - c).length() - r;
        let weight = |_: Vec3| 1.0;
        self.modify(l, c, r + k, k, carve, (Vec3::Z, -(r + k), r + k), None, None, &weight, &shape)
    }

    /// One dab of the smooth brush at level `l`: the surface under the brush, as heights along its overall normal (see
    /// [`Ground`], read from the field as it is now, so scrubbing goes on smoothing), is smoothed at about a quarter
    /// of the radius, and each sample moves `alpha` of the way (with the falloff, as the plane brushes' does) to its
    /// distance from that smoothed surface. The smoothing keeps the form at the brush's scale: the heights less what
    /// three times taking away the blurred heights leaves (1 - (1 - blur)^3) keep any polynomial up to the fifth
    /// degree as it is, so a fresh ball stays a ball (a blur alone flattens it under the brush, twice the blurred
    /// heights less those blurred twice raised it by 0.7 of a step under a brush half its radius), while bumps,
    /// ridges and terraces narrower than the blur go. Like the plane brushes it works
    /// on the surface facing along the normal, down to 0.35 of the radius under it; finer levels under it take the
    /// smoothed surface (their detail is buried, or carried with `keep_detail`). Returns the normal and what it did.
    pub fn smooth_dab(&mut self, l: usize, at: Vec3, along: Vec3, r: f32, hardness: f32, alpha: f32) -> Option<(Vec3, LevelDab)> {
        let l = l.min(self.levels() - 1);
        let on = self.onto_base(l, at, along);
        if self.normal_upto(on, l, true) == Vec3::ZERO {
            return None;
        }
        let first = self.area_normal(l, on, along, r, false);
        let n = self.area_normal(l, on, first, r, false);
        let (vx, band) = (self.voxel(l), self.band(l));
        let depth = 0.35 * r;
        // Columns out past the radius by the blur's reach, so the blur inside it sees both sides.
        let heights = self.ground(l, on, n, 1.75 * r, depth + band, -(2.0 * depth + band), false);
        let rho = ((r / heights.step / 4.0).round() as i32).max(1);
        // The form at the brush's scale, a quartic fitted to the heights, is kept as it is and only what departs from it
        // is smoothed: columns that meet no surface (past a ball's horizon under a wide brush) leave the blur one-sided
        // near them, and blurring the heights themselves there lifted the surface under the brush (a ball smoothed by a
        // brush half its radius rose by 0.65 of a step).
        let size = heights.size();
        let fit = {
            let (mut m, mut v) = ([[0.0f64; 15]; 15], [0.0f64; 15]);
            let scale = 1.5 * r / heights.step;
            for (i, &h) in heights.h.iter().enumerate() {
                let (a, b) = (((i as i32 % size) - heights.half) as f64 / scale as f64, ((i as i32 / size) - heights.half) as f64 / scale as f64);
                let d2 = a * a + b * b;
                if !h.is_finite() || d2 >= 1.0 {
                    continue;
                }
                let w = (1.0 - d2) * (1.0 - d2);
                let f = quartic_terms(a, b);
                for j in 0..15 {
                    v[j] += w * f[j] * h as f64;
                    for k in 0..15 {
                        m[j][k] += w * f[j] * f[k];
                    }
                }
            }
            solve(m, v).map(|c| move |i: usize| {
                let (a, b) = (((i as i32 % size) - heights.half) as f64 / scale as f64, ((i as i32 / size) - heights.half) as f64 / scale as f64);
                quartic_terms(a, b).iter().zip(c.iter()).map(|(t, c)| t * c).sum::<f64>() as f32
            })
        };
        let Some(fit) = fit else { return None };
        // What departs from the fit, out to where it was fitted (fading to nothing over its last sixth), and nothing
        // past it or where no surface was met: the blur near the grid's edge, or a horizon, then reads the fitted form
        // going on rather than nothing (one-sided, it sank the middle of a ball by a tenth of a step).
        let mut heights_off = heights.clone();
        let scale = 1.5 * r / heights.step;
        for (i, h) in heights_off.h.iter_mut().enumerate() {
            let (a, b) = (((i as i32 % size) - heights.half) as f32 / scale, ((i as i32 / size) - heights.half) as f32 / scale);
            let x = ((1.0 - (a * a + b * b).sqrt()) * 6.0).clamp(0.0, 1.0);
            *h = if h.is_finite() { (*h - fit(i)) * x * x * (3.0 - 2.0 * x) } else { 0.0 };
        }
        let mut once = heights_off.clone();
        once.smooth(rho);
        let mut twice = once.clone();
        twice.smooth(rho);
        let mut thrice = twice.clone();
        thrice.smooth(rho);
        let mut target = heights.clone();
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (i, t) in target.h.iter_mut().enumerate() {
            if t.is_finite() {
                let h = *t;
                *t = fit(i) + 3.0 * (once.h[i] - twice.h[i]) + thrice.h[i];
                let (a, b) = ((i as i32 % heights.size() - heights.half) as f32, (i as i32 / heights.size() - heights.half) as f32);
                if (a * a + b * b).sqrt() * heights.step <= r + 2.0 * heights.step {
                    lo = lo.min(h.min(*t));
                    hi = hi.max(h.max(*t));
                }
            }
        }
        if !lo.is_finite() {
            return None;
        }
        let hard = hardness.clamp(0.0, 0.95);
        let weight = move |p: Vec3| {
            let q = p - on;
            let t = (q - n * q.dot(n)).length();
            let x = ((t / r - hard) / (1.0 - hard)).clamp(0.0, 1.0);
            alpha * (1.0 - x * x * (3.0 - 2.0 * x))
        };
        let shape = move |p: Vec3| target.distance_smooth(p).unwrap_or(f32::INFINITY);
        let (s_lo, s_hi) = (lo - depth, hi + band);
        let reach = (r * r + s_lo.abs().max(s_hi.abs()).powi(2)).sqrt();
        let dab = self.modify(l, on, reach, vx, false, (n, s_lo, s_hi), Some(Push { on, n, r, hard }), Some(depth), &weight, &shape);
        Some((n, dab))
    }

    // ------------------------------------------------------------------ strokes and history

    /// Start a stroke: until [`Self::end_stroke`], brushes are placed on the surface as it is now.
    pub fn begin_stroke(&mut self) {
        self.stroke = Some(Shared::default());
        self.carry.clear();
        self.carry_push.clear();
        self.recent.clear();
        self.moved.clear();
        for p in self.pending.iter_mut().chain(self.carried.iter_mut()) {
            p.clear();
        }
    }

    pub fn in_stroke(&self) -> bool {
        self.stroke.is_some()
    }

    /// End the stroke: move the form it built at fine levels down a level, remove the specks it left, and keep it for
    /// [`Self::undo`].
    pub fn end_stroke(&mut self) -> StrokeEnd {
        let t = Instant::now();
        if self.stroke.as_ref().is_none_or(|s| s.is_empty()) {
            self.stroke = None;
            return StrokeEnd::default();
        }
        let mut out = self.stroke_end_steps();
        self.keep_stroke();
        out.ms = t.elapsed().as_secs_f64() * 1e3;
        out
    }

    /// What ending a stroke changes in the field: the rest of its carrying, distances again, form moved down, specks.
    fn stroke_end_steps(&mut self) -> StrokeEnd {
        let mut out = StrokeEnd::default();
        self.carry_pending(None);
        self.drop_empty();
        let touched: Vec<Key> = self.stroke.as_ref().unwrap().keys().copied().collect();
        self.redistance(&touched, false);
        // Moving form down a level moves the coarser level's values toward the finer surface, which puts its
        // surface right but leaves the field around it compressed (its slope fell to half a step from the surface):
        // the finer level's detail against such a field has a slope of its own along the normal, and a later stroke
        // carrying it a fraction of a step off the surface read that slope as a lattice of dips. So distances again
        // where it moved.
        let rebased = self.rebase(&touched);
        out.rebased = rebased.len();
        self.redistance(&rebased, true);
        // And the blocks beside them: a coarser block moved toward the finer surface beside one that was not, with
        // only the moved one measured again, left samples inside the solid shallower than any distance (a phantom
        // surface beneath the real one, where the finer level's blocks end), which a later stroke cutting down to it
        // showed as dents and steps a block wide. Their surface stays where it is.
        let rebased_set: HashSet<Key> = rebased.iter().copied().collect();
        let mut beside: Vec<Key> = rebased
            .iter()
            .flat_map(|&(l, bc)| (0..27).map(move |k| (l, bc + IVec3::new(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1))))
            .filter(|k| !rebased_set.contains(k) && self.blocks[k.0 as usize].contains_key(&k.1))
            .collect();
        beside.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        beside.dedup();
        self.redistance(&beside, false);
        let touched: Vec<Key> = self.stroke.as_ref().unwrap().keys().copied().collect();
        out.specks = self.remove_specks(&touched, SPECK_VERTS);
        out
    }

    /// Drop the blocks the stroke touched at finer levels that hold nothing but the level below (making parents for
    /// carried detail leaves some), finest first, so a parent left with no children can go too.
    fn drop_empty(&mut self) {
        let Some(stroke) = &self.stroke else { return };
        let mut keys: Vec<Key> = stroke.keys().copied().filter(|k| k.0 > 0).collect();
        keys.sort_unstable_by_key(|k| (std::cmp::Reverse(k.0), k.1.z, k.1.y, k.1.x));
        for (m8, bc) in keys {
            let m = m8 as usize;
            if !self.blocks[m].contains_key(&bc) {
                continue;
            }
            if m + 1 < self.levels() && (0..8).any(|q| self.blocks[m + 1].contains_key(&(bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)))) {
                continue;
            }
            // What the level below would show here instead, as `grid` interpolates it where a level has no block.
            let (tol, band) = (0.02 * self.voxel(m), self.band(m));
            let below = self.grid(m - 1, (bc * B).div_euclid(IVec3::splat(2)) - IVec3::ONE, (bc * B + B3).div_euclid(IVec3::splat(2)) + IVec3::ONE, false);
            let s = &self.blocks[m][&bc];
            let empty = s.iter().enumerate().all(|(i, v)| (v - below.trilinear((bc * B + cell_of(i)).as_vec3() * 0.5).clamp(-band, band)).abs() <= tol);
            if empty {
                self.keep((m8, bc));
                self.blocks[m].remove(&bc);
                self.dirty.insert((m8, bc));
            }
        }
    }

    /// End a stroke on another thread: [`EndJob::run`] does the field's part on a copy of the volume, and
    /// [`Self::end_stroke_apply`] takes its result. Nothing may change the volume in between (the caller waits for the
    /// job before the next stroke, an undo or a save); what it changes is then exactly the copy's. `None` when the
    /// stroke changed nothing (it has ended).
    pub fn end_stroke_take(&mut self) -> Option<EndJob> {
        if self.stroke.as_ref().is_none_or(|s| s.is_empty()) {
            self.stroke = None;
            return None;
        }
        // Carrying the job's copy cannot do (it starts with nothing queued).
        self.carry_pending(None);
        let mut vol = self.snapshot();
        vol.pending = self.pending.clone();
        Some(EndJob { vol })
    }

    /// Take what an [`EndJob`] did, and keep the stroke for [`Self::undo`].
    pub fn end_stroke_apply(&mut self, done: EndDone) -> StrokeEnd {
        let EndDone { vol, mut end } = done;
        let t = Instant::now();
        self.blocks = vol.blocks;
        self.inside0 = vol.inside0;
        self.stroke = vol.stroke;
        self.dirty.extend(vol.dirty);
        self.keep_stroke();
        end.ms += t.elapsed().as_secs_f64() * 1e3;
        end
    }

    /// Put the stroke just ended in the history, dropping the oldest strokes past [`HISTORY_BYTES`].
    fn keep_stroke(&mut self) {
        let saved = self.stroke.take().unwrap_or_default();
        self.history.push(saved);
        self.redo.clear();
        let size = |m: &Shared<Key, Was>| m.values().map(|w| 16 + if matches!(w, Was::Stored(_)) { N * 4 } else { 0 }).sum::<usize>();
        let mut total: usize = self.history.iter().map(size).sum();
        while total > HISTORY_BYTES && self.history.len() > 1 {
            total -= size(&self.history.remove(0));
        }
    }

    pub fn undo(&mut self) -> bool {
        self.stroke = None;
        for p in &mut self.pending {
            p.clear();
        }
        let Some(entry) = self.history.pop() else { return false };
        let back = self.swap_in(entry);
        self.redo.push(back);
        true
    }

    pub fn redo(&mut self) -> bool {
        self.stroke = None;
        for p in &mut self.pending {
            p.clear();
        }
        let Some(entry) = self.redo.pop() else { return false };
        let back = self.swap_in(entry);
        self.history.push(back);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    fn swap_in(&mut self, entry: Shared<Key, Was>) -> Shared<Key, Was> {
        let mut back = Shared::default();
        for (&k, was) in entry.iter() {
            let was = was.clone();
            let (l, bc) = (k.0 as usize, k.1);
            let now = match self.blocks[l].remove(&bc) {
                Some(b) => Was::Stored(b),
                None if l == 0 && self.inside0.remove(&bc) => Was::Inside,
                None => Was::Missing,
            };
            if l == 0 {
                self.inside0.remove(&bc);
            }
            match was {
                Was::Stored(b) => {
                    self.blocks[l].insert(bc, b);
                }
                Was::Inside => {
                    self.inside0.insert(bc);
                }
                Was::Missing => {}
            }
            back.insert(k, now);
            self.dirty.insert(k);
        }
        back
    }

    /// Make the touched blocks' samples distances again. Plane brushes move samples part of the way to a shape, so deep
    /// inside clay a stroke built up, or far above what it cut, values stay near zero instead of growing with the
    /// distance to the surface; the sign is right, but anything that reads distances (carrying detail, moving form down
    /// a level, the next brush's plane) would be misled. Samples next to the surface (a neighbour along an axis on the
    /// other side) keep their values, so the surface does not move; every other sample becomes its distance to the
    /// surface those samples place, within the band: the solution of `|grad d| = 1` grown out from them (swept across
    /// the block and the samples around it, as narrow-band level sets are re-initialised). That distance is smooth from
    /// sample to sample, as it must be: a later stroke moves the surface onto what are now samples a few steps off it,
    /// and its blends start from their values. Taken instead as the distance to the nearest point where the surface
    /// crossed a lattice edge, each was off by its own part of the crossings' spacing (up to half a step, too far), and
    /// a stroke raising the surface a couple of steps over earlier clay brought that grain up as scattered pits, and as
    /// a broken stair along an earlier stroke's wall.
    /// With `near_too`, the samples next to the surface become distances as well (see below); otherwise they keep
    /// their values, which place the surface.
    fn redistance(&mut self, touched: &[Key], near_too: bool) {
        for l in 0..self.levels() {
            let blocks: Vec<IVec3> = touched.iter().filter(|k| k.0 as usize == l && self.blocks[l].contains_key(&k.1)).map(|k| k.1).collect();
            if blocks.is_empty() {
                continue;
            }
            if l == 0 && self.redistance0(&blocks, near_too) {
                continue;
            }
            let (vx, band) = (self.voxel(l), self.band(l));
            // Distances out to the band (level 0 has its own sweep, `redistance0`, unless its blocks span too much).
            let reach = Self::redistance_reach(vx, band);
            let cap = (band / vx).min(reach as f32);
            let me = &*self;
            let fixed: Vec<(IVec3, Block)> = blocks
                .par_iter()
                .filter_map(|&bc| {
                    let lo = bc * B - IVec3::splat(reach.max(1));
                    let g = me.grid(l, lo, bc * B + IVec3::splat(B - 1 + reach.max(1)), false);
                    let n = g.n;
                    let hi = lo + n - IVec3::ONE;
                    // A sample's distance (in steps) to the plane through where the surface crosses its own lattice
                    // edges: the intercepts along each axis that crosses, one over the root of the sum of their
                    // inverse squares. Where one axis crosses, both ends of the edge keep their crossing exactly;
                    // elsewhere it moves by what the plane model is off, a grain at the sample's scale, which is why
                    // it is taken only with `near_too`, where moving form down a level left the field compressed
                    // around its surface (carried detail reading that a step or two off the old surface came out as a
                    // lattice of dips). No gradient is used: the field's gradient is no guide to the surface's normal
                    // inside fresh clay. Infinite for a sample with no crossing edge.
                    let intercept = |p: IVec3| -> f32 {
                        let a = g.at(p);
                        let mut inv2 = 0.0f32;
                        for axis in 0..3 {
                            let mut t_near = f32::INFINITY;
                            for dir in [-1, 1] {
                                let mut j = p;
                                j[axis] += dir;
                                if j[axis] < lo[axis] || j[axis] > hi[axis] {
                                    continue;
                                }
                                let b = g.at(j);
                                if (a < 0.0) != (b < 0.0) {
                                    t_near = t_near.min(a / (a - b));
                                }
                            }
                            if t_near.is_finite() {
                                inv2 += 1.0 / (t_near * t_near).max(1e-12);
                            }
                        }
                        if inv2 > 0.0 { 1.0 / inv2.sqrt() } else { f32::INFINITY }
                    };
                    // The samples next to the surface seed the distances, with their own values (or with `near_too`,
                    // their distances to the plane through their crossings); every other sample starts at the cap.
                    // Held with a border of one sample all round that reads as no neighbour (infinitely far), so the
                    // sweep reads each sample's six neighbours without asking whether they are in the box.
                    let pn = n + IVec3::splat(2);
                    let pat = |q: IVec3| (((q.z + 1) * pn.y + q.y + 1) * pn.x + q.x + 1) as usize;
                    let plen = (pn.x * pn.y * pn.z) as usize;
                    let mut d = vec![f32::INFINITY; plen];
                    let mut seed = vec![false; plen];
                    let mut any_seed = false;
                    for z in 0..n.z {
                        for y in 0..n.y {
                            for x in 0..n.x {
                                let q = IVec3::new(x, y, z);
                                let k = pat(q);
                                d[k] = cap;
                                let t = intercept(lo + q);
                                if t.is_finite() {
                                    seed[k] = true;
                                    any_seed = true;
                                    d[k] = if near_too { t } else { g.at(lo + q).abs() / vx }.min(cap);
                                }
                            }
                        }
                    }
                    if any_seed {
                        // Fast sweeping: the upwind solution of |grad d| = 1 (exact for a plane), swept once in each of
                        // the eight orders of the axes' directions (a second round changes nothing that shows).
                        // A sample's update is never less than its least neighbour, and no sample holds more than the
                        // cap: a row whose own samples and the four rows beside it all hold the cap (most of the box,
                        // past a few steps from the surface) changes nothing, and is passed over. Each row's least
                        // value is kept as the sweep lowers it.
                        let row = |y: i32, z: i32| ((z + 1) * pn.y + y + 1) as usize;
                        let mut least_in = vec![f32::INFINITY; (pn.y * pn.z) as usize];
                        for z in 0..n.z {
                            for y in 0..n.y {
                                let k0 = pat(IVec3::new(0, y, z));
                                least_in[row(y, z)] = d[k0..k0 + n.x as usize].iter().copied().fold(f32::INFINITY, f32::min);
                            }
                        }
                        let (sy, sz) = (pn.x as usize, (pn.x * pn.y) as usize);
                        for o in 0..8 {
                            let (fx, fy, fz) = (o & 1 == 1, o & 2 == 2, o & 4 == 4);
                            for zi in 0..n.z {
                                let z = if fz { n.z - 1 - zi } else { zi };
                                for yi in 0..n.y {
                                    let y = if fy { n.y - 1 - yi } else { yi };
                                    let r = row(y, z);
                                    let near = least_in[r].min(least_in[r - 1]).min(least_in[r + 1]).min(least_in[r - pn.y as usize]).min(least_in[r + pn.y as usize]);
                                    if near >= cap {
                                        continue;
                                    }
                                    let mut low = least_in[r];
                                    for xi in 0..n.x {
                                        let x = if fx { n.x - 1 - xi } else { xi };
                                        let k = pat(IVec3::new(x, y, z));
                                        if seed[k] {
                                            continue;
                                        }
                                        let mut a = [d[k - 1].min(d[k + 1]), d[k - sy].min(d[k + sy]), d[k - sz].min(d[k + sz])];
                                        a.sort_unstable_by(f32::total_cmp);
                                        let mut u = a[0] + 1.0;
                                        if u > a[1] {
                                            u = 0.5 * (a[0] + a[1] + (2.0 - (a[0] - a[1]) * (a[0] - a[1])).max(0.0).sqrt());
                                            if u > a[2] {
                                                let s = a[0] + a[1] + a[2];
                                                let disc = s * s - 3.0 * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - 1.0);
                                                u = (s + disc.max(0.0).sqrt()) / 3.0;
                                            }
                                        }
                                        if u < d[k] {
                                            d[k] = u;
                                            low = low.min(u);
                                        }
                                    }
                                    least_in[r] = low;
                                }
                            }
                        }
                    }
                    let old = &me.blocks[l][&bc];
                    let mut s: Block = Box::new(**old);
                    let mut any = false;
                    for (i, v) in s.iter_mut().enumerate() {
                        let p = bc * B + cell_of(i);
                        let k = pat(p - lo);
                        if seed[k] && !near_too {
                            continue;
                        }
                        let dist = (d[k] * vx).min(band);
                        let next = if g.at(p) < 0.0 { -dist } else { dist };
                        if next != *v {
                            *v = next;
                            any = true;
                        }
                    }
                    any.then_some((bc, s))
                })
                .collect();
            for (bc, s) in fixed {
                self.put(l, bc, s);
            }
        }
    }

    /// [`Self::redistance`] at level 0, out to its whole band (sixteen steps). Swept over a box around each block, as
    /// the finer levels are, the box would have to reach sixteen steps past the block on every side (forty samples
    /// across for its eight) and the sweep would cost several times the stroke. Instead the blocks the stroke touched
    /// and the stored blocks within the band of them are swept together: each block is swept on its own (in the eight
    /// orders), reading its neighbours' faces as they were after the last round, and rounds go on until nothing
    /// changes, only blocks next to one that changed being swept again. Every sample then holds the same solution of
    /// `|grad d| = 1` a sweep over the whole region gives. Unstored blocks (all at the band) are no neighbour. False
    /// (nothing done) when the touched blocks span too wide a box to index.
    fn redistance0(&mut self, blocks: &[IVec3], near_too: bool) -> bool {
        let (vx, band) = (self.voxel(0), self.band(0));
        let cap = band / vx;
        let halo = (cap / B as f32).ceil() as i32;
        let (mut lo, mut hi) = (blocks[0], blocks[0]);
        for &b in blocks {
            lo = lo.min(b);
            hi = hi.max(b);
        }
        lo -= IVec3::splat(halo);
        hi += IVec3::splat(halo);
        let dim = hi - lo + IVec3::ONE;
        let total = dim.x as usize * dim.y as usize * dim.z as usize;
        if total > 1 << 26 {
            return false;
        }
        let cell = |c: IVec3| ((c.z - lo.z) as usize * dim.y as usize + (c.y - lo.y) as usize) * dim.x as usize + (c.x - lo.x) as usize;
        // The touched blocks, grown by the halo along each axis in turn (a cube around each).
        let mut mark = vec![false; total];
        for &b in blocks {
            mark[cell(b)] = true;
        }
        for axis in 0..3 {
            let step = [1usize, dim.x as usize, dim.x as usize * dim.y as usize][axis];
            let len = dim[axis] as usize;
            let prev = mark.clone();
            for k in 0..total {
                if !prev[k] {
                    continue;
                }
                let at = (k / step) % len;
                for o in at.saturating_sub(halo as usize)..(at + halo as usize + 1).min(len) {
                    mark[k - at * step + o * step] = true;
                }
            }
        }
        let me = &*self;
        let mut index = vec![u32::MAX; total];
        let mut region: Vec<IVec3> = Vec::new();
        let mut vals: Vec<&[f32; N]> = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let c = IVec3::new(x, y, z);
                    let k = cell(c);
                    if mark[k]
                        && let Some(s) = me.blocks[0].get(&c)
                    {
                        index[k] = region.len() as u32;
                        region.push(c);
                        vals.push(&**s);
                    }
                }
            }
        }
        // Each block's neighbours across its faces: -x, +x, -y, +y, -z, +z.
        const FACE: [IVec3; 6] = [IVec3::NEG_X, IVec3::X, IVec3::NEG_Y, IVec3::Y, IVec3::NEG_Z, IVec3::Z];
        let nb: Vec<[u32; 6]> = region
            .iter()
            .map(|&c| {
                FACE.map(|f| {
                    let q = c + f;
                    if q.cmplt(lo).any() || q.cmpgt(hi).any() { u32::MAX } else { index[cell(q)] }
                })
            })
            .collect();
        // A sample's value, `p` within the block or one past it along one axis (in the neighbour across that face).
        let value = |r: usize, p: IVec3| -> Option<f32> {
            for axis in 0..3 {
                if p[axis] < 0 || p[axis] >= B {
                    let f = 2 * axis + (p[axis] >= B) as usize;
                    let n = nb[r][f];
                    if n == u32::MAX {
                        return None;
                    }
                    let mut q = p;
                    q[axis] = q[axis].rem_euclid(B);
                    return Some(vals[n as usize][local(q)]);
                }
            }
            Some(vals[r][local(p)])
        };
        // The seeds (samples next to the surface, as in `redistance`) and every other sample at the cap.
        let mut d: Vec<Box<[f32; N]>> = Vec::with_capacity(region.len());
        let mut seeds: Vec<SignMask> = Vec::with_capacity(region.len());
        (0..region.len())
            .into_par_iter()
            .map(|r| {
                let mut dd = Box::new([cap; N]);
                let mut seed = [0u64; 8];
                for (i, out) in dd.iter_mut().enumerate() {
                    let p = cell_of(i);
                    let a = vals[r][i];
                    let mut inv2 = 0.0f32;
                    for axis in 0..3 {
                        let mut t_near = f32::INFINITY;
                        for dir in [-1, 1] {
                            let mut q = p;
                            q[axis] += dir;
                            if let Some(b) = value(r, q)
                                && (a < 0.0) != (b < 0.0)
                            {
                                t_near = t_near.min(a / (a - b));
                            }
                        }
                        if t_near.is_finite() {
                            inv2 += 1.0 / (t_near * t_near).max(1e-12);
                        }
                    }
                    if inv2 > 0.0 {
                        seed[i / 64] |= 1 << (i % 64);
                        *out = if near_too { 1.0 / inv2.sqrt() } else { a.abs() / vx }.min(cap);
                    }
                }
                (dd, seed)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(dd, s)| {
                d.push(dd);
                seeds.push(s);
            });
        let is_seed = |r: usize, i: usize| seeds[r][i / 64] >> (i % 64) & 1 == 1;
        // Rounds of sweeping: a block is swept when it or a neighbour changed in the last round (at first, the blocks
        // holding seeds).
        let mut changed: Vec<bool> = seeds.iter().map(|s| s.iter().any(|&m| m != 0)).collect();
        const P: usize = (B + 2) as usize;
        for _ in 0..64 {
            let active: Vec<usize> = (0..region.len()).filter(|&r| changed[r] || nb[r].iter().any(|&n| n != u32::MAX && changed[n as usize])).collect();
            if active.is_empty() {
                break;
            }
            let swept: Vec<(usize, Option<Box<[f32; N]>>)> = active
                .par_iter()
                .map(|&r| {
                    // The block with a border of its neighbours' faces (none: infinitely far).
                    let mut pd = [f32::INFINITY; P * P * P];
                    let pat = |x: usize, y: usize, z: usize| (z * P + y) * P + x;
                    for i in 0..N {
                        let p = cell_of(i);
                        pd[pat(p.x as usize + 1, p.y as usize + 1, p.z as usize + 1)] = d[r][i];
                    }
                    for (f, &n) in nb[r].iter().enumerate() {
                        if n == u32::MAX {
                            continue;
                        }
                        let (axis, far) = (f / 2, f % 2 == 1);
                        for a in 0..B {
                            for b in 0..B {
                                let mut q = IVec3::ZERO;
                                q[(axis + 1) % 3] = a;
                                q[(axis + 2) % 3] = b;
                                q[axis] = if far { 0 } else { B - 1 };
                                let mut t = q + IVec3::ONE;
                                t[axis] = if far { B + 1 } else { 0 };
                                pd[pat(t.x as usize, t.y as usize, t.z as usize)] = d[n as usize][local(q)];
                            }
                        }
                    }
                    let (sy, sz) = (P, P * P);
                    let mut any = false;
                    for o in 0..8 {
                        let (fx, fy, fz) = (o & 1 == 1, o & 2 == 2, o & 4 == 4);
                        for zi in 0..B as usize {
                            let z = if fz { B as usize - 1 - zi } else { zi };
                            for yi in 0..B as usize {
                                let y = if fy { B as usize - 1 - yi } else { yi };
                                for xi in 0..B as usize {
                                    let x = if fx { B as usize - 1 - xi } else { xi };
                                    if is_seed(r, (z * 64 + y * 8 + x) as usize) {
                                        continue;
                                    }
                                    let k = pat(x + 1, y + 1, z + 1);
                                    let mut a = [pd[k - 1].min(pd[k + 1]), pd[k - sy].min(pd[k + sy]), pd[k - sz].min(pd[k + sz])];
                                    a.sort_unstable_by(f32::total_cmp);
                                    if a[0] >= pd[k] {
                                        continue;
                                    }
                                    let mut u = a[0] + 1.0;
                                    if u > a[1] {
                                        u = 0.5 * (a[0] + a[1] + (2.0 - (a[0] - a[1]) * (a[0] - a[1])).max(0.0).sqrt());
                                        if u > a[2] {
                                            let s = a[0] + a[1] + a[2];
                                            let disc = s * s - 3.0 * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - 1.0);
                                            u = (s + disc.max(0.0).sqrt()) / 3.0;
                                        }
                                    }
                                    if u < pd[k] - 1e-5 {
                                        pd[k] = u;
                                        any = true;
                                    }
                                }
                            }
                        }
                    }
                    if !any {
                        return (r, None);
                    }
                    let mut out = Box::new([0.0f32; N]);
                    for (i, v) in out.iter_mut().enumerate() {
                        let p = cell_of(i);
                        *v = pd[pat(p.x as usize + 1, p.y as usize + 1, p.z as usize + 1)];
                    }
                    (r, Some(out))
                })
                .collect();
            changed.iter_mut().for_each(|c| *c = false);
            for (r, out) in swept {
                if let Some(out) = out {
                    d[r] = out;
                    changed[r] = true;
                }
            }
        }
        // The touched blocks take the distances (the seeds keep their values unless `near_too`).
        let fixed: Vec<(IVec3, Block)> = blocks
            .par_iter()
            .filter_map(|&bc| {
                let r = index[cell(bc)] as usize;
                let old = vals[r];
                let mut s: Block = Box::new(*old);
                let mut any = false;
                for (i, v) in s.iter_mut().enumerate() {
                    if is_seed(r, i) && !near_too {
                        continue;
                    }
                    let dist = (d[r][i] * vx).min(band);
                    let next = if old[i] < 0.0 { -dist } else { dist };
                    if next != *v {
                        *v = next;
                        any = true;
                    }
                }
                any.then_some((bc, s))
            })
            .collect();
        for (bc, s) in fixed {
            self.put(0, bc, s);
        }
        true
    }

    /// Fine work that built real form, moved down a level: at each level from the finest, under each touched block,
    /// the level below moves toward this level's surface, smoothed to the coarser lattice, by however far the two differ
    /// beyond [`REBASE_KEEP`] of this level's band. Small detail (a groove, a scratch) stays where it was sculpted;
    /// broad form built up at a fine level ends up in the level below too, where a distant view sees it and a later
    /// coarse stroke treats it as form. The finer level's values do not change, so neither does the surface. Returns the
    /// coarser blocks changed.
    fn rebase(&mut self, touched: &[Key]) -> Vec<Key> {
        let mut changed_total = Vec::new();
        let mut region: HashSet<IVec3> = HashSet::default();
        for m in (1..self.levels()).rev() {
            let mut blocks: Vec<IVec3> = touched.iter().filter(|k| k.0 as usize == m).map(|k| k.1).chain(region.drain()).filter(|bc| self.blocks[m].contains_key(bc)).collect();
            blocks.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            blocks.dedup();
            if blocks.is_empty() {
                continue;
            }
            let (fband, cband) = (self.band(m), self.band(m - 1));
            let keep = REBASE_KEEP * fband;
            let mut under: Vec<IVec3> = blocks.iter().map(|fb| fb.div_euclid(IVec3::splat(2))).collect();
            under.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            under.dedup();
            let me = &*self;
            // Smoothing about a coarse step wide (binomial, 17 taps at the fine step): a feature counts as form only
            // where it is broad enough for the coarse level to hold; narrow detail, however tall, stays fine.
            let mut wts = [1.0f32; 17];
            for (k, w) in wts.iter_mut().enumerate() {
                let mut c = 1.0f64;
                for j in 0..k {
                    c = c * (16 - j) as f64 / (j + 1) as f64;
                }
                *w = (c / 65536.0) as f32;
            }
            let changes: Vec<(IVec3, Block)> = under
                .par_iter()
                .filter_map(|&cb| {
                    // This level's field around the coarse block, smoothed at the coarse points, one axis at a time.
                    const R: i32 = 8;
                    const NF: usize = (2 * B + 2 * R - 1) as usize;
                    let lo = cb * B * 2 - IVec3::splat(R);
                    let g = me.grid(m, lo, lo + IVec3::splat(NF as i32 - 1), false);
                    let at = |x: usize, y: usize, z: usize| g.v[(z * NF + y) * NF + x];
                    let mut ax = vec![0.0f32; 8 * NF * NF];
                    for z in 0..NF {
                        for y in 0..NF {
                            for x in 0..8 {
                                ax[(z * NF + y) * 8 + x] = (0..17).map(|k| wts[k] * at(2 * x + k, y, z)).sum();
                            }
                        }
                    }
                    let mut ay = vec![0.0f32; 8 * 8 * NF];
                    for z in 0..NF {
                        for y in 0..8 {
                            for x in 0..8 {
                                ay[(z * 8 + y) * 8 + x] = (0..17).map(|k| wts[k] * ax[(z * NF + 2 * y + k) * 8 + x]).sum();
                            }
                        }
                    }
                    let mut target = [f32::NAN; N];
                    for (i, t) in target.iter_mut().enumerate() {
                        let c = cb * B + cell_of(i);
                        // Wherever this level has a block, not only under the ones touched: moved coarse samples
                        // beside unmoved ones, block by block, left a lattice of steps after a big stroke carried
                        // the fine levels.
                        if !me.blocks[m].contains_key(&(c * 2).div_euclid(B3)) {
                            continue;
                        }
                        let l = cell_of(i);
                        let (x, y, z) = (l.x as usize, l.y as usize, l.z as usize);
                        *t = (0..17).map(|k| wts[k] * ay[((2 * z + k) * 8 + y) * 8 + x]).sum();
                    }
                    let mut s = me.block_values(m - 1, cb, false);
                    let mut any = false;
                    // A few passes: each moves the coarse field at most about a band toward the target, since it compares
                    // only within the finer band.
                    for _ in 0..4 {
                        let mut moved = false;
                        for (v, t) in s.iter_mut().zip(target.iter()) {
                            if t.is_nan() {
                                continue;
                            }
                            let d = t - v.clamp(-fband, fband);
                            let step = d.signum() * (d.abs() - keep).max(0.0);
                            if step != 0.0 {
                                *v = (*v + step).clamp(-cband, cband);
                                moved = true;
                            }
                        }
                        any |= moved;
                        if !moved {
                            break;
                        }
                    }
                    any.then_some((cb, s))
                })
                .collect();
            for (cb, s) in changes {
                self.put(m - 1, cb, s);
                region.insert(cb);
                changed_total.push((m as u8 - 1, cb));
            }
        }
        changed_total
    }

    /// Remove the specks among the surface around `touched`: closed pieces with at most `max_verts` vertices, all in
    /// blocks the stroke touched. Each is removed from the field, by giving the samples it encloses the sign around them.
    /// Returns how many were removed.
    fn remove_specks(&mut self, touched: &[Key], max_verts: usize) -> usize {
        let cap = self.levels() - 1;
        let region: HashSet<Key> = touched.iter().copied().filter(|k| self.block(*k).is_some()).collect();
        if region.is_empty() {
            return 0;
        }
        let work = self.extract_set(&region.iter().copied().collect::<Vec<_>>(), cap);
        let (surf, _) = self.extract(&work, cap);
        // Number the vertices, join them along quads.
        let mut first: HashMap<Key, u32> = HashMap::default();
        let mut total = 0u32;
        for (k, s) in &surf {
            first.insert(*k, total);
            total += s.verts.len() as u32;
        }
        let mut parent: Vec<u32> = (0..total).collect();
        fn root(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        let mut open = vec![false; total as usize];
        // Vertices no quad uses (a cell next to finer ones always has one) are not surface, let alone a speck.
        let mut used = vec![false; total as usize];
        for s in surf.values() {
            for q in &s.quads {
                let ids: Vec<Option<u32>> = q.iter().map(|(k, c, p)| Some(first.get(k)? + surf[k].vertex(*c, *p)? as u32)).collect();
                let known: Vec<u32> = ids.iter().flatten().copied().collect();
                for &v in &known {
                    used[v as usize] = true;
                }
                for w in known.windows(2) {
                    let (a, c) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
                    if a != c {
                        let o = open[a as usize] || open[c as usize];
                        parent[a.max(c) as usize] = a.min(c);
                        open[a.min(c) as usize] = o;
                    }
                }
                if ids.iter().any(|i| i.is_none()) && let Some(&k) = known.first() {
                    let r = root(&mut parent, k);
                    open[r as usize] = true;
                }
            }
        }
        // A piece with a vertex outside the stroke's blocks reaches the rest of the surface.
        for (k, s) in &surf {
            if !region.contains(k) {
                for v in 0..s.verts.len() as u32 {
                    let r = root(&mut parent, first[k] + v);
                    open[r as usize] = true;
                }
            }
        }
        let mut size: HashMap<u32, usize> = HashMap::default();
        for i in 0..total {
            let r = root(&mut parent, i);
            if used[i as usize] {
                *size.entry(r).or_default() += 1;
            }
        }
        let specks: HashSet<u32> = size.iter().filter(|&(&r, &n)| n <= max_verts && !open[r as usize]).map(|(&r, _)| r).collect();
        if specks.is_empty() {
            return 0;
        }
        // The cells (level, lattice cell) of each speck, and which pieces every cell holds.
        let mut cells_of: HashMap<u32, Vec<(usize, IVec3)>> = HashMap::default();
        let mut pieces_at: HashMap<(usize, IVec3), Vec<u32>> = HashMap::default();
        for (k, s) in &surf {
            let l = k.0 as usize;
            for (i, &info) in s.cells.iter().enumerate() {
                let count = (info >> 10) & 7;
                if count == 0 {
                    continue;
                }
                let c = k.1 * B + cell_of(i);
                for p in 0..count {
                    let r = root(&mut parent, first[k] + (info & 1023) + p);
                    pieces_at.entry((l, c)).or_default().push(r);
                    if specks.contains(&r) {
                        cells_of.entry(r).or_default().push((l, c));
                    }
                }
            }
        }
        let mut edits: Vec<(usize, IVec3, f32)> = Vec::new();
        for (r, cells) in &cells_of {
            // The samples at its cells' corners, where they are stored: the side with fewer is what the speck encloses.
            let mut corners: Vec<(usize, IVec3)> = cells.iter().flat_map(|&(l, c)| (0..8).map(move |k| (l, c + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1)))).map(|(l, p)| self.canonical(l, p, &Show::upto(cap))).collect();
            corners.sort_unstable_by_key(|(l, p)| (*l, p.z, p.y, p.x));
            corners.dedup();
            let value = |(l, p): (usize, IVec3)| self.sample(l, p, false);
            let neg = corners.iter().filter(|&&c| value(c) < 0.0).count();
            let pos = corners.len() - neg;
            if neg == pos {
                continue;
            }
            let enclosed_inside = neg < pos;
            for &(l, p) in &corners {
                if (value((l, p)) < 0.0) != enclosed_inside || !self.blocks[l].contains_key(&p.div_euclid(B3)) {
                    continue;
                }
                let to = if enclosed_inside { 0.5 * self.voxel(l) } else { -0.5 * self.voxel(l) };
                let shared = (0..8).any(|k| pieces_at.get(&(l, p - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).is_some_and(|ps| ps.iter().any(|q| q != r)));
                if !shared {
                    edits.push((l, p, to));
                }
            }
        }
        let mut flipped: Vec<HashMap<IVec3, SignMask>> = vec![HashMap::default(); self.levels()];
        for (l, p, to) in edits {
            let bc = p.div_euclid(B3);
            self.keep((l as u8, bc));
            let i = local(p);
            if let Some(s) = self.blocks[l].get_mut(&bc) {
                Arc::make_mut(s)[i] = to;
            }
            self.dirty.insert((l as u8, bc));
            flipped[l].entry(bc).or_insert([0; 8])[i / 64] |= 1 << (i % 64);
        }
        for (l, f) in flipped.into_iter().enumerate() {
            self.clear_checkerboards(l, f);
        }
        cells_of.len()
    }

    // ------------------------------------------------------------------ the surface

    /// The level and lattice point whose value level `l`'s point `i` takes in a view capped at `cap`: the finest level
    /// holding a block there.
    fn canonical(&self, l: usize, i: IVec3, v: &Show) -> (usize, IVec3) {
        let (mut l, mut i) = (l, i);
        while self.shows(v, l + 1, (i * 2).div_euclid(B3)) {
            l += 1;
            i *= 2;
        }
        (l, i)
    }

    /// Whether level `l`'s cell `q` is split by finer cells in a view capped at `cap`.
    fn covered(&self, l: usize, q: IVec3, v: &Show) -> bool {
        self.shows(v, l + 1, (q * 2).div_euclid(B3))
    }

    /// Whether the view `v` draws level `l`'s block `bc` (stored, not past the cap, not hidden).
    fn shows(&self, v: &Show, l: usize, bc: IVec3) -> bool {
        l <= v.cap && l < self.blocks.len() && self.blocks[l].contains_key(&bc) && !v.hidden.is_some_and(|h| h.contains(&(l as u8, bc)))
    }

    /// The smallest cell holding level `l`'s cell `q` (which must not be covered): its level and coordinates, if that
    /// level has a block there (level 0 cells of unstored blocks have none).
    fn leaf_down(&self, l: usize, q: IVec3, v: &Show) -> Option<(usize, IVec3)> {
        for m in (0..=l).rev() {
            let c = q.div_euclid(IVec3::splat(1 << (l - m)));
            if self.shows(v, m, c.div_euclid(B3)) {
                return Some((m, c));
            }
            if m == 0 {
                return None;
            }
        }
        None
    }

    /// The stored blocks whose 9x9x9 samples ([`Self::gather9`]) changed when the blocks `changed` changed (samples or
    /// existence): each of those still stored, the blocks just below it at its level (whose first layer past them is its
    /// samples), and the finer blocks near it (whose first layer past them may fall where their level has no block and
    /// so read its level, interpolated).
    pub fn upload_set(&self, changed: &[Key]) -> Vec<Key> {
        self.upload_set_in(changed, &Show::upto(self.levels() - 1))
    }

    /// [`Self::upload_set`] for a view `v` (whose samples are [`Self::gather9_hiding`] with its hidden blocks):
    /// `changed` may also name blocks it began or stopped showing. Only blocks it shows are listed.
    pub fn upload_set_in(&self, changed: &[Key], v: &Show) -> Vec<Key> {
        let levels = self.levels();
        // At each changed block's level: it and the seven below it, named once however many changed blocks name them.
        let span = |_l: usize, _m: usize, c: IVec3| (c - IVec3::ONE, c);
        let near = Self::near(changed, levels - 1, false, &span, 8, |&(l8, bc), prev, out| {
            if (l8 as usize) < levels {
                // After the block just below along x: only the four it did not name.
                let named = prev == Some(&(l8, bc - IVec3::X));
                for o in 0..8 {
                    if named && (o & 1) == 1 {
                        continue;
                    }
                    out.push((l8, bc - IVec3::new(o & 1, (o >> 1) & 1, o >> 2 & 1)));
                }
            }
        });
        let mut out: Vec<Key> = near.par_iter().copied().filter(|&(l, c)| self.shows(v, l as usize, c)).collect();
        // Finer blocks whose samples or first layer past them meet a changed block's box (grown by a sample for
        // interpolation), found through the nesting. Only a block with a missing neighbour above it reads its first
        // layer past it from coarser levels; one inside a finer region reads its neighbours, which this change leaves
        // alone.
        let finest = (0..levels).rev().find(|&m| self.level_len(m) > 0).unwrap_or(0);
        let per = self.per_level(changed, finest);
        let first = (0..finest).find(|&l| !per[l].is_empty());
        let shown: Vec<std::sync::OnceLock<Option<Named>>> = (0..levels).map(|_| std::sync::OnceLock::new()).collect();
        // A block among the eight below a changed block of its own level is listed above already.
        let named = |m: usize, c: IVec3| Self::any_in(&per[m], c, c + IVec3::ONE);
        for l in first.unwrap_or(finest)..finest {
            if per[l].is_empty() {
                continue;
            }
            // The box of changed block `bc`, in level `l` lattice units, is `bc*B-1 ..= bc*B+B+1`; a finer block `c`
            // (lattice `c*B ..= c*B+B`, at `s` finer steps a step) meets it when some changed block lies in this range.
            let meets = |c: IVec3, s: i32| {
                let d = B * s;
                let (lo, hi) = ((c * B - IVec3::splat(s + 1)).div_euclid(IVec3::splat(d)), (c * B + IVec3::splat(B + s)).div_euclid(IVec3::splat(d)));
                Self::any_in(&per[l], lo, hi)
            };
            // Of the level's blocks named, those next to a changed block of the level itself.
            let frontier: Vec<IVec3> = Self::at_level(&near, l as u8).par_iter().map(|k| k.1).filter(|c| meets(*c, 1)).collect();
            let edge = |_m: usize, c: IVec3, shows: &dyn Fn(IVec3) -> bool| (1..8).any(|o| !shows(c + IVec3::new(o & 1, (o >> 1) & 1, o >> 2 & 1)));
            self.descend(v, levels - 1, l, frontier, &meets, &named, &edge, &shown, &mut out);
        }
        Self::unique(out)
    }

    /// The blocks to re-extract when the blocks `changed` changed (samples or existence), in a view capped at `cap`:
    /// every stored block whose cells read those samples, whose cells a change in which blocks exist splits or joins,
    /// or whose quads point at their vertices. That is, at every level, the stored blocks meeting a changed block's box
    /// grown by two of its cells.
    pub fn extract_set(&self, changed: &[Key], cap: usize) -> Vec<Key> {
        self.extract_set_in(changed, &Show::upto(cap))
    }

    /// [`Self::extract_set`] in the view `v`: `changed` may also name blocks the view began or stopped showing.
    pub fn extract_set_in(&self, changed: &[Key], v: &Show) -> Vec<Key> {
        self.extract_set_where(changed, v, &|_| true)
    }

    /// [`Self::extract_set_in`], listing only the blocks `keep` passes (those it fails are left out as they are found,
    /// which costs less than taking them out after).
    pub fn extract_set_where(&self, changed: &[Key], v: &Show, keep: &(dyn Fn(Key) -> bool + Sync)) -> Vec<Key> {
        let cap = v.cap.min(self.levels() - 1);
        // At each changed block's level and the coarser ones: the few blocks meeting its box, named once however many
        // changed blocks name them (neighbouring changed blocks name the same ones over and over).
        // The blocks at level `m` meeting the box of level `l` block `c`.
        let span = |l: usize, m: usize, c: IVec3| {
            let s = 1i32 << (l - m);
            let (lo, hi) = (c * B - IVec3::splat(2), c * B + IVec3::splat(B + 2));
            (lo.div_euclid(IVec3::splat(s)).div_euclid(B3), (hi - IVec3::ONE).div_euclid(IVec3::splat(s)).div_euclid(B3))
        };
        let near = Self::near(changed, cap, true, &span, 36, |&(l8, bc), prev, out| {
            let l = l8 as usize;
            if l > cap {
                return;
            }
            // After the block just below along x: only the blocks past what it named.
            let after = (prev == Some(&(l8, bc - IVec3::X))).then_some(bc - IVec3::X);
            for m in 0..=l {
                let (a, b) = span(l, m, bc);
                let x0 = after.map_or(a.x, |p| a.x.max(span(l, m, p).1.x + 1));
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in x0..=b.x {
                            out.push((m as u8, IVec3::new(x, y, z)));
                        }
                    }
                }
            }
        });
        let mut out: Vec<Key> = near.par_iter().copied().filter(|&(m, c)| self.shows(v, m as usize, c) && keep((m, c))).collect();
        // Finer levels: shown descendants of the boxes, found through the nesting. Only a block at the edge of its
        // region (a neighbour missing) reads coarser samples or points its quads at coarser cells; one inside it is
        // untouched by a coarser change.
        let finest = (0..=cap).rev().find(|&m| self.level_len(m) > 0).unwrap_or(0);
        let per = self.per_level(changed, finest);
        let first = (0..finest).find(|&l| !per[l].is_empty());
        let shown: Vec<std::sync::OnceLock<Option<Named>>> = (0..=cap).map(|_| std::sync::OnceLock::new()).collect();
        // A block next to a changed block of its own level is listed above already.
        let named = |m: usize, c: IVec3| Self::any_in(&per[m], c - IVec3::ONE, c + IVec3::ONE);
        for l in first.unwrap_or(finest)..finest {
            if per[l].is_empty() {
                continue;
            }
            // The box of changed block `bc`, in level `l` lattice units, is `bc*B-2 .. bc*B+B+2`; a finer block `c`
            // (lattice `c*B .. c*B+B`, at `s` finer steps a step) meets it when some changed block lies in this range.
            let meets = |c: IVec3, s: i32| {
                let d = B * s;
                let (lo, hi) = ((c * B - IVec3::splat((B + 2) * s)).div_euclid(IVec3::splat(d)) + IVec3::ONE, (c * B + IVec3::splat(B + 2 * s - 1)).div_euclid(IVec3::splat(d)));
                Self::any_in(&per[l], lo, hi)
            };
            // Of the level's blocks named, those next to a changed block of the level itself.
            let frontier: Vec<IVec3> = Self::at_level(&near, l as u8).par_iter().map(|k| k.1).filter(|c| meets(*c, 1)).collect();
            let edge = |m: usize, c: IVec3, shows: &dyn Fn(IVec3) -> bool| keep((m as u8, c)) && (0..27).any(|q| q != 13 && !shows(c + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1)));
            self.descend(v, cap, l, frontier, &meets, &named, &edge, &shown, &mut out);
        }
        Self::unique(out)
    }

    /// A key's place in the order (level, then z, y, x), as one number: quicker to compare than the tuple.
    #[inline]
    fn order(k: &Key) -> u128 {
        let c = |v: i32| (v as u32 as u128) ^ (1 << 31);
        ((k.0 as u128) << 96) | (c(k.1.z) << 64) | (c(k.1.y) << 32) | c(k.1.x)
    }

    /// `keys` in order, each once (sorted over the threads).
    fn unique(mut keys: Vec<Key>) -> Vec<Key> {
        keys.par_sort_unstable_by_key(Self::order);
        keys.dedup();
        keys
    }

    /// The blocks `span` (changed level, named level, changed block: the box of blocks named) names for the blocks of
    /// `changed` up to `cap`, at their own level (and the coarser ones, with `coarser`), in order and each once. Where
    /// a changed level's blocks lie in a box of fair size they are marked as bits over it, over the threads, and read
    /// off in order; else what `each` names is gathered ([`Self::unique_chunked`]).
    fn near(changed: &[Key], cap: usize, coarser: bool, span: &(dyn Fn(usize, usize, IVec3) -> (IVec3, IVec3) + Sync), per: usize, each: impl Fn(&Key, Option<&Key>, &mut Vec<Key>) + Sync) -> Vec<Key> {
        use std::sync::atomic::{AtomicU64, Ordering};
        #[derive(Clone, Copy)]
        struct Grid {
            lo: IVec3,
            n: IVec3,
        }
        let index = |g: &Grid, c: IVec3| {
            let d = c - g.lo;
            ((d.z * g.n.y + d.y) * g.n.x + d.x) as usize
        };
        // The box of the changed blocks at each level.
        let mut boxes: Vec<Option<(IVec3, IVec3)>> = vec![None; cap + 1];
        for &(l, bc) in changed {
            if (l as usize) <= cap {
                let e = &mut boxes[l as usize];
                *e = Some(e.map_or((bc, bc), |(a, b)| (a.min(bc), b.max(bc))));
            }
        }
        let levels: Vec<usize> = (0..=cap).filter(|&l| boxes[l].is_some()).collect();
        // Per changed level: bits over the box it names at its own level, marked by every thread (neighbouring blocks
        // mostly fall to one thread), and at each coarser level, whose cells many blocks over the threads share, so
        // each thread marks a copy of its own, joined after.
        let mut same: Vec<Option<(Grid, Vec<AtomicU64>)>> = (0..=cap).map(|_| None).collect();
        let mut coarse: Vec<(usize, usize, Grid)> = Vec::new();
        let mut blank: Vec<Vec<u64>> = Vec::new();
        for &l in &levels {
            let (lo, hi) = boxes[l].unwrap();
            for m in if coarser { 0..=l } else { l..=l } {
                let (a, b) = (span(l, m, lo).0, span(l, m, hi).1);
                let n = b - a + IVec3::ONE;
                let cells = n.x as i64 * n.y as i64 * n.z as i64;
                if cells > Named::MOST_BITS {
                    return Self::unique_chunked(changed, per, each);
                }
                let words = ((cells + 63) / 64) as usize;
                let g = Grid { lo: a, n };
                if m == l {
                    same[l] = Some((g, (0..words).map(|_| AtomicU64::new(0)).collect()));
                } else {
                    coarse.push((l, m, g));
                    blank.push(vec![0u64; words]);
                }
            }
        }
        let mark = |mine: &mut Vec<Vec<u64>>, l: usize, bc: IVec3| {
            if let Some((g, bits)) = &same[l] {
                let (a, b) = span(l, l, bc);
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            let i = index(g, IVec3::new(x, y, z));
                            let (w, bit) = (&bits[i >> 6], 1u64 << (i & 63));
                            if w.load(Ordering::Relaxed) & bit == 0 {
                                w.fetch_or(bit, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
            for (ci, (cl, m, g)) in coarse.iter().enumerate() {
                if *cl != l {
                    continue;
                }
                let (a, b) = span(l, *m, bc);
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            let i = index(g, IVec3::new(x, y, z));
                            mine[ci][i >> 6] |= 1 << (i & 63);
                        }
                    }
                }
            }
        };
        let merged: Vec<Vec<u64>> = changed
            .par_chunks(4096)
            .fold(
                || blank.clone(),
                |mut mine, chunk| {
                    for &(l8, bc) in chunk {
                        if (l8 as usize) <= cap {
                            mark(&mut mine, l8 as usize, bc);
                        }
                    }
                    mine
                },
            )
            .reduce(
                || blank.clone(),
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        for (p, q) in x.iter_mut().zip(y) {
                            *p |= *q;
                        }
                    }
                    a
                },
            );
        // Read off level by level, each grid's bits in z, y, x order.
        let read = |g: &Grid, words: &mut dyn Iterator<Item = u64>, m: usize, out: &mut Vec<Key>| {
            for (w, mut bits) in words.enumerate() {
                while bits != 0 {
                    let i = w * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let (x, rest) = (i as i32 % g.n.x, i as i32 / g.n.x);
                    out.push((m as u8, g.lo + IVec3::new(x, rest % g.n.y, rest / g.n.y)));
                }
            }
        };
        let mut out: Vec<Key> = Vec::new();
        for m in 0..=cap {
            for &l in &levels {
                if m == l {
                    if let Some((g, bits)) = &same[l] {
                        read(g, &mut bits.iter().map(|w| w.load(Ordering::Relaxed)), m, &mut out);
                    }
                } else if let Some(ci) = coarse.iter().position(|(cl, cm, _)| *cl == l && *cm == m) {
                    read(&coarse[ci].2, &mut merged[ci].iter().copied(), m, &mut out);
                }
            }
        }
        // Grids of one level from several changed levels may overlap.
        if levels.len() > 1 { Self::unique(out) } else { out }
    }

    /// What `each` names for the keys of `changed` (about `per` a key), in order and each once: a chunk of keys at a
    /// time on each thread, the repeats within a chunk dropped (as a set) before the chunks are merged.
    fn unique_chunked(changed: &[Key], per: usize, each: impl Fn(&Key, Option<&Key>, &mut Vec<Key>) + Sync) -> Vec<Key> {
        let chunks: Vec<Vec<Key>> = changed
            .par_chunks(1024)
            .map(|chunk| {
                let mut out = Vec::with_capacity(per * 4);
                let mut seen: HashSet<Key> = HashSet::default();
                seen.reserve(chunk.len() * per / 4);
                let mut prev: Option<&Key> = None;
                for k in chunk {
                    out.clear();
                    each(k, prev, &mut out);
                    seen.extend(out.iter().copied());
                    prev = Some(k);
                }
                seen.into_iter().collect()
            })
            .collect();
        Self::unique(chunks.concat())
    }

    /// The keys of `sorted` (as [`Self::unique`] orders them) at level `l`.
    fn at_level(sorted: &[Key], l: u8) -> &[Key] {
        let a = sorted.partition_point(|k| k.0 < l);
        let b = a + sorted[a..].partition_point(|k| k.0 == l);
        &sorted[a..b]
    }

    /// The blocks of `changed` at each level up to `cap`.
    fn per_level(&self, changed: &[Key], cap: usize) -> Vec<Named> {
        (0..=cap).map(|l| Named::of(changed.iter().filter(|k| k.0 as usize == l).map(|k| k.1))).collect()
    }

    /// Whether `set` holds a block in the box `lo..=hi`.
    fn any_in(set: &Named, lo: IVec3, hi: IVec3) -> bool {
        (lo.z..=hi.z).any(|z| (lo.y..=hi.y).any(|y| (lo.x..=hi.x).any(|x| set.contains(IVec3::new(x, y, z)))))
    }

    /// The shown blocks of the levels finer than `l` up to `cap` under the level `l` blocks `frontier` that `meets`
    /// (a block, its steps per level `l` step) passes, level by level; those `edge` (given the level, the block and
    /// whether a block of the level shows) passes are added to `out`, unless `named` (level, block) says the block
    /// is listed already by its own level's changed blocks.
    fn descend(&self, v: &Show, cap: usize, l: usize, mut frontier: Vec<IVec3>, meets: &(dyn Fn(IVec3, i32) -> bool + Sync), named: &(dyn Fn(usize, IVec3) -> bool + Sync), edge: &(dyn Fn(usize, IVec3, &dyn Fn(IVec3) -> bool) -> bool + Sync), shown: &[std::sync::OnceLock<Option<Named>>], out: &mut Vec<Key>) {
        for m in l + 1..=cap {
            // Every block has its parent, so a level without blocks has none finer.
            if frontier.is_empty() || self.level_len(m) == 0 {
                return;
            }
            let s = 1i32 << (m - l);
            // The level's shown blocks as bits (made when the level is first reached): each block found looks up the
            // many around it.
            let bits = shown[m].get_or_init(|| self.shown_bits(m, v));
            let shows_m = |c: IVec3| match bits {
                Some(b) => b.contains(c),
                None => self.shows(v, m, c),
            };
            // Children of distinct blocks are distinct: nothing to merge.
            let next: Vec<(IVec3, bool)> = frontier
                .par_iter()
                .flat_map_iter(|p| {
                    (0..8).filter_map(|q| {
                        let c = *p * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                        (shows_m(c) && meets(c, s)).then(|| (c, !named(m, c) && edge(m, c, &shows_m)))
                    })
                })
                .collect();
            out.extend(next.iter().filter(|x| x.1).map(|x| (m as u8, x.0)));
            frontier = next.into_iter().map(|x| x.0).collect();
        }
    }

    /// Level `m`'s blocks the view shows, as bits over their box; none when that is too spread out. A level can hold
    /// hundreds of thousands of blocks: its box and its bits are worked out over the threads, a shard each.
    fn shown_bits(&self, m: usize, v: &Show) -> Option<Named> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let map = &self.blocks[m];
        if map.is_empty() || m > v.cap {
            return None;
        }
        let shards = map.shards();
        let (lo, hi) = shards.par_iter().map(|s| s.keys().fold((IVec3::MAX, IVec3::MIN), |(lo, hi), c| (lo.min(*c), hi.max(*c)))).reduce(|| (IVec3::MAX, IVec3::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        let n = hi - lo + IVec3::ONE;
        if n.x as i64 * n.y as i64 * n.z as i64 > Named::MOST_BITS {
            return None;
        }
        let words = ((n.x as i64 * n.y as i64 * n.z as i64 + 63) / 64) as usize;
        let bits: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();
        shards.par_iter().for_each(|s| {
            for c in s.keys() {
                let d = *c - lo;
                let i = ((d.z * n.y + d.y) * n.x + d.x) as usize;
                bits[i >> 6].fetch_or(1 << (i & 63), Ordering::Relaxed);
            }
        });
        // All of them, less the hidden ones (few, against looking each block up).
        let mut named = Named::Bits { lo, n, bits: bits.into_iter().map(|b| b.into_inner()).collect() };
        if let Some(hidden) = v.hidden {
            for k in hidden {
                if k.0 as usize == m {
                    named.remove(k.1);
                }
            }
        }
        Some(named)
    }

    /// [`Self::upload_set_in`] worked out plainly: the reference the chunked one is tested against. At each changed
    /// block's level, it and the seven blocks below it; and the finer blocks under those, level by level, whose box
    /// meets the box of any changed block of that level (grown by a sample), listed where a block above them is missing.
    /// (Descending from each changed block's own blocks alone, as this first did, missed a finer block under the
    /// blocks below one changed block that meets the box of another next to it, which the chunked one finds.)
    #[doc(hidden)]
    pub fn upload_set_in_reference(&self, changed: &[Key], v: &Show) -> Vec<Key> {
        let mut out: Vec<Key> = Vec::new();
        for l in 0..self.levels() {
            let at_l: Vec<IVec3> = changed.iter().filter(|k| k.0 as usize == l).map(|k| k.1).collect();
            if at_l.is_empty() {
                continue;
            }
            let mut frontier: Vec<IVec3> = at_l.iter().flat_map(|&bc| (0..8).map(move |o| bc - IVec3::new(o & 1, (o >> 1) & 1, o >> 2 & 1))).collect();
            frontier.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            frontier.dedup();
            out.extend(frontier.iter().filter(|&&n| self.shows(v, l, n)).map(|&n| (l as u8, n)));
            for m in l + 1..self.levels() {
                let s = 1i32 << (m - l);
                let meets = |c: IVec3| {
                    let (clo, chi) = (c * B, c * B + B3);
                    at_l.iter().any(|&bc| {
                        let (flo, fhi) = ((bc * B - IVec3::ONE) * s, (bc * B + IVec3::splat(B + 1)) * s);
                        !(chi.cmplt(flo).any() || clo.cmpgt(fhi).any())
                    })
                };
                let mut next = Vec::new();
                for p in &frontier {
                    for q in 0..8 {
                        let c = *p * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                        if !self.shows(v, m, c) || !meets(c) {
                            continue;
                        }
                        if (1..8).any(|o| !self.shows(v, m, c + IVec3::new(o & 1, (o >> 1) & 1, o >> 2 & 1))) {
                            out.push((m as u8, c));
                        }
                        next.push(c);
                    }
                }
                if next.is_empty() {
                    break;
                }
                next.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                next.dedup();
                frontier = next;
            }
        }
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out.dedup();
        out
    }

    /// [`Self::extract_set_where`] a changed block at a time, as it was first written: the reference the chunked one
    /// is tested against.
    #[doc(hidden)]
    pub fn extract_set_where_reference(&self, changed: &[Key], v: &Show, keep: &(dyn Fn(Key) -> bool + Sync)) -> Vec<Key> {
        let cap = v.cap.min(self.levels() - 1);
        let mut v: Vec<Key> = changed.par_iter().flat_map_iter(|&(l, bc)| {
            let mut out: Vec<Key> = Vec::new();
            let l = l as usize;
            if l > cap {
                return out;
            }
            let lo = bc * B - IVec3::splat(2);
            let hi = bc * B + IVec3::splat(B + 2);
            for m in 0..=l {
                let s = 1i32 << (l - m);
                let (a, b) = (lo.div_euclid(IVec3::splat(s)).div_euclid(B3), (hi - IVec3::ONE).div_euclid(IVec3::splat(s)).div_euclid(B3));
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            let c = IVec3::new(x, y, z);
                            if self.shows(v, m, c) && keep((m as u8, c)) {
                                out.push((m as u8, c));
                            }
                        }
                    }
                }
            }
            let mut frontier: Vec<IVec3> = Vec::new();
            {
                let (a, b) = (lo.div_euclid(B3), (hi - IVec3::ONE).div_euclid(B3));
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            frontier.push(IVec3::new(x, y, z));
                        }
                    }
                }
            }
            for m in l + 1..=cap {
                let s = 1i32 << (m - l);
                let (flo, fhi) = (lo * s, hi * s);
                let mut next = Vec::new();
                for p in &frontier {
                    for q in 0..8 {
                        let c = *p * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                        let (clo, chi) = (c * B, c * B + B3);
                        if chi.cmple(flo).any() || clo.cmpge(fhi).any() {
                            continue;
                        }
                        if self.shows(v, m, c) {
                            if keep((m as u8, c)) && (0..27).any(|q| q != 13 && !self.shows(v, m, c + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1))) {
                                out.push((m as u8, c));
                            }
                            next.push(c);
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
            out
        }).collect();
        v.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        v.dedup();
        v
    }

    /// The surface of `work` (stored blocks), in a view capped at `cap`, by the rules the GPU extractor follows:
    ///
    /// * The cells of a block are its level's; a cell covered by a finer block is not a leaf and has no vertex. A
    ///   lattice point takes the value of the finest level holding a block there (read from the block's own samples or,
    ///   for the first layer past it, its level's field there).
    /// * A leaf cell that a finer cell touches (among its 26 neighbours) is a transition cell: it has exactly one vertex,
    ///   at the mean of the crossings on its faces split once, or failing that, near its centre. Any other cell has a
    ///   vertex per surface piece ([`piece_table`]), at the mean of that piece's crossings.
    /// * A lattice edge whose ends differ in sign gives a quad when it is the smallest edge there: none of the four cells
    ///   around it is finer than it. The first of those cells at the edge's level (in the order: the cell starting at
    ///   the edge's start, then the one below along the second axis, then below along the first, then below along both)
    ///   owns it; where two corners are one coarser cell the quad is a triangle.
    pub fn extract(&self, work: &[Key], cap: usize) -> (HashMap<Key, BlockSurface>, ExtractStats) {
        self.extract_in(work, &Show::upto(cap))
    }

    /// [`Self::extract`] in the view `v`: the blocks of `work` it shows.
    pub fn extract_in(&self, work: &[Key], v: &Show) -> (HashMap<Key, BlockSurface>, ExtractStats) {
        let t = Instant::now();
        let v = &Show { cap: v.cap.min(self.levels() - 1), hidden: v.hidden };
        let table = piece_table();
        let mut stats = ExtractStats::default();
        let mut surf: HashMap<Key, BlockSurface> = work.par_iter().filter(|k| self.shows(v, k.0 as usize, k.1)).map(|&k| (k, self.block_vertices(k, v, &table))).collect();
        // Quads read the vertices of blocks outside `work` too: extract those (vertices only) on demand.
        let mut extra: HashMap<Key, BlockSurface> = HashMap::default();
        let quads: Vec<(Key, Vec<[Corner; 4]>, usize)> = {
            let need: HashSet<Key> = work
                .par_iter()
                .filter(|k| surf.contains_key(k))
                .flat_map_iter(|&k| self.quad_neighbours(k, v))
                .filter(|k| !surf.contains_key(k))
                .collect();
            extra.extend(need.into_par_iter().map(|k| (k, self.block_vertices(k, v, &table))).collect::<Vec<_>>());
            let all = |k: &Key| surf.get(k).or(extra.get(k));
            work.par_iter().filter(|k| surf.contains_key(k)).map(|&k| {
                let (q, dropped) = self.block_quads(k, v, &table, &all);
                (k, q, dropped)
            }).collect()
        };
        for (k, q, dropped) in quads {
            stats.dropped += dropped;
            for quad in &q {
                if quad[0] == quad[1] || quad[1] == quad[2] || quad[2] == quad[3] || quad[3] == quad[0] {
                    stats.triangles += 1;
                } else {
                    stats.quads += 1;
                }
            }
            surf.get_mut(&k).unwrap().quads = q;
        }
        stats.blocks = surf.len();
        stats.verts = surf.values().map(|s| s.verts.len()).sum();
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        (surf, stats)
    }

    /// Blocks whose vertices block `k`'s quads can point at: the stored blocks around it at its level and coarser.
    fn quad_neighbours(&self, k: Key, v: &Show) -> Vec<Key> {
        let (l, bc) = (k.0 as usize, k.1);
        let mut out = Vec::new();
        let lo = bc * B - IVec3::ONE;
        let hi = bc * B + B3;
        for m in 0..=l.min(v.cap) {
            let s = 1i32 << (l - m);
            let (a, b) = (lo.div_euclid(IVec3::splat(s)).div_euclid(B3), hi.div_euclid(IVec3::splat(s)).div_euclid(B3));
            for z in a.z..=b.z {
                for y in a.y..=b.y {
                    for x in a.x..=b.x {
                        let c = IVec3::new(x, y, z);
                        if self.shows(v, m, c) {
                            out.push((m as u8, c));
                        }
                    }
                }
            }
        }
        out
    }

    /// The value at level `l`'s point `i` for a cell of block `bc` (whose 9x9x9 samples are `g`).
    fn corner_value(&self, l: usize, bc: IVec3, g: &[f32; 729], i: IVec3, v: &Show) -> f32 {
        let (m, j) = self.canonical(l, i, v);
        if m > l {
            return as_held(self.blocks[m][&j.div_euclid(B3)][local(j)], self.band(m));
        }
        let d = i - bc * B;
        g[(d.z * 81 + d.y * 9 + d.x) as usize]
    }

    fn block_vertices(&self, k: Key, v: &Show, table: &[u32; 256]) -> BlockSurface {
        let (l, bc) = (k.0 as usize, k.1);
        // Every value as the GPU's 16-bit samples hold it, so that the surface here is the GPU's to the last bit of
        // arithmetic: a value interpolated from full-precision samples can land on the other side of zero.
        let mut g = self.gather9_in(l, bc, v);
        let (held_band, reach) = (self.band(l), self.sent_reach(l));
        for v in g.iter_mut() {
            *v = as_held(v.clamp(-reach, reach), held_band);
        }
        let (vx, band) = (self.voxel(l), self.band(l));
        let base = bc * B;
        let mut out = BlockSurface { cells: vec![0; N], ..Default::default() };
        // Which level `l` cells around the block are split by finer ones: a 10x10x10 neighbourhood.
        let mut finer = [false; 1000];
        if l < v.cap {
            for z in -1..9 {
                for y in -1..9 {
                    for x in -1..9 {
                        let q = base + IVec3::new(x, y, z);
                        finer[((z + 1) * 100 + (y + 1) * 10 + (x + 1)) as usize] = self.covered(l, q, v);
                    }
                }
            }
        }
        let fin = |o: IVec3| finer[((o.z + 1) * 100 + (o.y + 1) * 10 + (o.x + 1)) as usize];
        // Vertices so far, counting those of cells past the budget: once one cell does not fit, no later one does.
        let mut total = 0usize;
        for i in 0..N {
            let c = cell_of(i);
            if fin(c) {
                continue;
            }
            // Within the band, as the GPU's 16-bit samples hold them: mid-stroke, fresh clay can lie past it.
            let d: [f32; 8] = std::array::from_fn(|q| self.corner_value(l, bc, &g, base + c + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32), v).clamp(-band, band));
            let config = (0..8).fold(0u32, |a, q| a | ((d[q] < 0.0) as u32) << q);
            let transition = l < v.cap && (0..27).any(|q| q != 13 && fin(c + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1)));
            let corner_pos = |q: usize| Vec3::new(corner(q)[0] as f32, corner(q)[1] as f32, corner(q)[2] as f32);
            // Where in the cell each of its vertices lies.
            let mut at: Vec<Vec3> = Vec::with_capacity(4);
            if transition {
                // Crossings on the cell's faces split once: values at the half-step points from the finer level where it
                // holds them, else interpolated from the corners.
                let mut sum = Vec3::ZERO;
                let mut n = 0.0;
                let half = |h: IVec3| -> f32 {
                    // `h` in half-cell units (0..2 per axis).
                    let p = (base + c) * 2 + h;
                    let (m, j) = self.canonical(l + 1, p, v);
                    if m > l + 1 || self.shows(v, l + 1, p.div_euclid(B3)) {
                        let j = if m > l + 1 { j } else { p };
                        // As the 16-bit samples hold it (within the band: mid-stroke, carried detail can lie past it).
                        let m = m.max(l + 1);
                        return as_held(self.blocks[m][&j.div_euclid(B3)][local(j)], self.band(m));
                    }
                    trilinear(h.as_vec3() * 0.5, d)
                };
                for axis in 0..3 {
                    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                    // Half-edges along `axis` on the cell's boundary: those with u or w at 0 or 2.
                    for a in 0..3 {
                        for b in 0..3 {
                            if a == 1 && b == 1 {
                                continue;
                            }
                            for s in 0..2 {
                                let mut p0 = IVec3::ZERO;
                                p0[u] = a;
                                p0[w] = b;
                                p0[axis] = s;
                                let mut p1 = p0;
                                p1[axis] += 1;
                                let (va, vb) = (half(p0), half(p1));
                                if (va < 0.0) != (vb < 0.0) {
                                    sum += (p0.as_vec3() + (p1 - p0).as_vec3() * (va / (va - vb))) * 0.5;
                                    n += 1.0;
                                }
                            }
                        }
                    }
                }
                let f = if n > 0.0 {
                    sum / n
                } else {
                    // Near the centre, moved onto the field's zero along its gradient: if there is one to speak of and
                    // the zero lies within the cell. Deep in fresh clay the corners all sit at the band, and the
                    // gradient left is rounding, which would fling the vertex anywhere.
                    let m = Vec3::splat(0.5);
                    let gr = trilinear_gradient(m, d);
                    let step = gr * trilinear(m, d) / gr.length_squared().max(1e-20);
                    if gr.length_squared() > (0.05 * vx) * (0.05 * vx) && step.length_squared() <= 1.0 { m - step } else { m }
                };
                at.push(f);
            } else {
                if config == 0 || config == 255 {
                    continue;
                }
                let entry = table[config as usize];
                let pieces = (entry >> 24) as usize;
                let mut sum = [Vec3::ZERO; 4];
                let mut cnt = [0.0f32; 4];
                for (e, [a, b]) in EDGES.iter().enumerate() {
                    if (d[*a] < 0.0) == (d[*b] < 0.0) {
                        continue;
                    }
                    let p = ((entry >> (2 * e)) & 3) as usize;
                    sum[p] += corner_pos(*a).lerp(corner_pos(*b), d[*a] / (d[*a] - d[*b]));
                    cnt[p] += 1.0;
                }
                for p in 0..pieces {
                    at.push(sum[p] / cnt[p].max(1.0));
                }
            }
            let first = total;
            total += at.len();
            if first + at.len() > VERTS_PER_BLOCK {
                continue;
            }
            for f in &at {
                let f = f.clamp(Vec3::ZERO, Vec3::ONE);
                out.verts.push(self.origin + (base + c).as_vec3() * vx + f * vx);
                out.normals.push(trilinear_gradient(f, d));
            }
            out.cells[i] = first as u32 | (at.len() as u32) << 10 | config << 13 | (transition as u32) << 21;
        }
        out
    }

    /// The corner a quad around edge `edge` (cell `(m, q)`'s cube edge, when the cell is at the quad's level) takes at
    /// that cell, if the cell has a vertex.
    fn quad_corner<'a>(&self, m: usize, q: IVec3, edge: Option<usize>, table: &[u32; 256], surf: &impl Fn(&Key) -> Option<&'a BlockSurface>) -> Option<Corner> {
        let key = (m as u8, q.div_euclid(B3));
        let s = surf(&key)?;
        let info = s.cells[local(q)];
        if (info >> 10) & 7 == 0 {
            return None;
        }
        let piece = match edge {
            Some(e) if info >> 21 & 1 == 0 => (table[((info >> 13) & 255) as usize] >> (2 * e)) & 3,
            _ => 0,
        };
        Some((key, local(q) as u16, piece as u8))
    }

    fn block_quads<'a>(&self, k: Key, v: &Show, table: &[u32; 256], surf: &impl Fn(&Key) -> Option<&'a BlockSurface>) -> (Vec<[Corner; 4]>, usize) {
        let (l, bc) = (k.0 as usize, k.1);
        let g = self.gather9_in(l, bc, v);
        let base = bc * B;
        let mut out = Vec::new();
        let mut dropped = 0;
        let Some(me) = surf(&k) else { return (out, 0) };
        for i in 0..N {
            let c = base + cell_of(i);
            if self.covered(l, c, v) {
                continue;
            }
            // Its crossing edges have both ends at its corners: none when they are all one sign.
            let config = (me.cells[i] >> 13) & 255;
            if config == 0 || config == 255 {
                continue;
            }
            for [a0, b0] in EDGES.iter() {
                let (ca, cb) = (corner(*a0), corner(*b0));
                let axis = (0..3).find(|&x| ca[x] != cb[x]).unwrap();
                let start = c + IVec3::new(ca[0] as i32, ca[1] as i32, ca[2] as i32);
                let mut end = start;
                end[axis] += 1;
                let va = self.corner_value(l, bc, &g, start, v);
                let vb = self.corner_value(l, bc, &g, end, v);
                if (va < 0.0) == (vb < 0.0) {
                    continue;
                }
                let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                let mut eu = IVec3::ZERO;
                eu[u] = 1;
                let mut ev = IVec3::ZERO;
                ev[w] = 1;
                // The four cells around the edge, in quad order, and the corner each starts the edge at.
                let around = [start - eu - ev, start - ev, start, start - eu];
                if around.iter().any(|&q| self.covered(l, q, v)) {
                    continue;
                }
                let level_of = |q: IVec3| self.leaf_down(l, q, v).map(|(m, _)| m);
                // The owner: the first at this level in the order start, below along v, below along u, below along both.
                let owner = [2usize, 1, 3, 0].into_iter().find(|&j| level_of(around[j]) == Some(l));
                if owner.map(|j| around[j]) != Some(c) {
                    continue;
                }
                let corners: Option<Vec<Corner>> = around
                    .iter()
                    .map(|&q| {
                        let (m, qc) = self.leaf_down(l, q, v)?;
                        let edge = (m == l).then(|| {
                            let lo = start - q;
                            let k0 = (lo.x | lo.y << 1 | lo.z << 2) as usize;
                            EDGE_OF[k0][k0 | 1 << axis] as usize
                        });
                        self.quad_corner(m, qc, edge, table, surf)
                    })
                    .collect();
                let Some(cs) = corners else {
                    dropped += 1;
                    continue;
                };
                out.push(if va < 0.0 { [cs[0], cs[1], cs[2], cs[3]] } else { [cs[0], cs[3], cs[2], cs[1]] });
            }
        }
        (out, dropped)
    }

    /// The whole surface (levels up to `cap`) as one mesh, for pictures, checks and export.
    pub fn surface(&self, cap: usize) -> (PolyMesh, ExtractStats) {
        self.surface_in(&Show::upto(cap))
    }

    /// [`Self::surface`] in the view `v`.
    pub fn surface_in(&self, v: &Show) -> (PolyMesh, ExtractStats) {
        let work: Vec<Key> = self.keys();
        let (surf, stats) = self.extract_in(&work, v);
        let mut first: HashMap<Key, u32> = HashMap::default();
        let mut positions = Vec::new();
        let mut keys: Vec<&Key> = surf.keys().collect();
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        for k in &keys {
            first.insert(**k, positions.len() as u32);
            positions.extend_from_slice(&surf[*k].verts);
        }
        let mut faces = Vec::new();
        for k in &keys {
            for q in &surf[*k].quads {
                let Some(ids) = q.iter().map(|(b, c, p)| Some(first[b] + surf[b].vertex(*c, *p)? as u32)).collect::<Option<Vec<u32>>>() else { continue };
                let mut uniq: Vec<u32> = Vec::with_capacity(4);
                for v in ids {
                    if uniq.last() != Some(&v) && uniq.first() != Some(&v) || uniq.is_empty() {
                        uniq.push(v);
                    }
                }
                match uniq.len() {
                    4 => faces.push([uniq[0], uniq[1], uniq[2], uniq[3]]),
                    3 => faces.push([uniq[0], uniq[1], uniq[2], NO_VERT]),
                    _ => {}
                }
            }
        }
        (PolyMesh { positions, faces }, stats)
    }

    // ------------------------------------------------------------------ files

    /// Write the field (not the history) to `w`: a header, the level 0 blocks wholly inside, then each level's stored
    /// blocks with their samples at 16 bits.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"SVOL")?;
        // Version 3: level 0 holds distances out to its band (version 2, six steps).
        w.write_all(&3u32.to_le_bytes())?;
        for f in [self.voxel0, self.origin.x, self.origin.y, self.origin.z] {
            w.write_all(&f.to_le_bytes())?;
        }
        w.write_all(&(self.levels() as u32).to_le_bytes())?;
        let mut inside: Vec<IVec3> = self.inside0.iter().copied().collect();
        inside.sort_unstable_by_key(|v| (v.z, v.y, v.x));
        w.write_all(&(inside.len() as u32).to_le_bytes())?;
        for c in inside {
            for x in c.to_array() {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        let mut buf = Vec::with_capacity(N * 2);
        for l in 0..self.levels() {
            let band = self.band(l);
            let mut keys: Vec<IVec3> = self.blocks[l].keys().copied().collect();
            keys.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            w.write_all(&(keys.len() as u32).to_le_bytes())?;
            for bc in keys {
                for x in bc.to_array() {
                    w.write_all(&x.to_le_bytes())?;
                }
                buf.clear();
                for &v in self.blocks[l][&bc].iter() {
                    buf.extend_from_slice(&quantize(v, band).to_le_bytes());
                }
                w.write_all(&buf)?;
            }
        }
        Ok(())
    }

    /// Read a volume written by [`Self::write_to`]. A single-resolution file (version 1, `Volume::write_to`) becomes
    /// level 0, with `levels` levels in all.
    pub fn read_from(r: &mut impl std::io::Read, levels: usize) -> std::io::Result<LevelVolume> {
        use std::io::{Error, ErrorKind};
        let bad = |m: &str| Error::new(ErrorKind::InvalidData, m.to_string());
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        if &b4 != b"SVOL" {
            return Err(bad("not a volume file"));
        }
        let u32_ = |r: &mut dyn std::io::Read| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let version = u32_(r)?;
        if version == 1 {
            // The rest of a version 1 file, read by the single-resolution volume.
            let mut rest = b"SVOL".to_vec();
            rest.extend_from_slice(&1u32.to_le_bytes());
            r.read_to_end(&mut rest)?;
            let old = super::volume::Volume::read_from(&mut rest.as_slice())?;
            return Ok(LevelVolume::from_volume(&old, levels));
        }
        if version != 2 && version != 3 {
            return Err(bad("unknown volume file version"));
        }
        let mut f = [0f32; 4];
        for v in &mut f {
            *v = f32::from_bits(u32_(r)?);
        }
        let n = u32_(r)? as usize;
        if n == 0 || n > MAX_LEVELS {
            return Err(bad("bad level count"));
        }
        let ivec = |r: &mut dyn std::io::Read| -> std::io::Result<IVec3> {
            let mut b = [0u8; 12];
            r.read_exact(&mut b)?;
            Ok(IVec3::new(i32::from_le_bytes(b[0..4].try_into().unwrap()), i32::from_le_bytes(b[4..8].try_into().unwrap()), i32::from_le_bytes(b[8..12].try_into().unwrap())))
        };
        let mut v = LevelVolume { voxel0: f[0], origin: Vec3::new(f[1], f[2], f[3]), blocks: vec![Shared::default(); n.max(levels.min(MAX_LEVELS))], inside0: SharedSet::default(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::default(), carry: Shared::default(), carry_push: Vec::new(), recent: HashSet::default(), carry_level: 0, pending: vec![HashSet::default(); MAX_LEVELS], moved: HashMap::default(), carried: vec![HashSet::default(); MAX_LEVELS], carry_budget: None, keep_detail: false, uncached_reads: false };
        for _ in 0..u32_(r)? {
            v.inside0.insert(ivec(r)?);
        }
        let mut buf = vec![0u8; N * 2];
        // A version 2 file's level 0 holds, past six steps from the surface, only that bound: it becomes the band, the
        // bound level 0 holds now.
        let old_bound = if version == 2 { quantize(Self::SENT0_STEPS * v.voxel0, v.band(0)) as u16 } else { u16::MAX };
        for l in 0..n {
            let band = v.band(l);
            for _ in 0..u32_(r)? {
                let bc = ivec(r)?;
                r.read_exact(&mut buf)?;
                let mut s = Box::new([0f32; N]);
                for (i, x) in s.iter_mut().enumerate() {
                    let q = i16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]);
                    *x = if l == 0 && q.unsigned_abs() == old_bound { (q.signum() as f32) * band } else { q as f32 / 32767.0 * band };
                }
                v.blocks[l].insert(bc, s);
            }
        }
        // Every block's parent must exist.
        for l in (1..n).rev() {
            let missing: Vec<IVec3> = v.blocks[l].keys().map(|bc| bc.div_euclid(IVec3::splat(2))).filter(|p| !v.blocks[l - 1].contains_key(p)).collect();
            if !missing.is_empty() {
                return Err(bad("a block's parent is missing"));
            }
        }
        v.dirty = v.keys().into_iter().collect();
        Ok(v)
    }

    /// A single-resolution volume as level 0 of a new one with `levels` levels.
    pub fn from_volume(old: &super::volume::Volume, levels: usize) -> LevelVolume {
        let (origin, dims) = old.grid();
        let mut v = LevelVolume { voxel0: old.voxel, origin, blocks: vec![Shared::default(); levels.clamp(1, MAX_LEVELS)], inside0: SharedSet::default(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::default(), carry: Shared::default(), carry_push: Vec::new(), recent: HashSet::default(), carry_level: 0, pending: vec![HashSet::default(); MAX_LEVELS], moved: HashMap::default(), carried: vec![HashSet::default(); MAX_LEVELS], carry_budget: None, keep_detail: false, uncached_reads: false };
        let band = v.band(0);
        let stored: HashMap<u32, &[f32; N]> = old.stored().collect();
        for (b, &sign) in old.signs().iter().enumerate() {
            let c = old.coords_of(b as u32);
            match sign {
                0 => {
                    let s = stored[&(b as u32)];
                    let mut out = Box::new([0f32; N]);
                    for (x, y) in out.iter_mut().zip(s.iter()) {
                        // The old band was narrower: beyond it, the old clamp stands.
                        *x = if y.abs() >= old.band * 0.999 { y.signum() * band } else { *y };
                    }
                    v.blocks[0].insert(c, out);
                }
                s if s < 0 => {
                    v.inside0.insert(c);
                }
                _ => {}
            }
        }
        let _ = dims;
        v.dirty = v.keys().into_iter().collect();
        v
    }
}

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

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::remesh::{EDGES, EDGE_OF, FACES, corner, find, union};
use crate::mesh::{NO_VERT, PolyMesh};

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

pub type Block = Box<[f32; N]>;
/// A block's level and coordinates (in blocks of that level).
pub type Key = (u8, IVec3);
/// One bit per sample.
type SignMask = [u64; 8];

/// A block as it was before a stroke changed it.
#[derive(Clone)]
enum Was {
    Missing,
    /// A level 0 block wholly inside the solid.
    Inside,
    Stored(Block),
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
pub fn quantize(v: f32, band: f32) -> i16 {
    let q = (v / band * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
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
                    for k in 0..4 {
                        if inside(face[k]) {
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
        for e in 0..12 {
            if !crossing[e] {
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

pub struct LevelVolume {
    pub voxel0: f32,
    origin: Vec3,
    blocks: Vec<HashMap<IVec3, Block>>,
    /// Level 0 blocks wholly inside the solid (any other level 0 block not stored is wholly outside).
    inside0: HashSet<IVec3>,
    /// While a stroke is under way: each block as it was before the stroke first changed it. Dabs are placed on the
    /// surface as it was when the stroke began, and the stroke is undone from it.
    stroke: Option<HashMap<Key, Was>>,
    history: Vec<HashMap<Key, Was>>,
    redo: Vec<HashMap<Key, Was>>,
    /// Blocks whose samples or existence changed since [`Self::drain_dirty`] last ran.
    dirty: HashSet<Key>,
    /// During a stroke: its change so far at level `carry_level` (the level its dabs write), per block, which finer
    /// levels' detail is carried by.
    carry: HashMap<IVec3, Block>,
    carry_level: usize,
}

impl LevelVolume {
    /// `field` (negative inside) over `[-half, half]^3` at level 0's step `voxel0`, with `levels` levels in all.
    pub fn from_field<F: Fn(Vec3) -> f32 + Sync>(field: &F, half: f32, voxel0: f32, levels: usize) -> LevelVolume {
        let levels = levels.clamp(1, MAX_LEVELS);
        let origin = Vec3::ZERO;
        let mut v = LevelVolume { voxel0, origin, blocks: vec![HashMap::new(); levels], inside0: HashSet::new(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::new(), carry: HashMap::new(), carry_level: 0 };
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

    /// Distances kept at level `l`: 2.5 lattice steps, but 8 at level 0, so an edit there can carry finer detail as far
    /// as it moves the surface.
    pub fn band(&self, l: usize) -> f32 {
        if l == 0 { 8.0 * self.voxel0 } else { 2.5 * self.voxel(l) }
    }

    /// Stored blocks per level, and the bytes their samples take at 16 bits.
    pub fn memory(&self) -> (Vec<usize>, usize) {
        let counts: Vec<usize> = self.blocks.iter().map(|m| m.len()).collect();
        let bytes = counts.iter().sum::<usize>() * N * 2;
        (counts, bytes)
    }

    pub fn block(&self, k: Key) -> Option<&Block> {
        self.blocks.get(k.0 as usize)?.get(&k.1)
    }

    pub fn exists(&self, l: usize, bc: IVec3) -> bool {
        self.blocks[l].contains_key(&bc)
    }

    /// Every stored block.
    pub fn keys(&self) -> Vec<Key> {
        let mut out: Vec<Key> = self.blocks.iter().enumerate().flat_map(|(l, m)| m.keys().map(move |bc| (l as u8, *bc))).collect();
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out
    }

    /// Blocks whose samples or existence changed since the last call.
    pub fn drain_dirty(&mut self) -> Vec<Key> {
        let mut out: Vec<Key> = self.dirty.drain().collect();
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
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
    /// the stroke began, with `base`), with the normal there. Marched on level 0, whose band reaches farthest, then
    /// narrowed down on the finer levels.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max: usize, base: bool) -> Option<(Vec3, Vec3)> {
        let value = |p: Vec3| self.value_upto(p, max, base);
        let v0 = self.voxel0;
        let fine = self.voxel(max.min(self.levels() - 1));
        let coarse = |p: Vec3| self.value_at(0, (p - self.origin) / v0, base);
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
        while t < far {
            let v = coarse(origin + dir * t);
            // Close to the coarse surface, step on the finest field, which detail may have moved.
            if v < 2.5 * v0 {
                let mut s = t;
                let end = t + 5.0 * v0;
                while s < end {
                    if value(origin + dir * s) < 0.0 {
                        hit = true;
                        break;
                    }
                    prev = s;
                    s += (0.5 * fine).max(0.15 * value(origin + dir * s).abs());
                }
                t = s;
                if hit {
                    break;
                }
                continue;
            }
            prev = t;
            t += v.max(0.3 * v0);
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
                    let stored = self.stored(l, bc, base);
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
            let coarse = self.grid(l - 1, lo.div_euclid(IVec3::splat(2)), (hi + IVec3::ONE).div_euclid(IVec3::splat(2)), base);
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

    /// Block `bc` of level `l`'s 512 samples, stored or interpolated from coarser levels.
    fn block_values(&self, l: usize, bc: IVec3, base: bool) -> Block {
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
    pub fn gather9(&self, l: usize, bc: IVec3) -> Box<[f32; 729]> {
        let g = self.grid(l, bc * B, bc * B + IVec3::splat(B), false);
        let mut out = Box::new([0.0f32; 729]);
        out.copy_from_slice(&g.v);
        out
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
    fn ensure(&mut self, l: usize, bc: IVec3) {
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
    fn modify(&mut self, l: usize, c: Vec3, reach: f32, k: f32, carve: bool, slab: (Vec3, f32, f32), weight: &(dyn Fn(Vec3) -> f32 + Sync), shape: &(dyn Fn(Vec3) -> f32 + Sync)) -> LevelDab {
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
                    // Far outside the shape, combining changes nothing.
                    let sd = shape(centre);
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
        // New samples for each candidate, worked out in parallel from the field as it is.
        let origin = self.origin;
        let me = &*self;
        let built: Vec<(IVec3, Block, Block, SignMask)> = cand
            .par_iter()
            .filter_map(|&bc| {
                let old = me.block_values(l, bc, false);
                let mut s = old.clone();
                let mut d = Box::new([0.0f32; N]);
                let mut flips = [0u64; 8];
                let mut any = false;
                let corner = origin + (bc * B).as_vec3() * vx;
                for (i, v) in s.iter_mut().enumerate() {
                    let p = corner + cell_of(i).as_vec3() * vx;
                    let w = weight(p);
                    if w <= 0.0 {
                        continue;
                    }
                    let blob = shape(p);
                    let full = if carve { v.max(-blob) } else { v.min(blob) };
                    let next = (*v + w * (full - *v)).clamp(-band, band);
                    if next != *v {
                        if (next < 0.0) != (*v < 0.0) {
                            flips[i / 64] |= 1 << (i % 64);
                        }
                        d[i] = next - *v;
                        *v = next;
                        any = true;
                    }
                }
                any.then_some((bc, s, d, flips))
            })
            .collect();
        // Outside a stroke, the dab is a stroke of its own: detail is carried from the field as it was before it.
        let own = self.stroke.is_none();
        if own {
            self.begin_stroke();
        }
        if self.carry_level != l {
            self.carry.clear();
            self.carry_level = l;
        }
        let mut flipped: HashMap<IVec3, SignMask> = HashMap::new();
        let mut changed = Vec::with_capacity(built.len());
        for (bc, s, d, flips) in built {
            if !self.blocks[l].contains_key(&bc) {
                stats.created += 1;
            }
            self.put(l, bc, s);
            let total = self.carry.entry(bc).or_insert_with(|| Box::new([0.0; N]));
            for (t, x) in total.iter_mut().zip(d.iter()) {
                *t += x;
            }
            changed.push(bc);
            if flips.iter().any(|&m| m != 0) {
                flipped.insert(bc, flips);
            }
        }
        stats.changed = changed.len();
        self.clear_checkerboards(l, flipped);
        stats.carried = self.carry_detail(l, &changed);
        if own {
            self.stroke = None;
            self.carry.clear();
        }
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        stats
    }

    /// After a dab at level `l` changed `changed` (blocks of that level), every finer level's own detail (its values
    /// less the level below, both within its band) moves with the surface: along the normal, as far as the stroke has
    /// moved it so far. The detail is always taken from the surface as it was when the stroke began and moved by the
    /// stroke's whole change, so however many dabs a stroke has, its detail is resampled once. Coarsest first, so each
    /// level builds on the new level below. Returns the blocks rebuilt.
    fn carry_detail(&mut self, l: usize, changed: &[IVec3]) -> usize {
        if changed.is_empty() || l + 1 >= self.levels() {
            return 0;
        }
        let mut carried = 0;
        // The dab's footprint at level `l`, with a ring of blocks for detail moved across block edges.
        let mut frontier: Vec<IVec3> = changed.iter().flat_map(|&b| (0..27).map(move |q| b + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1))).collect();
        frontier.sort_unstable_by_key(|v| (v.z, v.y, v.x));
        frontier.dedup();
        for m in l + 1..self.levels() {
            // This level's blocks under the footprint, found through the nesting.
            let mut region: Vec<IVec3> = frontier.iter().flat_map(|&p| (0..8).map(move |q| p * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1))).filter(|c| self.blocks[m].contains_key(c)).collect();
            region.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            region.dedup();
            if region.is_empty() {
                break;
            }
            let k = 1i32 << (m - l);
            let vm = self.voxel(m);
            // Where the detail goes: the blocks it is in, and those its box covers once moved with the surface.
            let dest: Vec<IVec3> = {
                let me = &*self;
                let mut dest: Vec<IVec3> = region
                    .par_iter()
                    .flat_map_iter(|&bc| {
                        let mut out = vec![bc];
                        for c in 0..9 {
                            let corner = if c == 8 { Vec3::splat(0.5) } else { Vec3::new((c & 1) as f32, ((c >> 1) & 1) as f32, ((c >> 2) & 1) as f32) };
                            let at = (bc * B).as_vec3() + corner * B as f32;
                            let g = at / k as f32;
                            let d = me.carry_delta_at(g);
                            if d != 0.0 {
                                let dn = |o: Vec3| me.value_at(l, g + o, false) - me.value_at(l, g - o, false);
                                let n = Vec3::new(dn(Vec3::X), dn(Vec3::Y), dn(Vec3::Z)).normalize_or_zero();
                                out.push((at + n * (-d / vm)).floor().as_ivec3().div_euclid(B3));
                            }
                        }
                        out
                    })
                    .collect();
                dest.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                dest.dedup();
                dest
            };
            for &bc in &dest {
                if m > 1 {
                    self.ensure(m - 1, bc.div_euclid(IVec3::splat(2)));
                }
            }
            let me = &*self;
            let built: Vec<(IVec3, Block)> = dest.par_iter().filter_map(|&bc| me.carry_block(l, m, bc).map(|s| (bc, s))).collect();
            for (bc, s) in built {
                self.put(m, bc, s);
                carried += 1;
            }
            frontier = region;
        }
        carried
    }

    /// The stroke's change so far at level `l`'s lattice coordinates `g` (zero where it changed nothing).
    fn carry_delta_at(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        trilinear(f, std::array::from_fn(|q| {
            let p = i + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32);
            self.carry.get(&p.div_euclid(B3)).map_or(0.0, |d| d[local(p)])
        }))
    }

    /// Block `bc` of level `m` (finer than the stroke's level `l`) rebuilt: the new level below plus this level's detail
    /// from when the stroke began, moved with the surface. `None` if that is what the block holds already.
    fn carry_block(&self, l: usize, m: usize, bc: IVec3) -> Option<Block> {
        let k = 1i32 << (m - l);
        let (vm, fband) = (self.voxel(m), self.band(m));
        let lo = bc * B;
        // The stroke's change and the level `l` field over this block's footprint.
        let (llo, lhi) = (lo.div_euclid(IVec3::splat(k)) - IVec3::ONE, (lo + B3).div_euclid(IVec3::splat(k)) + IVec3::splat(2));
        let ln = lhi - llo + IVec3::ONE;
        let mut dgrid = Grid { lo: llo, n: ln, v: vec![0.0; (ln.x * ln.y * ln.z) as usize] };
        let mut most = 0.0f32;
        for z in llo.z..=lhi.z {
            for y in llo.y..=lhi.y {
                for x in llo.x..=lhi.x {
                    let p = IVec3::new(x, y, z);
                    if let Some(d) = self.carry.get(&p.div_euclid(B3)) {
                        let at = dgrid.idx(p);
                        dgrid.v[at] = d[local(p)];
                        most = most.max(d[local(p)].abs());
                    }
                }
            }
        }
        let current = self.blocks[m].get(&bc);
        let differs = |s: &Block| current.is_none_or(|c| **c != **s);
        if most == 0.0 {
            // Nothing moved here: the detail is as it was.
            let base = self.block_values(m, bc, true);
            return differs(&base).then_some(base);
        }
        let lgrid = self.grid(l, llo, lhi, false);
        // Detail as it was, over the box the moved detail comes from.
        let reach = (most / vm).ceil() as i32 + 2;
        let (blo, bhi) = (lo - IVec3::splat(reach), lo + IVec3::splat(B + reach));
        let base_m = self.grid(m, blo, bhi, true);
        let base_below = self.grid(m - 1, blo.div_euclid(IVec3::splat(2)), (bhi + IVec3::ONE).div_euclid(IVec3::splat(2)), true);
        let below = self.grid(m - 1, lo.div_euclid(IVec3::splat(2)) - IVec3::ONE, (lo + B3).div_euclid(IVec3::splat(2)) + IVec3::ONE, false);
        let detail_at = |j: IVec3| -> f32 {
            let j = j.clamp(blo, bhi);
            base_m.at(j) - base_below.trilinear(j.as_vec3() * 0.5).clamp(-fband, fband)
        };
        let kf = k as f32;
        let mut s = Box::new([0.0f32; N]);
        for (i, v) in s.iter_mut().enumerate() {
            let p = lo + cell_of(i);
            let g = p.as_vec3() / kf;
            let d = dgrid.trilinear(g);
            if d == 0.0 {
                *v = base_m.at(p);
                continue;
            }
            let dn = |o: Vec3| lgrid.trilinear(g + o) - lgrid.trilinear(g - o);
            let n = Vec3::new(dn(Vec3::X * 0.5), dn(Vec3::Y * 0.5), dn(Vec3::Z * 0.5)).normalize_or_zero();
            // The surface moved out by `-d` here: the detail now here was `-d` farther in.
            let q = p.as_vec3() + n * (d / vm);
            let j = q.floor().as_ivec3();
            let f = q - j.as_vec3();
            let detail = trilinear(f, std::array::from_fn(|r| detail_at(j + IVec3::new((r & 1) as i32, ((r >> 1) & 1) as i32, ((r >> 2) & 1) as i32))));
            *v = (below.trilinear(p.as_vec3() * 0.5) + detail).clamp(-fband, fband);
        }
        differs(&s).then_some(s)
    }

    /// Remove checkerboard faces (a cell face whose corners alternate in sign, which pieces cannot represent cleanly)
    /// from level `l`'s stored field, by filling one outside corner of each, repeatedly, as `volume.rs` does. Only the
    /// blocks with a sample whose sign changed (`flipped`) and the ones below them can have gained one.
    fn clear_checkerboards(&mut self, l: usize, mut flipped: HashMap<IVec3, SignMask>) {
        let fill = -1e-3 * self.voxel(l);
        for _ in 0..32 {
            if flipped.is_empty() {
                break;
            }
            let mut look: Vec<IVec3> = flipped.keys().flat_map(|&b| (0..8).map(move |k| b - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).filter(|b| self.blocks[l].contains_key(b)).collect();
            look.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            look.dedup();
            let me = &*self;
            let mut flips: Vec<IVec3> = look
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
                            out.push(base + *q);
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
            flips.sort_unstable_by_key(|p| (p.z, p.y, p.x));
            flips.dedup();
            flipped.clear();
            for p in flips {
                let bc = p.div_euclid(B3);
                self.keep((l as u8, bc));
                let i = local(p);
                if let Some(s) = self.blocks[l].get_mut(&bc) {
                    s[i] = fill;
                }
                self.dirty.insert((l as u8, bc));
                flipped.entry(bc).or_insert([0; 8])[i / 64] |= 1 << (i % 64);
            }
        }
    }

    // ------------------------------------------------------------------ plane brushes

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
    fn area_normal(&self, l: usize, c: Vec3, n: Vec3, r: f32) -> Vec3 {
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
        sum.try_normalize().filter(|u| u.dot(n) > 0.3).unwrap_or(n)
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
        let first = self.area_normal(l, on, along, r);
        let n = self.area_normal(l, on, first, r);
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
        let reach = 1.1 * r + depth;
        let dab = if trim {
            let shape = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                (plane - u).max(u - (level + depth)).max((q - n * u).length() - r)
            };
            self.modify(l, on, reach, k, true, (n, plane, level + depth), &weight, &shape)
        } else {
            let shape = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                (u - plane).max((level - depth) - u).max((q - n * u).length() - r)
            };
            self.modify(l, on, reach, k, false, (n, level - depth, plane), &weight, &shape)
        };
        Some((n, dab))
    }

    /// Add (or with `carve`, remove) a soft ball of radius `r` at `c`, blended over `k`, at level `l`.
    pub fn ball_dab(&mut self, l: usize, c: Vec3, r: f32, k: f32, carve: bool) -> LevelDab {
        let l = l.min(self.levels() - 1);
        let shape = move |p: Vec3| (p - c).length() - r;
        let weight = |_: Vec3| 1.0;
        self.modify(l, c, r + k, k, carve, (Vec3::Z, -(r + k), r + k), &weight, &shape)
    }

    // ------------------------------------------------------------------ strokes and history

    /// Start a stroke: until [`Self::end_stroke`], brushes are placed on the surface as it is now.
    pub fn begin_stroke(&mut self) {
        self.stroke = Some(HashMap::new());
        self.carry.clear();
    }

    pub fn in_stroke(&self) -> bool {
        self.stroke.is_some()
    }

    /// End the stroke: move the form it built at fine levels down a level, remove the specks it left, and keep it for
    /// [`Self::undo`].
    pub fn end_stroke(&mut self) -> StrokeEnd {
        let t = Instant::now();
        let mut out = StrokeEnd::default();
        if self.stroke.as_ref().is_none_or(|s| s.is_empty()) {
            self.stroke = None;
            return out;
        }
        let touched: Vec<Key> = self.stroke.as_ref().unwrap().keys().copied().collect();
        self.redistance(&touched);
        out.rebased = self.rebase(&touched);
        let touched: Vec<Key> = self.stroke.as_ref().unwrap().keys().copied().collect();
        out.specks = self.remove_specks(&touched, SPECK_VERTS);
        let saved = self.stroke.take().unwrap_or_default();
        self.history.push(saved);
        self.redo.clear();
        let size = |m: &HashMap<Key, Was>| m.values().map(|w| 16 + if matches!(w, Was::Stored(_)) { N * 4 } else { 0 }).sum::<usize>();
        let mut total: usize = self.history.iter().map(size).sum();
        while total > HISTORY_BYTES && self.history.len() > 1 {
            total -= size(&self.history.remove(0));
        }
        out.ms = t.elapsed().as_secs_f64() * 1e3;
        out
    }

    pub fn undo(&mut self) -> bool {
        self.stroke = None;
        let Some(entry) = self.history.pop() else { return false };
        let back = self.swap_in(entry);
        self.redo.push(back);
        true
    }

    pub fn redo(&mut self) -> bool {
        self.stroke = None;
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

    fn swap_in(&mut self, entry: HashMap<Key, Was>) -> HashMap<Key, Was> {
        let mut back = HashMap::with_capacity(entry.len());
        for (k, was) in entry {
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
    /// a level, the next brush's plane) would be misled. Samples next to the surface keep their values, so the surface
    /// does not move; every other sample becomes its distance to the nearest crossing of the surface with the lattice,
    /// within the band.
    fn redistance(&mut self, touched: &[Key]) {
        for l in 0..self.levels() {
            let blocks: Vec<IVec3> = touched.iter().filter(|k| k.0 as usize == l && self.blocks[l].contains_key(&k.1)).map(|k| k.1).collect();
            if blocks.is_empty() {
                continue;
            }
            let (vx, band) = (self.voxel(l), self.band(l));
            let reach = (band / vx).ceil() as i32 + 1;
            let me = &*self;
            let fixed: Vec<(IVec3, Block)> = blocks
                .par_iter()
                .filter_map(|&bc| {
                    let lo = bc * B - IVec3::splat(reach);
                    let g = me.grid(l, lo, bc * B + IVec3::splat(B - 1 + reach), false);
                    // Where the surface crosses the lattice's edges, in lattice units.
                    let mut crossings: Vec<Vec3> = Vec::new();
                    for z in 0..g.n.z {
                        for y in 0..g.n.y {
                            for x in 0..g.n.x {
                                let i = lo + IVec3::new(x, y, z);
                                let a = g.at(i);
                                for (axis, inside) in [(0, x + 1 < g.n.x), (1, y + 1 < g.n.y), (2, z + 1 < g.n.z)] {
                                    if !inside {
                                        continue;
                                    }
                                    let mut j = i;
                                    j[axis] += 1;
                                    let b = g.at(j);
                                    if (a < 0.0) != (b < 0.0) {
                                        crossings.push(i.as_vec3().lerp(j.as_vec3(), a / (a - b)));
                                    }
                                }
                            }
                        }
                    }
                    let old = &me.blocks[l][&bc];
                    let mut s = old.clone();
                    let mut any = false;
                    for (i, v) in s.iter_mut().enumerate() {
                        let p = bc * B + cell_of(i);
                        let a = g.at(p);
                        // Next to the surface: the value places it.
                        let near = (0..6).any(|q| {
                            let mut o = IVec3::ZERO;
                            o[q / 2] = if q % 2 == 0 { 1 } else { -1 };
                            (g.at(p + o) < 0.0) != (a < 0.0)
                        });
                        if near {
                            continue;
                        }
                        let pf = p.as_vec3();
                        let d2 = crossings.iter().map(|c| c.distance_squared(pf)).fold(f32::INFINITY, f32::min);
                        let d = (d2.sqrt() * vx).min(band);
                        let next = if a < 0.0 { -d } else { d };
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

    /// Fine work that built real form, moved down a level: at each level from the finest, under each touched block,
    /// the level below moves toward this level's surface, smoothed to the coarser lattice, by however far the two differ
    /// beyond [`REBASE_KEEP`] of this level's band. Small detail (a groove, a scratch) stays where it was sculpted;
    /// broad form built up at a fine level ends up in the level below too, where a distant view sees it and a later
    /// coarse stroke treats it as form. The finer level's values do not change, so neither does the surface. Returns the
    /// coarser blocks changed.
    fn rebase(&mut self, touched: &[Key]) -> usize {
        let mut changed_total = 0;
        let mut region: HashSet<IVec3> = HashSet::new();
        for m in (1..self.levels()).rev() {
            let mut blocks: Vec<IVec3> = touched.iter().filter(|k| k.0 as usize == m).map(|k| k.1).chain(region.drain()).filter(|bc| self.blocks[m].contains_key(bc)).collect();
            blocks.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            blocks.dedup();
            if blocks.is_empty() {
                continue;
            }
            let (fband, cband) = (self.band(m), self.band(m - 1));
            let keep = REBASE_KEEP * fband;
            let fine: HashSet<IVec3> = blocks.iter().copied().collect();
            let mut under: Vec<IVec3> = blocks.iter().map(|fb| fb.div_euclid(IVec3::splat(2))).collect();
            under.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            under.dedup();
            let me = &*self;
            let wts = [1.0f32, 4.0, 6.0, 4.0, 1.0];
            let changes: Vec<(IVec3, Block)> = under
                .par_iter()
                .filter_map(|&cb| {
                    // This level's field around the coarse block, smoothed at each coarse point under a touched block.
                    let lo = cb * B * 2 - IVec3::splat(2);
                    let g = me.grid(m, lo, lo + IVec3::splat(2 * B + 2), false);
                    let mut target = [f32::NAN; N];
                    for (i, t) in target.iter_mut().enumerate() {
                        let c = cb * B + cell_of(i);
                        if !fine.contains(&(c * 2).div_euclid(B3)) {
                            continue;
                        }
                        let mut sum = 0.0;
                        for z in -2..=2i32 {
                            for y in -2..=2i32 {
                                for x in -2..=2i32 {
                                    sum += wts[(x + 2) as usize] * wts[(y + 2) as usize] * wts[(z + 2) as usize] * g.at(c * 2 + IVec3::new(x, y, z));
                                }
                            }
                        }
                        *t = sum / 4096.0;
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
                changed_total += 1;
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
        let mut first: HashMap<Key, u32> = HashMap::new();
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
        for s in surf.values() {
            for q in &s.quads {
                let ids: Vec<Option<u32>> = q.iter().map(|(k, c, p)| Some(first.get(k)? + surf[k].vertex(*c, *p)? as u32)).collect();
                let known: Vec<u32> = ids.iter().flatten().copied().collect();
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
        let mut size: HashMap<u32, usize> = HashMap::new();
        for i in 0..total {
            let r = root(&mut parent, i);
            *size.entry(r).or_default() += 1;
        }
        let specks: HashSet<u32> = size.iter().filter(|&(&r, &n)| n <= max_verts && !open[r as usize]).map(|(&r, _)| r).collect();
        if specks.is_empty() {
            return 0;
        }
        // The cells (level, lattice cell) of each speck, and which pieces every cell holds.
        let mut cells_of: HashMap<u32, Vec<(usize, IVec3)>> = HashMap::new();
        let mut pieces_at: HashMap<(usize, IVec3), Vec<u32>> = HashMap::new();
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
            let mut corners: Vec<(usize, IVec3)> = cells.iter().flat_map(|&(l, c)| (0..8).map(move |k| (l, c + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1)))).map(|(l, p)| self.canonical(l, p, cap)).collect();
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
        let mut flipped: Vec<HashMap<IVec3, SignMask>> = vec![HashMap::new(); self.levels()];
        for (l, p, to) in edits {
            let bc = p.div_euclid(B3);
            self.keep((l as u8, bc));
            let i = local(p);
            if let Some(s) = self.blocks[l].get_mut(&bc) {
                s[i] = to;
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
    fn canonical(&self, l: usize, i: IVec3, cap: usize) -> (usize, IVec3) {
        let (mut l, mut i) = (l, i);
        while l < cap && self.blocks[l + 1].contains_key(&(i * 2).div_euclid(B3)) {
            l += 1;
            i *= 2;
        }
        (l, i)
    }

    /// Whether level `l`'s cell `q` is split by finer cells in a view capped at `cap`.
    fn covered(&self, l: usize, q: IVec3, cap: usize) -> bool {
        l < cap && self.blocks[l + 1].contains_key(&(q * 2).div_euclid(B3))
    }

    /// The smallest cell holding level `l`'s cell `q` (which must not be covered): its level and coordinates, if that
    /// level has a block there (level 0 cells of unstored blocks have none).
    fn leaf_down(&self, l: usize, q: IVec3) -> Option<(usize, IVec3)> {
        for m in (0..=l).rev() {
            let c = q.div_euclid(IVec3::splat(1 << (l - m)));
            if self.blocks[m].contains_key(&c.div_euclid(B3)) {
                return Some((m, c));
            }
            if m == 0 {
                return None;
            }
        }
        None
    }

    /// The blocks to re-extract when the blocks `changed` changed (samples or existence), in a view capped at `cap`:
    /// every stored block whose cells read those samples, whose cells a change in which blocks exist splits or joins,
    /// or whose quads point at their vertices. That is, at every level, the stored blocks meeting a changed block's box
    /// grown by two of its cells.
    pub fn extract_set(&self, changed: &[Key], cap: usize) -> Vec<Key> {
        let mut out: HashSet<Key> = HashSet::new();
        let cap = cap.min(self.levels() - 1);
        for &(l, bc) in changed {
            let l = l as usize;
            if l > cap {
                continue;
            }
            // The box in level `l` lattice units, grown by two cells.
            let lo = bc * B - IVec3::splat(2);
            let hi = bc * B + IVec3::splat(B + 2);
            // Coarser levels and this one: the few blocks meeting the box.
            for m in 0..=l {
                let s = 1i32 << (l - m);
                let (a, b) = (lo.div_euclid(IVec3::splat(s)).div_euclid(B3), (hi - IVec3::ONE).div_euclid(IVec3::splat(s)).div_euclid(B3));
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            let c = IVec3::new(x, y, z);
                            if self.blocks[m].contains_key(&c) {
                                out.insert((m as u8, c));
                            }
                        }
                    }
                }
            }
            // Finer levels: stored descendants of the box, found through the nesting.
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
                        if self.blocks[m].contains_key(&c) {
                            out.insert((m as u8, c));
                            next.push(c);
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
        }
        let mut v: Vec<Key> = out.into_iter().collect();
        v.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
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
        let t = Instant::now();
        let cap = cap.min(self.levels() - 1);
        let table = piece_table();
        let mut stats = ExtractStats::default();
        let mut surf: HashMap<Key, BlockSurface> = work.par_iter().filter(|k| self.block(**k).is_some() && (k.0 as usize) <= cap).map(|&k| (k, self.block_vertices(k, cap, &table))).collect();
        // Quads read the vertices of blocks outside `work` too: extract those (vertices only) on demand.
        let mut extra: HashMap<Key, BlockSurface> = HashMap::new();
        let quads: Vec<(Key, Vec<[Corner; 4]>, usize)> = {
            let need: HashSet<Key> = work
                .par_iter()
                .filter(|k| surf.contains_key(k))
                .flat_map_iter(|&k| self.quad_neighbours(k, cap))
                .filter(|k| !surf.contains_key(k))
                .collect();
            extra.extend(need.into_par_iter().map(|k| (k, self.block_vertices(k, cap, &table))).collect::<Vec<_>>());
            let all = |k: &Key| surf.get(k).or(extra.get(k));
            work.par_iter().filter(|k| surf.contains_key(k)).map(|&k| {
                let (q, dropped) = self.block_quads(k, cap, &table, &all);
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
    fn quad_neighbours(&self, k: Key, cap: usize) -> Vec<Key> {
        let (l, bc) = (k.0 as usize, k.1);
        let mut out = Vec::new();
        let lo = bc * B - IVec3::ONE;
        let hi = bc * B + B3;
        for m in 0..=l.min(cap) {
            let s = 1i32 << (l - m);
            let (a, b) = (lo.div_euclid(IVec3::splat(s)).div_euclid(B3), hi.div_euclid(IVec3::splat(s)).div_euclid(B3));
            for z in a.z..=b.z {
                for y in a.y..=b.y {
                    for x in a.x..=b.x {
                        let c = IVec3::new(x, y, z);
                        if self.blocks[m].contains_key(&c) {
                            out.push((m as u8, c));
                        }
                    }
                }
            }
        }
        out
    }

    /// The value at level `l`'s point `i` for a cell of block `bc` (whose 9x9x9 samples are `g`).
    fn corner_value(&self, l: usize, bc: IVec3, g: &[f32; 729], i: IVec3, cap: usize) -> f32 {
        let (m, j) = self.canonical(l, i, cap);
        if m > l {
            return self.blocks[m][&j.div_euclid(B3)][local(j)];
        }
        let d = i - bc * B;
        g[(d.z * 81 + d.y * 9 + d.x) as usize]
    }

    fn block_vertices(&self, k: Key, cap: usize, table: &[u32; 256]) -> BlockSurface {
        let (l, bc) = (k.0 as usize, k.1);
        let g = self.gather9(l, bc);
        let vx = self.voxel(l);
        let base = bc * B;
        let mut out = BlockSurface { cells: vec![0; N], ..Default::default() };
        // Which level `l` cells around the block are split by finer ones: a 10x10x10 neighbourhood.
        let mut finer = [false; 1000];
        if l < cap {
            for z in -1..9 {
                for y in -1..9 {
                    for x in -1..9 {
                        let q = base + IVec3::new(x, y, z);
                        finer[((z + 1) * 100 + (y + 1) * 10 + (x + 1)) as usize] = self.covered(l, q, cap);
                    }
                }
            }
        }
        let fin = |o: IVec3| finer[((o.z + 1) * 100 + (o.y + 1) * 10 + (o.x + 1)) as usize];
        for i in 0..N {
            let c = cell_of(i);
            if fin(c) {
                continue;
            }
            let d: [f32; 8] = std::array::from_fn(|q| self.corner_value(l, bc, &g, base + c + IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, ((q >> 2) & 1) as i32), cap));
            let config = (0..8).fold(0u32, |a, q| a | ((d[q] < 0.0) as u32) << q);
            let transition = l < cap && (0..27).any(|q| q != 13 && fin(c + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1)));
            let first = out.verts.len() as u32;
            let corner_pos = |q: usize| Vec3::new(corner(q)[0] as f32, corner(q)[1] as f32, corner(q)[2] as f32);
            let push = |out: &mut BlockSurface, f: Vec3| {
                let f = f.clamp(Vec3::ZERO, Vec3::ONE);
                out.verts.push(self.origin + (base + c).as_vec3() * vx + f * vx);
                out.normals.push(trilinear_gradient(f, d));
            };
            let count = if transition {
                // Crossings on the cell's faces split once: values at the half-step points from the finer level where it
                // holds them, else interpolated from the corners.
                let mut sum = Vec3::ZERO;
                let mut n = 0.0;
                let half = |h: IVec3| -> f32 {
                    // `h` in half-cell units (0..2 per axis).
                    let p = (base + c) * 2 + h;
                    let (m, j) = self.canonical(l + 1, p, cap);
                    if m > l + 1 || self.blocks[l + 1].contains_key(&p.div_euclid(B3)) {
                        let j = if m > l + 1 { j } else { p };
                        return self.blocks[m.max(l + 1)][&j.div_euclid(B3)][local(j)];
                    }
                    trilinear(h.as_vec3() * 0.5, d)
                };
                for axis in 0..3 {
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    // Half-edges along `axis` on the cell's boundary: those with u or v at 0 or 2.
                    for a in 0..3 {
                        for b in 0..3 {
                            if a == 1 && b == 1 {
                                continue;
                            }
                            for s in 0..2 {
                                let mut p0 = IVec3::ZERO;
                                p0[u] = a;
                                p0[v] = b;
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
                    // Near the centre, moved onto the field's zero along its gradient.
                    let m = Vec3::splat(0.5);
                    let gr = trilinear_gradient(m, d);
                    m - gr * trilinear(m, d) / gr.length_squared().max(1e-12)
                };
                if first as usize + 1 > VERTS_PER_BLOCK {
                    continue;
                }
                push(&mut out, f);
                1
            } else {
                if config == 0 || config == 255 {
                    continue;
                }
                let entry = table[config as usize];
                let pieces = (entry >> 24) as usize;
                if first as usize + pieces > VERTS_PER_BLOCK {
                    continue;
                }
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
                    push(&mut out, sum[p] / cnt[p].max(1.0));
                }
                pieces as u32
            };
            out.cells[i] = first | count << 10 | config << 13 | (transition as u32) << 21;
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

    fn block_quads<'a>(&self, k: Key, cap: usize, table: &[u32; 256], surf: &impl Fn(&Key) -> Option<&'a BlockSurface>) -> (Vec<[Corner; 4]>, usize) {
        let (l, bc) = (k.0 as usize, k.1);
        let g = self.gather9(l, bc);
        let base = bc * B;
        let mut out = Vec::new();
        let mut dropped = 0;
        let Some(me) = surf(&k) else { return (out, 0) };
        for i in 0..N {
            let c = base + cell_of(i);
            if self.covered(l, c, cap) {
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
                let va = self.corner_value(l, bc, &g, start, cap);
                let vb = self.corner_value(l, bc, &g, end, cap);
                if (va < 0.0) == (vb < 0.0) {
                    continue;
                }
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let mut eu = IVec3::ZERO;
                eu[u] = 1;
                let mut ev = IVec3::ZERO;
                ev[v] = 1;
                // The four cells around the edge, in quad order, and the corner each starts the edge at.
                let around = [start - eu - ev, start - ev, start, start - eu];
                if around.iter().any(|&q| self.covered(l, q, cap)) {
                    continue;
                }
                let level_of = |q: IVec3| self.leaf_down(l, q).map(|(m, _)| m);
                // The owner: the first at this level in the order start, below along v, below along u, below along both.
                let owner = [2usize, 1, 3, 0].into_iter().find(|&j| level_of(around[j]) == Some(l));
                if owner.map(|j| around[j]) != Some(c) {
                    continue;
                }
                let corners: Option<Vec<Corner>> = around
                    .iter()
                    .map(|&q| {
                        let (m, qc) = self.leaf_down(l, q)?;
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
        let work: Vec<Key> = self.keys();
        let (surf, stats) = self.extract(&work, cap);
        let mut first: HashMap<Key, u32> = HashMap::new();
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
        w.write_all(&2u32.to_le_bytes())?;
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
        if version != 2 {
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
        let mut v = LevelVolume { voxel0: f[0], origin: Vec3::new(f[1], f[2], f[3]), blocks: vec![HashMap::new(); n.max(levels.min(MAX_LEVELS))], inside0: HashSet::new(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::new(), carry: HashMap::new(), carry_level: 0 };
        for _ in 0..u32_(r)? {
            v.inside0.insert(ivec(r)?);
        }
        let mut buf = vec![0u8; N * 2];
        for l in 0..n {
            let band = v.band(l);
            for _ in 0..u32_(r)? {
                let bc = ivec(r)?;
                r.read_exact(&mut buf)?;
                let mut s = Box::new([0f32; N]);
                for (i, x) in s.iter_mut().enumerate() {
                    *x = i16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]) as f32 / 32767.0 * band;
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
        let mut v = LevelVolume { voxel0: old.voxel, origin, blocks: vec![HashMap::new(); levels.clamp(1, MAX_LEVELS)], inside0: HashSet::new(), stroke: None, history: Vec::new(), redo: Vec::new(), dirty: HashSet::new(), carry: HashMap::new(), carry_level: 0 };
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

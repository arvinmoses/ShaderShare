//! SPIKE (measurement only): sculpting in a volume instead of on a mesh.
//!
//! The question this answers: can a brush dab change a sparse distance field and re-extract only the surface it touched
//! fast enough to sculpt with, at 1-20 million triangle detail? It implements just enough to time that honestly:
//!
//! * **Storage.** The lattice is cut into blocks of 8x8x8 samples. Blocks within a few lattice steps of the surface hold
//!   all their samples (distances clamped to +-`band`); every other block holds only whether it is inside or outside.
//! * **Dabs.** Clay is a smooth union of a soft blob into the field, carving a smooth subtraction. Unions and
//!   subtractions stay correct on a clamped field (a displacement would not: far from the surface the clamped value
//!   understates the distance, so pushing it would conjure surface out of nothing). A dab touches only the blocks around
//!   it, turning inside/outside blocks into full ones where the blob reaches.
//! * **Extraction.** Naive surface nets per block: one vertex per cell the surface crosses, one quad per crossing lattice
//!   edge. A dab rebuilds the blocks it changed plus the neighbours whose cells read those samples. The manifold pieces
//!   of `remesh.rs` are left out here; they add a fixed, small cost per cell.
//!
//! It is not wired to the app or the document. GPU upload is reported as bytes, not timed.

use std::collections::HashMap;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

const B: i32 = 8;
const N: usize = 512;

/// A quad's corners, each as (block, cell).
pub type Quad = [(u32, u16); 4];
/// A block as first built: its index, inside (-1) / outside (1) / stored (0), and its samples when stored.
type Filled = (u32, i8, Option<Box<[f32; N]>>);
/// A block after refinement: its index, its samples when it holds surface, else whether it is inside (-1) or out (1).
type Refilled = (u32, Option<Box<[f32; N]>>, i8);

/// A sculpt-able solid stored as a sparse, clamped distance field.
pub struct Volume {
    pub voxel: f32,
    pub band: f32,
    origin: Vec3,
    dims: IVec3,
    /// Per block: `-1` inside, `1` outside, `0` stored in full in `dense`.
    signs: Vec<i8>,
    dense: HashMap<u32, Box<[f32; N]>>,
    meshes: HashMap<u32, BlockMesh>,
    /// Blocks whose surface was rebuilt since [`Volume::drain_changed`] last ran, for a viewport to re-upload.
    changed: std::collections::HashSet<u32>,
    /// While a stroke is under way: each block's samples (or sign) from before the stroke first changed it, so dabs
    /// are placed on the surface as it was when the stroke began.
    saved: Option<HashMap<u32, Saved>>,
}

/// A block as it was: its sign, and its samples when stored.
type Saved = (i8, Option<Box<[f32; N]>>);

/// The surface inside one block: a vertex per crossed cell, and the quads whose lattice edges start in this block.
#[derive(Default)]
struct BlockMesh {
    verts: Vec<Vec3>,
    cell: Vec<u32>,
    /// Each corner as (block, cell); resolved to indices when the whole mesh is assembled.
    quads: Vec<Quad>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DabStats {
    pub field_ms: f64,
    pub extract_ms: f64,
    pub blocks_touched: usize,
    pub blocks_rebuilt: usize,
    pub new_dense: usize,
    pub verts: usize,
    pub quads: usize,
    /// What a viewport would upload for the rebuilt blocks: 8-byte vertices and 6 32-bit indices per quad.
    pub upload_bytes: usize,
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    -smin(-a, -b, k)
}

const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

impl Volume {
    /// A volume over `[-half, half]^3` holding `field` (negative inside), at lattice spacing `voxel`.
    pub fn from_field<F: Fn(Vec3) -> f32 + Sync>(field: &F, half: f32, voxel: f32) -> Volume {
        Volume::from_field_with_band(field, half, voxel, 2.5 * voxel)
    }

    /// [`Self::from_field`] keeping true distances out to `band` instead of a few lattice steps. A coarse level needs
    /// a wide band: refining carries detail back along the distance the surface moved, which must not be clamped.
    pub fn from_field_with_band<F: Fn(Vec3) -> f32 + Sync>(field: &F, half: f32, voxel: f32, band: f32) -> Volume {
        let blocks = ((2.0 * half / voxel) / B as f32).ceil() as i32 + 2;
        let dims = IVec3::splat(blocks);
        let origin = Vec3::splat(-half - voxel * B as f32);
        let mut v = Volume { voxel, band, origin, dims, signs: vec![1; (blocks as usize).pow(3)], dense: HashMap::new(), meshes: HashMap::new(), changed: Default::default(), saved: None };
        let reach = 0.5 * B as f32 * voxel * 3f32.sqrt() + band;
        let total = v.signs.len();
        let filled: Vec<Filled> = (0..total as u32)
            .into_par_iter()
            .map(|b| {
                let lo = v.block_origin(b);
                let c = field(lo + Vec3::splat(0.5 * (B - 1) as f32 * voxel));
                if c.abs() > reach {
                    return (b, if c < 0.0 { -1 } else { 1 }, None);
                }
                let mut s = Box::new([0.0f32; N]);
                for (i, slot) in s.iter_mut().enumerate() {
                    let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                    *slot = field(lo + l.as_vec3() * voxel).clamp(-band, band);
                }
                (b, 0, Some(s))
            })
            .collect();
        for (b, sign, s) in filled {
            v.signs[b as usize] = sign;
            if let Some(s) = s {
                v.dense.insert(b, s);
            }
        }
        v
    }

    fn block_coords(&self, b: u32) -> IVec3 {
        let d = self.dims;
        IVec3::new(b as i32 % d.x, (b as i32 / d.x) % d.y, b as i32 / (d.x * d.y))
    }

    fn block_index(&self, c: IVec3) -> Option<u32> {
        (c.cmpge(IVec3::ZERO).all() && c.cmplt(self.dims).all()).then(|| ((c.z * self.dims.y + c.y) * self.dims.x + c.x) as u32)
    }

    fn block_origin(&self, b: u32) -> Vec3 {
        self.origin + (self.block_coords(b) * B).as_vec3() * self.voxel
    }

    /// Bytes held by the stored samples, at 32 bits and at the 16 bits a real version would use.
    pub fn memory(&self) -> (usize, usize) {
        let n = self.dense.len() * N;
        (n * 4 + self.signs.len(), n * 2 + self.signs.len())
    }

    pub fn dense_blocks(&self) -> usize {
        self.dense.len()
    }

    /// Extract every block's surface.
    pub fn extract_all(&mut self) -> (f64, usize, usize) {
        let t = Instant::now();
        let all: Vec<u32> = self.dense.keys().copied().collect();
        // Cells in a block read samples of the blocks above it, so the blocks just below dense ones can hold surface.
        let mut with_lower: Vec<u32> = all.iter().flat_map(|&b| self.lower_neighbours(b)).collect();
        with_lower.sort_unstable();
        with_lower.dedup();
        self.rebuild(&with_lower);
        let (verts, quads) = self.meshes.values().fold((0, 0), |(v, q), m| (v + m.verts.len(), q + m.quads.len()));
        (t.elapsed().as_secs_f64() * 1e3, verts, quads)
    }

    fn lower_neighbours(&self, b: u32) -> Vec<u32> {
        let c = self.block_coords(b);
        (0..8).filter_map(|k| self.block_index(c - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).collect()
    }

    /// Rebuild the surface of `blocks`: vertices first (every block), then quads (which read neighbours' vertices).
    fn rebuild(&mut self, blocks: &[u32]) {
        self.changed.extend(blocks.iter().copied());
        let built: Vec<(u32, BlockMesh)> = blocks.par_iter().map(|&b| (b, self.block_vertices(b))).collect();
        for (b, m) in built {
            if m.verts.is_empty() {
                self.meshes.remove(&b);
            } else {
                self.meshes.insert(b, m);
            }
        }
        let quads: Vec<(u32, Vec<Quad>)> = blocks.par_iter().map(|&b| (b, self.block_quads(b))).collect();
        for (b, q) in quads {
            if let Some(m) = self.meshes.get_mut(&b) {
                m.quads = q;
            }
        }
    }

    /// The 9x9x9 samples a block's cells read (its own and the first layer of the blocks above it), gathered with one
    /// lookup per block instead of one per sample.
    fn gather(&self, b: u32) -> Box<[f32; 729]> {
        let c = self.block_coords(b);
        let mut out = Box::new([0.0f32; 729]);
        for k in 0..8 {
            let off = IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1);
            let nb = self.block_index(c + off);
            let (dense, uniform) = match nb {
                Some(n) => (self.dense.get(&n), self.signs[n as usize] as f32 * self.band),
                None => (None, self.band),
            };
            // The part of this neighbour the cube covers: all of it for the block itself, one layer for the others.
            let range = |o: i32| if o == 0 { 0..8 } else { 0..1 };
            for z in range(off.z) {
                for y in range(off.y) {
                    for x in range(off.x) {
                        let v = dense.map_or(uniform, |d| d[(z * 64 + y * 8 + x) as usize]);
                        let (gx, gy, gz) = (x + 8 * off.x, y + 8 * off.y, z + 8 * off.z);
                        out[(gz * 81 + gy * 9 + gx) as usize] = v;
                    }
                }
            }
        }
        out
    }

    fn block_vertices(&self, b: u32) -> BlockMesh {
        let base = self.block_coords(b) * B;
        let g = self.gather(b);
        let mut m = BlockMesh::default();
        // Nothing crosses: all one sign.
        let neg = g.iter().filter(|&&v| v < 0.0).count();
        if neg == 0 || neg == 729 {
            return m;
        }
        m.cell = vec![u32::MAX; N];
        let at = |x: i32, y: i32, z: i32| g[(z * 81 + y * 9 + x) as usize];
        for i in 0..N {
            let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
            let d: [f32; 8] = std::array::from_fn(|k| at(l.x + (k & 1) as i32, l.y + ((k >> 1) & 1) as i32, l.z + ((k >> 2) & 1) as i32));
            let inside = d.iter().filter(|&&x| x < 0.0).count();
            if inside == 0 || inside == 8 {
                continue;
            }
            let (mut sum, mut n) = (Vec3::ZERO, 0.0);
            for e in EDGES {
                let (a, c) = (d[e[0]], d[e[1]]);
                if (a < 0.0) != (c < 0.0) {
                    let pa = Vec3::new((e[0] & 1) as f32, ((e[0] >> 1) & 1) as f32, ((e[0] >> 2) & 1) as f32);
                    let pc = Vec3::new((e[1] & 1) as f32, ((e[1] >> 1) & 1) as f32, ((e[1] >> 2) & 1) as f32);
                    sum += pa.lerp(pc, a / (a - c));
                    n += 1.0;
                }
            }
            m.cell[i] = m.verts.len() as u32;
            m.verts.push(self.origin + ((base + l).as_vec3() + sum / n) * self.voxel);
        }
        m
    }

    fn block_quads(&self, b: u32) -> Vec<Quad> {
        let base = self.block_coords(b) * B;
        let g = self.gather(b);
        let neg = g.iter().filter(|&&v| v < 0.0).count();
        if neg == 0 || neg == 729 {
            return Vec::new();
        }
        let at = |l: IVec3| g[(l.z * 81 + l.y * 9 + l.x) as usize];
        let mut out = Vec::new();
        let cell_of = |c: IVec3| -> Option<(u32, u16)> {
            let blk = self.block_index(c.div_euclid(IVec3::splat(B)))?;
            let l = c.rem_euclid(IVec3::splat(B));
            let idx = (l.z * 64 + l.y * 8 + l.x) as u16;
            self.meshes.get(&blk).filter(|m| m.cell.get(idx as usize).is_some_and(|&v| v != u32::MAX)).map(|_| (blk, idx))
        };
        for i in 0..N {
            let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
            let f0 = at(l);
            for axis in 0..3 {
                let mut step = IVec3::ZERO;
                step[axis] = 1;
                let f1 = at(l + step);
                if (f0 < 0.0) == (f1 < 0.0) {
                    continue;
                }
                let s = base + l;
                let (c1, c2) = ((axis + 1) % 3, (axis + 2) % 3);
                let around = |q1: i32, q2: i32| {
                    let mut c = s;
                    c[c1] -= 1 - q1;
                    c[c2] -= 1 - q2;
                    cell_of(c)
                };
                if let (Some(a), Some(bq), Some(c), Some(d)) = (around(0, 0), around(1, 0), around(1, 1), around(0, 1)) {
                    out.push(if f0 < 0.0 { [a, bq, c, d] } else { [a, d, c, bq] });
                }
            }
        }
        out
    }

    /// The blocks a blob of radius `r` at `c`, blended over `k`, can change: those its surface passes through
    /// (sample by sample), and those wholly inside it (they become solid or empty). Only blocks the blob's own surface
    /// passes through change sample by sample; out of its reach nothing changes: a thin shell of blocks, not a ball.
    pub fn dab_shell(&self, c: Vec3, r: f32, k: f32) -> (Vec<u32>, Vec<u32>) {
        self.shell_of(c, r, r, k)
    }

    /// [`Self::dab_shell`] for a shape that lies within `outer` of `c` and holds the ball of radius `inner` around it.
    fn shell_of(&self, c: Vec3, outer: f32, inner: f32, k: f32) -> (Vec<u32>, Vec<u32>) {
        let r = outer;
        let reach = r + k + self.band;
        let lo = ((c - Vec3::splat(reach) - self.origin) / (self.voxel * B as f32)).floor().as_ivec3();
        let hi = ((c + Vec3::splat(reach) - self.origin) / (self.voxel * B as f32)).floor().as_ivec3();
        let shell = self.band + k;
        let (mut touched, mut swallowed) = (Vec::new(), Vec::new());
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let Some(b) = self.block_index(IVec3::new(x, y, z)) else { continue };
                    let bmin = self.block_origin(b);
                    let bmax = bmin + Vec3::splat((B - 1) as f32 * self.voxel);
                    let near = (c - c.clamp(bmin, bmax)).length();
                    let far = (c - bmin).abs().max((c - bmax).abs()).length();
                    if near - r > shell {
                        continue;
                    }
                    if far < inner - shell {
                        swallowed.push(b);
                    } else {
                        touched.push(b);
                    }
                }
            }
        }
        (touched, swallowed)
    }

    /// Block coordinates of block `b`, and the block at coordinates `c` (if inside the grid).
    pub fn coords_of(&self, b: u32) -> IVec3 {
        self.block_coords(b)
    }
    pub fn block_at(&self, c: IVec3) -> Option<u32> {
        self.block_index(c)
    }
    pub fn grid(&self) -> (Vec3, IVec3) {
        (self.origin, self.dims)
    }
    /// `-1` inside, `1` outside, `0` stored, per block.
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }
    pub fn stored(&self) -> impl Iterator<Item = (u32, &[f32; N])> {
        self.dense.iter().map(|(b, s)| (*b, &**s))
    }

    /// Add (`carve` false) or remove a soft blob of radius `r` centred at `c`, blended over `k`, and re-extract.
    pub fn dab(&mut self, c: Vec3, r: f32, k: f32, carve: bool) -> DabStats {
        self.modify(c, r, r, k, carve, &|p: Vec3| (p - c).length() - r)
    }

    /// Add or remove a flat disc: an ellipsoid centred at `c`, `r` across and `h` high along `n`, blended over `k`.
    /// Centred on the surface, clay raises it by at most `h` and carving digs at most `h`, however often dabs overlap.
    pub fn dab_disc(&mut self, c: Vec3, n: Vec3, r: f32, h: f32, k: f32, carve: bool) -> DabStats {
        let (r2, h2) = (r * r, h * h);
        self.modify(c, r, h, k, carve, &|p: Vec3| {
            // An ellipsoid's distance, closely approximated (exact on its axes).
            let q = p - c;
            let qn = q.dot(n);
            let qt2 = (q.length_squared() - qn * qn).max(0.0);
            let k0 = (qt2 / r2 + qn * qn / h2).sqrt();
            let k1 = (qt2 / (r2 * r2) + qn * qn / (h2 * h2)).sqrt();
            if k1 < 1e-20 { -h } else { k0 * (k0 - 1.0) / k1 }
        })
    }

    /// Start a stroke: until [`Self::end_stroke`], raycasts and normals see the surface as it is now.
    pub fn begin_stroke(&mut self) {
        self.saved = Some(HashMap::new());
    }

    pub fn end_stroke(&mut self) {
        self.saved = None;
    }

    /// Combine a shape (distance `shape`, within `outer` of `c` and holding the ball of radius `inner`) into the field.
    fn modify(&mut self, c: Vec3, outer: f32, inner: f32, k: f32, carve: bool, shape: &(dyn Fn(Vec3) -> f32 + Sync)) -> DabStats {
        let mut stats = DabStats::default();
        let t = Instant::now();
        let reach = Vec3::splat(outer + k + self.band);
        self.ensure_room(c - reach, c + reach);
        let (touched, swallowed) = self.shell_of(c, outer, inner, k);
        if let Some(saved) = self.saved.as_mut() {
            for &b in touched.iter().chain(&swallowed) {
                saved.entry(b).or_insert_with(|| (self.signs[b as usize], self.dense.get(&b).cloned()));
            }
        }
        stats.blocks_touched = touched.len() + swallowed.len();
        let fill: i8 = if carve { 1 } else { -1 };
        // A swallowed block changes only if it was not already all `fill`.
        let mut changed: Vec<u32> = Vec::new();
        for &b in &swallowed {
            if self.dense.remove(&b).is_some() || self.signs[b as usize] != fill {
                changed.push(b);
            }
            self.signs[b as usize] = fill;
        }
        // Inside/outside blocks the blob's surface crosses become full ones. An inside block stays inside under clay; an
        // outside one stays outside under carving.
        for &b in &touched {
            let sign = self.signs[b as usize];
            if sign != 0 && ((sign > 0 && !carve) || (sign < 0 && carve)) {
                self.dense.insert(b, Box::new([sign as f32 * self.band; N]));
                self.signs[b as usize] = 0;
                stats.new_dense += 1;
            }
        }
        let (voxel, band, origin, dims) = (self.voxel, self.band, self.origin, self.dims);
        // Take the touched blocks out, change them in parallel, put them back.
        let mut blocks: Vec<(u32, Box<[f32; N]>, bool)> = touched.iter().filter_map(|b| self.dense.remove(b).map(|s| (*b, s, false))).collect();
        // During a stroke the shape is blended into the field as it was when the stroke began, then combined with the
        // field as it is: dabbing the same place again changes nothing, where blending into the current field would
        // deepen the blend with every dab.
        let saved = self.saved.as_ref();
        blocks.par_iter_mut().for_each(|(b, s, moved)| {
            let bc = IVec3::new(*b as i32 % dims.x, (*b as i32 / dims.x) % dims.y, *b as i32 / (dims.x * dims.y));
            let lo = origin + (bc * B).as_vec3() * voxel;
            let before = saved.and_then(|m| m.get(b));
            for (i, v) in s.iter_mut().enumerate() {
                let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                let p = lo + l.as_vec3() * voxel;
                let blob = shape(p);
                let base = before.map_or(*v, |(sign, samples)| samples.as_ref().map_or(*sign as f32 * band, |o| o[i]));
                let next = (if carve { smax(base, -blob, k).max(*v) } else { smin(base, blob, k).min(*v) }).clamp(-band, band);
                if next != *v {
                    *v = next;
                    *moved = true;
                }
            }
        });
        for (b, s, moved) in blocks {
            self.dense.insert(b, s);
            if moved {
                changed.push(b);
            }
        }
        // Only the blocks whose samples changed (and the ones below them, whose cells read those samples) are rebuilt.
        let touched = changed;
        stats.field_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut dirty: Vec<u32> = touched.iter().flat_map(|&b| self.lower_neighbours(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        self.rebuild(&dirty);
        stats.extract_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.blocks_rebuilt = dirty.len();
        for b in &dirty {
            if let Some(m) = self.meshes.get(b) {
                stats.verts += m.verts.len();
                stats.quads += m.quads.len();
            }
        }
        stats.upload_bytes = stats.verts * 8 + stats.quads * 6 * 4;
        stats
    }

    /// The whole surface as one mesh, for pictures and checks.
    pub fn mesh(&self) -> crate::mesh::PolyMesh {
        let mut offset: HashMap<u32, u32> = HashMap::new();
        let mut positions = Vec::new();
        for (&b, m) in &self.meshes {
            offset.insert(b, positions.len() as u32);
            positions.extend_from_slice(&m.verts);
        }
        let faces = self
            .meshes
            .values()
            .flat_map(|m| m.quads.iter())
            .map(|q| q.map(|(b, c)| offset[&b] + self.meshes[&b].cell[c as usize]))
            .collect();
        crate::mesh::PolyMesh { positions, faces }
    }
}

// ---------------------------------------------------------------------------------------------
// SPIKE 2 (measurement only): a coarse level for large brushes.
//
// A large brush changes broad form, so it can work on a coarse copy of the field (4x the lattice step, 64x fewer
// samples) while the stroke is under way, and the viewport can show that coarse surface: a brush that size is used
// zoomed out, where the difference does not show. When the stroke ends the fine field is rebuilt once: the new coarse
// shape plus the old fine detail, carried along with the surface as it moved, so detail rides on the change the way it
// does in multiresolution sculpting.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    pub field_ms: f64,
    pub extract_ms: f64,
    pub blocks_checked: usize,
    pub blocks_rebuilt: usize,
}

impl Volume {
    /// A copy of the field without its surface, to remember how it was before a stroke.
    pub fn field_copy(&self) -> Volume {
        Volume { voxel: self.voxel, band: self.band, origin: self.origin, dims: self.dims, signs: self.signs.clone(), dense: self.dense.clone(), meshes: HashMap::new(), changed: Default::default(), saved: None }
    }

    fn sample(&self, i: IVec3) -> f32 {
        let Some(b) = self.block_index(i.div_euclid(IVec3::splat(B))) else { return self.band };
        match self.dense.get(&b) {
            Some(s) => {
                let l = i.rem_euclid(IVec3::splat(B));
                s[(l.z * 64 + l.y * 8 + l.x) as usize]
            }
            None => self.signs[b as usize] as f32 * self.band,
        }
    }

    /// The field at any point, interpolated between lattice samples.
    pub fn value_at(&self, p: Vec3) -> f32 {
        let g = (p - self.origin) / self.voxel;
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let s = |dx: i32, dy: i32, dz: i32| self.sample(i + IVec3::new(dx, dy, dz));
        let x00 = s(0, 0, 0) + (s(1, 0, 0) - s(0, 0, 0)) * f.x;
        let x10 = s(0, 1, 0) + (s(1, 1, 0) - s(0, 1, 0)) * f.x;
        let x01 = s(0, 0, 1) + (s(1, 0, 1) - s(0, 0, 1)) * f.x;
        let x11 = s(0, 1, 1) + (s(1, 1, 1) - s(0, 1, 1)) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        y0 + (y1 - y0) * f.z
    }

    /// The lattice samples around the box `[lo, lo + size]`, gathered once (5 per axis covers a fine block of up to two
    /// of this field's cells, plus one more for a gradient), so a fine block reads its coarse field without lookups.
    fn local(&self, lo: Vec3) -> Local {
        let start = ((lo - self.origin) / self.voxel).floor().as_ivec3();
        let mut v = [0.0f32; 125];
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    v[(z * 25 + y * 5 + x) as usize] = self.sample(start + IVec3::new(x, y, z));
                }
            }
        }
        Local { v, origin: self.origin + start.as_vec3() * self.voxel, voxel: self.voxel }
    }

    /// Rebuild this (fine) field inside `region` after a stroke changed the coarse field from `old_coarse` to
    /// `new_coarse`: the new coarse shape plus this field's old detail (its difference from the old coarse shape),
    /// carried along the surface normal by however far the coarse surface moved.
    pub fn refine_from(&mut self, old_coarse: &Volume, new_coarse: &Volume, region: (Vec3, Vec3)) -> RefineStats {
        let mut stats = RefineStats::default();
        let t = Instant::now();
        let step = self.voxel * B as f32;
        let lo = ((region.0 - self.origin) / step).floor().as_ivec3();
        let hi = ((region.1 - self.origin) / step).floor().as_ivec3();
        let mut candidates = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    if let Some(b) = self.block_index(IVec3::new(x, y, z)) {
                        candidates.push(b);
                    }
                }
            }
        }
        stats.blocks_checked = candidates.len();
        let reach = 0.5 * B as f32 * self.voxel * 3f32.sqrt() + self.band + new_coarse.band;
        let band = self.band;
        let this = &*self;
        // New samples for each block that can hold surface now or did before: computed from the old field, written after.
        let rebuilt: Vec<Refilled> = candidates
            .par_iter()
            .filter_map(|&b| {
                let lo = this.block_origin(b);
                let centre = lo + Vec3::splat(0.5 * (B - 1) as f32 * this.voxel);
                let was_dense = this.dense.contains_key(&b);
                let near_now = new_coarse.value_at(centre).abs() < reach;
                let moved = (new_coarse.value_at(centre) - old_coarse.value_at(centre)).abs() > 1e-7;
                if !moved || (!was_dense && !near_now) {
                    return None;
                }
                let mut s = Box::new([0.0f32; N]);
                let (mut any_in, mut any_out) = (false, false);
                let (new_local, old_local) = (new_coarse.local(lo), old_coarse.local(lo));
                for (i, slot) in s.iter_mut().enumerate() {
                    let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                    let p = lo + l.as_vec3() * this.voxel;
                    let (cn, grad) = new_local.value_and_gradient(p);
                    let v = if cn.abs() > 2.0 * band {
                        // Far from the new surface: the detail (at most a band thick) cannot bring it back.
                        cn.signum() * band
                    } else {
                        let co = old_local.value_and_gradient(p).0;
                        // Where this point's surface came from before the stroke.
                        let from = p + grad.normalize_or_zero() * (cn - co);
                        let fine_old = this.value_at(from);
                        let detail = if fine_old.abs() < band { fine_old - old_coarse.value_at(from) } else { 0.0 };
                        (cn + detail).clamp(-band, band)
                    };
                    any_in |= v < 0.0;
                    any_out |= v >= 0.0;
                    *slot = v;
                }
                Some(if any_in && any_out { (b, Some(s), 0) } else { (b, None, if any_in { -1 } else { 1 }) })
            })
            .collect();
        let mut changed = Vec::with_capacity(rebuilt.len());
        for (b, s, sign) in rebuilt {
            changed.push(b);
            match s {
                Some(s) => {
                    self.dense.insert(b, s);
                    self.signs[b as usize] = 0;
                }
                None => {
                    self.dense.remove(&b);
                    self.signs[b as usize] = sign;
                }
            }
        }
        stats.field_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut dirty: Vec<u32> = changed.iter().flat_map(|&b| self.lower_neighbours(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        self.rebuild(&dirty);
        stats.extract_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.blocks_rebuilt = dirty.len();
        stats
    }
}

/// A small cube of one field's samples, for reading it many times without lookups.
struct Local {
    v: [f32; 125],
    origin: Vec3,
    voxel: f32,
}

impl Local {
    /// Trilinear value and its gradient at `p`, which must lie inside the cube.
    fn value_and_gradient(&self, p: Vec3) -> (f32, Vec3) {
        let g = ((p - self.origin) / self.voxel).clamp(Vec3::ZERO, Vec3::splat(3.999));
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let s = |dx: i32, dy: i32, dz: i32| self.v[((i.z + dz) * 25 + (i.y + dy) * 5 + (i.x + dx)) as usize];
        let (c000, c100, c010, c110) = (s(0, 0, 0), s(1, 0, 0), s(0, 1, 0), s(1, 1, 0));
        let (c001, c101, c011, c111) = (s(0, 0, 1), s(1, 0, 1), s(0, 1, 1), s(1, 1, 1));
        let x00 = c000 + (c100 - c000) * f.x;
        let x10 = c010 + (c110 - c010) * f.x;
        let x01 = c001 + (c101 - c001) * f.x;
        let x11 = c011 + (c111 - c011) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        let value = y0 + (y1 - y0) * f.z;
        let dx = ((c100 - c000) * (1.0 - f.y) + (c110 - c010) * f.y) * (1.0 - f.z) + ((c101 - c001) * (1.0 - f.y) + (c111 - c011) * f.y) * f.z;
        let dy = (x10 - x00) * (1.0 - f.z) + (x11 - x01) * f.z;
        let dz = y1 - y0;
        (value, Vec3::new(dx, dy, dz) / self.voxel)
    }
}

// ---------------------------------------------------------------------------------------------
// For the app's volume playground: what changed, each block's surface, and finding the surface under the cursor.
// ---------------------------------------------------------------------------------------------

impl Volume {
    /// Blocks whose surface was rebuilt since the last call.
    pub fn drain_changed(&mut self) -> Vec<u32> {
        let mut out: Vec<u32> = self.changed.drain().collect();
        out.sort_unstable();
        out
    }

    /// Block `b`'s surface: for each of its 512 cells the index of its vertex in `verts` (or `u32::MAX`), the
    /// vertices, and the quads whose lattice edges start in it. `None` if the block holds no surface.
    pub fn surface_of(&self, b: u32) -> Option<(&[u32], &[Vec3], &[Quad])> {
        self.meshes.get(&b).filter(|m| !m.verts.is_empty()).map(|m| (m.cell.as_slice(), m.verts.as_slice(), m.quads.as_slice()))
    }

    /// The box the lattice covers.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.origin, self.origin + (self.dims * B).as_vec3() * self.voxel)
    }

    /// Make the lattice cover the box `[lo, hi]` with a block to spare, growing it (with room to keep growing) when it
    /// does not. Growing renumbers every block and rebuilds the whole surface: rare, and the price of an unbounded solid.
    pub fn ensure_room(&mut self, lo: Vec3, hi: Vec3) -> bool {
        let step = self.voxel * B as f32;
        let need_lo = ((lo - self.origin) / step).floor().as_ivec3() - IVec3::ONE;
        let need_hi = ((hi - self.origin) / step).floor().as_ivec3() + IVec3::ONE;
        if need_lo.cmpge(IVec3::ZERO).all() && need_hi.cmplt(self.dims).all() {
            return false;
        }
        let spare = IVec3::splat((self.dims.max_element() / 4).max(4));
        let shift = IVec3::select(need_lo.cmplt(IVec3::ZERO), spare - need_lo, IVec3::ZERO);
        let grow_hi = IVec3::select(need_hi.cmpge(self.dims), need_hi + IVec3::ONE - self.dims + spare, IVec3::ZERO);
        let dims = self.dims + shift + grow_hi;
        let remap = |old: &Volume, b: u32| {
            let c = old.block_coords(b) + shift;
            ((c.z * dims.y + c.y) * dims.x + c.x) as u32
        };
        // Everything past the old lattice is outside, as the old lattice's border was.
        let mut signs = vec![1i8; (dims.x * dims.y * dims.z) as usize];
        for (b, &sign) in self.signs.iter().enumerate() {
            signs[remap(self, b as u32) as usize] = sign;
        }
        let dense = std::mem::take(&mut self.dense).into_iter().map(|(b, s)| (remap(self, b), s)).collect();
        let saved = self.saved.take().map(|m| m.into_iter().map(|(b, s)| (remap(self, b), s)).collect());
        self.signs = signs;
        self.dense = dense;
        self.saved = saved;
        self.origin -= shift.as_vec3() * step;
        self.dims = dims;
        self.meshes.clear();
        self.changed.clear();
        self.extract_all();
        true
    }

    /// A sample as it was when the stroke began (as it is now, between strokes).
    fn base_sample(&self, i: IVec3) -> f32 {
        if let Some(saved) = &self.saved
            && let Some(b) = self.block_index(i.div_euclid(IVec3::splat(B)))
            && let Some((sign, samples)) = saved.get(&b)
        {
            let l = i.rem_euclid(IVec3::splat(B));
            return samples.as_ref().map_or(*sign as f32 * self.band, |s| s[(l.z * 64 + l.y * 8 + l.x) as usize]);
        }
        self.sample(i)
    }

    /// The field at `p` as it was when the stroke began, interpolated.
    pub fn base_value_at(&self, p: Vec3) -> f32 {
        let g = (p - self.origin) / self.voxel;
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let s = |dx: i32, dy: i32, dz: i32| self.base_sample(i + IVec3::new(dx, dy, dz));
        let x00 = s(0, 0, 0) + (s(1, 0, 0) - s(0, 0, 0)) * f.x;
        let x10 = s(0, 1, 0) + (s(1, 1, 0) - s(0, 1, 0)) * f.x;
        let x01 = s(0, 0, 1) + (s(1, 0, 1) - s(0, 0, 1)) * f.x;
        let x11 = s(0, 1, 1) + (s(1, 1, 1) - s(0, 1, 1)) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        y0 + (y1 - y0) * f.z
    }

    /// Outward normal at `p` of the surface as it was when the stroke began.
    pub fn normal_at(&self, p: Vec3) -> Vec3 {
        self.gradient_at(p)
    }

    /// `p` moved onto the surface as it was when the stroke began, along that surface's normal.
    pub fn onto_base(&self, mut p: Vec3) -> Vec3 {
        for _ in 0..3 {
            let v = self.base_value_at(p);
            if v.abs() >= self.band * 0.999 {
                break;
            }
            p -= self.gradient_at(p) * v;
        }
        p
    }

    fn gradient_at(&self, p: Vec3) -> Vec3 {
        let h = self.voxel;
        let d = |o: Vec3| self.base_value_at(p + o) - self.base_value_at(p - o);
        Vec3::new(d(Vec3::X * h), d(Vec3::Y * h), d(Vec3::Z * h)).normalize_or_zero()
    }

    /// Where a ray from `origin` along `dir` (normalised) first enters the solid (as it was when the stroke began),
    /// with the surface normal there.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        let (lo, hi) = self.bounds();
        // Clip the ray to the lattice's box.
        let inv = dir.recip();
        let (t0, t1) = ((lo - origin) * inv, (hi - origin) * inv);
        let (near, far) = (t0.min(t1).max_element().max(0.0), t0.max(t1).min_element());
        if near > far {
            return None;
        }
        // March by the field (a lower bound on the distance), at least a third of a step at a time.
        let mut t = near;
        let mut prev = t;
        let mut v = self.base_value_at(origin + dir * t);
        if v < 0.0 {
            return None;
        }
        while t < far {
            prev = t;
            t += v.max(0.3 * self.voxel);
            v = self.base_value_at(origin + dir * t);
            if v < 0.0 {
                break;
            }
        }
        if v >= 0.0 {
            return None;
        }
        // Narrow down the crossing.
        let (mut a, mut b) = (prev, t);
        for _ in 0..12 {
            let m = 0.5 * (a + b);
            if self.base_value_at(origin + dir * m) < 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        let p = origin + dir * b;
        Some((p, self.gradient_at(p)))
    }
}

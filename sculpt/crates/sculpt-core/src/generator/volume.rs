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
type Quad = [(u32, u16); 4];
/// A block as first built: its index, inside (-1) / outside (1) / stored (0), and its samples when stored.
type Filled = (u32, i8, Option<Box<[f32; N]>>);

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
}

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
        let band = 2.5 * voxel;
        let blocks = ((2.0 * half / voxel) / B as f32).ceil() as i32 + 2;
        let dims = IVec3::splat(blocks);
        let origin = Vec3::splat(-half - voxel * B as f32);
        let mut v = Volume { voxel, band, origin, dims, signs: vec![1; (blocks as usize).pow(3)], dense: HashMap::new(), meshes: HashMap::new() };
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

    /// Add (`carve` false) or remove a soft blob of radius `r` centred at `c`, blended over `k`, and re-extract.
    pub fn dab(&mut self, c: Vec3, r: f32, k: f32, carve: bool) -> DabStats {
        let mut stats = DabStats::default();
        let t = Instant::now();
        let reach = r + k + self.band;
        let lo = ((c - Vec3::splat(reach) - self.origin) / (self.voxel * B as f32)).floor().as_ivec3();
        let hi = ((c + Vec3::splat(reach) - self.origin) / (self.voxel * B as f32)).floor().as_ivec3();
        // Only blocks the blob's own surface passes through can change sample by sample. Blocks wholly inside the blob
        // just become solid (clay) or empty (carving), and blocks out of its reach are left alone: a thin shell of
        // blocks instead of a ball of them.
        let shell = self.band + k;
        let mut touched = Vec::new();
        let mut swallowed = Vec::new();
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
                    if far < r - shell {
                        swallowed.push(b);
                    } else {
                        touched.push(b);
                    }
                }
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
        blocks.par_iter_mut().for_each(|(b, s, moved)| {
            let bc = IVec3::new(*b as i32 % dims.x, (*b as i32 / dims.x) % dims.y, *b as i32 / (dims.x * dims.y));
            let lo = origin + (bc * B).as_vec3() * voxel;
            for (i, v) in s.iter_mut().enumerate() {
                let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                let p = lo + l.as_vec3() * voxel;
                let blob = (p - c).length() - r;
                let next = (if carve { smax(*v, -blob, k) } else { smin(*v, blob, k) }).clamp(-band, band);
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

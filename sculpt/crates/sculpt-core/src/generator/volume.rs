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
//! * **Extraction.** Surface nets per block, manifold the way `remesh.rs` is: a cell's surface is split into pieces (its
//!   crossing edges linked across the cell's faces), each with its own vertex, and checkerboard faces are removed from
//!   the stored field first. A dab rebuilds the blocks it changed plus the neighbours whose cells read those samples.
//!
//! It is not wired to the app or the document. GPU upload is reported as bytes, not timed.

use std::collections::HashMap;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::remesh::{EDGE_OF, FACES, corner, find, union};

const B: i32 = 8;
const N: usize = 512;

/// A quad's corners, each as (block, cell | edge << 9): the vertex of the surface piece holding that cube edge of that
/// cell. Naming the cell rather than a vertex number keeps a quad valid when the block it points into is rebuilt.
pub type Quad = [(u32, u16); 4];

/// Sign changes per block, one bit per sample.
type SignMask = [u64; 8];
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
    /// Finished strokes, oldest first: each block a stroke changed, as it was before. Undoing one swaps it back in and
    /// moves what it replaced to `redo`.
    history: Vec<HashMap<u32, Saved>>,
    redo: Vec<HashMap<u32, Saved>>,
}

/// Bytes of stroke history kept before the oldest strokes are forgotten.
const HISTORY_BYTES: usize = 1 << 30;

/// Surface pieces with at most this many vertices, closed and left alone by a stroke, are specks: removed from the field
/// when the stroke ends. A few voxels across; anything bigger is real form and stays.
pub const SPECK_VERTS: usize = 64;

/// A block as it was: its sign, and its samples when stored.
type Saved = (i8, Option<Box<[f32; N]>>);

/// The surface inside one block: a vertex per surface piece of each crossed cell, and the quads whose lattice edges
/// start in this block.
#[derive(Default)]
struct BlockMesh {
    verts: Vec<Vec3>,
    /// Per cell: its first vertex, or `u32::MAX` if no surface crosses it.
    cell: Vec<u32>,
    /// Cells holding more than one piece, sorted: which piece each of the cell's 12 edges belongs to, 2 bits each.
    multi: Vec<(u16, u32)>,
    /// Each corner as (block, cell | edge << 9); resolved to vertices when the mesh is assembled or uploaded.
    quads: Vec<Quad>,
}

impl BlockMesh {
    /// The vertex of the piece holding cube edge `edge` of cell `cell`.
    fn vertex(&self, cell: usize, edge: usize) -> Option<u16> {
        let first = *self.cell.get(cell)?;
        if first == u32::MAX {
            return None;
        }
        let piece = self.multi.binary_search_by_key(&(cell as u16), |m| m.0).map_or(0, |i| (self.multi[i].1 >> (2 * edge)) & 3);
        Some((first + piece) as u16)
    }
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

/// How much of a clay dab's height each dab adds once the first layer is down.
pub const BUILD_STEP: f32 = 0.05;

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
        let mut v = Volume { voxel, band, origin, dims, signs: vec![1; (blocks as usize).pow(3)], dense: HashMap::new(), meshes: HashMap::new(), changed: Default::default(), saved: None, history: Vec::new(), redo: Vec::new() };
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
        self.clear_checkerboards(all.iter().map(|&b| (b, [u64::MAX; 8])).collect());
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
        // The blocks just above point into these ones, whose vertex numbering a rebuild changes: their quads still name
        // the same cells, but a viewport holding vertex numbers must redo them.
        for &b in blocks {
            let c = self.block_coords(b);
            let above: Vec<u32> = (1..8).filter_map(|k| self.block_index(c + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).filter(|u| self.meshes.contains_key(u)).collect();
            self.changed.extend(above);
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
            // Which cube edges cross, and how they join into pieces across the cell's faces (as in `remesh.rs`).
            let crossing: [bool; 12] = std::array::from_fn(|e| (d[EDGES[e][0]] < 0.0) != (d[EDGES[e][1]] < 0.0));
            let mut parent: [u8; 12] = std::array::from_fn(|i| i as u8);
            for face in FACES {
                let edge = |k: usize| EDGE_OF[face[k]][face[(k + 1) % 4]] as usize;
                let (mut on, mut count) = ([0usize; 4], 0);
                for k in 0..4 {
                    if crossing[edge(k)] {
                        on[count] = k;
                        count += 1;
                    }
                }
                match count {
                    2 => union(&mut parent, edge(on[0]), edge(on[1])),
                    // A checkerboard face (one the field cleanup missed): each inside corner cut off by itself, the
                    // same choice in both cells sharing the face.
                    4 => {
                        for k in 0..4 {
                            if d[face[k]] < 0.0 {
                                union(&mut parent, edge((k + 3) % 4), edge(k));
                            }
                        }
                    }
                    _ => {}
                }
            }
            let mut piece_of_root = [u8::MAX; 12];
            let (mut pieces, mut packed) = (0usize, 0u32);
            let (mut sum, mut cnt) = ([Vec3::ZERO; 4], [0.0f32; 4]);
            for e in 0..12 {
                if !crossing[e] {
                    continue;
                }
                let root = find(&mut parent, e);
                if piece_of_root[root] == u8::MAX {
                    piece_of_root[root] = pieces as u8;
                    pieces += 1;
                }
                let piece = (piece_of_root[root] as usize).min(3);
                packed |= (piece as u32) << (2 * e);
                let (a, c) = (EDGES[e][0], EDGES[e][1]);
                let (pa, pc) = (corner(a), corner(c));
                let (pa, pc) = (Vec3::new(pa[0] as f32, pa[1] as f32, pa[2] as f32), Vec3::new(pc[0] as f32, pc[1] as f32, pc[2] as f32));
                sum[piece] += pa.lerp(pc, d[a] / (d[a] - d[c]));
                cnt[piece] += 1.0;
            }
            m.cell[i] = m.verts.len() as u32;
            if pieces > 1 {
                m.multi.push((i as u16, packed));
            }
            for p in 0..pieces.min(4) {
                m.verts.push(self.origin + ((base + l).as_vec3() + sum[p] / cnt[p].max(1.0)) * self.voxel);
            }
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
        // Cell `c` (lattice coordinates) and its cube edge `edge`, as a quad corner. A cell around a crossing edge always
        // holds surface, so no lookup is needed.
        let corner_of = |c: IVec3, edge: usize| -> Option<(u32, u16)> {
            let blk = self.block_index(c.div_euclid(IVec3::splat(B)))?;
            let l = c.rem_euclid(IVec3::splat(B));
            Some((blk, (l.z * 64 + l.y * 8 + l.x) as u16 | (edge as u16) << 9))
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
                // The four cells around the lattice edge from `s` along `axis`, and that edge's number in each.
                let around = |q1: i32, q2: i32| {
                    let mut c = s;
                    c[c1] -= 1 - q1;
                    c[c2] -= 1 - q2;
                    let mut lo = [0usize; 3];
                    lo[c1] = (1 - q1) as usize;
                    lo[c2] = (1 - q2) as usize;
                    let k0 = lo[0] | lo[1] << 1 | lo[2] << 2;
                    corner_of(c, EDGE_OF[k0][k0 | 1 << axis] as usize)
                };
                if let (Some(a), Some(bq), Some(c), Some(d)) = (around(0, 0), around(1, 0), around(1, 1), around(0, 1)) {
                    out.push(if f0 < 0.0 { [a, bq, c, d] } else { [a, d, c, bq] });
                }
            }
        }
        out
    }

    /// Remove checkerboard faces (a cell face whose corners alternate in sign, which pieces cannot represent cleanly)
    /// from the stored field, by filling one outside corner of each, repeatedly, as `remesh.rs` does. Only the blocks with
    /// a sample whose sign changed (`flipped`) and the ones below them can have gained one. Returns the blocks whose
    /// samples changed.
    fn clear_checkerboards(&mut self, mut flipped: HashMap<u32, SignMask>) -> Vec<u32> {
        let mut changed: Vec<u32> = Vec::new();
        let fill = -1e-3 * self.voxel;
        for _ in 0..32 {
            // Faces start in the block holding their lowest corner: the flipped blocks and the ones below them.
            let mut look: Vec<u32> = flipped.keys().flat_map(|&b| self.lower_neighbours(b)).collect();
            look.sort_unstable();
            look.dedup();
            let flips: Vec<IVec3> = {
                let me = &*self;
                let mut found: Vec<IVec3> = look
                    .par_iter()
                    .flat_map_iter(|&b| {
                        let base = me.block_coords(b) * B;
                        let g = me.gather(b);
                        let idx = |l: IVec3| (l.z * 81 + l.y * 9 + l.x) as usize;
                        // Inside bits of the 9x9x9 samples, a row of 9 along x per (y, z): a face test then covers a
                        // whole row of faces in a few bit operations.
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
                        let mut out = Vec::new();
                        let mut take = |l: IVec3, u: usize, v: usize| {
                            let mut c = [l; 4];
                            c[1][u] += 1;
                            c[2][u] += 1;
                            c[2][v] += 1;
                            c[3][v] += 1;
                            // Fill an outside corner in a stored block (a uniform one has no samples to change).
                            let stored = |p: &&IVec3| me.block_index((base + **p).div_euclid(IVec3::splat(B))).is_some_and(|nb| me.dense.contains_key(&nb));
                            if let Some(p) = c.iter().filter(stored).find(|p| g[idx(**p)] >= 0.0) {
                                out.push(base + *p);
                            }
                        };
                        for z in 0..8 {
                            for y in 0..8 {
                                let a = rows[z][y];
                                let (ny, nz, nyz) = (rows[z][y + 1], rows[z + 1][y], rows[z + 1][y + 1]);
                                // Faces in the x-y plane: corners (x, y), (x+1, y), (x+1, y+1), (x, y+1).
                                let xy = (a ^ (a >> 1)) & !((a >> 1) ^ ny) & !(a ^ (ny >> 1)) & 0xFF;
                                // In the z-x plane: corners (z, x), (z+1, x), (z+1, x+1), (z, x+1).
                                let zx = (a ^ nz) & !(a ^ (nz >> 1)) & !(nz ^ (a >> 1)) & 0xFF;
                                // In the y-z plane, for every x at once: (y, z), (y+1, z), (y+1, z+1), (y, z+1).
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
                found.sort_unstable_by_key(|p| (p.z, p.y, p.x));
                found.dedup();
                found
            };
            if flips.is_empty() {
                break;
            }
            flipped.clear();
            for p in flips {
                let Some(b) = self.block_index(p.div_euclid(IVec3::splat(B))) else { continue };
                self.keep(b);
                let l = p.rem_euclid(IVec3::splat(B));
                let i = (l.z * 64 + l.y * 8 + l.x) as usize;
                if let Some(s) = self.dense.get_mut(&b) {
                    s[i] = fill;
                }
                flipped.entry(b).or_insert([0; 8])[i / 64] |= 1 << (i % 64);
                changed.push(b);
            }
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// During a stroke, remember block `b` as it was before the stroke first changes it.
    fn keep(&mut self, b: u32) {
        if let Some(saved) = self.saved.as_mut() {
            saved.entry(b).or_insert_with(|| (self.signs[b as usize], self.dense.get(&b).cloned()));
        }
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

    /// Add or remove a low cap on the surface at `c` (outward normal `n`): `r` in radius and `h` high (or deep), its
    /// rim meeting the surface at a shallow angle so strokes blend into each other, blended over `k`. Clay raises the
    /// surface by at most `h` and carving digs at most `h`, however often dabs overlap.
    pub fn dab_disc(&mut self, c: Vec3, n: Vec3, r: f32, h: f32, k: f32, carve: bool) -> DabStats {
        // A sphere through the rim and the top, cut to the brush's reach so it cannot poke out of thin parts.
        let big = (r * r + h * h) / (2.0 * h);
        let centre = if carve { c + n * (big - h) } else { c - n * (big - h) };
        let reach = 1.05 * r;
        self.modify(c, reach, h, k, carve, &|p: Vec3| ((p - centre).length() - big).max((p - c).length() - reach))
    }

    /// The surface's overall normal under a brush of radius `r` at `c` (as it was when the stroke began): the plane
    /// through rings of surface points around `c`, found along `n`. Heights vary little over small bumps, so the plane
    /// does not tip with every one the brush passes over, as the normals there would.
    pub fn area_normal(&self, c: Vec3, n: Vec3, r: f32) -> Vec3 {
        let (t1, t2) = n.any_orthonormal_pair();
        let mut sum = Vec3::ZERO;
        for (ring, weight) in [(0.5, 0.6), (0.8, 1.0), (1.0, 0.8)] {
            let pts: Option<Vec<Vec3>> = (0..12)
                .map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 12.0;
                    self.along_onto_base(c + (t1 * a.cos() + t2 * a.sin()) * ring * r, n, r)
                })
                .collect();
            // A ring that leaves the surface (over an edge) says nothing about the plane.
            let Some(pts) = pts else { continue };
            // The ring's polygon normal (Newell's method), which a plane fit through its points matches.
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

    /// Where the line through `p` along `n` meets the surface as it was when the stroke began, within `limit` of `p`.
    fn along_onto_base(&self, p: Vec3, n: Vec3, limit: f32) -> Option<Vec3> {
        self.along_onto(p, n, limit, true)
    }

    /// Where the line through `p` along `n` meets the surface (as it was when the stroke began, or as it is now).
    fn along_onto(&self, mut p: Vec3, n: Vec3, limit: f32, base: bool) -> Option<Vec3> {
        let start = p;
        for _ in 0..48 {
            let v = if base { self.base_value_at(p) } else { self.value_at(p) };
            if v.abs() < 0.01 * self.voxel {
                break;
            }
            p -= n * v;
            if (p - start).length() > limit {
                return None;
            }
        }
        Some(p)
    }

    /// One dab of a plane brush, the way the clay and trim brushes of mesh sculpting work. The brush plane takes its
    /// tilt from the surface as it was when the stroke began (so it follows the original form and holds steady), and
    /// its height from the surface as it is now, under the brush.
    ///
    /// * Clay (`trim` false) lifts the plane `h` above the surface's average height under the brush and fills
    ///   everything below it: cavities fill first, bumps above the plane are left alone, and passes over the same place
    ///   keep building, since each one sees what the stroke has added already.
    /// * Trim lowers the plane `h` below that average and cuts away everything above it: it flattens.
    ///
    /// Each dab moves the surface `alpha` of the way to the plane, fully out to `hardness` of the radius and easing to
    /// nothing at the rim, so edges are square at high hardness and soft at low, and overlapping dabs blend smoothly.
    /// `along` is the rough normal near `at`. Returns the plane's normal.
    #[allow(clippy::too_many_arguments)]
    pub fn plane_dab(&mut self, at: Vec3, along: Vec3, r: f32, h: f32, hardness: f32, alpha: f32, trim: bool) -> Option<Vec3> {
        let on = self.onto_base(at, along);
        if self.normal_at(on) == Vec3::ZERO {
            return None;
        }
        let first = self.area_normal(on, along, r);
        let n = self.area_normal(on, first, r);
        // Heights under the brush along the plane's normal, from the point brushed: of the surface the stroke began on,
        // and of what the stroke has added since.
        let (t1, t2) = n.any_orthonormal_pair();
        let (mut base_sum, mut added_sum, mut weight) = (0.0, 0.0, 0.0);
        for (ring, w, count) in [(0.0, 1.0, 1), (0.5, 0.8, 8), (0.9, 0.5, 12)] {
            for k in 0..count {
                let a = k as f32 * std::f32::consts::TAU / count as f32;
                let p = on + (t1 * a.cos() + t2 * a.sin()) * ring * r;
                if let (Some(b), Some(c)) = (self.along_onto(p, n, r, true), self.along_onto(p, n, r, false)) {
                    base_sum += (b - on).dot(n) * w;
                    added_sum += (c - b).dot(n) * w;
                    weight += w;
                }
            }
        }
        let (base, added) = if weight > 0.0 { (base_sum / weight, added_sum / weight) } else { (0.0, 0.0) };
        // Over a cavity the original surface's average lies above the point brushed, so clay fills it; over a convex
        // form it lies below, and taking it would leave clay barely building, so the point itself sets the level.
        // Clay then rides on what the stroke has added (it builds up as you scrub); trim does not, so a stroke cuts
        // down to one plane however long you scrub.
        let ground = if trim { base.min(0.0) } else { base.max(0.0) };
        // Clay's first layer comes at once; beyond it each dab adds a small step, so a pass (about a dozen overlapping
        // dabs at the app's spacing) builds about one more layer rather than a dozen.
        let level = if trim { ground } else { ground + added };
        let plane = if trim { ground - h } else { (ground + h).max(level + BUILD_STEP * h) };
        // Each dab moves the surface `alpha` of the way to the plane at its centre, less toward the edge: full out to
        // `hardness` of the radius, easing to nothing at the rim, so dabs blend into each other with no edge of their own.
        let hard = hardness.clamp(0.0, 0.95);
        let depth = 0.35 * r;
        let weight = move |p: Vec3| {
            let q = p - on;
            let u = q.dot(n);
            let t = (q - n * u).length();
            let x = ((t / r - hard) / (1.0 - hard)).clamp(0.0, 1.0);
            alpha * (1.0 - x * x * (3.0 - 2.0 * x))
        };
        let k = (0.5 * h).max(self.voxel);
        let reach = 1.1 * r + depth;
        if trim {
            // Everything above the plane, out to the brush's edge and up to `depth` over the average.
            let shape = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                (plane - u).max(u - (level + depth)).max((q - n * u).length() - r)
            };
            self.modify_in(on, reach, 0.0, k, true, Some((n, plane, level + depth)), Some(&weight), &shape);
        } else {
            // Everything below the plane, out to the brush's edge and down to `depth` under the average: deep enough to
            // fill a cavity, shallow enough not to come out of the far side of a thin part.
            let shape = move |p: Vec3| {
                let q = p - on;
                let u = q.dot(n);
                (u - plane).max((level - depth) - u).max((q - n * u).length() - r)
            };
            self.modify_in(on, reach, 0.0, k, false, Some((n, level - depth, plane)), Some(&weight), &shape);
        }
        Some(n)
    }

    /// Start a stroke: until [`Self::end_stroke`], raycasts and normals see the surface as it is now.
    pub fn begin_stroke(&mut self) {
        self.saved = Some(HashMap::new());
    }

    /// End the stroke: remove the specks it left (see [`SPECK_VERTS`]) and keep it for [`Self::undo`]. Returns the
    /// specks removed.
    pub fn end_stroke(&mut self) -> usize {
        let Some(saved) = self.saved.as_ref() else { return 0 };
        let mut region: Vec<u32> = saved.keys().flat_map(|&b| self.lower_neighbours(b)).collect();
        region.sort_unstable();
        region.dedup();
        let specks = self.remove_specks(&region, SPECK_VERTS);
        let saved = self.saved.take().unwrap_or_default();
        if !saved.is_empty() {
            self.history.push(saved);
            self.redo.clear();
            let size = |m: &HashMap<u32, Saved>| m.values().map(|(_, s)| 8 + s.as_ref().map_or(0, |_| N * 4)).sum::<usize>();
            let mut total: usize = self.history.iter().map(size).sum();
            while total > HISTORY_BYTES && self.history.len() > 1 {
                total -= size(&self.history.remove(0));
            }
        }
        specks
    }

    /// Undo the last stroke. False if there is none.
    pub fn undo(&mut self) -> bool {
        self.saved = None;
        let Some(entry) = self.history.pop() else { return false };
        let back = self.swap_in(entry);
        self.redo.push(back);
        true
    }

    /// Redo the last undone stroke. False if there is none.
    pub fn redo(&mut self) -> bool {
        self.saved = None;
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

    /// Put `entry`'s blocks back and rebuild their surface; returns what they replaced.
    fn swap_in(&mut self, entry: HashMap<u32, Saved>) -> HashMap<u32, Saved> {
        let mut back = HashMap::with_capacity(entry.len());
        for (b, (sign, samples)) in entry {
            back.insert(b, (self.signs[b as usize], self.dense.remove(&b)));
            self.signs[b as usize] = sign;
            if let Some(s) = samples {
                self.dense.insert(b, s);
            }
        }
        let mut dirty: Vec<u32> = back.keys().flat_map(|&b| self.lower_neighbours(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        self.rebuild(&dirty);
        back
    }

    /// Remove the specks among the surface of `region` (blocks already rebuilt): closed pieces with at most `max_verts`
    /// vertices, every one inside `region`. Each is removed from the field, by giving the samples it encloses the sign
    /// around them, so no rebuild brings it back. Returns how many were removed.
    pub fn remove_specks(&mut self, region: &[u32], max_verts: usize) -> usize {
        let inside: std::collections::HashSet<u32> = region.iter().copied().filter(|b| self.meshes.contains_key(b)).collect();
        // Number the vertices of the region's blocks, and join them along quads.
        let mut first: HashMap<u32, u32> = HashMap::new();
        let mut total = 0u32;
        for &b in &inside {
            first.insert(b, total);
            total += self.meshes[&b].verts.len() as u32;
        }
        let mut parent: Vec<u32> = (0..total).collect();
        let mut open = vec![false; total as usize];
        fn root(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        let node = |b: u32, c: u16, me: &Volume| -> Option<u32> { Some(first.get(&b)? + me.vertex_of(b, c)? as u32) };
        // Quads of the region's blocks, and of the blocks just above it (which point into it).
        let mut quad_blocks: Vec<u32> = inside.iter().copied().collect();
        for &b in &inside {
            let c = self.block_coords(b);
            quad_blocks.extend((1..8).filter_map(|k| self.block_index(c + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).filter(|u| !inside.contains(u) && self.meshes.contains_key(u)));
        }
        quad_blocks.sort_unstable();
        quad_blocks.dedup();
        for &qb in &quad_blocks {
            for q in &self.meshes[&qb].quads {
                let ids: Vec<Option<u32>> = q.iter().map(|&(b, c)| node(b, c, self)).collect();
                // A quad reaching outside the region (or owned by a block outside it) joins its piece to the rest of
                // the surface: that piece is not a speck.
                let leaves = !inside.contains(&qb) || ids.iter().any(|i| i.is_none());
                let known: Vec<u32> = ids.into_iter().flatten().collect();
                for w in known.windows(2) {
                    let (a, c) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
                    if a != c {
                        parent[a.max(c) as usize] = a.min(c);
                        let o = open[a as usize] || open[c as usize];
                        open[a.min(c) as usize] = o;
                    }
                }
                if leaves && let Some(&k) = known.first() {
                    let r = root(&mut parent, k);
                    open[r as usize] = true;
                }
            }
        }
        // Pieces by size.
        let mut size: HashMap<u32, usize> = HashMap::new();
        for i in 0..total {
            let r = root(&mut parent, i);
            *size.entry(r).or_default() += 1;
        }
        let specks: std::collections::HashSet<u32> = size.iter().filter(|&(&r, &n)| n <= max_verts && !open[r as usize]).map(|(&r, _)| r).collect();
        if specks.is_empty() {
            return 0;
        }
        // Which piece each crossed cell's vertices belong to.
        let mut cells_of: HashMap<u32, Vec<IVec3>> = HashMap::new();
        let mut piece_at: HashMap<IVec3, Vec<u32>> = HashMap::new();
        for &b in &inside {
            let m = &self.meshes[&b];
            let base = self.block_coords(b) * B;
            // Vertices are numbered in cell order, so a cell's run ends where the next crossed cell's begins.
            let crossed: Vec<(usize, u32)> = m.cell.iter().enumerate().filter(|&(_, &f)| f != u32::MAX).map(|(i, &f)| (i, f)).collect();
            for (k, &(i, f)) in crossed.iter().enumerate() {
                let next = crossed.get(k + 1).map_or(m.verts.len() as u32, |&(_, n)| n);
                let c = base + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                for v in f..next {
                    let r = root(&mut parent, first[&b] + v);
                    piece_at.entry(c).or_default().push(r);
                    if specks.contains(&r) {
                        cells_of.entry(r).or_default().push(c);
                    }
                }
            }
        }
        let mut edits: Vec<(IVec3, f32)> = Vec::new();
        for (r, cells) in &cells_of {
            // The samples at its cells' corners: the side with fewer is what the speck encloses.
            let mut corners: Vec<IVec3> = cells.iter().flat_map(|&c| (0..8).map(move |k| c + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).collect();
            corners.sort_unstable_by_key(|p| (p.z, p.y, p.x));
            corners.dedup();
            let neg = corners.iter().filter(|&&p| self.sample(p) < 0.0).count();
            let pos = corners.len() - neg;
            if neg == pos {
                continue;
            }
            let enclosed_inside = neg < pos;
            let to = if enclosed_inside { 0.5 * self.voxel } else { -0.5 * self.voxel };
            for &p in &corners {
                if (self.sample(p) < 0.0) != enclosed_inside {
                    continue;
                }
                // Leave a sample that another piece's cell also uses.
                let shared = (0..8).any(|k| {
                    let c = p - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1);
                    piece_at.get(&c).is_some_and(|ps| ps.iter().any(|q| q != r))
                });
                if !shared {
                    edits.push((p, to));
                }
            }
        }
        let mut flipped: HashMap<u32, SignMask> = HashMap::new();
        for (p, to) in edits {
            let Some(b) = self.block_index(p.div_euclid(IVec3::splat(B))) else { continue };
            if !self.dense.contains_key(&b) {
                continue;
            }
            self.keep(b);
            let l = p.rem_euclid(IVec3::splat(B));
            let i = (l.z * 64 + l.y * 8 + l.x) as usize;
            self.dense.get_mut(&b).unwrap()[i] = to;
            flipped.entry(b).or_insert([0; 8])[i / 64] |= 1 << (i % 64);
        }
        let mut changed: Vec<u32> = flipped.keys().copied().collect();
        changed.extend(self.clear_checkerboards(flipped));
        let mut dirty: Vec<u32> = changed.iter().flat_map(|&b| self.lower_neighbours(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        self.rebuild(&dirty);
        cells_of.len()
    }

    /// Write the field (not the surface or history) to `w`: a small header, the per-block signs run-length coded, and
    /// each stored block's samples at 16 bits.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"SVOL")?;
        w.write_all(&1u32.to_le_bytes())?;
        for f in [self.voxel, self.band, self.origin.x, self.origin.y, self.origin.z] {
            w.write_all(&f.to_le_bytes())?;
        }
        for d in self.dims.to_array() {
            w.write_all(&d.to_le_bytes())?;
        }
        let mut runs: Vec<(i8, u32)> = Vec::new();
        for &s in &self.signs {
            match runs.last_mut() {
                Some((v, n)) if *v == s => *n += 1,
                _ => runs.push((s, 1)),
            }
        }
        w.write_all(&(runs.len() as u32).to_le_bytes())?;
        for (v, n) in runs {
            w.write_all(&[v as u8])?;
            w.write_all(&n.to_le_bytes())?;
        }
        let mut stored: Vec<u32> = self.dense.keys().copied().collect();
        stored.sort_unstable();
        w.write_all(&(stored.len() as u32).to_le_bytes())?;
        let mut buf = Vec::with_capacity(N * 2);
        for b in stored {
            w.write_all(&b.to_le_bytes())?;
            buf.clear();
            for &v in self.dense[&b].iter() {
                // Kept on the same side of zero: the surface must not move by a voxel through rounding.
                let mut q = (v / self.band * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
                if v < 0.0 && q == 0 {
                    q = -1;
                }
                buf.extend_from_slice(&q.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
        Ok(())
    }

    /// Read a field written by [`Self::write_to`] and extract its surface.
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Volume> {
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
        if u32_(r)? != 1 {
            return Err(bad("unknown volume file version"));
        }
        let mut f = [0f32; 5];
        for v in &mut f {
            *v = f32::from_bits(u32_(r)?);
        }
        let mut dims = [0i32; 3];
        for d in &mut dims {
            *d = u32_(r)? as i32;
        }
        let dims = IVec3::from_array(dims);
        let total = (dims.x as usize) * (dims.y as usize) * (dims.z as usize);
        if dims.min_element() <= 0 || total > 1 << 31 {
            return Err(bad("bad lattice size"));
        }
        let mut signs = Vec::with_capacity(total);
        for _ in 0..u32_(r)? {
            let mut v = [0u8; 1];
            r.read_exact(&mut v)?;
            let n = u32_(r)? as usize;
            if signs.len() + n > total {
                return Err(bad("sign runs overflow the lattice"));
            }
            signs.extend(std::iter::repeat_n(v[0] as i8, n));
        }
        if signs.len() != total {
            return Err(bad("sign runs do not cover the lattice"));
        }
        let band = f[1];
        let mut dense = HashMap::new();
        let mut buf = vec![0u8; N * 2];
        for _ in 0..u32_(r)? {
            let b = u32_(r)?;
            if b as usize >= total {
                return Err(bad("block out of range"));
            }
            r.read_exact(&mut buf)?;
            let mut s = Box::new([0f32; N]);
            for (i, v) in s.iter_mut().enumerate() {
                *v = i16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]) as f32 / 32767.0 * band;
            }
            dense.insert(b, s);
        }
        let mut v = Volume { voxel: f[0], band, origin: Vec3::new(f[2], f[3], f[4]), dims, signs, dense, meshes: HashMap::new(), changed: Default::default(), saved: None, history: Vec::new(), redo: Vec::new() };
        v.extract_all();
        Ok(v)
    }


    /// Combine a shape (distance `shape`, within `outer` of `c` and holding the ball of radius `inner`) into the field.
    fn modify(&mut self, c: Vec3, outer: f32, inner: f32, k: f32, carve: bool, shape: &(dyn Fn(Vec3) -> f32 + Sync)) -> DabStats {
        self.modify_in(c, outer, inner, k, carve, None, None, shape)
    }

    /// [`Self::modify`] for a shape that also lies within the slab `lo..hi` along `n` from `c`: only the blocks near it
    /// are visited. With `weight`, each sample moves only that fraction of the way to the combined field (as a brush
    /// with falloff moves a mesh part of the way): overlapping dabs then blend into one smooth change instead of each
    /// leaving its own edge.
    #[allow(clippy::too_many_arguments)]
    fn modify_in(
        &mut self,
        c: Vec3,
        outer: f32,
        inner: f32,
        k: f32,
        carve: bool,
        slab: Option<(Vec3, f32, f32)>,
        weight: Option<&(dyn Fn(Vec3) -> f32 + Sync)>,
        shape: &(dyn Fn(Vec3) -> f32 + Sync),
    ) -> DabStats {
        let mut stats = DabStats::default();
        let t = Instant::now();
        let reach = Vec3::splat(outer + k + self.band);
        self.ensure_room(c - reach, c + reach);
        let (mut touched, swallowed) = self.shell_of(c, outer, inner, k);
        if let Some((n, lo, hi)) = slab {
            let margin = 0.5 * B as f32 * self.voxel * 3f32.sqrt() + self.band + k;
            let half = 0.5 * (B - 1) as f32 * self.voxel;
            touched.retain(|&b| {
                let u = (self.block_origin(b) + Vec3::splat(half) - c).dot(n);
                u > lo - margin && u < hi + margin
            });
        }
        for &b in touched.iter().chain(&swallowed) {
            self.keep(b);
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
        let mut blocks: Vec<(u32, Box<[f32; N]>, bool, SignMask)> = touched.iter().filter_map(|b| self.dense.remove(b).map(|s| (*b, s, false, [0; 8]))).collect();
        // During a stroke the shape is blended into the field as it was when the stroke began, then combined with the
        // field as it is: dabbing the same place again changes nothing, where blending into the current field would
        // deepen the blend with every dab.
        let saved = self.saved.as_ref();
        blocks.par_iter_mut().for_each(|(b, s, moved, signs)| {
            let bc = IVec3::new(*b as i32 % dims.x, (*b as i32 / dims.x) % dims.y, *b as i32 / (dims.x * dims.y));
            let lo = origin + (bc * B).as_vec3() * voxel;
            let before = saved.and_then(|m| m.get(b));
            for (i, v) in s.iter_mut().enumerate() {
                let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                let p = lo + l.as_vec3() * voxel;
                let next = if let Some(weight) = weight {
                    let w = weight(p);
                    if w <= 0.0 {
                        continue;
                    }
                    let blob = shape(p);
                    let full = if carve { v.max(-blob) } else { v.min(blob) };
                    (*v + w * (full - *v)).clamp(-band, band)
                } else {
                    let blob = shape(p);
                    let base = before.map_or(*v, |(sign, samples)| samples.as_ref().map_or(*sign as f32 * band, |o| o[i]));
                    (if carve { smax(base, -blob, k).max(*v) } else { smin(base, blob, k).min(*v) }).clamp(-band, band)
                };
                if next != *v {
                    if (next < 0.0) != (*v < 0.0) {
                        signs[i / 64] |= 1 << (i % 64);
                    }
                    *v = next;
                    *moved = true;
                }
            }
        });
        let mut flipped: HashMap<u32, SignMask> = HashMap::new();
        for (b, s, moved, signs) in blocks {
            self.dense.insert(b, s);
            if moved {
                changed.push(b);
            }
            if signs.iter().any(|&m| m != 0) {
                flipped.insert(b, signs);
            }
        }
        // Only the blocks whose samples changed (and the ones below them, whose cells read those samples) are rebuilt,
        // after the checkerboard faces the change made are filled.
        changed.extend(self.clear_checkerboards(flipped));
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
            .filter_map(|q| {
                let v: Option<Vec<u32>> = q.iter().map(|&(b, c)| Some(offset[&b] + self.vertex_of(b, c)? as u32)).collect();
                v.map(|v| [v[0], v[1], v[2], v[3]])
            })
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
        Volume { voxel: self.voxel, band: self.band, origin: self.origin, dims: self.dims, signs: self.signs.clone(), dense: self.dense.clone(), meshes: HashMap::new(), changed: Default::default(), saved: None, history: Vec::new(), redo: Vec::new() }
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

    /// The vertex of block `b` that quad corner `corner` (cell | edge << 9) names.
    pub fn vertex_of(&self, b: u32, corner: u16) -> Option<u16> {
        self.meshes.get(&b)?.vertex((corner & 511) as usize, (corner >> 9) as usize)
    }

    /// Block `b`'s surface: its vertices, and the quads whose lattice edges start in it, each corner as (block,
    /// cell | edge << 9), which [`Self::vertex_of`] resolves. `None` if the block holds no surface.
    pub fn surface_of(&self, b: u32) -> Option<(&[Vec3], &[Quad])> {
        self.meshes.get(&b).filter(|m| !m.verts.is_empty()).map(|m| (m.verts.as_slice(), m.quads.as_slice()))
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
        let history: Vec<HashMap<u32, Saved>> = std::mem::take(&mut self.history).into_iter().map(|m| m.into_iter().map(|(b, s)| (remap(self, b), s)).collect()).collect();
        let redo: Vec<HashMap<u32, Saved>> = std::mem::take(&mut self.redo).into_iter().map(|m| m.into_iter().map(|(b, s)| (remap(self, b), s)).collect()).collect();
        self.history = history;
        self.redo = redo;
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
    /// Far from the surface (where the stored distance is clamped) it first walks along `dir`, the rough normal.
    pub fn onto_base(&self, mut p: Vec3, dir: Vec3) -> Vec3 {
        for _ in 0..32 {
            let v = self.base_value_at(p);
            if v.abs() < self.band * 0.999 {
                break;
            }
            p -= dir * v;
        }
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

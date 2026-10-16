//! Remeshing a signed distance field into a clean quad surface.
//!
//! This is the extractor behind a Remesh checkpoint. It is surface nets with two changes that matter for sculpting:
//!
//! * **Manifold by construction.** A cell whose corners form more than one inside region (a thin wall, two surfaces
//!   passing through one voxel) gets one vertex per region instead of one shared vertex, with inside corners joined
//!   along cube edges only. That rule makes every edge of the output belong to exactly two quads.
//! * **Sparse.** The lattice is cut into blocks of samples. A block whose centre is farther from the surface than the
//!   block can reach is never sampled and stores one value, the centre's. A distance-like field, which is what the
//!   generators produce, is mostly far from the surface, so evaluation, memory and the extraction passes all follow the
//!   surface area, not the volume, and run in parallel over the blocks that matter.
//!
//! The field is any `Fn(Vec3) -> f32`, negative inside. It should be roughly distance-like (gradient near 1);
//! [`RemeshParams::lipschitz`] says how far it may exceed that, which sets how conservatively blocks are skipped.

use glam::Vec3;
use rayon::prelude::*;

use crate::geom::Aabb;
use crate::mesh::{Face, PolyMesh};

/// Samples per block edge for the sparse evaluation.
const BLOCK: usize = 8;
/// Cells owned by one block (a cell belongs to the block holding its lowest corner).
const BLOCK_CELLS: usize = BLOCK * BLOCK * BLOCK;

#[derive(Clone, Copy, Debug)]
pub struct RemeshParams {
    /// Distance between lattice samples, in world units.
    pub voxel: f32,
    /// Upper bound on the field's gradient length. 1 for an exact distance; noise or sharp blends need more.
    pub lipschitz: f32,
}

impl RemeshParams {
    pub fn new(voxel: f32) -> RemeshParams {
        RemeshParams { voxel, lipschitz: 1.5 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Remeshed {
    pub positions: Vec<Vec3>,
    pub quads: Vec<[u32; 4]>,
    /// Field evaluations made, against the lattice's total sample count.
    pub evaluated: usize,
    pub total: usize,
}

impl Remeshed {
    pub fn into_poly_mesh(self) -> PolyMesh {
        let faces: Vec<Face> = self.quads;
        PolyMesh { positions: self.positions, faces }
    }
}

/// Number of edges not shared by exactly two quads: 0 for a closed surface where no edge is torn or overloaded.
pub fn non_manifold_edges(quads: &[[u32; 4]]) -> usize {
    let mut uses = std::collections::HashMap::new();
    for q in quads {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

// Cube corners are numbered by bits: x = bit 0, y = bit 1, z = bit 2.
const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

fn corner(i: usize) -> [usize; 3] {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1]
}

/// What one block's cells produced: vertices, and for each cell its first vertex (block-local) and corner regions.
struct BlockCells {
    positions: Vec<Vec3>,
    start: Vec<u32>,
    regions: Vec<u16>,
}

/// Extract the zero surface of `f` inside `bounds`. Anything that touches the bounds is left open there, so the
/// bounds must enclose the shape with a margin.
pub fn remesh<F: Fn(Vec3) -> f32 + Sync>(f: &F, bounds: Aabb, params: &RemeshParams) -> Remeshed {
    let vs = params.voxel;
    let n = ((bounds.extent() / vs).ceil().as_uvec3() + glam::UVec3::splat(2)).to_array().map(|v| v as usize);
    let total = n[0] * n[1] * n[2];
    let pos = |x: usize, y: usize, z: usize| bounds.min + Vec3::new(x as f32, y as f32, z as f32) * vs;

    // ---- sparse field evaluation, one block of samples at a time
    let nb = [n[0].div_ceil(BLOCK), n[1].div_ceil(BLOCK), n[2].div_ceil(BLOCK)];
    let block_count = nb[0] * nb[1] * nb[2];
    let reach = 0.5 * BLOCK as f32 * vs * 3f32.sqrt() * params.lipschitz;
    let block_origin = |b: usize| [(b % nb[0]) * BLOCK, ((b / nb[0]) % nb[1]) * BLOCK, (b / (nb[0] * nb[1])) * BLOCK];
    let block_dims = |lo: [usize; 3]| [(lo[0] + BLOCK).min(n[0]) - lo[0], (lo[1] + BLOCK).min(n[1]) - lo[1], (lo[2] + BLOCK).min(n[2]) - lo[2]];
    // A skipped block holds one value; a sampled block holds all of its samples, x fastest.
    let blocks: Vec<Vec<f32>> = (0..block_count)
        .into_par_iter()
        .map(|b| {
            let lo = block_origin(b);
            let dims = block_dims(lo);
            let centre = pos(lo[0], lo[1], lo[2]) + Vec3::splat(0.5 * BLOCK as f32 * vs);
            let c = f(centre);
            if c.abs() > reach {
                return vec![c];
            }
            let mut vals = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        vals.push(f(pos(lo[0] + x, lo[1] + y, lo[2] + z)));
                    }
                }
            }
            vals
        })
        .collect();
    let evaluated: usize = blocks.iter().map(|v| if v.len() == 1 { 1 } else { v.len() + 1 }).sum();
    let sample = |x: usize, y: usize, z: usize| -> f32 {
        let b = ((z / BLOCK) * nb[1] + y / BLOCK) * nb[0] + x / BLOCK;
        let v = &blocks[b];
        if v.len() == 1 {
            return v[0];
        }
        let lo = block_origin(b);
        let dims = block_dims(lo);
        v[((z - lo[2]) * dims[1] + (y - lo[1])) * dims[0] + (x - lo[0])]
    };

    // A block matters if it or a block just above it (in any axis) was sampled: those hold every cell that has a
    // corner in a sampled block. Two skipped blocks side by side cannot straddle the surface.
    let sampled = |b: usize| blocks[b].len() > 1;
    let active: Vec<usize> = (0..block_count)
        .into_par_iter()
        .filter(|&b| {
            let (bx, by, bz) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
            (0..8).any(|k| {
                let (x, y, z) = (bx + (k & 1), by + ((k >> 1) & 1), bz + ((k >> 2) & 1));
                x < nb[0] && y < nb[1] && z < nb[2] && sampled((z * nb[1] + y) * nb[0] + x)
            })
        })
        .collect();
    let mut slot_of = vec![u32::MAX; block_count];
    for (s, &b) in active.iter().enumerate() {
        slot_of[b] = s as u32;
    }

    // ---- one vertex per inside region of each cell, in parallel over active blocks
    let cell_limit = [n[0] - 1, n[1] - 1, n[2] - 1];
    let cells: Vec<BlockCells> = active
        .par_iter()
        .map(|&b| {
            let lo = block_origin(b);
            let mut out = BlockCells { positions: Vec::new(), start: vec![u32::MAX; BLOCK_CELLS], regions: vec![0; BLOCK_CELLS] };
            for lz in 0..BLOCK.min(cell_limit[2].saturating_sub(lo[2])) {
                for ly in 0..BLOCK.min(cell_limit[1].saturating_sub(lo[1])) {
                    for lx in 0..BLOCK.min(cell_limit[0].saturating_sub(lo[0])) {
                        let (x, y, z) = (lo[0] + lx, lo[1] + ly, lo[2] + lz);
                        let d: [f32; 8] = std::array::from_fn(|i| {
                            let c = corner(i);
                            sample(x + c[0], y + c[1], z + c[2])
                        });
                        let inside: u32 = (0..8).filter(|&i| d[i] < 0.0).fold(0, |m, i| m | 1 << i);
                        if inside == 0 || inside == 0xFF {
                            continue;
                        }
                        // Join inside corners that share a cube edge; each group is one region of the solid in this cell.
                        let mut region = [u8::MAX; 8];
                        let mut regions = 0u8;
                        for i in 0..8 {
                            if inside >> i & 1 == 0 || region[i] != u8::MAX {
                                continue;
                            }
                            let mut stack = vec![i];
                            region[i] = regions;
                            while let Some(c) = stack.pop() {
                                for e in EDGES {
                                    let other = if e[0] == c {
                                        e[1]
                                    } else if e[1] == c {
                                        e[0]
                                    } else {
                                        continue;
                                    };
                                    if inside >> other & 1 == 1 && region[other] == u8::MAX {
                                        region[other] = regions;
                                        stack.push(other);
                                    }
                                }
                            }
                            regions += 1;
                        }
                        let mut sum = [Vec3::ZERO; 4];
                        let mut cnt = [0.0f32; 4];
                        for e in EDGES {
                            let (a, bb) = (e[0], e[1]);
                            if (d[a] < 0.0) == (d[bb] < 0.0) {
                                continue;
                            }
                            let (ins, outc) = if d[a] < 0.0 { (a, bb) } else { (bb, a) };
                            let t = d[ins] / (d[ins] - d[outc]);
                            let (pi, po) = (corner(ins), corner(outc));
                            let p = Vec3::new(pi[0] as f32, pi[1] as f32, pi[2] as f32).lerp(Vec3::new(po[0] as f32, po[1] as f32, po[2] as f32), t);
                            let r = region[ins] as usize;
                            sum[r] += p;
                            cnt[r] += 1.0;
                        }
                        let local = (lz * BLOCK + ly) * BLOCK + lx;
                        out.start[local] = out.positions.len() as u32;
                        let mut packed = 0u16;
                        for (i, &r) in region.iter().enumerate() {
                            if r != u8::MAX {
                                packed |= (r as u16) << (2 * i);
                            }
                        }
                        out.regions[local] = packed;
                        let origin = pos(x, y, z);
                        for (s, c) in sum.iter().zip(cnt).take(regions as usize) {
                            out.positions.push(origin + *s / c.max(1.0) * vs);
                        }
                    }
                }
            }
            out
        })
        .collect();

    // Global vertex numbering: blocks in order.
    let mut offset = Vec::with_capacity(cells.len());
    let mut positions: Vec<Vec3> = Vec::with_capacity(cells.iter().map(|c| c.positions.len()).sum());
    for c in &cells {
        offset.push(positions.len() as u32);
        positions.extend_from_slice(&c.positions);
    }
    // The vertex of the region holding corner `corner_index` of cell (x, y, z), or `None` if that cell has no surface.
    let vertex_of = |x: usize, y: usize, z: usize, corner_index: usize| -> Option<u32> {
        let b = ((z / BLOCK) * nb[1] + y / BLOCK) * nb[0] + x / BLOCK;
        let slot = slot_of[b];
        if slot == u32::MAX {
            return None;
        }
        let local = ((z % BLOCK) * BLOCK + y % BLOCK) * BLOCK + x % BLOCK;
        let c = &cells[slot as usize];
        let start = c.start[local];
        (start != u32::MAX).then(|| offset[slot as usize] + start + ((c.regions[local] >> (2 * corner_index)) & 3) as u32)
    };

    // ---- one quad per lattice edge that crosses the surface, in parallel over active blocks
    let quads: Vec<[u32; 4]> = active
        .par_iter()
        .flat_map_iter(|&b| {
            let lo = block_origin(b);
            let dims = block_dims(lo);
            let mut out: Vec<[u32; 4]> = Vec::new();
            for z in lo[2]..lo[2] + dims[2] {
                for y in lo[1]..lo[1] + dims[1] {
                    for x in lo[0]..lo[0] + dims[0] {
                        if x == 0 || y == 0 || z == 0 {
                            continue;
                        }
                        let f0 = sample(x, y, z);
                        for axis in 0..3 {
                            let (nx, ny, nz) = (x + (axis == 0) as usize, y + (axis == 1) as usize, z + (axis == 2) as usize);
                            if nx >= n[0] || ny >= n[1] || nz >= n[2] || x >= cell_limit[0] || y >= cell_limit[1] || z >= cell_limit[2] {
                                continue;
                            }
                            let f1 = sample(nx, ny, nz);
                            if (f0 < 0.0) == (f1 < 0.0) {
                                continue;
                            }
                            let (c1, c2) = ((axis + 1) % 3, (axis + 2) % 3);
                            // The cell at offset (q1, q2) around the edge, and the vertex of the region holding the inside end.
                            let vertex = |q1: usize, q2: usize| -> Option<u32> {
                                let mut p = [x, y, z];
                                p[c1] -= 1 - q1;
                                p[c2] -= 1 - q2;
                                let mut local = [0usize; 3];
                                local[c1] = 1 - q1;
                                local[c2] = 1 - q2;
                                local[axis] = if f0 < 0.0 { 0 } else { 1 };
                                vertex_of(p[0], p[1], p[2], local[0] | local[1] << 1 | local[2] << 2)
                            };
                            if let (Some(a), Some(bq), Some(c), Some(d)) = (vertex(0, 0), vertex(1, 0), vertex(1, 1), vertex(0, 1)) {
                                out.push(if f0 < 0.0 { [a, bq, c, d] } else { [a, d, c, bq] });
                            }
                        }
                    }
                }
            }
            out
        })
        .collect();
    Remeshed { positions, quads, evaluated, total }
}

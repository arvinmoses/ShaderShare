//! Remeshing a signed distance field into a clean quad surface.
//!
//! This is the extractor behind a Remesh checkpoint. It is surface nets with two changes that matter for sculpting:
//!
//! * **Manifold by construction.** A cell whose corners form more than one inside region (a thin wall, two surfaces
//!   passing through one voxel) gets one vertex per region instead of one shared vertex, with inside corners joined
//!   along cube edges only. That rule makes every edge of the output belong to exactly two quads.
//! * **Sparse.** The field is evaluated in blocks, and a block whose centre is farther from the surface than the block
//!   can reach is skipped and filled with the centre's value. A distance-like field, which is what the generators
//!   produce, is mostly far from the surface, so the cost follows the surface area, not the volume.
//!
//! The field is any `Fn(Vec3) -> f32`, negative inside. It should be roughly distance-like (gradient near 1);
//! [`RemeshParams::lipschitz`] says how far it may exceed that, which sets how conservatively blocks are skipped.

use glam::Vec3;
use rayon::prelude::*;

use crate::geom::Aabb;
use crate::mesh::{Face, PolyMesh};

/// Samples per block edge for the sparse evaluation.
const BLOCK: usize = 8;

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

/// Extract the zero surface of `f` inside `bounds`. Anything that touches the bounds is left open there, so the
/// bounds must enclose the shape with a margin.
pub fn remesh<F: Fn(Vec3) -> f32 + Sync>(f: &F, bounds: Aabb, params: &RemeshParams) -> Remeshed {
    let vs = params.voxel;
    let n = ((bounds.extent() / vs).ceil().as_uvec3() + glam::UVec3::splat(2)).to_array().map(|v| v as usize);
    let total = n[0] * n[1] * n[2];
    let at = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let pos = |x: usize, y: usize, z: usize| bounds.min + Vec3::new(x as f32, y as f32, z as f32) * vs;

    // ---- sparse field evaluation, one block of samples at a time
    let nb = [n[0].div_ceil(BLOCK), n[1].div_ceil(BLOCK), n[2].div_ceil(BLOCK)];
    let reach = 0.5 * BLOCK as f32 * vs * 3f32.sqrt() * params.lipschitz;
    let block_range = |b: usize| {
        let (bx, by, bz) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
        let lo = [bx * BLOCK, by * BLOCK, bz * BLOCK];
        let hi = [(lo[0] + BLOCK).min(n[0]), (lo[1] + BLOCK).min(n[1]), (lo[2] + BLOCK).min(n[2])];
        (lo, hi)
    };
    let blocks: Vec<(usize, Vec<f32>)> = (0..nb[0] * nb[1] * nb[2])
        .into_par_iter()
        .map(|b| {
            let (lo, hi) = block_range(b);
            let centre = pos(lo[0], lo[1], lo[2]) + Vec3::splat(0.5 * BLOCK as f32 * vs);
            let c = f(centre);
            if c.abs() > reach {
                return (b, vec![c]);
            }
            let mut vals = Vec::with_capacity((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        vals.push(f(pos(x, y, z)));
                    }
                }
            }
            (b, vals)
        })
        .collect();
    let mut vals = vec![0.0f32; total];
    let mut evaluated = 0;
    for (b, v) in blocks {
        let (lo, hi) = block_range(b);
        evaluated += v.len() + usize::from(v.len() == 1);
        let mut k = 0;
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    vals[at(x, y, z)] = if v.len() == 1 {
                        v[0]
                    } else {
                        k += 1;
                        v[k - 1]
                    };
                }
            }
        }
    }

    // ---- one vertex per inside region of each cell
    let (cx, cy, cz) = (n[0] - 1, n[1] - 1, n[2] - 1);
    let cidx = |x: usize, y: usize, z: usize| (z * cy + y) * cx + x;
    // For each cell: index of its first vertex, and the region of each corner (2 bits each).
    let mut cell_start = vec![u32::MAX; cx * cy * cz];
    let mut cell_regions = vec![0u16; cx * cy * cz];
    let mut positions: Vec<Vec3> = Vec::new();
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                let d: [f32; 8] = std::array::from_fn(|i| {
                    let c = corner(i);
                    vals[at(x + c[0], y + c[1], z + c[2])]
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
                    let (a, b) = (e[0], e[1]);
                    if (d[a] < 0.0) == (d[b] < 0.0) {
                        continue;
                    }
                    let (ins, out) = if d[a] < 0.0 { (a, b) } else { (b, a) };
                    let t = d[ins] / (d[ins] - d[out]);
                    let (pi, po) = (corner(ins), corner(out));
                    let p = Vec3::new(pi[0] as f32, pi[1] as f32, pi[2] as f32).lerp(Vec3::new(po[0] as f32, po[1] as f32, po[2] as f32), t);
                    let r = region[ins] as usize;
                    sum[r] += p;
                    cnt[r] += 1.0;
                }
                cell_start[cidx(x, y, z)] = positions.len() as u32;
                let mut packed = 0u16;
                for i in 0..8 {
                    if region[i] != u8::MAX {
                        packed |= (region[i] as u16) << (2 * i);
                    }
                }
                cell_regions[cidx(x, y, z)] = packed;
                let origin = pos(x, y, z);
                for r in 0..regions as usize {
                    positions.push(origin + sum[r] / cnt[r].max(1.0) * vs);
                }
            }
        }
    }

    // ---- one quad per lattice edge that crosses the surface
    let mut quads = Vec::new();
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
                    // The cell at offset (q1, q2) around the edge, and the vertex of the region holding the inside end.
                    let vertex = |q1: usize, q2: usize| -> u32 {
                        let mut p = [x, y, z];
                        p[c1] -= 1 - q1;
                        p[c2] -= 1 - q2;
                        let ci = cidx(p[0], p[1], p[2]);
                        let mut local = [0usize; 3];
                        local[c1] = 1 - q1;
                        local[c2] = 1 - q2;
                        local[axis] = if f0 < 0.0 { 0 } else { 1 };
                        let corner_index = local[0] | local[1] << 1 | local[2] << 2;
                        cell_start[ci] + ((cell_regions[ci] >> (2 * corner_index)) & 3) as u32
                    };
                    let q = [vertex(0, 0), vertex(1, 0), vertex(1, 1), vertex(0, 1)];
                    quads.push(if f0 < 0.0 { q } else { [q[0], q[3], q[2], q[1]] });
                }
            }
        }
    }
    Remeshed { positions, quads, evaluated, total }
}

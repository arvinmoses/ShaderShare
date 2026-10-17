//! Remeshing a signed distance field into a clean quad surface.
//!
//! This is the extractor behind a Remesh checkpoint. It is surface nets with two changes that matter for sculpting:
//!
//! * **Manifold by construction.** Each cell's surface is split into pieces: crossing edges are linked when they lie on
//!   the same cell face and the face's arc joins them, and a face whose four edges all cross (the checkerboard case) is
//!   always resolved by cutting off each inside corner. Both cells on either side of a face see the same arcs, so the
//!   pieces match across faces, and each piece gets its own vertex. That makes every edge of the output belong to
//!   exactly two quads, including where two surfaces pass through one voxel.
//! * **No checkerboard faces.** A cell face whose four corners alternate in sign is the one place a surface can pass a
//!   face twice, which one vertex per piece cannot represent. Before extraction one sample of each such face is flipped,
//!   repeatedly, which changes the surface only at voxel scale (a pocket of a single sample is filled or dropped).
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

/// Even out the quads: each pass slides every vertex part of the way to the middle of its neighbours, but only within
/// the surface's tangent plane, so the shape stays put while thin quads and nearly coincident vertices (where the
/// surface passes close to a lattice point) spread out. Thin quads are what a brush flips inside out first.
pub fn relax(mesh: &mut Remeshed, passes: u32) {
    let n = mesh.positions.len();
    if n == 0 || passes == 0 {
        return;
    }
    // Neighbours along quad edges, and the quads around each vertex for its normal.
    let mut rings: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (qi, q) in mesh.quads.iter().enumerate() {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            rings[a as usize].push(b);
            rings[b as usize].push(a);
            around[a as usize].push(qi as u32);
        }
    }
    for r in &mut rings {
        r.sort_unstable();
        r.dedup();
    }
    for _ in 0..passes {
        let p = &mesh.positions;
        let quads = &mesh.quads;
        let next: Vec<Vec3> = (0..n)
            .into_par_iter()
            .map(|v| {
                let ring = &rings[v];
                if ring.is_empty() {
                    return p[v];
                }
                let normal = around[v]
                    .iter()
                    .map(|&qi| {
                        let q = quads[qi as usize];
                        (p[q[2] as usize] - p[q[0] as usize]).cross(p[q[3] as usize] - p[q[1] as usize])
                    })
                    .sum::<Vec3>()
                    .normalize_or_zero();
                let mean = ring.iter().map(|&u| p[u as usize]).sum::<Vec3>() / ring.len() as f32;
                let step = mean - p[v];
                p[v] + (step - normal * step.dot(normal)) * 0.5
            })
            .collect();
        mesh.positions = next;
    }
}

/// Drop connected pieces with fewer than `min_fraction` of all quads, and the vertices only they used. Returns how many
/// pieces went. The largest piece is always kept.
pub fn drop_small_pieces(mesh: &mut Remeshed, min_fraction: f32) -> usize {
    if min_fraction <= 0.0 || mesh.quads.is_empty() {
        return 0;
    }
    let n = mesh.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for q in &mesh.quads {
        let r0 = root(&mut parent, q[0]);
        for &v in &q[1..] {
            let r = root(&mut parent, v);
            if r != r0 {
                parent[r as usize] = r0;
            }
        }
    }
    let roots: Vec<u32> = mesh.quads.iter().map(|q| root(&mut parent, q[0])).collect();
    let mut size: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &r in &roots {
        *size.entry(r).or_default() += 1;
    }
    let largest = size.values().copied().max().unwrap_or(0);
    let keep_at = ((mesh.quads.len() as f32 * min_fraction).ceil() as usize).min(largest);
    let dropped = size.values().filter(|&&s| s < keep_at).count();
    if dropped == 0 {
        return 0;
    }
    let kept: Vec<[u32; 4]> = mesh.quads.iter().zip(&roots).filter(|(_, r)| size[r] >= keep_at).map(|(q, _)| *q).collect();
    let mut remap = vec![u32::MAX; n];
    let mut positions = Vec::with_capacity(n);
    for q in &kept {
        for &v in q {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = positions.len() as u32;
                positions.push(mesh.positions[v as usize]);
            }
        }
    }
    mesh.quads = kept.into_iter().map(|q| q.map(|v| remap[v as usize])).collect();
    mesh.positions = positions;
    dropped
}

/// The edges not shared by exactly two quads, with how many quads use each, for diagnosing a bad extraction.
pub fn non_manifold_edge_list(quads: &[[u32; 4]]) -> Vec<((u32, u32), u32)> {
    let mut uses = std::collections::HashMap::new();
    for q in quads {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
        }
    }
    uses.into_iter().filter(|&(_, n)| n != 2).collect()
}

// Cube corners are numbered by bits: x = bit 0, y = bit 1, z = bit 2.
const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

fn corner(i: usize) -> [usize; 3] {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1]
}

/// The six cell faces as cycles of corners, so consecutive corners share a cube edge.
const FACES: [[usize; 4]; 6] = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]];

/// Index into [`EDGES`] of the cube edge joining corners `a` and `b`, or 255 if they are not adjacent.
const EDGE_OF: [[u8; 8]; 8] = {
    let mut t = [[255u8; 8]; 8];
    let mut i = 0;
    while i < 12 {
        t[EDGES[i][0]][EDGES[i][1]] = i as u8;
        t[EDGES[i][1]][EDGES[i][0]] = i as u8;
        i += 1;
    }
    t
};

/// Union-find over the 12 cube edges of a cell.
fn find(parent: &mut [u8; 12], mut i: usize) -> usize {
    while parent[i] as usize != i {
        parent[i] = parent[parent[i] as usize];
        i = parent[i] as usize;
    }
    i
}

fn union(parent: &mut [u8; 12], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb) as u8;
    }
}

/// The field value at lattice sample `(x, y, z)`, from block storage (a skipped block holds one value).
fn sample_at(blocks: &[Vec<f32>], nb: [usize; 3], n: [usize; 3], x: usize, y: usize, z: usize) -> f32 {
    let b = ((z / BLOCK) * nb[1] + y / BLOCK) * nb[0] + x / BLOCK;
    let v = &blocks[b];
    if v.len() == 1 {
        return v[0];
    }
    let (lo, w, h) = ((x / BLOCK * BLOCK, y / BLOCK * BLOCK, z / BLOCK * BLOCK), (n[0] - x / BLOCK * BLOCK).min(BLOCK), (n[1] - y / BLOCK * BLOCK).min(BLOCK));
    v[((z - lo.2) * h + (y - lo.1)) * w + (x - lo.0)]
}

/// What one block's cells produced: vertices, and for each cell its first vertex (block-local) and the surface piece
/// each of its 12 edges belongs to (2 bits per edge).
struct BlockCells {
    positions: Vec<Vec3>,
    start: Vec<u32>,
    pieces: Vec<u32>,
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
    let mut blocks: Vec<Vec<f32>> = (0..block_count)
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

    // ---- remove checkerboard faces: flip one sample of each until none is left
    let block_of = |p: &[usize; 3]| ((p[2] / BLOCK) * nb[1] + p[1] / BLOCK) * nb[0] + p[0] / BLOCK;
    for _ in 0..32 {
        let flips: Vec<[usize; 3]> = {
            let blocks = &blocks;
            let mut found: Vec<[usize; 3]> = active
                .par_iter()
                .flat_map_iter(|&b| {
                    let lo = block_origin(b);
                    let dims = block_dims(lo);
                    let mut out = Vec::new();
                    for z in lo[2]..lo[2] + dims[2] {
                        for y in lo[1]..lo[1] + dims[1] {
                            for x in lo[0]..lo[0] + dims[0] {
                                for axis in 0..3 {
                                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                                    let mut c = [[x, y, z]; 4];
                                    c[1][u] += 1;
                                    c[2][u] += 1;
                                    c[2][v] += 1;
                                    c[3][v] += 1;
                                    if c[2][u] >= n[u] || c[2][v] >= n[v] {
                                        continue;
                                    }
                                    let inside = c.map(|p| sample_at(blocks, nb, n, p[0], p[1], p[2]) < 0.0);
                                    if inside[0] == inside[2] && inside[1] == inside[3] && inside[0] != inside[1] {
                                        // Fill an outside corner when we can, so every flip removes outside and the loop must end.
                                        // Only a sampled block can change (a skipped block has one shared value).
                                        let editable = |p: &&[usize; 3]| blocks[block_of(p)].len() > 1;
                                        let pick = c.iter().filter(editable).find(|p| sample_at(blocks, nb, n, p[0], p[1], p[2]) >= 0.0).or_else(|| c.iter().find(editable));
                                        if let Some(p) = pick {
                                            out.push(*p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out
                })
                .collect();
            found.sort_unstable();
            found.dedup();
            found
        };
        if flips.is_empty() {
            break;
        }
        for p in flips {
            let b = block_of(&p);
            let lo = block_origin(b);
            let dims = block_dims(lo);
            let v = &mut blocks[b][((p[2] - lo[2]) * dims[1] + (p[1] - lo[1])) * dims[0] + (p[0] - lo[0])];
            *v = if *v < 0.0 { vs * 1e-3 } else { -vs * 1e-3 };
        }
    }
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


    // ---- one vertex per inside region of each cell, in parallel over active blocks
    let cell_limit = [n[0] - 1, n[1] - 1, n[2] - 1];
    let cells: Vec<BlockCells> = active
        .par_iter()
        .map(|&b| {
            let lo = block_origin(b);
            let mut out = BlockCells { positions: Vec::new(), start: vec![u32::MAX; BLOCK_CELLS], pieces: vec![0; BLOCK_CELLS] };
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
                        // Which cube edges cross the surface, and how their pieces join across each face.
                        let mut crossing = [false; 12];
                        for (e, c) in crossing.iter_mut().enumerate() {
                            *c = (d[EDGES[e][0]] < 0.0) != (d[EDGES[e][1]] < 0.0);
                        }
                        let mut parent: [u8; 12] = std::array::from_fn(|i| i as u8);
                        for face in FACES {
                            let edge = |k: usize| EDGE_OF[face[k]][face[(k + 1) % 4]] as usize;
                            let on: Vec<usize> = (0..4).filter(|&k| crossing[edge(k)]).collect();
                            match on.len() {
                                2 => union(&mut parent, edge(on[0]), edge(on[1])),
                                // The checkerboard face: cut each inside corner off by itself, so the face never
                                // joins two inside corners. Both cells sharing the face make the same choice.
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
                        // One piece per group of linked crossing edges, numbered in edge order.
                        let mut piece_of_root = [u8::MAX; 12];
                        let mut pieces = 0usize;
                        let mut packed = 0u32;
                        let mut sum = [Vec3::ZERO; 4];
                        let mut cnt = [0.0f32; 4];
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
                            let (a, bb) = (EDGES[e][0], EDGES[e][1]);
                            let t = d[a] / (d[a] - d[bb]);
                            let (pa, pb) = (corner(a), corner(bb));
                            sum[piece] += Vec3::new(pa[0] as f32, pa[1] as f32, pa[2] as f32).lerp(Vec3::new(pb[0] as f32, pb[1] as f32, pb[2] as f32), t);
                            cnt[piece] += 1.0;
                        }
                        let regions = pieces.min(4) as u8;
                        let local = (lz * BLOCK + ly) * BLOCK + lx;
                        out.start[local] = out.positions.len() as u32;
                        out.pieces[local] = packed;
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
    // The vertex of the surface piece holding cube edge `edge` of cell (x, y, z), or `None` if that cell has no surface.
    let vertex_of = |x: usize, y: usize, z: usize, edge: usize| -> Option<u32> {
        let b = ((z / BLOCK) * nb[1] + y / BLOCK) * nb[0] + x / BLOCK;
        let slot = slot_of[b];
        if slot == u32::MAX {
            return None;
        }
        let local = ((z % BLOCK) * BLOCK + y % BLOCK) * BLOCK + x % BLOCK;
        let c = &cells[slot as usize];
        let start = c.start[local];
        (start != u32::MAX).then(|| offset[slot as usize] + start + ((c.pieces[local] >> (2 * edge)) & 3))
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
                                let mut low = [0usize; 3];
                                low[c1] = 1 - q1;
                                low[c2] = 1 - q2;
                                let lo = low[0] | low[1] << 1 | low[2] << 2;
                                vertex_of(p[0], p[1], p[2], EDGE_OF[lo][lo | 1 << axis] as usize)
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

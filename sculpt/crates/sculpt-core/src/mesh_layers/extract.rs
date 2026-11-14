//! The volume's display surface as one named mesh.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec3;
use rayon::prelude::*;

use crate::generator::level_volume::{BlockSurface, ExtractStats, HashMap, Key, LevelVolume, Show, key_order};
use crate::mesh::{Face, NO_VERT, PolyMesh, face_len, face_verts};
use crate::{Error, Result};

/// One extraction: the mesh, each vertex's name, and each vertex's normal (the field's gradient, normalized), all in
/// the mesh's vertex order (the canonical order). Also where each vertex sits in its block: see [`super::MeshLayers::blocks`].
pub struct Extracted {
    pub mesh: PolyMesh,
    pub keys: Vec<u64>,
    pub normals: Vec<Vec3>,
    /// Every block, in the mesh's order, with its first vertex (its vertices run to the next block's first).
    pub blocks: Vec<(Key, u32)>,
    /// Per vertex: its index in its block's `BlockSurface::verts` (the GPU's vertex order).
    pub vi: Vec<u16>,
    pub stats: ExtractStats,
}

/// The volume's display surface as one mesh, with each vertex's name. Vertices no face uses are left out.
pub fn extract_named(vol: &LevelVolume, cap: usize) -> Result<(PolyMesh, Vec<u64>, ExtractStats)> {
    let e = extract_all(vol, cap)?;
    Ok((e.mesh, e.keys, e.stats))
}

/// [`extract_named`] with the vertex normals too.
pub fn extract_all(vol: &LevelVolume, cap: usize) -> Result<Extracted> {
    let (surf, stats) = extract_blocks(vol, cap);
    assemble(&surf, stats)
}

/// Every shown block's surface, as [`LevelVolume::extract_in`] gives it.
pub(super) fn extract_blocks(vol: &LevelVolume, cap: usize) -> (HashMap<Key, BlockSurface>, ExtractStats) {
    let work: Vec<Key> = vol.keys();
    vol.extract_in(&work, &Show::upto(cap))
}

/// The mesh of the block surfaces `surf` (every block shown, as a full extraction gives them): blocks in order (level,
/// z, y, x), each block's vertices in its own order; faces as `surface_in` builds them.
pub fn assemble(surf: &HashMap<Key, BlockSurface>, stats: ExtractStats) -> Result<Extracted> {
    let mut blocks: Vec<(Key, &BlockSurface)> = surf.iter().map(|(k, s)| (*k, s)).collect();
    blocks.par_sort_unstable_by_key(|(k, _)| key_order(k));
    for (k, _) in &blocks {
        if k.1.abs().max_element() >= 1 << 15 {
            return Err(Error::InvalidData(format!("block {:?} is too far out to be named", k.1)));
        }
    }
    // Where each block's vertices start, and each block's place in the order (one lookup per quad corner).
    let mut first = Vec::with_capacity(blocks.len() + 1);
    let mut total = 0usize;
    for (_, s) in &blocks {
        first.push(total as u32);
        total += s.verts.len();
    }
    first.push(total as u32);
    let place: HashMap<Key, u32> = blocks.iter().enumerate().map(|(i, (k, _))| (*k, i as u32)).collect();
    // Positions, names and normals, block by block, each written in place.
    let mut positions = vec![Vec3::ZERO; total];
    let mut names = vec![u64::MAX; total];
    let mut normals = vec![Vec3::ZERO; total];
    let mut vi: Vec<u16> = vec![0; total];
    vi.par_chunks_mut(4096).enumerate().for_each(|(c, out)| {
        let at = c * 4096;
        let mut b = first.partition_point(|&f| f as usize <= at) - 1;
        for (k, o) in out.iter_mut().enumerate() {
            while first[b + 1] as usize <= at + k {
                b += 1;
            }
            *o = (at + k - first[b] as usize) as u16;
        }
    });
    {
        let mut parts = Vec::with_capacity(blocks.len());
        let (mut p, mut n, mut m) = (&mut positions[..], &mut names[..], &mut normals[..]);
        for (_, s) in &blocks {
            let len = s.verts.len();
            let (a, rest_p) = p.split_at_mut(len);
            let (b, rest_n) = n.split_at_mut(len);
            let (c, rest_m) = m.split_at_mut(len);
            parts.push((a, b, c));
            (p, n, m) = (rest_p, rest_n, rest_m);
        }
        parts.into_par_iter().zip(blocks.par_iter()).for_each(|((pos, nam, nor), (k, s))| {
            pos.copy_from_slice(&s.verts);
            for (o, g) in nor.iter_mut().zip(&s.normals) {
                *o = g.normalize_or_zero();
            }
            for (cell, &info) in s.cells.iter().enumerate() {
                let count = ((info >> 10) & 7) as usize;
                let start = (info & 1023) as usize;
                for p in 0..count {
                    if let Some(slot) = nam.get_mut(start + p) {
                        *slot = name(*k, cell, p);
                    }
                }
            }
        });
    }
    if names.par_iter().any(|&n| n == u64::MAX) {
        return Err(Error::InvalidData("a surface vertex has no cell".into()));
    }
    // The faces, as `surface_in` builds them.
    let corner = |(b, c, p): &(Key, u16, u8)| -> Option<u32> {
        let i = *place.get(b)? as usize;
        Some(first[i] + blocks[i].1.vertex(*c, *p)? as u32)
    };
    let face_lists: Vec<Vec<Face>> = blocks
        .par_iter()
        .map(|(_, s)| {
            let mut faces = Vec::with_capacity(s.quads.len());
            for q in &s.quads {
                let (Some(a), Some(b), Some(c), Some(d)) = (corner(&q[0]), corner(&q[1]), corner(&q[2]), corner(&q[3])) else { continue };
                let mut uniq = [0u32; 4];
                let mut len = 0usize;
                for v in [a, b, c, d] {
                    if len == 0 || uniq[len - 1] != v && uniq[0] != v {
                        uniq[len] = v;
                        len += 1;
                    }
                }
                match len {
                    4 => faces.push(uniq),
                    3 => faces.push([uniq[0], uniq[1], uniq[2], NO_VERT]),
                    _ => {}
                }
            }
            faces
        })
        .collect();
    let mut faces: Vec<Face> = Vec::with_capacity(face_lists.iter().map(|f| f.len()).sum());
    for f in face_lists {
        faces.extend_from_slice(&f);
    }
    // Drop vertices no face uses (they would be loose in the document).
    let used: Vec<AtomicBool> = (0..total).into_par_iter().map(|_| AtomicBool::new(false)).collect();
    faces.par_iter().for_each(|f| {
        for &v in face_verts(f) {
            used[v as usize].store(true, Ordering::Relaxed);
        }
    });
    if used.par_iter().any(|u| !u.load(Ordering::Relaxed)) {
        let mut remap = vec![NO_VERT; total];
        let mut next = 0u32;
        let (mut p2, mut n2, mut m2, mut v2) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
        let mut b = 0;
        for i in 0..total {
            while b < blocks.len() && first[b] as usize <= i {
                // Block `b` starts here: its first kept vertex is the next one kept.
                first[b] = next;
                b += 1;
            }
            if used[i].load(Ordering::Relaxed) {
                remap[i] = next;
                next += 1;
                p2.push(positions[i]);
                n2.push(names[i]);
                m2.push(normals[i]);
                v2.push(vi[i]);
            }
        }
        for f in &mut first[b..] {
            *f = next;
        }
        faces.par_iter_mut().for_each(|f| {
            for v in f.iter_mut().filter(|v| **v != NO_VERT) {
                *v = remap[*v as usize];
            }
        });
        positions = p2;
        names = n2;
        normals = m2;
        vi = v2;
    }
    let blocks = blocks.iter().zip(&first).map(|((k, _), f)| (*k, *f)).collect();
    Ok(Extracted { mesh: PolyMesh { positions, faces }, keys: names, normals, blocks, vi, stats })
}

/// The block of a vertex name.
pub(super) fn key_of(name: u64) -> Key {
    let c = |s: u32| ((name >> s) & 0xFFFF) as i32 - (1 << 15);
    ((name >> 61) as u8, glam::IVec3::new(c(13), c(29), c(45)))
}

/// A vertex name: level (3 bits), block z, y, x (16 bits each, offset), cell (9 bits), piece (2 bits). Ascending in
/// the extraction order.
fn name(k: Key, cell: usize, piece: usize) -> u64 {
    let o = |x: i32| ((x + (1 << 15)) as u64) & 0xFFFF;
    (k.0 as u64) << 61 | o(k.1.z) << 45 | o(k.1.y) << 29 | o(k.1.x) << 13 | (cell as u64) << 4 | piece as u64
}

/// Edges not shared by exactly two faces (open or non-manifold), and edges two faces walk the same way (flipped).
pub fn edge_defects(faces: &[Face]) -> (usize, usize) {
    let mut uses: std::collections::HashMap<(u32, u32), (u32, u32)> = std::collections::HashMap::with_capacity(faces.len() * 2);
    for f in faces {
        let n = face_len(f);
        for k in 0..n {
            let (a, b) = (f[k], f[(k + 1) % n]);
            let e = uses.entry((a.min(b), a.max(b))).or_insert((0, 0));
            if a < b {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
    let bad = uses.values().filter(|(f, b)| f + b != 2).count();
    let flipped = uses.values().filter(|(f, b)| f + b == 2 && *f != 1).count();
    (bad, flipped)
}

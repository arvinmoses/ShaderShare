//! The volume's display surface as one named mesh.

use glam::Vec3;
use rayon::prelude::*;

use crate::generator::level_volume::{ExtractStats, Key, LevelVolume, Show};
use crate::mesh::{Face, NO_VERT, PolyMesh, face_len, face_verts};
use crate::{Error, Result};

/// The volume's display surface as one mesh, with each vertex's name. Vertices no face uses are left out.
pub fn extract_named(vol: &LevelVolume, cap: usize) -> Result<(PolyMesh, Vec<u64>, ExtractStats)> {
    let v = Show::upto(cap);
    let work: Vec<Key> = vol.keys();
    let (surf, stats) = vol.extract_in(&work, &v);
    let mut blocks: Vec<&Key> = surf.keys().collect();
    blocks.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
    let mut first: std::collections::HashMap<Key, u32> = std::collections::HashMap::with_capacity(blocks.len());
    let mut total = 0usize;
    for k in &blocks {
        first.insert(**k, total as u32);
        total += surf[*k].verts.len();
    }
    // Positions and names, block by block, in parallel.
    let per_block: Vec<(Vec<Vec3>, Vec<u64>)> = blocks
        .par_iter()
        .map(|k| {
            let s = &surf[*k];
            let mut names = vec![u64::MAX; s.verts.len()];
            for (cell, &info) in s.cells.iter().enumerate() {
                let count = ((info >> 10) & 7) as usize;
                let start = (info & 1023) as usize;
                for p in 0..count {
                    if let Some(slot) = names.get_mut(start + p) {
                        *slot = name(**k, cell, p);
                    }
                }
            }
            (s.verts.clone(), names)
        })
        .collect();
    let mut positions = Vec::with_capacity(total);
    let mut names = Vec::with_capacity(total);
    for (p, n) in per_block {
        positions.extend(p);
        names.extend(n);
    }
    if names.iter().any(|&n| n == u64::MAX) {
        return Err(Error::InvalidData("a surface vertex has no cell".into()));
    }
    for k in &blocks {
        if k.1.abs().max_element() >= 1 << 15 {
            return Err(Error::InvalidData(format!("block {:?} is too far out to be named", k.1)));
        }
    }
    // The faces, as `surface_in` builds them.
    let face_lists: Vec<Vec<Face>> = blocks
        .par_iter()
        .map(|k| {
            let mut faces = Vec::with_capacity(surf[*k].quads.len());
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
            faces
        })
        .collect();
    let mut faces: Vec<Face> = face_lists.into_iter().flatten().collect();
    // Drop vertices no face uses (they would be loose in the document).
    let mut used = vec![false; positions.len()];
    for f in &faces {
        for &v in face_verts(f) {
            used[v as usize] = true;
        }
    }
    if used.iter().any(|u| !u) {
        let mut remap = vec![NO_VERT; positions.len()];
        let mut next = 0u32;
        let (mut p2, mut n2) = (Vec::with_capacity(positions.len()), Vec::with_capacity(positions.len()));
        for i in 0..positions.len() {
            if used[i] {
                remap[i] = next;
                next += 1;
                p2.push(positions[i]);
                n2.push(names[i]);
            }
        }
        for f in &mut faces {
            for v in f.iter_mut().filter(|v| **v != NO_VERT) {
                *v = remap[*v as usize];
            }
        }
        positions = p2;
        names = n2;
    }
    Ok((PolyMesh { positions, faces }, names, stats))
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

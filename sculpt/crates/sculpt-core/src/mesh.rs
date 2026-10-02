//! Polygon mesh storage and adjacency.
//!
//! Faces are fixed-size `[u32; 4]`: quads, or triangles with `NO_VERT` in the
//! last slot. Sculpt meshes are quad-dominant (everything above the base level
//! is all-quads after Catmull-Clark), so this keeps faces flat and SIMD/GPU
//! friendly without an n-gon indirection.

use glam::Vec3;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::{Error, Result};

pub const NO_VERT: u32 = u32::MAX;

pub type Face = [u32; 4];

#[inline]
pub fn face_len(f: &Face) -> usize {
    if f[3] == NO_VERT { 3 } else { 4 }
}

#[inline]
pub fn face_verts(f: &Face) -> &[u32] {
    &f[..face_len(f)]
}

/// Area-weighted (unnormalized) face normal.
#[inline]
pub fn face_normal(positions: &[Vec3], f: &Face) -> Vec3 {
    let p0 = positions[f[0] as usize];
    let p1 = positions[f[1] as usize];
    let p2 = positions[f[2] as usize];
    if f[3] == NO_VERT {
        (p1 - p0).cross(p2 - p0)
    } else {
        let p3 = positions[f[3] as usize];
        (p2 - p0).cross(p3 - p1)
    }
}

/// Triangles of a face (fan split for quads).
#[inline]
pub fn face_triangles(f: &Face) -> SmallVec<[[u32; 3]; 2]> {
    let mut t = SmallVec::new();
    t.push([f[0], f[1], f[2]]);
    if f[3] != NO_VERT {
        t.push([f[0], f[2], f[3]]);
    }
    t
}

/// Plain polygon soup used for import/export and construction.
#[derive(Clone, Debug, Default)]
pub struct PolyMesh {
    pub positions: Vec<Vec3>,
    pub faces: Vec<Face>,
}

impl PolyMesh {
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len() as u32;
        for (i, f) in self.faces.iter().enumerate() {
            let len = face_len(f);
            for k in 0..len {
                if f[k] >= n {
                    return Err(Error::InvalidMesh(format!("face {i} references vertex {} of {n}", f[k])));
                }
                if f[k] == f[(k + 1) % len] {
                    return Err(Error::InvalidMesh(format!("face {i} is degenerate")));
                }
            }
        }
        if self.faces.is_empty() {
            return Err(Error::InvalidMesh("mesh has no faces".into()));
        }
        Ok(())
    }

    /// Append another mesh (used for multi-part test scenes).
    pub fn merge(&mut self, other: &PolyMesh) {
        let off = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.faces.extend(other.faces.iter().map(|f| {
            let mut g = *f;
            for v in g.iter_mut().filter(|v| **v != NO_VERT) {
                *v += off;
            }
            g
        }));
    }

    pub fn translate(&mut self, t: Vec3) {
        for p in &mut self.positions {
            *p += t;
        }
    }
}

/// Compressed sparse rows: `items[offsets[i]..offsets[i+1]]` belong to row `i`.
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub offsets: Vec<u32>,
    pub items: Vec<u32>,
}

impl Csr {
    #[inline]
    pub fn row(&self, i: usize) -> &[u32] {
        &self.items[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Counting-sort construction from `(row, item)` pairs. Items keep input order.
    pub fn from_pairs(rows: usize, pairs: impl Iterator<Item = (u32, u32)> + Clone) -> Csr {
        let mut offsets = vec![0u32; rows + 1];
        for (r, _) in pairs.clone() {
            offsets[r as usize + 1] += 1;
        }
        for i in 0..rows {
            offsets[i + 1] += offsets[i];
        }
        let mut cursor = offsets.clone();
        let mut items = vec![0u32; offsets[rows] as usize];
        for (r, it) in pairs {
            let c = &mut cursor[r as usize];
            items[*c as usize] = it;
            *c += 1;
        }
        Csr { offsets, items }
    }
}

/// Vertex adjacency used by smoothing, normals, geodesics and blurs.
#[derive(Clone, Debug, Default)]
pub struct Topology {
    /// One-ring vertex neighbours (sorted, unique).
    pub vert_verts: Csr,
    /// Faces incident to each vertex.
    pub vert_faces: Csr,
}

impl Topology {
    pub fn build(vertex_count: usize, faces: &[Face]) -> Topology {
        let vert_faces = Csr::from_pairs(
            vertex_count,
            faces
                .iter()
                .enumerate()
                .flat_map(|(fi, f)| face_verts(f).iter().map(move |&v| (v, fi as u32))),
        );

        let rings: Vec<SmallVec<[u32; 8]>> = (0..vertex_count)
            .into_par_iter()
            .map(|v| {
                let mut ring: SmallVec<[u32; 8]> = SmallVec::new();
                for &fi in vert_faces.row(v) {
                    let f = &faces[fi as usize];
                    let n = face_len(f);
                    let k = f[..n].iter().position(|&x| x as usize == v).unwrap();
                    ring.push(f[(k + 1) % n]);
                    ring.push(f[(k + n - 1) % n]);
                }
                ring.sort_unstable();
                ring.dedup();
                ring
            })
            .collect();

        let mut offsets = Vec::with_capacity(vertex_count + 1);
        offsets.push(0u32);
        let mut total = 0u32;
        for r in &rings {
            total += r.len() as u32;
            offsets.push(total);
        }
        let mut items = Vec::with_capacity(total as usize);
        for r in &rings {
            items.extend_from_slice(r);
        }
        Topology { vert_verts: Csr { offsets, items }, vert_faces }
    }
}

/// Normal of a single vertex from its incident faces.
#[inline]
pub fn vertex_normal(positions: &[Vec3], faces: &[Face], topo: &Topology, v: usize) -> Vec3 {
    let mut n = Vec3::ZERO;
    for &f in topo.vert_faces.row(v) {
        n += face_normal(positions, &faces[f as usize]);
    }
    n.normalize_or(Vec3::Z)
}

pub fn compute_normals(positions: &[Vec3], faces: &[Face], topo: &Topology) -> Vec<Vec3> {
    (0..positions.len())
        .into_par_iter()
        .map(|v| vertex_normal(positions, faces, topo, v))
        .collect()
}

/// Apply an `old -> new` vertex permutation to a per-vertex array.
pub fn permute<T: Copy + Default + Send + Sync>(old_to_new: &[u32], data: &[T]) -> Vec<T> {
    let mut out = vec![T::default(); data.len()];
    for (old, &new) in old_to_new.iter().enumerate() {
        out[new as usize] = data[old];
    }
    out
}

/// Inverse of [`permute`]: bring internally ordered data back to the source order.
pub fn unpermute<T: Copy + Default + Send + Sync>(old_to_new: &[u32], data: &[T]) -> Vec<T> {
    old_to_new.par_iter().map(|&new| data[new as usize]).collect()
}

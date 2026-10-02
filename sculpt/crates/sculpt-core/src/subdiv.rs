//! Catmull-Clark subdivision as a reusable linear operator.
//!
//! The topology plan is built once; [`Subdivider::apply`] then maps *any*
//! linear per-vertex attribute (positions, layer deltas, mask channels) to the
//! next level. Because the operator is linear, subdividing base positions and
//! each layer's deltas separately gives the same surface as subdividing the
//! composite, which is what lets sculpt layers survive a level change.
//!
//! New vertex order: `[vertex points | edge points | face points]`.

use std::ops::{Add, Mul};

use glam::Vec3;
use rayon::prelude::*;

use crate::mesh::{Csr, Face, NO_VERT, face_len, face_verts};

pub trait Attr: Copy + Send + Sync + Default + Add<Output = Self> + Mul<f32, Output = Self> {}
impl Attr for f32 {}
impl Attr for Vec3 {}

pub struct Subdivider {
    vertex_count: usize,
    faces: Vec<Face>,
    edges: Vec<[u32; 2]>,
    /// Up to two faces per edge; second is `NO_VERT` on boundaries.
    edge_faces: Vec<[u32; 2]>,
    vert_edges: Csr,
    vert_faces: Csr,
    new_faces: Vec<Face>,
}

impl Subdivider {
    pub fn new(vertex_count: usize, faces: &[Face]) -> Subdivider {
        let mut sides: Vec<(u64, u32)> = Vec::with_capacity(faces.len() * 4);
        for (fi, f) in faces.iter().enumerate() {
            let n = face_len(f);
            for i in 0..n {
                let (a, b) = (f[i], f[(i + 1) % n]);
                let key = ((a.min(b) as u64) << 32) | a.max(b) as u64;
                sides.push((key, (fi as u32) << 2 | i as u32));
            }
        }
        sides.par_sort_unstable_by_key(|s| s.0);

        let mut edges = Vec::with_capacity(sides.len() / 2 + 1);
        let mut edge_faces = Vec::with_capacity(sides.len() / 2 + 1);
        let mut face_edges = vec![[NO_VERT; 4]; faces.len()];
        let mut i = 0;
        while i < sides.len() {
            let key = sides[i].0;
            let e = edges.len() as u32;
            edges.push([(key >> 32) as u32, key as u32]);
            let mut ef = [NO_VERT; 2];
            let mut k = 0;
            while i < sides.len() && sides[i].0 == key {
                let (fi, side) = (sides[i].1 >> 2, sides[i].1 & 3);
                if k < 2 {
                    ef[k] = fi;
                }
                k += 1;
                face_edges[fi as usize][side as usize] = e;
                i += 1;
            }
            edge_faces.push(ef);
        }

        let vert_edges = Csr::from_pairs(
            vertex_count,
            edges.iter().enumerate().flat_map(|(e, ab)| [(ab[0], e as u32), (ab[1], e as u32)]),
        );
        let vert_faces = Csr::from_pairs(
            vertex_count,
            faces
                .iter()
                .enumerate()
                .flat_map(|(fi, f)| face_verts(f).iter().map(move |&v| (v, fi as u32))),
        );

        let v0 = vertex_count as u32;
        let e0 = v0 + edges.len() as u32;
        let new_faces: Vec<Face> = faces
            .par_iter()
            .enumerate()
            .flat_map_iter(|(fi, f)| {
                let n = face_len(f);
                let fe = face_edges[fi];
                (0..n).map(move |i| {
                    [f[i], v0 + fe[i], e0 + fi as u32, v0 + fe[(i + n - 1) % n]]
                })
            })
            .collect();

        Subdivider {
            vertex_count,
            faces: faces.to_vec(),
            edges,
            edge_faces,
            vert_edges,
            vert_faces,
            new_faces,
        }
    }

    pub fn new_vertex_count(&self) -> usize {
        self.vertex_count + self.edges.len() + self.faces.len()
    }

    pub fn new_faces(&self) -> &[Face] {
        &self.new_faces
    }

    pub fn apply<T: Attr>(&self, vals: &[T]) -> Vec<T> {
        assert_eq!(vals.len(), self.vertex_count);
        let face_pts: Vec<T> = self
            .faces
            .par_iter()
            .map(|f| {
                let vs = face_verts(f);
                let mut s = T::default();
                for &v in vs {
                    s = s + vals[v as usize];
                }
                s * (1.0 / vs.len() as f32)
            })
            .collect();

        let edge_pts: Vec<T> = (0..self.edges.len())
            .into_par_iter()
            .map(|e| {
                let [a, b] = self.edges[e];
                let [f0, f1] = self.edge_faces[e];
                let mid = vals[a as usize] + vals[b as usize];
                if f1 == NO_VERT {
                    mid * 0.5
                } else {
                    (mid + face_pts[f0 as usize] + face_pts[f1 as usize]) * 0.25
                }
            })
            .collect();

        let vert_pts: Vec<T> = (0..self.vertex_count)
            .into_par_iter()
            .map(|v| {
                let p = vals[v];
                let ve = self.vert_edges.row(v);
                let boundary: smallvec::SmallVec<[u32; 4]> = ve
                    .iter()
                    .copied()
                    .filter(|&e| self.edge_faces[e as usize][1] == NO_VERT)
                    .collect();
                let other = |e: u32| {
                    let [a, b] = self.edges[e as usize];
                    if a as usize == v { b } else { a }
                };
                if !boundary.is_empty() {
                    if boundary.len() == 2 {
                        return p * 0.75 + (vals[other(boundary[0]) as usize] + vals[other(boundary[1]) as usize]) * 0.125;
                    }
                    return p; // corner / non-manifold: keep sharp
                }
                let vf = self.vert_faces.row(v);
                if ve.is_empty() || vf.is_empty() {
                    return p;
                }
                let n = ve.len() as f32;
                let mut f_avg = T::default();
                for &f in vf {
                    f_avg = f_avg + face_pts[f as usize];
                }
                f_avg = f_avg * (1.0 / vf.len() as f32);
                let mut r_avg = T::default();
                for &e in ve {
                    r_avg = r_avg + (p + vals[other(e) as usize]) * 0.5;
                }
                r_avg = r_avg * (1.0 / n);
                (f_avg + r_avg * 2.0 + p * (n - 3.0)) * (1.0 / n)
            })
            .collect();

        let mut out = vert_pts;
        out.reserve(edge_pts.len() + face_pts.len());
        out.extend(edge_pts);
        out.extend(face_pts);
        out
    }
}

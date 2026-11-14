//! Live evening of the output: tangential relaxation.

use glam::Vec3;
use rayon::prelude::*;

use super::TriMesh;
use crate::mesh::Csr;

/// Move each vertex toward the area-weighted centre of its ring, within its tangent plane, `iterations` times at
/// strength `evenness` (0 does nothing). Border vertices stay. Topology is untouched.
pub fn relax(mesh: &mut TriMesh, evenness: f32, iterations: usize) {
    let e = evenness.clamp(0.0, 1.0);
    if e <= 0.0 || iterations == 0 || mesh.triangles.is_empty() {
        return;
    }
    let nv = mesh.positions.len();
    let vf = Csr::from_pairs(nv, mesh.triangles.iter().enumerate().flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))));
    // Border: an edge of v whose other end is seen an odd number of times around v.
    let border: Vec<bool> = (0..nv)
        .into_par_iter()
        .map(|v| {
            let mut ends: Vec<u32> = Vec::with_capacity(16);
            for &f in vf.row(v) {
                for &u in &mesh.triangles[f as usize] {
                    if u as usize != v {
                        ends.push(u);
                    }
                }
            }
            ends.sort_unstable();
            ends.chunk_by(|a, b| a == b).any(|c| c.len() != 2)
        })
        .collect();
    let lambda = 0.5 * e;
    for _ in 0..iterations {
        let p = &mesh.positions;
        let next: Vec<Vec3> = (0..nv)
            .into_par_iter()
            .map(|v| {
                let pv = p[v];
                if border[v] || vf.row(v).is_empty() {
                    return pv;
                }
                let (mut c, mut w, mut n) = (Vec3::ZERO, 0.0f32, Vec3::ZERO);
                for &f in vf.row(v) {
                    let t = mesh.triangles[f as usize];
                    let (a, b, cc) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                    let cr = (b - a).cross(cc - a);
                    let area = cr.length();
                    n += cr;
                    c += (a + b + cc) / 3.0 * area;
                    w += area;
                }
                if w <= 0.0 {
                    return pv;
                }
                let n = n.normalize_or_zero();
                let d = c / w - pv;
                pv + (d - n * n.dot(d)) * lambda
            })
            .collect();
        mesh.positions = next;
    }
}

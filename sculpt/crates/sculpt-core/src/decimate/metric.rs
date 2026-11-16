//! How far a decimated surface strays from the original: sampled two-way distances (mean and Hausdorff).

use std::collections::HashMap;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::TriMesh;
use crate::geom::Aabb;

#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceError {
    pub mean: f32,
    /// Largest sampled distance either way (a Hausdorff estimate).
    pub max: f32,
    /// The original's bounding-box diagonal, to read the others as a share of the model's size.
    pub diagonal: f32,
}

impl SurfaceError {
    pub fn mean_rel(&self) -> f32 {
        self.mean / self.diagonal.max(1e-12)
    }
    pub fn max_rel(&self) -> f32 {
        self.max / self.diagonal.max(1e-12)
    }
}

/// Samples up to `samples` points on each mesh (vertices and face centres) and measures each to the other surface.
pub fn surface_error(original: &TriMesh, decimated: &TriMesh, samples: usize) -> SurfaceError {
    let to_dec = Grid::new(decimated);
    let to_orig = Grid::new(original);
    let a = to_dec.distances(&sample_points(original, samples));
    let b = to_orig.distances(&sample_points(decimated, samples));
    let n = (a.len() + b.len()).max(1) as f64;
    let mean = (a.iter().chain(&b).map(|&d| d as f64).sum::<f64>() / n) as f32;
    let max = a.iter().chain(&b).copied().fold(0.0f32, f32::max);
    SurfaceError { mean, max, diagonal: Aabb::from_points(&original.positions).diagonal() }
}

fn sample_points(m: &TriMesh, samples: usize) -> Vec<Vec3> {
    let n = m.triangles.len();
    let step = (2 * n / samples.max(1)).max(1);
    let mut out = Vec::with_capacity(samples + 2);
    for t in m.triangles.iter().step_by(step) {
        let p = t.map(|v| m.positions[v as usize]);
        out.push((p[0] + p[1] + p[2]) / 3.0);
        out.push(p[0]);
    }
    out
}

struct Grid<'a> {
    m: &'a TriMesh,
    min: Vec3,
    h: f32,
    cells: HashMap<IVec3, Vec<u32>>,
}

impl<'a> Grid<'a> {
    fn new(m: &'a TriMesh) -> Grid<'a> {
        let b = Aabb::from_points(&m.positions);
        let area: f32 = m
            .triangles
            .iter()
            .map(|t| (m.positions[t[1] as usize] - m.positions[t[0] as usize]).cross(m.positions[t[2] as usize] - m.positions[t[0] as usize]).length() * 0.5)
            .sum();
        // About two triangles per cell.
        let h = (area / (m.triangles.len().max(1) as f32) * 4.0).sqrt().max(b.diagonal() * 1e-4).max(1e-9);
        let mut cells: HashMap<IVec3, Vec<u32>> = HashMap::new();
        for (f, t) in m.triangles.iter().enumerate() {
            let tb = Aabb::from_points(&t.map(|v| m.positions[v as usize]));
            let (lo, hi) = (((tb.min - b.min) / h).floor().as_ivec3(), ((tb.max - b.min) / h).floor().as_ivec3());
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        cells.entry(IVec3::new(x, y, z)).or_default().push(f as u32);
                    }
                }
            }
        }
        Grid { m, min: b.min, h, cells }
    }

    fn closest(&self, p: Vec3) -> f32 {
        let c = ((p - self.min) / self.h).floor().as_ivec3();
        let mut best = f32::INFINITY;
        for r in 0..64i32 {
            // Shell r: cells at Chebyshev distance r. Once a hit is closer than the shell's inner reach, stop.
            if best <= (r as f32 - 1.0).max(0.0) * self.h {
                break;
            }
            for z in -r..=r {
                for y in -r..=r {
                    for x in -r..=r {
                        if x.abs().max(y.abs()).max(z.abs()) != r {
                            continue;
                        }
                        if let Some(fs) = self.cells.get(&(c + IVec3::new(x, y, z))) {
                            for &f in fs {
                                let t = self.m.triangles[f as usize];
                                let q = closest_on_triangle(p, self.m.positions[t[0] as usize], self.m.positions[t[1] as usize], self.m.positions[t[2] as usize]);
                                best = best.min(p.distance(q));
                            }
                        }
                    }
                }
            }
        }
        best
    }

    fn distances(&self, pts: &[Vec3]) -> Vec<f32> {
        pts.par_iter().map(|&p| self.closest(p)).filter(|d| d.is_finite()).collect()
    }
}

/// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

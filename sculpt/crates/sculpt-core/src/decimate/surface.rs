//! The original surface, for projecting points back onto it: a uniform grid of the triangles (each listed in every
//! cell its box touches), searched nearest cell first with cells pruned by their distance.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use glam::{IVec3, Vec2, Vec3};
use rayon::prelude::*;

use super::TriMesh;
use super::metric::closest_on_triangle;
use crate::geom::Aabb;

/// Multiplicative hash for packed cell keys.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    fn write_u64(&mut self, k: u64) {
        self.0 = (k ^ (k >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type Cells = HashMap<u64, (u32, u32), BuildHasherDefault<KeyHasher>>;

const BITS: u32 = 21;
const MASK: i32 = (1 << BITS) - 1;

fn key(c: IVec3) -> u64 {
    ((c.x & MASK) as u64) | (((c.y & MASK) as u64) << BITS) | (((c.z & MASK) as u64) << (2 * BITS))
}

/// A triangle mesh ready for closest-point queries.
pub struct Surface {
    mesh: Arc<TriMesh>,
    min: Vec3,
    h: f32,
    cells: Cells,
    ids: Vec<u32>,
}

impl Surface {
    pub fn new(mesh: Arc<TriMesh>) -> Surface {
        let b = Aabb::from_points(&mesh.positions);
        let n = mesh.triangles.len().max(1);
        let area: f64 = mesh.triangles.par_iter().map(|t| tri_area(&mesh, t) as f64).sum();
        // About two mean edges a cell: a few triangles in each, each in a few cells.
        let h = ((area / n as f64).sqrt() as f32 * 2.0).max(b.diagonal() / (1 << (BITS - 1)) as f32).max(1e-9);
        let min = b.min;
        let mut pairs: Vec<(u64, u32)> = mesh
            .triangles
            .par_iter()
            .enumerate()
            .flat_map_iter(|(f, t)| {
                let tb = Aabb::from_points(&t.map(|v| mesh.positions[v as usize]));
                let (lo, hi) = (((tb.min - min) / h).floor().as_ivec3(), ((tb.max - min) / h).floor().as_ivec3());
                (lo.z..=hi.z).flat_map(move |z| (lo.y..=hi.y).flat_map(move |y| (lo.x..=hi.x).map(move |x| (key(IVec3::new(x, y, z)), f as u32))))
            })
            .collect();
        pairs.par_sort_unstable();
        let ids: Vec<u32> = pairs.par_iter().map(|p| p.1).collect();
        let mut cells = Cells::default();
        let mut i = 0;
        while i < pairs.len() {
            let k = pairs[i].0;
            let mut j = i + 1;
            while j < pairs.len() && pairs[j].0 == k {
                j += 1;
            }
            cells.insert(k, (i as u32, j as u32));
            i = j;
        }
        Surface { mesh, min, h, cells, ids }
    }

    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    /// The closest point on the surface to `p` within `max_dist`, if any.
    pub fn closest(&self, p: Vec3, max_dist: f32) -> Option<Vec3> {
        self.closest_hit(p, max_dist).map(|(q, _)| q)
    }

    /// The closest point on the surface to `p` within `max_dist` and the triangle it is on, if any.
    pub fn closest_hit(&self, p: Vec3, max_dist: f32) -> Option<(Vec3, u32)> {
        let rel = (p - self.min) / self.h;
        let c = rel.floor().as_ivec3();
        let frac = rel - c.as_vec3();
        let mut best = max_dist * max_dist;
        let mut hit = None;
        let reach = (max_dist / self.h).ceil().min(64.0) as i32;
        for r in 0..=reach {
            // Every cell of shell r is at least this far (r-1 whole cells plus the way out of the own cell).
            let shell = if r == 0 { 0.0 } else { (r as f32 - 1.0 + frac.min_element().min((1.0 - frac).min_element())) * self.h };
            if shell * shell >= best {
                break;
            }
            for z in -r..=r {
                for y in -r..=r {
                    for x in -r..=r {
                        if x.abs().max(y.abs()).max(z.abs()) != r {
                            continue;
                        }
                        // Distance from p to this cell's box.
                        let o = IVec3::new(x, y, z);
                        let lo = o.as_vec3() - frac;
                        let d = (lo.max(Vec3::ZERO) + (-(lo + Vec3::ONE)).max(Vec3::ZERO)) * self.h;
                        if d.length_squared() >= best {
                            continue;
                        }
                        let Some(&(s, e)) = self.cells.get(&key(c + o)) else { continue };
                        for &f in &self.ids[s as usize..e as usize] {
                            let t = self.mesh.triangles[f as usize];
                            let q = closest_on_triangle(
                                p,
                                self.mesh.positions[t[0] as usize],
                                self.mesh.positions[t[1] as usize],
                                self.mesh.positions[t[2] as usize],
                            );
                            let d2 = p.distance_squared(q);
                            if d2 < best {
                                best = d2;
                                hit = Some((q, f));
                            }
                        }
                    }
                }
            }
        }
        hit
    }

    /// The first point of the surface on the segment `o + t d`, `t` in `0..=max_t` (`d` of unit length), walking the
    /// grid cell by cell along it.
    pub fn ray(&self, o: Vec3, d: Vec3, max_t: f32) -> Option<Hit> {
        if d.length_squared() == 0.0 {
            return None;
        }
        let rel = (o - self.min) / self.h;
        let mut c = rel.floor().as_ivec3();
        let frac = rel - c.as_vec3();
        let mut step = IVec3::ZERO;
        let mut next = Vec3::splat(f32::INFINITY);
        let mut delta = Vec3::splat(f32::INFINITY);
        for k in 0..3 {
            if d[k] > 0.0 {
                step[k] = 1;
                delta[k] = self.h / d[k];
                next[k] = (1.0 - frac[k]) * delta[k];
            } else if d[k] < 0.0 {
                step[k] = -1;
                delta[k] = -self.h / d[k];
                next[k] = frac[k] * delta[k];
            }
        }
        let mut best: Option<Hit> = None;
        loop {
            if let Some(&(s, e)) = self.cells.get(&key(c)) {
                for &f in &self.ids[s as usize..e as usize] {
                    let t = self.mesh.triangles[f as usize];
                    let p = t.map(|v| self.mesh.positions[v as usize]);
                    if let Some((t, u, v)) = intersect(o, d, p[0], p[1], p[2])
                        && t >= 0.0
                        && t <= max_t
                        && best.is_none_or(|b| t < b.t)
                    {
                        best = Some(Hit { t, face: f, bary: Vec2::new(u, v) });
                    }
                }
            }
            let exit = next.min_element();
            if best.is_some_and(|b| b.t <= exit) || exit > max_t {
                return best;
            }
            let k = if next.x <= next.y && next.x <= next.z { 0 } else if next.y <= next.z { 1 } else { 2 };
            c[k] += step[k];
            next[k] += delta[k];
        }
    }
}

/// Where a ray met the surface: how far along it, the triangle, and the weights of the triangle's corners 1 and 2.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub face: u32,
    pub bary: Vec2,
}

/// Ray and triangle, either side (Moller-Trumbore): `(t, u, v)` with the hit at `a + u (b - a) + v (c - a)`.
fn intersect(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-6;
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det == 0.0 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    Some((e2.dot(q) * inv, u, v))
}

fn tri_area(m: &TriMesh, t: &[u32; 3]) -> f32 {
    let p = t.map(|v| m.positions[v as usize]);
    (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::quad_sphere_res;

    #[test]
    fn closest_matches_brute_force() {
        let m = Arc::new(TriMesh::from_poly(&quad_sphere_res(20, 1.0)));
        let s = Surface::new(m.clone());
        for i in 0..200 {
            let a = i as f32 * 0.37;
            let p = Vec3::new(a.sin() * 1.1, (a * 1.7).cos() * 0.9, (a * 0.3).sin() * 1.05);
            let got = s.closest(p, 10.0).unwrap();
            let want = m
                .triangles
                .iter()
                .map(|t| closest_on_triangle(p, m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]))
                .min_by(|x, y| p.distance_squared(*x).total_cmp(&p.distance_squared(*y)))
                .unwrap();
            assert!((p.distance(got) - p.distance(want)).abs() < 1e-5, "{p}: {got} vs {want}");
        }
    }

    #[test]
    fn ray_hits_sphere_from_inside_and_out() {
        let m = Arc::new(TriMesh::from_poly(&quad_sphere_res(20, 1.0)));
        let s = Surface::new(m);
        for i in 0..100 {
            let a = i as f32 * 0.61;
            let d = Vec3::new(a.sin(), (a * 1.3).cos(), (a * 0.7).sin()).normalize();
            let inside = s.ray(d * 0.5, d, 1.0).expect("hit from inside");
            assert!((inside.t - 0.5).abs() < 0.01, "{}", inside.t);
            let outside = s.ray(d * 1.5, -d, 1.0).expect("hit from outside");
            assert!((outside.t - 0.5).abs() < 0.01, "{}", outside.t);
            assert!(s.ray(d * 1.5, d, 1.0).is_none());
        }
    }
}

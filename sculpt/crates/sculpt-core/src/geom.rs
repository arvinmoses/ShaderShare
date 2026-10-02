//! Small geometric primitives shared by the spatial index, brushes and bakers.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline]
    pub fn grow(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.min(o.min), max: self.max.max(o.max) }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(&self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { self.max - self.min }
    }

    pub fn diagonal(&self) -> f32 {
        self.extent().length()
    }

    pub fn from_points(points: &[Vec3]) -> Aabb {
        let mut b = Aabb::EMPTY;
        for &p in points {
            b.grow(p);
        }
        b
    }

    #[inline]
    pub fn intersects_sphere(&self, c: Vec3, r: f32) -> bool {
        if self.is_empty() {
            return false;
        }
        let q = c.max(self.min).min(self.max);
        q.distance_squared(c) <= r * r
    }

    /// Slab test. Returns the entry distance if the ray hits within `[0, tmax]`.
    #[inline]
    pub fn ray_entry(&self, origin: Vec3, inv_dir: Vec3, tmax: f32) -> Option<f32> {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let tmin_v = t0.min(t1);
        let tmax_v = t0.max(t1);
        let enter = tmin_v.max_element().max(0.0);
        let exit = tmax_v.min_element().min(tmax);
        (enter <= exit).then_some(enter)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray { origin, dir: dir.normalize() }
    }
}

/// Two-sided Möller–Trumbore. Returns the hit distance along `dir`.
/// Barycentric bounds are padded slightly so rays through shared edges and
/// vertices can't slip between adjacent triangles.
#[inline]
pub fn ray_triangle(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    const EPS: f32 = 1e-5;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

/// Smoothstep on `[e0, e1]`.
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

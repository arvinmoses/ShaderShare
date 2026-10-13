//! Analytic signed distance functions for the generator primitives.
//!
//! All primitives are defined in a canonical local frame of roughly unit size
//! (see [`Primitive::local_half_extent`]); [`crate::generator::Instance`] places
//! them in the world. Negative distance = inside.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Torus tube radius relative to the unit major radius.
pub const TORUS_MINOR: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Primitive {
    /// Box with half extents 1.
    Box,
    /// Sphere of radius 1.
    Sphere,
    /// Cylinder along Y, radius 1, half height 1.
    Cylinder,
    /// Cone along Y: base radius 1 at y = -1, apex at y = +1.
    Cone,
    /// Right-triangle prism (ramp): legs along X and Y, half length 1 along Z.
    Wedge,
    /// Diamond (45 deg rotated square) prism along Z, half length 1. Centered on
    /// a 90 degree edge it removes a symmetric chamfer.
    ChamferCutter,
    /// Torus around Y, major radius 1, tube radius [`TORUS_MINOR`].
    Torus,
}

impl Primitive {
    pub const ALL: [Primitive; 7] = [
        Primitive::Box,
        Primitive::Sphere,
        Primitive::Cylinder,
        Primitive::Cone,
        Primitive::Wedge,
        Primitive::ChamferCutter,
        Primitive::Torus,
    ];

    /// Half extents of the tight local AABB.
    pub fn local_half_extent(self) -> Vec3 {
        match self {
            Primitive::Torus => Vec3::new(1.0 + TORUS_MINOR, TORUS_MINOR, 1.0 + TORUS_MINOR),
            _ => Vec3::ONE,
        }
    }

    /// Signed distance in the local frame (exact for all but non-uniformly
    /// scaled instances, where the instance applies a conservative factor).
    #[inline]
    pub fn sdf(self, p: Vec3) -> f32 {
        match self {
            Primitive::Sphere => p.length() - 1.0,
            Primitive::Box => {
                let q = p.abs() - Vec3::ONE;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            Primitive::Cylinder => {
                let d = Vec2::new(Vec2::new(p.x, p.z).length() - 1.0, p.y.abs() - 1.0);
                d.max_element().min(0.0) + d.max(Vec2::ZERO).length()
            }
            Primitive::Cone => sd_capped_cone(p, 1.0, 1.0, 0.0),
            Primitive::Wedge => {
                let d2 = sd_triangle(Vec2::new(p.x, p.y), Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0));
                extrude(d2, p.z)
            }
            Primitive::ChamferCutter => {
                let s = std::f32::consts::FRAC_1_SQRT_2;
                let u = (p.x + p.y) * s;
                let v = (p.x - p.y) * s;
                let q = Vec2::new(u.abs() - s, v.abs() - s);
                let d2 = q.max(Vec2::ZERO).length() + q.max_element().min(0.0);
                extrude(d2, p.z)
            }
            Primitive::Torus => {
                let q = Vec2::new(Vec2::new(p.x, p.z).length() - 1.0, p.y);
                q.length() - TORUS_MINOR
            }
        }
    }

    /// Distance and unit gradient (analytic for sphere and box, tetrahedral
    /// central differences of the exact SDF for the rest).
    #[inline]
    pub fn sdf_grad(self, p: Vec3) -> (f32, Vec3) {
        match self {
            Primitive::Sphere => {
                let l = p.length();
                (l - 1.0, if l > 1e-12 { p / l } else { Vec3::Y })
            }
            Primitive::Box => {
                let a = p.abs();
                let q = a - Vec3::ONE;
                let s = p.signum();
                let d = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);
                let g = if q.max_element() > 0.0 {
                    (s * q.max(Vec3::ZERO)).normalize_or(Vec3::Y)
                } else {
                    // inside: gradient points along the axis of the closest face
                    let m = q.max_position();
                    let mut g = Vec3::ZERO;
                    g[m] = s[m];
                    g
                };
                (d, g)
            }
            _ => {
                const E: f32 = 1e-3;
                let k0 = Vec3::new(1.0, -1.0, -1.0);
                let k1 = Vec3::new(-1.0, -1.0, 1.0);
                let k2 = Vec3::new(-1.0, 1.0, -1.0);
                let k3 = Vec3::ONE;
                let g = k0 * self.sdf(p + k0 * E) + k1 * self.sdf(p + k1 * E) + k2 * self.sdf(p + k2 * E) + k3 * self.sdf(p + k3 * E);
                (self.sdf(p), g.normalize_or(Vec3::Y))
            }
        }
    }
}

#[inline]
fn extrude(d2: f32, z: f32) -> f32 {
    let w = Vec2::new(d2, z.abs() - 1.0);
    w.max_element().min(0.0) + w.max(Vec2::ZERO).length()
}

fn sd_capped_cone(p: Vec3, h: f32, r1: f32, r2: f32) -> f32 {
    let q = Vec2::new(Vec2::new(p.x, p.z).length(), p.y);
    let k1 = Vec2::new(r2, h);
    let k2 = Vec2::new(r2 - r1, 2.0 * h);
    let ca = Vec2::new(q.x - q.x.min(if q.y < 0.0 { r1 } else { r2 }), q.y.abs() - h);
    let cb = q - k1 + k2 * ((k1 - q).dot(k2) / k2.length_squared()).clamp(0.0, 1.0);
    let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    s * ca.length_squared().min(cb.length_squared()).sqrt()
}

fn sd_triangle(p: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * (v0.dot(e0) / e0.dot(e0)).clamp(0.0, 1.0);
    let pq1 = v1 - e1 * (v1.dot(e1) / e1.dot(e1)).clamp(0.0, 1.0);
    let pq2 = v2 - e2 * (v2.dot(e2) / e2.dot(e2)).clamp(0.0, 1.0);
    let s = (e0.x * e2.y - e0.y * e2.x).signum();
    let a = Vec2::new(pq0.dot(pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let b = Vec2::new(pq1.dot(pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let c = Vec2::new(pq2.dot(pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = a.min(b).min(c);
    -d.x.sqrt() * d.y.signum()
}

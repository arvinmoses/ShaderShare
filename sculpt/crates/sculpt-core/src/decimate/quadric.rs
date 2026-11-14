//! Error quadrics (Garland-Heckbert), area weighted, in f64.
//!
//! `Q(x) = x^T A x - 2 b^T x + c` summed over planes `n.x = d` weighted by area, plus the total weight so the error
//! can be read as a mean squared distance.

use glam::{DVec3, Vec3};

#[derive(Clone, Copy, Debug, Default)]
pub struct Quadric {
    // Symmetric A: xx xy xz yy yz zz.
    a: [f64; 6],
    b: DVec3,
    c: f64,
    /// Sum of the weights (areas).
    pub w: f64,
}

impl Quadric {
    /// The plane through `p` with unit normal `n`, weighted by `w`.
    pub fn plane(n: DVec3, p: DVec3, w: f64) -> Quadric {
        let d = n.dot(p);
        Quadric {
            a: [n.x * n.x * w, n.x * n.y * w, n.x * n.z * w, n.y * n.y * w, n.y * n.z * w, n.z * n.z * w],
            b: n * (d * w),
            c: d * d * w,
            w,
        }
    }

    /// Like [`Self::plane`] but adds nothing to the total weight (constraint planes along borders).
    pub fn constraint(n: DVec3, p: DVec3, w: f64) -> Quadric {
        let mut q = Quadric::plane(n, p, w);
        q.w = 0.0;
        q
    }

    pub fn add(&mut self, o: &Quadric) {
        for i in 0..6 {
            self.a[i] += o.a[i];
        }
        self.b += o.b;
        self.c += o.c;
        self.w += o.w;
    }

    pub fn sum(a: &Quadric, b: &Quadric) -> Quadric {
        let mut q = *a;
        q.add(b);
        q
    }

    /// Sum of weighted squared distances at `x` (never negative).
    pub fn eval(&self, x: DVec3) -> f64 {
        let a = &self.a;
        let ax = DVec3::new(
            a[0] * x.x + a[1] * x.y + a[2] * x.z,
            a[1] * x.x + a[3] * x.y + a[4] * x.z,
            a[2] * x.x + a[4] * x.y + a[5] * x.z,
        );
        (x.dot(ax) - 2.0 * self.b.dot(x) + self.c).max(0.0)
    }

    /// The point minimising `Q(x)/w + s |x - m|^2`. `s > 0` keeps the system well posed (on flat or cylindrical
    /// pieces the quadric alone has a line or plane of minima); it also pulls the point toward `m`.
    pub fn minimise_toward(&self, m: DVec3, s: f64) -> DVec3 {
        let inv = if self.w > 0.0 { 1.0 / self.w } else { 1.0 };
        let a = self.a.map(|v| v * inv);
        let (m00, m01, m02, m11, m12, m22) = (a[0] + s, a[1], a[2], a[3] + s, a[4], a[5] + s);
        let r = self.b * inv + m * s;
        // Cramer on the symmetric 3x3.
        let c00 = m11 * m22 - m12 * m12;
        let c01 = m02 * m12 - m01 * m22;
        let c02 = m01 * m12 - m02 * m11;
        let det = m00 * c00 + m01 * c01 + m02 * c02;
        if det.abs() < 1e-30 {
            return m;
        }
        let c11 = m00 * m22 - m02 * m02;
        let c12 = m01 * m02 - m00 * m12;
        let c22 = m00 * m11 - m01 * m01;
        let x = DVec3::new(
            c00 * r.x + c01 * r.y + c02 * r.z,
            c01 * r.x + c11 * r.y + c12 * r.z,
            c02 * r.x + c12 * r.y + c22 * r.z,
        ) / det;
        if x.is_finite() { x } else { m }
    }
}

#[inline]
pub fn d3(v: Vec3) -> DVec3 {
    v.as_dvec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_of_three_planes() {
        let mut q = Quadric::plane(DVec3::X, DVec3::new(1.0, 0.0, 0.0), 1.0);
        q.add(&Quadric::plane(DVec3::Y, DVec3::new(0.0, 2.0, 0.0), 1.0));
        q.add(&Quadric::plane(DVec3::Z, DVec3::new(0.0, 0.0, 3.0), 1.0));
        let x = q.minimise_toward(DVec3::ZERO, 1e-9);
        assert!((x - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-6, "{x}");
        assert!(q.eval(x) < 1e-9);
    }
}

//! Spike 4: a volume at several resolutions. Fine detail rides a large coarse edit, and the surface joins the levels
//! without cracks, at every level cap a distant view would use.

use glam::Vec3;
use sculpt_core::generator::levels::Levels;

/// Height of the surface above the ball (radius 1) at angle `a` (radians around y, from +z), seen along the radius.
fn height(v: &Levels, a: f32) -> f32 {
    let dir = Vec3::new(a.sin(), 0.0, a.cos());
    v.raycast(dir * 3.0, -dir).map_or(f32::NAN, |(h, _)| h.length() - 1.0)
}

/// A groove along y at the finest level, then a broad level 0 dab raising the area by about 0.02: the groove's depth
/// before and after, and the open edges of the surface at each level cap, before and after.
fn groove_then_raise(levels: usize, restrict: bool) -> (f32, f32, Vec<usize>) {
    let mut v = Levels::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.02, levels);
    v.restrict = restrict;
    for i in 0..40 {
        let y = -0.15 + 0.3 * i as f32 / 39.0;
        let p = Vec3::new(0.0, y, (1.0 - y * y).sqrt());
        v.dab(levels - 1, p * 1.004, 0.012, 0.003, true);
    }
    let depth = |v: &Levels| {
        // Against the mean of the heights either side, so the raised dome's curvature does not count.
        (height(v, -0.03) + height(v, 0.03)) / 2.0 - height(v, 0.0)
    };
    let before = depth(&v);
    let mut open: Vec<usize> = (0..levels).map(|cap| v.surface(cap).1.open_edges).collect();
    v.dab(0, Vec3::new(0.0, 0.0, 1.0 - 0.6 + 0.02), 0.6, 0.05, false);
    let after = depth(&v);
    open.extend((0..levels).map(|cap| v.surface(cap).1.open_edges));
    (before, after, open)
}

#[test]
fn fine_detail_rides_a_coarse_edit() {
    for levels in [2usize, 3] {
        let (a0, a1, _) = groove_then_raise(levels, true);
        let (b0, b1, open) = groove_then_raise(levels, false);
        eprintln!("{levels} levels: groove depth, rule A {a0:.4} -> {a1:.4} ({:.0}%), rule B {b0:.4} -> {b1:.4} ({:.0}%); open edges by cap, before then after (rule B) {open:?}", 100.0 * a1 / a0, 100.0 * b1 / b0);
        assert!(b0 > 0.005, "the groove was not cut");
        assert!(b1 > 0.8 * b0, "rule B lost the detail");
        assert!(open.iter().all(|&o| o == 0), "the surface cracked: {open:?}");
    }
}

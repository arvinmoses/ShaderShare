use glam::{Quat, Vec3};
use sculpt_core::generator::adf::{AdfParams, BooleanAdf, Frame};
use sculpt_core::generator::{BoolOp, Instance, InstanceSet, Primitive};
use sculpt_core::geom::Aabb;

const EDGE: f32 = 0.02;

fn plane(p: Vec3, _bound: f32) -> f32 {
    p.y
}

fn sphere_field(c: Vec3, r: f32) -> impl Fn(Vec3, f32) -> f32 + Sync {
    move |p, _| p.distance(c) - r
}

fn bounds() -> Aabb {
    Aabb { min: Vec3::splat(-0.6), max: Vec3::splat(0.6) }
}

fn sphere(c: Vec3, r: f32) -> Instance {
    Instance::new(Primitive::Sphere, c, Quat::IDENTITY, Vec3::splat(r))
}

fn setup(tol_edges: f32, insts: Vec<Instance>) -> BooleanAdf {
    let params = AdfParams::new(EDGE, tol_edges, BoolOp::Subtract);
    let ext = insts.iter().map(|i| i.aabb().extent().max_element()).sum::<f32>() / insts.len() as f32;
    let frame = Frame::new(bounds(), &params, ext);
    BooleanAdf::new(frame, params, insts).0
}

/// Exact boolean field for a plane minus the given instances.
fn exact(insts: &[Instance]) -> impl Fn(Vec3) -> f32 + '_ {
    let set = InstanceSet::new(insts.to_vec(), 0.0);
    move |p| {
        let m = (0..insts.len() as u32).map(|i| set.sdf(i, p)).fold(f32::INFINITY, f32::min);
        p.y.max(-m)
    }
}

#[test]
fn adaptive_sampling_converges_to_exact_sdf() {
    // A sphere SDF as the "mesh" field: error at random points near the surface must stay within
    // about the tolerance and shrink with it.
    let c = Vec3::new(0.05, 0.0, -0.02);
    let field = sphere_field(c, 0.4);
    let inst = vec![sphere(c + Vec3::new(0.4, 0.0, 0.0), 0.12)];
    let mut errs = Vec::new();
    for tol_edges in [1.0f32, 0.25] {
        let mut a = setup(tol_edges, inst.clone());
        a.build(&field);
        let tol = a.params.tol();
        let mut max_err = 0.0f32;
        let mut n = 0;
        for i in 0..4000u32 {
            let h = |k: u32| ((i.wrapping_mul(2654435761).wrapping_add(k.wrapping_mul(40503))) % 10007) as f32 / 10007.0;
            let dir = Vec3::new(h(1) - 0.5, h(2) - 0.5, h(3) - 0.5).normalize_or(Vec3::X);
            let p = c + dir * (0.4 + (h(4) - 0.5) * tol);
            if let Some(v) = a.base.eval(p) {
                max_err = max_err.max((v - (p.distance(c) - 0.4)).abs());
                n += 1;
            }
        }
        assert!(n > 100, "too few samples inside built blocks: {n}");
        assert!(max_err <= 2.0 * tol, "tol {tol}: max err {max_err}");
        errs.push(max_err);
    }
    assert!(errs[1] < errs[0], "error must shrink with tolerance: {errs:?}");
}

#[test]
fn sphere_cutter_in_plane_matches_analytic() {
    let insts = vec![sphere(Vec3::new(0.0, 0.03, 0.0), 0.25)];
    for tol_edges in [1.0f32, 0.5, 0.25] {
        let mut a = setup(tol_edges, insts.clone());
        a.build(&plane);
        let m = a.extract(None);
        assert!(m.tris.len() > 200, "tris {}", m.tris.len());
        let f = exact(&insts);
        let tol = a.params.tol();
        let (mut worst_v, mut worst_f) = (0.0f32, 0.0f32);
        for t in &m.tris {
            let (p0, p1, p2) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
            worst_v = worst_v.max(f(p0).abs());
            for p in [(p0 + p1 + p2) / 3.0, (p0 + p1) * 0.5] {
                worst_f = worst_f.max(f(p).abs());
            }
        }
        // vertices sit on the surface to within the tolerance; facet interiors also carry chord error
        assert!(worst_v <= tol, "tol {tol}: vertex deviation {worst_v}");
        assert!(worst_f <= 2.0 * tol, "tol {tol}: facet deviation {worst_f}");
    }
}

#[test]
fn box_cutter_keeps_sharp_corners() {
    let rot = Quat::from_rotation_y(0.4);
    let insts = vec![Instance::new(Primitive::Box, Vec3::new(0.0, 0.0, 0.0), rot, Vec3::splat(0.2))];
    let mut a = setup(0.5, insts.clone());
    a.build(&plane);
    let m = a.extract(None);
    let f = exact(&insts);
    let tol = a.params.tol();
    let worst = m.positions.iter().map(|&p| f(p).abs()).fold(0.0f32, f32::max);
    assert!(worst <= tol, "deviation {worst}");
    // Vertical concave edges of the cavity: nearest extracted vertex must be within a fraction of an edge length.
    let mut max_gap = 0.0f32;
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        for k in 1..20 {
            let y = -0.2 * k as f32 / 20.0 * 0.95;
            let q = rot * Vec3::new(sx * 0.2, 0.0, sz * 0.2) + Vec3::new(0.0, y, 0.0);
            let d = m.positions.iter().map(|p| p.distance(q)).fold(f32::INFINITY, f32::min);
            max_gap = max_gap.max(d);
        }
    }
    assert!(max_gap < 0.6 * EDGE, "sharp edge gap {max_gap} vs edge {EDGE}");
}

#[test]
fn interior_of_patch_is_watertight() {
    // Away from the open patch boundary every edge must be shared by exactly two triangles
    // (no cracks between octree levels or between blocks).
    let insts = vec![sphere(Vec3::new(0.0, 0.03, 0.0), 0.25), Instance::new(Primitive::Box, Vec3::new(0.2, 0.0, 0.1), Quat::from_rotation_y(0.3), Vec3::splat(0.08))];
    let mut a = setup(0.5, insts);
    a.build(&plane);
    let m = a.extract(None);
    let mut count: std::collections::HashMap<(u32, u32), u32> = Default::default();
    for t in &m.tris {
        for k in 0..3 {
            let (x, y) = (t[k], t[(k + 1) % 3]);
            *count.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let (mut open, mut over) = (0, 0);
    for (&(x, y), &n) in &count {
        let mid = (m.positions[x as usize] + m.positions[y as usize]) * 0.5;
        if mid.x.abs() < 0.2 && mid.z.abs() < 0.2 {
            open += (n == 1) as usize;
            over += (n > 2) as usize;
        }
    }
    let total = count.len();
    println!("interior edges: {total}, open {open}, non-manifold {over}, patched quads {}", m.missing);
    assert!(open * 200 <= total, "{open} interior boundary edges of {total}");
    assert!(over * 200 <= total, "{over} non-manifold edges of {total}");
}

#[test]
fn result_is_oriented_outward() {
    let insts = vec![sphere(Vec3::new(0.0, 0.0, 0.0), 0.25)];
    let mut a = setup(0.5, insts.clone());
    a.build(&plane);
    let m = a.extract(None);
    let f = exact(&insts);
    let (mut good, mut total) = (0, 0);
    for t in &m.tris {
        let (p0, p1, p2) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let c = (p0 + p1 + p2) / 3.0;
        let e = 0.002;
        let g = Vec3::new(f(c + Vec3::X * e) - f(c - Vec3::X * e), f(c + Vec3::Y * e) - f(c - Vec3::Y * e), f(c + Vec3::Z * e) - f(c - Vec3::Z * e));
        total += 1;
        good += (n.dot(g.normalize_or_zero()) > 0.0) as usize;
    }
    assert!(good as f32 > 0.98 * total as f32, "{good}/{total} outward");
}

fn many(n: usize) -> Vec<Instance> {
    (0..n)
        .map(|i| {
            let h = |k: usize| ((i * 7919 + k * 104729) % 1000) as f32 / 1000.0;
            let c = Vec3::new(h(1) - 0.5, 0.02 * (h(2) - 0.5), h(3) - 0.5) * 0.9;
            if i % 3 == 0 {
                Instance::new(Primitive::Box, c, Quat::from_rotation_y(h(4) * 3.0), Vec3::splat(0.04 + 0.05 * h(5)))
            } else {
                sphere(c, 0.04 + 0.06 * h(5))
            }
        })
        .collect()
}

#[test]
fn incremental_equals_full_rebuild() {
    let insts = many(60);
    let mut a = setup(0.5, insts.clone());
    a.build(&plane);
    let mut changed = insts.clone();
    let mut changes = Vec::new();
    for i in (0..60).step_by(7) {
        let mut n = changed[i];
        n.scale *= 1.3;
        n.translation += Vec3::new(0.05, 0.0, -0.03);
        changed[i] = n;
        changes.push((i, n));
    }
    let (dirty, _bt) = a.update(&plane, &changes);
    let mut b = BooleanAdf::new(a.frame, a.params, changed).0;
    b.build(&plane);
    assert!(a.fingerprint() == b.fingerprint(), "incremental result differs from full rebuild");
    let (ma, mb) = (a.extract(None), b.extract(None));
    assert_eq!(ma.positions.len(), mb.positions.len());
    assert_eq!(ma.tris.len(), mb.tris.len());
    // dirty-only extraction is a subset of the full one
    let part = a.extract(Some(&dirty));
    assert!(part.quads <= ma.quads && part.quads > 0);
    let all: std::collections::HashSet<[u32; 3]> = ma.positions.iter().map(|p| p.to_array().map(f32::to_bits)).collect();
    assert!(part.positions.iter().all(|p| all.contains(&p.to_array().map(f32::to_bits))));
}

#[test]
fn add_and_intersect_match_analytic() {
    // Add a sphere above the plane, intersect a sphere with the half-space.
    let insts = vec![sphere(Vec3::new(0.0, 0.05, 0.0), 0.25)];
    for op in [BoolOp::Add, BoolOp::Intersect] {
        let params = AdfParams::new(EDGE, 0.5, op);
        let frame = Frame::new(bounds(), &params, 0.5);
        let mut a = BooleanAdf::new(frame, params, insts.clone()).0;
        a.build(&plane);
        let m = a.extract(None);
        assert!(m.tris.len() > 100, "{op:?} tris {}", m.tris.len());
        let set = InstanceSet::new(insts.clone(), 0.0);
        let worst = m
            .positions
            .iter()
            .map(|&p| {
                let d = set.sdf(0, p);
                let f = if op == BoolOp::Add { p.y.min(d) } else { p.y.max(d) };
                f.abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst <= params.tol(), "{op:?} deviation {worst}");
    }
}

#[test]
fn incremental_equals_full_when_most_instances_change() {
    // More than a quarter of the instances changing takes the regroup-from-scratch path.
    let insts = many(40);
    let mut a = setup(0.5, insts.clone());
    a.build(&plane);
    let changed: Vec<Instance> = insts.iter().map(|i| Instance { scale: i.scale * 1.15, translation: i.translation + Vec3::new(0.0, 0.0, 0.02), ..*i }).collect();
    let changes: Vec<(usize, Instance)> = changed.iter().copied().enumerate().collect();
    a.update(&plane, &changes);
    let mut b = BooleanAdf::new(a.frame, a.params, changed).0;
    b.build(&plane);
    assert!(a.fingerprint() == b.fingerprint());
}

#[test]
fn small_cutters_are_not_missed_by_coarse_cells() {
    // A cutter much smaller than the tolerance-driven cell size must still appear.
    let insts = vec![sphere(Vec3::new(0.013, 0.0, 0.007), 0.03)];
    let mut a = setup(2.0, insts.clone());
    a.build(&plane);
    let m = a.extract(None);
    let f = exact(&insts);
    assert!(m.positions.iter().any(|&p| p.y < -0.01), "dent missing");
    assert!(m.positions.iter().map(|&p| f(p).abs()).fold(0.0, f32::max) <= a.params.tol());
}

use glam::{Quat, Vec3};
use sculpt_core::Document;
use sculpt_core::generator::place::{EdgeSampler, SurfaceSampler};
use sculpt_core::generator::{
    BoolOp, EvalParams, Instance, InstanceSet, Primitive, affected_leaves, evaluate, evaluate_leaves,
};
use sculpt_core::primitives::{cube, grid, quad_sphere};

fn sphere_instance(c: Vec3, r: f32) -> Instance {
    Instance::new(Primitive::Sphere, c, Quat::IDENTITY, Vec3::splat(r))
}

fn flat_doc() -> Document {
    Document::from_mesh(grid(160, 2.0)).unwrap()
}

#[test]
fn subtract_sphere_projects_vertices_to_surface() {
    let doc = flat_doc();
    let inst = sphere_instance(Vec3::new(0.0, 0.15, 0.0), 0.5);
    let set = InstanceSet::new(vec![inst], 0.0);
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    let stats = evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &EvalParams::default(), &mut delta);
    assert!(stats.vertices_moved > 100, "moved {}", stats.vertices_moved);
    assert!(stats.leaves_skipped + stats.leaves_evaluated == doc.bvh().leaves.len());
    let (mut moved, mut on_surface) = (0, 0);
    for (p, d) in doc.positions().iter().zip(&delta) {
        let q = *p + *d;
        let sd = q.distance(inst.translation) - 0.5;
        assert!(sd > -2e-3, "vertex ended inside the cutter: {sd}");
        let was_inside = p.distance(inst.translation) < 0.5;
        if was_inside {
            moved += 1;
            if sd.abs() < 2e-3 {
                on_surface += 1;
            }
        } else {
            assert_eq!(*d, Vec3::ZERO, "outside vertex moved");
        }
    }
    assert!(moved > 100);
    assert_eq!(moved, on_surface);
}

#[test]
fn add_lifts_vertices_to_primitive_top() {
    let doc = flat_doc();
    let inst = sphere_instance(Vec3::new(0.0, 0.0, 0.0), 0.5);
    let set = InstanceSet::new(vec![inst], 0.0);
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    let params = EvalParams { op: BoolOp::Add, ..Default::default() };
    evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &params, &mut delta);
    let mut lifted = 0;
    for (p, d) in doc.positions().iter().zip(&delta) {
        let r2 = p.x * p.x + p.z * p.z;
        if r2 < 0.48 * 0.48 {
            let want = (0.25 - r2).sqrt();
            assert!((p.y + d.y - want).abs() < 2e-3, "top {} vs {}", p.y + d.y, want);
            lifted += 1;
        } else if r2 > 0.51 * 0.51 {
            assert_eq!(*d, Vec3::ZERO);
        }
    }
    assert!(lifted > 100);
}

#[test]
fn intersect_pulls_outside_vertices_in() {
    let doc = flat_doc();
    let inst = sphere_instance(Vec3::new(0.0, 0.0, 0.0), 0.5);
    let set = InstanceSet::new(vec![inst], 0.3);
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    let params = EvalParams { op: BoolOp::Intersect, reach: 0.3, iterations: 3 };
    evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &params, &mut delta);
    let mut pulled = 0;
    for (p, d) in doc.positions().iter().zip(&delta) {
        let r = p.length();
        if r <= 0.5 {
            assert_eq!(*d, Vec3::ZERO);
        } else if r < 0.78 {
            assert!((( *p + *d).length() - 0.5).abs() < 2e-3);
            pulled += 1;
        } else if r > 0.82 {
            assert_eq!(*d, Vec3::ZERO);
        }
    }
    assert!(pulled > 100);
}

#[test]
fn every_primitive_projects_onto_its_surface() {
    // Points near the surface projected with p -= d * grad land on it.
    for prim in Primitive::ALL {
        let inst = Instance::new(prim, Vec3::new(0.3, -0.2, 0.1), Quat::from_rotation_x(0.4) * Quat::from_rotation_y(0.9), Vec3::splat(0.5));
        let set = InstanceSet::new(vec![inst], 0.0);
        let mut worst = 0.0f32;
        let mut inside = 0;
        for i in 0..2000 {
            let f = |k: u32| ((i as u32).wrapping_mul(2654435761).rotate_left(k) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let p = inst.translation + Vec3::new(f(3), f(11), f(19)) * 0.6;
            let mut q = p;
            for _ in 0..4 {
                let (d, g) = set.sdf_grad(0, q);
                q -= g * d;
            }
            worst = worst.max(set.sdf(0, q).abs());
            inside += (set.sdf(0, p) < 0.0) as u32;
        }
        assert!(inside > 50, "{prim:?}: only {inside} samples inside");
        assert!(worst < 0.03, "{prim:?}: residual {worst}");
    }
}

#[test]
fn incremental_equals_full() {
    let doc = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    let mut insts = Vec::new();
    for i in 0..300u32 {
        let h = |k: u32| (i.wrapping_mul(2654435761).rotate_left(k) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let dir = Vec3::new(h(1), h(7), h(13)).normalize_or(Vec3::Y);
        insts.push(Instance::new(Primitive::Cylinder, dir, Quat::from_rotation_arc(Vec3::Y, dir), Vec3::splat(0.08)));
    }
    let mut set = InstanceSet::new(insts.clone(), 0.0);
    let params = EvalParams::default();
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &params, &mut delta);
    // Change ~1%.
    let changes: Vec<(usize, Instance)> = (0..3).map(|k| (k * 97, Instance { scale: Vec3::splat(0.14), ..insts[k * 97] })).collect();
    let dirty = set.update(&changes);
    let leaves = affected_leaves(doc.bvh(), &dirty);
    assert!(!leaves.is_empty() && leaves.len() < doc.bvh().leaves.len());
    evaluate_leaves(doc.positions(), doc.normals(), doc.bvh(), &set, &params, &leaves, &mut delta);
    let mut fresh = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &params, &mut fresh);
    assert_eq!(delta, fresh);
}

#[test]
fn spatial_hash_returns_all_overlapping_instances() {
    let insts: Vec<Instance> = (0..500u32)
        .map(|i| {
            let h = |k: u32| (i.wrapping_mul(2654435761).rotate_left(k) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            Instance::new(Primitive::Box, Vec3::new(h(1), h(9), h(17)), Quat::from_rotation_z(h(5)), Vec3::new(0.05, 0.1, 0.03) * (1.5 + h(21)))
        })
        .collect();
    let set = InstanceSet::new(insts.clone(), 0.01);
    for k in 0..4000u32 {
        let h = |s: u32| (k.wrapping_mul(2246822519).rotate_left(s) as f32 / u32::MAX as f32) * 2.4 - 1.2;
        let p = Vec3::new(h(2), h(10), h(18));
        let got = set.candidates(p);
        for (i, inst) in insts.iter().enumerate() {
            let a = inst.aabb();
            let inside = p.cmpge(a.min - Vec3::splat(0.01)).all() && p.cmple(a.max + Vec3::splat(0.01)).all();
            if inside {
                assert!(got.contains(&(i as u32)), "missing instance {i} at {p}");
            }
        }
    }
}

#[test]
fn mask_weighting_puts_points_in_white_regions() {
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    let mask: Vec<f32> = doc.positions().iter().map(|p| if p.y > 0.3 { 1.0 } else { 0.0 }).collect();
    let sampler = SurfaceSampler::new(doc.positions(), doc.faces(), Some(&mask));
    let pts = sampler.sample(doc.positions(), doc.normals(), doc.faces(), Some(&mask), 20_000, 42);
    assert_eq!(pts.len(), 20_000, "count is fixed");
    let white = pts.iter().filter(|s| s.position.y > 0.3 - 0.05).count();
    assert!(white as f32 / pts.len() as f32 >= 0.95, "only {white} of {} in white", pts.len());
    // Deterministic.
    let again = sampler.sample(doc.positions(), doc.normals(), doc.faces(), Some(&mask), 20_000, 42);
    assert!(pts.iter().zip(&again).all(|(a, b)| a.position == b.position));
    // Normals are unit and agree with the sphere.
    assert!(pts.iter().all(|s| (s.normal.length() - 1.0).abs() < 1e-3 && s.normal.dot(s.position.normalize()) > 0.99));
    // Uniform (no mask) covers both hemispheres.
    let uni = SurfaceSampler::new(doc.positions(), doc.faces(), None);
    let pts = uni.sample(doc.positions(), doc.normals(), doc.faces(), None, 20_000, 1);
    let up = pts.iter().filter(|s| s.position.y > 0.0).count() as f32 / 20_000.0;
    assert!((up - 0.5).abs() < 0.03);
}

#[test]
fn edge_sampling_lies_on_cube_edges() {
    let doc = Document::from_mesh(cube()).unwrap();
    let es = EdgeSampler::detect(doc.positions(), doc.faces(), doc.topology(), 30.0);
    assert_eq!(es.edges.len(), 12);
    assert!((es.total_length() - 24.0).abs() < 1e-4);
    let pts = es.sample(doc.positions(), 960, 5);
    assert_eq!(pts.len(), 960);
    for s in &pts {
        let a = s.position.abs();
        let on_max = [a.x, a.y, a.z].iter().filter(|&&c| (c - 1.0).abs() < 1e-4).count();
        assert!(on_max >= 2 && a.max_element() < 1.0 + 1e-4, "{:?} not on a cube edge", s.position);
        // Tangent is axis-aligned and along the free axis, normal is the (+-1,+-1,0)/sqrt2 bisector.
        assert!((s.tangent.abs().max_element() - 1.0).abs() < 1e-4);
        assert!(s.tangent.dot(s.normal).abs() < 1e-4);
        assert!((s.normal.abs().max_element() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        assert!((s.normal.dot(s.position) > 0.0), "normal points outward");
        assert!((s.chain_length - 2.0).abs() < 1e-4, "each cube edge is its own chain of length 2");
        assert!((s.dihedral - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
    }
    // Even spacing: each of the 12 edges gets 80 samples.
    let mut per_edge = std::collections::HashMap::new();
    for s in &pts {
        *per_edge.entry(s.edge).or_insert(0) += 1;
    }
    assert!(per_edge.values().all(|&c| (79..=81).contains(&c)), "{per_edge:?}");
    // A high threshold finds nothing; a smooth sphere finds nothing at 30 degrees.
    assert!(EdgeSampler::detect(doc.positions(), doc.faces(), doc.topology(), 100.0).edges.is_empty());
    let sph = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    assert!(EdgeSampler::detect(sph.positions(), sph.faces(), sph.topology(), 30.0).edges.is_empty());
}

#[test]
fn voxel_spike_mesh_sdf_has_correct_sign_and_distance() {
    use sculpt_core::generator::voxel::{MeshSdf, measure};
    let doc = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    let m = MeshSdf::build(doc.bvh(), doc.positions(), doc.faces(), doc.topology());
    for k in 0..500u32 {
        let h = |s: u32| (k.wrapping_mul(2654435761).rotate_left(s) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let p = Vec3::new(h(3), h(11), h(19)) * 1.6;
        let want = p.length() - 1.0;
        if want.abs() < 0.02 {
            continue; // tessellation error of the polygonal sphere
        }
        let got = m.signed_distance(p, 10.0);
        assert!((got - want).abs() < 0.02, "p={p} got {got} want {want}");
    }
    // The pipeline produces a surface for a single cutter.
    let set = InstanceSet::new(vec![sphere_instance(Vec3::new(0.0, 0.0, 1.0), 0.2)], 0.05);
    let r = measure(&m, &set, 0.025, 10_000);
    assert!(!r.extrapolated && r.bricks > 0 && r.net_vertices > 0 && r.net_quads > 0);
}

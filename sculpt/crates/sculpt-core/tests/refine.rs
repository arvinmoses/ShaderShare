use glam::{Quat, Vec3};
use sculpt_core::Document;
use sculpt_core::generator::refine::{MeshView, RefineParams, check_closed, compose, refine_all, refine_leaf};
use sculpt_core::generator::{EvalParams, Instance, InstanceSet, Primitive, evaluate};
use sculpt_core::primitives::{grid, quad_sphere};

fn view(doc: &Document) -> MeshView<'_> {
    MeshView { positions: doc.positions(), normals: doc.normals(), faces: doc.faces(), bvh: doc.bvh() }
}

/// Base positions with the displacement boolean (spike A) applied, as the engine draws them.
fn displaced(doc: &Document, set: &InstanceSet) -> Vec<Vec3> {
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(doc.positions(), doc.normals(), doc.bvh(), set, &EvalParams::default(), &mut delta);
    doc.positions().iter().zip(&delta).map(|(p, d)| *p + *d).collect()
}

fn rnd(i: u32, k: u32) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// Sphere cutter above a plane: the true intersection is a circle of radius sqrt(R^2 - c^2) in y = 0.
fn flat_case(n: u32, diameter_edges: f32, c_frac: f32, off: (f32, f32)) -> (Document, Instance, f32, f32) {
    let size = 2.0;
    let doc = Document::from_mesh(grid(n, size)).unwrap();
    let e = size / n as f32;
    let r = diameter_edges * e * 0.5;
    let c = c_frac * r;
    let center = Vec3::new(off.0 * e, c, off.1 * e);
    let inst = Instance::new(Primitive::Sphere, center, Quat::IDENTITY, Vec3::splat(r));
    let rho = (r * r - c * c).sqrt();
    (doc, inst, r, rho)
}

#[test]
fn rim_and_wall_vertices_lie_on_the_cutter_surface_and_the_intersection_curve() {
    for (dia, off) in [(1.5, (0.31, 0.17)), (3.0, (0.5, 0.5)), (6.0, (0.13, 0.77)), (3.0, (0.0, 0.0))] {
        let (doc, inst, r, rho) = flat_case(160, dia, 0.25, off);
        let set = InstanceSet::new(vec![inst], 0.0);
        let (patches, _) = refine_all(&view(&doc), &set, &RefineParams::default());
        assert!(!patches.is_empty(), "dia {dia}: no patch");
        let mut crease = 0;
        let mut wall = 0;
        for p in &patches {
            for &ci in &p.crease_vertices {
                let q = p.positions[ci as usize] - inst.translation;
                let on_sphere = (q.length() - r).abs();
                let on_plane = (q.y + inst.translation.y).abs();
                let circle = ((q.x * q.x + q.z * q.z).sqrt() - rho).abs();
                assert!(on_sphere < 2e-5 && on_plane < 2e-5 && circle < 2e-5, "dia {dia}: crease off the curve: {on_sphere} {on_plane} {circle}");
                crease += 1;
            }
            for pos in &p.positions {
                if pos.y < -1e-5 {
                    let sd = (*pos - inst.translation).length() - r;
                    assert!(sd.abs() < 2e-4 * r.max(0.01), "dia {dia}: wall vertex {sd} off the cutter surface");
                    wall += 1;
                } else {
                    assert!(pos.y.abs() < 2e-5, "surface vertex lifted: {}", pos.y);
                }
            }
        }
        assert!(crease >= 12, "dia {dia}: only {crease} rim vertices");
        assert!(wall >= 3, "dia {dia}: crater has {wall} wall vertices");
    }
}

#[test]
fn refinement_beats_the_unrefined_rim() {
    let (doc, inst, _r, rho) = flat_case(160, 3.0, 0.25, (0.3, 0.6));
    let e = 2.0 / 160.0;
    let set = InstanceSet::new(vec![inst], 0.0);
    // Spike A: vertices on the rim ring (unmoved neighbours of moved vertices).
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(doc.positions(), doc.normals(), doc.bvh(), &set, &EvalParams::default(), &mut delta);
    let mut worst_a = 0.0f32;
    for (v, d) in delta.iter().enumerate() {
        if *d != Vec3::ZERO {
            let p = doc.positions()[v];
            let q = p - inst.translation;
            let on_ring = doc.topology().vert_verts.row(v).iter().any(|&n| delta[n as usize] == Vec3::ZERO);
            if on_ring {
                // moved vertex next to an unmoved one: its original position vs the true circle
                worst_a = worst_a.max(((q.x * q.x + q.z * q.z).sqrt() - rho).abs());
            }
        }
    }
    let (patches, _) = refine_all(&view(&doc), &set, &RefineParams::default());
    let mut worst_r = 0.0f32;
    for p in &patches {
        for &ci in &p.crease_vertices {
            let q = p.positions[ci as usize] - inst.translation;
            worst_r = worst_r.max(((q.x * q.x + q.z * q.z).sqrt() - rho).abs());
        }
    }
    assert!(worst_a > 0.05 * e, "baseline unexpectedly sharp: {worst_a}");
    assert!(worst_r < 1e-5 && worst_r < worst_a * 0.01, "refined {worst_r} vs A {worst_a}");
}

fn mixed_instances(n: u32, edge: f32, level_radius: f32) -> Vec<Instance> {
    (0..n)
        .map(|i| {
            let dir = Vec3::new(rnd(i, 1) - 0.5, rnd(i, 2) - 0.5, rnd(i, 3) - 0.5).normalize_or(Vec3::Y);
            let prim = Primitive::ALL[(rnd(i, 4) * 7.0) as usize % 7];
            let half = edge * (0.75 + 3.0 * rnd(i, 5));
            let rot = Quat::from_axis_angle(Vec3::new(rnd(i, 6), rnd(i, 7), rnd(i, 8) + 0.1).normalize(), rnd(i, 9) * std::f32::consts::TAU);
            Instance::new(prim, dir * level_radius * (0.97 + 0.04 * rnd(i, 10)), rot, Vec3::splat(half))
        })
        .collect()
}

#[test]
fn no_cracks_on_a_closed_mesh_spheres_across_leaf_borders() {
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    assert!(doc.bvh().leaves.len() >= 8);
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let insts: Vec<Instance> = (0..400u32)
        .map(|i| {
            let dir = Vec3::new(rnd(i, 1) - 0.5, rnd(i, 2) - 0.5, rnd(i, 3) - 0.5).normalize_or(Vec3::Y);
            let r = edge * (0.75 + 3.0 * rnd(i, 4));
            Instance::new(Primitive::Sphere, dir * (1.0 + (rnd(i, 5) - 0.5) * 0.6 * r), Quat::IDENTITY, Vec3::splat(r))
        })
        .collect();
    let set = InstanceSet::new(insts, 0.0);
    let (patches, touched) = refine_all(&view(&doc), &set, &RefineParams::default());
    assert!(touched >= 8 && patches.len() >= 8, "only {} patches", patches.len());
    // Sanity: the unpatched mesh is closed, so any hole is the patches' fault.
    assert!(check_closed(&compose(doc.positions(), doc.normals(), doc.faces(), &[])).is_closed());
    let c = compose(&displaced(&doc, &set), doc.normals(), doc.faces(), &patches);
    let rep = check_closed(&c);
    assert!(rep.is_closed(), "cracks: {rep:?}");
    let added: usize = patches.iter().map(|p| p.stats.tris_after as usize).sum();
    assert!(added > patches.iter().map(|p| p.stats.tris_before as usize).sum::<usize>());
}

#[test]
fn no_cracks_with_all_primitives() {
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let set = InstanceSet::new(mixed_instances(300, edge, 1.0), 0.0);
    let (patches, _) = refine_all(&view(&doc), &set, &RefineParams::default());
    assert!(patches.len() >= 8);
    let c = compose(&displaced(&doc, &set), doc.normals(), doc.faces(), &patches);
    let rep = check_closed(&c);
    assert!(rep.is_closed(), "cracks: {rep:?}");
}

#[test]
fn leaves_are_independent_and_deterministic() {
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let set = InstanceSet::new(mixed_instances(200, edge, 1.0), 0.0);
    let (a, _) = refine_all(&view(&doc), &set, &RefineParams::default());
    let mut b = Vec::new();
    for l in (0..doc.bvh().leaves.len() as u32).rev() {
        b.extend(refine_leaf(&view(&doc), &set, &RefineParams::default(), l));
    }
    b.sort_by_key(|p| p.leaf);
    let mut a = a;
    a.sort_by_key(|p| p.leaf);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.indices, y.indices);
        assert_eq!(x.positions, y.positions);
    }
}

#[test]
fn untouched_leaves_get_no_patch_and_cost_scales_with_footprint() {
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let inst = Instance::new(Primitive::Sphere, Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, Vec3::splat(2.0 * edge));
    let set = InstanceSet::new(vec![inst], 0.0);
    let (patches, _) = refine_all(&view(&doc), &set, &RefineParams::default());
    let faces: usize = patches.iter().map(|p| p.stats.faces_replaced as usize).sum();
    assert!(!patches.is_empty() && patches.len() <= 4);
    // A cutter 4 edges across touches on the order of 10-60 faces, not a leaf (2048).
    assert!(faces > 5 && faces < 120, "{faces} faces replaced");
}

#[test]
fn crack_check_detects_a_naive_leaf_border_policy() {
    // Control experiment: if leaves skip the faces on their borders, patches meet unrefined
    // neighbours with T-junctions and the check must flag them.
    let doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let insts: Vec<Instance> = (0..400u32)
        .map(|i| {
            let dir = Vec3::new(rnd(i, 1) - 0.5, rnd(i, 2) - 0.5, rnd(i, 3) - 0.5).normalize_or(Vec3::Y);
            Instance::new(Primitive::Sphere, dir, Quat::IDENTITY, Vec3::splat(edge * (1.0 + 3.0 * rnd(i, 4))))
        })
        .collect();
    let set = InstanceSet::new(insts, 0.0);
    let params = RefineParams { debug_skip_leaf_borders: true, ..Default::default() };
    let (patches, _) = refine_all(&view(&doc), &set, &params);
    let rep = check_closed(&compose(&displaced(&doc, &set), doc.normals(), doc.faces(), &patches));
    assert!(rep.boundary_edges > 0, "expected cracks, got {rep:?}");
}

#[test]
fn no_cracks_dense_small_cutters_on_98k_faces() {
    // Stress: many 1.5-4 edge cutters, some within tolerance of mesh vertices and edges, plus
    // overlapping ones. The weld is bit exact, so any mismatch between a patch and the displaced
    // base mesh shows up as a boundary edge.
    let doc = Document::from_mesh(quad_sphere(7, 1.0)).unwrap();
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let mut insts: Vec<Instance> = (0..3000u32)
        .map(|i| {
            let dir = Vec3::new(rnd(i, 1) - 0.5, rnd(i, 2) - 0.5, rnd(i, 3) - 0.5).normalize_or(Vec3::Y);
            let r = edge * (0.75 + 1.25 * rnd(i, 4));
            Instance::new(Primitive::Sphere, dir * (1.0 + 0.3 * r * (rnd(i, 5) - 0.5)), Quat::IDENTITY, Vec3::splat(r))
        })
        .collect();
    // Cutters centred (almost) on mesh vertices.
    for v in (0..doc.vertex_count()).step_by(97).take(300) {
        // Lifted by a quarter radius: a centre exactly on the surface makes the radial projection
        // of spike A collapse the whole crater floor onto its rim (documented limitation).
        let c = doc.positions()[v] + doc.normals()[v] * (0.25 * edge * 1.1);
        insts.push(Instance::new(Primitive::Sphere, c, Quat::IDENTITY, Vec3::splat(edge * 1.1)));
    }
    let set = InstanceSet::new(insts, 0.0);
    let (patches, _) = refine_all(&view(&doc), &set, &RefineParams::default());
    let a_only = check_closed(&compose(&displaced(&doc, &set), doc.normals(), doc.faces(), &[]));
    let rep = check_closed(&compose(&displaced(&doc, &set), doc.normals(), doc.faces(), &patches));
    assert!(rep.is_closed(), "cracks: {rep:?} (A alone: {a_only:?})");
}

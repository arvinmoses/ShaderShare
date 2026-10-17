use sculpt_core::document::SolidSettings;
use sculpt_core::generator::layer::{GeneratorKind, GeneratorSpec, NoiseSpec};
use sculpt_core::generator::remesh::non_manifold_edges;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

fn spec(op: BoolOp, primitive: Primitive, count: u32, size: f32, lift: f32, seed: u64) -> GeneratorSpec {
    GeneratorSpec { primitive, op, count, seed, size, lift, min_spacing: size * 1.2, tilt: 50.0, ..GeneratorSpec::default() }
}

fn settings() -> SolidSettings {
    SolidSettings::new(0.05)
}

/// Volume of a closed quad mesh by the divergence theorem (each quad as two triangles).
fn volume(mesh: &sculpt_core::mesh::PolyMesh) -> f32 {
    let p = &mesh.positions;
    mesh.faces
        .iter()
        .map(|f| {
            let t = |a: u32, b: u32, c: u32| p[a as usize].dot(p[b as usize].cross(p[c as usize])) / 6.0;
            t(f[0], f[1], f[2]) + t(f[0], f[2], f[3])
        })
        .sum()
}

#[test]
fn solidify_without_generators_reproduces_the_surface() {
    let doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    let (mesh, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 0);
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
    let v = volume(&mesh);
    assert!((v - 4.19).abs() < 0.2, "a unit ball has volume 4.19, got {v}");
}

#[test]
fn subtracting_cutters_removes_volume_and_stays_closed() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    let (mesh, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 1);
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
    let v = volume(&mesh);
    assert!(v > 2.0 && v < 4.0, "cuts take some volume away, got {v}");
}

#[test]
fn adding_cutters_adds_volume() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("bumps", Placement::Top, spec(BoolOp::Add, Primitive::Sphere, 10, 0.25, 0.0, 6)).unwrap();
    let (mesh, _) = doc.solidify(&settings()).unwrap();
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
    assert!(volume(&mesh) > 4.4);
}

#[test]
fn a_second_layer_scatters_on_the_surface_the_first_made() {
    // Chips placed after big cuts must still land on the shape, not on the original ball: removing the first layer's
    // effect from the scatter would leave most chips cutting thin air, so the chipped rock would barely differ.
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("big", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 6, 0.6, 0.7, 7)).unwrap();
    let (before, _) = doc.solidify(&settings()).unwrap();
    doc.insert_generator("chips", Placement::Top, spec(BoolOp::Subtract, Primitive::Sphere, 40, 0.1, 0.6, 8)).unwrap();
    let (after, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 2);
    assert_eq!(non_manifold_edges(&after.faces), 0);
    assert!(volume(&after) < volume(&before) - 0.02, "chips took {} of volume", volume(&before) - volume(&after));
}

#[test]
fn hidden_generator_layers_are_left_out() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    let id = doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    let mut meta = doc.layer(id).unwrap().meta();
    meta.visible = false;
    doc.set_layer_meta(id, meta, false).unwrap();
    let (_, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 0);
}

#[test]
fn a_second_solidify_reuses_the_base_distances_and_gives_the_same_mesh() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    let (first, a) = doc.solidify(&settings()).unwrap();
    let (second, b) = doc.solidify(&settings()).unwrap();
    assert!(a.base_misses > 0 && a.base_hits < a.base_misses, "a cold call computes: {a:?}");
    assert_eq!(b.base_misses, 0, "nothing new to compute: {b:?}");
    assert!(b.base_hits > 0);
    assert_eq!(first.faces, second.faces);
    assert_eq!(first.positions, second.positions);
}

#[test]
fn editing_a_cutter_keeps_most_of_the_cache() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    let id = doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    doc.solidify(&settings()).unwrap();
    let mut meta = doc.layer(id).unwrap().meta();
    let mut g = meta.generator.clone().unwrap();
    g.size = 0.45;
    meta.generator = Some(g);
    doc.set_layer_meta(id, meta, false).unwrap();
    let (_, stats) = doc.solidify(&settings()).unwrap();
    assert!(stats.base_hits > stats.base_misses * 4, "most distances are reused: {stats:?}");
}

#[test]
fn changing_the_surface_invalidates_the_cache() {
    let mut doc = Document::from_mesh(quad_sphere(3, 1.0)).unwrap();
    doc.solidify(&settings()).unwrap();
    // Subdividing moves the surface's vertices, so the old distances must not be reused for it.
    doc.subdivide().unwrap();
    let (_, stats) = doc.solidify(&settings()).unwrap();
    assert!(stats.base_misses > 0, "{stats:?}");
}

fn noise_spec(amplitude: f32) -> GeneratorSpec {
    GeneratorSpec { kind: GeneratorKind::Noise, noise: NoiseSpec { amplitude, scale: 4.0, octaves: 3, seed: 2 }, ..GeneratorSpec::default() }
}

#[test]
fn a_noise_layer_roughens_the_live_surface_along_the_normals() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    let before: Vec<f32> = doc.positions().iter().map(|p| p.length()).collect();
    doc.insert_generator("grain", Placement::Top, noise_spec(0.1)).unwrap();
    let moved: Vec<f32> = doc.positions().iter().zip(&before).map(|(p, b)| p.length() - b).collect();
    let peak = moved.iter().fold(0.0f32, |m, d| m.max(d.abs()));
    assert!(peak > 0.02, "the surface moves, peak {peak}");
    assert!(peak <= 0.06, "by at most half the peak-to-peak amplitude (and a little for normals), peak {peak}");
    let mean = moved.iter().sum::<f32>() / moved.len() as f32;
    assert!(mean.abs() < 0.02, "noise is centred, mean {mean}");
}

#[test]
fn a_noise_layer_can_be_turned_off_and_back() {
    let mut doc = Document::from_mesh(quad_sphere(3, 1.0)).unwrap();
    let id = doc.insert_generator("grain", Placement::Top, noise_spec(0.1)).unwrap();
    let rough: Vec<_> = doc.positions().to_vec();
    let mut meta = doc.layer(id).unwrap().meta();
    meta.visible = false;
    doc.set_layer_meta(id, meta, false).unwrap();
    assert!(doc.positions().iter().all(|p| (p.length() - 1.0).abs() < 1e-3), "hidden noise leaves the sphere alone");
    let mut meta = doc.layer(id).unwrap().meta();
    meta.visible = true;
    doc.set_layer_meta(id, meta, false).unwrap();
    assert_eq!(doc.positions(), rough.as_slice());
}

#[test]
fn solidify_applies_noise_in_stack_order_and_stays_manifold() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    let (smooth, _) = doc.solidify(&settings()).unwrap();
    doc.insert_generator("grain", Placement::Top, noise_spec(0.12)).unwrap();
    let (rough, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 1);
    assert_eq!(non_manifold_edges(&rough.faces), 0);
    let spread = |m: &sculpt_core::mesh::PolyMesh| {
        let r: Vec<f32> = m.positions.iter().map(|p| p.length()).collect();
        r.iter().cloned().fold(0.0, f32::max) - r.iter().cloned().fold(f32::MAX, f32::min)
    };
    assert!(spread(&rough) > spread(&smooth) + 0.04, "noise roughens: {} against {}", spread(&rough), spread(&smooth));
}

#[test]
fn noise_after_cuts_roughens_the_cut_faces_too() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 6, 0.5, 0.7, 5)).unwrap();
    doc.insert_generator("grain", Placement::Top, noise_spec(0.08)).unwrap();
    let (mesh, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 2);
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
}

#[test]
fn a_snapshot_runs_on_another_thread_and_warms_the_documents_cache() {
    let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    let input = doc.solid_input();
    assert!(input.has_layers());
    // The document keeps changing while the snapshot is solidified elsewhere; the snapshot does not see it.
    let worker = std::thread::spawn(move || input.run(&settings()).unwrap());
    doc.insert_generator("more", Placement::Top, spec(BoolOp::Subtract, Primitive::Sphere, 20, 0.1, 0.6, 9)).unwrap();
    let (from_thread, thread_stats) = worker.join().unwrap();
    assert_eq!(thread_stats.layers, 1);
    // Solidifying the document afterwards starts from the distances the thread computed.
    let (_, stats) = doc.solidify(&settings()).unwrap();
    assert_eq!(stats.layers, 2);
    assert!(stats.base_hits > stats.base_misses, "the thread's distances are reused: {stats:?}");
    assert_eq!(non_manifold_edges(&from_thread.faces), 0);
}

#[test]
fn after_a_brush_stroke_the_kept_distances_give_exactly_the_cold_result() {
    use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
    let path: Vec<(glam::Vec3, f32)> = (0..=20).map(|i| (glam::Vec3::new(-0.3 + 0.6 * i as f32 / 20.0, 1.0, 0.0), 1.0)).collect();
    let stack = |doc: &mut Document| {
        doc.insert_generator("cuts", Placement::Top, spec(BoolOp::Subtract, Primitive::Box, 8, 0.4, 0.7, 5)).unwrap();
    };
    // Warm: solidify, stroke, solidify again (the second reuses what the stroke cannot have changed).
    let mut warm = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    stack(&mut warm);
    warm.solidify(&settings()).unwrap();
    brush::stroke(&mut warm, &mut ClayBuildup::default(), &BrushSettings { radius: 0.25, strength: 0.6, ..Default::default() }, &path).unwrap();
    let (warm_mesh, warm_stats) = warm.solidify(&settings()).unwrap();
    // Cold: the same stroke, solidified once from nothing.
    let mut cold = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    stack(&mut cold);
    brush::stroke(&mut cold, &mut ClayBuildup::default(), &BrushSettings { radius: 0.25, strength: 0.6, ..Default::default() }, &path).unwrap();
    let (cold_mesh, cold_stats) = cold.solidify(&settings()).unwrap();
    assert!(warm_stats.base_hits > warm_stats.base_misses, "most distances survive a local stroke: {warm_stats:?}");
    assert!(cold_stats.base_hits * 50 < cold_stats.base_misses, "a cold run computes nearly everything (block centres are looked up twice): {cold_stats:?}");
    assert_eq!(warm_mesh.faces, cold_mesh.faces);
    assert_eq!(warm_mesh.positions, cold_mesh.positions);
}

#[test]
fn intersecting_with_cutters_stays_closed() {
    // Intersect keeps only what lies inside some cutter, so far from every cutter the field must say outside. Treating
    // "no cutter nearby" as "no change" left slabs of the base behind and broke the surface open.
    for seed in [5, 9] {
        let mut doc = Document::from_mesh(quad_sphere(4, 1.0)).unwrap();
        doc.insert_generator("keep", Placement::Top, spec(BoolOp::Intersect, Primitive::Box, 10, 0.35, 0.0, seed)).unwrap();
        let (mesh, _) = doc.solidify(&settings()).unwrap();
        assert!(!mesh.faces.is_empty());
        assert_eq!(non_manifold_edges(&mesh.faces), 0, "seed {seed}");
        assert!(volume(&mesh) < 4.19 * 0.8, "only the parts inside cutters remain, volume {}", volume(&mesh));
    }
}

#[test]
fn the_app_scene_with_intersect_and_noise_stays_closed() {
    // The scene that tore in the app: 14 lifted boxes intersected with the sphere, a noise layer on top.
    let mut doc = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    doc.insert_generator("keep", Placement::Top, GeneratorSpec { primitive: Primitive::Box, op: BoolOp::Intersect, count: 14, size: 0.45, lift: 0.7, seed: 5, ..GeneratorSpec::default() }).unwrap();
    doc.insert_generator("grain", Placement::Top, noise_spec(0.03)).unwrap();
    let (mesh, _) = doc.solidify(&SolidSettings::new(0.02)).unwrap();
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
}

#[test]
fn loose_slivers_are_dropped_and_zero_keeps_them() {
    use sculpt_core::generator::remesh::{RemeshParams, drop_small_pieces, remesh};
    // A ball and a speck well apart.
    let field = |p: glam::Vec3| (p.length() - 0.8).min((p - glam::Vec3::new(1.15, 0.0, 0.0)).length() - 0.06);
    let bounds = sculpt_core::geom::Aabb { min: glam::Vec3::splat(-1.3), max: glam::Vec3::splat(1.4) };
    let whole = remesh(&field, bounds, &RemeshParams::new(0.04));
    let mut kept = whole.clone();
    assert_eq!(drop_small_pieces(&mut kept, 0.0), 0);
    assert_eq!(kept.quads.len(), whole.quads.len());
    let mut cleaned = whole.clone();
    assert_eq!(drop_small_pieces(&mut cleaned, 0.01), 1);
    assert!(cleaned.quads.len() < whole.quads.len());
    assert!(cleaned.positions.iter().all(|p| p.x < 1.0), "only the speck went");
    assert_eq!(non_manifold_edges(&cleaned.quads), 0);
}

#[test]
fn solidified_quads_are_never_slivers() {
    // Thin quads are what a brush turns inside out first; relaxing after extraction spreads them out.
    let mut doc = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, GeneratorSpec { primitive: Primitive::Box, op: BoolOp::Subtract, count: 14, size: 0.45, lift: 0.7, seed: 5, ..GeneratorSpec::default() }).unwrap();
    doc.insert_generator("grain", Placement::Top, noise_spec(0.03)).unwrap();
    let (mesh, _) = doc.solidify(&SolidSettings::new(0.02)).unwrap();
    let p = &mesh.positions;
    let thinnest = mesh
        .faces
        .iter()
        .map(|f| {
            let e: Vec<f32> = (0..4).map(|k| p[f[k] as usize].distance(p[f[(k + 1) % 4] as usize])).collect();
            e.iter().cloned().fold(f32::MAX, f32::min) / e.iter().cloned().fold(0.0, f32::max)
        })
        .fold(1.0f32, f32::min);
    assert!(thinnest > 0.05, "thinnest quad edge ratio {thinnest}");
    assert_eq!(non_manifold_edges(&mesh.faces), 0);
}

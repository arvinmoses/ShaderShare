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
    SolidSettings { voxel: 0.05 }
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

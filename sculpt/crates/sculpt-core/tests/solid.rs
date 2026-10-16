use sculpt_core::document::SolidSettings;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::remesh::non_manifold_edges;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

fn spec(op: BoolOp, primitive: Primitive, count: u32, size: f32, lift: f32, seed: u64) -> GeneratorSpec {
    GeneratorSpec { primitive, op, count, seed, size, lift, min_spacing: size * 1.2, tilt: 50.0, ..GeneratorSpec::default() }
}

fn settings() -> SolidSettings {
    SolidSettings { voxel: 0.05, grain: None }
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

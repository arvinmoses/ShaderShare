use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::document::SolidSettings;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::remesh::non_manifold_edges;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

const VOXEL: f32 = 0.04;

fn stack() -> Document {
    let mut doc = Document::from_mesh(quad_sphere(5, 1.0)).unwrap();
    doc.insert_generator("cuts", Placement::Top, GeneratorSpec { primitive: Primitive::Box, op: BoolOp::Subtract, count: 6, size: 0.4, lift: 0.7, seed: 5, min_spacing: 0.5, tilt: 40.0, ..GeneratorSpec::default() }).unwrap();
    doc
}

fn remesh(source: Document) -> Document {
    let (mesh, _) = source.solidify(&SolidSettings::new(VOXEL)).unwrap();
    Document::remeshed(mesh, source, VOXEL).unwrap()
}

/// A clay stroke across the top of the model.
fn stroke(doc: &mut Document) {
    let path: Vec<(Vec3, f32)> = (0..=20).map(|i| (Vec3::new(-0.3 + 0.6 * i as f32 / 20.0, 1.0, 0.1), 1.0)).collect();
    brush::stroke(doc, &mut ClayBuildup::default(), &BrushSettings { radius: 0.25, strength: 0.8, ..Default::default() }, &path).unwrap();
}

fn top_height(doc: &Document) -> f32 {
    doc.positions().iter().filter(|p| p.x.abs() < 0.2 && p.z.abs() < 0.2).map(|p| p.y).fold(f32::MIN, f32::max)
}

#[test]
fn remeshing_gives_a_clean_mesh_with_a_sculpt_layer_and_keeps_the_stack() {
    let top = remesh(stack());
    assert!(top.has_remesh_source());
    assert_eq!(top.layers().len(), 1);
    assert!(top.layers()[0].is_sculpt_target());
    assert_eq!(top.active_layer(), Some(top.layers()[0].id));
    assert!(top.layers().iter().all(|l| !l.is_generator()));
    assert_eq!(non_manifold_edges(top.faces()), 0);
}

#[test]
fn strokes_on_the_remeshed_mesh_go_into_the_sculpt_layer() {
    let mut top = remesh(stack());
    let before = top_height(&top);
    stroke(&mut top);
    assert!(top_height(&top) > before + 0.01, "the stroke raises the surface");
    assert!(!top.layers()[0].allocated_leaves().is_empty(), "the stroke is in the sculpt layer, not the base");
}

#[test]
fn rebuilding_from_an_unchanged_stack_keeps_the_sculpt_exactly() {
    let mut top = remesh(stack());
    stroke(&mut top);
    let sculpted = top.positions().to_vec();
    let source = top.take_remesh_source().unwrap();
    let again = top.rebuilt_from_source(source).unwrap();
    assert!(again.has_remesh_source());
    assert_eq!(again.vertex_count(), top.vertex_count());
    // Same mesh again, so every vertex finds itself and keeps its stroke (up to float rounding).
    let key = |p: &Vec3| (*p * 1e4).round().as_i64vec3().to_array();
    let known: std::collections::HashSet<_> = sculpted.iter().map(key).collect();
    for p in again.positions().iter().filter(|p| !known.contains(&key(p))) {
        let nearest = sculpted.iter().map(|q| q.distance(*p)).fold(f32::MAX, f32::min);
        assert!(nearest < 1e-5, "a vertex moved by {nearest}");
    }
}

#[test]
fn editing_the_stack_below_carries_the_sculpt_onto_the_new_mesh() {
    let mut top = remesh(stack());
    let unsculpted = top_height(&top);
    stroke(&mut top);
    let sculpted = top_height(&top);
    let mut source = top.take_remesh_source().unwrap();
    // Change the cuts: a different seed puts them elsewhere.
    let id = source.layers().iter().find(|l| l.is_generator()).unwrap().id;
    let mut meta = source.layer(id).unwrap().meta();
    let mut g = meta.generator.clone().unwrap();
    g.seed = 11;
    meta.generator = Some(g);
    source.set_layer_meta(id, meta, false).unwrap();
    let rebuilt = top.rebuilt_from_source(source).unwrap();
    assert_eq!(non_manifold_edges(rebuilt.faces()), 0);
    assert_eq!(rebuilt.layers().len(), 1);
    let height = top_height(&rebuilt);
    assert!(height > unsculpted + 0.5 * (sculpted - unsculpted), "the stroke survives: {unsculpted} -> {sculpted}, rebuilt {height}");
}


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


/// Edit the stack below `top` with `change` and carry the sculpt across as volume.
fn round_trip_as_volume(top: &mut Document, change: impl FnOnce(&mut Document)) -> Document {
    let mut source = top.take_remesh_source().unwrap();
    let settings = SolidSettings::new(VOXEL);
    let old = source.solid_input().field(&settings).unwrap();
    change(&mut source);
    top.rebuilt_as_volume(&old, source, &settings).unwrap()
}

fn reseed(source: &mut Document, seed: u64) {
    let id = source.layers().iter().find(|l| l.is_generator()).unwrap().id;
    let mut meta = source.layer(id).unwrap().meta();
    let mut g = meta.generator.clone().unwrap();
    g.seed = seed;
    meta.generator = Some(g);
    source.set_layer_meta(id, meta, false).unwrap();
}

fn stretched_edges(doc: &Document, limit: f32) -> usize {
    doc.faces().iter().flat_map(|f| (0..4).map(move |k| (f[k], f[(k + 1) % 4]))).filter(|&(a, b)| doc.positions()[a as usize].distance(doc.positions()[b as usize]) > limit).count()
}

#[test]
fn volume_carry_keeps_the_sculpt_and_bakes_it_into_the_base() {
    let mut top = remesh(stack());
    let unsculpted = top_height(&top);
    stroke(&mut top);
    let sculpted = top_height(&top);
    let out = round_trip_as_volume(&mut top, |_| {});
    assert_eq!(non_manifold_edges(out.faces()), 0);
    assert!(out.has_remesh_source());
    assert!(out.layers().iter().all(|l| l.allocated_leaves().is_empty()), "the strokes are in the base now, the layers start empty");
    let height = top_height(&out);
    assert!((height - sculpted).abs() < 2.0 * VOXEL, "the sculpt is kept: {unsculpted} -> {sculpted}, now {height}");
}

#[test]
fn volume_carry_over_a_changed_stack_cannot_fold() {
    let mut top = remesh(stack());
    let unsculpted = top_height(&top);
    stroke(&mut top);
    let sculpted = top_height(&top);
    let out = round_trip_as_volume(&mut top, |s| reseed(s, 11));
    assert_eq!(non_manifold_edges(out.faces()), 0);
    assert_eq!(stretched_edges(&out, 4.0 * VOXEL), 0, "no stretched edges: it is one extraction");
    let height = top_height(&out);
    assert!(height > unsculpted + 0.5 * (sculpted - unsculpted), "the stroke survives: {unsculpted} -> {sculpted}, now {height}");
}

#[test]
fn volume_carry_accumulates_over_rounds() {
    let mut top = remesh(stack());
    let side = |doc: &Document| doc.positions().iter().filter(|p| p.y.abs() < 0.2 && p.z.abs() < 0.2 && p.x > 0.0).map(|p| p.x).fold(f32::MIN, f32::max);
    let side_before = side(&top);
    stroke(&mut top);
    let first = top_height(&top);
    let mut out = round_trip_as_volume(&mut top, |s| reseed(s, 11));
    // A second stroke elsewhere, on the side, then another round.
    let path: Vec<(Vec3, f32)> = (0..=20).map(|i| (Vec3::new(1.0, -0.3 + 0.6 * i as f32 / 20.0, 0.1), 1.0)).collect();
    out.set_active_layer(Some(out.layers()[0].id)).unwrap();
    brush::stroke(&mut out, &mut ClayBuildup::default(), &BrushSettings { radius: 0.25, strength: 0.8, ..Default::default() }, &path).unwrap();
    let side_sculpted = side(&out);
    assert!(side_sculpted > side_before + 0.01);
    let again = round_trip_as_volume(&mut out, |s| reseed(s, 12));
    assert_eq!(non_manifold_edges(again.faces()), 0);
    assert!(top_height(&again) > first - 3.0 * VOXEL, "the first round's stroke is still there");
    assert!(side(&again) > side_before + 0.5 * (side_sculpted - side_before), "and the second's");
}

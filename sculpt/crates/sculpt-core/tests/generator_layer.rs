//! The generator layer inside the document: composition, strength, undo, save/load, subdivision.

use glam::Vec3;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::io::project;
use sculpt_core::lod::{LodParams, View};
use sculpt_core::primitives::{quad_sphere, quad_sphere_res};
use sculpt_core::{Document, LayerId, LayerKind, Placement};

fn sphere_doc(level: u32) -> Document {
    Document::from_mesh(quad_sphere(level, 1.0)).unwrap()
}

fn dents() -> GeneratorSpec {
    GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 30, seed: 7, size: 0.18, lift: 0.6, random_spin: true }
}

fn min_radius(doc: &Document) -> f32 {
    doc.positions().iter().map(|p| p.length()).fold(f32::MAX, f32::min)
}

fn max_radius(doc: &Document) -> f32 {
    doc.positions().iter().map(|p| p.length()).fold(0.0, f32::max)
}

fn moved(doc: &Document) -> usize {
    doc.positions().iter().zip(doc.base_positions()).filter(|(p, b)| (**p - **b).length() > 1e-6).count()
}

fn same(a: &[Vec3], b: &[Vec3], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (*x - *y).length() <= tol)
}

#[test]
fn subtract_cuts_into_the_surface_and_leaves_the_base_alone() {
    let mut doc = sphere_doc(3);
    let before = doc.positions().to_vec();
    let id = doc.insert_generator("Dents", Placement::Top, dents()).unwrap();
    assert_eq!(doc.layer(id).unwrap().kind, LayerKind::Generator);
    assert!(min_radius(&doc) < 0.99, "the cutters should push vertices inward, min radius {}", min_radius(&doc));
    assert!(moved(&doc) > 20);
    assert!(max_radius(&doc) <= 1.0 + 1e-4, "subtract never raises the surface");
    assert!(same(doc.base_positions(), &before, 0.0), "the base stays untouched");
}

#[test]
fn add_raises_the_surface() {
    let mut doc = sphere_doc(3);
    let spec = GeneratorSpec { op: BoolOp::Add, lift: 0.4, ..dents() };
    doc.insert_generator("Bumps", Placement::Top, spec).unwrap();
    assert!(max_radius(&doc) > 1.01, "max radius {}", max_radius(&doc));
}

#[test]
fn placement_is_deterministic_and_seed_dependent() {
    let (mut a, mut b, mut c) = (sphere_doc(3), sphere_doc(3), sphere_doc(3));
    a.insert_generator("g", Placement::Top, dents()).unwrap();
    b.insert_generator("g", Placement::Top, dents()).unwrap();
    c.insert_generator("g", Placement::Top, GeneratorSpec { seed: 8, ..dents() }).unwrap();
    assert!(same(a.positions(), b.positions(), 0.0), "same spec, same result");
    assert!(!same(a.positions(), c.positions(), 1e-6), "a different seed gives a different scatter");
}

#[test]
fn strength_and_visibility_scale_the_effect() {
    let mut doc = sphere_doc(3);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    let full = min_radius(&doc);
    doc.set_layer_opacity(id, 0.5).unwrap();
    let half = min_radius(&doc);
    assert!(half > full && half < 1.0, "half strength cuts less: {full} < {half} < 1");
    doc.set_layer_opacity(id, 1.0).unwrap();
    doc.set_layer_visible(id, false).unwrap();
    assert!(same(doc.positions(), doc.base_positions(), 1e-6), "hidden means untouched");
    doc.set_layer_visible(id, true).unwrap();
    assert!((min_radius(&doc) - full).abs() < 1e-6, "showing it again restores the cut");
}

#[test]
fn undo_and_redo_cover_insert_and_edit() {
    let mut doc = sphere_doc(3);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    let cut = doc.positions().to_vec();
    doc.set_generator(id, GeneratorSpec { count: 80, seed: 3, ..dents() }).unwrap();
    let edited = doc.positions().to_vec();
    assert!(!same(&cut, &edited, 1e-6));

    assert!(doc.undo());
    assert!(same(doc.positions(), &cut, 1e-6), "undo of the edit restores the previous scatter");
    assert!(doc.undo());
    assert!(same(doc.positions(), doc.base_positions(), 1e-6), "undo of the insert removes the layer");
    assert!(doc.redo() && doc.redo());
    assert!(same(doc.positions(), &edited, 1e-6), "redo replays both steps");
}

#[test]
fn incremental_edit_equals_a_fresh_build() {
    let mut a = sphere_doc(3);
    let id = a.insert_generator("g", Placement::Top, dents()).unwrap();
    let final_spec = GeneratorSpec { count: 55, seed: 11, size: 0.12, primitive: Primitive::Cylinder, ..dents() };
    a.set_generator(id, final_spec.clone()).unwrap();
    let mut b = sphere_doc(3);
    b.insert_generator("g", Placement::Top, final_spec).unwrap();
    assert!(same(a.positions(), b.positions(), 1e-6), "only the touched leaves were redone, and the result is the same");
}

#[test]
fn save_and_load_round_trip() {
    let mut doc = sphere_doc(3);
    let id = doc.insert_generator("Dents", Placement::Top, dents()).unwrap();
    doc.set_layer_opacity(id, 0.75).unwrap();
    let dir = std::env::temp_dir().join(format!("sculpt-gen-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    project::save(&doc, &dir).unwrap();
    let loaded = project::load(&dir).unwrap();
    let layer = loaded.layers().iter().find(|l| l.name == "Dents").expect("generator layer loads");
    assert_eq!(layer.kind, LayerKind::Generator);
    assert_eq!(layer.generator, Some(dents()));
    assert!(same(loaded.positions(), doc.positions(), 1e-5), "the loaded surface matches");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn survives_subdivision() {
    let mut doc = sphere_doc(2);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    doc.subdivide().unwrap();
    let layer = doc.layer(id).unwrap();
    assert_eq!(layer.generator.as_ref(), Some(&dents()), "the spec is kept");
    assert!(min_radius(&doc) < 0.99, "the cutters are re-placed on the denser surface");
}

#[test]
fn procedural_layers_refuse_destructive_edits() {
    let mut doc = sphere_doc(2);
    let sculpt = doc.insert_layer("Sculpt", Placement::Top).unwrap();
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    assert!(doc.flatten_layer(id).is_err());
    assert!(doc.merge_down(id).is_err());
    let above = doc.insert_layer("Above", Placement::Top).unwrap();
    assert!(doc.merge_down(above).is_err(), "merging into a generator layer is refused");
    assert!(doc.set_generator(sculpt, dents()).is_err(), "a sculpt layer has no generator");
}

#[test]
fn a_generator_layer_is_never_the_sculpt_target() {
    let mut doc = sphere_doc(2);
    let sculpt = doc.insert_layer("Sculpt", Placement::Top).unwrap();
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    assert_eq!(doc.active_layer(), Some(sculpt), "inserting a generator does not steal the sculpt target");
    assert!(doc.set_active_layer(Some(id)).is_err(), "strokes cannot be aimed at a generator layer");
    let copy = doc.duplicate_layer(id).unwrap();
    assert_eq!(doc.active_layer(), Some(sculpt), "duplicating a generator keeps the target");
    doc.delete_layer(sculpt).unwrap();
    assert_ne!(doc.active_layer(), Some(id));
    assert_ne!(doc.active_layer(), Some(copy), "deleting the target never falls onto a generator");
}

fn view_from(eye: Vec3, fov_deg: f32, px_height: f32) -> View {
    let proj = glam::camera::rh::proj::directx::perspective(fov_deg.to_radians(), 1.0, 0.01, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    View::from_view_proj(proj * view, eye, px_height / (2.0 * (fov_deg.to_radians() / 2.0).tan()))
}

#[test]
fn a_whole_mesh_edit_never_draws_out_of_date_lod_patches() {
    // A generator edit reaches nearly every leaf at once. The level-of-detail cut must not fall back to simplified
    // patches whose connectivity was chosen for the old surface: they tear the new dents into shards.
    let mut doc = Document::from_mesh(quad_sphere_res(70, 1.0)).unwrap();
    let id = doc.insert_generator("g", Placement::Top, GeneratorSpec { count: 400, size: 0.08, ..dents() }).unwrap();
    doc.build_lod(LodParams::default());
    doc.set_generator(id, GeneratorSpec { count: 500, seed: 9, size: 0.08, ..dents() }).unwrap();
    let tree = doc.lod().unwrap();
    assert!(tree.stale_nodes() > 1, "the edit should outdate the patches above the touched leaves");

    let view = view_from(Vec3::new(0.0, 0.0, 3.0), 35.0, 1080.0);
    let natural = tree.select(doc.bvh(), &view, 1.0, u64::MAX);
    // A budget well under what the stale areas need at full detail.
    let squeezed = tree.select(doc.bvh(), &view, 1.0, natural.triangles / 8);
    let stale_drawn = squeezed.ids.iter().filter(|&&n| tree.is_stale(n)).count();
    assert_eq!(stale_drawn, 0, "{stale_drawn} stale patches were drawn to stay inside the triangle budget");
}

#[test]
fn normals_match_a_full_recompute_after_generator_edits() {
    // Edits refresh normals only where something moved (and around it); the result must equal recomputing everything.
    let mut doc = sphere_doc(4);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    let check = |doc: &Document, what: &str| {
        let fresh = sculpt_core::mesh::compute_normals(doc.positions(), doc.faces(), doc.topology());
        let worst = doc.normals().iter().zip(&fresh).map(|(a, b)| (*a - *b).length()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{what}: stored normals differ from a full recompute by {worst}");
    };
    check(&doc, "after insert");
    doc.set_generator(id, GeneratorSpec { count: 90, seed: 3, size: 0.12, ..dents() }).unwrap();
    check(&doc, "after changing count, seed and size");
    doc.set_generator(id, GeneratorSpec { op: BoolOp::Add, lift: 0.4, ..dents() }).unwrap();
    check(&doc, "after switching to add");
    assert!(doc.undo());
    check(&doc, "after undo");
    doc.set_layer_visible(id, false).unwrap();
    check(&doc, "after hiding the layer");
}

#[test]
fn lazy_normals_defer_the_work_and_ensure_catches_up() {
    let mut doc = sphere_doc(4);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    doc.set_lazy_normals(true);
    doc.set_generator(id, GeneratorSpec { count: 90, seed: 3, ..dents() }).unwrap();
    assert!(doc.normals_stale(), "a generator edit leaves normals out of date while lazy");
    let fresh = |doc: &Document| {
        let full = sculpt_core::mesh::compute_normals(doc.positions(), doc.faces(), doc.topology());
        doc.normals().iter().zip(&full).map(|(a, b)| (*a - *b).length()).fold(0.0f32, f32::max)
    };
    assert!(fresh(&doc) > 1e-3, "the stored normals really are stale before ensure");
    doc.ensure_normals();
    assert!(!doc.normals_stale());
    assert!(fresh(&doc) < 1e-5, "ensure brings them to a full recompute");
    // Beginning a stroke is also a reader.
    doc.set_generator(id, GeneratorSpec { count: 20, seed: 5, ..dents() }).unwrap();
    assert!(doc.normals_stale());
    doc.begin_stroke("stroke");
    assert!(!doc.normals_stale(), "begin_stroke catches normals up");
    doc.end_stroke();
    // Turning laziness off also catches up, and later edits are eager again.
    doc.set_generator(id, GeneratorSpec { count: 70, seed: 6, ..dents() }).unwrap();
    doc.set_lazy_normals(false);
    assert!(!doc.normals_stale() && fresh(&doc) < 1e-5);
    doc.set_generator(id, GeneratorSpec { count: 30, seed: 8, ..dents() }).unwrap();
    assert!(!doc.normals_stale() && fresh(&doc) < 1e-5);
}

fn dense_doc() -> Document {
    // 6 * 110^2 = 72,600 quads: a few dozen leaves, enough for ordering and partial progress.
    Document::from_mesh(quad_sphere_res(110, 1.0)).unwrap()
}

fn edited(progressive: bool) -> (Document, LayerId) {
    let mut doc = dense_doc();
    doc.set_progressive(progressive);
    let id = doc.insert_generator("g", Placement::Top, dents()).unwrap();
    doc.set_generator(id, GeneratorSpec { count: 120, seed: 4, size: 0.1, ..dents() }).unwrap();
    (doc, id)
}

#[test]
fn progressive_edits_defer_then_converge_to_the_synchronous_result() {
    let (reference, _) = edited(false);
    let (mut doc, _) = edited(true);
    assert!(doc.pending_leaves() > 0, "a progressive edit leaves leaves waiting");
    assert!(!same(doc.positions(), reference.positions(), 1e-6), "the surface has not caught up yet");

    // Even a zero budget makes progress (at least one chunk per call), so a stream of tiny budgets finishes.
    let mut last = doc.pending_leaves();
    let mut calls = 0;
    while doc.pending_leaves() > 0 {
        let left = doc.progress_pending(std::time::Duration::ZERO, None, false);
        assert!(left < last, "every call retires some leaves ({last} -> {left})");
        last = left;
        calls += 1;
        assert!(calls < 1000);
    }
    assert!(calls > 1, "a zero budget spreads the work over several calls");
    assert!(same(doc.positions(), reference.positions(), 1e-6), "converged result equals the synchronous one");
}

#[test]
fn a_visible_only_pass_leaves_what_the_camera_cannot_see() {
    let (reference, _) = edited(false);
    let (mut doc, _) = edited(true);
    let before = doc.pending_leaves();
    let front = view_from(Vec3::new(0.0, 0.0, 3.0), 35.0, 1000.0);
    let left = doc.progress_pending(std::time::Duration::from_secs(5), Some(&front), true);
    assert!(left > 0 && left < before, "the leaves facing the camera were done and the rest wait ({before} -> {left})");
    // Later, with nothing hurrying, the same call without `visible_only` finishes the rest.
    let left = doc.progress_pending(std::time::Duration::from_secs(5), Some(&front), false);
    assert_eq!(left, 0);
    assert!(same(doc.positions(), reference.positions(), 1e-6));
}

#[test]
fn anything_that_needs_the_exact_surface_finishes_pending_work_first() {
    let (reference, _) = edited(false);
    let (mut doc, _) = edited(true);
    assert!(doc.pending_leaves() > 0);
    doc.begin_stroke("stroke");
    assert_eq!(doc.pending_leaves(), 0, "begin_stroke finishes pending work");
    doc.end_stroke();
    assert!(same(doc.positions(), reference.positions(), 1e-6));

    let (mut doc, id) = edited(true);
    doc.set_layer_visible(id, false).unwrap();
    doc.set_progressive(false);
    assert_eq!(doc.pending_leaves(), 0, "turning progressive off finishes pending work");
    assert!(same(doc.positions(), doc.base_positions(), 1e-6), "a hidden generator leaves the base surface");
}

#[test]
fn undo_while_work_is_pending_still_ends_at_the_right_surface() {
    let (mut doc, id) = edited(true);
    doc.finish_pending();
    let after_first = doc.positions().to_vec();
    doc.set_generator(id, GeneratorSpec { count: 300, seed: 12, size: 0.07, ..dents() }).unwrap();
    assert!(doc.pending_leaves() > 0);
    assert!(doc.undo());
    doc.finish_pending();
    assert!(same(doc.positions(), &after_first, 1e-6), "undo restores the previous surface once caught up");
}

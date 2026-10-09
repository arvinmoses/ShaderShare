//! Layer hierarchy: folders, ordering, grouping, duplicate, merge down, solo, undo and persistence.

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::geom::Ray;
use sculpt_core::io::project;
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, LayerId, Placement};

fn doc() -> Document {
    Document::from_mesh(quad_sphere(3, 1.0)).unwrap()
}

/// Sculpt a bump into the active layer.
fn bump(doc: &mut Document) {
    let s = BrushSettings { radius: 0.4, strength: 1.0, ..Default::default() };
    let path: Vec<(Vec3, f32)> = (0..=10).map(|i| (Vec3::new(-0.2 + 0.04 * i as f32, 1.0, 0.0), 1.0)).collect();
    brush::stroke(doc, &mut ClayBuildup::default(), &s, &path).unwrap();
}

fn names(doc: &Document) -> Vec<String> {
    doc.layer_tree().iter().map(|r| format!("{}{}", "-".repeat(r.depth), doc.layer(r.id).unwrap().name)).collect()
}

fn max_diff(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (*p - *q).length()).fold(0.0, f32::max)
}

fn top_y(doc: &Document) -> f32 {
    doc.raycast(&Ray::new(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y)).unwrap().point.y
}

#[test]
fn insert_move_and_tree_order_with_undo() {
    let mut d = doc();
    let a = d.insert_layer("A", Placement::Top).unwrap();
    let b = d.insert_layer("B", Placement::Top).unwrap();
    let f = d.insert_folder("F", Placement::Above(a)).unwrap();
    assert_eq!(names(&d), ["B", "F", "A"]);
    d.move_layer(b, Placement::Into(f)).unwrap();
    assert_eq!(names(&d), ["F", "-B", "A"]);
    d.move_layer(a, Placement::Above(b)).unwrap();
    assert_eq!(names(&d), ["F", "-A", "-B"]);
    d.move_layer(a, Placement::Below(f)).unwrap();
    assert_eq!(names(&d), ["F", "-B", "A"]);

    for expect in [vec!["F", "-A", "-B"], vec!["F", "-B", "A"], vec!["B", "F", "A"]] {
        assert!(d.undo());
        assert_eq!(names(&d), expect);
    }
    assert!(d.redo());
    assert_eq!(names(&d), ["F", "-B", "A"]);
}

#[test]
fn cannot_move_folder_into_itself_or_non_folder() {
    let mut d = doc();
    let outer = d.insert_folder("Outer", Placement::Top).unwrap();
    let inner = d.insert_folder("Inner", Placement::Into(outer)).unwrap();
    assert!(d.move_layer(outer, Placement::Into(inner)).is_err());
    assert!(d.move_layer(outer, Placement::Above(inner)).is_err());
    let l = d.insert_layer("L", Placement::Top).unwrap();
    assert!(d.move_layer(outer, Placement::Into(l)).is_err());
    assert_eq!(names(&d), ["L", "Outer", "-Inner"]);
}

#[test]
fn folder_strength_visibility_and_solo_scale_the_surface() {
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    let l = d.insert_layer("L", Placement::Into(f)).unwrap();
    let flat = top_y(&d);
    bump(&mut d);
    let full = top_y(&d);
    assert!(full > flat + 0.01);

    d.set_layer_opacity(f, 0.5).unwrap();
    let half = top_y(&d);
    assert!(half > flat && half < full, "folder strength scales children: {flat} {half} {full}");
    d.set_layer_opacity(f, 1.0).unwrap();
    assert!((top_y(&d) - full).abs() < 1e-5);

    d.set_layer_visible(f, false).unwrap();
    assert!((top_y(&d) - flat).abs() < 1e-5, "hidden folder hides children");
    d.set_layer_visible(f, true).unwrap();

    // Solo another layer: this one is excluded, and sculpting on it is refused.
    let other = d.insert_layer("Other", Placement::Top).unwrap();
    d.set_solo(Some(other)).unwrap();
    assert!((top_y(&d) - flat).abs() < 1e-5);
    d.set_active_layer(Some(l)).unwrap();
    assert!(brush::stroke(&mut d, &mut ClayBuildup::default(), &BrushSettings::default(), &[(Vec3::Y, 1.0)]).is_err());
    d.set_solo(Some(f)).unwrap();
    assert!((top_y(&d) - full).abs() < 1e-5, "soloing a folder shows its subtree");
    d.set_solo(None).unwrap();
    assert!((top_y(&d) - full).abs() < 1e-5);
}

#[test]
fn folders_cannot_be_sculpted_on() {
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    assert!(d.set_active_layer(Some(f)).is_err());
}

#[test]
fn duplicate_doubles_the_effect_and_undoes() {
    let mut d = doc();
    let l = d.insert_layer("Wrinkles", Placement::Top).unwrap();
    let flat = top_y(&d);
    bump(&mut d);
    let one = top_y(&d);
    let before = d.positions().to_vec();
    let copy = d.duplicate_layer(l).unwrap();
    assert_eq!(d.layer(copy).unwrap().name, "Wrinkles copy");
    assert_eq!(names(&d), ["Wrinkles copy", "Wrinkles"]);
    assert!((top_y(&d) - flat) > (one - flat) * 1.9);
    assert!(d.undo());
    assert_eq!(names(&d), ["Wrinkles"]);
    assert!(max_diff(&before, d.positions()) < 1e-6);
    assert!(d.redo());
    assert_eq!(names(&d), ["Wrinkles copy", "Wrinkles"]);
    // A second copy gets a unique name.
    let again = d.duplicate_layer(l).unwrap();
    assert_eq!(d.layer(again).unwrap().name, "Wrinkles copy 2");
}

#[test]
fn duplicate_folder_copies_subtree() {
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    d.insert_layer("A", Placement::Into(f)).unwrap();
    let inner = d.insert_folder("In", Placement::Into(f)).unwrap();
    d.insert_layer("B", Placement::Into(inner)).unwrap();
    d.duplicate_layer(f).unwrap();
    assert_eq!(names(&d), ["F copy", "-In", "--B", "-A", "F", "-In", "--B", "-A"]);
    let ids: Vec<LayerId> = d.layers().iter().map(|l| l.id).collect();
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(ids.len(), unique.len());
}

#[test]
fn merge_down_keeps_the_surface_and_undoes() {
    let mut d = doc();
    let low = d.insert_layer("Low", Placement::Top).unwrap();
    bump(&mut d);
    d.set_layer_opacity(low, 0.6).unwrap();
    let high = d.insert_layer("High", Placement::Top).unwrap();
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    brush::stroke(&mut d, &mut ClayBuildup::default(), &s, &[(Vec3::new(0.1, 1.0, 0.0), 1.0), (Vec3::new(0.2, 1.0, 0.0), 1.0)]).unwrap();
    d.set_layer_opacity(high, 0.5).unwrap();
    let before = d.positions().to_vec();

    let into = d.merge_down(high).unwrap();
    assert_eq!(into, low);
    assert_eq!(names(&d), ["Low"]);
    assert_eq!(d.layer(low).unwrap().opacity, 1.0);
    assert!(max_diff(&before, d.positions()) < 1e-4, "merge keeps the composite");
    assert_eq!(d.active_layer(), Some(low));

    assert!(d.undo());
    assert_eq!(names(&d), ["High", "Low"]);
    assert_eq!(d.layer(low).unwrap().opacity, 0.6);
    assert_eq!(d.layer(high).unwrap().opacity, 0.5);
    assert!(max_diff(&before, d.positions()) < 1e-4);
    assert!(d.redo());
    assert_eq!(names(&d), ["Low"]);
    assert!(max_diff(&before, d.positions()) < 1e-4);
}

#[test]
fn merge_down_refuses_bad_targets() {
    let mut d = doc();
    let a = d.insert_layer("A", Placement::Top).unwrap();
    assert!(d.merge_down(a).is_err(), "nothing below");
    let b = d.insert_layer("B", Placement::Top).unwrap();
    d.set_layer_locked(a, true).unwrap();
    assert!(d.merge_down(b).is_err(), "locked target");
    d.set_layer_locked(a, false).unwrap();
    d.set_layer_visible(b, false).unwrap();
    assert!(d.merge_down(b).is_err(), "hidden source");
}

#[test]
fn group_and_ungroup_round_trip() {
    let mut d = doc();
    let a = d.insert_layer("A", Placement::Top).unwrap();
    let b = d.insert_layer("B", Placement::Top).unwrap();
    let c = d.insert_layer("C", Placement::Top).unwrap();
    let g = d.group_layers(&[a, c], "G").unwrap();
    assert_eq!(names(&d), ["G", "-C", "-A", "B"]);
    assert_eq!(d.layer(g).unwrap().parent, None);
    assert!(d.group_layers(&[a, b], "bad").is_err(), "different parents");
    d.ungroup(g).unwrap();
    assert_eq!(names(&d), ["C", "A", "B"]);
    assert!(d.undo());
    assert_eq!(names(&d), ["G", "-C", "-A", "B"]);
    assert!(d.undo());
    assert_eq!(names(&d), ["C", "B", "A"]);
    assert!(d.layer(g).is_none());
    let _ = b;
}

#[test]
fn delete_restores_data_with_undo() {
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    d.insert_layer("A", Placement::Into(f)).unwrap();
    bump(&mut d);
    let with = d.positions().to_vec();
    let flat = top_y(&d);
    d.delete_layer(f).unwrap();
    assert!(d.layers().is_empty());
    assert!(top_y(&d) < flat - 0.01);
    assert!(d.undo());
    assert_eq!(names(&d), ["F", "-A"]);
    assert!(max_diff(&with, d.positions()) < 1e-6);
    assert!(d.redo());
    assert!(d.layers().is_empty());
}

#[test]
fn meta_edits_undo_and_slider_drags_coalesce() {
    let mut d = doc();
    let l = d.insert_layer("A", Placement::Top).unwrap();
    bump(&mut d);
    let full = top_y(&d);
    let mut m = d.layer(l).unwrap().meta();
    for step in 1..=5 {
        m.opacity = 1.0 - 0.1 * step as f32;
        d.set_layer_meta(l, m.clone(), true).unwrap();
    }
    assert!(top_y(&d) < full);
    // One undo reverts the whole drag.
    assert!(d.undo());
    assert_eq!(d.layer(l).unwrap().opacity, 1.0);
    assert!((top_y(&d) - full).abs() < 1e-5);
    assert!(d.redo());
    assert!((d.layer(l).unwrap().opacity - 0.5).abs() < 1e-6);
    // Rename is a separate step.
    m.name = "Renamed".into();
    d.set_layer_meta(l, m, false).unwrap();
    assert!(d.undo());
    assert_eq!(d.layer(l).unwrap().name, "A");
}

#[test]
fn folders_round_trip_and_old_projects_load() {
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    d.insert_layer("A", Placement::Into(f)).unwrap();
    bump(&mut d);
    let mut m = d.layer(f).unwrap().meta();
    m.opacity = 0.5;
    m.collapsed = true;
    d.set_layer_meta(f, m, false).unwrap();
    let dir = std::env::temp_dir().join(format!("sculpt-tree-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    project::save(&d, &dir).unwrap();
    let back = project::load(&dir).unwrap();
    assert_eq!(names(&back), ["F", "-A"]);
    assert!(back.layer(f).unwrap().collapsed);
    assert!(max_diff(d.positions(), back.positions()) < 1e-5);
    std::fs::remove_dir_all(&dir).unwrap();

    // A version 1 project (flat layers, no folder fields) still loads.
    let old = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_project");
    let v1 = project::load(&old).unwrap();
    assert_eq!(names(&v1), ["Second", "Old layer"]);
    assert!(v1.layers().iter().all(|l| l.parent.is_none() && !l.is_folder()));
    assert_eq!(v1.layers()[1].opacity, 1.0);
    assert_eq!(v1.layers()[0].opacity, 0.5);
    assert!(top_y(&v1) > 1.0, "old layer deltas still shape the surface");
}

#[test]
fn corrupt_parent_links_are_rejected() {
    let old = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_project");
    let dir = std::env::temp_dir().join(format!("sculpt-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    fn copy(a: &std::path::Path, b: &std::path::Path) {
        std::fs::create_dir_all(b).unwrap();
        for e in std::fs::read_dir(a).unwrap() {
            let e = e.unwrap();
            if e.path().is_dir() {
                copy(&e.path(), &b.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), b.join(e.file_name())).unwrap();
            }
        }
    }
    copy(&old, &dir);
    let manifest = dir.join("project.json");
    let text = std::fs::read_to_string(&manifest).unwrap().replacen("\"locked\": false,", "\"locked\": false,\n      \"parent\": 99,", 1);
    std::fs::write(&manifest, text).unwrap();
    assert!(project::load(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---------------------------------------------------------------- blend modes

fn set_blend(d: &mut Document, id: LayerId, blend: sculpt_core::LayerBlend) {
    let mut m = d.layer(id).unwrap().meta();
    m.blend = blend;
    d.set_layer_meta(id, m, false).unwrap();
}

#[test]
fn blend_modes_shape_the_surface_as_documented() {
    use sculpt_core::LayerBlend::*;
    let mut d = doc();
    let l = d.insert_layer("Bump", Placement::Top).unwrap();
    let flat = top_y(&d);
    bump(&mut d); // raises the top
    let raised = top_y(&d);
    assert!(raised > flat + 0.01);

    set_blend(&mut d, l, Subtract);
    assert!(top_y(&d) < flat - 0.01, "subtract turns the bump into a dent");
    set_blend(&mut d, l, Min);
    assert!((top_y(&d) - flat).abs() < 1e-5, "min ignores the raising layer");
    set_blend(&mut d, l, Max);
    assert!((top_y(&d) - raised).abs() < 1e-5, "max keeps the raising layer");
    set_blend(&mut d, l, Normal);
    assert!((top_y(&d) - raised).abs() < 1e-4, "normal over a bare base looks like add");
    set_blend(&mut d, l, Add);
    assert!((top_y(&d) - raised).abs() < 1e-5);

    // Undo walks back through every mode change.
    assert!(d.undo());
    assert!((top_y(&d) - raised).abs() < 1e-4, "back to Normal");
    assert!(d.undo());
    assert!((top_y(&d) - raised).abs() < 1e-5, "back to Max");
    assert!(d.undo());
    assert!((top_y(&d) - flat).abs() < 1e-5, "back to Min");
}

#[test]
fn normal_replaces_what_is_below_inside_its_footprint() {
    use sculpt_core::LayerBlend::*;
    let mut d = doc();
    let low = d.insert_layer("Low", Placement::Top).unwrap();
    bump(&mut d);
    let low_only = top_y(&d);
    let high = d.insert_layer("High", Placement::Top).unwrap();
    // A second, smaller bump over the first.
    let s = BrushSettings { radius: 0.15, strength: 1.0, ..Default::default() };
    brush::stroke(&mut d, &mut ClayBuildup::default(), &s, &[(Vec3::new(0.0, 1.0, 0.0), 1.0), (Vec3::new(0.02, 1.0, 0.0), 1.0)]).unwrap();
    let stacked = top_y(&d);
    assert!(stacked > low_only, "add stacks the second bump on the first");
    set_blend(&mut d, high, Normal);
    let replaced = top_y(&d);
    assert!(replaced < stacked, "normal drops the lower layer's contribution under its footprint");
    let _ = low;
}

#[test]
fn old_projects_load_in_add_mode_and_new_modes_round_trip() {
    use sculpt_core::LayerBlend::*;
    let old = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_project");
    let v1 = project::load(&old).unwrap();
    assert!(v1.layers().iter().all(|l| l.blend == Add));

    let mut d = doc();
    let l = d.insert_layer("A", Placement::Top).unwrap();
    bump(&mut d);
    set_blend(&mut d, l, Max);
    let dir = std::env::temp_dir().join(format!("sculpt-blend-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    project::save(&d, &dir).unwrap();
    let back = project::load(&dir).unwrap();
    assert_eq!(back.layer(l).unwrap().blend, Max);
    assert!(max_diff(d.positions(), back.positions()) < 1e-5);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn stroking_on_a_subtract_layer_shows_immediately_and_flatten_needs_add() {
    use sculpt_core::LayerBlend::*;
    let mut d = doc();
    let l = d.insert_layer("Dig", Placement::Top).unwrap();
    set_blend(&mut d, l, Subtract);
    let flat = top_y(&d);
    bump(&mut d); // positive delta, subtracted: a dent appears right away
    assert!(top_y(&d) < flat - 0.01);
    assert!(d.flatten_layer(l).is_err());
    let m = d.insert_layer("Other", Placement::Top).unwrap();
    assert!(d.merge_down(m).is_err(), "merge needs both layers in Add mode");
}

#[test]
fn blend_order_follows_the_folder_tree() {
    use sculpt_core::LayerBlend::*;
    let mut d = doc();
    let f = d.insert_folder("F", Placement::Top).unwrap();
    let inside = d.insert_layer("Inside", Placement::Into(f)).unwrap();
    bump(&mut d);
    let outside = d.insert_layer("Outside", Placement::Top).unwrap();
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    brush::stroke(&mut d, &mut ClayBuildup::default(), &s, &[(Vec3::new(0.0, 1.0, 0.0), 1.0), (Vec3::new(0.05, 1.0, 0.0), 1.0)]).unwrap();
    set_blend(&mut d, outside, Normal);
    let over = top_y(&d);
    // Moving the Normal layer under the folder lets the folder's layer win instead.
    d.move_layer(outside, Placement::Below(f)).unwrap();
    let under = top_y(&d);
    assert!((over - under).abs() > 1e-3, "order matters once a layer replaces: {over} vs {under}");
    let _ = inside;
}

#[test]
fn disabled_mask_applies_everywhere_and_undoes() {
    use sculpt_core::mask::MaskStack;
    let mut d = doc();
    let l = d.insert_layer("A", Placement::Top).unwrap();
    bump(&mut d);
    let full = top_y(&d);
    let flat = {
        let mut m = d.layer(l).unwrap().meta();
        m.mask = Some(MaskStack::new(0.0)); // black: hides the layer
        d.set_layer_meta(l, m, false).unwrap();
        top_y(&d)
    };
    assert!(flat < full - 0.01, "black mask hides the bump");
    let mut m = d.layer(l).unwrap().meta();
    m.mask.as_mut().unwrap().enabled = false;
    d.set_layer_meta(l, m, false).unwrap();
    assert!((top_y(&d) - full).abs() < 1e-5, "disabled mask is ignored");
    assert!(d.undo());
    assert!((top_y(&d) - flat).abs() < 1e-5, "undo re-enables the mask");
    assert!(d.undo());
    assert!((top_y(&d) - full).abs() < 1e-5, "undo removes the mask");
}

#[test]
fn previews_show_the_footprint_and_track_edits() {
    let mut d = doc();
    let l = d.insert_layer("Bump", Placement::Top).unwrap();
    let empty = d.delta_preview(l, 16).unwrap();
    assert!(empty.covered.iter().all(|c| !c), "an untouched layer has no footprint");
    let before = d.edit_serial();
    bump(&mut d);
    assert!(d.edit_serial() > before, "a stroke bumps the edit counter");
    let p = d.delta_preview(l, 16).unwrap();
    assert!(p.covered.iter().any(|c| *c) && p.covered.iter().any(|c| !c), "covers part of the grid");
    assert!(p.values.iter().cloned().fold(0.0, f32::max) > 0.9, "peak is normalised to 1");
    // Looking down -Z at a unit sphere, the +Y bump is in the top rows.
    let top: f32 = p.values[..16 * 8].iter().sum();
    let bottom: f32 = p.values[16 * 8..].iter().sum();
    assert!(top > bottom, "bump sits in the upper half of the preview");
    let f = d.insert_folder("F", Placement::Top).unwrap();
    assert!(d.delta_preview(f, 16).is_none());

    assert!(d.mask_preview(l, 16).is_none(), "no mask, no mask preview");
    let mut m = d.layer(l).unwrap().meta();
    m.mask = Some(sculpt_core::mask::MaskStack::new(0.25));
    d.set_layer_meta(l, m, false).unwrap();
    let mp = d.mask_preview(l, 16).unwrap();
    let covered: Vec<f32> = mp.values.iter().zip(&mp.covered).filter(|(_, c)| **c).map(|(v, _)| *v).collect();
    assert!(!covered.is_empty() && covered.iter().all(|v| (v - 0.25).abs() < 1e-5));
}

#[test]
fn batch_edit_is_one_undo_step() {
    let mut d = doc();
    let a = d.insert_layer("A", Placement::Top).unwrap();
    let b = d.insert_layer("B", Placement::Top).unwrap();
    let edits: Vec<_> = [a, b]
        .iter()
        .map(|id| {
            let mut m = d.layer(*id).unwrap().meta();
            m.opacity = 0.5;
            m.locked = true;
            (*id, m)
        })
        .collect();
    d.set_layers_meta(edits).unwrap();
    assert!(d.layer(a).unwrap().locked && d.layer(b).unwrap().locked);
    assert!(d.undo());
    assert!(!d.layer(a).unwrap().locked && !d.layer(b).unwrap().locked, "one undo reverts both");
    assert_eq!(d.layer(a).unwrap().opacity, 1.0);
    assert!(d.redo());
    assert_eq!(d.layer(b).unwrap().opacity, 0.5);
}

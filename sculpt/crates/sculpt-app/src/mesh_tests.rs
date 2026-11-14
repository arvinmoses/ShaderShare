//! Mesh layers in the volume playground, headless: refits that never miss a change, one history across volume steps,
//! layer strokes, masks and settings (bit for bit by vertex name), save and open, and (ignored, reporting numbers) the
//! times and memory at 2.5M triangles.
//!
//! `cargo test --release -p sculpt-app mesh_tests -- --nocapture` (and `-- --ignored` for the timing gates).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::Vec3;
use sculpt_core::generator::sculpt_layer::{LayerBlend as RowBlend, MaskKind, NoiseMask};
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers};

use super::*;

/// One frame as the app runs it: a frame's share of the dabs, the tick, the GPU taking the volume's changes, the rows.
fn frame(mode: &mut VolumeMode) -> RowsDirty {
    mode.run_frame_dabs();
    mode.tick();
    mode.vol.drain_dirty();
    mode.take_rows_dirty()
}

/// Frames until nothing is left to land (strokes' ends, refits, swaps).
fn settle_frames(mode: &mut VolumeMode) -> usize {
    let mut n = 0;
    loop {
        frame(mode);
        n += 1;
        if !mode.busy() && !mode.dabs_queued() {
            return n;
        }
        assert!(n < 20_000, "never settled");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// These tests one at a time: each loads every core, and side by side they slow the other tests' timing checks.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// A ball with a first mesh layer on it.
fn ball_with_layer(triangles: f32) -> VolumeMode {
    let mut mode = VolumeMode::ball(triangles);
    mode.add_mesh_layer();
    assert!(mode.mesh_building());
    settle_frames(&mut mode);
    assert!(mode.mesh.is_some(), "{:?}", mode.mesh_note);
    mode
}

/// A stroke of 16 dabs about direction `dir` on what `target` names, frames going on every fourth dab. `end`: the pen
/// lifted (else the stroke is left open).
fn stroke(mode: &mut VolumeMode, target: Target, dir: Vec3, r: f32, brush: Brush) -> bool {
    mode.target = target;
    mode.begin_stroke();
    let ahead = mode.ahead.is_some();
    let side = dir.any_orthonormal_vector();
    for k in 0..16 {
        let d = (dir.normalize() + side * (0.15 * (k as f32 / 15.0 - 0.5))).normalize();
        if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
            mode.queue_dab(hit, r, 0.002, 0.8, 0.3, brush);
        }
        if k % 4 == 3 {
            frame(mode);
        }
    }
    mode.end_stroke();
    ahead
}

fn ml(mode: &mut VolumeMode) -> &mut MeshLayers {
    &mut mode.mesh.as_mut().unwrap().layers
}

/// Every layer composited afresh (strokes add up dab by dab, so to rounding; a restore composites afresh).
fn fresh(ml: &mut MeshLayers) {
    let ids: Vec<(LayerId, f32)> = ml.doc.layers().iter().map(|l| (l.id, l.opacity)).collect();
    for (id, o) in ids {
        ml.doc.set_layer_opacity(id, o * 0.5 + 0.25).unwrap();
        ml.doc.set_layer_opacity(id, o).unwrap();
    }
}

use sculpt_core::LayerId;

/// Per vertex name, everything the layers hold or show there: base, each layer's offset and mask value, each
/// channel, freeze, and the composite (afresh).
fn state(ml: &mut MeshLayers) -> HashMap<u64, Vec<u32>> {
    fresh(ml);
    let doc = &ml.doc;
    let to_c = ml.internal_to_canonical();
    let deltas: Vec<Vec<Vec3>> = doc.layers().iter().map(|l| ml.layer_deltas(l.id).unwrap()).collect();
    let b = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    (0..doc.vertex_count())
        .map(|v| {
            let mut out: Vec<u32> = b(doc.base_positions()[v]).to_vec();
            for (l, d) in doc.layers().iter().zip(&deltas) {
                out.extend(b(d[v]));
                out.push(l.mask_values().map_or(u32::MAX, |m| m[v].to_bits()));
                out.push(l.opacity.to_bits());
                out.push(l.visible as u32);
            }
            for c in doc.channels().values() {
                out.push(c[v].to_bits());
            }
            out.push(doc.freeze()[v].to_bits());
            out.extend(b(doc.positions()[v]));
            (ml.keys()[to_c[v] as usize], out)
        })
        .collect()
}

/// (names only in one, names in both whose values differ).
fn differ(a: &HashMap<u64, Vec<u32>>, b: &HashMap<u64, Vec<u32>>) -> (usize, usize) {
    let only = a.keys().filter(|k| !b.contains_key(k)).count() + b.keys().filter(|k| !a.contains_key(k)).count();
    let diff = a.iter().filter(|(k, x)| b.get(k).is_some_and(|y| y != *x)).count();
    (only, diff)
}

/// The mesh the layers sit on is the volume's surface now, bit for bit (as a fresh extraction gives it).
fn base_is_fresh(mode: &mut VolumeMode, what: &str) {
    let fresh = MeshLayers::from_volume(&mode.vol, MeshLayerSettings::default()).unwrap();
    let ml = ml(mode);
    assert_eq!(ml.keys().len(), fresh.keys().len(), "{what}: vertex count");
    assert!(ml.keys() == fresh.keys(), "{what}: vertex names");
    let wrong = (0..ml.keys().len())
        .filter(|&c| {
            let (a, b) = (ml.doc.base_positions()[ml.doc.canonical_index(c)], fresh.doc.base_positions()[fresh.doc.canonical_index(c)]);
            a.to_array().map(f32::to_bits) != b.to_array().map(f32::to_bits)
        })
        .count();
    assert_eq!(wrong, 0, "{what}: {wrong} base positions differ from a fresh extraction");
    assert!(ml.blocks() == fresh.blocks(), "{what}: blocks");
    eprintln!("{what}: {} vertices, base bit-equal to a fresh extraction", ml.keys().len());
}

// --------------------------------------------------------------------------------------------------------- gates

/// After a scripted session (layer strokes, volume strokes, strokes run ahead of the last one's end, undo, redo),
/// the layers' mesh is the volume's surface bit for bit.
#[test]
fn refit_never_misses() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    stroke(&mut mode, Target::Form, Vec3::new(0.25, 0.1, 1.0), 0.12, Brush::Clay);
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after a volume stroke");
    // Strokes begun while the last one's end is still landing: run ahead on a copy, done for real later.
    stroke(&mut mode, Target::Form, Vec3::new(-0.5, 0.3, 0.8), 0.1, Brush::Trim);
    let ahead = stroke(&mut mode, Target::Form, Vec3::new(-0.4, 0.2, 0.9), 0.05, Brush::Clay);
    assert!(ahead, "the second stroke ran ahead");
    stroke(&mut mode, Target::Form, Vec3::new(-0.45, 0.25, 0.85), 0.2, Brush::Smooth);
    // A layer stroke right after: it waits for everything to land and refit.
    stroke(&mut mode, Target::Detail, Vec3::new(-0.45, 0.2, 0.9), 0.06, Brush::Clay);
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after strokes run ahead");
    assert!(mode.undo() && mode.undo());
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after two undos");
    assert!(mode.redo());
    stroke(&mut mode, Target::Form, Vec3::new(0.9, -0.2, 0.3), 0.15, Brush::Clay);
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after a redo and a stroke");
    for _ in 0..6 {
        mode.undo();
    }
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after undoing all");
    let m = mode.mesh.as_ref().unwrap();
    eprintln!("swaps {}, history {} redo {}", m.times.swaps, m.history.len(), m.redo.len());
}

/// A layer stroke, a volume stroke under it, undo twice, redo twice: every vertex's data by name bit for bit at
/// every step.
#[test]
fn layer_stroke_then_volume_stroke_under_it_undo_redo() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    let s0 = state(ml(&mut mode));
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    settle_frames(&mut mode);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).1 > 100, "the layer stroke changed little");
    stroke(&mut mode, Target::Form, Vec3::new(0.22, 0.1, 1.0), 0.1, Brush::Clay);
    settle_frames(&mut mode);
    let s2 = state(ml(&mut mode));
    let moved = differ(&s1, &s2);
    eprintln!("the volume stroke: {moved:?} (names only in one, differing)");
    assert!(moved.0 > 100, "the volume stroke renamed few vertices");
    let mut check = |mode: &mut VolumeMode, want: &HashMap<u64, Vec<u32>>, what: &str| {
        settle_frames(mode);
        let d = differ(&state(ml(mode)), want);
        eprintln!("{what}: {d:?}");
        assert_eq!(d, (0, 0), "{what}");
    };
    assert!(mode.undo());
    check(&mut mode, &s1, "undo the volume stroke");
    assert!(mode.undo());
    check(&mut mode, &s0, "undo the layer stroke");
    assert!(mode.redo());
    check(&mut mode, &s1, "redo the layer stroke");
    assert!(mode.redo());
    check(&mut mode, &s2, "redo the volume stroke");
    base_is_fresh(&mut mode, "at the end");
}

/// A layer stroke that changes nothing (every dab off the model) adds no step: not again the stroke before it, nor
/// an error after an opacity edit.
#[test]
fn a_stroke_that_misses_adds_no_step() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    let s0 = state(ml(&mut mode));
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    settle_frames(&mut mode);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).1 > 100, "the layer stroke changed little");
    let steps = mode.mesh.as_ref().unwrap().history.len();
    let miss = |mode: &mut VolumeMode| {
        mode.target = Target::Detail;
        mode.begin_stroke();
        mode.queue_dab(Vec3::new(50.0, 50.0, 50.0), 0.08, 0.002, 0.8, 0.3, Brush::Clay);
        frame(mode);
        mode.end_stroke();
        settle_frames(mode);
    };
    miss(&mut mode);
    assert_eq!(mode.mesh.as_ref().unwrap().history.len(), steps, "a stroke off the model made a step");
    mode.set_layer_opacity(0, 0.7);
    mode.close_sculpt_edit();
    let steps = mode.mesh.as_ref().unwrap().history.len();
    miss(&mut mode);
    assert_eq!(mode.mesh.as_ref().unwrap().history.len(), steps, "a stroke off the model after an edit made a step");
    assert!(mode.undo());
    settle_frames(&mut mode);
    assert!(mode.undo());
    settle_frames(&mut mode);
    assert_eq!(differ(&state(ml(&mut mode)), &s0), (0, 0), "two undos: the opacity, then the stroke");
}

/// Three volume strokes ending within one refit (one swap covering all three), over a layer: undo three times gives
/// the start back bit for bit, redo three times the end.
#[test]
fn three_volume_strokes_in_one_swap_undo_redo() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    stroke(&mut mode, Target::Detail, Vec3::new(0.0, 0.1, 1.0), 0.15, Brush::Clay);
    stroke(&mut mode, Target::Detail, Vec3::new(0.1, -0.1, 1.0), 0.1, Brush::Trim);
    settle_frames(&mut mode);
    let swaps = mode.mesh.as_ref().unwrap().times.swaps;
    let s0 = state(ml(&mut mode));
    let t = Instant::now();
    // One after another with no frame between: the later ones run ahead of the first's end.
    for (i, d) in [Vec3::new(0.05, 0.1, 1.0), Vec3::new(-0.05, 0.0, 1.0), Vec3::new(0.1, 0.0, 1.0)].into_iter().enumerate() {
        mode.target = Target::Form;
        mode.begin_stroke();
        let side = d.any_orthonormal_vector();
        for k in 0..12 {
            let dd = (d.normalize() + side * (0.1 * (k as f32 / 11.0 - 0.5))).normalize();
            let (hit, _) = mode.raycast(dd * 3.0, -dd).unwrap();
            mode.brush_dab(hit, 0.06 + 0.02 * i as f32, 0.002, 0.8, 0.3, if i == 1 { Brush::Trim } else { Brush::Clay });
        }
        mode.end_stroke();
    }
    eprintln!("three strokes ended in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    settle_frames(&mut mode);
    let m = mode.mesh.as_ref().unwrap();
    assert_eq!(m.times.swaps, swaps + 1, "one swap covers the three strokes");
    let s3 = state(ml(&mut mode));
    assert!(differ(&s0, &s3).0 > 100);
    for i in 0..3 {
        assert!(mode.undo(), "undo {i}");
    }
    settle_frames(&mut mode);
    let d = differ(&state(ml(&mut mode)), &s0);
    eprintln!("undo x3: {d:?}");
    assert_eq!(d, (0, 0), "undo x3");
    for i in 0..3 {
        assert!(mode.redo(), "redo {i}");
    }
    settle_frames(&mut mode);
    let d = differ(&state(ml(&mut mode)), &s3);
    eprintln!("redo x3: {d:?}");
    assert_eq!(d, (0, 0), "redo x3");
    base_is_fresh(&mut mode, "at the end");
}

/// A mask stroke and an opacity change undo and redo bit for bit; two layers with a paint and a noise mask save and
/// open bit for bit.
#[test]
fn masks_opacity_undo_redo_and_save_open() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    stroke(&mut mode, Target::Detail, Vec3::new(0.0, 0.1, 1.0), 0.15, Brush::Clay);
    mode.add_mesh_layer();
    assert_eq!(mode.sculpt_layers().unwrap().1.len(), 2);
    stroke(&mut mode, Target::Detail, Vec3::new(0.1, 0.0, 1.0), 0.12, Brush::Trim);
    // Layer 0: a painted mask, painted black along a stroke; a noise mask on layer 1.
    mode.set_active_layer(0);
    mode.add_mask(MaskKind::Paint { fill: 1.0 });
    assert!(matches!(mode.target, Target::Mask { show: false }), "{:?}", mode.target);
    settle_frames(&mut mode);
    let s0 = state(ml(&mut mode));
    stroke(&mut mode, Target::Mask { show: false }, Vec3::new(0.0, 0.1, 1.0), 0.06, Brush::Clay);
    settle_frames(&mut mode);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).1 > 50, "the mask stroke changed little");
    assert!(mode.undo());
    assert_eq!(differ(&state(ml(&mut mode)), &s0), (0, 0), "undo the mask stroke");
    assert!(mode.redo());
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "redo the mask stroke");
    // An opacity drag over frames, one step.
    for o in [0.9, 1.3, 1.7] {
        mode.set_layer_opacity(1, o);
        frame(&mut mode);
    }
    mode.close_sculpt_edit();
    let s2 = state(ml(&mut mode));
    assert!(differ(&s1, &s2).1 > 50);
    assert!(mode.undo());
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "undo the opacity drag");
    assert!(mode.redo());
    assert_eq!(differ(&state(ml(&mut mode)), &s2), (0, 0), "redo the opacity drag");
    mode.set_active_layer(1);
    mode.add_mask(MaskKind::Noise(NoiseMask::default()));
    mode.set_layer_blend(1, RowBlend::Raise);
    let info = mode.sculpt_layers().unwrap().1;
    assert_eq!(info[0].masks.len(), 1);
    assert_eq!(info[1].masks.len(), 1);
    assert!(matches!(info[1].masks[0].kind, MaskKind::Noise(_)));
    // A volume stroke too, so the file's mesh is not the first one.
    stroke(&mut mode, Target::Form, Vec3::new(-0.6, 0.3, 0.7), 0.1, Brush::Clay);
    settle_frames(&mut mode);
    let before = state(ml(&mut mode));
    base_is_fresh(&mut mode, "before saving");
    let mut bytes = Vec::new();
    mode.write_to(&mut bytes).unwrap();
    let mut back = VolumeMode::read_from(&mut std::io::Cursor::new(&bytes)).unwrap();
    assert!(back.mesh.is_some());
    base_is_fresh(&mut back, "opened");
    // The volume's own file form rounds its samples, so the surface moves by a hair on the first save: what the
    // layers hold (offsets, painted channels, settings) comes back bit for bit by name, and a second save and open
    // gives everything back bit for bit.
    let after = state(ml(&mut back));
    let only = |s: &HashMap<u64, Vec<u32>>, layers: usize| -> HashMap<u64, Vec<u32>> {
        s.iter().map(|(k, v)| {
            let mut out = Vec::new();
            for l in 0..layers {
                let at = 3 + 6 * l;
                out.extend_from_slice(&v[at..at + 3]);
                out.extend_from_slice(&v[at + 4..at + 6]);
            }
            out.extend_from_slice(&v[3 + 6 * layers..v.len() - 3]);
            (*k, out)
        }).collect()
    };
    let d = differ(&only(&before, 2), &only(&after, 2));
    eprintln!("save and open: {} bytes; layer data by name {d:?}; everything {:?}", bytes.len(), differ(&before, &after));
    assert_eq!(d, (0, 0), "save and open: layer data");
    let mut again = Vec::new();
    back.write_to(&mut again).unwrap();
    let mut back2 = VolumeMode::read_from(&mut std::io::Cursor::new(&again)).unwrap();
    let d2 = differ(&after, &state(ml(&mut back2)));
    eprintln!("second save and open: {d2:?}");
    assert_eq!(d2, (0, 0), "second save and open");
    // (Which mask is picked is not saved.)
    let rows = |m: &VolumeMode| m.sculpt_layers().unwrap().1.into_iter().map(|mut x| {
        x.active_mask = None;
        x
    }).collect::<Vec<_>>();
    assert_eq!(rows(&back), rows(&mode), "layer rows");
    // A file without mesh layers opens as before.
    let mut plain = Vec::new();
    VolumeMode::ball(2e5).write_to(&mut plain).unwrap();
    assert!(VolumeMode::read_from(&mut std::io::Cursor::new(&plain)).unwrap().mesh.is_none());
}

/// Carve the whole volume away as one step (the mesh it leaves has no faces), kept in the layers' history as a
/// stroke's end keeps it.
fn carve_all(mode: &mut VolumeMode) {
    mode.settle();
    ball_step(mode, Vec3::ZERO, 2.0, true);
}

/// A ball of radius `r` at `c` added (or carved) as one step, kept in the layers' history as a stroke's end keeps it.
/// (Straight on the volume: nothing settles first, so a refit under way is still under way.)
fn ball_step(mode: &mut VolumeMode, c: Vec3, r: f32, carve: bool) {
    let v = &mut mode.vol;
    assert_eq!(v.levels(), 1, "a ball over every block of a fine level takes for ever");
    v.begin_stroke();
    v.ball_dab(0, c, r, 0.05, carve);
    v.end_stroke();
    let id = v.last_id().unwrap();
    mode.mesh.as_mut().unwrap().note_step(id);
}

/// A one-level ball (so a carve over all of it is quick) with a first mesh layer and a layer stroke on it.
fn small_ball_with_stroke() -> VolumeMode {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 1.0 / 48.0, 1));
    mode.add_mesh_layer();
    settle_frames(&mut mode);
    assert!(mode.mesh.is_some(), "{:?}", mode.mesh_note);
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    settle_frames(&mut mode);
    mode
}

fn failed(mode: &VolumeMode) -> bool {
    let m = mode.mesh.as_ref().unwrap();
    assert_eq!(m.failure.is_some(), mode.mesh_failure().is_some());
    m.failure.is_some()
}

/// Undo or redo (`f`), frames until it lands, and whether its refit failed as it was meant to.
fn step(mode: &mut VolumeMode, f: fn(&mut VolumeMode) -> bool, fails: bool, what: &str) {
    assert!(f(mode), "{what}: nothing done");
    settle_frames(mode);
    assert_eq!(failed(mode), fails, "{what}: failed");
    assert!(mode.mesh_note.as_deref() != Some(crate::mesh_stack::DISAGREE), "{what}: the histories came apart");
}

fn sides(mode: &VolumeMode) -> (usize, usize) {
    let m = mode.mesh.as_ref().unwrap();
    (m.history.len(), m.redo.len())
}

/// Everything carved away: the layers cannot follow (no faces). Frames settle and no refit runs again, waiting for
/// the layers (finishing, saving) returns, the note says why; undo brings the layers back bit for bit, redo carves
/// again (its refit failing as it lands: the step is not lost), and undo after it brings them back again; the layer
/// stroke under it still undoes and redoes.
#[test]
fn layers_survive_the_volume_carved_away() {
    let _one = one_at_a_time();
    // One level, so the carve over the whole ball is quick.
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 1.0 / 48.0, 1));
    mode.add_mesh_layer();
    settle_frames(&mut mode);
    assert!(mode.mesh.is_some(), "{:?}", mode.mesh_note);
    let s0 = state(ml(&mut mode));
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    settle_frames(&mut mode);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).1 > 100, "the layer stroke changed little");
    carve_all(&mut mode);
    let carved = mode.vol.dirty_keys();
    settle_frames(&mut mode);
    assert!(failed(&mode), "the refit of nothing did not fail");
    let swaps = mode.mesh.as_ref().unwrap().times.swaps;
    for _ in 0..5 {
        frame(&mut mode);
        assert!(!ml(&mut mode).busy() && !mode.busy(), "a failed refit ran again");
    }
    // The carve's blocks seen again (the GPU not having taken them yet, as in the app): no refit.
    assert!(!carved.is_empty());
    let m = mode.mesh.as_mut().unwrap();
    m.tap(carved);
    assert!(!m.start_refit(&mode.vol) && !m.pending(), "the failed blocks seen again started a refit");
    // Layer strokes and fills wait, saying why; another note does not hide the failure.
    let steps = sides(&mode);
    stroke(&mut mode, Target::Detail, Vec3::new(-0.2, 0.1, 1.0), 0.08, Brush::Clay);
    mode.fill_mask(0.0);
    settle_frames(&mut mode);
    assert_eq!(sides(&mode), steps, "a layer stroke or fill made a step while the layers could not follow");
    assert!(mode.mesh_note.as_ref().is_some_and(|n| n.starts_with("No mask fill")), "{:?}", mode.mesh_note);
    assert!(failed(&mode));
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "a refused stroke changed the layers");
    mode.mesh.as_mut().unwrap().finish(&mode.vol);
    let mut bytes = Vec::new();
    mode.write_to(&mut bytes).unwrap();
    assert_eq!(mode.mesh.as_ref().unwrap().times.swaps, swaps);
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "the layers changed while they could not follow");
    let check = |mode: &mut VolumeMode, want: &HashMap<u64, Vec<u32>>, what: &str| {
        settle_frames(mode);
        let d = differ(&state(ml(mode)), want);
        eprintln!("{what}: {d:?}");
        assert_eq!(d, (0, 0), "{what}");
    };
    assert!(mode.undo());
    check(&mut mode, &s1, "undo the carve");
    assert!(!failed(&mode));
    base_is_fresh(&mut mode, "after undoing the carve");
    // The redo's own refit fails: the step goes back on the history all the same.
    let (h, r) = sides(&mode);
    assert!(mode.redo());
    settle_frames(&mut mode);
    assert!(failed(&mode));
    assert_eq!(sides(&mode), (h + 1, r - 1), "the carve's step was lost");
    assert!(mode.undo());
    check(&mut mode, &s1, "undo the carve again");
    assert!(!failed(&mode));
    assert!(mode.undo());
    check(&mut mode, &s0, "undo the layer stroke");
    assert!(mode.redo());
    check(&mut mode, &s1, "redo the layer stroke");
    assert!(mode.redo());
    settle_frames(&mut mode);
    assert!(failed(&mode));
    assert!(mode.undo());
    check(&mut mode, &s1, "undo the carve a third time");
    base_is_fresh(&mut mode, "at the end");
}

/// Through a state the layers cannot follow (everything carved away, V1) between two they can (V0, V2): carve (fails),
/// a new ball (V2), undo it (fails), undo the carve (V0), redo the carve (fails), redo the ball, undo it (fails), undo
/// the carve: V0's layer data comes back bit for bit both times, and V2's after the redo.
#[test]
fn a_failing_state_between_two_undo_redo() {
    let _one = one_at_a_time();
    let mut mode = small_ball_with_stroke();
    let s0 = state(ml(&mut mode));
    carve_all(&mut mode);
    settle_frames(&mut mode);
    assert!(failed(&mode));
    ball_step(&mut mode, Vec3::ZERO, 0.8, false);
    settle_frames(&mut mode);
    assert!(!failed(&mode), "the new ball did not refit");
    let s2 = state(ml(&mut mode));
    let check = |mode: &mut VolumeMode, want: &HashMap<u64, Vec<u32>>, what: &str| {
        let d = differ(&state(ml(mode)), want);
        eprintln!("{what}: {d:?}");
        assert_eq!(d, (0, 0), "{what}");
    };
    step(&mut mode, VolumeMode::undo, true, "undo the ball");
    step(&mut mode, VolumeMode::undo, false, "undo the carve");
    check(&mut mode, &s0, "V0 after undoing both");
    step(&mut mode, VolumeMode::redo, true, "redo the carve");
    step(&mut mode, VolumeMode::redo, false, "redo the ball");
    check(&mut mode, &s2, "V2 after redoing both");
    step(&mut mode, VolumeMode::undo, true, "undo the ball again");
    step(&mut mode, VolumeMode::undo, false, "undo the carve again");
    check(&mut mode, &s0, "V0 after undoing both again");
    base_is_fresh(&mut mode, "at the end");
}

/// Carve (fails), a new ball, undo it (fails), another ball in its place, undo it (fails), undo the carve: V0's
/// layer data comes back bit for bit.
#[test]
fn a_failing_state_then_a_new_step_from_it() {
    let _one = one_at_a_time();
    let mut mode = small_ball_with_stroke();
    let s0 = state(ml(&mut mode));
    carve_all(&mut mode);
    settle_frames(&mut mode);
    ball_step(&mut mode, Vec3::ZERO, 0.8, false);
    settle_frames(&mut mode);
    assert!(!failed(&mode));
    step(&mut mode, VolumeMode::undo, true, "undo the ball");
    ball_step(&mut mode, Vec3::new(0.1, 0.0, 0.0), 0.7, false);
    settle_frames(&mut mode);
    assert!(!failed(&mode), "the other ball did not refit");
    step(&mut mode, VolumeMode::undo, true, "undo the other ball");
    step(&mut mode, VolumeMode::undo, false, "undo the carve");
    let d = differ(&state(ml(&mut mode)), &s0);
    eprintln!("V0: {d:?}");
    assert_eq!(d, (0, 0), "V0's layer data");
    base_is_fresh(&mut mode, "at the end");
}

/// A file whose later chunks are damaged or cut short opens all the same, without them, saying so: the mesh layers
/// cut inside, their offset back damaged, their volume carved away (no mesh to put them on), a surface noise chunk
/// running past the end.
#[test]
fn damaged_files_open_without_what_is_damaged() {
    let _one = one_at_a_time();
    let mut mode = small_ball_with_stroke();
    let mut good = Vec::new();
    mode.write_to(&mut good).unwrap();
    let open = |bytes: &[u8], what: &str, gone: &str| -> VolumeMode {
        let m = VolumeMode::read_from(&mut std::io::Cursor::new(bytes)).unwrap_or_else(|e| panic!("{what}: refused: {e}"));
        let lossy = m.lossy_open.clone().unwrap_or_else(|| panic!("{what}: no note"));
        eprintln!("{what}: {lossy}");
        assert!(lossy.contains(gone), "{what}: {lossy}");
        assert_eq!(m.mesh_note.as_ref(), Some(&lossy), "{what}");
        assert!(!m.vol.keys().is_empty(), "{what}: the volume is gone");
        m
    };
    let back = VolumeMode::read_from(&mut std::io::Cursor::new(&good)).unwrap();
    assert!(back.mesh.is_some() && back.lossy_open.is_none(), "{:?}", back.lossy_open);
    // Cut inside the mesh layers.
    let m = open(&good[..good.len() - 40], "cut inside", "mesh layers");
    assert!(m.mesh.is_none());
    // The offset back damaged.
    let mut bad = good.clone();
    let n = bad.len();
    bad[n - 12..n - 4].copy_from_slice(&5u64.to_le_bytes());
    assert!(open(&bad, "offset back", "mesh layers").mesh.is_none());
    // A surface noise chunk after the volume, running past the end.
    let mut plain = Vec::new();
    VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 1.0 / 48.0, 1)).write_to(&mut plain).unwrap();
    plain.extend_from_slice(b"SNOI");
    plain.extend_from_slice(&1000u32.to_le_bytes());
    plain.extend_from_slice(b"{\"on\":true");
    open(&plain, "noise past the end", "surface noise");
    // Mesh layers on a volume with no surface left.
    carve_all(&mut mode);
    settle_frames(&mut mode);
    let mut empty = Vec::new();
    mode.write_to(&mut empty).unwrap();
    let m = VolumeMode::read_from(&mut std::io::Cursor::new(&empty)).unwrap();
    assert!(m.mesh.is_none() && m.lossy_open.as_ref().is_some_and(|l| l.contains("mesh layers")), "{:?}", m.lossy_open);
}

/// A redo whose refit has not landed when a new volume stroke ends: the redone step stays on the history under the
/// new one, so two undos give back the state before it, with no history lost.
#[test]
fn a_stroke_before_a_redo_lands_keeps_the_redone_step() {
    let _one = one_at_a_time();
    let mut mode = small_ball_with_stroke();
    let s0 = state(ml(&mut mode));
    ball_step(&mut mode, Vec3::new(0.0, 0.0, 1.0), 0.2, false);
    settle_frames(&mut mode);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).0 > 100);
    step(&mut mode, VolumeMode::undo, false, "undo the bump");
    assert_eq!(differ(&state(ml(&mut mode)), &s0), (0, 0), "undo the bump");
    assert!(mode.redo());
    assert!(mode.mesh.as_ref().unwrap().pending(), "the redo landed at once");
    // Straight on: no frame lands the redo first.
    ball_step(&mut mode, Vec3::new(0.0, 1.0, 0.0), 0.2, false);
    settle_frames(&mut mode);
    step(&mut mode, VolumeMode::undo, false, "undo the second bump");
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "after undoing the second bump");
    step(&mut mode, VolumeMode::undo, false, "undo the redone bump");
    assert_eq!(differ(&state(ml(&mut mode)), &s0), (0, 0), "after undoing the redone bump");
    // The layer stroke under them is still there to undo.
    assert!(matches!(mode.mesh.as_ref().unwrap().history.last(), Some(crate::mesh_stack::Entry::Layer { .. })));
}

/// A mask fill's undo puts back the fill the panel shows (and the mask's values), its redo the new one.
#[test]
fn undoing_a_fill_shows_the_fill_before() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    stroke(&mut mode, Target::Detail, Vec3::new(0.0, 0.1, 1.0), 0.15, Brush::Clay);
    mode.set_active_layer(0);
    mode.add_mask(MaskKind::Paint { fill: 1.0 });
    settle_frames(&mut mode);
    let fill = |mode: &VolumeMode| match mode.sculpt_layers().unwrap().1[0].masks[0].kind {
        MaskKind::Paint { fill } => fill,
        _ => panic!("not a painted mask"),
    };
    assert_eq!(fill(&mode), 1.0);
    let s0 = state(ml(&mut mode));
    mode.fill_mask(0.0);
    let s1 = state(ml(&mut mode));
    assert!(differ(&s0, &s1).1 > 100, "the fill changed little");
    assert_eq!(fill(&mode), 0.0);
    assert!(mode.undo());
    assert_eq!(fill(&mode), 1.0, "the panel's fill after undo");
    assert_eq!(differ(&state(ml(&mut mode)), &s0), (0, 0), "undo the fill");
    assert!(mode.redo());
    assert_eq!(fill(&mode), 0.0, "the panel's fill after redo");
    assert_eq!(differ(&state(ml(&mut mode)), &s1), (0, 0), "redo the fill");
}

/// The rows told of every change: a dab's leaves, a swap's blocks, a hidden layer's.
#[test]
fn rows_follow_dabs_swaps_and_hiding() {
    let _one = one_at_a_time();
    let mut mode = ball_with_layer(2e5);
    let r = frame(&mut mode);
    assert!(r.all || r.blocks.is_empty());
    let r = frame(&mut mode);
    assert!(r.is_empty(), "{r:?}");
    mode.target = Target::Detail;
    mode.begin_stroke();
    let d = Vec3::new(0.1, 0.0, 1.0).normalize();
    let (hit, _) = mode.raycast(d * 3.0, -d).unwrap();
    mode.queue_dab(hit, 0.05, 0.002, 0.8, 0.3, Brush::Clay);
    let r = frame(&mut mode);
    assert!(!r.all && !r.leaves.is_empty(), "a dab lists its leaves: {r:?}");
    mode.end_stroke();
    mode.target = Target::Form;
    stroke(&mut mode, Target::Form, Vec3::new(-0.6, 0.3, 0.7), 0.1, Brush::Clay);
    let mut swapped = RowsDirty::default();
    loop {
        let r = frame(&mut mode);
        swapped.blocks.extend(r.blocks);
        swapped.removed.extend(r.removed);
        if !mode.busy() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!swapped.blocks.is_empty(), "a swap lists its blocks");
    mode.set_layer_visible(0, false);
    assert!(frame(&mut mode).none);
    assert!(!mode.layers_shown());
    mode.set_layer_visible(0, true);
    assert!(frame(&mut mode).all);
}

// --------------------------------------------------------------------------------------- brushes on a layer

/// A stroke as the app gives it: dabs every 0.15 r along the arc from `from` to `to`, `passes` times back and forth in
/// one stroke, each dab where the pen's ray meets what `raycast` gives.
#[allow(clippy::too_many_arguments)]
fn app_stroke(mode: &mut VolumeMode, target: Target, from: Vec3, to: Vec3, r: f32, strength: f32, brush: Brush, passes: usize) {
    mode.target = target;
    mode.begin_stroke();
    let (a, b) = (from.normalize(), to.normalize());
    let n = (a.angle_between(b) / (0.15 * r)).ceil().max(1.0) as usize;
    for p in 0..passes {
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let t = if p % 2 == 0 { t } else { 1.0 - t };
            let d = a.lerp(b, t).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.queue_dab(hit, r, 0.004, strength, 0.3, brush);
            }
            if k % 3 == 2 {
                frame(mode);
            }
        }
    }
    mode.end_stroke();
    settle_frames(mode);
}

/// Radial height of what is shown (the layers' composite with `layer`, else the volume) along `d`.
fn height(mode: &mut VolumeMode, layer: bool, d: Vec3) -> f32 {
    mode.target = if layer { Target::Detail } else { Target::Form };
    mode.raycast(d * 3.0, -d).map_or(f32::NAN, |(p, _)| p.length())
}

/// Heights across the middle of the stroke from `from` to `to`, 65 samples over +-1.6 r.
fn across(mode: &mut VolumeMode, layer: bool, from: Vec3, to: Vec3, r: f32) -> Vec<f32> {
    let (a, b) = (from.normalize(), to.normalize());
    let mid = a.lerp(b, 0.5).normalize();
    let side = (b - a).cross(mid).normalize();
    (0..65).map(|i| height(mode, layer, (mid + side * (1.6 * r * (i as f32 / 32.0 - 1.0))).normalize())).collect()
}

/// The change from `before` to `after` furthest from zero.
fn extreme(before: &[f32], after: &[f32]) -> f32 {
    after.iter().zip(before).map(|(a, b)| a - b).fold(0.0f32, |m, x| if x.abs() > m.abs() { x } else { m })
}

/// Faces within `r` of `c` turned more than 90 degrees from the base's, of how many.
fn turned_over(ml: &MeshLayers, c: Vec3, r: f32) -> (usize, usize) {
    let doc = &ml.doc;
    let (base, p) = (doc.base_positions(), doc.positions());
    let (mut over, mut n) = (0, 0);
    for f in doc.faces() {
        let (a, b, d) = (f[0] as usize, f[1] as usize, f[2] as usize);
        if base[a].distance(c) > r {
            continue;
        }
        let nb = (base[b] - base[a]).cross(base[d] - base[a]).normalize_or_zero();
        let nc = (p[b] - p[a]).cross(p[d] - p[a]).normalize_or_zero();
        n += 1;
        if nb.dot(nc) < 0.0 {
            over += 1;
        }
    }
    (over, n)
}

/// Bumpiness of what is shown within `r` of `c`: the mean size of (ring average - position) along the base normal.
fn bumpiness(ml: &MeshLayers, c: Vec3, r: f32) -> f64 {
    let doc = &ml.doc;
    let (p, n, topo) = (doc.positions(), ml.base_normals(), doc.topology());
    let (mut s, mut k) = (0.0f64, 0usize);
    for v in 0..p.len() {
        if doc.base_positions()[v].distance(c) > r {
            continue;
        }
        let ring = topo.vert_verts.row(v);
        if ring.is_empty() {
            continue;
        }
        let avg = ring.iter().map(|&u| p[u as usize]).sum::<Vec3>() / ring.len() as f32;
        s += (avg - p[v]).dot(n[v]).abs() as f64;
        k += 1;
    }
    s / k.max(1) as f64
}

/// Smoothing smooths whatever the layer's blend and strength: a clay stroke on a layer, the layer set to Add or
/// Subtract at 50, 100 or 200%, then three smooth strokes: the bumpiness falls with each, no face turns over.
#[test]
fn smoothing_smooths_under_every_blend() {
    let _one = one_at_a_time();
    let (from, to) = (Vec3::new(-0.3, 0.1, 1.0), Vec3::new(0.3, 0.1, 1.0));
    let c = Vec3::new(0.0, 0.1, 1.0).normalize();
    let mut fails = Vec::new();
    for blend in [RowBlend::Add, RowBlend::Subtract] {
        for opacity in [0.5f32, 1.0, 2.0] {
            let mut mode = ball_with_layer(2.64e6);
            app_stroke(&mut mode, Target::Detail, from, to, 0.08, 0.8, Brush::Clay, 1);
            mode.set_layer_blend(0, blend);
            mode.set_layer_opacity(0, opacity);
            mode.close_sculpt_edit();
            settle_frames(&mut mode);
            let mut last = bumpiness(ml(&mut mode), c, 0.2);
            let mut line = format!("{} {:.0}%: bumpiness {last:.3e}", blend.label(), opacity * 100.0);
            for pass in 0..3 {
                app_stroke(&mut mode, Target::Detail, from, to, 0.1, 0.5, Brush::Smooth, 2);
                let b = bumpiness(ml(&mut mode), c, 0.2);
                let (over, nf) = turned_over(ml(&mut mode), c, 0.2);
                line += &format!(" -> {b:.3e} ({over} of {nf} faces turned over)");
                if b >= last || over > 0 {
                    fails.push(format!("{} {:.0}% smooth stroke {}: {last:.3e} -> {b:.3e}, {over} turned over", blend.label(), opacity * 100.0, pass + 1));
                }
                last = b;
            }
            eprintln!("{line}");
        }
    }
    assert!(fails.is_empty(), "{fails:#?}");
}

/// Clay and trim on a Subtract layer, three passes in one stroke, do what they do on an Add layer (they ran away: clay
/// 0.68 r deep, trim 1.9 r high). Clay stays within 1.3 clay steps of where it began. Trim on any layer still runs on
/// (its plane follows the cut down each dab, about 4.9 clay steps here): that is the trim's own matter, so on Subtract
/// it is held to what it does on Add.
#[test]
fn strokes_on_subtract_stay_bounded() {
    let _one = one_at_a_time();
    let (from, to) = (Vec3::new(-0.3, 0.1, 1.0), Vec3::new(0.3, 0.1, 1.0));
    let (r, strength) = (0.08f32, 0.8f32);
    let h = r * (0.015 + 0.08 * strength);
    let mut fails = Vec::new();
    for brush in [Brush::Clay, Brush::Trim] {
        let mut on = [0.0f32; 2];
        for (i, blend) in [RowBlend::Add, RowBlend::Subtract].into_iter().enumerate() {
            let mut mode = ball_with_layer(2.64e6);
            mode.set_layer_blend(0, blend);
            mode.close_sculpt_edit();
            let before = across(&mut mode, true, from, to, r);
            app_stroke(&mut mode, Target::Detail, from, to, r, strength, brush, 3);
            on[i] = extreme(&before, &across(&mut mode, true, from, to, r));
            eprintln!("{brush:?} on a {} layer, three passes: {:.3} r ({:.2} h)", blend.label(), on[i] / r, on[i] / h);
        }
        let [add, sub] = on;
        if !((sub - add).abs() <= 0.02 * add.abs()) {
            fails.push(format!("{brush:?}: {:.3} r on Subtract against {:.3} r on Add", sub / r, add / r));
        }
        if brush == Brush::Clay && !(sub.abs() <= 1.3 * h) {
            fails.push(format!("{brush:?}: {:.3} r, {:.2} h", sub / r, sub / h));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// Ctrl on a layer does what it does on the base: the app swaps the brush (Clay + Ctrl sends Trim) and sets `invert`;
/// Clay + Ctrl lowers and Trim + Ctrl raises.
#[test]
fn ctrl_on_a_layer_does_what_it_does_on_the_base() {
    let _one = one_at_a_time();
    let (from, to) = (Vec3::new(-0.3, 0.1, 1.0), Vec3::new(0.3, 0.1, 1.0));
    let r = 0.08;
    let mut fails = Vec::new();
    // (tool, the brush the app sends with Ctrl, whether it should raise)
    for (tool, sent, raises) in [("Clay", Brush::Trim, false), ("Trim", Brush::Clay, true)] {
        let mut mode = ball_with_layer(2.64e6);
        let before = across(&mut mode, true, from, to, r);
        mode.invert = true;
        app_stroke(&mut mode, Target::Detail, from, to, r, 0.8, sent, 1);
        mode.invert = false;
        let e = extreme(&before, &across(&mut mode, true, from, to, r));
        eprintln!("{tool} + Ctrl on a layer: {:.4} r", e / r);
        if (e > 0.0) != raises || e.abs() < 0.02 * r {
            fails.push(format!("{tool} + Ctrl: {:.4} r", e / r));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// Within `r` of `c` on a ball about the origin: the 99th percentile of the angle (degrees) between the vertex normals
/// of `p` (area weighted over `faces`) and the ball's own, and the spread (sd) of the radius.
fn off_ball(p: &[Vec3], base: &[Vec3], faces: &[sculpt_core::mesh::Face], c: Vec3, r: f32) -> (f64, f64) {
    let mut n = vec![Vec3::ZERO; p.len()];
    for f in faces {
        let vs = sculpt_core::mesh::face_verts(f);
        let fnrm = if vs.len() == 4 {
            (p[vs[2] as usize] - p[vs[0] as usize]).cross(p[vs[3] as usize] - p[vs[1] as usize])
        } else {
            (p[vs[1] as usize] - p[vs[0] as usize]).cross(p[vs[2] as usize] - p[vs[0] as usize])
        };
        for &v in vs {
            n[v as usize] += fnrm;
        }
    }
    let idx: Vec<usize> = (0..p.len()).filter(|&v| base[v].distance(c) < r).collect();
    let mut a: Vec<f64> = idx.iter().map(|&v| (n[v].normalize_or_zero().dot(p[v].normalize()).clamp(-1.0, 1.0) as f64).acos().to_degrees()).collect();
    let rm = idx.iter().map(|&v| p[v].length() as f64).sum::<f64>() / idx.len().max(1) as f64;
    let sd = (idx.iter().map(|&v| (p[v].length() as f64 - rm).powi(2)).sum::<f64>() / idx.len().max(1) as f64).sqrt();
    (pct(&mut a, 0.99), sd)
}

/// Layer 0's stored offsets within `r` of `c`: the mean of their part across the base normal, over the mean edge there.
fn sideways_in_edges(ml: &MeshLayers, c: Vec3, r: f32) -> f64 {
    let doc = &ml.doc;
    let d = ml.layer_deltas(doc.layers()[0].id).unwrap();
    let n = ml.base_normals();
    let (mut side, mut k, mut e, mut ke) = (0.0f64, 0usize, 0.0f64, 0usize);
    for v in 0..d.len() {
        if doc.base_positions()[v].distance(c) > r {
            continue;
        }
        side += (d[v] - n[v] * d[v].dot(n[v])).length() as f64;
        k += 1;
        for &u in doc.topology().vert_verts.row(v) {
            e += doc.base_positions()[v].distance(doc.base_positions()[u as usize]) as f64;
            ke += 1;
        }
    }
    (side / k.max(1) as f64) / (e / ke.max(1) as f64)
}

/// Eight smooth strokes on a plain ball on an Add layer leave it a ball: the stored offsets' sideways part under 1% of
/// an edge, the radius's spread under twice the base's, and with the layer switched to Subtract the normals' 99th
/// percentile off a ball's under twice the base's. (The ring average slid vertices 8 to 13% of an edge, spread the
/// radius 280 times and turned the normals 23 times as far.)
#[test]
fn smooth_on_a_plain_ball_stores_next_to_nothing() {
    let _one = one_at_a_time();
    let (from, to) = (Vec3::new(-0.3, 0.1, 1.0), Vec3::new(0.3, 0.1, 1.0));
    let c = Vec3::new(0.0, 0.1, 1.0).normalize();
    let mut mode = ball_with_layer(2.64e6);
    for _ in 0..8 {
        app_stroke(&mut mode, Target::Detail, from, to, 0.1, 1.0, Brush::Smooth, 2);
    }
    let r = 0.12;
    let side = sideways_in_edges(ml(&mut mode), c, r);
    let (base_p99, base_sd, add_p99, add_sd) = {
        let doc = &ml(&mut mode).doc;
        let (b, a) = (off_ball(doc.base_positions(), doc.base_positions(), doc.faces(), c, r), off_ball(doc.positions(), doc.base_positions(), doc.faces(), c, r));
        (b.0, b.1, a.0, a.1)
    };
    mode.set_layer_blend(0, RowBlend::Subtract);
    mode.close_sculpt_edit();
    settle_frames(&mut mode);
    let sub_p99 = {
        let doc = &ml(&mut mode).doc;
        off_ball(doc.positions(), doc.base_positions(), doc.faces(), c, r).0
    };
    eprintln!(
        "sideways {:.2}% of an edge; radius sd {:.3e} (base {base_sd:.3e}, {:.2} times); normals p99 off a ball's: base {base_p99:.3} deg, Add {add_p99:.3}, Subtract {sub_p99:.3} ({:.2} times)",
        side * 100.0,
        add_sd,
        add_sd / base_sd,
        sub_p99 / base_p99
    );
    assert!(side < 0.01, "sideways {:.2}% of an edge", side * 100.0);
    assert!(add_sd < 2.0 * base_sd, "radius sd {add_sd:.3e} against the base's {base_sd:.3e}");
    assert!(sub_p99 < 2.0 * base_p99, "normals p99 {sub_p99:.3} deg on Subtract against the base's {base_p99:.3}");
}

/// Heights of what is shown (`layer`: the layers' composite, else the volume) over a patch of directions about the
/// stroke in [`layer_smooth_matches_base_smooth`].
fn patch(mode: &mut VolumeMode, layer: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for j in 0..9 {
        for i in 0..25 {
            let d = Vec3::new(-0.15 + 0.3 * i as f32 / 24.0, 0.05 + 0.1 * j as f32 / 8.0, 1.0).normalize();
            out.push(height(mode, layer, d));
        }
    }
    out
}

fn rms_between(a: &[f32], b: &[f32]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// A field of small bumps on the volume, then the same smooth stroke on the volume and on a mesh layer over it: what is
/// left of the bumps is within 15% of each other, at 250k and at 2.5M triangles.
#[test]
fn layer_smooth_matches_base_smooth() {
    let _one = one_at_a_time();
    let (from, to) = (Vec3::new(-0.3, 0.1, 1.0), Vec3::new(0.3, 0.1, 1.0));
    let mut fails = Vec::new();
    for tris in [6.6e5f32, 6.6e6] {
        let mut left = [0.0f64; 2];
        for (k, layer) in [false, true].into_iter().enumerate() {
            let mut mode = VolumeMode::ball(tris);
            let plain = patch(&mut mode, false);
            for j in 0..3 {
                for i in 0..5 {
                    let d = Vec3::new(-0.2 + 0.1 * i as f32 + 0.03 * j as f32, 0.05 + 0.05 * j as f32, 1.0);
                    app_stroke(&mut mode, Target::Form, d, d, 0.03, 1.0, Brush::Clay, 1);
                }
            }
            if layer {
                mode.add_mesh_layer();
                settle_frames(&mut mode);
            }
            let bumps = patch(&mut mode, layer);
            app_stroke(&mut mode, if layer { Target::Detail } else { Target::Form }, from, to, 0.1, 0.5, Brush::Smooth, 2);
            let after = patch(&mut mode, layer);
            left[k] = rms_between(&after, &plain) / rms_between(&bumps, &plain);
            eprintln!(
                "{tris:.0} on the {}: bumps {:.3e} rms, after smoothing {:.3e} ({:.1}% left)",
                if layer { "layer" } else { "volume" },
                rms_between(&bumps, &plain),
                rms_between(&after, &plain),
                left[k] * 100.0
            );
        }
        if !((left[1] - left[0]).abs() <= 0.15 * left[0]) {
            fails.push(format!("{tris:.0}: {:.1}% left on the layer against {:.1}% on the volume", left[1] * 100.0, left[0] * 100.0));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

// ------------------------------------------------------------------------------------------- timing (ignored)

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// A ball of about 2.5M triangles with a mesh layer.
fn big() -> VolumeMode {
    let t = Instant::now();
    let mode = ball_with_layer(6.6e6);
    let tris: usize = mode.mesh.as_ref().unwrap().layers.doc.faces().iter().map(|f| sculpt_core::mesh::face_len(f) - 2).sum();
    eprintln!("ball and mesh layer: {tris} triangles, {:.1} s", t.elapsed().as_secs_f64());
    assert!(tris > 2_000_000);
    mode
}

/// Layer dabs at 2.5M triangles, brush over about 3k vertices: median < 1 ms, p95 < 2 ms for each of Clay, Trim and
/// Smooth, on an Add and on a Max layer; a 50k-vertex brush reported.
#[test]
#[ignore]
fn layer_dab_times_at_2_5m() {
    let _one = one_at_a_time();
    let mut mode = big();
    let verts_in = |mode: &VolumeMode, c: Vec3, r: f32| mode.mesh.as_ref().unwrap().layers.doc.positions().iter().filter(|p| p.distance(c) < r).count();
    let mut fails = Vec::new();
    for (blend, r) in [(RowBlend::Add, 0.098), (RowBlend::Raise, 0.098), (RowBlend::Add, 0.4), (RowBlend::Raise, 0.4)] {
        mode.set_layer_blend(0, blend);
        mode.target = Target::Detail;
        let d0 = Vec3::new(0.0, 0.0, 1.0);
        let (c, _) = mode.raycast(d0 * 3.0, -d0).unwrap();
        let n = verts_in(&mode, c, r);
        let brushes = [Brush::Clay, Brush::Trim, Brush::Smooth];
        let mut times: [Vec<f64>; 3] = Default::default();
        for s in 0..6 {
            let k = s % 3;
            mode.begin_stroke();
            for i in 0..60 {
                let d = Vec3::new(-0.4 + 0.8 * i as f32 / 59.0, 0.1 * (s / 3) as f32 - 0.05, 1.0).normalize();
                let (hit, _) = mode.raycast(d * 3.0, -d).unwrap();
                let t = Instant::now();
                mode.brush_dab(hit, r, 0.002, 0.7, 0.3, brushes[k]);
                times[k].push(t.elapsed().as_secs_f64() * 1e3);
            }
            mode.end_stroke();
            frame(&mut mode);
        }
        let t = Instant::now();
        let rows = frame(&mut mode);
        let frame_ms = t.elapsed().as_secs_f64() * 1e3;
        for (k, times) in times.iter_mut().enumerate() {
            let max = times.iter().cloned().fold(0.0, f64::max);
            let (med, p95) = (pct(times, 0.5), pct(times, 0.95));
            eprintln!("{} {:?} brush r {r} (~{n} vertices): dab median {med:.2} ms, p95 {p95:.2} ms, max {max:.2}", blend.label(), brushes[k]);
            if r < 0.1 && (med >= 1.0 || p95 >= 2.0) {
                fails.push(format!("{} {:?}: median {med:.2} p95 {p95:.2}", blend.label(), brushes[k]));
            }
        }
        eprintln!("{} r {r}: frame after {frame_ms:.1} ms ({} blocks)", blend.label(), rows.blocks.len());
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// The frames of a session at 2.5M triangles: volume stroke ends with layers (< 2 ms on the frame), 20 mixed strokes
/// with the pen up a while between them (no frame over 16 ms; the first mesh is made on its thread before), undo of
/// a volume step (waited for, reported).
#[test]
#[ignore]
fn session_frames_at_2_5m() {
    let _one = one_at_a_time();
    let mut mode = big();
    // A second layer all over, so every swap carries every vertex.
    mode.add_mesh_layer();
    for d in [Vec3::Z, -Vec3::Z, Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y] {
        stroke(&mut mode, Target::Detail, d, 0.9, Brush::Clay);
    }
    settle_frames(&mut mode);
    let mut frames: Vec<(f64, f64, &str)> = Vec::new();
    let mut end_frames = Vec::new();
    let mut waits = Vec::new();
    let timed = |mode: &mut VolumeMode, what: &'static str, frames: &mut Vec<(f64, f64, &str)>| {
        let t = Instant::now();
        let dabs = mode.run_frame_dabs();
        let td = t.elapsed().as_secs_f64() * 1e3;
        mode.tick();
        mode.vol.drain_dirty();
        let _ = mode.take_rows_dirty();
        let all = t.elapsed().as_secs_f64() * 1e3;
        frames.push((all, all - td, what));
        let _ = dabs;
    };
    for i in 0..20 {
        let layer = i % 3 == 2;
        let d = Vec3::new((i as f32 * 1.7).sin(), (i as f32 * 0.9).cos() * 0.6, (i as f32 * 1.3).cos()).normalize();
        mode.target = if layer { Target::Detail } else { Target::Form };
        let t = Instant::now();
        mode.begin_stroke();
        if layer {
            waits.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let side = d.any_orthonormal_vector();
        for k in 0..24 {
            let dd = (d + side * (0.2 * (k as f32 / 23.0 - 0.5))).normalize();
            if let Some((hit, _)) = mode.raycast(dd * 3.0, -dd) {
                mode.queue_dab(hit, if layer { 0.05 } else { 0.08 }, 0.002, 0.7, 0.3, Brush::Clay);
            }
            if k % 3 == 2 {
                timed(&mut mode, "stroke", &mut frames);
                std::thread::sleep(Duration::from_millis(8));
            }
        }
        let t = Instant::now();
        mode.end_stroke();
        end_frames.push(t.elapsed().as_secs_f64() * 1e3);
        // The pen up for a while: frames at about 60 a second.
        for _ in 0..30 {
            timed(&mut mode, if layer { "after a layer stroke" } else { "after a volume stroke" }, &mut frames);
            std::thread::sleep(Duration::from_millis(14));
        }
    }
    settle_frames(&mut mode);
    // Undo of a volume step: the frame it is asked on, and the frames until its refit lands.
    let mut undo_frames = Vec::new();
    let t = Instant::now();
    assert!(mode.undo());
    let undo_ms = t.elapsed().as_secs_f64() * 1e3;
    while mode.busy() {
        let t = Instant::now();
        frame(&mut mode);
        undo_frames.push(t.elapsed().as_secs_f64() * 1e3);
        std::thread::sleep(Duration::from_millis(14));
    }
    let undo_worst = undo_frames.iter().cloned().fold(0.0, f64::max);
    eprintln!("undo: {undo_ms:.1} ms on its frame, then {} frames, worst {undo_worst:.1} ms", undo_frames.len());
    assert!(undo_ms < 16.0 && undo_worst < 16.0, "undo frames over 16 ms");
    let worst = frames.iter().cloned().fold((0.0, 0.0, ""), |a, b| if b.1 > a.1 { b } else { a });
    let worst_all = frames.iter().map(|f| f.0).fold(0.0, f64::max);
    let m = mode.mesh.as_ref().unwrap();
    eprintln!(
        "session: {} frames, worst frame without dabs {:.1} ms ({}), worst with dabs {worst_all:.1} ms; stroke ends {:.2} ms max; layer stroke waits {:?} ms; {} swaps, last swap frame {:.1} ms; undo of a volume step {undo_ms:.0} ms",
        frames.len(),
        worst.1,
        worst.2,
        end_frames.iter().cloned().fold(0.0, f64::max),
        waits.iter().map(|w| w.round()).collect::<Vec<_>>(),
        m.times.swaps,
        m.times.swap_frame_ms
    );
    let mut over: Vec<f64> = frames.iter().filter(|f| f.1 > 16.0).map(|f| f.1).collect();
    over.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(end_frames.iter().all(|&e| e < 2.0), "stroke end {end_frames:?}");
    assert!(over.is_empty(), "{} frames over 16 ms: {:?}", over.len(), &over[..over.len().min(8)]);
}

/// Memory of the process (Windows: working set now, and its peak).
fn memory() -> (f64, f64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_ws: usize,
            ws: usize,
            a: usize,
            b: usize,
            c: usize,
            d: usize,
            pagefile: usize,
            peak_pagefile: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(p: isize, c: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
        unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        }
        return (c.ws as f64 / 1e6, c.peak_ws as f64 / 1e6);
    }
    #[allow(unreachable_code)]
    (0.0, 0.0)
}

/// Peak memory at 2.5M triangles with two layers all over, through 20 volume strokes: reported; over 1.5 GB above
/// the volume alone fails.
#[test]
#[ignore]
fn memory_at_2_5m() {
    let _one = one_at_a_time();
    let mut mode = VolumeMode::ball(6.6e6);
    for d in [Vec3::new(0.3, 0.2, 1.0), Vec3::new(-0.7, 0.1, 0.5)] {
        stroke(&mut mode, Target::Form, d, 0.1, Brush::Clay);
    }
    settle_frames(&mut mode);
    let (volume_only, _) = memory();
    mode.add_mesh_layer();
    settle_frames(&mut mode);
    for l in 0..2 {
        if l == 1 {
            mode.add_mesh_layer();
        }
        for d in [Vec3::Z, -Vec3::Z, Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y] {
            stroke(&mut mode, Target::Detail, d, 0.9, Brush::Clay);
        }
    }
    let (with_layers, _) = memory();
    let mut peak: f64 = 0.0;
    for i in 0..20 {
        let d = Vec3::new((i as f32 * 1.7).sin(), (i as f32 * 0.9).cos() * 0.6, (i as f32 * 1.3).cos()).normalize();
        stroke(&mut mode, Target::Form, d, 0.1, Brush::Clay);
        settle_frames(&mut mode);
        peak = peak.max(memory().0);
    }
    let (_, peak_ws) = memory();
    eprintln!(
        "memory: volume only {volume_only:.0} MB; with two layers all over {with_layers:.0} MB; through 20 volume strokes at most {peak:.0} MB (peak working set {peak_ws:.0} MB): {:.0} MB above the volume alone",
        peak - volume_only
    );
    assert!(peak - volume_only < 1500.0, "{:.0} MB above the volume alone", peak - volume_only);
}

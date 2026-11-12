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
fn stroke(mode: &mut VolumeMode, target: Target, dir: Vec3, r: f32, brush: Brush) {
    mode.target = target;
    mode.begin_stroke();
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
    let mut mode = ball_with_layer(2e5);
    stroke(&mut mode, Target::Detail, Vec3::new(0.2, 0.1, 1.0), 0.08, Brush::Clay);
    stroke(&mut mode, Target::Form, Vec3::new(0.25, 0.1, 1.0), 0.12, Brush::Clay);
    settle_frames(&mut mode);
    base_is_fresh(&mut mode, "after a volume stroke");
    // Strokes begun while the last one's end is still landing: run ahead on a copy, done for real later.
    stroke(&mut mode, Target::Form, Vec3::new(-0.5, 0.3, 0.8), 0.1, Brush::Trim);
    stroke(&mut mode, Target::Form, Vec3::new(-0.4, 0.2, 0.9), 0.05, Brush::Clay);
    assert!(mode.ahead.is_some(), "the second stroke ran ahead");
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

/// Three volume strokes ending within one refit (one swap covering all three), over a layer: undo three times gives
/// the start back bit for bit, redo three times the end.
#[test]
fn three_volume_strokes_in_one_swap_undo_redo() {
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

/// The rows told of every change: a dab's blocks, a swap's, a hidden layer's.
#[test]
fn rows_follow_dabs_swaps_and_hiding() {
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
    assert!(!r.all && !r.blocks.is_empty(), "a dab: {r:?}");
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

/// Layer dabs at 2.5M triangles, brush over about 3k vertices: median < 1 ms, p95 < 2 ms, on an Add and on a Max
/// layer; a 50k-vertex brush reported.
#[test]
#[ignore]
fn layer_dab_times_at_2_5m() {
    let mut mode = big();
    let verts_in = |mode: &VolumeMode, c: Vec3, r: f32| mode.mesh.as_ref().unwrap().layers.doc.positions().iter().filter(|p| p.distance(c) < r).count();
    let mut fails = Vec::new();
    for (blend, r) in [(RowBlend::Add, 0.098), (RowBlend::Raise, 0.098), (RowBlend::Add, 0.4), (RowBlend::Raise, 0.4)] {
        mode.set_layer_blend(0, blend);
        mode.target = Target::Detail;
        let d0 = Vec3::new(0.0, 0.0, 1.0);
        let (c, _) = mode.raycast(d0 * 3.0, -d0).unwrap();
        let n = verts_in(&mode, c, r);
        let mut times = Vec::new();
        for s in 0..4 {
            mode.begin_stroke();
            for k in 0..60 {
                let d = Vec3::new(-0.4 + 0.8 * k as f32 / 59.0, 0.15 * s as f32 - 0.2, 1.0).normalize();
                let (hit, _) = mode.raycast(d * 3.0, -d).unwrap();
                let t = Instant::now();
                mode.brush_dab(hit, r, 0.002, 0.7, 0.3, if s % 2 == 0 { Brush::Clay } else { Brush::Trim });
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            mode.end_stroke();
            frame(&mut mode);
        }
        let t = Instant::now();
        let rows = frame(&mut mode);
        let (med, p95) = (pct(&mut times, 0.5), pct(&mut times, 0.95));
        eprintln!("{} brush r {r} (~{n} vertices): dab median {med:.2} ms, p95 {p95:.2} ms, max {:.2}; frame after {:.1} ms ({} blocks)", blend.label(), times.last().unwrap(), t.elapsed().as_secs_f64() * 1e3, rows.blocks.len());
        if r < 0.1 && (med >= 1.0 || p95 >= 2.0) {
            fails.push(format!("{}: median {med:.2} p95 {p95:.2}", blend.label()));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// The frames of a session at 2.5M triangles: volume stroke ends with layers (< 2 ms on the frame), 20 mixed strokes
/// with the pen up a while between them (no frame over 16 ms; the first mesh is made on its thread before), undo of
/// a volume step (waited for, reported).
#[test]
#[ignore]
fn session_frames_at_2_5m() {
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
    let t = Instant::now();
    assert!(mode.undo());
    let undo_ms = t.elapsed().as_secs_f64() * 1e3;
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

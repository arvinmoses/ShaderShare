//! Mesh layers, speed: the block cache (B1: only the blocks a volume edit reached are extracted again, and the mesh
//! assembled from the kept blocks is the full extraction's bit for bit), the refit off the main thread (B2: the stroke's
//! end only starts a job; the swap applies the plan to the layers as they are by then), and the extracted normals
//! (B3). The 2.47M-triangle gates are ignored: run them in release.

use std::time::{Duration, Instant};

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::geom::Ray;
use sculpt_core::mesh::face_len;
use sculpt_core::mesh_layers::{BlockCache, MeshLayerSettings, MeshLayers, extract_all, extract_named};
use sculpt_core::{Document, LayerId};

mod common;
use common::*;

/// Directions all over the ball, the `i`th of a spiral.
fn spiral(i: usize, n: usize) -> Vec3 {
    let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
    let a = i as f32 * 2.399_963;
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * a.cos(), r * a.sin(), z)
}

/// The cache's mesh is the full extraction's, bit for bit: names, positions, normals and faces.
fn assert_same_as_full(cache: &BlockCache, vol: &LevelVolume, cap: usize, what: &str) {
    let a = cache.assemble().unwrap();
    let b = extract_all(vol, cap).unwrap();
    assert_eq!(a.keys, b.keys, "{what}: names");
    assert!(same(&a.mesh.positions, &b.mesh.positions), "{what}: positions");
    assert!(same(&a.normals, &b.normals), "{what}: normals");
    assert_eq!(a.mesh.faces, b.mesh.faces, "{what}: faces");
    let (m, k, _) = extract_named(vol, cap).unwrap();
    assert_eq!(k, a.keys, "{what}: extract_named names");
    assert!(same(&m.positions, &a.mesh.positions), "{what}: extract_named positions");
}

// ------------------------------------------------------------------------------------------------------------- B1

#[test]
fn b1_cached_assembly_is_the_full_extraction_after_50_strokes() {
    // Two levels: strokes at the finer one make new blocks there, and transition cells around them.
    let mut vol = ball(1.0 / 32.0, 2, R);
    let cap = usize::MAX;
    let mut cache = BlockCache::build(&vol, cap);
    vol.drain_dirty();
    assert_same_as_full(&cache, &vol, cap, "start");
    let (mut extracted, mut removed) = (0, 0);
    for i in 0..50 {
        // Every fifth stroke goes over the one before, the rest all over.
        let c = if i % 5 == 4 { spiral(i - 1, 50) } else { spiral(i, 50) };
        vol_stroke(&mut vol, c, [0.08, 0.15, 0.05][i % 3]);
        let dirty = vol.drain_dirty();
        let st = cache.update(&vol, &dirty);
        extracted += st.extracted;
        removed += st.removed;
        assert_same_as_full(&cache, &vol, cap, &format!("stroke {i}"));
    }
    eprintln!("B1: 50 strokes; {} blocks kept, {extracted} extracted again in all, {removed} dropped; bit-equal after each", cache.len());
}

#[test]
#[ignore]
fn b1_partial_extract_speed_at_2_5m_triangles() {
    let t = Instant::now();
    let mut vol = ball(1.0 / 512.0, 1, 0.5);
    eprintln!("B1: volume {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let cap = usize::MAX;
    let t = Instant::now();
    let mut cache = BlockCache::build(&vol, cap);
    eprintln!("B1: cache built in {:.0} ms, {} blocks", t.elapsed().as_secs_f64() * 1e3, cache.len());
    vol.drain_dirty();
    let t = Instant::now();
    let full = extract_all(&vol, cap).unwrap();
    let full_ms = t.elapsed().as_secs_f64() * 1e3;
    let tris: usize = full.mesh.faces.iter().map(|f| face_len(f) - 2).sum();
    eprintln!("B1: full extraction {full_ms:.0} ms, {tris} triangles");
    assert!(tris > 2_000_000);
    drop(full);
    let mut worst = 0.0f64;
    let mut worst_assemble = 0.0f64;
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.3, 0.8, -0.2), Vec3::new(1.0, -0.2, 0.1), Vec3::new(0.0, 0.0, 1.0), Vec3::new(-0.5, -0.5, 0.5)].into_iter().enumerate() {
        vol_stroke(&mut vol, c, 0.1);
        let dirty = vol.drain_dirty();
        let t = Instant::now();
        let st = cache.update(&vol, &dirty);
        let part = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let a = cache.assemble().unwrap();
        let asm = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "B1 stroke {i}: partial extract {part:.1} ms (set {:.1}, extract {:.1}); {} changed, {} extracted, {} dropped; assembly {asm:.1} ms ({} verts)",
            st.set_ms, st.extract_ms, st.changed, st.extracted, st.removed, a.keys.len()
        );
        worst = worst.max(part);
        worst_assemble = worst_assemble.max(asm);
    }
    assert_same_as_full(&cache, &vol, cap, "after the strokes");
    eprintln!("B1: worst partial extract {worst:.1} ms, worst assembly {worst_assemble:.1} ms");
    assert!(worst < 20.0, "partial extract took {worst:.1} ms");
}

// ------------------------------------------------------------------------------------------------------------- B3

#[test]
fn b3_normals_come_out_of_extraction() {
    let mut vol = ball(1.0 / 32.0, 2, R);
    vol_stroke(&mut vol, Vec3::new(0.3, 0.2, 1.0), 0.15);
    let e = extract_all(&vol, usize::MAX).unwrap();
    assert_eq!(e.normals.len(), e.mesh.positions.len());
    assert_eq!(e.normals.len(), e.keys.len());
    let off_unit = e.normals.iter().filter(|n| (n.length() - 1.0).abs() > 1e-5).count();
    // Away from the stroke the surface is the ball's: the normal is the direction from the centre.
    let stroke_dir = Vec3::new(0.3, 0.2, 1.0).normalize();
    let (mut checked, mut worst) = (0, 1.0f32);
    for (p, n) in e.mesh.positions.iter().zip(&e.normals) {
        if p.normalize().dot(stroke_dir) > 0.7 {
            continue;
        }
        checked += 1;
        worst = worst.min(n.dot(p.normalize()));
    }
    // In the mesh's order: each vertex's normal agrees with its faces' (outward, as the faces wind).
    let mut area = vec![Vec3::ZERO; e.mesh.positions.len()];
    for f in &e.mesh.faces {
        let k = face_len(f);
        let p = |i: usize| e.mesh.positions[f[i % k] as usize];
        let n = (p(1) - p(0)).cross(p(2) - p(0)) + if k == 4 { (p(2) - p(0)).cross(p(3) - p(0)) } else { Vec3::ZERO };
        for &v in &f[..k] {
            area[v as usize] += n;
        }
    }
    let disagree = area.iter().zip(&e.normals).filter(|(a, n)| a.normalize_or_zero().dot(**n) < 0.5).count();
    eprintln!("B3: {} normals, {off_unit} not unit; on the plain ball {checked} checked, worst dot with the radial {worst:.5}; {disagree} disagree with their faces", e.normals.len());
    assert_eq!(off_unit, 0);
    assert!(checked > 1000);
    assert!(worst > 0.99, "worst {worst}");
    assert!(disagree * 1000 < e.normals.len(), "{disagree} normals disagree with their faces");
}

// ------------------------------------------------------------------------------------------------------------- B2

/// A small clay stroke on layer `id` at the top of the ball, from direction `dir`.
fn layer_dab(doc: &mut Document, dir: Vec3, radius: f32) {
    let s = BrushSettings { radius, strength: 0.5, ..Default::default() };
    let mut clay = ClayBuildup { height: 0.1, ..Default::default() };
    let side = dir.any_orthonormal_vector();
    let path: Vec<(Vec3, f32)> = (0..=6)
        .filter_map(|k| {
            let d = (dir + side * (0.1 * (k as f32 / 6.0 - 0.5))).normalize();
            doc.raycast(&Ray::new(d * 3.0, -d)).map(|h| (h.point, 1.0))
        })
        .collect();
    brush::stroke(doc, &mut clay, &s, &path).unwrap();
}

fn layer_named(doc: &Document, name: &str) -> Option<LayerId> {
    doc.layers().iter().find(|l| l.name == name).map(|l| l.id)
}

/// The block-cached refit off the thread gives what the plain refit gives (the same carry, split into plan and apply),
/// and edits made to the layers while its job runs are there after the swap.
#[test]
fn b2_async_refit_matches_the_plain_refit_and_keeps_edits_made_meanwhile() {
    let mut vol = ball(1.0 / 64.0, 2, R);
    let mut plain = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let mut live = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let id = column(&mut plain.doc, 4, 0.12);
    assert_eq!(column(&mut live.doc, 4, 0.12), id);
    vol.drain_dirty();
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.8, 0.3, -0.4), Vec3::new(0.1, -0.9, 0.3)].into_iter().enumerate() {
        let stroke = vol_stroke(&mut vol, c, 0.1);
        let dirty = vol.drain_dirty();
        let region = MeshLayers::changed_box(&vol, &dirty);
        let (_, plain_patch) = plain.refit_patch(&vol, region).unwrap();
        let (st, patch) = live.refit_cached_patch(&vol, &dirty, region).unwrap();
        eprintln!("B2 small {i}: {st:?}");
        assert!(!st.discarded);
        assert_eq!((st.patch_len, patch.len()), (plain_patch.len(), plain_patch.len()), "stroke {i}: patch");
        assert!(!patch.is_empty() && live.take_patches().is_empty());
        assert_eq!(live.keys(), plain.keys(), "stroke {i}: names");
        assert!(same(live.doc.base_positions(), plain.doc.base_positions()), "stroke {i}: base");
        assert!(same(live.doc.positions(), plain.doc.positions()), "stroke {i}: composite");
        assert!(by_key(&live, id) == by_key(&plain, id), "stroke {i}: layer");
        let _ = stroke;
    }

    // Edits while a job runs: a dab on the column layer and a new layer with its own dab, both on top, away from the
    // volume stroke below.
    let stroke = vol_stroke(&mut vol, Vec3::new(0.3, 0.3, -1.0), 0.1);
    let dirty = vol.drain_dirty();
    let region = MeshLayers::changed_box(&vol, &dirty);
    let spent = live.refit_async(&vol, &dirty, region);
    assert!(live.busy());
    layer_dab(&mut live.doc, Vec3::new(0.2, 0.0, 1.0), 0.06);
    let id2 = live.doc.add_layer("meanwhile");
    layer_dab(&mut live.doc, Vec3::new(-0.2, 0.1, 1.0), 0.06);
    let still_busy = live.busy();
    let before = (by_key(&live, id), by_key(&live, id2));
    let st = loop {
        if let Some(r) = live.poll(&vol) {
            break r.unwrap();
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    eprintln!("B2 meanwhile: start {spent:.2} ms, edits done while the job ran: {still_busy}; {st:?}");
    assert!(!st.discarded);
    assert_eq!(layer_named(&live.doc, "meanwhile"), Some(id2), "the new layer is gone");
    let after = (by_key(&live, id), by_key(&live, id2));
    let gate = grown(&region, 3.0 * vol.voxel(vol.levels() - 1));
    for (which, (b, a)) in [(id, (&before.0, &after.0)), (id2, (&before.1, &after.1))] {
        let (mut checked, mut moved, mut lost) = (0usize, 0usize, 0usize);
        let mut sum = (0.0f64, 0.0f64);
        for (k, (p0, d0)) in b {
            sum.0 += d0.length() as f64;
            if inside(&gate, *p0) {
                continue;
            }
            let Some((p, d)) = a.get(k) else {
                lost += 1;
                continue;
            };
            checked += 1;
            if bits(&[*p, *d]) != bits(&[*p0, *d0]) {
                moved += 1;
            }
        }
        sum.1 = a.values().map(|(_, d)| d.length() as f64).sum();
        eprintln!("B2 meanwhile, layer {which:?}: {checked} vertices outside the stroke, {moved} differ, {lost} lost; |d| sum {:.4} -> {:.4}", sum.0, sum.1);
        assert!(checked > 1000);
        assert_eq!((moved, lost), (0, 0));
        assert!(sum.0 > 0.0 && (sum.1 / sum.0 - 1.0).abs() < 1e-3);
    }
    let _ = stroke;
}

/// Carve the whole ball away, as one stroke: the mesh it leaves has no faces.
fn carve_all(vol: &mut LevelVolume) {
    vol.begin_stroke();
    for l in 0..vol.levels() {
        vol.ball_dab(l, Vec3::ZERO, 2.0, 0.05, true);
    }
    vol.end_stroke();
}

/// A refit that cannot succeed (everything carved away: no faces) fails once: the job is not started again until new
/// blocks come, so waiting for it ends; undone, the layers come back bit for bit.
#[test]
fn b2_a_failed_refit_is_not_run_again_and_undo_brings_the_layers_back() {
    let (mut vol, mut ml) = small();
    let id = column(&mut ml.doc, 2, 0.12);
    vol.drain_dirty();
    let before = by_key(&ml, id);
    carve_all(&mut vol);
    let dirty = vol.drain_dirty();
    assert!(!dirty.is_empty());
    let region = MeshLayers::changed_box(&vol, &dirty);
    // refit_cached waits while the job is busy: with the job run again on every poll it never returned.
    let r = ml.refit_cached(&vol, &dirty, region);
    eprintln!("refit of nothing: {:?}", r.as_ref().err());
    assert!(r.is_err(), "a refit of an empty volume succeeded");
    assert!(!ml.busy() && ml.refit_failed());
    // Polled and waited for again with no new blocks: nothing runs.
    assert!(ml.poll(&vol).is_none() && ml.wait(&vol).is_none());
    assert!(!ml.busy());
    // The volume's undo brings new blocks: the job runs and the layers are as they were.
    assert!(vol.undo());
    let dirty = vol.drain_dirty();
    let st = ml.refit_cached(&vol, &dirty, MeshLayers::changed_box(&vol, &dirty)).unwrap();
    assert!(!st.discarded && !ml.refit_failed());
    assert!(by_key(&ml, id) == before, "the layers came back changed");
    // Redo fails again, and undo after it brings them back again.
    assert!(vol.redo());
    let dirty = vol.drain_dirty();
    assert!(ml.refit_cached(&vol, &dirty, MeshLayers::changed_box(&vol, &dirty)).is_err());
    assert!(!ml.busy());
    assert!(vol.undo());
    let dirty = vol.drain_dirty();
    ml.refit_cached(&vol, &dirty, MeshLayers::changed_box(&vol, &dirty)).unwrap();
    assert!(by_key(&ml, id) == before, "the layers came back changed after the second undo");
}

#[test]
#[ignore]
fn b2_async_refit_at_2_5m_triangles() {
    b2_at_2_5m(false);
}

/// The same with a second layer over the whole surface: the swap then carries every vertex.
#[test]
#[ignore]
fn b2_async_refit_at_2_5m_triangles_layer_everywhere() {
    b2_at_2_5m(true);
}

fn b2_at_2_5m(everywhere: bool) {
    let t = Instant::now();
    let mut vol = ball(1.0 / 512.0, 1, 0.5);
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let tris: usize = ml.doc.faces().iter().map(|f| face_len(f) - 2).sum();
    eprintln!("B2: volume + mesh layers {:.0} ms, {} verts, {tris} tris", t.elapsed().as_secs_f64() * 1e3, ml.doc.vertex_count());
    assert!(tris > 2_000_000);
    if everywhere {
        ml.doc.add_layer("all over");
        for d in [Vec3::Z, -Vec3::Z, Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y] {
            layer_dab(&mut ml.doc, d, 0.9);
        }
        let all = ml.doc.layers().last().unwrap().id;
        let d = ml.layer_deltas(all).unwrap();
        eprintln!("B2: layer all over moves {} of {} vertices", d.iter().filter(|v| **v != Vec3::ZERO).count(), d.len());
    }
    let id = ml.doc.add_layer("clay");
    for k in 0..4 {
        layer_dab(&mut ml.doc, Vec3::new(-0.05 + 0.03 * k as f32, 0.0, 1.0), 0.12);
    }
    vol.drain_dirty();
    let (mut worst_end, mut worst_swap, mut worst_frame, mut worst_latency) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.3, 0.8, -0.2), Vec3::new(1.0, -0.2, 0.1), Vec3::new(-0.6, -0.6, -0.3)].into_iter().enumerate() {
        let stroke = vol_stroke(&mut vol, c, 0.1);
        let dirty = vol.drain_dirty();
        let t0 = Instant::now();
        let start_ms = ml.refit_async(&vol, &dirty, stroke);
        let end_ms = t0.elapsed().as_secs_f64() * 1e3;
        // The artist goes on with the layers while the job runs (the first time).
        let mut edit_ms = 0.0;
        if i == 0 {
            let t = Instant::now();
            layer_dab(&mut ml.doc, Vec3::new(0.1, 0.1, 1.0), 0.05);
            edit_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        let (st, frame_ms) = loop {
            let t = Instant::now();
            if let Some(r) = ml.poll(&vol) {
                break (r.unwrap(), t.elapsed().as_secs_f64() * 1e3);
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let latency = t0.elapsed().as_secs_f64() * 1e3;
        let j = st.job;
        eprintln!(
            "B2 stroke {i}: stroke end {end_ms:.2} ms (in refit_async {start_ms:.2}); swap frame {frame_ms:.1} ms (apply {:.1}, recomposite {:.1}); latency {latency:.0} ms (job {:.0}: set {:.1}, extract {:.1}, assemble {:.1}, document {:.0}, names {:.1}, closest point {:.1}, geometry copy {:.1}); copied {}, projected {}; layer edit meanwhile {edit_ms:.0} ms",
            st.apply_ms, st.recomposite_ms, j.total_ms, j.cache.set_ms, j.cache.extract_ms, j.assemble_ms, j.build_ms, j.match_ms, j.project_ms, j.geom_ms, j.copied, j.projected
        );
        assert!(!st.discarded && !j.rebuilt_cache);
        worst_end = worst_end.max(end_ms);
        worst_swap = worst_swap.max(st.swap_ms);
        worst_frame = worst_frame.max(frame_ms);
        worst_latency = worst_latency.max(latency);
    }
    let (sum, _) = delta_stats(&ml, id);
    eprintln!("B2: worst stroke end {worst_end:.2} ms, swap {worst_swap:.1} ms (frame {worst_frame:.1}), latency {worst_latency:.0} ms; layer |d| sum {sum:.3}");
    assert!(worst_end < 2.0, "stroke end {worst_end:.2} ms");
    assert!(worst_frame < 16.0, "swap frame {worst_frame:.1} ms");
    assert!(worst_latency < 400.0, "latency {worst_latency:.0} ms");
}

// ------------------------------------------------------------------------------------------------------------ C0d

/// The swap frame at 2.47M triangles with a Max layer (its gate reads the base normals) or a noise-masked layer: the
/// basis and the mask values come from the job, so the main thread stays under a frame.
#[test]
#[ignore]
fn c0d_swap_with_a_max_layer_at_2_5m_triangles() {
    c0d_at_2_5m(true);
}

#[test]
#[ignore]
fn c0d_swap_with_a_noise_mask_at_2_5m_triangles() {
    c0d_at_2_5m(false);
}

fn c0d_at_2_5m(max: bool) {
    use sculpt_core::LayerBlend;
    use sculpt_core::mask::{MaskLayer, MaskSource, MaskStack};
    use sculpt_core::noise::NoiseParams;
    let what = if max { "Max" } else { "noise mask" };
    let t = Instant::now();
    let mut vol = ball(1.0 / 512.0, 1, 0.5);
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let tris: usize = ml.doc.faces().iter().map(|f| face_len(f) - 2).sum();
    eprintln!("C0d: volume + mesh layers {:.0} ms, {} verts, {tris} tris", t.elapsed().as_secs_f64() * 1e3, ml.doc.vertex_count());
    assert!(tris > 2_000_000);
    // Two layers on top: Max and Min, or each with a paint and a noise mask (as the panel makes them).
    for (n, at) in [-0.05f32, 0.1].into_iter().enumerate() {
        let id = ml.doc.add_layer(&format!("clay {n}"));
        for k in 0..4 {
            layer_dab(&mut ml.doc, Vec3::new(at + 0.03 * k as f32, 0.0, 1.0), 0.12);
        }
        if max {
            let mut m = ml.doc.layer(id).unwrap().meta();
            m.blend = if n == 0 { LayerBlend::Max } else { LayerBlend::Min };
            ml.doc.set_layer_meta(id, m, false).unwrap();
        } else {
            let paint = format!("paint.{n}");
            let half: Vec<f32> = ml.doc.base_positions().iter().map(|p| if p.x < at { 0.0 } else { 1.0 }).collect();
            ml.doc.set_channel(&paint, half).unwrap();
            let noise = NoiseParams { seed: n as u32, ..Default::default() };
            let stack = MaskStack::new(1.0)
                .with(MaskLayer::new("paint", MaskSource::Channel { name: paint }).blend(sculpt_core::mask::BlendMode::Multiply))
                .with(MaskLayer::new("noise", MaskSource::Noise(noise)).blend(sculpt_core::mask::BlendMode::Multiply));
            ml.doc.set_layer_mask(id, Some(stack)).unwrap();
        }
    }
    vol.drain_dirty();
    let mut worst = 0.0f64;
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.3, 0.8, -0.2), Vec3::new(1.0, -0.2, 0.1)].into_iter().enumerate() {
        let stroke = vol_stroke(&mut vol, c, 0.1);
        let dirty = vol.drain_dirty();
        ml.refit_async(&vol, &dirty, stroke);
        let (st, frame_ms) = loop {
            let t = Instant::now();
            if let Some(r) = ml.poll(&vol) {
                break (r.unwrap(), t.elapsed().as_secs_f64() * 1e3);
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        eprintln!(
            "C0d {what} stroke {i}: swap frame {frame_ms:.1} ms (swap {:.1}, apply {:.1}, recomposite {:.1}); job {:.0} ms",
            st.swap_ms, st.apply_ms, st.recomposite_ms, st.job.total_ms
        );
        assert!(!st.discarded);
        worst = worst.max(frame_ms);
    }
    eprintln!("C0d {what}: worst swap frame {worst:.1} ms");
    assert!(worst < 16.0, "swap frame {worst:.1} ms");
}

//! Mesh layers over a volume: layers live on the volume's extracted display surface, and survive volume edits by
//! vertex name. Gates G0 (exact layer ops), G2 (identity carry), G3 (carry speed, ignored), G4 (retention), G5
//! (extraction quality; the artist file part ignored), G6 (save and reopen).

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::geom::Ray;
use sculpt_core::mask::{MaskLayer, MaskSource, MaskStack};
use sculpt_core::mesh::face_len;
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers, edge_defects, extract_named};
use sculpt_core::{Document, LayerBlend};

mod common;
use common::*;

// ------------------------------------------------------------------------------------------------------------- G0

#[test]
fn g0_layer_ops_are_exact_on_the_extracted_mesh() {
    let (_vol, mut ml) = small();
    let doc = &mut ml.doc;
    let id = column(doc, 6, 0.12);
    let base = doc.base_positions().to_vec();
    let top = doc.raycast(&Ray::new(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z)).unwrap().point.z - R;
    eprintln!("G0: {} verts, column {top:.4} high", doc.vertex_count());
    assert!(top > 0.05, "the column did not build ({top})");
    // The reference: the sculpted state recomposited.
    doc.set_layer_opacity(id, 0.5).unwrap();
    doc.set_layer_opacity(id, 1.0).unwrap();
    let sculpted = doc.positions().to_vec();
    assert!(!same(&sculpted, &base));
    for o in [0.5f32, 1.5, 2.0] {
        doc.set_layer_opacity(id, o).unwrap();
        assert!(!same(doc.positions(), &sculpted), "opacity {o} changed nothing");
    }
    doc.set_layer_opacity(id, 1.0).unwrap();
    assert!(same(doc.positions(), &sculpted), "opacity back to 1.0 is not exact");
    let set_blend = |doc: &mut Document, b: LayerBlend| {
        let mut m = doc.layer(id).unwrap().meta();
        m.blend = b;
        doc.set_layer_meta(id, m, false).unwrap();
    };
    set_blend(doc, LayerBlend::Subtract);
    assert!(!same(doc.positions(), &sculpted));
    set_blend(doc, LayerBlend::Add);
    assert!(same(doc.positions(), &sculpted), "Subtract and back is not exact");
    doc.set_layer_visible(id, false).unwrap();
    assert!(same(doc.positions(), &base), "hidden is not the base");
    doc.set_layer_visible(id, true).unwrap();
    assert!(same(doc.positions(), &sculpted), "hide/show is not exact");
    // A painted mask over the x < 0 half.
    let half: Vec<f32> = doc.base_positions().iter().map(|p| if p.x < 0.0 { 0.0 } else { 1.0 }).collect();
    doc.set_channel("paint.half", half.clone()).unwrap();
    doc.set_layer_mask(id, Some(MaskStack::new(0.0).with(MaskLayer::new("painted", MaskSource::Channel { name: "paint.half".into() })))).unwrap();
    let (mut masked, mut kept_moved) = (0, 0);
    for v in 0..doc.vertex_count() {
        if half[v] == 0.0 {
            masked += 1;
            assert!(bits(&[doc.positions()[v]]) == bits(&[base[v]]), "masked vertex {v} moved");
        } else if doc.positions()[v] != base[v] {
            kept_moved += 1;
        }
    }
    eprintln!("G0: masked half {masked} vertices at base exactly; {kept_moved} kept vertices raised");
    assert!(kept_moved > 0);
    doc.set_layer_mask(id, None).unwrap();
    assert!(same(doc.positions(), &sculpted), "mask off is not exact");
}

// ------------------------------------------------------------------------------------------------------------- G2

#[test]
fn g2_identity_carry_away_from_the_volume_stroke() {
    let (vol, ml) = small();
    g2(vol, ml, false);
}

/// The same with a stroke at a finer level than the ball holds: the stroke makes finer blocks, and the surface there
/// changes level (transition cells around it).
#[test]
fn g2_identity_carry_with_a_finer_level_stroke() {
    let vol = ball(1.0 / 64.0, 2, R);
    let ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    g2(vol, ml, true);
}

/// `by_blocks`: the region is the box of the blocks the stroke changed (a finer-level stroke changes the volume's
/// surface past the stroke's own box: new blocks are whole blocks, and the cells around them turn into transition cells).
fn g2(mut vol: LevelVolume, mut ml: MeshLayers, by_blocks: bool) {
    let id = column(&mut ml.doc, 6, 0.12);
    let before = by_key(&ml, id);
    let (sum0, max0) = delta_stats(&ml, id);
    // A volume stroke on the far side (the bottom).
    vol.drain_dirty();
    let stroke = vol_stroke(&mut vol, Vec3::new(0.2, 0.1, -1.0), 0.1);
    let dirty = vol.drain_dirty();
    let blocks = MeshLayers::changed_box(&vol, &dirty);
    let region = if by_blocks { blocks } else { stroke };
    let st = ml.refit(&vol, region).unwrap();
    eprintln!("G2: {st:?}");
    eprintln!("G2: stroke box {:?}, changed blocks' box {:?}", stroke, blocks);
    let after = by_key(&ml, id);
    let cell = vol.voxel(vol.levels() - 1);
    let gate = grown(&region, 3.0 * cell);
    let (mut checked, mut base_diff, mut delta_diff, mut moved_outside) = (0usize, 0usize, 0usize, 0usize);
    for (k, (p, d)) in &after {
        if inside(&gate, *p) {
            continue;
        }
        let Some((p0, d0)) = before.get(k) else { continue };
        if inside(&gate, *p0) {
            continue;
        }
        checked += 1;
        if bits(&[*p]) != bits(&[*p0]) {
            base_diff += 1;
        }
        if bits(&[*d]) != bits(&[*d0]) {
            delta_diff += 1;
        }
    }
    // Volume changes outside the gate (would make the gate unfair rather than the carry wrong).
    for (k, (p0, _)) in &before {
        if !inside(&gate, *p0) && !after.contains_key(k) {
            moved_outside += 1;
        }
    }
    let (sum1, max1) = delta_stats(&ml, id);
    eprintln!(
        "G2: {checked} vertices outside the stroke + 3 cells with a surviving name; base differs at {base_diff}, delta at {delta_diff}; {moved_outside} names outside lost; layer |d| sum {sum0:.4} -> {sum1:.4}, max {max0:.5} -> {max1:.5}"
    );
    assert!(checked > 1000);
    assert!(st.projected > 0, "the stroke changed nothing");
    assert_eq!(base_diff, 0);
    assert_eq!(delta_diff, 0);
    assert_eq!(moved_outside, 0);
}

// ------------------------------------------------------------------------------------------------------------- G3

#[test]
#[ignore]
fn g3_carry_speed_at_2_5m_triangles() {
    let r = 0.5;
    let t = Instant::now();
    let mut vol = ball(1.0 / 512.0, 1, r);
    eprintln!("G3: volume {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let tris: usize = ml.doc.faces().iter().map(|f| face_len(f) - 2).sum();
    eprintln!("G3: extract + document {:.0} ms, {} verts, {tris} tris", t.elapsed().as_secs_f64() * 1e3, ml.doc.vertex_count());
    let t = Instant::now();
    let id = {
        // The column scaled to this ball.
        let doc = &mut ml.doc;
        let id = doc.add_layer("clay");
        let s = BrushSettings { radius: 0.12, strength: 0.5, ..Default::default() };
        let mut tall = ClayBuildup { height: 0.32, ..Default::default() };
        for _ in 0..4 {
            let path: Vec<(Vec3, f32)> = (0..=16).filter_map(|k| {
                let tt = k as f32 / 16.0;
                let x = -0.05 + 0.1 * if k <= 8 { tt * 2.0 } else { 2.0 - tt * 2.0 };
                doc.raycast(&Ray::new(Vec3::new(x, 0.0, 3.0), -Vec3::Z)).map(|h| (h.point, 1.0))
            }).collect();
            brush::stroke(doc, &mut tall, &s, &path).unwrap();
        }
        id
    };
    eprintln!("G3: column {:.0} ms, layer |d| {:?}", t.elapsed().as_secs_f64() * 1e3, delta_stats(&ml, id));
    let mut worst = 0.0f64;
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.3, 0.8, -0.2), Vec3::new(1.0, -0.2, 0.1)].into_iter().enumerate() {
        let stroke = vol_stroke(&mut vol, c, 0.1);
        let st = ml.refit(&vol, stroke).unwrap();
        eprintln!(
            "G3 stroke {i}: {} tris; extract {:.0} ms, document build (BVH, topology, normals) {:.0} ms, carry {:.1} ms (names {:.1}, closest point {:.1}, recomposite {:.1}); copied {}, projected {}, searched {} faces",
            st.triangles, st.extract_ms, st.build_ms, st.carry_ms, st.match_ms, st.project_ms, st.recomposite_ms, st.copied, st.projected, st.search_faces
        );
        worst = worst.max(st.carry_ms);
        assert!(st.triangles > 2_000_000);
    }
    assert!(worst < 50.0, "carry took {worst:.1} ms");
}

// ------------------------------------------------------------------------------------------------------------- G4

#[test]
fn g4_far_layer_survives_50_volume_strokes() {
    let (mut vol, mut ml) = small();
    let id = column(&mut ml.doc, 6, 0.12);
    let (sum0, max0) = delta_stats(&ml, id);
    let (mut copied, mut projected, mut ms) = (0, 0, 0.0);
    for i in 0..50 {
        // Around the lower half, away from the column on top.
        let a = i as f32 * 2.4;
        let c = Vec3::new(a.cos(), a.sin(), -0.6 - 0.4 * ((i * 7 % 5) as f32 / 4.0));
        let stroke = vol_stroke(&mut vol, c, 0.08);
        let st = ml.refit(&vol, stroke).unwrap();
        copied += st.copied;
        projected += st.projected;
        ms += st.extract_ms + st.build_ms + st.carry_ms;
    }
    let (sum1, max1) = delta_stats(&ml, id);
    let keep = sum1 / sum0;
    let max_err = (max1 - max0).abs() / max0;
    eprintln!("G4: 50 strokes, refit {:.1} ms each; copied {copied}, projected {projected}; |d| sum {sum0:.4} -> {sum1:.4} ({:.4}%), max {max0:.5} -> {max1:.5} ({:.4}% off)", ms / 50.0, keep * 100.0, max_err * 100.0);
    assert!(keep >= 0.999, "kept {:.4}%", keep * 100.0);
    assert!(max_err <= 0.001, "max off by {:.4}%", max_err * 100.0);
}

// ------------------------------------------------------------------------------------------------------------- G5

#[test]
fn g5_extracted_mesh_is_manifold() {
    // A one-level ball, and a two-level one with finer blocks where a stroke went (transition cells).
    let (_, ml) = small();
    let (bad, flipped) = edge_defects(ml.doc.faces());
    eprintln!("G5: ball, {} verts, {} faces: {bad} non-manifold/open edges, {flipped} flipped", ml.doc.vertex_count(), ml.doc.face_count());
    assert_eq!(bad, 0);
    {
        let vol = ball(1.0 / 64.0, 1, R);
        let (mesh, _, stats) = extract_named(&vol, 8).unwrap();
        whole_surface(&vol, &mesh, &stats, "ball");
    }
    let mut vol = ball(1.0 / 32.0, 2, R);
    vol_stroke(&mut vol, Vec3::new(0.3, 0.2, 1.0), 0.15);
    let (mesh, keys, stats) = extract_named(&vol, 8).unwrap();
    let (bad2, flipped2) = edge_defects(&mesh.faces);
    let mut sorted = keys.clone();
    sorted.dedup();
    eprintln!("G5: two levels after a fine stroke: {} verts, {} faces ({stats:?}): {bad2} non-manifold/open edges, {flipped2} flipped; names unique {}", mesh.positions.len(), mesh.faces.len(), sorted.len() == keys.len());
    assert_eq!(sorted.len(), keys.len(), "names repeat");
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "names are not in extraction order");
    assert_eq!(bad2, 0);
    assert_eq!((flipped, flipped2), (0, 0), "flipped edges");
    whole_surface(&vol, &mesh, &stats, "two levels");
    Document::from_mesh(mesh).unwrap();
}

/// The named mesh is the displayed surface whole: no quad the extraction dropped, none left out in naming, and the
/// same triangles as `surface_in` builds.
fn whole_surface(vol: &LevelVolume, mesh: &sculpt_core::mesh::PolyMesh, stats: &sculpt_core::generator::level_volume::ExtractStats, what: &str) {
    let tris = |faces: &[sculpt_core::mesh::Face]| -> usize { faces.iter().map(|f| face_len(f) - 2).sum() };
    let (shown, _) = vol.surface(8);
    let left_out = (stats.quads + stats.triangles) as i64 - mesh.faces.len() as i64;
    eprintln!("G5 {what}: {} quads dropped in extraction, {left_out} left out in naming; {} triangles, the displayed surface {}", stats.dropped, tris(&mesh.faces), tris(&shown.faces));
    assert_eq!(stats.dropped, 0, "{what}: quads dropped (holes)");
    assert_eq!(left_out, 0, "{what}: faces left out in naming");
    assert_eq!(tris(&mesh.faces), tris(&shown.faces), "{what}: not the displayed surface's triangles");
}

#[test]
#[ignore]
fn g5_artist_file_extraction() {
    let Ok(path) = std::env::var("SCULPT_ARTIST_FILE") else {
        eprintln!("SCULPT_ARTIST_FILE not set; skipped");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let vol = LevelVolume::read_from(&mut std::io::Cursor::new(&bytes[..]), 6).unwrap();
    eprintln!("G5 file: {} levels, blocks {:?}", vol.levels(), (0..vol.levels()).map(|l| vol.level_len(l)).collect::<Vec<_>>());
    let t = Instant::now();
    let (shown, s) = vol.surface(8);
    let shown_ms = t.elapsed().as_secs_f64() * 1e3;
    let shown_tris: usize = shown.faces.iter().map(|f| face_len(f) - 2).sum();
    drop(shown);
    let t = Instant::now();
    let (mesh, _keys, stats) = extract_named(&vol, 8).unwrap();
    let ex_ms = t.elapsed().as_secs_f64() * 1e3;
    let tris: usize = mesh.faces.iter().map(|f| face_len(f) - 2).sum();
    let (bad, flipped) = edge_defects(&mesh.faces);
    let t = Instant::now();
    let doc = Document::from_mesh(mesh).unwrap();
    let doc_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "G5 file: named extraction {tris} tris in {ex_ms:.0} ms (displayed surface {shown_tris} tris, {} quads + {} triangles by its stats, {shown_ms:.0} ms; ratio {:.3}); document {doc_ms:.0} ms, {} verts; {bad} non-manifold/open edges, {flipped} flipped",
        s.quads,
        s.triangles,
        tris as f64 / shown_tris as f64,
        doc.vertex_count()
    );
    assert_eq!(tris, shown_tris, "not the displayed surface's triangles");
    assert_eq!(bad, 0, "non-manifold edges on the artist's surface");
    assert_eq!(flipped, 0, "flipped edges on the artist's surface");
    assert_eq!(stats.dropped, 0, "quads dropped (holes)");
    assert_eq!(stats.quads + stats.triangles, doc.face_count(), "faces left out in naming");
}

// ------------------------------------------------------------------------------------------------------------- G6

#[test]
fn g6_save_and_reopen() {
    let (vol, mut ml) = small();
    let id = column(&mut ml.doc, 4, 0.12);
    let id2 = column(&mut ml.doc, 2, 0.06);
    let half: Vec<f32> = ml.doc.base_positions().iter().map(|p| (p.x * 10.0).clamp(0.0, 1.0)).collect();
    ml.doc.set_channel("paint.half", half).unwrap();
    ml.doc.set_layer_mask(id, Some(MaskStack::new(0.0).with(MaskLayer::new("painted", MaskSource::Channel { name: "paint.half".into() })))).unwrap();
    ml.doc.set_layer_opacity(id, 0.7).unwrap();
    let mut m = ml.doc.layer(id2).unwrap().meta();
    m.blend = LayerBlend::Subtract;
    m.opacity = 1.3;
    ml.doc.set_layer_meta(id2, m, false).unwrap();
    let dir = std::env::temp_dir().join(format!("mesh_layers_g6_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ml.save(&dir).unwrap();
    let (back, lost) = MeshLayers::load(&dir, &vol).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(lost, 0);
    assert_eq!(back.keys(), ml.keys());
    for lid in [id, id2] {
        let (a, b) = (ml.doc.layer(lid).unwrap(), back.doc.layer(lid).unwrap());
        assert_eq!(a.meta(), b.meta(), "layer meta");
        assert_eq!(a.opacity.to_bits(), b.opacity.to_bits());
        let (da, db) = (by_key(&ml, lid), by_key(&back, lid));
        assert_eq!(da.len(), db.len());
        for (k, (p, d)) in &da {
            let (p2, d2) = db[k];
            assert!(bits(&[*p, *d]) == bits(&[p2, d2]), "vertex {k:x} differs");
        }
    }
    // Mask values and the composite come back too.
    let to_c = |m: &MeshLayers, v: &[Vec3]| -> HashMap<u64, [u32; 3]> {
        let c = m.internal_to_canonical();
        v.iter().enumerate().map(|(i, p)| (m.keys()[c[i] as usize], [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])).collect()
    };
    assert!(to_c(&ml, ml.doc.positions()) == to_c(&back, back.doc.positions()), "composite differs");
    let mv = |m: &MeshLayers| -> HashMap<u64, u32> {
        let c = m.internal_to_canonical();
        m.doc.layer(id).unwrap().mask_values().unwrap().iter().enumerate().map(|(i, x)| (m.keys()[c[i] as usize], x.to_bits())).collect()
    };
    assert!(mv(&ml) == mv(&back), "mask values differ");
    eprintln!("G6: {} verts, two layers (mask + 0.7, Subtract 1.3) bit-equal after reopening", back.doc.vertex_count());
}

/// `vol` as a file holds it (its samples at 16 bits), opened with `levels` levels.
fn filed(vol: &LevelVolume, levels: usize) -> LevelVolume {
    let mut bytes = Vec::new();
    vol.write_to(&mut bytes).unwrap();
    LevelVolume::read_from(&mut std::io::Cursor::new(&bytes[..]), levels).unwrap()
}

/// The volume written to a file and read back, then the layers loaded onto it: every name is found and every offset
/// comes back bit for bit. So does every vertex when `vol` came from a file itself (`opened`); a volume never saved
/// holds its samples finer than the file does, and going through the file moves a few of its vertices by a
/// thousandth of a step.
fn g6b(vol: &LevelVolume, open_levels: usize, opened: bool, what: &str) {
    let mut ml = MeshLayers::from_volume(vol, MeshLayerSettings::default()).unwrap();
    let step = ml.step();
    let id = fill_layer(&mut ml.doc, "noise", |doc, v| doc.normals()[v].normalize_or_zero() * (0.5 * step * noise(doc.base_positions()[v])));
    let dir = std::env::temp_dir().join(format!("mesh_layers_g6b_{}_{}", what.replace(' ', "_"), std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ml.save(&dir).unwrap();
    let vol2 = filed(vol, open_levels);
    let (back, lost) = MeshLayers::load(&dir, &vol2).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (a, b) = (by_key(&ml, id), by_key(&back, id));
    let missing = a.keys().filter(|k| !b.contains_key(k)).count();
    let differ = a.iter().filter(|(k, (p, d))| b.get(k).is_some_and(|(p2, d2)| bits(&[*p, *d]) != bits(&[*p2, *d2]))).count();
    let delta_differ = a.iter().filter(|(k, (_, d))| b.get(k).is_some_and(|(_, d2)| bits(&[*d]) != bits(&[*d2]))).count();
    let far = a.iter().filter_map(|(k, (p, _))| b.get(k).map(|(p2, _)| p.distance(*p2))).fold(0.0f32, f32::max);
    eprintln!(
        "G6b {what}: {} levels saved, {} opened; {} vertices, {} after; lost {lost}, names missing {missing}, vertices differing {differ} (offsets {delta_differ}; base moved at most {:.5} step)",
        vol.levels(),
        vol2.levels(),
        a.len(),
        b.len(),
        far / step
    );
    assert_eq!(lost, 0, "{what}: names lost");
    assert_eq!(a.len(), b.len(), "{what}: vertex count");
    assert_eq!(missing, 0, "{what}: names missing");
    assert_eq!(delta_differ, 0, "{what}: offsets differ");
    if opened {
        assert_eq!(differ, 0, "{what}: vertices differ");
    } else {
        assert!(far <= 0.01 * step, "{what}: the file moved a vertex {:.5} step", far / step);
    }
}

#[test]
fn g6b_layers_survive_the_volume_file() {
    let mut vol = ball(1.0 / 32.0, 2, R);
    vol_stroke(&mut vol, Vec3::new(0.3, 0.2, 1.0), 0.15);
    vol_stroke(&mut vol, Vec3::new(-1.0, 0.2, 0.1), 0.1);
    g6b(&vol, 2, false, "same levels, never saved");
    g6b(&filed(&vol, 2), 2, true, "same levels, opened");
}

#[test]
fn g6b_layers_survive_a_file_opened_with_more_levels() {
    let mut vol = ball(1.0 / 32.0, 4, R);
    vol_stroke(&mut vol, Vec3::new(0.3, 0.2, 1.0), 0.1);
    let l = vol.levels() - 2;
    vol_stroke_on(&mut vol, Vec3::new(-R, 0.0, 0.0), -Vec3::X, 0.08, l, 0.008, 0.05, 1);
    g6b(&vol, 6, false, "four as six, never saved");
    g6b(&filed(&vol, 4), 6, true, "four as six, opened");
}

// ------------------------------------------------------------------------------------------------------------- G7

/// What the noise layer holds at a base point of the ball.
fn ball_noise(p: Vec3, step: f32) -> Vec3 {
    p.normalize() * (0.5 * step * noise(p))
}

#[test]
fn g7_noise_layer_through_50_volume_strokes() {
    let (mut vol, mut ml) = small();
    let step = ml.step();
    let id = fill_layer(&mut ml.doc, "noise", |doc, v| ball_noise(doc.base_positions()[v], step));
    let before = by_key(&ml, id);
    let mut dabs: Vec<Dabs> = Vec::new();
    let (mut copied, mut projected) = (0, 0);
    for i in 0..50 {
        // All over the ball.
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 50.0;
        let a = i as f32 * 2.399963;
        let c = Vec3::new(a.cos() * (1.0 - z * z).sqrt(), a.sin() * (1.0 - z * z).sqrt(), z);
        let d = vol_stroke_dabs(&mut vol, c, 0.08);
        let st = ml.refit(&vol, d.bounds()).unwrap();
        copied += st.copied;
        projected += st.projected;
        dabs.push(d);
    }
    let after = by_key(&ml, id);
    // Where the base did not change, the layer is what it was.
    let (mut same_base, mut mismatch) = (0usize, 0usize);
    for (k, (p, d)) in &after {
        let Some((p0, d0)) = before.get(k) else { continue };
        if bits(&[*p]) == bits(&[*p0]) {
            same_base += 1;
            mismatch += (bits(&[*d]) != bits(&[*d0])) as usize;
        }
    }
    // Outside every dab, the layer is the noise.
    let (mut n, mut err2, mut ref2, mut worst) = (0usize, 0.0f64, 0.0f64, 0.0f32);
    for (p, d) in after.values() {
        if !dabs.iter().all(|s| s.outside(*p, 2.0 * step)) {
            continue;
        }
        let want = ball_noise(*p, step);
        n += 1;
        err2 += (*d - want).length_squared() as f64;
        ref2 += want.length_squared() as f64;
        worst = worst.max((*d - want).length());
    }
    let rms = (err2 / ref2).sqrt();
    eprintln!(
        "G7: 50 strokes, copied {copied}, projected {projected}; {same_base} of {} vertices keep their base, {mismatch} of them with another delta; {n} outside the dabs + 2 steps: RMS error {:.3}% of the noise, worst {:.3} step",
        after.len(),
        rms * 100.0,
        worst / step
    );
    assert!(same_base > 1000 && n > 1000);
    assert_eq!(mismatch, 0, "vertices whose base did not change lost their delta");
    assert!(rms <= 0.01, "RMS error outside the dabs {:.3}%", rms * 100.0);
}

// ------------------------------------------------------------------------------------------------------------- G8

#[test]
#[ignore]
fn g8_coarse_layer_beside_a_finest_level_stroke() {
    let Ok(path) = std::env::var("SCULPT_ARTIST_FILE") else {
        eprintln!("SCULPT_ARTIST_FILE not set; skipped");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let mut vol = LevelVolume::read_from(&mut std::io::Cursor::new(&bytes[..]), 6).unwrap();
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let coarse = vol.voxel(2);
    let fine_l = vol.levels() - 1;
    let fine = vol.voxel(fine_l);
    eprintln!("G8: {} levels, blocks {:?}; {} verts", vol.levels(), (0..vol.levels()).map(|l| vol.level_len(l)).collect::<Vec<_>>(), ml.doc.vertex_count());
    // A layer on the vertices of level 0-2 cells: half a level 2 step along the normal.
    let to_c = ml.internal_to_canonical();
    let level: Vec<u8> = (0..ml.doc.vertex_count()).map(|v| (ml.keys()[to_c[v] as usize] >> 61) as u8).collect();
    let id = fill_layer(&mut ml.doc, "coarse", |doc, v| if level[v] <= 2 { doc.normals()[v].normalize_or_zero() * (0.5 * coarse) } else { Vec3::ZERO });
    // A place whose surface is level 0-2 for six level 2 steps around: of those, where the surface turns most (there
    // a coarse cell's surface and the finer cells' made over it lie farthest apart).
    let base = ml.doc.base_positions();
    let normals = ml.doc.normals();
    let coarse_verts: Vec<usize> = (0..base.len()).filter(|&v| level[v] <= 2).collect();
    let spot = coarse_verts
        .iter()
        .step_by((coarse_verts.len() / 400).max(1))
        .copied()
        .filter(|&v| (0..base.len()).all(|u| level[u] <= 2 || base[u].distance(base[v]) > 6.0 * coarse))
        .map(|v| {
            let n = normals[v].normalize_or_zero();
            let turn = (0..base.len()).filter(|&u| base[u].distance(base[v]) < 3.0 * coarse).map(|u| normals[u].normalize_or_zero().dot(n)).fold(1.0f32, f32::min);
            (v, turn)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("no coarse region in the file")
        .0;
    let (at, n) = (base[spot], ml.doc.normals()[spot].normalize());
    let r = 24.0 * fine;
    // Distances from the dabs across the surface (the stroke raises what is under it out of their balls).
    let across = |c: Vec3, d: &Dabs| d.hits.iter().map(|h| (c - h).reject_from_normalized(n).length()).fold(f32::INFINITY, f32::min);
    let near = |c: Vec3, d: &Dabs| (c - at).dot(n).abs() < 4.0 * coarse && across(c, d) > d.r && across(c, d) <= d.r + 4.0 * coarse;
    let outside = |c: Vec3, d: &Dabs| (c - at).dot(n).abs() > 4.0 * coarse || across(c, d) > d.r;
    vol.drain_dirty();
    let dabs = vol_stroke_on(&mut vol, at, n, r, fine_l, r * 0.095, r, 1);
    // The layer's mean size over the surface: beside the stroke, and everywhere outside it.
    let mean = |ml: &MeshLayers, keep: &dyn Fn(Vec3) -> bool| {
        let (sum, area) = delta_integral(ml, id, keep);
        sum / area
    };
    let sum0 = mean(&ml, &|c| near(c, &dabs));
    let all0 = mean(&ml, &|c| outside(c, &dabs));
    let dirty = vol.drain_dirty();
    let st = ml.refit(&vol, MeshLayers::changed_box(&vol, &dirty)).unwrap();
    let sum1 = mean(&ml, &|c| near(c, &dabs));
    let all1 = mean(&ml, &|c| outside(c, &dabs));
    eprintln!(
        "G8: a level {fine_l} stroke (radius {:.2} level 2 steps) at {at:?}; blocks now {:?}; copied {}, projected {}, searched {} faces, carry {:.1} ms; mean |delta| over the surface outside the dabs: within 4 level 2 steps {sum0:.6e} -> {sum1:.6e} ({:.3}%), everywhere {all0:.6e} -> {all1:.6e} ({:.4}%)",
        r / coarse,
        (0..vol.levels()).map(|l| vol.level_len(l)).collect::<Vec<_>>(),
        st.copied,
        st.projected,
        st.search_faces,
        st.carry_ms,
        sum1 / sum0 * 100.0,
        all1 / all0 * 100.0
    );
    assert!(st.projected > 0 && sum0 > 0.0);
    assert!(sum1 / sum0 >= 0.99, "kept {:.3}% beside the stroke", sum1 / sum0 * 100.0);
    assert!(all1 / all0 >= 0.99, "kept {:.3}% outside the stroke", all1 / all0 * 100.0);
}

// ------------------------------------------------------------------------------------------------------------ G10

/// A clay stroke five steps tall over a noise layer: under it the layer rides up with the surface, and where carried
/// vertices meet copied ones the layer does not jump.
#[test]
fn g10_noise_layer_rides_a_tall_volume_stroke() {
    let (mut vol, mut ml) = small();
    let step = ml.step();
    let id = fill_layer(&mut ml.doc, "noise", |doc, v| ball_noise(doc.base_positions()[v], step));
    let before = by_key(&ml, id);
    let top = Vec3::new(0.0, 0.0, R);
    let dabs = vol_stroke_on(&mut vol, top, Vec3::Z, 0.1, 0, 0.08, 0.06, 12);
    let height = vol.raycast(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z, 0, false).unwrap().0.z - R;
    let st = ml.refit(&vol, dabs.bounds()).unwrap();
    let d = ml.layer_deltas(id).unwrap();
    let to_c = ml.internal_to_canonical();
    let key = |v: usize| ml.keys()[to_c[v] as usize];
    let base = ml.doc.base_positions();
    let copied = |v: usize| before.get(&key(v)).is_some_and(|(p0, _)| bits(&[*p0]) == bits(&[base[v]]));
    // What the layer should hold at a vertex: the noise at the point of the ball under it, along the surface's
    // normal now.
    let want = |v: usize| -> Vec3 {
        let foot = base[v].normalize() * R;
        vol.normal_upto(base[v], 0, false) * (0.5 * step * noise(foot))
    };
    // Under the stroke: the height kept, signed by what it should be.
    let (mut got, mut all, mut under) = (0.0f64, 0.0f64, 0usize);
    for v in 0..base.len() {
        if copied(v) || base[v].length() - R < 1.0 * step {
            continue;
        }
        let w = want(v);
        under += 1;
        all += w.length() as f64;
        got += (d[v].dot(w) / w.length().max(1e-12)) as f64;
    }
    // Across the edge of the carried part: the error of carried vertices next to copied ones.
    let (mut jump, mut edges) = (0.0f32, 0usize);
    for f in ml.doc.faces() {
        let vs = sculpt_core::mesh::face_verts(f);
        for k in 0..vs.len() {
            let (a, b) = (vs[k] as usize, vs[(k + 1) % vs.len()] as usize);
            for (a, b) in [(a, b), (b, a)] {
                if copied(a) && !copied(b) {
                    edges += 1;
                    jump = jump.max(((d[b] - want(b)) - (d[a] - before[&key(a)].1)).length());
                }
            }
        }
    }
    eprintln!(
        "G10: the stroke raised the top {:.2} steps; copied {}, projected {} ({} found no old surface); {under} vertices raised a step or more keep {:.2}% of the layer's height; {edges} edges between copied and carried vertices, worst jump {:.4} step",
        height / step,
        st.copied,
        st.projected,
        st.unplaced,
        got / all * 100.0,
        jump / step
    );
    assert!(height >= 4.5 * step, "the stroke raised only {:.2} steps", height / step);
    assert!(under > 100 && edges > 20);
    assert!(got / all >= 0.95, "kept {:.2}% of the height under the stroke", got / all * 100.0);
    assert!(jump <= 0.1 * step, "jump {:.4} step at the edge of the carried part", jump / step);
}

// ------------------------------------------------------------------------------------------------------------- G9

/// A volume stroke undone: the surface is what it was, and with the stroke's patch put back so is every layer, bit
/// for bit.
fn g9(mut vol: LevelVolume, what: &str) {
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let step = ml.step();
    let id = fill_layer(&mut ml.doc, "noise", |doc, v| ball_noise(doc.base_positions()[v], step));
    let id2 = column(&mut ml.doc, 3, 0.12);
    let half: Vec<f32> = ml.doc.base_positions().iter().map(|p| (p.x * 10.0).clamp(0.0, 1.0)).collect();
    ml.doc.set_channel("paint.half", half).unwrap();
    let channel = |m: &MeshLayers| -> HashMap<u64, u32> {
        let c = m.internal_to_canonical();
        m.doc.channel("paint.half").unwrap().iter().enumerate().map(|(i, x)| (m.keys()[c[i] as usize], x.to_bits())).collect()
    };
    let composite = |m: &MeshLayers| -> HashMap<u64, [u32; 3]> {
        let c = m.internal_to_canonical();
        m.doc.positions().iter().enumerate().map(|(i, p)| (m.keys()[c[i] as usize], [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])).collect()
    };
    // (Composited afresh: strokes add up dab by dab, to rounding.)
    ml.doc.set_layer_opacity(id2, 0.5).unwrap();
    ml.doc.set_layer_opacity(id2, 1.0).unwrap();
    let before = [by_key(&ml, id), by_key(&ml, id2)];
    let (channel0, composite0) = (channel(&ml), composite(&ml));
    // A stroke over the side of the column, where both layers and the channel's edge lie.
    vol.drain_dirty();
    let stroke = vol_stroke(&mut vol, Vec3::new(0.25, 0.1, 1.0), 0.1);
    let (st, patch) = ml.refit_patch(&vol, stroke).unwrap();
    assert!(vol.undo(), "nothing to undo");
    let dirty = vol.drain_dirty();
    let st2 = ml.refit(&vol, MeshLayers::changed_box(&vol, &dirty)).unwrap();
    let differing = |ml: &MeshLayers| -> (usize, usize, usize) {
        let (mut names, mut base, mut delta) = (0, 0, 0);
        for (was, lid) in before.iter().zip([id, id2]) {
            let now = by_key(ml, lid);
            names += was.keys().filter(|k| !now.contains_key(k)).count() + now.keys().filter(|k| !was.contains_key(k)).count();
            for (k, (p, d)) in was {
                let Some((p2, d2)) = now.get(k) else { continue };
                base += (bits(&[*p]) != bits(&[*p2])) as usize;
                delta += (bits(&[*d]) != bits(&[*d2])) as usize;
            }
        }
        (names, base, delta)
    };
    let unrestored = differing(&ml);
    let back = ml.restore(&patch).unwrap();
    let restored = differing(&ml);
    let channel_diff = channel(&ml).iter().filter(|(k, x)| channel0.get(k) != Some(x)).count();
    let composite_diff = composite(&ml).iter().filter(|(k, x)| composite0.get(k) != Some(x)).count();
    eprintln!(
        "G9 {what}: the stroke carried {} vertices (patch of {}), the undo {}; after the undo (names, bases, offsets) differing {unrestored:?}; {back} put back: {restored:?}, channel values {channel_diff}, shown positions {composite_diff}",
        st.projected,
        patch.len(),
        st2.projected
    );
    assert!(st.projected > 0 && patch.len() > 0);
    assert_eq!((restored.0, restored.1), (0, 0), "{what}: the surface after the undo is not the surface before the stroke");
    assert_eq!(back, patch.len(), "{what}: patch entries with no vertex to go back to");
    assert_eq!(restored.2, 0, "{what}: offsets differ after the patch went back");
    assert_eq!(channel_diff, 0, "{what}: channel values differ");
    assert_eq!(composite_diff, 0, "{what}: shown positions differ");
}

#[test]
fn g9_volume_undo_brings_the_layers_back() {
    g9(ball(1.0 / 64.0, 1, R), "one level");
    g9(ball(1.0 / 64.0, 2, R), "a finer stroke");
}

//! Mesh layers over a volume: layers live on the volume's extracted display surface, and survive volume edits by
//! vertex name. Gates G0 (exact layer ops), G2 (identity carry), G3 (carry speed, ignored), G4 (retention), G5
//! (extraction quality; the artist file part ignored), G6 (save and reopen).

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::geom::{Aabb, Ray};
use sculpt_core::mask::{MaskLayer, MaskSource, MaskStack};
use sculpt_core::mesh::face_len;
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers, edge_defects, extract_named};
use sculpt_core::{Document, LayerBlend, LayerId};

const R: f32 = 0.5;

fn ball(voxel0: f32, levels: usize, r: f32) -> LevelVolume {
    LevelVolume::from_field(&|p: Vec3| p.length() - r, r + 0.3, voxel0, levels)
}

fn small() -> (LevelVolume, MeshLayers) {
    let vol = ball(1.0 / 64.0, 1, R);
    let ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    (vol, ml)
}

/// A clay column on top of the ball (+z), as the spike builds it: back-and-forth strokes over the top.
fn column(doc: &mut Document, strokes: usize, radius: f32) -> LayerId {
    let id = doc.add_layer("clay");
    let s = BrushSettings { radius, strength: 0.5, ..Default::default() };
    let mut tall = ClayBuildup { height: 0.32, ..Default::default() };
    for _ in 0..strokes {
        let mut path = Vec::new();
        for k in 0..=16 {
            let t = k as f32 / 16.0;
            let x = -0.05 + 0.1 * if k <= 8 { t * 2.0 } else { 2.0 - t * 2.0 };
            if let Some(h) = doc.raycast(&Ray::new(Vec3::new(x, 0.0, 3.0), -Vec3::Z)) {
                path.push((h.point, 1.0));
            }
        }
        brush::stroke(doc, &mut tall, &s, &path).unwrap();
    }
    id
}

/// A volume clay stroke at the finest level around direction `c` (a few dabs along a short arc); returns its box
/// (each dab's hit point grown by the radius).
fn vol_stroke(vol: &mut LevelVolume, c: Vec3, r: f32) -> Aabb {
    let c = c.normalize();
    let side = c.any_orthonormal_vector();
    let l = vol.levels() - 1;
    let mut b = Aabb::EMPTY;
    let mut along = Vec3::ZERO;
    vol.begin_stroke();
    for k in 0..=8 {
        let dir = (c + side * (0.15 * (k as f32 / 8.0 - 0.5))).normalize();
        let Some((hit, n)) = vol.raycast(dir * 3.0, -dir, l, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = vol.plane_dab(l, hit, rough, r, r * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        b.grow(hit - Vec3::splat(r));
        b.grow(hit + Vec3::splat(r));
    }
    vol.end_stroke();
    b
}

fn grown(b: &Aabb, by: f32) -> Aabb {
    Aabb { min: b.min - Vec3::splat(by), max: b.max + Vec3::splat(by) }
}

fn inside(b: &Aabb, p: Vec3) -> bool {
    p.cmpge(b.min).all() && p.cmple(b.max).all()
}

/// (sum, max) of |delta| of layer `id`.
fn delta_stats(ml: &MeshLayers, id: LayerId) -> (f64, f32) {
    let d = ml.layer_deltas(id).unwrap();
    (d.iter().map(|v| v.length() as f64).sum(), d.iter().map(|v| v.length()).fold(0.0, f32::max))
}

/// Per vertex name: (base, delta of `id`).
fn by_key(ml: &MeshLayers, id: LayerId) -> HashMap<u64, (Vec3, Vec3)> {
    let d = ml.layer_deltas(id).unwrap();
    let to_c = ml.internal_to_canonical();
    (0..ml.doc.vertex_count()).map(|v| (ml.keys()[to_c[v] as usize], (ml.doc.base_positions()[v], d[v]))).collect()
}

fn bits(a: &[Vec3]) -> Vec<[u32; 3]> {
    a.iter().map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect()
}

fn same(a: &[Vec3], b: &[Vec3]) -> bool {
    bits(a) == bits(b)
}

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
    Document::from_mesh(mesh).unwrap();
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
    let (mesh, _keys, _) = extract_named(&vol, 8).unwrap();
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
    let ratio = tris as f64 / shown_tris as f64;
    assert!((0.5..=2.0).contains(&ratio));
    assert_eq!(bad, 0, "non-manifold edges on the artist's surface");
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

//! Mesh layers, the API the app builds on: vertex places in the GPU's blocks (C0a), the blocks a swap changed (C0b),
//! layer and mask strokes as name-keyed patches (C0c), and the stream file form (C0e). (C0d, the swap frame with Max
//! layers and noise masks, is in mesh_layers_async.rs.)

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup, Falloff};
use sculpt_core::generator::level_volume::{Key, LevelVolume};
use sculpt_core::geom::Ray;
use sculpt_core::mask::{BlendMode, MaskLayer, MaskSource, MaskStack};
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers, SwapStats, extract_all};
use sculpt_core::noise::NoiseParams;
use sculpt_core::{Document, LayerBlend, LayerId, PaintTarget};

mod common;
use common::*;

/// A clay stroke on the active layer around direction `dir`.
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

/// A paint stroke on channel `name` around direction `dir`.
fn paint_dab(doc: &mut Document, name: &str, dir: Vec3, radius: f32, value: f32) {
    doc.begin_stroke("Paint");
    let side = dir.any_orthonormal_vector();
    for k in 0..=6 {
        let d = (dir + side * (0.1 * (k as f32 / 6.0 - 0.5))).normalize();
        if let Some(h) = doc.raycast(&Ray::new(d * 3.0, -d)) {
            doc.paint(&PaintTarget::Channel(name.into()), h.point, radius, &Falloff::default(), 0.8, value, None);
        }
    }
    doc.end_stroke();
}

/// Every layer composited afresh (strokes add up dab by dab, to rounding).
fn fresh(doc: &mut Document) {
    let ids: Vec<(LayerId, f32)> = doc.layers().iter().map(|l| (l.id, l.opacity)).collect();
    for (id, o) in ids {
        doc.set_layer_opacity(id, o * 0.5 + 0.25).unwrap();
        doc.set_layer_opacity(id, o).unwrap();
    }
}

/// Per vertex name, every value the layers hold or show there: base, each layer's offset and mask value, each channel,
/// freeze, composite.
fn state(ml: &MeshLayers) -> HashMap<u64, Vec<u32>> {
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

/// (names only in one, names in both whose values differ), over the names `keep` takes.
fn differ(a: &HashMap<u64, Vec<u32>>, b: &HashMap<u64, Vec<u32>>, keep: &dyn Fn(u64) -> bool) -> (usize, usize) {
    let only = a.keys().filter(|k| keep(**k) && !b.contains_key(k)).count() + b.keys().filter(|k| keep(**k) && !a.contains_key(k)).count();
    let diff = a.iter().filter(|(k, x)| keep(**k) && b.get(k).is_some_and(|y| y != *x)).count();
    (only, diff)
}

// ------------------------------------------------------------------------------------------------------------ C0a

/// Every vertex's (block, vi) is where the block surfaces put it, as the GPU draws it: built with the mesh, and
/// again by the refit job.
fn check_places(ml: &MeshLayers, vol: &LevelVolume, what: &str) {
    let cache = ml.block_cache().expect("no block cache");
    let blocks = ml.blocks();
    assert_eq!(blocks.len(), cache.len(), "{what}: blocks listed");
    let vi = ml.vertex_vi();
    let to_c = ml.internal_to_canonical();
    let c2i: Vec<usize> = (0..ml.doc.vertex_count()).map(|c| ml.doc.canonical_index(c)).collect();
    assert!(c2i.iter().enumerate().all(|(c, &v)| to_c[v] as usize == c), "{what}: internal -> canonical");
    let base = ml.doc.base_positions();
    let mut seen = 0usize;
    for (i, (key, _)) in blocks.iter().enumerate() {
        let surf = cache.block(key).unwrap_or_else(|| panic!("{what}: block {key:?} not held"));
        let r = ml.block_vertices(i);
        let mut last = None;
        for c in r.clone() {
            let name = ml.keys()[c];
            let (cell, piece) = (((name >> 4) & 511) as usize, (name & 15) as u32);
            let info = surf.cells[cell];
            let count = (info >> 10) & 7;
            // As vs_levels finds the vertex of a corner (cell, piece).
            let gpu_vi = (info & 1023) + piece.min(count - 1);
            assert!(piece < count, "{what}: piece {piece} of a cell with {count}");
            assert_eq!(vi[c] as u32, gpu_vi, "{what}: block {key:?} vertex {c}");
            assert_eq!(surf.vertex(cell as u16, piece as u8), Some(vi[c]), "{what}: BlockSurface::vertex");
            let p = surf.verts[vi[c] as usize];
            assert!(same(&[p], &[base[c2i[c]]]), "{what}: block {key:?} vertex {c}: position");
            assert!(last.is_none_or(|l| l < vi[c]), "{what}: vi not ascending in block {key:?}");
            assert_eq!(ml.block_of(c), i, "{what}: block_of");
            last = Some(vi[c]);
            seen += 1;
        }
    }
    assert_eq!(seen, ml.doc.vertex_count(), "{what}: vertices placed");
    // Base normals: the fresh base mesh's.
    let fresh_doc = Document::from_mesh(extract_all(vol, usize::MAX).unwrap().mesh).unwrap();
    let want: Vec<Vec3> = (0..ml.doc.vertex_count()).map(|c| fresh_doc.normals()[fresh_doc.canonical_index(c)]).collect();
    let got: Vec<Vec3> = c2i.iter().map(|&v| ml.base_normals()[v]).collect();
    assert!(same(&got, &want), "{what}: base normals");
    eprintln!("C0a {what}: {} blocks, {seen} vertices placed by (block, vi) as the GPU draws them; base normals bit-equal", blocks.len());
}

#[test]
fn c0a_vertex_places_match_the_block_surfaces() {
    // Two levels with a finer stroke: transition cells, cells with several pieces, blocks with dropped vertices.
    let mut vol = ball(1.0 / 32.0, 2, R);
    vol_stroke(&mut vol, Vec3::new(0.3, 0.2, 1.0), 0.15);
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    check_places(&ml, &vol, "built");
    column(&mut ml.doc, 2, 0.12);
    vol.drain_dirty();
    for (i, c) in [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.8, 0.3, 0.4)].into_iter().enumerate() {
        let stroke = vol_stroke(&mut vol, c, 0.12);
        let dirty = vol.drain_dirty();
        ml.refit_cached(&vol, &dirty, stroke).unwrap();
        check_places(&ml, &vol, &format!("after swap {i}"));
    }
    let stroke = vol_stroke(&mut vol, Vec3::new(0.0, -1.0, 0.0), 0.1);
    ml.refit(&vol, stroke).unwrap();
    // (The plain refit keeps no cache: compare with a fresh one.)
    let fresh_ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    assert_eq!(ml.blocks(), fresh_ml.blocks(), "plain refit: blocks");
    assert_eq!(ml.vertex_vi(), fresh_ml.vertex_vi(), "plain refit: vi");
    assert!(same(ml.base_normals(), fresh_ml.base_normals()), "plain refit: base normals");
}

// ------------------------------------------------------------------------------------------------------------ C0b

/// Per (block, vi): the base, the base normal, the composite's offset from the base and the composite normal.
fn rows(ml: &MeshLayers) -> HashMap<(Key, u16), [u32; 12]> {
    let doc = &ml.doc;
    let b = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    let mut out = HashMap::new();
    for (i, (key, _)) in ml.blocks().iter().enumerate() {
        for c in ml.block_vertices(i) {
            let v = doc.canonical_index(c);
            let mut row = [0u32; 12];
            row[0..3].copy_from_slice(&b(doc.base_positions()[v]));
            row[3..6].copy_from_slice(&b(ml.base_normals()[v]));
            row[6..9].copy_from_slice(&b(doc.positions()[v] - doc.base_positions()[v]));
            row[9..12].copy_from_slice(&b(doc.normals()[v]));
            out.insert((*key, ml.vertex_vi()[c]), row);
        }
    }
    out
}

#[test]
fn c0b_swap_lists_every_block_whose_rows_changed() {
    let mut vol = ball(1.0 / 64.0, 2, R);
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let step = ml.step();
    // An Add column with a paint and a noise mask, and a Max noise layer all over.
    let id = column(&mut ml.doc, 3, 0.12);
    let half: Vec<f32> = ml.doc.base_positions().iter().map(|p| (p.x * 10.0).clamp(0.0, 1.0)).collect();
    ml.doc.set_channel("paint.half", half).unwrap();
    let stack = MaskStack::new(1.0)
        .with(MaskLayer::new("paint", MaskSource::Channel { name: "paint.half".into() }).blend(BlendMode::Multiply))
        .with(MaskLayer::new("noise", MaskSource::Noise(NoiseParams::default())).blend(BlendMode::Multiply));
    ml.doc.set_layer_mask(id, Some(stack)).unwrap();
    let max = fill_layer(&mut ml.doc, "noise", |doc, v| doc.normals()[v].normalize_or_zero() * (0.5 * step * noise(doc.base_positions()[v])));
    let mut m = ml.doc.layer(max).unwrap().meta();
    m.blend = LayerBlend::Max;
    ml.doc.set_layer_meta(max, m, false).unwrap();
    vol.drain_dirty();
    let (mut listed, mut differing, mut total) = (0usize, 0usize, 0usize);
    for i in 0..6 {
        let c = [Vec3::new(0.2, 0.1, -1.0), Vec3::new(-0.8, 0.3, -0.4), Vec3::new(0.25, 0.1, 1.0), Vec3::new(0.9, -0.3, 0.2), Vec3::new(0.0, 0.8, -0.6), Vec3::new(0.25, 0.1, 1.0)][i];
        // Every other stroke at the finer level (new blocks), the rest at the coarser one.
        let l = if i % 2 == 0 { 1 } else { 0 };
        let dabs = vol_stroke_on(&mut vol, c.normalize() * R, c, 0.1, l, 0.01, 0.06, 1);
        let dirty = vol.drain_dirty();
        let before = rows(&ml);
        let old_blocks: HashSet<Key> = ml.blocks().iter().map(|b| b.0).collect();
        ml.refit_async(&vol, &dirty, dabs.bounds());
        let st: SwapStats = loop {
            if let Some(r) = ml.poll(&vol) {
                break r.unwrap();
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(!st.discarded);
        let after = rows(&ml);
        let new_blocks: HashSet<Key> = ml.blocks().iter().map(|b| b.0).collect();
        let changed: HashSet<Key> = st.blocks.changed.iter().copied().collect();
        let removed: HashSet<Key> = st.blocks.removed.iter().copied().collect();
        assert!(changed.is_subset(&new_blocks), "swap {i}: a changed block is not in the mesh");
        assert_eq!(removed, &old_blocks - &new_blocks, "swap {i}: removed blocks");
        assert!(new_blocks.iter().filter(|k| !old_blocks.contains(k)).all(|k| changed.contains(k)), "swap {i}: a new block is not listed");
        assert!(dirty.iter().all(|k| changed.contains(k) || removed.contains(k) || !new_blocks.contains(k)), "swap {i}: a changed volume block is not listed");
        // Outside the listed blocks every row is what it was, and no vertex came or went.
        let mut wrong: Vec<String> = Vec::new();
        for (k, row) in &after {
            if changed.contains(&k.0) {
                continue;
            }
            match before.get(k) {
                Some(r) if r == row => {}
                Some(r) => wrong.push(format!("{k:?}: {r:?} -> {row:?}")),
                None => wrong.push(format!("{k:?} is new")),
            }
        }
        for k in before.keys().filter(|k| new_blocks.contains(&k.0) && !changed.contains(&k.0)) {
            if !after.contains_key(k) {
                wrong.push(format!("{k:?} is gone"));
            }
        }
        let actually: HashSet<Key> = after.iter().filter(|(k, r)| before.get(k) != Some(r)).map(|(k, _)| k.0).collect();
        eprintln!(
            "C0b swap {i} (level {l}): {:?}; {} blocks, {} listed, {} with a row that differs; {} rows wrong outside the listed blocks",
            st.blocks,
            new_blocks.len(),
            changed.len(),
            actually.len(),
            wrong.len()
        );
        assert!(wrong.is_empty(), "swap {i}: {} rows changed outside the listed blocks, e.g. {:?}", wrong.len(), &wrong[..wrong.len().min(3)]);
        assert!(!changed.is_empty() && changed.len() * 4 < new_blocks.len(), "swap {i}: {} of {} listed", changed.len(), new_blocks.len());
        listed += changed.len();
        differing += actually.len();
        total += new_blocks.len();
    }
    eprintln!("C0b: 6 swaps, {listed} blocks listed of {total}, {differing} with a row that differs");
}

// ------------------------------------------------------------------------------------------------------------ C0c

/// A scene with a masked clay layer: (volume, layers, layer id).
fn c0c_scene() -> (LevelVolume, MeshLayers, LayerId) {
    let vol = ball(1.0 / 64.0, 2, R);
    let mut ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    let id = column(&mut ml.doc, 2, 0.12);
    let half: Vec<f32> = ml.doc.base_positions().iter().map(|p| (p.x * 10.0).clamp(0.0, 1.0)).collect();
    ml.doc.set_channel("paint.half", half).unwrap();
    let stack = MaskStack::new(1.0)
        .with(MaskLayer::new("paint", MaskSource::Channel { name: "paint.half".into() }).blend(BlendMode::Multiply))
        .with(MaskLayer::new("noise", MaskSource::Noise(NoiseParams::default())).blend(BlendMode::Multiply).opacity(0.5));
    ml.doc.set_layer_mask(id, Some(stack)).unwrap();
    fresh(&mut ml.doc);
    (vol, ml, id)
}

/// A layer stroke and a mask stroke, each as a patch: (state before, after the layer stroke, after both; the patches).
fn c0c_strokes(ml: &mut MeshLayers, id: LayerId) -> ([HashMap<u64, Vec<u32>>; 3], [sculpt_core::mesh_layers::Patch; 2]) {
    let s0 = state(ml);
    ml.doc.set_active_layer(Some(id)).unwrap();
    layer_dab(&mut ml.doc, Vec3::new(0.15, 0.05, 1.0), 0.1);
    let p1 = ml.stroke_patch().unwrap();
    fresh(&mut ml.doc);
    let s1 = state(ml);
    paint_dab(&mut ml.doc, "paint.half", Vec3::new(0.05, 0.0, 1.0), 0.1, 0.0);
    let p2 = ml.stroke_patch().unwrap();
    fresh(&mut ml.doc);
    let s2 = state(ml);
    assert!(!p1.is_empty() && !p2.is_empty());
    let all = |_: u64| true;
    assert!(differ(&s0, &s1, &all).1 > 0 && differ(&s1, &s2, &all).1 > 0, "a stroke changed nothing");
    ([s0, s1, s2], [p1, p2])
}

/// Undo both strokes by their patches, then redo them by the inverses, checking each state over the names `keep` takes.
fn c0c_round_trip(ml: &mut MeshLayers, s: &[HashMap<u64, Vec<u32>>; 3], p: &[sculpt_core::mesh_layers::Patch; 2], keep: &dyn Fn(u64) -> bool, what: &str) {
    let i2 = ml.restore_swap(&p[1]).unwrap();
    let a = differ(&state(ml), &s[1], keep);
    let i1 = ml.restore_swap(&p[0]).unwrap();
    let b = differ(&state(ml), &s[0], keep);
    let r1 = ml.restore_swap(&i1).unwrap();
    let c = differ(&state(ml), &s[1], keep);
    ml.restore_swap(&i2).unwrap();
    let d = differ(&state(ml), &s[2], keep);
    eprintln!(
        "C0c {what}: patches of {} and {} vertices, inverses {} and {} (redo's inverse {}); (names missing, vertices differing) undo mask {a:?}, undo layer {b:?}, redo layer {c:?}, redo mask {d:?}",
        p[0].len(),
        p[1].len(),
        i1.len(),
        i2.len(),
        r1.len()
    );
    assert_eq!((i1.len(), i2.len()), (p[0].len(), p[1].len()), "{what}: entries with no vertex to go back to");
    for (x, at) in [(a, "undo mask"), (b, "undo layer"), (c, "redo layer"), (d, "redo mask")] {
        assert_eq!(x, (0, 0), "{what}: {at}");
    }
}

#[test]
fn c0c_stroke_patches_undo_and_redo_bit_equal() {
    let (_vol, mut ml, id) = c0c_scene();
    let (s, p) = c0c_strokes(&mut ml, id);
    c0c_round_trip(&mut ml, &s, &p, &|_| true, "no swap");
}

/// Across a refit swap: the strokes, then a volume stroke elsewhere refitted (a new document: its own undo is gone).
/// Undone the way the app does it (the volume step first: undo, refit, its patch back) every value comes back bit
/// for bit; with the volume stroke left in place, every name its stroke did not reach does.
#[test]
fn c0c_stroke_patches_hold_across_a_refit_swap() {
    for undo_volume in [true, false] {
        let (mut vol, mut ml, id) = c0c_scene();
        let (s, p) = c0c_strokes(&mut ml, id);
        vol.drain_dirty();
        // Undone: beside the strokes on top. Left in place: underneath, away from them.
        let at = if undo_volume { Vec3::new(-0.3, 0.2, 0.9) } else { Vec3::new(0.3, 0.2, -0.9) };
        let stroke = vol_stroke(&mut vol, at, 0.1);
        let dirty = vol.drain_dirty();
        let (_, vp) = ml.refit_cached_patch(&vol, &dirty, stroke).unwrap();
        assert!(!ml.doc.can_undo(), "the swap kept the document's undo");
        let after_volume = state(&ml);
        if undo_volume {
            assert!(vol.undo());
            let dirty = vol.drain_dirty();
            ml.refit_cached(&vol, &dirty, MeshLayers::changed_box(&vol, &dirty)).unwrap();
            ml.restore(&vp).unwrap();
            fresh(&mut ml.doc);
            assert_eq!(differ(&state(&ml), &s[2], &|_| true), (0, 0), "volume undo");
            c0c_round_trip(&mut ml, &s, &p, &|_| true, "after a volume stroke undone");
        } else {
            // Names whose base the volume stroke did not move.
            let unmoved: HashSet<u64> = after_volume.iter().filter(|(k, v)| s[2].get(k).is_some_and(|w| w[..3] == v[..3])).map(|(k, _)| *k).collect();
            let near: HashSet<u64> = s[2].keys().filter(|k| !unmoved.contains(k)).copied().collect();
            eprintln!("C0c across a swap: the volume stroke moved or renamed {} vertices", near.len());
            assert!(near.len() > 100 && unmoved.len() > 1000);
            // Compared away from the volume stroke (by a brush radius past it).
            let far = |k: u64| unmoved.contains(&k) && s[2].get(&k).is_some_and(|v| {
                let p = Vec3::new(f32::from_bits(v[0]), f32::from_bits(v[1]), f32::from_bits(v[2]));
                p.distance(at.normalize() * R) > 0.25
            });
            c0c_round_trip(&mut ml, &s, &p, &far, "with a volume stroke in place");
        }
    }
}

// ------------------------------------------------------------------------------------------------------------ C0e

#[test]
fn c0e_stream_save_and_reopen() {
    let (vol, mut ml, id) = c0c_scene();
    let step = ml.step();
    let id2 = fill_layer(&mut ml.doc, "noise", |doc, v| doc.normals()[v].normalize_or_zero() * (0.5 * step * noise(doc.base_positions()[v])));
    let mut m = ml.doc.layer(id2).unwrap().meta();
    m.blend = LayerBlend::Max;
    m.opacity = 1.3;
    ml.doc.set_layer_meta(id2, m, false).unwrap();
    ml.doc.set_layer_opacity(id, 0.7).unwrap();
    let mut bytes = Vec::new();
    ml.write_to(&mut bytes).unwrap();
    let len = bytes.len();
    // Something after it in the file: left unread.
    bytes.extend_from_slice(b"NEXT");
    let mut cur = std::io::Cursor::new(&bytes[..]);
    let (back, lost) = MeshLayers::read_from(&mut cur, &vol).unwrap();
    assert_eq!(cur.position() as usize, len, "read_from read past its own bytes");
    assert_eq!(lost, 0);
    assert_eq!(back.keys(), ml.keys());
    for lid in [id, id2] {
        let (a, b) = (ml.doc.layer(lid).unwrap(), back.doc.layer(lid).unwrap());
        assert_eq!(a.meta(), b.meta(), "layer meta");
        assert_eq!(a.opacity.to_bits(), b.opacity.to_bits());
    }
    // Offsets, mask values, channels and the composite, by name.
    let mut a = ml;
    fresh(&mut a.doc);
    let d = differ(&state(&a), &state(&back), &|_| true);
    // The directory form writes the same.
    let dir = std::env::temp_dir().join(format!("mesh_layers_c0e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    a.save(&dir).unwrap();
    let (back2, _) = MeshLayers::load(&dir, &vol).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let d2 = differ(&state(&back2), &state(&back), &|_| true);
    eprintln!("C0e: {} verts, {len} bytes; stream reopen (names missing, vertices differing) {d:?}; directory vs stream {d2:?}", back.doc.vertex_count());
    assert_eq!(d, (0, 0), "stream reopen differs");
    assert_eq!(d2, (0, 0), "directory and stream differ");
}

#[test]
fn c0e_refresh_painted_masks_shows_a_mask_stroke_while_open() {
    let (_vol, mut ml, id) = c0c_scene();
    let before = ml.doc.layer(id).unwrap().mask_values().unwrap().to_vec();
    ml.doc.begin_stroke("Paint");
    let h = ml.doc.raycast(&Ray::new(Vec3::new(0.05, 0.0, 3.0), -Vec3::Z)).unwrap();
    ml.doc.paint(&PaintTarget::Channel("paint.half".into()), h.point, 0.1, &Falloff::default(), 0.8, 0.0, None);
    let open = ml.doc.layer(id).unwrap().mask_values().unwrap().to_vec();
    ml.doc.refresh_painted_masks().unwrap();
    let live = ml.doc.layer(id).unwrap().mask_values().unwrap().to_vec();
    ml.doc.end_stroke();
    let ended = ml.doc.layer(id).unwrap().mask_values().unwrap().to_vec();
    assert_eq!(open, before, "the mask changed before the refresh");
    assert_ne!(live, before, "the refresh did not show the stroke");
    assert_eq!(live, ended, "the refresh is not what the stroke's end gives");
}

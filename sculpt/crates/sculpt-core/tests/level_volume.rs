//! The multi-resolution volume: brushes at every level keep the surface closed, fine detail rides coarse strokes, the
//! surface rebuilt block by block matches a full rebuild, and strokes undo and save exactly.

use std::collections::HashMap;

use glam::Vec3;
use sculpt_core::generator::level_volume::{BlockSurface, Key, LevelVolume};
use sculpt_core::generator::levels::edge_counts;

fn ball(levels: usize) -> LevelVolume {
    LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, levels)
}

/// A clay (or trim) stroke at level `l` from direction `a` to `b` (each lands on the surface under it).
fn stroke(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, strength: f32, trim: bool) {
    v.begin_stroke();
    let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
    let mut along = Vec3::ZERO;
    for k in 0..=steps {
        let dir = a.lerp(b, k as f32 / steps as f32).normalize();
        let Some((hit, n)) = v.raycast(dir * 3.0, -dir, l, true) else { continue };
        let h = r * if trim { 0.015 + 0.06 * strength } else { 0.015 + 0.08 * strength };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, 0.5, 0.1 + 0.3 * strength, trim) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
    }
    v.end_stroke();
}

/// Open edges, non-manifold edges and dropped quads of the surface at `cap`.
fn check(v: &LevelVolume, cap: usize) -> (usize, usize, usize, usize) {
    let (mesh, stats) = v.surface(cap);
    let (open, nm) = edge_counts(&mesh.faces);
    (open, nm, stats.dropped, mesh.faces.len())
}

#[test]
fn a_fresh_ball_is_closed_at_every_cap() {
    let v = ball(4);
    for cap in 0..4 {
        let (open, nm, dropped, faces) = check(&v, cap);
        eprintln!("cap {cap}: {faces} faces, open {open}, non-manifold {nm}, dropped {dropped}");
        assert_eq!((open, nm, dropped), (0, 0, 0));
    }
}

#[test]
fn plane_brushes_at_every_level_keep_the_surface_closed() {
    let mut v = ball(4);
    let front = Vec3::new(0.0, 0.0, 1.0);
    // Level 0: broad clay and a trim; level 1: ridges; level 2: a trimmed flat; level 3: small clay.
    stroke(&mut v, 0, Vec3::new(-0.6, 0.2, 1.0), Vec3::new(0.6, 0.3, 1.0), 0.35, 0.8, false);
    stroke(&mut v, 0, Vec3::new(-0.5, -0.4, 1.0), Vec3::new(0.4, -0.3, 1.0), 0.3, 0.8, true);
    for i in 0..3 {
        let y = -0.2 + 0.15 * i as f32;
        stroke(&mut v, 1, Vec3::new(-0.4, y, 1.0), Vec3::new(0.3, y + 0.05, 1.0), 0.06, 1.0, false);
    }
    stroke(&mut v, 2, Vec3::new(0.1, 0.0, 1.0), Vec3::new(0.25, 0.05, 1.0), 0.03, 1.0, true);
    for i in 0..6 {
        let p = front + Vec3::new(0.05 * i as f32 - 0.1, 0.12, 0.0);
        stroke(&mut v, 3, p, p + Vec3::new(0.01, 0.0, 0.0), 0.012, 1.0, false);
    }
    let (counts, bytes) = v.memory();
    eprintln!("blocks per level {counts:?}, {:.1} MB", bytes as f64 / 1e6);
    for cap in 0..4 {
        let (open, nm, dropped, faces) = check(&v, cap);
        eprintln!("cap {cap}: {faces} faces, open {open}, non-manifold {nm}, dropped {dropped}");
        assert_eq!((open, dropped), (0, 0), "cap {cap}");
        assert!(nm <= 8, "cap {cap}: {nm} non-manifold edges");
    }
}

/// Height of the surface above the ball at angle `a` around y (from +z), seen along the radius.
fn height(v: &LevelVolume, a: f32) -> f32 {
    let dir = Vec3::new(a.sin(), 0.0, a.cos());
    v.raycast(dir * 3.0, -dir, v.levels() - 1, false).map_or(f32::NAN, |(h, _)| h.length() - 1.0)
}

#[test]
fn fine_detail_rides_a_coarse_clay_stroke() {
    let mut v = ball(3);
    // A groove along y at the finest level.
    v.begin_stroke();
    for i in 0..40 {
        let y = -0.15 + 0.3 * i as f32 / 39.0;
        let p = Vec3::new(0.0, y, (1.0 - y * y).sqrt());
        v.ball_dab(2, p * 1.003, 0.012, 0.003, true);
    }
    v.end_stroke();
    let depth = |v: &LevelVolume| (height(v, -0.03) + height(v, 0.03)) / 2.0 - height(v, 0.0);
    let (before, lift0) = (depth(&v), height(&v, 0.03));
    // Broad clay over it at level 0, scrubbed a few times.
    for _ in 0..3 {
        stroke(&mut v, 0, Vec3::new(0.0, -0.3, 1.0), Vec3::new(0.0, 0.3, 1.0), 0.4, 1.0, false);
    }
    let (after, lift1) = (depth(&v), height(&v, 0.03));
    eprintln!("groove depth {before:.4} -> {after:.4} ({:.0}%), the area rose {:.4}", 100.0 * after / before, lift1 - lift0);
    assert!(before > 0.004, "the groove was not cut");
    assert!(lift1 - lift0 > 0.01, "the clay did not build");
    assert!(after > 0.8 * before, "the coarse stroke lost the detail");
    let (open, _, dropped, _) = check(&v, 2);
    assert_eq!((open, dropped), (0, 0));
}

#[test]
fn form_built_at_a_fine_level_moves_down_a_level() {
    let mut v = ball(3);
    // A tall ridge of clay at level 2, built over many passes: far more than level 2's band above the level below.
    let top = |v: &LevelVolume, cap: usize| v.raycast(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z, cap, false).map_or(f32::NAN, |(h, _)| h.z - 1.0);
    for _ in 0..12 {
        stroke(&mut v, 2, Vec3::new(-0.06, 0.0, 1.0), Vec3::new(0.06, 0.0, 1.0), 0.03, 1.0, false);
        let most = |l: u8| v.keys().iter().filter(|k| k.0 == l).map(|k| v.detail_of(l as usize, k.1)).fold(0.0f32, f32::max);
        eprintln!("  heights by cap {:.4} {:.4} {:.4}; most detail at levels 1, 2: {:.4} {:.4}", top(&v, 0), top(&v, 1), top(&v, 2), most(1), most(2));
    }
    let built = top(&v, 2);
    let worst = v.keys().iter().filter(|k| k.0 == 2).map(|k| v.detail_of(2, k.1)).fold(0.0f32, f32::max);
    eprintln!("ridge {built:.4} high (level 2 band {:.4}); most level 2 detail after the rebase {worst:.4}; seen from afar (levels 0-1) {:.4}", v.band(2), top(&v, 1));
    assert!(built > 2.0 * v.band(2), "the ridge did not build");
    // A distant view (without level 2) still shows the form: it moved down a level.
    assert!(top(&v, 1) > 0.6 * built, "the form did not move down");
    let (open, _, dropped, _) = check(&v, 2);
    assert_eq!((open, dropped), (0, 0));
}

#[test]
fn extracting_only_what_changed_matches_a_full_extraction() {
    let mut v = ball(4);
    let cap = 3;
    let all = v.keys();
    let (mut kept, _) = v.extract(&all, cap);
    v.drain_dirty();
    let mut strokes = 0;
    for (l, r, trim) in [(0usize, 0.3f32, false), (1, 0.08, false), (2, 0.03, true), (3, 0.012, false), (0, 0.35, true), (2, 0.02, false)] {
        let a = Vec3::new(-0.2 + 0.1 * strokes as f32, 0.1, 1.0);
        stroke(&mut v, l, a, a + Vec3::new(0.2, 0.05, 0.0), r, 1.0, trim);
        strokes += 1;
        let dirty = v.drain_dirty();
        let work = v.extract_set(&dirty, cap);
        let (fresh, _) = v.extract(&work, cap);
        for k in &dirty {
            if v.block(*k).is_none() {
                kept.remove(k);
            }
        }
        for (k, s) in fresh {
            kept.insert(k, s);
        }
        let (full, _) = v.extract(&v.keys(), cap);
        assert_eq!(fingerprint(&kept), fingerprint(&full), "after stroke {strokes} (level {l})");
    }
}

/// Quads by where their corners are, as a sorted list.
fn fingerprint(s: &HashMap<Key, BlockSurface>) -> (usize, Vec<[i64; 3]>) {
    let pos = |(k, c, p): &(Key, u16, u8)| s[k].verts[s[k].vertex(*c, *p).expect("a quad points at a cell without a vertex") as usize];
    let mut out: Vec<[i64; 3]> = s
        .values()
        .flat_map(|b| b.quads.iter())
        .map(|q| {
            let c = q.iter().map(pos).sum::<Vec3>();
            [(c.x * 1e5) as i64, (c.y * 1e5) as i64, (c.z * 1e5) as i64]
        })
        .collect();
    out.sort_unstable();
    (out.len(), out)
}

#[test]
fn strokes_undo_redo_and_save_exactly() {
    let mut v = ball(4);
    let bytes = |v: &LevelVolume| {
        let mut out = Vec::new();
        v.write_to(&mut out).unwrap();
        out
    };
    let start = bytes(&v);
    stroke(&mut v, 0, Vec3::new(-0.3, 0.0, 1.0), Vec3::new(0.3, 0.1, 1.0), 0.3, 1.0, false);
    stroke(&mut v, 2, Vec3::new(-0.1, 0.0, 1.0), Vec3::new(0.1, 0.0, 1.0), 0.03, 1.0, true);
    stroke(&mut v, 3, Vec3::new(0.0, 0.05, 1.0), Vec3::new(0.05, 0.05, 1.0), 0.01, 1.0, false);
    let end = bytes(&v);
    while v.undo() {}
    assert_eq!(bytes(&v), start, "undo did not restore the start");
    while v.redo() {}
    assert_eq!(bytes(&v), end, "redo did not restore the end");
    let read = LevelVolume::read_from(&mut end.as_slice(), 4).unwrap();
    assert_eq!(bytes(&read), end, "a saved volume reads back differently");
    let (a, b) = (v.surface(3).0, read.surface(3).0);
    assert_eq!(a.faces.len(), b.faces.len());
}

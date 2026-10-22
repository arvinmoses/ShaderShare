//! The multi-resolution volume: brushes at every level keep the surface closed, fine detail rides coarse strokes, the
//! surface rebuilt block by block matches a full rebuild, and strokes undo and save exactly.

use sculpt_core::generator::level_volume::HashMap;

use glam::Vec3;
use sculpt_core::generator::level_volume::{BlockSurface, Key, LevelVolume};
use sculpt_core::generator::levels::edge_counts;

/// A ball whose coarse strokes keep the finer detail under them (the tests of carrying it want that; burying it is
/// tested on its own).
fn ball(levels: usize) -> LevelVolume {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, levels);
    v.keep_detail = true;
    v.keep_detail = true;
    v
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
    assert!(top(&v, 1) > 0.45 * built, "the form did not move down");
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

#[test]
fn specks_go_and_real_pieces_stay() {
    let mut v = ball(3);
    let pieces = |v: &LevelVolume| {
        let (mesh, _) = v.surface(2);
        // Count connected pieces of the mesh.
        let mut parent: Vec<u32> = (0..mesh.positions.len() as u32).collect();
        fn root(p: &mut [u32], mut i: u32) -> u32 {
            while p[i as usize] != i {
                p[i as usize] = p[p[i as usize] as usize];
                i = p[i as usize];
            }
            i
        }
        let mut used = vec![false; mesh.positions.len()];
        for f in &mesh.faces {
            let n = if f[3] == sculpt_core::mesh::NO_VERT { 3 } else { 4 };
            for k in 0..n {
                used[f[k] as usize] = true;
                let (a, b) = (root(&mut parent, f[k]), root(&mut parent, f[(k + 1) % n]));
                parent[a as usize] = b;
            }
        }
        let mut roots: Vec<u32> = (0..mesh.positions.len() as u32).filter(|&i| used[i as usize]).map(|i| root(&mut parent, i)).collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    };
    assert_eq!(pieces(&v), 1);
    // A stroke that leaves a crumb floating off the surface (a few steps across) and a real lump (many).
    v.begin_stroke();
    v.ball_dab(2, Vec3::new(0.0, 0.0, 1.03), 0.006, 0.002, false);
    v.ball_dab(2, Vec3::new(0.3, 0.0, 1.12), 0.04, 0.01, false);
    let before = pieces(&v);
    let end = v.end_stroke();
    let after = pieces(&v);
    eprintln!("pieces {before} -> {after}, specks removed {}", end.specks);
    assert_eq!(before, 3, "the stroke should have left a crumb and a lump");
    assert_eq!(end.specks, 1);
    assert_eq!(after, 2, "the lump must stay");
}

#[test]
fn fine_detail_rides_a_big_rise_without_spawning_blocks() {
    // A groove at level 4 (16 times finer than level 0), then broad clay at level 0 raising the area by many level 4
    // steps.
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
    v.keep_detail = true;
    v.begin_stroke();
    for i in 0..60 {
        let y = -0.05 + 0.1 * i as f32 / 59.0;
        let p = Vec3::new(0.0, y, (1.0 - y * y).sqrt());
        v.ball_dab(4, p * 1.0008, 0.003, 0.0008, true);
    }
    v.end_stroke();
    let depth = |v: &LevelVolume| (height(v, -0.008) + height(v, 0.008)) / 2.0 - height(v, 0.0);
    let (before, lift0, blocks0) = (depth(&v), height(&v, 0.03), v.memory().0);
    let t = std::time::Instant::now();
    for _ in 0..3 {
        stroke(&mut v, 0, Vec3::new(0.0, -0.2, 1.0), Vec3::new(0.0, 0.2, 1.0), 0.35, 1.0, false);
    }
    let (after, lift1, blocks1) = (depth(&v), height(&v, 0.03), v.memory().0);
    eprintln!("groove {before:.5} -> {after:.5} ({:.0}%), area rose {:.4} ({:.0} level 4 steps) in {:.0} ms; blocks per level {blocks0:?} -> {blocks1:?}", 100.0 * after / before, lift1 - lift0, (lift1 - lift0) / v.voxel(4), t.elapsed().as_secs_f64() * 1e3);
    assert!(before > 0.0008, "the groove was not cut");
    assert!(lift1 - lift0 > 20.0 * v.voxel(4), "the clay did not build");
    assert!(after > 0.7 * before, "the detail was lost");
    assert!(blocks1[4] < 3 * blocks0[4].max(1), "carrying the detail spawned blocks");
    let (open, _, dropped, _) = check(&v, 4);
    assert_eq!((open, dropped), (0, 0));
}

#[test]
fn fine_bumps_survive_a_level_0_stroke_carried_a_little_at_a_time() {
    let mut results = Vec::new();
    // Bumps 4 thousandths across: at least a couple of steps of the level they are made at.
    // The last: many strokes, raising the area far more than the band of any finer level.
    // And on a finer level 0 (about 1M triangles' step), whose band is far narrower than the rise.
    // With `jobs`, as the app carries: taken off the queue after each dab and brought in after the next one, so every
    // job's result lands on a volume a dab has changed since.
    for (fine, carve, strokes, voxel0, jobs) in [
        (5usize, false, 2, 0.02, false),
        (5, true, 2, 0.02, false),
        (4, false, 2, 0.02, false),
        (4, true, 2, 0.02, false),
        (3, false, 2, 0.02, false),
        (4, false, 6, 0.02, false),
        (3, false, 2, 0.005, false),
        (5, false, 2, 0.02, true),
        (4, false, 6, 0.02, true),
        (3, false, 2, 0.005, true),
    ] {
        let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel0, fine + 1);
        v.keep_detail = true;
        v.carry_budget = Some(if jobs { 0.0 } else { 2.0 });
        // A row of separate bumps (or pits) at a fine level, each its own stroke.
        let xs: Vec<f32> = (0..5).map(|i| -0.024 + 0.012 * i as f32).collect();
        for &x in &xs {
            v.begin_stroke();
            let p = Vec3::new(x, 0.0, 1.0).normalize();
            v.ball_dab(fine, p * if carve { 1.0035 } else { 1.0 }, 0.004, 0.001, carve);
            v.end_stroke();
        }
        // Each bump's height over the mean of the surface either side of it.
        // Each bump's height over the surface beside it, wherever it ended up within a little of where it was made.
        // Heights along the strokes' push (their planes face +z over the top of the ball): detail carried along it
        // stays where it was seen that way.
        let h2 = |v: &LevelVolume, x: f32, y: f32| {
            let p = Vec3::new(x, y, 1.0).normalize();
            v.raycast(Vec3::new(p.x, p.y, 3.0), -Vec3::Z, v.levels() - 1, false).map_or(f32::NAN, |(h, _)| h.z - 1.0)
        };
        let relief = |v: &LevelVolume| {
            xs.iter()
                .map(|&x| {
                    let mut best = 0.0f32;
                    for i in 0..61 {
                        let y = -0.03 + 0.001 * i as f32;
                        for j in 0..9 {
                            let x = x - 0.004 + 0.001 * j as f32;
                            let r = h2(v, x, y) - 0.5 * (h2(v, x - 0.006, y) + h2(v, x + 0.006, y));
                            if r.abs() > best.abs() {
                                best = r;
                            }
                        }
                    }
                    best.abs()
                })
                .sum::<f32>()
                / xs.len() as f32
        };
        let (before, top0, blocks0) = (relief(&v), height(&v, 0.0), v.memory().0);
        for _ in 0..strokes {
            v.begin_stroke();
            let mut along = Vec3::ZERO;
            let mut held: Option<sculpt_core::generator::level_volume::CarryJob> = None;
            for k in 0..=8 {
                let dir = Vec3::new(0.0, -0.1 + 0.2 * k as f32 / 8.0, 1.0).normalize();
                let Some((hit, n)) = v.raycast(dir * 3.0, -dir, 0, true) else { continue };
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, _)) = v.plane_dab(0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
                    along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
                }
                if jobs {
                    if let Some(job) = held.take() {
                        let done = job.run();
                        v.carry_apply(done);
                    }
                    held = v.carry_take(64);
                } else {
                    v.carry_pending(Some(2.0));
                }
            }
            if let Some(job) = held.take() {
                v.carry_apply(job.run());
            }
            v.end_stroke();
        }
        let after = relief(&v);
        eprintln!("level {fine} (level 0 step {voxel0}{}) {} after {strokes} strokes: relief {before:.5} -> {after:.5} ({:.0}%), the area rose {:.4}; blocks per level {blocks0:?} -> {:?}", if jobs { ", carried in jobs" } else { "" }, if carve { "pits" } else { "bumps" }, 100.0 * after / before, height(&v, 0.0) - top0, v.memory().0);
        results.push((before, height(&v, 0.0) - top0, after));
    }
    for (before, rose, after) in results {
        assert!(before > 0.0003, "the dabs did not make bumps");
        assert!(rose > 0.02, "the clay did not build");
        assert!(after > 0.55 * before, "the fine detail was lost");
    }
}

/// Ending a stroke on another thread (a job on a copy, its result taken back) leaves exactly what ending it at once
/// does, and undoes the same.
#[test]
fn a_stroke_ended_in_a_job_is_the_same_stroke() {
    let make = || {
        let mut v = ball(4);
        // Fine work, broad clay over it, then a ridge at level 2 built tall enough to move form down a level; the last
        // pass is left open.
        stroke(&mut v, 3, Vec3::new(0.0, 0.05, 1.0), Vec3::new(0.05, 0.05, 1.0), 0.01, 1.0, false);
        stroke(&mut v, 0, Vec3::new(-0.3, 0.0, 1.0), Vec3::new(0.3, 0.1, 1.0), 0.3, 1.0, false);
        for _ in 0..11 {
            stroke(&mut v, 2, Vec3::new(-0.06, 0.0, 1.0), Vec3::new(0.06, 0.0, 1.0), 0.03, 1.0, false);
        }
        let mut open = stroke_open(&mut v, 2, Vec3::new(-0.06, 0.0, 1.0), Vec3::new(0.06, 0.0, 1.0), 0.03);
        open.carry_pending(None);
        open
    };
    let bytes = |v: &LevelVolume| {
        let mut b = Vec::new();
        v.write_to(&mut b).unwrap();
        b
    };
    let (mut now, mut later) = (make(), make());
    let a = now.end_stroke();
    let job = later.end_stroke_take().expect("the stroke changed something");
    let b = later.end_stroke_apply(job.run());
    eprintln!("ended at once: {} rebased, {} specks; in a job: {} rebased, {} specks", a.rebased, a.specks, b.rebased, b.specks);
    assert_eq!((a.specks, a.rebased), (b.specks, b.rebased));
    assert!(a.rebased > 0, "the ridge should have moved form down a level");
    assert!(bytes(&now) == bytes(&later), "ending in a job left a different volume");
    assert!(now.undo() && later.undo());
    assert!(bytes(&now) == bytes(&later), "the strokes ended in a job undo differently");
}

/// [`stroke`] without its end.
fn stroke_open(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32) -> LevelVolume {
    let mut w = std::mem::replace(v, ball(1));
    w.begin_stroke();
    let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
    let mut along = Vec3::ZERO;
    for k in 0..=steps {
        let dir = a.lerp(b, k as f32 / steps as f32).normalize();
        let Some((hit, n)) = w.raycast(dir * 3.0, -dir, l, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = w.plane_dab(l, hit, rough, r, r * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
    }
    w
}

/// Across a groove cut at `level` along `along` through `patch`, at 40 stations over `+-half` along it and `+-width`
/// across, seen along `push`: how far the surface lies below the surface without the groove's level (the ground),
/// and where its bottom is across (the middle of where it is more than half as deep).
#[allow(clippy::too_many_arguments)]
fn groove_profile(v: &LevelVolume, level: usize, patch: Vec3, along: Vec3, across: Vec3, push: Vec3, half: f32, width: f32) -> Vec<(f32, f32)> {
    let all = v.levels() - 1;
    (0..40)
        .map(|i| {
            let s = -half + 2.0 * half * i as f32 / 39.0;
            let depths: Vec<(f32, f32)> = (0..81)
                .map(|j| {
                    let u = -width + 2.0 * width * j as f32 / 80.0;
                    let o = patch + along * s + across * u + push * 2.0;
                    let h = |cap: usize| v.raycast(o, -push, cap, false).map_or(f32::NAN, |(p, _)| p.dot(push));
                    (h(level - 1) - h(all), u)
                })
                .collect();
            let deepest = depths.iter().map(|q| q.0).fold(f32::NEG_INFINITY, f32::max);
            let (sum, weight) = depths.iter().fold((0.0, 0.0), |(s, w), q| {
                let x = (q.0 - 0.5 * deepest).max(0.0);
                (s + x * q.1, w + x)
            });
            (deepest, if weight > 0.0 { sum / weight } else { 0.0 })
        })
        .collect()
}

/// The review session's case: clay at level 1, four level 4 trim grooves close together over it, on a level 0 of 1M
/// triangles' step, then a big level 0 clay stroke right over them, its detail carried in jobs as the app carries it
/// (one job of a few hundred blocks at a time, started once a frame, two dabs a frame, so its result lands two dabs
/// late). Each groove stays whole, as deep along its length as it was, where they used to break into stepped pieces.
#[test]
fn trim_grooves_ride_a_big_stroke_carried_in_jobs() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6);
    v.keep_detail = true;
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    let plane_stroke = |v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool, jobs: bool| {
        v.begin_stroke();
        let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
        let mut along = Vec3::ZERO;
        let mut held: Option<sculpt_core::generator::level_volume::CarryJob> = None;
        for k in 0..=steps {
            let d = a.lerp(b, k as f32 / steps as f32).normalize();
            let Some((hit, n)) = v.raycast(d * 3.0, -d, l, true) else { continue };
            let h = r * if trim { 0.075 } else { 0.095 };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, 0.2, 0.4, trim) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            if jobs && k % 2 == 1 {
                if let Some(job) = held.take() {
                    v.carry_apply(job.run());
                }
                held = v.carry_take(384);
            }
        }
        if let Some(job) = held.take() {
            v.carry_apply(job.run());
        }
        v.end_stroke();
    };
    // Clay at level 1 either side of the grooves, so the levels between hold form, and the grooves sit on its slope.
    for k in 0..2 {
        let o = t2 * (0.05 * k as f32 - 0.025);
        plane_stroke(&mut v, 1, patch + o - t1 * 0.08, patch + o + t1 * 0.08, 0.06, false, false);
    }
    for k in 0..4 {
        let o = t2 * (0.004 * k as f32 - 0.006);
        plane_stroke(&mut v, 4, patch + o - t1 * 0.01, patch + o + t1 * 0.01, 0.0025, true, false);
    }
    let push = patch;
    let along = (t1 - push * t1.dot(push)).normalize();
    let across = push.cross(along);
    let grooves: Vec<Vec3> = (0..4).map(|k| patch + t2 * (0.004 * k as f32 - 0.006)).collect();
    let profiles = |v: &LevelVolume| -> Vec<Vec<(f32, f32)>> { grooves.iter().map(|&g| groove_profile(v, 4, g, along, across, push, 0.006, 0.0018)).collect() };
    let before = profiles(&v);
    let blocks0 = v.memory().0;
    v.carry_budget = Some(0.0);
    let t = std::time::Instant::now();
    plane_stroke(&mut v, 0, patch - t1 * 0.05, patch + t1 * 0.05, 0.29, false, true);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let after = profiles(&v);
    let rise = v.raycast(patch * 3.0, -patch, 0, false).map_or(0.0, |(p, _)| p.length() - 1.0);
    let median = |p: &[(f32, f32)]| {
        let mut d: Vec<f32> = p.iter().map(|x| x.0).collect();
        d.sort_by(f32::total_cmp);
        d[d.len() / 2]
    };
    let (open, _, dropped, _) = check(&v, 5);
    assert_eq!((open, dropped), (0, 0));
    assert!(rise > 0.02, "the clay did not build there");
    for (k, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        let least = a.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        let jump = a.windows(2).map(|w| (w[1].1 - w[0].1).abs()).fold(0.0f32, f32::max);
        eprintln!("groove {k}: depth median {:.5} -> {:.5}, least {least:.5} after; largest jump of its bottom between stations {jump:.5} (level 4 step {:.5})", median(b), median(a), v.voxel(4));
        if std::env::var("GROOVE_DUMP").is_ok() {
            eprintln!("   depth by station, in 1e-5: {}", a.iter().map(|x| format!("{:.0}", x.0 * 1e5)).collect::<Vec<_>>().join(" "));
            eprintln!("   bottom by station, in 1e-4: {}", a.iter().map(|x| format!("{:+.1}", x.1 * 1e4)).collect::<Vec<_>>().join(" "));
        }
        assert!(least > 0.7 * median(b), "groove {k} broke: somewhere it is {least:.5} deep, against {:.5}", median(b));
        assert!(jump < 2.0 * v.voxel(4), "groove {k} jumps sideways by {jump:.5}");
    }
    eprintln!("the ground rose {rise:.4} in {ms:.0} ms; blocks per level {blocks0:?} -> {:?}", v.memory().0);
}

/// A groove at level 4 on the slope a big level 0 clay stroke leaves at its rim: it rides along whole, as deep along
/// its length as it was and without jumping sideways, where it used to break into pieces cut off at block faces.
#[test]
fn a_groove_on_a_big_strokes_rim_rides_along_whole() {
    // The rim close to the groove (it rises 0.05 over two passes), and nearer the stroke's line (steeper).
    for (off, passes) in [(0.12f32, 1usize), (0.15, 2)] {
        let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6);
        v.keep_detail = true;
        let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
        let (t1, t2) = patch.any_orthonormal_pair();
        let plane_stroke = |v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool| {
            v.begin_stroke();
            let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
            let mut along = Vec3::ZERO;
            for k in 0..=steps {
                let d = a.lerp(b, k as f32 / steps as f32).normalize();
                let Some((hit, n)) = v.raycast(d * 3.0, -d, l, true) else { continue };
                let h = r * if trim { 0.075 } else { 0.095 };
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, 0.2, 0.4, trim) {
                    along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
                }
            }
            if std::env::var("RIM_NOEND").is_ok() && l == 0 {
                v.carry_pending(None);
                return;
            }
            v.end_stroke();
        };
        plane_stroke(&mut v, 4, patch - t1 * 0.012, patch + t1 * 0.012, 0.0025, true);
        // The clay's push (its planes face about the middle of the stroke), and the groove's frame seen along it.
        let push = (patch + t2 * off).normalize();
        let along = (t1 - push * t1.dot(push)).normalize();
        let across = push.cross(along);
        // Across the groove at stations along it, seen along the push: its depth below the ground and where its bottom
        // is. The ground is the surface without the groove's level (a groove this shallow is never moved down a
        // level): a line through the sides would read the rim's curve as lost depth, a quarter of it after one pass.
        let profile = |v: &LevelVolume| -> Vec<(f32, f32)> { groove_profile(v, 4, patch, along, across, push, 0.008, 0.008) };
        let before = profile(&v);
        let blocks0 = v.memory().0;
        // Big clay, its line `off` to the side, so the groove lies on the slope of its rim.
        for _ in 0..passes {
            plane_stroke(&mut v, 0, patch + t2 * off - t1 * 0.06, patch + t2 * off + t1 * 0.06, 0.29, false);
        }
        let after = profile(&v);
        if std::env::var("GROOVE_DUMP").is_ok() {
            for (i, (b, q)) in before.iter().zip(after.iter()).enumerate() {
                eprintln!("  station {i:2}: depth {:.5} -> {:.5}, bottom at {:+.5} -> {:+.5}", b.0, q.0, b.1, q.1);
            }
        }
        let median = |p: &[(f32, f32)]| {
            let mut d: Vec<f32> = p.iter().map(|x| x.0).collect();
            d.sort_by(f32::total_cmp);
            d[d.len() / 2]
        };
        let least = after.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        let jump = after.windows(2).map(|w| (w[1].1 - w[0].1).abs()).fold(0.0f32, f32::max);
        let rise = v.raycast(patch * 3.0, -patch, 0, false).map_or(0.0, |(p, _)| p.length() - 1.0);
        eprintln!(
            "clay {off} to the side, {passes} passes: groove depth along it: median {:.5} before, {:.5} after, least {least:.5} after; largest jump of its bottom between stations {jump:.5} (level 4 step {:.5}); the ground rose {rise:.4}; blocks per level {blocks0:?} -> {:?}",
            median(&before),
            median(&after),
            v.voxel(4),
            v.memory().0
        );
        let (open, _, dropped, _) = check(&v, 5);
        assert_eq!((open, dropped), (0, 0));
        assert!(rise > 0.01, "the clay did not build there");
        assert!(least > 0.7 * median(&before), "the groove broke: somewhere it is {least:.5} deep, against {:.5}", median(&before));
        assert!(jump < 2.0 * v.voxel(4), "the groove jumps sideways by {jump:.5}");
    }
}

/// The review session's patch, as its dabs were logged from the app: a level 0 clay stroke ending over it, two
/// level 2 clay strokes (the rolls), four level 4 trim grooves between them, six level 4 clay dots below, then the
/// big level 0 clay stroke whose line passes 0.18 to the side, so the patch lies on the brush's falloff slope and
/// gets two or three dabs of a third strength, its detail carried in jobs as the app carries it. After it the rolls
/// keep their shape, every groove and dot keeps its depth, and the surface between them is as smooth as before.
#[test]
fn the_review_sessions_patch_rides_its_big_stroke() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6);
    v.keep_detail = true;
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    let at = |a: f32, b: f32| (patch + t1 * a + t2 * b).normalize();
    // A stroke of the app's clay or trim brush at level `l`: dabs along the path at the app's spacing, each placed on
    // the surface as it was when the stroke began, the plane's tilt eased as the app eases it.
    let stroke = |v: &mut LevelVolume, l: usize, path: &[Vec3], r: f32, hard: f32, trim: bool, jobs: bool| {
        v.begin_stroke();
        let mut along = Vec3::ZERO;
        let mut held: Option<sculpt_core::generator::level_volume::CarryJob> = None;
        let mut k = 0;
        for w in path.windows(2) {
            let steps = ((w[1] - w[0]).length() / (0.15 * r)).ceil().max(1.0) as usize;
            for i in 0..steps {
                let d = w[0].lerp(w[1], i as f32 / steps as f32).normalize();
                let Some((hit, n)) = v.raycast(d * 3.0, -d, l, true) else { continue };
                let h = r * if trim { 0.075 } else { 0.095 };
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, hard, 0.4, trim) {
                    along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
                }
                k += 1;
                if jobs && k % 2 == 0 {
                    if let Some(job) = held.take() {
                        v.carry_apply(job.run());
                    }
                    held = v.carry_take(384);
                }
            }
        }
        if let Some(job) = held.take() {
            v.carry_apply(job.run());
        }
        v.end_stroke();
    };
    // Earlier coarse form: the first big clay stroke of the session ends over the patch.
    stroke(&mut v, 0, &[at(-0.8, 0.07), at(-0.4, 0.07), at(-0.05, 0.02), at(0.07, 0.07)], 0.46, 0.2, false, false);
    // The rolls: level 2 clay, wavy along t1 either side of the patch.
    for side in [1.0f32, -1.0] {
        let path: Vec<Vec3> = (0..=12).map(|i| { let a = -0.14 + 0.27 * i as f32 / 12.0; at(a, side * (0.045 + 0.02 * (a * 35.0).sin())) }).collect();
        stroke(&mut v, 2, &path, 0.042, 0.2, false, false);
    }
    // Four grooves: level 4 trim between the rolls, 0.014 apart.
    let grooves: Vec<f32> = vec![0.020, 0.006, -0.008, -0.022];
    for &b in &grooves {
        stroke(&mut v, 4, &[at(-0.032, b), at(0.030, b + 0.001)], 0.0034, 0.3, true, false);
    }
    // Six dots: level 4 clay, two dabs each, in a row below the grooves.
    let dots: Vec<Vec3> = (0..6).map(|i| at(-0.0234 + 0.0102 * i as f32, -0.0335)).collect();
    for &d in &dots {
        stroke(&mut v, 4, &[d, (d + t1 * 0.0005).normalize()], 0.00215, 0.2, false, false);
    }
    let push = patch;
    let along = (t1 - push * t1.dot(push)).normalize();
    let across = push.cross(along);
    let h = |v: &LevelVolume, p: Vec3, cap: usize| v.raycast(p + push * 2.0, -push, cap, false).map_or(f32::NAN, |(q, _)| q.dot(push));
    // The rolls: the surface over level 0 along a line across them, clear of the grooves and dots.
    let rolls = |v: &LevelVolume| -> Vec<f32> { (0..97).map(|j| { let p = patch + along * 0.08 + across * (-0.12 + 0.0025 * j as f32); h(v, p, 5) - h(v, p, 0) }).collect() };
    let groove_profiles = |v: &LevelVolume| -> Vec<Vec<(f32, f32)>> { grooves.iter().map(|&b| groove_profile(v, 4, patch + t2 * b, along, across, push, 0.02, 0.0035)).collect() };
    // Each dot's relief: the surface's peak within 0.0025 of it over its mean along that window's rim.
    let dot_heights = |v: &LevelVolume| -> Vec<f32> {
        dots.iter()
            .map(|&d| {
                let (mut peak, mut rim, mut n) = (f32::NEG_INFINITY, 0.0, 0);
                for i in 0..13 {
                    for j in 0..13 {
                        let p = d + along * (0.0004 * (i as f32 - 6.0)) + across * (0.0004 * (j as f32 - 6.0));
                        let x = h(v, p, 5);
                        peak = peak.max(x);
                        if i == 0 || i == 12 || j == 0 || j == 12 {
                            rim += x;
                            n += 1;
                        }
                    }
                }
                peak - rim / n as f32
            })
            .collect()
    };
    // Roughness over the fine work: the RMS second difference of the surface along lines between the grooves and
    // past the dots, 0.0025 apart, the finest level's step being 0.0003.
    let roughness_at = |v: &LevelVolume, cap: usize| -> f32 {
        let (mut sum, mut n) = (0.0f32, 0);
        for off in [0.013f32, -0.001, -0.015, -0.028, -0.040] {
            let base = patch + across * off - along * 0.03;
            let hs: Vec<f32> = (0..25).map(|j| h(v, base + along * (0.0025 * j as f32), cap)).collect();
            for w in hs.windows(3) {
                let d2 = w[0] - 2.0 * w[1] + w[2];
                sum += d2 * d2;
                n += 1;
            }
        }
        (sum / n as f32).sqrt()
    };
    let roughness = |v: &LevelVolume| roughness_at(v, 5);
    // And at the finest scale: the second difference at one level 4 step along the same lines.
    let grain_at = |v: &LevelVolume, origin: Vec3, cap: usize| -> f32 {
        let (mut sum, mut n) = (0.0f32, 0);
        for off in [0.013f32, -0.001, -0.015, -0.028, -0.040] {
            let base = origin + across * off - along * 0.03;
            let hs: Vec<f32> = (0..201).map(|j| h(v, base + along * (0.0003 * j as f32), cap)).collect();
            for w in hs.windows(3) {
                let d2 = w[0] - 2.0 * w[1] + w[2];
                sum += d2 * d2;
                n += 1;
            }
        }
        (sum / n as f32).sqrt()
    };
    let grain = |v: &LevelVolume, origin: Vec3| grain_at(v, origin, 5);
    let (rolls0, grooves0, dots0, rough0, blocks0) = (rolls(&v), groove_profiles(&v), dot_heights(&v), roughness(&v), v.memory().0);
    let grain0 = grain(&v, patch);
    let ground0 = h(&v, patch, 0);
    let across_profile = |v: &LevelVolume| -> Vec<f32> { (0..111).map(|j| { let b = -0.06 + 0.001 * j as f32; let p = patch + t2 * b; h(v, p, 5) - h(v, p, 3) }).collect() };
    let prof0 = across_profile(&v);
    // And along the rolls, through the dots (bumps over level 3), for the shift along them.
    let along_profile = |v: &LevelVolume, b: f32| -> Vec<f32> { (0..101).map(|j| { let a = -0.05 + 0.001 * j as f32; let p = patch + t2 * b + t1 * a; h(v, p, 5) - h(v, p, 3) }).collect() };
    let dots_row = -0.0335;
    let prof_along0 = along_profile(&v, dots_row);
    // The big stroke: five dabs of the app's spacing, its line 0.18 to the side of the patch.
    v.carry_budget = Some(0.0);
    let t = std::time::Instant::now();
    stroke(&mut v, 0, &[at(-0.011, 0.187), at(0.252, 0.169)], 0.445, 0.2, false, true);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    // The fine work is carried along the stroke's push, which tilts toward the dabs' centres 0.18 to the side, so
    // the whole pattern lands about 0.016 across from where it was (and not measurably along the stroke). Found by
    // matching the profile of the finest level over level 3 across the patch, and along the dots' row, before and
    // after; the grooves, dots, rolls and roughness are read where it landed.
    let prof1 = across_profile(&v);
    let shift = {
        let mut best = (f32::INFINITY, 0.0f32);
        for k in -40..=40i32 {
            let mut err = 0.0;
            for (j, &before) in prof0.iter().enumerate() {
                let jj = j as i32 + k;
                err += if (0..111).contains(&jj) { (before - prof1[jj as usize]).powi(2) } else { before.powi(2) };
            }
            if err < best.0 {
                best = (err, 0.001 * k as f32);
            }
        }
        best.1
    };
    let prof_along1 = along_profile(&v, dots_row + shift);
    let shift_along = {
        let mut best = (f32::INFINITY, 0.0f32);
        // Within the dots' spacing, which a match along their row would otherwise alias on.
        for k in -8..=8i32 {
            let mut err = 0.0;
            for (j, &before) in prof_along0.iter().enumerate() {
                let jj = j as i32 + k;
                err += if (0..101).contains(&jj) { (before - prof_along1[jj as usize]).powi(2) } else { before.powi(2) };
            }
            if err < best.0 {
                best = (err, 0.001 * k as f32);
            }
        }
        best.1
    };
    let shifted = patch + t2 * shift + t1 * shift_along;
    let grooves1: Vec<Vec<(f32, f32)>> = grooves.iter().map(|&b| groove_profile(&v, 4, shifted + t2 * b, along, across, push, 0.02, 0.0035)).collect();
    let dots1: Vec<f32> = dots
        .iter()
        .map(|&d| {
            let (mut peak, mut rim, mut n) = (f32::NEG_INFINITY, 0.0, 0);
            for i in 0..21 {
                for j in 0..13 {
                    let p = d + t2 * shift + along * (0.0005 * (i as f32 - 10.0)) + across * (0.0004 * (j as f32 - 6.0));
                    let x = h(&v, p, 5);
                    peak = peak.max(x);
                    if i == 0 || i == 20 || j == 0 || j == 12 {
                        rim += x;
                        n += 1;
                    }
                }
            }
            peak - rim / n as f32
        })
        .collect();
    let rough1 = {
        let (mut sum, mut n) = (0.0f32, 0);
        for off in [0.013f32, -0.001, -0.015, -0.028, -0.040] {
            let base = shifted + across * off - along * 0.03;
            let hs: Vec<f32> = (0..25).map(|j| h(&v, base + along * (0.0025 * j as f32), 5)).collect();
            for w in hs.windows(3) {
                let d2 = w[0] - 2.0 * w[1] + w[2];
                sum += d2 * d2;
                n += 1;
            }
        }
        (sum / n as f32).sqrt()
    };
    let rolls1: Vec<f32> = (0..97).map(|j| { let p = shifted + along * 0.08 + across * (-0.12 + 0.0025 * j as f32); h(&v, p, 5) - h(&v, p, 0) }).collect();
    let grain1 = grain(&v, shifted);
    if std::env::var("GROOVE_DUMP").is_ok() {
        eprintln!("   grain by cap before: {:?}", (0..=5).map(|c| format!("{:.7}", grain_at(&v, patch, c))).collect::<Vec<_>>());
        eprintln!("   grain by cap after: {:?}", (0..=5).map(|c| format!("{:.7}", grain_at(&v, shifted, c))).collect::<Vec<_>>());
    }
    eprintln!("the pattern landed {shift:+.4} across and {shift_along:+.4} along; grain (second difference at one level 4 step) {grain0:.7} -> {grain1:.7}");
    let rise = h(&v, patch, 0) - ground0;
    let span = |r: &[f32]| r.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - r.iter().cloned().fold(f32::INFINITY, f32::min);
    let roll_diff = rolls0.iter().zip(rolls1.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let median = |p: &[(f32, f32)]| {
        let mut d: Vec<f32> = p.iter().map(|x| x.0).collect();
        d.sort_by(f32::total_cmp);
        d[d.len() / 2]
    };
    eprintln!("rolls: height {:.4} -> {:.4}, largest change of the profile over level 0 {:.5}", span(&rolls0), span(&rolls1), roll_diff);
    if std::env::var("GROOVE_DUMP").is_ok() {
        eprintln!("   rolls before, in 1e-4: {}", rolls0.iter().map(|x| format!("{:.0}", x * 1e4)).collect::<Vec<_>>().join(" "));
        eprintln!("   rolls after,  in 1e-4: {}", rolls1.iter().map(|x| format!("{:.0}", x * 1e4)).collect::<Vec<_>>().join(" "));
    }
    for (k, (b, a)) in grooves0.iter().zip(grooves1.iter()).enumerate() {
        let least = a.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        let jump = a.windows(2).map(|w| (w[1].1 - w[0].1).abs()).fold(0.0f32, f32::max);
        eprintln!("groove {k}: depth median {:.5} -> {:.5}, least {least:.5}; largest jump of its bottom {jump:.5}", median(b), median(a));
        if std::env::var("GROOVE_DUMP").is_ok() {
            eprintln!("   depth by station before, in 1e-5: {}", b.iter().map(|x| format!("{:.0}", x.0 * 1e5)).collect::<Vec<_>>().join(" "));
            eprintln!("   depth by station, in 1e-5: {}", a.iter().map(|x| format!("{:.0}", x.0 * 1e5)).collect::<Vec<_>>().join(" "));
            eprintln!("   bottom by station, in 1e-4: {}", a.iter().map(|x| format!("{:+.1}", x.1 * 1e4)).collect::<Vec<_>>().join(" "));
        }
    }
    eprintln!("dots: {:.5?} -> {:.5?}", dots0, dots1);
    if std::env::var("GROOVE_DUMP").is_ok() {
        eprintln!("   roughness by cap after: {:?}", (0..=5).map(|c| format!("{:.6}", roughness_at(&v, c))).collect::<Vec<_>>());
    }
    eprintln!("roughness (RMS second difference over 0.0025): {rough0:.6} -> {rough1:.6}; the ground rose {rise:.4} in {ms:.0} ms; blocks per level {blocks0:?} -> {:?}", v.memory().0);
    let (open, _, dropped, _) = check(&v, 5);
    assert_eq!((open, dropped), (0, 0));
    assert!(rise > 0.01, "the clay did not build there");
    assert!(span(&rolls0) > 0.003, "the rolls did not build");
    // The bars: the rolls keep their height and shape; every groove and dot keeps three quarters of its depth
    // along its whole length; and the surface between them gets no rougher than three and a half times what the
    // rolls' own curvature and the clay brush's texture read before the stroke. That last bar is where the carry
    // stands, not where it should be: level 0 under the fine work reads 0.00001 while the stroke lasts, and the
    // rest is a lattice at the block's scale the carry still leaves at the levels between (3.4 times before,
    // against 4.0 for the lattice of diamonds this test was written for).
    assert!(span(&rolls1) > 0.8 * span(&rolls0), "the rolls lost height: {:.5} of {:.5}", span(&rolls1), span(&rolls0));
    // The valley between the rolls fills in part on the brush's falloff: their profile changes by 0.4 of their
    // height here (0.86 for the build this test was written against).
    assert!(roll_diff < 0.5 * span(&rolls0), "the rolls changed shape by {roll_diff:.5} of {:.5}", span(&rolls0));
    for (k, (b, a)) in grooves0.iter().zip(grooves1.iter()).enumerate() {
        let least = a.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        assert!(median(a) > 0.75 * median(b), "groove {k} faded: {:.5} against {:.5}", median(a), median(b));
        assert!(least > 0.75 * median(b), "groove {k} broke: {least:.5} against {:.5}", median(b));
    }
    for (k, (b, a)) in dots0.iter().zip(dots1.iter()).enumerate() {
        assert!(*a > 0.75 * b, "dot {k} faded: {a:.5} against {b:.5}");
    }
    assert!(rough1 < 3.5 * rough0, "the surface got rougher: {rough1:.6} against {rough0:.6}");
}

/// By default a coarse clay stroke buries the fine detail under it as much as it adds clay there, as clay does: bumps
/// under the middle of a big stroke go (it built 0.07 over them), bumps out at the edge of its plateau, where it built
/// about their height, stay.
#[test]
fn big_clay_buries_the_fine_detail_under_it() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
    assert!(!v.keep_detail, "burying is the default");
    // Two rows of bumps at level 4: one under the strokes' line (y = 0), one out on their falloff.
    let rows = [0.0f32, 0.29];
    let xs: Vec<f32> = (0..5).map(|i| -0.024 + 0.012 * i as f32).collect();
    for &y in &rows {
        for &x in &xs {
            v.begin_stroke();
            v.ball_dab(4, Vec3::new(x, y, 1.0).normalize(), 0.004, 0.001, false);
            v.end_stroke();
        }
    }
    // A bump's height over the ground beside it, seen along the strokes' push (+z), best within a little of its place.
    let h = |v: &LevelVolume, x: f32, y: f32| {
        let p = Vec3::new(x, y, 1.0).normalize();
        v.raycast(Vec3::new(p.x, p.y, 3.0), -Vec3::Z, v.levels() - 1, false).map_or(f32::NAN, |(q, _)| q.z)
    };
    let relief = |v: &LevelVolume, y0: f32| {
        xs.iter()
            .map(|&x0| {
                let mut best = 0.0f32;
                for i in 0..21 {
                    for j in 0..9 {
                        let (x, y) = (x0 - 0.004 + 0.001 * j as f32, y0 - 0.01 + 0.001 * i as f32);
                        best = best.max(h(v, x, y) - 0.5 * (h(v, x - 0.006, y) + h(v, x + 0.006, y)));
                    }
                }
                best
            })
            .sum::<f32>()
            / xs.len() as f32
    };
    let before: Vec<f32> = rows.iter().map(|&y| relief(&v, y)).collect();
    let ground0: Vec<f32> = rows.iter().map(|&y| h(&v, 0.06, y)).collect();
    for _ in 0..2 {
        v.begin_stroke();
        let mut along = Vec3::ZERO;
        for k in 0..=8 {
            let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, 0.0, 1.0).normalize();
            let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = v.plane_dab(0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
        }
        v.end_stroke();
    }
    let after: Vec<f32> = rows.iter().map(|&y| relief(&v, y)).collect();
    let rise: Vec<f32> = rows.iter().zip(&ground0).map(|(&y, g)| h(&v, 0.06, y) - g).collect();
    eprintln!("under the strokes' line: {:.5} -> {:.5} (the clay rose {:.4}); on their falloff: {:.5} -> {:.5} (rose {:.4})", before[0], after[0], rise[0], before[1], after[1], rise[1]);
    assert!(before.iter().all(|&b| b > 0.0003), "the dabs did not make bumps");
    assert!(after[0] < 0.25 * before[0], "the bumps under the clay were not buried");
    assert!(after[1] > 0.5 * before[1], "the bumps out on the falloff should mostly stay");
    let (open, _, dropped, _) = check(&v, 4);
    assert_eq!((open, dropped), (0, 0));
}

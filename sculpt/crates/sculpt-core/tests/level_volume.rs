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
    // And each groove's depth from end to end (0.001 apart, the deepest within a step across), for the shift along
    // them: the ends, where the depth falls to nothing, are the structure to match on.
    let depth_along = |v: &LevelVolume, b: f32| -> Vec<f32> {
        (0..91)
            .map(|i| {
                let s = -0.045 + 0.001 * i as f32;
                (-2..=2).map(|j| { let p = patch + t2 * b + along * s + across * (0.0005 * j as f32); h(v, p, 3) - h(v, p, 5) }).fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    };
    let grooves_along0: Vec<Vec<f32>> = grooves.iter().map(|&b| depth_along(&v, b)).collect();
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
    // Along the stroke, by matching the grooves' depth from end to end before and after, within a dot's spacing (a
    // profile along the dots' row instead aliased on their spacing, and read nothing where the row missed them).
    let shift_along = {
        let after: Vec<Vec<f32>> = grooves.iter().map(|&b| depth_along(&v, b + shift)).collect();
        let mut best = (f32::INFINITY, 0.0f32);
        for k in -8..=8i32 {
            let (mut err, mut n) = (0.0, 0);
            for (b, a) in grooves_along0.iter().zip(after.iter()) {
                for (j, &d0) in b.iter().enumerate() {
                    let jj = j as i32 + k;
                    if (0..91).contains(&jj) {
                        err += (d0 - a[jj as usize]).powi(2);
                        n += 1;
                    }
                }
            }
            let err = err / n as f32;
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
        // A groove's shallowest station keeps two thirds of its depth (0.75 to 0.85 of the median before, here; a
        // station reads a fifth of a level 4 step either way between builds).
        assert!(least > 0.65 * median(b), "groove {k} broke: {least:.5} against {:.5}", median(b));
    }
    for (k, (b, a)) in dots0.iter().zip(dots1.iter()).enumerate() {
        assert!(*a > 0.75 * b, "dot {k} faded: {a:.5} against {b:.5}");
    }
    // Held to what it was when the bar was 3.5 times the roughness before (0.000147 then, against an allowance of
    // 0.000182): the first big stroke's falloff no longer leaves a row of pits over the patch, so the roughness
    // before is lower (0.000036, from 0.000052) and the ratio no longer measures this stroke alone.
    assert!(rough1 < 0.00015, "the surface got rougher: {rough1:.6} against {rough0:.6}");
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

/// The slope of level 0's field at its samples within a step of the surface, in a box of `half` around `at`, as a
/// histogram in quarters (0-0.25, 0.25-0.5, ..., 1.75 and up).
fn slope_histogram(v: &LevelVolume, at: Vec3, half: f32) -> [usize; 8] {
    let vx = v.voxel(0);
    let (lo, hi) = (((at - Vec3::splat(half) - v.origin()) / vx).floor().as_ivec3(), ((at + Vec3::splat(half) - v.origin()) / vx).ceil().as_ivec3());
    let mut hist = [0usize; 8];
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let i = glam::IVec3::new(x, y, z);
                if v.sample(0, i, false).abs() > 0.7 * vx {
                    continue;
                }
                let d = |o: glam::IVec3| v.sample(0, i + o, false) - v.sample(0, i - o, false);
                let g = Vec3::new(d(glam::IVec3::X), d(glam::IVec3::Y), d(glam::IVec3::Z)).length() / (2.0 * vx);
                hist[((g / 0.25) as usize).min(7)] += 1;
            }
        }
    }
    hist
}

/// Level 0 once measured distances only six steps out from the surface when a stroke ended, farther samples holding
/// that as a bound (it now measures them out to its band, sixteen steps). A big clay stroke over a curved form has its
/// plane well beyond six steps above the surface at its rim (here 0.1, twenty steps), so its dabs build the new surface
/// out of samples that held the bound. Blended from the bound, the field around the new surface had the dab's weight
/// for a slope instead of one, and the surface showed a lattice of facets along the rim, mid-stroke and after: the
/// field near the new surface must keep its unit slope.
#[test]
fn a_big_clay_stroke_keeps_unit_slope_beyond_level_zeros_measured_distances() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 4);
    // Out on the big brush's falloff: 0.33 from the front, where the ball falls 0.056 below the brush's plane and the
    // brush still moves the surface a quarter of the way per dab.
    let rim = Vec3::new(0.33, 0.0, (1.0f32 - 0.33 * 0.33).sqrt());
    // An earlier, smaller stroke there: when it ends, the blocks it touched (to 0.1 and more above the surface) hold
    // distances out to the band, sixteen steps from the surface.
    stroke(&mut v, 0, (rim + Vec3::new(-0.05, -0.03, 0.0)).normalize(), (rim + Vec3::new(0.05, 0.03, 0.0)).normalize(), 0.3, 0.3, false);
    // Where level 0 once held only its bound.
    let bound = 6.0 * v.voxel(0);
    let (on_rim, n_rim) = v.raycast(rim * 3.0, -rim, 0, false).unwrap();
    let far: Vec<f32> = (7..=14).map(|k| v.value_upto(on_rim + n_rim * (k as f32 * v.voxel(0)), 0, false) / v.voxel(0)).collect();
    assert!(far.iter().zip(7..).all(|(&d, k)| (d - k as f32).abs() < 0.3), "the earlier stroke should leave distances seven to fourteen steps out, in steps: {far:?}");
    // The big stroke: three dabs of a brush of radius 0.45 at the front, its plane 0.045 above the surface at its
    // centre and 0.1 (twenty steps) above it at `rim`: the surface there ends up ten steps out, past the bound.
    let front = Vec3::new(0.0, 0.0, 1.0);
    let (hit, _) = v.raycast(front * 3.0, -front, 0, false).unwrap();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let mut mid = [0usize; 8];
    for k in 0..3 {
        let at = (hit + Vec3::new(0.01 * k as f32, 0.0, 0.0)).normalize();
        let rough = if along == Vec3::ZERO { v.normal_upto(at, 0, true) } else { along };
        let (n, dab) = v.plane_dab(0, at, rough, 0.45, 0.045, 0.5, 0.4, false).unwrap();
        along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        mid = slope_histogram(&v, rim, 0.06);
        eprintln!("dab {k} ({:.1} ms): level 0 slope near the surface around the rim, in quarters: {mid:?}", dab.ms);
    }
    v.end_stroke();
    let end = slope_histogram(&v, rim, 0.06);
    let rise = v.raycast(rim * 3.0, -rim, 0, false).map_or(f32::NAN, |(p, _)| (p - on_rim).dot(n_rim));
    eprintln!("after the stroke: {end:?}; the clay rose {rise:.4} at the rim (the bound is {bound})");
    let (open, _, dropped, _) = check(&v, 3);
    assert_eq!((open, dropped), (0, 0));
    assert!(rise > bound + 0.01, "the clay did not build beyond the bound at the rim: {rise}");
    for (name, h) in [("mid-stroke", mid), ("after", end)] {
        let total: usize = h.iter().sum();
        let low: usize = h[..3].iter().sum();
        assert!(total > 500, "{name}: too few samples near the surface: {total}");
        assert!(low * 20 < total, "{name}: {low} of {total} samples near the new surface have a slope under 0.75: {h:?}");
    }
}

/// How far the finest level's surface stands from the stroke's own level's, over the brush: a stale piece of old
/// surface left standing on the trimmed ground (or a ledge under clay) reads as a big gap, and fine work as a small one.
#[test]
#[ignore]
fn measure_steps_under_a_big_stroke() {
    for keep in [false, true] {
        for trim in [true, false] {
            let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.01, 6);
            v.keep_detail = keep;
            let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
            let (t1, t2) = patch.any_orthonormal_pair();
            let at = |a: f32, b: f32| (patch + t1 * a + t2 * b).normalize();
            let dab = |v: &mut LevelVolume, l: usize, d: Vec3, r: f32, trim: bool, along: &mut Vec3| {
                let Some((hit, n)) = v.raycast(d * 3.0, -d, l, true) else { return };
                let rough = if *along == Vec3::ZERO { n } else { *along };
                if let Some((n, _)) = v.plane_dab(l, hit, rough, r, r * if trim { 0.075 } else { 0.095 }, 0.3, 0.4, trim) {
                    *along = if *along == Vec3::ZERO { n } else { (*along * 0.5 + n * 0.5).normalize() };
                }
            };
            // Fine work: level 4 clay dabs scattered over the patch.
            for i in 0..40 {
                v.begin_stroke();
                let mut along = Vec3::ZERO;
                dab(&mut v, 4, at(-0.06 + 0.003 * i as f32, 0.04 * ((i * 7 % 11) as f32 / 11.0 - 0.5)), 0.006, false, &mut along);
                v.end_stroke();
            }
            let h = |v: &LevelVolume, p: Vec3, cap: usize| v.raycast(p + patch * 2.0, -patch, cap, false).map_or(f32::NAN, |(q, _)| q.dot(patch));
            let gap = |v: &LevelVolume, l: usize| {
                let mut worst = 0.0f32;
                for i in 0..21 {
                    for j in 0..21 {
                        let p = patch + t1 * (-0.2 + 0.02 * i as f32) + t2 * (-0.2 + 0.02 * j as f32);
                        let (a, b) = (h(v, p, 5), h(v, p, l));
                        if a.is_finite() && b.is_finite() {
                            worst = worst.max((a - b).abs());
                        }
                    }
                }
                worst
            };
            for l in [0usize, 1, 2] {
                let r = [0.4f32, 0.12, 0.06][l];
                let before = gap(&v, l);
                v.begin_stroke();
                let mut along = Vec3::ZERO;
                let mut mid = 0.0f32;
                for i in 0..14 {
                    dab(&mut v, l, at(-0.15 + 0.022 * i as f32, 0.0), r, trim, &mut along);
                    mid = mid.max(gap(&v, l));
                }
                v.end_stroke();
                eprintln!("keep {keep} trim {trim} level {l}: gap to the stroke's level before {before:.4}, worst during {mid:.4}, after {:.4}", gap(&v, l));
            }
        }
    }
}

/// A fresh 1M-triangle ball and one wide clay stroke at level 0 (the app's first stroke), as a shaded picture of the
/// result (a PGM at `SCULPT_PICTURE`): its rim should be as smooth as the brush's own falloff.
#[test]
#[ignore]
fn picture_a_wide_clay_stroke_on_a_fresh_ball() {
    let Ok(path) = std::env::var("SCULPT_PICTURE") else { return };
    let voxel = (4.0 * std::f32::consts::PI / (1e6f32 / 2.0)).sqrt();
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 6);
    let r: f32 = std::env::var("SCULPT_R").ok().and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let patch = Vec3::new(0.0, 0.3, 1.0).normalize();
    let (t1, _) = patch.any_orthonormal_pair();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let steps: usize = std::env::var("SCULPT_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    for i in 0..=steps {
        let d = (patch + t1 * (-0.6 + 1.2 * i as f32 / steps as f32)).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        let trim = std::env::var_os("SCULPT_TRIM").is_some();
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, r * if trim { 0.075 } else { 0.095 }, 0.2, 0.4, trim) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
    }
    v.end_stroke();
    let w = patch;
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    // `SCULPT_WIN=a,b,half`: only that window (default the whole ball), at 400 pixels, and the wall's ripple as a number.
    let win: Vec<f32> = std::env::var("SCULPT_WIN").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let (n, half, (ca, cb)) = if win.len() == 3 { (400, win[2], (win[0], win[1])) } else { (800, 0.8f32, (0.0, 0.0)) };
    if win.len() == 3 {
        // Height over a grid half a step apart; the ripple is the mean absolute second difference along both axes.
        let g = 0.5 * voxel;
        let m = (2.0 * half / g) as usize;
        let hts: Vec<f32> = (0..m * m).map(|k| { let (i, j) = (k % m, k / m); v.raycast(w * 4.0 + u * (ca - half + g * i as f32) + t * (cb - half + g * j as f32), -w, 0, false).map_or(f32::NAN, |(q, _)| q.dot(w)) }).collect();
        let (mut sum, mut across, mut along, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for j in 1..m - 1 { for i in 1..m - 1 {
            let c = hts[j * m + i];
            let (da, db) = ((hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs(), (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs());
            let d = da + db;
            if d.is_finite() { sum += d as f64; across += da as f64; along += db as f64; cnt += 1; }
        } }
        let steps = |x: f64| x / cnt as f64 / voxel as f64;
        eprintln!("RIPPLE mean |second difference| {:.3e} (in steps of {voxel:.4}: {:.4}; across the wall {:.4}, along it {:.4}) over {cnt} points", sum / cnt as f64, steps(sum), steps(across), steps(along));
    }
    let mut img = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let (a, b) = (ca - half + 2.0 * half * x as f32 / n as f32, cb + half - 2.0 * half * y as f32 / n as f32);
            if let Some((_, nrm)) = v.raycast(w * 4.0 + u * a + t * b, -w, 0, false) {
                img[y * n + x] = (35.0 + 220.0 * nrm.dot((w * 0.8 + u * -0.4 + t * 0.5).normalize()).max(0.0)) as u8;
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(path, out).unwrap();
}

/// The lattice a wide clay stroke's wall picks up at level 0 when it is scrubbed densely (sixty dabs along 1.2 of a
/// coarse ball, a brush of radius 0.7 so its plane stands above level 0's measured distances): the mean |second
/// difference| of the surface's height along the wall (where the wall's own shape is flat), in steps, sampled half a
/// step apart over a window on the wall. Samples holding only the bound on their distance read the ground under the
/// brush once; read again each dab (as any sample near the bound once was), they were set back to the surface's
/// distance while their neighbours accumulated the dabs', and the wall's foot came out dotted with a lattice of pits
/// (0.088 steps along the wall here, against 0.026 reading the ground once).
#[test]
fn a_densely_scrubbed_clay_stroke_leaves_no_lattice_on_its_wall() {
    let voxel = 0.01f32;
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 3);
    let r = 0.7f32;
    let patch = Vec3::new(0.0, 0.3, 1.0).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let steps = 60;
    for i in 0..=steps {
        let d = (patch + t1 * (-0.6 + 1.2 * i as f32 / steps as f32)).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, r * 0.095, 0.2, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
    }
    v.end_stroke();
    // A window across the wall (the rim is 0.7 to the side of the stroke's line), at the stroke's middle.
    let (half, g) = (0.3 * r, 0.5 * voxel);
    let m = (2.0 * half / g) as usize;
    let height = |a: f32, b: f32| v.raycast(patch * 4.0 + t2 * (-0.85 * r + a) + t1 * b, -patch, 0, false).map_or(f32::NAN, |(q, _)| q.dot(patch));
    let hts: Vec<f32> = (0..m * m).map(|k| height(-half + g * (k % m) as f32, -half + g * (k / m) as f32)).collect();
    let (mut across, mut along, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = hts[j * m + i];
            let (da, db) = ((hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs(), (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs());
            if (da + db).is_finite() {
                across += da as f64;
                along += db as f64;
                cnt += 1;
            }
        }
    }
    let steps = |x: f64| x / cnt as f64 / voxel as f64;
    eprintln!("wall ripple: across it {:.4}, along it {:.4} steps over {cnt} points", steps(across), steps(along));
    assert!(cnt > m * m / 2, "the window missed the surface: {cnt} of {} points", m * m);
    assert!(steps(along) < 0.05, "the wall is rippled along its length: mean |second difference| {:.4} steps", steps(along));
}

/// A clay stroke as wide as the ball (the app's first strokes, seen from far: radius 0.9 on a ball of radius 1) lays a
/// shell that follows the ball. Laid to the brush's plane, each dab was a flat disc tilted with its own normal, the
/// ball curving away 0.4 under its rim: the dabs stacked as terraces, with block-sized stairs where they met the ball.
#[test]
fn a_clay_stroke_as_wide_as_the_ball_lays_a_shell_that_follows_it() {
    let voxel = 0.01f32;
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 3);
    let r = 0.9f32;
    let h = 0.095 * r;
    let patch = Vec3::new(0.0, 0.3, 1.0).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    for i in 0..=12 {
        let d = (patch + t1 * (-0.2 + 0.4 * i as f32 / 12.0)).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, h, 0.2, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
    }
    v.end_stroke();
    // Heights over the ball on a grid two steps apart across the brush: nowhere much more than the brush's height,
    // and smooth (the mean |second difference| in steps).
    let (half, g) = (0.6f32, 2.0 * voxel);
    let m = (2.0 * half / g) as usize;
    let height = |a: f32, b: f32| {
        let d = (patch + t1 * a + t2 * b).normalize();
        v.raycast(d * 3.0, -d, 0, false).map_or(f32::NAN, |(q, _)| q.length() - 1.0)
    };
    let hts: Vec<f32> = (0..m * m).map(|k| height(-half + g * (k % m) as f32, -half + g * (k / m) as f32)).collect();
    let (mut sum, mut cnt, mut most) = (0.0f64, 0usize, 0.0f32);
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = hts[j * m + i];
            most = most.max(c);
            let d = (hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs() + (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs();
            if d.is_finite() {
                sum += d as f64;
                cnt += 1;
            }
        }
    }
    let ripple = sum / cnt as f64 / voxel as f64;
    eprintln!("the shell is at most {most:.4} high (the brush's height {h:.4}); ripple {ripple:.4} steps over {cnt} points");
    assert!(most > 0.5 * h, "the stroke built nothing: {most:.4}");
    assert!(most < 1.5 * h, "the stroke piled up beyond its height: {most:.4} against {h:.4}");
    assert!(ripple < 0.3, "the shell is terraced: mean |second difference| {ripple:.4} steps");
}

/// Far strokes over earlier work (a session's): fine clay dabs over a patch, then overlapping level 0 clay strokes of
/// mixed width over it, then a small brush and a big one. Prints each stroke's dab cost and blocks changed, the
/// surface's bumpiness over the patch, and with `SCULPT_PICTURE` a shaded picture of it.
#[test]
#[ignore]
fn far_mixed_strokes_over_fine_work() {
    let voxel = (4.0 * std::f32::consts::PI / (1e6f32 / 2.0)).sqrt();
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 6);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    let at = |a: f32, b: f32| (patch + t1 * a + t2 * b).normalize();
    let dab = |v: &mut LevelVolume, l: usize, d: Vec3, r: f32, along: &mut Vec3| -> Option<(f64, usize)> {
        let (hit, n) = v.raycast(d * 3.0, -d, l, true)?;
        let rough = if *along == Vec3::ZERO { n } else { *along };
        let (n, s) = v.plane_dab(l, hit, rough, r, r * 0.095, 0.2, 0.4, false)?;
        *along = if *along == Vec3::ZERO { n } else { (*along * 0.5 + n * 0.5).normalize() };
        Some((s.ms, s.changed + s.carried))
    };
    for i in 0..40 {
        v.begin_stroke();
        let mut along = Vec3::ZERO;
        dab(&mut v, 4, at(-0.06 + 0.003 * i as f32, 0.04 * ((i * 7 % 11) as f32 / 11.0 - 0.5)), 0.006, &mut along);
        v.end_stroke();
    }
    let only: Option<usize> = std::env::var("SCULPT_ONLY").ok().and_then(|s| s.parse().ok());
    let strokes = [(0.45f32, -0.05f32), (0.25, 0.03), (0.6, -0.02), (0.33, 0.08), (0.18, 0.0), (0.5, 0.05), (0.08, 0.01), (0.7, -0.04)];
    for (k, &(r, b)) in strokes.iter().enumerate() {
        if only.is_some_and(|o| o != k) || std::env::var("SCULPT_UPTO").ok().and_then(|s| s.parse::<usize>().ok()).is_some_and(|u| k > u) {
            continue;
        }
        let steps = ((0.4 / (0.15 * r)).ceil() as usize).max(1);
        v.begin_stroke();
        let mut along = Vec3::ZERO;
        let (mut ms, mut most, mut blocks) = (0.0f64, 0.0f64, 0usize);
        for i in 0..=steps {
            let a = -0.2 + 0.4 * i as f32 / steps as f32;
            if let Some((m, n)) = dab(&mut v, 0, at(if k % 2 == 0 { a } else { -a }, b + 0.05 * a), r, &mut along) {
                ms += m;
                most = most.max(m);
                blocks = blocks.max(n);
            }
        }
        let t = std::time::Instant::now();
        v.end_stroke();
        eprintln!("stroke {k} r {r}: {} dabs, {:.1} ms a dab (most {most:.1}), up to {blocks} blocks a dab; its end {:.0} ms", steps + 1, ms / (steps + 1) as f64, t.elapsed().as_secs_f64() * 1e3);
    }
    let cap = v.levels() - 1;
    let (half, g) = (0.3f32, 0.5 * voxel);
    let m = (2.0 * half / g) as usize;
    let height = |a: f32, b: f32| v.raycast(patch * 4.0 + t1 * a + t2 * b, -patch, cap, false).map_or(f32::NAN, |(q, _)| q.dot(patch));
    let hts: Vec<f32> = (0..m * m).map(|k| height(-half + g * (k % m) as f32, -half + g * (k / m) as f32)).collect();
    // Bumpiness: the mean |second difference| in steps, and specks: points where it is over a fifth of a step.
    let (mut sum, mut cnt, mut specks) = (0.0f64, 0usize, 0usize);
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = hts[j * m + i];
            let d = (hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs() + (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs();
            if d.is_finite() {
                sum += d as f64;
                cnt += 1;
                if d > 0.2 * voxel {
                    specks += 1;
                }
            }
        }
    }
    eprintln!("BUMPS mean |second difference| {:.4} steps; {specks} specks over {cnt} points", sum / cnt as f64 / voxel as f64);
    // `SCULPT_BUMPS`: the second difference as a picture (grey is flat, a speck is a dark or light dot).
    if let Ok(path) = std::env::var("SCULPT_BUMPS") {
        let mut img = vec![128u8; m * m];
        for j in 1..m - 1 {
            for i in 1..m - 1 {
                let d = hts[j * m + i - 1] + hts[j * m + i + 1] + hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 4.0 * hts[j * m + i];
                img[(m - 1 - j) * m + i] = if d.is_finite() { (128.0 + 400.0 * d / voxel).clamp(0.0, 255.0) as u8 } else { 0 };
            }
        }
        let mut out = format!("P5 {m} {m} 255
").into_bytes();
        out.extend(img);
        std::fs::write(path, out).unwrap();
    }
    if let Ok(path) = std::env::var("SCULPT_PICTURE") {
        let n = 600;
        let light = (t1 * -0.6 + t2 * 0.8).normalize();
        let mut img = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let (a, b) = (-half + 2.0 * half * x as f32 / n as f32, half - 2.0 * half * y as f32 / n as f32);
                // Shaded by the normal's tilt from the ball's (so only what the strokes made shows), steeply.
                if let Some((q, nrm)) = v.raycast(patch * 4.0 + t1 * a + t2 * b, -patch, cap, false) {
                    img[y * n + x] = (128.0 + 250.0 * (nrm - q.normalize()).dot(light)).clamp(0.0, 255.0) as u8;
                }
            }
        }
        let mut out = format!("P5 {n} {n} 255\n").into_bytes();
        out.extend(img);
        std::fs::write(path, out).unwrap();
    }
}

/// A wide clay stroke at level 0 whose brush is not quite wide enough to follow the form (radius 0.5 on a ball of 1,
/// seen from far) leaves no pits along its dabs' rims. Out on its falloff the ball curves away from the brush's plane
/// and the samples there hold only level 0's bound on their distance; the ground under the brush says how far, and
/// was taken only when it came within 0.9 of the bound (the bound otherwise), a jump of a tenth of the bound between
/// neighbouring samples where the ground crossed that line: each dab left a dotted row of pits along its rim (175 over
/// this window, against none now).
#[test]
fn a_wide_clay_strokes_falloff_leaves_no_pits() {
    let voxel = 0.005f32;
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 2);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    let at = |a: f32, b: f32| (patch + t1 * a + t2 * b).normalize();
    let r = 0.5f32;
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let steps = (0.4 / (0.15 * r)).ceil() as usize;
    for i in 0..=steps {
        let a = 0.2 - 0.4 * i as f32 / steps as f32;
        let Some((hit, n)) = v.raycast(at(a, 0.05 + 0.05 * a) * 3.0, -at(a, 0.05 + 0.05 * a), 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, r * 0.095, 0.2, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
    }
    v.end_stroke();
    // Heights half a step apart over the window; a pit is a point whose |second difference| is over a fifth of a step.
    let (half, g) = (0.3f32, 0.5 * voxel);
    let m = (2.0 * half / g) as usize;
    let height = |a: f32, b: f32| v.raycast(patch * 4.0 + t1 * a + t2 * b, -patch, 0, false).map_or(f32::NAN, |(q, _)| q.dot(patch));
    let hts: Vec<f32> = (0..m * m).map(|k| height(-half + g * (k % m) as f32, -half + g * (k / m) as f32)).collect();
    let (mut cnt, mut pits) = (0usize, 0usize);
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = hts[j * m + i];
            let d = (hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs() + (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs();
            if d.is_finite() {
                cnt += 1;
                pits += (d > 0.2 * voxel) as usize;
            }
        }
    }
    eprintln!("{pits} pits over {cnt} points");
    assert!(cnt > m * m / 2, "the window missed the surface: {cnt} of {} points", m * m);
    assert!(pits <= 10, "the stroke's falloff is pitted: {pits} points over a fifth of a step");
}

/// Level 0 samples between `lo` and `hi` steps from the surface (by their values) that stand out from their neighbours
/// along all three axes, each second difference past `by` steps the same way (a sample off by its own amount), and how
/// many samples were looked at.
fn grain(v: &LevelVolume, lo: f32, hi: f32, by: f32) -> (usize, usize) {
    use glam::IVec3;
    let vx = v.voxel(0);
    let (mut odd, mut all) = (0, 0);
    for bc in v.level_keys(0) {
        let b = v.block((0, bc)).unwrap();
        for (k, &s) in b.iter().enumerate() {
            if s.abs() < lo * vx || s.abs() > hi * vx {
                continue;
            }
            all += 1;
            let i = bc * 8 + IVec3::new((k % 8) as i32, ((k / 8) % 8) as i32, (k / 64) as i32);
            let second = [IVec3::X, IVec3::Y, IVec3::Z].map(|d| (v.sample(0, i + d, false) + v.sample(0, i - d, false) - 2.0 * s) / vx);
            if second.iter().all(|&x| x < -by) || second.iter().all(|&x| x > by) {
                odd += 1;
            }
        }
    }
    (odd, all)
}

/// A clay stroke at level 0 at the app's default strength (seen from far) along the polyline `path` (offsets along the
/// patch's tangents), its dabs spaced as the app spaces them.
fn far_clay_stroke(v: &mut LevelVolume, patch: Vec3, path: &[(f32, f32)], r: f32, hardness: f32) -> f64 {
    let (t1, t2) = patch.any_orthonormal_pair();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let mut pts = vec![path[0]];
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let steps = ((((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt() / (0.15 * r)).ceil() as usize).max(1);
        pts.extend((1..=steps).map(|i| (a.0 + (b.0 - a.0) * i as f32 / steps as f32, a.1 + (b.1 - a.1) * i as f32 / steps as f32)));
    }
    for (x, y) in pts {
        let d = (patch + t1 * x + t2 * y).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        // Strength 0.4: a height of 0.047 of the radius, 0.22 of the way each dab.
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, r * 0.047, hardness, 0.22, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
    }
    v.end_stroke().ms
}

/// A wide clay stroke over an earlier one (at level 0, seen from far) brings up no grain. The stroke's end makes the
/// samples off the surface distances again; measured to the nearest point where the surface crossed a lattice edge,
/// each was off by its own part of the crossings' spacing, up to half a step. That grain lay off the surface unseen
/// (236 samples standing out from their neighbours along all three axes here, against 6 now) until the next stroke
/// raised the surface onto it, where each such sample shades as a star-shaped pit or bump (455 at the surface after
/// the second stroke here, against 3 now). `SCULPT_PICTURE` writes the surface, flat shaded.
#[test]
fn a_clay_stroke_over_earlier_clay_brings_up_no_grain() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.01, 2);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    far_clay_stroke(&mut v, patch, &[(-0.3, -0.1), (0.3, 0.1)], 0.5, 0.5);
    let (off, looked) = grain(&v, 1.5, 5.0, 0.3);
    eprintln!("after the first stroke, {off} of {looked} samples off the surface stand out");
    assert!(looked > 100_000, "too few samples off the surface: {looked}");
    assert!(off <= 25, "the field off the surface is grainy: {off} samples stand out");
    far_clay_stroke(&mut v, patch, &[(-0.3, 0.05), (0.3, -0.05)], 0.5, 0.5);
    let (on, looked) = grain(&v, 0.0, 1.5, 0.3);
    eprintln!("after the second, {on} of {looked} samples at the surface stand out");
    if let Ok(path) = std::env::var("SCULPT_PICTURE") {
        mesh_picture(&v, 0, patch, 0.3, 900, &path);
    }
    assert!(on <= 30, "the second stroke brought up grain: {on} samples at the surface stand out");
}

/// A sample an earlier dab of the stroke blended keeps blending from its own value, even where that came out at level
/// 0's bound. Blended a little, out on a wide brush's falloff, a sample far off the surface can land within what
/// holding it at 16 bits would move the bound; read as the bound again, the next dab took the ground under the brush
/// for its distance instead while its neighbours went on from their own, and it jumped from them by up to half a step:
/// once the clay reached it, a pit. Here: over a stroke scrubbed back and forth, the samples a dab left within a
/// hundredth of a step of the bound, and how many of them the next dab moved unlike their neighbours the stroke
/// blended too (by more than a fifth of a step from their mean change): 3415 of 16749 when they were read as the bound,
/// 9 of 16981 when they were not but level 0 held distances only six steps out (its neighbours past that read the
/// ground under the brush), none of 30679 now that it holds them out to its band.
#[test]
fn a_sample_the_stroke_blended_onto_the_bound_keeps_its_own_distance() {
    use glam::IVec3;
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.01, 2);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    // An earlier stroke, ended: its samples off the surface hold the bound.
    far_clay_stroke(&mut v, patch, &[(-0.3, -0.1), (0.3, 0.1)], 0.5, 0.5);
    let (t1, t2) = patch.any_orthonormal_pair();
    let (vx, plateau) = (v.voxel(0), v.plateau(0));
    let start = v.preview_copy();
    let r = 0.5f32;
    let mut pts = Vec::new();
    for leg in 0..6 {
        for i in 0..=8 {
            let f = i as f32 / 8.0;
            let f = if leg % 2 == 0 { f } else { 1.0 - f };
            pts.push((-0.3 + 0.6 * f, 0.05 - 0.1 * f));
        }
    }
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let (mut onto, mut jumped) = (0usize, 0usize);
    let mut watch: Vec<IVec3> = Vec::new();
    for (x, y) in pts {
        let d = (patch + t1 * x + t2 * y).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let before = v.preview_copy();
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(0, hit, rough, r, r * 0.047, 0.5, 0.22, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
        // How the samples left on the bound last dab moved this one, against their neighbours the stroke blended too
        // (a neighbour still holding the bound reads the ground, as it should).
        let change = |i: IVec3| v.sample(0, i, false) - before.sample(0, i, false);
        for &i in &watch {
            let blended: Vec<f32> = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z].iter().map(|&o| i + o).filter(|&j| before.sample(0, j, false) != start.sample(0, j, false)).map(change).collect();
            if blended.len() >= 2 && (change(i) - blended.iter().sum::<f32>() / blended.len() as f32).abs() > 0.2 * vx {
                jumped += 1;
            }
        }
        // The samples this dab blended that came out within a hundredth of a step of the bound.
        watch.clear();
        for bc in v.level_keys(0) {
            let b = v.block((0, bc)).unwrap();
            for (k, &s) in b.iter().enumerate() {
                let i = bc * 8 + IVec3::new((k % 8) as i32, ((k / 8) % 8) as i32, (k / 64) as i32);
                if (s.abs() - plateau).abs() <= 0.01 * vx && s != before.sample(0, i, false) && s != start.sample(0, i, false) {
                    watch.push(i);
                }
            }
        }
        onto += watch.len();
    }
    v.end_stroke();
    eprintln!("{onto} samples blended onto the bound; {jumped} of them jumped from their neighbours on the next dab");
    assert!(onto > 0, "no sample was blended onto the bound: the test no longer looks at anything");
    assert!(jumped <= 2, "{jumped} samples blended onto the bound jumped from their neighbours on the next dab");
}

/// A wide clay stroke raising the surface over the samples off an earlier one's (seen from far, on the app's new ball)
/// shows no crease where the two met. Level 0 held distances only six steps out from the surface when a stroke ended,
/// and a plane dab read the samples beyond from the ground under the brush instead, a guess that near an earlier
/// stroke's wall came out unlike its neighbours' own distances: the field kinked along the line six steps off the
/// earlier surface, and once the later stroke's clay reached it, a beaded crease ran across it. Here: three crossing
/// strokes, and over a window across where the third rises over the earlier two, the points where the surface's
/// height bends by more than 0.08 of a step (its second differences half a step apart, summed over both axes): 13 with
/// the guess (4 past a tenth of a step), 5 now (none).
#[test]
fn a_wide_stroke_over_earlier_ones_shows_no_crease_where_they_met() {
    // The app's new ball: level 0 at twice the step a million-triangle ball names.
    let voxel = 2.0 * (4.0 * std::f32::consts::PI / (1e6f32 / 2.0)).sqrt();
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 7);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    for path in [[(-0.5, -0.2), (0.4, -0.15)], [(-0.3, 0.4), (0.05, -0.15)], [(0.4, 0.35), (-0.1, 0.1)]] {
        far_clay_stroke(&mut v, patch, &path, 0.5, 0.5);
    }
    let (t1, t2) = patch.any_orthonormal_pair();
    let (ca, cb, half, g) = (0.36f32, 0.24f32, 0.15f32, 0.5 * voxel);
    let m = (2.0 * half / g) as usize;
    let hts: Vec<f32> = (0..m * m).map(|k| v.raycast(patch * 4.0 + t1 * (ca - half + g * (k % m) as f32) + t2 * (cb - half + g * (k / m) as f32), -patch, 0, false).map_or(f32::NAN, |(q, _)| q.dot(patch))).collect();
    let (mut cnt, mut bent) = (0usize, 0usize);
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = hts[j * m + i];
            let d = (hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs() + (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs();
            if d.is_finite() {
                cnt += 1;
                bent += (d > 0.08 * voxel) as usize;
            }
        }
    }
    eprintln!("{bent} of {cnt} points bend by more than 0.08 of a step");
    assert!(cnt > m * m / 2, "the window missed the surface: {cnt} of {} points", m * m);
    assert!(bent <= 7, "a crease where the strokes met: {bent} points bend by more than 0.08 of a step");
}

/// The extracted surface (levels up to `cap`) seen straight down `-w` over a square `half` across around `w`, flat
/// shaded with a grazing light (each face its own shade), as an `n` by `n` PGM at `path`.
fn mesh_picture(v: &LevelVolume, cap: usize, w: Vec3, half: f32, n: usize, path: &str) {
    let (mesh, _) = v.surface(cap);
    let (u, t) = w.any_orthonormal_pair();
    let light = (w * 0.45 + u * 0.7 + t * 0.55).normalize();
    let mut depth = vec![f32::NEG_INFINITY; n * n];
    let mut img = vec![0u8; n * n];
    let px = |p: Vec3| ((p.dot(u) + half) / (2.0 * half) * n as f32, (half - p.dot(t)) / (2.0 * half) * n as f32, p.dot(w));
    for f in &mesh.faces {
        let nrm = sculpt_core::mesh::face_normal(&mesh.positions, f).normalize_or_zero();
        if nrm.dot(w) <= 0.0 {
            continue;
        }
        let shade = (30.0 + 225.0 * nrm.dot(light).max(0.0)) as u8;
        for tri in sculpt_core::mesh::face_triangles(f) {
            let [a, b, c] = tri.map(|i| px(mesh.positions[i as usize]));
            let (x0, x1) = (a.0.min(b.0).min(c.0).floor().max(0.0) as i32, a.0.max(b.0).max(c.0).ceil().min(n as f32 - 1.0) as i32);
            let (y0, y1) = (a.1.min(b.1).min(c.1).floor().max(0.0) as i32, a.1.max(b.1).max(c.1).ceil().min(n as f32 - 1.0) as i32);
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area.abs() < 1e-12 {
                continue;
            }
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (sx, sy) = (x as f32 + 0.5, y as f32 + 0.5);
                    let l1 = ((b.0 - sx) * (c.1 - sy) - (c.0 - sx) * (b.1 - sy)) / area;
                    let l2 = ((c.0 - sx) * (a.1 - sy) - (a.0 - sx) * (c.1 - sy)) / area;
                    let l3 = 1.0 - l1 - l2;
                    if l1 < 0.0 || l2 < 0.0 || l3 < 0.0 {
                        continue;
                    }
                    let z = l1 * a.2 + l2 * b.2 + l3 * c.2;
                    let k = y as usize * n + x as usize;
                    if z > depth[k] {
                        depth[k] = z;
                        img[k] = shade;
                    }
                }
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(path, out).unwrap();
    // And as the app shades it, by the field's gradient where the field's own surface lies (`path` with `.grad.pgm`).
    let h = 0.02 * v.voxel(0);
    let mut img = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let (a, b) = (-half + 2.0 * half * (x as f32 + 0.5) / n as f32, half - 2.0 * half * (y as f32 + 0.5) / n as f32);
            if let Some((q, _)) = v.raycast(w * 4.0 + u * a + t * b, -w, cap, false) {
                let d = |o: Vec3| v.value_upto(q + o, cap, false) - v.value_upto(q - o, cap, false);
                let nrm = Vec3::new(d(Vec3::X * h), d(Vec3::Y * h), d(Vec3::Z * h)).normalize_or_zero();
                img[y * n + x] = (30.0 + 225.0 * nrm.dot(light).max(0.0)) as u8;
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(format!("{path}.grad.pgm"), out).unwrap();
}

/// Many overlapping wide clay strokes at level 0 (seen from far, radius `SCULPT_RS` steps, 120 by default), scrubbed
/// back and forth, some ending on earlier ones, at the app's default strength and hardness `SCULPT_HARD` (0.5): after
/// each, the samples at the surface that stand out from their neighbours. `SCULPT_N` strokes (10); `SCULPT_PICTURE`
/// writes the surface at the end, flat shaded, over `SCULPT_WIN=a,b,half` (offsets along the patch's tangents).
#[test]
#[ignore]
fn far_scrubbed_clay_strokes() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(d);
    // `SCULPT_LEVELS=7`: the app's new ball (a coarser level 0, twice the step).
    let levels = env("SCULPT_LEVELS", 6.0) as usize;
    let voxel = (4.0 * std::f32::consts::PI / (1e6f32 / 2.0)).sqrt() * if levels >= 7 { 2.0 } else { 1.0 };
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, levels);
    let (hard, count, base_r) = (env("SCULPT_HARD", 0.5), env("SCULPT_N", 10.0) as usize, env("SCULPT_RS", 120.0));
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let mut rng = 12345u32;
    let mut rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    for k in 0..count {
        let r = voxel * (base_r + base_r / 6.0 * rand());
        let (a, b) = ((0.2 * rand(), 0.2 * rand()), (0.2 * rand(), 0.2 * rand()));
        let mid = (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
        // Scrubbed: there, back, and partway there again, in one stroke.
        let t = std::time::Instant::now();
        far_clay_stroke(&mut v, patch, &[a, b, a, mid], r, hard);
        let (on, looked) = grain(&v, 0.0, 1.5, 0.3);
        eprintln!("stroke {k} r {:.0} steps ({:.0} ms): {on} of {looked} samples at the surface stand out", r / voxel, t.elapsed().as_secs_f64() * 1e3);
    }
    if let Ok(path) = std::env::var("SCULPT_PICTURE") {
        let win: Vec<f32> = std::env::var("SCULPT_WIN").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        let (a, b, half) = if win.len() == 3 { (win[0], win[1], win[2]) } else { (0.0, 0.0, 0.35) };
        let (t1, t2) = patch.any_orthonormal_pair();
        mesh_picture(&v, 0, (patch + t1 * a + t2 * b).normalize(), half, 900, &path);
        relief_picture(&v, 0, (patch + t1 * a + t2 * b).normalize(), half, 900, &format!("{path}.relief.pgm"));
    }
}

/// The field's surface (levels up to `cap`) seen straight down `-w` over a square `half` across around `w`, shaded
/// steeply by its normal's tilt from the ball's (so only what strokes made shows), as an `n` by `n` PGM at `path`.
fn relief_picture(v: &LevelVolume, cap: usize, w: Vec3, half: f32, n: usize, path: &str) {
    let (u, t) = w.any_orthonormal_pair();
    let light = (u * -0.6 + t * 0.8).normalize();
    let mut img = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let (a, b) = (-half + 2.0 * half * (x as f32 + 0.5) / n as f32, half - 2.0 * half * (y as f32 + 0.5) / n as f32);
            if let Some((q, nrm)) = v.raycast(w * 4.0 + u * a + t * b, -w, cap, false) {
                img[y * n + x] = (128.0 + 400.0 * (nrm - q.normalize()).dot(light)).clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(path, out).unwrap();
}

/// Level 0's samples as the GPU is sent them: each block's 9x9x9, clamped to what is drawn from, at 16 bits.
fn sent0(v: &LevelVolume, bc: glam::IVec3) -> Vec<i16> {
    let (band, reach) = (v.band(0), v.sent_reach(0));
    let mut g = [0.0f32; 729];
    v.gather9_into(0, bc, &mut g);
    g.iter().map(|x| sculpt_core::generator::level_volume::quantize(x.clamp(-reach, reach), band)).collect()
}

/// A wide clay stroke at level 0 (seen from far) changes a slab of samples sixteen steps either side of the surface
/// across its whole disc, but what is drawn reads only those within the plateau, six steps: at the app's default
/// strength over an earlier stroke, the blocks whose samples as the GPU is sent them change are about two thirds of
/// those its dabs change (2617 of 3785 here; at full strength over a fresh ball, about half), and only they are sent
/// and extracted again (all of them were).
#[test]
fn a_wide_far_dab_sends_only_the_blocks_it_changes_near_the_surface() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.01, 2);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    // An earlier stroke, ended: what a later one works over.
    far_clay_stroke(&mut v, patch, &[(-0.3, -0.1), (0.3, 0.1)], 0.5, 0.5);
    let (t1, _) = patch.any_orthonormal_pair();
    v.drain_dirty();
    v.begin_stroke();
    let mut along = Vec3::ZERO;
    let (mut changed, mut sent) = (0usize, 0usize);
    for i in 0..6 {
        let d = (patch + t1 * (-0.2 + 0.08 * i as f32)).normalize();
        let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
        let before: HashMap<glam::IVec3, Vec<i16>> = v.level_keys(0).into_iter().map(|bc| (bc, sent0(&v, bc))).collect();
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(0, hit, rough, 0.5, 0.5 * 0.047, 0.5, 0.22, false) {
            along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
        }
        for (l, bc) in v.drain_dirty() {
            if l == 0 && v.exists(0, bc) {
                changed += 1;
                sent += (before.get(&bc) != Some(&sent0(&v, bc))) as usize;
            }
        }
    }
    v.end_stroke();
    eprintln!("the dabs changed {changed} level 0 blocks, {sent} of them as sent");
    assert!(changed > 3000, "the dabs changed too little to tell: {changed} blocks");
    assert!((sent as f32) < 0.8 * changed as f32, "{sent} of the {changed} blocks the dabs changed are sent again");
}

/// Wide far clay strokes crossing on the app's new ball (radius `SCULPT_R`, 0.5; hardness `SCULPT_HARD`, 0.5; the first
/// `SCULPT_UPTO` of three), each one's time and its end's, with `SCULPT_WIN=a,b,half` the surface's bending over that
/// window, and a picture shaded by the surface's tilt from the ball's at `SCULPT_PICTURE`.relief.pgm.
#[test]
#[ignore]
fn far_crossing_strokes_on_the_new_ball() {
    let path = std::env::var("SCULPT_PICTURE").ok();
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(d);
    let voxel = 2.0 * (4.0 * std::f32::consts::PI / (1e6f32 / 2.0)).sqrt();
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, 7);
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (r, hard) = (env("SCULPT_R", 0.5), env("SCULPT_HARD", 0.5));
    let strokes: [&[(f32, f32)]; 3] = [&[(-0.5, -0.2), (0.4, -0.15)], &[(-0.3, 0.4), (0.05, -0.15)], &[(0.4, 0.35), (-0.1, 0.1)]];
    let upto = env("SCULPT_UPTO", 3.0) as usize;
    for (k, s) in strokes.iter().enumerate().take(upto) {
        let t = std::time::Instant::now();
        let end = far_clay_stroke(&mut v, patch, s, r, hard);
        eprintln!("stroke {k}: {:.0} ms, its end {end:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    let win: Vec<f32> = std::env::var("SCULPT_WIN").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    if win.len() == 3 {
        let (t1, t2) = patch.any_orthonormal_pair();
        let (ca, cb, h) = (win[0], win[1], win[2]);
        let g = 0.5 * voxel;
        let m = (2.0 * h / g) as usize;
        let hts: Vec<f32> = (0..m * m).map(|k| v.raycast(patch * 4.0 + t1 * (ca - h + g * (k % m) as f32) + t2 * (cb - h + g * (k / m) as f32), -patch, 0, false).map_or(f32::NAN, |(q, _)| q.dot(patch))).collect();
        let (mut sum, mut cnt, mut big) = (0.0f64, 0usize, 0usize);
        for j in 1..m - 1 {
            for i in 1..m - 1 {
                let c = hts[j * m + i];
                let d = (hts[j * m + i - 1] + hts[j * m + i + 1] - 2.0 * c).abs() + (hts[(j - 1) * m + i] + hts[(j + 1) * m + i] - 2.0 * c).abs();
                if d.is_finite() {
                    sum += d as f64;
                    cnt += 1;
                    big += (d > 0.1 * voxel) as usize;
                }
            }
        }
        eprintln!("WINDOW mean |second difference| {:.4} steps, {big} over a tenth of a step, over {cnt} points", sum / cnt as f64 / voxel as f64);
    }
    if let Some(path) = path {
        relief_picture(&v, 0, patch, env("SCULPT_HALF", 0.9), 900, &format!("{path}.relief.pgm"));
    }
}

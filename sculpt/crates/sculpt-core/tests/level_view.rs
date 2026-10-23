//! Choosing the level region by region: far work falls back to coarser levels, near work shows in full, the surface of
//! what is shown closes, and rebuilding only around the blocks that changed or were shown or hidden matches a full
//! rebuild.

use glam::Vec3;
use sculpt_core::generator::level_view::LevelView;
use sculpt_core::generator::level_volume::{BlockSurface, HashMap, Key, LevelVolume};
use sculpt_core::generator::levels::edge_counts;

fn stroke(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool) {
    v.begin_stroke();
    let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
    for k in 0..=steps {
        let dir = a.lerp(b, k as f32 / steps as f32).normalize();
        let Some((hit, n)) = v.raycast(dir * 3.0, -dir, l, true) else { continue };
        v.plane_dab(l, hit, n, r, r * if trim { 0.075 } else { 0.095 }, 0.5, 0.4, trim);
    }
    v.end_stroke();
}

/// A ball with work at every level, on its front (+z) and its back (-z).
fn worked() -> LevelVolume {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    for side in [1.0f32, -1.0] {
        stroke(&mut v, 0, Vec3::new(-0.5, 0.2, side), Vec3::new(0.5, 0.3, side), 0.3, false);
        for i in 0..3 {
            let y = -0.2 + 0.15 * i as f32;
            stroke(&mut v, 1, Vec3::new(-0.4, y, side), Vec3::new(0.3, y + 0.05, side), 0.06, false);
        }
        stroke(&mut v, 2, Vec3::new(0.1, 0.0, side), Vec3::new(0.25, 0.05, side), 0.03, true);
        for i in 0..6 {
            let p = Vec3::new(0.05 * i as f32 - 0.1, 0.12, side);
            stroke(&mut v, 3, p, p + Vec3::new(0.01, 0.0, 0.0), 0.012, false);
        }
    }
    v
}

fn fresh_view(v: &LevelVolume) -> LevelView {
    let mut view = LevelView::default();
    view.note(v, &v.keys());
    view.refresh(v, usize::MAX);
    view
}

fn per_level(v: &LevelVolume, view: &LevelView) -> Vec<(usize, usize)> {
    (0..v.levels()).map(|l| {
        let all: Vec<Key> = v.keys().into_iter().filter(|k| k.0 as usize == l).collect();
        let hidden = all.iter().filter(|k| view.hides(**k)).count();
        (all.len() - hidden, hidden)
    }).collect()
}

/// Open edges, non-manifold edges, dropped quads, faces of what the view shows.
fn closed(v: &LevelVolume, view: &LevelView) -> (usize, usize, usize, usize) {
    let (mesh, stats) = v.surface_in(&view.show(v.levels() - 1));
    let (open, nm) = edge_counts(&mesh.faces);
    (open, nm, stats.dropped, mesh.faces.len())
}

const FOCAL: f32 = 1000.0;

#[test]
fn far_work_falls_back_to_coarser_levels_and_near_work_shows() {
    let v = worked();
    let mut view = fresh_view(&v);
    let (_, _, _, full) = closed(&v, &LevelView::default());
    // Near the front: the front's work shows, the back's falls back.
    let near = Vec3::new(0.0, 0.0, 1.25);
    view.choose(&v, near, FOCAL);
    let hidden = |view: &LevelView| v.keys().into_iter().filter(|k| view.hides(*k)).count();
    let near_hidden = hidden(&view);
    let (open, nm, dropped, faces) = closed(&v, &view);
    eprintln!("near: per level (shown, hidden) {:?}; {faces} faces of {full}; open {open}, non-manifold {nm}, dropped {dropped}", per_level(&v, &view));
    assert_eq!((open, dropped), (0, 0));
    // The front's finest work all shows.
    let front3 = v.keys().into_iter().filter(|k| k.0 == 3 && v.voxel(3) * (k.1.z as f32 * 8.0) + v.origin().z > 0.9).collect::<Vec<_>>();
    assert!(!front3.is_empty());
    assert!(front3.iter().all(|k| !view.hides(*k)), "near work was hidden");
    // Further off, more falls back.
    view.choose(&v, Vec3::new(0.0, 0.0, 6.0), FOCAL);
    let mid_hidden = hidden(&view);
    eprintln!("mid: per level (shown, hidden) {:?}", per_level(&v, &view));
    assert!(mid_hidden > near_hidden + 20, "moving away hid little: {near_hidden} -> {mid_hidden}");
    // From afar: only the coarse levels.
    view.choose(&v, Vec3::new(0.0, 0.0, 40.0), FOCAL);
    let (open, nm, dropped, faces_far) = closed(&v, &view);
    let lv = per_level(&v, &view);
    eprintln!("far: per level (shown, hidden) {lv:?}; {faces_far} faces; open {open}, non-manifold {nm}, dropped {dropped}");
    assert_eq!((open, dropped), (0, 0));
    assert_eq!(lv[3].0, 0, "level 3 shows from afar");
    assert_eq!(faces_far, v.surface(0).0.faces.len(), "from afar, the view is level 0's");
    // Back near: the same as the first near choice.
    view.choose(&v, near, FOCAL);
    let again = closed(&v, &view);
    assert_eq!((again.0, again.2), (0, 0));
}

/// Quads by where their corners are, as a sorted list, and how many corners point at no vertex.
fn fingerprint(s: &HashMap<Key, BlockSurface>) -> (usize, Vec<[i64; 3]>) {
    let mut missing = 0;
    let mut out: Vec<[i64; 3]> = Vec::new();
    for q in s.values().flat_map(|b| b.quads.iter()) {
        let mut c = Vec3::ZERO;
        for (k, cell, p) in q {
            match s.get(k).and_then(|b| Some(b.verts[b.vertex(*cell, *p)? as usize])) {
                Some(v) => c += v,
                None => missing += 1,
            }
        }
        out.push([(c.x * 1e5) as i64, (c.y * 1e5) as i64, (c.z * 1e5) as i64]);
    }
    out.sort_unstable();
    (missing, out)
}

#[test]
fn rebuilding_around_what_was_shown_or_hidden_matches_a_full_rebuild() {
    let mut v = worked();
    let mut view = fresh_view(&v);
    let cap = v.levels() - 1;
    let mut kept: HashMap<Key, BlockSurface> = v.extract_in(&v.keys(), &view.show(cap)).0;
    v.drain_dirty();
    let eyes = [Vec3::new(0.0, 0.0, 1.3), Vec3::new(0.0, 0.0, 6.0), Vec3::new(2.0, 0.5, -1.5), Vec3::new(0.0, 0.0, 30.0), Vec3::new(0.2, 0.1, 1.15)];
    for (i, eye) in eyes.iter().enumerate() {
        // A stroke now and then, as sculpting goes on between camera moves.
        if i % 2 == 1 {
            let a = Vec3::new(-0.2 + 0.1 * i as f32, 0.0, 1.0);
            stroke(&mut v, 1 + i % 3, a, a + Vec3::new(0.15, 0.05, 0.0), [0.06, 0.03, 0.012][i % 3], false);
        }
        let mut dirty = v.drain_dirty();
        view.note(&v, &dirty);
        view.refresh(&v, usize::MAX);
        dirty.extend(view.choose(&v, *eye, FOCAL));
        let show = view.show(cap);
        let work = v.extract_set_in(&dirty, &show);
        let (fresh, _) = v.extract_in(&work, &show);
        for k in &dirty {
            if v.block(*k).is_none() || view.hides(*k) {
                kept.remove(k);
            }
        }
        kept.extend(fresh);
        let (full, _) = v.extract_in(&v.keys(), &show);
        eprintln!("eye {eye}: {} flips/changes, per level (shown, hidden) {:?}", dirty.len(), per_level(&v, &view));
        assert_eq!(kept.len(), full.len(), "eye {i}: blocks drawn");
        assert_eq!(fingerprint(&kept), fingerprint(&full), "eye {i}");
        let (open, _, dropped, _) = closed(&v, &view);
        assert_eq!((open, dropped), (0, 0), "eye {i}");
    }
}

#[test]
fn a_block_being_sculpted_shows_until_its_detail_is_measured() {
    let mut v = worked();
    let mut view = fresh_view(&v);
    let far = Vec3::new(0.0, 0.0, 40.0);
    view.choose(&v, far, FOCAL);
    let p = Vec3::new(0.3, -0.3, 1.0);
    stroke(&mut v, 3, p, p + Vec3::new(0.02, 0.0, 0.0), 0.012, false);
    let dirty = v.drain_dirty();
    view.note(&v, &dirty);
    view.choose(&v, far, FOCAL);
    let fresh: Vec<Key> = dirty.iter().copied().filter(|k| k.0 == 3 && v.block(*k).is_some()).collect();
    assert!(!fresh.is_empty());
    assert!(fresh.iter().all(|k| !view.hides(*k)), "a block just sculpted was hidden before its detail was measured");
    view.refresh(&v, usize::MAX);
    view.choose(&v, far, FOCAL);
    assert!(fresh.iter().all(|k| view.hides(*k)), "measured, from afar, it should fall back");
    let (open, _, dropped, _) = closed(&v, &view);
    assert_eq!((open, dropped), (0, 0));
}

/// Triangles of what the view shows (a quad two, a triangle one).
fn triangles(v: &LevelVolume, view: &LevelView) -> usize {
    let (_, stats) = v.surface_in(&view.show(v.levels() - 1));
    stats.quads * 2 + stats.triangles
}

#[test]
fn a_triangle_budget_keeps_the_detail_that_shows_most() {
    let v = worked();
    let mut view = fresh_view(&v);
    let near = Vec3::new(0.0, 0.0, 1.25);
    view.choose(&v, near, FOCAL);
    let full = (triangles(&v, &view), view.stats.triangles);
    let base = {
        let mut only0 = fresh_view(&v);
        only0.choose(&v, Vec3::new(0.0, 0.0, 1e6), FOCAL);
        (triangles(&v, &only0), only0.stats.triangles)
    };
    eprintln!("no budget: {} triangles (estimated {}); level 0 alone {} (estimated {})", full.0, full.1, base.0, base.1);
    assert!((full.1 as f32 / full.0 as f32 - 1.0).abs() < 0.25, "estimate off: {full:?}");
    // Room for about a third of what the finer levels add.
    let budget = base.1 + (full.1 - base.1) / 3;
    view.budget = Some(budget);
    view.choose(&v, near, FOCAL);
    let got = (triangles(&v, &view), view.stats.triangles);
    let (open, _, dropped, _) = closed(&v, &view);
    eprintln!("budget {budget}: {} triangles (estimated {}), detail shown down to {:.2} px; per level (shown, hidden) {:?}; open {open}, dropped {dropped}", got.0, got.1, view.stats.tau_used, per_level(&v, &view));
    assert_eq!((open, dropped), (0, 0));
    assert!(got.1 as f32 <= budget as f32 * 1.1, "over budget");
    assert!(got.1 > base.1, "the budget left out every finer block");
    assert!(view.stats.tau_used > view.tau_px, "the budget did not raise the pixels");
    // What it kept is what is nearest: the front's finest work, right under the eye.
    let under: Vec<Key> = v.keys().into_iter().filter(|k| k.0 == 3 && v.voxel(3) * (k.1.z as f32 * 8.0) + v.origin().z > 0.9).collect();
    let kept = under.iter().filter(|k| !view.hides(**k)).count();
    let all3 = v.keys().into_iter().filter(|k| k.0 == 3 && !view.hides(*k)).count();
    eprintln!("front level 3 blocks kept {kept} of {}; level 3 shown in all {all3}", under.len());
    assert!(kept * 2 >= under.len(), "the nearest fine work went");
    // No budget again: back, all but blocks whose detail comes to under a pixel (hidden now, they need a whole one to
    // show again).
    view.budget = None;
    view.choose(&v, near, FOCAL);
    assert!(view.stats.triangles as f32 > full.1 as f32 * 0.95, "{} of {}", view.stats.triangles, full.1);
}

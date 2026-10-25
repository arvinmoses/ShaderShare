//! A detail layer over a form: its bumps ride a big stroke on the form whole, in place, without grain; the surface
//! stays closed; undo brings back both; the mask hides the layer where painted.

use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::detail_layer::Layered;
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::generator::levels::edge_counts;

fn check(v: &LevelVolume, cap: usize) -> (usize, usize, usize) {
    let (mesh, stats) = v.surface(cap);
    let (open, nm) = edge_counts(&mesh.faces);
    (open, nm, stats.dropped)
}

/// The layer with what it shows, as one.
struct Lv {
    l: Layered,
    s: LevelVolume,
}

impl Lv {
    fn new(mut s: LevelVolume, from: usize) -> Lv {
        let l = Layered::new(&mut s, from);
        Lv { l, s }
    }
    fn form(&self) -> &LevelVolume {
        self.l.form()
    }
    fn shown(&self) -> &LevelVolume {
        &self.s
    }
    fn blocks(&self) -> usize {
        self.l.blocks()
    }
    fn begin_stroke(&mut self) {
        self.l.begin_stroke();
    }
    #[allow(clippy::too_many_arguments)]
    fn plane_dab(&mut self, l: usize, at: Vec3, along: Vec3, r: f32, h: f32, hardness: f32, alpha: f32, trim: bool) -> Option<(Vec3, sculpt_core::generator::level_volume::LevelDab)> {
        self.l.plane_dab(&mut self.s, l, at, along, r, h, hardness, alpha, trim)
    }
    fn end_stroke(&mut self) {
        self.l.end_stroke(&mut self.s);
    }
    fn undo(&mut self) {
        self.l.undo(&mut self.s);
    }
    fn redo(&mut self) {
        self.l.redo(&mut self.s);
    }
    fn paint_mask(&mut self, c: Vec3, r: f32, to: f32) {
        self.l.paint_mask(&mut self.s, c, r, to);
    }
}

const ROWS: [f32; 2] = [0.0, 0.29];

fn xs() -> Vec<f32> {
    (0..5).map(|i| -0.024 + 0.012 * i as f32).collect()
}

/// The surface's height seen along -z over (x, y).
fn h(v: &LevelVolume, x: f32, y: f32) -> f32 {
    let p = Vec3::new(x, y, 1.0).normalize();
    v.raycast(Vec3::new(p.x, p.y, 3.0), -Vec3::Z, v.levels() - 1, false).map_or(f32::NAN, |(q, _)| q.z)
}

/// How far the shown surface stands out of the form at the form's point over (x, y), along the form's normal.
fn out_of(lv: &Lv, x: f32, y: f32) -> f32 {
    let p = Vec3::new(x, y, 1.0).normalize();
    let f = lv.form();
    let Some((q, n)) = f.raycast(Vec3::new(p.x, p.y, 3.0), -Vec3::Z, f.levels() - 1, false) else { return f32::NAN };
    let s = lv.shown();
    s.raycast(q + n * 0.02, -n, s.levels() - 1, false).map_or(f32::NAN, |(r, _)| 0.02 - (q + n * 0.02 - r).length())
}

/// Per row: the bumps' height out of the form (best within a little of each bump), averaged; and where each bump's
/// top is.
fn relief(lv: &Lv, y0: f32) -> (f32, Vec<(f32, f32)>) {
    let mut sum = 0.0;
    let mut at = Vec::new();
    for &x0 in &xs() {
        let (mut best, mut where_) = (0.0f32, (x0, y0));
        for i in 0..21 {
            for j in 0..9 {
                let (x, y) = (x0 - 0.004 + 0.001 * j as f32, y0 - 0.01 + 0.001 * i as f32);
                let r = out_of(lv, x, y);
                if r > best {
                    best = r;
                    where_ = (x, y);
                }
            }
        }
        sum += best;
        at.push(where_);
    }
    (sum / xs().len() as f32, at)
}

/// Each bump of a row: its centre (heights out of the form weighting the points around it), and its shape there: the
/// heights over a little grid around the centre.
fn shapes(lv: &Lv, y0: f32) -> Vec<(Vec3, Vec<f32>)> {
    xs().iter()
        .map(|&x0| {
            let (mut c, mut w) = (Vec3::ZERO, 0.0);
            for i in 0..17 {
                for j in 0..17 {
                    let (x, y) = (x0 - 0.0048 + 0.0006 * i as f32, y0 - 0.0048 + 0.0006 * j as f32);
                    let d = out_of(lv, x, y).max(0.0);
                    c += Vec3::new(x, y, 0.0) * d;
                    w += d;
                }
            }
            let c = c / w;
            let shape = (0..121).map(|k| out_of(lv, c.x - 0.005 + 0.001 * (k % 11) as f32, c.y - 0.005 + 0.001 * (k / 11) as f32)).collect();
            (c, shape)
        })
        .collect()
}

/// The most two rows' bumps differ in shape, each about its own centre, and the most their centres moved.
fn compare(a: &[(Vec3, Vec<f32>)], b: &[(Vec3, Vec<f32>)]) -> (f32, f32) {
    let mut shape = 0.0f32;
    let mut moved = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        moved = moved.max((x.0 - y.0).length());
        for (p, q) in x.1.iter().zip(&y.1) {
            shape = shape.max((p - q).abs());
        }
    }
    (shape, moved)
}

fn big_strokes(lv: &mut Lv) -> (f64, usize) {
    let (mut ms, mut dabs) = (0.0, 0);
    for _ in 0..2 {
        lv.begin_stroke();
        let mut along = Vec3::ZERO;
        for k in 0..=8 {
            let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, 0.0, 1.0).normalize();
            let Some((hit, n)) = lv.form().raycast(d * 3.0, -d, 0, true) else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            let t = Instant::now();
            if let Some((n, _)) = lv.plane_dab(0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            ms += t.elapsed().as_secs_f64() * 1e3;
            dabs += 1;
        }
        lv.end_stroke();
    }
    (ms, dabs)
}

#[test]
fn bumps_on_a_detail_layer_ride_a_big_stroke_whole() {
    rides(5, 4);
}

/// As the app's volume has it: a sixth level, finer than the work, and the layer taking a level with none of it.
#[test]
fn bumps_ride_a_layer_taken_from_a_level_coarser_than_the_work() {
    rides(6, 3);
}

fn rides(levels: usize, from: usize) {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, levels);
    for &y in &ROWS {
        for &x in &xs() {
            v.begin_stroke();
            v.ball_dab(4, Vec3::new(x, y, 1.0).normalize(), 0.004, 0.001, false);
            v.end_stroke();
        }
    }
    let ground0: Vec<f32> = ROWS.iter().map(|&y| h(&v, 0.06, y)).collect();
    let t = Instant::now();
    let mut lv = Lv::new(v, from);
    eprintln!("into a layer: {} blocks in {:.0} ms", lv.blocks(), t.elapsed().as_secs_f64() * 1e3);
    let before: Vec<(f32, Vec<(f32, f32)>)> = ROWS.iter().map(|&y| relief(&lv, y)).collect();
    let shapes0: Vec<_> = ROWS.iter().map(|&y| shapes(&lv, y)).collect();
    eprintln!("bumps out of the form: {:.5} {:.5}", before[0].0, before[1].0);
    assert!(before.iter().all(|r| r.0 > 0.003), "the layer does not hold the bumps");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    let (ms, dabs) = big_strokes(&mut lv);
    let after: Vec<(f32, Vec<(f32, f32)>)> = ROWS.iter().map(|&y| relief(&lv, y)).collect();
    let shapes1: Vec<_> = ROWS.iter().map(|&y| shapes(&lv, y)).collect();
    let rise: Vec<f32> = ROWS.iter().zip(&ground0).map(|(&y, g)| h(lv.shown(), 0.06, y) - g).collect();
    for r in 0..2 {
        eprintln!(
            "row {r}: bumps {:.5} -> {:.5} (the form rose {:.4}); each bump's shape about its centre changed at most {:.6}, its centre moved {:.5}",
            before[r].0, after[r].0, rise[r], compare(&shapes0[r], &shapes1[r]).0, compare(&shapes0[r], &shapes1[r]).1
        );
    }
    eprintln!("{:.1} ms a dab with the layer following; last follow {:?}", ms / dabs as f64, lv.l.last);
    assert!(rise[0] > 0.03, "the strokes did not build");
    for r in 0..2 {
        assert!(after[r].0 > 0.9 * before[r].0 && after[r].0 < 1.1 * before[r].0, "row {r}: the bumps did not ride the strokes whole");
        let (shape, moved) = compare(&shapes0[r], &shapes1[r]);
        assert!(moved < 0.0015, "row {r}: the bumps slid");
        assert!(shape < 0.12 * before[r].0, "row {r}: the bumps changed shape (softened or grainy)");
    }
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    // Undo brings back the form and the layer as they were; redo the strokes again.
    lv.undo();
    lv.undo();
    assert!((h(lv.shown(), 0.06, 0.0) - ground0[0]).abs() < 1e-4, "undo left the form raised");
    let undone = relief(&lv, 0.0).0;
    assert!((undone - before[0].0).abs() < 1e-5, "undo did not bring the layer back");
    lv.redo();
    lv.redo();
    let redone = relief(&lv, 0.0).0;
    assert!((redone - after[0].0).abs() < 1e-5, "redo is not the same strokes");

    // The mask hides the layer where painted.
    let p = Vec3::new(0.0, 0.29, 1.0).normalize();
    let at = lv.form().raycast(Vec3::new(p.x, p.y, 3.0), -Vec3::Z, 4, false).unwrap().0;
    lv.paint_mask(at, 0.1, 0.0);
    let masked: Vec<f32> = ROWS.iter().map(|&y| relief(&lv, y).0).collect();
    eprintln!("masked on the falloff row: {:.5} {:.5}", masked[0], masked[1]);
    assert!(masked[1] < 0.1 * after[1].0, "the mask did not hide the layer");
    assert!((masked[0] - after[0].0).abs() < 1e-5, "the mask reached the other row");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    // Hide all takes the layer away on both rows; Show all brings both back whole, past the painted mask.
    lv.l.fill_mask(&mut lv.s, 0.0);
    let hidden: Vec<f32> = ROWS.iter().map(|&y| relief(&lv, y).0).collect();
    eprintln!("hidden everywhere: {:.5} {:.5}", hidden[0], hidden[1]);
    assert!(hidden.iter().zip(&after).all(|(h, a)| *h < 0.05 * a.0), "hide all left the layer showing");
    lv.l.fill_mask(&mut lv.s, 1.0);
    let back: Vec<f32> = ROWS.iter().map(|&y| relief(&lv, y).0).collect();
    assert!(back.iter().zip(&after).all(|(b, a)| (b - a.0).abs() < 1e-5), "show all is not the layer unmasked");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));
}

#[test]
fn detail_painted_on_the_layer_goes_into_it_and_rides_the_form() {
    let v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
    let mut lv = Lv::new(v, 3);
    assert_eq!(lv.blocks(), 0);
    // A row of bumps painted on the layer: dabs on what is shown, then taken into the layer.
    for &x in &xs() {
        lv.s.begin_stroke();
        lv.s.ball_dab(4, Vec3::new(x, 0.0, 1.0).normalize(), 0.004, 0.001, false);
        lv.s.end_stroke();
        lv.l.take_detail(&mut lv.s);
    }
    let painted = relief(&lv, 0.0).0;
    let form_relief = xs().iter().map(|&x| h(lv.form(), x, 0.0) - h(lv.form(), x + 0.006, 0.0)).fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("painted on the layer: bumps {painted:.5} out of the form; the form itself {form_relief:.6}; {} layer blocks", lv.blocks());
    assert!(painted > 0.003, "the bumps are not in the layer");
    assert!(form_relief < 0.0002, "the bumps went into the form");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));
    let shapes0 = shapes(&lv, 0.0);
    big_strokes(&mut lv);
    let after = relief(&lv, 0.0).0;
    let (shape, moved) = compare(&shapes0, &shapes(&lv, 0.0));
    eprintln!("after the big strokes: {after:.5}, shape changed {shape:.6}, centres moved {moved:.5}");
    assert!((after - painted).abs() < 0.1 * painted && shape < 0.12 * painted && moved < 0.0015);
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));
    // Undo the strokes and the last bump.
    lv.undo();
    lv.undo();
    lv.undo();
    let left = shapes(&lv, 0.0);
    let last = left.last().unwrap().1.iter().fold(0.0f32, |a, b| a.max(*b));
    assert!(last < 0.0002, "undo did not take the last bump back out of the layer");
    assert!(left[0].1.iter().fold(0.0f32, |a, b| a.max(*b)) > 0.003, "undo took too much");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));
}

/// At the demo's scale (a ball of 1M triangles at level 0, six levels), how long following takes, part by part.
#[test]
#[ignore]
fn demo_scale_timing() {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6);
    for j in 0..12 {
        for i in 0..60 {
            let p = Vec3::new(-0.06 + 0.002 * i as f32, -0.06 + 0.01 * j as f32, 1.0).normalize();
            v.begin_stroke();
            v.ball_dab(4, p, 0.0015, 0.0004, true);
            v.end_stroke();
        }
    }
    let mut lv = Lv::new(v, 4);
    eprintln!("layer: {} blocks", lv.blocks());
    lv.begin_stroke();
    let mut along = Vec3::ZERO;
    for k in 0..=8 {
        let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, 0.0, 1.0).normalize();
        let Some((hit, n)) = lv.form().raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        let t = Instant::now();
        if let Some((n, _)) = lv.plane_dab(0, hit, rough, 0.5, 0.5 * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        eprintln!("dab {k}: {:.1} ms, layer {} blocks; follow {:?}", t.elapsed().as_secs_f64() * 1e3, lv.blocks(), lv.l.last);
    }
    let t = Instant::now();
    lv.end_stroke();
    eprintln!("end: {:.1} ms; follow {:?}", t.elapsed().as_secs_f64() * 1e3, lv.l.last);
}

/// Following in jobs that land late (here after every third dab) ends the same as following at once: the stroke's
/// last follow redoes all it touched from the layer as the stroke began.
#[test]
fn following_in_jobs_ends_as_following_at_once() {
    let make = || {
        let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
        for &y in &ROWS {
            for &x in &xs() {
                v.begin_stroke();
                v.ball_dab(4, Vec3::new(x, y, 1.0).normalize(), 0.004, 0.001, false);
                v.end_stroke();
            }
        }
        Lv::new(v, 4)
    };
    let mut at_once = make();
    big_strokes(&mut at_once);
    let mut jobs = make();
    jobs.l.deferred = true;
    for _ in 0..2 {
        jobs.l.begin_stroke();
        let mut along = Vec3::ZERO;
        for k in 0..=8 {
            let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, 0.0, 1.0).normalize();
            let Some((hit, n)) = jobs.l.form().raycast(d * 3.0, -d, 0, true) else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = jobs.l.plane_dab(&mut jobs.s, 0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            assert!(jobs.l.busy());
            if k % 3 == 2
                && let Some(job) = jobs.l.follow_take()
            {
                let done = job.run();
                jobs.l.follow_apply(&mut jobs.s, done);
            }
        }
        jobs.l.end_stroke(&mut jobs.s);
        assert!(jobs.l.busy(), "the stroke's last follow is left to do");
        jobs.l.settle(&mut jobs.s);
        assert!(!jobs.l.busy());
    }
    for &y in &ROWS {
        let (a, b) = (relief(&at_once, y).0, relief(&jobs, y).0);
        let (shape, moved) = compare(&shapes(&at_once, y), &shapes(&jobs, y));
        eprintln!("row {y}: at once {a:.5}, in jobs {b:.5}; shapes differ {shape:.7}, centres {moved:.6}");
        assert!((a - b).abs() < 2e-5 && shape < 2e-5, "following in jobs ended differently");
    }
    assert_eq!(check(jobs.shown(), 4), (0, 0, 0));
    jobs.undo();
    at_once.undo();
    assert!((relief(&at_once, 0.0).0 - relief(&jobs, 0.0).0).abs() < 2e-5);
}

/// Two layers of different resolutions: a broad bump on a coarse layer and a row of small ones on a fine layer over
/// it. Each rides a big stroke on the form whole; each can be hidden, its resolution changed, removed and brought
/// back; and the whole stack saves and opens again as it was.
#[test]
fn layers_of_two_resolutions_ride_together_and_save() {
    let v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
    let mut lv = Lv::new(v, 3);
    assert_eq!(lv.l.layers().len(), 1);
    let fine = 0;
    let broad = lv.l.add_layer(&mut lv.s, "Broad", 3);
    assert_eq!((lv.l.layers()[fine].level, lv.l.layers()[broad].level), (4, 3));
    // A broad bump on the coarse layer, then small bumps on the fine one, over and beside it.
    lv.l.set_active(broad);
    lv.s.begin_stroke();
    lv.s.ball_dab(3, Vec3::new(0.0, 0.0, 1.0), 0.012, 0.003, false);
    lv.s.end_stroke();
    lv.l.take_detail(&mut lv.s);
    lv.l.set_active(fine);
    for &x in &xs() {
        // On the surface shown (over the broad bump, where it is).
        let d = Vec3::new(x, 0.0, 1.0).normalize();
        let (at, _) = lv.s.raycast(d * 3.0, -d, 4, false).unwrap();
        lv.s.begin_stroke();
        lv.s.ball_dab(4, at, 0.004, 0.001, false);
        lv.s.end_stroke();
        lv.l.take_detail(&mut lv.s);
    }
    let infos = lv.l.layers();
    eprintln!("blocks: fine {} at level 4, broad {} at level 3", infos[fine].blocks, infos[broad].blocks);
    assert!(infos[broad].blocks > 0 && infos[fine].blocks > 0);
    // Each layer alone: the broad bump's top, and the small bumps' shapes.
    let alone = |lv: &mut Lv, k: usize| {
        let other = 1 - k;
        lv.l.set_visible(&mut lv.s, other, false);
        let out = (out_of(lv, 0.0, 0.0), shapes(lv, 0.0));
        lv.l.set_visible(&mut lv.s, other, true);
        out
    };
    let (top0, _) = alone(&mut lv, broad);
    let (_, small0) = alone(&mut lv, fine);
    let both0 = out_of(&lv, 0.0, 0.0);
    eprintln!("broad bump {top0:.5}; both layers there {both0:.5}");
    assert!(top0 > 0.004, "the broad bump is not in its layer");
    assert!(both0 > top0 + 0.002, "the small bump over the broad one does not add to it");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    big_strokes(&mut lv);
    let (top1, _) = alone(&mut lv, broad);
    let (_, small1) = alone(&mut lv, fine);
    let (shape, moved) = compare(&small0, &small1);
    eprintln!("after the big strokes: broad bump {top1:.5}; small bumps' shapes changed {shape:.6}, moved {moved:.5}");
    assert!((top1 - top0).abs() < 0.1 * top0, "the broad bump did not ride the strokes");
    assert!(shape < 0.0006 && moved < 0.0015, "the small bumps did not ride the strokes");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    // Saved and opened again: the same surface, the same layers.
    let mut file = Vec::new();
    lv.l.write_to(&mut file).unwrap();
    let (s2, l2) = Layered::read_from(&mut &file[..], 5).unwrap();
    let l2 = l2.expect("the layers were not saved");
    assert_eq!(l2.layers().len(), 2);
    let mut worst = 0.0f32;
    for j in 0..21 {
        for i in 0..21 {
            let (x, y) = (-0.05 + 0.005 * i as f32, -0.05 + 0.005 * j as f32);
            worst = worst.max((h(&s2, x, y) - h(lv.shown(), x, y)).abs());
        }
    }
    eprintln!("opened again: off by at most {worst:.7}");
    assert!(worst < 2e-5, "the opened stack does not show as it did");

    // The broad layer held finer, then coarser again: about the same bump.
    lv.l.set_level(&mut lv.s, broad, 4);
    let (top2, _) = alone(&mut lv, broad);
    lv.l.set_level(&mut lv.s, broad, 3);
    let (top3, _) = alone(&mut lv, broad);
    eprintln!("broad bump at level 4 {top2:.5}, back at level 3 {top3:.5}");
    assert!((top2 - top1).abs() < 0.05 * top1 && (top3 - top1).abs() < 0.1 * top1);
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));

    // Removed, and back with undo.
    lv.l.remove_layer(&mut lv.s, broad);
    let gone = out_of(&lv, 0.0, 0.0);
    lv.undo();
    let back = out_of(&lv, 0.0, 0.0);
    eprintln!("both layers at the broad bump {:.5}; with it removed {gone:.5}; back {back:.5}", out_of(&lv, 0.0, 0.0));
    assert!(gone < back - 0.004, "removing the layer did not take its bump away");
    assert_eq!(check(lv.shown(), 4), (0, 0, 0));
}

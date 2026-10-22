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
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
    for &y in &ROWS {
        for &x in &xs() {
            v.begin_stroke();
            v.ball_dab(4, Vec3::new(x, y, 1.0).normalize(), 0.004, 0.001, false);
            v.end_stroke();
        }
    }
    let ground0: Vec<f32> = ROWS.iter().map(|&y| h(&v, 0.06, y)).collect();
    let t = Instant::now();
    let mut lv = Lv::new(v, 4);
    eprintln!("into a layer: {} blocks in {:.0} ms", lv.blocks(), t.elapsed().as_secs_f64() * 1e3);
    let before: Vec<(f32, Vec<(f32, f32)>)> = ROWS.iter().map(|&y| relief(&lv, y)).collect();
    let shapes0: Vec<_> = ROWS.iter().map(|&y| shapes(&lv, y)).collect();
    eprintln!("bumps out of the form: {:.5} {:.5}", before[0].0, before[1].0);
    for r in 0..2 {
        assert!(before[r].0 > 0.003, "the layer does not hold the bumps");
    }
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

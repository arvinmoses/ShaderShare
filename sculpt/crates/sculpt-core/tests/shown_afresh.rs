//! What is shown live against the same layers composed afresh (as opening the file does): every operation on the
//! stack leaves the shown volume with exactly the blocks composing it afresh gives, none left over.

use glam::Vec3;
use sculpt_core::generator::detail_layer::Layered;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::generator::{BoolOp, Primitive};

const LEVELS: usize = 6;

/// Per level: blocks only live, only afresh, and the most a shared block's values differ.
fn against_afresh(s: &LevelVolume, l: &Layered) -> Vec<(usize, usize, f32)> {
    let mut buf = Vec::new();
    l.write_to(&mut buf).unwrap();
    let (again, _) = Layered::read_from(&mut &buf[..], LEVELS).unwrap();
    (0..s.levels())
        .map(|lv| {
            let (a, b): (std::collections::HashSet<_>, std::collections::HashSet<_>) = (s.level_keys(lv).into_iter().map(|v| v.to_array()).collect(), again.level_keys(lv).into_iter().map(|v| v.to_array()).collect());
            let most = a
                .intersection(&b)
                .map(|k| {
                    let bc = glam::IVec3::from_array(*k);
                    s.block((lv as u8, bc)).unwrap().iter().zip(again.block((lv as u8, bc)).unwrap().iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            (a.difference(&b).count(), b.difference(&a).count(), most)
        })
        .collect()
}

/// The blocks that differ after `step`, reported.
fn differ(step: &str, s: &LevelVolume, l: &Layered) -> usize {
    let d = against_afresh(s, l);
    eprintln!("{step}: per level (only live, only afresh, values apart) {d:?}");
    d.iter().map(|x| x.0 + x.1).sum()
}

/// A level `level` clay (or trim) stroke across the front at `y`, on a volume of its own.
fn stroke(s: &mut LevelVolume, level: usize, y: f32, r: f32, trim: bool) {
    s.begin_stroke();
    let mut along = Vec3::ZERO;
    for k in 0..=8 {
        let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, y, 1.0).normalize();
        let Some((hit, n)) = s.raycast(d * 3.0, -d, level, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = s.plane_dab(level, hit, rough, r, r * 0.095, 0.5, 0.4, trim) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
    }
    s.end_stroke();
}

/// A big level 0 clay stroke across the front at `y` on the form under the layers, following in jobs that land every
/// third dab, and the stroke's last follow after the pen is up (as the app runs it).
fn form_stroke(l: &mut Layered, s: &mut LevelVolume, y: f32) {
    l.begin_stroke();
    let mut along = Vec3::ZERO;
    for k in 0..=8 {
        let d = Vec3::new(-0.1 + 0.2 * k as f32 / 8.0, y, 1.0).normalize();
        let Some((hit, n)) = l.form().raycast(d * 3.0, -d, 0, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = l.plane_dab(s, 0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        if k % 3 == 2
            && let Some(job) = l.follow_take()
        {
            let done = job.run();
            l.follow_apply(s, done);
        }
    }
    l.end_stroke(s);
    l.settle(s);
}

/// The review session's operations in order: a layer made of fine work, big form strokes under it (ending in jobs),
/// undo and redo, its mask painted, a second finer layer painted on, a form stroke under both, a layer hidden and
/// shown, a generator added and edited. After each, the shown volume holds exactly the blocks composing the same
/// layers afresh gives. (A form stroke's end used to leave blocks at the levels between: the form dropped the parents
/// it had made for carried detail, and the stroke's last follow, mirroring them coarsest first, kept each as a parent
/// of the finer one it removed next. They then read the form as it was, whatever changed level 0 under them.)
#[test]
fn every_operation_leaves_shown_as_composed_afresh() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, LEVELS);
    // Coarse form, then fine work: grooves at level 4, dots at level 5 (as the review's patch).
    stroke(&mut s, 0, 0.0, 0.3, false);
    stroke(&mut s, 1, 0.05, 0.08, false);
    for k in 0..3 {
        stroke(&mut s, 4, -0.02 + 0.015 * k as f32, 0.006, true);
    }
    for k in 0..6 {
        s.begin_stroke();
        s.ball_dab(5, Vec3::new(-0.03 + 0.012 * k as f32, 0.03, 1.0).normalize(), 0.003, 0.001, false);
        s.end_stroke();
    }
    // A big stroke burying it, undone.
    stroke(&mut s, 0, 0.0, 0.3, false);
    s.undo();
    let mut bad = Vec::new();
    let mut check = |step: &str, s: &LevelVolume, l: &Layered| {
        if differ(step, s, l) > 0 {
            bad.push(step.to_string());
        }
    };
    let mut l = Layered::new(&mut s, 4);
    l.deferred = true;
    check("made a layer", &s, &l);
    form_stroke(&mut l, &mut s, 0.0);
    check("a form stroke under it", &s, &l);
    // A generator at level 0 over where that stroke was: its pits cut level 0 under the leftovers, which would show
    // the old surface across them at any cap above level 0.
    let pits = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 80, seed: 2, size: 0.1, lift: 0.1, min_spacing: 0.2, blend: 0.3, ..GeneratorSpec::default() };
    let k = l.add_generator(&mut s, "Pits", 0, &pits);
    check("pits cut at level 0", &s, &l);
    l.remove_layer(&mut s, k);
    check("pits removed", &s, &l);
    l.undo(&mut s);
    l.undo(&mut s);
    l.undo(&mut s);
    check("undo", &s, &l);
    l.redo(&mut s);
    check("redo", &s, &l);
    let p = Vec3::new(0.0, 0.0, 1.0);
    let at = l.form().raycast(p * 3.0, -p, 3, false).unwrap().0;
    l.paint_mask(&mut s, at, 0.05, 0.0);
    check("mask hidden", &s, &l);
    l.paint_mask(&mut s, at, 0.05, 1.0);
    check("mask shown", &s, &l);
    l.add_layer("Layer 2", 5);
    for k in 0..4 {
        let d = Vec3::new(-0.02 + 0.012 * k as f32, -0.03, 1.0).normalize();
        let (hit, _) = s.raycast(d * 3.0, -d, 5, false).unwrap();
        s.begin_stroke();
        s.ball_dab(5, hit, 0.003, 0.001, true);
        s.end_stroke();
        l.take_detail(&mut s);
    }
    check("painted on a second, finer layer", &s, &l);
    form_stroke(&mut l, &mut s, 0.02);
    check("a form stroke under both", &s, &l);
    l.set_visible(&mut s, 1, false);
    check("second layer hidden", &s, &l);
    l.set_visible(&mut s, 1, true);
    check("second layer shown", &s, &l);
    let size = 0.03 * 2.6;
    let spec = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 12, seed: 1, size, lift: 0.15, random_spin: true, min_spacing: 3.0 * size, size_var: 0.4, stretch_var: 0.35, tilt: 25.0, blend: 0.3, rough: 0.15, ..GeneratorSpec::default() };
    let k = l.add_generator(&mut s, "Rocks", 0, &spec);
    check("a generator added", &s, &l);
    l.set_generator(&mut s, k, &GeneratorSpec { count: 60, size: size * 1.5, ..spec.clone() });
    check("a generator edit", &s, &l);
    form_stroke(&mut l, &mut s, -0.02);
    check("a form stroke under the rocks", &s, &l);
    assert!(bad.is_empty(), "shown holds other blocks than composing it afresh gives, after: {bad:?}");
}

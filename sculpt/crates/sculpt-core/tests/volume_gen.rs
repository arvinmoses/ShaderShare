//! Generator layers on a volume: a primitive scattered over the form, joined to it (or cut from it) softly, in the
//! layer stack: the surface stays closed at every level, what it scatters stands where it should from near and far,
//! rides a big stroke on the form, edits as it builds, saves, undoes and fades under its mask.

use glam::{Quat, Vec3};
use sculpt_core::generator::detail_layer::Layered;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::generator::levels::edge_counts;
use sculpt_core::generator::{BoolOp, Instance, Primitive};

struct Lv {
    l: Layered,
    s: LevelVolume,
}

impl Lv {
    fn ball() -> Lv {
        let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
        let l = Layered::new(&mut s, 2);
        Lv { l, s }
    }
    fn add(&mut self, level: usize, spec: &GeneratorSpec) -> usize {
        self.l.add_generator(&mut self.s, "Rocks", level, spec)
    }
}

fn rocks() -> GeneratorSpec {
    GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 6, seed: 3, size: 0.1, lift: 0.2, random_spin: false, min_spacing: 0.45, blend: 0.25, ..GeneratorSpec::default() }
}

fn closed(v: &LevelVolume, cap: usize) -> (usize, usize, usize) {
    let (mesh, stats) = v.surface(cap);
    let (open, nm) = edge_counts(&mesh.faces);
    (open, nm, stats.dropped)
}

/// How far the surface (levels up to `cap`) reaches out from an instance's centre along its up axis.
fn top(v: &LevelVolume, i: &Instance, cap: usize) -> f32 {
    let up = i.rotation * Vec3::Y;
    v.raycast(i.translation + up * 0.5, -up, cap, false).map_or(f32::NAN, |(q, _)| (q - i.translation).dot(up))
}

/// Where the anchor of an instance lifted by `lift` of its size is, and the normal there.
fn anchor(i: &Instance, lift: f32) -> (Vec3, Vec3) {
    let up = i.rotation * Vec3::Y;
    (i.translation - up * (lift * i.scale.y), up)
}

/// How far the surface stands out along `n` at `p` (negative: a pit).
fn out_at(v: &LevelVolume, p: Vec3, n: Vec3) -> f32 {
    v.raycast(p + n * 0.5, -n, v.levels() - 1, false).map_or(f32::NAN, |(q, _)| (q - p).dot(n))
}

fn surface_key(v: &LevelVolume, cap: usize) -> Vec<[i64; 3]> {
    let (mesh, _) = v.surface(cap);
    let mut out: Vec<[i64; 3]> = mesh.positions.iter().map(|p| [(p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64]).collect();
    out.sort_unstable();
    out
}

#[test]
fn scattered_rocks_stand_on_the_form_closed_at_every_level() {
    let mut lv = Lv::ball();
    let k = lv.add(2, &rocks());
    let inst = lv.l.generator_instances(k);
    assert_eq!(inst.len(), 6, "count");
    // Apart by the spacing, each on the form.
    for (a, i) in inst.iter().enumerate() {
        let (p, _) = anchor(i, 0.2);
        let d = lv.l.form().value_upto(p, 1, false);
        assert!(d.abs() < 0.002, "anchor {a} off the form by {d}");
        for j in &inst[a + 1..] {
            assert!(anchor(j, 0.2).0.distance(p) >= 0.45 - 1e-4, "too close");
        }
    }
    for cap in 0..4 {
        let (open, nm, dropped) = closed(&lv.s, cap);
        let tops: Vec<f32> = inst.iter().map(|i| top(&lv.s, i, cap)).collect();
        eprintln!("cap {cap}: open {open}, non-manifold {nm}, dropped {dropped}; tops {tops:.4?} (radius 0.1)");
        assert_eq!((open, dropped), (0, 0), "cap {cap}");
        // From near (level 2 and finer) the rocks are round to within a step; from afar (levels 0 and 1) still there.
        let tol = if cap >= 2 { 0.0075 + 0.002 } else { 0.03 };
        for t in &tops {
            assert!((t - 0.1).abs() < tol, "cap {cap}: a rock's top at {t}");
        }
    }
}

#[test]
fn cutters_dig_pits() {
    let mut lv = Lv::ball();
    let spec = GeneratorSpec { op: BoolOp::Subtract, lift: 0.0, ..rocks() };
    let k = lv.add(2, &spec);
    for i in lv.l.generator_instances(k) {
        let (p, n) = anchor(&i, 0.0);
        let depth = -out_at(&lv.s, p, n);
        assert!((depth - 0.1).abs() < 0.012, "a pit {depth} deep (radius 0.1)");
    }
    let (open, _, dropped) = closed(&lv.s, 3);
    assert_eq!((open, dropped), (0, 0));
}

#[test]
fn rocks_ride_a_big_stroke_on_the_form() {
    let mut lv = Lv::ball();
    let k = lv.add(2, &rocks());
    let before = lv.l.generator_instances(k);
    // The rock nearest the front, and a big clay stroke over it.
    let (ri, rock) = before.iter().enumerate().max_by(|a, b| a.1.translation.z.total_cmp(&b.1.translation.z)).map(|(i, r)| (i, *r)).unwrap();
    let (p0, n0) = anchor(&rock, 0.2);
    let rise_form = |lv: &Lv| out_at(lv.l.form(), p0, n0);
    lv.l.begin_stroke();
    for s in 0..12 {
        let dir = (p0 + Vec3::new(-0.12 + 0.02 * s as f32, 0.0, 0.0)).normalize();
        let Some((hit, n)) = lv.l.form().raycast(dir * 3.0, -dir, 0, true) else { continue };
        let _ = lv.l.plane_dab(&mut lv.s, 0, hit, n, 0.35, 0.035, 0.5, 0.4, false);
    }
    lv.l.end_stroke(&mut lv.s);
    lv.l.settle(&mut lv.s);
    let rise = rise_form(&lv);
    let after = lv.l.generator_instances(k);
    let moved = (after[ri].translation - rock.translation).dot(n0);
    let t = top(&lv.s, &after[ri], 3);
    eprintln!("the form rose {rise:.4} under the rock; the rock moved {moved:.4} along the old normal; its top {t:.4} out of its centre");
    assert!(rise > 0.01, "the stroke did not build");
    assert!((moved - rise).abs() < 0.2 * rise + 0.002, "the rock did not ride the form");
    assert!((t - 0.1).abs() < 0.01, "the rock lost its shape: {t}");
    // Rocks far from the stroke stay exactly.
    for (a, b) in before.iter().zip(&after) {
        if a.translation.distance(rock.translation) > 0.9 {
            assert_eq!(a.translation, b.translation, "a far rock moved");
        }
    }
    let (open, _, dropped) = closed(&lv.s, 3);
    assert_eq!((open, dropped), (0, 0));
    // Undo puts the form and the rocks back.
    lv.l.undo(&mut lv.s);
    assert_eq!(lv.l.generator_instances(k)[ri].translation, rock.translation);
}

#[test]
fn an_edit_shows_what_building_it_afresh_shows_and_undoes() {
    let mut a = Lv::ball();
    let bare = surface_key(&a.s, 3);
    let k = a.add(2, &rocks());
    let bigger = GeneratorSpec { size: 0.13, blend: 0.4, ..rocks() };
    a.l.set_generator(&mut a.s, k, &bigger);
    let mut b = Lv::ball();
    b.add(2, &bigger);
    assert!(surface_key(&a.s, 3) == surface_key(&b.s, 3), "editing differs from building afresh");
    a.l.undo(&mut a.s);
    let mut c = Lv::ball();
    c.add(2, &rocks());
    assert!(surface_key(&a.s, 3) == surface_key(&c.s, 3), "undoing the edit differs");
    a.l.undo(&mut a.s);
    assert!(surface_key(&a.s, 3) == bare, "undoing the layer left something");
}

#[test]
fn generator_layers_save_and_open() {
    let mut lv = Lv::ball();
    lv.add(2, &rocks());
    lv.l.paint_mask(&mut lv.s, Vec3::new(0.0, 0.0, 1.0), 0.3, 0.0);
    let mut buf = Vec::new();
    lv.l.write_to(&mut buf).unwrap();
    let (shown, layered) = Layered::read_from(&mut &buf[..], 4).unwrap();
    let layered = layered.expect("layers");
    assert_eq!(layered.layers().len(), 2);
    assert!(layered.layers()[1].generator);
    // The same blocks, the same values to within the form's rounding in the file.
    let (a, b) = (lv.s.keys(), shown.keys());
    assert_eq!(a, b, "the blocks shown differ");
    let most = a.iter().map(|k| lv.s.block(*k).unwrap().iter().zip(shown.block(*k).unwrap().iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
    eprintln!("opened: {} blocks, values within {most:.2e}", a.len());
    assert!(most < 1e-4, "values differ by {most}");
}

#[test]
fn the_mask_fades_a_rock_away() {
    let mut lv = Lv::ball();
    let k = lv.add(2, &rocks());
    let rock = lv.l.generator_instances(k)[0];
    let (p, n) = anchor(&rock, 0.2);
    let was = out_at(&lv.s, p, n);
    lv.l.paint_mask(&mut lv.s, rock.translation, 0.3, 0.0);
    let now = out_at(&lv.s, p, n);
    eprintln!("the rock stood {was:.4} out of the form; masked away, {now:.4}");
    assert!(was > 0.11);
    assert!(now.abs() < 0.005, "the mask left {now}");
    let (open, _, dropped) = closed(&lv.s, 3);
    assert_eq!((open, dropped), (0, 0));
    let _ = Quat::IDENTITY;
}

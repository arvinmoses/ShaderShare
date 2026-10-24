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
    /// A ball at level 0 step `voxel`, with `levels`, layered from `from`.
    fn with(voxel: f32, levels: usize, from: usize) -> Lv {
        let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, levels);
        let l = Layered::new(&mut s, from);
        Lv { l, s }
    }
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

/// The surface seen from `eye` straight down `-w`, `half` wide each way, shaded, as a grey PGM at `path` (when
/// `SCULPT_GEN_PICTURE` names a folder).
fn picture(v: &LevelVolume, eye: Vec3, w: Vec3, half: f32, name: &str) {
    let Ok(dir) = std::env::var("SCULPT_GEN_PICTURE") else { return };
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    let n = 400;
    let mut img = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let (a, b) = (-half + 2.0 * half * x as f32 / n as f32, half - 2.0 * half * y as f32 / n as f32);
            if let Some((_, nrm)) = v.raycast(eye + u * a + t * b, -w, v.levels() - 1, false) {
                img[y * n + x] = (35.0 + 220.0 * nrm.dot((w * 0.8 + u * -0.4 + t * 0.5).normalize()).max(0.0)) as u8;
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255
").into_bytes();
    out.extend(img);
    std::fs::write(format!("{dir}/{name}.pgm"), out).unwrap();
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

/// At the app's scale: adding a generator, dragging its size, a seed change (scattered again), and a big form stroke
/// under it, timed.
#[test]
#[ignore]
fn timing_at_app_scale() {
    use std::time::Instant;
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, 0.0035 * 2.0, 6);
    let mut l = Layered::new(&mut s, 5);
    let size = 4.0 * s.voxel(0);
    let spec = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 12, seed: 1, size, lift: 0.15, random_spin: true, min_spacing: 3.0 * size, size_var: 0.4, stretch_var: 0.35, tilt: 25.0, blend: 0.3, rough: if std::env::var_os("SMOOTH").is_some() { 0.0 } else { 0.25 }, ..GeneratorSpec::default() };
    for level in [1usize, 2, 3] {
        let t = Instant::now();
        let k = l.add_generator(&mut s, "Rocks", level, &spec);
        let add = t.elapsed().as_secs_f64() * 1e3;
        let mut drag = Vec::new();
        l.begin_edit();
        for i in 1..=5 {
            let t = Instant::now();
            l.preview_generator(&mut s, k, &GeneratorSpec { size: size * (1.0 + 0.05 * i as f32), ..spec.clone() });
            drag.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let t = Instant::now();
        l.end_edit_in(&mut s);
        let release = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        l.set_generator(&mut s, k, &GeneratorSpec { seed: 9, ..spec.clone() });
        let reseed = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("level {level}: {} blocks; add {add:.0} ms, size drag (preview) {drag:.0?} ms, at its level on release {release:.0} ms, new seed {reseed:.0} ms", l.layers()[k].blocks);
        l.remove_layer(&mut s, k);
    }
}

#[test]
fn a_previewed_drag_ends_as_the_edit_made_at_once() {
    let mut a = Lv::ball();
    let k = a.add(3, &rocks());
    let to = GeneratorSpec { size: 0.1 + 0.005 * 4.0, ..rocks() };
    a.l.begin_edit();
    for i in 1..=4 {
        a.l.preview_generator(&mut a.s, k, &GeneratorSpec { size: 0.1 + 0.005 * i as f32, ..rocks() });
    }
    a.l.end_edit_in(&mut a.s);
    let mut b = Lv::ball();
    let kb = b.add(3, &rocks());
    b.l.set_generator(&mut b.s, kb, &to);
    assert!(surface_key(&a.s, 3) == surface_key(&b.s, 3), "the drag ended elsewhere");
    // One undo takes the whole drag back.
    a.l.undo(&mut a.s);
    let mut c = Lv::ball();
    c.add(3, &rocks());
    assert!(surface_key(&a.s, 3) == surface_key(&c.s, 3), "undo did not take the drag back whole");
}

/// Fine detail in layers (at two levels) under rocks: every rock stays round over it.
#[test]
fn rocks_over_detail_layers_stay_whole() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    // Fine work over the front, at levels 2 and 3, then layers made of it.
    s.begin_stroke();
    for i in 0..60 {
        let a = i as f32 * 0.37;
        let p = Vec3::new(0.25 * a.cos() * (i as f32 / 60.0), 0.25 * a.sin() * (i as f32 / 60.0), 1.0).normalize();
        s.ball_dab(3, p * 1.002, 0.02, 0.004, i % 2 == 0);
        s.ball_dab(2, p * 1.004, 0.05, 0.01, false);
    }
    s.end_stroke();
    let mut l = Layered::new(&mut s, 2);
    let spec = GeneratorSpec { count: 300, min_spacing: 0.13, size: 0.06, ..rocks() };
    for level in [0usize, 1, 2] {
        let k = l.add_generator(&mut s, "Rocks", level, &spec);
        let bad: Vec<(Vec3, f32)> = l.generator_instances(k).iter().map(|i| (i.translation, top(&s, i, 3))).filter(|(_, t)| (t - 0.06).abs() > 0.006).collect();
        let front = l.generator_instances(k).iter().filter(|i| i.translation.z > 0.85).count();
        eprintln!("generator at level {level}: {front} rocks over the front; tops off: {bad:?}");
        let (open, _, dropped) = closed(&s, 3);
        assert_eq!((open, dropped), (0, 0));
        assert!(bad.is_empty(), "level {level}: rocks not whole over the detail");
        l.remove_layer(&mut s, k);
    }
}

/// Rocks with a wide blend over form blocks of a finer level and over detail layers, pictured
/// (`SCULPT_GEN_PICTURE=dir`): the shown surface from the front, shaded, finest and level 0 alone.
#[test]
#[ignore]
fn picture_a_wide_blend() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    s.begin_stroke();
    for i in 0..30 {
        let p = Vec3::new(-0.3 + 0.02 * i as f32, 0.05 * (i as f32 * 0.7).sin(), 1.0).normalize();
        s.ball_dab(1, p * 1.003, 0.06, 0.012, false);
    }
    for i in 0..60 {
        let a = i as f32 * 0.37;
        let p = Vec3::new(0.3 * a.cos() * (i as f32 / 60.0), 0.3 * a.sin() * (i as f32 / 60.0), 1.0).normalize();
        s.ball_dab(3, p * 1.002, 0.02, 0.004, i % 2 == 0);
        s.ball_dab(2, p * 1.004, 0.05, 0.01, false);
    }
    s.end_stroke();
    let mut l = Layered::new(&mut s, 2);
    let spec = GeneratorSpec { count: 300, min_spacing: 0.13, size: 0.06, blend: 0.6, ..rocks() };
    l.add_generator(&mut s, "Rocks", 0, &spec);
    let dir = std::env::var("SCULPT_GEN_PICTURE").unwrap_or(std::env::temp_dir().to_string_lossy().into_owned());
    let n = 500;
    for cap in [0usize, 3] {
        let mut img = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let (px, py) = (-0.4 + 0.8 * x as f32 / n as f32, 0.4 - 0.8 * y as f32 / n as f32);
                if let Some((_, nrm)) = s.raycast(Vec3::new(px, py, 3.0), -Vec3::Z, cap, false) {
                    let lgt = Vec3::new(-0.4, 0.5, 0.8).normalize();
                    img[y * n + x] = (40.0 + 215.0 * nrm.dot(lgt).max(0.0)) as u8;
                }
            }
        }
        let mut out = format!("P5 {n} {n} 255
").into_bytes();
        out.extend(img);
        std::fs::write(format!("{dir}/blend_cap{cap}.pgm"), out).unwrap();
    }
}

/// A dumped volume (`SCULPT_GEN_DUMP=file.svol`) pictured around a direction (`SCULPT_GEN_AT=x,y,z`) at every cap, into
/// `SCULPT_GEN_PICTURE`.
#[test]
#[ignore]
fn picture_a_dump() {
    let path = std::env::var("SCULPT_GEN_DUMP").expect("SCULPT_GEN_DUMP");
    let v = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap()), 6).unwrap();
    let at: Vec<f32> = std::env::var("SCULPT_GEN_AT").unwrap_or("0.3,0.34,0.89".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let dir = std::env::var("SCULPT_GEN_PICTURE").unwrap_or(std::env::temp_dir().to_string_lossy().into_owned());
    let w = Vec3::new(at[0], at[1], at[2]).normalize();
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    let half: f32 = std::env::var("SCULPT_GEN_HALF").ok().and_then(|x| x.parse().ok()).unwrap_or(0.25);
    let n = 500;
    for cap in 0..v.levels() {
        let mut img = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let (a, b) = (-half + 2.0 * half * x as f32 / n as f32, half - 2.0 * half * y as f32 / n as f32);
                let o = w * 2.0 + u * a + t * b;
                if let Some((_, nrm)) = v.raycast(o, -w, cap, false) {
                    let lgt = (w * 0.8 + u * -0.4 + t * 0.5).normalize();
                    img[y * n + x] = (40.0 + 215.0 * nrm.dot(lgt).max(0.0)) as u8;
                }
            }
        }
        let mut out = format!("P5 {n} {n} 255\n").into_bytes();
        out.extend(img);
        std::fs::write(format!("{dir}/dump_cap{cap}.pgm"), out).unwrap();
    }
    eprintln!("blocks per level {:?}", v.memory().0);
}

/// The review's edits in order (count 12, 30, 60, size 1.5 times, blend 0.6) over detail layers at levels 4 and 5:
/// level 0 holds every rock whole after each.
#[test]
fn the_reviews_edits_keep_every_rock_whole_at_level_0() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 6);
    s.begin_stroke();
    for i in 0..80 {
        let a = i as f32 * 0.37;
        let p = Vec3::new(0.2 * a.cos() * (i as f32 / 80.0), 0.2 * a.sin() * (i as f32 / 80.0), 1.0).normalize();
        s.ball_dab(5, p * 1.0005, 0.004, 0.001, i % 2 == 0);
        s.ball_dab(4, p * 1.001, 0.01, 0.002, false);
    }
    s.end_stroke();
    let mut l = Layered::new(&mut s, 4);
    let size = 0.03 * 2.6;
    let mut spec = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 12, seed: 1, size, lift: 0.15, random_spin: true, min_spacing: 3.0 * size, size_var: 0.4, stretch_var: 0.35, tilt: 25.0, blend: 0.3, ..GeneratorSpec::default() };
    let k = l.add_generator(&mut s, "Generator 1", 0, &spec);
    let edits: [fn(&mut GeneratorSpec); 4] = [|s| s.count = 30, |s| s.count = 60, |s| s.size *= 1.5, |s| s.blend = 0.6];
    for (e, edit) in edits.iter().enumerate() {
        edit(&mut spec);
        l.set_generator(&mut s, k, &spec);
        // Against composing it afresh on the same stack: the same at level 0.
        let mut fresh = Vec::new();
        l.write_to(&mut fresh).unwrap();
        let (again, _) = Layered::read_from(&mut &fresh[..], 6).unwrap();
        let most = s.level_keys(0).iter().map(|bc| {
            let (a, b) = (s.block((0, *bc)).unwrap(), again.block((0, *bc)));
            b.map_or(f32::INFINITY, |b| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max))
        }).fold(0.0f32, f32::max);
        let (mut extra, mut missing) = (0, 0);
        for l in 0..s.levels() {
            let (a, b): (std::collections::HashSet<_>, std::collections::HashSet<_>) = (s.level_keys(l).into_iter().map(|v| v.to_array()).collect(), again.level_keys(l).into_iter().map(|v| v.to_array()).collect());
            extra += a.difference(&b).count();
            missing += b.difference(&a).count();
        }
        eprintln!("edit {}: level 0 against composing afresh: at most {most:.5} apart; blocks only live {extra}, only afresh {missing}", e + 1);
        assert!(most < 1e-3, "edit {}: level 0 differs from composing afresh by {most}", e + 1);
        assert_eq!((extra, missing), (0, 0), "edit {}: the blocks differ from composing afresh", e + 1);
    }
}

/// Jumps across level 0's block faces in a dump (`SCULPT_GEN_DUMP`): where neighbouring samples differ by more than a
/// distance field's would.
#[test]
#[ignore]
fn jumps_in_a_dump() {
    let path = std::env::var("SCULPT_GEN_DUMP").expect("SCULPT_GEN_DUMP");
    let v = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap()), 6).unwrap();
    let vx = v.voxel(0);
    let mut out = Vec::new();
    for bc in v.level_keys(0) {
        let a = v.block((0, bc)).unwrap();
        for axis in 0..3 {
            let mut o = glam::IVec3::ZERO;
            o[axis] = 1;
            let Some(b) = v.block((0, bc + o)) else { continue };
            let mut worst = 0.0f32;
            for p in 0..8 {
                for q in 0..8 {
                    let (mut i, mut j) = (glam::IVec3::ZERO, glam::IVec3::ZERO);
                    i[axis] = 7;
                    i[(axis + 1) % 3] = p;
                    i[(axis + 2) % 3] = q;
                    j[(axis + 1) % 3] = p;
                    j[(axis + 2) % 3] = q;
                    let (x, y) = (a[(i.x + 8 * i.y + 64 * i.z) as usize], b[(j.x + 8 * j.y + 64 * j.z) as usize]);
                    if x.abs() < 0.5 * v.band(0) || y.abs() < 0.5 * v.band(0) {
                        worst = worst.max((x - y).abs() / vx);
                    }
                }
            }
            if worst > 1.6 {
                out.push((worst, bc, axis, v.origin() + (bc * 8).as_vec3() * vx));
            }
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!("{} faces jump; worst: {:?}", out.len(), &out[..out.len().min(12)]);
}

/// Along one pixel's ray of [`picture_a_dump`] (`SCULPT_GEN_PIXEL=x,y`): level 0's field, and its blocks.
#[test]
#[ignore]
fn ray_in_a_dump() {
    let path = std::env::var("SCULPT_GEN_DUMP").expect("SCULPT_GEN_DUMP");
    let v = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap()), 6).unwrap();
    let px: Vec<f32> = std::env::var("SCULPT_GEN_PIXEL").unwrap().split(',').map(|x| x.parse().unwrap()).collect();
    let w = Vec3::new(0.3, 0.34, 0.89).normalize();
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    let (half, n) = (0.35f32, 500.0f32);
    let (a, b) = (-half + 2.0 * half * px[0] / n, half - 2.0 * half * px[1] / n);
    let o = w * 2.0 + u * a + t * b;
    let hit = v.raycast(o, -w, 0, false);
    eprintln!("hit {hit:?}");
    let Some((h, _)) = hit else { return };
    for k in -12..=12 {
        let p = h + w * (k as f32 * 0.005);
        let bc = ((p - v.origin()) / (v.voxel(0) * 8.0)).floor().as_ivec3();
        eprintln!("  {:+.3}: level 0 {:+.4}, block {bc} stored {}", k as f32 * 0.005, v.value_upto(p, 0, false), v.exists(0, bc));
    }
}

/// A live dump (`SCULPT_GEN_DUMP`) against its layers (`SCULPT_GEN_LAYERS`) composed afresh: the level 0 blocks that
/// differ, and whether the generators' blocks hold them.
#[test]
#[ignore]
fn live_against_afresh() {
    let open = |v: &str| std::io::BufReader::new(std::fs::File::open(std::env::var(v).unwrap()).unwrap());
    let live = LevelVolume::read_from(&mut open("SCULPT_GEN_DUMP"), 6).unwrap();
    let (afresh, layered) = Layered::read_from(&mut open("SCULPT_GEN_LAYERS"), 6).unwrap();
    let layered = layered.unwrap();
    let gens: Vec<usize> = layered.layers().iter().enumerate().filter(|(_, x)| x.generator).map(|(k, _)| k).collect();
    eprintln!("layers {:?}", layered.layers().iter().map(|x| (x.name.clone(), x.level, x.blocks, x.generator)).collect::<Vec<_>>());
    for l in 0..live.levels() {
        let mut keys: Vec<_> = live.level_keys(l);
        keys.extend(afresh.level_keys(l));
        keys.sort_unstable_by_key(|k| (k.z, k.y, k.x));
        keys.dedup();
        let mut differ = Vec::new();
        for bc in keys {
            let (a, b) = (live.block((l as u8, bc)), afresh.block((l as u8, bc)));
            let d = match (a, b) {
                (Some(a), Some(b)) => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max),
                _ => f32::INFINITY,
            };
            if d > 1e-3 {
                differ.push((bc, d, a.is_some(), b.is_some()));
            }
        }
        eprintln!("level {l}: {} blocks differ; first {:?}", differ.len(), &differ[..differ.len().min(6)]);
        if !differ.is_empty() {
            let inst = layered.generator_instances(gens[0]);
            let size = live.voxel(l) * 8.0;
            let near = differ.iter().filter(|(bc, ..)| {
                let c = live.origin() + (bc.as_vec3() + Vec3::splat(0.5)) * size;
                inst.iter().any(|i| { let b = i.aabb(); (c.clamp(b.min, b.max) - c).length() < 0.1 })
            }).count();
            let form = differ.iter().filter(|(bc, ..)| layered.form().exists(l, *bc)).count();
            let stale_values = differ.iter().filter(|(bc, ..)| {
                let (a, f) = (live.block((l as u8, *bc)), layered.form().block((l as u8, *bc)));
                a.zip(f).is_some_and(|(a, f)| a.iter().zip(f.iter()).any(|(x, y)| (x - y).abs() > 1e-4))
            }).count();
            eprintln!("   of them near a rock {near}; the form's own {form}; differing from the form's {stale_values}");
        }
    }
    // Lattice points across the first notch's jump: stored, against the form joined with the rocks by hand.
    {
        let spec = layered.generator(gens[0]).unwrap().clone();
        let inst = layered.generator_instances(gens[0]);
        let k = spec.blend * spec.size;
        let set = sculpt_core::generator::InstanceSet::new(inst.clone(), k);
        let v0 = afresh.voxel(0);
        let p0 = Vec3::new(0.5378121, 0.47325364, 1.0140688) + Vec3::new(0.3, 0.34, 0.89).normalize() * 0.027;
        let g = ((p0 - afresh.origin()) / v0).round().as_ivec3();
        for dz in -2..=2 {
            for dy in -2..=2 {
                let i = g + glam::IVec3::new(0, dy, dz);
                let p = afresh.origin() + i.as_vec3() * v0;
                let bc = i.div_euclid(glam::IVec3::splat(8));
                let l = i.rem_euclid(glam::IVec3::splat(8));
                let stored = afresh.block((0, bc)).map(|b| b[(l.x + 8 * l.y + 64 * l.z) as usize]);
                let c = layered.form().block((0, bc)).map(|b| b[(l.x + 8 * l.y + 64 * l.z) as usize]);
                let d = (0..set.len() as u32).map(|j| set.sdf(j, p)).fold(f32::INFINITY, |a, b| if a.is_infinite() { b } else { sculpt_core::generator::volume_gen::smin(a, b, k) });
                let want = c.map(|c| sculpt_core::generator::volume_gen::smin(c, d, k));
                eprintln!("  point {i} (block {bc}): stored {stored:?}, form {c:?}, rocks {d:+.4}, joined {want:?}");
            }
        }
    }
    // Along the ray of a notch: the level 0 blocks, whether the generator holds them, and the instances near.
    let k = gens[0];
    let inst = layered.generator_instances(k);
    let w = Vec3::new(0.3, 0.34, 0.89).normalize();
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    for px in [(300.0f32, 380.0f32), (390.0, 440.0), (350.0, 400.0)] {
        let (a, b) = (-0.35 + 0.7 * px.0 / 500.0, 0.35 - 0.7 * px.1 / 500.0);
        let o = w * 2.0 + u * a + t * b;
        let Some((h, _)) = afresh.raycast(o, -w, 0, false) else { continue };
        eprintln!("pixel {px:?}: hit {h}");
        for s in -6..=8 {
            let p = h + w * (s as f32 * 0.005);
            let bc = ((p - afresh.origin()) / (afresh.voxel(0) * 8.0)).floor().as_ivec3();
            let lo = afresh.origin() + (bc * 8).as_vec3() * afresh.voxel(0);
            let hi = lo + Vec3::splat(8.0 * afresh.voxel(0));
            let nearest = inst.iter().map(|i| {
                let bb = i.aabb();
                let d = (p.clamp(bb.min, bb.max) - p).length();
                (d, (lo.cmple(bb.max).all() && hi.cmpge(bb.min).all()))
            }).fold((f32::INFINITY, false), |a, b| if b.0 < a.0 { b } else { a });
            eprintln!("   {:+.3}: field {:+.4}; block {bc} held {}; nearest instance box {:.4} away (meets the block {})", s as f32 * 0.005, afresh.value_upto(p, 0, false), layered.generator_holds(k, 0, bc), nearest.0, nearest.1);
        }
    }
}

/// The layers of a dump (`SCULPT_GEN_LAYERS`) composed afresh, pictured as [`picture_a_dump`] does, at level 0.
#[test]
#[ignore]
fn picture_layers_afresh() {
    let (v, _) = Layered::read_from(&mut std::io::BufReader::new(std::fs::File::open(std::env::var("SCULPT_GEN_LAYERS").unwrap()).unwrap()), 6).unwrap();
    let dir = std::env::var("SCULPT_GEN_PICTURE").unwrap_or(std::env::temp_dir().to_string_lossy().into_owned());
    let w = Vec3::new(0.3, 0.34, 0.89).normalize();
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    let (half, n) = (0.35f32, 500usize);
    for cap in [0usize, 5] {
        let mut img = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let (a, b) = (-half + 2.0 * half * x as f32 / n as f32, half - 2.0 * half * y as f32 / n as f32);
                if let Some((_, nrm)) = v.raycast(w * 2.0 + u * a + t * b, -w, cap, false) {
                    img[y * n + x] = (40.0 + 215.0 * nrm.dot((w * 0.8 + u * -0.4 + t * 0.5).normalize()).max(0.0)) as u8;
                }
            }
        }
        let mut out = format!("P5 {n} {n} 255\n").into_bytes();
        out.extend(img);
        std::fs::write(format!("{dir}/afresh_cap{cap}.pgm"), out).unwrap();
    }
}

/// A blend wider than level 0 holds distances (six of its steps), over a form a stroke made distances again (so its
/// blocks hold a bound past six steps, and the blocks past them their band): level 0's field stays whole, with no
/// jump at any block's face.
#[test]
fn a_blend_wider_than_the_form_holds_distances_leaves_no_steps() {
    let mut lv = Lv::ball();
    lv.l.begin_stroke();
    for s in 0..10 {
        let dir = Vec3::new(-0.3 + 0.06 * s as f32, 0.1, 1.0).normalize();
        let Some((hit, n)) = lv.l.form().raycast(dir * 3.0, -dir, 0, true) else { continue };
        let _ = lv.l.plane_dab(&mut lv.s, 0, hit, n, 0.4, 0.03, 0.5, 0.4, false);
    }
    lv.l.end_stroke(&mut lv.s);
    lv.l.settle(&mut lv.s);
    let v0 = lv.s.voxel(0);
    let spec = GeneratorSpec { count: 8, min_spacing: 0.5, size: 0.25, blend: 0.9, ..rocks() };
    assert!(spec.blend * spec.size > 6.0 * v0);
    let k = lv.add(0, &spec);
    let _ = k;
    // Neighbouring samples near the surface differ by no more than a distance field's would.
    let mut worst = 0.0f32;
    let band = lv.s.band(0);
    for bc in lv.s.level_keys(0) {
        let Some(a) = lv.s.block((0, bc)) else { continue };
        for axis in 0..3 {
            let mut o = glam::IVec3::ZERO;
            o[axis] = 1;
            let Some(b) = lv.s.block((0, bc + o)) else { continue };
            for p in 0..8 {
                for q in 0..8 {
                    let (mut i, mut j) = (glam::IVec3::ZERO, glam::IVec3::ZERO);
                    i[axis] = 7;
                    i[(axis + 1) % 3] = p;
                    i[(axis + 2) % 3] = q;
                    j[(axis + 1) % 3] = p;
                    j[(axis + 2) % 3] = q;
                    let (x, y) = (a[(i.x + 8 * i.y + 64 * i.z) as usize], b[(j.x + 8 * j.y + 64 * j.z) as usize]);
                    if x.abs() < 3.0 * v0 || y.abs() < 3.0 * v0 {
                        worst = worst.max((x - y).abs() / v0);
                    }
                }
            }
        }
    }
    eprintln!("level 0 near the surface: neighbouring samples across block faces at most {worst:.2} steps apart (band {:.0} steps)", band / v0);
    assert!(worst < 1.6, "a step at a block's face: {worst} steps");
}

/// Rough rocks: their surface varies by about the roughness, within it, and stays closed at every level.
#[test]
fn rough_rocks_vary_within_their_roughness() {
    for kind in [sculpt_core::noise::NoiseKind::Fbm, sculpt_core::noise::NoiseKind::Cellular] {
        let mut lv = Lv::ball();
        let spec = GeneratorSpec { rough: 0.25, rough_scale: 2.5, rough_kind: kind, lift: 0.4, ..rocks() };
        let k = lv.add(2, &spec);
        let amp = 0.25 * 0.1;
        let (mut lo, mut hi, mut sum, mut sum2, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32, 0.0f32, 0.0f32);
        for i in lv.l.generator_instances(k) {
            let up = i.rotation * Vec3::Y;
            let (t1, t2) = (up.any_orthonormal_vector(), up.cross(up.any_orthonormal_vector()));
            for a in 0..16 {
                for e in 0..3 {
                    let (th, ph) = (a as f32 * 0.3927, e as f32 * 0.35);
                    let d = (up * ph.cos() + (t1 * th.cos() + t2 * th.sin()) * ph.sin()).normalize();
                    // Along `d` from the centre: the radius of the rock there (a sphere of 0.1, stretched none here).
                    let Some((q, _)) = lv.s.raycast(i.translation + d * 0.3, -d, 3, false) else { continue };
                    let r = (q - i.translation).dot(d);
                    lo = lo.min(r);
                    hi = hi.max(r);
                    sum += r;
                    sum2 += r * r;
                    n += 1.0;
                }
            }
        }
        let sd = (sum2 / n - (sum / n).powi(2)).max(0.0).sqrt();
        eprintln!("{kind:?}: radius {lo:.4} to {hi:.4}, spread {sd:.4} (smooth 0.1, roughness up to {amp})");
        assert!(sd > 0.2 * amp, "{kind:?}: hardly rough");
        assert!(lo > 0.1 - amp - 0.01 && hi < 0.1 + amp + 0.01, "{kind:?}: rougher than its roughness");
        for cap in 0..4 {
            let (open, _, dropped) = closed(&lv.s, cap);
            assert_eq!((open, dropped), (0, 0), "{kind:?} cap {cap}");
        }
    }
}

/// Rough rocks of each kind pictured from the front (`SCULPT_GEN_PICTURE`).
#[test]
#[ignore]
fn picture_rough_rocks() {
    let dir = std::env::var("SCULPT_GEN_PICTURE").unwrap_or(std::env::temp_dir().to_string_lossy().into_owned());
    for (name, kind) in [("fbm", sculpt_core::noise::NoiseKind::Fbm), ("ridged", sculpt_core::noise::NoiseKind::Ridged), ("cellular", sculpt_core::noise::NoiseKind::Cellular)] {
        let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5);
        let mut l = Layered::new(&mut s, 3);
        let spec = GeneratorSpec { count: 40, min_spacing: 0.3, size: 0.12, rough: 0.3, rough_scale: 1.5, rough_kind: kind, size_var: 0.4, stretch_var: 0.4, tilt: 25.0, random_spin: true, blend: 0.3, lift: 0.1, ..rocks() };
        l.add_generator(&mut s, "Rocks", 3, &spec);
        let n = 500;
        let mut img = vec![0u8; n * n];
        for y in 0..n {
            for x in 0..n {
                let (px, py) = (-0.6 + 1.2 * x as f32 / n as f32, 0.6 - 1.2 * y as f32 / n as f32);
                if let Some((_, nrm)) = s.raycast(Vec3::new(px, py, 3.0), -Vec3::Z, 4, false) {
                    img[y * n + x] = (35.0 + 220.0 * nrm.dot(Vec3::new(-0.4, 0.5, 0.8).normalize()).max(0.0)) as u8;
                }
            }
        }
        let mut out = format!("P5 {n} {n} 255\n").into_bytes();
        out.extend(img);
        std::fs::write(format!("{dir}/rough_{name}.pgm"), out).unwrap();
    }
}

/// Detail painted on a rock, in a layer over the generator: it stands on the rock, and when a stroke on the form
/// carries the rock along, the detail goes with it.
#[test]
fn detail_on_a_rock_rides_the_rock() {
    let mut lv = Lv::with(0.02, 5, 3);
    let k = lv.add(1, &GeneratorSpec { size: 0.12, lift: 0.2, ..rocks() });
    let top_layer = lv.l.add_layer("On the rocks", 4);
    assert!(top_layer > k);
    // The rock nearest the front, and bumps painted on its top (three, at level 4).
    let inst = lv.l.generator_instances(k);
    let (ri, rock) = inst.iter().enumerate().max_by(|a, b| a.1.translation.z.total_cmp(&b.1.translation.z)).map(|(i, r)| (i, *r)).unwrap();
    let up = rock.rotation * Vec3::Y;
    let side = up.any_orthonormal_vector();
    let spots: Vec<Vec3> = (0..3).map(|j| (up * 0.12 + side * (0.03 * (j as f32 - 1.0))).normalize() * 0.12).collect();
    let rock_out = |lv: &Lv, r: &Instance, d: Vec3| -> f32 {
        // In the rock's own frame: turned as the rock turned since the bumps were painted.
        let dir = (r.rotation * rock.rotation.inverse() * d).normalize();
        // From close by: a ray from far off can step over a bump this thin.
        lv.s.raycast(r.translation + dir * 0.2, -dir, 4, false).map_or(f32::NAN, |(q, _)| (q - r.translation).dot(dir))
    };
    let bare: Vec<f32> = spots.iter().map(|d| rock_out(&lv, &rock, *d)).collect();
    for d in &spots {
        lv.s.begin_stroke();
        lv.s.ball_dab(4, rock.translation + d.normalize() * (bare[0] + 0.002), 0.008, 0.002, false);
        lv.s.end_stroke();
        lv.l.take_detail(&mut lv.s);
    }
    let painted: Vec<f32> = spots.iter().zip(&bare).map(|(d, b)| rock_out(&lv, &rock, *d) - b).collect();
    eprintln!("bumps painted on the rock: {painted:.4?} out of it; layer blocks {}", lv.l.layers()[top_layer].blocks);
    assert!(painted.iter().all(|h| *h > 0.004), "the bumps did not stand on the rock");
    let (open, _, dropped) = closed(&lv.s, 4);
    assert_eq!((open, dropped), (0, 0));
    picture(&lv.s, rock.translation + up * 0.3, up, 0.2, "rock_detail_before");
    // Composing it afresh (as opening the file does) shows the same.
    {
        let mut buf = Vec::new();
        lv.l.write_to(&mut buf).unwrap();
        let (fresh, _) = Layered::read_from(&mut &buf[..], 5).unwrap();
        assert_eq!(lv.s.keys(), fresh.keys());
        assert!(lv.s.keys().iter().all(|k| lv.s.block(*k).unwrap().iter().zip(fresh.block(*k).unwrap().iter()).all(|(x, y)| (x - y).abs() < 1e-4)));
    }
    // A big stroke on the form under the rock.
    let p0 = rock.translation - up * (0.2 * 0.12);
    lv.l.begin_stroke();
    for s in 0..12 {
        let dir = (p0 + Vec3::new(-0.12 + 0.02 * s as f32, 0.0, 0.0)).normalize();
        let Some((hit, n)) = lv.l.form().raycast(dir * 3.0, -dir, 0, true) else { continue };
        let _ = lv.l.plane_dab(&mut lv.s, 0, hit, n, 0.35, 0.035, 0.5, 0.4, false);
    }
    lv.l.end_stroke(&mut lv.s);
    lv.l.settle(&mut lv.s);
    let moved = lv.l.generator_instances(k)[ri];
    let shift = (moved.translation - rock.translation).length();
    let tops: Vec<f32> = spots.iter().map(|d| rock_out(&lv, &moved, *d)).collect();
    eprintln!("the rock moved {shift:.4}; out of its centre now {tops:.4?} (bare {bare:.4?}, painted {:.4?})", spots.iter().zip(&bare).zip(&painted).map(|((_, b), p)| b + p).collect::<Vec<_>>());
    assert!(shift > 0.01, "the rock did not move");
    picture(&lv.s, moved.translation + up * 0.3, up, 0.2, "rock_detail_after");
    for ((t, b), p) in tops.iter().zip(&bare).zip(&painted) {
        assert!((t - (b + p)).abs() < 0.25 * p + 0.001, "a bump did not ride the rock: {t} against {}", b + p);
    }
    let (open, _, dropped) = closed(&lv.s, 4);
    assert_eq!((open, dropped), (0, 0));
}

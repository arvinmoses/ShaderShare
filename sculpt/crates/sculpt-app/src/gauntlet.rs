//! The brush checklist: scripted scenes of clay, trim and smooth as they are used (close and far, rows, crossings,
//! scrubbing, a trim zoomed out over clay made zoomed in, smoothing over it all), each scored for the marks a brush
//! should not leave and for how fast it went. Run with
//! `cargo test --release -p sculpt-app gauntlet -- --ignored --nocapture`; `SCULPT_GAUNTLET=name,name` picks scenes,
//! `SCULPT_GAUNTLET_OUT=dir` saves each scene's volume (`<name>.svol`) and its scores (`scores.txt`) for the renders.

use super::*;
use rayon::prelude::*;
use std::time::Instant;

/// One stroke of a scene: the brush, its radius and the pixel size at the brush (which pick its level as in the
/// app), its path in the scene's patch (offsets along its two axes), passes back and forth, and dabs a frame.
struct Line {
    brush: Brush,
    r: f32,
    px: f32,
    from: (f32, f32),
    to: (f32, f32),
    passes: usize,
    per_frame: usize,
    /// On a scene with a sculpt layer: whether the stroke works on the form under it rather than the layer.
    form: bool,
}

fn line(brush: Brush, r: f32, px: f32, from: (f32, f32), to: (f32, f32)) -> Line {
    Line { brush, r, px, from, to, passes: 1, per_frame: 2, form: false }
}

/// Clay rows across the patch, close (level 4): five rows, two passes each, a little apart.
fn rows(out: &mut Vec<Line>, r: f32, px: f32) {
    for row in 0..5 {
        for pass in 0..2 {
            let y = -0.03 + 0.014 * row as f32 + 0.003 * pass as f32;
            out.push(line(Brush::Clay, r, px, (-0.06, y), (0.06, y + 0.004)));
        }
    }
}

struct Scene {
    name: &'static str,
    /// What is scored: a square of this half-width on the patch.
    half: f32,
    lines: Vec<Line>,
    /// Strokes on a new sculpt layer (see `VolumeMode::new_sculpt_layer`) rather than on the volume.
    layer: bool,
}

fn scenes() -> Vec<Scene> {
    let mut out = Vec::new();
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    out.push(Scene { layer: false, name: "clay_close_rows", half: 0.08, lines: l });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    for k in 0..3 {
        let x = -0.03 + 0.03 * k as f32;
        l.push(line(Brush::Clay, 0.016, 0.0004, (x, -0.05), (x + 0.005, 0.05)));
    }
    out.push(Scene { layer: false, name: "clay_close_crossed", half: 0.08, lines: l });
    out.push(Scene { layer: false, name: "clay_close_scrubbed", half: 0.06, lines: vec![Line { passes: 6, ..line(Brush::Clay, 0.016, 0.0004, (-0.04, 0.0), (0.04, 0.005)) }] });
    out.push(Scene { layer: false, name: "clay_very_close", half: 0.03, lines: (0..3).map(|k| line(Brush::Clay, 0.008, 0.0002, (-0.02, -0.008 + 0.008 * k as f32), (0.02, -0.006 + 0.008 * k as f32))).collect() });
    out.push(Scene { layer: false, name: "clay_mid_crossed", half: 0.2, lines: vec![line(Brush::Clay, 0.08, 0.0015, (-0.15, -0.03), (0.15, 0.03)), line(Brush::Clay, 0.08, 0.0015, (-0.03, -0.15), (0.03, 0.15))] });
    out.push(Scene { layer: false, name: "clay_far_single", half: 0.45, lines: vec![line(Brush::Clay, 0.25, 0.003, (-0.4, -0.1), (0.4, 0.1))] });
    out.push(Scene { layer: false, name: "clay_far_crossed", half: 0.45, lines: vec![line(Brush::Clay, 0.25, 0.003, (-0.4, -0.1), (0.4, 0.1)), line(Brush::Clay, 0.25, 0.003, (-0.1, -0.4), (0.1, 0.4))] });
    out.push(Scene { layer: false, name: "trim_close_fresh", half: 0.08, lines: vec![line(Brush::Trim, 0.038, 0.000476, (-0.07, 0.0), (0.07, 0.01)), line(Brush::Trim, 0.03, 0.000476, (-0.03, -0.05), (0.0, 0.06))] });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    l.push(line(Brush::Trim, 0.02, 0.0004, (-0.07, -0.01), (0.07, 0.0)));
    l.push(line(Brush::Trim, 0.02, 0.0004, (0.0, -0.05), (0.01, 0.05)));
    out.push(Scene { layer: false, name: "trim_close_over_rows", half: 0.08, lines: l });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    l.push(line(Brush::Trim, 0.038, 0.000476, (-0.07, 0.0), (0.07, 0.01)));
    l.push(line(Brush::Trim, 0.03, 0.000476, (-0.03, -0.05), (0.0, 0.06)));
    out.push(Scene { layer: false, name: "trim_coarser_over_rows", half: 0.08, lines: l });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    l.push(Line { passes: 2, ..line(Brush::Smooth, 0.02, 0.0004, (-0.07, -0.01), (0.07, 0.0)) });
    out.push(Scene { layer: false, name: "smooth_close_over_rows", half: 0.08, lines: l });
    out.push(Scene { layer: false, name: "smooth_far_over_clay", half: 0.45, lines: vec![line(Brush::Clay, 0.25, 0.003, (-0.4, -0.1), (0.4, 0.1)), Line { passes: 2, ..line(Brush::Smooth, 0.25, 0.003, (-0.4, 0.05), (0.4, -0.05)) }] });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    for x in &mut l {
        x.per_frame = 8;
    }
    l.push(Line { per_frame: 8, ..line(Brush::Trim, 0.038, 0.000476, (-0.07, 0.0), (0.07, 0.01)) });
    out.push(Scene { layer: false, name: "fast_drags", half: 0.08, lines: l });
    // On a sculpt layer: close rows, big strokes, and the form moved under the rows by a big stroke.
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    out.push(Scene { layer: true, name: "layer_clay_close_rows", half: 0.08, lines: l });
    out.push(Scene { layer: true, name: "layer_clay_far_crossed", half: 0.45, lines: vec![line(Brush::Clay, 0.25, 0.003, (-0.4, -0.1), (0.4, 0.1)), line(Brush::Clay, 0.25, 0.003, (-0.1, -0.4), (0.1, 0.4))] });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    l.push(line(Brush::Clay, 0.25, 0.003, (-0.4, -0.05), (0.4, 0.05)));
    l.push(line(Brush::Trim, 0.038, 0.000476, (-0.07, 0.0), (0.07, 0.01)));
    out.push(Scene { layer: true, name: "layer_rows_then_big", half: 0.08, lines: l });
    let mut l = Vec::new();
    rows(&mut l, 0.016, 0.0004);
    l.push(Line { form: true, ..line(Brush::Clay, 0.25, 0.003, (-0.4, -0.05), (0.4, 0.05)) });
    out.push(Scene { layer: true, name: "layer_form_under_rows", half: 0.08, lines: l });
    out
}

#[derive(Default)]
struct Scores {
    /// Heights less their local mean (over three steps of the finest level stroked), rms and worst, in those steps.
    ripple: f64,
    worst: f32,
    /// Points standing off their neighbourhood by over a fifth of a step: dots, ticks and step edges.
    marks: usize,
    /// How far the surface's slope turns from its local mean (over the same window), rms in degrees: what shading
    /// shows of ripples too shallow to see in height. And points turned by over 4 degrees.
    shade: f64,
    shade_marks: usize,
    /// Samples inside the solid shallower than a distance field allows (hidden folds a later stroke breaks into).
    phantoms: usize,
    dab_mean: f64,
    dab_worst: f64,
    frame_worst: f64,
    end_worst: f64,
}

/// See `phantoms` in the core's tests: samples inside, away from the surface, shallower than both their neighbours
/// along an axis by more than a distance field can be.
fn phantoms(v: &LevelVolume, l: usize) -> usize {
    let (vx, band) = (v.voxel(l), v.band(l));
    v.level_keys(l)
        .par_iter()
        .map(|&bc| {
            let mut count = 0;
            for q in 0..512 {
                let i = bc * 8 + glam::IVec3::new(q % 8, (q / 8) % 8, q / 64);
                let s = v.sample(l, i, false);
                if s >= 0.0 || s <= -0.999 * band {
                    continue;
                }
                let axes = [glam::IVec3::X, glam::IVec3::Y, glam::IVec3::Z];
                if axes.iter().any(|&d| v.sample(l, i - d, false) >= 0.0 || v.sample(l, i + d, false) >= 0.0) {
                    continue;
                }
                let d0 = -s / vx;
                let most = (d0 * d0 + 1.0).sqrt() - d0 + 0.15;
                if axes.iter().any(|&d| s > v.sample(l, i - d, false).max(v.sample(l, i + d, false)) + most * vx) {
                    count += 1;
                }
            }
            count
        })
        .sum()
}

fn run(scene: &Scene, out: Option<&std::path::Path>) -> Scores {
    let mut mode = VolumeMode::ball(1e6);
    if scene.layer {
        mode.new_sculpt_layer();
    }
    let centre = Vec3::new(0.3, -0.05, 0.95).normalize();
    let (t1, t2) = (Vec3::Y.cross(centre).normalize(), centre.cross(Vec3::Y.cross(centre).normalize()));
    let mut s = Scores::default();
    let (mut dabs, mut finest) = (0usize, 0usize);
    for ln in &scene.lines {
        let (pa, pb) = (centre + t1 * ln.from.0 + t2 * ln.from.1, centre + t1 * ln.to.0 + t2 * ln.to.1);
        let steps = ((pa - pb).length() / (0.15 * ln.r)).ceil() as usize;
        if scene.layer {
            mode.target = if ln.form { Target::Form } else { Target::Detail };
        }
        mode.begin_stroke();
        let mut n = 0;
        for pass in 0..ln.passes {
            for k in 0..=steps {
                let t = k as f32 / steps as f32;
                let d = (if pass % 2 == 0 { pa.lerp(pb, t) } else { pb.lerp(pa, t) }).normalize();
                let Some((hit, _)) = mode.raycast(d * 3.0, -d) else { continue };
                // As the app does: the frame's dabs queued, a frame's share of them done, then the frame's tick.
                mode.queue_dab(hit, ln.r, ln.px, 0.8, 0.3, ln.brush);
                n += 1;
                if n % ln.per_frame == 0 {
                    let t0 = Instant::now();
                    let k = mode.run_frame_dabs();
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    s.dab_mean += ms;
                    s.dab_worst = s.dab_worst.max(ms / k.max(1) as f64);
                    dabs += k;
                    mode.tick();
                    s.frame_worst = s.frame_worst.max(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        finest = finest.max(mode.level);
        // The pen up: the frames that do what is queued, then those until the stroke's end is all in.
        while mode.dabs_queued() {
            let t0 = Instant::now();
            let k = mode.run_frame_dabs();
            s.dab_mean += t0.elapsed().as_secs_f64() * 1e3;
            dabs += k;
            mode.tick();
            s.frame_worst = s.frame_worst.max(t0.elapsed().as_secs_f64() * 1e3);
        }
        let t0 = Instant::now();
        mode.end_stroke();
        let mut worst = t0.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        while mode.busy() && start.elapsed().as_secs_f64() < 20.0 {
            let t0 = Instant::now();
            mode.tick();
            worst = worst.max(t0.elapsed().as_secs_f64() * 1e3);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        mode.settle();
        s.end_worst = s.end_worst.max(worst);
    }
    s.dab_mean /= dabs.max(1) as f64;
    // Heights over the patch along its normal, and what stands off its neighbourhood.
    let vx = mode.vol.voxel(finest);
    let g = (0.5 * vx).max(2.0 * scene.half / 500.0);
    let m = (2.0 * scene.half / g) as usize;
    let top = mode.vol.levels() - 1;
    let vol = &mode.vol;
    let h: Vec<f32> = (0..m * m)
        .into_par_iter()
        .map(|q| {
            let p = centre + t1 * (-scene.half + g * (q % m) as f32) + t2 * (-scene.half + g * (q / m) as f32);
            vol.raycast(p + centre * 0.1, -centre, top, false).map_or(f32::NAN, |(q, _)| (q - centre).dot(centre))
        })
        .collect();
    let k = ((1.5 * vx / g).ceil() as usize).max(1);
    let offs: Vec<f32> = (0..m * m)
        .into_par_iter()
        .map(|q| {
            let (i, j) = (q % m, q / m);
            if i < k || j < k || i + k >= m || j + k >= m {
                return f32::NAN;
            }
            let mut mean = 0.0;
            for b in j - k..=j + k {
                for a in i - k..=i + k {
                    mean += h[b * m + a];
                }
            }
            (h[q] - mean / ((2 * k + 1) * (2 * k + 1)) as f32) / vx
        })
        .collect();
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for &d in offs.iter().filter(|d| d.is_finite()) {
        sum += (d * d) as f64;
        cnt += 1;
        s.worst = s.worst.max(d.abs());
        s.marks += (d.abs() > 0.2) as usize;
    }
    s.ripple = (sum / cnt.max(1) as f64).sqrt();
    // Slopes by central differences, less their mean over the window.
    let slope = |q: usize| -> Option<(f32, f32)> {
        let (i, j) = (q % m, q / m);
        if i < 1 || j < 1 || i + 1 >= m || j + 1 >= m {
            return None;
        }
        let sx = (h[q + 1] - h[q - 1]) / (2.0 * g);
        let sy = (h[q + m] - h[q - m]) / (2.0 * g);
        (sx.is_finite() && sy.is_finite()).then_some((sx, sy))
    };
    let slopes: Vec<Option<(f32, f32)>> = (0..m * m).into_par_iter().map(slope).collect();
    let turns: Vec<f32> = (0..m * m)
        .into_par_iter()
        .filter_map(|q| {
            let (i, j) = (q % m, q / m);
            if i < k + 1 || j < k + 1 || i + k + 1 >= m || j + k + 1 >= m {
                return None;
            }
            let (sx, sy) = slopes[q]?;
            let (mut ax, mut ay, mut n) = (0.0, 0.0, 0.0);
            for b in j - k..=j + k {
                for a in i - k..=i + k {
                    if let Some((x, y)) = slopes[b * m + a] {
                        ax += x;
                        ay += y;
                        n += 1.0;
                    }
                }
            }
            let (dx, dy) = (sx - ax / n, sy - ay / n);
            Some((dx * dx + dy * dy).sqrt().atan().to_degrees())
        })
        .collect();
    s.shade = (turns.iter().map(|t| (t * t) as f64).sum::<f64>() / turns.len().max(1) as f64).sqrt();
    s.shade_marks = turns.iter().filter(|&&t| t > 4.0).count();
    s.phantoms = (1..mode.vol.levels()).map(|l| phantoms(&mode.vol, l)).sum();
    if let Some(dir) = out {
        mode.write_to(&mut std::fs::File::create(dir.join(format!("{}.svol", scene.name))).unwrap()).unwrap();
        std::fs::write(dir.join(format!("{}.offs", scene.name)), offs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    s
}

#[test]
#[ignore]
fn gauntlet() {
    let pick: Option<Vec<String>> = std::env::var("SCULPT_GAUNTLET").ok().map(|v| v.split(',').map(str::to_string).collect());
    let out = std::env::var("SCULPT_GAUNTLET_OUT").ok().map(std::path::PathBuf::from);
    if let Some(d) = &out {
        std::fs::create_dir_all(d).unwrap();
    }
    let mut table = format!("{:<24} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7}\n", "scene", "ripple", "worst", "marks", "shade", "turns", "phantom", "dab ms", "dab max", "frame", "end");
    for scene in scenes() {
        if pick.as_ref().is_some_and(|p| !p.iter().any(|n| n == scene.name)) {
            continue;
        }
        let s = run(&scene, out.as_deref());
        let row = format!(
            "{:<24} {:>7.4} {:>6.3} {:>6} {:>6.2} {:>6} {:>8} {:>7.2} {:>7.1} {:>7.1} {:>7.1}\n",
            scene.name, s.ripple, s.worst, s.marks, s.shade, s.shade_marks, s.phantoms, s.dab_mean, s.dab_worst, s.frame_worst, s.end_worst
        );
        eprint!("{row}");
        table += &row;
    }
    eprintln!("\n{table}");
    if let Some(d) = &out {
        std::fs::write(d.join("scores.txt"), table).unwrap();
    }
}

//! Diagnostics run by hand: each reads a file or a setting from the environment, and does nothing without it.

use super::*;
use super::tests::{bent, pieces, spikes};

/// One of the user's own files from before mesh layers (`SCULPT_OLD_LAYERS_FILE`, read only): it opens, merged,
/// with the note.
#[test]
#[ignore]
fn an_old_layers_file_opens_merged() {
    let Ok(path) = std::env::var("SCULPT_OLD_LAYERS_FILE") else { return };
    let bytes = std::fs::read(&path).unwrap();
    let t = Instant::now();
    let mode = VolumeMode::read_from(&mut &bytes[..]).unwrap();
    let (counts, mb) = mode.vol.memory();
    eprintln!("{path}: {} bytes, opened in {:.2} s; blocks per level {counts:?} ({:.0} MB); note {:?}", bytes.len(), t.elapsed().as_secs_f64(), mb as f64 / 1e6, mode.lossy_open);
    assert!(!mode.vol.keys().is_empty());
    assert!(mode.layer.is_none() && mode.mesh.is_none());
    assert!(mode.lossy_open.as_ref().is_some_and(|n| n.contains("old volume layers were merged")), "{:?}", mode.lossy_open);
}

/// Volumes dumped from the review's frames (`SCULPT_DEMO_DUMP`), in the folder `SCULPT_DUMPS`: their surfaces'
/// spiky vertices, by the finest level where they are, and a few of them.
#[test]
#[ignore]
fn dumped_frames_spikes() {
    let Ok(dir) = std::env::var("SCULPT_DUMPS") else { return };
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "svol")).collect();
    files.sort();
    for f in files {
        let vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(&f).unwrap()), OPEN_LEVELS).unwrap();
        let (mesh, _) = vol.surface(vol.levels() - 1);
        let sp = spikes(&mesh, 0.6);
        let lv: Vec<usize> = (0..vol.levels()).map(|l| sp.iter().filter(|p| vol.level_of(**p) == l).count()).collect();
        eprintln!("{}: {} vertices, {} spiky, by level {lv:?}; e.g. {:?}", f.file_name().unwrap().to_string_lossy(), mesh.positions.len(), sp.len(), &sp[..sp.len().min(4)]);
    }
}

/// A shaded height map of `vol`'s surface over a square `half` across around `c` (seen along its normal), as a PGM.
fn height_map(vol: &LevelVolume, c: Vec3, half: f32, px: usize, path: &std::path::Path) {
    use rayon::prelude::*;
    let n = c.normalize();
    let u = n.cross(Vec3::Y).normalize();
    let v = n.cross(u);
    let step = 2.0 * half / px as f32;
    let hs: Vec<f32> = (0..px * px)
        .into_par_iter()
        .map(|k| {
            let (i, j) = ((k % px) as f32, (k / px) as f32);
            let q = c + u * (-half + step * i) + v * (-half + step * j);
            vol.raycast(q + n * 0.3, -n, vol.levels() - 1, false).map_or(f32::NAN, |(hit, _)| (hit - c).dot(n))
        })
        .collect();
    let at = |i: usize, j: usize| hs[j.min(px - 1) * px + i.min(px - 1)];
    let light = Vec3::new(-0.5, 0.6, 0.6).normalize();
    let mut img = format!("P5 {px} {px} 255
").into_bytes();
    for j in 0..px {
        for i in 0..px {
            let dx = (at(i + 1, j) - at(i.saturating_sub(1), j)) / (2.0 * step);
            let dy = (at(i, j + 1) - at(i, j.saturating_sub(1))) / (2.0 * step);
            let nn = Vec3::new(-dx, -dy, 1.0).normalize();
            img.push((nn.dot(light).clamp(0.0, 1.0) * 255.0) as u8);
        }
    }
    std::fs::write(path, img).unwrap();
    // The finest level at each pixel, as grey.
    let levels: Vec<u8> = (0..px * px)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k % px, k / px);
            let h = at(i, j);
            if h.is_nan() {
                return 0;
            }
            let q = c + u * (-half + step * i as f32) + v * (-half + step * j as f32) + n * h;
            (vol.level_of(q) * 50) as u8
        })
        .collect();
    let mut lv = format!("P5 {px} {px} 255
").into_bytes();
    lv.extend(levels);
    std::fs::write(path.with_extension("lvl.pgm"), lv).unwrap();
    // Where the surface jumps between neighbouring pixels by more than four pixels' width: the levels and blocks
    // on its two sides.
    if std::env::var_os("SCULPT_REPLAY_JUMPS").is_some() {
        let mut seen = std::collections::BTreeMap::new();
        for j in 0..px - 1 {
            for i in 0..px - 1 {
                for (a, b) in [((i, j), (i + 1, j)), ((i, j), (i, j + 1))] {
                    let (ha, hb) = (at(a.0, a.1), at(b.0, b.1));
                    if (ha - hb).abs() > 4.0 * step {
                        let (hi, h) = if ha > hb { (a, ha) } else { (b, hb) };
                        let q = c + u * (-half + step * hi.0 as f32) + v * (-half + step * hi.1 as f32) + n * h;
                        let l = vol.level_of(q);
                        let bc = ((q - vol.origin()) / (vol.voxel(l) * 8.0)).floor().as_ivec3();
                        *seen.entry((l, bc.x, bc.y, bc.z)).or_insert(0) += 1;
                    }
                }
            }
        }
        eprintln!("{}: jumps at the high side's finest level and block: {:?}", path.display(), seen.iter().take(12).collect::<Vec<_>>());
    }
}

/// The level 0 field's slope at samples near the surface around `patch`, in quarters (0-0.25, ..., 1.75+).
fn slope(vol: &LevelVolume, patch: Vec3, half: f32) -> [usize; 8] {
            let vx = vol.voxel(0);
            let (lo, hi) = (((patch - Vec3::splat(half.max(0.1)) - vol.origin()) / vx).floor().as_ivec3(), ((patch + Vec3::splat(half.max(0.1)) - vol.origin()) / vx).ceil().as_ivec3());
            let mut hist = [0usize; 8];
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let i = glam::IVec3::new(x, y, z);
                        let f = vol.sample(0, i, false);
                        if f.abs() > 0.7 * vx {
                            continue;
                        }
                        let g = glam::Vec3::new(
                            vol.sample(0, i + glam::IVec3::X, false) - vol.sample(0, i - glam::IVec3::X, false),
                            vol.sample(0, i + glam::IVec3::Y, false) - vol.sample(0, i - glam::IVec3::Y, false),
                            vol.sample(0, i + glam::IVec3::Z, false) - vol.sample(0, i - glam::IVec3::Z, false),
                        ).length() / (2.0 * vx);
                        hist[((g / 0.25) as usize).min(7)] += 1;
                    }
                }
            }
            hist
}

/// The review's big stroke replayed on the volume dumped just before it (`SCULPT_REPLAY_DIR` holds `dump_01604.svol`
/// and the session's dab log `dabs.txt`), drawn around the patch before and after its dabs.
#[test]
#[ignore]
fn replay_the_big_stroke() {
    let Ok(dir) = std::env::var("SCULPT_REPLAY_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    let vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(dir.join("dump_01604.svol")).unwrap()), OPEN_LEVELS).unwrap();
    let log = std::fs::read_to_string(dir.join("dabs.txt")).unwrap();
    let last = log.rfind("BEGIN").unwrap();
    let dabs: Vec<Vec<f32>> = log[last..].lines().filter(|l| l.starts_with("DAB")).map(|l| l.split_whitespace().skip(1).take(7).map(|x| x.parse().unwrap()).collect()).collect();
    let carve: Vec<bool> = log[last..].lines().filter(|l| l.starts_with("DAB")).map(|l| l.ends_with("true")).collect();
    let mut mode = VolumeMode::new(vol);
    let patch = std::env::var("SCULPT_REPLAY_CENTER").ok().map(|c| {
        let v: Vec<f32> = c.split(',').map(|x| x.parse().unwrap()).collect();
        Vec3::new(v[0], v[1], v[2])
    }).unwrap_or(Vec3::new(0.30, 0.34, 0.89));
    let half: f32 = std::env::var("SCULPT_REPLAY_HALF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.12);
    // Level 0's samples in a lattice box around the view's centre, as text, per stage.
    let dump_box = |vol: &LevelVolume, name: &str| {
        let Ok(r) = std::env::var("SCULPT_REPLAY_BOX") else { return };
        let r: i32 = r.parse().unwrap();
        let c = ((patch - vol.origin()) / vol.voxel(0)).round().as_ivec3();
        let mut out = String::new();
        for z in c.z - r..=c.z + r {
            for y in c.y - r..=c.y + r {
                for x in c.x - r..=c.x + r {
                    let i = glam::IVec3::new(x, y, z);
                    out.push_str(&format!("{x} {y} {z} {} {}\n", vol.sample(0, i, false), vol.level_of(vol.origin() + i.as_vec3() * vol.voxel(0))));
                }
            }
        }
        std::fs::write(dir.join(format!("box_{name}.txt")), out).unwrap();
    };
    height_map(&mode.vol, patch, half, 700, &dir.join("map_0.pgm"));
    dump_box(&mode.vol, "0");
    mode.begin_stroke();
    for (k, d) in dabs.iter().enumerate() {
        mode.dab(Vec3::new(d[0], d[1], d[2]), d[3], d[4], d[5], d[6], carve[k]);
        mode.tick();
        height_map(&mode.vol, patch, half, 700, &dir.join(format!("map_{}.pgm", k + 1)));
        dump_box(&mode.vol, &format!("{}", k + 1));
        eprintln!("after dab {}: level 0 slope near the surface, in quarters: {:?}", k + 1, slope(&mode.vol, patch, half));
    }
    eprintln!("{:.1} ms per dab", mode.last_ms / mode.last_dabs.max(1) as f64);
    mode.end_stroke();
    mode.settle();
    eprintln!("after the end: level 0 slope near the surface, in quarters: {:?}", slope(&mode.vol, patch, half));
    height_map(&mode.vol, patch, half, 700, &dir.join("map_end.pgm"));
    dump_box(&mode.vol, "end");
}

/// The strokes of a dab log (`SCULPT_REPLAY_DABS`, from `SCULPT_DAB_LOG`) on the app's fresh ball; during the
/// strokes from `SCULPT_REPLAY_FROM` on (0), every `SCULPT_REPLAY_EVERY` dabs (8) the surface's bent points around
/// the stroke's first dab and its small pieces, and height maps in `SCULPT_REPLAY_OUT` when set.
#[test]
#[ignore]
fn replay_a_dab_log() {
    let Ok(path) = std::env::var("SCULPT_REPLAY_DABS") else { return };
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(d);
    let (from, every, half) = (env("SCULPT_REPLAY_FROM", 0.0) as usize, env("SCULPT_REPLAY_EVERY", 8.0) as usize, env("SCULPT_REPLAY_HALF", 0.06));
    let out = std::env::var("SCULPT_REPLAY_OUT").ok().map(std::path::PathBuf::from);
    let log = std::fs::read_to_string(path).unwrap();
    let mut mode = VolumeMode::ball(1e6);
    let strokes: Vec<&str> = log.split("BEGIN").filter(|s| s.contains("DAB")).collect();
    for (s, text) in strokes.iter().enumerate() {
        let dabs: Vec<(Vec<f32>, bool)> = text.lines().filter(|l| l.starts_with("DAB")).map(|l| (l.split_whitespace().skip(1).take(7).map(|x| x.parse().unwrap()).collect(), l.ends_with("true"))).collect();
        let c = Vec3::new(dabs[0].0[0], dabs[0].0[1], dabs[0].0[2]);
        let by = 0.2 * mode.vol.voxel(mode.vol.levels() - 1);
        let look = |mode: &VolumeMode, name: &str| {
            let (b, cnt) = bent(&mode.vol, c, half, 400, by);
            let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
            let small = pieces(&mesh).iter().filter(|&&p| p < sculpt_core::generator::level_volume::SPECK_VERTS).count();
            eprintln!("stroke {s} {name}: {b} of {cnt} points bent, {small} small pieces");
            if let Some(dir) = &out {
                height_map(&mode.vol, c, half, 700, &dir.join(format!("s{s}_{name}.pgm")));
            }
        };
        if s >= from {
            look(&mode, "before");
        }
        mode.begin_stroke();
        for (k, (d, carve)) in dabs.iter().enumerate() {
            mode.dab(Vec3::new(d[0], d[1], d[2]), d[3], d[4], d[5], d[6], *carve);
            mode.tick();
            if s >= from && k % every == every - 1 {
                look(&mode, &format!("dab{k:03}"));
            }
        }
        // `SCULPT_REPLAY_PROF=u0,v0,u1,v1`: heights at every cap along that line (offsets as the height maps').
        let profile = |mode: &VolumeMode, tag: &str| {
            let prof: Vec<f32> = std::env::var("SCULPT_REPLAY_PROF").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
            if prof.len() == 4 {
                let n = c.normalize();
                let u = n.cross(Vec3::Y).normalize();
                let v = n.cross(u);
                for i in 0..=60 {
                    let f = i as f32 / 60.0;
                    let q = c + u * (prof[0] + f * (prof[2] - prof[0])) + v * (prof[1] + f * (prof[3] - prof[1]));
                    let hs: Vec<String> = (0..mode.vol.levels()).map(|cap| mode.vol.raycast(q + n * 0.3, -n, cap, false).map_or("-".into(), |(hit, _)| format!("{:7.3}", (hit - c).dot(n) / mode.vol.voxel(4)))).collect();
                    let hit = mode.vol.raycast(q + n * 0.3, -n, mode.vol.levels() - 1, false).map(|(h, _)| h).unwrap_or(q);
                    println!("PROF {tag} {f:.3} level {} heights in level 4 steps by cap {}", mode.vol.level_of(hit), hs.join(" "));
                }
            }
        };
        if s >= from {
            profile(&mode, "last-dab");
        }
        mode.end_stroke();
        mode.settle();
        if s >= from {
            eprintln!("stroke {s}: {} dabs at level {} r {:.4} carve {}", dabs.len(), mode.level, dabs[0].0[3], dabs[0].1);
            look(&mode, "end");
            profile(&mode, "end");
        }
    }
}

/// A saved file's surface under a logged stroke (`SCULPT_DAB_LOG` lines), before and after replaying it: heights
/// along the first dab's normal over a grid around the stroke, and the level shown at each, written as raw f32
/// (`SCULPT_SURVEY_FILE`, `SCULPT_SURVEY_DABS`, `SCULPT_SURVEY_OUT`). For finding marks a stroke leaves.
#[test]
#[ignore]
fn survey_a_logged_stroke() {
    let (Ok(file), Ok(dabs), Ok(out)) = (std::env::var("SCULPT_SURVEY_FILE"), std::env::var("SCULPT_SURVEY_DABS"), std::env::var("SCULPT_SURVEY_OUT")) else {
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut mode = VolumeMode::read_from(&mut std::fs::File::open(&file).unwrap()).unwrap();
    let lines: Vec<Vec<f32>> = std::fs::read_to_string(&dabs)
        .unwrap()
        .lines()
        .filter_map(|l| l.strip_prefix("DAB "))
        .map(|l| l.split_whitespace().map(|w| match w { "true" => 1.0, "false" => 0.0, w => w.parse().unwrap_or(0.0) }).collect())
        .collect();
    let first = Vec3::new(lines[0][0], lines[0][1], lines[0][2]);
    let last = Vec3::new(lines[lines.len() - 1][0], lines[lines.len() - 1][1], lines[lines.len() - 1][2]);
    let top = mode.vol.levels() - 1;
    let chord = (first + last) * 0.5;
    let out_dir = chord.normalize();
    let mid = mode.vol.raycast(chord + out_dir * 0.3, -out_dir, top, false).unwrap().0;
    let n = mode.vol.normal_upto(mid, top, false);
    let ax = (last - first - n * (last - first).dot(n)).normalize();
    let ay = n.cross(ax);
    let half_x = (last - first).length() * 0.5 + 2.0 * lines[0][3];
    let half_y = 2.5 * lines[0][3];
    let step: f32 = std::env::var("SCULPT_SURVEY_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0005);
    let (w, h) = ((2.0 * half_x / step) as usize, (2.0 * half_y / step) as usize);
    let survey = |mode: &VolumeMode, name: &str| {
        let top = mode.vol.levels() - 1;
        let mut hs = Vec::with_capacity(w * h);
        let mut ls = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                let p = mid + ax * (-half_x + i as f32 * step) + ay * (-half_y + j as f32 * step);
                let o = p + n * 0.03;
                match mode.vol.raycast(o, -n, top, false) {
                    Some((q, _)) => {
                        hs.push((q - mid).dot(n));
                        ls.push(mode.vol.level_of(q - n * 1e-5) as f32);
                    }
                    None => {
                        hs.push(f32::NAN);
                        ls.push(-1.0);
                    }
                }
            }
        }
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(out.join(format!("{name}_h.f32")), bytes(&hs)).unwrap();
        std::fs::write(out.join(format!("{name}_l.f32")), bytes(&ls)).unwrap();
    };
    std::fs::write(out.join("size.txt"), format!("{w} {h} {step}")).unwrap();
    let v3 = |v: Vec3| format!("{} {} {}", v.x, v.y, v.z);
    std::fs::write(out.join("frame.txt"), format!("{}
{}
{}
{}
{} {}
", v3(mid), v3(ax), v3(ay), v3(n), half_x, half_y)).unwrap();
    if std::env::var_os("SCULPT_SURVEY_FRAME_ONLY").is_some() {
        return;
    }
    survey(&mode, "before");
    mode.begin_stroke();
    for d in &lines {
        mode.dab(Vec3::new(d[0], d[1], d[2]), d[3], d[4], d[5], d[6], d[7] > 0.5);
        mode.tick();
    }
    if std::env::var_os("SCULPT_SURVEY_MID").is_some() {
        mode.settle();
        survey(&mode, "mid");
    }
    mode.end_stroke();
    mode.settle();
    survey(&mode, "after");
    mode.write_to(&mut std::fs::File::create(out.join("after.svol")).unwrap()).unwrap();
    eprintln!("surveyed {w} x {h} at {step}; stroke at level {}", mode.level);
}

/// Heights of each level's own field (its samples, the coarser levels' where it holds none) along a row of a
/// survey's grid (`SCULPT_PROBE=j,i0,i1`, with `SCULPT_SURVEY_FILE` and the survey's `frame.txt` in
/// `SCULPT_SURVEY_OUT`), as the file holds them: which level a mark is in.
#[test]
#[ignore]
fn probe_levels_along_a_row() {
    let (Ok(file), Ok(out), Ok(probe)) = (std::env::var("SCULPT_SURVEY_FILE"), std::env::var("SCULPT_SURVEY_OUT"), std::env::var("SCULPT_PROBE")) else {
        return;
    };
    let mode = VolumeMode::read_from(&mut std::fs::File::open(&file).unwrap()).unwrap();
    let f: Vec<f32> = std::fs::read_to_string(std::path::Path::new(&out).join("frame.txt")).unwrap().split_whitespace().map(|w| w.parse().unwrap()).collect();
    let (mid, ax, ay, n) = (Vec3::new(f[0], f[1], f[2]), Vec3::new(f[3], f[4], f[5]), Vec3::new(f[6], f[7], f[8]), Vec3::new(f[9], f[10], f[11]));
    let (half_x, half_y) = (f[12], f[13]);
    let step: f32 = std::env::var("SCULPT_SURVEY_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0005);
    let pr: Vec<usize> = probe.split(',').filter_map(|w| w.parse().ok()).collect();
    let vol = &mode.vol;
    let field = |l: usize, p: Vec3| -> f32 {
        let g = (p - vol.origin()) / vol.voxel(l);
        let i = g.floor().as_ivec3();
        let t = g - i.as_vec3();
        let mut v = 0.0;
        for k in 0..8 {
            let o = glam::IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32);
            let wgt = (if o.x == 1 { t.x } else { 1.0 - t.x }) * (if o.y == 1 { t.y } else { 1.0 - t.y }) * (if o.z == 1 { t.z } else { 1.0 - t.z });
            v += wgt * vol.sample(l, i + o, false);
        }
        v
    };
    let height = |l: usize, p: Vec3| -> f32 {
        let mut u = 0.0f32;
        for _ in 0..30 {
            let v = field(l, p + n * u);
            u -= v;
            if v.abs() < 1e-7 {
                break;
            }
        }
        u
    };
    let top = vol.levels() - 1;
    use rayon::prelude::*;
    // `SCULPT_PROBE=map`: every level's own height over the survey's whole grid, as raw f32 (`own_<l>.f32`).
    if probe == "map" {
        let (w, h) = ((2.0 * half_x / step) as usize, (2.0 * half_y / step) as usize);
        for l in 0..=4.min(top) {
            let hs: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|k| {
                    let (i, j) = (k % w, k / w);
                    height(l, mid + ax * (-half_x + i as f32 * step) + ay * (-half_y + j as f32 * step))
                })
                .collect();
            std::fs::write(std::path::Path::new(&out).join(format!("own_{l}.f32")), hs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        }
        return;
    }
    // `SCULPT_PROBE=ray,j,i`: along the survey's ray at that grid point, near the surface: the level shown and its
    // value, and each level's own value.
    // `SCULPT_PROBE=lat,l,x0,x1,y,z0,z1`: level `l`'s samples over that box (x across, a row per y z).
    if probe.starts_with("lat") {
        let q: Vec<i32> = probe.split(',').skip(1).map(|w| w.parse().unwrap()).collect();
        for z in q[4]..=q[5] {
            let row: Vec<String> = (q[1]..=q[2]).map(|x| format!("{:6.0}", vol.sample(q[0] as usize, glam::IVec3::new(x, q[3], z), false) * 1e6)).collect();
            eprintln!("z {z:5} | {}", row.join(" "));
        }
        return;
    }
    if probe.starts_with("ray") {
        let p = mid + ax * (-half_x + pr[1] as f32 * step) + ay * (-half_y + pr[0] as f32 * step);
        let (q, _) = vol.raycast(p + n * 0.03, -n, top, false).unwrap();
        let u0 = (q - p).dot(n);
        for di in -8i32..=8 {
            let x = mid + ax * (-half_x + (pr[1] as i32 + di) as f32 * step) + ay * (-half_y + pr[0] as f32 * step);
            let Some((q, _)) = vol.raycast(x + n * 0.03, -n, top, false) else { continue };
            let keys: Vec<String> = (2..=4.min(top))
                .map(|l| {
                    let g = ((q - vol.origin()) / vol.voxel(l)).floor().as_ivec3();
                    let bc = g.div_euclid(glam::IVec3::splat(8));
                    format!("L{l} {:?}{} cell {:?}", bc.to_array(), if vol.exists(l, bc) { "*" } else { " " }, (g - bc * 8).to_array())
                })
                .collect();
            eprintln!("col {:4} h {:9.1} | {}", pr[1] as i32 + di, (q - p).dot(n) * 1e6, keys.join(" | "));
        }
        for k in -12..=12 {
            let u = u0 + k as f32 * 0.0001;
            let x = p + n * u;
            let own: Vec<String> = (2..=4.min(top)).map(|l| format!("{:9.1}", field(l, x) * 1e6)).collect();
            eprintln!("u {:9.1} | shown {} {:9.1} | own 2.. {}", u * 1e6, vol.level_of(x), vol.value(x) * 1e6, own.join(" "));
        }
        return;
    }
    for i in (pr[1]..pr[2]).step_by(pr.get(3).copied().unwrap_or(1)) {
        let p = mid + ax * (-half_x + i as f32 * step) + ay * (-half_y + pr[0] as f32 * step);
        let hs: Vec<String> = (0..=4.min(top)).map(|l| format!("{:9.2}", height(l, p) * 1e6)).collect();
        eprintln!("{i:4} shown {} | {}", vol.level_of(p), hs.join(" "));
    }
}

/// Close-up clay rows over one another, then a trim across them, on a new ball, as up close in the app (`SCULPT_ROWS_OUT`
/// names the `.svol` written): a scene to look at for marks (`SCULPT_DEMO_SCRIPT=look`, `SCULPT_LOOK_AT=0.3,-0.05,0.95`).
#[test]
#[ignore]
fn clay_rows_then_trim_file() {
    let Ok(out) = std::env::var("SCULPT_ROWS_OUT") else { return };
    let mut mode = VolumeMode::ball(1e6);
    let centre = Vec3::new(0.3, -0.05, 0.95).normalize();
    let (t1, t2) = (Vec3::Y.cross(centre).normalize(), centre.cross(Vec3::Y.cross(centre).normalize()));
    let line = |mode: &mut VolumeMode, a: (f32, f32), b: (f32, f32), r: f32, px: f32, trim: bool| {
        let (pa, pb) = (centre + t1 * a.0 + t2 * a.1, centre + t1 * b.0 + t2 * b.1);
        let steps = ((pa - pb).length() / (0.15 * r)).ceil() as usize;
        mode.begin_stroke();
        for k in 0..=steps {
            let d = pa.lerp(pb, k as f32 / steps as f32).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.dab(hit, r, px, 0.8, 0.3, trim);
            }
            mode.tick();
        }
        mode.end_stroke();
        mode.settle();
    };
    for row in 0..5 {
        for pass in 0..2 {
            let y = -0.03 + 0.014 * row as f32 + 0.003 * pass as f32;
            line(&mut mode, (-0.06, y), (0.06, y + 0.004), 0.016, 0.0004, false);
        }
    }
    line(&mut mode, (-0.07, 0.0), (0.07, 0.01), 0.038, 0.000476, true);
    line(&mut mode, (-0.03, -0.05), (0.0, 0.06), 0.03, 0.000476, true);
    eprintln!("blocks per level {:?}", mode.vol.memory().0);
    mode.write_to(&mut std::fs::File::create(&out).unwrap()).unwrap();
}

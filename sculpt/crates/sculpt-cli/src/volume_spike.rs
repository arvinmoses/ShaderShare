//! `sculpt-cli volume-spike [triangles, millions: 1,5,20] [--images DIR]`: can a volume brush keep up?
//!
//! For each detail level, a ball of radius 1 is stored as a sparse field at the lattice spacing that gives about that
//! many triangles (two per quad), extracted once, then sculpted with strokes of clay and carving dabs at three brush
//! sizes. Each dab is timed: changing the field, and re-extracting the blocks it touched.

use std::path::Path;

use sculpt_core::Document;
use sculpt_core::generator::volume::{DabStats, Volume};
use sculpt_core::glam::Vec3;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn longest_edge(mesh: &sculpt_core::mesh::PolyMesh) -> f32 {
    mesh.faces.iter().flat_map(|f| (0..4).map(move |k| (f[k], f[(k + 1) % 4]))).map(|(a, b)| mesh.positions[a as usize].distance(mesh.positions[b as usize])).fold(0.0, f32::max)
}

/// A stroke of `count` dabs across the top of the ball, each a blob sitting on the surface.
fn stroke(vol: &mut Volume, radius: f32, count: usize, carve: bool, phase: f32) -> Vec<DabStats> {
    (0..count)
        .map(|i| {
            let a = phase + i as f32 * radius * 0.25;
            let dir = Vec3::new(a.sin() * 0.6, 1.0, a.cos() * 0.6).normalize();
            // The blob's centre sits under the surface so it raises (or cuts) a lump of about half its radius.
            let centre = dir * (1.0 - 0.5 * radius);
            // A blend of a fifth of the radius: soft, and it keeps the band of changed samples thin.
            vol.dab(centre, radius, 0.2 * radius, carve)
        })
        .collect()
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let millions: Vec<f32> = args.first().filter(|a| !a.starts_with("--")).map_or(vec![1.0, 5.0, 20.0], |s| s.split(',').filter_map(|v| v.parse().ok()).collect());
    let images = args.iter().position(|a| a == "--images").and_then(|i| args.get(i + 1)).map(|s| s.to_string());
    println!("| triangles | voxel | build ms | blocks stored | memory 32-bit / 16-bit | full extract ms | quads |");
    println!("|---|---|---|---|---|---|---|");
    let mut dab_rows = Vec::new();
    for m in millions {
        // Surface nets gives about area / voxel^2 quads; two triangles each.
        let quads = m * 1e6 / 2.0;
        let voxel = (4.0 * std::f32::consts::PI / quads).sqrt();
        let t = std::time::Instant::now();
        let mut vol = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.4, voxel);
        let build = t.elapsed().as_secs_f64() * 1e3;
        let (extract, _verts, q) = vol.extract_all();
        let (m32, m16) = vol.memory();
        println!("| {m}M | {voxel:.5} | {build:.0} | {} | {:.0} / {:.0} MB | {extract:.0} | {q} |", vol.dense_blocks(), m32 as f64 / 1e6, m16 as f64 / 1e6);
        for (label, radius) in [("small", 0.03f32), ("medium", 0.1), ("large", 0.25)] {
            for carve in [false, true] {
                let s = stroke(&mut vol, radius, 24, carve, if carve { 1.5 } else { 0.0 });
                let field: Vec<f64> = s.iter().map(|d| d.field_ms).collect();
                let extract: Vec<f64> = s.iter().map(|d| d.extract_ms).collect();
                let total: Vec<f64> = s.iter().map(|d| d.field_ms + d.extract_ms).collect();
                let blocks = s.iter().map(|d| d.blocks_rebuilt).max().unwrap_or(0);
                let upload = s.iter().map(|d| d.upload_bytes).max().unwrap_or(0);
                dab_rows.push(format!(
                    "| {m}M | {label} ({:.0} voxels) | {} | {:.1} | {:.1} | {:.1} | {:.1} | {blocks} | {:.0} KB |",
                    radius / voxel,
                    if carve { "carve" } else { "clay" },
                    median(field),
                    median(extract),
                    median(total.clone()),
                    total.iter().cloned().fold(0.0, f64::max),
                    upload as f64 / 1024.0
                ));
            }
        }
    }
    println!();
    println!("| triangles | brush (radius) | dab | field ms | extract ms | total ms (median) | total ms (worst) | blocks rebuilt (most) | upload (most) |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in dab_rows {
        println!("{r}");
    }

    if let Some(dir) = images {
        demo(Path::new(&dir))?;
    }
    Ok(())
}

/// Two balls pulled into each other and a horn pulled far out of one, at about 1M triangles.
fn demo(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    let voxel = (4.0 * std::f32::consts::PI / 5e5).sqrt() * 0.8;
    let a = Vec3::new(-0.62, 0.0, 0.0);
    let b = Vec3::new(0.62, 0.0, 0.0);
    let mut vol = Volume::from_field(&|p: Vec3| ((p - a).length() - 0.5).min((p - b).length() - 0.5), 1.6, voxel);
    vol.extract_all();
    let views = [sculpt_core_render_view(0.0, 0.2), sculpt_core_render_view(0.9, 0.5)];
    crate::render::render(&Document::from_mesh(vol.mesh())?, &views, 420, &dir.join("volume_spike_before.png"))?;
    // A bridge of clay between the balls: the two surfaces meet and merge.
    for i in 0..=30 {
        let x = -0.15 + 0.3 * i as f32 / 30.0;
        vol.dab(Vec3::new(x, 0.0, 0.0), 0.16, 0.08, false);
    }
    // A horn pulled far out of the right ball, dab after dab: a mesh would stretch, a volume just grows.
    for i in 0..=60 {
        let t = i as f32 / 60.0;
        vol.dab(b + Vec3::new(0.3 + 0.6 * t, 0.45 + 0.3 * t, 0.0), 0.1 * (1.0 - 0.5 * t), 0.04, false);
    }
    // And a carved groove across the left ball.
    for i in 0..=30 {
        let t = i as f32 / 30.0;
        vol.dab(a + Vec3::new(-0.2, 0.55, -0.3 + 0.6 * t), 0.08, 0.03, true);
    }
    let mesh = vol.mesh();
    println!("demo: {} quads, longest edge {:.2} lattice steps, non-manifold edges {}", mesh.faces.len(), longest_edge(&mesh) / voxel, sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces));
    crate::render::render(&Document::from_mesh(mesh)?, &views, 420, &dir.join("volume_spike_after.png"))?;
    Ok(())
}

fn sculpt_core_render_view(yaw: f32, pitch: f32) -> crate::render::View {
    crate::render::View { yaw, pitch, overlay: None }
}

/// `sculpt-cli volume-spike levels [millions: 5,20] [--images DIR]`: large brushes on a coarse level.
pub(crate) fn levels(args: &[String]) -> Res<()> {
    use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
    let millions: Vec<f32> = args.first().filter(|a| !a.starts_with("--")).map_or(vec![5.0, 20.0], |s| s.split(',').filter_map(|v| v.parse().ok()).collect());
    let images = args.iter().position(|a| a == "--images").and_then(|i| args.get(i + 1)).map(|s| s.to_string());
    println!("| triangles | surface | coarse dab ms (median / worst) | refine after the stroke ms (field + extract) | direct fine dab ms (median / worst) | shape vs direct, voxels (mean / p95) | detail kept |");
    println!("|---|---|---|---|---|---|---|");
    let mut runs: Vec<(f32, bool)> = millions.iter().flat_map(|&m| [(m, false), (m, true)]).collect();
    if images.is_some() {
        runs.push((1.0, true));
    }
    for (m, bumpy) in runs {
        let voxel = (4.0 * std::f32::consts::PI / (m * 1e6 / 2.0)).sqrt();
        let coarse_voxel = 4.0 * voxel;
        // Fine bumps a dozen lattice steps across and a step and a half high: detail the coarse level cannot hold.
        let noise = Noise::new(&NoiseParams { kind: NoiseKind::Perlin, scale: 1.0 / (12.0 * voxel), octaves: 1, seed: 3, ..Default::default() });
        let amp = if bumpy { 1.5 * voxel } else { 0.0 };
        let fine_field = |p: Vec3| p.length() - 1.0 + amp * (noise.sample(p) - 0.5) * 2.0;
        let mut fine = Volume::from_field(&fine_field, 1.4, voxel);
        fine.extract_all();
        // The coarse level keeps true distances out past the largest move a stroke makes (the lump's height).
        let mut coarse = Volume::from_field_with_band(&|p: Vec3| p.length() - 1.0, 1.4, coarse_voxel, 0.2);
        coarse.extract_all();
        let before_mesh = (m <= 1.0).then(|| fine.mesh());
        let old_coarse = coarse.field_copy();
        // A large stroke, the same 24 dabs as the main benchmark's large brush.
        let (radius, k) = (0.25f32, 0.05f32);
        let centres: Vec<Vec3> = (0..24)
            .map(|i| {
                let a = i as f32 * radius * 0.25;
                Vec3::new(a.sin() * 0.6, 1.0, a.cos() * 0.6).normalize() * (1.0 - 0.5 * radius)
            })
            .collect();
        let coarse_times: Vec<f64> = centres.iter().map(|&c| { let d = coarse.dab(c, radius, k, false); d.field_ms + d.extract_ms }).collect();
        let coarse_mesh = (m <= 1.0).then(|| coarse.mesh());
        let reach = Vec3::splat(radius + k + 4.0 * coarse_voxel);
        let region = centres.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), c| (lo.min(*c - reach), hi.max(*c + reach)));
        let refine = fine.refine_from(&old_coarse, &coarse, region);
        // The same stroke straight on a fresh fine field.
        let mut direct = Volume::from_field(&fine_field, 1.4, voxel);
        direct.extract_all();
        let direct_times: Vec<f64> = centres.iter().map(|&c| { let d = direct.dab(c, radius, k, false); d.field_ms + d.extract_ms }).collect();
        let worst = |v: &[f64]| v.iter().cloned().fold(0.0, f64::max);
        // Shape: how far the direct result's surface is from the coarse-then-refined one, inside the stroke.
        let inside = |p: &Vec3| p.cmpge(region.0).all() && p.cmple(region.1).all();
        let direct_mesh = direct.mesh();
        // True distance from each direct-result vertex to the refined surface (not the clamped field).
        let refined_doc = Document::from_mesh(fine.mesh())?;
        let refined_sdf = sculpt_core::generator::voxel::MeshSdf::build(refined_doc.bvh(), refined_doc.positions(), refined_doc.faces(), refined_doc.topology());
        let mut gaps: Vec<f32> = direct_mesh.positions.iter().filter(|p| inside(p)).step_by(7).map(|&p| refined_sdf.signed_distance(p, 0.5).abs() / voxel).collect();
        gaps.sort_by(f32::total_cmp);
        let mean = gaps.iter().sum::<f32>() / gaps.len().max(1) as f32;
        let p95 = gaps.get((gaps.len() as f32 * 0.95) as usize).copied().unwrap_or(0.0);
        // Against the exact answer: the same dabs composed with true distances everywhere (no clamping), extracted at
        // fine detail. Which of the two paths is right?
        if !bumpy {
            let smin = |a: f32, b: f32, k: f32| {
                let h = (k - (a - b).abs()).max(0.0) / k;
                a.min(b) - h * h * k * 0.25
            };
            let exact_field = |p: Vec3| centres.iter().fold(p.length() - 1.0, |f, c| smin(f, (p - *c).length() - radius, k));
            let mut exact = Volume::from_field(&exact_field, 1.4, voxel);
            exact.extract_all();
            let exact_doc = Document::from_mesh(exact.mesh())?;
            let exact_sdf = sculpt_core::generator::voxel::MeshSdf::build(exact_doc.bvh(), exact_doc.positions(), exact_doc.faces(), exact_doc.topology());
            let off = |mesh: &sculpt_core::mesh::PolyMesh| {
                let mut g: Vec<f32> = mesh.positions.iter().filter(|p| inside(p)).step_by(7).map(|&p| exact_sdf.signed_distance(p, 0.5).abs() / voxel).collect();
                g.sort_by(f32::total_cmp);
                (g.iter().sum::<f32>() / g.len().max(1) as f32, g.get((g.len() as f32 * 0.95) as usize).copied().unwrap_or(0.0))
            };
            let (cm, cp) = off(&fine.mesh());
            let (dm, dp) = off(&direct_mesh);
            println!("  {m}M against the exact result, voxels (mean / p95): coarse then refined {cm:.2} / {cp:.2}; direct on the fine field {dm:.2} / {dp:.2}");
        }
        // Detail: how far the surface sits off the smooth coarse surface, on top of the new lump, against the same on the
        // untouched rest of the ball. 100% means the bumps rode up onto the lump unchanged; clay put on directly buries
        // them, so a direct stroke should score low here.
        let refined_mesh = fine.mesh();
        let lump_top: Vec<Vec3> = coarse.mesh().positions.into_iter().filter(|p| inside(p) && p.length() > 1.0 + 0.3 * radius).collect();
        let untouched: Vec<Vec3> = coarse.mesh().positions.into_iter().filter(|p| !inside(p)).take(20_000).collect();
        let spread = |field: &Volume, at: &[Vec3]| {
            let d: Vec<f32> = at.iter().map(|&p| field.value_at(p)).collect();
            let mean = d.iter().sum::<f32>() / d.len().max(1) as f32;
            (d.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d.len().max(1) as f32).sqrt()
        };
        let kept = if bumpy {
            let before = spread(&fine, &untouched);
            format!("{:.0}% (direct: {:.0}%)", 100.0 * spread(&fine, &lump_top) / before, 100.0 * spread(&direct, &lump_top) / before)
        } else {
            "-".into()
        };
        println!(
            "| {m}M | {} | {:.1} / {:.1} | {:.0} + {:.0} | {:.1} / {:.1} | {mean:.2} / {p95:.2} | {kept} |",
            if bumpy { "fine bumps" } else { "smooth" },
            median(coarse_times.clone()),
            worst(&coarse_times),
            refine.field_ms,
            refine.extract_ms,
            median(direct_times.clone()),
            worst(&direct_times),
        );
        if let (Some(dir), Some(before), Some(during)) = (&images, before_mesh, coarse_mesh) {
            let dir = Path::new(dir);
            std::fs::create_dir_all(dir)?;
            let views = [sculpt_core_render_view(0.3, 0.6)];
            crate::render::render(&Document::from_mesh(before)?, &views, 420, &dir.join("levels_1_before.png"))?;
            crate::render::render(&Document::from_mesh(during)?, &views, 420, &dir.join("levels_2_coarse_during_stroke.png"))?;
            crate::render::render(&Document::from_mesh(refined_mesh)?, &views, 420, &dir.join("levels_3_refined_after.png"))?;
            crate::render::render(&Document::from_mesh(direct_mesh)?, &views, 420, &dir.join("levels_4_direct_fine.png"))?;
        }
    }
    Ok(())
}

/// `sculpt-cli volume-spike video DIR`: frames of a short volume sculpting session, one dab a frame, the camera turning.
pub(crate) fn video(args: &[String]) -> Res<()> {
    let dir = Path::new(args.first().map_or("volume_video", |s| s.as_str()));
    std::fs::create_dir_all(dir)?;
    let voxel = 0.006f32;
    let a = Vec3::new(-0.62, 0.0, 0.0);
    let b = Vec3::new(0.62, 0.0, 0.0);
    let mut vol = Volume::from_field(&|p: Vec3| ((p - a).length() - 0.5).min((p - b).length() - 0.5), 1.7, voxel);
    vol.extract_all();
    // The session: (centre, radius, blend, carve), one dab a frame; `None` holds still for a frame.
    let mut dabs: Vec<Option<(Vec3, f32, f32, bool)>> = vec![None; 20];
    // Clay laid across the gap until the balls join.
    for i in 0..60 {
        let t = i as f32 / 59.0;
        dabs.push(Some((Vec3::new(-0.2 + 0.4 * t, 0.05 * (t * 6.0).sin(), 0.0), 0.14, 0.06, false)));
    }
    // A horn pulled up out of the right ball, tapering.
    for i in 0..80 {
        let t = i as f32 / 79.0;
        dabs.push(Some((b + Vec3::new(0.25 + 0.45 * t, 0.42 + 0.35 * t, 0.1 * t), 0.11 * (1.0 - 0.6 * t), 0.04, false)));
    }
    // A groove carved round the left ball.
    for i in 0..60 {
        let t = i as f32 / 59.0 * std::f32::consts::PI;
        let dir = Vec3::new(-0.9 * t.cos(), 0.9 * t.sin(), 0.6).normalize();
        dabs.push(Some((a + dir * 0.5, 0.075, 0.02, true)));
    }
    // Clay blobs dabbed onto the bridge.
    for i in 0..30 {
        let t = i as f32 / 29.0;
        dabs.push(Some((Vec3::new(-0.15 + 0.3 * t, 0.16, 0.08 * (t * 9.0).cos()), 0.07, 0.03, false)));
    }
    dabs.extend(std::iter::repeat_n(None, 30));
    // `--realtime`: frames for 60 fps playback at hand speed. A stroke of 60-80 dabs then takes half a second to two
    // thirds of one (two dabs a frame), and the pauses last as long as in the 30 fps version (two frames per hold).
    let realtime = args.iter().any(|a| a == "--realtime");
    let mut frames: Vec<Vec<(Vec3, f32, f32, bool)>> = Vec::new();
    for d in &dabs {
        match (d, realtime) {
            (None, true) => frames.extend([Vec::new(), Vec::new()]),
            (None, false) => frames.push(Vec::new()),
            (Some(d), true) if frames.last().is_some_and(|f| f.len() == 1) => frames.last_mut().unwrap().push(*d),
            (Some(d), _) => frames.push(vec![*d]),
        }
    }
    let mut times = Vec::new();
    let mut frame_ms: Vec<f64> = Vec::new();
    for (frame, batch) in frames.iter().enumerate() {
        let mut spent = 0.0;
        for &(c, r, k, carve) in batch {
            let s = vol.dab(c, r, k, carve);
            times.push(s.field_ms + s.extract_ms);
            spent += s.field_ms + s.extract_ms;
        }
        frame_ms.push(spent);
        // Sway across the front so every stroke stays in view, at the same speed in seconds either way.
        let seconds = frame as f32 / if realtime { 60.0 } else { 30.0 };
        let yaw = 0.15 + 0.7 * (seconds * 0.66).sin();
        let views = [sculpt_core_render_view(yaw, 0.35)];
        crate::render::render(&Document::from_mesh(vol.mesh())?, &views, 720, &dir.join(format!("frame_{frame:04}.png")))?;
    }
    let mesh = vol.mesh();
    println!("sculpting work per frame: worst {:.2} ms (a 60 fps frame is 16.7 ms)", frame_ms.iter().cloned().fold(0.0, f64::max));
    println!(
        "{} frames, {} dabs: median {:.2} ms, worst {:.2} ms; final {} quads, longest edge {:.2} lattice steps, non-manifold edges {}",
        frames.len(),
        times.len(),
        median(times.clone()),
        times.iter().cloned().fold(0.0, f64::max),
        mesh.faces.len(),
        longest_edge(&mesh) / voxel,
        sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces)
    );
    Ok(())
}

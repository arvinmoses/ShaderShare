//! `sculpt-cli volume-spike multilevel [--images DIR]`: a volume at four resolutions (spike 4, `generator::levels`).
//!
//! A ball is shaped with large dabs on level 0, given ridges on level 1, fine grooves on level 2 and tiny dots on level
//! 3, each at the brush size a zoomed-in view would give. Then a large dab on level 0 bulges the patch holding the fine
//! work, to see whether the detail rides on it. Reported: dab times per level, memory against one uniform lattice at
//! the finest step, the surface (cracks, size, time) at each level cap a distant view would use, and how much of the
//! fine detail survived the large edit.

use std::path::Path;
use std::time::Instant;

use sculpt_core::Document;
use sculpt_core::generator::levels::{LevelDab, Levels};
use sculpt_core::glam::Vec3;
use sculpt_core::mesh::{NO_VERT, PolyMesh};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Dabs along `path` (directions from the centre; each lands on the surface under it), `r` apart by `spacing`.
fn stroke(v: &mut Levels, level: usize, path: &[Vec3], r: f32, carve: bool, times: &mut Vec<LevelDab>) {
    for w in path.windows(2) {
        let steps = ((w[1] - w[0]).length() / (0.15 * r)).ceil().max(1.0) as usize;
        for k in 0..steps {
            let dir = w[0].lerp(w[1], k as f32 / steps as f32).normalize();
            // On the surface below this level, which the stroke barely changes: a dab placed on the one it just
            // carved would dig deeper and deeper.
            let Some((hit, n)) = v.raycast_upto(dir * 4.0, -dir, level.saturating_sub(1)) else { continue };
            // Half the ball above the surface for clay, half below for carving.
            let c = if carve { hit + n * 0.3 * r } else { hit - n * 0.3 * r };
            times.push(v.dab(level, c, r, 0.25 * r, carve));
        }
    }
}

fn arc(from: Vec3, to: Vec3, n: usize) -> Vec<Vec3> {
    (0..=n).map(|i| from.lerp(to, i as f32 / n as f32).normalize()).collect()
}

/// The faces with every corner within `r` of `c`, as their own mesh (for a close-up the renderer frames by itself).
fn around(mesh: &PolyMesh, c: Vec3, r: f32) -> PolyMesh {
    let keep: Vec<bool> = mesh.positions.iter().map(|p| (*p - c).length() < r).collect();
    let mut map = vec![u32::MAX; mesh.positions.len()];
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    for f in &mesh.faces {
        let n = if f[3] == NO_VERT { 3 } else { 4 };
        if !f[..n].iter().all(|&v| keep[v as usize]) {
            continue;
        }
        let mut g = [NO_VERT; 4];
        for k in 0..n {
            let v = f[k] as usize;
            if map[v] == u32::MAX {
                map[v] = positions.len() as u32;
                positions.push(mesh.positions[v]);
            }
            g[k] = map[v];
        }
        faces.push(g);
    }
    PolyMesh { positions, faces }
}

/// Heights of the surface along a line across the patch (across the grooves and the row of dots), seen along `-n`; and the part that is detail (the height
/// minus its running mean over `window` samples).
fn profile(v: &Levels, centre: Vec3, n: Vec3, across: Vec3, len: f32, count: usize, window: usize) -> Vec<f32> {
    let heights: Vec<f32> = (0..count)
        .map(|i| {
            let p = centre + across * (len * (i as f32 / (count - 1) as f32 - 0.5));
            v.raycast(p + n * 0.6, -n).map_or(f32::NAN, |(h, _)| (h - p).dot(n))
        })
        .collect();
    (0..count)
        .map(|i| {
            let lo = i.saturating_sub(window);
            let hi = (i + window + 1).min(count);
            let mean = heights[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            heights[i] - mean
        })
        .collect()
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let images = args.iter().position(|a| a == "--images").and_then(|i| args.get(i + 1)).map(|s| s.to_string());
    let levels = 4;
    let voxel0 = 0.02;
    let t = Instant::now();
    let mut v = Levels::from_field(&|p: Vec3| p.length() - 1.0, 1.7, voxel0, levels);
    // `--rule a`: coarser levels take each dab's change, filtered. Default (rule b): they do not, and capped views
    // fold the finer detail in when drawn.
    v.restrict = args.windows(2).any(|w| w[0] == "--rule" && w[1] == "a");
    println!("rule {}", if v.restrict { "a: coarser levels take every change, filtered" } else { "b: each level holds its own work; capped views fold finer detail in when drawn" });
    println!("ball at level 0 (step {voxel0}) in {:.0} ms; {levels} levels, the finest at step {}", t.elapsed().as_secs_f64() * 1e3, v.voxel(levels - 1));
    let mut times: Vec<Vec<LevelDab>> = vec![Vec::new(); levels];

    // Level 0: broad form, large brushes.
    for i in 0..6 {
        let a = i as f32 * 1.1;
        let d = Vec3::new(a.sin() * 0.8, 0.5 + 0.3 * (i as f32).cos(), a.cos() * 0.8 + 0.6);
        stroke(&mut v, 0, &arc(d, d + Vec3::new(0.3, -0.2, 0.0), 4), 0.35, i == 4, &mut times[0]);
    }
    // Level 1: ridges across the front.
    for i in 0..3 {
        let y = -0.25 + 0.2 * i as f32;
        stroke(&mut v, 1, &arc(Vec3::new(-0.6, y, 1.0), Vec3::new(0.2, y + 0.1, 1.0), 10), 0.06, false, &mut times[1]);
    }
    // Level 2: fine grooves in a patch; level 3: tiny dots inside it.
    let patch = Vec3::new(0.35, -0.35, 1.0).normalize();
    let (hit, n) = v.raycast(patch * 4.0, -patch).ok_or("missed the patch")?;
    let (t1, t2) = n.any_orthonormal_pair();
    for i in 0..4 {
        let off = t2 * (0.03 * (i as f32 - 1.5));
        let a = (hit + off - t1 * 0.09).normalize();
        let b = (hit + off + t1 * 0.09).normalize();
        stroke(&mut v, 2, &arc(a, b, 6), 0.012, true, &mut times[2]);
    }
    for i in 0..10 {
        let p = hit + t1 * (0.012 * (i as f32 - 4.5)) + t2 * 0.075;
        stroke(&mut v, 3, &[p.normalize(), p.normalize()], 0.005, false, &mut times[3]);
    }
    for (l, ts) in times.iter().enumerate() {
        let ms: Vec<f64> = ts.iter().map(|d| d.ms).collect();
        let created: usize = ts.iter().map(|d| d.created).sum();
        println!(
            "level {l} (step {:.4}, brush radius {:.3}): {} dabs, median {:.2} ms, worst {:.2} ms; {created} blocks made at this level",
            v.voxel(l),
            [0.35, 0.06, 0.012, 0.005][l],
            ts.len(),
            median(ms.clone()),
            ms.iter().cloned().fold(0.0, f64::max)
        );
    }
    let (counts, bytes) = v.memory();
    let uniform = counts[0] * (1usize << (2 * (levels - 1))) * 512 * 2;
    println!("blocks per level {counts:?}: {:.1} MB at 16 bits; one uniform lattice at the finest step would need about {:.0} MB (level 0's surface blocks x 4 per level)", bytes as f64 / 1e6, uniform as f64 / 1e6);

    // The surface at each level cap: what a view from far to near would draw.
    let mut meshes = Vec::new();
    for cap in 0..levels {
        let (mesh, s) = v.surface(cap);
        println!(
            "surface up to level {cap}: {} quads + {} triangles in {:.0} ms; open edges {}, non-manifold {}; faces per level {:?}",
            s.quads,
            s.triangles,
            s.ms,
            s.open_edges,
            s.non_manifold_edges,
            &s.per_level[..levels]
        );
        meshes.push(mesh);
    }

    // A large level 0 dab under the patch: does the fine work ride on it?
    let window = 12;
    let before = profile(&v, hit, n, t2, 0.2, 201, window);
    // How much finer-than-level-1 detail sits on the surface along the line: the field with all levels, minus the
    // field up to level 1, at the surface seen up to level 1.
    let fine_detail = |v: &Levels, centre: Vec3| -> (f32, usize) {
        let mut most = 0.0f32;
        let mut n_some = 0;
        for i in 0..201 {
            let p = centre + t2 * (0.2 * (i as f32 / 200.0 - 0.5));
            if let Some((s, _)) = v.raycast_upto(p + n * 0.6, -n, 1) {
                let d = v.value(s) - v.value_upto(s, 1);
                most = most.max(d.abs());
                n_some += (d.abs() > 1e-4) as usize;
            }
        }
        (most, n_some)
    };
    let fd_before = fine_detail(&v, hit);
    let mut big = Vec::new();
    let sink: f32 = std::env::var("ML_SINK").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
    let bulge = hit - n * sink;
    big.push(v.dab(0, bulge, 0.33, 0.08, false));
    let (hit2, _) = v.raycast(patch * 4.0, -patch).ok_or("missed the patch")?;
    let after = profile(&v, hit2, n, t2, 0.2, 201, window);
    println!("finer-than-level-1 detail on the line: before {fd_before:?}, after {:?} (largest, samples with any)", fine_detail(&v, hit2));
    let ok: Vec<(f32, f32)> = before.iter().zip(&after).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
    let (sa, sb, sab) = ok.iter().fold((0.0, 0.0, 0.0), |(sa, sb, sab), (a, b)| (sa + a * a, sb + b * b, sab + a * b));
    println!(
        "large level 0 dab over the fine patch: {:.2} ms, {} other blocks updated; the patch moved out {:.3}; fine detail after vs before: correlation {:.2}, size {:.0}%",
        big[0].ms,
        big[0].others,
        (hit2 - hit).dot(n),
        sab / (sa.sqrt() * sb.sqrt()),
        100.0 * (sb / sa).sqrt()
    );
    let (full_after, s) = v.surface(levels - 1);
    println!("surface after: {} quads + {} triangles, open edges {}, non-manifold {}", s.quads, s.triangles, s.open_edges, s.non_manifold_edges);

    if let Some(dir) = images {
        let dir = Path::new(&dir);
        std::fs::create_dir_all(dir)?;
        let view = |yaw: f32, pitch: f32| crate::render::View { yaw, pitch, overlay: None };
        for (cap, mesh) in meshes.iter().enumerate() {
            crate::render::render(&Document::from_mesh(mesh.clone())?, &[view(0.35, 0.3)], 520, &dir.join(format!("multilevel_whole_cap{cap}.png")))?;
        }
        let close = |m: &PolyMesh| around(m, hit, 0.14);
        for (cap, mesh) in meshes.iter().enumerate() {
            crate::render::render(&Document::from_mesh(close(mesh))?, &[view(-0.35, 0.35)], 520, &dir.join(format!("multilevel_patch_cap{cap}.png")))?;
        }
        crate::render::render(&Document::from_mesh(around(&full_after, hit2, 0.14))?, &[view(-0.35, 0.35)], 520, &dir.join("multilevel_patch_after_bulge.png"))?;
        crate::render::render(&Document::from_mesh(around(&full_after, hit2, 0.5))?, &[view(-0.35, 0.35)], 520, &dir.join("multilevel_bulge_wide.png"))?;
        println!("images in {}", dir.display());
    }
    Ok(())
}

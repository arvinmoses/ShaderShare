//! `sculpt-cli volume-spike levels-carry DIR [LEVEL] [--pits]`: pictures of fine detail under a coarse stroke.
//!
//! A row of small bumps (or pits) is sculpted at `LEVEL` of a multi-resolution ball, then broad clay at level 0 raises
//! the area under them. The surface around them is drawn before and after (`levels_carry_before.png`, `..._after.png`),
//! to see whether the detail rode along.

use std::path::Path;

use sculpt_core::Document;
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::glam::Vec3;
use sculpt_core::mesh::{NO_VERT, PolyMesh};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

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

/// The demo's case: level 0 at about 1M triangles' step, grooves trimmed at level 4 by a small brush, then a big level
/// 0 clay stroke, detail carried 6 ms at a time as in the app. Prints the blocks per level after each part.
fn app_case() -> Res<()> {
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6);
    v.carry_budget = Some(6.0);
    let plane_stroke = |v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool| {
        v.begin_stroke();
        let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
        let mut along = Vec3::ZERO;
        for k in 0..=steps {
            let d = a.lerp(b, k as f32 / steps as f32).normalize();
            let Some((hit, n)) = v.raycast(d * 3.0, -d, l, true) else { continue };
            let h = r * if trim { 0.075 } else { 0.095 };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, 0.2, 0.4, trim) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            v.carry_pending(Some(6.0));
        }
        v.end_stroke();
    };
    let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
    let (t1, t2) = patch.any_orthonormal_pair();
    for k in 0..4 {
        let o = t2 * (0.004 * k as f32 - 0.006);
        plane_stroke(&mut v, 4, patch + o - t1 * 0.01, patch + o + t1 * 0.01, 0.0025, true);
    }
    println!("after the grooves: blocks per level {:?}", v.memory().0);
    plane_stroke(&mut v, 0, patch - t1 * 0.05, patch + t1 * 0.05, 0.29, false);
    println!("after the big stroke: blocks per level {:?}", v.memory().0);
    Ok(())
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    if args.iter().any(|a| a == "--app") {
        return app_case();
    }
    let dir = Path::new(args.first().ok_or("levels-carry DIR [LEVEL] [--pits]")?);
    let fine: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    let carve = args.iter().any(|a| a == "--pits");
    std::fs::create_dir_all(dir)?;
    let mut v = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, fine + 1);
    for i in 0..5 {
        let x = -0.024 + 0.012 * i as f32;
        v.begin_stroke();
        let p = Vec3::new(x, 0.0, 1.0).normalize();
        v.ball_dab(fine, p * if carve { 1.0035 } else { 1.0 }, 0.004, 0.001, carve);
        v.end_stroke();
    }
    let draw = |v: &LevelVolume, centre: Vec3, name: &str| -> Res<()> {
        let (mesh, s) = v.surface(v.levels() - 1);
        let part = around(&mesh, centre, 0.06);
        println!("{name}: {} faces near the bumps (of {}), open edges in the whole: {}", part.faces.len(), mesh.faces.len(), s.dropped);
        let view = |yaw: f32, pitch: f32| crate::render::View { yaw, pitch, overlay: None };
        crate::render::render(&Document::from_mesh(part)?, &[view(0.0, 0.0), view(0.0, 1.2)], 520, &dir.join(format!("levels_carry_{name}.png")))
    };
    draw(&v, Vec3::new(0.0, 0.0, 1.0), "before")?;
    for _ in 0..2 {
        v.begin_stroke();
        let mut along = Vec3::ZERO;
        for k in 0..=8 {
            let d = Vec3::new(0.0, -0.1 + 0.2 * k as f32 / 8.0, 1.0).normalize();
            let Some((hit, n)) = v.raycast(d * 3.0, -d, 0, true) else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = v.plane_dab(0, hit, rough, 0.3, 0.3 * 0.095, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
        }
        v.end_stroke();
    }
    let top = v.raycast(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z, v.levels() - 1, false).map_or(1.0, |(h, _)| h.z);
    draw(&v, Vec3::new(0.0, 0.0, top), "after")?;
    println!("images in {}", dir.display());
    Ok(())
}

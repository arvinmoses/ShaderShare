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

//! Decimation: targets hit, manifold kept, silhouette error at evenness 0 vs 1, and speed.

use std::time::Instant;

use glam::Vec3;
use sculpt_core::decimate::{Analysis, AnalyzeOptions, Target, TriMesh, check_manifold, relax, surface_error};
use sculpt_core::primitives::{grid, quad_sphere_res};

/// A closed, lumpy ball with a sharp-ish ridge: something with a silhouette to keep.
fn lumpy(res: u32) -> TriMesh {
    let mut m = TriMesh::from_poly(&quad_sphere_res(res, 1.0));
    for p in &mut m.positions {
        let d = p.normalize();
        let bumps = 0.08 * (5.0 * d.x).sin() * (7.0 * d.y).sin() * (6.0 * d.z).sin();
        let ridge = 0.12 * (1.0 - (d.y * 6.0).abs().min(1.0));
        *p = d * (1.0 + bumps + ridge);
    }
    m
}

fn edge_len_cv(m: &TriMesh) -> f32 {
    let mut l = Vec::new();
    for t in &m.triangles {
        for i in 0..3 {
            l.push(m.positions[t[i] as usize].distance(m.positions[t[(i + 1) % 3] as usize]));
        }
    }
    let mean = l.iter().sum::<f32>() / l.len() as f32;
    let var = l.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / l.len() as f32;
    var.sqrt() / mean
}

fn analyse(m: &TriMesh, e: f32) -> Analysis {
    Analysis::build(m, &AnalyzeOptions { evenness: e, ..Default::default() }, None).unwrap()
}

#[test]
fn targets_hit_and_closed_manifold_kept() {
    let m = lumpy(120); // 172,800 triangles, closed
    let input = check_manifold(m.positions.len(), &m.triangles);
    assert!(input.closed() && input.manifold(), "{input:?}");
    for e in [0.0, 0.5, 1.0] {
        let a = analyse(&m, e);
        assert_eq!(a.input_triangles(), m.triangles.len());
        assert!(a.min_triangles() <= 400, "e {e}: floor {}", a.min_triangles());
        for target in [Target::KeepPercent(50.0), Target::KeepPercent(10.0), Target::Triangles(5000), Target::Triangles(800)] {
            let want = target.triangles(a.input_triangles());
            let d = a.extract(target);
            let got = d.mesh.triangles.len();
            assert!(got.abs_diff(want) <= 2, "e {e} {target:?}: {got} for {want}");
            let r = check_manifold(d.mesh.positions.len(), &d.mesh.triangles);
            assert!(r.closed() && r.manifold(), "e {e} {target:?}: {r:?}");
            // Every vertex output is used.
            let mut used = vec![false; d.mesh.positions.len()];
            d.mesh.triangles.iter().flatten().for_each(|&v| used[v as usize] = true);
            assert!(used.iter().all(|u| *u), "e {e} {target:?}: unused vertices");
        }
        // Step 0 is the input itself.
        let d0 = a.extract_step(0);
        assert_eq!(d0.mesh.triangles.len(), m.triangles.len());
    }
}

#[test]
fn borders_stay_put() {
    // An open, wavy sheet: its border must stay on the square's edge and stay a single loop.
    let mut m = TriMesh::from_poly(&grid(100, 2.0));
    for p in &mut m.positions {
        p.y = 0.1 * (3.0 * p.x).sin() * (4.0 * p.z).cos();
    }
    let input = check_manifold(m.positions.len(), &m.triangles);
    let a = analyse(&m, 0.0);
    let d = a.extract(Target::KeepPercent(5.0));
    let r = check_manifold(d.mesh.positions.len(), &d.mesh.triangles);
    assert!(r.manifold(), "{r:?}");
    assert!(r.border_edges > 0 && r.border_edges < input.border_edges, "{r:?}");
    // Border vertices are still on the square's outline.
    let mut count = std::collections::HashMap::new();
    for t in &d.mesh.triangles {
        for i in 0..3 {
            let (x, y) = (t[i], t[(i + 1) % 3]);
            *count.entry((x.min(y), x.max(y))).or_insert(0) += 1;
        }
    }
    for ((x, y), c) in count {
        if c == 1 {
            for v in [x, y] {
                let p = d.mesh.positions[v as usize];
                let on = (p.x.abs() - 1.0).abs() < 1e-4 || (p.z.abs() - 1.0).abs() < 1e-4;
                assert!(on, "border vertex moved off the outline: {p}");
            }
        }
    }
}

#[test]
fn frozen_borders_keep_every_border_vertex() {
    let mut m = TriMesh::from_poly(&grid(60, 2.0));
    for p in &mut m.positions {
        p.y = 0.1 * (3.0 * p.x).sin() * (4.0 * p.z).cos();
    }
    let input = check_manifold(m.positions.len(), &m.triangles);
    let a = Analysis::build(&m, &AnalyzeOptions { freeze_borders: true, ..Default::default() }, None).unwrap();
    let d = a.extract(Target::KeepPercent(5.0));
    let r = check_manifold(d.mesh.positions.len(), &d.mesh.triangles);
    assert!(r.manifold(), "{r:?}");
    assert_eq!(r.border_edges, input.border_edges);
}

#[test]
fn evenness_trades_silhouette_for_even_triangles() {
    let m = lumpy(120);
    let target = Target::Triangles(6000);
    let mut rows = Vec::new();
    for e in [0.0f32, 0.5, 1.0] {
        let a = analyse(&m, e);
        let d = a.extract(target);
        let err = surface_error(&m, &d.mesh, 50_000);
        let cv = edge_len_cv(&d.mesh);
        println!("evenness {e}: {} tris, mean err {:.5}% max err {:.4}% of diagonal, edge length CV {cv:.3}", d.mesh.triangles.len(), 100.0 * err.mean_rel(), 100.0 * err.max_rel());
        rows.push((err, cv));
    }
    let (e0, e1) = (&rows[0], &rows[2]);
    assert!(e0.0.mean < e1.0.mean, "evenness 0 should hold the shape better: {} vs {}", e0.0.mean, e1.0.mean);
    assert!(e1.1 < e0.1, "evenness 1 should give more even edges: {} vs {}", e1.1, e0.1);
    // The middle sits between the two ends on both counts.
    assert!(rows[1].1 < e0.1 && rows[1].0.mean < e1.0.mean, "{:?}", rows.iter().map(|r| (r.0.mean, r.1)).collect::<Vec<_>>());
    // Shape held: at 3.5% of the input the mean error is a small share of the size.
    assert!(e0.0.mean_rel() < 0.002, "{}", e0.0.mean_rel());
}

#[test]
fn relax_keeps_topology_and_evens() {
    let m = lumpy(80);
    let a = analyse(&m, 0.0);
    let mut d = a.extract(Target::Triangles(4000)).mesh;
    let cv0 = edge_len_cv(&d);
    let tris = d.triangles.clone();
    relax(&mut d, 1.0, 5);
    assert_eq!(tris, d.triangles);
    let cv1 = edge_len_cv(&d);
    assert!(cv1 < cv0, "{cv1} vs {cv0}");
    let r = check_manifold(d.positions.len(), &d.triangles);
    assert!(r.manifold() && r.closed());
}

/// Speed: analysis and live extraction. Ignored by default; run with
/// `cargo test --release -p sculpt-core --test decimate -- --ignored --nocapture bench`.
#[test]
#[ignore]
fn bench() {
    let res: u32 = std::env::var("DECIMATE_RES").ok().and_then(|s| s.parse().ok()).unwrap_or(580); // ~4M tris
    let m = lumpy(res);
    println!("input {:.2}M triangles", m.triangles.len() as f64 / 1e6);
    for e in [0.0f32, 1.0] {
        let t = Instant::now();
        let a = analyse(&m, e);
        println!("evenness {e}: analysis {:.0} ms ({} rounds, floor {})", t.elapsed().as_secs_f64() * 1e3, a.stats.rounds, a.min_triangles());
        for pct in [50.0f32, 10.0, 1.0] {
            let t = Instant::now();
            let d = a.extract(Target::KeepPercent(pct));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let r = check_manifold(d.mesh.positions.len(), &d.mesh.triangles);
            let mms = t.elapsed().as_secs_f64() * 1e3;
            println!("  {pct}%: {} tris extract {ms:.1} ms, manifold check {mms:.0} ms closed {} manifold {}", d.mesh.triangles.len(), r.closed(), r.manifold());
        }
    }
    let _ = Vec3::ZERO;
}

/// A real volume surface: `DECIMATE_SVOL=path cargo test --release -p sculpt-core --test decimate -- --ignored --nocapture svol`.
#[test]
#[ignore]
fn svol() {
    let Some(path) = std::env::var_os("DECIMATE_SVOL") else { return };
    let vol = sculpt_core::generator::level_volume::LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap()), 8).unwrap();
    let e = sculpt_core::mesh_layers::extract_all(&vol, usize::MAX).unwrap();
    let m = TriMesh::from_poly(&e.mesh);
    let r = check_manifold(m.positions.len(), &m.triangles);
    println!("input {} tris, {} verts: {r:?}", m.triangles.len(), m.positions.len());
    let t = Instant::now();
    let a = analyse(&m, 0.0);
    println!("analysis {:.0} ms {:?}", t.elapsed().as_secs_f64() * 1e3, a.stats);
    let d = a.extract(Target::KeepPercent(2.0));
    println!("2%: {:?}", check_manifold(d.mesh.positions.len(), &d.mesh.triangles));
}

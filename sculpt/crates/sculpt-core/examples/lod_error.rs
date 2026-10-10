//! Compare each LOD patch's stored error with its true deviation from the full-resolution surface,
//! on a sphere carrying clay-like plateaus with steep edges (the case that shows seams).
use std::collections::HashMap;

use glam::Vec3;
use sculpt_core::Document;
use sculpt_core::lod::LodParams;
use sculpt_core::primitives::quad_sphere_res;

fn point_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Ericson, closest point on triangle.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// Largest distance from `verts` to the triangle soup `tris` (brute force with a coarse grid).
fn deviation(pos: &[Vec3], tris: &[u32], verts: &[u32]) -> f32 {
    let cell = 0.02f32;
    let key = |p: Vec3| ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (t, tri) in tris.chunks_exact(3).enumerate() {
        let (a, b, c) = (pos[tri[0] as usize], pos[tri[1] as usize], pos[tri[2] as usize]);
        let lo = a.min(b).min(c);
        let hi = a.max(b).max(c);
        let (k0, k1) = (key(lo), key(hi));
        for x in k0.0..=k1.0 {
            for y in k0.1..=k1.1 {
                for z in k0.2..=k1.2 {
                    grid.entry((x, y, z)).or_default().push(t);
                }
            }
        }
    }
    let mut worst = 0.0f32;
    for &v in verts {
        let p = pos[v as usize];
        let k = key(p);
        let mut best = f32::MAX;
        for r in 0..6 {
            for x in k.0 - r..=k.0 + r {
                for y in k.1 - r..=k.1 + r {
                    for z in k.2 - r..=k.2 + r {
                        if (x - k.0).abs().max((y - k.1).abs()).max((z - k.2).abs()) != r {
                            continue;
                        }
                        if let Some(list) = grid.get(&(x, y, z)) {
                            for &t in list {
                                let tri = &tris[t * 3..t * 3 + 3];
                                best = best.min(point_tri(p, pos[tri[0] as usize], pos[tri[1] as usize], pos[tri[2] as usize]));
                            }
                        }
                    }
                }
            }
            if best <= r as f32 * cell {
                break;
            }
        }
        worst = worst.max(best);
    }
    worst
}

fn main() {
    let res: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let height: f32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let mut doc = Document::from_mesh(quad_sphere_res(res, 1.0)).unwrap();
    doc.build_lod(LodParams::default());
    // Clay-like plateaus along a V: flat top, steep rim.
    let centres: Vec<Vec3> = (0..24).map(|i| {
        let t = i as f32 / 23.0;
        let x = -0.5 + t;
        let y = 0.3 - (x.abs()) * 0.6;
        Vec3::new(x, y, 1.0).normalize()
    }).collect();
    for c in &centres {
        let r = 0.08;
        let d = doc.compute_displacements(*c, r, |doc, v, dist| {
            let n = doc.normals()[v];
            let s = ((r - dist) / (0.08 * r)).clamp(0.0, 1.0);
            let target = height * s;
            // Build up to the plateau: only move what is below it.
            let cur = (doc.positions()[v].length() - 1.0).max(0.0);
            (target > cur).then(|| n * (target - cur))
        });
        doc.begin_stroke("clay");
        doc.apply_displacements(&d).unwrap();
        doc.end_stroke();
    }
    let report = |doc: &Document, label: &str| {
        let tree = doc.lod().unwrap();
        let pos = doc.positions();
        let mut rows = Vec::new();
        for (id, n) in tree.nodes.iter().enumerate() {
            let Some(_) = n.children else { continue };
            let b = &doc.bvh().nodes[id].bounds;
            // Only patches near the edits.
            if !centres.iter().any(|c| b.min.cmple(*c + 0.1).all() && b.max.cmpge(*c - 0.1).all()) {
                continue;
            }
            // Full-resolution vertices under this node: the leaves below it.
            let mut stack = vec![id as u32];
            let mut verts = Vec::new();
            while let Some(i) = stack.pop() {
                let m = &tree.nodes[i as usize];
                match m.children {
                    Some([l, r]) => stack.extend([l, r]),
                    None => verts.extend_from_slice(&tree.indices[m.first as usize..(m.first + m.count) as usize]),
                }
            }
            verts.sort_unstable();
            verts.dedup();
            let patch = &tree.indices[n.first as usize..(n.first + n.count) as usize];
            let true_err = deviation(pos, patch, &verts);
            rows.push((id, n.error, true_err, verts.len()));
        }
        let worst = rows.iter().map(|r| r.2 / r.1.max(1e-7)).fold(0.0f32, f32::max);
        let under = rows.iter().filter(|r| r.2 > r.1 * 1.5 + 1e-4).count();
        let moved = doc.positions().iter().map(|p| p.length() - 1.0).fold(0.0f32, f32::max);
        println!("max displacement {moved:.4}");
        println!("{label}: {} patches near edits; true/stored worst ratio {:.1}; {} underestimated by >1.5x", rows.len(), worst, under);
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        for r in rows.iter().take(6) {
            println!("  node {:>5}: stored {:.5}  true {:.5}  ({} verts)", r.0, r.1, r.2, r.3);
        }
    };
    doc.refresh_lod_now();
    report(&doc, "after edits + refresh");
    doc.build_lod(LodParams::default());
    report(&doc, "fresh build after edits");
}

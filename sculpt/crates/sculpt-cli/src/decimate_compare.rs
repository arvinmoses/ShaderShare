//! `sculpt-cli decimate-compare [out_dir] [seed] [voxel]`: a noisy rock built from scatter and noise generator layers,
//! decimated at a few targets and evenness settings, with wireframe pictures and error numbers.
//!
//! Writes `rock.obj`, one `keep_<pct>.png` grid per target (columns: input, evenness 0, 0.5, 1; rows: whole rock shaded,
//! close up with wireframe) and prints a table.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat3, Vec3};
use sculpt_core::decimate::{Analysis, AnalyzeOptions, Surface, Target, TriMesh, check_manifold, even_out, surface_error};
use sculpt_core::document::SolidSettings;
use sculpt_core::generator::layer::{GeneratorKind, GeneratorSpec, NoiseSpec};
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const TILE: usize = 480;
const SS: usize = 2;

pub(crate) fn run(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("decimate_compare", |s| s.as_str()));
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let voxel: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.006);
    std::fs::create_dir_all(out)?;

    let t = Instant::now();
    let input = build_rock(seed, voxel)?;
    println!("rock: {} tris, {} verts in {:.1} s", input.triangles.len(), input.positions.len(), t.elapsed().as_secs_f64());
    std::fs::write(out.join("rock.obj"), obj(&input))?;
    let input = Arc::new(input);
    let surface = Surface::new(input.clone());

    let evens = [0.0f32, 0.5, 1.0];
    let analyses: Vec<Analysis> = evens
        .iter()
        .map(|&e| {
            let t = Instant::now();
            let a = Analysis::build(&input, &AnalyzeOptions { evenness: e, ..Default::default() }, None).expect("not cancelled");
            println!("analysis at evenness {e}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
            a
        })
        .collect();

    let views = [View::whole(0.6, 0.35), View::close(0.6, 0.35, &input)];
    let input_tiles: Vec<Vec<u8>> = views.iter().map(|v| draw(&input, v, false)).collect();
    println!("| keep | evenness | tris | min angle | valence dev | mean err | max err | closed+manifold |");
    println!("|---|---|---|---|---|---|---|---|");
    for pct in [5.0f32, 1.0, 0.25] {
        let mut cols = vec![input_tiles.clone()];
        for (a, &e) in analyses.iter().zip(&evens) {
            let mut m = a.extract(Target::KeepPercent(pct)).mesh;
            if e > 0.0 {
                even_out(&mut m, Some(&surface), e, (e * 3.0).ceil() as usize, true);
            }
            let (ma, dev) = shape(&m);
            let err = surface_error(&input, &m, 300_000);
            let r = check_manifold(m.positions.len(), &m.triangles);
            println!(
                "| {pct}% | {e} | {} | {ma:.1}° | {dev:.2} | {:.4}% | {:.3}% | {} |",
                m.triangles.len(),
                100.0 * err.mean_rel(),
                100.0 * err.max_rel(),
                r.closed() && r.manifold()
            );
            cols.push(views.iter().enumerate().map(|(i, v)| draw(&m, v, i == 1)).collect());
        }
        save_grid(&cols, &out.join(format!("keep_{}.png", pct.to_string().replace('.', "_"))))?;
    }
    Ok(())
}

/// A lumpy ellipsoid with big and medium box cuts, chips, added knobs, and two noise layers, solidified.
fn build_rock(seed: u64, voxel: f32) -> Res<TriMesh> {
    let lumps = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 1.1, octaves: 3, seed: seed as u32, ..Default::default() });
    let mut mesh = quad_sphere(6, 1.0);
    for p in &mut mesh.positions {
        let dir = *p;
        *p = dir * Vec3::new(1.0, 0.72, 0.88) * (1.0 + (lumps.sample(dir) - 0.5) * 0.25);
    }
    let mut doc = Document::from_mesh(mesh)?;
    let scatter = |primitive, op, count, size, lift, spacing, tilt, k: u64, blend| GeneratorSpec {
        primitive,
        op,
        count,
        seed: seed + k,
        size,
        lift,
        min_spacing: spacing,
        size_var: 0.35,
        stretch_var: 0.25,
        tilt,
        lift_var: 0.12,
        blend,
        ..GeneratorSpec::default()
    };
    let noise = |amplitude, scale, octaves, k: u32| GeneratorSpec {
        kind: GeneratorKind::Noise,
        noise: NoiseSpec { amplitude, scale, octaves, seed: seed as u32 + k },
        ..GeneratorSpec::default()
    };
    doc.insert_generator("Big cuts", Placement::Top, scatter(Primitive::Box, BoolOp::Subtract, 14, 0.6, 0.8, 0.75, 55.0, 0, 0.12))?;
    doc.insert_generator("Knobs", Placement::Top, scatter(Primitive::Sphere, BoolOp::Add, 12, 0.12, -0.2, 0.4, 0.0, 3, 0.3))?;
    doc.insert_generator("Medium cuts", Placement::Top, scatter(Primitive::Box, BoolOp::Subtract, 28, 0.2, 0.9, 0.35, 50.0, 1, 0.25))?;
    doc.insert_generator("Chips", Placement::Top, scatter(Primitive::Sphere, BoolOp::Subtract, 40, 0.06, 0.65, 0.12, 0.0, 2, 0.5))?;
    doc.insert_generator("Lumps", Placement::Top, noise(0.04, 3.0, 3, 7))?;
    doc.insert_generator("Grain", Placement::Top, noise(0.012, 14.0, 4, 9))?;
    let (mesh, stats) = doc.solidify(&SolidSettings::new(voxel))?;
    println!("solidify: {} layers, {} quads (scatter {:.0} ms, extract {:.0} ms)", stats.layers, stats.quads, stats.scatter_ms, stats.extract_ms);
    Ok(TriMesh::from_poly(&mesh))
}

/// Smallest angle in degrees, and the mean squared deviation of valences from 6.
fn shape(m: &TriMesh) -> (f32, f32) {
    let mut min_angle = 180.0f32;
    for t in &m.triangles {
        let p = t.map(|v| m.positions[v as usize]);
        for i in 0..3 {
            min_angle = min_angle.min((p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i]).to_degrees());
        }
    }
    let mut val = vec![0i32; m.positions.len()];
    m.triangles.iter().flatten().for_each(|&v| val[v as usize] += 1);
    (min_angle, val.iter().map(|&n| ((n - 6) * (n - 6)) as f32).sum::<f32>() / val.len().max(1) as f32)
}

struct View {
    rot: Mat3,
    center: Vec3,
    scale: f32,
}

impl View {
    fn whole(yaw: f32, pitch: f32) -> View {
        View { rot: Mat3::from_rotation_x(pitch) * Mat3::from_rotation_y(yaw), center: Vec3::ZERO, scale: 0.0 }
    }

    /// Zoomed 4x on the front surface point nearest the view axis.
    fn close(yaw: f32, pitch: f32, m: &TriMesh) -> View {
        let rot = Mat3::from_rotation_x(pitch) * Mat3::from_rotation_y(yaw);
        let c = mid(m);
        let p = m
            .positions
            .iter()
            .filter(|p| {
                let q = rot * (**p - c);
                q.x.abs() < 0.05 && q.y.abs() < 0.05
            })
            .max_by(|a, b| (rot * (**a - c)).z.total_cmp(&(rot * (**b - c)).z))
            .copied()
            .unwrap_or(c);
        View { rot, center: p, scale: 4.0 }
    }
}

fn mid(m: &TriMesh) -> Vec3 {
    let (lo, hi) = m.positions.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    (lo + hi) * 0.5
}

/// Clay shading with optional wireframe, `TILE` square, supersampled.
fn draw(m: &TriMesh, view: &View, wire: bool) -> Vec<u8> {
    let s = TILE * SS;
    let (lo, hi) = m.positions.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let (center, scale) = if view.scale == 0.0 { ((lo + hi) * 0.5, 1.3 * s as f32 / (hi - lo).length()) } else { (view.center, view.scale * 0.85 * s as f32 / 2.6) };
    let pts: Vec<Vec3> = m
        .positions
        .iter()
        .map(|p| {
            let q = view.rot * (*p - center);
            Vec3::new(q.x * scale + s as f32 * 0.5, s as f32 * 0.5 - q.y * scale, q.z)
        })
        .collect();
    let mut nrm = vec![Vec3::ZERO; m.positions.len()];
    for t in &m.triangles {
        let p = t.map(|v| m.positions[v as usize]);
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        t.iter().for_each(|&v| nrm[v as usize] += n);
    }
    let nrm: Vec<Vec3> = nrm.iter().map(|n| view.rot * n.normalize_or(Vec3::Z)).collect();
    let mut depth = vec![f32::NEG_INFINITY; s * s];
    let mut px = vec![Vec3::new(0.16, 0.17, 0.19); s * s];
    let light = Vec3::new(-0.4, 0.6, 0.7).normalize();
    let half = (light + Vec3::Z).normalize();
    for t in &m.triangles {
        let [a, b, c] = t.map(|v| v as usize);
        let (pa, pb, pc) = (pts[a], pts[b], pts[c]);
        let area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
        if area.abs() < 1e-9 {
            continue;
        }
        let x0 = pa.x.min(pb.x).min(pc.x).floor().max(0.0) as usize;
        let x1 = (pa.x.max(pb.x).max(pc.x).ceil().max(0.0) as usize).min(s - 1);
        let y0 = pa.y.min(pb.y).min(pc.y).floor().max(0.0) as usize;
        let y1 = (pa.y.max(pb.y).max(pc.y).ceil().max(0.0) as usize).min(s - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        // Pixel distance to each edge is w_i * |2 area| / |opposite edge|.
        let el = [pb.truncate().distance(pc.truncate()), pc.truncate().distance(pa.truncate()), pa.truncate().distance(pb.truncate())];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((pb.x - fx) * (pc.y - fy) - (pb.y - fy) * (pc.x - fx)) / area;
                let w1 = ((pc.x - fx) * (pa.y - fy) - (pc.y - fy) * (pa.x - fx)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = pa.z * w0 + pb.z * w1 + pc.z * w2;
                let i = y * s + x;
                if z <= depth[i] {
                    continue;
                }
                depth[i] = z;
                let n = (nrm[a] * w0 + nrm[b] * w1 + nrm[c] * w2).normalize_or(Vec3::Z);
                let diff = n.dot(light).max(0.0);
                let rim = (1.0 - n.z.abs()).powi(3) * 0.3;
                let spec = n.dot(half).max(0.0).powi(24) * 0.2;
                let mut col = Vec3::new(0.78, 0.62, 0.52) * (0.18 + 0.82 * diff) + Vec3::splat(rim + spec);
                if wire {
                    let d = (w0 * area.abs() / el[0]).min(w1 * area.abs() / el[1]).min(w2 * area.abs() / el[2]);
                    let k = (1.0 - (d - 0.3 * SS as f32).max(0.0) / SS as f32).clamp(0.0, 1.0);
                    col = col.lerp(Vec3::new(0.08, 0.1, 0.16), k * 0.7);
                }
                px[i] = col;
            }
        }
    }
    let mut rgb = vec![0u8; TILE * TILE * 3];
    for y in 0..TILE {
        for x in 0..TILE {
            let mut c = Vec3::ZERO;
            for dy in 0..SS {
                for dx in 0..SS {
                    c += px[(y * SS + dy) * s + x * SS + dx];
                }
            }
            c /= (SS * SS) as f32;
            for k in 0..3 {
                rgb[(y * TILE + x) * 3 + k] = (c[k].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            }
        }
    }
    rgb
}

/// Columns of tiles side by side, rows stacked, with a 4 px gap.
fn save_grid(cols: &[Vec<Vec<u8>>], path: &Path) -> Res<()> {
    let gap = 4;
    let rows = cols[0].len();
    let (w, h) = (cols.len() * TILE + (cols.len() - 1) * gap, rows * TILE + (rows - 1) * gap);
    let mut rgb = vec![40u8; w * h * 3];
    for (ci, col) in cols.iter().enumerate() {
        for (ri, tile) in col.iter().enumerate() {
            for y in 0..TILE {
                let o = ((ri * (TILE + gap) + y) * w + ci * (TILE + gap)) * 3;
                rgb[o..o + TILE * 3].copy_from_slice(&tile[y * TILE * 3..(y + 1) * TILE * 3]);
            }
        }
    }
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    Ok(())
}

fn obj(m: &TriMesh) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for p in &m.positions {
        let _ = writeln!(s, "v {} {} {}", p.x, p.y, p.z);
    }
    for t in &m.triangles {
        let _ = writeln!(s, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
    }
    s
}

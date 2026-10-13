//! `sculpt-cli adf-bench`: Spike 6, adaptively sampled distance field boolean.
//!
//! ```text
//! sculpt-cli adf-bench perf    [levels=7,8,9,10] [counts=1000,10000,100000] [--tol=0.5] [--detail[=AMP_EDGES,WAVELENGTH_EDGES]] [--test=face7|center|full19] [--op=sub|add|int]
//! sculpt-cli adf-bench quality [out_dir] [--images]
//! ```
//! `perf` times cold build, per-edit re-adaptation and extraction on the quad-sphere benchmark
//! meshes (6 * 4^level faces). `quality` measures deviation from the exact analytic boolean and
//! compares against uniform-grid surface nets (the uniform-brick approach).

use std::path::Path;
use std::time::Instant;

use glam::{Quat, Vec3};
use sculpt_core::Document;
use sculpt_core::generator::adf::{AdfMesh, AdfParams, BooleanAdf, Frame, TestPoints, uniform_surface_nets};
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::voxel::MeshSdf;
use sculpt_core::generator::{BoolOp, Instance, InstanceSet, Primitive};
use sculpt_core::geom::Aabb;
use sculpt_core::mesh::{NO_VERT, PolyMesh};
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
use sculpt_core::primitives::quad_sphere;

use crate::generator_bench::{make_instances, median, ms, rss_mb};
use crate::render;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run(args: &[String]) -> Res<()> {
    match args.first().map(|s| s.as_str()) {
        Some("perf") => perf(&args[1..]),
        Some("quality") => quality(&args[1..]),
        Some("qcost") => qcost(&args[1..]),
        Some("demo") => demo(&args[1..]),
        _ => Err("usage: adf-bench perf [levels] [counts] [flags] | adf-bench quality [out_dir] [--images]".into()),
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().find_map(|a| {
        if a == name {
            Some("")
        } else {
            a.strip_prefix(name).and_then(|r| r.strip_prefix('='))
        }
    })
}

fn parse_test(args: &[String]) -> TestPoints {
    match flag(args, "--test").unwrap_or("face7") {
        "center" => TestPoints::Center,
        "full19" => TestPoints::Full19,
        _ => TestPoints::Face7,
    }
}

fn list(args: &[String], i: usize, default: &str) -> Vec<usize> {
    args.iter()
        .filter(|a| !a.starts_with("--"))
        .nth(i)
        .map_or(default, |s| s.as_str())
        .split(',')
        .filter_map(|x| x.parse().ok())
        .collect()
}

/// `detail = (amplitude, wavelength)` in mesh edge lengths: fbm relief displaced along the normal
/// after subdivision, i.e. real edge-scale sculpt-like detail.
fn build_doc(level: usize, detail: Option<(f32, f32)>) -> Res<Document> {
    let mut doc = Document::from_mesh(quad_sphere((level as u32).min(6), 1.0))?;
    while (doc.level() as usize) + level.min(6) < level {
        doc.subdivide()?;
    }
    if let Some((amp, wl)) = detail {
        let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
        let noise = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 1.0 / (wl * edge), octaves: 2, lacunarity: 2.0, gain: 0.5, seed: 3, offset: Vec3::ZERO });
        let d = doc.compute_displacements(Vec3::ZERO, 10.0, |doc, v, _| {
            let p = doc.positions()[v];
            Some(doc.normals()[v] * (2.0 * amp * edge * (noise.sample(p) - 0.5)))
        });
        doc.apply_displacements(&d)?;
    }
    Ok(doc)
}

fn perf(args: &[String]) -> Res<()> {
    let levels = list(args, 0, "7,8,9,10");
    let counts = list(args, 1, "1000,10000,100000");
    let tols: Vec<f32> = flag(args, "--tol").unwrap_or("0.5").split(',').filter_map(|x| x.parse().ok()).collect();
    let detail = flag(args, "--detail").map(|v| {
        let mut it = v.split(',').filter_map(|x| x.parse::<f32>().ok());
        (it.next().unwrap_or(1.0), it.next().unwrap_or(8.0))
    });
    let test = parse_test(args);
    let block_scale: f32 = flag(args, "--block").and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let op = match flag(args, "--op").unwrap_or("sub") {
        "add" => BoolOp::Add,
        "int" => BoolOp::Intersect,
        _ => BoolOp::Subtract,
    };
    println!("threads: {}  test points: {test:?}  block scale: {block_scale}  op: {op:?}  detail: {detail:?}", rayon::current_num_threads());
    println!("\n| faces | N | tol (edges) | bake | members | base ADF | adapt | cold total | extract | tris | warm rebuild (base cached) | edit ALL: update+adapt | edit ALL: extract | edit 1%: update+adapt | edit 1%: extract | blocks | base cells | cells | surface cells | max depth | mesh queries | ADF MB |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &level in &levels {
        let t0 = Instant::now();
        let doc = build_doc(level, detail)?;
        let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
        let sampler = SurfaceSampler::new(pos, faces, None);
        let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
        let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
        let diag = doc.bounds().diagonal() * 2.0;
        let field = |p: Vec3, bound: f32| msdf.signed_distance(p, bound.min(diag));
        eprintln!("level {level}: {} faces, edge {edge:.5}, mesh+sdf setup {:.1} s, RSS {:.0} MB", doc.face_count(), t0.elapsed().as_secs_f64(), rss_mb("VmRSS"));
        for &n in &counts {
            let pts = sampler.sample(pos, nrm, faces, None, n, 7);
            let instances = make_instances(&pts, edge, 1.0, false);
            let mut bounds = doc.bounds();
            for i in &instances {
                bounds = bounds.union(&i.aabb());
            }
            let mean_ext = instances.iter().map(|i| i.aabb().extent().max_element()).sum::<f32>() / n as f32;
            for &tol in &tols {
                let mut params = AdfParams::new(edge, tol, op);
                params.test = test;
                params.block_scale = block_scale;
                let frame = Frame::new(bounds, &params, mean_ext);
                let (mut adf, bake) = BooleanAdf::new(frame, params, instances.clone());
                let bt = adf.build(&field);
                let t = Instant::now();
                let mesh = adf.extract(None);
                let t_ex = ms(t);
                eprintln!("extract: {} quads, {} quads with a vertexless cell (patched to a triangle or dropped)", mesh.quads, mesh.missing);
                let st = adf.stats();
                eprintln!("base: {} queries, {} cache hits, {:.0} ms CPU in mesh queries", st.base.mesh_queries, st.base.cache_hits, st.base.query_cpu_ms);
                // Warm: base ADF cached, combined tree rebuilt from scratch.
                let bt2 = adf.build(&field);

                // Edit ALL: every instance changes size.
                let scaled = |s: f32| -> Vec<(usize, Instance)> { instances.iter().enumerate().map(|(i, x)| (i, Instance { scale: x.scale * s, ..*x })).collect() };
                let (mut ua, mut ue) = (Vec::new(), Vec::new());
                for rep in 0..4 {
                    let ch = if rep % 2 == 0 { scaled(1.1) } else { scaled(1.0) };
                    let t = Instant::now();
                    let (dirty, _) = adf.update(&field, &ch);
                    let a = ms(t);
                    let t = Instant::now();
                    let _ = adf.extract(Some(&dirty));
                    let e = ms(t);
                    if rep > 0 {
                        ua.push(a);
                        ue.push(e);
                    }
                }
                // Make sure the state is the original one again (rep 3 restores scale 1.0).
                // Edit 1%: 1% of the instances change by x1.3.
                let k = (n / 100).max(1);
                let (mut pa, mut pe) = (Vec::new(), Vec::new());
                for rep in 0..5 {
                    let ch: Vec<(usize, Instance)> = (0..k)
                        .map(|j| {
                            let i = j * (n / k);
                            (i, if rep % 2 == 0 { Instance { scale: instances[i].scale * 1.3, ..instances[i] } } else { instances[i] })
                        })
                        .collect();
                    let t = Instant::now();
                    let (dirty, _) = adf.update(&field, &ch);
                    let a = ms(t);
                    let t = Instant::now();
                    let _ = adf.extract(Some(&dirty));
                    let e = ms(t);
                    if rep > 0 {
                        pa.push(a);
                        pe.push(e);
                    }
                }
                println!(
                    "| {} | {n} | {tol} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {:.0} | {:.0} | {:.0} | {:.1} | {:.1} | {} | {} | {} | {} | {} | {:.1}M | {:.0} |",
                    doc.face_count(),
                    bake,
                    bt.members_ms,
                    bt.base_ms,
                    bt.adapt_ms,
                    bt.total_ms() + bake,
                    t_ex,
                    mesh.tris.len(),
                    bt2.members_ms + bt2.adapt_ms,
                    median(ua),
                    median(ue),
                    median(pa),
                    median(pe),
                    st.blocks,
                    st.base.nodes,
                    st.cells,
                    st.surface_leaves,
                    st.max_depth_reached,
                    st.base.mesh_queries as f64 / 1e6,
                    st.bytes as f64 / 1e6
                );
            }
        }
        eprintln!("RSS {:.0} MB, peak {:.0} MB", rss_mb("VmRSS"), rss_mb("VmHWM"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Quality
// ---------------------------------------------------------------------------------------------

struct Scene {
    name: &'static str,
    insts: Vec<Instance>,
    /// Sample points on the analytic sharp features of the result.
    features: Vec<Vec3>,
    /// The 8 analytic corner points.
    corners: Vec<Vec3>,
}

const EDGE: f32 = 0.01;

fn scenes() -> Vec<Scene> {
    let sphere = Instance::new(Primitive::Sphere, Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.12));
    let rot = Quat::from_rotation_y(0.5);
    let a = 0.1f32;
    let cube = Instance::new(Primitive::Box, Vec3::ZERO, rot, Vec3::splat(a));
    let mut feats = Vec::new();
    let mut corners = Vec::new();
    // Cavity of a box centred on the plane y = 0: walls from y = 0 to y = -a.
    let pt = |x: f32, y: f32, z: f32| rot * Vec3::new(x, 0.0, z) + Vec3::new(0.0, y, 0.0);
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        corners.push(pt(sx * a, 0.0, sz * a));
        corners.push(pt(sx * a, -a, sz * a));
        for k in 0..=20 {
            let t = k as f32 / 20.0;
            feats.push(pt(sx * a, -a * t, sz * a)); // vertical concave edges
        }
    }
    for (sa, sb) in [(-1.0f32, 0.0f32), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
        for k in 0..=20 {
            let t = k as f32 / 20.0 * 2.0 - 1.0;
            let (x, z) = if sa != 0.0 { (sa * a, t * a) } else { (t * a, sb * a) };
            feats.push(pt(x, 0.0, z)); // rim (convex) edges
            feats.push(pt(x, -a, z)); // floor (concave) edges
        }
    }
    vec![
        Scene { name: "sphere", insts: vec![sphere], features: vec![Vec3::new(0.12, 0.0, 0.0), Vec3::new(-0.12, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.12), Vec3::new(0.0, 0.0, -0.12)], corners: vec![] },
        Scene { name: "cube", insts: vec![cube], features: feats, corners },
    ]
}

struct Metrics {
    vertex_max: f32,
    vertex_mean: f32,
    facet_max: f32,
    feature_max: f32,
    feature_mean: f32,
    corner_max: f32,
    tris: usize,
}

fn measure(mesh: &AdfMesh, exact: &impl Fn(Vec3) -> f32, scene: &Scene) -> Metrics {
    let vmax = mesh.positions.iter().map(|&p| exact(p).abs()).fold(0.0f32, f32::max);
    let vmean = mesh.positions.iter().map(|&p| exact(p).abs()).sum::<f32>() / mesh.positions.len().max(1) as f32;
    let mut fmax = 0.0f32;
    for t in &mesh.tris {
        let (p0, p1, p2) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
        for p in [(p0 + p1 + p2) / 3.0, (p0 + p1) * 0.5, (p1 + p2) * 0.5, (p0 + p2) * 0.5] {
            fmax = fmax.max(exact(p).abs());
        }
    }
    // Distance from analytic feature points to the extracted *surface*: a crisp edge lies on it,
    // a rounded or chamfered one is missed by up to the amount the corner was cut.
    let near = |q: Vec3| {
        mesh.tris
            .iter()
            .map(|t| dist_point_tri(q, mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]))
            .fold(f32::INFINITY, f32::min)
    };
    let fd: Vec<f32> = scene.features.iter().map(|&q| near(q)).collect();
    let cd: Vec<f32> = scene.corners.iter().map(|&q| near(q)).collect();
    Metrics {
        vertex_max: vmax,
        vertex_mean: vmean,
        facet_max: fmax,
        feature_max: fd.iter().cloned().fold(0.0, f32::max),
        feature_mean: fd.iter().sum::<f32>() / fd.len().max(1) as f32,
        corner_max: cd.iter().cloned().fold(0.0, f32::max),
        tris: mesh.tris.len(),
    }
}

fn dist_point_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Ericson, Real-Time Collision Detection 5.1.5
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return p.distance(a);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return p.distance(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return p.distance(a + ab * (d1 / (d1 - d3)));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return p.distance(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return p.distance(a + ac * (d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return p.distance(b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let inv = 1.0 / (va + vb + vc);
    p.distance(a + ab * (vb * inv) + ac * (vc * inv))
}

fn to_doc(m: &AdfMesh) -> Res<Document> {
    let faces = m.tris.iter().map(|t| [t[0], t[1], t[2], NO_VERT]).collect();
    Ok(Document::from_mesh(PolyMesh { positions: m.positions.clone(), faces })?)
}

fn quality(args: &[String]) -> Res<()> {
    let out = Path::new(args.iter().find(|a| !a.starts_with("--")).map_or("docs/generator/img", |s| s.as_str()));
    let images = args.iter().any(|a| a == "--images");
    if images {
        std::fs::create_dir_all(out)?;
    }
    let plane = |p: Vec3, _b: f32| p.y;
    let bounds = Aabb { min: Vec3::splat(-0.25), max: Vec3::splat(0.25) };
    println!("Edge length unit e = {EDGE}. Deviations are in units of e (max over vertices / over facet samples).\n");
    println!("| scene | method | tol or voxel | vertex max | vertex mean | facet max | feature miss max | feature miss mean | corner miss max | tris | cells or voxels |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for scene in scenes() {
        let set = InstanceSet::new(scene.insts.clone(), 0.0);
        let exact = |p: Vec3| {
            let m = (0..scene.insts.len() as u32).map(|i| set.sdf(i, p)).fold(f32::INFINITY, f32::min);
            p.y.max(-m)
        };
        let row = |method: &str, par: &str, m: &Metrics, cells: usize| {
            println!(
                "| {} | {method} | {par} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} |",
                scene.name,
                m.vertex_max / EDGE,
                m.vertex_mean / EDGE,
                m.facet_max / EDGE,
                m.feature_max / EDGE,
                m.feature_mean / EDGE,
                m.corner_max / EDGE,
                m.tris,
                cells
            );
        };
        for tol in [1.0f32, 0.5, 0.25] {
            let mut params = AdfParams::new(EDGE, tol, BoolOp::Subtract);
            params.test = parse_test(args);
            let frame = Frame::new(bounds, &params, 0.2);
            let mut adf = BooleanAdf::new(frame, params, scene.insts.clone()).0;
            adf.build(&plane);
            let mesh = adf.extract(None);
            row("ADF + DC", &format!("{tol}e"), &measure(&mesh, &exact, &scene), adf.stats().cells);
            if images && (tol - 0.5).abs() < 1e-6 {
                render::render(&to_doc(&mesh)?, &views(), 520, &out.join(format!("adf_{}_dc.png", scene.name)))?;
                cell_slice(&adf, out.join(format!("adf_{}_cells.png", scene.name)).as_path())?;
            }
        }
        for vs in [2.0f32, 1.0, 0.5] {
            let mesh = uniform_surface_nets(&exact, bounds, vs * EDGE);
            let n = ((bounds.extent() / (vs * EDGE)).ceil() + Vec3::ONE).to_array().iter().map(|v| *v as usize).product::<usize>();
            row("uniform nets", &format!("{vs}e voxel"), &measure(&mesh, &exact, &scene), n);
            if images && (vs - 1.0).abs() < 1e-6 {
                render::render(&to_doc(&mesh)?, &views(), 520, &out.join(format!("uniform_{}_nets.png", scene.name)))?;
            }
        }
    }
    Ok(())
}

fn views() -> Vec<render::View> {
    vec![render::View { yaw: 0.5, pitch: -0.9, overlay: None }, render::View { yaw: 1.2, pitch: -0.6, overlay: None }]
}

/// Cell structure on the slice y = -0.03 (inside the cavity), coloured by depth.
fn cell_slice(adf: &BooleanAdf, path: &Path) -> Res<()> {
    let size = 520usize;
    let y = -0.03f32;
    let (lo, span) = (-0.2f32, 0.4f32);
    let mut img = vec![[245u8, 245, 245]; size * size];
    let px = |v: f32| ((v - lo) / span * size as f32) as i32;
    let leaves = adf.leaf_boxes();
    let maxd = leaves.iter().map(|l| l.2).max().unwrap_or(1);
    for (min, s, d, has_v) in leaves {
        if y < min.y || y >= min.y + s {
            continue;
        }
        let (x0, x1, z0, z1) = (px(min.x), px(min.x + s), px(min.z), px(min.z + s));
        let t = (d as f32 - (maxd as f32 - 4.0)).max(0.0) / 4.0;
        let fill = [(235.0 - 190.0 * t) as u8, (240.0 - 120.0 * t) as u8, (250.0 - 60.0 * t) as u8];
        for zz in z0.max(0)..z1.min(size as i32) {
            for xx in x0.max(0)..x1.min(size as i32) {
                let edge = xx == x0 || zz == z0;
                img[zz as usize * size + xx as usize] = if edge { [40, 40, 50] } else if has_v { [230, 120, 60] } else { fill };
            }
        }
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, size as u32, size as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let data: Vec<u8> = img.iter().flat_map(|c| c.iter().copied()).collect();
    enc.write_header()?.write_image_data(&data)?;
    Ok(())
}

/// Render the ADF boolean patch of mixed primitives cut into a level-7 sphere.
fn demo(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("docs/generator/img", |s| s.as_str()));
    std::fs::create_dir_all(out)?;
    let doc = build_doc(7, None)?;
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let sampler = SurfaceSampler::new(pos, faces, None);
    let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
    let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
    let diag = doc.bounds().diagonal() * 2.0;
    let field = |p: Vec3, bound: f32| msdf.signed_distance(p, bound.min(diag));
    // Instances only on the +Z cap so the patch is one connected region that fits a close-up.
    let pts: Vec<_> = sampler
        .sample(pos, nrm, faces, None, 4000, 11)
        .into_iter()
        .filter(|s| s.position.z > 0.8 && s.position.x.abs() < 0.3 && s.position.y.abs() < 0.3)
        .take(60)
        .collect();
    let instances = make_instances(&pts, edge, 2.5, true);
    let mut bounds = doc.bounds();
    for i in &instances {
        bounds = bounds.union(&i.aabb());
    }
    let mean_ext = instances.iter().map(|i| i.aabb().extent().max_element()).sum::<f32>() / instances.len() as f32;
    let params = AdfParams::new(edge, 0.5, BoolOp::Subtract);
    let frame = Frame::new(bounds, &params, mean_ext);
    let mut adf = BooleanAdf::new(frame, params, instances).0;
    let bt = adf.build(&field);
    let mesh = adf.extract(None);
    println!("{} instances, cold {:.0} ms, {} tris, {} cells", pts.len(), bt.total_ms(), mesh.tris.len(), adf.stats().cells);
    let views = [render::View { yaw: 0.0, pitch: 0.0, overlay: None }, render::View { yaw: 0.6, pitch: -0.5, overlay: None }];
    render::render(&to_doc(&mesh)?, &views, 560, &out.join("adf_demo_patch.png"))?;
    Ok(())
}

/// Mesh closest-point query cost versus the search bound (in mesh edges), at random surface points.
fn qcost(args: &[String]) -> Res<()> {
    let level = list(args, 0, "9")[0];
    let doc = build_doc(level, None)?;
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let sampler = SurfaceSampler::new(pos, faces, None);
    let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
    let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
    let pts = sampler.sample(pos, nrm, faces, None, 20000, 5);
    println!("| bound (edges) | us per query (1 thread) |");
    println!("|---|---|");
    for off in [0.0f32, 1.0, 4.0, 8.0, 16.0] {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for (i, s) in pts.iter().enumerate() {
            let sgn = if i % 2 == 0 { 1.0 } else { -1.0 };
            let p = s.position + s.normal * (edge * off * sgn);
            acc += msdf.signed_distance(p, (off + 1.0) * edge * 1.2 + 1e-4);
        }
        println!("| offset {off} edges, tight bound | {:.2} (acc {acc:.1}) |", t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64);
    }
    for b in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 1e6] {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for (i, s) in pts.iter().enumerate() {
            let p = s.position + s.normal * (edge * 0.3 * ((i % 7) as f32 - 3.0) / 3.0);
            acc += msdf.signed_distance(p, (b * edge).min(2.0));
        }
        println!("| {b} | {:.2} (acc {acc:.1}) |", t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64);
    }
    Ok(())
}

//! Headless benchmark, CPU/GPU comparison and PNG for the GPU displacement boolean.
//!
//! ```text
//! sculpt-app --gpu-bench [LEVELS] [COUNTS] [--op subtract|add|intersect|all] [--mixed]
//!                        [--reps N] [--cpu] [--check] [--png out.png]
//! ```
//! LEVELS are quad-sphere levels (7,8,9,10 = 98k / 393k / 1.57M / 6.29M faces), COUNTS instance
//! counts (default 1000,10000,100000). No window is created.

use std::time::Instant;

use glam::Vec3;
use sculpt_core::Document;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::{BoolOp, EvalParams, Instance, InstanceSet, Primitive, evaluate, frame_from_normal};
use sculpt_core::mesh::face_triangles;
use sculpt_core::primitives::quad_sphere;

use super::{GpuBoolean, headless};

fn rng(i: u64, k: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// Same instance recipe as `sculpt-cli generator-bench`: surface samples, aligned to the
/// normal with random spin, diameter 3..10 mesh edges (times `size`).
fn make_instances(doc: &Document, sampler: &SurfaceSampler, n: usize, edge: f32, mixed: bool, size: f32) -> Vec<Instance> {
    let pts = sampler.sample(doc.positions(), doc.normals(), doc.faces(), None, n, 7);
    pts.iter()
        .enumerate()
        .map(|(i, s)| {
            let d = edge * (3.0 + 7.0 * rng(i as u64, 1)) * size;
            let rot = frame_from_normal(s.normal, rng(i as u64, 2) * std::f32::consts::TAU);
            if mixed {
                let prim = Primitive::ALL[(rng(i as u64, 3) * 7.0) as usize % 7];
                let sc = Vec3::new(0.7 + 0.3 * rng(i as u64, 4), 0.7 + 0.3 * rng(i as u64, 5), 0.7 + 0.3 * rng(i as u64, 6)) * d * 0.5;
                Instance::new(prim, s.position, rot, sc)
            } else {
                Instance::new(Primitive::Sphere, s.position, rot, Vec3::splat(d * 0.5))
            }
        })
        .collect()
}

/// A quad-sphere document plus instances scattered on it.
pub struct Scene {
    pub doc: Document,
    pub edge: f32,
    pub instances: Vec<Instance>,
    sampler: SurfaceSampler,
    mixed: bool,
    size: f32,
}

impl Scene {
    pub fn new(level: usize, n: usize, mixed: bool, size: f32) -> Scene {
        let base = level.min(6);
        let mut doc = Document::from_mesh(quad_sphere(base as u32, 1.0)).expect("mesh");
        while (doc.level() as usize) + base < level {
            doc.subdivide().expect("subdivide");
        }
        let sampler = SurfaceSampler::new(doc.positions(), doc.faces(), None);
        let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
        let mut s = Scene { doc, edge, instances: Vec::new(), sampler, mixed, size };
        s.set_count(n);
        s
    }

    pub fn set_count(&mut self, n: usize) {
        self.instances = make_instances(&self.doc, &self.sampler, n, self.edge, self.mixed, self.size);
    }

    /// Reach used for Add / Intersect (and the AABB margin); Subtract needs none.
    pub fn params(&self, op: BoolOp) -> (EvalParams, f32) {
        let reach = 2.0 * self.edge;
        (EvalParams { op, reach, iterations: 3 }, if op == BoolOp::Subtract { 0.0 } else { reach })
    }

    /// Run the GPU boolean and the CPU generator on this scene and compare the per-vertex deltas.
    pub fn compare(&self, gb: &mut GpuBoolean, op: BoolOp) -> Comparison {
        let (params, margin) = self.params(op);
        gb.set_instances(&self.instances, margin);
        gb.evaluate(&params);
        let stats = gb.stats();
        let gpu = gb.read_delta();
        let set = InstanceSet::new(self.instances.clone(), margin);
        let mut cpu = vec![Vec3::ZERO; self.doc.vertex_count()];
        evaluate(self.doc.positions(), self.doc.normals(), self.doc.bvh(), &set, &params, &mut cpu);
        let errs: Vec<f32> = gpu.iter().zip(&cpu).map(|(g, c)| (*g - *c).length()).collect();
        Comparison {
            max_err: errs.iter().cloned().fold(0.0, f32::max),
            mean_err: errs.iter().map(|e| *e as f64).sum::<f64>() / errs.len() as f64,
            moved_cpu: cpu.iter().filter(|d| **d != Vec3::ZERO).count(),
            moved_gpu: gpu.iter().filter(|d| **d != Vec3::ZERO).count(),
            max_delta: cpu.iter().map(|d| d.length()).fold(0.0, f32::max),
            errs,
            cpu,
            gpu,
            stats,
        }
    }
}

pub struct Comparison {
    pub errs: Vec<f32>,
    pub cpu: Vec<Vec3>,
    pub gpu: Vec<Vec3>,
    pub max_err: f32,
    pub mean_err: f64,
    pub max_delta: f32,
    pub moved_cpu: usize,
    pub moved_gpu: usize,
    pub stats: super::GpuStats,
}

impl Comparison {
    pub fn frac_within(&self, tol: f32) -> f64 {
        self.errs.iter().filter(|e| **e <= tol).count() as f64 / self.errs.len() as f64
    }
    pub fn count_over(&self, tol: f32) -> usize {
        self.errs.iter().filter(|e| **e > tol).count()
    }
}

impl std::fmt::Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "verts {}, moved cpu/gpu {}/{}, max |delta| {:.5}, max err {:.2e}, mean err {:.2e}, active leaves {}, grid entries {}/{}",
            self.errs.len(),
            self.moved_cpu,
            self.moved_gpu,
            self.max_delta,
            self.max_err,
            self.mean_err,
            self.stats.active_leaves,
            self.stats.entries_used,
            self.stats.entries_capacity
        )
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map_or("n/a".into(), |x| format!("{x:.2}"))
}

// ------------------------------------------------------------------ rasterizer (PNG)

/// Smooth-shaded orthographic clay render of `pos` over `doc`'s faces, with a per-vertex red tint.
fn render_panel(doc: &Document, pos: &[Vec3], tint: &[f32], size: usize, out: &mut [u8], stride: usize, x_off: usize) {
    let (yaw, pitch) = (0.55f32, 0.35f32);
    let rot = glam::Mat3::from_rotation_x(pitch) * glam::Mat3::from_rotation_y(yaw);
    let scale = size as f32 * 0.46;
    let pts: Vec<Vec3> = pos.iter().map(|p| { let q = rot * *p; Vec3::new(q.x * scale + size as f32 * 0.5, size as f32 * 0.5 - q.y * scale, q.z) }).collect();
    let mut nrm = vec![Vec3::ZERO; pos.len()];
    for f in doc.faces() {
        for [a, b, c] in face_triangles(f) {
            let n = (pos[b as usize] - pos[a as usize]).cross(pos[c as usize] - pos[a as usize]);
            for v in [a, b, c] {
                nrm[v as usize] += n;
            }
        }
    }
    let nrm: Vec<Vec3> = nrm.iter().map(|n| rot * n.normalize_or(Vec3::Z)).collect();
    let mut depth = vec![f32::NEG_INFINITY; size * size];
    let light = Vec3::new(-0.4, 0.6, 0.7).normalize();
    for f in doc.faces() {
        for [a, b, c] in face_triangles(f) {
            let (pa, pb, pc) = (pts[a as usize], pts[b as usize], pts[c as usize]);
            let area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = pa.x.min(pb.x).min(pc.x).floor().max(0.0) as usize;
            let x1 = (pa.x.max(pb.x).max(pc.x).ceil().max(0.0) as usize).min(size - 1);
            let y0 = pa.y.min(pb.y).min(pc.y).floor().max(0.0) as usize;
            let y1 = (pa.y.max(pb.y).max(pc.y).ceil().max(0.0) as usize).min(size - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((pb.x - px) * (pc.y - py) - (pb.y - py) * (pc.x - px)) / area;
                    let w1 = ((pc.x - px) * (pa.y - py) - (pc.y - py) * (pa.x - px)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let z = pa.z * w0 + pb.z * w1 + pc.z * w2;
                    if z <= depth[y * size + x] {
                        continue;
                    }
                    depth[y * size + x] = z;
                    let n = (nrm[a as usize] * w0 + nrm[b as usize] * w1 + nrm[c as usize] * w2).normalize_or(Vec3::Z);
                    let t = (tint[a as usize] * w0 + tint[b as usize] * w1 + tint[c as usize] * w2).clamp(0.0, 1.0);
                    let diff = n.dot(light).max(0.0);
                    let spec = n.dot((light + Vec3::Z).normalize()).max(0.0).powi(24) * 0.2;
                    let base = Vec3::new(0.78, 0.62, 0.52).lerp(Vec3::new(0.95, 0.1, 0.1), t);
                    let col = base * (0.2 + 0.8 * diff) + Vec3::splat(spec);
                    let o = (y * stride + x_off + x) * 3;
                    for k in 0..3 {
                        out[o + k] = (col[k].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    }
                }
            }
        }
    }
}

/// Rows: Subtract, Add, Intersect. Columns: CPU, GPU, GPU tinted by |GPU - CPU| (red = 1% of an edge).
fn render_png(gb: &mut GpuBoolean, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let scene = Scene::new(6, 60, true, 1.6);
    gb.upload_mesh(scene.doc.positions(), scene.doc.normals(), scene.doc.bvh());
    let size = 420usize;
    let ops = [BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect];
    let (w, h) = (size * 3, size * ops.len());
    let mut rgb = vec![14u8; w * h * 3];
    for (row, op) in ops.iter().enumerate() {
        let c = scene.compare(gb, *op);
        println!("png row {op:?}: {c}");
        let base = scene.doc.positions();
        let cpu: Vec<Vec3> = base.iter().zip(&c.cpu).map(|(p, d)| *p + *d).collect();
        let gpu: Vec<Vec3> = base.iter().zip(&c.gpu).map(|(p, d)| *p + *d).collect();
        let zero = vec![0.0f32; base.len()];
        let tint: Vec<f32> = c.errs.iter().map(|e| e / (0.01 * scene.edge)).collect();
        let band = &mut rgb[row * size * w * 3..(row + 1) * size * w * 3];
        render_panel(&scene.doc, &cpu, &zero, size, band, w, 0);
        render_panel(&scene.doc, &gpu, &zero, size, band, w, size);
        render_panel(&scene.doc, &gpu, &tint, size, band, w, 2 * size);
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    println!("wrote {}", path.display());
    Ok(())
}

// ------------------------------------------------------------------ benchmark

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let positional: Vec<&String> = {
        let mut skip = false;
        args.iter()
            .filter(|a| {
                if skip {
                    skip = false;
                    return false;
                }
                if matches!(a.as_str(), "--op" | "--reps" | "--png") {
                    skip = true;
                    return false;
                }
                !a.starts_with("--")
            })
            .collect()
    };
    let list = |i: usize, default: &str| -> Vec<usize> { positional.get(i).map_or(default, |s| s.as_str()).split(',').filter_map(|x| x.parse().ok()).collect() };
    let levels = list(0, "7,8,9,10");
    let counts = list(1, "1000,10000,100000");
    let reps: usize = value("--reps").and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let mixed = flag("--mixed");
    let ops: Vec<BoolOp> = match value("--op").as_deref() {
        Some("add") => vec![BoolOp::Add],
        Some("intersect") => vec![BoolOp::Intersect],
        Some("all") => vec![BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect],
        _ => vec![BoolOp::Subtract],
    };

    let Some((device, queue, info)) = headless() else {
        eprintln!("no wgpu adapter available (need Vulkan / Metal / DX12 or a software Vulkan driver such as lavapipe)");
        std::process::exit(2);
    };
    println!("adapter: {} | {:?} | {:?} | driver: {} {}", info.name, info.device_type, info.backend, info.driver, info.driver_info);
    let mut gb = GpuBoolean::new(device, queue);
    println!("timestamp queries: {} | mixed primitives: {mixed} | reps (median): {reps} | cpu threads: {}", gb.has_timestamps(), rayon::current_num_threads());
    if let Some(p) = value("--png") {
        return render_png(&mut gb, std::path::Path::new(&p));
    }
    if gb.has_timestamps() {
        println!("grid/cull/eval = GPU timestamp queries; wall = submit to fence-wait return");
    } else {
        println!("no timestamp queries: grid/cull/eval unavailable, use wall (submit to fence-wait return)");
    }

    for &level in &levels {
        let t = Instant::now();
        let mut scene = Scene::new(level, counts[0], mixed, 1.0);
        let (nv, nf) = (scene.doc.vertex_count(), scene.doc.face_count());
        let build_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        gb.upload_mesh(scene.doc.positions(), scene.doc.normals(), scene.doc.bvh());
        let mesh_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("\n## level {level}: {nf} faces, {nv} verts, {} leaves (CPU mesh build {build_s:.1} s, one-time mesh upload {mesh_ms:.1} ms)\n", scene.doc.bvh().leaves.len());
        println!("| op | instances | bake ms | upload ms | grid ms | cull ms | eval ms | gpu sum ms | wall ms | slider total ms | active/total leaves | cells | entries |{}", if flag("--cpu") { " cpu ms | cpu/gpu |" } else { "" });
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|{}", if flag("--cpu") { "---|---|" } else { "" });
        for &n in &counts {
            scene.set_count(n);
            for &op in &ops {
                let (params, margin) = scene.params(op);
                let mut up = super::UploadStats::default();
                let mut times = Vec::new();
                for rep in 0..=reps {
                    // Slider drag: every instance changes, so bake + upload + dispatch each time.
                    up = gb.set_instances(&scene.instances, margin);
                    let d = gb.evaluate(&params);
                    if rep > 0 {
                        times.push((up.bake_ms, up.upload_ms, d));
                    }
                }
                let st = gb.stats();
                let m = |f: &dyn Fn(&(f64, f64, super::DispatchTimes)) -> f64| median(times.iter().map(f).collect());
                let (bake, upload, wall) = (m(&|t| t.0), m(&|t| t.1), m(&|t| t.2.wall_ms));
                let has = gb.has_timestamps();
                let (grid, cull, eval) = if has { (Some(m(&|t| t.2.gpu_grid_ms.unwrap())), Some(m(&|t| t.2.gpu_cull_ms.unwrap())), Some(m(&|t| t.2.gpu_eval_ms.unwrap()))) } else { (None, None, None) };
                let sum = grid.zip(cull).zip(eval).map(|((a, b), c)| a + b + c);
                let mut row = format!(
                    "| {op:?} | {n} | {bake:.2} | {upload:.2} | {} | {} | {} | {} | {wall:.2} | {:.2} | {}/{} | {} | {} |",
                    fmt_opt(grid),
                    fmt_opt(cull),
                    fmt_opt(eval),
                    fmt_opt(sum),
                    bake + upload + wall,
                    st.active_leaves,
                    scene.doc.bvh().leaves.len(),
                    up.cells,
                    up.entries
                );
                if st.entries_used > st.entries_capacity {
                    row.push_str(" ENTRY OVERFLOW");
                }
                if flag("--cpu") {
                    let set = InstanceSet::new(scene.instances.clone(), margin);
                    let mut out = vec![Vec3::ZERO; nv];
                    evaluate(scene.doc.positions(), scene.doc.normals(), scene.doc.bvh(), &set, &params, &mut out);
                    let mut ts = Vec::new();
                    for _ in 0..3 {
                        let t = Instant::now();
                        let set = InstanceSet::new(scene.instances.clone(), margin);
                        evaluate(scene.doc.positions(), scene.doc.normals(), scene.doc.bvh(), &set, &params, &mut out);
                        ts.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    let cpu = median(ts);
                    row.push_str(&format!(" {cpu:.1} | {:.2}x |", cpu / (bake + upload + wall)));
                }
                println!("{row}");
                if flag("--check") {
                    let c = scene.compare(&mut gb, op);
                    println!("  check {op:?} n={n}: {c}; within 1e-4 edge: {:.5}, over 0.1 edge: {}", c.frac_within(1e-4 * scene.edge), c.count_over(0.1 * scene.edge));
                }
            }
        }
        println!("GPU buffers held: {:.0} MB", gb.gpu_bytes() as f64 / 1e6);
    }
    Ok(())
}

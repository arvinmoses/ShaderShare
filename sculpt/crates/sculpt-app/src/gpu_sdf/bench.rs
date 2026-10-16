//! Headless benchmark: `sculpt-app --gpu-sdf-bench [levels] [counts] [options]`.
//!
//! ```text
//! levels   quad-sphere levels, 7,8,9,10 = 98k / 393k / 1.6M / 6.3M faces   (default 7,8,9,10)
//! counts   instance counts                                                 (default 1000,10000,100000)
//! --voxels 1,0.5      voxel size as a multiple of the mean mesh edge length (default 1,0.5)
//! --max-bricks N      run at most N bricks per configuration and extrapolate by the brick ratio
//!                     (default: 6144 on a software adapter, 150000 on a real GPU; 0 = as many as one
//!                     storage binding of the adapter holds, i.e. exact wherever the cache fits)
//! --frames K          timed edit frames per configuration (the fastest is reported, default 5)
//! --mixed             all 7 primitives with random non-uniform scale instead of spheres
//! --no-resample       skip the CPU nearest-surface resampling measurement
//! --op subtract|union|intersect, --smooth K   boolean operation and smooth blend radius in voxels (default subtract, 0)
//! ```

use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;
use sculpt_core::Document;
use sculpt_core::generator::place::{SurfaceSample, SurfaceSampler};
use sculpt_core::generator::voxel::MeshSdf;
use sculpt_core::generator::{Instance, Primitive, frame_from_normal};
use sculpt_core::primitives::quad_sphere;

use super::{Gpu, Keep, Kernels, MeshGpu, Op, Sdf, config_for, enumerate_bricks, enumerate_with, prune_mesh, unpack_key};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub fn rng(i: u64, k: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// Same construction as `sculpt-cli generator-bench`: level L = 6 * 4^L quads.
pub fn make_doc(level: usize) -> Res<Document> {
    let mut doc = Document::from_mesh(quad_sphere((level as u32).min(6), 1.0))?;
    while (doc.level() as usize) + level.min(6) < level {
        doc.subdivide()?;
    }
    Ok(doc)
}

/// Instances centred on the surface, oriented to the normal. Diameter `lo..hi` mesh edges.
pub fn make_instances(samples: &[SurfaceSample], edge: f32, lo: f32, hi: f32, mixed: bool) -> Vec<Instance> {
    samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = edge * (lo + (hi - lo) * rng(i as u64, 1));
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

pub fn mean_edge(doc: &Document) -> f32 {
    let sampler = SurfaceSampler::new(doc.positions(), doc.faces(), None);
    (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt()
}

fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

fn fmt_count(v: usize) -> String {
    if v >= 1_000_000 { format!("{:.2}M", v as f64 / 1e6) } else { format!("{:.0}k", v as f64 / 1e3) }
}

fn fmt_ms(v: f64) -> String {
    if v >= 1000.0 { format!("{:.2} s", v / 1e3) } else if v >= 10.0 { format!("{v:.0} ms") } else { format!("{v:.2} ms") }
}

pub fn run(args: &[String]) -> Res<()> {
    let list = |i: usize, default: &str| -> Vec<f64> {
        args.get(i).filter(|s| !s.starts_with("--")).map_or(default, |s| s.as_str()).split(',').filter_map(|x| x.parse().ok()).collect()
    };
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let levels: Vec<usize> = list(0, "7,8,9,10").into_iter().map(|v| v as usize).collect();
    let counts: Vec<usize> = list(1, "1000,10000,100000").into_iter().map(|v| v as usize).collect();
    let voxels: Vec<f32> = opt("--voxels").unwrap_or("1,0.5".into()).split(',').filter_map(|x| x.parse().ok()).collect();
    let frames: usize = opt("--frames").and_then(|v| v.parse().ok()).unwrap_or(5);
    let mixed = args.iter().any(|a| a == "--mixed");
    let resample = !args.iter().any(|a| a == "--no-resample");
    let op = match opt("--op").as_deref() {
        None | Some("subtract") => Op::Subtract,
        Some("union") => Op::Union,
        Some("intersect") => Op::Intersect,
        Some(o) => return Err(format!("unknown --op {o} (subtract|union|intersect)").into()),
    };
    let smooth: f32 = opt("--smooth").and_then(|v| v.parse().ok()).unwrap_or(0.0); // blend radius in voxels

    let gpu = Gpu::new()?;
    println!("adapter: {}", gpu.describe());
    println!("cpu threads: {}   limits: max_buffer {} MB, max_storage_binding {} MB, workgroups/dim {}", rayon::current_num_threads(), gpu.limits.max_buffer_size >> 20, gpu.limits.max_storage_buffer_binding_size >> 20, gpu.limits.max_compute_workgroups_per_dimension);
    let kernels = Kernels::new(&gpu);
    println!("shader + {} pipelines compiled in {:.0} ms", 10, kernels.compile_ms);
    let max_bricks: usize = opt("--max-bricks").and_then(|v| v.parse().ok()).unwrap_or(if gpu.is_software() { 6144 } else { 150_000 });
    println!("op: {op:?}, smooth {smooth} voxels");
    println!("max bricks per run: {}   primitives: {}   frames: {frames}", if max_bricks == 0 { "unlimited".into() } else { max_bricks.to_string() }, if mixed { "7 mixed" } else { "spheres" });

    for &level in &levels {
        let t = Instant::now();
        let doc = make_doc(level)?;
        println!("\n## level {level}: {} faces, {} verts (built in {:.1} s, RSS {:.0} MB)", doc.face_count(), doc.vertex_count(), t.elapsed().as_secs_f64(), rss_mb());
        let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
        let edge = mean_edge(&doc);
        let t = Instant::now();
        let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
        let t_build = ms(t);
        let t = Instant::now();
        let flat = msdf.flatten();
        let t_flat = ms(t);
        let (u_bytes, f_bytes) = ((flat.nodes.len() * 32 + flat.tris.len() * 4) as u64, pos.len() as u64 * 24);
        let fits = u_bytes.max(f_bytes) <= gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size);
        let mesh_full = if fits { Some(MeshGpu::upload(&gpu, &flat, pos, msdf.pseudo_normals(), 0)) } else { None };
        println!(
            "mesh distance structure: CPU build {:.0} ms, flatten {:.0} ms, upload {}; {} nodes, {} triangles, {:.0} MB; mean edge {:.5}",
            t_build,
            t_flat,
            mesh_full.as_ref().map_or("n/a".into(), |m| format!("{:.0} ms", m.upload_ms)),
            flat.nodes.len(),
            flat.tris.len() / 3,
            (u_bytes + f_bytes) as f64 / 1e6,
            edge
        );
        if !fits {
            println!("  the whole mesh does not fit one storage binding ({} MB limit): each run uploads the sub-tree around its brick window instead", gpu.limits.max_storage_buffer_binding_size >> 20);
        }
        let bounds = doc.bounds();
        let sampler = SurfaceSampler::new(pos, faces, None);
        let mut resample_us: Option<f64> = None;
        let mut notes: Vec<String> = Vec::new();

        println!("\n| Faces | N | voxel | bricks (all / run) | pairs | avg list | cold: enum+assign | cold: classify | cold: base fill | cold total | base queries/s | frame: enum+assign | list | combine | verts | indices | **frame eval** | verts (owned / buffer) | tris | cache MB f32 (f16) | overlay MB | resample (CPU) | extrap. x |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        for &n in &counts {
            let samples = sampler.sample(pos, nrm, faces, None, n, 7);
            let insts = make_instances(&samples, edge, 3.0, 10.0, mixed);
            for &vx in &voxels {
                let voxel = vx * edge;
                let origin = bounds.min - Vec3::splat(4.0 * 8.0 * voxel);
                let en0 = enumerate_bricks(&insts, origin, voxel, 1);
                let limit_cap = (gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size) / 4000 * 9 / 10) as usize; // base pool must fit one binding
                let cap = if max_bricks == 0 { limit_cap } else { max_bricks.min(limit_cap) };
                let mut pruned_holder: Option<MeshGpu> = None;
                let need_window = mesh_full.is_none();
                let (keep, en) = if !need_window && en0.keys_all <= cap {
                    (Keep::ALL, en0)
                } else if !need_window {
                    let keep = Keep::stride(en0.keys_all.div_ceil(cap).max(1) as u32);
                    (keep, enumerate_with(&insts, origin, voxel, keep))
                } else {
                    // Brick window around the middle brick, grown until it holds about `cap` bricks and the
                    // sub-tree of the mesh it can query still fits one storage binding.
                    let c = unpack_key(en0.keys[en0.keys.len() / 2]);
                    let win_of = |h: i32| ([c[0] - h, c[1] - h, c[2] - h], [c[0] + h, c[1] + h, c[2] + h]);
                    let (mut lo_h, mut hi_h) = (1i32, 600i32);
                    while lo_h < hi_h {
                        let h = (lo_h + hi_h + 1) / 2;
                        let n = en0.keys.iter().filter(|&&k| Keep { stride: 1, win: Some(win_of(h)) }.keeps(k)).count();
                        if n <= cap { lo_h = h } else { hi_h = h - 1 }
                    }
                    let mut h = lo_h;
                    loop {
                        let win = win_of(h);
                        // Mesh sub-tree: window plus a generous query reach (12 bricks).
                        let bsz = 8.0 * voxel;
                        let wlo = origin + Vec3::new(win.0[0] as f32, win.0[1] as f32, win.0[2] as f32) * bsz - Vec3::splat(12.0 * bsz);
                        let whi = origin + Vec3::new(win.1[0] as f32 + 1.0, win.1[1] as f32 + 1.0, win.1[2] as f32 + 1.0) * bsz + Vec3::splat(12.0 * bsz);
                        let p = prune_mesh(&flat, pos, msdf.pseudo_normals(), wlo, whi);
                        let bytes = (p.flat.nodes.len() * 32 + p.flat.tris.len() * 4).max(p.positions.len() * 24) as u64;
                        if bytes <= gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size) || h == 1 {
                            pruned_holder = Some(MeshGpu::upload(&gpu, &p.flat, &p.positions, &p.pseudo, p.root));
                            break (Keep { stride: 1, win: Some(win) }, enumerate_with(&insts, origin, voxel, Keep { stride: 1, win: Some(win) }));
                        }
                        h = (h * 4 / 5).max(1);
                    }
                };
                let mesh: &MeshGpu = pruned_holder.as_ref().or(mesh_full.as_ref()).expect("mesh");
                let scale = en.keys_all as f64 / en.keys.len().max(1) as f64;
                let mut cfg = config_for(&en, origin, voxel, op, keep, 16384);
                cfg.smooth_k = smooth * voxel;
                let mut sdf = Sdf::new(&gpu, &kernels, mesh, cfg, insts.len());
                sdf.upload_instances(&insts);
                let cold = sdf.cold_build();
                let near = sdf.near_samples();
                let (visits, tests, nq) = sdf.query_stats();
                // Edit frames: nudge every instance by a few percent of a voxel, like a slider drag.
                let mut runs = Vec::new();
                for k in 0..frames.max(1) {
                    let jit: Vec<Instance> = insts
                        .iter()
                        .enumerate()
                        .map(|(i, x)| {
                            let d = Vec3::new(rng(i as u64, 10 + k as u64) - 0.5, rng(i as u64, 20 + k as u64) - 0.5, rng(i as u64, 30 + k as u64) - 0.5) * 0.1 * voxel;
                            Instance::new(x.primitive, x.translation + d, x.rotation, x.scale)
                        })
                        .collect();
                    let up = sdf.upload_instances(&jit);
                    let mut s = sdf.frame();
                    s.ms_upload = up;
                    runs.push(s);
                }
                runs.sort_by(|a, b| a.ms_eval().total_cmp(&b.ms_eval()));
                let fr = runs[0].clone(); // fastest of the frames: the container is shared, so contention only ever adds time
                // Scaled (extrapolated) stage times; enumeration/list building are reported as measured.
                let xs = |v: f64| v * scale;
                let (cold_cl, cold_fill) = (xs(cold.ms_classify), xs(cold.ms_fill));
                let cold_total = cold.ms_enum + cold.ms_assign + cold_cl + cold_fill;
                let (comb, vert, idx) = (xs(fr.ms_combine), xs(fr.ms_verts), xs(fr.ms_indices));
                let eval = fr.ms_enum + fr.ms_assign + fr.ms_reset + fr.ms_list + comb + vert + idx;
                let verts_all = fr.verts_owned as f64 * scale;
                let tris_all = fr.indices as f64 / 3.0 * scale;
                let qps = if cold.ms_fill > 0.0 { near as f64 / (cold.ms_fill / 1e3) } else { 0.0 };
                if resample && resample_us.is_none() && fr.verts > 0 {
                    resample_us = Some(measure_resample(&gpu, &sdf, &msdf, voxel, fr.verts.min(200_000) as usize));
                }
                let m = sdf.memory();
                let bricks_all = en.keys_all as f64;
                let base_mb = bricks_all * 4000.0 / 1e6;
                let overlay_mb = (fr.verts as f64 * 32.0 + fr.indices as f64 * 4.0) * scale / 1e6;
                println!(
                    "| {} | {} | {}e | {} / {} | {} | {:.1} | {} | {} | {} | **{}** | {:.1}M | {} | {} | {} | {} | {} | **{}** | {:.2}M / {:.2}M | {:.2}M | {:.0} ({:.0}) | {:.0} | {} | {} |",
                    doc.face_count(),
                    insts.len(),
                    vx,
                    en.keys_all,
                    en.keys.len(),
                    fmt_count(en.pairs_all),
                    en.pairs_all as f64 / bricks_all.max(1.0),
                    fmt_ms(cold.ms_enum + cold.ms_assign),
                    fmt_ms(cold_cl),
                    fmt_ms(cold_fill),
                    fmt_ms(cold_total),
                    qps / 1e6,
                    fmt_ms(fr.ms_enum + fr.ms_assign),
                    fmt_ms(fr.ms_list),
                    fmt_ms(comb),
                    fmt_ms(vert),
                    fmt_ms(idx),
                    fmt_ms(eval),
                    verts_all / 1e6,
                    fr.verts as f64 * scale / 1e6,
                    tris_all / 1e6,
                    base_mb,
                    base_mb / 2.0,
                    overlay_mb,
                    resample_us.map_or("-".into(), |us| fmt_ms(verts_all * us / 1e3)),
                    if scale > 1.0 { format!("{scale:.1}{}", if keep.win.is_some() { " (window)" } else { "" }) } else { "1 (exact)".into() },
                );
                if fr.overflow != 0 {
                    println!("  WARNING overflow flags {:#x} (1 slots, 2 pairs, 4 verts, 8 idx, 16 table, 32 coords)", fr.overflow);
                }
                notes.push(format!("N={} voxel={}e: {visits:.1} node visits + {tests:.1} triangle tests per base query ({nq} queries sampled); {:.0}% of cached samples are near-surface", insts.len(), vx, near as f64 / (sdf.cached_bricks().max(1) as f64 * 1000.0) * 100.0));
                eprintln!(
                    "    [{} bricks cached ({} new in frame), run GPU buffers {:.0} MB, upload {:.1} ms, build-new {:.0} ms, CPU brick enumeration (harness only) {:.0} ms]",
                    sdf.cached_bricks(),
                    fr.new_bricks,
                    m.total() as f64 / 1e6,
                    fr.ms_upload,
                    fr.ms_build_new,
                    en.ms
                );
            }
        }
        println!("\nBase query work (counting traversal, first 256 bricks of each cache):");
        for n in &notes {
            println!("* {n}");
        }
        if let Some(us) = resample_us {
            println!("\nCPU nearest-surface resample (closest-point query per new vertex, all {} threads): {:.2} us/vertex", rayon::current_num_threads(), us);
        }
    }
    Ok(())
}

/// Time nearest-surface queries from overlay vertices to the base mesh, the work a commit has to
/// do to carry per-vertex data (layers, masks, channels) onto the new surface. Returns microseconds
/// per vertex across all threads.
fn measure_resample(gpu: &Gpu, sdf: &Sdf, msdf: &MeshSdf, voxel: f32, n: usize) -> f64 {
    let v = gpu.read_f32(&sdf.verts, 0, n * 8);
    let pts: Vec<Vec3> = v.as_chunks::<8>().0.iter().map(|c| Vec3::new(c[0], c[1], c[2])).collect();
    let t = Instant::now();
    let s: f32 = pts.par_iter().map(|&p| msdf.signed_distance(p, 4.0 * voxel)).sum();
    let us = t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64;
    std::hint::black_box(s);
    us
}

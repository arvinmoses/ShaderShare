//! `sculpt-cli generator-bench`: timing for the Generator Boolean spike.
//!
//! ```text
//! sculpt-cli generator-bench [levels=7,8,9,10] [counts=1000,10000,100000] [--no-voxel] [--mixed]
//! ```
//! Levels are the same quad-sphere subdivision levels `bench` uses
//! (6 * 4^level faces: 7 = 98k, 8 = 393k, 9 = 1.57M, 10 = 6.29M).

use std::time::Instant;

use glam::Vec3;
use sculpt_core::Document;
use sculpt_core::generator::place::{EdgeSampler, SurfaceSampler};
use sculpt_core::generator::voxel::{self, MeshSdf};
use sculpt_core::generator::{
    BoolOp, EvalParams, Instance, InstanceSet, Primitive, affected_leaves, evaluate, evaluate_leaves, frame_from_normal,
};
use sculpt_core::primitives::quad_sphere;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub(crate) fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub(crate) fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Median of `reps` timed runs after one untimed warm-up.
fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f();
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        last = f();
        times.push(ms(t));
    }
    (median(times), last)
}

pub(crate) fn rng(i: u64, k: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

fn rss_mb(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

pub(crate) fn make_instances(samples: &[sculpt_core::generator::place::SurfaceSample], edge: f32, size: f32, mixed: bool) -> Vec<Instance> {
    samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // diameter 3..10 mesh edge lengths
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

pub fn run(args: &[String]) -> Res<()> {
    let list = |i: usize, default: &str| -> Vec<usize> {
        args.get(i)
            .filter(|s| !s.starts_with("--"))
            .map_or(default, |s| s.as_str())
            .split(',')
            .filter_map(|x| x.parse().ok())
            .collect()
    };
    let levels = list(0, "7,8,9,10");
    let counts = list(1, "1000,10000,100000");
    let do_voxel = !args.iter().any(|a| a == "--no-voxel");
    let mixed = args.iter().any(|a| a == "--mixed");
    println!("threads: {}  mixed primitives: {mixed}", rayon::current_num_threads());

    for &level in &levels {
        let t = Instant::now();
        let mut doc = Document::from_mesh(quad_sphere((level as u32).min(6), 1.0))?;
        while (doc.level() as usize) + level.min(6) < level {
            doc.subdivide()?;
        }
        println!("\n## level {level}: {} faces, {} verts, {} leaves (built in {:.1} s, RSS {:.0} MB)", doc.face_count(), doc.vertex_count(), doc.bvh().leaves.len(), t.elapsed().as_secs_f64(), rss_mb("VmRSS"));
        let (pos, nrm, faces, bvh) = (doc.positions(), doc.normals(), doc.faces(), doc.bvh());

        // --- Placement building blocks.
        let (t_cdf, sampler) = timed(3, || SurfaceSampler::new(pos, faces, None));
        let area = sampler.total_weight() as f32;
        let edge = (area / doc.face_count() as f32).sqrt();
        println!("mean edge length {edge:.5}, surface CDF build {t_cdf:.1} ms ({:.0} MB)", sampler.memory_bytes() as f64 / 1e6);

        // --- Edge sampling: same topology mapped onto a cube so it has real sharp edges.
        {
            let cube_pos: Vec<Vec3> = pos.iter().map(|p| *p / p.abs().max_element()).collect();
            let (t_det, es) = timed(1, || EdgeSampler::detect(&cube_pos, faces, doc.topology(), 30.0));
            let (t_s, pts) = timed(5, || es.sample(&cube_pos, 100_000, 3));
            println!(
                "edge placement (cube-mapped mesh, 30 deg): detect+chains {t_det:.1} ms (cold), sample 100k pts {t_s:.2} ms, {} sharp edges, {} pts",
                es.edges.len(),
                pts.len()
            );
        }

        let mesh_sdf = if do_voxel {
            let t = Instant::now();
            let m = MeshSdf::build(bvh, pos, faces, doc.topology());
            println!("voxel: mesh closest-point structure built in {:.0} ms, {:.0} MB", ms(t), m.memory_bytes() as f64 / 1e6);
            Some(m)
        } else {
            None
        };

        println!("\n| N | placement cold ms | placement warm ms | set build (bake+hash) ms | hash rebuild ms | slider: update+hash ms | slider: boolean ms | slider total ms | 1%: update+hash ms | 1%: leaf query ms | 1%: boolean ms | 1% total ms | leaves hit all / 1% / total | verts moved |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        for &n in &counts {
            // Placement cold = CDF + sample + orient/scale.
            let (t_cold, _) = timed(3, || {
                let s = SurfaceSampler::new(pos, faces, None);
                let pts = s.sample(pos, nrm, faces, None, n, 7);
                make_instances(&pts, edge, 1.0, mixed)
            });
            let (t_warm, instances) = timed(5, || {
                let pts = sampler.sample(pos, nrm, faces, None, n, 7);
                make_instances(&pts, edge, 1.0, mixed)
            });
            let params = EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 3 };
            let (t_build, mut set) = timed(3, || InstanceSet::new(instances.clone(), 0.0));
            let (t_hash, _) = timed(5, || set.rebuild_grid());
            let mut delta = vec![Vec3::ZERO; doc.vertex_count()];

            // Slider drag: every instance changes size.
            let bigger: Vec<Instance> = instances.iter().map(|i| Instance { scale: i.scale * 1.1, ..*i }).collect();
            let (t_upd, _) = timed(3, || set.set_all(bigger.clone()));
            let (t_eval, stats) = timed(5, || evaluate(pos, nrm, bvh, &set, &params, &mut delta));

            // 1% of instances change.
            set.set_all(instances.clone());
            evaluate(pos, nrm, bvh, &set, &params, &mut delta);
            let k = (n / 100).max(1);
            let changes: Vec<(usize, Instance)> = (0..k).map(|j| (j * (n / k), Instance { scale: instances[j * (n / k)].scale * 1.3, ..instances[j * (n / k)] })).collect();
            let mut tu = Vec::new();
            let mut tq = Vec::new();
            let mut te = Vec::new();
            let mut hit = 0;
            for rep in 0..6 {
                // alternate between the changed and original state
                let ch: Vec<(usize, Instance)> = if rep % 2 == 0 { changes.clone() } else { changes.iter().map(|&(i, _)| (i, instances[i])).collect() };
                let t = Instant::now();
                let dirty = set.update(&ch);
                let a = ms(t);
                let t = Instant::now();
                let leaves = affected_leaves(bvh, &dirty);
                let b = ms(t);
                let t = Instant::now();
                evaluate_leaves(pos, nrm, bvh, &set, &params, &leaves, &mut delta);
                let c = ms(t);
                hit = leaves.len();
                if rep > 0 {
                    tu.push(a);
                    tq.push(b);
                    te.push(c);
                }
            }
            let (a, b, c) = (median(tu), median(tq), median(te));
            println!(
                "| {n} | {t_cold:.1} | {t_warm:.2} | {t_build:.1} | {t_hash:.2} | {t_upd:.1} | {t_eval:.1} | {:.1} | {a:.2} | {b:.2} | {c:.2} | {:.2} | {}/{hit}/{} | {} |",
                t_upd + t_eval,
                a + b + c,
                stats.leaves_evaluated,
                bvh.leaves.len(),
                stats.vertices_moved
            );
        }

        // Cost of each op at every count (slider case). Add/Intersect need a reach margin.
        for &n in &counts {
            let pts = sampler.sample(pos, nrm, faces, None, n, 7);
            let instances = make_instances(&pts, edge, 1.0, mixed);
            let reach = 2.0 * edge;
            let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
            for op in [BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect] {
                let margin = if op == BoolOp::Subtract { 0.0 } else { reach };
                let set = InstanceSet::new(instances.clone(), margin);
                let params = EvalParams { op, reach, iterations: 3 };
                let (t, s) = timed(3, || evaluate(pos, nrm, bvh, &set, &params, &mut delta));
                println!("op {op:?} (N={n}, reach {margin:.4}): {t:.1} ms, {} verts moved, {} leaves", s.vertices_moved, s.leaves_evaluated);
            }
        }

        // --- Spike B.
        if let Some(mesh_sdf) = &mesh_sdf {
            println!("\n| N | voxel | bricks (dup) | bricks | near / far | voxels | bricks ms | classify ms | mesh SDF ms | combine ms | nets ms | total ms | net verts | net quads | brick MB | sampled |");
            println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
            for &n in &counts {
                let pts = sampler.sample(pos, nrm, faces, None, n, 7);
                let instances = make_instances(&pts, edge, 1.0, mixed);
                for ratio in [0.5f32, 1.0] {
                    let vs = edge * ratio;
                    let set = InstanceSet::new(instances.clone(), 2.0 * vs);
                    let r = voxel::measure(mesh_sdf, &set, vs, 2000);
                    println!(
                        "| {n} | {ratio}e | {} | {} | {}/{} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {:.0} | {} |",
                        r.bricks_with_duplicates,
                        r.bricks,
                        r.bricks_near,
                        r.bricks_far,
                        r.voxels,
                        r.ms_bricks,
                        r.ms_classify,
                        r.ms_mesh_sdf,
                        r.ms_combine,
                        r.ms_nets,
                        r.ms_total(),
                        r.net_vertices,
                        r.net_quads,
                        r.brick_memory_bytes as f64 / 1e6,
                        if r.extrapolated { format!("{} bricks, extrapolated", r.sample_bricks) } else { "all".into() }
                    );
                }
            }
        }
        println!("RSS {:.0} MB, peak {:.0} MB", rss_mb("VmRSS"), rss_mb("VmHWM"));
    }
    Ok(())
}

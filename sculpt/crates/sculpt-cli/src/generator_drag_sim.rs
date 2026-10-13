//! `sculpt-cli generator-drag-sim`: replays a slider drag against the Generator scheduler.
//!
//! ```text
//! sculpt-cli generator-drag-sim [levels=7,9,10] [counts=1000,10000,100000]
//!     [--view overview|zoom|both] [--frames 120] [--budget-ms 12] [--baseline-stride 4]
//!     [--slack 0.15] [--no-baseline] [--trace]
//! ```
//! Levels are quad-sphere subdivision levels as in `generator-bench` (7 = 98k, 9 = 1.57M, 10 = 6.29M faces).
//!
//! The drag has three phases of `frames / 3` frames each, replayed at a paced 60 Hz:
//!
//! * **size**: every instance grows from x1.0 to x1.3 (`Scheduler::map_all`, incremental hash).
//! * **count**: the active instance count falls to 50% and comes back (prefix of a shuffled placement,
//!   `Scheduler::apply_changes` with deactivation).
//! * **seed**: a new seed every frame (placement + instance set built on a worker thread via
//!   `Scheduler::request_replace`; the newest request wins).
//!
//! After each phase the drag is released and idle frames run until the visible cut is fresh and the background
//! full-quality job has landed. The result is then compared bit for bit with `evaluate()` on the same set.
//! The baseline repeats the same parameter changes with an unscheduled full recompute (rebuild the instance
//! set, evaluate every vertex) on every `baseline-stride`-th frame.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::Vec3;
use rayon::slice::ParallelSliceMut;
use sculpt_core::Document;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::schedule::{EvalMesh, FrameReport, Mode, SchedConfig, Scheduler};
use sculpt_core::generator::{BoolOp, EvalParams, Instance, InstanceSet, evaluate};
use sculpt_core::lod::{LodParams, View};
use sculpt_core::mesh::Face;
use sculpt_core::primitives::quad_sphere;

use crate::generator_bench::{make_instances, median, ms, rng};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const FRAME: Duration = Duration::from_micros(16_667);

struct Ctx {
    mesh: Arc<EvalMesh>,
    faces: Arc<Vec<Face>>,
    sampler: Arc<SurfaceSampler>,
    edge: f32,
}

/// `n` instances at seeded stratified-random positions, in a fixed shuffled order so any prefix is an even
/// spread (the count slider activates and deactivates a prefix).
fn placement(ctx: &Ctx, n: usize, seed: u64) -> Vec<Instance> {
    let pts = ctx.sampler.sample(&ctx.mesh.positions, &ctx.mesh.normals, &ctx.faces, None, n, seed);
    let inst = make_instances(&pts, ctx.edge, 1.0, false);
    let mut order: Vec<usize> = (0..inst.len()).collect();
    order.par_sort_unstable_by_key(|&i| ((rng(i as u64, 77 + seed) * 1e7) as u64, i));
    order.into_iter().map(|i| inst[i]).collect()
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn make_view(dist: f32, aspect: f32) -> View {
    let (h, fov) = (1080.0f32, 35f32.to_radians());
    let focal = h / (2.0 * (fov / 2.0).tan());
    let eye = Vec3::new(0.0, 0.0, dist);
    let proj = glam::camera::rh::proj::directx::perspective(fov, aspect, dist * 0.01, dist * 100.0);
    let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    View::from_view_proj(proj * view, eye, focal)
}

#[derive(Default)]
struct PhaseLog {
    name: &'static str,
    frame_ms: Vec<f64>,
    fresh_items: Vec<f64>,
    fresh_leaves: Vec<f64>,
    visible: Vec<f64>,
    backface: Vec<f64>,
    skipped: Vec<f64>,
    evaluated_verts: Vec<f64>,
    cut_tris: Vec<f64>,
    change_ms: Vec<f64>,
    select_ms: Vec<f64>,
    eval_ms: Vec<f64>,
    over_budget: usize,
    max_stale_age: u32,
    base_ms: Vec<f64>,
    vis_frames: u32,
    vis_ms: f64,
    full_frames: u32,
    full_ms: f64,
    exact: bool,
    hash: (u64, u64),
    idle_max_ms: f64,
}

fn log_frame(log: &mut PhaseLog, rep: &FrameReport, frame_ms: f64, change_ms: f64, budget_ms: f64) {
    log.change_ms.push(change_ms);
    log.select_ms.push(rep.select_ms);
    log.eval_ms.push(rep.eval_ms);
    log.frame_ms.push(frame_ms);
    log.fresh_items.push(100.0 * rep.fresh_after as f64 / rep.visible_items.max(1) as f64);
    log.fresh_leaves.push(100.0 * rep.fresh_vert_frac as f64);
    log.visible.push(rep.visible_items as f64);
    log.backface.push(rep.culled as f64);
    log.skipped.push((rep.skipped_tiny + rep.skipped_subpixel) as f64);
    log.evaluated_verts.push(rep.evaluated_verts as f64);
    log.cut_tris.push(rep.cut_triangles as f64);
    log.max_stale_age = log.max_stale_age.max(rep.max_stale_age);
    if frame_ms > budget_ms + 4.0 {
        log.over_budget += 1;
    }
}

fn pace(start: Instant) {
    let spent = start.elapsed();
    if spent < FRAME {
        std::thread::sleep(FRAME - spent);
    }
}

/// Idle frames after a release until the visible cut is fresh and the background job landed.
fn settle(sched: &mut Scheduler, view: &View, budget_ms: f64, log: &mut PhaseLog) {
    let t_release = Instant::now();
    sched.release();
    let (mut vis_done, mut frames) = (false, 0u32);
    while frames < 900 {
        let start = Instant::now();
        let deadline = start + Duration::from_secs_f64(budget_ms / 1e3);
        let rep = sched.frame(view, deadline, Mode::Settle);
        log.idle_max_ms = log.idle_max_ms.max(ms(start));
        frames += 1;
        if !vis_done && rep.fresh_after == rep.visible_items && !sched.build_pending() {
            vis_done = true;
            log.vis_frames = frames;
            log.vis_ms = t_release.elapsed().as_secs_f64() * 1e3;
        }
        if sched.is_fully_converged() {
            break;
        }
        pace(start);
    }
    log.full_frames = frames;
    log.full_ms = t_release.elapsed().as_secs_f64() * 1e3;
    let mut full = vec![Vec3::ZERO; sched.out().len()];
    let mesh = sched.mesh().clone();
    evaluate(&mesh.positions, &mesh.normals, &mesh.bvh, sched.set(), sched.params(), &mut full);
    log.exact = sched.is_fully_converged() && full.iter().zip(sched.out()).all(|(a, b)| a == b);
}

struct Opts {
    frames: usize,
    budget_ms: f64,
    stride: usize,
    slack: f32,
    baseline: bool,
    trace: bool,
}

fn run_case(ctx: &Arc<Ctx>, lod: &Arc<sculpt_core::lod::LodTree>, n: usize, view: &View, o: &Opts) -> Vec<PhaseLog> {
    let params = EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 3 };
    let base0 = Arc::new(placement(ctx, n, 7));
    let set = InstanceSet::with_slack((*base0).clone(), 0.0, o.slack);
    let t_setup = Instant::now();
    let mut sched = Scheduler::new(ctx.mesh.clone(), lod.clone(), set, params, SchedConfig::default());
    println!("    scheduler setup {:.2} s (node vertex lists, normal cones, edge lengths)", t_setup.elapsed().as_secs_f64());
    let m = o.frames / 3;
    let mut logs: Vec<PhaseLog> = ["size", "count", "seed"].into_iter().map(|name| PhaseLog { name, ..Default::default() }).collect();

    // Converge at the starting parameters (not part of the replay).
    {
        let mut warm = PhaseLog::default();
        settle(&mut sched, view, o.budget_ms, &mut warm);
    }

    let mut cur: Arc<Vec<Instance>> = base0;
    let mut active = n;
    let mut bset_scratch;
    let mut bout = if o.baseline { vec![Vec3::ZERO; ctx.mesh.positions.len()] } else { Vec::new() };

    for (pi, log) in logs.iter_mut().enumerate() {
        for k in 0..m {
            let frac = k as f32 / (m - 1).max(1) as f32;
            let start = Instant::now();
            let deadline = start + Duration::from_secs_f64(o.budget_ms / 1e3);
            match pi {
                0 => {
                    // A global size slider: this frame's factor relative to the last one.
                    let (s, prev) = (1.0 + 0.3 * frac, 1.0 + 0.3 * (k.saturating_sub(1)) as f32 / (m - 1).max(1) as f32);
                    sched.rescale_all(if k == 0 { s } else { s / prev });
                }
                1 => {
                    let tri = 1.0 - (2.0 * frac - 1.0).abs();
                    let want = ((n as f32) * (1.0 - 0.5 * tri)) as usize;
                    if want < active {
                        let off: Vec<usize> = (want..active).collect();
                        sched.apply_changes(&[], &off);
                    } else if want > active {
                        let on: Vec<(usize, Instance)> = (active..want).map(|i| (i, cur[i])).collect();
                        sched.apply_changes(&on, &[]);
                    }
                    active = want;
                }
                _ => {
                    let c = ctx.clone();
                    let slack = o.slack;
                    let seed = 100 + k as u64;
                    sched.request_replace(move || InstanceSet::with_slack(placement(&c, n, seed), 0.0, slack));
                }
            }
            let change_ms = ms(start);
            let rep = sched.frame(view, deadline, Mode::Drag);
            let frame_ms = ms(start);
            if o.trace && frame_ms > o.budget_ms + 4.0 {
                println!(
                    "    slow frame: {} k={k} {frame_ms:.1} ms (change {change_ms:.1}, select {:.2}, eval {:.1}; {} nodes {} verts, stale {} deadline_hit {} swapped {})",
                    log.name, rep.select_ms, rep.eval_ms, rep.evaluated_items, rep.evaluated_verts, rep.visible_items - rep.fresh_after, rep.deadline_hit, rep.set_swapped
                );
            }
            log_frame(log, &rep, frame_ms, change_ms, o.budget_ms);
            pace(start);

            // Unscheduled baseline on the same parameters.
            if o.baseline && k % o.stride == 0 {
                let t = Instant::now();
                bset_scratch = match pi {
                    0 => {
                        let s = 1.0 + 0.3 * frac;
                        InstanceSet::new(cur.iter().map(|i| Instance { scale: i.scale * s, ..*i }).collect(), 0.0)
                    }
                    1 => {
                        let tri = 1.0 - (2.0 * frac - 1.0).abs();
                        let want = ((n as f32) * (1.0 - 0.5 * tri)) as usize;
                        InstanceSet::new(cur[..want.max(1)].to_vec(), 0.0)
                    }
                    _ => InstanceSet::new(placement(ctx, n, 100 + k as u64), 0.0),
                };
                evaluate(&ctx.mesh.positions, &ctx.mesh.normals, &ctx.mesh.bvh, &bset_scratch, &params, &mut bout);
                log.base_ms.push(ms(t));
            }
        }
        // Phase end: remember the instances the next phase starts from, release, settle.
        if pi == 0 {
            let s = 1.3;
            cur = Arc::new(cur.iter().map(|i| Instance { scale: i.scale * s, ..*i }).collect());
        }
        log.hash = (sched.set().stats.reindexed, sched.set().stats.compactions);
        settle(&mut sched, view, o.budget_ms, log);
        if pi == 1 {
            // Bring back to full count so the seed phase starts from N active instances.
            active = n;
        }
    }
    // Cancellation drill: release, wait until the background job is running, then start a new drag.
    {
        let cancelled0 = sched.stats.jobs_cancelled;
        sched.release();
        let mut running = false;
        for _ in 0..400 {
            let start = Instant::now();
            sched.frame(view, start + Duration::from_secs_f64(o.budget_ms / 1e3), Mode::Settle);
            if sched.job_running() {
                running = true;
                break;
            }
            pace(start);
        }
        let t = Instant::now();
        sched.rescale_all(1.1);
        let cancel_ms = ms(t);
        let mut drill = PhaseLog::default();
        settle(&mut sched, view, o.budget_ms, &mut drill);
        println!(
            "    cancellation drill: job running at interrupt {running}, cancelled {} (new drag call took {cancel_ms:.1} ms), final result exact: {}",
            sched.stats.jobs_cancelled - cancelled0,
            drill.exact
        );
    }
    println!(
        "    scheduler: jobs started {} cancelled {} adopted {}, sets swapped {}, builds superseded {}; hash re-registrations {}, compactions {}",
        sched.stats.jobs_started,
        sched.stats.jobs_cancelled,
        sched.stats.jobs_adopted,
        sched.stats.sets_swapped,
        sched.stats.builds_superseded,
        sched.set().stats.reindexed,
        sched.set().stats.compactions,
    );
    logs
}

/// Cost of keeping the spatial hash current: full rebuild versus the incremental paths.
fn hash_report(ctx: &Ctx, n: usize, slack: f32) {
    let base = placement(ctx, n, 7);
    let med = |mut v: Vec<f64>| median(std::mem::take(&mut v));
    let mut set = InstanceSet::with_slack(base.clone(), 0.0, slack);
    let full: Vec<f64> = (0..5).map(|_| { let t = Instant::now(); set.rebuild_grid(); ms(t) }).collect();
    let build: Vec<f64> = (0..3).map(|_| { let t = Instant::now(); let _ = InstanceSet::with_slack(base.clone(), 0.0, slack); ms(t) }).collect();
    let old_path: Vec<f64> = {
        let k = (n / 100).max(1);
        let ch: Vec<(usize, Instance)> = (0..k).map(|j| (j * (n / k), Instance { scale: base[j * (n / k)].scale * 1.3, ..base[j * (n / k)] })).collect();
        (0..4).map(|_| { let t = Instant::now(); set.update(&ch); ms(t) }).collect()
    };
    let mut set = InstanceSet::with_slack(base.clone(), 0.0, slack);
    let k = (n / 100).max(1);
    let one_pct: Vec<f64> = (0..8)
        .map(|rep| {
            let f = if rep % 2 == 0 { 1.15 } else { 1.0 };
            let ch: Vec<(usize, Instance)> = (0..k).map(|j| (j * (n / k), Instance { scale: base[j * (n / k)].scale * f, ..base[j * (n / k)] })).collect();
            let t = Instant::now();
            set.update_incremental(&ch, &[]);
            ms(t)
        })
        .collect();
    let mut set = InstanceSet::with_slack(base.clone(), 0.0, slack);
    let mut cur = 1.0f32;
    let small: Vec<f64> = (0..8)
        .map(|_| {
            cur *= 1.004;
            let (b, c) = (&base, cur);
            let t = Instant::now();
            set.map_all(&|i, _| Some(Instance { scale: b[i].scale * c, ..b[i] }));
            ms(t)
        })
        .collect();
    let esc = set.stats;
    let mut set = InstanceSet::with_slack(base.clone(), 0.0, slack);
    let big: Vec<f64> = (0..4)
        .map(|rep| {
            let c = if rep % 2 == 0 { 1.4 } else { 1.0 };
            let b = &base;
            let t = Instant::now();
            set.map_all(&|i, _| Some(Instance { scale: b[i].scale * c, ..b[i] }));
            ms(t)
        })
        .collect();
    println!(
        "\n#### spatial hash, N={n} (4 threads): build {:.1} ms | full grid rebuild {:.1} ms | old update(1% changed)=rebuild {:.1} ms | incremental 1% changed (x1.15) {:.2} ms | map_all x1.004 per step {:.2} ms (re-registered {} of {n} over 8 steps) | map_all x1.4 jump {:.2} ms (median of alternating)",
        med(build),
        med(full),
        med(old_path),
        med(one_pct),
        med(small),
        esc.reindexed,
        med(big),
    );
}

pub fn run(args: &[String]) -> Res<()> {
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let pos_list = |i: usize, default: &str| -> Vec<usize> {
        args.get(i)
            .filter(|s| !s.starts_with("--"))
            .map_or(default, |s| s.as_str())
            .split(',')
            .filter_map(|x| x.parse().ok())
            .collect()
    };
    let levels = pos_list(0, "7,9,10");
    let counts = pos_list(1, "1000,10000,100000");
    let views = match flag("--view").as_deref() {
        Some("overview") => vec![("overview", 4.0f32)],
        Some("zoom") => vec![("zoom", 2.0)],
        _ => vec![("overview", 4.0), ("zoom", 2.0)],
    };
    let o = Opts {
        frames: flag("--frames").and_then(|s| s.parse().ok()).unwrap_or(120),
        budget_ms: flag("--budget-ms").and_then(|s| s.parse().ok()).unwrap_or(10.0),
        stride: flag("--baseline-stride").and_then(|s| s.parse().ok()).unwrap_or(4),
        slack: flag("--slack").and_then(|s| s.parse().ok()).unwrap_or(0.15),
        baseline: !args.iter().any(|a| a == "--no-baseline"),
        trace: args.iter().any(|a| a == "--trace"),
    };
    println!(
        "threads: {}  frames {} (3 phases)  budget {} ms  paced at 60 Hz  baseline stride {}  hash slack {}",
        rayon::current_num_threads(),
        o.frames,
        o.budget_ms,
        o.stride,
        o.slack
    );

    for &level in &levels {
        let t = Instant::now();
        let mut doc = Document::from_mesh(quad_sphere((level as u32).min(6), 1.0))?;
        while (doc.level() as usize) + level.min(6) < level {
            doc.subdivide()?;
        }
        let t_doc = t.elapsed().as_secs_f64();
        let t = Instant::now();
        doc.build_lod(LodParams::default());
        let lod = doc.lod().expect("built").clone();
        let t_lod = t.elapsed().as_secs_f64();
        let mesh = Arc::new(EvalMesh::from_document(&doc));
        let faces = Arc::new(doc.faces().to_vec());
        let sampler = Arc::new(SurfaceSampler::new(&mesh.positions, &faces, None));
        let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
        let nv = doc.vertex_count();
        let nf = doc.face_count();
        let leaves = mesh.bvh.leaves.len();
        drop(doc);
        println!("\n## level {level}: {nf} faces, {nv} verts, {leaves} leaves (mesh {t_doc:.1} s, LOD tree {t_lod:.1} s)");
        let ctx = Arc::new(Ctx { mesh, faces, sampler, edge });
        for &n in &counts {
            if args.iter().any(|a| a == "--hash-only") || !args.iter().any(|a| a == "--no-hash") {
                hash_report(&ctx, n, o.slack);
            }
            if args.iter().any(|a| a == "--hash-only") {
                continue;
            }
            for &(vname, dist) in &views {
                let view = make_view(dist, 16.0 / 9.0);
                println!("\n### {nf} faces, {n} instances, view {vname}");
                let t = Instant::now();
                let logs = run_case(&ctx, &lod, n, &view, &o);
                println!("    (case wall time {:.1} s)", t.elapsed().as_secs_f64());
                println!("\n| faces | N | view | phase | frame ms med | p95 | max | frames > budget+4ms | fresh nodes % med | min | fresh verts % med | visible nodes | subtrees culled | skipped | verts eval/frame | cut tris | max stale age (frames) | change / select / eval ms (med) | hash re-reg / rebuilds | baseline med | p95 | max | visible converged (frames / ms) | full converged (frames / ms) | idle frame max ms | exact |");
                println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
                let mut all_ms = Vec::new();
                for mut l in logs {
                    all_ms.extend(l.frame_ms.iter().copied());
                    println!(
                        "| {nf} | {n} | {vname} | {} | {:.1} | {:.1} | {:.1} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {:.1} / {:.2} / {:.1} | {} / {} | {} | {} | {} | {} / {:.0} | {} / {:.0} | {:.0} | {} |",
                        l.name,
                        median(l.frame_ms.clone()),
                        pct(&mut l.frame_ms, 0.95),
                        pct(&mut l.frame_ms, 1.0),
                        l.over_budget,
                        median(l.fresh_items.clone()),
                        pct(&mut l.fresh_items, 0.0),
                        median(l.fresh_leaves.clone()),
                        median(l.visible.clone()),
                        median(l.backface.clone()),
                        median(l.skipped.clone()),
                        median(l.evaluated_verts.clone()),
                        median(l.cut_tris.clone()),
                        l.max_stale_age,
                        median(l.change_ms.clone()),
                        median(l.select_ms.clone()),
                        median(l.eval_ms.clone()),
                        l.hash.0,
                        l.hash.1,
                        if l.base_ms.is_empty() { "-".into() } else { format!("{:.0}", median(l.base_ms.clone())) },
                        if l.base_ms.is_empty() { "-".into() } else { format!("{:.0}", pct(&mut l.base_ms, 0.95)) },
                        if l.base_ms.is_empty() { "-".into() } else { format!("{:.0}", pct(&mut l.base_ms, 1.0)) },
                        l.vis_frames,
                        l.vis_ms,
                        l.full_frames,
                        l.full_ms,
                        l.idle_max_ms,
                        if l.exact { "yes" } else { "NO" },
                    );
                }
                println!(
                    "    all {} drag frames: median {:.1} ms, p95 {:.1} ms, max {:.1} ms",
                    all_ms.len(),
                    median(all_ms.clone()),
                    pct(&mut all_ms, 0.95),
                    pct(&mut all_ms, 1.0)
                );
            }
        }
    }
    Ok(())
}

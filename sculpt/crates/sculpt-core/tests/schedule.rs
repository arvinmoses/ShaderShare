use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Quat, Vec3};
use sculpt_core::Document;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::schedule::{EvalMesh, Mode, SchedConfig, Scheduler};
use sculpt_core::generator::{BoolOp, EvalParams, Instance, InstanceSet, Primitive, evaluate, frame_from_normal};
use sculpt_core::lod::{LodParams, LodTree, View};
use sculpt_core::primitives::quad_sphere;

struct Fixture {
    mesh: Arc<EvalMesh>,
    lod: Arc<LodTree>,
    instances: Vec<Instance>,
}

fn fixture(n: usize) -> Fixture {
    let mut doc = Document::from_mesh(quad_sphere(6, 1.0)).unwrap();
    doc.subdivide().unwrap();
    doc.build_lod(LodParams::default());
    let lod = doc.lod().unwrap().clone();
    let mesh = Arc::new(EvalMesh::from_document(&doc));
    let sampler = SurfaceSampler::new(doc.positions(), doc.faces(), None);
    let edge = (sampler.total_weight() as f32 / doc.face_count() as f32).sqrt();
    let pts = sampler.sample(doc.positions(), doc.normals(), doc.faces(), None, n, 5);
    let instances = pts
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = edge * (3.0 + 5.0 * ((i * 7919 % 101) as f32 / 101.0));
            Instance::new(Primitive::Sphere, s.position, frame_from_normal(s.normal, i as f32), Vec3::splat(d * 0.5))
        })
        .collect();
    Fixture { mesh, lod, instances }
}

fn view(dist: f32) -> View {
    let (h, fov) = (1080.0f32, 35f32.to_radians());
    let focal = h / (2.0 * (fov / 2.0).tan());
    let eye = Vec3::new(0.0, 0.0, dist);
    let proj = glam::camera::rh::proj::directx::perspective(fov, 16.0 / 9.0, dist * 0.01, dist * 100.0);
    let v = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    View::from_view_proj(proj * v, eye, focal)
}

fn params() -> EvalParams {
    EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 3 }
}

fn full(mesh: &EvalMesh, set: &InstanceSet, p: &EvalParams) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; mesh.positions.len()];
    evaluate(&mesh.positions, &mesh.normals, &mesh.bvh, set, p, &mut out);
    out
}

/// Cold-start packing is one slice per worker thread, so the tests that need a deadline to cut a pass short pin a small
/// pool instead of depending on the host's core count.
fn on_small_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(f)
}

fn far() -> Instant {
    Instant::now() + Duration::from_secs(3600)
}

fn sched(f: &Fixture, cfg: SchedConfig) -> Scheduler {
    Scheduler::new(f.mesh.clone(), f.lod.clone(), InstanceSet::with_slack(f.instances.clone(), 0.0, 0.15), params(), cfg)
}

#[test]
fn background_job_converges_to_the_full_evaluation_exactly() {
    let f = fixture(3000);
    let mut s = sched(&f, SchedConfig::default());
    let v = view(4.0);
    // Drag: grow all instances for a while with a tiny budget, so most nodes stay stale.
    let base = f.instances.clone();
    for k in 0..12 {
        let scale = 1.0 + 0.04 * k as f32;
        let b = base.clone();
        s.map_all(&move |i, _| Some(Instance { scale: b[i].scale * scale, ..b[i] }));
        s.frame(&v, Instant::now() + Duration::from_micros(300), Mode::Drag);
    }
    assert!(!s.is_fully_converged());
    s.release();
    s.wait_job();
    assert!(s.is_fully_converged());
    let want = full(&f.mesh, s.set(), s.params());
    assert!(s.out() == want.as_slice(), "converged buffer differs from the full evaluation");
    // ... and from a set built from scratch (the incremental hash must not change any result).
    let final_instances: Vec<Instance> = base.iter().map(|i| Instance { scale: i.scale * (1.0 + 0.04 * 11.0), ..*i }).collect();
    let fresh = InstanceSet::new(final_instances, 0.0);
    assert!(s.out() == full(&f.mesh, &fresh, s.params()).as_slice());
}

#[test]
fn frame_loop_alone_converges_when_given_time() {
    let f = fixture(2000);
    // Leaf-level cut, no culling: the frame loop must reproduce the full evaluation on its own.
    let cfg = SchedConfig { tri_budget: u64::MAX, backface_cull: false, min_node_px: 0.0, min_instance_px: 0.0, min_spacing_px: 0.0, verts_per_instance: 1e9, ..Default::default() };
    let mut s = sched(&f, cfg);
    let v = view(4.0);
    let mut last = None;
    for _ in 0..40 {
        let r = s.frame(&v, far(), Mode::Settle);
        let done = r.fresh_after == r.visible_items;
        last = Some(r);
        if done {
            break;
        }
    }
    let r = last.unwrap();
    assert_eq!(r.fresh_after, r.visible_items);
    assert!(s.out() == full(&f.mesh, s.set(), s.params()).as_slice());
}

#[test]
fn cancellation_discards_the_old_job_and_never_mixes_parameters() {
    let f = fixture(3000);
    let mut s = sched(&f, SchedConfig { job_chunk_leaves: 2, job_after_visible: false, ..Default::default() });
    let v = view(4.0);
    s.release();
    assert!(s.job_running());
    // A newer drag arrives while the job runs.
    let base = f.instances.clone();
    s.map_all(&move |i, _| Some(Instance { scale: base[i].scale * 1.3, ..base[i] }));
    assert!(!s.job_running());
    assert_eq!(s.stats.jobs_cancelled, 1);
    assert!(!s.is_fully_converged());
    s.frame(&v, Instant::now() + Duration::from_millis(2), Mode::Drag);
    s.release();
    s.wait_job();
    assert_eq!(s.stats.jobs_adopted, 1, "only the job for the latest parameters is adopted");
    assert!(s.is_fully_converged());
    assert!(s.out() == full(&f.mesh, s.set(), s.params()).as_slice());
    // Changing parameters after adoption makes the buffer stale again until the next job.
    s.set_params(EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 1 });
    assert!(!s.is_fully_converged());
    s.release();
    s.wait_job();
    assert!(s.out() == full(&f.mesh, s.set(), s.params()).as_slice());
}

#[test]
fn replaced_set_lands_without_blocking_and_restarts_the_job() {
    let f = fixture(2000);
    let mut s = sched(&f, SchedConfig::default());
    let v = view(4.0);
    let other: Vec<Instance> = f.instances.iter().map(|i| Instance { translation: i.translation * 1.0, scale: i.scale * 1.2, ..*i }).collect();
    let o2 = other.clone();
    s.request_replace(move || InstanceSet::with_slack(o2, 0.0, 0.15));
    s.release();
    s.wait_builds();
    s.frame(&v, far(), Mode::Settle);
    s.wait_job();
    assert!(s.is_fully_converged());
    assert!(s.out() == full(&f.mesh, &InstanceSet::new(other, 0.0), s.params()).as_slice());
}

#[test]
fn no_visible_node_starves_under_constant_change() {
    let f = fixture(3000);
    let mut s = sched(&f, SchedConfig::default());
    let v = view(4.0);
    let base = f.instances.clone();
    let mut visible = 0;
    let mut worst = 0;
    for k in 0..300 {
        let scale = 1.0 + 0.2 * ((k % 20) as f32 / 20.0);
        let b = base.clone();
        s.map_all(&move |i, _| Some(Instance { scale: b[i].scale * scale, ..b[i] }));
        // Deadline already passed: exactly the minimal batch runs every frame.
        let r = s.frame(&v, Instant::now(), Mode::Drag);
        visible = r.visible_items;
        if k > 100 {
            worst = worst.max(r.max_stale_age);
        }
    }
    let bound = 4 * visible as u32 + 8;
    assert!(worst <= bound, "a visible node waited {worst} frames (bound {bound}, {visible} visible nodes)");
}

#[test]
fn view_aware_work_is_a_fraction_of_the_mesh() {
    let f = fixture(3000);
    let mut s = sched(&f, SchedConfig::default());
    let r = s.frame(&view(4.0), far(), Mode::Drag);
    assert!(r.culled > 0, "the far side of a sphere is not scheduled");
    let total = f.mesh.positions.len();
    assert!(r.evaluated_verts < total, "{} of {total}", r.evaluated_verts);
    // Zoomed in, the frustum removes more.
    let mut z = sched(&f, SchedConfig::default());
    let rz = z.frame(&view(1.5), far(), Mode::Drag);
    assert!(rz.cut_nodes > 0 && rz.evaluated_verts < r.evaluated_verts, "zoomed {} vs overview {}", rz.evaluated_verts, r.evaluated_verts);
}

#[test]
fn deadline_bounds_the_pass_and_work_carries_over() {
    on_small_pool(|| {
        let f = fixture(3000);
        let mut s = sched(&f, SchedConfig::default());
        let v = view(4.0);
        let r1 = s.frame(&v, Instant::now(), Mode::Drag);
        assert!(r1.deadline_hit);
        assert!(r1.fresh_after < r1.visible_items);
        let mut prev = r1.fresh_after;
        for _ in 0..200 {
            let r = s.frame(&v, Instant::now() + Duration::from_micros(500), Mode::Drag);
            assert!(r.fresh_after >= prev.min(r.visible_items), "progress must not be lost without a change");
            prev = r.fresh_after;
            if !r.deadline_hit {
                break;
            }
        }
        assert!(s.visible_converged(&v));
    });
}

#[test]
fn incremental_hash_matches_a_fresh_build() {
    let f = fixture(2500);
    let mut set = InstanceSet::with_slack(f.instances.clone(), 0.0, 0.1);
    let mut cur = f.instances.clone();
    // Sparse moves, a global resize, deactivation and reactivation.
    let moves: Vec<(usize, Instance)> = (0..200)
        .map(|k| {
            let i = k * 12;
            (i, Instance { translation: cur[i].translation + Vec3::new(0.03, -0.02, 0.01), rotation: Quat::IDENTITY, ..cur[i] })
        })
        .collect();
    for &(i, inst) in &moves {
        cur[i] = inst;
    }
    set.update_incremental(&moves, &[3, 4, 5]);
    let c2 = cur.clone();
    set.map_all(&move |i, x| if i == 3 || i == 5 { None } else { Some(Instance { scale: c2[i].scale * (1.0 + 0.0001 * i as f32), ..*x }) });
    for (i, c) in cur.iter_mut().enumerate() {
        c.scale *= 1.0 + 0.0001 * i as f32;
    }
    let back: Vec<(usize, Instance)> = vec![(4, cur[4])];
    set.update_incremental(&back, &[]);
    let mut live = cur.clone();
    let mut off = vec![false; cur.len()];
    off[3] = true;
    off[5] = true;
    // Reference: a fresh set without the deactivated instances (swap them out of reach).
    for (i, o) in off.iter().enumerate() {
        if *o {
            live[i].scale = Vec3::splat(1e-6);
            live[i].translation = Vec3::splat(100.0);
        }
    }
    let fresh = InstanceSet::new(live, 0.0);
    let mesh = &*fixture_mesh(&f);
    let p = params();
    assert!(full(mesh, &set, &p) == full(mesh, &fresh, &p), "incremental hash changed the result");
    // Brute-force candidate coverage.
    for (k, v) in mesh.positions.iter().step_by(997).enumerate() {
        let got = set.candidates(*v);
        for (i, inst) in cur.iter().enumerate() {
            if off[i] {
                continue;
            }
            let a = inst.aabb();
            if v.cmpge(a.min).all() && v.cmple(a.max).all() {
                assert!(got.contains(&(i as u32)), "sample {k}: missing instance {i}");
            }
        }
    }
    assert!(set.stats.reindexed > 0);
}

fn fixture_mesh(f: &Fixture) -> Arc<EvalMesh> {
    f.mesh.clone()
}

#[test]
fn full_quality_job_waits_for_the_visible_cut() {
    on_small_pool(|| {
        let f = fixture(2000);
        let mut s = sched(&f, SchedConfig::default());
        let v = view(4.0);
        s.release();
        assert!(s.job_deferred() && !s.job_running());
        // No time: visible nodes stay stale, so the job must not start yet.
        let r = s.frame(&v, Instant::now(), Mode::Settle);
        assert!(r.fresh_after < r.visible_items && !s.job_running());
        // Enough time: the visible cut finishes and the job starts in the same frame.
        let r = s.frame(&v, far(), Mode::Settle);
        assert_eq!(r.fresh_after, r.visible_items);
        assert!(s.job_running() && !s.job_deferred());
        s.wait_job();
        assert!(s.is_fully_converged());
        assert!(s.out() == full(&f.mesh, s.set(), s.params()).as_slice());
    });
}

#[test]
fn rescale_all_matches_a_rebuilt_set() {
    let f = fixture(2500);
    let mut set = InstanceSet::with_slack(f.instances.clone(), 0.0, 0.1);
    for r in [1.1f32, 1.1, 0.95, 1.2] {
        set.rescale_all(r);
    }
    let total = 1.1f32 * 1.1 * 0.95 * 1.2;
    let fresh = InstanceSet::new(f.instances.iter().map(|i| Instance { scale: i.scale * total, ..*i }).collect(), 0.0);
    let p = params();
    let (a, b) = (full(&f.mesh, &set, &p), full(&f.mesh, &fresh, &p));
    let worst = a.iter().zip(&b).map(|(x, y)| (*x - *y).length()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "rescaled set differs from a rebuilt one by {worst}");
    assert!(a.iter().any(|d| *d != Vec3::ZERO));
}

//! Interactive scheduler for the Generator Boolean (spike 4).
//!
//! The displacement boolean ([`super::evaluate`]) costs 58 ms at 10k instances and 278 ms at 100k on a
//! 6.3M-face mesh when every vertex is recomputed. A slider drag has about 16 ms. This module closes the gap
//! with two techniques that share one data structure, the LOD tree's cut:
//!
//! 1. **View-aware evaluation.** Work is scheduled per *cut node* (an LOD node the renderer would draw this
//!    frame). Only the vertices a node's triangles reference are evaluated, so a coarse node costs a fraction
//!    of its full-resolution leaves, nodes outside the frustum or facing away are never touched, and nodes
//!    that are tiny on screen or whose instances are sub-pixel are skipped while a drag is in progress.
//! 2. **Progressive refinement.** While a parameter is dragged, [`Scheduler::frame`] runs a time-budgeted pass
//!    over the stale visible nodes in priority order (large and near first, with an age term so no visible node
//!    starves), carries unfinished nodes over to the next frame, and leaves the rest showing their last
//!    complete result. [`Scheduler::release`] starts a full-quality evaluation of every vertex on a background
//!    thread. A newer drag cancels it (at the next chunk boundary) and its result is never mixed with the
//!    new parameters; an uncancelled job's result replaces the displacement buffer wholesale, so the
//!    converged buffer is bit-identical to [`super::evaluate`] on the same instance set.
//!
//! Changes enter through [`Scheduler::apply_changes`] / [`Scheduler::map_all`] (incremental spatial hash, only
//! the leaves under changed instances become stale), [`Scheduler::set_params`] (everything stale) or
//! [`Scheduler::request_replace`] (a new instance set built on a worker thread, swapped in when ready, so a
//! seed or distribution change never stalls a frame).
//!
//! The scheduler does not look at the brush path: it owns a snapshot of positions and normals ([`EvalMesh`]).
//! Sculpting under a live generator must call [`Scheduler::mark_leaves`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::{Dirty, EvalParams, Instance, InstanceSet, affected_leaves, eval_vertex, evaluate_leaves};
use crate::bvh::{Bvh, NodeKind, split_ranges_mut};
use crate::lod::{Cut, LodTree, View};

/// Mesh data the evaluator reads, in the engine's internal vertex order. Cloned from the document so a
/// background job can own it.
pub struct EvalMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub bvh: Bvh,
}

impl EvalMesh {
    pub fn from_document(doc: &crate::Document) -> EvalMesh {
        EvalMesh { positions: doc.positions().to_vec(), normals: doc.normals().to_vec(), bvh: doc.bvh().clone() }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SchedConfig {
    /// LOD pixel tolerance passed to [`LodTree::select`].
    pub tau_px: f32,
    /// Triangle cap passed to [`LodTree::select`].
    pub tri_budget: u64,
    /// While dragging, skip nodes whose bounding box diagonal projects to fewer pixels than this.
    pub min_node_px: f32,
    /// While dragging, skip nodes whose instances (mean extent) project to fewer pixels than this.
    pub min_instance_px: f32,
    /// The cut refines (below the LOD tree's own error test) until a node touched by instances has vertices
    /// at most `max(min_spacing_px, instance_px / verts_per_instance)` pixels apart.
    pub min_spacing_px: f32,
    pub verts_per_instance: f32,
    /// While dragging, the cut is coarsened by this factor (pixel tolerances are multiplied), so a frame's
    /// budget covers more of the screen; `Mode::Settle` uses the full detail.
    pub drag_detail_scale: f32,
    /// Skip nodes facing away from the camera (normal cone test).
    pub backface_cull: bool,
    /// Priority bonus per frame since a node was last evaluated: `score *= 1 + age / age_scale`.
    pub age_scale: f32,
    /// Every this many frames the pass serves the *longest-waiting* stale node first instead of the highest
    /// priority one, which bounds how long any visible node can wait (`0` disables).
    pub fairness_period: u32,
    /// Smallest slice of a node's vertices evaluated at once when time is short.
    pub min_slice_verts: usize,
    /// Largest number of node slices evaluated between two deadline checks (the overshoot bound).
    pub max_batch_items: usize,
    /// Start the background full-quality job only once the visible cut is fresh (`release` defers it), so the
    /// foreground catches up first. `false` starts it inside `release`.
    pub job_after_visible: bool,
    /// Leaves per cancellation check in the background job.
    pub job_chunk_leaves: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig {
            tau_px: 1.0,
            tri_budget: 3_000_000,
            min_node_px: 3.0,
            min_instance_px: 0.75,
            min_spacing_px: 2.0,
            verts_per_instance: 3.0,
            drag_detail_scale: 3.0,
            backface_cull: true,
            age_scale: 6.0,
            fairness_period: 4,
            min_slice_verts: 256,
            max_batch_items: 32,
            job_after_visible: true,
            job_chunk_leaves: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// A parameter is being dragged: skip tiny nodes and sub-pixel instances.
    Drag,
    /// Idle or just released: refine everything visible.
    Settle,
}

/// What one [`Scheduler::frame`] call did.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameReport {
    pub frame: u32,
    pub cut_nodes: usize,
    pub cut_triangles: u64,
    /// Subtrees rejected by the frustum or the backface test while selecting the cut.
    pub culled: usize,
    /// Cut nodes that are candidates for this frame (visible).
    pub visible_items: usize,
    /// Visible nodes whose displacement matches the current parameters, before and after this frame's pass.
    pub fresh_before: usize,
    pub fresh_after: usize,
    /// `fresh_after` weighted by the number of BVH leaves each node covers, as a fraction of the visible total.
    pub fresh_leaf_frac: f32,
    /// Fraction of the visible nodes' vertices that hold a current result (partly evaluated nodes count).
    pub fresh_vert_frac: f32,
    pub evaluated_items: usize,
    pub evaluated_verts: usize,
    /// Nodes whose bounds touch no instance: zero-filled without evaluation.
    pub zero_filled_items: usize,
    pub skipped_tiny: usize,
    pub skipped_subpixel: usize,
    /// Largest `frame - last_evaluated` among visible nodes that are still stale (starvation indicator).
    pub max_stale_age: u32,
    pub select_ms: f64,
    pub eval_ms: f64,
    /// The pass stopped because the deadline arrived (work remains) rather than because it ran out of work.
    pub deadline_hit: bool,
    pub set_swapped: bool,
    pub job_adopted: bool,
    pub job_running: bool,
}

/// Cumulative counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedStats {
    pub jobs_started: u32,
    pub jobs_cancelled: u32,
    pub jobs_adopted: u32,
    pub sets_swapped: u32,
    pub builds_superseded: u32,
}

struct Job {
    epoch: u32,
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<Option<Vec<Vec3>>>,
}

type BuildFn = Box<dyn FnOnce() -> InstanceSet + Send>;

struct Build {
    handle: JoinHandle<InstanceSet>,
}

pub struct Scheduler {
    mesh: Arc<EvalMesh>,
    lod: Arc<LodTree>,
    set: Arc<InstanceSet>,
    params: EvalParams,
    cfg: SchedConfig,
    out: Vec<Vec3>,
    /// Unique vertices each node's triangles reference (what a node costs and what must be fresh to draw it).
    node_verts: Vec<Box<[u32]>>,
    parent: Vec<u32>,
    leaf_node: Vec<u32>,
    /// BVH leaves under each node.
    node_leaves: Vec<u32>,
    /// Typical triangle edge length of each node's own (possibly simplified) surface, world units.
    node_edge: Vec<f32>,
    /// Normal cone per node: unit axis and half-angle in radians (>= pi/2 means no backface culling).
    cones: Vec<(Vec3, f32)>,
    epoch: u32,
    /// Generation a finished full-quality job last covered.
    full_gen: u32,
    /// A full-quality job is wanted for the current parameters (set by `release`, cleared by any change).
    released: bool,
    /// `release` was called and the job has not started yet (waiting for the visible cut).
    job_deferred: bool,
    /// Small pool for instance-set builds so they do not take every core from the frame loop.
    bg_pool: Arc<rayon::ThreadPool>,
    need: Vec<u32>,
    done: Vec<u32>,
    last_eval: Vec<u32>,
    frame: u32,
    job: Option<Job>,
    build: Option<Build>,
    queued_build: Option<BuildFn>,
    /// Smoothed CPU seconds per evaluated vertex over all nodes, and per node (0 = not measured yet).
    cpu_per_vert: f64,
    node_cost: Vec<f32>,
    /// Epoch at which each vertex of `out` was last evaluated (a node can span several frames).
    vert_epoch: Vec<u32>,
    /// Rotating start of the next slice of each node's vertex list.
    cursor: Vec<u32>,
    /// Nodes whose vertices were written by the last `frame` (for incremental GPU upload).
    updated: Vec<u32>,
    /// Smoothed ratio of wall time to ideal parallel time (CPU time / threads): imbalance and contention.
    wall_scale: f64,
    pub stats: SchedStats,
}

impl Scheduler {
    /// `lod` must have been built from the same mesh (same BVH, same vertex order). Computes per-node vertex
    /// lists and normal cones (a few hundred ms at 6M faces); every node starts stale.
    pub fn new(mesh: Arc<EvalMesh>, lod: Arc<LodTree>, set: InstanceSet, params: EvalParams, cfg: SchedConfig) -> Scheduler {
        let nn = mesh.bvh.nodes.len();
        assert_eq!(lod.nodes.len(), nn, "LOD tree and BVH must match");
        let node_verts: Vec<Box<[u32]>> = lod
            .nodes
            .par_iter()
            .map(|n| {
                let mut v = lod.indices[n.first as usize..(n.first + n.count) as usize].to_vec();
                v.sort_unstable();
                v.dedup();
                v.into_boxed_slice()
            })
            .collect();
        let mut parent = vec![u32::MAX; nn];
        let mut leaf_node = vec![0u32; mesh.bvh.leaves.len()];
        for (i, n) in mesh.bvh.nodes.iter().enumerate() {
            match n.kind {
                NodeKind::Inner { left, right } => {
                    parent[left as usize] = i as u32;
                    parent[right as usize] = i as u32;
                }
                NodeKind::Leaf { leaf } => leaf_node[leaf as usize] = i as u32,
            }
        }
        // Children have larger ids than their parent, so reverse order is bottom-up.
        let mut node_leaves = vec![1u32; nn];
        let mut cones = vec![(Vec3::Y, std::f32::consts::PI); nn];
        let leaf_cones: Vec<(usize, (Vec3, f32))> = (0..nn)
            .into_par_iter()
            .filter(|&i| matches!(mesh.bvh.nodes[i].kind, NodeKind::Leaf { .. }))
            .map(|i| (i, cone_of(&node_verts[i], &mesh.normals)))
            .collect();
        for (i, c) in leaf_cones {
            cones[i] = c;
        }
        for i in (0..nn).rev() {
            if let NodeKind::Inner { left, right } = mesh.bvh.nodes[i].kind {
                node_leaves[i] = node_leaves[left as usize] + node_leaves[right as usize];
                cones[i] = merge_cones(cones[left as usize], cones[right as usize]);
            }
        }
        let node_edge: Vec<f32> = lod
            .nodes
            .par_iter()
            .map(|n| {
                let tris = lod.indices[n.first as usize..(n.first + n.count) as usize].as_chunks::<3>().0.iter();
                let count = tris.len().max(1) as f32;
                let area: f64 = tris
                    .map(|t| {
                        let (a, b, c) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
                        0.5 * (b - a).cross(c - a).length() as f64
                    })
                    .sum();
                1.52 * (area as f32 / count).sqrt()
            })
            .collect();
        let n_verts = mesh.positions.len();
        Scheduler {
            mesh,
            lod,
            set: Arc::new(set),
            params,
            cfg,
            out: vec![Vec3::ZERO; n_verts],
            node_verts,
            parent,
            leaf_node,
            node_leaves,
            node_edge,
            cones,
            epoch: 1,
            full_gen: 0,
            released: false,
            job_deferred: false,
            bg_pool: Arc::new(rayon::ThreadPoolBuilder::new().num_threads((rayon::current_num_threads() / 4).max(1)).build().expect("pool")),
            need: vec![1; nn],
            done: vec![0; nn],
            last_eval: vec![0; nn],
            frame: 0,
            job: None,
            build: None,
            queued_build: None,
            cpu_per_vert: 0.0,
            node_cost: vec![0.0; nn],
            vert_epoch: vec![0; n_verts],
            cursor: vec![0; nn],
            updated: Vec::new(),
            wall_scale: 1.3,
            stats: SchedStats::default(),
        }
    }

    /// Current displacement per vertex (internal order). Nodes not yet refreshed hold their last result.
    pub fn out(&self) -> &[Vec3] {
        &self.out
    }
    pub fn set(&self) -> &InstanceSet {
        &self.set
    }
    pub fn params(&self) -> &EvalParams {
        &self.params
    }
    pub fn config(&self) -> &SchedConfig {
        &self.cfg
    }
    pub fn config_mut(&mut self) -> &mut SchedConfig {
        &mut self.cfg
    }
    pub fn generation(&self) -> u32 {
        self.epoch
    }
    pub fn mesh(&self) -> &Arc<EvalMesh> {
        &self.mesh
    }
    /// Vertices a node must have fresh to be drawn.
    pub fn node_vertex_count(&self, node: u32) -> usize {
        self.node_verts[node as usize].len()
    }
    /// Nodes (LOD/BVH node ids) with vertices written by the last [`Scheduler::frame`]; upload
    /// [`Scheduler::node_vertices`] of each. When the report says `job_adopted`, the whole buffer changed.
    pub fn updated_nodes(&self) -> &[u32] {
        &self.updated
    }
    /// The vertices a node's triangles reference.
    pub fn node_vertices(&self, node: u32) -> &[u32] {
        &self.node_verts[node as usize]
    }
    pub fn job_running(&self) -> bool {
        self.job.is_some()
    }
    /// `release` was called and the job is waiting for the visible cut to finish.
    pub fn job_deferred(&self) -> bool {
        self.job_deferred
    }
    pub fn build_pending(&self) -> bool {
        self.build.is_some() || self.queued_build.is_some()
    }

    // ---- changes -------------------------------------------------------------------------------------

    /// Move / resize some instances (and deactivate others). Cancels a running full-quality job. Only the
    /// leaves under the old and new boxes become stale.
    pub fn apply_changes(&mut self, changes: &[(usize, Instance)], deactivate: &[usize]) {
        self.cancel_job();
        let dirty = self.set_mut().update_incremental(changes, deactivate);
        self.mark(dirty);
    }

    /// Change every instance in parallel (`None` deactivates). Everything becomes stale.
    pub fn map_all(&mut self, f: &(impl Fn(usize, &Instance) -> Option<Instance> + Sync)) {
        self.cancel_job();
        let dirty = self.set_mut().map_all(f);
        self.mark(dirty);
    }

    /// Multiply every active instance's size by `ratio` about its centre (a global size slider). Everything
    /// becomes stale. Far cheaper than [`Scheduler::map_all`] at 100k instances.
    pub fn rescale_all(&mut self, ratio: f32) {
        self.cancel_job();
        let dirty = self.set_mut().rescale_all(ratio);
        self.mark(dirty);
    }

    /// Change the boolean parameters (op, reach, iterations). Everything becomes stale.
    pub fn set_params(&mut self, params: EvalParams) {
        self.cancel_job();
        self.params = params;
        self.mark(Dirty::All);
    }

    /// Note that vertices in these BVH leaves changed under the generator (sculpting below it). The caller
    /// must have updated `mesh` (a new [`EvalMesh`] via [`Scheduler::replace_mesh`]).
    pub fn mark_leaves(&mut self, leaves: &[u32]) {
        self.cancel_job();
        self.epoch += 1;
        for &l in leaves {
            self.mark_leaf_node(self.leaf_node[l as usize]);
        }
    }

    /// Install an updated snapshot of the mesh (same topology). Call [`Scheduler::mark_leaves`] after.
    pub fn replace_mesh(&mut self, mesh: Arc<EvalMesh>) {
        self.cancel_job();
        assert_eq!(mesh.positions.len(), self.mesh.positions.len());
        self.mesh = mesh;
    }

    /// Ask for a whole new instance set, built on a worker thread. The old set and its results stay on
    /// screen until the new one is ready (swapped in by a later [`Scheduler::frame`]). Only one build runs at
    /// a time; a request made while one runs replaces any earlier queued request.
    pub fn request_replace(&mut self, build: impl FnOnce() -> InstanceSet + Send + 'static) {
        if self.build.is_some() {
            if self.queued_build.is_some() {
                self.stats.builds_superseded += 1;
            }
            self.queued_build = Some(Box::new(build));
        } else {
            self.build = Some(self.spawn_build(Box::new(build)));
        }
    }

    /// Block until the in-flight build (and any queued one) has been swapped in. For tests and simulators
    /// that need deterministic ordering; an app should poll instead.
    pub fn wait_builds(&mut self) {
        while self.build.is_some() {
            if let Some(b) = self.build.take() {
                self.land_build(b);
            }
        }
    }

    fn spawn_build(&self, build: BuildFn) -> Build {
        let pool = self.bg_pool.clone();
        Build { handle: std::thread::spawn(move || pool.install(build)) }
    }

    fn land_build(&mut self, b: Build) {
        let set = b.handle.join().expect("instance build panicked");
        let released = self.released;
        self.cancel_job();
        self.released = released;
        let old = std::mem::replace(&mut self.set, Arc::new(set));
        std::thread::spawn(move || drop(old));
        self.stats.sets_swapped += 1;
        self.mark(Dirty::All);
        if let Some(next) = self.queued_build.take() {
            self.build = Some(self.spawn_build(next));
        } else if released {
            self.release();
        }
    }

    fn set_mut(&mut self) -> &mut InstanceSet {
        Arc::get_mut(&mut self.set).expect("no job may hold the instance set while it is mutated")
    }

    fn mark(&mut self, dirty: Dirty) {
        self.epoch += 1;
        let g = self.epoch;
        match dirty {
            Dirty::All => self.need.fill(g),
            Dirty::Boxes(boxes) => {
                if boxes.len() > 20_000 {
                    self.need.fill(g);
                } else {
                    let leaves = affected_leaves(&self.mesh.bvh, &boxes);
                    for l in leaves {
                        self.mark_leaf_node(self.leaf_node[l as usize]);
                    }
                }
            }
        }
    }

    fn mark_leaf_node(&mut self, node: u32) {
        let g = self.epoch;
        let mut n = node;
        while n != u32::MAX && self.need[n as usize] != g {
            self.need[n as usize] = g;
            n = self.parent[n as usize];
        }
    }

    // ---- background full-quality job ---------------------------------------------------------------

    fn cancel_job(&mut self) {
        self.released = false;
        self.job_deferred = false;
        if let Some(j) = self.job.take() {
            j.cancel.store(true, Ordering::Relaxed);
            // Wait (at most one chunk) until the job no longer references the instance set, then detach it.
            if j.handle.is_finished() {
                let _ = j.handle.join();
            } else {
                while Arc::strong_count(&self.set) > 1 && !j.handle.is_finished() {
                    std::thread::yield_now();
                }
            }
            self.stats.jobs_cancelled += 1;
        }
    }

    /// The drag ended: start evaluating every vertex at full quality on a background thread. Cancelled by the
    /// next change. Its result replaces [`Scheduler::out`] when it finishes (seen by a later `frame`).
    pub fn release(&mut self) {
        self.cancel_job();
        self.released = true;
        if self.cfg.job_after_visible {
            self.job_deferred = true;
        } else {
            self.start_job();
        }
    }

    fn start_job(&mut self) {
        self.job_deferred = false;
        let (mesh, set, params, epoch) = (self.mesh.clone(), self.set.clone(), self.params, self.epoch);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let chunk = self.cfg.job_chunk_leaves.max(1);
        let handle = std::thread::spawn(move || {
            // Zero pages from the allocator, not a memset: a cancel arriving at once must not wait for 75 MB of writes.
            let mut out: Vec<Vec3> = bytemuck::allocation::zeroed_vec(mesh.positions.len());
            let leaves: Vec<u32> = (0..mesh.bvh.leaves.len() as u32).collect();
            for c in leaves.chunks(chunk) {
                if flag.load(Ordering::Relaxed) {
                    // Let go of the instance set first: the caller waits only for that, not for this thread
                    // to free its (large) output buffer.
                    drop(set);
                    return None;
                }
                evaluate_leaves(&mesh.positions, &mesh.normals, &mesh.bvh, &set, &params, c, &mut out);
            }
            Some(out)
        });
        self.stats.jobs_started += 1;
        self.job = Some(Job { epoch, cancel, handle });
    }

    fn poll_job(&mut self) -> bool {
        if self.job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            let j = self.job.take().unwrap();
            if let Ok(Some(out)) = j.handle.join()
                && j.epoch == self.epoch
            {
                self.out = out;
                self.done.fill(j.epoch);
                self.vert_epoch.fill(j.epoch);
                self.full_gen = j.epoch;
                self.stats.jobs_adopted += 1;
                return true;
            }
        }
        false
    }

    /// Block until the background job finishes (tests).
    pub fn wait_job(&mut self) {
        if self.job_deferred {
            self.start_job();
        }
        if let Some(j) = &self.job {
            while !j.handle.is_finished() {
                std::thread::yield_now();
            }
        }
        self.poll_job();
    }

    /// Every vertex holds the full-quality result for the current parameters and no build is outstanding.
    pub fn is_fully_converged(&self) -> bool {
        self.full_gen == self.epoch && self.job.is_none() && self.build.is_none() && self.queued_build.is_none()
    }

    /// Every visible node (per `view`, honouring the culling in [`SchedConfig`]) is fresh. Cheap check for
    /// "nothing left for the frame loop to do".
    pub fn visible_converged(&self, view: &View) -> bool {
        let cut = self.select_cut(view, Mode::Settle);
        cut.ids.iter().all(|&id| self.done[id as usize] >= self.need[id as usize])
    }

    /// The nodes to draw and evaluate this frame: the LOD tree's cut, refined where instances need vertices,
    /// minus whatever the frustum and the normal cones reject. Draw this cut (not a plain `LodTree::select`)
    /// so the displaced vertices the scheduler evaluates are the ones on screen.
    pub fn select_cut(&self, view: &View, mode: Mode) -> Cut {
        let mut scale = if mode == Mode::Drag { self.cfg.drag_detail_scale.max(1.0) } else { 1.0 };
        loop {
            let mut cut = Cut { tau_px: self.cfg.tau_px * scale, ..Default::default() };
            self.visit(0, view, scale, &mut cut);
            if cut.triangles <= self.cfg.tri_budget || scale > 1e4 {
                return cut;
            }
            scale *= 1.4;
        }
    }

    fn visit(&self, id: u32, view: &View, scale: f32, cut: &mut Cut) {
        let b = &self.mesh.bvh.nodes[id as usize].bounds;
        if b.is_empty() || !view.sees(b) || self.is_backfacing(id, view) {
            cut.culled += 1;
            return;
        }
        let node = &self.lod.nodes[id as usize];
        if let Some([l, r]) = node.children
            && (view.error_px(b, node.error) > self.cfg.tau_px * scale || self.needs_detail(id, b, view, scale))
        {
            self.visit(l, view, scale, cut);
            self.visit(r, view, scale, cut);
            return;
        }
        cut.nodes += 1;
        cut.ids.push(id);
        cut.triangles += node.count as u64 / 3;
        match cut.ranges.last_mut() {
            Some((first, count)) if *first + *count == node.first => *count += node.count,
            _ => cut.ranges.push((node.first, node.count)),
        }
    }

    /// Does a node holding instances have vertices too far apart on screen to show them?
    fn needs_detail(&self, id: u32, b: &crate::geom::Aabb, view: &View, scale: f32) -> bool {
        if !self.set.touches(b) {
            return false;
        }
        let dist = (view.eye.clamp(b.min, b.max) - view.eye).length().max(1e-3);
        let inst_px = self.set.cell_size() * view.focal_px / dist;
        if inst_px < self.cfg.min_instance_px {
            return false;
        }
        let spacing_px = self.node_edge[id as usize] * view.focal_px / dist;
        spacing_px > (self.cfg.min_spacing_px * scale).max(inst_px / self.cfg.verts_per_instance)
    }

    // ---- per-frame pass -----------------------------------------------------------------------------

    fn is_backfacing(&self, id: u32, view: &View) -> bool {
        if !self.cfg.backface_cull {
            return false;
        }
        let (axis, theta) = self.cones[id as usize];
        if theta >= std::f32::consts::FRAC_PI_2 {
            return false;
        }
        let b = &self.mesh.bvh.nodes[id as usize].bounds;
        let d = b.center() - view.eye;
        axis.dot(d) >= (theta + 0.02).sin() * d.length() + 0.5 * b.diagonal()
    }

    /// One frame of work until `deadline` (at least one small batch always runs, so a stale visible node is
    /// refreshed no matter how little time is left). Call after this frame's changes were applied.
    pub fn frame(&mut self, view: &View, deadline: Instant, mode: Mode) -> FrameReport {
        self.frame += 1;
        self.updated.clear();
        let mut rep = FrameReport { frame: self.frame, ..Default::default() };
        if self.build.as_ref().is_some_and(|b| b.handle.is_finished()) {
            let b = self.build.take().unwrap();
            self.land_build(b);
            rep.set_swapped = true;
        }
        rep.job_adopted = self.poll_job();
        rep.job_running = self.job.is_some();

        let t_sel = Instant::now();
        let mesh = self.mesh.clone();
        let set = self.set.clone();
        let cut = self.select_cut(view, mode);
        rep.select_ms = t_sel.elapsed().as_secs_f64() * 1e3;
        rep.cut_nodes = cut.ids.len();
        rep.cut_triangles = cut.triangles;
        rep.culled = cut.culled as usize;

        struct Item {
            id: u32,
            score: f32,
            age: u32,
            /// Vertices of the node still older than the node's required epoch.
            stale: usize,
        }
        let g = self.epoch;
        let frame = self.frame;
        let mut pending: Vec<Item> = Vec::new();
        let mut zero: Vec<u32> = Vec::new();
        let (mut leaf_total, mut leaf_fresh) = (0u64, 0u64);
        let (mut verts_total, mut verts_fresh) = (0u64, 0u64);
        let drag = mode == Mode::Drag;
        for &id in &cut.ids {
            rep.visible_items += 1;
            let iu = id as usize;
            let len = self.node_verts[iu].len();
            let w = self.node_leaves[iu] as u64;
            leaf_total += w;
            verts_total += len as u64;
            if self.done[iu] >= self.need[iu] {
                rep.fresh_before += 1;
                leaf_fresh += w;
                verts_fresh += len as u64;
                continue;
            }
            // A node is stale only if some vertex is older than the node's required epoch.
            let need = self.need[iu];
            let stale = self.node_verts[iu].iter().filter(|&&v| self.vert_epoch[v as usize] < need).count();
            if stale == 0 {
                self.done[iu] = need;
                rep.fresh_before += 1;
                leaf_fresh += w;
                verts_fresh += len as u64;
                continue;
            }
            verts_fresh += (len - stale) as u64;
            rep.max_stale_age = rep.max_stale_age.max(frame - self.last_eval[iu]);
            let b = &mesh.bvh.nodes[iu].bounds;
            if !set.touches(b) {
                zero.push(id);
                continue;
            }
            let dist = (view.eye.clamp(b.min, b.max) - view.eye).length().max(1e-3);
            let size_px = b.diagonal() * view.focal_px / dist;
            let inst_px = set.cell_size() * view.focal_px / dist;
            if drag && size_px < self.cfg.min_node_px {
                rep.skipped_tiny += 1;
                continue;
            }
            if drag && inst_px < self.cfg.min_instance_px {
                rep.skipped_subpixel += 1;
                continue;
            }
            let age = frame - self.last_eval[iu];
            pending.push(Item { id, score: size_px * (1.0 + inst_px.min(32.0) / 8.0) * (1.0 + age as f32 / self.cfg.age_scale), age, stale });
        }
        let mut fresh_now = rep.fresh_before;

        // Nodes no instance can touch: their displacement is zero everywhere.
        for &id in &zero {
            let iu = id as usize;
            let mut was_stale = 0u64;
            let need = self.need[iu];
            for &v in self.node_verts[iu].iter() {
                was_stale += (self.vert_epoch[v as usize] < need) as u64;
                self.out[v as usize] = Vec3::ZERO;
                self.vert_epoch[v as usize] = g;
            }
            self.done[iu] = g;
            self.last_eval[iu] = frame;
            self.updated.push(id);
            fresh_now += 1;
            leaf_fresh += self.node_leaves[iu] as u64;
            verts_fresh += was_stale;
        }
        rep.zero_filled_items = zero.len();

        if self.cfg.fairness_period > 0 && frame.is_multiple_of(self.cfg.fairness_period) {
            pending.sort_unstable_by(|a, b| b.age.cmp(&a.age).then(b.score.total_cmp(&a.score)).then(a.id.cmp(&b.id)));
        } else {
            pending.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        }
        let t_eval = Instant::now();
        let nthreads = rayon::current_num_threads();
        let min_slice = self.cfg.min_slice_verts.max(1);
        let mut i = 0;
        let mut progressed = false;
        while i < pending.len() {
            let remaining = deadline.saturating_duration_since(Instant::now()).as_secs_f64();
            if progressed && remaining <= 0.0 {
                break;
            }
            // Pack slices until the predicted wall time reaches the deadline. A node is split when it does
            // not fit (its stale vertices are taken from a rotating start, so no vertex starves), which keeps
            // one expensive node from blowing the frame.
            let mut slices: Vec<(u32, Vec<u32>)> = Vec::new();
            let (mut predicted, mut verts, mut j) = (0.0f64, 0usize, i);
            while j < pending.len() && slices.len() < self.cfg.max_batch_items {
                let id = pending[j].id as usize;
                let stale = pending[j].stale;
                if stale == 0 {
                    j += 1;
                    continue;
                }
                let per = if self.node_cost[id] > 0.0 { self.node_cost[id] as f64 } else { self.cpu_per_vert };
                let per_wall = per * self.wall_scale / nthreads as f64;
                let fit = if per_wall > 0.0 { ((remaining - predicted) / per_wall).max(0.0) as usize } else { usize::MAX };
                let mut take = stale.min(fit);
                if slices.is_empty() {
                    take = take.max(stale.min(min_slice));
                } else if take < min_slice.min(stale) {
                    break;
                }
                let list = &self.node_verts[id];
                let need = self.need[id];
                let mut picked = Vec::with_capacity(take);
                let mut at = self.cursor[id] as usize % list.len().max(1);
                for _ in 0..list.len() {
                    if picked.len() == take {
                        break;
                    }
                    let v = list[at];
                    if self.vert_epoch[v as usize] < need {
                        picked.push(v);
                    }
                    at += 1;
                    if at == list.len() {
                        at = 0;
                    }
                }
                self.cursor[id] = at as u32;
                predicted += per_wall * picked.len() as f64;
                verts += picked.len();
                // Vertices shared with a node evaluated earlier in this frame are already fresh: if fewer than
                // asked were stale, the node is complete.
                pending[j].stale = if picked.len() < take { 0 } else { pending[j].stale - picked.len() };
                slices.push((id as u32, picked));
                if pending[j].stale > 0 {
                    break;
                }
                j += 1;
                if self.cpu_per_vert == 0.0 && slices.len() >= nthreads {
                    break;
                }
            }
            let t_batch = Instant::now();
            let cpu = eval_slices(&mesh, &set, &self.params, &slices, &mut self.out, &mut self.vert_epoch, g);
            let mut batch_cpu = 0.0;
            for ((id, picked), &c) in slices.iter().zip(&cpu) {
                let id = *id as usize;
                let per = c / picked.len().max(1) as f64;
                let nc = &mut self.node_cost[id];
                *nc = if *nc == 0.0 { per as f32 } else { 0.5 * *nc + 0.5 * per as f32 };
                batch_cpu += c;
                self.last_eval[id] = frame;
                if self.updated.last() != Some(&(id as u32)) {
                    self.updated.push(id as u32);
                }
                verts_fresh += picked.len() as u64;
            }
            while i < pending.len() && pending[i].stale == 0 {
                let id = pending[i].id as usize;
                self.done[id] = g;
                leaf_fresh += self.node_leaves[id] as u64;
                fresh_now += 1;
                rep.evaluated_items += 1;
                i += 1;
            }
            let ideal = batch_cpu / nthreads as f64;
            if ideal > 1e-5 {
                let ratio = (t_batch.elapsed().as_secs_f64() / ideal).clamp(1.0, 6.0);
                self.wall_scale = ratio.max(0.7 * self.wall_scale + 0.3 * ratio);
            }
            let sample = batch_cpu / verts.max(1) as f64;
            self.cpu_per_vert = if self.cpu_per_vert == 0.0 { sample } else { 0.7 * self.cpu_per_vert + 0.3 * sample };
            rep.evaluated_verts += verts;
            progressed = true;
        }
        self.updated.sort_unstable();
        self.updated.dedup();
        rep.deadline_hit = i < pending.len();
        rep.eval_ms = t_eval.elapsed().as_secs_f64() * 1e3;
        rep.fresh_after = fresh_now;
        rep.fresh_leaf_frac = if leaf_total == 0 { 1.0 } else { leaf_fresh as f32 / leaf_total as f32 };
        rep.fresh_vert_frac = if verts_total == 0 { 1.0 } else { (verts_fresh as f64 / verts_total as f64).min(1.0) as f32 };
        if self.job_deferred && self.job.is_none() && rep.fresh_after == rep.visible_items && !self.build_pending() {
            self.start_job();
            rep.job_running = true;
        }
        rep
    }
}

/// Evaluate the listed vertices of each node, scatter the displacements into `out` and stamp them with
/// `epoch`. Returns the CPU seconds spent on each slice.
fn eval_slices(mesh: &EvalMesh, set: &InstanceSet, params: &EvalParams, slices: &[(u32, Vec<u32>)], out: &mut [Vec3], stamp: &mut [u32], epoch: u32) -> Vec<f64> {
    let mut ranges = Vec::with_capacity(slices.len());
    let mut total = 0usize;
    for (_, v) in slices {
        ranges.push(total..total + v.len());
        total += v.len();
    }
    let mut buf = vec![Vec3::ZERO; total];
    let nanos: Vec<AtomicU64> = slices.iter().map(|_| AtomicU64::new(0)).collect();
    split_ranges_mut(&mut buf, ranges).into_par_iter().zip(slices.par_iter().zip(nanos.par_iter())).for_each(|(slice, ((_, verts), ns))| {
        slice.par_chunks_mut(256).enumerate().for_each(|(c, chunk)| {
            let t = Instant::now();
            for (k, o) in chunk.iter_mut().enumerate() {
                let v = verts[c * 256 + k] as usize;
                let p = mesh.positions[v];
                *o = eval_vertex(p, mesh.normals[v], set, params) - p;
            }
            ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        });
    });
    let mut at = 0;
    for (_, verts) in slices {
        for &v in verts {
            out[v as usize] = buf[at];
            stamp[v as usize] = epoch;
            at += 1;
        }
    }
    nanos.iter().map(|n| n.load(Ordering::Relaxed) as f64 * 1e-9).collect()
}

/// Mean normal and the largest angle from it among `verts`.
fn cone_of(verts: &[u32], normals: &[Vec3]) -> (Vec3, f32) {
    let sum: Vec3 = verts.iter().map(|&v| normals[v as usize]).sum();
    let axis = sum.try_normalize().unwrap_or(Vec3::Y);
    let min_dot = verts.iter().map(|&v| normals[v as usize].dot(axis)).fold(1.0f32, f32::min);
    (axis, min_dot.clamp(-1.0, 1.0).acos())
}

fn merge_cones(a: (Vec3, f32), b: (Vec3, f32)) -> (Vec3, f32) {
    if a.1 >= std::f32::consts::PI || b.1 >= std::f32::consts::PI {
        return (a.0, std::f32::consts::PI);
    }
    let axis = (a.0 + b.0).try_normalize().unwrap_or(a.0);
    let ang = |c: (Vec3, f32)| axis.dot(c.0).clamp(-1.0, 1.0).acos() + c.1;
    (axis, ang(a).max(ang(b)).min(std::f32::consts::PI))
}

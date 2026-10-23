//! Generator Boolean layer: spike code.
//!
//! * [`sdf`]: analytic primitive signed distance fields.
//! * [`instances`]: instance transforms and the spatial hash over their AABBs.
//! * [`schedule`]: LOD-aware, time-budgeted progressive evaluation (interactive drag scheduler).
//! * [`place`]: mask-weighted surface sampling and sharp-edge sampling.
//! * [`refine`]: SPIKE 3, adaptive boundary refinement (temporary crisp patches).
//! * [`voxel`]: SPIKE B timing code (sparse brick SDF + surface nets). Measurement
//!   only, not a feature implementation.
//!
//! Spike A (this file): the **displacement boolean**. Each vertex near an
//! instance is moved onto the instance surface, producing a per-vertex delta
//! relative to the input positions, so topology, per-vertex layers, masks and
//! undo are untouched.
//!
//! Per-op rules (all in world space, `p` input position, `n` input normal):
//!
//! * `Subtract`: vertices with `sdf < 0` for any instance are projected to
//!   the cutter surface, `p -= d * grad`, up to `iterations` times, each time
//!   using the most-penetrating instance (so overlapping cutters converge to
//!   the union boundary).
//! * `Intersect`: if `p` is outside every nearby instance but within `reach`
//!   of the closest one, it is pulled in the same way (`p -= d * grad`, d > 0).
//!   Vertices inside any instance, or farther than `reach`, are untouched.
//! * `Add`: a heightfield union along the normal. A ray from `p` along `+n`
//!   (sphere traced on the instance SDF, bisection-refined) is intersected with
//!   the primitive. If `p` is inside the primitive it moves to the exit point (the
//!   primitive's top as seen along `n`). If `p` is outside, but the ray enters
//!   the primitive within `reach`, it moves to the entry point (glues a floating
//!   primitive to the surface). Rays that miss leave the vertex alone. Add can
//!   therefore never make an overhang or undercut; the result is single-valued
//!   along each vertex normal.

pub mod adf;
pub mod density;
pub mod instances;
pub mod layer;
pub mod detail_layer;
pub mod level_view;
pub mod level_volume;
pub mod levels;
pub mod mapping;
pub mod place;
pub mod refine;
pub mod remesh;
pub mod schedule;
pub mod sdf;
pub mod volume;
pub mod voxel;

use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use instances::{Cands, Dirty, HashStats, Instance, InstanceSet, frame_from_edge, frame_from_normal};
pub use sdf::Primitive;

use crate::bvh::{Bvh, NodeKind, split_ranges_mut};
use crate::geom::Aabb;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoolOp {
    Add,
    Subtract,
    Intersect,
}

#[derive(Clone, Copy, Debug)]
pub struct EvalParams {
    pub op: BoolOp,
    /// World-space influence distance for Add (entry glue) and Intersect (pull-in).
    /// Must equal the margin the [`InstanceSet`] was built with (or be smaller).
    pub reach: f32,
    /// Projection iterations for Subtract / Intersect.
    pub iterations: u32,
}

impl Default for EvalParams {
    fn default() -> Self {
        EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 3 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalStats {
    pub leaves_evaluated: usize,
    pub leaves_skipped: usize,
    pub vertices_moved: usize,
}

/// Evaluate one vertex. Returns the new position.
#[inline]
pub fn eval_vertex(p: Vec3, n: Vec3, set: &InstanceSet, params: &EvalParams) -> Vec3 {
    let cands = set.candidates(p);
    if cands.is_empty() {
        return p;
    }
    match params.op {
        BoolOp::Subtract => {
            // Pass 1 at p: find the most-penetrating instance and remember every
            // candidate that could still matter. SDFs are 1-Lipschitz and the
            // total motion is at most `iterations * |d1|`, so an instance with
            // d(p) beyond that can never contain a later iterate.
            let mut near: smallvec::SmallVec<[(u32, f32); 24]> = smallvec::SmallVec::new();
            let mut best = (0.0f32, u32::MAX);
            for &i in cands {
                if !set.inflated_aabb_contains(i, p) {
                    continue;
                }
                let d = set.sdf(i, p);
                if d < best.0 {
                    best = (d, i);
                }
                near.push((i, d));
            }
            if best.1 == u32::MAX {
                return p;
            }
            let horizon = params.iterations as f32 * -best.0;
            let mut q = p - set.sdf_grad(best.1, p).1 * best.0;
            for _ in 1..params.iterations {
                let mut b = (0.0f32, u32::MAX);
                for &(i, d0) in &near {
                    if d0 > horizon {
                        continue;
                    }
                    let d = set.sdf(i, q);
                    if d < b.0 {
                        b = (d, i);
                    }
                }
                if b.1 == u32::MAX {
                    break;
                }
                q -= set.sdf_grad(b.1, q).1 * b.0;
            }
            q
        }
        BoolOp::Intersect => {
            let mut q = p;
            for _ in 0..params.iterations {
                let mut best = (f32::INFINITY, Vec3::ZERO);
                for &i in cands {
                    if !set.inflated_aabb_contains(i, q) {
                        continue;
                    }
                    let (d, g) = set.sdf_grad(i, q);
                    if d < best.0 {
                        best = (d, g);
                    }
                }
                // Inside some instance, or nothing in reach: nothing to do.
                if best.0 <= 0.0 || best.0 > params.reach {
                    break;
                }
                q -= best.1 * best.0;
            }
            q
        }
        BoolOp::Add => {
            let mut best_t = f32::NEG_INFINITY;
            for &i in cands {
                if !set.inflated_aabb_contains(i, p) {
                    continue;
                }
                if let Some(t) = ray_surface(set, i, p, n, params.reach) {
                    // Inside -> exit (largest t wins); glued -> entry (kept only if nothing taller).
                    best_t = best_t.max(t);
                }
            }
            if best_t.is_finite() { p + n * best_t } else { p }
        }
    }
}

/// Distance along `n` from `p` to the surface the Add rule pulls to, or None.
fn ray_surface(set: &InstanceSet, i: u32, p: Vec3, n: Vec3, reach: f32) -> Option<f32> {
    let scale = set.min_scale(i);
    let eps = 1e-4 * scale;
    let mut t = 0.0f32;
    let mut d = set.sdf(i, p);
    if d > eps {
        // Outside: sphere trace toward the primitive, give up past `reach`.
        let mut hit = false;
        for _ in 0..32 {
            t += d;
            if t > reach {
                return None;
            }
            d = set.sdf(i, p + n * t);
            if d <= eps {
                hit = true;
                break;
            }
        }
        return hit.then_some(t);
    }
    // Inside (or on the surface): march out, then bisect the crossing.
    let step_min = 0.05 * scale;
    let mut b = t;
    let mut a;
    for _ in 0..64 {
        a = b;
        b += (-d).max(step_min);
        d = set.sdf(i, p + n * b);
        if d > 0.0 {
            for _ in 0..8 {
                let m = 0.5 * (a + b);
                if set.sdf(i, p + n * m) > 0.0 {
                    b = m;
                } else {
                    a = m;
                }
            }
            return Some(0.5 * (a + b));
        }
    }
    None
}

/// Evaluate every leaf. `out` has one entry per vertex (internal order).
pub fn evaluate(positions: &[Vec3], normals: &[Vec3], bvh: &Bvh, set: &InstanceSet, params: &EvalParams, out: &mut [Vec3]) -> EvalStats {
    let all: Vec<u32> = (0..bvh.leaves.len() as u32).collect();
    evaluate_leaves(positions, normals, bvh, set, params, &all, out)
}

/// Evaluate the given (ascending, unique) leaves, overwriting their vertex range in `out`.
pub fn evaluate_leaves(
    positions: &[Vec3],
    normals: &[Vec3],
    bvh: &Bvh,
    set: &InstanceSet,
    params: &EvalParams,
    leaves: &[u32],
    out: &mut [Vec3],
) -> EvalStats {
    let ranges = leaves.iter().map(|&l| bvh.leaves[l as usize].owned_range());
    let slices = split_ranges_mut(out, ranges);
    let (evaluated, moved) = slices
        .into_par_iter()
        .zip(leaves.par_iter())
        .map(|(slice, &li)| {
            let leaf = &bvh.leaves[li as usize];
            if !set.touches(&leaf.bounds) {
                slice.fill(Vec3::ZERO);
                return (0usize, 0usize);
            }
            let base = leaf.owned.start as usize;
            let mut moved = 0;
            for (k, o) in slice.iter_mut().enumerate() {
                let v = base + k;
                let p = positions[v];
                let q = eval_vertex(p, normals[v], set, params);
                *o = q - p;
                moved += (*o != Vec3::ZERO) as usize;
            }
            (1, moved)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    EvalStats { leaves_evaluated: evaluated, leaves_skipped: leaves.len() - evaluated, vertices_moved: moved }
}

/// Leaves whose bounds overlap any of the boxes (sorted, unique).
pub fn affected_leaves(bvh: &Bvh, boxes: &[Aabb]) -> Vec<u32> {
    let mut out: Vec<u32> = boxes
        .par_iter()
        .flat_map_iter(|b| {
            let mut found = Vec::new();
            let mut stack = vec![0u32];
            while let Some(n) = stack.pop() {
                let Some(node) = bvh.nodes.get(n as usize) else { break };
                let nb = &node.bounds;
                if nb.is_empty() || nb.min.cmpgt(b.max).any() || nb.max.cmplt(b.min).any() {
                    continue;
                }
                match node.kind {
                    NodeKind::Leaf { leaf } => {
                        let lb = &bvh.leaves[leaf as usize].bounds;
                        if !(lb.min.cmpgt(b.max).any() || lb.max.cmplt(b.min).any()) {
                            found.push(leaf);
                        }
                    }
                    NodeKind::Inner { left, right } => {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            }
            found
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

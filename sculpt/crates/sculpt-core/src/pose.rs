//! ZBrush-style posing: topological masking + weighted transforms.
//!
//! Workflow (Transpose / Gizmo with topological mask):
//! 1. pick a vertex on the part to move, get weights by geodesic distance
//!    ([`Document::topological_weights`]); optionally soften with blur;
//! 2. rotate / translate / scale around a pivot with those weights.
//!
//! Poses transform the base mesh *and rotate every layer's deltas* by the
//! same per-vertex rotation, so sculpted detail on layers follows the limb
//! instead of shearing off.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::{Quat, Vec3};
use rayon::prelude::*;

use crate::document::Document;
use crate::geom::smoothstep;
use crate::mask::blur;
use crate::{Error, Result, undo};

#[derive(PartialEq)]
struct Item(f32, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0) // min-heap
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PoseTransform {
    Rotate { pivot: Vec3, axis: Vec3, angle: f32 },
    Translate { offset: Vec3 },
    Scale { pivot: Vec3, factor: f32 },
}

impl Document {
    /// Dijkstra over mesh edges from `seed`, up to `max_distance`.
    pub fn geodesic_distances(&self, seed: u32, max_distance: f32) -> Vec<(u32, f32)> {
        let mut dist: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        let mut heap = BinaryHeap::new();
        dist.insert(seed, 0.0);
        heap.push(Item(0.0, seed));
        while let Some(Item(d, v)) = heap.pop() {
            if d > dist[&v] {
                continue;
            }
            let p = self.positions[v as usize];
            for &u in self.topo.vert_verts.row(v as usize) {
                let nd = d + p.distance(self.positions[u as usize]);
                if nd <= max_distance && dist.get(&u).is_none_or(|&old| nd < old) {
                    dist.insert(u, nd);
                    heap.push(Item(nd, u));
                }
            }
        }
        dist.into_iter().collect()
    }

    /// Movable weights for posing: 1 within `radius` (geodesic) of `seed`,
    /// fading to 0 over the outer `softness` fraction, then blurred.
    pub fn topological_weights(&self, seed: u32, radius: f32, softness: f32, blur_iterations: u32) -> Vec<f32> {
        let mut w = vec![0.0; self.vertex_count()];
        let inner = radius * (1.0 - softness.clamp(0.0, 1.0));
        for (v, d) in self.geodesic_distances(seed, radius) {
            w[v as usize] = 1.0 - smoothstep(inner, radius.max(inner + 1e-6), d);
        }
        if blur_iterations > 0 { blur(&w, &self.topo, blur_iterations) } else { w }
    }

    /// Apply a weighted pose transform (one undo step). Freeze is respected.
    pub fn pose(&mut self, weights: &[f32], xf: PoseTransform) -> Result<()> {
        if weights.len() != self.vertex_count() {
            return Err(Error::InvalidData("pose weights size mismatch".into()));
        }
        self.begin_stroke("Pose");
        let leaves = self.bvh.leaves.len() as u32;
        let layer_ids: Vec<_> = self.layers.iter().map(|l| l.id).collect();
        for l in 0..leaves {
            self.snapshot(&undo::Target::Base, l);
            for &id in &layer_ids {
                self.snapshot(&undo::Target::Layer(id), l);
            }
        }

        let freeze = &self.freeze;
        let w_at = |v: usize| (weights[v] * (1.0 - freeze[v])).clamp(0.0, 1.0);
        let rot_at = |w: f32| -> (Quat, Vec3, f32) {
            match xf {
                PoseTransform::Rotate { pivot, axis, angle } => (Quat::from_axis_angle(axis.normalize_or(Vec3::Y), angle * w), pivot, 1.0),
                PoseTransform::Translate { .. } => (Quat::IDENTITY, Vec3::ZERO, 1.0),
                PoseTransform::Scale { pivot, factor } => (Quat::IDENTITY, pivot, 1.0 + (factor - 1.0) * w),
            }
        };
        self.base.par_iter_mut().enumerate().for_each(|(v, p)| {
            let w = w_at(v);
            if w <= 0.0 {
                return;
            }
            let (q, pivot, s) = rot_at(w);
            *p = pivot + q * ((*p - pivot) * s);
            if let PoseTransform::Translate { offset } = xf {
                *p += offset * w;
            }
        });
        let bvh_leaves = &self.bvh.leaves;
        for layer in &mut self.layers {
            layer.chunks.par_iter_mut().enumerate().for_each(|(l, c)| {
                let Some(chunk) = c else { return };
                let start = bvh_leaves[l].owned.start as usize;
                for (k, d) in chunk.iter_mut().enumerate() {
                    let w = w_at(start + k);
                    if w > 0.0 {
                        let (q, _, s) = rot_at(w);
                        *d = q * (*d * s);
                    }
                }
            });
        }
        self.end_stroke();
        self.recomposite_all();
        Ok(())
    }
}

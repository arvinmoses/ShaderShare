//! Mesh-derived mask attributes: curvature, cavity, ambient occlusion, thickness.
//!
//! Bakes are explicit (like Substance's "bake mesh maps") and land in named
//! per-vertex channels, so mask stacks reference them like any other data.

use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bvh::Bvh;
use crate::geom::Ray;
use crate::mask::blur;
use crate::mesh::{Face, Topology};
use crate::noise::hash3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshAttribute {
    /// 0.5 = flat, > 0.5 convex, < 0.5 concave.
    Curvature,
    /// 1 in crevices, 0 elsewhere.
    Cavity,
    /// 1 = fully open, 0 = fully occluded.
    AmbientOcclusion,
    /// 0 = thin, 1 = thick (relative to `thickness_distance`).
    Thickness,
}

impl MeshAttribute {
    pub const ALL: [MeshAttribute; 4] =
        [MeshAttribute::Curvature, MeshAttribute::Cavity, MeshAttribute::AmbientOcclusion, MeshAttribute::Thickness];

    pub fn channel_name(self) -> &'static str {
        match self {
            MeshAttribute::Curvature => "mesh.curvature",
            MeshAttribute::Cavity => "mesh.cavity",
            MeshAttribute::AmbientOcclusion => "mesh.ao",
            MeshAttribute::Thickness => "mesh.thickness",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BakeSettings {
    /// Contrast applied after percentile normalization.
    pub curvature_contrast: f32,
    /// Blur iterations: larger values pick up broader forms.
    pub curvature_blur: u32,
    pub ao_rays: u32,
    /// As a fraction of the mesh bounding-box diagonal.
    pub ao_distance: f32,
    pub thickness_rays: u32,
    pub thickness_distance: f32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        BakeSettings {
            curvature_contrast: 1.0,
            curvature_blur: 2,
            ao_rays: 32,
            ao_distance: 0.25,
            thickness_rays: 16,
            thickness_distance: 0.25,
        }
    }
}

pub struct BakeInput<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub faces: &'a [Face],
    pub topology: &'a Topology,
    pub bvh: &'a Bvh,
}

/// Signed, scale-normalized mean curvature mapped to `[0, 1]`.
pub fn curvature(input: &BakeInput, s: &BakeSettings) -> Vec<f32> {
    let raw: Vec<f32> = (0..input.positions.len())
        .into_par_iter()
        .map(|v| {
            let ring = input.topology.vert_verts.row(v);
            if ring.is_empty() {
                return 0.0;
            }
            let p = input.positions[v];
            let mut avg = Vec3::ZERO;
            let mut len = 0.0;
            for &u in ring {
                let q = input.positions[u as usize];
                avg += q;
                len += q.distance(p);
            }
            avg /= ring.len() as f32;
            len /= ring.len() as f32;
            if len <= 0.0 { 0.0 } else { input.normals[v].dot(p - avg) / len }
        })
        .collect();
    let raw = blur(&raw, input.topology, s.curvature_blur);

    // Normalize by the 95th percentile so the result adapts to mesh density.
    let mut mags: Vec<f32> = raw.iter().map(|c| c.abs()).collect();
    let k = ((mags.len() as f32 * 0.95) as usize).min(mags.len().saturating_sub(1));
    let scale = if mags.is_empty() { 1.0 } else { *mags.select_nth_unstable_by(k, f32::total_cmp).1 };
    let scale = if scale > 1e-12 { s.curvature_contrast / scale } else { 0.0 };
    raw.par_iter().map(|c| (0.5 + 0.5 * (c * scale)).clamp(0.0, 1.0)).collect()
}

pub fn cavity_from_curvature(curv: &[f32]) -> Vec<f32> {
    curv.par_iter().map(|c| ((0.5 - c) * 2.0).clamp(0.0, 1.0)).collect()
}

fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let t = n.any_orthonormal_vector();
    (t, n.cross(t))
}

/// Cosine-weighted, per-vertex rotated Fibonacci hemisphere.
#[inline]
fn hemisphere_dir(n: Vec3, i: u32, count: u32, rot: f32) -> Vec3 {
    let (t, b) = tangent_frame(n);
    let u = (i as f32 + 0.5) / count as f32;
    let r = u.sqrt();
    let phi = std::f32::consts::TAU * (i as f32 * 0.618_034 + rot);
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - u).sqrt()).normalize()
}

pub fn ambient_occlusion(input: &BakeInput, s: &BakeSettings) -> Vec<f32> {
    let diag = input.bvh.bounds().diagonal();
    let dist = s.ao_distance * diag;
    let eps = 1e-4 * diag;
    let rays = s.ao_rays.max(1);
    (0..input.positions.len())
        .into_par_iter()
        .map(|v| {
            let n = input.normals[v];
            let o = input.positions[v] + n * eps;
            let rot = (hash3(v as i32, 7, 13, 1) & 0xffff) as f32 / 65535.0;
            let mut hits = 0;
            for i in 0..rays {
                let ray = Ray { origin: o, dir: hemisphere_dir(n, i, rays, rot) };
                if input.bvh.occluded(input.positions, input.faces, &ray, dist) {
                    hits += 1;
                }
            }
            1.0 - hits as f32 / rays as f32
        })
        .collect()
}

pub fn thickness(input: &BakeInput, s: &BakeSettings) -> Vec<f32> {
    let diag = input.bvh.bounds().diagonal();
    let dist = s.thickness_distance * diag;
    let eps = 1e-4 * diag;
    let rays = s.thickness_rays.max(1);
    (0..input.positions.len())
        .into_par_iter()
        .map(|v| {
            let n = -input.normals[v];
            let o = input.positions[v] + n * eps;
            let rot = (hash3(v as i32, 3, 5, 2) & 0xffff) as f32 / 65535.0;
            let mut total = 0.0;
            for i in 0..rays {
                let ray = Ray { origin: o, dir: hemisphere_dir(n, i, rays, rot) };
                total += input.bvh.raycast(input.positions, input.faces, &ray, dist).map_or(dist, |h| h.t);
            }
            (total / rays as f32 / dist).clamp(0.0, 1.0)
        })
        .collect()
}

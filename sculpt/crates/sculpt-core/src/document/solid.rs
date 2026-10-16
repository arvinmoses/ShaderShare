//! Solidify: turn the layer stack into one clean quad mesh through a signed distance field.
//!
//! The live stack moves vertices, which is fast but can fold the surface and never changes topology. Solidify treats the
//! same stack as a solid instead. The sculpted surface (base plus plain layers) is the starting field, and each
//! generator layer, bottom to top, combines its instances with it: subtract cuts, add joins, intersect keeps the overlap,
//! each with a smooth blend. One manifold extraction at the end gives a mesh with no inside geometry and no folds.
//!
//! A generator layer scatters on the surface *below* it, so each layer after the first gets a coarse extraction of the
//! field so far to scatter on. That is what lets small chips land on the faces a big cut made.

use std::time::Instant;

use glam::Vec3;

use super::Document;
use crate::generator::layer::GeneratorSpec;
use crate::generator::remesh::{RemeshParams, remesh};
use crate::generator::voxel::MeshSdf;
use crate::generator::{BoolOp, InstanceSet};
use crate::geom::Aabb;
use crate::layer_ops::Placement;
use crate::mesh::PolyMesh;
use crate::noise::{Noise, NoiseKind, NoiseParams};
use crate::{Error, Result};

/// Fine noise added to the field before extraction, until noise is a layer of its own.
#[derive(Clone, Copy, Debug)]
pub struct Grain {
    /// Peak displacement in world units (the field moves by up to half of it either way).
    pub amplitude: f32,
    /// Cycles per world unit.
    pub scale: f32,
    pub seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SolidSettings {
    /// Lattice spacing of the final extraction, in world units.
    pub voxel: f32,
    pub grain: Option<Grain>,
}

#[derive(Clone, Debug, Default)]
pub struct SolidStats {
    pub layers: usize,
    pub quads: usize,
    pub scatter_ms: f64,
    pub extract_ms: f64,
    pub evaluated: usize,
    pub total: usize,
}

/// One generator layer ready to be evaluated as part of the field.
struct Term {
    set: InstanceSet,
    op: BoolOp,
    /// Blend width in world units: the layer's `blend` times its typical instance size.
    blend: f32,
}

/// Smooth maximum: `max(a, b)` blended over a band of width `k` (polynomial). `k = 0` is a plain maximum.
fn smax(a: f32, b: f32, k: f32) -> f32 {
    if k <= 1e-9 {
        return a.max(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Distance to the nearest instance of `set` near `p` (negative inside one); `INFINITY` if none is within `limit`.
/// An instance whose box is farther than `limit` cannot change the field at `p`, so it is skipped without evaluating it.
fn nearest(set: &InstanceSet, p: Vec3, limit: f32) -> f32 {
    let mut best = f32::INFINITY;
    for &i in set.candidates(p) {
        let b = set.instances()[i as usize].aabb();
        if (p - p.clamp(b.min, b.max)).length() > limit {
            continue;
        }
        best = best.min(set.sdf(i, p));
    }
    best
}

fn apply_terms(mut f: f32, terms: &[Term], p: Vec3) -> f32 {
    for t in terms {
        // Only an instance within |f| + blend of p can move the field there (further away it changes the value, never the sign).
        let d = nearest(&t.set, p, f.abs() + t.blend);
        if !d.is_finite() {
            continue;
        }
        f = match t.op {
            BoolOp::Subtract => smax(f, -d, t.blend),
            BoolOp::Add => -smax(-f, -d, t.blend),
            BoolOp::Intersect => smax(f, d, t.blend),
        };
    }
    f
}

impl Document {
    /// The surface of the base plus plain sculpt layers, leaving generator layers out.
    fn plain_positions(&self) -> Vec<Vec3> {
        let order = self.composite_order();
        let eps = (self.bvh.bounds().extent().length() * 1e-4).max(1e-9);
        let mut out = self.base.clone();
        for &i in &order {
            let layer = &self.layers[i];
            if layer.gen_cache.is_some() {
                continue;
            }
            for (l, chunk) in layer.chunks.iter().enumerate() {
                let Some(chunk) = chunk else { continue };
                let r = self.bvh.leaves[l].owned_range();
                for (k, v) in r.clone().enumerate() {
                    let w = layer.weight(v);
                    if layer.blend == crate::layers::LayerBlend::Add {
                        out[v] += chunk[k] * w;
                    } else {
                        let acc = out[v] - self.base[v];
                        out[v] = self.base[v] + layer.blend.apply(acc, chunk[k], w, self.normals[v], eps);
                    }
                }
            }
        }
        out
    }

    /// Solidify the stack into a clean quad mesh. The document is not changed; the caller decides what to do with the
    /// result (start a new document from it, which is what a Remesh checkpoint does).
    pub fn solidify(&self, settings: &SolidSettings) -> Result<(PolyMesh, SolidStats)> {
        let mut stats = SolidStats::default();
        let plain = Document::from_mesh(PolyMesh { positions: self.plain_positions(), faces: self.faces.clone() })?;
        let sdf = MeshSdf::build(plain.bvh(), plain.positions(), plain.faces(), plain.topology());
        let diagonal = plain.bounds().extent().length().max(1e-3);
        let bound = diagonal * 2.0;

        // Generator layers that contribute, bottom to top.
        let specs: Vec<GeneratorSpec> = self
            .composite_order()
            .into_iter()
            .filter(|&i| self.layers[i].gen_cache.is_some() && self.layers[i].scale > 0.0)
            .filter_map(|i| self.layers[i].generator.clone())
            .collect();
        stats.layers = specs.len();
        let reach = specs.iter().map(|s| s.size * (1.0 + s.size_var) * (1.0 + s.stretch_var) * 1.8 + s.blend * s.size).fold(0.0, f32::max);
        let pad = reach + 4.0 * settings.voxel;
        let bounds = Aabb { min: plain.bounds().min - Vec3::splat(pad), max: plain.bounds().max + Vec3::splat(pad) };
        // Blocks are skipped when their centre is this far from the surface, so the cutters must be findable that far out.
        let margin = 8.0 * settings.voxel * 0.87 * 1.5;
        let coarse_voxel = (bounds.extent().max_element() / 48.0).max(settings.voxel);

        let grain = settings.grain.map(|g| (Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: g.scale, octaves: 4, seed: g.seed, ..Default::default() }), g.amplitude));
        let base_field = |p: Vec3| sdf.signed_distance(p, bound);

        let mut terms: Vec<Term> = Vec::new();
        let t = Instant::now();
        for spec in &specs {
            // The surface so far is what this layer scatters on: the sculpted surface for the first, a coarse extraction after.
            let mut surface = if terms.is_empty() {
                Document::from_mesh(PolyMesh { positions: plain.positions().to_vec(), faces: plain.faces().to_vec() })?
            } else {
                let coarse = remesh(&|p: Vec3| apply_terms(base_field(p), &terms, p), bounds, &RemeshParams { voxel: coarse_voxel, lipschitz: 1.5 });
                if coarse.quads.is_empty() {
                    return Err(Error::InvalidData("the stack leaves nothing to scatter on".into()));
                }
                Document::from_mesh(coarse.into_poly_mesh())?
            };
            let id = surface.insert_generator("solid", Placement::Top, spec.clone())?;
            let instances = surface.layer(id).and_then(|l| l.generator_cache()).map(|c| c.instances().to_vec()).unwrap_or_default();
            if instances.is_empty() {
                continue;
            }
            let mean_size = instances.iter().map(|i| i.scale.max_element()).sum::<f32>() / instances.len() as f32;
            let blend = spec.blend * mean_size;
            terms.push(Term { set: InstanceSet::new(instances, margin + blend), op: spec.op, blend });
        }
        stats.scatter_ms = t.elapsed().as_secs_f64() * 1e3;

        let field = |p: Vec3| {
            let f = apply_terms(base_field(p), &terms, p);
            match &grain {
                Some((noise, amplitude)) => f + (noise.sample(p) - 0.5) * amplitude,
                None => f,
            }
        };
        let t = Instant::now();
        // Gradient above 1 where blends overlap and grain is added, so skip blocks conservatively.
        let lipschitz = 1.3 + grain.as_ref().map_or(0.0, |(_, a)| a * 2.0);
        let out = remesh(&field, bounds, &RemeshParams { voxel: settings.voxel, lipschitz });
        stats.extract_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.quads = out.quads.len();
        stats.evaluated = out.evaluated;
        stats.total = out.total;
        Ok((out.into_poly_mesh(), stats))
    }
}

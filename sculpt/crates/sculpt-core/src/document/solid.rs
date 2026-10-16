//! Solidify: turn the layer stack into one clean quad mesh through a signed distance field.
//!
//! The live stack moves vertices, which is fast but can fold the surface and never changes topology. Solidify treats the
//! same stack as a solid instead. The sculpted surface (base plus plain layers) is the starting field, and each
//! generator layer, bottom to top, combines its instances with it: subtract cuts, add joins, intersect keeps the overlap,
//! each with a smooth blend. One manifold extraction at the end gives a mesh with no inside geometry and no folds.
//!
//! A generator layer scatters on the surface *below* it, so each layer after the first gets a coarse extraction of the
//! field so far to scatter on. That is what lets small chips land on the faces a big cut made.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;

use super::Document;
use crate::generator::layer::{GeneratorKind, GeneratorSpec};
use crate::generator::remesh::{RemeshParams, remesh};
use crate::generator::voxel::MeshSdf;
use crate::generator::{BoolOp, InstanceSet};
use crate::geom::Aabb;
use crate::layer_ops::Placement;
use crate::mesh::PolyMesh;
use crate::noise::Noise;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SolidSettings {
    /// Lattice spacing of the final extraction, in world units.
    pub voxel: f32,
}

/// Hashes a key that is already well mixed.
#[derive(Default)]
struct Mixed(u64);

impl Hasher for Mixed {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 ^= self.0 >> 32;
    }
}

const SHARDS: usize = 64;
/// Entries past which the cache starts over, so a long session cannot grow it without bound.
const CACHE_LIMIT: usize = 12_000_000;

/// Distances from the sculpted surface, remembered by lattice point between solidify calls. They depend only on that
/// surface and the lattice spacing, not on the generator layers, so editing a cutter reuses all of them. The lattice is
/// aligned to multiples of the spacing in world space, so a changed margin around the shape still lands on the same points.
pub(crate) struct BaseCache {
    surface: u64,
    voxel_bits: u32,
    shards: Vec<Mutex<HashMap<u64, f32, BuildHasherDefault<Mixed>>>>,
    entries: AtomicUsize,
}

impl BaseCache {
    fn new(surface: u64, voxel: f32) -> BaseCache {
        BaseCache { surface, voxel_bits: voxel.to_bits(), shards: (0..SHARDS).map(|_| Mutex::default()).collect(), entries: AtomicUsize::new(0) }
    }

    fn key(p: Vec3, voxel: f32) -> u64 {
        let q = |v: f32| ((v / voxel).round() as i64 & 0x1F_FFFF) as u64;
        q(p.x) | q(p.y) << 21 | q(p.z) << 42
    }

    /// The cached value at `p`, or `compute()` stored for next time. Returns whether it was a hit.
    fn get(&self, p: Vec3, voxel: f32, compute: impl FnOnce() -> f32) -> (f32, bool) {
        let key = Self::key(p, voxel);
        let shard = &self.shards[(key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize % SHARDS];
        if let Some(&v) = shard.lock().unwrap().get(&key) {
            return (v, true);
        }
        let v = compute();
        shard.lock().unwrap().insert(key, v);
        self.entries.fetch_add(1, Ordering::Relaxed);
        (v, false)
    }
}

/// A cheap fingerprint of the sculpted surface, to know when cached distances are stale.
fn fingerprint(positions: &[Vec3]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ positions.len() as u64;
    for p in positions {
        for v in p.to_array() {
            h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

#[derive(Clone, Debug, Default)]
pub struct SolidStats {
    /// Distances to the sculpted surface taken from the cache, and computed.
    pub base_hits: usize,
    pub base_misses: usize,
    pub layers: usize,
    pub quads: usize,
    pub scatter_ms: f64,
    pub extract_ms: f64,
    pub evaluated: usize,
    pub total: usize,
}

/// One generator layer ready to be evaluated as part of the field.
enum Term {
    Instances {
        set: InstanceSet,
        op: BoolOp,
        /// Blend width in world units: the layer's `blend` times its typical instance size.
        blend: f32,
    },
    /// Noise along the normals: the field moves by up to half the amplitude either way.
    Noise { noise: Noise, amplitude: f32 },
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
        match t {
            Term::Noise { noise, amplitude } => f += (noise.sample(p) - 0.5) * amplitude,
            Term::Instances { set, op, blend } => {
                // Only an instance within |f| + blend of p can move the field there (further away it changes the value, never the sign).
                let d = nearest(set, p, f.abs() + blend);
                if !d.is_finite() {
                    continue;
                }
                f = match op {
                    BoolOp::Subtract => smax(f, -d, *blend),
                    BoolOp::Add => -smax(-f, -d, *blend),
                    BoolOp::Intersect => smax(f, d, *blend),
                };
            }
        }
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
        let scatter = |s: &&GeneratorSpec| s.kind == GeneratorKind::Scatter;
        let reach = specs.iter().filter(scatter).map(|s| s.size * (1.0 + s.size_var) * (1.0 + s.stretch_var) * 1.8 + s.blend * s.size).fold(0.0, f32::max);
        // Noise moves the surface by up to its amplitude and steepens the field by about amplitude x scale.
        let noise_pad: f32 = specs.iter().filter(|s| s.kind == GeneratorKind::Noise).map(|s| s.noise.amplitude).sum();
        let noise_slope: f32 = specs.iter().filter(|s| s.kind == GeneratorKind::Noise).map(|s| s.noise.amplitude * s.noise.scale).sum();
        let pad = reach + noise_pad + 4.0 * settings.voxel;
        let snap = |v: Vec3| (v / settings.voxel).floor() * settings.voxel;
        let bounds = Aabb { min: snap(plain.bounds().min - Vec3::splat(pad)), max: plain.bounds().max + Vec3::splat(pad) };
        // Blocks are skipped when their centre is this far from the surface, so the cutters must be findable that far out.
        let margin = 8.0 * settings.voxel * 0.87 * 1.5;
        let coarse_voxel = (bounds.extent().max_element() / 48.0).max(settings.voxel);

        let base_field = |p: Vec3| sdf.signed_distance(p, bound);
        // The final extraction reuses base distances from earlier calls when the surface and spacing are unchanged.
        let cache = {
            let surface = fingerprint(plain.positions());
            let mut slot = self.solid_cache.lock().unwrap();
            match slot.as_ref() {
                Some(c) if c.surface == surface && c.voxel_bits == settings.voxel.to_bits() && c.entries.load(Ordering::Relaxed) < CACHE_LIMIT => c.clone(),
                _ => {
                    let fresh = Arc::new(BaseCache::new(surface, settings.voxel));
                    *slot = Some(fresh.clone());
                    fresh
                }
            }
        };
        let (hits, misses) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let cached_base = |p: Vec3| {
            let (v, hit) = cache.get(p, settings.voxel, || base_field(p));
            if hit { hits.fetch_add(1, Ordering::Relaxed) } else { misses.fetch_add(1, Ordering::Relaxed) };
            v
        };

        let mut terms: Vec<Term> = Vec::new();
        let t = Instant::now();
        for spec in &specs {
            if spec.kind == GeneratorKind::Noise {
                terms.push(Term::Noise { noise: spec.noise.build(), amplitude: spec.noise.amplitude });
                continue;
            }
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
            terms.push(Term::Instances { set: InstanceSet::new(instances, margin + blend), op: spec.op, blend });
        }
        stats.scatter_ms = t.elapsed().as_secs_f64() * 1e3;

        let field = |p: Vec3| apply_terms(cached_base(p), &terms, p);
        let t = Instant::now();
        // Gradient above 1 where blends overlap and grain is added, so skip blocks conservatively.
        let lipschitz = 1.3 + noise_slope;
        let out = remesh(&field, bounds, &RemeshParams { voxel: settings.voxel, lipschitz });
        stats.extract_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.base_hits = hits.load(Ordering::Relaxed);
        stats.base_misses = misses.load(Ordering::Relaxed);
        stats.quads = out.quads.len();
        stats.evaluated = out.evaluated;
        stats.total = out.total;
        Ok((out.into_poly_mesh(), stats))
    }
}

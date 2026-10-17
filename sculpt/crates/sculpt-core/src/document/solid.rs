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
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;

use super::Document;
use crate::generator::layer::{GeneratorKind, GeneratorSpec};
use crate::generator::remesh::{RemeshParams, remesh};
use crate::generator::voxel::MeshSdf;
use crate::generator::{BoolOp, Instance, InstanceSet};
use crate::geom::Aabb;
use crate::mesh::PolyMesh;
use crate::noise::Noise;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SolidSettings {
    /// Lattice spacing of the final extraction, in world units.
    pub voxel: f32,
    /// Loose pieces with fewer than this fraction of the surface are dropped (crossing cuts can slice slivers right
    /// off). `0` keeps everything.
    pub min_piece: f32,
}

impl SolidSettings {
    pub fn new(voxel: f32) -> SolidSettings {
        SolidSettings { voxel, min_piece: 0.0025 }
    }
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

    /// The lattice point a key was made from.
    fn point(key: u64, voxel: f32) -> Vec3 {
        // Each coordinate is 21 bits, two's complement.
        let d = |bits: u64| (((bits & 0x1F_FFFF) as i64) << 43 >> 43) as f32 * voxel;
        Vec3::new(d(key), d(key >> 21), d(key >> 42))
    }

    /// A copy for `surface`, which differs from this cache's surface only inside `changed`. A distance stays exact if
    /// it is shorter than the way to the changed region: the closest point it measured lies outside the region, and
    /// nothing in the region can have come closer than the region itself. Everything else is dropped.
    fn carried_over(&self, surface: u64, changed: &Aabb) -> BaseCache {
        let voxel = f32::from_bits(self.voxel_bits);
        let out = BaseCache::new(surface, voxel);
        let mut kept = 0;
        for (from, to) in self.shards.iter().zip(&out.shards) {
            let from = from.lock().unwrap();
            let mut to = to.lock().unwrap();
            for (&key, &v) in from.iter() {
                let p = BaseCache::point(key, voxel);
                let gap = (p - p.clamp(changed.min, changed.max)).length();
                if v.abs() < gap {
                    to.insert(key, v);
                    kept += 1;
                }
            }
        }
        out.entries.store(kept, Ordering::Relaxed);
        out
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

/// What solidify keeps between calls, shared by the document and every snapshot of it. Each part is keyed by what it
/// depends on, so a slider drag on one layer reuses everything the drag does not touch.
#[derive(Default)]
pub(crate) struct SolidCaches {
    /// Base distances, one per lattice spacing (the drag preview's and the full-detail one); two at most.
    bases: Vec<Arc<BaseCache>>,
    /// The sculpted surface as a document, with its distance structure.
    plain: Option<Arc<PlainSurface>>,
    /// Instances each generator layer scattered, keyed by the layer and everything below it.
    scatters: HashMap<u64, Arc<Vec<Instance>>>,
}

/// The sculpted surface, ready to measure distances to and to scatter the first layer on.
pub(crate) struct PlainSurface {
    surface: u64,
    /// The surface's vertices in the snapshot's order, to find what the next stroke changed.
    positions: Vec<Vec3>,
    sdf: MeshSdf<'static>,
    bounds: Aabb,
    doc: Mutex<Document>,
}

/// Scatter results kept at most; past this the table starts over.
const SCATTER_LIMIT: usize = 64;

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
    /// Loose pieces too small to keep.
    pub pieces_dropped: usize,
    pub base_hits: usize,
    pub base_misses: usize,
    pub layers: usize,
    pub quads: usize,
    /// Building the sculpted surface and its distance structure.
    pub prep_ms: f64,
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

/// Distance to the nearest instance of `set` near `p` (negative inside one), or `None` if none is within `limit`, with a
/// lower bound on that distance. An instance whose box is farther than `limit` cannot change a cut or a join at `p`, so
/// it is not evaluated; an instance the spatial table does not list for `p` is at least the table's margin away.
fn nearest(set: &InstanceSet, p: Vec3, limit: f32) -> (Option<f32>, f32) {
    let mut best: Option<f32> = None;
    let mut lower = set.margin();
    for &i in set.candidates(p) {
        let b = set.instances()[i as usize].aabb();
        let gap = (p - p.clamp(b.min, b.max)).length();
        if gap > limit {
            lower = lower.min(gap);
            continue;
        }
        let d = set.sdf(i, p);
        best = Some(best.map_or(d, |b: f32| b.min(d)));
    }
    (best, lower)
}

fn apply_terms(mut f: f32, terms: &[Term], p: Vec3) -> f32 {
    for t in terms {
        match t {
            Term::Noise { noise, amplitude } => f += (noise.sample(p) - 0.5) * amplitude,
            Term::Instances { set, op, blend } => {
                // Only an instance within |f| + blend of p can move a cut or a join there (further away it changes the
                // value, never the sign).
                let (near, lower) = nearest(set, p, f.abs() + blend);
                let d = match (near, op) {
                    (Some(d), _) => d,
                    // Intersect keeps only what is inside some instance: with none near, p is outside all of them by
                    // at least `lower`. Skipping the term instead would keep the base here and tear the field.
                    (None, BoolOp::Intersect) => lower,
                    (None, _) => continue,
                };
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
    /// The instances `spec` would scatter on this document's base surface, without adding a layer.
    pub(crate) fn generator_instances(&mut self, spec: &GeneratorSpec) -> Vec<Instance> {
        let basis = self.generator_basis();
        let density = self.density_for(spec, None, false);
        self.ensure_mapping_channels(spec);
        let inputs = crate::generator::layer::BuildInputs { basis: &basis, density: density.as_ref(), channels: &self.channels, base: &self.base, faces: &self.faces, bvh: &self.bvh };
        crate::generator::layer::GeneratorCache::build(spec, &inputs, None).instances().to_vec()
    }

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

    /// Everything solidify reads, copied out so it can run on another thread while the document keeps being edited.
    pub fn solid_input(&self) -> SolidInput {
        let specs = self
            .composite_order()
            .into_iter()
            .filter(|&i| self.layers[i].gen_cache.is_some() && self.layers[i].scale > 0.0)
            .filter_map(|i| self.layers[i].generator.clone())
            .collect();
        SolidInput { positions: self.plain_positions(), faces: self.faces.clone(), specs, cache: self.solid_cache.clone() }
    }

    /// Solidify the stack into a clean quad mesh. The document is not changed; the caller decides what to do with the
    /// result (start a new document from it, which is what a Remesh checkpoint does).
    pub fn solidify(&self, settings: &SolidSettings) -> Result<(PolyMesh, SolidStats)> {
        self.solid_input().run(settings)
    }
}

/// A snapshot of a document's stack for [`SolidInput::run`]. It shares the document's distance cache, so a preview run
/// on a worker thread warms the cache for the next one.
pub struct SolidInput {
    positions: Vec<Vec3>,
    faces: Vec<crate::mesh::Face>,
    specs: Vec<GeneratorSpec>,
    cache: Arc<Mutex<SolidCaches>>,
}

impl SolidInput {
    /// Whether there is any generator layer to solidify.
    pub fn has_layers(&self) -> bool {
        !self.specs.is_empty()
    }

    /// Bounds of the sculpted surface. Unlike the live mesh's, they do not move while generator layers are edited, so a
    /// detail setting turned into a voxel size with them gives the same lattice every time, which keeps the caches warm.
    pub fn surface_bounds(&self) -> Aabb {
        Aabb::from_points(&self.positions)
    }

    pub fn run(&self, settings: &SolidSettings) -> Result<(PolyMesh, SolidStats)> {
        let mut stats = SolidStats::default();
        let t_prep = Instant::now();
        let surface = fingerprint(&self.positions);
        // The sculpted surface and its distance structure, built once per surface.
        let plain = {
            let mut caches = self.cache.lock().unwrap();
            match &caches.plain {
                Some(p) if p.surface == surface => p.clone(),
                _ => {
                    // A sculpt stroke moves part of the surface: keep the distances it cannot have changed.
                    if let Some(old) = caches.plain.clone()
                        && old.positions.len() == self.positions.len()
                    {
                        let mut changed = Aabb::EMPTY;
                        for (a, b) in old.positions.iter().zip(&self.positions) {
                            if a != b {
                                changed.grow(*a);
                                changed.grow(*b);
                            }
                        }
                        // A moved vertex changes every face around it, and the sign test reads the normals of the
                        // vertices around those faces: two rings, so two of the longest edges (old or new) further out.
                        if changed.min.x <= changed.max.x {
                            let longest = |pos: &[Vec3]| {
                                self.faces
                                    .iter()
                                    .map(|f| {
                                        let n = crate::mesh::face_len(f);
                                        (0..n).map(|k| pos[f[k] as usize].distance(pos[f[(k + 1) % n] as usize])).fold(0.0f32, f32::max)
                                    })
                                    .fold(0.0f32, f32::max)
                            };
                            let grow = 2.0 * longest(&old.positions).max(longest(&self.positions));
                            changed = Aabb { min: changed.min - Vec3::splat(grow), max: changed.max + Vec3::splat(grow) };
                        }
                        let carried: Vec<Arc<BaseCache>> = caches.bases.iter().filter(|c| c.surface == old.surface).map(|c| Arc::new(c.carried_over(surface, &changed))).collect();
                        caches.bases = carried;
                    }
                    let doc = Document::from_mesh(PolyMesh { positions: self.positions.clone(), faces: self.faces.clone() })?;
                    let sdf = MeshSdf::build_owned(doc.bvh().clone(), doc.positions().to_vec(), doc.faces().to_vec(), doc.topology());
                    let p = Arc::new(PlainSurface { surface, positions: self.positions.clone(), sdf, bounds: doc.bounds(), doc: Mutex::new(doc) });
                    caches.plain = Some(p.clone());
                    caches.scatters.clear();
                    p
                }
            }
        };
        stats.prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;
        let sdf = &plain.sdf;
        let diagonal = plain.bounds.extent().length().max(1e-3);
        let bound = diagonal * 2.0;

        // Generator layers that contribute, bottom to top.
        let specs = &self.specs;
        stats.layers = specs.len();
        let reach_of = |s: &GeneratorSpec| match s.kind {
            GeneratorKind::Scatter => s.size * (1.0 + s.size_var) * (1.0 + s.stretch_var) * 1.8 + s.blend * s.size,
            // Noise moves the surface by up to its amplitude.
            GeneratorKind::Noise => s.noise.amplitude,
        };
        let reach = specs.iter().map(reach_of).fold(0.0, f32::max) + specs.iter().filter(|s| s.kind == GeneratorKind::Noise).map(|s| s.noise.amplitude).sum::<f32>();
        // Noise steepens the field by about amplitude x scale.
        let noise_slope: f32 = specs.iter().filter(|s| s.kind == GeneratorKind::Noise).map(|s| s.noise.amplitude * s.noise.scale).sum();
        let pad = reach + 4.0 * settings.voxel;
        let snap = |v: Vec3, step: f32| (v / step).floor() * step;
        let bounds = Aabb { min: snap(plain.bounds.min - Vec3::splat(pad), settings.voxel), max: plain.bounds.max + Vec3::splat(pad) };
        // Blocks are skipped when their centre is this far from the surface, so the cutters must be findable that far out.
        let margin = 8.0 * settings.voxel * 0.87 * 1.5;
        // The surfaces later layers scatter on depend only on the layers below them (not on the final detail, nor on the
        // layers above), so they stay cached while a higher layer is dragged.
        let coarse_voxel = plain.bounds.extent().max_element() / 40.0;

        let base_field = |p: Vec3| sdf.signed_distance(p, bound);
        // The final extraction reuses base distances from earlier calls when the surface and spacing are unchanged.
        let cache = {
            // One cache per lattice spacing, so a coarse preview while dragging and the full-detail one after do not
            // evict each other. The two most recently used are kept.
            let mut caches = self.cache.lock().unwrap();
            let slots = &mut caches.bases;
            slots.retain(|c| c.surface == surface && c.entries.load(Ordering::Relaxed) < CACHE_LIMIT);
            match slots.iter().position(|c| c.voxel_bits == settings.voxel.to_bits()) {
                Some(i) => {
                    let c = slots.remove(i);
                    slots.push(c.clone());
                    c
                }
                None => {
                    let fresh = Arc::new(BaseCache::new(surface, settings.voxel));
                    slots.push(fresh.clone());
                    if slots.len() > 2 {
                        slots.remove(0);
                    }
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
        // What the layers so far amount to, so a layer's scatter is found again when nothing below it changed.
        let mut below = DefaultHasher::new();
        surface.hash(&mut below);
        let mut lower_reach = 0.0f32;
        for spec in specs {
            let text = format!("{spec:?}");
            if spec.kind == GeneratorKind::Noise {
                terms.push(Term::Noise { noise: spec.noise.build(), amplitude: spec.noise.amplitude });
                text.hash(&mut below);
                lower_reach += spec.noise.amplitude;
                continue;
            }
            let key = {
                let mut h = below.clone();
                text.hash(&mut h);
                h.finish()
            };
            let known = self.cache.lock().unwrap().scatters.get(&key).cloned();
            let instances = match known {
                Some(v) => v,
                None => {
                    // The surface so far is what this layer scatters on: the sculpted surface for the first, a coarse
                    // extraction of the layers below after that.
                    let found = if terms.is_empty() {
                        plain.doc.lock().unwrap().generator_instances(spec)
                    } else {
                        let p = lower_reach + 4.0 * coarse_voxel;
                        let lower = Aabb { min: snap(plain.bounds.min - Vec3::splat(p), coarse_voxel), max: plain.bounds.max + Vec3::splat(p) };
                        let coarse = remesh(&|p: Vec3| apply_terms(base_field(p), &terms, p), lower, &RemeshParams { voxel: coarse_voxel, lipschitz: 1.5 + noise_slope });
                        if coarse.quads.is_empty() {
                            return Err(Error::InvalidData("the stack leaves nothing to scatter on".into()));
                        }
                        Document::from_mesh(coarse.into_poly_mesh())?.generator_instances(spec)
                    };
                    let found = Arc::new(found);
                    let mut caches = self.cache.lock().unwrap();
                    if caches.scatters.len() >= SCATTER_LIMIT {
                        caches.scatters.clear();
                    }
                    caches.scatters.insert(key, found.clone());
                    found
                }
            };
            text.hash(&mut below);
            lower_reach = lower_reach.max(reach_of(spec));
            if instances.is_empty() {
                continue;
            }
            let mean_size = instances.iter().map(|i| i.scale.max_element()).sum::<f32>() / instances.len() as f32;
            let blend = spec.blend * mean_size;
            terms.push(Term::Instances { set: InstanceSet::new(instances.to_vec(), margin + blend), op: spec.op, blend });
        }
        stats.scatter_ms = t.elapsed().as_secs_f64() * 1e3;

        let field = |p: Vec3| apply_terms(cached_base(p), &terms, p);
        let t = Instant::now();
        // Gradient above 1 where blends overlap and grain is added, so skip blocks conservatively.
        let lipschitz = 1.3 + noise_slope;
        let mut out = remesh(&field, bounds, &RemeshParams { voxel: settings.voxel, lipschitz });
        stats.pieces_dropped = crate::generator::remesh::drop_small_pieces(&mut out, settings.min_piece);
        stats.extract_ms = t.elapsed().as_secs_f64() * 1e3;
        stats.base_hits = hits.load(Ordering::Relaxed);
        stats.base_misses = misses.load(Ordering::Relaxed);
        stats.quads = out.quads.len();
        stats.evaluated = out.evaluated;
        stats.total = out.total;
        Ok((out.into_poly_mesh(), stats))
    }
}

//! The generator layer: a procedural layer that scatters a primitive over the surface and adds, subtracts or
//! intersects it against the layers below.
//!
//! A generator layer stores parameters only ([`GeneratorSpec`]). Its placement and spatial hash are derived data
//! ([`GeneratorCache`]), rebuilt whenever the spec or the topology changes. During compositing it acts as a *function
//! layer*: it reads the positions accumulated from the layers beneath it and returns displaced positions, which are then
//! blended by the layer's strength and mask. Placement is taken from the base surface, so it stays put while you sculpt.
//!
//! The derived data is staged so an edit redoes only what it invalidates, which is what keeps a slider drag cheap on a
//! dense mesh:
//!
//! | stage | holds | rebuilt when |
//! |---|---|---|
//! | [`SurfaceBasis`] | base normals and the per-face sampling table | the base surface or topology changes |
//! | samples | surface points with their normals | count or seed changes |
//! | instances and index | cutter transforms and the spatial hash | any parameter changes (cost is per cutter) |

use std::sync::Arc;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::density::{Density, DensityField};
use super::mapping::{AttributeRow, Mapper};
use super::place::{SurfaceSample, SurfaceSampler};
use super::{BoolOp, EvalParams, Instance, InstanceSet, Primitive, affected_leaves, eval_vertex, frame_from_normal};
use crate::bvh::Bvh;
use crate::geom::Aabb;
use crate::mesh::{self, Face, Topology};
use crate::noise::{Noise, NoiseKind, NoiseParams};

/// What a generator layer does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorKind {
    /// Scatter a primitive over the surface and combine it with the layers below.
    #[default]
    Scatter,
    /// Roughen the surface with noise along its normals.
    Noise,
}

/// Parameters of a noise layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseSpec {
    /// Peak-to-peak displacement along the surface normal, in world units.
    pub amplitude: f32,
    /// Cycles per world unit.
    pub scale: f32,
    pub octaves: u32,
    pub seed: u32,
}

impl Default for NoiseSpec {
    fn default() -> Self {
        NoiseSpec { amplitude: 0.03, scale: 9.0, octaves: 4, seed: 1 }
    }
}

impl NoiseSpec {
    pub fn build(&self) -> Noise {
        Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: self.scale, octaves: self.octaves.max(1), seed: self.seed, ..Default::default() })
    }
}

/// User-editable generator parameters. Cheap to copy, so undo can swap them like any other layer property.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratorSpec {
    /// What the layer does; the scatter parameters below are unused by a noise layer, and `noise` by a scatter layer.
    #[serde(default)]
    pub kind: GeneratorKind,
    #[serde(default)]
    pub noise: NoiseSpec,
    pub primitive: Primitive,
    pub op: BoolOp,
    /// Number of instances scattered over the surface.
    pub count: u32,
    pub seed: u64,
    /// Primitive scale: the canonical primitive's half extent times this, in world units.
    pub size: f32,
    /// Offset of the primitive's centre along the surface normal, in multiples of `size`. Positive lifts it out of
    /// the surface, negative sinks it in. A subtracting primitive centred exactly on the surface (0) cannot dig: the
    /// vertices inside it slide sideways to its rim, so cutters want a positive lift.
    pub lift: f32,
    /// Spin each instance by a random angle around the surface normal.
    pub random_spin: bool,
    /// Smallest distance between two instance centres, in world units; `0` places them independently. If the surface
    /// cannot hold `count` instances this far apart, fewer are placed.
    #[serde(default)]
    pub min_spacing: f32,
    /// Random variation of each instance's size, `0` (all alike) to `1` (anywhere from nothing to double).
    #[serde(default)]
    pub size_var: f32,
    /// Random variation of each axis separately, on top of `size_var`, so instances come out stretched or squashed.
    #[serde(default)]
    pub stretch_var: f32,
    /// Largest random lean away from the surface normal, in degrees.
    #[serde(default)]
    pub tilt: f32,
    /// Random variation of `lift`, in multiples of the instance's size.
    #[serde(default)]
    pub lift_var: f32,
    /// Where on the surface the instances gather. The count stays fixed; a density redistributes it.
    #[serde(default)]
    pub density: Density,
    /// Rows that let the surface set each instance's size, height, lift and lean. See [`super::mapping`].
    #[serde(default)]
    pub mapping: Vec<AttributeRow>,
    /// How softly this layer meets the surface when the stack is solidified, as a fraction of each instance's size:
    /// `0` is a hard cut, around `0.25` rounds the join. The live displacement preview ignores it.
    #[serde(default = "default_blend")]
    pub blend: f32,
}

fn default_blend() -> f32 {
    0.25
}

impl Default for GeneratorSpec {
    fn default() -> Self {
        GeneratorSpec {
            kind: GeneratorKind::Scatter,
            noise: NoiseSpec::default(),
            primitive: Primitive::Sphere,
            op: BoolOp::Subtract,
            count: 100,
            seed: 1,
            size: 0.05,
            lift: 0.6,
            random_spin: true,
            min_spacing: 0.0,
            size_var: 0.0,
            stretch_var: 0.0,
            tilt: 0.0,
            lift_var: 0.0,
            density: Density::Even,
            mapping: Vec::new(),
            blend: default_blend(),
        }
    }
}

/// Facts about the base surface that every generator edit needs and none changes. Building it walks every face and
/// vertex, so it is kept by the document and shared by all generator layers until the base or topology changes.
#[derive(Debug)]
pub struct SurfaceBasis {
    normals: Vec<Vec3>,
    sampler: SurfaceSampler,
}

impl SurfaceBasis {
    /// Per-vertex normals of the base surface.
    pub(crate) fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    pub fn build(base: &[Vec3], faces: &[Face], topo: &Topology) -> SurfaceBasis {
        SurfaceBasis { normals: mesh::compute_normals(base, faces, topo), sampler: SurfaceSampler::new(base, faces, None) }
    }
}

/// Everything about the surface a build reads, gathered so the build's signature stays small.
pub struct BuildInputs<'a> {
    pub basis: &'a Arc<SurfaceBasis>,
    pub density: Option<&'a Arc<DensityField>>,
    pub channels: &'a std::collections::BTreeMap<String, Vec<f32>>,
    pub base: &'a [Vec3],
    pub faces: &'a [Face],
    pub bvh: &'a Bvh,
}

/// Placement and spatial hash derived from a spec and the base surface.
#[derive(Clone, Debug)]
pub struct GeneratorCache {
    /// The parameters this cache was built for; the next build diffs against them to see which stages to redo.
    spec: GeneratorSpec,
    basis: Arc<SurfaceBasis>,
    density: Option<Arc<DensityField>>,
    samples: Arc<Vec<SurfaceSample>>,
    pub(crate) set: InstanceSet,
    pub(crate) params: EvalParams,
    /// Leaves whose bounds an instance can reach, ascending.
    pub(crate) leaves: Vec<u32>,
    /// A noise layer's table, shared between rebuilds while its parameters are unchanged.
    noise: Option<Arc<Noise>>,
}

impl GeneratorCache {
    /// Scatter `spec.count` instances over `base` and index them, reusing whatever `previous` still has right.
    pub fn build(spec: &GeneratorSpec, inputs: &BuildInputs<'_>, previous: Option<&GeneratorCache>) -> GeneratorCache {
        let BuildInputs { basis, density, channels, base, faces, bvh } = *inputs;
        if spec.kind == GeneratorKind::Noise {
            // A noise layer places nothing; it reaches every leaf.
            let noise = match previous.and_then(|p| p.noise.as_ref().map(|n| (p, n))) {
                Some((p, n)) if p.spec.noise.scale == spec.noise.scale && p.spec.noise.octaves == spec.noise.octaves && p.spec.noise.seed == spec.noise.seed => n.clone(),
                _ => Arc::new(spec.noise.build()),
            };
            return GeneratorCache {
                spec: spec.clone(),
                basis: basis.clone(),
                density: None,
                samples: Arc::new(Vec::new()),
                set: InstanceSet::new(Vec::new(), 0.0),
                params: EvalParams { op: BoolOp::Subtract, reach: 0.0, iterations: 1 },
                leaves: (0..bvh.leaves.len() as u32).collect(),
                noise: Some(noise),
            };
        }
        let same_density = |p: &GeneratorCache| match (&p.density, density) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        let samples = match previous {
            Some(p)
                if Arc::ptr_eq(&p.basis, basis)
                    && same_density(p)
                    && p.spec.count == spec.count
                    && p.spec.seed == spec.seed
                    && p.spec.min_spacing == spec.min_spacing =>
            {
                p.samples.clone()
            }
            _ => Arc::new(match density {
                Some(d) => scatter(&d.sampler, Some(&d.values), base, &basis.normals, faces, spec),
                None => scatter(&basis.sampler, None, base, &basis.normals, faces, spec),
            }),
        };
        let seed = spec.seed;
        let mapper = Mapper::new(&spec.mapping, channels);
        let count = samples.len();
        let instances = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let i = i as u64;
                // What the surface says about this instance (empty rows say nothing).
                let m = mapper.map(s, i as usize, count, |stream, k| unit_random(seed ^ MAP_STREAM ^ stream as u64, k));
                let spin = if spec.random_spin { unit_random(seed, i) * std::f32::consts::TAU } else { 0.0 };
                // Each kind of variation draws from its own stream, so changing one never reshuffles another.
                let size = spec.size * (1.0 + spec.size_var * signed_random(seed ^ SIZE_STREAM, i)).max(0.05) * m.size.max(0.05);
                let mut axes = Vec3::new(
                    1.0 + spec.stretch_var * signed_random(seed ^ STRETCH_STREAM, 3 * i),
                    1.0 + spec.stretch_var * signed_random(seed ^ STRETCH_STREAM, 3 * i + 1),
                    1.0 + spec.stretch_var * signed_random(seed ^ STRETCH_STREAM, 3 * i + 2),
                )
                .max(Vec3::splat(0.05));
                axes.y *= m.stretch.max(0.05);
                let lift = spec.lift + spec.lift_var * signed_random(seed ^ LIFT_STREAM, i) + m.lift;
                let max_tilt = (spec.tilt + m.tilt_degrees).clamp(0.0, 89.0).to_radians();
                let up = lean(s.normal, max_tilt * unit_random(seed ^ TILT_STREAM, i), unit_random(seed ^ TILT_STREAM, i + (1 << 40)) * std::f32::consts::TAU);
                Instance::new(spec.primitive, s.position + s.normal * (lift * size), frame_from_normal(up, spin), axes * size)
            })
            .collect();
        // The largest an instance can grow, which bounds how far it can reach.
        let mapped_growth: f32 = spec.mapping.iter().map(|r| r.max_factor()).product();
        let biggest = spec.size * (1.0 + spec.size_var) * (1.0 + spec.stretch_var) * mapped_growth;
        let reach = if spec.op == BoolOp::Subtract { 0.0 } else { 0.25 * biggest };
        let set = InstanceSet::new(instances, reach);
        let boxes: Vec<Aabb> = (0..set.len()).filter(|&i| set.is_active(i)).map(|i| set.aabb(i)).collect();
        let leaves = affected_leaves(bvh, &boxes);
        GeneratorCache { spec: spec.clone(), basis: basis.clone(), density: density.cloned(), samples, set, params: EvalParams { op: spec.op, reach, iterations: 3 }, leaves, noise: None }
    }

    /// The density the instances were scattered by, if any.
    pub fn density(&self) -> Option<&Arc<DensityField>> {
        self.density.as_ref()
    }

    /// The cutters as placed, for inspection.
    pub fn instances(&self) -> &[Instance] {
        self.set.instances()
    }

    /// True when `self` and `other` share the same scattered points, so an edit between them did not resample.
    pub fn shares_samples_with(&self, other: &GeneratorCache) -> bool {
        Arc::ptr_eq(&self.samples, &other.samples)
    }

    /// Displace the vertices of one leaf. `pos` is the leaf's slice of accumulated positions, `first` its first vertex
    /// index, and `weight` the layer's per-vertex strength. Positions are blended toward the evaluated result.
    #[inline]
    pub(crate) fn apply_leaf(&self, bounds: &Aabb, pos: &mut [Vec3], normals: &[Vec3], first: usize, weight: impl Fn(usize) -> f32) {
        if let Some(noise) = &self.noise {
            let amplitude = self.spec.noise.amplitude;
            for (k, p) in pos.iter_mut().enumerate() {
                let v = first + k;
                let w = weight(v);
                if w != 0.0 {
                    *p += normals[v] * ((noise.sample(*p) - 0.5) * amplitude * w);
                }
            }
            return;
        }
        if !self.set.touches(bounds) {
            return;
        }
        for (k, p) in pos.iter_mut().enumerate() {
            let v = first + k;
            let w = weight(v);
            if w == 0.0 {
                continue;
            }
            let q = eval_vertex(*p, normals[v], &self.set, &self.params);
            *p += (q - *p) * w;
        }
    }
}

const SIZE_STREAM: u64 = 0x5129_A1C3_0F7D_E001;
const STRETCH_STREAM: u64 = 0x7C15_93B2_44D1_F002;
const LIFT_STREAM: u64 = 0x3AF8_61E9_B2C4_D003;
const TILT_STREAM: u64 = 0x96D2_0E47_58A3_C004;
const ORDER_STREAM: u64 = 0xE410_7B6D_91F2_A005;
const MAP_STREAM: u64 = 0x1D8B_44F0_C6A7_E006;

/// Deterministic value in `[-1, 1)`.
fn signed_random(seed: u64, i: u64) -> f32 {
    unit_random(seed, i) * 2.0 - 1.0
}

/// `normal` leaned by `angle` towards the direction `around` radians round it (`angle` 0 returns it unchanged).
fn lean(normal: Vec3, angle: f32, around: f32) -> Vec3 {
    if angle <= 1e-6 {
        return normal;
    }
    let (t, b) = {
        let t = normal.any_orthonormal_vector();
        (t, normal.cross(t))
    };
    let axis = (t * around.cos() + b * around.sin()).normalize_or(t);
    (normal * angle.cos() + axis * angle.sin()).normalize_or(normal)
}

/// Scatter `spec.count` points over the surface. With a minimum spacing this is dart throwing: candidates (in a random
/// order, so no region of the mesh fills first) are accepted when no accepted point is within the spacing, until `count`
/// are placed or fresh candidates stop finding room.
fn scatter(sampler: &SurfaceSampler, mask: Option<&[f32]>, base: &[Vec3], normals: &[Vec3], faces: &[Face], spec: &GeneratorSpec) -> Vec<SurfaceSample> {
    let count = spec.count as usize;
    if spec.min_spacing <= 0.0 || count == 0 {
        return sampler.sample(base, normals, faces, mask, count, spec.seed);
    }
    let cell = spec.min_spacing;
    let key = |p: Vec3| ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32);
    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>> = std::collections::HashMap::new();
    let mut accepted: Vec<SurfaceSample> = Vec::with_capacity(count);
    let min2 = spec.min_spacing * spec.min_spacing;
    let mut dry_batches = 0;
    for batch in 0u64..24 {
        let candidates = sampler.sample(base, normals, faces, mask, (count * 3).max(256), spec.seed.wrapping_add(batch.wrapping_mul(0x9E37_79B9)));
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| unit_random(spec.seed ^ ORDER_STREAM ^ batch, a as u64).total_cmp(&unit_random(spec.seed ^ ORDER_STREAM ^ batch, b as u64)));
        let before = accepted.len();
        for ci in order {
            if accepted.len() >= count {
                break;
            }
            let c = candidates[ci];
            let (kx, ky, kz) = key(c.position);
            let crowded = (-1..=1).any(|dx| {
                (-1..=1).any(|dy| {
                    (-1..=1).any(|dz| grid.get(&(kx + dx, ky + dy, kz + dz)).is_some_and(|v| v.iter().any(|&j| accepted[j as usize].position.distance_squared(c.position) < min2)))
                })
            });
            if !crowded {
                grid.entry((kx, ky, kz)).or_default().push(accepted.len() as u32);
                accepted.push(c);
            }
        }
        if accepted.len() >= count {
            break;
        }
        // Two batches in a row that found almost nothing: the surface is full.
        dry_batches = if accepted.len() - before < count / 50 + 1 { dry_batches + 1 } else { 0 };
        if dry_batches >= 2 {
            break;
        }
    }
    accepted
}

/// Deterministic value in `[0, 1)` from a seed and an index (splitmix64).
fn unit_random(seed: u64, i: u64) -> f32 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

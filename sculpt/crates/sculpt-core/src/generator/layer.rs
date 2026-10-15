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

use super::place::{SurfaceSample, SurfaceSampler};
use super::{BoolOp, EvalParams, Instance, InstanceSet, Primitive, affected_leaves, eval_vertex, frame_from_normal};
use crate::bvh::Bvh;
use crate::geom::Aabb;
use crate::mesh::{self, Face, Topology};

/// User-editable generator parameters. Cheap to copy, so undo can swap them like any other layer property.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratorSpec {
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
}

impl Default for GeneratorSpec {
    fn default() -> Self {
        GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 100, seed: 1, size: 0.05, lift: 0.6, random_spin: true }
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

/// Placement and spatial hash derived from a spec and the base surface.
#[derive(Clone, Debug)]
pub struct GeneratorCache {
    /// The parameters this cache was built for; the next build diffs against them to see which stages to redo.
    spec: GeneratorSpec,
    basis: Arc<SurfaceBasis>,
    samples: Arc<Vec<SurfaceSample>>,
    pub(crate) set: InstanceSet,
    pub(crate) params: EvalParams,
    /// Leaves whose bounds an instance can reach, ascending.
    pub(crate) leaves: Vec<u32>,
}

impl GeneratorCache {
    /// Scatter `spec.count` instances over `base` and index them, reusing whatever `previous` still has right.
    pub fn build(spec: &GeneratorSpec, basis: &Arc<SurfaceBasis>, base: &[Vec3], faces: &[Face], bvh: &Bvh, previous: Option<&GeneratorCache>) -> GeneratorCache {
        let samples = match previous {
            Some(p) if Arc::ptr_eq(&p.basis, basis) && p.spec.count == spec.count && p.spec.seed == spec.seed => p.samples.clone(),
            _ => Arc::new(basis.sampler.sample(base, &basis.normals, faces, None, spec.count as usize, spec.seed)),
        };
        let instances = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let spin = if spec.random_spin { unit_random(spec.seed, i as u64) * std::f32::consts::TAU } else { 0.0 };
                Instance::new(spec.primitive, s.position + s.normal * (spec.lift * spec.size), frame_from_normal(s.normal, spin), Vec3::splat(spec.size))
            })
            .collect();
        let reach = if spec.op == BoolOp::Subtract { 0.0 } else { 0.25 * spec.size };
        let set = InstanceSet::new(instances, reach);
        let boxes: Vec<Aabb> = (0..set.len()).filter(|&i| set.is_active(i)).map(|i| set.aabb(i)).collect();
        let leaves = affected_leaves(bvh, &boxes);
        GeneratorCache { spec: spec.clone(), basis: basis.clone(), samples, set, params: EvalParams { op: spec.op, reach, iterations: 3 }, leaves }
    }

    /// True when `self` and `other` share the same scattered points, so an edit between them did not resample.
    pub fn shares_samples_with(&self, other: &GeneratorCache) -> bool {
        Arc::ptr_eq(&self.samples, &other.samples)
    }

    /// Displace the vertices of one leaf. `pos` is the leaf's slice of accumulated positions, `first` its first vertex
    /// index, and `weight` the layer's per-vertex strength. Positions are blended toward the evaluated result.
    #[inline]
    pub(crate) fn apply_leaf(&self, bounds: &Aabb, pos: &mut [Vec3], normals: &[Vec3], first: usize, weight: impl Fn(usize) -> f32) {
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

/// Deterministic value in `[0, 1)` from a seed and an index (splitmix64).
fn unit_random(seed: u64, i: u64) -> f32 {
    let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

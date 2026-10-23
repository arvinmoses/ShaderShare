//! A generator layer on a volume: a primitive scattered over the form's surface and joined to it, or cut from it, with
//! a soft blend, exactly, in the field (no limit to what is single-valued along a normal, as on a mesh).
//!
//! * **Anchors.** Each instance hangs from a point of the form's surface and its normal. Points are drawn over level
//!   0's surface by the lattice edges it crosses (so about evenly by area, with no surface to extract), apart by the
//!   spec's spacing if it has one, and moved onto the finest surface the form has. When a stroke moves the form, each
//!   anchor is found again on the new surface along its normal, so what was scattered rides the form. A spec edit
//!   that does not change the count, seed or spacing keeps the anchors.
//! * **Instances** come from the anchors as on a mesh ([`super::layer`]): size, stretch, lift, tilt and spin varied by
//!   the seed, and the mapping rows that read the surface's slope.
//! * **Joining.** Instances join one another and the surface below with a polynomial smooth minimum over `blend` times
//!   the spec's size (a subtracting layer, a smooth maximum against them), so a rock sits in the ground rounded where
//!   it meets it.
//! * **Blocks.** What the layer changes is held at every level up to its own (level 0 all the way through each
//!   instance's box, finer levels in a shell around its surface), so the inside of a large instance reads it from the
//!   coarser levels, and a distant view that leaves fine levels out keeps it. Past its own level, the form's own
//!   blocks within reach take it in too.

use std::sync::Arc;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::layer::{GeneratorSpec, ORDER_STREAM, biggest_instance, place_instance, unit_random};
use super::level_volume::{HashSet, LevelVolume};
use super::mapping::Mapper;
use super::place::SurfaceSample;
use super::{BoolOp, InstanceSet};
use crate::geom::Aabb;

const B: i32 = 8;

/// A point of the surface an instance hangs from, with the surface's outward normal there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub p: Vec3,
    pub n: Vec3,
}

#[derive(Clone)]
pub struct VolumeGen {
    pub spec: GeneratorSpec,
    anchors: Arc<Vec<Anchor>>,
    set: Arc<InstanceSet>,
}

/// Polynomial smooth minimum over `k`.
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Where the field `f` changes sign along `d` from `p`, nearest `p`, within `reach` (searched in steps of `step`).
fn nearest_root(f: &dyn Fn(Vec3) -> f32, p: Vec3, d: Vec3, step: f32, reach: f32) -> Option<Vec3> {
    let f0 = f(p);
    if f0 == 0.0 {
        return Some(p);
    }
    let steps = (reach / step).ceil() as i32;
    for k in 1..=steps {
        for s in [1.0f32, -1.0] {
            let (a, b) = (s * (k - 1) as f32 * step, s * k as f32 * step);
            let (fa, fb) = (f(p + d * a), f(p + d * b));
            if (fa < 0.0) != (fb < 0.0) {
                let (mut lo, mut hi, mut flo) = (a, b, fa);
                for _ in 0..24 {
                    let m = 0.5 * (lo + hi);
                    let fm = f(p + d * m);
                    if (fm < 0.0) == (flo < 0.0) {
                        lo = m;
                        flo = fm;
                    } else {
                        hi = m;
                    }
                }
                return Some(p + d * (0.5 * (lo + hi)));
            }
        }
    }
    None
}

/// Points over the form's level 0 surface, `count` (or fewer when spaced and the surface is full), drawn by `seed`.
fn scatter(form: &LevelVolume, spec: &GeneratorSpec) -> Vec<Anchor> {
    let count = spec.count as usize;
    if count == 0 {
        return Vec::new();
    }
    // Level 0's crossing edges, block by block, in a fixed order (so a seed always draws the same points).
    let mut keys = form.level_keys(0);
    keys.sort_unstable_by_key(|k| (k.z, k.y, k.x));
    let weights: Vec<u32> = keys.par_iter().map(|bc| form.surface_quads(0, *bc)).collect();
    let mut cdf = Vec::with_capacity(weights.len());
    let mut total = 0u64;
    for w in &weights {
        total += *w as u64;
        cdf.push(total);
    }
    if total == 0 {
        return Vec::new();
    }
    let (v0, origin, top) = (form.voxel(0), form.origin(), form.levels() - 1);
    let draw = |i: u64| -> Option<Anchor> {
        let pick = ((unit_random(spec.seed, 2 * i) as f64 * total as f64) as u64).min(total - 1);
        let b = cdf.partition_point(|&c| c <= pick);
        let bc = keys[b];
        let s = form.block((0, bc))?;
        // The block's crossing edges, in order; the draw picks one, and a point along it where the field is zero.
        let mut edges: Vec<(IVec3, usize, f32)> = Vec::new();
        for z in 0..8i32 {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let i = (x + 8 * y + 64 * z) as usize;
                    for (axis, (ok, j)) in [(x < 7, i + 1), (y < 7, i + 8), (z < 7, i + 64)].into_iter().enumerate() {
                        if ok && (s[i] < 0.0) != (s[j] < 0.0) {
                            edges.push((IVec3::new(x, y, z), axis, s[i] / (s[i] - s[j])));
                        }
                    }
                }
            }
        }
        let (c, axis, t) = *edges.get(((unit_random(spec.seed, 2 * i + 1) * edges.len() as f32) as usize).min(edges.len().max(1) - 1))?;
        let mut q = (bc * B + c).as_vec3();
        q[axis] += t;
        let p = origin + q * v0;
        snap(form, Anchor { p, n: form.normal_upto(p, top, false) }, 2.0 * v0)
    };
    if spec.min_spacing <= 0.0 {
        return (0..count as u64).into_par_iter().filter_map(draw).collect();
    }
    // Dart throwing, as on a mesh: candidates in a random order, kept when none kept is within the spacing.
    let cell = spec.min_spacing;
    let key = |p: Vec3| (p / cell).floor().as_ivec3();
    let mut grid: std::collections::HashMap<IVec3, Vec<u32>> = std::collections::HashMap::new();
    let mut kept: Vec<Anchor> = Vec::with_capacity(count);
    let min2 = cell * cell;
    let mut dry = 0;
    for batch in 0u64..24 {
        let n = (count * 3).max(256) as u64;
        let mut cands: Vec<(f32, Anchor)> = (0..n).into_par_iter().filter_map(|j| draw(batch * n + j).map(|a| (unit_random(spec.seed ^ ORDER_STREAM ^ batch, j), a))).collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let before = kept.len();
        for (_, a) in cands {
            if kept.len() >= count {
                break;
            }
            let k = key(a.p);
            let crowded = (0..27).any(|o| grid.get(&(k + IVec3::new(o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1))).is_some_and(|v| v.iter().any(|&j| kept[j as usize].p.distance_squared(a.p) < min2)));
            if !crowded {
                grid.entry(k).or_default().push(kept.len() as u32);
                kept.push(a);
            }
        }
        if kept.len() >= count {
            break;
        }
        dry = if kept.len() - before < count / 50 + 1 { dry + 1 } else { 0 };
        if dry >= 2 {
            break;
        }
    }
    kept
}

/// `a` moved onto the form's finest surface along its normal (within `reach`), with the normal there.
fn snap(form: &LevelVolume, a: Anchor, reach: f32) -> Option<Anchor> {
    let top = form.levels() - 1;
    let f = |p: Vec3| form.value_upto(p, top, false);
    // Still on the surface (it did not move here): kept exactly, so anchors do not drift stroke by stroke.
    if a.n != Vec3::ZERO && f(a.p).abs() < 1e-3 * form.voxel(top) {
        return Some(a);
    }
    let n = if a.n == Vec3::ZERO { form.normal_upto(a.p, top, false) } else { a.n };
    let p = nearest_root(&f, a.p, n.normalize_or(Vec3::Y), 0.25 * form.voxel(top).max(form.voxel(0) / 8.0), reach)?;
    let n2 = form.normal_upto(p, top, false);
    Some(Anchor { p, n: if n2 == Vec3::ZERO { n } else { n2 } })
}

impl VolumeGen {
    /// Scatter `spec` over `form`, keeping `previous`'s anchors where its count, seed and spacing are the same.
    pub fn build(form: &LevelVolume, spec: &GeneratorSpec, previous: Option<&VolumeGen>) -> VolumeGen {
        let anchors = match previous {
            Some(p) if p.spec.count == spec.count && p.spec.seed == spec.seed && p.spec.min_spacing == spec.min_spacing => p.anchors.clone(),
            _ => Arc::new(scatter(form, spec)),
        };
        Self::with_anchors(spec, anchors)
    }

    fn with_anchors(spec: &GeneratorSpec, anchors: Arc<Vec<Anchor>>) -> VolumeGen {
        let channels = std::collections::BTreeMap::new();
        let mapper = Mapper::new(&spec.mapping, &channels);
        let count = anchors.len();
        let instances = anchors
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let s = SurfaceSample { position: a.p, normal: a.n, face: 0, verts: [0; 3], bary: [1.0, 0.0, 0.0] };
                place_instance(spec, &mapper, &s, i, count)
            })
            .collect();
        let k = spec.blend.max(0.0) * spec.size;
        VolumeGen { spec: spec.clone(), anchors, set: Arc::new(InstanceSet::new(instances, k)) }
    }

    /// The same, with anchors as saved.
    pub fn from_anchors(spec: &GeneratorSpec, anchors: Vec<Anchor>) -> VolumeGen {
        Self::with_anchors(spec, Arc::new(anchors))
    }

    pub fn anchors(&self) -> &[Anchor] {
        &self.anchors
    }

    pub fn instances(&self) -> &[super::Instance] {
        self.set.instances()
    }

    /// Each anchor found again on `form` along its normal, as far as `reach`; one with no surface within it stays.
    pub fn follow(&self, form: &LevelVolume, reach: f32) -> VolumeGen {
        let moved: Vec<Anchor> = self.anchors.par_iter().map(|a| snap(form, *a, reach).unwrap_or(*a)).collect();
        if moved == *self.anchors {
            return self.clone();
        }
        Self::with_anchors(&self.spec, Arc::new(moved))
    }

    /// The blend's width, in world units.
    pub fn blend(&self) -> f32 {
        self.spec.blend.max(0.0) * self.spec.size
    }

    fn subtracts(&self) -> bool {
        self.spec.op == BoolOp::Subtract
    }

    /// Whether any instance's reach meets the box `lo..hi` grown by `pad`.
    pub fn touches(&self, lo: Vec3, hi: Vec3, pad: f32) -> bool {
        self.set.touches(&Aabb { min: lo - Vec3::splat(pad), max: hi + Vec3::splat(pad) })
    }

    /// Instances whose reach meets the box `lo..hi` grown by `pad`.
    pub fn near(&self, lo: Vec3, hi: Vec3, pad: f32, out: &mut Vec<u32>) {
        out.clear();
        self.set.candidates_in_box(&Aabb { min: lo - Vec3::splat(pad), max: hi + Vec3::splat(pad) }, out);
    }

    /// The instances' own field at `p` (joined smoothly), from candidates `near`.
    pub fn field(&self, p: Vec3, near: &[u32]) -> f32 {
        let k = self.blend();
        near.iter().fold(f32::INFINITY, |a, &i| if a.is_infinite() { self.set.sdf(i, p) } else { smin(a, self.set.sdf(i, p), k) })
    }

    /// The field `c` below at `p` with this layer on it, faded by `mask` (0 leaves `c`).
    pub fn apply(&self, c: f32, p: Vec3, near: &[u32], mask: f32) -> f32 {
        if near.is_empty() || mask <= 0.0 {
            return c;
        }
        let d = self.field(p, near);
        if !d.is_finite() {
            return c;
        }
        let k = self.blend();
        let out = if self.subtracts() { -smin(-c, d, k) } else { smin(c, d, k) };
        c + (out - c) * mask.min(1.0)
    }

    /// The blocks of level `l` this layer changes, when it lives at level `own`: level 0's through each instance's box,
    /// a shell around the instances' surface at levels up to `own`, and past it the form's own blocks within reach.
    pub fn blocks(&self, form: &LevelVolume, l: usize, own: usize) -> HashSet<IVec3> {
        let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
        let k = self.blend();
        let pad = k + band + v;
        let size = v * B as f32;
        if l > own {
            // Past its own level: the form's blocks within reach (found among the form's, far fewer than the boxes').
            return form.level_keys(l).into_par_iter().filter(|bc| {
                let lo = origin + (*bc * B).as_vec3() * v;
                self.touches(lo, lo + Vec3::splat(size), pad)
            }).collect::<Vec<_>>().into_iter().collect();
        }
        let mut out: HashSet<IVec3> = HashSet::default();
        for i in 0..self.set.len() {
            if !self.set.is_active(i) {
                continue;
            }
            let b = self.set.aabb(i);
            let lo = ((b.min - Vec3::splat(pad) - origin) / size).floor().as_ivec3();
            let hi = ((b.max + Vec3::splat(pad) - origin) / size).floor().as_ivec3();
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        out.insert(IVec3::new(x, y, z));
                    }
                }
            }
        }
        if l == 0 {
            return out;
        }
        let half = 0.5 * 3f32.sqrt() * size;
        let keep = |bc: &IVec3| -> bool {
            let lo = origin + (*bc * B).as_vec3() * v;
            let c = lo + Vec3::splat(0.5 * size);
            let mut near = Vec::new();
            self.near(lo, lo + Vec3::splat(size), pad, &mut near);
            near.iter().any(|&i| self.set.sdf(i, c).abs() <= half + pad)
        };
        out.into_par_iter().filter(keep).collect::<Vec<_>>().into_iter().collect()
    }

    /// How far it reaches past its instances' boxes, for the blocks to redo around an instance that moved.
    pub fn reach(&self) -> f32 {
        biggest_instance(&self.spec) + self.blend()
    }
}

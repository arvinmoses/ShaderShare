//! Instances of primitives and the uniform-grid spatial hash over their AABBs.
//!
//! The hash is a sparse grid: only occupied cells exist, found by open
//! addressing, so rebuild cost scales with the number of instances (not with
//! the volume of the scene). Cell size is the mean instance extent, so an
//! instance covers at most ~8 cells.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::sdf::Primitive;
use crate::geom::Aabb;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub primitive: Primitive,
    pub rotation: Quat,
    pub translation: Vec3,
    /// Per-axis half-size multiplier of the canonical primitive.
    pub scale: Vec3,
}

impl Instance {
    pub fn new(primitive: Primitive, translation: Vec3, rotation: Quat, scale: Vec3) -> Instance {
        Instance { primitive, rotation, translation, scale }
    }

    /// World-space AABB (exact for the rotated local box, conservative for the shape).
    pub fn aabb(&self) -> Aabb {
        let m = Mat3::from_quat(self.rotation);
        let h = self.primitive.local_half_extent() * self.scale;
        if self.primitive == Primitive::Sphere {
            // Exact AABB of the rotated ellipsoid.
            let (c0, c1, c2) = (m.x_axis * h.x, m.y_axis * h.y, m.z_axis * h.z);
            let e = (c0 * c0 + c1 * c1 + c2 * c2).sqrt();
            return Aabb { min: self.translation - e, max: self.translation + e };
        }
        let e = Vec3::new(
            m.x_axis.x.abs() * h.x + m.y_axis.x.abs() * h.y + m.z_axis.x.abs() * h.z,
            m.x_axis.y.abs() * h.x + m.y_axis.y.abs() * h.y + m.z_axis.y.abs() * h.z,
            m.x_axis.z.abs() * h.x + m.y_axis.z.abs() * h.y + m.z_axis.z.abs() * h.z,
        );
        Aabb { min: self.translation - e, max: self.translation + e }
    }
}

/// Rotation taking local +Y to `normal`, with `spin` radians around it.
pub fn frame_from_normal(normal: Vec3, spin: f32) -> Quat {
    Quat::from_rotation_arc(Vec3::Y, normal.normalize_or(Vec3::Y)) * Quat::from_rotation_y(spin)
}

/// Rotation taking local +Z to `tangent` and local +Y to `up` (made orthogonal).
/// Used to lay edge cutters along an edge with the cutter's diamond axis along
/// the edge normal.
pub fn frame_from_edge(tangent: Vec3, up: Vec3) -> Quat {
    let z = tangent.normalize_or(Vec3::Z);
    let x = up.cross(z).normalize_or(z.any_orthonormal_vector());
    let y = z.cross(x);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

/// Precomputed inverse transform for fast queries.
#[derive(Clone, Copy, Debug)]
struct Baked {
    inv_rot: Mat3,
    rot: Mat3,
    inv_scale: Vec3,
    t: Vec3,
    min_scale: f32,
    prim: Primitive,
    /// Sphere with uniform scale: rotation is irrelevant (fast path).
    ball: bool,
}

impl Baked {
    fn new(i: &Instance) -> Baked {
        let rot = Mat3::from_quat(i.rotation);
        Baked {
            inv_rot: rot.transpose(),
            rot,
            inv_scale: i.scale.recip(),
            t: i.translation,
            min_scale: i.scale.min_element(),
            prim: i.primitive,
            ball: i.primitive == Primitive::Sphere && (i.scale.max_element() - i.scale.min_element()) <= 1e-6 * i.scale.max_element(),
        }
    }
    #[inline]
    fn local(&self, p: Vec3) -> Vec3 {
        (self.inv_rot * (p - self.t)) * self.inv_scale
    }
    #[inline]
    fn sdf(&self, p: Vec3) -> f32 {
        if self.ball {
            return (p - self.t).length() - self.min_scale;
        }
        self.prim.sdf(self.local(p)) * self.min_scale
    }
    #[inline]
    fn sdf_grad(&self, p: Vec3) -> (f32, Vec3) {
        let (d, g) = self.prim.sdf_grad(self.local(p));
        (d * self.min_scale, (self.rot * (g * self.inv_scale)).normalize_or(Vec3::Y))
    }
}

const EMPTY: u64 = u64::MAX;
const COORD_BITS: u32 = 21;

#[derive(Clone, Debug, Default)]
struct Grid {
    origin: Vec3,
    inv_cell: f32,
    cell: f32,
    mask: usize,
    keys: Vec<u64>,
    /// `start` offsets into `items`; slot `s` holds `items[start[s]..start[s+1]]`
    /// only conceptually, so lengths are stored separately.
    start: Vec<u32>,
    len: Vec<u32>,
    items: Vec<u32>,
    occupied: usize,
}

#[inline]
fn hash(key: u64) -> u64 {
    let mut x = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x
}

#[inline]
fn pack(x: u32, y: u32, z: u32) -> u64 {
    x as u64 | (y as u64) << COORD_BITS | (z as u64) << (2 * COORD_BITS)
}

impl Grid {
    #[inline]
    fn find(&self, key: u64) -> Option<usize> {
        if self.keys.is_empty() {
            return None;
        }
        let mut s = hash(key) as usize & self.mask;
        loop {
            let k = self.keys[s];
            if k == key {
                return Some(s);
            }
            if k == EMPTY {
                return None;
            }
            s = (s + 1) & self.mask;
        }
    }

    /// Cell coordinate range of an AABB, clamped to the grid domain.
    #[inline]
    fn cell_range(&self, b: &Aabb) -> ([u32; 3], [u32; 3]) {
        let lo = ((b.min - self.origin) * self.inv_cell).floor().max(Vec3::ZERO);
        let hi = ((b.max - self.origin) * self.inv_cell).floor().max(Vec3::ZERO);
        let lim = ((1u32 << COORD_BITS) - 1) as f32;
        let c = |v: Vec3| [v.x.min(lim) as u32, v.y.min(lim) as u32, v.z.min(lim) as u32];
        (c(lo), c(hi))
    }
}

/// A set of instances plus the spatial hash used to find which ones are near a point.
#[derive(Clone, Debug)]
pub struct InstanceSet {
    instances: Vec<Instance>,
    baked: Vec<Baked>,
    /// Instance AABBs inflated by `margin`.
    aabbs: Vec<Aabb>,
    margin: f32,
    bounds: Aabb,
    grid: Grid,
}

impl InstanceSet {
    /// `margin` inflates every AABB (the boolean's reach), so a point within
    /// `margin` of an instance finds it.
    pub fn new(instances: Vec<Instance>, margin: f32) -> InstanceSet {
        let mut s = InstanceSet {
            instances,
            baked: Vec::new(),
            aabbs: Vec::new(),
            margin,
            bounds: Aabb::EMPTY,
            grid: Grid::default(),
        };
        s.rebuild();
        s
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }
    pub fn aabb(&self, i: usize) -> Aabb {
        self.aabbs[i]
    }
    pub fn margin(&self) -> f32 {
        self.margin
    }
    pub fn cell_size(&self) -> f32 {
        self.grid.cell
    }
    pub fn occupied_cells(&self) -> usize {
        self.grid.occupied
    }
    pub fn memory_bytes(&self) -> usize {
        self.instances.len() * (std::mem::size_of::<Instance>() + std::mem::size_of::<Baked>() + std::mem::size_of::<Aabb>())
            + self.grid.keys.len() * 16
            + self.grid.items.len() * 4
    }

    /// Replace the instances and rebuild everything (bake + grid).
    pub fn set_all(&mut self, instances: Vec<Instance>) {
        self.instances = instances;
        self.rebuild();
    }

    /// Replace a subset of instances; returns the old and new inflated AABBs
    /// (feed them to [`super::affected_leaves`]). The grid is rebuilt.
    pub fn update(&mut self, changes: &[(usize, Instance)]) -> Vec<Aabb> {
        let dirty = self.update_bake_only(changes);
        self.rebuild_grid();
        dirty
    }

    /// Like [`InstanceSet::update`] but leaves the spatial hash stale (callers that never use
    /// [`InstanceSet::candidates`], such as the ADF boolean, skip its rebuild).
    pub fn update_bake_only(&mut self, changes: &[(usize, Instance)]) -> Vec<Aabb> {
        let mut dirty = Vec::with_capacity(changes.len() * 2);
        for &(i, inst) in changes {
            dirty.push(self.aabbs[i]);
            self.instances[i] = inst;
            self.baked[i] = Baked::new(&inst);
            self.aabbs[i] = inflate(inst.aabb(), self.margin);
            dirty.push(self.aabbs[i]);
        }
        dirty
    }

    /// Bake transforms and AABBs only; no spatial hash (see [`InstanceSet::update_bake_only`]).
    pub fn bake_only(instances: Vec<Instance>, margin: f32) -> InstanceSet {
        let mut s = InstanceSet { instances, baked: Vec::new(), aabbs: Vec::new(), margin, bounds: Aabb::EMPTY, grid: Grid::default() };
        s.baked = s.instances.par_iter().map(Baked::new).collect();
        s.aabbs = s.instances.par_iter().map(|i| inflate(i.aabb(), margin)).collect();
        s
    }

    /// Re-bake every instance and rebuild the grid.
    pub fn rebuild(&mut self) {
        let m = self.margin;
        self.baked = self.instances.par_iter().map(Baked::new).collect();
        self.aabbs = self.instances.par_iter().map(|i| inflate(i.aabb(), m)).collect();
        self.rebuild_grid();
    }

    /// Rebuild only the spatial hash from the current AABBs.
    pub fn rebuild_grid(&mut self) {
        let n = self.aabbs.len();
        if n == 0 {
            self.grid = Grid::default();
            self.bounds = Aabb::EMPTY;
            return;
        }
        let (bounds, ext_sum) = self
            .aabbs
            .par_iter()
            .fold(
                || (Aabb::EMPTY, 0.0f64),
                |(b, s), a| (b.union(a), s + a.extent().max_element() as f64),
            )
            .reduce(|| (Aabb::EMPTY, 0.0), |a, b| (a.0.union(&b.0), a.1 + b.1));
        self.bounds = bounds;
        let mean_ext = (ext_sum / n as f64) as f32;
        let cell = mean_ext.max(bounds.extent().max_element() / (1u32 << (COORD_BITS - 1)) as f32).max(1e-9);
        let mut g = Grid { origin: bounds.min, inv_cell: 1.0 / cell, cell, ..Default::default() };

        let ranges: Vec<([u32; 3], [u32; 3])> = self.aabbs.par_iter().map(|a| g.cell_range(a)).collect();
        let entries: usize = ranges
            .par_iter()
            .map(|(lo, hi)| ((hi[0] - lo[0] + 1) as usize) * ((hi[1] - lo[1] + 1) as usize) * ((hi[2] - lo[2] + 1) as usize))
            .sum();
        // Distinct cells are bounded by both the entry count and the cells in the bounds
        // (instances overlap heavily on a surface, so the latter is usually far smaller).
        let dims = ((bounds.extent() * g.inv_cell).floor() + Vec3::ONE).as_u64vec3();
        let cells_in_bounds = dims.x.saturating_mul(dims.y).saturating_mul(dims.z) as usize;
        let cap = (entries.min(cells_in_bounds) * 2).next_power_of_two().max(16);
        g.mask = cap - 1;

        // Pass 1: lock-free insert of cell keys + per-cell counts.
        let akeys: Vec<AtomicU64> = (0..cap).map(|_| AtomicU64::new(EMPTY)).collect();
        let counts: Vec<AtomicU32> = (0..cap).map(|_| AtomicU32::new(0)).collect();
        let slot_of = |key: u64| -> usize {
            let mut s = hash(key) as usize & g.mask;
            loop {
                match akeys[s].compare_exchange(EMPTY, key, Relaxed, Relaxed) {
                    Ok(_) => return s,
                    Err(k) if k == key => return s,
                    _ => s = (s + 1) & g.mask,
                }
            }
        };
        ranges.par_iter().for_each(|(lo, hi)| {
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        counts[slot_of(pack(x, y, z))].fetch_add(1, Relaxed);
                    }
                }
            }
        });

        // Prefix sum over slots (sequential: `cap` u32 adds).
        let mut start = vec![0u32; cap + 1];
        let mut len = vec![0u32; cap];
        let mut run = 0u32;
        let mut occupied = 0usize;
        for s in 0..cap {
            let c = counts[s].load(Relaxed);
            start[s] = run;
            len[s] = c;
            run += c;
            occupied += (c > 0) as usize;
        }
        start[cap] = run;

        // Pass 2: scatter instance ids.
        let cursor: Vec<AtomicU32> = start[..cap].iter().map(|&s| AtomicU32::new(s)).collect();
        let items: Vec<AtomicU32> = (0..run).map(|_| AtomicU32::new(0)).collect();
        ranges.par_iter().enumerate().for_each(|(id, (lo, hi))| {
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let s = slot_of(pack(x, y, z));
                        let at = cursor[s].fetch_add(1, Relaxed);
                        items[at as usize].store(id as u32, Relaxed);
                    }
                }
            }
        });
        let mut items: Vec<u32> = items.into_iter().map(AtomicU32::into_inner).collect();

        // Make per-cell order independent of thread scheduling (determinism).
        let multi: Vec<std::ops::Range<usize>> = (0..cap)
            .filter(|&s| len[s] > 1)
            .map(|s| start[s] as usize..(start[s] + len[s]) as usize)
            .collect();
        crate::bvh::split_ranges_mut(&mut items, multi).into_par_iter().for_each(|sl| sl.sort_unstable());

        g.keys = akeys.into_iter().map(AtomicU64::into_inner).collect();
        g.start = start;
        g.len = len;
        g.items = items;
        g.occupied = occupied;
        self.grid = g;
    }

    /// Instances whose inflated AABB overlaps the cell containing `p`
    /// (a superset of the instances that can influence `p`).
    #[inline]
    pub fn candidates(&self, p: Vec3) -> &[u32] {
        let g = &self.grid;
        let c = (p - g.origin) * g.inv_cell;
        if c.min_element() < 0.0 || c.max_element() >= (1u32 << COORD_BITS) as f32 {
            return &[];
        }
        match g.find(pack(c.x as u32, c.y as u32, c.z as u32)) {
            Some(s) => &g.items[g.start[s] as usize..(g.start[s] + g.len[s]) as usize],
            None => &[],
        }
    }

    /// Instances whose inflated AABB overlaps `b` (sorted, unique).
    pub fn candidates_in_box(&self, b: &Aabb, out: &mut Vec<u32>) {
        out.clear();
        if self.grid.keys.is_empty() || b.is_empty() || b.min.cmpgt(self.bounds.max).any() || b.max.cmplt(self.bounds.min).any() {
            return;
        }
        let (lo, hi) = self.grid.cell_range(b);
        let cells = (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64;
        if cells > 100_000 {
            out.extend((0..self.aabbs.len() as u32).filter(|&i| self.aabb_overlaps(i, b)));
            return;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    if let Some(s) = self.grid.find(pack(x, y, z)) {
                        out.extend_from_slice(&self.grid.items[self.grid.start[s] as usize..(self.grid.start[s] + self.grid.len[s]) as usize]);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out.retain(|&i| self.aabb_overlaps(i, b));
    }

    #[inline]
    fn aabb_overlaps(&self, i: u32, b: &Aabb) -> bool {
        let a = &self.aabbs[i as usize];
        !(a.min.cmpgt(b.max).any() || a.max.cmplt(b.min).any())
    }

    /// Does any instance (inflated) potentially overlap this box?
    pub fn touches(&self, b: &Aabb) -> bool {
        if self.grid.keys.is_empty() || b.is_empty() || b.min.cmpgt(self.bounds.max).any() || b.max.cmplt(self.bounds.min).any() {
            return false;
        }
        let (lo, hi) = self.grid.cell_range(b);
        let cells = (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64;
        if cells > 4096 {
            return true;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    if self.grid.find(pack(x, y, z)).is_some() {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline]
    pub fn sdf(&self, i: u32, p: Vec3) -> f32 {
        self.baked[i as usize].sdf(p)
    }

    #[inline]
    pub fn sdf_grad(&self, i: u32, p: Vec3) -> (f32, Vec3) {
        self.baked[i as usize].sdf_grad(p)
    }

    /// Smallest local half-size, i.e. the distance scale of instance `i`.
    #[inline]
    pub fn min_scale(&self, i: u32) -> f32 {
        self.baked[i as usize].min_scale
    }

    #[inline]
    pub fn inflated_aabb_contains(&self, i: u32, p: Vec3) -> bool {
        let a = &self.aabbs[i as usize];
        p.cmpge(a.min).all() && p.cmple(a.max).all()
    }
}

fn inflate(mut a: Aabb, m: f32) -> Aabb {
    a.min -= Vec3::splat(m);
    a.max += Vec3::splat(m);
    a
}

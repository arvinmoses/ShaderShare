//! Instances of primitives and the uniform-grid spatial hash over their AABBs.
//!
//! The hash is a sparse grid: only occupied cells exist, found by open
//! addressing, so rebuild cost scales with the number of instances (not with
//! the volume of the scene). Cell size is the mean instance extent, so an
//! instance covers at most ~8 cells.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
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
    /// Cells added since the last full rebuild by incrementally updated instances that left their
    /// registered cell range. Looked up after the static table.
    overlay: HashMap<u64, Vec<u32>, BuildHasherDefault<KeyHasher>>,
    overlay_entries: usize,
    csr_entries: usize,
}

/// Cheap hasher for already-mixed `u64` cell keys.
#[derive(Default)]
pub struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = hash(v);
    }
}

type CellRange = ([u32; 3], [u32; 3]);
/// Marks an inactive instance (no cells).
const NO_RANGE: CellRange = ([1; 3], [0; 3]);

#[inline]
fn range_is_empty(r: &CellRange) -> bool {
    r.0[0] > r.1[0] || r.0[1] > r.1[1] || r.0[2] > r.1[2]
}

#[inline]
fn range_contains(outer: &CellRange, inner: &CellRange) -> bool {
    (0..3).all(|k| outer.0[k] <= inner.0[k] && inner.1[k] <= outer.1[k])
}

#[inline]
fn range_cells(r: &CellRange) -> usize {
    if range_is_empty(r) { 0 } else { (0..3).map(|k| (r.1[k] - r.0[k] + 1) as usize).product() }
}

/// Instances found near a point: the static table's cell plus the overlay's cell.
#[derive(Clone, Copy)]
pub struct Cands<'a>(&'a [u32], &'a [u32]);

impl Cands<'_> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_empty()
    }
    pub fn contains(&self, id: &u32) -> bool {
        self.0.contains(id) || self.1.contains(id)
    }
}

impl<'a> IntoIterator for Cands<'a> {
    type Item = &'a u32;
    type IntoIter = std::iter::Chain<std::slice::Iter<'a, u32>, std::slice::Iter<'a, u32>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().chain(self.1.iter())
    }
}

/// What an incremental update changed, for dirty-leaf tracking.
#[derive(Clone, Debug)]
pub enum Dirty {
    /// Most instances changed; treat every leaf as dirty.
    All,
    /// Old and new inflated AABBs of the changed instances.
    Boxes(Vec<Aabb>),
}

/// Bookkeeping of the incremental hash, for the simulator and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct HashStats {
    /// Instances re-registered in the overlay since construction (cumulative).
    pub reindexed: u64,
    /// Full rebuilds triggered by an incremental update (overlay too big or left the grid domain).
    pub compactions: u64,
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
    /// Fraction of its extent an instance may grow or drift before it leaves its registered cell range.
    slack: f32,
    /// Cell range each instance is registered under in the static table (built from the slack-inflated box).
    reg: Vec<CellRange>,
    /// Range registered in the overlay, when the instance has left `reg`.
    ov: Vec<Option<CellRange>>,
    /// Deactivated instances (slot kept, no AABB).
    off: Vec<bool>,
    pub stats: HashStats,
}

impl InstanceSet {
    /// `margin` inflates every AABB (the boolean's reach), so a point within
    /// `margin` of an instance finds it.
    pub fn new(instances: Vec<Instance>, margin: f32) -> InstanceSet {
        Self::with_slack(instances, margin, 0.0)
    }

    /// Like [`InstanceSet::new`], registering every instance under cells covering its box inflated by up to
    /// `slack` of its extent (randomised per instance so re-registrations spread over frames). Instances
    /// updated through [`InstanceSet::update_incremental`] / [`InstanceSet::map_all`] that stay inside
    /// that range cost nothing in the hash.
    pub fn with_slack(instances: Vec<Instance>, margin: f32, slack: f32) -> InstanceSet {
        let mut s = InstanceSet {
            instances,
            baked: Vec::new(),
            aabbs: Vec::new(),
            margin,
            bounds: Aabb::EMPTY,
            grid: Grid::default(),
            slack,
            reg: Vec::new(),
            ov: Vec::new(),
            off: Vec::new(),
            stats: HashStats::default(),
        };
        s.rebuild();
        s
    }

    /// Bake transforms and AABBs only, with no spatial hash (for callers that index instances themselves,
    /// like the ADF boolean). Pair with [`InstanceSet::update_bake_only`].
    pub fn bake_only(instances: Vec<Instance>, margin: f32) -> InstanceSet {
        let n = instances.len();
        let baked = instances.par_iter().map(Baked::new).collect();
        let aabbs = instances.par_iter().map(|i| inflate(i.aabb(), margin)).collect();
        InstanceSet {
            instances,
            baked,
            aabbs,
            margin,
            bounds: Aabb::EMPTY,
            grid: Grid::default(),
            slack: 0.0,
            reg: vec![NO_RANGE; n],
            ov: vec![None; n],
            off: vec![false; n],
            stats: HashStats::default(),
        }
    }

    /// Replace a subset of instances without touching any hash; returns the old and new boxes (dirty regions).
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

    pub fn slack(&self) -> f32 {
        self.slack
    }

    /// Is the instance currently part of the set (not deactivated)?
    pub fn is_active(&self, i: usize) -> bool {
        !self.off[i]
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
        self.off.clear();
        self.rebuild();
    }

    /// Replace a subset of instances; returns the old and new inflated AABBs
    /// (feed them to [`super::affected_leaves`]). The grid is rebuilt.
    pub fn update(&mut self, changes: &[(usize, Instance)]) -> Vec<Aabb> {
        let mut dirty = Vec::with_capacity(changes.len() * 2);
        for &(i, inst) in changes {
            dirty.push(self.aabbs[i]);
            self.instances[i] = inst;
            self.baked[i] = Baked::new(&inst);
            self.aabbs[i] = inflate(inst.aabb(), self.margin);
            self.off[i] = false;
            dirty.push(self.aabbs[i]);
        }
        self.rebuild_grid();
        dirty
    }


    /// Registered box of instance `i`: its inflated AABB plus a per-instance random share of the slack.
    #[inline]
    fn reg_box(&self, i: usize, a: &Aabb) -> Aabb {
        reg_box_of(self.slack, i, a)
    }

    #[inline]
    fn range_of(&self, a: &Aabb) -> CellRange {
        if a.is_empty() { NO_RANGE } else { self.grid.cell_range(a) }
    }

    /// Replace a subset of instances and update the spatial hash **incrementally**: instances still inside
    /// their registered cell range cost nothing; the rest are re-registered in a small overlay. `deactivate`
    /// removes instances from the set (their slot stays; re-activate with a change). Falls back to a full
    /// grid rebuild when the overlay grows past a third of the table or an instance leaves the grid domain.
    pub fn update_incremental(&mut self, changes: &[(usize, Instance)], deactivate: &[usize]) -> Dirty {
        let mut dirty = Vec::with_capacity((changes.len() + deactivate.len()) * 2);
        let mut touched: Vec<u32> = Vec::with_capacity(changes.len() + deactivate.len());
        for &(i, inst) in changes {
            dirty.push(self.aabbs[i]);
            self.instances[i] = inst;
            self.baked[i] = Baked::new(&inst);
            self.aabbs[i] = inflate(inst.aabb(), self.margin);
            self.off[i] = false;
            dirty.push(self.aabbs[i]);
            touched.push(i as u32);
        }
        for &i in deactivate {
            dirty.push(self.aabbs[i]);
            self.aabbs[i] = Aabb::EMPTY;
            self.off[i] = true;
            touched.push(i as u32);
        }
        self.reindex(&touched);
        Dirty::Boxes(dirty)
    }

    /// Update every instance in parallel with `f(index, current) -> Some(new)`; `None` deactivates it
    /// (`Some` on a deactivated instance re-activates it). The hash is updated incrementally as in
    /// [`InstanceSet::update_incremental`]; instances that stay inside their registered cells cost no serial work.
    pub fn map_all(&mut self, f: &(impl Fn(usize, &Instance) -> Option<Instance> + Sync)) -> Dirty {
        let (margin, slack) = (self.margin, self.slack);
        let (grid, reg, ov) = (&self.grid, &self.reg, &self.ov);
        let has_grid = !grid.keys.is_empty();
        let escape = std::sync::atomic::AtomicBool::new(false);
        let (touched, bounds) = (&mut self.instances, &mut self.baked, &mut self.aabbs, &mut self.off)
            .into_par_iter()
            .enumerate()
            .fold(
                || (Vec::<u32>::new(), Aabb::EMPTY),
                |(mut t, mut b), (i, (inst, baked, aabb, off))| {
                    let (new_aabb, new_off) = match f(i, inst) {
                        Some(n) => {
                            *inst = n;
                            *baked = Baked::new(&n);
                            (inflate(n.aabb(), margin), false)
                        }
                        None => (Aabb::EMPTY, true),
                    };
                    *off = new_off;
                    let old = *aabb;
                    *aabb = new_aabb;
                    if old.min == new_aabb.min && old.max == new_aabb.max {
                        if !new_aabb.is_empty() {
                            b = b.union(&new_aabb);
                        }
                        return (t, b);
                    }
                    if new_aabb.is_empty() {
                        if ov[i].is_some() {
                            t.push(i as u32);
                        }
                        return (t, b);
                    }
                    b = b.union(&new_aabb);
                    if has_grid {
                        let rb = reg_box_of(slack, i, &new_aabb);
                        if rb.min.cmplt(grid.origin).any() {
                            escape.store(true, Relaxed);
                        }
                        let need = grid.cell_range(&new_aabb);
                        let covered = range_contains(&reg[i], &need);
                        if covered {
                            if ov[i].is_some() {
                                t.push(i as u32);
                            }
                        } else if !ov[i].is_some_and(|o| range_contains(&o, &need)) {
                            t.push(i as u32);
                        }
                    }
                    (t, b)
                },
            )
            .reduce(
                || (Vec::new(), Aabb::EMPTY),
                |mut a, b| {
                    a.0.extend(b.0);
                    (a.0, a.1.union(&b.1))
                },
            );
        self.bounds = self.bounds.union(&bounds);
        if !has_grid || escape.load(Relaxed) {
            if has_grid {
                self.stats.compactions += 1;
            }
            self.rebuild_grid();
        } else {
            let mut touched = touched;
            touched.sort_unstable();
            self.reindex(&touched);
        }
        Dirty::All
    }

    /// Multiply the scale of every active instance by `ratio` (a global size slider), keeping each instance's
    /// centre. Much cheaper than [`InstanceSet::map_all`] because rotations, bake matrices and the box centre
    /// are untouched. The hash is updated incrementally like `map_all`.
    pub fn rescale_all(&mut self, ratio: f32) -> Dirty {
        let (margin, slack) = (self.margin, self.slack);
        let (grid, reg, ov) = (&self.grid, &self.reg, &self.ov);
        let has_grid = !grid.keys.is_empty();
        let escape = std::sync::atomic::AtomicBool::new(false);
        let (touched, bounds) = (&mut self.instances, &mut self.baked, &mut self.aabbs, &self.off)
            .into_par_iter()
            .enumerate()
            .fold(
                || (Vec::<u32>::new(), Aabb::EMPTY),
                |(mut t, mut b), (i, (inst, baked, aabb, off))| {
                    if *off {
                        return (t, b);
                    }
                    inst.scale *= ratio;
                    baked.inv_scale /= ratio;
                    baked.min_scale *= ratio;
                    let c = aabb.center();
                    let h = (aabb.max - aabb.min) * 0.5 - Vec3::splat(margin);
                    let nh = h * ratio + Vec3::splat(margin);
                    *aabb = Aabb { min: c - nh, max: c + nh };
                    b = b.union(aabb);
                    if has_grid {
                        let rb = reg_box_of(slack, i, aabb);
                        if rb.min.cmplt(grid.origin).any() {
                            escape.store(true, Relaxed);
                        }
                        let need = grid.cell_range(aabb);
                        if range_contains(&reg[i], &need) {
                            if ov[i].is_some() {
                                t.push(i as u32);
                            }
                        } else if !ov[i].is_some_and(|o| range_contains(&o, &need)) {
                            t.push(i as u32);
                        }
                    }
                    (t, b)
                },
            )
            .reduce(
                || (Vec::new(), Aabb::EMPTY),
                |mut a, b| {
                    a.0.extend(b.0);
                    (a.0, a.1.union(&b.1))
                },
            );
        self.bounds = self.bounds.union(&bounds);
        if !has_grid || escape.load(Relaxed) {
            if has_grid {
                self.stats.compactions += 1;
            }
            self.rebuild_grid();
        } else {
            let mut touched = touched;
            touched.sort_unstable();
            self.reindex(&touched);
        }
        Dirty::All
    }

    /// Bring the hash in line with the current AABBs of `touched` instances.
    fn reindex(&mut self, touched: &[u32]) {
        if self.grid.keys.is_empty() {
            self.rebuild_grid();
            return;
        }
        let mut domain_escape = false;
        // Re-registering is serial hash work: past this many instances a full (parallel) rebuild is cheaper.
        let limit = 2048.max(self.instances.len() / 8);
        let mut moved = 0usize;
        for &t in touched {
            if moved > limit {
                domain_escape = true;
                break;
            }
            let i = t as usize;
            let a = self.aabbs[i];
            if !a.is_empty() {
                self.bounds = self.bounds.union(&a);
            }
            let rb = self.reg_box(i, &a);
            if !a.is_empty() && rb.min.cmplt(self.grid.origin).any() {
                domain_escape = true;
                continue;
            }
            let need = self.range_of(&a);
            let in_csr = range_is_empty(&need) || range_contains(&self.reg[i], &need);
            if in_csr {
                self.overlay_remove(i);
            } else if self.ov[i].is_some_and(|o| range_contains(&o, &need)) {
                // already covered by the overlay registration
            } else {
                self.overlay_remove(i);
                let r = self.range_of(&rb);
                self.overlay_insert(i, r);
                self.stats.reindexed += 1;
                moved += 1;
            }
        }
        if domain_escape || self.grid.overlay_entries > 2048.max(self.grid.csr_entries / 3) {
            self.stats.compactions += 1;
            self.rebuild_grid();
        }
    }

    fn overlay_remove(&mut self, i: usize) {
        if let Some(r) = self.ov[i].take() {
            for z in r.0[2]..=r.1[2] {
                for y in r.0[1]..=r.1[1] {
                    for x in r.0[0]..=r.1[0] {
                        if let Some(l) = self.grid.overlay.get_mut(&pack(x, y, z))
                            && let Some(at) = l.iter().position(|&v| v == i as u32)
                        {
                            l.swap_remove(at);
                            self.grid.overlay_entries -= 1;
                        }
                    }
                }
            }
        }
    }

    fn overlay_insert(&mut self, i: usize, r: CellRange) {
        for z in r.0[2]..=r.1[2] {
            for y in r.0[1]..=r.1[1] {
                for x in r.0[0]..=r.1[0] {
                    self.grid.overlay.entry(pack(x, y, z)).or_default().push(i as u32);
                    self.grid.overlay_entries += 1;
                }
            }
        }
        self.ov[i] = Some(r);
    }

    /// Entries in the overlay and in the static table (to judge compaction).
    pub fn overlay_len(&self) -> (usize, usize) {
        (self.grid.overlay_entries, self.grid.csr_entries)
    }

    /// Fold the overlay back into the static table (a full grid rebuild) if it holds more than `frac` of it.
    pub fn compact_if(&mut self, frac: f32) -> bool {
        if self.grid.overlay_entries as f32 > frac * self.grid.csr_entries as f32 {
            self.stats.compactions += 1;
            self.rebuild_grid();
            return true;
        }
        false
    }

    /// Re-bake every instance and rebuild the grid.
    pub fn rebuild(&mut self) {
        let m = self.margin;
        self.baked = self.instances.par_iter().map(Baked::new).collect();
        self.off.resize(self.instances.len(), false);
        let off = &self.off;
        self.aabbs = self.instances.par_iter().enumerate().map(|(i, x)| if off[i] { Aabb::EMPTY } else { inflate(x.aabb(), m) }).collect();
        self.rebuild_grid();
    }

    /// Rebuild only the spatial hash from the current AABBs.
    pub fn rebuild_grid(&mut self) {
        let n = self.aabbs.len();
        let active = self.aabbs.par_iter().filter(|a| !a.is_empty()).count();
        self.ov = vec![None; n];
        if active == 0 {
            self.grid = Grid::default();
            self.bounds = Aabb::EMPTY;
            self.reg = vec![NO_RANGE; n];
            return;
        }
        let (bounds, ext_sum) = self
            .aabbs
            .par_iter()
            .filter(|a| !a.is_empty())
            .fold(
                || (Aabb::EMPTY, 0.0f64),
                |(b, s), a| (b.union(a), s + a.extent().max_element() as f64),
            )
            .reduce(|| (Aabb::EMPTY, 0.0), |a, b| (a.0.union(&b.0), a.1 + b.1));
        self.bounds = bounds;
        let mean_ext = (ext_sum / active as f64) as f32;
        let cell = mean_ext.max(bounds.extent().max_element() / (1u32 << (COORD_BITS - 1)) as f32).max(1e-9);
        // Pad the origin so instances that drift or grow slightly past the bounds stay inside the domain.
        let pad = if self.slack > 0.0 { Vec3::splat(2.0 * cell) } else { Vec3::ZERO };
        let mut g = Grid { origin: bounds.min - pad, inv_cell: 1.0 / cell, cell, ..Default::default() };

        let this = &*self;
        let ranges: Vec<CellRange> = this
            .aabbs
            .par_iter()
            .enumerate()
            .map(|(i, a)| if a.is_empty() { NO_RANGE } else { g.cell_range(&this.reg_box(i, a)) })
            .collect();
        let entries: usize = ranges.par_iter().map(range_cells).sum();
        // Distinct cells are bounded by both the entry count and the cells in the bounds
        // (instances overlap heavily on a surface, so the latter is usually far smaller).
        let (rlo, rhi) = ranges
            .par_iter()
            .filter(|r| !range_is_empty(r))
            .fold(
                || ([u32::MAX; 3], [0u32; 3]),
                |(mut lo, mut hi), r| {
                    for k in 0..3 {
                        lo[k] = lo[k].min(r.0[k]);
                        hi[k] = hi[k].max(r.1[k]);
                    }
                    (lo, hi)
                },
            )
            .reduce(
                || ([u32::MAX; 3], [0u32; 3]),
                |a, b| ([a.0[0].min(b.0[0]), a.0[1].min(b.0[1]), a.0[2].min(b.0[2])], [a.1[0].max(b.1[0]), a.1[1].max(b.1[1]), a.1[2].max(b.1[2])]),
            );
        let cells_in_bounds = (0..3).map(|k| (rhi[k].saturating_sub(rlo[k]) as u64) + 1).fold(1u64, |a, b| a.saturating_mul(b)) as usize;
        let distinct_cap = entries.min(cells_in_bounds);
        let cap = (distinct_cap * 2).next_power_of_two().max(16);
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
        g.csr_entries = entries;
        self.grid = g;
        self.reg = ranges;
    }

    /// Instances whose inflated AABB overlaps the cell containing `p`
    /// (a superset of the instances that can influence `p`).
    #[inline]
    pub fn candidates(&self, p: Vec3) -> Cands<'_> {
        let g = &self.grid;
        let c = (p - g.origin) * g.inv_cell;
        if c.min_element() < 0.0 || c.max_element() >= (1u32 << COORD_BITS) as f32 {
            return Cands(&[], &[]);
        }
        let key = pack(c.x as u32, c.y as u32, c.z as u32);
        let a: &[u32] = match g.find(key) {
            Some(s) => &g.items[g.start[s] as usize..(g.start[s] + g.len[s]) as usize],
            None => &[],
        };
        let b: &[u32] = if g.overlay.is_empty() { &[] } else { g.overlay.get(&key).map_or(&[], |v| v.as_slice()) };
        Cands(a, b)
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
                    let key = pack(x, y, z);
                    if self.grid.find(key).is_some_and(|s| self.grid.len[s] > 0) || self.grid.overlay.get(&key).is_some_and(|v| !v.is_empty()) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline]
    fn box_hits(&self, i: u32, b: &Aabb) -> bool {
        let a = &self.aabbs[i as usize];
        !(a.min.cmpgt(b.max).any() || a.max.cmplt(b.min).any())
    }

    /// All instances whose inflated AABB overlaps `b`, sorted by index (static table plus overlay).
    pub fn candidates_in_box(&self, b: &Aabb, out: &mut Vec<u32>) {
        out.clear();
        if self.grid.keys.is_empty() || b.is_empty() || b.min.cmpgt(self.bounds.max).any() || b.max.cmplt(self.bounds.min).any() {
            return;
        }
        let (lo, hi) = self.grid.cell_range(b);
        let cells = (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64;
        if cells > 100_000 {
            out.extend((0..self.aabbs.len() as u32).filter(|&i| self.box_hits(i, b)));
            return;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let key = pack(x, y, z);
                    if let Some(s) = self.grid.find(key) {
                        out.extend_from_slice(&self.grid.items[self.grid.start[s] as usize..(self.grid.start[s] + self.grid.len[s]) as usize]);
                    }
                    if let Some(v) = self.grid.overlay.get(&key) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out.retain(|&i| self.box_hits(i, b));
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

/// Box an instance is registered under: its AABB plus a per-instance random share of `slack`.
#[inline]
fn reg_box_of(slack: f32, i: usize, a: &Aabb) -> Aabb {
    if slack <= 0.0 || a.is_empty() {
        return *a;
    }
    let h = (hash(i as u64 ^ 0xA5A5) & 0xFFFF) as f32 / 65535.0;
    let pad = Vec3::splat(slack * (0.5 + h) * a.extent().max_element());
    Aabb { min: a.min - pad, max: a.max + pad }
}

fn inflate(mut a: Aabb, m: f32) -> Aabb {
    a.min -= Vec3::splat(m);
    a.max += Vec3::splat(m);
    a
}

//! SPIKE 4 (measurement only): a volume held at several resolutions, so detail exists only where it was sculpted and
//! can be shown or hidden by distance, the way an adaptive distance field gives a drawing program detail at any zoom.
//!
//! The question it answers: what storage layout should the GPU engine be built on? It implements just enough to judge
//! one candidate honestly:
//!
//! * **Levels.** Level 0 is the coarse field; each level above halves the lattice step. Every level uses 8x8x8 blocks
//!   on one shared lattice (level `l + 1`'s point `2i` is level `l`'s point `i`), so each coarse cell splits exactly into
//!   eight finer cells and a finer block covers a 4x4x4 patch of coarser cells. A block exists at a level only where
//!   something was sculpted at that level or finer. Where a level has no block, its values are the coarser level's,
//!   interpolated.
//! * **Values, not residuals.** Each level stores the field itself, and holds only what was sculpted at it or at a
//!   coarser level (rule B). After a dab at level `l`, every finer level's own detail (its difference from the level
//!   below) moves with the surface, along the normal by as far as the edit moved it, so fine detail rides on a large
//!   edit. A view capped below the finest level adds the finer levels' detail back, filtered to its step, when it is
//!   drawn. (Rule A, [`Levels::restrict`], instead passes each change down to the coarser levels, filtered: a large
//!   edit then replaces the coarse copy of finer detail, and most of that detail is lost.)
//! * **Surface.** Dual contouring over the tree of cells: one vertex per smallest cell the surface touches (at the mean
//!   of the crossings around it), one quad (or triangle, where two of its cells are one coarser cell) per smallest
//!   crossing lattice edge. A lattice point shared by several levels takes the finest one's value, so cells of
//!   different levels agree on every crossing and the surface closes. A level cap gives what a distant view would draw.
//!
//! Left out: the manifold pieces and checkerboard cleanup of `volume.rs`, plane brushes, undo, the GPU. Each level's
//! samples are clamped to its own band (2.5 of its lattice steps, 8 at level 0), as in `volume.rs`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use crate::mesh::{NO_VERT, PolyMesh};

const B: i32 = 8;
const N: usize = 512;

type Block = Box<[f32; N]>;
/// A smallest crossing edge: its level, start and axis, the four cells around it, where it crosses, and whether its
/// start is inside.
type Crossing = (usize, IVec3, usize, [(usize, IVec3); 4], Vec3, bool);

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(IVec3::splat(B));
    (l.z * 64 + l.y * 8 + l.x) as usize
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    -smin(-a, -b, k)
}

/// A sparse distance field at several resolutions.
pub struct Levels {
    pub voxel0: f32,
    origin: Vec3,
    /// Stored blocks per level, by block coordinates.
    blocks: Vec<HashMap<IVec3, Block>>,
    /// Level 0 blocks wholly inside the solid (any other level 0 block not stored is wholly outside).
    inside0: HashSet<IVec3>,
    /// Whether a dab's change is passed, filtered, to the coarser levels (rule A). Without it (rule B) each level holds
    /// only what was sculpted at it or coarser, and a distant view would have to filter the finer levels when drawn.
    pub restrict: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LevelDab {
    pub ms: f64,
    /// Blocks changed at the dab's level, blocks created there, and blocks updated at other levels.
    pub changed: usize,
    pub created: usize,
    pub others: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LevelSurface {
    pub ms: f64,
    pub quads: usize,
    pub triangles: usize,
    /// Edges with one face (a crack or hole) and with more than two.
    pub open_edges: usize,
    pub non_manifold_edges: usize,
    /// Quads per level of the cells they join (the finest of their four).
    pub per_level: [usize; 8],
}

impl Levels {
    pub fn levels(&self) -> usize {
        self.blocks.len()
    }

    pub fn voxel(&self, l: usize) -> f32 {
        self.voxel0 / (1u32 << l) as f32
    }

    /// Distances kept at level `l`: 2.5 lattice steps, but 8 at level 0, so a large edit can carry finer detail as
    /// far as it moves the surface.
    pub fn band(&self, l: usize) -> f32 {
        if l == 0 { 8.0 * self.voxel0 } else { 2.5 * self.voxel(l) }
    }

    /// Stored blocks per level, and the bytes they take at 16 bits a sample.
    pub fn memory(&self) -> (Vec<usize>, usize) {
        let counts: Vec<usize> = self.blocks.iter().map(|m| m.len()).collect();
        let bytes = counts.iter().sum::<usize>() * N * 2;
        (counts, bytes)
    }

    /// `field` over `[-half, half]^3` at level 0's step `voxel0`, with `levels` levels in all (the finer ones empty).
    pub fn from_field<F: Fn(Vec3) -> f32 + Sync>(field: &F, half: f32, voxel0: f32, levels: usize) -> Levels {
        let origin = Vec3::splat(-half - voxel0 * B as f32);
        let n = ((2.0 * half / voxel0) / B as f32).ceil() as i32 + 2;
        let band = 8.0 * voxel0;
        let reach = 0.5 * B as f32 * voxel0 * 3f32.sqrt() + band;
        let coords: Vec<IVec3> = (0..n * n * n).map(|b| IVec3::new(b % n, (b / n) % n, b / (n * n))).collect();
        let built: Vec<(IVec3, Option<Block>, bool)> = coords
            .par_iter()
            .map(|&bc| {
                let lo = origin + (bc * B).as_vec3() * voxel0;
                let c = field(lo + Vec3::splat(0.5 * (B - 1) as f32 * voxel0));
                if c.abs() > reach {
                    return (bc, None, c < 0.0);
                }
                let mut s = Box::new([0.0f32; N]);
                for (i, v) in s.iter_mut().enumerate() {
                    let l = IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                    *v = field(lo + l.as_vec3() * voxel0).clamp(-band, band);
                }
                (bc, Some(s), false)
            })
            .collect();
        let mut v = Levels { voxel0, origin, blocks: vec![HashMap::new(); levels], inside0: HashSet::new(), restrict: false };
        for (bc, s, inside) in built {
            match s {
                Some(s) => {
                    v.blocks[0].insert(bc, s);
                }
                None if inside => {
                    v.inside0.insert(bc);
                }
                None => {}
            }
        }
        v
    }

    /// The sample at level `l`'s lattice point `i`: stored, or else the coarser level's, interpolated.
    pub fn sample(&self, l: usize, i: IVec3) -> f32 {
        let bc = i.div_euclid(IVec3::splat(B));
        if let Some(b) = self.blocks[l].get(&bc) {
            return b[local(i)];
        }
        if l == 0 {
            return if self.inside0.contains(&bc) { -self.band(0) } else { self.band(0) };
        }
        let band = self.band(l);
        if i.x & 1 == 0 && i.y & 1 == 0 && i.z & 1 == 0 {
            return self.sample(l - 1, i / 2).clamp(-band, band);
        }
        self.value_at(l - 1, i.as_vec3() * 0.5).clamp(-band, band)
    }

    /// Level `l`'s field at continuous lattice coordinates `g` (in level `l` steps), interpolated.
    fn value_at(&self, l: usize, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let s = |dx: i32, dy: i32, dz: i32| self.sample(l, i + IVec3::new(dx, dy, dz));
        let x00 = s(0, 0, 0) + (s(1, 0, 0) - s(0, 0, 0)) * f.x;
        let x10 = s(0, 1, 0) + (s(1, 1, 0) - s(0, 1, 0)) * f.x;
        let x01 = s(0, 0, 1) + (s(1, 0, 1) - s(0, 0, 1)) * f.x;
        let x11 = s(0, 1, 1) + (s(1, 1, 1) - s(0, 1, 1)) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        y0 + (y1 - y0) * f.z
    }

    /// The field at world point `p`, from the finest level that holds it.
    pub fn value(&self, p: Vec3) -> f32 {
        self.value_upto(p, self.levels() - 1)
    }

    /// [`Self::value`] using levels up to `max` only.
    pub fn value_upto(&self, p: Vec3, max: usize) -> f32 {
        for l in (0..=max.min(self.levels() - 1)).rev() {
            let g = (p - self.origin) / self.voxel(l);
            if l == 0 || self.blocks[l].contains_key(&g.floor().as_ivec3().div_euclid(IVec3::splat(B))) {
                return self.value_at(l, g);
            }
        }
        unreachable!()
    }

    /// Outward normal at `p`, from the finest level there.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        let (l, _) = self.leaf(p, self.levels() - 1);
        let h = self.voxel(l);
        let d = |o: Vec3| self.value(p + o) - self.value(p - o);
        Vec3::new(d(Vec3::X * h), d(Vec3::Y * h), d(Vec3::Z * h)).normalize_or_zero()
    }

    /// Where a ray from `origin` along `dir` (normalised) first enters the solid, with the normal there: marched on
    /// level 0 (whose band reaches farthest), then narrowed down on the finest field.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        self.raycast_upto(origin, dir, self.levels() - 1)
    }

    /// [`Self::raycast`] against the surface of levels up to `max` only (what a stroke at a finer level started on).
    pub fn raycast_upto(&self, origin: Vec3, dir: Vec3, max: usize) -> Option<(Vec3, Vec3)> {
        let value = |p: Vec3| self.value_upto(p, max);
        let v0 = self.voxel0;
        let coarse = |p: Vec3| self.value_at(0, (p - self.origin) / v0);
        let (mut t, mut prev) = (0.0f32, 0.0f32);
        let far = 20.0;
        let mut hit = false;
        while t < far {
            let v = coarse(origin + dir * t);
            // Close to the coarse surface, switch to the finest field, which detail may have moved.
            if v < 2.0 * v0 {
                let mut s = t;
                while s < t + 6.0 * v0 {
                    if value(origin + dir * s) < 0.0 {
                        hit = true;
                        t = s;
                        break;
                    }
                    prev = s;
                    s += 0.25 * self.voxel(self.levels() - 1).max(0.1 * v0);
                }
                if hit {
                    break;
                }
                t = s;
                continue;
            }
            prev = t;
            t += v.max(0.3 * v0);
        }
        if !hit {
            return None;
        }
        let (mut a, mut b) = (prev, t);
        for _ in 0..16 {
            let m = 0.5 * (a + b);
            if value(origin + dir * m) < 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        let p = origin + dir * b;
        Some((p, self.normal(p)))
    }

    /// A block of level `l`, made from the coarser level if it does not exist yet (true if it was made). Its parent
    /// at the level below is made first: the levels nest, which the surface relies on to join them without cracks.
    fn ensure(&mut self, l: usize, bc: IVec3) -> bool {
        if self.blocks[l].contains_key(&bc) {
            return false;
        }
        if l > 1 {
            self.ensure(l - 1, (bc * B).div_euclid(IVec3::splat(2 * B)));
        }
        let mut s = Box::new([0.0f32; N]);
        for (i, v) in s.iter_mut().enumerate() {
            let p = bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
            *v = self.sample(l, p);
        }
        self.blocks[l].insert(bc, s);
        true
    }

    /// Add (or with `carve`, remove) a soft ball of radius `r` at `c`, blended over `k`, written at level `l`.
    pub fn dab(&mut self, l: usize, c: Vec3, r: f32, k: f32, carve: bool) -> LevelDab {
        let t = Instant::now();
        let mut stats = LevelDab::default();
        let (vx, band) = (self.voxel(l), self.band(l));
        let step = vx * B as f32;
        let reach = r + k + band;
        let lo = ((c - Vec3::splat(reach) - self.origin) / step).floor().as_ivec3();
        let hi = ((c + Vec3::splat(reach) - self.origin) / step).floor().as_ivec3();
        let half = 0.5 * (B - 1) as f32 * vx;
        let diag = half * 3f32.sqrt();
        // Blocks the ball's surface passes near, and that hold surface or will: no finer blocks deep inside or far out.
        let mut touched = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let bc = IVec3::new(x, y, z);
                    let centre = self.origin + (bc * B).as_vec3() * vx + Vec3::splat(half);
                    let ball = (centre - c).length() - r;
                    if ball.abs() > diag + band + k {
                        continue;
                    }
                    if ball > diag && self.value(centre).abs() > diag + band {
                        continue;
                    }
                    touched.push(bc);
                }
            }
        }
        let mut created = HashSet::new();
        for &bc in &touched {
            if self.ensure(l, bc) {
                created.insert(bc);
            }
        }
        // The finer levels around the dab, as they were, to carry their detail from.
        let scale = |m: usize| 1i32 << (m - l);
        let old: Vec<HashMap<IVec3, Block>> = (0..self.levels())
            .map(|m| {
                if m < l {
                    return HashMap::new();
                }
                if m == l {
                    return touched.iter().map(|bc| (*bc, self.blocks[l][bc].clone())).collect();
                }
                let (a, b) = (lo * scale(m), (hi + IVec3::ONE) * scale(m));
                self.blocks[m].iter().filter(|(bc, _)| bc.cmpge(a).all() && bc.cmplt(b).all()).map(|(bc, s)| (*bc, s.clone())).collect()
            })
            .collect();
        // The change at level `l`, kept to carry it to the finer levels.
        let origin = self.origin;
        let mut delta: HashMap<IVec3, Block> = HashMap::new();
        let mut work: Vec<(IVec3, Block)> = touched.iter().map(|bc| (*bc, self.blocks[l].remove(bc).unwrap())).collect();
        let deltas: Vec<Option<Block>> = work
            .par_iter_mut()
            .map(|(bc, s)| {
                let mut d = Box::new([0.0f32; N]);
                let mut any = false;
                for (i, v) in s.iter_mut().enumerate() {
                    let p = origin + (*bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32)).as_vec3() * vx;
                    let blob = (p - c).length() - r;
                    let next = (if carve { smax(*v, -blob, k) } else { smin(*v, blob, k) }).clamp(-band, band);
                    if next != *v {
                        d[i] = next - *v;
                        *v = next;
                        any = true;
                    }
                }
                any.then_some(d)
            })
            .collect();
        for ((bc, s), d) in work.into_iter().zip(deltas) {
            match d {
                Some(d) => {
                    self.blocks[l].insert(bc, s);
                    delta.insert(bc, d);
                }
                // A block made for the dab that it did not change only repeats the coarser level: drop it.
                None if created.contains(&bc) => {}
                None => {
                    self.blocks[l].insert(bc, s);
                }
            }
        }
        stats.changed = delta.len();
        stats.created = created.iter().filter(|bc| delta.contains_key(bc)).count();
        // Coarser levels take the change, filtered: each coarse point the weighted mean (1-2-1 per axis) of the change
        // at the finer points around it. A distant view sees the change smoothed, and detail finer than its step fades
        // instead of aliasing; unchanged places stay exactly as they were.
        let mut fine_delta: HashMap<IVec3, Block> = if self.restrict { delta.clone() } else { HashMap::new() };
        for m in (0..l).rev() {
            let fd = |p: IVec3| fine_delta.get(&p.div_euclid(IVec3::splat(B))).map_or(0.0, |d| d[local(p)]);
            // Coarse points whose filter reaches a changed fine block: 4b..4b+4 per axis, so up to 8 coarse blocks.
            let mut under: Vec<IVec3> = Vec::new();
            for fb in fine_delta.keys() {
                for k in 0..8 {
                    let corner = *fb * 4 + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1) * 4;
                    under.push(corner.div_euclid(IVec3::splat(B)));
                }
            }
            under.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            under.dedup();
            let changes: Vec<(IVec3, Block)> = under
                .par_iter()
                .filter_map(|&cb| {
                    let mut d = Box::new([0.0f32; N]);
                    let mut any = false;
                    for (i, v) in d.iter_mut().enumerate() {
                        let c = cb * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                        let mut sum = 0.0;
                        for z in -1..=1i32 {
                            for y in -1..=1i32 {
                                for x in -1..=1i32 {
                                    let w = ((2 - x.abs()) * (2 - y.abs()) * (2 - z.abs())) as f32;
                                    sum += w * fd(c * 2 + IVec3::new(x, y, z));
                                }
                            }
                        }
                        if sum != 0.0 {
                            *v = sum / 64.0;
                            any = true;
                        }
                    }
                    any.then_some((cb, d))
                })
                .collect();
            let cband = self.band(m);
            for (cb, d) in &changes {
                if m == 0 && !self.blocks[0].contains_key(cb) {
                    let fill = if self.inside0.remove(cb) { -self.band(0) } else { self.band(0) };
                    self.blocks[0].insert(*cb, Box::new([fill; N]));
                } else {
                    self.ensure(m, *cb);
                }
                let coarse = self.blocks[m].get_mut(cb).unwrap();
                for (v, dv) in coarse.iter_mut().zip(d.iter()) {
                    *v = (*v + dv).clamp(-cband, cband);
                }
                stats.others += 1;
            }
            fine_delta = changes.into_iter().collect();
        }
        // Finer levels: each one's own detail (its difference from the level below, as both were) moves with the
        // surface: along the normal, as far as this dab moved it. Coarsest first, so each builds on the new level below.
        let dvalue = |g: Vec3| -> f32 {
            // The change at level `l`, at `g` in level `l` steps (zero where no block changed).
            let i = g.floor().as_ivec3();
            let f = g - i.as_vec3();
            let s = |dx: i32, dy: i32, dz: i32| {
                let p = i + IVec3::new(dx, dy, dz);
                delta.get(&p.div_euclid(IVec3::splat(B))).map_or(0.0, |d| d[local(p)])
            };
            trilinear(f, [s(0, 0, 0), s(1, 0, 0), s(0, 1, 0), s(1, 1, 0), s(0, 0, 1), s(1, 0, 1), s(0, 1, 1), s(1, 1, 1)])
        };
        for m in l + 1..self.levels() {
            if old[m].is_empty() {
                continue;
            }
            let k = scale(m) as f32;
            let (vm, fband) = (self.voxel(m), self.band(m));
            // Where the detail goes: the blocks it was in, and every block its box covers once moved with the surface.
            let dest: Vec<IVec3> = {
            let me = &*self;
            let normal = |g: Vec3| -> Vec3 {
                let d = |o: Vec3| me.value_at(l, g + o) - me.value_at(l, g - o);
                Vec3::new(d(Vec3::X), d(Vec3::Y), d(Vec3::Z)).normalize_or_zero()
            };
            let mut dest: Vec<IVec3> = old[m].keys().copied().collect();
            for bc in old[m].keys() {
                for c in 0..9 {
                    let corner = if c == 8 { Vec3::splat(0.5) } else { Vec3::new((c & 1) as f32, ((c >> 1) & 1) as f32, ((c >> 2) & 1) as f32) };
                    let at = (*bc * B).as_vec3() + corner * B as f32;
                    let d = dvalue(at / k);
                    if d != 0.0 {
                        let to = at + normal(at / k) * (-d / vm);
                        dest.push(to.floor().as_ivec3().div_euclid(IVec3::splat(B)));
                    }
                }
            }
            dest.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            dest.dedup();
            dest
            };
            // New blocks (and their parents) exist before the new values are worked out, reading the level below.
            for &bc in &dest {
                if m > 1 {
                    self.ensure(m - 1, (bc * B).div_euclid(IVec3::splat(2 * B)));
                }
            }
            let me = &*self;
            // Level `m - 1` as it was, at level `m` lattice point `i`.
            let old_below = |i: IVec3| -> f32 {
                let g = i.as_vec3() * 0.5;
                let j = g.floor().as_ivec3();
                let f = g - j.as_vec3();
                let s = |dx: i32, dy: i32, dz: i32| {
                    let p = j + IVec3::new(dx, dy, dz);
                    old[m - 1].get(&p.div_euclid(IVec3::splat(B))).map_or_else(|| me.sample(m - 1, p), |b| b[local(p)])
                };
                trilinear(f, [s(0, 0, 0), s(1, 0, 0), s(0, 1, 0), s(1, 1, 0), s(0, 0, 1), s(1, 0, 1), s(0, 1, 1), s(1, 1, 1)])
            };
            // Level `m`'s detail as it was, at `q` in level `m` steps: zero where it held no block.
            let detail = |q: Vec3| -> f32 {
                let j = q.floor().as_ivec3();
                let f = q - j.as_vec3();
                let s = |dx: i32, dy: i32, dz: i32| {
                    let p = j + IVec3::new(dx, dy, dz);
                    // Both clamped to this level's band: where either saturates (away from the surface) there is no
                    // detail, only the two bands' difference.
                    old[m].get(&p.div_euclid(IVec3::splat(B))).map_or(0.0, |b| b[local(p)] - old_below(p).clamp(-fband, fband))
                };
                trilinear(f, [s(0, 0, 0), s(1, 0, 0), s(0, 1, 0), s(1, 1, 0), s(0, 0, 1), s(1, 0, 1), s(0, 1, 1), s(1, 1, 1)])
            };
            let normal = |g: Vec3| -> Vec3 {
                let d = |o: Vec3| me.value_at(l, g + o) - me.value_at(l, g - o);
                Vec3::new(d(Vec3::X), d(Vec3::Y), d(Vec3::Z)).normalize_or_zero()
            };
            let built: Vec<(IVec3, Option<Block>)> = dest
                .par_iter()
                .map(|&bc| {
                    let mut s: Block = me.blocks[m].get(&bc).cloned().unwrap_or_else(|| {
                        let mut s = Box::new([0.0f32; N]);
                        for (i, v) in s.iter_mut().enumerate() {
                            *v = me.sample(m, bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32));
                        }
                        s
                    });
                    let mut any = false;
                    for (i, v) in s.iter_mut().enumerate() {
                        let p = bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                        let g = p.as_vec3() / k;
                        let d = dvalue(g);
                        if d == 0.0 {
                            continue;
                        }
                        // The surface moved out by `-d` here: the detail now here was `-d` farther in.
                        let q = p.as_vec3() + normal(g) * (d / vm);
                        let next = (me.value_at(m - 1, p.as_vec3() * 0.5) + detail(q)).clamp(-fband, fband);
                        if next != *v {
                            *v = next;
                            any = true;
                        }
                    }
                    (bc, any.then_some(s))
                })
                .collect();
            for (bc, s) in built {
                if let Some(s) = s {
                    self.blocks[m].insert(bc, s);
                    stats.others += 1;
                }
            }
        }
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        stats
    }

    /// The smallest cell at world point `p`, among levels up to `max`: its level and lattice coordinates.
    fn leaf(&self, p: Vec3, max: usize) -> (usize, IVec3) {
        for l in (0..=max).rev() {
            let cell = ((p - self.origin) / self.voxel(l)).floor().as_ivec3();
            if l == 0 || self.blocks[l].contains_key(&cell.div_euclid(IVec3::splat(B))) {
                return (l, cell);
            }
        }
        unreachable!()
    }

    /// The detail of the levels finer than `cap`, filtered to `cap`'s step: per `cap` block, what to add to its samples
    /// so a view capped there still shows the finer work, smoothed. Each finer level's own detail (its values less the
    /// level below, both within its band) is folded down one level at a time with the 1-2-1 filter.
    fn folded_detail(&self, cap: usize) -> HashMap<IVec3, Block> {
        let mut acc: HashMap<IVec3, Block> = HashMap::new();
        for m in (cap + 1..self.levels()).rev() {
            let band = self.band(m);
            // This level's detail, plus what was folded into it from finer levels.
            let cur: HashMap<IVec3, Block> = self.blocks[m]
                .par_iter()
                .map(|(bc, s)| {
                    let mut r = Box::new([0.0f32; N]);
                    for (i, v) in r.iter_mut().enumerate() {
                        let p = *bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                        *v = s[i] - self.value_at(m - 1, p.as_vec3() * 0.5).clamp(-band, band) + acc.get(bc).map_or(0.0, |a| a[i]);
                    }
                    (*bc, r)
                })
                .collect();
            let at = |p: IVec3| cur.get(&p.div_euclid(IVec3::splat(B))).map_or(0.0, |d| d[local(p)]);
            let mut under: Vec<IVec3> = cur.keys().flat_map(|fb| (0..8).map(move |k| (*fb * 4 + IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1) * 4).div_euclid(IVec3::splat(B)))).collect();
            under.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            under.dedup();
            acc = under
                .par_iter()
                .filter_map(|&cb| {
                    let mut d = Box::new([0.0f32; N]);
                    let mut any = false;
                    for (i, v) in d.iter_mut().enumerate() {
                        let c = cb * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                        let mut sum = 0.0;
                        for z in -1..=1i32 {
                            for y in -1..=1i32 {
                                for x in -1..=1i32 {
                                    sum += ((2 - x.abs()) * (2 - y.abs()) * (2 - z.abs())) as f32 * at(c * 2 + IVec3::new(x, y, z));
                                }
                            }
                        }
                        if sum != 0.0 {
                            *v = sum / 64.0;
                            any = true;
                        }
                    }
                    any.then_some((cb, d))
                })
                .collect();
        }
        acc
    }

    /// The surface, using levels up to `max` (a distant view caps it; the finest view uses them all).
    pub fn surface(&self, max: usize) -> (PolyMesh, LevelSurface) {
        let t = Instant::now();
        let max = max.min(self.levels() - 1);
        let mut stats = LevelSurface::default();
        // Without restriction the levels below hold none of the finer work: a capped view adds it back, filtered.
        let folded = if self.restrict || max == self.levels() - 1 { HashMap::new() } else { self.folded_detail(max) };
        let band_max = self.band(max);
        // A lattice point shared by several levels takes the finest one's value (and sign), whichever level's cell
        // asks: cells of different levels then agree on every crossing they share, and the surface closes.
        let sample = |l: usize, i: IVec3| -> f32 {
            let (mut l, mut i) = (l, i);
            while l < max && self.blocks[l + 1].contains_key(&(i * 2).div_euclid(IVec3::splat(B))) {
                l += 1;
                i *= 2;
            }
            let v = self.sample(l, i);
            if l == max && let Some(d) = folded.get(&i.div_euclid(IVec3::splat(B))) {
                return (v + d[local(i)]).clamp(-band_max, band_max);
            }
            v
        };
        // The smallest cells that can hold surface: those of each level's blocks (and, at level 0, the blocks just
        // below stored ones, whose cells read their samples) that no finer block covers.
        let mut cells: Vec<(usize, IVec3)> = Vec::new();
        for l in 0..=max {
            let mut owners: Vec<IVec3> = self.blocks[l].keys().copied().collect();
            if l == 0 {
                let below: Vec<IVec3> = owners.iter().flat_map(|&b| (0..8).map(move |k| b - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).collect();
                owners.extend(below);
                owners.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                owners.dedup();
            }
            for bc in owners {
                for i in 0..N {
                    let c = bc * B + IVec3::new((i % 8) as i32, ((i / 8) % 8) as i32, (i / 64) as i32);
                    if l < max && self.blocks[l + 1].contains_key(&(c * 2).div_euclid(IVec3::splat(B))) {
                        continue;
                    }
                    cells.push((l, c));
                }
            }
        }
        // Each smallest crossing edge, once: from the cells' 12 edges, keeping those no finer cell splits.
        let found: Vec<Crossing> = cells
            .par_iter()
            .flat_map_iter(|&(l, c)| {
                let vx = self.voxel(l);
                let mut out = Vec::new();
                let s: [f32; 8] = std::array::from_fn(|k| sample(l, c + IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32)));
                if s.iter().all(|&v| v < 0.0) || s.iter().all(|&v| v >= 0.0) {
                    return out;
                }
                for axis in 0..3 {
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    for k in 0..4 {
                        // The cell's four edges along `axis`, by where they start in the other two directions.
                        let mut start = c;
                        start[u] += k & 1;
                        start[v] += (k >> 1) & 1;
                        let mut end = start;
                        end[axis] += 1;
                        let (a, b) = (sample(l, start), sample(l, end));
                        if (a < 0.0) == (b < 0.0) {
                            continue;
                        }
                        let mut mid = start.as_vec3();
                        mid[axis] += 0.5;
                        let around = |q1: f32, q2: f32| {
                            let mut p = mid;
                            p[u] += q1;
                            p[v] += q2;
                            self.leaf(self.origin + p * vx, max)
                        };
                        let quad = [around(-0.25, -0.25), around(0.25, -0.25), around(0.25, 0.25), around(-0.25, 0.25)];
                        if quad.iter().any(|q| q.0 > l) {
                            continue;
                        }
                        let pa = self.origin + start.as_vec3() * vx;
                        let pb = self.origin + end.as_vec3() * vx;
                        out.push((l, start, axis, quad, pa.lerp(pb, a / (a - b)), a < 0.0));
                    }
                }
                out
            })
            .collect();
        // Every cell around an edge reports it: keep each (level, start, axis) once.
        let mut seen: HashSet<(usize, IVec3, usize)> = HashSet::new();
        let mut id: HashMap<(usize, IVec3), u32> = HashMap::new();
        let mut sum: Vec<(Vec3, f32)> = Vec::new();
        let mut faces = Vec::new();
        for (l, start, axis, quad, x, inside_first) in found {
            if !seen.insert((l, start, axis)) {
                continue;
            }
            let mut ids = [0u32; 4];
            for (k, q) in quad.iter().enumerate() {
                let n = id.len() as u32;
                let v = *id.entry(*q).or_insert(n);
                if v as usize == sum.len() {
                    sum.push((Vec3::ZERO, 0.0));
                }
                sum[v as usize].0 += x;
                sum[v as usize].1 += 1.0;
                ids[k] = v;
            }
            let order = if inside_first { [ids[0], ids[1], ids[2], ids[3]] } else { [ids[0], ids[3], ids[2], ids[1]] };
            // Where two of the four cells are one coarser cell, the quad is a triangle.
            let mut uniq: Vec<u32> = Vec::with_capacity(4);
            for v in order {
                if !uniq.contains(&v) {
                    uniq.push(v);
                }
            }
            match uniq.len() {
                4 => {
                    faces.push([uniq[0], uniq[1], uniq[2], uniq[3]]);
                    stats.quads += 1;
                }
                3 => {
                    faces.push([uniq[0], uniq[1], uniq[2], NO_VERT]);
                    stats.triangles += 1;
                }
                _ => continue,
            }
            stats.per_level[l.min(7)] += 1;
        }
        let positions = sum.iter().map(|(s, n)| *s / *n).collect();
        let mesh = PolyMesh { positions, faces };
        let (open, nm) = edge_counts(&mesh.faces);
        stats.open_edges = open;
        stats.non_manifold_edges = nm;
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        (mesh, stats)
    }
}

fn trilinear(f: Vec3, c: [f32; 8]) -> f32 {
    let x00 = c[0] + (c[1] - c[0]) * f.x;
    let x10 = c[2] + (c[3] - c[2]) * f.x;
    let x01 = c[4] + (c[5] - c[4]) * f.x;
    let x11 = c[6] + (c[7] - c[6]) * f.x;
    let y0 = x00 + (x10 - x00) * f.y;
    let y1 = x01 + (x11 - x01) * f.y;
    y0 + (y1 - y0) * f.z
}

/// Edges with one face, and edges with more than two, over quads and triangles.
pub fn edge_counts(faces: &[[u32; 4]]) -> (usize, usize) {
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        let n = if f[3] == NO_VERT { 3 } else { 4 };
        for k in 0..n {
            let (a, b) = (f[k], f[(k + 1) % n]);
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    (count.values().filter(|&&c| c == 1).count(), count.values().filter(|&&c| c > 2).count())
}

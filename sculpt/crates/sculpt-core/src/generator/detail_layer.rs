//! Detail layers over a sculpted form: fine work kept as heights over the form's surface, each layer with a mask and a
//! resolution of its own, so sculpting the form carries them all along (as sculpting a lower subdivision level does),
//! a mask fades its layer anywhere, and a coarse layer costs little.
//!
//! * **Heights.** A layer holds, at each lattice point of its level near the form's surface, `h`, how far its detail
//!   stands out of the form at the point of its surface nearest (the same along the normal), and a mask `m` (0 to 1).
//!   Layers are displacements along the form's normal and add up: what is shown is the form pushed out by the sum
//!   of the visible layers' `m h`. Their order does not change what is shown.
//! * **Resolution.** Each layer lives at one level, coarse or fine. What is shown at a level holds the layers at that
//!   level or coarser (finer ones show at their own level), so a distant view that leaves fine levels out leaves fine
//!   layers out too.
//! * **Following the form.** When a stroke moves the form, each new surface point takes the height of the point of
//!   the surface it came from: found along the normal (or the stroke's push) by where the fields change sign, which
//!   holds mid-stroke, when the fields are not distances yet. The move is measured on each layer's own lattice, every
//!   four steps, and interpolated; heights are read from the layers as the stroke began, so a stroke resamples them
//!   once, and where the surface hardly moved they are copied exactly. It runs in jobs off the frame when deferred.
//! * **Shown.** A second volume holds what is drawn: the form's blocks, and at each layer's level the composed field,
//!   made a distance again near the surface (its slope can be steeper than one on a groove's wall) so the extractor's
//!   crossings and the 16-bit samples stay exact.

use std::io::{Read, Write};
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::layer::GeneratorSpec;
use super::level_volume::{EndDone, EndJob, HashMap, HashSet, Key, LevelDab, LevelVolume, Shared, Stored, StrokeEnd};
use super::volume_gen::{Anchor, GpuGen, GpuInstance, VolumeGen};

const B: i32 = 8;
const N: usize = 512;
/// Lattice steps between the points the move is measured at.
const EVERY: i32 = 4;
/// The most a height may be, in steps of the layers' first (coarsest) level, whatever a layer's own: more is form,
/// and belongs in the form. A layer's resolution then never limits what it holds.
const MOST: f32 = 6.0;

/// The tallest height `layer` holds.
fn tallest(layer: &Layer) -> f32 {
    layer.h.values().flat_map(|b| b.iter()).fold(0.0f32, |a, b| a.max(b.abs()))
}
/// A sample whose source moved less than this (in steps) copies its height rather than resampling it.
const SNAP: f32 = 0.05;

fn cell(i: usize) -> IVec3 {
    IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32)
}

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(IVec3::splat(B));
    (l.x + 8 * l.y + 64 * l.z) as usize
}

/// Trilinear weights of corner `q` (bits x, y, z) at fraction `f`.
fn corner_weight(q: i32, f: Vec3) -> f32 {
    (if q & 1 == 1 { f.x } else { 1.0 - f.x }) * (if (q >> 1) & 1 == 1 { f.y } else { 1.0 - f.y }) * (if (q >> 2) & 1 == 1 { f.z } else { 1.0 - f.z })
}

fn corner_offset(q: i32) -> IVec3 {
    IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)
}

/// One layer: its name, level, whether it shows, and its samples (heights and mask) per block of its level; or, for a
/// generator, what it scatters (its mask still per block of its level).
#[derive(Clone)]
struct Layer {
    name: String,
    level: usize,
    visible: bool,
    h: Shared<IVec3, Stored>,
    m: Shared<IVec3, Stored>,
    scatter: Option<GenPart>,
}

/// A generator layer's scatter, and the blocks it changes at each level ([`VolumeGen::blocks`]).
#[derive(Clone)]
struct GenPart {
    g: VolumeGen,
    blocks: std::sync::Arc<Vec<HashSet<IVec3>>>,
    /// The finest level it is held at in full: the layer's own, or coarser while a drag previews it.
    upto: usize,
}

/// The finest level a generator's drag preview holds it at.
const PREVIEW: usize = 1;

impl GenPart {
    fn new(form: &LevelVolume, g: VolumeGen, own: usize) -> GenPart {
        let g = g.fitted(form.voxel(own));
        let blocks = (0..form.levels()).map(|l| g.blocks(form, l, own)).collect();
        GenPart { g, blocks: std::sync::Arc::new(blocks), upto: own }
    }
}

impl Layer {
    fn new(name: &str, level: usize) -> Layer {
        Layer { name: name.to_string(), level, visible: true, h: Shared::default(), m: Shared::default(), scatter: None }
    }

    /// Whether the layer has block `bc` of level `l` (a generator, at any level it changes).
    fn holds(&self, l: usize, bc: IVec3) -> bool {
        match &self.scatter {
            Some(g) => g.blocks.get(l).is_some_and(|b| b.contains(&bc)),
            None => l == self.level && self.h.contains_key(&bc),
        }
    }

    /// Every block the layer has, with its level.
    fn all_keys(&self) -> Vec<(usize, IVec3)> {
        match &self.scatter {
            Some(g) => g.blocks.iter().enumerate().flat_map(|(l, b)| b.iter().map(move |bc| (l, *bc))).collect(),
            None => self.h.keys().map(|bc| (self.level, *bc)).collect(),
        }
    }

    fn len(&self) -> usize {
        match &self.scatter {
            Some(g) => g.blocks.iter().map(|b| b.len()).sum(),
            None => self.h.len(),
        }
    }
}

/// What a layer is, for a layer list.
#[derive(Clone, Debug)]
pub struct LayerInfo {
    pub name: String,
    pub level: usize,
    pub visible: bool,
    pub blocks: usize,
    /// A generator layer (scattering a primitive) rather than heights.
    pub generator: bool,
}

/// A generator previewed coarser, at its own level again: composed on another thread (see
/// [`Layered::generator_take`]).
pub struct GenJob {
    form: LevelVolume,
    stack: Vec<Layer>,
    from: usize,
    k: usize,
    part: GenPart,
    /// The preview it replaces: if the layer holds another by the time it is done, it is dropped.
    preview: std::sync::Arc<Vec<HashSet<IVec3>>>,
    targets: Vec<(usize, IVec3)>,
}

/// What a [`GenJob`] composed.
pub struct GenDone {
    k: usize,
    part: GenPart,
    preview: std::sync::Arc<Vec<HashSet<IVec3>>>,
    blocks: Vec<(usize, IVec3, Option<Stored>)>,
    pub ms: f64,
}

impl GenJob {
    pub fn run(self) -> GenDone {
        let t = Instant::now();
        let blocks = compose_all(&self.form, &self.stack, self.from - 1, &self.targets);
        GenDone { k: self.k, part: self.part, preview: self.preview, blocks, ms: t.elapsed().as_secs_f64() * 1e3 }
    }
}

/// A layer map's samples over a box of lattice points, read once.
struct Dense {
    lo: IVec3,
    n: IVec3,
    v: Vec<f32>,
}

impl Dense {
    fn of(map: &Shared<IVec3, Stored>, lo: IVec3, hi: IVec3, default: f32) -> Dense {
        let n = hi - lo + IVec3::ONE;
        let mut v = vec![default; (n.x * n.y * n.z) as usize];
        let (blo, bhi) = (lo.div_euclid(IVec3::splat(B)), hi.div_euclid(IVec3::splat(B)));
        for bz in blo.z..=bhi.z {
            for by in blo.y..=bhi.y {
                for bx in blo.x..=bhi.x {
                    let bc = IVec3::new(bx, by, bz);
                    let Some(s) = map.get(&bc) else { continue };
                    let (a, b) = ((bc * B).max(lo), (bc * B + IVec3::splat(B - 1)).min(hi));
                    for z in a.z..=b.z {
                        for y in a.y..=b.y {
                            for x in a.x..=b.x {
                                let i = IVec3::new(x, y, z);
                                let d = i - lo;
                                v[((d.z * n.y + d.y) * n.x + d.x) as usize] = s[local(i)];
                            }
                        }
                    }
                }
            }
        }
        Dense { lo, n, v }
    }

    fn at(&self, i: IVec3) -> f32 {
        let d = (i - self.lo).clamp(IVec3::ZERO, self.n - IVec3::ONE);
        self.v[((d.z * self.n.y + d.y) * self.n.x + d.x) as usize]
    }

    /// Whether every sample is `x`.
    fn all(&self, x: f32) -> bool {
        self.v.iter().all(|&v| v == x)
    }

    fn linear(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        (0..8).map(|q| corner_weight(q, f) * self.at(i + corner_offset(q))).sum()
    }

    /// Catmull-Rom at lattice coordinates `g`.
    fn cubic(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let (wx, wy, wz) = (catmull(f.x), catmull(f.y), catmull(f.z));
        let mut sum = 0.0;
        for (z, wz) in wz.iter().enumerate() {
            for (y, wy) in wy.iter().enumerate() {
                for (x, wx) in wx.iter().enumerate() {
                    sum += wx * wy * wz * self.at(i + IVec3::new(x as i32 - 1, y as i32 - 1, z as i32 - 1));
                }
            }
        }
        sum
    }
}

/// One block's samples.
type Samples = [f32; N];

/// A layer map's samples read where needed, straight from its blocks (the last block read kept at hand).
struct Sparse<'a> {
    map: &'a Shared<IVec3, Stored>,
    default: f32,
    last: std::cell::Cell<Option<(IVec3, Option<&'a Samples>)>>,
}

impl<'a> Sparse<'a> {
    fn new(map: &'a Shared<IVec3, Stored>, default: f32) -> Sparse<'a> {
        Sparse { map, default, last: std::cell::Cell::new(None) }
    }

    fn at(&self, i: IVec3) -> f32 {
        let bc = i.div_euclid(IVec3::splat(B));
        let block = match self.last.get() {
            Some((k, b)) if k == bc => b,
            _ => {
                let b = self.map.get(&bc).map(|b| &**b);
                self.last.set(Some((bc, b)));
                b
            }
        };
        block.map_or(self.default, |b| b[local(i)])
    }

    fn linear(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        (0..8).map(|q| corner_weight(q, f) * self.at(i + corner_offset(q))).sum()
    }

    fn cubic(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let (wx, wy, wz) = (catmull(f.x), catmull(f.y), catmull(f.z));
        let mut sum = 0.0;
        for (z, wz) in wz.iter().enumerate() {
            for (y, wy) in wy.iter().enumerate() {
                for (x, wx) in wx.iter().enumerate() {
                    sum += wx * wy * wz * self.at(i + IVec3::new(x as i32 - 1, y as i32 - 1, z as i32 - 1));
                }
            }
        }
        sum
    }
}

/// Catmull-Rom weights at `t` for the four samples around it.
fn catmull(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

/// Whether `map` has a block over any of the lattice box `lo..=hi`.
fn any_block(map: &Shared<IVec3, Stored>, lo: IVec3, hi: IVec3) -> bool {
    let (blo, bhi) = (lo.div_euclid(IVec3::splat(B)), hi.div_euclid(IVec3::splat(B)));
    (blo.z..=bhi.z).any(|z| (blo.y..=bhi.y).any(|y| (blo.x..=bhi.x).any(|x| map.contains_key(&IVec3::new(x, y, z)))))
}

/// Where `f` changes sign along the line through `p` along `d`, toward the surface from `p`: stepping by the field's
/// value (`lo` to `hi`), then halving. Only signs are trusted, since mid-stroke the field is not a distance.
fn root(f: &dyn Fn(Vec3) -> f32, p: Vec3, d: Vec3, lo: f32, hi: f32, reach: f32) -> Option<Vec3> {
    // No direction (a flat field, far inside or out): nothing to find.
    if d == Vec3::ZERO {
        return None;
    }
    let f0 = f(p);
    if f0 == 0.0 {
        return Some(p);
    }
    let inside = f0 < 0.0;
    // Both ways along the line, a step each in turn, the expected way (into the solid from outside) first.
    let first = if inside { d } else { -d };
    let mut t = [0.0f32; 2];
    let mut v = [f0; 2];
    while t[0] < reach || t[1] < reach {
        for k in 0..2 {
            if t[k] >= reach {
                continue;
            }
            let dir = if k == 0 { first } else { -first };
            let next = t[k] + (0.8 * v[k].abs()).clamp(lo, hi);
            let fv = f(p + dir * next);
            if (fv < 0.0) != inside {
                let (mut a, mut b) = (t[k], next);
                for _ in 0..14 {
                    let m = 0.5 * (a + b);
                    if (f(p + dir * m) < 0.0) == inside {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                return Some(p + dir * (0.5 * (a + b)));
            }
            t[k] = next;
            v[k] = fv;
        }
    }
    None
}

/// The form's field at level `l` over the lattice box `lo..=hi` (x fastest): its own samples where it has a block there,
/// else level `below`'s (the form's finest), interpolated but not held to level `l`'s band, since heights can stand
/// out farther.
fn unclamped(form: &LevelVolume, l: usize, below: usize, lo: IVec3, hi: IVec3) -> Vec<f32> {
    let bc = lo.div_euclid(IVec3::splat(B)) + IVec3::ONE;
    if l <= below || form.exists(l, bc) {
        return form.samples(l, lo, hi);
    }
    let k = 1i32 << (l - below);
    let (clo, chi) = (lo.div_euclid(IVec3::splat(k)), hi.div_euclid(IVec3::splat(k)) + IVec3::ONE);
    let coarse = form.samples(below, clo, chi);
    let cn = chi - clo + IVec3::ONE;
    let at = |i: IVec3| coarse[((i.z * cn.y + i.y) * cn.x + i.x) as usize];
    let n = hi - lo + IVec3::ONE;
    let mut out = Vec::with_capacity((n.x * n.y * n.z) as usize);
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let g = IVec3::new(x, y, z).as_vec3() / k as f32 - clo.as_vec3();
                let i = g.floor().as_ivec3().clamp(IVec3::ZERO, cn - IVec3::splat(2));
                let f = g - i.as_vec3();
                out.push((0..8).map(|q| corner_weight(q, f) * at(i + corner_offset(q))).sum());
            }
        }
    }
    out
}

/// The form's outward normal at `p` from the finest level up to `below` whose band reaches `p` (farther out, a fine
/// level's field is flat, and has none).
fn normal(form: &LevelVolume, p: Vec3, below: usize) -> Vec3 {
    for l in (0..=below).rev() {
        if form.value_upto(p, l, false).abs() < 0.8 * form.band(l) {
            let n = form.normal_upto(p, l, false);
            if n != Vec3::ZERO {
                return n;
            }
        }
    }
    form.normal_upto(p, 0, false)
}

/// The surface detail layer `k`'s heights stand on: the form, joined with the generators shown under the layer. Away
/// from them it is the form exactly.
struct Below<'a> {
    form: &'a LevelVolume,
    below: usize,
    gens: Vec<(&'a VolumeGen, &'a Shared<IVec3, Stored>, usize)>,
}

impl<'a> Below<'a> {
    fn of(form: &'a LevelVolume, stack: &'a [Layer], k: usize, below: usize) -> Below<'a> {
        let gens = stack[..k.min(stack.len())].iter().filter(|x| x.visible).filter_map(|x| x.scatter.as_ref().map(|g| (&g.g, &x.m, x.level))).collect();
        Below { form, below, gens }
    }

    /// Whether a generator under the layer reaches `p`.
    fn reached(&self, p: Vec3) -> bool {
        self.gens.iter().any(|(g, ..)| g.touches(p, p, g.pad()))
    }

    /// The form's distance at `p` for a join: from the finest level whose band holds it, as far as level 0 holds
    /// distances (as composing reads it).
    fn distance(&self, p: Vec3) -> f32 {
        let reach = 0.95 * self.form.plateau(0);
        for l in (0..=self.below).rev() {
            let v = self.form.value_upto(p, l, false);
            if v.abs() < 0.8 * self.form.band(l) {
                return v.clamp(-reach, reach);
            }
        }
        self.form.value_upto(p, 0, false).clamp(-reach, reach)
    }

    /// Its field at `p`, the form read up to `level`.
    fn value(&self, p: Vec3, level: usize) -> f32 {
        let mut out: Option<f32> = None;
        let mut near = Vec::new();
        for (g, m, gl) in &self.gens {
            g.near(p, p, g.pad(), &mut near);
            if near.is_empty() {
                continue;
            }
            let c = out.unwrap_or_else(|| self.distance(p));
            out = Some(g.apply(c, p, &near, mask_at(self.form, m, *gl, p)));
        }
        out.unwrap_or_else(|| self.form.value_upto(p, level, false))
    }

    /// Where `p`, near what a generator under the layer scattered, was in `was` (the same generators as they were):
    /// what they scatter moves whole, so a point on it came from the same place on it before. With how much that
    /// holds: all where the instance is nearer than the form, none where the form is, easing across the join.
    fn rigid(&self, was: &Below, p: Vec3) -> Option<(Vec3, f32)> {
        if self.gens.len() != was.gens.len() {
            return None;
        }
        let mut near = Vec::new();
        let mut best: Option<(f32, Vec3, f32)> = None;
        for ((g1, ..), (g0, ..)) in self.gens.iter().zip(&was.gens) {
            if g1.instances().len() != g0.instances().len() {
                continue;
            }
            g1.near(p, p, g1.pad(), &mut near);
            let Some((j, d)) = near.iter().map(|&j| (j, g1.sdf(j, p))).min_by(|a, b| a.1.total_cmp(&b.1)) else { continue };
            if best.is_some_and(|b| b.0 <= d) {
                continue;
            }
            let (a, b) = (g1.instances()[j as usize], g0.instances()[j as usize]);
            let local = (a.rotation.inverse() * (p - a.translation)) / a.scale;
            best = Some((d, b.translation + b.rotation * (local * b.scale), g1.blend().max(1e-6)));
        }
        let (d, from, k) = best?;
        let c = self.distance(p);
        let t = ((c - d) / k + 0.5).clamp(0.0, 1.0);
        (t > 0.0).then_some((from, t * t * (3.0 - 2.0 * t)))
    }

    /// Its outward normal at `p`.
    fn normal(&self, p: Vec3) -> Vec3 {
        if !self.reached(p) {
            return normal(self.form, p, self.below);
        }
        let h = 0.5 * self.form.voxel(self.below);
        let f = |q: Vec3| self.value(q, self.below);
        Vec3::new(f(p + Vec3::X * h) - f(p - Vec3::X * h), f(p + Vec3::Y * h) - f(p - Vec3::Y * h), f(p + Vec3::Z * h) - f(p - Vec3::Z * h)).normalize_or_zero()
    }
}

/// A mask of a layer at level `l` (absent: whole) at world point `p`, linearly.
fn mask_at(form: &LevelVolume, m: &Shared<IVec3, Stored>, l: usize, p: Vec3) -> f32 {
    if m.is_empty() {
        return 1.0;
    }
    let g = (p - form.origin()) / form.voxel(l);
    let i = g.floor().as_ivec3();
    let f = g - i.as_vec3();
    (0..8)
        .map(|q| {
            let j = i + corner_offset(q);
            corner_weight(q, f) * m.get(&j.div_euclid(IVec3::splat(B))).map_or(1.0, |b| b[local(j)])
        })
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

/// The visible layers' masked heights summed, over the lattice box `lo..=hi` of level `l` (x fastest): the layers at
/// level `l` or coarser (with `finer`, all of them), but for layer `except`.
fn offsets(stack: &[Layer], l: usize, lo: IVec3, hi: IVec3, finer: bool, except: Option<usize>) -> Vec<f32> {
    offsets_of(stack, l, lo, hi, &|k, layer| Some(k) != except && (finer || layer.level <= l))
}

/// [`offsets`] of the visible detail layers `which` picks.
fn offsets_of(stack: &[Layer], l: usize, lo: IVec3, hi: IVec3, which: &dyn Fn(usize, &Layer) -> bool) -> Vec<f32> {
    let n = hi - lo + IVec3::ONE;
    let mut out = vec![0.0f32; (n.x * n.y * n.z) as usize];
    for (k, layer) in stack.iter().enumerate() {
        if !layer.visible || layer.scatter.is_some() || !which(k, layer) {
            continue;
        }
        // The layer's lattice over the box: coarser, read linearly; finer, at the points it shares.
        let (scale, up) = if layer.level <= l { (1i32 << (l - layer.level), false) } else { (1i32 << (layer.level - l), true) };
        let (llo, lhi) = if up { (lo * scale, hi * scale) } else { (lo.div_euclid(IVec3::splat(scale)), hi.div_euclid(IVec3::splat(scale)) + IVec3::ONE) };
        if !any_block(&layer.h, llo, lhi) {
            continue;
        }
        let (h, m) = (Dense::of(&layer.h, llo, lhi, 0.0), Dense::of(&layer.m, llo, lhi, 1.0));
        for z in 0..n.z {
            for y in 0..n.y {
                for x in 0..n.x {
                    let i = lo + IVec3::new(x, y, z);
                    let s = if up {
                        let j = i * scale;
                        m.at(j) * h.at(j)
                    } else if scale == 1 {
                        m.at(i) * h.at(i)
                    } else {
                        let g = i.as_vec3() / scale as f32;
                        m.linear(g) * h.linear(g)
                    };
                    out[((z * n.y + y) * n.x + x) as usize] += s;
                }
            }
        }
    }
    out
}

/// Whether block `bc` of level `l` is composed (a visible layer there has it).
fn composed(stack: &[Layer], l: usize, bc: IVec3) -> bool {
    stack.iter().any(|x| x.visible && x.holds(l, bc))
}

/// The shown blocks layer changes `changes` (level, block) reach: themselves, and the finer layers' blocks under them
/// (which hold the coarser layers too).
fn affected(stack: &[Layer], changes: &[(usize, IVec3)]) -> Vec<(usize, IVec3)> {
    let top = stack.iter().map(|x| if x.scatter.is_some() { x.all_keys().iter().map(|k| k.0).max().unwrap_or(0) } else { x.level }).max().unwrap_or(0).max(changes.iter().map(|c| c.0).max().unwrap_or(0));
    let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); top + 1];
    for &(l, bc) in changes {
        per[l].insert(bc);
    }
    let mut out: HashSet<(usize, IVec3)> = changes.iter().copied().collect();
    for layer in stack.iter().filter(|x| x.visible) {
        for (lf, bc) in layer.all_keys() {
            if (0..lf.min(top + 1)).any(|lk| !per[lk].is_empty() && per[lk].contains(&(bc >> (lf - lk) as i32))) {
                out.insert((lf, bc));
            }
        }
    }
    out.into_iter().collect()
}

/// What is shown at level `l` for blocks `keys`: the form pushed out by the layers at that level or coarser, or (`None`)
/// nothing of its own where no visible layer at that level has the block.
fn compose_blocks(form: &LevelVolume, stack: &[Layer], l: usize, below: usize, keys: &[IVec3]) -> Vec<(IVec3, Option<Stored>)> {
    let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
    // The generators shown, in the stack's order, each with its mask's lattice scale against this level's.
    let gens: Vec<(&VolumeGen, Option<&Shared<IVec3, Stored>>, f32)> =
        stack.iter().filter(|x| x.visible).filter_map(|x| x.scatter.as_ref().map(|g| (&g.g, (!x.m.is_empty()).then_some(&x.m), 2f32.powi(x.level as i32 - l as i32)))).collect();
    // Detail layers over a generator ride what is under them, generators included: composed in the stack's order.
    let ordered = stack.iter().position(|x| x.visible && x.scatter.is_some()).is_some_and(|g| stack[g + 1..].iter().any(|x| x.visible && x.scatter.is_none() && x.level <= l));
    keys.par_iter()
        .map(|&bc| {
            if !composed(stack, l, bc) {
                return (bc, None);
            }
            let (lo, hi) = (bc * B - IVec3::ONE, bc * B + IVec3::splat(B));
            let n = 10;
            if ordered {
                let c = compose_ordered(form, stack, l, below, lo, hi);
                return (bc, Some(finish(&c, n, v, band)));
            }
            let s = offsets(stack, l, lo, hi, false, None);
            let most = s.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            // The form's field around the block, as far out as the heights reach.
            let pad = (most / v).ceil() as i32 + 2;
            let (plo, phi) = (lo - IVec3::splat(pad), hi + IVec3::splat(pad));
            let pn = phi - plo + IVec3::ONE;
            let field = unclamped(form, l, below, plo, phi);
            let pa = |i: IVec3| {
                let d = (i - plo).clamp(IVec3::ZERO, pn - IVec3::ONE);
                field[((d.z * pn.y + d.y) * pn.x + d.x) as usize]
            };
            let tri = |g: Vec3| {
                let i = g.floor().as_ivec3();
                let f = g - i.as_vec3();
                (0..8).map(|q| corner_weight(q, f) * pa(i + corner_offset(q))).sum::<f32>()
            };
            // The generators' instances within reach of the block, and their masks.
            let (wlo, whi) = (origin + lo.as_vec3() * v, origin + hi.as_vec3() * v);
            let near: Vec<(&VolumeGen, Option<Sparse>, f32, Vec<u32>)> = gens
                .iter()
                .filter_map(|&(g, m, scale)| {
                    let mut near = Vec::new();
                    g.near(wlo, whi, g.pad() + v, &mut near);
                    (!near.is_empty()).then(|| (g, m.map(|m| Sparse::new(m, 1.0)), scale, near))
                })
                .collect();
            // A soft join reads the form's distance out to its blend's width, farther than a fine level holds one (it
            // clamps at its band): where this level's is near its clamp, level 0's (which reaches 16 of its steps)
            // takes over, so every level joins the same surface.
            let sat = if l <= below { band } else { form.band(below) };
            // And past what level 0 holds as distances (six of its steps: past them, a stored block holds a bound and
            // an unstored one its band) the form counts as that far, everywhere alike: a blend wider than that reads
            // the same value on both sides of every block's face.
            let reach = 0.95 * form.plateau(0);
            let coarse = (l > 0 && !near.is_empty()).then(|| {
                let s = 1i32 << l;
                let (clo, chi) = (lo.div_euclid(IVec3::splat(s)), hi.div_euclid(IVec3::splat(s)) + IVec3::ONE);
                (clo, chi - clo + IVec3::ONE, form.samples(0, clo, chi), s as f32)
            });
            let far = |i: IVec3, c: f32| -> f32 {
                let Some((clo, cn, g, s)) = &coarse else { return c };
                let t = ((c.abs() - 0.5 * sat) / (0.4 * sat)).clamp(0.0, 1.0);
                if t == 0.0 {
                    return c;
                }
                let q = i.as_vec3() / *s - clo.as_vec3();
                let b = q.floor().as_ivec3().clamp(IVec3::ZERO, *cn - IVec3::splat(2));
                let f = q - b.as_vec3();
                let c0: f32 = (0..8).map(|k| {
                    let j = b + corner_offset(k);
                    corner_weight(k, f) * g[((j.z * cn.y + j.y) * cn.x + j.x) as usize]
                }).sum();
                let t = t * t * (3.0 - 2.0 * t);
                c + (c0 - c) * t
            };
            let c: Vec<f32> = (0..n * n * n)
                .map(|k| {
                    let i = lo + IVec3::new(k % n, (k / n) % n, k / (n * n));
                    let s = s[k as usize];
                    // Away from the surface the height moves nothing that shows.
                    let mut c = if s == 0.0 || pa(i).abs() > band + s.abs() + v {
                        pa(i)
                    } else {
                        // The form's field where this point would be with the height taken off, along the normal: zero
                        // exactly the height out from the form's surface, however steep or shallow the field is.
                        let g = Vec3::new(pa(i + IVec3::X) - pa(i - IVec3::X), pa(i + IVec3::Y) - pa(i - IVec3::Y), pa(i + IVec3::Z) - pa(i - IVec3::Z));
                        tri(i.as_vec3() - g.normalize_or_zero() * (s / v))
                    };
                    // Then each generator on it, in order.
                    if !near.is_empty() {
                        c = far(i, c).clamp(-reach, reach);
                    }
                    for (g, mask, scale, near) in &near {
                        let m = mask.as_ref().map_or(1.0, |m| m.linear(i.as_vec3() * *scale).clamp(0.0, 1.0));
                        c = g.apply(c, origin + i.as_vec3() * v, near, m);
                    }
                    c
                })
                .collect();
            (bc, Some(finish(&c, n, v, band)))
        })
        .collect()
}

/// A block's samples from its composed field `c` (`n` per side, a sample past it all round): steeper than a distance on
/// a groove's wall, made one again, so a crossing sits where it is; and held to the band.
fn finish(c: &[f32], n: i32, v: f32, band: f32) -> Stored {
    let at = |x: i32, y: i32, z: i32| c[(x + n * (y + n * z)) as usize];
    let mut out = Box::new([0.0f32; N]);
    for (i, o) in out.iter_mut().enumerate() {
        let q = cell(i) + IVec3::ONE;
        let (x, y, z) = (q.x, q.y, q.z);
        let g = Vec3::new(at(x + 1, y, z) - at(x - 1, y, z), at(x, y + 1, z) - at(x, y - 1, z), at(x, y, z + 1) - at(x, y, z - 1)) / (2.0 * v);
        *o = (at(x, y, z) / g.length().max(1.0)).clamp(-band, band);
    }
    Stored::from(out)
}

/// What is shown over the lattice box `lo..=hi` of level `l` (x fastest), composed in the stack's order: the form,
/// then each run of detail layers pushing out what is under it (along its normal), and each generator joined to it.
fn compose_ordered(form: &LevelVolume, stack: &[Layer], l: usize, below: usize, lo: IVec3, hi: IVec3) -> Vec<f32> {
    let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
    let s_all = offsets(stack, l, lo, hi, false, None);
    let most = s_all.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let pad = (most / v).ceil() as i32 + 2;
    let (plo, phi) = (lo - IVec3::splat(pad), hi + IVec3::splat(pad));
    let pn = phi - plo + IVec3::ONE;
    let idx = |i: IVec3| {
        let d = (i - plo).clamp(IVec3::ZERO, pn - IVec3::ONE);
        ((d.z * pn.y + d.y) * pn.x + d.x) as usize
    };
    let points: Vec<IVec3> = (0..pn.x * pn.y * pn.z).map(|k| plo + IVec3::new(k % pn.x, (k / pn.x) % pn.y, k / (pn.x * pn.y))).collect();
    let mut g = unclamped(form, l, below, plo, phi);
    let sat = if l <= below { band } else { form.band(below) };
    let reach = 0.95 * form.plateau(0);
    // Level 0's distance over the box, for joins wider than this level holds one (see `compose_blocks`).
    let coarse = (l > 0).then(|| {
        let s = 1i32 << l;
        let (clo, chi) = (plo.div_euclid(IVec3::splat(s)), phi.div_euclid(IVec3::splat(s)) + IVec3::ONE);
        (clo, chi - clo + IVec3::ONE, form.samples(0, clo, chi), s as f32)
    });
    let far = |i: IVec3, c: f32| -> f32 {
        let Some((clo, cn, g0, s)) = &coarse else { return c };
        let t = ((c.abs() - 0.5 * sat) / (0.4 * sat)).clamp(0.0, 1.0);
        if t == 0.0 {
            return c;
        }
        let q = i.as_vec3() / *s - clo.as_vec3();
        let b = q.floor().as_ivec3().clamp(IVec3::ZERO, *cn - IVec3::splat(2));
        let f = q - b.as_vec3();
        let c0: f32 = (0..8).map(|k| {
            let j = b + corner_offset(k);
            corner_weight(k, f) * g0[((j.z * cn.y + j.y) * cn.x + j.x) as usize]
        }).sum();
        let t = t * t * (3.0 - 2.0 * t);
        c + (c0 - c) * t
    };
    // A run of detail layers (indices `run`) pushing out the field so far.
    let push = |g: &Vec<f32>, run: &[usize]| -> Vec<f32> {
        let s = offsets_of(stack, l, plo, phi, &|k, layer| layer.level <= l && run.contains(&k));
        if s.iter().all(|x| *x == 0.0) {
            return g.clone();
        }
        let pa = |i: IVec3| g[idx(i)];
        let tri = |q: Vec3| {
            let i = q.floor().as_ivec3();
            let f = q - i.as_vec3();
            (0..8).map(|c| corner_weight(c, f) * pa(i + corner_offset(c))).sum::<f32>()
        };
        points
            .par_iter()
            .enumerate()
            .map(|(k, &i)| {
                let s = s[k];
                if s == 0.0 || pa(i).abs() > band + s.abs() + v {
                    return pa(i);
                }
                let n = Vec3::new(pa(i + IVec3::X) - pa(i - IVec3::X), pa(i + IVec3::Y) - pa(i - IVec3::Y), pa(i + IVec3::Z) - pa(i - IVec3::Z));
                tri(i.as_vec3() - n.normalize_or_zero() * (s / v))
            })
            .collect()
    };
    let mut run: Vec<usize> = Vec::new();
    for (k, x) in stack.iter().enumerate() {
        if !x.visible {
            continue;
        }
        let Some(part) = x.scatter.as_ref() else {
            run.push(k);
            continue;
        };
        if !run.is_empty() {
            g = push(&g, &run);
            run.clear();
        }
        let gg = &part.g;
        let mut near = Vec::new();
        gg.near(origin + plo.as_vec3() * v, origin + phi.as_vec3() * v, gg.pad() + v, &mut near);
        if near.is_empty() {
            continue;
        }
        let scale = 2f32.powi(x.level as i32 - l as i32);
        let mask = (!x.m.is_empty()).then(|| Sparse::new(&x.m, 1.0));
        g = points
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let m = mask.as_ref().map_or(1.0, |m| m.linear(i.as_vec3() * scale).clamp(0.0, 1.0));
                gg.apply(far(i, g[k]).clamp(-reach, reach), origin + i.as_vec3() * v, &near, m)
            })
            .collect();
    }
    if !run.is_empty() {
        g = push(&g, &run);
    }
    let n = hi - lo + IVec3::ONE;
    (0..n.x * n.y * n.z).map(|k| g[idx(lo + IVec3::new(k % n.x, (k / n.x) % n.y, k / (n.x * n.y)))]).collect()
}

/// Block `bc` of level `l`'s 9x9x9 samples (its own and the first layer past it) as composed without the generators:
/// the form pushed out by the detail layers, read for a join (level 0's distance where this level's is near its clamp,
/// and no farther than level 0 holds distances) where `join` (a generator reaches it). What the GPU joins them to.
fn base_block(form: &LevelVolume, stack: &[Layer], l: usize, below: usize, bc: IVec3, join: bool) -> Box<[f32; 729]> {
    let (v, band) = (form.voxel(l), form.band(l));
    let (lo, hi) = (bc * B, bc * B + IVec3::splat(B));
    let s = offsets(stack, l, lo, hi, false, None);
    let most = s.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let pad = (most / v).ceil() as i32 + 2;
    let (plo, phi) = (lo - IVec3::splat(pad), hi + IVec3::splat(pad));
    let pn = phi - plo + IVec3::ONE;
    let field = unclamped(form, l, below, plo, phi);
    let pa = |i: IVec3| {
        let d = (i - plo).clamp(IVec3::ZERO, pn - IVec3::ONE);
        field[((d.z * pn.y + d.y) * pn.x + d.x) as usize]
    };
    let tri = |g: Vec3| {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        (0..8).map(|q| corner_weight(q, f) * pa(i + corner_offset(q))).sum::<f32>()
    };
    let sat = if l <= below { band } else { form.band(below) };
    let reach = 0.95 * form.plateau(0);
    let coarse = (l > 0 && join).then(|| {
        let s = 1i32 << l;
        let (clo, chi) = (lo.div_euclid(IVec3::splat(s)), hi.div_euclid(IVec3::splat(s)) + IVec3::ONE);
        (clo, chi - clo + IVec3::ONE, form.samples(0, clo, chi), s as f32)
    });
    let far = |i: IVec3, c: f32| -> f32 {
        let Some((clo, cn, g, s)) = &coarse else { return c };
        let t = ((c.abs() - 0.5 * sat) / (0.4 * sat)).clamp(0.0, 1.0);
        if t == 0.0 {
            return c;
        }
        let q = i.as_vec3() / *s - clo.as_vec3();
        let b = q.floor().as_ivec3().clamp(IVec3::ZERO, *cn - IVec3::splat(2));
        let f = q - b.as_vec3();
        let c0: f32 = (0..8).map(|k| {
            let j = b + corner_offset(k);
            corner_weight(k, f) * g[((j.z * cn.y + j.y) * cn.x + j.x) as usize]
        }).sum();
        let t = t * t * (3.0 - 2.0 * t);
        c + (c0 - c) * t
    };
    let mut out = Box::new([0.0f32; 729]);
    for (k, o) in out.iter_mut().enumerate() {
        let i = lo + IVec3::new(k as i32 % 9, (k as i32 / 9) % 9, k as i32 / 81);
        let s = s[k];
        let c = if s == 0.0 || pa(i).abs() > band + s.abs() + v {
            pa(i)
        } else {
            let g = Vec3::new(pa(i + IVec3::X) - pa(i - IVec3::X), pa(i + IVec3::Y) - pa(i - IVec3::Y), pa(i + IVec3::Z) - pa(i - IVec3::Z));
            tri(i.as_vec3() - g.normalize_or_zero() * (s / v))
        };
        *o = if join { far(i, c).clamp(-reach, reach) } else { c };
    }
    out
}

/// What the GPU joins the generators to the field with, for an edit: the generators shown (in the stack's order) and
/// their instances, and per block the instances near it of each. See [`Layered::gpu_generator`].
pub struct GpuPlan {
    pub gens: Vec<GpuGen>,
    pub instances: Vec<GpuInstance>,
    pub cands: Vec<u32>,
    /// Per block: its key, and per generator (as `gens`) the range of `cands` near it and its mask's samples there
    /// (an index into `masks`, `u32::MAX` where the mask is whole).
    pub blocks: Vec<(Key, Vec<[u32; 4]>)>,
    /// Masks' 9x9x9 samples, block by block.
    pub masks: Vec<Box<[f32; 729]>>,
    /// [`Layered::base_epoch`] when it was made.
    pub epoch: u64,
}

/// Compose the shown blocks `targets` (level, block), level by level.
fn compose_all(form: &LevelVolume, stack: &[Layer], below: usize, targets: &[(usize, IVec3)]) -> Vec<(usize, IVec3, Option<Stored>)> {
    let mut per: HashMap<usize, Vec<IVec3>> = HashMap::default();
    for &(l, bc) in targets {
        per.entry(l).or_default().push(bc);
    }
    let mut out = Vec::with_capacity(targets.len());
    for (l, keys) in per {
        out.extend(compose_blocks(form, stack, l, below, &keys).into_iter().map(|(bc, s)| (l, bc, s)));
    }
    out
}

/// Layer blocks (of `layer`, at level `l`) under any of the form's blocks `region`.
fn under(l: usize, region: &[Key], layer: &Layer) -> Vec<IVec3> {
    let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); l + 1];
    for &(m, bc) in region {
        if (m as usize) <= l {
            per[m as usize].insert(bc);
        }
    }
    layer.h.keys().filter(|bc| (0..=l).any(|m| !per[m].is_empty() && per[m].contains(&(**bc >> (l - m) as i32)))).copied().collect()
}

/// Every block of `layer` (at any level) under any of the form's blocks `region`, with its level.
fn under_all(region: &[Key], layer: &Layer) -> Vec<(usize, IVec3)> {
    if layer.scatter.is_none() {
        return under(layer.level, region, layer).into_iter().map(|bc| (layer.level, bc)).collect();
    }
    let top = region.iter().map(|k| k.0 as usize).max().unwrap_or(0);
    let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); top + 1];
    for &(m, bc) in region {
        per[m as usize].insert(bc);
    }
    layer.all_keys().into_iter().filter(|&(l, bc)| (0..=l.min(top)).any(|m| !per[m].is_empty() && per[m].contains(&(bc >> (l - m) as i32)))).collect()
}

/// What a follow did.
#[derive(Clone, Copy, Debug, Default)]
pub struct FollowStats {
    pub ms: f64,
    /// Layer blocks worked out again, and the points the move was measured at.
    pub blocks: usize,
    pub points: usize,
    /// Its parts: finding where the layers' blocks went, measuring the move, the heights, and composing.
    pub parts_ms: [f64; 4],
}

enum Undo {
    /// A stroke on the form, with the layers as they were before it.
    Form(Vec<Layer>),
    /// A change to the layers alone.
    Layers(Vec<Layer>),
}

/// A block's corner moves (each `None` where the search found no surface).
type Corners = [Option<Vec3>; 8];

/// Corner moves a block was last followed with, keyed by (layer, block).
type Used = HashMap<(usize, IVec3), Corners>;

/// A form with detail layers over it.
pub struct Layered {
    form: LevelVolume,
    /// The first level the layers took work from: the form keeps the coarser ones, layers live at it or finer.
    from: usize,
    stack: Vec<Layer>,
    active: usize,
    /// While a layer itself is edited (its mask painted): the layers as the edit began.
    edit: Option<Vec<Layer>>,
    /// While a stroke on the form is under way: the form and the layers as it began, and the blocks it changed.
    start: Option<(LevelVolume, Vec<Layer>)>,
    touched: HashSet<Key>,
    history: Vec<Undo>,
    redo: Vec<Undo>,
    pub last: FollowStats,
    /// Follows taken in, this stroke.
    pub follows: usize,
    /// Whether dabs leave following to [`Self::follow_take`] (jobs on other threads) rather than doing it at once.
    pub deferred: bool,
    /// Blocks of the form dabs changed over the layers, waiting for a follow; the pen up, waiting for the last one; a
    /// follow under way.
    pending: HashSet<Key>,
    ending: bool,
    running: bool,
    /// Per level, the blocks over any layer's.
    covered: Vec<HashSet<IVec3>>,
    /// The move at each layer block's corners when it was last followed this stroke: a block whose move has not
    /// changed since is left as it is.
    used: Used,
    /// Layer blocks follows changed this stroke: its last follow works them all out again.
    changed: HashSet<(usize, IVec3)>,
    /// Bumped whenever what is under the generators may have changed (the form, the detail layers, which show): a
    /// base field (see [`Self::base_samples`]) read before it is stale.
    base_epoch: u64,
}

/// A follow to work out on another thread: copies of what it reads (see [`Layered::follow_take`]).
pub struct FollowJob {
    form: LevelVolume,
    form0: LevelVolume,
    stack0: Vec<Layer>,
    stack: Vec<Layer>,
    from: usize,
    region: Vec<Key>,
    end: bool,
    used: Used,
    /// Layer blocks to work out besides those under the region (at the end, all the stroke changed).
    also: Vec<(usize, IVec3)>,
}

/// One layer's new blocks (heights and mask; `None` where it holds nothing any more).
type LayerBlocks = Vec<(IVec3, Option<(Stored, Stored)>)>;

/// What a [`FollowJob`] worked out, for [`Layered::follow_apply`].
pub struct FollowDone {
    form: LevelVolume,
    region: Vec<Key>,
    end: bool,
    layers: Vec<(usize, LayerBlocks)>,
    /// Generator layers as the follow left them (the stroke's last finds their anchors again on the form).
    gens: Vec<(usize, GenPart)>,
    shown: Vec<(usize, IVec3, Option<Stored>)>,
    used: Vec<((usize, IVec3), Corners)>,
    stats: FollowStats,
}

impl FollowJob {
    /// Blocks of the form it covers.
    pub fn len(&self) -> usize {
        self.region.len()
    }

    pub fn is_empty(&self) -> bool {
        self.region.is_empty()
    }

    /// Whether it is a stroke's last follow (the pen is up: nothing else wants the threads).
    pub fn last(&self) -> bool {
        self.end
    }

    pub fn run(mut self) -> FollowDone {
        let t = Instant::now();
        let mut stats = FollowStats::default();
        let mut layers = Vec::with_capacity(self.stack.len());
        let mut used = Vec::new();
        let mut changes = Vec::new();
        let mut gens = Vec::new();
        for k in 0..self.stack.len() {
            if let Some(part) = self.stack[k].scatter.clone() {
                let own = self.stack[k].level;
                let part = if self.end {
                    // The anchors found again on the form, and every block the layer changed or changes now.
                    let reach = (16.0 * self.form.voxel(0)).max(2.0 * part.g.spec.size);
                    let moved = GenPart::new(&self.form, part.g.follow(&self.form, reach), own);
                    changes.extend(self.stack[k].all_keys());
                    self.stack[k].scatter = Some(moved.clone());
                    changes.extend(self.stack[k].all_keys());
                    moved
                } else {
                    // Its blocks under the region, and form blocks the stroke made within its reach.
                    let mut blocks = (*part.blocks).clone();
                    for &(m, bc) in &self.region {
                        let m = m as usize;
                        if blocks[m].contains(&bc) {
                            changes.push((m, bc));
                        } else if self.form.exists(m, bc) {
                            let (v, origin) = (self.form.voxel(m), self.form.origin());
                            let lo = origin + (bc * B).as_vec3() * v;
                            if part.g.touches(lo, lo + Vec3::splat(B as f32 * v), part.g.pad() + self.form.band(m) + v) {
                                blocks[m].insert(bc);
                                changes.push((m, bc));
                            }
                        }
                    }
                    let part = GenPart { g: part.g, blocks: std::sync::Arc::new(blocks), upto: part.upto };
                    self.stack[k].scatter = Some(part.clone());
                    part
                };
                gens.push((k, part));
                continue;
            }
            let (built, u, s) = follow_heights(&self, k);
            stats.blocks += s.blocks;
            stats.points += s.points;
            for p in 0..3 {
                stats.parts_ms[p] += s.parts_ms[p];
            }
            let layer = &mut self.stack[k];
            for (bc, hm) in &built {
                match hm {
                    Some((h, m)) => {
                        layer.h.insert(*bc, h.clone());
                        layer.m.insert(*bc, m.clone());
                    }
                    None => {
                        layer.h.remove(bc);
                        layer.m.remove(bc);
                    }
                }
                changes.push((layer.level, *bc));
            }
            used.extend(u.into_iter().map(|(bc, c)| ((k, bc), c)));
            layers.push((k, built));
        }
        let t3 = Instant::now();
        let targets = affected(&self.stack, &changes);
        let shown = compose_all(&self.form, &self.stack, self.from - 1, &targets);
        stats.parts_ms[3] = t3.elapsed().as_secs_f64() * 1e3;
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        FollowDone { form: self.form, region: self.region, end: self.end, layers, gens, shown, used, stats }
    }
}

/// Layer `k`'s heights over the form as a follow job's copy has it, for the layer's blocks its region reaches (those
/// whose move changed since they were last followed, mid-stroke).
#[allow(clippy::type_complexity)]
fn follow_heights(job: &FollowJob, k: usize) -> (LayerBlocks, Vec<(IVec3, Corners)>, FollowStats) {
    let (form, form0, end) = (&job.form, &job.form0, job.end);
    let current = &job.stack[k];
    let empty = Layer::new("", current.level);
    let layer0 = job.stack0.get(k).filter(|x| x.level == current.level).unwrap_or(&empty);
    let region = &job.region;
    let l = current.level;
    let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
    let (top, below) = (form.levels() - 1, job.from - 1);
    let (under1, under0) = (Below::of(form, &job.stack, k, below), Below::of(form0, &job.stack0, k, below));
    let new = |p: Vec3| under1.value(p, top);
    let old = |p: Vec3| under0.value(p, top);
    let reach = 64.0 * form.voxel(below.saturating_sub(2));
    // The longest step a search takes: the form's fields are distances out to its finest level's band, and only
    // signs are trusted, so a step may go that far.
    let far = (4.0 * v).max(form.band(below));
    // How far heights may stand out, and how far this layer's do (what decides which blocks hold any of it).
    let limit = MOST * form.voxel(job.from);
    let tall = tallest(layer0).max(tallest(current)).min(limit) + v;
    let mut src = under(l, region, layer0);
    // At the end, the layer's blocks over generators under it: what they scatter moved with the form wherever it
    // hangs from, which is not where these blocks are.
    if end && !(under0.gens.is_empty() && under1.gens.is_empty()) {
        let size = v * B as f32;
        let over = |bc: &IVec3| {
            let lo = origin + (*bc * B).as_vec3() * v;
            under0.gens.iter().chain(&under1.gens).any(|(g, ..)| g.touches(lo, lo + Vec3::splat(size), g.pad() + band))
        };
        let had: HashSet<IVec3> = src.iter().copied().collect();
        src.extend(layer0.h.keys().filter(|bc| !had.contains(*bc) && over(bc)).copied().collect::<Vec<_>>());
    }
    let mut dest: HashSet<IVec3> = src.iter().copied().collect();
    dest.extend(under(l, region, current));
    dest.extend(job.also.iter().filter(|a| a.0 == k).map(|a| a.1));
    if dest.is_empty() {
        return (Vec::new(), Vec::new(), FollowStats::default());
    }
    // Where the surface in each source block went: blocks there may need the layer too.
    let near = 4.0 * 3f32.sqrt() * v + band + tall;
    let t0 = Instant::now();
    let grown: Vec<IVec3> = src
        .par_iter()
        .flat_map_iter(|&bc| {
            // Mid-stroke from the block's centre (a preview); at the end from its corners as well.
            (if end { 0..9 } else { 8..9 })
                .filter_map(|q| {
                    let g = if q == 8 { IVec3::splat(B / 2) } else { corner_offset(q) * B };
                    let c = origin + (bc * B + g).as_vec3() * v;
                    let n0 = under0.normal(c);
                    let y0 = root(&old, c, n0, 0.25 * v, far, reach)?;
                    let u = form.push_dir(y0, n0);
                    let y1 = root(&new, y0, u, 0.25 * v, far, reach)?;
                    Some(((c + y1 - y0 - origin) / (v * B as f32)).floor().as_ivec3())
                })
                .filter(|b| new(origin + (*b * B + IVec3::splat(B / 2)).as_vec3() * v).abs() < near)
                .collect::<Vec<_>>()
        })
        .collect();
    // With their face neighbours: the surface of a block can land across a face from where its points did (the
    // corners' distances below drop those that hold nothing).
    let faces = [IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z];
    let grown: HashSet<IVec3> = grown.into_iter().flat_map(|b| std::iter::once(b).chain(faces.iter().map(move |f| b + *f))).collect();
    dest.extend(grown);
    let dest: Vec<IVec3> = dest.into_iter().collect();
    // Each point's move (in steps) and its distance from the new surface (found by the search, so exact however far
    // out the fields hold only a bound).
    let measure = |i: IVec3| -> Option<(Vec3, f32)> {
        let p = origin + i.as_vec3() * v;
        let n1 = under1.normal(p);
        let o = root(&new, p, n1, 0.25 * v, far, reach).and_then(|y1| {
            let n0 = under0.normal(y1);
            let n0 = if n0 == Vec3::ZERO { n1 } else { n0 };
            let u = form.push_dir(y1, n0);
            // As far out from the surface the stroke began on, along its normal there, as `p` is out of the new one:
            // where the stroke tilted the surface, the same offset along the new normal would land beside the point it
            // came from.
            root(&old, y1, u, 0.25 * v, far, reach).map(|y0| {
                let n = under0.normal(y0);
                let n = if n == Vec3::ZERO { n1 } else { n };
                (y0 + n * (p - y1).dot(n1) - p, (p - y1).dot(n1))
            })
        });
        // On what a generator scattered: where it was, moved whole (eased into the search across the join).
        let o = match under1.rigid(&under0, p) {
            Some((from, w)) => {
                let d = o.map_or_else(|| under1.value(p, top), |o| o.1);
                Some((match o {
                    Some((m, _)) if w < 1.0 => m * (1.0 - w) + (from - p) * w,
                    _ => from - p,
                }, d))
            }
            None => o,
        };
        o.map(|(o, d)| (o / v, d))
    };
    // The move, measured every `EVERY` steps: from each point to the new surface, and from there back to the surface
    // the stroke began on. First at the blocks' corners: a block whose corners all lie farther from the new surface
    // than a height can reach (and half the block across) holds nothing that shows, and needs no more. Mid-stroke the
    // corners are all (a preview); at the end every `EVERY` steps.
    let t1 = Instant::now();
    let corner_points: Vec<IVec3> = dest.iter().flat_map(|bc| (0..8).map(move |q| (*bc + corner_offset(q)) * B)).collect::<HashSet<_>>().into_iter().collect();
    let mut moves: HashMap<IVec3, Option<(Vec3, f32)>> = corner_points.par_iter().map(|&i| (i, measure(i))).collect();
    let keep = band + tall + 4.0 * 3f32.sqrt() * v;
    let (dest, gone): (Vec<IVec3>, Vec<IVec3>) = dest.into_iter().partition(|bc| (0..8).any(|q| moves.get(&((*bc + corner_offset(q)) * B)).copied().flatten().is_some_and(|(_, d)| d.abs() < keep)));
    // Those the layer held go (the surface left them).
    let gone: Vec<IVec3> = gone.into_iter().filter(|bc| current.h.contains_key(bc)).collect();
    let every = if end { EVERY } else { B };
    let per = B / every + 1;
    let points: Vec<IVec3> = dest
        .iter()
        .flat_map(|bc| (0..per * per * per).map(move |q| bc * B + IVec3::new(q % per, (q / per) % per, q / (per * per)) * every))
        .collect::<HashSet<_>>()
        .into_iter()
        .filter(|p| !moves.contains_key(p))
        .collect();
    moves.extend(points.par_iter().map(|&i| (i, measure(i))).collect::<Vec<_>>());
    let t2 = Instant::now();
    let corners = |bc: IVec3| -> Corners { std::array::from_fn(|q| moves.get(&((bc + corner_offset(q as i32)) * B)).copied().flatten().map(|m| m.0)) };
    // Mid-stroke, a block whose corners moved less than a quarter step since it was last followed stays as it is.
    let same = |a: &Corners, b: &Corners| {
        a.iter().zip(b).all(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => (*x - *y).length() < 0.25,
            (None, None) => true,
            _ => false,
        })
    };
    let dest: Vec<IVec3> = if end { dest } else { dest.into_iter().filter(|bc| job.used.get(&(k, *bc)).is_none_or(|u| !same(u, &corners(*bc)))).collect() };
    let used: Vec<(IVec3, Corners)> = if end { Vec::new() } else { dest.iter().map(|&bc| (bc, corners(bc))).collect() };
    let built: LayerBlocks = dest
        .par_iter()
        .map(|&bc| {
            let at = |i: IVec3| moves.get(&i).copied().flatten();
            let corner = bc * B;
            // Interpolated from the corners measured (where a search found no surface, the others say). Samples
            // farther from the surface than any height reaches need none.
            let reach_out = band + tall;
            let offs: Vec<Option<Vec3>> = (0..N)
                .map(|i| {
                    let c = cell(i);
                    let base = c.div_euclid(IVec3::splat(every)) * every;
                    let f = (c - base).as_vec3() / every as f32;
                    let (mut o, mut dist, mut sum) = (Vec3::ZERO, 0.0, 0.0);
                    for q in 0..8 {
                        let w = corner_weight(q, f);
                        if let Some((m, e)) = at(corner + base + corner_offset(q) * every) {
                            o += w * m;
                            dist += w * e;
                            sum += w;
                        }
                    }
                    (sum > 1e-6 && (dist / sum).abs() < reach_out).then(|| o / sum)
                })
                .collect();
            // Where the block's heights come from: with none of the layer there, there is nothing to read.
            let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
            for (i, o) in offs.iter().enumerate() {
                if let Some(o) = o {
                    let g = (corner + cell(i)).as_vec3() + *o;
                    lo = lo.min(g.floor().as_ivec3());
                    hi = hi.max(g.ceil().as_ivec3());
                }
            }
            if lo.x > hi.x {
                return (bc, None);
            }
            let (lo, hi) = (lo - IVec3::splat(2), hi + IVec3::splat(2));
            let (blo, bhi) = (lo.div_euclid(IVec3::splat(B)), hi.div_euclid(IVec3::splat(B)));
            let span = bhi - blo + IVec3::ONE;
            if span.x as i64 * span.y as i64 * span.z as i64 <= 64 && !any_block(&layer0.h, lo, hi) {
                return (bc, None);
            }
            // Mid-stroke a preview, read linearly straight from the blocks; at the end with the cubic, from the source
            // region read once (unless it is large: a block the stroke stretched or tore across, read as needed).
            let (h0, m0) = (Sparse::new(&layer0.h, 0.0), Sparse::new(&layer0.m, 1.0));
            let dense = end.then(|| -> Option<(Dense, Dense, bool)> {
                let n = hi - lo + IVec3::ONE;
                if n.x as i64 * n.y as i64 * n.z as i64 > 32 * 32 * 32 {
                    return None;
                }
                let m = Dense::of(&layer0.m, lo, hi, 1.0);
                let shown = m.all(1.0);
                Some((Dense::of(&layer0.h, lo, hi, 0.0), m, shown))
            });
            let read = |g: Vec3| match &dense {
                Some(Some((h, m, shown))) => (h.cubic(g), if *shown { 1.0 } else { m.cubic(g).clamp(0.0, 1.0) }),
                Some(None) => (h0.cubic(g), m0.cubic(g).clamp(0.0, 1.0)),
                None => (h0.linear(g), m0.linear(g).clamp(0.0, 1.0)),
            };
            let mut h = Box::new([0.0f32; N]);
            let mut m = Box::new([1.0f32; N]);
            let mut any = false;
            for i in 0..N {
                let x = corner + cell(i);
                let (hv, mv) = match offs[i] {
                    None => (0.0, 1.0),
                    Some(o) if o.length() < SNAP => (h0.at(x), m0.at(x)),
                    Some(o) => read(x.as_vec3() + o),
                };
                h[i] = hv.clamp(-limit, limit);
                m[i] = mv;
                any |= h[i].abs() > 1e-4 * v;
            }
            (bc, any.then(|| (Stored::from(h), Stored::from(m))))
        })
        .collect();
    let mut built = built;
    built.extend(gone.into_iter().map(|bc| (bc, None)));
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    let stats = FollowStats { ms: 0.0, blocks: built.len(), points: points.len() + corner_points.len(), parts_ms: [ms(t0, t1), ms(t1, t2), ms(t2, Instant::now()), 0.0] };
    (built, used, stats)
}

impl Layered {
    /// Put the work `shown` holds at levels `from` and finer in a layer of its own over the rest, at the finest level
    /// that work reaches. `shown` becomes what is drawn: the form with the layer on it.
    pub fn new(shown: &mut LevelVolume, from: usize) -> Layered {
        assert!(from >= 1 && from < shown.levels());
        // The finest level the work reaches (the finest of all, on a volume with none yet).
        let level = shown.keys().iter().map(|k| k.0 as usize).filter(|&l| l >= from).max().unwrap_or(shown.levels() - 1);
        let full = shown.snapshot();
        let mut form = shown.snapshot();
        for l in from..form.levels() {
            form.remove_level(l);
        }
        form.drain_dirty();
        form.clear_history();
        let mut out = Layered::with_form(form, from, vec![Layer::new("Detail", level)]);
        let keys = out.finest_under(&full, &full.keys(), level);
        out.take_heights(shown, &full, 0, &keys, &[]);
        shown.clear_history();
        out
    }

    fn with_form(form: LevelVolume, from: usize, stack: Vec<Layer>) -> Layered {
        Layered {
            form,
            from,
            stack,
            active: 0,
            edit: None,
            start: None,
            touched: HashSet::default(),
            history: Vec::new(),
            redo: Vec::new(),
            last: FollowStats::default(),
            follows: 0,
            deferred: false,
            pending: HashSet::default(),
            ending: false,
            running: false,
            covered: Vec::new(),
            used: HashMap::default(),
            changed: HashSet::default(),
            base_epoch: 0,
        }
    }

    /// Blocks of level `l` under `keys` (of levels `from` and finer) near the surface of `vol`.
    fn finest_under(&self, vol: &LevelVolume, keys: &[Key], l: usize) -> Vec<IVec3> {
        let (v, band, origin) = (vol.voxel(l), vol.band(l), vol.origin());
        let near = 4.0 * 3f32.sqrt() * v + band + MOST * vol.voxel(self.from);
        let mut out: HashSet<IVec3> = HashSet::default();
        for &(m, bc) in keys {
            let m = m as usize;
            if m < self.from {
                continue;
            }
            if m > l {
                out.insert(bc >> (m - l) as i32);
                continue;
            }
            let k = 1i32 << (l - m);
            for q in 0..k * k * k {
                out.insert(bc * k + IVec3::new(q % k, (q / k) % k, q / (k * k)));
            }
        }
        let out: Vec<IVec3> = out.into_iter().collect();
        out.into_par_iter().filter(|bc| vol.value_upto(origin + (*bc * B + IVec3::splat(B / 2)).as_vec3() * v, l, false).abs() < near).collect()
    }

    /// Make layer `k`'s heights over its blocks `keys` what `detail` stands out of the form there, less the other
    /// layers (keeping its mask, and dividing by it), and show them; `shown` then holds the form's blocks again where it
    /// held work the layers now have. `also` are other shown blocks the work changed.
    fn take_heights(&mut self, shown: &mut LevelVolume, detail: &LevelVolume, k: usize, keys: &[IVec3], also: &[Key]) {
        let l = self.stack[k].level;
        let (form, stack) = (&self.form, &self.stack);
        let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
        let below = self.from - 1;
        let limit = MOST * form.voxel(self.from);
        let built: Vec<(IVec3, Option<Box<[f32; N]>>)> = keys
            .par_iter()
            .map(|&bc| {
                let under = Below::of(form, stack, k, below);
                let base = |p: Vec3| under.value(p, below);
                let top = |p: Vec3| detail.value_upto(p, detail.levels() - 1, false);
                let others = offsets(stack, l, bc * B, bc * B + IVec3::splat(B - 1), true, Some(k));
                let mask = stack[k].m.get(&bc).cloned();
                let old = stack[k].h.get(&bc).cloned();
                let mut h = Box::new([0.0f32; N]);
                let mut any = false;
                for (i, out) in h.iter_mut().enumerate() {
                    let m = mask.as_ref().map_or(1.0, |m| m[i]);
                    let was = old.as_ref().map_or(0.0, |h| h[i]);
                    *out = was;
                    // Where the mask hides the layer, painting cannot say what it holds.
                    if m < 0.05 {
                        any |= out.abs() > 1e-4 * v;
                        continue;
                    }
                    let x = origin + (bc * B + cell(i)).as_vec3() * v;
                    if base(x).abs() > band + limit {
                        *out = 0.0;
                        continue;
                    }
                    let n = under.normal(x);
                    let Some(foot) = root(&base, x, n, 0.25 * v, 2.0 * v, 8.0 * band) else { continue };
                    let n = under.normal(foot);
                    let Some(y) = root(&top, foot, n, 0.1 * v, 0.5 * v, limit + v) else { continue };
                    *out = (((y - foot).dot(n) - others[i]) / m).clamp(-limit, limit);
                    any |= out.abs() > 1e-4 * v;
                }
                (bc, any.then_some(h))
            })
            .collect();
        let layer = &mut self.stack[k];
        for (bc, h) in built {
            match h {
                Some(h) => {
                    layer.h.insert(bc, Stored::from(h));
                    if !layer.m.contains_key(&bc) {
                        layer.m.insert(bc, Stored::from(Box::new([1.0f32; N])));
                    }
                }
                None => {
                    layer.h.remove(&bc);
                    layer.m.remove(&bc);
                }
            }
        }
        // The form's blocks again, coarsest first, where `shown` held work that the layers have (and is not theirs).
        let mut back: Vec<Key> = detail.keys().into_iter().chain(shown.keys()).filter(|k| (k.0 as usize) >= self.from && !composed(&self.stack, k.0 as usize, k.1)).collect();
        back.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        back.dedup();
        for key in back {
            shown.mirror(&self.form, key);
        }
        let mut changes: Vec<(usize, IVec3)> = keys.iter().map(|&bc| (l, bc)).collect();
        changes.extend(also.iter().filter(|a| composed(&self.stack, a.0 as usize, a.1)).map(|a| (a.0 as usize, a.1)));
        self.compose(shown, &changes);
    }

    /// A stroke on what is shown (`shown`'s last stroke, on the active layer) has ended: take it into the layer's
    /// heights, to undo with the layers' history rather than with the stroke's own.
    pub fn take_detail(&mut self, shown: &mut LevelVolume) {
        self.base_epoch += 1;
        let keys = shown.last_stroke_keys();
        shown.forget_last();
        if keys.is_empty() || self.stack.is_empty() {
            return;
        }
        let was = self.stack.clone();
        let detail = shown.snapshot();
        let k = self.active;
        let blocks = self.finest_under(&detail, &keys, self.stack[k].level);
        // Coarser blocks the stroke changed (making parents, moving form down) go back to the form's.
        let mut coarse: Vec<Key> = keys.iter().copied().filter(|k| (k.0 as usize) < self.from).collect();
        coarse.sort_unstable_by_key(|k| k.0);
        for key in coarse {
            shown.mirror(&self.form, key);
        }
        self.stack[k].visible = true;
        self.take_heights(shown, &detail, k, &blocks, &keys);
        self.history.push(Undo::Layers(was));
        self.redo.clear();
    }

    /// The first level the layers took work from: strokes on a layer work there or finer, on the form coarser.
    pub fn from(&self) -> usize {
        self.from
    }

    /// The form under the layers: brushes find the surface on it.
    pub fn form(&self) -> &LevelVolume {
        &self.form
    }

    /// The active layer's level (strokes on it work no finer).
    pub fn level(&self) -> usize {
        self.stack.get(self.active).map_or(self.form.levels() - 1, |x| x.level)
    }

    /// The tallest height a layer holds (for checks).
    pub fn tallest(&self) -> f32 {
        self.stack.iter().map(tallest).fold(0.0f32, f32::max)
    }

    /// The active layer's blocks.
    pub fn blocks(&self) -> usize {
        self.stack.get(self.active).map_or(0, |x| x.len())
    }

    /// The layers, bottom first.
    pub fn layers(&self) -> Vec<LayerInfo> {
        self.stack.iter().map(|x| LayerInfo { name: x.name.clone(), level: x.level, visible: x.visible, blocks: x.len(), generator: x.scatter.is_some() }).collect()
    }

    /// The layer strokes on the layers (and mask strokes) work on.
    pub fn active(&self) -> usize {
        self.active
    }

    pub fn set_active(&mut self, k: usize) {
        if k < self.stack.len() {
            self.active = k;
        }
    }

    /// A new empty layer at `level` (no coarser than the layers' first level) on top, made active.
    pub fn add_layer(&mut self, name: &str, level: usize) -> usize {
        self.base_epoch += 1;
        let level = level.clamp(self.from, self.form.levels() - 1);
        self.history.push(Undo::Layers(self.stack.clone()));
        self.redo.clear();
        self.stack.push(Layer::new(name, level));
        self.active = self.stack.len() - 1;
        self.active
    }

    /// A new generator layer on top, scattering `spec` over the form, made active. It changes the field at levels up to
    /// `level` (any, coarse or fine: a generator's resolution is its own).
    pub fn add_generator(&mut self, shown: &mut LevelVolume, name: &str, level: usize, spec: &GeneratorSpec) -> usize {
        let level = level.min(self.form.levels() - 1);
        self.history.push(Undo::Layers(self.stack.clone()));
        self.redo.clear();
        let mut layer = Layer::new(name, level);
        layer.scatter = Some(GenPart::new(&self.form, VolumeGen::build(&self.form, spec, None), level));
        let changes = layer.all_keys();
        self.stack.push(layer);
        self.active = self.stack.len() - 1;
        self.compose(shown, &changes);
        self.active
    }

    /// The base field's epoch (see `base_epoch`).
    pub fn base_epoch(&self) -> u64 {
        self.base_epoch
    }

    /// Block `key`'s 9x9x9 samples composed without the generators, as the GPU joins them to (see [`base_block`]).
    pub fn base_samples(&self, key: Key) -> Box<[f32; 729]> {
        let (l, bc) = (key.0 as usize, key.1);
        let (v, origin) = (self.form.voxel(l), self.form.origin());
        let lo = origin + (bc * B).as_vec3() * v;
        let hi = lo + Vec3::splat(B as f32 * v);
        let join = self.stack.iter().filter(|x| x.visible).filter_map(|x| x.scatter.as_ref()).any(|g| g.g.touches(lo, hi, g.g.pad() + v));
        base_block(&self.form, &self.stack, l, self.from - 1, bc, join)
    }

    /// Whether the GPU can join the generators for an edit: no detail layer shown above one (those are composed in the
    /// stack's order, on the CPU).
    pub fn gpu_can_join(&self) -> bool {
        let first = self.stack.iter().position(|x| x.visible && x.scatter.is_some());
        let ordered = first.is_some_and(|g| self.stack[g + 1..].iter().any(|x| x.visible && x.scatter.is_none()));
        !ordered
    }

    /// Change generator layer `k`'s parameters, the GPU to show it at its own level: the stack takes the new scatter
    /// (and the blocks it no longer changes the form's again, and blocks it now changes their base, so they exist);
    /// the GPU joins the generators to the blocks the plan lists. Its exact field comes from a job
    /// ([`Self::generator_take`]), as after a preview. `None` (and nothing changed) where the GPU cannot.
    pub fn gpu_generator(&mut self, shown: &mut LevelVolume, k: usize, spec: &GeneratorSpec) -> Option<GpuPlan> {
        let old = self.stack.get(k).and_then(|x| x.scatter.clone())?;
        // At level 1 or coarser the CPU's preview is the layer at its own level already, and sooner.
        if !self.gpu_can_join() || self.stack[k].level <= PREVIEW {
            return None;
        }
        if self.edit.is_none() && old.g.spec != *spec {
            self.history.push(Undo::Layers(self.stack.clone()));
            self.redo.clear();
        }
        let own = self.stack[k].level;
        let mut changes = self.stack[k].all_keys();
        let mut part = GenPart::new(&self.form, VolumeGen::build(&self.form, spec, Some(&old.g)), own);
        // Shown by the GPU: the job brings the field at its own level.
        part.upto = usize::MAX;
        self.stack[k].scatter = Some(part);
        changes.extend(self.stack[k].all_keys());
        let targets = affected(&self.stack, &changes);
        // On the CPU: blocks no longer changed by any layer, the form's again; blocks new to the layer, their base.
        let (gone, kept): (Vec<(usize, IVec3)>, Vec<(usize, IVec3)>) = targets.into_iter().partition(|&(l, bc)| !composed(&self.stack, l, bc));
        let fresh: Vec<(usize, IVec3, Option<Stored>)> = kept
            .par_iter()
            .filter(|&&(l, bc)| !shown.exists(l, bc))
            .map(|&(l, bc)| {
                let b = self.base_samples((l as u8, bc));
                let mut own = Box::new([0.0f32; N]);
                for (i, o) in own.iter_mut().enumerate() {
                    let c = cell(i);
                    *o = b[(c.x + 9 * c.y + 81 * c.z) as usize].clamp(-self.form.band(l), self.form.band(l));
                }
                (l, bc, Some(Stored::from(own)))
            })
            .collect();
        let mut put: Vec<(usize, IVec3, Option<Stored>)> = gone.into_iter().map(|(l, bc)| (l, bc, None)).collect();
        put.extend(fresh);
        self.put_shown(shown, put, &self.form);
        // The plan: the generators shown, and the instances near each block.
        let mut gens = Vec::new();
        let mut instances = Vec::new();
        let mut parts: Vec<(&VolumeGen, u32, &Shared<IVec3, Stored>, usize)> = Vec::new();
        for x in self.stack.iter().filter(|x| x.visible) {
            if let Some(g) = &x.scatter {
                parts.push((&g.g, instances.len() as u32, &x.m, x.level));
                gens.push(g.g.gpu_params());
                instances.extend(g.g.gpu_instances());
            }
        }
        let origin = self.form.origin();
        type Lists = Vec<(Vec<u32>, Option<Box<[f32; 729]>>)>;
        let form = &self.form;
        let per: Vec<(Key, Lists)> = kept
            .par_iter()
            .map(|&(l, bc)| {
                let v = form.voxel(l);
                let lo = origin + (bc * B).as_vec3() * v;
                let hi = lo + Vec3::splat(B as f32 * v);
                let mut near = Vec::new();
                let lists = parts
                    .iter()
                    .map(|(g, first, m, gl)| {
                        g.near(lo, hi, g.pad() + v, &mut near);
                        // The mask at the block's samples, where it is painted near (else whole): any of its blocks over
                        // the block's box, at the mask's lattice.
                        let painted = !near.is_empty() && !m.is_empty() && {
                            let mv = form.voxel(*gl);
                            let (a, z) = (((lo - origin) / mv).floor().as_ivec3() - IVec3::ONE, ((hi - origin) / mv).ceil().as_ivec3() + IVec3::ONE);
                            let (a, z) = (a.div_euclid(IVec3::splat(B)), z.div_euclid(IVec3::splat(B)));
                            (a.z..=z.z).any(|bz| (a.y..=z.y).any(|by| (a.x..=z.x).any(|bx| m.contains_key(&IVec3::new(bx, by, bz)))))
                        };
                        let mask = painted.then(|| {
                            let mut out = Box::new([1.0f32; 729]);
                            for (k, o) in out.iter_mut().enumerate() {
                                let i = bc * B + IVec3::new(k as i32 % 9, (k as i32 / 9) % 9, k as i32 / 81);
                                *o = mask_at(form, m, *gl, origin + i.as_vec3() * v);
                            }
                            out
                        });
                        let mask = mask.filter(|m| m.iter().any(|x| *x < 0.9999));
                        (near.iter().map(|j| j + first).collect(), mask)
                    })
                    .collect();
                ((l as u8, bc), lists)
            })
            .collect();
        let mut cands = Vec::new();
        let mut masks = Vec::new();
        let blocks = per
            .into_iter()
            .map(|(key, lists)| {
                let ranges = lists
                    .into_iter()
                    .map(|(c, mask)| {
                        let first = cands.len() as u32;
                        cands.extend(c);
                        let mi = match mask {
                            Some(m) => {
                                masks.push(m);
                                masks.len() as u32 - 1
                            }
                            None => u32::MAX,
                        };
                        [first, cands.len() as u32 - first, mi, 0]
                    })
                    .collect();
                (key, ranges)
            })
            .collect();
        Some(GpuPlan { gens, instances, cands, blocks, masks, epoch: self.base_epoch })
    }

    /// Generator layer `k`'s parameters.
    pub fn generator(&self, k: usize) -> Option<&GeneratorSpec> {
        self.stack.get(k)?.scatter.as_ref().map(|g| &g.g.spec)
    }

    /// Whether generator layer `k` changes block `bc` of level `l`, for checks.
    pub fn generator_holds(&self, k: usize, l: usize, bc: IVec3) -> bool {
        self.stack.get(k).is_some_and(|x| x.scatter.is_some() && x.holds(l, bc))
    }

    /// Generator layer `k`'s instances as placed, for checks.
    pub fn generator_instances(&self, k: usize) -> Vec<super::Instance> {
        self.stack.get(k).and_then(|x| x.scatter.as_ref()).map_or(Vec::new(), |g| g.g.instances().to_vec())
    }

    /// Change generator layer `k`'s parameters (scattered again only if the count, seed or spacing changed). Within
    /// an edit ([`Self::begin_edit`]) a run of changes undoes as one.
    pub fn set_generator(&mut self, shown: &mut LevelVolume, k: usize, spec: &GeneratorSpec) {
        self.set_generator_at(shown, k, spec, false);
    }

    /// [`Self::set_generator`], shown no finer than level 1 for now: a slider drag's preview, many times cheaper on a
    /// fine layer. [`Self::set_generator`] (or [`Self::end_edit`]) shows it at its own level again.
    pub fn preview_generator(&mut self, shown: &mut LevelVolume, k: usize, spec: &GeneratorSpec) {
        self.set_generator_at(shown, k, spec, true);
    }

    fn set_generator_at(&mut self, shown: &mut LevelVolume, k: usize, spec: &GeneratorSpec, preview: bool) {
        let Some(old) = self.stack.get(k).and_then(|x| x.scatter.clone()) else { return };
        let own = self.stack[k].level;
        let upto = if preview { own.min(PREVIEW) } else { own };
        if old.g.spec == *spec && old.upto == upto {
            return;
        }
        if self.edit.is_none() && old.g.spec != *spec {
            self.history.push(Undo::Layers(self.stack.clone()));
            self.redo.clear();
        }
        let mut changes = self.stack[k].all_keys();
        let mut part = GenPart::new(&self.form, VolumeGen::build(&self.form, spec, Some(&old.g)), upto);
        part.upto = upto;
        self.stack[k].scatter = Some(part);
        changes.extend(self.stack[k].all_keys());
        self.compose(shown, &changes);
    }

    /// Remove layer `k`.
    pub fn remove_layer(&mut self, shown: &mut LevelVolume, k: usize) {
        self.base_epoch += 1;
        if k >= self.stack.len() {
            return;
        }
        self.history.push(Undo::Layers(self.stack.clone()));
        self.redo.clear();
        let gone = self.stack.remove(k);
        self.active = self.active.min(self.stack.len().saturating_sub(1));
        let changes = gone.all_keys();
        self.compose(shown, &changes);
    }

    pub fn rename(&mut self, k: usize, name: &str) {
        if let Some(x) = self.stack.get_mut(k) {
            x.name = name.to_string();
        }
    }

    /// Show or hide layer `k`.
    pub fn set_visible(&mut self, shown: &mut LevelVolume, k: usize, on: bool) {
        self.base_epoch += 1;
        let Some(x) = self.stack.get_mut(k) else { return };
        if x.visible == on {
            return;
        }
        x.visible = on;
        let changes = x.all_keys();
        self.compose(shown, &changes);
    }

    /// Hold layer `k` at another level: finer keeps its detail on a denser lattice; coarser keeps what that lattice
    /// can hold (averaged over each new step).
    pub fn set_level(&mut self, shown: &mut LevelVolume, k: usize, level: usize) {
        self.base_epoch += 1;
        if let Some(part) = self.stack.get(k).and_then(|x| x.scatter.clone()) {
            // A generator: the same scatter, changing the field to another level.
            let level = level.min(self.form.levels() - 1);
            if self.stack[k].level == level {
                return;
            }
            self.history.push(Undo::Layers(self.stack.clone()));
            self.redo.clear();
            let mut changes = self.stack[k].all_keys();
            self.stack[k].level = level;
            self.stack[k].m = Shared::default();
            self.stack[k].scatter = Some(GenPart::new(&self.form, part.g, level));
            changes.extend(self.stack[k].all_keys());
            self.compose(shown, &changes);
            return;
        }
        let level = level.clamp(self.from, self.form.levels() - 1);
        let Some(old) = self.stack.get(k).cloned() else { return };
        if old.level == level {
            return;
        }
        self.history.push(Undo::Layers(self.stack.clone()));
        self.redo.clear();
        let mut targets: HashSet<IVec3> = HashSet::default();
        for bc in old.h.keys() {
            if level > old.level {
                let s = 1i32 << (level - old.level);
                for q in 0..s * s * s {
                    targets.insert(*bc * s + IVec3::new(q % s, (q / s) % s, q / (s * s)));
                }
            } else {
                targets.insert(*bc >> (old.level - level) as i32);
            }
        }
        let targets: Vec<IVec3> = targets.into_iter().collect();
        let v = self.form.voxel(level);
        let limit = MOST * self.form.voxel(self.from);
        let built: Vec<(IVec3, Option<(Stored, Stored)>)> = targets
            .par_iter()
            .map(|&bc| {
                let (hs, ms) = (Sparse::new(&old.h, 0.0), Sparse::new(&old.m, 1.0));
                let mut h = Box::new([0.0f32; N]);
                let mut m = Box::new([1.0f32; N]);
                let mut any = false;
                for i in 0..N {
                    let x = bc * B + cell(i);
                    let (hv, mv) = if level > old.level {
                        let g = x.as_vec3() / (1i32 << (level - old.level)) as f32;
                        (hs.cubic(g), ms.linear(g).clamp(0.0, 1.0))
                    } else {
                        // The mean over the new step (a tent over the old lattice points within it).
                        let s = 1i32 << (old.level - level);
                        let c = x * s;
                        let (mut sh, mut sm, mut sw) = (0.0, 0.0, 0.0);
                        for dz in -s + 1..s {
                            for dy in -s + 1..s {
                                for dx in -s + 1..s {
                                    let w = ((s - dx.abs()) * (s - dy.abs()) * (s - dz.abs())) as f32;
                                    let j = c + IVec3::new(dx, dy, dz);
                                    sh += w * hs.at(j);
                                    sm += w * ms.at(j);
                                    sw += w;
                                }
                            }
                        }
                        (sh / sw, sm / sw)
                    };
                    h[i] = hv.clamp(-limit, limit);
                    m[i] = mv;
                    any |= h[i].abs() > 1e-4 * v;
                }
                (bc, any.then(|| (Stored::from(h), Stored::from(m))))
            })
            .collect();
        let mut layer = Layer { name: old.name.clone(), level, visible: old.visible, h: Shared::default(), m: Shared::default(), scatter: None };
        for (bc, hm) in built {
            if let Some((h, m)) = hm {
                layer.h.insert(bc, h);
                layer.m.insert(bc, m);
            }
        }
        let mut changes: Vec<(usize, IVec3)> = old.h.keys().map(|bc| (old.level, *bc)).collect();
        changes.extend(layer.h.keys().map(|bc| (level, *bc)));
        self.stack[k] = layer;
        self.compose(shown, &changes);
    }

    /// Begin a stroke on the form (the last one's following all in: see [`Self::settle`]).
    pub fn begin_stroke(&mut self) {
        self.start = Some((self.form.snapshot(), self.stack.clone()));
        self.touched.clear();
        self.pending.clear();
        self.used.clear();
        self.changed.clear();
        self.follows = 0;
        self.covered = vec![HashSet::default(); self.form.levels()];
        for k in 0..self.stack.len() {
            if self.stack[k].scatter.is_some() {
                for (l, bc) in self.stack[k].all_keys() {
                    self.cover(l, &[bc]);
                }
                continue;
            }
            let keys: Vec<IVec3> = self.stack[k].h.keys().copied().collect();
            self.cover(self.stack[k].level, &keys);
        }
        self.form.begin_stroke();
    }

    /// Whether a generator shown reaches block `bc` of level `l` (a form block a stroke made there shows with it on).
    fn generators_reach(&self, l: usize, bc: IVec3) -> bool {
        let (v, origin) = (self.form.voxel(l), self.form.origin());
        let lo = origin + (bc * B).as_vec3() * v;
        let hi = lo + Vec3::splat(B as f32 * v);
        self.stack.iter().filter(|x| x.visible).filter_map(|x| x.scatter.as_ref()).any(|g| g.g.touches(lo, hi, g.g.pad() + self.form.band(l) + v))
    }

    /// Mark the blocks over blocks `keys` of level `l`, at every level up to it.
    fn cover(&mut self, l: usize, keys: &[IVec3]) {
        if self.covered.len() <= l {
            return;
        }
        for &bc in keys {
            for m in 0..=l {
                self.covered[m].insert(bc >> (l - m) as i32);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plane_dab(&mut self, shown: &mut LevelVolume, l: usize, at: Vec3, along: Vec3, r: f32, h: f32, hardness: f32, alpha: f32, trim: bool) -> Option<(Vec3, LevelDab)> {
        let out = self.form.plane_dab(l, at, along, r, h, hardness, alpha, trim);
        self.dabbed(shown);
        out
    }

    pub fn ball_dab(&mut self, shown: &mut LevelVolume, l: usize, c: Vec3, r: f32, k: f32, carve: bool) -> LevelDab {
        let out = self.form.ball_dab(l, c, r, k, carve);
        self.dabbed(shown);
        out
    }

    /// After a dab on the form: what it changed away from the layers shows at once; over them, it waits for a follow
    /// (at once, unless deferred), which shows it with the layers on it.
    fn dabbed(&mut self, shown: &mut LevelVolume) {
        self.base_epoch += 1;
        let dirty = self.form.drain_dirty();
        self.touched.extend(dirty.iter().copied());
        if self.start.is_none() {
            for &k in &dirty {
                shown.mirror(&self.form, k);
            }
            let changes: Vec<(usize, IVec3)> = self.stack.iter().flat_map(|x| under_all(&dirty, x)).collect();
            self.compose(shown, &changes);
            let stale = self.stale_under(shown, &dirty, &self.form);
            self.tidy(shown, stale, None);
            return;
        }
        for k in dirty {
            let m = k.0 as usize;
            if self.covered.get(m).is_some_and(|c| c.contains(&k.1)) || self.generators_reach(m, k.1) {
                self.pending.insert(k);
            } else {
                shown.mirror(&self.form, k);
            }
        }
        if !self.deferred {
            self.settle(shown);
        }
    }

    /// End the stroke on the form; the layers follow the form as it is now, its distances made again (at once, or in a
    /// last job when deferred).
    pub fn end_stroke(&mut self, shown: &mut LevelVolume) -> StrokeEnd {
        let out = self.form.end_stroke();
        let dirty = self.form.drain_dirty();
        self.touched.extend(dirty);
        if self.start.is_some() {
            self.ending = true;
        }
        if !self.deferred {
            self.settle(shown);
        }
        out
    }

    /// End the stroke on the form on another thread: [`EndJob::run`] works out the form's end on a copy, and
    /// [`Self::end_form_apply`] takes it, leaving the layers' last follow to do. `None` when the form's stroke changed
    /// nothing (it has ended, and the last follow is left to do).
    pub fn end_form_take(&mut self) -> Option<EndJob> {
        let job = self.form.end_stroke_take();
        if job.is_none() && self.start.is_some() {
            self.ending = true;
        }
        job
    }

    pub fn end_form_apply(&mut self, done: EndDone) -> StrokeEnd {
        self.base_epoch += 1;
        let out = self.form.end_stroke_apply(done);
        let dirty = self.form.drain_dirty();
        self.touched.extend(dirty);
        if self.start.is_some() {
            self.ending = true;
        }
        out
    }

    /// Whether following is still to do (queued, under way, or the stroke's last).
    pub fn busy(&self) -> bool {
        self.running || self.ending || !self.pending.is_empty()
    }

    /// The next follow to work out on another thread: what dabs queued, or once the pen is up, the stroke's last (over
    /// everything it touched, exactly). `None` while one is under way, or with nothing to do.
    pub fn follow_take(&mut self) -> Option<FollowJob> {
        if self.running {
            return None;
        }
        let (form0, stack0) = self.start.as_ref()?;
        let end = self.ending;
        let region: Vec<Key> = if end {
            self.pending.clear();
            self.touched.iter().copied().collect()
        } else {
            self.pending.drain().collect()
        };
        if region.is_empty() && !end {
            return None;
        }
        self.running = true;
        Some(FollowJob {
            form: self.form.snapshot(),
            form0: form0.snapshot(),
            stack0: stack0.clone(),
            stack: self.stack.clone(),
            from: self.from,
            region,
            end,
            used: if end { HashMap::default() } else { self.used.clone() },
            also: if end { self.changed.iter().copied().collect() } else { Vec::new() },
        })
    }

    /// Take what a follow worked out: the layers' blocks, and what is shown over the blocks it covered (the form as the
    /// job read it, the layers on it).
    pub fn follow_apply(&mut self, shown: &mut LevelVolume, done: FollowDone) {
        self.base_epoch += 1;
        let t = Instant::now();
        self.running = false;
        let FollowDone { form, region, end, layers, gens, shown: blocks, used, mut stats } = done;
        for (k, part) in gens {
            if let Some(x) = self.stack.get_mut(k).filter(|x| x.scatter.is_some()) {
                x.scatter = Some(part);
                let keys = x.all_keys();
                for (l, bc) in keys {
                    self.cover(l, &[bc]);
                }
            }
        }
        let mut sorted = region.clone();
        sorted.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        // (The layers' levels only where no layer is composed: their blocks there are composed below, or stay.)
        for &key in &sorted {
            if !composed(&self.stack, key.0 as usize, key.1) {
                shown.mirror(&form, key);
            }
        }
        for (k, built) in layers {
            let level = self.stack[k].level;
            if !end {
                self.changed.extend(built.iter().map(|(bc, _)| (k, *bc)));
            }
            let mut made = Vec::new();
            let layer = &mut self.stack[k];
            for (bc, hm) in built {
                match hm {
                    Some((h, m)) => {
                        layer.h.insert(bc, h);
                        layer.m.insert(bc, m);
                        made.push(bc);
                    }
                    None => {
                        layer.h.remove(&bc);
                        layer.m.remove(&bc);
                    }
                }
            }
            self.cover(level, &made);
        }
        self.used.extend(used);
        self.put_shown(shown, blocks, &form);
        let stale = self.stale_under(shown, &region, &form);
        self.tidy(shown, stale, Some(&form));
        stats.ms += t.elapsed().as_secs_f64() * 1e3;
        self.last = stats;
        self.follows += 1;
        if end {
            self.ending = false;
            self.used.clear();
            self.changed.clear();
            if let Some((_, stack)) = self.start.take() {
                self.history.push(Undo::Form(stack));
                self.redo.clear();
            }
        }
    }

    /// Do the following still to do, here and now.
    pub fn settle(&mut self, shown: &mut LevelVolume) {
        while let Some(job) = self.follow_take() {
            let done = job.run();
            self.follow_apply(shown, done);
        }
    }

    /// Fade the active layer toward `to` (0 hides it, 1 shows it) within `r` of `c`, softly.
    pub fn paint_mask(&mut self, shown: &mut LevelVolume, c: Vec3, r: f32, to: f32) {
        self.base_epoch += 1;
        let Some(level) = self.stack.get(self.active).map(|x| x.level) else { return };
        let (v, origin) = (self.form.voxel(level), self.form.origin());
        let lo = ((c - Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
        let hi = ((c + Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
        let was = self.stack.clone();
        let mut changed = Vec::new();
        let layer = &mut self.stack[self.active];
        for bz in lo.z..=hi.z {
            for by in lo.y..=hi.y {
                for bx in lo.x..=hi.x {
                    let bc = IVec3::new(bx, by, bz);
                    // A generator's mask starts out whole wherever it changes the field.
                    let start = if layer.scatter.as_ref().is_some_and(|g| g.blocks[level].contains(&bc)) { Some(Stored::from(Box::new([1.0f32; N]))) } else { None };
                    let Some(m) = layer.m.get(&bc).cloned().or(start) else { continue };
                    let mut m = Box::new(*m);
                    for (i, x) in m.iter_mut().enumerate() {
                        let d = (origin + (bc * B + cell(i)).as_vec3() * v - c).length() / r;
                        let t = (1.0 - d).clamp(0.0, 1.0);
                        let w = t * t * (3.0 - 2.0 * t);
                        *x += w * (to - *x);
                    }
                    layer.m.insert(bc, Stored::from(m));
                    changed.push((level, bc));
                }
            }
        }
        // A generator's mask reaches what it changes at every level: its blocks within the stroke.
        if let Some(g) = layer.scatter.as_ref().filter(|_| !changed.is_empty()) {
            changed = g
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(l, b)| {
                    let s = self.form.voxel(l) * B as f32;
                    b.iter()
                        .filter(move |bc| {
                            let lo = origin + bc.as_vec3() * s;
                            (c.clamp(lo, lo + Vec3::splat(s)) - c).length() <= r + s
                        })
                        .map(move |bc| (l, *bc))
                })
                .collect();
        }
        if !changed.is_empty() {
            if self.edit.is_none() {
                self.history.push(Undo::Layers(was));
                self.redo.clear();
            }
            self.compose(shown, &changed);
        }
    }

    /// Begin an edit of a layer itself (a mask stroke): undone as one.
    pub fn begin_edit(&mut self) {
        self.edit = Some(self.stack.clone());
    }

    /// The next generator layer shown coarser than its own level (by a drag's preview) as a job composing it at its
    /// own: [`GenJob::run`] on another thread, then [`Self::generator_apply`]. `None` when none is.
    pub fn generator_take(&mut self) -> Option<GenJob> {
        let k = (0..self.stack.len()).find(|&k| self.stack[k].scatter.as_ref().is_some_and(|p| p.upto != self.stack[k].level))?;
        let preview = self.stack[k].scatter.clone()?;
        let own = self.stack[k].level;
        let part = GenPart::new(&self.form, preview.g.clone(), own);
        let mut stack = self.stack.clone();
        let mut changes = stack[k].all_keys();
        stack[k].scatter = Some(part.clone());
        changes.extend(stack[k].all_keys());
        let targets = affected(&stack, &changes);
        Some(GenJob { form: self.form.snapshot(), stack, from: self.from, k, part, preview: preview.blocks, targets })
    }

    /// Take what a [`GenJob`] composed, unless the layer changed meanwhile (then a later job brings it). True if taken.
    pub fn generator_apply(&mut self, shown: &mut LevelVolume, done: GenDone) -> bool {
        let GenDone { k, part, preview, blocks, .. } = done;
        let Some(x) = self.stack.get_mut(k) else { return false };
        if !x.scatter.as_ref().is_some_and(|p| std::sync::Arc::ptr_eq(&p.blocks, &preview)) {
            return false;
        }
        x.scatter = Some(part);
        self.put_shown(shown, blocks, &self.form);
        true
    }

    pub fn end_edit_in(&mut self, shown: &mut LevelVolume) {
        // A generator previewed coarser during the edit: at its own level again.
        for k in 0..self.stack.len() {
            if let Some(part) = self.stack[k].scatter.clone().filter(|p| p.upto != self.stack[k].level) {
                let spec = part.g.spec.clone();
                self.set_generator_at(shown, k, &spec, false);
            }
        }
        self.end_edit();
    }

    pub fn end_edit(&mut self) {
        if let Some(was) = self.edit.take() {
            self.history.push(Undo::Layers(was));
            self.redo.clear();
        }
    }

    pub fn undo(&mut self, shown: &mut LevelVolume) -> bool {
        let Some(entry) = self.history.pop() else { return false };
        let back = self.swap(shown, entry, true);
        self.redo.push(back);
        true
    }

    pub fn redo(&mut self, shown: &mut LevelVolume) -> bool {
        let Some(entry) = self.redo.pop() else { return false };
        let back = self.swap(shown, entry, false);
        self.history.push(back);
        true
    }

    fn swap(&mut self, shown: &mut LevelVolume, entry: Undo, undo: bool) -> Undo {
        self.base_epoch += 1;
        let (stack, form) = match entry {
            Undo::Form(s) => (s, true),
            Undo::Layers(s) => (s, false),
        };
        if form {
            if undo {
                self.form.undo();
            } else {
                self.form.redo();
            }
        }
        let old = std::mem::replace(&mut self.stack, stack);
        self.active = self.active.min(self.stack.len().saturating_sub(1));
        let region = self.form.drain_dirty();
        for &k in &region {
            shown.mirror(&self.form, k);
        }
        // Every layer block either stack holds that differs, or whose form changed.
        let mut changes: HashSet<(usize, IVec3)> = HashSet::default();
        let same_shape = old.len() == self.stack.len() && old.iter().zip(&self.stack).all(|(a, b)| a.level == b.level && a.visible == b.visible);
        if same_shape {
            for (a, b) in old.iter().zip(&self.stack) {
                if a.scatter.is_some() || b.scatter.is_some() {
                    let same = match (&a.scatter, &b.scatter) {
                        (Some(x), Some(y)) => std::sync::Arc::ptr_eq(&x.blocks, &y.blocks) && a.m.len() == b.m.len() && b.m.iter().all(|(bc, m)| a.m.get(bc).is_some_and(|o| std::sync::Arc::ptr_eq(o, m))),
                        _ => false,
                    };
                    if !same {
                        changes.extend(a.all_keys());
                        changes.extend(b.all_keys());
                    }
                    continue;
                }
                for (bc, h) in b.h.iter() {
                    if a.h.get(bc).is_none_or(|o| !std::sync::Arc::ptr_eq(o, h)) || a.m.get(bc).zip(b.m.get(bc)).is_none_or(|(x, y)| !std::sync::Arc::ptr_eq(x, y)) {
                        changes.insert((b.level, *bc));
                    }
                }
                for bc in a.h.keys() {
                    if !b.h.contains_key(bc) {
                        changes.insert((a.level, *bc));
                    }
                }
            }
        } else {
            for x in old.iter().chain(&self.stack) {
                changes.extend(x.all_keys());
            }
        }
        for x in &self.stack {
            changes.extend(under_all(&region, x));
        }
        let changes: Vec<(usize, IVec3)> = changes.into_iter().collect();
        self.compose(shown, &changes);
        let stale = self.stale_under(shown, &region, &self.form);
        self.tidy(shown, stale, None);
        if form { Undo::Form(old) } else { Undo::Layers(old) }
    }

    /// Blocks shown holds only as parents (neither the form's nor a layer's) under the form's blocks `region`: they read
    /// the form as it was when made, whatever the layers do there.
    fn stale_under(&self, shown: &LevelVolume, region: &[Key], form: &LevelVolume) -> Vec<Key> {
        let top = self.form.levels() - 1;
        let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); top + 1];
        for &(m, bc) in region {
            per[m as usize].insert(bc);
        }
        // Such blocks are parents of the layers' (read through their blocks, not every block shown).
        let mut out: HashSet<Key> = HashSet::default();
        for x in self.stack.iter().filter(|x| x.scatter.is_none()) {
            let l = x.level;
            for &bc in x.h.keys() {
                for m in 1..l {
                    let a = bc >> (l - m) as i32;
                    if !out.contains(&(m as u8, a)) && !form.exists(m, a) && !composed(&self.stack, m, a) && shown.exists(m, a) && (0..=m).any(|k| per[k].contains(&(a >> (m - k) as i32))) {
                        out.insert((m as u8, a));
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    /// Blocks `keys` that shown holds only as parents of finer ones (neither the form's nor a layer's): dropped where
    /// they hold none any more (finest first, so a parent left with none goes too), else made again from the level
    /// below (coarsest first, so each reads the one below as it is now).
    fn tidy(&self, shown: &mut LevelVolume, keys: Vec<Key>, form: Option<&LevelVolume>) {
        let form = form.unwrap_or(&self.form);
        let mut keys: Vec<Key> = keys
            .into_iter()
            .filter(|&(m, bc)| shown.exists(m as usize, bc) && !form.exists(m as usize, bc) && !composed(&self.stack, m as usize, bc))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        keys.sort_unstable_by_key(|k| (std::cmp::Reverse(k.0), k.1.z, k.1.y, k.1.x));
        let mut kept = Vec::with_capacity(keys.len());
        for (m, bc) in keys {
            let mu = m as usize;
            let children = mu + 1 < shown.levels() && (0..8).any(|q| shown.exists(mu + 1, bc * 2 + corner_offset(q)));
            if children {
                kept.push((m, bc));
            } else {
                shown.mirror(form, (m, bc));
            }
        }
        kept.reverse();
        for (m, bc) in kept {
            shown.refill(m as usize, bc);
        }
    }

    /// Put composed blocks into `shown` (or take its own away where no layer is composed there), coarsest first, and
    /// tidy their parents.
    fn put_shown(&self, shown: &mut LevelVolume, mut blocks: Vec<(usize, IVec3, Option<Stored>)>, form: &LevelVolume) {
        blocks.sort_unstable_by_key(|b| (b.0, b.1.z, b.1.y, b.1.x));
        let mut parents = Vec::new();
        for (l, bc, s) in blocks {
            match s {
                Some(s) => shown.put_stored(l, bc, s),
                None => shown.mirror(form, (l as u8, bc)),
            }
            parents.extend((1..l).map(|m| (m as u8, bc >> (l - m) as i32)));
        }
        self.tidy(shown, parents, Some(form));
    }

    /// Write what is shown for layer changes `changes` (level, block): those blocks, and the finer layers' over them.
    fn compose(&mut self, shown: &mut LevelVolume, changes: &[(usize, IVec3)]) {
        let targets = affected(&self.stack, changes);
        let built = compose_all(&self.form, &self.stack, self.from - 1, &targets);
        self.put_shown(shown, built, &self.form);
    }

    /// Write the form and the layers (the form as a volume file, the layers after it).
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        self.form.write_to(w)?;
        w.write_all(b"SLAY")?;
        for x in [2u32, self.from as u32, self.stack.len() as u32, self.active as u32] {
            w.write_all(&x.to_le_bytes())?;
        }
        for layer in &self.stack {
            w.write_all(&(layer.name.len() as u32).to_le_bytes())?;
            w.write_all(layer.name.as_bytes())?;
            w.write_all(&(layer.level as u32).to_le_bytes())?;
            w.write_all(&[layer.visible as u8])?;
            // What kind of layer (version 2): heights, or a generator (its parameters, anchors and mask).
            w.write_all(&[layer.scatter.is_some() as u8])?;
            if let Some(g) = &layer.scatter {
                let spec = serde_json::to_vec(&g.g.spec).map_err(std::io::Error::other)?;
                w.write_all(&(spec.len() as u32).to_le_bytes())?;
                w.write_all(&spec)?;
                w.write_all(&(g.g.anchors().len() as u32).to_le_bytes())?;
                for a in g.g.anchors() {
                    for x in a.p.to_array().into_iter().chain(a.n.to_array()) {
                        w.write_all(&x.to_le_bytes())?;
                    }
                }
                let mut keys: Vec<IVec3> = layer.m.keys().copied().collect();
                keys.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                w.write_all(&(keys.len() as u32).to_le_bytes())?;
                for bc in keys {
                    for x in bc.to_array() {
                        w.write_all(&x.to_le_bytes())?;
                    }
                    let buf: Vec<u8> = layer.m[&bc].iter().flat_map(|v| v.to_le_bytes()).collect();
                    w.write_all(&buf)?;
                }
                continue;
            }
            let mut keys: Vec<IVec3> = layer.h.keys().copied().collect();
            keys.sort_unstable_by_key(|v| (v.z, v.y, v.x));
            w.write_all(&(keys.len() as u32).to_le_bytes())?;
            let mut buf = Vec::with_capacity(N * 8);
            for bc in keys {
                for x in bc.to_array() {
                    w.write_all(&x.to_le_bytes())?;
                }
                buf.clear();
                let m = layer.m.get(&bc);
                for i in 0..N {
                    buf.extend_from_slice(&layer.h[&bc][i].to_le_bytes());
                }
                for i in 0..N {
                    buf.extend_from_slice(&m.map_or(1.0f32, |m| m[i]).to_le_bytes());
                }
                w.write_all(&buf)?;
            }
        }
        Ok(())
    }

    /// Read a volume file: what is shown, and the layers if it has any (`None` for a file of one volume).
    pub fn read_from(r: &mut impl Read, levels: usize) -> std::io::Result<(LevelVolume, Option<Layered>)> {
        use std::io::{Error, ErrorKind};
        let mut form = LevelVolume::read_from(r, levels)?;
        let mut magic = [0u8; 4];
        match r.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok((form, None)),
            Err(e) => return Err(e),
        }
        let bad = |m: &str| Error::new(ErrorKind::InvalidData, m.to_string());
        // A section after the volume that is not ours (the app's surface noise): no layers; the caller reads on.
        if &magic == b"SNOI" {
            return Ok((form, None));
        }
        if &magic != b"SLAY" {
            return Err(bad("not a layer section"));
        }
        let u32s = |r: &mut dyn Read| -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let version = u32s(r)?;
        if version != 1 && version != 2 {
            return Err(bad("unknown layer section version"));
        }
        let (from, count, active) = (u32s(r)? as usize, u32s(r)? as usize, u32s(r)? as usize);
        let top = form.levels() - 1;
        if from == 0 || from > top {
            return Err(bad("layers' first level out of range"));
        }
        let mut stack = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u32s(r)? as usize;
            let mut name = vec![0u8; len.min(4096)];
            r.read_exact(&mut name)?;
            let level = u32s(r)? as usize;
            let mut vis = [0u8; 1];
            r.read_exact(&mut vis)?;
            let mut kind = [0u8; 1];
            if version >= 2 {
                r.read_exact(&mut kind)?;
            }
            if kind[0] == 1 {
                let mut layer = Layer::new(&String::from_utf8_lossy(&name), level.min(top));
                layer.visible = vis[0] != 0;
                let len = u32s(r)? as usize;
                let mut spec = vec![0u8; len.min(1 << 20)];
                r.read_exact(&mut spec)?;
                let spec: GeneratorSpec = serde_json::from_slice(&spec).map_err(|e| bad(&format!("generator parameters: {e}")))?;
                let n = u32s(r)? as usize;
                let mut anchors = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    let mut f = [0f32; 6];
                    for x in &mut f {
                        *x = f32::from_bits(u32s(r)?);
                    }
                    anchors.push(Anchor { p: Vec3::new(f[0], f[1], f[2]), n: Vec3::new(f[3], f[4], f[5]) });
                }
                let masks = u32s(r)? as usize;
                let mut buf = vec![0u8; N * 4];
                for _ in 0..masks {
                    let mut c = [0i32; 3];
                    for x in &mut c {
                        *x = u32s(r)? as i32;
                    }
                    r.read_exact(&mut buf)?;
                    let m: Box<[f32; N]> = Box::new(std::array::from_fn(|i| f32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]])));
                    layer.m.insert(IVec3::from_array(c), Stored::from(m));
                }
                layer.scatter = Some(GenPart::new(&form, VolumeGen::from_anchors(&spec, anchors), layer.level));
                stack.push(layer);
                continue;
            }
            let level = level.clamp(from, top);
            let mut layer = Layer::new(&String::from_utf8_lossy(&name), level);
            layer.visible = vis[0] != 0;
            let blocks = u32s(r)? as usize;
            let mut buf = vec![0u8; N * 8];
            for _ in 0..blocks {
                let mut c = [0i32; 3];
                for x in &mut c {
                    *x = u32s(r)? as i32;
                }
                r.read_exact(&mut buf)?;
                let f = |i: usize| f32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
                let h: Box<[f32; N]> = Box::new(std::array::from_fn(f));
                let m: Box<[f32; N]> = Box::new(std::array::from_fn(|i| f(N + i)));
                let bc = IVec3::from_array(c);
                layer.h.insert(bc, Stored::from(h));
                layer.m.insert(bc, Stored::from(m));
            }
            stack.push(layer);
        }
        form.drain_dirty();
        let mut shown = form.snapshot();
        let mut out = Layered::with_form(form, from, stack);
        out.active = active.min(out.stack.len().saturating_sub(1));
        let changes: Vec<(usize, IVec3)> = out.stack.iter().flat_map(|x| x.all_keys()).collect();
        out.compose(&mut shown, &changes);
        shown.clear_history();
        Ok((shown, Some(out)))
    }
}

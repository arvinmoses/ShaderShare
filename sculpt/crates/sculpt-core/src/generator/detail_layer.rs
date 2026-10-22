//! A detail layer over a sculpted form: fine work kept as heights over the form's surface, with a mask, so sculpting
//! the form carries it along (as sculpting a lower subdivision level does), and the mask fades it anywhere.
//!
//! * **Heights.** At each lattice point of one fine level `L` near the form's surface, the layer holds `h`, how far the
//!   detail stands out of the form at the point of its surface nearest (the same along the normal), and a mask `m`
//!   from 0 to 1. What is shown is the form pushed out by `m * h`: the field `φ - m h`, where `φ` is the form's.
//! * **Following the form.** When a stroke moves the form, each new surface point takes the height of the point of
//!   the surface it came from: found along the normal (or the stroke's push) by where the fields change sign, which
//!   holds mid-stroke, when the fields are not distances yet. The move is measured every four lattice steps and
//!   interpolated, so it is smooth; the heights are read from the layer as the stroke began, so a stroke resamples
//!   them once, never a dab after a dab, and where the surface hardly moved they are copied exactly.
//! * **Shown.** A second volume holds what is drawn: the form's blocks, and at level `L` the composed field, made a
//!   distance again near the surface (its slope can be steeper than one on a groove's wall) so the extractor's
//!   crossings and the 16-bit samples stay exact.

use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::level_volume::{HashMap, HashSet, Key, LevelDab, LevelVolume, Shared, Stored, StrokeEnd};

const B: i32 = 8;
const N: usize = 512;
/// Lattice steps between the points the move is measured at.
const EVERY: i32 = 4;
/// The most a height may be, in level `L` steps: more is form, and belongs in the form.
const MOST: f32 = 6.0;
/// A sample whose source moved less than this (in steps) copies its height rather than resampling it.
const SNAP: f32 = 0.05;

fn cell(i: usize) -> IVec3 {
    IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32)
}

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(IVec3::splat(B));
    (l.x + 8 * l.y + 64 * l.z) as usize
}

/// The layer's samples: heights and mask, per level `L` block.
#[derive(Clone, Default)]
struct Layer {
    h: Shared<IVec3, Stored>,
    m: Shared<IVec3, Stored>,
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
        let mut sum = 0.0;
        for q in 0..8 {
            let d = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
            let w = (if d.x == 1 { f.x } else { 1.0 - f.x }) * (if d.y == 1 { f.y } else { 1.0 - f.y }) * (if d.z == 1 { f.z } else { 1.0 - f.z });
            sum += w * self.at(i + d);
        }
        sum
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

/// Where `f` changes sign along the line through `p` along `d`, toward the surface from `p`: stepping by the field's
/// value (`lo` to `hi`), then halving. Only signs are trusted, since mid-stroke the field is not a distance.
fn root(f: &dyn Fn(Vec3) -> f32, p: Vec3, d: Vec3, lo: f32, hi: f32, reach: f32) -> Option<Vec3> {
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
/// else level `below`'s (the form's finest under the layer), interpolated but not held to level `l`'s band, since
/// heights can stand out farther.
fn unclamped(form: &LevelVolume, l: usize, below: usize, lo: IVec3, hi: IVec3) -> Vec<f32> {
    let bc = lo.div_euclid(IVec3::splat(B)) + IVec3::ONE;
    if form.exists(l, bc) {
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
                let mut v = 0.0;
                for q in 0..8 {
                    let d = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                    let w = (if d.x == 1 { f.x } else { 1.0 - f.x }) * (if d.y == 1 { f.y } else { 1.0 - f.y }) * (if d.z == 1 { f.z } else { 1.0 - f.z });
                    v += w * at(i + d);
                }
                out.push(v);
            }
        }
    }
    out
}

/// The form's outward normal at `p` from the finest level up to `below` whose band reaches `p` (farther out, a fine
/// level's field is flat, and has none).
fn normal(form: &LevelVolume, p: Vec3, below: usize, _base: bool) -> Vec3 {
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

/// What a follow did.
#[derive(Clone, Copy, Debug, Default)]
pub struct FollowStats {
    pub ms: f64,
    /// Layer blocks worked out again, and the points the move was measured at.
    pub blocks: usize,
    pub points: usize,
    /// Its parts: finding where the layer's blocks went, measuring the move, the heights, and composing.
    pub parts_ms: [f64; 4],
}

enum Undo {
    /// A stroke on the form, with the layer as it was before it.
    Form(Layer),
    /// A change to the layer alone.
    Layer(Layer),
}

/// A form with one detail layer over it.
pub struct Layered {
    form: LevelVolume,
    /// The layer's level (the finest), and the first level whose work it took from the form.
    level: usize,
    from: usize,
    /// While the layer itself is edited (its mask painted): the layer as the edit began.
    edit: Option<Layer>,
    layer: Layer,
    /// While a stroke on the form is under way: the form and the layer as it began, and the blocks it changed.
    start: Option<(LevelVolume, Layer)>,
    touched: HashSet<Key>,
    history: Vec<Undo>,
    redo: Vec<Undo>,
    pub last: FollowStats,
}

impl Layered {
    /// Put the work `shown` holds at levels `from` and finer in a layer of its own over the rest, held at the finest
    /// level. `shown` becomes what is drawn: the form with the layer on it.
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
        let mut out = Layered { form, level, from, edit: None, layer: Layer::default(), start: None, touched: HashSet::default(), history: Vec::new(), redo: Vec::new(), last: FollowStats::default() };
        let keys = out.finest_under(&full, &full.keys());
        out.take_heights(shown, &full, &keys);
        shown.clear_history();
        out
    }

    /// Finest-level blocks under `keys` (of levels `from` and finer) near the surface of `vol`.
    fn finest_under(&self, vol: &LevelVolume, keys: &[Key]) -> Vec<IVec3> {
        let l = self.level;
        let (v, band, origin) = (vol.voxel(l), vol.band(l), vol.origin());
        let near = 4.0 * 3f32.sqrt() * v + band + MOST * v;
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

    /// Make the layer's heights over finest-level blocks `keys` what `detail` stands out of the form there (keeping the
    /// mask), and show them; `shown` then holds the form's blocks again at the levels the layer took.
    fn take_heights(&mut self, shown: &mut LevelVolume, detail: &LevelVolume, keys: &[IVec3]) {
        let (l, form) = (self.level, &self.form);
        let (v, band, origin) = (form.voxel(l), form.band(l), form.origin());
        let below = self.from - 1;
        let built: Vec<(IVec3, Option<Box<[f32; N]>>)> = keys
            .par_iter()
            .map(|&bc| {
                let base = |p: Vec3| form.value_upto(p, below, false);
                let top = |p: Vec3| detail.value_upto(p, l, false);
                let mut h = Box::new([0.0f32; N]);
                let mut any = false;
                for (i, out) in h.iter_mut().enumerate() {
                    let x = origin + (bc * B + cell(i)).as_vec3() * v;
                    if base(x).abs() > band + MOST * v {
                        continue;
                    }
                    let n = normal(form, x, below, false);
                    let Some(foot) = root(&base, x, n, 0.25 * v, 2.0 * v, 8.0 * band) else { continue };
                    let n = normal(form, foot, below, false);
                    let Some(y) = root(&top, foot, n, 0.1 * v, 0.5 * v, MOST * v + v) else { continue };
                    *out = (y - foot).dot(n).clamp(-MOST * v, MOST * v);
                    any |= out.abs() > 1e-4 * v;
                }
                (bc, any.then_some(h))
            })
            .collect();
        for (bc, h) in built {
            match h {
                Some(h) => {
                    self.layer.h.insert(bc, Stored::from(h));
                    if !self.layer.m.contains_key(&bc) {
                        self.layer.m.insert(bc, Stored::from(Box::new([1.0f32; N])));
                    }
                }
                None => {
                    self.layer.h.remove(&bc);
                    self.layer.m.remove(&bc);
                }
            }
        }
        // The form's blocks again, coarsest first, where `shown` held the work the layer now has.
        let mut coarse: Vec<Key> = detail.keys().into_iter().chain(shown.keys()).filter(|k| (k.0 as usize) >= self.from && (k.0 as usize) < l).collect();
        coarse.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        coarse.dedup();
        for k in coarse {
            shown.mirror(&self.form, k);
        }
        let stray: Vec<Key> = shown.keys().into_iter().filter(|k| k.0 as usize == l && !self.layer.h.contains_key(&k.1)).collect();
        for k in stray {
            shown.mirror(&self.form, k);
        }
        self.compose(shown, keys);
    }

    /// A stroke on what is shown (`shown`'s last stroke, on the layer) has ended: take it into the layer's heights,
    /// to undo with the layer's history rather than with the stroke's own.
    pub fn take_detail(&mut self, shown: &mut LevelVolume) {
        let keys = shown.last_stroke_keys();
        shown.forget_last();
        if keys.is_empty() {
            return;
        }
        let was = self.layer.clone();
        let detail = shown.snapshot();
        let blocks = self.finest_under(&detail, &keys);
        // Coarser blocks the stroke changed (making parents, moving form down) go back to the form's.
        let mut coarse: Vec<Key> = keys.iter().copied().filter(|k| (k.0 as usize) < self.from).collect();
        coarse.sort_unstable_by_key(|k| k.0);
        for k in coarse {
            shown.mirror(&self.form, k);
        }
        self.take_heights(shown, &detail, &blocks);
        self.history.push(Undo::Layer(was));
        self.redo.clear();
    }

    /// The first level the layer took work from: strokes on the layer work there or finer.
    pub fn from(&self) -> usize {
        self.from
    }

    /// The form under the layer: brushes find the surface on it.
    pub fn form(&self) -> &LevelVolume {
        &self.form
    }

    pub fn level(&self) -> usize {
        self.level
    }

    /// The tallest height the layer holds (for checks).
    pub fn tallest(&self) -> f32 {
        self.layer.h.values().flat_map(|b| b.iter()).fold(0.0f32, |a, b| a.max(b.abs()))
    }

    /// The layer's blocks.
    pub fn blocks(&self) -> usize {
        self.layer.h.len()
    }

    pub fn begin_stroke(&mut self) {
        self.start = Some((self.form.snapshot(), self.layer.clone()));
        self.touched.clear();
        self.form.begin_stroke();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plane_dab(&mut self, shown: &mut LevelVolume, l: usize, at: Vec3, along: Vec3, r: f32, h: f32, hardness: f32, alpha: f32, trim: bool) -> Option<(Vec3, LevelDab)> {
        let out = self.form.plane_dab(l, at, along, r, h, hardness, alpha, trim);
        self.follow(shown, false);
        out
    }

    pub fn ball_dab(&mut self, shown: &mut LevelVolume, l: usize, c: Vec3, r: f32, k: f32, carve: bool) -> LevelDab {
        let out = self.form.ball_dab(l, c, r, k, carve);
        self.follow(shown, false);
        out
    }

    /// End the stroke on the form, and carry the layer over the form as it is now, its distances made again.
    pub fn end_stroke(&mut self, shown: &mut LevelVolume) -> StrokeEnd {
        let out = self.form.end_stroke();
        self.follow(shown, true);
        if let Some((_, layer)) = self.start.take() {
            self.history.push(Undo::Form(layer));
            self.redo.clear();
        }
        out
    }

    /// Fade the layer toward `to` (0 hides it, 1 shows it) within `r` of `c`, softly.
    pub fn paint_mask(&mut self, shown: &mut LevelVolume, c: Vec3, r: f32, to: f32) {
        let (v, origin) = (self.form.voxel(self.level), self.form.origin());
        let lo = ((c - Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
        let hi = ((c + Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
        let was = self.layer.clone();
        let mut changed = Vec::new();
        for bz in lo.z..=hi.z {
            for by in lo.y..=hi.y {
                for bx in lo.x..=hi.x {
                    let bc = IVec3::new(bx, by, bz);
                    let Some(m) = self.layer.m.get(&bc) else { continue };
                    let mut m = Box::new(**m);
                    for (i, x) in m.iter_mut().enumerate() {
                        let d = (origin + (bc * B + cell(i)).as_vec3() * v - c).length() / r;
                        let t = (1.0 - d).clamp(0.0, 1.0);
                        let w = t * t * (3.0 - 2.0 * t);
                        *x += w * (to - *x);
                    }
                    self.layer.m.insert(bc, Stored::from(m));
                    changed.push(bc);
                }
            }
        }
        if !changed.is_empty() {
            if self.edit.is_none() {
                self.history.push(Undo::Layer(was));
                self.redo.clear();
            }
            self.compose(shown, &changed);
        }
    }

    /// Begin an edit of the layer itself (a mask stroke): undone as one.
    pub fn begin_edit(&mut self) {
        self.edit = Some(self.layer.clone());
    }

    pub fn end_edit(&mut self) {
        if let Some(was) = self.edit.take() {
            self.history.push(Undo::Layer(was));
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
        let (layer, form) = match entry {
            Undo::Form(l) => (l, true),
            Undo::Layer(l) => (l, false),
        };
        if form {
            if undo {
                self.form.undo();
            } else {
                self.form.redo();
            }
        }
        let old = std::mem::replace(&mut self.layer, layer);
        let region = self.form.drain_dirty();
        for &k in &region {
            shown.mirror(&self.form, k);
        }
        // Every layer block either version holds that differs, or whose form changed.
        let mut keys: HashSet<IVec3> = HashSet::default();
        for (bc, h) in self.layer.h.iter() {
            if old.h.get(bc).is_none_or(|o| !std::sync::Arc::ptr_eq(o, h)) || old.m.get(bc).zip(self.layer.m.get(bc)).is_none_or(|(a, b)| !std::sync::Arc::ptr_eq(a, b)) {
                keys.insert(*bc);
            }
        }
        for bc in old.h.keys() {
            if !self.layer.h.contains_key(bc) {
                keys.insert(*bc);
            }
        }
        keys.extend(self.under(&region, &self.layer));
        let keys: Vec<IVec3> = keys.into_iter().collect();
        self.compose(shown, &keys);
        let stale = self.stale_under(shown, &region);
        self.tidy(shown, stale);
        if form { Undo::Form(old) } else { Undo::Layer(old) }
    }

    /// Layer blocks (of `layer`) under any of the form's blocks `region`.
    fn under(&self, region: &[Key], layer: &Layer) -> Vec<IVec3> {
        let l = self.level;
        let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); l + 1];
        for &(m, bc) in region {
            if (m as usize) <= l {
                per[m as usize].insert(bc);
            }
        }
        layer.h.keys().filter(|bc| (0..=l).any(|m| !per[m].is_empty() && per[m].contains(&(**bc >> (l - m) as i32)))).copied().collect()
    }

    /// Carry the layer over the form as the stroke has left it. Mid-stroke, only where the last dab changed it; at the
    /// end, everywhere the stroke did, now that the form's fields are distances again.
    fn follow(&mut self, shown: &mut LevelVolume, end: bool) {
        let t = Instant::now();
        let dirty = self.form.drain_dirty();
        for &k in &dirty {
            shown.mirror(&self.form, k);
        }
        self.touched.extend(dirty.iter().copied());
        let Some((form0, layer0)) = &self.start else {
            let keys = self.under(&dirty, &self.layer);
            self.compose(shown, &keys);
            let stale = self.stale_under(shown, &dirty);
            self.tidy(shown, stale);
            return;
        };
        let region: Vec<Key> = if end { self.touched.iter().copied().collect() } else { dirty };
        let (v, band, origin) = (self.form.voxel(self.level), self.form.band(self.level), self.form.origin());
        let (form, top, below) = (&self.form, self.form.levels() - 1, self.from - 1);
        let new = |p: Vec3| form.value_upto(p, top, false);
        let old = |p: Vec3| form0.value_upto(p, top, false);
        let reach = 64.0 * form.voxel(below.saturating_sub(2));
        // The longest step a search takes: the form's fields are distances out to its finest level's band, and only
        // signs are trusted, so a step may go that far.
        let far = (4.0 * v).max(form.band(below));
        let src = self.under(&region, layer0);
        let mut dest: HashSet<IVec3> = src.iter().copied().collect();
        dest.extend(self.under(&region, &self.layer));
        // Where the surface in each source block went: blocks there may need the layer too.
        let near = 4.0 * 3f32.sqrt() * v + band + MOST * v;
        let t0 = Instant::now();
        let grown: Vec<IVec3> = src
            .par_iter()
            .flat_map_iter(|&bc| {
                (0..9)
                    .filter_map(|q| {
                        let g = if q == 8 { IVec3::splat(B / 2) } else { IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1) * B };
                        let c = origin + (bc * B + g).as_vec3() * v;
                        let n0 = normal(form0, c, below, false);
                        let y0 = root(&old, c, n0, 0.25 * v, far, reach)?;
                        let u = form.push_dir(y0, n0);
                        let y1 = root(&new, y0, u, 0.25 * v, far, reach)?;
                        Some(((c + y1 - y0 - origin) / (v * B as f32)).floor().as_ivec3())
                    })
                    .filter(|b| new(origin + (*b * B + IVec3::splat(B / 2)).as_vec3() * v).abs() < near)
                    .collect::<Vec<_>>()
            })
            .collect();
        dest.extend(grown);
        let dest: Vec<IVec3> = dest.into_iter().collect();
        // The move, measured every `EVERY` steps: from each point to the new surface, and from there back to the
        // surface the stroke began on.
        // Mid-stroke at the blocks' corners only (a preview); at the end every `EVERY` steps.
        let every = if end { EVERY } else { B };
        let per = B / every + 1;
        let points: HashSet<IVec3> = dest.iter().flat_map(|bc| (0..per * per * per).map(move |q| bc * B + IVec3::new(q % per, (q / per) % per, q / (per * per)) * every)).collect();
        let points: Vec<IVec3> = points.into_iter().collect();
        // Each point's move (in steps) and its distance from the new surface (found by the search, so exact however far
        // out the fields hold only a bound).
        let measure = |i: IVec3| -> Option<(Vec3, f32)> {
            let p = origin + i.as_vec3() * v;
            let n1 = normal(form, p, below, false);
            let o = root(&new, p, n1, 0.25 * v, far, reach)
                .and_then(|y1| {
                    let n0 = normal(form0, y1, below, false);
                    let n0 = if n0 == Vec3::ZERO { n1 } else { n0 };
                    let u = form.push_dir(y1, n0);
                    // As far out from the surface the stroke began on, along its normal there, as `p` is out of the
                    // new one: where the stroke tilted the surface, the same offset along the new normal would land
                    // beside the point it came from.
                    root(&old, y1, u, 0.25 * v, far, reach).map(|y0| {
                        let n = normal(form0, y0, below, false);
                        let n = if n == Vec3::ZERO { n1 } else { n };
                        (y0 + n * (p - y1).dot(n1) - p, (p - y1).dot(n1))
                    })
                });
            o.map(|(o, d)| (o / v, d))
        };
        let t1 = Instant::now();
        let moves: HashMap<IVec3, Option<(Vec3, f32)>> = points.par_iter().map(|&i| (i, measure(i))).collect();
        let t2 = Instant::now();
        let built: Vec<(IVec3, Option<(Stored, Stored)>)> = dest
            .par_iter()
            .map(|&bc| {
                let at = |i: IVec3| moves.get(&i).copied().flatten();
                let corner = bc * B;
                // Interpolated from the corners measured (where a search found no surface, the others say).
                // Samples farther from the surface than any height reaches need none.
                let reach_out = band + MOST * v;
                let offs: Vec<Option<Vec3>> = (0..N)
                    .map(|i| {
                        let c = cell(i);
                        let base = c.div_euclid(IVec3::splat(every)) * every;
                        let f = (c - base).as_vec3() / every as f32;
                        let (mut o, mut dist, mut sum) = (Vec3::ZERO, 0.0, 0.0);
                        for q in 0..8 {
                            let d = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                            let w = (if d.x == 1 { f.x } else { 1.0 - f.x }) * (if d.y == 1 { f.y } else { 1.0 - f.y }) * (if d.z == 1 { f.z } else { 1.0 - f.z });
                            if let Some((m, e)) = at(corner + base + d * every) {
                                o += w * m;
                                dist += w * e;
                                sum += w;
                            }
                        }
                        (sum > 1e-6 && (dist / sum).abs() < reach_out).then(|| o / sum)
                    })
                    .collect();
                // Mid-stroke a preview, read linearly straight from the blocks; at the end with the cubic, from the source
                // region read once.
                let (h0, m0) = (Sparse::new(&layer0.h, 0.0), Sparse::new(&layer0.m, 1.0));
                // (Unless the source region is large: a block the stroke stretched or tore across, read as needed.)
                let dense = end.then(|| -> Option<(Dense, Dense, bool)> {
                    let (mut lo, mut hi) = (corner, corner + IVec3::splat(B - 1));
                    for (i, o) in offs.iter().enumerate() {
                        if let Some(o) = o {
                            let g = (corner + cell(i)).as_vec3() + *o;
                            lo = lo.min(g.floor().as_ivec3());
                            hi = hi.max(g.ceil().as_ivec3());
                        }
                    }
                    let (lo, hi) = (lo - IVec3::splat(2), hi + IVec3::splat(2));
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
                    h[i] = hv.clamp(-MOST * v, MOST * v);
                    m[i] = mv;
                    any |= h[i].abs() > 1e-4 * v;
                }
                (bc, any.then(|| (Stored::from(h), Stored::from(m))))
            })
            .collect();
        let mut keys = Vec::with_capacity(built.len());
        for (bc, hm) in built {
            match hm {
                Some((h, m)) => {
                    self.layer.h.insert(bc, h);
                    self.layer.m.insert(bc, m);
                }
                None => {
                    self.layer.h.remove(&bc);
                    self.layer.m.remove(&bc);
                }
            }
            keys.push(bc);
        }
        let t3 = Instant::now();
        self.compose(shown, &keys);
        let stale = self.stale_under(shown, &region);
        self.tidy(shown, stale);
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        self.last = FollowStats { ms: 0.0, blocks: keys.len(), points: points.len(), parts_ms: [ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, Instant::now())] };
        self.last.ms = t.elapsed().as_secs_f64() * 1e3;
    }

    /// Blocks shown holds only as parents (the form has none there) under the form's blocks `region`: they read the
    /// form as it was when made, whatever the layer does there.
    fn stale_under(&self, shown: &LevelVolume, region: &[Key]) -> Vec<Key> {
        let l = self.level;
        let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); l + 1];
        for &(m, bc) in region {
            if (m as usize) <= l {
                per[m as usize].insert(bc);
            }
        }
        shown
            .keys()
            .into_iter()
            .filter(|&(m, bc)| {
                let m = m as usize;
                m >= 1 && m < l && !self.form.exists(m, bc) && (0..=m).any(|k| per[k].contains(&(bc >> (m - k) as i32)))
            })
            .collect()
    }

    /// Blocks `keys` that shown holds only as parents of finer ones (the form has none there): dropped where they hold
    /// none any more (finest first, so a parent left with none goes too), else made again from the level below
    /// (coarsest first, so each reads the one below as it is now).
    fn tidy(&self, shown: &mut LevelVolume, keys: Vec<Key>) {
        let mut keys: Vec<Key> = keys.into_iter().filter(|&(m, bc)| shown.exists(m as usize, bc)).collect::<HashSet<_>>().into_iter().collect();
        keys.sort_unstable_by_key(|k| (std::cmp::Reverse(k.0), k.1.z, k.1.y, k.1.x));
        let mut kept = Vec::with_capacity(keys.len());
        for (m, bc) in keys {
            let mu = m as usize;
            let children = mu + 1 < shown.levels() && (0..8).any(|q| shown.exists(mu + 1, bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)));
            if children {
                kept.push((m, bc));
            } else {
                shown.mirror(&self.form, (m, bc));
            }
        }
        kept.reverse();
        for (m, bc) in kept {
            shown.refill(m as usize, bc);
        }
    }

    /// Write what is shown at level `L` for layer blocks `keys`: the form pushed out by the layer, or the form alone
    /// where the layer holds nothing.
    fn compose(&mut self, shown: &mut LevelVolume, keys: &[IVec3]) {
        let (l, below) = (self.level, self.from - 1);
        let (v, band) = (self.form.voxel(l), self.form.band(l));
        let (form, layer) = (&self.form, &self.layer);
        let built: Vec<(IVec3, Option<Stored>)> = keys
            .par_iter()
            .map(|&bc| {
                if !layer.h.contains_key(&bc) {
                    return (bc, None);
                }
                let (lo, hi) = (bc * B - IVec3::ONE, bc * B + IVec3::splat(B));
                let (h, m) = (Dense::of(&layer.h, lo, hi, 0.0), Dense::of(&layer.m, lo, hi, 1.0));
                let most = h.v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
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
                    let mut sum = 0.0;
                    for q in 0..8 {
                        let d = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                        let w = (if d.x == 1 { f.x } else { 1.0 - f.x }) * (if d.y == 1 { f.y } else { 1.0 - f.y }) * (if d.z == 1 { f.z } else { 1.0 - f.z });
                        sum += w * pa(i + d);
                    }
                    sum
                };
                let n = 10;
                let c: Vec<f32> = (0..n * n * n)
                    .map(|k| {
                        let i = lo + IVec3::new(k % n, (k / n) % n, k / (n * n));
                        let s = m.at(i) * h.at(i);
                        // Away from the surface the height moves nothing that shows.
                        if s == 0.0 || pa(i).abs() > band + s.abs() + v {
                            return pa(i);
                        }
                        // The form's field where this point would be with the height taken off, along the normal: zero
                        // exactly the height out from the form's surface, however steep or shallow the field is.
                        let g = Vec3::new(pa(i + IVec3::X) - pa(i - IVec3::X), pa(i + IVec3::Y) - pa(i - IVec3::Y), pa(i + IVec3::Z) - pa(i - IVec3::Z));
                        let dir = g.normalize_or_zero();
                        tri(i.as_vec3() - dir * (s / v))
                    })
                    .collect();
                let at = |x: i32, y: i32, z: i32| c[(x + n * (y + n * z)) as usize];
                let mut out = Box::new([0.0f32; N]);
                for (i, o) in out.iter_mut().enumerate() {
                    let q = cell(i) + IVec3::ONE;
                    let (x, y, z) = (q.x, q.y, q.z);
                    let g = Vec3::new(at(x + 1, y, z) - at(x - 1, y, z), at(x, y + 1, z) - at(x, y - 1, z), at(x, y, z + 1) - at(x, y, z - 1)) / (2.0 * v);
                    // Steeper than a distance on a groove's wall: made one again, so a crossing sits where it is.
                    *o = (at(x, y, z) / g.length().max(1.0)).clamp(-band, band);
                }
                (bc, Some(Stored::from(out)))
            })
            .collect();
        for (bc, s) in built {
            match s {
                Some(s) => shown.put_stored(l, bc, s),
                None => shown.mirror(&self.form, (l as u8, bc)),
            }
        }
        let parents: Vec<Key> = keys.iter().flat_map(|&bc| (1..l).map(move |m| (m as u8, bc >> (l - m) as i32))).filter(|&(m, a)| !self.form.exists(m as usize, a)).collect();
        self.tidy(shown, parents);
    }
}

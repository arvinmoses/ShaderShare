//! Sculpt layers: each layer keeps what its strokes changed, at every level and block they changed, an opacity, and
//! a stack of masks.
//!
//! * **What is shown is the truth.** Strokes on a layer sculpt the volume itself, at full speed and at whatever level
//!   the brush picks. When one ends, the layer takes the stroke's change (the field after it less the field before,
//!   block by block, at every level it reached) into its own: `shown = base + Σ w Δ`, `w` the layer's weight (0 to 1):
//!   its opacity (up to 2, exaggerated) times its masks, 0 where it is hidden.
//! * **A layer's blend** shapes what shows of its change: as sculpted, turned over, only its raises, or only its
//!   carves. Strokes on it move what shows by what they did, as far as the blend shows it; the rest is the base's.
//! * **Masks** stack as in a paint program: each is painted (starting from a fill) or noise, blended onto what is
//!   under it (multiply, add, subtract, ...) at its own opacity, starting from white. Any change to a layer's weight
//!   moves the volume by the change times `Δ`, only where the layer holds any.
//! * **Sculpting where a layer shows partly** keeps the layer's look: the stroke's change goes in divided by the
//!   weight, so what shows is what was sculpted. Where the weight is under a quarter, a growing part of it goes to the
//!   base instead (so nothing grows without bound where the layer barely shows).
//! * A stroke on the base (no layer) changes nothing kept here: layers do not follow the surface it moves.
//!
//! Every change of the volume here happens inside one of its strokes (begun and ended by the caller), so the volume's
//! own undo steps it back; the caller keeps copies of the layers beside it (cheap: their maps share blocks).

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::level_volume::{HashMap, Key, LevelVolume, Shared, Stored};
use crate::mask::BlendMode;
use crate::noise::{Noise, NoiseKind, NoiseParams};

const B: i32 = 8;
const N: usize = 512;
/// Under this weight, sculpting a layer puts part of the change in the base.
const LOW: f32 = 0.25;

fn cell(i: usize) -> IVec3 {
    IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32)
}

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(IVec3::splat(B));
    (l.x + 8 * l.y + 64 * l.z) as usize
}

fn corner_weight(o: IVec3, f: Vec3) -> f32 {
    (if o.x == 1 { f.x } else { 1.0 - f.x }) * (if o.y == 1 { f.y } else { 1.0 - f.y }) * (if o.z == 1 { f.z } else { 1.0 - f.z })
}

/// Where the lattice points of a level are.
struct Frame {
    origin: Vec3,
    voxel0: f32,
}

impl Frame {
    fn of(vol: &LevelVolume) -> Frame {
        Frame { origin: vol.origin(), voxel0: vol.voxel(0) }
    }

    fn pos(&self, l: usize, i: IVec3) -> Vec3 {
        self.origin + i.as_vec3() * (self.voxel0 / (1u32 << l) as f32)
    }
}

/// A noise mask: noise in world space, remapped about `balance` by `contrast` (0 soft, 1 a hard edge).
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseMask {
    pub params: NoiseParams,
    pub balance: f32,
    pub contrast: f32,
    pub invert: bool,
}

impl Default for NoiseMask {
    fn default() -> Self {
        NoiseMask { params: NoiseParams { kind: NoiseKind::Fbm, scale: 6.0, octaves: 4, ..Default::default() }, balance: 0.5, contrast: 0.6, invert: false }
    }
}

impl NoiseMask {
    fn remap(&self, raw: f32) -> f32 {
        let w = (1.0 - self.contrast).clamp(0.002, 1.0);
        let t = ((raw - self.balance) / w + 0.5).clamp(0.0, 1.0);
        let v = t * t * (3.0 - 2.0 * t);
        if self.invert { 1.0 - v } else { v }
    }
}

/// What a mask is, for a mask list (and to change one: give it back changed).
#[derive(Clone, Debug, PartialEq)]
pub enum MaskKind {
    /// Painted, from `fill` (0 black, 1 white) where nothing is painted.
    Paint { fill: f32 },
    Noise(NoiseMask),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub kind: MaskKind,
}

#[derive(Clone)]
enum Source {
    /// Per level, the painted values at the samples the layer's change holds (a block with none is all `fill`).
    Paint { fill: f32, m: Vec<Shared<IVec3, Stored>> },
    /// The noise, built, and its raw values per block worked out so far.
    Noise { noise: NoiseMask, built: Arc<Noise>, cache: Arc<Mutex<HashMap<Key, Stored>>> },
}

impl Source {
    fn noise(noise: NoiseMask) -> Source {
        let built = Arc::new(Noise::new(&noise.params));
        Source::Noise { noise, built, cache: Arc::default() }
    }
}

#[derive(Clone)]
struct MaskLayer {
    name: String,
    visible: bool,
    opacity: f32,
    blend: BlendMode,
    source: Source,
}

impl MaskLayer {
    fn info(&self) -> MaskInfo {
        let kind = match &self.source {
            Source::Paint { fill, .. } => MaskKind::Paint { fill: *fill },
            Source::Noise { noise, .. } => MaskKind::Noise(noise.clone()),
        };
        MaskInfo { name: self.name.clone(), visible: self.visible, opacity: self.opacity, blend: self.blend, kind }
    }

    /// The mask's values in block `bc` of level `l` (before blending), `d` the layer's change.
    fn values(&self, d: &[Shared<IVec3, Stored>], frame: &Frame, l: usize, bc: IVec3) -> Box<[f32; N]> {
        match &self.source {
            Source::Paint { fill, m } => paint_block(m, *fill, d, l, bc),
            Source::Noise { noise, built, cache } => {
                let key = (l as u8, bc);
                let raw = cache.lock().unwrap().get(&key).cloned();
                let raw = raw.unwrap_or_else(|| {
                    let mut v = Box::new([0.0f32; N]);
                    for (i, x) in v.iter_mut().enumerate() {
                        *x = built.sample(frame.pos(l, bc * B + cell(i)));
                    }
                    let s = Stored::from(v);
                    cache.lock().unwrap().insert(key, s.clone());
                    s
                });
                let mut out = Box::new([0.0f32; N]);
                for i in 0..N {
                    out[i] = noise.remap(raw[i]);
                }
                out
            }
        }
    }
}

/// A painted mask at level `l`'s lattice point `i`: held where the change is held, else the coarser level's,
/// interpolated (`fill` at level 0).
fn paint_sample(m: &[Shared<IVec3, Stored>], fill: f32, d: &[Shared<IVec3, Stored>], l: usize, i: IVec3) -> f32 {
    let bc = i.div_euclid(IVec3::splat(B));
    if d[l].contains_key(&bc) {
        return m[l].get(&bc).map_or(fill, |b| b[local(i)]);
    }
    if l == 0 {
        return fill;
    }
    let g = i.as_vec3() * 0.5;
    let j = g.floor().as_ivec3();
    let f = g - j.as_vec3();
    if f == Vec3::ZERO {
        return paint_sample(m, fill, d, l - 1, j);
    }
    let mut out = 0.0;
    for q in 0..8 {
        let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
        let w = corner_weight(o, f);
        if w > 0.0 {
            out += w * paint_sample(m, fill, d, l - 1, j + o);
        }
    }
    out
}

fn paint_block(m: &[Shared<IVec3, Stored>], fill: f32, d: &[Shared<IVec3, Stored>], l: usize, bc: IVec3) -> Box<[f32; N]> {
    if d[l].contains_key(&bc) {
        return match m[l].get(&bc) {
            Some(b) => Box::new(**b),
            None => Box::new([fill; N]),
        };
    }
    let mut out = Box::new([0.0f32; N]);
    for (i, x) in out.iter_mut().enumerate() {
        *x = paint_sample(m, fill, d, l, bc * B + cell(i));
    }
    out
}

fn set_paint(m: &mut Shared<IVec3, Stored>, fill: f32, bc: IVec3, v: Box<[f32; N]>) {
    if v.iter().all(|&x| x == fill) {
        m.remove(&bc);
    } else {
        m.insert(bc, Stored::from(v));
    }
}

/// How a layer's change shows (the field is a distance: below 0 is inside, so raising makes it smaller).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerBlend {
    /// As sculpted.
    #[default]
    Add,
    /// Turned over: what was raised is carved, and what was carved is raised.
    Subtract,
    /// Only what the layer raises shows: the higher of the surface with it and without it ("Max").
    Raise,
    /// Only what the layer carves shows: the lower of the surface with it and without it ("Min").
    Carve,
}

impl LayerBlend {
    pub const ALL: [LayerBlend; 4] = [LayerBlend::Add, LayerBlend::Subtract, LayerBlend::Raise, LayerBlend::Carve];

    pub fn label(self) -> &'static str {
        match self {
            LayerBlend::Add => "Add",
            LayerBlend::Subtract => "Subtract",
            LayerBlend::Raise => "Max",
            LayerBlend::Carve => "Min",
        }
    }

    /// What shows of a held change `d` (unweighted).
    fn shape(self, d: f32) -> f32 {
        match self {
            LayerBlend::Add => d,
            LayerBlend::Subtract => -d,
            LayerBlend::Raise => d.min(0.0),
            LayerBlend::Carve => d.max(0.0),
        }
    }

    /// What shows, kept within what this blend can show.
    fn clip(self, s: f32) -> f32 {
        match self {
            LayerBlend::Raise => s.min(0.0),
            LayerBlend::Carve => s.max(0.0),
            _ => s,
        }
    }

    /// A change to hold that shows `s` (a value `clip` keeps).
    fn unshape(self, s: f32) -> f32 {
        if self == LayerBlend::Subtract { -s } else { s }
    }
}

/// The most a layer's opacity goes to: over 1 it shows its change exaggerated.
pub const MAX_OPACITY: f32 = 2.0;

/// How much of a stroke's change a layer at weight `w` takes, unweighted (`w` times it is what shows of it).
fn gain(w: f32) -> f32 {
    if w <= 0.0 { 0.0 } else { w / (w.max(LOW) * w.max(LOW)) }
}

/// A point of the surface a layer was sculpted on (its base), at a sample near it, and how far the layer moves that
/// surface along its normal (unweighted and unshaped, as `d`): the layer as heights. Short changes show from `d`
/// exactly; where a layer is tall (over [`TALL`] bands), its other weights and blends are built from the heights, so a
/// lift turned over is as deep as it was high, and halved is half as high.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Column {
    /// The sample, within its block.
    i: u16,
    /// Its distance from the base.
    b: f32,
    /// The base's normal there.
    n: Vec3,
    /// How far the layer moves the base, as a change of the field (as `d`: negative raises it).
    h: f32,
}

type Cols = Arc<Vec<Column>>;

/// A column taller than this many bands of its level is built from its height when the layer's weight or blend changes.
const TALL: f32 = 1.5;

impl Column {
    /// Where on the base it stands.
    fn foot(&self, frame: &Frame, l: usize, bc: IVec3) -> Vec3 {
        frame.pos(l, bc * B + cell(self.i as usize)) - self.n * self.b
    }

    fn tall(&self, band: f32) -> bool {
        self.h.abs() > TALL * band
    }

    /// The sample its weight is read at (block and index): a tall column's where its top shows as sculpted (where a
    /// mask is painted on it), any other's where it stands.
    fn weighed(&self, frame: &Frame, l: usize, bc: IVec3, band: f32) -> (IVec3, usize) {
        if !self.tall(band) {
            return (bc, self.i as usize);
        }
        let top = self.foot(frame, l, bc) - self.n * self.h;
        let g = ((top - frame.origin) / (frame.voxel0 / (1u32 << l) as f32)).round().as_ivec3();
        (g.div_euclid(IVec3::splat(B)), local(g))
    }
}

/// How far along `n` from `s` the surface of `vol` (levels up to `l`) lies: the first crossing, outward where `s` is
/// inside, inward where it is outside.
fn height(vol: &LevelVolume, l: usize, s: Vec3, n: Vec3) -> f32 {
    let st = 0.5 * vol.voxel(l);
    let reach = 4.0 * vol.band(0);
    let f = |t: f32| vol.value_upto(s + n * t, l, false);
    let mut prev = f(0.0);
    let dir = if prev < 0.0 { 1.0 } else { -1.0 };
    let mut t = 0.0;
    while t < reach {
        // A distance (or a bound on one) is a safe step.
        let next = t + (0.8 * prev.abs()).max(st);
        let v = f(dir * next);
        if (v < 0.0) != (prev < 0.0) {
            let (mut a, mut b, mut va, mut vb) = (t, next, prev, v);
            for _ in 0..5 {
                let m = 0.5 * (a + b);
                let vm = f(dir * m);
                if (vm < 0.0) == (va < 0.0) {
                    (a, va) = (m, vm);
                } else {
                    (b, vb) = (m, vm);
                }
            }
            return dir * (a + (b - a) * va / (va - vb));
        }
        (t, prev) = (next, v);
    }
    dir * reach
}

/// The gradient of a block's values at cell `c`, per step (one-sided at its faces).
fn grad(a: &[f32; N], c: IVec3) -> Vec3 {
    let inb = |p: IVec3| p.cmpge(IVec3::ZERO).all() && p.cmplt(IVec3::splat(B)).all();
    let v = |p: IVec3| a[(p.x + 8 * p.y + 64 * p.z) as usize];
    let mut g = Vec3::ZERO;
    for (ax, e) in [IVec3::X, IVec3::Y, IVec3::Z].into_iter().enumerate() {
        g[ax] = match (inb(c - e), inb(c + e)) {
            (true, true) => 0.5 * (v(c + e) - v(c - e)),
            (false, true) => v(c + e) - v(c),
            (true, false) => v(c) - v(c - e),
            _ => 0.0,
        };
    }
    g
}

/// What block `bc` of level `l` reads from the coarser levels (as where the volume holds no block there).
fn from_coarser(vol: &LevelVolume, frame: &Frame, l: usize, bc: IVec3) -> Box<[f32; N]> {
    let band = vol.band(l);
    let mut v = Box::new([0.0f32; N]);
    for (i, x) in v.iter_mut().enumerate() {
        *x = vol.value_upto(frame.pos(l, bc * B + cell(i)), l - 1, false).clamp(-band, band);
    }
    v
}

/// A column whose shown height changes, for building the surface where the layer is tall.
struct Moving {
    s: Vec3,
    n: Vec3,
    /// What the layer showed of it, and shows now (heights).
    h0: f32,
    h1: f32,
    /// How the layer shows there now: its weight, turned over (negative) for Subtract.
    k: f32,
    /// Where along it the layer's surface can lie, before and now (and a margin past).
    lo: f32,
    hi: f32,
    /// How far along it the layer's surface lay before or lies now (for what was built before and goes).
    reach: f32,
    tall: bool,
    /// Whether the layer shows it now at full weight as sculpted, or only its raises or carves, or not at all: what the
    /// changes held give exactly.
    pure: bool,
}

/// Where along a column of height `h` (raised positive, at full) the layer's surface can lie at the weights `ks`
/// (negative turned over), and its own height at full: from below the lowest to above the highest, with a margin.
fn span_of(h: f32, ks: &[f32], band: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (h.min(0.0), h.max(0.0));
    for &k in ks {
        lo = lo.min(k * h);
        hi = hi.max(k * h);
    }
    let m = 0.25 * h.abs() + 2.0 * band;
    (lo - m, hi + m)
}

/// How far to the side of a column a sample `t` along it takes its value from it (columns spread apart over a
/// rounded base).
fn side_reach(vx: f32, t: f32) -> f32 {
    1.5 * vx + 0.08 * t.abs()
}

/// The layer's columns about a level's points: those whose stretch reaches each, blended smoothly by how far to the
/// side each stands (one column alone would flip a point beside a steep wall between a tall column and a short one).
struct ColField<'a> {
    mv: &'a [Moving],
    /// Per block, the columns whose stretch reaches it.
    blocks: HashMap<IVec3, Vec<usize>>,
    vx: f32,
}

/// What the columns about a point say there: the base under it and its normal, how the layer shows there (`k`), and
/// whether it is tall there and shown exactly by the changes held (the nearest column's).
struct Along {
    s: Vec3,
    n: Vec3,
    k: f32,
    tall: bool,
    pure: bool,
}

impl Along {
    /// Where the layer at full holds what shows at `p` (drawn out by `1/k` along the normal, or mirrored).
    fn to_full(&self, p: Vec3) -> Vec3 {
        let t = (p - self.s).dot(self.n);
        p + self.n * (t / self.k - t)
    }

    /// Where what the layer at full holds at `q` shows.
    fn to_shown(&self, q: Vec3) -> Vec3 {
        let t = (q - self.s).dot(self.n);
        p_scale(q, self.n, t, self.k)
    }
}

fn p_scale(q: Vec3, n: Vec3, t: f32, k: f32) -> Vec3 {
    q + n * (t * k - t)
}

impl<'a> ColField<'a> {
    fn new(frame: &Frame, vx: f32, band: f32, mv: &'a [Moving]) -> ColField<'a> {
        let bs = vx * B as f32;
        let mut hits: Vec<(IVec3, usize)> = (0..mv.len())
            .into_par_iter()
            .flat_map_iter(|k| {
                let m = &mv[k];
                // The blocks whose middles lie near enough its stretch, out of the box about it.
                let side = band + side_reach(vx, m.lo.abs().max(m.hi.abs()));
                let r = side + 0.5 * 3f32.sqrt() * bs;
                let (e0, e1) = (m.s + m.n * m.lo - frame.origin, m.s + m.n * m.hi - frame.origin);
                let (a, b) = (((e0.min(e1) - Vec3::splat(side)) / bs).floor().as_ivec3(), ((e0.max(e1) + Vec3::splat(side)) / bs).floor().as_ivec3());
                let mut out: Vec<(IVec3, usize)> = Vec::new();
                for z in a.z..=b.z {
                    for y in a.y..=b.y {
                        for x in a.x..=b.x {
                            let bc = IVec3::new(x, y, z);
                            let o = frame.origin + (bc.as_vec3() + 0.5) * bs - m.s;
                            let t = o.dot(m.n).clamp(m.lo, m.hi);
                            if (o - m.n * t).length_squared() <= r * r {
                                out.push((bc, k));
                            }
                        }
                    }
                }
                out
            })
            .collect();
        hits.par_sort_unstable_by_key(|(bc, k)| (bc.z, bc.y, bc.x, *k));
        let mut blocks: HashMap<IVec3, Vec<usize>> = HashMap::default();
        blocks.reserve(hits.len() / 64);
        for run in hits.chunk_by(|a, b| a.0 == b.0) {
            blocks.insert(run[0].0, run.iter().map(|h| h.1).collect());
        }
        ColField { mv, blocks, vx }
    }

    /// The columns about `p` (in block `bc`), if any reaches it.
    fn at(&self, bc: IVec3, p: Vec3) -> Option<Along> {
        self.at_among(self.blocks.get(&bc)?, p)
    }

    /// Of `ks`, those that can reach within `slack` of `c`.
    fn near(&self, ks: &[usize], c: Vec3, slack: f32) -> Vec<usize> {
        ks.iter()
            .copied()
            .filter(|&k| {
                let m = &self.mv[k];
                let o = c - m.s;
                let t = o.dot(m.n).clamp(m.lo, m.hi);
                let r = side_reach(self.vx, m.lo.abs().max(m.hi.abs())) + slack;
                (o - m.n * t).length_squared() <= r * r
            })
            .collect()
    }

    /// The columns of `ks` about `p`, if any reaches it.
    fn at_among(&self, ks: &[usize], p: Vec3) -> Option<Along> {
        let (mut wsum, mut foot, mut normal, mut kw) = (0.0f32, Vec3::ZERO, Vec3::ZERO, 0.0f32);
        let mut best: Option<(f32, usize)> = None;
        let mut tall = false;
        for &k in ks {
            let m = &self.mv[k];
            let o = p - m.s;
            let t = o.dot(m.n);
            if t < m.lo || t > m.hi {
                continue;
            }
            let d2 = (o - m.n * t).length_squared();
            let r = side_reach(self.vx, t);
            if d2 > r * r {
                continue;
            }
            tall |= m.tall;
            if best.is_none_or(|b| d2 < b.0) {
                best = Some((d2, k));
            }
            let q = 1.0 - d2 / (r * r);
            let w = q * q + 1e-6;
            wsum += w;
            foot += w * m.s;
            normal += w * m.n;
            kw += w * m.k;
        }
        let (_, k) = best?;
        let kk = kw / wsum;
        let kk = if kk.abs() < 0.02 { 0.02f32.copysign(kk) } else { kk };
        Some(Along { s: foot / wsum, n: normal.normalize_or_zero(), k: kk, tall, pure: self.mv[k].pure })
    }
}

/// The surface at level `l` where columns `mv` move it, where the layer is tall: the layer at full (`full`: the volume
/// with it shown as sculpted, per block it holds) drawn out or pressed down along its base's normal by its weight, or
/// mirrored through its base where it is turned over, so walls and overhangs keep their shape. Per block reached: its
/// values, and which were built (1) or no column reaches (2); the rest are left to the changes held.
fn build_tall(read: &LevelVolume, frame: &Frame, l: usize, mv: &[Moving], full: &HashMap<IVec3, Box<[f32; N]>>, lin: Option<&Shared<IVec3, Stored>>) -> Vec<(IVec3, Box<[f32; N]>, Box<[u8; N]>)> {
    let (vx, band) = (read.voxel(l), read.band(l));
    // The layer at full at lattice point `i`: where it holds no change, the base, as far from it along the normal
    // (`s`, `n`) as the point lies.
    let at = |i: IVec3, s: Vec3, n: Vec3| -> f32 {
        let bc = i.div_euclid(IVec3::splat(B));
        if let Some(b) = full.get(&bc) {
            return b[local(i)];
        }
        if let Some(b) = lin.and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
            return b[local(i)];
        }
        (frame.pos(l, i) - s).dot(n).clamp(-band, band)
    };
    let full_at = |q: Vec3, s: Vec3, n: Vec3| -> f32 {
        let g = (q - frame.origin) / vx;
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let bc = i.div_euclid(IVec3::splat(B));
        let li = i - bc * B;
        if li.cmplt(IVec3::splat(B - 1)).all() {
            let blk: Option<&[f32; N]> = full.get(&bc).map(|b| &**b).or_else(|| lin.and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()).map(|b| &**b));
            if let Some(b) = blk {
                let mut v = 0.0;
                for c in 0..8 {
                    let o = IVec3::new(c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let w = corner_weight(o, f);
                    if w > 0.0 {
                        v += w * b[local(i + o)];
                    }
                }
                return v;
            }
        }
        let mut v = 0.0;
        for c in 0..8 {
            let o = IVec3::new(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let w = corner_weight(o, f);
            if w > 0.0 {
                v += w * at(i + o, s, n);
            }
        }
        v
    };
    let field = ColField::new(frame, vx, band, mv);
    let blocks: Vec<IVec3> = field.blocks.iter().filter(|(_, ks)| ks.iter().any(|&k| mv[k].tall)).map(|(bc, _)| *bc).collect();
    blocks
        .into_par_iter()
        .map(|bc| {
            let mut v = read.block_values(l, bc, false);
            // 1: built; 2: no column reaches it (as where a layer's top is wider than its base).
            let mut set = Box::new([0u8; N]);
            // The columns' say every other sample (it changes smoothly); between, it is interpolated where the eight
            // about agree that the layer is built there, and asked again where they do not.
            const C: i32 = B / 2 + 1;
            // The columns that can reach each two-sample cell (and its far corner), found by quarters first.
            let all = &field.blocks[&bc];
            let lat = |g: IVec3| frame.pos(l, bc * B) + g.as_vec3() * vx;
            let quarters: Vec<Vec<usize>> = (0..8).map(|q| field.near(all, lat(IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1) * 4 + IVec3::splat(2)), 2.0 * 3f32.sqrt() * vx + 0.01 * vx)).collect();
            let cells: Vec<Vec<usize>> = (0..64)
                .map(|j| {
                    let c = IVec3::new(j % 4, (j / 4) % 4, j / 16);
                    let q = c / 2;
                    field.near(&quarters[(q.x + 2 * (q.y + 2 * q.z)) as usize], lat(c * 2 + IVec3::ONE), 3f32.sqrt() * vx + 0.01 * vx)
                })
                .collect();
            let cell_of = |g: IVec3| -> &[usize] {
                let c = (g / 2).min(IVec3::splat(3));
                &cells[(c.x + 4 * (c.y + 4 * c.z)) as usize]
            };
            let coarse: Vec<Option<Along>> = (0..C * C * C)
                .map(|j| {
                    let c = IVec3::new(j % C, (j / C) % C, j / (C * C)) * 2;
                    let g = bc * B + c;
                    if c.cmplt(IVec3::splat(B)).all() {
                        field.at_among(cell_of(c), frame.pos(l, g))
                    } else {
                        field.at(g.div_euclid(IVec3::splat(B)), frame.pos(l, g))
                    }
                })
                .collect();
            let built = |a: &Option<Along>| a.as_ref().is_some_and(|a| a.tall && !a.pure);
            for i in 0..N {
                let c = cell(i);
                let p = frame.pos(l, bc * B + c);
                let lo = c / 2;
                let odd = c - lo * 2;
                let mut mixed: Option<Along> = None;
                if (0..8).all(|q| {
                    let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1) * odd;
                    let g = lo + o;
                    built(&coarse[(g.x + C * (g.y + C * g.z)) as usize])
                }) {
                    let (mut s, mut n, mut k, mut wsum) = (Vec3::ZERO, Vec3::ZERO, 0.0f32, 0.0f32);
                    for q in 0..8 {
                        let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                        if (o * (IVec3::ONE - odd)) != IVec3::ZERO {
                            continue;
                        }
                        let g = lo + o;
                        let a = coarse[(g.x + C * (g.y + C * g.z)) as usize].as_ref().unwrap();
                        s += a.s;
                        n += a.n;
                        k += a.k;
                        wsum += 1.0;
                    }
                    mixed = Some(Along { s: s / wsum, n: n.normalize_or_zero(), k: k / wsum, tall: true, pure: false });
                }
                let got = match mixed {
                    Some(a) => Some(a),
                    None => field.at_among(cell_of(c), p),
                };
                match got {
                    Some(a) if a.tall && !a.pure => {
                        let f = full_at(a.to_full(p), a.s, a.n);
                        // Held at the band stays there; nearer, distances along the normal scale by the weight.
                        let mag = if f.abs() >= 0.999 * band { f } else { a.k.abs() * f };
                        v[i] = (mag * a.k.signum()).clamp(-band, band);
                        set[i] = 1;
                    }
                    None => set[i] = 2,
                    _ => {}
                }
            }
            (bc, v, set)
        })
        .collect()
}

#[derive(Clone)]
pub struct SculptLayer {
    pub name: String,
    pub visible: bool,
    opacity: f32,
    blend: LayerBlend,
    /// Per level, the change the layer's strokes made, per block.
    d: Vec<Shared<IVec3, Stored>>,
    /// Per level, the layer as heights over its base, per block (at the samples near the base).
    cols: Vec<Shared<IVec3, Cols>>,
    /// Per level, the blocks where what shows was built from its heights.
    built: Vec<Shared<IVec3, ()>>,
    /// Bottom first.
    masks: Vec<MaskLayer>,
    active_mask: Option<usize>,
}

impl SculptLayer {
    fn new(name: &str, levels: usize) -> SculptLayer {
        SculptLayer { name: name.to_string(), visible: true, opacity: 1.0, blend: LayerBlend::Add, d: vec![Shared::default(); levels], cols: vec![Shared::default(); levels], built: vec![Shared::default(); levels], masks: Vec::new(), active_mask: None }
    }

    fn fit(&mut self, levels: usize) {
        if self.cols.len() < levels {
            self.cols.resize(levels, Shared::default());
        }
        if self.built.len() < levels {
            self.built.resize(levels, Shared::default());
        }
        if self.d.len() < levels {
            self.d.resize(levels, Shared::default());
            for mk in &mut self.masks {
                if let Source::Paint { m, .. } = &mut mk.source {
                    m.resize(levels, Shared::default());
                }
            }
        }
    }

    pub fn blocks(&self) -> usize {
        self.d.iter().map(|d| d.len()).sum()
    }

    /// The layer's weight in block `bc` of level `l`: opacity times its masks (0 hidden).
    fn weight(&self, frame: &Frame, l: usize, bc: IVec3) -> Box<[f32; N]> {
        if !self.visible || self.opacity <= 0.0 {
            return Box::new([0.0; N]);
        }
        let mut acc = Box::new([1.0f32; N]);
        for mk in self.masks.iter().filter(|mk| mk.visible && mk.opacity > 0.0) {
            let v = mk.values(&self.d, frame, l, bc);
            for i in 0..N {
                let b = mk.blend.blend(acc[i], v[i]).clamp(0.0, 1.0);
                acc[i] += mk.opacity * (b - acc[i]);
            }
        }
        if self.opacity != 1.0 {
            acc.iter_mut().for_each(|x| *x *= self.opacity);
        }
        acc
    }

    /// Block `bc` of level `l` is about to hold a change: painted masks there take what showed through them from
    /// coarser levels (else they would read their fill there).
    fn adopt_masks(&mut self, l: usize, bc: IVec3) {
        if self.d[l].contains_key(&bc) {
            return;
        }
        let d = &self.d;
        for mk in &mut self.masks {
            if let Source::Paint { fill, m } = &mut mk.source {
                let v = paint_block(m, *fill, d, l, bc);
                set_paint(&mut m[l], *fill, bc, v);
            }
        }
    }

    /// The change at level `l`'s lattice point `i`: held, or else the coarser level's, interpolated (as the volume
    /// reads a level where it holds no block).
    fn d_sample(&self, l: usize, i: IVec3) -> f32 {
        if let Some(b) = self.d[l].get(&i.div_euclid(IVec3::splat(B))) {
            return b[local(i)];
        }
        if l == 0 {
            return 0.0;
        }
        self.d_value_at(l - 1, i.as_vec3() * 0.5)
    }

    fn d_value_at(&self, l: usize, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        if f == Vec3::ZERO {
            return self.d_sample(l, i);
        }
        let mut out = 0.0;
        for q in 0..8 {
            let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
            let w = corner_weight(o, f);
            if w > 0.0 {
                out += w * self.d_sample(l, i + o);
            }
        }
        out
    }

    /// Whether any coarser level holds a change over block `bc` of level `l`.
    fn under(&self, l: usize, bc: IVec3) -> bool {
        let (lo, hi) = (bc * B, bc * B + IVec3::splat(B));
        (0..l).any(|m| {
            let s = (l - m) as i32;
            let (a, b) = ((lo >> s).div_euclid(IVec3::splat(B)), (hi >> s).div_euclid(IVec3::splat(B)));
            (a.z..=b.z).any(|z| (a.y..=b.y).any(|y| (a.x..=b.x).any(|x| self.d[m].contains_key(&IVec3::new(x, y, z)))))
        })
    }

    /// What `vol` reads in block `bc` of level `l` with this layer's part taken away (as it reads a level: held, or
    /// the coarser level's interpolated and clipped to the band). Blocks worked out go in `cache`.
    fn without(&self, vol: &LevelVolume, frame: &Frame, l: usize, bc: IVec3, cache: &mut HashMap<Key, Box<[f32; N]>>) -> Box<[f32; N]> {
        if let Some(b) = cache.get(&(l as u8, bc)) {
            return b.clone();
        }
        let mut out = Box::new([0.0f32; N]);
        if let Some(b) = vol.block((l as u8, bc)) {
            match self.d[l].get(&bc) {
                Some(d) => {
                    let w = self.weight(frame, l, bc);
                    for i in 0..N {
                        out[i] = b[i] - w[i] * self.blend.shape(d[i]);
                    }
                }
                None => out.copy_from_slice(&b[..]),
            }
        } else if l == 0 {
            for i in 0..N {
                out[i] = vol.sample(0, bc * B + cell(i), false);
            }
        } else {
            // Level l-1's points this block reads: a 5x5x5 patch from its corner.
            let base = bc * (B / 2);
            let mut patch = [0.0f32; 125];
            let mut held: Option<(IVec3, Box<[f32; N]>)> = None;
            for (n, x) in patch.iter_mut().enumerate() {
                let j = base + IVec3::new((n % 5) as i32, ((n / 5) % 5) as i32, (n / 25) as i32);
                let pb = j.div_euclid(IVec3::splat(B));
                if held.as_ref().is_none_or(|h| h.0 != pb) {
                    held = Some((pb, self.without(vol, frame, l - 1, pb, cache)));
                }
                *x = held.as_ref().unwrap().1[local(j)];
            }
            let band = vol.band(l);
            for i in 0..N {
                let c = cell(i);
                let (h, f) = (c / 2, (c % 2).as_vec3() * 0.5);
                let mut v = 0.0;
                for q in 0..8 {
                    let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
                    let w = corner_weight(o, f);
                    if w > 0.0 {
                        let t = h + o;
                        v += w * patch[(t.x + 5 * t.y + 25 * t.z) as usize];
                    }
                }
                out[i] = v.clamp(-band, band);
            }
        }
        cache.insert((l as u8, bc), out.clone());
        out
    }

    /// Block `bc` of level `l` a step made, where `before` (the volume before the step) held none and the layer
    /// holds none: what showed through it of the layer from coarser levels is the layer's there now (else it would
    /// stay in the block whatever the layer did after).
    fn inherit(&mut self, before: &LevelVolume, frame: &Frame, l: usize, bc: IVec3, cache: &mut HashMap<Key, Box<[f32; N]>>) {
        if self.d[l].contains_key(&bc) || !self.under(l, bc) {
            return;
        }
        self.adopt_masks(l, bc);
        let w = self.weight(frame, l, bc);
        let with = before.block_values(l, bc, false);
        let without = self.without(before, frame, l, bc, cache);
        let mut d = Box::new([0.0f32; N]);
        for i in 0..N {
            let shown = with[i] - without[i];
            // Stored unweighted, as every change is; where the layer barely shows, as the coarser levels hold it.
            d[i] = if w[i] > 0.01 { self.blend.unshape(self.blend.clip(shown / w[i])) } else { self.d_value_at(l - 1, (bc * B + cell(i)).as_vec3() * 0.5) };
        }
        self.d[l].insert(bc, Stored::from(d));
    }

    /// Whether the layer is tall anywhere at level `l` (then all of it there is built from the layer at full where it
    /// shows otherwise than the changes held give exactly, so no seam runs between a tall part and a short one).
    fn tall_at(&self, l: usize, band: f32) -> bool {
        self.cols.get(l).is_some_and(|m| m.values().any(|cs| cs.iter().any(|c| c.tall(band))))
    }

    /// The columns a change of the layer's weight at `keys` (or anywhere, `all`) can move: each with its level, block,
    /// and where its weight is read. Tall columns standing beside those are taken too (their surface is built with
    /// theirs).
    fn pick_cols(&self, frame: &Frame, vol: &LevelVolume, keys: &[Key], all: bool) -> Vec<(usize, IVec3, Column, IVec3, usize)> {
        let keyset: std::collections::HashSet<Key> = keys.iter().copied().collect();
        let mut out = Vec::new();
        for l in 0..self.cols.len().min(vol.levels()) {
            let band = vol.band(l);
            if !self.tall_at(l, band) {
                continue;
            }
            let entries: Vec<(IVec3, Cols)> = self.cols[l].iter().map(|(bc, c)| (*bc, c.clone())).collect();
            let near = |bc: IVec3, c: &Column| {
                let (wb, _) = c.weighed(frame, l, bc, band);
                all || keyset.contains(&(l as u8, bc)) || keyset.contains(&(l as u8, wb))
            };
            // Blocks where a tall column was picked: the tall columns of those around them come too.
            let mut around: std::collections::HashSet<IVec3> = std::collections::HashSet::new();
            if !all {
                for (bc, cols) in &entries {
                    if cols.iter().any(|c| near(*bc, c)) {
                        for q in 0..27 {
                            around.insert(*bc + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1));
                        }
                    }
                }
            }
            for (bc, cols) in &entries {
                for c in cols.iter() {
                    if near(*bc, c) || around.contains(bc) {
                        let (wb, wi) = c.weighed(frame, l, *bc, band);
                        out.push((l, *bc, *c, wb, wi));
                    }
                }
            }
        }
        out
    }

    /// Where the layer is tall and shows other than as sculpted (turned over, or at a part weight: built from the
    /// layer at full drawn out or mirrored), what a stroke did to what shows goes back into the layer at full the same
    /// way: the layer becomes what shows what is there now. Per block of the layer at full this reaches: its change,
    /// which samples were worked out so, and what the changes held give there now.
    /// Also the samples the stroke changed there (their change goes in only so).
    #[allow(clippy::type_complexity)]
    fn unwarp(&self, vol: &LevelVolume, before: &LevelVolume, lin: &[Shared<IVec3, Stored>], frame: &Frame, changes: &[(usize, IVec3, Box<[f32; N]>)]) -> (Vec<(usize, IVec3, Box<[f32; N]>, Box<[bool; N]>, Box<[f32; N]>)>, HashMap<(usize, IVec3), Box<[bool; N]>>) {
        let mut out = Vec::new();
        let mut zone: HashMap<(usize, IVec3), Box<[bool; N]>> = HashMap::default();
        for l in 0..self.cols.len().min(vol.levels()) {
            if !changes.iter().any(|c| c.0 == l) {
                continue;
            }
            let (vx, band) = (vol.voxel(l), vol.band(l));
            // The layer's tall columns as it shows now.
            let mut wcache: HashMap<IVec3, Box<[f32; N]>> = HashMap::default();
            let mut mv = Vec::new();
            if !self.tall_at(l, band) {
                continue;
            }
            for (bc, cols) in self.cols[l].iter() {
                for c in cols.iter() {
                    let (wb, wi) = c.weighed(frame, l, *bc, band);
                    let w = wcache.entry(wb).or_insert_with(|| self.weight(frame, l, wb))[wi];
                    let pure = w == 0.0 || (w == 1.0 && self.blend != LayerBlend::Subtract);
                    let k = if self.blend == LayerBlend::Subtract { -w } else { w };
                    let h = -w * self.blend.shape(c.h);
                    let (lo, hi) = span_of(-c.h, &[k], band);
                    mv.push(Moving { s: c.foot(frame, l, *bc), n: c.n, h0: h, h1: h, k, lo, hi, reach: lo.abs().max(hi.abs()), tall: true, pure });
                }
            }
            if !mv.iter().any(|m| !m.pure) {
                continue;
            }
            let field = ColField::new(frame, vx, band, &mv);
            // The blocks of the layer at full what the stroke changed comes from.
            let bs = vx * B as f32;
            let mut reach: std::collections::HashSet<IVec3> = std::collections::HashSet::new();
            for (_, bc, d) in changes.iter().filter(|c| c.0 == l) {
                for i in 0..N {
                    if d[i] == 0.0 {
                        continue;
                    }
                    let p = frame.pos(l, *bc * B + cell(i));
                    if let Some(a) = field.at(*bc, p).filter(|a| a.tall && !a.pure) {
                        zone.entry((l, *bc)).or_insert_with(|| Box::new([false; N]))[i] = true;
                        let q = a.to_full(p);
                        for c in 0..8 {
                            let o = IVec3::new(c & 1, (c >> 1) & 1, (c >> 2) & 1).as_vec3() * 2.0 - Vec3::ONE;
                            reach.insert(((q + o * vx - frame.origin) / bs).floor().as_ivec3());
                        }
                    }
                }
            }
            let reach: Vec<IVec3> = reach.into_iter().collect();
            let done: Vec<(usize, IVec3, Box<[f32; N]>, Box<[bool; N]>, Box<[f32; N]>)> = reach
                .par_iter()
                .filter_map(|&bc| {
                    let held = match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                        Some(x) => Box::new(**x),
                        None => before.block_values(l, bc, false),
                    };
                    let w = self.weight(frame, l, bc);
                    let od = self.d[l].get(&bc);
                    let mut nd = match od {
                        Some(x) => Box::new(**x),
                        None => Box::new([0.0; N]),
                    };
                    let mut lin_now = held.clone();
                    let mut done = Box::new([false; N]);
                    for i in 0..N {
                        let q = frame.pos(l, bc * B + cell(i));
                        let Some(a) = field.at(bc, q).filter(|a| a.tall && !a.pure) else { continue };
                        let v = vol.value_upto(a.to_shown(q), l, false).clamp(-band, band);
                        let full = if v.abs() >= 0.999 * band { v * a.k.signum() } else { v / a.k };
                        let base = held[i] - w[i] * self.blend.shape(od.map_or(0.0, |x| x[i]));
                        nd[i] = full.clamp(-band, band) - base;
                        lin_now[i] = base + w[i] * self.blend.shape(nd[i]);
                        done[i] = true;
                    }
                    done.iter().any(|&x| x).then_some((l, bc, nd, done, lin_now))
                })
                .collect();
            out.extend(done);
        }
        (out, zone)
    }

    /// The layer's columns a stroke moved (from `before` to `vol`), with new ones where it changed a block the base lies
    /// in and the layer had none there: per block, its columns now. `changes` are the blocks it changed, with the
    /// layer's weights there.
    fn stroke_heights(&self, vol: &LevelVolume, before: &LevelVolume, lin: &[Shared<IVec3, Stored>], frame: &Frame, changes: &[(usize, IVec3, Box<[f32; N]>)], weights: &[Box<[f32; N]>]) -> Vec<(usize, IVec3, Vec<Column>)> {
        let changed: std::collections::HashSet<Key> = changes.iter().map(|(l, bc, _)| (*l as u8, *bc)).collect();
        let mut todo: HashMap<Key, Option<usize>> = changes.iter().enumerate().map(|(k, (l, bc, _))| ((*l as u8, *bc), Some(k))).collect();
        // Tall columns standing elsewhere whose stretch the stroke reached (as far as the layer could show them).
        for l in 0..self.cols.len().min(vol.levels()) {
            let (vx, band) = (vol.voxel(l), vol.band(l));
            let bs = vx * B as f32;
            let entries: Vec<(IVec3, Cols)> = self.cols[l].iter().map(|(bc, c)| (*bc, c.clone())).collect();
            let hit: Vec<IVec3> = entries
                .par_iter()
                .filter(|(bc, cols)| {
                    cols.iter().any(|c| {
                        if c.h.abs() < 0.5 * band {
                            return false;
                        }
                        let s = c.foot(frame, l, *bc);
                        let reach = MAX_OPACITY * c.h.abs() + band;
                        let mut t = -reach;
                        while t <= reach {
                            let k = (l as u8, ((s + c.n * t - frame.origin) / bs).floor().as_ivec3());
                            if changed.contains(&k) {
                                return true;
                            }
                            t += 0.5 * bs;
                        }
                        false
                    })
                })
                .map(|(bc, _)| *bc)
                .collect();
            for bc in hit {
                todo.entry((l as u8, bc)).or_insert(None);
            }
        }
        let todo: Vec<(Key, Option<usize>)> = todo.into_iter().collect();
        let measured: Vec<(usize, IVec3, Vec<Column>, Vec<f32>, Vec<f32>, bool)> = todo
            .par_iter()
            .map(|&((l8, bc), k)| {
                let l = l8 as usize;
                let (vx, band) = (vol.voxel(l), vol.band(l));
                let mut cols: Vec<Column> = self.cols.get(l).and_then(|m| m.get(&bc)).map(|c| (**c).clone()).unwrap_or_default();
                let w = match k {
                    Some(k) => weights[k].clone(),
                    None => self.weight(frame, l, bc),
                };
                let mut any = false;
                let mut wcache: HashMap<IVec3, Box<[f32; N]>> = HashMap::default();
                let mut col_w = |c: &Column| -> f32 {
                    let (wb, wi) = c.weighed(frame, l, bc, band);
                    if wb == bc {
                        return w[wi];
                    }
                    wcache.entry(wb).or_insert_with(|| self.weight(frame, l, wb))[wi]
                };
                if k.is_some() {
                    // New columns where the base lies in this block and none stands.
                    let mut have = [false; N];
                    for c in &cols {
                        have[c.i as usize] = true;
                    }
                    // Where the layer stands at coarser levels about this block, its new columns here take their
                    // base and height from those (the field here shows the layer already, not its base).
                    let (lo, hi) = (frame.pos(l, bc * B), frame.pos(l, bc * B + IVec3::splat(B - 1)));
                    let mut coarse: Vec<(Vec3, Vec3, f32, f32)> = Vec::new();
                    for m in (0..l).rev() {
                        let vm = vol.voxel(m);
                        let bsm = vm * B as f32;
                        let a = ((lo - Vec3::splat(2.0 * vm) - frame.origin) / bsm).floor().as_ivec3();
                        let z = ((hi + Vec3::splat(2.0 * vm) - frame.origin) / bsm).floor().as_ivec3();
                        for cz in a.z..=z.z {
                            for cy in a.y..=z.y {
                                for cx in a.x..=z.x {
                                    let cb = IVec3::new(cx, cy, cz);
                                    if let Some(cs) = self.cols.get(m).and_then(|x| x.get(&cb)) {
                                        coarse.extend(cs.iter().map(|c| (c.foot(frame, m, cb), c.n, c.h, vm)));
                                    }
                                }
                            }
                        }
                        if !coarse.is_empty() {
                            break;
                        }
                    }
                    if !coarse.is_empty() {
                        for i in 0..N {
                            if have[i] {
                                continue;
                            }
                            let p = frame.pos(l, bc * B + cell(i));
                            let mut best: Option<(f32, Column)> = None;
                            for &(f, n, h, vm) in &coarse {
                                let o = p - f;
                                let t = o.dot(n);
                                let side = (o - n * t).length_squared();
                                if t.abs() < 0.5 * vx && side < vm * vm && best.is_none_or(|b| side < b.0) {
                                    best = Some((side, Column { i: i as u16, b: t, n, h }));
                                }
                            }
                            if let Some((_, c)) = best {
                                cols.push(c);
                                have[i] = true;
                                any = true;
                            }
                        }
                    }
                    // As the changes held give it where a surface was built.
                    let bv = match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                        Some(x) => Box::new(**x),
                        None => before.block_values(l, bc, false),
                    };
                    let od = self.d[l].get(&bc);
                    let mut base = [0.0f32; N];
                    for i in 0..N {
                        base[i] = bv[i] - w[i] * self.blend.shape(od.map_or(0.0, |o| o[i]));
                    }
                    for i in 0..N {
                        if have[i] || !coarse.is_empty() || base[i].abs() >= 0.5 * vx || od.is_some_and(|o| o[i].abs() >= 1.5 * band) {
                            continue;
                        }
                        let n = grad(&base, cell(i)).normalize_or_zero();
                        if n != Vec3::ZERO {
                            cols.push(Column { i: i as u16, b: base[i], n, h: 0.0 });
                            any = true;
                        }
                    }
                }
                let mut dhs = Vec::with_capacity(cols.len());
                let mut ws = Vec::with_capacity(cols.len());
                for c in &cols {
                    let s = c.foot(frame, l, bc);
                    dhs.push(height(before, l, s, c.n) - height(vol, l, s, c.n));
                    ws.push(col_w(c));
                }
                let _ = vx;
                (l, bc, cols, dhs, ws, any)
            })
            .collect();
        // Each column's change is the middle one of those around it: where the line up from the base grazes a steep
        // wall, one column's crossing can lie far from its neighbours'.
        let mut out = Vec::new();
        for l in 0..vol.levels() {
            let vx = vol.voxel(l);
            let here: Vec<&(usize, IVec3, Vec<Column>, Vec<f32>, Vec<f32>, bool)> = measured.iter().filter(|m| m.0 == l).collect();
            if here.is_empty() {
                continue;
            }
            let mut grid: HashMap<IVec3, Vec<(Vec3, f32)>> = HashMap::default();
            for m in &here {
                for (c, dh) in m.2.iter().zip(&m.3) {
                    let f = c.foot(frame, l, m.1);
                    grid.entry(((f - frame.origin) / vx).floor().as_ivec3()).or_default().push((f, *dh));
                }
            }
            let done: Vec<(usize, IVec3, Vec<Column>)> = here
                .par_iter()
                .filter_map(|m| {
                    let (bc, mut cols) = (m.1, m.2.clone());
                    let mut any = m.5;
                    for (j, c) in cols.iter_mut().enumerate() {
                        let f = c.foot(frame, l, bc);
                        let g = ((f - frame.origin) / vx).floor().as_ivec3();
                        let mut near: Vec<f32> = Vec::new();
                        for q in 0..27 {
                            if let Some(v) = grid.get(&(g + IVec3::new(q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1))) {
                                near.extend(v.iter().filter(|(p, _)| (*p - f).length_squared() <= 2.25 * vx * vx).map(|x| x.1));
                            }
                        }
                        near.sort_unstable_by(|a, b| a.total_cmp(b));
                        let dh = if near.is_empty() { m.3[j] } else { near[near.len() / 2] };
                        if dh.abs() > 1e-3 * vx {
                            let w = m.4[j];
                            let was = self.blend.shape(c.h);
                            let to = self.blend.clip(was + dh * gain(w));
                            if to != was {
                                c.h = self.blend.unshape(to);
                                any = true;
                            }
                        }
                    }
                    any.then_some((l, bc, cols))
                })
                .collect();
            out.extend(done);
        }
        out
    }

    /// Every block the layer holds, coarsest level first.
    fn keys(&self) -> Vec<Key> {
        let mut out: Vec<Key> = self.d.iter().enumerate().flat_map(|(l, d)| d.keys().map(move |bc| (l as u8, *bc))).collect();
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out
    }

    fn new_mask(&self, name: String, kind: MaskKind) -> MaskLayer {
        // The first mask sets the layer's look; ones above it cut into it.
        let blend = if self.masks.is_empty() { BlendMode::Normal } else { BlendMode::Multiply };
        let source = match kind {
            MaskKind::Paint { fill } => Source::Paint { fill, m: vec![Shared::default(); self.d.len()] },
            MaskKind::Noise(n) => Source::noise(n),
        };
        MaskLayer { name, visible: true, opacity: 1.0, blend, source }
    }
}

/// What a layer is, for a layer list.
#[derive(Clone, Debug, PartialEq)]
pub struct SculptInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: LayerBlend,
    pub blocks: usize,
    /// Bottom first.
    pub masks: Vec<MaskInfo>,
    /// The mask strokes paint (or that is picked), if any: else strokes sculpt the layer.
    pub active_mask: Option<usize>,
}

#[derive(Clone, Default)]
pub struct SculptLayers {
    stack: Vec<SculptLayer>,
    active: usize,
    /// Per level, where a surface was built from heights, what the changes held give there instead (the volume as
    /// sculpted, weighted): kept so that showing a layer as sculpted again, or hiding it, is exact.
    lin: Vec<Shared<IVec3, Stored>>,
}

impl SculptLayers {
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn layers(&self) -> Vec<SculptInfo> {
        self.stack
            .iter()
            .map(|x| SculptInfo {
                name: x.name.clone(),
                visible: x.visible,
                opacity: x.opacity,
                blend: x.blend,
                blocks: x.blocks(),
                masks: x.masks.iter().map(|m| m.info()).collect(),
                active_mask: x.active_mask,
            })
            .collect()
    }

    pub fn active(&self) -> usize {
        self.active
    }

    pub fn set_active(&mut self, k: usize) {
        if k < self.stack.len() {
            self.active = k;
        }
    }

    /// Whether layer `k` is taller than the band anywhere (its other weights and blends are built, which costs more).
    pub fn tall(&self, k: usize, vol: &LevelVolume) -> bool {
        self.stack.get(k).is_some_and(|x| (0..vol.levels()).any(|l| x.tall_at(l, vol.band(l))))
    }

    pub fn visible(&self, k: usize) -> bool {
        self.stack.get(k).is_some_and(|x| x.visible)
    }

    pub fn rename(&mut self, k: usize, name: &str) {
        if let Some(x) = self.stack.get_mut(k) {
            x.name = name.to_string();
        }
    }

    /// The mask of layer `k` strokes paint, if any (`None`: strokes sculpt the layer).
    pub fn active_mask(&self, k: usize) -> Option<usize> {
        self.stack.get(k).and_then(|x| x.active_mask)
    }

    /// Pick mask `mi` of layer `k` (or the layer itself) as what strokes work on.
    pub fn set_active_mask(&mut self, k: usize, mi: Option<usize>) {
        if let Some(x) = self.stack.get_mut(k) {
            x.active_mask = mi.filter(|&i| i < x.masks.len());
        }
    }

    /// Whether layer `k`'s picked mask is painted (strokes paint it).
    pub fn paints(&self, k: usize) -> bool {
        self.stack.get(k).is_some_and(|x| x.active_mask.is_some_and(|i| matches!(x.masks[i].source, Source::Paint { .. })))
    }

    /// A new empty layer on top, made active.
    pub fn add(&mut self, name: &str, levels: usize) -> usize {
        self.stack.push(SculptLayer::new(name, levels));
        self.active = self.stack.len() - 1;
        self.active
    }

    /// Change layer `k` by `edit`, and the volume by the change of its weight (inside a stroke of `vol`): at `keys`,
    /// or every block it holds.
    fn reweigh(&mut self, vol: &mut LevelVolume, k: usize, keys: Option<Vec<Key>>, edit: impl FnOnce(&mut SculptLayer)) {
        let frame = Frame::of(vol);
        let Some(layer) = self.stack.get_mut(k) else { return };
        layer.fit(vol.levels());
        let all = keys.is_none();
        let mut keys = keys.unwrap_or_else(|| layer.keys());
        // A block the volume does not hold follows the coarser levels, moved already.
        keys.retain(|&(l, bc)| l == 0 || vol.exists(l as usize, bc));
        let picked = layer.pick_cols(&frame, vol, &keys, all);
        let wkeys: Vec<Key> = {
            let mut v: Vec<Key> = picked.iter().map(|p| (p.0 as u8, p.3)).collect();
            v.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
            v.dedup();
            v
        };
        let weigh = |layer: &SculptLayer| -> HashMap<Key, Box<[f32; N]>> { wkeys.par_iter().map(|&(l, bc)| ((l, bc), layer.weight(&frame, l as usize, bc))).collect() };
        let cw0 = weigh(layer);
        let w0: Vec<Box<[f32; N]>> = keys.par_iter().map(|&(l, bc)| layer.weight(&frame, l as usize, bc)).collect();
        let was = layer.blend;
        edit(layer);
        let (layer, read) = (&*layer, &*vol);
        let cw1 = weigh(layer);
        let w1: Vec<Box<[f32; N]>> = keys.par_iter().map(|&(l, bc)| layer.weight(&frame, l as usize, bc)).collect();
        let lin = &self.lin;
        let moved: Vec<(usize, IVec3, Box<[f32; N]>)> = keys
            .par_iter()
            .zip(w0.par_iter())
            .zip(w1.par_iter())
            .filter_map(|((&(l8, bc), a), b)| {
                let l = l8 as usize;
                let d = layer.d[l].get(&bc)?;
                if b == a && layer.blend == was {
                    return None;
                }
                let mut v = match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                    Some(x) => Box::new(**x),
                    None => read.block_values(l, bc, false),
                };
                for i in 0..N {
                    v[i] += b[i] * layer.blend.shape(d[i]) - a[i] * was.shape(d[i]);
                }
                Some((l, bc, v))
            })
            .collect();
        // Where the layer is tall, its surface is built from its heights instead.
        let mut tall: Vec<(usize, IVec3, Box<[f32; N]>, Box<[u8; N]>)> = Vec::new();
        let any_impure = std::cell::Cell::new(false);
        let mut restore: Vec<(usize, IVec3)> = Vec::new();
        for l in 0..layer.cols.len().min(read.levels()) {
            let band = read.band(l);
            let mv: Vec<Moving> = picked
                .iter()
                .filter(|p| p.0 == l)
                .filter_map(|&(_, bc, c, wb, wi)| {
                    let key = (l as u8, wb);
                    let (a, b) = (cw0[&key][wi], cw1[&key][wi]);
                    let (h0, h1) = (-a * was.shape(c.h), -b * layer.blend.shape(c.h));
                    let tall = true;
                    let k = if layer.blend == LayerBlend::Subtract { -b } else { b };
                    let k0 = if was == LayerBlend::Subtract { -a } else { a };
                    // Where it shows now only (what is built must not hang on what showed before), and as far as it
                    // showed before too, for what goes.
                    let (lo, hi) = span_of(-c.h, &[k], band);
                    let (wlo, whi) = span_of(-c.h, &[k0, k], band);
                    (tall || h0 != h1).then(|| Moving { s: c.foot(&frame, l, bc), n: c.n, h0, h1, k, lo, hi, reach: wlo.abs().max(whi.abs()), tall, pure: b == 0.0 || (b == 1.0 && layer.blend != LayerBlend::Subtract) })
                })
                .collect();
            // What was built about the columns goes back to what the changes give, unless built again.
            if !all && !mv.is_empty() {
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for m in &mv {
                    lo = lo.min(m.s - Vec3::splat(m.reach + band));
                    hi = hi.max(m.s + Vec3::splat(m.reach + band));
                }
                let bsz = read.voxel(l) * B as f32;
                restore.extend(layer.built[l].keys().filter(|bc| {
                    let c = frame.pos(l, **bc * B);
                    c.cmpge(lo - Vec3::splat(bsz)).all() && c.cmple(hi).all()
                }).map(|bc| (l, *bc)));
            }
            // Shown everywhere as the changes held give exactly: nothing to build, only what was built goes.
            if mv.iter().all(|m| m.pure) {
                continue;
            }
            if mv.iter().any(|m| m.tall && m.h0 != m.h1) {
                any_impure.set(any_impure.get() || mv.iter().any(|m| m.tall && !m.pure));
                // The volume with the layer shown at full in its blend, where it holds a change.
                let at_full = if layer.blend == LayerBlend::Subtract { LayerBlend::Add } else { layer.blend };
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for m in &mv {
                    lo = lo.min(m.s - Vec3::splat(m.reach + band));
                    hi = hi.max(m.s + Vec3::splat(m.reach + band));
                }
                let bsz = read.voxel(l) * B as f32;
                let index: HashMap<IVec3, usize> = keys.iter().enumerate().filter(|(_, k)| k.0 as usize == l).map(|(j, k)| (k.1, j)).collect();
                let near: Vec<IVec3> = layer.d[l]
                    .keys()
                    .copied()
                    .filter(|bc| {
                        let c = frame.pos(l, *bc * B);
                        c.cmpge(lo - Vec3::splat(bsz)).all() && c.cmple(hi).all()
                    })
                    .collect();
                let full: HashMap<IVec3, Box<[f32; N]>> = near
                    .par_iter()
                    .filter_map(|&bc| {
                        let d = layer.d[l].get(&bc)?;
                        // A block the volume does not hold of itself (none, or made for a surface built) shows none
                        // of the layer: the layer at full is read about it as where it holds none.
                        if (l > 0 && !read.exists(l, bc)) || lin.get(l).and_then(|m| m.get(&bc)).is_some_and(|x| x[0].is_nan()) {
                            return None;
                        }
                        // Its weight before the change (where the change did not reach, as it is).
                        let a = match index.get(&bc) {
                            Some(&j) => w0[j].clone(),
                            None => layer.weight(&frame, l, bc),
                        };
                        let mut v = match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                            Some(x) => Box::new(**x),
                            None => read.block_values(l, bc, false),
                        };
                        for i in 0..N {
                            v[i] += at_full.shape(d[i]) - a[i] * was.shape(d[i]);
                        }
                        Some((bc, v))
                    })
                    .collect();
                tall.extend(build_tall(read, &frame, l, &mv, &full, lin.get(l)).into_iter().map(|(bc, v, set)| (l, bc, v, set)));
            }
        }
        // Per block: what the changes held give, and what shows (built from heights where it was).
        let mut out: HashMap<(usize, IVec3), (Box<[f32; N]>, Option<Box<[f32; N]>>)> = moved.into_iter().map(|(l, bc, v)| ((l, bc), (v, None))).collect();
        let impure = any_impure.get();
        let merged: Vec<((usize, IVec3), Box<[f32; N]>, Option<Box<[f32; N]>>)> = tall
            .into_par_iter()
            .map(|(l, bc, v, set)| {
                let held = match out.get(&(l, bc)) {
                    Some(e) => e.0.clone(),
                    None => match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                        Some(x) => Box::new(**x),
                        None => read.block_values(l, bc, false),
                    },
                };
                // Where no column reaches but the layer holds a tall change, the layer cannot be built from heights
                // (it overhangs its base): what shows there is the volume without it.
                let held_d = layer.d[l].get(&bc);
                let band = read.band(l);
                let drop = |i: usize| set[i] == 2 && impure && held_d.is_some_and(|d| d[i].abs() >= TALL * band);
                if !(0..N).any(|i| set[i] == 1 || drop(i)) {
                    // Reached, but shown as the changes give it: what was built there goes.
                    return ((l, bc), held, None);
                }
                let wd = held_d.map(|_| layer.weight(&frame, l, bc));
                // A block made for a surface built takes the rest from the coarser levels as they are once built
                // (marked here, filled when written).
                let made = (l > 0 && !read.exists(l, bc)) || lin.get(l).and_then(|m| m.get(&bc)).is_some_and(|x| x[0].is_nan());
                let mut shown = if made { Box::new([f32::NAN; N]) } else { held.clone() };
                for i in 0..N {
                    if set[i] == 1 {
                        shown[i] = v[i];
                    } else if drop(i) {
                        let (d, w) = (held_d.unwrap(), wd.as_ref().unwrap());
                        shown[i] = held[i] - w[i] * layer.blend.shape(d[i]);
                    }
                }
                ((l, bc), held, Some(shown))
            })
            .collect();
        for (key, held, shown) in merged {
            out.insert(key, (held, shown));
        }
        // A change of the whole layer gives back what the changes give wherever it built before and does not now.
        // What was built and is not now (anywhere for a change of the whole layer) shows as the changes give it.
        let mut again = restore;
        if all {
            again.extend(layer.built.iter().enumerate().flat_map(|(l, m)| m.keys().map(move |bc| (l, *bc))));
        }
        for (l, bc) in again {
            out.entry((l, bc)).or_insert_with(|| {
                let held = match lin.get(l).and_then(|m| m.get(&bc)).filter(|x| !x[0].is_nan()) {
                    Some(x) => Box::new(**x),
                    None => read.block_values(l, bc, false),
                };
                (held, None)
            });
        }
        let mut out: Vec<((usize, IVec3), (Box<[f32; N]>, Option<Box<[f32; N]>>))> = out.into_iter().collect();
        out.sort_unstable_by_key(|((l, bc), _)| (*l, bc.z, bc.y, bc.x));
        if self.lin.len() < vol.levels() {
            self.lin.resize(vol.levels(), Shared::default());
        }
        // Coarsest first, so a block made here (or made back) reads the coarser levels as they are now.
        let mut unmake = Vec::new();
        let mut fill: Vec<(usize, IVec3, Box<[f32; N]>)> = Vec::new();
        let built = &mut self.stack[k].built;
        for ((l, bc), (held, shown)) in out {
            if shown.is_some() {
                built[l].insert(bc, ());
            } else {
                built[l].remove(&bc);
            }
            let exists = l == 0 || vol.exists(l, bc);
            let made = self.lin[l].get(&bc).is_some_and(|x| x[0].is_nan());
            match shown {
                Some(v) => {
                    // A block made for the surface built holds, as what the changes give, only that it was made.
                    let held = if !exists || made { Box::new([f32::NAN; N]) } else { held };
                    if !exists {
                        // Its parents the volume makes with it are made here too.
                        let mut pb = bc;
                        for m in (1..l).rev() {
                            pb = pb.div_euclid(IVec3::splat(2));
                            if vol.exists(m, pb) {
                                break;
                            }
                            self.lin[m].insert(pb, Stored::from(Box::new([f32::NAN; N])));
                            built[m].insert(pb, ());
                        }
                        vol.ensure(l, bc);
                    }
                    let mut v = v;
                    if v.iter().any(|x| x.is_nan()) {
                        // (Filled at the end.)
                        fill.push((l, bc, v.clone()));
                        let band = vol.band(l);
                        v.iter_mut().filter(|x| x.is_nan()).for_each(|x| *x = band);
                    }
                    self.lin[l].insert(bc, Stored::from(held));
                    vol.set_block(l, bc, v);
                }
                None if made => {
                    self.lin[l].remove(&bc);
                    unmake.push((l, bc));
                }
                None if exists => {
                    self.lin[l].remove(&bc);
                    vol.set_block(l, bc, held);
                }
                None => {}
            }
        }
        // Blocks made for a surface built go again, finest first (a block with a finer one in it stays, read from the
        // coarser levels).
        unmake.sort_unstable_by_key(|(l, bc)| (std::cmp::Reverse(*l), bc.z, bc.y, bc.x));
        for (l, bc) in unmake {
            vol.remove_block(l, bc);
            if vol.exists(l, bc) {
                let v = from_coarser(vol, &frame, l, bc);
                vol.set_block(l, bc, v);
            }
        }
        // Made blocks take what was not built from the coarser levels once those are all as they now show (a level at a
        // time, coarsest first).
        let mut fill = fill;
        while !fill.is_empty() {
            let l = fill.iter().map(|f| f.0).min().unwrap();
            let (now, rest): (Vec<_>, Vec<_>) = fill.into_iter().partition(|f| f.0 == l);
            fill = rest;
            let read = &*vol;
            let done: Vec<(IVec3, Box<[f32; N]>)> = now
                .into_par_iter()
                .map(|(_, bc, mut v)| {
                    let c = from_coarser(read, &frame, l, bc);
                    for i in 0..N {
                        if v[i].is_nan() {
                            v[i] = c[i];
                        }
                    }
                    (bc, v)
                })
                .collect();
            for (bc, v) in done {
                vol.set_block(l, bc, v);
            }
        }
    }

    /// Take `vol`'s last step (a stroke just ended, on layer `on` or on the base) into the layers.
    pub fn take_stroke(&mut self, vol: &LevelVolume, on: Option<usize>) {
        if self.stack.is_empty() {
            return;
        }
        let Some(before) = vol.before_last() else { return };
        let frame = Frame::of(vol);
        let keys = vol.last_stroke_keys();
        // Blocks it made, coarsest first: each layer's change shows through them as it did.
        let mut made: Vec<Key> = keys.iter().copied().filter(|&(l, bc)| l > 0 && vol.exists(l as usize, bc) && !before.exists(l as usize, bc)).collect();
        made.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        for layer in &mut self.stack {
            layer.fit(vol.levels());
            let mut cache = HashMap::default();
            for &(l, bc) in &made {
                layer.inherit(&before, &frame, l as usize, bc, &mut cache);
            }
        }
        // What the stroke did moves what the changes held give too, where that is kept.
        let lin_moved: Vec<(usize, IVec3, Box<[f32; N]>)> = keys
            .par_iter()
            .filter_map(|&(l8, bc)| {
                let l = l8 as usize;
                let held = self.lin.get(l)?.get(&bc).filter(|x| !x[0].is_nan())?;
                let (a, b) = (vol.block_values(l, bc, false), before.block_values(l, bc, false));
                let mut v = Box::new(**held);
                for i in 0..N {
                    v[i] += a[i] - b[i];
                }
                Some((l, bc, v))
            })
            .collect();
        let Some(layer) = on.and_then(|k| self.stack.get_mut(k)) else {
            for (l, bc, v) in lin_moved {
                self.lin[l].insert(bc, Stored::from(v));
            }
            return;
        };
        let lin_old = &self.lin;
        let lin_old_copy: HashMap<(usize, IVec3), Stored> = lin_moved.iter().filter_map(|(l, bc, _)| Some(((*l, *bc), lin_old.get(*l)?.get(bc)?.clone()))).collect();
        let changes: Vec<(usize, IVec3, Box<[f32; N]>)> = keys
            .par_iter()
            .filter_map(|&(l8, bc)| {
                let l = l8 as usize;
                let (a, b) = (vol.block_values(l, bc, false), before.block_values(l, bc, false));
                let mut d = Box::new([0.0f32; N]);
                let mut any = false;
                for i in 0..N {
                    d[i] = a[i] - b[i];
                    any |= d[i] != 0.0;
                }
                any.then_some((l, bc, d))
            })
            .collect();
        for &(l, bc, _) in &changes {
            layer.adopt_masks(l, bc);
        }
        let layer_ref = &*layer;
        let weights: Vec<Box<[f32; N]>> = changes.par_iter().map(|(l, bc, _)| layer_ref.weight(&frame, *l, *bc)).collect();
        let (unwarped, zone) = layer_ref.unwarp(vol, &before, lin_old, &frame, &changes);
        let heights = layer_ref.stroke_heights(vol, &before, lin_old, &frame, &changes, &weights);
        for (l, bc, cols) in heights {
            if cols.is_empty() {
                layer.cols[l].remove(&bc);
            } else {
                layer.cols[l].insert(bc, Arc::new(cols));
            }
        }
        let mut lin_set: HashMap<(usize, IVec3), (Box<[f32; N]>, Box<[bool; N]>)> = HashMap::default();
        let mut unwarped_d: HashMap<(usize, IVec3), (Box<[f32; N]>, Box<[bool; N]>)> = HashMap::default();
        for (l, bc, nd, done, held) in unwarped {
            lin_set.insert((l, bc), (held, done.clone()));
            unwarped_d.insert((l, bc), (nd, done));
        }
        for ((l, bc, d), w) in changes.into_iter().zip(weights) {
            let mut nd = match layer.d[l].get(&bc) {
                Some(od) => Box::new(**od),
                None => Box::new([0.0; N]),
            };
            let blend = layer.blend;
            let un = unwarped_d.remove(&(l, bc));
            let zn = zone.get(&(l, bc));
            for i in 0..N {
                if let Some((ud, done)) = &un
                    && done[i]
                {
                    nd[i] = ud[i];
                    continue;
                }
                if zn.is_some_and(|z| z[i]) {
                    continue;
                }
                if d[i] != 0.0 {
                    // What shows of it moves by the stroke's share, as far as the blend shows it (the rest is the
                    // base's); where that is nothing, the held change stays.
                    let was = blend.shape(nd[i]);
                    let to = blend.clip(was + d[i] * gain(w[i]));
                    if to != was {
                        nd[i] = blend.unshape(to);
                    }
                }
            }
            layer.d[l].insert(bc, Stored::from(nd));
        }
        // Blocks of the layer at full the stroke reached only through the mirroring or drawing out.
        for ((l, bc), (ud, done)) in unwarped_d {
            let mut nd = match layer.d[l].get(&bc) {
                Some(od) => Box::new(**od),
                None => Box::new([0.0; N]),
            };
            for i in 0..N {
                if done[i] {
                    nd[i] = ud[i];
                }
            }
            layer.d[l].insert(bc, Stored::from(nd));
        }
        for (l, bc, mut v) in lin_moved {
            // In the zone, the stroke's change went into the layer at full only (where it reached): what the changes
            // held give there is as it was.
            if let (Some(z), Some(old)) = (zone.get(&(l, bc)), lin_old_copy.get(&(l, bc))) {
                for i in 0..N {
                    if z[i] {
                        v[i] = old[i];
                    }
                }
            }
            if let Some((held, done)) = lin_set.remove(&(l, bc)) {
                for i in 0..N {
                    if done[i] {
                        v[i] = held[i];
                    }
                }
            }
            self.lin[l].insert(bc, Stored::from(v));
        }
        for ((l, bc), (held, done)) in lin_set {
            if let Some(old) = self.lin[l].get(&bc).filter(|x| !x[0].is_nan()) {
                let mut v = Box::new(**old);
                for i in 0..N {
                    if done[i] {
                        v[i] = held[i];
                    }
                }
                self.lin[l].insert(bc, Stored::from(v));
            }
        }
    }

    /// Show or hide layer `k` (inside a stroke of `vol`).
    pub fn set_visible(&mut self, vol: &mut LevelVolume, k: usize, on: bool) {
        if self.stack.get(k).is_none_or(|x| x.visible == on) {
            return;
        }
        self.reweigh(vol, k, None, |x| x.visible = on);
    }

    /// Set how layer `k`'s change shows (inside a stroke of `vol`).
    pub fn set_blend(&mut self, vol: &mut LevelVolume, k: usize, blend: LayerBlend) {
        if self.stack.get(k).is_none_or(|x| x.blend == blend) {
            return;
        }
        self.reweigh(vol, k, None, |x| x.blend = blend);
    }

    /// Set layer `k`'s opacity (inside a stroke of `vol`).
    pub fn set_opacity(&mut self, vol: &mut LevelVolume, k: usize, opacity: f32) {
        let opacity = opacity.clamp(0.0, MAX_OPACITY);
        if self.stack.get(k).is_none_or(|x| x.opacity == opacity) {
            return;
        }
        self.reweigh(vol, k, None, |x| x.opacity = opacity);
    }

    /// Add a mask on top of layer `k`'s masks, picked (inside a stroke of `vol`).
    pub fn add_mask(&mut self, vol: &mut LevelVolume, k: usize, kind: MaskKind) -> Option<usize> {
        let x = self.stack.get(k)?;
        let n = x.masks.iter().filter(|m| matches!((&m.source, &kind), (Source::Paint { .. }, MaskKind::Paint { .. }) | (Source::Noise { .. }, MaskKind::Noise(_)))).count();
        let name = match kind {
            MaskKind::Paint { .. } => format!("Paint {}", n + 1),
            MaskKind::Noise(_) => format!("Noise {}", n + 1),
        };
        let mk = x.new_mask(name, kind);
        self.reweigh(vol, k, None, |x| {
            x.masks.push(mk);
            x.active_mask = Some(x.masks.len() - 1);
        });
        self.stack[k].active_mask
    }

    /// Remove mask `mi` of layer `k` (inside a stroke of `vol`).
    pub fn remove_mask(&mut self, vol: &mut LevelVolume, k: usize, mi: usize) {
        if self.stack.get(k).is_none_or(|x| mi >= x.masks.len()) {
            return;
        }
        self.reweigh(vol, k, None, |x| {
            x.masks.remove(mi);
            x.active_mask = match x.active_mask {
                Some(a) if a == mi => None,
                Some(a) if a > mi => Some(a - 1),
                a => a,
            };
        });
    }

    /// Move mask `mi` of layer `k` up (`up`, toward the top) or down one (inside a stroke of `vol`).
    pub fn move_mask(&mut self, vol: &mut LevelVolume, k: usize, mi: usize, up: bool) {
        let Some(x) = self.stack.get(k) else { return };
        let to = if up { mi + 1 } else { mi.wrapping_sub(1) };
        if mi >= x.masks.len() || to >= x.masks.len() {
            return;
        }
        self.reweigh(vol, k, None, |x| {
            x.masks.swap(mi, to);
            if x.active_mask == Some(mi) {
                x.active_mask = Some(to);
            } else if x.active_mask == Some(to) {
                x.active_mask = Some(mi);
            }
        });
    }

    /// Change mask `mi` of layer `k` to `info` (inside a stroke of `vol`). A painted mask given a new fill is filled:
    /// what was painted on it goes.
    pub fn set_mask(&mut self, vol: &mut LevelVolume, k: usize, mi: usize, info: MaskInfo) {
        let Some(old) = self.stack.get(k).and_then(|x| x.masks.get(mi)).map(|m| m.info()) else { return };
        if old == info {
            return;
        }
        let looks = MaskInfo { name: old.name.clone(), ..info.clone() } != old;
        let apply = move |x: &mut SculptLayer| {
            let mk = &mut x.masks[mi];
            mk.name = info.name;
            mk.visible = info.visible;
            mk.opacity = info.opacity.clamp(0.0, 1.0);
            mk.blend = info.blend;
            match (&mut mk.source, info.kind) {
                (Source::Paint { fill, m }, MaskKind::Paint { fill: to }) => {
                    if *fill != to {
                        *fill = to.clamp(0.0, 1.0);
                        m.iter_mut().for_each(|m| *m = Shared::default());
                    }
                }
                (Source::Noise { noise, .. }, MaskKind::Noise(n)) if *noise != n => {
                    if noise.params == n.params {
                        *noise = n;
                    } else {
                        mk.source = Source::noise(n);
                    }
                }
                _ => {}
            }
        };
        if looks {
            self.reweigh(vol, k, None, apply);
        } else {
            apply(&mut self.stack[k]);
        }
    }

    /// The painted mask strokes on layer `k` paint: the picked one (or with none picked, the top painted one), or a
    /// new white one if the layer has no masks.
    fn paint_target(&mut self, k: usize) -> Option<usize> {
        let x = self.stack.get_mut(k)?;
        if x.masks.is_empty() {
            // White under no masks: the layer looks just as it did.
            let mk = x.new_mask("Paint 1".into(), MaskKind::Paint { fill: 1.0 });
            x.masks.push(mk);
            x.active_mask = Some(0);
        }
        if x.active_mask.is_none() {
            x.active_mask = x.masks.iter().rposition(|m| matches!(m.source, Source::Paint { .. }));
        }
        x.active_mask.filter(|&i| matches!(x.masks[i].source, Source::Paint { .. }))
    }

    /// Fill layer `k`'s picked painted mask with `to` (inside a stroke of `vol`).
    pub fn fill_mask(&mut self, vol: &mut LevelVolume, k: usize, to: f32) {
        let Some(mi) = self.paint_target(k) else { return };
        let mut info = self.stack[k].masks[mi].info();
        info.kind = MaskKind::Paint { fill: to };
        if info == self.stack[k].masks[mi].info() {
            // The same fill: what was painted on it goes still.
            self.reweigh(vol, k, None, |x| {
                if let Source::Paint { m, .. } = &mut x.masks[mi].source {
                    m.iter_mut().for_each(|m| *m = Shared::default());
                }
            });
        } else {
            self.set_mask(vol, k, mi, info);
        }
    }

    /// Paint layer `k`'s picked painted mask toward `to` within `r` of `c`, smoothly to nothing at `r` (inside a
    /// stroke of `vol`).
    pub fn paint_mask(&mut self, vol: &mut LevelVolume, k: usize, c: Vec3, r: f32, to: f32) {
        let Some(mi) = self.paint_target(k) else { return };
        let layer = &mut self.stack[k];
        layer.fit(vol.levels());
        let frame = Frame::of(vol);
        let mut keys = Vec::new();
        for l in 0..layer.d.len().min(vol.levels()) {
            if layer.d[l].is_empty() {
                continue;
            }
            let v = vol.voxel(l);
            let lo = ((c - Vec3::splat(r) - frame.origin) / (v * B as f32)).floor().as_ivec3();
            let hi = ((c + Vec3::splat(r) - frame.origin) / (v * B as f32)).floor().as_ivec3();
            let span = (hi - lo + IVec3::ONE).as_i64vec3();
            if span.x * span.y * span.z > layer.d[l].len() as i64 {
                keys.extend(layer.d[l].keys().copied().filter(|bc| bc.cmpge(lo).all() && bc.cmple(hi).all()).map(|bc| (l as u8, bc)));
            } else {
                for z in lo.z..=hi.z {
                    for y in lo.y..=hi.y {
                        for x in lo.x..=hi.x {
                            let bc = IVec3::new(x, y, z);
                            if layer.d[l].contains_key(&bc) {
                                keys.push((l as u8, bc));
                            }
                        }
                    }
                }
            }
        }
        if keys.is_empty() {
            return;
        }
        // Only blocks the brush's sphere reaches.
        keys.retain(|&(l, bc)| {
            let v = frame.voxel0 / (1u32 << l) as f32;
            let lo = frame.pos(l as usize, bc * B);
            let near = c.clamp(lo, lo + Vec3::splat(v * (B - 1) as f32));
            (near - c).length() < r
        });
        if keys.is_empty() {
            return;
        }
        let Source::Paint { fill, m } = &layer.masks[mi].source else { return };
        let fill = *fill;
        let painted: Vec<(usize, IVec3, Box<[f32; N]>)> = keys
            .par_iter()
            .map(|&(l8, bc)| {
                let l = l8 as usize;
                let mut v = match m[l].get(&bc) {
                    Some(b) => Box::new(**b),
                    None => Box::new([fill; N]),
                };
                for i in 0..N {
                    let t = (1.0 - (frame.pos(l, bc * B + cell(i)) - c).length() / r).clamp(0.0, 1.0);
                    v[i] += t * t * (3.0 - 2.0 * t) * (to - v[i]);
                }
                (l, bc, v)
            })
            .collect();
        self.reweigh(vol, k, Some(keys), |x| {
            let Source::Paint { fill, m } = &mut x.masks[mi].source else { return };
            for (l, bc, v) in painted {
                set_paint(&mut m[l], *fill, bc, v);
            }
        });
    }

    /// Remove layer `k` and what it shows (inside a stroke of `vol`).
    pub fn remove(&mut self, vol: &mut LevelVolume, k: usize) {
        if k >= self.stack.len() {
            return;
        }
        self.set_visible(vol, k, false);
        self.stack.remove(k);
        if self.active >= self.stack.len() {
            self.active = self.stack.len().saturating_sub(1);
        }
    }

    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        let u32w = |w: &mut dyn Write, x: u32| w.write_all(&x.to_le_bytes());
        let f32w = |w: &mut dyn Write, x: f32| w.write_all(&x.to_le_bytes());
        let maps = |w: &mut dyn Write, maps: &[Shared<IVec3, Stored>]| -> std::io::Result<()> {
            for map in maps {
                let mut keys: Vec<IVec3> = map.keys().copied().collect();
                keys.sort_unstable_by_key(|k| (k.z, k.y, k.x));
                w.write_all(&(keys.len() as u32).to_le_bytes())?;
                for bc in keys {
                    for c in bc.to_array() {
                        w.write_all(&c.to_le_bytes())?;
                    }
                    let s = map.get(&bc).unwrap();
                    let mut bytes = Vec::with_capacity(N * 4);
                    for v in s.iter() {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.write_all(&bytes)?;
                }
            }
            Ok(())
        };
        u32w(w, V4)?;
        u32w(w, self.stack.len() as u32)?;
        u32w(w, self.active as u32)?;
        for x in &self.stack {
            u32w(w, x.name.len() as u32)?;
            w.write_all(x.name.as_bytes())?;
            w.write_all(&[x.visible as u8])?;
            f32w(w, x.opacity)?;
            w.write_all(&[LayerBlend::ALL.iter().position(|&b| b == x.blend).unwrap_or(0) as u8])?;
            u32w(w, x.d.len() as u32)?;
            maps(w, &x.d)?;
            for map in &x.cols {
                let mut keys: Vec<IVec3> = map.keys().copied().collect();
                keys.sort_unstable_by_key(|k| (k.z, k.y, k.x));
                u32w(w, keys.len() as u32)?;
                for bc in keys {
                    let cols = map.get(&bc).unwrap();
                    let mut bytes = Vec::with_capacity(16 + cols.len() * 22);
                    for c in bc.to_array() {
                        bytes.extend_from_slice(&c.to_le_bytes());
                    }
                    bytes.extend_from_slice(&(cols.len() as u32).to_le_bytes());
                    for c in cols.iter() {
                        bytes.extend_from_slice(&c.i.to_le_bytes());
                        for v in [c.b, c.n.x, c.n.y, c.n.z, c.h] {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    w.write_all(&bytes)?;
                }
            }
            u32w(w, x.masks.len() as u32)?;
            u32w(w, x.active_mask.map_or(u32::MAX, |i| i as u32))?;
            for mk in &x.masks {
                u32w(w, mk.name.len() as u32)?;
                w.write_all(mk.name.as_bytes())?;
                w.write_all(&[mk.visible as u8, BlendMode::ALL.iter().position(|&b| b == mk.blend).unwrap_or(0) as u8])?;
                f32w(w, mk.opacity)?;
                match &mk.source {
                    Source::Paint { fill, m } => {
                        w.write_all(&[0])?;
                        f32w(w, *fill)?;
                        u32w(w, m.len() as u32)?;
                        maps(w, m)?;
                    }
                    Source::Noise { noise, .. } => {
                        let p = &noise.params;
                        w.write_all(&[1, KINDS.iter().position(|&k| k == p.kind).unwrap_or(1) as u8, noise.invert as u8])?;
                        for v in [p.scale, p.lacunarity, p.gain, p.offset.x, p.offset.y, p.offset.z, noise.balance, noise.contrast] {
                            f32w(w, v)?;
                        }
                        u32w(w, p.octaves)?;
                        u32w(w, p.seed)?;
                    }
                }
            }
        }
        u32w(w, self.lin.len() as u32)?;
        maps(w, &self.lin)?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> std::io::Result<SculptLayers> {
        fn u32r(r: &mut dyn Read) -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        }
        fn f32r(r: &mut dyn Read) -> std::io::Result<f32> {
            Ok(f32::from_bits(u32r(r)?))
        }
        fn bytes(r: &mut dyn Read, n: usize) -> std::io::Result<Vec<u8>> {
            let mut b = vec![0u8; n];
            r.read_exact(&mut b)?;
            Ok(b)
        }
        fn name(r: &mut dyn Read) -> std::io::Result<String> {
            let len = u32r(r)? as usize;
            if len > 4096 {
                return Err(bad("name too long"));
            }
            Ok(String::from_utf8_lossy(&bytes(r, len)?).into_owned())
        }
        fn maps(r: &mut dyn Read, out: &mut [Shared<IVec3, Stored>]) -> std::io::Result<()> {
            let mut buf = vec![0u8; 12 + N * 4];
            for map in out.iter_mut() {
                let count = u32r(r)? as usize;
                for _ in 0..count {
                    r.read_exact(&mut buf)?;
                    let c = |i: usize| i32::from_le_bytes(buf[4 * i..4 * i + 4].try_into().unwrap());
                    let bc = IVec3::new(c(0), c(1), c(2));
                    let mut s = Box::new([0.0f32; N]);
                    for (i, v) in s.iter_mut().enumerate() {
                        let o = 12 + 4 * i;
                        *v = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
                    }
                    map.insert(bc, Stored::from(s));
                }
            }
            Ok(())
        }
        fn levels(r: &mut dyn Read) -> std::io::Result<usize> {
            let n = u32r(r)? as usize;
            if n > 32 {
                return Err(bad("too many levels"));
            }
            Ok(n)
        }
        fn bad(s: &str) -> std::io::Error {
            std::io::Error::new(std::io::ErrorKind::InvalidData, s.to_string())
        }
        let first = u32r(r)?;
        let v4 = first == V4;
        let v3 = first == V3 || v4;
        let v2 = first == V2 || v3;
        let n = if v2 { u32r(r)? } else { first } as usize;
        let active = u32r(r)? as usize;
        if n > 4096 {
            return Err(bad("too many sculpt layers"));
        }
        let mut stack = Vec::with_capacity(n);
        for _ in 0..n {
            let nm = name(r)?;
            let visible = bytes(r, 1)?[0] != 0;
            let opacity = if v2 { f32r(r)? } else { 1.0 };
            let blend = if v3 { LayerBlend::ALL.get(bytes(r, 1)?[0] as usize).copied().unwrap_or_default() } else { LayerBlend::Add };
            let lv = levels(r)?;
            let mut x = SculptLayer::new(&nm, lv);
            x.visible = visible;
            x.opacity = opacity;
            x.blend = blend;
            maps(r, &mut x.d)?;
            if v4 {
                for map in x.cols.iter_mut() {
                    let count = u32r(r)? as usize;
                    for _ in 0..count {
                        let head = bytes(r, 16)?;
                        let c = |i: usize| i32::from_le_bytes(head[4 * i..4 * i + 4].try_into().unwrap());
                        let bc = IVec3::new(c(0), c(1), c(2));
                        let n = c(3) as usize;
                        if n > N {
                            return Err(bad("too many columns"));
                        }
                        let body = bytes(r, n * 22)?;
                        let cols: Vec<Column> = body
                            .chunks_exact(22)
                            .map(|b| {
                                let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
                                Column { i: u16::from_le_bytes([b[0], b[1]]).min(N as u16 - 1), b: f(2), n: Vec3::new(f(6), f(10), f(14)), h: f(18) }
                            })
                            .collect();
                        map.insert(bc, Arc::new(cols));
                    }
                }
            }
            if !v2 {
                // One mask then, held with the change: a painted one, white where nothing was painted.
                let mut m = vec![Shared::default(); lv];
                maps(r, &mut m)?;
                if m.iter().any(|m| !m.is_empty()) {
                    x.masks.push(MaskLayer { name: "Paint 1".into(), visible: true, opacity: 1.0, blend: BlendMode::Normal, source: Source::Paint { fill: 1.0, m } });
                }
                stack.push(x);
                continue;
            }
            let count = u32r(r)? as usize;
            if count > 4096 {
                return Err(bad("too many masks"));
            }
            let am = u32r(r)?;
            for _ in 0..count {
                let nm = name(r)?;
                let vb = bytes(r, 2)?;
                let opacity = f32r(r)?;
                let blend = BlendMode::ALL.get(vb[1] as usize).copied().unwrap_or_default();
                let source = match bytes(r, 1)?[0] {
                    0 => {
                        let fill = f32r(r)?;
                        let ml = levels(r)?;
                        let mut m = vec![Shared::default(); ml.max(lv)];
                        maps(r, &mut m[..ml])?;
                        Source::Paint { fill, m }
                    }
                    _ => {
                        let kb = bytes(r, 2)?;
                        let mut f = [0.0f32; 8];
                        for v in &mut f {
                            *v = f32r(r)?;
                        }
                        let (octaves, seed) = (u32r(r)?, u32r(r)?);
                        let params = NoiseParams {
                            kind: KINDS.get(kb[0] as usize).copied().unwrap_or_default(),
                            scale: f[0],
                            octaves: octaves.min(12),
                            lacunarity: f[1],
                            gain: f[2],
                            seed,
                            offset: Vec3::new(f[3], f[4], f[5]),
                        };
                        Source::noise(NoiseMask { params, balance: f[6], contrast: f[7], invert: kb[1] != 0 })
                    }
                };
                x.masks.push(MaskLayer { name: nm, visible: vb[0] != 0, opacity, blend, source });
            }
            x.active_mask = (am != u32::MAX && (am as usize) < x.masks.len()).then_some(am as usize);
            stack.push(x);
        }
        let mut lin = Vec::new();
        if v4 {
            let lv = levels(r)?;
            lin = vec![Shared::default(); lv];
            maps(r, &mut lin)?;
        }
        Ok(SculptLayers { active: active.min(n.saturating_sub(1)), stack, lin })
    }
}

/// Marks the layers' second form (with opacity and mask stacks); the first began with the count of layers.
const V2: u32 = 0x3253_4C59;
/// The third adds each layer's blend.
const V3: u32 = 0x3353_4C59;
/// The fourth adds each layer's heights.
const V4: u32 = 0x3453_4C59;
const KINDS: [NoiseKind; 5] = [NoiseKind::Perlin, NoiseKind::Fbm, NoiseKind::Ridged, NoiseKind::Turbulence, NoiseKind::Cellular];

#[cfg(test)]
mod tests {
    use super::*;

    fn ball() -> LevelVolume {
        LevelVolume::from_field(&|p: Vec3| p.length() - 0.5, 1.0, 1.0 / 16.0, 4)
    }

    /// Whether two fields show one surface: no sample on the other side of it, and samples near it (within half the
    /// band, where values are not clipped) within `tol` of a step.
    fn same(a: &LevelVolume, b: &LevelVolume, tol: f32) {
        let (flips, worst) = off(a, b);
        assert!(flips == 0 && worst <= tol, "{flips} samples change side, near the surface off by {worst} steps");
    }

    /// Samples on the other side of the surface, and the most samples near it are off by, in steps.
    fn off(a: &LevelVolume, b: &LevelVolume) -> (usize, f32) {
        let (mut worst, mut flips) = (0.0f32, 0);
        for l in 0..a.levels() {
            let mut keys = a.level_keys(l);
            keys.extend(b.level_keys(l));
            let half = 0.5 * a.band(l);
            for bc in keys {
                let (x, y) = (a.block_values(l, bc, false), b.block_values(l, bc, false));
                for i in 0..N {
                    if (x[i] < 0.0) != (y[i] < 0.0) && x[i].abs().max(y[i].abs()) > 1e-4 * a.voxel(l) {
                        flips += 1;
                    }
                    if x[i].abs() < half || y[i].abs() < half {
                        worst = worst.max((x[i] - y[i]).abs() / a.voxel(l));
                    }
                }
            }
        }
        (flips, worst)
    }

    fn differs(a: &LevelVolume, b: &LevelVolume) -> bool {
        (0..a.levels()).any(|l| a.level_keys(l).into_iter().any(|bc| a.block_values(l, bc, false) != b.block_values(l, bc, false)))
    }

    fn stroke(vol: &mut LevelVolume, l: usize, at: Vec3) {
        stroke_h(vol, l, at, 0.03);
    }

    fn stroke_h(vol: &mut LevelVolume, l: usize, at: Vec3, h: f32) {
        vol.begin_stroke();
        for i in 0..6 {
            let p = at + Vec3::new(0.02 * i as f32, 0.0, 0.0);
            vol.plane_dab(l, p, p.normalize(), 0.12, h, 0.5, 1.0, false);
        }
        vol.end_stroke();
    }

    fn step(vol: &mut LevelVolume, f: impl FnOnce(&mut LevelVolume)) {
        vol.begin_stroke();
        f(vol);
        vol.end_stroke_plain();
    }

    /// Hiding a layer takes its strokes away exactly; showing it puts them back; a black fill hides it; undo brings
    /// the fill back exactly.
    #[test]
    fn hide_show_and_mask() {
        let mut vol = ball();
        let start = vol.snapshot();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        stroke(&mut vol, 3, Vec3::new(0.05, 0.03, 0.5));
        layers.take_stroke(&vol, Some(k));
        assert!(layers.layers()[0].blocks > 0);
        let sculpted = vol.snapshot();

        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);
        step(&mut vol, |v| layers.set_visible(v, k, true));
        same(&vol, &sculpted, 0.01);
        step(&mut vol, |v| layers.fill_mask(v, k, 0.0));
        same(&vol, &start, 0.01);
        assert!(vol.undo());
        same(&vol, &sculpted, 1e-5);
    }

    /// Sculpting where a layer shows half keeps it whole: hiding it again leaves what was there first.
    #[test]
    fn stroke_over_mask() {
        let mut vol = ball();
        let start = vol.snapshot();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        step(&mut vol, |v| layers.fill_mask(v, k, 0.5));
        stroke(&mut vol, 2, Vec3::new(0.02, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        step(&mut vol, |v| layers.set_opacity(v, k, 0.3));
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);
    }

    /// Opacity, noise and painted masks with blends move the volume; taking them away again gives what was sculpted.
    #[test]
    fn mask_stack() {
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        stroke(&mut vol, 3, Vec3::new(0.05, 0.03, 0.5));
        layers.take_stroke(&vol, Some(k));
        let sculpted = vol.snapshot();

        step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
        assert!(differs(&vol, &sculpted));
        let noise = step_ret(&mut vol, |v| layers.add_mask(v, k, MaskKind::Noise(NoiseMask { params: NoiseParams { scale: 20.0, ..Default::default() }, ..Default::default() }))).unwrap();
        let paint = step_ret(&mut vol, |v| layers.add_mask(v, k, MaskKind::Paint { fill: 0.0 })).unwrap();
        let mut info = layers.layers()[k].masks[paint].clone();
        info.blend = BlendMode::Add;
        step(&mut vol, |v| layers.set_mask(v, k, paint, info));
        step(&mut vol, |v| layers.paint_mask(v, k, Vec3::new(0.03, 0.0, 0.5), 0.05, 1.0));
        let mut info = layers.layers()[k].masks[noise].clone();
        if let MaskKind::Noise(n) = &mut info.kind {
            n.params.seed = 7;
            n.contrast = 0.9;
        }
        step(&mut vol, |v| layers.set_mask(v, k, noise, info));
        step(&mut vol, |v| layers.move_mask(v, k, 0, true));
        assert!(differs(&vol, &sculpted));
        // Taken away: the layer at full, as sculpted.
        step(&mut vol, |v| layers.remove_mask(v, k, 1));
        step(&mut vol, |v| layers.remove_mask(v, k, 0));
        step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
        assert!(layers.layers()[k].masks.is_empty());
        same(&vol, &sculpted, 0.01);

        // A stroke through a noise mask; hiding and showing stay exact.
        step(&mut vol, |v| {
            layers.add_mask(v, k, MaskKind::Noise(NoiseMask { params: NoiseParams { scale: 20.0, ..Default::default() }, ..Default::default() }));
        });
        stroke(&mut vol, 3, Vec3::new(-0.04, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        step(&mut vol, |v| layers.set_visible(v, k, false));
        let hidden = vol.snapshot();
        // Hiding then showing is exact at any stack.
        step(&mut vol, |v| layers.set_visible(v, k, true));
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &hidden, 0.01);
    }

    /// Every sample of `a` on one side of `b`'s (`below`: a no more than b), within `tol` of a step.
    fn side(a: &LevelVolume, b: &LevelVolume, below: bool, tol: f32) {
        let mut worst = 0.0f32;
        for l in 0..a.levels() {
            for bc in a.level_keys(l) {
                let (x, y) = (a.block_values(l, bc, false), b.block_values(l, bc, false));
                for i in 0..N {
                    let over = if below { x[i] - y[i] } else { y[i] - x[i] };
                    worst = worst.max(over / a.voxel(l));
                }
            }
        }
        assert!(worst <= tol, "off the side by {worst} steps");
    }

    fn trim(vol: &mut LevelVolume, l: usize, at: Vec3, h: f32) {
        vol.begin_stroke();
        for i in 0..6 {
            let p = at + Vec3::new(0.0, 0.02 * i as f32, 0.0);
            vol.plane_dab(l, p, p.normalize(), 0.1, h, 0.5, 1.0, true);
        }
        vol.end_stroke();
    }

    /// A detail layer's blend (its changes within the band of every level, where the changes held give every blend
    /// exactly): turned over, only its raises, only its carves, and back as sculpted; strokes on a turned layer show as
    /// sculpted; hiding is exact in every blend.
    #[test]
    fn layer_blends() {
        let mut vol = ball();
        let start = vol.snapshot();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke_h(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5), 0.012);
        layers.take_stroke(&vol, Some(k));
        trim(&mut vol, 3, Vec3::new(0.0, 0.25, 0.43), 0.012);
        layers.take_stroke(&vol, Some(k));
        let sculpted = vol.snapshot();

        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Raise));
        side(&vol, &start, true, 1e-3);
        assert!(differs(&vol, &sculpted) && differs(&vol, &start));
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Carve));
        side(&vol, &start, false, 1e-3);
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);
        step(&mut vol, |v| layers.set_visible(v, k, true));
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        assert!(differs(&vol, &sculpted));
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
        same(&vol, &sculpted, 0.01);
        step(&mut vol, |v| layers.set_opacity(v, k, 2.0));
        assert!(differs(&vol, &sculpted));
        step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
        same(&vol, &sculpted, 0.01);

        // A stroke on the turned layer shows as sculpted, and turning it back shows it turned.
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        let turned = vol.snapshot();
        stroke_h(&mut vol, 3, Vec3::new(0.05, 0.03, 0.5), 0.012);
        layers.take_stroke(&vol, Some(k));
        let more = vol.snapshot();
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        same(&vol, &more, 0.01);
        assert!(differs(&vol, &turned));
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);

        let mut bytes = Vec::new();
        layers.write_to(&mut bytes).unwrap();
        let back = SculptLayers::read_from(&mut &bytes[..]).unwrap();
        assert_eq!(back.layers(), layers.layers());
    }

    fn step_ret<T>(vol: &mut LevelVolume, f: impl FnOnce(&mut LevelVolume) -> T) -> T {
        vol.begin_stroke();
        let out = f(vol);
        vol.end_stroke_plain();
        out
    }

    #[test]
    fn write_and_read() {
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        step(&mut vol, |v| layers.paint_mask(v, k, Vec3::new(0.0, 0.0, 0.5), 0.05, 0.0));
        step(&mut vol, |v| {
            layers.add_mask(v, k, MaskKind::Noise(NoiseMask::default()));
        });
        step(&mut vol, |v| layers.set_opacity(v, k, 0.7));
        let mut bytes = Vec::new();
        layers.write_to(&mut bytes).unwrap();
        let back = SculptLayers::read_from(&mut &bytes[..]).unwrap();
        let (a, b) = (&back.layers()[0], &layers.layers()[0]);
        assert_eq!((a.blocks, &a.name, a.opacity, &a.masks, a.active_mask), (b.blocks, &b.name, b.opacity, &b.masks, b.active_mask));
        // The weights, heights and what the changes held give read back the same.
        let frame = Frame::of(&vol);
        for (l, bc) in layers.stack[0].keys() {
            assert_eq!(layers.stack[0].weight(&frame, l as usize, bc), back.stack[0].weight(&frame, l as usize, bc));
        }
        for l in 0..vol.levels() {
            for (bc, cols) in layers.stack[0].cols[l].iter() {
                assert_eq!(Some(cols), back.stack[0].cols[l].get(bc));
            }
            assert_eq!(layers.lin.get(l).map_or(0, |m| m.len()), back.lin.get(l).map_or(0, |m| m.len()));
        }
    }

    /// The finest-level height of the surface over the ball at `x` along +z, in finest steps.
    fn top(vol: &LevelVolume, x: f32) -> f32 {
        let hit = vol.raycast(Vec3::new(x, 0.0, 2.0), -Vec3::Z, 3, false).map_or(f32::NAN, |h| h.0.z);
        (hit - (0.25 - x * x).sqrt()) / vol.voxel(3)
    }

    /// Lines along z over the lift crossing the surface more than once (a surface where there is none, or an overhang).
    fn strays(vol: &LevelVolume) -> usize {
        let mut lines = 0;
        let st = 0.5 * vol.voxel(3);
        for yi in -10..=10 {
            for xi in -12..=12 {
                let (x, y) = (0.015 * xi as f32, 0.015 * yi as f32);
                let mut z = 1.3f32;
                let mut prev = vol.value_upto(Vec3::new(x, y, z), 3, false);
                let mut n = 0;
                while z > 0.2 {
                    z -= st;
                    let v = vol.value_upto(Vec3::new(x, y, z), 3, false);
                    if (v < 0.0) != (prev < 0.0) {
                        n += 1;
                    }
                    prev = v;
                }
                lines += (n > 1) as usize;
            }
        }
        lines
    }

    /// A layer far taller than the band (58 finest steps): turned over it is a dent as deep, doubled twice as high,
    /// halved half as high, with no surface where there is none; at full, hidden and shown again it is exactly as it
    /// was.
    #[test]
    fn tall_layer() {
        let mut vol = ball();
        let start = vol.snapshot();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        for _ in 0..12 {
            stroke(&mut vol, 3, Vec3::new(-0.05, 0.0, 0.5));
            layers.take_stroke(&vol, Some(k));
        }
        let sculpted = vol.snapshot();
        let h = top(&vol, 0.0);
        assert!(h > 50.0, "lift {h}");
        // (The lift's own overhangs cross some lines more than once.)
        let own = strays(&vol);

        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        assert!(top(&vol, 0.0) < -0.9 * h, "turned over {} for a lift of {h}", top(&vol, 0.0));
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
        same(&vol, &sculpted, 0.01);
        step(&mut vol, |v| layers.set_opacity(v, k, 2.0));
        assert!((top(&vol, 0.0) / h - 2.0).abs() < 0.1, "doubled {} for a lift of {h}", top(&vol, 0.0));
        step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
        assert!((top(&vol, 0.0) / h - 0.5).abs() < 0.1, "halved {} for a lift of {h}", top(&vol, 0.0));
        assert!(strays(&vol) <= own, "{} lines cross the surface more than once at half, {own} at full", strays(&vol));
        step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
        same(&vol, &sculpted, 0.01);
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);
        assert_eq!(strays(&vol), 0);
        step(&mut vol, |v| layers.set_visible(v, k, true));
        same(&vol, &sculpted, 0.01);
    }

    /// Volumes with a tall layer in each blend and weight, saved for screenshots (`SCULPT_SHOTS` names a folder).
    #[test]
    #[ignore]
    fn tall_shots() {
        let Ok(dir) = std::env::var("SCULPT_SHOTS") else { return };
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        for _ in 0..12 {
            vol.begin_stroke();
            for i in 0..8 {
                let y = -0.07 + 0.02 * i as f32;
                let p = Vec3::new((0.25 - y * y).sqrt(), y, 0.0);
                vol.plane_dab(3, p, p.normalize(), 0.12, 0.03, 0.5, 1.0, false);
            }
            vol.end_stroke();
            layers.take_stroke(&vol, Some(k));
        }
        let save = |vol: &LevelVolume, layers: &SculptLayers, name: &str| {
            let mut bytes = Vec::new();
            vol.write_to(&mut bytes).unwrap();
            let mut l = Vec::new();
            layers.write_to(&mut l).unwrap();
            bytes.extend_from_slice(b"SCLY");
            bytes.extend_from_slice(&(l.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&l);
            std::fs::write(format!("{dir}/{name}.svol"), bytes).unwrap();
        };
        save(&vol, &layers, "a_sculpted");
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        save(&vol, &layers, "b_subtract");
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
        step(&mut vol, |v| layers.set_opacity(v, k, 2.0));
        save(&vol, &layers, "c_200");
        step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
        save(&vol, &layers, "d_50");
        step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
        step(&mut vol, |v| layers.paint_mask(v, k, Vec3::new(0.95, 0.07, 0.0), 0.25, 0.0));
        save(&vol, &layers, "e_masked");
        // Turned over where it faces the camera, to see into the dent.
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        for _ in 0..12 {
            stroke(&mut vol, 3, Vec3::new(-0.05, 0.0, 0.5));
            layers.take_stroke(&vol, Some(k));
        }
        save(&vol, &layers, "f_front");
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Subtract));
        save(&vol, &layers, "g_front_subtract");
        step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
        step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
        save(&vol, &layers, "h_front_50");
    }

    /// Lines along z over the front crossing the surface more than once, between `x0` and `x1`.
    fn strays_in(vol: &LevelVolume, x0: f32, x1: f32) -> usize {
        let mut lines = 0;
        let st = 0.5 * vol.voxel(3);
        for yi in -12..=12 {
            let mut x = x0;
            while x <= x1 {
                let y = 0.0125 * yi as f32;
                let mut z = 1.3f32;
                let mut prev = vol.value_upto(Vec3::new(x, y, z), 3, false);
                let mut n = 0;
                while z > 0.2 {
                    z -= st;
                    let v = vol.value_upto(Vec3::new(x, y, z), 3, false);
                    if (v < 0.0) != (prev < 0.0) {
                        n += 1;
                    }
                    prev = v;
                }
                lines += (n > 1) as usize;
                x += 0.0125;
            }
        }
        lines
    }

    /// A tall raise and a deep carve on one layer, in every blend: each shows as it should at full and half, and comes
    /// back exactly at full however its weight went there (heights and lines crossing the surface more than once are
    /// printed with `--nocapture`; `SCULPT_SHOTS` saves volumes and cross sections for screenshots).
    #[test]
    fn tall_blends() {
        let dir = std::env::var("SCULPT_SHOTS").ok();
        let start = ball();
        let make = || {
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        for _ in 0..10 {
            stroke(&mut vol, 3, Vec3::new(-0.25, 0.0, 0.433));
            layers.take_stroke(&vol, Some(k));
        }
        for _ in 0..8 {
            vol.begin_stroke();
            for i in 0..6 {
                let x = 0.17 + 0.02 * i as f32;
                let p = Vec3::new(x, 0.0, (0.25 - x * x).sqrt());
                vol.plane_dab(3, p, p.normalize(), 0.1, 0.03, 0.5, 1.0, true);
            }
            vol.end_stroke();
            layers.take_stroke(&vol, Some(k));
        }
        (vol, layers, k)
        };
        let (mut vol, mut layers, k) = make();
        let sculpted = vol.snapshot();
        // Heights along the ball's normal through the middle of each, in finest steps.
        let along = |vol: &LevelVolume, x: f32| {
            let d = Vec3::new(x, 0.0, (0.25 - x * x).sqrt()).normalize();
            let hit = vol.raycast(d * 2.0, -d, 3, false).map_or(f32::NAN, |h| h.0.length());
            (hit - 0.5) / vol.voxel(3)
        };
        let (rx, cx) = (-0.2, 0.22);
        let at = |vol: &LevelVolume| format!("raise {:6.1} carve {:6.1} strays {:3}", along(vol, rx), along(vol, cx), strays_in(vol, -0.45, 0.35));
        println!("BLEND sculpted        {}", at(&vol));
        // Cross sections through y = 0 and y = 0.05 (P5 images, inside dark), written beside the shots.
        let slice = |vol: &LevelVolume, name: &str| {
            let Some(dir) = &dir else { return };
            for (yn, y) in [("0", 0.0f32), ("05", 0.05)] {
                let (w, h) = (360usize, 300usize);
                let mut img = Vec::with_capacity(w * h);
                for j in 0..h {
                    for i in 0..w {
                        let p = Vec3::new(-0.45 + 0.9 * i as f32 / w as f32, y, 0.95 - 0.75 * j as f32 / h as f32);
                        let v = vol.value_upto(p, 3, false) / vol.voxel(3);
                        img.push(if v < 0.0 { (60.0 + 20.0 * v.max(-2.5)) as u8 } else { (200.0 + 20.0 * v.min(2.5)) as u8 });
                    }
                }
                let mut bytes = format!("P5 {w} {h} 255
").into_bytes();
                bytes.extend(img);
                std::fs::write(format!("{dir}/slice_{name}_{yn}.pgm"), bytes).unwrap();
            }
        };
        slice(&vol, "add");

        let save = |vol: &LevelVolume, layers: &SculptLayers, name: &str| {
            let Some(dir) = &dir else { return };
            let mut bytes = Vec::new();
            vol.write_to(&mut bytes).unwrap();
            let mut l = Vec::new();
            layers.write_to(&mut l).unwrap();
            bytes.extend_from_slice(b"SCLY");
            bytes.extend_from_slice(&(l.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&l);
            std::fs::write(format!("{dir}/{name}.svol"), bytes).unwrap();
        };
        save(&vol, &layers, "a_add");
        for (name, blend) in [("subtract", LayerBlend::Subtract), ("max", LayerBlend::Raise), ("min", LayerBlend::Carve)] {
            (vol, layers, _) = make();
            step(&mut vol, |v| layers.set_blend(v, k, blend));
            println!("BLEND {name:8} 100%    {}", at(&vol));
            let (r, c) = (along(&vol, rx), along(&vol, cx));
            let (r0, c0) = (along(&sculpted, rx), along(&sculpted, cx));
            let (er, ec) = match blend {
                LayerBlend::Subtract => (-r0, -c0),
                LayerBlend::Raise => (r0, 0.0),
                _ => (0.0, c0),
            };
            assert!((r - er).abs() < 0.1 * r0.abs() + 1.0 && (c - ec).abs() < 0.1 * c0.abs() + 1.0, "{name}: raise {r} carve {c}, wanted {er} {ec}");
            slice(&vol, name);
            let before = vol.snapshot();
            save(&vol, &layers, &format!("{name}_100"));
            step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
            println!("BLEND {name:8} 50%     {}", at(&vol));
            let (r2, c2) = (along(&vol, rx), along(&vol, cx));
            assert!((r2 - 0.5 * er).abs() < 0.1 * r0.abs() + 1.0 && (c2 - 0.5 * ec).abs() < 0.1 * c0.abs() + 1.0, "{name} at half: raise {r2} carve {c2}, wanted {} {}", 0.5 * er, 0.5 * ec);
            slice(&vol, &format!("{name}50"));
            save(&vol, &layers, &format!("{name}_50"));
            step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
            same(&vol, &before, 0.01);
            step(&mut vol, |v| layers.set_opacity(v, k, 0.999));
            step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
            same(&vol, &before, 0.01);
            let over = 2.0;
            step(&mut vol, |v| layers.set_opacity(v, k, over));
            println!("BLEND {name:8} {over}x    {}", at(&vol));
            step(&mut vol, |v| layers.set_opacity(v, k, 1.0));
            same(&vol, &before, 0.01);
            step(&mut vol, |v| layers.set_blend(v, k, LayerBlend::Add));
            same(&vol, &sculpted, 0.01);
        }
        step(&mut vol, |v| layers.set_visible(v, k, false));
        same(&vol, &start, 0.01);
    }

    /// How long a bigger layer takes at stroke ends and when its weight or blend changes (printed).
    #[test]
    #[ignore]
    fn layer_timings() {
        use std::time::Instant;
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        let (mut ends, mut most) = (0.0f64, 0.0f64);
        let n = 48;
        for j in 0..n {
            let a = j as f32 * 2.399;
            let z = 1.0 - 2.0 * (j as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).sqrt();
            let at = Vec3::new(r * a.cos(), r * a.sin(), z) * 0.5;
            for _ in 0..(1 + j % 4) {
                stroke(&mut vol, 3, at);
                let t = Instant::now();
                layers.take_stroke(&vol, Some(k));
                let ms = t.elapsed().as_secs_f64() * 1e3;
                ends += ms;
                most = most.max(ms);
            }
        }
        let tall = (0..vol.levels()).filter(|&l| layers.stack[k].tall_at(l, vol.band(l))).count();
        println!("TIME layer {} blocks, tall at {tall} levels; stroke ends {ends:.0} ms in all, worst {most:.1} ms", layers.layers()[k].blocks);
        let timed = |name: &str, vol: &mut LevelVolume, f: &mut dyn FnMut(&mut LevelVolume)| {
            let t = Instant::now();
            step(vol, |v| f(v));
            println!("TIME {name:10} {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        };
        timed("50%", &mut vol, &mut |v| layers.set_opacity(v, k, 0.5));
        timed("60%", &mut vol, &mut |v| layers.set_opacity(v, k, 0.6));
        timed("100%", &mut vol, &mut |v| layers.set_opacity(v, k, 1.0));
        timed("subtract", &mut vol, &mut |v| layers.set_blend(v, k, LayerBlend::Subtract));
        timed("add", &mut vol, &mut |v| layers.set_blend(v, k, LayerBlend::Add));
        timed("hide", &mut vol, &mut |v| layers.set_visible(v, k, false));
        timed("show", &mut vol, &mut |v| layers.set_visible(v, k, true));
        step(&mut vol, |v| layers.set_opacity(v, k, 0.5));
        let t = Instant::now();
        stroke(&mut vol, 3, Vec3::new(0.0, 0.0, 0.5));
        let s = Instant::now();
        layers.take_stroke(&vol, Some(k));
        println!("TIME stroke at 50%: stroke {:.0} ms, layer at its end {:.0} ms", (s - t).as_secs_f64() * 1e3, s.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        step(&mut vol, |v| layers.paint_mask(v, k, Vec3::new(0.0, 0.0, 0.5), 0.08, 0.0));
        println!("TIME mask dab {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

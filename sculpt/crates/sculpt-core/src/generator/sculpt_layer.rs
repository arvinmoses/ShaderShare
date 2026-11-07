//! Sculpt layers: each layer keeps what its strokes changed, at every level and block they changed, an opacity, and
//! a stack of masks.
//!
//! * **What is shown is the truth.** Strokes on a layer sculpt the volume itself, at full speed and at whatever level
//!   the brush picks. When one ends, the layer takes the stroke's change (the field after it less the field before,
//!   block by block, at every level it reached) into its own: `shown = base + Σ w Δ`, `w` the layer's weight (0 to 1):
//!   its opacity times its masks, 0 where it is hidden.
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

/// How much of a stroke's change a layer at weight `w` takes, unweighted (`w` times it is what shows of it).
fn gain(w: f32) -> f32 {
    if w <= 0.0 { 0.0 } else { w / (w.max(LOW) * w.max(LOW)) }
}

#[derive(Clone)]
pub struct SculptLayer {
    pub name: String,
    pub visible: bool,
    opacity: f32,
    /// Per level, the change the layer's strokes made, per block.
    d: Vec<Shared<IVec3, Stored>>,
    /// Bottom first.
    masks: Vec<MaskLayer>,
    active_mask: Option<usize>,
}

impl SculptLayer {
    fn new(name: &str, levels: usize) -> SculptLayer {
        SculptLayer { name: name.to_string(), visible: true, opacity: 1.0, d: vec![Shared::default(); levels], masks: Vec::new(), active_mask: None }
    }

    fn fit(&mut self, levels: usize) {
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
        if self.opacity < 1.0 {
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
                        out[i] = b[i] - w[i] * d[i];
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
            d[i] = if w[i] > 0.01 { shown / w[i] } else { self.d_value_at(l - 1, (bc * B + cell(i)).as_vec3() * 0.5) };
        }
        self.d[l].insert(bc, Stored::from(d));
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
        let mut keys = keys.unwrap_or_else(|| layer.keys());
        // A block the volume does not hold follows the coarser levels, moved already.
        keys.retain(|&(l, bc)| l == 0 || vol.exists(l as usize, bc));
        let w0: Vec<Box<[f32; N]>> = keys.par_iter().map(|&(l, bc)| layer.weight(&frame, l as usize, bc)).collect();
        edit(layer);
        let (layer, read) = (&*layer, &*vol);
        let moved: Vec<(usize, IVec3, Box<[f32; N]>)> = keys
            .par_iter()
            .zip(w0.par_iter())
            .filter_map(|(&(l8, bc), a)| {
                let l = l8 as usize;
                let d = layer.d[l].get(&bc)?;
                let b = layer.weight(&frame, l, bc);
                if b == *a {
                    return None;
                }
                let mut v = read.block_values(l, bc, false);
                for i in 0..N {
                    v[i] += (b[i] - a[i]) * d[i];
                }
                Some((l, bc, v))
            })
            .collect();
        for (l, bc, v) in moved {
            vol.set_block(l, bc, v);
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
        let Some(layer) = on.and_then(|k| self.stack.get_mut(k)) else { return };
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
        for ((l, bc, d), w) in changes.into_iter().zip(weights) {
            let mut nd = match layer.d[l].get(&bc) {
                Some(od) => Box::new(**od),
                None => Box::new([0.0; N]),
            };
            for i in 0..N {
                if d[i] != 0.0 {
                    nd[i] += d[i] * gain(w[i]);
                }
            }
            layer.d[l].insert(bc, Stored::from(nd));
        }
    }

    /// Show or hide layer `k` (inside a stroke of `vol`).
    pub fn set_visible(&mut self, vol: &mut LevelVolume, k: usize, on: bool) {
        if self.stack.get(k).is_none_or(|x| x.visible == on) {
            return;
        }
        self.reweigh(vol, k, None, |x| x.visible = on);
    }

    /// Set layer `k`'s opacity (inside a stroke of `vol`).
    pub fn set_opacity(&mut self, vol: &mut LevelVolume, k: usize, opacity: f32) {
        let opacity = opacity.clamp(0.0, 1.0);
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
        u32w(w, V2)?;
        u32w(w, self.stack.len() as u32)?;
        u32w(w, self.active as u32)?;
        for x in &self.stack {
            u32w(w, x.name.len() as u32)?;
            w.write_all(x.name.as_bytes())?;
            w.write_all(&[x.visible as u8])?;
            f32w(w, x.opacity)?;
            u32w(w, x.d.len() as u32)?;
            maps(w, &x.d)?;
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
        let v2 = first == V2;
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
            let lv = levels(r)?;
            let mut x = SculptLayer::new(&nm, lv);
            x.visible = visible;
            x.opacity = opacity;
            maps(r, &mut x.d)?;
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
        Ok(SculptLayers { active: active.min(n.saturating_sub(1)), stack })
    }
}

/// Marks the layers' second form (with opacity and mask stacks); the first began with the count of layers.
const V2: u32 = 0x3253_4C59;
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
        assert!(flips == 0 && worst <= tol, "{flips} samples change side, near the surface off by {worst} steps");
    }

    fn differs(a: &LevelVolume, b: &LevelVolume) -> bool {
        (0..a.levels()).any(|l| a.level_keys(l).into_iter().any(|bc| a.block_values(l, bc, false) != b.block_values(l, bc, false)))
    }

    fn stroke(vol: &mut LevelVolume, l: usize, at: Vec3) {
        vol.begin_stroke();
        for i in 0..6 {
            let p = at + Vec3::new(0.02 * i as f32, 0.0, 0.0);
            vol.plane_dab(l, p, p.normalize(), 0.12, 0.03, 0.5, 1.0, false);
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
        // The weights read back are the same.
        let frame = Frame::of(&vol);
        for (l, bc) in layers.stack[0].keys() {
            assert_eq!(layers.stack[0].weight(&frame, l as usize, bc), back.stack[0].weight(&frame, l as usize, bc));
        }
    }
}

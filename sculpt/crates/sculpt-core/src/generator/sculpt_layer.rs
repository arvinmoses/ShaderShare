//! Sculpt layers: each layer keeps what its strokes changed, at every level and block they changed, and a mask.
//!
//! * **What is shown is the truth.** Strokes on a layer sculpt the volume itself, at full speed and at whatever level
//!   the brush picks. When one ends, the layer takes the stroke's change (the field after it less the field before,
//!   block by block, at every level it reached) into its own: `shown = base + Σ w Δ`, `w` the layer's mask (0 to 1)
//!   where it shows, 0 where it is hidden.
//! * **Masks and visibility** change `w`; the volume moves by the change times `Δ`, only where the layer holds any.
//! * **Sculpting where a layer is masked** shows it there again: the samples a stroke changes take what showed plus
//!   the stroke as the layer's change, and a mask of 1.
//! * A stroke on the base (no layer) changes nothing kept here: layers do not follow the surface it moves.
//!
//! Every change of the volume here happens inside one of its strokes (begun and ended by the caller), so the volume's
//! own undo steps it back; the caller keeps copies of the layers beside it (cheap: their maps share blocks).

use std::io::{Read, Write};

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::level_volume::{HashMap, Key, LevelVolume, Shared, Stored};

const B: i32 = 8;
const N: usize = 512;

fn cell(i: usize) -> IVec3 {
    IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32)
}

fn local(i: IVec3) -> usize {
    let l = i.rem_euclid(IVec3::splat(B));
    (l.x + 8 * l.y + 64 * l.z) as usize
}

#[derive(Clone)]
pub struct SculptLayer {
    pub name: String,
    pub visible: bool,
    /// Per level, the change the layer's strokes made, per block.
    d: Vec<Shared<IVec3, Stored>>,
    /// Per level, the mask at the same samples (a block with none is all 1).
    m: Vec<Shared<IVec3, Stored>>,
}

impl SculptLayer {
    fn new(name: &str, levels: usize) -> SculptLayer {
        SculptLayer { name: name.to_string(), visible: true, d: vec![Shared::default(); levels], m: vec![Shared::default(); levels] }
    }

    fn fit(&mut self, levels: usize) {
        if self.d.len() < levels {
            self.d.resize(levels, Shared::default());
            self.m.resize(levels, Shared::default());
        }
    }

    pub fn blocks(&self) -> usize {
        self.d.iter().map(|d| d.len()).sum()
    }

    fn mask(&self, l: usize, bc: IVec3) -> Box<[f32; N]> {
        match self.m[l].get(&bc) {
            Some(m) => Box::new(**m),
            None => Box::new([1.0; N]),
        }
    }

    fn set_mask(&mut self, l: usize, bc: IVec3, m: Box<[f32; N]>) {
        if m.iter().all(|&x| x == 1.0) {
            self.m[l].remove(&bc);
        } else {
            self.m[l].insert(bc, Stored::from(m));
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
            let w = (if o.x == 1 { f.x } else { 1.0 - f.x }) * (if o.y == 1 { f.y } else { 1.0 - f.y }) * (if o.z == 1 { f.z } else { 1.0 - f.z });
            if w > 0.0 {
                out += w * self.d_sample(l, i + o);
            }
        }
        out
    }

    /// The mask at level `l`'s lattice point `i`, likewise (1 where nothing is held).
    fn m_sample(&self, l: usize, i: IVec3) -> f32 {
        let bc = i.div_euclid(IVec3::splat(B));
        if self.d[l].contains_key(&bc) {
            return self.m[l].get(&bc).map_or(1.0, |m| m[local(i)]);
        }
        if l == 0 {
            return 1.0;
        }
        let g = i.as_vec3() * 0.5;
        let j = g.floor().as_ivec3();
        let f = g - j.as_vec3();
        let mut out = 0.0;
        for q in 0..8 {
            let o = IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
            let w = (if o.x == 1 { f.x } else { 1.0 - f.x }) * (if o.y == 1 { f.y } else { 1.0 - f.y }) * (if o.z == 1 { f.z } else { 1.0 - f.z });
            if w > 0.0 {
                out += w * self.m_sample(l - 1, j + o);
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
    fn without(&self, vol: &LevelVolume, l: usize, bc: IVec3, cache: &mut HashMap<Key, Box<[f32; N]>>) -> Box<[f32; N]> {
        if let Some(b) = cache.get(&(l as u8, bc)) {
            return b.clone();
        }
        let mut out = Box::new([0.0f32; N]);
        if let Some(b) = vol.block((l as u8, bc)) {
            let d = self.d[l].get(&bc);
            let m = self.m[l].get(&bc);
            for i in 0..N {
                let w = if !self.visible { 0.0 } else { m.map_or(1.0, |m| m[i]) };
                out[i] = b[i] - w * d.map_or(0.0, |d| d[i]);
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
                    held = Some((pb, self.without(vol, l - 1, pb, cache)));
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
                    let w = (if o.x == 1 { f.x } else { 1.0 - f.x }) * (if o.y == 1 { f.y } else { 1.0 - f.y }) * (if o.z == 1 { f.z } else { 1.0 - f.z });
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
    fn inherit(&mut self, before: &LevelVolume, l: usize, bc: IVec3, cache: &mut HashMap<Key, Box<[f32; N]>>) {
        if self.d[l].contains_key(&bc) || !self.under(l, bc) {
            return;
        }
        let with = before.block_values(l, bc, false);
        let without = self.without(before, l, bc, cache);
        let (mut d, mut m) = (Box::new([0.0f32; N]), Box::new([1.0f32; N]));
        for i in 0..N {
            let p = bc * B + cell(i);
            m[i] = self.m_sample(l, p);
            let shown = with[i] - without[i];
            // Stored unweighted, as every change is; where the layer barely shows, as the coarser levels hold it.
            d[i] = if self.visible && m[i] > 0.01 { shown / m[i] } else { self.d_value_at(l - 1, p.as_vec3() * 0.5) };
        }
        self.d[l].insert(bc, Stored::from(d));
        self.set_mask(l, bc, m);
    }

    /// Move the volume by `dw` times the layer's change, at block `bc` of level `l`. A block the volume does not
    /// hold follows the coarser levels, moved already.
    fn shift(&self, vol: &mut LevelVolume, l: usize, bc: IVec3, dw: &[f32; N]) {
        if l > 0 && !vol.exists(l, bc) {
            return;
        }
        let Some(d) = self.d[l].get(&bc) else { return };
        if dw.iter().all(|&x| x == 0.0) {
            return;
        }
        let mut v = vol.block_values(l, bc, false);
        for i in 0..N {
            v[i] += dw[i] * d[i];
        }
        vol.set_block(l, bc, v);
    }

    /// Every block the layer holds, coarsest level first.
    fn keys(&self) -> Vec<Key> {
        let mut out: Vec<Key> = self.d.iter().enumerate().flat_map(|(l, d)| d.keys().map(move |bc| (l as u8, *bc))).collect();
        out.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        out
    }
}

/// What a layer is, for a layer list.
#[derive(Clone, Debug)]
pub struct SculptInfo {
    pub name: String,
    pub visible: bool,
    pub blocks: usize,
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
        self.stack.iter().map(|x| SculptInfo { name: x.name.clone(), visible: x.visible, blocks: x.blocks() }).collect()
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

    /// A new empty layer on top, made active.
    pub fn add(&mut self, name: &str, levels: usize) -> usize {
        self.stack.push(SculptLayer::new(name, levels));
        self.active = self.stack.len() - 1;
        self.active
    }

    /// Take `vol`'s last step (a stroke just ended, on layer `on` or on the base) into the layers.
    pub fn take_stroke(&mut self, vol: &LevelVolume, on: Option<usize>) {
        if self.stack.is_empty() {
            return;
        }
        let Some(before) = vol.before_last() else { return };
        let keys = vol.last_stroke_keys();
        // Blocks it made, coarsest first: each layer's change shows through them as it did.
        let mut made: Vec<Key> = keys.iter().copied().filter(|&(l, bc)| l > 0 && vol.exists(l as usize, bc) && !before.exists(l as usize, bc)).collect();
        made.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        for layer in &mut self.stack {
            layer.fit(vol.levels());
            let mut cache = HashMap::default();
            for &(l, bc) in &made {
                layer.inherit(&before, l as usize, bc, &mut cache);
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
        let vis = layer.visible;
        for (l, bc, d) in changes {
            let mut m = layer.mask(l, bc);
            let mut nd = match layer.d[l].get(&bc) {
                Some(od) => Box::new(**od),
                None => Box::new([0.0; N]),
            };
            for i in 0..N {
                if d[i] != 0.0 {
                    let w = if vis { m[i] } else { 0.0 };
                    nd[i] = w * nd[i] + d[i];
                    m[i] = 1.0;
                }
            }
            layer.d[l].insert(bc, Stored::from(nd));
            layer.set_mask(l, bc, m);
        }
    }

    /// Show or hide layer `k` (inside a stroke of `vol`).
    pub fn set_visible(&mut self, vol: &mut LevelVolume, k: usize, on: bool) {
        let Some(layer) = self.stack.get_mut(k) else { return };
        if layer.visible == on {
            return;
        }
        let sign = if on { 1.0 } else { -1.0 };
        for (l8, bc) in layer.keys() {
            let l = l8 as usize;
            let mut dw = layer.mask(l, bc);
            dw.iter_mut().for_each(|x| *x *= sign);
            layer.shift(vol, l, bc, &dw);
        }
        layer.visible = on;
    }

    /// Set layer `k`'s mask to `to` everywhere (inside a stroke of `vol`).
    pub fn fill_mask(&mut self, vol: &mut LevelVolume, k: usize, to: f32) {
        let Some(layer) = self.stack.get_mut(k) else { return };
        for (l8, bc) in layer.keys() {
            let l = l8 as usize;
            let m = layer.mask(l, bc);
            if layer.visible {
                let mut dw = Box::new([0.0f32; N]);
                for i in 0..N {
                    dw[i] = to - m[i];
                }
                layer.shift(vol, l, bc, &dw);
            }
            layer.set_mask(l, bc, Box::new([to; N]));
        }
    }

    /// Paint layer `k`'s mask toward `to` within `r` of `c`, smoothly to nothing at `r` (inside a stroke of `vol`).
    pub fn paint_mask(&mut self, vol: &mut LevelVolume, k: usize, c: Vec3, r: f32, to: f32) {
        let Some(layer) = self.stack.get_mut(k) else { return };
        let origin = vol.origin();
        for l in 0..layer.d.len().min(vol.levels()) {
            if layer.d[l].is_empty() {
                continue;
            }
            let v = vol.voxel(l);
            let lo = ((c - Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
            let hi = ((c + Vec3::splat(r) - origin) / (v * B as f32)).floor().as_ivec3();
            let span = (hi - lo + IVec3::ONE).as_i64vec3();
            let keys: Vec<IVec3> = if span.x * span.y * span.z > layer.d[l].len() as i64 {
                layer.d[l].keys().copied().filter(|bc| bc.cmpge(lo).all() && bc.cmple(hi).all()).collect()
            } else {
                let mut out = Vec::new();
                for z in lo.z..=hi.z {
                    for y in lo.y..=hi.y {
                        for x in lo.x..=hi.x {
                            let bc = IVec3::new(x, y, z);
                            if layer.d[l].contains_key(&bc) {
                                out.push(bc);
                            }
                        }
                    }
                }
                out
            };
            for bc in keys {
                let mut m = layer.mask(l, bc);
                let mut dw = Box::new([0.0f32; N]);
                for i in 0..N {
                    let d = (origin + (bc * B + cell(i)).as_vec3() * v - c).length() / r;
                    let t = (1.0 - d).clamp(0.0, 1.0);
                    let w = t * t * (3.0 - 2.0 * t);
                    let was = m[i];
                    m[i] += w * (to - m[i]);
                    dw[i] = m[i] - was;
                }
                if layer.visible {
                    layer.shift(vol, l, bc, &dw);
                }
                layer.set_mask(l, bc, m);
            }
        }
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
        u32w(w, self.stack.len() as u32)?;
        u32w(w, self.active as u32)?;
        for x in &self.stack {
            u32w(w, x.name.len() as u32)?;
            w.write_all(x.name.as_bytes())?;
            w.write_all(&[x.visible as u8])?;
            u32w(w, x.d.len() as u32)?;
            for map in x.d.iter().chain(x.m.iter()) {
                let mut keys: Vec<IVec3> = map.keys().copied().collect();
                keys.sort_unstable_by_key(|k| (k.z, k.y, k.x));
                u32w(w, keys.len() as u32)?;
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
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> std::io::Result<SculptLayers> {
        fn u32r(r: &mut dyn Read) -> std::io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        }
        let bad = |s: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, s.to_string());
        let n = u32r(r)? as usize;
        let active = u32r(r)? as usize;
        if n > 4096 {
            return Err(bad("too many sculpt layers"));
        }
        let mut stack = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u32r(r)? as usize;
            if len > 4096 {
                return Err(bad("sculpt layer name too long"));
            }
            let mut name = vec![0u8; len];
            r.read_exact(&mut name)?;
            let mut vis = [0u8; 1];
            r.read_exact(&mut vis)?;
            let levels = u32r(r)? as usize;
            if levels > 32 {
                return Err(bad("too many levels"));
            }
            let mut x = SculptLayer::new(&String::from_utf8_lossy(&name), levels);
            x.visible = vis[0] != 0;
            for j in 0..2 * levels {
                let count = u32r(r)? as usize;
                let mut bytes = vec![0u8; 12 + N * 4];
                for _ in 0..count {
                    r.read_exact(&mut bytes)?;
                    let c = |i: usize| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
                    let bc = IVec3::new(c(0), c(1), c(2));
                    let mut s = Box::new([0.0f32; N]);
                    for (i, v) in s.iter_mut().enumerate() {
                        let o = 12 + 4 * i;
                        *v = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                    }
                    let map = if j < levels { &mut x.d[j] } else { &mut x.m[j - levels] };
                    map.insert(bc, Stored::from(s));
                }
            }
            stack.push(x);
        }
        Ok(SculptLayers { active: active.min(n.saturating_sub(1)), stack })
    }
}

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

    fn stroke(vol: &mut LevelVolume, l: usize, at: Vec3) {
        vol.begin_stroke();
        for i in 0..6 {
            let p = at + Vec3::new(0.02 * i as f32, 0.0, 0.0);
            vol.plane_dab(l, p, p.normalize(), 0.12, 0.03, 0.5, 1.0, false);
        }
        vol.end_stroke();
    }

    /// Hiding a layer takes its strokes away exactly; showing it puts them back; a half mask is half way; a stroke on
    /// the base under it stays.
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

        vol.begin_stroke();
        layers.set_visible(&mut vol, k, false);
        vol.end_stroke_plain();
        same(&vol, &start, 0.01);

        vol.begin_stroke();
        layers.set_visible(&mut vol, k, true);
        vol.end_stroke_plain();
        same(&vol, &sculpted, 0.01);

        vol.begin_stroke();
        layers.fill_mask(&mut vol, k, 0.0);
        vol.end_stroke_plain();
        same(&vol, &start, 0.01);

        // Undo brings the mask step back exactly.
        assert!(vol.undo());
        same(&vol, &sculpted, 1e-5);
    }

    /// Sculpting where a layer is hidden by its mask shows what the stroke made; hiding it again leaves the base.
    #[test]
    fn stroke_over_mask() {
        let mut vol = ball();
        let start = vol.snapshot();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        vol.begin_stroke();
        layers.paint_mask(&mut vol, k, Vec3::new(0.0, 0.0, 0.5), 0.3, 0.0);
        vol.end_stroke_plain();
        stroke(&mut vol, 2, Vec3::new(0.02, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        vol.begin_stroke();
        layers.set_visible(&mut vol, k, false);
        vol.end_stroke_plain();
        same(&vol, &start, 0.01);
    }

    #[test]
    fn write_and_read() {
        let mut vol = ball();
        let mut layers = SculptLayers::default();
        let k = layers.add("Layer 1", vol.levels());
        stroke(&mut vol, 2, Vec3::new(0.0, 0.0, 0.5));
        layers.take_stroke(&vol, Some(k));
        let mut bytes = Vec::new();
        layers.write_to(&mut bytes).unwrap();
        let back = SculptLayers::read_from(&mut &bytes[..]).unwrap();
        assert_eq!(back.layers()[0].blocks, layers.layers()[0].blocks);
        assert_eq!(back.layers()[0].name, "Layer 1");
    }
}

//! Which levels a view draws, region by region.
//!
//! Every stored block above level 0 holds detail over the level below it. A view shows a block only while that detail
//! (with the most any finer block under it adds) would come to more than `tau_px` pixels on screen, seen from the eye.
//! So far work falls back to coarser levels, near work shows in full, and finer blocks that add next to nothing (left
//! by smoothing, or by a big stroke over work since buried) are not drawn at all.
//!
//! What is hidden nests as the stored blocks do: a hidden block's finer blocks are hidden with it. The surface of the
//! blocks shown closes, as the surface of the stored ones does, since the extractor's rules only ask which blocks are
//! there ([`Show`]).
//!
//! A block's own detail is [`LevelVolume::detail_of`], measured again after it changes (a few hundred blocks a frame,
//! in parallel); until then it counts as unbounded, so a block being sculpted always shows.

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::level_volume::{HashMap, HashSet, Key, LevelVolume, Show};

const B: i32 = 8;

/// How much less than `tau_px` a shown block's detail must come to before it is hidden: between the two it keeps what
/// it is, so a slowly moving camera does not make blocks flicker in and out.
const KEEP: f32 = 0.5;

/// Of a finer block's quads, the share a view showing it adds: the rest replace the coarser cells it covers (each a
/// quarter as many over the same surface).
const ADDS: f32 = 0.75;

/// How far over the triangle budget the blocks already shown may keep a view before some are hidden.
const OVER: f32 = 1.1;

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewStats {
    pub shown: usize,
    pub hidden: usize,
    /// Blocks whose detail is still to be measured.
    pub stale: usize,
    /// Blocks shown or hidden by the last [`LevelView::choose`].
    pub flips: usize,
    pub ms: f64,
    /// About how many triangles the blocks shown draw, and the pixels a block's detail had to come to (more than
    /// `tau_px` where the triangle budget asked for less).
    pub triangles: u64,
    pub tau_used: f32,
}

pub struct LevelView {
    hidden: HashSet<Key>,
    /// Each block's own detail over the level below.
    own: HashMap<Key, f32>,
    /// The most the surface moves if a block and all under it are left out: its own detail plus its finer blocks' most.
    tree: HashMap<Key, f32>,
    /// Blocks to measure, each with the sync it last changed in: one changed in the latest is left for later, as a
    /// stroke under way is likely to change it again.
    stale: HashMap<Key, u64>,
    syncs: u64,
    /// About how many quads each block gives the surface ([`LevelVolume::surface_quads`]), and level 0's in all.
    quads: HashMap<Key, u32>,
    base: u64,
    /// The most triangles to draw: past it, blocks show only for more detail than `tau_px`, as far as it takes. Level
    /// 0 always shows, whatever it costs.
    pub budget: Option<u64>,
    /// Detail, in pixels, a block must add to be shown.
    pub tau_px: f32,
    /// Whether to choose at all: off, every stored block shows.
    pub on: bool,
    /// The view last chosen for, and whether anything changed since.
    last: Option<(Vec3, f32, f32, Option<u64>)>,
    changed: bool,
    pub stats: ViewStats,
}

impl Default for LevelView {
    fn default() -> Self {
        LevelView { hidden: HashSet::default(), own: HashMap::default(), tree: HashMap::default(), stale: HashMap::default(), syncs: 0, quads: HashMap::default(), base: 0, budget: None, tau_px: 1.0, on: true, last: None, changed: true, stats: ViewStats::default() }
    }
}

fn children(vol: &LevelVolume, k: Key) -> impl Iterator<Item = Key> + '_ {
    let l = k.0 as usize + 1;
    (0..8).filter_map(move |q| {
        let c = k.1 * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1);
        (l < vol.levels() && vol.exists(l, c)).then_some((l as u8, c))
    })
}

impl LevelView {
    pub fn hidden(&self) -> &HashSet<Key> {
        &self.hidden
    }

    pub fn show(&self, cap: usize) -> Show<'_> {
        Show { cap, hidden: Some(&self.hidden) }
    }

    pub fn hides(&self, k: Key) -> bool {
        self.hidden.contains(&k)
    }

    /// Forget everything (a new volume).
    pub fn clear(&mut self) {
        let (tau, on, budget) = (self.tau_px, self.on, self.budget);
        *self = LevelView { tau_px: tau, on, budget, ..Default::default() };
    }

    /// The blocks `dirty` changed (samples or existence): their detail is to be measured again, and a block that came
    /// under a hidden one is hidden with it.
    pub fn note(&mut self, vol: &LevelVolume, dirty: &[Key]) {
        self.syncs += 1;
        if dirty.is_empty() {
            return;
        }
        // What each costs to draw, measured at once (a cheap count of sign changes).
        let counted: Vec<(Key, u32)> = dirty.par_iter().map(|&k| (k, vol.surface_quads(k.0 as usize, k.1))).collect();
        for (k, q) in counted {
            let was = if q > 0 || vol.exists(k.0 as usize, k.1) { self.quads.insert(k, q) } else { self.quads.remove(&k) };
            if k.0 == 0 {
                self.base = self.base + q as u64 - was.unwrap_or(0) as u64;
            }
        }
        let mut sorted: Vec<Key> = dirty.to_vec();
        sorted.sort_unstable_by_key(|k| k.0);
        let mut up: Vec<Key> = Vec::new();
        for &k in &sorted {
            let (l, bc) = (k.0 as usize, k.1);
            // Level 0 always shows: its detail is never asked for.
            if l == 0 {
                continue;
            }
            if !vol.exists(l, bc) {
                self.own.remove(&k);
                self.stale.remove(&k);
                self.hidden.remove(&k);
                if self.tree.remove(&k).is_some() {
                    up.push(k);
                }
                continue;
            }
            if self.hidden.contains(&(k.0 - 1, bc.div_euclid(IVec3::splat(2)))) {
                self.hidden.insert(k);
            }
            self.own.remove(&k);
            self.stale.insert(k, self.syncs);
            // Already unbounded (still to be measured since an earlier change): so is everything above it.
            if self.tree.get(&k) != Some(&f32::INFINITY) {
                up.push(k);
            }
        }
        self.retree(vol, up);
        self.changed = true;
    }

    /// Measure the detail of up to `most` changed blocks.
    pub fn refresh(&mut self, vol: &LevelVolume, most: usize) {
        if self.stale.is_empty() {
            return;
        }
        let now = self.syncs;
        let take: Vec<Key> = self.stale.iter().filter(|(_, at)| **at < now || most == usize::MAX).map(|(k, _)| *k).take(most).collect();
        if take.is_empty() {
            return;
        }
        let measured: Vec<(Key, f32)> = take.par_iter().map(|&k| (k, vol.detail_of(k.0 as usize, k.1))).collect();
        for (k, e) in &measured {
            self.stale.remove(k);
            self.own.insert(*k, *e);
        }
        self.retree(vol, take);
        self.changed = true;
    }

    /// Work out the tree detail of `keys` and every block above them, finest first.
    fn retree(&mut self, vol: &LevelVolume, keys: Vec<Key>) {
        let mut per: Vec<HashSet<IVec3>> = vec![HashSet::default(); vol.levels()];
        for k in keys {
            if (k.0 as usize) < per.len() {
                per[k.0 as usize].insert(k.1);
            }
        }
        for l in (1..per.len()).rev() {
            let here: Vec<IVec3> = per[l].drain().collect();
            for bc in here {
                let k = (l as u8, bc);
                if vol.exists(l, bc) {
                    let own = self.own.get(&k).copied().unwrap_or(f32::INFINITY);
                    let below = children(vol, k).map(|c| self.tree.get(&c).copied().unwrap_or(f32::INFINITY)).fold(0.0f32, f32::max);
                    let t = own + below;
                    if self.tree.insert(k, t) == Some(t) {
                        continue;
                    }
                }
                per[l - 1].insert(bc.div_euclid(IVec3::splat(2)));
            }
        }
    }

    /// Choose what to show from `eye` (with `focal_px` pixels per unit at distance one). Returns the blocks shown or
    /// hidden since the last choice.
    pub fn choose(&mut self, vol: &LevelVolume, eye: Vec3, focal_px: f32) -> Vec<Key> {
        let t = std::time::Instant::now();
        if !self.on {
            let flips: Vec<Key> = self.hidden.drain().collect();
            self.last = None;
            self.stats = ViewStats { shown: 0, hidden: 0, stale: self.stale.len(), flips: flips.len(), ms: t.elapsed().as_secs_f64() * 1e3, triangles: 0, tau_used: 0.0 };
            return flips;
        }
        let now = (eye, focal_px, self.tau_px, self.budget);
        if !self.changed && self.last == Some(now) {
            self.stats.flips = 0;
            return Vec::new();
        }
        self.last = Some(now);
        self.changed = false;
        let origin = vol.origin();
        let px_of = |k: Key| -> f32 {
            let (l, bc) = (k.0 as usize, k.1);
            let vx = vol.voxel(l);
            let lo = origin + (bc * B).as_vec3() * vx;
            let hi = lo + Vec3::splat(B as f32 * vx);
            let d = (eye.clamp(lo, hi) - eye).length();
            let e = self.tree.get(&k).copied().unwrap_or(f32::INFINITY);
            if d <= 0.0 { f32::INFINITY } else { e * focal_px / d }
        };
        let cost = |k: &Key| self.quads.get(k).copied().unwrap_or(0) as f32 * ADDS * 2.0;
        // Within a budget: the pixels at which the blocks with the most detail on screen first add up to it. A block's
        // detail on screen is never less than any finer block's under it (its tree detail holds theirs, and it is no
        // further away), so every threshold shows a set that nests.
        let (mut tau, mut keep) = (self.tau_px, self.tau_px * KEEP);
        if let Some(budget) = self.budget {
            let mut all: Vec<(f32, f32)> = self.tree.par_iter().map(|(k, _)| (px_of(*k), cost(k))).filter(|(px, _)| *px > keep).collect();
            all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let room = budget as f32 - self.base as f32 * 2.0;
            let cut = |room: f32| -> f32 {
                let mut sum = 0.0;
                for (px, c) in &all {
                    sum += c;
                    if sum > room {
                        return *px;
                    }
                }
                0.0
            };
            tau = tau.max(cut(room));
            keep = keep.max(cut(room + (OVER - 1.0) * budget as f32));
        }
        let mut triangles = self.base as f32 * 2.0;
        let mut flips: Vec<Key> = Vec::new();
        let mut hide_roots: Vec<Key> = Vec::new();
        // Level 0 always shows; level by level down from there, through the blocks shown.
        let mut shown = vol.level_len(0);
        let mut candidates: Vec<Key> = vol.level_keys(1).into_iter().map(|bc| (1u8, bc)).collect();
        while !candidates.is_empty() {
            let decided: Vec<(Key, bool, bool)> = candidates
                .par_iter()
                .map(|&k| {
                    let px = px_of(k);
                    let was_hidden = self.hidden.contains(&k);
                    let show = if was_hidden { px > tau } else { px > keep };
                    (k, show, was_hidden)
                })
                .collect();
            let mut next = Vec::new();
            for (k, show, was_hidden) in decided {
                if show {
                    if was_hidden {
                        self.hidden.remove(&k);
                        flips.push(k);
                    }
                    next.push(k);
                } else if !was_hidden {
                    hide_roots.push(k);
                }
            }
            shown += next.len();
            triangles += next.iter().map(&cost).sum::<f32>();
            candidates = next.par_iter().flat_map_iter(|&p| children(vol, p).collect::<Vec<_>>()).collect();
        }
        // A hidden block takes its finer blocks with it.
        let mut stack = hide_roots;
        while let Some(k) = stack.pop() {
            if self.hidden.insert(k) {
                flips.push(k);
            }
            stack.extend(children(vol, k).filter(|c| !self.hidden.contains(c)));
        }
        self.stats = ViewStats { shown, hidden: self.hidden.len(), stale: self.stale.len(), flips: flips.len(), ms: t.elapsed().as_secs_f64() * 1e3, triangles: triangles as u64, tau_used: tau };
        flips
    }
}

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
//! A block's own detail is [`LevelVolume::detail_of`], measured again after it changes (a few thousand blocks a frame,
//! in parallel); until then it counts as unbounded, so a block being sculpted always shows.
//!
//! The blocks are kept as a tree of nodes in flat arrays (a node per stored block, its parent and children by index),
//! so a choice walks level by level over arrays in parallel, with no lookups: a few milliseconds for half a million
//! blocks.

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

/// No node.
const NONE: u32 = u32::MAX;
/// Not to be measured.
const MEASURED: u64 = u64::MAX;
/// Blocks measured per batch while measuring within a time.
const MEASURE_BATCH: usize = 2048;

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
    /// The node of each stored block.
    index: HashMap<Key, u32>,
    /// Per node: its block (level 255: a free node), its parent, its children by octant, its own detail over the level
    /// below, the most the surface moves if it and all under it are left out (its own detail plus its finer blocks'
    /// most), about how many quads it gives the surface ([`LevelVolume::surface_quads`]), whether it is hidden, and the
    /// sync it last changed in while its detail is still to be measured (one changed in the latest sync is left for
    /// later, as a stroke under way is likely to change it again).
    keys: Vec<Key>,
    parent: Vec<u32>,
    children: Vec<[u32; 8]>,
    own: Vec<f32>,
    tree: Vec<f32>,
    quads: Vec<u32>,
    hid: Vec<bool>,
    stale_at: Vec<u64>,
    /// The nodes of each level, and each node's place in its level's list.
    at_level: Vec<Vec<u32>>,
    in_level: Vec<u32>,
    free: Vec<u32>,
    /// The hidden blocks, as the extractor and the GPU's sync read them.
    hidden: HashSet<Key>,
    /// Nodes to measure (some may have been measured or freed since they were listed).
    stale: Vec<u32>,
    n_stale: usize,
    syncs: u64,
    /// Level 0's quads in all.
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
    /// Seconds a block takes to fade in or out as it hands over to the level below (0: at once). While it fades out
    /// it stays shown; it is hidden once it has faded.
    pub fade_s: f32,
    /// The time, in seconds, fades are measured by ([`Self::set_clock`]).
    clock: f64,
    /// Nodes shown and fading in or out.
    fades: HashMap<u32, Fade>,
    /// Blocks whose fade began, turned or ended since [`Self::take_fade_changes`].
    fade_changed: Vec<Key>,
    /// Per node, for a choice: its detail in pixels, and whether it is shown by this choice (so its children are
    /// considered).
    px: Vec<f32>,
    shown_now: Vec<bool>,
    /// Nodes made since the last choice: a block that just came and would not show is hidden at once (nothing of
    /// it was drawn to fade out).
    fresh: Vec<u32>,
    is_fresh: Vec<bool>,
}

/// A block fading: how far in it was (0 the level below's surface, 1 its own) at `at`, and which way it goes.
#[derive(Clone, Copy, Debug)]
struct Fade {
    t0: f32,
    at: f64,
    rising: bool,
}

impl Default for LevelView {
    fn default() -> Self {
        LevelView {
            index: HashMap::default(),
            keys: Vec::new(),
            parent: Vec::new(),
            children: Vec::new(),
            own: Vec::new(),
            tree: Vec::new(),
            quads: Vec::new(),
            hid: Vec::new(),
            stale_at: Vec::new(),
            at_level: Vec::new(),
            in_level: Vec::new(),
            free: Vec::new(),
            hidden: HashSet::default(),
            stale: Vec::new(),
            n_stale: 0,
            syncs: 0,
            base: 0,
            budget: None,
            tau_px: 1.0,
            on: true,
            last: None,
            changed: true,
            stats: ViewStats::default(),
            fade_s: 0.0,
            clock: 0.0,
            fades: HashMap::default(),
            fade_changed: Vec::new(),
            px: Vec::new(),
            shown_now: Vec::new(),
            fresh: Vec::new(),
            is_fresh: Vec::new(),
        }
    }
}

/// A block's octant among its parent's children.
fn octant(bc: IVec3) -> usize {
    let r = bc.rem_euclid(IVec3::splat(2));
    (r.x | (r.y << 1) | (r.z << 2)) as usize
}

fn parent_key(k: Key) -> Key {
    (k.0 - 1, k.1.div_euclid(IVec3::splat(2)))
}

/// What a choice does to a block.
const KEEP_AS_IS: u8 = 0;
const SHOW: u8 = 1;
const SHOW_MEASURED: u8 = 2;
const REFADE: u8 = 3;
const REFADE_MEASURED: u8 = 4;
const HIDE: u8 = 5;

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
        let (tau, on, budget, fade_s, clock) = (self.tau_px, self.on, self.budget, self.fade_s, self.clock);
        *self = LevelView { tau_px: tau, on, budget, fade_s, clock, ..Default::default() };
    }

    /// Count blocks as hidden for now (as the GPU's sync reads what it has not been given yet); returns those that
    /// were not hidden already, for [`Self::unhide`] after.
    pub fn hide_for_now(&mut self, keys: &[Key]) -> Vec<Key> {
        keys.iter().copied().filter(|k| self.hidden.insert(*k)).collect()
    }

    /// Blocks [`Self::hide_for_now`] hid, as they were.
    pub fn unhide(&mut self, keys: &[Key]) {
        for k in keys {
            self.hidden.remove(k);
        }
    }

    /// The time fades are measured by, in seconds.
    pub fn set_clock(&mut self, now: f64) {
        self.clock = now;
    }

    /// Whether any block is fading.
    pub fn fading(&self) -> bool {
        !self.fades.is_empty()
    }

    /// How far a shown block has faded in at time `now` is `a + b * now` (clamped to 0..1): `(1, 0)` for one not
    /// fading.
    pub fn fade_line(&self, k: Key) -> (f32, f32) {
        match self.index.get(&k).and_then(|n| self.fades.get(n)) {
            None => (1.0, 0.0),
            Some(f) => {
                let rate = if f.rising { 1.0 } else { -1.0 } / self.fade_s.max(1e-6);
                ((f.t0 as f64 - rate as f64 * f.at) as f32, rate)
            }
        }
    }

    /// The blocks whose fade began, turned or ended since last asked.
    pub fn take_fade_changes(&mut self) -> Vec<Key> {
        std::mem::take(&mut self.fade_changed)
    }

    fn fade_now(&self, n: u32) -> f32 {
        self.fades.get(&n).map_or(1.0, |f| {
            let d = ((self.clock - f.at) / self.fade_s.max(1e-6) as f64) as f32;
            (if f.rising { f.t0 + d } else { f.t0 - d }).clamp(0.0, 1.0)
        })
    }

    fn set_fade(&mut self, n: u32, t0: f32, rising: bool) {
        self.fades.insert(n, Fade { t0, at: self.clock, rising });
        self.fade_changed.push(self.keys[n as usize]);
    }

    fn end_fade(&mut self, n: u32) {
        if self.fades.remove(&n).is_some() {
            self.fade_changed.push(self.keys[n as usize]);
        }
    }

    fn set_hidden(&mut self, n: u32, hidden: bool) {
        self.hid[n as usize] = hidden;
        let k = self.keys[n as usize];
        if hidden {
            self.hidden.insert(k);
        } else {
            self.hidden.remove(&k);
        }
    }

    /// Hide node `n` and every block under it that is not hidden yet, ending their fades. Returns the blocks hidden.
    fn hide_tree(&mut self, n: u32, flips: &mut Vec<Key>) {
        let mut stack = vec![n];
        while let Some(n) = stack.pop() {
            if !self.hid[n as usize] {
                self.set_hidden(n, true);
                flips.push(self.keys[n as usize]);
            }
            self.end_fade(n);
            stack.extend(self.children[n as usize].iter().copied().filter(|&c| c != NONE && !self.hid[c as usize]));
        }
    }

    /// Fades that ran their course: those that faded in are shown, those that faded out are hidden with every block
    /// under them. Returns the blocks hidden.
    fn retire(&mut self) -> Vec<Key> {
        if self.fades.is_empty() {
            return Vec::new();
        }
        let now = self.clock;
        let fade = self.fade_s.max(1e-6) as f64;
        let done: Vec<(u32, bool)> = self.fades.iter().filter(|(_, f)| if f.rising { f.t0 as f64 + (now - f.at) / fade >= 1.0 } else { f.t0 as f64 - (now - f.at) / fade <= 0.0 }).map(|(n, f)| (*n, f.rising)).collect();
        let mut flips = Vec::new();
        for (n, rising) in done {
            self.end_fade(n);
            if !rising {
                self.hide_tree(n, &mut flips);
            }
        }
        flips
    }

    // ---------------------------------------------------------------- the nodes

    /// The node of block `k`, made (with its parents, which the volume keeps) if there is none.
    fn node(&mut self, k: Key) -> u32 {
        if let Some(&n) = self.index.get(&k) {
            return n;
        }
        let parent = if k.0 > 0 { self.node(parent_key(k)) } else { NONE };
        let n = match self.free.pop() {
            Some(n) => {
                self.keys[n as usize] = k;
                self.parent[n as usize] = parent;
                self.children[n as usize] = [NONE; 8];
                self.own[n as usize] = f32::INFINITY;
                self.tree[n as usize] = f32::INFINITY;
                self.quads[n as usize] = 0;
                self.hid[n as usize] = false;
                self.stale_at[n as usize] = MEASURED;
                n
            }
            None => {
                self.keys.push(k);
                self.parent.push(parent);
                self.children.push([NONE; 8]);
                self.own.push(f32::INFINITY);
                self.tree.push(f32::INFINITY);
                self.quads.push(0);
                self.hid.push(false);
                self.stale_at.push(MEASURED);
                self.in_level.push(0);
                self.is_fresh.push(false);
                self.keys.len() as u32 - 1
            }
        };
        self.is_fresh[n as usize] = true;
        self.fresh.push(n);
        if parent != NONE {
            self.children[parent as usize][octant(k.1)] = n;
        }
        let l = k.0 as usize;
        if self.at_level.len() <= l {
            self.at_level.resize_with(l + 1, Vec::new);
        }
        self.in_level[n as usize] = self.at_level[l].len() as u32;
        self.at_level[l].push(n);
        self.index.insert(k, n);
        n
    }

    /// Forget node `n` (its block went). Its children, if any are left, lose their parent.
    fn remove(&mut self, n: u32) {
        let k = self.keys[n as usize];
        self.index.remove(&k);
        let p = self.parent[n as usize];
        if p != NONE {
            self.children[p as usize][octant(k.1)] = NONE;
        }
        for c in self.children[n as usize] {
            if c != NONE {
                self.parent[c as usize] = NONE;
            }
        }
        let l = k.0 as usize;
        let at = self.in_level[n as usize] as usize;
        let last = self.at_level[l].pop().expect("the node's level has it");
        if at < self.at_level[l].len() {
            self.at_level[l][at] = last;
            self.in_level[last as usize] = at as u32;
        }
        if self.stale_at[n as usize] != MEASURED {
            self.n_stale -= 1;
        }
        self.keys[n as usize] = (255, IVec3::ZERO);
        self.parent[n as usize] = NONE;
        self.children[n as usize] = [NONE; 8];
        self.stale_at[n as usize] = MEASURED;
        self.hid[n as usize] = false;
        self.is_fresh[n as usize] = false;
        self.free.push(n);
    }

    /// The blocks `dirty` changed (samples or existence): their detail is to be measured again, and a block that came
    /// under a hidden one is hidden with it.
    pub fn note(&mut self, vol: &LevelVolume, dirty: &[Key]) {
        self.note_at(vol, dirty, false);
    }

    /// [`Self::note`], with `measured` the detail measured now rather than later (for a big change landing: what
    /// would not show from here need never go up).
    pub fn note_at(&mut self, vol: &LevelVolume, dirty: &[Key], measured: bool) {
        self.syncs += 1;
        if dirty.is_empty() {
            return;
        }
        // What each costs to draw, measured at once (a cheap count of sign changes), and whether it is still there.
        let mut counted: Vec<(Key, u32, bool, f32)> = dirty
            .par_iter()
            .map(|&k| {
                let (l, bc) = (k.0 as usize, k.1);
                let exists = vol.exists(l, bc);
                let own = if measured && exists && l > 0 { vol.detail_of(l, bc) } else { f32::INFINITY };
                (k, vol.surface_quads(l, bc), exists, own)
            })
            .collect();
        // Coarsest first (a parent before its children), blocks that went last and finest first (a child before its
        // parent).
        counted.sort_by_key(|c| if c.2 { c.0.0 as i32 } else { 255 - c.0.0 as i32 });
        let syncs = self.syncs;
        let mut up: Vec<u32> = Vec::new();
        for &(k, q, exists, own) in &counted {
            if !exists {
                if let Some(&n) = self.index.get(&k) {
                    if k.0 == 0 {
                        self.base -= self.quads[n as usize] as u64;
                    }
                    self.set_hidden(n, false);
                    self.end_fade(n);
                    let p = self.parent[n as usize];
                    if p != NONE && k.0 > 1 {
                        up.push(p);
                    }
                    self.remove(n);
                }
                continue;
            }
            let n = self.node(k);
            let was = std::mem::replace(&mut self.quads[n as usize], q);
            if k.0 == 0 {
                self.base = self.base + q as u64 - was as u64;
                continue;
            }
            self.own[n as usize] = own;
            if own.is_finite() {
                if self.stale_at[n as usize] != MEASURED {
                    self.n_stale -= 1;
                    self.stale_at[n as usize] = MEASURED;
                }
            } else {
                if self.stale_at[n as usize] == MEASURED {
                    self.n_stale += 1;
                    self.stale.push(n);
                }
                self.stale_at[n as usize] = syncs;
            }
            let p = self.parent[n as usize];
            if p != NONE && self.hid[p as usize] && !self.hid[n as usize] {
                self.set_hidden(n, true);
            }
            // Already unbounded (still to be measured since an earlier change): so is everything above it, unless
            // measured now.
            if own.is_finite() || self.tree[n as usize] != f32::INFINITY {
                up.push(n);
            }
            // A block changed shows as it is at once: no fade lags behind the work.
            if !self.fades.is_empty() {
                self.end_fade(n);
            }
        }
        self.retree(up);
        self.changed = true;
    }

    /// Measure the detail of up to `most` changed blocks. Returns how many were measured.
    pub fn refresh(&mut self, vol: &LevelVolume, most: usize) -> usize {
        if self.stale.is_empty() {
            return 0;
        }
        let now = self.syncs;
        let mut take: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < self.stale.len() && take.len() < most {
            let n = self.stale[i];
            let at = self.stale_at[n as usize];
            if at == MEASURED {
                // Measured or gone since it was listed.
                self.stale.swap_remove(i);
            } else if at < now || most == usize::MAX {
                take.push(n);
                self.stale.swap_remove(i);
            } else {
                i += 1;
            }
        }
        if take.is_empty() {
            return 0;
        }
        let keys = &self.keys;
        let measured: Vec<f32> = take.par_iter().map(|&n| vol.detail_of(keys[n as usize].0 as usize, keys[n as usize].1)).collect();
        for (&n, &e) in take.iter().zip(&measured) {
            self.stale_at[n as usize] = MEASURED;
            self.n_stale -= 1;
            self.own[n as usize] = e;
        }
        let count = take.len();
        self.retree(take);
        self.changed = true;
        count
    }

    /// [`Self::refresh`] in batches until about `ms` milliseconds are spent (always at least one batch).
    pub fn refresh_within(&mut self, vol: &LevelVolume, ms: f64) {
        let t = std::time::Instant::now();
        loop {
            if self.refresh(vol, MEASURE_BATCH) == 0 || t.elapsed().as_secs_f64() * 1e3 >= ms {
                return;
            }
        }
    }

    /// Work out the tree detail of `nodes` and every block above them, finest first.
    fn retree(&mut self, nodes: Vec<u32>) {
        let levels = self.at_level.len();
        let mut per: Vec<Vec<u32>> = vec![Vec::new(); levels];
        for n in nodes {
            let l = self.keys[n as usize].0 as usize;
            if l < levels {
                per[l].push(n);
            }
        }
        for l in (1..levels).rev() {
            let mut here = std::mem::take(&mut per[l]);
            here.sort_unstable();
            here.dedup();
            for n in here {
                let own = self.own[n as usize];
                // Unbounded (to be measured): so is the tree, whatever is below.
                let t = if own.is_infinite() { own } else { own + self.children[n as usize].iter().filter(|&&c| c != NONE).map(|&c| self.tree[c as usize]).fold(0.0f32, f32::max) };
                if self.tree[n as usize] == t {
                    continue;
                }
                self.tree[n as usize] = t;
                let p = self.parent[n as usize];
                if p != NONE && l > 1 {
                    per[l - 1].push(p);
                }
            }
        }
    }

    /// Within a budget: of the blocks whose detail on screen is more than `keep` pixels, sorted by it, the pixels at
    /// which their costs first add up to more than each of `rooms` (0 if they never do). A histogram over the pixels
    /// finds the bin, and only that bin is sorted. Blocks still to be measured (unbounded detail) are left out: they
    /// show whatever the budget says, as work under way must, and count against it once measured.
    fn cuts(&self, keep: f32, rooms: [f32; 2]) -> [f32; 2] {
        const BINS: usize = 1 << 16;
        let (px, quads, keys) = (&self.px, &self.quads, &self.keys);
        let n = px.len();
        let bin = |p: f32| -> usize { (p.to_bits() >> 16) as usize };
        let cost = |i: usize| quads[i] as f32 * ADDS * 2.0;
        let counts = |i: usize| keys[i].0 != 0 && keys[i].0 != 255 && px[i] > keep && px[i].is_finite();
        let hist: Vec<f64> = (0..n)
            .into_par_iter()
            .with_min_len(1 << 15)
            .fold(
                || vec![0.0f64; BINS],
                |mut h, i| {
                    if counts(i) {
                        h[bin(px[i])] += cost(i) as f64;
                    }
                    h
                },
            )
            .reduce(
                || vec![0.0f64; BINS],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += y;
                    }
                    a
                },
            );
        let mut out = [0.0f32; 2];
        for (r, &room) in rooms.iter().enumerate() {
            let mut sum = 0.0f64;
            let mut found = None;
            for b in (0..BINS).rev() {
                if sum + hist[b] > room as f64 {
                    found = Some(b);
                    break;
                }
                sum += hist[b];
            }
            let Some(b) = found else { continue };
            let mut members: Vec<(f32, f32)> = (0..n).into_par_iter().with_min_len(1 << 15).filter(|&i| counts(i) && bin(px[i]) == b).map(|i| (px[i], cost(i))).collect();
            members.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let mut s = sum as f32;
            let mut at = members.last().map_or(0.0, |m| m.0);
            for (p, c) in &members {
                s += c;
                if s > room {
                    at = *p;
                    break;
                }
            }
            out[r] = at;
        }
        out
    }

    /// Choose what to show from `eye` (with `focal_px` pixels per unit at distance one). Returns the blocks shown or
    /// hidden since the last choice.
    pub fn choose(&mut self, vol: &LevelVolume, eye: Vec3, focal_px: f32) -> Vec<Key> {
        let t = std::time::Instant::now();
        if !self.on {
            let flips: Vec<Key> = self.hidden.drain().collect();
            for h in self.hid.iter_mut() {
                *h = false;
            }
            let fading: Vec<u32> = self.fades.keys().copied().collect();
            for n in fading {
                self.end_fade(n);
            }
            for n in self.fresh.drain(..) {
                self.is_fresh[n as usize] = false;
            }
            self.last = None;
            self.stats = ViewStats { shown: 0, hidden: 0, stale: self.n_stale, flips: flips.len(), ms: t.elapsed().as_secs_f64() * 1e3, triangles: 0, tau_used: 0.0 };
            return flips;
        }
        let mut flips = self.retire();
        let now = (eye, focal_px, self.tau_px, self.budget);
        if !self.changed && self.last == Some(now) {
            self.stats.flips = flips.len();
            self.stats.hidden = self.hidden.len();
            return flips;
        }
        self.last = Some(now);
        self.changed = false;
        let origin = vol.origin();
        let voxels: Vec<f32> = (0..self.at_level.len()).map(|l| vol.voxel(l)).collect();
        // Each block's detail in pixels, seen from the eye.
        let n = self.keys.len();
        self.px.clear();
        let (keys, tree) = (&self.keys, &self.tree);
        self.px.par_extend((0..n).into_par_iter().map(|i| {
            let (l, bc) = (keys[i].0 as usize, keys[i].1);
            if l == 0 || l >= voxels.len() {
                return 0.0;
            }
            let vx = voxels[l];
            let lo = origin + (bc * B).as_vec3() * vx;
            let hi = lo + Vec3::splat(B as f32 * vx);
            let d = (eye.clamp(lo, hi) - eye).length();
            let e = tree[i];
            if d <= 0.0 { f32::INFINITY } else { e * focal_px / d }
        }));
        // Within a budget: the pixels at which the blocks with the most detail on screen first add up to it. A block's
        // detail on screen is never less than any finer block's under it (its tree detail holds theirs, and it is no
        // further away), so every threshold shows a set that nests.
        let (mut tau, mut keep) = (self.tau_px, self.tau_px * KEEP);
        if let Some(budget) = self.budget {
            let room = budget as f32 - self.base as f32 * 2.0;
            let [a, b] = self.cuts(keep, [room, room + (OVER - 1.0) * budget as f32]);
            tau = tau.max(a);
            keep = keep.max(b);
        }
        let mut triangles = self.base as f32 * 2.0;
        let fading = self.fade_s > 0.0;
        self.shown_now.clear();
        self.shown_now.resize(n, false);
        let mut shown = self.at_level.first().map_or(0, |l| l.len());
        for &n0 in self.at_level.first().map_or(&[][..], |l| &l[..]) {
            self.shown_now[n0 as usize] = true;
        }
        // Level 0 always shows; level by level down from there. A block whose parent is not shown by this choice
        // (hidden, or going) goes with it, with everything under it.
        for l in 1..self.at_level.len() {
            let nodes = std::mem::take(&mut self.at_level[l]);
            let (parent, shown_now, hid, px, tree, fades, quads) = (&self.parent, &self.shown_now, &self.hid, &self.px, &self.tree, &self.fades, &self.quads);
            let decided: Vec<(u8, bool, f32)> = nodes
                .par_iter()
                .map(|&n| {
                    let i = n as usize;
                    let p = parent[i];
                    if p == NONE || !shown_now[p as usize] {
                        return (if hid[i] { KEEP_AS_IS } else { HIDE }, false, 0.0);
                    }
                    let was_hidden = hid[i];
                    let show = if was_hidden { px[i] > tau } else { px[i] > keep };
                    if !show {
                        return (if was_hidden { KEEP_AS_IS } else { HIDE }, false, 0.0);
                    }
                    let cost = quads[i] as f32 * ADDS * 2.0;
                    // Fading in from the level below, unless its detail is still to be measured (work under way
                    // shows at once).
                    let measured = tree[i].is_finite();
                    if was_hidden {
                        (if measured { SHOW_MEASURED } else { SHOW }, true, cost)
                    } else if fades.get(&n).is_some_and(|f| !f.rising) {
                        // Wanted again while fading out: back in from where it got to.
                        (if measured { REFADE_MEASURED } else { REFADE }, true, cost)
                    } else {
                        (KEEP_AS_IS, true, cost)
                    }
                })
                .collect();
            for (&n, &(act, show, cost)) in nodes.iter().zip(&decided) {
                self.shown_now[n as usize] = show;
                if show {
                    shown += 1;
                    triangles += cost;
                }
                match act {
                    SHOW | SHOW_MEASURED => {
                        self.set_hidden(n, false);
                        flips.push(self.keys[n as usize]);
                        if fading && act == SHOW_MEASURED {
                            self.set_fade(n, 0.0, true);
                        }
                    }
                    REFADE_MEASURED => {
                        let from = self.fade_now(n);
                        self.set_fade(n, from, true);
                    }
                    REFADE => self.end_fade(n),
                    // A hidden block takes its finer blocks with it. Fading, they fade out first (from where they
                    // are), and are hidden once they have; one that just came goes at once.
                    HIDE => {
                        if fading && !self.is_fresh[n as usize] {
                            if self.fades.get(&n).is_none_or(|f| f.rising) {
                                let from = self.fade_now(n);
                                self.set_fade(n, from, false);
                            }
                        } else {
                            self.set_hidden(n, true);
                            flips.push(self.keys[n as usize]);
                        }
                    }
                    _ => {}
                }
            }
            self.at_level[l] = nodes;
        }
        for n in self.fresh.drain(..) {
            self.is_fresh[n as usize] = false;
        }
        self.stats = ViewStats { shown, hidden: self.hidden.len(), stale: self.n_stale, flips: flips.len(), ms: t.elapsed().as_secs_f64() * 1e3, triangles: triangles as u64, tau_used: tau };
        flips
    }
}

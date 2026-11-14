//! Mesh layers in the volume playground: the layers (`sculpt_core::mesh_layers::MeshLayers`, a mesh taken from the
//! volume with per-vertex offsets on it), what the volume changed since they were last refitted, the one undo history
//! of volume steps, layer strokes and layer settings, and what the GPU rows need redrawn.
//!
//! The volume keeps the big form; the layers ride on its surface. After a volume stroke the real volume (never the
//! copy strokes run ahead on) is refitted off the main thread and swapped in a frame or so later. What a swap could not
//! carry by name it hands back as a patch, filed under the volume state the layers left; whenever a later swap (an
//! undo, a redo, or a refit that failed and then went on) brings the layers back to that state, it goes back exactly.
//! So every state shown after a swap comes back bit for bit, however the layers got there.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use sculpt_core::generator::level_volume::{Key, LevelVolume};
use sculpt_core::mesh_layers::{MeshLayers, Patch, SwapStats};
use sculpt_core::{DirtySet, LayerId, LayerMeta};

/// What the GPU rows need redrawn since the last take.
#[derive(Default, Debug, Clone)]
pub struct RowsDirty {
    /// Rows of every block are stale (first layer made, file opened, layer removed, opacity/blend/visibility/mask
    /// change touching everything, undo of a volume step).
    pub all: bool,
    /// Blocks whose rows changed (swap `SwapStats.blocks.changed`, or blocks touched by a layer dab / restore).
    pub blocks: Vec<Key>,
    /// Blocks gone from the mesh (swap `SwapStats.blocks.removed`).
    pub removed: Vec<Key>,
    /// No visible mesh layer remains: clear every row and turn `layers_shown` off.
    pub none: bool,
    /// Layer dabs, mask and restore changes: the live document's BVH leaves whose vertices moved (much faster than
    /// `blocks`). Only valid for the document as it is at the take; leaves pending when a swap lands are turned into
    /// `blocks` first.
    pub leaves: Vec<u32>,
}

impl RowsDirty {
    pub fn is_empty(&self) -> bool {
        !self.all && !self.none && self.blocks.is_empty() && self.removed.is_empty() && self.leaves.is_empty()
    }
}

/// One step of the history (and, on the redo side, the step to do again).
pub(crate) enum Entry {
    /// A step of the volume's own history, named by `LevelVolume::last_id` (on the redo side `redo_id`): only a check
    /// that the two histories still agree. `pre` and `post`: the volume states (this stack's own names) before and
    /// after it.
    Volume { id: u64, pre: u64, post: u64 },
    /// A layer or mask stroke (or a mask fill): restoring the patch undoes it, and gives the patch that redoes it. A
    /// fill also holds its mask's fill to put back (the panel shows it), `None` inside when none was set.
    Layer { patch: Patch, fill: Option<(String, Option<f32>)> },
    /// A layer's settings (name, opacity, visibility, blend, masks) before and after.
    Meta { id: LayerId, before: LayerMeta, after: LayerMeta },
}

/// What the last swap, and the last refit's start, cost on the main thread.
#[derive(Clone, Debug, Default)]
pub struct MeshTimes {
    pub refit_start_ms: f64,
    pub swap: Option<SwapStats>,
    /// The frame's whole share of polling and swapping, last time a swap came in.
    pub swap_frame_ms: f64,
    pub swaps: usize,
    /// Waits for a refit inside an undo, a redo or before a layer stroke.
    pub wait_ms: f64,
}

pub struct MeshStack {
    pub layers: MeshLayers,
    /// Volume blocks changed since the last refit started.
    pub(crate) dirty: HashSet<Key>,
    /// Blocks a refit done at once (undo, redo) already took: not counted again when next seen still waiting for the
    /// GPU (only the next look; any stroke clears it).
    pub(crate) skip: HashSet<Key>,
    pub(crate) history: Vec<Entry>,
    pub(crate) redo: Vec<Entry>,
    /// What each swap could not carry, filed under the volume state the layers left (newest last): put back when
    /// the layers next arrive at that state, however they get there. A patch belongs to the state left, not to a
    /// step: refits that fail, or that cover several steps, leave the layers at states other than the step's ends.
    lost: HashMap<u64, Vec<Patch>>,
    /// Per state the layers arrived at: the state they came from (by the last swap that brought them there).
    came_from: HashMap<u64, u64>,
    /// The volume's state, the state the layers are at (they lag behind while a refit runs or after one failed), and
    /// the state the refit under way goes to.
    vol_state: u64,
    doc_state: u64,
    job_state: Option<u64>,
    next_state: u64,
    /// The blocks of the refit under way, and of the refits that failed since one last succeeded: seen again with
    /// nothing new (the GPU not having taken them yet), they start no refit.
    job_keys: Vec<Key>,
    failed_keys: HashSet<Key>,
    rows: RowsDirty,
    shown: bool,
    /// Per leaf of the document: the blocks its vertices sit in (worked out when first needed), for one topology.
    leaf_blocks: (u64, Vec<Option<Box<[Key]>>>),
    /// Per layer: the mask strokes paint (index into its mask stack), if one is picked.
    pub(crate) active_mask: HashMap<LayerId, Option<usize>>,
    /// Per painted mask's channel: what it was last filled with.
    pub(crate) fills: HashMap<String, f32>,
    /// A change going on over frames (a slider dragged): the layer as it began.
    pub(crate) open: Option<(LayerId, LayerMeta)>,
    pub times: MeshTimes,
    /// The last refit failed (the layers could not follow the volume), with its error: until one succeeds.
    pub failure: Option<String>,
    /// Something for the status line, once (an undo that could not be done).
    pub note: Option<String>,
}

/// Take the mesh layers' chunk off the end of a `.svol` file's bytes, if it has one (see `VolumeMode::write_to`):
/// returns its body ("MLAY" and its length left out), the bytes before it left in `bytes`.
pub fn split_mlay(bytes: &mut Vec<u8>) -> std::io::Result<Option<Vec<u8>>> {
    let n = bytes.len();
    if n < 24 || &bytes[n - 4..] != b"MLAY" {
        return Ok(None);
    }
    let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "a damaged mesh layers chunk");
    let back = usize::try_from(u64::from_le_bytes(bytes[n - 12..n - 4].try_into().unwrap())).map_err(|_| bad())?;
    // (A chunk is at least its two marks and two lengths.)
    if back < 24 {
        return Err(bad());
    }
    let start = n.checked_sub(back).ok_or_else(bad)?;
    if &bytes[start..start + 4] != b"MLAY" {
        return Err(bad());
    }
    let len = usize::try_from(u64::from_le_bytes(bytes[start + 4..start + 12].try_into().unwrap())).map_err(|_| bad())?;
    if len.checked_add(24) != Some(back) {
        return Err(bad());
    }
    let body = bytes[start + 12..start + 12 + len].to_vec();
    bytes.truncate(start);
    Ok(Some(body))
}

/// The note when the layers' history and the volume's no longer agree.
pub(crate) const DISAGREE: &str = "The layers' history no longer matched the volume's: it starts afresh";

impl MeshStack {
    /// The mesh layers as `VolumeMode::write_to` wrote them (the chunk's body), on a fresh mesh taken from `vol`.
    pub fn read_from(body: &[u8], vol: &LevelVolume) -> std::io::Result<MeshStack> {
        let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
        let mut cur = std::io::Cursor::new(body);
        let (layers, lost) = MeshLayers::read_from(&mut cur, vol).map_err(|e| err(e.to_string()))?;
        let at = cur.position() as usize;
        let mut stack = MeshStack::new(layers, HashSet::new());
        if lost > 0 {
            // (Told by the opener, with anything else left out.)
            stack.note = Some(format!("the layer data of {lost} vertices (no vertex of their name to go back to)"));
        }
        // (The notes are optional: damaged, they are left out.)
        if body.len() >= at + 8 {
            let len = u64::from_le_bytes(body[at..at + 8].try_into().unwrap());
            let end = usize::try_from(len).ok().and_then(|len| (at + 8).checked_add(len));
            let fills = end
                .and_then(|end| body.get(at + 8..end))
                .and_then(|json| serde_json::from_slice::<serde_json::Value>(json).ok())
                .and_then(|v| v.get("fills").and_then(|f| serde_json::from_value::<HashMap<String, f32>>(f.clone()).ok()));
            match fills {
                Some(f) => stack.fills = f,
                None => eprintln!("mesh layers: the mask fills could not be read; left out"),
            }
        }
        Ok(stack)
    }

    pub fn new(mut layers: MeshLayers, dirty: HashSet<Key>) -> MeshStack {
        // What a swap replaces is freed when the volume mode says (between strokes): freed at once it slowed the dabs
        // of a stroke under way.
        layers.set_hold_old(true);
        MeshStack {
            layers,
            dirty,
            skip: HashSet::new(),
            history: Vec::new(),
            redo: Vec::new(),
            lost: HashMap::new(),
            came_from: HashMap::new(),
            vol_state: 0,
            doc_state: 0,
            job_state: None,
            next_state: 1,
            job_keys: Vec::new(),
            failed_keys: HashSet::new(),
            rows: RowsDirty { all: true, ..Default::default() },
            shown: false,
            leaf_blocks: (0, Vec::new()),
            active_mask: HashMap::new(),
            fills: HashMap::new(),
            open: None,
            times: MeshTimes::default(),
            failure: None,
            note: None,
        }
    }

    /// Volume blocks changed (as the volume's dirty set holds them, before whoever drains it takes them).
    pub fn tap(&mut self, keys: impl IntoIterator<Item = Key>) {
        let skip = std::mem::take(&mut self.skip);
        for k in keys {
            if !skip.contains(&k) {
                self.dirty.insert(k);
            }
        }
    }

    /// A stroke begins: whatever the volume shows next is new.
    pub fn forget_skip(&mut self) {
        self.skip.clear();
    }

    /// The volume kept a step named `id`: it gets an entry, and what there was to redo goes.
    pub fn note_step(&mut self, id: u64) {
        let (pre, post) = (self.vol_state, self.new_state());
        self.history.push(Entry::Volume { id, pre, post });
        self.vol_state = post;
        // A new step: the blocks a failed refit had are worth another try with it.
        self.dirty.extend(self.failed_keys.drain());
        self.redo.clear();
        self.trim();
    }

    /// Something else was done (a layer stroke, a setting): it goes on the history, and what there was to redo goes.
    pub(crate) fn push(&mut self, e: Entry) {
        self.history.push(e);
        self.redo.clear();
        self.trim();
    }

    /// The oldest step past 512 goes, and the patches of states no step reaches any more.
    fn trim(&mut self) {
        if self.history.len() > 512 {
            self.history.remove(0);
        }
        let mut keep: HashSet<u64> = [self.vol_state, self.doc_state].into_iter().collect();
        keep.extend(self.job_state);
        for e in self.history.iter().chain(&self.redo) {
            if let Entry::Volume { pre, post, .. } = e {
                keep.extend([*pre, *post]);
            }
        }
        // A state no step reaches any more: what its patches hold of vertices the layers carried there by name from the
        // state they came from is that state's data too (a refit that skipped a failed state can bring the layers
        // back there straight from a later one), so they go to the nearest such state still reached, newest last.
        let came = &self.came_from;
        let kept_from = |s: u64| -> Option<u64> {
            let mut at = s;
            for _ in 0..=came.len() {
                let c = *came.get(&at)?;
                if keep.contains(&c) {
                    return Some(c);
                }
                at = c;
            }
            None
        };
        let gone: Vec<(u64, Option<u64>)> = self.lost.keys().filter(|s| !keep.contains(s)).map(|&s| (s, kept_from(s))).collect();
        let from: Vec<(u64, Option<u64>)> = came.keys().filter(|s| keep.contains(s)).map(|&s| (s, kept_from(s))).collect();
        for (s, to) in gone {
            let patches = self.lost.remove(&s).unwrap_or_default();
            if let Some(c) = to {
                self.lost.entry(c).or_default().extend(patches);
            }
        }
        // (Each state still reached keeps where it came from, past the states gone.)
        self.came_from = from.into_iter().filter_map(|(s, c)| Some((s, c?))).collect();
    }

    fn new_state(&mut self) -> u64 {
        let s = self.next_state;
        self.next_state += 1;
        s
    }

    /// Drop the history (a layer removed: its strokes can no longer be put back).
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.redo.clear();
        self.trim();
    }

    /// Whether a refit is under way or waiting (blocks changed since the last one began).
    pub fn pending(&self) -> bool {
        self.layers.busy() || !self.dirty.is_empty()
    }

    /// Start refitting to `vol` (the real volume, at rest) if blocks changed and no refit runs. Returns whether one
    /// started.
    pub fn start_refit(&mut self, vol: &LevelVolume) -> bool {
        if self.dirty.is_empty() || self.layers.busy() {
            return false;
        }
        if self.dirty.iter().all(|k| self.failed_keys.contains(k)) {
            // Only blocks a failed refit had (still waiting for the GPU): it would fail again. They wait in the layers.
            self.dirty.clear();
            return false;
        }
        let mut keys: Vec<Key> = self.dirty.drain().collect();
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        let b = MeshLayers::changed_box(vol, &keys);
        self.times.refit_start_ms = self.layers.refit_async(vol, &keys, b);
        self.started(keys);
        true
    }

    /// A refit was asked for with `keys`: if it runs, it goes to the volume's state now.
    fn started(&mut self, keys: Vec<Key>) {
        if self.layers.busy() {
            self.job_state = Some(self.vol_state);
            self.job_keys = keys;
        }
    }

    /// Take a finished refit in (with `wait`, the one under way): what it could not carry is filed under the state
    /// the layers left, and what was filed under the state they arrive at goes back. Returns whether a swap came in.
    pub fn poll(&mut self, vol: &LevelVolume, wait: bool) -> bool {
        if !self.layers.busy() {
            return false;
        }
        let t = Instant::now();
        // What the layers changed since the rows were last told is told now: the swap's own list holds only what it
        // changes from here.
        self.collect_doc();
        if !self.rows.leaves.is_empty() {
            // Leaves name the document the swap replaces: as blocks they still mean something after it.
            let ls = std::mem::take(&mut self.rows.leaves);
            let blocks = self.leaf_blocks(&ls);
            self.rows.blocks.extend(blocks);
        }
        let r = if wait { self.layers.wait(vol) } else { self.layers.poll(vol) };
        let Some(r) = r else { return false };
        let to = self.job_state.take().unwrap_or(self.vol_state);
        let swapped = match r {
            Ok(st) if !st.discarded => {
                let _ = self.layers.doc.take_geometry_dirty();
                self.rows.blocks.extend_from_slice(&st.blocks.changed);
                self.rows.removed.extend_from_slice(&st.blocks.removed);
                self.times.swap = Some(st);
                self.times.swaps += 1;
                self.failure = None;
                true
            }
            Ok(_) => false,
            Err(e) => {
                // (Once: the job is not run again until the volume changes.)
                eprintln!("mesh layers: refit failed: {e}");
                self.failure = Some(e.to_string());
                // The layers stay at the state they are at.
                self.failed_keys.extend(self.job_keys.drain(..));
                false
            }
        };
        let patches = self.layers.take_patches();
        if swapped {
            let left = self.doc_state;
            let fwd: Vec<Patch> = patches.into_iter().filter(|p| !p.is_empty()).collect();
            if !fwd.is_empty() {
                self.lost.entry(left).or_default().extend(fwd);
            }
            // Newest first: where two hold a vertex, the older (nearer the state as it was) wins.
            for p in self.lost.remove(&to).unwrap_or_default().iter().rev() {
                if let Err(e) = self.layers.restore(p) {
                    eprintln!("mesh layers: restore failed: {e}");
                }
            }
            if to != left {
                self.came_from.insert(to, left);
            }
            self.doc_state = to;
            self.failed_keys.clear();
            self.job_keys.clear();
            self.trim();
            self.times.swap_frame_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        if self.layers.busy() {
            // A job started again inside (its base changed meanwhile): it goes to the volume's state now.
            self.job_state = Some(self.vol_state);
        }
        swapped
    }

    /// Every change the volume made so far, refitted and swapped in, now.
    pub fn finish(&mut self, vol: &LevelVolume) {
        let t = Instant::now();
        let mut waited = false;
        loop {
            if self.layers.busy() {
                self.poll(vol, true);
                waited = true;
            } else if !self.start_refit(vol) {
                break;
            }
        }
        if waited {
            self.times.wait_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// After `vol`'s undo or redo of a step: start refitting to it, with every block it has not sent yet. (No refit
    /// runs: [`Self::finish`] came first.)
    fn start_landing(&mut self, vol: &LevelVolume) {
        let t = Instant::now();
        let waiting = vol.dirty_keys();
        let mut keys: Vec<Key> = waiting.iter().copied().chain(self.dirty.drain()).collect();
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        keys.dedup();
        let b = MeshLayers::changed_box(vol, &keys);
        self.layers.refit_async(vol, &keys, b);
        // (Not started when the last refit failed and the step changed no block: the layers stay where they are.)
        self.started(keys);
        // Those blocks still wait for the GPU: the next look at the volume's changes leaves them out.
        self.skip = waiting.into_iter().collect();
        self.times.refit_start_ms = t.elapsed().as_secs_f64() * 1e3;
    }

    /// Undo the newest step: a layer stroke or setting, or a volume step (`vol.undo`, a refit started, its patches
    /// back once it swaps in: until then the volume shows undone and the layers as they were where it did not change). With no entry left, the volume's own next step if it has one. Returns whether anything was undone.
    pub fn undo(&mut self, vol: &mut LevelVolume) -> bool {
        self.close_open();
        self.finish(vol);
        loop {
            match self.history.pop() {
                None => {
                    // The volume's own older step: it gets an entry, from a state of its own.
                    if !vol.undo() {
                        return false;
                    }
                    let (pre, post) = (self.new_state(), self.vol_state);
                    if let Some(id) = vol.redo_id() {
                        self.redo.push(Entry::Volume { id, pre, post });
                    }
                    self.vol_state = pre;
                    self.start_landing(vol);
                    return true;
                }
                Some(Entry::Layer { patch, fill }) => {
                    return match self.layers.restore_swap(&patch) {
                        Ok(inv) => {
                            let fill = self.swap_fill(fill);
                            self.redo.push(Entry::Layer { patch: inv, fill });
                            true
                        }
                        Err(e) => self.drop_history("undo", e),
                    };
                }
                Some(Entry::Meta { id, before, after }) => {
                    return match self.layers.doc.set_layer_meta(id, before.clone(), false) {
                        Ok(()) => {
                            self.redo.push(Entry::Meta { id, before, after });
                            true
                        }
                        Err(e) => self.drop_history("undo", e),
                    };
                }
                Some(Entry::Volume { id, pre, post }) => {
                    if vol.last_id() != Some(id) {
                        // The two histories no longer agree: the older entries can only be wrong.
                        self.history.clear();
                        self.trim();
                        self.note = Some(DISAGREE.into());
                        continue;
                    }
                    vol.undo();
                    // On the redo side at once (its refit may land later, or fail): anything done meanwhile comes after it.
                    if let Some(id) = vol.redo_id() {
                        self.redo.push(Entry::Volume { id, pre, post });
                    }
                    self.vol_state = pre;
                    self.start_landing(vol);
                    return true;
                }
            }
        }
    }

    /// Redo what was last undone. Returns whether anything was.
    pub fn redo(&mut self, vol: &mut LevelVolume) -> bool {
        self.close_open();
        self.finish(vol);
        match self.redo.pop() {
            None => false,
            Some(Entry::Layer { patch, fill }) => match self.layers.restore_swap(&patch) {
                Ok(inv) => {
                    let fill = self.swap_fill(fill);
                    self.history.push(Entry::Layer { patch: inv, fill });
                    true
                }
                Err(e) => self.drop_history("redo", e),
            },
            Some(Entry::Meta { id, before, after }) => match self.layers.doc.set_layer_meta(id, after.clone(), false) {
                Ok(()) => {
                    self.history.push(Entry::Meta { id, before, after });
                    true
                }
                Err(e) => self.drop_history("redo", e),
            },
            Some(Entry::Volume { id, pre, post }) => {
                if vol.redo_id() != Some(id) {
                    self.redo.clear();
                    self.trim();
                    self.note = Some(DISAGREE.into());
                    return false;
                }
                vol.redo();
                // On the history at once, under whatever comes while its refit runs (a stroke begun before it lands).
                if let Some(id) = vol.last_id() {
                    self.history.push(Entry::Volume { id, pre, post });
                }
                self.vol_state = post;
                self.trim();
                self.start_landing(vol);
                true
            }
        }
    }

    /// An undo or redo of a layer step that could not be done (nothing of it was): the history goes, as when the two
    /// histories disagree (keeping the step would block every older one behind it), and the note says why. Returns
    /// false: nothing was undone.
    fn drop_history(&mut self, what: &str, e: impl std::fmt::Display) -> bool {
        eprintln!("mesh layers: {what} failed: {e}");
        self.clear_history();
        self.note = Some(format!("Could not {what} on the layers ({e}): their history starts afresh"));
        false
    }

    /// Put back the fill a mask fill's step holds (`None` inside: none was set), giving what it replaced.
    fn swap_fill(&mut self, fill: Option<(String, Option<f32>)>) -> Option<(String, Option<f32>)> {
        let (name, to) = fill?;
        let was = match to {
            Some(f) => self.fills.insert(name.clone(), f),
            None => self.fills.remove(&name),
        };
        Some((name, was))
    }

    /// A slider drag still open: kept as one step.
    pub(crate) fn close_open(&mut self) {
        if let Some((id, before)) = self.open.take()
            && let Some(after) = self.layers.doc.layer(id).map(|l| l.meta())
            && after != before
        {
            self.push(Entry::Meta { id, before, after });
        }
    }

    // ------------------------------------------------------------------------------------------------- GPU rows

    /// Whether a visible mesh layer exists.
    pub fn layers_shown(&self) -> bool {
        self.layers.doc.layers().iter().any(|l| l.visible)
    }

    /// What the rows need redrawn since the last call.
    pub fn take_rows_dirty(&mut self) -> RowsDirty {
        self.collect_doc();
        let shown = self.layers_shown();
        if shown != self.shown {
            if shown {
                self.rows.all = true;
            } else {
                self.rows.none = true;
            }
            self.shown = shown;
        }
        let mut out = std::mem::take(&mut self.rows);
        if !shown {
            out.all = false;
            out.blocks.clear();
            out.removed.clear();
            out.leaves.clear();
        } else if out.all {
            out.blocks.clear();
            out.removed.clear();
            out.leaves.clear();
        } else {
            out.leaves.sort_unstable();
            out.leaves.dedup();
            let order = |k: &Key| (k.0, k.1.z, k.1.y, k.1.x);
            out.blocks.sort_unstable_by_key(order);
            out.blocks.dedup();
            out.removed.sort_unstable_by_key(order);
            out.removed.dedup();
            // A block gone and back again in one take is listed as there.
            if !out.removed.is_empty() {
                let there: HashSet<Key> = self.layers.blocks().iter().map(|b| b.0).collect();
                out.removed.retain(|k| !there.contains(k));
                out.blocks.retain(|k| there.contains(k));
            }
        }
        out
    }

    /// Count every row as stale.
    pub fn rows_all(&mut self) {
        self.rows.all = true;
    }

    /// The document's changed leaves, as blocks, into the rows to redraw.
    pub(crate) fn collect_doc(&mut self) {
        match self.layers.doc.take_geometry_dirty() {
            DirtySet::None => {}
            DirtySet::All => self.rows.all = true,
            DirtySet::Leaves(ls) => {
                if self.rows.all {
                    return;
                }
                self.rows.leaves.extend_from_slice(&ls);
            }
        }
    }

    /// The blocks the vertices of `leaves` sit in.
    pub(crate) fn leaf_blocks(&mut self, leaves: &[u32]) -> Vec<Key> {
        let doc = &self.layers.doc;
        let topo = doc.topology_id();
        let n = doc.bvh().leaves.len();
        if self.leaf_blocks.0 != topo || self.leaf_blocks.1.len() != n {
            self.leaf_blocks = (topo, vec![None; n]);
        }
        let i2c = self.layers.internal_to_canonical();
        let blocks = self.layers.blocks();
        let mut out = Vec::new();
        for &l in leaves {
            let slot = &mut self.leaf_blocks.1[l as usize];
            if slot.is_none() {
                let mut ks: Vec<Key> = Vec::new();
                let mut last = usize::MAX;
                for v in doc.bvh().leaves[l as usize].owned_range() {
                    let b = self.layers.block_of(i2c[v] as usize);
                    if b != last {
                        ks.push(blocks[b].0);
                        last = b;
                    }
                }
                ks.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
                ks.dedup();
                *slot = Some(ks.into_boxed_slice());
            }
            out.extend_from_slice(slot.as_ref().unwrap());
        }
        out
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::split_mlay;

    /// `before`, then a mesh layers chunk holding `body` with the given offset back and length (as written: the
    /// body's length plus 24, and its length).
    fn file(before: &[u8], body: &[u8], back: u64, len: u64) -> Vec<u8> {
        let mut out = before.to_vec();
        out.extend_from_slice(b"MLAY");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(&back.to_le_bytes());
        out.extend_from_slice(b"MLAY");
        out
    }

    #[test]
    fn a_whole_chunk_comes_off() {
        let body = b"layers and notes";
        let mut bytes = file(b"the volume", body, body.len() as u64 + 24, body.len() as u64);
        assert_eq!(split_mlay(&mut bytes).unwrap().as_deref(), Some(&body[..]));
        assert_eq!(bytes, b"the volume");
        let mut plain = b"a volume with no chunk at all, long enough".to_vec();
        assert!(split_mlay(&mut plain).unwrap().is_none());
    }

    /// A damaged offset or length is an error, never a panic or an overflow.
    #[test]
    fn a_damaged_chunk_is_an_error() {
        let body = b"layers and notes";
        let n = body.len() as u64;
        let total = file(b"the volume", body, n + 24, n).len() as u64;
        for (back, len) in [(0, n), (5, n), (12, n), (23, n), (total + 1, n), (u64::MAX, n), (n + 24, u64::MAX), (n + 24, u64::MAX - 11), (n + 24, n + 1), (n + 24, 0)] {
            let mut bytes = file(b"the volume", body, back, len);
            let r = split_mlay(&mut bytes);
            assert!(r.as_ref().is_err_and(|e| e.kind() == std::io::ErrorKind::InvalidData), "back {back}, len {len}: {r:?}");
        }
    }
}

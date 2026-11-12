//! Mesh layers in the volume playground: the layers (`sculpt_core::mesh_layers::MeshLayers`, a mesh taken from the
//! volume with per-vertex offsets on it), what the volume changed since they were last refitted, the one undo history
//! of volume steps, layer strokes and layer settings, and what the GPU rows need redrawn.
//!
//! The volume keeps the big form; the layers ride on its surface. After a volume stroke the real volume (never the
//! copy strokes run ahead on) is refitted off the main thread and swapped in a frame or so later. What a swap could not
//! carry by name it hands back as a patch, kept with the oldest volume step the swap covers, so undoing that step puts
//! it back exactly. Undoing or redoing a volume step refits at once and keeps that refit's own patch for the other
//! direction, so every state shown after a swap comes back bit for bit.

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
    /// that the two histories still agree. `back`: what to restore, newest last, once the step is undone (on the redo
    /// side: once it is done again) and the layers refitted.
    Volume { serial: u64, id: u64, back: Vec<Patch> },
    /// A layer or mask stroke (or a mask fill): restoring the patch undoes it, and gives the patch that redoes it.
    Layer(Patch),
    /// A layer's settings (name, opacity, visibility, blend, masks) before and after.
    Meta { id: LayerId, before: LayerMeta, after: LayerMeta },
}

/// A volume step undone or redone whose refit is under way: once it swaps in, what goes back (`back`, restored newest
/// first) and the step's entry on the other side, with the refit's own patch.
struct Landing {
    serial: u64,
    /// The step's name on the side it goes to (`redo_id` after an undo, `last_id` after a redo).
    id: Option<u64>,
    back: Vec<Patch>,
    /// It goes on the redo side (an undo), else the history (a redo).
    to_redo: bool,
    /// The history as it was then: anything done meanwhile drops the entry (what there was to redo goes).
    epoch: u64,
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
    next_serial: u64,
    /// Volume steps since the last refit started, and those the refit under way covers (oldest first).
    uncovered: Vec<u64>,
    cover: Vec<u64>,
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
    /// An undo or redo of a volume step whose refit is under way.
    landing: Option<Landing>,
    /// Bumped by every new step.
    epoch: u64,
    pub times: MeshTimes,
}

/// Take the mesh layers' chunk off the end of a `.svol` file's bytes, if it has one (see `VolumeMode::write_to`):
/// returns its body ("MLAY" and its length left out), the bytes before it left in `bytes`.
pub fn split_mlay(bytes: &mut Vec<u8>) -> std::io::Result<Option<Vec<u8>>> {
    let n = bytes.len();
    if n < 24 || &bytes[n - 4..] != b"MLAY" {
        return Ok(None);
    }
    let back = u64::from_le_bytes(bytes[n - 12..n - 4].try_into().unwrap()) as usize;
    let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "a damaged mesh layers chunk");
    let start = n.checked_sub(back).ok_or_else(bad)?;
    if &bytes[start..start + 4] != b"MLAY" {
        return Err(bad());
    }
    let len = u64::from_le_bytes(bytes[start + 4..start + 12].try_into().unwrap()) as usize;
    if start + 12 + len + 12 != n {
        return Err(bad());
    }
    let body = bytes[start + 12..start + 12 + len].to_vec();
    bytes.truncate(start);
    Ok(Some(body))
}

impl MeshStack {
    /// The mesh layers as `VolumeMode::write_to` wrote them (the chunk's body), on a fresh mesh taken from `vol`.
    pub fn read_from(body: &[u8], vol: &LevelVolume) -> std::io::Result<MeshStack> {
        let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
        let mut cur = std::io::Cursor::new(body);
        let (layers, lost) = MeshLayers::read_from(&mut cur, vol).map_err(|e| err(e.to_string()))?;
        if lost > 0 {
            eprintln!("mesh layers: {lost} vertices' data had no vertex of their name to go back to");
        }
        let at = cur.position() as usize;
        let mut stack = MeshStack::new(layers, HashSet::new());
        if body.len() >= at + 8 {
            let len = u64::from_le_bytes(body[at..at + 8].try_into().unwrap()) as usize;
            if let Some(json) = body.get(at + 8..at + 8 + len)
                && let Ok(v) = serde_json::from_slice::<serde_json::Value>(json)
                && let Some(f) = v.get("fills").and_then(|f| serde_json::from_value::<HashMap<String, f32>>(f.clone()).ok())
            {
                stack.fills = f;
            }
        }
        Ok(stack)
    }

    pub fn new(layers: MeshLayers, dirty: HashSet<Key>) -> MeshStack {
        MeshStack {
            layers,
            dirty,
            skip: HashSet::new(),
            history: Vec::new(),
            redo: Vec::new(),
            next_serial: 1,
            uncovered: Vec::new(),
            cover: Vec::new(),
            rows: RowsDirty { all: true, ..Default::default() },
            shown: false,
            leaf_blocks: (0, Vec::new()),
            active_mask: HashMap::new(),
            fills: HashMap::new(),
            open: None,
            landing: None,
            epoch: 0,
            times: MeshTimes::default(),
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
        let serial = self.next_serial;
        self.next_serial += 1;
        self.history.push(Entry::Volume { serial, id, back: Vec::new() });
        self.uncovered.push(serial);
        self.redo.clear();
        self.epoch += 1;
        if self.history.len() > 512 {
            self.history.remove(0);
        }
    }

    /// Something else was done (a layer stroke, a setting): it goes on the history, and what there was to redo goes.
    pub(crate) fn push(&mut self, e: Entry) {
        self.history.push(e);
        self.redo.clear();
        self.epoch += 1;
        if self.history.len() > 512 {
            self.history.remove(0);
        }
    }

    /// Drop the history (a layer removed: its strokes can no longer be put back).
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.redo.clear();
        self.uncovered.clear();
        self.cover.clear();
        self.epoch += 1;
    }

    /// Whether a refit is under way or waiting (blocks changed since the last one began).
    pub fn pending(&self) -> bool {
        self.layers.busy() || !self.dirty.is_empty() || self.landing.is_some()
    }

    /// Start refitting to `vol` (the real volume, at rest) if blocks changed and no refit runs. Returns whether one
    /// started.
    pub fn start_refit(&mut self, vol: &LevelVolume) -> bool {
        if self.dirty.is_empty() || self.layers.busy() {
            return false;
        }
        let mut keys: Vec<Key> = self.dirty.drain().collect();
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        let b = MeshLayers::changed_box(vol, &keys);
        self.times.refit_start_ms = self.layers.refit_async(vol, &keys, b);
        self.cover.extend(self.uncovered.drain(..));
        true
    }

    /// Take a finished refit in (with `wait`, the one under way), filing its patch on the oldest volume step it
    /// covers. Returns whether a swap came in.
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
        let swapped = match r {
            Ok(st) if !st.discarded => {
                let _ = self.layers.doc.take_geometry_dirty();
                self.rows.blocks.extend_from_slice(&st.blocks.changed);
                self.rows.removed.extend_from_slice(&st.blocks.removed);
                self.times.swap = Some(st);
                self.times.swaps += 1;
                true
            }
            Ok(_) => false,
            Err(e) => {
                eprintln!("mesh layers: refit failed: {e}");
                false
            }
        };
        let patches = self.layers.take_patches();
        if swapped && let Some(l) = self.landing.take() {
            // An undo or redo of a volume step: its refit is in. What it could not carry goes with the step to the
            // other side; what the step kept goes back.
            let fwd = patches.into_iter().next().unwrap_or_default();
            for p in l.back.iter().rev() {
                if let Err(e) = self.layers.restore(p) {
                    eprintln!("mesh layers: restore failed: {e}");
                }
            }
            if let (true, Some(id)) = (l.epoch == self.epoch, l.id) {
                let e = Entry::Volume { serial: l.serial, id, back: vec![fwd] };
                if l.to_redo { self.redo.push(e) } else { self.history.push(e) }
            }
            self.times.swap_frame_ms = t.elapsed().as_secs_f64() * 1e3;
            return true;
        }
        if swapped {
            let oldest = self.cover.first().copied();
            let entry = oldest.and_then(|s| self.history.iter_mut().find(|e| matches!(e, Entry::Volume { serial, .. } if *serial == s)));
            if let Some(Entry::Volume { back, .. }) = entry {
                back.extend(patches.into_iter().filter(|p| !p.is_empty()));
            }
            // (A swap covering no step kept in the history: nothing undoes into it, its patch has no use.)
            self.cover.clear();
            if self.layers.busy() {
                // (Not by this stack's rules: a refit is started only when none runs. Count it as covering what waits.)
                self.cover.extend(self.uncovered.drain(..));
            }
            self.times.swap_frame_ms = t.elapsed().as_secs_f64() * 1e3;
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

    /// After `vol`'s undo or redo of a step: start refitting to it, with every block it has not sent yet; `l` lands
    /// when the swap comes in. (No refit runs: [`Self::finish`] came first.)
    fn start_landing(&mut self, vol: &LevelVolume, l: Landing) {
        let t = Instant::now();
        let waiting = vol.dirty_keys();
        let mut keys: Vec<Key> = waiting.iter().copied().chain(self.dirty.drain()).collect();
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        keys.dedup();
        let b = MeshLayers::changed_box(vol, &keys);
        self.layers.refit_async(vol, &keys, b);
        // (Steps since the last refit are in this one; it covers none for undo: its patch goes with `l`.)
        self.uncovered.clear();
        self.landing = Some(l);
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
                    if !vol.undo() {
                        return false;
                    }
                    let serial = self.serial();
                    let l = Landing { serial, id: vol.redo_id(), back: Vec::new(), to_redo: true, epoch: self.epoch };
                    self.start_landing(vol, l);
                    return true;
                }
                Some(Entry::Layer(p)) => {
                    match self.layers.restore_swap(&p) {
                        Ok(inv) => self.redo.push(Entry::Layer(inv)),
                        Err(e) => eprintln!("mesh layers: undo failed: {e}"),
                    }
                    return true;
                }
                Some(Entry::Meta { id, before, after }) => {
                    let _ = self.layers.doc.set_layer_meta(id, before.clone(), false);
                    self.redo.push(Entry::Meta { id, before, after });
                    return true;
                }
                Some(Entry::Volume { serial, id, back }) => {
                    if vol.last_id() != Some(id) {
                        // The two histories no longer agree: the older entries can only be wrong.
                        self.history.clear();
                        continue;
                    }
                    vol.undo();
                    let l = Landing { serial, id: vol.redo_id(), back, to_redo: true, epoch: self.epoch };
                    self.start_landing(vol, l);
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
            Some(Entry::Layer(p)) => {
                match self.layers.restore_swap(&p) {
                    Ok(inv) => self.history.push(Entry::Layer(inv)),
                    Err(e) => eprintln!("mesh layers: redo failed: {e}"),
                }
                true
            }
            Some(Entry::Meta { id, before, after }) => {
                let _ = self.layers.doc.set_layer_meta(id, after.clone(), false);
                self.history.push(Entry::Meta { id, before, after });
                true
            }
            Some(Entry::Volume { serial, id, back }) => {
                if vol.redo_id() != Some(id) {
                    self.redo.clear();
                    return false;
                }
                vol.redo();
                let l = Landing { serial, id: vol.last_id(), back, to_redo: false, epoch: self.epoch };
                self.start_landing(vol, l);
                true
            }
        }
    }

    fn serial(&mut self) -> u64 {
        let s = self.next_serial;
        self.next_serial += 1;
        s
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

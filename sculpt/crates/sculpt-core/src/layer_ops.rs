//! Layer tree: folders, ordering, grouping, duplicate, merge down, solo.
//!
//! The document keeps layers in one flat `Vec`. Each node names its parent
//! folder, and the relative order of siblings is their order in the `Vec`
//! (last = top of the stack). Because layer deltas add, order never changes the
//! composited surface: only strength, visibility and solo do, and those are
//! folded into a per-layer `scale` by [`Document::refresh_scales`]. That keeps
//! the sculpting hot path (`weight = scale * mask`) free of any tree walking.
//!
//! Every structural edit is a list of reversible [`StructOp`]s recorded in the
//! normal undo stack, so undo and redo treat strokes and structure alike.

use std::collections::{BTreeSet, HashMap};

use crate::document::Document;
use crate::layers::{LayerId, LayerKind, LayerMeta, SculptLayer};
use crate::undo::{self, Data, Record, StructOp};
use crate::{Error, Result};

/// One row of the flattened layer tree, top of the stack first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeRow {
    pub id: LayerId,
    pub depth: usize,
}

/// Where a layer is put relative to another node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Same parent, directly above `target` in the stack.
    Above(LayerId),
    /// Same parent, directly below `target`.
    Below(LayerId),
    /// Inside folder `target`, on top of its children.
    Into(LayerId),
    /// Top level, on top of everything.
    Top,
}

impl Document {
    // ---------------------------------------------------------------- queries

    pub fn solo(&self) -> Option<LayerId> {
        self.solo
    }

    /// Direct children of `parent` (`None` = top level), bottom of the stack first.
    pub fn children(&self, parent: Option<LayerId>) -> Vec<LayerId> {
        self.layers.iter().filter(|l| l.parent == parent).map(|l| l.id).collect()
    }

    /// The whole tree in display order: top of the stack first, folders before their children.
    pub fn layer_tree(&self) -> Vec<TreeRow> {
        let mut out = Vec::with_capacity(self.layers.len());
        self.walk(None, 0, &mut out);
        out
    }

    fn walk(&self, parent: Option<LayerId>, depth: usize, out: &mut Vec<TreeRow>) {
        for id in self.children(parent).into_iter().rev() {
            out.push(TreeRow { id, depth });
            if self.layers[self.layer_index_of(id)].is_folder() {
                self.walk(Some(id), depth + 1, out);
            }
        }
    }

    fn layer_index_of(&self, id: LayerId) -> usize {
        self.layers.iter().position(|l| l.id == id).expect("layer id in tree")
    }

    /// `id` and everything below it in the tree, parents first.
    pub fn subtree(&self, id: LayerId) -> Vec<LayerId> {
        let mut out = vec![id];
        let mut i = 0;
        while i < out.len() {
            let cur = out[i];
            out.extend(self.layers.iter().filter(|l| l.parent == Some(cur)).map(|l| l.id));
            i += 1;
        }
        out
    }

    pub fn is_ancestor(&self, ancestor: LayerId, of: LayerId) -> bool {
        let mut cur = self.layer(of).and_then(|l| l.parent);
        // Bounded so a corrupt file with a parent cycle cannot hang the loader.
        for _ in 0..=self.layers.len() {
            match cur {
                Some(p) if p == ancestor => return true,
                Some(p) => cur = self.layer(p).and_then(|l| l.parent),
                None => return false,
            }
        }
        true
    }

    // ------------------------------------------------------------------- solo

    /// Show base plus only this layer (and its subtree). Non-destructive; not undoable.
    pub fn set_solo(&mut self, id: Option<LayerId>) -> Result<()> {
        if let Some(id) = id {
            self.layer_index(id)?;
        }
        self.solo = id;
        self.refresh_scales();
        Ok(())
    }

    /// Recompute every layer's effective scale and recomposite the layers whose scale changed.
    pub(crate) fn refresh_scales(&mut self) {
        let mut leaves = BTreeSet::new();
        self.refresh_scales_into(&mut leaves);
        self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
    }

    pub(crate) fn refresh_scales_into(&mut self, leaves: &mut BTreeSet<u32>) {
        let by_id: HashMap<LayerId, usize> = self.layers.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
        let solo_chain: BTreeSet<LayerId> = match self.solo {
            Some(s) if by_id.contains_key(&s) => {
                let mut chain = BTreeSet::from([s]);
                let mut cur = self.layers[by_id[&s]].parent;
                while let Some(p) = cur {
                    chain.insert(p);
                    cur = by_id.get(&p).and_then(|&i| self.layers[i].parent);
                }
                chain
            }
            _ => BTreeSet::new(),
        };
        let soloing = !solo_chain.is_empty();
        for i in 0..self.layers.len() {
            let mut scale = 1.0f32;
            let mut in_solo = !soloing;
            let mut cur = Some(i);
            while let Some(c) = cur {
                let l = &self.layers[c];
                let on_chain = solo_chain.contains(&l.id);
                in_solo |= on_chain;
                // The soloed layer and its ancestors show even if hidden.
                if !l.visible && !on_chain {
                    scale = 0.0;
                }
                scale *= l.opacity;
                cur = l.parent.and_then(|p| by_id.get(&p).copied());
            }
            if !in_solo {
                scale = 0.0;
            }
            let layer = &mut self.layers[i];
            if layer.scale != scale {
                layer.scale = scale;
                leaves.extend(layer.allocated_leaves());
            }
        }
    }

    // --------------------------------------------------------- undo plumbing

    /// Apply `op`, returning its inverse. Leaves needing a recomposite are added to `leaves`.
    pub(crate) fn apply_struct(&mut self, op: StructOp, leaves: &mut BTreeSet<u32>) -> StructOp {
        let inverse = match op {
            StructOp::Insert { index, layer } => {
                leaves.extend(layer.allocated_leaves());
                self.layers.insert(index.min(self.layers.len()), *layer);
                self.scalar_dirty.mark_all();
                StructOp::Remove { index }
            }
            StructOp::Remove { index } => {
                let layer = self.layers.remove(index);
                leaves.extend(layer.allocated_leaves());
                self.scalar_dirty.mark_all();
                StructOp::Insert { index, layer: Box::new(layer) }
            }
            StructOp::Place { id, parent, index } => {
                let from = self.layer_index_of(id);
                let mut layer = self.layers.remove(from);
                let inverse = StructOp::Place { id, parent: layer.parent, index: from };
                layer.parent = parent;
                self.layers.insert(index.min(self.layers.len()), layer);
                inverse
            }
            StructOp::Meta { id, meta } => {
                let i = self.layer_index_of(id);
                let before = self.layers[i].meta();
                let mask_changed = before.mask != meta.mask;
                self.layers[i].set_meta(meta);
                if mask_changed {
                    let values = self.layers[i].mask.clone().and_then(|s| self.evaluate_mask(&s).ok());
                    self.layers[i].mask_values = values;
                    self.scalar_dirty.mark_all();
                    leaves.extend(self.layers[i].allocated_leaves());
                }
                StructOp::Meta { id, meta: before }
            }
            StructOp::Active { id } => {
                let before = self.active;
                self.active = id;
                StructOp::Active { id: before }
            }
        };
        self.refresh_scales_into(leaves);
        inverse
    }

    /// Run `ops` as one undoable step.
    fn exec(&mut self, label: &str, ops: Vec<StructOp>) {
        self.begin_stroke(label);
        self.exec_in_open(ops);
        self.end_stroke();
    }

    fn exec_in_open(&mut self, ops: Vec<StructOp>) {
        let mut leaves = BTreeSet::new();
        for op in ops {
            let inverse = self.apply_struct(op, &mut leaves);
            let group = self.undo.open.as_mut().expect("open undo group");
            group.records.push(Record { target: undo::Target::Structure, leaf: 0, data: Data::Op(inverse) });
        }
        self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
    }

    fn next_id(&mut self) -> LayerId {
        let id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        id
    }

    /// Vec index at which a node placed by `at` lands, and its parent. `moving` is excluded from the count.
    fn resolve(&self, at: Placement, moving: Option<LayerId>) -> Result<(Option<LayerId>, usize)> {
        let skip = moving.and_then(|m| self.layers.iter().position(|l| l.id == m));
        let adjust = |i: usize| if skip.is_some_and(|s| s < i) { i - 1 } else { i };
        let len = self.layers.len() - usize::from(skip.is_some());
        Ok(match at {
            Placement::Top => (None, len),
            Placement::Above(t) => {
                let i = self.layer_index(t)?;
                (self.layers[i].parent, adjust(i) + 1)
            }
            Placement::Below(t) => {
                let i = self.layer_index(t)?;
                (self.layers[i].parent, adjust(i))
            }
            Placement::Into(f) => {
                let i = self.layer_index(f)?;
                if !self.layers[i].is_folder() {
                    return Err(Error::InvalidData(format!("'{}' is not a folder", self.layers[i].name)));
                }
                (Some(f), len)
            }
        })
    }

    fn unique_name(&self, base: &str) -> String {
        if !self.layers.iter().any(|l| l.name == base) {
            return base.into();
        }
        (2..).map(|n| format!("{base} {n}")).find(|n| !self.layers.iter().any(|l| &l.name == n)).unwrap()
    }

    // --------------------------------------------------------------- commands

    /// New empty layer at `at`; it becomes the active sculpt target. Undoable.
    pub fn insert_layer(&mut self, name: &str, at: Placement) -> Result<LayerId> {
        self.insert_node(name, at, LayerKind::Layer, "Add layer")
    }

    /// New empty folder at `at`. Undoable.
    pub fn insert_folder(&mut self, name: &str, at: Placement) -> Result<LayerId> {
        self.insert_node(name, at, LayerKind::Folder, "Add folder")
    }

    fn insert_node(&mut self, name: &str, at: Placement, kind: LayerKind, label: &str) -> Result<LayerId> {
        let (parent, index) = self.resolve(at, None)?;
        let id = self.next_id();
        let leaves = self.bvh.leaves.len();
        let mut layer = match kind {
            LayerKind::Layer => SculptLayer::new(id, name, leaves),
            LayerKind::Folder => SculptLayer::new_folder(id, name, leaves),
        };
        layer.parent = parent;
        let mut ops = vec![StructOp::Insert { index, layer: Box::new(layer) }];
        if kind == LayerKind::Layer {
            ops.push(StructOp::Active { id: Some(id) });
        }
        self.exec(label, ops);
        Ok(id)
    }

    /// Remove a node and everything under it. Undoable (the data moves into the undo stack).
    pub fn delete_layer(&mut self, id: LayerId) -> Result<()> {
        self.layer_index(id)?;
        let doomed = self.subtree(id);
        let mut idx: Vec<usize> = doomed.iter().map(|d| self.layer_index_of(*d)).collect();
        idx.sort_unstable_by(|a, b| b.cmp(a));
        let mut ops: Vec<StructOp> = idx.into_iter().map(|index| StructOp::Remove { index }).collect();
        if self.active.is_some_and(|a| doomed.contains(&a)) {
            let next = self.layers.iter().rev().find(|l| !l.is_folder() && !doomed.contains(&l.id)).map(|l| l.id);
            ops.push(StructOp::Active { id: next });
        }
        if self.solo.is_some_and(|s| doomed.contains(&s)) {
            self.solo = None;
        }
        self.exec("Delete layer", ops);
        Ok(())
    }

    /// Copy a layer (or a folder with its subtree) directly above the original. Undoable.
    pub fn duplicate_layer(&mut self, id: LayerId) -> Result<LayerId> {
        self.layer_index(id)?;
        let sources = self.subtree(id);
        let mut remap: HashMap<LayerId, LayerId> = HashMap::new();
        let mut ops = Vec::new();
        let first_slot = self.layer_index_of(id) + 1;
        let root_name = self.unique_name(&format!("{} copy", self.layers[self.layer_index_of(id)].name));
        for (n, (src, at)) in sources.iter().zip(first_slot..).enumerate() {
            let new_id = self.next_id();
            remap.insert(*src, new_id);
            let mut copy = self.layers[self.layer_index_of(*src)].clone();
            copy.id = new_id;
            copy.parent = if n == 0 { copy.parent } else { copy.parent.and_then(|p| remap.get(&p).copied()) };
            if n == 0 {
                copy.name = root_name.clone();
            }
            ops.push(StructOp::Insert { index: at, layer: Box::new(copy) });
        }
        let new_root = remap[&id];
        if !self.layers[self.layer_index_of(id)].is_folder() {
            ops.push(StructOp::Active { id: Some(new_root) });
        }
        self.exec("Duplicate layer", ops);
        Ok(new_root)
    }

    /// Move a node (with its subtree) to a new place. Refuses moves that would nest a folder inside itself.
    pub fn move_layer(&mut self, id: LayerId, at: Placement) -> Result<()> {
        self.layer_index(id)?;
        let target = match at {
            Placement::Above(t) | Placement::Below(t) | Placement::Into(t) => Some(t),
            Placement::Top => None,
        };
        if let Some(t) = target
            && (t == id || self.is_ancestor(id, t))
        {
            return Err(Error::InvalidData("cannot move a folder into itself".into()));
        }
        let (parent, index) = self.resolve(at, Some(id))?;
        let i = self.layer_index_of(id);
        if self.layers[i].parent == parent && index == i {
            return Ok(());
        }
        self.exec("Move layer", vec![StructOp::Place { id, parent, index }]);
        Ok(())
    }

    /// Put sibling nodes into a new folder placed where the topmost of them was. Undoable.
    pub fn group_layers(&mut self, ids: &[LayerId], name: &str) -> Result<LayerId> {
        let first = *ids.first().ok_or_else(|| Error::InvalidData("nothing to group".into()))?;
        let parent = self.layers[self.layer_index(first)?].parent;
        let mut order: Vec<usize> = Vec::new();
        for id in ids {
            let i = self.layer_index(*id)?;
            if self.layers[i].parent != parent {
                return Err(Error::InvalidData("only siblings can be grouped".into()));
            }
            order.push(i);
        }
        order.sort_unstable();
        order.dedup();
        let top = *order.last().unwrap();
        let folder_id = self.next_id();
        let mut folder = SculptLayer::new_folder(folder_id, name, self.bvh.leaves.len());
        folder.parent = parent;
        let moving: Vec<LayerId> = order.iter().map(|&i| self.layers[i].id).collect();
        self.begin_stroke("Group layers");
        self.exec_in_open(vec![StructOp::Insert { index: top + 1, layer: Box::new(folder) }]);
        for id in &moving {
            // Each member lands on top of the folder's children, so relative order is kept.
            let end = self.layers.len() - 1;
            self.exec_in_open(vec![StructOp::Place { id: *id, parent: Some(folder_id), index: end }]);
        }
        self.end_stroke();
        Ok(folder_id)
    }

    /// Dissolve a folder, leaving its children in its place. Undoable.
    pub fn ungroup(&mut self, folder: LayerId) -> Result<()> {
        let fi = self.layer_index(folder)?;
        if !self.layers[fi].is_folder() {
            return Err(Error::InvalidData("not a folder".into()));
        }
        let parent = self.layers[fi].parent;
        let kids = self.children(Some(folder));
        self.begin_stroke("Ungroup");
        for (n, kid) in kids.iter().enumerate() {
            let at = self.layer_index_of(folder) + 1 + n;
            // `index` counts with the kid removed; the kid always sat above the folder, so no shift.
            self.exec_in_open(vec![StructOp::Place { id: *kid, parent, index: at.min(self.layers.len() - 1) }]);
        }
        let fi = self.layer_index_of(folder);
        self.exec_in_open(vec![StructOp::Remove { index: fi }]);
        self.end_stroke();
        Ok(())
    }

    /// Merge `id` into the layer directly below it (same folder). The lower layer keeps its name and
    /// becomes the sum of both at their current strengths and masks. Undoable.
    pub fn merge_down(&mut self, id: LayerId) -> Result<LayerId> {
        let si = self.layer_index(id)?;
        let src = &self.layers[si];
        if src.is_folder() {
            return Err(Error::InvalidData("flatten a folder instead of merging it down".into()));
        }
        let parent = src.parent;
        let Some(di) = (0..si).rev().find(|&i| self.layers[i].parent == parent) else {
            return Err(Error::InvalidData("nothing below to merge into".into()));
        };
        let (dst, src) = (&self.layers[di], &self.layers[si]);
        if dst.is_folder() {
            return Err(Error::InvalidData("the layer below is a folder".into()));
        }
        for l in [dst, src] {
            if l.locked {
                return Err(Error::LayerLocked(l.name.clone()));
            }
            if !l.visible {
                return Err(Error::LayerHidden(l.name.clone()));
            }
        }
        let dst_id = dst.id;
        let leaves: Vec<u32> = {
            let mut s: BTreeSet<u32> = dst.allocated_leaves().into_iter().collect();
            s.extend(src.allocated_leaves());
            s.into_iter().collect()
        };
        let mut after = dst.meta();
        after.opacity = 1.0;
        after.mask = None;

        self.begin_stroke("Merge down");
        for &l in &leaves {
            self.snapshot(&undo::Target::Layer(dst_id), l);
        }
        let own = |layer: &SculptLayer, v: usize| layer.opacity * layer.mask_values.as_ref().map_or(1.0, |m| m[v]);
        for &l in &leaves {
            let range = self.bvh.leaves[l as usize].owned_range();
            let len = range.len();
            let mut merged = vec![glam::Vec3::ZERO; len];
            for layer in [&self.layers[di], &self.layers[si]] {
                if let Some(chunk) = &layer.chunks[l as usize] {
                    for (k, m) in merged.iter_mut().enumerate() {
                        *m += chunk[k] * own(layer, range.start + k);
                    }
                }
            }
            self.layers[di].chunks[l as usize] = Some(merged.into_boxed_slice());
        }
        // Dst's own mask/strength are now baked in. Swap metadata through the undo path.
        let remove_at = self.layer_index_of(id);
        self.exec_in_open(vec![StructOp::Meta { id: dst_id, meta: after }, StructOp::Remove { index: remove_at }]);
        if self.active == Some(id) {
            self.exec_in_open(vec![StructOp::Active { id: Some(dst_id) }]);
        }
        self.recomposite(&leaves);
        self.end_stroke();
        Ok(dst_id)
    }

    /// Change a layer's name, strength, visibility, lock, collapse state or mask as one undoable edit.
    /// With `coalesce`, consecutive edits of the same layer (a slider drag) share one undo step.
    pub fn set_layer_meta(&mut self, id: LayerId, meta: LayerMeta, coalesce: bool) -> Result<()> {
        let i = self.layer_index(id)?;
        if self.layers[i].meta() == meta {
            return Ok(());
        }
        if self.layers[i].is_folder() && meta.mask.is_some() {
            return Err(Error::InvalidData("folders cannot carry a mask yet".into()));
        }
        let merge = coalesce
            && self.undo.redo.is_empty()
            && self.undo.open.is_none()
            && self.undo.undo.last().is_some_and(|g| {
                g.label == "Edit layer" && matches!(g.records.as_slice(), [Record { data: Data::Op(StructOp::Meta { id: m, .. }), .. }] if *m == id)
            });
        if merge {
            // Keep the original "before" in the existing record; just apply the new state.
            let mut leaves = BTreeSet::new();
            let _ = self.apply_struct(StructOp::Meta { id, meta }, &mut leaves);
            self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
        } else {
            self.exec("Edit layer", vec![StructOp::Meta { id, meta }]);
        }
        Ok(())
    }
}

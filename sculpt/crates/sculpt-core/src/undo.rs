//! Chunked copy-on-write undo.
//!
//! The first time a stroke touches a leaf of some target (base, a layer, the
//! freeze mask or a channel), that leaf's data is snapshotted. Undoing swaps
//! snapshots back, so cost scales with what the stroke touched, never with
//! mesh size.

use std::collections::HashSet;

use glam::Vec3;

use crate::layers::{Chunk, LayerId, LayerMeta, SculptLayer};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Target {
    Base,
    Layer(LayerId),
    Freeze,
    Channel(String),
    /// Layer-list structure (insert, remove, move, rename...). `leaf` is unused.
    Structure,
}

/// A reversible edit of the layer list. Applying one returns its inverse.
#[derive(Debug)]
pub(crate) enum StructOp {
    Insert { index: usize, layer: Box<SculptLayer> },
    Remove { index: usize },
    /// Move `id` under `parent` at vec position `index` (counted with `id` removed).
    Place { id: LayerId, parent: Option<LayerId>, index: usize },
    Meta { id: LayerId, meta: LayerMeta },
    Active { id: Option<LayerId> },
}

#[derive(Debug)]
pub(crate) enum Data {
    Op(StructOp),
    Positions(Box<[Vec3]>),
    Deltas(Chunk),
    Scalars(Box<[f32]>),
    /// Channel did not exist before this group.
    Absent,
}

#[derive(Debug)]
pub(crate) struct Record {
    pub target: Target,
    pub leaf: u32,
    pub data: Data,
}

#[derive(Debug, Default)]
pub(crate) struct Group {
    pub label: String,
    pub records: Vec<Record>,
    seen: HashSet<(Target, u32)>,
}

impl Group {
    pub fn new(label: &str) -> Group {
        Group { label: label.into(), ..Default::default() }
    }

    /// True if `(target, leaf)` still needs a snapshot in this group.
    pub fn wants(&mut self, target: &Target, leaf: u32) -> bool {
        self.seen.insert((target.clone(), leaf))
    }
}

#[derive(Debug)]
pub(crate) struct UndoStack {
    pub undo: Vec<Group>,
    pub redo: Vec<Group>,
    pub open: Option<Group>,
    pub limit: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        UndoStack { undo: Vec::new(), redo: Vec::new(), open: None, limit: 64 }
    }
}

impl UndoStack {
    pub fn commit(&mut self) {
        if let Some(g) = self.open.take()
            && !g.records.is_empty() {
                self.undo.push(g);
                self.redo.clear();
                if self.undo.len() > self.limit {
                    self.undo.remove(0);
                }
            }
    }

    pub fn clear(&mut self) {
        *self = UndoStack { limit: self.limit, ..Default::default() };
    }
}

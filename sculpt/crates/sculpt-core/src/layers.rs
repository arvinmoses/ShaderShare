//! Mudbox-style sculpt layers.
//!
//! Each layer stores *offsets* from the layers below, sparsely, one chunk per
//! spatial leaf. The displayed surface is
//!
//! ```text
//! P(v) = base(v) + Σ_layers  opacity · mask(v) · delta(v)
//! ```
//!
//! so the strength slider and the (procedural or painted) layer mask scale a
//! layer's effect per vertex without touching its data, and untouched regions
//! of a layer cost no memory.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::mask::MaskStack;

pub type Chunk = Option<Box<[Vec3]>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Holds sculpt deltas.
    #[default]
    Layer,
    /// Organises other nodes. Its strength and visibility scale its subtree.
    Folder,
}

/// How a sculpt layer's offset combines with the offsets of the layers below it.
///
/// `A` is the offset accumulated so far, `L` the layer's stored offset, `s` its
/// strength times mask (and ancestors' strengths), `n` the surface normal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerBlend {
    /// `A + s·L`. Layers stack, which is how sculpt layers always behaved.
    #[default]
    Add,
    /// `A − s·L`. The layer digs where it would have built up.
    Subtract,
    /// Within the layer's footprint, replace what is below: `A + s·(L − A)`.
    Normal,
    /// Apply the layer only where it raises the surface (`L·n > 0`).
    Max,
    /// Apply the layer only where it lowers the surface (`L·n < 0`).
    Min,
}

impl LayerBlend {
    pub const ALL: [LayerBlend; 5] = [LayerBlend::Normal, LayerBlend::Add, LayerBlend::Subtract, LayerBlend::Min, LayerBlend::Max];

    pub fn label(self) -> &'static str {
        match self {
            LayerBlend::Normal => "Normal",
            LayerBlend::Add => "Add",
            LayerBlend::Subtract => "Subtract",
            LayerBlend::Min => "Min",
            LayerBlend::Max => "Max",
        }
    }

    /// Four-letter label for the layer row.
    pub fn short(self) -> &'static str {
        match self {
            LayerBlend::Normal => "Norm",
            LayerBlend::Add => "Add",
            LayerBlend::Subtract => "Sub",
            LayerBlend::Min => "Min",
            LayerBlend::Max => "Max",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            LayerBlend::Normal => "Replaces the layers below inside this layer's footprint",
            LayerBlend::Add => "Stacks on top of the layers below",
            LayerBlend::Subtract => "Digs where the layer would build up",
            LayerBlend::Min => "Only the parts that lower the surface",
            LayerBlend::Max => "Only the parts that raise the surface",
        }
    }

    /// True when a stroke can be shown by simply adding the dab to the surface.
    pub fn is_linear(self) -> bool {
        matches!(self, LayerBlend::Add | LayerBlend::Subtract)
    }

    /// New accumulated offset. `eps` is the length below which a stored offset counts as "no data".
    #[inline]
    pub fn apply(self, acc: Vec3, layer: Vec3, s: f32, normal: Vec3, eps: f32) -> Vec3 {
        match self {
            LayerBlend::Add => acc + layer * s,
            LayerBlend::Subtract => acc - layer * s,
            LayerBlend::Normal => {
                let coverage = (layer.length() / eps).min(1.0);
                acc + (layer - acc) * (s * coverage)
            }
            LayerBlend::Max => if layer.dot(normal) > 0.0 { acc + layer * s } else { acc },
            LayerBlend::Min => if layer.dot(normal) < 0.0 { acc + layer * s } else { acc },
        }
    }
}

/// The user-editable, cheap-to-copy part of a layer. Undo records swap these.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerMeta {
    pub name: String,
    pub opacity: f32,
    pub visible: bool,
    pub locked: bool,
    pub collapsed: bool,
    pub blend: LayerBlend,
    pub mask: Option<MaskStack>,
}

#[derive(Clone, Debug)]
pub struct SculptLayer {
    pub id: LayerId,
    pub kind: LayerKind,
    /// Containing folder; `None` for top level.
    pub parent: Option<LayerId>,
    /// Folder shown collapsed in the layer list (persisted).
    pub collapsed: bool,
    pub blend: LayerBlend,
    /// Effective strength after ancestors and solo (derived, kept by the document).
    pub(crate) scale: f32,
    pub name: String,
    /// Strength slider. Mudbox allows going past 100% and negative.
    pub opacity: f32,
    pub visible: bool,
    pub locked: bool,
    /// Mask definition; evaluated into `mask_values`.
    pub mask: Option<MaskStack>,
    pub(crate) mask_values: Option<Vec<f32>>,
    pub(crate) chunks: Vec<Chunk>,
}

impl SculptLayer {
    pub(crate) fn new_folder(id: LayerId, name: &str, leaves: usize) -> SculptLayer {
        SculptLayer { kind: LayerKind::Folder, ..SculptLayer::new(id, name, leaves) }
    }

    pub fn is_folder(&self) -> bool {
        self.kind == LayerKind::Folder
    }

    /// Effective strength: own strength times every ancestor's, zero when hidden or soloed out.
    pub fn effective_scale(&self) -> f32 {
        self.scale
    }

    pub fn meta(&self) -> LayerMeta {
        LayerMeta {
            name: self.name.clone(),
            opacity: self.opacity,
            visible: self.visible,
            locked: self.locked,
            collapsed: self.collapsed,
            blend: self.blend,
            mask: self.mask.clone(),
        }
    }

    pub(crate) fn set_meta(&mut self, m: LayerMeta) {
        self.name = m.name;
        self.opacity = m.opacity;
        self.visible = m.visible;
        self.locked = m.locked;
        self.collapsed = m.collapsed;
        self.blend = m.blend;
        self.mask = m.mask;
    }

    pub(crate) fn new(id: LayerId, name: &str, leaves: usize) -> SculptLayer {
        SculptLayer {
            id,
            kind: LayerKind::Layer,
            parent: None,
            collapsed: false,
            blend: LayerBlend::Add,
            scale: 1.0,
            name: name.into(),
            opacity: 1.0,
            visible: true,
            locked: false,
            mask: None,
            mask_values: None,
            chunks: vec![None; leaves],
        }
    }

    /// Effective per-vertex weight of this layer.
    #[inline]
    pub fn weight(&self, v: usize) -> f32 {
        self.scale * self.mask_values.as_ref().map_or(1.0, |m| m[v])
    }

    pub fn mask_values(&self) -> Option<&[f32]> {
        self.mask_values.as_deref()
    }

    /// Leaves with stored deltas.
    pub fn allocated_leaves(&self) -> Vec<u32> {
        self.chunks.iter().enumerate().filter(|(_, c)| c.is_some()).map(|(i, _)| i as u32).collect()
    }

    /// Approximate memory held by deltas, in bytes.
    pub fn delta_bytes(&self) -> usize {
        self.chunks.iter().flatten().map(|c| c.len() * std::mem::size_of::<Vec3>()).sum()
    }
}

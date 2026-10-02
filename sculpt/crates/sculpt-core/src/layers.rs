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

#[derive(Clone, Debug)]
pub struct SculptLayer {
    pub id: LayerId,
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
    pub(crate) fn new(id: LayerId, name: &str, leaves: usize) -> SculptLayer {
        SculptLayer {
            id,
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
        if !self.visible {
            return 0.0;
        }
        self.opacity * self.mask_values.as_ref().map_or(1.0, |m| m[v])
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

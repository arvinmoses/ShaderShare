//! UI-only state of the layer panel. The document stays the source of truth for layers.

use std::time::Instant;

use egui::Pos2;
use sculpt_core::LayerId;

use super::dragdrop::RowGeom;

use super::selection::LayerSelection;

/// Rows being dragged.
#[derive(Clone, Debug)]
pub struct DragState {
    pub ids: Vec<LayerId>,
}

#[derive(Default)]
pub struct PanelState {
    pub selection: LayerSelection,
    /// Row being renamed inline, with its edit buffer.
    pub rename: Option<(LayerId, String)>,
    pub drag: Option<DragState>,
    /// Last copied mask, for Paste mask.
    pub mask_clip: Option<sculpt_core::mask::MaskStack>,
    /// Collapsed folder the drag is hovering, and since when (spring-open).
    pub spring: Option<(LayerId, Instant)>,
    /// Rows as drawn last frame (hit testing for drops, headless checks).
    pub rows: Vec<RowGeom>,
    /// Replaces the real pointer during a drag. Lets headless runs and tests place a drop.
    pub pointer_override: Option<Pos2>,
}

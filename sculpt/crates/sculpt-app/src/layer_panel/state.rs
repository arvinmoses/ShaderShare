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
    /// Radial menu centre, when open.
    pub radial: Option<Pos2>,
    /// A mask op being dragged: its layer and its index in the mask stack.
    pub op_drag: Option<(LayerId, usize)>,
    /// What the current tool would edit, refreshed each frame (drives the thumbnail frames).
    pub paint_kind: Option<super::target::TargetKind>,
    /// Dense rows (28 px) instead of comfortable ones (36 px). Saved in settings.
    pub compact: bool,
    pub thumbs: super::thumbs::ThumbCache,
    /// Tab quick-switcher, when open.
    pub switcher: Option<super::switcher::Switcher>,
    /// Last copied mask, for Paste mask.
    /// The layer it came from (so its paint can be copied) and the mask.
    pub mask_clip: Option<(sculpt_core::LayerId, sculpt_core::mask::MaskStack)>,
    /// Collapsed folder the drag is hovering, and since when (spring-open).
    pub spring: Option<(LayerId, Instant)>,
    /// Rows as drawn last frame (hit testing for drops, headless checks).
    pub rows: Vec<RowGeom>,
    /// Replaces the real pointer during a drag. Lets headless runs and tests place a drop.
    pub pointer_override: Option<Pos2>,
    /// Treat Ctrl as held during a drag (copy). Same purpose as `pointer_override`.
    pub copy_override: bool,
}

//! What the layer panel's rows show of a layer and its masks, and the edits they give back.

use sculpt_core::LayerBlend;
use sculpt_core::mask::BlendMode;
use sculpt_core::noise::{NoiseKind, NoiseParams};

/// The most a layer's opacity goes to: over 1 it shows its change exaggerated.
pub const MAX_OPACITY: f32 = 2.0;

/// The blends a layer row offers (a layer set to `Normal` shows as `Add`).
pub const ROW_BLENDS: [LayerBlend; 4] = [LayerBlend::Add, LayerBlend::Subtract, LayerBlend::Max, LayerBlend::Min];

/// A noise mask: noise in world space, remapped about `balance` by `contrast` (0 soft, 1 a hard edge).
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseMask {
    pub params: NoiseParams,
    pub balance: f32,
    pub contrast: f32,
    pub invert: bool,
}

impl Default for NoiseMask {
    fn default() -> Self {
        NoiseMask { params: NoiseParams { kind: NoiseKind::Fbm, scale: 6.0, octaves: 4, ..Default::default() }, balance: 0.5, contrast: 0.6, invert: false }
    }
}

/// What a mask is, for a mask list (and to change one: give it back changed).
#[derive(Clone, Debug, PartialEq)]
pub enum MaskKind {
    /// Painted, from `fill` (0 black, 1 white) where nothing is painted.
    Paint { fill: f32 },
    Noise(NoiseMask),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub kind: MaskKind,
}

/// A layer as its row shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    /// One of [`ROW_BLENDS`].
    pub blend: LayerBlend,
    /// How much it holds (vertices, roughly).
    pub blocks: usize,
    /// Bottom first.
    pub masks: Vec<MaskInfo>,
    /// The mask strokes paint (or that is picked), if any: else strokes sculpt the layer.
    pub active_mask: Option<usize>,
}

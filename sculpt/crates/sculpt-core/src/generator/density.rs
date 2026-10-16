//! Where a generator layer puts its instances: an even scatter, or a density field that pulls them toward some places.
//!
//! A density field is a mask (white = every spawn wants to be here, black = none) built from the same mask machinery the
//! layers use. The instance count stays fixed: the field redistributes it, importance-sampling the surface by area times
//! density, so a dense patch gets crowded and a black one stays bare, instead of thinning a fixed scatter after the fact.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::place::SurfaceSampler;
use crate::mask::{Levels, MaskLayer, MaskSource, MaskStack};
use crate::mesh::Face;
use crate::noise::NoiseParams;

/// How the instances are spread over the surface.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Density {
    /// The same everywhere (by surface area).
    #[default]
    Even,
    /// Instances gather in clumps with bare ground between. `scale` is the size of the patches (larger = bigger patches),
    /// `coverage` the share of the surface that is dense (0 to 1), `softness` how gradually a clump thins out at its edge.
    Clusters { scale: f32, coverage: f32, softness: f32, seed: u32 },
    /// More instances where the surface faces `axis` (for example rubble settles toward the ground with `axis` = -Y, or
    /// dust gathers on top with +Y). `sharpness` narrows how tightly it has to face that way.
    Facing { axis: [f32; 3], sharpness: f32 },
    /// Painted by hand into a per-vertex channel (the mask paint tool writes `paint.mask`).
    Painted { channel: String },
}

impl Density {
    /// The mask stack that realises this density, or `None` for an even scatter.
    pub fn to_stack(&self) -> Option<MaskStack> {
        match self {
            Density::Even => None,
            Density::Clusters { scale, coverage, softness, seed } => {
                let half = softness.clamp(0.02, 1.0) * 0.5;
                let centre = 1.0 - coverage.clamp(0.0, 1.0);
                let (lo, hi) = ((centre - half).clamp(0.0, 0.98), (centre + half).clamp(0.02, 1.0));
                let noise = NoiseParams { scale: *scale, seed: *seed, ..Default::default() };
                Some(MaskStack::new(0.0).with(MaskLayer::new("Clusters", MaskSource::Noise(noise)).levels(Levels::range(lo, hi.max(lo + 0.01)))))
            }
            Density::Facing { axis, sharpness } => {
                let axis = Vec3::from_array(*axis).normalize_or(Vec3::NEG_Y);
                Some(MaskStack::new(0.0).with(MaskLayer::new("Facing", MaskSource::Direction { axis, sharpness: sharpness.max(0.05) })))
            }
            Density::Painted { channel } => Some(MaskStack::new(0.0).with(MaskLayer::new("Painted", MaskSource::Channel { name: channel.clone() }))),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Density::Even => "Even",
            Density::Clusters { .. } => "Clusters",
            Density::Facing { .. } => "Facing",
            Density::Painted { .. } => "Painted",
        }
    }

    /// A sensible starting point for each kind, for the panel's mode switch.
    pub fn default_clusters() -> Density {
        Density::Clusters { scale: 3.0, coverage: 0.45, softness: 0.25, seed: 1 }
    }

    pub fn default_facing() -> Density {
        Density::Facing { axis: [0.0, -1.0, 0.0], sharpness: 1.5 }
    }

    pub fn default_painted() -> Density {
        Density::Painted { channel: "paint.mask".into() }
    }
}

/// A density evaluated over a mesh: the per-vertex values and the area-times-density sampling table built from them.
/// Building it walks every face, so it is kept until the density or the surface it was made for changes.
#[derive(Debug)]
pub struct DensityField {
    pub(crate) stack: MaskStack,
    pub(crate) sampler: SurfaceSampler,
    pub(crate) values: Vec<f32>,
}

impl DensityField {
    pub fn new(stack: MaskStack, base: &[Vec3], faces: &[Face], values: Vec<f32>) -> DensityField {
        let sampler = SurfaceSampler::new(base, faces, Some(&values));
        DensityField { stack, sampler, values }
    }

    /// Density at each vertex, 0 to 1.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Total of area times density over the surface: zero means nothing can be placed.
    pub fn total_weight(&self) -> f64 {
        self.sampler.total_weight()
    }
}

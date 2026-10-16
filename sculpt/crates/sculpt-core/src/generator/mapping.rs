//! Attribute mapping: letting the surface decide how each instance looks.
//!
//! A generator layer carries an ordered list of rows. Each row reads one *source* at the instance's point on the surface
//! (how steeply it faces, how high it is, how hollow it is, a noise, a painted channel, a random number, its place in the
//! scatter), turns it into a value `t` from 0 to 1 through a *remap*, and applies it to one *target* property with a signed
//! `strength`. So "bigger rubble in the hollows" is the row `Cavity -> Size, strength +1`, and "smaller on steep slopes"
//! is `Slope -> Size, strength -0.6, inverted`.
//!
//! The rows are a fixed table, not a node graph: the order only matters for how their effects are listed, since sizes
//! multiply and the other targets add.

use std::collections::BTreeMap;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::place::SurfaceSample;
use crate::bake::MeshAttribute;
use crate::noise::{Noise, NoiseParams};

/// What a row reads at an instance's point on the surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttributeSource {
    /// How squarely the surface faces `axis` (1 facing it, 0 edge-on or away): up for flat tops, down for undersides.
    Slope { axis: [f32; 3] },
    /// Position along `axis`, from 0 at `from` to 1 at `to` (world units).
    Height { axis: [f32; 3], from: f32, to: f32 },
    /// A smooth noise of the point's position; `scale` is cycles per world unit.
    Noise { scale: f32, seed: u32 },
    /// A baked mesh attribute (cavity, curvature, ambient occlusion, thickness). Baked on demand.
    Mesh { attribute: MeshAttribute },
    /// Any per-vertex channel, for example one painted with the mask tool.
    Channel { name: String },
    /// A random number per instance, from its own stream.
    Random { seed: u32 },
    /// The instance's position in the scatter order, 0 for the first and 1 for the last.
    Index,
}

impl AttributeSource {
    pub fn label(&self) -> &'static str {
        match self {
            AttributeSource::Slope { .. } => "Slope",
            AttributeSource::Height { .. } => "Height",
            AttributeSource::Noise { .. } => "Noise",
            AttributeSource::Mesh { attribute } => match attribute {
                MeshAttribute::Curvature => "Curvature",
                MeshAttribute::Cavity => "Cavity",
                MeshAttribute::AmbientOcclusion => "Ambient occlusion",
                MeshAttribute::Thickness => "Thickness",
            },
            AttributeSource::Channel { .. } => "Painted channel",
            AttributeSource::Random { .. } => "Random",
            AttributeSource::Index => "Order",
        }
    }

    /// The baked attribute this source needs to exist before it can be read.
    pub fn needs_bake(&self) -> Option<MeshAttribute> {
        match self {
            AttributeSource::Mesh { attribute } => Some(*attribute),
            _ => None,
        }
    }

    /// The per-vertex channel this source reads, if any.
    pub fn channel(&self) -> Option<&str> {
        match self {
            AttributeSource::Mesh { attribute } => Some(attribute.channel_name()),
            AttributeSource::Channel { name } => Some(name),
            _ => None,
        }
    }
}

/// What a row changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeTarget {
    /// Multiplies the size by `1 + strength * t`.
    Size,
    /// Multiplies the height along the surface normal by `1 + strength * t`, leaving the width alone.
    Stretch,
    /// Adds `strength * t` to the lift, in multiples of the size.
    Lift,
    /// Adds `strength * t` degrees to how far the instance may lean from the surface normal.
    Tilt,
}

impl AttributeTarget {
    pub const ALL: [AttributeTarget; 4] = [AttributeTarget::Size, AttributeTarget::Stretch, AttributeTarget::Lift, AttributeTarget::Tilt];

    pub fn label(self) -> &'static str {
        match self {
            AttributeTarget::Size => "Size",
            AttributeTarget::Stretch => "Height",
            AttributeTarget::Lift => "Lift",
            AttributeTarget::Tilt => "Tilt",
        }
    }
}

/// Turns a source value into `t`, 0 to 1: values at or below `in_min` give 0, at or above `in_max` give 1, in between
/// follow a power curve (`gamma` above 1 bunches values toward 0), and `invert` flips the result.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Remap {
    pub in_min: f32,
    pub in_max: f32,
    pub gamma: f32,
    pub invert: bool,
}

impl Default for Remap {
    fn default() -> Self {
        Remap { in_min: 0.0, in_max: 1.0, gamma: 1.0, invert: false }
    }
}

impl Remap {
    #[inline]
    pub fn apply(&self, v: f32) -> f32 {
        // `in_max` below `in_min` reverses the mapping; a zero-width range is a hard threshold.
        let span = self.in_max - self.in_min;
        let t = if span.abs() < 1e-6 { f32::from(v >= self.in_max) } else { ((v - self.in_min) / span).clamp(0.0, 1.0) }.powf(self.gamma.max(1e-3));
        if self.invert { 1.0 - t } else { t }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttributeRow {
    pub source: AttributeSource,
    #[serde(default)]
    pub remap: Remap,
    pub target: AttributeTarget,
    pub strength: f32,
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

impl AttributeRow {
    pub fn new(source: AttributeSource, target: AttributeTarget, strength: f32) -> AttributeRow {
        AttributeRow { source, remap: Remap::default(), target, strength, enabled: true }
    }

    /// The most this row can multiply a size or height by (its target at `t` = 0 or 1, whichever is larger).
    pub fn max_factor(&self) -> f32 {
        if self.enabled && matches!(self.target, AttributeTarget::Size | AttributeTarget::Stretch) { (1.0 + self.strength).max(1.0) } else { 1.0 }
    }
}

/// What the rows add up to for one instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mapped {
    pub size: f32,
    pub stretch: f32,
    pub lift: f32,
    pub tilt_degrees: f32,
}

impl Default for Mapped {
    fn default() -> Self {
        Mapped { size: 1.0, stretch: 1.0, lift: 0.0, tilt_degrees: 0.0 }
    }
}

/// A row ready to be read many times: noise tables are built once.
enum Reader<'a> {
    Slope(Vec3),
    Height(Vec3, f32, f32),
    Noise(Box<Noise>),
    Channel(Option<&'a [f32]>),
    Random(u32),
    Index,
}

/// The rows of one spec, prepared against a mesh's channels, ready to evaluate instance by instance.
pub struct Mapper<'a> {
    rows: Vec<(&'a AttributeRow, Reader<'a>)>,
}

impl<'a> Mapper<'a> {
    pub fn new(rows: &'a [AttributeRow], channels: &'a BTreeMap<String, Vec<f32>>) -> Mapper<'a> {
        let rows = rows
            .iter()
            .filter(|r| r.enabled && r.strength != 0.0)
            .map(|r| {
                let reader = match &r.source {
                    AttributeSource::Slope { axis } => Reader::Slope(Vec3::from_array(*axis).normalize_or(Vec3::Y)),
                    AttributeSource::Height { axis, from, to } => Reader::Height(Vec3::from_array(*axis).normalize_or(Vec3::Y), *from, *to),
                    AttributeSource::Noise { scale, seed } => Reader::Noise(Box::new(Noise::new(&NoiseParams { scale: *scale, seed: *seed, ..Default::default() }))),
                    AttributeSource::Mesh { .. } | AttributeSource::Channel { .. } => Reader::Channel(r.source.channel().and_then(|n| channels.get(n)).map(|v| v.as_slice())),
                    AttributeSource::Random { seed } => Reader::Random(*seed),
                    AttributeSource::Index => Reader::Index,
                };
                (r, reader)
            })
            .collect();
        Mapper { rows }
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The source's value for instance `index` of `count` at `sample`, before the remap.
    fn read(reader: &Reader<'_>, sample: &SurfaceSample, index: usize, count: usize, random: impl Fn(u32, u64) -> f32) -> f32 {
        match reader {
            Reader::Slope(axis) => sample.normal.dot(*axis).clamp(0.0, 1.0),
            Reader::Height(axis, from, to) => {
                let span = to - from;
                if span.abs() < 1e-9 { 0.0 } else { (sample.position.dot(*axis) - from) / span }
            }
            Reader::Noise(noise) => noise.sample(sample.position),
            Reader::Channel(Some(values)) => sample.verts.iter().zip(sample.bary).map(|(&v, w)| values.get(v as usize).copied().unwrap_or(0.0) * w).sum(),
            Reader::Channel(None) => 0.0,
            Reader::Random(seed) => random(*seed, index as u64),
            Reader::Index => {
                if count > 1 { index as f32 / (count - 1) as f32 } else { 0.0 }
            }
        }
    }

    /// What every row does to instance `index` of `count`, scattered at `sample`.
    pub fn map(&self, sample: &SurfaceSample, index: usize, count: usize, random: impl Fn(u32, u64) -> f32 + Copy) -> Mapped {
        let mut out = Mapped::default();
        for (row, reader) in &self.rows {
            let t = row.remap.apply(Self::read(reader, sample, index, count, random));
            let effect = row.strength * t;
            match row.target {
                AttributeTarget::Size => out.size *= 1.0 + effect,
                AttributeTarget::Stretch => out.stretch *= 1.0 + effect,
                AttributeTarget::Lift => out.lift += effect,
                AttributeTarget::Tilt => out.tilt_degrees += effect,
            }
        }
        out
    }
}

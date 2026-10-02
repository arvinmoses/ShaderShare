//! Substance-style mask stacks.
//!
//! A [`MaskStack`] is an ordered list of [`MaskLayer`]s composited bottom to
//! top, each with a source (fill, hand-painted/imported channel, procedural
//! noise, mesh attribute, direction, gradient), levels, blur, blend mode,
//! opacity, and optionally its *own* nested mask. That nesting is what lets
//! you, for example, paint where noise is allowed to show.
//!
//! Stacks are plain serde data: they are saved verbatim in project files and
//! can be written by external tools.

use std::collections::BTreeMap;

use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bake::MeshAttribute;
use crate::mesh::Topology;
use crate::noise::{Noise, NoiseParams};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskStack {
    /// Value below the first layer (0 = black, 1 = white).
    #[serde(default)]
    pub base: f32,
    #[serde(default)]
    pub layers: Vec<MaskLayer>,
}

impl MaskStack {
    pub fn new(base: f32) -> MaskStack {
        MaskStack { base, layers: Vec::new() }
    }

    pub fn with(mut self, layer: MaskLayer) -> MaskStack {
        self.layers.push(layer);
        self
    }

    /// Every channel this stack reads, including nested masks.
    pub fn channels(&self, out: &mut Vec<String>) {
        for l in &self.layers {
            match &l.source {
                MaskSource::Channel { name } => out.push(name.clone()),
                MaskSource::Mesh { attribute } => out.push(attribute.channel_name().to_string()),
                _ => {}
            }
            if let Some(m) = &l.mask {
                m.channels(out);
            }
        }
    }

    pub fn uses_normals(&self) -> bool {
        self.layers.iter().any(|l| {
            matches!(l.source, MaskSource::Direction { .. }) || l.mask.as_ref().is_some_and(|m| m.uses_normals())
        })
    }

    /// Mesh attributes (bakes) this stack needs.
    pub fn required_bakes(&self, out: &mut Vec<MeshAttribute>) {
        for l in &self.layers {
            if let MaskSource::Mesh { attribute } = l.source
                && !out.contains(&attribute) {
                    out.push(attribute);
                }
            if let Some(m) = &l.mask {
                m.required_bakes(out);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskLayer {
    #[serde(default)]
    pub name: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    pub source: MaskSource,
    #[serde(default)]
    pub blend: BlendMode,
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default)]
    pub levels: Levels,
    #[serde(default)]
    pub invert: bool,
    /// Topological blur iterations applied to the source.
    #[serde(default)]
    pub blur: u32,
    /// Per-vertex opacity for this layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<Box<MaskStack>>,
}

fn yes() -> bool {
    true
}
fn one() -> f32 {
    1.0
}

impl MaskLayer {
    pub fn new(name: &str, source: MaskSource) -> MaskLayer {
        MaskLayer {
            name: name.into(),
            enabled: true,
            source,
            blend: BlendMode::Normal,
            opacity: 1.0,
            levels: Levels::default(),
            invert: false,
            blur: 0,
            mask: None,
        }
    }
    pub fn blend(mut self, b: BlendMode) -> Self {
        self.blend = b;
        self
    }
    pub fn opacity(mut self, o: f32) -> Self {
        self.opacity = o;
        self
    }
    pub fn levels(mut self, l: Levels) -> Self {
        self.levels = l;
        self
    }
    pub fn inverted(mut self) -> Self {
        self.invert = true;
        self
    }
    pub fn blurred(mut self, iterations: u32) -> Self {
        self.blur = iterations;
        self
    }
    pub fn masked(mut self, m: MaskStack) -> Self {
        self.mask = Some(Box::new(m));
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaskSource {
    Fill { value: f32 },
    /// Per-vertex channel: hand painted, baked, or imported from another program.
    Channel { name: String },
    Noise(NoiseParams),
    /// Mesh-derived attribute (must be baked first; see [`crate::bake`]).
    Mesh { attribute: MeshAttribute },
    /// How much the surface faces `axis` (e.g. "dust settles on top").
    Direction { axis: Vec3, #[serde(default = "one")] sharpness: f32 },
    /// Linear ramp of position along `axis` between `from` and `to`.
    Gradient { axis: Vec3, from: f32, to: f32 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Add,
    Subtract,
    Screen,
    Overlay,
    Max,
    Min,
    Difference,
}

impl BlendMode {
    #[inline]
    pub fn blend(self, a: f32, b: f32) -> f32 {
        match self {
            BlendMode::Normal => b,
            BlendMode::Multiply => a * b,
            BlendMode::Add => a + b,
            BlendMode::Subtract => a - b,
            BlendMode::Screen => 1.0 - (1.0 - a) * (1.0 - b),
            BlendMode::Overlay => {
                if a < 0.5 { 2.0 * a * b } else { 1.0 - 2.0 * (1.0 - a) * (1.0 - b) }
            }
            BlendMode::Max => a.max(b),
            BlendMode::Min => a.min(b),
            BlendMode::Difference => (a - b).abs(),
        }
    }
}

/// Photoshop/Substance style levels remap.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Levels {
    pub in_min: f32,
    pub in_max: f32,
    pub gamma: f32,
    pub out_min: f32,
    pub out_max: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Levels { in_min: 0.0, in_max: 1.0, gamma: 1.0, out_min: 0.0, out_max: 1.0 }
    }
}

impl Levels {
    pub fn range(in_min: f32, in_max: f32) -> Levels {
        Levels { in_min, in_max, ..Default::default() }
    }

    #[inline]
    pub fn apply(&self, v: f32) -> f32 {
        if *self == Levels::default() {
            return v;
        }
        let span = (self.in_max - self.in_min).max(1e-6);
        let t = ((v - self.in_min) / span).clamp(0.0, 1.0).powf(1.0 / self.gamma.max(1e-3));
        self.out_min + t * (self.out_max - self.out_min)
    }
}

/// Data a mask stack can read.
pub struct MaskContext<'a> {
    /// Rest-pose positions, so procedural masks don't swim while sculpting.
    pub positions: &'a [Vec3],
    /// Rest-pose normals (may be empty if no source needs them).
    pub normals: &'a [Vec3],
    pub topology: &'a Topology,
    pub channels: &'a BTreeMap<String, Vec<f32>>,
}

pub fn evaluate(stack: &MaskStack, ctx: &MaskContext) -> Result<Vec<f32>> {
    let n = ctx.positions.len();
    let mut out = vec![stack.base.clamp(0.0, 1.0); n];
    for layer in stack.layers.iter().filter(|l| l.enabled && l.opacity > 0.0) {
        let mut src = evaluate_source(&layer.source, ctx)?;
        if layer.blur > 0 {
            src = blur(&src, ctx.topology, layer.blur);
        }
        let opacity = match &layer.mask {
            Some(m) => Some(evaluate(m, ctx)?),
            None => None,
        };
        out.par_iter_mut().enumerate().for_each(|(v, a)| {
            let mut b = layer.levels.apply(src[v]);
            if layer.invert {
                b = 1.0 - b;
            }
            let o = layer.opacity * opacity.as_ref().map_or(1.0, |m| m[v]);
            let blended = layer.blend.blend(*a, b);
            *a = (*a + (blended - *a) * o).clamp(0.0, 1.0);
        });
    }
    Ok(out)
}

fn evaluate_source(src: &MaskSource, ctx: &MaskContext) -> Result<Vec<f32>> {
    let n = ctx.positions.len();
    let channel = |name: &str| -> Result<Vec<f32>> {
        let c = ctx.channels.get(name).ok_or_else(|| Error::MissingChannel(name.to_string()))?;
        if c.len() != n {
            return Err(Error::MissingChannel(format!("{name} has {} values, mesh has {n}", c.len())));
        }
        Ok(c.clone())
    };
    Ok(match src {
        MaskSource::Fill { value } => vec![value.clamp(0.0, 1.0); n],
        MaskSource::Channel { name } => channel(name)?,
        MaskSource::Mesh { attribute } => channel(attribute.channel_name())?,
        MaskSource::Noise(params) => {
            let noise = Noise::new(params);
            ctx.positions.par_iter().map(|&p| noise.sample(p)).collect()
        }
        MaskSource::Direction { axis, sharpness } => {
            let a = axis.normalize_or(Vec3::Y);
            ctx.normals.par_iter().map(|nrm| nrm.dot(a).clamp(0.0, 1.0).powf(sharpness.max(1e-3))).collect()
        }
        MaskSource::Gradient { axis, from, to } => {
            let a = axis.normalize_or(Vec3::Y);
            let span = if (to - from).abs() < 1e-9 { 1e-9 } else { to - from };
            ctx.positions.par_iter().map(|p| ((p.dot(a) - from) / span).clamp(0.0, 1.0)).collect()
        }
    })
}

/// Jacobi blur of a scalar field over vertex one-rings.
pub fn blur(values: &[f32], topo: &Topology, iterations: u32) -> Vec<f32> {
    let mut cur = values.to_vec();
    let mut next = vec![0.0; cur.len()];
    for _ in 0..iterations {
        next.par_iter_mut().enumerate().for_each(|(v, out)| {
            let ring = topo.vert_verts.row(v);
            let s: f32 = ring.iter().map(|&u| cur[u as usize]).sum();
            *out = (cur[v] + s) / (1 + ring.len()) as f32;
        });
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

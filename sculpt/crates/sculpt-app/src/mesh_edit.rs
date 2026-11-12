//! The layer panel's rows driving the mesh layers: the same `SculptInfo` / `MaskInfo` the volume's own sculpt layers
//! give, and the same edits (opacity 0-200%, blend, visibility, name, a paint mask and a noise mask per layer), each
//! a step of the one history.

use sculpt_core::generator::sculpt_layer::{LayerBlend as RowBlend, MaskInfo, MaskKind, NoiseMask, SculptInfo};
use sculpt_core::mask::{BlendMode, Levels, MaskLayer, MaskSource, MaskStack};
use sculpt_core::{LayerBlend, LayerId, LayerMeta, PaintTarget};

use crate::mesh_stack::{Entry, MeshStack};

/// How a row's blend shows on a mesh layer.
pub fn to_core(b: RowBlend) -> LayerBlend {
    match b {
        RowBlend::Add => LayerBlend::Add,
        RowBlend::Subtract => LayerBlend::Subtract,
        RowBlend::Raise => LayerBlend::Max,
        RowBlend::Carve => LayerBlend::Min,
    }
}

pub fn from_core(b: LayerBlend) -> RowBlend {
    match b {
        LayerBlend::Subtract => RowBlend::Subtract,
        LayerBlend::Max => RowBlend::Raise,
        LayerBlend::Min => RowBlend::Carve,
        LayerBlend::Add | LayerBlend::Normal => RowBlend::Add,
    }
}

/// A noise mask's look as levels on the raw noise: `balance` the middle, `contrast` how short the ramp.
fn noise_levels(n: &NoiseMask) -> Levels {
    let w = (1.0 - n.contrast).clamp(0.002, 1.0);
    Levels::range(n.balance - 0.5 * w, n.balance + 0.5 * w)
}

/// Layers holding more vertices than this move when a slider is let go, not every frame.
const LIVE_VERTICES: usize = 600_000;

impl MeshStack {
    pub fn layer_id(&self, k: usize) -> Option<LayerId> {
        self.layers.doc.layers().get(k).map(|l| l.id)
    }

    fn index_of(&self, id: LayerId) -> Option<usize> {
        self.layers.doc.layers().iter().position(|l| l.id == id)
    }

    /// The active layer's index.
    pub fn active(&self) -> Option<usize> {
        self.layers.doc.active_layer().and_then(|id| self.index_of(id))
    }

    /// The layers bottom first, as the rows show them, and the active one.
    pub fn infos(&self) -> Option<(usize, Vec<SculptInfo>)> {
        let doc = &self.layers.doc;
        if doc.layers().is_empty() {
            return None;
        }
        let leaf_len = (doc.vertex_count() / doc.bvh().leaves.len().max(1)).max(1);
        let out = doc
            .layers()
            .iter()
            .map(|l| SculptInfo {
                name: l.name.clone(),
                visible: l.visible,
                opacity: l.opacity,
                blend: from_core(l.blend),
                // (Vertices, roughly: the leaves its strokes reached.)
                blocks: l.allocated_leaves().len() * leaf_len,
                masks: l.mask.as_ref().map(|s| s.layers.iter().map(|m| self.mask_info(m)).collect()).unwrap_or_default(),
                active_mask: self.active_mask.get(&l.id).copied().flatten(),
            })
            .collect();
        Some((self.active().unwrap_or(0), out))
    }

    fn mask_info(&self, m: &MaskLayer) -> MaskInfo {
        let kind = match &m.source {
            MaskSource::Noise(params) => {
                let (lo, hi) = (m.levels.in_min, m.levels.in_max);
                MaskKind::Noise(NoiseMask { params: params.clone(), balance: 0.5 * (lo + hi), contrast: (1.0 - (hi - lo)).clamp(0.0, 0.99), invert: m.invert })
            }
            MaskSource::Channel { name } => MaskKind::Paint { fill: self.fills.get(name).copied().unwrap_or(1.0) },
            _ => MaskKind::Paint { fill: 1.0 },
        };
        MaskInfo { name: m.name.clone(), visible: m.enabled, opacity: m.opacity, blend: m.blend, kind }
    }

    /// A new empty layer on top, made active. (Not a step of the history: there is nothing on it to undo.)
    pub fn add_layer(&mut self) -> LayerId {
        self.close_open();
        let n = self.layers.doc.layers().len();
        let id = self.layers.doc.add_layer(&format!("Layer {}", n + 1));
        self.active_mask.insert(id, None);
        id
    }

    /// Remove layer `k` and what it holds. Its strokes can no longer be put back: the history goes.
    pub fn remove_layer(&mut self, k: usize) {
        self.close_open();
        let Some(id) = self.layer_id(k) else { return };
        if self.layers.doc.remove_layer(id).is_ok() {
            self.active_mask.remove(&id);
            self.clear_history();
            self.rows_all();
        }
    }

    pub fn set_active(&mut self, k: usize) {
        if let Some(id) = self.layer_id(k) {
            let _ = self.layers.doc.set_active_layer(Some(id));
        }
    }

    /// One change to layer `k`'s settings, one step.
    fn meta_step(&mut self, k: usize, f: impl FnOnce(&mut LayerMeta)) {
        self.close_open();
        let Some(id) = self.layer_id(k) else { return };
        let Some(before) = self.layers.doc.layer(id).map(|l| l.meta()) else { return };
        let mut after = before.clone();
        f(&mut after);
        if after == before {
            return;
        }
        if let Err(e) = self.layers.doc.set_layer_meta(id, after.clone(), false) {
            eprintln!("mesh layers: {e}");
            return;
        }
        self.push(Entry::Meta { id, before, after });
    }

    /// A change going on over frames (a slider dragged): one step once [`MeshStack::close_open`] is called.
    fn meta_drag(&mut self, k: usize, f: impl FnOnce(&mut sculpt_core::Document, LayerId) -> sculpt_core::Result<()>) {
        let Some(id) = self.layer_id(k) else { return };
        if self.open.as_ref().is_some_and(|o| o.0 != id) {
            self.close_open();
        }
        if self.open.is_none() {
            let Some(before) = self.layers.doc.layer(id).map(|l| l.meta()) else { return };
            self.open = Some((id, before));
        }
        if let Err(e) = f(&mut self.layers.doc, id) {
            eprintln!("mesh layers: {e}");
        }
    }

    pub fn rename(&mut self, k: usize, name: &str) {
        self.meta_step(k, |m| m.name = name.to_string());
    }

    pub fn set_visible(&mut self, k: usize, on: bool) {
        self.meta_step(k, |m| m.visible = on);
    }

    pub fn set_blend(&mut self, k: usize, b: RowBlend) {
        self.meta_step(k, |m| m.blend = to_core(b));
    }

    pub fn set_opacity(&mut self, k: usize, opacity: f32) {
        let o = opacity.clamp(0.0, sculpt_core::generator::sculpt_layer::MAX_OPACITY);
        self.meta_drag(k, |doc, id| doc.set_layer_opacity(id, o));
    }

    /// Whether layer `k` is small enough for a slider to move it every frame (else it moves when let go).
    pub fn live_edits(&self, k: usize) -> bool {
        let doc = &self.layers.doc;
        let Some(l) = doc.layers().get(k) else { return true };
        let leaf_len = doc.vertex_count() / doc.bvh().leaves.len().max(1);
        l.allocated_leaves().len() * leaf_len <= LIVE_VERTICES
    }

    // ------------------------------------------------------------------------------------------------------ masks

    /// The mask strokes paint on layer `k`, if one is picked.
    pub fn active_mask(&self, k: usize) -> Option<usize> {
        self.layer_id(k).and_then(|id| self.active_mask.get(&id).copied().flatten())
    }

    /// Pick layer `k`, and its mask `mi` or the layer itself. Returns what a picked painted mask was filled with.
    pub fn select_mask(&mut self, k: usize, mi: Option<usize>) -> Option<f32> {
        let id = self.layer_id(k)?;
        let _ = self.layers.doc.set_active_layer(Some(id));
        let n = self.layers.doc.layer(id).and_then(|l| l.mask.as_ref()).map_or(0, |s| s.layers.len());
        let mi = mi.filter(|&i| i < n);
        self.active_mask.insert(id, mi);
        let info = self.infos()?.1.get(k)?.masks.get(mi?)?.clone();
        Some(match info.kind {
            MaskKind::Paint { fill } => fill,
            MaskKind::Noise(_) => 1.0,
        })
    }

    /// The channel strokes paint: the active layer's picked mask, if it is painted.
    pub fn paint_channel(&self) -> Option<String> {
        let k = self.active()?;
        let mi = self.active_mask(k)?;
        let id = self.layer_id(k)?;
        match &self.layers.doc.layer(id)?.mask.as_ref()?.layers.get(mi)?.source {
            MaskSource::Channel { name } => Some(name.clone()),
            _ => None,
        }
    }

    /// A channel name no mask uses yet.
    fn new_channel(&self, id: LayerId) -> String {
        let doc = &self.layers.doc;
        (1..).map(|i| format!("mask.l{}.{i}", id.0)).find(|n| doc.channel(n).is_none()).unwrap()
    }

    /// Add a mask to the active layer, picked: at most one painted and one noise mask a layer (one of that kind
    /// there already is picked instead). Returns its index.
    pub fn add_mask(&mut self, kind: MaskKind) -> Option<usize> {
        let k = self.active()?;
        let id = self.layer_id(k)?;
        let stack = self.layers.doc.layer(id)?.mask.clone().unwrap_or_else(|| MaskStack::new(1.0));
        let paint = matches!(kind, MaskKind::Paint { .. });
        if let Some(i) = stack.layers.iter().position(|m| matches!(m.source, MaskSource::Channel { .. }) == paint) {
            self.select_mask(k, Some(i));
            return Some(i);
        }
        // The first mask sets the layer's look; one above it cuts into it.
        let blend = if stack.layers.is_empty() { BlendMode::Normal } else { BlendMode::Multiply };
        let layer = match kind {
            MaskKind::Paint { fill } => {
                let name = self.new_channel(id);
                let n = self.layers.doc.vertex_count();
                self.layers.doc.set_channel(&name, vec![fill; n]).ok()?;
                self.fills.insert(name.clone(), fill);
                MaskLayer::new("Paint", MaskSource::Channel { name }).blend(blend)
            }
            MaskKind::Noise(n) => {
                let mut m = MaskLayer::new("Noise", MaskSource::Noise(n.params.clone())).blend(blend).levels(noise_levels(&n));
                m.invert = n.invert;
                m
            }
        };
        let mut stack = stack;
        stack.layers.push(layer);
        let mi = stack.layers.len() - 1;
        self.meta_step(k, |m| m.mask = Some(stack));
        self.select_mask(k, Some(mi));
        Some(mi)
    }

    pub fn remove_mask(&mut self, k: usize, mi: usize) {
        let Some(id) = self.layer_id(k) else { return };
        let Some(mut stack) = self.layers.doc.layer(id).and_then(|l| l.mask.clone()) else { return };
        if mi >= stack.layers.len() {
            return;
        }
        stack.layers.remove(mi);
        let stack = (!stack.layers.is_empty()).then_some(stack);
        self.meta_step(k, |m| m.mask = stack);
        let a = self.active_mask.entry(id).or_default();
        *a = match *a {
            Some(x) if x == mi => None,
            Some(x) if x > mi => Some(x - 1),
            x => x,
        };
    }

    pub fn move_mask(&mut self, k: usize, mi: usize, up: bool) {
        let Some(id) = self.layer_id(k) else { return };
        let Some(mut stack) = self.layers.doc.layer(id).and_then(|l| l.mask.clone()) else { return };
        let to = if up { mi + 1 } else { mi.wrapping_sub(1) };
        if mi >= stack.layers.len() || to >= stack.layers.len() {
            return;
        }
        stack.layers.swap(mi, to);
        self.meta_step(k, |m| m.mask = Some(stack));
        let a = self.active_mask.entry(id).or_default();
        if *a == Some(mi) {
            *a = Some(to);
        } else if *a == Some(to) {
            *a = Some(mi);
        }
    }

    /// Change mask `mi` of layer `k` (a drag is one step).
    pub fn set_mask(&mut self, k: usize, mi: usize, info: MaskInfo) {
        let Some(id) = self.layer_id(k) else { return };
        let Some(mut stack) = self.layers.doc.layer(id).and_then(|l| l.mask.clone()) else { return };
        let Some(m) = stack.layers.get_mut(mi) else { return };
        m.name = info.name;
        m.enabled = info.visible;
        m.opacity = info.opacity.clamp(0.0, 1.0);
        m.blend = info.blend;
        match (&info.kind, &mut m.source) {
            (MaskKind::Noise(n), MaskSource::Noise(p)) => {
                *p = n.params.clone();
                m.levels = noise_levels(n);
                m.invert = n.invert;
            }
            (MaskKind::Paint { fill }, MaskSource::Channel { name }) => {
                self.fills.insert(name.clone(), *fill);
            }
            _ => {}
        }
        self.meta_drag(k, |doc, id| doc.set_layer_mask(id, Some(stack)));
    }

    /// Fill the active layer's picked painted mask (or its painted mask, one made if there is none) with `to`: a
    /// stroke of its own. Returns the mask's index.
    pub fn fill_mask(&mut self, to: f32) -> Option<usize> {
        self.close_open();
        let k = self.active()?;
        let id = self.layer_id(k)?;
        let painted = |s: &MeshStack| -> Option<(usize, String)> {
            let stack = s.layers.doc.layer(id)?.mask.as_ref()?;
            stack.layers.iter().enumerate().rev().find_map(|(i, m)| match &m.source {
                MaskSource::Channel { name } => Some((i, name.clone())),
                _ => None,
            })
        };
        let (mi, name) = match painted(self) {
            Some(x) => x,
            None => {
                self.add_mask(MaskKind::Paint { fill: 1.0 })?;
                painted(self)?
            }
        };
        let doc = &mut self.layers.doc;
        let b = doc.bounds();
        let (c, r) = ((b.min + b.max) * 0.5, (b.max - b.min).length() * 0.6 + 1.0);
        doc.begin_stroke("Fill mask");
        let flat = sculpt_core::brush::Falloff { hardness: 1.0 };
        // Twice: the second pass lands exactly on `to` wherever rounding left the first a hair off.
        for _ in 0..2 {
            doc.paint(&PaintTarget::Channel(name.clone()), c, r, &flat, 1.0, to, None);
        }
        doc.end_stroke();
        match self.layers.stroke_patch() {
            Ok(p) if !p.is_empty() => self.push(Entry::Layer(p)),
            Ok(_) => {}
            Err(e) => eprintln!("mesh layers: {e}"),
        }
        self.fills.insert(name, to);
        self.active_mask.insert(id, Some(mi));
        Some(mi)
    }
}

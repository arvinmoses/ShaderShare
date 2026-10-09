//! Mask actions: one-step presets, disable, invert, copy and paste, and per-op edits.
//!
//! Everything works on the editing copy of the active layer's mask (`app.mask_edit`) and marks it
//! dirty; `panels::apply_mask_edit` then writes it back as one undoable step once the pointer is up.

use sculpt_core::LayerId;
use sculpt_core::bake::MeshAttribute;
use sculpt_core::mask::MaskSource;
use sculpt_core::noise::{NoiseKind, NoiseParams};

use crate::app::{SculptApp, Selection};

/// What a mask preset builds. Each is one step: the mask exists and the new op is selected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preset {
    White,
    Black,
    Bake(MeshAttribute),
    Noise(NoiseKind),
    Paint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAction {
    /// Disable or enable the whole mask (Shift+click on its thumbnail).
    Toggle,
    /// Swap white and black.
    Invert,
    Copy,
    Paste,
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpAction {
    Toggle,
    Duplicate,
    /// Towards the top of the stack, which is up in the list.
    Raise,
    Lower,
    Delete,
}

pub const BAKES: [(MeshAttribute, &str); 4] = [
    (MeshAttribute::Curvature, "Curvature"),
    (MeshAttribute::Cavity, "Cavity"),
    (MeshAttribute::AmbientOcclusion, "Ambient occlusion"),
    (MeshAttribute::Thickness, "Thickness"),
];

pub const NOISES: [(NoiseKind, &str); 5] = [
    (NoiseKind::Fbm, "Clouds (fBm)"),
    (NoiseKind::Ridged, "Ridged"),
    (NoiseKind::Cellular, "Cellular"),
    (NoiseKind::Perlin, "Perlin"),
    (NoiseKind::Turbulence, "Turbulence"),
];

fn label_and_source(p: Preset) -> Option<(String, MaskSource)> {
    Some(match p {
        Preset::White | Preset::Black => return None,
        Preset::Bake(a) => {
            let name = BAKES.iter().find(|(b, _)| *b == a).map_or("Bake", |(_, n)| *n);
            (name.into(), MaskSource::Mesh { attribute: a })
        }
        Preset::Noise(k) => {
            let name = NOISES.iter().find(|(n, _)| *n == k).map_or("Noise", |(_, n)| *n);
            (name.into(), MaskSource::Noise(NoiseParams { kind: k, ..Default::default() }))
        }
        Preset::Paint => ("Paint".into(), MaskSource::Channel { name: String::new() }),
    })
}

/// Build a mask (or add an op to the existing one) on layer `id` and make it the paint target.
pub fn apply_preset(app: &mut SculptApp, id: LayerId, preset: Preset) {
    crate::panels::select_layer(app, Some(id), Selection::Layer);
    app.layers.selection.select_only(id);
    match label_and_source(preset) {
        None => {
            let base = if preset == Preset::White { 1.0 } else { 0.0 };
            if app.mask_edit.is_none() {
                crate::panels::add_mask(app, base);
            }
        }
        Some((label, source)) => {
            if app.mask_edit.is_none() {
                // Start from black so the op shows through.
                crate::panels::add_mask(app, 0.0);
            }
            crate::panels::add_effect(app, &label, source);
        }
    }
}

pub fn mask_action(app: &mut SculptApp, id: LayerId, action: MaskAction) {
    crate::panels::select_layer(app, Some(id), Selection::Layer);
    app.layers.selection.select_only(id);
    match action {
        MaskAction::Remove => {
            if let Some(d) = app.doc.as_mut()
                && let Some(l) = d.layer(id)
            {
                let mut m = l.meta();
                m.mask = None;
                let _ = d.set_layer_meta(id, m, false);
            }
            app.mask_edit = None;
            app.last_active = None;
            crate::panels::sync_mask_edit(app);
            app.selection = Selection::Layer;
            return;
        }
        MaskAction::Copy => {
            app.layers.mask_clip = app.mask_edit.as_ref().map(|(_, s)| s.clone());
            app.status = if app.layers.mask_clip.is_some() { "Mask copied".into() } else { "This layer has no mask".into() };
            return;
        }
        MaskAction::Paste => {
            let Some(clip) = app.layers.mask_clip.clone() else { return };
            app.mask_edit = Some((id, clip));
            app.expanded.insert(id);
            app.selection = Selection::Mask;
        }
        MaskAction::Toggle | MaskAction::Invert => {
            let Some((_, stack)) = app.mask_edit.as_mut() else { return };
            if action == MaskAction::Toggle {
                stack.enabled = !stack.enabled;
            } else {
                stack.base = 1.0 - stack.base;
            }
        }
    }
    app.mask_dirty = true;
}

/// Edit one op of the active layer's mask. `index` counts from the bottom of the mask stack.
pub fn op_action(app: &mut SculptApp, id: LayerId, index: usize, action: OpAction) {
    crate::panels::select_layer(app, Some(id), Selection::Effect(index));
    app.layers.selection.select_only(id);
    let Some((_, stack)) = app.mask_edit.as_mut() else { return };
    if index >= stack.layers.len() {
        return;
    }
    let select = match action {
        OpAction::Toggle => {
            stack.layers[index].enabled = !stack.layers[index].enabled;
            index
        }
        OpAction::Duplicate => {
            let mut copy = stack.layers[index].clone();
            copy.name = format!("{} copy", if copy.name.is_empty() { "Op" } else { &copy.name });
            stack.layers.insert(index + 1, copy);
            index + 1
        }
        OpAction::Raise if index + 1 < stack.layers.len() => {
            stack.layers.swap(index, index + 1);
            index + 1
        }
        OpAction::Lower if index > 0 => {
            stack.layers.swap(index, index - 1);
            index - 1
        }
        OpAction::Raise | OpAction::Lower => return,
        OpAction::Delete => {
            stack.layers.remove(index);
            app.selection = Selection::Mask;
            app.mask_dirty = true;
            return;
        }
    };
    app.selection = Selection::Effect(select);
    app.mask_dirty = true;
}

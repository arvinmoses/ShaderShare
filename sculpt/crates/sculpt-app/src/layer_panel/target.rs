//! What a stroke would change right now, and whether it would be refused.
//!
//! One answer feeds the viewport chip at the brush, the Properties breadcrumb, the status bar and the
//! guard that stops a stroke from starting, so they can never disagree.

use egui::Color32;
use sculpt_core::mask::{MaskSource, MaskStack};
use sculpt_core::{Document, LayerId};

use crate::app::{SculptApp, Selection};
use crate::theme::UiColors;
use crate::tools::Tool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// Sculpt offsets of a layer (or the base mesh).
    Delta,
    /// A mask's paint channel.
    Mask,
    Freeze,
}

#[derive(Clone, Debug)]
pub struct Target {
    pub kind: TargetKind,
    /// Path to the thing edited, e.g. `["Face", "Pores", "Mask", "Paint"]`.
    pub path: Vec<String>,
    /// Layer strength to show next to the name, when the target is a layer.
    pub strength: Option<f32>,
    /// Why a stroke would do nothing, in words fit for a status bar.
    pub refusal: Option<String>,
}

impl Target {
    pub fn color(&self, c: &UiColors) -> Color32 {
        if self.refusal.is_some() {
            return c.danger();
        }
        match self.kind {
            TargetKind::Delta => c.target_delta(),
            TargetKind::Mask => c.target_mask(),
            TargetKind::Freeze => c.target_freeze(),
        }
    }

    /// One short line: `Wrinkles › Mask › Paint`.
    pub fn text(&self) -> String {
        self.path.join(" › ")
    }
}

pub fn resolve(app: &SculptApp) -> Option<Target> {
    let doc = app.doc.as_ref()?;
    Some(decide(doc, app.tool, app.selection, app.layers.selection.primary(), app.mask_edit.as_ref()))
}

/// The decision itself, free of UI state so it can be tested against plain documents.
pub fn decide(doc: &Document, tool: Tool, selection: Selection, primary: Option<LayerId>, mask_edit: Option<&(LayerId, MaskStack)>) -> Target {
    if tool == Tool::Freeze {
        return Target { kind: TargetKind::Freeze, path: vec!["Freeze".into()], strength: None, refusal: None };
    }
    if tool == Tool::Pose {
        return Target { kind: TargetKind::Delta, path: vec!["Pose".into()], strength: None, refusal: None };
    }
    let selected = primary.and_then(|id| doc.layer(id));
    let active = doc.active_layer().and_then(|id| doc.layer(id));

    // Path of names from the top-level folder down to `layer`.
    let path_to = |layer: &sculpt_core::SculptLayer| {
        let mut path = vec![layer.name.clone()];
        let mut cur = layer.parent;
        while let Some(p) = cur.and_then(|id| doc.layer(id)) {
            path.push(p.name.clone());
            cur = p.parent;
        }
        path.reverse();
        path
    };

    if let Some(f) = selected.filter(|l| l.is_folder()) {
        return Target { kind: TargetKind::Delta, path: path_to(f), strength: Some(f.opacity), refusal: Some("A folder cannot be sculpted on: select a layer inside it".into()) };
    }
    let Some(layer) = active else {
        return Target { kind: TargetKind::Delta, path: vec!["Base".into()], strength: None, refusal: None };
    };
    let mut path = path_to(layer);

    if tool == Tool::MaskPaint {
        path.push("Mask".into());
        let ops = mask_edit.filter(|(id, _)| *id == layer.id).map(|(_, s)| &s.layers);
        let refusal = match (selection, ops) {
            (_, None) => Some("This layer has no mask. Use Mask ▾ › Hand-painted".to_string()),
            (Selection::Effect(i), Some(ops)) => match ops.get(i).map(|o| (&o.source, &o.name)) {
                Some((MaskSource::Channel { .. }, name)) => {
                    path.push(if name.is_empty() { "Paint".into() } else { name.clone() });
                    None
                }
                Some((_, name)) => Some(format!("'{}' is procedural. Select a Paint op to paint the mask", if name.is_empty() { "This op" } else { name })),
                None => Some("Select a Paint op to paint the mask".into()),
            },
            _ => Some("Select the mask's Paint op to paint it".to_string()),
        };
        return Target { kind: TargetKind::Mask, path, strength: Some(layer.opacity), refusal };
    }

    let refusal = if layer.locked {
        Some(format!("'{}' is locked", layer.name))
    } else if !layer.visible {
        Some(format!("'{}' is hidden", layer.name))
    } else if layer.effective_scale() == 0.0 && layer.opacity != 0.0 {
        Some(format!("'{}' is hidden by another layer's solo or a hidden folder", layer.name))
    } else {
        None
    };
    Target { kind: TargetKind::Delta, path, strength: Some(layer.opacity), refusal }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sculpt_core::mask::MaskLayer;
    use sculpt_core::noise::NoiseParams;
    use sculpt_core::primitives::quad_sphere;
    use sculpt_core::{LayerMeta, Placement};

    fn doc() -> Document {
        Document::from_mesh(quad_sphere(1, 1.0)).unwrap()
    }

    fn edit(doc: &mut Document, id: LayerId, f: impl FnOnce(&mut LayerMeta)) {
        let mut m = doc.layer(id).unwrap().meta();
        f(&mut m);
        doc.set_layer_meta(id, m, false).unwrap();
    }

    #[test]
    fn sculpt_tools_target_the_active_layer_and_path_includes_folders() {
        let mut d = doc();
        let f = d.insert_folder("Face", Placement::Top).unwrap();
        let l = d.insert_layer("Pores", Placement::Into(f)).unwrap();
        let t = decide(&d, Tool::ClayBuildup, Selection::Layer, Some(l), None);
        assert_eq!((t.kind, t.text(), t.refusal.is_none()), (TargetKind::Delta, "Face › Pores".to_string(), true));
        // Even with the mask selected, a sculpt tool still sculpts the layer.
        let t = decide(&d, Tool::Smooth, Selection::Mask, Some(l), None);
        assert_eq!(t.kind, TargetKind::Delta);
    }

    #[test]
    fn base_is_the_target_without_a_layer_and_pose_never_refuses() {
        let d = doc();
        let t = decide(&d, Tool::ClayBuildup, Selection::Layer, None, None);
        assert_eq!(t.text(), "Base");
        assert!(t.refusal.is_none());
        let mut d = doc();
        let l = d.insert_layer("L", Placement::Top).unwrap();
        edit(&mut d, l, |m| m.locked = true);
        assert!(decide(&d, Tool::Pose, Selection::Layer, Some(l), None).refusal.is_none());
    }

    #[test]
    fn locked_hidden_and_soloed_out_layers_refuse() {
        let mut d = doc();
        let a = d.insert_layer("A", Placement::Top).unwrap();
        let b = d.insert_layer("B", Placement::Top).unwrap();
        d.set_active_layer(Some(a)).unwrap();
        edit(&mut d, a, |m| m.locked = true);
        assert!(decide(&d, Tool::ClayBuildup, Selection::Layer, Some(a), None).refusal.unwrap().contains("locked"));
        edit(&mut d, a, |m| {
            m.locked = false;
            m.visible = false;
        });
        assert!(decide(&d, Tool::ClayBuildup, Selection::Layer, Some(a), None).refusal.unwrap().contains("hidden"));
        edit(&mut d, a, |m| m.visible = true);
        d.set_solo(Some(b)).unwrap();
        assert!(decide(&d, Tool::ClayBuildup, Selection::Layer, Some(a), None).refusal.unwrap().contains("solo"));
        d.set_solo(None).unwrap();
        assert!(decide(&d, Tool::ClayBuildup, Selection::Layer, Some(a), None).refusal.is_none());
    }

    #[test]
    fn selected_folder_refuses_sculpting() {
        let mut d = doc();
        let f = d.insert_folder("F", Placement::Top).unwrap();
        d.insert_layer("L", Placement::Into(f)).unwrap();
        let t = decide(&d, Tool::ClayBuildup, Selection::Layer, Some(f), None);
        assert!(t.refusal.unwrap().contains("folder"));
    }

    #[test]
    fn mask_paint_needs_a_selected_paint_op() {
        let mut d = doc();
        let l = d.insert_layer("Wrinkles", Placement::Top).unwrap();
        // No mask at all.
        assert!(decide(&d, Tool::MaskPaint, Selection::Layer, Some(l), None).refusal.unwrap().contains("no mask"));
        let stack = MaskStack::new(0.0)
            .with(MaskLayer::new("Breakup", MaskSource::Noise(NoiseParams::default())))
            .with(MaskLayer::new("Hand", MaskSource::Channel { name: "paint.x".into() }));
        let edit_copy = (l, stack);
        // Layer selected, not the op.
        assert!(decide(&d, Tool::MaskPaint, Selection::Layer, Some(l), Some(&edit_copy)).refusal.is_some());
        // A procedural op.
        let t = decide(&d, Tool::MaskPaint, Selection::Effect(0), Some(l), Some(&edit_copy));
        assert!(t.refusal.unwrap().contains("procedural"));
        // The Paint op: allowed, purple, and the path says so.
        let t = decide(&d, Tool::MaskPaint, Selection::Effect(1), Some(l), Some(&edit_copy));
        assert_eq!((t.kind, t.text(), t.refusal.is_none()), (TargetKind::Mask, "Wrinkles › Mask › Hand".to_string(), true));
    }

    #[test]
    fn freeze_tool_targets_freeze() {
        let d = doc();
        assert_eq!(decide(&d, Tool::Freeze, Selection::Layer, None, None).kind, TargetKind::Freeze);
    }
}

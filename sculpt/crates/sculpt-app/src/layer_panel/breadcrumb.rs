//! The paint target, spelled out: `Wrinkles › Mask › Breakup`, coloured like the target frame.

use egui::{Color32, RichText, Ui};

use crate::app::{SculptApp, Selection};

pub struct Crumbs {
    pub parts: Vec<String>,
    pub color: Color32,
    /// What a stroke would change, in plain words.
    pub verb: &'static str,
}

/// Path to the selected thing, and the colour of what a stroke would edit.
pub fn crumbs(app: &SculptApp) -> Option<Crumbs> {
    let doc = app.doc.as_ref()?;
    let colors = &app.theme.ui;
    let primary = app.layers.selection.primary().or_else(|| doc.active_layer());
    let Some(layer) = primary.and_then(|id| doc.layer(id)) else {
        return Some(Crumbs { parts: vec!["Base".into()], color: colors.target_delta(), verb: "sculpts the base mesh" });
    };
    let mut parts: Vec<String> = Vec::new();
    let mut cur = layer.parent;
    while let Some(p) = cur.and_then(|id| doc.layer(id)) {
        parts.push(p.name.clone());
        cur = p.parent;
    }
    parts.reverse();
    parts.push(layer.name.clone());
    if layer.is_folder() {
        return Some(Crumbs { parts, color: colors.text_weak.0, verb: "folders cannot be sculpted on" });
    }
    // Mask and op selection only mean something on the active layer.
    let sel = if doc.active_layer() == Some(layer.id) { app.selection } else { Selection::Layer };
    Some(match sel {
        Selection::Layer => Crumbs { parts, color: colors.target_delta(), verb: "strokes sculpt this layer" },
        Selection::Mask => {
            parts.push("Mask".into());
            Crumbs { parts, color: colors.target_mask(), verb: "Mask Paint edits this mask" }
        }
        Selection::Effect(i) => {
            parts.push("Mask".into());
            let name = app.mask_edit.as_ref().and_then(|(_, s)| s.layers.get(i)).map(|l| if l.name.is_empty() { "Op".to_string() } else { l.name.clone() });
            parts.push(name.unwrap_or_else(|| "Op".into()));
            Crumbs { parts, color: colors.target_mask(), verb: "Mask Paint edits this op" }
        }
    })
}

pub fn show(app: &SculptApp, ui: &mut Ui) {
    let Some(c) = crumbs(app) else { return };
    let fs = app.theme.metrics.font_size;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
        ui.painter().circle_filled(dot.center(), 4.0, c.color);
        for (i, p) in c.parts.iter().enumerate() {
            if i > 0 {
                ui.label(RichText::new("›").color(app.theme.weak_text()));
            }
            let last = i + 1 == c.parts.len();
            let t = RichText::new(p).size(fs);
            ui.label(if last { t.strong() } else { t.color(app.theme.weak_text()) });
        }
    });
    ui.label(RichText::new(c.verb).size(fs * 0.85).color(app.theme.weak_text()));
}

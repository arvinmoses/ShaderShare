//! The paint target, spelled out: `Wrinkles › Mask › Breakup`, coloured like the target frame.

use egui::{Color32, RichText, Ui};

use crate::app::SculptApp;

pub struct Crumbs {
    pub parts: Vec<String>,
    pub color: Color32,
    /// What a stroke would change, or why it would not.
    pub verb: String,
}

/// Path to the paint target for the current tool, and the colour of what a stroke would edit.
pub fn crumbs(app: &SculptApp) -> Option<Crumbs> {
    let n = app.layers.selection.len();
    if n > 1 {
        return Some(Crumbs { parts: vec![format!("{n} layers selected")], color: app.theme.weak_text(), verb: "changes apply to all of them".into() });
    }
    let t = super::target::resolve(app)?;
    let verb = match (&t.refusal, t.kind) {
        (Some(why), _) => format!("⊘ {why}"),
        (None, super::target::TargetKind::Delta) => "strokes sculpt this".into(),
        (None, super::target::TargetKind::Mask) => "Mask Paint edits this".into(),
        (None, super::target::TargetKind::Freeze) => "strokes freeze the surface".into(),
    };
    Some(Crumbs { color: t.color(&app.theme.ui), parts: t.path, verb })
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
    let hint_color = if c.verb.starts_with('⊘') { app.theme.ui.danger() } else { app.theme.weak_text() };
    ui.add(egui::Label::new(RichText::new(&c.verb).size(fs * 0.85).color(hint_color)).wrap());
}

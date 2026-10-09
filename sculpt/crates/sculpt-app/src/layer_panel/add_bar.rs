//! The strip of add/edit buttons directly under the layer list, so the pointer
//! never travels far from the row it just touched. Everything inserts above
//! the selection (see `command::insert_point`).

use egui::{Align2, FontId, Rect, Sense, Ui, vec2};

use super::command::{self, LayerCommand};
use crate::app::SculptApp;
use crate::icons::Icon;
use crate::keymap::Command;

pub const BAR_H: f32 = 30.0;
/// Extra height below the list for item spacing around the bar.
pub const SLACK: f32 = 12.0;

/// Icon plus short label; `menu` adds a drop-down chevron.
struct BarButton {
    icon: Icon,
    label: &'static str,
    tip: &'static str,
    key: Option<Command>,
    enabled: bool,
    menu: bool,
}

impl BarButton {
    fn new(icon: Icon, label: &'static str, tip: &'static str) -> BarButton {
        BarButton { icon, label, tip, key: None, enabled: true, menu: false }
    }
    fn key(mut self, k: Command) -> BarButton {
        self.key = Some(k);
        self
    }
    fn enabled(mut self, on: bool) -> BarButton {
        self.enabled = on;
        self
    }
    fn menu(mut self) -> BarButton {
        self.menu = true;
        self
    }
    fn show(self, ui: &mut Ui, app: &SculptApp) -> egui::Response {
        let font = FontId::proportional(app.theme.metrics.font_size * 0.92);
        let text_w = ui.painter().layout_no_wrap(self.label.to_owned(), font.clone(), egui::Color32::WHITE).size().x;
        let gap = if self.label.is_empty() { 0.0 } else { 4.0 };
        let w = 4.0 + 16.0 + gap + text_w + if self.menu { 12.0 } else { 0.0 } + 6.0;
        let (rect, resp) = ui.allocate_exact_size(vec2(w, BAR_H - 6.0), if self.enabled { Sense::click() } else { Sense::hover() });
        let v = ui.visuals();
        if resp.hovered() && self.enabled {
            ui.painter().rect_filled(rect, 3.0, v.widgets.hovered.bg_fill);
        }
        let color = if !self.enabled { app.theme.weak_text().gamma_multiply(0.6) } else if resp.hovered() { v.strong_text_color() } else { v.text_color() };
        self.icon.paint(ui.painter(), Rect::from_min_size(rect.left_center() + vec2(4.0, -8.0), vec2(16.0, 16.0)), color);
        ui.painter().text(rect.left_center() + vec2(24.0, 0.0), Align2::LEFT_CENTER, self.label, font, color);
        if self.menu {
            Icon::ChevronDown.paint(ui.painter(), Rect::from_center_size(rect.right_center() - vec2(10.0, 0.0), vec2(12.0, 12.0)), color);
        }
        let mut tip = self.tip.to_owned();
        if let Some(s) = self.key.and_then(|k| app.keymap.shortcut_text(ui.ctx(), k)) {
            tip = format!("{tip} ({s})");
        }
        resp.on_hover_text(tip)
    }
}

pub fn show(app: &mut SculptApp, ui: &mut Ui) {
    let has_sel = app.layers.selection.primary().is_some() || app.doc.as_ref().is_some_and(|d| d.active_layer().is_some());
    let active = app.doc.as_ref().and_then(|d| d.active_layer());
    ui.add_space(3.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 1.0;
        ui.add_space(2.0);
        if BarButton::new(Icon::Plus, "Layer", "New sculpt layer above the selection").key(Command::NewLayer).show(ui, app).clicked() {
            command::execute(app, LayerCommand::NewLayer);
        }
        if BarButton::new(Icon::Folder, "Folder", "New folder").key(Command::NewFolder).show(ui, app).clicked() {
            command::execute(app, LayerCommand::NewFolder);
        }
        let mask = BarButton::new(Icon::Mask, "Mask", if active.is_some() { "Add a mask in one step, or edit this layer's mask" } else { "Select a sculpt layer to add a mask (folders and Base cannot have one)" }).enabled(active.is_some()).menu().show(ui, app);
        if let Some(id) = active {
            super::menus::add_mask_menu_on(app, &mask, id);
        }
        let op = BarButton::new(Icon::Effect, "Op", if active.is_some() { "Add an op to the mask" } else { "Select a sculpt layer first" }).enabled(active.is_some()).menu().show(ui, app);
        egui::Popup::menu(&op).show(|ui| crate::panels::effect_menu(app, ui));
        if BarButton::new(Icon::Duplicate, "", "Duplicate").key(Command::LayerDuplicate).enabled(has_sel).show(ui, app).clicked() {
            command::execute(app, LayerCommand::Duplicate);
        }
        if BarButton::new(Icon::Merge, "", "Merge down").key(Command::LayerMergeDown).enabled(has_sel).show(ui, app).clicked() {
            command::execute(app, LayerCommand::MergeDown);
        }
        if BarButton::new(Icon::Trash, "", "Delete").key(Command::LayerDelete).enabled(has_sel).show(ui, app).clicked() {
            command::execute(app, LayerCommand::Delete);
        }
    });
}

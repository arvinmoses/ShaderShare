//! Radial menu at the cursor (Q): the handful of layer actions you reach for mid-stroke, arranged
//! around the pointer so every one is a short move away. Click an item, Esc or click elsewhere to close.

use egui::{Align2, Color32, Context, FontId, Id, Order, Pos2, Rect, Sense, Stroke, vec2};

use super::command::{self, LayerCommand, MetaEdit};
use super::mask_ops::{MaskAction, Preset};
use crate::app::SculptApp;
use crate::keymap::Command;

const RADIUS: f32 = 82.0;
const ITEM: egui::Vec2 = vec2(86.0, 30.0);

/// What a radial item does when picked.
#[derive(Clone, Debug, PartialEq)]
enum Pick {
    Layer(LayerCommand),
    Ui(Command),
}

struct Item {
    label: String,
    hint: &'static str,
    pick: Pick,
    enabled: bool,
}

/// Where item `i` of `n` sits: first straight up, then clockwise.
pub fn slot(center: Pos2, i: usize, n: usize, radius: f32) -> Pos2 {
    let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * i as f32 / n as f32;
    center + vec2(a.cos(), a.sin()) * radius
}

pub fn toggle(app: &mut SculptApp, ctx: &Context) {
    let at = ctx.input(|i| i.pointer.hover_pos()).unwrap_or_else(|| ctx.content_rect().center());
    app.layers.radial = if app.layers.radial.is_some() { None } else { Some(at) };
}

fn items(app: &SculptApp) -> Vec<Item> {
    let doc = app.doc.as_ref();
    let active = doc.and_then(|d| d.active_layer());
    let layer = active.and_then(|id| doc.and_then(|d| d.layer(id)));
    let has_mask = app.mask_edit.as_ref().is_some_and(|(i, _)| Some(*i) == active);
    let (visible, locked) = layer.map_or((true, false), |l| (l.visible, l.locked));
    let id = active;
    let on_layer = |cmd: Option<LayerCommand>| cmd.map(Pick::Layer);
    let mask_pick = id.map(|id| {
        if has_mask {
            LayerCommand::Mask { id, action: MaskAction::Toggle }
        } else {
            LayerCommand::AddMask { id, preset: Preset::Black }
        }
    });
    vec![
        Item { label: "New layer".into(), hint: "Ctrl+L", pick: Pick::Layer(LayerCommand::NewLayer), enabled: true },
        Item { label: (if has_mask { "Mask on/off" } else { "Add mask" }).into(), hint: "Shift+M", pick: on_layer(mask_pick.clone()).unwrap_or(Pick::Ui(Command::ToggleTarget)), enabled: id.is_some() },
        Item { label: "Layer/Mask".into(), hint: "M", pick: Pick::Ui(Command::ToggleTarget), enabled: has_mask },
        Item { label: "View mask".into(), hint: "Alt+M", pick: Pick::Ui(Command::ViewMask), enabled: has_mask },
        Item { label: "Duplicate".into(), hint: "Ctrl+D", pick: Pick::Layer(LayerCommand::Duplicate), enabled: id.is_some() },
        Item { label: "Solo".into(), hint: "S", pick: id.map_or(Pick::Ui(Command::LayerSolo), |id| Pick::Layer(LayerCommand::ToggleSolo(id))), enabled: id.is_some() },
        Item { label: (if visible { "Hide" } else { "Show" }).into(), hint: "Shift+H", pick: id.map_or(Pick::Ui(Command::LayerHide), |id| Pick::Layer(LayerCommand::Edit { id, edit: MetaEdit::Visible(!visible), coalesce: false })), enabled: id.is_some() },
        Item { label: (if locked { "Unlock" } else { "Lock" }).into(), hint: "Shift+L", pick: id.map_or(Pick::Ui(Command::LayerLock), |id| Pick::Layer(LayerCommand::Edit { id, edit: MetaEdit::Locked(!locked), coalesce: false })), enabled: id.is_some() },
    ]
}

pub fn show(app: &mut SculptApp, ctx: &Context) {
    let Some(center) = app.layers.radial else { return };
    let items = items(app);
    let mut picked: Option<Pick> = None;
    let mut close = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    let accent = app.theme.ui.accent.0;
    let weak = app.theme.weak_text();
    let pointer = ctx.input(|i| i.pointer.hover_pos());

    egui::Area::new(Id::new("layer_radial")).order(Order::Foreground).fixed_pos(center - vec2(RADIUS + ITEM.x, RADIUS + ITEM.y)).show(ctx, |ui| {
        let size = vec2((RADIUS + ITEM.x) * 2.0, (RADIUS + ITEM.y) * 2.0);
        let (frame, _) = ui.allocate_exact_size(size, Sense::hover());
        let c = frame.center();
        let p = ui.painter();
        p.circle_filled(c, RADIUS + 10.0, Color32::from_black_alpha(90));
        p.circle_stroke(c, RADIUS + 10.0, Stroke::new(1.0, weak.gamma_multiply(0.5)));
        p.circle_filled(c, 5.0, accent);
        for (n, item) in items.iter().enumerate() {
            let rect = Rect::from_center_size(slot(c, n, items.len(), RADIUS), ITEM);
            let resp = ui.interact(rect, Id::new(("radial", n)), if item.enabled { Sense::click() } else { Sense::hover() });
            let hot = resp.hovered() && item.enabled;
            let fill = if hot { accent } else { Color32::from_black_alpha(215) };
            p.rect_filled(rect, 15.0, fill);
            p.rect_stroke(rect, 15.0, Stroke::new(1.0, accent.gamma_multiply(if item.enabled { 0.9 } else { 0.3 })), egui::StrokeKind::Inside);
            let text = if item.enabled { Color32::WHITE } else { weak.gamma_multiply(0.7) };
            p.text(rect.center(), Align2::CENTER_CENTER, &item.label, FontId::proportional(12.5), text);
            if item.enabled {
                resp.clone().on_hover_text(item.hint);
            }
            if resp.clicked() {
                picked = Some(item.pick.clone());
            }
        }
    });
    if let (Some(p), None) = (pointer, &picked) {
        let far = (p - center).length() > RADIUS + ITEM.x + 70.0;
        if far || ctx.input(|i| i.pointer.any_click() && (i.pointer.interact_pos().unwrap_or(p) - center).length() < 14.0) {
            close = true;
        }
    }
    if let Some(pick) = picked {
        app.layers.radial = None;
        match pick {
            Pick::Layer(cmd) => command::execute(app, cmd),
            Pick::Ui(cmd) => {
                super::switcher::run(app, ctx, cmd);
            }
        }
    } else if close {
        app.layers.radial = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_start_at_the_top_and_run_clockwise() {
        let c = Pos2::new(100.0, 100.0);
        let top = slot(c, 0, 8, 50.0);
        assert!((top - Pos2::new(100.0, 50.0)).length() < 1e-3);
        let right = slot(c, 2, 8, 50.0);
        assert!((right - Pos2::new(150.0, 100.0)).length() < 1e-3);
        let bottom = slot(c, 4, 8, 50.0);
        assert!((bottom - Pos2::new(100.0, 150.0)).length() < 1e-3);
    }

    #[test]
    fn every_slot_is_the_same_distance_from_the_centre() {
        let c = Pos2::new(0.0, 0.0);
        for i in 0..8 {
            assert!(((slot(c, i, 8, 80.0) - c).length() - 80.0).abs() < 1e-3);
        }
    }
}

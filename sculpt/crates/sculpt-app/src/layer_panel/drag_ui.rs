//! Drawing and finishing a row drag: insertion line, folder outline, refusal, ghost, auto-scroll, spring-open.

use std::time::{Duration, Instant};

use egui::{Align2, Color32, FontId, Key, Rect, Stroke, StrokeKind, Ui, vec2};

use super::command::{self, LayerCommand, MetaEdit};
use super::dragdrop::{self, DropSpot, RowGeom, Zone, find_spot};
use crate::app::SculptApp;

const SPRING_OPEN: Duration = Duration::from_millis(600);
const EDGE: f32 = 26.0;

/// Call after the rows are drawn. `list` is the visible part of the list.
pub fn update(app: &mut SculptApp, ui: &mut Ui, rows: &[RowGeom], list: Rect) {
    let Some(drag) = app.layers.drag.clone() else { return };
    let ctx = ui.ctx().clone();
    let pointer = app.layers.pointer_override.or_else(|| ctx.input(|i| i.pointer.interact_pos()));
    let (released, escape, copy) = ctx.input(|i| (i.pointer.any_released(), i.key_pressed(Key::Escape), i.modifiers.command));
    let copy = copy || app.layers.copy_override;
    let released = released && app.layers.pointer_override.is_none();
    if escape {
        app.layers.drag = None;
        app.layers.spring = None;
        return;
    }
    ctx.request_repaint_after(Duration::from_millis(50));

    let spot = pointer.filter(|p| list.contains(*p)).and_then(|p| app.doc.as_ref().and_then(|d| find_spot(d, rows, &drag.ids, p)));
    if let Some(p) = pointer {
        autoscroll(ui, list, p);
    }
    paint(app, ui, list, spot, &drag.ids, pointer, copy);
    spring_open(app, spot);

    if released {
        app.layers.drag = None;
        app.layers.spring = None;
        if let Some(s) = spot.filter(|s| s.valid) {
            let at = dragdrop::placement(s.target, s.zone);
            command::execute(app, LayerCommand::Move { ids: drag.ids, at, copy });
        } else if spot.is_some() {
            app.status = "Cannot drop a folder into itself".into();
        }
    }
}

fn autoscroll(ui: &mut Ui, list: Rect, p: egui::Pos2) {
    let speed = if p.y < list.top() + EDGE && p.y > list.top() - EDGE {
        (list.top() + EDGE - p.y) / EDGE * 14.0
    } else if p.y > list.bottom() - EDGE && p.y < list.bottom() + EDGE {
        -(p.y - (list.bottom() - EDGE)) / EDGE * 14.0
    } else {
        0.0
    };
    if speed != 0.0 {
        ui.scroll_with_delta(vec2(0.0, speed));
    }
}

/// A collapsed folder under the drag opens after it has been hovered for a moment.
fn spring_open(app: &mut SculptApp, spot: Option<DropSpot>) {
    let hovered_folder = spot.filter(|s| s.zone == Zone::Into && s.valid).map(|s| s.target);
    let Some(folder) = hovered_folder else {
        app.layers.spring = None;
        return;
    };
    let collapsed = app.doc.as_ref().and_then(|d| d.layer(folder)).is_some_and(|l| l.collapsed);
    match app.layers.spring {
        Some((id, since)) if id == folder => {
            if collapsed && since.elapsed() >= SPRING_OPEN {
                command::execute(app, LayerCommand::Edit { id: folder, edit: MetaEdit::Collapsed(false), coalesce: true });
                app.layers.spring = None;
            }
        }
        _ => app.layers.spring = Some((folder, Instant::now())),
    }
}

fn paint(app: &SculptApp, ui: &Ui, list: Rect, spot: Option<DropSpot>, ids: &[sculpt_core::LayerId], pointer: Option<egui::Pos2>, copy: bool) {
    let painter = ui.painter_at(list);
    let colors = &app.theme.ui;
    if let Some(s) = spot {
        let color = if s.valid { colors.accent.0 } else { colors.danger() };
        match s.zone {
            Zone::Into => {
                painter.rect_stroke(s.rect.shrink(1.0), 3.0, Stroke::new(2.0, color), StrokeKind::Inside);
            }
            Zone::Above | Zone::Below => {
                let y = if s.zone == Zone::Above { s.rect.top() } else { s.rect.bottom() };
                painter.line_segment([egui::pos2(s.rect.left() + 4.0, y), egui::pos2(s.rect.right() - 4.0, y)], Stroke::new(2.5, color));
                painter.circle_filled(egui::pos2(s.rect.left() + 4.0, y), 3.5, color);
            }
        }
    }
    // Ghost row at the pointer: first name plus a count.
    if let (Some(p), Some(doc)) = (pointer, app.doc.as_ref()) {
        let first = ids.first().and_then(|id| doc.layer(*id)).map(|l| l.name.clone()).unwrap_or_default();
        let text = match (ids.len(), copy) {
            (1, false) => first,
            (1, true) => format!("{first}  (copy)"),
            (n, false) => format!("{first}  +{}", n - 1),
            (n, true) => format!("{first}  +{}  (copy)", n - 1),
        };
        let font = FontId::proportional(app.theme.metrics.font_size);
        let galley = ui.painter().layout_no_wrap(text, font, Color32::WHITE);
        let size = galley.size() + vec2(16.0, 8.0);
        let rect = Rect::from_min_size(p + vec2(12.0, -size.y / 2.0), size);
        let layer_painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("layer_drag_ghost")));
        layer_painter.rect_filled(rect, 4.0, colors.widget_active.0.gamma_multiply(0.85));
        layer_painter.text(rect.left_center() + vec2(8.0, 0.0), Align2::LEFT_CENTER, galley.text(), FontId::proportional(app.theme.metrics.font_size), Color32::WHITE);
    }
}

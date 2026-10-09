//! Tab quick-switcher: a small copy of the layer list at the cursor, so changing layers while
//! sculpting never means reaching for the side panel. Type to filter, Up/Down and Enter to pick.

use egui::{Align2, Context, Id, Key, Modifiers, Order, Pos2, RichText, Sense, vec2};
use sculpt_core::LayerId;

use crate::app::{SculptApp, Selection};
use crate::keymap::Command;
use crate::viewport::OverlayKind;

pub struct Switcher {
    pub at: Pos2,
    pub filter: String,
    pub cursor: usize,
}

/// Keymap commands that act on the UI rather than the document. True when handled.
pub fn run(app: &mut SculptApp, ctx: &Context, cmd: Command) -> bool {
    match cmd {
        Command::LayerSwitcher => {
            let at = ctx.input(|i| i.pointer.hover_pos()).unwrap_or_else(|| ctx.content_rect().center());
            app.layers.switcher = if app.layers.switcher.is_some() { None } else { Some(Switcher { at, filter: String::new(), cursor: 0 }) };
            true
        }
        Command::ToggleTarget => {
            toggle_target(app);
            true
        }
        Command::ViewMask => {
            app.overlay = if app.overlay == OverlayKind::LayerMask { OverlayKind::None } else { OverlayKind::LayerMask };
            true
        }
        _ => false,
    }
}

/// M: flip the paint target between the layer's sculpt data and its mask.
fn toggle_target(app: &mut SculptApp) {
    let Some(id) = app.doc.as_ref().and_then(|d| d.active_layer()) else { return };
    let has_mask = app.mask_edit.as_ref().is_some_and(|(i, _)| *i == id);
    if !has_mask {
        app.status = "This layer has no mask".into();
        return;
    }
    let next = if app.selection == Selection::Layer { Selection::Mask } else { Selection::Layer };
    crate::panels::select_layer(app, Some(id), next);
}

pub fn show(app: &mut SculptApp, ctx: &Context) {
    let Some(sw) = app.layers.switcher.as_mut() else { return };
    let Some(doc) = app.doc.as_ref() else { return };
    let tree = doc.layer_tree();
    let filter = sw.filter.to_lowercase();
    let rows: Vec<(LayerId, usize, String, f32, bool, bool)> = tree
        .iter()
        .filter_map(|r| doc.layer(r.id).map(|l| (l.id, r.depth, l.name.clone(), l.opacity, l.visible, l.is_folder())))
        .filter(|(_, _, name, ..)| filter.is_empty() || name.to_lowercase().contains(&filter))
        .collect();
    let current = doc.active_layer();

    let (down, up, enter, esc) = ctx.input_mut(|i| {
        (i.consume_key(Modifiers::NONE, Key::ArrowDown), i.consume_key(Modifiers::NONE, Key::ArrowUp), i.consume_key(Modifiers::NONE, Key::Enter), i.consume_key(Modifiers::NONE, Key::Escape))
    });
    if down && sw.cursor + 1 < rows.len() {
        sw.cursor += 1;
    }
    if up {
        sw.cursor = sw.cursor.saturating_sub(1);
    }
    sw.cursor = sw.cursor.min(rows.len().saturating_sub(1));

    let mut picked: Option<LayerId> = if enter { rows.get(sw.cursor).map(|r| r.0) } else { None };
    let accent = app.theme.ui.accent.0;
    let weak = app.theme.weak_text();
    let at = sw.at;
    let area = egui::Area::new(Id::new("layer_switcher")).order(Order::Foreground).fixed_pos(at - vec2(20.0, 14.0)).constrain(true).show(ctx, |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            ui.set_width(250.0);
            let te = ui.add(egui::TextEdit::singleline(&mut sw.filter).hint_text("Switch layer…").desired_width(f32::INFINITY));
            if !te.has_focus() {
                te.request_focus();
            }
            ui.add_space(2.0);
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for (n, (id, depth, name, strength, visible, folder)) in rows.iter().enumerate() {
                    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
                    if n == sw.cursor || resp.hovered() {
                        ui.painter().rect_filled(rect, 3.0, ui.visuals().widgets.hovered.bg_fill);
                    }
                    if Some(*id) == current {
                        ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0, accent), egui::StrokeKind::Inside);
                    }
                    let tint = if *visible { ui.visuals().text_color() } else { weak };
                    let x = rect.left() + 8.0 + *depth as f32 * 12.0;
                    let mut text_x = x;
                    if *folder {
                        crate::icons::Icon::Folder.paint(ui.painter(), egui::Rect::from_center_size(egui::pos2(x + 7.0, rect.center().y), vec2(14.0, 14.0)), weak);
                        text_x += 18.0;
                    }
                    ui.painter().text(egui::pos2(text_x, rect.center().y), Align2::LEFT_CENTER, name, egui::FontId::proportional(13.0), tint);
                    ui.painter().text(egui::pos2(rect.right() - 8.0, rect.center().y), Align2::RIGHT_CENTER, format!("{:.0}%", strength * 100.0), egui::FontId::proportional(11.0), weak);
                    if resp.clicked() {
                        picked = Some(*id);
                    }
                }
                if rows.is_empty() {
                    ui.label(RichText::new("No match").color(weak));
                }
            });
        });
    });
    let outside = ctx.input(|i| i.pointer.hover_pos()).is_some_and(|p| !area.response.rect.expand(80.0).contains(p));
    let clicked_away = ctx.input(|i| i.pointer.any_click()) && ctx.input(|i| i.pointer.hover_pos()).is_some_and(|p| !area.response.rect.contains(p));
    if let Some(id) = picked {
        app.layers.switcher = None;
        let folder = app.doc.as_ref().and_then(|d| d.layer(id)).is_some_and(|l| l.is_folder());
        app.layers.selection.select_only(id);
        if !folder {
            crate::panels::select_layer(app, Some(id), Selection::Layer);
        }
    } else if esc || outside || clicked_away {
        app.layers.switcher = None;
    }
}

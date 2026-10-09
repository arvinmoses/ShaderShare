//! One layer or folder row: eye, disclosure, content thumbnail, mask thumbnail,
//! name, and a right-aligned cluster of solo, lock and strength.
//!
//! The row is laid out by rectangle arithmetic (not nested layouts) so every
//! hit area is exact and the cost per row stays tiny.

use egui::{Align2, Color32, CursorIcon, FontId, Key, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, UiBuilder, vec2};
use sculpt_core::LayerId;

use super::command::{self, LayerCommand, MetaEdit};
use super::dragdrop::RowGeom;
use super::menus;
use super::selection::ClickMods;
use super::state::DragState;
use super::tree::Node;
use crate::app::{SculptApp, Selection};
use crate::icons::Icon;
use crate::theme::bold;
use crate::viewport::OverlayKind;

pub const ROW_H: f32 = 36.0;
pub const INDENT: f32 = 14.0;
const THUMB: f32 = 26.0;
const SMALL: f32 = 16.0;

/// Cut `text` to `max` pixels, keeping both ends ("Wrinkle…_v2").
fn elide(ui: &Ui, text: &str, font: &FontId, max: f32) -> String {
    let width = |s: &str| ui.painter().layout_no_wrap(s.to_owned(), font.clone(), Color32::WHITE).size().x;
    if width(text) <= max {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    for keep in (2..chars.len()).rev() {
        let (head, tail) = (keep - keep / 2, keep / 2);
        let s: String = chars[..head].iter().chain(['…'].iter()).chain(chars[chars.len() - tail..].iter()).collect();
        if width(&s) <= max {
            return s;
        }
    }
    "…".into()
}

fn mods(ui: &Ui) -> ClickMods {
    let m = ui.input(|i| i.modifiers);
    ClickMods { toggle: m.command, range: m.shift }
}

/// A small clickable icon cell: where, what to draw, and its tooltip.
struct Cell {
    rect: Rect,
    key: (LayerId, &'static str),
    icon: Icon,
    on: bool,
    tint: Color32,
    tip: &'static str,
}

impl Cell {
    fn show(self, ui: &mut Ui) -> egui::Response {
        let resp = ui.interact(self.rect, ui.id().with(self.key), Sense::click());
        if resp.hovered() {
            ui.painter().rect_filled(self.rect, 2.0, ui.visuals().widgets.hovered.bg_fill);
        } else if self.on {
            ui.painter().rect_filled(self.rect, 2.0, ui.visuals().widgets.inactive.bg_fill.gamma_multiply(0.7));
        }
        self.icon.paint(ui.painter(), self.rect.shrink(self.rect.width() * 0.15), self.tint);
        resp.on_hover_text(self.tip)
    }
}

pub fn show(app: &mut SculptApp, ui: &mut Ui, n: &Node, order: &[LayerId], geoms: &mut Vec<RowGeom>) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), Sense::hover());
    geoms.push(RowGeom { id: n.id, rect, is_folder: n.is_folder });
    let row = ui.interact(rect, ui.id().with(("row", n.id)), Sense::click_and_drag());

    let selected = app.layers.selection.contains(n.id);
    let primary = app.layers.selection.primary() == Some(n.id);
    let ui_colors = app.theme.ui.clone();
    let fs = app.theme.metrics.font_size;
    let dim = if n.visible && !n.excluded { 1.0 } else { 0.45 };
    let text_color = if primary { ui.visuals().strong_text_color() } else { ui.visuals().text_color() }.gamma_multiply(dim);
    let weak = app.theme.weak_text().gamma_multiply(dim);

    // Background: selection tint with a 1 px accent outline, like Painter.
    let painter = ui.painter_at(rect);
    if selected {
        painter.rect_filled(rect, 0.0, ui_colors.row_selected());
        painter.rect_stroke(rect.shrink(0.5), 0.0, Stroke::new(1.0, ui_colors.accent.0), StrokeKind::Inside);
    } else if row.hovered() {
        painter.rect_filled(rect, 0.0, ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.4));
    }
    painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, ui_colors.separator.0));

    // Left cluster.
    let cy = rect.center().y;
    let mut x = rect.left() + 4.0;
    let square = |x: f32, s: f32| Rect::from_center_size(Pos2::new(x + s / 2.0, cy), vec2(s, s));

    let eye_rect = square(x, 20.0);
    x += 22.0 + n.depth as f32 * INDENT;
    let disc_rect = square(x, 16.0);
    x += 18.0;
    let content_rect = Rect::from_min_size(Pos2::new(x, cy - THUMB / 2.0 - 1.0), vec2(THUMB, THUMB));
    x += THUMB + 4.0;
    let mask_rect = Rect::from_min_size(Pos2::new(x, content_rect.top()), vec2(THUMB, THUMB));
    x += THUMB + 6.0;
    let name_left = x;

    // Right cluster, built from the edge inward.
    let mut rx = rect.right() - 6.0;
    let strength_rect = Rect::from_min_size(Pos2::new(rx - 50.0, cy - 10.0), vec2(50.0, 20.0));
    rx -= 54.0;
    let blend_rect = Rect::from_min_size(Pos2::new(rx - 46.0, cy - 10.0), vec2(46.0, 20.0));
    if !n.is_folder {
        rx -= 48.0;
    }
    let lock_rect = square(rx - SMALL, SMALL);
    rx -= SMALL + 2.0;
    let solo_rect = square(rx - SMALL, SMALL);
    rx -= SMALL + 4.0;
    let name_right = rx;

    // Eye. Alt+click solos, as in Photoshop.
    let eye = Cell { rect: eye_rect, key: (n.id, "eye"), icon: if n.visible { Icon::Eye } else { Icon::EyeOff }, on: false, tint: text_color, tip: "Visibility (Alt+click: solo)" }.show(ui);
    if eye.clicked() {
        if ui.input(|i| i.modifiers.alt) {
            command::execute(app, LayerCommand::ToggleSolo(n.id));
        } else {
            command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Visible(!n.visible), coalesce: false });
        }
    }

    // Disclosure: folder children, or a layer's mask ops.
    let expandable = if n.is_folder { n.has_children } else { n.has_mask };
    if expandable {
        let open = if n.is_folder { !n.collapsed } else { app.expanded.contains(&n.id) };
        let d = Cell { rect: disc_rect, key: (n.id, "disc"), icon: if open { Icon::ChevronDown } else { Icon::ChevronRight }, on: false, tint: weak, tip: if open { "Collapse" } else { "Expand" } }.show(ui);
        if d.clicked() {
            if n.is_folder {
                command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Collapsed(open), coalesce: false });
            } else if open {
                app.expanded.remove(&n.id);
            } else {
                app.expanded.insert(n.id);
            }
        }
    }

    // Content thumbnail (or the folder glyph).
    let doc_active = app.doc.as_ref().and_then(|d| d.active_layer());
    let target_content = primary && !n.is_folder && doc_active == Some(n.id) && app.selection == Selection::Layer;
    let target_mask = primary && n.has_mask && doc_active == Some(n.id) && matches!(app.selection, Selection::Mask | Selection::Effect(_));
    let thumb_bg = app.theme.viewport.background_bottom.0.gamma_multiply(dim);
    painter.rect_filled(content_rect, 3.0, thumb_bg);
    if n.is_folder {
        let icon = if n.collapsed { Icon::Folder } else { Icon::FolderOpen };
        icon.paint(&painter, content_rect.shrink(4.0), ui_colors.text_weak.0.gamma_multiply(dim + 0.3));
    } else {
        Icon::Base.paint(&painter, content_rect.shrink(3.0), ui_colors.target_delta().gamma_multiply(dim));
    }
    if target_content {
        painter.rect_stroke(content_rect.expand(1.0), 3.0, Stroke::new(2.0, ui_colors.target_delta()), StrokeKind::Outside);
    }
    let content = ui.interact(content_rect, ui.id().with((n.id, "content")), Sense::click());
    if content.clicked() {
        select(app, n, order, ClickMods::default());
        if !n.is_folder {
            crate::panels::select_layer(app, Some(n.id), Selection::Layer);
        }
    }
    content.on_hover_text(if n.is_folder { "Folder" } else { "Sculpt delta: click to sculpt on this layer" });

    // Mask thumbnail, or an empty slot that offers "+" on hover.
    let mask_resp = ui.interact(mask_rect, ui.id().with((n.id, "mask")), Sense::click());
    if !n.is_folder {
        if n.has_mask {
            let base = app.mask_edit.as_ref().filter(|(i, _)| *i == n.id).map(|(_, s)| s.base).or_else(|| app.doc.as_ref().and_then(|d| d.layer(n.id)).and_then(|l| l.mask.as_ref()).map(|m| m.base)).unwrap_or(1.0);
            let g = (base.clamp(0.0, 1.0) * 255.0) as u8;
            painter.rect_filled(mask_rect, 3.0, Color32::from_gray(g).gamma_multiply(dim));
            Icon::Mask.paint(&painter, mask_rect.shrink(6.0), if g > 140 { Color32::from_gray(60) } else { Color32::from_gray(190) }.gamma_multiply(dim));
            if target_mask {
                painter.rect_stroke(mask_rect.expand(1.0), 3.0, Stroke::new(2.0, ui_colors.target_mask()), StrokeKind::Outside);
            }
            // Bar under the thumbnail: lit when the mask has ops, like Painter's effects line.
            let bar = Rect::from_min_size(Pos2::new(mask_rect.left(), mask_rect.bottom() + 2.0), vec2(THUMB, 2.0));
            painter.rect_filled(bar, 1.0, if n.mask_ops > 0 { ui_colors.target_mask() } else { ui_colors.separator.0 });
            if mask_resp.clicked() {
                if ui.input(|i| i.modifiers.alt) {
                    app.overlay = if app.overlay == OverlayKind::LayerMask { OverlayKind::None } else { OverlayKind::LayerMask };
                } else {
                    select(app, n, order, ClickMods::default());
                    crate::panels::select_layer(app, Some(n.id), Selection::Mask);
                }
            }
            menus::mask_menu_on(app, &mask_resp, n.id);
            mask_resp.on_hover_text("Mask. Click: paint target. Alt+click: view in viewport");
        } else if row.hovered() || mask_resp.hovered() {
            painter.rect_stroke(mask_rect, 3.0, Stroke::new(1.0, weak), StrokeKind::Inside);
            Icon::Plus.paint(&painter, mask_rect.shrink(7.0), weak);
            menus::add_mask_menu_on(app, &mask_resp, n.id);
            mask_resp.on_hover_text("Add a mask");
        }
    }
    // Bar under the content thumbnail (content effects do not exist yet, so it stays quiet).
    painter.rect_filled(Rect::from_min_size(Pos2::new(content_rect.left(), content_rect.bottom() + 2.0), vec2(THUMB, 2.0)), 1.0, ui_colors.separator.0);

    // Name, inline rename, or elided label.
    let name_rect = Rect::from_min_max(Pos2::new(name_left, rect.top()), Pos2::new(name_right.max(name_left + 20.0), rect.bottom()));
    let font = if n.is_folder || primary { FontId::new(fs, bold()) } else { FontId::proportional(fs) };
    if app.layers.rename.as_ref().is_some_and(|(id, _)| *id == n.id) {
        rename_field(app, ui, name_rect, n);
    } else {
        let label = elide(ui, &n.name, &font, name_rect.width());
        painter.text(Pos2::new(name_rect.left(), cy), Align2::LEFT_CENTER, label, font, text_color);
    }

    // Solo and lock: always visible when on, otherwise only while the row is hovered or selected.
    let show_idle = row.hovered() || selected;
    if n.soloed || show_idle {
        let tint = if n.soloed { ui_colors.accent.0 } else { weak };
        if (Cell { rect: solo_rect, key: (n.id, "solo"), icon: Icon::Solo, on: n.soloed, tint, tip: "Solo: show only this layer (S)" }).show(ui).clicked() {
            command::execute(app, LayerCommand::ToggleSolo(n.id));
        }
    }
    if n.locked || show_idle {
        let tint = if n.locked { text_color } else { weak };
        if (Cell { rect: lock_rect, key: (n.id, "lock"), icon: if n.locked { Icon::Lock } else { Icon::Unlock }, on: n.locked, tint, tip: "Lock (Shift+L)" }).show(ui).clicked() {
            command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Locked(!n.locked), coalesce: false });
        }
    }
    if !n.is_folder {
        blend_button(app, ui, blend_rect, n, weak);
    }
    strength_field(app, ui, strength_rect, n, weak);

    // Row-level interaction (lowest priority: sub-widgets above took their clicks).
    if row.double_clicked() {
        command::execute(app, LayerCommand::Rename);
    } else if row.clicked() {
        select(app, n, order, mods(ui));
        if !n.is_folder && !mods(ui).toggle && !mods(ui).range {
            crate::panels::select_layer(app, Some(n.id), Selection::Layer);
        }
    }
    if row.secondary_clicked() && !app.layers.selection.contains(n.id) {
        select(app, n, order, ClickMods::default());
        if !n.is_folder {
            crate::panels::select_layer(app, Some(n.id), Selection::Layer);
        }
    }
    if row.drag_started() {
        if !app.layers.selection.contains(n.id) {
            select(app, n, order, ClickMods::default());
        }
        let ids = app.layers.selection.in_order(order);
        app.layers.drag = Some(DragState { ids });
    }
    egui::Popup::context_menu(&row).show(|ui| menus::row_menu(app, ui, n));
    if app.layers.drag.is_some() {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
    }
}

fn select(app: &mut SculptApp, n: &Node, order: &[LayerId], m: ClickMods) {
    app.layers.selection.click(n.id, m, order);
    if n.is_folder {
        app.selection = Selection::Layer;
    }
}

fn rename_field(app: &mut SculptApp, ui: &mut Ui, rect: Rect, n: &Node) {
    let Some((_, text)) = app.layers.rename.as_mut() else { return };
    let escape = ui.input(|i| i.key_pressed(Key::Escape));
    let te = ui.put(rect.shrink2(vec2(0.0, 6.0)), egui::TextEdit::singleline(text).margin(vec2(4.0, 0.0)));
    if !te.has_focus() && !te.lost_focus() {
        te.request_focus();
    }
    if escape {
        app.layers.rename = None;
    } else if te.lost_focus()
        && let Some((_, text)) = app.layers.rename.take()
    {
        command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Name(text), coalesce: false });
    }
}

/// Drag to scrub, click to type. Range is -100% to 200% like the engine's strength.
fn strength_field(app: &mut SculptApp, ui: &mut Ui, rect: Rect, n: &Node, weak: Color32) {
    let mut pct = n.strength * 100.0;
    let mut child = ui.new_child(UiBuilder::new().max_rect(rect));
    let v = child.visuals_mut();
    v.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.override_text_color = Some(weak.gamma_multiply(1.6));
    let dv = child.add_sized(rect.size(), egui::DragValue::new(&mut pct).range(-100.0..=200.0).speed(0.5).suffix("%").max_decimals(0));
    if dv.changed() {
        // A drag is one undo step; typed values are separate steps.
        let coalesce = dv.dragged() && !dv.drag_started();
        command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Strength(pct / 100.0), coalesce });
    }
    dv.on_hover_text("Strength: drag to scrub, click to type");
}

/// Blend mode of a sculpt layer: a short label that opens the list of modes.
fn blend_button(app: &mut SculptApp, ui: &mut Ui, rect: Rect, n: &Node, weak: Color32) {
    let resp = ui.interact(rect, ui.id().with((n.id, "blend")), Sense::click());
    let hot = resp.hovered() || n.blend != sculpt_core::LayerBlend::Add;
    if resp.hovered() {
        ui.painter().rect_filled(rect, 3.0, ui.visuals().widgets.hovered.bg_fill);
    }
    let color = if hot { weak.gamma_multiply(1.6) } else { weak };
    let font = FontId::proportional(app.theme.metrics.font_size * 0.88);
    ui.painter().text(rect.left_center() + vec2(4.0, 0.0), Align2::LEFT_CENTER, n.blend.short(), font, color);
    Icon::ChevronDown.paint(ui.painter(), Rect::from_center_size(rect.right_center() - vec2(8.0, 0.0), vec2(10.0, 10.0)), color);
    let resp = resp.on_hover_text(format!("Blend: {}. {}", n.blend.label(), n.blend.hint()));
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_min_width(150.0);
        for mode in sculpt_core::LayerBlend::ALL {
            let r = ui.selectable_label(n.blend == mode, mode.label()).on_hover_text(mode.hint());
            if r.clicked() {
                command::execute(app, LayerCommand::Edit { id: n.id, edit: MetaEdit::Blend(mode), coalesce: false });
                ui.close();
            }
        }
    });
}

//! Panels in a Substance Painter / Mudbox hybrid layout:
//!
//! * menu bar, then a Painter-style **context toolbar** (brush, size, strength, falloff);
//! * right dock: the volume's **LAYERS** (form, detail, generator and surface noise layers, and what strokes edit)
//!   over **PROPERTIES** (the selected layer's settings and the brush);
//! * bottom: Mudbox **tool tray** (brushes / falloff) and a status bar.

use egui::{Align, Color32, Layout, Rect, RichText, Sense, Ui, UiBuilder, vec2};
use sculpt_core::brush::Falloff;

use crate::app::{Dialog, SculptApp, fmt_count};
use crate::icons::{Icon, icon_button};
use crate::keymap::Command;
use crate::theme::{Hex, bold};
use crate::tools::Tool;
use crate::volume_mode::{Target, VolumeMode};
use sculpt_core::generator::sculpt_layer::{LayerBlend, MAX_OPACITY, MaskInfo, MaskKind, NoiseMask};
use sculpt_core::mask::BlendMode;
use sculpt_core::noise::NoiseKind;

const ROW_H: f32 = 26.0;

// ------------------------------------------------------------------ helpers

fn cmd_button(app: &mut SculptApp, ui: &mut Ui, cmd: Command) {
    let mut b = egui::Button::new(cmd.label());
    if let Some(s) = app.keymap.shortcut_text(ui.ctx(), cmd) {
        b = b.shortcut_text(s);
    }
    if ui.add(b).clicked() {
        let ctx = ui.ctx().clone();
        app.run(&ctx, cmd);
        ui.close();
    }
}

/// Uppercase panel title bar with optional right-aligned controls.
pub(crate) fn panel_header(ui: &mut Ui, app: &SculptApp, title: &str, right: impl FnOnce(&mut Ui)) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, app.theme.ui.header());
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(1.0, app.theme.ui.separator.0));
    let mut child = ui.new_child(UiBuilder::new().max_rect(rect.shrink2(vec2(10.0, 0.0))).layout(Layout::left_to_right(Align::Center)));
    child.label(RichText::new(title).size(app.theme.metrics.font_size * 0.8).family(bold()).color(app.theme.weak_text()).extra_letter_spacing(0.6));
    child.with_layout(Layout::right_to_left(Align::Center), right);
}

/// Section header inside Properties.
pub(crate) fn section(ui: &mut Ui, font_size: f32, id: &str, title: &str, body: impl FnOnce(&mut Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).size(font_size * 0.82).family(bold()).extra_letter_spacing(0.5))
        .id_salt(id)
        .default_open(true)
        .show(ui, |ui| {
            // Fixed width: sizing from available width would feed back into the
            // panel's own width and grow it every frame.
            ui.spacing_mut().slider_width = 150.0;
            body(ui);
        });
}

/// Label column + widget, like Painter's property rows.
pub(crate) fn prop<R>(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(78.0, ui.spacing().interact_size.y), Sense::hover());
        ui.painter().text(r.left_center(), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(ui.style().text_styles[&egui::TextStyle::Body].size * 0.92), ui.visuals().weak_text_color());
        add(ui)
    })
    .inner
}

/// What strokes edit right now, in words and the colour that marks it.
fn target_text(app: &SculptApp) -> (String, Color32) {
    let m = &app.volume;
    let ui = &app.theme.ui;
    let Some(layer) = m.layer.as_ref() else {
        let Some((active, layers)) = m.sculpt_layers() else { return ("Strokes edit the volume".into(), ui.accent.0) };
        let Some(info) = layers.get(active) else { return ("Strokes edit the volume".into(), ui.accent.0) };
        let name = &info.name;
        let mask = info.active_mask.and_then(|i| info.masks.get(i));
        return match (m.target, mask) {
            (Target::Form, _) => ("Strokes edit the base (under every layer)".into(), ui.accent.0),
            (Target::Mask { .. }, Some(mk)) if matches!(mk.kind, MaskKind::Noise(_)) => (format!("{} is noise: set it in Properties (strokes do nothing)", mk.name), ui.target_mask()),
            (Target::Mask { show }, mk) => {
                let mk = mk.map_or("its mask".to_string(), |mk| mk.name.clone());
                (format!("Strokes paint {mk} {} ({} {name}; X swaps)", if show { "white" } else { "black" }, if show { "showing" } else { "hiding" }), ui.target_mask())
            }
            _ => (format!("Strokes edit {name}"), ui.target_delta()),
        };
    };
    let name = layer.layers().get(layer.active()).map_or(String::new(), |l| l.name.clone());
    match m.routed() {
        // A generator or noise layer holds no sculpting: strokes go to the form under it.
        Target::Form if m.target == Target::Detail => (format!("Strokes edit the form ({name} has no sculpting of its own)"), ui.accent.0),
        Target::Form => (format!("Strokes edit the form (levels 0 to {})", layer.from() - 1), ui.accent.0),
        Target::Detail => (format!("Strokes edit {name}"), ui.target_delta()),
        Target::Mask { show: false } => (format!("Strokes hide {name} (its mask)"), ui.target_mask()),
        Target::Mask { show: true } => (format!("Strokes show {name} again (its mask)"), ui.target_mask()),
        Target::Density { more } => (format!("Strokes paint where {name}'s rocks {}", if more { "may go" } else { "stay out" }), ui.target_freeze()),
    }
}

// ----------------------------------------------------------------- menu bar

/// The new balls: what a fresh one shows (its level 0), and the size given to [`VolumeMode::ball`] (its level 1, the
/// step of the balls these labels once named).
const BALLS: [(&str, f32); 3] = [("Ball, 250k triangles", 1e6), ("Ball, 750k triangles", 3e6), ("Ball, 1.25M triangles", 5e6)];

pub fn menu_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::top("menu").frame(egui::Frame::side_top_panel(ui.style()).fill(app.theme.ui.window.0)).show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            let (logo, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
            Icon::Base.paint(ui.painter(), logo, app.theme.ui.accent.0);
            ui.add_space(4.0);
            ui.menu_button("File", |ui| {
                ui.menu_button("New", |ui| {
                    for (label, tris) in BALLS {
                        if ui.button(label).clicked() {
                            app.dialog = Some(Dialog::New(tris));
                            ui.close();
                        }
                    }
                });
                cmd_button(app, ui, Command::Open);
                cmd_button(app, ui, Command::Save);
                if ui.button("Save As…").clicked() {
                    app.dialog = Some(Dialog::SaveAs(app.volume_path.clone().unwrap_or("untitled.svol".into())));
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                cmd_button(app, ui, Command::Undo);
                cmd_button(app, ui, Command::Redo);
                ui.separator();
                if ui.button("Keyboard shortcuts…").clicked() {
                    app.show_keymap = true;
                    ui.close();
                }
            });
            ui.menu_button("Volume", |ui| volume_menu(app, ui));
            ui.menu_button("Display", |ui| {
                cmd_button(app, ui, Command::Frame);
                cmd_button(app, ui, Command::ToggleHud);
                ui.separator();
                let m = &mut app.volume;
                let top = m.vol.levels() - 1;
                ui.add(egui::Slider::new(&mut m.cap, 0..=top).text("Detail shown up to level"));
                ui.checkbox(&mut m.by_distance, "Level by distance").on_hover_text(
                    "Each region shows only the levels whose detail would show on screen from here: far work is drawn from coarser levels, near work in full.",
                );
                ui.add_enabled(m.by_distance, egui::Slider::new(&mut m.level_px, 0.25..=4.0).logarithmic(true).text("Detail shown down to (pixels)"));
                ui.add_enabled(m.by_distance, egui::Checkbox::new(&mut m.budget_on, "Triangle budget")).on_hover_text(
                    "Past this many triangles, only the detail that shows most is drawn: the pixels a block's detail must come to rise as far as it takes. Level 0 always shows.",
                );
                ui.add_enabled(m.by_distance && m.budget_on, egui::Slider::new(&mut m.budget_m, 0.5..=40.0).logarithmic(true).text("Million triangles"));
                ui.checkbox(&mut m.tint, "Tint by level");
                if let Some(vp) = app.viewport.as_mut() {
                    ui.checkbox(&mut vp.cull, "Skip clusters out of view or facing away");
                }
            });
            ui.menu_button("Theme", |ui| {
                let names: Vec<String> = app.themes.themes.iter().map(|t| t.name.clone()).collect();
                for n in names {
                    if ui.radio(app.theme.name == n, &n).clicked() {
                        let t = app.themes.get(&n);
                        let ctx = ui.ctx().clone();
                        app.set_theme(&ctx, t);
                    }
                }
                ui.separator();
                if ui.button("Theme editor…").clicked() {
                    app.theme_editor = Some(app.theme.clone());
                    ui.close();
                }
            });
            // Document title, centred like Painter's project name.
            let title = app.volume_path.as_deref().map_or("Untitled".to_string(), crate::app::file_name);
            let full = ui.max_rect();
            ui.painter().text(
                egui::pos2(full.center().x, full.center().y),
                egui::Align2::CENTER_CENTER,
                format!("Sculpt  ·  {title}"),
                egui::FontId::new(app.theme.metrics.font_size * 0.9, egui::FontFamily::Proportional),
                app.theme.weak_text(),
            );
        });
    });
}

const SCULPT_HOVER: &str = "An empty layer on top to sculpt on, on a mesh taken from the volume's surface: it keeps what its strokes change. Its opacity (up to 200%), blend and masks (one painted, one noise) fade it anywhere. Sculpting the base under it carries it along.";
/// Asks whether to merge the volume's own sculpt layers so mesh layers can be added.
const MERGE_ASK: &str = "mesh_merge_ask";
const NOISE_HOVER: &str = "A layer of noise over the whole surface, drawn by the GPU as it is shown: its coarse octaves move the surface, every octave shades it, finer than any level. Its mask fades it. Brushes work on the surface under it.";
const GENERATOR_HOVER: &str = "A layer that scatters a primitive over the form and joins it softly (or cuts it): rocks to start with. Its settings are under Properties; sculpting the form carries it along.";

/// The Volume menu: new layers, merging them, and how big brushes treat detail.
fn volume_menu(app: &mut SculptApp, ui: &mut Ui) {
    let m = &mut app.volume;
    let top = m.vol.levels() - 1;
    let mesh = m.mesh.is_some() || m.mesh_building();
    if ui.add_enabled(m.layer.is_none(), egui::Button::new("New sculpt layer")).on_hover_text(SCULPT_HOVER).clicked() {
        if m.has_volume_layers() {
            ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(MERGE_ASK), true));
        } else {
            m.add_mesh_layer();
        }
        ui.close();
    }
    if ui.add_enabled(!mesh, egui::Button::new("New generator layer")).on_hover_text(GENERATOR_HOVER).clicked() {
        m.add_generator();
        ui.close();
    }
    if ui.add_enabled(!mesh, egui::Button::new("New surface noise layer")).on_hover_text(NOISE_HOVER).clicked() {
        m.add_noise();
        ui.close();
    }
    match m.layer.as_ref().map(|l| l.from()) {
        // (Mesh layers stay layers: nothing merges them into the volume yet.)
        None if mesh => {}
        None if m.sculpt_layers().is_some() => {
            if ui.button("Merge the layers").on_hover_text("One volume again, as shown. Undo starts afresh.").clicked() {
                m.merge_layer();
                ui.close();
            }
        }
        None => {
            ui.menu_button("Move detail to a layer", |ui| {
                ui.label(RichText::new("The work at this level and finer goes into a detail layer over the form.").small());
                for l in 1..=top {
                    if ui.button(format!("From level {l}  (step {:.5})", m.vol.voxel(l))).clicked() {
                        m.make_layer(l);
                        ui.close();
                    }
                }
            });
        }
        Some(_) => {
            if ui.button("Merge the layers into the form").on_hover_text("One volume again, as shown. Undo starts afresh.").clicked() {
                m.merge_layer();
                ui.close();
            }
        }
    }
    ui.separator();
    ui.checkbox(&mut m.vol.keep_detail, "Big brushes keep finer detail").on_hover_text(
        "On: a big brush moves the finer detail under it along with the new surface, as sculpting at a lower subdivision level does.\nOff: it builds over the detail like clay, filling grooves and covering bumps where it builds above them.",
    );
}

// ----------------------------------------------------------- context toolbar

pub fn context_bar(app: &mut SculptApp, ui: &mut Ui) {
    let fill = app.theme.ui.panel.0;
    egui::Panel::top("context").exact_size(34.0).frame(egui::Frame::side_top_panel(ui.style()).fill(fill)).show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            let tool = app.tool;
            let (r, _) = ui.allocate_exact_size(vec2(22.0, 22.0), Sense::hover());
            tool.paint_icon(ui.painter(), r, ui.visuals().strong_text_color());
            ui.label(RichText::new(tool.label()).family(bold()));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);
            ui.spacing_mut().slider_width = 110.0;
            let p = app.tools.params_mut(tool);
            ui.label(RichText::new("Size").weak());
            ui.add(egui::Slider::new(&mut p.size_px, 2.0..=600.0).logarithmic(true).max_decimals(0));
            ui.label(RichText::new("Strength").weak());
            ui.add(egui::Slider::new(&mut p.strength, 0.0..=1.0).max_decimals(2));
            ui.label(RichText::new("Falloff").weak());
            ui.add(egui::Slider::new(&mut p.hardness, 0.0..=0.95).max_decimals(2)).on_hover_text("Hardness: 0 = soft, 1 = hard edge");
            if tool != Tool::Smooth {
                ui.label(RichText::new("Shift smooth").weak());
                ui.add(egui::Slider::new(&mut app.tools.params_mut(Tool::Smooth).strength, 0.0..=1.0).max_decimals(2)).on_hover_text("Strength of the Smooth brush, which Shift gives with any brush");
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if icon_button(ui, Icon::Eye, 22.0, app.hud, "Engine readout (H)").clicked() {
                    let ctx = ui.ctx().clone();
                    app.run(&ctx, Command::ToggleHud);
                }
                let (text, color) = target_text(app);
                ui.label(RichText::new(text).color(color));
            });
        });
    });
}

// --------------------------------------------------------------- status bar

pub fn status_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::bottom("status").exact_size(22.0).frame(egui::Frame::side_top_panel(ui.style()).fill(app.theme.ui.window.0)).show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            let weak = app.theme.weak_text();
            ui.label(RichText::new(app.tool.hint()).color(weak));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let m = &app.volume;
                let blocks: usize = m.vol.memory().0.iter().sum();
                ui.label(RichText::new(format!("{} blocks  ·  about {} triangles shown", fmt_count(blocks), fmt_count(m.view.triangles as usize))).color(weak));
                ui.separator();
                ui.label(RichText::new(format!("Pen: {} {:.2}", app.pen.source.label(), app.pen.pressure)).color(weak)).on_hover_text(&app.pen.status);
                // Transient messages fade out after a few seconds.
                if app.status != app.status_seen.0 {
                    app.status_seen = (app.status.clone(), std::time::Instant::now());
                }
                let age = app.status_seen.1.elapsed().as_secs_f32();
                if !app.status.is_empty() && age < 6.0 {
                    let alpha = (6.0 - age).clamp(0.0, 1.0);
                    ui.separator();
                    ui.add(egui::Label::new(RichText::new(&app.status).color(weak.gamma_multiply(alpha))).truncate());
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                }
            });
        });
    });
}

// --------------------------------------------------------------------- tray

pub fn tray(app: &mut SculptApp, ui: &mut Ui) {
    let tile = app.theme.metrics.tray_tile;
    let fill = app.theme.ui.tray.0;
    egui::Panel::bottom("tray").exact_size(tile + 36.0).frame(egui::Frame::side_top_panel(ui.style()).fill(fill)).show(ui, |ui| {
        let tab_id = egui::Id::new("tray_falloff");
        let mut falloff = ui.data(|d| d.get_temp::<bool>(tab_id)).unwrap_or(false);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (on, label) in [(false, "Brushes"), (true, "Falloff")] {
                let sel = falloff == on;
                let text = if sel { RichText::new(label).strong() } else { RichText::new(label).color(app.theme.weak_text()) };
                if ui.add(egui::Button::selectable(sel, text).corner_radius(0)).clicked() {
                    falloff = on;
                }
            }
        });
        ui.data_mut(|d| d.insert_temp(tab_id, falloff));
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if falloff {
                falloff_tray(app, ui, tile);
                return;
            }
            for t in Tool::ALL {
                let selected = app.tool == t;
                let key = app.tool_hotkey(ui.ctx(), t);
                let resp = tile_ui(ui, app, tile, selected, |p, r, c| t.paint_icon(p, r, c), t.label(), &key);
                if resp.on_hover_text(t.hint()).clicked() {
                    app.select_tool(t);
                }
            }
        });
    });
}

fn tile_ui(ui: &mut Ui, app: &SculptApp, tile: f32, selected: bool, icon: impl FnOnce(&egui::Painter, Rect, Color32), label: &str, key: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(tile * 1.2, tile), Sense::click());
    let v = ui.visuals();
    let (bg, fg) = if selected {
        (app.theme.ui.row_selected(), v.strong_text_color())
    } else if resp.hovered() {
        (v.widgets.hovered.bg_fill, v.strong_text_color())
    } else {
        (v.widgets.inactive.bg_fill.gamma_multiply(0.6), v.text_color())
    };
    let painter = ui.painter();
    let radius = app.theme.metrics.corner_radius as f32 + 2.0;
    painter.rect_filled(rect, radius, bg);
    painter.line_segment([rect.left_top() + vec2(radius, 0.5), rect.right_top() + vec2(-radius, 0.5)], egui::Stroke::new(1.0, Color32::from_white_alpha(if selected { 40 } else { 14 })));
    if selected {
        painter.rect_filled(Rect::from_min_max(rect.left_bottom() - vec2(0.0, 2.0), rect.right_bottom()), 0.0, app.theme.ui.accent.0);
    }
    let icon_rect = Rect::from_center_size(rect.center_top() + vec2(0.0, tile * 0.36), vec2(tile * 0.42, tile * 0.42));
    icon(painter, icon_rect, fg);
    painter.text(rect.center_bottom() - vec2(0.0, tile * 0.17), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(app.theme.metrics.font_size * 0.82), fg);
    if !key.is_empty() {
        painter.text(rect.right_top() + vec2(-4.0, 3.0), egui::Align2::RIGHT_TOP, key, egui::FontId::monospace(9.5), fg.gamma_multiply(0.6));
    }
    resp
}

/// Mudbox's Falloff tray: curve presets applied to the current brush.
fn falloff_tray(app: &mut SculptApp, ui: &mut Ui, tile: f32) {
    let current = app.tools.params(app.tool).hardness;
    for (label, h) in [("Soft", 0.0f32), ("Smooth", 0.2), ("Medium", 0.45), ("Firm", 0.7), ("Hard", 0.9)] {
        let selected = (current - h).abs() < 0.03;
        let resp = tile_ui(
            ui,
            app,
            tile,
            selected,
            |p, r, c| {
                let f = Falloff { hardness: h };
                let pts: Vec<egui::Pos2> = (0..=32)
                    .map(|i| {
                        let t = i as f32 / 32.0;
                        egui::pos2(r.left() + r.width() * t, r.bottom() - r.height() * f.eval(t))
                    })
                    .collect();
                p.line_segment([r.left_bottom(), r.right_bottom()], egui::Stroke::new(1.0, c.gamma_multiply(0.4)));
                p.add(egui::Shape::line(pts, egui::Stroke::new(1.8, c)));
            },
            label,
            "",
        );
        if resp.on_hover_text(format!("Falloff hardness {h:.2} for {}", app.tool.label())).clicked() {
            app.tools.params_mut(app.tool).hardness = h;
        }
    }
}

// --------------------------------------------------------------- right dock

pub fn right_panel(app: &mut SculptApp, ui: &mut Ui) {
    let fill = app.theme.ui.panel.0;
    egui::Panel::right("dock").default_size(340.0).min_size(280.0).frame(egui::Frame::side_top_panel(ui.style()).fill(fill).inner_margin(0)).show(ui, |ui| {
        let total = ui.available_height();
        // The stack keeps 30% to 70% of the dock whatever the window size, so it never collapses to a few rows.
        egui::Panel::top("layers_panel")
            .resizable(true)
            .default_size(total * 0.45)
            .size_range((total * 0.3).max(180.0)..=(total * 0.7))
            .frame(egui::Frame::NONE.fill(fill))
            .show(ui, |ui| layers_panel(app, ui));
        egui::CentralPanel::no_frame().show(ui, |ui| properties_panel(app, ui));
    });
}

/// Row background + a child Ui for its contents. `bar` marks the selected row's left edge.
fn row_frame(ui: &mut Ui, app: &SculptApp, indent: f32, selected: bool, bar: Color32) -> (egui::Response, Ui) {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), Sense::click());
    let bg = if selected {
        app.theme.ui.row_selected()
    } else if resp.hovered() {
        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.5)
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 0.0, bg);
    if selected {
        ui.painter().rect_filled(Rect::from_min_max(rect.left_top(), rect.left_bottom() + vec2(3.0, 0.0)), 0.0, bar);
    }
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(1.0, app.theme.ui.separator.0));
    let inner = rect.shrink2(vec2(4.0, 0.0)).with_min_x(rect.left() + 6.0 + indent).with_max_x(rect.right() - 6.0);
    let child = ui.new_child(UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)));
    (resp, child)
}

/// A small explanatory paragraph, inset like the rows' contents.
fn note(ui: &mut Ui, text: &str, color: Color32) {
    egui::Frame::NONE.inner_margin(egui::Margin::symmetric(10, 0)).show(ui, |ui| ui.label(RichText::new(text).small().color(color)));
}

/// What a layer row asked for, done after the rows are drawn.
#[derive(Default)]
struct RowActions {
    select: Option<usize>,
    visible: Option<(usize, bool)>,
    remove: Option<usize>,
    target: Option<Target>,
    /// A sculpt layer, and its mask or the layer itself, picked.
    pick: Option<(usize, Option<usize>)>,
    mask_set: Option<(usize, usize, MaskInfo)>,
    mask_remove: Option<(usize, usize)>,
    add_mask: Option<MaskKind>,
    opacity: Option<(usize, f32)>,
    blend: Option<(usize, LayerBlend)>,
}

/// A row's opacity, as a percentage to drag (or click and type), `max` the most.
fn row_opacity(ui: &mut Ui, opacity: f32, max: f32, hover: &str) -> Option<f32> {
    let mut pct = opacity * 100.0;
    let r = ui.add(egui::DragValue::new(&mut pct).range(0.0..=max * 100.0).speed(0.5).suffix("%").fixed_decimals(0).custom_formatter(|v, _| format!("{v:.0}")));
    let r = r.on_hover_text(hover);
    (r.changed() && (pct / 100.0 - opacity).abs() > 1e-6).then_some(pct / 100.0)
}

/// A row's blend, picked from `all`.
fn row_blend<T: Copy + PartialEq>(ui: &mut Ui, fs: f32, id: impl std::hash::Hash + std::fmt::Debug, now: T, all: &[T], label: impl Fn(T) -> &'static str, hover: impl Fn(T) -> &'static str) -> Option<T> {
    let mut out = None;
    egui::ComboBox::from_id_salt(id).width(fs * 5.2).selected_text(RichText::new(label(now)).size(fs * 0.85)).show_ui(ui, |ui| {
        for &b in all {
            let r = ui.selectable_label(now == b, label(b));
            let h = hover(b);
            let r = if h.is_empty() { r } else { r.on_hover_text(h) };
            if r.clicked() && now != b {
                out = Some(b);
            }
        }
    });
    out
}

fn layer_blend_hover(b: LayerBlend) -> &'static str {
    match b {
        LayerBlend::Add => "The layer's strokes as sculpted.",
        LayerBlend::Subtract => "Turned over: what the layer raised is carved, and what it carved is raised.",
        LayerBlend::Raise => "The higher surface of the base with the layer and without it: only what the layer raises shows.",
        LayerBlend::Carve => "The lower surface of the base with the layer and without it: only what the layer carves shows.",
    }
}

fn noise_label(k: NoiseKind) -> &'static str {
    match k {
        NoiseKind::Perlin => "Smooth",
        NoiseKind::Fbm => "Clouds",
        NoiseKind::Ridged => "Ridged",
        NoiseKind::Turbulence => "Turbulence",
        NoiseKind::Cellular => "Cells",
    }
}

/// The volume's stack: its layers top first, then the form under them; the selected row is what strokes edit.
fn layers_panel(app: &mut SculptApp, ui: &mut Ui) {
    panel_header(ui, app, "LAYERS", |_| {});
    let weak = app.theme.weak_text();
    let fs = app.theme.metrics.font_size;
    let (accent, detail, mask) = (app.theme.ui.accent.0, app.theme.ui.target_delta(), app.theme.ui.target_mask());
    let mut act = RowActions::default();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let m = &app.volume;
        match m.layer.as_ref() {
            None if m.sculpt_layers().is_some() => {
                let (active, layers) = m.sculpt_layers().unwrap();
                let on_mask = matches!(m.target, Target::Mask { .. });
                for k in (0..layers.len()).rev() {
                    let info = &layers[k];
                    let selected = active == k && m.target != Target::Form && !(on_mask && info.active_mask.is_some());
                    let (resp, mut row) = row_frame(ui, app, 0.0, selected, detail);
                    if icon_button(&mut row, if info.visible { Icon::Eye } else { Icon::EyeOff }, 20.0, false, "Show or hide this layer").clicked() {
                        act.visible = Some((k, !info.visible));
                    }
                    let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                    Icon::Paint.paint(row.painter(), r, row.visuals().text_color());
                    let name = RichText::new(&info.name);
                    row.label(if info.visible { name } else { name.color(weak) }).on_hover_text(format!("{} blocks", fmt_count(info.blocks)));
                    row.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let hover = if m.mesh.is_some() { "Remove this layer and what it shows (undo cannot bring it back: the history goes)" } else { "Remove this layer and what it shows (undo brings it back)" };
                        if icon_button(ui, Icon::Trash, 20.0, false, hover).clicked() {
                            act.remove = Some(k);
                        }
                        if let Some(o) = row_opacity(ui, info.opacity, MAX_OPACITY, "The layer's opacity: drag, or click to type. Over 100% exaggerates it.") {
                            act.opacity = Some((k, o));
                        }
                        if let Some(b) = row_blend(ui, fs, ("row_layer_blend", k), info.blend, &LayerBlend::ALL, LayerBlend::label, layer_blend_hover) {
                            act.blend = Some((k, b));
                        }
                    });
                    if resp.clicked() {
                        act.pick = Some((k, None));
                    }
                    // Its masks, top first.
                    for mi in (0..info.masks.len()).rev() {
                        let mk = &info.masks[mi];
                        let sel = active == k && on_mask && info.active_mask == Some(mi);
                        let (resp, mut row) = row_frame(ui, app, 26.0, sel, mask);
                        if icon_button(&mut row, if mk.visible { Icon::Eye } else { Icon::EyeOff }, 20.0, false, "Show or hide this mask").clicked() {
                            act.mask_set = Some((k, mi, MaskInfo { visible: !mk.visible, ..mk.clone() }));
                        }
                        let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                        let icon = if matches!(mk.kind, MaskKind::Noise(_)) { Icon::Noise } else { Icon::Mask };
                        icon.paint(row.painter(), r, mask);
                        let name = RichText::new(&mk.name);
                        row.label(if mk.visible { name } else { name.color(weak) });
                        row.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if icon_button(ui, Icon::Trash, 20.0, false, "Remove this mask").clicked() {
                                act.mask_remove = Some((k, mi));
                            }
                            if let Some(o) = row_opacity(ui, mk.opacity, 1.0, "The mask's opacity: drag, or click to type.") {
                                act.mask_set = Some((k, mi, MaskInfo { opacity: o, ..mk.clone() }));
                            }
                            if let Some(b) = row_blend(ui, fs, ("row_mask_blend", k, mi), mk.blend, &BlendMode::ALL, BlendMode::label, |_| "How this mask goes onto the masks under it.") {
                                act.mask_set = Some((k, mi, MaskInfo { blend: b, ..mk.clone() }));
                            }
                        });
                        if resp.clicked() {
                            act.pick = Some((k, Some(mi)));
                        }
                    }
                    if active == k {
                        let (_, mut row) = row_frame(ui, app, 26.0, false, mask);
                        if row.small_button("+ Paint mask").on_hover_text("A mask to paint, white to start (the layer looks the same). Strokes paint it at once.").clicked() {
                            act.add_mask = Some(MaskKind::Paint { fill: 1.0 });
                        }
                        if row.small_button("+ Noise mask").on_hover_text("Noise that fades the layer: set its look in Properties.").clicked() {
                            act.add_mask = Some(MaskKind::Noise(NoiseMask::default()));
                        }
                        let painted = info.active_mask.and_then(|i| info.masks.get(i)).is_some_and(|mk| matches!(mk.kind, MaskKind::Paint { .. }));
                        if on_mask && painted {
                            let (_, mut row) = row_frame(ui, app, 26.0, false, mask);
                            row.label(RichText::new("Paint").color(weak));
                            for (show, label, hover) in [(false, "Black", "Strokes hide the layer where they go (X swaps)."), (true, "White", "Strokes show it again (X swaps).")] {
                                if row.selectable_label(m.target == Target::Mask { show }, label).on_hover_text(hover).clicked() {
                                    act.target = Some(Target::Mask { show });
                                }
                            }
                        }
                    }
                }
                let (resp, mut row) = row_frame(ui, app, 0.0, m.target == Target::Form, accent);
                row.add_space(24.0);
                let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                Icon::Base.paint(row.painter(), r, app.theme.viewport.clay.0);
                row.label("Base");
                if resp.clicked() {
                    act.target = Some(Target::Form);
                }
                ui.add_space(8.0);
                if m.mesh.is_some() {
                    note(ui, "Each layer keeps what its strokes change on a mesh taken from the volume's surface, and rides along when the base is sculpted. Its opacity, blend and masks (one painted, one noise) fade it. Pick the painted mask to paint it.", weak);
                } else {
                    note(ui, "Each layer keeps what its strokes change, at any detail. Its opacity and masks fade it: masks stack bottom to top, each blended onto those under it. Pick a painted mask to paint it. The base is what no layer holds.", weak);
                }
            }
            None => {
                let (_, mut row) = row_frame(ui, app, 0.0, true, accent);
                let (r, _) = row.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
                Icon::Base.paint(row.painter(), r, app.theme.viewport.clay.0);
                row.label("Volume");
                row.with_layout(Layout::right_to_left(Align::Center), |ui| ui.label(RichText::new(format!("{} levels", m.vol.levels())).size(fs * 0.85).color(weak)));
                ui.add_space(8.0);
                note(ui, "One volume: strokes work on all of it. + Layer adds a sculpt layer (maskable, its resolution growing with the work on it), a generator or surface noise.", weak);
            }
            Some(layer) => {
                let (active, layers) = (layer.active(), layer.layers());
                for k in (0..layers.len()).rev() {
                    let info = &layers[k];
                    let selected = active == k && m.target != Target::Form;
                    let bar = match m.routed() {
                        Target::Mask { .. } => mask,
                        Target::Detail => detail,
                        _ => accent,
                    };
                    let (resp, mut row) = row_frame(ui, app, 0.0, selected, bar);
                    if icon_button(&mut row, if info.visible { Icon::Eye } else { Icon::EyeOff }, 20.0, false, "Show or hide this layer").clicked() {
                        act.visible = Some((k, !info.visible));
                    }
                    let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                    let icon = if info.noise {
                        Icon::Noise
                    } else if info.generator {
                        Icon::Scatter
                    } else {
                        Icon::Paint
                    };
                    icon.paint(row.painter(), r, row.visuals().text_color());
                    let name = RichText::new(&info.name);
                    row.label(if info.visible { name } else { name.color(weak) });
                    row.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icon_button(ui, Icon::Trash, 20.0, false, "Remove this layer (undo brings it back)").clicked() {
                            act.remove = Some(k);
                        }
                        let what = if info.noise {
                            "noise".to_string()
                        } else {
                            format!("L{}  ·  {} blocks", info.level, fmt_count(info.blocks))
                        };
                        ui.label(RichText::new(what).size(fs * 0.85).color(weak));
                    });
                    if resp.clicked() {
                        act.select = Some(k);
                    }
                    // The active layer's mask (and a generator's painted density): what else strokes can paint.
                    if active == k {
                        let (_, mut row) = row_frame(ui, app, 26.0, false, mask);
                        let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                        Icon::Mask.paint(row.painter(), r, mask);
                        for (show, label, hover) in [(false, "Hide", "Strokes hide this layer where they go."), (true, "Show", "Strokes show it again.")] {
                            if row.selectable_label(m.target == Target::Mask { show }, label).on_hover_text(hover).clicked() {
                                act.target = Some(Target::Mask { show });
                            }
                        }
                        if layer.generator(k).is_some_and(|s| matches!(s.density, sculpt_core::generator::density::Density::Painted { .. })) {
                            row.separator();
                            for (more, label, hover) in [(false, "Less", "Paint where the rocks stay out. They scatter again when the stroke ends."), (true, "More", "Paint where the rocks may go.")] {
                                if row.selectable_label(m.target == Target::Density { more }, label).on_hover_text(hover).clicked() {
                                    act.target = Some(Target::Density { more });
                                }
                            }
                        }
                    }
                }
                let (resp, mut row) = row_frame(ui, app, 0.0, m.target == Target::Form, accent);
                row.add_space(24.0);
                let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                Icon::Base.paint(row.painter(), r, app.theme.viewport.clay.0);
                row.label("Form");
                row.with_layout(Layout::right_to_left(Align::Center), |ui| ui.label(RichText::new(format!("levels 0 to {}", layer.from() - 1)).size(fs * 0.85).color(weak)));
                if resp.clicked() {
                    act.target = Some(Target::Form);
                }
                ui.add_space(8.0);
                note(ui, "Sculpting the form carries every layer along. Each layer has its own resolution: coarse ones cost little, fine ones hold the smallest detail.", weak);
            }
        }
        ui.add_space(6.0);
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(10, 0)).show(ui, |ui| {
            ui.menu_button(RichText::new("+ Layer"), |ui| volume_menu(app, ui));
        });
        if app.volume.mesh_building() {
            note(ui, "Making the mesh for layers from the volume...", weak);
        } else if let Some(n) = app.volume.mesh_note.as_ref() {
            note(ui, n, weak);
        }
    });
    mesh_merge_question(app, ui.ctx());
    let m = &mut app.volume;
    if let Some(k) = act.select {
        m.set_active_layer(k);
        m.target = Target::Detail;
    }
    if let Some((k, mi)) = act.pick {
        m.select_mask(k, mi);
    }
    if let Some(kind) = act.add_mask {
        m.add_mask(kind);
    }
    if let Some((k, mi, x)) = act.mask_set {
        m.set_mask(k, mi, x);
    }
    if let Some((k, mi)) = act.mask_remove {
        m.remove_mask(k, mi);
    }
    if let Some(t) = act.target {
        m.target = t;
    }
    if let Some((k, on)) = act.visible {
        m.set_layer_visible(k, on);
    }
    if let Some((k, o)) = act.opacity {
        m.set_layer_opacity(k, o);
    }
    if let Some((k, b)) = act.blend {
        m.set_layer_blend(k, b);
    }
    if let Some(k) = act.remove {
        m.remove_layer(k);
    }
}

/// The volume has sculpt layers of its own (an older file) and a mesh layer was asked for: merge them first?
fn mesh_merge_question(app: &mut SculptApp, ctx: &egui::Context) {
    let id = egui::Id::new(MERGE_ASK);
    if !ctx.data(|d| d.get_temp::<bool>(id).unwrap_or(false)) {
        return;
    }
    let (mut yes, mut no) = (false, false);
    egui::Window::new("New sculpt layer").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        ui.label("This volume has sculpt layers of its own (from an older file). Sculpt layers now live on a mesh taken from the volume, and the two kinds do not go together.");
        ui.label("Merge the volume's layers into it as shown (their history goes) and add the new layer?");
        ui.horizontal(|ui| {
            yes = ui.button("Merge and add").clicked();
            no = ui.button("Cancel").clicked();
        });
    });
    if yes || no {
        ctx.data_mut(|d| d.remove_temp::<bool>(id));
    }
    if yes {
        app.volume.add_mesh_layer();
    }
}

// --------------------------------------------------------------- properties

fn properties_panel(app: &mut SculptApp, ui: &mut Ui) {
    panel_header(ui, app, "PROPERTIES", |_| {});
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
            layer_props(app, ui);
            if app.volume.target != Target::Form {
                crate::generator_ui::volume_generator_props(app, ui);
                crate::generator_ui::volume_noise_props(app, ui);
            }
            ui.add_space(6.0);
            brush_props(app, ui);
        });
    });
}

/// A name to edit: what it was changed to, once the field is left.
fn name_field(ui: &mut Ui, salt: (&str, usize, usize), name: &str) -> Option<String> {
    let id = ui.id().with(salt);
    let mut text = ui.data(|d| d.get_temp::<String>(id)).unwrap_or(name.to_string());
    let r = prop(ui, "Name", |ui| ui.text_edit_singleline(&mut text));
    if r.lost_focus() {
        ui.data_mut(|d| d.remove_temp::<String>(id));
        if text != name && !text.trim().is_empty() {
            return Some(text);
        }
    } else if r.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text));
    }
    None
}

/// A sculpt layer's mask: its blend, opacity, and what it is (painted, or noise and its look).
#[allow(clippy::too_many_arguments)]
fn mask_props(m: &mut VolumeMode, ui: &mut Ui, fs: f32, weak: Color32, k: usize, mi: usize, mk: MaskInfo, count: usize) {
    let mut x = mk.clone();
    let (mut fill, mut remove, mut moved, mut show) = (None, false, None, None);
    let title = if matches!(mk.kind, MaskKind::Noise(_)) { "NOISE MASK" } else { "PAINTED MASK" };
    section(ui, fs, "mask", title, |ui| {
        if let Some(name) = name_field(ui, ("mask_name", k, mi), &mk.name) {
            x.name = name;
        }
        prop(ui, "", |ui| ui.checkbox(&mut x.visible, "Visible"));
        let mut pct = x.opacity * 100.0;
        if prop(ui, "Opacity", |ui| ui.add(egui::Slider::new(&mut pct, 0.0..=100.0).suffix("%").fixed_decimals(0))).changed() {
            x.opacity = pct / 100.0;
        }
        prop(ui, "Blend", |ui| {
            egui::ComboBox::from_id_salt(("mask_blend", k, mi)).selected_text(x.blend.label()).show_ui(ui, |ui| {
                for b in BlendMode::ALL {
                    ui.selectable_value(&mut x.blend, b, b.label());
                }
            })
        });
        match &mut x.kind {
            MaskKind::Paint { .. } => {
                prop(ui, "Paint", |ui| {
                    for (s, label) in [(false, "Black"), (true, "White")] {
                        if ui.selectable_label(m.target == Target::Mask { show: s }, label).clicked() {
                            show = Some(s);
                        }
                    }
                });
                prop(ui, "Fill", |ui| {
                    if ui.button("Black").on_hover_text("Fill it black: what was painted on it goes. One undo.").clicked() {
                        fill = Some(0.0);
                    }
                    if ui.button("White").on_hover_text("Fill it white: what was painted on it goes. One undo.").clicked() {
                        fill = Some(1.0);
                    }
                });
                ui.label(RichText::new("Strokes paint this mask (X swaps black and white).").small().color(weak));
            }
            MaskKind::Noise(n) => {
                prop(ui, "Kind", |ui| {
                    egui::ComboBox::from_id_salt(("mask_noise", k, mi)).selected_text(noise_label(n.params.kind)).show_ui(ui, |ui| {
                        for kind in [NoiseKind::Fbm, NoiseKind::Perlin, NoiseKind::Ridged, NoiseKind::Turbulence, NoiseKind::Cellular] {
                            ui.selectable_value(&mut n.params.kind, kind, noise_label(kind));
                        }
                    })
                });
                prop(ui, "Scale", |ui| ui.add(egui::Slider::new(&mut n.params.scale, 0.5..=200.0).logarithmic(true)));
                prop(ui, "Detail", |ui| ui.add(egui::Slider::new(&mut n.params.octaves, 1..=8)));
                prop(ui, "Balance", |ui| ui.add(egui::Slider::new(&mut n.balance, 0.0..=1.0)));
                prop(ui, "Contrast", |ui| ui.add(egui::Slider::new(&mut n.contrast, 0.0..=0.99)));
                prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut n.params.seed)));
                prop(ui, "", |ui| ui.checkbox(&mut n.invert, "Invert"));
            }
        }
        prop(ui, "Order", |ui| {
            if ui.add_enabled(mi + 1 < count, egui::Button::new("Up")).clicked() {
                moved = Some(true);
            }
            if ui.add_enabled(mi > 0, egui::Button::new("Down")).clicked() {
                moved = Some(false);
            }
            if ui.button("Remove").clicked() {
                remove = true;
            }
        });
        ui.label(RichText::new("Masks stack bottom to top from white, each blended onto those under it.").small().color(weak));
        if !m.live_edits(k) {
            ui.label(RichText::new("A big layer: sliders apply when let go.").small().color(weak));
        }
    });
    if x != mk {
        m.set_mask(k, mi, x);
    }
    if let Some(s) = show {
        m.target = Target::Mask { show: s };
    }
    if let Some(to) = fill {
        m.fill_mask(to);
        m.target = Target::Mask { show: to < 0.5 };
    }
    if let Some(up) = moved {
        m.move_mask(k, mi, up);
    }
    if remove {
        m.remove_mask(k, mi);
    }
}

/// The selected layer's own settings: its name, resolution, visibility and mask; or the form's, or the whole volume's.
fn layer_props(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let m = &mut app.volume;
    let top = m.vol.levels() - 1;
    let level_text = |m: &VolumeMode, l: usize| format!("level {l}  (step {:.5})", m.vol.voxel(l));
    if m.layer.is_none()
        && m.target != Target::Form
        && let Some((k, layers)) = m.sculpt_layers()
    {
        let Some(info) = layers.get(k).cloned() else { return };
        if matches!(m.target, Target::Mask { .. })
            && let Some(mi) = info.active_mask
            && let Some(mk) = info.masks.get(mi).cloned()
        {
            mask_props(m, ui, fs, weak, k, mi, mk, info.masks.len());
            return;
        }
        let live = m.live_edits(k);
        let (mut rename, mut visible, mut fill, mut opacity, mut add, mut blend) = (None, None, None, None, None, None);
        section(ui, fs, "layer", "SCULPT LAYER", |ui| {
            if let Some(name) = name_field(ui, ("layer_name", k, 0), &info.name) {
                rename = Some(name);
            }
            let mut on = info.visible;
            if prop(ui, "", |ui| ui.checkbox(&mut on, "Visible")).changed() {
                visible = Some(on);
            }
            let mut pct = info.opacity * 100.0;
            let slider = egui::Slider::new(&mut pct, 0.0..=MAX_OPACITY * 100.0).suffix("%").fixed_decimals(0);
            if prop(ui, "Opacity", |ui| ui.add(slider).on_hover_text("Over 100% shows the layer's strokes exaggerated.")).changed() {
                opacity = Some(pct / 100.0);
            }
            prop(ui, "Blend", |ui| {
                egui::ComboBox::from_id_salt(("layer_blend", k)).selected_text(info.blend.label()).show_ui(ui, |ui| {
                    for b in LayerBlend::ALL {
                        let hover = layer_blend_hover(b);
                        if ui.selectable_label(info.blend == b, b.label()).on_hover_text(hover).clicked() && info.blend != b {
                            blend = Some(b);
                        }
                    }
                })
            });
            prop(ui, "Blocks", |ui| ui.label(fmt_count(info.blocks)));
            if !live {
                ui.label(RichText::new("A big layer: its sliders apply when let go.").small().color(weak));
            }
            prop(ui, "Masks", |ui| {
                if ui.button("+ Paint").on_hover_text("A mask to paint, white to start. Strokes paint it at once.").clicked() {
                    add = Some(MaskKind::Paint { fill: 1.0 });
                }
                if ui.button("+ Noise").on_hover_text("Noise that fades the layer.").clicked() {
                    add = Some(MaskKind::Noise(NoiseMask::default()));
                }
            });
            prop(ui, "", |ui| {
                if ui.button("Hide all").on_hover_text("Fill the top painted mask black (one is made if there is none), then paint the layer in where wanted. One undo.").clicked() {
                    fill = Some(0.0);
                }
                if ui.button("Show all").on_hover_text("Fill the top painted mask white. One undo.").clicked() {
                    fill = Some(1.0);
                }
            });
        });
        if let Some(name) = rename {
            m.rename_layer(k, &name);
        }
        if let Some(on) = visible {
            m.set_layer_visible(k, on);
        }
        if let Some(o) = opacity {
            m.set_layer_opacity(k, o);
        }
        if let Some(b) = blend {
            m.set_layer_blend(k, b);
        }
        if let Some(kind) = add {
            m.add_mask(kind);
        }
        if let Some(to) = fill {
            m.select_mask(k, None);
            m.fill_mask(to);
            let mi = m.active_mask(k);
            m.select_mask(k, mi);
            m.target = Target::Mask { show: to < 0.5 };
        }
        return;
    }
    let Some(layer) = m.layer.as_ref() else {
        section(ui, fs, "volume", "VOLUME", |ui| {
            let (counts, bytes) = m.vol.memory();
            prop(ui, "Levels", |ui| ui.label(format!("{}  (finest step {:.5})", counts.len(), m.vol.voxel(top))));
            prop(ui, "Blocks", |ui| ui.label(format!("{}  ·  {:.0} MB", fmt_count(counts.iter().sum()), bytes as f64 / 1e6)));
            ui.label(RichText::new("Strokes work at the level that suits the brush's size on screen.").small().color(weak));
        });
        return;
    };
    let from = layer.from();
    if m.target == Target::Form {
        section(ui, fs, "form", "FORM", |ui| {
            prop(ui, "Levels", |ui| ui.label(format!("0 to {}", from - 1)));
            prop(ui, "Blocks", |ui| ui.label(fmt_count(layer.form().memory().0.iter().sum())));
            ui.label(RichText::new("The shape under the layers. Sculpting it carries every layer along.").small().color(weak));
        });
        return;
    }
    let k = layer.active();
    let Some(info) = layer.layers().get(k).cloned() else { return };
    let kind = if info.noise {
        "SURFACE NOISE LAYER"
    } else if info.generator {
        "GENERATOR LAYER"
    } else {
        "DETAIL LAYER"
    };
    let (mut rename, mut level, mut visible, mut fill) = (None, None, None, None);
    section(ui, fs, "layer", kind, |ui| {
        let id = ui.id().with(("layer_name", k));
        let mut name = ui.data(|d| d.get_temp::<String>(id)).unwrap_or(info.name.clone());
        let r = prop(ui, "Name", |ui| ui.text_edit_singleline(&mut name));
        if r.lost_focus() {
            ui.data_mut(|d| d.remove_temp::<String>(id));
            if name != info.name && !name.trim().is_empty() {
                rename = Some(name);
            }
        } else if r.has_focus() {
            ui.data_mut(|d| d.insert_temp(id, name));
        }
        let mut on = info.visible;
        if prop(ui, "", |ui| ui.checkbox(&mut on, "Visible")).changed() {
            visible = Some(on);
        }
        if !info.noise {
            // A generator's resolution can be any level; a detail layer's, the layers' first or finer.
            let first = if info.generator { 0 } else { from };
            let mut l = info.level;
            prop(ui, "Resolution", |ui| {
                egui::ComboBox::from_id_salt(("layer_level", k)).selected_text(level_text(m, l)).show_ui(ui, |ui| {
                    for c in first..=top {
                        ui.selectable_value(&mut l, c, level_text(m, c));
                    }
                })
            });
            if l != info.level {
                level = Some(l);
            }
            prop(ui, "Blocks", |ui| ui.label(fmt_count(info.blocks)));
        }
        prop(ui, "Mask", |ui| {
            if ui.button("Hide all").on_hover_text("Hide this layer everywhere, then paint it in where wanted. One undo.").clicked() {
                fill = Some(0.0);
            }
            if ui.button("Show all").on_hover_text("Clear this layer's mask: it shows everywhere again. One undo.").clicked() {
                fill = Some(1.0);
            }
        });
    });
    if let Some(name) = rename {
        m.settle();
        if let Some(layer) = m.layer.as_mut() {
            layer.rename(k, &name);
        }
    }
    if let Some(on) = visible {
        m.set_layer_visible(k, on);
    }
    if let Some(l) = level {
        m.set_layer_level(k, l);
    }
    if let Some(to) = fill {
        m.fill_mask(to);
        // Hidden everywhere: the next strokes paint it back in.
        if to < 1.0 {
            m.target = Target::Mask { show: true };
        }
    }
}

fn brush_props(app: &mut SculptApp, ui: &mut Ui) {
    let tool = app.tool;
    let title = format!("BRUSH — {}", tool.label().to_uppercase());
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    section(ui, fs, "brush", &title, |ui| {
        let p = app.tools.params_mut(tool);
        prop(ui, "Size", |ui| ui.add(egui::Slider::new(&mut p.size_px, 2.0..=600.0).logarithmic(true).suffix(" px").max_decimals(0)));
        prop(ui, "Strength", |ui| ui.add(egui::Slider::new(&mut p.strength, 0.0..=1.0)));
        prop(ui, "Falloff", |ui| ui.add(egui::Slider::new(&mut p.hardness, 0.0..=0.95)));
        if tool != Tool::Smooth {
            let s = app.tools.params_mut(Tool::Smooth);
            prop(ui, "Shift smooth", |ui| ui.add(egui::Slider::new(&mut s.strength, 0.0..=1.0))).on_hover_text("Strength of the Smooth brush, which Shift gives with any brush");
        }
        ui.label(RichText::new(tool.hint()).small().color(weak));
    });
}

// ------------------------------------------------------------ windows

pub fn dialogs(app: &mut SculptApp, ctx: &egui::Context) {
    let Some(d) = app.dialog.as_mut() else { return };
    let (title, field) = match d {
        Dialog::Open(p) => ("Open volume (.svol)", Some(p)),
        Dialog::SaveAs(p) => ("Save volume as (.svol)", Some(p)),
        Dialog::New(_) => ("New volume", None),
    };
    let mut ok = false;
    let mut cancel = false;
    egui::Window::new(title).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        match field {
            Some(p) => {
                ui.label("Path:");
                let r = ui.add(egui::TextEdit::singleline(p).desired_width(380.0));
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    ok = true;
                }
            }
            None => {
                ui.label("Replace the current volume? Unsaved changes are lost.");
            }
        }
        ui.horizontal(|ui| {
            ok |= ui.button("OK").clicked();
            cancel = ui.button("Cancel").clicked();
        });
    });
    if ok {
        let d = app.dialog.take().unwrap();
        app.run_dialog_action(d);
    } else if cancel {
        app.dialog = None;
    }
}

fn color_row(ui: &mut Ui, label: &str, c: &mut Hex) -> bool {
    ui.label(label);
    let r = ui.color_edit_button_srgba(&mut c.0).changed();
    ui.end_row();
    r
}

fn optional_color_row(ui: &mut Ui, label: &str, c: &mut Option<Hex>, fallback: Color32) -> bool {
    let mut v = c.unwrap_or(Hex(fallback));
    let changed = color_row(ui, label, &mut v);
    if changed {
        *c = Some(v);
    }
    changed
}

pub fn theme_editor(app: &mut SculptApp, ctx: &egui::Context) {
    let Some(mut t) = app.theme_editor.take() else { return };
    let mut open = true;
    let mut changed = false;
    let mut save = false;
    egui::Window::new("Theme editor").open(&mut open).default_width(320.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut t.name);
        });
        changed |= ui.checkbox(&mut t.dark, "Dark base").changed();
        egui::CollapsingHeader::new("Interface").default_open(true).show(ui, |ui| {
            egui::Grid::new("ui_colors").num_columns(2).show(ui, |ui| {
                let header_fallback = t.ui.header();
                let row_fallback = t.ui.row_selected();
                let (detail_fallback, mask_fallback, density_fallback) = (t.ui.target_delta(), t.ui.target_mask(), t.ui.target_freeze());
                let c = &mut t.ui;
                for (l, h) in [
                    ("Window / menu", &mut c.window),
                    ("Panel", &mut c.panel),
                    ("Tray", &mut c.tray),
                    ("Widget", &mut c.widget),
                    ("Widget hover", &mut c.widget_hover),
                    ("Widget active", &mut c.widget_active),
                    ("Text", &mut c.text),
                    ("Weak text", &mut c.text_weak),
                    ("Accent", &mut c.accent),
                    ("Separator", &mut c.separator),
                ] {
                    changed |= color_row(ui, l, h);
                }
                changed |= optional_color_row(ui, "Panel header", &mut c.header, header_fallback);
                changed |= optional_color_row(ui, "Selected row", &mut c.row_selected, row_fallback);
                changed |= optional_color_row(ui, "Detail layer target", &mut c.target_delta, detail_fallback);
                changed |= optional_color_row(ui, "Mask target", &mut c.target_mask, mask_fallback);
                changed |= optional_color_row(ui, "Density target", &mut c.target_freeze, density_fallback);
            });
        });
        egui::CollapsingHeader::new("Viewport").default_open(true).show(ui, |ui| {
            egui::Grid::new("vp_colors").num_columns(2).show(ui, |ui| {
                let c = &mut t.viewport;
                for (l, h) in [("Background top", &mut c.background_top), ("Background bottom", &mut c.background_bottom), ("Clay", &mut c.clay), ("Brush cursor", &mut c.cursor), ("HUD text", &mut c.hud_text)] {
                    changed |= color_row(ui, l, h);
                }
            });
        });
        egui::CollapsingHeader::new("Metrics").show(ui, |ui| {
            let m = &mut t.metrics;
            changed |= ui.add(egui::Slider::new(&mut m.font_size, 9.0..=22.0).text("Font size")).changed();
            changed |= ui.add(egui::Slider::new(&mut m.spacing, 2.0..=14.0).text("Spacing")).changed();
            changed |= ui.add(egui::Slider::new(&mut m.corner_radius, 0..=10).text("Corner radius")).changed();
            changed |= ui.add(egui::Slider::new(&mut m.tray_tile, 40.0..=120.0).text("Tray tile")).changed();
        });
        ui.separator();
        ui.horizontal(|ui| {
            save = ui.button("Save theme").clicked();
            ui.label(RichText::new(app.themes.dir.display().to_string()).small().weak());
        });
    });
    if changed {
        t.apply(ctx);
        app.theme = t.clone();
    }
    if save {
        match app.themes.save(&t) {
            Ok(p) => app.status = format!("Saved theme to {}", p.display()),
            Err(e) => app.status = format!("Theme save failed: {e}"),
        }
        app.set_theme(ctx, t.clone());
    }
    if open {
        app.theme_editor = Some(t);
    }
}

pub fn keymap_window(app: &mut SculptApp, ctx: &egui::Context) {
    if !app.show_keymap {
        return;
    }
    let mut open = true;
    egui::Window::new("Keyboard shortcuts").open(&mut open).show(ctx, |ui| {
        ui.label(RichText::new(format!("Override in {}", crate::app::config_dir().join("keymap.json").display())).weak());
        egui::Grid::new("keys").striped(true).show(ui, |ui| {
            for (s, b) in &app.keymap.bindings {
                ui.label(b.command.label());
                ui.monospace(ctx.format_shortcut(s));
                ui.end_row();
            }
            for (k, what) in [("Alt + LMB / RMB drag", "Orbit"), ("Alt + MMB / MMB drag", "Pan"), ("Alt + RMB drag / wheel", "Zoom"), ("Ctrl + stroke", "Add / carve swapped")] {
                ui.label(what);
                ui.monospace(k);
                ui.end_row();
            }
        });
    });
    app.show_keymap = open;
}

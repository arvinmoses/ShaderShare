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
    let Some(layer) = m.layer.as_ref() else { return ("Strokes edit the volume".into(), ui.accent.0) };
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

const BALLS: [(&str, f32); 3] = [("Ball, 1M triangles", 1e6), ("Ball, 3M triangles", 3e6), ("Ball, 5M triangles", 5e6)];

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

const NOISE_HOVER: &str = "A layer of noise over the whole surface, drawn by the GPU as it is shown: its coarse octaves move the surface, every octave shades it, finer than any level. Its mask fades it. Brushes work on the surface under it.";
const GENERATOR_HOVER: &str = "A layer that scatters a primitive over the form and joins it softly (or cuts it): rocks to start with. Its settings are under Properties; sculpting the form carries it along.";

/// The Volume menu: new layers, merging them, and how big brushes treat detail.
fn volume_menu(app: &mut SculptApp, ui: &mut Ui) {
    let m = &mut app.volume;
    let top = m.vol.levels() - 1;
    if ui.button("New generator layer").on_hover_text(GENERATOR_HOVER).clicked() {
        m.add_generator();
        ui.close();
    }
    if ui.button("New surface noise layer").on_hover_text(NOISE_HOVER).clicked() {
        m.add_noise();
        ui.close();
    }
    match m.layer.as_ref().map(|l| l.from()) {
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
        Some(from) => {
            ui.menu_button("New detail layer", |ui| {
                for l in from..=top {
                    if ui.button(format!("At level {l}  (step {:.5})", m.vol.voxel(l))).clicked() {
                        m.add_layer(l);
                        ui.close();
                    }
                }
            });
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
            None => {
                let (_, mut row) = row_frame(ui, app, 0.0, true, accent);
                let (r, _) = row.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
                Icon::Base.paint(row.painter(), r, app.theme.viewport.clay.0);
                row.label("Volume");
                row.with_layout(Layout::right_to_left(Align::Center), |ui| ui.label(RichText::new(format!("{} levels", m.vol.levels())).size(fs * 0.85).color(weak)));
                ui.add_space(8.0);
                note(ui, "One volume: strokes work on all of it. Add a generator or surface noise layer, or move the finer detail to a layer, to work in layers.", weak);
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
    });
    let m = &mut app.volume;
    if let Some(k) = act.select {
        m.set_active_layer(k);
        m.target = Target::Detail;
    }
    if let Some(t) = act.target {
        m.target = t;
    }
    if let Some((k, on)) = act.visible {
        m.set_layer_visible(k, on);
    }
    if let Some(k) = act.remove {
        m.remove_layer(k);
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

/// The selected layer's own settings: its name, resolution, visibility and mask; or the form's, or the whole volume's.
fn layer_props(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let m = &mut app.volume;
    let top = m.vol.levels() - 1;
    let level_text = |m: &VolumeMode, l: usize| format!("level {l}  (step {:.5})", m.vol.voxel(l));
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

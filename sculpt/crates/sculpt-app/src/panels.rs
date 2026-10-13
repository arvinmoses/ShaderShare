//! Panels in a Substance Painter / Mudbox hybrid layout:
//!
//! * menu bar, then a Painter-style **context toolbar** (active tool, size,
//!   strength, falloff, overlay);
//! * right dock: **LAYERS** stack (layers with masks and mask effects nested
//!   beneath, Painter-style) over **PROPERTIES** (edits the selection);
//! * bottom: Mudbox **tool tray** (Sculpt / Paint / Pose / Falloff) and a
//!   status bar.

use egui::{Align, Color32, Layout, Rect, RichText, Sense, Ui, UiBuilder, vec2};
use sculpt_core::LayerId;
use sculpt_core::bake::MeshAttribute;
use sculpt_core::brush::{Falloff, SmoothMode};
use sculpt_core::mask::{BlendMode, MaskLayer, MaskSource, MaskStack};
use sculpt_core::noise::{NoiseKind, NoiseParams};

use crate::app::{Dialog, SculptApp, Selection, fmt_count};
use crate::icons::{Icon, icon_button};
use crate::keymap::Command;
use crate::theme::{Hex, bold};
use crate::tools::{PoseMode, Tool, Tray};
use crate::viewport::OverlayKind;

const EFFECT_ROW_H: f32 = 24.0;

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
fn section(ui: &mut Ui, font_size: f32, id: &str, title: &str, body: impl FnOnce(&mut Ui)) {
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
fn prop<R>(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(78.0, ui.spacing().interact_size.y), Sense::hover());
        ui.painter().text(r.left_center(), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(ui.style().text_styles[&egui::TextStyle::Body].size * 0.92), ui.visuals().weak_text_color());
        add(ui)
    })
    .inner
}

fn source_icon(s: &MaskSource) -> Icon {
    match s {
        MaskSource::Fill { .. } => Icon::Fill,
        MaskSource::Channel { .. } => Icon::Paint,
        MaskSource::Noise(_) => Icon::Noise,
        MaskSource::Mesh { attribute: MeshAttribute::Curvature } => Icon::Curvature,
        MaskSource::Mesh { attribute: MeshAttribute::Cavity } => Icon::Cavity,
        MaskSource::Mesh { attribute: MeshAttribute::AmbientOcclusion } => Icon::Occlusion,
        MaskSource::Mesh { attribute: MeshAttribute::Thickness } => Icon::Thickness,
        MaskSource::Direction { .. } => Icon::Direction,
        MaskSource::Gradient { .. } => Icon::Gradient,
    }
}

fn source_label(s: &MaskSource) -> &'static str {
    match s {
        MaskSource::Fill { .. } => "Fill",
        MaskSource::Channel { .. } => "Paint",
        MaskSource::Noise(_) => "Noise",
        MaskSource::Mesh { attribute: MeshAttribute::Curvature } => "Curvature",
        MaskSource::Mesh { attribute: MeshAttribute::Cavity } => "Cavity",
        MaskSource::Mesh { attribute: MeshAttribute::AmbientOcclusion } => "Ambient Occlusion",
        MaskSource::Mesh { attribute: MeshAttribute::Thickness } => "Thickness",
        MaskSource::Direction { .. } => "Direction",
        MaskSource::Gradient { .. } => "Gradient",
    }
}

const BLENDS: [BlendMode; 9] = [
    BlendMode::Normal,
    BlendMode::Multiply,
    BlendMode::Add,
    BlendMode::Subtract,
    BlendMode::Screen,
    BlendMode::Overlay,
    BlendMode::Max,
    BlendMode::Min,
    BlendMode::Difference,
];

// ----------------------------------------------------------------- menu bar

pub fn menu_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::top("menu").frame(egui::Frame::side_top_panel(ui.style()).fill(app.theme.ui.window.0)).show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            let (logo, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
            Icon::Base.paint(ui.painter(), logo, app.theme.ui.accent.0);
            ui.add_space(4.0);
            ui.menu_button("File", |ui| {
                ui.menu_button("New sphere", |ui| {
                    for (level, label) in [(5u32, "6k faces"), (6, "25k"), (7, "98k"), (8, "393k"), (9, "1.6M"), (10, "6.3M")] {
                        if ui.button(format!("Level {level} — {label}")).clicked() {
                            app.dialog = Some(Dialog::NewSphere(level));
                            ui.close();
                        }
                    }
                });
                cmd_button(app, ui, Command::Open);
                cmd_button(app, ui, Command::Save);
                if ui.button("Save As…").clicked() {
                    app.dialog = Some(Dialog::SaveAs(app.project_path.as_ref().map(|p| p.display().to_string()).unwrap_or("untitled.sculpt".into())));
                    ui.close();
                }
                ui.separator();
                if ui.button("Import OBJ…").clicked() {
                    app.dialog = Some(Dialog::ImportObj(String::new()));
                    ui.close();
                }
                if ui.button("Export OBJ…").clicked() {
                    app.dialog = Some(Dialog::ExportObj("export.obj".into()));
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                cmd_button(app, ui, Command::Undo);
                cmd_button(app, ui, Command::Redo);
                ui.separator();
                cmd_button(app, ui, Command::InvertFreeze);
                cmd_button(app, ui, Command::ClearFreeze);
                ui.separator();
                if ui.button("Keyboard shortcuts…").clicked() {
                    app.show_keymap = true;
                    ui.close();
                }
            });
            ui.menu_button("Mesh", |ui| {
                cmd_button(app, ui, Command::Subdivide);
                cmd_button(app, ui, Command::NewLayer);
                ui.separator();
                ui.menu_button("Bake mesh maps", |ui| {
                    for a in MeshAttribute::ALL {
                        if ui.button(source_label(&MaskSource::Mesh { attribute: a })).clicked() {
                            app.bake(a);
                            ui.close();
                        }
                    }
                });
                if ui.button("Store rest pose").on_hover_text("Re-anchor procedural masks to the current base mesh").clicked() {
                    if let Some(d) = app.doc.as_mut() {
                        let _ = d.store_rest_pose();
                    }
                    ui.close();
                }
            });
            ui.menu_button("Display", |ui| {
                cmd_button(app, ui, Command::FrameMesh);
                cmd_button(app, ui, Command::ToggleHud);
                ui.add(egui::Slider::new(&mut app.overlay_strength, 0.0..=1.0).text("Overlay opacity"));
                ui.separator();
                ui.label("Viewport detail (dense meshes)");
                for d in crate::app::ViewDetail::ALL {
                    if ui.radio(app.view_detail == d, d.name()).on_hover_text(d.hint()).clicked() {
                        app.view_detail = d;
                        app.save_settings();
                    }
                }
                ui.separator();
                if ui.button("Mesh info…").clicked() {
                    app.show_mesh_info = true;
                    ui.close();
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
            let title = app.project_path.as_ref().and_then(|p| p.file_name()).map_or("Untitled".to_string(), |n| n.to_string_lossy().into_owned());
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
            ui.toggle_value(&mut p.front_faces_only, "Front only").on_hover_text("Ignore back-facing surfaces");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if icon_button(ui, Icon::Eye, 22.0, app.hud, "Performance HUD (H)").clicked() {
                    app.hud = !app.hud;
                }
                let current = overlay_label(app);
                egui::ComboBox::from_id_salt("overlay").selected_text(current).width(130.0).show_ui(ui, |ui| {
                    let mut overlays = vec![("None".to_string(), OverlayKind::None), ("Freeze".into(), OverlayKind::Freeze), ("Layer mask".into(), OverlayKind::LayerMask)];
                    if let Some(d) = &app.doc {
                        overlays.extend(d.channels().keys().map(|n| (n.clone(), OverlayKind::Channel(n.clone()))));
                    }
                    for (label, kind) in overlays {
                        ui.selectable_value(&mut app.overlay, kind, label);
                    }
                });
                ui.label(RichText::new("Overlay").weak());
            });
        });
    });
}

fn overlay_label(app: &SculptApp) -> String {
    match &app.overlay {
        OverlayKind::None => "None".into(),
        OverlayKind::Freeze => "Freeze".into(),
        OverlayKind::LayerMask => "Layer mask".into(),
        OverlayKind::Channel(n) => n.clone(),
        OverlayKind::Custom => "Pose weights".into(),
    }
}

// --------------------------------------------------------------- status bar

pub fn status_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::bottom("status").exact_size(22.0).frame(egui::Frame::side_top_panel(ui.style()).fill(app.theme.ui.window.0)).show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            let weak = app.theme.weak_text();
            ui.label(RichText::new(app.tool.hint()).color(weak));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if let Some(d) = &app.doc {
                    ui.label(RichText::new(format!("{} faces", fmt_count(d.face_count()))).color(weak));
                    ui.separator();
                    let _ = d;
                    if let Some(c) = crate::layer_panel::breadcrumb::crumbs(app) {
                        ui.label(RichText::new(format!("{} — {}", c.parts.join(" › "), c.verb)).color(c.color));
                    }
                    ui.separator();
                }
                ui.label(RichText::new(format!("Pen: {} {:.2}", app.pen.source.label(), app.pen.pressure)).color(weak));
                // Transient messages fade out after a few seconds.
                if app.status != app.status_seen.0 {
                    app.status_seen = (app.status.clone(), std::time::Instant::now());
                }
                let age = app.status_seen.1.elapsed().as_secs_f32();
                if !app.status.is_empty() && age < 6.0 {
                    let alpha = (6.0 - age).clamp(0.0, 1.0);
                    ui.separator();
                    ui.label(RichText::new(&app.status).color(weak.gamma_multiply(alpha)));
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
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (t, label) in [(Tray::Sculpt, "Sculpt Tools"), (Tray::Paint, "Paint Tools"), (Tray::Pose, "Pose Tools"), (Tray::Falloff, "Falloff")] {
                let sel = app.tray == t;
                let text = if sel { RichText::new(label).strong() } else { RichText::new(label).color(app.theme.weak_text()) };
                if ui.add(egui::Button::selectable(sel, text).corner_radius(0)).clicked() {
                    app.tray = t;
                }
            }
        });
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if app.tray == Tray::Falloff {
                falloff_tray(app, ui, tile);
                return;
            }
            let tray_tools: Vec<Tool> = Tool::ALL.into_iter().filter(|t| t.tray() == app.tray).collect();
            for t in tray_tools {
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

/// Mudbox's Falloff tray: curve presets applied to the current tool.
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
        // The stack keeps 42% to 75% of the dock whatever the window size, so it never collapses to a few rows.
        egui::Panel::top("layers_panel")
            .resizable(true)
            .default_size(total * 0.55)
            .size_range((total * 0.42).max(200.0)..=(total * 0.75))
            .frame(egui::Frame::NONE.fill(fill))
            .show(ui, |ui| crate::layer_panel::layers_panel(app, ui));
        egui::CentralPanel::no_frame().show(ui, |ui| properties_panel(app, ui));
    });
}

/// Keep the mask editing copy pointed at the active layer. Returns true if
/// the active layer changed since the copy was made.
pub(crate) fn sync_mask_edit(app: &mut SculptApp) -> bool {
    let Some(doc) = app.doc.as_ref() else { return false };
    let active = doc.active_layer();
    if app.mask_edit.as_ref().map(|(id, _)| *id) == active && active.is_some() {
        return false;
    }
    let stored = active.and_then(|id| doc.layer(id).and_then(|l| l.mask.clone()).map(|s| (id, s)));
    let changed = app.mask_edit.as_ref().map(|(id, _)| *id) != stored.as_ref().map(|(id, _)| *id) || app.last_active != active;
    app.mask_edit = stored;
    app.last_active = active;
    changed
}

pub(crate) fn select_layer(app: &mut SculptApp, id: Option<LayerId>, sel: Selection) {
    if let Some(d) = app.doc.as_mut() {
        let _ = d.set_active_layer(id);
    }
    sync_mask_edit(app);
    app.selection = sel;
    match sel {
        Selection::Layer => {
            if app.overlay == OverlayKind::LayerMask {
                app.overlay = OverlayKind::None;
            }
        }
        Selection::Mask => app.overlay = OverlayKind::LayerMask,
        Selection::Effect(i) => {
            app.overlay = OverlayKind::LayerMask;
            // Selecting a Paint effect makes Mask Paint draw into it (Painter behaviour).
            if let Some(MaskSource::Channel { name }) = app.mask_edit.as_ref().and_then(|(_, s)| s.layers.get(i)).map(|l| l.source.clone()) {
                app.tools.mask_channel = name;
            }
        }
    }
}

pub(crate) fn add_mask(app: &mut SculptApp, base: f32) {
    let Some(id) = app.doc.as_ref().and_then(|d| d.active_layer()) else { return };
    app.mask_edit = Some((id, MaskStack::new(base)));
    app.expanded.insert(id);
    app.mask_dirty = true;
    app.selection = Selection::Mask;
    app.overlay = OverlayKind::LayerMask;
}

pub(crate) fn add_effect(app: &mut SculptApp, label: &str, src: MaskSource) {
    if app.mask_edit.is_none() {
        add_mask(app, 1.0);
    }
    let Some((id, stack)) = app.mask_edit.as_mut() else { return };
    let blend = if stack.layers.is_empty() && stack.base >= 1.0 { BlendMode::Multiply } else { BlendMode::Normal };
    let src = match src {
        MaskSource::Channel { .. } => MaskSource::Channel { name: format!("paint.layer{}", id.0) },
        s => s,
    };
    let is_paint = matches!(src, MaskSource::Channel { .. });
    stack.layers.push(MaskLayer::new(label, src).blend(blend));
    let idx = stack.layers.len() - 1;
    let id = *id;
    app.expanded.insert(id);
    app.mask_dirty = true;
    select_layer(app, Some(id), Selection::Effect(idx));
    if is_paint {
        app.select_tool(Tool::MaskPaint);
        app.overlay = OverlayKind::LayerMask;
    }
}

pub(crate) fn effect_menu(app: &mut SculptApp, ui: &mut Ui) {
    let items: [(&str, MaskSource); 9] = [
        ("Paint", MaskSource::Channel { name: String::new() }),
        ("Fill", MaskSource::Fill { value: 1.0 }),
        ("Noise", MaskSource::Noise(NoiseParams::default())),
        ("Curvature", MaskSource::Mesh { attribute: MeshAttribute::Curvature }),
        ("Cavity", MaskSource::Mesh { attribute: MeshAttribute::Cavity }),
        ("Ambient Occlusion", MaskSource::Mesh { attribute: MeshAttribute::AmbientOcclusion }),
        ("Thickness", MaskSource::Mesh { attribute: MeshAttribute::Thickness }),
        ("Direction", MaskSource::Direction { axis: glam::Vec3::Y, sharpness: 1.0 }),
        ("Gradient", MaskSource::Gradient { axis: glam::Vec3::Y, from: -1.0, to: 1.0 }),
    ];
    for (label, src) in items {
        let resp = ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
            source_icon(&src).paint(ui.painter(), r, ui.visuals().text_color());
            ui.add(egui::Button::new(label).frame(false))
        });
        if resp.inner.clicked() {
            add_effect(app, label, src);
            ui.close();
        }
    }
}

/// Row background + a child Ui for its contents.
fn row_frame(ui: &mut Ui, app: &SculptApp, height: f32, indent: f32, selected: bool) -> (egui::Response, Ui) {
    row_frame_tinted(ui, app, height, indent, selected, None)
}

/// Like `row_frame`, with an optional selection fill (mask rows use the mask colour so they read differently from layers).
fn row_frame_tinted(ui: &mut Ui, app: &SculptApp, height: f32, indent: f32, selected: bool, tint: Option<Color32>) -> (egui::Response, Ui) {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click_and_drag());
    let bg = if selected {
        tint.unwrap_or_else(|| app.theme.ui.row_selected())
    } else if resp.hovered() {
        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.5)
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 0.0, bg);
    if selected {
        let bar = if tint.is_some() { app.theme.ui.target_mask() } else { app.theme.ui.accent.0 };
        ui.painter().rect_filled(Rect::from_min_max(rect.left_top(), rect.left_bottom() + vec2(2.0, 0.0)), 0.0, bar);
    }
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(1.0, app.theme.ui.separator.0));
    // Same right edge as the layer rows' strength column, so numbers line up down the list.
    let inner = rect.shrink2(vec2(4.0, 0.0)).with_min_x(rect.left() + 4.0 + indent).with_max_x(rect.right() - 6.0);
    let child = ui.new_child(UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)));
    (resp, child)
}

pub(crate) fn mask_rows(app: &mut SculptApp, ui: &mut Ui, id: LayerId, depth: usize) {
    let base_indent = depth as f32 * crate::layer_panel::row::INDENT;
    let active = app.doc.as_ref().unwrap().active_layer() == Some(id);
    let stack = if active { app.mask_edit.as_ref().map(|(_, s)| s.clone()) } else { app.doc.as_ref().unwrap().layer(id).and_then(|l| l.mask.clone()) };
    let Some(stack) = stack else { return };
    let weak = app.theme.weak_text();

    // Effects, top of the stack first.
    let mut op_rows: Vec<crate::layer_panel::op_drag::OpRow> = Vec::new();
    for i in (0..stack.layers.len()).rev() {
        let e = &stack.layers[i];
        let selected = active && app.selection == Selection::Effect(i);
        let tint = app.theme.ui.target_mask().gamma_multiply(0.32);
        let (resp, mut row) = row_frame_tinted(ui, app, EFFECT_ROW_H, 40.0 + base_indent, selected, Some(tint));
        op_rows.push((i, resp.rect));
        if resp.drag_started() && active {
            app.layers.op_drag = Some((id, i));
        }
        row.spacing_mut().item_spacing.x = 4.0;
        if icon_button(&mut row, if e.enabled { Icon::Eye } else { Icon::EyeOff }, 18.0, false, "Enable").clicked() {
            select_layer(app, Some(id), Selection::Effect(i));
            if let Some((_, s)) = app.mask_edit.as_mut() {
                s.layers[i].enabled = !s.layers[i].enabled;
                app.mask_dirty = true;
            }
        }
        let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
        source_icon(&e.source).paint(row.painter(), r, row.visuals().text_color());
        let title = if e.name.is_empty() { source_label(&e.source).to_string() } else { e.name.clone() };
        row.label(RichText::new(title).size(app.theme.metrics.font_size * 0.92));
        row.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(format!("{:?} {:.0}%", e.blend, e.opacity * 100.0)).size(app.theme.metrics.font_size * 0.85).color(weak));
        });
        if resp.clicked() {
            select_layer(app, Some(id), Selection::Effect(i));
        }
        let (count, enabled) = (stack.layers.len(), e.enabled);
        egui::Popup::context_menu(&resp).show(|ui| crate::layer_panel::menus::op_menu(app, ui, id, i, count, enabled));
    }
    crate::layer_panel::op_drag::update(app, ui, id, &op_rows);
    // Mask base value row.
    let selected = active && app.selection == Selection::Mask;
    let (resp, mut row) = row_frame_tinted(ui, app, EFFECT_ROW_H, 40.0 + base_indent, selected, Some(app.theme.ui.target_mask().gamma_multiply(0.32)));
    let (r, _) = row.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
    Icon::Mask.paint(row.painter(), r, row.visuals().text_color());
    row.label(RichText::new(if stack.base >= 0.5 { "White mask" } else { "Black mask" }).size(app.theme.metrics.font_size * 0.92).color(weak));
    if resp.clicked() {
        select_layer(app, Some(id), Selection::Mask);
    }
    egui::Popup::context_menu(&resp).show(|ui| crate::layer_panel::menus::add_mask_items(app, ui, id));
}

pub(crate) fn base_row(app: &mut SculptApp, ui: &mut Ui) {
    let selected = app.doc.as_ref().unwrap().active_layer().is_none();
    let (resp, mut row) = row_frame(ui, app, crate::layer_panel::row::Metrics::of(app.layers.compact).row_h, 0.0, selected);
    row.add_space(40.0);
    let (sw, _) = row.allocate_exact_size(vec2(22.0, 22.0), Sense::hover());
    row.painter().rect_filled(sw, 2.0, app.theme.viewport.background_bottom.0);
    Icon::Base.paint(row.painter(), sw.shrink(2.0), app.theme.viewport.clay.0.gamma_multiply(0.7));
    row.label(RichText::new("Base").italics());
    if resp.clicked() {
        app.layers.selection.clear();
        select_layer(app, None, Selection::Layer);
    }
}

// --------------------------------------------------------------- properties

fn properties_panel(app: &mut SculptApp, ui: &mut Ui) {
    // Folders can be selected without becoming the sculpt target, so Properties follows the primary row.
    let target = app.doc.as_ref().and_then(|d| app.layers.selection.primary().or(d.active_layer()).and_then(|id| d.layer(id)).map(|l| (l.id, l.is_folder())));
    let active = app.doc.as_ref().and_then(|d| d.active_layer());
    panel_header(ui, app, "PROPERTIES", |_| {});
    egui::Frame::NONE.inner_margin(egui::Margin::symmetric(10, 5)).show(ui, |ui| crate::layer_panel::breadcrumb::show(app, ui));
    ui.painter().line_segment([ui.cursor().left_top(), ui.cursor().right_top()], egui::Stroke::new(1.0, app.theme.ui.separator.0));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
            match (app.selection, target) {
                _ if app.layers.selection.len() > 1 => multi_props(app, ui),
                (Selection::Effect(i), Some((id, false))) if active == Some(id) => effect_props(app, ui, i),
                (Selection::Mask, Some((id, false))) if active == Some(id) => mask_props(app, ui),
                (_, Some((id, _))) => layer_props(app, ui, id),
                _ => {
                    ui.label(RichText::new("Sculpting directly on the base mesh. Add a layer (+) to sculpt non-destructively.").color(app.theme.weak_text()));
                }
            }
            ui.add_space(6.0);
            brush_props(app, ui);
        });
    });
}

/// Several rows selected: shared fields, with a mixed state where they differ. Each change applies to all.
fn multi_props(app: &mut SculptApp, ui: &mut Ui) {
    use crate::layer_panel::command::{self, LayerCommand, MetaEdit};
    let Some(doc) = app.doc.as_ref() else { return };
    let order: Vec<LayerId> = doc.layer_tree().iter().map(|r| r.id).collect();
    let ids = app.layers.selection.in_order(&order);
    let metas: Vec<_> = ids.iter().filter_map(|id| doc.layer(*id)).map(|l| (l.is_folder(), l.meta())).collect();
    let Some(first) = metas.first().map(|(_, m)| m.clone()) else { return };
    let all = |f: &dyn Fn(&sculpt_core::LayerMeta) -> bool| metas.iter().all(|(_, m)| f(m));
    let (mixed_strength, mixed_vis, mixed_lock) = (!all(&|m| m.opacity == first.opacity), !all(&|m| m.visible == first.visible), !all(&|m| m.locked == first.locked));
    let sculpt: Vec<_> = metas.iter().filter(|(f, _)| !f).map(|(_, m)| m.blend).collect();
    let mixed_blend = sculpt.windows(2).any(|w| w[0] != w[1]);
    let blend = sculpt.first().copied();
    let fs = app.theme.metrics.font_size;
    section(ui, fs, "multi", &format!("{} LAYERS", ids.len()), |ui| {
        // Strength is applied when the drag ends, so one gesture is one undo step.
        let key = ui.id().with("multi_strength");
        let shown = ui.data(|d| d.get_temp::<f32>(key)).unwrap_or(first.opacity * 100.0);
        let mut pct = shown;
        let r = prop(ui, if mixed_strength { "Strength (mixed)" } else { "Strength" }, |ui| ui.add(egui::Slider::new(&mut pct, -100.0..=200.0).suffix("%").max_decimals(0)));
        if r.dragged() || r.has_focus() {
            ui.data_mut(|d| d.insert_temp(key, pct));
        }
        if r.drag_stopped() || r.lost_focus() || (r.changed() && !r.dragged() && !r.has_focus()) {
            ui.data_mut(|d| d.remove_temp::<f32>(key));
            command::execute(app, LayerCommand::EditMany { ids: ids.clone(), edit: MetaEdit::Strength(pct / 100.0) });
        }
        if let Some(current) = blend {
            prop(ui, "Blend", |ui| {
                egui::ComboBox::from_id_salt("multi_blend").selected_text(if mixed_blend { "Mixed" } else { current.label() }).show_ui(ui, |ui| {
                    for mode in sculpt_core::LayerBlend::ALL {
                        if ui.selectable_label(!mixed_blend && current == mode, mode.label()).clicked() {
                            command::execute(app, LayerCommand::EditMany { ids: ids.clone(), edit: MetaEdit::Blend(mode) });
                        }
                    }
                })
            });
        }
        let (mut v, mut lk) = (first.visible, first.locked);
        prop(ui, "", |ui| {
            if ui.add(egui::Checkbox::new(&mut v, "Visible").indeterminate(mixed_vis)).changed() {
                command::execute(app, LayerCommand::EditMany { ids: ids.clone(), edit: MetaEdit::Visible(v) });
            }
            if ui.add(egui::Checkbox::new(&mut lk, "Locked").indeterminate(mixed_lock)).changed() {
                command::execute(app, LayerCommand::EditMany { ids: ids.clone(), edit: MetaEdit::Locked(lk) });
            }
        });
        prop(ui, "", |ui| {
            if ui.button("Group").on_hover_text("Put the selection in a new folder (Ctrl+G)").clicked() {
                command::execute(app, LayerCommand::Group);
            }
            if ui.button("Duplicate").clicked() {
                command::execute(app, LayerCommand::Duplicate);
            }
        });
    });
}

fn layer_props(app: &mut SculptApp, ui: &mut Ui, id: LayerId) {
    use crate::layer_panel::command::{self, LayerCommand, MetaEdit};
    let Some(l) = app.doc.as_ref().and_then(|d| d.layer(id)) else { return };
    let (name, opacity, visible, locked, has_mask, is_folder, blend) = (l.name.clone(), l.opacity, l.visible, l.locked, l.mask.is_some(), l.is_folder(), l.blend);
    let fs = app.theme.metrics.font_size;
    section(ui, fs, "layer", if is_folder { "FOLDER" } else { "LAYER" }, |ui| {
        let mut n = name.clone();
        if prop(ui, "Name", |ui| ui.text_edit_singleline(&mut n)).lost_focus() && n != name {
            command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Name(n), coalesce: false });
        }
        let mut pct = opacity * 100.0;
        let slider = prop(ui, "Strength", |ui| ui.add(egui::Slider::new(&mut pct, -100.0..=200.0).suffix("%").max_decimals(0)));
        if slider.changed() {
            command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Strength(pct / 100.0), coalesce: slider.dragged() && !slider.drag_started() });
        }
        if !is_folder {
            prop(ui, "Blend", |ui| {
                egui::ComboBox::from_id_salt(("layer_blend", id)).selected_text(blend.label()).show_ui(ui, |ui| {
                    for mode in sculpt_core::LayerBlend::ALL {
                        if ui.selectable_label(blend == mode, mode.label()).on_hover_text(mode.hint()).clicked() {
                            command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Blend(mode), coalesce: false });
                        }
                    }
                })
            });
        }
        let (mut v, mut lk) = (visible, locked);
        prop(ui, "", |ui| {
            if ui.checkbox(&mut v, "Visible").changed() {
                command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Visible(v), coalesce: false });
            }
            if ui.checkbox(&mut lk, "Locked").changed() {
                command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Locked(lk), coalesce: false });
            }
        });
        if is_folder {
            let kids = app.doc.as_ref().map_or(0, |d| d.children(Some(id)).len());
            prop(ui, "Contains", |ui| ui.label(RichText::new(format!("{kids} item{}", if kids == 1 { "" } else { "s" })).weak()));
            return;
        }
        prop(ui, "Mask", |ui| {
            if has_mask {
                if ui.button("Edit mask").clicked() {
                    select_layer(app, Some(id), Selection::Mask);
                }
            } else {
                if ui.button("Add white").clicked() {
                    add_mask(app, 1.0);
                }
                if ui.button("Add black").clicked() {
                    add_mask(app, 0.0);
                }
            }
        });
        let mb = app.doc.as_ref().unwrap().layer(id).map_or(0, |l| l.delta_bytes());
        prop(ui, "Data", |ui| ui.label(RichText::new(format!("{:.1} MB sparse", mb as f64 / 1e6)).weak()));
    });
}

fn mask_props(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    section(ui, fs, "mask", "MASK", |ui| {
        let Some((_, stack)) = app.mask_edit.as_mut() else { return };
        let mut changed = prop(ui, "Base value", |ui| ui.add(egui::Slider::new(&mut stack.base, 0.0..=1.0))).changed();
        prop(ui, "", |ui| {
            if ui.button("White").clicked() {
                stack.base = 1.0;
                changed = true;
            }
            if ui.button("Black").clicked() {
                stack.base = 0.0;
                changed = true;
            }
        });
        prop(ui, "Overlay", |ui| {
            let mut show = app.overlay == OverlayKind::LayerMask;
            if ui.checkbox(&mut show, "Show in viewport").changed() {
                app.overlay = if show { OverlayKind::LayerMask } else { OverlayKind::None };
            }
        });
        ui.label(RichText::new("Add effects with the ✦ button in the Layers panel.").small().color(app.theme.weak_text()));
        if changed {
            app.mask_dirty = true;
        }
    });
}

fn effect_props(app: &mut SculptApp, ui: &mut Ui, i: usize) {
    let weak = app.theme.weak_text();
    let fs = app.theme.metrics.font_size;
    let label = app.mask_edit.as_ref().and_then(|(_, s)| s.layers.get(i)).map_or("EFFECT".to_string(), |e| source_label(&e.source).to_uppercase());
    section(ui, fs, "effect", &label, |ui| {
        let Some((_, stack)) = app.mask_edit.as_mut() else { return };
        let n = stack.layers.len();
        if i >= n {
            return;
        }
        let mut changed = false;
        let mut action = None;
        {
            let e = &mut stack.layers[i];
            prop(ui, "Name", |ui| changed |= ui.text_edit_singleline(&mut e.name).lost_focus());
            prop(ui, "", |ui| {
                changed |= ui.checkbox(&mut e.enabled, "Enabled").changed();
                changed |= ui.checkbox(&mut e.invert, "Invert").changed();
            });
            prop(ui, "Blend", |ui| {
                egui::ComboBox::from_id_salt("blend").selected_text(format!("{:?}", e.blend)).show_ui(ui, |ui| {
                    for b in BLENDS {
                        changed |= ui.selectable_value(&mut e.blend, b, format!("{b:?}")).changed();
                    }
                });
            });
            changed |= prop(ui, "Opacity", |ui| ui.add(egui::Slider::new(&mut e.opacity, 0.0..=1.0))).changed();
            ui.add_space(4.0);
            changed |= source_props(ui, &mut e.source, weak);
            ui.add_space(4.0);
            ui.label(RichText::new("Adjustments").strong().size(ui.style().text_styles[&egui::TextStyle::Body].size * 0.9));
            changed |= prop(ui, "Levels in", |ui| {
                let a = ui.add(egui::DragValue::new(&mut e.levels.in_min).range(0.0..=1.0).speed(0.005)).changed();
                let b = ui.add(egui::DragValue::new(&mut e.levels.in_max).range(0.0..=1.0).speed(0.005)).changed();
                a | b
            });
            changed |= prop(ui, "Gamma", |ui| ui.add(egui::Slider::new(&mut e.levels.gamma, 0.1..=5.0).logarithmic(true))).changed();
            changed |= prop(ui, "Blur", |ui| ui.add(egui::Slider::new(&mut e.blur, 0..=20))).changed();
            ui.add_space(4.0);
            prop(ui, "Order", |ui| {
                if icon_button(ui, Icon::ArrowUp, 22.0, false, "Move up").clicked() && i + 1 < n {
                    action = Some(i + 1);
                }
                if icon_button(ui, Icon::ArrowDown, 22.0, false, "Move down").clicked() && i > 0 {
                    action = Some(i - 1);
                }
            });
        }
        if let Some(j) = action {
            stack.layers.swap(i, j);
            app.selection = Selection::Effect(j);
            changed = true;
        }
        if let MaskSource::Channel { name } = &stack.layers[i.min(stack.layers.len() - 1)].source
            && ui.button("Paint into this mask").clicked() {
                app.tools.mask_channel = name.clone();
                app.select_tool(Tool::MaskPaint);
                app.overlay = OverlayKind::LayerMask;
            }
        if changed {
            app.mask_dirty = true;
        }
    });
}

fn source_props(ui: &mut Ui, src: &mut MaskSource, weak: Color32) -> bool {
    let mut changed = false;
    match src {
        MaskSource::Fill { value } => changed |= prop(ui, "Value", |ui| ui.add(egui::Slider::new(value, 0.0..=1.0))).changed(),
        MaskSource::Channel { name } => {
            prop(ui, "Channel", |ui| changed |= ui.text_edit_singleline(name).lost_focus());
            ui.label(RichText::new("Hand-painted with Mask Paint (or imported data).").small().color(weak));
        }
        MaskSource::Noise(p) => {
            prop(ui, "Type", |ui| {
                egui::ComboBox::from_id_salt("noise_kind").selected_text(format!("{:?}", p.kind)).show_ui(ui, |ui| {
                    for k in [NoiseKind::Fbm, NoiseKind::Perlin, NoiseKind::Ridged, NoiseKind::Turbulence, NoiseKind::Cellular] {
                        changed |= ui.selectable_value(&mut p.kind, k, format!("{k:?}")).changed();
                    }
                });
            });
            changed |= prop(ui, "Scale", |ui| ui.add(egui::Slider::new(&mut p.scale, 0.1..=50.0).logarithmic(true))).changed();
            changed |= prop(ui, "Octaves", |ui| ui.add(egui::Slider::new(&mut p.octaves, 1..=8))).changed();
            changed |= prop(ui, "Roughness", |ui| ui.add(egui::Slider::new(&mut p.gain, 0.1..=0.9))).changed();
            changed |= prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut p.seed))).changed();
        }
        MaskSource::Mesh { attribute } => {
            ui.label(RichText::new(format!("Reads the '{}' bake. Re-bake from Mesh ▸ Bake mesh maps after sculpting.", attribute.channel_name())).small().color(weak));
        }
        MaskSource::Direction { axis, sharpness } => {
            changed |= axis_prop(ui, axis);
            changed |= prop(ui, "Sharpness", |ui| ui.add(egui::Slider::new(sharpness, 0.1..=8.0))).changed();
        }
        MaskSource::Gradient { axis, from, to } => {
            changed |= axis_prop(ui, axis);
            changed |= prop(ui, "Range", |ui| {
                let a = ui.add(egui::DragValue::new(from).speed(0.01)).changed();
                let b = ui.add(egui::DragValue::new(to).speed(0.01)).changed();
                a | b
            });
        }
    }
    changed
}

fn axis_prop(ui: &mut Ui, axis: &mut glam::Vec3) -> bool {
    prop(ui, "Axis", |ui| {
        let mut changed = false;
        for (label, v) in [("X", glam::Vec3::X), ("Y", glam::Vec3::Y), ("Z", glam::Vec3::Z), ("−Y", -glam::Vec3::Y)] {
            if ui.selectable_label(*axis == v, label).clicked() {
                *axis = v;
                changed = true;
            }
        }
        changed
    })
}

fn brush_props(app: &mut SculptApp, ui: &mut Ui) {
    let tool = app.tool;
    let title = format!("BRUSH — {}", tool.label().to_uppercase());
    let fs = app.theme.metrics.font_size;
    section(ui, fs, "brush", &title, |ui| {
        let p = app.tools.params_mut(tool);
        prop(ui, "Size", |ui| ui.add(egui::Slider::new(&mut p.size_px, 2.0..=600.0).logarithmic(true).suffix(" px").max_decimals(0)));
        prop(ui, "Strength", |ui| ui.add(egui::Slider::new(&mut p.strength, 0.0..=1.0)));
        prop(ui, "Falloff", |ui| ui.add(egui::Slider::new(&mut p.hardness, 0.0..=0.95)));
        prop(ui, "", |ui| ui.checkbox(&mut p.front_faces_only, "Front faces only"));
        match tool {
            Tool::ClayBuildup => {
                let c = &mut app.tools.clay;
                prop(ui, "Height", |ui| ui.add(egui::Slider::new(&mut c.height, 0.02..=0.6)));
                prop(ui, "Squareness", |ui| ui.add(egui::Slider::new(&mut c.squareness, 2.0..=10.0)));
                prop(ui, "", |ui| ui.checkbox(&mut c.accumulate, "Accumulate within stroke"));
            }
            Tool::TrimDynamic => {
                let t = &mut app.tools.trim;
                prop(ui, "Depth", |ui| ui.add(egui::Slider::new(&mut t.depth, 0.0..=0.3)));
                prop(ui, "Smooth border", |ui| ui.add(egui::Slider::new(&mut t.smooth_border, 0.0..=1.0)));
            }
            Tool::Smooth => {
                prop(ui, "Mode", |ui| {
                    ui.selectable_value(&mut app.tools.smooth.mode, SmoothMode::Laplacian, "Strong");
                    ui.selectable_value(&mut app.tools.smooth.mode, SmoothMode::Surface, "Keep form");
                });
            }
            Tool::Move => {
                prop(ui, "", |ui| ui.checkbox(&mut app.tools.move_topological, "Topological"));
            }
            Tool::MaskPaint => {
                prop(ui, "Channel", |ui| ui.text_edit_singleline(&mut app.tools.mask_channel));
            }
            Tool::Pose => {
                prop(ui, "Mode", |ui| {
                    ui.selectable_value(&mut app.tools.pose_mode, PoseMode::Rotate, "Rotate");
                    ui.selectable_value(&mut app.tools.pose_mode, PoseMode::Translate, "Translate");
                });
                prop(ui, "Joint softness", |ui| ui.add(egui::Slider::new(&mut app.tools.pose_softness, 0.0..=1.0)));
                prop(ui, "Mask blur", |ui| ui.add(egui::Slider::new(&mut app.tools.pose_blur, 0..=10)));
            }
            Tool::Freeze => {}
        }
    });
}

/// Push the edited mask stack to the document (deferred until mouse release
/// so slider drags don't re-evaluate the mask every frame).
/// Write the mask editing copy back to the document as one undoable step.
pub fn apply_mask_edit(app: &mut SculptApp) {
    let Some((id, stack)) = app.mask_edit.clone() else { return };
    let write = move |d: &mut sculpt_core::Document| -> Result<(), String> {
        let Some(layer) = d.layer(id) else { return Ok(()) };
        let mut meta = layer.meta();
        meta.mask = Some(stack);
        d.set_layer_meta(id, meta, false).map_err(|e| e.to_string())
    };
    if app.doc.as_ref().is_some_and(|d| d.vertex_count() > 400_000) {
        app.start_doc_job("Evaluating mask", write);
    } else if let Some(d) = app.doc.as_mut()
        && let Err(e) = write(d)
    {
        app.status = e;
    }
}

// ------------------------------------------------------------ windows

pub fn mesh_info(app: &mut SculptApp, ctx: &egui::Context) {
    if !app.show_mesh_info {
        return;
    }
    let mut open = true;
    egui::Window::new("Mesh info").open(&mut open).resizable(false).show(ctx, |ui| {
        let Some(doc) = &app.doc else { return };
        egui::Grid::new("obj").num_columns(2).striped(true).show(ui, |ui| {
            let mb: usize = doc.layers().iter().map(|l| l.delta_bytes()).sum();
            for (k, v) in [
                ("Faces", fmt_count(doc.face_count())),
                ("Vertices", fmt_count(doc.vertex_count())),
                ("Subdivisions", doc.level().to_string()),
                ("Spatial leaves", doc.bvh().leaves.len().to_string()),
                ("Layers", doc.layers().len().to_string()),
                ("Layer data", format!("{:.1} MB", mb as f64 / 1e6)),
                ("Channels", doc.channels().keys().cloned().collect::<Vec<_>>().join(", ")),
                ("Project", app.project_path.as_ref().map_or("unsaved".into(), |p| p.display().to_string())),
                ("Pen", format!("{} — {}", app.pen.source.label(), app.pen.status)),
            ] {
                ui.label(RichText::new(k).weak());
                ui.label(v);
                ui.end_row();
            }
        });
    });
    app.show_mesh_info = open;
}

pub fn dialogs(app: &mut SculptApp, ctx: &egui::Context) {
    let Some(d) = app.dialog.as_mut() else { return };
    let (title, field) = match d {
        Dialog::Open(p) => ("Open project (.sculpt folder)", Some(p)),
        Dialog::SaveAs(p) => ("Save project as", Some(p)),
        Dialog::ImportObj(p) => ("Import OBJ", Some(p)),
        Dialog::ExportObj(p) => ("Export OBJ (composited)", Some(p)),
        Dialog::NewSphere(_) => ("New sphere", None),
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
                ui.label("Replace the current document? Unsaved changes are lost.");
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
            });
        });
        egui::CollapsingHeader::new("Viewport").default_open(true).show(ui, |ui| {
            egui::Grid::new("vp_colors").num_columns(2).show(ui, |ui| {
                let c = &mut t.viewport;
                for (l, h) in [
                    ("Background top", &mut c.background_top),
                    ("Background bottom", &mut c.background_bottom),
                    ("Clay", &mut c.clay),
                    ("Mask overlay", &mut c.overlay),
                    ("Freeze overlay", &mut c.freeze),
                    ("Brush cursor", &mut c.cursor),
                    ("HUD text", &mut c.hud_text),
                ] {
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
            for (k, what) in [("Alt + LMB / RMB drag", "Orbit"), ("Alt + MMB / MMB drag", "Pan"), ("Alt + RMB drag / wheel", "Zoom"), ("Shift + stroke", "Smooth"), ("Ctrl + stroke", "Invert")] {
                ui.label(what);
                ui.monospace(k);
                ui.end_row();
            }
        });
    });
    app.show_keymap = open;
}

//! Mudbox-style panels: menu bar, bottom tool tray, right-hand Layers /
//! Properties / Object tabs, status bar, dialogs and the theme editor.

use egui::{Color32, RichText, Ui};
use sculpt_core::bake::MeshAttribute;
use sculpt_core::brush::SmoothMode;
use sculpt_core::mask::{BlendMode, Levels, MaskLayer, MaskSource, MaskStack};
use sculpt_core::noise::{NoiseKind, NoiseParams};

use crate::app::{Dialog, RightTab, SculptApp, fmt_count};
use crate::keymap::Command;
use crate::theme::Hex;
use crate::tools::{PoseMode, Tool, Tray};
use crate::viewport::OverlayKind;

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

pub fn menu_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::top("menu").show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
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
                ui.label(RichText::new("Bake mesh maps").weak());
                for a in MeshAttribute::ALL {
                    if ui.button(format!("{a:?}")).clicked() {
                        app.bake(a);
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Store rest pose").on_hover_text("Re-anchor procedural masks to the current base mesh").clicked() {
                    if let Some(d) = app.doc.as_mut() {
                        let _ = d.store_rest_pose();
                    }
                    ui.close();
                }
            });
            ui.menu_button("Display", |ui| {
                ui.label(RichText::new("Overlay").weak());
                let mut overlays = vec![("None", OverlayKind::None), ("Freeze", OverlayKind::Freeze), ("Active layer mask", OverlayKind::LayerMask)];
                if let Some(d) = &app.doc {
                    for name in d.channels().keys() {
                        overlays.push((Box::leak(name.clone().into_boxed_str()), OverlayKind::Channel(name.clone())));
                    }
                }
                for (label, kind) in overlays {
                    if ui.radio(app.overlay == kind, label).clicked() {
                        app.overlay = kind;
                    }
                }
                ui.add(egui::Slider::new(&mut app.overlay_strength, 0.0..=1.0).text("Overlay strength"));
                ui.separator();
                cmd_button(app, ui, Command::FrameMesh);
                cmd_button(app, ui, Command::ToggleHud);
                ui.separator();
                ui.label(RichText::new("Theme").weak());
                let names: Vec<String> = app.themes.themes.iter().map(|t| t.name.clone()).collect();
                for n in names {
                    if ui.radio(app.theme.name == n, &n).clicked() {
                        let t = app.themes.get(&n);
                        let ctx = ui.ctx().clone();
                        app.set_theme(&ctx, t);
                    }
                }
                if ui.button("Theme editor…").clicked() {
                    app.theme_editor = Some(app.theme.clone());
                    ui.close();
                }
            });
        });
    });
}

pub fn status_bar(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::bottom("status").exact_size(22.0).show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            if let Some(d) = &app.doc {
                let layer = d.active_layer().and_then(|id| d.layer(id)).map_or("Base mesh".to_string(), |l| l.name.clone());
                ui.label(format!("{} faces · sculpting on: {layer}", fmt_count(d.face_count())));
                ui.separator();
            }
            ui.label(RichText::new(app.tool.hint()).color(app.theme.weak_text()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(&app.status).color(app.theme.weak_text()));
            });
        });
    });
}

pub fn tray(app: &mut SculptApp, ui: &mut Ui) {
    let tile = app.theme.metrics.tray_tile;
    egui::Panel::bottom("tray").exact_size(tile + 40.0).frame(egui::Frame::side_top_panel(ui.style()).fill(app.theme.ui.tray.0)).show(ui, |ui| {
        ui.horizontal(|ui| {
            for (t, label) in [(Tray::Sculpt, "Sculpt Tools"), (Tray::Paint, "Paint Tools"), (Tray::Pose, "Pose Tools")] {
                if ui.selectable_label(app.tray == t, label).clicked() {
                    app.tray = t;
                }
            }
        });
        ui.horizontal(|ui| {
            let tray_tools: Vec<Tool> = Tool::ALL.into_iter().filter(|t| t.tray() == app.tray).collect();
            for t in tray_tools {
                let selected = app.tool == t;
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(tile * 1.25, tile), egui::Sense::click());
                let v = ui.visuals();
                let (bg, fg) = if selected {
                    (v.selection.bg_fill, v.strong_text_color())
                } else if resp.hovered() {
                    (v.widgets.hovered.bg_fill, v.text_color())
                } else {
                    (v.widgets.inactive.bg_fill, v.text_color())
                };
                let painter = ui.painter();
                painter.rect_filled(rect, app.theme.metrics.corner_radius as f32, bg);
                let icon = egui::Rect::from_center_size(rect.center_top() + egui::vec2(0.0, tile * 0.33), egui::vec2(tile * 0.42, tile * 0.42));
                t.paint_icon(painter, icon, fg);
                let font = egui::FontId::proportional(app.theme.metrics.font_size * 0.85);
                painter.text(rect.center_bottom() - egui::vec2(0.0, tile * 0.2), egui::Align2::CENTER_CENTER, t.label(), font.clone(), fg);
                let key = app.tool_hotkey(ui.ctx(), t);
                painter.text(rect.right_top() + egui::vec2(-4.0, 3.0), egui::Align2::RIGHT_TOP, key, egui::FontId::monospace(10.0), fg.gamma_multiply(0.7));
                if resp.on_hover_text(t.hint()).clicked() {
                    app.select_tool(t);
                }
            }
        });
    });
}

pub fn right_panel(app: &mut SculptApp, ui: &mut Ui) {
    egui::Panel::right("right").default_size(330.0).min_size(260.0).show(ui, |ui| {
        ui.horizontal(|ui| {
            for (t, label) in [(RightTab::Layers, "Layers"), (RightTab::Properties, "Properties"), (RightTab::Object, "Object")] {
                if ui.selectable_label(app.right_tab == t, label).clicked() {
                    app.right_tab = t;
                }
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| match app.right_tab {
            RightTab::Layers => layers_tab(app, ui),
            RightTab::Properties => properties_tab(app, ui),
            RightTab::Object => object_tab(app, ui),
        });
    });
}

fn layers_tab(app: &mut SculptApp, ui: &mut Ui) {
    let Some(doc) = app.doc.as_mut() else {
        ui.label("Loading…");
        return;
    };
    ui.horizontal(|ui| {
        if ui.button("+ New").clicked() {
            let n = doc.layers().len() + 1;
            doc.add_layer(&format!("Layer {n}"));
        }
        let active = doc.active_layer();
        ui.add_enabled_ui(active.is_some(), |ui| {
            if ui.button("Flatten").on_hover_text("Bake into the base mesh").clicked() {
                let _ = doc.flatten_layer(active.unwrap());
            }
            if ui.button("Delete").clicked() {
                let _ = doc.remove_layer(active.unwrap());
            }
        });
    });
    ui.add_space(4.0);
    let ids: Vec<_> = doc.layers().iter().rev().map(|l| l.id).collect();
    for id in ids {
        let l = doc.layer(id).unwrap().clone_header();
        let selected = doc.active_layer() == Some(id);
        egui::Frame::group(ui.style()).fill(if selected { ui.visuals().selection.bg_fill.gamma_multiply(0.35) } else { Color32::TRANSPARENT }).show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut vis = l.visible;
                if ui.checkbox(&mut vis, "").on_hover_text("Visible").changed() {
                    let _ = doc.set_layer_visible(id, vis);
                }
                if ui.selectable_label(l.locked, if l.locked { "🔒" } else { "🔓" }).on_hover_text("Lock").clicked() {
                    let _ = doc.set_layer_locked(id, !l.locked);
                }
                if ui.selectable_label(selected, RichText::new(&l.name).strong()).clicked() {
                    let _ = doc.set_active_layer(Some(id));
                }
                if l.has_mask {
                    ui.label(RichText::new("M").strong().color(app.theme.viewport.overlay.0)).on_hover_text("Has a mask");
                }
            });
            let mut pct = l.opacity * 100.0;
            if ui.add(egui::Slider::new(&mut pct, -100.0..=200.0).suffix("%").text("Strength")).changed() {
                let _ = doc.set_layer_opacity(id, pct / 100.0);
            }
        });
    }
    let base_sel = doc.active_layer().is_none();
    if ui.selectable_label(base_sel, RichText::new("Base mesh").italics()).clicked() {
        let _ = doc.set_active_layer(None);
    }

    ui.add_space(8.0);
    ui.separator();
    mask_editor(app, ui);
}

trait Header {
    fn clone_header(&self) -> LayerHeader;
}
struct LayerHeader {
    name: String,
    opacity: f32,
    visible: bool,
    locked: bool,
    has_mask: bool,
}
impl Header for sculpt_core::SculptLayer {
    fn clone_header(&self) -> LayerHeader {
        LayerHeader { name: self.name.clone(), opacity: self.opacity, visible: self.visible, locked: self.locked, has_mask: self.mask.is_some() }
    }
}

// ------------------------------------------------------------------ masks

fn source_label(s: &MaskSource) -> &'static str {
    match s {
        MaskSource::Fill { .. } => "Fill",
        MaskSource::Channel { .. } => "Painted / imported channel",
        MaskSource::Noise(_) => "Noise",
        MaskSource::Mesh { attribute: MeshAttribute::Curvature } => "Curvature",
        MaskSource::Mesh { attribute: MeshAttribute::Cavity } => "Cavity",
        MaskSource::Mesh { attribute: MeshAttribute::AmbientOcclusion } => "Ambient occlusion",
        MaskSource::Mesh { attribute: MeshAttribute::Thickness } => "Thickness",
        MaskSource::Direction { .. } => "Direction",
        MaskSource::Gradient { .. } => "Gradient",
    }
}

fn mask_editor(app: &mut SculptApp, ui: &mut Ui) {
    let Some(doc) = app.doc.as_ref() else { return };
    let Some(id) = doc.active_layer() else {
        ui.label(RichText::new("Select a layer to edit its mask.").color(app.theme.weak_text()));
        return;
    };
    let name = doc.layer(id).unwrap().name.clone();
    if app.mask_edit.as_ref().is_none_or(|(lid, _)| *lid != id) {
        let stack = doc.layer(id).unwrap().mask.clone().unwrap_or_else(|| MaskStack::new(1.0));
        app.mask_edit = Some((id, stack));
    }
    let has_mask = doc.layer(id).unwrap().mask.is_some();
    ui.horizontal(|ui| {
        ui.heading(format!("Mask — {name}"));
    });
    ui.horizontal(|ui| {
        if !has_mask && ui.button("Add mask").clicked() {
            app.mask_dirty = true;
        }
        if has_mask && ui.button("Remove mask").clicked() {
            if let Some(d) = app.doc.as_mut() {
                let _ = d.set_layer_mask(id, None);
            }
            app.mask_edit = None;
            return;
        }
        if ui.selectable_label(app.overlay == OverlayKind::LayerMask, "Show").clicked() {
            app.overlay = if app.overlay == OverlayKind::LayerMask { OverlayKind::None } else { OverlayKind::LayerMask };
        }
    });
    let mask_channel = app.tools.mask_channel.clone();
    let Some((_, stack)) = app.mask_edit.as_mut() else { return };
    let mut changed = false;
    changed |= ui.add(egui::Slider::new(&mut stack.base, 0.0..=1.0).text("Base value")).changed();

    let mut remove = None;
    let mut swap = None;
    let n = stack.layers.len();
    for i in (0..n).rev() {
        let layer = &mut stack.layers[i];
        let title = if layer.name.is_empty() { source_label(&layer.source).to_string() } else { layer.name.clone() };
        egui::CollapsingHeader::new(format!("{}  ·  {:?} {:.0}%", title, layer.blend, layer.opacity * 100.0)).id_salt(("mask", i)).default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut layer.enabled, "On").changed();
                changed |= ui.checkbox(&mut layer.invert, "Invert").changed();
                if ui.small_button("Up").on_hover_text("Move up").clicked() && i + 1 < n {
                    swap = Some((i, i + 1));
                }
                if ui.small_button("Down").on_hover_text("Move down").clicked() && i > 0 {
                    swap = Some((i, i - 1));
                }
                if ui.small_button("×").on_hover_text("Remove").clicked() {
                    remove = Some(i);
                }
            });
            egui::ComboBox::from_id_salt(("blend", i)).selected_text(format!("{:?}", layer.blend)).show_ui(ui, |ui| {
                for b in [BlendMode::Normal, BlendMode::Multiply, BlendMode::Add, BlendMode::Subtract, BlendMode::Screen, BlendMode::Overlay, BlendMode::Max, BlendMode::Min, BlendMode::Difference] {
                    changed |= ui.selectable_value(&mut layer.blend, b, format!("{b:?}")).changed();
                }
            });
            changed |= ui.add(egui::Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity")).changed();
            changed |= source_ui(ui, &mut layer.source, i);
            ui.horizontal(|ui| {
                ui.label("Levels");
                changed |= ui.add(egui::DragValue::new(&mut layer.levels.in_min).range(0.0..=1.0).speed(0.005).prefix("in ")).changed();
                changed |= ui.add(egui::DragValue::new(&mut layer.levels.in_max).range(0.0..=1.0).speed(0.005)).changed();
                changed |= ui.add(egui::DragValue::new(&mut layer.levels.gamma).range(0.1..=5.0).speed(0.01).prefix("γ ")).changed();
            });
            changed |= ui.add(egui::Slider::new(&mut layer.blur, 0..=20).text("Blur")).changed();
        });
    }
    if let Some((a, b)) = swap {
        stack.layers.swap(a, b);
        changed = true;
    }
    if let Some(i) = remove {
        stack.layers.remove(i);
        changed = true;
    }
    ui.menu_button("+ Add mask layer", |ui| {
        let add: Option<MaskLayer> = [
            ("Noise", MaskSource::Noise(NoiseParams::default())),
            ("Painted channel", MaskSource::Channel { name: mask_channel.clone() }),
            ("Curvature", MaskSource::Mesh { attribute: MeshAttribute::Curvature }),
            ("Cavity", MaskSource::Mesh { attribute: MeshAttribute::Cavity }),
            ("Ambient occlusion", MaskSource::Mesh { attribute: MeshAttribute::AmbientOcclusion }),
            ("Thickness", MaskSource::Mesh { attribute: MeshAttribute::Thickness }),
            ("Direction", MaskSource::Direction { axis: glam::Vec3::Y, sharpness: 1.0 }),
            ("Gradient", MaskSource::Gradient { axis: glam::Vec3::Y, from: -1.0, to: 1.0 }),
            ("Fill", MaskSource::Fill { value: 1.0 }),
        ]
        .into_iter()
        .find_map(|(label, src)| ui.button(label).clicked().then(|| MaskLayer::new(label, src)));
        if let Some(l) = add {
            let blend = if stack.layers.is_empty() && stack.base >= 1.0 { BlendMode::Multiply } else { BlendMode::Normal };
            stack.layers.push(l.blend(blend).levels(Levels::default()));
            changed = true;
            ui.close();
        }
    });
    if changed {
        app.mask_dirty = true;
    }
}

fn source_ui(ui: &mut Ui, src: &mut MaskSource, i: usize) -> bool {
    let mut changed = false;
    match src {
        MaskSource::Fill { value } => changed |= ui.add(egui::Slider::new(value, 0.0..=1.0).text("Value")).changed(),
        MaskSource::Channel { name } => {
            ui.horizontal(|ui| {
                ui.label("Channel");
                changed |= ui.text_edit_singleline(name).lost_focus();
            });
        }
        MaskSource::Noise(p) => {
            egui::ComboBox::from_id_salt(("noise", i)).selected_text(format!("{:?}", p.kind)).show_ui(ui, |ui| {
                for k in [NoiseKind::Fbm, NoiseKind::Perlin, NoiseKind::Ridged, NoiseKind::Turbulence, NoiseKind::Cellular] {
                    changed |= ui.selectable_value(&mut p.kind, k, format!("{k:?}")).changed();
                }
            });
            changed |= ui.add(egui::Slider::new(&mut p.scale, 0.1..=50.0).logarithmic(true).text("Scale")).changed();
            changed |= ui.add(egui::Slider::new(&mut p.octaves, 1..=8).text("Octaves")).changed();
            changed |= ui.add(egui::Slider::new(&mut p.gain, 0.1..=0.9).text("Roughness")).changed();
            changed |= ui.add(egui::DragValue::new(&mut p.seed).prefix("seed ")).changed();
        }
        MaskSource::Mesh { attribute } => {
            ui.label(RichText::new(format!("Reads bake '{}' (Mesh ▸ Bake to refresh)", attribute.channel_name())).weak());
        }
        MaskSource::Direction { axis, sharpness } => {
            changed |= axis_ui(ui, axis);
            changed |= ui.add(egui::Slider::new(sharpness, 0.1..=8.0).text("Sharpness")).changed();
        }
        MaskSource::Gradient { axis, from, to } => {
            changed |= axis_ui(ui, axis);
            changed |= ui.add(egui::DragValue::new(from).speed(0.01).prefix("from ")).changed();
            changed |= ui.add(egui::DragValue::new(to).speed(0.01).prefix("to ")).changed();
        }
    }
    changed
}

fn axis_ui(ui: &mut Ui, axis: &mut glam::Vec3) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Axis");
        for (label, v) in [("X", glam::Vec3::X), ("Y", glam::Vec3::Y), ("Z", glam::Vec3::Z), ("-Y", -glam::Vec3::Y)] {
            if ui.selectable_label(*axis == v, label).clicked() {
                *axis = v;
                changed = true;
            }
        }
    });
    changed
}

/// Push the edited mask stack to the document (deferred until mouse release
/// so slider drags don't re-evaluate the mask every frame).
pub fn apply_mask_edit(app: &mut SculptApp) {
    let Some((id, stack)) = app.mask_edit.clone() else { return };
    let big = app.doc.as_ref().is_some_and(|d| d.vertex_count() > 400_000);
    if big {
        app.start_doc_job("Evaluating mask", move |d| d.set_layer_mask(id, Some(stack)).map_err(|e| e.to_string()));
    } else if let Some(d) = app.doc.as_mut()
        && let Err(e) = d.set_layer_mask(id, Some(stack)) {
            app.status = e.to_string();
        }
}

// ------------------------------------------------------------- properties

fn properties_tab(app: &mut SculptApp, ui: &mut Ui) {
    let tool = app.tool;
    ui.heading(tool.label());
    ui.label(RichText::new(tool.hint()).color(app.theme.weak_text()));
    ui.add_space(6.0);
    let p = app.tools.params_mut(tool);
    ui.add(egui::Slider::new(&mut p.size_px, 2.0..=600.0).logarithmic(true).text("Size (px)"));
    ui.add(egui::Slider::new(&mut p.strength, 0.0..=1.0).text("Strength"));
    ui.add(egui::Slider::new(&mut p.hardness, 0.0..=0.95).text("Falloff hardness"));
    ui.checkbox(&mut p.front_faces_only, "Front faces only");
    ui.separator();
    match tool {
        Tool::ClayBuildup => {
            let c = &mut app.tools.clay;
            ui.add(egui::Slider::new(&mut c.height, 0.02..=0.6).text("Plane height"));
            ui.add(egui::Slider::new(&mut c.squareness, 2.0..=10.0).text("Squareness"));
            ui.checkbox(&mut c.accumulate, "Accumulate within a stroke");
        }
        Tool::TrimDynamic => {
            let t = &mut app.tools.trim;
            ui.add(egui::Slider::new(&mut t.depth, 0.0..=0.3).text("Depth"));
            ui.add(egui::Slider::new(&mut t.smooth_border, 0.0..=1.0).text("Smooth border"));
        }
        Tool::Smooth => {
            ui.radio_value(&mut app.tools.smooth.mode, SmoothMode::Laplacian, "Laplacian (strong)");
            ui.radio_value(&mut app.tools.smooth.mode, SmoothMode::Surface, "Surface relax (keeps form)");
        }
        Tool::Move => {
            ui.checkbox(&mut app.tools.move_topological, "Topological (ignore unconnected parts)");
        }
        Tool::MaskPaint => {
            ui.horizontal(|ui| {
                ui.label("Channel");
                ui.text_edit_singleline(&mut app.tools.mask_channel);
            });
        }
        Tool::Pose => {
            ui.radio_value(&mut app.tools.pose_mode, PoseMode::Rotate, "Rotate");
            ui.radio_value(&mut app.tools.pose_mode, PoseMode::Translate, "Translate");
            ui.add(egui::Slider::new(&mut app.tools.pose_softness, 0.0..=1.0).text("Joint softness"));
            ui.add(egui::Slider::new(&mut app.tools.pose_blur, 0..=10).text("Mask blur"));
            ui.label(RichText::new("Size sets how far the topological mask reaches.").weak());
        }
        Tool::Freeze => {}
    }
}

fn object_tab(app: &mut SculptApp, ui: &mut Ui) {
    let Some(doc) = &app.doc else { return };
    egui::Grid::new("obj").num_columns(2).show(ui, |ui| {
        ui.label("Faces");
        ui.label(fmt_count(doc.face_count()));
        ui.end_row();
        ui.label("Vertices");
        ui.label(fmt_count(doc.vertex_count()));
        ui.end_row();
        ui.label("Level");
        ui.label(doc.level().to_string());
        ui.end_row();
        ui.label("Spatial leaves");
        ui.label(doc.bvh().leaves.len().to_string());
        ui.end_row();
        ui.label("Layers");
        ui.label(doc.layers().len().to_string());
        ui.end_row();
        let mb: usize = doc.layers().iter().map(|l| l.delta_bytes()).sum();
        ui.label("Layer data");
        ui.label(format!("{:.1} MB", mb as f64 / 1e6));
        ui.end_row();
        ui.label("Project");
        ui.label(app.project_path.as_ref().map_or("unsaved".into(), |p| p.display().to_string()));
        ui.end_row();
    });
    ui.separator();
    ui.label(RichText::new("Input").strong());
    ui.label(format!("Pen: {} — {}", app.pen.source.label(), app.pen.status));
    ui.separator();
    ui.label(RichText::new("Channels").strong());
    for name in doc.channels().keys() {
        ui.label(name);
    }
}

// ------------------------------------------------------------ dialogs etc.

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
                let c = &mut t.ui;
                for (l, h) in [
                    ("Window", &mut c.window),
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

//! Generator layer UI: the Properties section, the edit-to-update timer, and a scripted parameter sweep used to record
//! demos (`--demo-sweep`).
//!
//! Every parameter change goes through the layer command path (`MetaEdit::Generator`), so undo, coalescing of slider
//! drags and the timer all behave the same for the panel and for the sweep.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use egui::{RichText, Ui};
use sculpt_core::{Document, LayerId};
use sculpt_core::generator::density::Density;
use sculpt_core::bake::MeshAttribute;
use sculpt_core::generator::layer::{GeneratorKind, GeneratorSpec};
use sculpt_core::generator::mapping::{AttributeRow, AttributeSource, AttributeTarget};
use sculpt_core::generator::{BoolOp, Primitive};

use crate::app::SculptApp;
use crate::layer_panel::command::{self, LayerCommand, MetaEdit};
use crate::panels::{prop, section};

/// How long the edit calls took, so the panel can show how fast the surface answers.
///
/// With progressive updates an edit returns almost at once and the surface catches up over the next frames, so the timer
/// also tracks how long that takes and the most any one frame spent on it.
#[derive(Default)]
pub struct EditTimer {
    recent: VecDeque<f32>,
    last_ms: f32,
    slowest_ms: f32,
    last_edit: Option<Instant>,
    /// When the current burst of edits began, until the surface has caught up.
    burst_start: Option<Instant>,
    settle_ms: Option<f32>,
    work_max_ms: f32,
    work_last_ms: f32,
}

impl EditTimer {
    pub fn record(&mut self, ms: f32) {
        if self.burst_start.is_none() {
            self.work_max_ms = 0.0;
            self.settle_ms = None;
        }
        let now = Instant::now();
        self.last_edit = Some(now);
        self.burst_start.get_or_insert(now);
        self.last_ms = ms;
        self.slowest_ms = self.slowest_ms.max(ms);
        self.recent.push_back(ms);
        if self.recent.len() > 60 {
            self.recent.pop_front();
        }
    }

    /// Time since the last generator edit (an hour when there has been none).
    pub fn since_edit(&self) -> std::time::Duration {
        self.last_edit.map_or(std::time::Duration::from_secs(3600), |t| t.elapsed())
    }

    /// True from an edit until the surface has caught up with it.
    pub fn waiting(&self) -> bool {
        self.burst_start.is_some()
    }

    /// Note one frame's catch-up work (`ms`) and how many leaves are still waiting. Records the settle time at zero.
    pub fn note_progress(&mut self, ms: f32, left: usize) {
        self.work_last_ms = ms;
        self.work_max_ms = self.work_max_ms.max(ms);
        if left == 0
            && let Some(start) = self.burst_start.take()
        {
            self.settle_ms = Some(start.elapsed().as_secs_f32() * 1e3);
        }
    }

    /// (time to settle, work the last frame did, the most any frame did) for the latest burst of edits.
    pub fn catch_up(&self) -> (Option<f32>, f32, f32) {
        (self.settle_ms, self.work_last_ms, self.work_max_ms)
    }

    /// (last, average of the recent edits, slowest ever), in milliseconds.
    pub fn summary(&self) -> Option<(f32, f32, f32)> {
        (!self.recent.is_empty()).then(|| (self.last_ms, self.recent.iter().sum::<f32>() / self.recent.len() as f32, self.slowest_ms))
    }
}

const AXES: [(&str, [f32; 3]); 6] = [("Down (ground)", [0.0, -1.0, 0.0]), ("Up (top)", [0.0, 1.0, 0.0]), ("+X side", [1.0, 0.0, 0.0]), ("-X side", [-1.0, 0.0, 0.0]), ("+Z side", [0.0, 0.0, 1.0]), ("-Z side", [0.0, 0.0, -1.0])];

/// A drop-down for a direction, from the six axes. Returns true if the choice changed.
fn axis_combo(ui: &mut Ui, id: egui::Id, axis: &mut [f32; 3]) -> bool {
    let current = AXES.iter().find(|(_, a)| a == axis).map_or("Custom", |(n, _)| *n);
    let mut changed = false;
    egui::ComboBox::from_id_salt(id).selected_text(current).show_ui(ui, |ui| {
        for (name, a) in AXES {
            if ui.selectable_label(current == name, name).clicked() {
                *axis = a;
                changed = true;
            }
        }
    });
    changed
}

/// Rows offered by the "Add row" menu, each a sensible starting point to adjust.
fn row_presets(vertical_range: (f32, f32)) -> Vec<(&'static str, AttributeRow)> {
    let mut steep = AttributeRow::new(AttributeSource::Slope { axis: [0.0, 1.0, 0.0] }, AttributeTarget::Size, -0.6);
    steep.remap.invert = true;
    let mut low = AttributeRow::new(AttributeSource::Height { axis: [0.0, 1.0, 0.0], from: vertical_range.0, to: vertical_range.1 }, AttributeTarget::Size, 1.0);
    low.remap.invert = true;
    let mut convex = AttributeRow::new(AttributeSource::Mesh { attribute: MeshAttribute::Curvature }, AttributeTarget::Lift, 0.4);
    convex.remap.in_min = 0.45;
    convex.remap.in_max = 0.75;
    vec![
        ("Cavity -> bigger in the hollows", AttributeRow::new(AttributeSource::Mesh { attribute: MeshAttribute::Cavity }, AttributeTarget::Size, 1.0)),
        ("Steep slopes -> smaller", steep),
        ("Low down -> bigger", low),
        ("Noise -> varied size", AttributeRow::new(AttributeSource::Noise { scale: 3.0, seed: 1 }, AttributeTarget::Size, 0.8)),
        ("Convex edges -> stand proud", convex),
        ("Random -> lean", AttributeRow::new(AttributeSource::Random { seed: 1 }, AttributeTarget::Tilt, 25.0)),
        ("Order -> growing size", AttributeRow::new(AttributeSource::Index, AttributeTarget::Size, 1.0)),
    ]
}

fn source_options(vertical_range: (f32, f32)) -> Vec<AttributeSource> {
    vec![
        AttributeSource::Slope { axis: [0.0, 1.0, 0.0] },
        AttributeSource::Height { axis: [0.0, 1.0, 0.0], from: vertical_range.0, to: vertical_range.1 },
        AttributeSource::Noise { scale: 3.0, seed: 1 },
        AttributeSource::Mesh { attribute: MeshAttribute::Cavity },
        AttributeSource::Mesh { attribute: MeshAttribute::Curvature },
        AttributeSource::Mesh { attribute: MeshAttribute::AmbientOcclusion },
        AttributeSource::Mesh { attribute: MeshAttribute::Thickness },
        AttributeSource::Channel { name: "paint.mask".into() },
        AttributeSource::Random { seed: 1 },
        AttributeSource::Index,
    ]
}

fn op_label(op: BoolOp) -> &'static str {
    match op {
        BoolOp::Add => "Add",
        BoolOp::Subtract => "Subtract",
        BoolOp::Intersect => "Intersect",
    }
}

fn primitive_label(p: Primitive) -> &'static str {
    match p {
        Primitive::Box => "Box",
        Primitive::Sphere => "Sphere",
        Primitive::Cylinder => "Cylinder",
        Primitive::Cone => "Cone",
        Primitive::Wedge => "Wedge",
        Primitive::ChamferCutter => "Chamfer cutter",
        Primitive::Torus => "Torus",
    }
}

/// True while the user is actually working the widget. egui widgets can report a change on their own (a slider rounding the
/// value it is shown), and that must not become an edit: it would fight whatever else set the value.
fn touched(r: &egui::Response) -> bool {
    r.dragged() || r.drag_started() || r.drag_stopped() || r.has_focus() || r.lost_focus() || r.clicked() || r.is_pointer_button_down_on()
}

/// Record a widget edit. A drag in progress coalesces into one undo step.
fn note(r: &egui::Response, changed: &mut bool, coalesce: &mut bool) {
    if r.changed() && touched(r) {
        *changed = true;
        *coalesce = r.dragged() && !r.drag_started();
    }
}

/// The Properties section for a generator layer.
pub(crate) fn generator_props(app: &mut SculptApp, ui: &mut Ui, id: LayerId) {
    let Some(spec) = app.doc.as_ref().and_then(|d| d.layer(id)).and_then(|l| l.generator.clone()) else { return };
    if spec.kind == GeneratorKind::Noise {
        noise_props(app, ui, id, spec);
        return;
    }
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let timing = app.gen_timer.summary();
    let (settle, work_last, work_max) = app.gen_timer.catch_up();
    let pending = app.doc.as_ref().map_or(0, |d| d.pending_leaves());
    let vertical_range = app.doc.as_ref().map_or((-1.0, 1.0), |d| (d.bounds().min.y, d.bounds().max.y));
    let frame_ms = app.stats.frame_ms;
    let mut next = spec.clone();
    let (mut changed, mut coalesce) = (false, false);
    section(ui, fs, "generator", "GENERATOR", |ui| {
        prop(ui, "Operation", |ui| {
            egui::ComboBox::from_id_salt(("gen_op", id))
                .selected_text(op_label(next.op))
                .show_ui(ui, |ui| {
                    for op in [BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect] {
                        if ui.selectable_label(next.op == op, op_label(op)).clicked() {
                            next.op = op;
                            changed = true;
                        }
                    }
                })
                .response
        });
        prop(ui, "Primitive", |ui| {
            egui::ComboBox::from_id_salt(("gen_prim", id)).selected_text(primitive_label(next.primitive)).show_ui(ui, |ui| {
                for p in Primitive::ALL {
                    if ui.selectable_label(next.primitive == p, primitive_label(p)).clicked() {
                        next.primitive = p;
                        changed = true;
                    }
                }
            })
        });
        note(&prop(ui, "Count", |ui| ui.add(egui::Slider::new(&mut next.count, 1..=100_000).logarithmic(true))), &mut changed, &mut coalesce);
        note(&prop(ui, "Size", |ui| ui.add(egui::Slider::new(&mut next.size, 0.005..=0.5).logarithmic(true).max_decimals(3))), &mut changed, &mut coalesce);
        note(&prop(ui, "Lift", |ui| ui.add(egui::Slider::new(&mut next.lift, -1.0..=1.5).max_decimals(2))), &mut changed, &mut coalesce);
        note(&prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut next.seed).speed(0.2))), &mut changed, &mut coalesce);
        note(&prop(ui, "Blend", |ui| ui.add(egui::Slider::new(&mut next.blend, 0.0..=1.0).max_decimals(2))).on_hover_text("How softly this layer meets the surface when the stack is solidified, as a fraction of each instance's size. The live preview stays hard-edged."), &mut changed, &mut coalesce);
        note(&prop(ui, "Spacing", |ui| ui.add(egui::Slider::new(&mut next.min_spacing, 0.0..=0.5).max_decimals(3))).on_hover_text("Smallest distance between two instances. 0 places them independently; if the surface is full, fewer are placed."), &mut changed, &mut coalesce);
        note(&prop(ui, "Size variation", |ui| ui.add(egui::Slider::new(&mut next.size_var, 0.0..=1.0).max_decimals(2))), &mut changed, &mut coalesce);
        note(&prop(ui, "Stretch", |ui| ui.add(egui::Slider::new(&mut next.stretch_var, 0.0..=1.0).max_decimals(2))).on_hover_text("Varies each axis separately, so instances come out stretched or squashed."), &mut changed, &mut coalesce);
        note(&prop(ui, "Tilt", |ui| ui.add(egui::Slider::new(&mut next.tilt, 0.0..=90.0).suffix(" deg").max_decimals(0))).on_hover_text("Largest random lean away from the surface normal."), &mut changed, &mut coalesce);
        note(&prop(ui, "Lift jitter", |ui| ui.add(egui::Slider::new(&mut next.lift_var, 0.0..=1.0).max_decimals(2))), &mut changed, &mut coalesce);
        ui.add_space(4.0);
        prop(ui, "Density", |ui| {
            egui::ComboBox::from_id_salt(("gen_density", id))
                .selected_text(next.density.label())
                .show_ui(ui, |ui| {
                    let options = [("Even", Density::Even), ("Clusters", Density::default_clusters()), ("Facing", Density::default_facing()), ("Painted", Density::default_painted())];
                    for (name, option) in options {
                        if ui.selectable_label(next.density.label() == name, name).clicked() && next.density.label() != name {
                            next.density = option;
                            changed = true;
                        }
                    }
                })
                .response
                .on_hover_text("Where the instances gather. The count stays fixed; the density redistributes it.")
        });
        match &mut next.density {
            Density::Even => {}
            Density::Clusters { scale, coverage, softness, seed } => {
                note(&prop(ui, "Patch size", |ui| ui.add(egui::Slider::new(scale, 0.5..=12.0).logarithmic(true).max_decimals(1))).on_hover_text("Larger is bigger patches. Smaller is finer, more frequent clumps."), &mut changed, &mut coalesce);
                note(&prop(ui, "Coverage", |ui| ui.add(egui::Slider::new(coverage, 0.0..=1.0).max_decimals(2))).on_hover_text("How much of the surface is dense."), &mut changed, &mut coalesce);
                note(&prop(ui, "Softness", |ui| ui.add(egui::Slider::new(softness, 0.02..=1.0).max_decimals(2))), &mut changed, &mut coalesce);
                note(&prop(ui, "Pattern seed", |ui| ui.add(egui::DragValue::new(seed).speed(0.2))), &mut changed, &mut coalesce);
            }
            Density::Facing { axis, sharpness } => {
                changed |= prop(ui, "Faces", |ui| axis_combo(ui, egui::Id::new(("gen_facing", id)), axis));
                note(&prop(ui, "Tightness", |ui| ui.add(egui::Slider::new(sharpness, 0.3..=6.0).logarithmic(true).max_decimals(1))).on_hover_text("Higher means the surface has to face that way more squarely."), &mut changed, &mut coalesce);
            }
            Density::Painted { channel } => {
                let before = channel.clone();
                let r = prop(ui, "Channel", |ui| ui.text_edit_singleline(channel));
                if r.lost_focus() && *channel != before {
                    changed = true;
                }
                ui.label(RichText::new("Paint it with the Mask tool: white pulls instances in, black keeps them out.").small().color(weak));
            }
        }
        // Attribute rows: the surface decides how each instance looks.
        ui.add_space(6.0);
        ui.label(RichText::new("ATTRIBUTES").size(fs * 0.8).color(weak).extra_letter_spacing(0.6));
        ui.label(RichText::new("Let the surface set size, height, lift and lean.").small().color(weak));
        let mut remove = None;
        for (n, row) in next.mapping.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut row.enabled, "").changed() {
                        changed = true;
                    }
                    egui::ComboBox::from_id_salt(("gen_row_source", id, n)).width(120.0).selected_text(row.source.label()).show_ui(ui, |ui| {
                        for option in source_options(vertical_range) {
                            if ui.selectable_label(row.source.label() == option.label(), option.label()).clicked() && row.source.label() != option.label() {
                                row.source = option;
                                changed = true;
                            }
                        }
                    });
                    ui.label("->");
                    egui::ComboBox::from_id_salt(("gen_row_target", id, n)).width(70.0).selected_text(row.target.label()).show_ui(ui, |ui| {
                        for t in AttributeTarget::ALL {
                            if ui.selectable_label(row.target == t, t.label()).clicked() && row.target != t {
                                row.target = t;
                                changed = true;
                            }
                        }
                    });
                    if ui.small_button("x").on_hover_text("Remove this row").clicked() {
                        remove = Some(n);
                    }
                });
                match &mut row.source {
                    AttributeSource::Slope { axis } => {
                        changed |= prop(ui, "Facing", |ui| axis_combo(ui, egui::Id::new(("gen_row_axis", id, n)), axis));
                    }
                    AttributeSource::Height { axis, from, to } => {
                        changed |= prop(ui, "Along", |ui| axis_combo(ui, egui::Id::new(("gen_row_axis", id, n)), axis));
                        note(&prop(ui, "From", |ui| ui.add(egui::DragValue::new(from).speed(0.01).max_decimals(3))), &mut changed, &mut coalesce);
                        note(&prop(ui, "To", |ui| ui.add(egui::DragValue::new(to).speed(0.01).max_decimals(3))), &mut changed, &mut coalesce);
                    }
                    AttributeSource::Noise { scale, seed } => {
                        note(&prop(ui, "Scale", |ui| ui.add(egui::Slider::new(scale, 0.5..=12.0).logarithmic(true).max_decimals(1))), &mut changed, &mut coalesce);
                        note(&prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(seed).speed(0.2))), &mut changed, &mut coalesce);
                    }
                    AttributeSource::Random { seed } => note(&prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(seed).speed(0.2))), &mut changed, &mut coalesce),
                    AttributeSource::Channel { name } => {
                        let before = name.clone();
                        let r = prop(ui, "Channel", |ui| ui.text_edit_singleline(name));
                        if r.lost_focus() && *name != before {
                            changed = true;
                        }
                    }
                    AttributeSource::Mesh { .. } | AttributeSource::Index => {}
                }
                let (lo, hi) = match row.target {
                    AttributeTarget::Tilt => (-90.0, 90.0),
                    _ => (-3.0, 3.0),
                };
                note(&prop(ui, "Strength", |ui| ui.add(egui::Slider::new(&mut row.strength, lo..=hi).max_decimals(2))).on_hover_text("Size and height are multiplied by 1 + strength x value. Lift adds it, in sizes. Lean adds it, in degrees."), &mut changed, &mut coalesce);
                note(&prop(ui, "From value", |ui| ui.add(egui::Slider::new(&mut row.remap.in_min, 0.0..=1.0).max_decimals(2))).on_hover_text("Source values at or below this count as 0."), &mut changed, &mut coalesce);
                note(&prop(ui, "To value", |ui| ui.add(egui::Slider::new(&mut row.remap.in_max, 0.0..=1.0).max_decimals(2))).on_hover_text("Source values at or above this count as 1."), &mut changed, &mut coalesce);
                prop(ui, "", |ui| {
                    if ui.checkbox(&mut row.remap.invert, "Invert").changed() {
                        changed = true;
                    }
                });
            });
        }
        if let Some(n) = remove {
            next.mapping.remove(n);
            changed = true;
        }
        ui.menu_button("+ Add row", |ui| {
            for (name, preset) in row_presets(vertical_range) {
                if ui.button(name).clicked() {
                    next.mapping.push(preset);
                    changed = true;
                    ui.close();
                }
            }
        });
        prop(ui, "", |ui| {
            if ui.checkbox(&mut next.random_spin, "Random spin").changed() {
                changed = true;
            }
        });
        ui.add_space(4.0);
        let line = match timing {
            Some((last, avg, slow)) => format!("Edit to surface  {last:5.1} ms   avg {avg:5.1}   slowest {slow:5.1}"),
            None => "Edit to surface  (no edits yet)".to_string(),
        };
        ui.label(RichText::new(line).monospace().size(fs * 0.85));
        let settle_text = match (pending, settle) {
            (0, Some(ms)) => format!("settled in {ms:5.0} ms"),
            (0, None) => "settled".to_string(),
            (n, _) => format!("catching up, {n} leaves left"),
        };
        ui.label(RichText::new(format!("Surface       {settle_text}")).monospace().size(fs * 0.85));
        ui.label(RichText::new(format!("Catch-up work {work_last:5.1} ms/frame   most {work_max:5.1}")).monospace().size(fs * 0.85).color(weak));
        ui.label(RichText::new(format!("Whole frame   {frame_ms:5.1} ms")).monospace().size(fs * 0.85).color(weak));
    });
    if solidify_section(app, ui) {
        return;
    }
    if changed && next != spec {
        command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Generator(next), coalesce });
    }
}

/// The Properties section for a noise layer.
fn noise_props(app: &mut SculptApp, ui: &mut Ui, id: LayerId, spec: GeneratorSpec) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let longest = app.doc.as_ref().map_or(2.0, |d| d.bounds().extent().max_element());
    let mut next = spec.clone();
    let (mut changed, mut coalesce) = (false, false);
    section(ui, fs, "noise", "NOISE", |ui| {
        ui.label(RichText::new("Roughens the surface along its normals. Layers above it work on the roughened surface; Solidify keeps it.").size(fs * 0.85).color(weak));
        note(&prop(ui, "Amount", |ui| ui.add(egui::Slider::new(&mut next.noise.amplitude, 0.0..=0.1 * longest).max_decimals(3))).on_hover_text("Peak-to-peak displacement, in world units."), &mut changed, &mut coalesce);
        note(&prop(ui, "Scale", |ui| ui.add(egui::Slider::new(&mut next.noise.scale, 0.5..=80.0).logarithmic(true).max_decimals(1))).on_hover_text("Features per world unit: higher is finer."), &mut changed, &mut coalesce);
        note(&prop(ui, "Detail", |ui| ui.add(egui::Slider::new(&mut next.noise.octaves, 1..=8))), &mut changed, &mut coalesce);
        note(&prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut next.noise.seed).speed(0.2))), &mut changed, &mut coalesce);
    });
    if solidify_section(app, ui) {
        return;
    }
    if changed && next != spec {
        command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Generator(next), coalesce });
    }
}

/// The Solidify section shared by every generator layer. Returns true if the stack was just solidified, which leaves
/// the layer being shown gone.
fn solidify_section(app: &mut SculptApp, ui: &mut Ui) -> bool {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let mut solidify_now = false;
    section(ui, fs, "solidify", "SOLIDIFY", |ui| {
        ui.label(RichText::new("Turn the whole stack into one clean, closed mesh. Cuts blend softly and nothing folds. The layers are consumed; Undo brings them back.").size(fs * 0.85).color(weak));
        prop(ui, "Detail", |ui| ui.add(egui::Slider::new(&mut app.solid_ui.divisions, 40..=400).logarithmic(true)).on_hover_text("Samples along the longest side. Higher keeps finer detail and takes longer."));
        if ui.button("Solidify stack").clicked() {
            solidify_now = true;
        }
    });
    if solidify_now {
        solidify(app);
    }
    solidify_now
}

/// A new generator layer of `kind` above the selection, with sizes that suit the model.
pub(crate) fn new_generator_spec(kind: GeneratorKind, longest: f32) -> GeneratorSpec {
    match kind {
        GeneratorKind::Scatter => GeneratorSpec { count: 40, size: 0.08 * longest, seed: 1, ..GeneratorSpec::default() },
        GeneratorKind::Noise => GeneratorSpec { kind, noise: sculpt_core::generator::layer::NoiseSpec { amplitude: 0.015 * longest, scale: 22.0 / longest, octaves: 4, seed: 1 }, ..GeneratorSpec::default() },
    }
}

/// Settings of the Solidify section.
pub struct SolidUi {
    pub divisions: u32,
}

impl Default for SolidUi {
    fn default() -> Self {
        SolidUi { divisions: 150 }
    }
}

/// Replace the document with its solidified stack, keeping the old one for undo.
pub(crate) fn solidify(app: &mut SculptApp) {
    let Some(doc) = app.doc.as_ref() else { return };
    let longest = doc.bounds().extent().max_element();
    let settings = sculpt_core::document::SolidSettings { voxel: longest / app.solid_ui.divisions as f32 };
    let started = Instant::now();
    match doc.solidify(&settings).and_then(|(mesh, stats)| Document::from_mesh(mesh).map(|d| (d, stats))) {
        Ok((new, stats)) => {
            app.status = format!("Solidified {} layers into {} quads in {:.0} ms", stats.layers, stats.quads, started.elapsed().as_secs_f64() * 1e3);
            app.solidify_backup = app.doc.replace(new);
            app.layers.selection.clear();
            app.maybe_build_lod();
        }
        Err(e) => app.status = format!("Solidify failed: {e}"),
    }
}

/// Put back the document from before the last Solidify.
pub(crate) fn undo_solidify(app: &mut SculptApp) {
    if let Some(old) = app.solidify_backup.take() {
        app.doc = Some(old);
        app.layers.selection.clear();
        app.status = "Undo solidify".into();
        app.maybe_build_lod();
    }
}

// ------------------------------------------------------------------- sweep

/// Seconds the scripted sweep runs before the app closes itself.
const SWEEP_SECONDS: f32 = 38.0;
/// Quiet seconds before the first change, so a screen recorder can attach.
const SWEEP_LEAD: f32 = 4.0;

/// A timeline that moves one parameter at a time through its range and back, so a screen recording shows each control
/// driving the surface.
pub struct Sweep {
    id: LayerId,
    base: GeneratorSpec,
    start: Instant,
    shots: Option<Shots>,
}

/// Frames the app saves itself at chosen sweep times (`--demo-sweep-shots`), straight from its own framebuffer, so
/// they do not depend on window focus or what else is on screen.
struct Shots {
    dir: PathBuf,
    times: Vec<f32>,
    next: usize,
    pending: Option<PathBuf>,
}

impl Sweep {
    pub fn new(id: LayerId, base: GeneratorSpec) -> Sweep {
        Sweep { id, base, start: Instant::now(), shots: None }
    }

    pub fn with_shots(mut self, dir: PathBuf, mut times: Vec<f32>) -> Sweep {
        times.sort_by(f32::total_cmp);
        self.shots = Some(Shots { dir, times, next: 0, pending: None });
        self
    }
}

fn spec_at(t: f32, base: &GeneratorSpec) -> GeneratorSpec {
    let mut s = base.clone();
    // 0 -> 1 -> 0 over the phase, 0 outside it.
    let wave = |from: f32, to: f32| {
        let x = ((t - from) / (to - from)).clamp(0.0, 1.0);
        0.5 - 0.5 * (x * std::f32::consts::TAU).cos()
    };
    let log_lerp = |a: f32, b: f32, w: f32| (a.ln() + (b.ln() - a.ln()) * w).exp();
    let within = |from: f32, to: f32| t >= from && t < to;
    if within(3.0, 9.0) {
        s.count = log_lerp(base.count as f32, 3000.0, wave(3.0, 9.0)).round() as u32;
    } else if within(9.0, 15.0) {
        s.size = log_lerp(base.size, 0.2, wave(9.0, 15.0));
    } else if within(15.0, 19.0) {
        s.lift = base.lift - 0.9 * wave(15.0, 19.0);
    } else if within(19.0, 23.0) {
        s.seed = base.seed + ((t - 19.0) / 0.12) as u64;
    } else if within(23.0, 27.0) {
        s.primitive = Primitive::ALL[((t - 23.0) / 0.55) as usize % Primitive::ALL.len()];
    } else if within(27.0, 31.0) {
        s.op = [BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect][((t - 27.0) / 1.3) as usize % 3];
    } else if within(31.0, 37.0) {
        s.count = log_lerp(base.count as f32, 20_000.0, wave(31.0, 37.0)).round() as u32;
    }
    s
}

/// Advance the sweep by one frame. Edits go through the same command as the sliders.
pub(crate) fn tick_sweep(app: &mut SculptApp, ctx: &egui::Context) {
    let Some(mut sweep) = app.sweep.take() else { return };
    let t = sweep.start.elapsed().as_secs_f32() - SWEEP_LEAD;
    let finished = match &sweep.shots {
        Some(s) => s.next >= s.times.len() && s.pending.is_none() && t > s.times.last().copied().unwrap_or(0.0) + 0.5,
        None => t > SWEEP_SECONDS,
    };
    if finished || t > SWEEP_SECONDS + 30.0 {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        return;
    }
    let (id, spec) = (sweep.id, spec_at(t, &sweep.base));
    let current = app.doc.as_ref().and_then(|d| d.layer(id)).and_then(|l| l.generator.clone());
    if current.as_ref() != Some(&spec) {
        command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Generator(spec), coalesce: true });
    }
    if let Some(shots) = sweep.shots.as_mut()
        && shots.pending.is_none()
        && shots.next < shots.times.len()
        && t >= shots.times[shots.next]
    {
        shots.pending = Some(shots.dir.join(format!("shot_{:02}_t{:04.1}.png", shots.next, shots.times[shots.next])));
        shots.next += 1;
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
    }
    app.sweep = Some(sweep);
    ctx.request_repaint();
}

/// Save the frame the app was asked for by [`tick_sweep`], once the renderer hands it back.
pub(crate) fn collect_screenshot(app: &mut SculptApp, ctx: &egui::Context) {
    let Some(shots) = app.sweep.as_mut().and_then(|s| s.shots.as_mut()) else { return };
    let Some(path) = shots.pending.clone() else { return };
    let image = ctx.input(|i| i.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
    if let Some(image) = image {
        write_png(&path, &image);
        shots.pending = None;
    }
}

fn write_png(path: &std::path::Path, image: &egui::ColorImage) {
    let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
    let result = std::fs::File::create(path).map_err(|e| e.to_string()).and_then(|file| {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())
    });
    match result {
        Ok(()) => println!("saved {}", path.display()),
        Err(e) => eprintln!("sweep shot: {e}"),
    }
}

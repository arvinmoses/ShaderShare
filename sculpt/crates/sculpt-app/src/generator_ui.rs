//! Generator layer UI: the Properties section, the edit-to-update timer, and a scripted parameter sweep used to record
//! demos (`--demo-sweep`).
//!
//! Every parameter change goes through the layer command path (`MetaEdit::Generator`), so undo, coalescing of slider
//! drags and the timer all behave the same for the panel and for the sweep.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use egui::{RichText, Ui};
use sculpt_core::LayerId;
use sculpt_core::generator::layer::GeneratorSpec;
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
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let timing = app.gen_timer.summary();
    let (settle, work_last, work_max) = app.gen_timer.catch_up();
    let pending = app.doc.as_ref().map_or(0, |d| d.pending_leaves());
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
    if changed && next != spec {
        command::execute(app, LayerCommand::Edit { id, edit: MetaEdit::Generator(next), coalesce });
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

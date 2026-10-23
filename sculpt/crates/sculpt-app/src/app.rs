//! Application state and the per-frame loop.
//!
//! Frame order is chosen for latency: commands → panels → *viewport input is
//! turned into dabs and applied* → dirty leaves uploaded → scene rendered →
//! presented, all within one frame. Dab work has a time budget; samples that
//! don't fit carry over to the next frame instead of stalling the UI. Long
//! operations (bake, subdivide, load) run on a worker thread.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Sense, Vec2};
use glam::{Vec2 as GVec2, Vec3};
use sculpt_core::bake::{BakeSettings, MeshAttribute};
use sculpt_core::brush::{Brush, BrushSettings, Dab, Falloff, MoveBrush, PaintBrush, StrokeSampler};
use sculpt_core::io::{obj, project};
use sculpt_core::pose::PoseTransform;
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, PaintTarget, SurfaceHit};

use crate::camera::Camera;
use crate::keymap::{Command, Keymap};
use crate::pen::Pen;
use crate::theme::{Theme, ThemeLibrary};
use crate::tools::{PoseMode, Tool, ToolBox, Tray};
use crate::viewport::{OverlayKind, UploadStats, Viewport};

/// Max time per frame spent applying dabs before deferring to the next frame.
const DAB_BUDGET: Duration = Duration::from_millis(12);

pub struct Options {
    /// Start on a synthetic quad sphere with this many quads per cube edge (6·n² quads).
    pub sphere_res: Option<u32>,
    pub project: Option<PathBuf>,
    pub level: u32,
    pub theme: Option<String>,
    pub test_frames: Option<u32>,
    pub screenshot: Option<PathBuf>,
    /// Create a masked detail layer on startup (for demos and screenshots).
    pub demo_layers: bool,
    /// Add a generator layer on startup (for demos and screenshots): see `parse_generator_demo`.
    pub demo_generator: Option<sculpt_core::generator::layer::GeneratorSpec>,
    /// Sweep the demo generator's parameters on a timeline, then close (for screen recordings).
    pub demo_sweep: bool,
    /// Save the app's own frame at these sweep times (seconds) into this folder, then close.
    pub sweep_shots: Option<(PathBuf, Vec<f32>)>,
    /// Orbit the camera for this many frames and report timings, waiting for the GPU each frame.
    pub bench_orbit: Option<u32>,
    /// Multiply the initial framing distance (below 1 zooms in), for close-up tests and screenshots.
    pub zoom: f32,
    /// Run the scripted volume playground session, saving every frame into this folder (`volume_demo.rs`).
    pub demo_volume: Option<PathBuf>,
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub pos: Pos2,
    pub pressure: f32,
}
pub type SampleIn = Sample;

enum Stroke {
    None,
    /// Button is down but the pen hasn't reached the surface yet: start the
    /// stroke at the first sample that hits (ZBrush/Mudbox behaviour).
    Waiting { modifiers: egui::Modifiers },
    Brush {
        brush: Box<dyn Brush>,
        settings: BrushSettings,
        sampler: StrokeSampler,
        normal: Vec3,
        /// Spaced dabs not yet applied (the per-frame budget is enforced per dab).
        dabs: VecDeque<(Vec3, Vec3, f32)>,
        last_hit: Option<Vec3>,
    },
    Move { brush: MoveBrush, start: Pos2, world_per_px: f32 },
    Pose { pivot: Vec3, grab: Vec3, start: Pos2, world_per_px: f32, previewed: bool },
    /// A stroke in the volume playground: dabs every `spacing` along the surface since `last`.
    Volume { carve: bool, size_px: f32, strength: f32, hardness: f32, last: Option<Vec3>, carry: f32 },
}

pub enum Dialog {
    Open(String),
    SaveAs(String),
    ImportObj(String),
    ExportObj(String),
    NewSphere(u32),
    /// The volume playground's file (`.svol`).
    OpenVolume(String),
    SaveVolume(String),
}

struct Job {
    label: String,
    started: Instant,
    handle: JoinHandle<Result<Document, String>>,
}

#[derive(Default, Clone, Copy)]
pub struct FrameStats {
    pub frame_ms: f32,
    pub interval_ms: f32,
    pub input_ms: f32,
    pub dabs: usize,
    pub upload: UploadStats,
    pub render_ms: f32,
    pub lod: crate::viewport::LodStats,
    /// UI-thread time spent handing LOD patches to / from the background re-simplifier.
    pub lod_refresh_ms: f32,
    pub pending: usize,
}

/// What the Properties panel edits, within the active layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Layer,
    Mask,
    /// Index into the active layer's mask stack.
    Effect(usize),
}

pub struct SculptApp {
    pub doc: Option<Document>,
    job: Option<Job>,
    start_zoom: f32,
    /// How faithfully dense meshes are drawn (Display menu).
    pub view_detail: ViewDetail,
    /// Hard, per-triangle shading (nothing stored or uploaded for normals) instead of smooth per-vertex normals.
    pub flat_shading: bool,
    /// `SCULPT_LOD_TAU` overrides the preset's tolerance (benchmarks).
    lod_tau_override: Option<f32>,
    /// Background re-simplification of patches outdated by edits: (topology id, result channel).
    lod_refresh: Option<(u64, std::sync::mpsc::Receiver<sculpt_core::lod::RefreshDone>)>,
    /// Triangles per background batch; grows while the worker outpaces frames, shrinks when it lags.
    lod_batch: usize,
    /// When the last dab landed; refresh waits for the pen to rest so it never competes with a stroke.
    last_edit: Instant,
    pub(crate) viewport: Option<Viewport>,
    pub camera: Camera,
    pub themes: ThemeLibrary,
    pub theme: Theme,
    pub theme_editor: Option<Theme>,
    pub keymap: Keymap,
    pub tool: Tool,
    pub tools: ToolBox,
    stroke: Stroke,
    pending: VecDeque<Sample>,
    pub overlay: OverlayKind,
    pub overlay_strength: f32,
    pose_weights: Option<Vec<f32>>,
    pose_weights_changed: bool,
    pub hud: bool,
    pub stats: FrameStats,
    last_frame: Option<Instant>,
    pub project_path: Option<PathBuf>,
    pub dialog: Option<Dialog>,
    pub status: String,
    pub selection: Selection,
    /// Layers whose mask effects are shown in the stack.
    pub expanded: std::collections::HashSet<sculpt_core::LayerId>,
    pub layers: crate::layer_panel::PanelState,
    pub show_mesh_info: bool,
    /// Active layer when the mask editing copy was last synced.
    pub last_active: Option<sculpt_core::LayerId>,
    /// Last status text seen by the status bar and when it appeared.
    pub status_seen: (String, Instant),
    pub tray: Tray,
    pub mask_dirty: bool,
    pub pen: Pen,
    test: Option<crate::test_driver::TestDriver>,
    cursor: Option<(Vec3, f32, f32)>,
    viewport_rect: Rect,
    last_theme_poll: Instant,
    pub show_keymap: bool,
    /// Editing copy of the active layer's mask stack.
    pub mask_edit: Option<(sculpt_core::LayerId, sculpt_core::mask::MaskStack)>,
    demo_layers: bool,
    demo_generator: Option<sculpt_core::generator::layer::GeneratorSpec>,
    demo_sweep: bool,
    sweep_shots: Option<(PathBuf, Vec<f32>)>,
    /// How long generator edits took, shown in the generator Properties section.
    pub gen_timer: crate::generator_ui::EditTimer,
    pub sweep: Option<crate::generator_ui::Sweep>,
    /// Settings of the Solidify section: resolution along the longest side, and how much grain to add.
    pub solid_ui: crate::generator_ui::SolidUi,
    /// The document before the last Solidify, kept so undo can bring it back.
    pub solidify_backup: Option<Document>,
    /// The live preview of the solidified stack (Display menu, and the Solidify section).
    pub solid_preview: crate::solid_preview::SolidPreview,
    /// The sculpt waiting aside while the procedural stack below it is edited.
    pub remesh_top: Option<Document>,
    /// The stack as it was when editing below began, to measure the sculpt against when it is carried as volume.
    pub remesh_old_stack: Option<sculpt_core::document::SolidInput>,
    /// The volume playground, when open: sculpted and drawn instead of the document.
    pub volume_mode: Option<crate::volume_mode::VolumeMode>,
    /// Where the volume playground was last saved or opened.
    pub volume_path: Option<String>,
    /// The scripted session being recorded, if any.
    volume_demo: Option<crate::volume_demo::VolumeDemo>,
}

pub fn config_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SCULPT_HOME") {
        return PathBuf::from(p);
    }
    #[cfg(target_os = "windows")]
    if let Ok(p) = std::env::var("APPDATA") {
        return PathBuf::from(p).join("Sculpt");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    #[cfg(target_os = "macos")]
    return PathBuf::from(home).join("Library/Application Support/Sculpt");
    #[allow(unreachable_code)]
    std::env::var("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(home).join(".config")).join("sculpt")
}

impl SculptApp {
    pub fn new(cc: &eframe::CreationContext, opts: Options) -> SculptApp {
        let cfg = config_dir();
        let themes = ThemeLibrary::load(&cfg.join("themes"));
        let settings: serde_json::Value = std::fs::read_to_string(cfg.join("settings.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let theme_name = opts.theme.clone().or_else(|| settings["theme"].as_str().map(String::from)).unwrap_or_else(|| "Mudbox Dark".into());
        let theme = themes.get(&theme_name);
        crate::theme::install_fonts(&cc.egui_ctx);
        theme.apply(&cc.egui_ctx);
        let keymap = Keymap::load(&cfg.join("keymap.json"));
        let tools = serde_json::from_value(settings["tools"].clone()).unwrap_or_default();

        if let Ok(mode) = std::env::var("SCULPT_SHADING") {
            println!("shading override: {mode}");
        }
        let mut app = SculptApp {
            doc: None,
            job: None,
            start_zoom: opts.zoom,
            lod_refresh: None,
            lod_batch: LOD_REFRESH_BATCH_MIN,
            last_edit: Instant::now(),
            view_detail: ViewDetail::Balanced,
            flat_shading: match std::env::var("SCULPT_SHADING").as_deref() {
                Ok("smooth") => false,
                Ok("flat") => true,
                _ => settings["flat_shading"].as_bool().unwrap_or(true),
            },
            lod_tau_override: std::env::var("SCULPT_LOD_TAU").ok().and_then(|v| v.parse().ok()),
            viewport: cc.wgpu_render_state.as_ref().map(Viewport::new),
            camera: Camera::default(),
            themes,
            theme,
            theme_editor: None,
            keymap,
            tool: Tool::ClayBuildup,
            tools,
            stroke: Stroke::None,
            pending: VecDeque::new(),
            overlay: OverlayKind::None,
            overlay_strength: 0.75,
            pose_weights: None,
            pose_weights_changed: false,
            hud: false,
            stats: FrameStats::default(),
            last_frame: None,
            project_path: None,
            dialog: None,
            status: String::new(),
            selection: Selection::Layer,
            expanded: Default::default(),
            layers: Default::default(),
            show_mesh_info: false,
            last_active: None,
            status_seen: (String::new(), Instant::now()),
            tray: Tray::Sculpt,
            mask_dirty: false,
            pen: Pen::new(cc),
            test: opts.bench_orbit.or(opts.test_frames).map(|f| {
                let mut d = crate::test_driver::TestDriver::new(f, opts.screenshot.clone());
                d.orbit = opts.bench_orbit.is_some();
                d
            }),
            cursor: None,
            viewport_rect: Rect::NOTHING,
            last_theme_poll: Instant::now(),
            show_keymap: false,
            mask_edit: None,
            demo_layers: opts.demo_layers,
            demo_generator: opts.demo_generator,
            demo_sweep: opts.demo_sweep,
            sweep_shots: opts.sweep_shots,
            gen_timer: Default::default(),
            solid_ui: Default::default(),
            solidify_backup: None,
            solid_preview: Default::default(),
            remesh_top: None,
            remesh_old_stack: None,
            volume_mode: None,
            volume_path: None,
            sweep: None,
            volume_demo: opts.demo_volume.map(crate::volume_demo::VolumeDemo::new),
        };
        app.layers.compact = settings["layers_compact"].as_bool().unwrap_or(false);
        app.view_detail = settings["view_detail"].as_str().and_then(ViewDetail::from_name).unwrap_or(ViewDetail::Balanced);
        for e in app.themes.errors.iter().chain(&app.keymap.errors) {
            eprintln!("config: {e}");
        }
        match (opts.project, opts.sphere_res) {
            (None, Some(n)) => app.start_job(&format!("Creating {n}x{n}-per-face sphere"), move || {
                Document::from_mesh(sculpt_core::primitives::quad_sphere_res(n, 1.0)).map_err(|e| e.to_string())
            }),
            (Some(p), _) => app.start_job("Opening project", move || project::load(&p).map_err(|e| e.to_string())),
            (None, None) => {
                let level = opts.level;
                app.start_job(&format!("Creating level {level} sphere"), move || new_sphere(level));
            }
        }
        app
    }

    pub fn save_settings(&self) {
        let cfg = config_dir();
        let _ = std::fs::create_dir_all(&cfg);
        // SCULPT_SHADING overrides the shading for one run (benchmarks, screenshots) and is never saved.
        let flat_shading = if std::env::var_os("SCULPT_SHADING").is_some() {
            std::fs::read_to_string(cfg.join("settings.json")).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()).and_then(|v| v["flat_shading"].as_bool()).unwrap_or(true)
        } else {
            self.flat_shading
        };
        let v = serde_json::json!({ "theme": self.theme.name, "tools": self.tools, "layers_compact": self.layers.compact, "view_detail": self.view_detail.name(), "flat_shading": flat_shading });
        let _ = std::fs::write(cfg.join("settings.json"), serde_json::to_string_pretty(&v).unwrap());
    }

    pub fn set_theme(&mut self, ctx: &egui::Context, theme: Theme) {
        theme.apply(ctx);
        self.theme = theme;
        self.save_settings();
    }

    pub fn busy(&self) -> Option<(&str, f32)> {
        self.job.as_ref().map(|j| (j.label.as_str(), j.started.elapsed().as_secs_f32()))
    }

    /// Run `f` on a worker thread; the document is unavailable until it finishes.
    pub fn start_job(&mut self, label: &str, f: impl FnOnce() -> Result<Document, String> + Send + 'static) {
        self.end_stroke();
        self.job = Some(Job { label: label.into(), started: Instant::now(), handle: std::thread::spawn(f) });
    }

    /// Run `f` on the current document on a worker thread.
    pub fn start_doc_job(&mut self, label: &str, f: impl FnOnce(&mut Document) -> Result<(), String> + Send + 'static) {
        let Some(mut doc) = self.doc.take() else { return };
        self.start_job(label, move || f(&mut doc).map(|_| doc));
    }

    fn poll_job(&mut self) {
        if self.job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            let job = self.job.take().unwrap();
            let secs = job.started.elapsed().as_secs_f32();
            match job.handle.join() {
                Ok(Ok(doc)) => {
                    let first = self.doc.is_none() && self.camera.distance == Camera::default().distance;
                    self.status = format!("{} — {:.2}s", job.label, secs);
                    self.doc = Some(doc);
                    if first {
                        self.camera.frame(&self.doc.as_ref().unwrap().bounds());
                        self.camera.distance *= self.start_zoom;
                    }
                    if std::mem::take(&mut self.demo_layers) {
                        self.setup_demo_layers();
                    }
                    if let (Some(spec), Some(doc)) = (self.demo_generator.take(), self.doc.as_mut())
                        && let Ok(id) = doc.insert_generator("Generator", sculpt_core::Placement::Top, spec.clone())
                    {
                        // Select the row so Properties shows it (it can never become the sculpt target).
                        self.layers.selection.select_only(id);
                        if self.demo_sweep {
                            let mut sweep = crate::generator_ui::Sweep::new(id, spec);
                            if let Some((dir, times)) = self.sweep_shots.take() {
                                sweep = sweep.with_shots(dir, times);
                            }
                            self.sweep = Some(sweep);
                        }
                    }
                    if let Some(m) = std::env::var("SCULPT_VOLUME").ok().and_then(|v| v.parse::<f32>().ok()).or(self.volume_demo.as_ref().map(|_| 1.0)) {
                        self.open_volume(m * 1e6);
                    }
                    if std::env::var("SCULPT_DEMO_NOISE").is_ok() {
                        crate::layer_panel::command::execute(self, crate::layer_panel::command::LayerCommand::NewGenerator(sculpt_core::generator::layer::GeneratorKind::Noise));
                    }
                    if std::env::var("SCULPT_DEMO_PREVIEW").is_ok() {
                        self.solid_preview.on = true;
                    }
                    if std::env::var("SCULPT_DEMO_SOLIDIFY").is_ok() {
                        crate::generator_ui::solidify(self);
                    }
                    self.maybe_build_lod();
                }
                Ok(Err(e)) => self.status = format!("{} failed: {e}", job.label),
                Err(_) => self.status = format!("{} crashed", job.label),
            }
            if self.doc.is_none() && self.job.is_none() {
                self.start_job("Creating sphere", || new_sphere(6));
            }
        }
    }

    /// Dense meshes get a level-of-detail tree so drawing costs pixels, not triangles.
    pub(crate) fn maybe_build_lod(&mut self) {
        let Some(doc) = self.doc.as_ref() else { return };
        if doc.lod().is_none() && std::env::var_os("SCULPT_NO_LOD").is_none() && doc.face_count() * 2 >= LOD_MIN_TRIANGLES {
            let mut params = sculpt_core::lod::LodParams::default();
            if let Some(w) = std::env::var("SCULPT_LOD_NORMAL_WEIGHT").ok().and_then(|v| v.parse().ok()) {
                params.normal_weight = w;
            }
            self.start_doc_job("Building level of detail", move |d| {
                d.build_lod(params);
                Ok(())
            });
        }
    }

    fn setup_demo_layers(&mut self) {
        use sculpt_core::mask::{BlendMode, Levels, MaskLayer, MaskSource, MaskStack};
        use sculpt_core::noise::NoiseParams;
        let Some(doc) = self.doc.as_mut() else { return };
        // A small hierarchy so the stack shows folders, nesting and a mask.
        use sculpt_core::Placement;
        let face = doc.insert_folder("Face", Placement::Top).ok();
        if let Some(f) = face {
            let _ = doc.insert_layer("Wrinkles", Placement::Into(f));
            let _ = doc.insert_layer("Pores", Placement::Into(f));
        }
        let _ = doc.insert_layer("Skin tone variation", Placement::Top);
        let id = doc.insert_layer("Detail", Placement::Top).expect("top-level layer");
        let stack = MaskStack::new(0.0)
            .with(MaskLayer::new("Breakup", MaskSource::Noise(NoiseParams { scale: 3.0, seed: 4, ..Default::default() })).levels(Levels::range(0.4, 0.6)))
            .with(MaskLayer::new("Top light", MaskSource::Direction { axis: Vec3::Y, sharpness: 2.0 }).blend(BlendMode::Screen).opacity(0.6));
        let _ = doc.set_layer_mask(id, Some(stack.clone()));
        let _ = doc.set_layer_opacity(id, 0.85);
        self.mask_edit = Some((id, stack));
        self.last_active = Some(id);
        self.expanded.insert(id);
        self.selection = Selection::Effect(0);
        self.overlay = OverlayKind::LayerMask;
        self.overlay_strength = 0.45;
    }

    // ------------------------------------------------------------- commands

    pub fn run(&mut self, ctx: &egui::Context, cmd: Command) {
        // The volume playground has its own undo history and file.
        if self.volume_mode.is_some() && matches!(cmd, Command::Undo | Command::Redo | Command::Save | Command::Open) {
            self.end_stroke();
            let mode = self.volume_mode.as_mut().unwrap();
            match cmd {
                Command::Undo => self.status = if mode.undo() { "Undo".into() } else { "Nothing to undo".into() },
                Command::Redo => self.status = if mode.redo() { "Redo".into() } else { "Nothing to redo".into() },
                Command::Save => self.dialog = Some(Dialog::SaveVolume(self.volume_path.clone().unwrap_or("untitled.svol".into()))),
                _ => self.dialog = Some(Dialog::OpenVolume(String::new())),
            }
            return;
        }
        let tool_of = Tool::ALL.iter().find(|t| t.command() == cmd).copied();
        if let Some(t) = tool_of {
            self.select_tool(t);
            return;
        }
        if crate::layer_panel::switcher::run(self, ctx, cmd) {
            return;
        }
        if let Some(lc) = crate::layer_panel::command::from_keymap(cmd, self) {
            crate::layer_panel::command::execute(self, lc);
            return;
        }
        if cmd == Command::Undo && crate::remesh_ui::undo_remesh(self) {
            return;
        }
        if cmd == Command::Undo && self.doc.as_ref().is_some_and(|d| !d.can_undo()) && self.solidify_backup.is_some() {
            crate::generator_ui::undo_solidify(self);
            return;
        }
        let Some(doc) = self.doc.as_mut() else { return };
        match cmd {
            Command::Undo => {
                self.status = if doc.undo() { "Undo".into() } else { "Nothing to undo".into() };
                crate::layer_panel::command::after_change(self);
            }
            Command::Redo => {
                self.status = if doc.redo() { "Redo".into() } else { "Nothing to redo".into() };
                crate::layer_panel::command::after_change(self);
            }
            Command::Save => match &self.project_path {
                Some(p) => self.save_to(p.clone()),
                None => self.dialog = Some(Dialog::SaveAs("untitled.sculpt".into())),
            },
            Command::Open => self.dialog = Some(Dialog::Open(String::new())),
            Command::BrushSizeUp | Command::BrushSizeDown => {
                let p = self.tools.params_mut(self.tool);
                p.size_px = (p.size_px * if cmd == Command::BrushSizeUp { 1.15 } else { 1.0 / 1.15 }).clamp(2.0, 1000.0);
            }
            Command::StrengthUp | Command::StrengthDown => {
                let p = self.tools.params_mut(self.tool);
                p.strength = (p.strength + if cmd == Command::StrengthUp { 0.05 } else { -0.05 }).clamp(0.0, 1.0);
            }
            Command::FrameMesh => self.camera.frame(&doc.bounds()),
            Command::Subdivide => {
                let faces = doc.face_count() * 4;
                self.start_doc_job(&format!("Subdividing to {faces} faces"), |d| d.subdivide().map_err(|e| e.to_string()));
            }
            Command::ToggleHud => self.hud = !self.hud,
            Command::CycleOverlay => {
                self.overlay = match self.overlay {
                    OverlayKind::None => OverlayKind::Freeze,
                    OverlayKind::Freeze => OverlayKind::LayerMask,
                    OverlayKind::LayerMask => OverlayKind::Channel(self.tools.mask_channel.clone()),
                    _ => OverlayKind::None,
                };
            }
            Command::InvertFreeze => {
                let v: Vec<f32> = doc.freeze().iter().map(|f| 1.0 - f).collect();
                let _ = doc.set_freeze(v);
            }
            Command::ClearFreeze => {
                let n = doc.vertex_count();
                let _ = doc.set_freeze(vec![0.0; n]);
            }
            _ => {}
        }
        let _ = ctx;
    }

    /// True while a brush, move or pose stroke is in progress.
    pub fn is_stroking(&self) -> bool {
        !matches!(self.stroke, Stroke::None)
    }

    pub fn select_tool(&mut self, t: Tool) {
        self.end_stroke();
        self.tool = t;
        self.tray = t.tray();
        self.overlay = match t {
            Tool::Freeze => OverlayKind::Freeze,
            Tool::MaskPaint => OverlayKind::Channel(self.tools.mask_channel.clone()),
            Tool::Pose if self.pose_weights.is_some() => OverlayKind::Custom,
            _ if matches!(self.overlay, OverlayKind::Freeze | OverlayKind::Custom) => OverlayKind::None,
            _ => self.overlay.clone(),
        };
    }

    pub fn save_to(&mut self, path: PathBuf) {
        let Some(doc) = &self.doc else { return };
        let t = Instant::now();
        match project::save(doc, &path) {
            Ok(()) => {
                self.status = format!("Saved {} ({:.2}s)", path.display(), t.elapsed().as_secs_f32());
                self.project_path = Some(path);
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    pub fn run_dialog_action(&mut self, d: Dialog) {
        match d {
            Dialog::Open(p) => {
                let path = PathBuf::from(p);
                self.project_path = Some(path.clone());
                self.start_job("Opening project", move || project::load(&path).map_err(|e| e.to_string()));
            }
            Dialog::SaveAs(p) => self.save_to(PathBuf::from(p)),
            Dialog::SaveVolume(p) => {
                let t = Instant::now();
                let Some(mode) = self.volume_mode.as_mut() else { return };
                mode.settle();
                let r = std::fs::File::create(&p).and_then(|f| {
                    let mut w = std::io::BufWriter::new(f);
                    mode.write_to(&mut w)?;
                    std::io::Write::flush(&mut w)
                });
                self.status = match r {
                    Ok(()) => format!("Saved {p} ({:.2}s)", t.elapsed().as_secs_f32()),
                    Err(e) => format!("Save failed: {e}"),
                };
                self.volume_path = Some(p);
            }
            Dialog::OpenVolume(p) => {
                let t = Instant::now();
                let r = std::fs::File::open(&p).and_then(|f| crate::volume_mode::VolumeMode::read_from(&mut std::io::BufReader::new(f)));
                match r {
                    Ok(mut mode) => {
                        self.end_stroke();
                        let (lo, hi) = mode.vol.bounds();
                        let generation = self.volume_mode.as_ref().map_or(0, |m| m.generation) + 1;
                        mode.generation = generation;
                        self.volume_mode = Some(mode);
                        if let Some(vp) = self.viewport.as_mut() {
                            vp.clear_volume();
                        }
                        self.camera.frame(&sculpt_core::geom::Aabb { min: lo, max: hi });
                        self.status = format!("Opened {p} ({:.2}s)", t.elapsed().as_secs_f32());
                        self.volume_path = Some(p);
                    }
                    Err(e) => self.status = format!("Open failed: {e}"),
                }
            }
            Dialog::ImportObj(p) => {
                self.project_path = None;
                self.start_job("Importing OBJ", move || {
                    let m = obj::read(std::path::Path::new(&p)).map_err(|e| e.to_string())?;
                    Document::from_mesh(m).map_err(|e| e.to_string())
                });
            }
            Dialog::ExportObj(p) => {
                if let Some(doc) = self.doc.as_mut() {
                    doc.finish_pending();
                    self.status = match obj::write(std::path::Path::new(&p), &doc.export_mesh(true)) {
                        Ok(()) => format!("Exported {p}"),
                        Err(e) => format!("Export failed: {e}"),
                    };
                }
            }
            Dialog::NewSphere(level) => {
                self.project_path = None;
                self.start_job(&format!("Creating level {level} sphere"), move || new_sphere(level));
            }
        }
    }

    pub fn bake(&mut self, attr: MeshAttribute) {
        self.start_doc_job(&format!("Baking {attr:?}"), move |d| {
            d.bake(attr, &BakeSettings::default());
            d.refresh_layer_masks().map_err(|e| e.to_string())
        });
        self.overlay = OverlayKind::Channel(attr.channel_name().into());
    }

    // ---------------------------------------------------------------- strokes

    fn end_stroke(&mut self) {
        self.pending.clear();
        let stroke = std::mem::replace(&mut self.stroke, Stroke::None);
        if let (Stroke::Volume { .. }, Some(mode)) = (&stroke, self.volume_mode.as_mut()) {
            mode.end_stroke();
        }
        if let Some(doc) = self.doc.as_mut() {
            match stroke {
                Stroke::Move { mut brush, .. } => {
                    brush.end();
                    doc.end_stroke();
                }
                Stroke::Brush { .. } => doc.end_stroke(),
                _ => {}
            }
        }
    }

    fn begin_stroke(&mut self, pos: Pos2, modifiers: egui::Modifiers) {
        if self.volume_mode.is_some() {
            let carve = match self.tool {
                Tool::ClayBuildup => modifiers.command,
                Tool::TrimDynamic => !modifiers.command,
                _ => {
                    self.status = "The volume playground has Clay Buildup (adds) and Trim Dynamic (carves); Ctrl swaps them.".into();
                    return;
                }
            };
            let p = self.tools.params(self.tool).clone();
            self.stroke = Stroke::Volume { carve, size_px: p.size_px, strength: p.strength, hardness: p.hardness, last: None, carry: 0.0 };
            if let Some(mode) = self.volume_mode.as_mut() {
                mode.begin_stroke();
            }
            self.pending.push_front(Sample { pos, pressure: self.pen.pressure });
            return;
        }
        // Strokes under generator layers would be displaced by them again: sculpt on a remesh of the stack instead.
        if self.doc.as_ref().is_some_and(crate::remesh_ui::needs_remesh) {
            if self.remesh_top.is_some() {
                self.status = "Sculpting waits while you edit the stack below: click Done in Properties to go back to it.".into();
                return;
            }
            if !crate::remesh_ui::remesh_to_sculpt(self) {
                return;
            }
        }
        // A stroke that would change nothing is refused up front, with the reason in the status bar.
        if let Some(why) = crate::layer_panel::target::resolve(self).and_then(|t| t.refusal) {
            self.status = why;
            return;
        }
        let Some(doc) = self.doc.as_mut() else { return };
        let size = self.viewport_rect.size();
        let local = pos - self.viewport_rect.min;
        let ray = self.camera.ray(GVec2::new(local.x, local.y), GVec2::new(size.x, size.y));
        let Some(hit) = doc.raycast(&ray) else {
            self.stroke = Stroke::Waiting { modifiers };
            return;
        };
        let wpp = self.camera.world_per_pixel(hit.point, size.y);
        // Shift: temporary smooth (ZBrush/Mudbox convention). Ctrl: invert.
        let tool = if modifiers.shift && matches!(self.tool, Tool::ClayBuildup | Tool::TrimDynamic | Tool::Move) { Tool::Smooth } else { self.tool };
        let p = self.tools.params(tool).clone();
        let settings = BrushSettings {
            radius: p.size_px * wpp,
            strength: p.strength,
            falloff: Falloff { hardness: p.hardness },
            spacing: 0.12,
            front_faces_only: p.front_faces_only,
            invert: modifiers.command,
        };
        let brush: Box<dyn Brush> = match tool {
            Tool::ClayBuildup => Box::new(self.tools.clay.clone()),
            Tool::TrimDynamic => Box::new(self.tools.trim.clone()),
            Tool::Smooth => Box::new(self.tools.smooth.clone()),
            Tool::Freeze => Box::new(PaintBrush { target: PaintTarget::Freeze, value: 1.0 }),
            Tool::MaskPaint => Box::new(PaintBrush { target: PaintTarget::Channel(self.tools.mask_channel.clone()), value: 1.0 }),
            Tool::Move => {
                doc.begin_stroke("Move");
                let mut mv = MoveBrush::new(self.tools.move_topological);
                mv.begin(doc, &hit, &settings);
                self.stroke = Stroke::Move { brush: mv, start: pos, world_per_px: wpp };
                return;
            }
            Tool::Pose => {
                self.begin_pose(&hit, pos, wpp, p.size_px);
                return;
            }
        };
        doc.begin_stroke(brush.name());
        self.stroke = Stroke::Brush {
            brush,
            sampler: StrokeSampler::new(settings.radius * settings.spacing),
            settings,
            normal: hit.normal,
            dabs: VecDeque::new(),
            last_hit: None,
        };
        self.pending.push_front(Sample { pos, pressure: self.pen.pressure });
    }

    fn begin_pose(&mut self, hit: &SurfaceHit, pos: Pos2, wpp: f32, size_px: f32) {
        let doc = self.doc.as_ref().unwrap();
        let reach = size_px * wpp;
        let w = doc.topological_weights(hit.vertex, reach, self.tools.pose_softness, self.tools.pose_blur);
        // Pivot = centroid of the soft band (the "joint").
        let (mut sum, mut n) = (Vec3::ZERO, 0.0);
        for (v, &x) in w.iter().enumerate() {
            if x > 0.05 && x < 0.6 {
                sum += doc.positions()[v];
                n += 1.0;
            }
        }
        let pivot = if n > 0.0 { sum / n } else { hit.point };
        self.pose_weights = Some(w);
        self.pose_weights_changed = true;
        self.overlay = OverlayKind::Custom;
        self.stroke = Stroke::Pose { pivot, grab: hit.point, start: pos, world_per_px: wpp, previewed: false };
    }

    fn pose_transform(&self, pivot: Vec3, grab: Vec3, drag: Vec2, wpp: f32) -> PoseTransform {
        let world = (self.camera.right() * drag.x - self.camera.up() * drag.y) * wpp;
        match self.tools.pose_mode {
            PoseMode::Translate => PoseTransform::Translate { offset: world },
            PoseMode::Rotate => {
                let arm = grab - pivot;
                let axis = arm.cross(world);
                let angle = world.length() / arm.length().max(1e-6);
                if axis.length_squared() < 1e-12 {
                    PoseTransform::Translate { offset: Vec3::ZERO }
                } else {
                    PoseTransform::Rotate { pivot, axis: axis.normalize(), angle }
                }
            }
        }
    }

    /// Apply queued samples within the time budget.
    fn process_samples(&mut self, last_pos: Option<Pos2>) -> usize {
        while let Stroke::Waiting { modifiers } = self.stroke {
            let Some(s) = self.pending.pop_front() else { return 0 };
            self.stroke = Stroke::None;
            self.pen.pressure = s.pressure;
            self.begin_stroke(s.pos, modifiers);
        }
        let start = Instant::now();
        let mut dabs = 0;
        let size = self.viewport_rect.size();
        let camera = self.camera.clone();
        match &mut self.stroke {
            Stroke::Volume { carve, size_px, strength, hardness, last, carry } => {
                let Some(mode) = self.volume_mode.as_mut() else { return 0 };
                mode.last_ms = 0.0;
                mode.last_dabs = 0;
                while let Some(s) = self.pending.pop_front() {
                    let local = s.pos - self.viewport_rect.min;
                    let ray = camera.ray(GVec2::new(local.x, local.y), GVec2::new(size.x, size.y));
                    let Some((hit, _)) = mode.raycast(ray.origin, ray.dir) else {
                        *last = None;
                        continue;
                    };
                    let px = camera.world_per_pixel(hit, size.y);
                    let radius = *size_px * px;
                    let spacing = (0.15 * radius).max(mode.voxel());
                    let from = match *last {
                        // Re-entering the surface, or jumping across a gap, starts the spacing over.
                        Some(p) if p.distance(hit) < radius * 4.0 => p,
                        _ => {
                            mode.dab(hit, radius, px, *strength * s.pressure, *hardness, *carve);
                            dabs += 1;
                            *last = Some(hit);
                            *carry = 0.0;
                            continue;
                        }
                    };
                    // Dabs every `spacing` along the way from the last one, each moved onto the surface the stroke began on.
                    let gap = from.distance(hit);
                    let mut along = spacing - *carry;
                    while along <= gap {
                        let q = from.lerp(hit, along / gap);
                        mode.dab(q, radius, px, *strength * s.pressure, *hardness, *carve);
                        dabs += 1;
                        along += spacing;
                    }
                    *carry = gap - (along - spacing);
                    *last = Some(hit);
                }
                if dabs > 0 {
                    self.status = format!("Volume: {dabs} dabs this frame at level {} (step {:.5}), {:.2} ms ({:.2} ms each), {} blocks", mode.level, mode.voxel(), mode.last_ms, mode.last_ms / dabs as f64, mode.blocks);
                }
            }
            Stroke::Brush { brush, settings, sampler, normal, dabs: queue, last_hit } => {
                let Some(doc) = self.doc.as_mut() else { return 0 };
                loop {
                    // Apply already-spaced dabs first, strictly within budget.
                    while let Some((point, dir, pressure)) = queue.pop_front() {
                        if start.elapsed() > DAB_BUDGET {
                            queue.push_front((point, dir, pressure));
                            return dabs;
                        }
                        let h = doc.project_to_surface(point, *normal, settings.radius);
                        let Some(h) = h else { continue };
                        let dab = Dab { center: h.point, normal: h.normal, direction: dir, pressure };
                        if let Err(e) = brush.dab(doc, &dab, settings) {
                            self.status = e.to_string();
                            self.pending.clear();
                            queue.clear();
                            return dabs;
                        }
                        dabs += 1;
                    }
                    let Some(s) = self.pending.pop_front() else { break };
                    let local = s.pos - self.viewport_rect.min;
                    let ray = camera.ray(GVec2::new(local.x, local.y), GVec2::new(size.x, size.y));
                    let Some(hit) = doc.raycast(&ray) else {
                        // Pen left the surface: lift, don't bridge the gap.
                        *last_hit = None;
                        continue;
                    };
                    // Re-entering the surface or jumping across a depth
                    // discontinuity restarts spacing instead of interpolating.
                    if last_hit.is_none_or(|p| p.distance(hit.point) > settings.radius * 4.0) {
                        *sampler = StrokeSampler::new(settings.radius * settings.spacing);
                    }
                    *last_hit = Some(hit.point);
                    *normal = hit.normal;
                    queue.extend(sampler.push(hit.point, s.pressure));
                }
            }
            Stroke::Move { brush, start: s0, world_per_px } => {
                self.pending.clear();
                if let (Some(p), Some(doc)) = (last_pos, self.doc.as_mut()) {
                    let d = p - *s0;
                    let offset = (camera.right() * d.x - camera.up() * d.y) * *world_per_px;
                    if brush.drag(doc, offset).is_ok() {
                        dabs += 1;
                    }
                }
            }
            Stroke::Pose { pivot, grab, start: s0, world_per_px, previewed } => {
                self.pending.clear();
                // Live preview on meshes small enough to re-pose every frame.
                let live = self.doc.as_ref().is_some_and(|d| d.vertex_count() < 1_500_000);
                if let (Some(p), true) = (last_pos, live) {
                    let (pivot, grab, wpp, d) = (*pivot, *grab, *world_per_px, p - *s0);
                    let was = std::mem::replace(previewed, true);
                    let xf = self.pose_transform(pivot, grab, d, wpp);
                    let doc = self.doc.as_mut().unwrap();
                    if was {
                        doc.undo();
                    }
                    let w = self.pose_weights.as_ref().unwrap();
                    let _ = doc.pose(w, xf);
                    dabs += 1;
                }
            }
            Stroke::None | Stroke::Waiting { .. } => self.pending.clear(),
        }
        dabs
    }

    fn finish_pose(&mut self, end: Pos2) {
        if let Stroke::Pose { pivot, grab, start, world_per_px, previewed } = std::mem::replace(&mut self.stroke, Stroke::None) {
            let xf = self.pose_transform(pivot, grab, end - start, world_per_px);
            let doc = self.doc.as_mut().unwrap();
            if previewed {
                doc.undo();
            }
            if (end - start).length() > 1.0 {
                let _ = doc.pose(self.pose_weights.as_ref().unwrap(), xf);
            }
        }
    }

    // --------------------------------------------------------------- viewport

    /// Open the volume playground on a fresh ball of about `triangles` triangles.
    pub fn open_volume(&mut self, triangles: f32) {
        self.end_stroke();
        let t = Instant::now();
        let mut mode = crate::volume_mode::VolumeMode::ball(triangles);
        mode.generation = self.volume_mode.as_ref().map_or(0, |m| m.generation) + 1;
        self.camera.frame(&sculpt_core::geom::Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) });
        self.status = format!("Volume playground: {:.1}M-triangle ball in {:.0} ms. Clay Buildup adds, Trim Dynamic carves, Ctrl swaps. Mesh menu to leave.", triangles / 1e6, t.elapsed().as_secs_f64() * 1e3);
        self.volume_mode = Some(mode);
        if self.tool != Tool::ClayBuildup && self.tool != Tool::TrimDynamic {
            self.select_tool(Tool::ClayBuildup);
        }
    }

    /// Back to the document.
    pub fn close_volume(&mut self) {
        self.end_stroke();
        self.volume_mode = None;
        if let Some(vp) = self.viewport.as_mut() {
            vp.clear_volume();
        }
        if let Some(doc) = self.doc.as_ref() {
            self.camera.frame(&doc.bounds());
        }
        self.status = "Left the volume playground".into();
    }

    /// Time since the last sculpt stroke dab.
    pub fn last_edit_age(&self) -> Duration {
        self.last_edit.elapsed()
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.viewport_rect = rect;
        let ctx = ui.ctx().clone();
        let input_start = Instant::now();

        let (mods, scroll, events) = ctx.input(|i| (i.modifiers, i.smooth_scroll_delta.y, i.events.clone()));
        self.pen.update(&events);
        let delta = resp.drag_delta();
        let dv = GVec2::new(delta.x, delta.y);
        let navigating = mods.alt || resp.dragged_by(egui::PointerButton::Middle) || resp.dragged_by(egui::PointerButton::Secondary);
        if navigating {
            if (mods.alt && resp.dragged_by(egui::PointerButton::Primary)) || (!mods.alt && resp.dragged_by(egui::PointerButton::Secondary)) {
                self.camera.orbit(dv);
            } else if resp.dragged_by(egui::PointerButton::Middle) {
                self.camera.pan(dv, rect.height());
            } else if mods.alt && resp.dragged_by(egui::PointerButton::Secondary) {
                self.camera.dolly((delta.x - delta.y) * 0.005);
            }
        }
        if resp.hovered() && scroll != 0.0 {
            self.camera.dolly(scroll * 0.0015);
        }

        // Sculpt input. Every pointer sample of the frame is used, not just the last.
        let busy = self.job.is_some();
        if !busy && !navigating {
            if (resp.drag_started_by(egui::PointerButton::Primary) || (self.test.is_none() && resp.clicked_by(egui::PointerButton::Primary)))
                && let Some(p) = resp.interact_pointer_pos() {
                    let origin = ctx.input(|i| i.pointer.press_origin()).unwrap_or(p);
                    self.begin_stroke(origin, mods);
                }
            if !matches!(self.stroke, Stroke::None) && resp.dragged_by(egui::PointerButton::Primary) {
                for e in &events {
                    match e {
                        egui::Event::PointerMoved(p) => self.pending.push_back(Sample { pos: *p, pressure: self.pen.pressure }),
                        egui::Event::Touch { pos, force, .. } => {
                            self.pending.push_back(Sample { pos: *pos, pressure: force.unwrap_or(self.pen.pressure) })
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(test) = &mut self.test
            && self.doc.is_some() && self.job.is_none() {
                test.drive(rect, &mut self.pending);
            }
        self.apply_test_stroke_edges(mods);
        self.step_volume_demo(&ctx, rect);

        let last = resp.interact_pointer_pos();
        let dabs = self.process_samples(last);
        if dabs > 0 {
            self.last_edit = Instant::now();
        }
        let released = resp.drag_stopped_by(egui::PointerButton::Primary) || (resp.clicked_by(egui::PointerButton::Primary) && self.test.is_none());
        if released && self.pending.is_empty() && !matches!(&self.stroke, Stroke::Brush { dabs, .. } if !dabs.is_empty()) {
            if matches!(self.stroke, Stroke::Pose { .. }) {
                self.finish_pose(last.unwrap_or(rect.center()));
            } else {
                self.end_stroke();
            }
        }
        self.stats.input_ms = input_start.elapsed().as_secs_f32() * 1e3;
        self.stats.dabs = dabs;
        self.stats.pending = self.pending.len() + if let Stroke::Brush { dabs, .. } = &self.stroke { dabs.len() } else { 0 };

        // Brush cursor.
        self.cursor = None;
        if let (Some(mode), Some(hover)) = (&self.volume_mode, resp.hover_pos().or(last))
            && !navigating {
                let local = hover - rect.min;
                let ray = self.camera.ray(GVec2::new(local.x, local.y), GVec2::new(rect.width(), rect.height()));
                if let Some((hit, _)) = mode.raycast(ray.origin, ray.dir) {
                    let p = self.tools.params(self.tool);
                    self.cursor = Some((hit, p.size_px * self.camera.world_per_pixel(hit, rect.height()), p.hardness));
                }
            }
        if let (Some(doc), Some(hover), None) = (&self.doc, resp.hover_pos().or(last), &self.volume_mode)
            && !navigating && !busy {
                let local = hover - rect.min;
                let ray = self.camera.ray(GVec2::new(local.x, local.y), GVec2::new(rect.width(), rect.height()));
                if let Some(hit) = doc.raycast(&ray) {
                    let p = self.tools.params(self.tool);
                    self.cursor = Some((hit.point, p.size_px * self.camera.world_per_pixel(hit.point, rect.height()), p.hardness));
                }
            }

        // What a stroke would edit, next to the brush.
        if matches!(self.stroke, Stroke::None)
            && !navigating
            && !busy
            && !egui::Popup::is_any_open(&ctx)
            && let (Some(hover), Some(target)) = (resp.hover_pos(), crate::layer_panel::target::resolve(self))
        {
            crate::layer_panel::hud::chip(&ui.painter_at(rect), hover, &target, &self.theme.ui, self.theme.metrics.font_size, rect);
        }
        egui::Popup::context_menu(&resp).show(|ui| crate::layer_panel::menus::viewport_menu(self, ui));

        // Upload + render.
        let ppp = ctx.pixels_per_point();
        let px = [(rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32];
        crate::solid_preview::tick(self, ui.ctx());
        let showing_preview = self.solid_preview.shown();
        if let (Some(rs), Some(vp)) = (frame.wgpu_render_state(), self.viewport.as_mut()) {
            if let Some(mode) = self.volume_mode.as_mut() {
                mode.tick();
                if let Some(specks) = mode.take_specks()
                    && specks > 0
                {
                    self.status = format!("Removed {specks} speck{} the stroke left", if specks == 1 { "" } else { "s" });
                }
                if mode.busy() {
                    ctx.request_repaint();
                }
                vp.sync_volume(rs, mode);
                self.stats.upload = vp.last_upload;
            } else if let Some(doc) = self.doc.as_mut() {
                // Meshes dense enough that a generator edit would not fit a frame update their surface a little each frame.
                let want_progressive = match std::env::var("SCULPT_PROGRESSIVE").as_deref() {
                    Ok("off") => false,
                    Ok("on") => true,
                    _ => doc.face_count() >= PROGRESSIVE_MIN_FACES,
                };
                if doc.progressive() != want_progressive {
                    doc.set_progressive(want_progressive);
                }
                let editing = self.gen_timer.since_edit() < Duration::from_millis(150);
                if doc.pending_leaves() > 0 || self.gen_timer.waiting() {
                    // The camera decides what matters: visible and large on screen first, and while the edit is still being
                    // dragged, only what can be seen. Zooming in or moving closer shifts the work with it.
                    let aspect = px[0].max(1) as f32 / px[1].max(1) as f32;
                    let view_proj = self.camera.proj(aspect) * self.camera.view();
                    let focal_px = px[1] as f32 / (2.0 * (self.camera.fov_y * 0.5).tan());
                    let view = sculpt_core::lod::View::from_view_proj(view_proj, self.camera.eye(), focal_px);
                    let budget = Duration::from_millis(if editing { 6 } else { 14 });
                    let tp = Instant::now();
                    let left = doc.progress_pending(budget, Some(&view), editing);
                    self.gen_timer.note_progress(tp.elapsed().as_secs_f32() * 1e3, left);
                    if left > 0 {
                        ctx.request_repaint();
                    }
                }
                let idle = matches!(self.stroke, Stroke::None) && self.pending.is_empty() && self.last_edit.elapsed() >= LOD_REFRESH_IDLE && doc.pending_leaves() == 0 && !editing;
                let tr = Instant::now();
                if poll_lod_refresh(doc, &mut self.lod_refresh, &mut self.lod_batch, idle) {
                    ctx.request_repaint();
                }
                self.stats.lod_refresh_ms = tr.elapsed().as_secs_f32() * 1e3;
                let custom = self.pose_weights.as_deref();
                vp.shading = if self.flat_shading { crate::viewport::Shading::Flat } else { crate::viewport::Shading::Smooth };
                // Flat shading never reads normals, so the engine can skip computing them after layer and generator edits.
                if doc.lazy_normals() != self.flat_shading {
                    doc.set_lazy_normals(self.flat_shading);
                }
                let changed = std::mem::take(&mut self.pose_weights_changed);
                match self.solid_preview.doc.as_mut() {
                    Some(preview) if self.solid_preview.on => vp.sync(rs, preview, &self.overlay, custom, changed),
                    _ => vp.sync(rs, doc, &self.overlay, custom, changed),
                }
                self.stats.upload = vp.last_upload;
            }
            let t = Instant::now();
            let tau = self.lod_tau_override.or(self.view_detail.tau_px());
            let tau = tau.map(|t| if navigating { t * 2.0 } else { t });
            let lod = self.doc.as_ref().zip(tau).filter(|_| !showing_preview && self.volume_mode.is_none()).and_then(|(d, tau)| {
                d.lod().map(|tree| crate::viewport::LodInput { tree, bvh: d.bvh(), topology_id: d.topology_id(), tau_px: tau, budget: LOD_BUDGET })
            });
            let tex = vp.render(rs, px, &self.camera, &self.theme, self.overlay_strength, self.cursor, lod);
            self.stats.lod = vp.lod_stats;
            if std::env::var_os("SCULPT_LOD_LOG").is_some() && self.stats.lod.active {
                let stale = self.doc.as_ref().and_then(|d| d.lod()).map_or(0, |t| t.stale_nodes());
                eprintln!("lod: {} tris, {} patches, tau {:.2}px, select {:.2} ms, stale {}, refresh hand-off {:.2} ms, encode {:.1} ms, upload {:.2} ms {} KB, edit {:.1} ms, frame {:.1} ms", self.stats.lod.triangles, self.stats.lod.nodes, self.stats.lod.tau_px, self.stats.lod.select_ms, stale, self.stats.lod_refresh_ms, t.elapsed().as_secs_f32() * 1e3, self.stats.upload.ms, self.stats.upload.bytes / 1024, self.gen_timer.summary().map_or(0.0, |s| s.0), self.stats.frame_ms);
            }
            self.stats.render_ms = t.elapsed().as_secs_f32() * 1e3;
            if let Some(tex) = tex {
                ui.painter().image(tex, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
        }
        self.paint_hud(ui, rect);
        if let Stroke::Pose { pivot, .. } = &self.stroke {
            let size = GVec2::new(rect.width(), rect.height());
            if let (Some(a), Some(p)) = (self.camera.project(*pivot, size), last) {
                let a = rect.min + Vec2::new(a.x, a.y);
                ui.painter().line_segment([a, p], egui::Stroke::new(2.0, self.theme.viewport.cursor.0));
                ui.painter().circle_filled(a, 5.0, self.theme.viewport.cursor.0);
            }
        }
        if !matches!(self.stroke, Stroke::None) || !self.pending.is_empty() || self.test.is_some() {
            ctx.request_repaint();
        }
    }

    /// One frame of the scripted volume session: apply what it says, and ask for the frame's image.
    fn step_volume_demo(&mut self, ctx: &egui::Context, rect: Rect) {
        let Some(demo) = self.volume_demo.as_mut() else { return };
        if demo.waiting.is_some() || self.volume_mode.is_none() {
            ctx.request_repaint();
            return;
        }
        if demo.finished() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        let f = demo.step(rect, &self.camera);
        let shot = demo.current_frame() - 1;
        if let Some(c) = f.camera {
            self.camera = c;
        }
        if f.end {
            self.end_stroke();
        }
        if f.undo || f.redo {
            self.end_stroke();
            if let Some(m) = self.volume_mode.as_mut() {
                if f.undo {
                    m.undo();
                } else {
                    m.redo();
                }
            }
        }
        if let Some(m) = self.volume_mode.as_mut() {
            if let Some(c) = f.cap {
                m.cap = c.min(m.vol.levels() - 1);
            }
            if let Some(k) = f.keep_detail {
                m.vol.keep_detail = k;
            }
            if let Some(from) = f.make_layer {
                m.make_layer(from);
            }
            if let Some(t) = f.target {
                m.target = t;
            }
            if let Some(t) = f.tint {
                m.tint = t;
            }
        }
        if let (Some(on), Some(vp)) = (f.cull, self.viewport.as_mut()) {
            vp.cull = on;
        }
        if let Some((tool, size, strength, p)) = f.begin {
            self.select_tool(tool);
            let params = self.tools.params_mut(tool);
            params.size_px = size;
            params.strength = strength;
            self.pen.pressure = 1.0;
            self.begin_stroke(p, egui::Modifiers::NONE);
        }
        for p in f.samples {
            self.pending.push_back(Sample { pos: p, pressure: 1.0 });
        }
        if let Some(demo) = self.volume_demo.as_mut() {
            demo.waiting = Some(shot);
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        ctx.request_repaint();
    }

    /// Save the scripted session's frame once the renderer hands it back.
    fn collect_volume_demo_frame(&mut self, ctx: &egui::Context) {
        let Some(demo) = self.volume_demo.as_mut() else { return };
        let Some(n) = demo.waiting else { return };
        let image = ctx.input(|i| i.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
        if let Some(image) = image {
            let path = demo.frame_path(n);
            let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
            let r = std::fs::File::create(&path).map_err(|e| e.to_string()).and_then(|file| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                enc.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())
            });
            if let Err(e) = r {
                eprintln!("demo frame {n}: {e}");
            }
            if n % 150 == 0 {
                println!("demo: frame {n} ({:.0} s)", n as f32 / crate::volume_demo::FPS);
            }
            demo.waiting = None;
        }
    }

    /// The test driver signals stroke begin/end through `TestDriver::edges`.
    fn apply_test_stroke_edges(&mut self, mods: egui::Modifiers) {
        let Some(test) = &mut self.test else { return };
        let edges = std::mem::take(&mut test.edges);
        if edges.end {
            // Finish the previous stroke with only its own samples.
            let next: VecDeque<Sample> = if edges.begin.is_some() { std::mem::take(&mut self.pending) } else { VecDeque::new() };
            // Let the previous stroke finish its deferred dabs before lifting.
            while matches!(&self.stroke, Stroke::Brush { dabs, .. } if !dabs.is_empty()) {
                self.process_samples(None);
            }
            self.end_stroke();
            self.pending = next;
        }
        if let Some(tool) = edges.tool {
            let queued = std::mem::take(&mut self.pending);
            self.select_tool(tool);
            self.pending = queued;
        }
        if let Some(p) = edges.begin {
            let queued = std::mem::take(&mut self.pending);
            self.begin_stroke(p, mods);

            self.pending.extend(queued);
        }
    }

    fn paint_hud(&self, ui: &egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let col = self.theme.viewport.hud_text.0;
        if let Some((label, secs)) = self.busy() {
            let c = rect.center();
            painter.rect_filled(Rect::from_center_size(c, Vec2::new(360.0, 56.0)), 6.0, Color32::from_black_alpha(170));
            painter.text(c, egui::Align2::CENTER_CENTER, format!("{label}…  {secs:.1}s"), egui::FontId::proportional(16.0), Color32::WHITE);
        }
        self.paint_axis_gizmo(&painter, rect);
        if self.overlay == OverlayKind::LayerMask {
            // Make the mode unmistakable, and say how to leave it.
            let name = self.doc.as_ref().and_then(|d| d.active_layer().and_then(|id| d.layer(id))).map_or("mask", |l| l.name.as_str()).to_string();
            let text = format!("Viewing mask: {name}   ·   Alt+M or Esc to exit");
            let font = egui::FontId::proportional(self.theme.metrics.font_size);
            let galley = painter.layout_no_wrap(text, font.clone(), Color32::WHITE);
            let pill = Rect::from_center_size(Pos2::new(rect.center().x, rect.top() + 22.0), galley.size() + Vec2::new(26.0, 10.0));
            painter.rect_filled(pill, 12.0, Color32::from_black_alpha(185));
            painter.rect_stroke(pill, 12.0, egui::Stroke::new(1.5, self.theme.ui.target_mask()), egui::StrokeKind::Inside);
            painter.text(pill.center(), egui::Align2::CENTER_CENTER, galley.text(), font, Color32::WHITE);
        }
        if let Some(m) = &self.volume_mode {
            // The volume engine at a glance: its levels, what the GPU rebuilt, what it drew.
            let (counts, bytes) = m.vol.memory();
            let per: Vec<String> = counts.iter().map(|&c| fmt_count(c)).collect();
            // While recording, the frame rate counts the frame captures: show the frame's own work instead.
            let pace = if self.volume_demo.is_some() {
                format!("CPU frame {:.1} ms", self.stats.frame_ms)
            } else {
                format!("{:.0} fps", if self.stats.interval_ms > 0.0 { 1000.0 / self.stats.interval_ms } else { 0.0 })
            };
            let mut lines = vec![
                format!("Volume  ·  {} levels, showing 0-{}  ·  blocks per level {}  ·  {:.0} MB", counts.len(), m.cap, per.join(" / "), bytes as f64 / 1e6),
                format!("Last stroke at level {} (step {:.5})  ·  {:.2} ms a dab on the CPU", m.stroke_level, m.vol.voxel(m.stroke_level), m.dab_ms),
                format!("Surface rebuilt on the GPU: {} blocks in {:.2} ms ({} blocks sent, {:.2} ms CPU)", fmt_count(m.sync.extracted), m.gpu_ms, fmt_count(m.sync.uploaded), m.sync.cpu_ms),
                format!("Drawing {} of {} pages of quads{}  ·  {pace}", fmt_count(m.pages_drawn as usize), fmt_count(m.pages_used as usize), if self.viewport.as_ref().is_some_and(|v| v.cull) { " (the rest out of view or facing away)" } else { " (culling off)" }),
            ];
            if let Some(layer) = &m.layer {
                let f = layer.last;
                lines.push(format!(
                    "Detail layer: {} blocks  ·  last follow {:.1} ms over {} blocks, {} points (find {:.1}, move {:.1}, heights {:.1}, compose {:.1})",
                    fmt_count(layer.blocks()),
                    f.ms,
                    fmt_count(f.blocks),
                    fmt_count(f.points),
                    f.parts_ms[0],
                    f.parts_ms[1],
                    f.parts_ms[2],
                    f.parts_ms[3]
                ));
            }
            let font = egui::FontId::proportional(12.5);
            let width = lines.iter().map(|l| painter.layout_no_wrap(l.clone(), font.clone(), Color32::WHITE).size().x).fold(0.0f32, f32::max);
            painter.rect_filled(Rect::from_min_size(rect.min + Vec2::new(4.0, 4.0), Vec2::new(width + 14.0, 17.0 * lines.len() as f32 + 8.0)), 6.0, Color32::from_black_alpha(150));
            let mut y = rect.min.y + 8.0;
            for l in lines {
                painter.text(Pos2::new(rect.min.x + 11.0, y), egui::Align2::LEFT_TOP, l, font.clone(), Color32::from_gray(225));
                y += 17.0;
            }
            // The scripted session's caption.
            if let Some((text, alpha)) = self.volume_demo.as_ref().and_then(|d| d.caption.clone()) {
                let font = egui::FontId::proportional(19.0);
                let galley = painter.layout(text, font, Color32::WHITE.gamma_multiply(alpha), rect.width() - 120.0);
                let size = galley.size() + Vec2::new(36.0, 20.0);
                let pill = Rect::from_center_size(Pos2::new(rect.center().x, rect.max.y - 30.0 - size.y * 0.5), size);
                painter.rect_filled(pill, 10.0, Color32::from_black_alpha((190.0 * alpha) as u8));
                painter.galley(pill.min + Vec2::new(18.0, 10.0), galley, Color32::WHITE);
            }
            return;
        }
        if !self.hud {
            // Minimal Mudbox-style readout; H shows the full performance HUD.
            if let Some(doc) = &self.doc {
                let fps = if self.stats.interval_ms > 0.0 { 1000.0 / self.stats.interval_ms } else { 0.0 };
                let text = format!("{} faces   ·   {:.0} fps", fmt_count(doc.face_count()), fps);
                painter.text(Pos2::new(rect.max.x - 12.0, rect.max.y - 10.0), egui::Align2::RIGHT_BOTTOM, text, egui::FontId::proportional(11.0), col.gamma_multiply(0.65));
            }
            return;
        }
        let s = &self.stats;
        let mut lines = vec![format!(
            "{:>5.1} fps   frame {:>5.2} ms   cpu {:>5.2} ms",
            if s.interval_ms > 0.0 { 1000.0 / s.interval_ms } else { 0.0 },
            s.interval_ms,
            s.frame_ms
        )];
        lines.push(format!("input+dabs {:>5.2} ms ({} dabs, {} queued)   render {:>4.2} ms", s.input_ms, s.dabs, s.pending, s.render_ms));
        if s.lod.active {
            lines.push(format!("{} detail: {} tris in {} patches, tau {:.1}px, select {:.2} ms", self.view_detail.name(), fmt_count(s.lod.triangles as usize), s.lod.nodes, s.lod.tau_px, s.lod.select_ms));
        } else if self.view_detail == ViewDetail::Full && self.doc.as_ref().is_some_and(|d| d.lod().is_some()) {
            lines.push("Full detail: every triangle (Display > Viewport detail)".to_string());
        }
        lines.push(format!("upload {:>7.1} KB in {} ranges ({:.2} ms)", s.upload.bytes as f32 / 1024.0, s.upload.ranges, s.upload.ms));
        lines.push(format!("pen: {}  pressure {:.2}", self.pen.source.label(), self.pen.pressure));
        if let Some(doc) = &self.doc {
            lines.push(format!("{} faces · {} verts · {} subdivisions", fmt_count(doc.face_count()), fmt_count(doc.vertex_count()), doc.level()));
        }
        let mut y = rect.min.y + 8.0;
        for l in lines {
            painter.text(Pos2::new(rect.min.x + 10.0, y), egui::Align2::LEFT_TOP, l, egui::FontId::monospace(12.0), col);
            y += 16.0;
        }
    }

    /// XYZ orientation triad in the bottom-left corner of the viewport.
    fn paint_axis_gizmo(&self, painter: &egui::Painter, rect: Rect) {
        let center = Pos2::new(rect.min.x + 42.0, rect.max.y - 42.0);
        let len = 26.0;
        painter.circle_filled(center, len + 8.0, Color32::from_black_alpha(40));
        let (right, up, fwd) = (self.camera.right(), self.camera.up(), self.camera.forward());
        let mut axes = [
            (Vec3::X, Color32::from_rgb(232, 86, 86), "X"),
            (Vec3::Y, Color32::from_rgb(120, 200, 90), "Y"),
            (Vec3::Z, Color32::from_rgb(80, 140, 240), "Z"),
        ];
        // Far axes first so near ones draw on top.
        axes.sort_by(|a, b| b.0.dot(fwd).total_cmp(&a.0.dot(fwd)));
        for (axis, color, label) in axes {
            let dir = Vec2::new(axis.dot(right), -axis.dot(up));
            let tip = center + dir * len;
            let facing_away = axis.dot(fwd) > 0.0;
            let c = if facing_away { color.gamma_multiply(0.55) } else { color };
            painter.line_segment([center, tip], egui::Stroke::new(2.0, c));
            painter.circle_filled(tip, 7.5, c);
            painter.text(tip, egui::Align2::CENTER_CENTER, label, egui::FontId::new(9.5, crate::theme::bold()), Color32::from_black_alpha(220));
        }
    }

    pub fn tool_hotkey(&self, ctx: &egui::Context, t: Tool) -> String {
        self.keymap.shortcut_text(ctx, t.command()).unwrap_or_default()
    }
}

/// How faithfully dense meshes are drawn: the level-of-detail tolerance, or the full mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewDetail {
    Draft,
    Balanced,
    Sharp,
    /// Every triangle, no level of detail: the reference for what the mesh really is.
    Full,
}

impl ViewDetail {
    pub const ALL: [ViewDetail; 4] = [ViewDetail::Draft, ViewDetail::Balanced, ViewDetail::Sharp, ViewDetail::Full];

    /// Screen-space error tolerance in pixels; `None` draws every triangle.
    pub fn tau_px(self) -> Option<f32> {
        match self {
            ViewDetail::Draft => Some(2.0),
            ViewDetail::Balanced => Some(1.0),
            ViewDetail::Sharp => Some(0.5),
            ViewDetail::Full => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ViewDetail::Draft => "Draft",
            ViewDetail::Balanced => "Balanced",
            ViewDetail::Sharp => "Sharp",
            ViewDetail::Full => "Full",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            ViewDetail::Draft => "Fastest; up to 2 px of simplification",
            ViewDetail::Balanced => "About one pixel of simplification",
            ViewDetail::Sharp => "Half-pixel; roughly twice the triangles",
            ViewDetail::Full => "Every triangle; slow on very dense meshes",
        }
    }

    fn from_name(s: &str) -> Option<ViewDetail> {
        ViewDetail::ALL.into_iter().find(|d| d.name() == s)
    }
}

/// How long the pen must rest before outdated LOD patches are re-simplified.
const LOD_REFRESH_IDLE: std::time::Duration = std::time::Duration::from_millis(250);
/// Triangles handed to the background re-simplifier at a time. The copy on the UI thread is about
/// 0.5 ms per 32k triangles, so the batch only grows when frames are slow enough to hide it.
const LOD_REFRESH_BATCH_MIN: usize = 32_768;
const LOD_REFRESH_BATCH_MAX: usize = 262_144;

/// Catch level-of-detail patches up with edits without ever blocking the UI thread on simplification:
/// install a finished batch if one is back, and when `idle`, copy out the next batch for a worker thread.
/// Edited areas draw at full detail until then, so the picture is always correct. Returns true while work remains.
fn poll_lod_refresh(doc: &mut Document, slot: &mut Option<(u64, std::sync::mpsc::Receiver<sculpt_core::lod::RefreshDone>)>, batch_tris: &mut usize, idle: bool) -> bool {
    use std::sync::mpsc::TryRecvError;
    if let Some((topo, rx)) = slot {
        match rx.try_recv() {
            Ok(done) => {
                doc.apply_lod_refresh(*topo, done);
                *slot = None;
                // Finished within a frame: the worker sat idle, so give it more next time.
                *batch_tris = (*batch_tris * 2).min(LOD_REFRESH_BATCH_MAX);
            }
            Err(TryRecvError::Empty) => {
                *batch_tris = (*batch_tris / 2).max(LOD_REFRESH_BATCH_MIN);
                return true;
            }
            Err(TryRecvError::Disconnected) => *slot = None,
        }
    }
    let stale = doc.lod().is_some_and(|t| t.stale_nodes() > 0);
    if idle && let Some((topo, batch)) = doc.gather_lod_refresh(*batch_tris) {
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("lod refresh".into()).spawn(move || {
            let _ = tx.send(batch.run());
        });
        if spawned.is_ok() {
            *slot = Some((topo, rx));
        }
    }
    stale
}

/// Meshes at or above this many triangles are drawn through the LOD tree.
const LOD_MIN_TRIANGLES: usize = 1_500_000;
/// Most triangles drawn per frame, whatever the pixel tolerance asks for.
const LOD_BUDGET: u64 = 6_000_000;
/// From this many faces up, generator edits update the surface progressively instead of all at once.
const PROGRESSIVE_MIN_FACES: usize = 600_000;

pub fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

pub fn new_sphere(level: u32) -> Result<Document, String> {
    let base = level.min(6);
    let mut doc = Document::from_mesh(quad_sphere(base, 1.0)).map_err(|e| e.to_string())?;
    while doc.level() + base < level {
        doc.subdivide().map_err(|e| e.to_string())?;
    }
    Ok(doc)
}

impl eframe::App for SculptApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let t0 = Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = prev.elapsed().as_secs_f32() * 1e3;
            self.stats.interval_ms = if self.stats.interval_ms == 0.0 { dt } else { self.stats.interval_ms * 0.9 + dt * 0.1 };
        }
        self.last_frame = Some(t0);
        self.poll_job();
        crate::generator_ui::collect_screenshot(self, &ctx);
        self.collect_volume_demo_frame(&ctx);
        crate::generator_ui::tick_sweep(self, &ctx);
        if self.job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        if self.last_theme_poll.elapsed() > Duration::from_secs(1) {
            self.last_theme_poll = Instant::now();
            if self.themes.poll_changes() && self.theme_editor.is_none() {
                let t = self.themes.get(&self.theme.name);
                t.apply(&ctx);
                self.theme = t;
            }
        }

        let esc_free = self.layers.drag.is_none() && self.layers.rename.is_none() && self.layers.switcher.is_none();
        if self.overlay == OverlayKind::LayerMask && esc_free && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            self.overlay = OverlayKind::None;
        }
        for cmd in self.keymap.poll(&ctx) {
            self.run(&ctx, cmd);
        }
        crate::panels::menu_bar(self, ui);
        crate::panels::context_bar(self, ui);
        crate::panels::status_bar(self, ui);
        crate::panels::tray(self, ui);
        crate::panels::right_panel(self, ui);
        crate::layer_panel::switcher::show(self, &ctx);
        crate::layer_panel::radial::show(self, &ctx);
        crate::panels::dialogs(self, &ctx);
        crate::panels::theme_editor(self, &ctx);
        crate::panels::keymap_window(self, &ctx);
        crate::panels::mesh_info(self, &ctx);
        if self.mask_dirty && !ctx.input(|i| i.pointer.any_down()) {
            self.mask_dirty = false;
            crate::panels::apply_mask_edit(self);
        }

        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport_ui(ui, frame));

        self.stats.frame_ms = t0.elapsed().as_secs_f32() * 1e3;
        if let Some(test) = &mut self.test {
            let doc_ready = self.doc.is_some() && self.job.is_none();
            test.record(&self.stats, doc_ready, self.doc.as_ref());
            let settled = self.doc.as_ref().and_then(|d| d.lod()).is_none_or(|t| t.stale_nodes() == 0);
            test.finish_if_done(&ctx, settled);
            for e in ctx.input(|i| i.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = e {
                    test.save_screenshot(&image);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    fn on_exit(&mut self) {
        self.save_settings();
    }
}

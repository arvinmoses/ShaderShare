//! Application state and the per-frame loop.
//!
//! Frame order is chosen for latency: commands → panels → *viewport input is turned into dabs on the volume* → the
//! changed blocks sent to the GPU and the surface rebuilt there → scene rendered → presented, all within one frame.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Sense, Vec2};
use glam::{Vec2 as GVec2, Vec3};

use crate::camera::Camera;
use crate::keymap::{Command, Keymap};
use crate::pen::Pen;
use crate::theme::{Theme, ThemeLibrary};
use crate::tools::{Tool, ToolBox};
use crate::viewport::Viewport;
use crate::volume_mode::VolumeMode;

pub struct Options {
    /// A `.svol` file to open instead of a new ball.
    pub file: Option<PathBuf>,
    /// The new ball's size: about this many triangles at level 0.
    pub triangles: f32,
    pub theme: Option<String>,
    /// Multiply the initial framing distance (below 1 zooms in), for close-up screenshots.
    pub zoom: f32,
    /// Save the window here once the volume has settled, then close.
    pub screenshot: Option<PathBuf>,
    /// Run the scripted session, saving every frame into this folder (`volume_demo.rs`).
    pub demo_volume: Option<PathBuf>,
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub pos: Pos2,
    pub pressure: f32,
}

/// A stroke in progress: dabs every `spacing` along the surface since `last`.
struct Stroke {
    carve: bool,
    size_px: f32,
    strength: f32,
    hardness: f32,
    last: Option<Vec3>,
    carry: f32,
}

pub enum Dialog {
    Open(String),
    SaveAs(String),
    /// Replace the volume with a new ball of about this many triangles.
    New(f32),
}

#[derive(Default, Clone, Copy)]
pub struct FrameStats {
    pub frame_ms: f32,
    pub interval_ms: f32,
}

/// `--screenshot`: the frame to save once the volume has settled.
struct Shot {
    path: PathBuf,
    frames: u32,
    requested: bool,
}

pub struct SculptApp {
    /// The volume being sculpted, with its layers.
    pub volume: VolumeMode,
    /// Where the volume was last saved or opened.
    pub volume_path: Option<String>,
    pub(crate) viewport: Option<Viewport>,
    pub camera: Camera,
    pub themes: ThemeLibrary,
    pub theme: Theme,
    pub theme_editor: Option<Theme>,
    pub keymap: Keymap,
    pub tool: Tool,
    pub tools: ToolBox,
    stroke: Option<Stroke>,
    pending: VecDeque<Sample>,
    /// The engine readout over the viewport (H).
    pub hud: bool,
    pub stats: FrameStats,
    last_frame: Option<Instant>,
    pub dialog: Option<Dialog>,
    pub status: String,
    /// Last status text seen by the status bar and when it appeared.
    pub status_seen: (String, Instant),
    pub pen: Pen,
    cursor: Option<(Vec3, f32, f32)>,
    viewport_rect: Rect,
    last_theme_poll: Instant,
    pub show_keymap: bool,
    /// The scripted session being recorded, if any.
    volume_demo: Option<crate::volume_demo::VolumeDemo>,
    shot: Option<Shot>,
    /// The GPU surface is still fading levels or landing a big change.
    settling: bool,
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

/// Write an egui screenshot as a PNG. `fast` trades file size for encoding time (frames saved while recording).
pub fn write_png(path: &std::path::Path, image: &egui::ColorImage, fast: bool) -> Result<(), String> {
    let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    if fast {
        enc.set_compression(png::Compression::Fast);
    }
    enc.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())
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
        let tools = ToolBox::from_settings(&settings["tools"]);
        // `SCULPT_VOLUME=M`: the new ball's millions of triangles (older scripts).
        let triangles = std::env::var("SCULPT_VOLUME").ok().and_then(|v| v.parse::<f32>().ok()).map_or(opts.triangles, |m| m * 1e6);

        let mut app = SculptApp {
            volume: VolumeMode::ball(triangles),
            volume_path: None,
            viewport: cc.wgpu_render_state.as_ref().map(Viewport::new),
            camera: Camera::default(),
            themes,
            theme,
            theme_editor: None,
            keymap,
            tool: Tool::ClayBuildup,
            tools,
            stroke: None,
            pending: VecDeque::new(),
            hud: settings["hud"].as_bool().unwrap_or(true),
            stats: FrameStats::default(),
            last_frame: None,
            dialog: None,
            status: String::new(),
            status_seen: (String::new(), Instant::now()),
            pen: Pen::new(cc),
            cursor: None,
            viewport_rect: Rect::NOTHING,
            last_theme_poll: Instant::now(),
            show_keymap: false,
            volume_demo: opts.demo_volume.map(crate::volume_demo::VolumeDemo::new),
            shot: opts.screenshot.map(|path| Shot { path, frames: 0, requested: false }),
            settling: false,
        };
        for e in app.themes.errors.iter().chain(&app.keymap.errors) {
            eprintln!("config: {e}");
        }
        app.frame_volume();
        app.status = format!("{:.1}M-triangle ball. Clay Buildup adds, Trim Dynamic carves, Ctrl swaps them.", triangles / 1e6);
        if let Some(p) = opts.file {
            app.open(p.display().to_string());
            app.frame_volume();
        }
        app.camera.distance *= opts.zoom;
        app
    }

    pub fn save_settings(&self) {
        let cfg = config_dir();
        let _ = std::fs::create_dir_all(&cfg);
        let v = serde_json::json!({ "theme": self.theme.name, "tools": self.tools, "hud": self.hud });
        let _ = std::fs::write(cfg.join("settings.json"), serde_json::to_string_pretty(&v).unwrap());
    }

    pub fn set_theme(&mut self, ctx: &egui::Context, theme: Theme) {
        theme.apply(ctx);
        self.theme = theme;
        self.save_settings();
    }

    // ------------------------------------------------------------- commands

    pub fn run(&mut self, _ctx: &egui::Context, cmd: Command) {
        if let Some(t) = Tool::ALL.iter().find(|t| t.command() == cmd).copied() {
            self.select_tool(t);
            return;
        }
        match cmd {
            Command::Undo | Command::Redo => {
                self.end_stroke();
                let done = if cmd == Command::Undo { self.volume.undo() } else { self.volume.redo() };
                self.status = match (cmd, done) {
                    (Command::Undo, true) => "Undo",
                    (Command::Undo, false) => "Nothing to undo",
                    (_, true) => "Redo",
                    (_, false) => "Nothing to redo",
                }
                .into();
            }
            Command::Save => match self.volume_path.clone() {
                Some(p) => self.save(p),
                None => self.dialog = Some(Dialog::SaveAs("untitled.svol".into())),
            },
            Command::Open => self.dialog = Some(Dialog::Open(self.volume_path.clone().unwrap_or_default())),
            Command::BrushSizeUp | Command::BrushSizeDown => {
                let p = self.tools.params_mut(self.tool);
                p.size_px = (p.size_px * if cmd == Command::BrushSizeUp { 1.15 } else { 1.0 / 1.15 }).clamp(2.0, 1000.0);
            }
            Command::StrengthUp | Command::StrengthDown => {
                let p = self.tools.params_mut(self.tool);
                p.strength = (p.strength + if cmd == Command::StrengthUp { 0.05 } else { -0.05 }).clamp(0.0, 1.0);
            }
            Command::Frame => self.frame_volume(),
            Command::ToggleHud => {
                self.hud = !self.hud;
                self.save_settings();
            }
            Command::ToolClayBuildup | Command::ToolTrimDynamic => {}
        }
    }

    pub fn select_tool(&mut self, t: Tool) {
        self.end_stroke();
        self.tool = t;
    }

    pub fn frame_volume(&mut self) {
        let (lo, hi) = self.volume.vol.bounds();
        self.camera.frame(&sculpt_core::geom::Aabb { min: lo, max: hi });
    }

    /// Put `mode` in place of the volume, drawn afresh.
    fn replace_volume(&mut self, mut mode: VolumeMode) {
        self.end_stroke();
        mode.generation = self.volume.generation + 1;
        self.volume = mode;
        if let Some(vp) = self.viewport.as_mut() {
            vp.clear_volume();
        }
    }

    /// A fresh ball of about `triangles` triangles in place of the volume.
    pub fn new_volume(&mut self, triangles: f32) {
        let t = Instant::now();
        self.replace_volume(VolumeMode::ball(triangles));
        self.volume_path = None;
        self.frame_volume();
        self.status = format!("New {:.1}M-triangle ball in {:.0} ms", triangles / 1e6, t.elapsed().as_secs_f64() * 1e3);
    }

    pub fn open(&mut self, p: String) {
        let t = Instant::now();
        match std::fs::File::open(&p).and_then(|f| VolumeMode::read_from(&mut std::io::BufReader::new(f))) {
            Ok(mode) => {
                self.replace_volume(mode);
                self.frame_volume();
                self.status = format!("Opened {} ({:.2}s)", file_name(&p), t.elapsed().as_secs_f32());
                self.volume_path = Some(p);
            }
            Err(e) => self.status = format!("Open failed: {p}: {e}"),
        }
    }

    pub fn save(&mut self, p: String) {
        let t = Instant::now();
        self.end_stroke();
        let mode = &mut self.volume;
        mode.settle();
        let r = std::fs::File::create(&p).and_then(|f| {
            let mut w = std::io::BufWriter::new(f);
            mode.write_to(&mut w)?;
            std::io::Write::flush(&mut w)
        });
        match r {
            Ok(()) => {
                self.status = format!("Saved {} ({:.2}s)", file_name(&p), t.elapsed().as_secs_f32());
                self.volume_path = Some(p);
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    pub fn run_dialog_action(&mut self, d: Dialog) {
        match d {
            Dialog::Open(p) => self.open(p),
            Dialog::SaveAs(p) => self.save(p),
            Dialog::New(triangles) => self.new_volume(triangles),
        }
    }

    // ---------------------------------------------------------------- strokes

    fn end_stroke(&mut self) {
        self.pending.clear();
        if self.stroke.take().is_some() {
            self.volume.end_stroke();
        }
    }

    fn begin_stroke(&mut self, pos: Pos2, modifiers: egui::Modifiers) {
        let p = self.tools.params(self.tool).clone();
        self.stroke = Some(Stroke { carve: self.tool.carves(modifiers.command), size_px: p.size_px, strength: p.strength, hardness: p.hardness, last: None, carry: 0.0 });
        self.volume.begin_stroke();
        self.pending.push_front(Sample { pos, pressure: self.pen.pressure });
    }

    /// Turn queued pen samples into dabs. Returns how many were applied.
    fn process_samples(&mut self) -> usize {
        let mut dabs = 0;
        let size = self.viewport_rect.size();
        let camera = self.camera.clone();
        let Some(Stroke { carve, size_px, strength, hardness, last, carry }) = self.stroke.as_mut() else {
            self.pending.clear();
            return 0;
        };
        let mode = &mut self.volume;
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
            self.status = format!("{dabs} dabs this frame at level {} (step {:.5}), {:.2} ms ({:.2} ms each)", mode.level, mode.voxel(), mode.last_ms, mode.last_ms / dabs as f64);
        }
        dabs
    }

    // --------------------------------------------------------------- viewport

    fn viewport_ui(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.viewport_rect = rect;
        let ctx = ui.ctx().clone();

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
        if !navigating {
            if (resp.drag_started_by(egui::PointerButton::Primary) || resp.clicked_by(egui::PointerButton::Primary))
                && let Some(p) = resp.interact_pointer_pos()
            {
                let origin = ctx.input(|i| i.pointer.press_origin()).unwrap_or(p);
                self.begin_stroke(origin, mods);
            }
            if self.stroke.is_some() && resp.dragged_by(egui::PointerButton::Primary) {
                for e in &events {
                    match e {
                        egui::Event::PointerMoved(p) => self.pending.push_back(Sample { pos: *p, pressure: self.pen.pressure }),
                        egui::Event::Touch { pos, force, .. } => self.pending.push_back(Sample { pos: *pos, pressure: force.unwrap_or(self.pen.pressure) }),
                        _ => {}
                    }
                }
            }
        }
        self.step_volume_demo(&ctx, rect);

        let last = resp.interact_pointer_pos();
        self.process_samples();
        let released = resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked_by(egui::PointerButton::Primary);
        if released && self.pending.is_empty() {
            self.end_stroke();
        }

        // Brush cursor.
        self.cursor = None;
        if let Some(hover) = resp.hover_pos().or(last)
            && !navigating
        {
            let local = hover - rect.min;
            let ray = self.camera.ray(GVec2::new(local.x, local.y), GVec2::new(rect.width(), rect.height()));
            if let Some((hit, _)) = self.volume.raycast(ray.origin, ray.dir) {
                let p = self.tools.params(self.tool);
                self.cursor = Some((hit, p.size_px * self.camera.world_per_pixel(hit, rect.height()), p.hardness));
            }
        }

        // Upload + render.
        let ppp = ctx.pixels_per_point();
        let px = [(rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32];
        if let (Some(rs), Some(vp)) = (frame.wgpu_render_state(), self.viewport.as_mut()) {
            let mode = &mut self.volume;
            mode.tick();
            if let Some(specks) = mode.take_specks()
                && specks > 0
            {
                self.status = format!("Removed {specks} speck{} the stroke left", if specks == 1 { "" } else { "s" });
            }
            if mode.busy() {
                ctx.request_repaint();
            }
            let focal_px = px[1].max(1) as f32 / (2.0 * (self.camera.fov_y * 0.5).tan());
            // A scripted session runs on its own clock, and may choose the levels from an eye of its own.
            let demo = self.volume_demo.as_ref();
            let eye = demo.and_then(|d| d.level_eye).unwrap_or(self.camera.eye());
            self.settling = vp.sync_volume(rs, mode, eye, focal_px, demo.map(|d| d.clock()));
            if self.settling {
                ctx.request_repaint();
            }
            if let Some(tex) = vp.render(rs, px, &self.camera, &self.theme, self.cursor) {
                ui.painter().image(tex, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
        }
        self.paint_hud(ui, rect);
        if self.stroke.is_some() || !self.pending.is_empty() || self.shot.is_some() {
            ctx.request_repaint();
        }
    }

    /// One frame of the scripted volume session: apply what it says, and ask for the frame's image.
    fn step_volume_demo(&mut self, ctx: &egui::Context, rect: Rect) {
        let Some(demo) = self.volume_demo.as_mut() else { return };
        if demo.waiting.is_some() {
            ctx.request_repaint();
            return;
        }
        if demo.finished() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        let f = demo.step(rect, &self.camera);
        let shot = demo.current_frame() - 1;
        // For looking into a frame: the volume as it is when that frame begins (`SCULPT_DEMO_DUMP=frame,frame,...`).
        if let Ok(list) = std::env::var("SCULPT_DEMO_DUMP")
            && list.split(',').any(|x| x.trim().parse::<u32>().ok() == Some(shot))
            && let Ok(file) = std::fs::File::create(demo.dir.join(format!("dump_{shot:05}.svol")))
        {
            let m = &mut self.volume;
            let _ = m.vol.write_to(&mut std::io::BufWriter::new(file));
            // And with its layers, to work them out again.
            if let Ok(file) = std::fs::File::create(demo.dir.join(format!("layers_{shot:05}.svol"))) {
                let _ = m.write_to(&mut std::io::BufWriter::new(file));
            }
        }
        // And a full rebuild of the GPU's copy at these frames (`SCULPT_DEMO_REBUILD=frame,...`).
        if let Ok(list) = std::env::var("SCULPT_DEMO_REBUILD")
            && list.split(',').any(|x| x.trim().parse::<u32>().ok() == Some(shot))
        {
            self.volume.generation += 1;
        }
        if let Some(c) = f.camera {
            self.camera = c;
        }
        if f.end {
            self.end_stroke();
        }
        if f.undo || f.redo {
            self.end_stroke();
            if f.undo {
                self.volume.undo();
            } else {
                self.volume.redo();
            }
        }
        let m = &mut self.volume;
        if let Some(c) = f.cap {
            m.cap = c.min(m.vol.levels() - 1);
        }
        if let Some(k) = f.keep_detail {
            m.vol.keep_detail = k;
        }
        if let Some(from) = f.make_layer {
            m.make_layer(from);
        }
        if let Some(l) = f.add_layer {
            m.add_layer(l);
        }
        if let Some((k, on)) = f.layer_visible {
            m.set_layer_visible(k, on);
        }
        if let Some((k, l)) = f.layer_level {
            m.set_layer_level(k, l);
        }
        if let Some(k) = f.remove_layer {
            m.remove_layer(k);
        }
        if let Some(k) = f.active_layer {
            m.set_active_layer(k);
        }
        if let Some(to) = f.fill_mask {
            m.fill_mask(to);
        }
        if let Some(level) = f.bake_noise
            && let Some(k) = m.layer.as_ref().and_then(|l| l.layers().iter().rposition(|x| x.noise))
        {
            m.bake_noise(k, level);
        }
        if let Some(t) = f.target {
            m.target = t;
        }
        if let Some(t) = f.tint {
            m.tint = t;
        }
        if let Some(on) = f.by_distance {
            m.by_distance = on;
        }
        if f.add_generator {
            m.add_generator();
        }
        if let Some(edit) = f.noise {
            // The topmost surface noise layer (made if there is none).
            let find = |m: &VolumeMode| m.layer.as_ref().and_then(|l| l.layers().iter().rposition(|x| x.noise));
            if find(m).is_none() {
                m.add_noise();
            }
            if let Some(k) = find(m)
                && let Some(mut n) = m.layer.as_ref().and_then(|l| l.noise(k)).copied()
            {
                edit(&mut n);
                m.set_noise(k, n, false);
            }
        }
        if let Some(edit) = f.generator
            && let Some(layer) = m.layer.as_ref()
        {
            let k = layer.active();
            if let Some(mut spec) = layer.generator(k).cloned() {
                edit(&mut spec);
                m.set_generator(k, &spec, false);
            }
        }
        if f.save_open {
            // Through a file in the frames' folder: what is shown afterwards is what the file holds.
            let path = self.volume_demo.as_ref().map(|d| d.dir.join("round_trip.svol")).unwrap_or_default();
            let m = &mut self.volume;
            let saved = std::fs::File::create(&path).and_then(|f| m.write_to(&mut std::io::BufWriter::new(f)));
            let opened = saved.and_then(|_| std::fs::File::open(&path)).and_then(|f| VolumeMode::read_from(&mut std::io::BufReader::new(f)));
            match opened {
                Ok(mut mode) => {
                    mode.target = m.target;
                    self.replace_volume(mode);
                }
                Err(e) => eprintln!("demo: the file round trip failed: {e}"),
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
            // Encoded on the frame: the fast setting keeps the frame's own work readable in its readout.
            if let Err(e) = write_png(&demo.frame_path(n), &image, true) {
                eprintln!("demo frame {n}: {e}");
            }
            if n % 150 == 0 {
                println!("demo: frame {n} ({:.0} s)", n as f32 / crate::volume_demo::FPS);
            }
            demo.waiting = None;
        }
    }

    /// `--screenshot`: once the volume has settled (or after a while regardless), save the window and close.
    fn step_screenshot(&mut self, ctx: &egui::Context) {
        let Some(shot) = self.shot.as_mut() else { return };
        if shot.requested {
            let image = ctx.input(|i| i.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
            if let Some(image) = image {
                match write_png(&shot.path, &image, false) {
                    Ok(()) => println!("screenshot saved to {}", shot.path.display()),
                    Err(e) => eprintln!("screenshot: {e}"),
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            return;
        }
        shot.frames += 1;
        let settled = !self.volume.busy() && !self.settling;
        if (shot.frames >= 30 && settled) || shot.frames >= 900 {
            shot.requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        ctx.request_repaint();
    }

    fn paint_hud(&self, ui: &egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        self.paint_axis_gizmo(&painter, rect);
        let m = &self.volume;
        // The scripted session's caption.
        if let Some((text, alpha)) = self.volume_demo.as_ref().and_then(|d| d.caption.clone()) {
            let font = egui::FontId::proportional(19.0);
            let galley = painter.layout(text, font, Color32::WHITE.gamma_multiply(alpha), rect.width() - 120.0);
            let size = galley.size() + Vec2::new(36.0, 20.0);
            let pill = Rect::from_center_size(Pos2::new(rect.center().x, rect.max.y - 30.0 - size.y * 0.5), size);
            painter.rect_filled(pill, 10.0, Color32::from_black_alpha((190.0 * alpha) as u8));
            painter.galley(pill.min + Vec2::new(18.0, 10.0), galley, Color32::WHITE);
        }
        if !self.hud && self.volume_demo.is_none() {
            return;
        }
        // The volume engine at a glance: its levels, what the GPU rebuilt, what it drew.
        let (counts, bytes) = m.vol.memory();
        let upload = self.viewport.as_ref().map(|v| v.last_upload).unwrap_or_default();
        let per: Vec<String> = counts.iter().map(|&c| fmt_count(c)).collect();
        // While recording, the frame rate counts the frame captures: show the frame's own work instead.
        let pace = if self.volume_demo.is_some() {
            format!("CPU frame {:.1} ms", self.stats.frame_ms)
        } else {
            format!("{:.0} fps", if self.stats.interval_ms > 0.0 { 1000.0 / self.stats.interval_ms } else { 0.0 })
        };
        let mut lines = vec![
            format!("Volume  ·  {} levels, showing 0-{}  ·  blocks per level {}  ·  {:.0} MB", counts.len(), m.cap, per.join(" / "), bytes as f64 / 1e6),
            if m.by_distance {
                format!(
                    "Level by distance (detail down to {:.2} px{}): {} blocks shown, {} left to coarser levels{}  ·  about {} triangles{}  ·  {:.2} ms",
                    m.view.tau_used.max(m.level_px),
                    if m.budget_on && m.view.tau_used > m.level_px * 1.001 { ", raised for the budget" } else { "" },
                    fmt_count(m.view.shown),
                    fmt_count(m.view.hidden),
                    if m.view.stale > 0 { format!(", {} to measure", fmt_count(m.view.stale)) } else { String::new() },
                    fmt_count(m.view.triangles as usize),
                    if m.budget_on { format!(" of {:.1}M", m.budget_m) } else { String::new() },
                    m.view.ms
                )
            } else {
                "Level by distance off: every level shown".to_string()
            },
            format!("Last stroke at level {} (step {:.5})  ·  {:.2} ms a dab on the CPU", m.stroke_level, m.vol.voxel(m.stroke_level), m.dab_ms),
            format!(
                "Surface rebuilt on the GPU: {} blocks in {:.2} ms ({} blocks sent, {:.2} ms CPU{})  ·  this frame {} sent, {:.0} KB in {:.2} ms",
                fmt_count(m.sync.extracted),
                m.gpu_ms,
                fmt_count(m.sync.uploaded),
                m.sync.cpu_ms,
                if m.sync.pending > 0 { format!(", {} still to land", fmt_count(m.sync.pending)) } else { String::new() },
                fmt_count(upload.ranges),
                upload.bytes as f64 / 1024.0,
                upload.ms
            ),
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

/// The last part of a path, for messages and the title.
pub fn file_name(p: &str) -> String {
    std::path::Path::new(p).file_name().map_or(p.to_string(), |n| n.to_string_lossy().into_owned())
}

pub fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
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
        self.collect_volume_demo_frame(&ctx);
        self.step_screenshot(&ctx);
        if self.last_theme_poll.elapsed() > Duration::from_secs(1) {
            self.last_theme_poll = Instant::now();
            if self.themes.poll_changes() && self.theme_editor.is_none() {
                let t = self.themes.get(&self.theme.name);
                t.apply(&ctx);
                self.theme = t;
            }
        }

        for cmd in self.keymap.poll(&ctx) {
            self.run(&ctx, cmd);
        }
        crate::panels::menu_bar(self, ui);
        crate::panels::context_bar(self, ui);
        crate::panels::status_bar(self, ui);
        crate::panels::tray(self, ui);
        crate::panels::right_panel(self, ui);
        crate::panels::dialogs(self, &ctx);
        crate::panels::theme_editor(self, &ctx);
        crate::panels::keymap_window(self, &ctx);

        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport_ui(ui, frame));

        self.stats.frame_ms = t0.elapsed().as_secs_f32() * 1e3;
    }

    fn on_exit(&mut self) {
        self.save_settings();
    }
}

//! Live preview of the solidified stack.
//!
//! The viewport shows what Solidify would make whenever the stack has generator layers: one clean closed mesh with soft
//! blends, instead of the live displacement, which cannot make a shape stand out of the surface (added spheres come out
//! jammed against it). It is on by default. The live layers stay what you edit and the preview follows them.
//!
//! Edits never wait for it. The preview is built on a worker thread with its own small pool from a snapshot of the stack;
//! a new one starts as soon as the last has finished and something changed. While a slider is being dragged it is built
//! at a third of the detail, so it keeps up with the hand; once the hand stops, a full-detail one
//! replaces it. The last finished result stays on screen until the next replaces it. While a
//! sculpt stroke is under way the live mesh is shown instead, so the brush is never drawn on a stale surface.

use std::hash::{Hash, Hasher};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sculpt_core::Document;
use sculpt_core::document::{SolidSettings, SolidStats};

use crate::app::SculptApp;

/// How long after the last brush dab the live mesh stays on screen, so a stroke is seen as it is made.
const AFTER_STROKE: Duration = Duration::from_millis(300);
/// An edit this recent counts as a drag still in progress: preview coarsely.
const DRAGGING: Duration = Duration::from_millis(150);

/// The preview's own worker threads, a third of the machine, so a preview in flight never competes with a live edit for the
/// shared pool when the hand starts moving again.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 3).max(2);
        rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("solid-preview-{i}")).build().expect("preview thread pool")
    })
}

type Finished = Result<(Document, SolidStats, f32, bool), String>;

pub struct SolidPreview {
    pub on: bool,
    /// The last finished preview, shown in place of the live mesh.
    pub doc: Option<Document>,
    job: Option<JoinHandle<Finished>>,
    started: Instant,
    last_since: Duration,
    signature: u64,
    dirty: bool,
    /// A sculpt stroke is under way (or just ended): show the live mesh.
    hold_live: bool,
    /// The preview on screen was built at drag detail and wants a full-detail pass.
    shown_coarse: bool,
    /// A brush stroke changed the surface since the last preview started: the next one is built at drag detail first.
    stroked: bool,
    last_dab_age: Duration,
    /// One line for the Solidify section.
    pub status: String,
    /// `SCULPT_PREVIEW_LOG`: frame intervals and preview build times, printed once a second.
    log: Option<Log>,
}

struct Log {
    last_frame: Instant,
    window: Instant,
    frames: Vec<f32>,
    builds: Vec<String>,
}

impl Default for SolidPreview {
    fn default() -> Self {
        SolidPreview { on: std::env::var("SCULPT_SOLID_PREVIEW").as_deref() != Ok("off"), doc: None, job: None, started: Instant::now(), last_since: Duration::ZERO, signature: 0, dirty: true, hold_live: false, shown_coarse: false, stroked: false, last_dab_age: Duration::ZERO, status: String::new(), log: std::env::var_os("SCULPT_PREVIEW_LOG").map(|_| Log { last_frame: Instant::now(), window: Instant::now(), frames: Vec::new(), builds: Vec::new() }) }
    }
}

impl SolidPreview {
    /// True while the viewport is showing the preview instead of the live mesh.
    pub fn shown(&self) -> bool {
        self.on && self.doc.is_some() && !self.hold_live
    }

    /// Forget the current preview (the document it was made from is gone).
    pub fn reset(&mut self) {
        self.doc = None;
        self.dirty = true;
        self.status.clear();
    }
}

/// A fingerprint of everything in the layer stack that changes what a solidify would produce, apart from sculpt strokes
/// (those are seen through the app's edit clock).
fn signature(doc: &Document) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in doc.layers() {
        // The cheap-to-copy part of a layer holds everything the stack depends on: strength, visibility, the generator.
        format!("{:?}", l.meta()).hash(&mut h);
        l.effective_scale().to_bits().hash(&mut h);
    }
    h.finish()
}

/// Called once a frame: collect a finished preview, notice edits, and start the next preview when editing has paused.
pub fn tick(app: &mut SculptApp, ctx: &egui::Context) {
    let since = app.last_edit_age().min(app.gen_timer.since_edit());
    let stroking = app.is_stroking() || app.last_edit_age() < AFTER_STROKE;
    let divisions = app.solid_ui.divisions;
    let Some(doc) = app.doc.as_ref() else { return };
    let dab_age = app.last_edit_age();
    let p = &mut app.solid_preview;
    p.hold_live = stroking;
    if dab_age < p.last_dab_age {
        p.stroked = true;
    }
    p.last_dab_age = dab_age;
    if let Some(log) = p.log.as_mut() {
        log.frames.push(log.last_frame.elapsed().as_secs_f32() * 1e3);
        log.last_frame = Instant::now();
        if log.window.elapsed() >= Duration::from_secs(1) {
            let mut f = std::mem::take(&mut log.frames);
            f.sort_by(f32::total_cmp);
            let pick = |q: f32| f.get(((f.len() as f32 - 1.0) * q) as usize).copied().unwrap_or(0.0);
            let builds: Vec<String> = std::mem::take(&mut log.builds);
            eprintln!("preview-log: frames {} median {:.1} ms p95 {:.1} max {:.1}; builds [{}]", f.len(), pick(0.5), pick(0.95), pick(1.0), builds.join(" "));
            log.window = Instant::now();
        }
    }
    if !p.on {
        if p.doc.is_some() {
            p.reset();
        }
        return;
    }
    if p.job.as_ref().is_some_and(|j| j.is_finished()) {
        match p.job.take().unwrap().join() {
            Ok(Ok((preview, stats, ms, coarse))) => {
                p.status = format!("{} layers, {} quads, {ms:.0} ms{}", stats.layers, stats.quads, if coarse { " (drag detail)" } else { "" });
                p.doc = Some(preview);
                p.shown_coarse = coarse;
                if let Some(log) = p.log.as_mut() {
                    log.builds.push(format!(
                        "{ms:.0}{} (prep {:.0} scatter {:.0} extract {:.0}, {} quads, {} new distances, {} samples)",
                        if coarse { "c" } else { "" },
                        stats.prep_ms,
                        stats.scatter_ms,
                        stats.extract_ms,
                        stats.quads,
                        stats.base_misses,
                        stats.evaluated
                    ));
                }
            }
            Ok(Err(e)) => p.status = format!("preview failed: {e}"),
            Err(_) => p.status = "preview crashed".into(),
        }
    }
    // A new edit shows as the edit clock restarting; layer changes show in the signature.
    let sig = signature(doc);
    if since < p.last_since || sig != p.signature {
        p.dirty = true;
    }
    p.last_since = since;
    p.signature = sig;

    let has_layers = doc.layers().iter().any(|l| l.generator.is_some());
    if !has_layers {
        // Nothing to solidify: show the live mesh.
        p.doc = None;
        p.dirty = false;
        p.status = "no generator layers".into();
        return;
    }
    // Strokes change the surface every dab; wait for the stroke to end rather than snapshot the mesh at each one.
    let dragging = since < DRAGGING;
    if p.job.is_none() && !stroking && (p.dirty || (p.shown_coarse && !dragging)) {
        // After a stroke much of the surface's distances are recomputed: show a quick result first.
        let coarse = dragging || p.stroked;
        p.stroked = false;
        let divisions = if coarse { (divisions / 3).max(32) } else { divisions };
        let input = doc.solid_input();
        let settings = SolidSettings { voxel: input.surface_bounds().extent().max_element() / divisions as f32 };
        p.dirty = false;
        p.started = Instant::now();
        p.job = Some(std::thread::spawn(move || {
            let work = || {
                let t = Instant::now();
                let (mesh, stats) = input.run(&settings).map_err(|e| e.to_string())?;
                let preview = Document::from_mesh(mesh).map_err(|e| e.to_string())?;
                Ok((preview, stats, t.elapsed().as_secs_f32() * 1e3, coarse))
            };
            // Always on the preview's own threads: the next brush stroke or slider drag can start at any moment, and
            // must find the rest of the machine free.
            pool().install(work)
        }));
        p.status = "updating...".into();
    }
    if p.job.is_some() || p.dirty || p.shown_coarse {
        ctx.request_repaint_after(Duration::from_millis(40));
    }
}

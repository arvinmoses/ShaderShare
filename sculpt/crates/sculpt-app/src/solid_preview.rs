//! Live preview of the solidified stack.
//!
//! The viewport shows what Solidify would make whenever the stack has generator layers: one clean closed mesh with soft
//! blends, instead of the live displacement, which cannot make a shape stand out of the surface (added spheres come out
//! jammed against it). It is on by default. The live layers stay what you edit and the preview follows them.
//!
//! Edits never wait for it. The preview is built on a worker thread with its own small pool from a snapshot of the stack;
//! a new one starts as soon as the last has finished and something changed, so while a slider is dragged the solid
//! updates several times a second, and the last finished result stays on screen until the next replaces it. While a
//! sculpt stroke is under way the live mesh is shown instead, so the brush is never drawn on a stale surface.

use std::hash::{Hash, Hasher};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sculpt_core::Document;
use sculpt_core::document::{SolidSettings, SolidStats};

use crate::app::SculptApp;

/// How long after the last brush dab the live mesh stays on screen, so a stroke is seen as it is made.
const AFTER_STROKE: Duration = Duration::from_millis(300);

/// The preview's own worker threads, a third of the machine, so a preview in flight never competes with a live edit for the
/// shared pool when the hand starts moving again.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 3).max(2);
        rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("solid-preview-{i}")).build().expect("preview thread pool")
    })
}

type Finished = Result<(Document, SolidStats, f32), String>;

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
    /// One line for the Solidify section.
    pub status: String,
}

impl Default for SolidPreview {
    fn default() -> Self {
        SolidPreview { on: true, doc: None, job: None, started: Instant::now(), last_since: Duration::ZERO, signature: 0, dirty: true, hold_live: false, status: String::new() }
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
    let p = &mut app.solid_preview;
    p.hold_live = stroking;
    if !p.on {
        if p.doc.is_some() {
            p.reset();
        }
        return;
    }
    if p.job.as_ref().is_some_and(|j| j.is_finished()) {
        match p.job.take().unwrap().join() {
            Ok(Ok((preview, stats, ms))) => {
                p.status = format!("{} layers, {} quads, {ms:.0} ms", stats.layers, stats.quads);
                p.doc = Some(preview);
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
    if p.dirty && p.job.is_none() && !stroking {
        let longest = doc.bounds().extent().max_element();
        let settings = SolidSettings { voxel: longest / divisions as f32 };
        let input = doc.solid_input();
        p.dirty = false;
        p.started = Instant::now();
        p.job = Some(std::thread::spawn(move || {
            pool().install(|| {
                let t = Instant::now();
                let (mesh, stats) = input.run(&settings).map_err(|e| e.to_string())?;
                let preview = Document::from_mesh(mesh).map_err(|e| e.to_string())?;
                Ok((preview, stats, t.elapsed().as_secs_f32() * 1e3))
            })
        }));
        p.status = "updating...".into();
    }
    if p.job.is_some() || p.dirty {
        ctx.request_repaint_after(Duration::from_millis(40));
    }
}

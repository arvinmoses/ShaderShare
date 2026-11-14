//! The volume playground: sculpting a solid that never stretches, at whatever detail the view calls for.
//!
//! The solid is a distance field at several resolutions (`sculpt_core::generator::level_volume`): level 0 holds the
//! whole form, each level after it halves the step, and finer levels exist only where something was sculpted at them.
//! A stroke works at the level its brush calls for: a lattice step about a pixel on screen where the brush is, but no
//! finer than a 48th of its radius, so a big brush works on coarse form (and the finer detail under it rides along)
//! and a small one zoomed in carves detail as fine as the screen shows. The viewport keeps the surface on the GPU
//! (`gpu_levels`), built there from the blocks a stroke changed. Strokes can be undone, and the volume saved and opened
//! (`.svol`). Over the volume: the detail and generator stack (`Layered`), and mesh layers on its surface
//! (`mesh_stack`).

use std::sync::{Arc, OnceLock};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Instant;

use crate::tools::Brush;
use glam::Vec3;
use sculpt_core::generator::detail_layer::{FollowDone, GenDone, Layered};
use sculpt_core::LayerBlend;
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::surface_noise::SurfaceNoise;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::generator::level_volume::{CarryDone, EndDone, Key, LevelVolume};

use crate::layer_ui::{LayerInfo, MaskInfo, MaskKind};
use crate::mesh_stack::{MeshStack, RowsDirty};
use crate::ms;

/// While strokes run ahead of the work still landing: what a frame spends doing them for real is this less what the
/// frame's own dabs took, at least [`CATCH_UP_MIN_MS`]. The copy is what is drawn, so the pen does not notice; the
/// real state only catches up later.
const CATCH_UP_MS: f64 = 14.0;
const CATCH_UP_MIN_MS: f64 = 2.0;
/// The mesh layers' refit after volume strokes starts once the pen has been up this long (while the form is what strokes
/// work on): back-to-back strokes share one, and none runs under the next stroke's dabs. Whatever needs the layers
/// current (a layer stroke, undo, redo, save) refits at once instead.
const REFIT_AFTER_MS: f64 = 250.0;
/// What a refit's swap replaced is freed (if the next refit has not freed it already) once the pen and the swap have
/// both been still this long: freeing it slows a stroke's dabs for a moment.
const FREE_OLD_AFTER_MS: f64 = 1000.0;
/// The least time a frame spends on queued dabs (see [`VolumeMode::run_frame_dabs`]).
const DAB_FRAME_MS: f64 = 9.0;

/// Read once: whether the environment variable `name` is set.
fn env_flag(cell: &'static OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| std::env::var_os(name).is_some())
}

/// `SCULPT_DAB_LOG`: print every dab, as given (to replay a session's strokes).
fn dab_log() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    env_flag(&F, "SCULPT_DAB_LOG")
}

/// `SCULPT_WAIT_LOG`: print how long strokes waited for work landing.
fn wait_log() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    env_flag(&F, "SCULPT_WAIT_LOG")
}

/// `SCULPT_LAYER_LOG`: print the time of each form dab under a detail layer.
fn layer_log() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    env_flag(&F, "SCULPT_LAYER_LOG")
}

/// One dab of a stroke, as the pen gives it.
#[derive(Clone, Copy, Debug)]
pub struct DabInput {
    /// The surface point under the pen.
    pub at: Vec3,
    /// In world units.
    pub radius: f32,
    /// The size of a pixel at the brush: with the radius, it picks the level a volume stroke works at.
    pub px: f32,
    pub strength: f32,
    pub hardness: f32,
    pub brush: Brush,
}

/// What a stroke was given, kept to do it again.
#[derive(Clone, Copy, Debug)]
enum Event {
    /// A stroke begun on this (routed) target, and when.
    Begin(Target, Instant),
    Dab(DabInput),
    End,
}

/// Strokes begun while the last one's work (its end, the layers' following, a generator's job) is still landing.
/// They are drawn at once on a copy of what is shown, and done for real, dab for dab, once that work is in: a few
/// dabs a frame until they have caught up, when the real state is drawn again. What they end with is what they end
/// with after waiting.
struct Ahead {
    /// The real state (what is shown, the layers, the stroke under way's level, whether chosen, its brush normal, its
    /// target, whether it works on a layer); `VolumeMode` holds the copy's meanwhile.
    vol: LevelVolume,
    layer: Option<Layered>,
    level: usize,
    chosen: bool,
    stroke_n: Vec3,
    target: Target,
    on_layer: bool,
    /// Blocks either side changed since the copy was made: drawn again from the real state when it comes back.
    changed: std::collections::HashSet<Key>,
    /// The strokes as given, and how many of their events the real state has done.
    events: Vec<Event>,
    done: usize,
    since: Instant,
}

impl Ahead {
    /// Trade the real state and the copy's.
    fn swap(&mut self, m: &mut VolumeMode) {
        std::mem::swap(&mut self.vol, &mut m.vol);
        std::mem::swap(&mut self.layer, &mut m.layer);
        std::mem::swap(&mut self.level, &mut m.level);
        std::mem::swap(&mut self.chosen, &mut m.chosen);
        std::mem::swap(&mut self.stroke_n, &mut m.stroke_n);
        std::mem::swap(&mut self.target, &mut m.target);
        std::mem::swap(&mut self.on_layer, &mut m.on_layer);
    }
}

/// Levels in a new volume: level 0 and six finer, the finest 64 times finer. Level 0 is coarse, for wide brushes seen
/// from far (a brush of radius 0.7 on the ball of radius 1 is about 70 of its steps); level 1 is the step a ball's
/// triangle count names.
pub const LEVELS: usize = 7;
/// Levels a volume opened from a file has at least: files from before the coarse level 0 have six, and open as they
/// were made (their level 0 is the step their triangle count named).
pub const OPEN_LEVELS: usize = 6;
/// The finest step a brush works at, in pixels on screen at the brush.
const PIXELS_PER_STEP: f32 = 1.0;
/// The most lattice steps across a brush's radius: bigger brushes work a level coarser.
const MAX_STEPS_PER_RADIUS: f32 = 48.0;
/// Finer blocks a carrying job takes at most: one job runs at a time, on its own threads, while the frames go on.
const CARRY_JOB: usize = 384;

/// Level 0's step for a ball of radius 1 whose level 1 alone would be about `triangles` triangles.
pub fn ball_voxel0(triangles: f32) -> f32 {
    2.0 * (4.0 * std::f32::consts::PI / (triangles / 2.0)).sqrt()
}

/// About how many triangles a fresh ball made for `triangles` shows: its level 0's.
pub fn ball_triangles(triangles: f32) -> f32 {
    triangles / 4.0
}

/// The level of `vol` a brush of `radius` (world units) works at, `px` world units being a pixel at the brush: the
/// coarsest whose step is no coarser than a pixel nor than a 48th of the radius, else level 0.
pub fn level_for(vol: &LevelVolume, radius: f32, px: f32) -> usize {
    let want = (px * PIXELS_PER_STEP).max(radius / MAX_STEPS_PER_RADIUS);
    let mut l = 0;
    while l + 1 < vol.levels() && vol.voxel(l + 1) >= want {
        l += 1;
    }
    l
}

/// A slider's change put off till the pointer lets go (a layer too big to move every frame).
enum PendingSlider {
    Opacity(usize, f32),
    Mask(usize, usize, MaskInfo),
}

/// What strokes work on once the volume has a detail layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    /// The form under the layer: the layer's detail follows it.
    Form,
    /// The detail layer: strokes paint on what is shown, and go into the layer when they end.
    Detail,
    /// The layer's mask: hiding the layer where painted, or (`show`) showing it again.
    Mask { show: bool },
    /// The generator layer's painted density: `more` lets rocks in, else keeps them out. The rocks are scattered again
    /// when the stroke ends.
    Density { more: bool },
}

pub struct VolumeMode {
    /// What is drawn (with a detail layer, the form with the layer on it).
    pub vol: LevelVolume,
    /// The detail layer over the form, if any, and what strokes work on.
    pub layer: Option<Layered>,
    pub target: Target,
    /// The stroke under way works on the detail layer (taken into it when it ends).
    on_layer: bool,
    /// A slider's change to a mesh layer too big to move every frame: done when the pointer lets go.
    pending_slider: Option<PendingSlider>,
    /// The detail layer following the form, off the frame (see `Layered::follow_take`), and the end of a stroke on the
    /// form under it.
    follow_job: Option<Receiver<FollowDone>>,
    form_end: Option<Receiver<EndDone>>,
    /// Bumped when `vol` is replaced, so the viewport rebuilds its GPU copy.
    pub generation: u64,
    /// The level the stroke under way works at.
    pub level: usize,
    /// Show levels up to this one (a distant view leaves the finer ones out).
    pub cap: usize,
    /// Choose the level shown region by region, by distance: a block shows while its detail comes to more than
    /// `level_px` pixels on screen.
    pub by_distance: bool,
    pub level_px: f32,
    /// At most this many million triangles drawn (when on): past it, only the detail that shows most.
    pub budget_on: bool,
    pub budget_m: f32,
    /// What that choice last came to.
    pub view: sculpt_core::generator::level_view::ViewStats,
    /// How long the last generator edit took to show (scattering, placing and composing), and whether a drag of one
    /// of its parameters is under way.
    pub generator_ms: f64,
    pub generator_full_ms: f64,
    generator_edit: bool,
    /// A generator being composed at its own level on another thread (its preview shown meanwhile).
    gen_job: Option<Receiver<GenDone>>,
    /// An edit for the GPU to join the generators for, on the next sync.
    pub gpu_plan: Option<sculpt_core::generator::detail_layer::GpuPlan>,
    /// How long the last join on the GPU took to start, on the CPU.
    pub join_ms: f64,
    /// Tint the surface by the level each part comes from.
    pub tint: bool,
    /// Timing of the last frame's dabs, for the status line.
    pub last_ms: f64,
    pub last_dabs: usize,
    pub blocks: usize,
    /// What the viewport's last GPU sync did, and what the last frame drew: pages of quads drawn, of those in use.
    pub sync: crate::gpu_levels::SyncStats,
    pub pages_drawn: u32,
    pub pages_used: u32,
    /// The last stroke's level and time per dab, for the readout.
    pub stroke_level: usize,
    pub dab_ms: f64,
    /// What this frame's dabs took so far (see [`VolumeMode::tick`]).
    frame_dab_ms: f64,
    /// When the pen last went up, and whether it is down.
    pen_up: Instant,
    pen_down: bool,
    /// Dabs given but not done yet: (at, radius, px, strength, hardness, brush).
    queued: std::collections::VecDeque<DabInput>,
    /// GPU time of the last surface rebuild.
    pub gpu_ms: f64,
    /// The brush normal of the stroke's last dab, which the next one leans toward.
    stroke_n: Vec3,
    /// Whether the stroke under way has chosen its level yet.
    chosen: bool,
    /// Carrying detail under a coarse stroke runs here, off the frame, a job at a time.
    pool: Arc<rayon::ThreadPool>,
    /// Generator jobs run here, on as many threads as cores below the frame's priority: in the frame's own pool they
    /// held up the next stroke's dabs until they were done.
    work: Arc<rayon::ThreadPool>,
    job: Option<Receiver<CarryDone>>,
    /// The last carrying job: blocks, and its time on its threads.
    pub carry_blocks: usize,
    pub carry_ms: f64,
    /// The pen is up but detail is still being carried: the stroke ends once it is all in.
    ending: bool,
    /// The stroke's end being worked out off the frame (see `LevelVolume::end_stroke_take`).
    end_job: Option<Receiver<EndDone>>,
    /// How long the last stroke's end took on its threads.
    pub end_ms: f64,
    /// Specks the last stroke to end left, removed, until read.
    ended_specks: Option<usize>,
    /// Stops the generator job under way (when a form stroke makes its result stale).
    gen_cancel: Option<Arc<AtomicBool>>,
    /// Strokes running ahead of work still landing (see [`Ahead`]).
    ahead: Option<Box<Ahead>>,
    /// The last stroke's begin, until its first dab.
    begun: Option<Instant>,
    /// From the last stroke's begin: to its first dab drawn, to its begin for real, and (when it ran ahead) to the
    /// real state catching up.
    pub first_dab_ms: f64,
    pub lag_ms: f64,
    pub catch_up_ms: f64,
    /// Form strokes that took a previewed generator at its own level rather than wait for its job.
    pub adopted: usize,
    /// Mesh layers on the volume's surface (see `mesh_stack`): never part of the copy strokes run ahead on.
    pub mesh: Option<MeshStack>,
    /// The first mesh layer's mesh being taken from the volume on another thread.
    mesh_build: Option<MeshBuild>,
    /// A stroke on a mesh layer or its mask under way.
    mesh_stroke: Option<MeshStroke>,
    /// Mesh layers went away: the rows are told once.
    rows_none: bool,
    /// A note for the status line about the mesh layers (made, waited for, refused), until read.
    pub mesh_note: Option<String>,
    /// What the file it was opened from held that could not be read and was left out (saving it over that file would
    /// lose it for good).
    pub lossy_open: Option<String>,
    /// Ctrl held for this stroke, as the app sees it. The app already swaps the brush for Ctrl (Clay for Trim and
    /// back), so no brush inverts on top of that.
    pub invert: bool,
}

/// The mesh of the first mesh layer, being taken from the volume off the frame.
struct MeshBuild {
    rx: Receiver<Result<sculpt_core::mesh_layers::MeshLayers, String>>,
    /// Volume blocks changed since the snapshot it is taken from.
    dirty: std::collections::HashSet<Key>,
    started: Instant,
}

/// A stroke on a mesh layer (or its painted mask).
struct MeshStroke {
    /// The layer itself (`Detail`) or its mask (`Mask`).
    target: Target,
    /// The last dab's centre and its area's normal.
    last: Option<Vec3>,
    n: Vec3,
    /// The channel a mask stroke paints.
    channel: Option<String>,
    /// A mask stroke shows as it goes while refreshing the masks stays under a frame.
    live: bool,
    painted: bool,
    /// Refused (the layers could not follow the volume): its dabs do nothing and it makes no step.
    refused: bool,
}

impl VolumeMode {
    /// A ball of radius 1 whose level 1 alone would be about `triangles` triangles: its level 0, all a fresh ball
    /// holds, has twice the step, a quarter as many (see [`ball_triangles`]).
    pub fn ball(triangles: f32) -> VolumeMode {
        VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, ball_voxel0(triangles), LEVELS))
    }

    pub fn new(mut vol: LevelVolume) -> VolumeMode {
        let cap = vol.levels() - 1;
        // Dabs only queue the carrying; jobs on the pool do it.
        vol.carry_budget = Some(0.0);
        // A quarter of the cores: the dabs and the sync want the rest, and more would only slow them.
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let threads = (cores / 4).max(2);
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("carry {i}")).build().expect("a thread pool"));
        // Below the frame's threads, so a dab never waits for its share of the cores behind them.
        let work = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(cores).thread_name(|i| format!("work {i}")).start_handler(|_| sculpt_core::lower_priority()).build().expect("a thread pool"));
        VolumeMode {
            vol,
            layer: None,
            target: Target::Detail,
            on_layer: false,
            pending_slider: None,
            follow_job: None,
            form_end: None,
            generation: 1,
            level: 0,
            cap,
            by_distance: true,
            level_px: 1.0,
            budget_on: true,
            budget_m: 6.0,
            generator_ms: 0.0,
            generator_full_ms: 0.0,
            generator_edit: false,
            gen_job: None,
            gpu_plan: None,
            join_ms: 0.0,
            view: Default::default(),
            tint: false,
            last_ms: 0.0,
            last_dabs: 0,
            blocks: 0,
            sync: Default::default(),
            pages_drawn: 0,
            pages_used: 0,
            stroke_level: 0,
            dab_ms: 0.0,
            frame_dab_ms: 0.0,
            pen_up: Instant::now(),
            pen_down: false,
            queued: Default::default(),
            gpu_ms: 0.0,
            stroke_n: Vec3::ZERO,
            chosen: false,
            pool,
            work,
            job: None,
            carry_blocks: 0,
            carry_ms: 0.0,
            ending: false,
            end_job: None,
            end_ms: 0.0,
            ended_specks: None,
            gen_cancel: None,
            ahead: None,
            begun: None,
            first_dab_ms: 0.0,
            lag_ms: 0.0,
            catch_up_ms: 0.0,
            adopted: 0,
            mesh: None,
            mesh_build: None,
            mesh_stroke: None,
            rows_none: false,
            mesh_note: None,
            lossy_open: None,
            invert: false,
        }
    }

    /// Spacing floor for dabs: the step of the level the stroke works at.
    pub fn voxel(&self) -> f32 {
        self.vol.voxel(self.level)
    }

    /// The level a brush of `radius` (world units) works at, `px` world units being a pixel at the brush.
    pub fn level_for(&self, radius: f32, px: f32) -> usize {
        level_for(&self.vol, radius, px)
    }

    /// One dab of Clay Buildup (or, with `carve`, Trim Dynamic) at surface point `at`: a plane brush whose plane
    /// follows the surface the stroke began on. `px` is the size of a pixel at the brush, which (with the radius) picks
    /// the level the stroke works at, on its first dab.
    /// (The app's strokes go through [`Self::brush_dab`]; the tests' clay and trim dabs through here.)
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn dab(&mut self, at: Vec3, radius: f32, px: f32, strength: f32, hardness: f32, carve: bool) {
        self.brush_dab(DabInput { at, radius, px, strength, hardness, brush: if carve { Brush::Trim } else { Brush::Clay } });
    }

    /// One dab of `brush`: Clay Buildup and Trim Dynamic as [`Self::dab`], or Smooth (see `LevelVolume::smooth_dab`),
    /// whose strength is how far each dab moves the surface toward its smoothed self.
    pub fn brush_dab(&mut self, d: DabInput) {
        let DabInput { at, radius, px, strength, hardness, brush } = d;
        // For replaying a session's strokes (`SCULPT_DAB_LOG`): every dab, as given.
        if dab_log() {
            eprintln!("DAB {} {} {} {radius} {px} {strength} {hardness} {}{}", at.x, at.y, at.z, brush == Brush::Trim, if brush == Brush::Smooth { " smooth" } else { "" });
        }
        // On a mesh layer or its mask: nothing of the volume's.
        if self.mesh_stroke.is_some() {
            self.mesh_dab(d);
            if let Some(t) = self.begun.take() {
                self.first_dab_ms = ms(t);
            }
            return;
        }
        // Running ahead: kept for the real state, and drawn on the copy.
        if let Some(a) = self.ahead.as_mut() {
            a.events.push(Event::Dab(d));
        }
        self.dab_now(d);
        self.note_copy();
        if let Some(t) = self.begun.take() {
            self.first_dab_ms = ms(t);
            if wait_log() {
                eprintln!("first dab {:.1} ms after the stroke began{}", self.first_dab_ms, if self.ahead.is_some() { " (ahead of work still landing)" } else { "" });
            }
        }
    }

    /// A dab for [`Self::run_dabs`] to do: the app's strokes queue theirs, so a fast drag's many dabs spread over
    /// frames instead of holding one.
    pub fn queue_dab(&mut self, d: DabInput) {
        self.queued.push_back(d);
    }

    /// Whether dabs are queued.
    pub fn dabs_queued(&self) -> bool {
        !self.queued.is_empty()
    }

    /// Queued dabs, in order, for about `budget_ms` (at least one). Returns how many were done.
    pub fn run_dabs(&mut self, budget_ms: f64) -> usize {
        let t = Instant::now();
        let mut n = 0;
        while let Some(d) = self.queued.pop_front() {
            self.brush_dab(d);
            n += 1;
            if ms(t) > budget_ms {
                break;
            }
        }
        n
    }

    /// One frame's share of queued dabs: `DAB_FRAME_MS` of them, or half the time all would take if that is more,
    /// so a burst spreads over a few frames and the stroke never trails the pen by much more than two frames' work.
    pub fn run_frame_dabs(&mut self) -> usize {
        let t = Instant::now();
        let all = self.queued.len() as f64 * self.dab_ms.max(0.05);
        let n = self.run_dabs(DAB_FRAME_MS.max(0.5 * all));
        self.frame_dab_ms += ms(t);
        n
    }

    /// Blocks the copy changed, while running ahead (see [`Ahead`]).
    fn note_copy(&mut self) {
        if let Some(a) = self.ahead.as_mut() {
            a.changed.extend(self.vol.dirty_keys());
        }
    }

    /// A dab on what `vol` and `layer` hold now: the real state, or while running ahead the copy.
    fn dab_now(&mut self, d: DabInput) {
        let DabInput { at, radius, px, strength, hardness, brush } = d;
        let carve = brush == Brush::Trim;
        let t = Instant::now();
        let target = self.layer.is_some().then(|| self.routed());
        // Smoothing a mask or the painted density is not a thing yet: Shift there does nothing.
        if brush == Brush::Smooth && matches!(target, Some(Target::Mask { .. } | Target::Density { .. })) {
            return;
        }
        if let (Some(Target::Mask { show }), Some(layer)) = (target, self.layer.as_mut()) {
            layer.paint_mask(&mut self.vol, at, radius, if show { 1.0 } else { 0.0 });
            return;
        }
        if let (Some(Target::Density { more }), Some(layer)) = (target, self.layer.as_mut()) {
            layer.paint_density(at, radius, if more { 1.0 } else { 0.0 });
            return;
        }
        if !self.chosen {
            self.level = self.level_for(radius, px);
            // The form is the levels under the layer's; the layer, its own.
            match (target, self.layer.as_mut()) {
                (Some(Target::Form), Some(layer)) => self.level = self.level.min(layer.from() - 1),
                // A layer's own resolution follows the work: a stroke finer than it raises it first.
                (Some(Target::Detail), Some(layer)) => {
                    self.level = self.level.max(layer.from());
                    if self.level > layer.level() {
                        let k = layer.active();
                        layer.raise_level(&mut self.vol, k, self.level);
                    }
                    self.level = self.level.min(layer.level());
                }
                _ => {}
            }
            self.chosen = true;
        }
        let form = target == Some(Target::Form);
        let along = if self.stroke_n != Vec3::ZERO {
            self.stroke_n
        } else if let (true, Some(layer)) = (form, self.layer.as_ref()) {
            layer.form().normal_upto(at, self.level, true)
        } else {
            self.vol.normal_upto(at, self.level, true)
        };
        if along == Vec3::ZERO {
            return;
        }
        let shape = brush.shape(strength);
        let (height, alpha) = (radius * shape.height, shape.alpha);
        let dabbed = match (form, self.layer.as_mut()) {
            (true, Some(layer)) if brush == Brush::Smooth => layer.smooth_dab(&mut self.vol, self.level, at, along, radius, hardness, alpha),
            (_, _) if brush == Brush::Smooth => self.vol.smooth_dab(self.level, at, along, radius, hardness, alpha),
            (true, Some(layer)) => {
                let out = layer.plane_dab(&mut self.vol, self.level, at, along, radius, height, hardness, alpha, carve);
                if layer_log() {
                    eprintln!("form dab at level {} r {radius:.3}: {:.1} ms; layer {} blocks; follow {:?}", self.level, ms(t), layer.blocks(), layer.last);
                }
                out
            }
            _ => self.vol.plane_dab(self.level, at, along, radius, height, hardness, alpha, carve),
        };
        let Some((n, _)) = dabbed else { return };
        // The original surface's plane is steady already; easing it a little more smooths the turn over sharp forms.
        self.stroke_n = if self.stroke_n == Vec3::ZERO { n } else { (self.stroke_n * 0.5 + n * 0.5).try_normalize().unwrap_or(n) };
        let ms = ms(t);
        self.last_ms += ms;
        self.last_dabs += 1;
        self.stroke_level = self.level;
        self.dab_ms = if self.dab_ms == 0.0 { ms } else { self.dab_ms * 0.8 + ms * 0.2 };
    }

    /// Once a frame: bring in a carrying job that finished, and start the next on what is queued (see
    /// `LevelVolume::carry_take`); while strokes run ahead, the real state's work, and as much of the strokes for real
    /// as a frame allows.
    pub fn tick(&mut self) {
        if self.ahead.is_none() {
            self.tick_now();
        } else {
            self.with_real(|m| m.tick_now());
            // A frame that ran dabs of its own does no catch-up dab it has no room for (a wide brush's dab is most
            // of a frame); the frames without (the pen resting, between dabs, up) do at least one.
            let budget = (CATCH_UP_MS - self.frame_dab_ms).max(CATCH_UP_MIN_MS);
            self.catch_up_with(Some(budget), false, self.frame_dab_ms > 0.0);
        }
        self.frame_dab_ms = 0.0;
        self.mesh_tick();
    }

    /// While the layers' last refit failed, why (shown beside `mesh_note` until a refit succeeds).
    pub fn mesh_failure(&self) -> Option<String> {
        self.mesh.as_ref()?.failure.as_ref().map(|e| format!("Layers could not follow the volume: {e}"))
    }

    /// `f` on the real state (while running ahead, the copy's put aside meanwhile).
    fn with_real<R>(&mut self, f: impl FnOnce(&mut VolumeMode) -> R) -> R {
        let Some(mut a) = self.ahead.take() else { return f(self) };
        a.swap(self);
        let out = f(self);
        a.swap(self);
        // As the viewport would have taken them, a frame at a time (the mesh layers told first).
        let dirty = a.vol.drain_dirty();
        self.tap_keys(&dirty);
        a.changed.extend(dirty);
        self.ahead = Some(a);
        out
    }

    /// Begin running ahead: the copy is drawn from here on.
    fn run_ahead(&mut self) {
        let mut copy = self.vol.preview_copy();
        // Its dabs only queue carrying: nothing runs it.
        copy.carry_budget = Some(0.0);
        // What the GPU has not been sent yet, sent from the copy (which holds the same).
        copy.mark_dirty(self.vol.dirty_keys());
        let real = std::mem::replace(&mut self.vol, copy);
        let layer = self.layer.as_ref().map(|l| l.preview_copy());
        let real_layer = std::mem::replace(&mut self.layer, layer);
        self.ahead = Some(Box::new(Ahead {
            vol: real,
            layer: real_layer,
            level: self.level,
            chosen: self.chosen,
            stroke_n: self.stroke_n,
            target: self.target,
            on_layer: self.on_layer,
            changed: Default::default(),
            events: Vec::new(),
            done: 0,
            since: Instant::now(),
        }));
    }

    /// Do the strokes run ahead for real, as far as the work they must come after allows, for at most `budget_ms`
    /// (at least one event); once all are done, draw the real state again.
    fn catch_up(&mut self, budget_ms: Option<f64>) -> bool {
        self.catch_up_with(budget_ms, false, false)
    }

    /// [`Self::catch_up`], the next begin going ahead whatever is still landing.
    fn catch_up_forced(&mut self) -> bool {
        debug_assert!(false, "a stroke run ahead still waits once everything has landed");
        self.catch_up_with(None, true, false)
    }

    /// Returns whether any event was done. `fit`: not even a first dab unless about one dab's time is left of the
    /// budget.
    fn catch_up_with(&mut self, budget_ms: Option<f64>, force: bool, fit: bool) -> bool {
        let t = Instant::now();
        let mut any = false;
        loop {
            let Some(event) = self.ahead.as_ref().and_then(|a| a.events.get(a.done).copied()) else { break };
            let first = !any;
            if let (true, Some(b), Event::Dab { .. }) = (fit, budget_ms, event)
                && ms(t) + self.dab_ms > b
            {
                break;
            }
            let done = self.with_real(|m| match event {
                Event::Begin(target, at) => {
                    if m.waiting_for(target) && !(force && first) {
                        return false;
                    }
                    m.target = target;
                    m.begin_now(true);
                    m.lag_ms = ms(at);
                    true
                }
                Event::Dab(d) => {
                    m.dab_now(d);
                    true
                }
                Event::End => {
                    m.end_now();
                    true
                }
            });
            if !done {
                break;
            }
            any = true;
            self.ahead.as_mut().unwrap().done += 1;
            if budget_ms.is_some_and(|b| ms(t) > b) {
                break;
            }
        }
        if self.ahead.as_ref().is_some_and(|a| a.done == a.events.len()) {
            self.come_back();
            any = true;
        }
        any
    }

    /// The strokes run ahead are all done for real: draw the real state again, every block either side changed sent.
    fn come_back(&mut self) {
        let Some(mut a) = self.ahead.take() else { return };
        let mut changed = std::mem::take(&mut a.changed);
        changed.extend(self.vol.dirty_keys());
        let target = self.target;
        a.swap(self);
        // The pen's target stays the one chosen.
        self.target = target;
        self.vol.mark_dirty(changed);
        self.catch_up_ms = ms(a.since);
        if wait_log() {
            eprintln!("caught up {:.0} ms after running ahead ({} events)", self.catch_up_ms, a.events.len());
        }
    }

    /// Whether a stroke on `target` must come after work still landing.
    fn waiting_for(&self, target: Target) -> bool {
        let layer = self.layer.as_ref();
        // A form stroke takes a generator still previewed at its own level instead (see `Layered::adopt_generators`).
        let generators = self.gen_job.is_some() || layer.is_some_and(|l| l.previewing());
        self.ending
            || self.carrying()
            || self.end_job.is_some()
            || self.follow_job.is_some()
            || self.form_end.is_some()
            || layer.is_some_and(|l| l.busy())
            || (generators && target != Target::Form)
    }

    fn tick_now(&mut self) {
        self.generator_jobs(false);
        let t = Instant::now();
        // The detail layer following the form: bring in a job that finished, start the next on what dabs queued.
        if let Some(layer) = self.layer.as_mut() {
            if let Some(rx) = &self.form_end {
                match rx.try_recv() {
                    Ok(done) => {
                        let end = layer.end_form_apply(done);
                        self.end_ms = end.ms;
                        self.ended_specks = Some(end.specks);
                        self.form_end = None;
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => self.form_end = None,
                }
            }
            if let Some(rx) = &self.follow_job {
                match rx.try_recv() {
                    Ok(done) => {
                        layer.follow_apply(&mut self.vol, done);
                        self.follow_job = None;
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => self.follow_job = None,
                }
            }
            // (A stroke's last also works out what is shown after it, so the frame only takes that.)
            if self.follow_job.is_none()
                && let Some(job) = layer.follow_take_landing(&self.vol)
            {
                // On the carrying threads (a quarter of the cores), so the dabs and the sync keep the rest: the
                // stroke's last too, as the next stroke's dabs may run beside it (on all the cores it slowed them
                // several times over, and finished hardly sooner). But a last follow composes every generator's
                // blocks again, which wants every core: with generators, below the frame's priority on all of them.
                let (tx, rx) = channel();
                let pool = if job.last() && layer.layers().iter().any(|x| x.generator) { &self.work } else { &self.pool };
                pool.spawn(move || {
                    let _ = tx.send(job.run());
                });
                self.follow_job = Some(rx);
            }
        }
        if let Some(rx) = &self.job {
            match rx.try_recv() {
                Ok(done) => {
                    self.carry_blocks = done.blocks;
                    self.carry_ms = done.ms;
                    self.vol.carry_apply(done);
                    self.job = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.job = None,
            }
        }
        // Once the pen is up, all that is left in one job (the stroke's end, and the next stroke, come after it).
        let most = if self.ending { usize::MAX } else { CARRY_JOB };
        if self.job.is_none()
            && let Some(job) = self.vol.carry_take(most)
        {
            let (tx, rx) = channel();
            self.pool.spawn(move || {
                let _ = tx.send(job.run());
            });
            self.job = Some(rx);
        }
        if self.ending && !self.carrying() {
            match &self.end_job {
                None => self.start_end(),
                Some(rx) => match rx.try_recv() {
                    Ok(done) => self.take_end(done),
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => {
                        self.end_job = None;
                        self.ending = false;
                    }
                },
            }
        }
        self.last_ms += ms(t);
    }

    /// Work out the stroke's end off the frame, on the carrying threads (the next stroke's dabs may run beside it).
    fn start_end(&mut self) {
        match self.vol.end_stroke_take() {
            Some(job) => {
                let (tx, rx) = channel();
                self.pool.spawn(move || {
                    let _ = tx.send(job.run());
                });
                self.end_job = Some(rx);
            }
            None => self.ending = false,
        }
    }

    fn take_end(&mut self, done: EndDone) {
        let was = self.vol.last_id();
        let end = self.vol.end_stroke_apply(done);
        // (Always on the real state: the copy run ahead on never ends a stroke.)
        if let (Some(m), Some(id)) = (self.mesh.as_mut(), self.vol.last_id())
            && Some(id) != was
        {
            m.note_step(id);
        }
        if std::mem::take(&mut self.on_layer)
            && let Some(layer) = self.layer.as_mut()
        {
            layer.take_detail(&mut self.vol);
        }
        self.end_ms = end.ms;
        self.ended_specks = Some(end.specks);
        self.end_job = None;
        self.ending = false;
    }

    /// Finish everything still being worked out, now (strokes run ahead done for real, a stroke's end, the layers'
    /// following, generator jobs): before anything that needs the volume settled (undo, saving, editing the layers).
    pub fn settle(&mut self) {
        self.close_slider_edit();
        while self.ahead.is_some() {
            self.with_real(|m| m.settle_now());
            // (Everything has landed: a begin that would still wait is a bug, and goes ahead rather than hang.)
            if !self.catch_up(None) {
                self.catch_up_forced();
            }
        }
        self.settle_now();
    }

    fn settle_now(&mut self) {
        self.generator_jobs(true);
        if let Some(layer) = self.layer.as_mut() {
            if let Some(rx) = self.form_end.take()
                && let Ok(done) = rx.recv()
            {
                let end = layer.end_form_apply(done);
                self.end_ms = end.ms;
                self.ended_specks = Some(end.specks);
            }
            if let Some(rx) = self.follow_job.take()
                && let Ok(done) = rx.recv()
            {
                layer.follow_apply(&mut self.vol, done);
            }
            layer.settle(&mut self.vol);
        }
        if !self.ending {
            return;
        }
        self.finish_job();
        if self.end_job.is_none() {
            self.start_end();
        }
        if let Some(rx) = self.end_job.take() {
            match rx.recv() {
                Ok(done) => self.take_end(done),
                Err(_) => self.ending = false,
            }
        }
    }

    /// Whether frames should keep coming to finish carrying or ending a stroke.
    pub fn busy(&self) -> bool {
        self.mesh.as_ref().is_some_and(|m| m.pending())
            || self.mesh_build.is_some()
            || self.ahead.is_some()
            || self.ending
            || self.carrying()
            || self.follow_job.is_some() || self.form_end.is_some() || self.gen_job.is_some() || self.layer.as_ref().is_some_and(|l| l.busy())
    }

    /// The specks the last stroke to end left, removed, once.
    pub fn take_specks(&mut self) -> Option<usize> {
        self.ended_specks.take()
    }

    /// Wait for the carrying job under way, if any, and bring it in.
    fn finish_job(&mut self) {
        if let Some(rx) = self.job.take()
            && let Ok(done) = rx.recv()
        {
            self.vol.carry_apply(done);
        }
    }

    /// Whether detail is still being carried (queued, or a job under way).
    pub fn carrying(&self) -> bool {
        self.job.is_some() || self.vol.carry_busy()
    }

    /// Begin a stroke. Its first dab shows at once: if work it must come after is still landing (the last stroke's
    /// end, the layers' following, a generator's job), it runs ahead on a copy until that work is in (see [`Ahead`]).
    pub fn begin_stroke(&mut self) {
        self.close_slider_edit();
        while self.run_dabs(f64::INFINITY) > 0 {}
        if dab_log() {
            eprintln!("BEGIN");
        }
        let now = Instant::now();
        self.begun = Some(now);
        self.pen_down = true;
        if let Some(m) = self.mesh.as_mut() {
            m.forget_skip();
        }
        if self.on_mesh() && self.mesh_begin() {
            return;
        }
        let target = self.routed();
        if self.ahead.is_none() && self.waiting_for(target) {
            self.run_ahead();
        }
        match self.ahead.as_mut() {
            Some(a) => {
                a.events.push(Event::Begin(target, now));
                self.begin_now(false);
                self.note_copy();
            }
            None => {
                self.begin_now(true);
                self.lag_ms = 0.0;
            }
        }
    }

    /// Begin a stroke on what `vol` and `layer` hold now: the real state, or (not `real`) the copy run ahead on.
    fn begin_now(&mut self, real: bool) {
        let target = self.routed();
        match (self.layer.as_mut(), target) {
            (Some(layer), Target::Form) => {
                // A generator still previewed: held at its own level now, its job's result stale (and stopped).
                if real && layer.previewing() {
                    self.adopted += 1;
                    layer.adopt_generators();
                    self.gen_job = None;
                    if let Some(c) = self.gen_cancel.take() {
                        c.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                layer.begin_stroke();
            }
            (Some(layer), Target::Mask { .. } | Target::Density { .. }) => layer.begin_edit(),
            (layer, _) => {
                self.on_layer = layer.is_some();
                self.vol.begin_stroke();
            }
        }
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
    }

    /// The pen is up: the stroke ends off the frame, once the detail it queued is all carried, frames going on
    /// meanwhile (see [`Self::take_specks`]).
    pub fn end_stroke(&mut self) {
        // Every dab given is done first.
        while self.run_dabs(f64::INFINITY) > 0 {}
        self.pen_up = Instant::now();
        self.pen_down = false;
        if self.mesh_stroke.is_some() {
            self.mesh_end();
            return;
        }
        if let Some(a) = self.ahead.as_mut() {
            // Ended for real once its dabs are; the copy's stroke is simply left.
            a.events.push(Event::End);
            self.stroke_n = Vec3::ZERO;
            self.chosen = false;
            return;
        }
        self.end_now();
    }

    fn end_now(&mut self) {
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
        let target = self.routed();
        match (self.layer.as_mut(), target) {
            (Some(layer), Target::Form) => {
                // The form's end off the frame too; the layer's last follow comes after it (see `tick`).
                if layer.form().in_stroke()
                    && let Some(job) = layer.end_form_take()
                {
                    let (tx, rx) = channel();
                    self.pool.spawn(move || {
                        let _ = tx.send(job.run());
                    });
                    self.form_end = Some(rx);
                }
            }
            (Some(layer), Target::Mask { .. }) => layer.end_edit(),
            (Some(layer), Target::Density { .. }) => {
                // The rocks scatter again for what was painted (a preview at once, its own level from a job).
                layer.end_density_stroke(&mut self.vol);
                self.start_generator_job();
            }
            _ => {
                if self.vol.in_stroke() {
                    self.ending = true;
                }
            }
        }
    }

    /// What strokes work on: with a generator layer selected, its mask, or else the form under it (a generator holds
    /// no sculpting of its own).
    pub fn routed(&self) -> Target {
        match (self.target, self.layer.as_ref()) {
            (Target::Detail, Some(layer)) if layer.generator(layer.active()).is_some() || layer.noise(layer.active()).is_some() => Target::Form,
            (t, _) => t,
        }
    }

    /// A new generator layer on top (rocks scattered over the form, to start with), made active. With no layers yet,
    /// the stack is made first, the finest level's work becoming a detail layer.
    pub fn add_generator(&mut self) {
        if self.refuse_with_mesh("generator") {
            return;
        }
        self.settle();
        let top = self.vol.levels() - 1;
        if self.layer.is_none() {
            self.make_layer(top);
        }
        let Some(layer) = self.layer.as_mut() else { return };
        let n = layer.layers().iter().filter(|x| x.generator).count();
        // Rocks a few hundredths of the object across, at the coarsest level giving them about a dozen steps a radius.
        let (lo, hi) = self.vol.bounds();
        let size = 0.03 * (hi - lo).max_element();
        let level = (0..=top).find(|&l| self.vol.voxel(l) <= size / 12.0).unwrap_or(top);
        let spec = GeneratorSpec {
            primitive: Primitive::Sphere,
            op: BoolOp::Add,
            count: 12,
            seed: 1 + n as u64,
            size,
            lift: 0.15,
            random_spin: true,
            min_spacing: 3.0 * size,
            size_var: 0.4,
            stretch_var: 0.35,
            tilt: 25.0,
            blend: 0.3,
            // Plain primitives to start with: Roughness turns them to stones when wanted.
            rough: 0.0,
            ..GeneratorSpec::default()
        };
        let t = Instant::now();
        layer.add_generator(&mut self.vol, &format!("Generator {}", n + 1), level, &spec);
        self.generator_ms = ms(t);
        self.target = Target::Detail;
    }

    /// A new surface noise layer on top (drawn by the GPU over the whole surface), made active; with no layers yet,
    /// the stack is made first.
    pub fn add_noise(&mut self) {
        let (lo, hi) = self.vol.bounds();
        let extent = (hi - lo).max_element().max(1e-3);
        self.add_noise_with(SurfaceNoise { on: true, amplitude: 0.006 * extent, scale: 6.0 / extent, octaves: 6, kind: 0, seed: 1 });
    }

    pub fn add_noise_with(&mut self, n: SurfaceNoise) {
        if self.refuse_with_mesh("surface noise") {
            return;
        }
        self.settle();
        if self.layer.is_none() {
            self.make_layer(self.vol.levels() - 1);
        }
        if let Some(layer) = self.layer.as_mut() {
            let count = layer.layers().iter().filter(|x| x.noise).count();
            layer.add_noise(&format!("Surface noise {}", count + 1), n);
            self.target = Target::Detail;
        }
    }

    /// Change surface noise layer `k`'s settings; while `dragging`, the changes undo as one.
    pub fn set_noise(&mut self, k: usize, n: SurfaceNoise, dragging: bool) {
        // (Strokes run ahead are done for real first: the change goes into the real layers.)
        if self.ahead.is_some() {
            self.settle();
        }
        let Some(layer) = self.layer.as_mut() else { return };
        if dragging && !self.generator_edit {
            layer.begin_edit();
            self.generator_edit = true;
        }
        layer.set_noise(k, n);
        if !dragging && std::mem::take(&mut self.generator_edit) {
            layer.end_edit();
        }
    }

    /// Bake surface noise layer `k` into a new detail layer at `level` (see `Layered::bake_noise`).
    pub fn bake_noise(&mut self, k: usize, level: usize) {
        self.settle();
        if let Some(layer) = self.layer.as_mut()
            && let Some(at) = layer.bake_noise(&mut self.vol, k, level)
        {
            layer.set_active(at);
            self.target = Target::Detail;
        }
    }

    /// The surface noise shown, if any: its settings, its mask's blocks, and when either last changed.
    pub fn shown_noise(&self) -> Option<(SurfaceNoise, &sculpt_core::generator::level_volume::Shared<glam::IVec3, sculpt_core::generator::level_volume::Stored>, u64)> {
        let layer = self.layer.as_ref()?;
        layer.shown_noise().map(|(n, m)| (n, m, layer.noise_epoch()))
    }

    /// Change generator layer `k`'s parameters. While `dragging`, the changes undo as one, until a change that is not.
    pub fn set_generator(&mut self, k: usize, spec: &GeneratorSpec, dragging: bool) {
        // Mid-drag, the job composing an earlier frame's rocks at their own level is stale: stopped, not waited on
        // (no job starts again until the drag is over: see `start_generator_job`).
        if self.generator_edit {
            self.cancel_generator_job();
        }
        self.settle();
        let Some(layer) = self.layer.as_mut() else { return };
        if dragging && !self.generator_edit {
            layer.begin_edit();
            self.generator_edit = true;
        }
        // Shown no finer than level 1 at once (a few milliseconds); at its own level from a job once the drag is over,
        // the frames going on meanwhile.
        let t = Instant::now();
        // At its own level on the GPU where the stack lets it; else no finer than level 1 here.
        match layer.gpu_generator(&mut self.vol, k, spec) {
            Some(plan) => self.gpu_plan = Some(plan),
            None => layer.preview_generator(&mut self.vol, k, spec),
        }
        self.generator_ms = ms(t);
        if !dragging {
            self.end_generator_edit();
        }
    }

    /// Start composing a previewed generator at its own level on another thread, unless one is under way.
    fn start_generator_job(&mut self) {
        // While a drag goes on, each frame's change would make it stale: it starts once the drag is over.
        if self.gen_job.is_some() || self.generator_edit {
            return;
        }
        // (It also works out what is shown once its result goes in, so the frame only takes that.)
        let vol = &self.vol;
        let Some(job) = self.layer.as_mut().and_then(|l| l.generator_take_landing(vol)) else { return };
        self.gen_cancel = Some(job.canceller());
        let (tx, rx) = channel();
        let t = Instant::now();
        self.work.spawn(move || {
            let mut done = job.run();
            done.ms = ms(t);
            let _ = tx.send(done);
        });
        self.gen_job = Some(rx);
    }

    /// Stop the generator job under way, if any (its result is never taken).
    fn cancel_generator_job(&mut self) {
        self.gen_job = None;
        if let Some(c) = self.gen_cancel.take() {
            c.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Take a generator job that finished (and start the next, if a newer edit wants one); with `wait`, the one under
    /// way and every one still wanted, now.
    fn generator_jobs(&mut self, wait: bool) {
        loop {
            self.start_generator_job();
            let Some(rx) = self.gen_job.as_ref() else { return };
            let done = if wait { rx.recv().ok() } else { rx.try_recv().ok() };
            let Some(done) = done else {
                if wait {
                    self.gen_job = None;
                }
                return;
            };
            self.gen_job = None;
            if let Some(layer) = self.layer.as_mut() {
                let ms = done.ms;
                if layer.generator_apply(&mut self.vol, done) {
                    self.generator_full_ms = ms;
                }
            }
            if !wait {
                // The next, if any, starts on the next frame's tick.
                self.start_generator_job();
                return;
            }
        }
    }

    /// A drag of a generator's parameter is over: shown at its own level again, one undo step for all of it.
    pub fn end_generator_edit(&mut self) {
        if self.ahead.is_some() {
            self.settle();
        }
        if std::mem::take(&mut self.generator_edit)
            && let Some(layer) = self.layer.as_mut()
        {
            layer.end_edit();
        }
        self.start_generator_job();
    }

    /// Move the work at levels `from` and finer into a detail layer over the rest (the form).
    pub fn make_layer(&mut self, from: usize) {
        if self.refuse_with_mesh("detail") {
            return;
        }
        self.settle();
        let from = from.clamp(1, self.vol.levels() - 1);
        let mut layer = Layered::new(&mut self.vol, from);
        // Dabs on the form only queue the layer's following: jobs off the frame do it.
        layer.deferred = true;
        self.layer = Some(layer);
        self.target = Target::Detail;
    }

    /// The mesh layers' rows, if any: the active one, and each (with a slider's change put off shown as done).
    pub fn layer_rows(&self) -> Option<(usize, Vec<LayerInfo>)> {
        let mut out = self.mesh.as_ref()?.infos()?;
        match &self.pending_slider {
            Some(PendingSlider::Opacity(k, o)) => {
                if let Some(x) = out.1.get_mut(*k) {
                    x.opacity = *o;
                }
            }
            Some(PendingSlider::Mask(k, mi, info)) => {
                if let Some(m) = out.1.get_mut(*k).and_then(|x| x.masks.get_mut(*mi)) {
                    *m = info.clone();
                }
            }
            None => {}
        }
        Some(out)
    }

    pub fn rename_layer(&mut self, k: usize, name: &str) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.rename(k, name);
        } else if let Some(layer) = self.layer.as_mut() {
            layer.rename(k, name);
        }
    }

    /// Keep the change going on over frames, if any, as one undo (a change put off till now done first).
    pub fn close_slider_edit(&mut self) {
        if let Some(m) = self.mesh.as_mut() {
            match self.pending_slider.take() {
                Some(PendingSlider::Opacity(k, o)) => m.set_opacity(k, o),
                Some(PendingSlider::Mask(k, mi, info)) => m.set_mask(k, mi, info),
                None => {}
            }
            m.close_open();
        }
    }

    /// Set mesh layer `k`'s opacity (a drag is one undo).
    pub fn set_layer_opacity(&mut self, k: usize, opacity: f32) {
        if !self.live_edits(k) {
            self.pending_slider = Some(PendingSlider::Opacity(k, opacity));
            return;
        }
        if let Some(m) = self.mesh.as_mut() {
            m.set_opacity(k, opacity);
        }
    }

    /// Set how mesh layer `k`'s strokes show (one undo).
    pub fn set_layer_blend(&mut self, k: usize, blend: LayerBlend) {
        if let Some(m) = self.mesh.as_mut() {
            m.set_blend(k, blend);
        }
    }

    /// Whether mesh layer `k` is small enough for a slider to move it every frame (else it moves when let go).
    pub fn live_edits(&self, k: usize) -> bool {
        self.mesh.as_ref().is_none_or(|m| m.live_edits(k))
    }

    /// Pick mesh layer `k`, and its mask `mi` or (`None`) the layer itself, as what strokes work on: a painted mask
    /// picked makes strokes paint it (the colour it is not filled with first).
    pub fn select_mask(&mut self, k: usize, mi: Option<usize>) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.close_open();
            let fill = m.select_mask(k, mi);
            self.target = match (fill, self.target) {
                (None, _) => Target::Detail,
                (Some(_), Target::Mask { show }) => Target::Mask { show },
                (Some(f), _) => Target::Mask { show: f < 0.5 },
            };
        }
    }

    /// Add a mask on top of the active mesh layer's, picked (strokes paint a painted one).
    pub fn add_mask(&mut self, kind: MaskKind) {
        if let Some(m) = self.mesh.as_mut() {
            self.pending_slider = None;
            let k = m.active();
            let mi = m.add_mask(kind);
            if let Some(k) = k {
                self.select_mask(k, mi);
            }
        }
    }

    pub fn remove_mask(&mut self, k: usize, mi: usize) {
        if let Some(m) = self.mesh.as_mut() {
            m.remove_mask(k, mi);
            if m.active() == Some(k) && m.active_mask(k).is_none() && matches!(self.target, Target::Mask { .. }) {
                self.target = Target::Detail;
            }
        }
    }

    pub fn move_mask(&mut self, k: usize, mi: usize, up: bool) {
        if let Some(m) = self.mesh.as_mut() {
            m.move_mask(k, mi, up);
        }
    }

    /// Change mask `mi` of mesh layer `k` (a drag is one undo).
    pub fn set_mask(&mut self, k: usize, mi: usize, info: MaskInfo) {
        if !self.live_edits(k) {
            self.pending_slider = Some(PendingSlider::Mask(k, mi, info));
            return;
        }
        if let Some(m) = self.mesh.as_mut() {
            m.set_mask(k, mi, info);
        }
    }

    /// A new empty layer at `level` on top, made what strokes work on.
    pub fn add_layer(&mut self, level: usize) {
        self.settle();
        if let Some(layer) = self.layer.as_mut() {
            let n = layer.layers().len();
            layer.add_layer(&mut self.vol, &format!("Layer {}", n + 1), level);
            self.target = Target::Detail;
        }
    }

    pub fn remove_layer(&mut self, k: usize) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.finish(&self.vol);
            m.remove_layer(k);
            if matches!(self.target, Target::Mask { .. }) {
                self.target = Target::Detail;
            }
        } else if let Some(layer) = self.layer.as_mut() {
            layer.remove_layer(&mut self.vol, k);
        }
    }

    pub fn set_layer_visible(&mut self, k: usize, on: bool) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.set_visible(k, on);
        } else if let Some(layer) = self.layer.as_mut() {
            layer.set_visible(&mut self.vol, k, on);
        }
    }

    /// Hide the active layer everywhere (`to` 0, then paint it in with Mask: show) or show it everywhere (`to` 1).
    pub fn fill_mask(&mut self, to: f32) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            // (A stroke on the layers: the volume's changes are in first.)
            m.finish(&self.vol);
            if m.failure.is_some() {
                self.mesh_note = self.mesh_failure().map(|f| format!("No mask fill until the layers follow the volume again. {f}"));
                return;
            }
            m.fill_mask(to);
        } else if let Some(layer) = self.layer.as_mut() {
            layer.fill_mask(&mut self.vol, to);
        }
    }

    /// Hold layer `k` at another level (its resolution).
    pub fn set_layer_level(&mut self, k: usize, level: usize) {
        self.settle();
        if let Some(layer) = self.layer.as_mut() {
            layer.set_level(&mut self.vol, k, level);
        }
    }

    pub fn set_active_layer(&mut self, k: usize) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.set_active(k);
        } else if let Some(layer) = self.layer.as_mut() {
            layer.set_active(k);
        }
    }

    /// Before a save opens any file: everything settled and the layers caught up, so the waits come before a file is
    /// touched ([`Self::write_to`] then finds nothing left to wait for).
    pub fn prepare_save(&mut self) {
        self.settle();
        if let Some(m) = self.mesh.as_mut() {
            m.finish(&self.vol);
            m.close_open();
        }
    }

    /// Write the volume (with its layers, if any) as a `.svol` file.
    pub fn write_to(&mut self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.settle();
        match &self.layer {
            Some(layer) => layer.write_to(w)?,
            None => self.vol.write_to(w)?,
        }
        // Mesh layers last: "MLAY", the length of what follows, the layers' stream, the app's own notes; then the
        // offset of "MLAY" from the end and "MLAY" again, so a reader finds it from the end and reads the rest as
        // before.
        if let Some(m) = self.mesh.as_mut() {
            m.finish(&self.vol);
            m.close_open();
            let mut body = Vec::new();
            m.layers.write_to(&mut body).map_err(|e| std::io::Error::other(e.to_string()))?;
            let notes = serde_json::to_vec(&serde_json::json!({ "fills": m.fills })).map_err(|e| std::io::Error::other(e.to_string()))?;
            body.extend_from_slice(&(notes.len() as u64).to_le_bytes());
            body.extend_from_slice(&notes);
            w.write_all(b"MLAY")?;
            w.write_all(&(body.len() as u64).to_le_bytes())?;
            w.write_all(&body)?;
            w.write_all(&(body.len() as u64 + 24).to_le_bytes())?;
            w.write_all(b"MLAY")?;
        }
        Ok(())
    }

    /// A volume read from a `.svol` file, with its layers and surface noise if it has them.
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<VolumeMode> {
        let mut bytes = Vec::new();
        r.read_to_end(&mut bytes)?;
        // What could not be read and was left out: the volume opens whatever its later chunks hold.
        let mut lost: Vec<String> = Vec::new();
        // Whether the file held the old volume layers' records (an `SCLY` chunk), left out.
        let mut merged = false;
        // Mesh layers, if any, are the last chunk: taken off, the rest reads as before.
        let mlay = crate::mesh_stack::split_mlay(&mut bytes).unwrap_or_else(|e| {
            lost.push(format!("the mesh layers ({e})"));
            None
        });
        let read = |b: &[u8]| -> std::io::Result<_> {
            let mut cur = std::io::Cursor::new(b);
            let (vol, layer) = Layered::read_from(&mut cur, OPEN_LEVELS)?;
            Ok((vol, layer, cur.position() as usize))
        };
        let (vol, layer, pos) = match read(&bytes) {
            Ok(x) => x,
            Err(e) => {
                // A mesh layers chunk left on (damaged, or cut short with its closing mark) stops the volume's reader:
                // the volume is read up to where the chunk starts, if a place its mark sits gives one.
                let mut to = bytes.len();
                let mut found = None;
                for _ in 0..8 {
                    let Some(i) = bytes[..to].windows(4).rposition(|w| w == b"MLAY") else { break };
                    if let Ok(x) = read(&bytes[..i]) {
                        found = Some((i, x));
                        break;
                    }
                    to = i;
                }
                let Some((i, x)) = found else { return Err(e) };
                bytes.truncate(i);
                if !lost.iter().any(|l| l.starts_with("the mesh layers")) {
                    lost.push("the mesh layers (the file ends inside them)".into());
                }
                x
            }
        };
        let mut mode = VolumeMode::new(vol);
        if let Some(mut layer) = layer {
            layer.deferred = true;
            mode.layer = Some(layer);
        }
        // A chunk's body as the file gives its length, checked against the file (a cut or damaged file).
        let chunk = |at: usize, len: u64| -> std::io::Result<&[u8]> {
            usize::try_from(len)
                .ok()
                .and_then(|len| at.checked_add(len))
                .and_then(|end| bytes.get(at..end))
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "it runs past the end of the file"))
        };
        // Where the chunks read so far end.
        let mut end = pos;
        // The old volume layers' records (files from before mesh layers): "SCLY", their length, their stream. The
        // volume holds what they showed already, so they are left out: the form opens as it was shown, one volume.
        // (The layers' reader took the mark already.)
        if mode.layer.is_none() && pos >= 4 && &bytes[pos - 4..pos] == b"SCLY" && bytes.len() >= pos + 8 {
            let len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            merged = true;
            if chunk(pos + 8, len).is_ok() {
                end = pos + 8 + len as usize;
            }
        }
        // Surface noise as files before it was a layer kept it: after the layers, or in their place (its mark read
        // already, then). It becomes a layer.
        let at = if bytes[pos..].starts_with(b"SNOI") {
            Some(pos + 4)
        } else if mode.layer.is_none() && pos < bytes.len() && pos >= 4 && &bytes[pos - 4..pos] == b"SNOI" {
            Some(pos)
        } else {
            None
        };
        if let Some(at) = at
            && bytes.len() >= at + 4
        {
            let len = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            let noise = chunk(at + 4, len as u64)
                .and_then(|json| serde_json::from_slice::<SurfaceNoise>(json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
            match noise {
                Ok(n) => {
                    end = at + 4 + len as usize;
                    if n.on {
                        mode.add_noise_with(n);
                    }
                }
                Err(e) => lost.push(format!("the surface noise ({e})")),
            }
        }
        match mlay {
            Some(body) => match MeshStack::read_from(&body, &mode.vol) {
                Ok(mut stack) => {
                    lost.extend(stack.note.take());
                    mode.mesh = Some(stack);
                    mode.target = Target::Detail;
                }
                Err(e) => lost.push(format!("the mesh layers ({e})")),
            },
            // A mesh layers chunk cut short (its closing mark gone): the file ends inside it.
            None if bytes.get(end..).is_some_and(|rest| rest.starts_with(b"MLAY")) && !lost.iter().any(|l| l.starts_with("the mesh layers")) => {
                lost.push("the mesh layers (the file ends inside them)".into());
            }
            None => {}
        }
        let mut notes = Vec::new();
        if merged {
            notes.push("This file's old volume layers were merged into the form as shown".to_string());
        }
        if !lost.is_empty() {
            notes.push(format!("Opened without {}", lost.join("; ")));
        }
        if !notes.is_empty() {
            let what = notes.join(". ");
            eprintln!("{what}");
            mode.mesh_note = Some(what.clone());
            mode.lossy_open = Some(what);
        }
        Ok(mode)
    }

    /// Merge the detail layers into the form: one volume again, as shown. Undo starts afresh.
    pub fn merge_layer(&mut self) {
        self.settle();
        self.layer = None;
        self.vol.clear_history();
    }

    pub fn undo(&mut self) -> bool {
        self.settle();
        self.stroke_n = Vec3::ZERO;
        if let Some(m) = self.mesh.as_mut() {
            self.pending_slider = None;
            m.tap(self.vol.dirty_keys());
            let done = m.undo(&mut self.vol);
            if let Some(n) = m.note.take() {
                self.mesh_note = Some(n);
            }
            return done;
        }
        match self.layer.as_mut() {
            Some(layer) => layer.undo(&mut self.vol),
            None => self.vol.undo(),
        }
    }

    pub fn redo(&mut self) -> bool {
        self.settle();
        self.stroke_n = Vec3::ZERO;
        if let Some(m) = self.mesh.as_mut() {
            self.pending_slider = None;
            m.tap(self.vol.dirty_keys());
            let done = m.redo(&mut self.vol);
            if let Some(n) = m.note.take() {
                self.mesh_note = Some(n);
            }
            return done;
        }
        match self.layer.as_mut() {
            Some(layer) => layer.redo(&mut self.vol),
            None => self.vol.redo(),
        }
    }

    /// The surface under a ray: during a stroke, the one it began on at its level (so dabs do not climb what the stroke
    /// builds); otherwise the surface shown.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        // On a mesh layer or its mask: its surface as shown.
        if let Some(m) = self.mesh.as_ref()
            && (self.mesh_stroke.is_some() || (self.on_mesh() && !self.vol.in_stroke()))
        {
            let hit = m.layers.doc.raycast(&sculpt_core::geom::Ray::new(origin, dir))?;
            return Some((hit.point, hit.normal));
        }
        if let Some(layer) = self.layer.as_ref().filter(|l| self.target == Target::Form && l.form().in_stroke() && self.chosen) {
            return layer.form().raycast(origin, dir, self.level, true);
        }
        if self.vol.in_stroke() && self.chosen {
            self.vol.raycast(origin, dir, self.level, true)
        } else {
            self.vol.raycast(origin, dir, self.cap, false)
        }
    }

}

/// Mesh layers (see `mesh_stack`): strokes on them and their masks, driving their refits, the rows to redraw.
impl VolumeMode {
    /// Whether strokes go to the mesh layers: there are some (or the first is being made), and a layer or its mask is
    /// what strokes work on.
    fn on_mesh(&self) -> bool {
        (self.mesh.is_some() || self.mesh_build.is_some()) && matches!(self.target, Target::Detail | Target::Mask { .. })
    }

    /// With mesh layers, a `what` layer is not made (a note says so).
    fn refuse_with_mesh(&mut self, what: &str) -> bool {
        if self.mesh.is_none() && self.mesh_build.is_none() {
            return false;
        }
        self.mesh_note = Some(format!("A {what} layer cannot go with mesh layers yet"));
        true
    }

    /// The mesh layers (the real ones, never a copy run ahead on).
    pub fn mesh_layers(&self) -> Option<&sculpt_core::mesh_layers::MeshLayers> {
        self.mesh.as_ref().map(|m| &m.layers)
    }

    /// Whether a visible mesh layer exists.
    #[cfg(test)]
    pub fn layers_shown(&self) -> bool {
        self.mesh.as_ref().is_some_and(|m| m.layers_shown())
    }

    /// What the GPU rows need redrawn since the last call.
    pub fn take_rows_dirty(&mut self) -> RowsDirty {
        match self.mesh.as_mut() {
            Some(m) => m.take_rows_dirty(),
            None => RowsDirty { none: std::mem::take(&mut self.rows_none), ..Default::default() },
        }
    }

    /// Whether the first mesh layer's mesh is still being made.
    pub fn mesh_building(&self) -> bool {
        self.mesh_build.is_some()
    }

    /// The mask strokes paint on layer `k`, if one is picked.
    pub fn active_mask(&self, k: usize) -> Option<usize> {
        self.mesh.as_ref().and_then(|m| m.active_mask(k))
    }

    /// A new empty mesh layer on top, made what strokes work on. The first takes the mesh from the volume on another
    /// thread (a note says when it is in).
    pub fn add_mesh_layer(&mut self) {
        if self.layer.is_some() {
            self.mesh_note = Some("Mesh layers cannot go with detail, generator or surface noise layers yet".into());
            return;
        }
        self.settle();
        self.pending_slider = None;
        self.target = Target::Detail;
        if let Some(m) = self.mesh.as_mut() {
            m.add_layer();
            return;
        }
        if self.mesh_build.is_some() {
            return;
        }
        let snap = self.vol.preview_copy();
        // The mesh shows the levels the GPU draws, so its vertices are the drawn ones.
        let settings = sculpt_core::mesh_layers::MeshLayerSettings { cap: self.cap, ..Default::default() };
        let (tx, rx) = channel();
        std::thread::Builder::new()
            .name("mesh-layers-first".into())
            .spawn(move || {
                let r = sculpt_core::mesh_layers::MeshLayers::from_volume(&snap, settings).map_err(|e| e.to_string());
                let _ = tx.send(r);
            })
            .expect("spawn the mesh layer build");
        self.mesh_build = Some(MeshBuild { rx, dirty: Default::default(), started: Instant::now() });
        self.mesh_note = Some("Making the mesh for layers...".into());
    }

    /// Take the first mesh layer's mesh in if it is made (with `wait`, waiting for it).
    fn mesh_take_build(&mut self, wait: bool) {
        let Some(b) = self.mesh_build.as_ref() else { return };
        let r = if wait {
            b.rx.recv().map_err(|_| "the mesh build stopped".to_string()).and_then(|r| r)
        } else {
            match b.rx.try_recv() {
                Ok(r) => r,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => Err("the mesh build stopped".into()),
            }
        };
        let b = self.mesh_build.take().unwrap();
        match r {
            Ok(ml) => {
                let vertices = ml.doc.vertex_count();
                let mut stack = MeshStack::new(ml, b.dirty);
                stack.add_layer();
                self.mesh = Some(stack);
                self.mesh_note = Some(format!("Mesh layers ready: {} vertices, {:.1} s", vertices, b.started.elapsed().as_secs_f64()));
            }
            Err(e) => self.mesh_note = Some(format!("Could not make the mesh for layers: {e}")),
        }
    }

    /// Volume blocks changed: the mesh layers (or the mesh being made for them) are told.
    fn tap_keys(&mut self, keys: &[Key]) {
        if let Some(m) = self.mesh.as_mut() {
            m.tap(keys.iter().copied());
        } else if let Some(b) = self.mesh_build.as_mut() {
            b.dirty.extend(keys.iter().copied());
        }
    }

    /// The real volume's changes not yet drained, told to the mesh layers (while running ahead, `with_real` does it).
    fn tap(&mut self) {
        if self.ahead.is_some() || (self.mesh.is_none() && self.mesh_build.is_none()) {
            return;
        }
        let keys = self.vol.dirty_keys();
        self.tap_keys(&keys);
    }

    /// Whether the real volume is at rest: no stroke under way, ending, carrying or run ahead.
    fn at_rest(&self) -> bool {
        self.ahead.is_none() && !self.ending && self.end_job.is_none() && !self.carrying() && !self.vol.in_stroke() && self.layer.is_none()
    }

    /// Once a frame, after the volume's own work: the mesh layers told what the volume changed, a finished refit
    /// swapped in, a new one started once the volume is at rest and the pen has been up `REFIT_AFTER_MS`; during a
    /// mask stroke, the masks shown as painted.
    fn mesh_tick(&mut self) {
        self.mesh_take_build(false);
        self.tap();
        // (With a layer or its mask picked the next stroke is on the layers: it would only wait for the refit.)
        let waited = self.target != Target::Form || ms(self.pen_up) >= REFIT_AFTER_MS;
        let rest = self.at_rest() && waited;
        let idle = !self.pen_down && ms(self.pen_up) >= FREE_OLD_AFTER_MS;
        let Some(m) = self.mesh.as_mut() else { return };
        if idle && m.layers.old_held_ms().is_some_and(|t| t >= FREE_OLD_AFTER_MS) {
            m.layers.free_old();
        }
        if let Some(st) = self.mesh_stroke.as_mut() {
            if st.live && st.painted {
                let t = Instant::now();
                let _ = m.layers.doc.refresh_painted_masks();
                // Over a frame: shown when the stroke ends instead.
                if ms(t) > 12.0 {
                    st.live = false;
                }
                st.painted = false;
            }
            return;
        }
        let real = self.ahead.as_ref().map_or(&self.vol, |a| &a.vol);
        m.poll(real, false);
        if rest {
            m.start_refit(&self.vol);
        }
    }

    /// Begin a stroke on the active mesh layer or its mask: once everything the volume did is in the layers (a wait
    /// only within a moment of a volume stroke). Returns false if there are no mesh layers after all.
    fn mesh_begin(&mut self) -> bool {
        self.settle();
        self.mesh_take_build(true);
        self.tap();
        let target = self.target;
        let Some(m) = self.mesh.as_mut() else { return false };
        m.close_open();
        let t = Instant::now();
        m.finish(&self.vol);
        let waited = ms(t);
        if waited > 50.0 {
            self.mesh_note = Some(format!("Waited {waited:.0} ms for the layers to follow the volume"));
        }
        if m.failure.is_some() {
            // The layers sit on a surface the volume no longer has: a stroke there would be lost or wrong.
            self.mesh_note = self.mesh_failure().map(|f| format!("No layer strokes until the layers follow the volume again. {f}"));
            self.mesh_stroke = Some(MeshStroke { target, last: None, n: Vec3::ZERO, channel: None, live: false, painted: false, refused: true });
            return true;
        }
        let mut channel = None;
        match target {
            Target::Mask { .. } => channel = m.paint_channel(),
            _ => {
                let k = match m.active() {
                    Some(k) => k,
                    None => {
                        m.add_layer();
                        m.active().unwrap_or(0)
                    }
                };
                // A hidden layer sculpted on shows again first (a step of its own).
                if m.layers.doc.layers().get(k).is_some_and(|l| !l.visible) {
                    m.set_visible(k, true);
                }
            }
        }
        m.layers.doc.begin_stroke(if matches!(target, Target::Mask { .. }) { "Paint mask" } else { "Sculpt layer" });
        self.mesh_stroke = Some(MeshStroke { target, last: None, n: Vec3::ZERO, channel, live: true, painted: false, refused: false });
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
        self.lag_ms = 0.0;
        true
    }

    /// One dab of a stroke on a mesh layer (Clay Buildup, Trim Dynamic or Smooth) or its mask. Ctrl already swapped the
    /// brush (Clay for Trim and back), as on the volume, so the layer brushes are never inverted on top of that.
    fn mesh_dab(&mut self, d: DabInput) {
        use sculpt_core::brush::{self as b, Brush as _};
        let DabInput { at, radius, strength, hardness, brush, .. } = d;
        let (Some(m), Some(st)) = (self.mesh.as_mut(), self.mesh_stroke.as_mut().filter(|st| !st.refused)) else { return };
        let t = Instant::now();
        let doc = &mut m.layers.doc;
        let s = strength.clamp(0.0, 1.0);
        let falloff = b::Falloff { hardness: hardness.clamp(0.0, 0.999) };
        let Some((_, n)) = doc.sample_area(at, radius, (st.n != Vec3::ZERO).then_some(st.n)) else { return };
        st.n = n;
        match st.target {
            Target::Mask { show } => {
                // (Smoothing a mask is not a thing yet; a noise mask picked has nothing to paint.)
                let Some(ch) = st.channel.clone().filter(|_| brush != Brush::Smooth) else { return };
                doc.paint(&sculpt_core::PaintTarget::Channel(ch), at, radius, &falloff, 0.3 + 0.7 * s, if show { 1.0 } else { 0.0 }, Some(n));
                st.painted = true;
            }
            _ => {
                let dab = b::Dab { center: at, normal: n, direction: st.last.map_or(Vec3::ZERO, |p| at - p), pressure: 1.0 };
                let shape = brush.shape(strength);
                let settings = b::BrushSettings { radius, strength: shape.alpha, falloff, spacing: 0.12, front_faces_only: true, invert: false };
                let r = match brush {
                    Brush::Clay => b::ClayBuildup { height: shape.height, ..Default::default() }.dab(doc, &dab, &settings),
                    Brush::Trim => b::TrimDynamic { depth: shape.height, ..Default::default() }.dab(doc, &dab, &settings),
                    Brush::Smooth => b::FormSmooth.dab(doc, &dab, &settings),
                };
                if let Err(e) = r {
                    eprintln!("mesh layer dab: {e}");
                }
                st.last = Some(at);
            }
        }
        let ms = ms(t);
        self.last_ms += ms;
        self.last_dabs += 1;
        self.dab_ms = if self.dab_ms == 0.0 { ms } else { self.dab_ms * 0.8 + ms * 0.2 };
    }

    /// The stroke on a mesh layer or its mask ends: one step of the history.
    fn mesh_end(&mut self) {
        let Some(st) = self.mesh_stroke.take() else { return };
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
        let Some(m) = self.mesh.as_mut().filter(|_| !st.refused) else { return };
        // (A stroke that changed nothing, the pen off the model, is no step.)
        match m.layers.take_stroke_patch() {
            Ok(Some(patch)) => m.push(crate::mesh_stack::Entry::Layer { patch, fill: None }),
            Ok(None) => {}
            Err(e) => eprintln!("mesh layers: {e}"),
        }
    }
}

#[cfg(test)]
mod gauntlet;

#[cfg(test)]
mod mesh_tests;

#[cfg(test)]
mod form_speed_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod diagnostics;

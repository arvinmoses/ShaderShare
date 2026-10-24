//! The volume playground: sculpting a solid that never stretches, at whatever detail the view calls for.
//!
//! The solid is a distance field at several resolutions (`sculpt_core::generator::level_volume`): level 0 holds the
//! whole form, each level after it halves the step, and finer levels exist only where something was sculpted at them.
//! A stroke works at the level its brush calls for: a lattice step about a pixel on screen where the brush is, but no
//! finer than a 48th of its radius, so a big brush works on coarse form (and the finer detail under it rides along)
//! and a small one zoomed in carves detail as fine as the screen shows. The viewport keeps the surface on the GPU
//! (`gpu_levels`), built there from the blocks a stroke changed. Strokes can be undone, and the volume saved and opened
//! (`.svol`); layers and the procedural stack are not here yet.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::detail_layer::{FollowDone, GenDone, Layered};
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::surface_noise::SurfaceNoise;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::generator::level_volume::{CarryDone, EndDone, LevelVolume};

/// Levels in a new volume: level 0 and five finer, the finest 32 times finer.
pub const LEVELS: usize = 6;
/// The finest step a brush works at, in pixels on screen at the brush.
const PIXELS_PER_STEP: f32 = 1.0;
/// The most lattice steps across a brush's radius: bigger brushes work a level coarser.
const MAX_STEPS_PER_RADIUS: f32 = 48.0;
/// Finer blocks a carrying job takes at most: one job runs at a time, on its own threads, while the frames go on.
const CARRY_JOB: usize = 384;

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
    /// GPU time of the last surface rebuild.
    pub gpu_ms: f64,
    /// The brush normal of the stroke's last dab, which the next one leans toward.
    stroke_n: Vec3,
    /// Whether the stroke under way has chosen its level yet.
    chosen: bool,
    /// Carrying detail under a coarse stroke runs here, off the frame, a job at a time.
    pool: Arc<rayon::ThreadPool>,
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
}

impl VolumeMode {
    /// A ball of radius 1 whose level 0 alone is about `triangles` triangles.
    pub fn ball(triangles: f32) -> VolumeMode {
        let voxel = (4.0 * std::f32::consts::PI / (triangles / 2.0)).sqrt();
        VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, LEVELS))
    }

    pub fn new(mut vol: LevelVolume) -> VolumeMode {
        let cap = vol.levels() - 1;
        // Dabs only queue the carrying; jobs on the pool do it.
        vol.carry_budget = Some(0.0);
        // A quarter of the cores: the dabs and the sync want the rest, and more would only slow them.
        let threads = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).max(2);
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("carry {i}")).build().expect("a thread pool"));
        VolumeMode {
            vol,
            layer: None,
            target: Target::Detail,
            on_layer: false,
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
            gpu_ms: 0.0,
            stroke_n: Vec3::ZERO,
            chosen: false,
            pool,
            job: None,
            carry_blocks: 0,
            carry_ms: 0.0,
            ending: false,
            end_job: None,
            end_ms: 0.0,
            ended_specks: None,
        }
    }

    /// Spacing floor for dabs: the step of the level the stroke works at.
    pub fn voxel(&self) -> f32 {
        self.vol.voxel(self.level)
    }

    /// The level a brush of `radius` (world units) works at, `px` world units being a pixel at the brush.
    pub fn level_for(&self, radius: f32, px: f32) -> usize {
        let want = (px * PIXELS_PER_STEP).max(radius / MAX_STEPS_PER_RADIUS);
        let mut l = 0;
        while l + 1 < self.vol.levels() && self.vol.voxel(l + 1) >= want {
            l += 1;
        }
        l
    }

    /// One dab of Clay Buildup (or, with `carve`, Trim Dynamic) at surface point `at`: a plane brush whose plane
    /// follows the surface the stroke began on. `px` is the size of a pixel at the brush, which (with the radius) picks
    /// the level the stroke works at, on its first dab.
    pub fn dab(&mut self, at: Vec3, radius: f32, px: f32, strength: f32, hardness: f32, carve: bool) {
        let t = Instant::now();
        // For replaying a session's strokes (`SCULPT_DAB_LOG`): every dab, as given.
        if std::env::var_os("SCULPT_DAB_LOG").is_some() {
            eprintln!("DAB {} {} {} {radius} {px} {strength} {hardness} {carve}", at.x, at.y, at.z);
        }
        let target = self.layer.as_ref().map(|_| self.routed());
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
            match (target, self.layer.as_ref()) {
                (Some(Target::Form), Some(layer)) => self.level = self.level.min(layer.from() - 1),
                (Some(Target::Detail), Some(layer)) => self.level = self.level.clamp(layer.from(), layer.level()),
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
        let s = strength.clamp(0.0, 1.0);
        let height = radius * if carve { 0.015 + 0.06 * s } else { 0.015 + 0.08 * s };
        let dabbed = match (form, self.layer.as_mut()) {
            (true, Some(layer)) => {
                let out = layer.plane_dab(&mut self.vol, self.level, at, along, radius, height, hardness, 0.1 + 0.3 * s, carve);
                if std::env::var_os("SCULPT_LAYER_LOG").is_some() {
                    eprintln!("form dab at level {} r {radius:.3}: {:.1} ms; layer {} blocks; follow {:?}", self.level, t.elapsed().as_secs_f64() * 1e3, layer.blocks(), layer.last);
                }
                out
            }
            _ => self.vol.plane_dab(self.level, at, along, radius, height, hardness, 0.1 + 0.3 * s, carve),
        };
        let Some((n, _)) = dabbed else { return };
        // The original surface's plane is steady already; easing it a little more smooths the turn over sharp forms.
        self.stroke_n = if self.stroke_n == Vec3::ZERO { n } else { (self.stroke_n * 0.5 + n * 0.5).try_normalize().unwrap_or(n) };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.last_ms += ms;
        self.last_dabs += 1;
        self.stroke_level = self.level;
        self.dab_ms = if self.dab_ms == 0.0 { ms } else { self.dab_ms * 0.8 + ms * 0.2 };
    }

    /// Once a frame: bring in a carrying job that finished, and start the next on what is queued (see
    /// `LevelVolume::carry_take`).
    pub fn tick(&mut self) {
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
            if self.follow_job.is_none()
                && let Some(job) = layer.follow_take()
            {
                // Mid-stroke on the carrying threads (a quarter of the cores), so the dabs and the sync keep the rest;
                // the stroke's last on all of them.
                let (tx, rx) = channel();
                if job.last() {
                    rayon::spawn(move || {
                        let _ = tx.send(job.run());
                    });
                } else {
                    let pool = self.pool.clone();
                    self.pool.spawn(move || {
                        let _ = tx.send(pool.install(|| job.run()));
                    });
                }
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
        if self.job.is_none()
            && let Some(job) = self.vol.carry_take(CARRY_JOB)
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
        self.last_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    /// Work out the stroke's end on the shared threads (nothing else wants them once the pen is up).
    fn start_end(&mut self) {
        match self.vol.end_stroke_take() {
            Some(job) => {
                let (tx, rx) = channel();
                rayon::spawn(move || {
                    let _ = tx.send(job.run());
                });
                self.end_job = Some(rx);
            }
            None => self.ending = false,
        }
    }

    fn take_end(&mut self, done: EndDone) {
        let end = self.vol.end_stroke_apply(done);
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

    /// Finish a stroke whose end is still being worked out, now: before anything that needs the volume settled (a
    /// new stroke, undo, saving).
    pub fn settle(&mut self) {
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
        self.ending || self.carrying() || self.follow_job.is_some() || self.form_end.is_some() || self.gen_job.is_some() || self.layer.as_ref().is_some_and(|l| l.busy())
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

    pub fn begin_stroke(&mut self) {
        if std::env::var_os("SCULPT_DAB_LOG").is_some() {
            eprintln!("BEGIN");
        }
        self.settle();
        let target = self.routed();
        match (self.layer.as_mut(), target) {
            (Some(layer), Target::Form) => layer.begin_stroke(),
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
                    rayon::spawn(move || {
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
    fn routed(&self) -> Target {
        match (self.target, self.layer.as_ref()) {
            (Target::Detail, Some(layer)) if layer.generator(layer.active()).is_some() || layer.noise(layer.active()).is_some() => Target::Form,
            (t, _) => t,
        }
    }

    /// A new generator layer on top (rocks scattered over the form, to start with), made active. With no layers yet,
    /// the stack is made first, the finest level's work becoming a detail layer.
    pub fn add_generator(&mut self) {
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
        // (Roughness keeps the octaves that level holds; a finer level shows more of it, at more cost.)
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
            rough: 0.15,
            ..GeneratorSpec::default()
        };
        let t = Instant::now();
        layer.add_generator(&mut self.vol, &format!("Generator {}", n + 1), level, &spec);
        self.generator_ms = t.elapsed().as_secs_f64() * 1e3;
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
        self.generator_ms = t.elapsed().as_secs_f64() * 1e3;
        if !dragging {
            self.end_generator_edit();
        }
    }

    /// Start composing a previewed generator at its own level on another thread, unless one is under way.
    fn start_generator_job(&mut self) {
        if self.gen_job.is_some() {
            return;
        }
        let Some(job) = self.layer.as_mut().and_then(|l| l.generator_take()) else { return };
        let (tx, rx) = channel();
        let t = Instant::now();
        rayon::spawn(move || {
            let mut done = job.run();
            done.ms = t.elapsed().as_secs_f64() * 1e3;
            let _ = tx.send(done);
        });
        self.gen_job = Some(rx);
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
        if std::mem::take(&mut self.generator_edit)
            && let Some(layer) = self.layer.as_mut()
        {
            layer.end_edit();
        }
        self.start_generator_job();
    }

    /// Move the work at levels `from` and finer into a detail layer over the rest (the form).
    pub fn make_layer(&mut self, from: usize) {
        self.settle();
        let from = from.clamp(1, self.vol.levels() - 1);
        let mut layer = Layered::new(&mut self.vol, from);
        // Dabs on the form only queue the layer's following: jobs off the frame do it.
        layer.deferred = true;
        self.layer = Some(layer);
        self.target = Target::Detail;
    }

    /// A new empty layer at `level` on top, made what strokes work on.
    pub fn add_layer(&mut self, level: usize) {
        self.settle();
        if let Some(layer) = self.layer.as_mut() {
            let n = layer.layers().len();
            layer.add_layer(&format!("Layer {}", n + 1), level);
            self.target = Target::Detail;
        }
    }

    pub fn remove_layer(&mut self, k: usize) {
        self.settle();
        if let Some(layer) = self.layer.as_mut() {
            layer.remove_layer(&mut self.vol, k);
        }
    }

    pub fn set_layer_visible(&mut self, k: usize, on: bool) {
        self.settle();
        if let Some(layer) = self.layer.as_mut() {
            layer.set_visible(&mut self.vol, k, on);
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
        if let Some(layer) = self.layer.as_mut() {
            layer.set_active(k);
        }
    }

    /// Write the volume (with its layers, if any) as a `.svol` file.
    pub fn write_to(&mut self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.settle();
        match &self.layer {
            Some(layer) => layer.write_to(w)?,
            None => self.vol.write_to(w)?,
        }
        Ok(())
    }

    /// A volume read from a `.svol` file, with its layers and surface noise if it has them.
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<VolumeMode> {
        let mut bytes = Vec::new();
        r.read_to_end(&mut bytes)?;
        let mut cur = std::io::Cursor::new(&bytes[..]);
        let (vol, layer) = Layered::read_from(&mut cur, LEVELS)?;
        let pos = cur.position() as usize;
        let mut mode = VolumeMode::new(vol);
        if let Some(mut layer) = layer {
            layer.deferred = true;
            mode.layer = Some(layer);
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
            let len = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
            if let Some(json) = bytes.get(at + 4..at + 4 + len) {
                let n: SurfaceNoise = serde_json::from_slice(json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
                if n.on {
                    mode.add_noise_with(n);
                }
            }
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
        match self.layer.as_mut() {
            Some(layer) => layer.undo(&mut self.vol),
            None => self.vol.undo(),
        }
    }

    pub fn redo(&mut self) -> bool {
        self.settle();
        self.stroke_n = Vec3::ZERO;
        match self.layer.as_mut() {
            Some(layer) => layer.redo(&mut self.vol),
            None => self.vol.redo(),
        }
    }

    /// The surface under a ray: during a stroke, the one it began on at its level (so dabs do not climb what the stroke
    /// builds); otherwise the surface shown.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// At the demo's scale (a ball of 1M triangles at level 0, a layer of a few thousand blocks), a big brush on the
    /// form: what a frame costs while the layer follows off the frame, and how long it takes to catch up.
    #[test]
    #[ignore]
    fn big_form_strokes_under_a_layer_stay_cheap_a_frame() {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, LEVELS));
        for j in 0..12 {
            for i in 0..60 {
                let p = Vec3::new(-0.06 + 0.002 * i as f32, -0.06 + 0.01 * j as f32, 1.0).normalize();
                mode.vol.begin_stroke();
                mode.vol.ball_dab(4, p, 0.0015, 0.0004, true);
                mode.vol.end_stroke();
            }
        }
        let layered = std::env::var_os("NO_LAYER").is_none();
        if layered {
            mode.make_layer(4);
            mode.target = Target::Form;
        }
        let blocks = mode.layer.as_ref().map_or(0, |l| l.blocks());
        let (mut dab_ms, mut tick_ms) = (0.0f64, 0.0f64);
        let mut worst = 0.0f64;
        let mut total = 0.0f64;
        let mut frames = 0;
        mode.begin_stroke();
        for k in 0..=24 {
            let d = Vec3::new(-0.1 + 0.2 * k as f32 / 24.0, 0.0, 1.0).normalize();
            let t = Instant::now();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.dab(hit, 0.5, 0.05, 1.0, 0.5, false);
            }
            dab_ms += t.elapsed().as_secs_f64() * 1e3;
            let t2 = Instant::now();
            mode.tick();
            tick_ms += t2.elapsed().as_secs_f64() * 1e3;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            worst = worst.max(ms);
            total += ms;
            frames += 1;
        }
        let t = Instant::now();
        mode.end_stroke();
        let pen_up = t.elapsed().as_secs_f64() * 1e3;
        let landed = mode.layer.as_ref().map_or(0, |l| l.follows);
        let t = Instant::now();
        let mut ticks = 0;
        while mode.busy() {
            mode.tick();
            ticks += 1;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let catch_up = t.elapsed().as_secs_f64() * 1e3;
        let last = mode.layer.as_ref().map(|l| l.last).unwrap_or_default();
        eprintln!("dabs {:.1} ms, ticks {:.1} ms a frame; {landed} follows landed during the stroke", dab_ms / frames as f64, tick_ms / frames as f64);
        eprintln!("layer {blocks} blocks: a frame (dab and tick) {:.1} ms on average, {worst:.1} at most; pen up {pen_up:.1} ms; caught up {catch_up:.0} ms later ({ticks} ticks); last follow {last:?}", total / frames as f64);
        assert!(!mode.busy());
    }

    /// A volume with two layers of different resolutions saves and opens again with both, showing the same.
    #[test]
    fn layers_save_and_open_with_the_volume() {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
        let dot = Vec3::new(0.0, 0.0, 1.0);
        mode.vol.begin_stroke();
        mode.vol.ball_dab(4, dot, 0.004, 0.001, false);
        mode.vol.end_stroke();
        mode.make_layer(3);
        mode.add_layer(3);
        mode.begin_stroke();
        let (hit, _) = mode.raycast(Vec3::new(0.03, 0.0, 3.0), -Vec3::Z).unwrap();
        mode.dab(hit, 0.012, 0.01, 1.0, 0.5, false);
        mode.end_stroke();
        mode.settle();
        let top = |m: &VolumeMode, x: f32| m.vol.raycast(Vec3::new(x, 0.0, 3.0), -Vec3::Z, 4, false).map_or(f32::NAN, |(p, _)| p.z);
        let mut file = Vec::new();
        mode.write_to(&mut file).unwrap();
        let opened = VolumeMode::read_from(&mut &file[..]).unwrap();
        let layers = opened.layer.as_ref().map(|l| l.layers()).unwrap_or_default();
        eprintln!("opened {} layers: {:?}", layers.len(), layers.iter().map(|l| (l.level, l.blocks)).collect::<Vec<_>>());
        assert_eq!(layers.len(), 2);
        assert!(layers[1].blocks > 0, "the stroke on the second layer was not kept");
        for x in [0.0, 0.03] {
            assert!((top(&opened, x) - top(&mode, x)).abs() < 2e-5, "opened, the surface is not where it was");
        }
    }

    /// Pieces of a surface: their sizes in vertices.
    fn pieces(mesh: &sculpt_core::mesh::PolyMesh) -> Vec<usize> {
        let n = mesh.positions.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for f in &mesh.faces {
            let k = if f[3] == sculpt_core::mesh::NO_VERT { 3 } else { 4 };
            for j in 1..k {
                let (a, b) = (root(&mut parent, f[0] as usize), root(&mut parent, f[j] as usize));
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut used = vec![false; n];
        for f in &mesh.faces {
            for &v in f.iter().filter(|&&v| v != sculpt_core::mesh::NO_VERT) {
                used[v as usize] = true;
            }
        }
        let mut size = std::collections::HashMap::new();
        for i in (0..n).filter(|&i| used[i]) {
            *size.entry(root(&mut parent, i)).or_insert(0usize) += 1;
        }
        size.into_values().collect()
    }

    /// Spiky vertices: farther from the mean of their neighbours than `k` times the mean edge to them, with where
    /// they are.
    fn spikes(mesh: &sculpt_core::mesh::PolyMesh, k: f32) -> Vec<Vec3> {
        let n = mesh.positions.len();
        let mut sum = vec![Vec3::ZERO; n];
        let mut len = vec![0.0f32; n];
        let mut cnt = vec![0u32; n];
        for f in &mesh.faces {
            let m = if f[3] == sculpt_core::mesh::NO_VERT { 3 } else { 4 };
            for j in 0..m {
                let (a, b) = (f[j] as usize, f[(j + 1) % m] as usize);
                if a == b {
                    continue;
                }
                let (pa, pb) = (mesh.positions[a], mesh.positions[b]);
                sum[a] += pb;
                sum[b] += pa;
                len[a] += (pa - pb).length();
                len[b] += (pa - pb).length();
                cnt[a] += 1;
                cnt[b] += 1;
            }
        }
        (0..n).filter(|&i| cnt[i] > 0 && (mesh.positions[i] - sum[i] / cnt[i] as f32).length() > k * len[i] / cnt[i] as f32).map(|i| mesh.positions[i]).collect()
    }

    /// Volumes dumped from the review's frames (`SCULPT_DEMO_DUMP`), in the folder `SCULPT_DUMPS`: their surfaces'
    /// spiky vertices, by the finest level where they are, and a few of them.
    #[test]
    #[ignore]
    fn dumped_frames_spikes() {
        let Ok(dir) = std::env::var("SCULPT_DUMPS") else { return };
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "svol")).collect();
        files.sort();
        for f in files {
            let vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(&f).unwrap()), LEVELS).unwrap();
            let (mesh, _) = vol.surface(vol.levels() - 1);
            let sp = spikes(&mesh, 0.6);
            let lv: Vec<usize> = (0..LEVELS).map(|l| sp.iter().filter(|p| vol.level_of(**p) == l).count()).collect();
            eprintln!("{}: {} vertices, {} spiky, by level {lv:?}; e.g. {:?}", f.file_name().unwrap().to_string_lossy(), mesh.positions.len(), sp.len(), &sp[..sp.len().min(4)]);
        }
    }

    /// A shaded height map of `vol`'s surface over a square `half` across around `c` (seen along its normal), as a PGM.
    fn height_map(vol: &LevelVolume, c: Vec3, half: f32, px: usize, path: &std::path::Path) {
        use rayon::prelude::*;
        let n = c.normalize();
        let u = n.cross(Vec3::Y).normalize();
        let v = n.cross(u);
        let step = 2.0 * half / px as f32;
        let hs: Vec<f32> = (0..px * px)
            .into_par_iter()
            .map(|k| {
                let (i, j) = ((k % px) as f32, (k / px) as f32);
                let q = c + u * (-half + step * i) + v * (-half + step * j);
                vol.raycast(q + n * 0.3, -n, vol.levels() - 1, false).map_or(f32::NAN, |(hit, _)| (hit - c).dot(n))
            })
            .collect();
        let at = |i: usize, j: usize| hs[j.min(px - 1) * px + i.min(px - 1)];
        let light = Vec3::new(-0.5, 0.6, 0.6).normalize();
        let mut img = format!("P5 {px} {px} 255
").into_bytes();
        for j in 0..px {
            for i in 0..px {
                let dx = (at(i + 1, j) - at(i.saturating_sub(1), j)) / (2.0 * step);
                let dy = (at(i, j + 1) - at(i, j.saturating_sub(1))) / (2.0 * step);
                let nn = Vec3::new(-dx, -dy, 1.0).normalize();
                img.push((nn.dot(light).clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
        std::fs::write(path, img).unwrap();
        // The finest level at each pixel, as grey.
        let levels: Vec<u8> = (0..px * px)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k % px, k / px);
                let h = at(i, j);
                if h.is_nan() {
                    return 0;
                }
                let q = c + u * (-half + step * i as f32) + v * (-half + step * j as f32) + n * h;
                (vol.level_of(q) * 50) as u8
            })
            .collect();
        let mut lv = format!("P5 {px} {px} 255
").into_bytes();
        lv.extend(levels);
        std::fs::write(path.with_extension("lvl.pgm"), lv).unwrap();
        // Where the surface jumps between neighbouring pixels by more than four pixels' width: the levels and blocks
        // on its two sides.
        if std::env::var_os("SCULPT_REPLAY_JUMPS").is_some() {
            let mut seen = std::collections::BTreeMap::new();
            for j in 0..px - 1 {
                for i in 0..px - 1 {
                    for (a, b) in [((i, j), (i + 1, j)), ((i, j), (i, j + 1))] {
                        let (ha, hb) = (at(a.0, a.1), at(b.0, b.1));
                        if (ha - hb).abs() > 4.0 * step {
                            let (hi, h) = if ha > hb { (a, ha) } else { (b, hb) };
                            let q = c + u * (-half + step * hi.0 as f32) + v * (-half + step * hi.1 as f32) + n * h;
                            let l = vol.level_of(q);
                            let bc = ((q - vol.origin()) / (vol.voxel(l) * 8.0)).floor().as_ivec3();
                            *seen.entry((l, bc.x, bc.y, bc.z)).or_insert(0) += 1;
                        }
                    }
                }
            }
            eprintln!("{}: jumps at the high side's finest level and block: {:?}", path.display(), seen.iter().take(12).collect::<Vec<_>>());
        }
    }

    /// The level 0 field's slope at samples near the surface around `patch`, in quarters (0-0.25, ..., 1.75+).
    fn slope(vol: &LevelVolume, patch: Vec3, half: f32) -> [usize; 8] {
                let vx = vol.voxel(0);
                let (lo, hi) = (((patch - Vec3::splat(half.max(0.1)) - vol.origin()) / vx).floor().as_ivec3(), ((patch + Vec3::splat(half.max(0.1)) - vol.origin()) / vx).ceil().as_ivec3());
                let mut hist = [0usize; 8];
                for z in lo.z..=hi.z {
                    for y in lo.y..=hi.y {
                        for x in lo.x..=hi.x {
                            let i = glam::IVec3::new(x, y, z);
                            let f = vol.sample(0, i, false);
                            if f.abs() > 0.7 * vx {
                                continue;
                            }
                            let g = glam::Vec3::new(
                                vol.sample(0, i + glam::IVec3::X, false) - vol.sample(0, i - glam::IVec3::X, false),
                                vol.sample(0, i + glam::IVec3::Y, false) - vol.sample(0, i - glam::IVec3::Y, false),
                                vol.sample(0, i + glam::IVec3::Z, false) - vol.sample(0, i - glam::IVec3::Z, false),
                            ).length() / (2.0 * vx);
                            hist[((g / 0.25) as usize).min(7)] += 1;
                        }
                    }
                }
                hist
    }

    /// The review's big stroke replayed on the volume dumped just before it (`SCULPT_REPLAY_DIR` holds `dump_01604.svol`
    /// and the session's dab log `dabs.txt`), drawn around the patch before and after its dabs.
    #[test]
    #[ignore]
    fn replay_the_big_stroke() {
        let Ok(dir) = std::env::var("SCULPT_REPLAY_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        let vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(dir.join("dump_01604.svol")).unwrap()), LEVELS).unwrap();
        let log = std::fs::read_to_string(dir.join("dabs.txt")).unwrap();
        let last = log.rfind("BEGIN").unwrap();
        let dabs: Vec<Vec<f32>> = log[last..].lines().filter(|l| l.starts_with("DAB")).map(|l| l.split_whitespace().skip(1).take(7).map(|x| x.parse().unwrap()).collect()).collect();
        let carve: Vec<bool> = log[last..].lines().filter(|l| l.starts_with("DAB")).map(|l| l.ends_with("true")).collect();
        let mut mode = VolumeMode::new(vol);
        let patch = std::env::var("SCULPT_REPLAY_CENTER").ok().map(|c| {
            let v: Vec<f32> = c.split(',').map(|x| x.parse().unwrap()).collect();
            Vec3::new(v[0], v[1], v[2])
        }).unwrap_or(Vec3::new(0.30, 0.34, 0.89));
        let half: f32 = std::env::var("SCULPT_REPLAY_HALF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.12);
        // Level 0's samples in a lattice box around the view's centre, as text, per stage.
        let dump_box = |vol: &LevelVolume, name: &str| {
            let Ok(r) = std::env::var("SCULPT_REPLAY_BOX") else { return };
            let r: i32 = r.parse().unwrap();
            let c = ((patch - vol.origin()) / vol.voxel(0)).round().as_ivec3();
            let mut out = String::new();
            for z in c.z - r..=c.z + r {
                for y in c.y - r..=c.y + r {
                    for x in c.x - r..=c.x + r {
                        let i = glam::IVec3::new(x, y, z);
                        out.push_str(&format!("{x} {y} {z} {} {}\n", vol.sample(0, i, false), vol.level_of(vol.origin() + i.as_vec3() * vol.voxel(0))));
                    }
                }
            }
            std::fs::write(dir.join(format!("box_{name}.txt")), out).unwrap();
        };
        height_map(&mode.vol, patch, half, 700, &dir.join("map_0.pgm"));
        dump_box(&mode.vol, "0");
        mode.begin_stroke();
        for (k, d) in dabs.iter().enumerate() {
            mode.dab(Vec3::new(d[0], d[1], d[2]), d[3], d[4], d[5], d[6], carve[k]);
            mode.tick();
            height_map(&mode.vol, patch, half, 700, &dir.join(format!("map_{}.pgm", k + 1)));
            dump_box(&mode.vol, &format!("{}", k + 1));
            eprintln!("after dab {}: level 0 slope near the surface, in quarters: {:?}", k + 1, slope(&mode.vol, patch, half));
        }
        eprintln!("{:.1} ms per dab", mode.last_ms / mode.last_dabs.max(1) as f64);
        mode.end_stroke();
        mode.settle();
        eprintln!("after the end: level 0 slope near the surface, in quarters: {:?}", slope(&mode.vol, patch, half));
        height_map(&mode.vol, patch, half, 700, &dir.join("map_end.pgm"));
        dump_box(&mode.vol, "end");
    }

    /// A big level 0 stroke over sculpted form and fine work, as the review's: what its surface looks like mid-stroke
    /// (small loose pieces, open or tangled edges), dab by dab.
    #[test]
    #[ignore]
    fn a_big_stroke_mid_stroke_surface() {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, LEVELS));
        let stroke = |mode: &mut VolumeMode, a: Vec3, b: Vec3, r: f32, px: f32, carve: bool, dabs: usize, watch: bool| {
            mode.begin_stroke();
            for k in 0..=dabs {
                let d = a.lerp(b, k as f32 / dabs as f32).normalize();
                if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                    mode.dab(hit, r, px, 1.0, 0.5, carve);
                }
                mode.tick();
                if watch {
                    let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
                    let (open, nm) = sculpt_core::generator::levels::edge_counts(&mesh.faces);
                    let mut sizes = pieces(&mesh);
                    sizes.sort_unstable();
                    let small = sizes.iter().filter(|&&s| s < sculpt_core::generator::level_volume::SPECK_VERTS).count();
                    let sp = spikes(&mesh, 0.6);
                    let lv: Vec<usize> = (0..LEVELS).map(|l| sp.iter().filter(|p| mode.vol.level_of(**p) == l).count()).collect();
                    eprintln!("dab {k} at level {}: {} pieces ({small} small), open edges {open}, tangled {nm}; {} spiky vertices, by the finest level there {lv:?}", mode.level, sizes.len(), sp.len());
                }
            }
            mode.end_stroke();
            mode.settle();
        };
        // Broad form, then zoomed-in clay, then fine trims.
        for (y, r) in [(0.05f32, 0.2f32), (-0.1, 0.18)] {
            stroke(&mut mode, Vec3::new(-0.2, y, 1.0), Vec3::new(0.2, y, 1.0), r, 0.02, false, 12, false);
        }
        stroke(&mut mode, Vec3::new(0.0, 0.2, 1.0), Vec3::new(0.05, -0.15, 1.0), 0.1, 0.02, true, 10, false);
        for y in [-0.02f32, 0.02] {
            stroke(&mut mode, Vec3::new(-0.08, y, 1.0), Vec3::new(0.08, y, 1.0), 0.03, 0.0006, false, 16, false);
        }
        for k in 0..5 {
            let y = -0.03 + 0.012 * k as f32;
            stroke(&mut mode, Vec3::new(-0.05, y, 1.0), Vec3::new(0.05, y + 0.003, 1.0), 0.004, 0.0001, true, 12, false);
        }
        let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
        eprintln!("before: {} pieces, {} spiky vertices; blocks per level {:?}", pieces(&mesh).len(), spikes(&mesh, 0.6).len(), mode.vol.memory().0);
        stroke(&mut mode, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.1, 0.02, 1.0), 0.45, 0.02, false, 8, true);
        let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
        eprintln!("after the stroke ended: {} pieces, {} spiky vertices", pieces(&mesh).len(), spikes(&mesh, 0.6).len());
    }

    /// With a detail layer, a mask stroke hides the layer where it passes, and undo brings it back.
    #[test]
    fn a_generator_edit_lands_at_its_own_level_off_the_frame() {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
        mode.add_generator();
        let k = mode.layer.as_ref().unwrap().active();
        let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
        spec.size *= 1.3;
        mode.set_generator(k, &spec, false);
        // Frames go on until the job is in.
        let start = Instant::now();
        while mode.busy() && start.elapsed().as_secs() < 20 {
            mode.tick();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(!mode.busy(), "the job never came in");
        let layer = mode.layer.as_mut().unwrap();
        assert_eq!(layer.generator(k), Some(&spec));
        assert!(layer.generator_take().is_none(), "still shown as a preview");
    }

    #[test]
    fn surface_noise_saves_as_a_layer_and_older_files_open_with_it() {
        let n = SurfaceNoise { on: true, amplitude: 0.02, scale: 7.0, octaves: 4, kind: 1, seed: 9 };
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.05, 3));
        mode.add_noise_with(n);
        let mut buf = Vec::new();
        mode.write_to(&mut buf).unwrap();
        let back = VolumeMode::read_from(&mut &buf[..]).unwrap();
        assert_eq!(back.shown_noise().map(|s| s.0), Some(n));
        // A file of one volume and its surface noise after it, as before noise was a layer.
        let vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.05, 3);
        let mut old = Vec::new();
        vol.write_to(&mut old).unwrap();
        let json = serde_json::to_vec(&n).unwrap();
        old.extend_from_slice(b"SNOI");
        old.extend_from_slice(&(json.len() as u32).to_le_bytes());
        old.extend_from_slice(&json);
        let back = VolumeMode::read_from(&mut &old[..]).unwrap();
        assert_eq!(back.shown_noise().map(|s| s.0), Some(n));
    }

    #[test]
    fn a_mask_stroke_hides_the_detail_layer() {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
        let dot = Vec3::new(0.0, 0.0, 1.0);
        mode.vol.begin_stroke();
        mode.vol.ball_dab(4, dot, 0.004, 0.001, false);
        mode.vol.end_stroke();
        let top = |m: &VolumeMode| m.vol.raycast(dot * 3.0, -dot, 4, false).map_or(f32::NAN, |(p, _)| p.z);
        let raised = top(&mode);
        mode.make_layer(3);
        assert!((top(&mode) - raised).abs() < 2e-4, "the layer does not show the dot");
        mode.target = Target::Mask { show: false };
        mode.begin_stroke();
        let (hit, _) = mode.raycast(dot * 3.0, -dot).unwrap();
        for _ in 0..4 {
            mode.dab(hit, 0.02, 0.001, 1.0, 0.5, false);
        }
        mode.end_stroke();
        let hidden = top(&mode);
        eprintln!("dot top {raised:.5}, hidden {hidden:.5}");
        assert!(hidden < raised - 0.002, "the mask did not hide the dot");
        assert!(mode.undo());
        assert!((top(&mode) - raised).abs() < 2e-4, "undo did not bring the dot back");
    }

    /// Big brushes work coarse, small brushes zoomed in work fine, and nothing finer than the volume has.
    #[test]
    fn brushes_pick_their_level_from_their_size_on_screen() {
        let mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, LEVELS));
        // Zoomed out: a pixel is a level 0 step or more.
        assert_eq!(mode.level_for(0.3, 0.02), 0);
        // A large brush zoomed in still works coarse: its radius is many steps of any finer level.
        assert_eq!(mode.level_for(0.5, 0.0005), 0);
        // A small brush zoomed in: as fine as a pixel.
        assert_eq!(mode.level_for(0.02, 0.0012), 4);
        assert_eq!(mode.level_for(0.004, 0.00001), LEVELS - 1);
    }

    /// Scrubbing a big brush back and forth over bumpy clay: the brush turns a little per dab, far less than the
    /// surface normal under it does.
    #[test]
    fn scrubbing_turns_the_brush_gently() {
        let mut mode = VolumeMode::ball(4e5);
        let px = 0.05;
        for i in 0..30 {
            let a = i as f32 * 2.4;
            let at = Vec3::new(0.35 * a.cos() * (i as f32 / 30.0), 0.35 * a.sin() * (i as f32 / 30.0), 1.0).normalize();
            mode.begin_stroke();
            if let Some((hit, _)) = mode.raycast(at * 3.0, -at) {
                for _ in 0..10 {
                    mode.dab(hit, 0.08, px, 1.0, 0.9, false);
                }
            }
            mode.end_stroke();
        }
        let (mut turn, mut raw_turn, mut worst, mut n) = (0.0f32, 0.0f32, 0.0f32, 0);
        let (mut last, mut last_raw) = (Vec3::ZERO, Vec3::ZERO);
        mode.begin_stroke();
        for i in 0..240 {
            let x = 0.4 * ((i as f32) * 0.08).sin();
            let y = 0.05 * ((i as f32) * 0.31).sin();
            let at = Vec3::new(x, y, 1.0).normalize();
            let Some((hit, _)) = mode.raycast(at * 3.0, -at) else { continue };
            let raw = mode.vol.normal_upto(hit, mode.vol.levels() - 1, true);
            mode.dab(hit, 0.3, px, 0.7, 0.5, false);
            if last != Vec3::ZERO {
                let a = last.angle_between(mode.stroke_n).to_degrees();
                turn += a;
                worst = worst.max(a);
                raw_turn += last_raw.angle_between(raw).to_degrees();
                n += 1;
            }
            last = mode.stroke_n;
            last_raw = raw;
        }
        mode.end_stroke();
        let (mean, raw_mean) = (turn / n as f32, raw_turn / n as f32);
        eprintln!("{:.2} ms per dab of radius 0.3", mode.last_ms / mode.last_dabs as f64);
        eprintln!("per dab the brush turned {mean:.2} deg on average ({worst:.2} at most); the normal under it turned {raw_mean:.2}");
        assert!(mean < raw_mean / 3.0 && worst < 6.0);
    }
}

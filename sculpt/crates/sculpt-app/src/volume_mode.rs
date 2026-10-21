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
use sculpt_core::generator::level_volume::{CarryDone, LevelVolume};

/// Levels in a new volume: level 0 and five finer, the finest 32 times finer.
pub const LEVELS: usize = 6;
/// The finest step a brush works at, in pixels on screen at the brush.
const PIXELS_PER_STEP: f32 = 1.0;
/// The most lattice steps across a brush's radius: bigger brushes work a level coarser.
const MAX_STEPS_PER_RADIUS: f32 = 48.0;
/// Finer blocks a carrying job takes at most: one job runs at a time, on its own threads, while the frames go on.
const CARRY_JOB: usize = 384;

pub struct VolumeMode {
    pub vol: LevelVolume,
    /// Bumped when `vol` is replaced, so the viewport rebuilds its GPU copy.
    pub generation: u64,
    /// The level the stroke under way works at.
    pub level: usize,
    /// Show levels up to this one (a distant view leaves the finer ones out).
    pub cap: usize,
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
            generation: 1,
            level: 0,
            cap,
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
        if !self.chosen {
            self.level = self.level_for(radius, px);
            self.chosen = true;
        }
        let along = if self.stroke_n == Vec3::ZERO { self.vol.normal_upto(at, self.level, true) } else { self.stroke_n };
        if along == Vec3::ZERO {
            return;
        }
        let s = strength.clamp(0.0, 1.0);
        let height = radius * if carve { 0.015 + 0.06 * s } else { 0.015 + 0.08 * s };
        let Some((n, _)) = self.vol.plane_dab(self.level, at, along, radius, height, hardness, 0.1 + 0.3 * s, carve) else { return };
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
        let t = Instant::now();
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
            self.finish_stroke();
        }
        self.last_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    fn finish_stroke(&mut self) {
        self.ending = false;
        self.ended_specks = Some(self.vol.end_stroke().specks);
    }

    /// Finish a stroke whose detail is still being carried, now: before anything that needs the volume settled (a
    /// new stroke, undo, saving).
    pub fn settle(&mut self) {
        if self.ending {
            self.finish_job();
            self.finish_stroke();
        }
    }

    /// Whether frames should keep coming to finish carrying.
    pub fn busy(&self) -> bool {
        self.ending || self.carrying()
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
        self.settle();
        self.vol.begin_stroke();
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
    }

    /// The pen is up: the stroke ends once the detail it queued is all carried, frames going on meanwhile (see
    /// [`Self::take_specks`]).
    pub fn end_stroke(&mut self) {
        self.stroke_n = Vec3::ZERO;
        self.chosen = false;
        if self.vol.in_stroke() {
            self.ending = true;
        }
        if !self.carrying() {
            self.settle();
        }
    }

    pub fn undo(&mut self) -> bool {
        self.settle();
        self.stroke_n = Vec3::ZERO;
        self.vol.undo()
    }

    pub fn redo(&mut self) -> bool {
        self.settle();
        self.stroke_n = Vec3::ZERO;
        self.vol.redo()
    }

    /// The surface under a ray: during a stroke, the one it began on at its level (so dabs do not climb what the stroke
    /// builds); otherwise the surface shown.
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
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

//! The Decimate window (a "decimation master"): analyse the surface shown once, then take it down to any triangle count
//! live, see it in the viewport in place of the volume, and export it as OBJ.
//!
//! * **Input** is the surface the user sees: the mesh layers' composite when the volume has mesh layers, else the
//!   volume's own surface at the levels shown (`mesh_layers::extract_all`).
//! * **Analyze** builds a [`Analysis`] on a thread of its own (progress, cancel). The count slider (triangles or % kept)
//!   then extracts from it live, every change while dragging: a pass over the output only.
//! * **Even triangles** `e` in 0..1 is a gradient. The shown mesh is always `even_out(extract(analysis), e)`: edge
//!   flips toward valence 6 and relaxation projected back onto the input (built once per input, off the frame).
//!   While the slider drags, the analysis stays the one already built and the evening alone follows `e` (fewer
//!   iterations on big outputs, to stay interactive): the bridge. When it is let go, the analysis built at `e` (rounded
//!   to 0.05) is used: from the cache at once if it was built before, else built in the background and switched to when
//!   ready. Relaxing on top stays, at strength `e`, so the drag and the exact result look alike and `e` evens vertex
//!   spacing as well as choosing which edges go.
//! * Closed/manifold and the surface error are measured off the frame when a slider is let go.
//! * **Tessellate + displace** previews the final asset: quick UVs on the output, a height map baked from the input
//!   (low resolution while a slider drags, the size set when let go), and the output split `level`² per triangle and
//!   displaced by it, shown in place of the output. **Normal map** shades the low or tessellated mesh from an
//!   object-space normal map baked with the height map.
//! * Closing the window drops the preview (and the analyses) and shows the volume again.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Instant;

use eframe::egui_wgpu;
use egui::{RichText, Ui};
use sculpt_core::decimate::{
    Analysis, AnalyzeOptions, Baker, ManifoldReport, Maps, Progress, Surface, SurfaceError, Target, TriMesh, UvMesh, bad_faces, check_manifold, even_out,
    quick_uv,
    surface_error, tessellate,
};
use glam::Vec2;

use crate::app::{SculptApp, fmt_count};
use crate::panels::{prop, section};
use crate::volume_mode::VolumeMode;

/// Analyses kept for evenness values let go before (each is several times the input's size).
const CACHE: usize = 3;
/// Map sizes offered, and the size baked while a slider drags.
const MAP_SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const LIVE_MAP: usize = 256;
/// Height of the panel's main buttons.
const BUTTON_H: f32 = 24.0;
/// The tessellated preview is held to about this many triangles (the level drops on big outputs).
const TESS_BUDGET: usize = 3_000_000;
/// Outputs bigger than this are not previewed (UVs and tessellation run on the frame).
const PREVIEW_MAX: usize = 1_000_000;

/// Points sampled on each mesh for the surface error.
const ERROR_SAMPLES: usize = 100_000;

/// `--decimate KEEP_PERCENT,EVENNESS[,wire][,source][,dock]`: open the window, analyse the startup mesh at that evenness
/// and show it at that share kept (for `--screenshot`); `source` shows the input instead, `dock` docks the window.
#[derive(Clone, Copy, Debug)]
pub struct Script {
    pub keep: f32,
    pub evenness: f32,
    pub wire: bool,
    pub source: bool,
    pub dock: bool,
}

impl Script {
    pub fn parse(s: &str) -> Option<Script> {
        let mut it = s.split(',').map(str::trim);
        let keep: f32 = it.next()?.parse().ok()?;
        let evenness: f32 = match it.next() {
            Some(e) => e.parse().ok()?,
            None => 0.0,
        };
        let flags: Vec<&str> = it.collect();
        let has = |f: &[&str]| flags.iter().any(|x| f.contains(x));
        Some(Script {
            keep: keep.clamp(0.01, 100.0),
            evenness: evenness.clamp(0.0, 1.0),
            wire: has(&["wire", "1", "on", "true"]),
            source: has(&["source"]),
            dock: has(&["dock"]),
        })
    }
}

/// An analysis being built off the frame, at evenness `key / 20`.
struct Job {
    key: i32,
    progress: Arc<Progress>,
    rx: Receiver<Option<Analysis>>,
}

/// Maps for output `generation` at `res`, baked off the frame (with their bake time).
struct BakeJob {
    generation: u64,
    res: usize,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Option<(Maps, f32)>>,
}

/// Closed/manifold and the surface error of output `generation`, measured off the frame.
struct Check {
    generation: u64,
    rx: Receiver<(ManifoldReport, SurfaceError, f32)>,
}

pub struct Decimate {
    pub open: bool,
    rs: Option<egui_wgpu::RenderState>,
    script: Option<Script>,
    script_started: bool,
    /// The surface analysed, where it came from, and how long taking it took.
    input: Option<Arc<TriMesh>>,
    input_from: &'static str,
    input_ms: f32,
    /// The input ready for projection (built off the frame after Analyze), and its build while under way.
    surface: Option<Arc<Surface>>,
    surface_rx: Option<Receiver<Surface>>,
    /// Finished analyses by evenness key (`round(e * 20)`), most recently used last.
    cache: Vec<(i32, Arc<Analysis>)>,
    /// The analysis the output comes from.
    current: Option<(i32, Arc<Analysis>)>,
    job: Option<Job>,
    /// Count the target in triangles (else as a share kept).
    by_count: bool,
    triangles: usize,
    keep: f32,
    evenness: f32,
    /// Lock every border vertex (open meshes keep their outline); part of every analysis, so changing it drops them.
    freeze_borders: bool,
    pub wire: bool,
    /// The extraction at the target (before relaxing), the step it is at, and the analysis key it is of.
    base: Option<(i32, usize, TriMesh)>,
    /// What is shown: the extraction relaxed.
    result: Option<Arc<TriMesh>>,
    generation: u64,
    extract_ms: f32,
    relax_ms: f32,
    /// Show the input (the source surface) in place of the output, to compare.
    show_source: bool,
    /// The controls sit in a panel beside the right dock instead of a floating window.
    pub docked: bool,
    /// Tint thin and knotted triangles of the output red, and which they are.
    show_bad: bool,
    bad: Arc<Vec<bool>>,
    bad_count: usize,
    check: Option<Check>,
    /// The last measurement, and the output generation it is of.
    measured: Option<(u64, ManifoldReport, SurfaceError, f32)>,
    /// The output changed since the GPU's copy.
    gpu_dirty: bool,
    /// The export dialog's path while it is open.
    export: Option<String>,
    note: String,
    /// Tessellate + displace preview: on, normal map on, level, map size, strength.
    displace: bool,
    nmap: bool,
    /// The maps changed since the GPU's copy.
    map_dirty: bool,
    level: usize,
    map_res: usize,
    strength: f32,
    baker: Option<Arc<Baker>>,
    /// UVs and maps of output `generation` (maps at the size given), the tessellated mesh shown, and its timings.
    uv: Option<(u64, Arc<UvMesh>)>,
    bake_job: Option<BakeJob>,
    uv_ms: f32,
    bake_ms: f32,
    maps: Option<(u64, Maps)>,
    tess: Option<(Arc<TriMesh>, Arc<Vec<[Vec2; 3]>>)>,
    tess_note: String,
}

fn key_of(e: f32) -> i32 {
    (e.clamp(0.0, 1.0) * 20.0).round() as i32
}

fn evenness_of(key: i32) -> f32 {
    key as f32 / 20.0
}

/// Relaxation passes at evenness `e`: fewer while a slider drags a big output.
fn relax_iterations(e: f32, triangles: usize, live: bool) -> usize {
    if e <= 0.0 {
        return 0;
    }
    let full = (e * 3.0).ceil() as usize;
    if live && triangles > 1_000_000 { full.min(1) } else { full }
}

/// The surface the user sees, as one triangle mesh: the mesh layers' composite if the volume has mesh layers, else
/// the volume's surface at the levels shown. Settles the volume first (strokes ending, layers refitting).
pub fn take_input(mode: &mut VolumeMode) -> Result<(TriMesh, &'static str), String> {
    mode.settle();
    if let Some(stack) = mode.mesh.as_mut() {
        stack.finish(&mode.vol);
    }
    if let Some(ml) = mode.mesh_layers() {
        return Ok((TriMesh::from_poly(&ml.doc.export_mesh(true)), "mesh layers"));
    }
    let e = sculpt_core::mesh_layers::extract_all(&mode.vol, mode.cap).map_err(|e| e.to_string())?;
    Ok((TriMesh::from_poly(&e.mesh), "volume"))
}

impl Decimate {
    pub fn new(rs: Option<egui_wgpu::RenderState>, script: Option<Script>) -> Decimate {
        Decimate {
            open: false,
            rs,
            script,
            script_started: false,
            input: None,
            input_from: "",
            input_ms: 0.0,
            surface: None,
            surface_rx: None,
            cache: Vec::new(),
            current: None,
            job: None,
            by_count: false,
            triangles: 0,
            keep: 10.0,
            evenness: 0.0,
            freeze_borders: false,
            wire: true,
            base: None,
            result: None,
            generation: 0,
            extract_ms: 0.0,
            relax_ms: 0.0,
            show_source: false,
            docked: false,
            show_bad: true,
            bad: Arc::new(Vec::new()),
            bad_count: 0,
            check: None,
            measured: None,
            gpu_dirty: false,
            export: None,
            note: String::new(),
            displace: false,
            nmap: false,
            map_dirty: false,
            level: 8,
            map_res: 1024,
            strength: 1.0,
            baker: None,
            uv: None,
            bake_job: None,
            uv_ms: 0.0,
            bake_ms: 0.0,
            maps: None,
            tess: None,
            tess_note: String::new(),
        }
    }

    /// The decimated mesh is shown in place of the volume (and sculpting waits).
    pub fn previewing(&self) -> bool {
        self.open && (self.result.is_some() || self.job.is_some())
    }

    /// Work still under way that a screenshot should wait for.
    pub fn busy(&self) -> bool {
        self.job.is_some() || self.bake_job.is_some() || self.surface_rx.is_some() || self.check.is_some() || self.gpu_dirty || (self.script.is_some() && !self.script_started)
    }

    fn target(&self) -> Target {
        if self.by_count { Target::Triangles(self.triangles) } else { Target::KeepPercent(self.keep) }
    }

    /// Take the surface shown and analyse it at the evenness set (the analyses of an earlier surface are dropped).
    fn analyze(&mut self, mode: &mut VolumeMode) {
        let t = Instant::now();
        match take_input(mode) {
            Ok((mesh, from)) if !mesh.triangles.is_empty() => {
                self.input_ms = t.elapsed().as_secs_f32() * 1e3;
                let mesh = Arc::new(mesh);
                self.input = Some(mesh.clone());
                self.surface = None;
                self.baker = None;
                self.uv = None;
                self.cancel_bake();
                self.maps = None;
                self.tess = None;
                let (tx, rx) = channel();
                let spawned = std::thread::Builder::new().name("decimate surface".into()).spawn(move || {
                    let _ = tx.send(Surface::new(mesh));
                });
                self.surface_rx = spawned.ok().map(|_| rx);
                self.input_from = from;
                self.cache.clear();
                self.current = None;
                self.base = None;
                self.result = None;
                self.measured = None;
                self.check = None;
                self.start_job(key_of(self.evenness));
            }
            Ok(_) => self.note = "Nothing to decimate: the surface is empty".into(),
            Err(e) => self.note = format!("Could not take the surface: {e}"),
        }
    }

    /// Build the analysis at evenness `key` off the frame (one at a time: a running one is stopped).
    fn start_job(&mut self, key: i32) {
        self.cancel_job();
        let Some(input) = self.input.clone() else { return };
        let progress = Arc::new(Progress::default());
        let (tx, rx) = channel();
        let p = progress.clone();
        let freeze_borders = self.freeze_borders;
        let spawned = std::thread::Builder::new().name("decimate analysis".into()).spawn(move || {
            let opts = AnalyzeOptions { evenness: evenness_of(key), freeze_borders, ..Default::default() };
            let _ = tx.send(Analysis::build(&input, &opts, Some(&p)));
        });
        match spawned {
            Ok(_) => self.job = Some(Job { key, progress, rx }),
            Err(e) => self.note = format!("Could not start the analysis: {e}"),
        }
    }

    fn cancel_job(&mut self) {
        if let Some(j) = self.job.take() {
            j.progress.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn cached(&mut self, key: i32) -> Option<Arc<Analysis>> {
        let i = self.cache.iter().position(|(k, _)| *k == key)?;
        let entry = self.cache.remove(i);
        let a = entry.1.clone();
        self.cache.push(entry);
        Some(a)
    }

    fn remember(&mut self, key: i32, a: Arc<Analysis>) {
        self.cache.retain(|(k, _)| *k != key);
        self.cache.push((key, a));
        while self.cache.len() > CACHE {
            // The oldest that is not shown.
            let cur = self.current.as_ref().map(|c| c.0);
            match self.cache.iter().position(|(k, _)| Some(*k) != cur) {
                Some(i) => {
                    self.cache.remove(i);
                }
                None => break,
            }
        }
    }

    /// Show analysis `a` (at `key`): extracted again at the target.
    fn use_analysis(&mut self, key: i32, a: Arc<Analysis>) {
        if !self.by_count && self.triangles == 0 {
            self.triangles = Target::KeepPercent(self.keep).triangles(a.input_triangles());
        }
        self.current = Some((key, a));
        self.refresh(true, false);
    }

    /// Make the output again: extract at the target (`extract`, else keep the extraction) and relax at the evenness
    /// set. `live` while a slider drags: fewer relax passes, and nothing measured.
    fn refresh(&mut self, extract: bool, live: bool) {
        let Some((key, a)) = self.current.clone() else { return };
        if self.by_count {
            self.triangles = self.triangles.clamp(a.min_triangles(), a.input_triangles());
        }
        let step = a.step_for(self.target());
        if extract || self.base.as_ref().is_none_or(|b| b.0 != key || b.1 != step) {
            let t = Instant::now();
            let d = a.extract_step(step);
            self.extract_ms = t.elapsed().as_secs_f32() * 1e3;
            self.base = Some((key, d.step, d.mesh));
        }
        let base = &self.base.as_ref().unwrap().2;
        let t = Instant::now();
        let mut out = base.clone();
        let iterations = relax_iterations(self.evenness, out.triangles.len(), live);
        let flips = !live || out.triangles.len() <= 1_000_000;
        even_out(&mut out, self.surface.as_deref(), self.evenness, iterations, flips);
        self.relax_ms = t.elapsed().as_secs_f32() * 1e3;
        let bad = bad_faces(&out);
        self.bad_count = bad.iter().filter(|x| **x).count();
        self.bad = Arc::new(bad);
        self.result = Some(Arc::new(out));
        self.generation += 1;
        self.gpu_dirty = true;
        self.preview(live);
        if !live {
            self.start_check();
        }
    }

    /// The tessellate + displace / normal map preview of the output. UVs are made again only when the output changed;
    /// a small map is baked at once when there is none for this output, and the size set is baked off the frame
    /// (`live`: not, while a slider drags) and swapped in when ready.
    fn preview(&mut self, live: bool) {
        self.tess = None;
        self.gpu_dirty = true;
        if !self.displace && !self.nmap {
            self.cancel_bake();
            return;
        }
        let (Some(out), Some(baker)) = (self.result.clone(), self.baker.clone()) else {
            self.tess_note = "Waiting for the input surface…".into();
            return;
        };
        if out.triangles.len() > PREVIEW_MAX {
            self.cancel_bake();
            self.tess_note = format!("The preview is for outputs up to {} triangles", fmt_count(PREVIEW_MAX));
            return;
        }
        let g = self.generation;
        if self.uv.as_ref().is_none_or(|u| u.0 != g) {
            let t = Instant::now();
            self.uv = Some((g, Arc::new(quick_uv(&out, 4.0 / 1024.0))));
            self.uv_ms = t.elapsed().as_secs_f32() * 1e3;
            self.maps = None;
            self.cancel_bake();
        }
        let uv = self.uv.as_ref().unwrap().1.clone();
        let max_dist = 1.5 * mean_edge(&out);
        if self.maps.as_ref().is_none_or(|m| m.0 != g) {
            let t = Instant::now();
            let Some(m) = baker.bake(&uv, LIVE_MAP.min(self.map_res), max_dist, None) else { return };
            self.bake_ms = t.elapsed().as_secs_f32() * 1e3;
            self.maps = Some((g, m));
            self.map_dirty = true;
        }
        let res = self.map_res;
        if live {
            self.cancel_bake();
        } else if self.maps.as_ref().is_some_and(|m| m.1.res != res) && self.bake_job.as_ref().is_none_or(|j| j.generation != g || j.res != res) {
            self.cancel_bake();
            let cancel = Arc::new(AtomicBool::new(false));
            let (tx, rx) = channel();
            let c = cancel.clone();
            let spawned = std::thread::Builder::new().name("decimate bake".into()).spawn(move || {
                let t = Instant::now();
                let m = baker.bake(&uv, res, max_dist, Some(&c));
                let _ = tx.send(m.map(|m| (m, t.elapsed().as_secs_f32() * 1e3)));
            });
            if spawned.is_ok() {
                self.bake_job = Some(BakeJob { generation: g, res, cancel, rx });
            }
        }
        self.build_view();
    }

    /// The mesh shown for the preview from the UVs and maps there are: tessellated and displaced, or the output split
    /// along its UV seams (for the normal map alone).
    fn build_view(&mut self) {
        let (Some((_, uv)), Some((_, maps)), Some(out)) = (self.uv.as_ref(), self.maps.as_ref(), self.result.as_ref()) else { return };
        let baking = match self.bake_job.as_ref() {
            Some(j) => format!(" ({}² baking…)", j.res),
            None => String::new(),
        };
        let maps_note = format!("{} charts, map {}²{baking}  ·  UV {:.1} ms, bake {:.0} ms", uv.charts, maps.res, self.uv_ms, self.bake_ms);
        if !self.displace {
            let low = TriMesh { positions: uv.positions.clone(), triangles: uv.triangles.clone() };
            let corners: Vec<[Vec2; 3]> = uv.triangles.iter().map(|t| t.map(|v| uv.uvs[v as usize])).collect();
            self.tess_note = maps_note;
            self.tess = Some((Arc::new(low), Arc::new(corners)));
        } else {
            let fit = ((TESS_BUDGET as f32 / out.triangles.len().max(1) as f32).sqrt() as usize).max(1);
            let level = self.level.min(fit);
            let t = Instant::now();
            let tess = tessellate(uv, maps, level, self.strength);
            self.tess_note = format!(
                "{} triangles (level {level}{})  ·  {maps_note}, tessellate {:.0} ms",
                fmt_count(tess.mesh.triangles.len()),
                if level < self.level { ", held down for size" } else { "" },
                t.elapsed().as_secs_f32() * 1e3,
            );
            self.tess = Some((Arc::new(tess.mesh), Arc::new(tess.corner_uvs)));
        }
        self.gpu_dirty = true;
    }

    fn cancel_bake(&mut self) {
        if let Some(j) = self.bake_job.take() {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The mesh to show and its corner UVs: the input when the source is asked for, the preview when one is on (the tessellated mesh, or the output split
    /// along its UV seams for the normal map), else the output.
    fn shown(&self) -> Option<(Arc<TriMesh>, Option<Arc<Vec<[Vec2; 3]>>>)> {
        if self.show_source && self.result.is_some() {
            return self.input.as_ref().map(|m| (m.clone(), None));
        }
        match self.tess.as_ref() {
            Some((m, uv)) if self.displace || self.nmap => Some((m.clone(), Some(uv.clone()))),
            _ => self.result.as_ref().map(|m| (m.clone(), None)),
        }
    }

    /// Measure the output off the frame: closed/manifold, and the surface error against the input.
    fn start_check(&mut self) {
        let (Some(input), Some(out)) = (self.input.clone(), self.result.clone()) else { return };
        let (tx, rx) = channel();
        let generation = self.generation;
        let spawned = std::thread::Builder::new().name("decimate check".into()).spawn(move || {
            let t = Instant::now();
            let report = check_manifold(out.positions.len(), &out.triangles);
            let err = surface_error(&input, &out, ERROR_SAMPLES);
            let _ = tx.send((report, err, t.elapsed().as_secs_f32() * 1e3));
        });
        self.check = spawned.ok().map(|_| Check { generation, rx });
    }

    /// Take in finished jobs.
    fn poll(&mut self) {
        if let Some(rx) = self.surface_rx.as_ref() {
            match rx.try_recv() {
                Ok(s) => {
                    self.surface_rx = None;
                    let s = Arc::new(s);
                    self.baker = Some(Arc::new(Baker::new(s.clone())));
                    self.surface = Some(s);
                    if (self.displace || self.nmap) && self.evenness <= 0.0 && self.result.is_some() {
                        self.preview(false);
                    }
                    // Even again, now projected onto the input.
                    if self.evenness > 0.0 && self.base.is_some() {
                        self.refresh(false, false);
                    }
                }
                Err(TryRecvError::Disconnected) => self.surface_rx = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some(j) = self.job.as_ref() {
            match j.rx.try_recv() {
                Ok(Some(a)) => {
                    let key = j.key;
                    self.job = None;
                    let a = Arc::new(a);
                    self.remember(key, a.clone());
                    // Switched to if it is still the evenness wanted (or nothing is shown yet).
                    if key == key_of(self.evenness) || self.current.is_none() {
                        self.use_analysis(key, a);
                    }
                }
                Ok(None) | Err(TryRecvError::Disconnected) => {
                    self.job = None;
                    self.note = "Analysis cancelled".into();
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some(j) = self.bake_job.as_ref() {
            match j.rx.try_recv() {
                Ok(Some((m, ms))) => {
                    let g = j.generation;
                    self.bake_job = None;
                    if self.uv.as_ref().is_some_and(|u| u.0 == g) {
                        self.maps = Some((g, m));
                        self.bake_ms = ms;
                        self.map_dirty = true;
                        self.build_view();
                    }
                }
                Ok(None) | Err(TryRecvError::Disconnected) => self.bake_job = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some(c) = self.check.as_ref() {
            match c.rx.try_recv() {
                Ok((report, err, ms)) => {
                    let g = c.generation;
                    self.check = None;
                    if g == self.generation {
                        self.measured = Some((g, report, err, ms));
                    }
                }
                Err(TryRecvError::Disconnected) => self.check = None,
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    /// The evenness slider let go: use the analysis at this evenness if there is one, else build it (the one shown,
    /// relaxed, bridges until then).
    fn evenness_released(&mut self) {
        let key = key_of(self.evenness);
        if self.current.as_ref().is_some_and(|c| c.0 == key) {
            self.refresh(false, false);
        } else if let Some(a) = self.cached(key) {
            self.use_analysis(key, a);
        } else {
            self.refresh(false, false);
            if self.job.as_ref().is_none_or(|j| j.key != key) {
                self.start_job(key);
            }
        }
    }

    /// Drop the preview and everything analysed: the volume shows again.
    fn close(&mut self) {
        self.cancel_job();
        self.open = false;
        self.input = None;
        self.surface = None;
        self.surface_rx = None;
        self.baker = None;
        self.uv = None;
        self.cancel_bake();
        self.maps = None;
        self.tess = None;
        self.cache.clear();
        self.current = None;
        self.base = None;
        self.result = None;
        self.check = None;
        self.measured = None;
        self.export = None;
        self.gpu_dirty = false;
    }

    fn export_obj(&mut self, path: &str) -> String {
        let Some(out) = self.result.as_ref() else { return "Nothing to export".into() };
        let t = Instant::now();
        match sculpt_core::io::obj::write(std::path::Path::new(path), &out.to_poly()) {
            Ok(()) => format!("Exported {} triangles to {} ({:.2}s)", fmt_count(out.triangles.len()), crate::app::file_name(path), t.elapsed().as_secs_f32()),
            Err(e) => format!("Export failed: {path}: {e}"),
        }
    }

    fn controls(&mut self, ui: &mut Ui, mode: &mut VolumeMode) {
        let weak = ui.visuals().weak_text_color();
        let fs = ui.style().text_styles[&egui::TextStyle::Body].size;
        let note = |ui: &mut Ui, text: &str| {
            if !text.is_empty() {
                ui.add(egui::Label::new(RichText::new(text).small().color(weak)).wrap());
            }
        };
        // Analyse (again), or the analysis under way.
        section(ui, fs, "decimate_source", "SOURCE", |ui| {
            if let Some(j) = self.job.as_ref() {
                let what = if self.current.is_some() { format!("Building evenness {:.2}", evenness_of(j.key)) } else { "Analysing".to_string() };
                let frac = j.progress.fraction();
                ui.add(egui::ProgressBar::new(frac).desired_height(18.0).text(RichText::new(format!("{what}…  {:.0}%", frac * 100.0)).small()));
                ui.ctx().request_repaint();
                ui.add_space(2.0);
                if ui.add(egui::Button::new("Cancel analysis").min_size(egui::vec2(0.0, BUTTON_H))).clicked() {
                    self.cancel_job();
                    self.note = "Analysis cancelled".into();
                }
            } else {
                let label = if self.input.is_some() { "Analyze again" } else { "Analyze" };
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new(label).min_size(egui::vec2(110.0, BUTTON_H)))
                        .on_hover_text("Take the surface shown and work out every collapse once; the sliders then decimate it live")
                        .clicked()
                    {
                        self.analyze(mode);
                    }
                    ui.add_space(6.0);
                    let r = ui.checkbox(&mut self.freeze_borders, "Freeze borders").on_hover_text("Keep every border vertex where it is (open meshes keep their outline). The volume's surface is closed, so it rarely matters.");
                    // The analyses were built with the other setting: build again at the evenness set.
                    if r.changed() && self.input.is_some() {
                        self.cache.clear();
                        // What is shown stays as a bridge only (no key matches it).
                        if let Some(c) = self.current.as_mut() {
                            c.0 = -1;
                        }
                        self.start_job(key_of(self.evenness));
                    }
                });
            }
            if self.current.is_none() {
                note(ui, &self.note);
            }
        });
        let Some((_, a)) = self.current.clone() else {
            return;
        };

        // The target: triangles or a share kept.
        let (input, floor) = (a.input_triangles(), a.min_triangles());
        let shown = self.result.as_ref().map_or(0, |r| r.triangles.len());
        section(ui, fs, "decimate_target", "TARGET", |ui| self.target_rows(ui, input, floor, shown));
        section(ui, fs, "decimate_display", "DISPLAY", |ui| self.display_rows(ui, shown));
        section(ui, fs, "decimate_preview", "RENDER PREVIEW", |ui| self.preview_rows(ui));
        section(ui, fs, "decimate_result", "RESULT", |ui| self.result_rows(ui, &a, input));

        // Export and close.
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(2.0);
        let out = self.result.as_ref().map_or(0, |r| r.triangles.len());
        ui.horizontal(|ui| {
            if ui.add_enabled(out > 0, egui::Button::new(RichText::new("Export OBJ…").strong()).min_size(egui::vec2(110.0, BUTTON_H))).clicked() {
                self.export = Some(self.export.clone().unwrap_or_else(|| "decimated.obj".into()));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new("Close").min_size(egui::vec2(70.0, BUTTON_H))).on_hover_text("Drop the preview and show the volume again").clicked() {
                    self.open = false;
                }
            });
        });
    }

    /// The target (a share kept or a triangle count) and evenness.
    fn target_rows(&mut self, ui: &mut Ui, input: usize, floor: usize, shown: usize) {
        let mut by_count = self.by_count;
        prop(ui, "Count by", |ui| {
            ui.selectable_value(&mut by_count, false, "% kept");
            ui.selectable_value(&mut by_count, true, "Triangles");
        });
        if by_count != self.by_count {
            self.by_count = by_count;
            // Same output, given the other way.
            if by_count {
                self.triangles = shown.max(floor);
            } else {
                self.keep = (100.0 * shown as f32 / input.max(1) as f32).clamp(0.01, 100.0);
            }
        }
        let (mut live, mut release) = (false, false);
        let r = if self.by_count {
            prop(ui, "Triangles", |ui| ui.add(egui::Slider::new(&mut self.triangles, floor.max(1)..=input).logarithmic(true)))
        } else {
            let lo = (100.0 * floor as f32 / input.max(1) as f32).max(0.01);
            prop(ui, "Keep", |ui| ui.add(egui::Slider::new(&mut self.keep, lo..=100.0).logarithmic(true).suffix(" %").max_decimals(2)))
        };
        live |= r.changed() && r.dragged();
        release |= r.drag_stopped() || (r.changed() && !r.dragged());
        if release {
            self.refresh(true, false);
        } else if live {
            self.refresh(true, true);
        }

        let r = prop(ui, "Evenness", |ui| ui.add(egui::Slider::new(&mut self.evenness, 0.0..=1.0).max_decimals(2)))
            .on_hover_text("0 keeps detail where the shape bends (fewer triangles where it is flat); 1 evens the triangles out. Dragging relaxes the result live; letting go builds the analysis at this evenness (kept for next time).");
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            self.evenness_released();
        } else if r.changed() {
            self.refresh(false, true);
        }
    }

    /// What the viewport shows: the output or its source, wireframe, bad triangles.
    fn display_rows(&mut self, ui: &mut Ui, shown: usize) {
        let src = self.input.as_ref().map_or(0, |m| m.triangles.len());
        let mut show_source = self.show_source;
        prop(ui, "Show", |ui| {
            ui.selectable_value(&mut show_source, false, format!("Decimated {}", fmt_count(shown)))
                .on_hover_text("The decimated output (with the preview, when one is on)");
            ui.selectable_value(&mut show_source, true, format!("Source {}", fmt_count(src)))
                .on_hover_text("The surface it was made from, to compare (same view, same wireframe)");
        });
        if show_source != self.show_source {
            self.show_source = show_source;
            self.gpu_dirty = true;
        }
        prop(ui, "", |ui| ui.checkbox(&mut self.wire, "Wireframe"));
        let weak = ui.visuals().weak_text_color();
        let r = prop(ui, "", |ui| {
            let r = ui.checkbox(&mut self.show_bad, "Bad triangles");
            if self.result.is_some() {
                let found = RichText::new(format!("{} found", fmt_count(self.bad_count))).small();
                ui.label(if self.bad_count > 0 && self.show_bad { found.color(egui::Color32::from_rgb(230, 110, 100)) } else { found.color(weak) });
            }
            r
        });
        if r.on_hover_text("Tint red the output's triangles with an angle under 10° or an edge under a tenth of the mean (knots); not shown on the tessellated preview").changed() {
            self.gpu_dirty = true;
        }
    }

    /// Tessellate + displace and the normal map: roughly how the asset renders.
    fn preview_rows(&mut self, ui: &mut Ui) {
        let weak = ui.visuals().weak_text_color();
        let r = prop(ui, "", |ui| ui.checkbox(&mut self.displace, "Tessellate + displace")).on_hover_text(
            "Quick UVs on the output, a height map baked from the input, and the output split and displaced by it: roughly how the asset looks rendered with displacement",
        );
        let r2 = prop(ui, "", |ui| ui.checkbox(&mut self.nmap, "Normal map"))
            .on_hover_text("Shade from a normal map baked from the input (on the tessellated mesh, or on the output when tessellation is off)");
        if r.changed() || r2.changed() {
            self.preview(false);
        }
        if self.displace || self.nmap {
            ui.add_space(2.0);
            let r = ui.add_enabled_ui(self.displace, |ui| prop(ui, "Level", |ui| ui.add(egui::Slider::new(&mut self.level, 1..=32))).on_hover_text("Each triangle is split level × level")).inner;
            let r2 = ui.add_enabled_ui(self.displace, |ui| prop(ui, "Strength", |ui| ui.add(egui::Slider::new(&mut self.strength, 0.0..=2.0).max_decimals(2)))).inner;
            let mut res = self.map_res;
            prop(ui, "Map size", |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for s in MAP_SIZES {
                    ui.selectable_value(&mut res, s, if s >= 1024 { format!("{}K", s / 1024) } else { s.to_string() });
                }
            });
            let res_changed = res != self.map_res;
            self.map_res = res;
            if r.changed() || r2.changed() {
                self.preview(r.dragged() || r2.dragged());
            } else if r.drag_stopped() || r2.drag_stopped() || res_changed {
                self.preview(false);
            }
            if !self.tess_note.is_empty() {
                ui.add(egui::Label::new(RichText::new(&self.tess_note).small().color(weak)).wrap());
            }
        }
    }

    /// What came out: counts, the mesh's shape and error, then the timings.
    fn result_rows(&mut self, ui: &mut Ui, a: &Analysis, input: usize) {
        let weak = ui.visuals().weak_text_color();
        let out = self.result.as_ref().map_or(0, |r| r.triangles.len());
        stat(ui, "Input", RichText::new(fmt_count(input)).strong(), Some(&format!("triangles · {}", self.input_from)));
        stat(ui, "Output", RichText::new(fmt_count(out)).strong(), Some(&format!("triangles · {:.2}%", 100.0 * out as f64 / input.max(1) as f64)));
        match self.measured.as_ref().filter(|m| m.0 == self.generation) {
            Some((_, report, err, _)) => {
                let shape = match (report.closed(), report.manifold()) {
                    (true, true) => "Closed, manifold".to_string(),
                    (false, true) => format!("Manifold, {} border edges", fmt_count(report.border_edges)),
                    _ => format!("Not manifold: {} edges, {} vertices", fmt_count(report.nonmanifold_edges), fmt_count(report.nonmanifold_vertices)),
                };
                stat(ui, "Surface", RichText::new(shape), None);
                stat(ui, "Error", RichText::new(format!("{:.3}% mean  ·  {:.3}% max", err.mean_rel() * 100.0, err.max_rel() * 100.0)), None)
                    .on_hover_text("Distance from the input, as a share of its bounding box diagonal");
            }
            None if self.check.is_some() => {
                stat(ui, "Surface", RichText::new("Measuring…").color(weak), None);
            }
            None => {
                stat(ui, "Surface", RichText::new("Measured when the slider is let go").color(weak), None);
            }
        }

        // Timings, quieter.
        ui.add_space(4.0);
        let relaxed = if self.relax_ms > 0.05 { format!(" + relax {:.1} ms", self.relax_ms) } else { String::new() };
        let lag = if key_of(self.evenness) != key_of(a.evenness()) { "  (relaxed toward the evenness set until its own is built)" } else { "" };
        let small = |ui: &mut Ui, s: String| {
            ui.add(egui::Label::new(RichText::new(s).small().color(weak)).wrap());
        };
        small(ui, format!("Analysis at {:.2}: {:.0} ms, {} rounds{lag}", a.evenness(), a.stats.ms, a.stats.rounds));
        small(ui, format!("Extract {:.1} ms{relaxed}  ·  surface taken in {:.0} ms", self.extract_ms, self.input_ms));
        if !self.note.is_empty() {
            ui.add_space(2.0);
            small(ui, self.note.clone());
        }
    }

    /// The export dialog (a path, as the app's Open and Save As take one).
    fn export_dialog(&mut self, ctx: &egui::Context) -> Option<String> {
        let path = self.export.as_mut()?;
        let (mut ok, mut cancel) = (false, false);
        egui::Window::new("Export decimated mesh (.obj)").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.label(RichText::new("Path").small().color(ui.visuals().weak_text_color()));
            let r = ui.add(egui::TextEdit::singleline(path).desired_width(380.0));
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                ok = true;
            }
            ui.add_space(6.0);
            ui.allocate_ui_with_layout(egui::vec2(380.0, BUTTON_H), egui::Layout::right_to_left(egui::Align::Center), |ui| {
                cancel = ui.add(egui::Button::new("Cancel").min_size(egui::vec2(70.0, BUTTON_H))).clicked();
                ok |= ui.add(egui::Button::new(RichText::new("Export").strong()).min_size(egui::vec2(70.0, BUTTON_H))).clicked();
            });
        });
        if ok {
            let p = self.export.take().unwrap();
            let status = self.export_obj(&p);
            // Keep the path for the next export.
            self.export = None;
            self.note = status.clone();
            return Some(status);
        }
        if cancel {
            self.export = None;
        }
        None
    }
}

/// A result row: the label column, a value, and a quieter unit after it.
fn stat(ui: &mut Ui, label: &str, value: RichText, unit: Option<&str>) -> egui::Response {
    prop(ui, label, |ui| {
        let r = ui.label(value);
        if let Some(u) = unit {
            ui.label(RichText::new(u).small().color(ui.visuals().weak_text_color()));
        }
        r
    })
}

fn mean_edge(m: &TriMesh) -> f32 {
    let s: f32 = m.triangles.iter().map(|t| (0..3).map(|i| m.positions[t[i] as usize].distance(m.positions[t[(i + 1) % 3] as usize])).sum::<f32>()).sum();
    s / (3 * m.triangles.len().max(1)) as f32
}

/// The menu entry that opens the window.
pub fn menu_item(app: &mut SculptApp, ui: &mut Ui) {
    if ui.button("Decimate…").on_hover_text("Decimate the surface shown and export it (OBJ)").clicked() {
        app.decimate.open = true;
        ui.close();
    }
}

/// The window, once a frame (before the viewport draws): sliders, jobs taken in, the preview sent to the GPU.
pub fn window(app: &mut SculptApp, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let ctx = &ctx;
    let d = &mut app.decimate;
    if let Some(s) = d.script
        && !d.script_started
    {
        d.script_started = true;
        d.open = true;
        d.keep = s.keep;
        d.evenness = s.evenness;
        d.wire = s.wire;
        d.show_source = s.source;
        d.docked = s.dock;
        d.analyze(&mut app.volume);
    }
    d.poll();
    if d.open {
        let (mut open, mut dock) = (true, d.docked);
        if d.docked {
            let fill = app.theme.ui.panel.0;
            egui::Panel::right("decimate_dock")
                .resizable(true)
                .default_size(360.0)
                .min_size(320.0)
                .frame(egui::Frame::side_top_panel(ui.style()).fill(fill).inner_margin(0))
                .show(ui, |ui| {
                    crate::panels::panel_header(ui, app, "DECIMATE", |ui| {
                        if ui.small_button("Close").on_hover_text("Close (the volume shows again)").clicked() {
                            open = false;
                        }
                        if ui.small_button("Float").on_hover_text("Undock into a floating window").clicked() {
                            dock = false;
                        }
                    });
                    let (d, mode) = (&mut app.decimate, &mut app.volume);
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
                            // Fixed: sizing from the available width would feed back into the panel's width.
                            ui.spacing_mut().slider_width = 150.0;
                            d.controls(ui, mode);
                        });
                    });
                });
        } else {
            let mode = &mut app.volume;
            egui::Window::new("Decimate").open(&mut open).default_width(360.0).resizable(false).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Dock").on_hover_text("Dock beside the Layers panel").clicked() {
                            dock = true;
                        }
                    });
                });
                d.controls(ui, mode);
            });
        }
        let d = &mut app.decimate;
        d.docked = dock;
        if !open {
            d.open = false;
        }
        if let Some(status) = d.export_dialog(ctx) {
            app.status = status;
        }
    }
    let d = &mut app.decimate;
    if !d.open && (d.input.is_some() || d.job.is_some()) {
        d.close();
    }
    if d.job.is_some() || d.check.is_some() || d.bake_job.is_some() {
        ctx.request_repaint();
    }

    // The preview on the GPU, in place of the volume.
    let (Some(rs), Some(vp)) = (d.rs.as_ref(), app.viewport.as_mut()) else {
        d.gpu_dirty = false;
        return;
    };
    match (d.previewing(), d.shown()) {
        (true, Some((out, uvs))) => {
            let draw = vp.decimate.get_or_insert_with(|| crate::decimate_draw::DecimateDraw::new(&rs.device, &vp.bgl));
            if d.gpu_dirty {
                let bad = (d.show_bad && !d.show_source && uvs.is_none()).then(|| d.bad.as_slice());
                draw.upload(&rs.device, &rs.queue, &out, uvs.as_deref().map(|u| u.as_slice()), bad);
                d.gpu_dirty = false;
            }
            if d.map_dirty
                && let Some((_, maps)) = d.maps.as_ref()
            {
                draw.upload_map(&rs.device, &rs.queue, maps);
                d.map_dirty = false;
            }
            draw.use_nmap = d.nmap;
            draw.wire = d.wire;
            draw.shown = true;
        }
        _ => {
            d.gpu_dirty = false;
            if let Some(draw) = vp.decimate.as_mut()
                && draw.showing()
            {
                draw.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default ball's surface is one closed manifold mesh, and analysing then extracting it round-trips: the full
    /// count gives the input back, a share kept stays closed and manifold near its count, and evening keeps it so.
    #[test]
    fn ball_analyse_extract_round_trip() {
        let mut mode = VolumeMode::ball(2e5);
        let (input, from) = take_input(&mut mode).expect("the ball's surface");
        assert_eq!(from, "volume");
        assert!(input.triangles.len() > 10_000, "{} triangles", input.triangles.len());
        let r = check_manifold(input.positions.len(), &input.triangles);
        assert!(r.closed() && r.manifold(), "input: {r:?}");

        let a = Analysis::build(&input, &AnalyzeOptions::default(), None).expect("not cancelled");
        assert_eq!(a.input_triangles(), input.triangles.len());
        let full = a.extract(Target::KeepPercent(100.0));
        assert_eq!(full.mesh.triangles.len(), input.triangles.len());
        let r = check_manifold(full.mesh.positions.len(), &full.mesh.triangles);
        assert!(r.closed() && r.manifold(), "full: {r:?}");

        let want = Target::KeepPercent(10.0).triangles(input.triangles.len());
        let mut d = a.extract(Target::KeepPercent(10.0)).mesh;
        let got = d.triangles.len();
        assert!((got as f64 - want as f64).abs() <= want as f64 * 0.02 + 4.0, "kept {got} for {want}");
        let r = check_manifold(d.positions.len(), &d.triangles);
        assert!(r.closed() && r.manifold(), "10%: {r:?}");
        let err = surface_error(&input, &d, 20_000);
        assert!(err.max_rel() < 0.02, "error {err:?}");

        let iterations = relax_iterations(1.0, d.triangles.len(), false);
        let surface = Surface::new(Arc::new(input.clone()));
        even_out(&mut d, Some(&surface), 1.0, iterations, true);
        let r = check_manifold(d.positions.len(), &d.triangles);
        assert!(r.closed() && r.manifold(), "evened: {r:?}");
        assert_eq!(crate::decimate_draw::vertex_normals(&d).len(), d.positions.len());
    }

    #[test]
    fn script_parses() {
        let s = Script::parse("25,0.5,wire").unwrap();
        assert_eq!((s.keep, s.evenness, s.wire), (25.0, 0.5, true));
        let s = Script::parse("10").unwrap();
        assert_eq!((s.keep, s.evenness, s.wire), (10.0, 0.0, false));
        let s = Script::parse("1,1,source,dock").unwrap();
        assert!(s.source && s.dock && !s.wire);
        assert!(Script::parse("x,1").is_none());
        assert_eq!(key_of(0.52), 10);
        assert_eq!(key_of(0.53), 11);
    }
}

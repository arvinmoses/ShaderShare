//! `--demo-volume DIR`: a scripted session in the volume playground, recorded frame by frame from the app's own
//! framebuffer (for a review video; nothing else on screen is captured).
//!
//! The script runs on its own clock, a frame at a time at [`FPS`]: each frame the app applies what the script says
//! (camera, brush, pen samples, level cap, tint, culling, undo), sculpts and draws as it would for a person, and saves
//! the frame before the script moves on. Real timings show in the viewport's readout, so the video plays smoothly
//! however long the recording takes, while the numbers on screen are the live ones.

use std::path::PathBuf;

use egui::{Pos2, Rect};
use glam::Vec3;

use crate::camera::Camera;
use crate::tools::Tool;

pub const FPS: f32 = 30.0;
/// Pen samples per frame along a stroke.
const SAMPLES: usize = 4;

/// A camera pose: what it looks at, from how far, from which direction.
#[derive(Clone, Copy)]
struct Pose {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl Pose {
    /// Looking straight at the ball's surface at direction `d`, from `distance` away.
    fn at(d: Vec3, distance: f32) -> Pose {
        let n = d.normalize();
        Pose { target: n, distance, yaw: n.x.atan2(n.z), pitch: n.y.asin() }
    }

    fn whole(distance: f32, yaw: f32, pitch: f32) -> Pose {
        Pose { target: Vec3::ZERO, distance, yaw, pitch }
    }

    fn lerp(a: Pose, b: Pose, t: f32) -> Pose {
        let s = t * t * (3.0 - 2.0 * t);
        Pose { target: a.target.lerp(b.target, s), distance: (a.distance.ln() + (b.distance.ln() - a.distance.ln()) * s).exp(), yaw: a.yaw + (b.yaw - a.yaw) * s, pitch: a.pitch + (b.pitch - a.pitch) * s }
    }
}

enum Act {
    /// Move the camera to a pose over `dur` seconds.
    Camera(Pose),
    /// A stroke along a polyline of viewport fractions over `dur` seconds.
    Stroke { tool: Tool, size: f32, strength: f32, path: Vec<(f32, f32)> },
    Cap(usize),
    /// Sweep the level cap from one level to another over `dur`.
    CapSweep(usize, usize),
    Tint(bool),
    Cull(bool),
    /// Choosing the level by distance on or off.
    ByDistance(bool),
    /// A new generator layer (rocks), and a change of its parameters (applied to the active generator's).
    AddGenerator,
    Generator(fn(&mut sculpt_core::generator::layer::GeneratorSpec)),
    /// A change of the surface noise.
    Noise(fn(&mut sculpt_core::surface_noise::SurfaceNoise)),
    /// Move the work at this level and finer into a detail layer.
    MakeLayer(usize),
    /// What strokes work on, with a detail layer.
    Target(crate::volume_mode::Target),
    /// A new layer at this level, and showing or hiding a layer.
    AddLayer(usize),
    LayerVisible(usize, bool),
    /// Hold a layer at another level; remove a layer; make a layer active.
    LayerLevel(usize, usize),
    RemoveLayer(usize),
    ActiveLayer(usize),
    /// The active layer's whole mask: 0 hides it everywhere, 1 shows it.
    FillMask(f32),
    /// Bake the surface noise into a detail layer at this level.
    BakeNoise(usize),
    /// Save the volume with its layers and open it again (what a file holds is what is shown).
    SaveOpen,
    Undo,
    Redo,
    /// A caption for `dur` seconds.
    Caption(&'static str),
}

struct Item {
    at: f32,
    dur: f32,
    act: Act,
}

/// What the app should do this frame.
#[derive(Default)]
pub struct Frame {
    pub camera: Option<Camera>,
    /// A stroke starts: its tool, size in pixels, strength and first point.
    pub begin: Option<(Tool, f32, f32, Pos2)>,
    pub samples: Vec<Pos2>,
    pub end: bool,
    pub cap: Option<usize>,
    pub tint: Option<bool>,
    pub keep_detail: Option<bool>,
    pub make_layer: Option<usize>,
    pub target: Option<crate::volume_mode::Target>,
    pub add_layer: Option<usize>,
    pub layer_visible: Option<(usize, bool)>,
    pub layer_level: Option<(usize, usize)>,
    pub remove_layer: Option<usize>,
    pub active_layer: Option<usize>,
    pub fill_mask: Option<f32>,
    pub bake_noise: Option<usize>,
    pub save_open: bool,
    pub cull: Option<bool>,
    pub by_distance: Option<bool>,
    pub add_generator: bool,
    pub noise: Option<fn(&mut sculpt_core::surface_noise::SurfaceNoise)>,
    pub generator: Option<fn(&mut sculpt_core::generator::layer::GeneratorSpec)>,
    pub undo: bool,
    pub redo: bool,
}

pub struct VolumeDemo {
    pub dir: PathBuf,
    items: Vec<Item>,
    frame: u32,
    end: f32,
    /// Whether the stroke item `i` has begun.
    begun: Vec<bool>,
    fired: Vec<bool>,
    /// The frame whose image the app is waiting for.
    pub waiting: Option<u32>,
    pub caption: Option<(String, f32)>,
}

fn line(a: (f32, f32), b: (f32, f32), wiggle: f32, n: usize) -> Vec<(f32, f32)> {
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let w = wiggle * (t * std::f32::consts::TAU * 1.5).sin();
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t + w)
        })
        .collect()
}

impl VolumeDemo {
    pub fn new(dir: PathBuf) -> VolumeDemo {
        let patch = Vec3::new(0.30, 0.34, 0.89);
        let start = Pose::whole(4.4, 0.0, 0.12);
        let mut items = Vec::new();
        let mut at = |t: f32, dur: f32, act: Act| items.push(Item { at: t, dur, act });
        use Act::*;
        at(0.0, 0.0, Camera(start));
        at(0.0, 5.0, Caption("The volume engine on the GPU: a ball whose level 0 alone is 1M triangles, and five finer levels for detail"));
        // Big brushes: coarse form at level 0.
        at(5.0, 10.0, Caption("A big brush works at level 0: broad form. The surface of every block it touched is rebuilt on the GPU"));
        at(5.5, 2.4, Stroke { tool: Tool::ClayBuildup, size: 150.0, strength: 0.9, path: line((0.33, 0.40), (0.62, 0.36), 0.02, 12) });
        at(8.2, 2.4, Stroke { tool: Tool::ClayBuildup, size: 130.0, strength: 0.9, path: line((0.36, 0.55), (0.64, 0.60), 0.02, 12) });
        at(10.9, 2.2, Stroke { tool: Tool::TrimDynamic, size: 120.0, strength: 0.8, path: line((0.40, 0.72), (0.62, 0.70), 0.015, 12) });
        at(13.3, 1.5, Stroke { tool: Tool::ClayBuildup, size: 90.0, strength: 1.0, path: line((0.45, 0.30), (0.50, 0.50), 0.0, 8) });
        // Zoom in, and brushes get finer.
        at(15.0, 3.0, Camera(Pose::at(patch, 1.1)));
        at(15.0, 4.0, Caption("Zoom in: the same brush sizes on screen now work at finer levels, about a lattice step per pixel"));
        at(18.5, 2.0, Stroke { tool: Tool::ClayBuildup, size: 45.0, strength: 1.0, path: line((0.36, 0.42), (0.64, 0.44), 0.03, 12) });
        at(20.7, 2.0, Stroke { tool: Tool::ClayBuildup, size: 45.0, strength: 1.0, path: line((0.36, 0.56), (0.64, 0.58), 0.03, 12) });
        at(23.0, 3.0, Camera(Pose::at(patch, 0.32)));
        at(23.0, 9.0, Caption("Closer still: grooves and dots at levels 4 and 5, 16 to 32 times finer than level 0, only where they are carved"));
        for k in 0..4 {
            let y = 0.38 + 0.08 * k as f32;
            at(26.2 + 1.2 * k as f32, 1.0, Stroke { tool: Tool::TrimDynamic, size: 14.0, strength: 1.0, path: line((0.38, y), (0.62, y + 0.01), 0.004, 10) });
        }
        for k in 0..6 {
            let x = 0.40 + 0.04 * k as f32;
            at(31.2 + 0.35 * k as f32, 0.25, Stroke { tool: Tool::ClayBuildup, size: 9.0, strength: 1.0, path: line((x, 0.69), (x + 0.002, 0.69), 0.0, 2) });
        }
        // Where the detail lives.
        at(34.0, 2.5, Camera(Pose::at(patch, 0.75)));
        at(34.0, 0.0, Tint(true));
        at(34.0, 5.5, Caption("Tinted by level: each level holds blocks only where it was sculpted (the readout counts them)"));
        // Detail shown by level cap.
        at(39.5, 8.0, CapSweep(5, 0));
        at(39.5, 8.0, Caption("Showing fewer levels, as a distant view would: the finest detail goes first, the form stays"));
        at(47.5, 3.0, CapSweep(0, 5));
        at(50.5, 0.0, Cap(5));
        at(50.5, 0.0, Tint(false));
        // A big coarse stroke over the detail.
        at(50.5, 2.5, Camera(Pose::at(patch, 3.0)));
        at(53.0, 7.0, Caption("A big level 0 stroke over the fine work: like clay, it builds over it and buries the grooves and dots under it"));
        at(53.5, 2.5, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.50, 0.40), (0.60, 0.42), 0.0, 8) });
        at(56.5, 3.0, Camera(Pose::at(patch * 1.05, 0.34)));
        // The same stroke on the form under a detail layer.
        at(60.0, 0.0, Undo);
        at(60.0, 2.0, Camera(Pose::at(patch, 3.0)));
        at(60.0, 9.0, Caption("Undone. Now the grooves and dots go into a detail layer (Layers panel), and the big stroke works on the form under it: the layer rides along"));
        at(61.5, 0.0, MakeLayer(4));
        at(62.0, 0.0, Target(crate::volume_mode::Target::Form));
        at(62.5, 2.5, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.50, 0.40), (0.60, 0.42), 0.0, 8) });
        at(65.5, 3.0, Camera(Pose::at(patch * 1.05, 0.34)));
        at(69.0, 4.5, Caption("Undo and redo work on the stack: the form's stroke, and the layer as it was"));
        at(69.5, 0.0, Undo);
        at(71.0, 0.0, Redo);
        // The layer's mask.
        at(73.5, 0.0, Target(crate::volume_mode::Target::Mask { show: false }));
        at(73.5, 6.5, Caption("Every layer has a mask: painting it hides the detail layer there, and painting it back shows it again"));
        at(74.5, 1.8, Stroke { tool: Tool::ClayBuildup, size: 80.0, strength: 1.0, path: line((0.40, 0.615), (0.80, 0.615), 0.0, 12) });
        at(77.0, 0.0, Target(crate::volume_mode::Target::Mask { show: true }));
        at(77.3, 1.2, Stroke { tool: Tool::ClayBuildup, size: 80.0, strength: 1.0, path: line((0.40, 0.615), (0.58, 0.615), 0.0, 8) });
        at(79.0, 0.0, Target(crate::volume_mode::Target::Detail));
        // A second layer, at a resolution of its own.
        at(79.5, 0.0, AddLayer(5));
        at(79.5, 6.0, Caption("A second layer, finer (level 5, the first is level 4): fine strokes painted on it go into it. Layers add up, each at its own resolution"));
        for k in 0..4 {
            let x = 0.46 + 0.07 * k as f32;
            at(80.5 + 1.0 * k as f32, 0.8, Stroke { tool: Tool::TrimDynamic, size: 12.0, strength: 1.0, path: line((x, 0.40), (x + 0.04, 0.66), 0.0, 10) });
        }
        at(85.0, 2.0, Camera(Pose::at(patch, 3.0)));
        at(85.0, 9.0, Caption("The form again, under both layers: each rides along, at its own resolution"));
        at(85.5, 0.0, Target(crate::volume_mode::Target::Form));
        at(87.0, 2.5, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.48, 0.42), (0.58, 0.44), 0.0, 8) });
        at(90.0, 3.0, Camera(Pose::at(patch * 1.11, 0.34)));
        at(94.0, 5.0, Caption("Each layer can be hidden: the finer one off, and on again"));
        at(94.5, 0.0, LayerVisible(1, false));
        at(96.5, 0.0, LayerVisible(1, true));
        at(97.5, 0.0, Target(crate::volume_mode::Target::Detail));
        // Culling.
        at(98.0, 3.0, Camera(Pose::whole(3.4, 0.9, 0.25)));
        at(98.5, 8.0, Caption("Each block's surface is a cluster: those out of view or facing away are skipped (see the pages drawn)"));
        at(101.0, 0.0, Cull(false));
        at(103.0, 0.0, Cull(true));
        // The level by distance.
        at(103.5, 2.5, Camera(Pose::at(patch, 0.6)));
        at(103.5, 0.0, Tint(true));
        at(104.0, 6.0, Caption("The level by distance: each region shows only the levels whose detail would show on screen from here (tinted by level)"));
        at(106.5, 9.0, Camera(Pose::at(patch, 9.0)));
        at(110.0, 6.0, Caption("Pulling back, the fine levels hand over to coarser ones as their detail drops under a pixel: the readout counts the blocks left out"));
        at(116.0, 0.0, ByDistance(false));
        at(116.0, 2.5, Caption("Off: every level drawn, from afar too"));
        at(118.5, 0.0, ByDistance(true));
        at(118.5, 2.0, Caption("On again"));
        at(120.5, 2.5, Camera(Pose::at(patch, 0.6)));
        at(120.5, 3.0, Caption("Back in close, the detail is all there"));
        at(124.0, 0.0, Tint(false));
        // A generator layer.
        at(124.0, 2.5, Camera(Pose::whole(3.0, 0.35, 0.25)));
        at(126.5, 0.0, AddGenerator);
        at(126.5, 6.0, Caption("A generator layer in the same stack: rough rocks scattered over the form, joined to it softly, in the field (no overhang limit)"));
        at(130.0, 0.0, Generator(|s| s.count = 30));
        at(131.0, 0.0, Generator(|s| s.count = 60));
        at(132.5, 0.0, Generator(|s| s.size *= 1.5));
        at(133.5, 0.0, Generator(|s| s.blend = 0.6));
        at(132.5, 4.0, Caption("Count, size, blend: each edit is shown at once (a drag previews it no finer than level 1)"));
        at(134.8, 0.0, Generator(|s| s.rough_kind = sculpt_core::noise::NoiseKind::Ridged));
        at(135.6, 0.0, Generator(|s| s.rough = 0.0));
        at(134.8, 1.7, Caption("Roughness: ridged, then none"));
        at(136.4, 0.0, Generator(|s| { s.rough = 0.15; s.rough_kind = sculpt_core::noise::NoiseKind::Fbm; }));
        at(136.5, 0.0, Generator(|s| s.op = sculpt_core::generator::BoolOp::Subtract));
        at(136.5, 3.0, Caption("Subtract: the same scatter cuts pits"));
        at(139.5, 0.0, Generator(|s| s.op = sculpt_core::generator::BoolOp::Add));
        at(140.0, 2.5, Camera(Pose::at(patch, 2.2)));
        at(140.5, 7.0, Caption("Sculpting the form under it carries the rocks along: each hangs from a point of the surface, found again after the stroke"));
        at(141.0, 0.0, Target(crate::volume_mode::Target::Form));
        at(142.5, 2.5, Stroke { tool: Tool::ClayBuildup, size: 260.0, strength: 1.0, path: line((0.35, 0.45), (0.65, 0.5), 0.0, 10) });
        at(145.5, 2.5, Stroke { tool: Tool::ClayBuildup, size: 260.0, strength: 1.0, path: line((0.65, 0.55), (0.35, 0.5), 0.0, 10) });
        at(148.5, 0.0, Target(crate::volume_mode::Target::Mask { show: false }));
        at(148.5, 4.5, Caption("Its mask fades it where painted, as every layer's does"));
        at(149.0, 2.5, Stroke { tool: Tool::ClayBuildup, size: 200.0, strength: 1.0, path: line((0.40, 0.40), (0.60, 0.42), 0.0, 8) });
        at(152.0, 0.0, Target(crate::volume_mode::Target::Detail));
        at(153.0, 2.5, Camera(Pose::whole(4.4, 0.0, 0.12)));
        at(153.0, 3.0, Caption("Next: generators of other kinds (noise, edges), and the layers' work on the GPU"));
        let mut end = 156.5;
        // `SCULPT_DEMO_SCRIPT=noise`: only the surface noise, on the plain ball (a quick look).
        if std::env::var("SCULPT_DEMO_SCRIPT").as_deref() == Ok("noise") {
            items.clear();
            let mut at = |t: f32, dur: f32, act: Act| items.push(Item { at: t, dur, act });
            at(0.0, 0.0, Camera(Pose::whole(3.2, 0.3, 0.2)));
            at(0.5, 0.0, Noise(|n| {
                n.on = true;
                n.amplitude = 0.02;
                n.scale = 3.0;
                n.octaves = 7;
            }));
            at(0.5, 3.0, Caption("Surface noise, drawn by the GPU: coarse octaves move the surface, every octave shades it"));
            at(3.5, 0.0, Noise(|n| n.kind = 1));
            at(3.5, 2.0, Caption("Ridged"));
            at(5.5, 0.0, Target(crate::volume_mode::Target::Mask { show: false }));
            at(5.5, 3.0, Caption("Its mask fades it where painted"));
            at(6.0, 2.0, Stroke { tool: Tool::ClayBuildup, size: 220.0, strength: 1.0, path: line((0.35, 0.45), (0.6, 0.5), 0.0, 10) });
            at(8.5, 2.5, Camera(Pose::at(patch, 0.4)));
            at(8.5, 3.0, Caption("Close up: octaves finer than any level, as shading"));
            end = 12.0;
        }
        // `SCULPT_DEMO_SCRIPT=gens`: only a generator and its edits, on the plain ball (a quick look).
        if std::env::var("SCULPT_DEMO_SCRIPT").as_deref() == Ok("gens") {
            items.clear();
            let mut at = |t: f32, dur: f32, act: Act| items.push(Item { at: t, dur, act });
            at(0.0, 0.0, Camera(Pose::whole(3.0, 0.35, 0.25)));
            at(0.5, 0.0, AddGenerator);
            at(2.0, 0.0, Generator(|s| s.count = 40));
            at(3.5, 0.0, Generator(|s| s.size *= 1.5));
            at(5.0, 0.0, Generator(|s| s.blend = 0.6));
            at(6.5, 0.0, Generator(|s| s.rough_kind = sculpt_core::noise::NoiseKind::Ridged));
            at(8.0, 0.0, Generator(|s| s.op = sculpt_core::generator::BoolOp::Subtract));
            end = 9.5;
        }
        // `SCULPT_DEMO_SCRIPT=stress`: the stress test's kinds of edits in the app, for looking at the frames: strokes
        // of every size at every level on the form and on layers, layers added, moved between levels, hidden and
        // removed, masks, a generator edited, noise baked, a file round trip, undo and redo, and the camera moving
        // near and far so the levels hand over.
        if std::env::var("SCULPT_DEMO_SCRIPT").as_deref() == Ok("stress") {
            items.clear();
            let mut at = |t: f32, dur: f32, act: Act| items.push(Item { at: t, dur, act });
            at(0.0, 0.0, Camera(start));
            at(0.0, 3.0, Caption("Stress: big strokes at level 0"));
            at(0.3, 1.6, Stroke { tool: Tool::ClayBuildup, size: 150.0, strength: 0.9, path: line((0.33, 0.40), (0.62, 0.36), 0.02, 12) });
            at(2.1, 1.6, Stroke { tool: Tool::TrimDynamic, size: 120.0, strength: 0.8, path: line((0.40, 0.72), (0.62, 0.70), 0.015, 12) });
            at(3.9, 1.6, Stroke { tool: Tool::ClayBuildup, size: 260.0, strength: 1.0, path: line((0.35, 0.55), (0.65, 0.58), 0.0, 10) });
            at(5.7, 2.0, Camera(Pose::at(patch, 1.1)));
            at(5.7, 3.0, Caption("Zoomed in: finer levels"));
            at(7.9, 1.4, Stroke { tool: Tool::ClayBuildup, size: 45.0, strength: 1.0, path: line((0.36, 0.42), (0.64, 0.44), 0.03, 12) });
            at(9.5, 2.0, Camera(Pose::at(patch, 0.32)));
            at(9.5, 4.0, Caption("Grooves and dots at levels 4 and 5"));
            for k in 0..3 {
                let y = 0.40 + 0.08 * k as f32;
                at(11.7 + 0.8 * k as f32, 0.7, Stroke { tool: Tool::TrimDynamic, size: 14.0, strength: 1.0, path: line((0.38, y), (0.62, y + 0.01), 0.004, 10) });
            }
            for k in 0..4 {
                let x = 0.40 + 0.05 * k as f32;
                at(14.2 + 0.3 * k as f32, 0.2, Stroke { tool: Tool::ClayBuildup, size: 9.0, strength: 1.0, path: line((x, 0.69), (x + 0.002, 0.69), 0.0, 2) });
            }
            // The fine work into a layer; a big stroke on the form under it.
            at(15.5, 0.0, MakeLayer(4));
            at(15.5, 0.0, Target(crate::volume_mode::Target::Form));
            at(15.6, 2.0, Camera(Pose::at(patch, 3.0)));
            at(15.6, 4.0, Caption("Layer made from level 4; a big form stroke under it"));
            at(17.8, 1.8, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.50, 0.40), (0.60, 0.42), 0.0, 8) });
            at(19.8, 2.0, Camera(Pose::at(patch * 1.05, 0.34)));
            at(21.9, 0.0, Undo);
            at(22.4, 0.0, Redo);
            at(21.9, 2.0, Caption("Undo, redo"));
            // Masks.
            at(23.0, 0.0, Target(crate::volume_mode::Target::Mask { show: false }));
            at(23.0, 4.0, Caption("Mask hidden, shown, hidden everywhere, painted back"));
            at(23.2, 1.2, Stroke { tool: Tool::ClayBuildup, size: 80.0, strength: 1.0, path: line((0.40, 0.615), (0.80, 0.615), 0.0, 12) });
            at(24.6, 0.0, Target(crate::volume_mode::Target::Mask { show: true }));
            at(24.7, 0.8, Stroke { tool: Tool::ClayBuildup, size: 80.0, strength: 1.0, path: line((0.40, 0.615), (0.58, 0.615), 0.0, 8) });
            at(25.7, 0.0, FillMask(0.0));
            at(26.2, 0.8, Stroke { tool: Tool::ClayBuildup, size: 120.0, strength: 1.0, path: line((0.40, 0.45), (0.60, 0.50), 0.0, 8) });
            at(27.2, 0.0, FillMask(1.0));
            at(27.3, 0.0, Target(crate::volume_mode::Target::Detail));
            // A second, finer layer painted on, moved coarser and back.
            at(27.5, 0.0, AddLayer(5));
            at(27.5, 4.0, Caption("A level 5 layer painted on, held at level 4, then 5 again"));
            for k in 0..3 {
                let x = 0.46 + 0.08 * k as f32;
                at(27.7 + 0.8 * k as f32, 0.6, Stroke { tool: Tool::TrimDynamic, size: 12.0, strength: 1.0, path: line((x, 0.40), (x + 0.04, 0.66), 0.0, 10) });
            }
            at(30.3, 0.0, LayerLevel(1, 4));
            at(31.0, 0.0, LayerLevel(1, 5));
            // The form again under both, then a generator.
            at(31.5, 1.5, Camera(Pose::at(patch, 3.0)));
            at(31.5, 0.0, Target(crate::volume_mode::Target::Form));
            at(31.5, 3.0, Caption("The form under both layers"));
            at(33.1, 1.6, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.48, 0.42), (0.58, 0.44), 0.0, 8) });
            at(34.8, 1.5, Camera(Pose::whole(3.0, 0.35, 0.25)));
            at(36.4, 0.0, AddGenerator);
            at(36.4, 5.0, Caption("A generator: count, size, cut, add, ridged; its mask painted"));
            at(37.2, 0.0, Generator(|s| s.count = 30));
            at(37.9, 0.0, Generator(|s| s.size *= 1.4));
            at(38.6, 0.0, Generator(|s| s.op = sculpt_core::generator::BoolOp::Subtract));
            at(39.3, 0.0, Generator(|s| s.op = sculpt_core::generator::BoolOp::Add));
            at(40.0, 0.0, Generator(|s| s.rough_kind = sculpt_core::noise::NoiseKind::Ridged));
            at(40.5, 0.0, Target(crate::volume_mode::Target::Mask { show: false }));
            at(40.6, 1.2, Stroke { tool: Tool::ClayBuildup, size: 200.0, strength: 1.0, path: line((0.40, 0.40), (0.60, 0.42), 0.0, 8) });
            at(42.0, 0.0, Target(crate::volume_mode::Target::Form));
            at(42.0, 3.0, Caption("A big form stroke under the rocks"));
            at(42.2, 1.6, Stroke { tool: Tool::ClayBuildup, size: 240.0, strength: 1.0, path: line((0.35, 0.45), (0.65, 0.5), 0.0, 10) });
            at(44.0, 0.0, Target(crate::volume_mode::Target::Detail));
            // Surface noise, baked; the file round trip.
            at(44.2, 0.0, Noise(|n| {
                n.on = true;
                n.amplitude = 0.015;
                n.scale = 4.0;
                n.octaves = 6;
            }));
            at(44.2, 4.0, Caption("Surface noise, baked into a level 3 layer; saved and opened"));
            at(45.8, 0.0, BakeNoise(3));
            at(47.3, 0.0, SaveOpen);
            // Near and far: the levels hand over.
            at(48.2, 2.0, Camera(Pose::at(patch, 0.6)));
            at(48.2, 3.0, Caption("Close in"));
            at(50.4, 5.0, Camera(Pose::at(patch, 9.0)));
            at(50.4, 5.0, Caption("Pulling back: fine levels hand over to coarse"));
            at(55.6, 3.0, Camera(Pose::at(patch, 0.4)));
            at(55.6, 3.0, Caption("And back in"));
            at(58.8, 0.0, ByDistance(false));
            at(59.3, 0.0, ByDistance(true));
            // A layer removed and brought back; a last big trim.
            at(59.8, 0.0, RemoveLayer(0));
            at(59.8, 3.0, Caption("The first layer removed, undone"));
            at(60.8, 0.0, Undo);
            at(61.5, 1.5, Camera(Pose::whole(3.4, 0.9, 0.25)));
            at(61.5, 0.0, Target(crate::volume_mode::Target::Form));
            at(63.1, 1.6, Stroke { tool: Tool::TrimDynamic, size: 200.0, strength: 1.0, path: line((0.35, 0.50), (0.65, 0.52), 0.0, 10) });
            at(64.9, 0.0, Tint(true));
            at(65.6, 0.0, Tint(false));
            end = 66.3;
        }
        let n = items.len();
        VolumeDemo { dir, items, frame: 0, end, begun: vec![false; n], fired: vec![false; n], waiting: None, caption: None }
    }

    /// Past the script's end (or `SCULPT_DEMO_END` seconds, for a quick look at its start).
    pub fn finished(&self) -> bool {
        let end = std::env::var("SCULPT_DEMO_END").ok().and_then(|v| v.parse().ok()).unwrap_or(self.end);
        self.frame as f32 / FPS > end
    }

    /// The camera pose at time `t`: the latest camera move, eased from where the one before it ended.
    fn pose_at(&self, t: f32) -> Option<Pose> {
        let moves: Vec<&Item> = self.items.iter().filter(|i| matches!(i.act, Act::Camera(_))).collect();
        let mut current: Option<Pose> = None;
        for m in moves {
            let Act::Camera(p) = m.act else { continue };
            if t < m.at {
                break;
            }
            let from = current.unwrap_or(p);
            let k = if m.dur <= 0.0 { 1.0 } else { ((t - m.at) / m.dur).clamp(0.0, 1.0) };
            current = Some(Pose::lerp(from, p, k));
            if k < 1.0 {
                return current;
            }
        }
        current
    }

    /// Advance the script by one frame.
    pub fn step(&mut self, rect: Rect, camera: &Camera) -> Frame {
        let t = self.frame as f32 / FPS;
        let dt = 1.0 / FPS;
        let mut f = Frame::default();
        if let Some(p) = self.pose_at(t) {
            let mut c = camera.clone();
            c.target = p.target;
            c.distance = p.distance;
            c.yaw = p.yaw;
            c.pitch = p.pitch;
            f.camera = Some(c);
        }
        let at = |(x, y): (f32, f32)| Pos2::new(rect.min.x + x * rect.width(), rect.min.y + y * rect.height());
        self.caption = None;
        for (i, item) in self.items.iter().enumerate() {
            let inside = t >= item.at && t < item.at + item.dur.max(dt);
            match &item.act {
                Act::Caption(text) if inside => self.caption = Some((text.to_string(), ((t - item.at) / 0.4).min((item.at + item.dur - t) / 0.4).clamp(0.0, 1.0))),
                Act::Stroke { tool, size, strength, path } => {
                    if t >= item.at && !self.begun[i] {
                        self.begun[i] = true;
                        f.begin = Some((*tool, *size, *strength, at(path[0])));
                    }
                    if self.begun[i] && !self.fired[i] {
                        // Samples along the path for this frame's slice of the stroke.
                        let pos = |u: f32| {
                            let s = u.clamp(0.0, 1.0) * (path.len() - 1) as f32;
                            let k = (s.floor() as usize).min(path.len() - 2);
                            let a = path[k];
                            let b = path[k + 1];
                            let w = s - k as f32;
                            at((a.0 + (b.0 - a.0) * w, a.1 + (b.1 - a.1) * w))
                        };
                        for j in 1..=SAMPLES {
                            let u = (t - item.at + dt * j as f32 / SAMPLES as f32) / item.dur;
                            f.samples.push(pos(u));
                        }
                        if t + dt >= item.at + item.dur {
                            self.fired[i] = true;
                            f.end = true;
                        }
                    }
                }
                Act::CapSweep(a, b) if inside => {
                    let k = ((t - item.at) / item.dur).clamp(0.0, 1.0);
                    f.cap = Some((*a as f32 + (*b as f32 - *a as f32) * k).round() as usize);
                }
                _ if t >= item.at && !self.fired[i] => {
                    match item.act {
                        Act::Cap(c) => f.cap = Some(c),
                        Act::Tint(on) => f.tint = Some(on),
                        Act::MakeLayer(from) => f.make_layer = Some(from),
                        Act::Target(t) => f.target = Some(t),
                        Act::AddLayer(l) => f.add_layer = Some(l),
                        Act::LayerVisible(k, on) => f.layer_visible = Some((k, on)),
                        Act::LayerLevel(k, l) => f.layer_level = Some((k, l)),
                        Act::RemoveLayer(k) => f.remove_layer = Some(k),
                        Act::ActiveLayer(k) => f.active_layer = Some(k),
                        Act::FillMask(to) => f.fill_mask = Some(to),
                        Act::BakeNoise(l) => f.bake_noise = Some(l),
                        Act::SaveOpen => f.save_open = true,
                        Act::Cull(on) => f.cull = Some(on),
                        Act::ByDistance(on) => f.by_distance = Some(on),
                        Act::AddGenerator => f.add_generator = true,
                        Act::Generator(edit) => f.generator = Some(edit),
                        Act::Noise(edit) => f.noise = Some(edit),
                        Act::Undo => f.undo = true,
                        Act::Redo => f.redo = true,
                        _ => continue,
                    }
                    self.fired[i] = true;
                }
                _ => {}
            }
        }
        self.frame += 1;
        f
    }

    /// Path of frame `n`'s image.
    pub fn frame_path(&self, n: u32) -> PathBuf {
        self.dir.join(format!("frame_{n:05}.png"))
    }

    pub fn current_frame(&self) -> u32 {
        self.frame
    }
}

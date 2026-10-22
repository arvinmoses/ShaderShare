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
    /// Whether big brushes keep the finer detail under them (Mesh menu).
    KeepDetail(bool),
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
    pub cull: Option<bool>,
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
        // The same stroke keeping the detail.
        at(60.0, 0.0, Undo);
        at(60.0, 2.0, Camera(Pose::at(patch, 3.0)));
        at(60.0, 9.0, Caption("Undone, and again with \"Big brushes keep finer detail\" on (Mesh menu): the detail rides along on the new form, as at a lower subdivision level"));
        at(62.0, 0.0, KeepDetail(true));
        at(62.5, 2.5, Stroke { tool: Tool::ClayBuildup, size: 170.0, strength: 1.0, path: line((0.50, 0.40), (0.60, 0.42), 0.0, 8) });
        at(65.5, 3.0, Camera(Pose::at(patch * 1.05, 0.34)));
        at(69.0, 5.0, Caption("Undo and redo: the whole stroke, at every level it touched"));
        at(70.0, 0.0, Undo);
        at(71.5, 0.0, Redo);
        at(72.5, 0.0, Undo);
        at(73.5, 0.0, Redo);
        at(74.0, 0.0, KeepDetail(false));
        // Culling.
        at(74.0, 3.0, Camera(Pose::whole(3.4, 0.9, 0.25)));
        at(74.5, 8.0, Caption("Each block's surface is a cluster: those out of view or facing away are skipped (see the pages drawn)"));
        at(77.0, 0.0, Cull(false));
        at(79.0, 0.0, Cull(true));
        at(79.5, 3.0, Camera(Pose::whole(3.4, 2.2, 0.1)));
        at(82.5, 2.5, Camera(Pose::whole(4.4, 0.0, 0.12)));
        at(82.5, 3.0, Caption("Next: choosing the level per region by distance and by how much detail each region holds"));
        let end = 86.0;
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
                        Act::KeepDetail(on) => f.keep_detail = Some(on),
                        Act::Cull(on) => f.cull = Some(on),
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

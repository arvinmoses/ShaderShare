//! `--test-strokes N`: drives synthetic pen strokes through the real input
//! path (samples → raycast → dabs → dirty-leaf upload → render) for N frames,
//! prints per-frame timing percentiles, optionally saves a screenshot, and
//! exits. Used to validate that the UI layer adds no cost to sculpting.

use std::path::PathBuf;
use std::time::Instant;

use egui::{Pos2, Rect};
use sculpt_core::Document;

use crate::app::FrameStats;
use crate::tools::Tool;

/// Samples per frame: a 240 Hz pen at 60 fps.
const SAMPLES_PER_FRAME: usize = 4;
const STROKE_FRAMES: u32 = 45;
const WARMUP: u32 = 20;
const TOOLS: [Tool; 4] = [Tool::ClayBuildup, Tool::TrimDynamic, Tool::ClayBuildup, Tool::Smooth];

#[derive(Default)]
pub struct StrokeEdges {
    pub end: bool,
    pub tool: Option<Tool>,
    pub begin: Option<Pos2>,
}

struct Rec {
    interval_ms: f32,
    frame_ms: f32,
    input_ms: f32,
    render_ms: f32,
    upload_kb: f32,
    dabs: usize,
}

pub struct TestDriver {
    total: u32,
    frame: u32,
    pub edges: StrokeEdges,
    records: Vec<Rec>,
    screenshot: Option<PathBuf>,
    requested: bool,
    active: bool,
    last_time: Option<Instant>,
    mesh: String,
}

impl TestDriver {
    pub fn new(total: u32, screenshot: Option<PathBuf>) -> TestDriver {
        TestDriver {
            total,
            frame: 0,
            edges: StrokeEdges::default(),
            records: Vec::new(),
            screenshot,
            requested: false,
            active: false,
            last_time: None,
            mesh: String::new(),
        }
    }

    fn pos(rect: Rect, segment: u32, t: f32) -> Pos2 {
        let lanes = [-0.16, -0.05, 0.06, 0.0];
        let x = rect.left() + rect.width() * (0.36 + 0.28 * t);
        let y = rect.center().y + rect.height() * (lanes[segment as usize % 4] + 0.06 * (t * std::f32::consts::TAU * 1.5).sin());
        Pos2::new(x, y)
    }

    /// Queue this frame's synthetic pen samples.
    pub fn drive(&mut self, rect: Rect, pending: &mut std::collections::VecDeque<crate::app::SampleIn>) {
        if self.frame < WARMUP {
            return;
        }
        let local = self.frame - WARMUP;
        if local >= self.total {
            if self.active {
                self.edges.end = true;
                self.active = false;
            }
            return;
        }
        let seg = local / STROKE_FRAMES;
        let idx = local % STROKE_FRAMES;
        if idx == 0 {
            if self.active {
                self.edges.end = true;
            }
            self.edges.tool = Some(TOOLS[seg as usize % TOOLS.len()]);
            self.edges.begin = Some(Self::pos(rect, seg, 0.0));
            self.active = true;
        }
        for k in 1..=SAMPLES_PER_FRAME {
            let t = (idx as f32 + k as f32 / SAMPLES_PER_FRAME as f32) / STROKE_FRAMES as f32;
            pending.push_back(crate::app::SampleIn { pos: Self::pos(rect, seg, t), pressure: 0.5 + 0.5 * (t * std::f32::consts::PI).sin() });
        }
    }

    pub fn record(&mut self, s: &FrameStats, ready: bool, doc: Option<&Document>) {
        if !ready {
            return;
        }
        let now = Instant::now();
        let interval = self.last_time.map_or(0.0, |t| (now - t).as_secs_f32() * 1e3);
        self.last_time = Some(now);
        if let Some(d) = doc {
            self.mesh = format!("{} faces / {} verts", d.face_count(), d.vertex_count());
        }
        if self.frame >= WARMUP && self.frame < WARMUP + self.total {
            self.records.push(Rec {
                interval_ms: interval,
                frame_ms: s.frame_ms,
                input_ms: s.input_ms,
                render_ms: s.render_ms,
                upload_kb: s.upload.bytes as f32 / 1024.0,
                dabs: s.dabs,
            });
        }
        self.frame += 1;
    }

    pub fn finish_if_done(&mut self, ctx: &egui::Context) {
        if self.requested || self.frame < WARMUP + self.total + 5 {
            return;
        }
        self.requested = true;
        self.report();
        if self.screenshot.is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        } else {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn report(&self) {
        fn pct(mut v: Vec<f32>, p: f32) -> f32 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(f32::total_cmp);
            v[((v.len() - 1) as f32 * p).round() as usize]
        }
        let col = |f: &dyn Fn(&Rec) -> f32| -> (f32, f32, f32) {
            let v: Vec<f32> = self.records.iter().map(f).collect();
            (pct(v.clone(), 0.5), pct(v.clone(), 0.95), pct(v, 1.0))
        };
        let dabs: usize = self.records.iter().map(|r| r.dabs).sum();
        println!("== sculpt-app stroke test: {} frames, {} dabs, mesh {} ==", self.records.len(), dabs, self.mesh);
        println!("{:<34} {:>9} {:>9} {:>9}", "metric (per frame)", "median", "p95", "max");
        for (name, (a, b, c)) in [
            ("frame-to-frame interval (ms)", col(&|r| r.interval_ms)),
            ("CPU: whole frame incl. UI (ms)", col(&|r| r.frame_ms)),
            ("CPU: input + dabs (ms)", col(&|r| r.input_ms)),
            ("CPU: viewport encode+submit (ms)", col(&|r| r.render_ms)),
            ("GPU upload (KB)", col(&|r| r.upload_kb)),
            ("dabs", col(&|r| r.dabs as f32)),
        ] {
            println!("{name:<34} {a:>9.2} {b:>9.2} {c:>9.2}");
        }
    }

    pub fn save_screenshot(&self, image: &egui::ColorImage) {
        let Some(path) = &self.screenshot else { return };
        let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
        let file = match std::fs::File::create(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("screenshot: {e}");
                return;
            }
        };
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), image.size[0] as u32, image.size[1] as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Err(e) = enc.write_header().and_then(|mut w| w.write_image_data(&rgba)) {
            eprintln!("screenshot: {e}");
        } else {
            println!("screenshot saved to {}", path.display());
        }
    }
}

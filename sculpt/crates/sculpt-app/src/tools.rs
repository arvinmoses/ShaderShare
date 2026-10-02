//! Tool definitions and per-tool settings (Mudbox: every tool remembers its
//! own size and strength).

use std::collections::BTreeMap;

use sculpt_core::brush::{ClayBuildup, Smooth, TrimDynamic};
use serde::{Deserialize, Serialize};

use crate::keymap::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tool {
    ClayBuildup,
    TrimDynamic,
    Move,
    Smooth,
    Freeze,
    MaskPaint,
    Pose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tray {
    Sculpt,
    Paint,
    Pose,
}

impl Tool {
    pub const ALL: [Tool; 7] = [Tool::ClayBuildup, Tool::TrimDynamic, Tool::Move, Tool::Smooth, Tool::Freeze, Tool::MaskPaint, Tool::Pose];

    pub fn label(self) -> &'static str {
        match self {
            Tool::ClayBuildup => "Clay Buildup",
            Tool::TrimDynamic => "Trim Dynamic",
            Tool::Move => "Move",
            Tool::Smooth => "Smooth",
            Tool::Freeze => "Freeze",
            Tool::MaskPaint => "Mask Paint",
            Tool::Pose => "Pose",
        }
    }

    pub fn tray(self) -> Tray {
        match self {
            Tool::Freeze | Tool::MaskPaint => Tray::Paint,
            Tool::Pose => Tray::Pose,
            _ => Tray::Sculpt,
        }
    }

    pub fn command(self) -> Command {
        match self {
            Tool::ClayBuildup => Command::ToolClayBuildup,
            Tool::TrimDynamic => Command::ToolTrimDynamic,
            Tool::Move => Command::ToolMove,
            Tool::Smooth => Command::ToolSmooth,
            Tool::Freeze => Command::ToolFreeze,
            Tool::MaskPaint => Command::ToolMaskPaint,
            Tool::Pose => Command::ToolPose,
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::ClayBuildup => "Builds clay up to a plane. Ctrl: dig. Shift: smooth.",
            Tool::TrimDynamic => "Shaves to a dynamic plane. Ctrl: fill.",
            Tool::Move => "Drag the surface. Topological option keeps nearby parts still.",
            Tool::Smooth => "Relax the surface (also Shift with any sculpt tool).",
            Tool::Freeze => "Paint freeze to protect areas. Ctrl: unfreeze.",
            Tool::MaskPaint => "Paint the mask channel used by layer masks. Ctrl: erase.",
            Tool::Pose => "Click a limb to mask it topologically, drag to rotate.",
        }
    }

    /// Vector icon for tray tiles (drawn, so it never depends on font coverage).
    pub fn paint_icon(self, p: &egui::Painter, r: egui::Rect, c: egui::Color32) {
        use egui::{Pos2, Shape, Stroke, pos2};
        let st = Stroke::new(1.8, c);
        let at = |x: f32, y: f32| pos2(r.left() + r.width() * x, r.top() + r.height() * y);
        let curve = |f: &dyn Fn(f32) -> f32| -> Vec<Pos2> { (0..=24).map(|i| { let t = i as f32 / 24.0; at(0.1 + 0.8 * t, f(t)) }).collect() };
        match self {
            Tool::ClayBuildup => {
                // Surface with a flat-topped clay strip on it.
                p.add(Shape::line(curve(&|t| 0.8 - 0.08 * (t * 3.1).sin()), st));
                p.add(Shape::convex_polygon(vec![at(0.3, 0.75), at(0.36, 0.45), at(0.64, 0.45), at(0.7, 0.75)], c.gamma_multiply(0.55), st));
            }
            Tool::TrimDynamic => {
                p.add(Shape::line(curve(&|t| 0.75 - 0.45 * (t * std::f32::consts::PI).sin()), Stroke::new(1.2, c.gamma_multiply(0.5))));
                p.add(Shape::line(curve(&|t| (0.75 - 0.45 * (t * std::f32::consts::PI).sin()).max(0.5)), st));
                p.line_segment([at(0.15, 0.5), at(0.85, 0.5)], Stroke::new(1.0, c));
            }
            Tool::Move => {
                let ctr = r.center();
                for d in [egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0), egui::vec2(0.0, 1.0), egui::vec2(0.0, -1.0)] {
                    let tip = ctr + d * r.width() * 0.4;
                    p.line_segment([ctr, tip], st);
                    let side = egui::vec2(-d.y, d.x) * r.width() * 0.1;
                    p.line_segment([tip, tip - d * r.width() * 0.12 + side], st);
                    p.line_segment([tip, tip - d * r.width() * 0.12 - side], st);
                }
            }
            Tool::Smooth => {
                for k in 0..3 {
                    let y0 = 0.3 + k as f32 * 0.2;
                    let amp = 0.08 * (1.0 - k as f32 * 0.4);
                    p.add(Shape::line(curve(&|t| y0 + amp * (t * 12.0).sin()), st));
                }
            }
            Tool::Freeze => {
                let ctr = r.center();
                for k in 0..3 {
                    let a = k as f32 * std::f32::consts::PI / 3.0;
                    let d = egui::vec2(a.cos(), a.sin()) * r.width() * 0.38;
                    p.line_segment([ctr - d, ctr + d], st);
                }
                p.circle_stroke(ctr, r.width() * 0.08, st);
            }
            Tool::MaskPaint => {
                let ctr = r.center();
                let rad = r.width() * 0.36;
                p.circle_stroke(ctr, rad, st);
                let half: Vec<Pos2> = (0..=20).map(|i| { let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / 20.0; ctr + egui::vec2(a.cos(), a.sin()) * rad }).collect();
                p.add(Shape::convex_polygon(half, c.gamma_multiply(0.6), Stroke::NONE));
            }
            Tool::Pose => {
                p.line_segment([at(0.2, 0.8), at(0.5, 0.55)], Stroke::new(3.0, c));
                p.line_segment([at(0.5, 0.55), at(0.8, 0.25)], Stroke::new(3.0, c));
                p.circle_filled(at(0.5, 0.55), r.width() * 0.07, c);
                let arc: Vec<Pos2> = (0..=12).map(|i| { let a = -0.2 - 1.3 * i as f32 / 12.0; at(0.5, 0.55) + egui::vec2(a.cos(), a.sin()) * r.width() * 0.32 }).collect();
                p.add(Shape::line(arc, Stroke::new(1.2, c.gamma_multiply(0.7))));
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolParams {
    /// Brush radius in screen points (like ZBrush draw size).
    pub size_px: f32,
    pub strength: f32,
    pub hardness: f32,
    pub front_faces_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoseMode {
    Rotate,
    Translate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolBox {
    pub params: BTreeMap<Tool, ToolParams>,
    pub clay: ClayBuildup,
    pub trim: TrimDynamic,
    pub smooth: Smooth,
    pub move_topological: bool,
    pub mask_channel: String,
    pub pose_mode: PoseMode,
    pub pose_softness: f32,
    pub pose_blur: u32,
}

impl Default for ToolBox {
    fn default() -> Self {
        let p = |size_px, strength, hardness, front| ToolParams { size_px, strength, hardness, front_faces_only: front };
        let params = BTreeMap::from([
            (Tool::ClayBuildup, p(40.0, 0.6, 0.2, true)),
            (Tool::TrimDynamic, p(50.0, 0.7, 0.3, true)),
            (Tool::Move, p(120.0, 1.0, 0.0, false)),
            (Tool::Smooth, p(45.0, 0.5, 0.2, true)),
            (Tool::Freeze, p(60.0, 1.0, 0.5, true)),
            (Tool::MaskPaint, p(60.0, 1.0, 0.3, true)),
            (Tool::Pose, p(150.0, 1.0, 0.0, false)),
        ]);
        ToolBox {
            params,
            clay: ClayBuildup::default(),
            trim: TrimDynamic::default(),
            smooth: Smooth::default(),
            move_topological: true,
            mask_channel: "paint.mask".into(),
            pose_mode: PoseMode::Rotate,
            pose_softness: 0.5,
            pose_blur: 2,
        }
    }
}

impl ToolBox {
    pub fn params(&self, t: Tool) -> &ToolParams {
        &self.params[&t]
    }
    pub fn params_mut(&mut self, t: Tool) -> &mut ToolParams {
        self.params.get_mut(&t).unwrap()
    }
}

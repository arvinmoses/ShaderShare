//! The volume brushes and their settings (every brush remembers its own size, strength and falloff).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::keymap::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tool {
    ClayBuildup,
    TrimDynamic,
}

impl Tool {
    pub const ALL: [Tool; 2] = [Tool::ClayBuildup, Tool::TrimDynamic];

    pub fn label(self) -> &'static str {
        match self {
            Tool::ClayBuildup => "Clay Buildup",
            Tool::TrimDynamic => "Trim Dynamic",
        }
    }

    pub fn command(self) -> Command {
        match self {
            Tool::ClayBuildup => Command::ToolClayBuildup,
            Tool::TrimDynamic => Command::ToolTrimDynamic,
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::ClayBuildup => "Adds to the volume. Ctrl: carves.",
            Tool::TrimDynamic => "Carves the volume. Ctrl: adds.",
        }
    }

    /// Whether a stroke carves, with Ctrl held or not.
    pub fn carves(self, ctrl: bool) -> bool {
        match self {
            Tool::ClayBuildup => ctrl,
            Tool::TrimDynamic => !ctrl,
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolParams {
    /// Brush radius in screen points (like ZBrush draw size).
    pub size_px: f32,
    pub strength: f32,
    pub hardness: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolBox {
    pub params: BTreeMap<Tool, ToolParams>,
}

impl Default for ToolBox {
    fn default() -> Self {
        let p = |size_px, strength, hardness| ToolParams { size_px, strength, hardness };
        ToolBox { params: BTreeMap::from([(Tool::ClayBuildup, p(40.0, 0.6, 0.2)), (Tool::TrimDynamic, p(50.0, 0.7, 0.3))]) }
    }
}

impl ToolBox {
    /// From saved settings: brushes and fields it does not know (older versions had more) are skipped.
    pub fn from_settings(v: &serde_json::Value) -> ToolBox {
        let mut tools = ToolBox::default();
        if let Some(saved) = v["params"].as_object() {
            for (name, p) in saved {
                let tool = serde_json::from_value::<Tool>(serde_json::Value::String(name.clone()));
                let size = p["size_px"].as_f64();
                if let (Ok(t), Some(size)) = (tool, size) {
                    let d = tools.params_mut(t);
                    d.size_px = size as f32;
                    d.strength = p["strength"].as_f64().map_or(d.strength, |x| x as f32);
                    d.hardness = p["hardness"].as_f64().map_or(d.hardness, |x| x as f32);
                }
            }
        }
        tools
    }

    pub fn params(&self, t: Tool) -> &ToolParams {
        &self.params[&t]
    }
    pub fn params_mut(&mut self, t: Tool) -> &mut ToolParams {
        self.params.get_mut(&t).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn older_settings_keep_the_brushes_they_share() {
        let old = serde_json::json!({
            "params": {
                "ClayBuildup": { "size_px": 77.0, "strength": 0.4, "hardness": 0.5, "front_faces_only": true },
                "Pose": { "size_px": 150.0, "strength": 1.0, "hardness": 0.0, "front_faces_only": false }
            },
            "clay": { "height": 0.2 },
            "move_topological": true
        });
        let t = ToolBox::from_settings(&old);
        assert_eq!(t.params(Tool::ClayBuildup), &ToolParams { size_px: 77.0, strength: 0.4, hardness: 0.5 });
        assert_eq!(t.params(Tool::TrimDynamic), ToolBox::default().params(Tool::TrimDynamic));
        let round = ToolBox::from_settings(&serde_json::to_value(&t).unwrap());
        assert_eq!(round.params, t.params);
    }
}

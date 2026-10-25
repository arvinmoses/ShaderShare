//! Small vector icons (drawn with the painter, so they never depend on font
//! coverage and always follow the theme's text color).

use egui::{Color32, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2, pos2, vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Eye,
    EyeOff,
    Trash,
    Mask,
    /// Surface noise.
    Noise,
    /// A detail layer (painted by strokes).
    Paint,
    /// A generator layer: rocks scattered on a surface.
    Scatter,
    /// The clay ball: the volume, its form, the app.
    Base,
}

impl Icon {
    pub fn paint(self, p: &Painter, r: Rect, c: Color32) {
        let s = Stroke::new(1.4, c);
        let at = |x: f32, y: f32| pos2(r.left() + r.width() * x, r.top() + r.height() * y);
        let w = r.width();
        match self {
            Icon::Eye | Icon::EyeOff => {
                let top: Vec<Pos2> = (0..=16).map(|i| { let t = i as f32 / 16.0; at(0.08 + 0.84 * t, 0.5 - 0.3 * (t * std::f32::consts::PI).sin()) }).collect();
                let bot: Vec<Pos2> = (0..=16).map(|i| { let t = i as f32 / 16.0; at(0.08 + 0.84 * t, 0.5 + 0.3 * (t * std::f32::consts::PI).sin()) }).collect();
                p.add(Shape::line(top, s));
                p.add(Shape::line(bot, s));
                p.circle_filled(r.center(), w * 0.13, c);
                if self == Icon::EyeOff {
                    p.line_segment([at(0.15, 0.85), at(0.85, 0.15)], Stroke::new(1.6, c));
                }
            }
            Icon::Trash => {
                p.line_segment([at(0.18, 0.28), at(0.82, 0.28)], s);
                p.line_segment([at(0.4, 0.18), at(0.6, 0.18)], s);
                p.add(Shape::closed_line(vec![at(0.27, 0.32), at(0.73, 0.32), at(0.68, 0.86), at(0.32, 0.86)], s));
            }
            Icon::Mask => {
                let b = r.shrink(w * 0.16);
                p.rect_stroke(b, 1.5, s, egui::StrokeKind::Inside);
                p.rect_filled(Rect::from_min_max(b.min, pos2(b.center().x, b.max.y)), 1.5, c);
            }
            Icon::Noise => {
                for (i, (x, y)) in [(0.25, 0.3), (0.6, 0.22), (0.78, 0.55), (0.4, 0.55), (0.22, 0.78), (0.62, 0.8)].into_iter().enumerate() {
                    p.circle_filled(at(x, y), w * (0.06 + 0.03 * (i % 3) as f32), c);
                }
            }
            Icon::Paint => {
                p.line_segment([at(0.78, 0.18), at(0.42, 0.56)], Stroke::new(2.0, c));
                p.add(Shape::convex_polygon(vec![at(0.42, 0.52), at(0.5, 0.62), at(0.3, 0.84), at(0.18, 0.84), at(0.2, 0.72)], c, Stroke::NONE));
            }
            Icon::Scatter => {
                let ground: Vec<Pos2> = (0..=16).map(|i| { let t = i as f32 / 16.0; at(0.06 + 0.88 * t, 0.84 - 0.06 * (t * std::f32::consts::PI).sin()) }).collect();
                p.add(Shape::line(ground, s));
                for (x, y, rad) in [(0.28, 0.66, 0.14), (0.58, 0.6, 0.2), (0.82, 0.7, 0.1)] {
                    p.circle_filled(at(x, y), w * rad, c);
                }
            }
            Icon::Base => {
                // Little shaded clay ball: dark core, lit cap, specular dot.
                let ctr = r.center();
                let rad = w * 0.36;
                p.circle_filled(ctr, rad, c.gamma_multiply(0.45));
                p.circle_filled(ctr - vec2(rad * 0.18, rad * 0.18), rad * 0.8, c.gamma_multiply(0.75));
                p.circle_filled(ctr - vec2(rad * 0.3, rad * 0.3), rad * 0.5, c);
                p.circle_filled(ctr - vec2(rad * 0.42, rad * 0.42), rad * 0.16, Color32::from_white_alpha(170));
            }
        }
    }
}

/// Square, frameless icon button. Highlights on hover and when `on`.
pub fn icon_button(ui: &mut Ui, icon: Icon, size: f32, on: bool, tooltip: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    let v = ui.visuals();
    if resp.hovered() {
        ui.painter().rect_filled(rect, 2.0, v.widgets.hovered.bg_fill);
    } else if on {
        ui.painter().rect_filled(rect, 2.0, v.widgets.inactive.bg_fill);
    }
    let color = if resp.hovered() || on { v.strong_text_color() } else { v.text_color() };
    icon.paint(ui.painter(), rect.shrink(size * 0.15), color);
    if tooltip.is_empty() { resp } else { resp.on_hover_text(tooltip) }
}

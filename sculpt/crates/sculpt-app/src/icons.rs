//! Small vector icons (drawn with the painter, so they never depend on font
//! coverage and always follow the theme's text color).

use egui::{Color32, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2, pos2, vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Eye,
    EyeOff,
    Lock,
    Unlock,
    Plus,
    Trash,
    Mask,
    Effect,
    ChevronRight,
    ChevronDown,
    ArrowUp,
    ArrowDown,
    Noise,
    Paint,
    Fill,
    Curvature,
    Cavity,
    Occlusion,
    Thickness,
    Direction,
    Gradient,
    Base,
    Folder,
    FolderOpen,
    Solo,
    Duplicate,
    Merge,
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
            Icon::Lock | Icon::Unlock => {
                let body = Rect::from_min_max(at(0.22, 0.45), at(0.78, 0.88));
                p.rect_filled(body, 1.5, c);
                let lift = if self == Icon::Unlock { 0.12 } else { 0.0 };
                let arc: Vec<Pos2> = (0..=12)
                    .map(|i| {
                        let a = std::f32::consts::PI * (1.0 + i as f32 / 12.0);
                        pos2(r.center().x + a.cos() * w * 0.2, r.top() + r.height() * (0.45 - lift) + a.sin() * r.height() * 0.28)
                    })
                    .collect();
                p.add(Shape::line(arc, s));
            }
            Icon::Plus => {
                p.line_segment([at(0.5, 0.18), at(0.5, 0.82)], Stroke::new(1.8, c));
                p.line_segment([at(0.18, 0.5), at(0.82, 0.5)], Stroke::new(1.8, c));
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
            Icon::Effect => {
                let ctr = r.center();
                for k in 0..4 {
                    let a = k as f32 * std::f32::consts::FRAC_PI_4 * 2.0 + 0.4;
                    let d = vec2(a.cos(), a.sin());
                    p.line_segment([ctr + d * w * 0.12, ctr + d * w * 0.38], s);
                }
                p.circle_filled(ctr, w * 0.08, c);
            }
            Icon::ChevronRight => {
                p.add(Shape::line(vec![at(0.38, 0.25), at(0.64, 0.5), at(0.38, 0.75)], s));
            }
            Icon::ChevronDown => {
                p.add(Shape::line(vec![at(0.25, 0.38), at(0.5, 0.64), at(0.75, 0.38)], s));
            }
            Icon::ArrowUp | Icon::ArrowDown => {
                let (a, b) = if self == Icon::ArrowUp { (0.2, 0.8) } else { (0.8, 0.2) };
                p.line_segment([at(0.5, a), at(0.5, b)], s);
                p.add(Shape::line(vec![at(0.28, a + (b - a) * 0.35), at(0.5, a), at(0.72, a + (b - a) * 0.35)], s));
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
            Icon::Fill => {
                p.rect_filled(r.shrink(w * 0.2), 2.0, c);
            }
            Icon::Curvature | Icon::Cavity => {
                let sign = if self == Icon::Curvature { -1.0 } else { 1.0 };
                let pts: Vec<Pos2> = (0..=16).map(|i| { let t = i as f32 / 16.0; at(0.12 + 0.76 * t, 0.55 + sign * 0.3 * (t * std::f32::consts::PI).sin()) }).collect();
                p.add(Shape::line(pts, Stroke::new(1.8, c)));
            }
            Icon::Occlusion => {
                p.circle_stroke(r.center(), w * 0.32, s);
                p.circle_filled(r.center() + vec2(w * 0.08, w * 0.08), w * 0.2, c);
            }
            Icon::Thickness => {
                p.line_segment([at(0.2, 0.25), at(0.8, 0.25)], Stroke::new(1.0, c));
                p.line_segment([at(0.2, 0.75), at(0.8, 0.75)], Stroke::new(1.0, c));
                p.line_segment([at(0.5, 0.3), at(0.5, 0.7)], s);
            }
            Icon::Direction => {
                p.line_segment([at(0.5, 0.82), at(0.5, 0.2)], Stroke::new(1.8, c));
                p.add(Shape::line(vec![at(0.3, 0.4), at(0.5, 0.2), at(0.7, 0.4)], Stroke::new(1.8, c)));
            }
            Icon::Gradient => {
                let b = r.shrink(w * 0.18);
                for k in 0..5 {
                    let x0 = b.left() + b.width() * k as f32 / 5.0;
                    let cell = Rect::from_min_max(pos2(x0, b.top()), pos2(x0 + b.width() / 5.0, b.bottom()));
                    p.rect_filled(cell, 0.0, c.gamma_multiply(0.2 + 0.2 * k as f32));
                }
            }
            Icon::Folder | Icon::FolderOpen => {
                let body = Rect::from_min_max(at(0.1, 0.3), at(0.9, 0.82));
                p.add(Shape::line(vec![at(0.1, 0.3), at(0.1, 0.2), at(0.4, 0.2), at(0.48, 0.3)], s));
                if self == Icon::Folder {
                    p.rect_filled(body, 1.5, c.gamma_multiply(0.85));
                } else {
                    p.rect_stroke(body, 1.5, s, egui::StrokeKind::Inside);
                    p.add(Shape::convex_polygon(vec![at(0.2, 0.82), at(0.32, 0.5), at(0.96, 0.5), at(0.84, 0.82)], c.gamma_multiply(0.85), Stroke::NONE));
                }
            }
            Icon::Solo => {
                // Dot with a ring: "this one only".
                p.circle_stroke(r.center(), w * 0.34, s);
                p.circle_filled(r.center(), w * 0.14, c);
            }
            Icon::Duplicate => {
                let a = Rect::from_min_max(at(0.14, 0.3), at(0.66, 0.86));
                let b = Rect::from_min_max(at(0.34, 0.14), at(0.86, 0.7));
                p.rect_stroke(a, 1.5, s, egui::StrokeKind::Inside);
                p.rect_filled(b, 1.5, c.gamma_multiply(0.55));
                p.rect_stroke(b, 1.5, s, egui::StrokeKind::Inside);
            }
            Icon::Merge => {
                p.line_segment([at(0.25, 0.15), at(0.5, 0.5)], s);
                p.line_segment([at(0.75, 0.15), at(0.5, 0.5)], s);
                p.line_segment([at(0.5, 0.5), at(0.5, 0.78)], s);
                p.add(Shape::line(vec![at(0.34, 0.64), at(0.5, 0.8), at(0.66, 0.64)], s));
                p.line_segment([at(0.2, 0.9), at(0.8, 0.9)], Stroke::new(1.8, c));
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

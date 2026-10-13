//! The chip that follows the brush: what a stroke would edit, in the target's colour.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, vec2};

use super::target::Target;
use crate::theme::UiColors;

/// Draw the chip just below-right of `at`, kept inside `bounds`.
pub fn chip(painter: &Painter, at: Pos2, target: &Target, colors: &UiColors, font_size: f32, bounds: Rect) {
    let color = target.color(colors);
    let mut text = target.text();
    if let Some(s) = target.strength.filter(|_| target.refusal.is_none()) {
        text = format!("{text}  {:.0}%", s * 100.0);
    }
    if let Some(why) = &target.refusal {
        text = format!("⊘ {why}");
    }
    let font = FontId::proportional(font_size * 0.92);
    let galley = painter.layout_no_wrap(text, font.clone(), Color32::WHITE);
    let size = galley.size() + vec2(24.0, 10.0);
    let mut min = at + vec2(18.0, 18.0);
    min.x = min.x.min(bounds.right() - size.x - 6.0).max(bounds.left() + 6.0);
    min.y = min.y.min(bounds.bottom() - size.y - 6.0).max(bounds.top() + 6.0);
    let rect = Rect::from_min_size(min, size);
    painter.rect_filled(rect, 5.0, Color32::from_black_alpha(190));
    painter.rect_stroke(rect, 5.0, egui::Stroke::new(1.5, color), egui::StrokeKind::Inside);
    painter.circle_filled(rect.left_center() + vec2(10.0, 0.0), 4.0, color);
    painter.text(rect.left_center() + vec2(20.0, 0.0), Align2::LEFT_CENTER, galley.text(), font, Color32::WHITE);
}

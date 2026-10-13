//! Drag a mask op row to reorder the ops of its layer.
//!
//! Rows are listed top of the stack first, so a higher row is a higher index in the mask stack.

use egui::{Pos2, Rect, Stroke, Ui};
use sculpt_core::LayerId;

use super::command::{self, LayerCommand};
use super::mask_ops::OpAction;
use crate::app::SculptApp;

/// One drawn op row: its index in the mask stack (bottom = 0) and where it was drawn.
pub type OpRow = (usize, Rect);

/// Index the dragged op ends up at, given the pointer height, or `None` when the pointer is not over a row.
/// `rows` come from one layer's stack. The result is the final index after the move.
pub fn drop_index(rows: &[OpRow], from: usize, pointer_y: f32) -> Option<usize> {
    let (over, rect) = rows.iter().find(|(_, r)| r.y_range().contains(pointer_y))?;
    let upper_half = pointer_y < rect.center().y;
    // Dropping on the upper half puts the op above that row (a higher index), the lower half below it.
    let slot = if upper_half { *over + 1 } else { *over };
    // `slot` counts positions with the dragged op still in the list; removing it first shifts the later ones down.
    Some(if from < slot { slot - 1 } else { slot })
}

/// Height of the insertion line for a drop landing at `to`. The dragged op ends up just below the row of the
/// op that will sit directly above it, or above the whole list when it lands on top.
pub fn line_y(rows: &[OpRow], from: usize, to: usize) -> Option<f32> {
    // Final indices of the other ops once the dragged one has left the list.
    let rest: Vec<(usize, Rect)> = rows.iter().filter(|(i, _)| *i != from).map(|(i, r)| (if *i > from { *i - 1 } else { *i }, *r)).collect();
    match rest.iter().filter(|(i, _)| *i >= to).min_by_key(|(i, _)| *i) {
        Some((_, r)) => Some(r.bottom()),
        None => rest.iter().max_by_key(|(i, _)| *i).map(|(_, r)| r.top()),
    }
}

/// Call after drawing a layer's op rows. Draws the insertion line and applies the move on release.
pub fn update(app: &mut SculptApp, ui: &mut Ui, layer: LayerId, rows: &[OpRow]) {
    let Some((owner, from)) = app.layers.op_drag else { return };
    if owner != layer {
        return;
    }
    let ctx = ui.ctx().clone();
    let (pointer, released, escape) = ctx.input(|i| (i.pointer.interact_pos(), i.pointer.any_released(), i.key_pressed(egui::Key::Escape)));
    let pointer = app.layers.pointer_override.or(pointer);
    let released = released && app.layers.pointer_override.is_none();
    if escape {
        app.layers.op_drag = None;
        return;
    }
    let target = pointer.and_then(|p| drop_index(rows, from, p.y));
    if let (Some(to), Some(rect)) = (target, rows.first().map(|r| r.1))
        && let Some(y) = line_y(rows, from, to)
    {
        let (l, r) = (rect.left() + 20.0, rect.right() - 8.0);
        ui.painter().line_segment([Pos2::new(l, y), Pos2::new(r, y)], Stroke::new(2.5, app.theme.ui.target_mask()));
    }
    ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
    ctx.request_repaint();
    if released {
        app.layers.op_drag = None;
        if let Some(to) = target.filter(|t| *t != from) {
            command::execute(app, LayerCommand::Op { id: layer, index: from, action: OpAction::MoveTo(to) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    /// Four ops, index 3 drawn at the top (y 0..24), index 0 at the bottom.
    fn rows() -> Vec<OpRow> {
        (0..4).map(|i| (3 - i, Rect::from_min_max(pos2(0.0, i as f32 * 24.0), pos2(100.0, (i + 1) as f32 * 24.0)))).collect()
    }

    #[test]
    fn dropping_on_the_upper_half_goes_above_that_row() {
        // Drag op 0 (bottom row) onto the upper half of op 2 (second row from the top, y 24..48).
        assert_eq!(drop_index(&rows(), 0, 28.0), Some(2));
        // Lower half: just below op 2, which is index 2 after the move because op 0 left from underneath.
        assert_eq!(drop_index(&rows(), 0, 44.0), Some(1));
    }

    #[test]
    fn moving_down_accounts_for_the_gap_left_behind() {
        // Drag op 3 (top row) onto the lower half of op 1 (y 48..72).
        assert_eq!(drop_index(&rows(), 3, 68.0), Some(1));
        // Upper half of op 1 puts it above op 1, i.e. index 2 once it has left the top.
        assert_eq!(drop_index(&rows(), 3, 52.0), Some(2));
    }

    #[test]
    fn the_line_sits_between_the_rows_the_op_will_land_between() {
        let rows = rows(); // op 3 at y 0..24, op 2 at 24..48, op 1 at 48..72, op 0 at 72..96
        // Op 0 moved to the top (index 3): the line is above the first row.
        assert_eq!(line_y(&rows, 0, 3), Some(0.0));
        // Op 0 moved to index 2: between op 3 (top) and op 2.
        assert_eq!(line_y(&rows, 0, 2), Some(24.0));
        // Op 3 moved to the bottom (index 0): below the last remaining row, op 0's bottom edge.
        assert_eq!(line_y(&rows, 3, 0), Some(96.0));
    }

    #[test]
    fn dropping_on_itself_changes_nothing_and_outside_rows_is_none() {
        let from = 2;
        let own = drop_index(&rows(), from, 30.0).unwrap();
        assert_eq!(own, from, "the upper half of its own row keeps its place");
        assert_eq!(drop_index(&rows(), from, 500.0), None);
    }
}

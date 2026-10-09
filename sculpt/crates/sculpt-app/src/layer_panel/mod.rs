//! The LAYERS panel: a Painter-style layer stack for sculpt layers.
//!
//! * `selection` multi-select model, `tree` rows read from the document,
//!   `row` one row, `menus` context menus, `add_bar` the buttons under the list,
//!   `dragdrop` + `drag_ui` row drag and drop, `command` every document change.
//!
//! Nothing here writes the document except through [`command::execute`].

pub mod add_bar;
pub mod breadcrumb;
pub mod command;
pub mod drag_ui;
pub mod hud;
pub mod dragdrop;
pub mod mask_ops;
pub mod menus;
pub mod row;
pub mod selection;
pub mod state;
pub mod switcher;
pub mod target;
pub mod thumbs;
pub mod tree;

use egui::{Rect, Sense, Ui, vec2};
use sculpt_core::LayerId;

pub use state::PanelState;

use crate::app::{SculptApp, Selection};

pub fn layers_panel(app: &mut SculptApp, ui: &mut Ui) {
    let header = Rect::from_min_size(ui.cursor().min, vec2(ui.available_width(), 24.0));
    crate::panels::panel_header(ui, app, "LAYERS", |_| {});
    let toggle = Rect::from_min_size(header.right_top() + vec2(-30.0, 2.0), vec2(22.0, 20.0));
    let tip = if app.layers.compact { "Comfortable rows" } else { "Compact rows" };
    let on = app.layers.compact;
    let density = ui.interact(toggle, ui.id().with("density"), Sense::click());
    if density.hovered() {
        ui.painter().rect_filled(toggle, 3.0, ui.visuals().widgets.hovered.bg_fill);
    }
    crate::icons::Icon::Density.paint(ui.painter(), toggle.shrink(3.0), if on { app.theme.ui.accent.0 } else { app.theme.weak_text() });
    if density.on_hover_text(tip).clicked() {
        app.layers.compact = !app.layers.compact;
        app.save_settings();
    }
    if crate::panels::sync_mask_edit(app) {
        app.selection = Selection::Layer;
    }
    if app.doc.is_none() {
        ui.label("Loading…");
        return;
    }
    seed_selection(app);
    app.layers.paint_kind = target::resolve(app).filter(|t| t.refusal.is_none()).map(|t| t.kind);
    app.layers.thumbs.begin_frame();
    if let Some(d) = app.doc.as_ref() {
        app.layers.thumbs.retain(d);
    }

    // Reserve the bar plus the spacing egui adds around it. Under-reserving makes the panel
    // grow by the shortfall every frame, because its height comes from this content.
    let list_h = (ui.available_height() - add_bar::BAR_H - add_bar::SLACK).max(40.0);
    let list_rect = Rect::from_min_size(ui.cursor().min, vec2(ui.available_width(), list_h));
    let mut rows = Vec::new();
    egui::ScrollArea::vertical().max_height(list_h).auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let nodes = tree::visible_nodes(app, app.doc.as_ref().unwrap());
        let order: Vec<LayerId> = nodes.iter().map(|n| n.id).collect();
        for n in &nodes {
            row::show(app, ui, n, &order, &mut rows);
            if !n.is_folder && n.has_mask && app.expanded.contains(&n.id) {
                crate::panels::mask_rows(app, ui, n.id, n.depth);
            }
        }
        crate::panels::base_row(app, ui);
        let rest = ui.available_size_before_wrap();
        if rest.y > 4.0 {
            let (_, empty) = ui.allocate_exact_size(rest, Sense::click());
            egui::Popup::context_menu(&empty).show(|ui| menus::empty_menu(app, ui));
        }
    });
    drag_ui::update(app, ui, &rows, list_rect);
    app.layers.rows = rows;
    add_bar::show(app, ui);
}

/// Keep the selection pointing at something real: the active layer if nothing else is selected.
fn seed_selection(app: &mut SculptApp) {
    let Some(doc) = app.doc.as_ref() else { return };
    if app.layers.selection.is_empty()
        && let Some(a) = doc.active_layer()
    {
        app.layers.selection.select_only(a);
    }
}

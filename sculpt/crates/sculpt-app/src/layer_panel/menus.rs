//! Right-click menus and the add-mask menu, per kind of row.
//!
//! Items that cannot apply are greyed, not hidden (as in Painter), and every
//! item shows its hotkey so menus teach the keyboard.

use egui::{Response, Ui};
use sculpt_core::LayerId;

use super::command::{self, LayerCommand};
use super::tree::Node;
use crate::app::{SculptApp, Selection};
use crate::keymap::Command;

/// One menu entry: label, optional hotkey from the keymap, enabled flag. Returns true when clicked.
fn item(app: &SculptApp, ui: &mut Ui, label: &str, key: Option<Command>, enabled: bool) -> bool {
    let mut b = egui::Button::new(label);
    if let Some(s) = key.and_then(|k| app.keymap.shortcut_text(ui.ctx(), k)) {
        b = b.shortcut_text(s);
    }
    let clicked = ui.add_enabled(enabled, b).clicked();
    if clicked {
        ui.close();
    }
    clicked
}

fn run(app: &mut SculptApp, ui: &Ui, cmd: LayerCommand) {
    let _ = ui;
    command::execute(app, cmd);
}

/// Menu for a layer or folder row.
pub fn row_menu(app: &mut SculptApp, ui: &mut Ui, n: &Node) {
    let multi = app.layers.selection.len() > 1;
    let can_merge = !n.is_folder && !multi;
    ui.set_min_width(210.0);
    if item(app, ui, "Rename", Some(Command::LayerRename), true) {
        run(app, ui, LayerCommand::Rename);
    }
    if item(app, ui, "Duplicate", Some(Command::LayerDuplicate), true) {
        run(app, ui, LayerCommand::Duplicate);
    }
    ui.separator();
    if item(app, ui, "Group into folder", Some(Command::LayerGroup), true) {
        run(app, ui, LayerCommand::Group);
    }
    if n.is_folder && item(app, ui, "Ungroup", Some(Command::LayerUngroup), !multi) {
        run(app, ui, LayerCommand::Ungroup);
    }
    if item(app, ui, "Merge down", Some(Command::LayerMergeDown), can_merge) {
        run(app, ui, LayerCommand::MergeDown);
    }
    ui.separator();
    if !n.is_folder {
        ui.menu_button("Add mask", |ui| add_mask_items(app, ui, n.id));
        ui.separator();
    }
    let solo_label = if n.soloed { "Unsolo" } else { "Solo" };
    if item(app, ui, solo_label, Some(Command::LayerSolo), true) {
        run(app, ui, LayerCommand::ToggleSolo(n.id));
    }
    let lock_label = if n.locked { "Unlock" } else { "Lock" };
    if item(app, ui, lock_label, Some(Command::LayerLock), true) {
        run(app, ui, LayerCommand::Edit { id: n.id, edit: command::MetaEdit::Locked(!n.locked), coalesce: false });
    }
    let hide_label = if n.visible { "Hide" } else { "Show" };
    if item(app, ui, hide_label, Some(Command::LayerHide), true) {
        run(app, ui, LayerCommand::Edit { id: n.id, edit: command::MetaEdit::Visible(!n.visible), coalesce: false });
    }
    ui.separator();
    if !n.is_folder && !multi && item(app, ui, "Flatten into base mesh", None, true) {
        run(app, ui, LayerCommand::Flatten);
    }
    if item(app, ui, "Delete", Some(Command::LayerDelete), true) {
        run(app, ui, LayerCommand::Delete);
    }
}

/// Menu for empty space below the rows.
pub fn empty_menu(app: &mut SculptApp, ui: &mut Ui) {
    if item(app, ui, "New layer", Some(Command::NewLayer), true) {
        run(app, ui, LayerCommand::NewLayer);
    }
    if item(app, ui, "New folder", Some(Command::NewFolder), true) {
        run(app, ui, LayerCommand::NewFolder);
    }
}

/// White / black / remove. One step each, then the mask is the paint target.
pub fn add_mask_items(app: &mut SculptApp, ui: &mut Ui, id: LayerId) {
    let has_mask = app.doc.as_ref().and_then(|d| d.layer(id)).is_some_and(|l| l.mask.is_some()) || app.mask_edit.as_ref().is_some_and(|(i, _)| *i == id);
    if item(app, ui, "White mask (shows everything)", None, !has_mask) {
        crate::panels::select_layer(app, Some(id), Selection::Layer);
        crate::panels::add_mask(app, 1.0);
    }
    if item(app, ui, "Black mask (hides everything)", None, !has_mask) {
        crate::panels::select_layer(app, Some(id), Selection::Layer);
        crate::panels::add_mask(app, 0.0);
    }
    if item(app, ui, "Remove mask", None, has_mask) {
        crate::panels::select_layer(app, Some(id), Selection::Layer);
        if let Some(d) = app.doc.as_mut() {
            let _ = d.set_layer_mask(id, None);
        }
        app.mask_edit = None;
        app.selection = Selection::Layer;
    }
}

/// Left-click menu on the "+" slot.
pub fn add_mask_menu_on(app: &mut SculptApp, resp: &Response, id: LayerId) {
    egui::Popup::menu(resp).show(|ui| add_mask_items(app, ui, id));
}

/// Right-click menu on a mask thumbnail.
pub fn mask_menu_on(app: &mut SculptApp, resp: &Response, id: LayerId) {
    egui::Popup::context_menu(resp).show(|ui| {
        ui.set_min_width(190.0);
        add_mask_items(app, ui, id);
    });
}

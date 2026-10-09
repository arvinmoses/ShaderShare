//! Right-click menus and the add-mask menu, per kind of row.
//!
//! Items that cannot apply are greyed, not hidden (as in Painter), and every
//! item shows its hotkey so menus teach the keyboard.

use egui::{Response, Ui};
use sculpt_core::LayerId;

use super::command::{self, LayerCommand};
use super::mask_ops::{self, MaskAction, OpAction, Preset};
use super::tree::Node;
use crate::app::SculptApp;
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
    if n.is_folder && item(app, ui, "Flatten folder", Some(Command::LayerFlatten), !multi) {
        run(app, ui, LayerCommand::FlattenFolder);
    }
    if !n.is_folder && !multi && item(app, ui, "Flatten into base mesh", None, true) {
        run(app, ui, LayerCommand::Flatten);
    }
    if item(app, ui, "Delete", Some(Command::LayerDelete), true) {
        run(app, ui, LayerCommand::Delete);
    }
}

/// Right-click menu in the viewport: the active layer's menu, then the global adds. Same items as the
/// panel, so nothing needs the pointer to leave the model.
pub fn viewport_menu(app: &mut SculptApp, ui: &mut Ui) {
    let id = app.layers.selection.primary().or_else(|| app.doc.as_ref().and_then(|d| d.active_layer()));
    let node = app.doc.as_ref().and_then(|d| super::tree::visible_nodes(app, d).into_iter().find(|n| Some(n.id) == id));
    if let Some(n) = node {
        ui.label(egui::RichText::new(&n.name).strong());
        ui.separator();
        row_menu(app, ui, &n);
        ui.separator();
    }
    empty_menu(app, ui);
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

/// Everything about a layer's mask in one menu. Without a mask it builds one in a single step
/// (white, black, from a bake, from noise, hand-painted); with one it adds ops and edits the mask.
pub fn add_mask_items(app: &mut SculptApp, ui: &mut Ui, id: LayerId) {
    let stack = app.mask_edit.as_ref().filter(|(i, _)| *i == id).map(|(_, s)| s.clone()).or_else(|| app.doc.as_ref().and_then(|d| d.layer(id)).and_then(|l| l.mask.clone()));
    let has_mask = stack.is_some();
    ui.set_min_width(200.0);
    let go = |app: &mut SculptApp, ui: &mut Ui, cmd: LayerCommand| {
        command::execute(app, cmd);
        ui.close();
    };
    if !has_mask {
        if item(app, ui, "White mask (shows everything)", None, true) {
            go(app, ui, LayerCommand::AddMask { id, preset: Preset::White });
        }
        if item(app, ui, "Black mask (hides everything)", None, true) {
            go(app, ui, LayerCommand::AddMask { id, preset: Preset::Black });
        }
        ui.separator();
    } else {
        ui.label(egui::RichText::new("ADD OP").small().weak());
    }
    ui.menu_button("From bake", |ui| {
        for (attr, label) in mask_ops::BAKES {
            if item(app, ui, label, None, true) {
                go(app, ui, LayerCommand::AddMask { id, preset: Preset::Bake(attr) });
            }
        }
    });
    ui.menu_button("From noise", |ui| {
        for (kind, label) in mask_ops::NOISES {
            if item(app, ui, label, None, true) {
                go(app, ui, LayerCommand::AddMask { id, preset: Preset::Noise(kind) });
            }
        }
    });
    if item(app, ui, "Hand-painted", None, true) {
        go(app, ui, LayerCommand::AddMask { id, preset: Preset::Paint });
    }
    ui.separator();
    let clip = app.layers.mask_clip.is_some();
    if has_mask {
        let enabled = stack.as_ref().is_none_or(|s| s.enabled);
        if item(app, ui, if enabled { "Disable mask" } else { "Enable mask" }, None, true) {
            go(app, ui, LayerCommand::Mask { id, action: MaskAction::Toggle });
        }
        if item(app, ui, "Invert mask", None, true) {
            go(app, ui, LayerCommand::Mask { id, action: MaskAction::Invert });
        }
        if item(app, ui, "Copy mask", None, true) {
            go(app, ui, LayerCommand::Mask { id, action: MaskAction::Copy });
        }
    }
    if item(app, ui, "Paste mask", None, clip) {
        go(app, ui, LayerCommand::Mask { id, action: MaskAction::Paste });
    }
    if has_mask {
        ui.separator();
        if item(app, ui, "Remove mask", None, true) {
            go(app, ui, LayerCommand::Mask { id, action: MaskAction::Remove });
        }
    }
}

/// Right-click menu on a mask op row. `index` counts from the bottom of the mask stack.
pub fn op_menu(app: &mut SculptApp, ui: &mut Ui, id: LayerId, index: usize, count: usize, enabled: bool) {
    ui.set_min_width(170.0);
    let go = |app: &mut SculptApp, ui: &mut Ui, action: OpAction| {
        command::execute(app, LayerCommand::Op { id, index, action });
        ui.close();
    };
    if item(app, ui, if enabled { "Disable" } else { "Enable" }, None, true) {
        go(app, ui, OpAction::Toggle);
    }
    if item(app, ui, "Duplicate", None, true) {
        go(app, ui, OpAction::Duplicate);
    }
    ui.separator();
    if item(app, ui, "Move up", None, index + 1 < count) {
        go(app, ui, OpAction::Raise);
    }
    if item(app, ui, "Move down", None, index > 0) {
        go(app, ui, OpAction::Lower);
    }
    ui.separator();
    if item(app, ui, "Delete", None, true) {
        go(app, ui, OpAction::Delete);
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

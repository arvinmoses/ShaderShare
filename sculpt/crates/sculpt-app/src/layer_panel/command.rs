//! Every change the layer panel makes to the document goes through here.
//!
//! The UI builds a [`LayerCommand`] and calls [`execute`]; it never touches the
//! document directly. That keeps undo consistent (each command is one engine
//! step) and gives hotkeys, menus and buttons a single behaviour to share.

use sculpt_core::{LayerBlend, LayerId, LayerMeta, Placement};

use crate::app::{SculptApp, Selection};
use crate::keymap::Command;

#[derive(Clone, Debug, PartialEq)]
pub enum MetaEdit {
    Name(String),
    Strength(f32),
    Visible(bool),
    Locked(bool),
    Collapsed(bool),
    Blend(LayerBlend),
}

impl MetaEdit {
    fn apply(&self, m: &mut LayerMeta) {
        match self {
            MetaEdit::Name(n) => m.name = n.clone(),
            MetaEdit::Strength(s) => m.opacity = *s,
            MetaEdit::Visible(v) => m.visible = *v,
            MetaEdit::Locked(l) => m.locked = *l,
            MetaEdit::Collapsed(c) => m.collapsed = *c,
            MetaEdit::Blend(b) => m.blend = *b,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerCommand {
    NewLayer,
    NewFolder,
    /// Duplicate the selection.
    Duplicate,
    MergeDown,
    /// Bake the layer into the base mesh. Not undoable: clears history.
    Flatten,
    /// Delete the selection.
    Delete,
    /// Group the selection into a new folder.
    Group,
    Ungroup,
    /// Start inline rename of the primary row.
    Rename,
    ToggleSolo(LayerId),
    Edit { id: LayerId, edit: MetaEdit, coalesce: bool },
    /// Move rows, or copy them there when `copy` is set (Ctrl+drag).
    Move { ids: Vec<LayerId>, at: Placement, copy: bool },
}

/// The layer command a keymap command stands for, if any.
pub fn from_keymap(cmd: Command, app: &SculptApp) -> Option<LayerCommand> {
    let primary = app.layers.selection.primary().or_else(|| app.doc.as_ref().and_then(|d| d.active_layer()));
    Some(match cmd {
        Command::NewLayer => LayerCommand::NewLayer,
        Command::NewFolder => LayerCommand::NewFolder,
        Command::LayerDuplicate => LayerCommand::Duplicate,
        Command::LayerMergeDown => LayerCommand::MergeDown,
        Command::LayerDelete => LayerCommand::Delete,
        Command::LayerGroup => LayerCommand::Group,
        Command::LayerUngroup => LayerCommand::Ungroup,
        Command::LayerRename => LayerCommand::Rename,
        Command::LayerSolo => LayerCommand::ToggleSolo(primary?),
        Command::LayerLock => {
            let id = primary?;
            let locked = app.doc.as_ref()?.layer(id)?.locked;
            LayerCommand::Edit { id, edit: MetaEdit::Locked(!locked), coalesce: false }
        }
        Command::LayerHide => {
            let id = primary?;
            let visible = app.doc.as_ref()?.layer(id)?.visible;
            LayerCommand::Edit { id, edit: MetaEdit::Visible(!visible), coalesce: false }
        }
        _ => return None,
    })
}

/// Rows the selection stands for, top of the stack first. Falls back to the active layer.
fn targets(app: &SculptApp) -> Vec<LayerId> {
    let Some(doc) = app.doc.as_ref() else { return Vec::new() };
    let order: Vec<LayerId> = doc.layer_tree().iter().map(|r| r.id).collect();
    let picked = app.layers.selection.in_order(&order);
    if picked.is_empty() { doc.active_layer().into_iter().collect() } else { picked }
}

/// Where a new row goes: above the selected layer, inside a selected folder, else on top.
fn insert_point(app: &SculptApp) -> Placement {
    let Some(doc) = app.doc.as_ref() else { return Placement::Top };
    match app.layers.selection.primary().or_else(|| doc.active_layer()).and_then(|id| doc.layer(id)) {
        Some(l) if l.is_folder() => Placement::Into(l.id),
        Some(l) => Placement::Above(l.id),
        None => Placement::Top,
    }
}

fn fail(app: &mut SculptApp, what: &str, e: impl std::fmt::Display) {
    app.status = format!("{what}: {e}");
}

pub fn execute(app: &mut SculptApp, cmd: LayerCommand) {
    if app.doc.is_none() {
        return;
    }
    match cmd {
        LayerCommand::NewLayer => {
            let at = insert_point(app);
            let doc = app.doc.as_mut().unwrap();
            let n = doc.layers().iter().filter(|l| !l.is_folder()).count() + 1;
            match doc.insert_layer(&format!("Layer {n}"), at) {
                Ok(id) => {
                    reveal(app, id);
                    app.expanded.insert(id);
                    crate::panels::select_layer(app, Some(id), Selection::Layer);
                    app.layers.selection.select_only(id);
                }
                Err(e) => fail(app, "New layer", e),
            }
        }
        LayerCommand::NewFolder => {
            let at = insert_point(app);
            let doc = app.doc.as_mut().unwrap();
            let n = doc.layers().iter().filter(|l| l.is_folder()).count() + 1;
            match doc.insert_folder(&format!("Folder {n}"), at) {
                Ok(id) => {
                    reveal(app, id);
                    app.layers.selection.select_only(id);
                    app.selection = Selection::Layer;
                }
                Err(e) => fail(app, "New folder", e),
            }
        }
        LayerCommand::Duplicate => {
            let ids = targets(app);
            let mut last = None;
            for id in ids {
                match app.doc.as_mut().unwrap().duplicate_layer(id) {
                    Ok(copy) => last = Some(copy),
                    Err(e) => fail(app, "Duplicate", e),
                }
            }
            if let Some(id) = last {
                reveal(app, id);
                app.layers.selection.select_only(id);
            }
        }
        LayerCommand::MergeDown => {
            let Some(id) = targets(app).first().copied() else { return };
            match app.doc.as_mut().unwrap().merge_down(id) {
                Ok(into) => {
                    app.layers.selection.select_only(into);
                    app.status = "Merged down".into();
                }
                Err(e) => fail(app, "Merge down", e),
            }
        }
        LayerCommand::Flatten => {
            let Some(id) = targets(app).first().copied() else { return };
            match app.doc.as_mut().unwrap().flatten_layer(id) {
                Ok(()) => app.status = "Flattened into the base mesh (undo history cleared)".into(),
                Err(e) => fail(app, "Flatten", e),
            }
        }
        LayerCommand::Delete => {
            for id in targets(app) {
                // A selected folder takes its children with it, so they may already be gone.
                if app.doc.as_ref().unwrap().layer(id).is_some()
                    && let Err(e) = app.doc.as_mut().unwrap().delete_layer(id)
                {
                    fail(app, "Delete", e);
                }
            }
            app.layers.selection.clear();
        }
        LayerCommand::Group => {
            let ids = targets(app);
            if ids.is_empty() {
                return;
            }
            let doc = app.doc.as_mut().unwrap();
            let n = doc.layers().iter().filter(|l| l.is_folder()).count() + 1;
            match doc.group_layers(&ids, &format!("Folder {n}")) {
                Ok(folder) => app.layers.selection.select_only(folder),
                Err(e) => fail(app, "Group", e),
            }
        }
        LayerCommand::Ungroup => {
            let Some(id) = app.layers.selection.primary() else { return };
            match app.doc.as_mut().unwrap().ungroup(id) {
                Ok(()) => app.layers.selection.clear(),
                Err(e) => fail(app, "Ungroup", e),
            }
        }
        LayerCommand::Rename => {
            let Some(id) = targets(app).first().copied() else { return };
            let name = app.doc.as_ref().unwrap().layer(id).map(|l| l.name.clone()).unwrap_or_default();
            app.layers.rename = Some((id, name));
            return;
        }
        LayerCommand::ToggleSolo(id) => {
            let doc = app.doc.as_mut().unwrap();
            let next = if doc.solo() == Some(id) { None } else { Some(id) };
            if let Err(e) = doc.set_solo(next) {
                fail(app, "Solo", e);
            }
            return;
        }
        LayerCommand::Edit { id, edit, coalesce } => {
            let doc = app.doc.as_mut().unwrap();
            let Some(layer) = doc.layer(id) else { return };
            let mut meta = layer.meta();
            edit.apply(&mut meta);
            if let MetaEdit::Name(n) = &edit {
                meta.name = n.trim().to_string();
                if meta.name.is_empty() {
                    return;
                }
            }
            if let Err(e) = doc.set_layer_meta(id, meta, coalesce) {
                fail(app, "Edit layer", e);
            }
            return;
        }
        LayerCommand::Move { ids, at, copy } => {
            for id in ids {
                let doc = app.doc.as_mut().unwrap();
                let result = if copy { doc.duplicate_layer(id).and_then(|c| doc.move_layer(c, at)) } else { doc.move_layer(id, at) };
                if let Err(e) = result {
                    fail(app, "Move", e);
                    break;
                }
            }
            if let Placement::Into(f) = at {
                // Show what was just dropped.
                reveal_in(app, f);
            }
        }
    }
    after_change(app);
}

/// Open every collapsed folder above `id` so the row is visible.
fn reveal(app: &mut SculptApp, id: LayerId) {
    let Some(doc) = app.doc.as_ref() else { return };
    let mut chain = Vec::new();
    let mut cur = doc.layer(id).and_then(|l| l.parent);
    while let Some(p) = cur {
        chain.push(p);
        cur = doc.layer(p).and_then(|l| l.parent);
    }
    for f in chain {
        reveal_in(app, f);
    }
}

fn reveal_in(app: &mut SculptApp, folder: LayerId) {
    let doc = app.doc.as_mut().unwrap();
    if let Some(l) = doc.layer(folder).filter(|l| l.collapsed) {
        let mut m = l.meta();
        m.collapsed = false;
        let _ = doc.set_layer_meta(folder, m, true);
    }
}

/// After any structural change (command, undo or redo): drop stale selection and resync mask editing.
pub fn after_change(app: &mut SculptApp) {
    let Some(doc) = app.doc.as_ref() else { return };
    app.layers.selection.retain(|id| doc.layer(id).is_some());
    if app.layers.rename.as_ref().is_some_and(|(id, _)| doc.layer(*id).is_none()) {
        app.layers.rename = None;
    }
    // Force the mask editing copy to re-read the (possibly different) active layer.
    app.mask_edit = None;
    app.last_active = None;
    if crate::panels::sync_mask_edit(app) {
        app.selection = Selection::Layer;
    }
    if let Some(a) = app.doc.as_ref().and_then(|d| d.active_layer())
        && app.layers.selection.is_empty()
    {
        app.layers.selection.select_only(a);
    }
}

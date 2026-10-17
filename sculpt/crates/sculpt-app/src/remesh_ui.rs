//! The Remesh checkpoint in the app: sculpting on a procedural stack, and going back to edit it.
//!
//! The first brush stroke on a stack with generator layers remeshes it into a clean mesh and starts a sculpt layer on
//! top; the procedural stack is kept underneath. "Edit stack below" swaps the stack back in to change it; "Done" remeshes
//! it again and carries the sculpt layers onto the new mesh. Undo straight after the remesh puts the stack back.

use std::time::Instant;

use egui::{RichText, Ui};
use sculpt_core::Document;
use sculpt_core::document::SolidSettings;

use crate::app::SculptApp;
use crate::panels::section;

/// True if strokes on `doc` would land under generator layers, which would displace them again.
pub fn needs_remesh(doc: &Document) -> bool {
    doc.layers().iter().any(|l| l.is_generator())
}

fn after_swap(app: &mut SculptApp) {
    app.layers.selection.clear();
    app.solid_preview.reset();
    app.maybe_build_lod();
}

/// Remesh the stack into a clean mesh to sculpt on, keeping the stack underneath. Returns false (with the reason in the
/// status bar) if it could not.
pub fn remesh_to_sculpt(app: &mut SculptApp) -> bool {
    let Some(doc) = app.doc.take() else { return false };
    let started = Instant::now();
    let input = doc.solid_input();
    // The same voxel size as the preview, so the remesh starts from its cached work.
    let voxel = input.surface_bounds().extent().max_element() / app.solid_ui.divisions as f32;
    let result = input.run(&SolidSettings { voxel }).map(|(mesh, _)| mesh);
    match result {
        Ok(mesh) => match Document::remeshed(mesh, doc, voxel) {
            Ok(top) => {
                app.status = format!("Remeshed the stack to sculpt on ({:.0} ms). Edit stack below changes it; your strokes follow.", started.elapsed().as_secs_f64() * 1e3);
                app.doc = Some(top);
                after_swap(app);
                true
            }
            Err(e) => {
                app.status = format!("Remesh failed: {e}");
                false
            }
        },
        Err(e) => {
            app.doc = Some(doc);
            app.status = format!("Remesh failed: {e}");
            false
        }
    }
}

/// Swap the procedural stack back in to edit it; the sculpt waits aside.
pub fn edit_stack_below(app: &mut SculptApp) {
    let Some(mut top) = app.doc.take() else { return };
    match top.take_remesh_source() {
        Some(source) => {
            app.remesh_top = Some(top);
            app.doc = Some(source);
            app.status = "Editing the stack below your sculpt. Done remeshes it and brings your strokes back.".into();
            after_swap(app);
        }
        None => app.doc = Some(top),
    }
}

/// Remesh the edited stack and carry the sculpt layers onto it.
pub fn finish_stack_edit(app: &mut SculptApp) {
    let (Some(top), Some(source)) = (app.remesh_top.take(), app.doc.take()) else { return };
    let started = Instant::now();
    let rebuilt = source.solidify(&SolidSettings { voxel: top.remesh_voxel() }).and_then(|(mesh, _)| top.reprojected_onto(mesh));
    match rebuilt {
        Ok(mut new) => {
            new.set_remesh_source(source, top.remesh_voxel());
            app.status = format!("Remeshed and carried your sculpt across ({:.0} ms)", started.elapsed().as_secs_f64() * 1e3);
            app.doc = Some(new);
        }
        Err(e) => {
            // Keep both as they were.
            app.status = format!("Could not remesh the stack: {e}");
            app.doc = Some(source);
            app.remesh_top = Some(top);
            return;
        }
    }
    after_swap(app);
}

/// Undo straight after a remesh (nothing sculpted yet): put the stack back.
pub fn undo_remesh(app: &mut SculptApp) -> bool {
    let Some(doc) = app.doc.as_mut() else { return false };
    if doc.can_undo() || !doc.has_remesh_source() || app.remesh_top.is_some() || doc.layers().iter().any(|l| l.delta_bytes() > 0) {
        return false;
    }
    let source = doc.take_remesh_source().unwrap();
    app.doc = Some(source);
    app.status = "Undo remesh".into();
    after_swap(app);
    true
}

/// The Remesh section at the top of Properties.
pub fn sections(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    if app.remesh_top.is_some() {
        let mut done = false;
        section(ui, fs, "remesh", "STACK BELOW YOUR SCULPT", |ui| {
            ui.label(RichText::new("You are editing the procedural stack under your sculpt layers. Done remeshes it and carries your strokes onto the new surface.").size(fs * 0.85).color(weak));
            done = ui.button("Done: back to sculpting").clicked();
        });
        if done {
            finish_stack_edit(app);
        }
    } else if app.doc.as_ref().is_some_and(|d| d.has_remesh_source()) {
        let mut edit = false;
        section(ui, fs, "remesh", "REMESHED", |ui| {
            ui.label(RichText::new("Sculpting on a clean mesh made from the procedural stack.").size(fs * 0.85).color(weak));
            edit = ui.button("Edit stack below").on_hover_text("Change the generators and noise under your sculpt; your strokes follow when you are done.").clicked();
        });
        if edit {
            edit_stack_below(app);
        }
    }
}

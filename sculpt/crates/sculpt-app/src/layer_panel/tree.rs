//! The rows the layer list shows, read once per frame from the document.

use sculpt_core::{Document, LayerBlend, LayerId};

use crate::app::SculptApp;

/// Everything a row needs to draw, copied out of the document so drawing can mutate the app freely.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: LayerId,
    pub depth: usize,
    pub name: String,
    pub is_folder: bool,
    pub visible: bool,
    pub locked: bool,
    pub strength: f32,
    pub blend: LayerBlend,
    pub collapsed: bool,
    pub has_children: bool,
    pub has_mask: bool,
    /// Number of ops in the mask stack.
    pub mask_ops: usize,
    pub soloed: bool,
    /// Contributes nothing right now (hidden ancestor or another layer soloed).
    pub excluded: bool,
}

/// Visible rows, top of the stack first. Children of collapsed folders are skipped.
pub fn visible_nodes(app: &SculptApp, doc: &Document) -> Vec<Node> {
    let mut out = Vec::new();
    let mut hide_below: Option<usize> = None;
    for row in doc.layer_tree() {
        if let Some(d) = hide_below {
            if row.depth > d {
                continue;
            }
            hide_below = None;
        }
        let Some(l) = doc.layer(row.id) else { continue };
        let editing = app.mask_edit.as_ref().filter(|(i, _)| *i == l.id).map(|(_, s)| s.layers.len());
        let mask_ops = editing.or_else(|| l.mask.as_ref().map(|m| m.layers.len())).unwrap_or(0);
        let has_children = l.is_folder() && doc.layers().iter().any(|c| c.parent == Some(l.id));
        out.push(Node {
            id: l.id,
            depth: row.depth,
            name: l.name.clone(),
            is_folder: l.is_folder(),
            visible: l.visible,
            locked: l.locked,
            strength: l.opacity,
            blend: l.blend,
            collapsed: l.collapsed,
            has_children,
            has_mask: l.mask.is_some() || editing.is_some(),
            mask_ops,
            soloed: doc.solo() == Some(l.id),
            excluded: l.effective_scale() == 0.0 && l.opacity != 0.0 && l.visible,
        });
        if l.is_folder() && l.collapsed {
            hide_below = Some(row.depth);
        }
    }
    out
}

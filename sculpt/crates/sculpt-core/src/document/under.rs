//! The surface under the active layer: the base and the layers below it, as they composite. Layer brushes steer by it,
//! so each layer works over what the ones below it make, as it would were they merged into the base. The active layer
//! plays no part (its tilt, turning under its own strokes, folded faces), and neither do the layers above it.
//!
//! Nothing but the active layer changes during a stroke, so what is worked out for a leaf holds until the stroke ends;
//! it is made for the leaves a dab reaches, as it reaches them.

use std::sync::Arc;

use glam::Vec3;
use rayon::prelude::*;

use super::Document;
use crate::layers::LayerId;
use crate::mesh::NO_VERT;

/// Per leaf: the surface's (place, normal) at each of its vertices; empty where no layer below holds anything there
/// (the base's own); `None` where not yet worked out.
type Leaves = Vec<Option<Box<[(Vec3, Vec3)]>>>;

/// What the open stroke has worked out, for one active layer and topology.
pub(crate) struct UnderCache {
    active: LayerId,
    topology: u64,
    leaves: Arc<Leaves>,
}

/// The surface under the active layer about a dab (see the module's notes).
pub(crate) struct Under {
    /// The layers below the active one that hold something, in the order they composite.
    below: Vec<usize>,
    eps: f32,
    base_normals: Arc<Vec<Vec3>>,
    leaves: Arc<Leaves>,
}

impl Under {
    /// Where vertex `v` sits on the surface.
    #[inline]
    pub(crate) fn pos(&self, doc: &Document, v: usize) -> Vec3 {
        if self.below.is_empty() {
            return doc.base[v];
        }
        match self.known(doc, v) {
            Some(Some(p)) => p.0,
            Some(None) => doc.base[v],
            None => self.point(doc, v),
        }
    }

    /// The surface's normal at vertex `v`.
    #[inline]
    pub(crate) fn normal(&self, doc: &Document, v: usize) -> Vec3 {
        if self.below.is_empty() {
            return self.base_normals[v];
        }
        match self.known(doc, v) {
            Some(Some(p)) => p.1,
            Some(None) => self.base_normals[v],
            None => self.normal_at(doc, v),
        }
    }

    /// What is worked out at `v`: `Some(None)` where its leaf is the base's own.
    #[inline]
    fn known(&self, doc: &Document, v: usize) -> Option<Option<(Vec3, Vec3)>> {
        let l = doc.bvh.vert_leaf[v] as usize;
        let c = self.leaves.get(l)?.as_deref()?;
        Some((!c.is_empty()).then(|| c[v - doc.bvh.leaves[l].owned.start as usize]))
    }

    fn point(&self, doc: &Document, v: usize) -> Vec3 {
        let mut acc = Vec3::ZERO;
        for &i in &self.below {
            let layer = &doc.layers[i];
            let off = doc.layer_offset(&layer.chunks, v);
            // (Nothing held: no blend changes what is below.)
            if off != Vec3::ZERO {
                acc = layer.blend.apply(acc, off, layer.weight(v), self.base_normals[v], self.eps);
            }
        }
        doc.base[v] + acc
    }

    fn normal_at(&self, doc: &Document, v: usize) -> Vec3 {
        let mut n = Vec3::ZERO;
        for &f in doc.topo.vert_faces.row(v) {
            let f = &doc.faces[f as usize];
            let p = |k: usize| self.point(doc, f[k] as usize);
            n += if f[3] == NO_VERT { (p(1) - p(0)).cross(p(2) - p(0)) } else { (p(2) - p(0)).cross(p(3) - p(1)) };
        }
        n.normalize_or(Vec3::Z)
    }
}

impl Document {
    /// The surface under the active layer, worked out for the leaves within `reach` of `center` (others are worked out
    /// as asked for, slower). With no layer below holding anything, the base's.
    pub(crate) fn under(&mut self, center: Vec3, reach: f32) -> Under {
        let base_normals = self.base_normals();
        let eps = (self.bvh.bounds().extent().length() * 1e-4).max(1e-9);
        let below: Vec<usize> = match self.active.and_then(|a| self.layer_index(a).ok()) {
            Some(active) => self
                .composite_order()
                .into_iter()
                .take_while(|&i| i != active)
                .filter(|&i| {
                    let l = &self.layers[i];
                    l.gen_cache.is_none() && l.scale != 0.0 && l.chunks.iter().any(|c| c.is_some())
                })
                .collect(),
            None => Vec::new(),
        };
        let mut under = Under { below, eps, base_normals, leaves: Arc::new(Vec::new()) };
        let Some(active) = self.active.filter(|_| !under.below.is_empty()) else {
            return under;
        };
        let mut leaves = match self.under_cache.take() {
            Some(c) if c.active == active && c.topology == self.topology_id => c.leaves,
            _ => Arc::new(Vec::new()),
        };
        {
            let list = Arc::make_mut(&mut leaves);
            list.resize(self.bvh.leaves.len(), None);
            let mut near = Vec::new();
            self.bvh.query_sphere(center, reach, &mut near);
            near.retain(|&l| list[l as usize].is_none());
            under.leaves = Arc::new(Vec::new());
            let doc = &*self;
            let made: Vec<(u32, Box<[(Vec3, Vec3)]>)> = near
                .par_iter()
                .map(|&l| {
                    let held = under.below.iter().any(|&i| doc.layers[i].chunks.get(l as usize).is_some_and(|c| c.is_some()));
                    let out: Box<[(Vec3, Vec3)]> = if held {
                        doc.bvh.leaves[l as usize].owned_range().map(|v| (under.point(doc, v), under.normal_at(doc, v))).collect()
                    } else {
                        Box::new([])
                    };
                    (l, out)
                })
                .collect();
            for (l, out) in made {
                list[l as usize] = Some(out);
            }
        }
        under.leaves = leaves.clone();
        self.under_cache = Some(UnderCache { active, topology: self.topology_id, leaves });
        under
    }
}

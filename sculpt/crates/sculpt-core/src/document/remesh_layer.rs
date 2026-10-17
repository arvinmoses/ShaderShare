//! The Remesh checkpoint: a clean mesh to sculpt on, with the procedural stack kept underneath.
//!
//! Sculpting directly on a stack with generator layers fights them: strokes land under the generators, which displace
//! the stroked vertices again. So before sculpting, the stack is solidified into a clean, closed mesh, and that mesh is
//! what the brushes work on, in ordinary sculpt layers. The stack it came from is kept as the document's remesh source.
//!
//! To change the stack later, the source is taken out, edited like any document, and put back with
//! [`Document::rebuilt_from_source`]: the stack is solidified again, and every sculpt layer, painted channel and freeze
//! value is carried onto the new mesh. Each new vertex takes what the old surface held at its closest point,
//! interpolated across that triangle. Strokes survive moderate changes below them; where a cut removes the sculpted
//! surface, those strokes go with it, and new surface starts unsculpted.

use std::collections::BTreeMap;

use glam::Vec3;
use rayon::prelude::*;

use super::{Document, SolidSettings};
use crate::bvh::Bvh;
use crate::generator::voxel::MeshSdf;
use crate::layer_ops::Placement;
use crate::layers::SculptLayer;
use crate::mesh::{PolyMesh, Topology};
use crate::Result;

impl Document {
    /// Whether this document is a remeshed stack, with the stack it came from kept underneath.
    pub fn has_remesh_source(&self) -> bool {
        self.remesh_source.is_some()
    }

    /// The lattice spacing the stack was remeshed at.
    pub fn remesh_voxel(&self) -> f32 {
        self.remesh_voxel
    }

    /// A new document to sculpt on: `mesh` (the solidified `source`) with one empty sculpt layer, active, and `source`
    /// kept underneath.
    pub fn remeshed(mesh: PolyMesh, source: Document, voxel: f32) -> Result<Document> {
        let mut top = Document::from_mesh(mesh)?;
        top.insert_layer("Sculpt", Placement::Top)?;
        top.undo.clear();
        top.remesh_source = Some(Box::new(source));
        top.remesh_voxel = voxel;
        Ok(top)
    }

    /// Put a stack underneath this document (after remeshing it at `voxel`).
    pub fn set_remesh_source(&mut self, source: Document, voxel: f32) {
        self.remesh_source = Some(Box::new(source));
        self.remesh_voxel = voxel;
    }

    /// Take the stack underneath out to edit it. Put it back with [`Document::rebuilt_from_source`].
    pub fn take_remesh_source(&mut self) -> Option<Document> {
        self.remesh_source.take().map(|b| *b)
    }

    /// Solidify the edited `source` again and carry this document's layers onto the result.
    pub fn rebuilt_from_source(&self, source: Document) -> Result<Document> {
        let (mesh, _) = source.solidify(&SolidSettings { voxel: self.remesh_voxel })?;
        let mut out = self.reprojected_onto(mesh)?;
        out.remesh_source = Some(Box::new(source));
        out.remesh_voxel = self.remesh_voxel;
        Ok(out)
    }

    /// A document on `mesh` holding this one's layers, channels and freeze, each new vertex taking the value at its
    /// closest point on this document's base surface.
    pub fn reprojected_onto(&self, mesh: PolyMesh) -> Result<Document> {
        let mut out = Document::from_mesh(mesh)?;
        // Closest points on the old base surface. Its own tree follows the sculpted positions, so build one for the base.
        let (mut positions, mut faces) = (self.base.clone(), self.faces.clone());
        let (bvh, old_to_new) = Bvh::build(&mut positions, &mut faces);
        let mut new_to_old = vec![0u32; old_to_new.len()];
        for (old, &new) in old_to_new.iter().enumerate() {
            new_to_old[new as usize] = old as u32;
        }
        let topo = Topology::build(positions.len(), &faces);
        let sdf = MeshSdf::build_owned(bvh, positions, faces, &topo);
        let bound = self.bvh.bounds().extent().length() * 4.0 + 1.0;
        let samples: Vec<([u32; 3], [f32; 3])> = out
            .base
            .par_iter()
            .map(|&p| match sdf.closest(p, bound) {
                Some((_, tri, w)) => (tri.map(|v| new_to_old[v as usize]), w),
                None => ([0; 3], [1.0, 0.0, 0.0]),
            })
            .collect();
        let carry_vec = |vals: &[Vec3]| -> Vec<Vec3> { samples.par_iter().map(|(t, w)| vals[t[0] as usize] * w[0] + vals[t[1] as usize] * w[1] + vals[t[2] as usize] * w[2]).collect() };
        let carry = |vals: &[f32]| -> Vec<f32> { samples.par_iter().map(|(t, w)| vals[t[0] as usize] * w[0] + vals[t[1] as usize] * w[1] + vals[t[2] as usize] * w[2]).collect() };

        out.freeze = carry(&self.freeze).into_iter().map(|f| f.clamp(0.0, 1.0)).collect();
        out.channels = self.channels.iter().map(|(k, v)| (k.clone(), carry(v))).collect::<BTreeMap<_, _>>();
        let layers: Vec<SculptLayer> = self
            .layers
            .iter()
            .map(|layer| {
                let mut dense = vec![Vec3::ZERO; self.vertex_count()];
                for l in layer.allocated_leaves() {
                    let r = self.bvh.leaves[l as usize].owned_range();
                    dense[r].copy_from_slice(layer.chunks[l as usize].as_ref().unwrap());
                }
                let moved = carry_vec(&dense);
                let chunks = out
                    .bvh
                    .leaves
                    .par_iter()
                    .map(|leaf| {
                        let s = &moved[leaf.owned_range()];
                        s.iter().any(|d| *d != Vec3::ZERO).then(|| s.into())
                    })
                    .collect();
                SculptLayer { chunks, mask_values: None, ..layer.clone_meta() }
            })
            .collect();
        out.layers = layers;
        out.active = self.active;
        out.solo = self.solo;
        out.next_layer_id = self.next_layer_id;
        for i in 0..out.layers.len() {
            if let Some(stack) = out.layers[i].mask.clone() {
                out.layers[i].mask_values = out.mask_values_for(&stack)?;
            }
        }
        out.refresh_scales();
        out.recomposite_all();
        Ok(out)
    }
}

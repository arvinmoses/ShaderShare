//! The sculpt document: mesh, spatial index, layers, masks, channels and undo.
//!
//! All per-vertex arrays use the *internal* vertex order produced by the BVH
//! (contiguous per leaf). Data crossing the API boundary to files or other
//! programs uses the *canonical* order (import order, then Catmull-Clark order
//! after each subdivision); see [`Document::to_canonical`].

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use glam::Vec3;
use rayon::prelude::*;

use crate::bake::{self, BakeInput, BakeSettings, MeshAttribute};
use crate::bvh::{Bvh, split_ranges_mut};
use crate::geom::{Aabb, Ray};
use crate::layers::{Chunk, LayerId, SculptLayer};
use crate::mask::{self, MaskContext, MaskStack};
use crate::mesh::{self, Face, NO_VERT, PolyMesh, Topology, face_normal, face_verts};
use crate::subdiv::Subdivider;
use crate::undo::{self, Group, Record, UndoStack};
use crate::{Error, Result};

/// Per-leaf sparse vertex displacements produced by a brush kernel.
/// Indices are local to the leaf's owned range.
#[derive(Debug, Default)]
pub struct Displacements {
    pub(crate) leaves: Vec<(u32, Vec<(u32, Vec3)>)>,
}

impl Displacements {
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
    pub fn vertex_count(&self) -> usize {
        self.leaves.iter().map(|(_, l)| l.len()).sum()
    }
}

/// Which spatial leaves changed since the renderer last asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirtySet {
    None,
    All,
    /// Sorted leaf ids; upload each leaf's owned vertex range.
    Leaves(Vec<u32>),
}

#[derive(Debug, Default)]
pub(crate) struct DirtyTracker {
    all: bool,
    marks: Vec<bool>,
    any: bool,
}

impl DirtyTracker {
    fn all() -> DirtyTracker {
        DirtyTracker { all: true, ..Default::default() }
    }
    pub(crate) fn mark(&mut self, leaf: u32, leaves: usize) {
        if self.all {
            return;
        }
        if self.marks.len() != leaves {
            self.marks = vec![false; leaves];
        }
        self.marks[leaf as usize] = true;
        self.any = true;
    }
    pub(crate) fn mark_all(&mut self) {
        self.all = true;
    }
    fn take(&mut self) -> DirtySet {
        let out = if self.all {
            DirtySet::All
        } else if self.any {
            DirtySet::Leaves(self.marks.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i as u32).collect())
        } else {
            DirtySet::None
        };
        self.all = false;
        self.any = false;
        self.marks.iter_mut().for_each(|m| *m = false);
        out
    }
}

static NEXT_TOPOLOGY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn new_topology_id() -> u64 {
    NEXT_TOPOLOGY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaintTarget {
    /// Mudbox "freeze": 1 = fully protected from sculpting and posing.
    Freeze,
    /// A named per-vertex channel (hand-painted mask data).
    Channel(String),
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub face: u32,
    /// Closest corner of the hit face.
    pub vertex: u32,
}

#[derive(Clone, Copy, Debug)]
enum SculptTarget {
    Base,
    Layer(usize),
}

pub struct Document {
    pub(crate) faces: Vec<Face>,
    pub(crate) base: Vec<Vec3>,
    /// Rest pose: the reference space procedural masks are evaluated in.
    /// Posing never moves it, so noise sticks to the surface like a texture.
    pub(crate) rest: Vec<Vec3>,
    pub(crate) positions: Vec<Vec3>,
    pub(crate) normals: Vec<Vec3>,
    pub(crate) topo: Topology,
    pub(crate) bvh: Bvh,
    pub(crate) freeze: Vec<f32>,
    pub(crate) channels: BTreeMap<String, Vec<f32>>,
    pub(crate) layers: Vec<SculptLayer>,
    pub(crate) active: Option<LayerId>,
    pub(crate) solo: Option<LayerId>,
    /// Bumped on every change a cache could care about (see `edit_serial`).
    pub(crate) serial: u64,
    pub(crate) next_layer_id: u32,
    /// `canonical index -> internal index`.
    pub(crate) canonical_to_internal: Vec<u32>,
    pub(crate) undo: UndoStack,
    /// Composite positions at stroke start, captured lazily per touched leaf.
    pub(crate) stroke_origin: Vec<Option<Box<[Vec3]>>>,
    pub(crate) dirty_channels: BTreeSet<String>,
    pub(crate) level: u32,
    /// Positions/normals changed (renderer upload tracking).
    pub(crate) geometry_dirty: DirtyTracker,
    /// Freeze / channels / layer masks changed (overlay upload tracking).
    pub(crate) scalar_dirty: DirtyTracker,
    pub(crate) topology_id: u64,
    /// Level-of-detail tree for dense meshes, and the topology it was built for.
    pub(crate) lod: Option<(u64, std::sync::Arc<crate::lod::LodTree>)>,
    /// Free-form, round-tripped by the file format (UI state, app metadata).
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Document {
    pub fn from_mesh(mesh: PolyMesh) -> Result<Document> {
        mesh.validate()?;
        let PolyMesh { mut positions, mut faces } = mesh;
        let (bvh, perm) = Bvh::build(&mut positions, &mut faces);
        let topo = Topology::build(positions.len(), &faces);
        let normals = mesh::compute_normals(&positions, &faces, &topo);
        let n = positions.len();
        Ok(Document {
            faces,
            base: positions.clone(),
            rest: positions.clone(),
            positions,
            normals,
            topo,
            bvh,
            freeze: vec![0.0; n],
            channels: BTreeMap::new(),
            layers: Vec::new(),
            active: None,
            solo: None,
            serial: 0,
            next_layer_id: 1,
            canonical_to_internal: perm,
            undo: UndoStack::default(),
            stroke_origin: Vec::new(),
            dirty_channels: BTreeSet::new(),
            level: 0,
            geometry_dirty: DirtyTracker::all(),
            scalar_dirty: DirtyTracker::all(),
            topology_id: new_topology_id(),
            lod: None,
            metadata: BTreeMap::new(),
        })
    }

    // ----------------------------------------------------------------- access

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }
    pub fn base_positions(&self) -> &[Vec3] {
        &self.base
    }
    pub fn rest_positions(&self) -> &[Vec3] {
        &self.rest
    }

    /// Make the current base mesh the new rest pose (re-anchors procedural masks).
    pub fn store_rest_pose(&mut self) -> Result<()> {
        self.rest = self.base.clone();
        self.refresh_layer_masks()
    }
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }
    pub fn topology(&self) -> &Topology {
        &self.topo
    }
    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }
    pub fn freeze(&self) -> &[f32] {
        &self.freeze
    }
    pub fn channel(&self, name: &str) -> Option<&[f32]> {
        self.channels.get(name).map(|c| c.as_slice())
    }
    pub fn channels(&self) -> &BTreeMap<String, Vec<f32>> {
        &self.channels
    }
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Unique id of the current face/vertex layout; changes on subdivision
    /// and load, signalling renderers to rebuild index buffers.
    pub fn topology_id(&self) -> u64 {
        self.topology_id
    }

    /// Build (or rebuild) the level-of-detail tree for the current topology. Takes seconds on tens of millions of triangles.
    pub fn build_lod(&mut self, params: crate::lod::LodParams) {
        let tree = crate::lod::LodTree::build(&self.positions, &self.faces, &self.bvh, params);
        self.lod = Some((self.topology_id, std::sync::Arc::new(tree)));
    }

    /// Re-simplify level-of-detail patches that edits have outdated, for at most `budget`. Returns how many remain.
    pub fn refresh_lod(&mut self, budget: std::time::Duration) -> usize {
        let Some((t, tree)) = self.lod.as_mut() else { return 0 };
        if *t != self.topology_id {
            return 0;
        }
        match std::sync::Arc::get_mut(tree) {
            Some(tree) => tree.refresh(&self.positions, budget),
            None => 0,
        }
    }

    /// Pool ranges the last [`refresh_lod`](Self::refresh_lod) rewrote, for the renderer to copy.
    pub fn take_lod_updates(&mut self) -> Vec<(u32, u32)> {
        match self.lod.as_mut().and_then(|(_, t)| std::sync::Arc::get_mut(t)) {
            Some(tree) => tree.take_updates(),
            None => Vec::new(),
        }
    }

    /// The level-of-detail tree, if one was built for the current topology.
    pub fn lod(&self) -> Option<&std::sync::Arc<crate::lod::LodTree>> {
        self.lod.as_ref().filter(|(t, _)| *t == self.topology_id).map(|(_, tree)| tree)
    }

    /// Leaves whose positions/normals changed since the last call.
    pub fn take_geometry_dirty(&mut self) -> DirtySet {
        self.geometry_dirty.take()
    }

    /// Leaves whose freeze, channel or layer-mask values changed since the last call.
    pub fn take_scalar_dirty(&mut self) -> DirtySet {
        self.scalar_dirty.take()
    }
    pub fn bounds(&self) -> Aabb {
        self.bvh.bounds()
    }

    /// Internal -> canonical order for export.
    pub fn to_canonical<T: Copy + Default + Send + Sync>(&self, data: &[T]) -> Vec<T> {
        mesh::unpermute(&self.canonical_to_internal, data)
    }

    /// Canonical -> internal order for import.
    pub fn from_canonical<T: Copy + Default + Send + Sync>(&self, data: &[T]) -> Vec<T> {
        mesh::permute(&self.canonical_to_internal, data)
    }

    pub fn canonical_index(&self, canonical: usize) -> usize {
        self.canonical_to_internal[canonical] as usize
    }

    /// Faces with canonical vertex indices.
    pub fn canonical_faces(&self) -> Vec<Face> {
        let mut internal_to_canonical = vec![0u32; self.vertex_count()];
        for (c, &i) in self.canonical_to_internal.iter().enumerate() {
            internal_to_canonical[i as usize] = c as u32;
        }
        self.faces
            .par_iter()
            .map(|f| f.map(|v| if v == NO_VERT { NO_VERT } else { internal_to_canonical[v as usize] }))
            .collect()
    }

    /// Composited (or base-only) mesh in canonical order.
    pub fn export_mesh(&self, composited: bool) -> PolyMesh {
        let src = if composited { &self.positions } else { &self.base };
        PolyMesh { positions: self.to_canonical(src), faces: self.canonical_faces() }
    }

    // --------------------------------------------------------------- channels

    /// Set a channel from internally ordered values.
    pub fn set_channel(&mut self, name: &str, values: Vec<f32>) -> Result<()> {
        if values.len() != self.vertex_count() {
            return Err(Error::InvalidData(format!("channel {name}: {} values for {} vertices", values.len(), self.vertex_count())));
        }
        self.channels.insert(name.into(), values);
        self.dirty_channels.insert(name.into());
        self.scalar_dirty.mark_all();
        self.undo.clear();
        self.refresh_dependent_masks()
    }

    /// Import a channel produced by another program, in canonical vertex order.
    pub fn import_channel(&mut self, name: &str, canonical_values: &[f32]) -> Result<()> {
        if canonical_values.len() != self.vertex_count() {
            return Err(Error::InvalidData(format!(
                "channel {name}: {} values for {} vertices",
                canonical_values.len(),
                self.vertex_count()
            )));
        }
        let v = self.from_canonical(canonical_values);
        self.set_channel(name, v)
    }

    pub fn remove_channel(&mut self, name: &str) {
        self.channels.remove(name);
        self.scalar_dirty.mark_all();
        self.undo.clear();
    }

    pub fn set_freeze(&mut self, values: Vec<f32>) -> Result<()> {
        if values.len() != self.vertex_count() {
            return Err(Error::InvalidData("freeze size mismatch".into()));
        }
        self.freeze = values;
        self.scalar_dirty.mark_all();
        self.undo.clear();
        Ok(())
    }

    // ----------------------------------------------------------------- layers

    pub fn layers(&self) -> &[SculptLayer] {
        &self.layers
    }

    pub fn layer(&self, id: LayerId) -> Option<&SculptLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub(crate) fn layer_index(&self, id: LayerId) -> Result<usize> {
        self.layers.iter().position(|l| l.id == id).ok_or(Error::NoSuchLayer(id.0))
    }

    pub fn active_layer(&self) -> Option<LayerId> {
        self.active
    }

    /// `None` sculpts directly on the base mesh.
    pub fn set_active_layer(&mut self, id: Option<LayerId>) -> Result<()> {
        if let Some(id) = id {
            let i = self.layer_index(id)?;
            if self.layers[i].is_folder() {
                return Err(Error::InvalidData(format!("'{}' is a folder and cannot be sculpted on", self.layers[i].name)));
            }
        }
        self.active = id;
        Ok(())
    }

    /// Adds a layer on top of the stack and makes it active.
    pub fn add_layer(&mut self, name: &str) -> LayerId {
        let id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        self.layers.push(SculptLayer::new(id, name, self.bvh.leaves.len()));
        self.active = Some(id);
        id
    }

    pub fn remove_layer(&mut self, id: LayerId) -> Result<()> {
        self.delete_layer(id)
    }

    pub fn rename_layer(&mut self, id: LayerId, name: &str) -> Result<()> {
        let i = self.layer_index(id)?;
        self.layers[i].name = name.into();
        Ok(())
    }

    /// The strength slider. Only leaves where the layer has data are recomposited.
    pub fn set_layer_opacity(&mut self, id: LayerId, opacity: f32) -> Result<()> {
        let i = self.layer_index(id)?;
        if self.layers[i].opacity != opacity {
            self.layers[i].opacity = opacity;
            self.refresh_scales();
        }
        Ok(())
    }

    pub fn set_layer_visible(&mut self, id: LayerId, visible: bool) -> Result<()> {
        let i = self.layer_index(id)?;
        if self.layers[i].visible != visible {
            self.layers[i].visible = visible;
            self.refresh_scales();
        }
        Ok(())
    }

    pub fn set_layer_locked(&mut self, id: LayerId, locked: bool) -> Result<()> {
        let i = self.layer_index(id)?;
        self.layers[i].locked = locked;
        Ok(())
    }

    /// Attach (or clear) a mask stack controlling where the layer applies.
    pub fn set_layer_mask(&mut self, id: LayerId, stack: Option<MaskStack>) -> Result<()> {
        let i = self.layer_index(id)?;
        let values = match &stack {
            Some(s) => self.mask_values_for(s)?,
            None => None,
        };
        let layer = &mut self.layers[i];
        layer.mask = stack;
        layer.mask_values = values;
        self.scalar_dirty.mark_all();
        let leaves = layer.allocated_leaves();
        self.recomposite(&leaves);
        Ok(())
    }

    /// Re-evaluate every layer mask (after bakes, imports or paint strokes).
    pub fn refresh_layer_masks(&mut self) -> Result<()> {
        let mut leaves = BTreeSet::new();
        for i in 0..self.layers.len() {
            if let Some(stack) = self.layers[i].mask.clone() {
                let values = self.mask_values_for(&stack)?;
                self.layers[i].mask_values = values;
                leaves.extend(self.layers[i].allocated_leaves());
                self.scalar_dirty.mark_all();
            }
        }
        self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
        Ok(())
    }

    fn refresh_dependent_masks(&mut self) -> Result<()> {
        if self.dirty_channels.is_empty() {
            return Ok(());
        }
        let dirty = std::mem::take(&mut self.dirty_channels);
        let mut leaves = BTreeSet::new();
        for i in 0..self.layers.len() {
            let Some(stack) = self.layers[i].mask.clone() else { continue };
            let mut used = Vec::new();
            stack.channels(&mut used);
            if used.iter().any(|c| dirty.contains(c)) {
                self.scalar_dirty.mark_all();
                let values = self.mask_values_for(&stack)?;
                self.layers[i].mask_values = values;
                leaves.extend(self.layers[i].allocated_leaves());
            }
        }
        self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
        Ok(())
    }

    /// Bake `layer` into the base mesh at its current strength and remove it.
    pub fn flatten_layer(&mut self, id: LayerId) -> Result<()> {
        let i = self.layer_index(id)?;
        if self.layers[i].is_folder() {
            return Err(Error::InvalidData("flatten works on layers, not folders".into()));
        }
        if self.layers[i].blend != crate::layers::LayerBlend::Add {
            return Err(Error::InvalidData("only layers in Add mode can be flattened into the base".into()));
        }
        // One undo step: the touched base leaves are snapshotted, then the layer leaves the stack.
        self.begin_stroke("Flatten layer");
        let leaves = self.layers[i].allocated_leaves();
        for &l in &leaves {
            self.snapshot(&undo::Target::Base, l);
        }
        for &l in &leaves {
            let r = self.bvh.leaves[l as usize].owned_range();
            let chunk = self.layers[i].chunks[l as usize].as_ref().unwrap();
            for (k, v) in r.enumerate() {
                self.base[v] += chunk[k] * self.layers[i].weight(v);
            }
        }
        let mut ops = vec![undo::StructOp::Remove { index: i }];
        if self.active == Some(id) {
            let next = self.layers.iter().rev().find(|l| !l.is_folder() && l.id != id).map(|l| l.id);
            ops.push(undo::StructOp::Active { id: next });
        }
        self.exec_in_open(ops);
        self.end_stroke();
        Ok(())
    }

    // ------------------------------------------------------------------ masks

    /// Evaluate a mask stack, baking any mesh attributes it needs first.
    /// Procedural sources read the rest pose (positions and normals), so the
    /// result is deterministic and unaffected by sculpting or posing.
    /// Per-vertex values for a layer's mask, or `None` (applies everywhere) when the mask is disabled.
    pub fn mask_values_for(&mut self, stack: &MaskStack) -> Result<Option<Vec<f32>>> {
        if !stack.enabled {
            return Ok(None);
        }
        self.evaluate_mask(stack).map(Some)
    }

    pub fn evaluate_mask(&mut self, stack: &MaskStack) -> Result<Vec<f32>> {
        let mut bakes = Vec::new();
        stack.required_bakes(&mut bakes);
        for b in bakes {
            if !self.channels.contains_key(b.channel_name()) {
                self.bake(b, &BakeSettings::default());
            }
        }
        let rest_normals = if stack.uses_normals() { mesh::compute_normals(&self.rest, &self.faces, &self.topo) } else { Vec::new() };
        mask::evaluate(
            stack,
            &MaskContext { positions: &self.rest, normals: &rest_normals, topology: &self.topo, channels: &self.channels },
        )
    }

    /// Bake a mesh attribute from the current composited surface into its channel.
    pub fn bake(&mut self, attribute: MeshAttribute, settings: &BakeSettings) {
        let input = BakeInput {
            positions: &self.positions,
            normals: &self.normals,
            faces: &self.faces,
            topology: &self.topo,
            bvh: &self.bvh,
        };
        let values = match attribute {
            MeshAttribute::Curvature => bake::curvature(&input, settings),
            MeshAttribute::Cavity => bake::cavity_from_curvature(&bake::curvature(&input, settings)),
            MeshAttribute::AmbientOcclusion => bake::ambient_occlusion(&input, settings),
            MeshAttribute::Thickness => bake::thickness(&input, settings),
        };
        self.channels.insert(attribute.channel_name().into(), values);
        self.dirty_channels.insert(attribute.channel_name().into());
        self.scalar_dirty.mark_all();
    }

    // --------------------------------------------------------------- queries

    pub fn raycast(&self, ray: &Ray) -> Option<SurfaceHit> {
        let hit = self.bvh.raycast(&self.positions, &self.faces, ray, f32::INFINITY)?;
        let f = &self.faces[hit.face as usize];
        let vertex = *face_verts(f)
            .iter()
            .min_by(|a, b| {
                self.positions[**a as usize].distance_squared(hit.point).total_cmp(&self.positions[**b as usize].distance_squared(hit.point))
            })
            .unwrap();
        let mut normal = face_normal(&self.positions, f).normalize_or(Vec3::Z);
        if normal.dot(ray.dir) > 0.0 {
            normal = -normal;
        }
        Some(SurfaceHit { point: hit.point, normal, face: hit.face, vertex })
    }

    /// Snap a point near the surface onto it (used to re-project interpolated dabs).
    pub fn project_to_surface(&self, point: Vec3, normal: Vec3, search: f32) -> Option<SurfaceHit> {
        let n = normal.normalize_or(Vec3::Z);
        let ray = Ray { origin: point + n * search, dir: -n };
        let hit = self.bvh.raycast(&self.positions, &self.faces, &ray, search * 2.0)?;
        let f = &self.faces[hit.face as usize];
        let vertex = face_verts(f)[0];
        Some(SurfaceHit { point: hit.point, normal: face_normal(&self.positions, f).normalize_or(n), face: hit.face, vertex })
    }

    /// Falloff-weighted average position and normal of the surface in a sphere.
    /// With `front`, only vertices facing that direction contribute.
    pub fn sample_area(&self, center: Vec3, radius: f32, front: Option<Vec3>) -> Option<(Vec3, Vec3)> {
        let mut leaves = Vec::new();
        self.bvh.query_sphere(center, radius, &mut leaves);
        let r2 = radius * radius;
        let (sp, sn, sw) = leaves
            .par_iter()
            .map(|&l| {
                let mut acc = (Vec3::ZERO, Vec3::ZERO, 0.0f32);
                for v in self.bvh.leaves[l as usize].owned_range() {
                    let p = self.positions[v];
                    let d2 = p.distance_squared(center);
                    if d2 > r2 {
                        continue;
                    }
                    let n = self.normals[v];
                    if front.is_some_and(|f| n.dot(f) <= 0.0) {
                        continue;
                    }
                    let w = 1.0 - d2 / r2;
                    acc.0 += p * w;
                    acc.1 += n * w;
                    acc.2 += w;
                }
                acc
            })
            .reduce(|| (Vec3::ZERO, Vec3::ZERO, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        (sw > 0.0).then(|| (sp / sw, sn.normalize_or(front.unwrap_or(Vec3::Z))))
    }

    // ------------------------------------------------------- brush pipeline

    /// Where vertex `v` was when the current stroke began.
    #[inline]
    pub fn stroke_origin(&self, v: usize) -> Vec3 {
        let l = self.bvh.vert_leaf[v] as usize;
        match self.stroke_origin.get(l).and_then(|c| c.as_ref()) {
            Some(c) => c[v - self.bvh.leaves[l].owned.start as usize],
            None => self.positions[v],
        }
    }

    /// Evaluate `kernel(doc, vertex, distance)` for every vertex within
    /// `radius` of `center`, in parallel per leaf. Results are pre-scaled by
    /// the freeze mask.
    pub fn compute_displacements<F>(&self, center: Vec3, radius: f32, kernel: F) -> Displacements
    where
        F: Fn(&Document, usize, f32) -> Option<Vec3> + Sync,
    {
        let mut leaves = Vec::new();
        self.bvh.query_sphere(center, radius, &mut leaves);
        let r2 = radius * radius;
        let leaves = leaves
            .par_iter()
            .filter_map(|&l| {
                let leaf = &self.bvh.leaves[l as usize];
                let start = leaf.owned.start as usize;
                let mut list = Vec::new();
                for v in leaf.owned_range() {
                    let d2 = self.positions[v].distance_squared(center);
                    if d2 > r2 {
                        continue;
                    }
                    let free = 1.0 - self.freeze[v];
                    if free <= 0.0 {
                        continue;
                    }
                    if let Some(d) = kernel(self, v, d2.sqrt()) {
                        let d = d * free;
                        if d != Vec3::ZERO && d.is_finite() {
                            list.push(((v - start) as u32, d));
                        }
                    }
                }
                (!list.is_empty()).then_some((l, list))
            })
            .collect();
        Displacements { leaves }
    }

    fn sculpt_target(&self) -> Result<SculptTarget> {
        match self.active {
            None => Ok(SculptTarget::Base),
            Some(id) => {
                let i = self.layer_index(id)?;
                let l = &self.layers[i];
                if l.locked {
                    return Err(Error::LayerLocked(l.name.clone()));
                }
                if l.is_folder() {
                    return Err(Error::InvalidData(format!("'{}' is a folder and cannot be sculpted on", l.name)));
                }
                if !l.visible || (l.scale == 0.0 && l.opacity != 0.0) {
                    return Err(Error::LayerHidden(l.name.clone()));
                }
                Ok(SculptTarget::Layer(i))
            }
        }
    }

    /// Write displacements into the active layer (or base) and update the
    /// composite, normals and bounds of the touched leaves.
    pub fn apply_displacements(&mut self, disp: &Displacements) -> Result<()> {
        if disp.is_empty() {
            return Ok(());
        }
        let target = self.sculpt_target()?;
        let leaf_ids: Vec<u32> = disp.leaves.iter().map(|(l, _)| *l).collect();
        let utarget = match target {
            SculptTarget::Base => undo::Target::Base,
            SculptTarget::Layer(i) => undo::Target::Layer(self.layers[i].id),
        };
        if self.stroke_origin.len() != self.bvh.leaves.len() {
            self.stroke_origin = vec![None; self.bvh.leaves.len()];
        }
        for &l in &leaf_ids {
            self.snapshot(&utarget, l);
            if self.stroke_origin[l as usize].is_none() {
                let r = self.bvh.leaves[l as usize].owned_range();
                self.stroke_origin[l as usize] = Some(self.positions[r].into());
            }
        }
        let ranges: Vec<Range<usize>> = leaf_ids.iter().map(|&l| self.bvh.leaves[l as usize].owned_range()).collect();
        let pos_slices = split_ranges_mut(&mut self.positions, ranges.iter().cloned());

        match target {
            SculptTarget::Base => {
                let base_slices = split_ranges_mut(&mut self.base, ranges.iter().cloned());
                disp.leaves.par_iter().zip(pos_slices).zip(base_slices).for_each(|(((_, list), pos), base)| {
                    for &(k, d) in list {
                        base[k as usize] += d;
                        pos[k as usize] += d;
                    }
                });
            }
            SculptTarget::Layer(i) => {
                let leaves = &self.bvh.leaves;
                let layer = &mut self.layers[i];
                for &l in &leaf_ids {
                    let len = leaves[l as usize].owned_len();
                    layer.chunks[l as usize].get_or_insert_with(|| vec![Vec3::ZERO; len].into_boxed_slice());
                }
                let (scale, blend) = (layer.scale, layer.blend);
                let sign = if blend == crate::layers::LayerBlend::Subtract { -1.0 } else { 1.0 };
                let mask_values = layer.mask_values.as_deref();
                let mut chunk_refs: Vec<&mut Box<[Vec3]>> = Vec::with_capacity(leaf_ids.len());
                let mut wanted = leaf_ids.iter().peekable();
                for (l, c) in layer.chunks.iter_mut().enumerate() {
                    if wanted.peek().is_some_and(|&&w| w as usize == l) {
                        wanted.next();
                        chunk_refs.push(c.as_mut().unwrap());
                    }
                }
                disp.leaves.par_iter().zip(chunk_refs).zip(pos_slices).zip(ranges.par_iter()).for_each(
                    |((((_, list), chunk), pos), r)| {
                        for &(k, d) in list {
                            let v = r.start + k as usize;
                            let w = scale * mask_values.map_or(1.0, |m| m[v]);
                            chunk[k as usize] += d;
                            if blend.is_linear() {
                                pos[k as usize] += d * (w * sign);
                            }
                        }
                    },
                );
            }
        }
        // Layers whose blend is not a plain add/subtract are re-composited, since a dab can change
        // what is shown in ways that depend on the layers beneath.
        if let SculptTarget::Layer(i) = target
            && !self.layers[i].blend.is_linear()
        {
            self.recomposite(&leaf_ids);
        } else {
            self.geometry_changed(&leaf_ids);
        }
        Ok(())
    }

    /// Paint a scalar target (freeze or a named channel) with falloff.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        target: &PaintTarget,
        center: Vec3,
        radius: f32,
        falloff: &crate::brush::Falloff,
        strength: f32,
        value: f32,
        front: Option<Vec3>,
    ) {
        if let PaintTarget::Channel(name) = target
            && !self.channels.contains_key(name) {
                self.snapshot_absent_channel(name);
                self.channels.insert(name.clone(), vec![0.0; self.vertex_count()]);
            }
        let mut leaves = Vec::new();
        self.bvh.query_sphere(center, radius, &mut leaves);
        let r2 = radius * radius;
        let values: &[f32] = match target {
            PaintTarget::Freeze => &self.freeze,
            PaintTarget::Channel(n) => &self.channels[n],
        };
        let updates: Vec<(u32, Vec<(u32, f32)>)> = leaves
            .par_iter()
            .filter_map(|&l| {
                let leaf = &self.bvh.leaves[l as usize];
                let mut list = Vec::new();
                for v in leaf.owned_range() {
                    let d2 = self.positions[v].distance_squared(center);
                    if d2 > r2 || front.is_some_and(|f| self.normals[v].dot(f) <= 0.0) {
                        continue;
                    }
                    let a = (falloff.eval(d2.sqrt() / radius) * strength).clamp(0.0, 1.0);
                    if a > 0.0 {
                        list.push((v as u32, values[v] + (value - values[v]) * a));
                    }
                }
                (!list.is_empty()).then_some((l, list))
            })
            .collect();
        let utarget = match target {
            PaintTarget::Freeze => undo::Target::Freeze,
            PaintTarget::Channel(n) => undo::Target::Channel(n.clone()),
        };
        for (l, _) in &updates {
            self.snapshot(&utarget, *l);
        }
        let values = match target {
            PaintTarget::Freeze => &mut self.freeze,
            PaintTarget::Channel(n) => {
                self.dirty_channels.insert(n.clone());
                self.channels.get_mut(n).unwrap()
            }
        };
        let nleaves = self.bvh.leaves.len();
        for (l, list) in updates {
            self.scalar_dirty.mark(l, nleaves);
            for (v, x) in list {
                values[v as usize] = x;
            }
        }
    }

    // ------------------------------------------------------- geometry upkeep

    /// Rebuild composite positions for `leaves` from base + layers.
    pub(crate) fn recomposite(&mut self, leaves: &[u32]) {
        if leaves.is_empty() {
            return;
        }
        let mut leaves = leaves.to_vec();
        leaves.sort_unstable();
        leaves.dedup();
        let ranges: Vec<Range<usize>> = leaves.iter().map(|&l| self.bvh.leaves[l as usize].owned_range()).collect();
        let order = self.composite_order();
        let eps = (self.bvh.bounds().extent().length() * 1e-4).max(1e-9);
        let slices = split_ranges_mut(&mut self.positions, ranges.iter().cloned());
        let (base, layers, normals) = (&self.base, &self.layers, &self.normals);
        slices.into_par_iter().zip(leaves.par_iter()).zip(ranges.par_iter()).for_each(|((pos, &l), r)| {
            pos.copy_from_slice(&base[r.clone()]);
            for &i in &order {
                let layer = &layers[i];
                let Some(chunk) = &layer.chunks[l as usize] else { continue };
                if layer.blend == crate::layers::LayerBlend::Add {
                    for (k, p) in pos.iter_mut().enumerate() {
                        *p += chunk[k] * layer.weight(r.start + k);
                    }
                } else {
                    for (k, p) in pos.iter_mut().enumerate() {
                        let v = r.start + k;
                        let acc = *p - base[v];
                        *p = base[v] + layer.blend.apply(acc, chunk[k], layer.weight(v), normals[v], eps);
                    }
                }
            }
        });
        self.geometry_changed(&leaves);
    }

    pub(crate) fn recomposite_all(&mut self) {
        let all: Vec<u32> = (0..self.bvh.leaves.len() as u32).collect();
        self.recomposite(&all);
    }

    /// Normals and bounds after positions of vertices owned by `leaves` moved.
    pub(crate) fn geometry_changed(&mut self, leaves: &[u32]) {
        self.serial += 1;
        let mut mark = vec![false; self.bvh.leaves.len()];
        for &l in leaves {
            mark[l as usize] = true;
            for &n in &self.bvh.leaf_neighbors[l as usize] {
                mark[n as usize] = true;
            }
        }
        let set: Vec<u32> = mark.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i as u32).collect();
        let (positions, faces, topo, bvh) = (&self.positions, &self.faces, &self.topo, &self.bvh);
        let normals: Vec<Vec<(u32, Vec3)>> = set
            .par_iter()
            .map(|&l| {
                bvh.leaves[l as usize]
                    .verts
                    .iter()
                    .map(|&v| (v, mesh::vertex_normal(positions, faces, topo, v as usize)))
                    .collect()
            })
            .collect();
        let nleaves = self.bvh.leaves.len();
        for &l in &set {
            self.geometry_dirty.mark(l, nleaves);
        }
        if let Some(tree) = self.lod.as_mut().filter(|(t, _)| *t == self.topology_id).and_then(|(_, tree)| std::sync::Arc::get_mut(tree)) {
            tree.mark_leaves(&set);
        }
        for list in normals {
            for (v, n) in list {
                self.normals[v as usize] = n;
                // Normals of borrowed vertices belong to their owner leaf's upload range.
                self.geometry_dirty.mark(self.bvh.vert_leaf[v as usize], nleaves);
            }
        }
        self.bvh.refit_leaves(&self.positions, &self.faces, &set);
    }

    // ------------------------------------------------------------------- undo

    pub(crate) fn snapshot(&mut self, target: &undo::Target, leaf: u32) {
        let group = self.undo.open.get_or_insert_with(|| Group::new("Edit"));
        if !group.wants(target, leaf) {
            return;
        }
        let r = self.bvh.leaves[leaf as usize].owned_range();
        let data = match target {
            undo::Target::Base => undo::Data::Positions(self.base[r].into()),
            undo::Target::Layer(id) => {
                let l = self.layers.iter().find(|l| l.id == *id).unwrap();
                undo::Data::Deltas(l.chunks[leaf as usize].clone())
            }
            undo::Target::Freeze => undo::Data::Scalars(self.freeze[r].into()),
            undo::Target::Channel(n) => undo::Data::Scalars(self.channels[n][r].into()),
            undo::Target::Structure => unreachable!("structure edits are recorded by layer_ops"),
        };
        group.records.push(Record { target: target.clone(), leaf, data });
    }

    pub(crate) fn snapshot_absent_channel(&mut self, name: &str) {
        let group = self.undo.open.get_or_insert_with(|| Group::new("Edit"));
        let target = undo::Target::Channel(name.into());
        // Leaf u32::MAX marks "whole channel".
        if group.wants(&target, u32::MAX) {
            group.records.push(Record { target, leaf: u32::MAX, data: undo::Data::Absent });
        }
    }

    /// Start an undoable operation (a brush stroke, a pose, ...).
    pub fn begin_stroke(&mut self, label: &str) {
        self.end_stroke();
        self.undo.open = Some(Group::new(label));
    }

    /// Finish the open operation and refresh masks fed by painted channels.
    pub fn end_stroke(&mut self) {
        self.stroke_origin.iter_mut().for_each(|c| *c = None);
        self.undo.commit();
        // Mask stacks only fail on missing data, which painting cannot cause.
        let _ = self.refresh_dependent_masks();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.undo.is_empty() || self.undo.open.as_ref().is_some_and(|g| !g.records.is_empty())
    }

    pub fn undo_labels(&self) -> Vec<&str> {
        self.undo.undo.iter().map(|g| g.label.as_str()).collect()
    }

    pub fn undo(&mut self) -> bool {
        self.end_stroke();
        let Some(group) = self.undo.undo.pop() else { return false };
        let inverse = self.swap_group(group);
        self.undo.redo.push(inverse);
        true
    }

    pub fn redo(&mut self) -> bool {
        self.end_stroke();
        let Some(group) = self.undo.redo.pop() else { return false };
        let inverse = self.swap_group(group);
        self.undo.undo.push(inverse);
        true
    }

    fn swap_group(&mut self, group: Group) -> Group {
        let mut inverse = Group::new(&group.label);
        let mut leaves = BTreeSet::new();
        for rec in group.records.into_iter().rev() {
            let Record { target, leaf, data } = rec;
            let data = match data {
                undo::Data::Op(op) => {
                    let inverse_op = self.apply_struct(op, &mut leaves);
                    inverse.records.push(Record { target, leaf, data: undo::Data::Op(inverse_op) });
                    continue;
                }
                other => other,
            };
            if leaf == u32::MAX {
                // Whole-channel creation/removal.
                if let undo::Target::Channel(name) = &target {
                    let old = self.channels.remove(name);
                    let restored = match data {
                        undo::Data::Absent => None,
                        undo::Data::Scalars(s) => Some(s.into_vec()),
                        _ => unreachable!(),
                    };
                    if let Some(r) = restored {
                        self.channels.insert(name.clone(), r);
                    }
                    let data = old.map_or(undo::Data::Absent, |o| undo::Data::Scalars(o.into_boxed_slice()));
                    inverse.records.push(Record { target: target.clone(), leaf, data });
                    self.dirty_channels.insert(name.clone());
                    self.scalar_dirty.mark_all();
                }
                continue;
            }
            let r = self.bvh.leaves[leaf as usize].owned_range();
            let current = match (&target, data) {
                (undo::Target::Base, undo::Data::Positions(p)) => {
                    let cur: Box<[Vec3]> = self.base[r.clone()].into();
                    self.base[r].copy_from_slice(&p);
                    leaves.insert(leaf);
                    undo::Data::Positions(cur)
                }
                (undo::Target::Layer(id), undo::Data::Deltas(c)) => {
                    let Some(layer) = self.layers.iter_mut().find(|l| l.id == *id) else { continue };
                    let cur: Chunk = std::mem::replace(&mut layer.chunks[leaf as usize], c);
                    leaves.insert(leaf);
                    undo::Data::Deltas(cur)
                }
                (undo::Target::Freeze, undo::Data::Scalars(s)) => {
                    let cur: Box<[f32]> = self.freeze[r.clone()].into();
                    self.freeze[r].copy_from_slice(&s);
                    self.scalar_dirty.mark(leaf, self.bvh.leaves.len());
                    undo::Data::Scalars(cur)
                }
                (undo::Target::Channel(n), undo::Data::Scalars(s)) => {
                    let Some(ch) = self.channels.get_mut(n) else { continue };
                    let cur: Box<[f32]> = ch[r.clone()].into();
                    ch[r].copy_from_slice(&s);
                    self.scalar_dirty.mark(leaf, self.bvh.leaves.len());
                    self.dirty_channels.insert(n.clone());
                    undo::Data::Scalars(cur)
                }
                _ => unreachable!("undo record kind mismatch"),
            };
            inverse.records.push(Record { target, leaf, data: current });
        }
        let _ = self.refresh_dependent_masks();
        self.recomposite(&leaves.into_iter().collect::<Vec<_>>());
        inverse
    }

    // ------------------------------------------------------------ subdivide

    /// Catmull-Clark subdivide everything: base, every layer's deltas, freeze
    /// and channels. Layers keep working at the new level.
    pub fn subdivide(&mut self) -> Result<()> {
        self.end_stroke();
        // Subdivide in canonical order so the new canonical order is exactly
        // Catmull-Clark order of the canonical mesh, reproducible by other tools.
        let sub = Subdivider::new(self.vertex_count(), &self.canonical_faces());
        let mut faces = sub.new_faces().to_vec();
        let mut positions = sub.apply(&self.to_canonical(&self.positions));
        let (bvh, perm) = Bvh::build(&mut positions, &mut faces);

        let lift = |vals: &[f32]| mesh::permute(&perm, &sub.apply(&self.to_canonical(vals)));
        let base = mesh::permute(&perm, &sub.apply(&self.to_canonical(&self.base)));
        let rest = mesh::permute(&perm, &sub.apply(&self.to_canonical(&self.rest)));
        let freeze: Vec<f32> = lift(&self.freeze).into_iter().map(|f| f.clamp(0.0, 1.0)).collect();
        let channels: BTreeMap<String, Vec<f32>> = self.channels.iter().map(|(k, v)| (k.clone(), lift(v))).collect();

        let layers: Vec<SculptLayer> = self
            .layers
            .iter()
            .map(|layer| {
                let mut dense = vec![Vec3::ZERO; self.vertex_count()];
                for l in layer.allocated_leaves() {
                    let r = self.bvh.leaves[l as usize].owned_range();
                    dense[r].copy_from_slice(layer.chunks[l as usize].as_ref().unwrap());
                }
                let dense = mesh::permute(&perm, &sub.apply(&self.to_canonical(&dense)));
                let chunks = bvh
                    .leaves
                    .par_iter()
                    .map(|leaf| {
                        let s = &dense[leaf.owned_range()];
                        s.iter().any(|d| *d != Vec3::ZERO).then(|| s.into())
                    })
                    .collect();
                SculptLayer { chunks, mask_values: None, ..layer.clone_meta() }
            })
            .collect();

        let topo = Topology::build(positions.len(), &faces);
        let normals = mesh::compute_normals(&positions, &faces, &topo);
        self.faces = faces;
        self.base = base;
        self.rest = rest;
        self.positions = positions;
        self.normals = normals;
        self.topo = topo;
        self.bvh = bvh;
        self.freeze = freeze;
        self.channels = channels;
        self.layers = layers;
        self.canonical_to_internal = perm;
        self.level += 1;
        self.topology_id = new_topology_id();
        self.geometry_dirty.mark_all();
        self.scalar_dirty.mark_all();
        self.undo.clear();
        self.stroke_origin.clear();
        for i in 0..self.layers.len() {
            if let Some(stack) = self.layers[i].mask.clone() {
                self.layers[i].mask_values = self.mask_values_for(&stack)?;
            }
        }
        self.recomposite_all();
        Ok(())
    }

    /// Internal: rebuild from canonical-order parts (used by the loader).
    pub(crate) fn from_parts(parts: crate::io::project::Parts) -> Result<Document> {
        let mut doc = Document::from_mesh(PolyMesh { positions: parts.base, faces: parts.faces })?;
        doc.level = parts.level;
        if let Some(rest) = parts.rest {
            doc.rest = doc.from_canonical(&rest);
        }
        doc.metadata = parts.metadata;
        doc.freeze = doc.from_canonical(&parts.freeze);
        doc.channels = parts.channels.into_iter().map(|(k, v)| (k, doc.from_canonical(&v))).collect();
        for pl in parts.layers {
            let mut layer = SculptLayer::new(pl.id, &pl.name, doc.bvh.leaves.len());
            layer.opacity = pl.opacity;
            layer.visible = pl.visible;
            layer.locked = pl.locked;
            layer.kind = pl.kind;
            layer.parent = pl.parent;
            layer.collapsed = pl.collapsed;
            layer.blend = pl.blend;
            layer.mask = pl.mask;
            for (ci, d) in pl.indices.iter().zip(pl.deltas) {
                let v = doc.canonical_index(*ci as usize);
                let l = doc.bvh.vert_leaf[v] as usize;
                let leaf = &doc.bvh.leaves[l];
                let len = leaf.owned_len();
                let chunk = layer.chunks[l].get_or_insert_with(|| vec![Vec3::ZERO; len].into_boxed_slice());
                chunk[v - leaf.owned.start as usize] = d;
            }
            doc.next_layer_id = doc.next_layer_id.max(pl.id.0 + 1);
            doc.layers.push(layer);
        }
        for l in &doc.layers {
            if let Some(p) = l.parent
                && !doc.layers.iter().any(|f| f.id == p && f.is_folder())
            {
                return Err(Error::Format(format!("layer '{}' names a missing parent folder", l.name)));
            }
        }
        for l in &doc.layers {
            if doc.is_ancestor(l.id, l.id) {
                return Err(Error::Format(format!("folder '{}' contains itself", l.name)));
            }
        }
        doc.active = parts.active.filter(|id| doc.layers.iter().any(|l| l.id == *id && !l.is_folder()));
        doc.refresh_scales_into(&mut Default::default());
        for i in 0..doc.layers.len() {
            if let Some(stack) = doc.layers[i].mask.clone() {
                doc.layers[i].mask_values = doc.mask_values_for(&stack)?;
            }
        }
        doc.recomposite_all();
        doc.dirty_channels.clear();
        Ok(doc)
    }
}

impl SculptLayer {
    fn clone_meta(&self) -> SculptLayer {
        SculptLayer {
            id: self.id,
            kind: self.kind,
            parent: self.parent,
            collapsed: self.collapsed,
            blend: self.blend,
            scale: self.scale,
            name: self.name.clone(),
            opacity: self.opacity,
            visible: self.visible,
            locked: self.locked,
            mask: self.mask.clone(),
            mask_values: None,
            chunks: Vec::new(),
        }
    }
}

//! Sculpt layers on a mesh extracted from a [`LevelVolume`] (option B: the volume holds the big form, layers are
//! per-vertex offsets on its surface).
//!
//! The mesh is the volume's own adaptive display surface ([`LevelVolume::extract_in`] over every stored block, the same
//! mesh [`LevelVolume::surface_in`] builds), so it has the density the artist sees. Each of its vertices is named by
//! (block, cell, piece): the surface piece of one lattice cell. Those names are stable: a volume edit only renames or
//! moves the vertices of the cells it reached, so after an edit the layers are carried onto the re-extracted mesh by
//! copying each vertex's data from the vertex of the same name wherever its base position did not change (bit for
//! bit) and it lies outside the changed box, and by closest point on the old base surface only for the rest.
//!
//! The mesh itself is never saved: [`MeshLayers::save`] writes the layers against the vertex names, and
//! [`MeshLayers::load`] extracts the mesh again from the volume and puts the data back by name.

mod cache;
mod carry;
mod extract;
mod file;
mod job;

pub use cache::{BlockCache, CacheUpdate};
pub use extract::{Extracted, assemble, edge_defects, extract_all, extract_named};
pub use job::{JobStats, SwapStats};

use std::time::Instant;

use glam::IVec3;

use crate::document::Document;
use crate::generator::level_volume::{Key, LevelVolume};
use crate::geom::Aabb;
use crate::layers::LayerId;
use crate::mesh::face_len;
use crate::Result;
use glam::Vec3;

/// How the mesh is taken from the volume and how layers are carried.
#[derive(Clone, Copy, Debug)]
pub struct MeshLayerSettings {
    /// The finest volume level the mesh shows (as `Show::upto`).
    pub cap: usize,
    /// Vertices within this many lattice steps of the changed box are carried by closest point even when their name
    /// and position survived.
    pub margin_steps: f32,
    /// The lattice step the carry's distances are measured in; `None` takes the finest shown level's voxel that holds
    /// blocks.
    pub step: Option<f32>,
}

impl Default for MeshLayerSettings {
    fn default() -> Self {
        MeshLayerSettings { cap: usize::MAX, margin_steps: 2.0, step: None }
    }
}

/// What one [`MeshLayers::refit`] did and how long each part took.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarryStats {
    /// Extracting the surface from the volume (with the vertex names).
    pub extract_ms: f64,
    /// `Document::from_mesh`: BVH, topology, normals.
    pub build_ms: f64,
    /// Matching names, copying, projecting, building the layers and recompositing.
    pub carry_ms: f64,
    /// Of `carry_ms`: matching names.
    pub match_ms: f64,
    /// Of `carry_ms`: the closest-point part (sub-mesh, its tree, the queries).
    pub project_ms: f64,
    /// Of `carry_ms`: recompositing the leaves the layers touch.
    pub recomposite_ms: f64,
    pub vertices: usize,
    pub triangles: usize,
    /// Vertices whose data was copied by name.
    pub copied: usize,
    /// Vertices carried by closest point.
    pub projected: usize,
    /// Old faces the closest-point search ran over.
    pub search_faces: usize,
}

pub struct MeshLayers {
    pub doc: Document,
    /// The vertex names, in the document's canonical order (the extraction order, which is ascending).
    keys: Vec<u64>,
    settings: MeshLayerSettings,
    step: f32,
    /// Refitting off the main thread: the block cache, the old base geometry, the job under way.
    live: job::Live,
}

impl MeshLayers {
    /// Extract the volume's surface and start with no layers.
    pub fn from_volume(vol: &LevelVolume, settings: MeshLayerSettings) -> Result<MeshLayers> {
        let cache = BlockCache::build(vol, settings.cap);
        let ex = cache.assemble()?;
        let doc = Document::from_mesh(ex.mesh)?;
        let live = job::Live::new(cache, job::Geom::of_fresh(&doc, &ex.keys));
        Ok(MeshLayers { doc, keys: ex.keys, settings, step: step_of(vol, &settings), live })
    }

    pub fn settings(&self) -> &MeshLayerSettings {
        &self.settings
    }

    /// The vertex names in canonical order.
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// Internal -> canonical index.
    pub fn internal_to_canonical(&self) -> Vec<u32> {
        invert(&self.doc.canonical_to_internal)
    }

    /// Layer `id`'s stored offsets, one per vertex in internal order.
    pub fn layer_deltas(&self, id: LayerId) -> Option<Vec<Vec3>> {
        let layer = self.doc.layer(id)?;
        let mut out = vec![Vec3::ZERO; self.doc.vertex_count()];
        for (l, c) in layer.chunks.iter().enumerate() {
            if let Some(c) = c {
                out[self.doc.bvh.leaves[l].owned_range()].copy_from_slice(c);
            }
        }
        Some(out)
    }

    /// The lattice step the carry uses.
    pub fn step(&self) -> f32 {
        self.step
    }

    /// Changed box of the blocks `keys` (as `LevelVolume::drain_dirty` gives them).
    pub fn changed_box(vol: &LevelVolume, keys: &[Key]) -> Aabb {
        let mut b = Aabb::EMPTY;
        for &(l, bc) in keys {
            let s = vol.voxel(l as usize) * 8.0;
            b.grow(vol.origin() + bc.as_vec3() * s);
            b.grow(vol.origin() + (bc + IVec3::ONE).as_vec3() * s);
        }
        b
    }

    /// Re-extract the surface after a volume edit confined to `changed` (world box) and carry the layers, channels and
    /// freeze onto it.
    pub fn refit(&mut self, vol: &LevelVolume, changed: Aabb) -> Result<CarryStats> {
        let mut st = CarryStats::default();
        let t = Instant::now();
        let (mesh, keys, _) = extract_named(vol, self.settings.cap)?;
        st.extract_ms = ms(t);
        let t = Instant::now();
        let new = Document::from_mesh(mesh)?;
        st.build_ms = ms(t);
        self.step = step_of(vol, &self.settings);
        let t = Instant::now();
        let (doc, s2) = carry::carry(&self.doc, &self.keys, new, &keys, changed, self.settings.margin_steps * self.step, self.step)?;
        st.carry_ms = ms(t);
        st.match_ms = s2.match_ms;
        st.project_ms = s2.project_ms;
        st.recomposite_ms = s2.recomposite_ms;
        st.copied = s2.copied;
        st.projected = s2.projected;
        st.search_faces = s2.search_faces;
        st.vertices = doc.vertex_count();
        st.triangles = doc.faces.iter().map(|f| face_len(f) - 2).sum();
        self.doc = doc;
        self.keys = keys;
        Ok(st)
    }
}

/// Refitting off the main thread (see `job`).
impl MeshLayers {
    /// After a volume edit: count the blocks `dirty` (as [`LevelVolume::drain_dirty`] gives them) and the world box
    /// `changed` as changed, and start a refit job on another thread if none runs (one that runs takes them when it
    /// ends). Costs the main thread a snapshot of the volume. Returns the milliseconds spent here.
    pub fn refit_async(&mut self, vol: &LevelVolume, dirty: &[Key], changed: Aabb) -> f64 {
        let t = Instant::now();
        self.live.note(dirty, changed);
        self.live.start(vol, &self.doc, &self.keys, self.settings);
        ms(t)
    }

    /// If the refit job finished: carry the layers as they are now onto its mesh and swap it in (then start the next
    /// job, from `vol`, if more changes came meanwhile). `None` while it runs or when none does.
    pub fn poll(&mut self, vol: &LevelVolume) -> Option<Result<SwapStats>> {
        self.live.poll(false, vol, &mut self.doc, &mut self.keys, &mut self.step, self.settings)
    }

    /// [`Self::poll`], waiting for the job under way.
    pub fn wait(&mut self, vol: &LevelVolume) -> Option<Result<SwapStats>> {
        self.live.poll(true, vol, &mut self.doc, &mut self.keys, &mut self.step, self.settings)
    }

    /// Whether a refit job is under way.
    pub fn busy(&self) -> bool {
        self.live.busy()
    }

    /// [`Self::refit_async`] and wait for it (and for any job under way first): the block-cached refit, at once.
    pub fn refit_cached(&mut self, vol: &LevelVolume, dirty: &[Key], changed: Aabb) -> Result<SwapStats> {
        let mut last = None;
        while self.busy() {
            last = self.wait(vol);
        }
        self.refit_async(vol, dirty, changed);
        while self.busy() {
            last = self.wait(vol);
        }
        last.unwrap_or_else(|| Ok(SwapStats::default()))
    }

    /// The block cache, while no job holds it.
    pub fn block_cache(&self) -> Option<&BlockCache> {
        self.live.cache.as_ref()
    }
}

pub(super) fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub(super) fn invert(perm: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p as usize] = i as u32;
    }
    out
}

/// The finest shown level's voxel among levels that hold blocks.
pub(super) fn step_of(vol: &LevelVolume, s: &MeshLayerSettings) -> f32 {
    if let Some(step) = s.step {
        return step;
    }
    let top = s.cap.min(vol.levels() - 1);
    let l = (0..=top).rev().find(|&l| vol.level_len(l) > 0).unwrap_or(0);
    vol.voxel(l)
}

//! Sculpt layers on a mesh extracted from a [`LevelVolume`] (option B: the volume holds the big form, layers are
//! per-vertex offsets on its surface).
//!
//! The mesh is the volume's own adaptive display surface ([`LevelVolume::extract_in`] over every stored block, the same
//! mesh [`LevelVolume::surface_in`] builds), so it has the density the artist sees. Each of its vertices is named by
//! (block, cell, piece): the surface piece of one lattice cell. Those names are stable: a volume edit only renames or
//! moves the vertices of the cells it reached, so after an edit the layers are carried onto the re-extracted mesh by
//! copying each vertex's data from the vertex of the same name wherever its base position did not change (bit for
//! bit), and from the old surface under it only for the rest.
//!
//! What a carry did not copy it hands back as a [`Patch`]: when the volume edit is undone and the surface is as it
//! was, [`MeshLayers::restore`] puts that data back exactly.
//!
//! The mesh itself is never saved: [`MeshLayers::save`] writes the layers against the vertex names, and
//! [`MeshLayers::load`] extracts the mesh again from the volume and puts the data back by name.

mod cache;
mod carry;
mod extract;
mod file;
mod job;
mod patch;

pub use cache::{BlockCache, CacheUpdate};
pub use extract::{Extracted, assemble, edge_defects, extract_all, extract_named};
pub use job::{BlockChanges, JobStats, SwapStats};

use std::ops::Range;
use std::sync::Arc;
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
    /// Not used: a vertex whose name and position survived is always copied. (Kept for saved files.)
    pub margin_steps: f32,
    /// What [`MeshLayers::step`] gives; `None` takes the finest shown level's voxel that holds blocks. The carry
    /// measures each vertex's distances in its own level's step.
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
    /// Vertices carried from the old surface under them.
    pub projected: usize,
    /// Of `projected`: vertices no old surface was found under (they start with no layer data).
    pub unplaced: usize,
    /// Old faces the closest-point search ran over.
    pub search_faces: usize,
}

/// What one [`MeshLayers::refit`] carried without copying: the layer offsets, freeze and channels of every old vertex
/// whose name or base position did not survive, by name. [`MeshLayers::restore`] puts them back.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    /// The vertex names, ascending.
    names: Vec<u64>,
    /// Each one's base position.
    base: Vec<Vec3>,
    /// Every layer there was, with its offsets at those vertices (`None`: all zero).
    layers: Vec<(LayerId, Option<Vec<Vec3>>)>,
    freeze: Option<Vec<f32>>,
    channels: Vec<(String, Vec<f32>)>,
}

impl Patch {
    /// How many vertices it holds.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Where each vertex sits among the volume's blocks, internal -> canonical, and the base mesh's normals: built with
/// each new mesh (in the refit job), never changed after.
#[derive(Default)]
pub(super) struct Layout {
    /// Every block, in canonical order, with its first canonical vertex.
    pub(super) blocks: Vec<(Key, u32)>,
    /// Per canonical vertex: its index in its block's GPU vertices.
    pub(super) vi: Vec<u16>,
    pub(super) i2c: Vec<u32>,
    /// Per internal vertex: the base mesh's normal (as `Document::from_mesh` computed it, before any layer).
    pub(super) base_normals: Vec<Vec3>,
}

impl Layout {
    /// For `doc`, fresh from `Document::from_mesh` of the extraction that gave `blocks` and `vi`.
    pub(super) fn new(blocks: Vec<(Key, u32)>, vi: Vec<u16>, doc: &Document) -> Layout {
        Layout { blocks, vi, i2c: invert(&doc.canonical_to_internal), base_normals: doc.normals.clone() }
    }
}

pub struct MeshLayers {
    pub doc: Document,
    /// The vertex names, in the document's canonical order (the extraction order, which is ascending).
    keys: Vec<u64>,
    layout: Arc<Layout>,
    settings: MeshLayerSettings,
    step: f32,
    /// The volume as the mesh was extracted from it: the next carry finds the old surface in it.
    pre: LevelVolume,
    /// Refitting off the main thread: the block cache, the old base geometry, the job under way, the patches of its
    /// swaps.
    live: job::Live,
}

impl MeshLayers {
    /// Extract the volume's surface and start with no layers.
    pub fn from_volume(vol: &LevelVolume, settings: MeshLayerSettings) -> Result<MeshLayers> {
        let cache = BlockCache::build(vol, settings.cap);
        let ex = cache.assemble()?;
        let doc = Document::from_mesh(ex.mesh)?;
        let layout = Arc::new(Layout::new(ex.blocks, ex.vi, &doc));
        let live = job::Live::new(cache, job::Geom::of(&doc, &ex.keys, layout.clone()));
        Ok(MeshLayers { doc, keys: ex.keys, layout, settings, step: step_of(vol, &settings), pre: vol.snapshot(), live })
    }

    pub fn settings(&self) -> &MeshLayerSettings {
        &self.settings
    }

    /// The vertex names in canonical order.
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// Internal -> canonical index (kept with the mesh).
    pub fn internal_to_canonical(&self) -> &[u32] {
        &self.layout.i2c
    }

    /// Every block of the volume the mesh was taken from (as the block cache holds them: every stored block shown up
    /// to `cap`), in canonical order, each with its first canonical vertex. Block `i`'s vertices are the canonical
    /// range [`Self::block_vertices`]`(i)`; a block whose vertices no face uses has an empty range.
    ///
    /// How a vertex maps onto the GPU's surface: canonical vertex `c` of block `key` is the vertex `vi = `
    /// [`Self::vertex_vi`]`()[c]` of that block's `BlockSurface::verts`, the one `vs_levels` draws for a corner (cell,
    /// piece) with `vi = (cells[cell] & 1023) + min(piece, count - 1)`. Its name ([`Self::keys`]) holds the block, the
    /// cell (bits 4-12) and the piece (bits 0-3), and `vi = (cells[cell] & 1023) + piece`. So per-vertex data for the
    /// GPU goes at `row_start[slot of key] + vi`, read from internal vertex `doc.canonical_index(c)`. Vertices of a block
    /// that no face uses (dropped from the mesh) have no canonical vertex: their rows stay zero.
    pub fn blocks(&self) -> &[(Key, u32)] {
        &self.layout.blocks
    }

    /// The canonical vertices of block `i` of [`Self::blocks`].
    pub fn block_vertices(&self, i: usize) -> Range<usize> {
        let b = &self.layout.blocks;
        let end = b.get(i + 1).map_or(self.keys.len(), |n| n.1 as usize);
        b[i].1 as usize..end
    }

    /// The block (index into [`Self::blocks`]) holding canonical vertex `c`.
    pub fn block_of(&self, c: usize) -> usize {
        // The last block starting at or before `c` (empty blocks share a start with the next).
        self.layout.blocks.partition_point(|b| b.1 as usize <= c) - 1
    }

    /// Per canonical vertex: its index in its block's GPU vertex order (see [`Self::blocks`]).
    pub fn vertex_vi(&self) -> &[u16] {
        &self.layout.vi
    }

    /// Per internal vertex: the base mesh's normal (area-weighted, as `Document::from_mesh` computes it before any
    /// layer), the normal Max/Min layers gate on.
    pub fn base_normals(&self) -> &[Vec3] {
        &self.layout.base_normals
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

    /// Re-extract the surface after a volume edit and carry the layers, channels and freeze onto it. (`changed`, the
    /// edit's world box, is not read: what changed is told from the vertex names and positions.)
    pub fn refit(&mut self, vol: &LevelVolume, changed: Aabb) -> Result<CarryStats> {
        Ok(self.refit_patch(vol, changed)?.0)
    }

    /// [`Self::refit`], also giving what the carry did not copy: kept beside the volume's undo step, it lets
    /// [`Self::restore`] put that data back once the edit is undone.
    pub fn refit_patch(&mut self, vol: &LevelVolume, _changed: Aabb) -> Result<(CarryStats, Patch)> {
        let mut st = CarryStats::default();
        let t = Instant::now();
        let ex = extract_all(vol, self.settings.cap)?;
        let (keys, blocks, vi) = (ex.keys, ex.blocks, ex.vi);
        st.extract_ms = ms(t);
        let t = Instant::now();
        let new = Document::from_mesh(ex.mesh)?;
        let layout = Arc::new(Layout::new(blocks, vi, &new));
        st.build_ms = ms(t);
        self.step = step_of(vol, &self.settings);
        let t = Instant::now();
        let plan = carry::plan(carry::Geo::of(&self.doc, &self.keys), carry::Geo::of(&new, &keys), &new.normals, &self.pre, vol, self.settings.cap);
        let (doc, patch, recomposite_ms) = carry::apply(&self.doc, &self.keys, new, &plan, &Default::default())?;
        st.carry_ms = ms(t);
        st.match_ms = plan.match_ms;
        st.project_ms = plan.project_ms;
        st.recomposite_ms = recomposite_ms;
        st.copied = plan.copied;
        st.projected = plan.projected;
        st.unplaced = plan.unplaced;
        st.search_faces = plan.search_faces;
        st.vertices = doc.vertex_count();
        st.triangles = doc.faces.iter().map(|f| face_len(f) - 2).sum();
        self.doc = doc;
        self.keys = keys;
        self.layout = layout;
        self.pre = vol.snapshot();
        Ok((st, patch))
    }

    /// Put back what a carry did not copy, wherever a vertex of the same name sits at the same base position again
    /// (after the volume edit was undone and the surface refitted). Returns how many vertices that was.
    pub fn restore(&mut self, patch: &Patch) -> Result<usize> {
        carry::restore(&mut self.doc, &self.keys, patch)
    }
}

/// Refitting off the main thread (see `job`).
impl MeshLayers {
    /// After a volume edit: count the blocks `dirty` (as [`LevelVolume::drain_dirty`] gives them) as changed, and
    /// start a refit job on another thread if none runs (one that runs takes them when it ends). Costs the main thread
    /// two snapshots of the volume. (`changed`, the edit's world box, is not read, as in [`Self::refit`].) Returns the
    /// milliseconds spent here.
    pub fn refit_async(&mut self, vol: &LevelVolume, dirty: &[Key], _changed: Aabb) -> f64 {
        let t = Instant::now();
        self.live.note(dirty);
        self.live.start(vol, &self.pre, &self.doc, &self.keys, &self.layout, self.settings);
        ms(t)
    }

    /// If the refit job finished: carry the layers as they are now onto its mesh and swap it in (then start the next
    /// job, from `vol`, if more changes came meanwhile). `None` while it runs or when none does. What the carry did not
    /// copy is kept as a [`Patch`]: take it with [`Self::take_patches`].
    /// Its [`SwapStats::blocks`] lists the blocks whose per-vertex data the swap changed.
    pub fn poll(&mut self, vol: &LevelVolume) -> Option<Result<SwapStats>> {
        self.live.poll(false, vol, &mut self.pre, &mut self.doc, &mut self.keys, &mut self.layout, &mut self.step, self.settings)
    }

    /// [`Self::poll`], waiting for the job under way.
    pub fn wait(&mut self, vol: &LevelVolume) -> Option<Result<SwapStats>> {
        self.live.poll(true, vol, &mut self.pre, &mut self.doc, &mut self.keys, &mut self.layout, &mut self.step, self.settings)
    }

    /// The patches of the swaps since the last call, oldest first: one per swap, which may cover several
    /// [`Self::refit_async`] calls when strokes came while a job ran. To undo, [`Self::restore`] them newest first.
    pub fn take_patches(&mut self) -> Vec<Patch> {
        std::mem::take(&mut self.live.patches)
    }

    /// Whether a refit job is under way.
    pub fn busy(&self) -> bool {
        self.live.busy()
    }

    /// With `hold`, what a swap replaces (the old mesh, its document and volume) is kept until [`Self::free_old`] or
    /// the next refit job's start (which frees it beside the job) rather than freed at once off the thread: freeing
    /// it slows work on every core for a moment (70 to 80 ms at 5M), so a caller can choose when. At most one swap's
    /// worth is held: the next swap frees the last.
    pub fn set_hold_old(&mut self, hold: bool) {
        self.live.hold_old = hold;
        if !hold {
            self.live.free_old();
        }
    }

    /// Free what the last swap replaced, if held (off the thread, below normal priority).
    pub fn free_old(&mut self) {
        self.live.free_old();
    }

    /// How long ago the swap whose replaced mesh is held came in (ms), if one is held.
    pub fn old_held_ms(&self) -> Option<f64> {
        self.live.old_held_ms()
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

    /// [`Self::refit_cached`], also giving its own swap's patch (as [`Self::refit_patch`]); patches of jobs that were
    /// under way before stay for [`Self::take_patches`].
    pub fn refit_cached_patch(&mut self, vol: &LevelVolume, dirty: &[Key], changed: Aabb) -> Result<(SwapStats, Patch)> {
        while self.busy() {
            let _ = self.wait(vol);
        }
        let before = self.live.patches.len();
        let st = self.refit_cached(vol, dirty, changed)?;
        let patch = if self.live.patches.len() > before { self.live.patches.pop().unwrap_or_default() } else { Patch::default() };
        Ok((st, patch))
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

//! Refitting off the main thread: a stroke's end only snapshots the volume and starts a job; the job brings the block
//! cache up to date, assembles the mesh, builds its document and works out the carry plan ([`carry::plan`]: geometry
//! only, from the old base kept in an `Arc` ([`Geom`]), the volume the old mesh was extracted from and the one the new
//! mesh is); the main thread then applies it ([`carry::apply`]) to the layers as they are by then (so layer edits made
//! meanwhile survive) and recomposites. The result is the plain refit's bit for bit.

use std::collections::HashSet;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::cache::{BlockCache, CacheUpdate};
use super::carry::{self, Geo};
use super::extract::key_of;
use super::{Layout, MeshLayerSettings, Patch, ms, step_of};
use crate::document::Document;
use crate::generator::level_volume::{Key, LevelVolume};
use crate::mask::{MaskContext, MaskStack, SourceValues};
use crate::mesh::{Face, face_len, face_verts};
use crate::{Error, Result};

/// The old base geometry a plan reads, shared with the job: base positions and faces (internal order), canonical ->
/// internal, the vertex names (canonical order), and which document state it is of.
pub(super) struct Geom {
    base: Vec<glam::Vec3>,
    faces: Vec<Face>,
    c2i: Vec<u32>,
    keys: Vec<u64>,
    ids: (u64, u64),
    /// Where its vertices sit in the blocks, and its base normals.
    layout: Arc<Layout>,
}

impl Geom {
    pub(super) fn of(doc: &Document, keys: &[u64], layout: Arc<Layout>) -> Geom {
        Geom { base: doc.base.clone(), faces: doc.faces.clone(), c2i: doc.canonical_to_internal.clone(), keys: keys.to_vec(), ids: ids(doc), layout }
    }

    fn geo(&self) -> Geo<'_> {
        Geo { base: &self.base, faces: &self.faces, canon: &self.c2i, keys: &self.keys }
    }

    pub(super) fn is_of(&self, doc: &Document) -> bool {
        self.ids == ids(doc)
    }
}

/// Which base geometry a document holds: its topology and base epoch.
fn ids(doc: &Document) -> (u64, u64) {
    (doc.topology_id, doc.base_epoch)
}

/// What a job did, off the main thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct JobStats {
    /// Bringing the block cache up to date (or building it, when there was none).
    pub cache: CacheUpdate,
    pub rebuilt_cache: bool,
    /// Assembling the mesh from the blocks.
    pub assemble_ms: f64,
    /// `Document::from_mesh`.
    pub build_ms: f64,
    /// Copying the new base geometry for the next plan.
    pub geom_ms: f64,
    /// Block places, internal -> canonical and base normals of the new mesh.
    pub layout_ms: f64,
    /// The base surface's normals and sampler for Max/Min layers (when one was there at the start).
    pub basis_ms: f64,
    /// Mask sources that read no channel (noise and the like), on the new mesh.
    pub masks_ms: f64,
    /// Finding the blocks whose vertices changed.
    pub changes_ms: f64,
    /// The carry plan: matching names, and the closest-point part.
    pub match_ms: f64,
    pub project_ms: f64,
    pub copied: usize,
    pub projected: usize,
    /// Of `projected`: vertices no old surface was found under.
    pub unplaced: usize,
    pub search_faces: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub total_ms: f64,
}

/// The blocks whose per-vertex data a swap may have changed: anything kept per (block, vertex `vi`) outside them is
/// still right after the swap (base position, base normal, offset of the composite from the base, composite normal).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct BlockChanges {
    /// Blocks of the new mesh: new, changed in the volume (every key given to the `refit_async` calls the swap
    /// covers), or with a vertex that is new, gone, at another `vi`, moved, or has another base normal, and every
    /// block sharing a face with such a vertex. Every block when the swap changed what all of them show (a
    /// Normal-blend layer's coverage length moved with the bounds, or a mask blurs) or the block cache was rebuilt.
    /// In (level, z, y, x) order.
    pub changed: Vec<Key>,
    /// Blocks the old mesh had and the new one has not.
    pub removed: Vec<Key>,
}

impl std::fmt::Debug for BlockChanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockChanges {{ changed: {}, removed: {} }}", self.changed.len(), self.removed.len())
    }
}

/// What one swap did on the main thread.
#[derive(Clone, Debug, Default)]
pub struct SwapStats {
    pub job: JobStats,
    /// The whole swap on the main thread (apply, recomposite, taking the result in).
    pub swap_ms: f64,
    /// Of `swap_ms`: carrying the layers, channels and freeze by the plan, and recompositing.
    pub apply_ms: f64,
    pub recomposite_ms: f64,
    /// From the job's start to the swap's end.
    pub latency_ms: f64,
    /// The document's base changed while the job ran: its result was dropped and the job started again.
    pub discarded: bool,
    /// Vertices in the [`Patch`] this swap kept (see [`super::MeshLayers::take_patches`]).
    pub patch_len: usize,
    /// The blocks re-extracted or removed by the swap (empty when nothing was swapped in).
    pub blocks: BlockChanges,
}

// ------------------------------------------------------------------------------------------------- the job

/// One refit to run off the main thread.
pub(super) struct Job {
    /// The volume now (the new mesh's).
    vol: LevelVolume,
    /// The volume the old mesh was extracted from.
    pre: LevelVolume,
    dirty: Vec<Key>,
    cache: Option<BlockCache>,
    old: Arc<Geom>,
    settings: MeshLayerSettings,
    /// The layers' mask stacks at the start: their sources that read no channel are worked out here.
    stacks: Vec<MaskStack>,
    /// A Max or Min layer was there at the start: the base surface's basis is built here.
    basis: bool,
}

/// A job's result, for [`carry::apply`] on the main thread.
pub(super) struct Done {
    doc: Document,
    keys: Vec<u64>,
    layout: Arc<Layout>,
    sources: SourceValues,
    changes: BlockChanges,
    geom: Arc<Geom>,
    plan: carry::Plan,
    step: f32,
    /// The volume the new mesh was extracted from: the next carry's `pre`.
    post: LevelVolume,
    old_ids: (u64, u64),
    stats: JobStats,
}

impl Job {
    /// Run it: the block cache comes back whatever happens to the rest.
    fn run(self) -> (Option<BlockCache>, Result<Done>) {
        let t0 = Instant::now();
        let Job { vol, pre, dirty, cache, old, settings, stacks, basis } = self;
        let mut st = JobStats::default();
        let cache = match cache {
            Some(mut c) if c.cap() == settings.cap => {
                st.cache = c.update(&vol, &dirty);
                c
            }
            _ => {
                let t = Instant::now();
                st.rebuilt_cache = true;
                let c = BlockCache::build(&vol, settings.cap);
                st.cache.extract_ms = ms(t);
                c
            }
        };
        let done = (|| {
            let t = Instant::now();
            let ex = cache.assemble()?;
            st.assemble_ms = ms(t);
            let t = Instant::now();
            let mut doc = Document::from_mesh(ex.mesh)?;
            st.build_ms = ms(t);
            let t = Instant::now();
            let layout = Arc::new(Layout::new(ex.blocks, ex.vi, &doc));
            st.layout_ms = ms(t);
            if basis {
                let t = Instant::now();
                doc.generator_basis();
                st.basis_ms = ms(t);
            }
            let t = Instant::now();
            let no_channels = Default::default();
            let ctx = MaskContext { positions: &doc.rest, normals: &layout.base_normals, topology: &doc.topo, channels: &no_channels };
            let sources = SourceValues::build(&stacks, &ctx)?;
            st.masks_ms = ms(t);
            let t = Instant::now();
            let changes = block_changes(&old, Geo::of(&doc, &ex.keys), &layout, st.rebuilt_cache, &dirty);
            st.changes_ms = ms(t);
            let step = step_of(&vol, &settings);
            let plan = carry::plan(old.geo(), Geo::of(&doc, &ex.keys), &doc.normals, &pre, &vol, settings.cap);
            st.match_ms = plan.match_ms;
            st.project_ms = plan.project_ms;
            st.copied = plan.copied;
            st.projected = plan.projected;
            st.unplaced = plan.unplaced;
            st.search_faces = plan.search_faces;
            let t = Instant::now();
            let geom = Arc::new(Geom::of(&doc, &ex.keys, layout.clone()));
            st.geom_ms = ms(t);
            st.vertices = doc.vertex_count();
            st.triangles = doc.faces.iter().map(|f| face_len(f) - 2).sum();
            st.total_ms = ms(t0);
            Ok(Done { doc, keys: ex.keys, layout, sources, changes, geom, plan, step, post: vol, old_ids: old.ids, stats: st })
        })();
        // The old geometry and volume may be the last references: let them go here rather than on the main thread.
        drop(old);
        drop(pre);
        (Some(cache), done)
    }
}

/// The refit state kept beside the document: the block cache and the old geometry while idle, the job while one
/// runs, the changes since it started, and the patches of the swaps not yet taken.
pub(super) struct Live {
    pub(super) cache: Option<BlockCache>,
    geom: Option<Arc<Geom>>,
    running: Option<Running>,
    pending: Vec<Key>,
    has_pending: bool,
    pub(super) patches: Vec<Patch>,
}

struct Running {
    handle: JoinHandle<(Option<BlockCache>, Result<Done>)>,
    dirty: Vec<Key>,
    started: Instant,
}

impl Live {
    pub(super) fn new(cache: BlockCache, geom: Geom) -> Live {
        Live { cache: Some(cache), geom: Some(Arc::new(geom)), running: None, pending: Vec::new(), has_pending: false, patches: Vec::new() }
    }

    pub(super) fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Count `dirty` as waiting for the next job.
    pub(super) fn note(&mut self, dirty: &[Key]) {
        self.pending.extend_from_slice(dirty);
        self.has_pending = true;
    }

    /// Start a job for what is waiting, if any and none runs. `pre` is the volume `doc`'s mesh was extracted from.
    pub(super) fn start(&mut self, vol: &LevelVolume, pre: &LevelVolume, doc: &Document, keys: &[u64], layout: &Arc<Layout>, settings: MeshLayerSettings) {
        if self.running.is_some() || !self.has_pending {
            return;
        }
        // A document whose base changed by other means (a plain refit, a bake): take its geometry again, and extract
        // everything (which blocks changed since the cache last saw the volume is not known).
        if !self.geom.as_ref().is_some_and(|g| g.is_of(doc)) {
            self.geom = Some(Arc::new(Geom::of(doc, keys, layout.clone())));
            self.cache = None;
        }
        let mut dirty = std::mem::take(&mut self.pending);
        dirty.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        dirty.dedup();
        self.has_pending = false;
        let stacks = doc.layers.iter().filter_map(|l| l.mask.clone()).filter(|s| s.enabled).collect();
        let basis = doc.layers.iter().any(|l| l.blend.uses_normal());
        let job = Job { vol: vol.snapshot(), pre: pre.snapshot(), dirty: dirty.clone(), cache: self.cache.take(), old: self.geom.clone().unwrap(), settings, stacks, basis };
        let handle = std::thread::Builder::new().name("mesh-layers-refit".into()).spawn(move || job.run()).expect("spawn the refit job");
        self.running = Some(Running { handle, dirty, started: Instant::now() });
    }

    /// Take in the finished job, if one finished (with `wait`, waiting for it): apply its plan to `doc`'s layers as
    /// they are now, swap the new document in (`pre` becoming the volume it was extracted from) and keep the carry's
    /// patch; then start the next job if changes are waiting.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn poll(
        &mut self,
        wait: bool,
        vol: &LevelVolume,
        pre: &mut LevelVolume,
        doc: &mut Document,
        keys: &mut Vec<u64>,
        layout: &mut Arc<Layout>,
        step: &mut f32,
        settings: MeshLayerSettings,
    ) -> Option<Result<SwapStats>> {
        if !wait && !self.running.as_ref()?.handle.is_finished() {
            return None;
        }
        let r = self.running.take()?;
        let t = Instant::now();
        let (cache, done) = match r.handle.join() {
            Ok(x) => x,
            Err(_) => (None, Err(Error::InvalidData("the refit job panicked".into()))),
        };
        self.cache = cache;
        let out = match done {
            // Nothing was swapped in: its change goes with the next job.
            Err(e) => {
                self.note(&r.dirty);
                Err(e)
            }
            // Made for another base than the document holds now (changed by other means meanwhile): again.
            Ok(done) if done.old_ids != ids(doc) => {
                self.note(&r.dirty);
                self.geom = None;
                Ok(SwapStats { job: done.stats, discarded: true, swap_ms: ms(t), ..Default::default() })
            }
            Ok(done) => {
                let Done { doc: new, keys: new_keys, layout: new_layout, sources, mut changes, geom, plan, step: new_step, post, stats, .. } = done;
                let ta = Instant::now();
                match carry::apply(doc, keys, new, &plan, &sources) {
                    Ok((new, patch, recomposite_ms)) => {
                        let apply_ms = ms(ta);
                        if rows_everywhere(doc, &new) {
                            changes.changed = new_layout.blocks.iter().map(|b| b.0).collect();
                        }
                        let old = std::mem::replace(doc, new);
                        let old_keys = std::mem::replace(keys, new_keys);
                        let old_layout = std::mem::replace(layout, new_layout);
                        let old_pre = std::mem::replace(pre, post);
                        *step = new_step;
                        let old_geom = self.geom.replace(geom);
                        let patch_len = patch.len();
                        self.patches.push(patch);
                        // Freeing the old mesh takes a while: not on this thread.
                        std::thread::spawn(move || drop((old, old_keys, old_layout, old_geom, old_pre, plan, sources)));
                        let swap_ms = ms(t);
                        Ok(SwapStats { job: stats, swap_ms, apply_ms, recomposite_ms, latency_ms: r.started.elapsed().as_secs_f64() * 1e3, discarded: false, patch_len, blocks: changes })
                    }
                    Err(e) => {
                        self.note(&r.dirty);
                        Err(e)
                    }
                }
            }
        };
        self.start(vol, pre, doc, keys, layout, settings);
        Some(out)
    }
}

/// Whether a swap from `old` to `new` changed what every vertex shows: a Normal-blend layer whose coverage length
/// (from the bounds) moved, or a mask that blurs (its values spread past the changed blocks).
fn rows_everywhere(old: &Document, new: &Document) -> bool {
    fn blurs(s: &MaskStack) -> bool {
        s.layers.iter().any(|l| l.blur > 0 || l.mask.as_ref().is_some_and(|m| blurs(m)))
    }
    let eps = |d: &Document| (d.bvh.bounds().extent().length() * 1e-4).max(1e-9).to_bits();
    let gated = new.layers.iter().any(|l| l.blend == crate::layers::LayerBlend::Normal);
    (gated && eps(old) != eps(new)) || new.layers.iter().any(|l| l.mask.as_ref().is_some_and(blurs))
}

fn bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

/// The blocks of the new mesh whose vertices' data may differ from the old mesh's (see [`BlockChanges`]), and every
/// block of `dirty` (the volume's changed blocks the job covers: the view may have dropped their rows). All of them
/// when the cache was `rebuilt`.
fn block_changes(old: &Geom, new: Geo, layout: &Layout, rebuilt: bool, dirty: &[Key]) -> BlockChanges {
    let (ol, nl) = (&old.layout, layout);
    // Per new vertex (canonical): new name, or another place in its block, base position or base normal than the old
    // vertex of its name.
    let moved_c: Vec<bool> = new
        .keys
        .par_iter()
        .enumerate()
        .map(|(c, k)| match old.keys.binary_search(k) {
            Ok(oc) => {
                let (ov, nv) = (old.c2i[oc] as usize, new.canon[c] as usize);
                ol.vi[oc] != nl.vi[c] || bits(old.base[ov]) != bits(new.base[nv]) || bits(ol.base_normals[ov]) != bits(nl.base_normals[nv])
            }
            Err(_) => true,
        })
        .collect();
    let mut set: HashSet<Key> = new.keys.par_iter().zip(moved_c.par_iter()).filter(|(_, m)| **m).map(|(k, _)| key_of(*k)).collect();
    // Old vertices gone.
    set.extend(old.keys.par_iter().filter(|k| new.keys.binary_search(k).is_err()).map(|k| key_of(*k)).collect::<HashSet<Key>>());
    // Every block sharing a face with such a vertex (its normals may have turned).
    let moved = crate::mesh::permute(new.canon, &moved_c);
    let ring: HashSet<Key> = new
        .faces
        .par_iter()
        .filter(|f| face_verts(f).iter().any(|&v| moved[v as usize]))
        .flat_map_iter(|f| face_verts(f).iter().map(|&v| key_of(new.keys[nl.i2c[v as usize] as usize])).collect::<Vec<_>>())
        .collect();
    set.extend(ring);
    let old_blocks: HashSet<Key> = ol.blocks.iter().map(|b| b.0).collect();
    let new_blocks: HashSet<Key> = nl.blocks.iter().map(|b| b.0).collect();
    if rebuilt {
        set.extend(new_blocks.iter().copied());
    }
    set.extend(dirty.iter().copied());
    set.extend(new_blocks.iter().filter(|k| !old_blocks.contains(k)).copied());
    let order = |k: &Key| (k.0, k.1.z, k.1.y, k.1.x);
    let mut changed: Vec<Key> = set.into_iter().filter(|k| new_blocks.contains(k)).collect();
    changed.sort_unstable_by_key(order);
    let mut removed: Vec<Key> = old_blocks.iter().filter(|k| !new_blocks.contains(k)).copied().collect();
    removed.sort_unstable_by_key(order);
    BlockChanges { changed, removed }
}

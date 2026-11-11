//! Refitting off the main thread: a stroke's end only snapshots the volume and starts a job; the job brings the block
//! cache up to date, assembles the mesh, builds its document and works out the carry plan ([`carry::plan`]: geometry
//! only, from the old base kept in an `Arc` ([`Geom`]), the volume the old mesh was extracted from and the one the new
//! mesh is); the main thread then applies it ([`carry::apply`]) to the layers as they are by then (so layer edits made
//! meanwhile survive) and recomposites. The result is the plain refit's bit for bit.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use super::cache::{BlockCache, CacheUpdate};
use super::carry::{self, Geo};
use super::{MeshLayerSettings, Patch, ms, step_of};
use crate::document::Document;
use crate::generator::level_volume::{Key, LevelVolume};
use crate::mesh::{Face, face_len};
use crate::{Error, Result};

/// The old base geometry a plan reads, shared with the job: base positions and faces (internal order), canonical ->
/// internal, the vertex names (canonical order), and which document state it is of.
pub(super) struct Geom {
    base: Vec<glam::Vec3>,
    faces: Vec<Face>,
    c2i: Vec<u32>,
    keys: Vec<u64>,
    ids: (u64, u64),
}

impl Geom {
    pub(super) fn of(doc: &Document, keys: &[u64]) -> Geom {
        Geom { base: doc.base.clone(), faces: doc.faces.clone(), c2i: doc.canonical_to_internal.clone(), keys: keys.to_vec(), ids: ids(doc) }
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

/// What one swap did on the main thread.
#[derive(Clone, Copy, Debug, Default)]
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
}

/// A job's result, for [`carry::apply`] on the main thread.
pub(super) struct Done {
    doc: Document,
    keys: Vec<u64>,
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
        let Job { vol, pre, dirty, cache, old, settings } = self;
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
            let doc = Document::from_mesh(ex.mesh)?;
            st.build_ms = ms(t);
            let step = step_of(&vol, &settings);
            let plan = carry::plan(old.geo(), Geo::of(&doc, &ex.keys), &doc.normals, &pre, &vol, settings.cap);
            st.match_ms = plan.match_ms;
            st.project_ms = plan.project_ms;
            st.copied = plan.copied;
            st.projected = plan.projected;
            st.unplaced = plan.unplaced;
            st.search_faces = plan.search_faces;
            let t = Instant::now();
            let geom = Arc::new(Geom::of(&doc, &ex.keys));
            st.geom_ms = ms(t);
            st.vertices = doc.vertex_count();
            st.triangles = doc.faces.iter().map(|f| face_len(f) - 2).sum();
            st.total_ms = ms(t0);
            Ok(Done { doc, keys: ex.keys, geom, plan, step, post: vol, old_ids: old.ids, stats: st })
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
    pub(super) fn start(&mut self, vol: &LevelVolume, pre: &LevelVolume, doc: &Document, keys: &[u64], settings: MeshLayerSettings) {
        if self.running.is_some() || !self.has_pending {
            return;
        }
        // A document whose base changed by other means (a plain refit, a bake): take its geometry again, and extract
        // everything (which blocks changed since the cache last saw the volume is not known).
        if !self.geom.as_ref().is_some_and(|g| g.is_of(doc)) {
            self.geom = Some(Arc::new(Geom::of(doc, keys)));
            self.cache = None;
        }
        let mut dirty = std::mem::take(&mut self.pending);
        dirty.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        dirty.dedup();
        self.has_pending = false;
        let job = Job { vol: vol.snapshot(), pre: pre.snapshot(), dirty: dirty.clone(), cache: self.cache.take(), old: self.geom.clone().unwrap(), settings };
        let handle = std::thread::Builder::new().name("mesh-layers-refit".into()).spawn(move || job.run()).expect("spawn the refit job");
        self.running = Some(Running { handle, dirty, started: Instant::now() });
    }

    /// Take in the finished job, if one finished (with `wait`, waiting for it): apply its plan to `doc`'s layers as
    /// they are now, swap the new document in (`pre` becoming the volume it was extracted from) and keep the carry's
    /// patch; then start the next job if changes are waiting.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn poll(&mut self, wait: bool, vol: &LevelVolume, pre: &mut LevelVolume, doc: &mut Document, keys: &mut Vec<u64>, step: &mut f32, settings: MeshLayerSettings) -> Option<Result<SwapStats>> {
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
                let Done { doc: new, keys: new_keys, geom, plan, step: new_step, post, stats, .. } = done;
                let ta = Instant::now();
                match carry::apply(doc, keys, new, &plan) {
                    Ok((new, patch, recomposite_ms)) => {
                        let apply_ms = ms(ta);
                        let old = std::mem::replace(doc, new);
                        let old_keys = std::mem::replace(keys, new_keys);
                        let old_pre = std::mem::replace(pre, post);
                        *step = new_step;
                        let old_geom = self.geom.replace(geom);
                        let patch_len = patch.len();
                        self.patches.push(patch);
                        // Freeing the old mesh takes a while: not on this thread.
                        std::thread::spawn(move || drop((old, old_keys, old_geom, old_pre, plan)));
                        let swap_ms = ms(t);
                        Ok(SwapStats { job: stats, swap_ms, apply_ms, recomposite_ms, latency_ms: r.started.elapsed().as_secs_f64() * 1e3, discarded: false, patch_len })
                    }
                    Err(e) => {
                        self.note(&r.dirty);
                        Err(e)
                    }
                }
            }
        };
        self.start(vol, pre, doc, keys, settings);
        Some(out)
    }
}

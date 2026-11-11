//! Refitting off the main thread: a stroke's end only snapshots the volume and starts a job; the job brings the block
//! cache up to date, assembles the mesh, builds its document and works out the carry PLAN (geometry only: for each new
//! vertex, the old vertex of its name or the closest point on the old base); the main thread then APPLIES the plan to
//! the layers as they are by then (so layer edits made meanwhile survive) and recomposites.
//!
//! The plan and apply halves below follow `carry::carry` (as of the split into extract/carry/file): the plan is its
//! name match and closest-point part, reading only the old base geometry, kept in an `Arc` ([`Geom`]); the apply is
//! its layer, channel and freeze part.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::cache::{BlockCache, CacheUpdate};
use super::{MeshLayerSettings, ms, step_of};
use crate::bvh::Bvh;
use crate::document::Document;
use crate::generator::level_volume::{Key, LevelVolume};
use crate::generator::voxel::MeshSdf;
use crate::geom::Aabb;
use crate::layers::SculptLayer;
use crate::mesh::{Face, NO_VERT, Topology, face_len, face_verts};
use crate::{Error, Result};

/// The old base geometry a plan reads, shared with the job: base positions, faces and base normals (internal order),
/// canonical -> internal, the vertex names (canonical order), and which document state it is of.
pub(super) struct Geom {
    base: Vec<Vec3>,
    faces: Vec<Face>,
    normals: Vec<Vec3>,
    c2i: Vec<u32>,
    keys: Vec<u64>,
    ids: (u64, u64),
}

impl Geom {
    /// Of a fresh document (no layers yet: its normals are its base's).
    pub(super) fn of_fresh(doc: &Document, keys: &[u64]) -> Geom {
        Geom { base: doc.base.clone(), faces: doc.faces.clone(), normals: doc.normals.clone(), c2i: doc.canonical_to_internal.clone(), keys: keys.to_vec(), ids: ids(doc) }
    }

    /// Of any document (the base normals worked out again).
    pub(super) fn of(doc: &Document, keys: &[u64]) -> Geom {
        let normals = crate::mesh::compute_normals(&doc.base, &doc.faces, &doc.topo);
        Geom { normals, ..Geom::of_fresh(doc, keys) }
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
}

/// Closest point on the old base: old triangle (internal), weights, how much to keep, how much of the full offset.
#[derive(Clone, Copy)]
struct Proj {
    tri: [u32; 3],
    w: [f32; 3],
    keep: f32,
    whole: f32,
}

/// A carry source: an old vertex, `PROJ | slot` into the projections, or nothing.
const NONE: u32 = u32::MAX;
const PROJ: u32 = 1 << 31;

/// Per new internal vertex where its data comes from, and the closest points.
pub(super) struct Plan {
    src: Vec<u32>,
    proj: Vec<Proj>,
    /// Old base normals at the projected triangles' corners.
    normals: Vec<[Vec3; 3]>,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ------------------------------------------------------------------------------------------------- plan (geometry)

/// The carry plan from `old` onto `new` (fresh from `Document::from_mesh`, names `new_keys` in canonical order): reads
/// only geometry.
pub(super) fn plan(old: &Geom, new: &Document, new_keys: &[u64], changed: Aabb, margin: f32, step: f32, st: &mut JobStats) -> Plan {
    let grown = if changed.is_empty() { changed } else { Aabb { min: changed.min - Vec3::splat(margin), max: changed.max + Vec3::splat(margin) } };
    let inside = |p: Vec3| !grown.is_empty() && p.cmpge(grown.min).all() && p.cmple(grown.max).all();
    let new_n = new.vertex_count();
    let t = Instant::now();
    let by_canon: Vec<u32> = new_keys
        .par_iter()
        .enumerate()
        .map(|(c, k)| match old.keys.binary_search(k) {
            Ok(oc) => {
                let ov = old.c2i[oc];
                let p = new.base[new.canonical_to_internal[c] as usize];
                if old.base[ov as usize] == p && !inside(p) { ov } else { NONE }
            }
            Err(_) => NONE,
        })
        .collect();
    let mut src: Vec<u32> = crate::mesh::permute(&new.canonical_to_internal, &by_canon);
    drop(by_canon);
    st.match_ms = ms(t);
    let t = Instant::now();
    let todo: Vec<u32> = src.par_iter().enumerate().filter(|(_, s)| **s == NONE).map(|(v, _)| v as u32).collect();
    st.copied = new_n - todo.len();
    st.projected = todo.len();
    let mut proj: Vec<Proj> = Vec::new();
    if !todo.is_empty() {
        let mut region = Aabb::EMPTY;
        for &v in &todo {
            region.grow(new.base[v as usize]);
        }
        let reach = 3.0 * step;
        let region = Aabb { min: region.min - Vec3::splat(reach), max: region.max + Vec3::splat(reach) };
        let near = |p: Vec3| p.cmpge(region.min).all() && p.cmple(region.max).all();
        let faces: Vec<Face> = old.faces.par_iter().filter(|f| face_verts(f).iter().any(|&v| near(old.base[v as usize]))).copied().collect();
        st.search_faces = faces.len();
        if !faces.is_empty() {
            let mut local = std::collections::HashMap::<u32, u32>::new();
            let mut back: Vec<u32> = Vec::new();
            let mut sub_faces = faces;
            for f in &mut sub_faces {
                for v in f.iter_mut().filter(|v| **v != NO_VERT) {
                    *v = *local.entry(*v).or_insert_with(|| {
                        back.push(*v);
                        back.len() as u32 - 1
                    });
                }
            }
            let mut pos: Vec<Vec3> = back.iter().map(|&v| old.base[v as usize]).collect();
            let (bvh, perm) = Bvh::build(&mut pos, &mut sub_faces);
            let mut back2 = vec![0u32; back.len()];
            for (i, &p) in perm.iter().enumerate() {
                back2[p as usize] = back[i];
            }
            let topo = Topology::build(pos.len(), &sub_faces);
            let sdf = MeshSdf::build_owned(bvh, pos, sub_faces, &topo);
            let pseudo = sdf.pseudo_normals();
            let bound = reach;
            proj = todo
                .par_iter()
                .map(|&v| {
                    let p = new.base[v as usize];
                    let n = new.normals[v as usize];
                    match sdf.closest(p, bound) {
                        Some((cp, tri, w)) => {
                            let old_n = (pseudo[tri[0] as usize] * w[0] + pseudo[tri[1] as usize] * w[1] + pseudo[tri[2] as usize] * w[2]).normalize_or_zero();
                            let near = 1.0 - smoothstep(1.0 * step, 2.5 * step, p.distance(cp));
                            let agree = n.normalize_or_zero().dot(old_n);
                            let facing = smoothstep(0.2, 0.6, agree);
                            let whole = smoothstep(0.9, 0.98, agree);
                            Proj { tri: tri.map(|t| back2[t as usize]), w, keep: near * facing, whole }
                        }
                        None => Proj { tri: [0; 3], w: [1.0, 0.0, 0.0], keep: 0.0, whole: 0.0 },
                    }
                })
                .collect();
            for (slot, &v) in todo.iter().enumerate() {
                src[v as usize] = PROJ | slot as u32;
            }
        }
    }
    // Old base normals at the projected triangles' corners (as `vertex_normal` gives them: the old document's own).
    let normals: Vec<[Vec3; 3]> = proj.par_iter().map(|p| p.tri.map(|v| old.normals[v as usize].normalize_or_zero())).collect();
    st.project_ms = ms(t);
    Plan { src, proj, normals }
}

// ------------------------------------------------------------------------------------------------- apply (main thread)

/// `new` with `old`'s layers, channels and freeze (as they are now) carried onto it by `plan`. Returns it and the
/// time spent recompositing.
pub(super) fn apply(old: &Document, mut new: Document, plan: &Plan) -> Result<(Document, f64)> {
    let new_n = new.vertex_count();
    let Plan { src, proj, normals } = plan;
    let old_leaf = &old.bvh.vert_leaf;
    let old_get = |layer: &SculptLayer, v: u32| -> Vec3 {
        let l = old_leaf[v as usize] as usize;
        match &layer.chunks[l] {
            Some(c) => c[v as usize - old.bvh.leaves[l].owned.start as usize],
            None => Vec3::ZERO,
        }
    };
    let new_normals = &new.normals;
    let carry_vec = |layer: &SculptLayer, v: usize| -> Vec3 {
        let s = src[v];
        if s == NONE {
            Vec3::ZERO
        } else if s & PROJ == 0 {
            old_get(layer, s)
        } else {
            let slot = (s & !PROJ) as usize;
            let Proj { tri, w, keep, whole } = proj[slot];
            if keep <= 0.0 {
                return Vec3::ZERO;
            }
            let d = tri.map(|o| old_get(layer, o));
            let full = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
            let height: f32 = (0..3).map(|k| d[k].dot(normals[slot][k]) * w[k]).sum();
            let along = new_normals[v].normalize_or_zero() * height;
            (along + (full - along) * whole) * keep
        }
    };
    let carry_f = |vals: &[f32]| -> Vec<f32> {
        src.par_iter()
            .map(|&s| {
                if s == NONE {
                    0.0
                } else if s & PROJ == 0 {
                    vals[s as usize]
                } else {
                    let Proj { tri, w, .. } = proj[(s & !PROJ) as usize];
                    vals[tri[0] as usize] * w[0] + vals[tri[1] as usize] * w[1] + vals[tri[2] as usize] * w[2]
                }
            })
            .collect()
    };
    new.freeze = if old.freeze.par_iter().any(|&f| f != 0.0) { carry_f(&old.freeze).into_iter().map(|f| f.clamp(0.0, 1.0)).collect() } else { vec![0.0; new_n] };
    new.channels = old.channels.iter().map(|(k, v)| (k.clone(), carry_f(v))).collect::<BTreeMap<_, _>>();
    let leaves = &new.bvh.leaves;
    let mut touched = vec![false; leaves.len()];
    let mut layers = Vec::with_capacity(old.layers.len());
    for layer in &old.layers {
        let chunks: Vec<Option<Box<[Vec3]>>> = if layer.chunks.iter().any(|c| c.is_some()) {
            leaves
                .par_iter()
                .map(|leaf| {
                    let r = leaf.owned_range();
                    let vals: Vec<Vec3> = r.map(|v| carry_vec(layer, v)).collect();
                    vals.iter().any(|d| *d != Vec3::ZERO).then(|| vals.into_boxed_slice())
                })
                .collect()
        } else {
            vec![None; leaves.len()]
        };
        for (i, c) in chunks.iter().enumerate() {
            touched[i] |= c.is_some();
        }
        layers.push(SculptLayer {
            id: layer.id,
            kind: layer.kind,
            parent: layer.parent,
            collapsed: layer.collapsed,
            blend: layer.blend,
            scale: layer.scale,
            name: layer.name.clone(),
            opacity: layer.opacity,
            visible: layer.visible,
            locked: layer.locked,
            mask: layer.mask.clone(),
            mask_values: None,
            chunks,
            generator: layer.generator.clone(),
            gen_cache: None,
        });
    }
    new.layers = layers;
    new.active = old.active;
    new.solo = old.solo;
    new.next_layer_id = old.next_layer_id;
    new.metadata = old.metadata.clone();
    new.level = old.level;
    for i in 0..new.layers.len() {
        if let Some(stack) = new.layers[i].mask.clone() {
            new.layers[i].mask_values = new.mask_values_for(&stack)?;
        }
    }
    new.refresh_scales();
    let t = Instant::now();
    let touched: Vec<u32> = touched.iter().enumerate().filter(|(_, t)| **t).map(|(i, _)| i as u32).collect();
    new.recomposite(&touched);
    let recomposite_ms = ms(t);
    new.dirty_channels.clear();
    new.undo.clear();
    Ok((new, recomposite_ms))
}

// ------------------------------------------------------------------------------------------------- the job

/// One refit to run off the main thread.
pub(super) struct Job {
    vol: LevelVolume,
    dirty: Vec<Key>,
    changed: Aabb,
    cache: Option<BlockCache>,
    old: Arc<Geom>,
    settings: MeshLayerSettings,
}

/// A job's result, for [`apply`] on the main thread.
pub(super) struct Done {
    pub(super) doc: Document,
    pub(super) keys: Vec<u64>,
    pub(super) geom: Arc<Geom>,
    pub(super) plan: Plan,
    pub(super) step: f32,
    pub(super) old_ids: (u64, u64),
    pub(super) stats: JobStats,
}

impl Job {
    /// Run it: the block cache comes back whatever happens to the rest.
    pub(super) fn run(self) -> (Option<BlockCache>, Result<Done>) {
        let t0 = Instant::now();
        let Job { vol, dirty, changed, cache, old, settings } = self;
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
            let plan = plan(&old, &doc, &ex.keys, changed, settings.margin_steps * step, step, &mut st);
            let t = Instant::now();
            let geom = Arc::new(Geom::of_fresh(&doc, &ex.keys));
            st.geom_ms = ms(t);
            st.vertices = doc.vertex_count();
            st.triangles = doc.faces.iter().map(|f| face_len(f) - 2).sum();
            st.total_ms = ms(t0);
            Ok(Done { doc, keys: ex.keys, geom, plan, step, old_ids: old.ids, stats: st })
        })();
        // The old geometry may be the last reference: let it go here rather than on the main thread.
        drop(old);
        drop(vol);
        (Some(cache), done)
    }
}

/// The refit state kept beside the document: the block cache and the old geometry while idle, the job while one
/// runs, and the changes since it started.
pub(super) struct Live {
    pub(super) cache: Option<BlockCache>,
    pub(super) geom: Option<Arc<Geom>>,
    running: Option<Running>,
    pending: Vec<Key>,
    pending_box: Aabb,
    has_pending: bool,
}

struct Running {
    handle: JoinHandle<(Option<BlockCache>, Result<Done>)>,
    dirty: Vec<Key>,
    changed: Aabb,
    started: Instant,
}

fn union(a: Aabb, b: Aabb) -> Aabb {
    if a.is_empty() {
        b
    } else if b.is_empty() {
        a
    } else {
        a.union(&b)
    }
}

impl Live {
    pub(super) fn new(cache: BlockCache, geom: Geom) -> Live {
        Live { cache: Some(cache), geom: Some(Arc::new(geom)), running: None, pending: Vec::new(), pending_box: Aabb::EMPTY, has_pending: false }
    }

    pub(super) fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Count `dirty` (and the world box `changed`) as waiting for the next job.
    pub(super) fn note(&mut self, dirty: &[Key], changed: Aabb) {
        if !self.has_pending {
            self.pending_box = Aabb::EMPTY;
        }
        self.pending.extend_from_slice(dirty);
        self.pending_box = union(self.pending_box, changed);
        self.has_pending = true;
    }

    /// Start a job for what is waiting, if any and none runs.
    pub(super) fn start(&mut self, vol: &LevelVolume, doc: &Document, keys: &[u64], settings: MeshLayerSettings) {
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
        let changed = std::mem::replace(&mut self.pending_box, Aabb::EMPTY);
        self.has_pending = false;
        let job = Job { vol: vol.snapshot(), dirty: dirty.clone(), changed, cache: self.cache.take(), old: self.geom.clone().unwrap(), settings };
        let handle = std::thread::Builder::new().name("mesh-layers-refit".into()).spawn(move || job.run()).expect("spawn the refit job");
        self.running = Some(Running { handle, dirty, changed, started: Instant::now() });
    }

    /// Take in the finished job, if one finished (with `wait`, waiting for it): apply its plan to `doc`'s layers as
    /// they are now and swap the new document in; then start the next job if changes are waiting.
    pub(super) fn poll(&mut self, wait: bool, vol: &LevelVolume, doc: &mut Document, keys: &mut Vec<u64>, step: &mut f32, settings: MeshLayerSettings) -> Option<Result<SwapStats>> {
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
                self.note(&r.dirty, r.changed);
                Err(e)
            }
            // Made for another base than the document holds now (changed by other means meanwhile): again.
            Ok(done) if done.old_ids != ids(doc) => {
                self.note(&r.dirty, r.changed);
                self.geom = None;
                Ok(SwapStats { job: done.stats, discarded: true, swap_ms: ms(t), ..Default::default() })
            }
            Ok(done) => {
                let Done { doc: new, keys: new_keys, geom, plan, step: new_step, stats, .. } = done;
                let ta = Instant::now();
                match apply(doc, new, &plan) {
                    Ok((new, recomposite_ms)) => {
                        let apply_ms = ms(ta);
                        let old = std::mem::replace(doc, new);
                        let old_keys = std::mem::replace(keys, new_keys);
                        *step = new_step;
                        let old_geom = self.geom.replace(geom);
                        // Freeing the old mesh takes a while: not on this thread.
                        std::thread::spawn(move || drop((old, old_keys, old_geom, plan)));
                        let swap_ms = ms(t);
                        Ok(SwapStats { job: stats, swap_ms, apply_ms, recomposite_ms, latency_ms: r.started.elapsed().as_secs_f64() * 1e3, discarded: false })
                    }
                    Err(e) => {
                        self.note(&r.dirty, r.changed);
                        Err(e)
                    }
                }
            }
        };
        self.start(vol, doc, keys, settings);
        Some(out)
    }
}

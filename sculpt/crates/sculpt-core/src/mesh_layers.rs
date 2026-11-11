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

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bvh::Bvh;
use crate::document::Document;
use crate::generator::level_volume::{ExtractStats, Key, LevelVolume, Show};
use crate::generator::voxel::MeshSdf;
use crate::geom::Aabb;
use crate::io::project::LayerEntry;
use crate::layers::{LayerId, SculptLayer};
use crate::mesh::{Face, NO_VERT, PolyMesh, Topology, face_len, face_verts};
use crate::{Error, Result};

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
}

/// The volume's display surface as one mesh, with each vertex's name. Vertices no face uses are left out.
pub fn extract_named(vol: &LevelVolume, cap: usize) -> Result<(PolyMesh, Vec<u64>, ExtractStats)> {
    let v = Show::upto(cap);
    let work: Vec<Key> = vol.keys();
    let (surf, stats) = vol.extract_in(&work, &v);
    let mut blocks: Vec<&Key> = surf.keys().collect();
    blocks.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
    let mut first: std::collections::HashMap<Key, u32> = std::collections::HashMap::with_capacity(blocks.len());
    let mut total = 0usize;
    for k in &blocks {
        first.insert(**k, total as u32);
        total += surf[*k].verts.len();
    }
    // Positions and names, block by block, in parallel.
    let per_block: Vec<(Vec<Vec3>, Vec<u64>)> = blocks
        .par_iter()
        .map(|k| {
            let s = &surf[*k];
            let mut names = vec![u64::MAX; s.verts.len()];
            for (cell, &info) in s.cells.iter().enumerate() {
                let count = ((info >> 10) & 7) as usize;
                let start = (info & 1023) as usize;
                for p in 0..count {
                    if let Some(slot) = names.get_mut(start + p) {
                        *slot = name(**k, cell, p);
                    }
                }
            }
            (s.verts.clone(), names)
        })
        .collect();
    let mut positions = Vec::with_capacity(total);
    let mut names = Vec::with_capacity(total);
    for (p, n) in per_block {
        positions.extend(p);
        names.extend(n);
    }
    if names.iter().any(|&n| n == u64::MAX) {
        return Err(Error::InvalidData("a surface vertex has no cell".into()));
    }
    for k in &blocks {
        if k.1.abs().max_element() >= 1 << 15 {
            return Err(Error::InvalidData(format!("block {:?} is too far out to be named", k.1)));
        }
    }
    // The faces, as `surface_in` builds them.
    let face_lists: Vec<Vec<Face>> = blocks
        .par_iter()
        .map(|k| {
            let mut faces = Vec::with_capacity(surf[*k].quads.len());
            for q in &surf[*k].quads {
                let Some(ids) = q.iter().map(|(b, c, p)| Some(first[b] + surf[b].vertex(*c, *p)? as u32)).collect::<Option<Vec<u32>>>() else { continue };
                let mut uniq: Vec<u32> = Vec::with_capacity(4);
                for v in ids {
                    if uniq.last() != Some(&v) && uniq.first() != Some(&v) || uniq.is_empty() {
                        uniq.push(v);
                    }
                }
                match uniq.len() {
                    4 => faces.push([uniq[0], uniq[1], uniq[2], uniq[3]]),
                    3 => faces.push([uniq[0], uniq[1], uniq[2], NO_VERT]),
                    _ => {}
                }
            }
            faces
        })
        .collect();
    let mut faces: Vec<Face> = face_lists.into_iter().flatten().collect();
    // Drop vertices no face uses (they would be loose in the document).
    let mut used = vec![false; positions.len()];
    for f in &faces {
        for &v in face_verts(f) {
            used[v as usize] = true;
        }
    }
    if used.iter().any(|u| !u) {
        let mut remap = vec![NO_VERT; positions.len()];
        let mut next = 0u32;
        let (mut p2, mut n2) = (Vec::with_capacity(positions.len()), Vec::with_capacity(positions.len()));
        for i in 0..positions.len() {
            if used[i] {
                remap[i] = next;
                next += 1;
                p2.push(positions[i]);
                n2.push(names[i]);
            }
        }
        for f in &mut faces {
            for v in f.iter_mut().filter(|v| **v != NO_VERT) {
                *v = remap[*v as usize];
            }
        }
        positions = p2;
        names = n2;
    }
    Ok((PolyMesh { positions, faces }, names, stats))
}

/// A vertex name: level (3 bits), block z, y, x (16 bits each, offset), cell (9 bits), piece (2 bits). Ascending in
/// the extraction order.
fn name(k: Key, cell: usize, piece: usize) -> u64 {
    let o = |x: i32| ((x + (1 << 15)) as u64) & 0xFFFF;
    (k.0 as u64) << 61 | o(k.1.z) << 45 | o(k.1.y) << 29 | o(k.1.x) << 13 | (cell as u64) << 4 | piece as u64
}

/// Edges not shared by exactly two faces (open or non-manifold), and edges two faces walk the same way (flipped).
pub fn edge_defects(faces: &[Face]) -> (usize, usize) {
    let mut uses: std::collections::HashMap<(u32, u32), (u32, u32)> = std::collections::HashMap::with_capacity(faces.len() * 2);
    for f in faces {
        let n = face_len(f);
        for k in 0..n {
            let (a, b) = (f[k], f[(k + 1) % n]);
            let e = uses.entry((a.min(b), a.max(b))).or_insert((0, 0));
            if a < b {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
    }
    let bad = uses.values().filter(|(f, b)| f + b != 2).count();
    let flipped = uses.values().filter(|(f, b)| f + b == 2 && *f != 1).count();
    (bad, flipped)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

impl MeshLayers {
    /// Extract the volume's surface and start with no layers.
    pub fn from_volume(vol: &LevelVolume, settings: MeshLayerSettings) -> Result<MeshLayers> {
        let (mesh, keys, _) = extract_named(vol, settings.cap)?;
        let doc = Document::from_mesh(mesh)?;
        Ok(MeshLayers { doc, keys, settings, step: step_of(vol, &settings) })
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
        let (doc, s2) = carry(&self.doc, &self.keys, new, &keys, changed, self.settings.margin_steps * self.step, self.step)?;
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

    // ------------------------------------------------------------------ files

    /// Write the layers, channels and freeze, keyed by vertex name, to `dir` (a directory). The mesh is not written.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir.join("blobs"))?;
        let doc = &self.doc;
        let to_canon = self.internal_to_canonical();
        let keys_blob = write_blob(dir, "keys", "u64", &self.keys)?;
        let freeze = if doc.freeze.iter().any(|&f| f != 0.0) { Some(write_blob(dir, "freeze", "f32", &doc.to_canonical(&doc.freeze))?) } else { None };
        let mut channels = Vec::new();
        for (i, (name, values)) in doc.channels.iter().enumerate() {
            channels.push((name.clone(), write_blob(dir, &format!("channel_{i}"), "f32", &doc.to_canonical(values))?));
        }
        let mut layers = Vec::new();
        for layer in &doc.layers {
            let mut sparse: Vec<(u32, Vec3)> = Vec::new();
            for l in layer.allocated_leaves() {
                let start = doc.bvh.leaves[l as usize].owned.start as usize;
                for (k, d) in layer.chunks[l as usize].as_ref().unwrap().iter().enumerate() {
                    if *d != Vec3::ZERO {
                        sparse.push((to_canon[start + k], *d));
                    }
                }
            }
            sparse.sort_unstable_by_key(|s| s.0);
            let (idx, deltas): (Vec<u32>, Vec<Vec3>) = sparse.into_iter().unzip();
            let tag = format!("layer_{}", layer.id.0);
            layers.push(SavedLayer {
                entry: LayerEntry {
                    id: layer.id.0,
                    name: layer.name.clone(),
                    opacity: layer.opacity,
                    visible: layer.visible,
                    locked: layer.locked,
                    kind: layer.kind,
                    parent: layer.parent.map(|p| p.0),
                    collapsed: layer.collapsed,
                    blend: layer.blend,
                    mask: layer.mask.clone(),
                    generator: layer.generator.clone(),
                    indices: None,
                    deltas: None,
                },
                indices: write_blob(dir, &format!("{tag}_indices"), "u32", &idx)?,
                deltas: write_blob(dir, &format!("{tag}_deltas"), "f32x3", &deltas)?,
            });
        }
        let m = Manifest {
            format: FORMAT.into(),
            version: 1,
            cap: self.settings.cap,
            margin_steps: self.settings.margin_steps,
            step: self.settings.step,
            keys: keys_blob,
            freeze,
            channels,
            layers,
            active: doc.active.map(|l| l.0),
            solo: doc.solo.map(|l| l.0),
            next_layer_id: doc.next_layer_id,
            metadata: doc.metadata.clone(),
        };
        std::fs::write(dir.join(MANIFEST), serde_json::to_string_pretty(&m)?)?;
        Ok(())
    }

    /// Read what [`Self::save`] wrote, extracting the mesh again from `vol`. Data whose vertex name the surface no
    /// longer has is dropped (the second value counts those vertices).
    pub fn load(dir: &Path, vol: &LevelVolume) -> Result<(MeshLayers, usize)> {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST))?)?;
        if m.format != FORMAT {
            return Err(Error::Format(format!("not a {FORMAT} manifest")));
        }
        let settings = MeshLayerSettings { cap: m.cap, margin_steps: m.margin_steps, step: m.step };
        let mut out = MeshLayers::from_volume(vol, settings)?;
        let saved_keys: Vec<u64> = read_blob(dir, &m.keys, "u64")?;
        // Saved canonical index -> internal index of the new document.
        let to_new: Vec<u32> = saved_keys.par_iter().map(|k| out.keys.binary_search(k).map_or(NO_VERT, |c| out.doc.canonical_to_internal[c])).collect();
        let lost = to_new.iter().filter(|&&v| v == NO_VERT).count();
        let n = out.doc.vertex_count();
        let scalar = |b: &Blob| -> Result<Vec<f32>> {
            let vals: Vec<f32> = read_blob(dir, b, "f32")?;
            if vals.len() != saved_keys.len() {
                return Err(Error::Format(format!("{}: {} values for {} vertices", b.path, vals.len(), saved_keys.len())));
            }
            let mut dense = vec![0.0f32; n];
            for (v, x) in to_new.iter().zip(vals) {
                if *v != NO_VERT {
                    dense[*v as usize] = x;
                }
            }
            Ok(dense)
        };
        let doc = &mut out.doc;
        if let Some(b) = &m.freeze {
            doc.freeze = scalar(b)?;
        }
        for (name, b) in &m.channels {
            let vals = scalar(b)?;
            doc.channels.insert(name.clone(), vals);
        }
        let leaves = doc.bvh.leaves.len();
        for sl in m.layers {
            let e = sl.entry;
            let idx: Vec<u32> = read_blob(dir, &sl.indices, "u32")?;
            let deltas: Vec<Vec3> = read_blob(dir, &sl.deltas, "f32x3")?;
            if idx.len() != deltas.len() || idx.iter().any(|&i| i as usize >= saved_keys.len()) {
                return Err(Error::Format(format!("layer '{}' has inconsistent sparse data", e.name)));
            }
            let mut layer = SculptLayer::new(LayerId(e.id), &e.name, leaves);
            layer.opacity = e.opacity;
            layer.visible = e.visible;
            layer.locked = e.locked;
            layer.kind = e.kind;
            layer.parent = e.parent.map(LayerId);
            layer.collapsed = e.collapsed;
            layer.blend = e.blend;
            layer.mask = e.mask;
            layer.generator = e.generator;
            for (ci, d) in idx.iter().zip(deltas) {
                let v = to_new[*ci as usize];
                if v == NO_VERT {
                    continue;
                }
                let v = v as usize;
                let l = doc.bvh.vert_leaf[v] as usize;
                let leaf = &doc.bvh.leaves[l];
                let len = leaf.owned_len();
                let chunk = layer.chunks[l].get_or_insert_with(|| vec![Vec3::ZERO; len].into_boxed_slice());
                chunk[v - leaf.owned.start as usize] = d;
            }
            doc.layers.push(layer);
        }
        doc.next_layer_id = m.next_layer_id.max(doc.layers.iter().map(|l| l.id.0 + 1).max().unwrap_or(1));
        doc.active = m.active.map(LayerId).filter(|id| doc.layers.iter().any(|l| l.id == *id && l.is_sculpt_target()));
        doc.solo = m.solo.map(LayerId).filter(|id| doc.layers.iter().any(|l| l.id == *id));
        doc.metadata = m.metadata;
        doc.refresh_scales();
        for i in 0..doc.layers.len() {
            if let Some(stack) = doc.layers[i].mask.clone() {
                doc.layers[i].mask_values = doc.mask_values_for(&stack)?;
            }
        }
        doc.recomposite_all();
        doc.dirty_channels.clear();
        doc.undo.clear();
        Ok((out, lost))
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn invert(perm: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p as usize] = i as u32;
    }
    out
}

/// The finest shown level's voxel among levels that hold blocks.
fn step_of(vol: &LevelVolume, s: &MeshLayerSettings) -> f32 {
    if let Some(step) = s.step {
        return step;
    }
    let top = s.cap.min(vol.levels() - 1);
    let l = (0..=top).rev().find(|&l| vol.level_len(l) > 0).unwrap_or(0);
    vol.voxel(l)
}

#[derive(Default)]
struct Carried {
    match_ms: f64,
    project_ms: f64,
    recomposite_ms: f64,
    copied: usize,
    projected: usize,
    search_faces: usize,
}

/// `new` (fresh from `Document::from_mesh`, names `new_keys` in canonical order) with `old`'s layers, channels and
/// freeze carried onto it.
fn carry(old: &Document, old_keys: &[u64], mut new: Document, new_keys: &[u64], changed: Aabb, margin: f32, step: f32) -> Result<(Document, Carried)> {
    let mut st = Carried::default();
    let grown = if changed.is_empty() { changed } else { Aabb { min: changed.min - Vec3::splat(margin), max: changed.max + Vec3::splat(margin) } };
    let inside = |p: Vec3| !grown.is_empty() && p.cmpge(grown.min).all() && p.cmple(grown.max).all();
    // Name match, per new internal vertex: the old vertex to copy from, or `NONE`.
    let new_n = new.vertex_count();
    let t = Instant::now();
    let by_canon: Vec<u32> = new_keys
        .par_iter()
        .enumerate()
        .map(|(c, k)| match old_keys.binary_search(k) {
            Ok(oc) => {
                let ov = old.canonical_to_internal[oc];
                let p = new.base[new.canonical_to_internal[c] as usize];
                if old.base[ov as usize] == p && !inside(p) { ov } else { NONE }
            }
            Err(_) => NONE,
        })
        .collect();
    let mut src: Vec<u32> = crate::mesh::permute(&new.canonical_to_internal, &by_canon);
    drop(by_canon);
    st.match_ms = ms(t);
    // The rest: closest point on the old base surface near them. `src` then holds `PROJ | slot` into `proj`.
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
            // A sub-mesh of the old base: its own vertex numbers, and back.
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
    // Old base normals at the projected triangles' corners, for splitting an offset into height and the rest.
    let normals: Vec<[Vec3; 3]> = proj
        .par_iter()
        .map(|p| p.tri.map(|v| crate::mesh::vertex_normal(&old.base, &old.faces, &old.topo, v as usize).normalize_or_zero()))
        .collect();
    st.project_ms = ms(t);

    let old_leaf = &old.bvh.vert_leaf;
    let old_get = |layer: &SculptLayer, v: u32| -> Vec3 {
        let l = old_leaf[v as usize] as usize;
        match &layer.chunks[l] {
            Some(c) => c[v as usize - old.bvh.leaves[l].owned.start as usize],
            None => Vec3::ZERO,
        }
    };
    let new_normals = &new.normals;
    let (src, proj) = (&src, &proj);
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
    new.freeze = if old.freeze.iter().any(|&f| f != 0.0) { carry_f(&old.freeze).into_iter().map(|f| f.clamp(0.0, 1.0)).collect() } else { vec![0.0; new_n] };
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
    st.recomposite_ms = ms(t);
    new.dirty_channels.clear();
    new.undo.clear();
    Ok((new, st))
}

// ---------------------------------------------------------------------- file format

const FORMAT: &str = "sculpt-mesh-layers";
const MANIFEST: &str = "mesh_layers.json";

#[derive(Serialize, Deserialize)]
struct Blob {
    path: String,
    dtype: String,
    count: u64,
}

#[derive(Serialize, Deserialize)]
struct SavedLayer {
    #[serde(flatten)]
    entry: LayerEntry,
    #[serde(rename = "key_indices")]
    indices: Blob,
    #[serde(rename = "key_deltas")]
    deltas: Blob,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    cap: usize,
    margin_steps: f32,
    step: Option<f32>,
    /// Vertex names in the order of every per-vertex blob.
    keys: Blob,
    freeze: Option<Blob>,
    channels: Vec<(String, Blob)>,
    layers: Vec<SavedLayer>,
    active: Option<u32>,
    solo: Option<u32>,
    next_layer_id: u32,
    metadata: BTreeMap<String, serde_json::Value>,
}

fn write_blob<T: bytemuck::Pod>(dir: &Path, name: &str, dtype: &str, data: &[T]) -> Result<Blob> {
    let rel = format!("blobs/{name}.bin");
    std::fs::write(dir.join(&rel), bytemuck::cast_slice(data))?;
    Ok(Blob { path: rel, dtype: dtype.into(), count: data.len() as u64 })
}

fn read_blob<T: bytemuck::Pod>(dir: &Path, b: &Blob, dtype: &str) -> Result<Vec<T>> {
    if b.dtype != dtype {
        return Err(Error::Format(format!("{}: expected {dtype}, found {}", b.path, b.dtype)));
    }
    if b.path.contains("..") || Path::new(&b.path).is_absolute() {
        return Err(Error::Format(format!("blob path escapes the directory: {}", b.path)));
    }
    let bytes = std::fs::read(dir.join(&b.path))?;
    if bytes.len() != b.count as usize * std::mem::size_of::<T>() {
        return Err(Error::Format(format!("{}: expected {} values, found {} bytes", b.path, b.count, bytes.len())));
    }
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}

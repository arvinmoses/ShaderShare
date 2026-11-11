//! Carrying layers, channels and freeze from one extracted mesh onto the next.
//!
//! In two parts: [`plan`] works out where each new vertex takes its data from (geometry only: the two meshes and the
//! volume before and after), and [`apply`] moves the data. A vertex whose name and base position both survived copies
//! its data. Any other is carried from the old surface under it: the point of the old surface it came from is found in
//! the old volume's field (down its gradient, or failing that along the new normal), the old mesh's triangle there
//! gives the data, and an offset's height goes along the new normal.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use super::{Patch, ms};
use crate::bvh::Bvh;
use crate::document::Document;
use crate::generator::level_volume::LevelVolume;
use crate::generator::voxel::MeshSdf;
use crate::geom::Aabb;
use crate::layers::SculptLayer;
use crate::mesh::{Face, NO_VERT, Topology, face_verts};
use crate::Result;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Where a carried vertex takes its data from on the old base: the old triangle (internal), weights, the old normals
/// at its corners, the new normal, how much to keep, and how much of the offset across the normal.
#[derive(Clone, Copy)]
struct Proj {
    tri: [u32; 3],
    w: [f32; 3],
    n: [Vec3; 3],
    dir: Vec3,
    keep: f32,
    whole: f32,
}

/// No old surface found: no data.
const NO_PROJ: Proj = Proj { tri: [NO_VERT; 3], w: [1.0, 0.0, 0.0], n: [Vec3::ZERO; 3], dir: Vec3::ZERO, keep: 0.0, whole: 0.0 };

/// A carry source: an old vertex, `PROJ | slot` into the projections, or nothing.
const NONE: u32 = u32::MAX;
const PROJ: u32 = 1 << 31;

/// A mesh's geometry as the plan reads it: base positions and faces (internal order), canonical -> internal, and the
/// vertex names in canonical order.
#[derive(Clone, Copy)]
pub(super) struct Geo<'a> {
    pub(super) base: &'a [Vec3],
    pub(super) faces: &'a [Face],
    pub(super) canon: &'a [u32],
    pub(super) keys: &'a [u64],
}

impl<'a> Geo<'a> {
    pub(super) fn of(doc: &'a Document, keys: &'a [u64]) -> Geo<'a> {
        Geo { base: &doc.base, faces: &doc.faces, canon: &doc.canonical_to_internal, keys }
    }
}

/// Where each new vertex takes its data from.
pub(super) struct Plan {
    /// Per new internal vertex.
    src: Vec<u32>,
    proj: Vec<Proj>,
    pub(super) match_ms: f64,
    pub(super) project_ms: f64,
    pub(super) copied: usize,
    pub(super) projected: usize,
    /// Of `projected`: vertices no old surface was found under (they start with no data).
    pub(super) unplaced: usize,
    pub(super) search_faces: usize,
}

/// The point of `pre`'s surface (levels up to `max`) that `p` came from: down the field's gradient, or failing that
/// (the field holds no distance that far out) along `dir`, the surface's normal at `p` now. `step` is `p`'s lattice
/// step.
fn foot(pre: &LevelVolume, max: usize, p: Vec3, dir: Vec3, step: f32) -> Option<Vec3> {
    let value = |q: Vec3| pre.value_upto(q, max, false);
    let v0 = value(p);
    let (mut q, mut v) = (p, v0);
    for _ in 0..4 {
        if v.abs() <= 0.02 * step {
            return Some(q);
        }
        let n = pre.normal_upto(q, max, false);
        if n == Vec3::ZERO {
            break;
        }
        q -= n * v;
        v = value(q);
    }
    if v.abs() <= 0.1 * step {
        return Some(q);
    }
    if dir == Vec3::ZERO {
        return None;
    }
    // Along the normal, towards the old surface: by the field's own distances until its sign turns, then halving.
    let d = if v0 > 0.0 { -dir } else { dir };
    let (mut t0, mut t1, mut v1) = (0.0f32, 0.0f32, v0);
    for _ in 0..64 {
        if (v1 < 0.0) != (v0 < 0.0) {
            break;
        }
        t0 = t1;
        t1 += v1.abs().max(0.25 * step);
        if t1 > 24.0 * step {
            return None;
        }
        v1 = value(p + d * t1);
    }
    if (v1 < 0.0) == (v0 < 0.0) {
        return None;
    }
    for _ in 0..12 {
        let t = 0.5 * (t0 + t1);
        if (value(p + d * t) < 0.0) == (v0 < 0.0) { t0 = t } else { t1 = t }
    }
    Some(p + d * (0.5 * (t0 + t1)))
}

/// Where each vertex of the new mesh takes its data from. `new_normals` are the new mesh's normals (internal order),
/// `pre` the volume the old mesh was extracted from, `post` the one the new mesh was, `cap` the finest level shown.
pub(super) fn plan(old: Geo, new: Geo, new_normals: &[Vec3], pre: &LevelVolume, post: &LevelVolume, cap: usize) -> Plan {
    let new_n = new.base.len();
    // Name match, per new vertex: the old vertex to copy from when its base position is the same bit for bit.
    let t = Instant::now();
    let by_canon: Vec<u32> = new
        .keys
        .par_iter()
        .enumerate()
        .map(|(c, k)| match old.keys.binary_search(k) {
            Ok(oc) => {
                let ov = old.canon[oc];
                if old.base[ov as usize] == new.base[new.canon[c] as usize] { ov } else { NONE }
            }
            Err(_) => NONE,
        })
        .collect();
    // The rest, with each one's level.
    let todo: Vec<(u32, u8)> = by_canon.par_iter().enumerate().filter(|(_, s)| **s == NONE).map(|(c, _)| (new.canon[c], (new.keys[c] >> 61) as u8)).collect();
    let mut src: Vec<u32> = crate::mesh::permute(new.canon, &by_canon);
    drop(by_canon);
    let mut out = Plan { src: Vec::new(), proj: Vec::new(), match_ms: ms(t), project_ms: 0.0, copied: new_n - todo.len(), projected: todo.len(), unplaced: 0, search_faces: 0 };
    let t = Instant::now();
    if !todo.is_empty() {
        let (max_pre, max_post) = (cap.min(pre.levels() - 1), cap.min(post.levels() - 1));
        // Each one's normal now, the point of the old surface under it, and the step its search is measured in: its
        // own level's or the old surface's there, whichever is coarser.
        struct Foot {
            dir: Vec3,
            at: Option<Vec3>,
            level: u8,
            step: f32,
        }
        let feet: Vec<Foot> = todo
            .par_iter()
            .map(|&(v, l)| {
                let p = new.base[v as usize];
                let field = post.normal_upto(p, max_post, false);
                let dir = if field == Vec3::ZERO { new_normals[v as usize].normalize_or_zero() } else { field };
                let step = post.voxel((l as usize).min(max_post));
                let at = foot(pre, max_pre, p, dir, step);
                let level = at.map_or(0, |q| pre.level_of(q).min(max_pre)) as u8;
                Foot { dir, at, level, step: step.max(pre.voxel(level as usize)) }
            })
            .collect();
        // The old faces to search: those with a vertex in the old surface's block at a foot, or in a block next to
        // it. (One box around every foot would take in everything between two strokes far apart.)
        let block = |p: Vec3, l: u8| ((p - pre.origin()) / (pre.voxel(l as usize) * 8.0)).floor().as_ivec3();
        let mut blocks: HashSet<(u8, IVec3)> = HashSet::new();
        let mut region = Aabb::EMPTY;
        for f in &feet {
            if let Some(q) = f.at {
                blocks.insert((f.level, block(q, f.level)));
            }
        }
        let mut levels: Vec<u8> = Vec::new();
        for &(l, b) in &blocks {
            if !levels.contains(&l) {
                levels.push(l);
            }
            let s = pre.voxel(l as usize) * 8.0;
            region.grow(pre.origin() + (b - IVec3::ONE).as_vec3() * s);
            region.grow(pre.origin() + (b + IVec3::splat(2)).as_vec3() * s);
        }
        let near: HashSet<(u8, IVec3)> = blocks.iter().flat_map(|&(l, b)| (0..27).map(move |k| (l, b + IVec3::new(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1)))).collect();
        let marked = |p: Vec3| p.cmpge(region.min).all() && p.cmple(region.max).all() && levels.iter().any(|&l| near.contains(&(l, block(p, l))));
        let faces: Vec<Face> = if blocks.is_empty() { Vec::new() } else { old.faces.par_iter().filter(|f| face_verts(f).iter().any(|&v| marked(old.base[v as usize]))).copied().collect() };
        out.search_faces = faces.len();
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
            out.proj = feet
                .par_iter()
                .map(|f| {
                    let Some(q) = f.at else { return NO_PROJ };
                    let Some((cp, tri, w)) = sdf.closest(q, 1.5 * f.step) else { return NO_PROJ };
                    let n = tri.map(|t| pseudo[t as usize].normalize_or_zero());
                    let old_n = (n[0] * w[0] + n[1] * w[1] + n[2] * w[2]).normalize_or_zero();
                    // Less where the old mesh lies off the old field's surface, and nothing where the surface now faces
                    // away from the old one under it (thin or folded forms: the other side's data). A wall a stroke
                    // raised faces sideways to the surface it rose from, and keeps its data.
                    let near = 1.0 - smoothstep(1.0 * f.step, 1.5 * f.step, q.distance(cp));
                    let agree = f.dir.dot(old_n);
                    let facing = smoothstep(-0.2, 0.2, agree);
                    let whole = smoothstep(0.9, 0.98, agree);
                    Proj { tri: tri.map(|t| back2[t as usize]), w, n, dir: f.dir, keep: near * facing, whole }
                })
                .collect();
        } else {
            out.proj = vec![NO_PROJ; todo.len()];
        }
        out.unplaced = out.proj.iter().filter(|p| p.tri[0] == NO_VERT).count();
        for (slot, &(v, _)) in todo.iter().enumerate() {
            src[v as usize] = PROJ | slot as u32;
        }
    }
    out.project_ms = ms(t);
    out.src = src;
    out
}

/// `new` (fresh from `Document::from_mesh`) with `old`'s layers, channels and freeze carried onto it as `plan` says,
/// what the carry did not copy (to put back with [`super::MeshLayers::restore`]), and the time recompositing took.
pub(super) fn apply(old: &Document, old_keys: &[u64], mut new: Document, plan: &Plan) -> Result<(Document, Patch, f64)> {
    let new_n = new.vertex_count();
    let old_leaf = &old.bvh.vert_leaf;
    let old_get = |layer: &SculptLayer, v: u32| -> Vec3 {
        let l = old_leaf[v as usize] as usize;
        match &layer.chunks[l] {
            Some(c) => c[v as usize - old.bvh.leaves[l].owned.start as usize],
            None => Vec3::ZERO,
        }
    };
    let (src, proj) = (&plan.src, &plan.proj);
    // The old vertices nothing copied from, by name: their data is carried by closest point at best.
    let patch = {
        let mut used = vec![false; old.vertex_count()];
        for &s in src.iter().filter(|s| **s != NONE && **s & PROJ == 0) {
            used[s as usize] = true;
        }
        let gone: Vec<(u64, u32)> = old_keys.par_iter().zip(old.canonical_to_internal.par_iter()).filter(|(_, v)| !used[**v as usize]).map(|(k, v)| (*k, *v)).collect();
        let scalar = |vals: &[f32]| -> Vec<f32> { gone.iter().map(|&(_, v)| vals[v as usize]).collect() };
        Patch {
            names: gone.iter().map(|g| g.0).collect(),
            base: gone.iter().map(|&(_, v)| old.base[v as usize]).collect(),
            layers: old
                .layers
                .iter()
                .map(|layer| {
                    let vals: Vec<Vec3> = if layer.chunks.iter().any(|c| c.is_some()) { gone.iter().map(|&(_, v)| old_get(layer, v)).collect() } else { Vec::new() };
                    (layer.id, vals.iter().any(|d| *d != Vec3::ZERO).then_some(vals))
                })
                .collect(),
            freeze: old.freeze.iter().any(|&f| f != 0.0).then(|| scalar(&old.freeze)),
            channels: old.channels.iter().map(|(k, v)| (k.clone(), scalar(v))).collect(),
        }
    };
    let carry_vec = |layer: &SculptLayer, v: usize| -> Vec3 {
        let s = src[v];
        if s == NONE {
            Vec3::ZERO
        } else if s & PROJ == 0 {
            old_get(layer, s)
        } else {
            let Proj { tri, w, n, dir, keep, whole } = proj[(s & !PROJ) as usize];
            if keep <= 0.0 {
                return Vec3::ZERO;
            }
            let d = tri.map(|o| old_get(layer, o));
            let full = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
            let height: f32 = (0..3).map(|k| d[k].dot(n[k]) * w[k]).sum();
            let along = dir * height;
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
                    if tri[0] == NO_VERT {
                        return 0.0;
                    }
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
    let recomposite_ms = ms(t);
    new.dirty_channels.clear();
    new.undo.clear();
    Ok((new, patch, recomposite_ms))
}

/// Put back what `patch` holds onto `doc` (names `keys` in canonical order), wherever a vertex of the same name sits
/// at the same base position bit for bit. Returns how many vertices that was.
pub(super) fn restore(doc: &mut Document, keys: &[u64], patch: &Patch) -> Result<usize> {
    // Per patch entry: the vertex it goes back to.
    let to: Vec<u32> = patch
        .names
        .par_iter()
        .zip(patch.base.par_iter())
        .map(|(k, p)| match keys.binary_search(k) {
            Ok(c) => {
                let v = doc.canonical_to_internal[c];
                let b = doc.base[v as usize];
                if [b.x.to_bits(), b.y.to_bits(), b.z.to_bits()] == [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()] { v } else { NONE }
            }
            Err(_) => NONE,
        })
        .collect();
    let back = to.iter().filter(|v| **v != NONE).count();
    if back == 0 {
        return Ok(0);
    }
    let mut touched: HashSet<u32> = HashSet::new();
    for (id, vals) in &patch.layers {
        let Some(layer) = doc.layers.iter_mut().find(|l| l.id == *id) else { continue };
        for (i, &v) in to.iter().enumerate().filter(|(_, v)| **v != NONE) {
            let d = vals.as_ref().map_or(Vec3::ZERO, |vals| vals[i]);
            let l = doc.bvh.vert_leaf[v as usize] as usize;
            let leaf = &doc.bvh.leaves[l];
            if d == Vec3::ZERO && layer.chunks[l].is_none() {
                continue;
            }
            let chunk = layer.chunks[l].get_or_insert_with(|| vec![Vec3::ZERO; leaf.owned_len()].into_boxed_slice());
            let slot = &mut chunk[v as usize - leaf.owned.start as usize];
            if *slot != d {
                *slot = d;
                touched.insert(l as u32);
            }
        }
    }
    let n = doc.vertex_count();
    let mut scalars = false;
    if let Some(vals) = &patch.freeze {
        for (i, &v) in to.iter().enumerate().filter(|(_, v)| **v != NONE) {
            scalars |= doc.freeze[v as usize] != vals[i];
            doc.freeze[v as usize] = vals[i];
        }
    }
    for (name, vals) in &patch.channels {
        let channel = doc.channels.entry(name.clone()).or_insert_with(|| vec![0.0; n]);
        for (i, &v) in to.iter().enumerate().filter(|(_, v)| **v != NONE) {
            scalars |= channel[v as usize] != vals[i];
            channel[v as usize] = vals[i];
        }
    }
    if scalars {
        // Masks read the channels: every masked layer is weighed again.
        for i in 0..doc.layers.len() {
            if let Some(stack) = doc.layers[i].mask.clone() {
                doc.layers[i].mask_values = doc.mask_values_for(&stack)?;
                touched.extend(doc.layers[i].touched_leaves());
            }
        }
        doc.scalar_dirty.mark_all();
        doc.dirty_channels.clear();
    }
    let mut touched: Vec<u32> = touched.into_iter().collect();
    touched.sort_unstable();
    doc.recomposite(&touched);
    Ok(back)
}

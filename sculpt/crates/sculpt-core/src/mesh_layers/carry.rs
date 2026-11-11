//! Carrying layers, channels and freeze from one extracted mesh onto the next.

use std::collections::BTreeMap;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;

use super::ms;
use crate::bvh::Bvh;
use crate::document::Document;
use crate::generator::voxel::MeshSdf;
use crate::geom::Aabb;
use crate::layers::SculptLayer;
use crate::mesh::{Face, NO_VERT, Topology, face_verts};
use crate::Result;

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

#[derive(Default)]
pub(super) struct Carried {
    pub(super) match_ms: f64,
    pub(super) project_ms: f64,
    pub(super) recomposite_ms: f64,
    pub(super) copied: usize,
    pub(super) projected: usize,
    pub(super) search_faces: usize,
}

/// `new` (fresh from `Document::from_mesh`, names `new_keys` in canonical order) with `old`'s layers, channels and
/// freeze carried onto it.
pub(super) fn carry(old: &Document, old_keys: &[u64], mut new: Document, new_keys: &[u64], changed: Aabb, margin: f32, step: f32) -> Result<(Document, Carried)> {
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

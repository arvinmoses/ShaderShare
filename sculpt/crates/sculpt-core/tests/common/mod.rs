//! Scenes and comparisons shared by the mesh-layer gate files.
#![allow(dead_code)]

use std::collections::HashMap;

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::geom::{Aabb, Ray};
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers};
use sculpt_core::{Document, LayerId};

pub const R: f32 = 0.5;

pub fn ball(voxel0: f32, levels: usize, r: f32) -> LevelVolume {
    LevelVolume::from_field(&|p: Vec3| p.length() - r, r + 0.3, voxel0, levels)
}

pub fn small() -> (LevelVolume, MeshLayers) {
    let vol = ball(1.0 / 64.0, 1, R);
    let ml = MeshLayers::from_volume(&vol, MeshLayerSettings::default()).unwrap();
    (vol, ml)
}

/// A clay column on top of the ball (+z), as the spike builds it: back-and-forth strokes over the top.
pub fn column(doc: &mut Document, strokes: usize, radius: f32) -> LayerId {
    let id = doc.add_layer("clay");
    let s = BrushSettings { radius, strength: 0.5, ..Default::default() };
    let mut tall = ClayBuildup { height: 0.32, ..Default::default() };
    for _ in 0..strokes {
        let mut path = Vec::new();
        for k in 0..=16 {
            let t = k as f32 / 16.0;
            let x = -0.05 + 0.1 * if k <= 8 { t * 2.0 } else { 2.0 - t * 2.0 };
            if let Some(h) = doc.raycast(&Ray::new(Vec3::new(x, 0.0, 3.0), -Vec3::Z)) {
                path.push((h.point, 1.0));
            }
        }
        brush::stroke(doc, &mut tall, &s, &path).unwrap();
    }
    id
}

/// Where a volume stroke's dabs landed, and their radius.
pub struct Dabs {
    pub hits: Vec<Vec3>,
    pub r: f32,
}

impl Dabs {
    /// Each dab's hit point grown by the radius.
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        for &h in &self.hits {
            b.grow(h - Vec3::splat(self.r));
            b.grow(h + Vec3::splat(self.r));
        }
        b
    }

    /// Whether `p` is farther than the radius plus `extra` from every dab.
    pub fn outside(&self, p: Vec3, extra: f32) -> bool {
        self.hits.iter().all(|h| h.distance(p) > self.r + extra)
    }

    /// The distance from `p` to the nearest dab's centre.
    pub fn distance(&self, p: Vec3) -> f32 {
        self.hits.iter().map(|h| h.distance(p)).fold(f32::INFINITY, f32::min)
    }
}

/// A volume clay stroke at the finest level around direction `c` (a few dabs along a short arc); returns its box
/// (each dab's hit point grown by the radius).
pub fn vol_stroke(vol: &mut LevelVolume, c: Vec3, r: f32) -> Aabb {
    vol_stroke_dabs(vol, c, r).bounds()
}

/// [`vol_stroke`], returning the dabs.
pub fn vol_stroke_dabs(vol: &mut LevelVolume, c: Vec3, r: f32) -> Dabs {
    let c = c.normalize();
    let side = c.any_orthonormal_vector();
    let l = vol.levels() - 1;
    let mut hits = Vec::new();
    let mut along = Vec3::ZERO;
    vol.begin_stroke();
    for k in 0..=8 {
        let dir = (c + side * (0.15 * (k as f32 / 8.0 - 0.5))).normalize();
        let Some((hit, n)) = vol.raycast(dir * 3.0, -dir, l, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = vol.plane_dab(l, hit, rough, r, r * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        hits.push(hit);
    }
    vol.end_stroke();
    Dabs { hits, r }
}

/// One clay stroke at level `l` on the surface near `at` (outward normal `n` there): `passes` runs of dabs back and
/// forth along a line `span` long, each dab's plane `h` above the surface.
#[allow(clippy::too_many_arguments)]
pub fn vol_stroke_on(vol: &mut LevelVolume, at: Vec3, n: Vec3, r: f32, l: usize, h: f32, span: f32, passes: usize) -> Dabs {
    let n = n.normalize();
    let side = n.any_orthonormal_vector();
    let mut hits = Vec::new();
    let mut along = Vec3::ZERO;
    vol.begin_stroke();
    for pass in 0..passes {
        for k in 0..=8 {
            let t = if pass % 2 == 0 { k } else { 8 - k } as f32 / 8.0 - 0.5;
            let from = at + side * (span * t) + n * (4.0 * r);
            let Some((hit, hn)) = vol.raycast(from, -n, l, true) else { continue };
            let rough = if along == Vec3::ZERO { hn } else { along };
            if let Some((pn, _)) = vol.plane_dab(l, hit, rough, r, h, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { pn } else { (along + pn).normalize() };
            }
            hits.push(hit);
        }
    }
    vol.end_stroke();
    Dabs { hits, r }
}

/// A smooth field between -1 and 1 with features about 0.2 across.
pub fn noise(p: Vec3) -> f32 {
    ((31.0 * p.x + 1.0).sin() + (29.0 * p.y + 2.0).sin() + (33.0 * p.z + 3.0).sin()) / 3.0
}

/// A new layer holding `f(vertex)` at every vertex (`f` reads the document as it is before the layer).
pub fn fill_layer(doc: &mut Document, name: &str, f: impl Fn(&Document, usize) -> Vec3 + Sync) -> LayerId {
    let id = doc.add_layer(name);
    let b = doc.bounds();
    let d = doc.compute_displacements(b.center(), b.extent().length() * 2.0 + 1.0, |doc, v, _| Some(f(doc, v)));
    doc.apply_displacements(&d).unwrap();
    id
}

/// The layer's |delta| summed over the base surface (each face's area times its corners' mean), over the faces
/// `keep` takes by their centre, and those faces' area.
pub fn delta_integral(ml: &MeshLayers, id: LayerId, keep: impl Fn(Vec3) -> bool) -> (f64, f64) {
    let d = ml.layer_deltas(id).unwrap();
    let base = ml.doc.base_positions();
    let (mut sum, mut total) = (0.0f64, 0.0f64);
    for f in ml.doc.faces() {
        let vs = sculpt_core::mesh::face_verts(f);
        let c = vs.iter().map(|&v| base[v as usize]).sum::<Vec3>() / vs.len() as f32;
        if !keep(c) {
            continue;
        }
        let mut area = 0.0;
        for k in 1..vs.len() - 1 {
            area += 0.5 * (base[vs[k] as usize] - base[vs[0] as usize]).cross(base[vs[k + 1] as usize] - base[vs[0] as usize]).length();
        }
        total += area as f64;
        sum += (area * vs.iter().map(|&v| d[v as usize].length()).sum::<f32>() / vs.len() as f32) as f64;
    }
    (sum, total)
}

pub fn grown(b: &Aabb, by: f32) -> Aabb {
    Aabb { min: b.min - Vec3::splat(by), max: b.max + Vec3::splat(by) }
}

pub fn inside(b: &Aabb, p: Vec3) -> bool {
    p.cmpge(b.min).all() && p.cmple(b.max).all()
}

/// (sum, max) of |delta| of layer `id`.
pub fn delta_stats(ml: &MeshLayers, id: LayerId) -> (f64, f32) {
    let d = ml.layer_deltas(id).unwrap();
    (d.iter().map(|v| v.length() as f64).sum(), d.iter().map(|v| v.length()).fold(0.0, f32::max))
}

/// Per vertex name: (base, delta of `id`).
pub fn by_key(ml: &MeshLayers, id: LayerId) -> HashMap<u64, (Vec3, Vec3)> {
    let d = ml.layer_deltas(id).unwrap();
    let to_c = ml.internal_to_canonical();
    (0..ml.doc.vertex_count()).map(|v| (ml.keys()[to_c[v] as usize], (ml.doc.base_positions()[v], d[v]))).collect()
}

pub fn bits(a: &[Vec3]) -> Vec<[u32; 3]> {
    a.iter().map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect()
}

pub fn same(a: &[Vec3], b: &[Vec3]) -> bool {
    bits(a) == bits(b)
}


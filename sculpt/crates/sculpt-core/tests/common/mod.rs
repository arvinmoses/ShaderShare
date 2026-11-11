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

/// A volume clay stroke at the finest level around direction `c` (a few dabs along a short arc); returns its box
/// (each dab's hit point grown by the radius).
pub fn vol_stroke(vol: &mut LevelVolume, c: Vec3, r: f32) -> Aabb {
    let c = c.normalize();
    let side = c.any_orthonormal_vector();
    let l = vol.levels() - 1;
    let mut b = Aabb::EMPTY;
    let mut along = Vec3::ZERO;
    vol.begin_stroke();
    for k in 0..=8 {
        let dir = (c + side * (0.15 * (k as f32 / 8.0 - 0.5))).normalize();
        let Some((hit, n)) = vol.raycast(dir * 3.0, -dir, l, true) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = vol.plane_dab(l, hit, rough, r, r * 0.095, 0.5, 0.4, false) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        b.grow(hit - Vec3::splat(r));
        b.grow(hit + Vec3::splat(r));
    }
    vol.end_stroke();
    b
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


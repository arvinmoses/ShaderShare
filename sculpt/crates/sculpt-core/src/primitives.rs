//! Starter meshes.

use glam::Vec3;

use crate::mesh::{Face, PolyMesh, face_normal, face_verts};
use crate::subdiv::Subdivider;

/// Unit cube (`[-1, 1]^3`) with outward-facing quads.
pub fn cube() -> PolyMesh {
    let positions: Vec<Vec3> = (0..8)
        .map(|i| Vec3::new((i & 1) as f32 * 2.0 - 1.0, ((i >> 1) & 1) as f32 * 2.0 - 1.0, ((i >> 2) & 1) as f32 * 2.0 - 1.0))
        .collect();
    let mut faces: Vec<Face> =
        vec![[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]];
    for f in &mut faces {
        let c: Vec3 = face_verts(f).iter().map(|&v| positions[v as usize]).sum::<Vec3>() / 4.0;
        if face_normal(&positions, f).dot(c) < 0.0 {
            f.reverse();
        }
    }
    PolyMesh { positions, faces }
}

/// All-quad sphere: a cube subdivided `level` times and projected to `radius`.
/// Face count is `6 * 4^level`.
pub fn quad_sphere(level: u32, radius: f32) -> PolyMesh {
    let mut m = cube();
    for _ in 0..level {
        let s = Subdivider::new(m.positions.len(), &m.faces);
        m = PolyMesh { positions: s.apply(&m.positions), faces: s.new_faces().to_vec() };
    }
    for p in &mut m.positions {
        *p = p.normalize() * radius;
    }
    m
}

/// Flat `n x n` quad grid in the XZ plane, `[-size/2, size/2]`, facing +Y.
pub fn grid(n: u32, size: f32) -> PolyMesh {
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            positions.push(Vec3::new(i as f32 / n as f32 - 0.5, 0.0, j as f32 / n as f32 - 0.5) * size);
        }
    }
    let idx = |i: u32, j: u32| j * (n + 1) + i;
    let mut faces = Vec::new();
    for j in 0..n {
        for i in 0..n {
            faces.push([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)]);
        }
    }
    PolyMesh { positions, faces }
}

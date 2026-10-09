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

/// All-quad sphere with `res x res` quads on each of the six cube faces (`6 * res^2` quads in total).
/// Unlike [`quad_sphere`] the size is not tied to powers of four, so benchmarks can ask for 10M quads.
pub fn quad_sphere_res(res: u32, radius: f32) -> PolyMesh {
    let n = res as usize;
    let side = n + 1;
    // Lattice point (x, y, z) on the cube surface, each in 0..=n, is shared by every face that touches it.
    let key = |x: usize, y: usize, z: usize| (x * side + y) * side + z;
    let on_surface = |x: usize, y: usize, z: usize| x == 0 || x == n || y == 0 || y == n || z == 0 || z == n;
    let mut index = vec![u32::MAX; side * side * side];
    let mut positions = Vec::with_capacity(6 * n * n + 2);
    let mut vert = |x: usize, y: usize, z: usize| -> u32 {
        debug_assert!(on_surface(x, y, z));
        let k = key(x, y, z);
        if index[k] == u32::MAX {
            index[k] = positions.len() as u32;
            let p = Vec3::new(x as f32, y as f32, z as f32) / n as f32 * 2.0 - Vec3::ONE;
            positions.push(p.normalize() * radius);
        }
        index[k]
    };
    let mut faces: Vec<Face> = Vec::with_capacity(6 * n * n);
    // Each cube face: a fixed axis value, and two free axes (a, b), wound so the quad faces outward.
    for (fixed, at, flip) in [(0usize, 0usize, true), (0, n, false), (1, 0, false), (1, n, true), (2, 0, true), (2, n, false)] {
        let (a, b) = match fixed {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        for i in 0..n {
            for j in 0..n {
                let mut corner = |di: usize, dj: usize| {
                    let mut c = [0usize; 3];
                    c[fixed] = at;
                    c[a] = i + di;
                    c[b] = j + dj;
                    vert(c[0], c[1], c[2])
                };
                let mut q = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
                if flip {
                    q.reverse();
                }
                faces.push(q);
            }
        }
    }
    PolyMesh { positions, faces }
}

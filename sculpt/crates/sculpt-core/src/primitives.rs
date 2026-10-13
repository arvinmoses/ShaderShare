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
    let m = n.saturating_sub(1);
    let vertex_count = 6 * n * n + 2;
    // Vertex ids: 8 cube corners, then 12 edges of `n - 1` points, then six faces of `(n - 1)^2` points.
    let edge_base = 8;
    let face_base = edge_base + 12 * m;
    // Id of lattice point (x, y, z), each in 0..=n and at least one on the cube surface (0 or n).
    let id = |p: [usize; 3]| -> u32 {
        let hi = |a: usize| p[a] == n;
        let on = |a: usize| p[a] == 0 || p[a] == n;
        let count = (0..3).filter(|&a| on(a)).count();
        (match count {
            3 => (hi(0) as usize) | (hi(1) as usize) << 1 | (hi(2) as usize) << 2,
            2 => {
                let free = (0..3).find(|&a| !on(a)).unwrap();
                let fixed: Vec<usize> = (0..3).filter(|&a| a != free).collect();
                let bits = (hi(fixed[0]) as usize) | (hi(fixed[1]) as usize) << 1;
                edge_base + free * 4 * m + bits * m + (p[free] - 1)
            }
            _ => {
                let k = (0..3).find(|&a| on(a)).unwrap();
                let (a, b) = match k {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                face_base + (k * 2 + hi(k) as usize) * m * m + (p[a] - 1) * m + (p[b] - 1)
            }
        }) as u32
    };
    let mut positions = vec![Vec3::ZERO; vertex_count];
    let mut faces: Vec<Face> = Vec::with_capacity(6 * n * n);
    // Each cube face: a fixed axis value and two free axes (a, b) in cyclic order, so `a x b` is the
    // fixed axis and a quad walked (0,0) -> (1,0) -> (1,1) faces +axis; the faces at 0 are reversed.
    for (fixed, at) in [(0usize, 0usize), (0, n), (1, 0), (1, n), (2, 0), (2, n)] {
        let flip = at == 0;
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
                    let v = id(c);
                    positions[v as usize] = (Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / n as f32 * 2.0 - Vec3::ONE).normalize() * radius;
                    v
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

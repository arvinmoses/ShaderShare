//! Hierarchical level of detail for very dense meshes.
//!
//! At tens of millions of triangles nearly every triangle covers less than a pixel, which is the worst
//! case for a GPU rasterizer. This module keeps a simplified copy of every patch of the mesh so the
//! renderer can draw roughly one triangle per pixel instead.
//!
//! The tree mirrors the sculpting BVH node for node. A leaf is a patch of ~2k quads at full detail. An
//! inner node is the *simplified union of its two children*, reduced to about [`LodParams::node_tris`]
//! triangles. Three properties make it fit an editable mesh:
//!
//! * **Shared vertices.** A simplified patch only references vertices that already exist in the mesh's
//!   vertex buffer. When a brush moves a vertex, every level that uses it moves too, with no extra upload.
//!   Normals come from the full-resolution surface, so shading does not pop between levels.
//! * **Crack-free.** Each simplification locks the border of the patch, so a patch always keeps every
//!   original vertex along its outer boundary. Two neighbouring patches at *different* levels therefore
//!   share exactly the same boundary edges, and no cut through the tree can open a gap.
//! * **Cheap selection.** Picking what to draw is a walk over a few thousand nodes: cull against the
//!   view frustum, and stop descending where the node's simplification error projects below a pixel
//!   threshold. The cost of a frame follows the number of pixels, not the number of triangles.
//!
//! The error stored per node is an upper bound at build time. Edits move retained vertices live, so
//! drawing stays faithful; only the shape of the collapsed-away detail drifts until a rebuild.

use glam::{Vec3, Vec4};
use meshopt::{SimplifyOptions, VertexDataAdapter};
use rayon::prelude::*;

use crate::bvh::{Bvh, NodeKind};
use crate::geom::Aabb;
use crate::mesh::{Face, face_triangles};

#[derive(Clone, Copy, Debug)]
pub struct LodParams {
    /// Triangles an inner node is simplified to.
    pub node_tris: usize,
}

impl Default for LodParams {
    fn default() -> Self {
        LodParams { node_tris: 4096 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LodNode {
    /// First index of this node's triangles in [`LodTree::indices`].
    pub first: u32,
    /// Number of indices (three per triangle).
    pub count: u32,
    /// Largest distance, in world units, between this node's surface and the full-resolution one.
    pub error: f32,
    /// BVH children, mirroring [`NodeKind::Inner`]; `None` for a leaf.
    pub children: Option<[u32; 2]>,
}

/// How the camera sees the mesh, in the terms selection needs.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: Vec3,
    /// Frustum planes, `xyz` pointing inward, so `dot(n, p) + w >= 0` means inside.
    pub planes: [Vec4; 6],
    /// Pixels spanned by one world unit at distance one (`height / (2 tan(fov/2))`).
    pub focal_px: f32,
}

impl View {
    /// Planes from a column-major view-projection matrix with a 0..1 depth range (Gribb/Hartmann).
    pub fn from_view_proj(m: glam::Mat4, eye: Vec3, focal_px: f32) -> View {
        let r = [m.row(0), m.row(1), m.row(2), m.row(3)];
        let raw = [r[3] + r[0], r[3] - r[0], r[3] + r[1], r[3] - r[1], r[2], r[3] - r[2]];
        let planes = raw.map(|p| {
            let n = p.truncate().length().max(1e-12);
            p / n
        });
        View { eye, planes, focal_px }
    }

    fn sees(&self, b: &Aabb) -> bool {
        let c = b.center();
        let h = b.extent() * 0.5;
        self.planes.iter().all(|p| {
            let r = h.x * p.x.abs() + h.y * p.y.abs() + h.z * p.z.abs();
            p.truncate().dot(c) + p.w + r >= 0.0
        })
    }

    /// Projected size, in pixels, of a world-space error `e` on the box nearest the eye.
    fn error_px(&self, b: &Aabb, e: f32) -> f32 {
        let d = (self.eye.clamp(b.min, b.max) - self.eye).length();
        e * self.focal_px / d.max(1e-4)
    }
}

/// What to draw this frame.
#[derive(Clone, Debug, Default)]
pub struct Cut {
    /// `(first_index, index_count)` into the pool, with neighbouring ranges already merged.
    pub ranges: Vec<(u32, u32)>,
    pub triangles: u64,
    pub nodes: u32,
    /// Nodes rejected by the frustum.
    pub culled: u32,
    /// The pixel threshold actually used (raised when the triangle budget forced it).
    pub tau_px: f32,
}

pub struct LodTree {
    pub nodes: Vec<LodNode>,
    /// Leaves' full-resolution triangles first, then every inner node's simplified triangles.
    pub indices: Vec<u32>,
    pub full_triangles: u64,
    pub params: LodParams,
}

/// Result of building one subtree: the node's own simplified triangles and error, plus the finished
/// results of every *inner* node below it (and itself).
struct Subtree {
    indices: Vec<u32>,
    error: f32,
    inner: Vec<(u32, Vec<u32>, f32)>,
}

impl LodTree {
    /// Builds the tree. `positions` and `faces` are in the engine's internal order, matching `bvh`.
    pub fn build(positions: &[Vec3], faces: &[Face], bvh: &Bvh, params: LodParams) -> LodTree {
        // Leaf triangles, in leaf order, are the start of the pool.
        let leaf_tris: Vec<Vec<u32>> = bvh
            .leaves
            .par_iter()
            .map(|leaf| {
                let mut t = Vec::with_capacity(leaf.faces.len() * 6);
                for &f in &leaf.faces {
                    for tri in face_triangles(&faces[f as usize]) {
                        t.extend_from_slice(&tri);
                    }
                }
                t
            })
            .collect();
        let full_triangles = leaf_tris.iter().map(|t| t.len() as u64 / 3).sum();

        let bytes: &[u8] = bytemuck::cast_slice(positions);
        let adapter = VertexDataAdapter::new(bytes, 12, 0).expect("positions are tightly packed f32x3");
        let root = Self::build_node(0, bvh, &leaf_tris, &adapter, params);

        // Lay the pool out: leaves first, so a cut made only of leaves is the plain mesh.
        let mut indices: Vec<u32> = Vec::with_capacity(leaf_tris.iter().map(Vec::len).sum::<usize>() + root.inner.iter().map(|(_, i, _)| i.len()).sum::<usize>());
        let mut nodes = vec![LodNode { first: 0, count: 0, error: 0.0, children: None }; bvh.nodes.len()];
        for (i, n) in bvh.nodes.iter().enumerate() {
            if let NodeKind::Leaf { leaf } = n.kind {
                let t = &leaf_tris[leaf as usize];
                nodes[i] = LodNode { first: indices.len() as u32, count: t.len() as u32, error: 0.0, children: None };
                indices.extend_from_slice(t);
            }
        }
        for (id, idx, error) in root.inner {
            let NodeKind::Inner { left, right } = bvh.nodes[id as usize].kind else { unreachable!("only inner nodes are listed") };
            nodes[id as usize] = LodNode { first: indices.len() as u32, count: idx.len() as u32, error, children: Some([left, right]) };
            indices.extend_from_slice(&idx);
        }
        LodTree { nodes, indices, full_triangles, params }
    }

    fn build_node(id: u32, bvh: &Bvh, leaf_tris: &[Vec<u32>], adapter: &VertexDataAdapter<'_>, params: LodParams) -> Subtree {
        match bvh.nodes[id as usize].kind {
            NodeKind::Leaf { leaf } => Subtree { indices: leaf_tris[leaf as usize].clone(), error: 0.0, inner: Vec::new() },
            NodeKind::Inner { left, right } => {
                // The two halves are independent, so rayon splits the work down the tree.
                let (l, r) = rayon::join(|| Self::build_node(left, bvh, leaf_tris, adapter, params), || Self::build_node(right, bvh, leaf_tris, adapter, params));
                let child_error = l.error.max(r.error);
                let mut inner = l.inner;
                inner.extend(r.inner);
                let mut merged = l.indices;
                merged.extend_from_slice(&r.indices);
                let target = params.node_tris * 3;
                let mut own = 0.0f32;
                let indices = if merged.len() > target {
                    let opts = SimplifyOptions::LockBorder | SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute;
                    meshopt::simplify(&merged, adapter, target, f32::MAX, opts, Some(&mut own))
                } else {
                    merged
                };
                // A parent is never more accurate than its worst child, so selection stays consistent.
                let error = child_error + own;
                inner.push((id, indices.clone(), error));
                Subtree { indices, error, inner }
            }
        }
    }

    /// Choose what to draw: nodes the frustum can see, descending until a node's error is below `tau_px`.
    /// If the result would exceed `budget_tris`, the threshold is raised until it fits (a frame-time guard).
    pub fn select(&self, bvh: &Bvh, view: &View, tau_px: f32, budget_tris: u64) -> Cut {
        let mut tau = tau_px.max(0.01);
        loop {
            let mut cut = Cut { tau_px: tau, ..Default::default() };
            self.visit(0, bvh, view, tau, &mut cut);
            if cut.triangles <= budget_tris || tau > 1e4 {
                return cut;
            }
            tau *= 1.4;
        }
    }

    fn visit(&self, id: u32, bvh: &Bvh, view: &View, tau: f32, cut: &mut Cut) {
        let bounds = &bvh.nodes[id as usize].bounds;
        if bounds.is_empty() || !view.sees(bounds) {
            cut.culled += 1;
            return;
        }
        let node = &self.nodes[id as usize];
        match node.children {
            Some([l, r]) if view.error_px(bounds, node.error) > tau => {
                self.visit(l, bvh, view, tau, cut);
                self.visit(r, bvh, view, tau, cut);
            }
            _ => {
                cut.nodes += 1;
                cut.triangles += node.count as u64 / 3;
                match cut.ranges.last_mut() {
                    Some((first, count)) if *first + *count == node.first => *count += node.count,
                    _ => cut.ranges.push((node.first, node.count)),
                }
            }
        }
    }

    /// Triangles in the pool across all levels.
    pub fn pool_triangles(&self) -> u64 {
        self.indices.len() as u64 / 3
    }
}

//! Mesh decimation in the manner of a "decimation master": analyse a triangle mesh once, then take it down to any
//! triangle count live.
//!
//! [`Analysis::build`] collapses edges cheapest first all the way to a floor and records every collapse
//! (a progressive mesh). The record is laid out so that at any step the faces still alive are a prefix of one face
//! order and the vertices still alive a prefix of one vertex order, so [`Analysis::extract`] at any target is a pass
//! over the output only, in parallel: no collapse is run again when the count slider moves.
//!
//! Evenness `e` in 0..=1 is a gradient, not a switch. The collapse cost is the geometric blend
//! `qem^(1-e) * len^e` of the quadric error (mean squared distance to the planes the collapse merges) and the edge's
//! squared length, and the new point minimises `Q(x)/A + w(e)|x - mid|^2`. At 0 that is pure quadric error metrics
//! (dense where the shape bends, coarse where it is flat); at 1 the shortest edge goes first, to its midpoint, which
//! evens the edges out. [`even_out`] then flips edges toward valence 6 and relaxes the output tangentially, projected
//! back onto the original [`Surface`], at strength `e`, live.
//!
//! Every collapse is guarded so the result stays manifold: link condition, no tetrahedron collapse, no flipped or
//! folded face, no sliver, border vertices only slide along the border, and capped valence. [`check_manifold`]
//! reports what the output is.
//!
//! The analysis runs in rounds over spatial tiles in parallel (tile seams locked, the grid shifted every round),
//! with each round's collapses merged by cost into the global order, then one last pass over the whole mesh.

mod analyze;
mod bake;
mod collapse;
mod even;
mod manifold;
mod metric;
mod quadric;
mod replay;
mod surface;

pub use analyze::{Analysis, AnalyzeOptions, Progress};
pub use bake::{Baker, Maps, Tessellated, UvMesh, quick_uv, tessellate, vertex_normals};
pub use even::{even_out, relax};
pub use manifold::{ManifoldReport, check_manifold};
pub use metric::{SurfaceError, surface_error};
pub use replay::Decimated;
pub use surface::{Hit, Surface};

use glam::Vec3;

use crate::mesh::{PolyMesh, face_triangles};

/// A triangle mesh: positions and counter-clockwise triangles.
#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    /// Triangles of a polygon mesh (quads split on their 0-2 diagonal, as [`face_triangles`]).
    pub fn from_poly(m: &PolyMesh) -> TriMesh {
        let mut triangles = Vec::with_capacity(m.faces.len() * 2);
        for f in &m.faces {
            triangles.extend(face_triangles(f));
        }
        TriMesh { positions: m.positions.clone(), triangles }
    }

    pub fn to_poly(&self) -> PolyMesh {
        PolyMesh {
            positions: self.positions.clone(),
            faces: self.triangles.iter().map(|t| [t[0], t[1], t[2], crate::mesh::NO_VERT]).collect(),
        }
    }
}

/// How a triangle target is given: a count, or a percentage of the input to keep (Decimation Master's "% of
/// decimation" is `100 - keep`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Triangles(usize),
    KeepPercent(f32),
}

impl Target {
    pub fn triangles(self, input: usize) -> usize {
        match self {
            Target::Triangles(n) => n,
            Target::KeepPercent(p) => ((input as f64) * (p.clamp(0.0, 100.0) as f64) / 100.0).round() as usize,
        }
    }
}

//! Sculpting engine core. The form is a volume ([`generator::level_volume`], a distance field at several resolutions)
//! with the detail and generator stack over it ([`generator::detail_layer`]), and mesh layers ride on its surface
//! ([`mesh_layers`]: per-vertex offsets on a mesh taken from the volume, with masks). The mesh document under those
//! ([`document`], [`layers`], [`mask`], [`brush`]) also carries the older mesh-sculpting path that sculpt-cli and the
//! tests still use. Headless by design; UI and GPU rendering live in separate crates on top of this one.

pub mod bake;
pub mod brush;
pub mod bvh;
pub mod document;
pub mod generator;
pub mod geom;
mod height_grid;
pub mod io;
pub mod layer_ops;
pub mod layers;
pub mod lod;
pub mod mask;
pub mod mesh;
pub mod mesh_layers;
pub mod noise;
pub mod pose;
pub mod preview;
pub mod primitives;
pub mod subdiv;
pub mod surface_noise;
mod undo;

pub use document::{DirtySet, Displacements, Document, PaintTarget, SurfaceHit};
pub use glam;
pub use layer_ops::{Placement, TreeRow};
pub use preview::Preview;
pub use layers::{LayerBlend, LayerId, LayerKind, LayerMeta, SculptLayer};

/// Run the calling thread below normal priority (background work: it takes what the frame leaves).
pub fn lower_priority() {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        // THREAD_PRIORITY_BELOW_NORMAL
        unsafe {
            SetThreadPriority(GetCurrentThread(), -1);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("missing channel: {0}")]
    MissingChannel(String),
    #[error("no layer with id {0}")]
    NoSuchLayer(u32),
    #[error("layer '{0}' is locked")]
    LayerLocked(String),
    #[error("layer '{0}' is hidden")]
    LayerHidden(String),
    #[error("file format: {0}")]
    Format(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

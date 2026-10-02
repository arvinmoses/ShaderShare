//! Sculpting engine core: dense-mesh storage, Mudbox-style layers,
//! Substance-style mask stacks, ZBrush-feel brushes and posing, and a
//! data-driven project format. Headless by design; UI and GPU rendering
//! live in separate crates on top of this one.

pub mod bake;
pub mod brush;
pub mod bvh;
pub mod document;
pub mod geom;
pub mod io;
pub mod layers;
pub mod mask;
pub mod mesh;
pub mod noise;
pub mod pose;
pub mod primitives;
pub mod subdiv;
mod undo;

pub use document::{Displacements, Document, PaintTarget, SurfaceHit};
pub use glam;
pub use layers::{LayerId, SculptLayer};

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

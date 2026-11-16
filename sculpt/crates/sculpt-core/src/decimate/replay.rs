//! What [`super::Analysis::extract`] gives.

use super::TriMesh;

/// The mesh after some number of collapses.
#[derive(Clone, Debug, Default)]
pub struct Decimated {
    pub mesh: TriMesh,
    /// Collapses replayed.
    pub step: usize,
}

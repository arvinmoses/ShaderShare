//! Tiny previews of a layer's delta and mask, for the layer list.
//!
//! A preview is a `res x res` grid seen from the front (looking down -Z at the rest pose). It is built
//! from a bounded number of vertices per spatial leaf, so cost scales with the leaf count, not the face
//! count, and it never touches the sculpt path.

use glam::Vec3;

use crate::document::Document;
use crate::layers::LayerId;

/// Values in `0..=1` per cell, plus which cells any sample landed in.
#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    pub res: usize,
    pub values: Vec<f32>,
    pub covered: Vec<bool>,
}

impl Document {
    /// Bumps whenever any geometry, layer data or mask changes. Caches compare it to know they are stale.
    pub fn edit_serial(&self) -> u64 {
        self.serial
    }

    /// Footprint of a layer's offsets: brighter where it moves the surface more. `None` for folders.
    pub fn delta_preview(&self, id: LayerId, res: usize) -> Option<Preview> {
        let layer = self.layer(id)?;
        if layer.is_folder() {
            return None;
        }
        let mut grid = Grid::new(self, res);
        for l in layer.allocated_leaves() {
            let leaf = &self.bvh.leaves[l as usize];
            let chunk = layer.chunks[l as usize].as_ref()?;
            for (k, v) in sample_indices(leaf.owned.start as usize, leaf.owned.end as usize) {
                let m = chunk[k - leaf.owned.start as usize].length();
                grid.splat(self.rest[v], m);
            }
        }
        Some(grid.finish(None))
    }

    /// The mask as grey values, or `None` when the layer has no (enabled) mask values to show.
    pub fn mask_preview(&self, id: LayerId, res: usize) -> Option<Preview> {
        let values = self.layer(id)?.mask_values()?;
        let mut grid = Grid::new(self, res);
        for leaf in &self.bvh.leaves {
            for (_, v) in sample_indices(leaf.owned.start as usize, leaf.owned.end as usize) {
                grid.splat(self.rest[v], values[v]);
            }
        }
        Some(grid.finish(Some(1.0)))
    }
}

/// Up to `PER_LEAF` evenly spaced vertex indices of `start..end`.
fn sample_indices(start: usize, end: usize) -> impl Iterator<Item = (usize, usize)> {
    let n = end.saturating_sub(start);
    let take = n.min(PER_LEAF);
    (0..take).map(move |i| {
        let v = start + if take <= 1 { 0 } else { i * (n - 1) / (take - 1) };
        (v, v)
    })
}

/// Samples per spatial leaf. A brush dab covers far more vertices than the stride skips.
const PER_LEAF: usize = 96;

struct Grid {
    res: usize,
    min: Vec3,
    scale: Vec3,
    sum: Vec<f32>,
    count: Vec<u32>,
}

impl Grid {
    fn new(doc: &Document, res: usize) -> Grid {
        let b = doc.bounds();
        let ext = b.extent().max(Vec3::splat(1e-9));
        let side = ext.x.max(ext.y);
        Grid { res, min: b.min - Vec3::new((side - ext.x) / 2.0, (side - ext.y) / 2.0, 0.0), scale: Vec3::splat(res as f32 / side), sum: vec![0.0; res * res], count: vec![0; res * res] }
    }

    /// Add `value` at the cell `p` falls in and its neighbours, so sparse samples still read as a shape.
    fn splat(&mut self, p: Vec3, value: f32) {
        let x = ((p.x - self.min.x) * self.scale.x) as isize;
        let y = ((p.y - self.min.y) * self.scale.y) as isize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (x + dx, y + dy);
                if cx < 0 || cy < 0 || cx >= self.res as isize || cy >= self.res as isize {
                    continue;
                }
                // The centre cell counts fully; neighbours count half so edges stay soft.
                let w = if dx == 0 && dy == 0 { 1.0 } else { 0.5 };
                let i = (self.res - 1 - cy as usize) * self.res + cx as usize;
                self.sum[i] += value * w;
                self.count[i] += if w == 1.0 { 2 } else { 1 };
            }
        }
    }

    /// Average per cell, then scale so the brightest cell is 1, or by `fixed` when given (masks are already 0..1).
    fn finish(self, fixed: Option<f32>) -> Preview {
        let covered: Vec<bool> = self.count.iter().map(|&c| c > 0).collect();
        let mut values: Vec<f32> = self.sum.iter().zip(&self.count).map(|(&s, &c)| if c == 0 { 0.0 } else { s / (c as f32 * 0.5) }).collect();
        let peak = fixed.unwrap_or_else(|| values.iter().cloned().fold(0.0, f32::max).max(1e-9));
        values.iter_mut().for_each(|v| *v = (*v / peak).clamp(0.0, 1.0));
        Preview { res: self.res, values, covered }
    }
}

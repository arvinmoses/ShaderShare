//! The volume playground: sculpting a solid that never stretches, to try the feel before the engine is built for real.
//!
//! A ball held as a sparse distance field (`sculpt_core::generator::volume`, the CPU engine from the spike). Clay adds a
//! soft blob of material where the brush passes, carving removes one; the surface is rebuilt only in the blocks a dab
//! changed. The viewport keeps one slot per block of surface in its GPU buffers and rewrites just the changed slots.
//! No undo, layers or saving: this is for judging the feel.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::volume::Volume;

/// Quads one block's slot can hold. More (only on very crinkled surface) are dropped and counted.
pub const QUADS_PER_SLOT: usize = 256;
/// Vertices per slot: one per cell.
pub const CELLS: usize = 512;

/// What the viewport has to rewrite.
#[derive(Default)]
pub struct VolumeUpload {
    /// Per slot: one position per cell (cells without surface hold zero).
    pub verts: Vec<(u32, Vec<Vec3>)>,
    /// Per slot: triangle indices (global vertex numbers, slot * 512 + cell).
    pub indices: Vec<(u32, Vec<u32>)>,
    /// Slots that no longer hold surface.
    pub cleared: Vec<u32>,
}

pub struct VolumeMode {
    pub vol: Volume,
    slot_of: HashMap<u32, u32>,
    free: Vec<u32>,
    /// One past the highest slot in use: the viewport draws slots below it.
    pub high: u32,
    pub capacity: u32,
    /// Quads dropped because a block had more than a slot holds.
    pub dropped: usize,
    /// Blocks holding surface.
    pub blocks: usize,
    /// Timing of the last frame's dabs, for the status line.
    pub last_ms: f64,
    pub last_dabs: usize,
    /// Set when the viewport must rebuild its buffers from scratch (first frame, or after growing).
    pub reset: bool,
}

impl VolumeMode {
    /// A ball of radius 1 at about `triangles` triangles.
    pub fn ball(triangles: f32) -> VolumeMode {
        let voxel = (4.0 * std::f32::consts::PI / (triangles / 2.0)).sqrt();
        let mut vol = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.8, voxel);
        vol.extract_all();
        let surface = vol.drain_changed().into_iter().filter(|&b| vol.surface_of(b).is_some()).count();
        let capacity = (surface * 3).max(4096) as u32;
        let mut mode = VolumeMode { vol, slot_of: HashMap::new(), free: (0..capacity).rev().collect(), high: 0, capacity, dropped: 0, blocks: 0, last_ms: 0.0, last_dabs: 0, reset: true };
        // Everything is new on the first upload.
        mode.vol.extract_all();
        mode
    }

    pub fn voxel(&self) -> f32 {
        self.vol.voxel
    }

    /// One dab at surface point `at` with outward `normal`: clay raises a lump about half the radius high, carving
    /// digs a groove as deep.
    pub fn dab(&mut self, at: Vec3, normal: Vec3, radius: f32, strength: f32, carve: bool) {
        let t = Instant::now();
        let depth = radius * (0.25 + 0.5 * strength.clamp(0.0, 1.0));
        let centre = if carve { at + normal * (radius - depth) } else { at - normal * (radius - depth) };
        self.vol.dab(centre, radius, 0.2 * radius, carve);
        self.last_ms += t.elapsed().as_secs_f64() * 1e3;
        self.last_dabs += 1;
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        self.vol.raycast(origin, dir)
    }

    /// Everything the viewport must rewrite since the last call.
    pub fn take_upload(&mut self) -> VolumeUpload {
        let changed = self.vol.drain_changed();
        let mut up = VolumeUpload::default();
        if changed.is_empty() {
            return up;
        }
        // Slots first (quads point at neighbours' slots), then the data.
        for &b in &changed {
            match (self.vol.surface_of(b).is_some(), self.slot_of.get(&b).copied()) {
                (true, None) => match self.free.pop() {
                    Some(s) => {
                        self.slot_of.insert(b, s);
                        self.high = self.high.max(s + 1);
                    }
                    None => self.dropped += 1,
                },
                (false, Some(s)) => {
                    self.slot_of.remove(&b);
                    self.free.push(s);
                    up.cleared.push(s);
                }
                _ => {}
            }
        }
        for &b in &changed {
            let (Some(&slot), Some((cell, verts, quads))) = (self.slot_of.get(&b), self.vol.surface_of(b)) else { continue };
            let positions: Vec<Vec3> = cell.iter().map(|&c| if c == u32::MAX { Vec3::ZERO } else { verts[c as usize] }).collect();
            up.verts.push((slot, positions));
            let mut idx = Vec::with_capacity(quads.len().min(QUADS_PER_SLOT) * 6);
            for q in quads {
                if idx.len() >= QUADS_PER_SLOT * 6 {
                    self.dropped += 1;
                    continue;
                }
                let Some(g) = q.iter().map(|&(blk, c)| self.slot_of.get(&blk).map(|&s| s * CELLS as u32 + c as u32)).collect::<Option<Vec<u32>>>() else { continue };
                idx.extend_from_slice(&[g[0], g[1], g[2], g[0], g[2], g[3]]);
            }
            up.indices.push((slot, idx));
        }
        self.blocks = self.slot_of.len();
        up
    }

    /// Bounds the surface can occupy (the lattice), which the viewport packs positions inside.
    pub fn bounds(&self) -> sculpt_core::geom::Aabb {
        let (min, max) = self.vol.bounds();
        sculpt_core::geom::Aabb { min, max }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After heavy clay and carving, every block of surface has a slot and every quad made it into one.
    #[test]
    fn every_quad_reaches_the_gpu_lists() {
        let mut mode = VolumeMode::ball(2e5);
        let _ = mode.take_upload();
        for i in 0..80 {
            let a = i as f32 * 0.08;
            let dir = Vec3::new(a.sin(), 0.7, a.cos()).normalize();
            mode.dab(dir, dir, 0.35, 0.7, i % 2 == 1);
            let _ = mode.take_upload();
        }
        // What the GPU lists hold, against what the volume says.
        let mut missing_slot = 0;
        let mut quads_volume = 0;
        for (&b, _) in mode.slot_of.iter() {
            if let Some((_, _, q)) = mode.vol.surface_of(b) {
                quads_volume += q.len();
                for quad in q {
                    if quad.iter().any(|(blk, _)| !mode.slot_of.contains_key(blk)) {
                        missing_slot += 1;
                    }
                }
            }
        }
        let mesh = mode.vol.mesh();
        eprintln!("dropped {}, quads pointing at a block without a slot {missing_slot}, quads in slotted blocks {quads_volume}, quads in the volume {}", mode.dropped, mesh.faces.len());
        assert_eq!(mode.dropped, 0);
        assert_eq!(missing_slot, 0);
        assert_eq!(quads_volume, mesh.faces.len());
        let open = sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces);
        let most = mode.slot_of.keys().filter_map(|&b| mode.vol.surface_of(b)).map(|(_, _, q)| q.len()).max().unwrap_or(0);
        eprintln!("non-manifold edges in the volume's surface: {open}; most quads in one block {most}");
    }
}

//! The volume playground: sculpting a solid that never stretches, to try the feel before the engine is built for real.
//!
//! A ball held as a sparse distance field (`sculpt_core::generator::volume`, the CPU engine from the spike). Clay adds a
//! soft blob of material where the brush passes, carving removes one; the surface is rebuilt only in the blocks a dab
//! changed. The viewport keeps one vertex slot per block of surface in its GPU buffers, and the block's quads in as many
//! fixed-size chunks as they need, each drawn by its own indirect entry; it rewrites just what changed. When the pools
//! run out they grow and everything is uploaded once more, so no surface is ever dropped.
//! No undo, layers or saving: this is for judging the feel.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::volume::Volume;

/// Quads in one chunk of the index buffer.
pub const QUADS_PER_CHUNK: usize = 128;
/// Vertices per slot: one per cell.
pub const CELLS: usize = 512;

/// What the viewport has to rewrite.
#[derive(Default)]
pub struct VolumeUpload {
    /// Per vertex slot: one position per cell (cells without surface hold zero).
    pub verts: Vec<(u32, Vec<Vec3>)>,
    /// Per chunk: triangle indices (global vertex numbers, slot * 512 + cell), at most a chunk's worth.
    pub chunks: Vec<(u32, Vec<u32>)>,
    /// Chunks that no longer hold anything.
    pub cleared: Vec<u32>,
}

/// A pool of numbered places that grows when it runs out.
struct Pool {
    free: Vec<u32>,
    capacity: u32,
    /// One past the highest place handed out.
    high: u32,
}

impl Pool {
    fn new(capacity: u32) -> Pool {
        Pool { free: (0..capacity).rev().collect(), capacity, high: 0 }
    }

    fn take(&mut self) -> Option<u32> {
        let s = self.free.pop()?;
        self.high = self.high.max(s + 1);
        Some(s)
    }

    /// Room for at least `more` places beyond what is free, with headroom.
    fn grow(&mut self, more: usize) {
        let next = ((self.capacity as usize + more) * 3 / 2) as u32;
        self.free.splice(0..0, (self.capacity..next).rev());
        self.capacity = next;
    }

    fn reset(&mut self) {
        *self = Pool::new(self.capacity);
    }
}

pub struct VolumeMode {
    pub vol: Volume,
    slot_of: HashMap<u32, u32>,
    chunks_of: HashMap<u32, Vec<u32>>,
    slots: Pool,
    chunks: Pool,
    /// Blocks holding surface.
    pub blocks: usize,
    /// Timing of the last frame's dabs, for the status line.
    pub last_ms: f64,
    pub last_dabs: usize,
    /// Set when the viewport must rebuild its buffers from scratch (first frame, or after a pool grew); the upload
    /// that sets it holds everything.
    pub reset: bool,
}

impl VolumeMode {
    /// A ball of radius 1 at about `triangles` triangles.
    pub fn ball(triangles: f32) -> VolumeMode {
        let voxel = (4.0 * std::f32::consts::PI / (triangles / 2.0)).sqrt();
        let mut vol = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.8, voxel);
        vol.extract_all();
        let surface = vol.drain_changed().into_iter().filter(|&b| vol.surface_of(b).is_some()).count();
        // Room for the surface to roughly double before the first growth.
        let cap = (surface * 2).max(4096) as u32;
        let mut mode = VolumeMode { vol, slot_of: HashMap::new(), chunks_of: HashMap::new(), slots: Pool::new(cap), chunks: Pool::new(cap), blocks: 0, last_ms: 0.0, last_dabs: 0, reset: true };
        // Everything is new on the first upload.
        mode.vol.extract_all();
        mode
    }

    pub fn voxel(&self) -> f32 {
        self.vol.voxel
    }

    /// Vertex slots and index chunks the viewport's buffers must hold.
    pub fn capacity(&self) -> (u32, u32) {
        (self.slots.capacity, self.chunks.capacity)
    }

    /// Indirect entries to draw: one past the highest chunk in use.
    pub fn draw_count(&self) -> u32 {
        self.chunks.high
    }

    /// One dab near surface point `at`: a flat disc on the surface the stroke began on. Clay raises it by up to a third
    /// of the radius at full strength, carving digs as deep; overlapping dabs in one stroke fill in rather than stack, so
    /// a stroke lays down (or cuts) one even layer, and the next stroke builds on it.
    pub fn dab(&mut self, at: Vec3, radius: f32, strength: f32, carve: bool) {
        let t = Instant::now();
        let on = self.vol.onto_base(at);
        let n = self.vol.normal_at(on);
        if n == Vec3::ZERO {
            return;
        }
        let height = radius * (0.04 + 0.3 * strength.clamp(0.0, 1.0));
        let grid = self.vol.grid();
        self.vol.dab_disc(on, n, radius, height, height, carve);
        if self.vol.grid() != grid {
            // The lattice grew and every block was renumbered: start the GPU side over.
            self.slot_of.clear();
            self.chunks_of.clear();
            self.slots.reset();
            self.chunks.reset();
            self.reset = true;
        }
        self.last_ms += t.elapsed().as_secs_f64() * 1e3;
        self.last_dabs += 1;
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        self.vol.raycast(origin, dir)
    }

    /// Everything the viewport must rewrite since the last call. If [`Self::reset`] is set afterwards, it is
    /// everything there is, for fresh buffers.
    pub fn take_upload(&mut self) -> VolumeUpload {
        let mut changed = self.vol.drain_changed();
        let mut up = VolumeUpload::default();
        if changed.is_empty() && !self.reset {
            return up;
        }
        // Vertex slots first (quads point at neighbours' slots).
        for &b in &changed {
            match (self.vol.surface_of(b).is_some(), self.slot_of.get(&b).copied()) {
                (true, None) => {
                    if self.slots.free.is_empty() {
                        self.slots.grow(1);
                        self.reset = true;
                    }
                    let s = self.slots.take().unwrap();
                    self.slot_of.insert(b, s);
                }
                (false, Some(s)) => {
                    self.slot_of.remove(&b);
                    self.slots.free.push(s);
                    for c in self.chunks_of.remove(&b).unwrap_or_default() {
                        self.chunks.free.push(c);
                        up.cleared.push(c);
                    }
                }
                _ => {}
            }
        }
        let per = QUADS_PER_CHUNK * 6;
        loop {
            if self.reset {
                // Fresh buffers: hand out every place again, in order, and upload every block.
                self.slots.reset();
                self.chunks.reset();
                self.chunks_of.clear();
                up = VolumeUpload::default();
                changed = self.slot_of.keys().copied().collect();
                changed.sort_unstable();
                for &b in &changed {
                    let s = self.slots.take().unwrap();
                    self.slot_of.insert(b, s);
                }
            }
            let lists: Vec<(u32, Vec<u32>)> = changed.iter().filter_map(|&b| Some((b, self.indices(b)?))).collect();
            // Chunks each block needs beyond those it has; spare ones go back first.
            let mut short = 0;
            for (b, idx) in &lists {
                let need = idx.len().div_ceil(per);
                let have = self.chunks_of.entry(*b).or_default();
                while have.len() > need {
                    let c = have.pop().unwrap();
                    self.chunks.free.push(c);
                    up.cleared.push(c);
                }
                short += need - have.len();
            }
            if short > self.chunks.free.len() {
                let more = short - self.chunks.free.len();
                self.chunks.grow(more);
                if !self.reset {
                    self.reset = true;
                    continue;
                }
            }
            for (b, idx) in lists {
                let have = self.chunks_of.get_mut(&b).unwrap();
                while have.len() < idx.len().div_ceil(per) {
                    have.push(self.chunks.take().unwrap());
                }
                for (i, part) in idx.chunks(per).enumerate() {
                    up.chunks.push((have[i], part.to_vec()));
                }
            }
            break;
        }
        for &b in &changed {
            let (Some(&slot), Some((cell, verts, _))) = (self.slot_of.get(&b), self.vol.surface_of(b)) else { continue };
            up.verts.push((slot, cell.iter().map(|&c| if c == u32::MAX { Vec3::ZERO } else { verts[c as usize] }).collect()));
        }
        self.blocks = self.slot_of.len();
        up
    }

    /// Block `b`'s triangles as global vertex numbers.
    fn indices(&self, b: u32) -> Option<Vec<u32>> {
        let (_, _, quads) = self.vol.surface_of(b)?;
        let mut idx = Vec::with_capacity(quads.len() * 6);
        for q in quads {
            // Every corner's block holds surface, so it has a slot.
            let Some(g) = q.iter().map(|(blk, c)| self.slot_of.get(blk).map(|&s| s * CELLS as u32 + *c as u32)).collect::<Option<Vec<u32>>>() else {
                debug_assert!(false, "a quad points at a block without surface");
                continue;
            };
            idx.extend_from_slice(&[g[0], g[1], g[2], g[0], g[2], g[3]]);
        }
        Some(idx)
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

    /// Big brushes along strokes, as in the app (the surface grows several times over, so the pools grow too): what
    /// the GPU buffers would draw is exactly the volume's surface.
    #[test]
    fn gpu_mirror_matches_the_volume() {
        let mut mode = VolumeMode::ball(1e6);
        let mut verts: HashMap<u32, Vec<Vec3>> = HashMap::new();
        let mut chunks: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut resets = 0;
        let mut apply = |mode: &mut VolumeMode| {
            let up = mode.take_upload();
            if mode.reset {
                verts.clear();
                chunks.clear();
                mode.reset = false;
                resets += 1;
            }
            for c in up.cleared {
                chunks.remove(&c);
            }
            for (s, p) in up.verts {
                verts.insert(s, p);
            }
            for (c, i) in up.chunks {
                assert!(i.len() <= QUADS_PER_CHUNK * 6 && c < mode.capacity().1);
                chunks.insert(c, i);
            }
            assert!(chunks.keys().all(|&c| c < mode.draw_count()));
        };
        apply(&mut mode);
        for stroke in 0..12 {
            let carve = stroke % 3 == 2;
            let radius = 0.25 + 0.05 * (stroke % 4) as f32;
            for i in 0..40 {
                let a = stroke as f32 * 0.9 + i as f32 * 0.03;
                let dir = Vec3::new(a.sin(), 0.3 * (stroke as f32).cos(), a.cos()).normalize();
                let Some((hit, n)) = mode.raycast(dir * 3.0, -dir) else { continue };
                let _ = n;
                mode.vol.begin_stroke();
                mode.dab(hit, radius, 0.8, carve);
                mode.vol.end_stroke();
                if i % 3 == 0 {
                    apply(&mut mode);
                }
            }
            apply(&mut mode);
        }
        // Drawn triangles against the volume's quads, by count and by where their corners are.
        let (mut drawn, mut sum, mut zero) = (0, Vec3::ZERO.as_dvec3(), 0);
        for i in chunks.values().flatten() {
            let p = verts[&(i / CELLS as u32)][(i % CELLS as u32) as usize];
            zero += (p == Vec3::ZERO) as usize;
            sum += p.as_dvec3();
            drawn += 1;
        }
        let mesh = mode.vol.mesh();
        let want: glam::DVec3 = mesh.faces.iter().flat_map(|f| [f[0], f[1], f[2], f[0], f[2], f[3]]).map(|v| mesh.positions[v as usize].as_dvec3()).sum();
        let open = sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces);
        eprintln!("drawn quads {}, volume quads {}, corners at zero {zero}, resets {resets}, non-manifold {open}, blocks {}, capacity {:?}", drawn / 6, mesh.faces.len(), mode.blocks, mode.capacity());
        assert_eq!(drawn / 6, mesh.faces.len());
        assert_eq!(zero, 0);
        assert!((sum - want).length() < 1e-3 * drawn as f64, "{sum} vs {want}");
    }
}

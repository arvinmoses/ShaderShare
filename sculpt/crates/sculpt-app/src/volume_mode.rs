//! The volume playground: sculpting a solid that never stretches, to try the feel before the engine is built for real.
//!
//! A ball held as a sparse distance field (`sculpt_core::generator::volume`, the CPU engine from the spike). Clay adds a
//! soft blob of material where the brush passes, carving removes one; the surface is rebuilt only in the blocks a dab
//! changed. The viewport keeps each block's vertices and quads in as many fixed-size chunks of its GPU buffers as they
//! need (a quad chunk drawn by its own indirect entry) and rewrites just what changed. When the pools run out they grow
//! and everything is uploaded once more, so no surface is ever dropped. Strokes can be undone, and the volume saved
//! and opened (`.svol`); layers and the procedural stack are not here yet.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::volume::Volume;

/// Quads in one chunk of the index buffer.
pub const QUADS_PER_CHUNK: usize = 128;
/// Vertices in one chunk of the vertex buffer.
pub const VERTS_PER_CHUNK: usize = 256;

/// What the viewport has to rewrite.
#[derive(Default)]
pub struct VolumeUpload {
    /// Per vertex chunk: its positions, at most a chunk's worth.
    pub verts: Vec<(u32, Vec<Vec3>)>,
    /// Per quad chunk: triangle indices (global vertex numbers, vertex chunk * 256 + place), at most a chunk's worth.
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
    /// Per block of surface: the vertex chunks and the quad chunks holding it.
    vchunks_of: HashMap<u32, Vec<u32>>,
    chunks_of: HashMap<u32, Vec<u32>>,
    vchunks: Pool,
    chunks: Pool,
    /// Blocks holding surface.
    pub blocks: usize,
    /// Timing of the last frame's dabs, for the status line.
    pub last_ms: f64,
    pub last_dabs: usize,
    /// The brush normal of the stroke's last dab, which the next one leans toward.
    stroke_n: Vec3,
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
        VolumeMode::new(vol)
    }

    /// The playground on `vol`, whose surface is already extracted.
    pub fn new(mut vol: Volume) -> VolumeMode {
        let surface = vol.drain_changed().into_iter().filter(|&b| vol.surface_of(b).is_some()).count();
        // Room for the surface to roughly double before the first growth.
        let cap = (surface * 2).max(4096) as u32;
        let mut mode = VolumeMode { vol, vchunks_of: HashMap::new(), chunks_of: HashMap::new(), vchunks: Pool::new(cap), chunks: Pool::new(cap), blocks: 0, last_ms: 0.0, last_dabs: 0, stroke_n: Vec3::ZERO, reset: true };
        // Everything is new on the first upload.
        mode.vol.extract_all();
        mode
    }

    pub fn voxel(&self) -> f32 {
        self.vol.voxel
    }

    /// Vertex chunks and quad chunks the viewport's buffers must hold.
    pub fn capacity(&self) -> (u32, u32) {
        (self.vchunks.capacity, self.chunks.capacity)
    }

    /// Indirect entries to draw: one past the highest chunk in use.
    pub fn draw_count(&self) -> u32 {
        self.chunks.high
    }

    /// One dab of Clay Buildup (or, with `carve`, Trim Dynamic) at surface point `at`: a plane brush whose plane
    /// follows the surface the stroke began on. Clay fills up to a plane a little above the surface under the brush
    /// (cavities first) and keeps building as you scrub; trim cuts down to a plane a little below it.
    pub fn dab(&mut self, at: Vec3, radius: f32, strength: f32, hardness: f32, carve: bool) {
        let t = Instant::now();
        let along = if self.stroke_n == Vec3::ZERO { self.vol.normal_at(at) } else { self.stroke_n };
        if along == Vec3::ZERO {
            return;
        }
        let s = strength.clamp(0.0, 1.0);
        let height = radius * if carve { 0.015 + 0.06 * s } else { 0.015 + 0.08 * s };
        let grid = self.vol.grid();
        let Some(n) = self.vol.plane_dab(at, along, radius, height, hardness, 0.1 + 0.3 * s, carve) else { return };
        // The original surface's plane is steady already; easing it a little more smooths the turn over sharp forms.
        self.stroke_n = if self.stroke_n == Vec3::ZERO { n } else { (self.stroke_n * 0.5 + n * 0.5).try_normalize().unwrap_or(n) };
        if self.vol.grid() != grid {
            // The lattice grew and every block was renumbered: start the GPU side over.
            self.restart_gpu();
        }
        self.last_ms += t.elapsed().as_secs_f64() * 1e3;
        self.last_dabs += 1;
    }

    pub fn begin_stroke(&mut self) {
        self.vol.begin_stroke();
        self.stroke_n = Vec3::ZERO;
    }

    /// Returns the specks the stroke left that were removed.
    pub fn end_stroke(&mut self) -> usize {
        self.stroke_n = Vec3::ZERO;
        self.vol.end_stroke()
    }

    pub fn undo(&mut self) -> bool {
        self.stroke_n = Vec3::ZERO;
        self.vol.undo()
    }

    pub fn redo(&mut self) -> bool {
        self.stroke_n = Vec3::ZERO;
        self.vol.redo()
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        self.vol.raycast(origin, dir)
    }

    /// Forget every GPU place: the next upload hands them out afresh and holds everything.
    fn restart_gpu(&mut self) {
        self.vchunks_of.clear();
        self.chunks_of.clear();
        self.vchunks.reset();
        self.chunks.reset();
        self.reset = true;
    }

    /// Everything the viewport must rewrite since the last call. If [`Self::reset`] is set afterwards, it is
    /// everything there is, for fresh buffers.
    pub fn take_upload(&mut self) -> VolumeUpload {
        let mut changed = self.vol.drain_changed();
        let mut up = VolumeUpload::default();
        if changed.is_empty() && !self.reset {
            return up;
        }
        let (vper, iper) = (VERTS_PER_CHUNK, QUADS_PER_CHUNK * 6);
        loop {
            if self.reset {
                // Fresh buffers: every block of surface is uploaded, into places handed out in order.
                let mut all: Vec<u32> = self.vchunks_of.keys().copied().chain(changed.iter().copied()).collect();
                all.sort_unstable();
                all.dedup();
                changed = all;
                self.vchunks_of.clear();
                self.chunks_of.clear();
                self.vchunks.reset();
                self.chunks.reset();
                up = VolumeUpload::default();
            }
            // Blocks that lost their surface give their places back.
            for &b in &changed {
                if self.vol.surface_of(b).is_none() {
                    self.vchunks.free.extend(self.vchunks_of.remove(&b).unwrap_or_default());
                    for c in self.chunks_of.remove(&b).unwrap_or_default() {
                        self.chunks.free.push(c);
                        up.cleared.push(c);
                    }
                }
            }
            // Vertex chunks first: quads point at their own block's vertices and their neighbours'.
            let vneed = |b: u32| self.vol.surface_of(b).map_or(0, |(v, _)| v.len().div_ceil(vper));
            let mut short = 0;
            for &b in &changed {
                let need = vneed(b);
                if need == 0 {
                    continue;
                }
                let have = self.vchunks_of.entry(b).or_default();
                while have.len() > need {
                    self.vchunks.free.push(have.pop().unwrap());
                }
                short += need - have.len();
            }
            if short > self.vchunks.free.len() {
                self.vchunks.grow(short - self.vchunks.free.len());
                if !self.reset {
                    self.reset = true;
                    continue;
                }
            }
            for &b in &changed {
                let need = vneed(b);
                if let Some(have) = self.vchunks_of.get_mut(&b) {
                    while have.len() < need {
                        have.push(self.vchunks.take().unwrap());
                    }
                }
            }
            // Then the quads, as global vertex numbers.
            let lists: Vec<(u32, Vec<u32>)> = changed.iter().filter_map(|&b| Some((b, self.indices(b)?))).collect();
            let mut short = 0;
            for (b, idx) in &lists {
                let need = idx.len().div_ceil(iper);
                let have = self.chunks_of.entry(*b).or_default();
                while have.len() > need {
                    let c = have.pop().unwrap();
                    self.chunks.free.push(c);
                    up.cleared.push(c);
                }
                short += need - have.len();
            }
            if short > self.chunks.free.len() {
                self.chunks.grow(short - self.chunks.free.len());
                if !self.reset {
                    self.reset = true;
                    continue;
                }
            }
            for (b, idx) in lists {
                let have = self.chunks_of.get_mut(&b).unwrap();
                while have.len() < idx.len().div_ceil(iper) {
                    have.push(self.chunks.take().unwrap());
                }
                for (i, part) in idx.chunks(iper).enumerate() {
                    up.chunks.push((have[i], part.to_vec()));
                }
            }
            break;
        }
        for &b in &changed {
            let (Some(have), Some((verts, _))) = (self.vchunks_of.get(&b), self.vol.surface_of(b)) else { continue };
            for (i, part) in verts.chunks(vper).enumerate() {
                up.verts.push((have[i], part.to_vec()));
            }
        }
        self.blocks = self.vchunks_of.len();
        up
    }

    /// Block `b`'s triangles as global vertex numbers.
    fn indices(&self, b: u32) -> Option<Vec<u32>> {
        let (_, quads) = self.vol.surface_of(b)?;
        let vper = VERTS_PER_CHUNK as u32;
        let mut idx = Vec::with_capacity(quads.len() * 6);
        for q in quads {
            // Every corner's block holds surface, so it has vertex chunks.
            let place = |blk: u32, corner: u16| {
                let v = self.vol.vertex_of(blk, corner)? as u32;
                self.vchunks_of.get(&blk)?.get((v / vper) as usize).map(|&c| c * vper + v % vper)
            };
            let Some(g) = q.iter().map(|&(blk, corner)| place(blk, corner)).collect::<Option<Vec<u32>>>() else {
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
        let mut gpu = Mirror::default();
        gpu.apply(&mut mode);
        for stroke in 0..12 {
            let carve = stroke % 3 == 2;
            let radius = 0.25 + 0.05 * (stroke % 4) as f32;
            for i in 0..40 {
                let a = stroke as f32 * 0.9 + i as f32 * 0.03;
                let dir = Vec3::new(a.sin(), 0.3 * (stroke as f32).cos(), a.cos()).normalize();
                let Some((hit, n)) = mode.raycast(dir * 3.0, -dir) else { continue };
                let _ = n;
                mode.begin_stroke();
                mode.dab(hit, radius, 0.8, 0.5, carve);
                mode.end_stroke();
                if i % 3 == 0 {
                    gpu.apply(&mut mode);
                }
            }
            gpu.apply(&mut mode);
        }
        gpu.check(&mode, "after the strokes");
        // Undo many strokes, then redo some: the GPU copy follows.
        for _ in 0..150 {
            mode.undo();
        }
        gpu.apply(&mut mode);
        gpu.check(&mode, "after undoing 150");
        for _ in 0..60 {
            mode.redo();
        }
        gpu.apply(&mut mode);
        gpu.check(&mode, "after redoing 60");
    }

    /// What the viewport's buffers would hold, applied from the uploads.
    #[derive(Default)]
    struct Mirror {
        verts: HashMap<u32, Vec<Vec3>>,
        chunks: HashMap<u32, Vec<u32>>,
        resets: usize,
    }

    impl Mirror {
        fn apply(&mut self, mode: &mut VolumeMode) {
            let up = mode.take_upload();
            if mode.reset {
                self.verts.clear();
                self.chunks.clear();
                mode.reset = false;
                self.resets += 1;
            }
            for c in up.cleared {
                self.chunks.remove(&c);
            }
            for (s, p) in up.verts {
                self.verts.insert(s, p);
            }
            for (c, i) in up.chunks {
                assert!(i.len() <= QUADS_PER_CHUNK * 6 && c < mode.capacity().1);
                self.chunks.insert(c, i);
            }
            assert!(self.chunks.keys().all(|&c| c < mode.draw_count()));
        }

        /// Drawn triangles against the volume's quads, by count and by where their corners are.
        fn check(&self, mode: &VolumeMode, when: &str) {
            let (verts, chunks, resets) = (&self.verts, &self.chunks, self.resets);
            let (mut drawn, mut sum, mut zero) = (0, Vec3::ZERO.as_dvec3(), 0);
            for i in chunks.values().flatten() {
                let p = verts[&(i / VERTS_PER_CHUNK as u32)][(i % VERTS_PER_CHUNK as u32) as usize];
                zero += (p == Vec3::ZERO) as usize;
                sum += p.as_dvec3();
                drawn += 1;
            }
            let mesh = mode.vol.mesh();
            let want: glam::DVec3 = mesh.faces.iter().flat_map(|f| [f[0], f[1], f[2], f[0], f[2], f[3]]).map(|v| mesh.positions[v as usize].as_dvec3()).sum();
            let open = sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces);
            eprintln!("{when}: drawn quads {}, volume quads {}, corners at zero {zero}, resets {resets}, non-manifold {open}, blocks {}, capacity {:?}", drawn / 6, mesh.faces.len(), mode.blocks, mode.capacity());
            assert_eq!(drawn / 6, mesh.faces.len(), "{when}");
            assert_eq!(zero, 0, "{when}");
            assert_eq!(open, 0, "{when}");
            assert!((sum - want).length() < 1e-3 * drawn as f64, "{when}: {sum} vs {want}");
        }
    }

    /// Scrubbing a big brush back and forth over bumpy clay: the brush turns a little per dab, far less than the
    /// surface normal under it does.
    #[test]
    fn scrubbing_turns_the_brush_gently() {
        let mut mode = VolumeMode::ball(4e5);
        // Bumps: small strokes of clay scattered over the top.
        for i in 0..30 {
            let a = i as f32 * 2.4;
            let at = Vec3::new(0.35 * a.cos() * (i as f32 / 30.0), 0.35 * a.sin() * (i as f32 / 30.0), 1.0).normalize();
            mode.begin_stroke();
            if let Some((hit, _)) = mode.raycast(at * 3.0, -at) {
                // Hard-edged and dabbed over until each bump stands up.
                for _ in 0..10 {
                    mode.dab(hit, 0.08, 1.0, 0.9, false);
                }
            }
            mode.end_stroke();
        }
        let _ = mode.take_upload();
        // One scrub: back and forth across the bumps.
        let (mut turn, mut raw_turn, mut worst, mut n) = (0.0f32, 0.0f32, 0.0f32, 0);
        let (mut last, mut last_raw) = (Vec3::ZERO, Vec3::ZERO);
        mode.begin_stroke();
        for i in 0..240 {
            let x = 0.4 * ((i as f32) * 0.08).sin();
            let y = 0.05 * ((i as f32) * 0.31).sin();
            let at = Vec3::new(x, y, 1.0).normalize();
            let Some((hit, _)) = mode.raycast(at * 3.0, -at) else { continue };
            let raw = mode.vol.normal_at(hit);
            mode.dab(hit, 0.3, 0.7, 0.5, false);
            if last != Vec3::ZERO {
                let a = last.angle_between(mode.stroke_n).to_degrees();
                turn += a;
                worst = worst.max(a);
                raw_turn += last_raw.angle_between(raw).to_degrees();
                n += 1;
            }
            last = mode.stroke_n;
            last_raw = raw;
        }
        mode.end_stroke();
        let (mean, raw_mean) = (turn / n as f32, raw_turn / n as f32);
        eprintln!("{:.2} ms per dab of radius 0.3 at 400k triangles", mode.last_ms / mode.last_dabs as f64);
        eprintln!("per dab the brush turned {mean:.2} deg on average ({worst:.2} at most); the normal under it turned {raw_mean:.2}");
        assert!(mean < raw_mean / 3.0 && worst < 6.0);
    }
}

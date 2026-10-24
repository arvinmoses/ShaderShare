//! Surface noise: noise over the whole surface, along its normal, drawn by the GPU as the surface is shown (moving the
//! vertices by the octaves the surface's own lattice holds, and shading every pixel by all of them) and the same here,
//! to the bit's worth of float arithmetic, so it can be baked into a layer as it was seen. [`WGSL`] is the shader's copy
//! of these functions; the two are kept in step by hand (`tests/surface_noise.rs` checks them against each other).
//!
//! Gradient noise from an integer hash (no permutation table, so the shader needs no buffer): at each lattice corner a
//! gradient from twelve, chosen by the hash, faded with the quintic.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// The shader's copy: `sn_height(p, octaves)` and what it calls, reading `SN` (see [`SurfaceNoise::uniform`]).
pub const WGSL: &str = include_str!("surface_noise.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceNoise {
    pub on: bool,
    /// The most it moves the surface, in world units.
    pub amplitude: f32,
    /// Bumps per world unit, of its first octave.
    pub scale: f32,
    pub octaves: u32,
    /// 0 smooth, 1 ridged.
    pub kind: u32,
    pub seed: u32,
}

impl Default for SurfaceNoise {
    fn default() -> Self {
        SurfaceNoise { on: false, amplitude: 0.01, scale: 12.0, octaves: 5, kind: 0, seed: 1 }
    }
}

const GAIN: f32 = 0.5;
const LACUNARITY: f32 = 2.0;

#[inline]
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    crate::noise::hash3(x, y, z, seed)
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// The corner's gradient (one of twelve, by its hash) dotted with the offset `d` from it.
#[inline]
fn corner(h: u32, d: Vec3) -> f32 {
    let h = h % 12;
    let (a, b) = match h / 4 {
        0 => (d.x, d.y),
        1 => (d.x, d.z),
        _ => (d.y, d.z),
    };
    (if h & 1 == 0 { a } else { -a }) + (if h & 2 == 0 { b } else { -b })
}

/// Gradient noise at `p`, about -1 to 1.
pub fn gradient(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
    let c = |dx: i32, dy: i32, dz: i32| corner(hash3(x + dx, y + dy, z + dz, seed), f - Vec3::new(dx as f32, dy as f32, dz as f32));
    let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
    lerp(w, lerp(v, lerp(u, c(0, 0, 0), c(1, 0, 0)), lerp(u, c(0, 1, 0), c(1, 1, 0))), lerp(v, lerp(u, c(0, 0, 1), c(1, 0, 1)), lerp(u, c(0, 1, 1), c(1, 1, 1))))
}

impl SurfaceNoise {
    /// How far it moves the surface at `p` along its normal, from its first `octaves` (at most its own).
    pub fn height(&self, p: Vec3, octaves: u32) -> f32 {
        let n = octaves.min(self.octaves).max(1);
        let (mut sum, mut amp, mut freq) = (0.0f32, 1.0f32, self.scale);
        for o in 0..n {
            let g = gradient(p * freq + Vec3::splat(o as f32 * 17.13), self.seed.wrapping_add(o));
            sum += amp * if self.kind == 1 { (1.0 - g.abs()) * (1.0 - g.abs()) } else { g };
            amp *= GAIN;
            freq *= LACUNARITY;
        }
        // Over the octaves the whole would have (so leaving fine ones out lowers nothing else), spread to about -1 to 1.
        let full: f32 = (0..self.octaves.max(1)).map(|o| GAIN.powi(o as i32)).sum();
        let s = sum / full;
        let b = if self.kind == 1 { (s - 0.45) * 2.2 } else { s * 2.2 };
        self.amplitude * b.clamp(-1.0, 1.0)
    }

    /// The shortest wavelength of octave `o`.
    pub fn wavelength(&self, o: u32) -> f32 {
        1.0 / (self.scale.max(1e-6) * LACUNARITY.powi(o as i32))
    }

    /// The octaves a lattice of step `voxel` holds: none shorter than `steps` of its steps.
    pub fn octaves_for(&self, voxel: f32, steps: f32) -> u32 {
        (0..self.octaves).take_while(|&o| self.wavelength(o) >= steps * voxel).count() as u32
    }

    /// The shader's parameters: (amplitude, scale, 0, 0) and (octaves, kind, seed, on).
    pub fn uniform(&self) -> ([f32; 4], [u32; 4]) {
        ([self.amplitude, self.scale, 0.0, 0.0], [self.octaves, self.kind, self.seed, self.on as u32])
    }
}

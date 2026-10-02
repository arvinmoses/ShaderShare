//! Seeded 3D procedural noise for world-space masks.
//!
//! Everything here is described by [`NoiseParams`], a plain serde struct, so
//! noise lives in project files as data and can be authored by other tools.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoiseKind {
    Perlin,
    #[default]
    Fbm,
    Ridged,
    Turbulence,
    /// Worley F1 distance.
    Cellular,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseParams {
    pub kind: NoiseKind,
    /// Frequency in cycles per world unit.
    pub scale: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub seed: u32,
    pub offset: Vec3,
}

impl Default for NoiseParams {
    fn default() -> Self {
        NoiseParams { kind: NoiseKind::Fbm, scale: 4.0, octaves: 5, lacunarity: 2.0, gain: 0.5, seed: 0, offset: Vec3::ZERO }
    }
}

pub struct Noise {
    params: NoiseParams,
    perm: [u8; 512],
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed.wrapping_mul(0x27d4_eb2d) ^ (x as u32).wrapping_mul(0x8da6_b343);
    h ^= (y as u32).wrapping_mul(0xd816_3841);
    h ^= (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn grad(h: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = h & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

impl Noise {
    pub fn new(params: &NoiseParams) -> Noise {
        let mut p: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut s = params.seed as u64 ^ 0x5EED;
        for i in (1..256).rev() {
            let j = (splitmix(&mut s) % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Noise { params: params.clone(), perm }
    }

    /// Improved Perlin noise, roughly in `[-1, 1]`.
    pub fn perlin(&self, p: Vec3) -> f32 {
        let pf = p.floor();
        let (xi, yi, zi) = ((pf.x as i32 & 255) as usize, (pf.y as i32 & 255) as usize, (pf.z as i32 & 255) as usize);
        let (x, y, z) = (p.x - pf.x, p.y - pf.y, p.z - pf.z);
        let (u, v, w) = (fade(x), fade(y), fade(z));
        let pm = &self.perm;
        let a = pm[xi] as usize + yi;
        let aa = pm[a] as usize + zi;
        let ab = pm[a + 1] as usize + zi;
        let b = pm[xi + 1] as usize + yi;
        let ba = pm[b] as usize + zi;
        let bb = pm[b + 1] as usize + zi;
        let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
        lerp(
            w,
            lerp(
                v,
                lerp(u, grad(pm[aa], x, y, z), grad(pm[ba], x - 1.0, y, z)),
                lerp(u, grad(pm[ab], x, y - 1.0, z), grad(pm[bb], x - 1.0, y - 1.0, z)),
            ),
            lerp(
                v,
                lerp(u, grad(pm[aa + 1], x, y, z - 1.0), grad(pm[ba + 1], x - 1.0, y, z - 1.0)),
                lerp(u, grad(pm[ab + 1], x, y - 1.0, z - 1.0), grad(pm[bb + 1], x - 1.0, y - 1.0, z - 1.0)),
            ),
        )
    }

    /// Distance to the nearest jittered feature point.
    pub fn cellular(&self, p: Vec3) -> f32 {
        let c = p.floor();
        let mut best = f32::MAX;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cx, cy, cz) = (c.x as i32 + dx, c.y as i32 + dy, c.z as i32 + dz);
                    let h = hash3(cx, cy, cz, self.params.seed);
                    let j = Vec3::new(
                        (h & 1023) as f32 / 1023.0,
                        ((h >> 10) & 1023) as f32 / 1023.0,
                        ((h >> 20) & 1023) as f32 / 1023.0,
                    );
                    let fp = Vec3::new(cx as f32, cy as f32, cz as f32) + j;
                    best = best.min(fp.distance_squared(p));
                }
            }
        }
        best.sqrt()
    }

    /// Sample in `[0, 1]`.
    pub fn sample(&self, p: Vec3) -> f32 {
        let pr = &self.params;
        let q = p * pr.scale + pr.offset;
        let octaves = pr.octaves.max(1);
        let v = match pr.kind {
            NoiseKind::Perlin => self.perlin(q) * 0.5 + 0.5,
            NoiseKind::Cellular => self.cellular(q).min(1.0),
            NoiseKind::Fbm | NoiseKind::Ridged | NoiseKind::Turbulence => {
                let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
                for o in 0..octaves {
                    let n = self.perlin(q * freq + Vec3::splat(o as f32 * 17.13));
                    sum += amp
                        * match pr.kind {
                            NoiseKind::Ridged => {
                                let r = 1.0 - n.abs();
                                r * r
                            }
                            NoiseKind::Turbulence => n.abs(),
                            _ => n,
                        };
                    norm += amp;
                    amp *= pr.gain;
                    freq *= pr.lacunarity;
                }
                let s = sum / norm;
                if pr.kind == NoiseKind::Fbm { s * 0.5 + 0.5 } else { s }
            }
        };
        v.clamp(0.0, 1.0)
    }
}

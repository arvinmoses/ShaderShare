//! Instance placement: mask-weighted surface sampling and sharp-edge sampling.
//!
//! Both samplers split into an expensive, cacheable build (CDF over faces /
//! edge detection + chains) and a cheap, deterministic `sample(n, seed)`, so
//! changing the instance count or seed in the UI only repeats the cheap part.

use glam::Vec3;
use rayon::prelude::*;

use crate::mesh::{Face, NO_VERT, Topology, face_len};

#[inline]
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform f32 in [0, 1) from a hash of (seed, a, b).
#[inline]
fn rnd(seed: u64, a: u64, b: u64) -> f32 {
    let h = splitmix(seed ^ splitmix(a.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ b));
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[inline]
fn tri_area(p: &[Vec3], a: u32, b: u32, c: u32) -> f32 {
    0.5 * (p[b as usize] - p[a as usize]).cross(p[c as usize] - p[a as usize]).length()
}

/// The two triangles of a face (second is None for a triangle face).
#[inline]
fn face_tris(f: &Face) -> [Option<[u32; 3]>; 2] {
    [Some([f[0], f[1], f[2]]), (f[3] != NO_VERT).then_some([f[0], f[2], f[3]])]
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    pub position: Vec3,
    /// Interpolated vertex normal (unit).
    pub normal: Vec3,
    pub face: u32,
    /// Triangle vertices (a face is split into [0,1,2] and [0,2,3]).
    pub verts: [u32; 3],
    /// Barycentric weights for `verts`.
    pub bary: [f32; 3],
}

/// Importance sampler over the surface: probability of a face is proportional to
/// `area * mean(mask over its vertices)`.
pub struct SurfaceSampler {
    /// Inclusive prefix sums of per-face weights.
    cdf: Vec<f64>,
    total: f64,
}

impl SurfaceSampler {
    /// Build the per-face CDF. `mask` is per vertex (None = uniform by area).
    pub fn new(positions: &[Vec3], faces: &[Face], mask: Option<&[f32]>) -> SurfaceSampler {
        let mut cdf: Vec<f64> = faces
            .par_iter()
            .map(|f| {
                let mut w = 0.0f64;
                for tri in face_tris(f).into_iter().flatten() {
                    let m = mask.map_or(1.0, |m| (m[tri[0] as usize] + m[tri[1] as usize] + m[tri[2] as usize]) / 3.0);
                    w += (tri_area(positions, tri[0], tri[1], tri[2]) * m.max(0.0)) as f64;
                }
                w
            })
            .collect();
        // Two-pass parallel inclusive scan.
        const CHUNK: usize = 1 << 16;
        let sums: Vec<f64> = cdf.par_chunks_mut(CHUNK).map(|c| {
            let mut s = 0.0;
            for x in c.iter_mut() {
                s += *x;
                *x = s;
            }
            s
        }).collect();
        let mut offs = vec![0.0f64; sums.len()];
        let mut run = 0.0;
        for (o, s) in offs.iter_mut().zip(&sums) {
            *o = run;
            run += s;
        }
        cdf.par_chunks_mut(CHUNK).zip(offs.par_iter()).for_each(|(c, &o)| c.iter_mut().for_each(|x| *x += o));
        SurfaceSampler { cdf, total: run }
    }

    pub fn total_weight(&self) -> f64 {
        self.total
    }

    pub fn memory_bytes(&self) -> usize {
        self.cdf.len() * 8
    }

    /// `n` stratified-jittered samples: sample `i` draws its CDF position from
    /// stratum `i` (even coverage, and spatially coherent order), then a point
    /// in the chosen triangle. With a mask, points are rejection-refined inside
    /// the face by the interpolated mask so face borders don't leak.
    pub fn sample(
        &self,
        positions: &[Vec3],
        normals: &[Vec3],
        faces: &[Face],
        mask: Option<&[f32]>,
        n: usize,
        seed: u64,
    ) -> Vec<SurfaceSample> {
        if self.total <= 0.0 || n == 0 {
            return Vec::new();
        }
        (0..n)
            .into_par_iter()
            .map(|i| {
                let u = ((i as f64 + rnd(seed, i as u64, 0) as f64) / n as f64) * self.total;
                let fi = self.cdf.partition_point(|&c| c <= u).min(faces.len() - 1);
                let f = &faces[fi];
                let lo = if fi == 0 { 0.0 } else { self.cdf[fi - 1] };
                let frac = ((u - lo) / (self.cdf[fi] - lo).max(1e-300)).clamp(0.0, 1.0);
                let tris = face_tris(f);
                // Choose triangle within the face by (masked) area.
                let w = |t: &[u32; 3]| {
                    tri_area(positions, t[0], t[1], t[2]) * mask.map_or(1.0, |m| (m[t[0] as usize] + m[t[1] as usize] + m[t[2] as usize]) / 3.0).max(0.0)
                };
                let tri = match tris[1] {
                    Some(t1) => {
                        let w0 = w(&tris[0].unwrap());
                        let w1 = w(&t1);
                        if (frac as f32) * (w0 + w1) < w0 { tris[0].unwrap() } else { t1 }
                    }
                    None => tris[0].unwrap(),
                };
                let mmax = mask.map_or(1.0, |m| m[tri[0] as usize].max(m[tri[1] as usize]).max(m[tri[2] as usize]));
                let mut bary = [1.0 / 3.0; 3];
                for k in 0..8u64 {
                    let (r1, r2) = (rnd(seed, i as u64, 1 + 2 * k), rnd(seed, i as u64, 2 + 2 * k));
                    let s = r1.sqrt();
                    bary = [1.0 - s, s * (1.0 - r2), s * r2];
                    match mask {
                        Some(m) if mmax > 0.0 => {
                            let mv = bary[0] * m[tri[0] as usize] + bary[1] * m[tri[1] as usize] + bary[2] * m[tri[2] as usize];
                            if rnd(seed, i as u64, 100 + k) * mmax <= mv {
                                break;
                            }
                        }
                        _ => break,
                    }
                }
                let pos = positions[tri[0] as usize] * bary[0] + positions[tri[1] as usize] * bary[1] + positions[tri[2] as usize] * bary[2];
                let nrm = (normals[tri[0] as usize] * bary[0] + normals[tri[1] as usize] * bary[1] + normals[tri[2] as usize] * bary[2])
                    .normalize_or(Vec3::Y);
                SurfaceSample { position: pos, normal: nrm, face: fi as u32, verts: tri, bary }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SharpEdge {
    pub a: u32,
    pub b: u32,
    pub faces: [u32; 2],
    pub length: f32,
    pub chain: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeSample {
    pub position: Vec3,
    /// Unit edge direction (from the lower to the higher vertex index).
    pub tangent: Vec3,
    /// Bisector of the two adjacent face normals (unit).
    pub normal: Vec3,
    /// Dihedral angle between the two faces, radians (0 = flat).
    pub dihedral: f32,
    /// Total length of the sharp-edge chain this point lies on.
    pub chain_length: f32,
    pub chain: u32,
    pub edge: u32,
}

pub struct EdgeSampler {
    pub edges: Vec<SharpEdge>,
    pub chain_lengths: Vec<f32>,
    face_normals: Vec<Vec3>,
    cdf: Vec<f64>,
}

impl EdgeSampler {
    /// Detect edges whose adjacent faces differ by more than `angle_deg` and
    /// group them into chains (runs through vertices with exactly two sharp
    /// edges). Boundary and non-manifold edges are ignored.
    pub fn detect(positions: &[Vec3], faces: &[Face], topo: &Topology, angle_deg: f32) -> EdgeSampler {
        let fnorm: Vec<Vec3> = faces
            .par_iter()
            .map(|f| crate::mesh::face_normal(positions, f).normalize_or(Vec3::Y))
            .collect();
        let cos_thr = angle_deg.to_radians().cos();
        let mut edges: Vec<SharpEdge> = (0..faces.len())
            .into_par_iter()
            .flat_map_iter(|fi| {
                let f = &faces[fi];
                let len = face_len(f);
                let mut found: smallvec::SmallVec<[SharpEdge; 1]> = smallvec::SmallVec::new();
                for k in 0..len {
                    let (a, b) = (f[k], f[(k + 1) % len]);
                    if a > b {
                        continue; // each edge is handled once, by its a < b traversal
                    }
                    let mut other = None;
                    let mut count = 0;
                    for &g in topo.vert_faces.row(a as usize) {
                        if g as usize != fi && faces[g as usize][..face_len(&faces[g as usize])].contains(&b) {
                            other = Some(g);
                            count += 1;
                        }
                    }
                    if let (Some(g), 1) = (other, count)
                        && fnorm[fi].dot(fnorm[g as usize]) < cos_thr
                    {
                        found.push(SharpEdge {
                            a,
                            b,
                            faces: [fi as u32, g],
                            length: positions[a as usize].distance(positions[b as usize]),
                            chain: 0,
                        });
                    }
                }
                found
            })
            .collect();
        // Chains: flood fill across degree-2 vertices.
        let mut inc: Vec<(u32, u32)> = edges.iter().enumerate().flat_map(|(i, e)| [(e.a, i as u32), (e.b, i as u32)]).collect();
        inc.par_sort_unstable();
        let row = |v: u32| {
            let s = inc.partition_point(|x| x.0 < v);
            let e = inc.partition_point(|x| x.0 <= v);
            &inc[s..e]
        };
        let mut chain_lengths: Vec<f32> = Vec::new();
        let mut seen = vec![false; edges.len()];
        let mut stack = Vec::new();
        for start in 0..edges.len() {
            if seen[start] {
                continue;
            }
            let id = chain_lengths.len() as u32;
            let mut total = 0.0f32;
            seen[start] = true;
            stack.push(start as u32);
            while let Some(e) = stack.pop() {
                edges[e as usize].chain = id;
                total += edges[e as usize].length;
                for v in [edges[e as usize].a, edges[e as usize].b] {
                    let r = row(v);
                    if r.len() == 2 {
                        for &(_, o) in r {
                            if !seen[o as usize] {
                                seen[o as usize] = true;
                                stack.push(o);
                            }
                        }
                    }
                }
            }
            chain_lengths.push(total);
        }
        let mut cdf = Vec::with_capacity(edges.len());
        let mut run = 0.0f64;
        for e in &edges {
            run += e.length as f64;
            cdf.push(run);
        }
        EdgeSampler { edges, chain_lengths, face_normals: fnorm, cdf }
    }

    pub fn total_length(&self) -> f64 {
        self.cdf.last().copied().unwrap_or(0.0)
    }

    /// `n` points evenly spaced by arc length over all sharp edges (the global
    /// phase comes from `seed`).
    pub fn sample(&self, positions: &[Vec3], n: usize, seed: u64) -> Vec<EdgeSample> {
        let total = self.total_length();
        if total <= 0.0 || n == 0 {
            return Vec::new();
        }
        let phase = rnd(seed, 0, 7) as f64;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let s = ((i as f64 + phase) / n as f64) * total;
                let ei = self.cdf.partition_point(|&c| c <= s).min(self.edges.len() - 1);
                let e = &self.edges[ei];
                let lo = if ei == 0 { 0.0 } else { self.cdf[ei - 1] };
                let t = ((s - lo) / e.length.max(1e-30) as f64).clamp(0.0, 1.0) as f32;
                let (pa, pb) = (positions[e.a as usize], positions[e.b as usize]);
                let n0 = self.face_normals[e.faces[0] as usize];
                let n1 = self.face_normals[e.faces[1] as usize];
                EdgeSample {
                    position: pa.lerp(pb, t),
                    tangent: (pb - pa).normalize_or(Vec3::X),
                    normal: (n0 + n1).normalize_or(n0),
                    dihedral: n0.dot(n1).clamp(-1.0, 1.0).acos(),
                    chain_length: self.chain_lengths[e.chain as usize],
                    chain: e.chain,
                    edge: ei as u32,
                }
            })
            .collect()
    }
}

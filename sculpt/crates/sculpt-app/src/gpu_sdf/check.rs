//! Correctness: GPU pipeline vs the CPU voxel code (`sculpt_core::generator::voxel` and
//! `InstanceSet`) on a small mesh, at every stage.
//!
//! 1. brick set equals the CPU enumeration;
//! 2. combined field equals `max(mesh_sd, -min instance_sd)` computed on the CPU at every
//!    lattice sample near the surface (reported in voxels);
//! 3. surface-nets vertices equal CPU surface nets run on the CPU field (same cell, same
//!    position), and the counts of owned vertices and sign-changing edges match exactly;
//! 4. the surface position error: |CPU boolean field| at every GPU vertex, in voxels;
//! 5. triangle winding agrees with the field gradient.

use std::collections::HashMap;

use glam::Vec3;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::voxel::{self, MeshSdf};
use sculpt_core::generator::InstanceSet;

use super::bench::{make_doc, make_instances, mean_edge};
use super::{Gpu, Kernels, MARGIN_VOXELS, MeshGpu, Op, Sdf, config_for, enumerate_bricks, unpack_key};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Default)]
pub struct CheckReport {
    pub op: String,
    pub faces: usize,
    pub instances: usize,
    pub voxel: f32,
    pub bricks_gpu: usize,
    pub bricks_cpu: usize,
    pub bricks_cpu_spike: usize,
    pub brick_sets_equal: bool,
    pub samples_compared: usize,
    pub field_max_diff_vox: f32,
    pub field_mean_diff_vox: f32,
    pub sign_mismatches: usize,
    pub verts_gpu: usize,
    pub verts_cpu: usize,
    pub vert_missing: usize,
    pub vert_max_diff_vox: f32,
    pub owned_gpu: u32,
    pub owned_cpu: usize,
    pub edges_gpu: u32,
    pub edges_cpu: usize,
    pub spike_net_vertices: usize,
    pub spike_equiv_vertices: usize,
    pub surf_err_mean_vox: f32,
    pub surf_err_p99_vox: f32,
    pub surf_err_max_vox: f32,
    pub winding_ok_frac: f32,
    pub tris: usize,
    pub overflow: u32,
}

impl CheckReport {
    pub fn passed(&self) -> bool {
        self.brick_sets_equal
            && self.overflow == 0
            && self.field_max_diff_vox < 0.05
            && self.sign_mismatches == 0
            && self.vert_missing == 0
            && self.vert_max_diff_vox < 0.05
            && self.owned_gpu as usize == self.owned_cpu
            && self.edges_gpu as usize == self.edges_cpu
            && self.surf_err_max_vox < 1.0
            && self.winding_ok_frac > 0.95
    }

    pub fn print(&self) {
        println!("correctness [{}]: {} faces, {} instances (all 7 primitives), voxel {:.2} x mean edge", self.op, self.faces, self.instances, self.voxel);
        let spike = |v: usize| if self.op.starts_with("Subtract") && !self.op.contains("smooth") { v.to_string() } else { "n/a (spike B measures hard subtract only)".into() };
        println!("  bricks: GPU {} / CPU enumeration {} (equal sets: {}) / CPU spike `measure` {} (it uses a 2-voxel band and no pad)", self.bricks_gpu, self.bricks_cpu, self.brick_sets_equal, spike(self.bricks_cpu_spike));
        println!(
            "  combined field vs CPU on {} near-surface samples: max {:.5} voxel, mean {:.6} voxel, sign mismatches {}",
            self.samples_compared, self.field_max_diff_vox, self.field_mean_diff_vox, self.sign_mismatches
        );
        println!(
            "  vertices: GPU {} (incl. pad duplicates) / CPU {}; missing {}; max position difference to CPU surface nets {:.5} voxel",
            self.verts_gpu, self.verts_cpu, self.vert_missing, self.vert_max_diff_vox
        );
        println!(
            "  owned vertices GPU {} / CPU {} (CPU spike `measure` counts {}; an independent recount of the spike's brick set gives {}); sign-changing owned edges GPU {} / CPU {}",
            self.owned_gpu,
            self.owned_cpu,
            spike(self.spike_net_vertices),
            self.spike_equiv_vertices,
            self.edges_gpu,
            self.edges_cpu
        );
        println!(
            "  surface position error |boolean SDF| at GPU vertices: mean {:.4}, p99 {:.4}, max {:.4} voxel; {} triangles, winding agrees with gradient for {:.2}%",
            self.surf_err_mean_vox,
            self.surf_err_p99_vox,
            self.surf_err_max_vox,
            self.tris,
            self.winding_ok_frac * 100.0
        );
        println!("  RESULT: {}", if self.passed() { "PASS" } else { "FAIL" });
    }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    -smin(-a, -b, k)
}

pub fn run(args: &[String]) -> Res<()> {
    let gpu = Gpu::new()?;
    println!("adapter: {}", gpu.describe());
    let level: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(5);
    let cases = [(Op::Subtract, 0.0, 1.0f32), (Op::Subtract, 0.0, 0.5), (Op::Union, 0.0, 0.5), (Op::Intersect, 0.0, 1.0), (Op::Subtract, 2.0, 0.5), (Op::Union, 2.0, 1.0)];
    for (op, smooth, vx) in cases {
        let r = check(&gpu, level, 60, vx, op, smooth)?;
        r.print();
        if !r.passed() {
            return Err("GPU/CPU mismatch".into());
        }
    }
    Ok(())
}

/// `level` 5 = 6144 faces. `vx` = voxel size in mesh edges.
pub fn check(gpu: &Gpu, level: usize, n: usize, vx: f32, op: Op, smooth_vox: f32) -> Res<CheckReport> {
    let doc = make_doc(level)?;
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let edge = mean_edge(&doc);
    let samples = SurfaceSampler::new(pos, faces, None).sample(pos, nrm, faces, None, n, 11);
    let insts = make_instances(&samples, edge, 3.0, 8.0, true);
    let voxel = vx * edge;
    let origin = doc.bounds().min - Vec3::splat(4.0 * 8.0 * voxel);
    let en = enumerate_bricks(&insts, origin, voxel, 1);

    let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
    let flat = msdf.flatten();
    let mesh = MeshGpu::upload(gpu, &flat, pos, msdf.pseudo_normals(), 0);
    let kernels = Kernels::new(gpu);
    let mut cfg = config_for(&en, origin, voxel, op, super::Keep::ALL, en.keys.len() as u32 + 16);
    cfg.smooth_k = smooth_vox * voxel;
    let k = cfg.smooth_k;
    let mut sdf = Sdf::new(gpu, &kernels, &mesh, cfg, insts.len());
    sdf.upload_instances(&insts);
    sdf.cold_build();
    let fr = sdf.frame();
    let dbg = sdf.read_debug();

    let set = InstanceSet::new(insts.clone(), MARGIN_VOXELS * voxel);
    let spike = if op == Op::Subtract && smooth_vox == 0.0 { voxel::measure(&msdf, &set, voxel, usize::MAX) } else { Default::default() };

    let mut r = CheckReport { op: format!("{op:?}{}", if smooth_vox > 0.0 { format!(", smooth {smooth_vox} voxels") } else { String::new() }), faces: doc.face_count(), instances: insts.len(), voxel: vx, overflow: fr.overflow, ..Default::default() };
    let mut gkeys = dbg.active_keys.clone();
    gkeys.sort_unstable();
    r.bricks_gpu = gkeys.len();
    r.bricks_cpu = en.keys.len();
    r.bricks_cpu_spike = spike.bricks;
    r.brick_sets_equal = gkeys == en.keys;
    r.spike_net_vertices = spike.net_vertices;

    // CPU boolean field at a world point, exactly the spike's combine.
    let cpu_field = |p: Vec3| -> f32 {
        let mesh_d = msdf.signed_distance(p, 1e3);
        let mut m = f32::INFINITY;
        for &i in set.candidates(p) {
            if set.inflated_aabb_contains(i, p) {
                m = m.min(set.sdf(i, p));
            }
        }
        let m = if !m.is_finite() && op == Op::Intersect { MARGIN_VOXELS * voxel } else { m };
        if !m.is_finite() {
            return mesh_d;
        }
        match op {
            Op::Subtract => smax(mesh_d, -m, k),
            Op::Union => smin(mesh_d, m, k),
            Op::Intersect => smax(mesh_d, m, k),
        }
    };
    let lattice = |key: u32, s: [i32; 3]| -> Vec3 {
        let b = unpack_key(key);
        let g = [b[0] * 8 + s[0], b[1] * 8 + s[1], b[2] * 8 + s[2]];
        origin + Vec3::new(g[0] as f32, g[1] as f32, g[2] as f32) * voxel
    };

    // --- 2. field comparison, and the CPU copy of every brick's field for the extraction checks.
    let mut cpu_fields: Vec<Vec<f32>> = Vec::new();
    let (mut max_d, mut sum_d, mut cnt) = (0.0f32, 0.0f64, 0usize);
    for (bi, &key) in dbg.active_keys.iter().enumerate() {
        let mut f = vec![0.0f32; super::SAMPLES];
        for (k, fk) in f.iter_mut().enumerate() {
            let s = [(k % 10) as i32 - 1, ((k / 10) % 10) as i32 - 1, (k / 100) as i32 - 1];
            *fk = cpu_field(lattice(key, s));
            let g = dbg.comb[bi * super::SAMPLES + k];
            if fk.abs().min(g.abs()) < 2.0 * voxel {
                let d = (*fk - g).abs() / voxel;
                max_d = max_d.max(d);
                sum_d += d as f64;
                cnt += 1;
                if (*fk < 0.0) != (g < 0.0) {
                    r.sign_mismatches += 1;
                }
            }
        }
        cpu_fields.push(f);
    }
    r.samples_compared = cnt;
    r.field_max_diff_vox = max_d;
    r.field_mean_diff_vox = (sum_d / cnt.max(1) as f64) as f32;

    // --- 3. CPU surface nets on the CPU field.
    const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    const EDGES: [[usize; 2]; 12] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    let at = |x: i32, y: i32, z: i32| (((z + 1) * 10 + (y + 1)) * 10 + (x + 1)) as usize;
    let mut cpu_verts: HashMap<[i32; 3], Vec3> = HashMap::new();
    let mut crossing = vec![[false; 729]; cpu_fields.len()];
    for (bi, &key) in dbg.active_keys.iter().enumerate() {
        let b = unpack_key(key);
        let f = &cpu_fields[bi];
        for cz in -1..8 {
            for cy in -1..8 {
                for cx in -1..8 {
                    let mut d = [0.0f32; 8];
                    let mut mask = 0u8;
                    for (i, c) in CORNERS.iter().enumerate() {
                        d[i] = f[at(cx + c[0] as i32, cy + c[1] as i32, cz + c[2] as i32)];
                        mask |= ((d[i] < 0.0) as u8) << i;
                    }
                    if mask == 0 || mask == 255 {
                        continue;
                    }
                    let (mut sum, mut cn) = (Vec3::ZERO, 0.0);
                    for e in EDGES {
                        let (a, bb) = (d[e[0]], d[e[1]]);
                        if (a < 0.0) != (bb < 0.0) {
                            let t = a / (a - bb);
                            let pa = Vec3::new(CORNERS[e[0]][0] as f32, CORNERS[e[0]][1] as f32, CORNERS[e[0]][2] as f32);
                            let pb = Vec3::new(CORNERS[e[1]][0] as f32, CORNERS[e[1]][1] as f32, CORNERS[e[1]][2] as f32);
                            sum += pa.lerp(pb, t);
                            cn += 1.0;
                        }
                    }
                    let g = [b[0] * 8 + cx, b[1] * 8 + cy, b[2] * 8 + cz];
                    let p = origin + (Vec3::new(g[0] as f32, g[1] as f32, g[2] as f32) + sum / cn) * voxel;
                    cpu_verts.insert(g, p);
                    crossing[bi][(((cz + 1) * 9 + (cy + 1)) * 9 + (cx + 1)) as usize] = true;
                    if cx >= 0 && cy >= 0 && cz >= 0 {
                        r.owned_cpu += 1;
                    }
                }
            }
        }
        // owned lattice edges with a sign change
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let s0 = f[at(x, y, z)] < 0.0;
                    for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        if (f[at(x + dx, y + dy, z + dz)] < 0.0) != s0 {
                            r.edges_cpu += 1;
                        }
                    }
                }
            }
        }
    }
    r.verts_cpu = cpu_verts.len();
    r.owned_gpu = fr.verts_owned;
    r.edges_gpu = fr.edges_owned;

    let nv = dbg.verts.len() / 8;
    r.verts_gpu = nv;
    let mut err = Vec::with_capacity(nv);
    let mut gpu_pos = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = Vec3::new(dbg.verts[v * 8], dbg.verts[v * 8 + 1], dbg.verts[v * 8 + 2]);
        gpu_pos.push(p);
        let c = ((p - origin) / voxel).floor().as_ivec3();
        match cpu_verts.get(&[c.x, c.y, c.z]) {
            Some(q) => r.vert_max_diff_vox = r.vert_max_diff_vox.max(p.distance(*q) / voxel),
            None => r.vert_missing += 1,
        }
        // --- 4. distance of the GPU vertex to the CPU boolean surface (|f| is a distance bound).
        err.push(cpu_field(p).abs() / voxel);
    }
    err.sort_by(|a, b| a.total_cmp(b));
    if !err.is_empty() {
        r.surf_err_mean_vox = err.iter().sum::<f32>() / err.len() as f32;
        r.surf_err_p99_vox = err[(err.len() as f32 * 0.99) as usize % err.len()];
        r.surf_err_max_vox = *err.last().unwrap();
    }

    // Cross-check the CPU spike's own count: re-count cells over the spike's brick set (raw AABB + 2 voxel band)
    // using the CPU field above.
    {
        let bsz = 8.0 * voxel;
        let mut keys: Vec<u32> = Vec::new();
        for inst in &insts {
            let a = inst.aabb();
            let lo = ((a.min - Vec3::splat(2.0 * voxel) - origin) / bsz).floor().as_ivec3();
            let hi = ((a.max + Vec3::splat(2.0 * voxel) - origin) / bsz).floor().as_ivec3();
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        keys.push(super::pack_key([x, y, z]));
                    }
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        let mut count = 0usize;
        for &key in &keys {
            let mut f = vec![0.0f32; 729];
            for z in 0..9 {
                for y in 0..9 {
                    for x in 0..9 {
                        f[(z * 9 + y) * 9 + x] = cpu_field(lattice(key, [x as i32, y as i32, z as i32]));
                    }
                }
            }
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let mut mask = 0u8;
                        for (i, c) in CORNERS.iter().enumerate() {
                            mask |= ((f[((z + c[2]) * 9 + y + c[1]) * 9 + x + c[0]] < 0.0) as u8) << i;
                        }
                        count += (mask != 0 && mask != 255) as usize;
                    }
                }
            }
        }
        r.spike_equiv_vertices = count;
    }

    // --- 5. winding.
    r.tris = dbg.idx.len() / 3;
    let mut ok = 0usize;
    for t in dbg.idx.as_chunks::<3>().0.iter() {
        let (a, b, c) = (gpu_pos[t[0] as usize], gpu_pos[t[1] as usize], gpu_pos[t[2] as usize]);
        let gn = (b - a).cross(c - a);
        let vn = Vec3::new(dbg.verts[t[0] as usize * 8 + 4], dbg.verts[t[0] as usize * 8 + 5], dbg.verts[t[0] as usize * 8 + 6]);
        if gn.dot(vn) >= 0.0 {
            ok += 1;
        }
    }
    r.winding_ok_frac = ok as f32 / r.tris.max(1) as f32;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_boolean_matches_cpu_voxel_code() {
        let Ok(gpu) = Gpu::new() else {
            eprintln!("no wgpu adapter: skipping GPU SDF test");
            return;
        };
        for (op, smooth, vx) in [(Op::Subtract, 0.0, 1.0), (Op::Subtract, 0.0, 0.5), (Op::Union, 0.0, 0.5), (Op::Intersect, 0.0, 1.0), (Op::Subtract, 2.0, 0.5)] {
            let r = check(&gpu, 4, 24, vx, op, smooth).expect("check");
            r.print();
            assert!(r.passed(), "GPU result differs from the CPU voxel code ({op:?}, smooth {smooth}, voxel {vx})");
        }
    }

    #[test]
    fn enumeration_stride_is_a_subset() {
        let doc = make_doc(3).unwrap();
        let edge = mean_edge(&doc);
        let sampler = SurfaceSampler::new(doc.positions(), doc.faces(), None);
        let s = sampler.sample(doc.positions(), doc.normals(), doc.faces(), None, 100, 3);
        let insts = make_instances(&s, edge, 3.0, 8.0, false);
        let origin = doc.bounds().min - Vec3::splat(1.0);
        let all = enumerate_bricks(&insts, origin, edge * 0.5, 1);
        let some = enumerate_bricks(&insts, origin, edge * 0.5, 4);
        assert_eq!(all.keys_all, some.keys_all);
        assert!(some.keys.len() < all.keys.len() && !some.keys.is_empty());
        assert!(some.keys.iter().all(|k| all.keys.binary_search(k).is_ok()));
    }
}

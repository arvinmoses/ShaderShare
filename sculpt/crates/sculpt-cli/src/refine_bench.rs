//! `sculpt-cli refine-bench`: measurements and images for the boundary refinement spike
//! (`sculpt-core/src/generator/refine.rs`, `docs/generator/spike3_refine.md`).
//!
//! ```text
//! (all: --crease=0.3 --wall=0.5 --levels=6 override the refinement targets)
//! refine-bench quality [faces=98000,1600000] [sizes=1.5,3,6]     rim accuracy vs the true circle (flat region), patch sizes
//! refine-bench time    [levels=7,8,9,10] [counts=1000,10000] [--small]   refine time, full recompute and 1% edit
//! refine-bench cracks  [level=7] [count=1000] [--mixed] [--naive]  watertightness of base + patches on a closed sphere
//!                      (--naive: control experiment, leaves skip their border faces -> cracks)
//! refine-bench dump    <out_dir>                                  OBJ + PNG with and without refinement
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use glam::{Quat, Vec3};
use sculpt_core::Document;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::refine::{MeshView, RefineParams, RefineStats, RefinedPatch, check_closed, compose, refine_all, refine_leaves};
use sculpt_core::generator::{EvalParams, Instance, InstanceSet, Primitive, affected_leaves, evaluate, evaluate_leaves, frame_from_normal};
use sculpt_core::io::obj;
use sculpt_core::mesh::{NO_VERT, PolyMesh, compute_normals};
use sculpt_core::primitives::{grid, quad_sphere};

use crate::render_mesh::{self, Cam, Scene};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `--crease=F` / `--wall=F` override the refinement edge-length targets.
fn params_from(args: &[String]) -> RefineParams {
    let mut p = RefineParams::default();
    for a in args {
        if let Some(v) = a.strip_prefix("--crease=").and_then(|v| v.parse().ok()) {
            p.crease_frac = v;
        }
        if let Some(v) = a.strip_prefix("--wall=").and_then(|v| v.parse().ok()) {
            p.wall_frac = v;
        }
        if let Some(v) = a.strip_prefix("--levels=").and_then(|v| v.parse().ok()) {
            p.max_levels = v;
        }
    }
    p
}

pub fn run(args: &[String]) -> Res<()> {
    match args.first().map(String::as_str) {
        Some("quality") => quality(&args[1..]),
        Some("time") => time(&args[1..]),
        Some("cracks") => cracks(&args[1..]),
        Some("dump") if args.len() >= 2 => dump(Path::new(&args[1])),
        _ => {
            eprintln!("usage: refine-bench quality [faces] [sizes] | time [levels] [counts] [--small] | cracks [level] [count] [--mixed] | dump <out_dir>");
            std::process::exit(2);
        }
    }
}

// ------------------------------------------------------------------ helpers

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f();
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        last = f();
        times.push(ms(t));
    }
    (median(times), last)
}

fn rng(i: u64, k: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

fn list_f(args: &[String], i: usize, default: &str) -> Vec<f64> {
    args.get(i).filter(|s| !s.starts_with("--")).map_or(default, |s| s.as_str()).split(',').filter_map(|x| x.parse().ok()).collect()
}

fn view(doc: &Document) -> MeshView<'_> {
    MeshView { positions: doc.positions(), normals: doc.normals(), faces: doc.faces(), bvh: doc.bvh() }
}

fn sphere_doc(level: usize) -> Res<Document> {
    let mut doc = Document::from_mesh(quad_sphere((level as u32).min(6), 1.0))?;
    while (doc.level() as usize) + level.min(6) < level {
        doc.subdivide()?;
    }
    Ok(doc)
}

/// Distance from `p` to the circle of radius `rho` around `c` in the plane y = 0.
fn dist_circle(p: Vec3, c: Vec3, rho: f32) -> f32 {
    let r = ((p.x - c.x).powi(2) + (p.z - c.z).powi(2)).sqrt();
    ((r - rho).powi(2) + p.y * p.y).sqrt()
}

fn dist_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[derive(Default)]
struct Acc {
    dev_all: Vec<f32>,
    dev_max: Vec<f32>,
    haus: Vec<f32>,
    lost: usize,
}

impl Acc {
    fn row(&self, e: f32, total: usize) -> String {
        let mean = if self.dev_all.is_empty() { f32::NAN } else { self.dev_all.iter().sum::<f32>() / self.dev_all.len() as f32 };
        let mx = self.dev_max.iter().cloned().fold(0.0, f32::max);
        let hm = median(self.haus.iter().map(|&x| x as f64).collect()) as f32;
        let hx = self.haus.iter().cloned().fold(0.0, f32::max);
        let mut sorted = self.haus.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let h95 = sorted.get(((sorted.len() as f32) * 0.95) as usize).copied().unwrap_or(f32::NAN);
        format!("{} / {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}", total - self.lost, total, mean / e, mx / e, hm / e, h95 / e, hx / e)
    }
}

// ------------------------------------------------------------------ quality

const LATTICE: usize = 16;

fn quality(args: &[String]) -> Res<()> {
    let faces_list = list_f(args, 0, "98000,1600000");
    let sizes = list_f(args, 1, "1.5,3,6");
    let params = params_from(args);
    println!("threads: {}  crease_frac {} wall_frac {} max_levels {} projection iterations {}", rayon::current_num_threads(), params.crease_frac, params.wall_frac, params.max_levels, params.eval.iterations);
    println!("\nRim accuracy: sphere cutters on a flat quad grid, lattice {LATTICE}x{LATTICE} (spacing 12 edges, random sub-edge offset and height c in [0, 0.5R]).");
    println!("Distances in mesh edges. Only cutters whose AABB touches a single BVH leaf are counted (rim of a refined cutter then lies in one patch).\n");
    let mut acc_rows = Vec::new();
    let mut size_rows = Vec::new();
    for &faces in &faces_list {
        let n = (faces.sqrt()).round() as u32;
        let e = 0.01f32;
        let doc = Document::from_mesh(grid(n, n as f32 * e))?;
        println!("grid {n}x{n}: {} faces, {} leaves", doc.face_count(), doc.bvh().leaves.len());
        let (pos, topo) = (doc.positions(), doc.topology());
        for &dia in &sizes {
            let dia = dia as f32;
            let r = dia * e * 0.5;
            let spacing = 12.0 * e;
            let mut cutters = Vec::new(); // (instance, center_xz, rho)
            for j in 0..LATTICE {
                for i in 0..LATTICE {
                    let id = (j * LATTICE + i) as u64;
                    let ox = (rng(id, 1) - 0.5) * e;
                    let oz = (rng(id, 2) - 0.5) * e;
                    let c = rng(id, 3) * 0.5 * r;
                    let cx = (i as f32 - (LATTICE as f32 - 1.0) / 2.0) * spacing + ox;
                    let cz = (j as f32 - (LATTICE as f32 - 1.0) / 2.0) * spacing + oz;
                    let inst = Instance::new(Primitive::Sphere, Vec3::new(cx, c, cz), Quat::IDENTITY, Vec3::splat(r));
                    cutters.push((inst, Vec3::new(cx, 0.0, cz), (r * r - c * c).sqrt()));
                }
            }
            let set = InstanceSet::new(cutters.iter().map(|c| c.0).collect(), 0.0);
            let ok: Vec<bool> = (0..cutters.len()).map(|k| affected_leaves(doc.bvh(), &[set.aabb(k)]).len() == 1).collect();
            let owner = |p: Vec3| -> Option<usize> {
                let fi = |v: f32| (v / spacing + (LATTICE as f32 - 1.0) / 2.0).round();
                let (i, j) = (fi(p.x), fi(p.z));
                if i < 0.0 || j < 0.0 || i >= LATTICE as f32 || j >= LATTICE as f32 {
                    return None;
                }
                Some(j as usize * LATTICE + i as usize)
            };
            let circle: Vec<Vec3> = (0..360).map(|k| { let a = k as f32 / 360.0 * std::f32::consts::TAU; Vec3::new(a.cos(), 0.0, a.sin()) }).collect();

            // --- Spike A rings.
            let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
            let (t_a, _) = timed(3, || evaluate(pos, doc.normals(), doc.bvh(), &set, &EvalParams::default(), &mut delta));
            let mut ring: Vec<Vec<Vec3>> = vec![Vec::new(); cutters.len()];
            let mut moved_any = vec![false; cutters.len()];
            for v in 0..doc.vertex_count() {
                let Some(k) = owner(pos[v]) else { continue };
                let moved = delta[v] != Vec3::ZERO;
                if moved {
                    moved_any[k] = true;
                }
                let nb = topo.vert_verts.row(v);
                let boundary = nb.iter().any(|&u| (delta[u as usize] != Vec3::ZERO) != moved);
                if boundary {
                    ring[k].push(pos[v] + delta[v]);
                }
            }
            let mut a = Acc::default();
            for (k, (_, c, rho)) in cutters.iter().enumerate() {
                if !ok[k] {
                    continue;
                }
                if !moved_any[k] {
                    a.lost += 1;
                    continue;
                }
                let devs: Vec<f32> = ring[k].iter().map(|&p| dist_circle(p, *c, *rho)).collect();
                let dmax = devs.iter().cloned().fold(0.0, f32::max);
                let cover = circle.iter().map(|&u| ring[k].iter().map(|&p| (*c + u * *rho).distance(p)).fold(f32::INFINITY, f32::min)).fold(0.0, f32::max);
                a.dev_all.extend(devs);
                a.dev_max.push(dmax);
                a.haus.push(dmax.max(cover));
            }
            let total = ok.iter().filter(|&&x| x).count();

            // --- Refined.
            let (t_r, (patches, _)) = timed(3, || refine_all(&view(&doc), &set, &params));
            let mut cr: Vec<Vec<Vec3>> = vec![Vec::new(); cutters.len()];
            let mut seg: Vec<Vec<(Vec3, Vec3)>> = vec![Vec::new(); cutters.len()];
            for p in &patches {
                for &ci in &p.crease_vertices {
                    let q = p.positions[ci as usize];
                    if let Some(k) = owner(q) {
                        cr[k].push(q);
                    }
                }
                for &[x, y] in &p.crease_edges {
                    let (qx, qy) = (p.positions[x as usize], p.positions[y as usize]);
                    if let Some(k) = owner(qx) {
                        seg[k].push((qx, qy));
                    }
                }
            }
            let mut rv = Acc::default();
            let mut rp = Acc::default();
            for (k, (_, c, rho)) in cutters.iter().enumerate() {
                if !ok[k] {
                    continue;
                }
                if cr[k].is_empty() {
                    rv.lost += 1;
                    rp.lost += 1;
                    continue;
                }
                let devs: Vec<f32> = cr[k].iter().map(|&p| dist_circle(p, *c, *rho)).collect();
                let dmax = devs.iter().cloned().fold(0.0, f32::max);
                let cover = circle.iter().map(|&u| cr[k].iter().map(|&p| (*c + u * *rho).distance(p)).fold(f32::INFINITY, f32::min)).fold(0.0, f32::max);
                rv.dev_all.extend(devs);
                rv.dev_max.push(dmax);
                rv.haus.push(dmax.max(cover));
                // polyline Hausdorff (chord sag included)
                let sag = seg[k]
                    .iter()
                    .flat_map(|&(x, y)| [0.25f32, 0.5, 0.75].map(|t| dist_circle(x.lerp(y, t), *c, *rho)))
                    .fold(0.0, f32::max);
                let cover_p = circle.iter().map(|&u| seg[k].iter().map(|&(x, y)| dist_segment(*c + u * *rho, x, y)).fold(f32::INFINITY, f32::min)).fold(0.0, f32::max);
                rp.dev_all.push(sag);
                rp.dev_max.push(sag);
                rp.haus.push(sag.max(cover_p));
            }
            acc_rows.push(format!("| {} | {dia} | A (no refinement) | {} |", doc.face_count(), a.row(e, total)));
            acc_rows.push(format!("| {} | {dia} | refined, rim vertices | {} |", doc.face_count(), rv.row(e, total)));
            acc_rows.push(format!("| {} | {dia} | refined, rim polyline | {} |", doc.face_count(), rp.row(e, total)));
            let stats = RefineStats::of(&patches, 0);
            let levels = patches.iter().map(|p| p.stats.levels).max().unwrap_or(0);
            let cn = cutters.len() as f64;
            size_rows.push(format!(
                "| {} | {dia} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {levels} | {t_a:.2} | {t_r:.2} |",
                doc.face_count(),
                stats.faces_replaced as f64 / cn,
                stats.tris_before as f64 / cn,
                stats.tris_after as f64 / cn,
                (stats.tris_after as f64 - stats.tris_before as f64) / cn,
                stats.new_vertices as f64 / cn,
            ));
        }
    }
    println!("\n| faces | cutter dia (edges) | method | cutters measured | boundary vertex dev mean | dev max | Hausdorff median | Hausdorff p95 | Hausdorff max |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in acc_rows {
        println!("{r}");
    }
    println!("\nPatch size per cutter ({} cutters), and time of the whole scene (256 cutters):\n", LATTICE * LATTICE);
    println!("| faces | cutter dia (edges) | base faces replaced | triangles before | triangles after | triangles added | new vertices | max levels | A evaluate ms (all) | refine ms (all) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for r in size_rows {
        println!("{r}");
    }
    Ok(())
}

// ------------------------------------------------------------------ time

fn sphere_instances(doc: &Document, sampler: &SurfaceSampler, edge: f32, n: usize, small: bool) -> Vec<Instance> {
    let pts = sampler.sample(doc.positions(), doc.normals(), doc.faces(), None, n, 7);
    pts.iter()
        .enumerate()
        .map(|(i, s)| {
            let d = if small { edge * (1.5 + 2.5 * rng(i as u64, 1)) } else { edge * (3.0 + 7.0 * rng(i as u64, 1)) };
            let rot = frame_from_normal(s.normal, rng(i as u64, 2) * std::f32::consts::TAU);
            // Centre slightly above the surface so the cutter leaves a crater.
            Instance::new(Primitive::Sphere, s.position + s.normal * (0.15 * d * 0.5), rot, Vec3::splat(d * 0.5))
        })
        .collect()
}

fn patch_mb(patches: &[RefinedPatch]) -> f64 {
    patches.iter().map(|p| p.positions.len() * 24 + p.indices.len() * 4).sum::<usize>() as f64 / 1e6
}

fn time(args: &[String]) -> Res<()> {
    let levels = list_f(args, 0, "7,8,9,10");
    let counts = list_f(args, 1, "1000,10000");
    let small = args.iter().any(|a| a == "--small");
    let params = params_from(args);
    println!("threads: {}  cutter diameter {} mesh edges (uniform), spheres", rayon::current_num_threads(), if small { "1.5-4" } else { "3-10" });
    println!("\n| faces | N | A full ms | refine full ms | leaves | faces replaced | tris after | patch MB | A 1% ms | refine 1% ms | 1% leaves | 1% faces replaced | 1% tris after |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &level in &levels {
        let level = level as usize;
        let doc = sphere_doc(level)?;
        let (pos, nrm, faces, bvh) = (doc.positions(), doc.normals(), doc.faces(), doc.bvh());
        let sampler = SurfaceSampler::new(pos, faces, None);
        let edge = ((sampler.total_weight() as f32) / doc.face_count() as f32).sqrt();
        for &n in &counts {
            let n = n as usize;
            let instances = sphere_instances(&doc, &sampler, edge, n, small);
            let mut set = InstanceSet::new(instances.clone(), 0.0);
            let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
            let (t_a, _) = timed(3, || evaluate(pos, nrm, bvh, &set, &EvalParams::default(), &mut delta));
            let (t_r, (patches, leaves)) = timed(3, || refine_all(&view(&doc), &set, &params));
            let st = RefineStats::of(&patches, leaves);

            // 1% edit: the same instances change as in generator-bench.
            let k = (n / 100).max(1);
            let changes: Vec<(usize, Instance)> = (0..k).map(|j| (j * (n / k), Instance { scale: instances[j * (n / k)].scale * 1.3, ..instances[j * (n / k)] })).collect();
            let (mut ta, mut tr, mut l1, mut f1, mut t1) = (Vec::new(), Vec::new(), 0, 0, 0);
            for rep in 0..7 {
                let ch: Vec<(usize, Instance)> = if rep % 2 == 0 { changes.clone() } else { changes.iter().map(|&(i, _)| (i, instances[i])).collect() };
                let t = Instant::now();
                let dirty = set.update(&ch);
                let leaves = affected_leaves(bvh, &dirty);
                let t_up = ms(t);
                let t = Instant::now();
                evaluate_leaves(pos, nrm, bvh, &set, &EvalParams::default(), &leaves, &mut delta);
                let t_ev = ms(t);
                let t = Instant::now();
                let p = refine_leaves(&view(&doc), &set, &params, &leaves);
                let t_rf = ms(t);
                if rep > 0 {
                    ta.push(t_up + t_ev);
                    tr.push(t_up + t_rf);
                }
                let s = RefineStats::of(&p, leaves.len());
                (l1, f1, t1) = (leaves.len(), s.faces_replaced, s.tris_after);
            }
            println!(
                "| {} | {n} | {t_a:.1} | {t_r:.1} | {leaves} | {} | {} | {:.1} | {:.2} | {:.2} | {l1} | {f1} | {t1} |",
                doc.face_count(),
                st.faces_replaced,
                st.tris_after,
                patch_mb(&patches),
                median(ta),
                median(tr)
            );
        }
    }
    println!("\nA 1% = instance update + dirty leaf lookup + evaluate those leaves; refine 1% = the same update/lookup + refine those leaves (patches only).");
    Ok(())
}

// ------------------------------------------------------------------ cracks

fn cracks(args: &[String]) -> Res<()> {
    let level = list_f(args, 0, "7")[0] as usize;
    let count = list_f(args, 1, "1000")[0] as usize;
    let mixed = args.iter().any(|a| a == "--mixed");
    let naive = args.iter().any(|a| a == "--naive");
    let doc = sphere_doc(level)?;
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let sampler = SurfaceSampler::new(pos, faces, None);
    let edge = ((sampler.total_weight() as f32) / doc.face_count() as f32).sqrt();
    let mut instances = sphere_instances(&doc, &sampler, edge, count, true);
    if mixed {
        for (i, ins) in instances.iter_mut().enumerate() {
            ins.primitive = Primitive::ALL[(rng(i as u64, 3) * 7.0) as usize % 7];
            ins.scale *= Vec3::new(0.7 + 0.3 * rng(i as u64, 4), 0.7 + 0.3 * rng(i as u64, 5), 0.7 + 0.3 * rng(i as u64, 6));
        }
    }
    let set = InstanceSet::new(instances, 0.0);
    let params = RefineParams { debug_skip_leaf_borders: naive, ..params_from(args) };
    let (patches, leaves) = refine_all(&view(&doc), &set, &params);
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(pos, nrm, doc.bvh(), &set, &EvalParams::default(), &mut delta);
    let a_pos: Vec<Vec3> = pos.iter().zip(&delta).map(|(p, d)| *p + *d).collect();
    let base = check_closed(&compose(pos, nrm, faces, &[]));
    let c = compose(&a_pos, nrm, faces, &patches);
    let rep = check_closed(&c);
    let mut seen: HashMap<(u32, u32, u32), (u32, bool)> = HashMap::new();
    let mut shared_new = 0usize;
    for p in &patches {
        for (i, q) in p.positions.iter().enumerate() {
            let k = (q.x.to_bits(), q.y.to_bits(), q.z.to_bits());
            let e = seen.entry(k).or_insert((p.leaf, false));
            if e.0 != p.leaf && !e.1 {
                e.1 = true;
                if p.base_vertex[i] == NO_VERT {
                    shared_new += 1;
                }
            }
        }
    }
    let shared = seen.values().filter(|v| v.1).count();
    let st = RefineStats::of(&patches, leaves);
    println!("mesh {} faces, {} leaves; {} instances ({}); {} leaves patched, {} faces replaced, {} tris after", doc.face_count(), doc.bvh().leaves.len(), count, if mixed { "all primitives" } else { "spheres 1.5-4 edges" }, leaves, st.faces_replaced, st.tris_after);
    println!("base mesh      : {base:?}");
    println!("base + patches : {rep:?}  closed = {}", rep.is_closed());
    println!("vertices shared bit-identically by patches of different leaves: {shared} ({shared_new} of them new vertices, not base vertices)");
    if !rep.is_closed() && !naive {
        std::process::exit(1);
    }
    Ok(())
}

// ------------------------------------------------------------------ dump

struct Meshes {
    a_pos: Vec<Vec3>,
    a_nrm: Vec<Vec3>,
    a_tris: Vec<[u32; 3]>,
    r_pos: Vec<Vec3>,
    r_nrm: Vec<Vec3>,
    r_tris: Vec<[u32; 3]>,
    rim: Vec<Vec3>,
    patches: usize,
    replaced: usize,
    tris_after: usize,
}

fn build(doc: &Document, set: &InstanceSet) -> Meshes {
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let mut delta = vec![Vec3::ZERO; doc.vertex_count()];
    evaluate(pos, nrm, doc.bvh(), set, &EvalParams::default(), &mut delta);
    let a_pos: Vec<Vec3> = pos.iter().zip(&delta).map(|(p, d)| *p + *d).collect();
    let a_nrm = compute_normals(&a_pos, faces, doc.topology());
    let a_tris: Vec<[u32; 3]> = faces.iter().flat_map(sculpt_core::mesh::face_triangles).collect();
    let (patches, _) = refine_all(&view(doc), set, &RefineParams::default());
    let c = compose(&a_pos, &a_nrm, faces, &patches);
    let off: Vec<u32> = patches.iter().scan(pos.len() as u32, |o, p| { let s = *o; *o += p.positions.len() as u32; Some(s) }).collect();
    let rim = patches.iter().zip(&off).flat_map(|(p, _)| p.crease_vertices.iter().map(|&i| p.positions[i as usize])).collect();
    let st = RefineStats::of(&patches, 0);
    Meshes { a_pos, a_nrm, a_tris, r_pos: c.positions, r_nrm: c.normals, r_tris: c.tris, rim, patches: patches.len(), replaced: st.faces_replaced, tris_after: st.tris_after }
}

fn write_obj(path: &Path, pos: &[Vec3], tris: &[[u32; 3]]) -> Res<()> {
    let m = PolyMesh { positions: pos.to_vec(), faces: tris.iter().map(|t| [t[0], t[1], t[2], NO_VERT]).collect() };
    obj::write(path, &m)?;
    Ok(())
}

fn dump(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    // ---- flat region with cutters of 1.5, 3 and 6 mesh edges (+ a box, a cylinder and a cone).
    let n = 140u32;
    let e = 0.01f32;
    let doc = Document::from_mesh(grid(n, n as f32 * e))?;
    let sph = |x: f32, z: f32, dia: f32| {
        let r = dia * e * 0.5;
        Instance::new(Primitive::Sphere, Vec3::new(x * e, 0.25 * r, z * e), Quat::IDENTITY, Vec3::splat(r))
    };
    let mut insts = vec![sph(-14.3, -6.0, 1.5), sph(-3.4, -6.0, 3.0), sph(11.2, -6.0, 6.0)];
    insts.push(Instance::new(Primitive::Box, Vec3::new(-12.0 * e, 1.0 * e, 10.0 * e), Quat::from_rotation_y(0.5), Vec3::new(2.5, 1.2, 1.7) * e));
    insts.push(Instance::new(Primitive::Cylinder, Vec3::new(-e, 0.2 * e, 10.0 * e), Quat::from_rotation_x(0.35), Vec3::new(1.6, 1.5, 1.6) * e));
    insts.push(Instance::new(Primitive::Cone, Vec3::new(12.0 * e, 0.5 * e, 10.0 * e), Quat::from_rotation_x(std::f32::consts::PI - 0.3), Vec3::new(2.5, 2.0, 2.5) * e));
    let set = InstanceSet::new(insts.clone(), 0.0);
    let m = build(&doc, &set);
    println!("flat: {} patches, {} base faces replaced, {} triangles after", m.patches, m.replaced, m.tris_after);
    write_obj(&dir.join("flat_A.obj"), &m.a_pos, &m.a_tris)?;
    write_obj(&dir.join("flat_refined.obj"), &m.r_pos, &m.r_tris)?;
    let sa = |wire: bool| Scene { positions: &m.a_pos, normals: &m.a_nrm, tris: &m.a_tris, wire, dots: &[] };
    let sr = |wire: bool, dots: bool| Scene { positions: &m.r_pos, normals: &m.r_nrm, tris: &m.r_tris, wire, dots: if dots { &m.rim } else { &[] } };
    let size = 560;
    let overview = Cam { center: Vec3::new(0.0, 0.0, 2.0 * e), half_extent: 24.0 * e, yaw: 0.0, pitch: 1.0 };
    render_mesh::render(&[(&sa(false), overview), (&sr(false, false), overview)], size, &dir.join("flat_overview.png"))?;
    for (name, x, z, he) in [("d1.5", -14.3f32, -6.0f32, 3.2f32), ("d3", -3.4, -6.0, 4.4), ("d6", 11.2, -6.0, 7.0), ("box", -12.0, 10.0, 6.0), ("cyl", -1.0, 10.0, 5.0), ("cone", 12.0, 10.0, 6.0)] {
        let cam = Cam { center: Vec3::new(x * e, 0.0, z * e), half_extent: he * e, yaw: 0.0, pitch: 1.0 };
        render_mesh::render(
            &[(&sa(true), cam), (&sa(false), cam), (&sr(true, true), cam), (&sr(false, false), cam)],
            400,
            &dir.join(format!("flat_{name}.png")),
        )?;
    }

    // ---- sphere (98k faces) with cutters of 1.5-6 edges in front.
    let doc = sphere_doc(7)?;
    let edge = (4.0 * std::f32::consts::PI / doc.face_count() as f32).sqrt();
    let insts: Vec<Instance> = (0..70u64)
        .map(|i| {
            let d = Vec3::new((rng(i, 1) - 0.5) * 0.62, (rng(i, 2) - 0.5) * 0.62, 1.0).normalize();
            let r = edge * (0.75 + 2.25 * rng(i, 3));
            Instance::new(Primitive::Sphere, d * (1.0 + 0.25 * r), Quat::IDENTITY, Vec3::splat(r))
        })
        .collect();
    let set = InstanceSet::new(insts, 0.0);
    let m = build(&doc, &set);
    println!("sphere: {} patches, {} base faces replaced, {} triangles after", m.patches, m.replaced, m.tris_after);
    write_obj(&dir.join("sphere_A.obj"), &m.a_pos, &m.a_tris)?;
    write_obj(&dir.join("sphere_refined.obj"), &m.r_pos, &m.r_tris)?;
    let sa = Scene { positions: &m.a_pos, normals: &m.a_nrm, tris: &m.a_tris, wire: false, dots: &[] };
    let sr = Scene { positions: &m.r_pos, normals: &m.r_nrm, tris: &m.r_tris, wire: false, dots: &[] };
    let cam = Cam { center: Vec3::new(0.0, 0.0, 1.0), half_extent: 0.2, yaw: 0.0, pitch: 0.0 };
    render_mesh::render(&[(&sa, cam), (&sr, cam)], 640, &dir.join("sphere_overview.png"))?;
    let cam2 = Cam { center: Vec3::new(0.0, 0.0, 1.0), half_extent: 0.07, yaw: 0.0, pitch: 0.0 };
    let (sa_w, sr_w) = (Scene { wire: true, ..Scene { positions: &m.a_pos, normals: &m.a_nrm, tris: &m.a_tris, wire: true, dots: &[] } }, Scene { positions: &m.r_pos, normals: &m.r_nrm, tris: &m.r_tris, wire: true, dots: &m.rim });
    render_mesh::render(&[(&sa_w, cam2), (&sr_w, cam2)], 640, &dir.join("sphere_zoom_wire.png"))?;
    Ok(())
}

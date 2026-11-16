//! The analysis: collapse the whole mesh down to a floor, in parallel rounds over spatial tiles, and keep the
//! collapse record that [`Analysis::extract`] replays to any count.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use glam::{DVec3, Vec3};
use rayon::prelude::*;

use super::collapse::{Engine, Params, Rec};
use super::quadric::{Quadric, d3};
use super::replay::Decimated;
use super::{Target, TriMesh};
use crate::geom::Aabb;
use crate::mesh::Csr;

#[derive(Clone, Copy, Debug)]
pub struct AnalyzeOptions {
    /// 0: keep the silhouette (quadric error); 1: even, near-equilateral triangles.
    pub evenness: f32,
    /// Collapse down to this share of the input's triangles (and at least `min_floor`).
    pub floor_ratio: f32,
    pub min_floor: usize,
    /// Below this many faces the rest is done in one piece, without tiles.
    pub single_pass_faces: usize,
    /// Faces per tile in the parallel rounds.
    pub tile_faces: usize,
    /// Share of the queued edges each parallel round aims to take (the round's cost threshold is this quantile).
    pub round_quantile: f64,
    pub max_valence: usize,
    /// Largest turn of a face normal a collapse may cause, in degrees.
    pub max_normal_turn_deg: f32,
    /// Keep every border vertex where it is (otherwise borders are decimated too, sliding only along themselves), so
    /// pieces that meet at their borders still weld after decimation.
    pub freeze_borders: bool,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        AnalyzeOptions {
            evenness: 0.0,
            floor_ratio: 0.001,
            min_floor: 64,
            single_pass_faces: 50_000,
            tile_faces: 60_000,
            round_quantile: 0.5,
            max_valence: 24,
            max_normal_turn_deg: 75.0,
            freeze_borders: false,
        }
    }
}

/// Watch or stop a running analysis from another thread.
#[derive(Default, Debug)]
pub struct Progress {
    /// 0..=1000.
    pub permille: AtomicU32,
    pub cancel: AtomicBool,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        self.permille.load(Ordering::Relaxed) as f32 / 1000.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnalyzeStats {
    pub ms: f32,
    pub rounds: usize,
    pub input_triangles: usize,
    pub floor_triangles: usize,
    /// Faces dropped before the analysis (a repeated vertex).
    pub degenerate: usize,
    pub border_edges: usize,
    pub nonmanifold_edges: usize,
    pub locked_vertices: usize,
}

/// A mesh analysed for decimation: replay to any triangle count with [`Self::extract`].
pub struct Analysis {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    evenness: f32,
    /// Per vertex: the step that removed it (`u32::MAX`: never) and the vertex it went into.
    rstep: Vec<u32>,
    parent: Vec<u32>,
    /// Used vertices, latest removed first; `vrank` its inverse.
    vorder: Vec<u32>,
    vrank: Vec<u32>,
    /// Faces, latest dying first.
    forder: Vec<u32>,
    /// Faces alive after `k` collapses, `k` in `0..=steps`.
    alive: Vec<u32>,
    /// Steps at which each vertex was moved (as the kept end), and where each step put it.
    hist: Csr,
    step_pos: Vec<Vec3>,
    pub stats: AnalyzeStats,
}

struct State {
    pos: Vec<Vec3>,
    quad: Vec<Quadric>,
    faces: Vec<[u32; 3]>,
    falive: Vec<bool>,
    valive: Vec<bool>,
    /// Live faces and vertices (refreshed each round), and a per-vertex scratch for tiles.
    lf: Vec<u32>,
    lv: Vec<u32>,
    vtile: Vec<u32>,
    border: Vec<bool>,
    locked: Vec<bool>,
    death: Vec<u32>,
    recs: Vec<Rec>,
}

impl Analysis {
    pub fn build(mesh: &TriMesh, opts: &AnalyzeOptions, progress: Option<&Progress>) -> Option<Analysis> {
        let t0 = Instant::now();
        let nv = mesh.positions.len();
        let mut stats = AnalyzeStats::default();

        // Faces with a repeated vertex take no part.
        let valid: Vec<bool> = mesh.triangles.par_iter().map(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]).collect();
        stats.degenerate = valid.iter().filter(|v| !**v).count();
        let nf_valid = mesh.triangles.len() - stats.degenerate;
        stats.input_triangles = nf_valid;
        let floor = ((nf_valid as f64 * opts.floor_ratio as f64) as usize).max(opts.min_floor);

        // Edges: borders get constraint planes, edges in more than two faces lock their ends.
        let mut edges: Vec<(u64, u32)> = mesh
            .triangles
            .par_iter()
            .enumerate()
            .filter(|(f, _)| valid[*f])
            .flat_map_iter(|(f, t)| {
                (0..3).map(move |i| {
                    let (a, b) = (t[i], t[(i + 1) % 3]);
                    (((a.min(b) as u64) << 32) | a.max(b) as u64, f as u32)
                })
            })
            .collect();
        edges.par_sort_unstable();
        let mut quad: Vec<Quadric> = vec![Quadric::default(); nv];
        let mut border = vec![false; nv];
        let mut locked = vec![false; nv];
        let mut used = vec![false; nv];
        {
            let mut i = 0;
            while i < edges.len() {
                let mut j = i + 1;
                while j < edges.len() && edges[j].0 == edges[i].0 {
                    j += 1;
                }
                let (a, b) = ((edges[i].0 >> 32) as usize, (edges[i].0 & 0xffff_ffff) as usize);
                used[a] = true;
                used[b] = true;
                if j - i == 1 {
                    stats.border_edges += 1;
                    border[a] = true;
                    border[b] = true;
                    let t = mesh.triangles[edges[i].1 as usize];
                    let p = t.map(|v| d3(mesh.positions[v as usize]));
                    let fnrm = (p[1] - p[0]).cross(p[2] - p[0]);
                    let (pa, pb) = (d3(mesh.positions[a]), d3(mesh.positions[b]));
                    let e = pb - pa;
                    let n = e.cross(fnrm).normalize_or_zero();
                    if n != DVec3::ZERO {
                        let q = Quadric::constraint(n, pa, 10.0 * e.length_squared());
                        quad[a].add(&q);
                        quad[b].add(&q);
                    }
                } else if j - i > 2 {
                    stats.nonmanifold_edges += 1;
                    locked[a] = true;
                    locked[b] = true;
                }
                i = j;
            }
        }
        drop(edges);
        if opts.freeze_borders {
            for (l, b) in locked.iter_mut().zip(&border) {
                *l |= *b;
            }
        }
        stats.locked_vertices = locked.iter().filter(|l| **l).count();

        // Face planes, area weighted.
        let vf = Csr::from_pairs(
            nv,
            mesh.triangles.iter().enumerate().filter(|(f, _)| valid[*f]).flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))),
        );
        let face_q: Vec<Quadric> = mesh
            .triangles
            .par_iter()
            .map(|t| {
                let p = t.map(|v| d3(mesh.positions[v as usize]));
                let c = (p[1] - p[0]).cross(p[2] - p[0]);
                let l = c.length();
                if l > 0.0 { Quadric::plane(c / l, p[0], 0.5 * l) } else { Quadric::default() }
            })
            .collect();
        quad.par_iter_mut().enumerate().for_each(|(v, q)| {
            for &f in vf.row(v) {
                q.add(&face_q[f as usize]);
            }
        });
        drop(face_q);
        drop(vf);

        // Scale for the cost's small constant: mean edge length.
        let mean_len = {
            let step = (mesh.triangles.len() / 4096).max(1);
            let (mut s, mut n) = (0.0f64, 0usize);
            for (f, t) in mesh.triangles.iter().enumerate().step_by(step) {
                if valid[f] {
                    s += mesh.positions[t[0] as usize].distance(mesh.positions[t[1] as usize]) as f64;
                    n += 1;
                }
            }
            if n > 0 { s / n as f64 } else { 1.0 }
        };
        let params = Params {
            evenness: opts.evenness as f64,
            eps: 1e-6 * mean_len.powi(4),
            max_valence: opts.max_valence,
            min_dot: (opts.max_normal_turn_deg as f64).to_radians().cos(),
            min_quality: 0.05 + 0.25 * opts.evenness as f64,
        };

        let mut st = State {
            pos: mesh.positions.clone(),
            quad,
            faces: mesh.triangles.clone(),
            falive: valid.clone(),
            valive: used.clone(),
            lf: (0..mesh.triangles.len() as u32).filter(|&f| valid[f as usize]).collect(),
            lv: (0..nv as u32).filter(|&v| used[v as usize]).collect(),
            vtile: vec![u32::MAX; nv],
            border,
            locked,
            death: vec![u32::MAX; mesh.triangles.len()],
            recs: Vec::new(),
        };
        let bounds = Aabb::from_points(&mesh.positions);
        let mut alive_faces = nf_valid;
        let total_to_remove = nf_valid.saturating_sub(floor).max(1) as f64;
        let set_progress = |alive_faces: usize| {
            if let Some(p) = progress {
                let done = (nf_valid - alive_faces) as f64 / total_to_remove;
                p.permille.store((done.clamp(0.0, 1.0) * 990.0) as u32, Ordering::Relaxed);
            }
        };
        let cancelled = || progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed));

        let trace = std::env::var_os("DECIMATE_TRACE").is_some();
        if trace {
            eprintln!("setup {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut round = 0usize;
        while alive_faces > floor {
            let tr = Instant::now();
            if cancelled() {
                return None;
            }
            let single = alive_faces <= opts.single_pass_faces;
            let removed = if single {
                one_round(&mut st, &bounds, None, 0, params, f64::INFINITY, floor, alive_faces, opts)
            } else {
                one_round(&mut st, &bounds, Some(round), opts.tile_faces, params, f64::NAN, floor, alive_faces, opts)
            };
            round += 1;
            alive_faces -= removed;
            if trace {
                eprintln!("round {round}: -{removed} -> {alive_faces} in {:.0} ms", tr.elapsed().as_secs_f64() * 1e3);
            }
            set_progress(alive_faces);
            if single || (removed as f64) < 0.01 * alive_faces as f64 {
                // Done, or the rounds stall: one last pass over everything left.
                if !single {
                    let removed = one_round(&mut st, &bounds, None, 0, params, f64::INFINITY, floor, alive_faces, opts);
                    alive_faces -= removed;
                    round += 1;
                }
                break;
            }
        }
        stats.rounds = round;
        stats.floor_triangles = alive_faces;

        let tr = Instant::now();
        let a = Analysis::from_record(mesh, &valid, &used, st, opts.evenness, stats, t0);
        if trace {
            eprintln!("record {:.0} ms", tr.elapsed().as_secs_f64() * 1e3);
        }
        if let Some(p) = progress {
            p.permille.store(1000, Ordering::Relaxed);
        }
        Some(a)
    }

    fn from_record(mesh: &TriMesh, valid: &[bool], used: &[bool], st: State, evenness: f32, mut stats: AnalyzeStats, t0: Instant) -> Analysis {
        let nv = mesh.positions.len();
        let steps = st.recs.len();
        let mut rstep = vec![u32::MAX; nv];
        let mut parent = vec![u32::MAX; nv];
        for (s, r) in st.recs.iter().enumerate() {
            rstep[r.removed as usize] = s as u32;
            parent[r.removed as usize] = r.kept;
        }
        let mut vorder: Vec<u32> = (0..nv as u32).filter(|&v| used[v as usize]).collect();
        vorder.par_sort_unstable_by_key(|&v| std::cmp::Reverse(rstep[v as usize]));
        let mut vrank = vec![u32::MAX; nv];
        for (i, &v) in vorder.iter().enumerate() {
            vrank[v as usize] = i as u32;
        }
        let mut forder: Vec<u32> = (0..mesh.triangles.len() as u32).filter(|&f| valid[f as usize]).collect();
        forder.par_sort_unstable_by_key(|&f| std::cmp::Reverse(st.death[f as usize]));
        let mut alive = Vec::with_capacity(steps + 1);
        let mut n = forder.len() as u32;
        alive.push(n);
        for r in &st.recs {
            n -= r.dead.iter().filter(|d| **d != u32::MAX).count() as u32;
            alive.push(n);
        }
        let hist = Csr::from_pairs(nv, st.recs.iter().enumerate().map(|(s, r)| (r.kept, s as u32)));
        let step_pos = st.recs.iter().map(|r| r.pos).collect();
        stats.ms = t0.elapsed().as_secs_f32() * 1e3;
        Analysis {
            positions: mesh.positions.clone(),
            triangles: mesh.triangles.clone(),
            evenness,
            rstep,
            parent,
            vorder,
            vrank,
            forder,
            alive,
            hist,
            step_pos,
            stats,
        }
    }

    pub fn evenness(&self) -> f32 {
        self.evenness
    }

    pub fn input_triangles(&self) -> usize {
        self.alive[0] as usize
    }

    /// The fewest triangles the record reaches.
    pub fn min_triangles(&self) -> usize {
        *self.alive.last().unwrap() as usize
    }

    pub fn steps(&self) -> usize {
        self.alive.len() - 1
    }

    /// Triangles left after `step` collapses.
    pub fn triangles_at(&self, step: usize) -> usize {
        self.alive[step.min(self.steps())] as usize
    }

    /// The step whose triangle count is nearest `target`.
    pub fn step_for(&self, target: Target) -> usize {
        let t = target.triangles(self.input_triangles()) as u32;
        // `alive` is non-increasing: first step at or below the target.
        let k = self.alive.partition_point(|&a| a > t);
        if k > self.steps() {
            return self.steps();
        }
        if k > 0 && self.alive[k - 1] - t < t - self.alive[k] { k - 1 } else { k }
    }

    pub fn extract(&self, target: Target) -> Decimated {
        self.extract_step(self.step_for(target))
    }

    /// The mesh after `step` collapses: a pass over the output only.
    pub fn extract_step(&self, step: usize) -> Decimated {
        let k = step.min(self.steps()) as u32;
        let nf = self.alive[k as usize] as usize;
        let nv = self.vorder.len() - k as usize;
        let positions: Vec<Vec3> = self.vorder[..nv].par_iter().map(|&v| self.position_at(v, k)).collect();
        let triangles: Vec<[u32; 3]> = self.forder[..nf]
            .par_iter()
            .map(|&f| self.triangles[f as usize].map(|v| self.vrank[self.rep(v, k) as usize]))
            .collect();
        Decimated { mesh: TriMesh { positions, triangles }, step: k as usize }
    }

    #[inline]
    fn rep(&self, mut v: u32, k: u32) -> u32 {
        while self.rstep[v as usize] < k {
            v = self.parent[v as usize];
        }
        v
    }

    #[inline]
    fn position_at(&self, v: u32, k: u32) -> Vec3 {
        let row = self.hist.row(v as usize);
        let i = row.partition_point(|&s| s < k);
        if i == 0 { self.positions[v as usize] } else { self.step_pos[row[i - 1] as usize] }
    }
}

/// One round: tiles in parallel (`tiled = Some(round)`) or the whole mesh as one piece. Returns faces removed.
#[allow(clippy::too_many_arguments)]
fn one_round(
    st: &mut State,
    bounds: &Aabb,
    tiled: Option<usize>,
    tile_faces: usize,
    params: Params,
    tau: f64,
    floor: usize,
    alive_faces: usize,
    opts: &AnalyzeOptions,
) -> usize {
    let trace = std::env::var_os("DECIMATE_TRACE").is_some();
    let t0 = Instant::now();
    let lap = |what: &str| {
        if trace {
            eprintln!("    {what} {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
    };
    {
        let (falive, valive) = (&st.falive, &st.valive);
        st.lf.retain(|&f| falive[f as usize]);
        st.lv.retain(|&v| valive[v as usize]);
    }
    let nl = st.lf.len();
    // Tile of every live face (u32::MAX: crosses tiles).
    let (tiles, face_tile): (usize, Vec<u32>) = match tiled {
        None => (1, vec![0; nl]),
        Some(round) => {
            // Equal-count tiles: live vertices in Morton order, cut into runs. The curve's origin moves every round so
            // the seams (faces whose vertices fall in two runs) move too.
            let ext = bounds.extent().max_element().max(1e-12) * 1.5;
            let shift = Vec3::new(0.37, 0.61, 0.23) * (round as f32 + 1.0);
            let off = (shift - shift.floor()) * (ext / 3.0);
            let scale = ((1u32 << 21) - 1) as f32 / ext;
            let mut order: Vec<(u64, u32)> = st
                .lv
                .par_iter()
                .map(|&v| {
                    let q = ((st.pos[v as usize] - bounds.min + off) * scale).max(Vec3::ZERO).as_uvec3();
                    (morton(q.x) | (morton(q.y) << 1) | (morton(q.z) << 2), v)
                })
                .collect();
            order.par_sort_unstable();
            let per = (tile_faces / 2).max(64);
            let tiles = order.len().div_ceil(per).max(1);
            for (i, &(_, v)) in order.iter().enumerate() {
                st.vtile[v as usize] = (i / per) as u32;
            }
            let vtile = &st.vtile;
            let ft: Vec<u32> = st
                .lf
                .par_iter()
                .map(|&f| {
                    let t = st.faces[f as usize];
                    let c = vtile[t[0] as usize];
                    if vtile[t[1] as usize] == c && vtile[t[2] as usize] == c { c } else { u32::MAX }
                })
                .collect();
            (tiles, ft)
        }
    };
    if tiles == 0 {
        return 0;
    }
    lap("tiles");

    // Seams: every vertex of a live face that crosses tiles is locked for this round.
    let mut seam = st.locked.clone();
    for (i, &f) in st.lf.iter().enumerate() {
        if face_tile[i] == u32::MAX {
            for &v in &st.faces[f as usize] {
                seam[v as usize] = true;
            }
        }
    }
    let tile_list = Csr::from_pairs(tiles, st.lf.iter().enumerate().filter(|(i, _)| face_tile[*i] != u32::MAX).map(|(i, &f)| (face_tile[i], f)));

    let mut engines: Vec<Engine> = (0..tiles)
        .into_par_iter()
        .map(|t| {
            let gf: Vec<u32> = tile_list.row(t).to_vec();
            let mut gv: Vec<u32> = gf.iter().flat_map(|&f| st.faces[f as usize]).collect();
            gv.sort_unstable();
            gv.dedup();
            let local = |v: u32| gv.binary_search(&v).unwrap() as u32;
            let faces: Vec<[u32; 3]> = gf.iter().map(|&f| st.faces[f as usize].map(local)).collect();
            let mut e = Engine::new(
                gv.clone(),
                gv.iter().map(|&v| st.pos[v as usize]).collect(),
                gv.iter().map(|&v| st.quad[v as usize]).collect(),
                gv.iter().map(|&v| seam[v as usize]).collect(),
                gv.iter().map(|&v| st.border[v as usize]).collect(),
                gf,
                faces,
                params,
            );
            e.seed();
            e
        })
        .collect();
    lap("seed");

    let tau = if tau.is_nan() {
        let mut costs = Vec::new();
        for e in &engines {
            e.queued_costs(&mut costs);
        }
        if costs.is_empty() {
            return 0;
        }
        let i = ((costs.len() as f64 * opts.round_quantile) as usize).min(costs.len() - 1);
        *costs.select_nth_unstable_by(i, |a, b| a.total_cmp(b)).1
    } else {
        tau
    };
    let excess = alive_faces.saturating_sub(floor);
    engines.par_iter_mut().for_each(|e| {
        // Each tile may take at most its share of what is left above the floor.
        let share = if tiles == 1 { excess } else { (excess as f64 * e.faces.len() as f64 / alive_faces as f64) as usize };
        let local_floor = e.live_faces.saturating_sub(share);
        e.run(tau, local_floor);
    });
    lap("run");

    // Write back, then merge the tiles' collapses by cost into the global order.
    let mut removed = 0;
    for e in &engines {
        for (l, &g) in e.gv.iter().enumerate() {
            st.pos[g as usize] = e.pos[l];
            st.quad[g as usize] = e.quad[l];
            st.valive[g as usize] = e.alive[l];
        }
        for (l, &g) in e.gf.iter().enumerate() {
            st.faces[g as usize] = e.faces[l].map(|v| e.gv[v as usize]);
            if !e.falive[l] {
                st.falive[g as usize] = false;
                removed += 1;
            }
        }
    }
    lap("write back");
    let mut heads: BinaryHeap<(std::cmp::Reverse<OrdF64>, usize, usize)> = engines
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.recs.is_empty())
        .map(|(t, e)| (std::cmp::Reverse(OrdF64(e.recs[0].cost)), t, 0))
        .collect();
    while let Some((_, t, i)) = heads.pop() {
        let r = engines[t].recs[i];
        let s = st.recs.len() as u32;
        for d in r.dead {
            if d != u32::MAX {
                st.death[d as usize] = s;
            }
        }
        st.recs.push(r);
        if i + 1 < engines[t].recs.len() {
            heads.push((std::cmp::Reverse(OrdF64(engines[t].recs[i + 1].cost)), t, i + 1));
        }
    }
    lap("merge");
    removed
}

/// Spread the low 21 bits of `x` to every third bit.
fn morton(x: u32) -> u64 {
    let mut x = (x as u64) & 0x1f_ffff;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

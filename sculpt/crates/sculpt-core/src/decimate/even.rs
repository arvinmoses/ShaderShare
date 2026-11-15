//! Live evening of the output, after isotropic remeshing (Botsch & Kobbelt 2004; Dunyach et al. 2013): short-edge
//! collapses paired with long-edge splits (so the counts stay as set) to clear knots the decimation left, edge flips toward valence 6 (4 on borders) that also clear slivers, then tangential
//! relaxation toward the area-weighted centre of each ring, projected back onto the original surface.
//!
//! Flips never cut across a crease (the new edge may not be sharper than the old one, past a small angle, nor stray
//! farther from the original surface), never fold
//! a face, and never make an edge that already exists, so the mesh stays manifold. A vertex move that would flip or
//! crush a face around it is not taken.

use glam::Vec3;
use rayon::prelude::*;
use smallvec::SmallVec;

use super::TriMesh;
use super::surface::Surface;
use crate::mesh::Csr;

/// A flip may not make the edge bend more than the old one did, except within this (cos): near-flat pairs flip freely.
const FLIP_BEND_COS: f32 = 0.995; // about 6 degrees
/// A relaxation move is refused if a face around the vertex would turn more than this (cos), or fall below
/// [`MOVE_MIN_QUALITY`] while getting worse.
const MOVE_TURN_COS: f32 = 0.5;
const MOVE_MIN_QUALITY: f32 = 0.1;
/// A ring whose faces all lie within about 10 degrees of its normal is flat; one with a face 35 degrees off is a crease.
const FLAT_COS: f32 = 0.985;
const CREASE_COS: f32 = 0.82;
/// Knot clearing: edges shorter than (KNOT_EDGE + KNOT_EDGE_EVEN·evenness) × the mean edge are collapsed, as is the
/// short edge of a face below KNOT_QUALITY; a collapse may not leave a vertex with more than KNOT_MAX_VALENCE edges.
const KNOT_EDGE: f32 = 0.1;
const KNOT_EDGE_EVEN: f32 = 0.25;
const KNOT_QUALITY: f32 = 0.1;
const KNOT_MAX_VALENCE: usize = 10;

/// Even out `mesh` at strength `evenness` (0 does nothing): `iterations` rounds of flips (when `flips`) and one
/// relaxation step each, projected onto `surface` when given. Border vertices stay; flips keep every vertex.
pub fn even_out(mesh: &mut TriMesh, surface: Option<&Surface>, evenness: f32, iterations: usize, flips: bool) {
    let e = evenness.clamp(0.0, 1.0);
    if e <= 0.0 || iterations == 0 || mesh.triangles.is_empty() {
        return;
    }
    let mut border = border_vertices(mesh);
    // Projection reaches a few mean edges at most (a vertex never moves farther in one step).
    let reach = 4.0 * mean_edge(mesh);
    for _ in 0..iterations {
        if flips {
            let nv = mesh.positions.len();
            if clear_knots(mesh, surface, &border, e, reach) > 0 && mesh.positions.len() != nv {
                border = border_vertices(mesh);
            }
            let mut vf = vertex_faces(mesh);
            for _ in 0..3 {
                if flip_pass(mesh, &mut vf, &border, surface, e, reach) == 0 {
                    break;
                }
            }
        }
        relax_step(mesh, surface, &border, e, reach);
    }
    if let Some(s) = surface {
        fit_faces(mesh, s, &border, reach);
    }
}

/// Vertices on the surface leave the faces between them as chords, inside convex parts and outside concave ones.
/// Move each vertex along its normal by the mean signed gap between the original surface and its faces' centres,
/// so the faces sit on the surface on average (as the quadrics would place them).
fn fit_faces(m: &mut TriMesh, s: &Surface, border: &[bool], reach: f32) {
    let p = &m.positions;
    let gaps: Vec<(f32, Vec3)> = m
        .triangles
        .par_iter()
        .map(|t| {
            let q = t.map(|v| p[v as usize]);
            let n = (q[1] - q[0]).cross(q[2] - q[0]);
            let c = (q[0] + q[1] + q[2]) / 3.0;
            let g = s.closest(c, reach).map_or(0.0, |h| (h - c).dot(n.normalize_or_zero()));
            (g, n)
        })
        .collect();
    let nv = p.len();
    let vf = Csr::from_pairs(nv, m.triangles.iter().enumerate().flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))));
    let next: Vec<Vec3> = (0..nv)
        .into_par_iter()
        .map(|v| {
            let row = vf.row(v);
            if border[v] || row.is_empty() {
                return p[v];
            }
            let (mut g, mut n, mut w) = (0.0f32, Vec3::ZERO, 0.0f32);
            for &f in row {
                let (gf, nf) = gaps[f as usize];
                let a = nf.length();
                g += gf * a;
                w += a;
                n += nf;
            }
            if w <= 0.0 {
                return p[v];
            }
            p[v] + n.normalize_or_zero() * (g / w)
        })
        .collect();
    m.positions = next;
}

/// [`even_out`] without a surface to project onto and without flips: tangential relaxation alone.
pub fn relax(mesh: &mut TriMesh, evenness: f32, iterations: usize) {
    even_out(mesh, None, evenness, iterations, false);
}

fn vertex_faces(m: &TriMesh) -> Vec<SmallVec<[u32; 8]>> {
    let mut vf: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); m.positions.len()];
    for (f, t) in m.triangles.iter().enumerate() {
        for &v in t {
            vf[v as usize].push(f as u32);
        }
    }
    vf
}

/// Vertices on an edge that does not have exactly two faces (borders and non-manifold edges).
fn border_vertices(m: &TriMesh) -> Vec<bool> {
    let nv = m.positions.len();
    let vf = Csr::from_pairs(nv, m.triangles.iter().enumerate().flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))));
    (0..nv)
        .into_par_iter()
        .map(|v| {
            let mut ends: SmallVec<[u32; 32]> = SmallVec::new();
            for &f in vf.row(v) {
                for &u in &m.triangles[f as usize] {
                    if u as usize != v {
                        ends.push(u);
                    }
                }
            }
            ends.sort_unstable();
            ends.chunk_by(|a, b| a == b).any(|c| c.len() != 2)
        })
        .collect()
}

fn mean_edge(m: &TriMesh) -> f32 {
    let s: f64 = m
        .triangles
        .par_iter()
        .map(|t| {
            let p = t.map(|v| m.positions[v as usize]);
            (p[0].distance(p[1]) + p[1].distance(p[2]) + p[2].distance(p[0])) as f64
        })
        .sum();
    (s / (3 * m.triangles.len().max(1)) as f64) as f32
}

/// 4√3·area / Σ edge²: 1 for an equilateral triangle, 0 for a sliver.
fn quality(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let s = a.distance_squared(b) + b.distance_squared(c) + c.distance_squared(a);
    if s <= 0.0 { 0.0 } else { 2.0 * 3f32.sqrt() * (b - a).cross(c - a).length() / s }
}

/// One sweep of flips over every edge. Returns how many were made.
fn flip_pass(m: &mut TriMesh, vf: &mut [SmallVec<[u32; 8]>], border: &[bool], surface: Option<&Surface>, e: f32, reach: f32) -> usize {
    let p = &m.positions;
    let t = &mut m.triangles;
    let val = |vf: &[SmallVec<[u32; 8]>], v: u32| vf[v as usize].len() as i32 + border[v as usize] as i32;
    let dev = |v: u32, n: i32| {
        let want = if border[v as usize] { 4 } else { 6 };
        (n - want) * (n - want)
    };
    let mut flips = 0;
    for f in 0..t.len() {
        for i in 0..3 {
            let tri = t[f];
            let (a, b, c) = (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]);
            if border[a as usize] && border[b as usize] {
                // Could be a border edge (or a chord between two border vertices): leave it.
                continue;
            }
            // The face across edge a->b runs b->a.
            let mut across = None;
            let mut count = 0;
            for &g in &vf[a as usize] {
                if g as usize != f && t[g as usize].contains(&b) {
                    count += 1;
                    across = Some(g as usize);
                }
            }
            if count != 1 {
                continue;
            }
            let g = across.unwrap();
            let tg = t[g];
            let j = tg.iter().position(|&u| u == b).unwrap();
            if tg[(j + 1) % 3] != a {
                continue;
            }
            let d = tg[(j + 2) % 3];
            if c == d || vf[c as usize].iter().any(|&h| t[h as usize].contains(&d)) {
                continue;
            }
            let (va, vb, vc, vd) = (val(vf, a), val(vf, b), val(vf, c), val(vf, d));
            let floor = |v: u32| if border[v as usize] { 2 } else { 3 };
            if va - 1 < floor(a) || vb - 1 < floor(b) {
                continue;
            }
            let before = dev(a, va) + dev(b, vb) + dev(c, vc) + dev(d, vd);
            let after = dev(a, va - 1) + dev(b, vb - 1) + dev(c, vc + 1) + dev(d, vd + 1);
            let (pa, pb, pc, pd) = (p[a as usize], p[b as usize], p[c as usize], p[d as usize]);
            let (n1, n2) = ((pb - pa).cross(pc - pa), (pa - pb).cross(pd - pb));
            let (m1, m2) = ((pd - pa).cross(pc - pa), (pb - pd).cross(pc - pd));
            let (l1, l2) = (m1.length(), m2.length());
            if l1 <= 1e-20 || l2 <= 1e-20 {
                continue;
            }
            let old_sum = n1 + n2;
            if m1.dot(old_sum) <= 0.0 || m2.dot(old_sum) <= 0.0 {
                continue;
            }
            let cos_new = m1.dot(m2) / (l1 * l2);
            let cos_old = n1.dot(n2) / (n1.length() * n2.length()).max(1e-30);
            if cos_new < cos_old.min(FLIP_BEND_COS) {
                continue;
            }
            let q_old = quality(pa, pb, pc).min(quality(pb, pa, pd));
            let q_new = quality(pa, pd, pc).min(quality(pd, pb, pc));
            let worth = (after < before && q_new >= 0.5 * q_old) || (after == before && q_new > q_old * 1.05) || (q_new > 2.0 * q_old && after <= before + 2);
            if !worth {
                continue;
            }
            // The new edge must lie as close to the original surface as the old one (a diagonal swap on a curved
            // quad changes its shape), give or take a little that grows with evenness.
            if let Some(s) = surface {
                let (mo, mn) = ((pa + pb) * 0.5, (pc + pd) * 0.5);
                let gap = |x: Vec3| s.closest(x, reach).map_or(reach, |q| q.distance(x));
                let slack = (0.005 + 0.03 * e) * 0.5 * (pa.distance(pb) + pc.distance(pd));
                if gap(mn) > gap(mo) + slack {
                    continue;
                }
            }
            t[f] = [a, d, c];
            t[g] = [d, b, c];
            vf[a as usize].retain(|h| *h as usize != g);
            vf[b as usize].retain(|h| *h as usize != f);
            vf[c as usize].push(g as u32);
            vf[d as usize].push(f as u32);
            flips += 1;
        }
    }
    flips
}

/// One step of tangential relaxation toward the area-weighted centre of each ring, then onto the surface.
fn relax_step(m: &mut TriMesh, surface: Option<&Surface>, border: &[bool], e: f32, reach: f32) {
    let nv = m.positions.len();
    let vf = Csr::from_pairs(nv, m.triangles.iter().enumerate().flat_map(|(f, t)| t.iter().map(move |&v| (v, f as u32))));
    let p = &m.positions;
    let tris = &m.triangles;
    // Without a surface to project onto, half steps (the old behaviour) so the shape does not shrink much.
    let lambda = if surface.is_some() { e } else { 0.5 * e };
    let next: Vec<Vec3> = (0..nv)
        .into_par_iter()
        .map(|v| {
            let pv = p[v];
            if border[v] || vf.row(v).is_empty() {
                return pv;
            }
            let (mut c, mut w, mut n) = (Vec3::ZERO, 0.0f32, Vec3::ZERO);
            for &f in vf.row(v) {
                let t = tris[f as usize];
                let (a, b, cc) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                let cr = (b - a).cross(cc - a);
                let area = cr.length();
                n += cr;
                c += (a + b + cc) / 3.0 * area;
                w += area;
            }
            if w <= 0.0 {
                return pv;
            }
            let n = n.normalize_or_zero();
            // Below full evenness, vertices where the ring bends sharply (where the quadrics put them to hold a
            // ridge or a corner) move less: at 1 every vertex moves the same.
            let mut bend = 1.0f32;
            for &f in vf.row(v) {
                let t = tris[f as usize];
                let fnrm = (p[t[1] as usize] - p[t[0] as usize]).cross(p[t[2] as usize] - p[t[0] as usize]).normalize_or_zero();
                bend = bend.min(fnrm.dot(n));
            }
            let flat = ((bend - CREASE_COS) / (FLAT_COS - CREASE_COS)).clamp(0.0, 1.0);
            let step = lambda * (e + (1.0 - e) * flat);
            let d = c / w - pv;
            let mut x = pv + (d - n * n.dot(d)) * step;
            if let Some(s) = surface {
                if let Some(q) = s.closest(x, reach) {
                    x = q;
                }
            }
            // Refuse a move that turns or crushes a face around v (the ring held where it is).
            for &f in vf.row(v) {
                let t = tris[f as usize];
                let o = t.map(|u| p[u as usize]);
                let q = t.map(|u| if u as usize == v { x } else { p[u as usize] });
                let (n0, n1) = ((o[1] - o[0]).cross(o[2] - o[0]), (q[1] - q[0]).cross(q[2] - q[0]));
                let (l0, l1) = (n0.length(), n1.length());
                if l1 <= 1e-20 || n0.dot(n1) < MOVE_TURN_COS * l0 * l1 {
                    return pv;
                }
                let qn = quality(q[0], q[1], q[2]);
                if qn < MOVE_MIN_QUALITY && qn < quality(o[0], o[1], o[2]) {
                    return pv;
                }
            }
            x
        })
        .collect();
    m.positions = next;
}

/// Knots: short edges the decimation left (a collapse it refused), often bunched where thin faces meet. Flips and
/// relaxation alone cannot clear them, so collapse edges much shorter than the mean (and the short edge of a sliver)
/// to their midpoint on the surface, then split as many of the longest edges so the face and vertex counts stay
/// as set (the freed slots are reused). Returns the number of collapses.
fn clear_knots(m: &mut TriMesh, surface: Option<&Surface>, border: &[bool], e: f32, reach: f32) -> usize {
    let mean = mean_edge(m);
    let short = (KNOT_EDGE + KNOT_EDGE_EVEN * e) * mean;
    let p = &m.positions;
    // Candidates: (length, a, b) with a < b.
    let mut cand: Vec<(f32, u32, u32)> = m
        .triangles
        .par_iter()
        .flat_map_iter(|t| {
            let q = t.map(|v| p[v as usize]);
            let bad = quality(q[0], q[1], q[2]) < KNOT_QUALITY;
            let lens = [0, 1, 2].map(|i| q[i].distance(q[(i + 1) % 3]));
            let lo = (0..3).min_by(|&i, &j| lens[i].total_cmp(&lens[j])).unwrap();
            let t = *t;
            (0..3).filter_map(move |i| {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                ((lens[i] < short || (bad && i == lo && lens[i] < mean)) && a < b).then_some((lens[i], a, b))
            })
        })
        .collect();
    if cand.is_empty() {
        return 0;
    }
    cand.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
    let mut vf = vertex_faces(m);
    let mut free_v: Vec<u32> = Vec::new();
    let mut free_f: Vec<u32> = Vec::new();
    for &(_, a, b) in &cand {
        if let Some((fa, fb)) = try_collapse(m, &mut vf, border, surface, reach, a, b) {
            free_v.push(b);
            free_f.extend([fa, fb]);
        }
    }
    let n = free_v.len();
    if n == 0 {
        return 0;
    }
    // Split the longest interior edges, one per collapse.
    let p = &m.positions;
    let mut long: Vec<(f32, u32, u32)> = m
        .triangles
        .iter()
        .flat_map(|t| (0..3).map(move |i| (t[i], t[(i + 1) % 3])))
        .filter(|&(a, b)| a < b && !(border[a as usize] && border[b as usize]))
        .map(|(a, b)| (p[a as usize].distance(p[b as usize]), a, b))
        .collect();
    let k = (4 * n).min(long.len());
    if k > 0 {
        long.select_nth_unstable_by(k - 1, |x, y| y.0.total_cmp(&x.0));
        long.truncate(k);
        long.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
    }
    for &(_, a, b) in &long {
        if free_v.is_empty() {
            break;
        }
        // Still an edge with one face each way (an earlier split may have cut it).
        let fs: SmallVec<[u32; 2]> = vf[a as usize].iter().copied().filter(|&f| m.triangles[f as usize].contains(&b)).collect();
        if fs.len() != 2 {
            continue;
        }
        let v = free_v.pop().unwrap();
        let (g1, g2) = (free_f.pop().unwrap(), free_f.pop().unwrap());
        let mid = (m.positions[a as usize] + m.positions[b as usize]) * 0.5;
        m.positions[v as usize] = surface.and_then(|s| s.closest(mid, reach)).unwrap_or(mid);
        vf[v as usize].clear();
        for (f, g) in [(fs[0], g1), (fs[1], g2)] {
            let t = m.triangles[f as usize];
            // The edge's direction in this face.
            let i = (0..3).find(|&i| (t[i] == a && t[(i + 1) % 3] == b) || (t[i] == b && t[(i + 1) % 3] == a)).unwrap();
            let (x, y, c) = (t[i], t[(i + 1) % 3], t[(i + 2) % 3]);
            m.triangles[f as usize] = [x, v, c];
            m.triangles[g as usize] = [v, y, c];
            vf[y as usize].retain(|h| *h != f);
            vf[y as usize].push(g);
            vf[c as usize].push(g);
            vf[v as usize].extend([f, g]);
        }
    }
    // Collapses whose slots found no edge to split (rare): drop the dead faces and vertices.
    if !free_v.is_empty() {
        compact(m, &free_f, &free_v);
    }
    n
}

/// Collapse edge a-b to its midpoint (kept in `a`) when that is safe and leaves the faces around it no worse.
/// Returns the two faces freed.
fn try_collapse(m: &mut TriMesh, vf: &mut [SmallVec<[u32; 8]>], border: &[bool], surface: Option<&Surface>, reach: f32, a: u32, b: u32) -> Option<(u32, u32)> {
    if border[a as usize] || border[b as usize] {
        return None;
    }
    let shared: SmallVec<[u32; 2]> = vf[a as usize].iter().copied().filter(|&f| m.triangles[f as usize].contains(&b)).collect();
    if shared.len() != 2 {
        return None;
    }
    let other = |f: u32| m.triangles[f as usize].iter().copied().find(|&u| u != a && u != b).unwrap();
    let (c, d) = (other(shared[0]), other(shared[1]));
    if c == d {
        return None;
    }
    // Link condition: a and b share exactly the neighbours c and d.
    let ring = |v: u32| -> SmallVec<[u32; 16]> {
        let mut r: SmallVec<[u32; 16]> = vf[v as usize].iter().flat_map(|&f| m.triangles[f as usize]).filter(|&u| u != v).collect();
        r.sort_unstable();
        r.dedup();
        r
    };
    let (ra, rb) = (ring(a), ring(b));
    if ra.iter().filter(|u| rb.contains(u)).count() != 2 {
        return None;
    }
    let valence = ra.len() + rb.len() - 4;
    if valence > KNOT_MAX_VALENCE || vf[c as usize].len() <= 3 || vf[d as usize].len() <= 3 {
        return None;
    }
    let p = &m.positions;
    let mid = (p[a as usize] + p[b as usize]) * 0.5;
    let x = surface.and_then(|s| s.closest(mid, reach)).unwrap_or(mid);
    // Faces that stay: no turn past the limit, nothing degenerate, and the worst of them no worse than before.
    let (mut q_old, mut q_new) = (f32::MAX, f32::MAX);
    for &f in vf[a as usize].iter().chain(vf[b as usize].iter()) {
        let t = m.triangles[f as usize];
        let o = t.map(|u| p[u as usize]);
        q_old = q_old.min(quality(o[0], o[1], o[2]));
        if shared.contains(&f) {
            continue;
        }
        let q = t.map(|u| if u == a || u == b { x } else { p[u as usize] });
        let (n0, n1) = ((o[1] - o[0]).cross(o[2] - o[0]), (q[1] - q[0]).cross(q[2] - q[0]));
        let (l0, l1) = (n0.length(), n1.length());
        if l1 <= 1e-20 || (l0 > 1e-20 && n0.dot(n1) < MOVE_TURN_COS * l0 * l1) {
            return None;
        }
        q_new = q_new.min(quality(q[0], q[1], q[2]));
    }
    if q_new < q_old.min(MOVE_MIN_QUALITY) {
        return None;
    }
    // Apply: b's faces go to a; the two shared faces die (left as degenerate [a, a, a] until reused).
    let (f1, f2) = (shared[0], shared[1]);
    m.positions[a as usize] = x;
    let bf = std::mem::take(&mut vf[b as usize]);
    for &f in &bf {
        if f == f1 || f == f2 {
            continue;
        }
        for u in &mut m.triangles[f as usize] {
            if *u == b {
                *u = a;
            }
        }
        vf[a as usize].push(f);
    }
    for v in [a, c, d] {
        vf[v as usize].retain(|h| *h != f1 && *h != f2);
    }
    m.triangles[f1 as usize] = [a, a, a];
    m.triangles[f2 as usize] = [a, a, a];
    Some((f1, f2))
}

/// Remove dead faces and vertices, renumbering the rest.
fn compact(m: &mut TriMesh, dead_f: &[u32], dead_v: &[u32]) {
    let mut keep_f = vec![true; m.triangles.len()];
    dead_f.iter().for_each(|&f| keep_f[f as usize] = false);
    let mut keep_v = vec![true; m.positions.len()];
    dead_v.iter().for_each(|&v| keep_v[v as usize] = false);
    let mut map = vec![u32::MAX; m.positions.len()];
    let mut pos = Vec::with_capacity(m.positions.len());
    for (v, &k) in keep_v.iter().enumerate() {
        if k {
            map[v] = pos.len() as u32;
            pos.push(m.positions[v]);
        }
    }
    m.triangles = m.triangles.iter().zip(&keep_f).filter(|(_, k)| **k).map(|(t, _)| t.map(|v| map[v as usize])).collect();
    m.positions = pos;
}

/// Faces worth a look: an angle under 10 degrees, or an edge under a tenth of the mean edge (a knot).
pub fn bad_faces(m: &TriMesh) -> Vec<bool> {
    let short = 0.1 * mean_edge(m);
    let thin = 10f32.to_radians();
    m.triangles
        .par_iter()
        .map(|t| {
            let p = t.map(|v| m.positions[v as usize]);
            (0..3).any(|i| p[i].distance(p[(i + 1) % 3]) < short || (p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i]) < thin)
        })
        .collect()
}

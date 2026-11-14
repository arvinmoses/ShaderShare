//! Quick preview maps for a decimated mesh: a fast, rough UV layout, a height (displacement) map and an object-space
//! normal map baked from the original surface, and a tessellate-and-displace of the low mesh by those maps.
//!
//! Speed over quality throughout: charts are triangles grouped by which of six axis directions they face (on
//! smoothed normals, small charts merged into a neighbour) and projected flat on that axis, packed in shelves. Each
//! texel casts along the low mesh's smoothed normal both ways to the original surface (the nearer hit wins; the
//! closest point when neither hits) for its height and the original's normal there. Baking is meant to be run at a
//! low resolution first and again higher, each pass cancellable.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use super::{Surface, TriMesh};

const NONE: u32 = u32::MAX;

/// A low mesh with UVs: vertices split along chart seams; `source` is each vertex's index in the mesh it came from.
/// Normals are the source mesh's smoothed vertex normals, so they agree across seams.
#[derive(Clone, Debug, Default)]
pub struct UvMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub triangles: Vec<[u32; 3]>,
    pub source: Vec<u32>,
    pub charts: usize,
}

/// Area-weighted vertex normals.
pub fn vertex_normals(m: &TriMesh) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; m.positions.len()];
    for t in &m.triangles {
        let p = t.map(|v| m.positions[v as usize]);
        let f = (p[1] - p[0]).cross(p[2] - p[0]);
        t.iter().for_each(|&v| n[v as usize] += f);
    }
    n.par_iter_mut().for_each(|v| *v = v.normalize_or(Vec3::Z));
    n
}

/// The six projection axes as (normal, u, v) with u x v = normal, so faces keep their winding in UV.
const AXES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y, Vec3::Z),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::Z, Vec3::X),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y, Vec3::X),
];

fn axis_of(n: Vec3) -> usize {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        if n.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if n.y >= 0.0 { 2 } else { 3 }
    } else if n.z >= 0.0 {
        4
    } else {
        5
    }
}

/// Charts smaller than this many faces are merged into a neighbour when they project onto its axis unfolded.
const SMALL_CHART: usize = 24;

/// A rough UV layout in 0..1 with `gutter` (a fraction of the map's side) around each chart.
pub fn quick_uv(m: &TriMesh, gutter: f32) -> UvMesh {
    let nf = m.triangles.len();
    let fnrm: Vec<Vec3> = m
        .triangles
        .par_iter()
        .map(|t| {
            let p = t.map(|v| m.positions[v as usize]);
            (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero()
        })
        .collect();

    // Face neighbours across edges.
    let mut open: HashMap<(u32, u32), u32> = HashMap::with_capacity(nf * 2);
    let mut nbr = vec![[NONE; 3]; nf];
    for (f, t) in m.triangles.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            let k = (a.min(b), a.max(b));
            match open.remove(&k) {
                Some(g) => {
                    let (g, j) = ((g / 3) as usize, (g % 3) as usize);
                    nbr[f][i] = g as u32;
                    nbr[g][j] = f as u32;
                }
                None => {
                    open.insert(k, (f * 3 + i) as u32);
                }
            }
        }
    }

    // Axis per face from smoothed normals, unless the face itself would fold on that axis.
    let mut s = fnrm.clone();
    for _ in 0..4 {
        s = (0..nf)
            .into_par_iter()
            .map(|f| {
                let a = nbr[f].iter().filter(|&&g| g != NONE).fold(s[f], |a, &g| a + s[g as usize]);
                a.normalize_or(fnrm[f])
            })
            .collect();
    }
    let axis: Vec<u8> = (0..nf)
        .map(|f| {
            let a = axis_of(s[f]);
            (if fnrm[f].dot(AXES[a].0) > 0.2 { a } else { axis_of(fnrm[f]) }) as u8
        })
        .collect();

    // Connected faces of one axis make a chart.
    let mut chart = vec![NONE; nf];
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut chart_axis: Vec<u8> = Vec::new();
    let mut stack = Vec::new();
    for f0 in 0..nf {
        if chart[f0] != NONE {
            continue;
        }
        let c = faces.len() as u32;
        let mut list = Vec::new();
        chart[f0] = c;
        stack.push(f0 as u32);
        while let Some(f) = stack.pop() {
            list.push(f);
            for &g in &nbr[f as usize] {
                if g != NONE && chart[g as usize] == NONE && axis[g as usize] == axis[f0] {
                    chart[g as usize] = c;
                    stack.push(g);
                }
            }
        }
        faces.push(list);
        chart_axis.push(axis[f0]);
    }

    // Merge small charts into the biggest neighbour they project onto without folding.
    let mut order: Vec<usize> = (0..faces.len()).filter(|&c| faces[c].len() < SMALL_CHART).collect();
    order.sort_by_key(|&c| faces[c].len());
    for c in order {
        if faces[c].is_empty() || faces[c].len() >= SMALL_CHART {
            continue;
        }
        let mut best: Option<usize> = None;
        for &f in &faces[c] {
            for &g in &nbr[f as usize] {
                if g == NONE {
                    continue;
                }
                let d = chart[g as usize] as usize;
                if d == c || best.is_some_and(|b| faces[b].len() >= faces[d].len()) {
                    continue;
                }
                let n = AXES[chart_axis[d] as usize].0;
                if faces[c].iter().all(|&h| fnrm[h as usize].dot(n) > 0.05) {
                    best = Some(d);
                }
            }
        }
        if let Some(d) = best {
            let moved = std::mem::take(&mut faces[c]);
            for &f in &moved {
                chart[f as usize] = d as u32;
            }
            faces[d].extend(moved);
        }
    }
    let charts: Vec<(Vec<u32>, u8)> = faces.into_iter().zip(chart_axis).filter(|(f, _)| !f.is_empty()).collect();

    // Split vertices per chart and project.
    let vn = vertex_normals(m);
    let mut out = UvMesh { charts: charts.len(), ..Default::default() };
    let mut stamp = vec![NONE; m.positions.len()];
    let mut idx = vec![0u32; m.positions.len()];
    let mut ranges = Vec::with_capacity(charts.len());
    for (ci, (list, a)) in charts.iter().enumerate() {
        let (_, u, v) = AXES[*a as usize];
        let start = out.positions.len();
        for &f in list {
            let t = m.triangles[f as usize];
            out.triangles.push(t.map(|s| {
                if stamp[s as usize] != ci as u32 {
                    stamp[s as usize] = ci as u32;
                    idx[s as usize] = out.positions.len() as u32;
                    let p = m.positions[s as usize];
                    out.positions.push(p);
                    out.normals.push(vn[s as usize]);
                    out.uvs.push(Vec2::new(p.dot(u), p.dot(v)));
                    out.source.push(s);
                }
                idx[s as usize]
            }));
        }
        let (lo, hi) = out.uvs[start..].iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        ranges.push((start..out.positions.len(), lo, hi - lo));
    }

    // Shelf packing, tallest first, a few widths tried for the squarest result.
    let mut by_height: Vec<usize> = (0..ranges.len()).collect();
    by_height.sort_by(|&a, &b| ranges[b].2.y.total_cmp(&ranges[a].2.y));
    let area: f32 = ranges.iter().map(|r| r.2.x * r.2.y).sum();
    let widest = ranges.iter().map(|r| r.2.x).fold(0.0f32, f32::max);
    let pack = |side: f32| {
        let g = gutter * side;
        let width = side.max(widest + 2.0 * g);
        let (mut x, mut y, mut shelf) = (0.0f32, 0.0f32, 0.0f32);
        let mut at = vec![Vec2::ZERO; ranges.len()];
        for &c in &by_height {
            let s = ranges[c].2 + Vec2::splat(2.0 * g);
            if x + s.x > width && x > 0.0 {
                y += shelf;
                x = 0.0;
                shelf = 0.0;
            }
            at[c] = Vec2::new(x + g, y + g);
            x += s.x;
            shelf = shelf.max(s.y);
        }
        (width.max(y + shelf), at)
    };
    let mut guess = area.sqrt().max(1e-9) * 1.15;
    let mut best = pack(guess);
    for _ in 0..8 {
        guess *= ((best.0 / guess).sqrt()).clamp(0.9, 1.2);
        let p = pack(guess);
        if p.0 < best.0 {
            best = p;
        }
    }
    let (side, at) = best;
    for (c, (r, lo, _)) in ranges.iter().enumerate() {
        for uv in &mut out.uvs[r.clone()] {
            *uv = (*uv - *lo + at[c]) / side;
        }
    }
    out
}

/// Baked maps, `res` square, row 0 at v = 0: height along the low mesh's normal, and the original's object-space
/// normal packed as `n * 0.5 + 0.5` in RGB.
#[derive(Clone, Debug)]
pub struct Maps {
    pub res: usize,
    pub height: Vec<f32>,
    pub normal: Vec<[u8; 4]>,
    /// How far each texel searched for the original surface.
    pub max_dist: f32,
}

impl Maps {
    fn texel(&self, uv: Vec2) -> (usize, usize, usize, usize, Vec2) {
        let r = self.res as f32;
        let p = (uv * r - 0.5).clamp(Vec2::ZERO, Vec2::splat(r - 1.0));
        let (x, y) = (p.x as usize, p.y as usize);
        (x, y, (x + 1).min(self.res - 1), (y + 1).min(self.res - 1), p - Vec2::new(x as f32, y as f32))
    }

    /// Bilinear height at `uv`.
    pub fn height_at(&self, uv: Vec2) -> f32 {
        let (x0, y0, x1, y1, f) = self.texel(uv);
        let h = |x, y| self.height[y * self.res + x];
        let a = h(x0, y0) + (h(x1, y0) - h(x0, y0)) * f.x;
        let b = h(x0, y1) + (h(x1, y1) - h(x0, y1)) * f.x;
        a + (b - a) * f.y
    }

    /// Bilinear normal at `uv`, unpacked and normalised.
    pub fn normal_at(&self, uv: Vec2) -> Vec3 {
        let (x0, y0, x1, y1, f) = self.texel(uv);
        let n = |x, y| {
            let c: [u8; 4] = self.normal[y * self.res + x];
            Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 127.5 - 1.0
        };
        let a = n(x0, y0).lerp(n(x1, y0), f.x);
        let b = n(x0, y1).lerp(n(x1, y1), f.x);
        a.lerp(b, f.y).normalize_or(Vec3::Z)
    }
}

fn pack_normal(n: Vec3) -> [u8; 4] {
    let c = ((n * 0.5 + 0.5) * 255.0).round().clamp(Vec3::ZERO, Vec3::splat(255.0));
    [c.x as u8, c.y as u8, c.z as u8, 255]
}

/// Bakes maps from one original surface; holds the original's vertex normals.
pub struct Baker {
    surface: Arc<Surface>,
    normals: Vec<Vec3>,
}

impl Baker {
    pub fn new(surface: Arc<Surface>) -> Baker {
        let normals = vertex_normals(surface.mesh());
        Baker { surface, normals }
    }

    /// The original's smoothed normal on triangle `f` at weights `b` of its corners 1 and 2.
    fn normal_on(&self, f: u32, b: Vec2) -> Vec3 {
        let t = self.surface.mesh().triangles[f as usize];
        let n = self.normals[t[0] as usize] * (1.0 - b.x - b.y) + self.normals[t[1] as usize] * b.x + self.normals[t[2] as usize] * b.y;
        n.normalize_or(Vec3::Z)
    }

    /// Height along `n` from `p` to the original surface and the original's normal there.
    fn sample(&self, p: Vec3, n: Vec3, max_dist: f32) -> Option<(f32, Vec3)> {
        let out = self.surface.ray(p, n, max_dist);
        let inn = self.surface.ray(p, -n, out.map_or(max_dist, |h| h.t));
        match (out, inn) {
            (_, Some(h)) => Some((-h.t, self.normal_on(h.face, h.bary))),
            (Some(h), None) => Some((h.t, self.normal_on(h.face, h.bary))),
            (None, None) => {
                let (q, f) = self.surface.closest_hit(p, max_dist)?;
                let t = self.surface.mesh().triangles[f as usize];
                let b = barycentric(q, t.map(|v| self.surface.mesh().positions[v as usize]));
                Some(((q - p).dot(n), self.normal_on(f, b)))
            }
        }
    }

    /// Bakes `uv`'s maps at `res`, searching up to `max_dist` from the low surface. `None` if cancelled.
    pub fn bake(&self, uv: &UvMesh, res: usize, max_dist: f32, cancel: Option<&AtomicBool>) -> Option<Maps> {
        let n = res * res;
        let r = res as f32;
        // Which triangle each texel centre is on, with texels up to a texel outside a triangle taking its plane.
        let mut face = vec![NONE; n];
        let mut bary = vec![Vec2::ZERO; n];
        let mut dist = vec![f32::INFINITY; n];
        for (f, t) in uv.triangles.iter().enumerate() {
            let p = t.map(|v| uv.uvs[v as usize] * r);
            let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
            if area.abs() < 1e-12 {
                continue;
            }
            let el = [p[1].distance(p[2]).max(1e-9), p[2].distance(p[0]).max(1e-9), p[0].distance(p[1]).max(1e-9)];
            let lo = (p[0].min(p[1]).min(p[2]) - 1.0).floor().max(Vec2::ZERO);
            let hi = (p[0].max(p[1]).max(p[2]) + 1.0).ceil().min(Vec2::splat(r - 1.0));
            if lo.x > hi.x || lo.y > hi.y {
                continue;
            }
            for y in lo.y as usize..=hi.y as usize {
                for x in lo.x as usize..=hi.x as usize {
                    let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = (p[1] - c).perp_dot(p[2] - c) / area;
                    let w1 = (p[2] - c).perp_dot(p[0] - c) / area;
                    let w2 = 1.0 - w0 - w1;
                    let d = [w0, w1, w2].iter().zip(el).map(|(w, e)| -w * area.abs() / e).fold(0.0f32, f32::max);
                    let i = y * res + x;
                    if d <= 1.0 && d < dist[i] {
                        dist[i] = d;
                        face[i] = f as u32;
                        bary[i] = Vec2::new(w1, w2);
                    }
                }
            }
        }
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        if cancelled() {
            return None;
        }

        let mut height = vec![f32::NAN; n];
        let mut normal = vec![[0u8; 4]; n];
        height.par_chunks_mut(res).zip(normal.par_chunks_mut(res)).enumerate().for_each(|(y, (hr, nr))| {
            if cancelled() {
                return;
            }
            for x in 0..res {
                let i = y * res + x;
                let f = face[i];
                if f == NONE {
                    continue;
                }
                let t = uv.triangles[f as usize];
                let b = bary[i];
                let w = [1.0 - b.x - b.y, b.x, b.y];
                let p: Vec3 = (0..3).map(|k| uv.positions[t[k] as usize] * w[k]).sum();
                let nn = (0..3).map(|k| uv.normals[t[k] as usize] * w[k]).sum::<Vec3>().normalize_or_zero();
                if nn == Vec3::ZERO {
                    continue;
                }
                let (h, hn) = self.sample(p, nn, max_dist).unwrap_or((0.0, nn));
                hr[x] = h;
                nr[x] = pack_normal(hn);
            }
        });
        if cancelled() {
            return None;
        }
        let passes = (res / 128).clamp(2, 16);
        dilate(res, &mut height, &mut normal, passes);
        Some(Maps { res, height, normal, max_dist })
    }
}

/// Weights of `p`'s corners 1 and 2 in triangle `t` (p on or near its plane).
fn barycentric(p: Vec3, t: [Vec3; 3]) -> Vec2 {
    let (e1, e2, d) = (t[1] - t[0], t[2] - t[0], p - t[0]);
    let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
    let (d1, d2) = (d.dot(e1), d.dot(e2));
    let den = d11 * d22 - d12 * d12;
    if den.abs() < 1e-30 {
        return Vec2::ZERO;
    }
    Vec2::new((d22 * d1 - d12 * d2) / den, (d11 * d2 - d12 * d1) / den)
}

/// Grows the baked texels into the empty ones around them `passes` times (the mean of filled neighbours), then fills
/// what is left with zero height and a neutral normal.
fn dilate(res: usize, height: &mut [f32], normal: &mut [[u8; 4]], passes: usize) {
    for _ in 0..passes {
        let (h0, n0) = (height.to_vec(), normal.to_vec());
        let changed: usize = height
            .par_chunks_mut(res)
            .zip(normal.par_chunks_mut(res))
            .enumerate()
            .map(|(y, (hr, nr))| {
                let mut changed = 0;
                for x in 0..res {
                    if !hr[x].is_nan() {
                        continue;
                    }
                    let (mut h, mut nv, mut k) = (0.0f32, Vec3::ZERO, 0);
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                            if xx < 0 || yy < 0 || xx >= res as i32 || yy >= res as i32 {
                                continue;
                            }
                            let j = yy as usize * res + xx as usize;
                            if !h0[j].is_nan() {
                                h += h0[j];
                                let c = n0[j];
                                nv += Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32);
                                k += 1;
                            }
                        }
                    }
                    if k > 0 {
                        hr[x] = h / k as f32;
                        let c = (nv / k as f32).round();
                        nr[x] = [c.x as u8, c.y as u8, c.z as u8, 255];
                        changed += 1;
                    }
                }
                changed
            })
            .sum();
        if changed == 0 {
            break;
        }
    }
    for (h, n) in height.iter_mut().zip(normal.iter_mut()) {
        if h.is_nan() {
            *h = 0.0;
            *n = [128, 128, 255, 255];
        }
    }
}

/// A tessellated, displaced mesh, welded across UV seams, with each triangle's corner UVs.
pub struct Tessellated {
    pub mesh: TriMesh,
    pub corner_uvs: Vec<[Vec2; 3]>,
}

/// Splits every triangle of `uv` into `level`² and moves each new vertex along the smoothed normal by the height map
/// times `scale`. Vertices on seams are shared by both sides and take the mean of their heights, so there are no
/// cracks.
pub fn tessellate(uv: &UvMesh, maps: &Maps, level: usize, scale: f32) -> Tessellated {
    let l = level.max(1);
    let ns = uv.source.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let mut src_pos = vec![Vec3::ZERO; ns];
    let mut src_nrm = vec![Vec3::Z; ns];
    for (v, &s) in uv.source.iter().enumerate() {
        src_pos[s as usize] = uv.positions[v];
        src_nrm[s as usize] = uv.normals[v];
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::with_capacity(uv.triangles.len() * 2);
    for t in &uv.triangles {
        for i in 0..3 {
            let (a, b) = (uv.source[t[i] as usize], uv.source[t[(i + 1) % 3] as usize]);
            let n = edges.len() as u32;
            edges.entry((a.min(b), a.max(b))).or_insert(n);
        }
    }
    let per_edge = l - 1;
    let per_face = if l >= 3 { (l - 1) * (l - 2) / 2 } else { 0 };
    let edge_base = ns;
    let face_base = edge_base + edges.len() * per_edge;
    let total = face_base + uv.triangles.len() * per_face;
    let mut base = vec![Vec3::ZERO; total];
    let mut nrm = vec![Vec3::Z; total];
    let mut hsum = vec![0.0f32; total];
    let mut hcount = vec![0u32; total];
    let mut tris = Vec::with_capacity(uv.triangles.len() * l * l);
    let mut corner_uvs = Vec::with_capacity(uv.triangles.len() * l * l);

    // Local (i, j) grid index -> vertex, uv; i towards corner 1, j towards corner 2.
    let grid = |i: usize, j: usize| i * (l + 1) + j;
    let mut vid = vec![0u32; (l + 1) * (l + 1)];
    let mut vuv = vec![Vec2::ZERO; (l + 1) * (l + 1)];
    for (f, t) in uv.triangles.iter().enumerate() {
        let s = t.map(|v| uv.source[v as usize]);
        let u = t.map(|v| uv.uvs[v as usize]);
        let mut interior = 0;
        for i in 0..=l {
            for j in 0..=l - i {
                let w = [(l - i - j) as f32 / l as f32, i as f32 / l as f32, j as f32 / l as f32];
                // On an edge (a, b) at step k of l from a: the shared vertex, placed from the lower source index.
                let on_edge = |a: u32, b: u32, k: usize| -> usize {
                    let (lo, hi, k) = if a < b { (a, b, k) } else { (b, a, l - k) };
                    edge_base + edges[&(lo, hi)] as usize * per_edge + (k - 1)
                };
                let (v, placed) = if i == 0 && j == 0 {
                    (s[0] as usize, None)
                } else if i == l {
                    (s[1] as usize, None)
                } else if j == l {
                    (s[2] as usize, None)
                } else if j == 0 {
                    (on_edge(s[0], s[1], i), Some((s[0], s[1], i)))
                } else if i + j == l {
                    (on_edge(s[1], s[2], j), Some((s[1], s[2], j)))
                } else if i == 0 {
                    (on_edge(s[0], s[2], j), Some((s[0], s[2], j)))
                } else {
                    interior += 1;
                    (face_base + f * per_face + interior - 1, None)
                };
                if hcount[v] == 0 {
                    let (p, n) = if v < ns {
                        (src_pos[v], src_nrm[v])
                    } else if let Some((a, b, k)) = placed {
                        let (lo, hi, k) = if a < b { (a, b, k) } else { (b, a, l - k) };
                        let x = k as f32 / l as f32;
                        (src_pos[lo as usize].lerp(src_pos[hi as usize], x), src_nrm[lo as usize].lerp(src_nrm[hi as usize], x))
                    } else {
                        (
                            (0..3).map(|k| src_pos[s[k] as usize] * w[k]).sum(),
                            (0..3).map(|k| src_nrm[s[k] as usize] * w[k]).sum(),
                        )
                    };
                    base[v] = p;
                    nrm[v] = n.normalize_or(Vec3::Z);
                }
                let tuv = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
                hsum[v] += maps.height_at(tuv);
                hcount[v] += 1;
                vid[grid(i, j)] = v as u32;
                vuv[grid(i, j)] = tuv;
            }
        }
        for i in 0..l {
            for j in 0..l - i {
                let a = [grid(i, j), grid(i + 1, j), grid(i, j + 1)];
                tris.push(a.map(|g| vid[g]));
                corner_uvs.push(a.map(|g| vuv[g]));
                if i + j + 1 < l {
                    let b = [grid(i + 1, j), grid(i + 1, j + 1), grid(i, j + 1)];
                    tris.push(b.map(|g| vid[g]));
                    corner_uvs.push(b.map(|g| vuv[g]));
                }
            }
        }
    }
    let positions = (0..total).map(|v| if hcount[v] == 0 { base[v] } else { base[v] + nrm[v] * (hsum[v] / hcount[v] as f32 * scale) }).collect();
    Tessellated { mesh: TriMesh { positions, triangles: tris }, corner_uvs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::quad_sphere_res;

    #[test]
    fn sphere_bakes_its_own_radius() {
        // A coarse sphere baked against a fine one: heights make up the chord gap, tessellation lands on the sphere.
        let fine = Arc::new(TriMesh::from_poly(&quad_sphere_res(40, 1.0)));
        let coarse = TriMesh::from_poly(&quad_sphere_res(4, 1.0));
        let uv = quick_uv(&coarse, 4.0 / 512.0);
        assert!(uv.uvs.iter().all(|p| (0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y)));
        let baker = Baker::new(Arc::new(Surface::new(fine)));
        let maps = baker.bake(&uv, 512, 0.3, None).unwrap();
        let t = tessellate(&uv, &maps, 6, 1.0);
        let check = super::super::check_manifold(t.mesh.positions.len(), &t.mesh.triangles);
        assert!(check.closed() && check.manifold(), "{check:?}");
        let worst = t.mesh.positions.iter().map(|p| (p.length() - 1.0).abs()).fold(0.0f32, f32::max);
        let before = coarse.triangles.iter().map(|t| (t.map(|v| coarse.positions[v as usize]).iter().sum::<Vec3>() / 3.0).length()).fold(1.0f32, f32::min);
        assert!(worst < 0.01, "worst {worst}, flat faces dip to {before}");
    }
}

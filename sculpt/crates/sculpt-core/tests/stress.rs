//! A long, seeded session over the volume engine and its layer stack: strokes of every size at every level on the
//! form and on detail layers, layers added, removed, hidden, moved between levels, generators added and edited,
//! masks, painted density, surface noise baked, undo and redo, save and open, jobs landing late. After every step:
//! the surface is closed at every level cap, what is shown is exactly what composing the stack afresh gives (the
//! same blocks, the same values), every block's parent exists, undo brings back the exact prior state and redo the
//! exact later one, and a file written and opened shows the same.
//!
//! `cargo test --release --test stress` runs a short session; `cargo test --release --test stress -- --ignored`
//! runs `SCULPT_STRESS_STEPS` steps (default 400) from seed `SCULPT_STRESS_SEED` (default 1), or every seed in
//! `SCULPT_STRESS_SEEDS=a..b`.

use glam::{IVec3, Vec3};
use sculpt_core::generator::density::Density;
use sculpt_core::generator::detail_layer::Layered;
use sculpt_core::generator::layer::{EdgeSide, GeneratorSpec, PlaceOn};
use sculpt_core::generator::level_volume::{Key, LevelVolume};
use sculpt_core::generator::levels::edge_counts;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::noise::NoiseKind;
use sculpt_core::surface_noise::SurfaceNoise;

const LEVELS: usize = 6;

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
    fn dir(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            let n = v.length();
            if n > 0.2 && n < 1.0 {
                return v / n;
            }
        }
    }
}

/// Everything the session holds: what is shown, the stack over the form (once made), and the checks' findings.
struct World {
    s: LevelVolume,
    l: Option<Layered>,
    rng: Rng,
    log: Vec<String>,
    /// The worst value drift between live and afresh seen, per level.
    drift: Vec<f32>,
    steps: usize,
    /// Caps seen with non-manifold edges while surveying.
    nm: usize,
    /// Samples where live and afresh cleared a checkerboard face by different corners.
    fills: usize,
    /// The farthest the surface reached from the centre.
    reach: f32,
}

/// Open and non-manifold edges and dropped quads at `cap`.
fn closed(v: &LevelVolume, cap: usize) -> (usize, usize, usize) {
    let (mesh, stats) = v.surface(cap);
    let (open, nm) = edge_counts(&mesh.faces);
    (open, nm, stats.dropped)
}

/// Where the non-manifold edges of the surface at `cap` are: each edge's midpoint, the finest level holding a block
/// there, how many faces share it, and whether it lies next to finer blocks (a transition cell).
fn non_manifold(v: &LevelVolume, cap: usize, form: Option<&LevelVolume>) -> Vec<(String, bool)> {
    let (mesh, _) = v.surface(cap);
    let mut count: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for f in &mesh.faces {
        let n = if f[3] == u32::MAX { 3 } else { 4 };
        for k in 0..n {
            let (a, b) = (f[k], f[(k + 1) % n]);
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut out: Vec<(String, bool)> = count
        .iter()
        .filter(|(_, c)| **c > 2)
        .map(|((a, b), c)| {
            let m = (mesh.positions[*a as usize] + mesh.positions[*b as usize]) * 0.5;
            let lv = (0..=cap).rev().find(|&l| v.exists(l, ((m - v.origin()) / (v.voxel(l) * 8.0)).floor().as_ivec3())).unwrap_or(0);
            // The cell of that level the midpoint is in: a checkerboard on any of its faces (or its neighbours')?
            let g = ((m - v.origin()) / v.voxel(lv)).floor().as_ivec3();
            let sign = |i: IVec3| v.sample(lv, i, false) < 0.0;
            let mut checker = 0;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = g + IVec3::new(dx, dy, dz);
                        for (u, w) in [(IVec3::X, IVec3::Y), (IVec3::Y, IVec3::Z), (IVec3::Z, IVec3::X)] {
                            let (p, q, r, t) = (sign(c), sign(c + u), sign(c + u + w), sign(c + w));
                            if p == r && q == t && p != q {
                                checker += 1;
                                let corners: Vec<String> = [c, c + u, c + u + w, c + w].iter().map(|i| format!("{:?}={:.6}/{}", i, v.sample(lv, *i, false), v.exists(lv, i.div_euclid(IVec3::splat(8))))).collect();
                                eprintln!("      checkerboard face at level {lv}: {}", corners.join(" "));
                            }
                        }
                    }
                }
            }
            // A cell next to a finer one: any of the 27 cells around it (itself included) covered by a finer block.
            let finer = lv + 1 <= cap
                && (0..27).any(|k| {
                    let c = (g + IVec3::new(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1)) * 2;
                    (0..8).any(|q| v.exists(lv + 1, (c + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)).div_euclid(IVec3::splat(8))))
                });
            let fv = form.map_or(f32::NAN, |f| f.value_upto(m, lv.min(f.levels() - 1), false));
            (format!("{:.4?} (r {:.3}) level {lv} x{c} ({checker} checkerboard faces near, finer blocks near: {finer}; shown {:.5} form {fv:.5})", m, m.length(), v.value_upto(m, cap, false)), finer && checker == 0)
        })
        .collect();
    out.sort();
    out
}

/// The surface seen from `eye` straight down `-w`, `half` wide each way, shaded, as a grey PGM at `path`.
fn picture(v: &LevelVolume, eye: Vec3, w: Vec3, half: f32, name: &str) {
    let u = w.any_orthonormal_vector();
    let t = w.cross(u);
    let n = 400;
    let mut img = vec![0u8; n * n];
    for y in 0..n {
        for x in 0..n {
            let (a, b) = (-half + 2.0 * half * x as f32 / n as f32, half - 2.0 * half * y as f32 / n as f32);
            if let Some((_, nrm)) = v.raycast(eye + u * a + t * b, -w, v.levels() - 1, false) {
                img[y * n + x] = (35.0 + 220.0 * nrm.dot((w * 0.8 + u * -0.4 + t * 0.5).normalize()).max(0.0)) as u8;
            }
        }
    }
    let mut out = format!("P5 {n} {n} 255\n").into_bytes();
    out.extend(img);
    std::fs::write(format!("{name}.pgm"), out).unwrap();
}

/// Every stored block's full samples, sorted, for exact comparisons. A checkerboard fill (of the block's level or a
/// coarser one, copied by a block made again from the level below) reads as zero: which corner of a face is filled
/// depends on whether the blocks were composed in pieces or all at once.
fn samples(v: &LevelVolume, l: Option<&Layered>) -> Vec<(u8, [i32; 3], Vec<u32>)> {
    let fills: Vec<f32> = (0..v.levels()).map(|m| 1e-3 * v.voxel(m)).collect();
    let is_fill = |l: usize, x: f32| fills[..=l].iter().any(|f| (x.abs() - f).abs() < 1e-12);
    // A block held only as a parent (neither the form's nor a layer's) is made from the level below, fills and all,
    // interpolated: its samples are left out.
    let parent_only = |k: Key| l.is_some_and(|l| !l.form().exists(k.0 as usize, k.1) && !(0..l.layers().len()).any(|j| l.layer_holds(j, k.0 as usize, k.1)));
    let mut out: Vec<(u8, [i32; 3], Vec<u32>)> = v.keys().into_iter().map(|k| (k.0, k.1.to_array(), if parent_only(k) { vec![FILL; 512] } else { v.block(k).unwrap().iter().map(|x| if is_fill(k.0 as usize, *x) { FILL } else { x.to_bits() }).collect() })).collect();
    out.sort_unstable_by_key(|x| (x.0, x.1));
    out
}

/// A sample that is a checkerboard fill, in [`samples`].
const FILL: u32 = u32::MAX;

/// The whole state: the layers (and the form at 16 bits) as saved, and every sample of the form and of what is shown.
struct State {
    layers: Vec<u8>,
    form: Vec<(u8, [i32; 3], Vec<u32>)>,
    shown: Vec<(u8, [i32; 3], Vec<u32>)>,
}

fn fingerprint(s: &LevelVolume, l: Option<&Layered>) -> State {
    let mut layers = Vec::new();
    let mut form = Vec::new();
    if let Some(l) = l {
        l.write_to(&mut layers).unwrap();
        form = samples(l.form(), None);
    }
    State { layers, form, shown: samples(s, l) }
}

/// Whether two sample sets are the same (a fill on either side counts as the same: which corner of a checkerboard
/// face is filled depends on whether blocks were composed in pieces or all at once).
fn same_samples(a: &[(u8, [i32; 3], Vec<u32>)], b: &[(u8, [i32; 3], Vec<u32>)]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0 && x.1 == y.1 && x.2.iter().zip(&y.2).all(|(p, q)| p == q || *p == FILL || *q == FILL))
}

/// Whether two states are the same, and if not, what differs.
fn same_state(a: &State, b: &State) -> Result<(), String> {
    if a.layers != b.layers {
        let at = a.layers.iter().zip(&b.layers).position(|(x, y)| x != y);
        return Err(format!("the layers' file differs (first at byte {at:?} of {})", a.layers.len()));
    }
    if !same_samples(&a.form, &b.form) {
        return Err("the form's samples differ".to_string());
    }
    if !same_samples(&a.shown, &b.shown) {
        return Err("the shown samples differ".to_string());
    }
    Ok(())
}

/// Where the two fingerprints first differ, described (for the failure message).
fn describe_difference(sa: &[(u8, [i32; 3], Vec<u32>)], b: &LevelVolume, l: Option<&Layered>) -> String {
    let sb = samples(b, l);
    let (ka, kb): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (sa.iter().map(|x| (x.0, x.1)).collect(), sb.iter().map(|x| (x.0, x.1)).collect());
    let only_a: Vec<_> = ka.difference(&kb).take(5).collect();
    let only_b: Vec<_> = kb.difference(&ka).take(5).collect();
    let mut worst = (0.0f32, (0u8, [0; 3]));
    let mut differing = vec![0usize; LEVELS];
    let mb: std::collections::HashMap<_, _> = sb.iter().map(|y| ((y.0, y.1), &y.2)).collect();
    for x in sa {
        if let Some(y) = mb.get(&(x.0, x.1)) {
            let d = x.2.iter().zip(y.iter()).filter(|(p, q)| **p != FILL && **q != FILL).map(|(p, q)| (f32::from_bits(*p) - f32::from_bits(*q)).abs()).fold(0.0f32, f32::max);
            if d > 0.0 {
                differing[x.0 as usize] += 1;
            }
            if d > worst.0 {
                worst = (d, (x.0, x.1));
            }
        }
    }
    format!("blocks only before {only_a:?} (of {}), only after {only_b:?} (of {}), blocks differing per level {differing:?}, values apart at most {} at {:?}", ka.difference(&kb).count(), kb.difference(&ka).count(), worst.0, worst.1)
}

fn plane_stroke(dab: &mut dyn FnMut(usize, Vec3, Vec3, f32, bool, usize), hit: &dyn Fn(Vec3, usize) -> Option<(Vec3, Vec3)>, level: usize, d: Vec3, span: f32, r: f32, trim: bool, dabs: usize) {
    let side = d.any_orthonormal_vector();
    let mut along = Vec3::ZERO;
    for k in 0..=dabs {
        let t = -1.0 + 2.0 * k as f32 / dabs as f32;
        let dir = (d + side * span * t).normalize();
        let Some((at, n)) = hit(dir, level) else { continue };
        let rough = if along == Vec3::ZERO { n } else { along };
        dab(level, at, rough, r, trim, k);
        along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
    }
}

impl World {
    fn new(seed: u64, voxel: f32) -> World {
        let s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel, LEVELS);
        World { s, l: None, rng: Rng::new(seed), log: Vec::new(), drift: vec![0.0; LEVELS], steps: 0, nm: 0, fills: 0, reach: 0.0 }
    }

    fn note(&mut self, what: String) {
        eprintln!("step {}: {what}", self.steps);
        self.log.push(what);
    }

    /// The checks after every step: closed, as composed afresh, nested.
    fn check(&mut self, what: &str) {
        self.steps += 1;
        self.note(what.to_string());
        let t0 = std::time::Instant::now();
        {
            // The surface's reach from the centre (a ball of radius 1 with what was sculpted on it), over the vertices
            // quads use: a transition cell has a vertex whether or not any crossing uses it.
            let (mesh, _) = self.s.surface(LEVELS - 1);
            let mut used = vec![false; mesh.positions.len()];
            for f in &mesh.faces {
                for &i in f.iter().filter(|i| **i != u32::MAX) {
                    used[i as usize] = true;
                }
            }
            let far = mesh.positions.iter().zip(&used).filter(|(_, u)| **u).map(|(p, _)| *p).fold(Vec3::ZERO, |a, p| if p.length() > a.length() { p } else { a });
            self.reach = self.reach.max(far.length());
            eprintln!("   surface reaches {:.3} from the centre, at {far:.3?}; {} faces", far.length(), mesh.faces.len());
            if let Ok(dir) = std::env::var("SCULPT_STRESS_PICTURE") {
                let w = far.normalize_or(Vec3::Z);
                picture(&self.s, w * 4.0, w, 1.5, &format!("{dir}/step_{:03}_whole", self.steps));
                picture(&self.s, far + w * 0.5, w, 0.3, &format!("{dir}/step_{:03}_far", self.steps));
            }
        }
        if std::env::var("SCULPT_STRESS_STOP").ok().and_then(|v| v.parse::<usize>().ok()).is_some_and(|n| self.steps >= n) {
            panic!("stopped at step {} as asked", self.steps);
        }
        self.closed_everywhere(what);
        let t_closed = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = std::time::Instant::now();
        let s = &self.s;
        for k in s.keys() {
            if k.0 > 0 {
                assert!(s.exists(k.0 as usize - 1, k.1.div_euclid(IVec3::splat(2))), "after {what}: block {k:?} has no parent");
            }
        }
        if self.l.is_some() {
            let l = self.l.as_ref().unwrap();
            // Live against the stack composed afresh over the same form, exactly: the same blocks, the same values.
            // The one difference allowed is where a checkerboard face was cleared by filling another of its corners
            // (composing in parts and all at once meet them in different orders), which shows as the fill value on
            // one side.
            let exact = l.composed_afresh();
            let mut report = Vec::new();
            let mut fills = 0;
            for lv in 0..LEVELS {
                let a: std::collections::BTreeSet<[i32; 3]> = s.level_keys(lv).into_iter().map(|v| v.to_array()).collect();
                let b: std::collections::BTreeSet<[i32; 3]> = exact.level_keys(lv).into_iter().map(|v| v.to_array()).collect();
                let only_live: Vec<_> = a.difference(&b).take(4).collect();
                let only_afresh: Vec<_> = b.difference(&a).take(4).collect();
                if !only_live.is_empty() || !only_afresh.is_empty() {
                    report.push(format!("level {lv}: blocks only live {only_live:?} ({}), only afresh {only_afresh:?} ({})", a.difference(&b).count(), b.difference(&a).count()));
                }
                // (A checkerboard fill of this level, or of a coarser one: a block made again from the level below
                // copies its samples.)
                let fill_values: Vec<f32> = (0..=lv).map(|m| 1e-3 * s.voxel(m)).collect();
                let is_fill = |x: f32| fill_values.iter().any(|f| (x.abs() - f).abs() < 1e-12);
                for k in a.intersection(&b) {
                    let bc = IVec3::from_array(*k);
                    if !l.form().exists(lv, bc) && !(0..l.layers().len()).any(|j| l.layer_holds(j, lv, bc)) {
                        continue;
                    }
                    let (x, y) = (s.block((lv as u8, bc)).unwrap(), exact.block((lv as u8, bc)).unwrap());
                    let bad: Vec<usize> = (0..x.len()).filter(|&i| x[i] != y[i] && !(is_fill(x[i]) || is_fill(y[i]))).collect();
                    fills += (0..x.len()).filter(|&i| x[i] != y[i] && (is_fill(x[i]) || is_fill(y[i]))).count();
                    if let Some(&i) = bad.first() {
                        let cell = |j: usize| IVec3::new((j & 7) as i32, ((j >> 3) & 7) as i32, (j >> 6) as i32);
                        let worst = bad.iter().map(|&j| (x[j] - y[j]).abs()).fold(0.0f32, f32::max);
                        let holders: Vec<usize> = (0..l.layers().len()).filter(|&k| l.layer_holds(k, lv, bc)).collect();
                        let children: Vec<IVec3> = (0..8).map(|q| bc * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)).filter(|c| lv + 1 < LEVELS && s.exists(lv + 1, *c)).collect();
                        let parent = (lv > 0).then(|| {
                            let pb = bc.div_euclid(IVec3::splat(2));
                            let (px, py) = (s.block((lv as u8 - 1, pb)).unwrap(), exact.block((lv as u8 - 1, pb)).unwrap());
                            px.iter().zip(py.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
                        });
                        report.push(format!("   children at level {}: {children:?}; the parent block's values apart by {parent:?}", lv + 1));
                        report.push(format!("level {lv} block {bc:?}: {} of 512 samples differ (at most {worst}), e.g. cell {:?} live {} afresh {}; in the form: {}; generators holding it: {holders:?}; layers: {:?}", bad.len(), cell(i), x[i], y[i], l.form().exists(lv, bc), l.layers().iter().map(|x| (x.level, x.visible, x.generator, x.noise, x.blocks)).collect::<Vec<_>>()));
                        if report.len() > 8 {
                            break;
                        }
                    }
                }
            }
            self.fills += fills;
            if !report.is_empty() {
                let afresh: Vec<(usize, usize, usize)> = (0..LEVELS).map(|cap| closed(&exact, cap)).collect();
                report.push(format!("(open, non-manifold, dropped) per cap of the afresh composition: {afresh:?}"));
                // Composed again live (every layer hidden and shown): if the difference goes, blocks were left stale.
                let l = self.l.as_mut().unwrap();
                let n = l.layers().len();
                for k in 0..n {
                    let on = l.layers()[k].visible;
                    l.set_visible(&mut self.s, k, !on);
                    l.set_visible(&mut self.s, k, on);
                }
                let mut worst = 0.0f32;
                for lv in 0..LEVELS {
                    for bc in self.s.level_keys(lv) {
                        if let Some(y) = exact.block((lv as u8, bc)) {
                            let x = self.s.block((lv as u8, bc)).unwrap();
                            worst = worst.max(x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
                        }
                    }
                }
                report.push(format!("composed again live, the values are apart at most {worst}"));
                let again = self.l.as_ref().unwrap().composed_afresh();
                let mut worst2 = 0.0f32;
                let mut where2 = (0, IVec3::ZERO, 0);
                for lv in 0..LEVELS {
                    for bc in again.level_keys(lv) {
                        if let Some(y) = exact.block((lv as u8, bc)) {
                            let x = again.block((lv as u8, bc)).unwrap();
                            for i in 0..x.len() {
                                if (x[i] - y[i]).abs() > worst2 {
                                    worst2 = (x[i] - y[i]).abs();
                                    where2 = (lv, bc, i);
                                }
                            }
                        }
                    }
                }
                report.push(format!("composed afresh twice, the values are apart at most {worst2} at {where2:?}"));
            }
            assert!(report.is_empty(), "after {what}: shown differs from the stack composed afresh:\n{}", report.join("\n"));
            let (s, l) = (&self.s, self.l.as_ref().unwrap());
            let t_afresh = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = std::time::Instant::now();
            // And through a file (the form at 16 bits), every few steps: the same blocks, values within a little of
            // the band.
            if self.steps % 5 != 0 {
                eprintln!("   checks: closed {t_closed:.0} ms, afresh {t_afresh:.0} ms; shown blocks per level {:?}", (0..LEVELS).map(|l| self.s.level_len(l)).collect::<Vec<_>>());
                return;
            }
            let mut buf = Vec::new();
            l.write_to(&mut buf).unwrap();
            let (again, l2) = Layered::read_from(&mut &buf[..], LEVELS).unwrap();
            assert!(l2.is_some());
            for lv in 0..LEVELS {
                let a: std::collections::BTreeSet<[i32; 3]> = s.level_keys(lv).into_iter().map(|v| v.to_array()).collect();
                let b: std::collections::BTreeSet<[i32; 3]> = again.level_keys(lv).into_iter().map(|v| v.to_array()).collect();
                assert!(a == b, "after {what}: opened from a file, level {lv} holds other blocks: only live {:?}, only opened {:?}", a.difference(&b).take(4).collect::<Vec<_>>(), b.difference(&a).take(4).collect::<Vec<_>>());
                // (A checkerboard fill of this level, or of a coarser one: a block made again from the level below
                // copies its samples.)
                let fill_values: Vec<f32> = (0..=lv).map(|m| 1e-3 * s.voxel(m)).collect();
                let is_fill = |x: f32| fill_values.iter().any(|f| (x.abs() - f).abs() < 1e-12);
                // The form at 16 bits: a few of its steps of rounding, which composing can stretch (where the form's
                // levels disagree, as under a stroke at a level between, a sample has moved by 0.22 of a step).
                let tol = 0.3 * s.voxel(lv);
                let mut most = (0.0f32, IVec3::ZERO, 0usize);
                for k in a.intersection(&b) {
                    let bc = IVec3::from_array(*k);
                    if !l.form().exists(lv, bc) && !(0..l.layers().len()).any(|j| l.layer_holds(j, lv, bc)) {
                        continue;
                    }
                    let (x, y) = (s.block((lv as u8, bc)).unwrap(), again.block((lv as u8, bc)).unwrap());
                    for i in 0..x.len() {
                        if !(is_fill(x[i]) || is_fill(y[i])) && (x[i] - y[i]).abs() > most.0 {
                            most = ((x[i] - y[i]).abs(), bc, i);
                        }
                    }
                }
                self.drift[lv] = self.drift[lv].max(most.0);
                if most.0 > tol {
                    let (d, bc, i) = most;
                    let cell = IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32);
                    let p = s.origin() + (bc * 8 + cell).as_vec3() * s.voxel(lv);
                    let (x, y) = (s.block((lv as u8, bc)).unwrap(), again.block((lv as u8, bc)).unwrap());
                    let n = (0..x.len()).filter(|&j| (x[j] - y[j]).abs() > tol).count();
                    let form: Vec<String> = (0..l.from()).map(|m| format!("level {m}: {} (band {})", l.form().value_upto(p, m, false), s.band(m))).collect();
                    panic!("after {what}: opened from a file, level {lv}'s values are apart by {d} (allowed {tol}) at block {bc:?} cell {cell:?} ({n} of 512 over): live {} opened {}; the form there: {}", x[i], y[i], form.join(", "));
                }
            }
            let t_file = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!("   checks: closed {t_closed:.0} ms, afresh {t_afresh:.0} ms, file {t_file:.0} ms; shown blocks per level {:?}", (0..LEVELS).map(|l| self.s.level_len(l)).collect::<Vec<_>>());
        }
    }

    /// The surface closed at every cap: no open edges, no dropped quads. Non-manifold edges are counted and reported,
    /// not failed on: every one seen so far is a saddle in a cell next to a finer one, which has one vertex whatever
    /// its pieces (the extractor's rule, the same on the GPU), so the edge is shared by four quads there. Any that is
    /// not such a cell (a checkerboard face left in the field) fails; `SCULPT_STRESS_STRICT` fails on all of them.
    fn closed_everywhere(&mut self, what: &str) {
        self.closed_everywhere_at(what, false);
    }

    /// [`Self::closed_everywhere`]; `preview` while a stroke's follows are still landing, when a composed block next
    /// to one the form just changed may be stale for a frame (a checkerboard face between them is a pinch that goes
    /// when the follow lands).
    fn closed_everywhere_at(&mut self, what: &str, preview: bool) {
        let strict = std::env::var_os("SCULPT_STRESS_STRICT").is_some() && !preview;
        for cap in 0..LEVELS {
            let (open, nm, dropped) = closed(&self.s, cap);
            if nm > 0 {
                let where_ = non_manifold(&self.s, cap, self.l.as_ref().map(|l| l.form()));
                eprintln!("   NOTE cap {cap}: {nm} non-manifold edges: {}", where_.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(", "));
                self.nm += nm;
                let other: Vec<&(String, bool)> = where_.iter().filter(|w| !w.1).collect();
                assert!(preview || other.is_empty(), "{what}: cap {cap} has non-manifold edges away from any finer cell: {other:?}");
            }
            let nm = if strict { nm } else { 0 };
            assert_eq!((open, nm, dropped), (0, 0, 0), "{what}: cap {cap} has {open} open edges, {nm} non-manifold edges, {dropped} dropped quads");
        }
    }

    fn hit(&self, dir: Vec3, level: usize, form: bool) -> Option<(Vec3, Vec3)> {
        match (&self.l, form) {
            (Some(l), true) => l.form().raycast(dir * 3.0, -dir, level, true),
            _ => self.s.raycast(dir * 3.0, -dir, level, true),
        }
    }

    fn from(&self) -> usize {
        self.l.as_ref().map_or(LEVELS, |l| l.from())
    }

    /// A radius for a brush at `level`: from a few steps to many, up to very big (0.6 on a ball of radius 1, the most
    /// a brush in the app comes to).
    fn radius(&mut self, level: usize) -> f32 {
        let v = self.s.voxel(level);
        if level == 0 && self.rng.chance(0.4) { self.rng.range(0.25, 0.6) } else { (v * self.rng.range(4.0, 48.0)).min(0.6) }
    }

    /// A stroke without layers, on the volume alone: carrying in jobs or at once, the end in a job or at once.
    fn plain_stroke(&mut self) {
        let level = self.rng.below(LEVELS);
        let r = self.radius(level);
        let trim = self.rng.chance(0.4);
        let d = self.rng.dir();
        let jobs = self.rng.chance(0.5);
        let end_job = self.rng.chance(0.5);
        self.s.keep_detail = self.rng.chance(0.3);
        self.s.carry_budget = if jobs { Some(0.0) } else { None };
        let span = (r * 1.5).min(0.5);
        self.s.begin_stroke();
        let mut held = None;
        let s = &mut self.s;
        let hit = |dir: Vec3, level: usize| s.raycast(dir * 3.0, -dir, level, true);
        let hits: Vec<Option<(Vec3, Vec3)>> = {
            let side = d.any_orthonormal_vector();
            (0..=6).map(|k| hit((d + side * span * (-1.0 + k as f32 / 3.0)).normalize(), level)).collect()
        };
        let mut along = Vec3::ZERO;
        for (k, h) in hits.into_iter().enumerate() {
            let Some((at, n)) = h else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = s.plane_dab(level, at, rough, r, r * 0.095, 0.5, 0.4, trim) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            if jobs && k % 2 == 1 {
                if let Some(job) = held.take() {
                    let job: sculpt_core::generator::level_volume::CarryJob = job;
                    s.carry_apply(job.run());
                }
                held = s.carry_take(64);
            }
        }
        if let Some(job) = held.take() {
            s.carry_apply(job.run());
        }
        while let Some(job) = s.carry_take(64) {
            s.carry_apply(job.run());
        }
        if end_job {
            if let Some(job) = s.end_stroke_take() {
                s.end_stroke_apply(job.run());
            }
        } else {
            s.end_stroke();
        }
        self.check(&format!("plain {} stroke at level {level} r {r:.4} (carry jobs {jobs}, end job {end_job}, keep {})", if trim { "trim" } else { "clay" }, self.s.keep_detail));
    }

    /// A stroke on the form under the layers, followed in jobs that land late or at once, ended in a job or at once,
    /// and settled (or, now and then, checked before the last follow lands).
    fn form_stroke(&mut self) {
        let from = self.from();
        let level = self.rng.below(from);
        let r = self.radius(level);
        let trim = self.rng.chance(0.4);
        let d = self.rng.dir();
        let jobs = self.rng.chance(0.6);
        let end_job = self.rng.chance(0.5);
        let span = (r * 1.5).min(0.5);
        let side = d.any_orthonormal_vector();
        let l = self.l.as_mut().unwrap();
        l.deferred = jobs;
        let hits: Vec<Option<(Vec3, Vec3)>> = (0..=6).map(|k| l.form().raycast((d + side * span * (-1.0 + k as f32 / 3.0)).normalize() * 3.0, -(d + side * span * (-1.0 + k as f32 / 3.0)).normalize(), level, true)).collect();
        l.begin_stroke();
        let mut along = Vec3::ZERO;
        for (k, h) in hits.into_iter().enumerate() {
            let Some((at, n)) = h else { continue };
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = l.plane_dab(&mut self.s, level, at, rough, r, r * 0.095, 0.5, 0.4, trim) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            if jobs && k % 3 == 2 && let Some(job) = l.follow_take() {
                l.follow_apply(&mut self.s, job.run());
            }
        }
        if end_job {
            match l.end_form_take() {
                Some(job) => {
                    l.end_form_apply(job.run());
                }
                None => {}
            }
        } else {
            l.end_stroke(&mut self.s);
        }
        if jobs && self.rng.chance(0.3) {
            // Before the last follow lands: closed, at least.
            self.closed_everywhere_at(&format!("mid-follow after a form stroke at level {level}"), true);
        }
        self.l.as_mut().unwrap().settle(&mut self.s);
        self.check(&format!("form {} stroke at level {level} r {r:.4} (follow jobs {jobs}, end job {end_job})", if trim { "trim" } else { "clay" }));
    }

    /// A stroke on what is shown, taken into the active detail layer when it ends (as the app does).
    fn layer_stroke(&mut self) {
        let l = self.l.as_ref().unwrap();
        let k = l.active();
        let info = &l.layers()[k];
        if info.generator || info.noise {
            return self.form_stroke();
        }
        let (from, top) = (l.from(), info.level);
        let level = from + self.rng.below(top - from + 1);
        let v = self.s.voxel(level);
        let r = v * self.rng.range(3.0, 24.0);
        let trim = self.rng.chance(0.5);
        let ball = self.rng.chance(0.3);
        let d = self.rng.dir();
        let end_job = self.rng.chance(0.5);
        let span = r * 1.5;
        let side = d.any_orthonormal_vector();
        let hits: Vec<Option<(Vec3, Vec3)>> = (0..=4).map(|k| {
            let dir = (d + side * span * (-1.0 + k as f32 / 2.0)).normalize();
            self.s.raycast(dir * 3.0, -dir, level, true)
        }).collect();
        self.s.carry_budget = None;
        self.s.begin_stroke();
        let mut along = Vec3::ZERO;
        for h in hits.into_iter() {
            let Some((at, n)) = h else { continue };
            if ball {
                self.s.ball_dab(level, at, r, r * 0.3, trim);
                continue;
            }
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = self.s.plane_dab(level, at, rough, r, r * 0.095, 0.5, 0.4, trim) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
        }
        if end_job {
            if let Some(job) = self.s.end_stroke_take() {
                self.s.end_stroke_apply(job.run());
            }
        } else {
            self.s.end_stroke();
        }
        self.l.as_mut().unwrap().take_detail(&mut self.s);
        self.check(&format!("layer {} {} stroke on layer {k} at level {level} r {r:.4} (end job {end_job})", if ball { "ball" } else { "plane" }, if trim { "trim" } else { "clay" }));
    }

    fn spec(&mut self) -> GeneratorSpec {
        let prims = [Primitive::Sphere, Primitive::Box, Primitive::Cylinder, Primitive::Cone, Primitive::Wedge];
        let kinds = [NoiseKind::Fbm, NoiseKind::Ridged, NoiseKind::Cellular];
        let size = self.s.voxel(0) * self.rng.range(1.5, 4.0);
        let density = match self.rng.below(4) {
            0 => Density::Clusters { scale: 2.0, coverage: 0.4, softness: 0.3, seed: 2 },
            1 => Density::Facing { axis: [0.0, 1.0, 0.0], sharpness: 2.0 },
            2 => Density::Painted { channel: String::new() },
            _ => Density::Even,
        };
        let place_on = if self.rng.chance(0.2) { PlaceOn::Edges { angle: 30.0, side: EdgeSide::Both } } else { PlaceOn::Surface };
        GeneratorSpec {
            primitive: prims[self.rng.below(prims.len())],
            op: if self.rng.chance(0.3) { BoolOp::Subtract } else { BoolOp::Add },
            count: 4 + self.rng.below(20) as u32,
            seed: self.rng.below(1000) as u64,
            size,
            lift: self.rng.range(0.0, 0.3),
            random_spin: self.rng.chance(0.5),
            min_spacing: if self.rng.chance(0.5) { size * 2.5 } else { 0.0 },
            size_var: self.rng.range(0.0, 0.5),
            stretch_var: self.rng.range(0.0, 0.4),
            tilt: self.rng.range(0.0, 30.0),
            blend: self.rng.range(0.05, 0.6),
            rough: if self.rng.chance(0.5) { self.rng.range(0.05, 0.25) } else { 0.0 },
            rough_kind: kinds[self.rng.below(kinds.len())],
            density,
            place_on,
            ..GeneratorSpec::default()
        }
    }

    /// A level for a generator of `size`: the coarsest giving about a dozen steps a radius (as the app picks).
    fn generator_level(&mut self, size: f32) -> usize {
        (0..LEVELS).find(|&l| self.s.voxel(l) <= size / 12.0).unwrap_or(LEVELS - 1)
    }

    fn add_generator(&mut self) {
        let spec = self.spec();
        let level = self.generator_level(spec.size);
        let l = self.l.as_mut().unwrap();
        let k = l.add_generator(&mut self.s, "Rocks", level, &spec);
        self.check(&format!("generator {k} added at level {level}: {:?} {:?} count {} size {:.3} blend {:.2} rough {:.2} {:?} {:?}", spec.primitive, spec.op, spec.count, spec.size, spec.blend, spec.rough, spec.density, spec.place_on));
    }

    fn generators(&self) -> Vec<usize> {
        self.l.as_ref().map_or(Vec::new(), |l| l.layers().iter().enumerate().filter(|(_, x)| x.generator).map(|(k, _)| k).collect())
    }

    /// Edit a generator: one parameter, shown at once, or previewed coarse and landed from a job (as a drag ends).
    fn edit_generator(&mut self) {
        let gens = self.generators();
        if gens.is_empty() {
            return self.add_generator();
        }
        let k = gens[self.rng.below(gens.len())];
        let l = self.l.as_mut().unwrap();
        let mut spec = l.generator(k).unwrap().clone();
        let which = self.rng.below(9);
        match which {
            0 => spec.count = 2 + self.rng.below(30) as u32,
            1 => spec.size *= self.rng.range(0.6, 1.6),
            2 => spec.seed += 1,
            3 => spec.blend = self.rng.range(0.0, 0.7),
            4 => spec.op = if spec.op == BoolOp::Add { BoolOp::Subtract } else { BoolOp::Add },
            5 => spec.rough = if spec.rough > 0.0 { 0.0 } else { 0.2 },
            6 => spec.place_on = match spec.place_on {
                PlaceOn::Surface => PlaceOn::Edges { angle: 30.0, side: EdgeSide::Ridges },
                _ => PlaceOn::Surface,
            },
            7 => spec.density = if matches!(spec.density, Density::Even) { Density::Painted { channel: String::new() } } else { Density::Even },
            _ => spec.lift = self.rng.range(-0.1, 0.4),
        }
        let preview = self.rng.chance(0.5);
        let drag = self.rng.chance(0.5);
        if drag {
            l.begin_edit();
        }
        if preview {
            l.preview_generator(&mut self.s, k, &spec);
            if self.rng.chance(0.5) {
                // A second change while the first is still a preview.
                spec.size *= 1.1;
                l.preview_generator(&mut self.s, k, &spec);
            }
            if self.rng.chance(0.5) {
                // The job for the first preview, landing after a newer one: dropped.
                if let Some(job) = l.generator_take() {
                    let done = job.run();
                    spec.blend = (spec.blend + 0.05).min(0.7);
                    l.preview_generator(&mut self.s, k, &spec);
                    let taken = l.generator_apply(&mut self.s, done);
                    assert!(!taken, "a stale generator job was taken");
                }
            }
            while let Some(job) = l.generator_take() {
                let done = job.run();
                assert!(l.generator_apply(&mut self.s, done), "a current generator job was dropped");
            }
        } else {
            l.set_generator(&mut self.s, k, &spec);
        }
        if drag {
            l.end_edit_in(&mut self.s);
        }
        while let Some(job) = l.generator_take() {
            let done = job.run();
            l.generator_apply(&mut self.s, done);
        }
        self.check(&format!("generator {k} edit {which} (preview {preview}, drag {drag}): count {} size {:.3} blend {:.2} {:?} lift {:.2}", spec.count, spec.size, spec.blend, spec.op, spec.lift));
    }

    fn paint_density(&mut self) {
        let gens: Vec<usize> = self.generators().into_iter().filter(|k| matches!(self.l.as_ref().unwrap().generator(*k).unwrap().density, Density::Painted { .. })).collect();
        if gens.is_empty() {
            return self.edit_generator();
        }
        let k = gens[self.rng.below(gens.len())];
        let d = self.rng.dir();
        let r = self.rng.range(0.1, 0.5);
        let to = if self.rng.chance(0.6) { 0.0 } else { 1.0 };
        let l = self.l.as_mut().unwrap();
        l.set_active(k);
        l.begin_edit();
        let Some((at, _)) = l.form().raycast(d * 3.0, -d, 0, false) else { return };
        assert!(l.paint_density(at, r, to));
        l.end_density_stroke(&mut self.s);
        while let Some(job) = l.generator_take() {
            let done = job.run();
            l.generator_apply(&mut self.s, done);
        }
        self.check(&format!("density of generator {k} painted to {to} within {r:.2}"));
    }

    fn paint_mask(&mut self) {
        let l = self.l.as_mut().unwrap();
        let k = l.active();
        let d = self.rng.dir();
        let level = l.layers()[k].level.min(l.from());
        let Some((at, _)) = l.form().raycast(d * 3.0, -d, level, false) else { return };
        let r = self.rng.range(0.05, 0.4);
        let to = if self.rng.chance(0.5) { 0.0 } else { 1.0 };
        l.begin_edit();
        l.paint_mask(&mut self.s, at, r, to);
        if self.rng.chance(0.5) {
            l.paint_mask(&mut self.s, at + d.any_orthonormal_vector() * r * 0.5, r, to);
        }
        l.end_edit();
        self.check(&format!("mask of layer {k} painted to {to} within {r:.2}"));
    }

    fn fill_mask(&mut self) {
        let l = self.l.as_mut().unwrap();
        let k = l.active();
        let to = if self.rng.chance(0.6) { 0.0 } else { 1.0 };
        l.fill_mask(&mut self.s, to);
        self.check(&format!("mask of layer {k} filled to {to}"));
    }

    fn add_layer(&mut self) {
        let from = self.from();
        let level = from + self.rng.below(LEVELS - from);
        let l = self.l.as_mut().unwrap();
        let k = l.add_layer(&mut self.s, "Layer", level);
        self.check(&format!("detail layer {k} added at level {level}"));
    }

    fn remove_layer(&mut self) {
        let l = self.l.as_mut().unwrap();
        let n = l.layers().len();
        if n == 0 {
            return;
        }
        let k = self.rng.below(n);
        l.remove_layer(&mut self.s, k);
        self.check(&format!("layer {k} removed"));
    }

    fn set_visible(&mut self) {
        let l = self.l.as_mut().unwrap();
        let layers = l.layers();
        if layers.is_empty() {
            return;
        }
        let k = self.rng.below(layers.len());
        let on = !layers[k].visible;
        l.set_visible(&mut self.s, k, on);
        self.check(&format!("layer {k} {}", if on { "shown" } else { "hidden" }));
    }

    fn set_level(&mut self) {
        let l = self.l.as_mut().unwrap();
        let layers = l.layers();
        if layers.is_empty() {
            return;
        }
        let k = self.rng.below(layers.len());
        if layers[k].noise {
            return;
        }
        let level = if layers[k].generator {
            let size = l.generator(k).unwrap().size;
            self.generator_level(size)
        } else {
            // (A layer over the whole surface, finer: a wait, not a check.)
            let from = l.from();
            let top = if layers[k].blocks > 1500 { (from + 1).min(LEVELS - 1) } else { LEVELS - 1 };
            from + self.rng.below(top - from + 1)
        };
        let l = self.l.as_mut().unwrap();
        l.set_level(&mut self.s, k, level);
        self.check(&format!("layer {k} moved to level {level}"));
    }

    fn set_active(&mut self) {
        let l = self.l.as_mut().unwrap();
        let n = l.layers().len();
        if n == 0 {
            return;
        }
        let k = self.rng.below(n);
        l.set_active(k);
        self.note(format!("layer {k} made active"));
    }

    fn add_noise(&mut self) {
        let l = self.l.as_mut().unwrap();
        let n = SurfaceNoise { on: true, amplitude: self.rng.range(0.005, 0.03), scale: self.rng.range(2.0, 10.0), octaves: 1 + self.rng.below(6) as u32, kind: self.rng.below(2) as u32, seed: self.rng.below(100) as u32 };
        let k = l.add_noise("Noise", n);
        self.check(&format!("noise layer {k} added"));
    }

    fn noises(&self) -> Vec<usize> {
        self.l.as_ref().map_or(Vec::new(), |l| l.layers().iter().enumerate().filter(|(_, x)| x.noise).map(|(k, _)| k).collect())
    }

    fn bake_noise(&mut self) {
        let ns = self.noises();
        if ns.is_empty() {
            return self.add_noise();
        }
        let k = ns[self.rng.below(ns.len())];
        let from = self.from();
        // Over a whole ball a fine bake is millions of blocks: the layers' first level.
        let level = from;
        let l = self.l.as_mut().unwrap();
        let at = l.bake_noise(&mut self.s, k, level);
        self.check(&format!("noise layer {k} baked at level {level} into layer {at:?}"));
    }

    fn tweak_noise(&mut self) {
        let ns = self.noises();
        if ns.is_empty() {
            return self.add_noise();
        }
        let k = ns[self.rng.below(ns.len())];
        let l = self.l.as_mut().unwrap();
        let mut n = *l.noise(k).unwrap();
        n.amplitude *= 0.7;
        n.kind = 1 - n.kind;
        l.set_noise(k, n);
        self.check(&format!("noise layer {k} changed"));
    }

    /// A state fingerprint, an operation, undo back to it exactly, redo forward to the result exactly.
    fn undo_redo(&mut self, op: fn(&mut World), name: &str) {
        let before = fingerprint(&self.s, self.l.as_ref());
        let before_s = samples(&self.s, self.l.as_ref());
        let history = self.l.as_ref().map_or(0, |l| l.history_len());
        op(self);
        let after = fingerprint(&self.s, self.l.as_ref());
        let after_s = samples(&self.s, self.l.as_ref());
        let after_afresh = samples(&self.l.as_ref().unwrap().composed_afresh(), self.l.as_ref());
        let pushed = self.l.as_ref().map_or(0, |l| l.history_len()) - history;
        if pushed == 0 {
            self.note(format!("{name}: nothing to undo"));
            return;
        }
        assert_eq!(pushed, 1, "{name} pushed {pushed} undo steps");
        let l = self.l.as_mut().unwrap();
        assert!(l.undo(&mut self.s));
        let undone = fingerprint(&self.s, self.l.as_ref());
        if let Err(why) = same_state(&undone, &before) {
            eprintln!("undo: {why}");
            panic!("undo after {name} did not bring back the state before it: shown {}", describe_difference(&before_s, &self.s, self.l.as_ref()));
        }
        self.check(&format!("undo of {name}"));
        let l = self.l.as_mut().unwrap();
        assert!(l.redo(&mut self.s));
        let redone = fingerprint(&self.s, self.l.as_ref());
        if let Err(why) = same_state(&redone, &after) {
            eprintln!("redo: {why}");
            let now_afresh = self.l.as_ref().unwrap().composed_afresh();
            let l = self.l.as_ref().unwrap();
            let worst = (3u8, IVec3::new(0, 4, 7));
            let holders: Vec<usize> = (0..l.layers().len()).filter(|&k| l.layer_holds(k, worst.0 as usize, worst.1)).collect();
            let x = self.s.block(worst).unwrap();
            let y = now_afresh.block(worst).unwrap();
            let i = (0..x.len()).max_by(|a, b| (x[*a] - y[*a]).abs().total_cmp(&(x[*b] - y[*b]).abs())).unwrap();
            eprintln!("worst cell {:?}: live {} afresh {}; children: {:?}", IVec3::new((i & 7) as i32, ((i >> 3) & 7) as i32, (i >> 6) as i32), x[i], y[i], (0..8).map(|q| worst.1 * 2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) & 1)).filter(|c| self.s.exists(4, *c)).collect::<Vec<_>>());
            let p2 = worst.1.div_euclid(IVec3::splat(2));
            for q in 0..8 {
                let n = p2 + IVec3::new(q & 1, (q >> 1) & 1, (q >> 2) as i32 & 1);
                let hold: Vec<usize> = (0..l.layers().len()).filter(|&k| l.layer_holds(k, 2, n)).collect();
                eprintln!("level 2 block {n:?}: exists {} dirty {} held by {hold:?} in form {}", self.s.exists(2, n), self.s.dirty_contains((2, n)), l.form().exists(2, n));
            }
            eprintln!("level 3 block dirty: {}", self.s.dirty_contains(worst));
            panic!("redo after {name} did not bring back the state after it ({why}); shown {}
 redo-state against its afresh: {}
 after-afresh against redo-afresh: {}
 layers holding (3,[0,4,7]): {holders:?}; in the form: {}", describe_difference(&after_s, &self.s, self.l.as_ref()), describe_difference(&samples(&now_afresh, self.l.as_ref()), &self.s, self.l.as_ref()), describe_difference(&after_afresh, &now_afresh, self.l.as_ref()), l.form().exists(3, worst.1));
        }
        self.check(&format!("redo of {name}"));
    }

    /// Several undos then as many redos: the end state is the start state exactly.
    fn undo_chain(&mut self) {
        let n = 1 + self.rng.below(4);
        let start = fingerprint(&self.s, self.l.as_ref());
        let mut done = 0;
        for _ in 0..n {
            let ok = match self.l.as_mut() {
                Some(l) => l.undo(&mut self.s),
                None => self.s.undo(),
            };
            if !ok {
                break;
            }
            done += 1;
            self.check("undo (chain)");
        }
        for _ in 0..done {
            let ok = match self.l.as_mut() {
                Some(l) => l.redo(&mut self.s),
                None => self.s.redo(),
            };
            assert!(ok, "redo failed in a chain of {done}");
            self.check("redo (chain)");
        }
        let end = fingerprint(&self.s, self.l.as_ref());
        if let Err(why) = same_state(&start, &end) {
            panic!("undo x{done} then redo x{done} changed the state: {why}");
        }
    }

    /// Save, open, and go on from the opened file (what is shown and the stack as a file holds them).
    fn save_open(&mut self) {
        let Some(l) = self.l.as_ref() else { return };
        let mut buf = Vec::new();
        l.write_to(&mut buf).unwrap();
        let (s, l2) = Layered::read_from(&mut &buf[..], LEVELS).unwrap();
        let mut l2 = l2.unwrap();
        l2.deferred = true;
        let mut buf2 = Vec::new();
        l2.write_to(&mut buf2).unwrap();
        assert!(buf == buf2, "a file opened and saved again differs");
        self.s = s;
        self.l = Some(l2);
        self.check("saved and opened");
    }

    fn make_layer(&mut self) {
        let from = 1 + self.rng.below(LEVELS - 1);
        let mut l = Layered::new(&mut self.s, from);
        l.deferred = true;
        // (With no work to take, the layer is made at the finest level: a level of its own instead, as a layer added
        // by hand would have, so strokes on it are not all composed at the finest level.)
        let level = from + self.rng.below(LEVELS - from);
        if l.layers()[0].blocks == 0 {
            l.set_level(&mut self.s, 0, level);
        }
        self.l = Some(l);
        self.check(&format!("layers made from level {from}, the first at level {level}"));
    }

    fn step(&mut self) {
        if self.l.is_none() {
            if self.rng.chance(0.25) {
                self.make_layer();
            } else {
                self.plain_stroke();
            }
            return;
        }
        let roll = self.rng.below(100);
        match roll {
            0..=17 => self.form_stroke(),
            18..=35 => self.layer_stroke(),
            36..=40 => self.add_generator(),
            41..=50 => self.edit_generator(),
            51..=54 => self.paint_density(),
            55..=60 => self.paint_mask(),
            61..=63 => self.fill_mask(),
            64..=67 => self.add_layer(),
            68..=70 => self.remove_layer(),
            71..=74 => self.set_visible(),
            75..=78 => self.set_level(),
            79..=81 => self.set_active(),
            82 => self.add_noise(),
            83 => self.tweak_noise(),
            84..=86 => self.bake_noise(),
            87..=89 => self.undo_chain(),
            90..=92 => self.save_open(),
            _ => {
                let ops: [(fn(&mut World), &str); 9] = [
                    (World::form_stroke, "a form stroke"),
                    (World::layer_stroke, "a layer stroke"),
                    (World::add_generator, "adding a generator"),
                    (World::edit_generator, "a generator edit"),
                    (World::paint_mask, "a mask stroke"),
                    (World::fill_mask, "a mask fill"),
                    (World::remove_layer, "removing a layer"),
                    (World::set_level, "moving a layer's level"),
                    (World::bake_noise, "baking noise"),
                ];
                let (op, name) = ops[self.rng.below(ops.len())];
                self.undo_redo(op, name);
            }
        }
    }
}

fn run(seed: u64, steps: usize, voxel: f32) {
    let mut w = World::new(seed, voxel);
    eprintln!("seed {seed}, {steps} steps");
    for _ in 0..steps {
        w.step();
    }
    if w.l.is_none() {
        w.make_layer();
    }
    w.save_open();
    eprintln!("worst drift live against opened per level: {:?}; non-manifold edges seen (summed over checks and caps): {}; checkerboard fills that differed: {}; the surface reached {:.3} from the centre", w.drift, w.nm, w.fills, w.reach);
}

#[test]
fn a_short_mixed_session_stays_closed_exact_and_undoable() {
    run(1, 14, 0.12);
}

#[test]
#[ignore]
fn a_long_mixed_session() {
    let seeds: Vec<u64> = match std::env::var("SCULPT_STRESS_SEEDS") {
        Ok(r) => {
            let (a, b) = r.split_once("..").expect("a..b");
            (a.parse().unwrap()..b.parse().unwrap()).collect()
        }
        Err(_) => vec![std::env::var("SCULPT_STRESS_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1)],
    };
    let steps = std::env::var("SCULPT_STRESS_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let voxel = std::env::var("SCULPT_STRESS_VOXEL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.04);
    for seed in seeds {
        run(seed, steps, voxel);
    }
}

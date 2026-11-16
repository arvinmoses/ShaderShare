//! ZBrush-feel brushes.
//!
//! The "feel" of ZBrush brushes comes mostly from *where the target plane is*
//! and *how the footprint is shaped*, so each brush below documents those
//! choices explicitly. All brushes run through
//! [`Document::compute_displacements`] (parallel, read-only) followed by
//! [`Document::apply_displacements`] (writes to the active layer).

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::document::{Displacements, Document, PaintTarget, SurfaceHit, Under};
use crate::geom::smoothstep;
use crate::Result;

/// Radial falloff: flat to `hardness`, then a smoothstep to zero at the rim.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Falloff {
    pub hardness: f32,
}

impl Default for Falloff {
    fn default() -> Self {
        Falloff { hardness: 0.2 }
    }
}

impl Falloff {
    #[inline]
    pub fn eval(&self, t: f32) -> f32 {
        if t >= 1.0 {
            return 0.0;
        }
        1.0 - smoothstep(self.hardness.min(0.999), 1.0, t)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushSettings {
    /// World-space radius.
    pub radius: f32,
    pub strength: f32,
    pub falloff: Falloff,
    /// Dab spacing as a fraction of the radius.
    pub spacing: f32,
    pub front_faces_only: bool,
    /// Alt-key behaviour (dig instead of build, etc.).
    pub invert: bool,
}

impl Default for BrushSettings {
    fn default() -> Self {
        BrushSettings { radius: 0.1, strength: 0.5, falloff: Falloff::default(), spacing: 0.12, front_faces_only: true, invert: false }
    }
}

/// One brush application on the surface.
#[derive(Clone, Copy, Debug)]
pub struct Dab {
    pub center: Vec3,
    pub normal: Vec3,
    /// Stroke tangent (may be zero for the first dab).
    pub direction: Vec3,
    pub pressure: f32,
}

pub trait Brush {
    fn name(&self) -> &'static str;
    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()>;
}

fn tangent_frame(n: Vec3, dir: Vec3) -> (Vec3, Vec3) {
    let d = dir - n * dir.dot(n);
    let x = if d.length_squared() > 1e-12 { d.normalize() } else { n.any_orthonormal_vector() };
    (x, n.cross(x))
}

// ---------------------------------------------------------------- Clay Buildup

/// ZBrush *Clay Buildup*: a rounded-square footprint (aligned to the stroke)
/// that pulls surface *up toward a plane* floating above the local average
/// surface. Vertices already above the plane are left alone, so overlapping
/// passes stack into ridges and plateaus instead of ballooning like Inflate,
/// and the square edges leave the characteristic stepped clay strokes.
///
/// The plane is re-sampled every dab, so on its own it would ride up on the
/// clay it just laid down. With `accumulate` off (the default) a single
/// stroke can raise a vertex at most one plane height above where it was at
/// stroke start; each new stroke builds another layer, which is the ZBrush
/// behaviour. With `accumulate` on there is no per-stroke cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClayBuildup {
    /// Plane height above the area center, as a fraction of the radius.
    pub height: f32,
    /// Superellipse exponent of the footprint: 2 = round, higher = squarer.
    pub squareness: f32,
    pub accumulate: bool,
}

impl Default for ClayBuildup {
    fn default() -> Self {
        ClayBuildup { height: 0.18, squareness: 4.0, accumulate: false }
    }
}

impl Brush for ClayBuildup {
    fn name(&self) -> &'static str {
        "Clay Buildup"
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let front = s.front_faces_only.then_some(dab.normal);
        let Some((area_c, area_n)) = doc.sample_area(dab.center, s.radius, front) else { return Ok(()) };
        let n = if s.invert { -area_n } else { area_n };
        let lift = s.radius * self.height * dab.pressure;
        let plane = area_c + n * lift;
        let cap = if self.accumulate { f32::INFINITY } else { lift };
        let (x, y) = tangent_frame(area_n, dab.direction);
        let (r, p, fade) = (s.radius, self.squareness.max(1.0), s.strength * dab.pressure);
        let falloff = s.falloff;
        // The superellipse corner reaches ~1.19r for p = 4.
        let reach = r * 2f32.powf(0.5 - 1.0 / p);
        let disp = doc.compute_displacements(dab.center, reach, |doc, v, _| {
            let pos = doc.positions[v];
            if front.is_some() && doc.normals[v].dot(area_n) <= 0.0 {
                return None;
            }
            let local = pos - dab.center;
            let (lx, ly) = (local.dot(x) / r, local.dot(y) / r);
            let d = (lx.abs().powf(p) + ly.abs().powf(p)).powf(1.0 / p);
            if d >= 1.0 {
                return None;
            }
            let h = (pos - plane).dot(n);
            if h >= 0.0 || h < -r {
                return None;
            }
            let risen = (pos - doc.stroke_origin(v)).dot(n);
            // A vertex moves as much as it faces the plane. On a steep wall (the inside of a cut) the lower vertices
            // would otherwise rise past the ones above them and fold the wall over itself; this way the floor fills
            // and the wall follows.
            let amount = (-h * falloff.eval(d) * fade * facing(doc.normals[v], area_n)).min(cap - risen);
            (amount > 0.0).then(|| n * amount)
        });
        doc.apply_displacements(&disp)
    }
}

// ----------------------------------------------------------------- Plane brush

/// Clay and trim on a mesh as the volume's do them (`LevelVolume::plane_dab`), so a layer's strokes look as the base's
/// do: a round brush whose plane is set by the surface as the stroke found it (each vertex's [`Document::stroke_origin`])
/// and what the stroke has laid since. Clay's plane stands `height` over that surface, and over what is laid by at least
/// [`crate::generator::level_volume::BUILD_STEP`] of it, so scrubbing keeps building, slowly; trim's stands `height`
/// under it. Each dab moves what lies below the plane (above it, trimming) its `strength` of the way there, fully out
/// to the falloff's hardness and easing to nothing at the rim. Under a brush wide enough that the form curves away
/// under it, clay follows the surface as the stroke found it instead of a flat plane (as the volume's does).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlaneBrush {
    /// How far the plane stands over (under, trimming) the surface, in world units.
    pub height: f32,
    pub trim: bool,
}

impl Brush for PlaneBrush {
    fn name(&self) -> &'static str {
        if self.trim { "Trim" } else { "Clay" }
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        use crate::generator::level_volume::BUILD_STEP;
        use rayon::prelude::*;
        let seed = dab.normal.normalize_or_zero();
        if seed == Vec3::ZERO {
            return Ok(());
        }
        // On a layer, along the normal of the surface under it there (as layer Smooth): the shown surface's turns toward
        // a groove's or a ridge's walls, and dabs along it moved offsets sideways until faces folded (shards).
        let n = match doc.active_layer() {
            Some(_) => {
                let under = doc.under(dab.center, s.radius);
                LayerSmooth::base_about(doc, &under, dab.center, seed, s.radius).map_or(seed, |b| b.0)
            }
            None => seed,
        };
        let (r, h) = (s.radius, self.height * dab.pressure);
        if !(r > 0.0 && h > 0.0 && s.strength * dab.pressure > 0.0) {
            return Ok(());
        }
        // On a Subtract layer the dab works on the surface as it would show were the layer Add (the base and the
        // other layers plus this one's offsets), and shows flipped: clay there digs, as the layer holds what clay
        // lays. `lift`: how far that surface is from the one shown.
        let flip = doc.active_layer().and_then(|id| doc.layer(id)).filter(|l| l.blend == crate::layers::LayerBlend::Subtract);
        let chunks = doc.active_layer_chunks().unwrap_or(&[]);
        let lift = |v: usize| flip.map_or(Vec3::ZERO, |l| doc.layer_offset(chunks, v) * (2.0 * l.weight(v)));
        let now_at = |v: usize| doc.positions[v] + lift(v);
        // (The rest stays as it was over a layer stroke, so where the stroke found the layer is mirrored about it.)
        let was_at = |v: usize| match flip {
            Some(_) => doc.positions[v] * 2.0 + lift(v) - doc.stroke_origin(v),
            None => doc.stroke_origin(v),
        };
        let c = match flip {
            Some(_) => {
                let mut near = Vec::new();
                doc.bvh.query_sphere(dab.center, 0.5 * r, &mut near);
                let (mut sum, mut k) = (Vec3::ZERO, 0.0f32);
                for &l in &near {
                    for v in doc.bvh.leaves[l as usize].owned_range() {
                        let d2 = doc.positions[v].distance_squared(dab.center);
                        if d2 < 0.25 * r * r {
                            let w = 1.0 - d2 / (0.25 * r * r);
                            sum += lift(v) * w;
                            k += w;
                        }
                    }
                }
                dab.center + if k > 0.0 { sum / k } else { Vec3::ZERO }
            }
            None => dab.center,
        };
        let out = if flip.is_some() { -1.0 } else { 1.0 };
        let (t1, t2) = n.any_orthonormal_pair();
        // Over the disc: the surface as the stroke found it, and as it is, along `n` from the dab's centre; and a
        // quadratic fitted to the first, the form at the brush's scale. The centre's own (within a quarter of the
        // radius) is where the stroke found the surface under the dab.
        let mut leaves = Vec::new();
        // (Shown, the vertices sit about as far from the dab as the flip moves its centre.)
        let shift = c.distance(dab.center);
        doc.bvh.query_sphere(dab.center, 1.5 * r + shift, &mut leaves);
        type Sums = ([f64; 5], [[f64; 6]; 6], [f64; 6]);
        let terms = |x: f64, y: f64| [1.0, x, y, x * x, x * y, y * y];
        let zero: Sums = ([0.0; 5], [[0.0; 6]; 6], [0.0; 6]);
        let (a, m, mv) = leaves
            .par_iter()
            .map(|&l| {
                let (mut a, mut m, mut mv) = zero;
                for v in doc.bvh.leaves[l as usize].owned_range() {
                    if doc.normals[v].dot(n) <= 0.0 {
                        continue;
                    }
                    let (was, now) = (was_at(v) - c, now_at(v) - c);
                    let t = (now - n * now.dot(n)).length() / r;
                    if t > 0.95 || now.dot(n).abs() > r {
                        continue;
                    }
                    let (hw, hn) = (was.dot(n) as f64, now.dot(n) as f64);
                    a[0] += hw;
                    a[1] += hn - hw;
                    a[2] += 1.0;
                    if t < 0.25 {
                        a[3] += hw;
                        a[4] += 1.0;
                    }
                    let f = terms((was.dot(t1) / r) as f64, (was.dot(t2) / r) as f64);
                    for j in 0..6 {
                        mv[j] += f[j] * hw;
                        for k in 0..6 {
                            m[j][k] += f[j] * f[k];
                        }
                    }
                }
                (a, m, mv)
            })
            .reduce(
                || zero,
                |x, y| (std::array::from_fn(|i| x.0[i] + y.0[i]), std::array::from_fn(|j| std::array::from_fn(|k| x.1[j][k] + y.1[j][k])), std::array::from_fn(|i| x.2[i] + y.2[i])),
            );
        if a[2] == 0.0 {
            return Ok(());
        }
        let on = if a[4] > 0.0 { a[3] / a[4] } else { a[0] / a[2] } as f32;
        let (base, added) = ((a[0] / a[2]) as f32 - on, (a[1] / a[2]) as f32);
        // (A small ridge keeps a fit over a coarse or one-sided disc from running wild; too few samples, none.)
        let fit = (a[2] >= 30.0)
            .then(|| {
                let mut m = m;
                let ridge = 1e-6 * (0..6).map(|j| m[j][j]).sum::<f64>();
                (0..6).for_each(|j| m[j][j] += ridge);
                crate::height_grid::solve(m, mv)
            })
            .flatten()
            .unwrap_or([0.0; 6]);
        // How far the form falls away under the brush, on average (a quadratic's mean over a disc is its centre plus
        // half its rim's fall).
        let mean_fall = ((1..6).map(|j| fit[j] * m[0][j] / a[2]).sum::<f64>() as f32).clamp(-r, 0.0);
        // The share of that fall the plane is laid over (see `fall`). The rest of the lift a flat plane gets over a
        // curved form is kept, as the volume's dabs keep it (matched stroke for stroke in the app's
        // `layer_clay_and_trim_match_the_base`): about a third of it where the form falls a third of `h` at the rim,
        // 85% once it falls twice `h`.
        let rim = (-2.0 * mean_fall).max(0.0) / h.max(1e-9);
        let x = ((rim - 0.4) / 1.6).clamp(0.0, 1.0);
        let lay = 0.65 - 0.5 * x * x * (3.0 - 2.0 * x);
        let bumps = base - lay * mean_fall.min(0.0);
        let plane = if self.trim {
            bumps.min(0.0) - h
        } else {
            let ground = bumps.max(0.0);
            (ground + h).max(ground + added + BUILD_STEP * h)
        };
        // How far the form curves away under the brush (the volume's measure): clay follows the surface the stroke
        // found, flat under a small brush, mixed between.
        let follow = if self.trim {
            0.0
        } else {
            let x = ((-base / r - 0.14) / 0.1).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        // The form's fall away from the centre (where it curves down only): the plane is laid over it, so what a dab
        // lays is as thick at its rim as at its centre, as the volume's is (a flat plane over a curved form stood
        // higher above it toward the rim, and overlapping dabs built a stroke's middle a third of `h` too high).
        let fall = move |x: f32, y: f32| {
            let f = terms(x as f64, y as f64);
            // (With its tilt: the dab's normal is the surface's as it is, not as the stroke found it.)
            lay * ((1..6).map(|j| f[j] * fit[j]).sum::<f64>() as f32).clamp(-r, 0.0)
        };
        let (alpha, falloff, trim) = ((s.strength * dab.pressure).clamp(0.0, 1.0), s.falloff, self.trim);
        let disp = doc.compute_displacements(dab.center, 1.2 * r + shift, |doc, v, _| {
            if doc.normals[v].dot(n) <= 0.0 {
                return None;
            }
            let q = now_at(v) - c;
            let u = q.dot(n);
            let w = falloff.eval((q - n * u).length() / r) * alpha;
            if w <= 0.0 {
                return None;
            }
            let was = was_at(v) - c;
            let under = was.dot(n) - on;
            let target = on + plane + follow * under + (1.0 - follow) * fall(was.dot(t1) / r, was.dot(t2) / r);
            let d = target - u;
            ((if trim { d < 0.0 } else { d > 0.0 }) && d.abs() < 2.0 * r).then(|| n * (d * w * out))
        });
        doc.apply_displacements(&disp)
    }
}

/// How squarely a vertex with normal `normal` faces a plane with normal `n`: 1 face on, falling to 0 edge on.
fn facing(normal: Vec3, n: Vec3) -> f32 {
    normal.normalize_or_zero().dot(n).clamp(0.0, 1.0)
}

// --------------------------------------------------------------- Trim Dynamic

/// ZBrush *Trim Dynamic*: each dab computes a plane from the local area
/// normal/center, sunk slightly below the average, and shaves away only what
/// sits above it. Because the plane is recomputed per dab it follows the form
/// and carves crisp planar facets. `smooth_border` relaxes the outer band of
/// the footprint (TrimSmoothBorder behaviour) so cuts blend instead of
/// stepping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrimDynamic {
    /// Plane depth below the area center, as a fraction of the radius.
    pub depth: f32,
    /// Fraction of the radius, measured inward from the rim, that is smoothed.
    pub smooth_border: f32,
}

impl Default for TrimDynamic {
    fn default() -> Self {
        TrimDynamic { depth: 0.03, smooth_border: 0.35 }
    }
}

impl Brush for TrimDynamic {
    fn name(&self) -> &'static str {
        "Trim Dynamic"
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let front = s.front_faces_only.then_some(dab.normal);
        let Some((area_c, n)) = doc.sample_area(dab.center, s.radius, front) else { return Ok(()) };
        // Inverted: fill hollows up to a plane slightly above instead.
        let sign = if s.invert { -1.0 } else { 1.0 };
        let plane = area_c - n * (sign * s.radius * self.depth * dab.pressure);
        let (r, fade, sb) = (s.radius, s.strength * dab.pressure, self.smooth_border.clamp(0.0, 1.0));
        let falloff = s.falloff;
        let disp = doc.compute_displacements(dab.center, r, |doc, v, dist| {
            if front.is_some() && doc.normals[v].dot(n) <= 0.0 {
                return None;
            }
            let pos = doc.positions[v];
            let t = dist / r;
            let h = (pos - plane).dot(n) * sign;
            let mut d = Vec3::ZERO;
            if h > 0.0 {
                // As in Clay Buildup: shaving a steep wall flat would fold it, so a vertex moves as much as it faces the plane.
                d -= n * (sign * h * falloff.eval(t) * fade * facing(doc.normals[v], n));
            }
            if sb > 0.0 && t > 1.0 - sb {
                let u = (t - (1.0 - sb)) / sb;
                let ring = doc.topo.vert_verts.row(v);
                if !ring.is_empty() {
                    let avg = ring.iter().map(|&u| doc.positions[u as usize]).sum::<Vec3>() / ring.len() as f32;
                    d += (avg - pos) * (4.0 * u * (1.0 - u) * fade);
                }
            }
            Some(d)
        });
        doc.apply_displacements(&disp)
    }
}

// --------------------------------------------------------------------- Smooth

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmoothMode {
    /// Uniform Laplacian: strong, slightly shrinks (ZBrush Shift default).
    #[default]
    Laplacian,
    /// Tangential relax only: evens out topology, keeps the form.
    Surface,
}

/// Smooth on a mesh document, by [`SmoothMode`].
/// (Old mesh-mode code: reached only from sculpt-cli and the tests, not the app.)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Smooth {
    pub mode: SmoothMode,
}

impl Brush for Smooth {
    fn name(&self) -> &'static str {
        "Smooth"
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let front = s.front_faces_only.then_some(dab.normal);
        let (r, fade, mode) = (s.radius, (s.strength * dab.pressure).clamp(0.0, 1.0), self.mode);
        let falloff = s.falloff;
        let disp = doc.compute_displacements(dab.center, r, |doc, v, dist| {
            let nrm = doc.normals[v];
            if front.is_some_and(|f| nrm.dot(f) <= 0.0) {
                return None;
            }
            let ring = doc.topo.vert_verts.row(v);
            if ring.is_empty() {
                return None;
            }
            let pos = doc.positions[v];
            let avg = ring.iter().map(|&u| doc.positions[u as usize]).sum::<Vec3>() / ring.len() as f32;
            let mut lap = avg - pos;
            if mode == SmoothMode::Surface {
                lap -= nrm * lap.dot(nrm);
            }
            Some(lap * (falloff.eval(dist / r) * fade))
        });
        doc.apply_displacements(&disp)
    }
}

// ---------------------------------------------------------------- Form smooth

/// Smoothing at the brush's scale, as the volume's smooth does it (`LevelVolume::smooth_dab`): the surface under the
/// brush as heights along the brush normal on a grid out to 1.75 r, a quartic fitted to them that keeps the form, what
/// departs from it blurred at about a quarter of the radius, and each vertex moved along the brush normal only, its
/// share and falloff of the way to that. Nothing slides sideways, a ball stays a ball, and the strength is the same at
/// any density. Vertices facing away from the brush normal are left out.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FormSmooth;

impl Brush for FormSmooth {
    fn name(&self) -> &'static str {
        "Smooth"
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let n = dab.normal.normalize_or_zero();
        if n == Vec3::ZERO {
            return Ok(());
        }
        let r = s.radius;
        let Some(target) = SmoothTarget::new(doc, dab.center, n, r) else { return Ok(()) };
        let (fade, falloff) = ((s.strength * dab.pressure).clamp(0.0, 1.0), s.falloff);
        let disp = doc.compute_displacements(dab.center, r, |doc, v, _| {
            if doc.normals[v].dot(n) <= 0.0 {
                return None;
            }
            let q = doc.positions[v] - dab.center;
            let t = (q - n * q.dot(n)).length() / r;
            let f = falloff.eval(t) * fade;
            if f <= 0.0 {
                return None;
            }
            target.toward(q).map(|d| n * (d * f))
        });
        doc.apply_displacements(&disp)
    }
}

/// Smoothing on a layer: [`FormSmooth`] on the active layer's own offsets, apart from the base and the other layers.
/// Each offset's part along the brush normal goes toward the same at its neighbours (the form at the brush's scale
/// kept), and the change is the layer's whatever its blend, opacity or mask: a layer smoothed in Add shows smoothed in
/// Subtract, and smoothing in either does the same to it. Bumps the base or other layers hold are left alone.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LayerSmooth;

impl LayerSmooth {

    /// Whether vertex `v`'s place on the surface under the layer ([`Under`]) is near the dab and faces `n`, that
    /// surface's normal there. What the layer itself shows plays no part: on a layer's steep or folded sides the shown
    /// surface turns sideways, and smoothing along that pushed offsets sideways (shards).
    fn base_facing(doc: &Document, under: &Under, v: usize, center: Vec3, n: Vec3, r: f32) -> bool {
        let depth = 0.35 * r;
        let h = (under.pos(doc, v) - center).dot(n);
        h <= depth && h >= -2.0 * depth && under.normal(doc, v).dot(n) > 0.2
    }

    /// The surface under the layer about the dab: its normal (at its vertices shown within `r` of `center` facing
    /// `seed`'s way), and the dab's centre on it (`center` less what the active layer and those above move them by).
    pub(crate) fn base_about(doc: &Document, under: &Under, center: Vec3, seed: Vec3, r: f32) -> Option<(Vec3, Vec3)> {
        use rayon::prelude::*;
        let mut leaves = Vec::new();
        doc.bvh.query_sphere(center, r, &mut leaves);
        let r2 = r * r;
        let (sum, shift, w) = leaves
            .par_iter()
            .map(|&l| {
                let mut acc = (Vec3::ZERO, Vec3::ZERO, 0.0f32);
                for v in doc.bvh.leaves[l as usize].owned_range() {
                    let d2 = doc.positions[v].distance_squared(center);
                    if d2 > r2 {
                        continue;
                    }
                    let bn = under.normal(doc, v);
                    if bn.dot(seed) > 0.0 {
                        let w = 1.0 - d2 / r2;
                        acc.0 += bn * w;
                        acc.1 += (doc.positions[v] - under.pos(doc, v)) * w;
                        acc.2 += w;
                    }
                }
                acc
            })
            .reduce(|| (Vec3::ZERO, Vec3::ZERO, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let n = sum.normalize_or_zero();
        (n != Vec3::ZERO && w > 0.0).then(|| (n, center - shift / w))
    }
}

impl Brush for LayerSmooth {
    fn name(&self) -> &'static str {
        "Smooth"
    }

    /// Heights are the offsets along the base's normal about the dab, measured over the base's own places, and they
    /// move along it. The offsets' parts across it go toward their neighbours' too, so sideways shards smooth away.
    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let seed = dab.normal.normalize_or_zero();
        if seed == Vec3::ZERO || doc.active_layer().is_none() {
            return Ok(());
        }
        let r = s.radius;
        let under = doc.under(dab.center, r);
        let Some((n, c)) = LayerSmooth::base_about(doc, &under, dab.center, seed, r) else { return Ok(()) };
        // (Shown, the vertices whose place under the layer is under the dab sit about as far off as the layers move
        // the centre, or twice that where they hold more than about it.)
        let shift = 2.0 * dab.center.distance(c);
        drop(under);
        let under = doc.under(c, 1.9 * r + shift);
        let Some(target) = SmoothTarget::of_layer(doc, &under, c, n, r, 1.9 * r + shift) else { return Ok(()) };
        let (fade, falloff) = ((s.strength * dab.pressure).clamp(0.0, 1.0), s.falloff);
        let chunks = doc.active_layer_chunks().unwrap_or(&[]);
        let across = |o: Vec3| o - n * o.dot(n);
        let disp = doc.compute_displacements(c, 1.15 * r + shift, |doc, v, _| {
            let at = under.pos(doc, v);
            let q = at - c;
            let t = (q - n * q.dot(n)).length() / r;
            if t >= 1.0 || !LayerSmooth::base_facing(doc, &under, v, c, n, r) {
                return None;
            }
            let f = falloff.eval(t) * fade;
            if f <= 0.0 {
                return None;
            }
            let off = doc.layer_offset(chunks, v);
            let p = SmoothTarget::layer_point(at, off, c, n) - c;
            let along = n * (target.toward(p).unwrap_or(0.0) * f);
            // The part across, where there is one (strokes along the base's normal make none): toward the mean of the
            // neighbours' (each vertex's ring through its faces).
            let own = across(off);
            let mut side = Vec3::ZERO;
            if own.length_squared() > (1e-3 * r) * (1e-3 * r) {
                let (mut sum, mut k) = (Vec3::ZERO, 0.0f32);
                for &fi in doc.topo.vert_faces.row(v) {
                    let face = &doc.faces[fi as usize];
                    for &u in &face[..crate::mesh::face_len(face)] {
                        let u = u as usize;
                        if u != v {
                            sum += across(doc.layer_offset(chunks, u));
                            k += 1.0;
                        }
                    }
                }
                if k > 0.0 {
                    side = (sum / k - own) * f;
                }
            }
            let d = along + side;
            (d != Vec3::ZERO).then_some(d)
        });
        doc.apply_displacements(&disp.raw())
    }
}

/// The smoothed surface under a dab (see [`FormSmooth`]), as how far each point under it is from its smoothed self
/// along the brush normal.
pub struct SmoothTarget {
    n: Vec3,
    t1: Vec3,
    t2: Vec3,
    step: f32,
    half: i32,
    /// `1 / (1.5 r)`: the fit's coordinates.
    inv_fit: f32,
    fit: [f64; 15],
    /// What departs from the fit, smoothed: `1 - (1 - blur)^3` of it, which keeps any polynomial up to the fifth degree.
    smoothed: Vec<f32>,
}

impl SmoothTarget {
    /// About this many columns either way out to 1.75 r, fewer where the mesh is coarser than that.
    const HALF: f32 = 20.0;

    pub fn new(doc: &Document, center: Vec3, n: Vec3, r: f32) -> Option<SmoothTarget> {
        // (The vertices shown near the dab, facing it.)
        let depth = 0.35 * r;
        let keep = |v: usize| {
            let shown = (doc.positions[v] - center).dot(n);
            doc.normals[v].dot(n) > 0.0 && shown <= depth && shown >= -2.0 * depth
        };
        let reach = ((1.75 * r).powi(2) + 4.0 * depth * depth).sqrt();
        Self::gather(doc, center, n, r, reach, keep, |v, _| doc.positions[v])
    }

    /// The same for the active layer's own offsets alone (see [`LayerSmooth`]): each vertex's height is its offset
    /// along `n`, across from where it sits under the layer, so nothing the base or other layers hold is smoothed. Its
    /// point is [`Self::layer_point`].
    /// `n` is the normal of the surface under the layer there, and only that surface near the dab facing it counts (see
    /// [`LayerSmooth`]); each vertex sits where that surface has it.
    pub(crate) fn of_layer(doc: &Document, under: &Under, center: Vec3, n: Vec3, r: f32, reach: f32) -> Option<SmoothTarget> {
        let keep = |v: usize| LayerSmooth::base_facing(doc, under, v, center, n, r);
        Self::gather(doc, center, n, r, reach, keep, |v, off| Self::layer_point(under.pos(doc, v), off.unwrap_or(Vec3::ZERO), center, n))
    }

    /// A layer vertex's point for [`Self::of_layer`]: on the plane through `center` across `n` above its base place
    /// `base`, raised by its offset `off` along `n`.
    #[inline]
    pub fn layer_point(base: Vec3, off: Vec3, center: Vec3, n: Vec3) -> Vec3 {
        base - n * (base - center).dot(n) + n * off.dot(n)
    }

    /// Samples of the surface facing along `n` over the disc out to 1.75 r about `center`, of the vertices shown from
    /// 0.35 r above it to 0.7 r below (as the volume's columns reach; a sheet behind, a fold or a far wall, is left out),
    /// as (across, across, height): each vertex's point from `point(v, the active layer's offset at v)`.
    fn gather(
        doc: &Document,
        center: Vec3,
        n: Vec3,
        r: f32,
        reach: f32,
        keep: impl Fn(usize) -> bool + Sync,
        point: impl Fn(usize, Option<Vec3>) -> Vec3 + Sync,
    ) -> Option<SmoothTarget> {
        use rayon::prelude::*;
        let (t1, t2) = n.any_orthonormal_pair();
        let out = 1.75 * r;
        let mut leaves = Vec::new();
        doc.bvh.query_sphere(center, reach, &mut leaves);
        let samples: Vec<Vec<[f32; 3]>> = leaves
            .par_iter()
            .map(|&l| {
                let mut list = Vec::new();
                let range = doc.bvh.leaves[l as usize].owned_range();
                let offs = doc.active_layer_offsets(l);
                for v in range.clone() {
                    if !keep(v) {
                        continue;
                    }
                    let q = point(v, offs.map(|o| o[v - range.start])) - center;
                    let (a, b, h) = (q.dot(t1), q.dot(t2), q.dot(n));
                    if a * a + b * b <= out * out {
                        list.push([a, b, h]);
                    }
                }
                list
            })
            .collect();
        Self::from_samples(samples, n, t1, t2, r)
    }

    fn from_samples(samples: Vec<Vec<[f32; 3]>>, n: Vec3, t1: Vec3, t2: Vec3, r: f32) -> Option<SmoothTarget> {
        use crate::height_grid::{blur, quartic_terms, solve};
        let out = 1.75 * r;
        // The columns are no closer than the mesh's edges, so each holds a vertex or two: the vertices under the radius
        // tell the spacing (an even triangle mesh has 2 / (sqrt 3 e^2) of them per unit of area).
        let inside = samples.iter().flatten().filter(|p| p[0] * p[0] + p[1] * p[1] < r * r).count();
        if inside == 0 {
            return None;
        }
        let edge = (2.0 * std::f32::consts::PI * r * r / (3f32.sqrt() * inside as f32)).sqrt();
        let step = (out / Self::HALF).max(edge);
        let half = (out / step).ceil() as i32 + 1;
        let size = 2 * half + 1;
        let cells = (size * size) as usize;
        // Each vertex's height spread over the four columns around it.
        let heights = Self::splat(samples.iter().flatten().map(|&[a, b, h]| (a, b, h)), step, half);
        // The form at the brush's scale: a quartic fitted to the heights out to 1.5 r (weighted down toward there).
        let scale = 1.5 * r / step;
        let at = |c: usize| (((c as i32 % size) - half) as f32 / scale, ((c as i32 / size) - half) as f32 / scale);
        let fit_to = |grid: &[f32]| {
            let (mut m, mut v) = ([[0.0f64; 15]; 15], [0.0f64; 15]);
            for (c, &h) in grid.iter().enumerate() {
                let (a, b) = at(c);
                let d2 = (a * a + b * b) as f64;
                if !h.is_finite() || d2 >= 1.0 {
                    continue;
                }
                let wt = (1.0 - d2) * (1.0 - d2);
                let f = quartic_terms(a as f64, b as f64);
                for j in 0..15 {
                    let wf = wt * f[j];
                    v[j] += wf * h as f64;
                    for k in j..15 {
                        m[j][k] += wf * f[k];
                    }
                }
            }
            for j in 0..15 {
                for k in 0..j {
                    m[j][k] = m[k][j];
                }
            }
            solve(m, v)
        };
        let eval = |fit: &[f64; 15], a: f32, b: f32| quartic_terms(a as f64, b as f64).iter().zip(fit.iter()).map(|(t, c)| t * c).sum::<f64>() as f32;
        let mut fit = fit_to(&heights)?;
        // What departs from the fit, each vertex's own. A column's mean height on a curved surface is not the height at
        // its middle, so the fit to the columns sits off the vertices by that (a little under a convex form): the
        // vertices' departures are fitted again and that taken into the fit. (Left in, it was taken away at the rim
        // below, where the departures fade, and each dab sank the surface there a little.)
        let inv_fit = 1.0 / (1.5 * r);
        let mut off = Self::splat(samples.iter().flatten().map(|&[a, b, h]| (a, b, h - eval(&fit, a * inv_fit, b * inv_fit))), step, half);
        let more = fit_to(&off)?;
        for (f, m) in fit.iter_mut().zip(more) {
            *f += m;
        }
        // Fading to nothing over the last sixth of where it was fitted, and nothing past it or where no surface was met
        // (the blur there reads the fitted form going on).
        for (c, h) in off.iter_mut().enumerate() {
            let (a, b) = at(c);
            let x = ((1.0 - (a * a + b * b).sqrt()) * 6.0).clamp(0.0, 1.0);
            *h = if h.is_finite() { (*h - eval(&more, a, b)) * x * x * (3.0 - 2.0 * x) } else { 0.0 };
        }
        let rho = ((r / step / 4.0).round() as i32).max(1);
        blur(&mut off, size, rho);
        let once = off.clone();
        blur(&mut off, size, rho);
        let twice = off.clone();
        blur(&mut off, size, rho);
        let smoothed = (0..cells).map(|c| 3.0 * (once[c] - twice[c]) + off[c]).collect();
        Some(SmoothTarget { n, t1, t2, step, half, inv_fit, fit, smoothed })
    }

    /// `values` at (across, across) spread over the four columns around each (`step` apart, `-half..=half` each way):
    /// each column their weighted mean, NaN where none reached it, unless three of the eight around it have one: then
    /// their mean (a column between vertices that missed it).
    fn splat(values: impl Iterator<Item = (f32, f32, f32)>, step: f32, half: i32) -> Vec<f32> {
        let size = 2 * half + 1;
        let cells = (size * size) as usize;
        let (mut wv, mut w) = (vec![0.0f32; cells], vec![0.0f32; cells]);
        for (a, b, value) in values {
            let (x, y) = (a / step + half as f32, b / step + half as f32);
            let (i, j) = (x.floor() as i32, y.floor() as i32);
            let (fx, fy) = (x - i as f32, y - j as f32);
            for (di, dj, k) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let (ci, cj) = (i + di, j + dj);
                if ci >= 0 && cj >= 0 && ci < size && cj < size {
                    let c = (cj * size + ci) as usize;
                    wv[c] += k * value;
                    w[c] += k;
                }
            }
        }
        let mut out: Vec<f32> = wv.iter().zip(&w).map(|(&s, &w)| if w > 1e-3 { s / w } else { f32::NAN }).collect();
        let holes: Vec<(usize, f32)> = (0..cells)
            .filter(|&c| out[c].is_nan())
            .filter_map(|c| {
                let (i, j) = ((c as i32) % size, (c as i32) / size);
                let (mut sum, mut k) = (0.0f32, 0);
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                    let (x, y) = (i + di, j + dj);
                    if x >= 0 && y >= 0 && x < size && y < size && out[(y * size + x) as usize].is_finite() {
                        sum += out[(y * size + x) as usize];
                        k += 1;
                    }
                }
                (k >= 3).then(|| (c, sum / k as f32))
            })
            .collect();
        for (c, v) in holes {
            out[c] = v;
        }
        out
    }

    /// How far the point `q` (from the dab's centre) is from its smoothed self along the brush normal (positive: it
    /// should rise); `None` past the grid.
    pub fn toward(&self, q: Vec3) -> Option<f32> {
        let (a, b, h) = (q.dot(self.t1), q.dot(self.t2), q.dot(self.n));
        let size = 2 * self.half + 1;
        let (x, y) = (a / self.step + self.half as f32, b / self.step + self.half as f32);
        let (i, j) = (x.floor() as i32, y.floor() as i32);
        if i < 0 || j < 0 || i + 1 >= size || j + 1 >= size {
            return None;
        }
        let (fx, fy) = (x - i as f32, y - j as f32);
        let g = |di: i32, dj: i32| self.smoothed[((j + dj) * size + i + di) as usize];
        let s = (g(0, 0) * (1.0 - fx) + g(1, 0) * fx) * (1.0 - fy) + (g(0, 1) * (1.0 - fx) + g(1, 1) * fx) * fy;
        let (fa, fb) = ((a * self.inv_fit) as f64, (b * self.inv_fit) as f64);
        let form = crate::height_grid::quartic_terms(fa, fb).iter().zip(self.fit.iter()).map(|(t, c)| t * c).sum::<f64>() as f32;
        Some(s - (h - form))
    }
}

// ----------------------------------------------------------------------- Move

/// ZBrush *Move*: grabs the vertices under the cursor once at stroke start,
/// with falloff weights, and drags them rigidly with the cursor. With
/// `topological` the selection grows by geodesic (edge) distance, so nearby
/// but unconnected or distant-along-surface parts (fingers, lips) stay put.
/// Per leaf: `(local vertex, weight, position at grab time)`.
type Grab = Vec<(u32, Vec<(u32, f32, Vec3)>)>;

/// The Move brush: grabs the vertices under the cursor as a stroke starts and drags them with it (see above).
/// (Old mesh-mode code: reached only from sculpt-cli and the tests, not the app.)
#[derive(Debug, Default)]
pub struct MoveBrush {
    pub topological: bool,
    grab: Option<Grab>,
}

impl MoveBrush {
    pub fn new(topological: bool) -> MoveBrush {
        MoveBrush { topological, grab: None }
    }

    pub fn begin(&mut self, doc: &Document, hit: &SurfaceHit, s: &BrushSettings) {
        let weighted: Vec<(u32, f32)> = if self.topological {
            doc.geodesic_distances(hit.vertex, s.radius)
                .into_iter()
                .map(|(v, d)| (v, s.falloff.eval(d / s.radius)))
                .collect()
        } else {
            let mut leaves = Vec::new();
            doc.bvh().query_sphere(hit.point, s.radius, &mut leaves);
            leaves
                .iter()
                .flat_map(|&l| doc.bvh().leaves[l as usize].owned_range())
                .filter_map(|v| {
                    let d = doc.positions()[v].distance(hit.point);
                    let facing = !s.front_faces_only || doc.normals()[v].dot(hit.normal) > 0.0;
                    (d < s.radius && facing).then(|| (v as u32, s.falloff.eval(d / s.radius)))
                })
                .collect()
        };
        let mut per_leaf: Grab = Vec::new();
        let mut sorted: Vec<(u32, f32)> = weighted
            .into_iter()
            .map(|(v, w)| (v, w * s.strength.max(0.0) * (1.0 - doc.freeze()[v as usize])))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        sorted.sort_unstable_by_key(|(v, _)| *v);
        for (v, w) in sorted {
            let l = doc.bvh().vert_leaf[v as usize];
            let local = v - doc.bvh().leaves[l as usize].owned.start;
            if per_leaf.last().is_none_or(|(pl, _)| *pl != l) {
                per_leaf.push((l, Vec::new()));
            }
            per_leaf.last_mut().unwrap().1.push((local, w, doc.positions()[v as usize]));
        }
        self.grab = Some(per_leaf);
    }

    /// Move the grabbed region by `offset` (total, since `begin`).
    pub fn drag(&mut self, doc: &mut Document, offset: Vec3) -> Result<()> {
        let Some(grab) = &self.grab else { return Ok(()) };
        let leaves = grab
            .iter()
            .map(|(l, items)| {
                let start = doc.bvh().leaves[*l as usize].owned.start as usize;
                let list = items
                    .iter()
                    .map(|&(k, w, orig)| (k, orig + offset * w - doc.positions()[start + k as usize]))
                    .collect();
                (*l, list)
            })
            .collect();
        doc.apply_displacements(&Displacements { leaves, raw: false })
    }

    pub fn end(&mut self) {
        self.grab = None;
    }

    pub fn grabbed_vertices(&self) -> usize {
        self.grab.as_ref().map_or(0, |g| g.iter().map(|(_, l)| l.len()).sum())
    }
}

// ---------------------------------------------------------------------- Paint

/// Paints freeze or a mask channel (towards `value`).
/// (Old mesh-mode code: reached only from sculpt-cli and the tests, not the app.)
#[derive(Clone, Debug)]
pub struct PaintBrush {
    pub target: PaintTarget,
    pub value: f32,
}

impl Brush for PaintBrush {
    fn name(&self) -> &'static str {
        "Paint"
    }

    fn dab(&mut self, doc: &mut Document, dab: &Dab, s: &BrushSettings) -> Result<()> {
        let value = if s.invert { 1.0 - self.value } else { self.value };
        let front = s.front_faces_only.then_some(dab.normal);
        doc.paint(&self.target, dab.center, s.radius, &s.falloff, s.strength * dab.pressure, value, front);
        Ok(())
    }
}

// --------------------------------------------------------------------- Stroke

/// Turns a stream of surface samples into evenly spaced dabs.
/// (Old mesh-mode code: reached only from sculpt-cli and the tests, not the app.)
#[derive(Debug)]
pub struct StrokeSampler {
    pub spacing: f32,
    last: Option<(Vec3, f32)>,
    carried: f32,
}

impl StrokeSampler {
    pub fn new(spacing: f32) -> StrokeSampler {
        StrokeSampler { spacing: spacing.max(1e-6), last: None, carried: 0.0 }
    }

    /// Returns `(point, direction, pressure)` for each dab to emit.
    pub fn push(&mut self, point: Vec3, pressure: f32) -> Vec<(Vec3, Vec3, f32)> {
        let Some((last, last_p)) = self.last else {
            self.last = Some((point, pressure));
            return vec![(point, Vec3::ZERO, pressure)];
        };
        let seg = point - last;
        let len = seg.length();
        if len <= 0.0 {
            return Vec::new();
        }
        let dir = seg / len;
        let mut out = Vec::new();
        let mut t = self.spacing - self.carried;
        while t <= len {
            let a = t / len;
            out.push((last + seg * a, dir, last_p + (pressure - last_p) * a));
            t += self.spacing;
        }
        self.carried = len - (t - self.spacing);
        self.last = Some((point, pressure));
        out
    }
}

/// Apply a brush along a path of points near the surface (one undo step).
/// Each dab is projected onto the current surface before being applied.
pub fn stroke(doc: &mut Document, brush: &mut dyn Brush, s: &BrushSettings, path: &[(Vec3, f32)]) -> Result<usize> {
    doc.begin_stroke(brush.name());
    let mut sampler = StrokeSampler::new(s.radius * s.spacing);
    let mut count = 0;
    let result = (|| {
        for &(p, pressure) in path {
            for (point, dir, pr) in sampler.push(p, pressure) {
                let guess = doc.sample_area(point, s.radius, None).map_or(Vec3::Z, |(_, n)| n);
                let Some(hit) = doc.project_to_surface(point, guess, s.radius) else { continue };
                brush.dab(doc, &Dab { center: hit.point, normal: hit.normal, direction: dir, pressure: pr }, s)?;
                count += 1;
            }
        }
        Ok(count)
    })();
    doc.end_stroke();
    result
}

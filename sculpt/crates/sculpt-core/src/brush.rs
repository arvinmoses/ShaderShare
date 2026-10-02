//! ZBrush-feel brushes.
//!
//! The "feel" of ZBrush brushes comes mostly from *where the target plane is*
//! and *how the footprint is shaped*, so each brush below documents those
//! choices explicitly. All brushes run through
//! [`Document::compute_displacements`] (parallel, read-only) followed by
//! [`Document::apply_displacements`] (writes to the active layer).

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::document::{Displacements, Document, PaintTarget, SurfaceHit};
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
            let amount = (-h * falloff.eval(d) * fade).min(cap - risen);
            (amount > 0.0).then(|| n * amount)
        });
        doc.apply_displacements(&disp)
    }
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
                d -= n * (sign * h * falloff.eval(t) * fade);
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

// ----------------------------------------------------------------------- Move

/// ZBrush *Move*: grabs the vertices under the cursor once at stroke start,
/// with falloff weights, and drags them rigidly with the cursor. With
/// `topological` the selection grows by geodesic (edge) distance, so nearby
/// but unconnected or distant-along-surface parts (fingers, lips) stay put.
/// Per leaf: `(local vertex, weight, position at grab time)`.
type Grab = Vec<(u32, Vec<(u32, f32, Vec3)>)>;

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
        doc.apply_displacements(&Displacements { leaves })
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

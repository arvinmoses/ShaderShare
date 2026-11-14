//! One piece of the mesh collapsed edge by edge, cheapest first, every collapse guarded so the result stays manifold.
//!
//! A piece holds whole faces. A vertex that also touches a face outside the piece is locked: it is never removed and
//! never moved, and no edge with a locked end collapses, so every collapse sees the full neighbourhood of both ends.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::{DVec3, Vec3};
use smallvec::SmallVec;

use super::quadric::{Quadric, d3};

#[derive(Clone, Copy, Debug)]
pub(super) struct Params {
    /// 0: quadric error only; 1: edge length only.
    pub evenness: f64,
    /// Added to the quadric error before blending (so flat pieces still order by length).
    pub eps: f64,
    pub max_valence: usize,
    /// A face's normal may turn at most this far: `cos` of the angle.
    pub min_dot: f64,
    /// Faces whose shape measure (1 equilateral, 0 a sliver) falls below this, and below what it was, are refused.
    pub min_quality: f64,
}

impl Params {
    /// How strongly the new point is pulled toward the edge's midpoint.
    fn pull(&self) -> f64 {
        let e = self.evenness.clamp(0.0, 1.0);
        1e-3 + e / (1.0 - e + 1e-4)
    }

    pub fn cost(&self, q: f64, len2: f64) -> f64 {
        let e = self.evenness.clamp(0.0, 1.0);
        let q = q + self.eps;
        let l = len2 * len2 + self.eps;
        if e <= 0.0 {
            q
        } else if e >= 1.0 {
            l
        } else {
            (q.ln() * (1.0 - e) + l.ln() * e).exp()
        }
    }
}

/// One collapse: `removed` merged into `kept`, which moved to `pos`; `dead` the faces it removed (global ids).
#[derive(Clone, Copy, Debug)]
pub(super) struct Rec {
    pub removed: u32,
    pub kept: u32,
    pub pos: Vec3,
    pub cost: f64,
    pub dead: [u32; 2],
}

#[derive(Clone, Copy)]
struct Cand {
    cost: f64,
    r: u32,
    k: u32,
    vr: u32,
    vk: u32,
    x: Vec3,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    // Min-heap on cost.
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.total_cmp(&self.cost)
    }
}

pub(super) struct Engine {
    pub gv: Vec<u32>,
    pub pos: Vec<Vec3>,
    pub quad: Vec<Quadric>,
    locked: Vec<bool>,
    border: Vec<bool>,
    pub alive: Vec<bool>,
    ver: Vec<u32>,
    pub gf: Vec<u32>,
    pub faces: Vec<[u32; 3]>,
    pub falive: Vec<bool>,
    vf: Vec<SmallVec<[u32; 8]>>,
    heap: BinaryHeap<Cand>,
    pub recs: Vec<Rec>,
    pub live_faces: usize,
    p: Params,
}

type Ring = SmallVec<[u32; 16]>;

impl Engine {
    /// `faces` index the local vertices `gv`; `locked`/`border` per local vertex.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gv: Vec<u32>,
        pos: Vec<Vec3>,
        quad: Vec<Quadric>,
        locked: Vec<bool>,
        border: Vec<bool>,
        gf: Vec<u32>,
        faces: Vec<[u32; 3]>,
        p: Params,
    ) -> Engine {
        let n = gv.len();
        let mut vf: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); n];
        for (i, f) in faces.iter().enumerate() {
            for &v in f {
                vf[v as usize].push(i as u32);
            }
        }
        let nf = faces.len();
        Engine {
            gv,
            pos,
            quad,
            locked,
            border,
            alive: vec![true; n],
            ver: vec![0; n],
            gf,
            faces,
            falive: vec![true; nf],
            vf,
            heap: BinaryHeap::new(),
            recs: Vec::new(),
            live_faces: nf,
            p,
        }
    }

    /// Fill the queue with every collapsible edge.
    pub fn seed(&mut self) {
        let mut cands = Vec::new();
        for v in 0..self.gv.len() as u32 {
            if self.locked[v as usize] {
                continue;
            }
            for n in self.ring(v) {
                if n > v && !self.locked[n as usize] {
                    if let Some(c) = self.evaluate(v, n) {
                        cands.push(c);
                    }
                }
            }
        }
        self.heap = BinaryHeap::from(cands);
    }

    /// Costs of everything queued (to pick a round's threshold).
    pub fn queued_costs(&self, out: &mut Vec<f64>) {
        out.extend(self.heap.iter().map(|c| c.cost));
    }

    fn ring(&self, v: u32) -> Ring {
        let mut r: Ring = SmallVec::new();
        for &f in &self.vf[v as usize] {
            for &u in &self.faces[f as usize] {
                if u != v {
                    r.push(u);
                }
            }
        }
        r.sort_unstable();
        r.dedup();
        r
    }

    fn evaluate(&self, a: u32, b: u32) -> Option<Cand> {
        let (ba, bb) = (self.border[a as usize], self.border[b as usize]);
        let shared = self.vf[a as usize].iter().filter(|&&f| self.faces[f as usize].contains(&b)).count();
        if shared == 0 || shared > 2 || (ba && bb && shared != 1) {
            return None;
        }
        let (pa, pb) = (self.pos[a as usize], self.pos[b as usize]);
        let q = Quadric::sum(&self.quad[a as usize], &self.quad[b as usize]);
        let (r, k, x) = if ba && !bb {
            (b, a, pa)
        } else if bb && !ba {
            (a, b, pb)
        } else {
            let x = q.minimise_toward((d3(pa) + d3(pb)) * 0.5, self.p.pull()).as_vec3();
            (a, b, x)
        };
        let len2 = d3(pa).distance_squared(d3(pb));
        let cost = self.p.cost(q.eval(d3(x)), len2);
        Some(Cand { cost, r, k, vr: self.ver[r as usize], vk: self.ver[k as usize], x })
    }

    fn push_edges(&mut self, v: u32) {
        if self.locked[v as usize] {
            return;
        }
        for n in self.ring(v) {
            if !self.locked[n as usize] {
                if let Some(c) = self.evaluate(v, n) {
                    self.heap.push(c);
                }
            }
        }
    }

    fn valence_floor(&self, v: u32) -> usize {
        if self.border[v as usize] { 2 } else { 3 }
    }

    /// Would collapsing `r` into `k` at `x` keep the mesh manifold and unfolded?
    fn allowed(&self, r: u32, k: u32, x: Vec3) -> bool {
        let shared: SmallVec<[u32; 2]> =
            self.vf[r as usize].iter().copied().filter(|&f| self.faces[f as usize].contains(&k)).collect();
        if shared.is_empty() || shared.len() > 2 {
            return false;
        }
        let mut opp: SmallVec<[u32; 2]> = shared
            .iter()
            .map(|&f| *self.faces[f as usize].iter().find(|&&u| u != r && u != k).unwrap())
            .collect();
        opp.sort_unstable();
        if opp.len() == 2 && opp[0] == opp[1] {
            return false;
        }
        let (rr, rk) = (self.ring(r), self.ring(k));
        // Link condition: the ends share exactly the vertices opposite the edge.
        let mut common: SmallVec<[u32; 4]> = SmallVec::new();
        let (mut i, mut j) = (0, 0);
        while i < rr.len() && j < rk.len() {
            match rr[i].cmp(&rk[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    common.push(rr[i]);
                    if common.len() > 2 {
                        return false;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        if common.as_slice() != opp.as_slice() {
            return false;
        }
        let new_val = rr.len() + rk.len() - common.len() - 2;
        if new_val < self.valence_floor(k) || new_val > self.p.max_valence {
            return false;
        }
        for &c in &opp {
            if self.ring(c).len() - 1 < self.valence_floor(c) {
                return false;
            }
        }
        // Faces that stay: no flip, no fold, no new sliver.
        let xd = d3(x);
        for &v in &[r, k] {
            for &f in &self.vf[v as usize] {
                if shared.contains(&f) || (v == k && self.faces[f as usize].contains(&r)) {
                    continue;
                }
                let t = self.faces[f as usize];
                let p = t.map(|u| d3(self.pos[u as usize]));
                let q = t.map(|u| if u == r || u == k { xd } else { d3(self.pos[u as usize]) });
                let n0 = (p[1] - p[0]).cross(p[2] - p[0]);
                let n1 = (q[1] - q[0]).cross(q[2] - q[0]);
                let (l0, l1) = (n0.length(), n1.length());
                if l1 <= 1e-30 || n0.dot(n1) < self.p.min_dot * l0 * l1 {
                    return false;
                }
                let qn = quality(&q, l1);
                if qn < self.p.min_quality && qn < quality(&p, l0) {
                    return false;
                }
            }
        }
        true
    }

    /// Collapse until the cheapest edge costs more than `tau` or only `floor` faces are left.
    pub fn run(&mut self, tau: f64, floor: usize) {
        while self.live_faces > floor {
            let Some(c) = self.heap.pop() else { break };
            if c.cost > tau {
                self.heap.push(c);
                break;
            }
            let (r, k) = (c.r as usize, c.k as usize);
            if !self.alive[r] || !self.alive[k] || self.ver[r] != c.vr || self.ver[k] != c.vk {
                continue;
            }
            if !self.allowed(c.r, c.k, c.x) {
                continue;
            }
            self.apply(c);
        }
    }

    fn apply(&mut self, c: Cand) {
        let (r, k) = (c.r, c.k);
        let mut dead = [u32::MAX; 2];
        let mut nd = 0;
        let rfaces = std::mem::take(&mut self.vf[r as usize]);
        for &f in &rfaces {
            let t = self.faces[f as usize];
            if t.contains(&k) {
                self.falive[f as usize] = false;
                dead[nd] = self.gf[f as usize];
                nd += 1;
                for &u in &t {
                    if u != r {
                        self.vf[u as usize].retain(|g| *g != f);
                    }
                }
            } else {
                for u in self.faces[f as usize].iter_mut() {
                    if *u == r {
                        *u = k;
                    }
                }
                self.vf[k as usize].push(f);
            }
        }
        self.live_faces -= nd;
        self.alive[r as usize] = false;
        self.pos[k as usize] = c.x;
        let qr = self.quad[r as usize];
        self.quad[k as usize].add(&qr);
        if self.border[r as usize] {
            self.border[k as usize] = true;
        }
        self.ver[k as usize] += 1;
        self.ver[r as usize] += 1;
        self.recs.push(Rec { removed: self.gv[r as usize], kept: self.gv[k as usize], pos: c.x, cost: c.cost, dead });
        self.push_edges(k);
    }
}

/// 4√3·area / Σ edge² : 1 for an equilateral triangle, 0 for a sliver. `twice_area` = |cross|.
fn quality(p: &[DVec3; 3], twice_area: f64) -> f64 {
    let s = p[0].distance_squared(p[1]) + p[1].distance_squared(p[2]) + p[2].distance_squared(p[0]);
    if s <= 0.0 { 0.0 } else { 2.0 * 3f64.sqrt() * twice_area / s }
}

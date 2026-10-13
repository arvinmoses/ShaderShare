// GPU displacement boolean (Generator Boolean layer, spike 1).
//
// A WGSL port of sculpt-core/src/generator (mod.rs, sdf.rs, instances.rs). Same rules:
//   Subtract: project vertices inside any cutter onto the cutter surface (<= iterations steps).
//   Intersect: pull vertices that are outside every nearby instance but within `reach` back in.
//   Add: heightfield union along the vertex normal (ray to the primitive, exit or entry point).
//
// Passes (all in one command buffer):
//   clear_heads, build_grid   uniform grid over instance AABBs (dense, linked-list cells)
//   mark_leaves               one workgroup per BVH leaf; appends leaves whose bounds touch an
//                             occupied cell to `work` and counts them in args[0] (indirect args)
//   clear_delta, eval         one thread per vertex of the active leaves; writes delta = new - old

struct Params {
    origin: vec4<f32>, // xyz grid origin, w = 1 / cell
    dims: vec4<u32>,   // grid dims xyz, w = instance count
    cfg: vec4<u32>,    // op (0 add, 1 subtract, 2 intersect), iterations, leaf count, vertex count
    misc: vec4<f32>,   // x = reach
    pool: vec4<u32>,   // x = entry capacity, y = chunks per leaf (dispatch y), z = cell count
}

struct Inst {
    c0: vec4<f32>,  // rotation column 0, w = translation x
    c1: vec4<f32>,
    c2: vec4<f32>,
    sc: vec4<f32>,  // xyz = 1 / scale, w = min scale
    bmin: vec4<f32>, // inflated AABB
    bmax: vec4<f32>,
    lo: vec4<u32>,  // first grid cell xyz, w = primitive | ball << 8
    hi: vec4<u32>,  // last grid cell xyz
}

struct Leaf {
    a: vec4<u32>,   // x = first owned vertex, y = owned count
    bmin: vec4<f32>,
    bmax: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> insts: array<Inst>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<u32>>;   // 0 = empty, else entry + 1
@group(0) @binding(3) var<storage, read_write> entries: array<vec2<u32>>;   // (instance, next entry + 1)
@group(0) @binding(4) var<storage, read> mesh: array<u32>;                  // per vertex 16 B: pos xyz (f32 bits), octahedral normal (2 x snorm16)
@group(0) @binding(5) var<storage, read_write> delta: array<f32>;           // per vertex: xyz
@group(0) @binding(6) var<storage, read> leaves: array<Leaf>;
@group(0) @binding(7) var<storage, read_write> work: array<u32>;            // active leaf ids
@group(0) @binding(8) var<storage, read_write> args: array<atomic<u32>>;    // [dispatch x, y, z, entry counter]

const NONE: u32 = 0xffffffffu;
const LIN: u32 = 262144u; // 4096 workgroups * 64 threads: row stride of 2D linearised dispatches

fn lin(gid: vec3<u32>) -> u32 {
    return gid.y * LIN + gid.x;
}

// ---------------------------------------------------------------- primitives

fn sgn(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0); // Rust's f32::signum (signum(0) = 1)
}

fn nrm_or(v: vec3<f32>, fb: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 1e-37 {
        return v * inverseSqrt(l2);
    }
    return fb;
}

fn extrude(d2: f32, z: f32) -> f32 {
    let w = vec2<f32>(d2, abs(z) - 1.0);
    return min(max(w.x, w.y), 0.0) + length(max(w, vec2<f32>(0.0)));
}

fn sd_capped_cone(p: vec3<f32>) -> f32 {
    // h = 1, r1 = 1, r2 = 0
    let q = vec2<f32>(length(p.xz), p.y);
    let k1 = vec2<f32>(0.0, 1.0);
    let k2 = vec2<f32>(-1.0, 2.0);
    let ca = vec2<f32>(q.x - min(q.x, select(0.0, 1.0, q.y < 0.0)), abs(q.y) - 1.0);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

fn sd_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sgn(e0.x * e2.y - e0.y * e2.x);
    let a = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let b = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let c = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = min(min(a, b), c);
    return -sqrt(d.x) * sgn(d.y);
}

// Prim ids follow Primitive::ALL: 0 Box, 1 Sphere, 2 Cylinder, 3 Cone, 4 Wedge, 5 ChamferCutter, 6 Torus.
fn prim_sdf(prim: u32, p: vec3<f32>) -> f32 {
    switch prim {
        case 0u: {
            let q = abs(p) - vec3<f32>(1.0);
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        }
        case 1u: {
            return length(p) - 1.0;
        }
        case 2u: {
            let d = vec2<f32>(length(p.xz) - 1.0, abs(p.y) - 1.0);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case 3u: {
            return sd_capped_cone(p);
        }
        case 4u: {
            let d2 = sd_triangle(p.xy, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0));
            return extrude(d2, p.z);
        }
        case 5u: {
            let s = 0.70710678;
            let u = (p.x + p.y) * s;
            let v = (p.x - p.y) * s;
            let q = vec2<f32>(abs(u) - s, abs(v) - s);
            let d2 = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
            return extrude(d2, p.z);
        }
        default: {
            let q = vec2<f32>(length(p.xz) - 1.0, p.y);
            return length(q) - 0.3;
        }
    }
}

// (distance, unit gradient) in the primitive's local frame.
fn prim_grad(prim: u32, p: vec3<f32>) -> vec4<f32> {
    if prim == 1u {
        let l = length(p);
        return vec4<f32>(l - 1.0, select(vec3<f32>(0.0, 1.0, 0.0), p / l, l > 1e-12));
    }
    if prim == 0u {
        let q = abs(p) - vec3<f32>(1.0);
        let s = vec3<f32>(sgn(p.x), sgn(p.y), sgn(p.z));
        let d = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        var g: vec3<f32>;
        if max(q.x, max(q.y, q.z)) > 0.0 {
            g = nrm_or(s * max(q, vec3<f32>(0.0)), vec3<f32>(0.0, 1.0, 0.0));
        } else {
            g = vec3<f32>(0.0);
            // glam's max_position: first index of the maximum
            if q.x >= q.y {
                if q.x >= q.z { g.x = s.x; } else { g.z = s.z; }
            } else if q.y >= q.z {
                g.y = s.y;
            } else {
                g.z = s.z;
            }
        }
        return vec4<f32>(d, g);
    }
    let e = 1e-3;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0);
    let g = k0 * prim_sdf(prim, p + k0 * e) + k1 * prim_sdf(prim, p + k1 * e) + k2 * prim_sdf(prim, p + k2 * e) + k3 * prim_sdf(prim, p + k3 * e);
    return vec4<f32>(prim_sdf(prim, p), nrm_or(g, vec3<f32>(0.0, 1.0, 0.0)));
}

// ---------------------------------------------------------------- instances

fn inst_local(ins: Inst, p: vec3<f32>) -> vec3<f32> {
    let d = p - vec3<f32>(ins.c0.w, ins.c1.w, ins.c2.w);
    return vec3<f32>(dot(ins.c0.xyz, d), dot(ins.c1.xyz, d), dot(ins.c2.xyz, d)) * ins.sc.xyz;
}

fn inst_sdf(i: u32, p: vec3<f32>) -> f32 {
    let ins = insts[i];
    if (ins.lo.w >> 8u) != 0u {
        return length(p - vec3<f32>(ins.c0.w, ins.c1.w, ins.c2.w)) - ins.sc.w;
    }
    return prim_sdf(ins.lo.w & 255u, inst_local(ins, p)) * ins.sc.w;
}

// xyz = world gradient, w = distance
fn inst_grad(i: u32, p: vec3<f32>) -> vec4<f32> {
    let ins = insts[i];
    let r = prim_grad(ins.lo.w & 255u, inst_local(ins, p));
    let g = r.yzw * ins.sc.xyz;
    let w = ins.c0.xyz * g.x + ins.c1.xyz * g.y + ins.c2.xyz * g.z;
    return vec4<f32>(nrm_or(w, vec3<f32>(0.0, 1.0, 0.0)), r.x * ins.sc.w);
}

fn aabb_contains(i: u32, p: vec3<f32>) -> bool {
    let ins = insts[i];
    return all(p >= ins.bmin.xyz) && all(p <= ins.bmax.xyz);
}

// Head of the cell list containing p (0 = none).
fn cell_head(p: vec3<f32>) -> u32 {
    let c = (p - P.origin.xyz) * P.origin.w;
    let d = vec3<f32>(P.dims.xyz);
    if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 || c.x >= d.x || c.y >= d.y || c.z >= d.z {
        return 0u;
    }
    let ci = vec3<u32>(c);
    return atomicLoad(&heads[ci.x + P.dims.x * (ci.y + P.dims.y * ci.z)]);
}

// ---------------------------------------------------------------- Add: ray to the primitive

// Distance along n from p to the surface the Add rule pulls to, or -1 for none.
fn ray_surface(i: u32, p: vec3<f32>, n: vec3<f32>, reach: f32) -> f32 {
    let scale = insts[i].sc.w;
    let eps = 1e-4 * scale;
    var t = 0.0;
    var d = inst_sdf(i, p);
    if d > eps {
        for (var k = 0; k < 32; k++) {
            t += d;
            if t > reach {
                return -1.0;
            }
            d = inst_sdf(i, p + n * t);
            if d <= eps {
                return t;
            }
        }
        return -1.0;
    }
    let step_min = 0.05 * scale;
    var b = t;
    var a = t;
    for (var k = 0; k < 64; k++) {
        a = b;
        b += max(-d, step_min);
        d = inst_sdf(i, p + n * b);
        if d > 0.0 {
            for (var j = 0; j < 8; j++) {
                let m = 0.5 * (a + b);
                if inst_sdf(i, p + n * m) > 0.0 {
                    b = m;
                } else {
                    a = m;
                }
            }
            return 0.5 * (a + b);
        }
    }
    return -1.0;
}

// ---------------------------------------------------------------- per-vertex evaluation

const NEAR_MAX: u32 = 16u;

fn eval_vertex(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let head = cell_head(p);
    if head == 0u {
        return p;
    }
    let iters = P.cfg.y;
    let op = P.cfg.x;
    if op == 1u {
        // Subtract. Pass 1 at p: most-penetrating instance + candidates that could still matter.
        var near_i: array<u32, 16>;
        var near_d: array<f32, 16>;
        var nn = 0u;
        var overflow = false;
        var best_d = 0.0;
        var best_i = NONE;
        var e = head;
        loop {
            if e == 0u { break; }
            let ent = entries[e - 1u];
            e = ent.y;
            let i = ent.x;
            if !aabb_contains(i, p) { continue; }
            let d = inst_sdf(i, p);
            if d < best_d {
                best_d = d;
                best_i = i;
            }
            if nn < NEAR_MAX {
                near_i[nn] = i;
                near_d[nn] = d;
                nn++;
            } else {
                overflow = true;
            }
        }
        if best_i == NONE {
            return p;
        }
        let horizon = f32(iters) * -best_d;
        var q = p - inst_grad(best_i, p).xyz * best_d;
        for (var it = 1u; it < iters; it++) {
            var bd = 0.0;
            var bi = NONE;
            if !overflow {
                for (var k = 0u; k < nn; k++) {
                    if near_d[k] > horizon { continue; }
                    let d = inst_sdf(near_i[k], q);
                    if d < bd {
                        bd = d;
                        bi = near_i[k];
                    }
                }
            } else {
                // More than NEAR_MAX candidates: walk the cell again (pruning is only an optimisation).
                var e2 = head;
                loop {
                    if e2 == 0u { break; }
                    let ent = entries[e2 - 1u];
                    e2 = ent.y;
                    let d = inst_sdf(ent.x, q);
                    if d < bd {
                        bd = d;
                        bi = ent.x;
                    }
                }
            }
            if bi == NONE { break; }
            q -= inst_grad(bi, q).xyz * bd;
        }
        return q;
    }
    if op == 2u {
        // Intersect.
        var q = p;
        let reach = P.misc.x;
        for (var it = 0u; it < iters; it++) {
            var bd = 3.0e38;
            var bg = vec3<f32>(0.0);
            var e = head;
            loop {
                if e == 0u { break; }
                let ent = entries[e - 1u];
                e = ent.y;
                if !aabb_contains(ent.x, q) { continue; }
                let r = inst_grad(ent.x, q);
                if r.w < bd {
                    bd = r.w;
                    bg = r.xyz;
                }
            }
            if bd <= 0.0 || bd > reach { break; }
            q -= bg * bd;
        }
        return q;
    }
    // Add.
    var best_t = -1.0;
    var e = head;
    loop {
        if e == 0u { break; }
        let ent = entries[e - 1u];
        e = ent.y;
        if !aabb_contains(ent.x, p) { continue; }
        best_t = max(best_t, ray_surface(ent.x, p, n, P.misc.x));
    }
    if best_t >= 0.0 {
        return p + n * best_t;
    }
    return p;
}

// ---------------------------------------------------------------- entry points

@compute @workgroup_size(64)
fn clear_heads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = lin(gid);
    if i < P.pool.z {
        atomicStore(&heads[i], 0u);
    }
}

@compute @workgroup_size(64)
fn clear_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = lin(gid);
    if i < P.cfg.w * 3u {
        delta[i] = 0.0;
    }
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= P.dims.w {
        return;
    }
    let ins = insts[id];
    for (var z = ins.lo.z; z <= ins.hi.z; z++) {
        for (var y = ins.lo.y; y <= ins.hi.y; y++) {
            for (var x = ins.lo.x; x <= ins.hi.x; x++) {
                let e = atomicAdd(&args[3], 1u);
                if e < P.pool.x {
                    let cell = x + P.dims.x * (y + P.dims.y * z);
                    let prev = atomicExchange(&heads[cell], e + 1u);
                    entries[e] = vec2<u32>(id, prev);
                }
            }
        }
    }
}

var<workgroup> wg_hit: atomic<u32>;

@compute @workgroup_size(64)
fn mark_leaves(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u {
        atomicStore(&wg_hit, 0u);
    }
    workgroupBarrier();
    let leaf_id = wid.x;
    let L = leaves[leaf_id];
    let d = vec3<f32>(P.dims.xyz);
    let lo_f = floor((L.bmin.xyz - P.origin.xyz) * P.origin.w);
    let hi_f = floor((L.bmax.xyz - P.origin.xyz) * P.origin.w);
    if !(any(hi_f < vec3<f32>(0.0)) || any(lo_f >= d)) {
        let lo = vec3<u32>(max(lo_f, vec3<f32>(0.0)));
        let hi = vec3<u32>(min(hi_f, d - vec3<f32>(1.0)));
        let n = hi - lo + vec3<u32>(1u);
        let total = n.x * n.y * n.z;
        if total > 4096u {
            atomicOr(&wg_hit, 1u);
        } else {
            for (var k = li; k < total; k += 64u) {
                let x = lo.x + k % n.x;
                let y = lo.y + (k / n.x) % n.y;
                let z = lo.z + k / (n.x * n.y);
                if atomicLoad(&heads[x + P.dims.x * (y + P.dims.y * z)]) != 0u {
                    atomicOr(&wg_hit, 1u);
                }
            }
        }
    }
    workgroupBarrier();
    if li == 0u && atomicLoad(&wg_hit) != 0u {
        let slot = atomicAdd(&args[0], 1u);
        work[slot] = leaf_id;
    }
}

fn oct_decode(w: u32) -> vec3<f32> {
    let e = unpack2x16snorm(w);
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if n.z < 0.0 {
        let t = (vec2<f32>(1.0) - abs(n.yx)) * vec2<f32>(sgn(n.x), sgn(n.y));
        n = vec3<f32>(t.x, t.y, n.z);
    }
    return normalize(n);
}

@compute @workgroup_size(64)
fn eval(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let L = leaves[work[wid.x]];
    let k = wid.y * 64u + li;
    if k >= L.a.y {
        return;
    }
    let v = L.a.x + k;
    let o = v * 4u;
    let p = vec3<f32>(bitcast<f32>(mesh[o]), bitcast<f32>(mesh[o + 1u]), bitcast<f32>(mesh[o + 2u]));
    let n = oct_decode(mesh[o + 3u]);
    let q = eval_vertex(p, n) - p;
    delta[v * 3u] = q.x;
    delta[v * 3u + 1u] = q.y;
    delta[v * 3u + 2u] = q.z;
}

// Render-time CSG preview (SPIKE 2). Everything here is per pixel *sample*:
// the base mesh only contributes its first front surface (F) and the next
// back surface behind it (B) per sample, rasterised by two depth+normal
// prepasses. The boolean itself is exact interval arithmetic along the view
// ray against analytic primitives found through a uniform world-space grid.

struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    clay: vec4<f32>,
    overlay: vec4<f32>,
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    cursor: vec4<f32>,
    cursor_color: vec4<f32>,
};

struct Csg {
    inv_view_proj: mat4x4<f32>,
    // x,y = target size in pixels, z = mode, w = heat-map flag
    size: vec4<f32>,
    // xyz = grid min corner, w = cell size
    grid_min: vec4<f32>,
    // xyz = cells per axis, w = instance count
    grid_dim: vec4<u32>,
    // rgb = cutter/primitive tint
    tint: vec4<f32>,
};

struct Inst {
    // xyz = centre, w = kind (0 sphere, 1 box)
    c: vec4<f32>,
    // sphere: x = radius; box: xyz = half extents
    h: vec4<f32>,
    // rotation quaternion (box), local = conj(q) * world
    q: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> g: Csg;

// ---------------------------------------------------------------- prepasses

struct PreOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_pre(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>) -> PreOut {
    var o: PreOut;
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.normal = normal;
    return o;
}

fn is_front(i: PreOut) -> bool {
    return dot(i.normal, u.eye.xyz - i.world) > 0.0;
}

// First surface facing the viewer: depth + smooth normal.
@fragment
fn fs_front(i: PreOut) -> @location(0) vec4<f32> {
    if (!is_front(i)) {
        discard;
    }
    return vec4<f32>(normalize(i.normal), 1.0);
}

// Nearest surface facing away from the viewer: the exit of the solid.
@fragment
fn fs_back(i: PreOut) -> @location(0) vec4<f32> {
    if (is_front(i)) {
        discard;
    }
    return vec4<f32>(-normalize(i.normal), 1.0);
}

// ------------------------------------------------------------------ CSG pass

@group(0) @binding(2) var<storage, read> insts: array<Inst>;
@group(0) @binding(3) var<storage, read> cells: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> cell_idx: array<u32>;
@group(0) @binding(5) var depth_f: texture_depth_multisampled_2d;
@group(0) @binding(6) var depth_b: texture_depth_multisampled_2d;
@group(0) @binding(7) var nrm_f: texture_multisampled_2d<f32>;
@group(0) @binding(8) var nrm_b: texture_multisampled_2d<f32>;

const INF: f32 = 1e30;
const MODE_SUBTRACT: u32 = 0u;
const MODE_ADD: u32 = 1u;
const MODE_INTERSECT: u32 = 2u;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn qconj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

/// Entry/exit distances of the ray against instance `i`; t0 > t1 means a miss.
fn isect(i: u32, o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let ins = insts[i];
    if (ins.c.w < 0.5) {
        let oc = o - ins.c.xyz;
        let b = dot(oc, d);
        let c = dot(oc, oc) - ins.h.x * ins.h.x;
        let disc = b * b - c;
        if (disc < 0.0) {
            return vec2<f32>(1.0, -1.0);
        }
        let s = sqrt(disc);
        return vec2<f32>(-b - s, -b + s);
    }
    let qc = qconj(ins.q);
    let lo = qrot(qc, o - ins.c.xyz);
    let ld = qrot(qc, d);
    let inv = 1.0 / select(ld, vec3<f32>(1e-20), abs(ld) < vec3<f32>(1e-20));
    let a = (-ins.h.xyz - lo) * inv;
    let b = (ins.h.xyz - lo) * inv;
    let tmin = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let tmax = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return vec2<f32>(tmin, tmax);
}

/// Outward normal of instance `i` at world point `p`.
fn inst_normal(i: u32, p: vec3<f32>) -> vec3<f32> {
    let ins = insts[i];
    if (ins.c.w < 0.5) {
        return normalize(p - ins.c.xyz);
    }
    let lp = qrot(qconj(ins.q), p - ins.c.xyz) / ins.h.xyz;
    let a = abs(lp);
    var n = vec3<f32>(0.0);
    if (a.x >= a.y && a.x >= a.z) {
        n.x = sign(lp.x);
    } else if (a.y >= a.z) {
        n.y = sign(lp.y);
    } else {
        n.z = sign(lp.z);
    }
    return qrot(ins.q, n);
}

/// Parametric overlap of the ray with the grid box.
fn grid_range(o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let lo = g.grid_min.xyz;
    let hi = lo + vec3<f32>(g.grid_dim.xyz) * g.grid_min.w;
    let inv = 1.0 / select(d, vec3<f32>(1e-20), abs(d) < vec3<f32>(1e-20));
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let tmin = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let tmax = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return vec2<f32>(tmin, tmax);
}

/// Amanatides-Woo cell walker along a ray, starting at parameter `t_start`.
struct Walker {
    cell: vec3<i32>,
    step: vec3<i32>,
    t_next: vec3<f32>,
    t_delta: vec3<f32>,
    t_out: f32,
    done: bool,
};

fn walker_init(o: vec3<f32>, d: vec3<f32>, t_start: f32) -> Walker {
    var w: Walker;
    let cs = g.grid_min.w;
    let p = (o + d * t_start - g.grid_min.xyz) / cs;
    let dim = vec3<i32>(g.grid_dim.xyz);
    w.cell = clamp(vec3<i32>(floor(p)), vec3<i32>(0), dim - vec3<i32>(1));
    w.step = vec3<i32>(sign(d));
    let inv = 1.0 / select(d, vec3<f32>(1e-20), abs(d) < vec3<f32>(1e-20));
    let next_edge = vec3<f32>(w.cell + max(w.step, vec3<i32>(0)));
    w.t_next = (g.grid_min.xyz + next_edge * cs - o) * inv;
    w.t_delta = abs(vec3<f32>(cs) * inv);
    w.t_out = min(min(w.t_next.x, w.t_next.y), w.t_next.z);
    w.done = false;
    return w;
}

fn walker_advance(w: ptr<function, Walker>) {
    let t = (*w).t_next;
    if (t.x <= t.y && t.x <= t.z) {
        (*w).cell.x += (*w).step.x;
        (*w).t_next.x += (*w).t_delta.x;
    } else if (t.y <= t.z) {
        (*w).cell.y += (*w).step.y;
        (*w).t_next.y += (*w).t_delta.y;
    } else {
        (*w).cell.z += (*w).step.z;
        (*w).t_next.z += (*w).t_delta.z;
    }
    let dim = vec3<i32>(g.grid_dim.xyz);
    if (any((*w).cell < vec3<i32>(0)) || any((*w).cell >= dim)) {
        (*w).done = true;
    }
    (*w).t_out = min(min((*w).t_next.x, (*w).t_next.y), (*w).t_next.z);
}

fn cell_range(c: vec3<i32>) -> vec2<u32> {
    let dim = g.grid_dim.xyz;
    return cells[(u32(c.z) * dim.y + u32(c.y)) * dim.x + u32(c.x)];
}

// What the ray finally sees.
struct Hit {
    kind: u32,        // 0 none, 1 mesh surface, 2 primitive wall
    t: f32,
    normal: vec3<f32>,
    work: u32,        // instance tests (for the heat map)
};

const MAX_CELLS: u32 = 400u;

fn trace(o: vec3<f32>, d: vec3<f32>, tf: f32, tb: f32, nf: vec3<f32>, nb: vec3<f32>, mode: u32) -> Hit {
    var h: Hit;
    h.kind = 0u;
    h.t = INF;
    h.normal = vec3<f32>(0.0, 1.0, 0.0);
    h.work = 0u;
    let has_mesh = tf < INF * 0.5;
    var gr = grid_range(o, d);
    gr.x = max(gr.x, 0.0);
    let grid_hit = gr.x <= gr.y;

    if (mode == MODE_SUBTRACT) {
        if (!has_mesh) { return h; }
        h.kind = 1u; h.t = tf; h.normal = nf;
        if (!grid_hit || tf > gr.y) { return h; }
        // `tc` is the first point at or after the mesh entry that is not
        // inside any cutter. Every cutter covering tc is registered in tc's
        // cell, so a cell is final as soon as tc lies inside it.
        var tc = tf;
        var who = 0xffffffffu;
        var w = walker_init(o, d, max(tf, gr.x));
        for (var s = 0u; s < MAX_CELLS; s++) {
            let r = cell_range(w.cell);
            for (var rep = 0u; rep < 8u; rep++) {
                var moved = false;
                for (var k = 0u; k < r.y; k++) {
                    let idx = cell_idx[r.x + k];
                    let seg = isect(idx, o, d);
                    h.work += 1u;
                    if (seg.x <= tc && tc < seg.y) {
                        tc = seg.y;
                        who = idx;
                        moved = true;
                    }
                }
                if (!moved) { break; }
            }
            if (tc <= w.t_out || tc >= tb) { break; }
            // tc left this cell: skip ahead to the cell that contains it.
            walker_advance(&w);
            for (var j = 0u; j < MAX_CELLS && !w.done && w.t_out < tc; j++) {
                walker_advance(&w);
            }
            if (w.done) { break; }
        }
        if (who != 0xffffffffu) {
            if (tc < tb) {
                h.kind = 2u;
                h.t = tc;
                h.normal = -inst_normal(who, o + d * tc);
            } else {
                // The ray leaves the solid while still inside cutters.
                h.kind = 0u;
                h.t = INF;
            }
        }
        return h;
    }

    // ADD / INTERSECT share the walk: find the best entry along the ray.
    if (mode == MODE_ADD) {
        if (has_mesh) { h.kind = 1u; h.t = tf; h.normal = nf; }
    } else if (!has_mesh) {
        return h;
    }
    if (!grid_hit) { return h; }
    let lo = select(gr.x, max(gr.x, tf), mode == MODE_INTERSECT);
    var best = INF;
    var best_who = 0xffffffffu;
    var w = walker_init(o, d, lo);
    for (var s = 0u; s < MAX_CELLS; s++) {
        let r = cell_range(w.cell);
        for (var k = 0u; k < r.y; k++) {
            let idx = cell_idx[r.x + k];
            let seg = isect(idx, o, d);
            h.work += 1u;
            if (mode == MODE_ADD) {
                if (seg.x <= seg.y && seg.y > 0.0 && seg.x > 0.0 && seg.x < best) {
                    best = seg.x;
                    best_who = idx;
                }
            } else {
                let t = max(seg.x, tf);
                if (seg.x <= seg.y && seg.y > tf && t < tb && t < best) {
                    best = t;
                    best_who = idx;
                }
            }
        }
        if (best <= w.t_out) { break; }
        walker_advance(&w);
        if (w.done) { break; }
    }
    if (mode == MODE_ADD) {
        if (best_who != 0xffffffffu && best < h.t) {
            h.kind = 2u;
            h.t = best;
            h.normal = inst_normal(best_who, o + d * best);
        }
    } else if (best_who != 0xffffffffu) {
        if (best <= tf) {
            h.kind = 1u; h.t = tf; h.normal = nf;
        } else {
            h.kind = 2u; h.t = best; h.normal = inst_normal(best_who, o + d * best);
        }
    }
    return h;
}

struct FsOut {
    @location(0) color: vec4<f32>,
};

struct BgOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) idx: u32) -> BgOut {
    let p = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.clip = vec4<f32>(p, 0.5, 1.0);
    return o;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let w = g.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return w.xyz / w.w;
}

fn clay_shade(base: vec3<f32>, normal: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let v = normalize(u.eye.xyz - world);
    var n = normalize(normal);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let key = normalize(v * 0.8 + u.up.xyz * 0.6 - u.right.xyz * 0.45);
    let fill = normalize(v * 0.4 - u.up.xyz * 0.3 + u.right.xyz * 0.8);
    let diffuse = max(dot(n, key), 0.0);
    let fill_d = max(dot(n, fill), 0.0) * 0.22;
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.3;
    let spec = pow(max(dot(n, normalize(key + v)), 0.0), 40.0) * 0.16;
    return base * (0.10 + 0.85 * diffuse + fill_d) + vec3<f32>(rim + spec);
}

fn heat_ramp(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    return vec3<f32>(smoothstep(0.35, 0.75, t), smoothstep(0.0, 0.35, t) * (1.0 - smoothstep(0.75, 1.0, t) * 0.6), 1.0 - smoothstep(0.2, 0.6, t));
}

@fragment
fn fs_csg(@builtin(position) pos: vec4<f32>, @builtin(sample_index) si: u32) -> FsOut {
    var out: FsOut;
    let px = vec2<i32>(floor(pos.xy));
    let ndc = vec2<f32>(pos.x / g.size.x * 2.0 - 1.0, 1.0 - pos.y / g.size.y * 2.0);
    let o = u.eye.xyz;
    let d = normalize(unproject(ndc, 0.5) - o);

    let zf = textureLoad(depth_f, px, si);
    let zb = textureLoad(depth_b, px, si);
    var tf = INF;
    var tb = INF;
    var nf = vec3<f32>(0.0, 1.0, 0.0);
    var nb = nf;
    if (zf < 1.0) {
        tf = distance(unproject(ndc, zf), o);
        nf = textureLoad(nrm_f, px, si).xyz;
        if (zb < 1.0) {
            tb = distance(unproject(ndc, zb), o);
            nb = textureLoad(nrm_b, px, si).xyz;
        }
    }
    // A front surface without a matching exit (open mesh): treat as thick.
    if (tf < INF * 0.5 && tb >= INF * 0.5) {
        tb = tf + 1e6;
    }

    let h = trace(o, d, tf, tb, nf, nb, u32(g.size.z));
    if (g.size.w > 0.5) {
        // Heat map: instance tests per sample, 0..64.
        let hv = f32(h.work) / 64.0;
        out.color = vec4<f32>(heat_ramp(hv), 1.0);
        if (h.kind == 0u && h.work == 0u) {
            discard;
        }
        return out;
    }
    if (h.kind == 0u) {
        discard;
    }
    let world = o + d * h.t;
    var base = u.clay.rgb;
    if (h.kind == 2u) {
        base = g.tint.rgb;
    }
    let c = clay_shade(base, h.normal, world);
    out.color = vec4<f32>(pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
    return out;
}

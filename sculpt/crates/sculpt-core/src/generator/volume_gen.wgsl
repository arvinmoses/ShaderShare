// A generator layer joined to a field, as `volume_gen.rs` has it: keep the two in step. Needs `surface_noise.wgsl`
// (for the roughness) and, from the including shader, `vg_inst` (the instances, `GpuInstance`) and `vg_cand` (indices
// into them).

struct GpuInstance {
    // The inverse rotation's columns, with the translation in w.
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    // 1 / scale, and the smallest scale.
    s: vec4<f32>,
    // The primitive, and whether it is a sphere scaled alike on every axis.
    k: vec4<u32>,
};

// A generator's parameters: (blend, roughness height, roughness bumps per unit, subtracts) and (roughness kind,
// octaves, seed, has roughness).
struct GpuGen {
    f: vec4<f32>,
    u: vec4<u32>,
};

fn vg_max2(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}

fn vg_extrude(d2: f32, z: f32) -> f32 {
    let w = vec2<f32>(d2, abs(z) - 1.0);
    return min(vg_max2(w), 0.0) + length(max(w, vec2<f32>(0.0)));
}

fn vg_capped_cone(p: vec3<f32>) -> f32 {
    // h = 1, r1 = 1, r2 = 0.
    let q = vec2<f32>(length(p.xz), p.y);
    let k1 = vec2<f32>(0.0, 1.0);
    let k2 = vec2<f32>(-1.0, 2.0);
    let ca = vec2<f32>(q.x - min(q.x, select(0.0, 1.0, q.y < 0.0)), abs(q.y) - 1.0);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

fn vg_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let a = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let b = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let c = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = min(min(a, b), c);
    return -sqrt(d.x) * sign(d.y);
}

// The primitive's distance in its own frame (`Primitive` order: box, sphere, cylinder, cone, wedge, chamfer, torus).
fn vg_prim(k: u32, p: vec3<f32>) -> f32 {
    switch k {
        case 0u: {
            let q = abs(p) - vec3<f32>(1.0);
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        }
        case 1u: {
            return length(p) - 1.0;
        }
        case 2u: {
            let d = vec2<f32>(length(p.xz) - 1.0, abs(p.y) - 1.0);
            return min(vg_max2(d), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case 3u: {
            return vg_capped_cone(p);
        }
        case 4u: {
            return vg_extrude(vg_triangle(p.xy, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0)), p.z);
        }
        case 5u: {
            let s = 0.70710677;
            let u = (p.x + p.y) * s;
            let v = (p.x - p.y) * s;
            let q = vec2<f32>(abs(u) - s, abs(v) - s);
            return vg_extrude(length(max(q, vec2<f32>(0.0))) + min(vg_max2(q), 0.0), p.z);
        }
        default: {
            let q = vec2<f32>(length(p.xz) - 1.0, p.y);
            return length(q) - 0.3;
        }
    }
}

fn vg_inst_sdf(i: u32, p: vec3<f32>) -> f32 {
    let a = vg_inst[i];
    let t = vec3<f32>(a.m0.w, a.m1.w, a.m2.w);
    if (a.k.y != 0u) {
        return length(p - t) - a.s.w;
    }
    let r = mat3x3<f32>(a.m0.xyz, a.m1.xyz, a.m2.xyz);
    return vg_prim(a.k.x, (r * (p - t)) * a.s.xyz) * a.s.w;
}

fn vg_smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) {
        return min(a, b);
    }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn vg_bump(kind: u32, n: f32) -> f32 {
    var b = (n - 0.5) * 3.0;
    if (kind == 1u) {
        b = (n - 0.4) * 2.5;
    } else if (kind == 2u) {
        b = (0.4 - n) * 2.5;
    }
    return clamp(b, -1.0, 1.0);
}

// The field `c` at `p` with generator `g` joined to it, from its candidates `vg_cand[first .. first + count]`.
fn vg_apply(g: GpuGen, c: f32, p: vec3<f32>, first: u32, count: u32) -> f32 {
    if (count == 0u) {
        return c;
    }
    let k = g.f.x;
    var d = vg_inst_sdf(vg_cand[first], p);
    for (var j = 1u; j < count; j++) {
        d = vg_smin(d, vg_inst_sdf(vg_cand[first + j], p), k);
    }
    let sub = g.f.w > 0.5;
    if (g.u.w != 0u) {
        let amp = g.f.y;
        let reach = select(c, -c, sub) + k + amp;
        if (d < reach) {
            d = d + amp * vg_bump(g.u.x, sn_rough(g.u.x, p * g.f.z, g.u.y, g.u.z));
        }
    }
    if (sub) {
        return -vg_smin(-c, d, k);
    }
    return vg_smin(c, d, k);
}

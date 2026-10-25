struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    clay: vec4<f32>,
    // rgb tint, a = strength
    overlay: vec4<f32>,
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    // xyz = center, w = radius (0 hides the cursor)
    cursor: vec4<f32>,
    cursor_color: vec4<f32>,
    // xyz = corner, xyz = size of the box the 16-bit vertex positions are measured in
    box_min: vec4<f32>,
    box_size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) overlay: f32,
    // The surface point before surface noise moved it (where the noise is read).
    @location(3) base: vec3<f32>,
};

fn decode(q: vec4<f32>) -> vec3<f32> {
    return u.box_min.xyz + q.xyz * u.box_size.xyz;
}

// Flat shading: one hard normal per triangle, taken from the screen derivatives of the position.
@vertex
fn vs_flat(@location(0) q: vec4<f32>) -> MeshOut {
    var o: MeshOut;
    let pos = decode(q);
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.normal = vec3<f32>(0.0, 1.0, 0.0);
    o.overlay = q.w;
    o.base = pos;
    return o;
}

@vertex
fn vs_smooth(@location(0) q: vec4<f32>, @location(1) normal: vec3<f32>) -> MeshOut {
    var o: MeshOut;
    let pos = decode(q);
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.normal = normal;
    o.overlay = q.w;
    o.base = pos;
    return o;
}

// Camera-relative studio clay (like a Mudbox/ZBrush matcap): soft wrapped
// key light, cool sky / warm ground ambient, a little subsurface warmth in
// the terminator, fresnel rim and a broad soft specular.
@fragment
fn fs_flat(i: MeshOut) -> @location(0) vec4<f32> {
    // Derivatives need uniform control flow, so take them before anything branches.
    let c = cross(dpdx(i.world), dpdy(i.world));
    let l2 = dot(c, c);
    let n = select(vec3<f32>(0.0, 1.0, 0.0), c * inverseSqrt(max(l2, 1e-30)), l2 > 1e-30);
    return shade(i, n);
}

@fragment
fn fs_smooth(i: MeshOut) -> @location(0) vec4<f32> {
    return shade(i, normalize(i.normal));
}

fn shade(i: MeshOut, normal: vec3<f32>) -> vec4<f32> {
    return lit(i, normal, mix(u.clay.rgb, u.overlay.rgb, clamp(i.overlay, 0.0, 1.0) * u.overlay.a));
}

fn lit(i: MeshOut, normal: vec3<f32>, base: vec3<f32>) -> vec4<f32> {
    let v = normalize(u.eye.xyz - i.world);
    var n = normal;
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let key = normalize(v * 0.75 + u.up.xyz * 0.65 - u.right.xyz * 0.5);
    let fill = normalize(v * 0.35 - u.up.xyz * 0.25 + u.right.xyz * 0.85);

    let ndl = dot(n, key);
    let wrap = clamp((ndl + 0.35) / 1.35, 0.0, 1.0);
    let diffuse = wrap * wrap;
    let sss = clamp(1.0 - abs(ndl + 0.05) * 3.0, 0.0, 1.0) * 0.12;
    let fill_d = max(dot(n, fill), 0.0) * 0.18;
    let hemi = mix(vec3<f32>(0.20, 0.17, 0.15), vec3<f32>(0.24, 0.26, 0.30), dot(n, u.up.xyz) * 0.5 + 0.5);
    let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
    let spec = pow(max(dot(n, normalize(key + v)), 0.0), 28.0) * 0.12 + pow(max(dot(n, normalize(key + v)), 0.0), 6.0) * 0.04;

    var c = base * (hemi + diffuse * 0.85 + fill_d) + base * vec3<f32>(1.0, 0.45, 0.3) * sss + vec3<f32>(spec) + vec3<f32>(0.75, 0.8, 0.9) * fres * 0.22;

    // Brush cursor: outer ring with a dark halo for contrast, inner falloff ring.
    if (u.cursor.w > 0.0) {
        let d = distance(i.world, u.cursor.xyz);
        let px = max(fwidth(d), 1e-6);
        let r = u.cursor.w;
        let halo = 1.0 - smoothstep(px * 1.2, px * 3.0, abs(d - r));
        c = mix(c, c * 0.35, halo * 0.6);
        let ring = 1.0 - smoothstep(0.0, px * 1.3, abs(d - r));
        if (d < r) {
            c = mix(c, u.cursor_color.rgb, 0.05);
        }
        let hard = u.cursor_color.a;
        if (hard > 0.02) {
            let inner = 1.0 - smoothstep(0.0, px * 1.1, abs(d - r * hard));
            c = mix(c, u.cursor_color.rgb, inner * 0.45);
        }
        c = mix(c, u.cursor_color.rgb, ring);
    }
    return vec4<f32>(pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_bg(@builtin(vertex_index) idx: u32) -> BgOut {
    let p = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.clip = vec4<f32>(p, 1.0, 1.0);
    o.uv = p * 0.5 + 0.5;
    return o;
}

// Studio backdrop: vertical gradient, soft radial vignette, and a tiny
// dither so the gradient never bands.
@fragment
fn fs_bg(i: BgOut) -> @location(0) vec4<f32> {
    var c = mix(u.bg_bottom.rgb, u.bg_top.rgb, smoothstep(0.0, 1.0, i.uv.y));
    let d = i.uv - vec2<f32>(0.5, 0.55);
    c = c * (1.0 - 0.35 * smoothstep(0.15, 0.85, dot(d, d) * 2.2));
    let g = pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    let noise = fract(sin(dot(i.clip.xy, vec2<f32>(12.9898, 78.233))) * 43758.547) - 0.5;
    return vec4<f32>(g + vec3<f32>(noise / 255.0), 1.0);
}

// ------------------------------------------------------------------ the multi-resolution volume (gpu_levels)
//
// Vertices are pulled from the buffers the GPU extractor writes: an index names (slot, piece, cell), the cell's word
// says where its vertices are, the slot's block coordinates and level place them.

struct LevelsU {
    origin: vec4<f32>,
    // x: level 0's step; y: tint by level when 1; z: 1 while any block fades; w: the time, in seconds.
    misc: vec4<f32>,
    // Surface noise: (amplitude, scale, -, -) and (octaves, kind, seed, on).
    noise_f: vec4<f32>,
    noise_i: vec4<u32>,
    // The noise mask's hash size less one, whether there is a mask, the slot table's size less one, the level cap.
    mask_i: vec4<u32>,
};

// The octaves a lattice of step `vx` holds: none shorter than four of its steps.
fn sn_octaves_for(vx: f32) -> u32 {
    var n = 0u;
    var wave = 1.0 / max(SN_F.y, 1e-6);
    for (var o = 0u; o < SN_I.x; o++) {
        if (wave < 4.0 * vx) {
            break;
        }
        n += 1u;
        wave *= 0.5;
    }
    return n;
}

@group(1) @binding(0) var<uniform> L: LevelsU;
@group(1) @binding(1) var<storage, read> lv_verts: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read> lv_cells: array<u32>;
@group(1) @binding(3) var<storage, read> lv_meta: array<vec4<i32>>;
@group(1) @binding(4) var<storage, read> lv_mask_hash: array<vec4<i32>>;
@group(1) @binding(5) var<storage, read> lv_mask: array<f32>;
// Per slot: its quad pages, then its vertex pages (count at word 26, numbers from 27), 40 words.
@group(1) @binding(6) var<storage, read> lv_slot_pages: array<u32>;
// Per slot, how far its block has faded in: a + b * time, clamped to 0..1.
@group(1) @binding(7) var<storage, read> lv_morph: array<vec2<f32>>;
// The slot table (as levels.wgsl reads it) and every slot's 9x9x9 samples.
@group(1) @binding(8) var<storage, read> lv_hash: array<vec4<i32>>;
@group(1) @binding(9) var<storage, read> lv_samples: array<u32>;
// Per pattern of inside corners: the pieces (top 8 bits) and each edge's piece (2 bits each).
@group(1) @binding(10) var<storage, read> lv_table: array<u32>;

// ------------------------------------------------------------------ hand-overs between levels
//
// A block shown fades in from the surface of the level below it, and one being hidden fades out to it before it goes:
// each vertex is drawn between where it is and where the coarser level's field has its surface nearby, found from the
// vertex along that field's gradient. Every vertex belongs to one cell of one slot and quads share them, so whatever
// it moves by, the surface stays closed. Extraction is untouched.

fn lv_fade(slot: u32) -> f32 {
    let m = lv_morph[slot];
    return clamp(m.x + m.y * L.misc.w, 0.0, 1.0);
}

fn lv_slot_of(l: u32, c: vec3<i32>) -> i32 {
    if (l > L.mask_i.w) {
        return -1;
    }
    let u = bitcast<vec3<u32>>(c);
    var h = ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u) ^ (l * 2654435761u)) & L.mask_i.z;
    for (var probe = 0u; probe <= L.mask_i.z; probe++) {
        let e = lv_hash[h];
        if (e.w == -1) {
            return -1;
        }
        if (e.w >= 0 && e.x == c.x && e.y == c.y && e.z == c.z && (u32(e.w) >> 24u) == l) {
            return e.w & 0xFFFFFF;
        }
        h = (h + 1u) & L.mask_i.z;
    }
    return -1;
}

fn lv_sample(s: u32, band: f32, p: vec3<i32>) -> f32 {
    let i = u32(p.x + 9 * p.y + 81 * p.z);
    let v = unpack2x16snorm(lv_samples[s * 365u + i / 2u]);
    return select(v.x, v.y, (i & 1u) == 1u) * band;
}

struct Near {
    pos: vec3<f32>,
    nrm: vec3<f32>,
};

// The surface of slot `s`'s field (block `bc` of level `l`) near `p`: two steps along its gradient, each at most a
// lattice step, and its normal there. Where the field is flat (outside its band) `p` stays.
fn lv_nearest(s: u32, l: u32, bc: vec3<i32>, p: vec3<f32>, n: vec3<f32>) -> Near {
    let vx = L.misc.x / f32(1u << l);
    let band = select(2.5 * vx, 16.0 * L.misc.x, l == 0u);
    var o: Near;
    o.pos = p;
    o.nrm = n;
    for (var it = 0; it < 2; it++) {
        let g = clamp((o.pos - L.origin.xyz) / vx - vec3<f32>(bc * 8), vec3<f32>(0.0), vec3<f32>(8.0));
        let i = vec3<i32>(min(floor(g), vec3<f32>(7.0)));
        let f = g - vec3<f32>(i);
        let c000 = lv_sample(s, band, i);
        let c100 = lv_sample(s, band, i + vec3<i32>(1, 0, 0));
        let c010 = lv_sample(s, band, i + vec3<i32>(0, 1, 0));
        let c110 = lv_sample(s, band, i + vec3<i32>(1, 1, 0));
        let c001 = lv_sample(s, band, i + vec3<i32>(0, 0, 1));
        let c101 = lv_sample(s, band, i + vec3<i32>(1, 0, 1));
        let c011 = lv_sample(s, band, i + vec3<i32>(0, 1, 1));
        let c111 = lv_sample(s, band, i + vec3<i32>(1, 1, 1));
        let x00 = mix(c000, c100, f.x);
        let x10 = mix(c010, c110, f.x);
        let x01 = mix(c001, c101, f.x);
        let x11 = mix(c011, c111, f.x);
        let v = mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
        let dx = mix(mix(c100 - c000, c110 - c010, f.y), mix(c101 - c001, c111 - c011, f.y), f.z);
        let dy = mix(mix(c010 - c000, c110 - c100, f.x), mix(c011 - c001, c111 - c101, f.x), f.z);
        let dz = mix(mix(c001 - c000, c101 - c100, f.x), mix(c011 - c010, c111 - c110, f.x), f.y);
        let gr = vec3<f32>(dx, dy, dz) / vx;
        let g2 = dot(gr, gr);
        if (g2 < 0.01) {
            break;
        }
        o.nrm = gr * inverseSqrt(g2);
        var step = gr * (v / g2);
        let len = length(step);
        if (len > vx) {
            step = step * (vx / len);
        }
        o.pos = o.pos - step;
    }
    return o;
}

// The surface near `x` of slot `s`'s field (block `bc`, level `l`) as drawn while it and the levels below fade: a
// blend, stage by stage, of where the coarser surfaces lie, each weighted by how far its block has faded in.
fn lv_chain(s0: u32, l0: u32, bc0: vec3<i32>, x: Near) -> Near {
    var acc = vec3<f32>(0.0);
    var nacc = vec3<f32>(0.0);
    var w = 1.0;
    var q = x;
    var s = s0;
    var l = l0;
    var bc = bc0;
    for (var k = 0; k < 8; k++) {
        let t = lv_fade(s);
        if (t >= 1.0 || l == 0u) {
            break;
        }
        acc += w * t * q.pos;
        nacc += w * t * q.nrm;
        w *= 1.0 - t;
        l -= 1u;
        bc = bc >> vec3<u32>(1u);
        let p = lv_slot_of(l, bc);
        if (p < 0) {
            break;
        }
        s = u32(p);
        q = lv_nearest(s, l, bc, q.pos, q.nrm);
    }
    acc += w * q.pos;
    nacc += w * q.nrm;
    return Near(acc, normalize(nacc + vec3<f32>(1e-20)));
}

// A cell's eight corner values: corners 0-3 and 4-7 (corner k at (k & 1, k >> 1 & 1, k >> 2)).
struct C8 {
    a: vec4<f32>,
    b: vec4<f32>,
};

fn c8_get(c: C8, k: u32) -> f32 {
    let v = select(c.a, c.b, k >= 4u);
    return v[k & 3u];
}

fn c8_off(k: u32) -> vec3<i32> {
    return vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
}

// Cube edge `e` (0-3 along x, 4-7 along y, 8-11 along z) runs from corner `c8_ea` to corner `c8_eb`.
fn c8_ea(e: u32) -> u32 {
    if (e < 4u) {
        return 2u * e;
    }
    if (e < 8u) {
        let j = e - 4u;
        return (j & 1u) | ((j >> 1u) << 2u);
    }
    return e - 8u;
}

fn c8_eb(e: u32) -> u32 {
    return c8_ea(e) | (1u << (e / 4u));
}

// Where in the cell its one piece's vertex goes (the mean of its edges' crossings), and how many crossings.
fn c8_vertex(c: C8) -> vec4<f32> {
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let ka = c8_ea(e);
        let kb = c8_eb(e);
        let va = c8_get(c, ka);
        let vb = c8_get(c, kb);
        if ((va < 0.0) != (vb < 0.0)) {
            sum += mix(vec3<f32>(c8_off(ka)), vec3<f32>(c8_off(kb)), va / (va - vb));
            n += 1.0;
        }
    }
    return vec4<f32>(sum / max(n, 1.0), n);
}

fn c8_gradient(f: vec3<f32>, c: C8) -> vec3<f32> {
    let dx = mix(mix(c.a.y - c.a.x, c.a.w - c.a.z, f.y), mix(c.b.y - c.b.x, c.b.w - c.b.z, f.y), f.z);
    let dy = mix(mix(c.a.z - c.a.x, c.a.w - c.a.y, f.x), mix(c.b.z - c.b.x, c.b.w - c.b.y, f.x), f.z);
    let dz = mix(mix(c.b.x - c.a.x, c.b.y - c.a.y, f.x), mix(c.b.z - c.a.z, c.b.w - c.a.w, f.x), f.y);
    return vec3<f32>(dx, dy, dz);
}

fn c8_set(c: ptr<function, C8>, k: u32, v: f32) {
    if (k < 4u) {
        (*c).a[k] = v;
    } else {
        (*c).b[k - 4u] = v;
    }
}

struct Blend {
    pos: vec3<f32>,
    nrm: vec3<f32>,
    ok: bool,
    transition: bool,
};

// The vertex of cell `lc` of slot `s` (block `bc`, level `l`; its parent's slot `ps`) at `p` with normal `n`, moved
// as its cell's one piece moves when each corner's sample is blended towards the parent's field by how far the block
// owning it has faded in (the slot's own corners by `t_own`, those on the layer past it by their neighbour's). Not ok
// when the blended field leaves the cell without a crossing.
fn lv_cell_blend(s: u32, ps: u32, l: u32, bc: vec3<i32>, lc: vec3<i32>, p: vec3<f32>, n: vec3<f32>, t_own: f32) -> Blend {
    var o = Blend(p, n, true, false);
    let edge = any(lc == vec3<i32>(7));
    if (t_own >= 1.0 && !edge) {
        return o;
    }
    // Each corner's owner's fade.
    var ts = C8(vec4<f32>(t_own), vec4<f32>(t_own));
    var any_fade = t_own < 1.0;
    if (edge) {
        for (var k = 0u; k < 8u; k++) {
            let at = lc + c8_off(k);
            let past = select(vec3<i32>(0), vec3<i32>(1), at == vec3<i32>(8));
            if (any(past != vec3<i32>(0))) {
                let ns = lv_slot_of(l, bc + past);
                // A neighbour not shown: the layer already reads the coarser level.
                let t = select(1.0, lv_fade(u32(max(ns, 0))), ns >= 0);
                c8_set(&ts, k, t);
                any_fade = any_fade || t < 1.0;
            }
        }
    }
    if (!any_fade) {
        return o;
    }
    let vx = L.misc.x / f32(1u << l);
    let band = select(2.5 * vx, 16.0 * L.misc.x, l == 0u);
    let pband = select(5.0 * vx, 16.0 * L.misc.x, l == 1u);
    // The parent's cell holding this one (a level `l` cell lies within one of level `l - 1`).
    let pbc = bc >> vec3<u32>(1u);
    let g0 = vec3<f32>(bc * 8 + lc) * 0.5 - vec3<f32>(pbc * 8);
    let pi = vec3<i32>(clamp(floor(g0), vec3<f32>(0.0), vec3<f32>(7.0)));
    var pc: C8;
    for (var k = 0u; k < 8u; k++) {
        c8_set(&pc, k, lv_sample(ps, pband, pi + c8_off(k)));
    }
    var own: C8;
    var mixed: C8;
    for (var k = 0u; k < 8u; k++) {
        let v = lv_sample(s, band, lc + c8_off(k));
        let f = vec3<f32>(lc + c8_off(k)) * 0.5 + vec3<f32>(bc * 8 - pbc * 16) * 0.5 - vec3<f32>(pi);
        let x00 = mix(pc.a.x, pc.a.y, f.x);
        let x10 = mix(pc.a.z, pc.a.w, f.x);
        let x01 = mix(pc.b.x, pc.b.y, f.x);
        let x11 = mix(pc.b.z, pc.b.w, f.x);
        let coarse = clamp(mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z), -band, band);
        c8_set(&own, k, v);
        c8_set(&mixed, k, mix(coarse, v, c8_get(ts, k)));
    }
    let a = c8_vertex(own);
    let b = c8_vertex(mixed);
    if (a.w == 0.0 || b.w == 0.0) {
        o.ok = false;
        return o;
    }
    o.pos = p + (b.xyz - a.xyz) * vx;
    let ga = c8_gradient(a.xyz, own);
    let gb = c8_gradient(b.xyz, mixed);
    o.nrm = normalize(n + gb * inverseSqrt(max(dot(gb, gb), 1e-30)) - ga * inverseSqrt(max(dot(ga, ga), 1e-30)));
    return o;
}

// The blocks one level finer around a cell (`c`, level `l`): their slots (-1 for none), the 2x2x2 from `lo`.
struct Around {
    lo: vec3<i32>,
    a: vec4<i32>,
    b: vec4<i32>,
};

fn around_of(l1: u32, c: vec3<i32>) -> Around {
    var r: Around;
    r.lo = (c * 2 - vec3<i32>(2)) >> vec3<u32>(3u);
    r.a = vec4<i32>(lv_slot_of(l1, r.lo), lv_slot_of(l1, r.lo + vec3<i32>(1, 0, 0)), lv_slot_of(l1, r.lo + vec3<i32>(0, 1, 0)), lv_slot_of(l1, r.lo + vec3<i32>(1, 1, 0)));
    r.b = vec4<i32>(lv_slot_of(l1, r.lo + vec3<i32>(0, 0, 1)), lv_slot_of(l1, r.lo + vec3<i32>(1, 0, 1)), lv_slot_of(l1, r.lo + vec3<i32>(0, 1, 1)), lv_slot_of(l1, r.lo + vec3<i32>(1, 1, 1)));
    return r;
}

// The slot of the finer block holding finer point or cell `p`.
fn around_slot(r: Around, p: vec3<i32>) -> i32 {
    let o = clamp((p >> vec3<u32>(3u)) - r.lo, vec3<i32>(0), vec3<i32>(1));
    let k = u32(o.x + 2 * o.y + 4 * o.z);
    let v = select(r.a, r.b, k >= 4u);
    return v[k & 3u];
}

// The finer field at finer point `p` as drawn: the finer block's sample, blended towards `coarse` (what the cell
// interpolates there) by how far that block has faded in; `coarse` where there is none. With `steady`, every block as
// if wholly in.
fn lv_finer_value(r: Around, l1: u32, p: vec3<i32>, coarse: f32, steady: bool) -> f32 {
    let s = around_slot(r, p);
    if (s < 0) {
        return coarse;
    }
    let v = lv_sample(u32(s), 2.5 * L.misc.x / f32(1u << l1), p & vec3<i32>(7));
    return select(mix(coarse, v, lv_fade(u32(s))), v, steady);
}

fn c8_trilinear(f: vec3<f32>, c: C8) -> f32 {
    let x00 = mix(c.a.x, c.a.y, f.x);
    let x10 = mix(c.a.z, c.a.w, f.x);
    let x01 = mix(c.b.x, c.b.y, f.x);
    let x11 = mix(c.b.z, c.b.w, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

fn c8_place(axis: u32, along: i32, u: i32, v: i32) -> vec3<i32> {
    if (axis == 0u) {
        return vec3<i32>(along, u, v);
    }
    if (axis == 1u) {
        return vec3<i32>(v, along, u);
    }
    return vec3<i32>(u, v, along);
}

// The vertex of cell `c` (level `l`, in block `bc` of slot `s`) as the extractor would place it in the field as drawn
// (finer blocks blended by their fades; with `steady`, as they are): a transition cell's when a finer block wholly in
// is next to it (or any finer block, with `steady`), else the piece nearest `near` (in the cell's steps) of a plain
// cell. Position in the cell's steps (0..1), normal, and whether there is one.
fn lv_cell_vertex(s: u32, l: u32, bc: vec3<i32>, c: vec3<i32>, near: vec3<f32>, steady: bool) -> Blend {
    var o = Blend(near, vec3<f32>(0.0, 1.0, 0.0), false, false);
    let vx = L.misc.x / f32(1u << l);
    let band = select(2.5 * vx, 16.0 * L.misc.x, l == 0u);
    let l1 = l + 1u;
    let r = around_of(l1, c);
    // The corners: the finer level's where a finer block holds them.
    var d: C8;
    for (var k = 0u; k < 8u; k++) {
        let i = c + c8_off(k);
        c8_set(&d, k, lv_finer_value(r, l1, i * 2, lv_sample(s, band, i - bc * 8), steady));
    }
    // A transition cell: a finer block next to it (not under it) that is drawn.
    var transition = false;
    for (var q = 0u; q < 27u; q++) {
        let off = vec3<i32>(i32(q % 3u) - 1, i32((q / 3u) % 3u) - 1, i32(q / 9u) - 1);
        if (q != 13u) {
            let fs = around_slot(r, (c + off) * 2);
            if (fs >= 0 && (steady || lv_fade(u32(fs)) >= 1.0)) {
                transition = true;
            }
        }
    }
    var f = vec3<f32>(0.5);
    if (transition) {
        var sum = vec3<f32>(0.0);
        var n = 0.0;
        for (var axis = 0u; axis < 3u; axis++) {
            for (var a = 0; a < 3; a++) {
                for (var b = 0; b < 3; b++) {
                    if (a == 1 && b == 1) {
                        continue;
                    }
                    for (var h = 0; h < 2; h++) {
                        let p0 = c8_place(axis, h, a, b);
                        let p1 = c8_place(axis, h + 1, a, b);
                        let va = lv_finer_value(r, l1, c * 2 + p0, c8_trilinear(vec3<f32>(p0) * 0.5, d), steady);
                        let vb = lv_finer_value(r, l1, c * 2 + p1, c8_trilinear(vec3<f32>(p1) * 0.5, d), steady);
                        if ((va < 0.0) != (vb < 0.0)) {
                            sum += (vec3<f32>(p0) + vec3<f32>(p1 - p0) * (va / (va - vb))) * 0.5;
                            n += 1.0;
                        }
                    }
                }
            }
        }
        if (n > 0.0) {
            f = sum / n;
        } else {
            let m = vec3<f32>(0.5);
            let gr = c8_gradient(m, d);
            let g2 = dot(gr, gr);
            let step = gr * (c8_trilinear(m, d) / max(g2, 1e-20));
            if (!(g2 > (0.05 * vx) * (0.05 * vx) && dot(step, step) <= 1.0)) {
                return o;
            }
            f = m - step;
        }
    } else {
        let a = select(vec4<u32>(0u), vec4<u32>(1u, 2u, 4u, 8u), d.a < vec4<f32>(0.0));
        let b = select(vec4<u32>(0u), vec4<u32>(16u, 32u, 64u, 128u), d.b < vec4<f32>(0.0));
        let config = a.x | a.y | a.z | a.w | b.x | b.y | b.z | b.w;
        if (config == 0u || config == 255u) {
            return o;
        }
        let entry = lv_table[config];
        let pieces = entry >> 24u;
        var best = 1e30;
        for (var k = 0u; k < pieces; k++) {
            var sum = vec3<f32>(0.0);
            var n = 0.0;
            for (var e = 0u; e < 12u; e++) {
                let ka = c8_ea(e);
                let kb = c8_eb(e);
                let va = c8_get(d, ka);
                let vb = c8_get(d, kb);
                if ((va < 0.0) != (vb < 0.0) && ((entry >> (2u * e)) & 3u) == k) {
                    sum += mix(vec3<f32>(c8_off(ka)), vec3<f32>(c8_off(kb)), va / (va - vb));
                    n += 1.0;
                }
            }
            let g = sum / max(n, 1.0);
            let dist = dot(g - near, g - near);
            if (n > 0.0 && dist < best) {
                best = dist;
                f = g;
            }
        }
    }
    f = clamp(f, vec3<f32>(0.0), vec3<f32>(1.0));
    o.pos = f;
    o.nrm = normalize(c8_gradient(f, d) + vec3<f32>(1e-20));
    o.ok = true;
    o.transition = transition;
    return o;
}

// How far the blocks one level finer next to a transition cell (cell `c`, level `l`) have faded in: the most of
// them, as the cell's vertex reads their samples.
fn lv_finer_fade(l: u32, c: vec3<i32>) -> f32 {
    let lo = (c * 2 - vec3<i32>(2)) >> vec3<u32>(3u);
    let hi = (c * 2 + vec3<i32>(3)) >> vec3<u32>(3u);
    var t = 0.0;
    var any = false;
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let s = lv_slot_of(l + 1u, vec3<i32>(x, y, z));
                if (s >= 0) {
                    t = max(t, lv_fade(u32(s)));
                    any = true;
                }
            }
        }
    }
    return select(1.0, t, any);
}

// The surface noise's mask at `p` (whole where none is painted): its block at level 0's lattice found by hash, then
// trilinear in the block's samples.
fn noise_mask(p: vec3<f32>) -> f32 {
    if (L.mask_i.y == 0u) {
        return 1.0;
    }
    let g = (p - L.origin.xyz) / L.misc.x;
    let b = vec3<i32>(floor(g / 8.0));
    var h = ((bitcast<u32>(b.x) * 73856093u) ^ (bitcast<u32>(b.y) * 19349663u) ^ (bitcast<u32>(b.z) * 83492791u)) & L.mask_i.x;
    var at = -1;
    for (var probe = 0u; probe <= L.mask_i.x; probe++) {
        let e = lv_mask_hash[h];
        if (e.w < 0) {
            break;
        }
        if (e.x == b.x && e.y == b.y && e.z == b.z) {
            at = e.w;
            break;
        }
        h = (h + 1u) & L.mask_i.x;
    }
    if (at < 0) {
        return 1.0;
    }
    let q = clamp(g - vec3<f32>(b * 8), vec3<f32>(0.0), vec3<f32>(8.0));
    let i = vec3<u32>(min(floor(q), vec3<f32>(7.0)));
    let f = q - vec3<f32>(i);
    let o = u32(at) * 729u + i.x + 9u * i.y + 81u * i.z;
    let x00 = mix(lv_mask[o], lv_mask[o + 1u], f.x);
    let x10 = mix(lv_mask[o + 9u], lv_mask[o + 10u], f.x);
    let x01 = mix(lv_mask[o + 81u], lv_mask[o + 82u], f.x);
    let x11 = mix(lv_mask[o + 90u], lv_mask[o + 91u], f.x);
    return clamp(mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z), 0.0, 1.0);
}

fn lv_oct_decode(w: u32) -> vec3<f32> {
    let e = unpack2x16snorm(w);
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (vec2<f32>(1.0) - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}

@vertex
fn vs_levels(@builtin(vertex_index) idx: u32) -> MeshOut {
    var o: MeshOut;
    let slot = idx >> 11u;
    let piece = (idx >> 9u) & 3u;
    let info = lv_cells[slot * 512u + (idx & 511u)];
    let count = (info >> 10u) & 7u;
    if (count == 0u) {
        // A corner with no vertex (never expected): outside the view, so the triangle is clipped away.
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    let vi = (info & 1023u) + min(piece, count - 1u);
    let v = lv_verts[lv_slot_pages[slot * 40u + 27u + (vi >> 6u)] * 64u + (vi & 63u)];
    let m = lv_meta[slot];
    let level = u32(max(m.w, 0));
    let vx = L.misc.x / f32(1u << level);
    let local = vec3<f32>(f32(v.x & 1023u), f32((v.x >> 10u) & 1023u), f32((v.x >> 20u) & 1023u)) / 1023.0 * 8.0;
    var base = L.origin.xyz + (vec3<f32>(m.xyz * 8) + local) * vx;
    // The field's gradient points out of the solid.
    var n = lv_oct_decode(v.y);
    SN_F = L.noise_f;
    SN_I = L.noise_i;
    // The surface noise's octaves this block's lattice holds (fractional while fading: those of the levels blended).
    var octaves = f32(sn_octaves_for(vx));
    if (L.misc.z > 0.5) {
        // Fading. A vertex of a block fading goes, as the block fades out, to where the coarser level's cell holding
        // it puts its vertex once the block is gone (so its cells collapse onto the coarser cells), and back as it
        // fades in. A transition cell's vertex, which reads finer blocks fading, and a cell's reading a fading
        // neighbour's samples on the layer past its block, move as their vertex moves with those blocks' samples
        // blended towards the coarser field. Then the coarser levels, should they be fading too.
        let ci = idx & 511u;
        let lc = vec3<i32>(i32(ci & 7u), i32((ci >> 3u) & 7u), i32(ci >> 6u));
        let t_own = lv_fade(slot);
        var q = Near(base, n);
        var s = slot;
        var l = level;
        var bc = m.xyz;
        var ps = -1;
        if (l > 0u) {
            ps = lv_slot_of(l - 1u, bc >> vec3<u32>(1u));
        }
        if (t_own < 1.0 && ps >= 0) {
            let pbc = bc >> vec3<u32>(1u);
            let fine = vec3<f32>(bc * 8) + local;
            let c = (bc * 8 + lc) >> vec3<u32>(1u);
            let r = lv_cell_vertex(u32(ps), l - 1u, pbc, c, fine * 0.5 - vec3<f32>(c), false);
            var to = Near(L.origin.xyz + (vec3<f32>(c) + r.pos) * (vx * 2.0), r.nrm);
            if (!r.ok) {
                to = lv_nearest(u32(ps), l - 1u, pbc, q.pos, q.nrm);
            }
            q = Near(mix(to.pos, q.pos, t_own), normalize(mix(to.nrm, q.nrm, t_own)));
            octaves = mix(f32(sn_octaves_for(vx * 2.0)), octaves, t_own);
            s = u32(ps);
            l -= 1u;
            bc = pbc;
        } else if (((info >> 21u) & 1u) == 1u) {
            let tf = lv_finer_fade(l, bc * 8 + lc);
            if (tf < 1.0) {
                let c = bc * 8 + lc;
                let near = local - vec3<f32>(lc);
                let now = lv_cell_vertex(slot, l, bc, c, near, false);
                let was = lv_cell_vertex(slot, l, bc, c, near, true);
                if (now.ok && was.ok) {
                    // Turning from a transition cell into a plain one happens as the last finer block next to it
                    // starts to fade: the change is let in by how far they have faded out.
                    let k = select(1.0 - tf, 1.0, now.transition);
                    q = Near(q.pos + (now.pos - was.pos) * vx * k, normalize(q.nrm + (now.nrm - was.nrm) * k));
                } else {
                    let own = lv_nearest(s, l, bc, q.pos, q.nrm);
                    q = Near(mix(own.pos, q.pos, tf), normalize(mix(own.nrm, q.nrm, tf)));
                }
            }
        } else if (ps >= 0 && count == 1u) {
            let b = lv_cell_blend(slot, u32(ps), l, bc, lc, q.pos, q.nrm, t_own);
            if (b.ok) {
                q = Near(b.pos, b.nrm);
            }
        }
        // The coarser levels: moved as their surface near the vertex moves while they fade.
        if (lv_fade(s) < 1.0 && l > 0u) {
            let x = lv_nearest(s, l, bc, q.pos, q.nrm);
            let y = lv_chain(s, l, bc, x);
            q = Near(q.pos + (y.pos - x.pos), normalize(q.nrm + (y.nrm - x.nrm)));
        }
        base = q.pos;
        n = q.nrm;
    }
    var pos = base;
    if (SN_I.w != 0u) {
        // Surface noise moves the vertex by the octaves this block's lattice holds; the rest only shade.
        let k0 = u32(floor(octaves));
        let fk = octaves - f32(k0);
        var h = 0.0;
        if (k0 > 0u) {
            h = sn_height(base, k0);
        }
        if (fk > 0.001) {
            h = mix(h, sn_height(base, k0 + 1u), fk);
        }
        if (k0 > 0u || fk > 0.001) {
            pos = base + n * h * noise_mask(base);
        }
    }
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.base = base;
    o.normal = n;
    o.overlay = f32(level);
    return o;
}

@fragment
fn fs_levels(i: MeshOut) -> @location(0) vec4<f32> {
    var base = u.clay.rgb;
    if (L.misc.y > 0.5) {
        // Level 0 the clay itself, then cool to warm as the levels get finer.
        let l = u32(clamp(round(i.overlay), 0.0, 7.0));
        if (l == 1u) {
            base = vec3<f32>(0.45, 0.62, 0.85);
        } else if (l == 2u) {
            base = vec3<f32>(0.42, 0.78, 0.62);
        } else if (l == 3u) {
            base = vec3<f32>(0.88, 0.80, 0.42);
        } else if (l == 4u) {
            base = vec3<f32>(0.92, 0.58, 0.38);
        } else if (l == 5u) {
            base = vec3<f32>(0.86, 0.42, 0.55);
        } else if (l >= 6u) {
            base = vec3<f32>(0.70, 0.45, 0.85);
        }
    }
    var n = normalize(i.normal);
    SN_F = L.noise_f;
    SN_I = L.noise_i;
    if (SN_I.w != 0u) {
        // Every octave shades: the normal tilted by the noise's slope along the surface (forward differences over an
        // eighth of its finest wavelength).
        let e = (1.0 / max(SN_F.y, 1e-6)) / f32(1u << min(SN_I.x, 16u)) * 0.125;
        let h0 = sn_height(i.base, SN_I.x);
        let g = vec3<f32>(sn_height(i.base + vec3<f32>(e, 0.0, 0.0), SN_I.x) - h0, sn_height(i.base + vec3<f32>(0.0, e, 0.0), SN_I.x) - h0, sn_height(i.base + vec3<f32>(0.0, 0.0, e), SN_I.x) - h0) / e * noise_mask(i.base);
        n = normalize(n - (g - n * dot(n, g)));
    }
    return lit(i, n, base);
}

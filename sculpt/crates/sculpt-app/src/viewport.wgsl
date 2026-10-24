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
    // x: level 0's step; y: tint by level when 1.
    misc: vec4<f32>,
    // Surface noise: (amplitude, scale, -, -) and (octaves, kind, seed, on).
    noise_f: vec4<f32>,
    noise_i: vec4<u32>,
    // The noise mask's hash size less one, and whether there is a mask.
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
    let v = lv_verts[slot * 768u + (info & 1023u) + min(piece, count - 1u)];
    let m = lv_meta[slot];
    let level = u32(max(m.w, 0));
    let vx = L.misc.x / f32(1u << level);
    let local = vec3<f32>(f32(v.x & 1023u), f32((v.x >> 10u) & 1023u), f32((v.x >> 20u) & 1023u)) / 1023.0 * 8.0;
    let base = L.origin.xyz + (vec3<f32>(m.xyz * 8) + local) * vx;
    // The field's gradient points out of the solid.
    let n = lv_oct_decode(v.y);
    var pos = base;
    SN_F = L.noise_f;
    SN_I = L.noise_i;
    if (SN_I.w != 0u) {
        // Surface noise moves the vertex by the octaves this block's lattice holds; the rest only shade.
        let k = sn_octaves_for(vx);
        if (k > 0u) {
            pos = base + n * sn_height(base, k) * noise_mask(base);
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

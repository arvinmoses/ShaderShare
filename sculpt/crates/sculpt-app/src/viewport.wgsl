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
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) overlay: f32,
};

@vertex
fn vs_mesh(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) overlay: f32) -> MeshOut {
    var o: MeshOut;
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.normal = normal;
    o.overlay = overlay;
    return o;
}

// Camera-relative studio clay (like a Mudbox/ZBrush matcap): soft wrapped
// key light, cool sky / warm ground ambient, a little subsurface warmth in
// the terminator, fresnel rim and a broad soft specular.
@fragment
fn fs_mesh(i: MeshOut) -> @location(0) vec4<f32> {
    let v = normalize(u.eye.xyz - i.world);
    var n = normalize(i.normal);
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

    let base = mix(u.clay.rgb, u.overlay.rgb, clamp(i.overlay, 0.0, 1.0) * u.overlay.a);
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

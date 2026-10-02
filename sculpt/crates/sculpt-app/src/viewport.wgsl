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

// Camera-relative "studio clay" lighting: the key light is fixed relative
// to the view (like a matcap), so forms read the same from every angle.
@fragment
fn fs_mesh(i: MeshOut) -> @location(0) vec4<f32> {
    let v = normalize(u.eye.xyz - i.world);
    var n = normalize(i.normal);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let key = normalize(v * 0.8 + u.up.xyz * 0.6 - u.right.xyz * 0.45);
    let fill = normalize(v * 0.4 - u.up.xyz * 0.3 + u.right.xyz * 0.8);
    let diffuse = max(dot(n, key), 0.0);
    let fill_d = max(dot(n, fill), 0.0) * 0.22;
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.3;
    let spec = pow(max(dot(n, normalize(key + v)), 0.0), 40.0) * 0.16;

    let base = mix(u.clay.rgb, u.overlay.rgb, clamp(i.overlay, 0.0, 1.0) * u.overlay.a);
    var c = base * (0.10 + 0.85 * diffuse + fill_d) + vec3<f32>(rim + spec);

    if (u.cursor.w > 0.0) {
        let d = distance(i.world, u.cursor.xyz);
        let px = max(fwidth(d), 1e-6);
        let ring = 1.0 - smoothstep(0.0, px * 1.5, abs(d - u.cursor.w));
        if (d < u.cursor.w) {
            c = mix(c, u.cursor_color.rgb, 0.07);
        }
        c = mix(c, u.cursor_color.rgb, ring);
    }
    return vec4<f32>(pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) t: f32,
};

@vertex
fn vs_bg(@builtin(vertex_index) idx: u32) -> BgOut {
    let p = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.clip = vec4<f32>(p, 1.0, 1.0);
    o.t = p.y * 0.5 + 0.5;
    return o;
}

@fragment
fn fs_bg(i: BgOut) -> @location(0) vec4<f32> {
    let c = mix(u.bg_bottom.rgb, u.bg_top.rgb, i.t);
    return vec4<f32>(pow(c, vec3<f32>(1.0 / 2.2)), 1.0);
}

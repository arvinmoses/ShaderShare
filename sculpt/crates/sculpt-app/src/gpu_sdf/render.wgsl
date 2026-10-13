// Headless preview of the overlay: clay shading, one colour per draw.

struct U {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
    light: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
}

@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VOut {
    var o: VOut;
    o.pos = u.view_proj * vec4<f32>(p, 1.0);
    o.n = n;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let n = normalize(i.n);
    let l = normalize(u.light.xyz);
    let diff = max(dot(n, l), 0.0);
    let hemi = 0.5 + 0.5 * n.y;
    let c = u.color.rgb * (0.25 + 0.2 * hemi + 0.7 * diff);
    return vec4<f32>(c, 1.0);
}

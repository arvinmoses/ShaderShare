// Surface noise, as `surface_noise.rs` has it: keep the two in step. Reads `SN_F` (amplitude, scale, -, -) and `SN_I`
// (octaves, kind, seed, on), which the including shader sets at the start of each entry point.

var<private> SN_F: vec4<f32>;
var<private> SN_I: vec4<u32>;

fn sn_hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var h = (seed * 0x27d4eb2du) ^ (bitcast<u32>(x) * 0x8da6b343u);
    h ^= bitcast<u32>(y) * 0xd8163841u;
    h ^= bitcast<u32>(z) * 0xcb1ab31fu;
    h ^= h >> 15u;
    h = h * 0x2c1b3c6du;
    h ^= h >> 12u;
    h = h * 0x297a2d39u;
    return h ^ (h >> 15u);
}

fn sn_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn sn_corner(h0: u32, d: vec3<f32>) -> f32 {
    let h = h0 % 12u;
    var a = d.y;
    var b = d.z;
    if (h / 4u == 0u) {
        a = d.x;
        b = d.y;
    } else if (h / 4u == 1u) {
        a = d.x;
        b = d.z;
    }
    return select(-a, a, (h & 1u) == 0u) + select(-b, b, (h & 2u) == 0u);
}

fn sn_c(i: vec3<i32>, f: vec3<f32>, o: vec3<i32>, seed: u32) -> f32 {
    return sn_corner(sn_hash3(i.x + o.x, i.y + o.y, i.z + o.z, seed), f - vec3<f32>(o));
}

fn sn_gradient(p: vec3<f32>, seed: u32) -> f32 {
    let fl = floor(p);
    let f = p - fl;
    let i = vec3<i32>(fl);
    let u = sn_fade(f.x);
    let v = sn_fade(f.y);
    let w = sn_fade(f.z);
    let x00 = mix(sn_c(i, f, vec3<i32>(0, 0, 0), seed), sn_c(i, f, vec3<i32>(1, 0, 0), seed), u);
    let x10 = mix(sn_c(i, f, vec3<i32>(0, 1, 0), seed), sn_c(i, f, vec3<i32>(1, 1, 0), seed), u);
    let x01 = mix(sn_c(i, f, vec3<i32>(0, 0, 1), seed), sn_c(i, f, vec3<i32>(1, 0, 1), seed), u);
    let x11 = mix(sn_c(i, f, vec3<i32>(0, 1, 1), seed), sn_c(i, f, vec3<i32>(1, 1, 1), seed), u);
    return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

// How far the noise moves the surface at `p` along its normal, from its first `octaves`.
fn sn_height(p: vec3<f32>, octaves: u32) -> f32 {
    let total = max(SN_I.x, 1u);
    let n = max(min(octaves, total), 1u);
    var sum = 0.0;
    var amp = 1.0;
    var freq = SN_F.y;
    for (var o = 0u; o < n; o++) {
        let g = sn_gradient(p * freq + vec3<f32>(f32(o) * 17.13), SN_I.z + o);
        if (SN_I.y == 1u) {
            sum += amp * (1.0 - abs(g)) * (1.0 - abs(g));
        } else {
            sum += amp * g;
        }
        amp *= 0.5;
        freq *= 2.0;
    }
    var full = 0.0;
    var a = 1.0;
    for (var o = 0u; o < total; o++) {
        full += a;
        a *= 0.5;
    }
    let s = sum / full;
    var b = s * 2.2;
    if (SN_I.y == 1u) {
        b = (s - 0.45) * 2.2;
    }
    return SN_F.x * clamp(b, -1.0, 1.0);
}

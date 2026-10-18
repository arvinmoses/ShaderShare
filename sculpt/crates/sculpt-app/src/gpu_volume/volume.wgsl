// SPIKE (measurement only): volume sculpting on the GPU. See mod.rs.
//
// The field lives in `samples`: 512 per stored block (8x8x8, x fastest), at the block's slot. `slots` maps every block
// of the grid to its slot, or -1 (outside) / -2 (inside) when nothing is stored for it. Each pass runs one thread per
// sample (or cell) of each block in `work`.

struct Params {
    origin: vec3<f32>,
    voxel: f32,
    centre: vec3<f32>,
    radius: f32,
    dims: vec3<i32>,
    band: f32,
    blend: f32,
    carve: u32,
    count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> samples: array<f32>;
@group(0) @binding(2) var<storage, read_write> slots: array<i32>;
// Per work item: block coordinates in xyz, slot (or fill sign for `fill`) in w.
@group(0) @binding(3) var<storage, read> work: array<vec4<i32>>;
// Per cell of every slot: the surface point, 3 x 10 bits inside the block (0..9 lattice steps), and a valid bit.
@group(0) @binding(4) var<storage, read_write> verts: array<u32>;
// Quads as four global cell ids (slot * 512 + cell).
@group(0) @binding(5) var<storage, read_write> quads: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;
// For `scatter`: (block, new slot table value) pairs.
@group(0) @binding(7) var<storage, read> changes: array<vec2<i32>>;

fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn block_index(c: vec3<i32>) -> i32 {
    if (any(c < vec3<i32>(0)) || any(c >= P.dims)) {
        return -1;
    }
    return (c.z * P.dims.y + c.y) * P.dims.x + c.x;
}

// The field at sample `s` (global lattice coordinates).
fn value(s: vec3<i32>) -> f32 {
    let b = block_index(s >> vec3<u32>(3u));
    if (b < 0) {
        return P.band;
    }
    let slot = slots[b];
    if (slot == -1) {
        return P.band;
    }
    if (slot == -2) {
        return -P.band;
    }
    let l = s & vec3<i32>(7);
    return samples[u32(slot) * 512u + u32(l.z * 64 + l.y * 8 + l.x)];
}

// Write a fill value into every sample of newly stored blocks.
@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= P.count * 512u) {
        return;
    }
    let w = work[id / 512u];
    samples[u32(w.w) * 512u + id % 512u] = f32(w.x) * P.band;
}

// Update the slot table.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= P.count) {
        return;
    }
    let c = changes[id];
    slots[c.x] = c.y;
}

// Blend the blob into every sample of the touched blocks.
@compute @workgroup_size(64)
fn dab(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= P.count * 512u) {
        return;
    }
    let w = work[id / 512u];
    let i = id % 512u;
    let l = vec3<i32>(i32(i % 8u), i32((i / 8u) % 8u), i32(i / 64u));
    let p = P.origin + vec3<f32>(w.xyz * 8 + l) * P.voxel;
    let blob = length(p - P.centre) - P.radius;
    let at = u32(w.w) * 512u + i;
    let v = samples[at];
    var next: f32;
    if (P.carve == 1u) {
        next = -smin(-v, blob, P.blend);
    } else {
        next = smin(v, blob, P.blend);
    }
    samples[at] = clamp(next, -P.band, P.band);
}

const CORNER = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));
const EDGE_A = array<u32, 12>(0u, 2u, 4u, 6u, 0u, 1u, 4u, 5u, 0u, 1u, 2u, 3u);
const EDGE_B = array<u32, 12>(1u, 3u, 5u, 7u, 2u, 3u, 6u, 7u, 4u, 5u, 6u, 7u);

// One vertex per cell the surface crosses, for every cell of the work blocks that have a slot.
@compute @workgroup_size(64)
fn vertices(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= P.count * 512u) {
        return;
    }
    let w = work[id / 512u];
    if (w.w < 0) {
        return;
    }
    let i = id % 512u;
    let l = vec3<i32>(i32(i % 8u), i32((i / 8u) % 8u), i32(i / 64u));
    let s = w.xyz * 8 + l;
    var d: array<f32, 8>;
    var inside = 0u;
    for (var k = 0u; k < 8u; k++) {
        d[k] = value(s + CORNER[k]);
        if (d[k] < 0.0) {
            inside++;
        }
    }
    let at = u32(w.w) * 512u + i;
    if (inside == 0u || inside == 8u) {
        verts[at] = 0u;
        return;
    }
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let a = d[EDGE_A[e]];
        let b = d[EDGE_B[e]];
        if ((a < 0.0) != (b < 0.0)) {
            sum += mix(vec3<f32>(CORNER[EDGE_A[e]]), vec3<f32>(CORNER[EDGE_B[e]]), a / (a - b));
            n += 1.0;
        }
    }
    // Position inside the block, 0..9 lattice steps, in 10 bits per axis.
    let q = vec3<u32>(clamp((vec3<f32>(l) + sum / n) / 9.0, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0);
    verts[at] = q.x | (q.y << 10u) | (q.z << 20u) | (1u << 31u);
}

// The cell id of cell `c` (global cell coordinates), if it has a vertex.
fn cell_id(c: vec3<i32>) -> u32 {
    let b = block_index(c >> vec3<u32>(3u));
    if (b < 0) {
        return 0xffffffffu;
    }
    let slot = slots[b];
    if (slot < 0) {
        return 0xffffffffu;
    }
    let l = c & vec3<i32>(7);
    let id = u32(slot) * 512u + u32(l.z * 64 + l.y * 8 + l.x);
    if ((verts[id] >> 31u) == 0u) {
        return 0xffffffffu;
    }
    return id;
}

// One quad per lattice edge from a sample of a work block that crosses the surface.
@compute @workgroup_size(64)
fn faces(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= P.count * 512u) {
        return;
    }
    let w = work[id / 512u];
    if (w.w < 0) {
        return;
    }
    let i = id % 512u;
    let l = vec3<i32>(i32(i % 8u), i32((i / 8u) % 8u), i32(i / 64u));
    let s = w.xyz * 8 + l;
    let f0 = value(s);
    for (var axis = 0; axis < 3; axis++) {
        var step = vec3<i32>(0);
        step[axis] = 1;
        let f1 = value(s + step);
        if ((f0 < 0.0) == (f1 < 0.0)) {
            continue;
        }
        let c1 = (axis + 1) % 3;
        let c2 = (axis + 2) % 3;
        var e1 = vec3<i32>(0);
        e1[c1] = 1;
        var e2 = vec3<i32>(0);
        e2[c2] = 1;
        let a = cell_id(s - e1 - e2);
        let b = cell_id(s - e2);
        let c = cell_id(s);
        let d = cell_id(s - e1);
        if (a == 0xffffffffu || b == 0xffffffffu || c == 0xffffffffu || d == 0xffffffffu) {
            continue;
        }
        let at = atomicAdd(&counter[0], 1u);
        if (at < arrayLength(&quads)) {
            if (f0 < 0.0) {
                quads[at] = vec4<u32>(a, b, c, d);
            } else {
                quads[at] = vec4<u32>(a, d, c, b);
            }
        }
    }
}

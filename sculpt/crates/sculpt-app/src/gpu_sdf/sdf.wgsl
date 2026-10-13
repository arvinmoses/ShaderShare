// SPIKE 5: sparse-brick SDF boolean on the GPU.
//
// One module, many compute entry points (see mod.rs for the pass order):
//
//   enum_insert   instances -> bricks they overlap -> hash table (keys only)
//   assign_slots  table entries -> cache slots + list of slots that need a base build
//   base_classify one mesh query at each new brick's centre
//   base_fill     mesh signed distance at every (padded) lattice sample of new bricks
//   begin_frame   zero the per-frame counters
//   reset_heads   empty the per-brick instance lists
//   list_build    instances -> per-brick linked lists + list of active bricks
//   combine       base (cached) + analytic instance SDFs -> combined field
//   verts         surface-nets vertex for every sign-changing cell
//   indices       two triangles per sign-changing lattice edge
//
// A brick is 8^3 voxels; it stores a 10^3 lattice (one sample of padding on every
// side) so that vertices and quads never need data from neighbouring bricks.

const EMPTY: u32 = 0xFFFFFFFFu;
const BR: i32 = 8;
const SD: i32 = 10;              // samples per brick edge (with padding)
const SAMPLES: u32 = 1000u;
const CD: i32 = 9;               // cells per brick edge (with padding): -1..7
const CELLS: u32 = 729u;
const LEAF: u32 = 0x80000000u;

struct Params {
    origin: vec3<f32>,
    vs: f32,
    n_inst: u32,
    table_mask: u32,
    frame: u32,
    op: u32,            // 0 subtract, 1 union, 2 intersect
    smooth_k: f32,      // blend radius between base and the instance union (0 = hard min/max)
    margin: f32,
    band: f32,          // brick enumeration band (margin + one voxel of padding)
    stride: u32,        // brick sampling for large benchmarks (1 = all bricks)
    chunk_start: u32,
    chunk_count: u32,
    slot_cap: u32,
    vert_cap: u32,
    idx_cap: u32,
    tri_off: u32,       // offset of the triangle list inside `mesh_u`
    pn_off: u32,        // offset of the pseudo-normals inside `mesh_f`
    n_dispatch_x: u32,  // workgroups in x of the current dispatch (for 2D folding)
    max_hits: u32,
    win_lo: u32,        // brick window (packed keys), inclusive
    win_hi: u32,
    root: u32,          // root node of the mesh tree
}

// Counter indices.
const C_NEXT_SLOT: u32 = 0u;
const C_BUILD: u32 = 1u;
const C_ACTIVE: u32 = 2u;
const C_PAIRS: u32 = 3u;
const C_VERTS: u32 = 4u;
const C_IDX: u32 = 5u;
const C_OWNED: u32 = 6u;     // vertices in cells owned by the brick (no padding duplicates)
const C_OVERFLOW: u32 = 7u;  // bit 0 slots, 1 pairs, 2 verts, 3 idx, 4 table full, 5 coords out of range
const C_QUADS_OWNED: u32 = 8u;

@group(0) @binding(0)  var<uniform> P: Params;
@group(0) @binding(1)  var<storage, read> insts: array<vec4<f32>>;
@group(0) @binding(2)  var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(3)  var<storage, read_write> ctr: array<atomic<u32>>;
@group(0) @binding(4)  var<storage, read_write> smeta: array<atomic<u32>>;
@group(0) @binding(5)  var<storage, read_write> pairs: array<u32>;
@group(0) @binding(6)  var<storage, read_write> lists: array<u32>;
@group(0) @binding(7)  var<storage, read_write> base: array<f32>;
@group(0) @binding(8)  var<storage, read_write> comb: array<f32>;
@group(0) @binding(9)  var<storage, read_write> cellv: array<u32>;
@group(0) @binding(10) var<storage, read_write> verts: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> idxbuf: array<u32>;
@group(0) @binding(12) var<storage, read> mesh_u: array<u32>;
@group(0) @binding(13) var<storage, read> mesh_f: array<f32>;

// ---------------------------------------------------------------- helpers

fn job_id(wid: vec3<u32>, lid: u32) -> u32 {
    return (wid.y * P.n_dispatch_x + wid.x) * 64u + lid;
}

fn hash32(k: u32) -> u32 {
    var x = k * 0x9E3779B1u;
    x = x ^ (x >> 15u);
    x = x * 0x85EBCA6Bu;
    x = x ^ (x >> 13u);
    return x;
}

fn pack_key(b: vec3<i32>) -> u32 {
    return u32(b.x) | (u32(b.y) << 10u) | (u32(b.z) << 20u);
}
fn unpack_key(k: u32) -> vec3<i32> {
    return vec3<i32>(i32(k & 1023u), i32((k >> 10u) & 1023u), i32((k >> 20u) & 1023u));
}
fn keep_brick(key: u32) -> bool {
    let b = unpack_key(key);
    if (any(b < unpack_key(P.win_lo)) || any(b > unpack_key(P.win_hi))) { return false; }
    return P.stride <= 1u || (hash32(key ^ 0x5bd1e995u) % P.stride) == 0u;
}
fn sidx(x: i32, y: i32, z: i32) -> u32 {
    return u32(((z + 1) * SD + (y + 1)) * SD + (x + 1));
}
fn cidx(x: i32, y: i32, z: i32) -> u32 {
    return u32(((z + 1) * CD + (y + 1)) * CD + (x + 1));
}
fn lattice_pos(brick: vec3<i32>, s: vec3<i32>) -> vec3<f32> {
    let g = brick * BR + s;
    return P.origin + vec3<f32>(g) * P.vs;
}

// Brick range overlapped by instance i (its inflated AABB grown by one voxel).
struct Range { lo: vec3<i32>, hi: vec3<i32>, ok: bool }
fn inst_range(i: u32) -> Range {
    let amin = insts[i * 6u + 4u].xyz;
    let amax = insts[i * 6u + 5u].xyz;
    let bsz = f32(BR) * P.vs;
    let pad = P.band - P.margin;
    let lo = vec3<i32>(floor((amin - vec3<f32>(pad) - P.origin) / bsz));
    let hi = vec3<i32>(floor((amax + vec3<f32>(pad) - P.origin) / bsz));
    var r: Range;
    r.lo = lo;
    r.hi = hi;
    r.ok = all(lo >= vec3<i32>(0)) && all(hi <= vec3<i32>(1023));
    return r;
}

fn table_find(key: u32) -> u32 {
    var h = hash32(key) & P.table_mask;
    for (var n = 0u; n <= P.table_mask; n = n + 1u) {
        let k = atomicLoad(&table[2u * h]);
        if (k == key) { return atomicLoad(&table[2u * h + 1u]); }
        if (k == EMPTY) { return EMPTY; }
        h = (h + 1u) & P.table_mask;
    }
    return EMPTY;
}

// ---------------------------------------------------------------- primitives

fn sgn(x: f32) -> f32 { return select(-1.0, 1.0, x >= 0.0); }

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

fn extrude(d2: f32, z: f32) -> f32 {
    let w = vec2<f32>(d2, abs(z) - 1.0);
    return min(max(w.x, w.y), 0.0) + length(max(w, vec2<f32>(0.0)));
}

fn prim_sdf(prim: u32, p: vec3<f32>) -> f32 {
    switch prim {
        case 0u: { // box
            let q = abs(p) - vec3<f32>(1.0);
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        }
        case 1u: { return length(p) - 1.0; }
        case 2u: { // cylinder
            let d = vec2<f32>(length(p.xz) - 1.0, abs(p.y) - 1.0);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case 3u: { return sd_capped_cone(p); }
        case 4u: { // wedge
            let d2 = sd_triangle(p.xy, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0));
            return extrude(d2, p.z);
        }
        case 5u: { // chamfer cutter
            let s = 0.70710678;
            let u = (p.x + p.y) * s;
            let v = (p.x - p.y) * s;
            let q = vec2<f32>(abs(u) - s, abs(v) - s);
            let d2 = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
            return extrude(d2, p.z);
        }
        default: { // torus
            let q = vec2<f32>(length(p.xz) - 1.0, p.y);
            return length(q) - 0.3;
        }
    }
}

fn inst_sdf(i: u32, p: vec3<f32>) -> f32 {
    let r0 = insts[i * 6u];
    let r1 = insts[i * 6u + 1u];
    let r2 = insts[i * 6u + 2u];
    let r3 = insts[i * 6u + 3u];
    let ball = insts[i * 6u + 5u].w > 0.5;
    if (ball) { return length(p - r3.xyz) - r3.w; }
    let prim = u32(insts[i * 6u + 4u].w);
    let l = vec3<f32>(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
    return prim_sdf(prim, l) * r3.w;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }

// ---------------------------------------------------------------- mesh closest point

fn aabb_d2(n: u32, p: vec3<f32>) -> f32 {
    let o = n * 8u;
    let lo = vec3<f32>(bitcast<f32>(mesh_u[o]), bitcast<f32>(mesh_u[o + 1u]), bitcast<f32>(mesh_u[o + 2u]));
    let hi = vec3<f32>(bitcast<f32>(mesh_u[o + 4u]), bitcast<f32>(mesh_u[o + 5u]), bitcast<f32>(mesh_u[o + 6u]));
    let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
    return dot(d, d);
}

fn vpos(v: u32) -> vec3<f32> {
    return vec3<f32>(mesh_f[3u * v], mesh_f[3u * v + 1u], mesh_f[3u * v + 2u]);
}
fn vpn(v: u32) -> vec3<f32> {
    let o = P.pn_off + 3u * v;
    return vec3<f32>(mesh_f[o], mesh_f[o + 1u], mesh_f[o + 2u]);
}

struct Cp { p: vec3<f32>, w: vec3<f32> }

fn closest_on_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Cp {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return Cp(a, vec3<f32>(1.0, 0.0, 0.0)); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return Cp(b, vec3<f32>(0.0, 1.0, 0.0)); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return Cp(a + ab * v, vec3<f32>(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return Cp(c, vec3<f32>(0.0, 0.0, 1.0)); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return Cp(a + ac * w, vec3<f32>(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Cp(b + (c - b) * w, vec3<f32>(0.0, 1.0 - w, w));
    }
    let inv = 1.0 / (va + vb + vc);
    let v = vb * inv;
    let w = vc * inv;
    return Cp(a + ab * v + ac * w, vec3<f32>(1.0 - v - w, v, w));
}

// Signed distance (negative inside) to the mesh, searching only within `bound`.
// Sign from angle-weighted vertex pseudo-normals interpolated at the closest point.
fn mesh_sd(p: vec3<f32>, bound: f32) -> f32 {
    var best = bound * bound;
    var hit_tri = EMPTY;
    var hit_cp = vec3<f32>(0.0);
    var hit_w = vec3<f32>(0.0);
    var sn: array<u32, 64>;
    var sdist: array<f32, 64>;
    var sp = 1;
    sn[0] = P.root;
    sdist[0] = 0.0;
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let d = sdist[sp];
        let n = sn[sp];
        if (d >= best) { continue; }
        let a = mesh_u[n * 8u + 3u];
        let b = mesh_u[n * 8u + 7u];
        if ((b & LEAF) == 0u) {
            let dl = aabb_d2(a, p);
            let dr = aabb_d2(b, p);
            let near_first = dl <= dr;
            let nn = select(b, a, near_first);
            let nd = select(dr, dl, near_first);
            let far_n = select(a, b, near_first);
            let far_d = select(dl, dr, near_first);
            if (far_d < best && sp < 63) { sn[sp] = far_n; sdist[sp] = far_d; sp = sp + 1; }
            if (nd < best && sp < 63) { sn[sp] = nn; sdist[sp] = nd; sp = sp + 1; }
        } else {
            let cnt = b & ~LEAF;
            for (var t = a; t < a + cnt; t = t + 1u) {
                let o = P.tri_off + 3u * t;
                let c = closest_on_tri(p, vpos(mesh_u[o]), vpos(mesh_u[o + 1u]), vpos(mesh_u[o + 2u]));
                let dd = dot(c.p - p, c.p - p);
                if (dd < best) { best = dd; hit_tri = t; hit_cp = c.p; hit_w = c.w; }
            }
        }
    }
    if (hit_tri == EMPTY) { return bound; }
    let o = P.tri_off + 3u * hit_tri;
    let nrm = vpn(mesh_u[o]) * hit_w.x + vpn(mesh_u[o + 1u]) * hit_w.y + vpn(mesh_u[o + 2u]) * hit_w.z;
    let d = sqrt(best);
    return select(d, -d, dot(p - hit_cp, nrm) < 0.0);
}

// Same traversal as mesh_sd, counting work instead of returning a distance:
// (inner/leaf node visits, triangle tests). Feeds the cost model in the report.
fn mesh_sd_count(p: vec3<f32>, bound: f32) -> vec2<u32> {
    var best = bound * bound;
    var sn: array<u32, 64>;
    var sdist: array<f32, 64>;
    var sp = 1;
    var visits = 0u;
    var tests = 0u;
    sn[0] = P.root;
    sdist[0] = 0.0;
    loop {
        if (sp == 0) { break; }
        sp = sp - 1;
        let d = sdist[sp];
        let n = sn[sp];
        if (d >= best) { continue; }
        visits = visits + 1u;
        let a = mesh_u[n * 8u + 3u];
        let b = mesh_u[n * 8u + 7u];
        if ((b & LEAF) == 0u) {
            let dl = aabb_d2(a, p);
            let dr = aabb_d2(b, p);
            let near_first = dl <= dr;
            let nn = select(b, a, near_first);
            let nd = select(dr, dl, near_first);
            let far_n = select(a, b, near_first);
            let far_d = select(dl, dr, near_first);
            if (far_d < best && sp < 63) { sn[sp] = far_n; sdist[sp] = far_d; sp = sp + 1; }
            if (nd < best && sp < 63) { sn[sp] = nn; sdist[sp] = nd; sp = sp + 1; }
        } else {
            let cnt = b & ~LEAF;
            for (var t = a; t < a + cnt; t = t + 1u) {
                let o = P.tri_off + 3u * t;
                let c = closest_on_tri(p, vpos(mesh_u[o]), vpos(mesh_u[o + 1u]), vpos(mesh_u[o + 2u]));
                tests = tests + 1u;
                let dd = dot(c.p - p, c.p - p);
                if (dd < best) { best = dd; }
            }
        }
    }
    return vec2<u32>(visits, tests);
}

// ---------------------------------------------------------------- passes

@compute @workgroup_size(64)
fn begin_frame(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x == 0u) {
        atomicStore(&ctr[C_ACTIVE], 0u);
        atomicStore(&ctr[C_PAIRS], 0u);
        atomicStore(&ctr[C_VERTS], 0u);
        atomicStore(&ctr[C_IDX], 0u);
        atomicStore(&ctr[C_OWNED], 0u);
        atomicStore(&ctr[C_QUADS_OWNED], 0u);
    }
}

@compute @workgroup_size(64)
fn reset_heads(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let s = job_id(wid, lid);
    if (s < atomicLoad(&ctr[C_NEXT_SLOT]) && s < P.slot_cap) { atomicStore(&smeta[s * 4u + 2u], EMPTY); }
}

@compute @workgroup_size(64)
fn enum_insert(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = job_id(wid, lid);
    if (i >= P.n_inst) { return; }
    let r = inst_range(i);
    if (!r.ok) { atomicOr(&ctr[C_OVERFLOW], 32u); return; }
    for (var z = r.lo.z; z <= r.hi.z; z = z + 1) {
        for (var y = r.lo.y; y <= r.hi.y; y = y + 1) {
            for (var x = r.lo.x; x <= r.hi.x; x = x + 1) {
                let key = pack_key(vec3<i32>(x, y, z));
                if (!keep_brick(key)) { continue; }
                var h = hash32(key) & P.table_mask;
                var done = false;
                for (var n = 0u; n <= P.table_mask && !done; n = n + 1u) {
                    let res = atomicCompareExchangeWeak(&table[2u * h], EMPTY, key);
                    if (res.exchanged || res.old_value == key) {
                        done = true;
                    } else if (res.old_value != EMPTY) {
                        h = (h + 1u) & P.table_mask;
                    }
                }
                if (!done) { atomicOr(&ctr[C_OVERFLOW], 16u); }
            }
        }
    }
}

@compute @workgroup_size(64)
fn assign_slots(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = job_id(wid, lid);
    if (h > P.table_mask) { return; }
    let key = atomicLoad(&table[2u * h]);
    if (key == EMPTY) { return; }
    if (atomicLoad(&table[2u * h + 1u]) != EMPTY) { return; }
    let s = atomicAdd(&ctr[C_NEXT_SLOT], 1u);
    if (s >= P.slot_cap) { atomicOr(&ctr[C_OVERFLOW], 1u); return; }
    atomicStore(&table[2u * h + 1u], s);
    atomicStore(&smeta[s * 4u], key);
    atomicStore(&smeta[s * 4u + 1u], 0u);
    atomicStore(&smeta[s * 4u + 2u], EMPTY);
    let b = atomicAdd(&ctr[C_BUILD], 1u);
    lists[b] = s;
}

fn brick_center(key: u32) -> vec3<f32> {
    return lattice_pos(unpack_key(key), vec3<i32>(0)) + vec3<f32>(3.5 * P.vs);
}

@compute @workgroup_size(64)
fn base_classify(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let j = job_id(wid, lid);
    if (j >= P.chunk_count) { return; }
    let s = lists[P.chunk_start + j];
    let key = atomicLoad(&smeta[s * 4u]);
    let dc = mesh_sd(brick_center(key), 1.0e30);
    atomicStore(&smeta[s * 4u + 3u], bitcast<u32>(dc));
}

@compute @workgroup_size(64)
fn base_fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = job_id(wid, lid);
    if (t >= P.chunk_count * SAMPLES) { return; }
    let j = t / SAMPLES;
    let k = t % SAMPLES;
    let s = lists[P.chunk_start + j];
    let key = atomicLoad(&smeta[s * 4u]);
    let dc = bitcast<f32>(atomicLoad(&smeta[s * 4u + 3u]));
    let hd = 0.5 * 1.7320508 * 9.0 * P.vs;                 // half diagonal of the padded lattice
    var d = dc;
    if (abs(dc) <= hd + P.band) {
        let x = i32(k % 10u) - 1;
        let y = i32((k / 10u) % 10u) - 1;
        let z = i32(k / 100u) - 1;
        let p = lattice_pos(unpack_key(key), vec3<i32>(x, y, z));
        d = mesh_sd(p, abs(dc) + hd + P.vs);
    }
    base[s * SAMPLES + k] = d;
}

@compute @workgroup_size(64)
fn list_build(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = job_id(wid, lid);
    if (i >= P.n_inst) { return; }
    let r = inst_range(i);
    if (!r.ok) { return; }
    for (var z = r.lo.z; z <= r.hi.z; z = z + 1) {
        for (var y = r.lo.y; y <= r.hi.y; y = y + 1) {
            for (var x = r.lo.x; x <= r.hi.x; x = x + 1) {
                let key = pack_key(vec3<i32>(x, y, z));
                if (!keep_brick(key)) { continue; }
                let s = table_find(key);
                if (s == EMPTY) { continue; }
                let old = atomicMax(&smeta[s * 4u + 1u], P.frame);
                if (old != P.frame) {
                    let a = atomicAdd(&ctr[C_ACTIVE], 1u);
                    lists[P.slot_cap + a] = s;
                }
                let n = atomicAdd(&ctr[C_PAIRS], 1u);
                if (2u * n + 1u >= arrayLength(&pairs)) { atomicOr(&ctr[C_OVERFLOW], 2u); continue; }
                pairs[2u * n] = i;
                pairs[2u * n + 1u] = atomicExchange(&smeta[s * 4u + 2u], n);
            }
        }
    }
}

@compute @workgroup_size(64)
fn combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = job_id(wid, lid);
    if (t >= P.chunk_count * SAMPLES) { return; }
    let j = t / SAMPLES;
    let k = t % SAMPLES;
    let s = lists[P.slot_cap + P.chunk_start + j];
    let key = atomicLoad(&smeta[s * 4u]);
    let x = i32(k % 10u) - 1;
    let y = i32((k / 10u) % 10u) - 1;
    let z = i32(k / 100u) - 1;
    let p = lattice_pos(unpack_key(key), vec3<i32>(x, y, z));
    let b = base[s * SAMPLES + k];
    var m = 1.0e30;
    var node = atomicLoad(&smeta[s * 4u + 2u]);
    var guard = 0u;
    loop {
        if (node == EMPTY || guard > P.max_hits) { break; }
        guard = guard + 1u;
        let i = pairs[2u * node];
        let lo = insts[i * 6u + 4u].xyz;
        let hi = insts[i * 6u + 5u].xyz;
        if (all(p >= lo) && all(p <= hi)) { m = min(m, inst_sdf(i, p)); }
        node = pairs[2u * node + 1u];
    }
    // No instance within `margin` of p: the true distance to the union is >= margin, so use that bound.
    // Only intersect needs it (subtract and union cannot change sign there), see the report.
    if (m >= 1.0e29 && P.op == 2u) { m = P.margin; }
    var v = b;
    if (m < 1.0e29) {
        if (P.op == 0u) {
            v = smax(b, -m, P.smooth_k);
        } else if (P.op == 1u) {
            v = smin(b, m, P.smooth_k);
        } else {
            v = smax(b, m, P.smooth_k);
        }
    }
    comb[j * SAMPLES + k] = v;
}

var<private> CORNER: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1));
var<private> EDGE: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
    vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u));

@compute @workgroup_size(64)
fn verts_pass(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = job_id(wid, lid);
    if (t >= P.chunk_count * CELLS) { return; }
    let j = t / CELLS;
    let c = t % CELLS;
    let cx = i32(c % 9u) - 1;
    let cy = i32((c / 9u) % 9u) - 1;
    let cz = i32(c / 81u) - 1;
    var d: array<f32, 8>;
    var mask = 0u;
    for (var q = 0u; q < 8u; q = q + 1u) {
        let o = CORNER[q];
        d[q] = comb[j * SAMPLES + sidx(cx + o.x, cy + o.y, cz + o.z)];
        if (d[q] < 0.0) { mask = mask | (1u << q); }
    }
    let slot_out = j * CELLS + c;
    if (mask == 0u || mask == 255u) { cellv[slot_out] = EMPTY; return; }
    var sum = vec3<f32>(0.0);
    var cnt = 0.0;
    for (var e = 0u; e < 12u; e = e + 1u) {
        let a = d[EDGE[e].x];
        let b = d[EDGE[e].y];
        if ((a < 0.0) != (b < 0.0)) {
            let tt = a / (a - b);
            sum = sum + mix(vec3<f32>(CORNER[EDGE[e].x]), vec3<f32>(CORNER[EDGE[e].y]), tt);
            cnt = cnt + 1.0;
        }
    }
    let s = lists[P.slot_cap + P.chunk_start + j];
    let brick = unpack_key(atomicLoad(&smeta[s * 4u]));
    let g = brick * BR + vec3<i32>(cx, cy, cz);
    let pos = P.origin + (vec3<f32>(g) + sum / cnt) * P.vs;
    // gradient of the trilinear interpolant at the cell centre
    let gx = (d[1] + d[3] + d[5] + d[7]) - (d[0] + d[2] + d[4] + d[6]);
    let gy = (d[2] + d[3] + d[6] + d[7]) - (d[0] + d[1] + d[4] + d[5]);
    let gz = (d[4] + d[5] + d[6] + d[7]) - (d[0] + d[1] + d[2] + d[3]);
    let nrm = normalize(vec3<f32>(gx, gy, gz) + vec3<f32>(1.0e-12));
    let v = atomicAdd(&ctr[C_VERTS], 1u);
    if (v >= P.vert_cap) { atomicOr(&ctr[C_OVERFLOW], 4u); cellv[slot_out] = EMPTY; return; }
    verts[2u * v] = vec4<f32>(pos, 1.0);
    verts[2u * v + 1u] = vec4<f32>(nrm, 0.0);
    cellv[slot_out] = v;
    if (cx >= 0 && cy >= 0 && cz >= 0) { atomicAdd(&ctr[C_OWNED], 1u); }
}

fn emit_quad(c0: u32, c1: u32, c2: u32, c3: u32, flip: bool) {
    if (c0 == EMPTY || c1 == EMPTY || c2 == EMPTY || c3 == EMPTY) { return; }
    let o = atomicAdd(&ctr[C_IDX], 6u);
    if (o + 6u > P.idx_cap) { atomicOr(&ctr[C_OVERFLOW], 8u); return; }
    if (!flip) {
        idxbuf[o] = c0; idxbuf[o + 1u] = c2; idxbuf[o + 2u] = c1;
        idxbuf[o + 3u] = c0; idxbuf[o + 4u] = c3; idxbuf[o + 5u] = c2;
    } else {
        idxbuf[o] = c0; idxbuf[o + 1u] = c1; idxbuf[o + 2u] = c2;
        idxbuf[o + 3u] = c0; idxbuf[o + 4u] = c2; idxbuf[o + 5u] = c3;
    }
}

@compute @workgroup_size(64)
fn indices_pass(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = job_id(wid, lid);
    if (t >= P.chunk_count * 512u) { return; }
    let j = t / 512u;
    let l = t % 512u;
    let x = i32(l % 8u);
    let y = i32((l / 8u) % 8u);
    let z = i32(l / 64u);
    let in0 = comb[j * SAMPLES + sidx(x, y, z)] < 0.0;
    let cb = j * CELLS;
    // x edge: cells (x, y-1..y, z-1..z); y edge: (x-1..x, y, z-1..z); z edge: (x-1..x, y-1..y, z)
    if ((comb[j * SAMPLES + sidx(x + 1, y, z)] < 0.0) != in0) {
        emit_quad(cellv[cb + cidx(x, y - 1, z - 1)], cellv[cb + cidx(x, y, z - 1)], cellv[cb + cidx(x, y, z)], cellv[cb + cidx(x, y - 1, z)], in0);
        atomicAdd(&ctr[C_QUADS_OWNED], 1u);
    }
    if ((comb[j * SAMPLES + sidx(x, y + 1, z)] < 0.0) != in0) {
        emit_quad(cellv[cb + cidx(x - 1, y, z - 1)], cellv[cb + cidx(x - 1, y, z)], cellv[cb + cidx(x, y, z)], cellv[cb + cidx(x, y, z - 1)], in0);
        atomicAdd(&ctr[C_QUADS_OWNED], 1u);
    }
    if ((comb[j * SAMPLES + sidx(x, y, z + 1)] < 0.0) != in0) {
        emit_quad(cellv[cb + cidx(x - 1, y - 1, z)], cellv[cb + cidx(x, y - 1, z)], cellv[cb + cidx(x, y, z)], cellv[cb + cidx(x - 1, y, z)], in0);
        atomicAdd(&ctr[C_QUADS_OWNED], 1u);
    }
}

// Work counters for the cost model: re-runs the base_fill queries of the first bricks of the
// build list with the counting traversal. ctr[9] node visits, ctr[10] triangle tests, ctr[11] queries.
@compute @workgroup_size(64)
fn query_stats(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = job_id(wid, lid);
    if (t >= P.chunk_count * SAMPLES) { return; }
    let j = t / SAMPLES;
    let k = t % SAMPLES;
    let s = lists[P.chunk_start + j];
    let key = atomicLoad(&smeta[s * 4u]);
    let dc = bitcast<f32>(atomicLoad(&smeta[s * 4u + 3u]));
    let hd = 0.5 * 1.7320508 * 9.0 * P.vs;
    if (abs(dc) > hd + P.band) { return; }
    let x = i32(k % 10u) - 1;
    let y = i32((k / 10u) % 10u) - 1;
    let z = i32(k / 100u) - 1;
    let c = mesh_sd_count(lattice_pos(unpack_key(key), vec3<i32>(x, y, z)), abs(dc) + hd + P.vs);
    atomicAdd(&ctr[9], c.x);
    atomicAdd(&ctr[10], c.y);
    atomicAdd(&ctr[11], 1u);
}

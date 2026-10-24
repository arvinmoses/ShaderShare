// The multi-resolution volume's surface, built on the GPU (see mod.rs and sculpt_core::generator::level_volume).
//
// Every stored block (any level) has a slot. Per slot: its 9x9x9 samples (its own 8x8x8 and the first layer past it, at
// its level, 16 bits each), and what extraction writes: a vertex per surface piece of its cells, a word per cell saying
// where its vertices are, and its quads in pages of 64, which the viewport draws straight from this buffer. A quad corner
// is (slot, piece, cell) packed in a 32-bit index, resolved to a vertex when drawn, so a quad stays valid when the block
// it points into is rebuilt.
//
// Passes, each over a list of slots (`work`, after the first `release_count` entries, which only give pages back):
//   release   give the slots' pages back to the free list.
//   vertices  one workgroup per slot: vertices, cell words, the slot's bounds and normal cone.
//   faces     one workgroup per slot: the quads of the smallest crossing edges its cells own, into fresh pages.
//   cull      one thread per slot: the pages of slots in view, as indirect draws.

const VSTRIDE: u32 = 768u;
const PAGE_QUADS: u32 = 64u;
const PAGE_INDICES: u32 = 384u;
const SLOT_WORDS: u32 = 40u;
const MAX_PAGES: u32 = 24u;
// Vertices go in pages of 64 from a pool (a block holds what it has, not the most it could): per slot, after the quad
// pages, how many vertex pages (word 26) and which (27..38).
const VPAGE: u32 = 64u;
const MAX_VPAGES: u32 = 12u;
const VP_N: u32 = 26u;
const VP_AT: u32 = 27u;
const SAMPLE_WORDS: u32 = 365u;

struct Params {
    origin: vec3<f32>,
    voxel0: f32,
    cap: u32,
    levels: u32,
    work_count: u32,
    hash_mask: u32,
    release_count: u32,
    page_capacity: u32,
    slot_count: u32,
    cull: u32,
    eye: vec4<f32>,
    planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> samples: array<u32>;
// Open addressing: (block coordinates, level << 24 | slot), w = -1 empty, -2 removed.
@group(0) @binding(2) var<storage, read> hash: array<vec4<i32>>;
// Per slot: block coordinates and level (-1 when free).
@group(0) @binding(3) var<storage, read> slot_meta: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read> work: array<u32>;
// Per vertex: position inside its block (10 bits per axis over 0..8 steps) and octahedral normal.
@group(0) @binding(5) var<storage, read_write> verts: array<vec2<u32>>;
// Per cell: first vertex (bits 0-9), pieces (10-12), inside corners (13-20), next to a finer cell (21).
@group(0) @binding(6) var<storage, read_write> cells: array<u32>;
@group(0) @binding(7) var<storage, read_write> pages: array<u32>;
// Per slot: pages, quads, then the page numbers; then its vertex pages (count, numbers).
@group(0) @binding(8) var<storage, read_write> slot_pages: array<u32>;
// The free quad pages, then (from `page_capacity` on) the free vertex pages.
@group(0) @binding(9) var<storage, read_write> free_pages: array<u32>;
// 0: free pages on the stack; 1: set when pages ran out; 2: draws written by `cull`; 3: quads dropped for want of pages;
// 4: free vertex pages; 5: set when vertex pages ran out.
@group(0) @binding(10) var<storage, read_write> status: array<atomic<i32>>;
// Per slot: bounds min (xyz, in steps inside the block), bounds max, normal cone (axis, cosine of its half-angle; w < -1
// for none).
@group(0) @binding(11) var<storage, read_write> clusters: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> table: array<u32>;
// DrawIndexedIndirectArgs, five words each.
@group(0) @binding(13) var<storage, read_write> draws: array<u32>;

fn voxel(l: u32) -> f32 {
    return P.voxel0 / f32(1u << l);
}

fn band(l: u32) -> f32 {
    if (l == 0u) {
        return 16.0 * P.voxel0;
    }
    return 2.5 * voxel(l);
}

fn hash_of(c: vec3<i32>, l: u32) -> u32 {
    let u = bitcast<vec3<u32>>(c);
    return (u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u) ^ (l * 2654435761u);
}

// The slot of block `c` of level `l`, or -1 (levels past the cap count as empty).
fn slot_of(l: u32, c: vec3<i32>) -> i32 {
    if (l > P.cap) {
        return -1;
    }
    var h = hash_of(c, l) & P.hash_mask;
    for (var probe = 0u; probe <= P.hash_mask; probe++) {
        let e = hash[h];
        if (e.w == -1) {
            return -1;
        }
        if (e.w >= 0 && e.x == c.x && e.y == c.y && e.z == c.z && (u32(e.w) >> 24u) == l) {
            return e.w & 0xFFFFFF;
        }
        h = (h + 1u) & P.hash_mask;
    }
    return -1;
}

fn idx9(p: vec3<i32>) -> u32 {
    return u32(p.x + 9 * p.y + 81 * p.z);
}

fn local8(p: vec3<i32>) -> u32 {
    let q = p & vec3<i32>(7);
    return u32(q.x + 8 * q.y + 64 * q.z);
}

fn sample9(s: u32, l: u32, i: u32) -> f32 {
    let v = unpack2x16snorm(samples[s * SAMPLE_WORDS + i / 2u]);
    return select(v.x, v.y, (i & 1u) == 1u) * band(l);
}

// No arrays live in a thread's own memory below (constant tables included): indexing them by a variable makes some
// drivers spill them to scratch memory, and on one Vulkan driver that lost the device once several workgroups ran.
// Corners and edges are worked out from their numbers' bits, a cell's eight corner values travel as two vec4s.

fn corner_off(k: u32) -> vec3<i32> {
    return vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
}

// Cube edge `e` (0-3 along x, 4-7 along y, 8-11 along z) runs from corner `edge_a` to corner `edge_b`.
fn edge_a(e: u32) -> u32 {
    if (e < 4u) {
        return 2u * e;
    }
    if (e < 8u) {
        let j = e - 4u;
        return (j & 1u) | ((j >> 1u) << 2u);
    }
    return e - 8u;
}

fn edge_b(e: u32) -> u32 {
    return edge_a(e) | (1u << (e / 4u));
}

// The cube edge starting at corner `k0` along `axis`.
fn edge_index(k0: u32, axis: u32) -> u32 {
    if (axis == 0u) {
        return k0 / 2u;
    }
    if (axis == 1u) {
        return 4u + (k0 & 1u) + ((k0 >> 2u) & 1u) * 2u;
    }
    return 8u + k0;
}

// A vector with `along` on `axis`, `u` on the axis after it and `v` on the one after that.
fn place(axis: u32, along: i32, u: i32, v: i32) -> vec3<i32> {
    if (axis == 0u) {
        return vec3<i32>(along, u, v);
    }
    if (axis == 1u) {
        return vec3<i32>(v, along, u);
    }
    return vec3<i32>(u, v, along);
}

fn comp(p: vec3<i32>, i: u32) -> i32 {
    return select(select(p.z, p.y, i == 1u), p.x, i == 0u);
}

// A cell's corner values: corners 0-3 and 4-7.
struct Corners {
    a: vec4<f32>,
    b: vec4<f32>,
};

fn dget(c: Corners, k: u32) -> f32 {
    let v = select(c.a, c.b, k >= 4u);
    return v[k & 3u];
}

fn config_of(c: Corners) -> u32 {
    let a = select(vec4<u32>(0u), vec4<u32>(1u, 2u, 4u, 8u), c.a < vec4<f32>(0.0));
    let b = select(vec4<u32>(0u), vec4<u32>(16u, 32u, 64u, 128u), c.b < vec4<f32>(0.0));
    return a.x | a.y | a.z | a.w | b.x | b.y | b.z | b.w;
}

fn trilinear(f: vec3<f32>, c: Corners) -> f32 {
    let x00 = mix(c.a.x, c.a.y, f.x);
    let x10 = mix(c.a.z, c.a.w, f.x);
    let x01 = mix(c.b.x, c.b.y, f.x);
    let x11 = mix(c.b.z, c.b.w, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

fn trilinear_gradient(f: vec3<f32>, c: Corners) -> vec3<f32> {
    let dx = mix(mix(c.a.y - c.a.x, c.a.w - c.a.z, f.y), mix(c.b.y - c.b.x, c.b.w - c.b.z, f.y), f.z);
    let dy = mix(mix(c.a.z - c.a.x, c.a.w - c.a.y, f.x), mix(c.b.z - c.b.x, c.b.w - c.b.y, f.x), f.z);
    let dz = mix(mix(c.b.x - c.a.x, c.b.y - c.a.y, f.x), mix(c.b.z - c.a.z, c.b.w - c.a.w, f.x), f.y);
    return vec3<f32>(dx, dy, dz);
}

fn oct_encode(n: vec3<f32>) -> u32 {
    let a = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    var e = a.xy;
    if (a.z < 0.0) {
        e = (vec2<f32>(1.0) - abs(a.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), a.xy >= vec2<f32>(0.0));
    }
    return pack2x16snorm(e);
}

fn oct_decode(w: u32) -> vec3<f32> {
    let e = unpack2x16snorm(w);
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (vec2<f32>(1.0) - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}

// ------------------------------------------------------------------------------ per-workgroup block state

// All of a workgroup's shared state in one array of atomics, at fixed places: one Vulkan driver laid separate workgroup
// variables over each other.
//   the block: slot, level, coordinates;
//   whether a block of the next level covers each group of 4x4x4 cells around it (groups -1..2 per axis);
//   the slots of the blocks around at the same level (-1..1 per axis);
//   prefix sums; the pages taken; bounds (min, max), normal sum (i32 bits), cone (offset by 32768), vertex count.
const W_SLOT: u32 = 0u;
const W_LEVEL: u32 = 1u;
const W_BC: u32 = 2u;
const W_FINE: u32 = 5u;
const W_NB: u32 = 69u;
const W_SCAN: u32 = 96u;
const W_TOT: u32 = 352u;
const W_PAGES: u32 = 368u;
const W_ATOM: u32 = 392u;
var<workgroup> wg: array<atomic<u32>, 404>;

fn wget(i: u32) -> u32 {
    return atomicLoad(&wg[i]);
}

fn wset(i: u32, v: u32) {
    atomicStore(&wg[i], v);
}

fn wg_bc() -> vec3<i32> {
    return vec3<i32>(bitcast<i32>(wget(W_BC)), bitcast<i32>(wget(W_BC + 1u)), bitcast<i32>(wget(W_BC + 2u)));
}

// A level `l` cell or point `q` near the block: is it under a block of the next level?
fn fine_at(q: vec3<i32>) -> bool {
    let g = (q >> vec3<u32>(2u)) - (wg_bc() * 2 - vec3<i32>(1));
    return wget(W_FINE + u32(g.x + 4 * g.y + 16 * g.z)) != 0u;
}

fn covered(l: u32, q: vec3<i32>) -> bool {
    return l < P.cap && fine_at(q);
}

// The value at level `l`'s point `i` (a corner of one of this block's cells): the finest level's that holds a block
// there, else this block's own sample.
fn corner_value(l: u32, slot: u32, bc: vec3<i32>, i: vec3<i32>) -> f32 {
    if (covered(l, i)) {
        var m = l + 1u;
        var j = i * 2;
        var s = slot_of(m, j >> vec3<u32>(3u));
        loop {
            if (m >= P.cap) {
                break;
            }
            let s2 = slot_of(m + 1u, (j * 2) >> vec3<u32>(3u));
            if (s2 < 0) {
                break;
            }
            s = s2;
            m += 1u;
            j = j * 2;
        }
        if (s >= 0) {
            return sample9(u32(s), m, idx9(j & vec3<i32>(7)));
        }
    }
    return sample9(slot, l, idx9(i - bc * 8));
}

fn corners(l: u32, slot: u32, bc: vec3<i32>, c: vec3<i32>) -> Corners {
    var o: Corners;
    o.a = vec4<f32>(corner_value(l, slot, bc, c), corner_value(l, slot, bc, c + vec3<i32>(1, 0, 0)), corner_value(l, slot, bc, c + vec3<i32>(0, 1, 0)), corner_value(l, slot, bc, c + vec3<i32>(1, 1, 0)));
    o.b = vec4<f32>(corner_value(l, slot, bc, c + vec3<i32>(0, 0, 1)), corner_value(l, slot, bc, c + vec3<i32>(1, 0, 1)), corner_value(l, slot, bc, c + vec3<i32>(0, 1, 1)), corner_value(l, slot, bc, c + vec3<i32>(1, 1, 1)));
    return o;
}

// The value at level `l + 1`'s point `p` on the boundary of a cell (`h` its offset in half cells), for a transition
// cell's vertex: the finer level's where it holds a block, else interpolated from the cell's corners `d`.
fn half_value(l: u32, p: vec3<i32>, h: vec3<i32>, d: Corners) -> f32 {
    var m = l + 1u;
    var j = p;
    var s = slot_of(m, p >> vec3<u32>(3u));
    loop {
        if (m >= P.cap) {
            break;
        }
        let s2 = slot_of(m + 1u, (j * 2) >> vec3<u32>(3u));
        if (s2 < 0) {
            break;
        }
        s = s2;
        m += 1u;
        j = j * 2;
    }
    if (s >= 0) {
        return sample9(u32(s), m, idx9(j & vec3<i32>(7)));
    }
    return trilinear(vec3<f32>(h) * 0.5, d);
}

// Load the block's slot, level and neighbourhood into workgroup memory, in two steps with a barrier after each. The
// barriers are the entry points' own: one Vulkan driver loses those placed inside a called function.
fn load_block_1(t: u32, w: u32) {
    if (t == 0u) {
        if (w < P.work_count) {
            let s = work[P.release_count + w];
            let m = slot_meta[s];
            wset(W_SLOT, s);
            wset(W_LEVEL, u32(max(m.w, 0)));
            wset(W_BC, bitcast<u32>(m.x));
            wset(W_BC + 1u, bitcast<u32>(m.y));
            wset(W_BC + 2u, bitcast<u32>(m.z));
        } else {
            wset(W_SLOT, 0xFFFFFFFFu);
            wset(W_LEVEL, 255u);
            wset(W_BC, 0u);
            wset(W_BC + 1u, 0u);
            wset(W_BC + 2u, 0u);
        }
        atomicStore(&wg[W_ATOM + 10], 0u);
        atomicStore(&wg[W_ATOM + 9], 65535u);
    }
    if (t < 3u) {
        atomicStore(&wg[W_ATOM + t], 0xFFFFFFFFu);
        atomicStore(&wg[W_ATOM + 3u + t], 0u);
        atomicStore(&wg[W_ATOM + 6u + t], 0u);
    }
}

fn load_block_2(t: u32) {
    let l = wget(W_LEVEL);
    let bc = wg_bc();
    if (t < 64u) {
        var f = 0u;
        if (l < P.cap) {
            let g = vec3<i32>(i32(t & 3u), i32((t >> 2u) & 3u), i32(t >> 4u));
            f = select(0u, 1u, slot_of(l + 1u, bc * 2 - vec3<i32>(1) + g) >= 0);
        }
        wset(W_FINE + t, f);
    } else if (t < 91u) {
        let q = t - 64u;
        let o = vec3<i32>(i32(q % 3u) - 1, i32((q / 3u) % 3u) - 1, i32(q / 9u) - 1);
        wset(W_NB + q, bitcast<u32>(slot_of(l, bc + o)));
    }
}

// Exclusive prefix sum over the workgroup, in two levels of 16: each thread puts its value in `wg_scan`, then (after a
// barrier) `scan_1` sums within its group of 16, then (after another) `scan_2` adds the groups before it. Returns
// (this thread's offset, the total).
fn scan_1(t: u32, v: u32) -> u32 {
    let g = t / 16u;
    var within = 0u;
    for (var i = g * 16u; i < t; i++) {
        within += wget(W_SCAN + i);
    }
    if ((t & 15u) == 15u) {
        wset(W_TOT + g, within + v);
    }
    return within;
}

fn scan_2(t: u32, within: u32) -> vec2<u32> {
    let g = t / 16u;
    var before = 0u;
    var total = 0u;
    for (var i = 0u; i < 16u; i++) {
        let x = wget(W_TOT + i);
        if (i < g) {
            before += x;
        }
        total += x;
    }
    return vec2<u32>(before + within, total);
}

// ------------------------------------------------------------------------------ release

@compute @workgroup_size(64)
fn release(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= P.release_count + P.work_count) {
        return;
    }
    let s = work[i];
    let nv = slot_pages[s * SLOT_WORDS + VP_N];
    if (nv > 0u) {
        let vbase = atomicAdd(&status[4], i32(nv));
        for (var k = 0u; k < nv; k++) {
            free_pages[P.page_capacity + u32(vbase) + k] = slot_pages[s * SLOT_WORDS + VP_AT + k];
        }
        slot_pages[s * SLOT_WORDS + VP_N] = 0u;
    }
    let n = slot_pages[s * SLOT_WORDS];
    if (n == 0u) {
        return;
    }
    let base = atomicAdd(&status[0], i32(n));
    for (var k = 0u; k < n; k++) {
        free_pages[u32(base) + k] = slot_pages[s * SLOT_WORDS + 2u + k];
    }
    slot_pages[s * SLOT_WORDS] = 0u;
    slot_pages[s * SLOT_WORDS + 1u] = 0u;
}

// Where vertex `i` of the block being extracted goes: in the vertex pages its workgroup took (at W_PAGES).
fn vat(i: u32) -> u32 {
    return wget(W_PAGES + (i >> 6u)) * VPAGE + (i & 63u);
}

// ------------------------------------------------------------------------------ vertices

fn cell_coords(bc: vec3<i32>, ci: u32) -> vec3<i32> {
    return bc * 8 + vec3<i32>(i32(ci & 7u), i32((ci >> 3u) & 7u), i32(ci >> 6u));
}

fn is_transition(l: u32, c: vec3<i32>) -> bool {
    if (l >= P.cap) {
        return false;
    }
    for (var q = 0u; q < 27u; q++) {
        if (q != 13u && fine_at(c + vec3<i32>(i32(q % 3u) - 1, i32((q / 3u) % 3u) - 1, i32(q / 9u) - 1))) {
            return true;
        }
    }
    return false;
}

// A cell's word without its first vertex: pieces, inside corners, transition. Zero for cells with no vertex.
fn cell_head(l: u32, slot: u32, bc: vec3<i32>, ci: u32) -> u32 {
    if (l > P.cap) {
        return 0u;
    }
    let c = cell_coords(bc, ci);
    if (covered(l, c)) {
        return 0u;
    }
    let config = config_of(corners(l, slot, bc, c));
    if (is_transition(l, c)) {
        return (1u << 10u) | (config << 13u) | (1u << 21u);
    }
    if (config == 0u || config == 255u) {
        return 0u;
    }
    return ((table[config] >> 24u) << 10u) | (config << 13u);
}

struct Vert {
    pos: vec3<f32>,
    nrm: vec3<f32>,
};

// Vertex `k` of a cell whose word (without its first vertex) is `head`: where it lies in the block, and its normal.
fn cell_vertex(l: u32, slot: u32, bc: vec3<i32>, ci: u32, head: u32, k: u32) -> Vert {
    let c = cell_coords(bc, ci);
    let lc = c - bc * 8;
    let d = corners(l, slot, bc, c);
    var f = vec3<f32>(0.5);
    if (((head >> 21u) & 1u) == 1u) {
        // Crossings on the cell's faces split once.
        var sum = vec3<f32>(0.0);
        var n = 0.0;
        for (var axis = 0u; axis < 3u; axis++) {
            for (var a = 0; a < 3; a++) {
                for (var b = 0; b < 3; b++) {
                    if (a == 1 && b == 1) {
                        continue;
                    }
                    for (var s = 0; s < 2; s++) {
                        let p0 = place(axis, s, a, b);
                        let p1 = place(axis, s + 1, a, b);
                        let va = half_value(l, c * 2 + p0, p0, d);
                        let vb = half_value(l, c * 2 + p1, p1, d);
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
            // Onto the zero along the gradient, if there is one to speak of and the zero lies within the cell (as the
            // reference does: deep in fresh clay the gradient left is rounding).
            let m = vec3<f32>(0.5);
            let gr = trilinear_gradient(m, d);
            let g2 = dot(gr, gr);
            let step = gr * (trilinear(m, d) / max(g2, 1e-20));
            let vx = voxel(l);
            let ok = g2 > (0.05 * vx) * (0.05 * vx) && dot(step, step) <= 1.0;
            f = select(m, m - step, ok);
        }
    } else {
        let entry = table[(head >> 13u) & 255u];
        var sum = vec3<f32>(0.0);
        var n = 0.0;
        for (var e = 0u; e < 12u; e++) {
            let ka = edge_a(e);
            let kb = edge_b(e);
            let va = dget(d, ka);
            let vb = dget(d, kb);
            if ((va < 0.0) != (vb < 0.0) && ((entry >> (2u * e)) & 3u) == k) {
                sum += mix(vec3<f32>(corner_off(ka)), vec3<f32>(corner_off(kb)), va / (va - vb));
                n += 1.0;
            }
        }
        f = sum / max(n, 1.0);
    }
    f = clamp(f, vec3<f32>(0.0), vec3<f32>(1.0));
    var v: Vert;
    v.pos = vec3<f32>(lc) + f;
    v.nrm = normalize(trilinear_gradient(f, d) + vec3<f32>(1e-20));
    return v;
}

fn write_cell(l: u32, slot: u32, bc: vec3<i32>, ci: u32, head: u32, first: u32) -> vec3<f32> {
    let n = (head >> 10u) & 7u;
    var nsum = vec3<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let v = cell_vertex(l, slot, bc, ci, head, k);
        let q = vec3<u32>(round(clamp(v.pos / 8.0, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0));
        verts[vat(first + k)] = vec2<u32>(q.x | (q.y << 10u) | (q.z << 20u), oct_encode(v.nrm));
        atomicMin(&wg[W_ATOM + 0], u32(clamp(v.pos.x / 8.0, 0.0, 1.0) * 65535.0));
        atomicMin(&wg[W_ATOM + 1], u32(clamp(v.pos.y / 8.0, 0.0, 1.0) * 65535.0));
        atomicMin(&wg[W_ATOM + 2], u32(clamp(v.pos.z / 8.0, 0.0, 1.0) * 65535.0));
        atomicMax(&wg[W_ATOM + 3], u32(ceil(clamp(v.pos.x / 8.0, 0.0, 1.0) * 65535.0)));
        atomicMax(&wg[W_ATOM + 4], u32(ceil(clamp(v.pos.y / 8.0, 0.0, 1.0) * 65535.0)));
        atomicMax(&wg[W_ATOM + 5], u32(ceil(clamp(v.pos.z / 8.0, 0.0, 1.0) * 65535.0)));
        nsum += v.nrm;
    }
    return nsum;
}

fn cone_of(slot: u32, first: u32, n: u32, axis: vec3<f32>) -> f32 {
    var cmin = 1.0;
    for (var k = 0u; k < n; k++) {
        cmin = min(cmin, dot(axis, oct_decode(verts[vat(first + k)].y)));
    }
    return cmin;
}

@compute @workgroup_size(256)
fn vertices(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // No thread returns before a barrier: one Vulkan driver drops barriers that follow a return, even a uniform one.
    let w = wid.x + wid.y * nwg.x;
    load_block_1(t, w);
    workgroupBarrier();
    load_block_2(t);
    workgroupBarrier();
    let slot = wget(W_SLOT);
    let l = wget(W_LEVEL);
    let live = slot != 0xFFFFFFFFu;
    let bc = wg_bc();
    let ha = cell_head(l, slot, bc, 2u * t);
    let hb = cell_head(l, slot, bc, 2u * t + 1u);
    let ca = (ha >> 10u) & 7u;
    let cb = (hb >> 10u) & 7u;
    wset(W_SCAN + t, ca + cb);
    workgroupBarrier();
    let within = scan_1(t, ca + cb);
    workgroupBarrier();
    let sc = scan_2(t, within);
    workgroupBarrier();
    // The vertex pages for them (none if the pool ran out: the block shows nothing until the pool grows).
    if (t == 0u) {
        var got = 0u;
        if (live) {
            let want = (min(sc.y, VSTRIDE) + VPAGE - 1u) / VPAGE;
            if (want > 0u) {
                let old = atomicSub(&status[4], i32(want));
                if (old < i32(want)) {
                    atomicAdd(&status[4], i32(want));
                    atomicStore(&status[5], 1);
                } else {
                    for (var k = 0u; k < want; k++) {
                        let page = free_pages[P.page_capacity + u32(old) - 1u - k];
                        wset(W_PAGES + k, page);
                        slot_pages[slot * SLOT_WORDS + VP_AT + k] = page;
                    }
                    got = want;
                }
            }
            slot_pages[slot * SLOT_WORDS + VP_N] = got;
        }
        wset(W_PAGES + MAX_VPAGES, got * VPAGE);
    }
    workgroupBarrier();
    let room = wget(W_PAGES + MAX_VPAGES);
    let first_a = sc.x;
    let first_b = sc.x + ca;
    // Kept while they fit: the cells past the block's vertex budget get none (a prefix of cells is kept).
    let keep_a = ca > 0u && first_a + ca <= room;
    let keep_b = cb > 0u && first_b + cb <= room;
    if (live) {
        cells[slot * 512u + 2u * t] = select(0u, ha | first_a, keep_a);
        cells[slot * 512u + 2u * t + 1u] = select(0u, hb | first_b, keep_b);
    }
    var nsum = vec3<f32>(0.0);
    if (keep_a) {
        nsum += write_cell(l, slot, bc, 2u * t, ha, first_a);
    }
    if (keep_b) {
        nsum += write_cell(l, slot, bc, 2u * t + 1u, hb, first_b);
    }
    let mine = select(0u, ca, keep_a) + select(0u, cb, keep_b);
    if (mine > 0u) {
        atomicAdd(&wg[W_ATOM + 10], mine);
        atomicAdd(&wg[W_ATOM + 6], bitcast<u32>(i32(nsum.x * 1024.0)));
        atomicAdd(&wg[W_ATOM + 7], bitcast<u32>(i32(nsum.y * 1024.0)));
        atomicAdd(&wg[W_ATOM + 8], bitcast<u32>(i32(nsum.z * 1024.0)));
    }
    storageBarrier();
    workgroupBarrier();
    let axis_raw = vec3<f32>(f32(bitcast<i32>(atomicLoad(&wg[W_ATOM + 6]))), f32(bitcast<i32>(atomicLoad(&wg[W_ATOM + 7]))), f32(bitcast<i32>(atomicLoad(&wg[W_ATOM + 8]))));
    let axis = axis_raw / max(length(axis_raw), 1e-6);
    var cmin = 1.0;
    if (keep_a) {
        cmin = min(cmin, cone_of(slot, first_a, ca, axis));
    }
    if (keep_b) {
        cmin = min(cmin, cone_of(slot, first_b, cb, axis));
    }
    if (mine > 0u) {
        atomicMin(&wg[W_ATOM + 9], u32(floor(cmin * 32767.0) + 32768.0));
    }
    workgroupBarrier();
    if (t == 0u && live) {
        let any = atomicLoad(&wg[W_ATOM + 10]) > 0u;
        let bmin = vec3<f32>(f32(atomicLoad(&wg[W_ATOM + 0])), f32(atomicLoad(&wg[W_ATOM + 1])), f32(atomicLoad(&wg[W_ATOM + 2]))) / 65535.0 * 8.0;
        let bmax = vec3<f32>(f32(atomicLoad(&wg[W_ATOM + 3])), f32(atomicLoad(&wg[W_ATOM + 4])), f32(atomicLoad(&wg[W_ATOM + 5]))) / 65535.0 * 8.0;
        if (any) {
            clusters[slot * 3u] = vec4<f32>(bmin, 1.0);
            clusters[slot * 3u + 1u] = vec4<f32>(bmax, 1.0);
            clusters[slot * 3u + 2u] = vec4<f32>(axis, (f32(atomicLoad(&wg[W_ATOM + 9])) - 32768.0) / 32767.0);
        } else {
            clusters[slot * 3u] = vec4<f32>(0.0);
            clusters[slot * 3u + 1u] = vec4<f32>(0.0);
            clusters[slot * 3u + 2u] = vec4<f32>(0.0, 0.0, 0.0, -2.0);
        }
    }
}

// ------------------------------------------------------------------------------ faces

// The slot of the block at this level holding level `l` cell `q` near this block (-1 if none).
fn same_level(q: vec3<i32>) -> i32 {
    let o = (q >> vec3<u32>(3u)) - wg_bc() + vec3<i32>(1);
    return bitcast<i32>(wget(W_NB + u32(o.x + 3 * o.y + 9 * o.z)));
}

// Around the edge from `start` along `axis`, in quad order: below along both other axes, below along the second, the
// cell starting at `start`, below along the first.
fn around_at(start: vec3<i32>, axis: u32, j: u32) -> vec3<i32> {
    let du = select(0, 1, j == 0u || j == 3u);
    let dv = select(0, 1, j == 0u || j == 1u);
    return start - place(axis, 0, du, dv);
}

// The quad corner at cell `q` (level `l`, not covered) for the edge starting at `start` along `axis`, or 0xFFFFFFFF if
// that cell has no vertex.
fn quad_corner(l: u32, q: vec3<i32>, start: vec3<i32>, axis: u32) -> u32 {
    let s = same_level(q);
    if (s >= 0) {
        let cell = local8(q);
        let info = cells[u32(s) * 512u + cell];
        if (((info >> 10u) & 7u) == 0u) {
            return 0xFFFFFFFFu;
        }
        var piece = 0u;
        if (((info >> 21u) & 1u) == 0u) {
            let lo = start - q;
            let k0 = u32(lo.x | (lo.y << 1u) | (lo.z << 2u));
            piece = (table[(info >> 13u) & 255u] >> (2u * edge_index(k0, axis))) & 3u;
        }
        return (u32(s) << 11u) | (piece << 9u) | cell;
    }
    // A coarser cell.
    for (var m = i32(l) - 1; m >= 0; m--) {
        let qm = q >> vec3<u32>(u32(i32(l) - m));
        let sm = slot_of(u32(m), qm >> vec3<u32>(3u));
        if (sm >= 0) {
            let cell = local8(qm);
            let info = cells[u32(sm) * 512u + cell];
            if (((info >> 10u) & 7u) == 0u) {
                return 0xFFFFFFFFu;
            }
            return (u32(sm) << 11u) | cell;
        }
    }
    return 0xFFFFFFFFu;
}

// The quads cell `ci` owns; with `write`, stored from quad number `base` on. Returns how many.
fn cell_quads(l: u32, slot: u32, bc: vec3<i32>, ci: u32, write: bool, base: u32) -> u32 {
    if (l > P.cap) {
        return 0u;
    }
    let c = cell_coords(bc, ci);
    if (covered(l, c)) {
        return 0u;
    }
    let d = corners(l, slot, bc, c);
    let config = config_of(d);
    if (config == 0u || config == 255u) {
        return 0u;
    }
    var n = 0u;
    for (var e = 0u; e < 12u; e++) {
        let ka = edge_a(e);
        let va = dget(d, ka);
        let vb = dget(d, edge_b(e));
        if ((va < 0.0) == (vb < 0.0)) {
            continue;
        }
        let axis = e / 4u;
        let lo = corner_off(ka);
        let start = c + lo;
        let a0 = around_at(start, axis, 0u);
        let a1 = around_at(start, axis, 1u);
        let a2 = around_at(start, axis, 2u);
        let a3 = around_at(start, axis, 3u);
        if (covered(l, a0) || covered(l, a1) || covered(l, a2) || covered(l, a3)) {
            continue;
        }
        // The owner: the first at this level in the order start, below along the second axis, below along the first,
        // below along both.
        var owner = 0u;
        if (same_level(a2) >= 0) {
            owner = 2u;
        } else if (same_level(a1) >= 0) {
            owner = 1u;
        } else if (same_level(a3) >= 0) {
            owner = 3u;
        }
        // This cell's place among the four.
        let lu = comp(lo, (axis + 1u) % 3u);
        let lv = comp(lo, (axis + 2u) % 3u);
        let mine = select(select(2u, 3u, lu == 1), select(1u, 0u, lu == 1), lv == 1);
        if (owner != mine) {
            continue;
        }
        if (write) {
            var c0 = quad_corner(l, a0, start, axis);
            var c1 = quad_corner(l, a1, start, axis);
            var c2 = quad_corner(l, a2, start, axis);
            var c3 = quad_corner(l, a3, start, axis);
            if (c0 == 0xFFFFFFFFu || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || c3 == 0xFFFFFFFFu) {
                // Counted as written, but drawn as nothing.
                c0 = 0u;
                c1 = 0u;
                c2 = 0u;
                c3 = 0u;
                atomicAdd(&status[3], 1);
            }
            if (va >= 0.0) {
                let t1 = c1;
                c1 = c3;
                c3 = t1;
            }
            let q = base + n;
            if (q < MAX_PAGES * PAGE_QUADS) {
                let page = wget(W_PAGES + q / PAGE_QUADS);
                if (page != 0xFFFFFFFFu) {
                    let at = page * PAGE_INDICES + (q % PAGE_QUADS) * 6u;
                    pages[at] = c0;
                    pages[at + 1u] = c1;
                    pages[at + 2u] = c2;
                    pages[at + 3u] = c0;
                    pages[at + 4u] = c2;
                    pages[at + 5u] = c3;
                }
            }
        }
        n += 1u;
    }
    return n;
}

@compute @workgroup_size(256)
fn faces(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wid.x + wid.y * nwg.x;
    load_block_1(t, w);
    workgroupBarrier();
    load_block_2(t);
    workgroupBarrier();
    let slot = wget(W_SLOT);
    let l = wget(W_LEVEL);
    let live = slot != 0xFFFFFFFFu;
    let bc = wg_bc();
    let na = cell_quads(l, slot, bc, 2u * t, false, 0u);
    let nb = cell_quads(l, slot, bc, 2u * t + 1u, false, 0u);
    wset(W_SCAN + t, na + nb);
    workgroupBarrier();
    let within = scan_1(t, na + nb);
    workgroupBarrier();
    let sc = scan_2(t, within);
    workgroupBarrier();
    if (t == 0u && live) {
        let quads = min(sc.y, MAX_PAGES * PAGE_QUADS);
        var want = (quads + PAGE_QUADS - 1u) / PAGE_QUADS;
        var got = 0u;
        if (want > 0u) {
            let old = atomicSub(&status[0], i32(want));
            if (old < i32(want)) {
                atomicAdd(&status[0], i32(want));
                atomicStore(&status[1], 1);
            } else {
                for (var k = 0u; k < want; k++) {
                    wset(W_PAGES + k, free_pages[u32(old) - 1u - k]);
                }
                got = want;
            }
        }
        for (var k = got; k < MAX_PAGES; k++) {
            wset(W_PAGES + k, 0xFFFFFFFFu);
        }
        slot_pages[slot * SLOT_WORDS] = got;
        slot_pages[slot * SLOT_WORDS + 1u] = select(0u, quads, got > 0u);
        for (var k = 0u; k < got; k++) {
            slot_pages[slot * SLOT_WORDS + 2u + k] = wget(W_PAGES + k);
        }
    }
    workgroupBarrier();
    cell_quads(l, slot, bc, 2u * t, true, sc.x);
    cell_quads(l, slot, bc, 2u * t + 1u, true, sc.x + na);
}

// ------------------------------------------------------------------------------ cull

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let s = gid.x + gid.y * nwg.x * 64u;
    if (s >= P.slot_count) {
        return;
    }
    let n = slot_pages[s * SLOT_WORDS];
    if (n == 0u) {
        return;
    }
    let m = slot_meta[s];
    if (m.w < 0 || u32(m.w) > P.cap) {
        return;
    }
    let l = u32(m.w);
    let vx = voxel(l);
    let corner = P.origin + vec3<f32>(m.xyz * 8) * vx;
    let lo = corner + clusters[s * 3u].xyz * vx;
    let hi = corner + clusters[s * 3u + 1u].xyz * vx;
    if (P.cull != 0u) {
        // Outside a frustum plane: every corner of the box on its far side.
        for (var k = 0u; k < 6u; k++) {
            let pl = P.planes[k];
            let far = select(lo, hi, pl.xyz >= vec3<f32>(0.0));
            if (dot(pl.xyz, far) + pl.w < 0.0) {
                return;
            }
        }
        // Facing away: every normal in the cone points from the eye, as seen from anywhere in the box.
        let cone = clusters[s * 3u + 2u];
        if (cone.w > 0.1) {
            let centre = 0.5 * (lo + hi);
            let radius = 0.5 * length(hi - lo);
            let to = centre - P.eye.xyz;
            let dist = length(to);
            if (dist > radius) {
                let sin_cone = sqrt(max(1.0 - cone.w * cone.w, 0.0));
                let sin_view = radius / dist;
                // Angle between the axis and the view direction beyond 90 degrees plus both half-angles.
                let c = dot(cone.xyz, to / dist);
                let cos_view = sqrt(max(1.0 - sin_view * sin_view, 0.0));
                // Every normal points away from every view direction when the axis is within 90 degrees less both
                // half-angles (the cone's and the box's as seen from the eye) of the view direction, with a margin
                // for normals taken from the field rather than the faces.
                let cos_sum = cone.w * cos_view - sin_cone * sin_view;
                let sin_sum = sin_cone * cos_view + cone.w * sin_view;
                if (cos_sum > 0.1 && c > sin_sum + 0.1) {
                    return;
                }
            }
        }
    }
    let quads = slot_pages[s * SLOT_WORDS + 1u];
    let at = u32(atomicAdd(&status[2], i32(n)));
    for (var k = 0u; k < n; k++) {
        let page = slot_pages[s * SLOT_WORDS + 2u + k];
        let q = min(quads - k * PAGE_QUADS, PAGE_QUADS);
        let d = (at + k) * 5u;
        draws[d] = q * 6u;
        draws[d + 1u] = 1u;
        draws[d + 2u] = page * PAGE_INDICES;
        draws[d + 3u] = 0u;
        draws[d + 4u] = 0u;
    }
}

# Generator Boolean: spike results

Follow-up: `spike4_lod_progressive.md` (LOD-aware evaluation, progressive refinement, incremental hash).

Measured on the sculpt engine at `crates/sculpt-core/src/generator/` (spike code, not wired into
`Document`). Design context: `DESIGN.md` in this folder.

## TL;DR

* **A (analytic-SDF displacement boolean) is interactive** up to
  10k instances at mesh sizes up to 1.6M faces (slider drag 4-34 ms), and 1k at 6.3M faces (16 ms).
  10k at 6.3M faces is borderline (58 ms, 80 ms with mixed primitives). 100k instances is **not**
  interactive at 1.6M+ faces on this machine (180-280 ms), mostly because every leaf is touched
  and the spatial hash is rebuilt from scratch (15-42 ms).
* **B (sparse brick SDF + surface nets) is not interactive at any density measured.** Cold it
  costs 1.9 s to 5 minutes. Even with the mesh distance field cached and free, 1k instances cost
  45-240 ms, 10k cost 0.18-2.5 s and 100k cost 1.3-14 s. It changes topology.
* **Recommendation: hybrid.** Live A as the layer's evaluator (keeps the per-vertex layer, mask and
  undo model). Optional explicit "Apply / Commit as solid" using B for true booleans (overhangs,
  crisp walls), run as a background job and not undoable per vertex. Details at the end.

## Machine and method

* Container, **4 cores** (`nproc` = 4), 15 GB RAM, no GPU. `cargo build --release` (thin LTO), all
  4 rayon threads. Absolute numbers will scale roughly with core count for the parallel parts
  (boolean evaluation, mesh SDF); the serial parts (prefix sums in the hash rebuild) will not.
* Meshes: the same `quad_sphere` + Catmull-Clark path as `sculpt-cli bench`: levels 7, 8, 9, 10 =
  98,304 / 393,216 / 1,572,864 / 6,291,456 faces (4096 leaves at the top).
* Instances: centred on the surface at points from the mask-weighted sampler (uniform mask),
  oriented to the surface normal with a random spin, diameter uniform in 3-10 mesh edge lengths
  (mean edge length measured per mesh: 0.0113, 0.00565, 0.00283, 0.00141 on a unit sphere).
  Main tables use spheres; a second run uses all 7 primitives with random non-uniform scale.
* Timings are the median of 3-5 runs after one warm-up. Spike B timings are **extrapolated** (see below).
* "Slider" = every instance changes size (x1.1): new transforms are baked, the hash is rebuilt,
  then every leaf is re-evaluated. "1%" = 1% of the instances change (x1.3 size): bake those, rebuild the
  hash, find leaves overlapping the old and new AABBs of the changed instances, re-evaluate only those.
  Placement is cached and not part of either (it only runs when seed, count, mask or topology change).

### Commands

```
cd sculpt
cargo build --release
cargo test --release --workspace
cargo clippy --release --workspace
./target/release/sculpt-cli generator-bench 7,8,9,10                    # spheres, A + B  (tables below)
./target/release/sculpt-cli generator-bench 7,8,9,10 --no-voxel --mixed # 7 primitives, A only
```

`generator-bench [levels] [counts] [--no-voxel] [--mixed]`; counts default to 1000,10000,100000.
Raw output: the markdown rows below are the program's output, reformatted only for units.

## Spike A: what was built

`generator/sdf.rs`: `Primitive` {Box, Sphere, Cylinder, Cone, Wedge, ChamferCutter, Torus} with exact
SDFs in a unit local frame and `sdf_grad` (analytic for sphere and box, tetrahedral differences of the exact SDF
otherwise). `Instance { primitive, rotation, translation, scale }`; non-uniform scale uses
`sdf * min(scale)`, a conservative lower bound (distances are under-estimated, so projection converges a little slower).

`generator/instances.rs`: `InstanceSet` = baked inverse transforms + AABBs inflated by a margin + a **sparse uniform-grid
spatial hash** (open addressing keyed by packed cell coordinates; cell = mean instance extent, so an instance
covers about 8 cells; lock-free parallel counting build, per-cell lists sorted for determinism). `update(changes)`
re-bakes only the changed instances and returns the old and new AABBs for leaf lookup.

`generator/mod.rs`: per-vertex rules, evaluated per leaf under rayon, output = per-vertex delta (`Vec3`) relative to the input positions.
Leaves whose bounds touch no occupied cell are skipped (their delta range is zeroed). `affected_leaves` walks the engine BVH
to turn changed AABBs into a sorted leaf list; `evaluate_leaves` re-evaluates only those. Incremental result equals a full evaluation
(unit test `incremental_equals_full`).

* **Subtract**: `p -= d * grad` for the most-penetrating instance, up to 3 iterations (candidates beyond
  `iterations * |d1|` are dropped after pass 1, since SDFs are 1-Lipschitz). Overlapping cutters converge to the union boundary.
* **Intersect**: vertices outside every nearby instance but within `reach` of the closest are pulled in the same way. Inside, or beyond `reach`: untouched.
* **Add (heightfield union along the normal)**: sphere-trace a ray from `p` along `+n` against the instance SDF, bisect-refine the crossing (8 steps).
  Vertex already inside the primitive moves to the **exit** point (the primitive's top as seen along `n`);
  vertex outside whose ray enters the primitive within `reach` moves to the **entry** point (glues a floating primitive); misses are untouched.
  Result is single-valued along each vertex normal, so **no overhangs or undercuts**.
* Fast path for uniformly scaled spheres (no rotation); sphere AABB is exact for the rotated ellipsoid.

`generator/place.rs`:
* `SurfaceSampler`: per-face CDF of `area * mean(vertex mask)` (parallel two-pass scan, f64), `sample(n, seed)`: stratified-jittered
  position in the CDF (one stratum per sample, so coverage is even and sample order is spatially coherent) then binary search, triangle pick, uniform barycentric point
  refined by rejection on the *interpolated* mask so face borders do not leak. Deterministic (hash of seed, index). Returns position, interpolated vertex normal, face, triangle verts and barycentrics.
* `EdgeSampler`: dihedral-angle detection (no sort, no edge map: for each face edge look the neighbour up in the vertex-face CSR),
  chains = flood fill through vertices with exactly two sharp edges, then points evenly spaced by arc length over the global CDF of edge lengths.
  Returns position, tangent, edge normal (bisector of the two face normals), dihedral angle, chain length, chain id.

### Tests (`crates/sculpt-core/tests/generator.rs`, 9 tests, all pass)

Subtract sphere from a flat grid (all moved vertices land within 2e-3 of the sphere surface, none end inside, outside vertices unchanged);
Add lifts to the sphere top; Intersect pulls in; all 7 primitives project onto their surface; incremental == full; spatial hash returns every overlapping instance
vs brute force; mask weighting puts >= 95% of 20k points in white regions (deterministic, unit normals); edge sampling on a cube lies on the 12 edges, tangent/bisector/chain length (2.0)/dihedral (90 deg) correct, 80 samples per edge; MeshSdf sign/distance on a sphere + the B pipeline produces a surface.

## Spike A results (ms, spheres, 4 cores)

### Placement

Cold = build the cacheable structure once; warm = what a seed or count change costs.

| Faces | CDF build (cold, once) | edge detect + chains (cold, once, 30 deg) | edge sample 100k pts (warm) | sharp edges found |
|---|---|---|---|---|
| 98,304 | 0.5 ms | 3.0 ms | 2.83 ms | 1536 |
| 393,216 | 1.7 ms | 6.7 ms | 2.17 ms | 3072 |
| 1,572,864 | 5.9 ms | 24.9 ms | 2.74 ms | 6144 |
| 6,291,456 | 24.7 ms | 116.9 ms | 2.65 ms | 12288 |

(Edge placement was run on the same topology mapped onto a cube so it has real sharp edges; a smooth sphere has none at 30 degrees. The cost of detection does not depend on how many edges are sharp, only the chain step does.)

Surface placement, per instance count (`placement cold` includes the face CDF build, `warm` reuses it; both include building instance transforms):

| Faces | N=1k cold / warm | N=10k cold / warm | N=100k cold / warm |
|---|---|---|---|
| 98,304 | 1.0 / 0.14 ms | 1.5 / 0.81 ms | 7.9 / 6.88 ms |
| 393,216 | 1.4 / 0.15 ms | 2.4 / 1.03 ms | 8.0 / 7.46 ms |
| 1,572,864 | 5.1 / 0.18 ms | 5.7 / 1.66 ms | 13.3 / 8.81 ms |
| 6,291,456 | 21.4 / 0.28 ms | 22.0 / 2.00 ms | 36.3 / 15.73 ms |

### Boolean (Subtract, spheres): slider drag and 1% incremental

| Faces | N | placement cold | placement warm | hash rebuild | SLIDER: update+hash | SLIDER: boolean | **SLIDER total** | 1%: update+hash | 1%: leaf query | 1%: boolean | **1% total** | leaves touched (all / 1% / total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 98,304 | 1,000 | 1.0 | 0.14 | 0.78 | 0.9 | 3.1 | **4.0** | 0.90 | 0.02 | 0.69 | **1.62** | 64/12/64 |
| 98,304 | 10,000 | 1.5 | 0.81 | 2.21 | 2.4 | 12.3 | **14.7** | 2.62 | 0.05 | 10.90 | **13.58** | 64/64/64 |
| 98,304 | 100,000 | 7.9 | 6.88 | 14.65 | 17.3 | 66.0 | **83.3** | 14.78 | 0.21 | 54.46 | **69.44** | 64/64/64 |
| 393,216 | 1,000 | 1.4 | 0.15 | 0.76 | 1.2 | 4.3 | **5.5** | 0.81 | 0.04 | 0.35 | **1.21** | 256/14/256 |
| 393,216 | 10,000 | 2.4 | 1.03 | 3.92 | 4.6 | 19.2 | **23.9** | 3.69 | 0.06 | 8.50 | **12.25** | 256/127/256 |
| 393,216 | 100,000 | 8.0 | 7.46 | 14.50 | 16.9 | 84.8 | **101.6** | 18.97 | 0.37 | 69.62 | **88.97** | 256/256/256 |
| 1,572,864 | 1,000 | 5.1 | 0.18 | 0.58 | 0.9 | 9.5 | **10.4** | 0.66 | 0.02 | 0.32 | **1.00** | 953/14/1024 |
| 1,572,864 | 10,000 | 5.7 | 1.66 | 4.55 | 4.8 | 29.5 | **34.3** | 5.15 | 0.12 | 4.91 | **10.17** | 1024/160/1024 |
| 1,572,864 | 100,000 | 13.3 | 8.81 | 29.91 | 35.1 | 143.4 | **178.5** | 31.99 | 0.34 | 109.70 | **142.03** | 1024/903/1024 |
| 6,291,456 | 1,000 | 21.4 | 0.28 | 0.63 | 0.7 | 14.8 | **15.5** | 0.63 | 0.02 | 0.23 | **0.88** | 1750/17/4096 |
| 6,291,456 | 10,000 | 22.0 | 2.00 | 4.26 | 4.6 | 53.9 | **58.5** | 4.99 | 0.10 | 2.31 | **7.40** | 4094/167/4096 |
| 6,291,456 | 100,000 | 36.3 | 15.73 | 40.26 | 46.8 | 231.1 | **277.9** | 41.61 | 0.54 | 74.46 | **116.61** | 4096/1480/4096 |

Notes on the table:

* "hash rebuild" is the full rebuild of the spatial hash alone. Bake + hash for a fresh set ("set build") is the same order (see the bench output).
* At 98k faces, 100k instances of 3-10 edges each means **every vertex sits inside about 57 instance AABBs**; it is a stress test of overlap, not a realistic density.
* Floor at 1k instances on big meshes (15 ms at 6.3M) is the per-vertex hash probe in the 1,750 leaves whose bounds touch an occupied cell (2.7M probes), not instance work. A per-instance scatter or finer-than-leaf bounds would cut it to about 1 ms.
* 100k instances at >= 1.6M faces: the 1% change still touches 900-1500 leaves because 1,000 changed instances are spread over the whole surface, and the full hash rebuild is 40 of the ~120 ms. An incremental hash (remove/insert only the changed instances) and per-leaf instance lists would fix the second; the first is inherent.

### Per-operation cost (full re-evaluation, ms)

Add and Intersect use a margin/reach of 2 mesh edges (bigger candidate sets, and Add sphere-traces). "Intersect ... verts moved 0" in the dense-sphere runs is real: with instances centred on the surface at that density every vertex is inside the union.

| Primitives | Faces | N | Subtract ms | Add ms | Intersect ms |
|---|---|---|---|---|---|
| spheres | 98,304 | 1,000 | 2.2 | 5.5 | 3.8 |
| spheres | 98,304 | 10,000 | 10.4 | 47.9 | 16.8 |
| spheres | 98,304 | 100,000 | 53.0 | 385.7 | 136.0 |
| spheres | 393,216 | 1,000 | 3.7 | 7.0 | 5.3 |
| spheres | 393,216 | 10,000 | 16.0 | 48.7 | 28.2 |
| spheres | 393,216 | 100,000 | 73.4 | 418.7 | 157.7 |
| spheres | 1,572,864 | 1,000 | 11.6 | 13.0 | 12.5 |
| spheres | 1,572,864 | 10,000 | 28.2 | 57.0 | 41.8 |
| spheres | 1,572,864 | 100,000 | 127.6 | 438.5 | 208.6 |
| spheres | 6,291,456 | 1,000 | 15.5 | 20.6 | 19.1 |
| spheres | 6,291,456 | 10,000 | 49.4 | 76.4 | 69.8 |
| spheres | 6,291,456 | 100,000 | 197.7 | 486.5 | 333.6 |
| mixed (7 prims, non-uniform scale) | 98,304 | 1,000 | 3.5 | 8.2 | 8.6 |
| mixed (7 prims, non-uniform scale) | 98,304 | 10,000 | 22.4 | 72.8 | 45.5 |
| mixed (7 prims, non-uniform scale) | 98,304 | 100,000 | 185.0 | 728.4 | 438.2 |
| mixed (7 prims, non-uniform scale) | 393,216 | 1,000 | 5.7 | 10.0 | 10.4 |
| mixed (7 prims, non-uniform scale) | 393,216 | 10,000 | 29.6 | 81.7 | 85.8 |
| mixed (7 prims, non-uniform scale) | 393,216 | 100,000 | 178.6 | 735.7 | 444.2 |
| mixed (7 prims, non-uniform scale) | 1,572,864 | 1,000 | 13.6 | 16.2 | 16.8 |
| mixed (7 prims, non-uniform scale) | 1,572,864 | 10,000 | 43.9 | 90.9 | 105.1 |
| mixed (7 prims, non-uniform scale) | 1,572,864 | 100,000 | 250.6 | 751.9 | 587.2 |
| mixed (7 prims, non-uniform scale) | 6,291,456 | 1,000 | 22.2 | 32.4 | 29.0 |
| mixed (7 prims, non-uniform scale) | 6,291,456 | 10,000 | 65.3 | 117.4 | 117.0 |
| mixed (7 prims, non-uniform scale) | 6,291,456 | 100,000 | 345.8 | 826.8 | 942.5 |

All 7 primitives, random non-uniform scale (slider total / 1% total, ms):

| Faces | N | slider total (update+hash+boolean) | 1% total |
|---|---|---|---|
| 98,304 | 1,000 | 4.8 | 1.84 |
| 98,304 | 10,000 | 25.9 | 24.20 |
| 98,304 | 100,000 | 243.8 | 196.54 |
| 393,216 | 1,000 | 8.5 | 1.67 |
| 393,216 | 10,000 | 38.7 | 24.08 |
| 393,216 | 100,000 | 277.9 | 205.22 |
| 1,572,864 | 1,000 | 12.9 | 1.10 |
| 1,572,864 | 10,000 | 52.0 | 12.84 |
| 1,572,864 | 100,000 | 296.7 | 284.68 |
| 6,291,456 | 1,000 | 21.7 | 1.20 |
| 6,291,456 | 10,000 | 79.6 | 7.65 |
| 6,291,456 | 100,000 | 440.6 | 187.21 |

## Spike B results (measure only)

### What was implemented

All in `generator/voxel.rs`, driven by `generator-bench`:

1. **Brick enumeration**: 8^3-voxel bricks overlapping each instance AABB + a 2-voxel band, packed to 63-bit keys, parallel sort + dedup. Full implementation, measured exactly.
2. **Classification**: one mesh query at each brick centre; |d| <= half-diagonal + band => "near" (needs per-sample exact distance), otherwise "far" (mesh distance taken as the centre value; only instance SDFs vary).
3. **Mesh signed distance** per lattice sample (9^3 per brick): real BVH closest-point query (my own per-leaf trees under the engine's coarse leaf hierarchy, children visited nearest-first, each query bounded by the previous sample's distance + 1 voxel), exact point-triangle distance, sign from **angle-weighted vertex pseudo-normals interpolated at the closest point**. Exact for smooth closed meshes; an approximation at creases and wrong on non-watertight or self-intersecting meshes. Unoptimized (about 1.6-3 us per query per core, derived from the near-brick counts and times).
4. **Combine**: `max(mesh, -min(instance sdf))` (subtract) via the same spatial hash. Real.
5. **Naive surface nets** inside each brick (vertex per sign-changing cell, quad per sign-changing lattice edge whose 4 cells are in the brick). Real, but quads that straddle brick borders are not emitted (undercounts quads by about 1/8) and there is no stitching to the untouched mesh.

**Honesty about extrapolation.** A full run at 100k instances is minutes to an hour of 4-core CPU, so steps 2-5 are timed on a **deterministic, evenly strided sample of 2,000 bricks** and multiplied by `bricks / 2000`
(every row below says "extrapolated"; step 1 and all counts are exact). The sample is spread across the whole brick list, so near/far mix is representative, but treat times as +-20%.
"Warm" = brick enumeration + combine + nets, i.e. **if the mesh distance bricks were cached across drags and free to recompute**; that is optimistic (cache invalidation, 8x memory at 0.5e, new bricks as instances move) but it is the best case for B.
A triangle-rasterizing mesh-to-volume (OpenVDB style) would probably beat my per-sample BVH queries by several times, which would lower the "cold" column but not the warm one.

Mesh closest-point structure (built once, rebuilt on topology change; includes pseudo-normals): 9 ms / 4 MB (98k), 30 ms / 16 MB (393k), 122 ms / 63 MB (1.6M), 539 ms / 252 MB (6.3M).

### Numbers (spheres, voxel size 0.5 and 1.0 mesh edge lengths)

| Faces | N | voxel | bricks (after dedup) | near / far | voxels | brick enum | classify | mesh SDF | combine | nets | **total (cold)** | **warm (mesh SDF cached)** | patch verts out | brick MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 98,304 | 1,000 | 0.5e | 18,984 | 12150/6834 | 9,719,808 | 1 ms | 22 ms | 3.6 s | 96 ms | 76 ms | **3.8 s** | **172 ms** | 862,044 | 56 |
| 98,304 | 1,000 | 1e | 4,288 | 3360/928 | 2,195,456 | 0 ms | 7 ms | 1.9 s | 27 ms | 17 ms | **1.9 s** | **45 ms** | 235,593 | 13 |
| 98,304 | 10,000 | 0.5e | 29,592 | 13316/16276 | 15,151,104 | 4 ms | 43 ms | 4.0 s | 511 ms | 85 ms | **4.6 s** | **600 ms** | 660,390 | 87 |
| 98,304 | 10,000 | 1e | 5,391 | 3429/1962 | 2,760,192 | 1 ms | 9 ms | 1.7 s | 157 ms | 18 ms | **1.9 s** | **176 ms** | 192,755 | 16 |
| 98,304 | 100,000 | 0.5e | 33,548 | 13587/19961 | 17,176,576 | 61 ms | 80 ms | 4.0 s | 4.7 s | 129 ms | **9.1 s** | **4.9 s** | 552,955 | 98 |
| 98,304 | 100,000 | 1e | 5,926 | 3440/2486 | 3,034,112 | 21 ms | 12 ms | 1.7 s | 1.3 s | 16 ms | **3.1 s** | **1.3 s** | 150,177 | 17 |
| 393,216 | 1,000 | 0.5e | 29,407 | 20747/8660 | 15,056,384 | 1 ms | 50 ms | 6.2 s | 114 ms | 122 ms | **6.5 s** | **237 ms** | 1,492,979 | 86 |
| 393,216 | 1,000 | 1e | 9,516 | 8412/1104 | 4,872,192 | 0 ms | 20 ms | 4.0 s | 32 ms | 37 ms | **4.1 s** | **69 ms** | 628,013 | 28 |
| 393,216 | 10,000 | 0.5e | 99,607 | 56527/43080 | 50,998,784 | 6 ms | 196 ms | 17.2 s | 824 ms | 512 ms | **18.7 s** | **1.3 s** | 3,881,037 | 291 |
| 393,216 | 10,000 | 1e | 19,548 | 13703/5845 | 10,008,576 | 3 ms | 53 ms | 7.3 s | 219 ms | 95 ms | **7.7 s** | **318 ms** | 1,014,062 | 57 |
| 393,216 | 100,000 | 0.5e | 125,806 | 55795/70011 | 64,412,672 | 43 ms | 290 ms | 16.4 s | 4.8 s | 355 ms | **21.9 s** | **5.2 s** | 2,303,193 | 368 |
| 393,216 | 100,000 | 1e | 22,633 | 13648/8985 | 11,588,096 | 15 ms | 62 ms | 7.1 s | 1.7 s | 107 ms | **9.0 s** | **1.8 s** | 680,133 | 66 |
| 1,572,864 | 1,000 | 0.5e | 32,112 | 23554/8558 | 16,441,344 | 1 ms | 61 ms | 7.0 s | 96 ms | 122 ms | **7.3 s** | **218 ms** | 1,701,904 | 94 |
| 1,572,864 | 1,000 | 1e | 12,054 | 10885/1169 | 6,171,648 | 0 ms | 28 ms | 5.3 s | 37 ms | 49 ms | **5.4 s** | **87 ms** | 840,700 | 35 |
| 1,572,864 | 10,000 | 0.5e | 238,784 | 163089/75695 | 122,257,408 | 6 ms | 557 ms | 51.3 s | 1.0 s | 937 ms | **53.8 s** | **1.9 s** | 11,735,875 | 698 |
| 1,572,864 | 10,000 | 1e | 61,238 | 50062/11176 | 31,353,856 | 3 ms | 176 ms | 26.8 s | 277 ms | 277 ms | **27.6 s** | **557 ms** | 3,618,217 | 179 |
| 1,572,864 | 100,000 | 0.5e | 451,238 | 214338/236900 | 231,033,856 | 55 ms | 1.6 s | 66.7 s | 7.6 s | 1.9 s | **77.9 s** | **9.6 s** | 12,647,750 | 1319 |
| 1,572,864 | 100,000 | 1e | 84,366 | 55175/29191 | 43,195,392 | 27 ms | 438 ms | 28.3 s | 1.9 s | 302 ms | **31.0 s** | **2.2 s** | 3,742,180 | 247 |
| 6,291,456 | 1,000 | 0.5e | 32,503 | 23646/8857 | 16,641,536 | 1 ms | 64 ms | 6.8 s | 90 ms | 130 ms | **7.1 s** | **221 ms** | 1,713,948 | 95 |
| 6,291,456 | 1,000 | 1e | 12,482 | 11277/1205 | 6,390,784 | 0 ms | 34 ms | 6.2 s | 35 ms | 53 ms | **6.3 s** | **88 ms** | 846,667 | 36 |
| 6,291,456 | 10,000 | 0.5e | 312,382 | 220073/92309 | 159,939,584 | 6 ms | 828 ms | 72.0 s | 1.1 s | 1.4 s | **75.4 s** | **2.5 s** | 16,192,790 | 913 |
| 6,291,456 | 10,000 | 1e | 110,065 | 97903/12162 | 56,353,280 | 2 ms | 353 ms | 49.9 s | 329 ms | 427 ms | **51.1 s** | **758 ms** | 7,555,632 | 322 |
| 6,291,456 | 100,000 | 0.5e | 1,432,322 | 855812/576510 | 733,348,864 | 63 ms | 3.8 s | 280.1 s | 8.5 s | 5.4 s | **297.8 s** | **14.0 s** | 64,277,598 | 4188 |
| 6,291,456 | 100,000 | 1e | 295,368 | 218425/76943 | 151,228,416 | 24 ms | 1.1 s | 108.9 s | 2.1 s | 1.1 s | **113.3 s** | **3.2 s** | 16,311,402 | 864 |

* "patch verts out" is the surface-net vertex count over **all bricks touched**, including the untouched mesh surface inside them. At 0.5e that is about 4x the vertex density of the source mesh in the patched region; at 1e about 1x (and then the patch is a lower-fidelity resample of the sculpted detail).
* Memory ("brick MB"): 9^3 floats per brick plus key, dense per brick (about 2.9 KB/brick). At 6.3M faces and 100k instances at 0.5e that is 4.2 GB, plus 252 MB for the closest-point structure and the existing document (about 1.5 GB RSS before any of this). Peak RSS of the whole bench process, which only ever holds the 2,000-brick sample, was 1.78 GB.
* A's memory for comparison: about 0.18 KB per instance (instance + baked transform + AABB) plus the hash (a few MB at 100k), and the per-vertex delta (12 B/vertex, 75 MB at 6.3M if dense; it can be chunked-sparse like layers).

### What B does to topology and the per-vertex model

* The result is a new triangle/quad soup over the patched bricks: vertex count and ids change with every parameter change (e.g. 0.9-64M new vertices above vs. the same number of original vertices removed).
* Per-vertex sculpt layers, channels, masks, freeze and undo chunks are all indexed by vertex id and leaf ranges; they cannot follow a surface that is regenerated on each edit. They would need a transfer (closest-point / barycentric resample onto the new vertices) each time, and the transfer is lossy and not invertible.
* The patch must be stitched to the untouched mesh (cracks at brick borders, T-junctions, different density), and the BVH/leaf ranges rebuilt for the affected region, which breaks "vertices renumbered per leaf" locality unless the whole mesh is rebuilt.
* So B only fits as an **explicit commit** that produces a new base mesh (like a remesh), not as a live layer that sits above other per-vertex layers and re-evaluates on every slider drag.

## Recommendation

| Instances | 98k-1.6M faces | 6.3M faces | Approach |
|---|---|---|---|
| up to 1k | A: 4-10 ms drag, ~1 ms for a 1% edit | A: 16 ms drag, 1 ms edit | **A live**. B warm costs 45-240 ms, cold 1.9-7.3 s: no. |
| 10k | A: 15-34 ms drag, 10-14 ms edit | A: 58 ms drag (80 ms with mixed primitives), 7 ms edit | **A live.** Borderline at 6.3M on 4 cores; fine with more cores or a coarser live preview (skip Add's ray trace, or use reach 0). B: 2-75 s cold, 0.18-2.5 s warm: commit only. |
| 100k | A: 83-180 ms drag, 70-140 ms edit | A: 280 ms drag, 117 ms edit | **A is not interactive here as built.** Use a cheaper live display (e.g. evaluate only visible leaves / a decimated LOD, or only the instances in the camera frustum / near the cursor), hash incremental update, then full A on release. B: 3 s - 5 min cold (30-300 s at >= 1.6M faces), 1.3-14 s warm: commit only, background job. |

**Hybrid (live A, committed B) is the recommended design**:

1. The generator layer stores parameters; its evaluator is A. Output is a per-vertex delta, so it composes with layer opacity, the layer mask, other layers, freeze and per-vertex undo for free (the layer's contribution is `opacity * mask * delta_A`, recomputed per dirty leaf).
2. "Apply / Convert to solid" runs B in a background job for the cases A cannot do (overhangs from Add, thin/crisp walls, true intersect), and replaces the base like a remesh. Do not try to keep layers live across it; either flatten layers first or resample them.
3. Prefer a cheap non-destructive commit before B: "bake generator to layer" = store the A delta as an ordinary sparse layer (zero topology change).

### Cheap wins not done in the spike (expected effect)

* Incremental spatial hash (insert/remove changed instances only): removes 15-42 ms from 100k-instance edits.
* Per-leaf instance lists / per-instance scatter instead of per-vertex hash probes: cuts the 15-20 ms floor at 1k instances on 6.3M faces to a few ms.
* Skip leaves that are off-screen or at low LOD while dragging; evaluate the remainder on release.
* Cache sphere/box fast paths for other primitives (only uniform spheres have one now).

## Failure modes

**A (displacement)**
* **Aliasing when features are small relative to the mesh.** Walls are made of existing vertices only. Under about 3 edge lengths across, a cutter covers a handful of vertices and the dent is ragged and faceted; under 1-2 edges across it can cover **no vertex and vanish** entirely or flicker as instances move. The minimum useful instance size is about 3 mesh edges (the benchmark's lower bound), preferably more; the UI should warn or the layer should require subdivision.
* **No new geometry.** A box or cylinder cutter's vertical walls do not exist: the rim triangle spans from an untouched vertex outside to a projected vertex inside, giving a sloped skirt of one edge width rather than a crisp wall, and floors smear. Edge-damage chamfers will look like soft bevels at the density of the mesh unless the mesh is much denser than the feature.
* **ADD is a heightfield along the vertex normal.** No overhangs, undercuts, hollows, or anything that curves back over the surface; side walls collapse into stretched triangles; where the normal tilts steeply relative to the primitive the top surface is distorted. Features thinner than a few edges that rise from the surface alias the same way. The "glue" `reach` (entry snap) is a heuristic.
* **INTERSECT is local**: it only pulls vertices within `reach` of an instance; it is not an intersection with the whole of the layers below.
* **Overlapping cutters**: 3 projection iterations approximate the union boundary; deep concave junctions can retain a small residual (under the test tolerance for spheres, not proven in general).
* **Non-uniform scale** uses a lower-bound SDF (`min scale`), so projection under-shoots and may need more iterations; the mixed-primitive run exercises it but accuracy is not asserted beyond uniform scale.
* **Normals** of the result need recomputing (dirty leaves + neighbours), and Add reads the normals of the layers below.
* Cost is linear in vertex-instance overlaps: dense overlaps (N x footprint >> vertex count) are slow regardless of mesh size (see 98k faces, 100k instances).
* Gradient for torus, cone, wedge, cylinder, chamfer is finite-difference (tetrahedral, eps 1e-3 in local units): fine for projection, slightly noisy exactly on creases.

**B (voxel)**
* Topology change; layers, masks, freeze, undo do not map (see above). Seams/cracks against untouched mesh.
* Resolution loss: sculpted detail finer than the voxel is filtered out (at 1e you are at the Nyquist limit of the mesh; at 0.5e the patch has 4x the vertices).
* Surface nets round sharp edges (dual contouring or a feature-preserving fix needed for crisp chamfers).
* Sign from pseudo-normals fails on non-watertight, self-intersecting or open sculpts, which sculpting produces.
* Memory: dense 9^3 bricks, 4 GB at the largest case before any optimization; cold cost is dominated by the mesh distance bricks (45-98% of the total; the rest is mostly the combine step at 100k instances, where each lattice sample tests every instance in its cell).
* Numbers are extrapolated from 2,000-brick samples and use an unoptimized BVH query; the order of magnitude, not the digits, is the finding.

## Caveats

* 4-core container, shared host, no GPU, one run per configuration (medians of 3-5 repetitions inside a run). Expect 2-3x better evaluation times on 8-16 cores for A.
* Instances are centred on the surface; real edge-damage placement concentrates along a few chains, which touches far fewer leaves than the uniform-surface distribution benchmarked here (the 1% incremental numbers would be better, 100k would be cheaper per instance).
* Placement is deterministic but not blue noise (stratified in the CDF, so even in area, not minimum-distance).
* Edge placement was benchmarked on the sphere topology mapped to a cube, so sharp edges are only the 12 cube edges (chains of length 2 at the coarsest, 2 x edges at others); organic sculpts have far more noisy edges and the chain step is the cost that grows with them.
* The design doc's boolean decision (`DESIGN.md`) should be updated from this file; no existing engine behaviour was changed (only `pub mod generator;` in `lib.rs`, the `generator-bench` command in `sculpt-cli`).

# Spike 2: render-time CSG preview for the Boolean layer

Question: can a generator Boolean (subtract / add / intersect against many
analytic instances) be shown with pixel-exact edges at any mesh density, with
no mesh change, purely in the viewport shader?

**Answer: yes for the preview, with stated limits.** Edges come from analytic
ray/primitive intersection, so they are as crisp as the display (4x MSAA,
exact per sample), identical at 98k and 6.3M faces, and per-pixel cost does
not grow with instance count (10 to 100k measured). It is not a way to
produce geometry, and it is exact only for ray-convex base meshes
(see "What fails").

## Scope note

The brief refers to `docs/generator/DESIGN.md`, `docs/generator/spike_results.md`
and `sculpt-core/src/generator/`. **None of these exist in this repository**
(only `ARCHITECTURE.md` and the engine core are present; no LOD-tree rendering
either). So the instances come from a stand-in placement in
`viewport_csg::place` (jittered Fibonacci lattice, spheres or oriented boxes).
The only contract with a real generator is the 48-byte `Inst` record
(`kind`, centre, radius or half extents, rotation quaternion); any placement
that emits those plugs in. The base mesh is the app's own sphere
(`--level 7` = 98,304 faces, `--level 10` = 6,291,456 faces).

## Code

- `crates/sculpt-app/src/viewport_csg/mod.rs`: options, placement, CPU grid
  build, pipelines, prepasses.
- `crates/sculpt-app/src/viewport_csg/csg.wgsl`: prepass + per-sample resolve.
- `viewport.rs`: 4 small hooks (opt-in field, `enable_csg`, scene setup on
  topology change, prepass + resolve in place of the mesh draw). With
  `csg == None` the render path is byte-for-byte the old one; `viewport.wgsl`
  and the sculpt/upload path are untouched.
- `scripts/csg_bench.sh`: benchmark.

## Algorithm

Per pixel **sample** (the resolve shader uses `@builtin(sample_index)`, so it
runs at sample rate and the 4x MSAA resolve averages exact per-sample results):

1. **Prepass F**: rasterise the base mesh, keep only fronts, store depth +
   smooth normal in 4x MSAA targets (`Depth32Float`, `Rgba16Float`).
2. **Prepass B**: same, keep only back faces, nearest one. This gives the
   exit of the solid, i.e. the mesh's inside interval `[F, B]` along the ray
   (the "depth-peel the back face" answer to the inside/outside question).
3. **Resolve** (one fullscreen draw in the existing main pass): reconstruct the
   view ray for the sample, walk the instance grid with a 3D DDA
   (Amanatides-Woo) starting at the mesh entry, and combine exact analytic
   intervals (sphere quadratic, oriented-box slabs):
   - **Subtract**: `t = F`; repeat: if some cutter interval contains `t`,
     `t = that cutter's exit` (covers any union of overlapping cutters
     exactly, since cutters covering `t` are all listed in `t`'s cell). Finish
     when `t` lies inside the current cell (early out). Result: `t == F` mesh
     surface; `F < t < B` cutter inner wall, normal = -cutter normal (the
     wall is shaded exactly where the ray exits the cutter inside solid
     material); `t >= B` ray left the solid inside a cutter: see-through hole.
   - **Add**: nearest of mesh front `F` and the earliest primitive *entry*;
     primitive surface is visible only where it is outside the mesh, which this
     gives for free. Early out once a cell starts beyond the best hit.
   - **Intersect**: first `t >= F`, `t < B`, covered by any primitive interval;
     `t == F` shows the mesh, otherwise the primitive surface inside the mesh.
4. Shade with the app's clay model; primitive walls tinted (subtract orange,
   add blue, intersect green).

Instance data: three read-only storage buffers: `Inst[]` (48 B each),
`cells[] = (offset, count)` and the flat `cell_idx[]`. Grid cell size is about
the instance diameter, capped at 128 cells per axis; an instance is listed in
every cell its bounding sphere touches (about 8 to 27 entries each). Build is
CPU, linear: 100k instances take 70 ms (placement 4 ms), so retuning a
parameter slider rebuilds in under a frame at 10k.

## Images (`docs/generator/img/`)

All made with the existing `--screenshot` path. Whole-window screenshots at
1500x1000.

| file | scene |
|---|---|
| `a_subtract_98k.png`, `a_subtract_6m.png` | 60 sphere cutters on a smooth sphere, 98k vs 6.3M faces |
| `a_zoom_98k.png`, `a_zoom_6m.png` | same crop, 6x nearest-neighbour zoom |
| `b_add_boxes.png` | 70 oriented boxes added |
| `c_intersect.png` | 40 spheres intersected (lens patches and inner caps) |
| `d_subtract_10k.png` | 10,000 cutters, many overlapping |
| `d_subtract_10k_heat.png` | same, heat map of instance tests per sample (blue 0 to red 64+) |

Edge equality, 98k vs 6.3M (whole viewport, 874k pixels): 1,945 pixels differ
at all, 294 by more than 24/255, all along the silhouette (the Catmull-Clark
levels shrink the sphere slightly; the CSG edges themselves coincide). The
zoom crops are visually indistinguishable.

## Commands

```sh
# build (first build is slow)
cargo build --release --workspace

# screenshots (headless: xvfb-run -a -s "-screen 0 2000x1300x24" <cmd>)
sculpt-app --size 1500x1000 --test-strokes 0 --screenshot out.png \
  --level 7  --csg-preview subtract  --csg-count 60 --csg-size 0.6     # a (98k)
sculpt-app ... --level 10 --csg-preview subtract --csg-count 60 --csg-size 0.6   # a (6.3M)
sculpt-app ... --level 7  --csg-preview add       --csg-count 70 --csg-size 0.3   # b
sculpt-app ... --level 7  --csg-preview intersect --csg-count 40 --csg-size 0.4   # c
sculpt-app ... --level 7  --csg-preview subtract  --csg-count 10000 --csg-size 0.6  # d
# add --csg-heat for the cost heat map, --csg-boxes for box cutters
```

`--test-strokes 0` = no synthetic sculpting; render ~25 frames, screenshot, exit.
`--csg-size` is instance size as a fraction of mean neighbour spacing (0.5 =
touching). `--csg-seed N` changes the placement.

### One-command benchmark

```sh
sculpt/scripts/csg_bench.sh            # your machine / real GPU
sculpt/scripts/csg_bench.sh --xvfb     # headless software adapter
# knobs: SIZE=1280x800 LEVELS="7 10" COUNTS="100 10000 100000"
```

It prints one `CSG_BENCH ... gpu_wall_ms median=` line per
(level, mode, count): prepasses + resolve + a `device.poll(wait)`, so it is
real submit-to-idle time per frame, not the CPU encode time shown in the HUD.

## Measurements (software adapter, NOT representative of a GPU)

Cloud VM, 4 cores, lavapipe (CPU Vulkan), 1000x700 window, 4x MSAA with
sample-rate resolve, median of 20 frames:

| base mesh | instances | subtract | add (boxes) | intersect |
|---|---|---|---|---|
| 98k | 1 (fixed cost) | 250 ms | | |
| 98k | 100 | 489 ms | 677 ms | 326 ms |
| 98k | 10,000 | 474 ms (408 in a second run) | 604 ms | 245 ms |
| 98k | 100,000 | 488 ms | 771 ms | 297 ms |
| 6.3M | 10,000 | 13,900 ms | not run | not run |

Run-to-run noise is about +-20%. What these numbers do say, and what they do not:

- Cost is flat in instance count from 100 to 100,000 (the grid and early outs
  work: about 6 to 15 instances per visited cell, usually one cell per sample).
  That is a property of the algorithm and does carry over to a GPU.
- Absolute times are lavapipe shading about 2.8M samples (1000x700 x4) on 4
  CPU cores; they say nothing about a real GPU. Expect it to be far
  cheaper there, but **this spike did not measure that**; run the script.
- At 6.3M faces the frame is 14 s because the two prepasses push 12.6M
  triangles through a CPU rasteriser. That is mesh-density cost of the
  prepasses (the normal viewport pays the same for one pass), not CSG cost. On
  a GPU 6.3M triangles x2 is a few ms to tens of ms, still the dominant term
  and the one to attack (see recommendations).

## Per-pixel cost analysis

Per sample: 2 depth + 2 normal texel loads, 2 unprojects, one ray-AABB test of
the grid, then per visited cell: one `cells[]` load plus `count` x (`Inst` load
+ analytic test: about 15 flops for a sphere, about 40 for a box incl.
quaternion rotates). Loop bounds are fixed (400 cells, 8 settle passes).

- Samples not near any instance (grid miss or only empty cells) cost the
  prepass loads plus about 1 to 3 empty-cell reads.
- Cost per sample is O(cells walked x instances per cell), independent of N.
  Instances per cell ~ (cell volume x density): with cell about 2 radii it is
  about 4 to 15 in the tests above. Dense unions (heavily overlapping cutters)
  add settle passes; capped at 8 so worst-case is bounded.
- Walk length: subtract stops at the first cell containing the answer; add
  stops when a cell begins past the best hit; intersect likewise. Rays that
  graze many empty cells cost most (the `MAX_CELLS` cap clamps them).
- Memory traffic: 100k instances = 4.8 MB, cells (128^3 x 8 B) up to 16 MB,
  indices about 3 MB. Fits the default 128 MB storage binding limit; uses 3
  storage buffers of the default 8.
- Sample-rate shading multiplies the resolve by 4. Mitigation, not done:
  shade at pixel rate, mark edge pixels (samples disagree in kind or cutter id
  from the cheap tests) in stencil, and run the sample-rate pass only on
  those. Edge pixels should be a small fraction of the frame (not measured).

## What works

- Subtract with arbitrary overlapping cutters (exact union via interval
  chaining), correct inner walls, crisp equal edges at both mesh densities.
- Add and intersect, including boxes with arbitrary rotation.
- Exact MSAA edges: the CSG boundary is resolved per sample, so there is no
  aliasing of the intersection curve and no need for a sample-mask hack.
- Instance count scaling (grid + early outs), and a heat-map debug view.
- Zero change to the mesh, sculpt hot path, uploads or default rendering.

## What fails or is approximate (measured or by construction)

1. **Non-convex base meshes (not implemented).** Inside/outside is a single
   interval `[F, B]` per sample. A hollow shell (outer + cavity surfaces), a
   torus seen through its hole, or a concave bumpy silhouette has several
   intervals per ray. A cutter wall found between two intervals would be shown
   floating in empty space, and a cut that opens a cavity will not show the
   cavity's inner surface. Fix is depth peeling: N layers of (F, B) pairs
   (2 extra prepasses per layer, same resolve loop per interval pair), or a
   per-pixel linked list. Not built here; cost is linear in layers.
2. **Thin shells.** Same root cause. Also depth precision: F and B from
   `Depth32Float` give about 6e-6 x distance of error, so a cutter wall
   thinner than that vs. a shell thinner than that flickers. Storing ray
   distance in an `R32Float` target instead would fix precision (needs 4x
   MSAA float32 render support, not universal).
3. **Open or non-manifold meshes.** No back face means no `B`; the code
   treats the solid as infinitely thick. Cuts then show walls to infinity.
4. **Edge position is limited by the mesh, not by the cutter.** The cut line
   is where the analytic cutter meets the *mesh surface at F*. At low density
   that surface has chord error, so the line is crisp but sits up to the chord
   error off the true sphere/cutter intersection. At 98k faces on a unit
   sphere the chord sag is on the order of 2e-5 of the radius (estimate, not measured), invisible here; on a
   coarse base mesh it is visible, and it is the reason this is a preview of
   the result on *this* mesh, not of the result on a re-meshed one.
5. **Normals** come from interpolated vertex normals at the mesh surface; wall
   normals are analytic and exact. Flat or hard-edged meshes shade smooth at
   the cut border.
6. **Silhouettes.** Mesh silhouette edges are MSAA-rasterised as usual (4
   samples). Cutter-carved silhouettes are exact per sample. Cutters that bite
   the silhouette from outside the mesh's screen footprint are fine for
   subtract (nothing to cut) but for add the primitive is drawn only through
   the grid walk, so anything outside the grid AABB is clipped (grid covers all
   instances by construction).
7. **No depth output.** The resolve does not write depth, so the existing
   cursor ring, overlays, and any later draw cannot depth-test against cut or
   added surfaces. Writing `frag_depth` from the hit `t` is a small change;
   sculpt brushes, however, raycast the CPU mesh and would hit material that
   the preview hides (cuts) or miss primitives that it shows (add).
8. **Camera inside a primitive** (add) or inside a cutter is not handled
   (entries `t < 0` are ignored).
9. **Overlay/mask colouring** from the regular fragment shader is not
   carried through; the resolve shades plain clay.
10. **Prepass cost scales with mesh density**: 2 extra full-mesh passes (the
    normal view draws one). At 6.3M faces this is the bottleneck.
11. **Aniso/large instances** (box with 50:1 aspect, or one instance covering
    the whole mesh) list in many cells; cost per cell grows, bounded by the
    cap but the grid should be sized from the *median*, not mean, radius.

## Recommendation

- **Role: interactive preview only, not the commit path.** It gives immediate,
  pixel-exact, density-independent feedback while the user drags parameters
  (the CPU grid rebuilds in about 1 ms at 10k, 70 ms at 100k, uploads are a
  few MB), which is exactly the tuning loop. It cannot output geometry, cannot
  feed sculpt brushes (CPU raycasts), and is exact only for ray-convex bases.
- **Gate it by base-mesh class.** Use it when the base is closed, and either
  ray-convex or peeled with 2 to 3 layers; otherwise fall back to the dimmed
  non-boolean mesh + instance gizmos. Detect non-convex rays cheaply: a heat
  flag when the number of front/back layers per pixel (a third peel prepass)
  exceeds the supported count.
- **Commit to real geometry** with a separate, resolution-aware step on
  "Apply": evaluate the same analytic SDF set (CPU, the same `Inst` data) and
  either (a) dual-contour/marching-cubes a narrow band around the cut curves
  on the existing multires level, projecting vertices onto the exact analytic
  surface (the preview already shows precisely which cells these are: the
  samples where `h.kind == 2` borders `h.kind == 1`), or (b) subdivide-and-
  snap the faces the cutter boundary crosses until the edge error is below a
  pixel at the working zoom. The preview doubles as the acceptance test: the
  committed mesh should render identically under `--csg-preview` off vs on
  (diff the screenshots, as done above for 98k/6.3M).
- **Next steps if this is adopted:** (1) write depth from the hit so overlays
  and tool cursors work; (2) edge-pixel-only sample-rate shading to cut the
  resolve about 3-4x; (3) a second peel layer for shells; (4) measure on a
  real GPU with `scripts/csg_bench.sh`; (5) consider rendering F/B once per
  camera change and caching them across parameter drags, since cutter edits
  do not change the base mesh, which makes slider drags cost only the resolve.

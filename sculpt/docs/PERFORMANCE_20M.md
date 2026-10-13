# 20-30 million triangles on a mid-range GPU

Goal: sculpt and orbit a 20M-triangle mesh smoothly on a laptop RTX 3070 (target) and an RTX 2070
(minimum, "more sluggish" is fine), without relying on a large GPU.

## Why a big mesh is slow, and what that implies

1. **Editing is not the problem.** Brush dabs run on the CPU, one rayon task per BVH leaf, and upload only
   the dirty leaves. At 6.3M faces a large dab is 7-11 ms and a small one about 1 ms
   (`ARCHITECTURE.md` §6). That already scales with brush size, not mesh size.
2. **Drawing is the problem.** A 1080p viewport has 2M pixels. 20M triangles is ~10 triangles per pixel,
   and a sphere seen from the front shows half of them. Hardware rasterizers shade pixels in 2x2 quads,
   so a one-pixel triangle costs about four pixel shades, and 4x MSAA multiplies the triangle setup and
   resolve work again. Cost goes with triangle count, which is wasteful: nobody can see 10 triangles in a pixel.
3. **So draw about one triangle per pixel.** That bounds frame cost by screen size, independent of mesh
   size and largely independent of the GPU generation. This is also what ZBrush and virtualized-geometry
   renderers (Nanite) rely on; there is no need for ZBrush itself to learn the trick, and nothing here
   needs it installed.

## The design (implemented)

### Hierarchical LOD that mirrors the sculpt BVH (`sculpt-core/src/lod.rs`)

* The sculpt BVH already splits the mesh into leaves of ~2048 quads. Each BVH **inner node** gets a
  simplified copy of its two children's triangles, reduced to about 4096 triangles
  (`meshopt` edge-collapse simplification, `LockBorder | Sparse | ErrorAbsolute`).
* Every level only **references vertices of the full mesh**. When a brush moves a vertex, every level that
  uses it moves too and the existing dirty-leaf upload already covers it. Normals come from the full-resolution
  surface, so shading does not pop between levels.
* **No cracks by construction.** Each patch keeps its whole outer border, so two neighbouring patches at
  different levels share exactly the same border edges. The test suite checks this directly: for many random
  cuts through the tree every undirected edge is used by exactly two triangles.
* Each node stores a world-space error (monotone toward the root: `max(child errors) + own simplification error`).
* **Normals count as error** (`LodParams::normal_weight`, default 0.3). Shading comes from the full-resolution
  normals of the vertices a patch keeps, so a purely geometric simplifier happily keeps a vertex from a stroke's
  steep rim as the corner of a large flat triangle, smearing the rim's dark shading across it: spiky seams and a
  preview that looks off from the real mesh. With normals as simplification attributes, rims survive as narrow
  strips and patches whose shading would change report a larger error, so they are drawn finer. Measured by
  rendering the same clay stroke with and without LOD (4.3M triangles, zoomed in): 99th-percentile pixel
  difference 13 grey levels without, 2 with, for 12% more triangles; on a 20M mesh covered in fine noise it
  costs 1.5x the triangles, on smooth forms nothing. (`examples/lod_error.rs` compares stored and true error.)

### Per-frame selection (CPU, ~0.02-0.25 ms)

Walk the tree from the root. Skip nodes outside the frustum. Stop descending where
`error * focal_px / distance_to_box <= tau` (Display > Viewport detail: Draft 2 px, Balanced 1 px,
Sharp 0.5 px, or Full to draw every triangle as a reference; the tolerance doubles while orbiting). A triangle budget guard raises
`tau` if the cut would exceed it. Adjacent index ranges are merged, written to an indirect buffer, and drawn
with `multi_draw_indexed_indirect` (needs only the downlevel `INDIRECT_EXECUTION` flag, so it works on any
Vulkan, DX12 or Metal device that wgpu supports).

### Editing a LOD'd mesh

An edit marks the leaves it touched and all their ancestors **stale**. Selection ignores the error of a stale
node and descends through it, so the edited area is drawn at full detail immediately and is always correct.
Once the pen has rested for 250 ms, the UI thread copies a few stale patches (their indices and corner
positions and normals, ~0.5 ms) to a single-core worker thread that compacts and re-simplifies them; finished
patches are installed only if no edit touched them meanwhile (a per-node epoch), the GPU pool ranges are
rewritten and the cut coarsens again. The batch grows while the worker outpaces frames. (An earlier version
did this on the UI thread and stalled small strokes for up to 284 ms; the stroke test's `SCULPT_TEST_TAPS=1`
mode reproduces that pattern.) If the stale area is so large that the
budget guard trips, selection stops forcing detail first, then loosens `tau`.

### Building it

The tree is built as a background job whenever a mesh of 1.5M triangles or more is loaded or subdivided. The
index pool holds the full mesh plus about 1.7x again for the coarse levels; for 20M triangles that is ~640 MB
of `u32` indices (the app asks wgpu for the adapter's maximum buffer size instead of the 256 MiB default,
see `gpu_setup` in `main.rs`; if a device cannot allocate it the plain draw path is used).

## Measured here (no GPU: 4-core VM, software Vulkan)

| | |
|---|---|
| Mesh | 10.0M quads = 20.0M triangles (cube-sphere) |
| Document + BVH build | 4.8-9 s |
| LOD build (4 cores) | 6.3-9.3 s; pool 53.7M triangles |
| Selection | 0.004-0.25 ms |
| Triangles drawn, smooth sphere | 90k-160k at 1 px |
| Triangles drawn, fractal relief | 180k-360k (mild), 1.0-2.8M (extreme 20M-triangle relief) at 1 px; 0.56-1.4M at 2 px |

Software rasterization is not a GPU number, but the *ratio* is informative. Orbiting the 20M-triangle mesh at
1080p on the same CPU rasterizer, including waiting for the frame to finish:

| | per frame (median) | triangles drawn |
|---|---|---|
| Full mesh (`SCULPT_NO_LOD=1`) | 5,280 ms | 20.0M |
| LOD cut, tau 1 px | 92 ms | 160k |

That is a 57x reduction for a smooth sphere. Real GPU numbers need your machine (see below).

## Validators

1. `cargo test -p sculpt-core --test lod`: watertight cuts, monotone errors, budget honoured, frustum culling,
   stale/refresh correctness.
2. `sculpt-cli lod-bench 10000000 [relief]` (10M quads = 20M triangles; relief 0.04 adds fine noise): build
   time, pool size, cut size at several distances. `SCULPT_LOD_NORMAL_WEIGHT` overrides the normal weight.
3. **On your GPU:** `sculpt-app --quads 1291 --bench-orbit 120 --size 1920x1080` prints median/p95/max for the
   whole frame, CPU encode+submit **including waiting for the GPU to finish** (so it is true GPU frame time),
   triangles drawn and selection time. Run it again with `SCULPT_NO_LOD=1` for the full-mesh baseline.
   `SCULPT_LOD_TAU=2` loosens the tolerance; `SCULPT_LOD_LOG=1` logs the cut every frame.
   Passing means: LOD run well under 16 ms on the 3070 Laptop and tolerable on the 2070.

## What is *not* done, in order of value

1. **Compute software rasterizer for sub-pixel clusters** (the Nanite / CuRast approach). The hardware path
   stops being efficient when clusters are tiny; a visibility-buffer compute pass using 64-bit atomics
   (wgpu `SHADER_INT64_ATOMIC_MIN_MAX`, Vulkan 1.2) removes the 2x2 quad cost completely. Needed only if the
   hardware path is still bound by small triangles at 1 px on the 2070.
2. **Pool memory**: 640 MB is fine on 8 GB cards, but 16-bit local indices per patch or a smaller
   `node_tris` would halve it.
3. **Hi-Z occlusion culling** for scenes with heavy self-occlusion (tight fists, interior cavities).
4. **Async partial rebuild** of the error values; today a heavy edit leaves coarse parents using their
   original error until refreshed (the stale mechanism hides this visually).
5. **Streaming build** for the 30M end (BVH + LOD in about 20 s on this 4-core box).

## References

* Brian Karis, Rune Stubbe, Graham Wihlidal, "A Deep Dive into Nanite Virtualized Geometry", SIGGRAPH 2021
  (cluster hierarchy, error-bounded cuts, software raster of micro-triangles).
* Cignoni et al., "Batched Multi Triangulation", IEEE Visualization 2005 (patch hierarchies with locked borders).
* Garland and Heckbert, "Surface Simplification Using Quadric Error Metrics", SIGGRAPH 1997.
* Arseny Kapoulkine, meshoptimizer (`simplify`, `Sparse`, `LockBorder`).
* Fatahalian et al., "Data-Parallel Rasterization of Micropolygons with Defocus and Motion Blur", HPG 2009
  (quad overhead of the hardware rasterizer on tiny primitives).

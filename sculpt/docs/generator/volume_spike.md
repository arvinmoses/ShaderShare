# Spike: volume sculpting (can a volume brush keep up?)

Code: `crates/sculpt-core/src/generator/volume.rs` (spike, not wired into the app or the document), benchmark
`sculpt-cli volume-spike [millions] [--images DIR]`. Measured on this machine (24 threads, 64 GB), CPU only.

## Question

Sculpting on a mesh stretches it; 3D-Coat's voxel mode never does, because the brush changes a volume and the surface
is rebuilt from it. Before committing to a volume brush engine: can a dab change a sparse distance field and rebuild only
the surface it touched in under about 10 ms, at the 10-20 million triangles the work typically uses?

## What the spike does

* The lattice is cut into blocks of 8x8x8 samples. Blocks near the surface store their samples (distances clamped to
  2.5 lattice steps); all others store only inside or outside.
* A clay dab is a smooth union of a soft ball into the field, a carve a smooth subtraction (blend width a fifth of the
  radius). Unions and subtractions stay correct on a clamped field; a displacement would not.
* Only blocks the ball's surface passes through are changed sample by sample; blocks wholly inside it become solid or
  empty; only blocks whose samples actually changed, and the ones below them, are re-extracted.
* Extraction is surface nets per block, each block reading its 9x9x9 neighbourhood with one lookup per neighbour.

## Results

Detail levels are a ball of radius 1 at the lattice spacing that gives about that many triangles. Strokes of 24 dabs,
brushes of radius 0.03, 0.1 and 0.25 (an eighth of the model's width).

| triangles | blocks stored | memory 32-bit / 16-bit | build ms | full extract ms |
|---|---|---|---|---|
| 1M | 18,287 | 38 / 19 MB | 8 | 21 |
| 5M | 92,271 | 193 / 99 MB | 27 | 101 |
| 20M | 368,519 | 786 / 409 MB | 119 | 381 |

| triangles | brush | dab | total ms, median | total ms, worst | upload, most |
|---|---|---|---|---|---|
| 1M | small (6 voxels) | clay / carve | 0.1 / 0.2 | 0.4 / 0.4 | 52 KB |
| 1M | medium (20) | clay / carve | 0.4 / 0.5 | 0.5 / 0.7 | 283 KB |
| 1M | large (50) | clay / carve | 1.2 / 1.5 | 1.5 / 2.8 | 1.8 MB |
| 5M | small (13) | clay / carve | 0.3 / 0.4 | 0.6 / 0.6 | 141 KB |
| 5M | medium (45) | clay / carve | 1.0 / 1.6 | 1.5 / 2.8 | 1.2 MB |
| 5M | large (112) | clay / carve | 6.1 / 8.5 | 10.1 / 23.4 | 8.4 MB |
| 20M | small (27) | clay / carve | 0.6 / 0.7 | 0.9 / 1.3 | 468 KB |
| 20M | medium (89) | clay / carve | 3.1 / 5.8 | 6.0 / 12.1 | 4.6 MB |
| 20M | large (223) | clay / carve | 30.1 / 51.9 | 54.1 / 137.7 | 33 MB |

For comparison, today's mesh brushes cost 0.8-1.7 ms a frame at 0.1-1.6M faces.

Images (`docs/generator/volume/`): two balls bridged with clay merge into one surface; a horn pulled far out of one
grows instead of stretching. The surface after it all: 642k quads, longest edge 1.34 lattice steps, no non-manifold
edges.

Two changes made the difference during the spike: gathering each block's neighbourhood once instead of one lookup
per sample (large brush at 1M: 37 to 3 ms), and rebuilding only blocks whose samples changed with a realistic blend
width (large brush at 20M: 150 to 30 ms).

## Verdict

**Go, for small and medium brushes at every size measured**: under 10 ms median everywhere, worst 12 ms (medium carve
at 20M). That is a working sculpting engine that cannot stretch, merges shapes that touch, and shares one field with the
procedural stack.

**Not yet for very large brushes on very dense models**: a brush an eighth of the model's width at 20M takes 30-52 ms
(worst 138 ms for carving), and would upload up to 33 MB. Two ways to close it, both in line with ideas already on the
table:

1. Large brushes work at a coarser level: a big dab only changes broad form, so it can change a coarser copy of the
   field and the fine level follows (the "object holds the main shape, detail tessellates in" idea).
2. Move the field update and extraction to the GPU: both are independent per block, which is what compute shaders do
   best; this also removes the upload, since the mesh would be made where it is drawn.

## Not measured

* GPU upload time (only bytes), and drawing per-block meshes in the viewport.
* Move, grab and smooth brushes, which have to warp or filter the field rather than add to it.
* Manifold pieces in extraction (`remesh.rs` adds a small, fixed cost per cell) and loose-piece cleanup.
* Layers, masks and undo in a volume; adaptive (ADF) storage for resolution independence; saving.

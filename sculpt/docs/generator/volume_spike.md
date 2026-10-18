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

# Spike 2: a coarse level for large brushes

Code: `Volume::refine_from` and friends in `volume.rs`; benchmark `sculpt-cli volume-spike levels [millions] [--images DIR]`.

A large brush works on a coarse copy of the field (4x the lattice step, 64x fewer samples, true distances kept out to
0.2 so nothing a stroke moves is clamped) while the stroke is under way, and the viewport shows that coarse surface.
When the stroke ends the fine field is rebuilt once: the new coarse shape plus the old fine detail, carried along the
surface normal by however far the coarse surface moved. The stroke is the large brush of the main table (radius 0.25,
24 dabs, blend 0.05). "Fine bumps" puts noise a dozen lattice steps across and 1.5 high on the ball, which the coarse
level cannot hold.

| triangles | surface | coarse dab ms (median / worst) | direct fine dab ms (median / worst) | refine once after the stroke ms | detail kept on the lump |
|---|---|---|---|---|---|
| 1M | fine bumps | 0.5 / 1.0 | 1.4 / 1.9 | 24 | 90% (direct: 50%) |
| 5M | smooth / bumps | 1.1-1.3 / 1.7 | 5.5 / 11 | 117-139 | 85% (direct: 28%) |
| 20M | smooth / bumps | 4.5-5.5 / 7.5-8.8 | 35 / 56-75 | 752-807 | 71% (direct: 0%) |

**Accuracy against the exact answer** (the same dabs composed with true distances everywhere, extracted at fine detail;
distance from each result's surface to the exact one, in fine lattice steps, mean / p95):

| triangles | coarse then refined | direct on the fine field |
|---|---|---|
| 5M | 0.03 / 0.18 | 2.36 / 7.31 |
| 20M | 0.02 / 0.07 | 7.02 / 19.76 |

The coarse path is not an approximation that costs accuracy: it is the accurate one. The fine field keeps distances only
2.5 lattice steps out, but a soft blend reaches much further (45 steps at 20M), so a dab blended straight into the fine
field mixes in clamped values and misshapes the fillet. The coarse level keeps true distances that far, so its blend
is right, and the fine detail then rides on it.

Images (`docs/generator/volume/levels_*.png`, 1M): the coarse surface shown during the stroke; the refined result with
the bumps carried up onto the new lump; the direct stroke, which buries them. The refined lump's flanks show faint
banding where detail was carried a long way at a slant.

## Verdict for large brushes

**Go.** A brush an eighth of the model's width at 20M costs 4.5-5.5 ms a dab on the coarse level (worst 8.8), against
35 ms on the fine field, and comes out closer to the exact shape. The one-off refinement after a stroke takes about
0.8 s at 20M (0.13 s at 5M), which would run in the background with nearest blocks first, but should come down further.

What this means for the engine: a small pyramid of levels, each brush working on the level whose distance range covers
its blend width, the finest level holding the detail. That is also the zoom-dependent detail idea: far away, the
viewport can show a coarser level.

Still open: GPU compute (spike 3), the banding on steep flanks, refining progressively near the camera first, and
whether detail should ride (as here) or be buried (as clay does) for each brush type.

# The volume engine: several resolutions, surface on the GPU

Code: `sculpt-core/src/generator/level_volume.rs` (the field, brushes, history, files, the reference extractor),
`sculpt-app/src/gpu_levels/` (the GPU copy, extraction, culling), `sculpt-app/src/volume_mode.rs` (the playground),
`sculpt-app/src/volume_demo.rs` (the scripted review session). Tests: `sculpt-core/tests/level_volume.rs`,
`sculpt-app --gpu-levels-check` (GPU against the reference), `sculpt-app --gpu-levels-check --bench`.
Video: `volume/volume_engine_review.mp4` (`sculpt-app --demo-volume DIR`, frames from the app's own framebuffer).

This is the engine spike 4 (`volume_spike.md`) chose the layout for.

## What it is

* **Levels.** Level 0 holds the whole form; each level after it halves the lattice step. Every level is cut into 8x8x8
  blocks keyed by (level, block coordinates), nested (a block's parent always exists), and a level holds blocks only
  where something was sculpted at it. A new ball has six levels; the finest is 32 times finer than level 0.
* **Each level holds its own work.** A brush works at the level its size on screen calls for: a lattice step about a
  pixel at the brush, but no finer than a 48th of its radius. A big brush works on coarse form, a small one zoomed in
  carves detail as fine as the screen shows.
* **Detail rides coarse strokes.** After a dab, every finer level's own detail (its values less the level below) moves
  with the surface, along the normal by as far as the stroke moved it. It is taken from the surface as it was when the
  stroke began and moved by the stroke's whole change, so it is resampled once per stroke however many dabs there are
  (resampling it every dab blurred a groove away to 8% over one long stroke; now 93% survives). Because only the
  stroke's total change matters, the work can also be done late: under a big brush over detailed areas the finer blocks
  are queued and rebuilt a few milliseconds' worth per frame, catching up within a few frames and fully when the pen
  lifts. Detail is taken only from blocks that held it when the stroke began, blocks it has moved out of are
  dropped (so moving detail leaves no trail of empty blocks), and blocks it was moved into are carried again with
  every dab (detail moved far can sit outside a dab's own reach). How far the surface moved is measured along the
  normal from the new surface back to the old one: the samples' change says the same only while the move stays within
  the band. Level 0 keeps distances out to 16 of its steps, wide enough for a big stroke's move at 1M triangles.
* **Form built at a fine level moves down.** When a stroke ends, wherever a finer level's surface lies further from
  the level below than small detail would (40% of its band), the level below moves toward it, smoothed over about one
  of its own steps. Narrow detail, however tall, stays where it was sculpted; broad form built up zoomed in also shows
  from afar (half a ridge's height with the finest level left out, against 6% before), and a later big stroke treats
  it as form. Without this a level's detail would outgrow its band and later coarse strokes would wreck it.
* **Distances again after a stroke.** Plane brushes move samples part of the way to a shape, which leaves a shallow
  field inside clay (about -0.005 throughout a ridge 0.04 tall): right in sign, wrong as a distance, and everything
  above reads distances. The touched blocks become distances to the surface again when the stroke ends; corners of the
  cells the surface crosses keep their values, so the surface and its shading do not move.
* **Plane brushes, checkerboard cleanup, speck removal, undo and `.svol`** as in the single-resolution playground, per
  level. Saving is version 2 (levels); version 1 files open as level 0.

## The surface, on the GPU

The CPU edits the field; the GPU holds every stored block's 9x9x9 samples (its own and the first layer past it, at
16 bits) and builds the surface there, block by block, into the buffers the viewport draws from. A hash table on the
GPU finds the slot of a (level, block). After a dab the CPU sends only the blocks that changed and those whose first
layer past them reads them, and the GPU rebuilds the blocks around them.

The rules (the same in the CPU's reference extractor, which the GPU is checked against exactly):

* A lattice point takes the value of the finest level holding a block there, so cells of different levels agree on
  every crossing and the surface closes without cracks.
* A cell has a vertex per surface piece (crossing edges linked across its faces, checkerboard faces resolved the same
  way on both sides). A cell next to a finer one has exactly one vertex, at the crossings on its faces split once.
* A crossing lattice edge gives a quad when none of the four cells around it is finer; one of them owns it. Quad
  corners are named by (slot, piece, cell) in the index buffer and resolved to a vertex when drawn, so rebuilding a
  block can never leave a neighbour's quads pointing at the wrong vertex.

Each block's surface is a cluster: its quads in pages of 64 (the index buffer), its bounds and its normal cone. Every
frame a cull pass writes the pages of the clusters in view and facing the eye as indirect draws, and the viewport
draws them in one call (with an indirect count where the device has it).

## Results

AMD Radeon AI PRO R9700, Vulkan.

**Correctness.** `--gpu-levels-check` sculpts strokes at all four levels of a test ball (clay and trim), undoes,
redoes and shows every level cap, and after each step compares the GPU's surface with the reference block by block:
every cell word (vertex numbering, pieces, corner signs) and every quad, corner by corner, in order, are identical;
vertices agree within 0.008 of a lattice step (16-bit samples). With `SCULPT_CHECK_EVERY_DAB=1` it compares after
every dab of every stroke, with detail carried 0.5 ms at a time so queued blocks arrive late, as in the app. It passed
on DX12 earlier in this work; DX12's shader compiler now fails on the vertex pass (see the gaps).

**Core tests** (`tests/level_volume.rs`): a fresh ball and plane brushes at every level give no open edges, no
non-manifold edges and no dropped quads at any level cap; rebuilding only the blocks a stroke touched matches a full
rebuild exactly; a groove keeps 93% of its depth under level 0 clay that raises the area 0.18, and a level 4 groove
102% under a rise of 116 of its steps, without spawning blocks (47 before, 35 after); bumps and pits at levels 3-5
keep 71-86% of their height under two level 0 clay strokes raising the area 0.064 (20 to 100 of their steps),
carried a little at a time, and 64% under six such strokes raising it 0.26 (200 steps); a ridge built at level 2 shows half its height from afar; a crumb left by a stroke is
removed and a real lump kept; undo, redo and saving restore exactly. `sculpt-cli volume-spike levels-carry DIR` draws
such bumps before and after.

**Speed** (`--gpu-levels-check --bench`): a ball whose level 0 alone is about 1M triangles (92k blocks with the
16-step band), five finer levels down to a step 32 times finer. Per dab of a stroke, medians:

| level | step | brush radius (steps) | dab, CPU | sync, CPU | until the GPU is done | blocks rebuilt (most) | pen lift |
|---|---|---|---|---|---|---|---|
| 0 | 0.0035 | 71 | 4.0 ms | 5.9 ms | 10.0 ms | 2798 | 77 ms |
| 1 | 0.0018 | 46 | 1.5 ms | 1.2 ms | 3.7 ms | 570 | 44 ms |
| 2 | 0.0009 | 46 | 1.5 ms | 1.4 ms | 4.2 ms | 607 | 50 ms |
| 3 | 0.0004 | 34 | 1.2 ms | 1.0 ms | 4.1 ms | 362 | 30 ms |
| 4 | 0.0002 | 37 | 1.4 ms | 1.1 ms | 4.6 ms | 589 | 37 ms |
| 5 | 0.0001 | 37 | 1.1 ms | 1.0 ms | 5.2 ms | 617 | 31 ms |

A level 0 stroke of radius 0.3 right over the fine work is the slow case: 72 ms a dab on the CPU (the dab's own
work; carrying is held to 6 ms of it) and 29 ms of syncing, up to 236 finer blocks carried a dab; the queued detail
then catches up over about 100 frames (1.8 s) and the pen lift takes 0.38 s. Nearly all of it is rebuilding the
carried blocks themselves (`carry_block`, about 2 ms per block across the cores); measuring how far the surface moved
is now about a tenth of it.

The first upload and full extraction of the 92k blocks takes 0.43 s (134 MB sent). After the strokes the samples
take 102 MB; one
lattice at the finest step everywhere would need about 60 GB. In the app the readout shows the GPU's own time for each
rebuild (timestamps), the blocks sent and rebuilt, and the pages drawn of those in use: on a 1M-triangle ball about
half are skipped as facing away.

## Found while building it

* Raycasts (the brush, the cursor) marched by the field of the coarsest level holding a point, which knows nothing of
  finer detail just ahead, and stepped over small bumps; they now allow for how far finer levels can raise the surface.
* A slot freed and reused in the same sync was released twice at once, so two blocks shared pages; caught by
  comparing the GPU with the reference after every dab (`SCULPT_CHECK_EVERY_DAB=1`).
* Where detail lands was worked out pushing forward along the normal at the source, and where each sample's detail
  comes from looking back along the normal at the sample; once a big stroke tilted the surface they disagreed by more
  than a block, the detail landed nowhere, and the emptied blocks were dropped (levels 4 and 5 vanished under a big
  stroke in the first recording). The landing point is now found by marching from the old surface, along its
  normal, to the new one, once per source block.
* Under a big level 0 stroke on a fine level 0 (1M triangles, a band of 0.04) the surface moved 0.065: the samples'
  change saturated at the band, and deep in the fresh clay its normal was zero, so detail stayed where it was and was
  buried; then blocks the detail had moved into kept an early dab's values (square platforms). Measuring the move,
  taking the old surface's normal at the source, carrying moved blocks again each dab and a wider level 0 band fixed
  it (the grooves ride along; some small steps remain).
* A cell next to a finer one, deep in fresh clay, with no crossing on its faces fell back to stepping onto the zero
  along the gradient; with every corner at the band the gradient was rounding, and the GPU and the reference put the
  vertex 0.9 steps apart. Both now keep the cell's centre unless the gradient is clear and the zero lies in the cell.
* Carrying detail sized its source grid by the stroke's whole movement in fine steps: under a big level 0 stroke that
  was hundreds of steps and gigabytes; it is now sized by where the samples actually come from.

## Driver workarounds (Vulkan, this AMD driver)

Two things in the compute shaders lost the device or silently gave wrong results on Vulkan while DX12 was fine:
arrays in a thread's own memory indexed by a variable (constant tables included), and several workgroup variables,
which the driver laid over each other (the atomics' initial values turned up in the block's slot number). The shaders
now work corners and edges out from bits, carry a cell's corner values as two vec4s, and keep all workgroup state in
one array of atomics at fixed places. The exact check runs on both backends.

## Known gaps

* **Level choice is per stroke, display cap is global.** The view shows every level, or all levels up to a cap set in
  the Mesh menu. Choosing the level per region by distance and by how much detail it holds (the error per block is
  stored for it), and a vertex budget, are next.
* **No occlusion culling yet**, only view and facing.
* **Big strokes over much fine detail are slow**: 70-100 ms a frame (a radius of 0.3 over a patch with detail at
  levels 2-5), and the carried detail then takes about 2 s to catch up. Lifting the pen after ordinary strokes costs
  30-80 ms (distances again, form moved down, speck search, all on the CPU). Rebuilding carried blocks should move to
  the GPU or a worker thread that does not hold up the frame.
* **DX12 fails to compile the vertex pass** (its FXC compiler aborts with an internal error). The app picks Vulkan on
  this machine; on DX12-only machines the volume mode would not start. Trying the newer DXC compiler, or splitting the
  transition-cell code out of the vertex pass, is next.
* **Small steps where carried detail meets the edge of a finer region**, and the finest dots come through faint.
* **Detail drifts sideways where clay builds a slope.** It moves along the surface's normal, so where a scrubbed
  stroke leaves the area sloping, small bumps end up off where they were, a few thousandths after a rise of 0.06 and
  up to 0.03 after 0.26 (they keep most of their height). Carrying along the brushes' own push direction instead
  would hold them in place under plane brushes.
* **The field is edited on the CPU.** Dabs cost 1-3 ms; the GPU does what grows with the model (extraction, drawing,
  culling). Moving dabs to compute shaders is possible later without changing the layout.

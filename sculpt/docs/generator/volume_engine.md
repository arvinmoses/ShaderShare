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
every dab of every stroke, with detail carried 0.5 ms at a time so queued blocks arrive late, then all over again
carrying in jobs that land a dab late, as the app's do. The reference reads every sample as the GPU's 16 bits hold
it, so both build the surface from the same numbers. It passed
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
| 0 | 0.0035 | 71 | 2.9 ms | 2.9 ms | 4.1 ms | 2381 | 77 ms |
| 1 | 0.0018 | 46 | 1.1 ms | 1.0 ms | 3.9 ms | 523 | 36 ms |
| 2 | 0.0009 | 46 | 1.2 ms | 1.1 ms | 4.2 ms | 539 | 41 ms |
| 3 | 0.0004 | 34 | 0.9 ms | 0.8 ms | 3.4 ms | 312 | 25 ms |
| 4 | 0.0002 | 37 | 1.0 ms | 0.9 ms | 3.0 ms | 347 | 31 ms |
| 5 | 0.0001 | 37 | 0.9 ms | 0.9 ms | 4.6 ms | 323 | 25 ms |

A level 0 stroke of radius 0.3 right over the fine work, timed frame by frame as the app runs it (two dabs a frame,
the app's spacing apart): 25 ms a frame on the main thread (dabs 13.9, sync 13.0, bringing in and starting carrying
jobs 0.1). The carrying runs on a quarter of the cores, off the frame. When the pen lifts, frames go on at 0.2 ms
each until the detail is all in (0.9 s, about 50 frames). Then the stroke ends as a job too (distances again, form
moved down, specks: 0.32 s on the shared threads, about 16 frames), and taking its result costs the main thread
1.5 ms, plus one frame's sync of about 25 ms for the blocks it changed.

Ending a stroke in a job works on a copy of the volume while nothing else may change it: the next stroke, an undo
or a save waits for it, and its result is then the copy's maps, taken as they are. A test checks that a stroke ended
this way is byte for byte the stroke ended at once, and undoes the same.

The dabs now cost about what the same dabs over bare clay do: a radius of 0.3 is 85 level 0 steps and changes about
2400 blocks a dab.

That case was 100 ms a frame and 1.8 s of catching up. What it took:

* **Carrying on another thread.** A job takes queued blocks (a level's only once the coarser ones queued are all
  taken, since carrying reads the level below as carried) with a copy of the volume. The worker carries them on
  that copy while the next dabs change the original, and the frame after it finishes brings the result in. A block
  whose ground a later dab changes is queued again, so a result that comes in late is replaced. The copy costs
  almost nothing: stored blocks are shared, and each level's map is split into 1024 shards by place, each copied
  only when written while a job still holds it. Keeping a block for undo is now sharing it, not copying it.

* Every point whose move a carried block reads is measured once per batch. Some were missed and measured again by
  every block reading them: three quarters of the carrying time.
* The move is measured every 4 lattice steps of the stroke's level and interpolated (it is smooth over a brush), and
  after a dab only the measurements and carried blocks within two blocks of what it changed are redone.
* The finer blocks under a dab are found from the finer levels' side (thousands of blocks) rather than by walking
  down from the 2000-odd coarse blocks a big dab touches.
* Before the worker, the carrying budget was spent once a frame rather than once per dab.
* The volume's maps hash block coordinates the fast way, not with the default hash, built to resist crafted keys.
* Syncing: a block whose 16-bit samples come out as the GPU already has them is neither sent nor rebuilt (about
  half under a big stroke). Block coordinates go only to slots given a new block, in runs of consecutive slots,
  not one small write each. The sets of blocks to send and rebuild are found in parallel.

The first upload and full extraction of the 92k blocks takes 0.24 s (134 MB sent). After the strokes the samples
take 102 MB; one lattice at the finest step everywhere would need about 60 GB. In the app the readout shows the GPU's own time for each
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
* Interpolated between full-precision samples, a value can land on the other side of zero from the same value
  interpolated between the GPU's 16-bit ones (0.0000008 against one 16-bit step of 0.000015): a cell then has one
  crossing more on one side, and its vertex moves 0.15 steps. Carrying in jobs happened to make such a cell. The
  reference now reads every sample as the GPU holds it.
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
* **Starting a stroke right after a big one waits** for the last one's end (up to about 0.3 s after a big stroke
  over detail, if the pen comes down at once).
* **Very big brushes on a fine level 0 cost about 7 ms a dab** (2400 blocks changed); a frame with several such dabs
  runs at 30-40 fps, detail or not.
* Carrying in jobs leaves a few empty in-between blocks that carrying all at once drops (up to 10 a level in the
  tests); they hold nothing and cost little.
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

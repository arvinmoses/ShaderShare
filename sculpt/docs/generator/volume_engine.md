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
* **A big stroke buries the fine detail under it, unless asked to keep it.** By default a coarse stroke works on the
  finer levels' blocks under it as it does on its own level, so it fills grooves and covers bumps wherever it builds
  above them, and leaves what sticks out of it: as clay does on one surface (ZBrush's top subdivision level, or a
  DynaMesh). With "Big brushes keep finer detail" (Mesh menu, `keep_detail`) the detail is carried along on the new
  surface instead, as sculpting at a lower subdivision level does; that is the carrying described here, and it still
  softens the finest work under a big rise (see the gaps). Detail that should ride on changing form (moving, posing,
  generators) belongs in layers: a detail layer kept as heights over the form beneath it, composed geometrically,
  rides along without the softening. `volume/bury_or_keep_detail.png`: before, buried (0.07 of clay over the patch),
  kept.
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
keep 94-107% of their height under two level 0 clay strokes raising the area 0.064 (20 to 100 of their steps),
carried a little at a time, and 92% under six such strokes raising it 0.26 (200 steps), in place as seen along the
brush; a level 4 groove on the sloping rim of a big stroke keeps its whole depth along its length (it used to break
into pieces cut off at block faces); a patch of mixed work (level 1 rolls, level 2-3 strokes, level 4 grooves and
dots) under a big level 0 stroke keeps its rolls, grooves and dots without new roughness; a ridge built at level 2 shows half its height from afar; a crumb left by a stroke is
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

## Carrying detail under big strokes, fixed

`volume/carry_big_stroke_compare.png` is one view after the demo's big stroke: top left as it was (grooves in stepped
pieces), top right a wrong turn on the way (rolls exaggerated, pebbly), bottom left as it is now, bottom right the
same with linear resampling. What was wrong, and what changed:

* **Where detail lands and where it is read back disagreed.** Which blocks to rebuild was worked out per source block
  along the stroke-start normal, the rebuild read detail back along the new normal; where a stroke tilts the surface
  (its rim) the blocks the detail needed were never built, so grooves were cut off at block faces. Both now follow
  the same direction, point by point.
* **That direction is the brush's push**, not the surface's normal: each dab's plane normal, weighted by the brush's
  falloff, blended between blocks. Along the normal, detail slid sideways wherever the stroke left a slope (0.027
  after six strokes; now 0.004). Near walls of hard brushes, where the push runs along the surface, the normal.
* **How far the surface moved is measured more finely where it varies**: on every 4th level 0 step, and where a cell's
  centre is off what its corners say by more than half a fine step, at every point of the cell. Form at the levels
  between (rolls rebased into level 0) made the move vary faster than the coarse lattice, so detail was read a few
  steps off and vanished.
* **Detail is taken and put back the same way**: the level as stored less the level below clamped to the level's band,
  added to the new level below clamped the same way, as reading a level from the one below does. Before, it was put
  back onto the unclamped level below, so samples just above a bump came out short and bumps shrank 8-10% a stroke.
  (Leaving both unclamped exaggerated everything a level stood off the one below: the wrong turn above.)
* **Detail is resampled with a cubic** (Catmull-Rom over 4x4x4) rather than linearly, which lost 3-15% of a feature a
  few steps wide each stroke.

* **A lattice of dips under a big stroke** (roughness 60 times what it was) came from level 0, not level 1: moving
  form down a level (`rebase`) placed level 0's surface right but left the field around it compressed (a slope of
  0.5-0.9 instead of 1), and the distances were made again before it, not after. Level 1's detail against level 0
  then sloped along the push, and a source a step or two off read it off by 0.001-0.002. The blocks `rebase` changes
  are now made distances afterwards, and every sample (those next to the surface too, which kept the brush's blend
  values) is set to the distance to its nearest crossing's tangent plane. Roughness under the stroke: 0.00058 to
  0.0001.

* **Fine work on a big brush's falloff was flattened or swallowed**: mid-stroke the stroke's own level is a blend,
  not a distance (its slope there is the dab's weight, about 0.3-0.7), so levels composed onto it by value landed at
  detail divided by that slope: form 1.6 times too high, a plateau that swallowed grooves. The level right above the
  stroke's now reads it as a distance marched along the carrying direction. `rebase` also moved coarse blocks only
  under touched fine blocks (a lattice a block across); it now moves them wherever the fine level has a block.
  A test replays the review session's patch dab for dab (`the_review_sessions_patch_rides_its_big_stroke`).

## A detail layer over the form

The way to keep fine work under big strokes without softening it: a layer stack, as sculpting a lower subdivision
level does in ZBrush, but with a mask on every layer. `sculpt_core::generator::detail_layer` (`Layered`); in the app,
the Layers panel shows the volume's stack while the playground is open. `volume/detail_layer_compare.png`: the demo's
patch before the big stroke, under it as one volume (buried), and under it with the grooves and dots in a layer.

* **The layer holds heights, not field values.** At each lattice point of the finest level near the form's surface:
  how far the detail stands out of the form at the nearest point of its surface (the same along the normal), and a
  mask from 0 to 1. Heights come from roots of the fields (where they change sign along the form's normal), so
  neither the band clamp nor a field that is not a distance mid-stroke enters them.
* **Making one**: "Move detail to a layer" takes the work at a chosen level and finer out of the volume into the layer
  (the form keeps the coarser levels). The bumps of the test come through as sculpted (0.00402 against 0.00402).
* **Sculpting the form carries it along.** Each new surface point takes the height of the point it came from: from a
  lattice point, along its normal to the new surface, then along the stroke's push back to the surface the stroke
  began on, and out from there along that surface's normal as far as the point was out of the new one. That last
  part matters on the rim of a big stroke, where the surface tilts (15 to 45 degrees in the test): offsetting along
  the new normal instead slid bumps 0.001 uphill and changed their shape by 45%. Heights are read from the layer as
  the stroke began, never dab after dab. Mid-stroke the move is measured at the blocks' corners and the heights read
  linearly, a preview; when the stroke ends, every 4 steps and with a cubic. A search that finds no surface leaves
  its point out (the corners that found one say), rather than reading as no move: that tore pieces out in the app.
* **What is drawn is the form read inward by the masked height**: the form's field at the point moved back along its
  normal by the height. That puts the surface exactly the height out whatever the field's slope (subtracting the
  height from the field would scale it by one over the slope, which strokes leave off one until they end).
* **Results** (`tests/detail_layer.rs`): two big level 0 clay strokes (the bury test's) over two rows of bumps. Under
  the strokes the form rose 0.068; the bumps went from 0.00399 to 0.00398 out of it, each bump's shape about its
  centre within 0.00033, centres moved 0.00038 (along the stroke's push). On the falloff, a 45 degree rim: 0.00382
  to 0.00381, shape within 0.00039, moved 0.00021. The surface stays closed. The same with the app's six levels and
  the layer taken from a level with no work of its own (`bumps_ride_a_layer_taken_from_a_level_coarser_than_the_work`).
* **Speed: following runs off the frame.** A dab on the form shows at once where no layer is under it, and queues
  what is; a job on a quarter of the cores (the carrying threads) follows the queue with the form as it was when the
  job began, and the frame takes its result in, the form there with the layer on it. When the pen lifts, the form's
  end and then the stroke's last follow (exact, over all the stroke touched and every layer block it changed) run on
  all the cores. At the demo's scale (a brush of radius 0.5 over a layer of 2400 blocks, a ball of 1M triangles):
  10.6 ms a frame with the layer against 11.4 without, pen up 0.2 ms, the layer caught up 0.7 s after
  (`big_form_strokes_under_a_layer_stay_cheap_a_frame`). Following in jobs ends exactly as following at once does
  (`following_in_jobs_ends_as_following_at_once`). On the way there: a block whose corners all lie farther from the
  new surface than a height reaches is dropped before any more is measured; a search with no direction (a flat field
  deep inside new clay) gives up at once; a block with none of the layer where its heights come from is skipped; a
  block whose move changed less than a quarter step since it was last followed is left as it is. Mid-stroke the
  layer is a preview that trails the brush: on a short stroke it may only show once the pen is up.
* **Painting on the layer**: a stroke with the layer selected works on what is shown, and when it ends the layer takes
  it in as heights (the form keeps none of it) and undoes it with its own history.
* **Its mask** hides it where painted ("Mask: hide" and "Mask: show" in the Layers panel), and rides the form as the
  heights do.
* **Undo and redo** run over the stack: a form stroke brings back the form and the layer as they were; a layer stroke
  or mask stroke the layer alone.

* **A stack of layers, each at its own resolution.** Layers are displacements along the form's normal and add up
  (their order does not change what is shown), each with its own mask, level and visibility. A level shows the
  layers at it or coarser, so a distant view that leaves fine levels out leaves fine layers out too. Following runs
  per layer on its own lattice, so a coarse layer costs little; heights may stand out as far as the layers' first
  level allows (6 of its steps), so changing a layer's resolution keeps what it holds (a broad bump 0.0120 tall:
  0.01196 moved a level finer, 0.01191 back). Painting goes into the layer selected, less what the others add there
  and divided by its mask. The Layers panel lists the stack top first (show, select, resolution, remove; new layer
  at a chosen resolution), and `.svol` files keep the layers after the form (a file without them opens as one
  volume). Tests: `layers_of_two_resolutions_ride_together_and_save` (a broad bump on a level 3 layer and small
  ones on a level 4 layer over it ride two big strokes, 0.0120 to 0.0119 and shapes within 0.0006; saved and opened
  within 0.0000045), `layers_save_and_open_with_the_volume`.

Not done yet: merging the layers back resets undo; layers per region of the form (a layer is the whole volume's);
the layers' following and composing on the GPU.

## The level by distance

Each region of the view shows only the levels whose detail would show on screen from where the eye is ("Level by
distance" in the Mesh menu, on by default, with the pixels it takes):

* **The measure.** A block above level 0 holds detail over the level below; how far its surface lies from that
  level's is stored per block, and per block the most its own detail and that of any block under it add up to. A
  block shows while that comes to more than a pixel (by default) at its distance from the eye, and goes once it falls
  under half that, so a slowly moving camera does not make blocks flicker. Level 0 always shows.
* **Hidden blocks are as if not stored.** What is hidden nests (a hidden block's finer blocks go with it), so the
  surface's rules, which only ask which blocks are there, close it as they close the stored surface. The one thing a
  hidden block changes for its shown neighbours is the first layer of samples past them: read as interpolated from
  the coarser levels, as it is where nothing is stored, so both sides of a face agree on it (read from the hidden
  block's own samples, the far view opened 11 edges in the test). On the GPU a hidden block gives up its slot, and
  showing it again uploads it, as a block that came or went does; the shaders are unchanged.
* **Measured after it changes.** A changed block's detail is measured again a few hundred blocks a frame, in parallel,
  and not while the stroke under way keeps changing it; until then it counts as unbounded and shows, so the block
  being sculpted always does.
* **Cost.** At about 1M triangles at level 0 with five finer levels (the bench): choosing takes 0.4-1 ms when the
  camera moves and nothing when it does not; a dab's sync costs 1.3-1.8 ms against 1.1-1.4 ms with every level shown;
  moving the camera from 2.6 to 1.15 away shows 1,100 blocks again in one 9 ms sync. From 8 away, 4,700 of the 5,000
  finer blocks are left to coarser levels.
* **Seen.** `volume/level_by_distance.png` (from the review, tinted by level): close in, the patch's levels; pulled
  back, none of the finer levels drawn (9.8k of 18.6k pages); off, every level drawn from afar too (14.0k of 22.8k
  pages); close in again, all there.
* **Tests.** `tests/level_view.rs`: near work shows and far work falls back (from afar the view is level 0's exactly),
  the surface closes in every view, a block being sculpted shows until measured, and rebuilding only around the
  blocks shown or hidden (with strokes between camera moves) matches rebuilding everything. The GPU check
  (`--gpu-levels-check`) compares six views from near and far, with strokes between, against the reference exactly.

Not done yet: a triangle budget (raising the pixels when a view would draw too much), blending between levels as
they hand over (a block appears or goes at once, about a pixel's change), and choosing by occlusion.

## Driver workarounds (Vulkan, this AMD driver)

Two things in the compute shaders lost the device or silently gave wrong results on Vulkan while DX12 was fine:
arrays in a thread's own memory indexed by a variable (constant tables included), and several workgroup variables,
which the driver laid over each other (the atomics' initial values turned up in the block's slot number). The shaders
now work corners and edges out from bits, carry a cell's corner values as two vec4s, and keep all workgroup state in
one array of atomics at fixed places. The exact check runs on both backends.

## Known gaps

* **Level choice by distance has no triangle budget yet** (see the level by distance): a view of a lot of near
  detail draws all of it.
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
* **The finest work on a slope fades at one end under a big stroke.** In the demo the lower two grooves and the dots
  come through faint: the finest detail shears sideways by about 0.004 over 0.012 there. Not yet diagnosed.
* **Kept under a big stroke ("Big brushes keep finer detail"), fine work comes through soft** (the detail layer above
  is the fix; this option stays for now) (`volume/carry_texture_compare.png`: before the stroke, the
  build before the last fixes, now): the grooves all survive but faint, with a fine grain and faint block seams, and
  the demo's dots are lost. The cause is the design of carrying: each finer level is carried as a difference of field
  values against the level below, and those fields are not distances while a stroke lasts, nor read at the same
  sample phase level to level. Carrying each level instead as its surface's height over the level below's surface,
  marched, and composing geometrically would remove it (see "Next" below).
* **Carrying under a big stroke catches up in about 2 s** after the pen lifts (the level above the stroke's is read as
  a marched distance), frames going on meanwhile.
* Bumps taller than 2.5 of their level's bands read 3-5% taller after two strokes.
* **The field is edited on the CPU.** Dabs cost 1-3 ms; the GPU does what grows with the model (extraction, drawing,
  culling). Moving dabs to compute shaders is possible later without changing the layout.

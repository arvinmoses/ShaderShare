# Local benchmark results (real GPU)

Machine: Windows 11 Home (build 26200), 24 CPU threads, rustc 1.97.0.
GPU: AMD Radeon AI PRO R9700 (discrete, Vulkan, AMD proprietary driver 25.20.42.14 / LLPC; a second R9700 and an
integrated Radeon are present, wgpu picked the first discrete one). Timestamp queries available.
Branch `ccr-32ffc09b-o5mvh1` at `d45439e`.

## Build and tests

- `cargo build --release --workspace`: clean, 1m33s cold.
- `cargo test --release --workspace --no-fail-fast`: all green after one test fix. Two timing tests in
  `sculpt-core/tests/schedule.rs` failed at first. Cause: with no cost estimate yet, the scheduler's first batch packs
  one slice per rayon thread before it checks the deadline, so with 24 threads (the cloud box had 4) one batch covered
  every visible item and `deadline_hit` was false. A test-environment assumption, not a scheduler bug. The two tests
  now run on a 2-thread pool.

## Spike 1: GPU displacement boolean vs CPU (`--gpu-bench 7,8,9,10 1000,10000,100000 --cpu`, Subtract)

"slider total" = bake + upload + GPU time + readback path; "wall" = submit to fence.

| faces | instances | GPU wall ms | slider total ms | CPU ms | CPU / GPU |
|---|---|---|---|---|---|
| 98k | 1k / 10k / 100k | 0.18 / 0.29 / 1.78 | 0.50 / 0.96 / 4.49 | 1.4 / 4.0 / 14.8 | 2.8x / 4.2x / 3.3x |
| 393k | 1k / 10k / 100k | 0.26 / 0.30 / 1.47 | 0.64 / 1.02 / 3.71 | 2.0 / 5.5 / 18.6 | 3.1x / 5.4x / 5.0x |
| 1.57M | 1k / 10k / 100k | 0.26 / 0.33 / 1.08 | 0.65 / 0.90 / 3.49 | 2.6 / 7.5 / 31.2 | 4.0x / 8.4x / 9.0x |
| 6.29M | 1k / 10k / 100k | 0.57 / 0.90 / 1.67 | 0.88 / 1.58 / 4.19 | 4.3 / 12.4 / 52.0 | 4.9x / 7.9x / 12.4x |

GPU buffers: 28 MB (98k) to 244 MB (6.3M). One-time mesh upload 103 ms at 6.3M. This is far better than the
3-35 ms the cloud estimate predicted at 6.3M. The CPU path is also fast on 24 threads (52 ms at 6.3M / 100k).

## CPU generator-bench (spike A/B, level 10, 6.3M faces)

Slider total (update + hash + boolean): 4.2 ms (1k), 12.6 ms (10k), 52.1 ms (100k). 1% edit: 0.95 / 3.4 / 30 ms.
Placement cold 11-17 ms, warm 0.2-6 ms. Edge detect + chains 26 ms cold; 100k edge points 1.3 ms.
Voxel (spike B) cold: 0.6-8.6 s (level 10 table, 1k-100k); not interactive, as before.

## Spike 4: progressive scheduler (`generator-drag-sim`, 6.3M faces, CPU)

| N | view | frame med / p95 / max ms | full recompute baseline med ms |
|---|---|---|---|
| 10k | zoom | 2.5-4.9 / 3.0-5.7 / <= 5.8 | 14-15 |
| 100k | overview | 5.5-6.6 (+seed 0.0) / 6.6-9.0 / <= 10.1 | 54-66 |
| 100k | zoom | 6.1-6.5 / 7.2-9.3 / <= 10.2 | 51-66 |

All 120 drag frames: 100k overview median 5.9 / p95 7.9 / max 10.1 ms. No frames over budget. Converges exactly.

## Spike 5: GPU sparse SDF (`--gpu-sdf-bench`, `--gpu-sdf-check`)

Correctness: all six checks PASS (subtract, union, intersect, smooth variants) vs CPU, 0 sign mismatches.
Level 10 warm frame eval: 1.8 ms (1k, 1e), 4.2 ms (1k, 0.5e), 12 ms (10k, 1e), 27 ms (10k, 0.5e), 23 ms (100k, 1e),
128 ms (100k, 0.5e). Cold: 136-165 ms (1k), 0.9-1.3 s (10k), 1.9-3.5 s (100k). Memory: GPU buffers up to ~2 GB,
cache up to 6.4 GB f32 (3.2 GB f16). Live-capable up to about 10k instances; heavy beyond.

## Spike 3: adaptive refinement (`refine-bench time`)

Full refine, 6.3M faces: 6 ms (1k), 78 ms (10k). 1% edit: 0.9 ms (1k), 7 ms (10k). Cheaper than the cloud numbers
(110-195 ms / 0.8-1.8 s), on 24 threads.

## Spike 6: ADF (`adf-bench perf`, level 10, 0.5 voxel)

1k: ~21 ms total. 10k: ~160 ms. 100k: ~580 ms (about 25-45x faster than the cloud's 14.7-26 s). About 1% of
quads have vertexless cells, as before.

## Not run

- `csg_bench.sh` (spike 2): the app has no `--csg-preview` flag, since the spike is parked and not wired into the
  current LOD viewport. Needs the port first.
- `adf-bench quality` / `refine-bench quality|cracks` (quality, not speed).

## Sculpt latency guard (after milestone 1)

`sculpt-app --level N --test-strokes 180` on this machine, discrete R9700, after adding the generator layer hook to
`Document::recomposite` (no baseline run on this machine before the change, so compare with the cloud figures only):

| faces | input + dabs median / p95 / max ms | whole frame incl. UI median ms |
|---|---|---|
| 98k | 1.18 / 1.99 / 4.97 | 1.63 |
| 1.57M | 2.18 / 3.51 / 10.33 | 2.69 |

## Edit-to-update speed in the app (milestone 1, scripted sweep)

`scripts/record_generator_demo.ps1` runs `sculpt-app --demo-sweep` (count, size, lift, seed, primitive, operation and a
20k-count stress, one at a time) and records the window in real time. The readout is the generator Properties
section: time spent in the edit command (`set_layer_meta`, which rebuilds placement and recomposites the touched
leaves), measured on the CPU. Whole-frame time and fps come from the app.

| mesh | edit to surface, average / slowest | frame rate during the sweep |
|---|---|---|
| 393k faces | 6.4 / 35 ms | 60 fps |
| 6.29M faces | 106 / 382 ms (about 65 ms in the ordinary phases; the 20k-count stress phase is the slow end) | drops to about 7 fps while editing |

Why it is slow at 6.3M: every edit rebuilds the whole placement (`SurfaceSampler` CDF over all faces and base normals
over all vertices) even when only the size or lift changed, and only then evaluates the touched leaves. Milestone 4
(staged caches: resample only when count, seed or the mask change; rebuild instances only for primitive, size, lift;
evaluate progressively) is what brings this back to the interactive budget.

## Torn dents while editing at 6.3M faces: cause and fix

Symptom (first 6.3M-face recording): during parameter sweeps the dents were ragged, with shard-like rims, while the same
scene was clean at 393k faces and clean once everything settled.

Cause: `LodTree::select` fell back to drawing *stale* simplified patches when refining the stale areas would exceed the
triangle budget (6M triangles in the app; the 6.3M-face mesh is 12.6M). A patch is a list of indices into the shared vertex
buffer whose connectivity was chosen for the old surface, so drawn over vertices that have since moved it tears the new
detail. Sculpt strokes touch few leaves, so the fallback never triggered before. A generator edit touches nearly all of them.

Fix: stale patches are never drawn. Stale areas always refine to current nodes, and the budget guard only coarsens areas
that are up to date, so after a whole-mesh edit the cut may exceed the budget until the background refresh catches up.
Regression test: `a_whole_mesh_edit_never_draws_out_of_date_lod_patches`. Evidence: `lod_stale_fix_before_after.png`
(the app's own frames at five sweep moments, before and after, same scene).

After the fix the whole-frame CPU time is about 3 ms and the window runs at about 56 fps at 6.3M faces during the sweep.
The edit itself still takes about 60-75 ms there (placement rebuilt on every edit), which milestone 4 addresses.

## Edit speed after staging the caches (milestone 2 speed pass)

`sculpt-cli generator-edit-bench 8,10 60`: median of 9 coalesced edits through `set_layer_meta`, alternating two values
(count 60/90, seed 7/8, size 0.10/0.12, lift 0.6/0.5, sphere/box, subtract/add).

| faces | stage | create | count | seed | size | lift | primitive | operation |
|---|---|---|---|---|---|---|---|---|
| 393k | before | 6.9 | 7.2 | 6.6 | 6.6 | 6.8 | 6.7 | 6.6 |
| 393k | after | 5.6 | 1.9 | 1.8 | 1.7 | 1.7 | 2.0 | 2.1 |
| 6.29M | before | 71.5 | 92.0 | 79.4 | 73.6 | 65.1 | 85.1 | 74.4 |
| 6.29M | after | 33.7 | 24.5 | 19.6 | 16.6 | 15.5 | 23.3 | 19.3 |

What changed, in order of effect (the profile of one edit at 6.3M faces was: cache rebuild 0.5 ms, evaluating the
cutters 5 ms, then 41-45 ms in `geometry_changed`):
1. Normals and dirty-marking only for leaves whose positions actually changed (`recomposite` compares before and after),
   instead of every leaf the cutters' bounding boxes reach. This also shrinks the GPU upload and the LOD staleness.
2. The per-leaf normal update writes each leaf's own vertices in place in parallel; only the few vertices a leaf borrows
   from outside the touched set are patched serially.
3. Placement is staged: the base normals and per-face sampling table are kept by the document (rebuilt only when the base or
   topology changes), the scattered points are reused when count and seed did not change, and only the cutter transforms and
   index are rebuilt per edit.
Guards: `normals_match_a_full_recompute_after_generator_edits`, plus the existing equivalence tests. Sculpt latency is no
worse (input + dabs median 1.09 ms at 98k faces, 1.66 ms at 1.57M).

In the app at 6.3M faces (scripted sweep readout): edit to surface averages 16-22 ms. The whole frame is 41-64 ms (11-19 fps)
during the heaviest phases (count in the thousands, 20k stress) because stale level-of-detail areas now draw at full detail
until the refresh catches up, and 2.7-2.9 ms (50-59 fps) in the lighter ones. A preview that does not touch mesh data during a
drag, or a faster LOD refresh, is the way to remove that. Not yet done.

## Flat shading with 8-byte vertices and lazy normals

Viewport vertices are now one 8-byte record (position as three 16-bit values inside the model's box, plus the overlay
scalar) instead of 28 bytes (position 12, normal 12, overlay 4). In flat shading (the default) there is no normals buffer:
each triangle's normal comes from the screen derivatives of its position, and the CPU skips recomputing normals after
generator, layer and undo edits (`Document::set_lazy_normals`; strokes, bakes, LOD refresh and `begin_stroke` call
`ensure_normals` first). Smooth shading is still available from Display > Shading and uses the old normals buffer.
A 16-bit step is about 1/65000 of the model's box, about 46 times finer than a triangle edge at 6.3M faces; the CPU copy
of the surface stays full precision and the box refits (re-sending everything) if the model outgrows it.

`generator-edit-bench` (median ms), eager normals against `--lazy`:

| faces | eager (count / size / lift) | lazy (count / size / lift) |
|---|---|---|
| 393k | 2.0 / 1.8 / 1.7 | 1.7 / 1.3 / 1.2 |
| 1.57M | 8.3 / 6.8 / 6.1 | 5.2 / 4.0 / 4.0 |
| 6.29M | 22.5 / 17.0 / 14.9 | 14.3 / 10.9 / 10.2 |

In the app at 6.3M faces (scripted sweep, same five moments, mode confirmed per run):

| | smooth | flat |
|---|---|---|
| edit to surface, average | 17-54 ms | 13-45 ms |
| whole frame in the heavy phases | 48.6 / 73.3 / 54.8 ms (14-20 fps) | 42.2 / 47.3 / 39.1 ms (17-24 fps) |
| whole frame in the light phases | 2.3-3.7 ms (45-59 fps) | 2.5-2.9 ms (50-59 fps) |

Sculpt strokes (level 9, 180 frames): input + dabs 1.68 ms median in both modes, GPU upload per frame 1981 KB (smooth)
against 792 KB (flat).

Look (`flat_vs_smooth.png`): at 393k faces flat is close to smooth, with slightly stair-stepped rims; at 98k faces seen
close up flat is plainly faceted (a visible grid in the dents). At 6.3M faces the two cannot be told apart at normal
distance. Rocks and rubble should suit hard normals; low-density meshes seen close up will want Smooth.

Test-setup trap found on the way: the app saves its settings on exit, so a run with a shading override once wrote
`flat_shading: false` into the real settings file and silently turned later "flat" runs smooth. `SCULPT_SHADING=flat|smooth`
now overrides for one run and is never saved (the app prints `shading override: ...` so a run can be confirmed).

## Where heavy-phase frame time goes at 6.3M faces (per-frame log, flat shading, scripted sweep)

`SCULPT_LOD_LOG=1` now also prints edit, upload and frame time. In the heavy phases the cost is the edit itself, not the
level-of-detail draw (the earlier guess was wrong): select 2 ms, encode 2 ms, upload about 5 ms, a UI-thread hand-off to the
LOD refresh of about 14 ms in some frames, against edits of 13-15 ms (median) and 128-323 ms in the 20,000-cutter stress.
The LOD never catches up during a continuous drag (about 4000 of 8000 patches stay stale, about 12.5M triangles drawn), which
costs little on this GPU.

Edit time against cutter count (`generator-edit-bench <levels> <count> <size> --lazy`, medians, ms):

| cutters (size) | 1.57M faces | 6.29M faces |
|---|---|---|
| 60 (0.10) | 4-5 | 10-17 |
| 1000 (0.04) | 8-11 | 27-37 |
| 5000 (0.025) | 11-17 | 37-56 |
| 20000 (0.015) | 19-30 | 57-101 |

The edit time grows with the number of vertices the cutters reach, so a cut at full density is the cost to remove: evaluate
what the view shows first (the progressive scheduler from the spikes, `generator/schedule.rs`) and finish the rest at rest.

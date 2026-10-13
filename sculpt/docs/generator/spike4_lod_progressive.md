# Generator Boolean spike 4: LOD-aware evaluation and progressive refinement

Follows `spike_results.md` (spike 3) and `DESIGN.md`. Question: can the CPU displacement boolean (spike A) stay inside an interactive frame (16 ms) while a slider is dragged, even at 100k instances on a 6.3M-face mesh, where spike 3 measured 278 ms?

## TL;DR

* **Yes for the frame budget, no for a fully live 100k display on this machine.** An engine-level scheduler (`generator/schedule.rs`) evaluates only the vertices of the LOD nodes in the current view cut, in priority order, under a deadline, and carries unfinished work to later frames. Measured over 120 frames of a size / count / seed drag at 98k, 1.6M and 6.3M faces with 1k, 10k and 100k instances (two cameras, 18 cases): **median frame 3 to 11 ms, p95 under 16.7 ms in 15 of 18 cases (worst 18.2 ms), max 9 to 38 ms** on a noisy shared 4-core box. The unscheduled full recompute of the same drag takes 8 to 641 ms per frame (median), so the scheduled frame is 2 to 6x cheaper at 1k instances, 3 to 18x at 10k and 21 to 59x at 100k.
* **What is on screen during the drag depends on density.** Share of the visible cut's vertices that are current at the end of a frame (median): 65 to 100% at 1k instances, 32 to 99% at 10k, 4 to 63% at 100k (the rest shows its last complete result). At 100k instances the cost per vertex, not scheduling, is the limit.
* **Converged result equals the full-quality evaluation exactly** (bit for bit, checked after every phase of every case and in tests), including after a released drag is interrupted by a new one. After release the visible cut is current in 1 to 12 frames up to 10k instances and 26 to 58 frames at 100k; the background full-quality job lands 17 ms to 1.3 s after release.
* **Cancellation never blocks input:** starting a new drag while the background job runs took 0.3 to 10.7 ms in the drill (wait for at most one 4-leaf chunk); the cancelled job's output is discarded.
* **Incremental spatial hash:** changing 1% of 100k instances costs 0.07 to 0.17 ms in the hash step versus a 72 to 128 ms full rebuild of the slack-registered grid on this box (the spike-3 grid took 15 to 42 ms on a quieter machine); a global size slider (every instance changes) costs 2 to 4 ms per frame at 100k including baking (`rescale_all`; `map_all` with an arbitrary function: 2 to 8 ms), because instances that stay inside their slack-registered cells need no hash work.
* **Recommendation:** adopt the scheduler as the Generator layer's live evaluator with an 8 to 10 ms pass budget. It makes up to 10k instances interactive at every mesh size tested (at 6.3M faces and 10k instances about 30 to 40% of the visible vertices are current during a size or count drag in the overview camera, converging 5 to 12 frames after release) and gives bounded-time progressive display at 100k. To make 100k feel live, add instance subsampling during the drag (see Recommendation). Details and caveats below; read the caveat about measurement noise first.

## What was built

| Piece | Where |
|---|---|
| Scheduler: view-aware cut, time-budgeted progressive passes, background full-quality job, cancellation, async instance-set swap | `crates/sculpt-core/src/generator/schedule.rs` |
| Incremental spatial hash: slack-registered cells plus an overlay, `update_incremental`, `map_all`, `rescale_all`, instance deactivation | `crates/sculpt-core/src/generator/instances.rs` |
| `Cut::ids` (node ids of the drawn nodes), `View::sees` and `View::error_px` made public | `crates/sculpt-core/src/lod.rs` (3 lines) |
| Deterministic drag simulator | `crates/sculpt-cli/src/generator_drag_sim.rs`, subcommand `generator-drag-sim` |
| Tests (10, all pass) | `crates/sculpt-core/tests/schedule.rs` |

Nothing on the sculpt brush path was touched. The scheduler is not wired into `Document` or the app yet (spike code, like spikes 1 to 3); the section "Calling it from the app" says how.

## How it works

### 1. View-aware evaluation at the LOD's vertex density

The unit of work is a **cut node**: a node of the hierarchical LOD tree that the renderer would draw this frame. The LOD tree mirrors the BVH node for node and its simplified patches only reference vertices that exist in the mesh's vertex buffer, so the vertices a cut node needs are exactly `unique(indices of that node)`. The scheduler caches that list per node (`node_verts`, built once, about 0.55 s and 70 MB at 6.3M faces) and evaluates only those vertices. A coarse node costs a fraction of the leaves under it, and the delta it writes is the one the renderer reads, because it is the same vertex buffer.

`Scheduler::select_cut(view, mode)` is the LOD tree's `select` with four changes:

* **Frustum and backface culling.** Subtrees outside the frustum, or whose normal cone faces away from the eye, are skipped (cones are built bottom-up from vertex normals at construction).
* **Generator-aware refinement.** A node that holds instances keeps refining until its triangle edge, in pixels, is at most `max(min_spacing_px, instance_px / verts_per_instance)` (defaults 2 px and 3 vertices per instance). The LOD tree's own error test knows nothing about displacement, so on a smooth sphere it would happily draw 4096 triangles for a model covered in cutters; the spike-3 failure mode "a cutter with fewer than about 3 vertices across aliases" would then show up at every LOD change.
* **Untouched nodes stay coarse.** A node whose bounds touch no instance cell is not refined for the generator and, when stale, is zero-filled without evaluating anything.
* **Drag bias.** While dragging, pixel tolerances are multiplied by `drag_detail_scale` (3), so the cut is about 2.5x coarser; `Mode::Settle` uses the full detail. Nodes whose box projects below `min_node_px` (3 px) or whose instances (mean extent) project below `min_instance_px` (0.75 px) are not scheduled while dragging. The renderer must draw the cut the scheduler evaluates, so use `select_cut` for the draw call instead of `LodTree::select` while a generator is active.

### 2. Progressive refinement under a frame budget

`Scheduler::frame(view, deadline, mode)` does one pass:

1. Land a finished instance-set build (swap) and a finished background job, if any.
2. Select the cut. For each cut node decide stale or fresh. Freshness is **per vertex**: every vertex records the epoch it was last evaluated at (`vert_epoch`), and a node is stale while one of its vertices is older than the epoch that last dirtied the node. Dirtiness comes from the change: `apply_changes` marks only the leaves under the old and new instance boxes (and their ancestors), `map_all`, `rescale_all`, `set_params` and a swapped instance set mark everything.
3. Order the stale nodes by `score = projected_size_px * (1 + min(instance_px, 32) / 8) * (1 + age / 6)`: on screen (the cut only contains visible nodes), large and near first, big instances first, with an age term. Every fourth frame the order is by age alone, so a visible node waits at most about `4 * visible_nodes` frames even if every frame changes every parameter (test `no_visible_node_starves_under_constant_change`).
4. Evaluate in batches until the deadline. A batch is packed from a cost model (per-node CPU seconds per vertex, an EMA, and a wall/CPU ratio that tracks imbalance and contention) so that its predicted wall time fits the time left. A node that does not fit is **split**: only the vertices that fit are taken, from a rotating start in the node's list, so no vertex of a big node starves. At least one small slice always runs, so a frame always makes progress. Overshoot is bounded by one batch (about 1 ms in the measurements) plus model error.
5. Everything not reached keeps its last complete result in the displacement buffer (`Scheduler::out()`), and carries over to the next frame with its age.

`FrameReport` gives the numbers a UI needs: fresh share by nodes, leaves and vertices, vertices evaluated, culled and skipped counts, the longest stale age, whether the deadline stopped the pass. `Scheduler::updated_nodes()` lists the nodes written by the last frame (upload `node_vertices(id)` for each).

### 3. Incremental instance spatial hash

The grid registers every instance under the cells of its box inflated by a per-instance random share (0.5 to 1.5) of `slack` (default 0.15) of its extent. Candidates are a superset by construction, and `eval_vertex` already tests the exact inflated box, so extra registration cannot change a result. An instance whose new box still fits its registered cells costs no hash work at all. The rest (an instance that grew or moved past its slack) is re-registered in a small overlay map; the overlay is consulted after the static table, and when too many instances escape (more than an eighth of the set, or the overlay outgrows a third of the table) the grid is rebuilt in parallel instead. `rescale_all(ratio)` (a global size slider), `map_all(f)` and `update_incremental(changes, deactivate)` are the entry points; deactivated instances keep their slot, so a count slider is "activate or deactivate a prefix" of a shuffled placement and never rebuilds anything. Hash maintenance numbers are in the results.

### 4. Release, the background job, and cancellation

`release()` marks the drag finished. The full-quality job (every vertex, `evaluate_leaves` in chunks of 4 leaves, into its own buffer) is started by the first `frame` after the visible cut is fresh, so the foreground catches up first and the two do not fight for cores. Any change (`apply_changes`, `map_all`, `rescale_all`, `set_params`, a swapped instance set, `mark_leaves`) cancels it: the cancel flag is checked between chunks, the caller waits only until the job lets go of the instance set (one chunk, measured at 0.3 to 10.7 ms in the cancellation drill), and the thread frees its buffer on its own. A cancelled job's output is never read. When a job finishes for the current epoch, its buffer replaces `out()` wholesale and every vertex is stamped current. A new instance set built by `request_replace` (seed change, new distribution) is built on a small worker pool and swapped in when ready; if the drag was already released, the job restarts on the new set.

### 5. Why the converged result equals the full evaluation exactly

`eval_vertex(p, n, set, params)` is a pure function of the vertex and the instance set, so a vertex's displacement does not depend on which LOD node, batch, thread or frame evaluated it. The progressive path writes exactly `eval_vertex(p) - p`, the same expression `evaluate_leaves` uses, and zero for untouched nodes, which is also what `evaluate_leaves` writes. The job's buffer is produced by `evaluate_leaves` itself. The simulator checks `out() == evaluate(...)` bit for bit after every phase (column Exact), as do the tests. The incremental hash is also checked against a freshly built set (test `incremental_hash_matches_a_fresh_build`); `rescale_all` matches a rebuilt set to 1e-4 (the scale is applied as a ratio, so the last bits differ).

## Calling it from the app

Once per document or topology change:

```rust
let mesh = Arc::new(EvalMesh::from_document(&doc));          // snapshot: positions, normals, BVH
let mut generator = Scheduler::new(mesh, doc.lod().unwrap().clone(), set, params, SchedConfig::default());
```

Every UI frame, after input and before drawing (`frame_start` is the time at the top of the frame):

```rust
// 1. Feed this frame's parameter changes. Each of these cancels a running full-quality job.
if size_slider.changed()  { generator.rescale_all(new_size / last_size); }
if count_slider.changed() { generator.apply_changes(&activate, &deactivate); }  // prefix of a shuffled placement
if seed_slider.changed()  { generator.request_replace(move || build_instance_set(seed)); } // worker thread
if op_or_reach_changed    { generator.set_params(params); }
if drag_ended             { generator.release(); }

// 2. Run the budgeted pass while dragging, and keep calling it until everything is current.
let mode = if slider_is_down { Mode::Drag } else { Mode::Settle };
if slider_is_down || !generator.is_fully_converged() {
    let budget = Duration::from_micros(16_667) - estimated_render_and_ui_cost;   // 8 to 10 ms is typical
    let report = generator.frame(&view, frame_start + budget, mode);
    if report.job_adopted { upload_all(generator.out()); }
    else { for &n in generator.updated_nodes() { upload(generator.node_vertices(n), generator.out()); } }
}

// 3. Draw the nodes the generator evaluated (cut.ranges index the LOD pool), with positions + out().
let cut = generator.select_cut(&view, mode);
```

Notes for the real integration:

* `frame` is cheap when nothing is stale (cut selection plus a freshness scan of the visible nodes, 0.05 to 0.8 ms in the runs), so calling it every frame is fine; stop when `is_fully_converged()`.
* Sculpting underneath a live generator: replace the snapshot (`replace_mesh`) and call `mark_leaves` with the touched leaves (the same list the LOD refresh uses). The normal cones are built from the snapshot at construction and are not refreshed; after large edits rebuild the scheduler.
* Vertex normals of the displaced surface are not recomputed by the scheduler (neither were they in spike 3).
* The scheduler snapshot is a copy of positions, normals and the BVH (about 12 + 12 bytes per vertex plus the BVH), plus 4 bytes per vertex of epoch stamps, about 70 MB of per-node vertex lists and 75 MB for the job's buffer at 6.3M faces. The document and LOD tree are shared, not copied.
* Keep the budget below the frame period by the cost of everything else in the frame; the pass respects the deadline it is given, not a fixed 16 ms.

## Commands

```
cd sculpt
cargo build --release --workspace
cargo test --release -p sculpt-core --test schedule          # scheduler tests
cargo clippy --release --workspace --all-targets             # 0 new warnings
./target/release/sculpt-cli generator-drag-sim 7,9,10 1000,10000,100000   # full matrix, about 12 minutes
./target/release/sculpt-cli generator-drag-sim 9 10000 --view zoom --trace # one case, prints slow frames
./target/release/sculpt-cli generator-drag-sim 10 100000 --hash-only       # only the hash maintenance numbers
```

`generator-drag-sim [levels] [counts] [--view overview|zoom|both] [--frames 120] [--budget-ms 10] [--baseline-stride 4] [--slack 0.15] [--no-baseline] [--no-hash] [--hash-only] [--trace]`. Levels are quad-sphere subdivision levels as in `generator-bench` (7 = 98k, 9 = 1.57M, 10 = 6.29M faces).

**What the replay does.** 120 frames in three 40-frame phases, paced at 60 Hz so background work gets real time: **size** (all instances grow x1.0 to x1.3, `rescale_all`), **count** (active instances fall to 50% and come back, `apply_changes` with deactivation), **seed** (a new placement every frame, `request_replace`, newest wins). After each phase the drag is released and idle frames run until the visible cut is current and the background job has landed; the buffer is then compared with `evaluate()`. After the last phase a cancellation drill releases, waits for the job to be running, starts a new drag, and checks that the final result is exact. Two cameras: `overview` (whole sphere at distance 4) and `zoom` (distance 2, about a third of the surface). The unscheduled baseline repeats the same parameters on every fourth frame with a full recompute (rebuild the instance set, evaluate every vertex). Instances are uniform spheres of 3 to 10 mesh edges, Subtract, the spike-3 setup.

## Failure modes and caveats

* **Measurement noise.** 4-core container shared with other sessions: `uptime` showed load averages between 5 and 19 during development and about 6 during the final runs. The spike-3 numbers were taken on a quieter machine (a re-run of `generator-bench` for 98k faces and 10k instances gave 20 ms for the boolean here against 12 ms in `spike_results.md`). Treat absolute milliseconds as pessimistic by up to 1.5 to 2x and tails (p95, max) as noisy; ratios between the scheduled and unscheduled columns are from the same run and are more reliable. One run per configuration.
* **100k instances is a progressive display, not a live one, on this machine.** Frame time is bounded, but only 5 to 30% of the visible vertices are current during a size or count drag at 1.6M and 6.3M faces (up to 50 to 80% for seed drags, which update in bursts), and it takes 20 to 65 frames after release to finish the visible cut. The stale parts show the last complete result. The cost per vertex at 100k instances is the problem (every vertex sits in dozens of instance boxes), not scheduling.
* **Seed and distribution changes lag.** At 100k instances a one-thread worker landed 11 or 12 of the 40 requested placements in a 40-frame phase (one every 3 to 4 frames, 28 superseded; at 10k instances 12 to 33 landed, at 1k all 39 or 40), so the display shows every third or fourth seed; intermediate seeds are dropped (newest wins). This is the right trade: the alternative is a 70 to 190 ms stall on the frame thread (full grid rebuild in the hash table).
* **Partially refreshed nodes.** A node can be mixed old and new displacement for a few frames (per-vertex freshness). The surface stays continuous (vertices move independently) but the frame may show a half-updated patch.
* **Coarse cut while dragging.** `drag_detail_scale` makes dents faceted during the drag; they snap to the settle cut after release. Cutters under about 3 vertices across alias in the cut as they do in the full evaluation (spike-3 caveat).
* **Subtract only, spheres only.** Add and Intersect cost 2 to 4x more per vertex (spike-3 table), so they will refresh a smaller share per frame; mixed primitives about 1.5 to 3x.
* **Backface cones are vertex-normal based and static.** They are conservative for smooth shapes, not proven for noisy normals, and are stale after sculpting (see above).
* **No GPU, no real renderer.** The simulator measures CPU time of the scheduler and evaluation on the frame thread plus background work; upload of changed vertices is not included.
* **Peak memory.** At 6.3M faces the process held the document (about 1.5 GB), the scheduler snapshot, LOD pool, node lists and a job buffer; a 6.3M-face, 100k-instance case stayed well under 6 GB RSS here.

## Results

All numbers below are produced by `sculpt-cli generator-drag-sim 7,9,10 1000,10000,100000` (4 threads, 10 ms budget, one run). Raw program output is the source of every row.

#### Per configuration (all 120 drag frames, three phases)

Scheduled = `Scheduler` with a 10 ms frame budget, measured per frame including applying the change. Unscheduled = rebuild the instance set and evaluate every vertex (the spike-3 path), median over sampled frames of the same drag. Fresh = share of the visible cut's vertices that hold a result for the current parameters at the end of the frame (median over frames, phases size / count / seed). Converge = after release: frames and ms until every visible node is current, then until the background full-quality job landed and the buffer equals `evaluate()` exactly.

| Faces | Instances | View | Scheduled frame ms (median / p95 / max) | Unscheduled full recompute ms (median) | Speedup at median | Fresh verts % (size / count / seed) | Visible converged after release (frames) | Full-quality job landed (ms) | Exact |
|---|---|---|---|---|---|---|---|---|---|
| 98k | 1,000 | overview | 3.3 / 5.7 / 9.4 | 8 | 2x | 100 / 100 / 100 | 1 | 17 | yes |
| 98k | 1,000 | zoom | 3.8 / 10.2 / 33.2 | 9 | 2x | 100 / 100 / 100 | 1 | 17 | yes |
| 98k | 10,000 | overview | 10.1 / 14.7 / 17.6 | 35 | 3x | 71 / 85 / 72 | 2 | 37 | yes |
| 98k | 10,000 | zoom | 10.1 / 13.4 / 15.3 | 28 | 3x | 90 / 82 / 89 | 4 | 51 | yes |
| 98k | 100,000 | overview | 10.9 / 14.8 / 18.4 | 305 | 28x | 5 / 6 / 13 | 26 | 714 | yes |
| 98k | 100,000 | zoom | 11.0 / 16.0 / 21.2 | 339 | 31x | 4 / 6 / 14 | 28 | 667 | yes |
| 1.6M | 1,000 | overview | 9.3 / 13.1 / 16.1 | 34 | 4x | 90 / 100 / 89 | 2 | 89 | yes |
| 1.6M | 1,000 | zoom | 5.9 / 11.0 / 19.1 | 35 | 6x | 99 / 100 / 99 | 1 | 87 | yes |
| 1.6M | 10,000 | overview | 10.1 / 14.6 / 20.5 | 90 | 9x | 38 / 44 / 99 | 5 | 203 | yes |
| 1.6M | 10,000 | zoom | 10.2 / 15.0 / 31.6 | 81 | 8x | 63 / 83 / 98 | 8 | 203 | yes |
| 1.6M | 100,000 | overview | 10.2 / 12.7 / 15.3 | 320 | 31x | 11 / 14 / 63 | 39 | 671 | yes |
| 1.6M | 100,000 | zoom | 10.1 / 10.7 / 25.9 | 208 | 21x | 28 / 31 / 57 | 31 | 323 | yes |
| 6.3M | 1,000 | overview | 10.0 / 10.6 / 11.4 | 34 | 3x | 93 / 100 / 77 | 5 | 229 | yes |
| 6.3M | 1,000 | zoom | 10.0 / 10.6 / 15.5 | 36 | 4x | 77 / 100 / 65 | 2 | 183 | yes |
| 6.3M | 10,000 | overview | 10.2 / 18.2 / 32.0 | 185 | 18x | 39 / 38 / 89 | 12 | 508 | yes |
| 6.3M | 10,000 | zoom | 10.1 / 14.8 / 21.3 | 175 | 17x | 32 / 93 / 59 | 5 | 312 | yes |
| 6.3M | 100,000 | overview | 10.2 / 17.7 / 38.4 | 605 | 59x | 16 / 13 / 38 | 58 | 1329 | yes |
| 6.3M | 100,000 | zoom | 10.8 / 17.8 / 23.3 | 641 | 59x | 8 / 13 / 27 | 50 | 1105 | yes |

#### Per phase detail

`change` is applying the parameter change (instance update, incremental hash, dirty marking) on the frame thread; `select` is cut selection; `eval` is the time-budgeted evaluation. Over budget = frames longer than budget + 4 ms (14 ms). Fresh nodes is the share of visible cut nodes completely current; max stale age is the longest a visible stale node waited since it was last touched, in frames.

| Faces | Instances | View | Phase | Frame ms med | p95 | max | Frames > 14 ms | Fresh nodes % | Fresh verts % | Verts evaluated / frame | Visible nodes | Cut triangles | Max stale age | change / select / eval ms | Unscheduled ms med | Visible converged (frames / ms) | Full job (frames / ms) | Longest idle frame ms | Exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 98k | 1,000 | overview | size | 3.1 | 3.8 | 4.9 | 0 | 100 | 100 | 35,488 | 16 | 65,536 | 2 | 0.1 / 0.03 / 2.9 | 7 | 1 / 0 | 2 / 17 | 0 | yes |
| 98k | 1,000 | overview | count | 3.2 | 5.6 | 7.0 | 0 | 100 | 100 | 31,052 | 16 | 65,536 | 5 | 0.1 / 0.04 / 3.0 | 8 | 1 / 0 | 2 / 17 | 0 | yes |
| 98k | 1,000 | overview | seed | 4.6 | 7.0 | 9.4 | 0 | 100 | 100 | 35,488 | 16 | 65,536 | 5 | 0.0 / 0.04 / 4.4 | 10 | 1 / 5 | 2 / 17 | 5 | yes |
| 98k | 1,000 | zoom | size | 3.0 | 8.3 | 22.5 | 2 | 100 | 100 | 35,488 | 16 | 65,536 | 3 | 0.2 / 0.03 / 2.7 | 9 | 1 / 0 | 2 / 17 | 0 | yes |
| 98k | 1,000 | zoom | count | 2.8 | 7.4 | 12.3 | 0 | 100 | 100 | 31,052 | 16 | 65,536 | 5 | 0.2 / 0.04 / 2.2 | 8 | 1 / 0 | 2 / 17 | 0 | yes |
| 98k | 1,000 | zoom | seed | 4.9 | 11.0 | 33.2 | 2 | 100 | 100 | 35,488 | 16 | 65,536 | 5 | 0.0 / 0.04 / 4.7 | 11 | 1 / 3 | 2 / 17 | 3 | yes |
| 98k | 10,000 | overview | size | 10.2 | 13.8 | 14.7 | 2 | 69 | 71 | 25,282 | 16 | 65,536 | 6 | 0.5 / 0.04 / 9.5 | 35 | 1 / 3 | 3 / 34 | 3 | yes |
| 98k | 10,000 | overview | count | 10.1 | 15.2 | 17.6 | 3 | 81 | 85 | 28,454 | 16 | 65,536 | 5 | 0.2 / 0.03 / 9.7 | 32 | 1 / 0 | 3 / 37 | 0 | yes |
| 98k | 10,000 | overview | seed | 10.1 | 15.5 | 16.1 | 6 | 69 | 72 | 20,326 | 16 | 65,536 | 6 | 0.0 / 0.03 / 10.0 | 35 | 2 / 19 | 4 / 50 | 10 | yes |
| 98k | 10,000 | zoom | size | 10.2 | 12.4 | 12.9 | 0 | 88 | 90 | 32,017 | 16 | 65,536 | 4 | 0.4 / 0.03 / 9.7 | 28 | 1 / 5 | 4 / 51 | 5 | yes |
| 98k | 10,000 | zoom | count | 10.1 | 14.7 | 15.3 | 4 | 81 | 82 | 28,066 | 16 | 65,536 | 7 | 0.2 / 0.03 / 9.8 | 25 | 1 / 2 | 4 / 51 | 2 | yes |
| 98k | 10,000 | zoom | seed | 10.1 | 13.1 | 13.9 | 0 | 88 | 89 | 18,208 | 16 | 65,536 | 6 | 0.0 / 0.03 / 9.9 | 32 | 4 / 54 | 7 / 101 | 10 | yes |
| 98k | 100,000 | overview | size | 11.0 | 14.9 | 18.4 | 5 | 0 | 5 | 1,727 | 16 | 65,536 | 28 | 3.3 / 0.04 / 8.0 | 273 | 12 / 203 | 34 / 585 | 22 | yes |
| 98k | 100,000 | overview | count | 10.6 | 12.4 | 15.5 | 1 | 0 | 6 | 2,001 | 16 | 65,536 | 40 | 1.0 / 0.04 / 9.5 | 305 | 17 / 273 | 43 / 714 | 15 | yes |
| 98k | 100,000 | overview | seed | 11.1 | 14.8 | 17.6 | 4 | 0 | 13 | 1,782 | 16 | 65,536 | 48 | 0.0 / 0.04 / 10.9 | 317 | 26 / 424 | 47 / 776 | 19 | yes |
| 98k | 100,000 | zoom | size | 10.8 | 14.4 | 15.0 | 4 | 0 | 4 | 1,280 | 16 | 65,536 | 35 | 3.3 / 0.04 / 7.7 | 342 | 14 / 221 | 40 / 667 | 16 | yes |
| 98k | 100,000 | zoom | count | 10.6 | 12.8 | 14.9 | 2 | 0 | 6 | 2,134 | 16 | 65,536 | 48 | 1.0 / 0.05 / 9.4 | 326 | 13 / 203 | 38 / 627 | 14 | yes |
| 98k | 100,000 | zoom | seed | 11.6 | 18.5 | 21.2 | 12 | 6 | 14 | 1,811 | 16 | 65,536 | 41 | 0.0 / 0.04 / 11.5 | 339 | 28 / 464 | 48 / 808 | 18 | yes |
| 1.6M | 1,000 | overview | size | 10.2 | 14.5 | 16.1 | 4 | 94 | 90 | 102,809 | 48 | 196,608 | 8 | 0.2 / 0.14 / 9.2 | 34 | 2 / 21 | 6 / 88 | 10 | yes |
| 1.6M | 1,000 | overview | count | 4.5 | 9.1 | 9.3 | 0 | 100 | 100 | 42,747 | 48 | 196,608 | 16 | 0.2 / 0.14 / 3.8 | 27 | 2 / 18 | 6 / 89 | 16 | yes |
| 1.6M | 1,000 | overview | seed | 10.0 | 12.2 | 13.1 | 0 | 94 | 89 | 98,068 | 48 | 196,608 | 16 | 0.0 / 0.11 / 9.3 | 35 | 2 / 26 | 7 / 106 | 12 | yes |
| 1.6M | 1,000 | zoom | size | 6.8 | 10.2 | 11.7 | 0 | 100 | 99 | 140,765 | 64 | 262,144 | 6 | 0.1 / 0.44 / 5.6 | 26 | 1 / 1 | 6 / 86 | 1 | yes |
| 1.6M | 1,000 | zoom | count | 1.8 | 6.9 | 12.2 | 0 | 100 | 100 | 17,841 | 64 | 262,144 | 36 | 0.2 / 0.49 / 1.1 | 41 | 1 / 1 | 6 / 89 | 2 | yes |
| 1.6M | 1,000 | zoom | seed | 7.7 | 11.8 | 19.1 | 1 | 100 | 99 | 140,765 | 64 | 262,144 | 54 | 0.0 / 0.45 / 6.6 | 35 | 1 / 8 | 6 / 87 | 8 | yes |
| 1.6M | 10,000 | overview | size | 10.1 | 13.7 | 14.6 | 1 | 40 | 38 | 43,513 | 48 | 196,608 | 17 | 0.4 / 0.07 / 9.2 | 94 | 3 / 42 | 15 / 241 | 10 | yes |
| 1.6M | 10,000 | overview | count | 10.6 | 13.7 | 13.9 | 0 | 44 | 44 | 46,745 | 48 | 196,608 | 21 | 0.2 / 0.07 / 9.7 | 81 | 3 / 42 | 13 / 203 | 13 | yes |
| 1.6M | 10,000 | overview | seed | 10.1 | 18.8 | 20.5 | 6 | 100 | 99 | 25,783 | 48 | 196,608 | 18 | 0.0 / 0.07 / 9.2 | 90 | 5 / 71 | 11 / 169 | 10 | yes |
| 1.6M | 10,000 | zoom | size | 11.3 | 14.4 | 16.6 | 3 | 64 | 63 | 90,310 | 64 | 262,144 | 12 | 0.3 / 0.16 / 10.1 | 78 | 1 / 5 | 10 / 153 | 5 | yes |
| 1.6M | 10,000 | zoom | count | 10.1 | 13.6 | 16.0 | 2 | 83 | 83 | 85,094 | 64 | 262,144 | 15 | 0.3 / 0.16 / 9.3 | 81 | 1 / 4 | 13 / 203 | 4 | yes |
| 1.6M | 10,000 | zoom | seed | 11.3 | 16.2 | 31.6 | 6 | 100 | 98 | 54,369 | 64 | 262,144 | 18 | 0.0 / 0.14 / 10.8 | 100 | 8 / 121 | 15 / 240 | 10 | yes |
| 1.6M | 100,000 | overview | size | 10.2 | 13.0 | 14.6 | 1 | 10 | 11 | 12,279 | 48 | 196,608 | 43 | 3.3 / 0.11 / 6.3 | 343 | 9 / 142 | 38 / 627 | 10 | yes |
| 1.6M | 100,000 | overview | count | 10.2 | 12.7 | 15.3 | 2 | 12 | 14 | 15,560 | 48 | 196,608 | 58 | 1.6 / 0.12 / 7.8 | 320 | 13 / 209 | 40 / 671 | 24 | yes |
| 1.6M | 100,000 | overview | seed | 10.1 | 11.1 | 11.8 | 0 | 56 | 63 | 21,649 | 48 | 196,608 | 53 | 0.0 / 0.10 / 9.5 | 260 | 39 / 647 | 54 / 895 | 15 | yes |
| 1.6M | 100,000 | zoom | size | 10.1 | 10.4 | 11.0 | 0 | 28 | 28 | 39,942 | 64 | 262,144 | 21 | 1.9 / 0.20 / 7.6 | 208 | 2 / 28 | 17 / 275 | 11 | yes |
| 1.6M | 100,000 | zoom | count | 10.1 | 10.3 | 11.6 | 0 | 31 | 31 | 43,829 | 64 | 262,144 | 20 | 1.3 / 0.23 / 8.1 | 200 | 3 / 36 | 20 / 323 | 10 | yes |
| 1.6M | 100,000 | zoom | seed | 10.1 | 11.5 | 25.9 | 1 | 53 | 57 | 30,265 | 64 | 262,144 | 25 | 0.0 / 0.19 / 9.6 | 312 | 31 / 512 | 50 / 828 | 15 | yes |
| 6.3M | 1,000 | overview | size | 10.2 | 10.8 | 11.4 | 0 | 100 | 93 | 121,274 | 48 | 196,608 | 16 | 0.2 / 0.04 / 9.1 | 34 | 5 / 77 | 14 / 229 | 11 | yes |
| 6.3M | 1,000 | overview | count | 4.5 | 5.7 | 6.6 | 0 | 100 | 100 | 43,729 | 48 | 196,608 | 24 | 0.2 / 0.04 / 3.9 | 35 | 4 / 55 | 13 / 212 | 11 | yes |
| 6.3M | 1,000 | overview | seed | 10.1 | 10.7 | 11.4 | 0 | 81 | 77 | 99,915 | 48 | 196,608 | 23 | 0.0 / 0.04 / 9.3 | 34 | 5 / 72 | 14 / 230 | 12 | yes |
| 6.3M | 1,000 | zoom | size | 10.1 | 12.4 | 15.5 | 2 | 79 | 77 | 173,865 | 104 | 425,984 | 13 | 0.2 / 0.74 / 7.9 | 36 | 2 / 21 | 13 / 212 | 11 | yes |
| 6.3M | 1,000 | zoom | count | 1.8 | 2.5 | 2.9 | 0 | 100 | 100 | 13,958 | 104 | 425,984 | 38 | 0.2 / 0.74 / 0.8 | 32 | 1 / 8 | 11 / 183 | 13 | yes |
| 6.3M | 1,000 | zoom | seed | 10.1 | 10.2 | 10.6 | 0 | 68 | 65 | 148,386 | 104 | 425,984 | 120 | 0.0 / 0.73 / 7.9 | 39 | 2 / 24 | 11 / 179 | 11 | yes |
| 6.3M | 10,000 | overview | size | 10.3 | 16.9 | 18.2 | 3 | 40 | 39 | 50,869 | 48 | 196,608 | 24 | 0.5 / 0.04 / 8.7 | 185 | 6 / 94 | 25 / 413 | 14 | yes |
| 6.3M | 10,000 | overview | count | 10.1 | 12.7 | 13.8 | 0 | 40 | 38 | 48,116 | 48 | 196,608 | 31 | 0.3 / 0.03 / 8.8 | 174 | 7 / 104 | 30 / 508 | 20 | yes |
| 6.3M | 10,000 | overview | seed | 10.2 | 21.2 | 32.0 | 11 | 90 | 89 | 28,812 | 48 | 196,608 | 36 | 0.0 / 0.05 / 9.2 | 186 | 12 / 190 | 38 / 636 | 17 | yes |
| 6.3M | 10,000 | zoom | size | 11.2 | 14.8 | 16.2 | 4 | 32 | 32 | 76,853 | 104 | 425,984 | 29 | 0.4 / 0.34 / 8.6 | 175 | 3 / 41 | 18 / 328 | 33 | yes |
| 6.3M | 10,000 | zoom | count | 10.0 | 11.2 | 11.9 | 0 | 93 | 93 | 91,446 | 104 | 425,984 | 28 | 0.4 / 0.37 / 8.7 | 164 | 2 / 24 | 19 / 312 | 10 | yes |
| 6.3M | 10,000 | zoom | seed | 10.3 | 16.6 | 21.3 | 8 | 57 | 59 | 66,628 | 104 | 425,984 | 30 | 0.0 / 0.31 / 9.3 | 179 | 5 / 71 | 18 / 296 | 11 | yes |
| 6.3M | 100,000 | overview | size | 10.1 | 13.1 | 20.4 | 2 | 15 | 16 | 20,811 | 48 | 196,608 | 60 | 2.2 / 0.04 / 7.0 | 434 | 16 / 261 | 71 / 1204 | 14 | yes |
| 6.3M | 100,000 | overview | count | 10.6 | 16.1 | 16.3 | 5 | 12 | 13 | 16,939 | 48 | 196,608 | 90 | 1.4 / 0.03 / 8.0 | 605 | 15 / 245 | 79 / 1329 | 12 | yes |
| 6.3M | 100,000 | overview | seed | 10.7 | 21.9 | 38.4 | 9 | 31 | 38 | 17,449 | 48 | 196,608 | 93 | 0.0 / 0.05 / 9.8 | 793 | 58 / 974 | 109 / 1856 | 22 | yes |
| 6.3M | 100,000 | zoom | size | 10.7 | 15.4 | 19.6 | 5 | 8 | 8 | 19,931 | 104 | 425,984 | 56 | 3.8 / 0.26 / 6.5 | 641 | 8 / 121 | 60 / 1017 | 16 | yes |
| 6.3M | 100,000 | zoom | count | 10.6 | 14.4 | 23.3 | 4 | 12 | 13 | 30,272 | 104 | 425,984 | 80 | 1.4 / 0.23 / 8.3 | 561 | 8 / 119 | 66 / 1105 | 14 | yes |
| 6.3M | 100,000 | zoom | seed | 12.1 | 19.3 | 20.4 | 11 | 23 | 27 | 26,594 | 104 | 425,984 | 84 | 0.0 / 0.21 / 10.9 | 818 | 50 / 836 | 91 / 1527 | 14 | yes |

#### Spatial hash maintenance

| Faces | Instances | Hash numbers (ms, 4 threads) |
|---|---|---|
| 98k | 1,000 | build 0.9 ms; full grid rebuild 0.9 ms; old update(1% changed)=rebuild 0.9 ms; incremental 1% changed (x1.15) 0.00 ms; map_all x1.004 per step 0.06 ms (re-registered 0 of 1000 over 8 steps); map_all x1.4 jump 0.08 ms (median of alternating) |
| 98k | 10,000 | build 19.4 ms; full grid rebuild 13.8 ms; old update(1% changed)=rebuild 15.0 ms; incremental 1% changed (x1.15) 0.01 ms; map_all x1.004 per step 0.32 ms (re-registered 0 of 10000 over 8 steps); map_all x1.4 jump 0.30 ms (median of alternating) |
| 98k | 100,000 | build 76.9 ms; full grid rebuild 72.0 ms; old update(1% changed)=rebuild 78.3 ms; incremental 1% changed (x1.15) 0.08 ms; map_all x1.004 per step 5.50 ms (re-registered 0 of 100000 over 8 steps); map_all x1.4 jump 11.87 ms (median of alternating) |
| 1.6M | 1,000 | build 1.4 ms; full grid rebuild 2.6 ms; old update(1% changed)=rebuild 3.1 ms; incremental 1% changed (x1.15) 0.00 ms; map_all x1.004 per step 0.09 ms (re-registered 0 of 1000 over 8 steps); map_all x1.4 jump 0.16 ms (median of alternating) |
| 1.6M | 10,000 | build 11.0 ms; full grid rebuild 13.5 ms; old update(1% changed)=rebuild 14.3 ms; incremental 1% changed (x1.15) 0.01 ms; map_all x1.004 per step 0.79 ms (re-registered 0 of 10000 over 8 steps); map_all x1.4 jump 0.77 ms (median of alternating) |
| 1.6M | 100,000 | build 148.0 ms; full grid rebuild 127.9 ms; old update(1% changed)=rebuild 124.1 ms; incremental 1% changed (x1.15) 0.07 ms; map_all x1.004 per step 8.04 ms (re-registered 0 of 100000 over 8 steps); map_all x1.4 jump 10.75 ms (median of alternating) |
| 6.3M | 1,000 | build 1.2 ms; full grid rebuild 1.2 ms; old update(1% changed)=rebuild 1.0 ms; incremental 1% changed (x1.15) 0.00 ms; map_all x1.004 per step 0.07 ms (re-registered 0 of 1000 over 8 steps); map_all x1.4 jump 0.11 ms (median of alternating) |
| 6.3M | 10,000 | build 8.1 ms; full grid rebuild 7.9 ms; old update(1% changed)=rebuild 8.4 ms; incremental 1% changed (x1.15) 0.01 ms; map_all x1.004 per step 0.35 ms (re-registered 0 of 10000 over 8 steps); map_all x1.4 jump 0.50 ms (median of alternating) |
| 6.3M | 100,000 | build 101.2 ms; full grid rebuild 106.1 ms; old update(1% changed)=rebuild 90.5 ms; incremental 1% changed (x1.15) 0.17 ms; map_all x1.004 per step 2.38 ms (re-registered 0 of 100000 over 8 steps); map_all x1.4 jump 3.24 ms (median of alternating) |

#### Cancellation drill and scheduler counters

| Faces | Instances | View | Drill | Counters |
|---|---|---|---|---|
| 98k | 1,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 3.0 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 40, builds superseded 0; hash re-registrations 0, compactions 0 |
| 98k | 1,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 2.2 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 40, builds superseded 0; hash re-registrations 0, compactions 0 |
| 98k | 10,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 3.7 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 33, builds superseded 7; hash re-registrations 0, compactions 0 |
| 98k | 10,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 3.1 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 28, builds superseded 12; hash re-registrations 0, compactions 0 |
| 98k | 100,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 2.6 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |
| 98k | 100,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 4.5 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |
| 1.6M | 1,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 9.9 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 40, builds superseded 0; hash re-registrations 0, compactions 0 |
| 1.6M | 1,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 9.1 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 39, builds superseded 1; hash re-registrations 0, compactions 0 |
| 1.6M | 10,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 8.3 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |
| 1.6M | 10,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 10.0 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 18, builds superseded 22; hash re-registrations 0, compactions 0 |
| 1.6M | 100,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 5.7 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 11, builds superseded 29; hash re-registrations 0, compactions 0 |
| 1.6M | 100,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 4.8 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |
| 6.3M | 1,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 0.3 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 40, builds superseded 0; hash re-registrations 0, compactions 0 |
| 6.3M | 1,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 0.4 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 40, builds superseded 0; hash re-registrations 0, compactions 0 |
| 6.3M | 10,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 4.1 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 13, builds superseded 27; hash re-registrations 0, compactions 0 |
| 6.3M | 10,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 0.4 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 18, builds superseded 22; hash re-registrations 0, compactions 0 |
| 6.3M | 100,000 | overview | job running at interrupt true, cancelled 1 (new drag call took 2.3 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |
| 6.3M | 100,000 | zoom | job running at interrupt true, cancelled 1 (new drag call took 10.7 ms), final result exact: true | jobs started 6 cancelled 1 adopted 5, sets swapped 12, builds superseded 28; hash re-registrations 0, compactions 0 |

Scheduler setup (node vertex lists, normal cones, edge lengths), once per mesh: 1.6M 0.11 s; 1.6M 0.20 s; 1.6M 0.21 s; 1.6M 0.22 s; 1.6M 0.25 s; 6.3M 0.52 s; 6.3M 0.54 s; 6.3M 0.55 s; 6.3M 0.80 s; 6.3M 0.82 s; 6.3M 0.88 s; 98k 0.01 s; 98k 0.02 s; 98k 0.03 s.

## Reading the results

* **Frame time is bounded by construction, not by luck.** Wherever the work exceeds the budget (every 10k and 100k case, and 1k at 6.3M faces) the median frame sits at the 10 ms budget plus about 0.2 to 1 ms. Tails come from the model being wrong for one batch (a noisy neighbour on the shared cores, a landing instance set, a cold page) and are the same few percent of frames in every configuration. A real app should leave 4 to 6 ms of headroom under 16.7 ms, as the 10 ms default does.
* **Where time goes at 100k instances.** The `change` column is 1.9 to 3.8 ms for a global size slider (`rescale_all`), 1.0 to 1.6 ms for the count slider, and about 0 for the seed slider (the build is on a worker). `select` is 0.04 to 0.8 ms. The rest of the budget is evaluation, which at 100k instances evaluates only about 2k to 44k vertices per frame, hence the small fresh share. Fresh share is highest for the seed phase because new instance sets arrive only a few times per second and the frames between them finish the previous set.
* **LOD effect.** The cut holds 65k to 426k triangles (16 to 104 nodes) against 196k to 12.6M triangles in the mesh. Together with backface culling (overview) and the frustum (zoom) a full refresh of the visible cut is tens of thousands to about 200k vertices (the 'Verts evaluated / frame' column of the 1k-instance rows) instead of 98k to 6.3M. The 6.3M-face unscheduled numbers (35 ms at 1k instances up to 641 ms at 100k) show what that is worth; the speedup grows with mesh size and with instance density.
* **`count` drags are cheaper than `size` drags** when few instances change, because only the leaves under the toggled instances are dirty; at 1k instances only the affected nodes are re-evaluated (frame 1.8 to 4.5 ms at 1.6M and 6.3M faces). At 10k and above the toggled instances touch every leaf anyway.
* **Convergence after release** includes the foreground catch-up and then the full-quality job: it shares the four cores with nothing else once the visible cut is current, so its time is close to the spike-3 full evaluation time scaled by this box's speed (6.3M faces, 100k instances: 1.1 to 1.3 s here against 278 ms measured in spike 3 on a quieter machine, with the size factor 1.3 and the noise accounting for the rest).
* **Idle frames after release** use the same deadline logic; the longest was 33 ms over all cases and 10 to 24 ms at 100k (column "Longest idle frame ms"), noise included. The one exception that can stall a frame is the grid rebuild that `rescale_all` / `map_all` fall back to when more than an eighth of the instances leave their slack (an instance-set jump rather than a drag); it did not trigger in these drags (`compactions 0` in the counters).

## Recommendation

1. **Adopt `generator::schedule::Scheduler` as the live evaluator of the Generator layer** (the hybrid from spike 3: live A, committed B). Call it from the frame loop as shown, with a pass budget of 8 to 10 ms, `Mode::Drag` while a slider is down and `Mode::Settle` afterwards, and draw `select_cut` instead of the plain LOD cut while a generator is active. Expect: fully live up to 1k instances at every mesh size, live-with-a-short-tail up to 10k (30 to 99% of the visible vertices current during a drag depending on mesh size; the tail finishes 2 to 12 frames after release), progressive at 100k.
2. **Keep the background job on release.** It is what guarantees the committed state is exactly the spike-3 evaluation and what lets the drag path be approximate (coarse cut, skipped tiny nodes, stale carry-over) without a quality risk. Cancellation is cheap enough (under about 11 ms measured, one chunk) to start a new drag at any moment.
3. **Use the incremental hash entry points that match the control**: `rescale_all` for a global size slider, `update_incremental` for edits to a few instances, `apply_changes` with deactivation for a count slider (shuffled placement, prefix active), `request_replace` for seed and distribution changes. Do not call `set_all` or `update` (full rebuild, 72 to 128 ms at 100k here) from a drag.
4. **To make 100k feel live, the next step is to reduce work per frame, not to schedule better.** In order of expected return: (a) evaluate a fixed pseudo-random 1/8 to 1/4 of the instances while dragging (the deactivate path already supports it; cost scales with the number of instances overlapping each vertex) and restore the full set on release; (b) raise `drag_detail_scale` to 5 to 6 at 6.3M faces; (c) per-leaf instance lists or a scatter evaluation so the probe floor (the 1k-instance 6.3M-face case, where 65 to 100% of the visible vertices are current during a drag) drops, as suggested in spike 3; (d) a uniform-sphere / box fast path for the other primitives. More cores help the parallel part directly (the container has 4).
5. **Before shipping:** recompute normals for the nodes the scheduler writes; hook `mark_leaves` into sculpting under a live generator; share the snapshot with the LOD refresh worker so the mesh is not copied twice; measure Add and Intersect with the scheduler (they cost 2 to 4x more per vertex, so expect a lower fresh share); decide whether partially refreshed nodes are acceptable visually or whether a node should be committed atomically (the cost is progress per frame at 100k instances, where one node takes longer than the whole budget).

### Status of the cheap wins listed in spike 3

| Spike-3 item | Status |
|---|---|
| Incremental spatial hash (remove the 15 to 42 ms rebuild at 100k) | Done: slack registration plus overlay, 0.07 to 0.17 ms for 1% changed instances; global resize 2 to 4 ms per frame |
| Skip leaves that are off-screen or at low LOD while dragging | Done: view cut with frustum, backface, LOD and tiny / sub-pixel skipping; full evaluation on release in a background job |
| Per-leaf instance lists / per-instance scatter | Not done (the 1k-instance floor at 6.3M faces remains) |
| Fast paths for primitives other than uniform spheres | Not done |

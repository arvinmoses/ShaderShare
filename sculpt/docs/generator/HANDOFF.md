# Hand-off prompt: build the Generator Boolean layer (local session)

Paste everything below the line into a new Claude Code session running **on the user's own machine**,
in a clone of `arvinmoses/ShaderShare`, on branch `ccr-32ffc09b-o5mvh1` (all work is on it and pushed).
The app lives in `sculpt/` (Cargo workspace: `sculpt-core`, `sculpt-cli`, `sculpt-app`).

---

## ROLE

You are taking over a Rust sculpting-app project. Earlier cloud sessions did research and six
performance spikes, but they had no real GPU and no display. You run locally, so you **can** measure
real GPU speed, see the app and iterate with the user. The user is a 3D artist who sculpts
(Mudbox / ZBrush / Substance Painter background). They value: legibility, intuitive workflow, speed
("snappy", never a one-minute rebuild), no cross-screen mouse travel, and elegant modular OOP code.
Discuss trade-offs with them before large decisions. Use subagents where useful (Opus for research and
review, Sonnet for implementation, Fable only for a final holistic review). Validate at every milestone,
and give them something they can evaluate at each milestone. Use plain language with no jargon.

Do not open a pull request unless asked. Commit with clear messages and push to `ccr-32ffc09b-o5mvh1`
(or a branch the user names). End commit messages with the attribution lines your session specifies.
Never put a model identifier in commit messages or code comments.

## THE PROJECT IN ONE PARAGRAPH

A Mudbox-style sculpting app with ZBrush-level density and brush feel. Engine (`sculpt-core`): a
chunked spatial hierarchy (leaves of about 2048 faces, vertices renumbered so each leaf owns a contiguous
range), lock-free parallel brush writes, sparse per-leaf layer deltas (`P = base + sum(opacity * mask * delta)`),
per-leaf undo, dirty-leaf GPU upload, Catmull-Clark subdivision as a linear operator (layers survive it),
mask stacks (noise, curvature, cavity, AO, thickness bakes, painting, levels, blur), a data-driven project
format (`project.json` + `blobs/*.bin`). App (`sculpt-app`): egui 0.36 + wgpu 30, themeable JSON themes,
latency-first input path (same-frame input to dabs to dirty-leaf upload to render). A separate cloud session
has been running a Substance-Painter-style layer UI pass (folders, drag-drop, context menus, radial menu,
live thumbnails, blend modes); its last blind score was 8.2/10 (`docs/ux/scores/m9.md`). It also added a
hierarchical LOD tree for very dense meshes. Read `README.md`, `docs/ARCHITECTURE.md` and
`docs/ux/substance_painter_layer_ux.md` first.

## THE FEATURE: GENERATOR BOOLEAN LAYER

A new layer type that creates a primitive and **adds, subtracts or intersects** it against the layers
underneath (the user dropped "divide"; their message was cut off, ask if it comes up). The layer also
**replicates** the primitive:

- **Ordered or chaotic** distributions (grid, spacing along edges, seeded random with minimum spacing).
- **Data-driven placement:** edges, curvature, surface, vertices, face centres, points.
- **Mesh data mapped onto instances:** align instances to the edge normal or the surface normal; map edge
  length (as a proportion of a unit value) to instance height; surface normal rotation is required.
- **Edge damage** is the motivating case: place cutters along sharp edges, aligned to the edge normal, sized
  from edge length, and subtract them.
- **Masks (the user's rule):** a mask is a *density field*, not a random-thinning filter. White pulls *all*
  spawns in; black gets none. Fixed count, redistributed (importance sampling via a per-face CDF). Users paint
  masks to force instances into areas. Edge and curvature selection should be masks too (add an **Edge bake**
  alongside curvature / cavity / AO / thickness). The layer's own mask separately scales boolean strength.
- **It must be snappy and interactive.** No full rebuild per parameter change. Cache each pipeline stage and
  invalidate only from the changed stage onward (Distribution, Attributes, Instances, spatial index, Boolean).
- **Custom OBJ primitives** are in the first version, plus the built-ins (box, sphere, cylinder, cone, wedge,
  chamfer cutter, torus).
- The user said **"it needs to be a boolean"** and floated voxels, then asked to explore more experimental
  options (adaptive distance fields like Foundry's Kanova / Mischief, GPU, render-time CSG).
- The user thinks **deltas** are important. See "Deltas" below.

Design doc: `docs/generator/DESIGN.md` (pipeline, caching, distribution, attribute mapping, layer integration,
UI, risks).

## WHAT THE SIX SPIKES FOUND (all code merged on this branch; each has a doc in `docs/generator/`)

All CPU numbers are from a 4-core container, shared and noisy (assume 1.5-2x pessimistic). **No GPU number
below is a real GPU measurement.** The container only had software Vulkan (llvmpipe).

| Spike | What | Key result |
|---|---|---|
| A (`spike_results.md`) | CPU displacement boolean: analytic SDF primitives, uniform-grid instance hash, per-leaf evaluation, vertices projected onto the cutter surface. Placement: mask-weighted surface sampling, dihedral-angle sharp edge sampling. | Interactive up to about 10k instances: slider drag 15-34 ms (up to 1.6M faces), 58-80 ms at 6.3M faces. 100k instances: 83-278 ms. Placement cold 21-36 ms at 6.3M faces, edge detect 117 ms once. |
| B (`spike_results.md`, `voxel.rs`) | CPU uniform-brick voxel boolean. | Not interactive: 1.9 s to 298 s cold; seconds even with cached mesh distance at 10k instances. Changes topology. (Later found: patch-vertex counts about 25% high, a likely bound bug, so timings are slightly pessimistic.) |
| 1 (`spike1_gpu_compute.md`) | GPU compute port of A (`sculpt-app/src/gpu_boolean/`, `--gpu-bench`). | Correct vs CPU for 3 ops and 7 primitives (3 of 98k vertices differ with 10k overlapping cutters: order-dependent tie-break, fixable). Real-GPU speed UNMEASURED; reasoned estimate 3-10 ms at 6.3M faces / 10k instances, 15-35 ms at 100k. |
| 2 (`spike2_render_csg.md`, `spike2_code/`) | Render-time CSG preview in the viewport shader, pixel-exact cut edges at any mesh density with no mesh change. | Edge identical at 98k and 6.3M faces; per-pixel cost flat 100 to 100k instances. Fails for concave meshes and thin shells (needs depth peeling), open meshes, no depth output. Built on the OLD pre-LOD viewport, so it is **parked**: copy `spike2_code/` and re-apply `hooks.patch` by hand. `sculpt/scripts/csg_bench.sh` benchmarks it. |
| 3 (`spike3_refine.md`, `refine.rs`) | Adaptive boundary refinement: split only faces along each cutter rim, rim vertices exactly on the cutter/surface intersection, crack-free by construction. Subtract only. | Rim accuracy 0.008-0.033 of an edge vs 0.3-1.0 for A. Slow: 110-195 ms for 1k instances full, 0.8-1.8 s at 10k; 1% edit 2-20 ms (1k). So use after the drag settles. Weak: centred-on-surface cutters collapse, overlapping cutters crinkle, sliver rim triangles. |
| 4 (`spike4_lod_progressive.md`, `schedule.rs`) | LOD-aware, time-budgeted progressive scheduler + incremental spatial hash + `generator-drag-sim`. | Frame time median 3-11 ms, p95 <= 16.7 ms in 15 of 18 cases; 2-6x faster than full recompute at 1k, 3-18x at 10k, 21-59x at 100k. 100k instances is a progressive display (4-63% of visible vertices current mid-drag), needs instance subsampling while dragging. Converged result bit-identical to full evaluation. Not wired to `Document` or the app; normals not recomputed. |
| 5 (`spike5_gpu_sdf.md`, `gpu_sdf/`) | GPU sparse-SDF boolean: brick cache, instance binning, analytic combine, surface nets. | Correct vs CPU. Real-GPU speed unmeasured (modelled 1-4 ms warm at 1k, 4-37 ms at 10k, 40-210 ms at 100k; the model is 4-12x optimistic vs llvmpipe). Heavy: cache up to 6.4 GB, overlay 3.5 GB. Does not map onto per-vertex layers. Recommend commit-only. |
| 6 (`spike6_adf.md`, `adf.rs`) | CPU Adaptively Sampled Distance Field boolean (Foundry Kanova's basis; Frisken/Perry, MERL). | Cold 14.7-26 s at 6.3M faces / 100k instances (vs 113-298 s uniform bricks); 12.5M cells vs 733M voxels, 0.6 GB vs 4.2 GB. Sharp corners exact (0.005 edge vs 0.33-1.44). Incremental: 1% edit 2-5 ms (1k), 20-170 ms (10k). Not a live slider evaluator. About 1% of quads have holes; open boundary, so commit only. Papers not fetched (blocked), so ADF claims are from memory. |

## RECOMMENDED ARCHITECTURE (from the findings)

1. **Live evaluator:** the CPU displacement boolean (A) driven by the progressive scheduler (4), per-leaf and
   delta-based. Optionally the GPU port (1) when a hardware adapter is present, with CPU as reference and
   fallback. Decide with real GPU numbers (see "First steps").
2. **Exact preview while tuning:** render-time CSG (2) once ported to the current LOD viewport. Preview only.
3. **Crisp edges after the drag settles:** adaptive refinement (3) on dirty leaves, drawn in place of the
   replaced faces.
4. **"Apply as solid" / convert to real geometry:** the ADF (6), optionally with the GPU SDF (5) as an
   accelerator if the real-GPU numbers justify it. It is a remesh event (loses subdivision levels, flattens
   layers) so it must be an explicit, undoable snapshot.

## DELTAS (the user's key idea; treat as a design pillar)

Our layers are already per-leaf sparse position deltas, and the displacement boolean's output is already a
per-vertex delta relative to the layers below. So a generator layer fits the engine as a *computed* delta layer.
Use this:

- **Incremental updates:** recompute deltas only in leaves touched by changed instances; upload only dirty leaves.
- **Undo/storage:** the result is procedural, so only parameters are stored (small commands). Cache per-leaf
  delta chunks, invalidated when lower layers change in those leaves.
- **Sculpting on top** keeps working (additive offsets). The generator is a *function layer*: its delta depends
  on positions below it, so it is evaluated in stack order, not summed independently.
- **Commit with parent weights:** refinement (3) creates new vertices; record each new vertex's parent vertices
  and weights. Then Apply can be a *local subdivision plus a delta layer*, and existing layers/masks carry over by the
  same linear weights, instead of a global remesh that flattens layers. Voxel/ADF vertices have no parents and
  need nearest-surface resampling (0.3-0.8 us/vertex on 4 threads), so keep them for full-new-topology cases.
- Catmull-Clark is linear, so deltas survive subdivision; the procedural generator recomputes at the new density.

## FIRST STEPS (you have a real GPU and display now)

1. `cd sculpt && cargo build --release --workspace && cargo test --release --workspace` (expect all green: it
   was green at the merge; clippy had no new warnings). Install Rust via rustup if missing.
2. Run the real benchmarks and record results in a new `docs/generator/local_results.md`, including the GPU
   model, driver, OS:
   - `cargo run --release -p sculpt-app -- --gpu-bench 7,8,9,10 1000,10000,100000 --cpu`
   - `cargo run --release -p sculpt-app -- --gpu-sdf-bench` (and `--gpu-sdf-check`, `--gpu-sdf-png <dir>`)
   - `sculpt/scripts/csg_bench.sh`
   - `cargo run --release -p sculpt-cli -- generator-bench`, `refine-bench`, `adf-bench`, `generator-drag-sim`
3. Decide with the user, using the real numbers: GPU live evaluator or CPU+scheduler; whether the render-time
   preview is worth porting; whether the GPU SDF is worth it as an accelerator for Apply.
4. The user may share **Foundry Kanova** screenshots / clips (they have a build). Treat them as the reference
   for how adaptive-resolution sculpting should feel, and ask what they like.
5. The Substance Painter layer UX research was blocked from Adobe's docs in earlier cloud sessions; the spec
   `docs/ux/substance_painter_layer_ux.md` was later corrected from Adobe screenshots. Ask the user for fresh
   screenshots if a layer-UI detail is in doubt.

## PROPOSED BUILD ORDER (confirm with the user)

1. **Engine generator layer** in `sculpt-core` (own module tree under `generator/`): parameters in the project
   file (new layer kind with a version bump; old projects must still load; keep an old-format fixture test),
   staged caches, evaluated in layer order, undoable, procedural. Single hook into the layer composite (the UI
   pass also touches `document.rs` and `layers.rs`; keep the footprint small to avoid conflicts).
2. **Attribute mapping table:** fixed ordered rows (source attribute, optional remap/unit value, target property),
   not a node graph. Sources: surface normal, edge normal, edge tangent, edge-chain length, local edge length,
   curvature, cavity, AO, thickness, density-mask value, noise, per-instance random, instance index. Targets:
   align axis to a direction, roll, scale X/Y/Z or uniform, height, offset along normal, density weight.
3. **Edge bake** (sharp-edge by dihedral threshold plus curvature ridges with hysteresis; expose the threshold).
4. **Custom OBJ primitives** (embedded as a blob).
5. **Wire the progressive scheduler** (`schedule.rs`) into `Document` and the app per frame; recompute normals.
6. **Generator UI** in the existing layer panel: row badge/icon, Properties sections (Primitive, Distribution,
   Attributes, Boolean), viewport instance gizmos while selected, presets ("Edge damage", "Surface pitting",
   "Rivets" (add), "Panel cuts").
7. **Refinement after settle** (3) with parent weights; then **Apply as solid** (6) as an explicit step.
8. **Render-time preview** (2) ported onto the LOD viewport, if the user wants it.

## VALIDATION (every milestone)

- Release build, `cargo clippy --release --workspace` with no new warnings, `cargo test --release --workspace`.
- Add tests for every engine feature: placement determinism, mask weighting (>=95% of points in white regions),
  edge sampling on a cube, GPU-vs-CPU equality (skip without an adapter), save/load round trip, old-format
  project fixture, undo/redo, subdivision survival, incremental result equals full recompute.
- Latency guard: sculpting must not regress. Run
  `./target/release/sculpt-app --level 9 --test-strokes 180 --screenshot out.png` before and after. Earlier
  figures (software Vulkan, 4 cores): input + dabs median 2.1 ms at 98k faces, 2.8 ms at 393k, 12.4 ms at 6.3M;
  panels about 0.5 ms/frame. Investigate any regression over about 10%.
- Interactivity guard: slider-drag frame time (use `generator-drag-sim` and a real drag in the app) must
  stay near 16 ms p95; report p95 and max.
- Screenshots in all four themes for UI work; a blind Opus reviewer scoring against the rubric in
  `docs/ux/substance_painter_layer_ux.md` section 9 if the layer UI changes.
- Update README / ARCHITECTURE and keep `keymap.json` defaults consistent.

## CODE-QUALITY RULES

Elegant, modular, proper OOP. Small types with one responsibility; traits where behaviour varies (e.g. a
`BooleanEvaluator` trait implemented by CPU displacement, GPU displacement, and the exact-solid backends; a
`Distribution` trait; an `AttributeSource` enum). UI actions go through the command layer so undo stays
consistent. New colours go through theme JSON, never hard-coded. Do not touch the sculpt hot path.

## KNOWN GOTCHAS

- egui 0.36 / wgpu 30 API differences (`Panel::top/show`, `MenuBar::new().ui`, `CentralPanel::no_frame().show`).
- Container-era spikes were run on a loaded 4-core box: re-measure everything locally before trusting a number.
- The spike branches' merges reconciled the instance set: the incremental-hash `InstanceSet` replaced the old one,
  and `bake_only` / `update_bake_only` (used by ADF) and `candidates_in_box` (used by refine) were re-added.
- Spike 2 (render CSG) is parked, not merged into the viewport.
- Spike 5 and 6 patch extraction have known hole/bound issues (documented in their docs).
- Axis-aligned rays gave NaN in the BVH slab test and rays through shared vertices slipped between triangles;
  both were fixed in the engine (EPS padding, safe-direction epsilon). Keep those fixes.
- Do not run `playwright install`; Chromium is pre-installed in cloud sessions (not relevant locally).

## WHAT TO TELL THE USER FIRST

Summarise (in a few lines) that you have read the hand-off, list the benchmark commands you are about to run
and what GPU you detected, ask for their GPU model and OS if you cannot tell, and ask whether they want to
share Kanova screenshots. Then run Step 2 and report the real numbers before deciding anything.

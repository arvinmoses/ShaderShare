# Generator Boolean layer: build plan

Decisions taken from the local benchmarks (`local_results.md`) and reference notes (`kanova_notes.md`, `zbrush_notes.md`):

- **Live evaluator:** CPU displacement boolean now, behind a `BooleanEvaluator` trait. The GPU displacement boolean
  (spike 1, 4 ms at 6.3M faces / 100k instances) plugs in later as a second implementation.
- **Preview and commit are separate stages.** Live is the displacement boolean; Apply as solid is explicit and undoable.
- **GPU SDF:** kept as an optional Apply backend, to be compared against the ADF in the app. Not on the live path.
- **Render-time CSG preview:** deferred to the end.

## Where it hooks in (small footprint)

A generator layer is a normal layer with a `generator` spec and no stored deltas. In `Document::recomposite`, the
per-leaf loop applies layers bottom to top; a generator layer evaluates the displacement boolean on the positions
accumulated so far, then blends the result with `weight(v)` (strength x layer mask). So it is a function layer,
evaluated in stack order, and sculpting on layers above or below keeps working.

- `layers.rs`: `LayerKind::Generator`, and `generator` + a derived cache on `SculptLayer`; the spec also lives in `LayerMeta`
  so the existing undo swap covers parameter edits.
- `generator/layer.rs` (new): `GeneratorSpec` (serialisable), `GeneratorCache` (placement and spatial hash), build and
  evaluate functions. All generator logic stays under `generator/`.
- `io/project.rs`: format version 4 adds an optional `generator` entry per layer. Older projects load unchanged.

## Direction set by the artist

- **Target:** organic work first: rocks, ruins, rubble piles and similar things you would sculpt. Hard surface comes later.
- **Priority:** iteration speed beats everything. A live preview is wanted, but never at the cost of how fast edits answer.
- **Soft blends:** transitions between a cutter and the surface are soft by default for organics, with a blend radius. Sharp
  edges are a later, separate target (feature-preserving mesher, angle-based sharp preservation).

## Milestones (each is something you can inspect)

| # | Outcome | How to check |
|---|---|---|
| M1 (done) | Generator layer in the engine: scatter, add / subtract / intersect, save and load, undo, subdivision, panel, edit timer. | Tests, demo clips. |
| M2 (in progress) | **Iteration speed.** Staged caches (done), changed-leaf-only updates (done), progressive evaluation and a cheaper level-of-detail catch-up for dense meshes (next). | `generator-edit-bench`, the in-app readout, frame time at 6.3M faces during a drag. |
| M3a (done) | **Minimum spacing** (dart throwing, no regional bias, fewer instances when the surface is full) and **per-instance variation**: size, per-axis stretch, tilt, lift jitter, each on its own random stream so editing one never reshuffles another. Old projects load unchanged. | `tests/generator_placement.rs`, `m3a_placement.png`. |
| M3b (done) | **Density:** the instance count stays fixed and a density field redistributes it (area times density, importance sampled). Even, Clusters (noise patches with coverage and softness), Facing (toward the ground, the top or a side) and Painted (a per-vertex channel, written by the mask paint tool). Built only when it changes, so size and variation edits stay cheap. | `tests/generator_placement.rs` (>= 95% of points in white regions, fixed count, all-black places nothing), `m3b_density.png`. |
| M3c | Edge placement along sharp edges with edge-aligned frames, and the Edge bake. | Tests on a cube, screenshots. |
| M4 (done) | **Attribute mapping table:** ordered rows of source -> remap -> target with a signed strength. Sources: slope, height, noise, baked cavity / curvature / ambient occlusion / thickness (baked on demand), any painted channel, random, order. Targets: size, height along the normal, lift, lean. Rows only rebuild the instances, never the scatter. Align-to-direction and edge sources wait for edge placement (M3c). | `tests/generator_mapping.rs`, `m4_mapping.png`. |
| M5 | Generator UI in the layer panel proper, viewport gizmos, presets, custom OBJ primitives. | Screenshots in all four themes, blind score. |
| M6 | **Soft-blend live preview** (screen space, approximate), strictly optional so it cannot slow edits. The same blend definition as Apply, shared between shader and Rust with parity tests. | Parity tests, real-GPU timing, side-by-side with Apply. |
| M7 | Refinement after settle with parent weights, then **Apply as solid** with the blend radius (surface nets first, staged and cancellable, diagnostics). | Tests, in-app comparison. |
| M8 | GPU evaluator behind the same trait, CPU as reference. | GPU-vs-CPU equality tests, real-GPU timings. |
| Later | Hard surface: sharp-feature mesher (dual contouring), angle-based sharp preservation, bevels. | |

Each milestone ends with: release build, `cargo clippy --release --workspace` with no new warnings,
`cargo test --release --workspace`, and the sculpt-latency guard from the hand-off.

## Remesh and solidify (added after the rock experiments)

Stacked displacement layers fold the surface and never change topology, so a rock built from several generators tore.
The fix is to treat the stack as a solid for one pass and extract a fresh surface.

- `generator/remesh.rs`: sparse, manifold surface nets over any signed distance field. One vertex per inside region of
  a cell, so every edge belongs to exactly two quads (tests include a sub-voxel wall, touching balls and a rippled
  field); blocks far from the surface are skipped. On the rock: 106 ms against 470 ms for plain surface nets, 0
  non-manifold edges against 40.
- `Document::solidify` (`document/solid.rs`): base plus plain layers is the starting field; each generator layer combines
  its instances with a smooth blend (`GeneratorSpec::blend`, a fraction of instance size); each layer scatters on a coarse
  extraction of the field below it. Returns a mesh; the document is unchanged.
- `sculpt-cli rock-layers`: the rock as real layers, solidified (about 660 ms at 66k quads, most of it the base mesh
  distance queries).
- The ADF was tested on the same rock (13 ms, 19k triangles) but leaves holes at level crossings (214 non-manifold edges),
  has no smooth blend and cannot show grain on its large triangles. Kept as a possible backend for thousands of cutters.

A noise layer (`GeneratorKind::Noise`) roughens along the normals in the live stack and becomes a field term in stack
order when solidified; the app has a Generator menu (Scatter, Noise) and a Solidify section. Solidify caches base
distances between calls (unchanged: 99 ms, one layer edited: 177 ms on the rock).

A live solid preview (Display menu, or the Solidify section) shows the solidified result in place of the live mesh. It is
built on a worker thread with its own small thread pool from a snapshot (`Document::solid_input`), only after editing
has paused for 120 ms, so edits never wait for it; the last finished preview stays up until the next replaces it. While
it is on, sculpt strokes are not visible until the preview catches up.

Not done: density and painted channels in the scatter, layer opacity and masks in
the field, GPU evaluation.

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
| M3 | **Placement language for rocks and rubble:** minimum spacing, density mask (white pulls all spawns in, fixed count), size / rotation variance per instance, edge placement, edge bake, clusters. | Tests (spacing, mask weighting >= 95%), screenshots. |
| M4 | Attribute mapping table (normal, edge normal, edge length, curvature, noise, random -> align, scale, height, offset). | Tests, an "edge damage" and a "rubble" preset. |
| M5 | Generator UI in the layer panel proper, viewport gizmos, presets, custom OBJ primitives. | Screenshots in all four themes, blind score. |
| M6 | **Soft-blend live preview** (screen space, approximate), strictly optional so it cannot slow edits. The same blend definition as Apply, shared between shader and Rust with parity tests. | Parity tests, real-GPU timing, side-by-side with Apply. |
| M7 | Refinement after settle with parent weights, then **Apply as solid** with the blend radius (surface nets first, staged and cancellable, diagnostics). | Tests, in-app comparison. |
| M8 | GPU evaluator behind the same trait, CPU as reference. | GPU-vs-CPU equality tests, real-GPU timings. |
| Later | Hard surface: sharp-feature mesher (dual contouring), angle-based sharp preservation, bevels. | |

Each milestone ends with: release build, `cargo clippy --release --workspace` with no new warnings,
`cargo test --release --workspace`, and the sculpt-latency guard from the hand-off.

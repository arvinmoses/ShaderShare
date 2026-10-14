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

## Milestones (each is something you can inspect)

| # | Outcome | How to check |
|---|---|---|
| M1 | A generator layer exists in the engine: scatter a primitive over the surface, add / subtract / intersect it, saves and loads, undoable, survives subdivision. | Tests (determinism, save/load, old-format fixture, undo, subdivision, incremental equals full), CLI demo that writes before/after meshes. |
| M2 | Placement language: density-field mask (white pulls all spawns in, fixed count, importance sampling), edge placement, grid / spacing, minimum spacing. Edge bake. | Tests (>=95% of points in white regions, cube edges), screenshots. |
| M3 | Attribute mapping table (normal, edge normal, edge length, curvature, noise, random -> align, scale, height, offset). | Tests, "edge damage" preset demo. |
| M4 | Staged caches and the progressive scheduler wired into `Document` and the app per frame; normals recomputed. | `generator-drag-sim` through `Document`, p95 near 16 ms. |
| M5 | Generator UI in the layer panel, viewport gizmos, presets, custom OBJ primitives. | Screenshots in all four themes, blind score. |
| M6 | Refinement after settle, with parent weights. Apply as solid (ADF, optional GPU SDF), staged and cancellable, with diagnostics. | Tests, in-app comparison. |
| M7 | GPU evaluator behind the same trait, CPU as reference. Render-time preview if wanted. | GPU-vs-CPU equality tests, real-GPU timings. |

Each milestone ends with: release build, `cargo clippy --release --workspace` with no new warnings,
`cargo test --release --workspace`, and the sculpt-latency guard from the hand-off.

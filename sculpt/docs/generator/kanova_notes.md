# Kanova reference notes (from the installed build, no GUI session)

Source: `tunables.ini`, `README.md` and a strings scan of `kanova.exe` (Foundry Kanova 1.0v1b8, build 2018-12-03).
No hands-on session was possible (desktop control of the app was not granted), so nothing here describes how it
*feels* to use. These are structural observations only.

## Representation
- Adaptively Sampled Distance Fields (ADF) per layer, octree based. Default 12 octree levels per layer
  (`DefaultOctreeLevels`); a per-layer max level can be changed after creation ("Change layer resolution").
- `DefaultSubdivisionBias` (curvature-driven detail) and `DefaultSurfaceBias` (detail at the surface) trade
  detail against speed and memory.

## Operations visible in the binary
- Add / subtract is a stroke mode (`toggleSubtract`), with a smooth-blending option (`setSmoothBlending`). It is not a
  separate procedural boolean layer, so our Generator layer is a different kind of feature.
- "Full Remesh", "Copy To Clay" (converts a reference model layer into clay), "Cleanup Current Layer", layer copy,
  layer-as-brush, per-layer transform, per-layer validate.
- Copy To Clay is staged and cancellable: initialising, building, finishing up, with the time logged per stage.
- A post-edit ADF cleanup pass reports the nodes and samples it touched and its time.

## Responsiveness policy (from `tunables.ini`)
- `StrokeExpiryMS=250`: stroke segments that waited longer than this are skipped.
- `DropSlowStrokes=true`: slow smooth / flatten / move / paint strokes process only the first segment.
- It protects feel by dropping work, never by queueing it. Our progressive scheduler does the same by time budget.
- Undo budget `MaxUndoMB=500`; an average stroke costs 5-10 MB of undo.

## What to borrow
1. Make Apply as solid staged and cancellable, with per-stage timings.
2. Keep a deferred cleanup pass after a drag settles (matches the refinement step).
3. Per-layer resolution as an explicit, changeable property.
4. Drop stale work rather than queue it.

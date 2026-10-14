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

## Deeper pass (class names and log strings in `kanova.exe`)

Binary-only (no source ships; the PDB path in the binary points at Foundry's `adfsculpt` tree). 102 `Foundry::Kanova`
classes were recoverable from RTTI.

### Threading and command flow
- One scene thread owns the OpenGL context and the scene. A separate OpenVR thread runs VR. Work arrives through a
  multi-producer multi-consumer command queue (`Command` / `StrokeCommand`; tests exist for the queue).
- Every user action is a command object with undo steps (`CreateLayerCommand` + `CreateLayerUndoStep`, `StrokeUndoStep`,
  `SetLayerTransformUndoStep`, ...). A stroke can be recorded and played back (`Record Strokes` / `Playback Strokes`),
  which is how they run performance tests (`MouseStrokeTests`, `RecordSceneStatsCommand`, start/stop timer commands).
- Back-pressure: "Scene has a backlog of N strokes" and "Skipped N expired strokes". Expired work is dropped, not run late.
- Per-stroke timing is logged in stages: "Sculpting for stroke took X ms", "Partial remeshing for stroke took X ms",
  "Partial mesh post-processing for stroke took X ms", "Creating undo step for stroke ...". So the cost is split into
  field edit, local remesh, post-process and undo capture, each measured.

### Meshing is separate from the field
- `IMesher` interface with `EnhancedSurfaceNetsMesher` (default) and `DualContouringMesher` ("Experimental"),
  switchable at runtime (`SetMesherTypeCommand`). Remeshing is partial: only the region a stroke touched.
- This is the same split we recommend: a distance field as truth, a swappable mesher on top. Our trait for Apply as solid
  should look the same (`Mesher` with surface nets and dual contouring backends, both run on dirty regions only).

### Cleanup scheduling
- "Idle-Time Cleanup" and "Just-In-Time Cleanup" are separate toggles (`SetIdleCleanupEnabledCommand`,
  `SetJustInTimeCleanupEnabledCommand`); cleanup prunes ADF nodes and samples and reports "ADF cleanup of N nodes and
  N samples completed in X ms". Cleanup is deferred off the stroke path, either when idle or just before it is needed.
- A "Lazy Strokes" option exists, and `DropSlowStrokes` / `StrokeExpiryMS` bound how much stroke work is kept.

### Mesh strokes are their boolean
- Log lines "Processing AddMesh stroke" and "Processing SubMesh stroke", and "stroke aborted because source layer is not
  a reference model layer". So Kanova does add and subtract with a **mesh** (a reference model layer used as a brush) as
  a stroke, evaluated into the field. "Use Layer As Brush" and "Save Layer As Brush" turn clay into reusable brushes.
- It is a stamped operation per stroke, not a live parametric layer. Ours stays procedural and re-evaluable, which is the
  difference we are selling.

### Debug views worth copying
- Viewport modes: octree all / inside nodes only / outside nodes only / surface nodes only / "Octree and Field,
  Interactive", plus "All Layers Wireframe" and `setShowADFBounds`. A node-cut overlay would make our LOD tree and the
  scheduler's fresh/stale nodes legible to the artist.

# ZBrush reference notes (ZBrush 2022, from the install; documentation and resource strings)

Sources: the bundled "What's New" PDFs (4R7, 4R8, 2018, 2019, 2021.6, 2022), `ZData/ZHStrings.zsc`, and resource
identifiers in `ZBrush.exe`. No source ships, and nothing here is measured. Where a point is my inference rather than
something the documents say, it is marked **(inferred)**.

## Live Boolean: a preview, then an explicit conversion (4R8)
- Two separate things. **Live Boolean** (Render > Render Booleans) previews the result in real time, and you can move,
  scale, rotate, duplicate, change the operator and keep sculpting while it runs. **Make Boolean Mesh** (Tool > SubTool >
  Boolean) converts the preview to real geometry in a new Tool. The docs say a multi-million-polygon model converts in
  "a few seconds", complex ones in "up to a few minutes". The preview is "not impacted" by the mesh-quality rules below.
- Operators live on each SubTool: Start flag, Add, Subtract, Intersect. Evaluated top to bottom, visible SubTools only.
  "Start" groups make independent results from one list. SubTool Folders (2019) let Live Boolean act on a folder.
- Any model can be subtracted from another regardless of polygon count.
- This is the same split we reached on our own: render-time preview (our spike 2), then an explicit, slower commit.

## What the commit step does to the mesh (the part artists notice)
- Only the area touched by the boolean is re-triangulated. Everything else keeps its original topology.
- PolyGroups and PolyPaint survive (paint loses quality where a dense mesh meets a coarse one). Masks have no effect.
- Destroyed by the commit: UVs and textures, subdivision levels (collapsed to the current level and **cannot be
  rebuilt**), creases, 3D layers, HD geometry, render-time effects (MicroMesh, surface noise).
- Parametric primitives, ZSpheres and instances must be converted to a PolyMesh first. ArrayMesh and NanoMesh are
  converted automatically during the boolean.
- Hidden polygons are treated as holes and closed. Meshes should be watertight, manifold, free of coplanar faces, and
  of similar density, or the cut produces long thin triangles that break the next boolean.
- Consequence for us: ZBrush's own commit loses layers and subdivision, which is exactly the loss our parent-weight
  design avoids. Say so in the UI before an irreversible Apply, and keep Apply undoable.

## Clean-boolean tooling visible in the binary
- Render and Remesh are separate commands (`DO_BOOLEAN_RENDER`, `DO_BOOLEAN_REMESH`), with an `INTERACTIVE` option for
  the render.
- Coplanar faces are detected for the render (`BOOLEAN_RENDER_COPLANAR_THRESHOLD`) and can be listed and stepped through
  SubTool by SubTool (`SHOW_COPLANAR`, `NEXT/PREV_COPLANAR_SUBTOOL`, inside/solo display modes, hatch lines with a density).
- After the remesh it reports problems: `SHOW_ISSUES`, `NEXT/PREV_ISSUES_SUBTOOL`, and result states named
  `BooleanCleanIssueNoResult`, `...SomeResult`, `...SomeResultWithWarnings`, `BooleanResultWithWarnings`.
- A sliver cleaner with a size limit (`BOOLEAN_CLEAN_SLIVERS`, `BOOLEAN_SLIVERS_MAX_SIZE`), and a pre-pass that fixes
  non-manifold edges when it can.
- Takeaway: "clean boolean" is mostly diagnosis plus cleanup. It tells the artist which SubTool has the problem and where,
  instead of silently producing a bad mesh. We should do the same, including partial results with warnings.

## Instancing: NanoMesh and ArrayMesh (4R7)
- Both are **instanced geometry**. Edit the one source mesh and every instance updates immediately.
- **NanoMesh**: instances ("Nanos") are linked to **placement polygons**, which can be a single polygon, any selection,
  or a whole PolyGroup. So placement is data-driven by polygon selection and PolyGroups. Per-instance size (W/L/H),
  offset and rotation (X/Y/Z) each have a matching variance slider (WVar, XOvar, XRvar...). A Random Distribution slider
  overwrites the pattern with random placement across the placement polygons, with a Random Seed. Several NanoMeshes can
  be layered on the same surface. "Edit Placement" hides the Nanos while the placement polygons are edited.
- **ArrayMesh**: parametric copies from offset / scale / rotate around a pivot, with a copy count. Multi-stage, where each
  stage can reuse the previous stage's content or just the last copy's location, so arrays can nest (tank tread example).
  Controllable through sliders or the TransPose line. Presets are saved files in `ZArraysPresets`. An ArrayMesh can be
  converted to a NanoMesh so each copy gets randomised variation.
- Instancing is therefore "one source + placement rule + per-instance variance", evaluated as render-time instances. They
  only become real polygons at boolean or convert time.
- Missing from what is documented: no instance count or timing figures, no mask-as-density-field, no edge-based or
  curvature-based placement. Polygroups and polygon selection are the placement language. Our density-field masks,
  edge placement and attribute mapping go beyond this, and are worth showing as a strength.

## Remeshing
- **DynaMesh** (4R2): voxel-like remesh at a uniform resolution set by a Resolution slider. Add / Sub / And (union,
  subtract, intersect) happen at remesh time and need a mask-free, visible mesh. A Create Shell option exists.
  Docs say it is for the early, low-to-mid resolution stage, not for fine detail.
- **Sculptris Pro** (2018): adaptive on-the-fly refinement. Topology density follows brush size, so finer strokes
  create denser topology. Works on boolean results and scan data. Also gives a "Tessimation" pass to adjust polycount.
- **ZRemesher** (2019, v3.0): automatic retopology that keeps creases, detects sharp angles, runs faster. It can be
  aborted with Esc, and shows progress. It fails with a clear message when symmetry has no polygons on the major side.
- Remesh by DynaMesh also acts as a union of all intersecting meshes in a SubTool.
- Two different tools for two different jobs: refine locally while sculpting (Sculptris Pro), and a deliberate global
  rebuild (DynaMesh, ZRemesher). That matches our refine-after-settle step versus Apply as solid.

## Engine hints
- `embree3.dll`, `tbb.dll`, `ompp.dll` and `libiomp5md.dll` ship with it. Embree (CPU ray tracing), TBB and OpenMP
  strongly suggest a CPU, multi-threaded geometry stack. **(inferred)** The documents do not say what Live Boolean's preview
  uses internally, so do not assume it is CPU- or GPU-based.
- The documents give no speed numbers for the live preview beyond "real time" and the commit estimate above.

## What to take
1. Keep preview and commit as two distinct stages with different guarantees. Done in our plan.
2. Our commit can do better than ZBrush's: no lost layers (parent weights), and undoable.
3. Add a diagnostics view: coplanar and thin-sliver highlighting, per-source warnings, partial results with warnings.
4. Offer "convert to PolyMesh first" for primitives and instances automatically rather than refusing.
5. Instances edit through one source and update everywhere. Our custom OBJ primitive should do the same.
6. Local refinement near edits (Sculptris Pro idea) is the right model for our post-settle refinement.

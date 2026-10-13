# Sculpt — architecture and roadmap

**Product stance:** Mudbox's UI and workflow (tray, layer panel with strength
sliders, freeze, clean object list), with ZBrush's mesh density and brush
feel. Texture layers follow Substance Painter. Everything on disk is data, not
opaque binary state.

This document covers what exists in `crates/sculpt-core` today, why it is
built this way, and what comes next.

---

## 1. Requirements → status

| Requirement | Status | Where |
|---|---|---|
| Sculpt layers (Mudbox) with strength slider | **Done** | `layers.rs`, `Document::set_layer_opacity` |
| Per-layer mask that scales the layer's effect per vertex | **Done** | `Document::set_layer_mask` (any `MaskStack`) |
| Freeze (Mudbox) | **Done**: weighted 0–1, paintable, respected by brushes + posing | `PaintTarget::Freeze` |
| Clay Buildup (ZBrush) | **Done** | `brush::ClayBuildup` |
| Trim Dynamic + smooth border (ZBrush) | **Done** | `brush::TrimDynamic` |
| Move + topological Move (ZBrush) | **Done** | `brush::MoveBrush` |
| Smooth (Laplacian / surface relax) | **Done** | `brush::Smooth` |
| ZBrush-style posing (topological mask + transpose) | **Done (engine)**; gizmo UI pending | `pose.rs` |
| ZBrush-level speed/density | **Foundation done**: 6.3M faces sculptable on 4 cores (see §6) | `bvh.rs`, `document.rs` |
| Procedural masking | **Done**: fbm / perlin / ridged / turbulence / cellular, gradient, direction | `noise.rs`, `mask.rs` |
| Mesh-based masks: curvature, cavity, AO, thickness | **Done** (explicit bakes, BVH ray-traced) | `bake.rs` |
| Substance-style mask layering (noise + hand paint) | **Done**: blend modes, levels, invert, blur, nested masks | `mask.rs` |
| Data-driven layered file format | **Done**: JSON manifest + typed raw blobs | `io/project.rs` |
| Import data/noise from other programs | **Done (v1)**: per-vertex channel files, mask stacks as JSON | `io/mod.rs`, `MaskStack` serde |
| Substance-style texture layers | **Designed**, deferred (§7) | — |
| Mudbox-style UI / GPU viewport | **Done (v1)**: egui + wgpu, themeable, data-driven hotkeys (§8) | `crates/sculpt-app` |

---

## 2. Core data model

```
Document
├── faces            [u32;4] quads (tri = last slot 0xFFFFFFFF)
├── base             base-layer positions
├── rest             rest pose: reference space for procedural masks
├── positions        composited result (what you see / brush against)
├── normals
├── freeze           f32 per vertex, 1 = fully protected
├── channels         name → f32 per vertex (painted masks, bakes, imports)
├── layers[]         SculptLayer { opacity, visible, locked, mask: MaskStack,
│                                  chunks: per-leaf sparse Vec3 deltas }
├── bvh              leaves own contiguous vertex ranges (see §6)
└── undo             chunked copy-on-write snapshots
```

**Composite rule (sculpt layers):**

```
P(v) = base(v) + Σ_layers  scale · mask(v) · delta(v)
```

Each layer also has a **blend mode** that says how its offset combines with the
offsets below it (`A` = offset so far, `L` = the layer's offset, `s` = scale·mask):
Add `A+sL` (default, and how old projects behave), Subtract `A−sL`, Normal
`A+s·c·(L−A)` where `c` is footprint coverage (the layer replaces what is below
inside the area it touches), Max (apply only where `L·n>0`) and Min (only where
`L·n<0`). Add and Subtract keep the fast incremental stroke path; the others
re-composite the touched leaves per dab. Because Normal, Min and Max make order
matter, the composite walks the layer tree bottom to top (`composite_order`).
Folders have no blend mode; they only scale their children.

`scale` is the layer's strength times every ancestor folder's strength, and is
zero when the layer or an ancestor is hidden or another layer is soloed. The
document refreshes it whenever structure, strength or visibility change, so the
sculpting path reads one `f32` and never walks the tree.

**Layer tree (`layer_ops.rs`).** Layers live in one flat `Vec`; each names its
parent folder (`parent`, `kind = Layer | Folder`). Because deltas add, sibling
order is organisational only: it never changes the surface. Commands:
`insert_layer`, `insert_folder`, `delete_layer`, `duplicate_layer` (subtree),
`move_layer(Placement::{Above, Below, Into, Top})` (refuses cycles),
`group_layers`, `ungroup`, `merge_down` (bakes both layers' strength and masks into
the lower one), `set_solo` (not undoable, view state) and `set_layer_meta`
(name, strength, visibility, lock, collapse, mask; `coalesce` joins a slider drag
into one step). Each command is a list of reversible `StructOp`s recorded in the
same undo stack as strokes, so removed layers keep their data in the stack and
undo/redo are exact. **Previews** (`preview.rs`): `delta_preview` and `mask_preview` sample at most 96 vertices per spatial leaf into a front-view grid, so cost follows the leaf count, not the face count. `edit_serial` bumps on every geometry or mask change; the UI cache (`layer_panel/thumbs.rs`) rebuilds a thumbnail only when the serial moved, it is over 0.5 s old, no stroke is in progress, and fewer than two were built this frame.

Duplicating a layer or pasting a mask gives the copy its own hand-painted channels (`paint.layer<id>`), recorded in the same undo step. `flatten_layer` is undoable (touched base leaves are snapshotted); `flatten_folder` bakes Add-mode layers into one new layer and hides the folder.

Project format v2 adds `kind`, `parent` and `collapsed`;
v1 files load unchanged (fixture `tests/fixtures/v1_project`).

* The **strength slider** (`opacity`) and the **layer mask** both scale a
  layer per vertex without changing its data. Moving the slider only
  recomposites leaves where the layer actually has deltas (5 ms at 1.5M faces,
  16 ms at 6.3M).
* Layer deltas are **sparse per leaf**: a layer that touches 2% of the mesh
  stores about 2% of the data.
* Brushes write into the active layer (or base when no layer is active),
  locked or hidden layers refuse strokes, and `flatten_layer` bakes a layer
  into the base.

**Two vertex orders.** Internally, vertices are renumbered so each spatial
leaf owns a contiguous range (§6). Everything that crosses the API boundary
(files, OBJ, imported channels) uses the **canonical order**: the import
order, then Catmull-Clark order after each subdivision. Data authored in
another program against your OBJ lines up exactly.

**Rest pose.** Procedural masks (noise, gradients, direction) are evaluated in
the rest pose, not the posed or sculpted surface. Noise sticks to the model
like a texture when you pose it, and mask results are deterministic: a project
reloads bit-exact. `store_rest_pose()` re-anchors on purpose.

**Subdivision** is Catmull-Clark implemented as a *linear operator*
(`subdiv::Subdivider::apply<T>`). Because it is linear, base positions, every
layer's deltas, freeze and all channels are each subdivided independently, and
layers survive subdivision intact.

---

## 3. Brushes: what gives them the ZBrush feel

All brushes use one two-phase pipeline:

1. `compute_displacements(center, radius, kernel)`: read-only, one rayon task
   per intersected leaf. Freeze is applied here.
2. `apply_displacements`: writes the active layer's chunks and the composite
   (disjoint per-leaf slices, no locks), then updates normals and bounds only
   for touched leaves and their neighbours.

Strokes are spaced dabs (`StrokeSampler`, spacing as a fraction of the radius,
pen pressure interpolated) and are re-projected onto the surface.

| Brush | Key behaviour | Main parameters |
|---|---|---|
| **Clay Buildup** | Rounded-square footprint (superellipse) aligned to the stroke. Pulls the surface *up to a plane* floating above the local area center; vertices above the plane are untouched, giving plateaus and stepped clay strokes rather than Inflate-style balloons. Per-stroke cap: one stroke rises at most one plane height (ZBrush "Accumulate" off), and repeat strokes build further. Alt (invert) digs. | `height`, `squareness`, `accumulate` |
| **Trim Dynamic** | Plane recomputed every dab from the local area normal/center, sunk slightly below. Only material above it is shaved, carving crisp planar facets that follow the form. `smooth_border` relaxes the outer band (TrimSmoothBorder) so cuts blend. | `depth`, `smooth_border` |
| **Move** | Grabs vertices + falloff weights once at stroke start and drags them rigidly. **Topological** mode grows the selection by geodesic distance, so nearby but unconnected parts (fingers, lips) stay put. | `topological` |
| **Smooth** | Uniform Laplacian (strong, like ZBrush Shift), or tangential "surface" relax that keeps the form. | `mode` |
| **Paint** | Paints freeze or any named channel toward a value; the hand-painting half of the mask system. | `target`, `value` |

Shared settings: radius, strength, falloff hardness, spacing,
front-faces-only, invert. All are serde, so brush presets are data too.

---

## 4. Masks (Substance-style)

`MaskStack { base, layers: [MaskLayer] }` composited bottom to top. Each
`MaskLayer` has:

* **source**: `fill`, `channel` (painted / baked / imported), `noise`,
  `mesh` (curvature, cavity, ambient_occlusion, thickness), `direction`,
  `gradient`
* **levels** (in/out range + gamma), **invert**, topological **blur**
* **blend**: normal, multiply, add, subtract, screen, overlay, max, min,
  difference
* **opacity**, plus an optional nested **mask**, so you can say "only where I
  painted, let the noise through"

The same `MaskStack` type drives layer masks now and will drive texture-layer
masks later (§7). Example, exactly as stored in `project.json`:

```json
{ "base": 0, "layers": [
  { "name": "breakup", "source": { "type": "noise", "kind": "fbm", "scale": 5, "seed": 3 },
    "levels": { "in_min": 0.35, "in_max": 0.65 } },
  { "name": "painted", "source": { "type": "channel", "name": "paint.detail" },
    "blend": "screen", "opacity": 0.8 },
  { "name": "no cavities", "source": { "type": "mesh", "attribute": "cavity" },
    "blend": "subtract", "opacity": 0.5 } ] }
```

**Mesh bakes** (`bake.rs`) are explicit, like Substance's "bake mesh maps",
and write channels `mesh.curvature`, `mesh.cavity`, `mesh.ao` and
`mesh.thickness`:

* curvature: signed one-ring mean curvature, multi-scale through blur, and
  normalized by the 95th percentile so it adapts to density
* AO: cosine-weighted, per-vertex rotated hemisphere rays against the BVH
* thickness: inward rays, mean hit distance

A stack that references a missing bake triggers it automatically.

---

## 5. Posing (ZBrush Transpose + topological masking)

* `geodesic_distances(seed, max)`: Dijkstra over mesh edges.
* `topological_weights(seed, radius, softness, blur)`: the "Ctrl-click a limb"
  mask, with a soft falloff band.
* `pose(weights, Rotate | Translate | Scale)`: one undo step, respects freeze.

Posing transforms the base **and rotates every layer's deltas** by the same
per-vertex rotation, so detail on layers turns with the limb instead of
shearing. Procedural masks stay attached via the rest pose. The Transpose
action-line / gizmo is UI work (§8) that calls these functions.

---

## 6. Performance

**Design:**

* **PBVH with owned contiguous ranges.** Faces are split into leaves of about
  2048. Vertices are renumbered so every leaf owns a contiguous range. Brush
  writes, layer chunks, undo snapshots and (later) GPU uploads are all
  per-leaf slices: no locks, no scattered writes, cache-friendly.
* **Per-leaf ray trees** (4 faces/node) for picking and bakes. This took AO
  from 16 s to 0.27 s on 98k faces.
* **Sparse everything**: layer deltas, undo (copy-on-write per touched leaf),
  normal/bounds updates (touched leaves + neighbours only).
* **Data-parallel** with rayon across leaves.

**Measured** (`sculpt-cli bench`, 4-core cloud VM, no GPU, release build):

| Mesh | Brush radius | Verts per dab | Clay | Trim | Smooth | Move update |
|---|---|---|---|---|---|---|
| 1.57M faces | small (0.05) | ~800 | 0.79 ms | 0.73 ms | 0.65 ms | |
| 1.57M faces | large (0.4) | ~53k | 3.5 ms | 2.7 ms | 2.6 ms | 2.2 ms |
| 6.29M faces | small (0.05) | ~3k | 1.1 ms | 1.0 ms | 0.95 ms | |
| 6.29M faces | large (0.4) | ~210k | 10.6 ms | 7.2 ms | 7.4 ms | 6.0 ms |

Also at 6.29M faces: undo 6.6 ms, layer slider 16 ms, whole-mesh fbm mask
0.39 s, peak RSS 1.7 GB.

**Dense meshes (20M+ triangles):** drawing is made independent of mesh size by a hierarchical LOD that mirrors the
BVH (`sculpt-core/src/lod.rs`); see [`PERFORMANCE_20M.md`](PERFORMANCE_20M.md).

**Next performance work, in priority order:**

1. **GPU viewport fed by dirty leaves**: one vertex buffer range per leaf;
   upload only `geometry_changed` leaves each frame. This is the biggest
   user-visible item.
2. **Subdivision speed**: building level 10 takes 7.8 s and should be under
   2 s (parallel BVH build, parallel CSR construction, no full topology
   rebuild per level).
3. **Multires level storage** (Mudbox step up/down): keep the lower levels
   and propagate deltas down by restriction and back up by subdivision.
4. Memory per vertex (now about 270 B at level 10): f16 or quantized normals,
   tighter topology storage, and drop `stroke_origin` and the snapshot after
   the stroke. Target 20–30M faces in 8 GB.
5. SIMD kernels (glam `Vec3A` / packed SoA) and a GPU compute path for very
   large brushes.
6. Leaf-local mask re-evaluation, so painting a channel used by a layer mask
   updates live instead of at stroke end.

---

## 7. Texture layers (Substance Painter style), deferred

Planned on the same foundations:

* **Storage:** UDIM tiles split into 128² sparse virtual-texture pages, only
  allocated where painted, with dirty-page tracking (the texture analogue of
  per-leaf chunks).
* **Layer stack:** fill/paint layers with channels (base color, roughness,
  metallic, height, normal, ...). Each layer has a `MaskStack` using the
  **same type** as sculpt layers, evaluated in texture space with GPU compute.
  Noise is evaluated in rest-pose object space; mesh bakes are rendered to
  texture once.
* **Painting:** project the brush into UV pages through the PBVH
  (screen-space projection for stencils and alphas).
* **Speed:** compute-shader compositing per dirty page and per channel, with
  results cached.
* Vertex-level `channels` remain the fast path for sculpt masks; texture
  masks reuse the same evaluation code with a different sample domain.

---

## 8. UI: Mudbox layout, ZBrush feel (`crates/sculpt-app`)

**Stack:** Rust, **egui 0.36** for panels, **wgpu 30** for the viewport, one
process, one GPU device. Chosen over Qt (via cxx-qt) and Tauri/React because
the viewport and pen input must not cross a process or language boundary, and
one language keeps the app easy to extend. Qt remains viable later: the
engine is UI-agnostic.

**Frame loop (latency first).** Per frame, on the UI thread:
`keymap commands → panels → every pointer sample of the frame → raycast →
stroke spacing → dabs → dirty-leaf GPU upload → render → present`. Input and
its result land in the same frame. Safeguards:

* **Per-dab time budget (12 ms):** dabs that don't fit are queued for the next
  frame, so a huge brush on a huge mesh makes the stroke trail slightly rather
  than freezing the UI.
* **Strokes lift over gaps:** leaving the silhouette or jumping across a depth
  discontinuity restarts spacing instead of interpolating dabs across the gap
  (this was a 236 ms hitch before the fix).
* **Strokes can start off the mesh** and begin when the pen reaches it.
* **Background jobs** (load, subdivide, bake, large-mesh mask evaluation) run
  on a worker thread; the viewport shows a progress overlay meanwhile.
* **`SurfaceConfig::LOW_LATENCY`** presentation.

**GPU data path.** Positions, normals and an overlay scalar live in separate
vertex buffers in the engine's internal vertex order. The engine tracks dirty
leaves (`Document::take_geometry_dirty` / `take_scalar_dirty`), and since
every leaf owns a contiguous vertex range, an update is a few `write_buffer`
calls for exactly the touched ranges. Index buffers rebuild only when
`topology_id` changes (subdivide / load). The scene renders at 4× MSAA into an
offscreen texture displayed by egui.

**Viewport shading:** camera-relative studio clay (key/fill/rim/spec), overlay
tint for freeze / layer mask / any channel / pose weights, and a ZBrush-style
brush ring drawn on the surface at the true world radius.

**Layout (Substance Painter / Mudbox hybrid):** menu bar; Painter-style
**context toolbar** (active tool, size, strength, falloff, front-faces,
overlay, HUD toggle); right dock with **LAYERS** over **PROPERTIES**, both
with uppercase title bars. The layer stack (`crates/sculpt-app/src/layer_panel/`) shows each layer or
folder as a 36 px row: eye, disclosure, content thumbnail, mask thumbnail
(or a hover "+" slot), middle-elided name, solo, lock and strength column;
mask ops nest beneath their layer, then *Base*. Directly under the list sits
an add bar (+ Layer, Folder, Mask, Op, duplicate, merge, delete) that inserts
above the selection. The module is split by responsibility: `selection`
(multi-select model), `tree` (rows read from the document), `row`, `menus`,
`add_bar`, `dragdrop` (pure drop-zone math) with `drag_ui`, `breadcrumb`, and
`target` (what a stroke edits, and when it is refused; drives the viewport chip, breadcrumb and stroke guard), `switcher` (Tab), `thumbs` and `command`, the only path by which the UI changes layers, so undo stays
consistent and hotkeys, menus and buttons share one behaviour. **PROPERTIES** edits the selection (layer, mask base,
or a single effect: blend, opacity, source parameters, levels, blur, order)
and then the brush; selecting a Paint effect retargets Mask Paint to that
effect's channel. Bottom: Mudbox **tray** (Sculpt / Paint / Pose tools and a
Falloff tray of curve presets) and a status bar (tool hint, pen, paint target, face count). All icons are vector-drawn, so nothing depends on font
glyph coverage. Visual style: Inter typography (semibold
letter-spaced section titles), accent-filled sliders with round handles,
soft popup/window shadows, a camera-relative clay material (wrapped key,
hemisphere ambient, subsurface warmth, fresnel rim) on a vignetted, dithered
backdrop, an XYZ orientation gizmo, and a brush cursor that shows both the
radius and the falloff hardness ring.

**Tools wired to the engine:** Clay Buildup, Trim Dynamic, Move (topological
option), Smooth (and Shift-smooth), Freeze paint, Mask paint, Pose
(click a limb → topological mask with soft joint, drag → rotate or translate,
live preview below 1.5M verts, applied on release above that).

**Theming:** a theme is JSON with UI colors, viewport colors (background
gradient, clay, overlays, cursor, HUD) and metrics (font size, spacing,
corner radius, tray tile size). Four built-ins (Mudbox Dark, Painter Dark, Studio Light,
High Contrast); user themes load from the config folder and **hot-reload when
the file changes**; *Display ▸ Theme editor* edits live and saves JSON. The
chosen theme and per-tool settings persist between sessions.

**Extensibility:** every action is a `Command`; menus, tray and hotkeys all
dispatch commands; bindings are JSON (`keymap.json`, user overrides per
command). Adding a feature = one enum variant + one handler. A scripting layer
(Python via pyo3 or Lua) can dispatch the same commands later.

**Pen pressure:** three sources with automatic fallback — octotablet (feature
`tablet`: Windows Ink + Wayland; unmaintained since 2024-03, so isolated
behind a feature), pointer force from winit (Windows pen), mouse = 1.0. The
HUD and Object tab show which source is live.

### Measured (cloud VM, 4 cores, **software** Vulkan — lavapipe)

Synthetic 240 Hz pen strokes through the real path (`--test-strokes`):

| Mesh | Input + dabs, median | p95 | max | GPU upload / frame | UI panels |
|---|---|---|---|---|---|
| 98k faces | 2.1 ms | 5.1 ms | 8.1 ms | 0.9 MB | ~0.5 ms |
| 393k faces | 2.8 ms | 5.0 ms | 10.6 ms | 1.3 MB | ~0.5 ms |
| 6.3M faces | 12.4 ms | 16.4 ms | 20.6 ms | 4.7 MB | ~0.5 ms |

*Viewport render time here (0.09–2.6 s/frame) is lavapipe rasterizing on the
same 4 CPU cores and is not representative; a discrete GPU draws 12.6M
triangles in a few ms.* What these numbers do show: panels cost ~0.5 ms per
frame, uploads are proportional to the brush, not the mesh, and no frame
stalls past the budget. At 6.3M faces with a large brush on 4 cores dabs
saturate the budget (the stroke trails the pen); more cores scale this
linearly — see §6.

**Still to validate on real hardware:** input-to-photon latency, tablet
pressure on your OS/tablet, GPU frame time at 6M+ faces.

**UI roadmap:** dockable/undockable panels (egui_dock), stamp/stencil/falloff
trays, camera bookmarks, wireframe, matcap library, Transpose action-line
gizmo, undo-history panel, node view for mask stacks, multi-object list.

---

## 9. File format (v1)

Directory `name.sculpt/`:

```
project.json          { format: "sculpt-project", version: 1, generator,
                        mesh: { vertex_count, face_count, level,
                                base_positions, faces, rest_positions? },
                        freeze?, channels[], layers[], active_layer,
                        metadata{} }
blobs/*.bin           raw little-endian arrays
```

* Every blob reference has `{ path, dtype: f32|u32, components, count }`,
  and the loader validates sizes and refuses paths that escape the project.
* All per-vertex data is in canonical order.
* Layers are stored sparse: `indices` (u32) + `deltas` (f32×3), plus name,
  opacity, visibility, lock and the full `MaskStack` JSON.
* `metadata` is a free-form JSON map for UI state and app extensions, and is
  round-tripped.
* Planned: a single-file packaging (zip of the same tree), texture pages,
  and multires levels as additional blobs.

**Interop today:**

* OBJ in and out, preserving vertex order.
* `sculpt-cli import-channel <project> <name> <file>` takes raw `.f32` or
  text (`value` per line or `index,value`).
* Mask stacks are plain JSON that external tools can generate.

**Next:** EXR/PNG sampled through UVs, Houdini/Blender attribute exporters,
and a JSON noise-graph spec (the `NoiseParams` schema already serves as v0).

---

## 10. Known limitations / open decisions

* Layer deltas are **object-space**; posing rotates them explicitly. A
  tangent-space option would let detail follow *sculpted* base changes too.
* On a layer at strength below 100%, the visible stroke is scaled by the
  strength (Mudbox behaviour). Move on such a layer converges to the cursor
  over several updates rather than instantly.
* Structural edits (subdivide, add/remove/flatten a layer, imports) clear
  undo history; property changes (slider, mask edits) are not yet undoable.
* Mask stacks that read painted channels refresh at stroke end, not per dab.
* Only one subdivision direction for now (no stepping down levels).

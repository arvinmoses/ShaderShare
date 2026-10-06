# Substance 3D Painter layer-stack UX: spec for a sculpt-layer clone

Status: research and spec. Target is the egui layer panel for sculpt delta layers
(`SculptLayer { opacity, visible, locked, mask: MaskStack }`, freeze, channels,
bakes). See `../ARCHITECTURE.md`.

**Confidence notes.** Verified in a second pass (see "Verified against" at the end)
by reading the official Experience League pages directly. `helpx.adobe.com`
returns "Access Denied" (the site's own bot protection, not the proxy), so the
Experience League mirror was used instead. Facts marked **[doc]** are
confirmed by those pages. **[obs]** marks long-standing SP behaviour seen in
tutorials that the pages do not state. **[spec]** marks values we choose.
Pixel sizes are always **[spec]**. Menu ordering changes between SP
versions, so the ordering given here is the order *we* will ship, modelled on
SP's grouping.

---

## 1. Anatomy of a layer row

### 1.1 SP reference (what the row contains)

One row per layer, folder, or effect. Bottom of the list composites first; the top
row composites last **[doc]**. Left to right, a layer row contains:

| # | Element | Notes |
|---|---------|-------|
| 1 | **Visibility toggle** (eye or checkbox) | Far left, in its own column. Click toggles. Dragging down the column toggles many rows (Photoshop convention, **[obs]**). |
| 2 | **Indent and disclosure arrow** | Only on folders, and on layers that have effects or a mask stack. Indent is about 1 thumbnail width per nesting level. |
| 3 | **Content thumbnail** | Square, about 32–40 px. Shows the layer's result *for the channel picked in the top-left view dropdown* **[doc]**. Paint layers carry a small brush badge and fill layers a bucket badge **[obs]**. Instanced layers carry an instance badge **[doc: layer instancing]**. |
| 4 | **Mask thumbnail** | Shown only if a mask exists. Same size, directly to the right of the content thumbnail. Greyscale. **Alt+LMB**: view the mask in the viewport. **Shift+LMB**: disable the mask, which then shows a red X / crossed state **[doc]**. |
| 5 | **Name** | Fills the remaining width. Double-click to rename inline **[obs]**. Elided with "…". |
| 6 | **Blend mode** (short label, e.g. `Norm`, `Mul`, `Ovr`) | Right-aligned. Click opens the blend list. It is **per channel**: it shows the mode for the channel in the view dropdown **[doc]**. Right-click → *Apply to all channels* **[doc 8.2]**. |
| 7 | **Opacity** (`100`) | Right-aligned under or next to the blend label. Click-drag scrubs, click types a value. Per channel **[doc]**. |

When a layer is expanded, the **effect rows** sit under it, indented. There is an
effect stack for the content and another for the mask **[doc]**. Each effect row
has its own eye, a type icon (paint, fill, generator, levels, filter, anchor,
compare mask, color selection), a name, and blend/opacity. Mask effects are drawn
under a "mask" sub-header and use greyscale thumbnails **[obs]**.

### 1.2 States

| State | SP look | Our spec |
|-------|---------|----------|
| Selected | Whole row filled with a lighter or accent tint. The **active thumbnail** (content *or* mask) gets an accent outline, and that is the paint target **[obs]**. | Row fill `accent@25%`, plus a 2 px accent frame around the active sub-target thumbnail. |
| Multi-selected | Same fill on every selected row. Only one row is the "active" one (shown by its thumbnail frame). | Same. The active row also gets a 2 px accent bar on its left edge. |
| Hover | Subtle lighten. Thumbnails show a pointer cursor. | `bg + 6%`. A tooltip on a thumbnail gives the click modifiers. |
| Hidden | Eye off. Row text and thumbnails dimmed to about 50% **[obs]**. | Eye off. Row alpha 0.45. Children inherit the dim. |
| Locked | SP has no per-layer lock. | **New:** padlock in the icon column. Strokes are refused, and the cursor shows ⊘ over the viewport. |
| Solo | Not native in SP (it is a common feature request) **[doc: community]**. | **New:** "S" pill, accent-filled when on. Every other row dims and shows a hollow eye. |
| Mask disabled | Red cross over the mask thumbnail **[obs]**. | Red diagonal over the mask thumbnail. The tooltip reads "Mask disabled (Shift-click)". |
| Mask viewed | Viewport shows the mask in greyscale. The thumbnail gets a highlight **[doc]**. | Mask thumbnail has a yellow frame. The viewport shows a banner "Viewing mask: <layer> — Esc to exit". |
| Dragging | Ghost row follows the cursor. A horizontal **insertion bar** marks above, between, or below. Dropping onto a folder highlights the folder **[doc]**. | See §2.4. |
| Instance | Instance badge on the thumbnail **[doc]**. | Defer. |

### 1.3 Row geometry for egui [spec]

- Row height: **36 px** for a layer with thumbnails and **24 px** for effect or mask-op rows. A density toggle sets 28/20.
- Columns from left: eye 20 · lock/solo 18 · indent (16 × depth) · disclosure 12 ·
  content thumb 28 · mask thumb 28 (or an empty slot that shows "+" on hover) · name (flex) · strength 44 (right-aligned, monospace).
- Thumbnails for sculpt rows: content = a mini matcap render of the delta
  (or a heat swatch of |delta| at fixed size). Mask = a greyscale preview of the
  mask's per-vertex values projected to UV or to a fixed sphere preview.
  Re-render lazily. Never block a frame on them.
- Font: 13 px name and 11 px numeric. Never wrap. Elide in the middle (`Wrinkle…_v2`).
- Blend-mode column is **dropped** for sculpt layers (see §7). The strength value moves into its slot.

---

## 2. Hierarchy rules

### 2.1 Node kinds (SP)

- **Paint layer**: a brush target. **Fill layer**: a procedural or material
  source that cannot be painted directly. **Folder**: holds layers. It has its own
  blend/opacity (Pass Through is available) and **can carry its own mask**
  **[doc]**.
- **Mask**: at most one per layer or folder. It is a *child stack*: in the
  expanded view it lists mask effects in order. The bottom effect evaluates first.
- **Effects** **[doc]**: Generator, Paint, Fill, Levels, Compare Mask, Filter,
  Anchor Point, plus Color Selection. Effects go on the **content** stack or the
  **mask** stack.
- **Geometry mask** **[doc 2021.1]**: per-layer visibility by mesh/UV tile. It is
  separate from the mask stack.

### 2.2 Nesting rules

1. Folders nest to any depth. Layers live in folders. Effects never contain children.
2. Masks are attached to exactly one layer or folder. A mask never appears as a free row.
3. Mask effects stay inside their mask. Content effects stay inside content. Dragging an effect from a mask to content is **allowed** in SP (it converts semantics) **[obs]**. For us, only mask ops are valid in masks, and the drop is refused with a red bar.
4. Effects can be copied and pasted across layers and stacks, with multi-select **[doc 2021.1]**.

### 2.3 Collapse and expand

- Disclosure arrow click: toggle one level.
- **Alt+click arrow** **[spec]**: toggle the whole subtree recursively.
- A new mask or effect auto-expands its parent so the user sees what was added (SP does this **[obs]**).
- The collapsed state is persisted per node in the project file.

### 2.4 Drag and drop [doc + spec]

| Drop zone (cursor y within target row) | Indicator | Result |
|---|---|---|
| Top 25% | 2 px accent line at the row's top edge, indented to the target depth | Insert above, as a sibling |
| Bottom 25% | Line at the bottom edge | Insert below, as a sibling |
| Middle 50% of a **folder** | Folder row outlined, 1 px accent box | Append into the folder (top of its children) |
| Middle of a **layer**, while dragging a mask/effect | Mask thumb outlined | Move or copy into that layer's mask |
| Invalid | Red line plus a ⊘ cursor | Nothing |

- **Ctrl+drag = duplicate** **[doc]**.
- Hover over a collapsed folder for 600 ms to spring it open.
- Near the top or bottom edge of the panel, the list auto-scrolls.
- Dragging a multi-selection moves the rows as a block and keeps their order.
- Esc cancels the drag.

---

## 3. Add flows

### 3.1 Toolbar (SP)

SP puts a row of icon buttons at the top right of the Layers panel, next to
the channel/view dropdown **[doc]**. Verified order, left to right: *Add effect*,
*Create mask* ▾ (white, black, bitmap, color selection, height combination),
*Create new paint layer*, *Create new fill layer*, *Add new smart materials*,
*Add new folder*, *Delete layer*. The paint and fill buttons insert **above the
current selection** **[doc]**; the pages do not state the insertion point for
the other buttons. Buttons with ▾ open a menu that lists the same items as the
matching right-click submenu.

**Ours** [spec]: put the bar at the **bottom of the panel, directly under the
list**, so the cursor travels a short way from the last-touched row. Left to
right: `+ Layer` · `+ Folder` · `Mask ▾` · `Op ▾` · `⧉ Duplicate` · `🗑`.
The bar is 24 px tall with icon and a 2–3 letter label. Each button also has a
hotkey (§5).

### 3.2 Right-click menus (our shipping order, grouped like SP)

SP shows a context menu on any row. Its items depend on the kind of row clicked
**[doc]**. Separators are shown as `---`.

**Layer row**
```
Add mask              ▸  Black (reveal by painting) | White (hide by painting) |
                         From bake ▸ Curvature / Cavity / AO / Thickness |
                         From noise ▸ fbm / ridged / cellular / … |
                         From freeze | From selection
Add mask op           ▸  (only if mask exists; same as Mask menu below)
---
Duplicate              Ctrl+D
Copy / Paste           Ctrl+C / Ctrl+V
Paste mask into this
Group into folder      Ctrl+G
---
Merge down             Ctrl+E
Flatten into base
---
Lock / Unlock          L        Solo   S        Hide/Show  H
Rename                 F2
---
Delete                 Del
```
SP's own *Create mask* submenu lists **Add white mask, Add black mask, Add bitmap
mask, Add mask with color selection, Add mask with height combination** **[doc]**.
In our menu, "From bake" and "From noise" take the place of SP's color-selection
and bitmap variants. SP achieves the same thing with a black mask plus a
Generator effect, which needs two steps. We do it in one.

**Mask thumbnail / mask header row**
```
Add op ▸   Paint | Fill (constant) | Noise ▸ … | Bake ▸ Curvature/Cavity/AO/Thickness |
           Levels | Blur | Invert | Gradient | Direction
---
View mask              Alt+click
Disable / enable mask  Shift+click
Invert mask
Clear mask (to black / to white)
Copy mask   /   Paste into mask       (SP: "Copy mask content", "Paste into mask" [doc])
Mask → freeze  /  Freeze → mask
---
Delete mask
```

**Mask op (effect) row**
```
Blend into mask ▸  Replace | Multiply | Add | Subtract | Max | Min | Screen | Overlay
Strength…
---
Move up / Move down    Ctrl+[ / Ctrl+]
Duplicate / Copy / Paste
Hide / Show
---
Delete
```

**Folder row**: the same as the layer menu, plus *Ungroup* (Shift+Ctrl+G) and
*Flatten folder* (SP 12: *Flatten group*, Ctrl+M, which makes a merged copy and
disables the source **[doc 12.0]**). *Merge down* is removed.

**Empty area below rows**: `New layer`, `New folder`, `Paste`.

### 3.3 Add-mask variants: what each one creates

| Variant | Creates | Selection after |
|---|---|---|
| Black | Mask stack `[Fill 0]` | **Mask** becomes the paint target. Brush is set to "reveal" |
| White | `[Fill 1]` | Mask becomes the paint target. Brush is set to "hide" |
| From bake X | `[Bake X, Levels]`. If X has no bake yet, it is baked first and a progress pill shows on the row | Levels op is selected, so the Properties panel shows the black/white points right away |
| From noise X | `[Noise X, Levels]` | Noise op is selected |
| From freeze | `[Paint ← copy of freeze]` | Mask selected |

---

## 4. Selection model and Properties panel

### 4.1 Model

- A **single active item** plus an optional multi-selection set. An item is one of
  `Layer(id)`, `LayerMask(id)`, `MaskOp(id, idx)`, `Folder(id)`, `Base`.
- Click on the **content thumb or name**: active = `Layer`, and the paint target becomes the layer deltas.
- Click on the **mask thumb**: active = `LayerMask`, and the paint target becomes the mask's top Paint op (one is auto-created if missing, which matches SP, where painting a mask needs a Paint effect **[obs]**).
- Click on a **mask op row**: active = that op. If it is a Paint op, it is also the paint target. Otherwise strokes are refused and the cursor shows ⊘ with the tooltip "Select a Paint op to paint".
- Ctrl+click toggles membership. Shift+click selects a range **[doc]**.
- Multi-selection is only for bulk ops (delete, group, hide, move). The Properties panel then shows shared fields with mixed values as "—".

### 4.2 Properties panel by selection

In SP, Properties shows whatever is selected: brush/material for a paint layer,
the material for a fill layer, and the effect parameters for an effect
**[doc]**.

| Active | Properties shows |
|---|---|
| Layer | Header: `● Layer "Wrinkles" — painting DELTAS` · strength slider · lock · visibility · delta stats (verts touched, max displacement) · *Mask: none / 3 ops [go to]* |
| LayerMask | Header: `◐ Mask of "Wrinkles" — painting MASK` · a compact ordered list of ops (each a clickable row) · invert, enabled |
| MaskOp | Header breadcrumb: `Wrinkles › Mask › Levels`. Op params: Noise (scale, octaves, seed, type), Bake (type, radius, samples, *Rebake*), Levels (in-black, in-white, gamma, histogram), Blur (radius, iterations), Paint (clear, fill value). Op blend mode and strength go at the top |
| Folder | Strength, visibility, folder mask summary |
| Base | "Base mesh — strokes write to base" · freeze summary |

### 4.3 Making the paint target unmistakable

SP shows the target only by the thumbnail frame, and that is easy to miss. That
is a known pain point. We use **three redundant cues**:
1. Accent frame on the target thumbnail in the layer row.
2. Breadcrumb at the top of Properties (`Wrinkles › Mask`).
3. A **viewport HUD chip** next to the brush cursor reading `→ Wrinkles · Mask`, coloured purple for mask, orange for delta, and blue for freeze.

Hotkey **`M`** (in our app) toggles the target between the layer's delta and its
mask without moving the cursor.

---

## 5. Viewing aids and hotkeys

| Action | SP | Ours |
|---|---|---|
| View mask in viewport | Alt+LMB on mask thumb **[doc]** | Same. Also **Alt+M** in the viewport for the active layer. Esc or a repeat exits |
| Disable mask | Shift+LMB on mask thumb, or RMB → Toggle mask **[doc]** | Same |
| Channel view | Top-left dropdown in the layer panel sets the per-channel display context **[doc]**. In the viewport, C cycles channels and M returns to material **[obs]** | **Overlay dropdown** in the viewport corner: Shaded · Mask(active) · Freeze · Delta heat · Bake ▸ … `C` cycles it at the cursor |
| Solo | Not native | **S** on the row, or Alt+click the eye (Photoshop/Blender convention). Solo shows base + that layer at its strength |
| Isolate geometry | Geometry Mask **[doc]** | Out of scope (use the existing hide/visibility tools) |
| Multi-select | Ctrl/Shift+click **[doc]** | Same |
| Copy / paste | Ctrl+C / Ctrl+V for layers and effects **[doc 2021.1]** | Same, including mask ops between layers |
| Duplicate | Ctrl+D, RMB → Duplicate, Ctrl+drag **[doc]** | Same |
| Group | Ctrl+G **[doc]** | Same. Shift+Ctrl+G ungroups |
| Merge / flatten | Ctrl+M flattens a group into a copy and disables the source **[doc 12.0]** | Ctrl+E merges down (ours; not an SP binding). *Flatten into base* (menu only, confirm). Ctrl+M flattens a folder to a new layer and hides the source |
| Rename | Double-click name **[obs]** | Double-click or F2. Enter commits, Esc cancels |
| Delete | Del, or the trash button **[doc]** | Del. Undo restores everything, including mask ops |
| New layer/folder | Toolbar | Ctrl+Shift+N for a layer, Ctrl+Shift+F for a folder |
| Strength scrub | Opacity field drag | Drag the number. **In viewport: hold `O` and drag horizontally** to scrub the active layer's strength at the cursor |

Hotkeys must work while the cursor is over the **viewport** too, acting on the
active item. Otherwise users are forced back to the panel.

---

## 6. What gives SP its feel (non-negotiables for 9/10)

1. **Two thumbnails, side by side, are the primary controls.** Content and mask
   are separate, equally sized, clickable targets on the same row. You can see the
   whole state of a layer (what, where, on/off) at a glance with no expansion.
2. **The mask is a stack.** Masks are built from ordered, visible, toggleable
   operations (generator → levels → paint), not a single bitmap. This is what
   makes SP's procedural-plus-hand-paint workflow possible.
3. **Density with discipline.** Rows are compact but separated by a 1 px
   divider. There is one accent colour. Numbers are right-aligned in a fixed
   column, so scanning down a column of strengths works.
4. **Modifier-click vocabulary on thumbnails.** Alt = look, Shift = toggle,
   Ctrl = add to selection. It is consistent, and it is shown in tooltips.
5. **Insert-above-selection.** Every add happens next to where you are
   looking. Nothing appends at the bottom of a long list.
6. **Context menus mirror the toolbar.** Every menu item names its hotkey, so
   users learn hotkeys from the menus.
7. **Properties follow selection exactly.** Click a Levels op and the panel
   shows Levels. There are no modes to switch.
8. **Viewport feedback is instant.** Changing a mask op or a strength updates the
   viewport within one frame (we already composite in about 5 ms, see ARCHITECTURE).
9. **Non-destructive by default.** Hide, disable mask, and solo never alter data.
   Flatten and merge are explicit, and SP 12 even keeps the source group.

---

## 7. Mapping: SP concept → sculpt equivalent → design

| SP concept | Sculpt equivalent | Recommended design |
|---|---|---|
| Paint layer | `SculptLayer` (sparse deltas) | Row with delta thumb and strength. Brushes write here |
| Fill layer | **Procedural displacement layer** (noise along normal, deferred) | Show a "Fill" badge and refuse brush strokes. Optional v2 |
| Folder | Folder of sculpt layers; folder strength multiplies children | Same row. Folder mask multiplies children's masks |
| Blend mode (per layer) | **N/A**. Deltas are additive in rest space. "Multiply" and similar have no geometric meaning | **Drop it.** Replace the column with **strength**, and allow strength values from −1 to 2 (negative inverts the layer, >1 exaggerates). If a mode is needed later, offer only *Add* (default) and *Replace-to-layer* (lerp the base toward base+delta) |
| Opacity | `opacity` = strength slider (Mudbox) | Right column, numeric with drag-scrub, 0–100 display |
| Channels (Base color, Roughness, Height…) | **N/A** for deltas. Closest is *view overlays* | Repurpose the channel dropdown as the **viewport overlay** selector (Shaded / Mask / Freeze / Delta heat / Bakes) |
| "Apply to all channels" | N/A | Drop |
| Layer mask | `SculptLayer.mask: MaskStack`, per-vertex 0–1 | Second thumbnail. Identical interaction |
| Mask effects: Fill | `Fill(const)` op | Same |
| Paint effect | `Paint` op (per-vertex channel) | Same. It is the default target when the mask thumb is clicked |
| Levels | `Levels` op | Same, with a histogram over vertex values |
| Filter (blur etc.) | `Blur` op (mesh-Laplacian smooth over N rings) | "Blur" in the op menu |
| Generator (curvature/AO/thickness/position) | `Bake` op (curvature, cavity, AO, thickness), plus Gradient/Direction | Bake op with *Rebake* button and stale badge ⟳ when topology changes |
| Fill with procedural (noise) | `Noise` op (fbm/perlin/ridged/turbulence/cellular), in rest space | Noise op |
| Effect blend mode inside a mask | `MaskStack` blend modes (valid: these are scalar fields) | **Keep** blend modes here: Replace, Multiply, Add, Subtract, Min, Max, Screen, Overlay |
| Compare mask / color selection | Selection → mask; freeze ↔ mask | "From selection" / "From freeze" variants |
| Anchor point | Reference to another layer's mask or a channel | v2: an "Ref" op that reads another mask or a named channel |
| Smart mask (preset) | Saved `MaskStack` JSON | "Mask presets" shelf. Drag one onto a mask thumb |
| Smart material | Saved layer + mask (detail preset) | v2 |
| Layer instancing | Shared mask stack across layers | v2 |
| Geometry mask | Hide/visibility of mesh parts | Out of scope here |
| Flatten group | Flatten folder → new layer | Ctrl+M, hides the source |
| (none) | **Freeze** (Mudbox) | Pinned **Freeze** pseudo-row at the top of the panel with a blue thumb. Clicking it makes freeze the paint target. It has its own mask-op context menu (Freeze from bake/noise) |
| (none) | **Base** | Pinned **Base** row at the bottom. It cannot be moved, deleted, or masked |
| (none) | **Lock** | Padlock column |

---

## 8. Cursor-local interaction (no cross-screen travel)

The user dislikes crossing the screen between viewport and panel. Rules:

1. **Viewport pie/radial at cursor** (hold `Space` or `Q`): New layer · Add
   mask ▸ · Toggle target (layer/mask) · Solo · View mask · Hide · Strength scrub.
   Release over an item to execute.
2. **Layer quick-switcher at cursor** (`Tab` in viewport): a compact popup copy
   of the layer list (names, eyes, strengths), anchored at the cursor. Type to
   filter, arrow keys to move, Enter to activate. It closes on mouse-out.
3. **Right-click in the viewport** opens the *same* context menu as the active
   layer row, anchored at the cursor. One menu definition serves both places.
4. **Hotkeys act on the active item from the viewport** (M, Alt+M, S, H, O-drag, Ctrl+D, Del).
5. **HUD chip** at the brush cursor shows target and strength (see §4.3). Scrolling with Alt over the chip changes strength.
6. **Menus open at the click point**, never at a fixed panel location. Submenus open toward the screen centre.
7. **Properties popover**: Ctrl+click a mask-op row (or `P` in the viewport)
   opens its parameters in a floating window next to the row or cursor, so the
   user does not have to reach for the docked Properties panel.
8. Docking: the layer panel and Properties should dock **on the same side**,
   stacked vertically, with Properties directly *below* the list. Then
   selection → parameters takes about 200 px of vertical travel, not a full screen width.

---

## 9. Scoring rubric (10 points total)

Score each criterion from 0 to 10, then multiply by its weight. Total = Σ(score ×
weight) / 100, which gives a value from 0 to 10.

### 9.1 Row anatomy (weight 20)
- **3**: A row has a name and a visibility checkbox. There is no mask thumbnail, strength sits in a separate panel, and states are hard to tell apart.
- **6**: Eye, name, and strength are on the row. A mask indicator exists but is not clickable as a target. Hidden and selected states are visible, but locked/solo/mask-disabled are missing or look the same.
- **9**: Eye, lock, content thumb, mask thumb, name, and right-aligned strength are all on one row. Every state in §1.2 is visually distinct. Thumbnails are clickable targets with modifier tooltips.
- **10**: All of 9, plus live thumbnails that update within about 1 s. Density toggle. Paint target framing is readable in a 50%-scaled screenshot.

### 9.2 Hierarchy (weight 20)
- **3**: A flat list. Masks are edited elsewhere.
- **6**: Folders exist, and the mask has ops in a list, but ops cannot be reordered or toggled. Drag-drop has no indicator.
- **9**: Folders nest. The mask stack expands under its layer with ordered, toggleable, reorderable ops. Drag-drop shows above/below/into indicators, refuses invalid drops, and Ctrl+drag duplicates.
- **10**: All of 9, plus spring-open folders, auto-scroll, recursive Alt-expand, and multi-row drag that keeps order.

### 9.3 Context menus and add-flows (weight 15)
- **3**: Adding a mask or op requires the main menu bar or a separate panel.
- **6**: A right-click menu exists on layers only. The add-mask types are generic (no bake/noise presets).
- **9**: Kind-specific menus (layer, mask, op, folder, empty area) list hotkeys. The add-mask variants include bake and noise in one step. Inserts go above the selection, and the new item is selected and expanded.
- **10**: All of 9, plus the same menu definitions are reachable from the viewport at the cursor.

### 9.4 Selection and Properties coherence (weight 15)
- **3**: Properties does not follow selection, or the paint target is ambiguous.
- **6**: Properties follows layer selection, but selecting a mask op does not show its params. The target is only implied.
- **9**: Every selection kind in §4.2 drives Properties. The breadcrumb, thumb frame, and HUD chip all agree on the paint target. Strokes on a non-paintable op are refused with feedback.
- **10**: All of 9, plus multi-select shows shared fields with mixed-value display. Ctrl+click opens a popover.

### 9.5 Locality: no cross-screen travel (weight 15)
- **3**: Common actions need trips between the panel, menu bar, and properties on opposite sides.
- **6**: The add bar sits next to the list. Some hotkeys exist, but they don't work while hovering the viewport.
- **9**: Menus open at the cursor. Viewport hotkeys cover target toggle, view mask, solo, strength scrub, and duplicate. Properties is docked under the list.
- **10**: All of 9, plus the viewport radial and the layer quick-switcher. All tasks in §9.7 are possible without the pointer leaving a 400 px radius.

### 9.6 Legibility (weight 15)
- **3**: Text is truncated or overlapping, and there is no alignment. Many accent colours compete.
- **6**: Rows are readable, but numbers are not column-aligned, and dark-on-dark states are hard to see.
- **9**: Single accent, aligned numeric column, middle-elided names, 1 px dividers, and dimmed hidden rows. Readable at 100% in a 1080p screenshot.
- **10**: All of 9, plus purpose colours (delta orange / mask purple / freeze blue) used consistently across row, HUD, and overlay. Contrast is AA or better.

### 9.7 Scripted task checklist (blind review from screenshots)

The reviewer gets a screenshot before and after each task, plus a screenshot of
any menu that was opened. Score each task pass/partial/fail, and note how many
clicks it took and how far the pointer travelled. Every task must be doable using
only the layer area, its menus, and viewport hotkeys.

1. **Create** a new layer named "Pores" above the current layer, inside folder "Face". (Expect: insert-above, inline rename.)
2. **Add a curvature mask** to "Pores" in one menu action. (Expect: the mask thumb appears, ops `[Bake Curvature, Levels]` are visible and expanded, and Levels is selected in Properties.)
3. **Add a noise op** (fbm) on top of that mask stack with blend *Multiply*. (Expect: op row with its blend label.)
4. **Hand-paint** into the mask: switch the target to the mask's Paint op. (Expect: purple frame on the mask thumb, breadcrumb `Pores › Mask › Paint`, HUD chip purple.)
5. **View the mask** in the viewport via Alt+click, then exit. (Expect: greyscale viewport, banner, framed thumb.)
6. **Disable the mask** with Shift+click, then re-enable it. (Expect: red diagonal, then restored.)
7. **Solo** "Pores", set its strength to 40% by drag, then un-solo. (Expect: other rows dimmed, number reads 40.)
8. **Reorder**: drag "Pores" below "Wrinkles", and Ctrl+drag a copy into folder "Body". (Expect: insertion bar, folder outline, and copy named "Pores copy".)
9. **Copy the mask** from "Pores" and paste it into "Wrinkles". (Expect: the mask thumb appears on Wrinkles with identical ops.)
10. **Lock** "Wrinkles", attempt a stroke (refused, with feedback), then **flatten** folder "Face" (Ctrl+M → new layer, source hidden).

---

## 10. Implementation notes (egui)

- Build the panel as a **flattened visible-row list** recomputed each frame from
  the tree and the collapse state. Each row stores `{node_ref, depth, kind}`.
  This makes range-select, drop-zone math, and keyboard nav trivial.
- Each row is one `ui.allocate_exact_size(…, Sense::click_and_drag())`, with
  sub-rects hit-tested manually (eye, lock, thumbs, name, strength) so modifier
  logic stays in one place.
- Drag: keep a `DragState { rows, ghost_offset, target: DropTarget }`. Draw the
  indicator in a `LayerId::new(Order::Foreground, …)` painter so it sits above
  rows.
- Context menus: `response.context_menu(|ui| build_menu(ui, MenuKind::…, &mut cmds))`.
  Reuse the same `build_menu` for viewport right-click with
  `egui::Area::new(...).fixed_pos(cursor)`.
- All actions emit `Command`s onto the existing undo stack. The UI never mutates
  `Document` directly.
- Thumbnails: an offscreen render queue (≤ 2 per frame) keyed by
  `(node, revision)`, served as egui `TextureHandle`s.

---

## Sources

- Use the Layer Stack (helpx): https://helpx.adobe.com/substance-3d-painter/using/layer-stack.html (also /substance-painter/using/layer-stack.html)
- Layer stack (Experience League): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/layer-stack
- Masking and effects: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/masking-and-effects ; https://helpx.adobe.com/substance-3d-painter/interface/layer-stack/masking-and-effects.html
- Creating layers: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/creating-layers
- Managing layers (multi-select, drag bar, group): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/managing-layers
- Layer instancing: https://helpx.adobe.com/substance-3d-painter/interface/layer-stack/layer-instancing.html
- Effects: https://helpx.adobe.com/substance-3d-painter/features/effects.html
- Blending modes: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/layer-stack/blending-modes
- Release notes 2021.1 (geometry mask, effect copy/paste, multi-select): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/old-versions/version-2021-1-7-1-0
- Release notes 8.2 (Apply to all channels): https://helpx.adobe.com/substance-3d-painter/release-notes/version-8-2.html
- Release notes 12.0 (Flatten, Ctrl+M, export from layer stack): https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/release-notes/version-12-0
- Shortcuts: https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/interface/settings/shortcuts
- Interface overview (Properties follows selection): https://helpx.adobe.com/substance-3d-painter/using/interface-overview.html
- Community request, isolate layer (solo not native): https://community.adobe.com/t5/substance-3d-painter-ideas/feature-suggestion-isolate-layer/idi-p/12850398
- Community custom shortcut set (mask toggle/view habits): https://github.com/cbuliarca/BCM_SubstancePainter_shortcuts/blob/master/README.md
- Adobe Substance magazine, smoother texturing UX: https://www.adobe.com/products/substance3d/magazine/a-smoother-texturing-experience-with-substance-3d-painter

---

## Verified against (second pass)

Pages read in full from Experience League. Corrections to the first draft are
listed under each.

- **Layer stack**: row = eye, content thumbnail, mask thumbnail (greyscale),
  name, opacity, blend mode, both **per channel** (top-left dropdown picks the
  channel). Paint, Fill and Folder layer types. Bottom layer draws first.
  Correction: toolbar order is as in 3.1 above. Our earlier order (smart
  material, smart mask first) was wrong.
- **Masking and effects**: Alt+LMB on a mask thumbnail isolates it in the
  viewport. Shift+LMB disables it temporarily and toggles it back. Mask
  right-click menu has copy, paste and invert. Re-adding or removing a mask
  destroys its effects. Ctrl while dropping a fill layer creates a mask at once.
  A line under each thumbnail shows effects: **grey = none, red = at least one**.
  Smart masks: Ctrl while dropping overwrites the effect list.
- **Managing layers**: drag shows a bar for the destination. Dropping on a
  folder nests. Multi-select with Ctrl/Cmd+click and Shift+click range. Group
  with right-click ▸ Group Layers or Ctrl+G.
- **Creating layers**: duplicate by right-click, Ctrl+D, or Ctrl+drag. Dropping
  assets (materials, smart materials, effects) onto the stack creates layers.
  Insertion point is not specified for drops.
- **Flatten layers**: Ctrl+M = merge selection. Flatten makes a new fill layer
  and **disables** (does not delete) the source. Only visible layers count.
  Correction: there is **no Ctrl+E merge down** in SP. It is our own shortcut.
- **Layer instancing**: paste as instance (right-click, or Ctrl+Shift+V per this
  page). Source and target icons act as navigation buttons. Only the source can
  be edited. Cycles are refused. The shortcuts page instead lists Ctrl+Shift+C/V
  as copy/paste layer *content*, so the two pages disagree; we do not copy
  either binding blindly.
- **Shortcuts**: Ctrl+C/X/V, Delete, Ctrl+D, Ctrl+G, Ctrl+Shift+C/V all work
  **only while the mouse is over the layer stack** **[doc]**. Quick-mask edit:
  U toggle, Y clear, I invert (global).

### Not verified

- `helpx.adobe.com` pages (blocked by the site), release-notes pages 2021.1, 8.2
  and 12.0 (not re-read this pass), row pixel sizes, exact right-click menu
  order, whether new items auto-expand, drop-zone proportions, spring-open timing
  and auto-scroll. These stay **[obs]** or **[spec]**.
- Screenshots: not downloaded, so row geometry is unconfirmed visually.
- `artstation.com` and `polycount.com` return 403 from a Cloudflare challenge,
  not the proxy; they were not usable.

### Implication for our design

SP's layer-stack hotkeys only work over the layer panel. The cursor-local rule
(hotkeys while hovering the viewport) is therefore a deliberate improvement,
not a clone.

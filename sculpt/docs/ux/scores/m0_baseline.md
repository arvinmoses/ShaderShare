# M0 baseline: blind UX score

Source: `m0_baseline.png` only, scored against the rubric in `../substance_painter_layer_ux.md` §9. No source code was read. Interaction criteria are scored on the affordances the screenshot shows. Anything that needs interaction to confirm is listed under "Not verifiable".

## Scores

| # | Criterion | Weight | Score | Weighted | Evidence (from screenshot) |
|---|-----------|-------:|------:|---------:|----------------------------|
| 9.1 | Row anatomy | 20 | 5 | 1.00 | Layer row "Detail" has a disclosure arrow, eye, content thumb (a generic sphere, not a delta render), name, a small mask glyph, a padlock, and a boxed `85%`. The mask is a tiny icon, not a second same-size thumbnail. There is no solo control. Op rows show the eye, a type icon, the name, and a right-aligned blend+value (`Scrn 60`, `Norm 100`). The selected state is clear (blue fill plus left bar). Hidden, locked-active, solo, and mask-disabled states are not shown. "Black mask" has no eye. |
| 9.2 | Hierarchy | 20 | 5 | 1.00 | "Detail" expands into ordered op rows (Top light, Breakup, Black mask). Op rows have eyes, so they can be toggled, and Properties has Order ↑/↓ buttons, so they can be reordered. There is no "Mask" sub-header, so it is unclear whether the ops belong to the content stack or the mask stack. No folder is visible. Base is pinned at the bottom. There is no Freeze row. Drag-drop indicators cannot be seen. |
| 9.3 | Context menus and add-flows | 15 | 4 | 0.60 | There is an icon-only toolbar at the top of the list (+, mask, effect, merge/down, trash). None of the buttons has a label or a ▾ to show variants. No context menu is shown, so there is no evidence of kind-specific menus, hotkey labels, or bake/noise one-step mask variants. |
| 9.4 | Selection and Properties coherence | 15 | 5 | 0.75 | Properties follows op selection ("PROPERTIES — DETAIL MASK EFFECT" shows the Noise params for Breakup), which is a good sign. The paint target is ambiguous, though. A noise op is selected, but the status bar reads "Sculpting on Detail". There is no thumbnail frame, no breadcrumb (`Detail › Mask › Breakup`), and no HUD chip. A Brush section is appended under the op params, so the panel does not show only what is selected. |
| 9.5 | Locality | 15 | 5 | 0.75 | Properties is docked directly under the layer list, which matches §8.8, and the add bar is adjacent to the list. The Overlay dropdown sits in the top toolbar at the far right, not in a viewport corner. The viewport has no HUD chip at the brush cursor and no visible radial or quick-switcher. |
| 9.6 | Legibility | 15 | 6 | 0.90 | The theme uses a single blue accent and rows have 1 px dividers, so it reads cleanly at 100%. The numeric column is inconsistent: `85%` is a boxed field, `Scrn 60` and `Norm 100` are small grey text, and Properties shows opacity as `1.000` rather than 100. Op-row text is small and low contrast. There are no purpose colours (delta orange, mask purple, freeze blue). The viewport tint in "Layer mask" overlay mode is blue and orange, not a readable greyscale mask. |
| | **Weighted total** | 100 | | **5.00 / 10** | Σ(score × weight) / 100 = (100 + 100 + 60 + 75 + 75 + 90) / 100 |

## Top 8 gaps, ranked by score impact

| Rank | Gap | Criteria hit | Est. gain |
|-----:|-----|--------------|----------:|
| 1 | The mask is a tiny glyph, not a second same-size clickable thumbnail beside the content thumb. It needs Alt (view) and Shift (disable) modifier tooltips and a 2 px accent frame on whichever thumb is the active paint target. | 9.1, 9.4 | +0.8 |
| 2 | Mask ops are not grouped under a "Mask" sub-header, so content effects and mask effects cannot be told apart. No folder is visible, and there is no drag insertion, into, or refuse indicator. | 9.2 | +0.6 |
| 3 | The paint target is ambiguous. With an op selected, the status bar reads "Sculpting on Detail". Add a breadcrumb in the Properties header, a HUD chip at the cursor, and a refusal (⊘ plus tooltip) when strokes hit a non-paint op. | 9.4, 9.5 | +0.6 |
| 4 | The add toolbar is icon-only with no ▾ variant menus. There is no visible evidence of kind-specific right-click menus with hotkeys, or of one-step "From bake / From noise" mask variants. | 9.3 | +0.5 |
| 5 | Nothing is local to the viewport. There is no HUD chip, radial, quick-switcher, or cursor-anchored menu, and the Overlay selector is in the top-right toolbar instead of a viewport corner. | 9.5 | +0.45 |
| 6 | The strength column is not uniform. A boxed `85%` sits next to grey `Norm 100` and `Scrn 60`, and Properties shows `1.000`. Use one monospace, right-aligned 0–100 column. Blend labels belong on mask-op rows only. | 9.6, 9.1 | +0.4 |
| 7 | States are missing or unshown. There is no solo pill, no lock column on op or base rows, no visible dimming for hidden rows, no mask-disabled red diagonal, and "Black mask" has no eye. There is no Freeze pseudo-row. | 9.1 | +0.4 |
| 8 | There are no purpose colours, and the overlay is unclear. "Layer mask" overlay mode tints the viewport blue and orange instead of a greyscale mask with a "Viewing mask" banner. Op-row text is small and low contrast. A Brush section clutters the op Properties view. | 9.6, 9.4 | +0.3 |

## Not verifiable from a single screenshot

These were scored on implied affordances only:
- Drag-drop behaviour and its indicators, and Ctrl+drag duplication.
- Context-menu contents and hotkey labels.
- Whether the thumbnails or mask glyph are clickable targets, and what the modifier clicks do.
- Inline rename.
- Viewport hotkeys (M, Alt+M, S, O-drag).
- Thumbnail live-update latency.
- Multi-select display in Properties.
- Contrast ratios (estimated by eye, not measured).
- Whether the Order ↑/↓ buttons and the op eyes actually work.

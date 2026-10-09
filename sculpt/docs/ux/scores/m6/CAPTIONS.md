# What each screenshot shows

Headless captures of the real app at 1600x1000 (software renderer). Menus and drags are frozen mid-action
by a test hook, because a still image cannot hold an open menu. Hotkeys, inline typing and live clicks cannot
be seen in stills. Panel crops show the right dock (layers over Properties).

- `view_mask_banner.png`: the viewport while a mask is viewed: the surface shows the mask as greyscale (white applies, dark hidden) and a banner names the layer and says how to exit.
- `more_states.png`: `nested` (a folder inside a folder), `drag_copy` (Ctrl+drag a row onto a folder: the ghost reads "(copy)"), `op_drag` (a mask op dragged to the top: purple insertion line), `folder_props` (Properties for a selected folder), `mask_base` (Properties for a mask's base value), and the earlier multi-select states view.
- `radial.png`: the radial menu at the cursor (Q): eight layer actions around the pointer.
- `full_default.png`: whole app. Layers top right, Properties below, paint target in the status bar.
- `rows_a.png`: default; `states` (Pores hidden, Wrinkles locked, both selected, Skin tone at 40%); `multi` (two rows selected, Properties shows shared fields); `compact` (dense rows); `blend` (layers set to Min, Max, Normal, Subtract); `solo` (Skin tone soloed, others dimmed).
- `rows_b.png`: `collapsed` folder; `rename` (inline edit of Wrinkles); `mask_off` (Detail's mask disabled); `preset` (a Curvature mask made in one step on Skin tone, selected); the add-mask menu on a layer without a mask, then on a layer with one.
- `paint_target.png`: a hand-painted mask was just added to Skin tone in one step, its Paint op is selected and Mask Paint is the tool. Properties header and thumbnail frame show the target.
- `paint_target_chip.png`: the brush chip in that same state.
- `menus.png`: right-click menus: on a layer, on a folder (note Flatten folder, Ctrl+M), on a mask op row, and on empty space below the rows.
- `drags.png`: dragging a row into a folder, between rows, and a refused drop of a folder into its own child.
- `viewport_chips.png`: the chip beside the brush for a layer, for a procedural mask op (refused) and for a locked layer (refused).
- `viewport_menu.png`: the right-click menu opened in the viewport (same items as the panel menu).
- `switcher.png`: the Tab layer switcher, a copy of the layer list at the cursor.
- `themes.png`: the `states` view in four themes.

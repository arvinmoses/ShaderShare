# Prompt for a new session: Substance-Painter-style layer UX for the sculpt app

Paste everything below the line into a new cloud session whose environment has
**Custom network access** with these Allowed domains (keep the default package-manager list):

```
helpx.adobe.com
experienceleague.adobe.com
substance3d.adobe.com
www.adobe.com
community.adobe.com
www.youtube.com
i.ytimg.com
www.artstation.com
80.lv
polycount.com
```

Repository: `arvinmoses/ShaderShare` (the app lives in `sculpt/`).
Branch: `ccr-32ffc09b-o5mvh1` (clone or check out this branch; it holds all current work).

---

## ROLE AND GOAL

You are continuing a UI pass on a Rust sculpting app (`sculpt/`, crates `sculpt-core`,
`sculpt-cli`, `sculpt-app`; egui 0.36 + wgpu 30). The user wants the **layer-management
experience to be very similar to Adobe Substance 3D Painter, but for sculpting**, and it must
reach **at least 9/10 similarity on a scoring rubric**. Judgement on the implications is
yours. You may change features if it makes the experience better.

Do NOT open a pull request unless the user asks. Develop on branch `ccr-32ffc09b-o5mvh1`,
commit with clear messages, push with `git push -u origin ccr-32ffc09b-o5mvh1`
(retry up to 4 times with 2s/4s/8s/16s backoff on network failure only).
End every commit message with:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

(Use the attribution lines your session's system reminder specifies, if it gives any.)
Never put a model identifier in commit messages, code comments or pushed files.

## USER'S PRIORITIES (non-negotiable)

1. **Legibility and intuitive workflow** come first.
2. **Avoid cross-screen interaction.** Moving the pointer from one side of the screen to the
   other to activate a command is disliked. Menus open at the cursor, hotkeys work while
   hovering the viewport, Properties is docked directly under the layer list, add controls sit
   next to the list.
3. **Sculpting latency must not regress.** The app's first priority is that UI never slows
   sculpting: same-frame input, 12 ms per-dab budget, dirty-leaf GPU upload. Panels cost about
   0.5 ms per frame today. Any new per-frame UI work (thumbnails, drag-drop, menus) must be
   cached/lazy and must not touch the sculpt path.
4. **Code quality:** elegant, modular, proper OOP principles. Prefer small types with one
   responsibility, traits where behaviour varies, no god functions. `panels.rs` is already
   about 1200 lines: split the layer UI into its own module tree (for example
   `layer_panel/{mod,row,tree,menus,dragdrop,thumbs,hud}.rs`) rather than growing it.
   All UI actions should go through a command layer (`Command`-style) so the UI does not
   mutate the document ad hoc and undo stays consistent.
5. **Validate at every milestone** (see below). The user wants something to evaluate at each
   milestone: screenshots plus a scored report. If a milestone would produce nothing
   evaluable, merge it into the next one and continue.
6. The user is open to a multi-layer plan. Discuss trade-offs with the user when a real
   decision is theirs, but proceed on sensible defaults otherwise.

## DECISIONS ALREADY MADE BY THE USER

- **Engine scope: extend the engine** (`sculpt-core`) for layer folders/groups, nesting, solo,
  duplicate, merge down, reorder, per-layer settings. The project file format
  (`project.json` + `blobs/*.bin`) must stay **backward compatible**: old projects load; add
  fields with serde defaults and a version bump.
- **Locality: cursor-local menus plus hotkeys.** Right-click context menus at the cursor, an
  in-viewport layer HUD, viewport hotkeys, add-flows next to the layer.
- **Review: screenshots plus a score report at each milestone.** The user is often away from
  their machine, so everything must be verifiable headless.
- **Models:** Opus 5.5 for UX research and blind scoring; Sonnet 5.5 for implementation and
  tests; Fable only for a final holistic review. Choose reasoning effort per task. You may
  define your own specialist agents (for example: UX researcher, egui implementer, engine
  implementer, blind reviewer, test writer).

## STEP 0: UNBLOCK THE RESEARCH (do this first)

The previous session's network policy blocked Adobe's docs, so the spec at
`sculpt/docs/ux/substance_painter_layer_ux.md` rests on search excerpts and tutorial
knowledge (claims are tagged **[doc]**, **[obs]**, **[spec]**).

1. Verify access: `curl -sS -o /dev/null -w "%{http_code}\n" https://helpx.adobe.com/substance-3d-painter/using/layer-stack.html`
   (expect 200). If you get 403 / "CONNECT tunnel failed", tell the user the exact host that is
   denied and that they must add it under Network access ▸ Custom ▸ Allowed domains, then
   work on whatever does not depend on it. Do not try to bypass the proxy.
2. With access, use WebFetch to read the official pages (layer stack, masking and effects,
   creating layers, managing layers, blending modes, effects, shortcuts, interface overview,
   layer instancing, release notes for 2021.1, 8.2, 12.0). Search for tutorials and
   screenshots too. If you can download images with curl into the scratchpad directory,
   open them with the Read tool to *see* them (WebFetch returns text only).
3. Update the spec: upgrade **[obs]** and **[spec]** claims to **[doc]** where verified,
   correct anything that was wrong (especially menu item order, toolbar order, row anatomy,
   drop behaviours, hotkeys), and add a "Verified against" section. Commit this before coding.
4. If the user attaches screenshots in chat, treat them as the highest authority.

## CURRENT STATE OF THE CODE

- `sculpt-core::layers::SculptLayer { id, name, opacity, visible, locked, mask: Option<MaskStack>, ... }`
  is flat: no folders, no ordering API, no duplicate/merge/solo. Composite is
  `P = base + Σ visible·opacity·mask·delta`. Layers survive Catmull-Clark subdivision.
  Document API: `add_layer, remove_layer, rename_layer, set_layer_opacity, set_layer_visible,
  set_layer_locked, set_layer_mask, flatten_layer`.
- `MaskStack` / `MaskLayer` (serde JSON): sources fill, channel, noise, mesh bake, direction,
  gradient; blend modes, levels, blur, nested masks. Bakes: curvature, cavity, AO, thickness.
- App UI (`crates/sculpt-app/src/`): `panels.rs` has `layers_panel`, `layer_row`, `mask_rows`,
  `base_row`, `row_frame`, `properties_panel`, `effect_props`, `effect_menu`, `add_mask`,
  `delete_selection`, `sync_mask_edit`. `app.rs` holds `Selection {Layer, Mask, Effect(usize)}`,
  `expanded`, `renaming`, `mask_edit`, `mask_dirty`. `icons.rs` has vector `Icon` and
  `icon_button`. `theme.rs` has JSON themes (hot reload) with `UiColors` (`header`,
  `row_selected` optional). Tools/tray in `tools.rs`; keymap in `keymap.json` + `keymap.rs`.
  Right panel = LAYERS over PROPERTIES.
- Baseline blind score: **5.0 / 10** (`docs/ux/scores/m0_baseline.md`, screenshot
  `docs/ux/scores/m0_baseline.png`). Top gaps: mask is a tiny icon not a second clickable
  thumbnail; no folders; paint target ambiguous (a noise op can be selected while the status bar
  says "Sculpting on Detail"); icon-only toolbar with no mask-type menus; nothing at the
  cursor in the viewport.

## DESIGN DECISIONS (from the spec; revisit after Step 0)

- Row: eye · lock/solo · indent+disclosure · content thumbnail · mask thumbnail (or "+" slot on
  hover) · name (middle-elided, double-click or F2 to rename) · right-aligned strength column
  (drag to scrub, click to type, range −100% to 200%). 36 px rows, 24 px op rows.
- **No per-layer blend modes** for sculpt layers (deltas are additive; no geometric meaning).
  Strength takes that column. Blend modes remain *inside mask stacks*.
- Solo and Lock are additions (not native in SP). Solo is non-destructive.
- Mask thumbnail clicks: plain = target the mask; Alt = view mask in viewport; Shift =
  disable mask (red diagonal). Mask is a stack of ops shown as children (eye, type icon,
  name, strength/blend), reorderable and toggleable.
- Add bar sits **directly under the list**; every add inserts **above the selection**, selects
  and expands the new item. Kind-specific context menus (layer, mask, op, folder, empty area)
  listing hotkeys. "Add mask ▸ From bake / From noise / From freeze" in one step.
- Folders nest; folder strength multiplies children; folder may have its own mask.
- Drag-drop: top/bottom quarter = above/below, middle half of a folder = into, invalid =
  red bar and refused, Ctrl+drag duplicates, spring-open collapsed folders after 600 ms,
  auto-scroll near edges, Esc cancels.
- Paint target shown three ways: thumbnail frame, Properties breadcrumb
  (`Wrinkles › Mask › Paint`), viewport HUD chip at the brush cursor (orange = delta,
  purple = mask, blue = freeze). Strokes on a non-paintable op are refused with feedback.
- Cursor-local: right-click in viewport opens the same menus at the cursor; hotkeys act on the
  active item from the viewport (M toggle layer/mask target, Alt+M view mask, S solo, H hide,
  L lock, Ctrl+D duplicate, Ctrl+G group, Ctrl+E merge down, Del delete, O-drag strength
  scrub); Tab opens a layer quick-switcher at the cursor; optional radial menu.
- Viewport overlay dropdown replaces SP's channel dropdown (Shaded · Mask · Freeze · Delta
  heat · Bakes).

## MILESTONES (each ends with a validated, scored, user-evaluable deliverable)

| # | Milestone | Target score |
|---|---|---|
| M1 | Row anatomy: two thumbnails, eye/lock/solo, strength column, inline rename, kind-specific context menus, add bar under the list inserting above selection, Properties breadcrumb, paint-target frame. Engine: solo, duplicate, merge down, reorder. | about 7 |
| M2 | Hierarchy: folders in engine (backward-compatible format), drag-drop with indicators, Ctrl+drag duplicate, spring-open, auto-scroll. | about 8 |
| M3 | Mask stack as children: reorder/toggle ops, one-step From bake / From noise masks, copy/paste mask, Alt-click view, Shift-click disable. | about 8.5 |
| M4 | Cursor-local: viewport context menus, paint-target HUD chip, viewport hotkeys, Tab switcher. | 9+ |
| M5 | Polish: live lazy thumbnails, density toggle, legibility pass, final blind score on the 10-task checklist. | 9+ |

## VALIDATION REQUIRED AT EVERY MILESTONE

1. `cd sculpt && cargo build --release -p sculpt-app` clean; `cargo clippy --release --workspace`
   0 warnings; `cargo fmt --check` if the repo uses it; `cargo test --release` all passing
   (currently 14 engine tests in `crates/sculpt-core/tests/engine.rs`). Add tests for every new
   engine feature: folder tree ops, reorder, duplicate, merge down, solo, undo/redo of each,
   save/load round trip, and loading an **old-format project** (keep a fixture).
2. **Latency guard:** run
   `xvfb-run -a ./target/release/sculpt-app --level 9 --test-strokes 180 --screenshot out.png`
   and compare "CPU: input + dabs" and "whole frame incl. UI" to the previous milestone. Earlier
   figures (software Vulkan, 4 cores): input+dabs median 2.1 ms at 98k faces, 2.8 ms at 393k,
   12.4 ms at 6.3M; panels about 0.5 ms/frame. Report any regression over about 10 percent
   and fix it before moving on. If the container lacks `libxkbcommon-x11`, install it with apt.
3. **Screenshots:** capture headless with `--demo-layers --size 1600x1000` (and the themes
   Mudbox Dark, Painter Dark, Studio Light, High Contrast). Add a screenshot hook or scripted
   scenarios to show menus open, drag in progress, expanded mask stacks, and the target HUD
   (use a temporary env-var hook if needed, and remove it afterwards; verify with grep).
   Save to `sculpt/docs/ux/scores/mN_*.png`.
4. **Blind scoring:** spawn an Opus reviewer that reads ONLY section 9 of the spec (rubric and
   10-task checklist) and the screenshots, never the code or your reasoning. It writes
   `docs/ux/scores/mN.md` with six criterion scores (row anatomy 20, hierarchy 20,
   menus/add-flows 15, selection/properties 15, locality 15, legibility 15), the weighted
   total, and ranked gaps. It must say what it could not verify from screenshots. Do not
   inflate: if a milestone scores below its target, fix the top gaps and re-score before
   proceeding. The end goal is **>= 9.0**.
5. Add the interaction checks screenshots cannot show as headless tests or scripted
   scenarios where possible (drop-zone math, selection model, command emit, hotkey map).
6. Update `sculpt/README.md` and `sculpt/docs/ARCHITECTURE.md` for new concepts and hotkeys,
   update `keymap.json` defaults, keep all four themes working (new colours go through
   `UiColors`/theme JSON, never hard-coded).
7. Commit and push at each milestone, then send the user the screenshots, the scored report
   and a short list of what to try, in plain language with no jargon.

## THINGS TO REMEMBER FROM EARLIER WORK

- egui 0.36 / wgpu 30 API differences: `Panel::top/show`, `MenuBar::new().ui`,
  `CentralPanel::no_frame().show`. Avoid unsafe pointer aliasing in panel code (pass sizes
  by value). Slider width must be fixed (150 px), not derived from available width, or the
  right panel grows every frame.
- Use vector icons from `icons.rs` (no font glyph dependence); add new `Icon` variants
  (Folder, Solo, Duplicate, Merge, Grip, etc.). Inter font is bundled.
- Thumbnails must be lazy and cached by `(node, revision)`; never block a frame.
- The user is often away from the machine and uses mobile; keep messages short, give
  screenshots, avoid asking for local testing unless necessary. If they ask how to test
  locally: install Rust via rustup, `cd sculpt`, `cargo run --release -p sculpt-app -- --demo-layers`
  (`--level 9` is 1.6M faces, `--level 10` is 6.3M, `--features tablet` for pen pressure on
  Windows/Wayland).

## FIRST ACTIONS

1. Do Step 0 (verify network, research, update the spec, commit).
2. Re-read `docs/ux/substance_painter_layer_ux.md` and `docs/ux/scores/m0_baseline.md`.
3. Post a short plan to the user (milestones above, adjusted by what the research found),
   then begin M1 without waiting unless a real trade-off needs their decision.

# Spike 2 code (render-time CSG preview), parked

This was built on the pre-UI, pre-LOD base (`c378864`), so it could not be merged
directly. To port it:

1. Copy `mod.rs` and `csg.wgsl` to `crates/sculpt-app/src/viewport_csg/` and add `mod viewport_csg;`.
2. Re-apply the four small hooks in `hooks.patch` (`viewport.rs`, `app.rs`, `main.rs`)
   by hand against the current LOD-based viewport.
3. Replace the stand-in instance generator (jittered Fibonacci lattice, 48-byte instance record)
   with the real generator output.

See `../spike2_render_csg.md` for results and the benchmark (`sculpt/scripts/csg_bench.sh`).

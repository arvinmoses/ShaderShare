# Local benchmark results (real GPU)

Machine: Windows 11 Home (build 26200), 24 CPU threads, rustc 1.97.0.
GPU: AMD Radeon AI PRO R9700 (discrete, Vulkan, AMD proprietary driver 25.20.42.14 / LLPC; a second R9700 and an
integrated Radeon are present, wgpu picked the first discrete one). Timestamp queries available.
Branch `ccr-32ffc09b-o5mvh1` at `d45439e`.

## Build and tests

- `cargo build --release --workspace`: clean, 1m33s cold.
- `cargo test --release --workspace --no-fail-fast`: all green after one test fix. Two timing tests in
  `sculpt-core/tests/schedule.rs` failed at first. Cause: with no cost estimate yet, the scheduler's first batch packs
  one slice per rayon thread before it checks the deadline, so with 24 threads (the cloud box had 4) one batch covered
  every visible item and `deadline_hit` was false. A test-environment assumption, not a scheduler bug. The two tests
  now run on a 2-thread pool.

## Spike 1: GPU displacement boolean vs CPU (`--gpu-bench 7,8,9,10 1000,10000,100000 --cpu`, Subtract)

"slider total" = bake + upload + GPU time + readback path; "wall" = submit to fence.

| faces | instances | GPU wall ms | slider total ms | CPU ms | CPU / GPU |
|---|---|---|---|---|---|
| 98k | 1k / 10k / 100k | 0.18 / 0.29 / 1.78 | 0.50 / 0.96 / 4.49 | 1.4 / 4.0 / 14.8 | 2.8x / 4.2x / 3.3x |
| 393k | 1k / 10k / 100k | 0.26 / 0.30 / 1.47 | 0.64 / 1.02 / 3.71 | 2.0 / 5.5 / 18.6 | 3.1x / 5.4x / 5.0x |
| 1.57M | 1k / 10k / 100k | 0.26 / 0.33 / 1.08 | 0.65 / 0.90 / 3.49 | 2.6 / 7.5 / 31.2 | 4.0x / 8.4x / 9.0x |
| 6.29M | 1k / 10k / 100k | 0.57 / 0.90 / 1.67 | 0.88 / 1.58 / 4.19 | 4.3 / 12.4 / 52.0 | 4.9x / 7.9x / 12.4x |

GPU buffers: 28 MB (98k) to 244 MB (6.3M). One-time mesh upload 103 ms at 6.3M. This is far better than the
3-35 ms the cloud estimate predicted at 6.3M. The CPU path is also fast on 24 threads (52 ms at 6.3M / 100k).

## CPU generator-bench (spike A/B, level 10, 6.3M faces)

Slider total (update + hash + boolean): 4.2 ms (1k), 12.6 ms (10k), 52.1 ms (100k). 1% edit: 0.95 / 3.4 / 30 ms.
Placement cold 11-17 ms, warm 0.2-6 ms. Edge detect + chains 26 ms cold; 100k edge points 1.3 ms.
Voxel (spike B) cold: 0.6-8.6 s (level 10 table, 1k-100k); not interactive, as before.

## Spike 4: progressive scheduler (`generator-drag-sim`, 6.3M faces, CPU)

| N | view | frame med / p95 / max ms | full recompute baseline med ms |
|---|---|---|---|
| 10k | zoom | 2.5-4.9 / 3.0-5.7 / <= 5.8 | 14-15 |
| 100k | overview | 5.5-6.6 (+seed 0.0) / 6.6-9.0 / <= 10.1 | 54-66 |
| 100k | zoom | 6.1-6.5 / 7.2-9.3 / <= 10.2 | 51-66 |

All 120 drag frames: 100k overview median 5.9 / p95 7.9 / max 10.1 ms. No frames over budget. Converges exactly.

## Spike 5: GPU sparse SDF (`--gpu-sdf-bench`, `--gpu-sdf-check`)

Correctness: all six checks PASS (subtract, union, intersect, smooth variants) vs CPU, 0 sign mismatches.
Level 10 warm frame eval: 1.8 ms (1k, 1e), 4.2 ms (1k, 0.5e), 12 ms (10k, 1e), 27 ms (10k, 0.5e), 23 ms (100k, 1e),
128 ms (100k, 0.5e). Cold: 136-165 ms (1k), 0.9-1.3 s (10k), 1.9-3.5 s (100k). Memory: GPU buffers up to ~2 GB,
cache up to 6.4 GB f32 (3.2 GB f16). Live-capable up to about 10k instances; heavy beyond.

## Spike 3: adaptive refinement (`refine-bench time`)

Full refine, 6.3M faces: 6 ms (1k), 78 ms (10k). 1% edit: 0.9 ms (1k), 7 ms (10k). Cheaper than the cloud numbers
(110-195 ms / 0.8-1.8 s), on 24 threads.

## Spike 6: ADF (`adf-bench perf`, level 10, 0.5 voxel)

1k: ~21 ms total. 10k: ~160 ms. 100k: ~580 ms (about 25-45x faster than the cloud's 14.7-26 s). About 1% of
quads have vertexless cells, as before.

## Not run

- `csg_bench.sh` (spike 2): the app has no `--csg-preview` flag, since the spike is parked and not wired into the
  current LOD viewport. Needs the port first.
- `adf-bench quality` / `refine-bench quality|cracks` (quality, not speed).

## Sculpt latency guard (after milestone 1)

`sculpt-app --level N --test-strokes 180` on this machine, discrete R9700, after adding the generator layer hook to
`Document::recomposite` (no baseline run on this machine before the change, so compare with the cloud figures only):

| faces | input + dabs median / p95 / max ms | whole frame incl. UI median ms |
|---|---|---|
| 98k | 1.18 / 1.99 / 4.97 | 1.63 |
| 1.57M | 2.18 / 3.51 / 10.33 | 2.69 |

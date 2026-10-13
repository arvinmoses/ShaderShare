#!/usr/bin/env bash
# One-command benchmark for the render-time CSG preview (SPIKE 2).
#   scripts/csg_bench.sh            # on a machine with a display / real GPU
#   scripts/csg_bench.sh --xvfb     # headless (software adapter; NOT representative)
# Prints one CSG_BENCH line per configuration: GPU-synchronised wall ms per frame.
# Env: SIZE=1280x800  LEVELS="7 10"  COUNTS="100 10000 100000"
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${SIZE:-1280x800}"
LEVELS="${LEVELS:-7 10}"          # 98k and 6.3M faces
COUNTS="${COUNTS:-100 10000 100000}"
cargo build --release -p sculpt-app >/dev/null
BIN=target/release/sculpt-app
XVFB="${1:-}"
wrap() {
  if [[ "$XVFB" == "--xvfb" ]]; then
    xvfb-run -a -s "-screen 0 1920x1200x24" "$@"
  else
    "$@"
  fi
}
for level in $LEVELS; do
  for mode in subtract add intersect; do
    for n in $COUNTS; do
      echo "== level $level ($((6 * 4 ** level)) faces) $mode x$n"
      # --test-strokes 0 = no sculpting, just render ~25 frames and exit.
      wrap "$BIN" --level "$level" --size "$SIZE" --csg-preview "$mode" --csg-count "$n" --csg-bench --test-strokes 0 2>&1 | grep -E "CSG_BENCH" || true
    done
  done
done

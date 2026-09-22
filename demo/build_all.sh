#!/bin/sh
# Rebuild the demo from every recorded session in the repo root.
#
# The build_demo.py CLI takes sessions explicitly rather than discovering them,
# so that publishing a review is always a deliberate act. This script is the
# current list; add a directory here to put it in the paper selector.
#
#   ./demo/build_all.sh
#   ./demo/build_all.sh --on-mismatch skip_trajectory
#
# Run from anywhere; paths resolve against the repo root.
set -e
cd "$(dirname "$0")/.."
exec python3 build_demo.py \
  review_0A4Uf88pog \
  review_0ACUx9pMWJ \
  review_0Af7UiJISU \
  review_0aj9su59IG \
  review_0aNfWttgHd \
  review_0JWhSwwXak_v3 \
  --out docs/index.html "$@"

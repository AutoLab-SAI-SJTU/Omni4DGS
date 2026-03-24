#!/usr/bin/env bash
set -euo pipefail

# Usage: bash dataset/generate_split.sh [image_dir] [output_dir]
# Default input: dataset/scene4/images
# Default output: alongside the image directory (train.txt / test.txt)
# TEST_MOD env var controls which frames go to test (default: 10).

IMAGE_DIR=${1:-dataset/scene4/images}
OUTPUT_DIR=${2:-$(dirname "$IMAGE_DIR")}
TEST_MOD=${TEST_MOD:-10}

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t frame_ids < <(
  find "$IMAGE_DIR" -maxdepth 1 -type f -name '*.jpg' -printf '%f\n' 2>/dev/null |
    sed -E 's/\.jpg$//' |
    grep -E '^[0-9]+$' |
    LC_ALL=C sort -n
)

if [[ ${#frame_ids[@]} -eq 0 ]]; then
  echo "No numeric jpg files found in: $IMAGE_DIR" >&2
  exit 1
fi

: >"$OUTPUT_DIR/train.txt"
: >"$OUTPUT_DIR/test.txt"

for id in "${frame_ids[@]}"; do
  if (( id % TEST_MOD == 0 )); then
    echo "$id" >>"$OUTPUT_DIR/test.txt"
  else
    echo "$id" >>"$OUTPUT_DIR/train.txt"
  fi
done

echo "Done. train/test written to: $OUTPUT_DIR (test every $TEST_MOD-th frame)"

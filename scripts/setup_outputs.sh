#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_PATH="$ROOT_DIR/outputs"

if [ -L "$OUTPUT_PATH" ]; then
  echo "outputs symlink present: $OUTPUT_PATH -> $(readlink "$OUTPUT_PATH")"
else
  mkdir -p "$OUTPUT_PATH"
  echo "outputs directory ready: $OUTPUT_PATH"
fi

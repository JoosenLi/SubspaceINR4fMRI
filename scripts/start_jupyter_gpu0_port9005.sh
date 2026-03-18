#!/usr/bin/env bash
set -euo pipefail

# HashINR notebook launcher:
# - always use physical GPU0
# - always use port 9005
# - never fall back to port 9002 via retries

export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hashinr_mplconfig}"

exec jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port=9005 \
  --ServerApp.port_retries=0 \
  "$@"

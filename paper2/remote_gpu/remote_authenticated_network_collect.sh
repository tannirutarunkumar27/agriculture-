#!/usr/bin/env bash
# Collect an already-completed authenticated multi-process deployment's receipts.
# No credentials or private keys are accepted by this wrapper.
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 PLAN_JSON RECEIPTS_DIRECTORY OUTPUT_DIRECTORY [INDEPENDENT_REVIEW_JSON]" >&2
  exit 64
fi

args=(--plan "$1" --receipts-root "$2" --output "$3")
if [ "$#" -eq 4 ]; then
  args+=(--review "$4")
fi
python3 -B -m paper2.restart.collect_authenticated_network_evidence "${args[@]}"

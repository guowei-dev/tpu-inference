#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# .buildkite/scripts/run_xla_autotune_shard.sh
# --------------------------------------------
# Host-side driver for one shard of the XLA OFAT autotune sweep.
#
# Args (positional):
#   $1  SLICE_INDEX  (1-based, matches {{matrix}})
#   $2  SLICE_COUNT  (total shard count)
#
# Optional env (set in the calling step's `env:`):
#   AUTOTUNE_MODEL          Default Qwen/Qwen3.5-397B-A17B-FP8
#   AUTOTUNE_TARGET_METRIC  Default total_token_throughput
#   AUTOTUNE_BASELINE_RUNS  Default 3
#
# Behaviour:
#   1.  Spawns a background watcher loop on the host that uploads every
#       new JSON artifact in $ARTIFACT_DIR via `buildkite-agent artifact
#       upload`, so per-trial results land in Buildkite *as soon as* the
#       trial finishes — no waiting for the whole shard.
#   2.  Calls run_in_docker.sh once for the whole shard.  The kernel-tuning
#       /tmp mount (already wired up in run_in_docker.sh) is reused as a
#       shared host↔container artifact directory — no run_in_docker.sh
#       changes required.
#   3.  Final cleanup pass uploads anything the watcher missed.

set -euo pipefail

SLICE_INDEX="${1:?usage: $0 <slice_index> <slice_count>}"
SLICE_COUNT="${2:?usage: $0 <slice_index> <slice_count>}"

AUTOTUNE_MODEL="${AUTOTUNE_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
AUTOTUNE_TARGET_METRIC="${AUTOTUNE_TARGET_METRIC:-total_token_throughput}"
AUTOTUNE_BASELINE_RUNS="${AUTOTUNE_BASELINE_RUNS:-3}"

# Reuse the /tmp/kernel_tuning mount that run_in_docker.sh already binds
# both inside and outside docker — gives us a shared artifact directory
# without touching run_in_docker.sh.
SHARED_ROOT="/tmp/kernel_tuning/xla_autotune"
ARTIFACT_DIR_HOST="${SHARED_ROOT}/shard_${SLICE_INDEX}_of_${SLICE_COUNT}"
ARTIFACT_DIR_CONTAINER="${ARTIFACT_DIR_HOST}"
mkdir -p "${ARTIFACT_DIR_HOST}"

echo "[xla-autotune] shard ${SLICE_INDEX}/${SLICE_COUNT} → ${ARTIFACT_DIR_HOST}"

# --------------------------------------------------------------------------
# Host-side watcher: uploads each *.json + summary.jsonl exactly once as
# soon as it appears.  Polls every 20 s.
# --------------------------------------------------------------------------
(
  cd "${ARTIFACT_DIR_HOST}"
  declare -A UPLOADED=()
  while true; do
    shopt -s nullglob
    for f in *.json summary.jsonl; do
      if [[ -f "$f" && -z "${UPLOADED[$f]:-}" ]]; then
        if buildkite-agent artifact upload "$f"; then
          UPLOADED["$f"]=1
        fi
      fi
    done
    sleep 20
  done
) &
WATCH_PID=$!
trap 'kill ${WATCH_PID} 2>/dev/null || true' EXIT INT TERM

set +e
.buildkite/scripts/run_in_docker.sh bash -c "
  set -euo pipefail
  cd /workspace/tpu_inference
  python3 .buildkite/xla_autotune/autotuner.py \
    --flag-list-file .buildkite/xla_autotune/flags.txt \
    --benchmark-args-json .buildkite/xla_autotune/config.json \
    --model '${AUTOTUNE_MODEL}' \
    --target-metric '${AUTOTUNE_TARGET_METRIC}' \
    --scheduler ofat \
    --slice-index ${SLICE_INDEX} --slice-count ${SLICE_COUNT} \
    --baseline-runs ${AUTOTUNE_BASELINE_RUNS} \
    --artifact-dir '${ARTIFACT_DIR_CONTAINER}'
"
RC=$?
set -e

# Stop the watcher, then do a final upload pass for anything that landed
# between its last tick and the docker exit.
kill "${WATCH_PID}" 2>/dev/null || true
wait "${WATCH_PID}" 2>/dev/null || true

(
  cd "${ARTIFACT_DIR_HOST}"
  shopt -s nullglob
  for f in *.json summary.jsonl; do
    [[ -f "$f" ]] && buildkite-agent artifact upload "$f" || true
  done
)

exit "${RC}"

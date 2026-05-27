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
#
# Host-side driver for one shard of the XLA autotune sweep.
#
# Usage: run_xla_autotune_shard.sh <slice_index> <slice_count>
#
# Env (optional, set in the pipeline step):
#   AUTOTUNE_MODEL          default: Qwen/Qwen3.5-397B-A17B-FP8
#   AUTOTUNE_TARGET_METRIC  default: total_token_throughput
#   AUTOTUNE_BASELINE_RUNS  default: 2
#   AUTOTUNE_FLAGS          default: .buildkite/xla_autotune/flags.txt
#   AUTOTUNE_CONFIG         optional JSON of VLLMTestParam overrides
#   AUTOTUNE_DRY_RUN        if non-empty, pass --dry-run to the autotuner
#                           (skips vllm serve + benchmark; useful for
#                           validating pipeline plumbing without TPU cost)
#
# A background watcher tails the artifact dir and uploads per-trial JSONs and
# completed log bundles to Buildkite as soon as they land — progress is
# visible without waiting for the whole shard to finish.

set -euo pipefail

SLICE_INDEX="${1:?usage: $0 <slice_index> <slice_count>}"
SLICE_COUNT="${2:?usage: $0 <slice_index> <slice_count>}"

AUTOTUNE_MODEL="${AUTOTUNE_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
AUTOTUNE_TARGET_METRIC="${AUTOTUNE_TARGET_METRIC:-total_token_throughput}"
AUTOTUNE_BASELINE_RUNS="${AUTOTUNE_BASELINE_RUNS:-2}"
AUTOTUNE_FLAGS="${AUTOTUNE_FLAGS:-.buildkite/xla_autotune/flags.txt}"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-}"

# /tmp/kernel_tuning is already bind-mounted host↔container by run_in_docker.sh,
# so we reuse it as a shared artifact directory.
SHARED_ROOT="/tmp/kernel_tuning/xla_autotune"
SHARD_DIRNAME="shard_${SLICE_INDEX}_of_${SLICE_COUNT}"
ARTIFACT_DIR="${SHARED_ROOT}/${SHARD_DIRNAME}"

# Wipe leftovers from any prior build on this agent VM, otherwise the watcher
# would re-upload stale per-trial JSONs before the new run overwrites them.
rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}"

echo "[xla-autotune] shard ${SLICE_INDEX}/${SLICE_COUNT} → ${ARTIFACT_DIR}"

# Watcher: ship per-trial JSONs as their mtime advances, and ship each log
# bundle exactly once when its sibling `<bundle>.done` marker appears.
(
  cd "${SHARED_ROOT}"
  declare -A LAST_MTIME=()
  declare -A UPLOADED_BUNDLE=()
  while true; do
    shopt -s nullglob
    for f in "${SHARD_DIRNAME}"/*.json "${SHARD_DIRNAME}/summary.jsonl"; do
      [[ -f "$f" ]] || continue
      cur_mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      if [[ "${LAST_MTIME[$f]:-0}" != "$cur_mtime" ]]; then
        if buildkite-agent artifact upload "$f"; then
          LAST_MTIME["$f"]=$cur_mtime
        fi
      fi
    done
    for marker in "${SHARD_DIRNAME}"/logs/*.done; do
      [[ -f "$marker" ]] || continue
      bundle_dir="${marker%.done}"
      [[ -d "$bundle_dir" ]] || continue
      if [[ -z "${UPLOADED_BUNDLE[$bundle_dir]:-}" ]]; then
        if buildkite-agent artifact upload "${bundle_dir}/**/*"; then
          UPLOADED_BUNDLE["$bundle_dir"]=1
        fi
      fi
    done
    sleep 20
  done
) &
WATCH_PID=$!
trap 'kill ${WATCH_PID} 2>/dev/null || true' EXIT INT TERM

EXTRA_ARGS=""
if [[ -n "${AUTOTUNE_CONFIG}" ]]; then
  EXTRA_ARGS+=" --benchmark-args-json '${AUTOTUNE_CONFIG}'"
fi
if [[ -n "${AUTOTUNE_DRY_RUN:-}" ]]; then
  EXTRA_ARGS+=" --dry-run"
fi

set +e
.buildkite/scripts/run_in_docker.sh bash -c "
  set -euo pipefail
  cd /workspace/tpu_inference
  python3 .buildkite/xla_autotune/autotuner.py \
    --flag-list-file '${AUTOTUNE_FLAGS}' \
    --model '${AUTOTUNE_MODEL}' \
    --target-metric '${AUTOTUNE_TARGET_METRIC}' \
    --slice-index ${SLICE_INDEX} --slice-count ${SLICE_COUNT} \
    --baseline-runs ${AUTOTUNE_BASELINE_RUNS} \
    --artifact-dir '${ARTIFACT_DIR}' \
    ${EXTRA_ARGS}
"
RC=$?
set -e

kill "${WATCH_PID}" 2>/dev/null || true
wait "${WATCH_PID}" 2>/dev/null || true

# Final sweep: pick up anything written between the watcher's last tick and
# the docker exit.
( cd "${SHARED_ROOT}" && buildkite-agent artifact upload "${SHARD_DIRNAME}/**/*" || true )

exit "${RC}"

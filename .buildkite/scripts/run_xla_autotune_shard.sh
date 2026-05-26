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
# Both paths are relative to the tpu_inference repo root (= the docker
# working directory).  Defaults run the full production sweep; override
# in the pipeline step to switch into the smoke-test config.
AUTOTUNE_FLAGS="${AUTOTUNE_FLAGS:-.buildkite/xla_autotune/flags.txt}"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-.buildkite/xla_autotune/config.json}"

# Reuse the /tmp/kernel_tuning mount that run_in_docker.sh already binds
# both inside and outside docker — gives us a shared artifact directory
# without touching run_in_docker.sh.
SHARED_ROOT="/tmp/kernel_tuning/xla_autotune"
ARTIFACT_DIR_HOST="${SHARED_ROOT}/shard_${SLICE_INDEX}_of_${SLICE_COUNT}"
ARTIFACT_DIR_CONTAINER="${ARTIFACT_DIR_HOST}"

# Always wipe any leftover artifact dir from a previous build on this agent
# VM so a re-trigger never picks up stale per-trial JSONs that the watcher
# would publish before the new run overwrites them.
rm -rf "${ARTIFACT_DIR_HOST}"
mkdir -p "${ARTIFACT_DIR_HOST}"

echo "[xla-autotune] shard ${SLICE_INDEX}/${SLICE_COUNT} → ${ARTIFACT_DIR_HOST}"

# --------------------------------------------------------------------------
# Host-side watcher.  Two responsibilities:
#
#   (1) Per-trial JSONs at top level (baseline_NN.json / cand_NNN.json) and
#       summary.jsonl: re-upload whenever mtime advances, so progressive
#       updates surface in BK without waiting for the shard to finish.
#
#   (2) Per-trial log bundles under logs/<tag>_EXP_<ts>/: each bundle is
#       written by vllm_test_framework.py and finalised by the autotuner
#       with a sibling marker file `<tag>_EXP_<ts>.done`.  When the marker
#       appears, every file inside the bundle is uploaded once; this gives
#       BK a complete, consistent log dir per trial — server log, env
#       dump (experiment_info.txt), benchmark stdout, the _tag.txt
#       descriptor, etc.
#
# All uploads run from $SHARED_ROOT so artifact paths in BK include the
# shard prefix (e.g. `shard_2_of_4/logs/autotune_cand_003_EXP_.../...`),
# which doubles as the descriptive tag in the BK UI.
# --------------------------------------------------------------------------
SHARD_DIRNAME="shard_${SLICE_INDEX}_of_${SLICE_COUNT}"
(
  cd "${SHARED_ROOT}"
  declare -A LAST_MTIME=()
  declare -A UPLOADED_BUNDLE=()
  while true; do
    shopt -s nullglob

    # (1) Top-level trial JSONs + summary.
    for f in "${SHARD_DIRNAME}"/*.json "${SHARD_DIRNAME}/summary.jsonl"; do
      [[ -f "$f" ]] || continue
      cur_mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      if [[ "${LAST_MTIME[$f]:-0}" != "$cur_mtime" ]]; then
        if buildkite-agent artifact upload "$f"; then
          LAST_MTIME["$f"]=$cur_mtime
        fi
      fi
    done

    # (2) Completed trial log bundles.
    for marker in "${SHARD_DIRNAME}"/logs/*.done; do
      [[ -f "$marker" ]] || continue
      bundle_dir="${marker%.done}"
      [[ -d "$bundle_dir" ]] || continue
      bundle_key="$bundle_dir"
      if [[ -z "${UPLOADED_BUNDLE[$bundle_key]:-}" ]]; then
        # Upload every file inside the bundle dir, preserving the
        # `shard_K_of_N/logs/<trial_id>_EXP_<ts>/...` path.
        if buildkite-agent artifact upload "${bundle_dir}/**/*"; then
          UPLOADED_BUNDLE["$bundle_key"]=1
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
    --flag-list-file '${AUTOTUNE_FLAGS}' \
    --benchmark-args-json '${AUTOTUNE_CONFIG}' \
    --model '${AUTOTUNE_MODEL}' \
    --target-metric '${AUTOTUNE_TARGET_METRIC}' \
    --scheduler ofat \
    --slice-index ${SLICE_INDEX} --slice-count ${SLICE_COUNT} \
    --baseline-runs ${AUTOTUNE_BASELINE_RUNS} \
    --artifact-dir '${ARTIFACT_DIR_CONTAINER}'
"
RC=$?
set -e

# Stop the watcher, then do a final recursive upload pass for anything
# that landed between its last tick and the docker exit — both top-level
# JSONs and every per-trial log bundle, complete or partial.
kill "${WATCH_PID}" 2>/dev/null || true
wait "${WATCH_PID}" 2>/dev/null || true

(
  cd "${SHARED_ROOT}"
  buildkite-agent artifact upload "${SHARD_DIRNAME}/**/*" || true
)

exit "${RC}"

#!/usr/bin/env python3
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

"""XLA flag auto-tuner.

Drives ``VLLMTestFramework`` to benchmark a list of candidate XLA / libtpu
flags against a baseline using one-factor-at-a-time (OFAT).

The CLI is sharded — each invocation handles ``--slice-index`` of
``--slice-count`` (1-based), so it fans out cleanly across a Buildkite
matrix step.  Every shard also runs ``--baseline-runs`` baseline trials so
the noise floor can be re-estimated per machine.

Each trial's record is written to ``<artifact-dir>/<trial_id>.json`` and
appended to ``summary.jsonl`` as soon as it finishes, so partial progress
survives a crash and the host-side watcher can ship results incrementally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_test_framework import (  # noqa: E402
    VLLMTestFramework,
    VLLMTestParam,
    VLLMTestTask,
)


@dataclass
class Trial:
    trial_id: str
    kind: str            # "baseline" | "candidate"
    flag: Optional[str]  # None for baseline; one extra flag for candidate
    extra_flags: List[str] = field(default_factory=list)


def slice_flags(flags: List[str], slice_index: int, slice_count: int) -> List[str]:
    """Contiguous-chunk shard: shard ``slice_index`` of ``slice_count``."""
    if slice_index < 1 or slice_index > slice_count:
        raise ValueError(f"slice_index {slice_index} out of [1, {slice_count}]")
    chunk = math.ceil(len(flags) / slice_count)
    start = (slice_index - 1) * chunk
    end = start + chunk if slice_index < slice_count else len(flags)
    return flags[start:end]


def plan_ofat(
    sliced_flags: List[str],
    baseline_flags: List[str],
    baseline_runs: int,
) -> List[Trial]:
    """Plan: ``baseline_runs`` baselines, then each candidate flag once."""
    trials: List[Trial] = []
    for i in range(baseline_runs):
        trials.append(Trial(
            trial_id=f"baseline_{i+1:02d}",
            kind="baseline",
            flag=None,
            extra_flags=list(baseline_flags),
        ))
    for idx, flag in enumerate(sliced_flags):
        trials.append(Trial(
            trial_id=f"cand_{idx+1:03d}",
            kind="candidate",
            flag=flag,
            extra_flags=list(baseline_flags) + [flag],
        ))
    return trials


def _load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _build_test_param(
    model: str,
    extra_flags: List[str],
    overrides: Dict[str, Any],
    tag: str,
) -> VLLMTestParam:
    p = VLLMTestParam()
    p.model_name = model
    p.tag = tag
    # Append; do not overwrite.  The framework's default LIBTPU set is what
    # the GCS-backed JAX compile cache is keyed on — replacing it would
    # invalidate every cache entry and force a ~90 min cold compile.
    p.extra_libtpu_init_args = list(p.extra_libtpu_init_args) + list(extra_flags)
    for k, v in overrides.items():
        if k.startswith("_"):  # allow `_comment` style JSON metadata
            continue
        if not hasattr(p, k):
            raise ValueError(f"unknown VLLMTestParam field: {k}")
        setattr(p, k, v)
    return p


def _run_trial(
    trial: Trial,
    model: str,
    overrides: Dict[str, Any],
    artifact_dir: str,
    target_metric: str,
    dry_run: bool,
    summary_fp,
) -> Dict[str, Any]:
    """Run one trial, persist its artifact, never raise."""
    started = datetime.utcnow().isoformat()
    t0 = time.time()
    record: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "kind": trial.kind,
        "flag": trial.flag,
        "extra_flags": trial.extra_flags,
        "model": model,
        "target_metric": target_metric,
        "started_utc": started,
        "success": False,
        "metrics": {},
        "target_value": None,
        "error": "",
    }
    logs_root = os.path.join(artifact_dir, "logs")
    os.makedirs(logs_root, exist_ok=True)
    trial_exp_dir: Optional[str] = None

    try:
        param = _build_test_param(
            model=model,
            extra_flags=trial.extra_flags,
            overrides=overrides,
            tag=f"autotune_{trial.trial_id}",
        )
        param.base_log_dir = logs_root
        fw = VLLMTestFramework(params=param, dry_run=dry_run)
        trial_exp_dir = fw.exp_dir
        results = fw.execute_task([VLLMTestTask.RUN_BENCHMARK_SERVING])
        result = results[0] if results else None
        if result is not None:
            record["success"] = bool(result.success)
            record["metrics"] = dict(result.metrics)
            record["error"] = result.error_message or ""
            for sub_metrics in result.metrics.values():
                if isinstance(sub_metrics, dict) and target_metric in sub_metrics:
                    record["target_value"] = sub_metrics[target_metric]
                    break
    except Exception as e:  # noqa: BLE001
        record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    record["duration_sec"] = round(time.time() - t0, 2)
    record["finished_utc"] = datetime.utcnow().isoformat()

    out_path = os.path.join(artifact_dir, f"{trial.trial_id}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    summary_fp.write(json.dumps(record) + "\n")
    summary_fp.flush()

    # Drop a human-readable _tag.txt + a .done marker the watcher uses to
    # detect that the log bundle is complete and safe to upload.
    if trial_exp_dir and os.path.exists(trial_exp_dir):
        try:
            with open(os.path.join(trial_exp_dir, "_tag.txt"), "w") as f:
                f.write(
                    f"trial_id        : {trial.trial_id}\n"
                    f"kind            : {trial.kind}\n"
                    f"flag            : {trial.flag or '(baseline)'}\n"
                    f"model           : {model}\n"
                    f"target_metric   : {target_metric}\n"
                    f"target_value    : {record['target_value']}\n"
                    f"success         : {record['success']}\n"
                    f"duration_sec    : {record['duration_sec']}\n"
                    f"started_utc     : {record['started_utc']}\n"
                    f"finished_utc    : {record['finished_utc']}\n"
                    f"error           : {record['error'].splitlines()[0] if record['error'] else ''}\n"
                )
            open(trial_exp_dir + ".done", "w").close()
        except Exception:  # noqa: BLE001
            pass

    print(
        f"[autotune] {trial.trial_id} kind={trial.kind} "
        f"success={record['success']} {target_metric}={record['target_value']} "
        f"dur={record['duration_sec']}s -> {out_path}",
        flush=True,
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag-list-file", required=True,
                    help="One XLA flag per line (e.g. --xla_tpu_foo=true).")
    ap.add_argument("--baseline-flags-file", default=None,
                    help="Flags applied to every trial (incl. baselines). "
                         "Defaults to none — pure stock baseline.")
    ap.add_argument("--model", required=True,
                    help="Model name; must appear in VLLMTestParam.model_configs.")
    ap.add_argument("--benchmark-args-json", default=None,
                    help="JSON file of VLLMTestParam field overrides.")
    ap.add_argument("--target-metric", default="total_token_throughput")
    ap.add_argument("--slice-index", type=int, default=1)
    ap.add_argument("--slice-count", type=int, default=1)
    ap.add_argument("--baseline-runs", type=int, default=2)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flags = _load_lines(args.flag_list_file)
    baseline_flags = (
        _load_lines(args.baseline_flags_file) if args.baseline_flags_file else []
    )
    overrides = _load_json(args.benchmark_args_json)

    sliced = slice_flags(flags, args.slice_index, args.slice_count)
    trials = plan_ofat(sliced, baseline_flags, args.baseline_runs)

    os.makedirs(args.artifact_dir, exist_ok=True)
    manifest = {
        "model": args.model,
        "scheduler": "ofat",
        "slice_index": args.slice_index,
        "slice_count": args.slice_count,
        "total_flags": len(flags),
        "sliced_flags": sliced,
        "baseline_flags": baseline_flags,
        "baseline_runs": args.baseline_runs,
        "target_metric": args.target_metric,
        "trial_count": len(trials),
        "started_utc": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(args.artifact_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[autotune] shard {args.slice_index}/{args.slice_count}: "
        f"{len(sliced)} candidate flags + {args.baseline_runs} baselines "
        f"= {len(trials)} trials",
        flush=True,
    )

    summary_path = os.path.join(args.artifact_dir, "summary.jsonl")
    failures = 0
    with open(summary_path, "w") as summary_fp:
        for trial in trials:
            rec = _run_trial(
                trial=trial,
                model=args.model,
                overrides=overrides,
                artifact_dir=args.artifact_dir,
                target_metric=args.target_metric,
                dry_run=args.dry_run,
                summary_fp=summary_fp,
            )
            if not rec["success"]:
                failures += 1

    print(
        f"[autotune] DONE shard {args.slice_index}/{args.slice_count}: "
        f"{len(trials)} trials, {failures} failure(s). "
        f"artifacts: {args.artifact_dir}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

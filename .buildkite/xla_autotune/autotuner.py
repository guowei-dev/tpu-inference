#!/usr/bin/env python3
"""XLA flag auto-tuner.

Drives `vllm_test_framework.VLLMTestFramework` to evaluate a list of
candidate XLA / libtpu flags against a baseline.

The CLI is sharded — each invocation handles `--slice-index` of
`--slice-count` (1-based), so it can be fanned out across N Buildkite VMs
on a single matrix step.  Every shard *also* runs `--baseline-runs`
baseline trials (default 3) so the noise floor can be re-estimated per
machine.

Schedulers
----------

* ``ofat``        — one-flag-at-a-time.  Implemented.
* ``orthogonal``  — placeholder.  Raises NotImplementedError.
* ``optuna``      — placeholder.  Raises NotImplementedError.

Output
------

For every trial the autotuner writes ``<trial-id>.json`` into
``--artifact-dir`` *immediately* after the trial finishes, so partial
progress survives a crash.  The file is also appended to a per-shard
``summary.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Make the sibling vllm_test_framework importable.  The framework module
# lives in the same directory as this script when shipped inside the
# tpu-inference repo (.buildkite/xla_autotune/), and at the repo root when
# run from the autotuner dev tree (/workspace/).  Try both.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SELF_DIR))
for _p in (_SELF_DIR, _REPO_ROOT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Tell the framework that we are driving it from inside the autotuner so
# it does NOT silently extend LIBTPU_INIT_ARGS with its hard-coded list of
# previously-autotuned defaults — we want a clean, controllable baseline.
os.environ.setdefault("VLLM_IN_AUTOTUNER", "1")

# vLLM's recompilation guard would raise on every trial because each
# candidate flag invalidates the JAX cache.  Disable it inside the
# autotuner.
os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"

from vllm_test_framework import (  # noqa: E402
    VLLMTestFramework,
    VLLMTestParam,
    VLLMTestTask,
)


# ---------------------------------------------------------------------------
# Trial planning
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One unit of work: launch vllm with `extra_flags` and benchmark it."""
    trial_id: str
    kind: str            # "baseline" | "candidate"
    flag: Optional[str]  # None for baseline; one extra flag for candidate
    extra_flags: List[str] = field(default_factory=list)


def slice_flags(flags: List[str], slice_index: int, slice_count: int) -> List[str]:
    """Split `flags` into `slice_count` contiguous chunks and return chunk `slice_index` (1-based).

    100 flags / 4 shards → shard 1 gets flags[0:25], shard 2 flags[25:50], etc.
    Trailing remainder is appended to the last shard.
    """
    if slice_index < 1 or slice_index > slice_count:
        raise ValueError(f"slice_index {slice_index} out of [1, {slice_count}]")
    chunk = math.ceil(len(flags) / slice_count)
    start = (slice_index - 1) * chunk
    end = start + chunk if slice_index < slice_count else len(flags)
    return flags[start:end]


class Scheduler:
    """Base scheduler: turns a sliced flag list into a sequence of Trials."""

    def __init__(self, baseline_flags: List[str], baseline_runs: int):
        self.baseline_flags = list(baseline_flags)
        self.baseline_runs = baseline_runs

    def plan(self, sliced_flags: List[str]) -> List[Trial]:
        raise NotImplementedError


class OFATScheduler(Scheduler):
    """One-factor-at-a-time: baseline ×N then each candidate flag once."""

    def plan(self, sliced_flags: List[str]) -> List[Trial]:
        trials: List[Trial] = []
        for i in range(self.baseline_runs):
            trials.append(Trial(
                trial_id=f"baseline_{i+1:02d}",
                kind="baseline",
                flag=None,
                extra_flags=list(self.baseline_flags),
            ))
        for idx, flag in enumerate(sliced_flags):
            trials.append(Trial(
                trial_id=f"cand_{idx+1:03d}",
                kind="candidate",
                flag=flag,
                extra_flags=list(self.baseline_flags) + [flag],
            ))
        return trials


class OrthogonalScheduler(Scheduler):
    """Placeholder — orthogonal-array experiment design."""

    def plan(self, sliced_flags: List[str]) -> List[Trial]:
        raise NotImplementedError("orthogonal scheduler not implemented yet")


class OptunaScheduler(Scheduler):
    """Placeholder — Optuna-driven Bayesian search."""

    def plan(self, sliced_flags: List[str]) -> List[Trial]:
        raise NotImplementedError("optuna scheduler not implemented yet")


SCHEDULERS = {
    "ofat": OFATScheduler,
    "orthogonal": OrthogonalScheduler,
    "optuna": OptunaScheduler,
}


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def _load_lines(path: str) -> List[str]:
    """Read a file as a list of non-empty, non-comment lines."""
    with open(path) as f:
        return [
            ln.strip()
            for ln in f
            if ln.strip() and not ln.strip().startswith("#")
        ]


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def _build_test_param(
    model: str,
    extra_flags: List[str],
    base_param_overrides: Dict[str, Any],
    tag: str,
) -> VLLMTestParam:
    p = VLLMTestParam()
    p.model_name = model
    p.tag = tag
    # Per vllm_test_framework convention: XLA / libtpu autotune flags go
    # into LIBTPU_INIT_ARGS, not XLA_FLAGS.
    p.extra_libtpu_init_args = list(extra_flags)
    for k, v in (base_param_overrides or {}).items():
        # Allow `_`-prefixed keys (e.g. `_comment`) as JSON-comment metadata.
        if k.startswith("_"):
            continue
        if not hasattr(p, k):
            raise ValueError(f"unknown VLLMTestParam field: {k}")
        setattr(p, k, v)
    return p


def _run_trial(
    trial: Trial,
    model: str,
    base_param_overrides: Dict[str, Any],
    artifact_dir: str,
    target_metric: str,
    dry_run: bool,
    summary_fp,
) -> Dict[str, Any]:
    """Execute one trial and persist its artifact.  Never raises."""
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
    try:
        param = _build_test_param(
            model=model,
            extra_flags=trial.extra_flags,
            base_param_overrides=base_param_overrides,
            tag=f"autotune_{trial.trial_id}",
        )
        fw = VLLMTestFramework(params=param, dry_run=dry_run)
        results = fw.execute_task([VLLMTestTask.RUN_BENCHMARK_SERVING])
        result = results[0] if results else None
        if result is not None:
            record["success"] = bool(result.success)
            record["metrics"] = dict(result.metrics)
            record["error"] = result.error_message or ""
            # Resolve the target value: the framework's metrics dict is keyed
            # by per-benchmark task name (e.g. "benchmark_1024_1024"); pick
            # the *first* sub-result's `target_metric` for the summary.
            for _sub, sub_metrics in result.metrics.items():
                if isinstance(sub_metrics, dict) and target_metric in sub_metrics:
                    record["target_value"] = sub_metrics[target_metric]
                    break
    except Exception as e:  # noqa: BLE001
        record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    record["duration_sec"] = round(time.time() - t0, 2)
    record["finished_utc"] = datetime.utcnow().isoformat()

    # Write per-trial artifact + append to summary.jsonl right away.
    out_path = os.path.join(artifact_dir, f"{trial.trial_id}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    summary_fp.write(json.dumps(record) + "\n")
    summary_fp.flush()
    print(
        f"[autotune] {trial.trial_id} kind={trial.kind} "
        f"success={record['success']} {target_metric}={record['target_value']} "
        f"dur={record['duration_sec']}s -> {out_path}",
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag-list-file", required=True,
                    help="One XLA flag per line, e.g. --xla_tpu_foo=true")
    ap.add_argument("--baseline-flags-file", default=None,
                    help="Flags applied to every trial (incl. baselines). "
                         "Defaults to none — pure stock baseline.")
    ap.add_argument("--model", required=True,
                    help="Model name, must be a key in MODEL_CONFIGS.")
    ap.add_argument("--benchmark-args-json", default=None,
                    help="Optional JSON file overriding VLLMTestParam fields "
                         "(benchmark_args, model_benchmark_configs, …).")
    ap.add_argument("--target-metric", default="output_throughput",
                    help="Metric key inside each per-benchmark result dict.")
    ap.add_argument("--scheduler", choices=list(SCHEDULERS), default="ofat")
    ap.add_argument("--slice-index", type=int, default=1)
    ap.add_argument("--slice-count", type=int, default=1)
    ap.add_argument("--baseline-runs", type=int, default=3)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flags = _load_lines(args.flag_list_file)
    baseline_flags = (
        _load_lines(args.baseline_flags_file) if args.baseline_flags_file else []
    )
    base_param_overrides = _load_json(args.benchmark_args_json) or {}

    sliced = slice_flags(flags, args.slice_index, args.slice_count)
    scheduler = SCHEDULERS[args.scheduler](
        baseline_flags=baseline_flags,
        baseline_runs=args.baseline_runs,
    )
    trials = scheduler.plan(sliced)

    os.makedirs(args.artifact_dir, exist_ok=True)
    manifest = {
        "model": args.model,
        "scheduler": args.scheduler,
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
                base_param_overrides=base_param_overrides,
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

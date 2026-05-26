"""vLLM benchmark-serving framework for the XLA autotune pipeline.

Drives a `vllm serve` subprocess and runs benchmark_serving against it,
recording metrics + logs in a per-experiment directory.  Designed for
benchmark / autotune workloads: the recompilation guard env var
(`VLLM_XLA_CHECK_RECOMPILATION`) is force-disabled because benchmark
trials legitimately re-lower HLO modules whenever flag sets change.

Configuration is a single dataclass, `VLLMTestParam`.  All defaults
match what production tpu-inference CI uses, so a fresh
`VLLMTestParam()` produces an `LIBTPU_INIT_ARGS` string byte-identical
to the one the GCS-backed JAX compile cache is keyed on.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Production defaults
# ---------------------------------------------------------------------------

# 15-flag production-tuned XLA / libtpu set used by tpu-inference CI.
# Listed individually here for readability.  Serialised back into a single
# trailing-comma-terminated string so the resulting LIBTPU_INIT_ARGS env
# var byte-matches what production sets — and that's the string the GCS
# JAX compile cache is keyed on, so any deviation costs ~90 min of cold
# recompile per trial.
_PRODUCTION_LIBTPU_FLAGS: List[str] = [
    "--xla_always_enable_all_gather_2d_asymmetric=true",
    "--xla_tpu_vmem_scavenging_mode=SAFE",
    "--xla_tpu_flowdown_critical_nodes=false",
    "--xla_sc_num_serialized_tables_to_optimize_hbm=65536",
    "--xla_tpu_enable_domain_passes=true",
    "--xla_tpu_enable_expression_constant_splitter=false",
    "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=65536",
    "--xla_tpu_use_single_sparse_core_for_all_reduce_offload=true",
    "--xla_tpu_enable_glp_ring_reordering=false",
    "--xla_tpu_enable_async_pincer_short_emitter_for_cf=true",
    "--xla_sc_hbm_spill_stack=0",
    "--xla_collective_optimize_constant_table=ENABLED",
    "--xla_jf_fusion_max_instruction_count_for_window_config=65536",
    "--xla_enable_post_msa_sync_slice_fusion=true",
    "--xla_tpu_enable_all_experimental_scheduler_features=true",
]
PRODUCTION_LIBTPU_FLAGS_STRING: str = ",".join(_PRODUCTION_LIBTPU_FLAGS) + ","


DEFAULT_MODEL: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8000
DEFAULT_BENCHMARK_SCRIPT_PATH: str = "/workspace/bench_serving/benchmark_serving.py"
DEFAULT_XPROF_GCS_BASE: str = "gs://guoweij-inference-test/vllm-profile"


# Per-model vllm serve arg lists.
DEFAULT_MODEL_CONFIGS: Dict[str, List[str]] = {
    "Qwen/Qwen3-0.6B": [
        "--tensor-parallel-size=8",
        "--max-model-len=1024",
    ],
    "Qwen/Qwen3-30B-A3B-FP8": [
        "--tensor-parallel-size=8",
        "--data-parallel-size=1",
        "--max-model-len=10240",
        "--max-num-batched-tokens=8192",
        "--max-num-seqs=512",
        "--port=8000",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.95",
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        "--quantization=fp8",
    ],
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": [
        "--tensor-parallel-size=8",
        "--data-parallel-size=1",
        "--max-model-len=10240",
        "--max-num-batched-tokens=8192",
        "--max-num-seqs=512",
        "--port=8000",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.95",
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        "--quantization=fp8",
    ],
    "Qwen/Qwen3.5-397B-A17B-FP8": [
        "--max-model-len=9216",
        "--max-num-batched-tokens=8192",
        "--max-num-seqs=512",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.9",
        "--tensor-parallel-size=8",
        "--async-scheduling",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        '--limit-mm-per-prompt={"image": 0, "video": 0}',
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
    ],
}

# Per-model env vars to set on the vllm serve subprocess.
DEFAULT_MODEL_ENV_CONFIGS: Dict[str, Dict[str, str]] = {
    "Qwen/Qwen3-30B-A3B-FP8": {
        "PHASED_PROFILING_DIR": "",
    },
    "Qwen/Qwen3.5-397B-A17B-FP8": {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_MOE_EP_KERNEL": "0",
        "ATTN_BUCKETIZED_NUM_REQS": "true",
        "ATTN_CUSTOM_NUM_REQS_BUCKETS": "32,64,128,256,512",
        "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_p_recurrent_kernel_d",
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_BATCHED_RPA_KERNEL": "1",
    },
}

# Default benchmark_serving CLI args (applied to every (input_len, output_len)
# pass).  Boolean true → bare flag; everything else → `--key=value`.
DEFAULT_BENCHMARK_ARGS: Dict[str, Any] = {
    "--ignore-eos": True,
    "--dataset-name": "random",
    "--backend": "vllm",
    "--random-range-ratio": "0.8",
    "--num-prompts": "64",
    "--max-concurrency": "64",
    "--percentile-metrics": "ttft,tpot,itl,e2el",
    "--save-result": True,
    "--save-detailed": True,
}

# Per-model list of (random-input-len, random-output-len) shape pairs that
# the benchmark client should run.  Each shape produces one trial result.
DEFAULT_MODEL_BENCHMARK_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "Qwen/Qwen3.5-397B-A17B-FP8": [
        {"random-input-len": 1024, "random-output-len": 8192},
        {"random-input-len": 8192, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-30B-A3B-FP8": [
        {"random-input-len": 1024, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": [
        {"random-input-len": 8192, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-0.6B": [
        {"random-input-len": 512, "random-output-len": 256},
    ],
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

class VLLMTestTask(Enum):
    RUN_BENCHMARK_SERVING = "run_benchmark_serving"


@dataclass
class VLLMTestResult:
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    commands_run: List[List[str]] = field(default_factory=list)
    error_message: str = ""


@dataclass
class VLLMTestParam:
    """All configuration needed to run one benchmark trial."""

    # --- model selection ---
    model_name: str = DEFAULT_MODEL
    model_configs: Dict[str, List[str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_CONFIGS)
    )
    model_env_configs: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_ENV_CONFIGS)
    )

    # --- server endpoint ---
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # --- XLA / libtpu flags applied to the vllm serve subprocess.  Defaults
    # to the 15-flag production string so the GCS JAX compile cache hits. ---
    extra_libtpu_init_args: List[str] = field(
        default_factory=lambda: [PRODUCTION_LIBTPU_FLAGS_STRING]
    )

    # --- benchmark client ---
    benchmark_script_path: str = DEFAULT_BENCHMARK_SCRIPT_PATH
    benchmark_args: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_BENCHMARK_ARGS)
    )
    model_benchmark_configs: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_BENCHMARK_CONFIGS)
    )
    # Number of warmup benchmark passes per shape; all discarded.  The
    # (warmup_runs+1)th pass is recorded as the trial result.
    warmup_runs: int = 1

    # --- output ---
    # Per-experiment log directory root.  When None, falls back to the
    # legacy `<this_file>/scripts/log/` location.
    base_log_dir: Optional[str] = None
    tag: str = ""

    xprof_gcs_base: str = DEFAULT_XPROF_GCS_BASE


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class VLLMTestFramework:
    """Drives one vllm serve subprocess and a benchmark client against it."""

    def __init__(self, params: VLLMTestParam = None, dry_run: bool = False):
        atexit.register(self.stop_server)
        self.params = params if params is not None else VLLMTestParam()
        self.dry_run = dry_run
        self.server_process: Optional[subprocess.Popen] = None
        self.printing_server_output: bool = True

        # Resolve the per-experiment log directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = self.params.base_log_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scripts/log"
        )
        folder_name = (
            f"{self.params.tag}_EXP_{timestamp}"
            if self.params.tag
            else f"EXP_{timestamp}"
        )
        self.exp_dir = os.path.join(self.base_log_dir, folder_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Configure logging.
        self.main_log_file = os.path.join(self.exp_dir, "framework_main.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.main_log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment directory created: {self.exp_dir}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def setup_environment(self) -> None:
        """Build self.env: parent env + libtpu/xla flags + model-specific overrides."""
        self.env = os.environ.copy()

        # XLA / libtpu autotune flags.  Single space-joined string — the
        # byte-level format the GCS JAX compile cache is keyed on.
        self.env["LIBTPU_INIT_ARGS"] = " ".join(self.params.extra_libtpu_init_args)

        # Benchmark trials legitimately re-lower HLO whenever flag sets
        # change; the recompilation guard would treat that as an error.
        self.env["VLLM_XLA_CHECK_RECOMPILATION"] = "0"

        # Per-model env vars (e.g., MODEL_IMPL_TYPE, kernel switches).
        model_env = self.params.model_env_configs.get(self.params.model_name, {})
        for k, v in model_env.items():
            self.env[k] = v

        # XProf profile directory.  An explicit per-model override wins;
        # an explicit "disable" sentinel removes it; otherwise stamp a
        # fresh path under xprof_gcs_base.
        prof_override = model_env.get("PHASED_PROFILING_DIR")
        if "PHASED_PROFILING_DIR" not in model_env:
            prof_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.env["PHASED_PROFILING_DIR"] = f"{self.params.xprof_gcs_base}/{prof_dt}"
        elif prof_override in ("", None, "0", "OFF", "false", "False"):
            self.env.pop("PHASED_PROFILING_DIR", None)

        # Persist the resolved env + key params for post-mortem inspection.
        info_path = os.path.join(self.exp_dir, "experiment_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Model Name: {self.params.model_name}\n")
            f.write(f"Start Time: {datetime.now()}\n")
            f.write(f"Host: {self.params.host}, Port: {self.params.port}\n")
            f.write("-" * 20 + " Key Env Vars " + "-" * 20 + "\n")
            f.write(f"LIBTPU_INIT_ARGS={self.env.get('LIBTPU_INIT_ARGS')}\n")
            f.write(
                f"PHASED_PROFILING_DIR={self.env.get('PHASED_PROFILING_DIR', '(unset)')}\n"
            )
            for k in sorted(model_env):
                if k != "PHASED_PROFILING_DIR":
                    f.write(f"{k}={self.env.get(k)}\n")
            f.write("-" * 20 + " All Env Vars " + "-" * 20 + "\n")
            for k, v in sorted(self.env.items()):
                f.write(f"{k}={v}\n")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """Launch vllm serve in the background; wait for the port to bind."""
        self.logger.info("Starting vLLM server...")

        cmd = [
            "vllm",
            "serve",
            self.params.model_name,
            "--port",
            str(self.params.port),
        ]
        for raw_arg in self.params.model_configs.get(self.params.model_name, []):
            if "${HF_HOME}" in raw_arg and "HF_HOME" in os.environ:
                raw_arg = raw_arg.replace("${HF_HOME}", os.environ["HF_HOME"])
            cmd.append(raw_arg)

        if self.dry_run:
            self.logger.info(f"[Dry Run] Would start vLLM server: {' '.join(cmd)}")
            return

        with open(os.path.join(self.exp_dir, "vllm_server.cmd"), "w") as f:
            f.write(" ".join(cmd))

        server_log_path = os.path.join(self.exp_dir, "vllm_server.log")
        self.server_process = subprocess.Popen(
            cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
            errors="replace",
        )

        def _log_reader() -> None:
            with open(server_log_path, "w") as f:
                for line in iter(self.server_process.stdout.readline, ""):
                    f.write(line)
                    f.flush()
                    if self.printing_server_output:
                        print(line, end="", flush=True)

        self.server_log_thread = threading.Thread(target=_log_reader, daemon=True)
        self.server_log_thread.start()

        try:
            self._wait_for_port()
        finally:
            self.printing_server_output = False
            self.logger.info(
                "Server startup phase complete; stopping console output."
            )

    def _wait_for_port(self) -> None:
        self.logger.info(f"Waiting for port {self.params.port} to be ready...")
        start = time.time()
        while True:
            if self.server_process and self.server_process.poll() is not None:
                rc = self.server_process.poll()
                self.logger.error(f"Server process exited; return code: {rc}")
                raise RuntimeError(f"Server failed to start with return code {rc}")
            try:
                with socket.create_connection(
                    (self.params.host, self.params.port), timeout=1
                ):
                    self.logger.info("Server ready.")
                    return
            except OSError:
                elapsed = int(time.time() - start)
                self.logger.info(f"Server not ready yet; waited {elapsed}s...")
                time.sleep(10)

    def stop_server(self) -> None:
        """Terminate the vllm server process group: SIGTERM then SIGKILL."""
        if not self.server_process:
            self.logger.info("No running server process.")
            return
        pid = self.server_process.pid
        self.logger.info(f"Closing server process group (PID: {pid})...")
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.server_process.wait(timeout=20)
                self.logger.info("Server process group exited gracefully.")
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    f"Server (PID: {pid}) did not exit within 20s; force-killing the whole process group..."
                )
                os.killpg(pgid, signal.SIGKILL)
                self.server_process.wait()
                self.logger.info(
                    "Server process group force-killed; device memory should be released."
                )
        except ProcessLookupError:
            self.logger.info("Process already exited or does not exist.")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Unexpected error while shutting down server: {e}")
        finally:
            self.server_process = None

    # ------------------------------------------------------------------
    # Subprocess runner
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task_name: str,
        cmd_list: List[str],
        log_path: Optional[str] = None,
        mode: str = "w",
        result: Optional[VLLMTestResult] = None,
    ) -> int:
        """Run a subprocess, streaming its output to a log file."""
        cmd_str = " ".join(cmd_list)
        self.logger.info(f"Starting task [{task_name}]")
        if result is not None:
            result.commands_run.append(cmd_list)

        with open(os.path.join(self.exp_dir, f"task_{task_name}.cmd"), "w") as f:
            f.write(cmd_str)

        if self.dry_run:
            self.logger.info(f"[Dry Run] Would run: {cmd_str}")
            return 0

        if log_path is None:
            log_path = os.path.join(self.exp_dir, f"task_{task_name}.log")

        with open(log_path, mode) as log_file:
            process = subprocess.Popen(
                cmd_list,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            try:
                while True:
                    ch = process.stdout.read(1)
                    if not ch:
                        break
                    log_file.write(ch)
                    log_file.flush()
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print(f"\n[Ctrl+C] Terminating task [{task_name}]...")
                process.terminate()
                process.wait()
                print(f"Task [{task_name}] terminated.")
                self.logger.warning(f"Task [{task_name}] interrupted by user.")
                return 130
            process.wait()

            if self.server_process and self.server_process.poll() is not None:
                server_code = self.server_process.poll()
                if server_code != 0:
                    self.logger.error(
                        f"Server crashed with code {server_code} during task [{task_name}]"
                    )
                    raise RuntimeError(f"Server crashed with code {server_code}")

        if process.returncode == 0:
            self.logger.info(f"Task [{task_name}] succeeded.")
        else:
            self.logger.error(
                f"Task [{task_name}] FAILED with return code {process.returncode}. See {log_path}"
            )
        return process.returncode

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark_serving(
        self, result: Optional[VLLMTestResult] = None
    ) -> VLLMTestResult:
        """Run benchmark_serving.py: warmup_runs warmup passes + 1 measured pass per shape."""
        if result is None:
            result = VLLMTestResult()
        result.success = True

        warmup_runs = max(0, int(self.params.warmup_runs))
        total_passes = warmup_runs + 1

        self.logger.info("Starting benchmark_serving runs")
        summary_log_path = os.path.join(self.exp_dir, "benchmark_serving_all.log")
        with open(summary_log_path, "w") as f:
            f.write(f"=== Benchmark Serving Results {datetime.now()} ===\n")

        configs = self.params.model_benchmark_configs.get(
            self.params.model_name,
            [{"random-input-len": 8192, "random-output-len": 1024}],
        )

        for config in configs:
            input_len = config.get("random-input-len")
            output_len = config.get("random-output-len")
            base_task_name = f"benchmark_{input_len}_{output_len}"
            measured_result_path: Optional[str] = None

            self.logger.info(
                f"Running config input_len={input_len} output_len={output_len} "
                f"with {warmup_runs} warmup + 1 measured pass(es)"
            )

            for pass_idx in range(total_passes):
                is_measured = pass_idx == total_passes - 1
                task_name = (
                    base_task_name
                    if is_measured
                    else f"{base_task_name}_warmup{pass_idx + 1}"
                )
                current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"{task_name}_result_{current_dt}.json"

                cmd = ["python3", self.params.benchmark_script_path]
                for k, v in self.params.benchmark_args.items():
                    if isinstance(v, bool) and v:
                        cmd.append(k)
                    else:
                        cmd.append(f"{k}={str(v)}")
                cmd.extend(
                    [
                        f"--random-input-len={input_len}",
                        f"--random-output-len={output_len}",
                        f"--model={self.params.model_name}",
                        f"--port={str(self.params.port)}",
                        f"--result-dir={self.exp_dir}",
                        f"--result-filename={result_filename}",
                    ]
                )

                rc = self._run_task(
                    task_name,
                    cmd,
                    log_path=summary_log_path,
                    mode="a",
                    result=result,
                )
                if rc != 0:
                    result.success = False
                    result.error_message += f"Task {task_name} failed with code {rc}. "
                    break

                if self.dry_run:
                    if is_measured:
                        result.metrics[base_task_name] = {
                            "request_throughput": 10.0,
                            "output_throughput": 100.0,
                            "total_token_throughput": 1000.0,
                            "mean_ttft_ms": 50.0,
                            "mean_tpot_ms": 5.0,
                        }
                        self.logger.info(
                            f"[Dry Run] Filled default metrics for {base_task_name}."
                        )
                    continue

                if is_measured:
                    measured_result_path = os.path.join(self.exp_dir, result_filename)

            if self.dry_run or not result.success:
                continue

            if measured_result_path and os.path.exists(measured_result_path):
                self._record_measured_metrics(
                    measured_result_path, base_task_name, result, warmup_runs
                )
            else:
                self.logger.warning(
                    f"Measured result file not found: {measured_result_path}"
                )
                result.success = False
                result.error_message += (
                    f"Result file not found for {base_task_name}. "
                )

        return result

    def _record_measured_metrics(
        self,
        result_path: str,
        task_name: str,
        result: VLLMTestResult,
        warmup_runs: int,
    ) -> None:
        try:
            with open(result_path) as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to read result file: {e}")
            result.success = False
            result.error_message += f"Failed to read result file for {task_name}: {e}. "
            return

        flat = {
            k: v
            for k, v in data.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        flat.update(
            {
                "request_throughput": data.get("request_throughput", "N/A"),
                "output_throughput": data.get("output_throughput", "N/A"),
                "total_token_throughput": data.get("total_token_throughput", "N/A"),
                "mean_ttft_ms": data.get("mean_ttft_ms", "N/A"),
                "mean_tpot_ms": data.get("mean_tpot_ms", "N/A"),
                "warmup_runs": warmup_runs,
            }
        )
        result.metrics[task_name] = flat

        def fmt(val: Any) -> str:
            return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

        self.logger.info("\n" + "=" * 10 + f" {task_name} key metrics " + "=" * 10)
        self.logger.info(f"Throughput (Request/s): {fmt(flat['request_throughput'])}")
        self.logger.info(
            f"Throughput (Output Token/s): {fmt(flat['output_throughput'])}"
        )
        self.logger.info(
            f"Throughput (Total Token/s): {fmt(flat['total_token_throughput'])}"
        )
        self.logger.info(f"Mean TTFT (ms): {fmt(flat['mean_ttft_ms'])}")
        self.logger.info(f"Mean TPOT (ms): {fmt(flat['mean_tpot_ms'])}")
        self.logger.info("=" * 30 + "\n")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def execute_task(self, tasks: List[VLLMTestTask]) -> List[VLLMTestResult]:
        """Set up env, start server, run each task, stop server."""
        results: List[VLLMTestResult] = []
        self.setup_environment()
        self.start_server()
        try:
            for task in tasks:
                trial_result = VLLMTestResult()
                if task == VLLMTestTask.RUN_BENCHMARK_SERVING:
                    self.run_benchmark_serving(trial_result)
                else:
                    trial_result.success = False
                    trial_result.error_message = f"Unsupported task: {task}"
                results.append(trial_result)
        finally:
            self.stop_server()
        return results

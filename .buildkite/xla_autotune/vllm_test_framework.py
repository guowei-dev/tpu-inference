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

"""vLLM serve + benchmark_serving driver used by the XLA autotune sweep.

Spins up `vllm serve` with a caller-supplied LIBTPU_INIT_ARGS string, then
runs benchmark_serving.py against it across a list of (input_len, output_len)
shape pairs.  Each shape runs `warmup_runs` warmup passes (discarded) plus
one measured pass.

Configuration is a single dataclass, ``VLLMTestParam``.  Production defaults
are kept here; the autotuner CLI / config.json layer them on top.
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


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCHMARK_SCRIPT_PATH: str = os.path.join(
    _THIS_DIR, "bench_serving", "benchmark_serving.py"
)
DEFAULT_MODEL: str = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8000


# Per-model `vllm serve` argument list.  Add a new key here when onboarding a
# new model; the autotuner / config.json can also override per-run.
DEFAULT_MODEL_CONFIGS: Dict[str, List[str]] = {
    "Qwen/Qwen3.5-397B-A17B-FP8": [
        "--tensor-parallel-size=8",
        "--max-model-len=9216",
        "--max-num-batched-tokens=1024",
        "--max-num-seqs=64",
        "--gpu-memory-utilization=0.9",
        "--no-enable-prefix-caching",
        "--async-scheduling",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        '--limit-mm-per-prompt={"image": 0, "video": 0}',
        "--kv-cache-dtype=fp8",
        "--enable-expert-parallel",
        '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}',
        "--mamba-ssm-cache-dtype=bfloat16",
    ],
}

# Per-model env vars set on the vllm serve subprocess.
DEFAULT_MODEL_ENV_CONFIGS: Dict[str, Dict[str, str]] = {
    "Qwen/Qwen3.5-397B-A17B-FP8": {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_MOE_EP_KERNEL": "0",
        "ATTN_BUCKETIZED_NUM_REQS": "true",
        "ATTN_CUSTOM_NUM_REQS_BUCKETS": "8,16,32,64",
        "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_p_recurrent_kernel_d",
        "NEW_MODEL_DESIGN": "1",
    },
}

DEFAULT_BENCHMARK_ARGS: Dict[str, Any] = {
    "--ignore-eos": True,
    "--dataset-name": "random",
    "--backend": "vllm",
    "--random-range-ratio": "0.8",
    "--num-prompts": "640",
    "--max-concurrency": "512",
    "--percentile-metrics": "ttft,tpot,itl,e2el",
    "--save-result": True,
}

DEFAULT_MODEL_BENCHMARK_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "Qwen/Qwen3.5-397B-A17B-FP8": [
        {"random-input-len": 8192, "random-output-len": 1024},
    ],
}


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

    model_name: str = DEFAULT_MODEL
    model_configs: Dict[str, List[str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_CONFIGS)
    )
    model_env_configs: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_ENV_CONFIGS)
    )

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # XLA / libtpu flags appended (space-joined) into LIBTPU_INIT_ARGS.  Each
    # entry must be a single `--flag=value`; libtpu uses SPACE — not comma —
    # as the flag separator.
    extra_libtpu_init_args: List[str] = field(default_factory=list)

    benchmark_script_path: str = DEFAULT_BENCHMARK_SCRIPT_PATH
    benchmark_args: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_BENCHMARK_ARGS)
    )
    model_benchmark_configs: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_BENCHMARK_CONFIGS)
    )
    # Number of warmup benchmark passes per shape (results discarded).  The
    # final (warmup_runs+1)th pass is the measured result.
    warmup_runs: int = 1

    base_log_dir: Optional[str] = None
    tag: str = ""

    # Optional GCS prefix for xprof traces.  When None, PHASED_PROFILING_DIR
    # is not exported and vLLM skips profile capture.
    xprof_gcs_base: Optional[str] = None


class VLLMTestFramework:
    """Drives one vllm serve subprocess and a benchmark client against it."""

    def __init__(self, params: Optional[VLLMTestParam] = None, dry_run: bool = False):
        atexit.register(self.stop_server)
        self.params = params if params is not None else VLLMTestParam()
        self.dry_run = dry_run
        self.server_process: Optional[subprocess.Popen] = None
        self.printing_server_output: bool = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = self.params.base_log_dir or os.path.join(os.getcwd(), "logs")
        folder = f"{self.params.tag}_EXP_{timestamp}" if self.params.tag else f"EXP_{timestamp}"
        self.exp_dir = os.path.join(self.base_log_dir, folder)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.main_log_file = os.path.join(self.exp_dir, "framework_main.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.main_log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment directory created: {self.exp_dir}")

    def setup_environment(self) -> None:
        self.env = os.environ.copy()

        # The GCS-backed JAX compile cache is keyed on the byte-exact
        # LIBTPU_INIT_ARGS string, so the autotuner controls every byte that
        # lands here via params.extra_libtpu_init_args.
        self.env["LIBTPU_INIT_ARGS"] = " ".join(self.params.extra_libtpu_init_args)

        # Each trial legitimately re-lowers HLO whenever the flag set changes;
        # vLLM's recompilation guard would otherwise abort the run.
        self.env["VLLM_XLA_CHECK_RECOMPILATION"] = "0"

        model_env = self.params.model_env_configs.get(self.params.model_name, {})
        for k, v in model_env.items():
            self.env[k] = v

        prof_override = model_env.get("PHASED_PROFILING_DIR")
        if "PHASED_PROFILING_DIR" not in model_env and self.params.xprof_gcs_base:
            prof_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.env["PHASED_PROFILING_DIR"] = f"{self.params.xprof_gcs_base}/{prof_dt}"
        elif prof_override in ("", None, "0", "OFF", "false", "False"):
            self.env.pop("PHASED_PROFILING_DIR", None)

        info_path = os.path.join(self.exp_dir, "experiment_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Model Name: {self.params.model_name}\n")
            f.write(f"Start Time: {datetime.now()}\n")
            f.write(f"Host: {self.params.host}, Port: {self.params.port}\n")
            f.write("-" * 20 + " Key Env Vars " + "-" * 20 + "\n")
            f.write(f"LIBTPU_INIT_ARGS={self.env.get('LIBTPU_INIT_ARGS')}\n")
            f.write(f"PHASED_PROFILING_DIR={self.env.get('PHASED_PROFILING_DIR', '(unset)')}\n")
            for k in sorted(model_env):
                if k != "PHASED_PROFILING_DIR":
                    f.write(f"{k}={self.env.get(k)}\n")
            f.write("-" * 20 + " All Env Vars " + "-" * 20 + "\n")
            for k, v in sorted(self.env.items()):
                f.write(f"{k}={v}\n")

    def start_server(self) -> None:
        self.logger.info("Starting vLLM server...")

        cmd = ["vllm", "serve", self.params.model_name, "--port", str(self.params.port)]
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
            self.logger.info("Server startup phase complete; stopping console output.")

    def _wait_for_port(self) -> None:
        self.logger.info(f"Waiting for port {self.params.port} to be ready...")
        start = time.time()
        while True:
            if self.server_process and self.server_process.poll() is not None:
                rc = self.server_process.poll()
                self.logger.error(f"Server process exited; return code: {rc}")
                raise RuntimeError(f"Server failed to start with return code {rc}")
            try:
                with socket.create_connection((self.params.host, self.params.port), timeout=1):
                    self.logger.info("Server ready.")
                    return
            except OSError:
                self.logger.info(f"Server not ready yet; waited {int(time.time() - start)}s...")
                time.sleep(10)

    def stop_server(self) -> None:
        if not self.server_process:
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
                self.logger.warning(f"Server (PID: {pid}) did not exit in 20s; SIGKILL.")
                os.killpg(pgid, signal.SIGKILL)
                self.server_process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Unexpected error while shutting down server: {e}")
        finally:
            self.server_process = None

    def _run_task(
        self,
        task_name: str,
        cmd_list: List[str],
        log_path: Optional[str] = None,
        mode: str = "w",
        result: Optional[VLLMTestResult] = None,
    ) -> int:
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
                process.terminate()
                process.wait()
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
                f"Task [{task_name}] FAILED with return code {process.returncode}. "
                f"See {log_path}"
            )
        return process.returncode

    def run_benchmark_serving(
        self, result: Optional[VLLMTestResult] = None
    ) -> VLLMTestResult:
        if result is None:
            result = VLLMTestResult()
        result.success = True

        warmup_runs = max(0, int(self.params.warmup_runs))
        total_passes = warmup_runs + 1

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
                    base_task_name if is_measured
                    else f"{base_task_name}_warmup{pass_idx + 1}"
                )
                result_filename = (
                    f"{task_name}_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

                cmd = ["python3", self.params.benchmark_script_path]
                for k, v in self.params.benchmark_args.items():
                    if isinstance(v, bool) and v:
                        cmd.append(k)
                    else:
                        cmd.append(f"{k}={str(v)}")
                cmd.extend([
                    f"--random-input-len={input_len}",
                    f"--random-output-len={output_len}",
                    f"--model={self.params.model_name}",
                    f"--port={self.params.port}",
                    f"--result-dir={self.exp_dir}",
                    f"--result-filename={result_filename}",
                ])

                rc = self._run_task(
                    task_name, cmd,
                    log_path=summary_log_path, mode="a", result=result,
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
                self.logger.warning(f"Measured result file not found: {measured_result_path}")
                result.success = False
                result.error_message += f"Result file not found for {base_task_name}. "

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
            k: v for k, v in data.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        flat["warmup_runs"] = warmup_runs
        result.metrics[task_name] = flat

        def fmt(v: Any) -> str:
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

        self.logger.info(f"===== {task_name} key metrics =====")
        for key in (
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
        ):
            self.logger.info(f"  {key}: {fmt(data.get(key, 'N/A'))}")

    def execute_task(self, tasks: List[VLLMTestTask]) -> List[VLLMTestResult]:
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

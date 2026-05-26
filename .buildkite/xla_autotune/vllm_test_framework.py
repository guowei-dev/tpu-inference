#!/usr/local/bin/python
import os
import subprocess
import threading
import time
import socket
import logging
import signal
import json
from datetime import datetime
import atexit
import argparse
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

class VLLMTestTask(Enum):
    RUN_BENCHMARK_SERVING = "run_benchmark_serving"

@dataclass
class VLLMTestResult:
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    commands_run: List[List[str]] = field(default_factory=list)
    error_message: str = ""

# =====================================================================
# User Configuration Area
# =====================================================================
# DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
# DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-FP8"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
# DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-397B-A17B-FP8"
# DEFAULT_MODEL_NAME = "/workspace/persistent/trim_models/qwen3-coder-4layers-fp8"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Model-specific vllm serve args
MODEL_CONFIGS = {
    "Qwen/Qwen3-0.6B": [
        "--tensor-parallel-size=8",
        "--max-model-len=1024"
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
        "--quantization=fp8"
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
        "--quantization=fp8"
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
        # "--download_dir=${HF_HOME}",
    ],
    "/workspace/persistent/trim_models/qwen3-coder-4layers-fp8": [
        "--max-model-len=9216",
        "--max-num-batched-tokens=32768",
        "--max-num-seqs=512",
        "--kv-cache-dtype=fp8",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.15",
        "--tensor-parallel-size=8",
        # "--download_dir=${HF_HOME}",
        "--async-scheduling",
        "--block-size=256",
        "--enable-expert-parallel"
    ]
}

# Model-specific env vars
MODEL_ENV_CONFIGS = {
    "Qwen/Qwen3-30B-A3B-FP8": {
        "PHASED_PROFILING_DIR": "",
    },
    "Qwen/Qwen3.5-397B-A17B-FP8": {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_MOE_EP_KERNEL": "0",
        "ATTN_BUCKETIZED_NUM_REQS": "true",
        "ATTN_CUSTOM_NUM_REQS_BUCKETS": "32,64,128,256,512",
        "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_p_recurrent_kernel_d",
        # "PHASED_PROFILING_DIR": "",
    },
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_BATCHED_RPA_KERNEL": "1",
    }
}

# Default env / XLA flag settings
XPROF_GCS_BASE = "gs://guoweij-inference-test/vllm-profile"
EXTRA_XLA_FLAGS = [
    # "--xla_dump_hlo_as_proto",
    # "--xla_dump_hlo_as_text",
    # "--xla_tpu_autofdo_verbose=true"
]

VERBOSE_DEBUG_LOG = False

VERBOSE_DEBUG_ENV = {
    "TPU_LIBRARY_PATH": "/workspace/internal_lib/_libtpu.so",   
    "TPU_STDERR_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    # "TPU_VLOG_LEVEL": "1",
    "TPU_VMODULE": "tensorcore_codec_base=0,deepsea_compiler*=5",
}
extra_env = {
    # "VLLM_IN_AUTOTUNER":"1",
    # "HF_HUB_OFFLINE":"1",   
    
    # "VMODULE": "GLOG_v=4",
    # "TF_CPP_MIN_LOG_LEVEL": "0",
    # "TF_CPP_VMODULE": "*xla*=4,tpu_compile_op=4"

    # "TF_CPP_MAX_VLOG_LEVEL": "0",
    # "TF_CPP_VMODULE": "window_config_assignment_prod=4" 

    # "TF_CPP_MIN_VLOG_LEVEL": "0",

    # "TF_CPP_MAX_VLOG_LEVEL": "5",
    # "TF_CPP_MIN_LOG_LEVEL": "0",


    # Verbose debug logs moved to VERBOSE_DEBUG_ENV
    # "TF_CPP_VMODULE": "autofdo_agent=5,tpu_compilation_environment=5,tpu_compile_op_impl=5,deepsea_compiler_backend=5"  
    # "TF_CPP_VMODULE": "deepsea_compiler_autotuner_util=1,autofdo_profile_utils=1,utils=2,sstable=2,hlo_module_tpu_compiler=1,autofdo_agent=5,tpu_compilation_environment=5,tpu_compile_op_impl=5"   

    # "TF_CPP_VMODULE": "window_config_assignment_prod=4"

}

EXTRA_LIBTPU_INIT_ARGS = [
    # "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
    # "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
    # "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
    # "--xla_jf_crs_combiner_threshold_in_bytes=0",
    # "--xla_tpu_scheduler_percent_shared_memory_limit=1000",

    # "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"


    # "--xla_tpu_autotune_flags=true",
    # "--xla_tpu_autotune_layouts=true",  
    # "--xla_tpu_autotune_database=/workspace/tuning_data/242108008"
    # "--xla_tpu_autofdo_profile_dir=/workspace/tuning_data/242108008/profiles",
    # "--vmodule=autotune_utils=3",

    # "--xla_tpu_autofdo_op_windows=true",
    # "--xla_tpu_autofdo_profile_file=/workspace/tuning_data/242108008/profiles/fdo_profiles.sstable",

    # "--xla_tpu_autofdo=true",
    # "--xla_tpu_autofdo_module_flags=true",
    # "--xla_tpu_autofdo_profile_file_module_flag=/workspace/tuning_data/profiles/module_flag_fdo_profiles.sstable",
    # "--xla_tpu_autofdo_module_layouts=true",
    # "--xla_tpu_autofdo_profile_file_module_layout=/workspace/tuning_data/profiles/module_layout_fdo_profiles.sstable",

]


autotuned_result_list = [
# "--xla_jf_enable_producer_consumer_multi_output_fusion=false",
# "--xla_tpu_enable_domain_passes=true",
# "--xla_collective_optimize_constant_table=ENABLED",
# "--xla_jf_fusion_max_instruction_count_for_window_config=65536"

# "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=65536",
# "--xla_tpu_enable_expression_constant_splitter=false",
# "--xla_tpu_use_single_sparse_core_for_all_reduce_offload=true",
# "--xla_tpu_overlay_enable_hlo_function_packs=false",
# "--xla_tpu_padding_overhead_threshold_with_copy_for_reduce_scatter=0.0",
# "--xla_mosaic_deprecated_allow_implicit_single_buffering=false",

# "--xla_tpu_overlay_enable_hlo_function_packs=false", 
# "--xla_mosaic_deprecated_allow_implicit_single_buffering=false", 
# "--xla_tpu_allow_deeply_nested_fusion_numerical_diff=false", 
# "--xla_should_allow_loop_variant_parameter_in_chain=AUTO", 
# "--xla_tpu_enable_expression_constant_splitter=false", 
# "--xla_tpu_enable_glp_ring_reordering=false",


# From ortho
"--xla_always_enable_all_gather_2d_asymmetric=true,"
"--xla_tpu_vmem_scavenging_mode=SAFE,"
"--xla_tpu_flowdown_critical_nodes=false,"
"--xla_sc_num_serialized_tables_to_optimize_hbm=65536,"
"--xla_tpu_enable_domain_passes=true,"
"--xla_tpu_enable_expression_constant_splitter=false,"
"--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=65536,"
"--xla_tpu_use_single_sparse_core_for_all_reduce_offload=true,"
"--xla_tpu_enable_glp_ring_reordering=false,"
"--xla_tpu_enable_async_pincer_short_emitter_for_cf=true,"
"--xla_sc_hbm_spill_stack=0,"
"--xla_collective_optimize_constant_table=ENABLED,"
"--xla_jf_fusion_max_instruction_count_for_window_config=65536,"
"--xla_enable_post_msa_sync_slice_fusion=true,"
"--xla_tpu_enable_all_experimental_scheduler_features=true,"


]

if not os.environ.get("VLLM_IN_AUTOTUNER"):
    # for k, v in autotuned_result.items():
    #     EXTRA_LIBTPU_INIT_ARGS.append(f"--{k}={v}")
    EXTRA_LIBTPU_INIT_ARGS.extend(autotuned_result_list)


# Benchmark Serving configs
BENCHMARK_SCRIPT_PATH = "/workspace/bench_serving/benchmark_serving.py"
MODEL_BENCHMARK_CONFIGS = {
    "Qwen/Qwen3.5-397B-A17B-FP8": [
        # {"random-input-len": 1024, "random-output-len": 1024},
        {"random-input-len": 1024, "random-output-len": 8192},
        {"random-input-len": 8192, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-30B-A3B-FP8": [
        {"random-input-len": 1024, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": [
        # {"random-input-len": 1024, "random-output-len": 1024}
        {"random-input-len": 8192, "random-output-len": 1024},
    ],
    "Qwen/Qwen3-0.6B": [
        {"random-input-len": 512, "random-output-len": 256},
    ]
}
# Other benchmark_serving args
BENCHMARK_ARGS = {
    # "--num-warmups": "64",
    "--ignore-eos": True,
    "--dataset-name": "random",
    "--backend": "vllm",
    "--random-range-ratio": "0.8",
    "--num-prompts": "64",
    "--max-concurrency": "64",
    # "--request-rate": "inf",
    "--percentile-metrics": "ttft,tpot,itl,e2el",
    "--save-result": True,
    "--save-detailed": True,
}
# =====================================================================

@dataclass
class VLLMTestParam:
    model_name: str = DEFAULT_MODEL_NAME
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model_configs: Dict[str, List[str]] = field(default_factory=lambda: dict(MODEL_CONFIGS))
    model_env_configs: Dict[str, Dict[str, str]] = field(default_factory=lambda: dict(MODEL_ENV_CONFIGS))
    xprof_gcs_base: str = XPROF_GCS_BASE
    extra_xla_flags: List[str] = field(default_factory=lambda: list(EXTRA_XLA_FLAGS))
    verbose_debug_log: bool = VERBOSE_DEBUG_LOG
    verbose_debug_env: Dict[str, str] = field(default_factory=lambda: dict(VERBOSE_DEBUG_ENV))
    extra_env: Dict[str, str] = field(default_factory=lambda: dict(extra_env))
    extra_libtpu_init_args: List[str] = field(default_factory=lambda: list(EXTRA_LIBTPU_INIT_ARGS))
    benchmark_script_path: str = BENCHMARK_SCRIPT_PATH
    model_benchmark_configs: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: dict(MODEL_BENCHMARK_CONFIGS))
    benchmark_args: Dict[str, Any] = field(default_factory=lambda: dict(BENCHMARK_ARGS))
    max_backups: int = 0
    tag: str = ""
    dump_xla: bool = False
    tpu_inference_commit: str = None
    vllm_commit: str = None
    torchtpu_vllm_commit: str = None
    # Number of benchmark warmup runs to perform before the measured run.
    # The vllm server itself is already precompiled; this is benchmark-client
    # warmup (first-connection latency, HTTP keepalive, request scheduler
    # state). The measured run is the last one; warmup results are discarded.
    warmup_runs: int = 1
    # Override for the per-experiment log directory root.  When None the
    # framework writes under `<this_file>/scripts/log/` (legacy behaviour).
    # The autotuner sets this to an artifact-mounted path so every trial's
    # full log bundle (vllm server log, env dump, benchmark stdout, etc.)
    # is reachable by the host-side Buildkite artifact watcher.
    base_log_dir: Optional[str] = None



class VLLMTestFramework:
    def __init__(self, params: VLLMTestParam = None, dry_run=False):
        atexit.register(self.stop_server)
        if params is None:
            params = VLLMTestParam()
        self.params = params
        self.dry_run = dry_run
        self.server_process = None
        
        # 1. Create experiment directory.  If params.base_log_dir is set,
        # use it (e.g. the autotuner points this at a host-mounted artifact
        # path); otherwise fall back to the legacy in-tree `scripts/log/`.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = (
            self.params.base_log_dir
            if self.params.base_log_dir
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/log")
        )
        if self.params.tag:
            folder_name = f"{self.params.tag}_EXP_{timestamp}"
        else:
            folder_name = f"EXP_{timestamp}"
        self.exp_dir = os.path.join(self.base_log_dir, folder_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 2. Initialise the framework main log
        self.main_log_file = os.path.join(self.exp_dir, "framework_main.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.main_log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment directory created: {self.exp_dir}")
        
        # 3. Clean up stale experiment directories
        self.cleanup_old_experiments()
        
        # 4. Check out requested repo commits
        self.checkout_repositories()

    def cleanup_old_experiments(self):
        """Trim stale experiment directories — keep the latest 3 live, move the rest to backup."""
        backup_dir = os.path.join(self.base_log_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        import glob
        import shutil
        
        # List all directories that match EXP_*
        exp_dirs = [d for d in glob.glob(os.path.join(self.base_log_dir, "EXP_*")) if os.path.isdir(d)]
        # Sort by name
        exp_dirs.sort()
        
        if len(exp_dirs) > 3:
            dirs_to_move = exp_dirs[:-3]
            self.logger.info("More than 3 experiment directories found; trimming to the latest 3...")
            for d in dirs_to_move:
                self.logger.info(f"Moving {d} to {backup_dir}")
                try:
                    shutil.move(d, backup_dir)
                except Exception as e:
                    self.logger.error(f"Move failed: {e}")
                    
        # Trim backup directory down to max_backups entries
        backup_dirs = [d for d in glob.glob(os.path.join(backup_dir, "EXP_*")) if os.path.isdir(d)]
        backup_dirs.sort()
        
        max_backups = self.params.max_backups
            
        if max_backups == 0:
            dirs_to_delete = backup_dirs
        elif len(backup_dirs) > max_backups:
            dirs_to_delete = backup_dirs[:-max_backups]
        else:
            dirs_to_delete = []
            
        if dirs_to_delete:
            self.logger.info(f"Backup directory exceeds {max_backups} entries; pruning...")
            for d in dirs_to_delete:
                self.logger.info(f"Deleting backup: {d}")
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    self.logger.error(f"Delete failed: {e}")

    def checkout_repositories(self):
        """Check out the supplied commit hashes for the tpu_inference, vllm and torchtpu_vllm repos."""
        repos = [
            ("tpu_inference", "/workspace/tpu_inference", self.params.tpu_inference_commit),
            ("vllm", "/workspace/vllm", self.params.vllm_commit),
            ("torchtpu_vllm", "/workspace/torchtpu_vllm", self.params.torchtpu_vllm_commit)
        ]
        for name, path, commit in repos:
            if commit:
                self.logger.info(f"Checking out {name} repo (path: {path}) at commit: {commit}")
                if self.dry_run:
                    self.logger.info(f"[Dry Run] Would run: git checkout {commit} in {path}")
                    continue
                
                if not os.path.exists(path):
                    self.logger.error(f"Repository path does not exist: {path}; cannot check out commit.")
                    raise FileNotFoundError(f"Repository path not found: {path}")
                
                try:
                    result = subprocess.run(
                        ["git", "checkout", commit],
                        cwd=path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True
                    )
                    self.logger.info(f"{name} repo checked out at {commit}.\nOutput: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"{name} repo checkout FAILED.\nError: {e.stdout.strip()}")
                    raise RuntimeError(f"Failed to checkout {commit} in {path}") from e


    def setup_environment(self):
        """Configure the subprocess environment and record it to disk."""
        self.env = os.environ.copy()
        xla_flags = self.params.extra_xla_flags.copy()
        if self.params.dump_xla:
            xla_flags.insert(0, f"--xla_dump_to={self.exp_dir}/xla_logs")
        self.env["XLA_FLAGS"] = " ".join(xla_flags)
        self.env["LIBTPU_INIT_ARGS"] = f"{' '.join(self.params.extra_libtpu_init_args)}"
        for k, v in self.params.extra_env.items():
            self.env[k] = v  
        
        # Inject model-specific env vars
        model_specific_env = self.params.model_env_configs.get(self.params.model_name, {})
        for k, v in model_specific_env.items():
            self.env[k] = v
        
        if self.params.verbose_debug_log:
            self.logger.info("Verbose debug logs enabled")
            for k, v in self.params.verbose_debug_env.items():
                self.env[k] = v

        gs_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        xprof_dir = f"{self.params.xprof_gcs_base}/{gs_datetime}"
        if "PHASED_PROFILING_DIR" not in model_specific_env:
            self.env["PHASED_PROFILING_DIR"] = xprof_dir
        elif model_specific_env.get("PHASED_PROFILING_DIR") in ("", None, "0", "OFF", "false", "False"):
            if "PHASED_PROFILING_DIR" in self.env:
                self.env.pop("PHASED_PROFILING_DIR")
        
        # Write the key configuration for this experiment to disk
        info_path = os.path.join(self.exp_dir, "experiment_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Model Name: {self.params.model_name}\n")
            f.write(f"Start Time: {datetime.now()}\n")
            f.write(f"Host: {self.params.host}, Port: {self.params.port}\n")
            if self.params.tpu_inference_commit:
                f.write(f"TPU Inference Commit: {self.params.tpu_inference_commit}\n")
                self.logger.info(f"[Key Info] TPU Inference Commit: {self.params.tpu_inference_commit}")
            if self.params.vllm_commit:
                f.write(f"vLLM Commit: {self.params.vllm_commit}\n")
                self.logger.info(f"[Key Info] vLLM Commit: {self.params.vllm_commit}")
            if self.params.torchtpu_vllm_commit:
                f.write(f"TorchTPU vLLM Commit: {self.params.torchtpu_vllm_commit}\n")
                self.logger.info(f"[Key Info] TorchTPU vLLM Commit: {self.params.torchtpu_vllm_commit}")
            f.write("-" * 20 + " Key Env Vars " + "-" * 20 + "\n")
            f.write(f"XLA_FLAGS={self.env.get('XLA_FLAGS')}\n")
            f.write(f"LIBTPU_INIT_ARGS={self.env.get('LIBTPU_INIT_ARGS')}\n")
            f.write(f"PHASED_PROFILING_DIR={self.env.get('PHASED_PROFILING_DIR')}\n")
            for k in sorted(model_specific_env.keys()):
                if k != "PHASED_PROFILING_DIR":
                    f.write(f"{k}={self.env.get(k)}\n")
            f.write("-" * 20 + " All Env Vars " + "-" * 20 + "\n")
            for k, v in sorted(self.env.items()):
                f.write(f"{k}={v}\n")

    def _run_task(self, task_name, cmd_list, log_path=None, mode="w", result=None):
        """
        Core execution flow:
          1. Record a .cmd file containing the exact command that was run.
          2. Record a .log file with stdout and stderr.
        """
        cmd_str = " ".join(cmd_list)
        self.logger.info(f"Starting task [{task_name}]")
        
        if result is not None:
            result.commands_run.append(cmd_list)

        # Save the exact command that was run
        with open(os.path.join(self.exp_dir, f"task_{task_name}.cmd"), "w") as f:
            f.write(cmd_str)

        if self.dry_run:
            self.logger.info(f"[Dry Run] Would run: {cmd_str}")
            return 0

        # Run the command and redirect its output to a dedicated log file
        if log_path is None:
            log_path = os.path.join(self.exp_dir, f"task_{task_name}.log")
        with open(log_path, mode) as log_file:
            process = subprocess.Popen(
                cmd_list,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='replace'
            )
            
            try:
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        break
                    log_file.write(char)
                    log_file.flush()
                    import sys
                    sys.stdout.write(char)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print(f"\n[Ctrl+C] Terminating task [{task_name}]...")
                process.terminate()
                process.wait()
                print(f"Task [{task_name}] terminated.")
                self.logger.warning(f"Task [{task_name}] interrupted by user.")
                return 130
                
            process.wait()

            # Check if server crashed during the task
            if self.server_process and self.server_process.poll() is not None:
                server_code = self.server_process.poll()
                if server_code != 0:
                    self.logger.error(f"Server crashed with code {server_code} during task [{task_name}]")
                    raise RuntimeError(f"Server crashed with code {server_code}")

        if process.returncode == 0:
            self.logger.info(f"Task [{task_name}] succeeded.")
        else:
            self.logger.error(f"Task [{task_name}] FAILED with return code {process.returncode}. See {log_path}")
        return process.returncode

    def start_server(self):
        """Launch the vllm server, streaming its log to the experiment directory."""
        self.logger.info("Starting vLLM server...")
        
        # Build the vllm serve command for the chosen model
        cmd = ["vllm", "serve", self.params.model_name, "--port", str(self.params.port)]
        
        # Pick up model-specific vllm serve args
        extra_args = self.params.model_configs.get(self.params.model_name, [])
        # Expand env-var placeholders such as ${HF_HOME}
        processed_args = []
        for arg in extra_args:
            if "${HF_HOME}" in arg and "HF_HOME" in os.environ:
                arg = arg.replace("${HF_HOME}", os.environ["HF_HOME"])
            processed_args.append(arg)
            
        cmd.extend(processed_args)
        
        if self.dry_run:
            self.logger.info(f"[Dry Run] Would start vLLM server: {' '.join(cmd)}")
            return

        server_log_path = os.path.join(self.exp_dir, "vllm_server.log")
        
        # Persist the server launch command
        with open(os.path.join(self.exp_dir, "vllm_server.cmd"), "w") as f:
            f.write(" ".join(cmd))

        # Capture stdout via PIPE; a daemon thread writes to the log and to the console
        self.server_process = subprocess.Popen(
            cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid, text=True, errors='replace'
        )
        
        self.printing_server_output = True
        
        def log_reader():
            with open(server_log_path, "w") as f:
                for line in iter(self.server_process.stdout.readline, ""):
                    f.write(line)
                    f.flush()
                    if self.printing_server_output:
                        print(line, end="", flush=True)
                        
        self.server_log_thread = threading.Thread(target=log_reader, daemon=True)
        self.server_log_thread.start()

        try:
            self._wait_for_port()
        finally:
            self.printing_server_output = False
            self.logger.info("Server startup phase complete; stopping console output.")

    def _wait_for_port(self):
        self.logger.info(f"Waiting for port {self.params.port} to be ready...")
        start = time.time()
        # while time.time() - start < 360000000000000000000000:
        while True:
            # Check if the server process has died
            if self.server_process and self.server_process.poll() is not None:
                returncode = self.server_process.poll()
                self.logger.error(f"Server process exited; return code: {returncode}")
                raise RuntimeError(f"Server failed to start with return code {returncode}")
                
            try:
                with socket.create_connection((self.params.host, self.params.port), timeout=1):
                    self.logger.info("Server ready.")
                    return
            except:
                elapsed = time.time() - start
                self.logger.info(f"Server not ready yet; waited {int(elapsed)}s...")
                time.sleep(10)
        raise TimeoutError("Server startup timed out.")

    def stop_server(self):
            """Optimised process-group shutdown logic."""
            if not self.server_process:
                self.logger.info("No running server process.")
                return

            pid = self.server_process.pid
            self.logger.info(f"Closing server process group (PID: {pid})...")

            try:
                # Get the process-group ID (a new group was created in start_server via os.setsid).
                pgid = os.getpgid(pid)
                
                # 1. Try a graceful shutdown via SIGTERM.
                os.killpg(pgid, signal.SIGTERM)
                
                # 2. Wait for the process to exit, with a timeout (TPU release is slow, ~15-20s).
                try:
                    self.server_process.wait(timeout=20)
                    self.logger.info("Server process group exited gracefully.")
                except subprocess.TimeoutExpired:
                    # 3. Still alive after the timeout — force-kill with SIGKILL.
                    self.logger.warning(f"Server (PID: {pid}) did not exit within 20s; force-killing the whole process group...")
                    os.killpg(pgid, signal.SIGKILL)
                    
                    # Wait again to fully reap the zombie process.
                    self.server_process.wait()
                    self.logger.info("Server process group force-killed; device memory should be released.")
                    
            except ProcessLookupError:
                self.logger.info("Process already exited or does not exist.")
            except Exception as e:
                self.logger.error(f"Unexpected error while shutting down server: {e}")
            finally:
                self.server_process = None

    # --- Concrete business tasks ---

    def run_mmlu(self):
        # Run via the OpenAI-compatible API and uvx.
        cmd = [
            "uvx", "--with", "lm-evaluation-harness@", "lm_eval", 
            "--model", "openai", 
            "--model_args", f"model={self.params.model_name},base_url=http://{self.params.host}:{self.params.port}/v1", 
            "--tasks", "mmlu_llama", 
            "--limit", "10"
        ]
        self._run_task("mmlu", cmd)

    def run_evalplus(self, dataset="humaneval"):
        # Run via uvx.
        # Create the directory evalplus expects up front.
        os.makedirs(os.path.join(self.exp_dir, "evalplus", dataset), exist_ok=True)
        cmd = [
            "uvx", "--from", "evalplus==0.3.1", "evalplus.evaluate", 
            "--model", self.params.model_name, 
            "--dataset", dataset, 
            "--backend", "openai",
            "--base-url", f"http://{self.params.host}:{self.params.port}/v1", 
            "--greedy",
            "--root", os.path.join(self.exp_dir, "evalplus")
        ]
        self._run_task(dataset, cmd)

    def run_benchmark_serving(self, result=None):
        """Run the benchmark_serving client.

        Each (input_len, output_len) config is executed `warmup_runs + 1`
        times back-to-back against the same already-warm vllm server.  All
        but the last execution are warmup passes whose result JSONs are
        ignored; only the metrics from the final pass are recorded.
        """
        if result is None:
            result = VLLMTestResult()

        self.logger.info("Starting benchmark_serving runs")

        summary_log_path = os.path.join(self.exp_dir, "benchmark_serving_all.log")
        with open(summary_log_path, "w") as f:
            f.write(f"=== Benchmark Serving Results {datetime.now()} ===\n")

        configs = self.params.model_benchmark_configs.get(self.params.model_name, [
            {"random-input-len": 8192, "random-output-len": 1024}
        ])

        result.success = True  # Assume success unless a task fails

        warmup_runs = max(0, int(getattr(self.params, "warmup_runs", 0)))
        total_passes = warmup_runs + 1

        for config in configs:
            input_len = config.get("random-input-len")
            output_len = config.get("random-output-len")

            base_task_name = f"benchmark_{input_len}_{output_len}"
            self.logger.info(
                f"Running config input_len={input_len} output_len={output_len} "
                f"with {warmup_runs} warmup + 1 measured pass(es)"
            )

            measured_result_path = None
            measured_task_name = base_task_name

            for pass_idx in range(total_passes):
                is_measured = (pass_idx == total_passes - 1)
                kind = "measured" if is_measured else f"warmup{pass_idx + 1}"
                task_name = base_task_name if is_measured else f"{base_task_name}_{kind}"

                cmd = ["python3", self.params.benchmark_script_path]
                for k, v in self.params.benchmark_args.items():
                    if isinstance(v, bool) and v:
                        cmd.append(k)
                    else:
                        cmd.append(f"{k}={str(v)}")
                cmd.append(f"--random-input-len={input_len}")
                cmd.append(f"--random-output-len={output_len}")
                cmd.append(f"--model={self.params.model_name}")
                cmd.append(f"--port={str(self.params.port)}")
                cmd.append(f"--result-dir={self.exp_dir}")

                current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"{task_name}_result_{current_dt}.json"
                cmd.append(f"--result-filename={result_filename}")

                returncode = self._run_task(
                    task_name, cmd, log_path=summary_log_path, mode="a", result=result,
                )

                if returncode != 0:
                    result.success = False
                    result.error_message += f"Task {task_name} failed with code {returncode}. "
                    break

                if self.dry_run:
                    if is_measured:
                        result.metrics[measured_task_name] = {
                            'request_throughput': 10.0,
                            'output_throughput': 100.0,
                            'total_token_throughput': 1000.0,
                            'mean_ttft_ms': 50.0,
                            'mean_tpot_ms': 5.0,
                        }
                        self.logger.info(f"[Dry Run] Filled default metrics for {measured_task_name}.")
                    continue

                if is_measured:
                    measured_result_path = os.path.join(self.exp_dir, result_filename)

            if self.dry_run or not result.success:
                continue

            if measured_result_path and os.path.exists(measured_result_path):
                try:
                    with open(measured_result_path, "r") as f:
                        res_data = json.load(f)

                    req_thr = res_data.get('request_throughput', 'N/A')
                    tok_thr = res_data.get('output_throughput', 'N/A')
                    tot_thr = res_data.get('total_token_throughput', 'N/A')
                    ttft = res_data.get('mean_ttft_ms', 'N/A')
                    tpot = res_data.get('mean_tpot_ms', 'N/A')

                    flat = {
                        k: v for k, v in res_data.items()
                        if isinstance(v, (int, float, str, bool)) or v is None
                    }
                    flat.update({
                        'request_throughput': req_thr,
                        'output_throughput': tok_thr,
                        'total_token_throughput': tot_thr,
                        'mean_ttft_ms': ttft,
                        'mean_tpot_ms': tpot,
                        'warmup_runs': warmup_runs,
                    })
                    result.metrics[measured_task_name] = flat

                    def fmt(val):
                        return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

                    self.logger.info(f"\n" + "=" * 10 + f" {measured_task_name} key metrics " + "=" * 10)
                    self.logger.info(f"Throughput (Request/s): {fmt(req_thr)}")
                    self.logger.info(f"Throughput (Output Token/s): {fmt(tok_thr)}")
                    self.logger.info(f"Throughput (Total Token/s): {fmt(tot_thr)}")
                    self.logger.info(f"Mean TTFT (ms): {fmt(ttft)}")
                    self.logger.info(f"Mean TPOT (ms): {fmt(tpot)}")
                    self.logger.info("=" * 30 + "\n")
                except Exception as e:
                    self.logger.error(f"Failed to read result file: {e}")
                    result.success = False
                    result.error_message += f"Failed to read result file for {measured_task_name}: {e}. "
            else:
                self.logger.warning(f"Measured result file not found: {measured_result_path}")
                result.success = False
                result.error_message += f"Result file not found for {measured_task_name}. "

        return result

    def execute_task(self, tasks: List[VLLMTestTask]) -> List[VLLMTestResult]:
        """Execute the supplied task list from an external caller."""
        results = []
        
        self.setup_environment()
        self.start_server()
        
        try:
            for task in tasks:
                result = VLLMTestResult()
                if task == VLLMTestTask.RUN_BENCHMARK_SERVING:
                    self.run_benchmark_serving(result)
                else:
                    result.success = False
                    result.error_message = f"Unsupported task: {task}"
                results.append(result)
        finally:
            self.stop_server()
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=[e.value for e in VLLMTestTask], help="Task to run")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode")
    parser.add_argument("--max-backups", type=int, default=10, help="Max number of experiment folders to keep in backup")
    parser.add_argument("--tag", type=str, default="", help="Tag prefix for experiment folder")
    parser.add_argument("--dump-xla", action="store_true", help="Enable XLA dumping to experiment directory")
    parser.add_argument("--tpu-inference-commit", type=str, default=None, help="Commit hash for tpu_inference repository to checkout")
    parser.add_argument("--vllm-commit", type=str, default=None, help="Commit hash for vllm repository to checkout")
    parser.add_argument("--torchtpu-vllm-commit", type=str, default=None, help="Commit hash for torchtpu_vllm repository to checkout")
    args = parser.parse_args()

    params = VLLMTestParam(
        max_backups=args.max_backups, 
        tag=args.tag, 
        dump_xla=args.dump_xla,
        tpu_inference_commit=args.tpu_inference_commit,
        vllm_commit=args.vllm_commit,
        torchtpu_vllm_commit=args.torchtpu_vllm_commit
    )
    framework = VLLMTestFramework(params=params, dry_run=args.dry_run)
    framework.setup_environment()

    if args.task:
        task_enum = VLLMTestTask(args.task)
        results = framework.execute_task([task_enum])
        result = results[0]
        print(f"Task completed with success: {result.success}")
        print(f"Metrics: {result.metrics}")
        exit(0 if result.success else 1)

    try:
        # Interactive / online task loop.
        framework.start_server()
        def display_help():
            print("\n" + "="*10 + " API & HELP " + "="*10)
            print("1 + Enter : Run HumanEval")
            print("2 + Enter : Run MBPP")
            print("3 + Enter : Run Benchmark Serving")
            print("exit or Ctrl+C * 2 : Exit server")
            print(f"XProf Dir : {framework.env.get('PHASED_PROFILING_DIR')}")
            print("=" * 34)

        display_help()
        
        ctrl_c_count = 0
        # print("\nStarting HumanEval...")
        # framework.run_evalplus(dataset="humaneval")
        # warmup
        print("\nRunning Benchmark Serving (warmup)...")
        framework.run_benchmark_serving()
        # first run
        print("\nRunning Benchmark Serving (first run)...")
        framework.run_benchmark_serving()

        framework.run_evalplus(dataset="humaneval")
        # framework.run_evalplus(dataset="mbpp")
        display_help()
        framework.printing_server_output = True  # Show server output by default
        
        while True:
            try:
                user_input = input("Enter command > ").strip()
                ctrl_c_count = 0  # Reset Ctrl+C counter
                
                if user_input == "1":
                    framework.printing_server_output = False
                    print("\nStarting HumanEval...")
                    framework.run_evalplus(dataset="humaneval")
                    framework.printing_server_output = True
                    display_help()
                elif user_input == "2":
                    framework.printing_server_output = False
                    print("\nStarting MBPP...")
                    framework.run_evalplus(dataset="mbpp")
                    framework.printing_server_output = True
                    display_help()
                elif user_input == "3":
                    framework.printing_server_output = False
                    print("\nStarting Benchmark Serving...")
                    framework.run_benchmark_serving()
                    framework.printing_server_output = True
                    display_help()
                elif user_input == "exit":
                    print("Received exit command; shutting down...")
                    break
                elif user_input == "":
                    display_help()
                else:
                    print(f"Invalid input: {user_input}")
                    display_help()
                    
            except KeyboardInterrupt:
                ctrl_c_count += 1
                if ctrl_c_count >= 2:
                    print("\nReceived Ctrl+C twice; exiting...")
                    break
                print("\nPress Ctrl+C again to confirm exit, or type exit.")
                time.sleep(0.5)
                display_help()
            
    except KeyboardInterrupt:
        print("\nReceived stop signal; shutting down server...")
        
    finally:
        framework.stop_server()
        print(f"\nExperiment complete. All data saved to: {framework.exp_dir}")
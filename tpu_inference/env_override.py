# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os
import sys
import types

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
# AOT compile is currently a Torch-only feature and thus we should not enable it
# for TPU
os.environ["VLLM_USE_AOT_COMPILE"] = "0"

# Handle XLA CPU compilation warning.
os.environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX2 " + os.environ.get(
    "XLA_FLAGS", "")

# TODO: Remove this when SMEM capacity optimization for batched rpa lands.
os.environ[
    "LIBTPU_INIT_ARGS"] = "--xla_tpu_use_dynamic_smem_negotiation=true " + os.environ.get(
        "LIBTPU_INIT_ARGS", "")

# TODO: Remove this when the Shardy partitioner segfault in libtpu 0.0.41 is
# fixed upstream. libtpu's InsertExplicitReshardsPass crashes inside
# mlir::sdy::redistributeAxes when lowering certain vision-tower JIT regions
# (observed on Qwen3.5-4B during test_speculative_decoding::test_mtp_correctness).
# Falling back to the legacy GSPMD partitioner avoids the crash. Honour any
# value the caller has already set so opt-in Shardy testing still works.
os.environ.setdefault("JAX_USE_SHARDY_PARTITIONER", "false")

# Monkeypatch vLLM to avoid ImportError: cannot import name 'SamplingParams' from 'vllm'
# in vllm/v1/... submodules due to circular imports or lazy loading failures.
try:
    import vllm
    import vllm.sampling_params
    if not hasattr(vllm, "SamplingParams"):
        vllm.SamplingParams = vllm.sampling_params.SamplingParams
    if not hasattr(vllm, "SamplingType"):
        vllm.SamplingType = vllm.sampling_params.SamplingType
    if not hasattr(vllm, "SamplingStatus"):
        from vllm.sampling_params import RequestOutputKind
        vllm.RequestOutputKind = RequestOutputKind
except ImportError:
    pass

# Bypass cutlass installation requirement. It is unconditionally imported by
# upstream vLLM (e.g. DeepSeek V4 ops), but only actually invoked on NVIDIA GPUs.
if "cutlass" not in sys.modules:
    sys.modules["cutlass"] = types.ModuleType("cutlass")

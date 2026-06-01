"""Minimal reproducer for the libtpu 0.0.41 Shardy InsertExplicitReshardsPass
segfault triggered by Qwen3.5-4B's vision-tower JIT compile.

This is NOT a pytest test — it's a standalone script intended for upstream
libtpu bug reports and manual verification of any flag/version workaround.
The name does not start with ``test_`` so pytest collection skips it.

Background
----------
Under libtpu 0.0.41 with JAX 0.10.1, instantiating
``LLM(model="Qwen/Qwen3.5-4B", tensor_parallel_size=2, ...)`` segfaults
during the precompile of ``vllm embed_multimodal`` inside libtpu's Shardy
``InsertExplicitReshardsPass`` (stack: ``redistributeAxes`` ->
``processOp`` -> ``runOnOperation`` -> ``InsertExplicitReshards``). The
crash is purely a libtpu regression — the same vLLM invocation completes
cleanly when libtpu is downgraded to 0.0.40 with JAX 0.10.1.

Verified version matrix (tpu7x, this repro):
    libtpu 0.0.40 + jax 0.10.1, Shardy default-on : OK
    libtpu 0.0.41 + jax 0.10.1, Shardy default-on : SEGFAULT
    libtpu 0.0.41 + jax 0.10.1, Shardy off (env)  : OK

Usage
-----
    python3 tests/repros/libtpu_shardy_qwen35_4b_segfault.py
        # default = Shardy on, expected to segfault on libtpu 0.0.41

    python3 tests/repros/libtpu_shardy_qwen35_4b_segfault.py --shardy-off
        # exits cleanly on libtpu 0.0.41

Notes for upstream bug filing
-----------------------------
* Pure-JAX programs that look similar (sharded matmul + reshape +
  softmax on a 2-device Auto-typed mesh) do NOT reproduce locally. The
  trigger is specific to the IR shape produced by torchax bridging
  vLLM's Qwen3-VL-derived PyTorch vision tower into JAX; we have not
  isolated a non-torchax minimal trigger.
* The pre-compile that crashes is ``embed_multimodal GridTHW(((1,4,4),))``
  with ``num_patches=16``, ``pixel_values`` of shape ``(16, 1536)`` bf16.
* No tpu_inference / vllm code change between libtpu 0.0.40 and 0.0.41
  runs; the env var ``JAX_USE_SHARDY_PARTITIONER`` is the only knob
  toggled.
"""

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--shardy-on", action="store_true",
                   help="Force Shardy on (default — segfaults on libtpu 0.0.41).")
    g.add_argument("--shardy-off", action="store_true",
                   help="Force Shardy off via JAX_USE_SHARDY_PARTITIONER=false "
                        "and LIBTPU_INIT_ARGS=--xla_use_shardy=false.")
    args = p.parse_args()

    # env *must* be set before `from vllm import LLM` so jax.config picks it up
    if args.shardy_off:
        os.environ["JAX_USE_SHARDY_PARTITIONER"] = "false"
        libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
        if "--xla_use_shardy" not in libtpu:
            os.environ["LIBTPU_INIT_ARGS"] = "--xla_use_shardy=false " + libtpu
        mode = "shardy-off"
    elif args.shardy_on:
        os.environ["JAX_USE_SHARDY_PARTITIONER"] = "true"
        libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
        os.environ["LIBTPU_INIT_ARGS"] = libtpu.replace("--xla_use_shardy=false ", "")
        mode = "shardy-on"
    else:
        mode = "default (Shardy on as of jax 0.7.0+)"

    print(f"== mode: {mode}", flush=True)
    print(f"== JAX_USE_SHARDY_PARTITIONER="
          f"{os.environ.get('JAX_USE_SHARDY_PARTITIONER')!r}", flush=True)
    print(f"== LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '')!r}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        max_model_len=128,
        max_num_seqs=10,
        tensor_parallel_size=2,
        model_loader_extra_config={"enable_weights_track": False},
        async_scheduling=False,
        seed=42,
        max_num_batched_tokens=1024,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.90,
    )
    print("== LLM created", flush=True)

    out = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=4))
    print(f"== generated: {out[0].outputs[0].text!r}", flush=True)
    print("== DONE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

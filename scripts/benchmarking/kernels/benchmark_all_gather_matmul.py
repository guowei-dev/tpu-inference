# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused all-gather-matmul kernel against the XLA GSPMD path.

Shapes default to the Llama-70B MLP up/gate projection under tensor
parallelism with a sequence-sharded residual: for each M in the sweep,

    all_gather(x[M // tp, K], axis=0) @ y[K, 2N // tp]

with K = 8192, 2N = 57344 (hidden and fused gate+up FFN dims), M covering
decode through prefill token counts (the serving token-padding buckets).

Compared paths per M:
  xla:    jit einsum with GSPMD shardings (the collective-matmul compiler
          modes used by the serving path), XLA inserts the all-gather.
  fused:  tpu_inference.kernels.collectives.all_gather_matmul.
Correctness is checked per cell against the XLA path; timing is the median
per-call device time of the compiled module read from the JAX profiler.

Run on a TPU host, e.g.:
    python benchmark_all_gather_matmul.py --m 256,1024,8192 --reps 20
"""

import argparse
import glob
import os
import shutil
import statistics
import tempfile

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference import utils
from tpu_inference.kernels.collectives import all_gather_matmul

AXIS = "x"

XLA_SERVE_FLAGS = {
    "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
    "xla_tpu_reduce_scatter_collective_matmul_mode": "post_spmd_conservative",
}


def device_time_ms(fn, args, *, iters, warmup):
    """Median per-call device time (ms) of the compiled module, via profiler."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    trace_dir = tempfile.mkdtemp()
    try:
        with jax.profiler.trace(trace_dir):
            out = None
            for _ in range(iters):
                out = fn(*args)
            jax.block_until_ready(out)
        trace = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"),
                          recursive=True)[0]
        plane = jax.profiler.ProfileData.from_file(trace).find_plane_with_name(
            "/device:TPU:0")
        durations = []
        for line in plane.lines:
            if line.name != "XLA Modules":
                continue
            for event in line.events:
                for name, value in event.stats:
                    if name == "device_duration_ps":
                        durations.append(value / 1e9)
        return statistics.median(durations) if durations else float("nan")
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


def rel_err(out, ref):
    diff = jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32)))
    scale = jnp.max(jnp.abs(ref.astype(jnp.float32)))
    return float(diff / scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="16,32,64,128,256,512,1024,2048,4096,8192",
                        help="comma-separated token counts to sweep")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=57344)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bn", type=int, default=None,
                        help="kernel n-block size; defaults to n // tp")
    parser.add_argument("--bk", type=int, default=None,
                        help="kernel k-block size; defaults to k")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    tp = jax.device_count()
    mesh = utils.make_optimized_mesh((tp, ), (AXIS, ))
    dtype = jnp.dtype(args.dtype)
    # Pass block sizes explicitly: the kernel's tuned-size fallback derives bn
    # from the global n, which over-sizes VMEM scratch on untuned shapes.
    bn = args.bn if args.bn is not None else args.n // tp
    bk = args.bk if args.bk is not None else args.k

    print(f"devices={tp} ({jax.devices()[0].device_kind}), k={args.k}, "
          f"n={args.n} (n/tp={args.n // tp}), dtype={dtype.name}, "
          f"bn={bn}, bk={bk}")
    print(f"{'M':>6} | {'xla (ms)':>9} | {'fused (ms)':>10} | "
          f"{'speedup':>7} | {'rel_err':>8} | note")

    for m in [int(v) for v in args.m.split(",")]:
        key_x, key_y = jax.random.split(jax.random.key(0), 2)
        x = jax.device_put(
            jax.random.normal(key_x, (m, args.k), dtype) * 0.1,
            NamedSharding(mesh, P(AXIS, None)))
        y = jax.device_put(
            jax.random.normal(key_y, (args.k, args.n), dtype) * 0.1,
            NamedSharding(mesh, P(None, AXIS)))

        xla_fn = jax.jit(
            lambda x, y: jnp.einsum("mk,kn->mn", x, y),
            out_shardings=NamedSharding(mesh, P(None, AXIS))).lower(
                x, y).compile(compiler_options=XLA_SERVE_FLAGS)
        ref = jax.block_until_ready(xla_fn(x, y))
        xla_ms = device_time_ms(xla_fn, (x, y), iters=args.iters,
                                warmup=args.warmup)

        try:
            all_gather_matmul.validate_inputs(x, y, tp)
        except ValueError as e:
            print(f"{m:>6} | {xla_ms:>9.3f} | {'SKIP':>10} | {'':>7} | "
                  f"{'':>8} | unsupported shape: {e}")
            continue

        def fused_fn(x, y):
            return all_gather_matmul.all_gather_matmul(
                x, y, mesh=mesh, axis_name=AXIS, collective_id=0, bn=bn,
                bk=bk)

        fused_jit = jax.jit(fused_fn)
        try:
            out = jax.block_until_ready(fused_jit(x, y))
        except Exception as e:  # noqa: BLE001 - report infeasible cells
            reason = str(e).splitlines()[0][:80]
            print(f"{m:>6} | {xla_ms:>9.3f} | {'FAIL':>10} | {'':>7} | "
                  f"{'':>8} | {type(e).__name__}: {reason}")
            continue
        fused_ms = device_time_ms(fused_jit, (x, y), iters=args.iters,
                                  warmup=args.warmup)
        print(f"{m:>6} | {xla_ms:>9.3f} | {fused_ms:>10.3f} | "
              f"{xla_ms / fused_ms:>6.2f}x | {rel_err(out, ref):>8.1e} |")


if __name__ == "__main__":
    main()

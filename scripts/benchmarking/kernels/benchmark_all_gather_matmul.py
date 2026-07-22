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
"""Benchmark the all-gather + matmul collective at Llama-70B TP shapes.

For each token count M in the sweep this measures

    all_gather(x[M // tp, K], axis=0) @ y[K, N // tp]

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_all_gather_matmul.py \
        --m 256,1024,8192
"""

import functools

import jax
from collective_bench_lib import AXIS, Impl, P, run, xla_einsum_builder
from jax.sharding import NamedSharding

EINSUM = "mk,kn->mn"
OUT_SPEC = P(None, AXIS)


def make_inputs(mesh, m, k, n, dtype):
    """x[M, K] token-sharded, y[K, N] column-sharded over the TP axis."""
    key_x, key_y = jax.random.split(jax.random.key(0), 2)
    x = jax.device_put(
        jax.random.normal(key_x, (m, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    y = jax.device_put(
        jax.random.normal(key_y, (k, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    return jax.block_until_ready((x, y))


def build_agmm_kernel(mesh, inputs, bn=None, bk=None, bm="auto"):
    """The in-tree fused all-gather-matmul ring kernel (order-sensitive:
    run it on the 'gray' mesh). Block defaults follow the v7x/tp8 tuning:
    bn = per-device N, full-K bk, m-tiling once the per-device M outgrows
    the VMEM working set."""
    from tpu_inference.kernels.collectives import all_gather_matmul as agmm
    x, y = inputs
    tp = mesh.shape[AXIS]
    agmm.validate_inputs(x, y, tp)
    if bn is None:
        bn = y.shape[1] // tp
    if bk is None:
        bk = x.shape[1]
    if bm == "auto":
        bm = 128 if x.shape[0] // tp > 128 else None
    fn = jax.jit(
        functools.partial(agmm.all_gather_matmul,
                          mesh=mesh,
                          axis_name=AXIS,
                          collective_id=0,
                          bn=bn,
                          bk=bk,
                          **({
                              "bm": bm
                          } if bm is not None else {})))
    return fn.lower(*inputs).compile()


IMPLEMENTATIONS = {
    "xla": Impl(xla_einsum_builder(EINSUM, OUT_SPEC)),
    "xla_serve": Impl(xla_einsum_builder(EINSUM, OUT_SPEC, "serve")),
    "agmm_kernel": Impl(build_agmm_kernel, mesh="gray"),
}

if __name__ == "__main__":
    run(make_inputs=make_inputs,
        einsum=EINSUM,
        out_spec=OUT_SPEC,
        implementations=IMPLEMENTATIONS,
        default_n=57344)  # N = fused gate+up FFN dim (2 x 28672)

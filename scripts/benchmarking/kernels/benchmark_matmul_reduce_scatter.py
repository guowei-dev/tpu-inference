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
"""Benchmark the matmul + reduce-scatter collective at Llama-70B TP shapes.

For each token count M in the sweep this measures

    reduce_scatter(a[M, N // tp] @ w[N // tp, K], axis=0)

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
        --m 256,1024,8192
"""

import functools

import jax
from collective_bench_lib import AXIS, Impl, P, run, xla_einsum_builder
from jax.sharding import NamedSharding

EINSUM = "mn,nk->mk"
OUT_SPEC = P(AXIS, None)


def make_inputs(mesh, m, k, n, dtype):
    """a[M, N] and w[N, K] both contraction-sharded over the TP axis."""
    key_a, key_w = jax.random.split(jax.random.key(0), 2)
    a = jax.device_put(
        jax.random.normal(key_a, (m, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    w = jax.device_put(
        jax.random.normal(key_w, (n, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    return jax.block_until_ready((a, w))


def build_mmrs_kernel(mesh, inputs, bm=None, bk_out=None, bnc=None):
    """The in-tree fused matmul-reduce-scatter ring kernel (order-sensitive:
    run it on the 'gray' mesh)."""
    from tpu_inference.kernels.collectives import matmul_reduce_scatter as mmrs
    a, w = inputs
    tp = mesh.shape[AXIS]
    mmrs.validate_inputs(a, w, tp)
    fn = jax.jit(
        functools.partial(mmrs.matmul_reduce_scatter,
                          mesh=mesh,
                          axis_name=AXIS,
                          collective_id=1,
                          bm=bm,
                          bk_out=bk_out,
                          bnc=bnc))
    return fn.lower(*inputs).compile()


IMPLEMENTATIONS = {
    "xla": Impl(xla_einsum_builder(EINSUM, OUT_SPEC)),
    "xla_serve": Impl(xla_einsum_builder(EINSUM, OUT_SPEC, "serve")),
    "mmrs_kernel": Impl(build_mmrs_kernel, mesh="gray"),
}

if __name__ == "__main__":
    run(make_inputs=make_inputs,
        einsum=EINSUM,
        out_spec=OUT_SPEC,
        implementations=IMPLEMENTATIONS,
        default_n=28672)  # N = FFN intermediate dim (the contraction axis)

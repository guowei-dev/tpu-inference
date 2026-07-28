# SPDX-License-Identifier: Apache-2.0
"""Tests for the hierarchical (twin-pair + hypercube) reduce-scatter kernel.

First tests for this kernel: correctness against lax.psum_scatter (fp32 tight
bound + bf16 comparative bound vs an fp32 golden — the bf16 wire adds make an
absolute bound meaningless, same discipline as matmul_reduce_scatter's tests)
and a 10-repetition bitwise-determinism check (race exposure: the kernel has
no start-of-call barrier, so cross-call buffer reuse is the risk surface).

The kernel assumes logical id == chip*2 + core; the mesh comes from
topology.make_collective_mesh('hier'), which constructs (and asserts) exactly
that layout from the live coords.
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax._src import test_util as jtu

from tpu_inference.kernels.collectives import hierarchical_reduce_scatter as hrs
from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()

AXIS = 'x'


def _make_parts(mesh, s, h, dtype, seed=0):
    """One distinct [S, H] partial per device, stacked on a sharded axis."""
    n = mesh.devices.size
    parts = jax.random.normal(jax.random.key(seed), (n, s, h), dtype) * 0.1
    return jax.block_until_ready(
        jax.device_put(parts, NamedSharding(mesh, P(AXIS, None, None))))


def _hier_rs(mesh, parts, num_micro_batches=2):
    n = mesh.devices.size
    fn = jax.jit(jax.shard_map(
        lambda ps: hrs.hierarchical_reduce_scatter_local(
            ps[0], n, num_micro_batches=num_micro_batches, axis_name=AXIS),
        mesh=mesh, in_specs=P(AXIS, None, None), out_specs=P(AXIS, None),
        check_vma=False))
    return fn(parts)


def _xla_rs(mesh, parts):
    fn = jax.jit(jax.shard_map(
        lambda ps: lax.psum_scatter(ps[0], AXIS, scatter_dimension=0,
                                    tiled=True),
        mesh=mesh, in_specs=P(AXIS, None, None), out_specs=P(AXIS, None),
        check_vma=False))
    return fn(parts)


def _golden(parts):
    """fp32 sum of the partials — the reduce-scatter output, unsharded."""
    return jnp.sum(parts.astype(jnp.float32), axis=0)


def _rel_err(out, golden):
    out = jax.device_put(out.astype(jnp.float32),
                         jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    golden = jax.device_put(golden,
                            jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    return float(jnp.max(jnp.abs(out - golden)) / jnp.max(jnp.abs(golden)))


class HierarchicalReduceScatterTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')
        self.mesh = topology.make_collective_mesh('hier', axis_name=AXIS)

    def test_fp32_close_to_xla_reference(self):
        # fp32 in, fp32 adds — only accumulation ORDER differs from
        # psum_scatter, so the bound can be tight.
        parts = _make_parts(self.mesh, 256, 1024, jnp.float32)
        out = jax.block_until_ready(_hier_rs(self.mesh, parts))
        ref = jax.block_until_ready(_xla_rs(self.mesh, parts))
        self.assertAllClose(out, ref, atol=1e-4, rtol=1e-4)

    def _bf16_case(self, s, h, num_micro_batches):
        parts = _make_parts(self.mesh, s, h, jnp.bfloat16)
        golden = _golden(parts)
        out = jax.block_until_ready(
            _hier_rs(self.mesh, parts, num_micro_batches))
        ref = jax.block_until_ready(_xla_rs(self.mesh, parts))
        kernel_err = _rel_err(out, golden)
        xla_err = _rel_err(ref, golden)
        # bf16 wire adds: comparative bound vs the XLA path, never absolute.
        self.assertLessEqual(kernel_err, 1.5 * max(xla_err, 1e-6))

    def test_bf16_no_worse_than_xla(self):
        self._bf16_case(256, 1024, num_micro_batches=2)

    def test_bf16_larger_shape_mb4(self):
        self._bf16_case(1024, 4096, num_micro_batches=4)

    def test_race_ten_runs_bitwise_identical(self):
        parts = _make_parts(self.mesh, 256, 1024, jnp.bfloat16)
        first = None
        for _ in range(10):
            out = np.asarray(
                jax.block_until_ready(_hier_rs(self.mesh, parts)))
            if first is None:
                first = out
            else:
                self.assertArraysEqual(out, first)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

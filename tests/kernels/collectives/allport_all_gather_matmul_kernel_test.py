# SPDX-License-Identifier: Apache-2.0
"""Tests for the all-port pipelined all-gather-matmul kernel.

Same numerics discipline as the hier/ring AG-MM tests: fp32 inputs make the
wire exact (tight bound, carries the schedule-correctness burden); bf16 gets
a comparative bound vs the XLA path; 10 repetitions expose races — the prime
risk surface here is the fine-grained interleaved choreography, whose static
schedule is separately verified by the CPU simulator in
allport_ag_schedule_test.py. The shape grid covers the M=128 floor (16 rows
-> two 8-row bands) and an uneven-band shape (M=192: 24 rows -> 8 + 16).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax._src import test_util as jtu

from tpu_inference.kernels.collectives import allport_all_gather_matmul as aag
from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()

AXIS = 'x'


def _make_inputs(mesh, m, k, n, dtype, seed=0):
    kx, ky = jax.random.split(jax.random.key(seed), 2)
    x = jax.device_put(
        jax.random.normal(kx, (m, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    y = jax.device_put(
        jax.random.normal(ky, (k, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    return jax.block_until_ready((x, y))


def _xla_reference(mesh, x, y):
    fn = jax.jit(jax.shard_map(
        lambda xs, ys: jnp.dot(
            lax.all_gather(xs, AXIS, axis=0, tiled=True), ys,
            preferred_element_type=jnp.float32).astype(xs.dtype),
        mesh=mesh, in_specs=(P(AXIS, None), P(None, AXIS)),
        out_specs=P(None, AXIS), check_vma=False))
    return fn(x, y)


def _golden(x, y):
    return jnp.einsum('mk,kn->mn', x.astype(jnp.float32),
                      y.astype(jnp.float32))


def _rel_err(out, golden):
    dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    out = jax.device_put(out.astype(jnp.float32), dev)
    golden = jax.device_put(golden, dev)
    return float(jnp.max(jnp.abs(out - golden)) / jnp.max(jnp.abs(golden)))


class AllportAllGatherMatmulTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')
        self.mesh = topology.make_collective_mesh('hier', axis_name=AXIS)
        self.kernel = jax.jit(functools.partial(
            aag.allport_all_gather_matmul, mesh=self.mesh, axis_name=AXIS))

    @parameterized.product(m=[128, 192, 1024], k=[512, 1024])
    def test_f32_exact(self, m, k):
        x, y = _make_inputs(self.mesh, m, k, 2048, jnp.float32)
        out = jax.block_until_ready(self.kernel(x, y))
        ref = jax.block_until_ready(_xla_reference(self.mesh, x, y))
        self.assertAllClose(out, ref, atol=1e-4, rtol=1e-4)

    def test_bf16_no_worse_than_xla(self):
        x, y = _make_inputs(self.mesh, 1024, 1024, 2048, jnp.bfloat16)
        golden = _golden(x, y)
        out = jax.block_until_ready(self.kernel(x, y))
        ref = jax.block_until_ready(_xla_reference(self.mesh, x, y))
        self.assertLessEqual(_rel_err(out, golden),
                             1.5 * max(_rel_err(ref, golden), 1e-6))

    def test_race_ten_runs_bitwise_identical(self):
        x, y = _make_inputs(self.mesh, 256, 512, 2048, jnp.bfloat16)
        first = None
        for _ in range(10):
            out = np.asarray(jax.block_until_ready(self.kernel(x, y)))
            if first is None:
                first = out
            else:
                self.assertArraysEqual(out, first)

    def test_rejects_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, 'divisible by 128'):
            aag.validate_inputs(jnp.zeros((32, 64)), jnp.zeros((64, 256)), 8)
        with self.assertRaisesRegex(ValueError, 'row bands'):
            aag.validate_inputs(jnp.zeros((8, 512)), jnp.zeros((512, 256)), 8)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

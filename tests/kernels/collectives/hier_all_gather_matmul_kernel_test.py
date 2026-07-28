# SPDX-License-Identifier: Apache-2.0
"""Tests for the hierarchical all-gather matmul kernel.

The gather itself is exact (no wire adds), so bf16 compares directly against
the all_gather + fp32-accumulated dot reference at the ring AGMM test's
tolerance; fp32 gets a tight bound; 10 repetitions expose races.
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

from tpu_inference.kernels.collectives import hier_all_gather_matmul as hag
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
        lambda xs, ys: jnp.dot(lax.all_gather(xs, AXIS, axis=0, tiled=True),
                               ys, preferred_element_type=jnp.float32
                               ).astype(xs.dtype),
        mesh=mesh, in_specs=(P(AXIS, None), P(None, AXIS)),
        out_specs=P(None, AXIS), check_vma=False))
    return fn(x, y)


class HierAllGatherMatmulTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')
        self.mesh = topology.make_collective_mesh('hier', axis_name=AXIS)
        self.kernel = jax.jit(functools.partial(
            hag.hier_all_gather_matmul, mesh=self.mesh, axis_name=AXIS))

    @parameterized.product(m=[256, 1024], k=[512, 1024])
    def test_bf16_matches_reference(self, m, k):
        x, y = _make_inputs(self.mesh, m, k, 2048, jnp.bfloat16)
        out = jax.block_until_ready(self.kernel(x, y))
        ref = jax.block_until_ready(_xla_reference(self.mesh, x, y))
        self.assertAllClose(out, ref, atol=1e-2, rtol=1e-2)

    def test_f32_tight(self):
        x, y = _make_inputs(self.mesh, 256, 512, 2048, jnp.float32)
        out = jax.block_until_ready(self.kernel(x, y))
        ref = jax.block_until_ready(_xla_reference(self.mesh, x, y))
        self.assertAllClose(out, ref, atol=1e-4, rtol=1e-4)

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
            hag.validate_inputs(jnp.zeros((32, 100)), jnp.zeros((100, 256)), 8)
        with self.assertRaisesRegex(ValueError, 'sublane'):
            hag.validate_inputs(jnp.zeros((4, 128)), jnp.zeros((128, 256)), 8)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

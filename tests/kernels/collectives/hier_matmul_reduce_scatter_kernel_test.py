# SPDX-License-Identifier: Apache-2.0
"""Tests for the hierarchical fused matmul-reduce-scatter kernel.

Same numerics discipline as matmul_reduce_scatter_kernel_test: fp32 inputs
make the wire exact so ring/schedule logic is validated tightly; bf16 gets a
comparative bound against the XLA path (absolute bounds are meaningless under
bf16 wire rounding); 10 repetitions expose races.
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

from tpu_inference.kernels.collectives import hier_matmul_reduce_scatter as hmr
from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()

AXIS = 'x'


def _make_inputs(mesh, m, n, k, dtype, seed=0):
    ka, kw = jax.random.split(jax.random.key(seed), 2)
    a = jax.device_put(
        jax.random.normal(ka, (m, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    w = jax.device_put(
        jax.random.normal(kw, (n, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    return jax.block_until_ready((a, w))


def _xla_reference(mesh, a, w):
    fn = jax.jit(jax.shard_map(
        lambda as_, ws: lax.psum_scatter(
            jnp.dot(as_, ws, preferred_element_type=jnp.float32
                    ).astype(as_.dtype),
            AXIS, scatter_dimension=0, tiled=True),
        mesh=mesh, in_specs=(P(None, AXIS), P(AXIS, None)),
        out_specs=P(AXIS, None), check_vma=False))
    return fn(a, w)


def _golden(a, w):
    return jnp.einsum('mn,nk->mk', a.astype(jnp.float32),
                      w.astype(jnp.float32))


def _rel_err(out, golden):
    dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    out = jax.device_put(out.astype(jnp.float32), dev)
    golden = jax.device_put(golden, dev)
    return float(jnp.max(jnp.abs(out - golden)) / jnp.max(jnp.abs(golden)))


class HierMatmulReduceScatterTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')
        self.mesh = topology.make_collective_mesh('hier', axis_name=AXIS)
        self.kernel = jax.jit(functools.partial(
            hmr.hier_matmul_reduce_scatter, mesh=self.mesh, axis_name=AXIS))

    @parameterized.product(m=[256, 1024], k=[512, 1024])
    def test_f32_exact(self, m, k):
        a, w = _make_inputs(self.mesh, m, 2048, k, jnp.float32)
        out = jax.block_until_ready(self.kernel(a, w))
        ref = jax.block_until_ready(_xla_reference(self.mesh, a, w))
        self.assertAllClose(out, ref, atol=1e-4, rtol=1e-4)

    def test_bf16_no_worse_than_xla(self):
        a, w = _make_inputs(self.mesh, 1024, 2048, 1024, jnp.bfloat16)
        golden = _golden(a, w)
        out = jax.block_until_ready(self.kernel(a, w))
        ref = jax.block_until_ready(_xla_reference(self.mesh, a, w))
        kernel_err = _rel_err(out, golden)
        xla_err = _rel_err(ref, golden)
        self.assertLessEqual(kernel_err, 1.5 * max(xla_err, 1e-6))

    def test_race_ten_runs_bitwise_identical(self):
        a, w = _make_inputs(self.mesh, 256, 2048, 512, jnp.bfloat16)
        first = None
        for _ in range(10):
            out = np.asarray(jax.block_until_ready(self.kernel(a, w)))
            if first is None:
                first = out
            else:
                self.assertArraysEqual(out, first)

    def test_rejects_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, 'divisible by 128'):
            hmr.validate_inputs(jnp.zeros((256, 64)), jnp.zeros((64, 100)), 8)
        with self.assertRaisesRegex(ValueError, 'sublane'):
            hmr.validate_inputs(jnp.zeros((8, 64)), jnp.zeros((64, 128)), 8)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

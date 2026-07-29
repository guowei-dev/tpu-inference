# SPDX-License-Identifier: Apache-2.0
"""Tests for the all-port pipelined matmul-reduce-scatter kernel.

fp32 inputs make the wire exact, so the tight bound there carries the
schedule-correctness burden; bf16 gets a comparative bound against the XLA
path, whose own error is the same size as the kernel's; and 10 repetitions
expose races, which is the prime risk surface for a fine-grained interleaved
choreography. The static schedule behind it is verified separately by the CPU
simulator in allport_schedule_test.py.

The largest case crosses the VMEM->HBM residency switch by shape alone, so
both run-slot residencies are covered.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.collectives import \
    allport_matmul_reduce_scatter as amr
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

    def _matmul_reduce_scatter(as_, ws):
        prod = jnp.dot(as_, ws, preferred_element_type=jnp.float32)
        return lax.psum_scatter(prod.astype(as_.dtype),
                                AXIS,
                                scatter_dimension=0,
                                tiled=True)

    fn = jax.jit(
        jax.shard_map(_matmul_reduce_scatter,
                      mesh=mesh,
                      in_specs=(P(None, AXIS), P(AXIS, None)),
                      out_specs=P(AXIS, None),
                      check_vma=False))
    return fn(a, w)


def _golden(a, w):
    return jnp.einsum('mn,nk->mk', a.astype(jnp.float32),
                      w.astype(jnp.float32))


def _rel_err(out, golden):
    dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    out = jax.device_put(out.astype(jnp.float32), dev)
    golden = jax.device_put(golden, dev)
    return float(jnp.max(jnp.abs(out - golden)) / jnp.max(jnp.abs(golden)))


class ValidateInputsTest(jtu.JaxTestCase):

    def test_rejects_unaligned_k(self):
        with self.assertRaisesRegex(ValueError, 'divisible by 128'):
            amr._validate_inputs(jnp.zeros((256, 64)), jnp.zeros((64, 100)), 8)

    def test_rejects_k_too_small_to_band(self):
        # k must split into num_dims non-empty 128-column bands; k=128 on 4
        # chips leaves the second band empty.
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            amr._validate_inputs(jnp.zeros((256, 64)), jnp.zeros((64, 128)), 8)

    def test_rejects_unaligned_rows(self):
        with self.assertRaisesRegex(ValueError, 'sublane'):
            amr._validate_inputs(jnp.zeros((8, 64)), jnp.zeros((64, 512)), 8)


class AllportMatmulReduceScatterTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')
        self.mesh = topology.make_collective_mesh(axis_name=AXIS)
        self.kernel = jax.jit(
            functools.partial(amr.allport_matmul_reduce_scatter,
                              mesh=self.mesh,
                              axis_name=AXIS))

    @parameterized.product(m=[256, 1024], k=[512, 1024])
    def test_f32_exact(self, m, k):
        a, w = _make_inputs(self.mesh, m, 2048, k, jnp.float32)
        out = jax.block_until_ready(self.kernel(a, w))
        ref = jax.block_until_ready(_xla_reference(self.mesh, a, w))
        self.assertAllClose(out, ref, atol=1e-4, rtol=1e-4)

    @parameterized.parameters(256, 1024)
    def test_bf16_no_worse_than_xla(self, m):
        # m=256 takes the whole-a path, whose own slots are at wire dtype;
        # m=1024 takes the chunked path with an fp32 accumulator.
        a, w = _make_inputs(self.mesh, m, 2048, 1024, jnp.bfloat16)
        golden = _golden(a, w)
        out = jax.block_until_ready(self.kernel(a, w))
        ref = jax.block_until_ready(_xla_reference(self.mesh, a, w))
        self.assertLessEqual(_rel_err(out, golden),
                             1.5 * max(_rel_err(ref, golden), 1e-6))

    def test_hbm_residency(self):
        # Chosen to exceed the scoped-VMEM budget so the run slots stream from
        # HBM: the store staging, the slot-0-aliased-to-output path and the
        # read-modify-write merges only run here.
        m, n, k = 8192, 8192, 4096
        self.assertGreater(
            amr.get_vmem_estimate_bytes(m // 8, n // 8, k, 2048, 4, 2048, 2) +
            8 * 1024 * 1024, amr._vmem_cap_bytes())
        a, w = _make_inputs(self.mesh, m, n, k, jnp.bfloat16)
        golden = _golden(a, w)
        out = jax.block_until_ready(self.kernel(a, w))
        ref = jax.block_until_ready(_xla_reference(self.mesh, a, w))
        self.assertLessEqual(_rel_err(out, golden),
                             1.5 * max(_rel_err(ref, golden), 1e-6))

    @parameterized.parameters((256, 2048, 512), (8192, 8192, 4096))
    def test_race_ten_runs_bitwise_identical(self, m, n, k):
        a, w = _make_inputs(self.mesh, m, n, k, jnp.bfloat16)
        first = None
        for _ in range(10):
            out = np.asarray(jax.block_until_ready(self.kernel(a, w)))
            if first is None:
                first = out
            else:
                self.assertArraysEqual(out, first)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

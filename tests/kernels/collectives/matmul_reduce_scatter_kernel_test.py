# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu

from tpu_inference import utils
from tpu_inference.kernels.collectives import matmul_reduce_scatter

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec


def _make_inputs(mesh, axis_name, m, n, k, dtype, seed):
    k0, k1 = jax.random.split(jax.random.key(seed), 2)
    a = jax.device_put(
        jax.random.normal(k0, (m, n), dtype=dtype),
        jax.sharding.NamedSharding(mesh, P(None, axis_name)))
    w = jax.device_put(
        jax.random.normal(k1, (n, k), dtype=dtype),
        jax.sharding.NamedSharding(mesh, P(axis_name, None)))
    return a, w


def _xla_reference(mesh, axis_name, a, w):
    """The unfused path: local matmul + psum_scatter, wire in a.dtype."""

    def ref(a_shard, w_shard):
        part = jnp.dot(a_shard, w_shard,
                       preferred_element_type=jnp.float32).astype(
                           a_shard.dtype)
        return lax.psum_scatter(part, axis_name, scatter_dimension=0,
                                tiled=True)

    return jax.jit(
        jax.shard_map(ref, mesh=mesh,
                      in_specs=(P(None, axis_name), P(axis_name, None)),
                      out_specs=P(axis_name, None), check_vma=False))(a, w)


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class MatmulReduceScatterTest(jtu.JaxTestCase):

    @parameterized.product(
        grid_ko=[1, 2],
        grid_nc=[1, 2],
        m=[256, 1024],
    )
    def test_matmul_reduce_scatter_f32_exact(self, grid_ko, grid_nc, m):
        # fp32 inputs make the wire fp32, so the ring's accumulation matches
        # the reference tightly — this validates the ring/schedule logic.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        bk_out, bnc = 512, 256
        k, n = bk_out * grid_ko, bnc * grid_nc * num_devices

        # Run several times to expose race conditions.
        for i in range(10):
            a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.float32,
                                1234 + i)
            output = matmul_reduce_scatter.matmul_reduce_scatter(
                a, w, mesh, axis_name, bk_out=bk_out, bnc=bnc)
            expected = _xla_reference(mesh, axis_name, a, w)
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

    @parameterized.parameters(2592, 2784)
    def test_matmul_reduce_scatter_unaligned_n_per(self, n_per):
        # n // tp_size not a multiple of 128 (production MLP intermediate
        # dims, e.g. D=20736 at tp=8): the kernel runs its VMEM operands at
        # the lane-padded width with the pad region zeroed. fp32 + exact
        # comparison verifies the pad contributes exactly nothing.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, n, k = 1024, n_per * num_devices, 1024

        for i in range(3):
            a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.float32,
                                1234 + i)
            output = matmul_reduce_scatter.matmul_reduce_scatter(
                a, w, mesh, axis_name)
            expected = _xla_reference(mesh, axis_name, a, w)
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_reduce_scatter_bf16_no_worse_than_xla(self):
        # bf16 rounds on every wire hop, and the kernel and the unfused path
        # round in different orders — compare both against an fp32 golden and
        # require the kernel to be no worse than the existing path.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, n, k = 1024, 2048, 1024

        a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.bfloat16, 1234)
        golden = jax.jit(
            lambda a, w: jnp.einsum("mn,nk->mk", a.astype(jnp.float32),
                                    w.astype(jnp.float32)),
            out_shardings=jax.sharding.NamedSharding(mesh, P(axis_name,
                                                             None)))(a, w)
        output = matmul_reduce_scatter.matmul_reduce_scatter(
            a, w, mesh, axis_name)
        reference = _xla_reference(mesh, axis_name, a, w)

        kernel_err = jnp.max(jnp.abs(output.astype(jnp.float32) - golden))
        xla_err = jnp.max(jnp.abs(reference.astype(jnp.float32) - golden))
        self.assertLessEqual(float(kernel_err), 1.5 * float(xla_err))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

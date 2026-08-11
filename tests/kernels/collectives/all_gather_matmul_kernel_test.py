# SPDX-License-Identifier: Apache-2.0

import os

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference import utils
from tpu_inference.kernels.collectives import all_gather_matmul

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

SpongeDir: str | None = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', None)


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class AllGatherMatmulTest(jtu.JaxTestCase):

    @parameterized.product(
        grid_k=[1, 2, 3],
        grid_n=[1, 2, 3],
        rhs_transpose=[True, False],
    )
    def test_all_gather_matmul(self, grid_k, grid_n, rhs_transpose):
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        bk, bn = 1024, 1024
        m, k, n = 1024, bk * grid_k, bn * grid_n * num_devices

        # Run the test 10 times to expose race conditions as much as possible.
        for i in range(10):
            # Create input data
            prng_key = jax.random.key(1234 + i)
            k0, k1 = jax.random.split(prng_key, 2)
            x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
            y_shape = (n, k) if rhs_transpose else (k, n)
            y_sharding = P(axis_name, None) if rhs_transpose else P(
                None, axis_name)
            y = jax.random.normal(k1, y_shape, dtype=jnp.bfloat16)
            sharded_x = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, P(axis_name, None)))
            sharded_y = jax.device_put(
                y, jax.sharding.NamedSharding(mesh, y_sharding))

            # Run the all_gather_matmul function
            output = all_gather_matmul.all_gather_matmul(
                sharded_x,
                sharded_y,
                mesh,
                axis_name,
                bk=bk,
                bn=bn,
                rhs_transpose=rhs_transpose,
            )
            y_for_dot = sharded_y.T if rhs_transpose else sharded_y
            expected_output = jnp.dot(sharded_x, y_for_dot)
            self.assertAllClose(output, expected_output, atol=1e-2, rtol=1e-2)

    @parameterized.parameters(
        (128, 1, False),  # fold=8, one group
        (256, 1, False),  # fold=8
        (256, 2, False),  # fold x grid_k
        (512, 1, True),   # fold=4 x rhs_transpose
        (512, 2, False),  # fold=4 x grid_k
    )
    def test_all_gather_matmul_folded_small_m(self, m, grid_k,
                                              rhs_transpose):
        # m_per_device < 128 folds the per-chunk dots into full-height
        # super-step dots; grid_n == 2 exercises the per-unit export path.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        bk, bn = 1024, 1024
        k, n = bk * grid_k, bn * 2 * num_devices

        for i in range(3):
            k0, k1 = jax.random.split(jax.random.key(7 + i), 2)
            x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
            y_shape = (n, k) if rhs_transpose else (k, n)
            y_sharding = P(axis_name, None) if rhs_transpose else P(
                None, axis_name)
            sharded_x = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, P(axis_name, None)))
            sharded_y = jax.device_put(
                jax.random.normal(k1, y_shape, dtype=jnp.bfloat16),
                jax.sharding.NamedSharding(mesh, y_sharding))

            output = all_gather_matmul.all_gather_matmul(
                sharded_x,
                sharded_y,
                mesh,
                axis_name,
                bk=bk,
                bn=bn,
                rhs_transpose=rhs_transpose,
            )
            y_for_dot = sharded_y.T if rhs_transpose else sharded_y
            expected_output = jnp.dot(sharded_x, y_for_dot)
            self.assertAllClose(output, expected_output, atol=1e-2, rtol=1e-2)

    @parameterized.parameters(
        (1024, 2592, False),
        (1024, 2592, True),
        (1024, 2784, False),
        (256, 2592, False),   # fold x unaligned n_per
        (512, 2784, True),    # fold x rhs_transpose x unaligned n_per
    )
    def test_all_gather_matmul_unaligned_n_per(self, m, n_per, rhs_transpose):
        # n // tp_size not a multiple of 128 (production MLP intermediate
        # dims, e.g. D=20736 at tp=8). bn defaults to n_per, so the kernel
        # emits the whole n extent and no block offset needs proving.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        k, n = 1024, n_per * num_devices

        for i in range(3):
            k0, k1 = jax.random.split(jax.random.key(99 + i), 2)
            x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
            y_shape = (n, k) if rhs_transpose else (k, n)
            y_sharding = P(axis_name, None) if rhs_transpose else P(
                None, axis_name)
            sharded_x = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, P(axis_name, None)))
            sharded_y = jax.device_put(
                jax.random.normal(k1, y_shape, dtype=jnp.bfloat16),
                jax.sharding.NamedSharding(mesh, y_sharding))

            output = all_gather_matmul.all_gather_matmul(
                sharded_x,
                sharded_y,
                mesh,
                axis_name,
                rhs_transpose=rhs_transpose,
            )
            y_for_dot = sharded_y.T if rhs_transpose else sharded_y
            expected_output = jnp.dot(sharded_x, y_for_dot)
            self.assertAllClose(output, expected_output, atol=1e-2, rtol=1e-2)

    def test_all_gather_matmul_rejects_ragged_bn(self):
        # A bn that blocks n must tile it exactly and lane-aligned; a ragged
        # tail block reads and writes out of bounds, which cost a slice.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, k, n = 1024, 1024, 2592 * num_devices
        x = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
        y = jax.ShapeDtypeStruct((k, n), jnp.bfloat16)
        with self.assertRaisesRegex(ValueError, 'multiple of 128'):
            jax.eval_shape(
                lambda a, b: all_gather_matmul.all_gather_matmul(
                    a, b, mesh, axis_name, bn=2560), x, y)

    @parameterized.product(rhs_transpose=[True, False])
    def test_all_gather_matmul_unrolled_matches_grid(self, rhs_transpose):
        # The unrolled-ring variant must be interchangeable with the grid
        # kernel on the single-block schedule: same result (bitwise, checked
        # across 10 runs to expose races) and the same reference tolerance.
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, k, n = 1024, 2048, 1024 * num_devices
        bn, bk = n // num_devices, k

        for i in range(10):
            k0, k1 = jax.random.split(jax.random.key(4321 + i), 2)
            x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
            y_shape = (n, k) if rhs_transpose else (k, n)
            y_sharding = P(axis_name, None) if rhs_transpose else P(
                None, axis_name)
            sharded_x = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, P(axis_name, None)))
            sharded_y = jax.device_put(
                jax.random.normal(k1, y_shape, dtype=jnp.bfloat16),
                jax.sharding.NamedSharding(mesh, y_sharding))

            kwargs = dict(bn=bn, bk=bk, rhs_transpose=rhs_transpose)
            out_grid = all_gather_matmul.all_gather_matmul(sharded_x,
                                                           sharded_y,
                                                           mesh,
                                                           axis_name,
                                                           unroll=False,
                                                           **kwargs)
            out_unrolled = all_gather_matmul.all_gather_matmul(sharded_x,
                                                               sharded_y,
                                                               mesh,
                                                               axis_name,
                                                               unroll=True,
                                                               **kwargs)
            self.assertArraysEqual(out_grid, out_unrolled)
            y_for_dot = sharded_y.T if rhs_transpose else sharded_y
            self.assertAllClose(out_unrolled,
                                jnp.dot(sharded_x, y_for_dot),
                                atol=1e-2,
                                rtol=1e-2)

    def test_all_gather_matmul_unrolled_rejects_multi_block(self):
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, k, n = 256, 1024, 512 * num_devices

        k0, k1 = jax.random.split(jax.random.key(1234), 2)
        sharded_x = jax.device_put(
            jax.random.normal(k0, (m, k), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(axis_name, None)))
        sharded_y = jax.device_put(
            jax.random.normal(k1, (k, n), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, axis_name)))

        with self.assertRaisesRegex(ValueError, 'unroll=True requires'):
            all_gather_matmul.all_gather_matmul(sharded_x,
                                                sharded_y,
                                                mesh,
                                                axis_name,
                                                bn=n // num_devices // 2,
                                                unroll=True)

    def test_all_gather_matmul_default_block_sizes(self):
        # Regression test: with no tuned entry, the default bn used to resolve
        # to the GLOBAL n instead of n // tp_size, issuing out-of-bounds DMA on
        # the per-device [k, n // tp_size] y block (TPU slice failure at
        # runtime when the oversized scratch still fit VMEM).
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, k, n = 256, 1024, 512 * num_devices  # no tuned entry for this shape

        k0, k1 = jax.random.split(jax.random.key(1234), 2)
        sharded_x = jax.device_put(
            jax.random.normal(k0, (m, k), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(axis_name, None)))
        sharded_y = jax.device_put(
            jax.random.normal(k1, (k, n), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, axis_name)))

        output = all_gather_matmul.all_gather_matmul(sharded_x, sharded_y,
                                                     mesh, axis_name)
        expected_output = jnp.dot(sharded_x, sharded_y)
        self.assertAllClose(output, expected_output, atol=1e-2, rtol=1e-2)

    def test_all_gather_matmul_rejects_oversized_bn(self):
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        m, k, n = 256, 1024, 512 * num_devices

        k0, k1 = jax.random.split(jax.random.key(1234), 2)
        sharded_x = jax.device_put(
            jax.random.normal(k0, (m, k), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(axis_name, None)))
        sharded_y = jax.device_put(
            jax.random.normal(k1, (k, n), dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, axis_name)))

        with self.assertRaisesRegex(ValueError, "bn .* must be <="):
            all_gather_matmul.all_gather_matmul(sharded_x,
                                                sharded_y,
                                                mesh,
                                                axis_name,
                                                bn=n)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

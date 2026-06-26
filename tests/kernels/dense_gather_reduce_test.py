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

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sparse_core.dense_gather_reduce import \
    dense_gather_reduce

jax.config.parse_flags_with_absl()


def reference_dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Reference implementation of dense gather reduce."""
    topk_weights_1d = topk_weights.reshape(-1)
    weights_for_mul = topk_weights_1d[:, None].astype(jnp.float32)
    gathered = x[indices].astype(jnp.float32)
    if topk_wgt_zero_nan:
        gathered = jnp.where(weights_for_mul == 0.0, 0.0,
                             gathered * weights_for_mul)
    else:
        gathered = gathered * weights_for_mul
    out = gathered.reshape(-1, reduce_group_size, gathered.shape[-1])
    out = jnp.sum(out, axis=1).astype(x.dtype)
    return out


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DenseGatherReduceTest(jtu.JaxTestCase):
    _test_cases = [
        dict(
            out_size=o,
            hidden_size=h,
            dtype=d,
            reduce_group_size=rg,
            topk_wgt_zero_nan=wzn,
        ) for o, h, d, rg, wzn in itertools.chain(
            itertools.product(
                [400, 840],
                [128, 512, 8192],
                [jnp.bfloat16, jnp.float32],
                [8, 5],
                [False, True],
            ),
            itertools.product(
                [16384],
                [7168],
                [jnp.bfloat16],
                [8],
                [False, True],
            ),
            itertools.product(
                [16384],
                [6144],
                [jnp.bfloat16],
                [8],
                [False],
            ),
            itertools.product(
                [20480],
                [4096],
                [jnp.bfloat16],
                [10],
                [False],
            ),
        )
    ]

    def setUp(self):
        super().setUp()
        try:
            tpu_info = pltpu.get_tpu_info()
            sc_info = tpu_info.sparse_core
        except ValueError:
            tpu_info = None
            sc_info = None
        if sc_info is None:
            self.skipTest("SparseCore is not available")
        if tpu_info.generation == 6:
            self.skipTest("dense_gather_reduce is not supported on TPUv6e")

    @parameterized.parameters(*_test_cases)
    def test_sc_dense_gather_reduce(self, out_size, hidden_size, dtype,
                                    reduce_group_size, topk_wgt_zero_nan):
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(
            key, (out_size // reduce_group_size, reduce_group_size),
            jnp.bfloat16)

        if topk_wgt_zero_nan:
            mask = jax.random.bernoulli(key, 0.2, topk_weights.shape)
            topk_weights = jnp.where(mask, 0.0, topk_weights)

        actual = dense_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
        )
        # Correctness check.
        desired = reference_dense_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
        )
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

    def test_nan_mitigation(self):
        """Verifies that topk_wgt_zero_nan=True correctly zeroes out NaNs."""
        try:
            sc_info = pltpu.get_tpu_info().sparse_core
        except ValueError:
            sc_info = None

        if sc_info is None:
            self.skipTest("SparseCore is not available")

        out_size = 16384
        hidden_size = 128
        reduce_group_size = 8

        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        nan_rows = [5, 12, 100, 500, 1000]
        x = x.at[nan_rows, :].set(jnp.nan)
        x = x.astype(jnp.bfloat16)

        indices = jax.random.permutation(key, out_size)

        topk_weights = jnp.ones(
            (out_size // reduce_group_size, reduce_group_size),
            dtype=jnp.bfloat16)

        for nan_row in nan_rows:
            gathered_indices = jnp.where(indices == nan_row)[0]
            for idx_val in gathered_indices:
                token_idx = idx_val // reduce_group_size
                topk_idx = idx_val % reduce_group_size
                topk_weights = topk_weights.at[token_idx, topk_idx].set(0.0)

        actual = dense_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size,
            topk_wgt_zero_nan=True,
        )

        self.assertFalse(jnp.isnan(actual).any())

        desired = reference_dense_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size,
            topk_wgt_zero_nan=True,
        )
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

        actual_with_nan = dense_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size,
            topk_wgt_zero_nan=False,
        )
        self.assertTrue(jnp.isnan(actual_with_nan).any())

    def test_debug_layout(self):
        out_size = 520
        hidden_size = 8192
        reduce_group_size = 8

        # x[r, c] = 1 if r % 8 == c else 0
        x = np.zeros((out_size, hidden_size), dtype=np.float32)
        for r in range(out_size):
            x[r, r % 8] = 1.0
        x = jnp.array(x)

        key = jax.random.key(0)
        indices = jax.random.permutation(key, out_size)
        # weights = arange(M)
        topk_weights = jnp.arange(out_size, dtype=jnp.float32).reshape(
            out_size // reduce_group_size, reduce_group_size)

        actual = dense_gather_reduce(x, indices, topk_weights,
                                     reduce_group_size)
        desired = reference_dense_gather_reduce(x, indices, topk_weights,
                                                reduce_group_size)
        try:
            np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            print("DEBUG ACTUAL (first 32 rows):\n", actual[:32, :])
            print("DEBUG DESIRED (first 32 rows):\n", desired[:32, :])
            raise e

    def test_sc_kernel_keeps_output_operand_aliasing(self):
        """Guards the jax 0.10.1 mpmd_map regression fix.

        The SparseCore ``tpu_custom_call`` must carry destination-passing
        ``output_to_operand_aliasing``; without it XLA drops the AllocateBuffer
        double-buffer and the SC kernels stop overlapping with TensorCore ops
        (~+15% device on the MoE serving path). This kernel must therefore be
        constructed via ``core_map_helper.kernel`` (not raw ``pl.kernel``, which
        lowers via ``mpmd_map`` and loses the alias). A revert to ``pl.kernel``
        regresses this test.
        """
        out_size, hidden_size, reduce_group_size = 16384, 512, 8
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size),
                              jnp.bfloat16)
        indices = jax.random.permutation(key, out_size).astype(jnp.int32)
        topk_weights = jax.random.normal(
            key, (out_size // reduce_group_size, reduce_group_size),
            jnp.bfloat16)

        compiled = jax.jit(
            functools.partial(dense_gather_reduce,
                              reduce_group_size=reduce_group_size)).lower(
                                  x, indices, topk_weights).compile()
        hlo = compiled.as_text()

        # The kernel must actually run on SparseCore (not the jax fallback)...
        self.assertIn("sc_dense_gather_reduce", hlo)
        self.assertIn('custom_call_target="tpu_custom_call"', hlo)
        # ...and the SC custom-call must keep destination-passing aliasing.
        kernel_lines = [
            line for line in hlo.splitlines()
            if "sc_dense_gather_reduce" in line
            and 'custom_call_target="tpu_custom_call"' in line
        ]
        self.assertTrue(kernel_lines, "SC custom-call not found in HLO")
        self.assertTrue(
            any("output_to_operand_aliasing" in line
                for line in kernel_lines),
            "SparseCore dense_gather_reduce lost output_to_operand_aliasing "
            "(mpmd_map regression) — it must be built via core_map_helper, "
            "not pl.kernel.")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

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
"""Tests for the fused-unpermute path of gmm_v2.

`gmm_v2(..., scatter_indices=idx)` writes grouped output row j to output row
idx[j] via per-row DMAs instead of contiguously. We check it is bit-exact to
scattering the unfused output, over the row-routing edge cases the fused path
introduces (group/sublane straddles, empty groups, skew, EP group_offset
windows, multiple n tiles), and that the flag misuses raise.
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

jax.config.parse_flags_with_absl()


def _inputs(m, k, n, group_sizes, num_src_groups=None, seed=0, bias=False,
            fp8=False, qblk=128):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(seed), 4)
    lhs = (jax.random.normal(k0, (m, k), jnp.float32) * 0.5).astype(
        jnp.bfloat16)
    num_groups = num_src_groups or len(group_sizes)
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.float32) * 0.05
    rhs_scale = None
    if fp8:
        rhs = rhs.astype(jnp.float8_e4m3fn)
        rhs_scale = jax.random.uniform(k3, (num_groups, k // qblk, 1, n),
                                       jnp.float32, 0.5, 2.0)
    else:
        rhs = rhs.astype(jnp.bfloat16)
    rhs_bias = (jax.random.normal(k3, (num_groups, 1, n), jnp.float32) *
                0.1 if bias else None)
    group_sizes = jnp.asarray(group_sizes, jnp.int32)
    scatter_idx = jax.random.permutation(k2, m).astype(jnp.int32)
    return lhs, rhs, rhs_scale, rhs_bias, group_sizes, scatter_idx


class GmmV2FuseUnpermuteTest(jtu.JaxTestCase):

    def _check(self, m, k, n, group_sizes, *, num_src_groups=None, goff=0,
               bias=False, fp8=False, fuse_act=None, vmem_limit=None,
               out_dtype=jnp.float32, seed=0):
        """Fused scatter output must be bit-exact to scattering the unfused
        output, over the rows this call computes (the group_offset window)."""
        lhs, rhs, rhs_scale, rhs_bias, gs, idx = _inputs(
            m, k, n, group_sizes, num_src_groups, seed=seed, bias=bias,
            fp8=fp8)
        goff_arr = jnp.array([goff], jnp.int32)
        kw = dict(fuse_act=fuse_act, maybe_quantize_lhs=True)
        if vmem_limit:
            kw["vmem_limit_bytes"] = vmem_limit

        unfused = jax.jit(lambda l, i: gmm_v2(
            l, rhs, gs, rhs_scale, rhs_bias, goff_arr,
            preferred_element_type=out_dtype, zero_initialize=True,
            **kw))(lhs, idx)
        fused = jax.jit(lambda l, i: gmm_v2(
            l, rhs, gs, rhs_scale, rhs_bias, goff_arr, scatter_indices=i,
            preferred_element_type=out_dtype, zero_initialize=False,
            **kw))(lhs, idx)

        # Rows this call computes: the group_offset window.
        num_local = rhs.shape[0]
        offs = np.cumsum(np.concatenate([[0], np.asarray(gs)]))
        w0, w1 = int(offs[goff]), int(offs[goff + num_local])
        idx_np = np.asarray(idx)
        rows = np.arange(w0, w1)
        self.assertGreater(len(rows), 0)
        got = np.asarray(fused)[idx_np[rows]]
        want = np.asarray(unfused)[rows]
        self.assertArraysEqual(got, want)

    @parameterized.named_parameters(
        ("balanced", [64] * 8),
        ("sublane_straddle", [3, 5, 9, 2, 129, 64, 200, 100]),
        ("empty_groups", [0, 128, 0, 200, 56, 0, 128, 0]),
        ("single_group", [512]),
        ("heavy_skew", [410, 14, 16, 8, 24, 16, 16, 8]),
    )
    def test_scatter_matches_unfused(self, group_sizes):
        self._check(sum(group_sizes), 256, 512, group_sizes)

    def test_m_not_tile_aligned(self):
        # m is sublane-aligned (the layer guarantees M % 16 == 0; gmm_v2
        # requires it on every path) but not tile_m-aligned.
        self._check(496, 256, 512, [100, 61, 39, 120, 80, 55, 41, 0])

    def test_ep_group_offset_window(self):
        # 32 global groups, 8 local (rhs), middle window; rows outside the
        # window are neither computed nor written.
        gs = [16] * 32
        self._check(512, 256, 512, gs, num_src_groups=8, goff=8)

    def test_multiple_n_tiles(self):
        # Force tile_n < n so each output row is scattered in column slices.
        self._check(512, 256, 1024, [64] * 8,
                    vmem_limit=2 * 1024 * 1024)

    def test_with_bias(self):
        self._check(512, 256, 512, [64] * 8, bias=True)

    def test_fp8_rhs_blockwise_scale(self):
        self._check(512, 256, 512, [64] * 8, fp8=True)

    def test_fuse_act_silu(self):
        self._check(512, 256, 512, [64] * 8, fuse_act="silu")

    def test_bf16_out(self):
        # bf16 rows are DMA-legal through the compact (1, 128)-tiled output.
        self._check(512, 256, 512, [64] * 8, out_dtype=jnp.bfloat16)

    def test_bf16_out_ep_window(self):
        self._check(512, 256, 512, [16] * 32, num_src_groups=8, goff=8,
                    out_dtype=jnp.bfloat16)

    def test_rejects_zero_initialize(self):
        lhs, rhs, _, _, gs, idx = _inputs(512, 256, 512, [64] * 8)
        with self.assertRaisesRegex(ValueError, "zero_initialize"):
            gmm_v2(lhs, rhs, gs, None, None, jnp.array([0], jnp.int32),
                   scatter_indices=idx,
                   preferred_element_type=jnp.float32,
                   zero_initialize=True)

    def test_rejects_gather_and_scatter(self):
        lhs, rhs, _, _, gs, idx = _inputs(512, 256, 512, [64] * 8)
        with self.assertRaises(AssertionError):
            gmm_v2(lhs.astype(jnp.float32), rhs, gs, None, None,
                   jnp.array([0], jnp.int32), gather_indices=idx,
                   scatter_indices=idx,
                   preferred_element_type=jnp.float32,
                   zero_initialize=False)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

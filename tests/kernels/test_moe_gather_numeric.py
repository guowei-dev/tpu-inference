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
"""Numeric equivalence battery for the MoE token-gather paths (moe-gather-perf solve).

Discriminates hypothesis H2 ("one-hot is *numerically* different, which is what
boosts e2e throughput") from H1/H3. Key facts established here:

  * PERMUTE one-hot (`one_hot(idx,B) @ x`) and `ragged_gather_v2` are BIT-EXACT to
    `x[idx]` -> a numeric difference cannot originate on the permute side.
  * COMBINE one-hot (`(one_hot*w*mask).sum(1) @ gmm2`) vs the SparseCore
    `ragged_gather_reduce` vs a plain fp32 reference: both match the fp32 reference
    within bf16 round-off with NO systematic (signed-mean) bias -> H2 refuted.
  * Edge cases that the one-hot path *could* get wrong: masked rows, out-of-range
    (padding) indices, duplicate indices, dtype.

Run:  proj moe-gather-perf && tpu-run -- python -m pytest \
        tests/kernels/test_moe_gather_numeric.py -s -v
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import (
    ragged_gather_reduce, _fallback_implementation)

jax.config.parse_flags_with_absl()
TOPK = 8


def _onehot_permute(x, idx):
    return jax.nn.one_hot(idx, x.shape[0], dtype=x.dtype) @ x


def _onehot_combine(gmm2, revert, w, mask, topk):
    oh = jax.nn.one_hot(revert.reshape(-1, topk), gmm2.shape[0], dtype=gmm2.dtype)
    comb = (oh * w.reshape(-1, topk)[..., None] *
            mask.reshape(-1, topk)[..., None]).sum(axis=1)
    return comb @ gmm2


def _ref_combine_f32(gmm2, revert, w, mask, topk):
    """Exact fp32 gather-weight-mask-reduce reference."""
    g = gmm2.astype(jnp.float32)[revert]
    g = g * w.astype(jnp.float32)[:, None] * mask.astype(jnp.float32)[:, None]
    return g.reshape(-1, topk, g.shape[-1]).sum(axis=1)


def _stats(impl, ref):
    """signed-mean bias, mean-abs err, max-abs err (all in fp32)."""
    d = impl.astype(jnp.float32) - ref.astype(jnp.float32)
    scale = float(jnp.mean(jnp.abs(ref))) + 1e-30
    return dict(bias=float(jnp.mean(d)), bias_rel=float(jnp.mean(d)) / scale,
                mae=float(jnp.mean(jnp.abs(d))), maxabs=float(jnp.max(jnp.abs(d))),
                maxrel=float(jnp.max(jnp.abs(d)) / scale))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEGatherNumericTest(jtu.JaxTestCase):

    # ---- PERMUTE bf16 (the PRODUCTION dtype): bit-exact for both paths ----
    @parameterized.product(B=[512, 2048], H=[4096, 7168])
    def test_permute_bit_exact_bf16(self, B, H):
        # The MoE permute uses dtype=activation=bf16 (fused_moe_gmm.py:747-750):
        # a one-hot row is a single 1.0, and 1.0*x_row sums nowhere on the native
        # bf16 MXU -> bit-exact. ragged_gather_v2 is a true gather -> bit-exact.
        M = B * TOPK
        k = jax.random.key(0)
        x = jax.random.normal(k, (B, H), jnp.float32).astype(jnp.bfloat16)
        idx = jax.random.permutation(
            k, jnp.arange(B, dtype=jnp.int32).repeat(TOPK))
        ref = x[idx]
        oh = jax.jit(_onehot_permute)(x, idx)
        rg = ragged_gather_v2(x, idx, jnp.array([0], jnp.int32),
                              jnp.array([M], jnp.int32))
        self.assertArraysEqual(oh, ref)   # exact, no tolerance
        self.assertArraysEqual(rg, ref)   # exact, no tolerance

    # ---- PERMUTE f32 (latent, NOT a production path): one-hot is NOT exact ----
    def test_permute_f32_onehot_inexact_but_ragged_exact(self):
        # The v7 MXU has no native f32; default-precision f32 matmul emulates it
        # with bf16 passes, so even `1.0 * x` rounds -> one-hot f32 permute is
        # ~1e-2 off (NOT bit-exact). A true gather (ragged_gather_v2 / x[idx]) is
        # exact for any dtype. precision=HIGHEST restores one-hot exactness,
        # proving the mechanism. This dtype is never used by the MoE permute.
        B, H = 512, 4096
        M = B * TOPK
        k = jax.random.key(0)
        x = jax.random.normal(k, (B, H), jnp.float32)
        idx = jax.random.permutation(
            k, jnp.arange(B, dtype=jnp.int32).repeat(TOPK))
        ref = x[idx]
        oh = jax.jit(_onehot_permute)(x, idx)
        rg = ragged_gather_v2(x, idx, jnp.array([0], jnp.int32),
                              jnp.array([M], jnp.int32))
        oh_hi = jax.jit(lambda a, i: jnp.matmul(
            jax.nn.one_hot(i, a.shape[0], dtype=a.dtype), a,
            precision=jax.lax.Precision.HIGHEST))(x, idx)
        self.assertArraysEqual(rg, ref)                       # gather exact
        self.assertFalse(bool(jnp.array_equal(oh, ref)))      # default NOT exact
        self.assertAllClose(oh, ref, atol=3e-2, rtol=3e-2)    # but close
        self.assertArraysEqual(oh_hi, ref)                    # HIGHEST -> exact
        print(f"\n[permute f32] one-hot default max|diff|="
              f"{float(jnp.max(jnp.abs(oh - ref))):.3e} (MXU bf16-pass emulation); "
              f"HIGHEST precision -> bit-exact. bf16 production path is exact.")

    # ---- COMBINE: all three match the fp32 reference; quantify the bias ----
    @parameterized.product(T=[256, 2048], H=[4096, 7168])
    def test_combine_matches_reference_and_bias(self, T, H):
        B = T * TOPK
        k = jax.random.key(1)
        gmm2 = (jax.random.normal(k, (B, H), jnp.float32) * 0.1).astype(jnp.bfloat16)
        revert = jax.random.permutation(k, jnp.arange(B, dtype=jnp.int32))
        w = (jax.random.uniform(k, (B,)) * 0.5).astype(jnp.bfloat16)
        mask = jnp.ones((B,), jnp.bool_)
        ref = _ref_combine_f32(gmm2, revert, w, mask, TOPK)

        oh = jax.jit(lambda *a: _onehot_combine(*a, TOPK))(gmm2, revert, w, mask)
        rg = ragged_gather_reduce(gmm2, revert, w, mask, TOPK)
        tc = _fallback_implementation(gmm2, revert, w, mask, TOPK)

        s_oh, s_rg, s_tc = _stats(oh, ref), _stats(rg, ref), _stats(tc, ref)
        print(f"\n[combine bias T={T} H={H}] (vs fp32 ref)")
        for nm, s in (("onehot", s_oh), ("ragged", s_rg), ("tcref ", s_tc)):
            print(f"   {nm}: signed_bias={s['bias']:+.3e} (rel {s['bias_rel']:+.2e}) "
                  f"mae={s['mae']:.3e} maxabs={s['maxabs']:.3e} maxrel={s['maxrel']:.2e}")
        # All three within bf16 round-off of the fp32 reference.
        for s in (s_oh, s_rg, s_tc):
            self.assertLess(s["maxrel"], 5e-2)
        # H2 discriminator: one-hot has NO systematic bias the others lack.
        # Its signed bias must be the same order as ragged's (round-off, ~0-mean).
        self.assertLess(abs(s_oh["bias_rel"]), 5e-3)
        self.assertLess(abs(s_oh["bias_rel"]),
                        abs(s_rg["bias_rel"]) + 5e-3)

    # ---- EDGE: masked rows must be exactly zero in all paths ----
    def test_combine_masked_rows_zero(self):
        T, H = 512, 4096
        B = T * TOPK
        k = jax.random.key(2)
        gmm2 = (jax.random.normal(k, (B, H), jnp.float32) * 0.1).astype(jnp.bfloat16)
        revert = jax.random.permutation(k, jnp.arange(B, dtype=jnp.int32))
        w = (jax.random.uniform(k, (B,)) * 0.5).astype(jnp.bfloat16)
        # Mask out every slot of token 0 and token 5 -> their output rows = 0.
        mask = jnp.ones((B,), jnp.bool_)
        m2d = mask.reshape(T, TOPK).at[0].set(False).at[5].set(False)
        mask = m2d.reshape(-1)
        oh = jax.jit(lambda *a: _onehot_combine(*a, TOPK))(gmm2, revert, w, mask)
        rg = ragged_gather_reduce(gmm2, revert, w, mask, TOPK)
        tc = _fallback_implementation(gmm2, revert, w, mask, TOPK)
        for nm, out in (("onehot", oh), ("ragged", rg), ("tcref", tc)):
            self.assertArraysEqual(out[0], jnp.zeros_like(out[0]),
                                   err_msg=f"{nm} token0 not zero")
            self.assertArraysEqual(out[5], jnp.zeros_like(out[5]),
                                   err_msg=f"{nm} token5 not zero")

    # ---- EDGE: out-of-range (padding) index -- document the disagreement ----
    def test_oob_index_behavior(self):
        """one_hot(OOB) -> all-zero row -> gathers ZEROS; XLA x[OOB] in jit CLAMPS
        to the edge row. They DISAGREE. This is the classic one-hot footgun -- but
        the MoE permute feeds in-range indices (arange.repeat(topk)), so it is
        latent, not hit. We assert the disagreement so the property is recorded."""
        S, H = 64, 128
        k = jax.random.key(3)
        x = jax.random.normal(k, (S, H), jnp.float32).astype(jnp.bfloat16)
        idx = jnp.array([0, 1, S + 5, S - 1] + [0] * 12, jnp.int32)  # one OOB=S+5
        oh = jax.jit(_onehot_permute)(x, idx)
        xla = jax.jit(lambda a, i: a[i])(x, idx)
        # one-hot zeros the OOB row; XLA clamps to a real (edge) row -> not equal.
        self.assertArraysEqual(oh[2], jnp.zeros_like(oh[2]))
        self.assertFalse(bool(jnp.array_equal(xla[2], jnp.zeros_like(xla[2]))))
        # In-range rows agree.
        self.assertArraysEqual(oh[0], xla[0])
        print("\n[oob] confirmed: one_hot(OOB)->zeros, x[OOB]->clamp; "
              "in-range rows agree (latent: MoE permute indices are in-range).")

    # ---- H2 under REALISTIC data: skewed/renormalized top-k weights + partial
    #      masks; compare one-hot DIRECTLY to the baselines element-wise. ----
    def test_combine_realistic_weights_no_harmful_divergence(self):
        T, H = 2048, 4096   # T=2048 -> ragged takes the SparseCore path (genuine)
        B = T * TOPK
        k = jax.random.key(7)
        gmm2 = (jax.random.normal(k, (B, H), jnp.float32) * 0.1).astype(jnp.bfloat16)
        revert = jax.random.permutation(k, jnp.arange(B, dtype=jnp.int32))
        # realistic: skewed top-k weights (softmax of random logits, sums to 1)
        w = jax.nn.softmax(jax.random.normal(jax.random.key(8), (T, TOPK)) * 3.0,
                           axis=-1).reshape(-1).astype(jnp.bfloat16)
        mask = (jax.random.uniform(jax.random.key(9), (B,)) > 0.3)   # ~30% invalid
        ref = _ref_combine_f32(gmm2, revert, w, mask, TOPK)
        oh = jax.jit(lambda *a: _onehot_combine(*a, TOPK))(gmm2, revert, w, mask)
        rg = ragged_gather_reduce(gmm2, revert, w, mask, TOPK)
        tc = _fallback_implementation(gmm2, revert, w, mask, TOPK)
        e_oh = float(jnp.max(jnp.abs(oh.astype(jnp.float32) - ref)))
        e_rg = float(jnp.max(jnp.abs(rg.astype(jnp.float32) - ref)))
        d_oh_tc = float(jnp.max(jnp.abs((oh - tc).astype(jnp.float32))))
        print(f"\n[realistic combine T={T}] oh-ref={e_oh:.3e} rg(SC)-ref={e_rg:.3e} "
              f"oh-vs-tcref(direct)={d_oh_tc:.3e}")
        # one-hot tracks the XLA gather-reduce closely (same math family)...
        self.assertLess(d_oh_tc, 5e-3)
        # ...and one-hot is NO LESS accurate than the SparseCore ragged it replaces
        # -> switching to one-hot is not a harmful numeric change (H2 refuted).
        self.assertLessEqual(e_oh, e_rg + 1e-4)

    # ---- EDGE: duplicate indices (a source row gathered by many slots) ----
    def test_duplicate_indices(self):
        S, H = 128, 512
        M = S * TOPK
        k = jax.random.key(4)
        x = jax.random.normal(k, (S, H), jnp.float32).astype(jnp.bfloat16)
        idx = jnp.zeros((M,), jnp.int32)  # every slot gathers row 0
        oh = jax.jit(_onehot_permute)(x, idx)
        ref = x[idx]
        self.assertArraysEqual(oh, ref)


if __name__ == "__main__":
    absltest.main()

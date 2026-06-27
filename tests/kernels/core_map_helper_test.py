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
"""Tests for ``core_map_helper.kernel`` -- the destination-passing wrapper that
keeps ``output_to_operand_aliasing`` on single-mesh SparseCore kernels.

Both lowerings (``core_map`` and ``mpmd``) must (1) produce numerically correct
results and (2) emit a ``tpu_custom_call`` carrying ``output_to_operand_aliasing``
(the destination-passing alias that XLA double-buffers). A raw ``pl.kernel`` (which
lowers via ``mpmd_map`` since jax 0.10.1) loses that alias -- the regression this
helper exists to prevent; the test below also documents that contrast.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import core_map_helper

jax.config.parse_flags_with_absl()

KERNEL_NAME = "test_sc_gather"


def _gather_body(idx_hbm, in_hbm, out_hbm, idx_vmem, blk_vmem, sem, *,
                 core_axis, subcore_axis, rows_per_core, hidden):
    """Minimal per-(core, subcore) row gather: out[r] = in[idx[r]]."""
    info = pltpu.get_tpu_info()
    lanes = info.num_lanes
    simd = info.sparse_core.num_lanes
    base = lax.axis_index((core_axis, subcore_axis)) * rows_per_core
    recv, send = sem.at[0], sem.at[1]

    @pl.loop(0, rows_per_core, step=simd)
    def _(r0):
        row0 = base + r0
        pltpu.sync_copy(idx_hbm.at[pl.ds(row0, simd)], idx_vmem)
        indices = idx_vmem[...]
        for j in range(simd):
            for c in range(0, hidden, lanes):
                pltpu.make_async_copy(in_hbm.at[indices[j], pl.ds(c, lanes)],
                                      blk_vmem.at[j, pl.ds(c, lanes)],
                                      recv).start()
        for _j in range(simd):
            for c in range(0, hidden, lanes):
                pltpu.make_async_copy(in_hbm.at[0, pl.ds(0, lanes)],
                                      blk_vmem.at[0, pl.ds(0, lanes)],
                                      recv).wait()
        d = pltpu.make_async_copy(
            blk_vmem, out_hbm.at[pl.ds(row0, simd), pl.ds(0, hidden)], send)
        d.start()
        d.wait()


def _make_gather(rows, hidden, lowering):
    sc = pltpu.get_tpu_info().sparse_core
    ncores = sc.num_cores * sc.num_subcores
    rpc = rows // ncores
    simd = sc.num_lanes
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    body = functools.partial(_gather_body, core_axis=mesh.core_axis_name,
                             subcore_axis=mesh.subcore_axis_name,
                             rows_per_core=rpc, hidden=hidden)
    return core_map_helper.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        mesh=mesh,
        scratch_types=[
            pltpu.VMEM((simd, ), jnp.int32),
            pltpu.VMEM((simd, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True,
                                             disable_bounds_checks=True),
        name=KERNEL_NAME,
        lowering=lowering,
    )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class CoreMapHelperTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        try:
            sc_info = pltpu.get_tpu_info().sparse_core
        except ValueError:
            sc_info = None
        if sc_info is None:
            self.skipTest("SparseCore is not available")

    @parameterized.parameters("core_map", "mpmd", "mpmd_out", "mpmd_fix")
    def test_correct_and_aliased(self, lowering):
        rows, hidden = 512, 512
        gather = _make_gather(rows, hidden, lowering)
        key = jax.random.key(0)
        x = jax.random.normal(jax.random.fold_in(key, 1), (rows, hidden),
                              jnp.float32)
        idx = jax.random.randint(jax.random.fold_in(key, 2), (rows, ), 0, rows,
                                 jnp.int32)

        # (1) numerics: out[r] == x[idx[r]]
        out = jax.jit(gather)(idx, x)
        np.testing.assert_allclose(out, x[idx], atol=0, rtol=0)

        # (2) the SC custom-call must carry destination-passing aliasing.
        hlo = jax.jit(gather).lower(idx, x).compile().as_text()
        self.assertIn('custom_call_target="tpu_custom_call"', hlo)
        kernel_lines = [
            line for line in hlo.splitlines()
            if KERNEL_NAME in line
            and 'custom_call_target="tpu_custom_call"' in line
        ]
        self.assertTrue(kernel_lines, f"{KERNEL_NAME} custom-call not in HLO")
        self.assertTrue(
            any("output_to_operand_aliasing" in line
                for line in kernel_lines),
            f"lowering={lowering!r} lost output_to_operand_aliasing")

    def test_invalid_lowering(self):
        with self.assertRaises(ValueError):
            _make_gather(512, 512, "bogus")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

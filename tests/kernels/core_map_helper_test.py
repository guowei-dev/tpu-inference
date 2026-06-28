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
"""Tests for ``core_map_helper.kernel``: destination-passing must be preserved and
read-only inputs must not be donated (no preservation copy)."""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import core_map_helper

KERNEL_NAME = "test_sc_gather"


def _gather_body(idx_hbm, in_hbm, out_hbm, idx_vmem, blk_vmem, sem, *, core_axis,
                 subcore_axis, rows_per_core, hidden):
    """out[r] = in[idx[r]] over the per-(core, subcore) row range."""
    lanes = pltpu.get_tpu_info().num_lanes
    base = lax.axis_index((core_axis, subcore_axis)) * rows_per_core
    recv, send = sem.at[0], sem.at[1]
    simd = blk_vmem.shape[0]

    @pl.loop(0, rows_per_core, step=simd)
    def _(r0):
        row0 = base + r0
        pltpu.sync_copy(idx_hbm.at[pl.ds(row0, simd)], idx_vmem)
        indices = idx_vmem[...]
        for j in range(simd):
            for c in range(0, hidden, lanes):
                pltpu.make_async_copy(in_hbm.at[indices[j], pl.ds(c, lanes)],
                                      blk_vmem.at[j, pl.ds(c, lanes)], recv).start()
        for j in range(simd):
            for c in range(0, hidden, lanes):
                pltpu.make_async_copy(in_hbm.at[indices[j], pl.ds(c, lanes)],
                                      blk_vmem.at[j, pl.ds(c, lanes)], recv).wait()
        d = pltpu.make_async_copy(
            blk_vmem, out_hbm.at[pl.ds(row0, simd), pl.ds(0, hidden)], send)
        d.start()
        d.wait()


def _make_gather(rows, hidden):
    sc = pltpu.get_tpu_info().sparse_core
    simd = sc.num_lanes
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core", subcore_axis_name="subcore")
    body = functools.partial(_gather_body, core_axis=mesh.core_axis_name,
                             subcore_axis=mesh.subcore_axis_name,
                             rows_per_core=rows // (sc.num_cores * sc.num_subcores),
                             hidden=hidden)
    return core_map_helper.kernel(
        body, out_type=jax.ShapeDtypeStruct((rows, hidden), jnp.float32), mesh=mesh,
        scratch_types=[pltpu.VMEM((simd, ), jnp.int32),
                       pltpu.VMEM((simd, hidden), jnp.float32),
                       pltpu.SemaphoreType.DMA((2, ))],
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True,
                                             disable_bounds_checks=True),
        name=KERNEL_NAME)


class CoreMapHelperTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        try:
            pltpu.get_tpu_info().sparse_core
        except ValueError:
            self.skipTest("SparseCore is not available")

    def test_correct_and_aliased(self):
        rows, hidden = 512, 512
        gather = _make_gather(rows, hidden)
        key = jax.random.key(0)
        x = jax.random.normal(jax.random.fold_in(key, 1), (rows, hidden), jnp.float32)
        idx = jax.random.randint(jax.random.fold_in(key, 2), (rows, ), 0, rows, jnp.int32)

        np.testing.assert_array_equal(jax.jit(gather)(idx, x), x[idx])

        hlo = jax.jit(gather).lower(idx, x).compile().as_text()
        cc = [l for l in hlo.splitlines()
              if KERNEL_NAME in l and 'custom_call_target="tpu_custom_call"' in l]
        self.assertTrue(cc and any("output_to_operand_aliasing" in l for l in cc))

    def test_no_input_preservation_copy(self):
        # With the input live past the kernel, aliasing a read-only input ref would
        # force a copy of it. Only the written output may be aliased -> no large copy.
        rows, hidden = 8192, 512
        gather = _make_gather(rows, hidden)

        def f(a, b, idx):
            x = a @ b
            return gather(idx, x), x

        specs = (jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
                 jax.ShapeDtypeStruct((hidden, hidden), jnp.float32),
                 jax.ShapeDtypeStruct((rows, ), jnp.int32))
        hlo = jax.jit(f).lower(*specs).compile().as_text()
        self.assertIn(KERNEL_NAME, hlo)  # the kernel compiled: the check below is not vacuous
        big = [l for l in hlo.splitlines()
               if re.search(r'=\s+f32\[%d,%d\][^ ]*\s+copy\(' % (rows, hidden), l)]
        self.assertEmpty(big, f"unexpected preservation copy of the kernel input:\n{big}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

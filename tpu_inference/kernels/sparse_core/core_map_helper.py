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
"""Run single-mesh SparseCore kernels with destination-passing output aliasing.

Drop-in replacement for ``pl.kernel``. The output ref is allocated outside the kernel
and closed over the body (rather than returned), giving ``mpmd_map`` a destination to
alias, so the output keeps ``output_to_operand_aliasing`` (double-buffered) instead of a
fresh allocation.
``mpmd_discharge_patch`` (imported here) restricts that aliasing to the written
output ref, so read-only inputs are not donated (and thus not copied before the kernel).
"""

from jax._src import api
from jax._src import core as jax_core
from jax._src import lax, tree_util
from jax._src.pallas import core as pl_core
from jax._src.pallas import mpmd as pl_mpmd

from tpu_inference.kernels.sparse_core import mpmd_discharge_patch  # noqa: F401


def _empty_out_ref(out_type):
    aval = pl_core._convert_out_shape_to_aval(out_type)
    memory_space = (None if isinstance(aval.memory_space, jax_core.MemorySpace)
                    else aval.memory_space)
    value = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    return jax_core.new_ref(value, memory_space=memory_space)


def kernel(body,
           *,
           out_type,
           mesh,
           scratch_types=(),
           compiler_params=None,
           interpret=False,
           cost_estimate=None,
           debug=False,
           name=None,
           metadata=None):
    """Wrap ``body`` as a jitted single-mesh SparseCore kernel runner."""
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type, ) if single_output else out_type

    @api.jit
    def run(*operands):
        arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
        out_refs = tree_util.tree_map(_empty_out_ref, out_types)

        # out_types=() so mpmd_map allocates nothing; the closed-over out_refs are
        # the destinations and pick up the input_output_alias on discharge.
        def _kernel(*scratch_refs, **scratch_kwrefs):
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        pl_mpmd.mpmd_map(
            [(mesh, _kernel)],
            out_types=(),
            scratch_types=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )()

        outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run

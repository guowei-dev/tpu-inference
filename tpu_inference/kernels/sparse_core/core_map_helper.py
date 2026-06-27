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

Since jax ``18597c6032`` (0.10.1), ``pl.kernel`` lowers through ``mpmd_map``
instead of ``core_map``. The default ``mpmd_map`` path allocates a *fresh* output
buffer per invocation, so the SparseCore ``tpu_custom_call`` loses
``output_to_operand_aliasing`` (destination-passing). Without that alias XLA stops
emitting the ``AllocateBuffer`` double-buffer for the kernel output, and the
latency-hiding scheduler can no longer pipeline async SC-offload gathers / weight
prefetch through the kernel-output shadow -> the SC kernels stop overlapping with
TensorCore ops (a measured ~+15% device / ~8% E2E regression on the MoE serving
path).

``kernel`` here is a drop-in replacement for ``pl.kernel`` that restores the
output alias by allocating the output ref *outside* the kernel (``jax_core.new_ref``
over ``lax.empty``) and closing over it in the kernel body -- the destination-passing
pattern. Two equivalent lowerings produce the identical aliased custom-call:

* ``lowering="mpmd"`` (default): stay on ``mpmd_map`` (jax's current ``pl.kernel``
  direction) but pass ``out_types=()`` and let the closed-over output ref carry the
  destination-passing alias. This recovers the same ``output_to_operand_aliasing`` +
  ``AllocateBuffer`` **without depending on ``core_map`` remaining available** -- the
  future-proof path now that jax routes ``pl.kernel`` through ``mpmd_map``.
* ``lowering="core_map"``: construct the kernel via ``pl_core.core_map`` directly,
  matching the pre-0.10.1 ``pl.kernel`` lowering. Kept as a fallback.

Both paths emit a byte-identical aliased custom-call (verified at the
after-optimizations HLO level): identical SC ``tpu_custom_call`` with the
destination operand + ``AllocateBuffer`` double-buffer. ``mpmd`` is the default so the
fix does not depend on ``core_map`` surviving the ``pl.kernel`` -> ``mpmd_map`` move.
"""

from jax._src import api
from jax._src import core as jax_core
from jax._src import lax, tree_util
from jax._src.pallas import core as pl_core
from jax._src.pallas import mpmd as pl_mpmd


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
           metadata=None,
           lowering="mpmd"):
    """Drop-in replacement for ``pl.kernel`` with destination-passing output.

    Args:
      lowering: ``"mpmd"`` (default) stays on ``mpmd_map`` (jax's current
        ``pl.kernel`` direction) and recovers the output aliasing via the
        closed-over output ref; ``"core_map"`` constructs the kernel via
        ``pl_core.core_map`` (the pre-0.10.1 lowering, kept as a fallback).
        Both yield an identical aliased custom-call.
    """
    if lowering not in ("core_map", "mpmd"):
        raise ValueError(
            f"lowering must be 'core_map' or 'mpmd', got {lowering!r}")
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type, ) if single_output else out_type

    @api.jit
    def run(*operands):
        # Allocate input + output refs OUTSIDE the kernel and close over them.
        # The output ref (lax.empty destination) is what carries the
        # output_to_operand_aliasing on the lowered custom-call.
        arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
        out_refs = tree_util.tree_map(_empty_out_ref, out_types)

        if lowering == "core_map":
            @pl_core.core_map(
                mesh,
                scratch_shapes=scratch_types,
                compiler_params=compiler_params,
                interpret=interpret,
                cost_estimate=cost_estimate,
                debug=debug,
                name=name,
                metadata=metadata,
            )
            def _(*scratch_refs, **scratch_kwrefs):
                return body(*arg_refs, *out_refs, *scratch_refs,
                            **scratch_kwrefs)
        else:  # lowering == "mpmd"
            # out_types=() so mpmd_map allocates nothing; the closed-over
            # out_refs (written then read after) become the destination buffers
            # and pick up the input_output_alias on discharge.
            def body_closed(*scratch_refs, **scratch_kwrefs):
                return body(*arg_refs, *out_refs, *scratch_refs,
                            **scratch_kwrefs)

            pl_mpmd.mpmd_map(
                [(mesh, body_closed)],
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

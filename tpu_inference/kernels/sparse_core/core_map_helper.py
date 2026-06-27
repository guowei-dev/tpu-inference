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
pattern. Two lowerings carry that output alias:

* ``lowering="core_map"`` (default): construct the kernel via ``pl_core.core_map``
  directly, matching the pre-0.10.1 ``pl.kernel`` lowering. ``core_map`` aliases only
  the *written* output ref, so it gets the double-buffer with no extra ordering edges.
* ``lowering="mpmd"``: stay on ``mpmd_map`` (jax's current ``pl.kernel`` direction)
  with ``out_types=()`` and let the closed-over output ref carry the alias.

Both emit the same aliased SC ``tpu_custom_call`` (destination operand +
``AllocateBuffer`` double-buffer) at the *kernel* level, BUT they are **not**
equivalent in the full serving graph. ``mpmd_map`` auto-aliases **all** closed-over
input refs (not just the written output), which adds ref-ordering edges that
serialize the serving MoE prefill. Measured on Qwen3.5-397B (TP=8 EP serve, jax
0.10.2): vs ``core_map``, ``mpmd`` is ~flat on decode (1K/8K) but **~-20% total
throughput / +38% TTFT on prefill (8K/1K)** -- the handoff "read-ref auto-aliasing"
axis, distinct from (and opposite-signed to) the output double-buffer it recovers.
**So ``core_map`` is the default** (output double-buffer WITHOUT the read-ref
serialization). ``mpmd`` is kept for the upstream discussion (it proves the alias is
recoverable under ``mpmd_map``); the real jax fix is for ``mpmd_map`` to alias only
written refs, like ``core_map``.
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
           lowering="mpmd_fix"):
    """Drop-in replacement for ``pl.kernel`` with destination-passing output.

    Args:
      lowering: ``"core_map"`` (default) constructs the kernel via
        ``pl_core.core_map`` -- aliases only the written output ref, the best
        serving choice. ``"mpmd"`` stays on ``mpmd_map`` and recovers the output
        alias too, but its read-ref auto-aliasing regresses serving prefill
        (~-20% on 397B 8K/1K); see the module docstring. Same kernel-level
        custom-call either way.
    """
    if lowering not in ("core_map", "mpmd", "mpmd_out", "mpmd_fix"):
        raise ValueError(
            "lowering must be 'core_map', 'mpmd', 'mpmd_out', or 'mpmd_fix', "
            f"got {lowering!r}")
    if lowering == "mpmd_fix":
        # Process-wide: re-register mpmd_map's state-discharge to alias only
        # WRITTEN refs (like core_map), so closed-over read-only input refs are
        # not donated -> no preservation copies of the live MoE tensor. The
        # thorough fix for the mpmd serving-prefill regression; see the module
        # docstring + dev_nexus solve/mpmd-noregress.
        from . import mpmd_discharge_patch
        mpmd_discharge_patch.apply()
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
        elif lowering in ("mpmd", "mpmd_fix"):
            # out_types=() so mpmd_map allocates nothing; the closed-over
            # arg_refs + out_refs (written then read after) become the
            # destination buffers and pick up the input_output_alias on
            # discharge. NOTE: stock mpmd_map discharge aliases + force-keeps ALL
            # closed-over ref invars (the inputs too) -> XLA inserts a
            # preservation copy of each live input (the gmm output) -> +118 GB/
            # step on 397B 8K prefill -> ~-20%. "mpmd" leaves that bug in place;
            # "mpmd_fix" applies mpmd_discharge_patch (alias written-only) above
            # -> no copies, while keeping this serve-compatible closed-over form
            # (unlike "mpmd_out", which passes operands and breaks the vLLM
            # shared-experts trace).
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
        else:  # lowering == "mpmd_out"
            # PASS the operands to mpmd_map (they become *value* invars, not
            # aliased) and close over ONLY the output ref. Then discharge's
            # io_indices = just the output ref -> output double-buffer WITHOUT
            # mpmd_map auto-aliasing/force-keeping the input refs. Intended to
            # match core_map's serving behavior on the mpmd_map lowering.
            # (Operands must be non-None: mpmd_map flattens away None, so the
            # body's positional inputs would shift — the SC kernels pass non-None.)
            n_in = len(tree_util.tree_leaves(operands))

            def body_out(*in_and_scratch, **scratch_kwrefs):
                in_refs = in_and_scratch[:n_in]
                scratch_refs = in_and_scratch[n_in:]
                return body(*in_refs, *out_refs, *scratch_refs,
                            **scratch_kwrefs)

            pl_mpmd.mpmd_map(
                [(mesh, body_out)],
                out_types=(),
                scratch_types=scratch_types,
                compiler_params=compiler_params,
                interpret=interpret,
                cost_estimate=cost_estimate,
                debug=debug,
                name=name,
                metadata=metadata,
            )(*operands)

        outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run

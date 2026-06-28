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
"""Alias only modified refs when discharging ``mpmd_map``.

The stock ``mpmd_map`` discharge aliases every ref operand (``input_output_aliases``
for all ``AbstractRef`` invars), so a read-only input ref is donated to a discarded
output. Since that input is still live, XLA must copy it before the kernel. ``core_map``
aliases only the refs the kernel modifies and pays no such copy.

This re-registers the discharge to alias only modified (written or accumulated) refs, so a
closed-over output ref still gets destination-passing while read-only inputs are left
alone. The rule mirrors jax's ``_mpmd_map_discharge_rule`` and differs only in the
``io_indices`` filter (marked below; ``diff -w`` against the stock to re-sync on a jax
bump). Importing this module installs it. Drop once jax aliases modified refs natively.
"""

from collections.abc import Sequence
from typing import Any

from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import state
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.mpmd import mpmd_map_p, mpmd_map_tracing_context
from jax._src.state import discharge as state_discharge


def _modified_ref_discharge(
    avals_in: Sequence[jax_core.AbstractValue],
    avals_out: Sequence[jax_core.AbstractValue],
    *args: Any,
    jaxprs,
    meshes,
    input_output_aliases,
    debug,
    interpret,
    compiler_params,
    cost_estimate,
    metadata,
    name,
    external_meshes,
    **_,
):
    # CHANGED vs jax's _mpmd_map_discharge_rule: alias only refs the kernel MODIFIES
    # (write/accumulate), not every AbstractRef -- donating a live read-only input forces
    # a preservation copy. AccumEffect is an intentional superset of core_map's
    # WriteEffect-only rule (an accumulate-only ref must still alias + flow its value back).
    modified = {
        jaxpr.invars.index(eff.input)
        for jaxpr in jaxprs
        for eff in jaxpr.effects
        if isinstance(eff, (state.WriteEffect, state.AccumEffect))
    }
    io_indices = [
        i
        for i, aval in enumerate(avals_in)
        if isinstance(aval, state.AbstractRef) and i in modified
    ]
    # --- below mirrors the stock rule verbatim ---
    num_in = len(avals_in)
    num_out_orig = len(avals_out)
    num_out_new = len(io_indices)

    new_jaxprs = []
    all_meshes = (*meshes, *external_meshes)

    def _rewrite_to_include_new_outputs(jaxpr):

        def new_body(*args):
            in_refs, orig_out_refs, new_out_refs, scratch_refs = util.split_list(
                args, [num_in, num_out_orig, num_out_new]
            )
            del new_out_refs
            jax_core.eval_jaxpr(
                jaxpr, (), *(in_refs + orig_out_refs + scratch_refs)
            )
            return ()

        all_in_avals = [v.aval for v in jaxpr.invars]
        in_avals_trace, orig_out_avals_trace, scratch_avals_trace = util.split_list(
            all_in_avals, [num_in, num_out_orig]
        )
        new_out_avals_trace = [avals_in[i] for i in io_indices]
        tracing_avals = (
            in_avals_trace
            + orig_out_avals_trace
            + new_out_avals_trace
            + scratch_avals_trace
        )

        debug_info = api_util.debug_info(
            "mpmd_map_discharge", new_body, tracing_avals, {}
        )
        wrapped_fun = lu.wrap_init(new_body, debug_info=debug_info)
        new_jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(wrapped_fun, tracing_avals)
        return new_jaxpr

    for mesh, jaxpr in zip(meshes, jaxprs):
        with mpmd_map_tracing_context(mesh, all_meshes):
            new_jaxprs.append(_rewrite_to_include_new_outputs(jaxpr))

    assert all(
        isinstance(avals_in[i], state.AbstractRef) for i in io_indices
    )
    new_out_avals = [avals_in[i].inner_aval for i in io_indices]  # pyrefly: ignore[missing-attribute]
    updated_out_avals = list(avals_out) + new_out_avals

    new_aliases = dict(input_output_aliases)
    for out_idx, in_idx in enumerate(io_indices):
        new_aliases[in_idx] = num_out_orig + out_idx

    res = mpmd_map_p.bind(
        *args,
        jaxprs=tuple(new_jaxprs),
        meshes=meshes,
        input_output_aliases=FrozenDict(new_aliases),
        out_avals=tuple(updated_out_avals),
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        metadata=metadata,
        name=name,
        external_meshes=external_meshes,
    )

    # Split the results into original outputs and updated refs.
    ans, updated_refs = util.split_list(res, [num_out_orig])
    new_invals = [None] * len(avals_in)
    for out_idx, in_idx in enumerate(io_indices):
        new_invals[in_idx] = updated_refs[out_idx]

    return new_invals, ans


state_discharge.register_discharge_rule(mpmd_map_p)(_modified_ref_discharge)

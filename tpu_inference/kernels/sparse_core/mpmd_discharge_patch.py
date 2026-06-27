"""Local patch: make mpmd_map's state-discharge alias only WRITTEN refs (like core_map's
default_mesh_discharge_rule), instead of every ref invar.

Root cause (397B serving-HLO diff): the stock _mpmd_map_discharge_rule aliases EVERY AbstractRef
invar (read + write). Aliasing a read-only input donates its buffer; since the SC reduce kernel's
input (the live gmm_v2 output) is needed downstream, XLA inserts a preservation copy -> +180 x
bf16[81920,4096] = +118 GB/step -> ~20% prefill regression. core_map aliases only WriteEffect refs,
so no copy. This patch restricts io_indices to written refs -> the upstream fix applied locally,
keeping the closed-over structure (serve-compatible; unlike passing operands which breaks the
vLLM shared-experts trace). Identical to the stock rule except the io_indices line.
"""
import jax._src.pallas.mpmd as m
state_discharge = m.state_discharge


def _written_only_discharge(avals_in, avals_out, *args, jaxprs, meshes,
                            input_output_aliases, debug, interpret, compiler_params,
                            cost_estimate, metadata, name, external_meshes, **_):
    # --- the only behavioral change: written refs only (vs all AbstractRef) ---
    written = set()
    for jaxpr in jaxprs:
        for eff in jaxpr.effects:
            if isinstance(eff, m.state.WriteEffect):
                try:
                    written.add(jaxpr.invars.index(eff.input))
                except ValueError:
                    pass
    io_indices = [i for i, aval in enumerate(avals_in)
                  if isinstance(aval, m.state.AbstractRef) and i in written]
    # --- rest is a faithful copy of _mpmd_map_discharge_rule ---
    num_in = len(avals_in)
    num_out_orig = len(avals_out)
    num_out_new = len(io_indices)
    new_jaxprs = []
    all_meshes = (*meshes, *external_meshes)

    def _rewrite_to_include_new_outputs(jaxpr):
        def new_body(*args):
            in_refs, orig_out_refs, new_out_refs, scratch_refs = m.util.split_list(
                args, [num_in, num_out_orig, num_out_new])
            del new_out_refs
            m.jax_core.eval_jaxpr(jaxpr, (), *(in_refs + orig_out_refs + scratch_refs))
            return ()
        all_in_avals = [v.aval for v in jaxpr.invars]
        in_avals_trace, orig_out_avals_trace, scratch_avals_trace = m.util.split_list(
            all_in_avals, [num_in, num_out_orig])
        new_out_avals_trace = [avals_in[i] for i in io_indices]
        tracing_avals = (in_avals_trace + orig_out_avals_trace
                         + new_out_avals_trace + scratch_avals_trace)
        debug_info = m.api_util.debug_info("mpmd_map_discharge", new_body, tracing_avals, {})
        wrapped_fun = m.lu.wrap_init(new_body, debug_info=debug_info)
        new_jaxpr, _, _ = m.pe.trace_to_jaxpr_dynamic(wrapped_fun, tracing_avals)
        return new_jaxpr

    for mesh, jaxpr in zip(meshes, jaxprs):
        with m.mpmd_map_tracing_context(mesh, all_meshes):
            new_jaxprs.append(_rewrite_to_include_new_outputs(jaxpr))

    assert all(isinstance(avals_in[i], m.state.AbstractRef) for i in io_indices)
    new_out_avals = [avals_in[i].inner_aval for i in io_indices]
    updated_out_avals = list(avals_out) + new_out_avals

    new_aliases = dict(input_output_aliases)
    for out_idx, in_idx in enumerate(io_indices):
        new_aliases[in_idx] = num_out_orig + out_idx

    res = m.mpmd_map_p.bind(
        *args, jaxprs=tuple(new_jaxprs), meshes=meshes,
        input_output_aliases=m.FrozenDict(new_aliases),
        out_avals=tuple(updated_out_avals), debug=debug, interpret=interpret,
        compiler_params=compiler_params, cost_estimate=cost_estimate,
        metadata=metadata, name=name, external_meshes=external_meshes)

    ans, updated_refs = m.util.split_list(res, [num_out_orig])
    new_invals = [None] * len(avals_in)
    for out_idx, in_idx in enumerate(io_indices):
        new_invals[in_idx] = updated_refs[out_idx]
    return new_invals, ans


_applied = False


def apply():
    global _applied
    if not _applied:
        state_discharge.register_discharge_rule(m.mpmd_map_p)(_written_only_discharge)
        _applied = True

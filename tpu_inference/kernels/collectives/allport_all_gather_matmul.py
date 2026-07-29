# SPDX-License-Identifier: Apache-2.0
"""All-port fused all-gather matmul (hierarchical schedule, ring-style
pipelining).

The dissemination mirror of allport_matmul_reduce_scatter: row-band b of each
core's x chunk trades dim (b+s) % D at round s (every ICI axis busy every
round — a row band is a complete [rows, K] sub-chunk, so each arrival is
dottable the moment it lands, with no reassembly), the whole held set
forwarded every round, and each completed chunk copied to the twin over the
D2D link as it completes. Bands >= 128 rows are dotted per band; thinner
bands are folded into one full-chunk dot at the last band's arrival (the MXU
runs 2.5x/10x below ideal at 64/32 rows — measured, probe_allport_ag_ablate). Unlike the RS ladder there is NO unlock gap: the round-0 sends
carry the core's own chunk and go out before any compute, so the wire
pipelines from t=0 exactly like the ring's. The emission order comes from
topology.allport_ag_program; arrival dots run one step behind their loads (a
depth-1 pipeline) so each HBM->VMEM load hides under the next wire wait.

Deadlock-freedom (the rule future edits must keep): every device emits the
SAME static sequence of DMA starts and semaphore waits — the ladder is
enumerated in RELATIVE labels l = q XOR my_chip, so slot indices and program
positions are device-invariant and only traced values (absolute labels,
partner ids, row offsets) differ. A wait on a shared cell then only needs the
partner to have issued its symmetric start, which it did at the same earlier
program position (induction over wait index; the CPU simulator in
allport_ag_schedule_test.py re-checks every edit). Never wrap a blocking or
signaling op in a device-dependent branch, and never make a wait's slot index
data-dependent.

Layout contract: logical id == chip * 2 + core (topology.make_collective_mesh
('hier')); the chunk owned by device d is rows [d*m_per, (d+1)*m_per) of the
gathered x; the gather buffer is addressed in GLOBAL chunk space (the id is
label-invariant across devices, so src slot == dst slot on both ends of every
copy) and every slot/semaphore cell is written exactly once per call. y is
resident in VMEM, streamed in n-column tiles under the first dot (each tile
waited once, at first use); dots tile the OUTPUT columns, so there is no
cross-tile accumulation. VMEM residency bounds the envelope to M <= ~2048 at
perdev spec shapes (larger M FAILs at compile — the ring kernel owns that
range). collective_id = 5.
"""
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import topology, util
from tpu_inference.kernels.collectives.hier_all_gather_matmul import \
    validate_inputs as _hier_validate  # same base input contract

_COLLECTIVE_ID = 5


def _row_bands(m_per, nbands):
    """Sublane-aligned row bands of one chunk; the last band absorbs the
    remainder (floor split keeps every band non-empty given
    m_per >= 8 * nbands)."""
    fcs = (m_per // nbands) // 8 * 8
    return [(b * fcs, fcs if b + 1 < nbands else m_per - b * fcs)
            for b in range(nbands)]


def validate_inputs(x, y, tp_size):
    _hier_validate(x, y, tp_size)
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    nbands = max(num_dims, 1)
    m_per = x.shape[0]
    if m_per < 8 * nbands:
        raise ValueError(f"m per device {m_per} too small for {nbands} "
                         f"8-aligned row bands")


def _allport_ag_kernel(
    x_hbm,
    y_hbm,
    out_hbm,
    gather_hbm,
    y_vmem,
    x_vmem,
    o_vmem,
    sem_seed,
    sem_x,
    sem_o,
    sems_y,
    sems_ici,
    sems_twin,
    *,
    sched,
    prog,
    num_dims,
    m_per,
    bn,
    row_bands,
    axis_name,
    ablate,
):
    n_per = y_vmem.shape[1]
    num_nt = n_per // bn
    my_id = lax.axis_index(axis_name)
    bit = lax.rem(my_id, 2)
    my_chip = lax.div(my_id, 2)
    twin = my_id + 1 - 2 * bit

    partners = [twin]
    for d in range(num_dims):
        partners.append((my_chip ^ (1 << d)) * 2 + bit)
    util.local_barrier_logical(partners)

    # y column-tile loads start immediately; each tile is waited once, at
    # first use (inside the first executed dot) — the 56 MiB y stream hides
    # under the early dots instead of standing exposed at the head.
    for nt in range(num_nt):
        pltpu.make_async_copy(y_hbm.at[:, pl.ds(nt * bn, bn)],
                              y_vmem.at[:, pl.ds(nt * bn, bn)],
                              sems_y.at[nt]).start()
    y_waited = [False] * num_nt

    def chunk_of(l, parity):
        """Global chunk id of relative label l, mine (0) / the twin's (1)."""
        b_ = bit if parity == 0 else 1 - bit
        return (my_chip ^ l) * 2 + b_

    def ici_op(key):
        s, b, l = key
        dim = sched["sends"][key]
        partner = (my_chip ^ (1 << dim)) * 2 + bit
        chunk = chunk_of(l, 0)
        r0, rw = row_bands[b]
        slot = sched["slot"][key]
        return pltpu.make_async_remote_copy(
            src_ref=gather_hbm.at[chunk, pl.ds(r0, rw), slice(None)],
            dst_ref=gather_hbm.at[chunk, pl.ds(r0, rw), slice(None)],
            send_sem=sems_ici.at[slot],
            recv_sem=sems_ici.at[slot],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def twin_op(l):
        chunk = chunk_of(l, 0)
        return pltpu.make_async_remote_copy(
            src_ref=gather_hbm.at[chunk],
            dst_ref=gather_hbm.at[chunk],
            send_sem=sems_twin.at[l],
            recv_sem=sems_twin.at[l],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    # Depth-1 dot pipeline: a dot's x load starts at its program point (its
    # data is guaranteed by the preceding wait); the MXU + out store run when
    # the NEXT dot is pushed, so the load hides under the next wire wait
    # instead of a serial start;wait per dot. Loads/dots are purely local —
    # deferring them does not touch the static send/wait sequence.
    pending = []       # at most one (spec, buf)
    o_pending = [None, None]
    dot_i = [0]

    def x_load(spec, buf):
        parity, l, r0, rw = spec
        return pltpu.make_async_copy(
            gather_hbm.at[chunk_of(l, parity), pl.ds(r0, rw), slice(None)],
            x_vmem.at[buf, pl.ds(0, rw), slice(None)], sem_x.at[buf])

    def flush_dot():
        spec, buf = pending.pop()
        parity, l, r0, rw = spec
        with jax.named_scope(f"dot_p{parity}l{l}r{r0}"):
            x_load(spec, buf).wait()
            if o_pending[buf] is not None:
                o_pending[buf].wait()

            def nt_body(nt):
                block = jnp.dot(x_vmem[buf, pl.ds(0, rw), :],
                                y_vmem[:, pl.ds(nt, bn)],
                                preferred_element_type=jnp.float32)
                o_vmem[buf, pl.ds(0, rw), pl.ds(nt, bn)] = block.astype(
                    o_vmem.dtype)

            if not all(y_waited):
                for nt in range(num_nt):
                    if not y_waited[nt]:
                        pltpu.make_async_copy(
                            y_hbm.at[:, pl.ds(nt * bn, bn)],
                            y_vmem.at[:, pl.ds(nt * bn, bn)],
                            sems_y.at[nt]).wait()
                        y_waited[nt] = True
                    nt_body(nt * bn)
            else:

                @pl.loop(0, num_nt)
                def _tiles(t):
                    nt_body(t * bn)

            o_copy = pltpu.make_async_copy(
                o_vmem.at[buf, pl.ds(0, rw), slice(None)],
                out_hbm.at[pl.ds(chunk_of(l, parity) * m_per + r0, rw),
                           slice(None)], sem_o.at[buf])
            o_copy.start()
            o_pending[buf] = o_copy

    def push_dot(spec):
        buf = dot_i[0] % 2
        x_load(spec, buf).start()
        if pending:
            flush_dot()
        pending.append((spec, buf))
        dot_i[0] += 1

    # ablate (instrumentation only, timing-ladder attribution; results are
    # WRONG for ablate != 3): 1 = own dots only (MXU + y-stream floor),
    # 2 = + ICI dissemination and its dots, 4 = + twin wires but not their
    # dots, 3 = full. Ops are skipped identically on every device (no
    # deadlock).
    skip = {
        1: ("send", "wait", "twin_start", "twin_wait"),
        2: ("twin_start", "twin_wait"),
        4: (),
        3: (),
    }[ablate]

    def skip_dot(spec):
        parity, _, l = spec
        if ablate == 1:
            return not (parity == 0 and l == 0)
        if ablate in (2, 4):
            return parity == 1
        return False

    # Sub-128-row dots starve the MXU (measured 2.5x/10x off ideal at 64/32
    # rows) — when the row bands are thinner than 128, fold a chunk's band
    # dots into ONE full-chunk dot fired at the LAST band's program point
    # (its data is complete there; the fold is local, the static send/wait
    # sequence is untouched). Bands >= 128 rows keep per-band dots (smaller
    # VMEM buffers, earlier starts).
    chunk_dots = min(rw for _, rw in row_bands) < 128
    nbands = max(num_dims, 1)
    bands_seen = {}

    def dot_specs(pspec):
        parity, b, l = pspec
        if not chunk_dots:
            r0, rw = row_bands[b]
            return [(parity, l, r0, rw)]
        seen = bands_seen.setdefault((parity, l), set())
        seen.add(b)
        if len(seen) == nbands:
            return [(parity, l, 0, m_per)]
        return []

    for op in prog:
        kind = op[0]
        if kind in skip:
            continue
        if kind == "seed":
            with jax.named_scope("seed"):
                cp = pltpu.make_async_copy(x_hbm,
                                           gather_hbm.at[chunk_of(0, 0)],
                                           sem_seed)
                cp.start()
                cp.wait()
        elif kind == "send":
            s, b, l = op[1]
            with jax.named_scope(f"ag_s{s}b{b}l{l}_send"):
                ici_op(op[1]).start()
        elif kind == "wait":
            s, b, l = op[1]
            with jax.named_scope(f"ag_s{s}b{b}l{l}_wait"):
                ici_op(op[1]).wait()
        elif kind == "twin_start":
            twin_op(op[1]).start()
        elif kind == "twin_wait":
            with jax.named_scope(f"ag_tw{op[1]}_wait"):
                twin_op(op[1]).wait()
        elif kind == "dot":
            if not skip_dot(op[1]):
                for spec in dot_specs(op[1]):
                    push_dot(spec)
    if pending:
        flush_dot()
    for buf in (0, 1):
        if o_pending[buf] is not None:
            o_pending[buf].wait()


def get_vmem_estimate_bytes(m_per, k, n_per, rows_dot, bn, itemsize):
    return (k * n_per * itemsize            # y resident
            + 2 * rows_dot * k * itemsize   # x double buffer
            + 2 * rows_dot * n_per * itemsize  # o staging double buffer
            + rows_dot * bn * 4)            # dot value


def allport_all_gather_matmul(
    x,
    y,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
    ablate: int = 3,
):
    """all_gather(x, axis=0) @ y, all-port pipelined hierarchical kernel.

    x: [M, K] P(axis, None), y: [K, N] P(None, axis) -> out [M, N] P(None,
    axis). Mesh from topology.make_collective_mesh('hier').
    """
    tp_size = mesh.shape[axis_name]
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    if num_dims == 0:
        raise ValueError("allport kernel needs >= 2 chips")
    m, k = x.shape
    m_per = m // tp_size
    n_per = y.shape[1] // tp_size
    bn = next(c for c in (2048, 1024, 512, 256, 128) if n_per % c == 0)
    row_bands = _row_bands(m_per, num_dims)
    # dot buffers hold full chunks when the kernel folds thin band dots
    rows_dot = m_per if min(rw for _, rw in row_bands) < 128 else max(
        rw for _, rw in row_bands)
    sched, prog = topology.allport_ag_program(num_dims)

    def per_device(x_local, y_local):
        validate_inputs(x_local, y_local, tp_size)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM(y_local.shape, y_local.dtype),
                pltpu.VMEM((2, rows_dot, k), x_local.dtype),
                pltpu.VMEM((2, rows_dot, y_local.shape[1]), x_local.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((y_local.shape[1] // bn, )),
                pltpu.SemaphoreType.DMA((sched["n_slots"], )),
                pltpu.SemaphoreType.DMA((2**num_dims, )),
            ),
            grid=(1, ),
        )
        kernel = functools.partial(
            _allport_ag_kernel,
            sched=sched,
            prog=prog,
            num_dims=num_dims,
            m_per=m_per,
            bn=bn,
            row_bands=row_bands,
            axis_name=axis_name,
            ablate=ablate,
        )
        out, _ = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((m, y_local.shape[1]), x_local.dtype),
                jax.ShapeDtypeStruct((tp_size, m_per, k), x_local.dtype),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=get_vmem_estimate_bytes(
                    m_per, k, y_local.shape[1], rows_dot, bn,
                    x_local.dtype.itemsize) + 8 * 1024 * 1024,
            ),
            name=f"allport_all_gather_matmul_d{num_dims}",
        )(x_local, y_local)
        return out

    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec(axis_name, None),
                  jax.sharding.PartitionSpec(None, axis_name)),
        out_specs=jax.sharding.PartitionSpec(None, axis_name),
        check_vma=False,
    )(x, y)

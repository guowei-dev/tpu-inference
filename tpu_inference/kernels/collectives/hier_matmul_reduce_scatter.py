# SPDX-License-Identifier: Apache-2.0
"""Hierarchical (twin-pair + hypercube) matmul reduce-scatter kernel.

Computes reduce_scatter(a @ w, axis=0) in the hierarchical_reduce_scatter
style, fused with the matmul and with the running sums kept VMEM-resident
end-to-end (the unfused reference's per-round HBM round-trip accumulate is
what makes it lose to the ring even on pure collectives).

Structure, for tp devices = num_chips chips x 2 cores (logical id must be
chip * 2 + core — build the mesh with topology.make_collective_mesh('hier')):
  - The M rows split into tp chunks; device d ends owning chunk d. Chunk
    2q + bit belongs to core `bit` of chip q.
  - Phase 1 (intra-chip twin, D2D): for every chip q, compute the twin-parity
    chunk's partial (fp32 MXU) into the run slot and send it to the twin;
    then compute the own-parity chunk and add the received twin partial.
    After phase 1 each core holds pair-sums for all chips' own-parity chunks.
  - Phase 2 (inter-chip hypercube, "concurrent dims"): the K columns split
    into num_dims bands; in round s band b trades along dim (b + s) %
    num_dims, so every ICI axis is active in every round (2 axes on a 2x2x1
    slice, 3 on 2x2x2 — the dimension count derives from the chip grid, so
    the kernel adapts across those topologies with no code change). Chunk
    resolution halves per round; after num_dims rounds the surviving slot is
    the device's own chunk, copied to the output.

Wire dtype = input dtype (bf16 in production): one rounding per hop, and the
hop count is 1 + log2(num_chips) (3 at tp=8) vs the ring's tp-1 (7), so the
error profile is tighter than the ring kernel's.

Remote landings use per-(round, band, op) HBM slots and semaphore cells that
are used exactly once per call (never reused), and the kernel opens with a
double barrier over {twin} | {hypercube partners} on collective_id 2.
"""
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util

_COLLECTIVE_ID = 2


def _band_bounds(k, num_dims):
    """Static (start, width) per band; last band absorbs the remainder."""
    fcs = ((k // num_dims + 127) // 128) * 128
    bounds = []
    for b in range(num_dims):
        start = min(b * fcs, k)
        width = min(fcs, k - start)
        bounds.append((start, width))
    return bounds


def _p2_slots(num_dims):
    """Static phase-2 slot index per (round, band, op) + total count."""
    slots, total = {}, 0
    for s in range(num_dims):
        ops = 2**(num_dims - 1 - s)
        for b in range(num_dims):
            for op in range(ops):
                slots[(s, b, op)] = total
                total += 1
    return slots, max(total, 1)


def _kept_label(op_idx, s, num_dims, my_chip, dim, dim_bit):
    """Chip label traded at (round s, rotated dim, op_idx): resolved rounds
    come from my own chip id, unresolved rounds from the op counter, and the
    target dim from dim_bit (the reference's _get_hypercube_chunk_idx)."""
    base = jnp.int32(0)
    # dims already traded in earlier rounds of THIS band's rotation:
    # rotation for band b is dim_r = (b + r) % num_dims for round r, and
    # `dim` = (b + s) % num_dims, so prior dims are (dim - s + r) % num_dims.
    for r in range(s):
        d = (dim - s + r) % num_dims
        base = base | (((my_chip >> d) & 1) << d)
    future = [(dim - s + r) % num_dims for r in range(s + 1, num_dims)]
    for bit_pos, d in enumerate(future):
        base = base | (jnp.int32((op_idx >> bit_pos) & 1) << d)
    return base | (dim_bit << dim)


def _hier_mmrs_kernel(
    a_hbm,
    w_hbm,
    out_hbm,
    recv1_hbm,
    recv2_hbm,
    w_vmem,
    a_vmem,
    run_vmem,
    acc_vmem,
    recv_vmem,
    sem_w,
    sem_a,
    sem_recv,
    sem_out,
    sems1,
    sems2,
    sems_load,
    *,
    num_chips,
    num_dims,
    m_per,
    bk,
    band_bounds,
    p2_slots,
    axis_name,
):
    my_id = lax.axis_index(axis_name)
    bit = lax.rem(my_id, 2)
    my_chip = lax.div(my_id, 2)
    twin = my_id + 1 - 2 * bit

    partners = [twin]
    for d in range(num_dims):
        partners.append((my_chip ^ (1 << d)) * 2 + bit)
    util.local_barrier_logical(partners)

    w_copy = pltpu.make_async_copy(w_hbm, w_vmem, sem_w)
    w_copy.start()
    w_copy.wait()

    k = w_vmem.shape[1]
    num_kt = k // bk

    def load_a(chunk_idx):
        a_copy = pltpu.make_async_copy(
            a_hbm.at[pl.ds(chunk_idx * m_per, m_per), :], a_vmem, sem_a)
        a_copy.start()
        a_copy.wait()

    def p1_send(q):
        """Twin exchange for chip q's chunks; the same cell carries my send
        and the twin's symmetric incoming copy, so one .wait() consumes both
        (the ring kernels' reconstructed-op pattern)."""
        return pltpu.make_async_remote_copy(
            src_ref=run_vmem.at[q],
            dst_ref=recv1_hbm.at[q],
            send_sem=sems1.at[q],
            recv_sem=sems1.at[q],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    # --- Phase 1: twin-parity partials out over D2D, own-parity merged in.
    # Traced loops (not Python-unrolled): the k-tiled fp32 dot value and the
    # w tile it reads stay one-body-deep on the Mosaic stack — the unrolled
    # form accumulated them across iterations and OOM'd scoped VMEM at K=8192.
    @pl.loop(0, num_chips)
    def _p1_compute_send(q):
        load_a(2 * q + (1 - bit))

        @pl.loop(0, num_kt)
        def _tile(i):
            kt = i * bk
            acc = jnp.dot(a_vmem[...], w_vmem[:, pl.ds(kt, bk)],
                          preferred_element_type=jnp.float32)
            run_vmem[q, slice(None), pl.ds(kt, bk)] = acc.astype(
                run_vmem.dtype)

        p1_send(q).start()

    @pl.loop(0, num_chips)
    def _p1_merge(q):
        # The own-parity dot runs into acc_vmem BEFORE waiting on the twin
        # exchange, so the D2D hides under the MXU rather than the reverse.
        load_a(2 * q + bit)

        @pl.loop(0, num_kt)
        def _tile(i):
            kt = i * bk
            acc_vmem[slice(None), pl.ds(kt, bk)] = jnp.dot(
                a_vmem[...], w_vmem[:, pl.ds(kt, bk)],
                preferred_element_type=jnp.float32)

        # My send q done (slot q is about to be overwritten) AND the twin's
        # chunk q arrived.
        p1_send(q).wait()

        @pl.loop(0, num_kt)
        def _merge(i):
            kt = i * bk
            r_copy = pltpu.make_async_copy(
                recv1_hbm.at[q, slice(None), pl.ds(kt, bk)],
                recv_vmem.at[0, slice(None), pl.ds(0, bk)], sem_recv)
            r_copy.start()
            r_copy.wait()
            merged = (acc_vmem[:, pl.ds(kt, bk)] +
                      recv_vmem[0, :, pl.ds(0, bk)].astype(jnp.float32))
            run_vmem[q, slice(None), pl.ds(kt, bk)] = merged.astype(
                run_vmem.dtype)

    # --- Phase 2: per round, every band trades on its rotated dim at once.
    def p2_send(s, b, op):
        dim = (b + s) % num_dims
        partner = (my_chip ^ (1 << dim)) * 2 + bit
        start, width = band_bounds[b]
        partner_bit = 1 - ((my_chip >> dim) & 1)
        send_label = _kept_label(op, s, num_dims, my_chip, dim, partner_bit)
        slot = p2_slots[(s, b, op)]
        return pltpu.make_async_remote_copy(
            src_ref=run_vmem.at[send_label,
                                slice(None),
                                pl.ds(start, width)],
            dst_ref=recv2_hbm.at[slot, slice(None), pl.ds(0, width)],
            send_sem=sems2.at[slot],
            recv_sem=sems2.at[slot],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    for s in range(num_dims):
        ops = 2**(num_dims - 1 - s)
        issued = []
        for b in range(num_dims):
            if band_bounds[b][1] <= 0:
                continue
            for op in range(ops):
                p2_send(s, b, op).start()
                issued.append((b, op))

        def load_op(idx, b, op, s=s):
            width = band_bounds[b][1]
            slot = p2_slots[(s, b, op)]
            return pltpu.make_async_copy(
                recv2_hbm.at[slot, slice(None), pl.ds(0, width)],
                recv_vmem.at[idx, slice(None), pl.ds(0, width)],
                sems_load.at[idx])

        for idx, (b, op) in enumerate(issued):
            # My send for this slot done (its source slot stays dead) AND the
            # partner's contribution arrived; then start the HBM->VMEM load
            # so all of the round's loads overlap the merges below.
            p2_send(s, b, op).wait()
            load_op(idx, b, op).start()
        for idx, (b, op) in enumerate(issued):
            dim = (b + s) % num_dims
            start, width = band_bounds[b]
            my_bit = (my_chip >> dim) & 1
            keep_label = _kept_label(op, s, num_dims, my_chip, dim, my_bit)
            load_op(idx, b, op).wait()
            for st in range(0, width, bk):
                sw = min(bk, width - st)
                merged = (
                    run_vmem[keep_label, :,
                             pl.ds(start + st, sw)].astype(jnp.float32) +
                    recv_vmem[idx, :, pl.ds(st, sw)].astype(jnp.float32))
                run_vmem[keep_label, slice(None),
                         pl.ds(start + st, sw)] = merged.astype(run_vmem.dtype)

    out_copy = pltpu.make_async_copy(run_vmem.at[my_chip], out_hbm, sem_out)
    out_copy.start()
    out_copy.wait()


def get_vmem_estimate_bytes(m_per, n_per, k, bk, num_chips, n_load, rv_w,
                            itemsize):
    """Scratch VMEM: resident w + a tile + run slots + fp32 acc + the
    phase-2 load slots, plus the k-tiled fp32 dot value."""
    w_bytes = n_per * k * itemsize
    a_bytes = m_per * n_per * itemsize
    run_bytes = num_chips * m_per * k * itemsize
    acc_bytes = m_per * k * 4
    recv_bytes = n_load * m_per * rv_w * itemsize
    dot_bytes = m_per * bk * 4
    return w_bytes + a_bytes + run_bytes + acc_bytes + recv_bytes + dot_bytes


def validate_inputs(a, w, tp_size):
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("a and w must be 2-D")
    if a.dtype != w.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {w.dtype}")
    m, n_per = a.shape
    n_per_w, k = w.shape
    if n_per != n_per_w:
        raise ValueError(f"contraction mismatch: {n_per} vs {n_per_w}")
    if tp_size % 2:
        raise ValueError(f"tp_size {tp_size} must pair 2 cores per chip")
    num_chips = tp_size // 2
    if num_chips & (num_chips - 1):
        raise ValueError(f"num_chips {num_chips} must be a power of two")
    if m % tp_size:
        raise ValueError(f"m {m} must be divisible by tp_size {tp_size}")
    if (m // tp_size) % 8:
        raise ValueError(f"m per device {m // tp_size} must be sublane-"
                         f"aligned (divisible by 8)")
    if k % 128:
        raise ValueError(f"k {k} must be divisible by 128")


def hier_matmul_reduce_scatter(
    a,
    w,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
):
    """reduce_scatter(a @ w, axis=0) via twin-pair D2D + hypercube rounds.

    a: [M, n_per] P(None, axis), w: [n_per, K] P(axis, None) ->
    out: [M // tp, K] P(axis, None). The mesh device order must satisfy
    logical id == chip * 2 + core (topology.make_collective_mesh('hier')).
    """
    tp_size = mesh.shape[axis_name]
    validate_inputs(a, w, tp_size)
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    m = a.shape[0]
    k = w.shape[1]
    m_per = m // tp_size
    bk = next(c for c in (2048, 1024, 512, 256, 128) if k % c == 0)
    band_bounds = _band_bounds(k, num_dims) if num_dims else []
    p2_slots, n_slots = _p2_slots(num_dims) if num_dims else ({}, 1)
    band_w = max((w_ for _, w_ in band_bounds), default=k)
    n_load = num_dims * 2**(num_dims - 1) if num_dims else 1
    rv_w = max(band_w, bk)

    def per_device(a_local, w_local):
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM(w_local.shape, w_local.dtype),
                pltpu.VMEM((m_per, a_local.shape[1]), a_local.dtype),
                pltpu.VMEM((num_chips, m_per, k), a_local.dtype),
                pltpu.VMEM((m_per, k), jnp.float32),
                pltpu.VMEM((n_load, m_per, rv_w), a_local.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_chips, )),
                pltpu.SemaphoreType.DMA((n_slots, )),
                pltpu.SemaphoreType.DMA((n_load, )),
            ),
            grid=(1, ),
        )
        kernel = functools.partial(
            _hier_mmrs_kernel,
            num_chips=num_chips,
            num_dims=num_dims,
            m_per=m_per,
            bk=bk,
            band_bounds=band_bounds,
            p2_slots=p2_slots,
            axis_name=axis_name,
        )
        out, _, _ = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((num_chips, m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((n_slots, m_per, band_w), a_local.dtype),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=get_vmem_estimate_bytes(
                    m_per, a_local.shape[1], k, bk, num_chips, n_load, rv_w,
                    a_local.dtype.itemsize) + 8 * 1024 * 1024,
            ),
        )(a_local, w_local)
        return out

    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec(None, axis_name),
                  jax.sharding.PartitionSpec(axis_name, None)),
        out_specs=jax.sharding.PartitionSpec(axis_name, None),
        check_vma=False,
    )(a, w)

# SPDX-License-Identifier: Apache-2.0
"""Hierarchical (hypercube + twin-pair) all-gather matmul kernel.

Computes all_gather(x, axis=0) @ y in the hierarchical style, the rotation
mirror of hier_matmul_reduce_scatter: instead of a tp-1-hop ring, the x
chunks disseminate over log2(num_chips) hypercube rounds between chips
(payload doubling each round) and one final twin exchange that moves the
whole other-parity half over the fast intra-chip D2D link. The matmul is
arrival-driven — each gathered chunk is dotted with the resident y shard as
soon as it lands, and each round's send is issued before the previous
round's arrivals are dotted so the wire hides under the MXU.

Layout contract: logical id == chip * 2 + core (use
topology.make_collective_mesh('hier')); the chunk owned by device d is rows
[d*m_per, (d+1)*m_per) of the gathered x. The gather buffer is addressed in
global chunk space and every slot/semaphore cell is written exactly once per
call; the kernel opens with a double barrier over {twin} | {hypercube
partners} on collective_id 3.
"""
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util

_COLLECTIVE_ID = 3


def _hier_agmm_kernel(
    x_hbm,
    y_hbm,
    out_hbm,
    gather_hbm,
    y_vmem,
    x_vmem,
    o_vmem,
    sem_y,
    sem_x,
    sem_o,
    sems_hyper,
    sems_twin,
    *,
    num_chips,
    num_dims,
    m_per,
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

    y_copy = pltpu.make_async_copy(y_hbm, y_vmem, sem_y)
    y_copy.start()

    # Seed the gather buffer with my own chunk (position = my chunk index).
    seed_copy = pltpu.make_async_copy(x_hbm, gather_hbm.at[my_chip * 2 + bit],
                                      sem_x)
    seed_copy.start()
    seed_copy.wait()
    y_copy.wait()

    def held_label(r, op):
        """Chip label of the op-th chunk I hold entering round r: dims < r
        enumerate the already-gathered subcube (op bits), dims >= r come from
        my own chip id."""
        base = jnp.int32(0)
        for bit_pos in range(r):
            base = base | (jnp.int32((op >> bit_pos) & 1) << bit_pos)
        for d in range(r, num_dims):
            base = base | (((my_chip >> d) & 1) << d)
        return base

    def hyper_send(r, op):
        """Round r sends my 2^r held chunks along dim r; the partner's
        symmetric copies land in my gather buffer at their own positions
        (written once — the partner's held set is disjoint from mine)."""
        partner = (my_chip ^ (1 << r)) * 2 + bit
        chunk = held_label(r, op) * 2 + bit
        return pltpu.make_async_remote_copy(
            src_ref=gather_hbm.at[chunk],
            dst_ref=gather_hbm.at[chunk],
            send_sem=sems_hyper.at[r, op],
            recv_sem=sems_hyper.at[r, op],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def twin_send(q):
        """The whole my-parity set to the twin (its other-parity half)."""
        chunk = q * 2 + bit
        return pltpu.make_async_remote_copy(
            src_ref=gather_hbm.at[chunk],
            dst_ref=gather_hbm.at[chunk],
            send_sem=sems_twin.at[q],
            recv_sem=sems_twin.at[q],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def dot_chunk(chunk):
        """out rows of one gathered chunk: x_chunk @ y."""
        x_copy = pltpu.make_async_copy(gather_hbm.at[chunk], x_vmem, sem_x)
        x_copy.start()
        x_copy.wait()
        o_vmem[...] = jnp.dot(x_vmem[...], y_vmem[...],
                              preferred_element_type=jnp.float32).astype(
                                  o_vmem.dtype)
        o_copy = pltpu.make_async_copy(
            o_vmem, out_hbm.at[pl.ds(chunk * m_per, m_per), :], sem_o)
        o_copy.start()
        o_copy.wait()

    # Round 0's send goes out before any local compute.
    if num_dims:
        hyper_send(0, 0).start()
    dot_chunk(my_chip * 2 + bit)

    for r in range(num_dims):
        ops = 2**r
        for op in range(ops):
            # My send done AND the partner's op-th chunk arrived.
            hyper_send(r, op).wait()
        if r + 1 < num_dims:
            for op in range(2**(r + 1)):
                hyper_send(r + 1, op).start()
        else:
            @pl.loop(0, num_chips)
            def _twin_out(q):
                twin_send(q).start()

        # Dot this round's arrivals (the partner's held set: my dims >= r+1
        # bits, op bits below, dim-r bit flipped) while the next wave flies.
        # Traced loop so the fp32 dot value stays one-body-deep.
        @pl.loop(0, ops)
        def _dot_arrivals(op):
            arrived = held_label(r, op) ^ (1 << r)
            dot_chunk(arrived * 2 + bit)

    if not num_dims:
        @pl.loop(0, num_chips)
        def _twin_out(q):
            twin_send(q).start()

    @pl.loop(0, num_chips)
    def _twin_in(q):
        twin_send(q).wait()
        dot_chunk(q * 2 + (1 - bit))


def get_vmem_estimate_bytes(m_per, k, n_per, itemsize):
    """Resident y + one x tile + the fp32 dot value + the o cast tile."""
    return (k * n_per * itemsize + m_per * k * itemsize +
            m_per * n_per * 4 + m_per * n_per * itemsize)


def validate_inputs(x, y, tp_size):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-D")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {y.dtype}")
    m_per, k = x.shape
    k_y, n_per = y.shape
    if k != k_y:
        raise ValueError(f"contraction mismatch: {k} vs {k_y}")
    if tp_size % 2:
        raise ValueError(f"tp_size {tp_size} must pair 2 cores per chip")
    num_chips = tp_size // 2
    if num_chips & (num_chips - 1):
        raise ValueError(f"num_chips {num_chips} must be a power of two")
    if m_per % 8:
        raise ValueError(f"m per device {m_per} must be sublane-aligned "
                         f"(divisible by 8)")
    if k % 128 or n_per % 128:
        raise ValueError(f"k {k} and n per device {n_per} must be divisible "
                         f"by 128")


def hier_all_gather_matmul(
    x,
    y,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
):
    """all_gather(x, axis=0) @ y via hypercube dissemination + twin exchange.

    x: [M, K] P(axis, None), y: [K, N] P(None, axis) -> out [M, N] P(None,
    axis). The mesh device order must satisfy logical id == chip * 2 + core
    (topology.make_collective_mesh('hier')).
    """
    tp_size = mesh.shape[axis_name]
    m, k = x.shape
    m_per = m // tp_size
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0

    def per_device(x_local, y_local):
        validate_inputs(x_local, y_local, tp_size)
        n_per = y_local.shape[1]
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
                pltpu.VMEM((m_per, k), x_local.dtype),
                pltpu.VMEM((m_per, n_per), x_local.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((max(num_dims, 1),
                                         max(2**(num_dims - 1), 1))),
                pltpu.SemaphoreType.DMA((num_chips, )),
            ),
            grid=(1, ),
        )
        kernel = functools.partial(
            _hier_agmm_kernel,
            num_chips=num_chips,
            num_dims=num_dims,
            m_per=m_per,
            axis_name=axis_name,
        )
        out, _ = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((m, n_per), x_local.dtype),
                jax.ShapeDtypeStruct((tp_size, m_per, k), x_local.dtype),
            ),
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=get_vmem_estimate_bytes(
                    m_per, k, n_per, x_local.dtype.itemsize) +
                8 * 1024 * 1024,
            ),
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

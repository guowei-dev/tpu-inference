# SPDX-License-Identifier: Apache-2.0
"""Matmul reduce-scatter kernel.

Computes reduce_scatter(a @ w, axis=0) as a bidirectional ring that overlaps
the inter-chip transfer of partially-reduced output chunks with the local
matmul: at ring step s each device computes its local partial for the
travelling output chunk, adds the accumulator received from its neighbor, and
forwards the sum; the chunk lands on its owner at the last step. The structure
mirrors all_gather_matmul with the rotation inverted (the OUTPUT accumulator
travels, not the input): each block's remote copy is issued as soon as that
block's add completes, and the receiver waits per block one ring step later.

Left halves of each chunk travel toward device-1, right halves toward device+1
(chunk schedules c_L(d,s) = (d+s+1) % tp, c_R(d,s) = (d-s-1) % tp; owner lands
at s = tp-1). Received accumulators land in per-step HBM slots (never reused,
so a neighbor running ahead cannot overwrite an unconsumed block) and
semaphores are per block (so one block's arrival can never satisfy another
block's wait). The outgoing buffer holds the full [m_per, k] step payload;
its per-block regions are disjoint and each region is only rewritten after the
previous step's copy from it has been waited.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util

P = jax.sharding.PartitionSpec


def _matmul_reduce_scatter_kernel(
    # Inputs
    a_hbm_ref,  # [m, n_pad]  (all m rows of this device's columns, lane-padded)
    w_hbm_ref,  # [n_per, k_out]
    # Outputs
    o_hbm_ref,  # [m_per, k_out]
    recv_hbm_ref,  # [num_devices - 1, m_per, k_out]  per-step landing slots
    # Scratches
    a_copy_sem,  # []
    w_copy_sem,  # []
    wire_copy_sem,  # []
    o_copy_sem,  # []
    send_sems,  # [grid_m, grid_ko]
    recv_sems,  # [grid_m, grid_ko]
    a_vmem_ref,  # [m_per, n_pad]  rows for this step's two chunks
    w_vmem_ref,  # [n_pad, k_out]  resident
    wire_vmem_ref,  # [bm, bk_out]  incoming accumulator stage
    out_vmem_ref,  # [m_per, k_out]  outgoing step payload (disjoint blocks)
    acc_vmem_ref,  # [bm, bk_out]  f32 local partial
    axis_name: str,
    bm: int,
    bk_out: int,
    bnc: int,
    n_actual: int,
):
    num_devices = pl.num_programs(0)
    grid_m = pl.num_programs(1)
    grid_nc = pl.num_programs(3)
    outer = pl.program_id(0)  # == ring schedule step s
    bm_i = pl.program_id(1)
    bko_i = pl.program_id(2)
    nc_i = pl.program_id(3)
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
    m_per_device, n_pad = a_vmem_ref.shape
    padded = n_pad != n_actual
    m_ppd = m_per_device // 2
    gm_half = grid_m // 2
    # chunk whose rows this outer step works on, per travel direction
    chunk_left = lax.rem(my_id + outer + 1, jnp.int32(num_devices))
    chunk_right = lax.rem(my_id - outer - 1 + 2 * num_devices,
                          jnp.int32(num_devices))
    is_left_block = bm_i < gm_half
    bm_rows = pl.ds(bm_i * bm, bm)
    ko_slice = pl.ds(bko_i * bk_out, bk_out)
    nc_slice = pl.ds(nc_i * bnc, bnc)

    def _start_or_wait(op, wait):
        if wait:
            op.wait()
        else:
            op.start()

    def _do_a_local_copy(wait: bool = False):
        # vmem row r == within-chunk row r: left half rows come from
        # chunk_left, right half rows from chunk_right.
        left_op = pltpu.make_async_copy(
            src_ref=a_hbm_ref.at[pl.ds(chunk_left * m_per_device, m_ppd), :],
            dst_ref=a_vmem_ref.at[:m_ppd, :],
            sem=a_copy_sem,
        )
        right_op = pltpu.make_async_copy(
            src_ref=a_hbm_ref.at[
                pl.ds(chunk_right * m_per_device + m_ppd, m_ppd), :],
            dst_ref=a_vmem_ref.at[m_ppd:, :],
            sem=a_copy_sem,
        )
        _start_or_wait(left_op, wait)
        _start_or_wait(right_op, wait)

    def _do_w_local_copy(wait: bool = False):
        if padded:
            # HBM holds only n_actual rows (grid_nc == 1 when padded); the
            # pad rows above n_actual are zeroed once and never DMA'd.
            src = w_hbm_ref.at[:, ko_slice]
            dst = w_vmem_ref.at[pl.ds(0, n_actual), ko_slice]
        else:
            src = w_hbm_ref.at[nc_slice, ko_slice]
            dst = w_vmem_ref.at[nc_slice, ko_slice]
        op = pltpu.make_async_copy(src_ref=src, dst_ref=dst, sem=w_copy_sem)
        _start_or_wait(op, wait)

    def _remote_copy_op(step):
        # SPMD-symmetric: as issuer this sends the block's region of the step
        # payload to the neighbor's landing slot `step`; re-constructed one
        # outer later, .wait() consumes this device's matching send signal and
        # the incoming copy's recv signal.
        target = jnp.where(is_left_block, left_neighbor, right_neighbor)
        return pltpu.make_async_remote_copy(
            src_ref=out_vmem_ref.at[bm_rows, ko_slice],
            dst_ref=recv_hbm_ref.at[step, bm_rows, ko_slice],
            send_sem=send_sems.at[bm_i, bko_i],
            recv_sem=recv_sems.at[bm_i, bko_i],
            device_id=(target, ),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _do_wire_local_copy(wait: bool = False):
        op = pltpu.make_async_copy(
            src_ref=recv_hbm_ref.at[outer - 1, bm_rows, ko_slice],
            dst_ref=wire_vmem_ref,
            sem=wire_copy_sem,
        )
        _start_or_wait(op, wait)

    def _do_o_local_copy(wait: bool = False):
        op = pltpu.make_async_copy(
            src_ref=out_vmem_ref.at[bm_rows, ko_slice],
            dst_ref=o_hbm_ref.at[bm_rows, ko_slice],
            sem=o_copy_sem,
        )
        _start_or_wait(op, wait)

    ### ------- Kernel start ------- ###

    @pl.when(jnp.logical_and(outer == 0, jnp.logical_and(
        bm_i == 0, jnp.logical_and(bko_i == 0, nc_i == 0))))
    @jax.named_scope("_barrier")
    def _barrier():
        util.local_barrier(left_neighbor, right_neighbor)

    # w's pad rows above n_actual never receive a DMA (a arrives lane-padded
    # with zeros from the wrapper — a ragged lane window is not a legal DMA
    # dst, but a ragged SUBLANE window is); zero them once so the dot's extra
    # terms are exactly zero (0 * an uninitialized-VMEM NaN would poison the
    # accumulator).
    if padded:

        @pl.when(jnp.logical_and(outer == 0, jnp.logical_and(
            bm_i == 0, jnp.logical_and(bko_i == 0, nc_i == 0))))
        @jax.named_scope("_zero_pad")
        def _zero_pad():
            w_vmem_ref.at[pl.ds(n_actual, n_pad - n_actual), :][...] = (
                jnp.zeros((n_pad - n_actual, w_vmem_ref.shape[1]),
                          w_vmem_ref.dtype))

    # w resident load: outer 0, first m block; one (nc, ko) tile per step.
    @pl.when(jnp.logical_and(outer == 0, bm_i == 0))
    @jax.named_scope("_start_w_local_copy")
    def _start_w_local_copy():
        _do_w_local_copy(wait=False)

    # a rows for this outer's two chunks: once per outer.
    a_copy_cond = jnp.logical_and(bm_i == 0,
                                  jnp.logical_and(bko_i == 0, nc_i == 0))

    @pl.when(a_copy_cond)
    @jax.named_scope("_start_a_local_copy")
    def _start_a_local_copy():
        _do_a_local_copy(wait=False)

    @pl.when(jnp.logical_and(outer == 0, bm_i == 0))
    @jax.named_scope("_wait_w_local_copy")
    def _wait_w_local_copy():
        _do_w_local_copy(wait=True)

    @pl.when(a_copy_cond)
    @jax.named_scope("_wait_a_local_copy")
    def _wait_a_local_copy():
        _do_a_local_copy(wait=True)

    # Incoming accumulator for this block: wait the copy the neighbor issued
    # at its previous outer (also consumes this device's matching send signal,
    # freeing the block's out_vmem region for rewrite), then stage it to VMEM.
    @pl.when(jnp.logical_and(outer > 0, nc_i == 0))
    @jax.named_scope("_wait_incoming_and_stage")
    def _wait_incoming_and_stage():
        _remote_copy_op(outer - 1).wait()
        _do_wire_local_copy(wait=False)

    @jax.named_scope("_mxu")
    def _mxu():
        block = jnp.dot(
            a_vmem_ref.at[bm_rows, nc_slice][...],
            w_vmem_ref.at[nc_slice, ko_slice][...],
            preferred_element_type=jnp.float32,
        )
        if grid_nc == 1:
            acc_vmem_ref[...] = block
        else:

            @pl.when(nc_i == 0)
            def _set():
                acc_vmem_ref[...] = block

            @pl.when(nc_i > 0)
            def _accumulate():
                acc_vmem_ref[...] += block

    _mxu()

    # Block complete: add the wire, write the payload region, forward or emit.
    @pl.when(nc_i == grid_nc - 1)
    @jax.named_scope("_finish_block")
    def _finish_block():

        @pl.when(outer == 0)
        def _no_wire():
            out_vmem_ref.at[bm_rows, ko_slice][...] = acc_vmem_ref[...].astype(
                out_vmem_ref.dtype)

        @pl.when(outer > 0)
        def _with_wire():
            _do_wire_local_copy(wait=True)
            out_vmem_ref.at[bm_rows, ko_slice][...] = (
                acc_vmem_ref[...] +
                wire_vmem_ref[...].astype(jnp.float32)).astype(
                    out_vmem_ref.dtype)

        @pl.when(outer < num_devices - 1)
        def _forward():
            _remote_copy_op(outer).start()

        @pl.when(outer == num_devices - 1)
        def _emit():
            _do_o_local_copy(wait=False)
            _do_o_local_copy(wait=True)

    ### ------- Kernel end ------- ###


def validate_inputs(a, w, tp_size):
    """Validates the inputs to the matmul_reduce_scatter kernel."""
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D, got shapes {a.shape} and {w.shape}.")
    if a.dtype != w.dtype:
        raise ValueError(
            f"Input dtypes must match, got {a.dtype} and {w.dtype}.")
    m, n_per = a.shape
    n_per_from_w, k_out = w.shape
    if n_per != n_per_from_w:
        raise ValueError(
            "Incompatible shapes for matmul: contracting dimension mismatch:"
            f" {a.shape} and {w.shape}.")
    if m % (tp_size * 2) != 0:
        raise ValueError(f"a.shape[0] ({m}) must be divisible by tp_size * 2 "
                         f"({tp_size * 2}).")
    if (m // tp_size // 2) % 8 != 0:
        raise ValueError(f"m ({m}) must be divisible by tp_size * 2 * 8 "
                         f"({tp_size * 2 * 8}).")
    if k_out % 128 != 0:
        raise ValueError(f"k_out ({k_out}) must be divisible by 128.")


def matmul_reduce_scatter(
    a: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    collective_id: int | None = 1,
    bm: int | None = None,
    bk_out: int | None = None,
    bnc: int | None = None,
):
    """Computes reduce_scatter(a @ w, axis=0) with a fused ring kernel.

  Args:
    a: LHS of the matmul, [m, n] sharded P(None, axis_name).
    w: RHS of the matmul, [n, k] sharded P(axis_name, None).
    mesh: JAX mesh.
    axis_name: Mesh axis to reduce-scatter over.
    collective_id: Barrier semaphore id; keep distinct from other collective
      kernels in the same program.
    bm: Output row-block size; must divide m // tp_size // 2.
    bk_out: Output column-block size; must divide k.
    bnc: Contraction block size; must divide n // tp_size. When n // tp_size
      is not a multiple of 128 the kernel runs its VMEM operands at the
      lane-padded width (pad region zeroed) and only the default bnc (the
      whole padded width) is supported.

  Returns:
    reduce_scatter(a @ w, axis=0): [m, k] sharded P(axis_name, None).
  """
    tp_size = mesh.shape[axis_name]
    validate_inputs(a, w, tp_size)
    m, n = a.shape
    _, k_out = w.shape
    n_per_device = n // tp_size
    m_per_device = m // tp_size
    m_ppd = m_per_device // 2
    # Mosaic requires the a operand's lane width and lane-dim block offsets
    # to be multiples of 128; an unaligned n_per_device runs at the padded
    # width with the VMEM pad region zeroed (HBM stays unpadded).
    n_pad = -(-n_per_device // 128) * 128
    if bm is None:
        bm = min(m_ppd, 256)
    if bk_out is None:
        bk_out = min(k_out, 2048)
    if bnc is None:
        bnc = n_pad
    if m_ppd % bm != 0:
        raise ValueError(f"bm ({bm}) must divide m // tp_size // 2 ({m_ppd}).")
    if k_out % bk_out != 0:
        raise ValueError(f"bk_out ({bk_out}) must divide k ({k_out}).")
    if n_pad != n_per_device and bnc != n_pad:
        raise ValueError(
            f"bnc ({bnc}) must equal the lane-padded contraction width "
            f"({n_pad}) when n // tp_size ({n_per_device}) is not a multiple "
            f"of 128.")
    if n_pad % bnc != 0:
        raise ValueError(
            f"bnc ({bnc}) must divide n // tp_size ({n_per_device}).")
    grid_m = m_per_device // bm
    grid_ko = k_out // bk_out
    grid_nc = n_pad // bnc

    out_shape = [
        jax.ShapeDtypeStruct((m_per_device, k_out), a.dtype),  # output
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, k_out),
                             a.dtype),  # per-step landing slots
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,  # a_copy_sem
            pltpu.SemaphoreType.DMA,  # w_copy_sem
            pltpu.SemaphoreType.DMA,  # wire_copy_sem
            pltpu.SemaphoreType.DMA,  # o_copy_sem
            pltpu.SemaphoreType.DMA((grid_m, grid_ko)),  # send_sems
            pltpu.SemaphoreType.DMA((grid_m, grid_ko)),  # recv_sems
            pltpu.VMEM((m_per_device, n_pad), a.dtype),  # a rows
            pltpu.VMEM((n_pad, k_out), w.dtype),  # w resident
            pltpu.VMEM((bm, bk_out), a.dtype),  # wire stage
            pltpu.VMEM((m_per_device, k_out), a.dtype),  # step payload
            pltpu.VMEM((bm, bk_out), jnp.float32),  # local partial acc
        ),
        grid=(tp_size, grid_m, grid_ko, grid_nc),
    )
    flops = 2 * m * n_per_device * k_out
    bytes_accessed = a.dtype.itemsize * (m * n_per_device +
                                         n_per_device * k_out +
                                         m_per_device * k_out)
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)
    vmem_bytes = (m_per_device * n_pad * a.dtype.itemsize +
                  n_pad * k_out * w.dtype.itemsize +
                  m_per_device * k_out * a.dtype.itemsize +
                  bm * bk_out * a.dtype.itemsize + 4 * bm * bk_out)

    @jax.jit(static_argnames=["bm", "bk_out", "bnc"])
    def _matmul_reduce_scatter_call(a, w, bm, bk_out, bnc):
        if n_pad != n_per_device:
            # A ragged lane window is not a legal Mosaic DMA dst, so the
            # kernel takes `a` lane-padded with zeros; w stays unpadded (its
            # contraction dim sits on sublanes, where a ragged window is
            # legal) and the kernel zeroes its VMEM pad rows instead.
            a = jnp.pad(a, ((0, 0), (0, n_pad - n_per_device)))
        return pl.pallas_call(
            functools.partial(
                _matmul_reduce_scatter_kernel,
                axis_name=axis_name,
                bm=bm,
                bk_out=bk_out,
                bnc=bnc,
                n_actual=n_per_device,
            ),
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=vmem_bytes + 8 * 1024 * 1024,
            ),
            cost_estimate=cost_estimate,
            name=f"matmul_reduce_scatter_kernel_bm_{bm}_bko_{bk_out}"
            f"_bnc_{bnc}",
        )(a, w)[0]

    shard_map_kernel = jax.jit(
        jax.shard_map(
            functools.partial(
                _matmul_reduce_scatter_call,
                bm=bm,
                bk_out=bk_out,
                bnc=bnc,
            ),
            mesh=mesh,
            in_specs=(P(None, axis_name), P(axis_name, None)),
            out_specs=P(axis_name, None),
            check_vma=False,
        ), )

    return shard_map_kernel(a, w)

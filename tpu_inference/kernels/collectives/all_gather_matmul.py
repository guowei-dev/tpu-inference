# SPDX-License-Identifier: Apache-2.0
"""All-gather matmul kernel."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import (
    all_gather_matmul_tuned_block_sizes, util)

P = jax.sharding.PartitionSpec


def _cdiv(x, y):
    return (x + y - 1) // y


# TODO(chengjiyao): try unrolling the loop instead of using pallas_call grid
# TODO(chengjiyao): try m tiling
# TODO(chengjiyao): try using [bm, bk] and [bk, bn] scratches memory shape for
# large bm
# TODO(chengjiyao): try splitting to two parts when n_per_device is large:
# output_0, gatherd_x = ag-matmul(x, y_0)
# output_1 = matmul(gatherd_x, y_1)
# output = concat(output_0, output_1)
# TODO(chengjiyao): investigate the register spilling
def _all_gather_kernel(
    # Inputs
    x_hbm_ref,  # [m_per_device, k]
    y_hbm_ref,  # [k, n_per_device]
    # Outputs
    o_hbm_ref,  # [m, n_per_device]
    x_hbm_scratch_ref,  # [num_devices - 1, m_per_device, k]
    # Scratches
    x_local_copy_sem,  # []
    y_local_copy_sem,  # []
    o_local_copy_sem,  # []
    send_sems,  # [2, num_devices - 1] for left and right
    recv_sems,  # [2, num_devices - 1] for left and right
    x_vmem_scratch_ref,  # [2, m_per_device, k]
    y_vmem_scratch_ref,  # [k, n_per_device]
    o_vmem_scratch_ref,  # [2, m_per_device, bn]
    acc_vmem_scratch_ref,  # [m_per_device, bn] of jnp.float32
    axis_name: str,
    bn: int,
    bk: int,
    bm: int,
    fold: int = 1,
    debug_mode=False,
    rhs_transpose: bool = False,
):
    """Pallas kernel for all-gather.

  Args:
    x_hbm_ref: LHS of the matmul before all-gather.
    y_hbm_ref: RHS of the matmul.
    o_hbm_ref: Output of the matmul.
    x_hbm_scratch_ref: Scratch memory for LHS of the matmul.
    x_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    y_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    o_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    send_sem: DMA semaphore for the remote send.
    capacity_sem: Capacity semaphore for the remote send.
    recv_sems: DMA semaphore for the remote receive.
    x_vmem_scratch_ref: Scratch memory for LHS of the matmul.
    y_vmem_scratch_ref: Scratch memory for RHS of the matmul.
    o_vmem_scratch_ref: Scratch memory for output of the matmul.
  """
    num_devices = pl.num_programs(0) - 2
    m_per_device, _ = x_hbm_ref.shape
    m_per_device_per_direction = m_per_device // 2
    # m tiling: the x/o VMEM working sets are [bm, ...] blocks so their size no
    # longer scales with m. grid_m == 1 (bm == m_per_device) preserves the
    # original 3-D grid and code paths exactly.
    grid_m = m_per_device // bm
    if grid_m > 1:
        grid_n = pl.num_programs(2)
        grid_k = pl.num_programs(3)
        bm_i = pl.program_id(1)
        bn_i = pl.program_id(2)
        bk_i = pl.program_id(3)
    else:
        grid_n = pl.num_programs(1)
        grid_k = pl.num_programs(2)
        bm_i = 0
        bn_i = pl.program_id(1)
        bk_i = pl.program_id(2)
    outer_step = pl.program_id(0)
    gn_by_gk = grid_n * grid_k
    steps_per_outer = grid_m * gn_by_gk
    global_step_id = (outer_step * steps_per_outer + bm_i * gn_by_gk +
                      bn_i * grid_k + bk_i)
    mxu_total_steps = num_devices * steps_per_outer
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
    x_hbm_receiving_slot = outer_step
    x_hbm_working_slot = outer_step - 1
    x_vmem_receiving_slot = (outer_step * grid_m + bm_i) % 2
    x_vmem_working_slot = (global_step_id - 1) // gn_by_gk % 2
    o_receiving_slot = lax.rem((global_step_id + grid_k - 1) // grid_k, 2)
    o_working_slot = 1 - o_receiving_slot
    bm_rows = pl.ds(bm_i * bm, bm)
    if fold > 1:
        # Compute-fold (small m_per_device): chunk c (0 = local, c >= 1 =
        # arrival slot c - 1) is copied at outer step c into row
        # (c % fold) * m_per_device of group buffer (c // fold) % 2, and ONE
        # [fold * m_per_device, bk] dot per (bn, bk) tile at each
        # fold-boundary outer computes the whole group's products at full MXU
        # row occupancy — a sub-128-row per-chunk dot starves the MXU at
        # ~rows/128. The ring hops, the travel halves and the o row mapping
        # are unchanged; per-row contraction order is preserved, so the
        # output is bitwise-equal to the per-chunk schedule.
        group_receiving_slot = lax.rem(outer_step // fold, 2)
        group_working_slot = lax.rem(outer_step // fold + 1, 2)
        fold_rows = pl.ds(lax.rem(outer_step, fold) * m_per_device,
                          m_per_device)

    def _n_window(idx):
        """The n window for block `idx`, or the whole extent at grid_n == 1.

        A traced `idx * bn` offset is what requires `128 | bn`: Mosaic must
        prove the tiled-dim index of the memref_slice is tile-aligned. At
        grid_n == 1 the offset is structurally zero, so emitting the whole
        extent instead makes an unaligned n // tp_size legal.
        """
        return slice(None) if grid_n == 1 else pl.ds(idx * bn, bn)

    def debug_print(msg, *args):
        if debug_mode:

            @pl.when(my_id == 0)
            def _debug_print():
                pl.debug_print(msg, *args)

    def _start_or_wait_copy(
        op: jax._src.pallas.mosaic.primitives.AsyncCopyDescriptor,
        wait: bool = False,
    ):
        if wait:
            op.wait()
        else:
            op.start()

    def _do_first_x_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first x local copy, x_vmem_receiving_slot={},"
            " bk_i={}",
            int(wait),
            x_vmem_receiving_slot,
            bk_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        src_rows = bm_rows if grid_m > 1 else slice(None)
        if fold > 1:
            dst = x_vmem_scratch_ref.at[group_receiving_slot, fold_rows,
                                        k_slice]
        else:
            dst = x_vmem_scratch_ref.at[x_vmem_receiving_slot, :, k_slice]
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[src_rows, k_slice],
            dst_ref=dst,
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_left_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent x left local copy,"
            " x_hbm_working_slot={}, x_vmem_receiving_slot={}, bk_i={}",
            int(wait),
            x_hbm_working_slot,
            x_vmem_receiving_slot,
            bk_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot,
                :m_per_device_per_direction,
                k_slice,
            ],
            dst_ref=x_vmem_scratch_ref.at[
                x_vmem_receiving_slot,
                :m_per_device_per_direction,
                k_slice,
            ],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_right_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent x right local copy,"
            " x_hbm_working_slot={}, x_vmem_receiving_slot={}, bk_i={}",
            int(wait),
            x_hbm_working_slot,
            x_vmem_receiving_slot,
            bk_i,
        )
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            dst_ref=x_vmem_scratch_ref.at[
                x_vmem_receiving_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_local_copy(wait: bool = False):
        if fold > 1:
            # chunk c = outer_step, one contiguous copy into its group row.
            x_local_copy_op = pltpu.make_async_copy(
                src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot, :,
                                             pl.ds(bk_i * bk, bk)],
                dst_ref=x_vmem_scratch_ref.at[group_receiving_slot, fold_rows,
                                              pl.ds(bk_i * bk, bk)],
                sem=x_local_copy_sem,
            )
            _start_or_wait_copy(x_local_copy_op, wait)
        elif grid_m > 1:
            # one bm block lies entirely on one side of the left/right split
            # (bm divides m_per_device_per_direction), so a single row-sliced
            # copy replaces the left/right pair.
            x_local_copy_op = pltpu.make_async_copy(
                src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot, bm_rows,
                                             pl.ds(bk_i * bk, bk)],
                dst_ref=x_vmem_scratch_ref.at[x_vmem_receiving_slot, :,
                                              pl.ds(bk_i * bk, bk)],
                sem=x_local_copy_sem,
            )
            _start_or_wait_copy(x_local_copy_op, wait)
        else:
            _do_subsequent_x_left_local_copy(wait)
            _do_subsequent_x_right_local_copy(wait)

    def _do_y_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do y local copy, bk_i={}, bn_i={}",
            int(wait),
            bk_i,
            bn_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        n_slice = _n_window(bn_i)
        if rhs_transpose:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[n_slice, k_slice],
                dst_ref=y_vmem_scratch_ref.at[n_slice, k_slice],
                sem=y_local_copy_sem,
            )
        else:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[k_slice, n_slice],
                dst_ref=y_vmem_scratch_ref.at[k_slice, n_slice],
                sem=y_local_copy_sem,
            )
        _start_or_wait_copy(y_local_copy_op, wait)

    def _do_first_left_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first left remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot,
                                         0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_first_right_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first right remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[
                x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_subsequent_left_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent left remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot,
                                         0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot,
                                         0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_subsequent_right_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent right remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot, m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[
                x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_mxu():
        working_global_step_id = global_step_id - 1
        working_bk_i = working_global_step_id % grid_k
        working_bn_i = working_global_step_id % gn_by_gk // grid_k
        debug_print(
            "[AGMM debug] do mxu, x_vmem_working_slot={}, o_receiving_slot={},"
            " working_bk_i={}, working_bn_i={}",
            x_vmem_working_slot,
            o_receiving_slot,
            working_bk_i,
            working_bn_i,
        )
        k_slice = pl.ds(working_bk_i * bk, bk)
        n_slice = _n_window(working_bn_i)

        if grid_k == 1:
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot][...]
                rhs = y_vmem_scratch_ref.at[n_slice, :][...]
                o_vmem_scratch_ref.at[o_receiving_slot][...] = lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
            else:
                o_vmem_scratch_ref.at[o_receiving_slot][...] = jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot][...],
                    y_vmem_scratch_ref.at[:, n_slice][...],
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
        else:
            # TODO(chengjiyao): optimize the vstore
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot, :,
                                            k_slice][...]
                rhs = y_vmem_scratch_ref.at[n_slice, k_slice][...]
                acc_vmem_scratch_ref[...] += lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            else:
                acc_vmem_scratch_ref[...] += jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot, :,
                                          k_slice][...],
                    y_vmem_scratch_ref.at[k_slice, n_slice][...],
                    preferred_element_type=jnp.float32,
                )

            @pl.when(working_bk_i == grid_k - 1)
            def _update():
                debug_print(
                    "[AGMM debug] update, o_receiving_slot={}",
                    o_receiving_slot,
                )
                o_vmem_scratch_ref.at[o_receiving_slot][
                    ...] = acc_vmem_scratch_ref[...].astype(
                        x_vmem_scratch_ref.dtype)
                # TODO(chengjiyao): based on the kyuyeunk' suggestion:
                # this logic can be more optimized. right now it does this.
                # line 316 performs dot
                # line 316 loads from acc_vmem_scartch_ref
                # line 316 adds resulting dot with acc_vmem_scratch_ref
                # line 316 stores result into acc_vmem_scratch_ref
                # line 335 loads from acc_vmem_scratch_ref again.
                # line 338 zero initializes & stores it to acc_vmem_scratch_ref
                # better way would be

                # perform dot
                # if working_bk_i != 0, load from acc_vmem_scratch_ref and add result
                # from previous step. If not, skip this process.
                # if working_bk_i == gk - 1, store the result from step 2 into
                # o_vmem_scratch_ref, if not, store it into acc_vmem_scratch_ref
                acc_vmem_scratch_ref[...] = jnp.zeros_like(
                    acc_vmem_scratch_ref)

    def _do_o_local_copy(wait: bool = False):
        working_global_step_id = global_step_id - grid_k - 1
        working_bn_i = (working_global_step_id % gn_by_gk) // grid_k
        n_slice = _n_window(working_bn_i)
        if grid_m > 1:
            # one bm block targets a single destination row range: chunk base
            # by travel direction (left half of the chunk = even m_ppd block,
            # right half = odd), plus the block's offset within its half.
            working_bm_i = (working_global_step_id %
                            steps_per_outer) // gn_by_gk
            offset = working_global_step_id // steps_per_outer
            left_base = ((my_id + offset) % num_devices *
                         2) * m_per_device_per_direction
            right_base = ((my_id - offset + num_devices) % num_devices * 2 +
                          1) * m_per_device_per_direction
            row0 = jnp.where(
                working_bm_i < grid_m // 2,
                left_base + working_bm_i * bm,
                right_base + working_bm_i * bm - m_per_device_per_direction,
            )
            o_local_copy_op = pltpu.make_async_copy(
                src_ref=o_vmem_scratch_ref.at[o_working_slot],
                dst_ref=o_hbm_ref.at[pl.ds(row0, bm), n_slice],
                sem=o_local_copy_sem,
            )
            _start_or_wait_copy(o_local_copy_op, wait)
            return
        offset = (global_step_id - 2) // gn_by_gk
        left_o_idx = (my_id + offset) % num_devices
        left_o_idx = left_o_idx * 2
        right_o_idx = (my_id - offset + num_devices) % num_devices
        right_o_idx = right_o_idx * 2 + 1
        debug_print(
            "[AGMM debug, wait={}] do o local copy, o_working_slot={},"
            " left_o_idx={}, right_o_idx={}, working_bn_i={}",
            int(wait),
            o_working_slot,
            left_o_idx,
            right_o_idx,
            working_bn_i,
        )
        o_left_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[
                o_working_slot, :m_per_device_per_direction],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * left_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        o_right_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[o_working_slot,
                                          m_per_device_per_direction:],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * right_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        _start_or_wait_copy(o_left_local_copy_op, wait)
        _start_or_wait_copy(o_right_local_copy_op, wait)

    def _do_mxu_folded():
        # One [fold * m_per_device, bk] dot per (bn, bk) tile over group
        # buffer (outer_step // fold - 1) % 2; same per-row contraction order
        # as the per-chunk dots.
        k_slice = pl.ds(bk_i * bk, bk)
        n_slice = _n_window(bn_i)
        unit = (outer_step // fold - 1) * grid_n + bn_i
        oslot = lax.rem(unit, 2)
        lhs = x_vmem_scratch_ref.at[group_working_slot, :, k_slice][...]
        if rhs_transpose:
            block = lax.dot_general(
                lhs,
                y_vmem_scratch_ref.at[n_slice, k_slice][...],
                dimension_numbers=(((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            block = jnp.dot(
                lhs,
                y_vmem_scratch_ref.at[k_slice, n_slice][...],
                preferred_element_type=jnp.float32,
            )
        if grid_k == 1:
            o_vmem_scratch_ref.at[oslot][...] = block.astype(
                x_vmem_scratch_ref.dtype)
        else:
            acc_vmem_scratch_ref[...] += block

            @pl.when(bk_i == grid_k - 1)
            def _update():
                o_vmem_scratch_ref.at[oslot][...] = acc_vmem_scratch_ref[
                    ...].astype(x_vmem_scratch_ref.dtype)
                acc_vmem_scratch_ref[...] = jnp.zeros_like(
                    acc_vmem_scratch_ref)

    def _do_o_export_folded(g_u, bn_u, wait: bool = False):
        # Export unit (g_u, bn_u): 2 * fold half-chunk DMAs; chunk
        # c = g_u * fold + j keeps the per-chunk o row mapping (offset = c).
        n_slice = _n_window(bn_u)
        slot = lax.rem(g_u * grid_n + bn_u, 2)
        half = m_per_device_per_direction
        for j in range(fold):
            c = g_u * fold + j
            left_o_idx = lax.rem(my_id + c, jnp.int32(num_devices)) * 2
            right_o_idx = lax.rem(my_id - c + jnp.int32(2 * num_devices),
                                  jnp.int32(num_devices)) * 2 + 1
            for row0, o_idx in (
                (j * m_per_device, left_o_idx),
                (j * m_per_device + half, right_o_idx),
            ):
                op = pltpu.make_async_copy(
                    src_ref=o_vmem_scratch_ref.at[slot,
                                                  pl.ds(row0, half), :],
                    dst_ref=o_hbm_ref.at[pl.ds(o_idx * half, half), n_slice],
                    sem=o_local_copy_sem,
                )
                _start_or_wait_copy(op, wait)

    ### ------- Kernel start ------- ###
    # TODO(chengjiyao): explore a fine-grained way to do the waits and signal

    debug_print(
        "===== starting a grid, outer_step={}, bn_i={}, bk_i={} =====",
        outer_step,
        bn_i,
        bk_i,
    )

    @pl.when(global_step_id == 0)
    @jax.named_scope("_start_first_remote_copy")
    def _start_first_remote_copy():
        if grid_k > 1:
            acc_vmem_scratch_ref[...] = jnp.zeros_like(acc_vmem_scratch_ref)
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        util.local_barrier(left_neighbor, right_neighbor)
        _do_first_left_remote_copy(wait=False)
        _do_first_right_remote_copy(wait=False)

    cond_start_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % steps_per_outer == 0,
    )

    @pl.when(cond_start_subsequent_remote_copy)
    @jax.named_scope("_start_subsequent_remote_copy")
    def _start_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=False)
        _do_subsequent_right_remote_copy(wait=False)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_start_first_local_x_copy")
    def _start_first_x_local_copy():
        _do_first_x_local_copy(wait=False)

    cond_subsequent_x_local_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices), bn_i == 0)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_start_subsequent_x_local_copy")
    def _start_subsequent_x_local_copy():
        _do_subsequent_x_local_copy(wait=False)

    y_copy_cond = (outer_step == 0 if grid_m == 1 else jnp.logical_and(
        outer_step == 0, bm_i == 0))

    @pl.when(y_copy_cond)
    @jax.named_scope("_start_y_local_copy")
    def _start_y_local_copy():
        _do_y_local_copy(wait=False)

    # Nothing before the first dot reads y, so the wait belongs at that dot —
    # `fold` outer steps in, or the next grid step when fold == 1 — not in the
    # step that issues the transfer, where it costs a hop for nothing.
    y_wait_cond = (outer_step == fold if fold > 1 else jnp.logical_and(
        global_step_id >= 1, global_step_id <= gn_by_gk))

    @pl.when(y_wait_cond)
    @jax.named_scope("_wait_y_local_copy")
    def _wait_y_local_copy():
        _do_y_local_copy(wait=True)

    if fold > 1:
        boundary = jnp.logical_and(
            lax.rem(outer_step, fold) == 0,
            jnp.logical_and(outer_step > 0, outer_step <= num_devices))
        g_cur = outer_step // fold - 1
        g_trail = (outer_step - 1) // fold - 1
        # unit (g, bn_i - 1) exports inside its boundary outer; a group's
        # last unit exports on the following outer step.
        exp_in = jnp.logical_and(boundary, bn_i > 0)
        exp_trail = jnp.logical_and(
            jnp.logical_and(lax.rem(outer_step, fold) == 1,
                            outer_step > fold),
            outer_step <= num_devices + 1)

        @pl.when(jnp.logical_and(exp_in, bk_i == 0))
        @jax.named_scope("_start_o_export")
        def _start_o_export_in():
            _do_o_export_folded(g_cur, bn_i - 1, wait=False)

        @pl.when(jnp.logical_and(exp_trail,
                                 jnp.logical_and(bn_i == 0, bk_i == 0)))
        @jax.named_scope("_start_o_export_trail")
        def _start_o_export_trail():
            _do_o_export_folded(g_trail, grid_n - 1, wait=False)

        @pl.when(boundary)
        @jax.named_scope("_mxu")
        def _mxu_folded():
            _do_mxu_folded()

        @pl.when(jnp.logical_and(exp_in, bk_i == grid_k - 1))
        @jax.named_scope("_wait_o_export")
        def _wait_o_export_in():
            _do_o_export_folded(g_cur, bn_i - 1, wait=True)

        @pl.when(jnp.logical_and(exp_trail,
                                 jnp.logical_and(bn_i == grid_n - 1,
                                                 bk_i == grid_k - 1)))
        @jax.named_scope("_wait_o_export_trail")
        def _wait_o_export_trail():
            _do_o_export_folded(g_trail, grid_n - 1, wait=True)
    else:

        def _get_start_o_local_copy_cond():
            if grid_k == 1:
                return jnp.logical_and(global_step_id >= 2, global_step_id
                                       < mxu_total_steps + 2)
            else:
                return jnp.logical_and(
                    jnp.logical_and(
                        global_step_id >= grid_k + 1,
                        global_step_id < mxu_total_steps + grid_k + 1,
                    ),
                    global_step_id % grid_k == 1,
                )

        @pl.when(_get_start_o_local_copy_cond())
        @jax.named_scope("_start_o_local_copy")
        def _start_o_local_copy():
            _do_o_local_copy(wait=False)

        @pl.when(
            jnp.logical_and(global_step_id >= 1, global_step_id
                            < 1 + mxu_total_steps))
        @jax.named_scope("_mxu")
        def _mxu():
            _do_mxu()

        def _get_wait_o_local_copy_cond():
            if grid_k == 1:
                return jnp.logical_and(global_step_id >= 2, global_step_id
                                       < mxu_total_steps + 2)
            else:
                return jnp.logical_and(
                    jnp.logical_and(
                        global_step_id >= grid_k + 1,
                        global_step_id < mxu_total_steps + grid_k + 1,
                    ),
                    global_step_id % grid_k == 0,
                )

        @pl.when(_get_wait_o_local_copy_cond())
        @jax.named_scope("_wait_o_local_copy")
        def _wait_o_local_copy():
            _do_o_local_copy(wait=True)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_wait_first_x_local_copy")
    def _wait_first_x_local_copy():
        _do_first_x_local_copy(wait=True)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_wait_subsequent_x_local_copy")
    def _wait_subsequent_x_local_copy():
        _do_subsequent_x_local_copy(wait=True)

    @pl.when(global_step_id == steps_per_outer - 1)
    @jax.named_scope("_wait_first_remote_copy")
    def _wait_first_remote_copy():
        _do_first_left_remote_copy(wait=True)
        _do_first_right_remote_copy(wait=True)

    cond_wait_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % steps_per_outer == steps_per_outer - 1,
    )

    @pl.when(cond_wait_subsequent_remote_copy)
    @jax.named_scope("_wait_subsequent_remote_copy")
    def _wait_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=True)
        _do_subsequent_right_remote_copy(wait=True)

    ### ------- Kernel end ------- ###


def _all_gather_kernel_unrolled(
    # Inputs
    x_hbm_ref,  # [m_per_device, k]
    y_hbm_ref,  # [k, n_per_device] (or transposed)
    # Outputs
    o_hbm_ref,  # [m, n_per_device]
    x_hbm_scratch_ref,  # [num_devices - 1, m_per_device, k]
    # Scratches
    x_local_copy_sem,
    y_local_copy_sem,
    o_local_copy_sem,
    send_sems,  # [2, num_devices - 1]
    recv_sems,  # [2, num_devices - 1]
    x_vmem_scratch_ref,  # [2, m_per_device, k]
    y_vmem_scratch_ref,  # [k, n_per_device]
    o_vmem_scratch_ref,  # [2, m_per_device, bn]
    acc_vmem_scratch_ref,  # unused (single k block)
    axis_name: str,
    rhs_transpose: bool = False,
):
    """Unrolled-ring variant of _all_gather_kernel for the single-block case
    (bn == n_per_device, bk == k, bm == m_per_device).

    Same schedule as the grid kernel — hop s and the local copy of hop s-1's
    chunk are started at step s, the MXU for chunk s-1 runs between the starts
    and the waits, and every hop is waited at the end of its own step — but as
    one pallas invocation with a Python loop, so slot and semaphore indices
    are static and there is no per-step grid re-entry. Profitable in the
    latency-bound mid-M range where the grid machinery is a visible fraction
    of the per-hop time.
    """
    del acc_vmem_scratch_ref
    num_devices = x_hbm_scratch_ref.shape[0] + 1
    m_per_device, _ = x_hbm_ref.shape
    half = m_per_device // 2
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))

    def _start_or_wait(op, wait):
        if wait:
            op.wait()
        else:
            op.start()

    def _hop(step, wait):
        # Ring hop `step`: send our (step 0) or the last-received (else) chunk
        # halves to the left/right neighbors' scratch slot `step`.
        src = x_hbm_ref if step == 0 else x_hbm_scratch_ref.at[step - 1]
        for direction, rows, neighbor in (
            (0, slice(0, half), left_neighbor),
            (1, slice(half, m_per_device), right_neighbor),
        ):
            _start_or_wait(
                pltpu.make_async_remote_copy(
                    src_ref=src.at[rows],
                    dst_ref=x_hbm_scratch_ref.at[step, rows],
                    send_sem=send_sems.at[direction, step],
                    recv_sem=recv_sems.at[direction, step],
                    device_id=(neighbor, ),
                    device_id_type=pl.DeviceIdType.MESH,
                ), wait)

    def _x_local(step, wait):
        src = x_hbm_ref if step == 0 else x_hbm_scratch_ref.at[step - 1]
        _start_or_wait(
            pltpu.make_async_copy(
                src_ref=src,
                dst_ref=x_vmem_scratch_ref.at[step % 2],
                sem=x_local_copy_sem,
            ), wait)

    def _y_local(wait):
        _start_or_wait(
            pltpu.make_async_copy(
                src_ref=y_hbm_ref,
                dst_ref=y_vmem_scratch_ref,
                sem=y_local_copy_sem,
            ), wait)

    def _mxu(step):
        lhs = x_vmem_scratch_ref.at[(step - 1) % 2][...]
        if rhs_transpose:
            out = lax.dot_general(
                lhs,
                y_vmem_scratch_ref[...],
                dimension_numbers=(((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            out = jnp.dot(lhs,
                          y_vmem_scratch_ref[...],
                          preferred_element_type=jnp.float32)
        o_vmem_scratch_ref.at[step % 2][...] = out.astype(
            x_vmem_scratch_ref.dtype)

    def _o_export(step, wait):
        # Export the MXU result of step - 1 (slot (step-1) % 2): its left half
        # is the chunk gathered from `offset` devices to the left, the right
        # half from `offset` to the right (same mapping as the grid kernel).
        offset = step - 2
        slot = (step - 1) % 2
        left_o_idx = (my_id + offset) % num_devices * 2
        right_o_idx = (my_id - offset + num_devices) % num_devices * 2 + 1
        for rows, o_idx in ((slice(0, half), left_o_idx),
                            (slice(half, m_per_device), right_o_idx)):
            _start_or_wait(
                pltpu.make_async_copy(
                    src_ref=o_vmem_scratch_ref.at[slot, rows],
                    dst_ref=o_hbm_ref.at[pl.ds(half * o_idx, half), :],
                    sem=o_local_copy_sem,
                ), wait)

    util.local_barrier(left_neighbor, right_neighbor)
    for step in range(num_devices + 2):
        if step == 0:
            _hop(0, wait=False)
            _x_local(0, wait=False)
            _y_local(wait=False)
            _x_local(0, wait=True)
            _hop(0, wait=True)
        elif step < num_devices:
            if step <= num_devices - 2:
                _hop(step, wait=False)
            _x_local(step, wait=False)
            if step >= 2:
                _o_export(step, wait=False)
            if step == 1:
                # First reader of y — see the grid kernel's `y_wait_cond`.
                _y_local(wait=True)
            _mxu(step)
            if step >= 2:
                _o_export(step, wait=True)
            _x_local(step, wait=True)
            if step <= num_devices - 2:
                _hop(step, wait=True)
        elif step == num_devices:
            _o_export(step, wait=False)
            _mxu(step)
            _o_export(step, wait=True)
        else:
            _o_export(step, wait=False)
            _o_export(step, wait=True)


# Measured scoped-VMEM ceiling (memory_space_assignment clamps requests here).
_VMEM_CAP_BYTES = 67043328


# FIXME(chengjiyao): make it accurate for the cases of quantization
def get_vmem_estimate_bytes(
    m,
    n,
    k,
    bn,
    acc_bytes,
    tp_size,
    x_dtype,
    y_dtype,
    out_dtype,
    bm=None,
    fold=1,
):
    """Returns the total vmem bytes used by the kernel."""
    m_per_device = m // tp_size
    n_per_device = n // tp_size
    if bm is None:
        bm = m_per_device
    y_vmem_bytes = (n_per_device * k * dtypes.itemsize_bits(y_dtype) // 8)
    total_bytes = (
        2 * fold * bm * k * dtypes.itemsize_bits(x_dtype) // 8
        # x_vmem_scratch_ref
        + y_vmem_bytes  # y_vmem_scratch_ref
        + 2 * fold * bm * bn * dtypes.itemsize_bits(out_dtype) // 8
        # o_vmem_scratch_ref
        + acc_bytes  # acc_vmem_scratch_ref, jnp.float32
    )
    return total_bytes


def validate_inputs(x, y, tp_size, rhs_transpose=False):
    """Validates the inputs to the all_gather_matmul kernel."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
    if x.dtype != y.dtype:
        raise ValueError(
            f"Input dtypes must match, got {x.dtype} and {y.dtype}.")
    m, k = x.shape
    if rhs_transpose:
        n, k_from_y = y.shape
    else:
        k_from_y, n = y.shape
    if k != k_from_y:
        raise ValueError(
            "Incompatible shapes for matmul: contracting dimension mismatch:"
            f" {x.shape} and {y.shape}.")

    if k % 128 != 0:
        raise ValueError(f"k ({k}) must be divisible by 128.")

    if n % 128 != 0:
        raise ValueError(f"n ({n}) must be divisible by 128.")

    m_per_device_per_direction = m // tp_size // 2
    if m_per_device_per_direction % 8 != 0:
        raise ValueError(f"m ({m}) must be divisible by {{tp_size * 2 * 8}}.")

    if m % (tp_size * 2) != 0:
        raise ValueError(
            f"x.shape[0] ({m}) must be divisible by tp_size * 2 ({tp_size * 2})'."
        )
    if n % tp_size != 0:
        raise ValueError(
            f"y.shape[{0 if rhs_transpose else 1}] ({n}) must be divisible by"
            f" tp_size ({tp_size}) on axis '{tp_size}'.")


def all_gather_matmul(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    collective_id: int | None = 0,
    bn: int | None = None,
    bk: int | None = None,
    bm: int | None = None,
    rhs_transpose: bool = False,
    unroll: bool | None = None,
    fold: int | None = None,
):
    """Performs all-gather on the input tensor and then a matmul.

  Args:
    x: LHS of the matmul before all-gather.
    y: RHS of the matmul. When n // tp_size is not a multiple of 128, store it
      with the row-major device layout (`jax.experimental.layout.Layout(
      major_to_minor=(0, 1))`): XLA's default for such a shape is column-major,
      which this kernel cannot take, so it relayouts the whole weight on every
      call — 1.45x at m = 256.
    mesh: JAX mesh.
    axis_name: Name of the axis to all-gather over.
    collective_id: An integer used for barrier semaphore allocation.
    bn: Number of blocks in the n dimension.
    bk: Number of blocks in the k dimension.
    bm: Row-block size for the x/output VMEM working set. Defaults to the full
      per-device chunk (m // tp_size); smaller values bound VMEM usage
      independently of m. Must divide m // tp_size // 2.
    rhs_transpose: If True, y is transposed.
    unroll: Use the unrolled-ring kernel (single-block schedules only, i.e.
      bn == n // tp_size, bk == k, bm == m // tp_size). None picks it
      automatically for such schedules; False forces the grid kernel.

  Returns:
    all-gather(x, axis=0) @ y
  """
    tp_size = mesh.shape[axis_name]
    validate_inputs(x, y, tp_size, rhs_transpose)
    m, k = x.shape
    if rhs_transpose:
        n, _ = y.shape
        y_in_spec = P(axis_name, None)
    else:
        _, n = y.shape
        y_in_spec = P(None, axis_name)
    m_per_device = m // tp_size
    n_per_device = n // tp_size
    tuned_bn, tuned_bk = (
        all_gather_matmul_tuned_block_sizes.get_tuned_block_sizes(
            m, n, k,
            jnp.dtype(x.dtype).name, tp_size))
    if bn is None:
        bn = tuned_bn if tuned_bn is not None else n_per_device
    if bk is None:
        bk = tuned_bk if tuned_bk is not None else k
    if bn > n_per_device:
        raise ValueError(
            f"bn ({bn}) must be <= n // tp_size ({n_per_device}): the kernel "
            "slices its per-device [k, n // tp_size] y block by bn, so a "
            "larger bn issues out-of-bounds DMA.")
    if bk > k:
        raise ValueError(f"bk ({bk}) must be <= k ({k}).")
    if bm is None:
        bm = m_per_device
    if bm != m_per_device:
        if bm <= 0 or bm % 8 != 0:
            raise ValueError(f"bm ({bm}) must be a positive multiple of 8.")
        if (m_per_device // 2) % bm != 0:
            raise ValueError(
                f"bm ({bm}) must divide m // tp_size // 2 "
                f"({m_per_device // 2}) so a row block stays on one side of "
                "the bidirectional split.")
    grid_m = m_per_device // bm
    grid_n = _cdiv(n_per_device, bn)
    grid_k = _cdiv(k, bk)
    if grid_n > 1 and (bn % 128 or n_per_device % bn):
        raise ValueError(
            f"bn ({bn}) must be a multiple of 128 and divide n // tp_size "
            f"({n_per_device}) when it blocks the n dimension: the block "
            "offset has to be tile-aligned and a ragged tail block reads and "
            "writes out of bounds. bn == n // tp_size has no such constraint.")
    single_block = grid_m == grid_n == grid_k == 1
    if unroll and not single_block:
        raise ValueError(
            "unroll=True requires the single-block schedule (bn == n // "
            f"tp_size, bk == k, bm == m // tp_size); got grid ({grid_m}, "
            f"{grid_n}, {grid_k}).")
    # Measured perf-neutral vs the grid kernel (bitwise-identical output), so
    # the proven grid path stays the default; the unrolled form is kept as the
    # simpler single-block schedule and the base for future ring variants.
    use_unroll = bool(unroll)
    # Compute-fold: a per-chunk dot below 128 rows starves the MXU at
    # ~rows/128 occupancy, so at m_per_device < 128 the kernel stacks `fold`
    # chunks per group buffer and dots them full-height at fold-boundary
    # outer steps. 128-row target — 256 concentrates compute into fewer
    # outers for no rate gain and loses 8-15% (fold sweep, v7x tp16).
    # Halved until it divides tp_size and the VMEM estimate fits the scoped
    # ceiling; fold == 1 keeps the original schedule byte-for-byte.
    if fold is None:
        fold = 1
        if bm == m_per_device and m_per_device < 128 and not use_unroll:
            fold = 128 // m_per_device
            while tp_size % fold:
                fold //= 2
            while fold > 1:
                acc_b = (fold * bm * bn * 4) if grid_k > 1 else 8 * 128 * 4
                if get_vmem_estimate_bytes(m, n, k, bn, acc_b, tp_size,
                                           x.dtype, y.dtype, x.dtype, bm=bm,
                                           fold=fold) <= _VMEM_CAP_BYTES:
                    break
                fold //= 2
    elif fold > 1 and (bm != m_per_device or tp_size % fold or use_unroll):
        raise ValueError(
            f"fold ({fold}) requires bm == m // tp_size ({m_per_device}), "
            f"fold | tp_size ({tp_size}) and the grid kernel.")
    acc_shape = (fold * bm, bn)
    # NOTE(chengjiyao): acc buffer is not used in the grid_k == 1 case.
    if grid_k == 1:
        acc_shape = (8, 128)
    acc_bytes = (acc_shape[0] * acc_shape[1] *
                 dtypes.itemsize_bits(jnp.float32)) // 8
    y_vmem_shape = (n_per_device, k) if rhs_transpose else (k, n_per_device)
    estimated_vmem_bytes = get_vmem_estimate_bytes(
        m,
        n,
        k,
        bn,
        acc_bytes,
        tp_size,
        x.dtype,
        y.dtype,
        x.dtype,
        bm=bm,
        fold=fold,
    )
    out_shape = [
        jax.ShapeDtypeStruct((m, n_per_device), x.dtype),  # output
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, k),
                             x.dtype),  # x HBM scratch
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
            pltpu.SemaphoreType.DMA,  # x_local_copy_sem
            pltpu.SemaphoreType.DMA,  # y_local_copy_sem
            pltpu.SemaphoreType.DMA,  # o_local_copy_sem
            pltpu.SemaphoreType.DMA(
                (2, tp_size - 1)),  # left and right send semaphores
            pltpu.SemaphoreType.DMA((
                2,
                tp_size - 1,
            )),  # left and right recv semaphores
            pltpu.VMEM((2, fold * bm, k), x.dtype),  # x vmem scratch
            pltpu.VMEM(y_vmem_shape, y.dtype),  # y vmem scratch
            pltpu.VMEM((2, fold * bm, bn), x.dtype),  # output vmem scratch
            pltpu.VMEM(acc_shape, jnp.float32),  # acc vmem scratch
        ),
        grid=((1, ) if use_unroll else
              (tp_size + 2, grid_m, grid_n, grid_k) if grid_m > 1 else
              (tp_size + 2, grid_n, grid_k)),
    )
    flops = 2 * m * k * n_per_device
    bytes_accessed = x.dtype.itemsize * (m * k + k * n_per_device +
                                         m * n_per_device)
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)

    if use_unroll:
        kernel_body = functools.partial(
            _all_gather_kernel_unrolled,
            axis_name=axis_name,
            rhs_transpose=rhs_transpose,
        )
    else:
        kernel_body = functools.partial(
            _all_gather_kernel,
            bn=bn,
            bk=bk,
            bm=bm,
            fold=fold,
            axis_name=axis_name,
            rhs_transpose=rhs_transpose,
        )
    kernel_name = get_kernel_name(bn, bk, bm, rhs_transpose)
    if use_unroll:
        kernel_name += "_unrolled"
    if fold > 1:
        kernel_name += f"_fold_{fold}"

    @jax.jit(static_argnames=["bn", "bk", "bm", "rhs_transpose"])
    def _all_gather_matmul_call(x, y, bn, bk, bm, rhs_transpose):
        return pl.pallas_call(
            kernel_body,
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                # fold == 1 keeps the exact historical limit; only the folded
                # stage can push the request past the scoped ceiling.
                vmem_limit_bytes=(min(estimated_vmem_bytes + 8 * 1024 * 1024,
                                      _VMEM_CAP_BYTES) if fold > 1 else
                                  estimated_vmem_bytes + 8 * 1024 * 1024),
            ),
            cost_estimate=cost_estimate,
            name=kernel_name,
        )(x, y)[0]

    shard_map_kernel = jax.jit(
        jax.shard_map(
            functools.partial(
                _all_gather_matmul_call,
                bn=bn,
                bk=bk,
                bm=bm,
                rhs_transpose=rhs_transpose,
            ),
            mesh=mesh,
            in_specs=(P(axis_name, None), y_in_spec),
            out_specs=P(None, axis_name),
            check_vma=False,
        ), )

    return shard_map_kernel(x, y)


def get_kernel_name(bn: int, bk: int, bm: int, rhs_transpose: bool):
    return (f"all_gather_matmul_kernel_bn_{bn}_bk_{bk}_bm_{bm}"
            f"_rhs_transpose_{rhs_transpose}")

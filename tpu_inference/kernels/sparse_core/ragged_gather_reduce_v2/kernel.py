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
"""SparseCore kernel body: gather / weighted segmented-reduce / scatter.

The kernel is a facade of top-level stages composed by ``functools.partial`` and
``emit_pipeline``: ``main_kernel`` stages this core's partition, ``_row_kernel``
runs one row-block (resolving the per-row scatter targets), and ``_col_kernel``
runs the weighted segmented reduction + scatter for one (sub-chunk, col-chunk).
Every stage reads the ``_Scratch`` pytree by attribute -- no positional unpacking.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import (config,
                                                                       memory_ref)


def _pack_scalars_to_vector(scalar_list: list[jax.Array],
                            num_simd_lanes: int) -> jax.Array:
    """Builds a lane vector from per-lane scalars.

    SparseCore cannot store individual scalars into VMEM lanes, so the vector is
    assembled with masked accumulation before being stored.
    """
    idx_vec = jnp.arange(num_simd_lanes)
    vec = jnp.zeros((num_simd_lanes, ), jnp.int32)
    for i in range(num_simd_lanes):
        vec += (idx_vec == i).astype(jnp.int32) * scalar_list[i]
    return vec


def _row_gather_spec(
    sorted_by_validity_vmem: jax.Array,
    sub: int,
    *,
    num_simd_lanes: int,
    row_chunk_size: int,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering sub-chunk ``sub``'s rows of a 1-D input."""
    return pl.BlockSpec(
        (pl.Indirect(num_simd_lanes), ),
        lambda i, s=sub: (sorted_by_validity_vmem[pl.ds(
            i * row_chunk_size + s * num_simd_lanes, num_simd_lanes)], ),
    )


def _col_gather_spec(
    src_indices_vmem: jax.Array,
    col_start: jax.Array,
    cfg: config._Config,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering ``col_chunk_size`` columns of the 2-D input.

    The row index is shifted by ``row_shift`` because ``x`` is gathered through a
    uint32 reinterpretation (bfloat16 packs two source rows per uint32 word).
    """
    return pl.BlockSpec(
        (pl.Indirect(cfg.num_simd_lanes), cfg.col_chunk_size),
        lambda s, c: (
            jnp.bitwise_right_shift(
                src_indices_vmem[pl.ds(s * cfg.num_simd_lanes, cfg.
                                       num_simd_lanes)],
                cfg.row_shift,
            ),
            col_start // cfg.col_chunk_size + c,
        ),
    )


def main_kernel(
    inputs: memory_ref._Inputs,
    out_hbm_ref: jax.Array,
    scratch: memory_ref._Scratch,
    *,
    cfg: config._Config,
):
    """Stages this core's partition into VMEM, then runs the row pipeline."""
    num_simd_lanes = cfg.num_simd_lanes

    # Step 1: Resolve this core's row/column partition and its column slice.
    core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
    row_partition_id = core_id // cfg.num_column_partitions
    col_partition_id = core_id % cfg.num_column_partitions

    row_partition_size_padded = (inputs.sorted_by_validity.shape[0] //
                                 cfg.num_row_partitions)
    row_start_padded = row_partition_id * row_partition_size_padded
    col_start = col_partition_id * cfg.col_size

    # Step 2: Stage this partition's row count and sort permutation into VMEM.
    recv_sem = scratch.sem.at[0]
    num_rows_dma = pltpu.make_async_copy(
        inputs.num_src_rows_per_row_partition.at[pl.ds(0, num_simd_lanes)],
        scratch.num_rows_per_row_partition_vmem,
        recv_sem,
    )
    sorted_dma = pltpu.make_async_copy(
        inputs.sorted_by_validity.at[pl.ds(row_start_padded,
                                           row_partition_size_padded)],
        scratch.sorted_by_validity_vmem,
        recv_sem,
    )
    num_rows_dma.start()
    sorted_dma.start()
    num_rows_dma.wait()
    sorted_dma.wait()

    num_rows_per_row_partition = scratch.num_rows_per_row_partition_vmem[...]
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    for i in range(cfg.num_row_partitions):
        num_rows_current_row_partition = jnp.where(
            row_partition_id == i,
            num_rows_per_row_partition[i],
            num_rows_current_row_partition,
        )
    num_row_blocks = pl.cdiv(num_rows_current_row_partition, cfg.row_chunk_size)

    # Step 3: Run the gather / weighted segmented-reduce / scatter pipeline.
    # Sentinel for the cross-block reduction carry (no previous group).
    scratch.prev_dst_row_smem[0] = -1

    # One gather per sub-chunk for ``indices``, then the same for
    # ``topk_weights``.
    row_gather_in_specs = tuple(
        _row_gather_spec(
            scratch.sorted_by_validity_vmem,
            sub,
            num_simd_lanes=num_simd_lanes,
            row_chunk_size=cfg.row_chunk_size,
        ) for sub in range(cfg.num_row_subchunks))

    row_pipeline = pltpu.emit_pipeline(
        functools.partial(
            _row_kernel,
            cfg=cfg,
            scratch=scratch,
            in_hbm_ref=inputs.x,
            out_hbm_ref=out_hbm_ref,
            num_rows_current_row_partition=num_rows_current_row_partition,
            col_start=col_start,
        ),
        grid=(num_row_blocks, ),
        in_specs=(row_gather_in_specs, row_gather_in_specs),
        out_specs=(),
    )
    row_pipeline(
        (inputs.indices, ) * cfg.num_row_subchunks,
        (inputs.topk_weights, ) * cfg.num_row_subchunks,
    )


def _row_kernel(
    src_indices_refs: tuple[jax.Array, ...],
    topk_weights_refs: tuple[jax.Array, ...],
    *,
    cfg: config._Config,
    scratch: memory_ref._Scratch,
    in_hbm_ref: jax.Array,
    out_hbm_ref: jax.Array,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
):
    """One row-block: stage indices/weights, resolve scatters, run col pipeline."""
    num_simd_lanes = cfg.num_simd_lanes
    row_chunk_size = cfg.row_chunk_size
    num_row_subchunks = cfg.num_row_subchunks

    # The SparseCore indirect DMA requires 32-bit elements, so x is gathered
    # through a uint32 reinterpretation. bfloat16 packs two source rows per
    # uint32 row (row index >> 1); float32 is 1:1 (row index unchanged).
    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)

    row_block_id = pl.program_id(0)

    # Destination output row of each source row in this block.
    dst_indices_list = [
        scratch.sorted_by_validity_vmem[pl.ds(
            row_block_id * row_chunk_size + s * num_simd_lanes,
            num_simd_lanes,
        )] // cfg.reduce_group_size for s in range(num_row_subchunks)
    ]

    # Stage the gathered indices/weights and the destinations in VMEM.
    for s in range(num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        scratch.src_indices_vmem[sub] = src_indices_refs[s][...]
        scratch.dst_indices_vmem[sub] = dst_indices_list[s]

        tw = topk_weights_refs[s][...]
        if cfg.topk_dtype == jnp.bfloat16:
            tw_f32 = plsc.bitcast(jnp.bitwise_left_shift(tw, 16), jnp.float32)
        else:
            tw_f32 = plsc.bitcast(tw, jnp.float32)
        scratch.tw_f32_vmem[sub] = tw_f32

    # For each sub-chunk, the destination of the row just before it -- the
    # seed for the segmented reduction's "same group as previous row" test.
    for s in range(num_row_subchunks):
        if s == 0:
            prev_dst = scratch.prev_dst_row_smem[0]
        else:
            prev_dst = dst_indices_list[s - 1][num_simd_lanes - 1]
        scratch.prev_dst_val_vmem[pl.ds(s * num_simd_lanes,
                                        num_simd_lanes)] = (jnp.broadcast_to(
                                            prev_dst, (num_simd_lanes, )))

    def get_dst_idx(global_idx):
        return dst_indices_list[global_idx //
                                num_simd_lanes][global_idx % num_simd_lanes]

    # For each source row, find the VMEM row that will hold its group's fully
    # reduced value -- the last row of the group within this block. Scanning
    # backwards, a row inherits its successor's merge target when they share
    # a destination, otherwise it is its own target.
    src_row_idx_in_vmem = []
    row_valid_vec = []
    for row_vmem_idx in reversed(range(row_chunk_size)):
        global_row_idx = row_block_id * row_chunk_size + row_vmem_idx
        row_valid_vec.append(global_row_idx < num_rows_current_row_partition)
        if row_vmem_idx == row_chunk_size - 1:
            src_row_idx_in_vmem.append(row_vmem_idx)
        else:
            same_group_as_next = jnp.logical_and(
                row_valid_vec[-2],
                get_dst_idx(row_vmem_idx) == get_dst_idx(row_vmem_idx + 1),
            ).astype(jnp.int32)
            src_row_idx_in_vmem.append(
                same_group_as_next * src_row_idx_in_vmem[-1] +
                (1 - same_group_as_next) * row_vmem_idx)
    src_row_idx_in_vmem.reverse()
    row_valid_vec.reverse()

    # Per source row, the (VMEM source row, HBM destination row) of its
    # scatter. Rows whose group is not yet fully reduced in this sub-chunk,
    # and padding rows, are routed to a throwaway row.
    garbage_dst = out_hbm_ref.shape[0] - 1
    dma_src_rows = []
    dma_dst_rows = []
    for s in range(num_row_subchunks):
        sub_src = []
        sub_dst = []
        for i in range(num_simd_lanes):
            global_idx = s * num_simd_lanes + i
            merge_target = src_row_idx_in_vmem[global_idx]
            is_final_write = jnp.logical_and(
                row_valid_vec[global_idx],
                merge_target < (s + 1) * num_simd_lanes,
            )
            sub_src.append(
                jnp.where(is_final_write, merge_target % num_simd_lanes, 0))
            sub_dst.append(
                jnp.where(is_final_write, dst_indices_list[s][i], garbage_dst))
        dma_src_rows.append(sub_src)
        dma_dst_rows.append(sub_dst)

    for s in range(num_row_subchunks):
        sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
        scratch.dma_src_row_vmem[sub] = _pack_scalars_to_vector(
            dma_src_rows[s], num_simd_lanes)
        scratch.dma_dst_row_vmem[sub] = _pack_scalars_to_vector(
            dma_dst_rows[s], num_simd_lanes)

    col_pipeline = pltpu.emit_pipeline(
        functools.partial(
            _col_kernel,
            cfg=cfg,
            scratch=scratch,
            out_hbm_ref=out_hbm_ref,
            col_start=col_start,
        ),
        grid=(num_row_subchunks, cfg.num_col_chunks),
        in_specs=_col_gather_spec(scratch.src_indices_vmem, col_start, cfg),
        out_specs=(),
    )
    col_pipeline(in_32b_hbm_ref)
    scratch.prev_dst_row_smem[0] = dst_indices_list[-1][num_simd_lanes - 1]


def _col_kernel(
    gather_ref: jax.Array,
    *,
    cfg: config._Config,
    scratch: memory_ref._Scratch,
    out_hbm_ref: jax.Array,
    col_start: jax.Array,
):
    """One (sub-chunk, col-chunk): weighted segmented-reduce, then scatter."""
    num_simd_lanes = cfg.num_simd_lanes
    col_chunk_size = cfg.col_chunk_size

    s = pl.program_id(0)
    c = pl.program_id(1)
    col_hbm_start = col_start + c * col_chunk_size
    send_sem = scratch.sem.at[1]

    row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
    tw_slice = scratch.tw_f32_vmem[row_slice]
    dst_slice = scratch.dst_indices_vmem[row_slice]
    src_idx_slice = scratch.src_indices_vmem[row_slice]
    prev_dst_vals_vec = scratch.prev_dst_val_vmem[row_slice]

    def col_loop(col_compute_offset):
        col_slice = pl.ds(col_compute_offset, num_simd_lanes)
        # Running sum, seeded by the carry from the previous sub-chunk.
        previous_accumulated_data = scratch.prev_iter_last_row_vmem[c,
                                                                    col_slice]

        for row_src in range(num_simd_lanes):
            val_u32 = gather_ref[row_src, col_slice]
            if cfg.in_dtype == jnp.bfloat16:
                # The two bfloat16 rows packed in one uint32 word sit in the low
                # (even row) or high (odd row) 16 bits. Shift the wanted half
                # into the float32 sign/exponent position and clear the rest.
                shift = jnp.where(
                    jnp.bitwise_and(src_idx_slice[row_src], 1) == 0, 16, 0)
                shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift),
                                          jnp.uint32(0xFFFF0000))
                data_f32 = plsc.bitcast(shifted, jnp.float32)
            else:
                data_f32 = plsc.bitcast(val_u32, jnp.float32)
            data_f32 *= tw_slice[row_src]

            # Reduction: accumulate while the destination group is unchanged,
            # restart otherwise. Sorting guarantees rows of one group are
            # contiguous.
            dst_row_hbm = dst_slice[row_src]
            if row_src == 0:
                prev_dst = prev_dst_vals_vec[0]
            else:
                prev_dst = dst_slice[row_src - 1]
            accumulated_data = jnp.where(
                dst_row_hbm == prev_dst,
                previous_accumulated_data + data_f32,
                data_f32,
            )
            previous_accumulated_data = accumulated_data

            # The output buffer stays float32: a bfloat16 output would be
            # (16, 128)-tiled and the per-row scatter below writes a single
            # row at an arbitrary, non-tile-aligned destination, which is only
            # legal for 32-bit elements. The cast happens in the wrapper.
            scratch.out_vmem[row_src, col_slice] = accumulated_data
            if row_src == num_simd_lanes - 1:
                scratch.prev_iter_last_row_vmem[c,
                                                col_slice] = accumulated_data

    plsc.parallel_loop(0, col_chunk_size, step=num_simd_lanes)(col_loop)

    # Scatter every source row's reduced value to its output row. Rows
    # that share a group write the same value (idempotent); rows routed to
    # the garbage destination are harmless.
    dma_src_row_slice = scratch.dma_src_row_vmem[row_slice]
    dma_dst_row_slice = scratch.dma_dst_row_vmem[row_slice]
    copies = []
    for i in range(num_simd_lanes):
        copy = pltpu.make_async_copy(
            scratch.out_vmem.at[dma_src_row_slice[i],
                                pl.ds(0, col_chunk_size)],
            out_hbm_ref.at[dma_dst_row_slice[i],
                           pl.ds(col_hbm_start, col_chunk_size)],
            send_sem,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()

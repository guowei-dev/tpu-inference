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
"""Public entry: dispatch, config, preprocessing, and the SparseCore launch."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import core_map_helper
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import (config,
                                                                       kernel,
                                                                       memory_ref)


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


def _preprocess(
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_row_partitions: int,
    num_simd_lanes: int,
    row_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sorts valid source rows to the front of each row partition.

    Returns:
      sorted_by_validity: original row index of each slot after the stable
        sort, flattened across partitions and padded to ``row_chunk_size``.
      num_src_rows_per_row_partition: valid row count per partition, padded to
        ``num_simd_lanes`` so the kernel can load it as a single vector.
      mask: per output group, whether the group has any valid source row.
    """
    row_partition_size = valid_rows_mask.shape[0] // num_row_partitions
    valid_rows_mask_2d = valid_rows_mask.reshape(num_row_partitions, -1)

    # Stable sort of a boolean key is a stable partition: valid rows keep their
    # relative order and move ahead of the invalid ones.
    sorted_by_validity = jnp.argsort(~valid_rows_mask_2d,
                                     descending=False,
                                     stable=True,
                                     axis=-1)
    sorted_by_validity += (jnp.arange(num_row_partitions)[:, None] *
                           row_partition_size)

    pad_to = config._align_to(row_partition_size, row_chunk_size)
    if pad_to > row_partition_size:
        sorted_by_validity = jnp.pad(
            sorted_by_validity,
            ((0, 0), (0, pad_to - row_partition_size)),
            constant_values=0,
        )
    sorted_by_validity = sorted_by_validity.reshape(-1)

    num_src_rows_per_row_partition = jnp.pad(
        jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
        (0, max(0, num_simd_lanes - num_row_partitions)),
    )
    mask = jnp.any(valid_rows_mask.reshape(-1, reduce_group_size), axis=-1)
    return (
        sorted_by_validity.astype(jnp.int32),
        num_src_rows_per_row_partition,
        mask,
    )


@functools.partial(jax.jit, static_argnames=("reduce_group_size", ))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

    Args:
      x: 2-D input features, ``(num_rows, hidden_size)``.
      indices: 1-D gather indices, ``(input_size,)``.
      topk_weights: 1-D per-row weights, ``(input_size,)``.
      valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
      reduce_group_size: number of consecutive rows summed into one output row.

    Returns:
      Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
    """
    # Step 1: Choose the implementation (TensorCore fallback or SparseCore).
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return _fallback_implementation(x, indices, topk_weights,
                                        valid_rows_mask, reduce_group_size)

    # For a small {input + output} both likely fit in TensorCore VMEM, where a
    # plain TC gather-reduce beats routing through SparseCore and HBM. This
    # also keeps the kernel off configs with num_row_partitions > num_simd_lanes.
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if (jnp.size(x) * dtype_bytes * 2
            < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6):
        return _fallback_implementation(x, indices, topk_weights,
                                        valid_rows_mask, reduce_group_size)

    # Step 2: Derive the kernel configuration (core grid and column tiling).
    hidden_size = x.shape[-1]
    input_size = indices.size
    num_simd_lanes = sc_info.num_lanes
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores

    num_column_partitions = config._calculate_num_column_partitions(
        hidden_size, input_size, num_cores, num_lanes, num_simd_lanes)
    num_row_partitions = num_cores // num_column_partitions
    assert (num_row_partitions <= num_simd_lanes
            ), f"{num_row_partitions=} must be <= {num_simd_lanes=}"
    num_row_subchunks, row_chunk_size = config._calculate_row_tiling(
        input_size, num_simd_lanes, num_row_partitions)

    aligned_hidden_size = config._align_to(hidden_size,
                                           128 * num_column_partitions)
    col_size = aligned_hidden_size // num_column_partitions
    col_chunk_size = config._calculate_col_chunk_size(col_size, num_simd_lanes)

    # Step 3: Pre-process inputs (weights, padding, sort by validity).
    # The kernel gathers x through a uint32 reinterpretation; carry the weights
    # the same way so they can be bitcast back to float32 on SparseCore.
    if topk_weights.dtype == jnp.bfloat16:
        topk_weights_u32 = jax.lax.bitcast_convert_type(
            topk_weights, jnp.uint16).astype(jnp.uint32)
    else:
        topk_weights_u32 = jax.lax.bitcast_convert_type(
            topk_weights, jnp.uint32)

    # Pad the input so each row partition holds a whole number of reduce
    # groups; no group is then split across two physical cores.
    padded_input_size = config._align_to(input_size,
                                         num_row_partitions * reduce_group_size)
    valid_rows_mask = jnp.pad(
        valid_rows_mask,
        (0, padded_input_size - input_size),
        constant_values=False,
    )

    sorted_by_validity, num_src_rows_per_row_partition, mask = _preprocess(
        valid_rows_mask,
        reduce_group_size,
        num_row_partitions,
        num_simd_lanes,
        row_chunk_size,
    )

    # Step 4: Launch the SparseCore kernel.
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    cfg = config._Config(
        num_row_partitions=num_row_partitions,
        num_column_partitions=num_column_partitions,
        reduce_group_size=reduce_group_size,
        col_size=col_size,
        col_chunk_size=col_chunk_size,
        num_row_subchunks=num_row_subchunks,
        num_simd_lanes=num_simd_lanes,
        topk_dtype=topk_weights.dtype,
        in_dtype=x.dtype,
        core_axis_name=vector_mesh.core_axis_name,
        subcore_axis_name=vector_mesh.subcore_axis_name,
    )

    row_partition_size_padded = sorted_by_validity.size // num_row_partitions

    # The output gets one extra row: the kernel's garbage scatter destination.
    out = core_map_helper.kernel(
        functools.partial(kernel.main_kernel, cfg=cfg),
        out_type=jax.ShapeDtypeStruct(
            (padded_input_size // reduce_group_size + 1, aligned_hidden_size),
            jnp.float32,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(memory_ref._Scratch.create(cfg,
                                                  row_partition_size_padded), ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v2",
    )(memory_ref._Inputs(
        num_src_rows_per_row_partition=num_src_rows_per_row_partition,
        x=x,
        indices=indices,
        topk_weights=topk_weights_u32,
        sorted_by_validity=sorted_by_validity,
    ), )

    # Step 5: Post-process the output (drop padding, zero empty groups, cast).
    out = out[:input_size // reduce_group_size, :hidden_size]
    out = jnp.where(mask[:input_size // reduce_group_size, None], out,
                    jnp.zeros_like(out))
    return out.astype(x.dtype)

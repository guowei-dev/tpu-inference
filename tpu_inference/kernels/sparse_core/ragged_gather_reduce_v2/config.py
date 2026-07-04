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
"""Kernel configuration and tiling/partitioning cost model."""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclasses.dataclass(frozen=True)
class _Config:
    num_row_partitions: int
    num_column_partitions: int
    reduce_group_size: int
    col_size: int
    col_chunk_size: int
    num_row_subchunks: int
    num_simd_lanes: int
    topk_dtype: Any
    in_dtype: Any
    core_axis_name: str
    subcore_axis_name: str

    @property
    def row_chunk_size(self) -> int:
        """Number of rows handled per row-pipeline block."""
        return self.num_simd_lanes * self.num_row_subchunks

    @property
    def num_col_chunks(self) -> int:
        return self.col_size // self.col_chunk_size

    @property
    def row_shift(self) -> int:
        """log2 of how many source rows pack into one uint32 gather element.

        The SparseCore indirect DMA requires 32-bit elements: bfloat16 packs two
        source rows per uint32 (shift 1), float32 is 1:1 (shift 0).
        """
        input_packing = 32 // jax.dtypes.itemsize_bits(self.in_dtype)
        return input_packing.bit_length() - 1


class _CostModelConstants:
    # Limit on the number of outer loop pipeline iterations. Too many iterations
    # cause high cumulative pipeline overhead (e.g., from frequent pipeline
    # startup/teardown bubbles). We try to find partitioning that does not exceed
    # this limit on iterations.
    MAX_ITERATIONS: int = 40

    # Upper cap on the column chunk size processed per inner pipeline step.
    # While larger chunk sizes help utilize bandwidth better, excessively large
    # chunk sizes cause large pipeline bubbles. We cap it here to balance
    # efficiency and bubble sizes.
    MAX_COL_CHUNK_SIZE: int = 1024


# ceil up to the nearest multiple of b.
def _align_to(a, b):
    return pl.cdiv(a, b) * b


def _calculate_num_column_partitions(hidden_size: int, input_size: int,
                                     num_cores: int, num_lanes: int,
                                     num_simd_lanes: int) -> int:
    """Calculates the number of row partitions."""
    # Each column partition should be multiple of 128 (number of lanes) due to
    # DMA requirements.
    # Prefer to use a large number of column partitions, as long as each
    # partition's size is not too small for DMA pipeline efficiency and each
    # partition's size can divide the hidden size.

    # Each column partition will do DMA pipelining on col_size.
    preferred_num_stages = 4
    num_column_partitions = 1
    while (num_cores % (num_column_partitions * 2) == 0
           and hidden_size % (num_lanes * num_column_partitions * 2) == 0
           and hidden_size //
           (num_column_partitions * 2 * num_lanes) >= preferred_num_stages):
        next_candidate = num_column_partitions * 2
        next_row_partitions = num_cores // next_candidate

        # Calculate exactly how many pipeline invocations (outer loop)
        num_row_subchunks, row_chunk_size = _calculate_row_tiling(
            input_size, num_simd_lanes, next_row_partitions)
        num_iterations = input_size // (row_chunk_size * next_row_partitions)

        # Ensure we satisfy the hardware constraint (num_row_partitions <= num_simd_lanes) first.
        if num_cores // num_column_partitions > num_simd_lanes:
            num_column_partitions = next_candidate
            continue

        # Too many iterations cause high cumulative pipeline overhead. Set the
        # limit based on empirical data.
        if num_iterations > _CostModelConstants.MAX_ITERATIONS:
            break

        num_column_partitions = next_candidate

    return num_column_partitions


def _calculate_row_tiling(
    input_size: int,
    num_simd_lanes: int,
    num_row_partitions: int,
) -> tuple[int, int]:
    """Calculates the number of row subchunks and row chunk size."""
    base_block_size = num_simd_lanes * num_row_partitions
    num_row_subchunks = max(1, min(4, pl.cdiv(input_size, base_block_size)))
    row_chunk_size = num_simd_lanes * num_row_subchunks
    return num_row_subchunks, row_chunk_size


def _calculate_col_chunk_size(col_size: int, num_simd_lanes: int) -> int:
    """Picks the column chunk size the inner pipeline gathers at a time.

    The chunk is the largest divisor of ``col_size`` whose gather double-buffer
    still fits comfortably in SparseCore VMEM.
    """
    generation = pltpu.get_tpu_info().generation
    match generation:
        case 6:
            target_bytes = int(256 * 1024 * 0.95)
        case 7:
            target_bytes = int(512 * 1024 * 0.95)
        case _:
            target_bytes = int(128 * 1024 * 0.95)

    # uint32 gather buffer, double-buffered by emit_pipeline.
    bytes_per_col = num_simd_lanes * 4 * 2
    max_safe_col = (target_bytes // bytes_per_col // 128) * 128

    # Larger chunk sizes cause larger pipeline bubbles, so cap it at 1024.
    max_safe_col = min(max_safe_col, _CostModelConstants.MAX_COL_CHUNK_SIZE)

    start_col = (min(col_size, max_safe_col) // 128) * 128
    for chunk in range(start_col, 127, -128):
        if col_size % chunk == 0:
            return chunk
    return 128

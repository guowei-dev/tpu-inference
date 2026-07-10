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

import dataclasses
from typing import Any

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IndexRef:
    num_src_rows_per_row_partition: Any
    indices: Any
    sorted_by_validity: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScratchRef:
    num_rows_per_row_partition_vmem: Any
    prev_iter_last_row_vmem: Any
    prev_dst_row_smem: Any
    sorted_by_validity_vmem: Any
    src_indices_vmem: Any
    dst_indices_vmem: Any
    tw_f32_vmem: Any
    dma_src_row_vmem: Any
    dma_dst_row_vmem: Any
    prev_dst_val_vmem: Any
    out_vmem: Any
    sem: Any

    @classmethod
    def create_scratch_types(cls, cfg: config.Config) -> Any:
        num_simd_lanes = cfg.sc_info.num_lanes
        indices_vmem = pltpu.VMEM((cfg.row_chunk_size, ), jnp.int32)
        return cls(
            num_rows_per_row_partition_vmem=pltpu.VMEM((num_simd_lanes, ),
                                                       jnp.int32),
            prev_iter_last_row_vmem=pltpu.VMEM(
                (cfg.col_size // cfg.col_chunk_size, cfg.col_chunk_size),
                jnp.float32,
            ),
            prev_dst_row_smem=pltpu.SMEM((1, ), jnp.int32),
            sorted_by_validity_vmem=pltpu.VMEM(
                (cfg.row_partition_size_padded, ), jnp.int32),
            src_indices_vmem=indices_vmem,
            dst_indices_vmem=indices_vmem,
            dma_src_row_vmem=indices_vmem,
            dma_dst_row_vmem=indices_vmem,
            prev_dst_val_vmem=indices_vmem,
            tw_f32_vmem=pltpu.VMEM((cfg.row_chunk_size, ), jnp.float32),
            out_vmem=pltpu.VMEM((num_simd_lanes, cfg.col_chunk_size),
                                jnp.float32),
            sem=pltpu.SemaphoreType.DMA((2, )),
        )

# SPDX-License-Identifier: Apache-2.0
"""Fused permute + grouped matmul.

Reads the LHS per-row from an un-permuted pool inside the kernel, so the MoE
dispatch permute never materialises. Same contract and bitwise-identical
results to `gmm_v2(lhs[gather_indices], ...)` over the shard window.

This is a rewrite of `gmm_v2`'s `gather_indices=` path around three
measurements (`dev_nexus/project/moe-fuse-permute-gmm/artifact/fpg_lab`):

  * The tax is PER-GM-TILE, not per launch. `num_gm` == the shard's local
    expert count (64) independently of T, and the tax fits
    `num_gm * (0.707 us + 6.42 ns * live_rows_per_tile)` -- 71% of it fixed
    per tile at the production point.
  * `calculate_tiling` grows `tile_m` to fill VMEM, but the shard's rows are
    spread over the expert groups, so a 512-row tile carries ~47 live rows at
    T=2048 (9% fill) while the extract and the issue/wait sites are all sized
    by `tile_m`. Matching `tile_m` to the live-row count is worth -39% of the
    tax with no kernel change at all.
  * The scalar/MXU co-issue is a CEILING problem here, not a scheduling one: a
    DMA-issue bundle can only share a VLIW bundle with a `vmat*` in the same
    region, and the old body put the issues in 32 `pl.when` regions and the
    dots in a `lax.switch`/`lax.cond` nest -- 321 regions, so at most 0.96% of
    the issues could ever co-issue. It realised 20.8% of that ceiling, i.e.
    the scheduler was already doing as well as on a kernel that reaches 21.5%.

so this kernel keeps the body straight-line and the issue chain in it:
`tile_m` is chosen from the live-row count (which makes the bucket switch
single-branch), the index table is a kernel-invariant SMEM operand rather than
a DMA-streamed window (no semaphore barrier in front of the address chain),
and the steady body is dispatched on `s % num_slots` so every slot index is a
Python constant (a traced `s % nb` hides the slots' disjointness from the
scheduler, which then orders the whole issue chain ahead of the wait).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm_v2 import (
    FusedWeightsRef,
    GmmConfigs,
    MetadataRef,
    TileSizes,
    WeightsRef,
    align_to,
    fill_metadata,
    generate_block_specs,
    get_cost_estimate,
    get_metadata,
    get_scope_name,
    inner_kernel,
    make_gmm_configs,
    zero_out_end,
    zero_out_start,
)

# v7x SMEM is 1 MiB and a scalar-prefetch operand cliffs well below it; the
# invariant index table costs 4 bytes per row.
SMEM_INDEX_LIMIT_BYTES = 480_000


def gather_tiling(size_m: int, size_group: int) -> int:
    """tile_m for the gather path.

    `calculate_tiling` maximises tile_m to fill VMEM, which is right for a
    dense LHS block and wrong here: rows are spread over `size_group` expert
    groups, so a gm tile only ever carries ~size_m/size_group live rows and
    everything sized by tile_m (the extract buffer, the per-row issue and wait
    sites) is paid in full regardless. Round the live-row count up to the
    128-row bucket granularity the matmul already selects.

    64 is allowed here. It is rejected by `gmm_v2`'s gather path, but for a
    reason that does not apply to this kernel: that path copies a
    `tile_m + 128` element index window into SMEM, and 192 is not a multiple
    of the 128-element lane tile ("Slice sizes along tiled dimensions must be
    aligned to tiles"). This kernel has no index window.
    """
    live = -(-size_m // max(size_group, 1))
    return int(min(512, max(64, align_to(live, 64))))


def _fused_permute_gmm_inner(
    # In (pipelined)
    tiled_rhs_ref,
    # Out (pipelined)
    tiled_out_ref,
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref: MetadataRef,
    gather_buf,  # [num_slots, tile_m, tile_k] int32 (bf16 pool) / lhs dtype
    gather_sem,  # DMA[num_slots]
    parity_buf,  # [num_slots, tile_m, 1] int32
    parity_sem,  # DMA[num_slots]
    *,
    cfgs: GmmConfigs,
    idx_smem,  # closed over: int32[padded_m] SMEM, kernel-invariant
    lhs_pool,  # closed over: [size_src(/packing), size_k] HBM
    parity_hbm,  # closed over: int32[padded_m, 1] HBM
    packing: int,
    num_slots: int,
):
    """Pipeline body. rhs/out stay pipelined; the LHS tile is gathered."""
    tile_m = cfgs.tiles.tile_m
    tile_k = cfgs.tiles.tile_k
    sublane = cfgs.dims.size_lhs_sublane
    shift = packing.bit_length() - 1  # bf16 pool: pool row = idx >> 1

    num_gm = pl.num_programs(1)
    num_k = pl.num_programs(2)
    s = (pl.program_id(0) * num_gm + pl.program_id(1)) * num_k + \
        pl.program_id(2)
    num_steps = pl.num_programs(0) * num_gm * num_k

    def m_base(step):
        """Sublane-aligned base row of `step`'s gm tile, and its live count.

        Mirrors inner_kernel's m_offset/m_end_local, so gathered row j is the
        row the contiguous path's tiled_lhs_ref[j] holds.
        """
        gm = lax.rem(step // num_k, num_gm)
        m_start = metadata_ref.gm_id_to_m_offset[gm]
        m_end = metadata_ref.gm_id_to_m_offset[gm + 1]
        m_offset = m_start - m_start % sublane
        return m_offset, m_end - m_offset

    def start_parity(step, slot):
        m_offset, _ = m_base(step)
        pltpu.make_async_copy(parity_hbm.at[pl.ds(m_offset, tile_m)],
                              parity_buf.at[slot], parity_sem.at[slot]).start()

    def wait_parity(slot):
        pltpu.make_async_copy(parity_hbm.at[pl.ds(0, tile_m)],
                              parity_buf.at[slot], parity_sem.at[slot]).wait()

    def start_rows(step, slot):
        """Issue the tile's row DMAs.

        Branch-free: the row count rides in the DMA *size* (0 or 1 rows), so a
        dead row moves no bytes and needs no `pl.when` region. That matters
        because a region boundary is exactly what stops an issue bundle from
        sharing a VLIW bundle with the matmul.

        Addresses come from the kernel-invariant SMEM table, so no semaphore
        wait sits in front of this scalar chain.
        """
        m_offset, live = m_base(step)
        k_base = lax.rem(step, num_k) * tile_k
        for j in range(tile_m):
            n = jnp.clip(live - j, 0, 1)
            row = idx_smem[m_offset + j]
            if packing > 1:
                row = row >> shift
            pltpu.make_async_copy(
                lhs_pool.at[pl.ds(row, n), pl.ds(k_base, tile_k)],
                gather_buf.at[slot, pl.ds(j, n)],
                gather_sem.at[slot],
            ).start()

    def wait_rows(step, slot):
        _, live = m_base(step)
        # One aggregate wait for the whole tile: same byte count as the
        # per-row waits, and measured bit-identical in schedule.
        pltpu.make_async_copy(
            lhs_pool.at[pl.ds(0, live), pl.ds(0, tile_k)],
            gather_buf.at[slot, pl.ds(0, live)],
            gather_sem.at[slot],
        ).wait()

    def extract(slot):
        """u32 row -> the bf16 row the index selected.

        The pool is viewed as int32, so word r packs bf16 rows 2r (low half)
        and 2r+1 (high half); shift the selected half up and mask.
        """
        if packing == 1:
            return gather_buf[slot]
        sh = 16 * (1 - jnp.bitwise_and(parity_buf[slot], 1))
        bits = jnp.bitwise_and(jnp.left_shift(gather_buf[slot], sh),
                               jnp.int32(-65536))
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(
            cfgs.lhs_cfgs.dtype)

    def run(slot):
        """One grid step, with `slot` a PYTHON int."""
        nxt = (slot + 1) % num_slots

        @pl.when(s + 1 < num_steps)
        def _():
            # Issue-first: feed the engine for the next tile before this
            # tile's data is touched.
            start_rows(s + 1, nxt)
            start_parity(s + 1, nxt)

        wait_rows(s, slot)
        if packing > 1:
            wait_parity(slot)
        inner_kernel(extract(slot), tiled_rhs_ref, tiled_out_ref,
                     partial_out_ref, acc_ref, metadata_ref, cfgs=cfgs)

    @pl.when(s == 0)
    def _():
        start_rows(0, 0)
        start_parity(0, 0)

    for slot in range(num_slots):
        # Dispatching on the parity makes every slot index inside the branch a
        # Python constant. A traced `s % num_slots` folds a runtime
        # multiply/add into every VMEM destination and semaphore address, and
        # hides the slots' disjointness from the scheduler.
        @pl.when(lax.rem(s, num_slots) == slot)
        def _(slot=slot):
            run(slot)


def kernel_main_fpg(
    # Scalar prefetch
    lhs_group_sizes_ref,
    group_offset_ref,
    idx_smem,
    # In
    lhs_ref,
    parity_hbm,
    rhs_ref: WeightsRef,
    # Out
    out_ref,
    # Scratch
    partial_out_ref,
    acc_ref,
    metadata_ref: MetadataRef,
    gather_buf,
    gather_sem,
    parity_buf,
    parity_sem,
    zero_ref,
    semaphore_ref,
    *,
    cfgs: GmmConfigs,
    packing: int,
    num_slots: int,
):
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref,
                           metadata_ref, cfgs=cfgs)

    if cfgs.zero_init:
        zero_size = zero_out_start(out_ref, zero_ref, semaphore_ref,
                                   metadata_ref, num_gm, dims=cfgs.dims)

    (_, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs,
                                                   rhs_buffer_count=2)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:],
                                  rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_spec = FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    lhs_pool = lhs_ref.bitcast(jnp.int32) if packing > 1 else lhs_ref
    body = functools.partial(_fused_permute_gmm_inner, cfgs=cfgs,
                             idx_smem=idx_smem, lhs_pool=lhs_pool,
                             parity_hbm=parity_hbm, packing=packing,
                             num_slots=num_slots)

    pipeline_fn = pltpu.emit_pipeline(body, grid=(num_n, num_gm, num_k),
                                      in_specs=(rhs_spec, ),
                                      out_specs=out_spec)

    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    pipeline_fn(rhs_ref, out_in,
                scratches=[partial_out_ref, acc_ref, metadata_ref, gather_buf,
                           gather_sem, parity_buf, parity_sem])

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


@functools.partial(jax.jit, static_argnames=[
    "tile_info", "vmem_limit_bytes", "precision", "preferred_element_type",
    "acc_dtype", "maybe_quantize_lhs", "zero_initialize", "fuse_act",
    "num_slots",
])
def fused_permute_gmm(
    lhs: jax.Array,  # [size_src, size_k] un-permuted pool
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    gather_indices: jax.Array,  # int32[size_m]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    tile_info=None,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    num_slots: int = 2,
) -> jax.Array:
    """Grouped matmul over `lhs[gather_indices]` without materialising it.

    Row `i` of the effective LHS is `lhs[gather_indices[i]]`; the output is
    [size_m, size_n] with size_m = gather_indices.shape[0].
    """
    if lhs.dtype.itemsize not in (2, 4):
        raise NotImplementedError(
            f"pool dtype {lhs.dtype} unsupported: the row read is an int32 "
            "view, so only 2- and 4-byte pools are implemented")
    if gather_indices.ndim != 1:
        raise ValueError("gather_indices must be 1-D")

    size_m = gather_indices.shape[0]
    if size_m * 4 > SMEM_INDEX_LIMIT_BYTES:
        raise NotImplementedError(
            f"{size_m} indices exceed the SMEM scalar-operand budget "
            f"({SMEM_INDEX_LIMIT_BYTES} B); use gmm_v2(gather_indices=)")

    packing = 4 // lhs.dtype.itemsize
    cfg_lhs = jax.ShapeDtypeStruct((size_m, lhs.shape[1]), lhs.dtype)
    if tile_info is None:
        tile_m = gather_tiling(size_m, group_sizes.shape[0])
        tile_info = TileSizes(tile_m=tile_m, tile_k=lhs.shape[1],
                              tile_n=align_to(rhs.shape[-1] //
                                              (2 if fuse_act else 1), 128),
                              bucket_base=tile_m)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(
            pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    cfgs = make_gmm_configs(
        cfg_lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset,
        tile_info=tile_info, vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type, acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize, fuse_act=fuse_act)

    dims, tiles = cfgs.dims, cfgs.tiles
    if group_offset is None:
        group_offset = jnp.zeros((1, ), jnp.int32)

    pool = lhs
    if packing > 1 and pool.shape[0] % packing:
        pool = jnp.pad(pool, ((0, packing - pool.shape[0] % packing), (0, 0)))

    # Pad the index table so a tile that runs past the last live row still
    # names a valid pool row (bounds checks are off).
    pad_to = align_to(dims.size_m, tiles.tile_m) + tiles.tile_m
    idx = jnp.pad(gather_indices.astype(jnp.int32), (0, pad_to - size_m),
                  mode="edge")

    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    # Scratch layout is gmm_v2's, verbatim, so inner_kernel and zero_out_*
    # keep their contract; the gather buffers are spliced in ahead of the
    # zero-init pair, which must stay last.
    num_lanes = pltpu.get_tpu_info().num_lanes
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        pltpu.VMEM((num_slots, tiles.tile_m, tiles.tile_k),
                   jnp.int32 if packing > 1 else lhs.dtype),
        pltpu.SemaphoreType.DMA((num_slots, )),
        pltpu.VMEM((num_slots, tiles.tile_m, 1), jnp.int32),
        pltpu.SemaphoreType.DMA((num_slots, )),
    ]
    if cfgs.zero_init:
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = min(2 * 1024 * 1024 // num_lanes // out_bytes,
                          dims.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1, )),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)

    return pl.pallas_call(
        functools.partial(kernel_main_fpg, cfgs=cfgs, packing=packing,
                          num_slots=num_slots),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(weight=pl.BlockSpec(memory_space=pltpu.HBM),
                           scale=rhs_scale_spec, bias=rhs_bias_spec),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs) + "-fpg",
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, idx, pool, idx[:, None],
      rhs_weights)[:, :cfgs.out_size_n]

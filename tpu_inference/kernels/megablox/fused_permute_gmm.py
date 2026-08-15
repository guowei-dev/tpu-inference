# SPDX-License-Identifier: Apache-2.0
"""Fused permute + grouped matmul.

Reads the LHS per-row from an un-permuted pool inside the kernel, so the MoE
dispatch permute never materialises. Bitwise-identical to
`gmm_v2(lhs[gather_indices], ...)` over the EP shard window.

A rewrite of `gmm_v2`'s `gather_indices=` path around four measurements
(`dev_nexus/project/moe-fuse-permute-gmm/artifact/fpg_lab`); at the production
regime (qwen397 EP8, T=2048) it moves the fused path from 73.9% of the pure-GMM
floor to 95.7%.

  * **The tax is per-gm-tile, not per launch.** `num_gm` == the shard's local
    expert count (64) independently of T, and the tax fits
    `num_gm * (0.707 us + 6.42 ns * live_rows_per_tile)` -- 71% of it fixed per
    tile. So `tile_m` is chosen from the live-row count (`gather_tiling`,
    ~`size_m / size_lhs_group`), not to fill VMEM: a 512-row tile carries ~47
    live rows here, and the extract and the per-row issue/wait sites are all
    sized by `tile_m` regardless. Worth -39% of the tax on its own.
  * **The index table is a kernel-invariant SMEM operand**, not a DMA-streamed
    window: no semaphore barrier in front of the address chain, and no
    `tile_m + 128` window copy (which is also what made `tile_m=64` illegal in
    the old path -- 192 is not a multiple of the 128-element lane tile).
  * **The steady body is emitted ONCE with a traced slot** (`traced_slots=`,
    default on). The parity-dispatched form (every slot index a Python
    constant) was this kernel's first answer and measured better on the
    original chassis; on the migrated chassis (rhs_buffers=3) the halved
    body wins -- traced+3-buffer is -3.5 us at T=2048 while constant slots
    alone are -1.4 (ledger fpg-tslot; the parent project's round 14 found
    the same inversion on its loop chassis). `traced_slots=False` keeps the
    parity form.
  * **The weight stream runs three deep** (`rhs_buffers=`, default 3 --
    `generate_block_specs`' own default). The inherited gather path dropped
    it to 2 to make room for a 16 MiB gather buffer at tile_m=512; with
    live-row tiles and a packed pool the gather scratch is ~1 MB and the
    third buffer pays, but ONLY together with the traced-slot body (alone it
    is -0.5 us, together -3.5; ledger fpg-rhs3).
  * **The pool is packed `int32[T, H/2]`** (`packed_pool=`), word `(r,c)` =
    bf16 `(r,c) | (r,c+H/2)<<16`. A gathered row moves 8 KB instead of the
    16 KB an int32 row-pair view costs, both halves are wanted so nothing is
    discarded, the parity column and its `[M,1]` XLA materialisation disappear,
    and `k` is the contracted axis so the unpack needs no weight permutation.
    This is the lever that clears the 95% bar (-17 us). In production it is a
    PRODUCER change -- 2 extra VPU ops, zero extra bytes -- never a repack pass.
    `pool_blocked=` accepts the alternative producer contract
    `bf16[T, H//128, 128]` (the parent's round-10 input form): dim 0 sits off
    the tiled pair so a plain bf16 per-row DMA is legal and there is no unpack
    at all; the dot pays one value-level reshape relayout instead. Measured:
    a small win only at small T with tile_m=32 (-0.4/-0.7 us @T=64), and
    +1.6 us at T=2048 -- the packed pool remains the default form.

`out_blocked=` declares the output `bf16[M, aligned_n//128, 128]` (the gdn-v3
shape) so bf16's sub-word packing sits off the axis the output DMA slices; it
is bitwise-exact and costs 0.1% here.

`coissue=` emits the next tile's row DMAs from inside `inner_kernel`, spread
over its matmul sites, which is the only way they share a scheduling region
with the dots -- a VLIW bundle can only pair two ops from one region. Injecting
the whole tile lifts the measured co-issue from 0% to 43.9% (the region
*ceiling* from 3.9% to 67.3%), but an injected row cannot be predicated (a
guard is a region boundary), so it reads `tile_m - live` dead rows per tile.
`coissue_rows` is therefore the depth knob, and it is a measured trade-off:
4 rows is the deepest prefix that still clears 95% of the floor.

    coissue_rows |  us  | % of ideal | region ceiling | issue co-issue
    -------------|------|------------|----------------|---------------
             0   | 191.7|   95.8%    |     4.4%       |     0%
             4   | 193.0|   95.1%    |     8.4%       |     1.0%
             8   | 193.9|   94.7%    |    12.3%       |     1.0%
            64   | 196.6|   93.5%    |      --        |      --
      64, no pool| 210.9|   87.1%    |    67.3%       |    43.9%

    (Measured on the original chassis. On the migrated traced-slot +
    3-buffer chassis every co-issue depth costs ~+5 us at T=2048 and is
    ~neutral at T=64 -- the tighter body has less slack to absorb the
    unguarded prefix's dead reads. The mechanism and the knob remain;
    whether to pay for the property is per-deployment.)

Note on the warm-up: it must use the SAME prefix/tail split as the steady
state. Starting all `tile_m` rows there while the steady wait expects the split
desyncs the gather semaphore and the kernel core-halts (E0200) on the first
tile with fewer than `tile_m` live rows -- and before that fix the combination
simply hung, which is what made it look like a 45-minute compile wall.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm_v2 import (
    FusedWeightsRef,
    IndexMaps,
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
    # Headroom above the MEAN group size, because a gm tile smaller than a
    # group SPLITS it and every split re-streams that group's whole weight
    # block (~2.8 us here): measured at T=1024, tile_m=32 (mean 20, so
    # groups of >32 rows split), num_gm went 64 -> 79 and the kernel lost
    # 28.6 us -- far more than the small-tile saving. group_sizes is traced,
    # so the largest group is unknowable at trace time; mean + 3*sqrt(mean)
    # covers a balanced (multinomial) router's spread. A skewed router still
    # splits its hot groups, which is correct, just not free.
    live += 3 * int(live**0.5 + 0.999)  # + 3*ceil(sqrt(mean))
    return int(min(512, max(32, align_to(live, 32))))


LANES = 128


def _blocked_out_spec(metadata_ref, cfgs):
    """gmm_v2's out BlockSpec with a trailing `LANES` minor dimension.

    A 2-D bf16[M, N] ref is tiled (16,128) with (2,1) packing: two adjacent
    ROWS share each 32-bit word, and rows are the axis the output DMA slices.
    Splitting the last dim moves the packing to the second-minor axis, wholly
    inside one M index, so the sliced axis becomes word-addressable.

    Caveat at THIS geometry: bf16's tiled pair is (16,128), so the
    second-minor dim is padded up to 16. With `out_size_n = 1024` that is
    8 -> 16, which DOUBLES the output array; blockpack only pays for itself
    when `aligned_n` is a multiple of 2048.
    """
    index_map = IndexMaps(metadata_ref, cfgs)
    bounded = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    def out_index_map(n_id, gm_id, k_id):
        rows, _, n = index_map.out_index_map(n_id, gm_id, k_id)
        return (rows, 0, n, 0)

    return pl.BlockSpec(
        (bounded, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n // LANES,
         LANES), out_index_map)


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
    lhs_buf,  # [num_slots, tile_m, tile_k] lhs dtype, or None
    *,
    cfgs: GmmConfigs,
    idx_smem,  # closed over: int32[padded_m] SMEM, kernel-invariant
    lhs_pool,  # closed over: [size_src(/packing), size_k] HBM
    parity_hbm,  # closed over: int32[padded_m, 1] HBM
    packing: int,
    num_slots: int,
    chunk: int,
    coissue: bool,
    packed_pool: bool,
    pool_blocked: bool,
    issue_spread: int,
    out_blocked: int,
    coissue_rows: int,
    traced_slots: bool,
    windowed_extract: bool,
):
    """Pipeline body. rhs/out stay pipelined; the LHS tile is gathered."""
    tile_m = cfgs.tiles.tile_m
    tile_k = cfgs.tiles.tile_k
    sublane = cfgs.dims.size_lhs_sublane
    shift = packing.bit_length() - 1  # bf16 pool: pool row = idx >> 1
    # A packed pool stores int32[T, H/2], word (r, c) = bf16 (r, c) in the low
    # half and (r, c + H/2) in the high half. A row is then 2H bytes instead of
    # the 4H the u32 row-pair view costs, the row index needs no shift, and the
    # parity column disappears entirely.
    row_k = tile_k // 2 if packed_pool else tile_k
    if pool_blocked:
        # bf16[T, H//128, 128] pool: dim 0 is off the tiled pair, so a plain
        # bf16 per-row DMA is legal and there is no unpack at all; the dot
        # pays one value-level reshape relayout instead (see extract()).
        row_k = tile_k // LANES

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

    def start_rows(step, slot, first=0):
        """Issue the tile's row DMAs.

        The row DMA size is a static 1: Mosaic rejects a slice whose size
        along a tiled dimension is dynamic ("Slice sizes along tiled
        dimensions must be aligned to tiles"), so a dead row cannot be
        expressed as a zero-row copy and has to be skipped by a predicate
        instead. `chunk` is how many rows share one predicate -- it trades
        wasted reads (up to chunk-1 dead rows per tile) against the number of
        regions the issue chain is cut into, and a region boundary is exactly
        what stops an issue bundle from sharing a VLIW bundle with the matmul.
        Keeping `tile_m` near the live-row count is what makes a large chunk
        affordable.

        Addresses come from the kernel-invariant SMEM table, so no semaphore
        wait sits in front of this scalar chain.
        """
        m_offset, live = m_base(step)
        k_base = lax.rem(step, num_k) * row_k
        for c in range(first, tile_m, chunk):

            @pl.when(c < live)
            def _(c=c):
                copies = [
                    pltpu.make_async_copy(
                        lhs_pool.at[pl.ds(_row(m_offset + j), 1),
                                    pl.ds(k_base, row_k)],
                        gather_buf.at[slot, pl.ds(j, 1)],
                        gather_sem.at[slot],
                    ) for j in range(c, min(c + chunk, tile_m))
                ]
                # Build every descriptor before starting any of them.
                for cp in copies:
                    cp.start()

    def _row(i):
        row = idx_smem[i]
        if packed_pool or pool_blocked:
            return row  # one physical row per logical row, no pairing
        return row >> shift if packing > 1 else row

    def wait_rows(step, slot, first=0):
        _, live = m_base(step)
        for c in range(first, tile_m, chunk):

            @pl.when(c < live)
            def _(c=c):
                for _ in range(min(c + chunk, tile_m) - c):
                    pltpu.make_async_copy(
                        lhs_pool.at[pl.ds(0, 1), pl.ds(0, row_k)],
                        gather_buf.at[slot, pl.ds(0, 1)],
                        gather_sem.at[slot],
                    ).wait()

    def extract(slot):
        """u32 row -> the bf16 row the index selected."""
        if pool_blocked:
            # The row arrived as plain bf16 whole words -- there is nothing
            # to unpack. The dot wants sublane=row, the blocked buffer has
            # sublane=block, so this reshape is a relayout; it replaces the
            # whole packed-pool unpack chain.
            return gather_buf[slot].reshape(tile_m, tile_k)
        if packed_pool:
            # Both halves are wanted, so there is no discarded read and no
            # parity to look up. k is the CONTRACTED axis, so the two halves
            # only have to end up in the original column order.
            bits = gather_buf[slot]
            lo = jax.lax.bitcast_convert_type(
                jnp.left_shift(bits, 16), jnp.float32).astype(
                    cfgs.lhs_cfgs.dtype)
            hi = jax.lax.bitcast_convert_type(
                jnp.bitwise_and(bits, jnp.int32(-65536)), jnp.float32).astype(
                    cfgs.lhs_cfgs.dtype)
            if lhs_buf is None:
                # Concat form: lands on a 128-multiple lane boundary, so it is
                # free -- but it keeps one [tile_m, tile_k] value live across
                # everything downstream, which is fatal once the issue chain
                # is injected into the matmul (co-issue + packed pool did not
                # finish compile+run in 45 min in that form).
                return jnp.concatenate([lo, hi], axis=1)
            half = tile_k // 2
            lhs_buf[slot, :, :half] = lo
            lhs_buf[slot, :, half:] = hi
            return lhs_buf[slot]
        if packing == 1:
            return gather_buf[slot]
        sh = 16 * (1 - jnp.bitwise_and(parity_buf[slot], 1))
        bits = jnp.bitwise_and(jnp.left_shift(gather_buf[slot], sh),
                               jnp.int32(-65536))
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(
            cfgs.lhs_cfgs.dtype)

    def extract_window(slot, rows):
        """`extract`, but only rows [0:rows) -- `rows` is a PYTHON int (a
        bucket size, called from inside inner_kernel's bucket branches via
        `lhs_fn`), so the unpack/reshape bills the BUCKET, not tile_m. This
        is what lets tile_m be a static CAPACITY instead of a price
        (solve tile-trilemma)."""
        if pool_blocked:
            return gather_buf[slot, :rows].reshape(rows, tile_k)
        if packed_pool:
            bits = gather_buf[slot, :rows]
            lo_ = jax.lax.bitcast_convert_type(
                jnp.left_shift(bits, 16), jnp.float32).astype(
                    cfgs.lhs_cfgs.dtype)
            hi_ = jax.lax.bitcast_convert_type(
                jnp.bitwise_and(bits, jnp.int32(-65536)), jnp.float32).astype(
                    cfgs.lhs_cfgs.dtype)
            return jnp.concatenate([lo_, hi_], axis=1)
        if packing == 1:
            return gather_buf[slot, :rows]
        sh = 16 * (1 - jnp.bitwise_and(parity_buf[slot, :rows], 1))
        bits = jnp.bitwise_and(jnp.left_shift(gather_buf[slot, :rows], sh),
                               jnp.int32(-65536))
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(
            cfgs.lhs_cfgs.dtype)

    def start_rows_unguarded(step, slot, lo, hi):
        """Rows [lo, hi) of `step`'s tile, with NO predicate.

        A predicate would put the issue chain in its own region, and a VLIW
        bundle can only pair two ops from one region -- which is why the
        guarded form measures 0% co-issue however few guards it has. Rows past
        the tile's live count are harmless: the index table is edge-padded, so
        they name a valid pool row, and inner_kernel masks them out of the
        result. The price is `tile_m - live` wasted row reads, which is what
        keeps `tile_m` pinned to the live-row count.
        """
        m_offset, _ = m_base(step)
        k_base = lax.rem(step, num_k) * row_k
        copies = [
            pltpu.make_async_copy(
                lhs_pool.at[pl.ds(_row(m_offset + j), 1),
                            pl.ds(k_base, row_k)],
                gather_buf.at[slot, pl.ds(j, 1)],
                gather_sem.at[slot],
            ) for j in range(lo, hi)
        ]
        for cp in copies:
            cp.start()

    def wait_rows_prefix(slot, n):
        for _ in range(n):
            pltpu.make_async_copy(
                lhs_pool.at[pl.ds(0, 1), pl.ds(0, row_k)],
                gather_buf.at[slot, pl.ds(0, 1)],
                gather_sem.at[slot],
            ).wait()

    def run(slot):
        """One grid step, with `slot` a PYTHON int."""
        nxt = (slot + 1) % num_slots

        if coissue:
            # HYBRID split. `coissue_rows` rows are issued from INSIDE
            # inner_kernel, so they share a scheduling region with the dots and
            # actually co-issue; the rest keep the ordinary guarded path
            # outside it. Two reasons not to inject the whole tile:
            #   * an unguarded row cannot be skipped, so injecting all of them
            #     reads `tile_m - live` dead rows per tile (5.7 us at T=2048),
            #     which costs more than the co-issue saves;
            #   * injecting all of them did not finish compile+run in 25-63 min
            #     once the packed pool was in -- at every `issue_spread`, with
            #     and without staging, and with a single slot.
            # A small in-region prefix gets the property at negligible cost:
            # rows below it are live on essentially every tile.
            n_co = min(coissue_rows, tile_m)

            def issue_fn(site, n_sites, nxt=nxt):
                n_use = n_sites if issue_spread <= 0 else min(n_sites,
                                                              issue_spread)
                if site >= n_use:
                    return
                per = -(-n_co // n_use)
                lo = site * per
                if lo >= n_co:
                    return

                @pl.when(s + 1 < num_steps)
                def _():
                    start_rows_unguarded(s + 1, nxt, lo, min(lo + per, n_co))

            # The wait has to mirror the split exactly or the semaphore
            # never balances: the injected prefix is unguarded, the tail
            # carries the same predicates its starts do.
            wait_rows_prefix(slot, n_co)
            wait_rows(s, slot, first=n_co)
            if packing > 1 and not packed_pool:
                wait_parity(slot)

            @pl.when(s + 1 < num_steps)
            def _():
                # the tail, guarded as usual, outside the dots' region
                start_rows(s + 1, nxt, first=n_co)
                if packing > 1 and not packed_pool:
                    start_parity(s + 1, nxt)

            inner_kernel(None if windowed_extract else extract(slot),
                         tiled_rhs_ref, tiled_out_ref,
                         partial_out_ref, acc_ref, metadata_ref, cfgs=cfgs,
                         issue_fn=issue_fn, out_blocked=out_blocked,
                         lhs_fn=functools.partial(extract_window, slot)
                         if windowed_extract else None)
            return

        @pl.when(s + 1 < num_steps)
        def _():
            # Issue-first: feed the engine for the next tile before this
            # tile's data is touched. The parent project's post-dot order
            # (round 14) was probed and measured +4.3..+8.2 us here -- on
            # the emit_pipeline chassis issue-first gives the next tile's
            # gather a full dot of cover (ledger fpg-postdot).
            start_rows(s + 1, nxt)
            if packing > 1 and not packed_pool:
                start_parity(s + 1, nxt)

        wait_rows(s, slot)
        if packing > 1 and not packed_pool:
            wait_parity(slot)
        inner_kernel(None if windowed_extract else extract(slot),
                     tiled_rhs_ref, tiled_out_ref,
                     partial_out_ref, acc_ref, metadata_ref, cfgs=cfgs,
                     out_blocked=out_blocked,
                     lhs_fn=functools.partial(extract_window, slot)
                     if windowed_extract else None)

    @pl.when(s == 0)
    def _():
        if coissue:
            # The warm-up must use the SAME split as the steady state, or the
            # gather semaphore never balances: the steady wait is
            # `n_co unguarded + the guarded tail`, so the warm-up has to start
            # exactly that. Starting all tile_m rows here core-halts (E0200)
            # as soon as a tile has fewer than tile_m live rows.
            start_rows_unguarded(0, 0, 0, min(coissue_rows, tile_m))
            start_rows(0, 0, first=min(coissue_rows, tile_m))
        else:
            start_rows(0, 0)
        if packing > 1 and not packed_pool:
            start_parity(0, 0)

    if traced_slots:
        # The parent's round-13/14 counter-structure: emit the body ONCE with
        # a traced slot. Halves the program (and its region count) at the
        # price of folding a runtime multiply/add into every VMEM destination
        # and semaphore address; which side wins is chassis-dependent and is
        # measured, not assumed.
        run(lax.rem(s, num_slots))
    else:
        for slot in range(num_slots):
            # Dispatching on the parity makes every slot index inside the
            # branch a Python constant, so it folds into the descriptor.
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
    lhs_buf,
    zero_ref,
    semaphore_ref,
    *,
    cfgs: GmmConfigs,
    packing: int,
    num_slots: int,
    chunk: int,
    coissue: bool,
    packed_pool: bool,
    pool_blocked: bool,
    issue_spread: int,
    out_blocked: int,
    coissue_rows: int,
    traced_slots: bool,
    rhs_buffers: int,
    windowed_extract: bool,
):
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref,
                           metadata_ref, cfgs=cfgs)

    if cfgs.zero_init:
        zero_size = zero_out_start(out_ref, zero_ref, semaphore_ref,
                                   metadata_ref, num_gm, dims=cfgs.dims)

    (_, rhs_spec), out_spec = generate_block_specs(
        metadata_ref, cfgs, rhs_buffer_count=rhs_buffers)
    if out_blocked:
        out_spec = _blocked_out_spec(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n:],
                                  rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_spec = FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    lhs_pool = (lhs_ref if packed_pool or pool_blocked or packing == 1
                else lhs_ref.bitcast(jnp.int32))
    body = functools.partial(_fused_permute_gmm_inner, cfgs=cfgs,
                             idx_smem=idx_smem, lhs_pool=lhs_pool,
                             parity_hbm=parity_hbm, packing=packing,
                             num_slots=num_slots, chunk=chunk,
                             coissue=coissue,
                             packed_pool=packed_pool,
                             pool_blocked=pool_blocked,
                             issue_spread=issue_spread,
                             out_blocked=out_blocked,
                             coissue_rows=coissue_rows,
                             traced_slots=traced_slots,
                             windowed_extract=windowed_extract)

    pipeline_fn = pltpu.emit_pipeline(body, grid=(num_n, num_gm, num_k),
                                      in_specs=(rhs_spec, ),
                                      out_specs=out_spec)

    if out_blocked:
        out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane,
                                 out_ref.shape[-2], out_ref.shape[-1])
    else:
        out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane,
                                 out_ref.shape[-1])
    pipeline_fn(rhs_ref, out_in,
                scratches=[partial_out_ref, acc_ref, metadata_ref, gather_buf,
                           gather_sem, parity_buf, parity_sem, lhs_buf])

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


@functools.partial(jax.jit, static_argnames=[
    "tile_info", "vmem_limit_bytes", "precision", "preferred_element_type",
    "acc_dtype", "maybe_quantize_lhs", "zero_initialize", "fuse_act",
    "num_slots", "chunk", "coissue", "packed_pool", "issue_spread",
    "out_blocked", "stage_lhs", "coissue_rows",
    "traced_slots", "rhs_buffers", "pool_blocked", "windowed_extract",
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
    chunk: int | None = None,
    coissue: bool = True,
    packed_pool: bool = False,
    issue_spread: int = 1,
    out_blocked: bool = False,
    stage_lhs: bool = False,
    coissue_rows: int = 4,
    traced_slots: bool = True,
    rhs_buffers: int = 3,
    pool_blocked: bool = False,
    windowed_extract: bool = False,
) -> jax.Array:
    """Grouped matmul over `lhs[gather_indices]` without materialising it.

    Row `i` of the effective LHS is `lhs[gather_indices[i]]`; the output is
    [size_m, size_n] with size_m = gather_indices.shape[0].
    """
    if packed_pool and lhs.dtype != jnp.int32:
        raise ValueError("packed_pool expects an int32[size_src, size_k/2] "
                         "pool; word (r,c) = bf16 (r,c) | (r,c+size_k/2)<<16")
    if pool_blocked:
        if packed_pool:
            raise ValueError("pool_blocked and packed_pool are two forms of "
                             "the same producer contract; pick one")
        if lhs.ndim != 3 or lhs.shape[-1] != LANES:
            raise ValueError("pool_blocked expects bf16[size_src, "
                             "size_k//128, 128]")
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

    if packed_pool:
        # lhs is int32[size_src, size_k/2]; the effective LHS is bf16.
        packing = 2
        cfg_lhs = jax.ShapeDtypeStruct((size_m, lhs.shape[1] * 2),
                                       jnp.bfloat16)
    elif pool_blocked:
        # lhs is bf16[size_src, size_k//128, 128]; one physical row per
        # logical row, whole 32-bit words, nothing to unpack.
        packing = 1
        cfg_lhs = jax.ShapeDtypeStruct((size_m, lhs.shape[1] * lhs.shape[2]),
                                       lhs.dtype)
    else:
        packing = 4 // lhs.dtype.itemsize
        cfg_lhs = jax.ShapeDtypeStruct((size_m, lhs.shape[1]), lhs.dtype)
    if tile_info is None:
        tile_m = gather_tiling(size_m, group_sizes.shape[0])
        tile_info = TileSizes(tile_m=tile_m, tile_k=cfg_lhs.shape[1],
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
    if chunk is None:
        # Measured at T=64 (live ~10 rows/tile): chunk=8 saves 1.9 us of
        # dead-row reads on a 32-row tile; 16 is right once tiles are 64+.
        chunk = 8 if tiles.tile_m <= 32 else 16
    if group_offset is None:
        group_offset = jnp.zeros((1, ), jnp.int32)

    pool = lhs
    if not (packed_pool or pool_blocked) and packing > 1 \
            and pool.shape[0] % packing:
        pool = jnp.pad(pool, ((0, packing - pool.shape[0] % packing), (0, 0)))

    # Pad the index table so a tile that runs past the last live row still
    # names a valid pool row (bounds checks are off).
    pad_to = align_to(dims.size_m, tiles.tile_m) + tiles.tile_m
    idx = jnp.pad(gather_indices.astype(jnp.int32), (0, pad_to - size_m),
                  mode="edge")

    parity = (idx[:, None] if packing > 1 and not packed_pool
              else jnp.zeros((1, 1), jnp.int32))

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
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n // LANES, LANES),
                   cfgs.out_dtype) if out_blocked else
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        pltpu.VMEM((num_slots, tiles.tile_m, tiles.tile_k // LANES, LANES),
                   lhs.dtype) if pool_blocked else
        pltpu.VMEM((num_slots, tiles.tile_m,
                    tiles.tile_k // 2 if packed_pool else tiles.tile_k),
                   jnp.int32 if packing > 1 else lhs.dtype),
        pltpu.SemaphoreType.DMA((num_slots, )),
        pltpu.VMEM((num_slots, tiles.tile_m, 1), jnp.int32),
        pltpu.SemaphoreType.DMA((num_slots, )),
        pltpu.VMEM((num_slots, tiles.tile_m, tiles.tile_k), cfgs.lhs_cfgs.dtype)
        if (packed_pool and stage_lhs) else None,
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
    out_init = (jax.ShapeDtypeStruct(
        (dims.size_m, aligned_n // LANES, LANES), cfgs.out_dtype)
        if out_blocked else
        jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype))

    out = pl.pallas_call(
        functools.partial(kernel_main_fpg, cfgs=cfgs, packing=packing,
                          num_slots=num_slots, chunk=chunk,
                             coissue=coissue,
                             packed_pool=packed_pool,
                             issue_spread=issue_spread,
                             out_blocked=LANES if out_blocked else 0,
                             coissue_rows=coissue_rows,
                             traced_slots=traced_slots,
                             rhs_buffers=rhs_buffers,
                             pool_blocked=pool_blocked,
                             windowed_extract=windowed_extract),
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
    )(group_sizes, group_offset, idx, pool,
      parity, rhs_weights)
    return out if out_blocked else out[:, :cfgs.out_size_n]

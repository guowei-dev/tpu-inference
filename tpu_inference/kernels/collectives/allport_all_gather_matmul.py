# SPDX-License-Identifier: Apache-2.0
"""All-port fused all-gather matmul (parity-plane hypercube dissemination).

`all_gather(x, axis=0) @ y` on a slice whose chip grid has every axis extent
<= 2, so the chip labels form a hypercube and every XOR partner is one hop
(topology.assert_xor_partners_adjacent).

Why this exists — the wire account, which IS the result:
a ring cycle visits 2 of a chip's 3 ICI ports and carries 15 chunks over them,
7.5P per port. A chip only has to IMPORT 14 remote chunks; spread over 3 ports
that is 4.67P, and 4.67P is a floor, not a tuning target. This schedule
attains it:

  * each core disseminates only its OWN parity plane over ICI (a 3-round
    hypercube ladder, 1 + 2 + 4 = 7P per core) and forwards every arrival to
    its twin over the on-chip D2D link, so a chip imports each remote chunk
    exactly once;
  * the chunk's ROWS are split into bands and band b trades dim (b + s) % dims
    at round s, so all three ICI ports carry traffic every round — that is
    what turns 7P per core into 2.33P per core per port;
  * a band slice is a complete [rows, k] sub-chunk, so it dots the moment it
    lands, and the exponential ladder hands the MXU 2, 2, 4 and 8 chunks in
    turn — dot heights that grow, instead of the ring's constant m // tp.

Deadlock-freedom (the rule future edits must keep) is allport_matmul_reduce_
scatter's, verbatim: every device emits the SAME static sequence of DMA starts
and semaphore waits, enumerated in RELATIVE chip labels l = q XOR my_chip, so
slot indices and program positions are device-invariant and only traced values
(partner ids, output row offsets) differ. Never wrap a blocking or signaling
op in a device-dependent branch, and never make a wait's slot index
data-dependent.

Layout contract: logical id == chip * 2 + core (topology.make_collective_mesh
('hier')); wire dtype = input dtype; fp32 accumulate; collective_id = 5.

Residency: arrivals land in an HBM gather buffer laid out in ARRIVAL order, so
each round's arrivals are a contiguous row range and the compute reads them
back through a small VMEM staging pipeline. y is VMEM-resident per n block;
when the whole per-device y does not fit, n is split into `grid_n` blocks of
one fixed width whose last block overlaps its predecessor backwards — every
VMEM ref is then used at its full extent, which is what keeps a lane-ragged
n // tp legal (all_gather_matmul's `_n_window` makes the same argument).
"""
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import topology, util

_COLLECTIVE_ID = 5

# Measured scoped-VMEM ceiling (memory_space_assignment clamps requests here).
_VMEM_CAP_BYTES = 67043328
_VMEM_SLACK_BYTES = 6 * 1024 * 1024

# Row granularity of a band. A bf16 VMEM ref is tiled (8, 128) with (2, 1)
# packing — 8 sublanes x 2 rows per word = 16 ROWS per tile — and Mosaic
# refuses a sublane offset it cannot prove tile-aligned ("Offsets along tiled
# dimensions must be aligned"). The output staging buffer is sliced at piece
# boundaries, so every band is a multiple of 16 rows.
_ROW_GRAN = 16

# One MXU pass. A dot shorter than this runs at roughly bm / 128 occupancy, so
# it is the threshold _pick_config trades an extra n block against.
_MXU_ROWS = 128


def _band_rows(m_per, num_bands):
    """Split m_per rows into num_bands pieces, each a multiple of _ROW_GRAN."""
    units = m_per // _ROW_GRAN
    base, rem = divmod(units, num_bands)
    return [(base + (1 if i < rem else 0)) * _ROW_GRAN
            for i in range(num_bands)]


def _plan(num_dims, m_per, order="head"):
    """Static gather-buffer layout, in expected-ARRIVAL order.

    Rows [0, m_per) are the local chunk and [m_per, 2 m_per) the twin's; the
    remaining pieces are laid out in the order the compute wants to consume
    them, which is the order they are expected to LAND.

    `order` picks that consumption order, and the right key is a message's
    ISSUE time, not the round that carries it: message (s, b, j) goes out as
    soon as ladder position j is in hand, so every round's `j == 0` message is
    issued at t = 0 and a round-2 arrival lands alongside a round-0 one, on a
    different port. (Ordering by DELIVERY position would be no different from
    round order: round s delivers exactly positions [2^s, 2^(s+1)).)

      round  the ladder's own order. A batch never spans two rounds, so it
             never waits on a late piece to dot early rows — but the MXU
             starves early, because only num_bands pieces per parity are at
             the head of the buffer.
      issue  sort by j: 3 x num_bands pieces per parity at the head. Measured
             a large-M lever and a mid-M regression (h8192_d20736 X=8192
             657 -> 590 us, X=1024 115 -> 129), because once rounds interleave
             a bm-row batch can span them and waits for the latest.
      head   only the j == 0 messages move to the front; the rest keep round
             order. THE DEFAULT: measured against `round` at 12 cells it wins
             9-11% from X = 4096 (h8192_d20736 X=8192 657 -> 605, X=4096
             342 -> 320, h4096_d20736 X=8192 384 -> 348) and is within noise
             everywhere else — no measured regression. `issue` beats it again
             on h8192 (581 / 304 / 174 at X = 8192 / 4096 / 2048) but costs
             +3.3% at h4096_d20736 X=2048, so it stays opt-in.
    """
    num_bands = max(1, min(num_dims, m_per // _ROW_GRAN))
    ladder = topology.allport_ag_ladder(num_dims, num_bands)
    rows = _band_rows(m_per, num_bands)
    starts = [sum(rows[:b]) for b in range(num_bands)]
    pieces = []

    def add(par, label, band, key):
        pieces.append(
            dict(par=par, label=label, band=band, rows=rows[band], key=key,
                 out_off=starts[band],
                 g_off=(pieces[-1]["g_off"] +
                        pieces[-1]["rows"] if pieces else 0)))

    for par in (0, 1):
        for b in range(num_bands):
            add(par, 0, b, None)
    arrivals = [(s, b, j) for s in range(num_dims)
                for b in range(num_bands) for j in range(1 << s)]
    # k = (s, b, j); k[2] == j is the position whose payload the message
    # carries, so it is also when the message can be issued.
    sort_key, wave_of = {
        "round": (lambda k: (k[0], k[2], k[1]), lambda k: k[0]),
        "issue": (lambda k: (k[2], k[0], k[1]), lambda k: k[2]),
        "head": (lambda k: (min(k[2], 1), k[0], k[2], k[1]),
                 lambda k: (min(k[2], 1), k[0])),
    }[order]
    arrivals.sort(key=sort_key)
    # Within a wave the ICI arrivals come before the D2D forwards of the same
    # wave, because a forward is that arrival relayed by the twin and can only
    # land later. Interleaving the two parities instead costs 12-16% at mid M.
    i = 0
    while i < len(arrivals):
        j = i
        while (j < len(arrivals)
               and wave_of(arrivals[j]) == wave_of(arrivals[i])):
            j += 1
        for par in (0, 1):
            for key in arrivals[i:j]:
                add(par, ladder["recvs"][key], key[1], key)
        i = j

    loc = {(p["par"], p["label"], p["band"]): (p["g_off"], p["rows"])
           for p in pieces}
    assert pieces[-1]["g_off"] + pieces[-1]["rows"] == m_per * (2 << num_dims)
    return ladder, pieces, loc


def _batches(pieces, bm, split_at):
    """Row segments of at most bm rows covering every piece, in arrival order.

    A batch is a list of (piece index, offset in that piece, rows); pieces are
    contiguous in the gather buffer, so a batch is one contiguous read. No
    batch straddles `split_at` — the local chunk is read from the kernel's own
    `x` input rather than from the gather buffer, and a DMA has one source.
    """
    out, cur, filled = [], [], 0
    for i in range(len(pieces)):
        left, off = pieces[i]["rows"], 0
        while left:
            take = min(left, bm - filled)
            at = pieces[i]["g_off"] + off
            if at < split_at:
                take = min(take, split_at - at)
            cur.append((i, off, take))
            off, left, filled = off + take, left - take, filled + take
            if filled == bm or at + take == split_at:
                out.append(cur)
                cur, filled = [], 0
    if cur:
        out.append(cur)
    return out


def _allport_ag_kernel(
    x_hbm,
    y_hbm,
    out_hbm,
    g_hbm,
    y_vmem,
    xs_vmem,
    o_vmem,
    sem_y,
    sem_xs,
    sem_o,
    sems_ici,
    sems_tw,
    *,
    ladder,
    pieces,
    loc,
    m_per,
    n_blocks,
    bm,
    xs_slots,
    arm_ahead,
    axis_name,
    ablate,
):
    num_dims = ladder["num_dims"]
    num_bands = ladder["num_bands"]
    my_id = lax.axis_index(axis_name)
    bit = lax.rem(my_id, 2)
    my_chip = lax.div(my_id, 2)
    twin = my_id + 1 - 2 * bit

    partners = [twin]
    for d in range(num_dims):
        partners.append((my_chip ^ (1 << d)) * 2 + bit)
    util.local_barrier_logical(partners)

    # ---- wire ----------------------------------------------------------
    ici_slot = {}
    for s in range(num_dims):
        for b in range(num_bands):
            for j in range(1 << s):
                ici_slot[(s, b, j)] = len(ici_slot)

    def src_of(off, rows):
        """Rows below m_per are the local chunk and live in `x`, not in the
        gather buffer — which is why no local copy is needed at all."""
        ref = x_hbm if off + rows <= m_per else g_hbm
        return ref.at[pl.ds(off, rows)]

    def ici_op(key):
        s, b, _ = key
        dim = ladder["dims"][(b, s)]
        so, sr = loc[(0, ladder["sends"][key], b)]
        do, _ = loc[(0, ladder["recvs"][key], b)]
        slot = ici_slot[key]
        return pltpu.make_async_remote_copy(
            src_ref=src_of(so, sr),
            dst_ref=g_hbm.at[pl.ds(do, sr)],
            send_sem=sems_ici.at[slot],
            recv_sem=sems_ici.at[slot],
            device_id=(my_chip ^ (1 << dim)) * 2 + bit,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def tw_op(slot, label, band):
        """Forward my parity's (label, band) slice into the twin's slot."""
        so, sr = loc[(0, label, band)]
        do, _ = loc[(1, label, band)]
        return pltpu.make_async_remote_copy(
            src_ref=src_of(so, sr),
            dst_ref=g_hbm.at[pl.ds(do, sr)],
            send_sem=sems_tw.at[slot],
            recv_sem=sems_tw.at[slot],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    tw_own = [(b, 0, b) for b in range(num_bands)]
    tw_keys, nxt = {}, num_bands
    for s in range(num_dims):
        tw_keys[s] = []
        for j in range(1 << s):
            for b in range(num_bands):
                tw_keys[s].append((nxt, ladder["recvs"][(s, b, j)], b))
                nxt += 1

    bn_max = n_blocks[-1][1]

    def y_win(width):
        """The y/out staging window for a block of `width` columns.

        The buffers are sized to the RAGGED last block, so that block is the
        whole extent and every other block is a 128 x 128-aligned sub-window —
        the only two forms Mosaic accepts on a tiled ref.
        """
        return slice(None) if width == bn_max else pl.ds(0, width)

    def y_load(nb):
        col, width = n_blocks[nb]
        return pltpu.make_async_copy(y_hbm.at[:, pl.ds(col, width)],
                                     y_vmem.at[:, y_win(width)], sem_y)

    # ---- compute -------------------------------------------------------
    dot_i, load_i = [0], [0]
    o_live = [None, None]

    def batch_off(batch):
        pi, poff, _ = batch[0]
        return pieces[pi]["g_off"] + poff

    def batch_rows(batch):
        return sum(r for _, _, r in batch)

    def xs_copy(batch, slot):
        return pltpu.make_async_copy(
            src_of(batch_off(batch), batch_rows(batch)),
            xs_vmem.at[slot, pl.ds(0, batch_rows(batch))], sem_xs.at[slot])

    def start_load(batch):
        xs_copy(batch, load_i[0] % xs_slots).start()
        load_i[0] += 1

    def emit_dot(batch, nb):
        i = dot_i[0]
        xs, os_ = i % xs_slots, i % 2
        rows = batch_rows(batch)
        xs_copy(batch, xs).wait()
        if o_live[os_] is not None:
            for cp in o_live[os_]:
                cp.wait()
            o_live[os_] = None
        col, width = n_blocks[nb]
        win = y_win(width)
        if ablate != 4:
            # fp32 accumulate, then one narrowing store. Mosaic's tpu.matmul
            # rejects a non-fp32 preferred_element_type, so the [bm, bn] fp32
            # landing value is structural, not a choice.
            o_vmem[os_, pl.ds(0, rows), win] = jnp.dot(
                xs_vmem[xs, pl.ds(0, rows), slice(None)], y_vmem[:, win],
                preferred_element_type=jnp.float32).astype(o_vmem.dtype)
        started, at = [], 0
        for (pi, poff, prows) in batch:
            p = pieces[pi]
            dev = ((my_chip ^ p["label"]) * 2 +
                   (bit if p["par"] == 0 else 1 - bit))
            cp = pltpu.make_async_copy(
                o_vmem.at[os_, pl.ds(at, prows), win],
                out_hbm.at[pl.ds(dev * m_per + p["out_off"] + poff, prows),
                           pl.ds(col, width)], sem_o.at[os_])
            cp.start()
            started.append(cp)
            at += prows
        o_live[os_] = started
        dot_i[0] += 1

    def run(batch_list, nb, arm=None, ahead=1):  # noqa: C901
        """Software-pipelined dot loop.

        `arm(i)` runs before batch i's staging load is issued and is where the
        arrival waits live; the load for batch i + depth is issued AFTER batch
        i's dot, so a blocking arrival wait sits behind as much already-staged
        compute as the pipeline is deep. It does not remove the stall: a static
        program's wait halts everything behind it, and the full kernel measures
        1.30-1.33x max(compute-only, wire-only) — the largest term still open.
        """
        if ablate == 2:  # wire only: arrivals still drain, no dot, no staging
            for i in range(len(batch_list)):
                if arm:
                    arm(i)
            return
        depth = min(xs_slots - 1, len(batch_list))
        # The arrival waits run `ahead` batches in front of the staging loads,
        # because an arrival is not only a dot's operand: the actions behind it
        # ISSUE the next rounds' messages. Tying the waits to the staging depth
        # made the dot loop gate when the wire advances, which is why the wire
        # inside the full kernel is longer than the wire measured alone.
        lead = min(max(depth, depth * ahead), len(batch_list))
        for i in range(lead):
            if arm:
                arm(i)
        for i in range(depth):
            start_load(batch_list[i])
        for i, batch in enumerate(batch_list):
            emit_dot(batch, nb)
            if arm and i + lead < len(batch_list):
                arm(i + lead)
            if i + depth < len(batch_list):
                start_load(batch_list[i + depth])
        for os_ in range(2):
            if o_live[os_] is not None:
                for cp in o_live[os_]:
                    cp.wait()
                o_live[os_] = None

    # Per-piece unlock actions. An arrival is not just a dot's operand: it is
    # also the payload of the NEXT round's message on that band and of the D2D
    # forward to the twin, so both are issued the instant it lands. Per PIECE
    # rather than per round, because the ladder delivers HALF its data in the
    # last round and a round-granular wait serialises that half behind the
    # wire.
    actions = {}

    def act(i, fn):
        actions.setdefault(i, []).append(fn)

    def ici_arrival(s, b, j, fwd_slot, fwd_label):
        def go():
            ici_op((s, b, j)).wait()
            if ablate != 5:
                tw_op(fwd_slot, fwd_label, b).start()
            # This arrival fills ladder position j + 2^s, which is the payload
            # of that position's message in EVERY later round — start them all
            # now. With the j == 0 messages issued up front, that covers the
            # ladder exactly once and leaves no message waiting on a round
            # boundary it does not actually depend on.
            for nxt_s in range(s + 1, num_dims):
                ici_op((nxt_s, b, j + (1 << s))).start()

        return go

    def tw_arrival(slot, label, band):
        def go():
            if ablate != 5:
                tw_op(slot, label, band).wait()

        return go

    for b in range(num_bands):
        act(num_bands + b, tw_arrival(*tw_own[b]))
    fwd = {}  # (s, b, j) -> its twin-forward slot
    for s in range(num_dims):
        t = 0
        for j in range(1 << s):
            for b in range(num_bands):
                fwd[(s, b, j)] = tw_keys[s][t]
                t += 1
    # The plan lays pieces out in expected-arrival order, so wire each action
    # to the piece it belongs to rather than to a round's offset.
    for i, pc in enumerate(pieces):
        if pc["key"] is None:
            continue
        slot, label, band = fwd[pc["key"]]
        if pc["par"] == 0:
            act(i, ici_arrival(*pc["key"], slot, label))
        else:
            act(i, tw_arrival(slot, label, band))

    def arm_for(lo, batch_list):
        """arm(i): run every pending action for the pieces batch i reads."""
        ends = [bt[-1][0] for bt in batch_list]
        nxt = [lo]

        def arm(i):
            while nxt[0] <= ends[i]:
                for fn in actions.get(nxt[0], ()):
                    fn()
                nxt[0] += 1

        return arm

    # ---- program -------------------------------------------------------
    y_load(0).start()
    if ablate not in (1, 5):
        for slot, label, band in tw_own:
            tw_op(slot, label, band).start()

    if ablate != 1:  # a started-but-never-waited DMA halts the NEXT call
        for st in range(num_dims):  # every message that depends on nothing
            for b in range(num_bands):
                ici_op((st, b, 0)).start()
    y_load(0).wait()

    # ONE pipelined loop over the whole arrival order, not one per round: the
    # per-piece arm already gates each batch on exactly the arrivals it reads,
    # so a per-round `run` only drained the staging pipeline at every round
    # boundary — and with half the data arriving in the last round that is
    # where the overlap has to hold.
    bl = _batches(pieces, bm, m_per)
    run(bl, 0, arm=None if ablate == 1 else arm_for(0, bl), ahead=arm_ahead)

    for nb in range(1, len(n_blocks)):
        if ablate == 2:
            break
        y_load(nb).start()
        y_load(nb).wait()
        run(_batches(pieces, bm, m_per), nb)


def _pick_config(m, m_per, n_per, k, itemsize):
    """(n blocks, bm, xs_slots, vmem bytes) — the best configuration that fits.

    A lane-ragged n // tp can only be sliced two ways: at a 128-aligned offset
    with a 128-multiple width, or as a ref's whole extent. So every block but
    the last is 128 x 128-aligned, the LAST block absorbs the ragged remainder,
    and the VMEM buffers are sized to that last block — the ragged block is
    then the buffer's full extent while the aligned ones are legal
    sub-windows. Nothing is re-computed and no column is padded.

    Taking the FIRST grid_n that fits is wrong, and expensively so: where the
    whole-y block only just fits (56.0 of the 57.9 MiB budget at k=8192,
    n_per=3584) grid_n=1 leaves the staging pipeline 1.9 MiB and bm collapses
    to 16 rows — an MXU at ~bm/128 occupancy while the wire account still looks
    perfect. Measured 532 -> 165 us at m=1024 on that shape, ABBA, spread 0.05%.
    So: the smallest grid_n whose dot reaches a full MXU pass (each extra block
    re-streams the gathered rows, and grid_n 3/4 measured 185/184 against
    grid_n 2's 165), falling back to the tallest dot when none does. Selects
    the identical config at all 96 cells of the recorded (H, D) x X grid.
    """
    budget = _VMEM_CAP_BYTES - _VMEM_SLACK_BYTES
    cands = []
    for grid_n in (1, 2, 3, 4, 6, 8):
        if grid_n == 1:
            head, bn = n_per, n_per
        else:
            head = (n_per // grid_n) // 128 * 128
            bn = n_per - (grid_n - 1) * head  # ragged tail, and the widest
            if head < 128 or bn < head or bn >= n_per:
                continue
        y_bytes = k * bn * itemsize
        if y_bytes > budget:
            continue
        # Staging depth BEFORE dot height, measured: bm 512 -> 1024 buys the
        # MXU's bm/(bm + 128) weight-reload term (0.80 -> 0.89) but costs the
        # third staging slot, and the compute path got 10% SLOWER for it
        # (301 -> 330 us at h4096_d20736 X=8192). The prefetch is worth more
        # than the reload.
        for xs_slots in (3, 2):
            best = next((bm for bm in (512, 384, 256, 192, 128, 64, 32, 16)
                         if bm <= m and not bm % _ROW_GRAN
                         and y_bytes + xs_slots * bm * k * itemsize
                         + bm * bn * 4 + 2 * bm * bn * itemsize <= budget), None)
            if best is not None:
                cands.append((grid_n, head, bn, best, xs_slots,
                              y_bytes + xs_slots * best * k * itemsize
                              + best * bn * 4 + 2 * best * bn * itemsize))
                break
    if not cands:
        raise ValueError(
            f"all-port AG-MM does not fit at m_per={m_per}, n_per={n_per}, "
            f"k={k}: one 128-column block of y alone needs "
            f"{k * 128 * itemsize / 2**20:.1f} MiB.")
    full = [c for c in cands if c[3] >= _MXU_ROWS]
    grid_n, head, bn, bm, xs_slots, need = (
        full[0] if full else max(cands, key=lambda c: (c[3], -c[0])))
    blocks = [(j * head, head) for j in range(grid_n - 1)]
    blocks.append(((grid_n - 1) * head, bn))
    return blocks, bm, xs_slots, need


def allport_all_gather_matmul(
    x,
    y,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
    arm_ahead: int = 1,
    order: str = "head",
    ablate: int = 0,
):
    """all_gather(x, axis=0) @ y, all-port parity-plane kernel.

    ablate (instrumentation only — the RESULT IS WRONG for ablate != 0):
    1 = compute path with no wire at all, 2 = wire path with no dots or
    staging, 4 = everything EXCEPT the MXU (full wire, full staging and output
    DMA traffic, no dot) — the discriminator between resource contention and
    scheduling stalls, 5 = full minus the D2D twin forwards, which is the one
    part of the wire whose issue point sits INSIDE the peer's dot loop. Ops are
    skipped identically on every device, so none of them deadlocks.

    x: [M, k] P(axis, None), y: [k, n] P(None, axis) -> out [M, n // tp]
    P(None, axis). Mesh from topology.make_collective_mesh('hier').

    y: when n // tp_size is not a multiple of 128, store it with the row-major
      device layout (`jax.experimental.layout.Layout(major_to_minor=(0, 1))`).
      XLA's default for such a shape is column-major, which a Pallas call
      cannot take, so it relayouts the whole weight on every call.
    """
    tp_size = mesh.shape[axis_name]
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"inputs must be 2D, got {x.shape} and {y.shape}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtypes must match, got {x.dtype} and {y.dtype}")
    m, k = x.shape
    if y.shape[0] != k:
        raise ValueError(f"contraction mismatch: {x.shape} vs {y.shape}")
    n = y.shape[1]
    if n % tp_size or m % tp_size:
        raise ValueError(f"m ({m}) and n ({n}) must divide tp ({tp_size})")
    m_per, n_per = m // tp_size, n // tp_size
    if m_per % _ROW_GRAN:
        raise ValueError(f"m // tp ({m_per}) must be a multiple of "
                         f"{_ROW_GRAN}")
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    if num_dims == 0 or (1 << num_dims) != num_chips:
        raise ValueError("all-port AG-MM needs a power-of-two chip count >= 2")

    ladder, pieces, loc = _plan(num_dims, m_per, order)
    n_blocks, bm, xs_slots, vmem_need = _pick_config(m, m_per, n_per, k,
                                                     x.dtype.itemsize)

    def per_device(x_local, y_local):
        hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
        bn = n_blocks[-1][1]  # the ragged last block is the WIDEST
        scratch_shapes = (
            pltpu.VMEM((k, bn), y_local.dtype),
            pltpu.VMEM((xs_slots, bm, k), x_local.dtype),
            pltpu.VMEM((2, bm, bn), x_local.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((xs_slots, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((len(ladder["sends"]), )),
            pltpu.SemaphoreType.DMA(
                (ladder["num_bands"] + len(ladder["sends"]), )),
        )
        kernel = functools.partial(
            _allport_ag_kernel,
            ladder=ladder,
            pieces=pieces,
            loc=loc,
            m_per=m_per,
            n_blocks=n_blocks,
            bm=bm,
            xs_slots=xs_slots,
            arm_ahead=arm_ahead,
            axis_name=axis_name,
            ablate=ablate,
        )
        outs = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((m, n_per), x_local.dtype),
                       jax.ShapeDtypeStruct((m, k), x_local.dtype)),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm, hbm],
                out_specs=(hbm, hbm),
                scratch_shapes=scratch_shapes,
                grid=(1, ),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=min(vmem_need + 4 * 1024 * 1024,
                                     _VMEM_CAP_BYTES),
            ),
            name=f"allport_all_gather_matmul_d{num_dims}",
        )(x_local, y_local)
        return outs[0]

    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec(axis_name, None),
                  jax.sharding.PartitionSpec(None, axis_name)),
        out_specs=jax.sharding.PartitionSpec(None, axis_name),
        check_vma=False,
    )(x, y)

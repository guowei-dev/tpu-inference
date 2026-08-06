# SPDX-License-Identifier: Apache-2.0
"""All-port fused matmul reduce-scatter (hierarchical schedule, ring-style
pipelining).

Same schedule as hier_matmul_reduce_scatter (twin-pair D2D + concurrent-dims
hypercube: band b trades dim (b+s) % D at round s, so every ICI axis is busy
every round) but with the phase barriers removed: the emission order comes
from topology.allport_program — twin pairs in popcount-descending order (the
first twin merge lights a round-0 send on EVERY dim), each merge immediately
starts the sends it unlocks, and ladder waits are hoisted between pair
computes. Round r+1's send of a region depends only on that region's round-r
merge, never on the whole round.

Deadlock-freedom (the rule future edits must keep): every device emits the
SAME static sequence of DMA starts and semaphore waits — the ladder is
enumerated in RELATIVE labels l = q XOR my_chip, so slot indices and program
positions are device-invariant and only traced values (absolute labels,
partner ids, row offsets) differ. A wait on a shared cell then only needs the
partner to have issued its symmetric start, which it did at the same earlier
program position (induction over wait index; the CPU simulator in
allport_schedule_test.py re-checks every edit). Never wrap a blocking or
signaling op in a device-dependent branch, and never make a wait's slot index
data-dependent.

Layout contract: logical id == chip * 2 + core (topology.make_collective_mesh
('hier')); wire dtype = input dtype; fp32 local accumulate; hops = 1 +
log2(num_chips). collective_id = 4.

Residency (v4a): below the scoped-VMEM budget the run slots live in VMEM
(_allport_kernel, the original form). Above it — M >= ~4096 at perdev spec
shapes — _allport_kernel_hbm keeps the run slots in HBM and streams every
touch through fixed VMEM tiles (the residency pattern proven by
hierarchical_reduce_scatter's HBM running sums + HBM-source remote copies);
run slot 0 is aliased to the output, message sizes are unchanged. Same static
program either way — the deadlock rule above is untouched.
"""
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import topology, util
from tpu_inference.kernels.collectives.hier_matmul_reduce_scatter import \
    validate_inputs  # same input contract as the hier kernel

_COLLECTIVE_ID = 4


def _band_bounds(k, num_dims):
    """K-column bands. (A row-band A/B measured within noise: the p2 stage
    is wire-bound on the SHARED chip ICI ports — 3P per chip port at D=2 —
    so banding geometry doesn't move it; columns won 4/5 Ms by 1-3%.)"""
    fcs = ((k // num_dims + 127) // 128) * 128
    return [(min(b * fcs, k), min(fcs, k - min(b * fcs, k)))
            for b in range(num_dims)]


def _allport_kernel(
    a_hbm,
    w_hbm,
    out_hbm,
    recv1_hbm,
    recv2_hbm,
    w_vmem,
    a_vmem,
    run_vmem,
    acc_vmem,
    recv_vmem,
    prod_vmem,
    sem_a,
    sem_recv,
    sem_out,
    sems_w,
    sems1,
    sems2,
    *,
    sched,
    prog,
    num_dims,
    m_per,
    bk,
    bk_merge,
    whole_a,
    use_prod,
    band_bounds,
    axis_name,
    ablate,
):
    k = w_vmem.shape[1]
    num_kt = k // bk
    my_id = lax.axis_index(axis_name)
    bit = lax.rem(my_id, 2)
    my_chip = lax.div(my_id, 2)
    twin = my_id + 1 - 2 * bit

    partners = [twin]
    for d in range(num_dims):
        partners.append((my_chip ^ (1 << d)) * 2 + bit)
    util.local_barrier_logical(partners)

    # w tile loads start immediately; each tile is waited once, at first use.
    for kt_i in range(num_kt):
        pltpu.make_async_copy(w_hbm.at[:, pl.ds(kt_i * bk, bk)],
                              w_vmem.at[:, pl.ds(kt_i * bk, bk)],
                              sems_w.at[kt_i]).start()
    w_waited = [False] * num_kt

    # a-chunk prefetch pipeline: chunk order = the program's compute order.
    compute_chunks = []  # (kind, l) in program order
    for op in prog:
        if op[0] in ("compute_tw", "compute_own"):
            compute_chunks.append(op)

    def chunk_rows(op):
        kind, l = op
        q = my_chip ^ l
        return q * 2 + (1 - bit) if kind == "compute_tw" else q * 2 + bit

    def a_load(i):
        return pltpu.make_async_copy(
            a_hbm.at[pl.ds(chunk_rows(compute_chunks[i]) * m_per, m_per), :],
            a_vmem.at[i % 2], sem_a)

    # whole_a (small M): ONE load of the full local a, and ONE weight sweep —
    # per w tile a single [M, bk] dot whose rows are then scattered into the
    # run/acc slots. The v3 per-chunk form re-sweeps all of w through the MXU
    # 2*num_chips times; at m_per <= 64 that re-sweep IS the floor (measured
    # 82.6 us "compute" vs 11.4 us for the same FLOPs as one matmul @M=256).
    if whole_a:
        a_all = pltpu.make_async_copy(a_hbm, a_vmem, sem_a)
        a_all.start()
        a_all.wait()
        for kt_i in range(num_kt):
            pltpu.make_async_copy(w_hbm.at[:, pl.ds(kt_i * bk, bk)],
                                  w_vmem.at[:, pl.ds(kt_i * bk, bk)],
                                  sems_w.at[kt_i]).wait()
            w_waited[kt_i] = True
            kt = kt_i * bk
            if use_prod:
                prod_vmem[:, :] = jnp.dot(a_vmem[:],
                                          w_vmem[:, pl.ds(kt, bk)],
                                          preferred_element_type=jnp.float32)
            for c_op in compute_chunks:
                kind_c, l_c = c_op
                rows = chunk_rows(c_op) * m_per
                if use_prod:
                    sl = prod_vmem[pl.ds(rows, m_per), :]
                else:  # tile-major per-chunk dot, w tile stationary
                    sl = jnp.dot(a_vmem[pl.ds(rows, m_per)],
                                 w_vmem[:, pl.ds(kt, bk)],
                                 preferred_element_type=jnp.float32)
                if kind_c == "compute_tw":
                    run_vmem[l_c, slice(None), pl.ds(kt, bk)] = sl.astype(
                        run_vmem.dtype)
                else:  # compute_own -> per-label acc slot (bf16, one extra
                    #      rounding of the own addend; gated by the err check)
                    acc_vmem[l_c, slice(None), pl.ds(kt, bk)] = sl.astype(
                        acc_vmem.dtype)
    else:
        a_load(0).start()
    compute_i = [0]  # python cell: index of the compute op being emitted

    def emit_compute(dst_ref, dst_is_acc, l):
        """dot a[chunk] @ w into dst (k-tiled); prefetch the next a chunk.

        The first chunk's tile loop is Python-unrolled so each w tile is
        waited exactly once, at first use; every later chunk uses a traced
        pl.loop so the fp32 dot values stay one-body-deep on the Mosaic
        stack (all w tiles are waited after chunk 0)."""
        i = compute_i[0]
        a_load(i).wait()
        if i + 1 < len(compute_chunks):
            a_load(i + 1).start()
        slot = i % 2

        def tile_body(kt):
            block = jnp.dot(a_vmem[slot], w_vmem[:, pl.ds(kt, bk)],
                            preferred_element_type=jnp.float32)
            if dst_is_acc:
                dst_ref[slice(None), pl.ds(kt, bk)] = block
            else:
                dst_ref[l, slice(None), pl.ds(kt, bk)] = block.astype(
                    run_vmem.dtype)

        if not all(w_waited):
            for kt_i in range(num_kt):
                if not w_waited[kt_i]:
                    pltpu.make_async_copy(w_hbm.at[:, pl.ds(kt_i * bk, bk)],
                                          w_vmem.at[:, pl.ds(kt_i * bk, bk)],
                                          sems_w.at[kt_i]).wait()
                    w_waited[kt_i] = True
                tile_body(kt_i * bk)
        else:

            @pl.loop(0, num_kt)
            def _tiles(t):
                tile_body(t * bk)

        compute_i[0] += 1

    def twin_op(l):
        return pltpu.make_async_remote_copy(
            src_ref=run_vmem.at[l],
            dst_ref=recv1_hbm.at[l],
            send_sem=sems1.at[l],
            recv_sem=sems1.at[l],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def p2_op(key):
        s, b, op_i = key
        dim = (b + s) % num_dims
        partner = (my_chip ^ (1 << dim)) * 2 + bit
        start, width = band_bounds[b]
        slot = sched["slot"][key]
        return pltpu.make_async_remote_copy(
            src_ref=run_vmem.at[sched["sends"][key],
                                slice(None),
                                pl.ds(start, width)],
            dst_ref=recv2_hbm.at[slot, slice(None), pl.ds(0, width)],
            send_sem=sems2.at[slot],
            recv_sem=sems2.at[slot],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    # ablate (instrumentation only, timing-ladder attribution; results are
    # WRONG for ablate != 3): 1 = computes only, 2 = + twin exchange,
    # 4 = full minus the p2 merge arithmetic (waits still consume their
    # symmetric signals), 3 = full. Ops are skipped identically on every
    # device (no deadlock).
    skip = {
        1: ("twin_start", "twin_wait", "twin_merge", "p2_start", "p2_wait",
            "p2_merge"),
        2: ("p2_start", "p2_wait", "p2_merge"),
        4: ("p2_merge", ),
        3: (),
    }[ablate]

    for op in prog:
        kind = op[0]
        if kind in skip:
            continue
        if kind == "compute_tw":
            if not whole_a:
                with jax.named_scope(f"p1_tw_l{op[1]}"):
                    emit_compute(run_vmem, False, op[1])
        elif kind == "twin_start":
            twin_op(op[1]).start()
        elif kind == "compute_own":
            if not whole_a:
                with jax.named_scope(f"p1_own_l{op[1]}"):
                    emit_compute(acc_vmem, True, op[1])
        elif kind == "twin_wait":
            with jax.named_scope(f"p1_wait_l{op[1]}"):
                twin_op(op[1]).wait()
        elif kind == "twin_merge":
            l = op[1]
            with jax.named_scope(f"p1_merge_l{l}"):
                # Pipelined loads: tile kt+1's HBM->VMEM copy flies while
                # tile kt merges (the serial start;wait form exposed the DMA
                # latency ~16x per call).
                nkt_m = k // bk_merge

                def tw_load(kt_i, tile):
                    return pltpu.make_async_copy(
                        recv1_hbm.at[l, slice(None),
                                     pl.ds(kt_i * bk_merge, bk_merge)],
                        recv_vmem.at[tile, slice(None), pl.ds(0, bk_merge)],
                        sem_recv.at[tile])

                tw_load(0, 0).start()
                for kt_i in range(nkt_m):
                    tile = kt_i % 2
                    if kt_i + 1 < nkt_m:
                        tw_load(kt_i + 1, (kt_i + 1) % 2).start()
                    tw_load(kt_i, tile).wait()
                    kt = kt_i * bk_merge
                    own = (acc_vmem[l, slice(None), pl.ds(kt, bk_merge)]
                           if whole_a else acc_vmem[:, pl.ds(kt, bk_merge)])
                    merged = (own.astype(jnp.float32) +
                              recv_vmem[tile, :, pl.ds(0, bk_merge)].astype(
                                  jnp.float32))
                    run_vmem[l, slice(None),
                             pl.ds(kt, bk_merge)] = merged.astype(
                                 run_vmem.dtype)
        elif kind == "p2_start":
            s, b, op_i = op[1]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_send"):
                p2_op(op[1]).start()
        elif kind == "p2_wait":
            s, b, op_i = op[1]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_wait"):
                p2_op(op[1]).wait()
        elif kind == "p2_merge":
            key = op[1]
            s, b, op_i = key
            start, width = band_bounds[b]
            target = sched["merges"][key]
            slot = sched["slot"][key]
            subtiles = [(st, min(bk_merge, width - st))
                        for st in range(0, width, bk_merge)]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_merge"):

                def p2_load(idx, tile):
                    st, sw = subtiles[idx]
                    return pltpu.make_async_copy(
                        recv2_hbm.at[slot, slice(None), pl.ds(st, sw)],
                        recv_vmem.at[tile, slice(None), pl.ds(0, sw)],
                        sem_recv.at[tile])

                p2_load(0, 0).start()
                for idx, (st, sw) in enumerate(subtiles):
                    tile = idx % 2
                    if idx + 1 < len(subtiles):
                        p2_load(idx + 1, (idx + 1) % 2).start()
                    p2_load(idx, tile).wait()
                    merged = (
                        run_vmem[target, :, pl.ds(start + st, sw)].astype(
                            jnp.float32) +
                        recv_vmem[tile, :, pl.ds(0, sw)].astype(jnp.float32))
                    run_vmem[target, slice(None),
                             pl.ds(start + st, sw)] = merged.astype(
                                 run_vmem.dtype)
        elif kind == "out":
            with jax.named_scope("out"):
                o_copy = pltpu.make_async_copy(run_vmem.at[0], out_hbm,
                                               sem_out)
                o_copy.start()
                o_copy.wait()


def _allport_kernel_hbm(
    a_hbm,
    w_hbm,
    out_hbm,
    recv1_hbm,
    recv2_hbm,
    run_hbm,
    w_vmem,
    a_vmem,
    acc_vmem,
    recv_vmem,
    runld_vmem,
    store_vmem,
    sem_a,
    sem_recv,
    sem_runld,
    sem_store,
    sems_w,
    sems1,
    sems2,
    *,
    sched,
    prog,
    num_dims,
    m_per,
    bk,
    bk_merge,
    whole_a,
    use_prod,
    band_bounds,
    axis_name,
    ablate,
):
    """HBM-resident run slots: the same static program as _allport_kernel,
    with every run-slot touch streamed through fixed VMEM tiles (2-slot store
    staging + 2-slot RMW loads). Slot 0 is ALIASED to out_hbm so the trailing
    'out' copy drops. All added DMA starts/waits are unconditional with static
    slot ids — the deadlock rule is preserved."""
    del bk_merge, whole_a, use_prod  # large-M path: chunked a, bk-tiled
    k = w_vmem.shape[1]
    num_kt = k // bk
    my_id = lax.axis_index(axis_name)
    bit = lax.rem(my_id, 2)
    my_chip = lax.div(my_id, 2)
    twin = my_id + 1 - 2 * bit

    partners = [twin]
    for d in range(num_dims):
        partners.append((my_chip ^ (1 << d)) * 2 + bit)
    util.local_barrier_logical(partners)

    for kt_i in range(num_kt):
        pltpu.make_async_copy(w_hbm.at[:, pl.ds(kt_i * bk, bk)],
                              w_vmem.at[:, pl.ds(kt_i * bk, bk)],
                              sems_w.at[kt_i]).start()
    w_waited = [False] * num_kt

    compute_chunks = []
    for op in prog:
        if op[0] in ("compute_tw", "compute_own"):
            compute_chunks.append(op)

    def chunk_rows(op):
        kind, l = op
        q = my_chip ^ l
        return q * 2 + (1 - bit) if kind == "compute_tw" else q * 2 + bit

    def a_load(i):
        return pltpu.make_async_copy(
            a_hbm.at[pl.ds(chunk_rows(compute_chunks[i]) * m_per, m_per), :],
            a_vmem.at[i % 2], sem_a)

    a_load(0).start()
    compute_i = [0]

    # Relative labels are static python ints, so slot 0's aliasing to the
    # output is a static ref choice, never a traced branch.
    def run_slice(l, col, width):
        if l == 0:
            return out_hbm.at[slice(None), pl.ds(col, width)]
        return run_hbm.at[l, slice(None), pl.ds(col, width)]

    def run_full(l):
        return out_hbm if l == 0 else run_hbm.at[l]

    # Store staging: slot reuse waits stay per-slot; DEPENDENCY waits are
    # per-label and happen only at the op that reads the label (twin/p2
    # sends, RMW loads, program end) — a merge never stalls on its own
    # stores. All bookkeeping is Python-side and static.
    pending = [None, None]           # slot -> in-flight cp
    pending_label = [None, None]     # slot -> label of that cp
    store_i = [0]

    def stage_store(l, col, width, value):
        st = store_i[0] % 2
        if pending[st] is not None:
            pending[st].wait()
            pending[st] = None
        store_vmem[st, slice(None), pl.ds(0, width)] = value
        cp = pltpu.make_async_copy(
            store_vmem.at[st, slice(None), pl.ds(0, width)],
            run_slice(l, col, width), sem_store.at[st])
        cp.start()
        pending[st] = cp
        pending_label[st] = l
        store_i[0] += 1

    def drain_label(l):
        for st in (0, 1):
            if pending[st] is not None and pending_label[st] == l:
                pending[st].wait()
                pending[st] = None

    def drain_stores():
        for st in (0, 1):
            if pending[st] is not None:
                pending[st].wait()
                pending[st] = None

    def emit_compute(dst_is_acc, l):
        """dot a[chunk] @ w tile-wise; run-destined tiles stream to HBM and
        are DRAINED before return (the next program op reads the slot)."""
        i = compute_i[0]
        a_load(i).wait()
        if i + 1 < len(compute_chunks):
            a_load(i + 1).start()
        slot = i % 2
        for kt_i in range(num_kt):
            if not w_waited[kt_i]:
                pltpu.make_async_copy(w_hbm.at[:, pl.ds(kt_i * bk, bk)],
                                      w_vmem.at[:, pl.ds(kt_i * bk, bk)],
                                      sems_w.at[kt_i]).wait()
                w_waited[kt_i] = True
            kt = kt_i * bk
            block = jnp.dot(a_vmem[slot], w_vmem[:, pl.ds(kt, bk)],
                            preferred_element_type=jnp.float32)
            if dst_is_acc:
                acc_vmem[slice(None), pl.ds(kt, bk)] = block
            else:
                stage_store(l, kt, bk, block.astype(out_hbm.dtype))
        compute_i[0] += 1

    def twin_op(l):
        return pltpu.make_async_remote_copy(
            src_ref=run_full(l),
            dst_ref=recv1_hbm.at[l],
            send_sem=sems1.at[l],
            recv_sem=sems1.at[l],
            device_id=twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def p2_op(key):
        s, b, op_i = key
        dim = (b + s) % num_dims
        partner = (my_chip ^ (1 << dim)) * 2 + bit
        start, width = band_bounds[b]
        slot = sched["slot"][key]
        return pltpu.make_async_remote_copy(
            src_ref=run_slice(sched["sends"][key], start, width),
            dst_ref=recv2_hbm.at[slot, slice(None), pl.ds(0, width)],
            send_sem=sems2.at[slot],
            recv_sem=sems2.at[slot],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    skip = {
        1: ("twin_start", "twin_wait", "twin_merge", "p2_start", "p2_wait",
            "p2_merge"),
        2: ("p2_start", "p2_wait", "p2_merge"),
        4: ("p2_merge", ),
        3: (),
    }[ablate]

    for op in prog:
        kind = op[0]
        if kind in skip:
            continue
        if kind == "compute_tw":
            with jax.named_scope(f"p1_tw_l{op[1]}"):
                emit_compute(False, op[1])
        elif kind == "twin_start":
            drain_label(op[1])
            twin_op(op[1]).start()
        elif kind == "compute_own":
            with jax.named_scope(f"p1_own_l{op[1]}"):
                emit_compute(True, op[1])
        elif kind == "twin_wait":
            with jax.named_scope(f"p1_wait_l{op[1]}"):
                twin_op(op[1]).wait()
        elif kind == "twin_merge":
            l = op[1]
            with jax.named_scope(f"p1_merge_l{l}"):

                def tw_load(kt_i, tile):
                    return pltpu.make_async_copy(
                        recv1_hbm.at[l, slice(None), pl.ds(kt_i * bk, bk)],
                        recv_vmem.at[tile, slice(None), pl.ds(0, bk)],
                        sem_recv.at[tile])

                tw_load(0, 0).start()
                for kt_i in range(num_kt):
                    tile = kt_i % 2
                    if kt_i + 1 < num_kt:
                        tw_load(kt_i + 1, (kt_i + 1) % 2).start()
                    tw_load(kt_i, tile).wait()
                    kt = kt_i * bk
                    merged = (acc_vmem[:, pl.ds(kt, bk)] +
                              recv_vmem[tile, :, pl.ds(0, bk)].astype(
                                  jnp.float32))
                    stage_store(l, kt, bk, merged.astype(out_hbm.dtype))
        elif kind == "p2_start":
            s, b, op_i = op[1]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_send"):
                drain_label(sched["sends"][op[1]])
                p2_op(op[1]).start()
        elif kind == "p2_wait":
            s, b, op_i = op[1]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_wait"):
                p2_op(op[1]).wait()
        elif kind == "p2_merge":
            key = op[1]
            s, b, op_i = key
            start, width = band_bounds[b]
            target = sched["merges"][key]
            slot = sched["slot"][key]
            subtiles = [(st, min(bk, width - st))
                        for st in range(0, width, bk)]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_merge"):

                def p2_load(idx, tile):
                    st, sw = subtiles[idx]
                    return pltpu.make_async_copy(
                        recv2_hbm.at[slot, slice(None), pl.ds(st, sw)],
                        recv_vmem.at[tile, slice(None), pl.ds(0, sw)],
                        sem_recv.at[tile])

                def run_load(idx, tile):
                    st, sw = subtiles[idx]
                    return pltpu.make_async_copy(
                        run_slice(target, start + st, sw),
                        runld_vmem.at[tile, slice(None), pl.ds(0, sw)],
                        sem_runld.at[tile])

                drain_label(target)  # RMW: prior stores to this cell land
                p2_load(0, 0).start()
                run_load(0, 0).start()
                for idx, (st, sw) in enumerate(subtiles):
                    tile = idx % 2
                    if idx + 1 < len(subtiles):
                        p2_load(idx + 1, (idx + 1) % 2).start()
                        run_load(idx + 1, (idx + 1) % 2).start()
                    p2_load(idx, tile).wait()
                    run_load(idx, tile).wait()
                    merged = (
                        runld_vmem[tile, :, pl.ds(0, sw)].astype(jnp.float32)
                        + recv_vmem[tile, :, pl.ds(0, sw)].astype(
                            jnp.float32))
                    stage_store(target, start + st, sw,
                                merged.astype(out_hbm.dtype))
        elif kind == "out":
            pass  # run slot 0 IS out_hbm
    drain_stores()  # the final label-0 (== out) stores must land


def get_vmem_estimate_bytes(m_per, n_per, k, bk, num_chips, stage_w,
                            itemsize):
    return (n_per * k * itemsize            # w resident
            + 2 * m_per * n_per * itemsize  # a double buffer
            + num_chips * m_per * k * itemsize  # run slots
            + m_per * k * 4                 # fp32 acc
            + 2 * m_per * stage_w * itemsize  # staging tiles
            + m_per * bk * 4)               # dot value


def get_vmem_estimate_bytes_hbm(m_per, n_per, k, bk, itemsize):
    return (n_per * k * itemsize            # w resident
            + 2 * m_per * n_per * itemsize  # a double buffer
            + m_per * k * 4                 # fp32 acc
            + 2 * m_per * bk * itemsize     # recv tiles
            + 2 * m_per * bk * itemsize     # run RMW load tiles
            + 2 * m_per * bk * itemsize     # store staging tiles
            + m_per * bk * 4)               # dot value


# Measured scoped-VMEM ceiling (memory_space_assignment clamps requests here).
_VMEM_CAP_BYTES = 67043328


def _scratch_bytes(scratch_shapes):
    """Bytes of the VMEM scratches a config declares — Mosaic must fit these,
    so the limit is never allowed below it."""
    return sum(
        math.prod(s.shape) * jnp.dtype(s.dtype).itemsize
        for s in scratch_shapes
        if getattr(s, "memory_space", None) == pltpu.MemorySpace.VMEM)


def _limit_covering(estimate, scratch_shapes):
    """vmem_limit = estimate + 8 MiB, raised only if that fails to cover the
    scratch actually declared.

    The estimators predate the whole-a variants: they model neither the
    per-label wire-dtype acc slots nor the one-sweep `prod` buffer, so on that
    path `estimate + 8 MiB` can land BELOW the declaration and Mosaic E1001s a
    shape that fits (v33: compiles iff limit >= declared scratch).

    The raise is CONDITIONAL, not a `max`, and that is load-bearing: a limit
    that already covers its declaration keeps its exact previous value, because
    a too-HIGH limit is byte-identical but 9% slower at m_per=32 on this kernel
    (STATUS 2026-07-29) and every recorded llama70b number was measured at the
    old value. Verified by compiled-HLO fingerprint equality across the fix
    (`probe_compile_only.py --tag pre/post`): an unconditional `max` moved
    M=256/512/1024; this form moves nothing that already compiled.
    """
    limit = estimate + 8 * 1024 * 1024
    declared = _scratch_bytes(scratch_shapes)
    if limit < declared:
        limit = declared + 8 * 1024 * 1024
    return min(limit, _VMEM_CAP_BYTES)


def allport_matmul_reduce_scatter(
    a,
    w,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
    ablate: int = 3,
    force_hbm: bool = None,
):
    """reduce_scatter(a @ w, axis=0), all-port pipelined hierarchical kernel.

    a: [M, n_per] P(None, axis), w: [n_per, K] P(axis, None) ->
    out [M // tp, K] P(axis, None). Mesh from
    topology.make_collective_mesh('hier'). Residency is picked statically per
    shape: VMEM run slots below the scoped-VMEM budget, HBM-streamed above
    (force_hbm overrides, for same-shape A/B of the HBM tax).
    """
    tp_size = mesh.shape[axis_name]
    validate_inputs(a, w, tp_size)
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    if num_dims == 0:
        raise ValueError("allport kernel needs >= 2 chips")
    m = a.shape[0]
    k = w.shape[1]
    m_per = m // tp_size
    n_per_est = a.shape[1] // tp_size
    band_bounds = _band_bounds(k, num_dims)
    band_w = max(w_ for _, w_ in band_bounds)
    sched, prog = topology.allport_program(num_dims)
    bk_v = next(c for c in (2048, 1024, 512, 256, 128) if k % c == 0)
    use_hbm = force_hbm
    if use_hbm is None:
        use_hbm = (get_vmem_estimate_bytes(
            m_per, n_per_est, k, bk_v, num_chips, max(bk_v, band_w),
            a.dtype.itemsize) + 8 * 1024 * 1024 > _VMEM_CAP_BYTES)
    # HBM mode: the staging/RMW buffers scale with bk and the wire messages
    # don't — pick the largest bk whose whole footprint still fits.
    if use_hbm:
        bk = next((c for c in (2048, 1024, 512, 256, 128)
                   if k % c == 0 and get_vmem_estimate_bytes_hbm(
                       m_per, n_per_est, k, c, a.dtype.itemsize) +
                   8 * 1024 * 1024 <= _VMEM_CAP_BYTES), None)
        if bk is None:
            # The bk-independent terms alone (resident w, the a double buffer,
            # the fp32 accumulator) exceed the budget, so no tiling saves it.
            floor = get_vmem_estimate_bytes_hbm(m_per, n_per_est, k, 128,
                                                a.dtype.itemsize)
            raise ValueError(
                f"allport MM-RS does not fit at m={m}, n_per={n_per_est}, "
                f"k={k}: the smallest bk (128) still needs "
                f"{floor / 2**20:.1f} MiB of VMEM, over the "
                f"{(_VMEM_CAP_BYTES - 8 * 1024 * 1024) / 2**20:.1f} MiB "
                "budget. Reduce m or n_per, or use the ring kernel.")
    else:
        bk = bk_v
    # Small M (VMEM path): one weight sweep + whole-a residency + full-width
    # merges — the floor is MXU weight re-sweeps and DMA-op latency, not
    # bytes (measured @256: "compute" 82.6 us vs 11.4 us of matmul).
    whole_a = (not use_hbm) and m_per <= 64
    bk_merge = k if whole_a else bk
    use_prod = False
    if whole_a:
        # One-sweep variant A (prod): a single [m, bk] dot per w tile, rows
        # then scattered. Stack cost ≈ a value + w tile + prod value, plus
        # the prod scratch — shrink bk until it fits; if nothing fits, fall
        # back to variant B (tile-major per-chunk dots against a stationary
        # w tile — no prod buffer, weights stay loaded across chunks).
        isz = a.dtype.itemsize
        fixed = (n_per_est * k * isz + m * n_per_est * isz +
                 2 * num_chips * m_per * k * isz + 2 * m_per * k * isz)
        for c in (2048, 1024, 512, 256, 128):
            if k % c == 0 and (fixed + 2 * m * c * 4 + m * n_per_est * isz +
                               n_per_est * c * isz + 4 * 1024 * 1024
                               <= _VMEM_CAP_BYTES):
                bk = c
                use_prod = True
                break

    def per_device(a_local, w_local):
        n_per = a_local.shape[1]
        hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
        if use_hbm:
            scratch_shapes = (
                pltpu.VMEM(w_local.shape, w_local.dtype),
                pltpu.VMEM((2, m_per, n_per), a_local.dtype),
                pltpu.VMEM((m_per, k), jnp.float32),
                pltpu.VMEM((2, m_per, bk), a_local.dtype),
                pltpu.VMEM((2, m_per, bk), a_local.dtype),
                pltpu.VMEM((2, m_per, bk), a_local.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((k // bk, )),
                pltpu.SemaphoreType.DMA((num_chips, )),
                pltpu.SemaphoreType.DMA((sched["n_slots"], )),
            )
            out_shape = (
                jax.ShapeDtypeStruct((m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((num_chips, m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((sched["n_slots"], m_per, band_w),
                                     a_local.dtype),
                jax.ShapeDtypeStruct((num_chips, m_per, k), a_local.dtype),
            )
            out_specs = (hbm, hbm, hbm, hbm)
            kernel_fn = _allport_kernel_hbm
            vmem_limit = _limit_covering(
                get_vmem_estimate_bytes_hbm(m_per, n_per, k, bk,
                                            a_local.dtype.itemsize),
                scratch_shapes)
        else:
            a_shape = ((m, n_per) if whole_a else (2, m_per, n_per))
            # whole_a: per-label own slots (wire dtype, one extra rounding);
            # chunked: one fp32 accumulator reused per pair.
            acc_spec = (pltpu.VMEM((num_chips, m_per, k), a_local.dtype)
                        if whole_a else pltpu.VMEM((m_per, k), jnp.float32))
            recv_w = max(bk_merge, band_w)
            prod_shape = ((m, bk) if whole_a and use_prod
                          else (8, 128))  # dummy if unused
            scratch_shapes = (
                pltpu.VMEM(w_local.shape, w_local.dtype),
                pltpu.VMEM(a_shape, a_local.dtype),
                pltpu.VMEM((num_chips, m_per, k), a_local.dtype),
                acc_spec,
                pltpu.VMEM((2, m_per, recv_w), a_local.dtype),
                pltpu.VMEM(prod_shape, jnp.float32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((k // bk, )),
                pltpu.SemaphoreType.DMA((num_chips, )),
                pltpu.SemaphoreType.DMA((sched["n_slots"], )),
            )
            out_shape = (
                jax.ShapeDtypeStruct((m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((num_chips, m_per, k), a_local.dtype),
                jax.ShapeDtypeStruct((sched["n_slots"], m_per, band_w),
                                     a_local.dtype),
            )
            out_specs = (hbm, hbm, hbm)
            kernel_fn = _allport_kernel
            # The estimator predates the whole-a variants: it models neither the
            # per-label bf16 acc slots nor the one-sweep prod buffer, so on that
            # path it can land BELOW the scratch actually declared and Mosaic
            # E1001s on a shape that fits. The limit must still cover the
            # declaration, so take the larger of the two.
            a_bytes = (m if whole_a else 2 * m_per) * n_per
            vmem_limit = _limit_covering(
                get_vmem_estimate_bytes(m_per, n_per, k, bk, num_chips,
                                        recv_w, a_local.dtype.itemsize) +
                (a_bytes - 2 * m_per * n_per) * a_local.dtype.itemsize,
                scratch_shapes)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm],
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=(1, ),
        )
        kernel = functools.partial(
            kernel_fn,
            sched=sched,
            prog=prog,
            num_dims=num_dims,
            m_per=m_per,
            bk=bk,
            bk_merge=bk_merge,
            whole_a=whole_a,
            use_prod=use_prod,
            band_bounds=band_bounds,
            axis_name=axis_name,
            ablate=ablate,
        )
        outs = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=vmem_limit,
            ),
            name=f"allport_matmul_reduce_scatter_d{num_dims}",
        )(a_local, w_local)
        return outs[0]

    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec(None, axis_name),
                  jax.sharding.PartitionSpec(axis_name, None)),
        out_specs=jax.sharding.PartitionSpec(axis_name, None),
        check_vma=False,
    )(a, w)

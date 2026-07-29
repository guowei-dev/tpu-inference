# SPDX-License-Identifier: Apache-2.0
"""All-port fused matmul reduce-scatter (hypercube schedule, ring-style
pipelining).

Twin-pair D2D + concurrent-dims hypercube: band b trades dim (b+s) % D at
round s, so every ICI axis is busy every round. The schedule carries no phase
barriers — the emission order comes from topology.allport_program: twin pairs
in popcount-descending order (the first twin merge lights a round-0 send on
EVERY dim), each merge immediately starts the sends it unlocks, and ladder
waits are hoisted between pair computes. Round r+1's send of a region depends
only on that region's round-r merge, never on the whole round.

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

Layout contract: logical id == chip * 2 + core (topology.make_collective_mesh);
wire dtype = input dtype; fp32 local accumulate, at wire dtype on the small-M
whole-a path; hops = 1 + log2(num_chips). collective_id = 4.

Residency: the run slots keep all 2^D relative-label cells at full K width, so
they scale as 2^D * m_per * K and are what caps M. Below the scoped-VMEM budget
they live in VMEM; above it they live in HBM and every touch is streamed
through fixed VMEM tiles, with run slot 0 aliased to the output. The choice is
static, per shape, and picks a run-slot implementation (`_VmemRunSlots` /
`_HbmRunSlots`) — `_emit` walks the one program either way, so the two
residencies cannot drift apart.
"""
import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import topology, util

P = jax.sharding.PartitionSpec

_COLLECTIVE_ID = 4

# memory_space_assignment clamps a scoped request to the chip's VMEM less a
# 64 KiB reservation; asking for more than that is what it refuses.
_VMEM_RESERVED_BYTES = 64 * 1024


def _vmem_cap_bytes():
    return pltpu.get_tpu_info().vmem_capacity_bytes - _VMEM_RESERVED_BYTES


# Column-tile candidates, largest first; every search below must agree.
_TILE_CANDIDATES = (2048, 1024, 512, 256, 128)


def _band_bounds(k, num_dims):
    """K-column bands, one per hypercube dim."""
    fcs = ((k // num_dims + 127) // 128) * 128
    return [(min(b * fcs, k), min(fcs, k - min(b * fcs, k)))
            for b in range(num_dims)]


def _validate_inputs(a, w, tp_size):
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("a and w must be 2-D")
    if a.dtype != w.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {w.dtype}")
    m, n_per = a.shape
    n_per_w, k = w.shape
    if n_per != n_per_w:
        raise ValueError(f"contraction mismatch: {n_per} vs {n_per_w}")
    if tp_size % 2:
        raise ValueError(f"tp_size {tp_size} must pair 2 cores per chip")
    num_chips = tp_size // 2
    if num_chips & (num_chips - 1):
        raise ValueError(f"num_chips {num_chips} must be a power of two")
    if num_chips < 2:
        raise ValueError(f"num_chips {num_chips} must be >= 2")
    if m % tp_size:
        raise ValueError(f"m {m} must be divisible by tp_size {tp_size}")
    if (m // tp_size) % 8:
        raise ValueError(f"m per device {m // tp_size} must be sublane-"
                         f"aligned (divisible by 8)")
    if k % 128:
        raise ValueError(f"k {k} must be divisible by 128")
    num_dims = int(math.log2(num_chips))
    if any(width == 0 for _, width in _band_bounds(k, num_dims)):
        raise ValueError(f"k {k} does not split into {num_dims} non-empty "
                         f"128-column bands")


@dataclasses.dataclass(frozen=True)
class _Config:
    """Static shape/schedule choices. Never a pytree — trace-time Python."""
    sched: Any
    prog: Any
    num_dims: int
    m_per: int
    bk: int
    bk_merge: int
    whole_a: bool
    band_bounds: Any
    axis_name: str


@dataclasses.dataclass(frozen=True)
class _Refs:
    """The refs every stage shares, bundled so a stage takes one argument."""
    a_hbm: Any
    w_hbm: Any
    out_hbm: Any
    recv1_hbm: Any
    recv2_hbm: Any
    w_vmem: Any
    a_vmem: Any
    acc_vmem: Any
    recv_vmem: Any
    sem_a: Any
    sem_recv: Any
    sems_w: Any
    sems1: Any
    sems2: Any


class _Stage:
    """Trace-time state shared by both residencies: ids, the w-tile waits and
    the a-chunk prefetch pipeline."""

    def __init__(self, refs, cfg):
        self.refs, self.cfg = refs, cfg
        self.k = refs.w_vmem.shape[1]
        self.num_kt = self.k // cfg.bk
        self.my_id = lax.axis_index(cfg.axis_name)
        self.bit = lax.rem(self.my_id, 2)
        self.my_chip = lax.div(self.my_id, 2)
        self.twin = self.my_id + 1 - 2 * self.bit
        self.chunks = [
            op for op in cfg.prog if op[0] in ("compute_tw", "compute_own")
        ]
        self.w_waited = [False] * self.num_kt
        self.compute_i = 0

    def barrier(self):
        partners = [self.twin]
        for d in range(self.cfg.num_dims):
            partners.append((self.my_chip ^ (1 << d)) * 2 + self.bit)
        util.local_barrier_logical(partners)

    def w_tile(self, kt_i):
        return pltpu.make_async_copy(
            self.refs.w_hbm.at[:, pl.ds(kt_i * self.cfg.bk, self.cfg.bk)],
            self.refs.w_vmem.at[:, pl.ds(kt_i * self.cfg.bk, self.cfg.bk)],
            self.refs.sems_w.at[kt_i])

    def start_w_loads(self):
        """w tile loads start immediately; each tile is waited once, at first
        use."""
        for kt_i in range(self.num_kt):
            self.w_tile(kt_i).start()

    def wait_w(self, kt_i):
        if not self.w_waited[kt_i]:
            self.w_tile(kt_i).wait()
            self.w_waited[kt_i] = True

    def chunk_rows(self, op):
        kind, lbl = op
        q = self.my_chip ^ lbl
        return q * 2 + (1 -
                        self.bit) if kind == "compute_tw" else q * 2 + self.bit

    def a_load(self, i):
        return pltpu.make_async_copy(
            self.refs.a_hbm.at[pl.ds(
                self.chunk_rows(self.chunks[i]) *
                self.cfg.m_per, self.cfg.m_per), :],
            self.refs.a_vmem.at[i % 2], self.refs.sem_a)

    def next_a_chunk(self):
        """Wait this chunk's a, start the next one's, return its buffer slot."""
        i = self.compute_i
        self.a_load(i).wait()
        if i + 1 < len(self.chunks):
            self.a_load(i + 1).start()
        self.compute_i += 1
        return i % 2

    def twin_copy(self, src_ref, lbl):
        return pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=self.refs.recv1_hbm.at[lbl],
            send_sem=self.refs.sems1.at[lbl],
            recv_sem=self.refs.sems1.at[lbl],
            device_id=self.twin,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def p2_copy(self, src_ref, key):
        s, b, _ = key
        dim = (b + s) % self.cfg.num_dims
        partner = (self.my_chip ^ (1 << dim)) * 2 + self.bit
        _, width = self.cfg.band_bounds[b]
        slot = self.cfg.sched["slot"][key]
        return pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=self.refs.recv2_hbm.at[slot,
                                           slice(None),
                                           pl.ds(0, width)],
            send_sem=self.refs.sems2.at[slot],
            recv_sem=self.refs.sems2.at[slot],
            device_id=partner,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    def recv_tile(self, src_ref, tile, width):
        return pltpu.make_async_copy(
            src_ref, self.refs.recv_vmem.at[tile,
                                            slice(None),
                                            pl.ds(0, width)],
            self.refs.sem_recv.at[tile])


class _VmemRunSlots:
    """Run slots resident in VMEM: every touch is a direct ref access."""

    def __init__(self, stage, run_vmem, prod_vmem, sem_out):
        self.stage, self.run = stage, run_vmem
        self.prod, self.sem_out = prod_vmem, sem_out

    def full(self, lbl):
        return self.run.at[lbl]

    def send_slice(self, lbl, start, width):
        return self.run.at[lbl, slice(None), pl.ds(start, width)]

    def before_send(self, lbl):
        del lbl  # writes to a VMEM slot have landed by the time we send it

    def compute(self, dst_is_acc, lbl):
        """dot a[chunk] @ w into the run slot or the accumulator (k-tiled).

        The first chunk's tile loop is Python-unrolled so each w tile is waited
        exactly once, at first use; every later chunk uses a traced pl.loop so
        the fp32 dot values stay one-body-deep on the Mosaic stack (all w tiles
        are waited after chunk 0)."""
        stage, cfg = self.stage, self.stage.cfg
        slot = stage.next_a_chunk()

        def tile_body(kt):
            block = jnp.dot(stage.refs.a_vmem[slot],
                            stage.refs.w_vmem[:, pl.ds(kt, cfg.bk)],
                            preferred_element_type=jnp.float32)
            if dst_is_acc:
                stage.refs.acc_vmem[slice(None), pl.ds(kt, cfg.bk)] = block
            else:
                self.run[lbl, slice(None),
                         pl.ds(kt, cfg.bk)] = block.astype(self.run.dtype)

        if not all(stage.w_waited):
            for kt_i in range(stage.num_kt):
                stage.wait_w(kt_i)
                tile_body(kt_i * cfg.bk)
        else:

            @pl.loop(0, stage.num_kt)
            def _tiles(t):
                tile_body(t * cfg.bk)

    def compute_all(self):
        """whole_a (small M): ONE load of the full local a, and ONE weight
        sweep — per w tile a single [M, bk] dot whose rows are then scattered
        into the run/acc slots. Per-chunk dots re-sweep all of w through the
        MXU 2*num_chips times, which at m_per <= 64 dominates the actual
        FLOPs."""
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs
        a_all = pltpu.make_async_copy(refs.a_hbm, refs.a_vmem, refs.sem_a)
        a_all.start()
        a_all.wait()
        for kt_i in range(stage.num_kt):
            stage.wait_w(kt_i)
            kt = kt_i * cfg.bk
            self.prod[:, :] = jnp.dot(refs.a_vmem[:],
                                      refs.w_vmem[:, pl.ds(kt, cfg.bk)],
                                      preferred_element_type=jnp.float32)
            for c_op in stage.chunks:
                kind_c, l_c = c_op
                rows = stage.chunk_rows(c_op) * cfg.m_per
                sl = self.prod[pl.ds(rows, cfg.m_per), :]
                if kind_c == "compute_tw":
                    self.run[l_c, slice(None),
                             pl.ds(kt, cfg.bk)] = sl.astype(self.run.dtype)
                else:  # compute_own -> per-label own slot at wire dtype
                    #      (one extra rounding of the own addend)
                    refs.acc_vmem[l_c, slice(None),
                                  pl.ds(kt, cfg.bk)] = sl.astype(
                                      refs.acc_vmem.dtype)

    def merge_twin(self, lbl):
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs
        # Double-buffered: tile kt+1's HBM->VMEM copy flies while tile kt
        # merges, so the DMA latency is paid once, not per tile.
        nkt_m = stage.k // cfg.bk_merge

        def load(kt_i, tile):
            return stage.recv_tile(
                refs.recv1_hbm.at[lbl,
                                  slice(None),
                                  pl.ds(kt_i * cfg.bk_merge, cfg.bk_merge)],
                tile, cfg.bk_merge)

        load(0, 0).start()
        for kt_i in range(nkt_m):
            tile = kt_i % 2
            if kt_i + 1 < nkt_m:
                load(kt_i + 1, (kt_i + 1) % 2).start()
            load(kt_i, tile).wait()
            kt = kt_i * cfg.bk_merge
            own = (refs.acc_vmem[lbl,
                                 slice(None),
                                 pl.ds(kt, cfg.bk_merge)]
                   if cfg.whole_a else refs.acc_vmem[:,
                                                     pl.ds(kt, cfg.bk_merge)])
            merged = (own.astype(jnp.float32) +
                      refs.recv_vmem[tile, :, pl.ds(0, cfg.bk_merge)].astype(
                          jnp.float32))
            self.run[lbl, slice(None),
                     pl.ds(kt, cfg.bk_merge)] = merged.astype(self.run.dtype)

    def merge_p2(self, key):
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs
        _, b, _ = key
        start, width = cfg.band_bounds[b]
        target = cfg.sched["merges"][key]
        slot = cfg.sched["slot"][key]
        subtiles = [(st, min(cfg.bk_merge, width - st))
                    for st in range(0, width, cfg.bk_merge)]

        def load(idx, tile):
            st, sw = subtiles[idx]
            return stage.recv_tile(
                refs.recv2_hbm.at[slot, slice(None),
                                  pl.ds(st, sw)], tile, sw)

        load(0, 0).start()
        for idx, (st, sw) in enumerate(subtiles):
            tile = idx % 2
            if idx + 1 < len(subtiles):
                load(idx + 1, (idx + 1) % 2).start()
            load(idx, tile).wait()
            merged = (
                self.run[target, slice(None),
                         pl.ds(start + st, sw)].astype(jnp.float32) +
                refs.recv_vmem[tile, slice(None),
                               pl.ds(0, sw)].astype(jnp.float32))
            self.run[target, slice(None),
                     pl.ds(start + st, sw)] = merged.astype(self.run.dtype)

    def finish(self):
        copy = pltpu.make_async_copy(self.run.at[0], self.stage.refs.out_hbm,
                                     self.sem_out)
        copy.start()
        copy.wait()


class _HbmRunSlots:
    """Run slots resident in HBM: every touch streams through fixed VMEM
    tiles (2-slot store staging + 2-slot RMW loads). Slot 0 is ALIASED to
    out_hbm so the trailing output copy drops.

    Store staging: slot reuse waits stay per-slot; DEPENDENCY waits are
    per-label and happen only at the op that reads the label (twin/p2 sends,
    RMW loads, program end) — a merge never stalls on its own stores. All
    bookkeeping is Python-side and static, so the emitted sequence of starts
    and waits is still identical on every device.
    """

    def __init__(self, stage, run_hbm, runld_vmem, store_vmem, sem_runld,
                 sem_store):
        self.stage, self.run = stage, run_hbm
        self.runld, self.store = runld_vmem, store_vmem
        self.sem_runld, self.sem_store = sem_runld, sem_store
        self.pending = [None, None]  # slot -> in-flight store copy
        self.pending_label = [None, None]
        self.store_i = 0

    # Relative labels are static python ints, so slot 0's aliasing to the
    # output is a static ref choice, never a traced branch.
    def full(self, lbl):
        return self.stage.refs.out_hbm if lbl == 0 else self.run.at[lbl]

    def send_slice(self, lbl, start, width):
        if lbl == 0:
            return self.stage.refs.out_hbm.at[slice(None), pl.ds(start, width)]
        return self.run.at[lbl, slice(None), pl.ds(start, width)]

    def _stage_store(self, lbl, col, width, value):
        st = self.store_i % 2
        if self.pending[st] is not None:
            self.pending[st].wait()
            self.pending[st] = None
        self.store[st, slice(None), pl.ds(0, width)] = value
        copy = pltpu.make_async_copy(
            self.store.at[st, slice(None), pl.ds(0, width)],
            self.send_slice(lbl, col, width), self.sem_store.at[st])
        copy.start()
        self.pending[st] = copy
        self.pending_label[st] = lbl
        self.store_i += 1

    def before_send(self, lbl):
        for st in (0, 1):
            if self.pending[st] is not None and self.pending_label[st] == lbl:
                self.pending[st].wait()
                self.pending[st] = None

    def compute(self, dst_is_acc, lbl):
        """dot a[chunk] @ w tile-wise; run-destined tiles stream to HBM."""
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs
        slot = stage.next_a_chunk()
        for kt_i in range(stage.num_kt):
            stage.wait_w(kt_i)
            kt = kt_i * cfg.bk
            block = jnp.dot(refs.a_vmem[slot],
                            refs.w_vmem[:, pl.ds(kt, cfg.bk)],
                            preferred_element_type=jnp.float32)
            if dst_is_acc:
                refs.acc_vmem[slice(None), pl.ds(kt, cfg.bk)] = block
            else:
                self._stage_store(lbl, kt, cfg.bk,
                                  block.astype(refs.out_hbm.dtype))

    def merge_twin(self, lbl):
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs

        def load(kt_i, tile):
            return stage.recv_tile(
                refs.recv1_hbm.at[lbl,
                                  slice(None),
                                  pl.ds(kt_i * cfg.bk, cfg.bk)], tile, cfg.bk)

        load(0, 0).start()
        for kt_i in range(stage.num_kt):
            tile = kt_i % 2
            if kt_i + 1 < stage.num_kt:
                load(kt_i + 1, (kt_i + 1) % 2).start()
            load(kt_i, tile).wait()
            kt = kt_i * cfg.bk
            merged = (
                refs.acc_vmem[:, pl.ds(kt, cfg.bk)] +
                refs.recv_vmem[tile, :, pl.ds(0, cfg.bk)].astype(jnp.float32))
            self._stage_store(lbl, kt, cfg.bk,
                              merged.astype(refs.out_hbm.dtype))

    def merge_p2(self, key):
        stage, cfg = self.stage, self.stage.cfg
        refs = stage.refs
        _, b, _ = key
        start, width = cfg.band_bounds[b]
        target = cfg.sched["merges"][key]
        slot = cfg.sched["slot"][key]
        subtiles = [(st, min(cfg.bk, width - st))
                    for st in range(0, width, cfg.bk)]

        def recv_load(idx, tile):
            st, sw = subtiles[idx]
            return stage.recv_tile(
                refs.recv2_hbm.at[slot, slice(None),
                                  pl.ds(st, sw)], tile, sw)

        def run_load(idx, tile):
            st, sw = subtiles[idx]
            return pltpu.make_async_copy(
                self.send_slice(target, start + st, sw),
                self.runld.at[tile, slice(None),
                              pl.ds(0, sw)], self.sem_runld.at[tile])

        self.before_send(target)  # RMW: prior stores to this cell must land
        recv_load(0, 0).start()
        run_load(0, 0).start()
        for idx, (st, sw) in enumerate(subtiles):
            tile = idx % 2
            if idx + 1 < len(subtiles):
                recv_load(idx + 1, (idx + 1) % 2).start()
                run_load(idx + 1, (idx + 1) % 2).start()
            recv_load(idx, tile).wait()
            run_load(idx, tile).wait()
            merged = (self.runld[tile, slice(None),
                                 pl.ds(0, sw)].astype(jnp.float32) +
                      refs.recv_vmem[tile, slice(None),
                                     pl.ds(0, sw)].astype(jnp.float32))
            self._stage_store(target, start + st, sw,
                              merged.astype(refs.out_hbm.dtype))

    def finish(self):
        for st in (0, 1):  # the final label-0 (== out) stores must land
            if self.pending[st] is not None:
                self.pending[st].wait()
                self.pending[st] = None


def _emit(stage, run_slots):
    """Walk topology.allport_program once.

    Both residencies walk this one program, so the REMOTE start/wait sequence
    stays device-invariant in either — the thing the deadlock argument rests
    on. Only the local staging DMAs differ between them.
    """
    cfg = stage.cfg
    stage.barrier()
    stage.start_w_loads()
    if cfg.whole_a:
        run_slots.compute_all()
    else:
        stage.a_load(0).start()

    for op in cfg.prog:
        kind = op[0]
        if kind == "compute_tw":
            if not cfg.whole_a:
                with jax.named_scope(f"p1_tw_l{op[1]}"):
                    run_slots.compute(False, op[1])
        elif kind == "twin_start":
            run_slots.before_send(op[1])
            stage.twin_copy(run_slots.full(op[1]), op[1]).start()
        elif kind == "compute_own":
            if not cfg.whole_a:
                with jax.named_scope(f"p1_own_l{op[1]}"):
                    run_slots.compute(True, op[1])
        elif kind == "twin_wait":
            with jax.named_scope(f"p1_wait_l{op[1]}"):
                stage.twin_copy(run_slots.full(op[1]), op[1]).wait()
        elif kind == "twin_merge":
            with jax.named_scope(f"p1_merge_l{op[1]}"):
                run_slots.merge_twin(op[1])
        elif kind in ("p2_start", "p2_wait"):
            s, b, op_i = op[1]
            label = cfg.sched["sends"][op[1]]
            start, width = cfg.band_bounds[b]
            src = run_slots.send_slice(label, start, width)
            if kind == "p2_start":
                with jax.named_scope(f"p2_s{s}b{b}k{op_i}_send"):
                    run_slots.before_send(label)
                    stage.p2_copy(src, op[1]).start()
            else:
                with jax.named_scope(f"p2_s{s}b{b}k{op_i}_wait"):
                    stage.p2_copy(src, op[1]).wait()
        elif kind == "p2_merge":
            s, b, op_i = op[1]
            with jax.named_scope(f"p2_s{s}b{b}k{op_i}_merge"):
                run_slots.merge_p2(op[1])
        elif kind == "out":
            with jax.named_scope("out"):
                run_slots.finish()


def _allport_kernel(
    # Inputs
    a_hbm,  # [m, n_per]
    w_hbm,  # [n_per, k]
    # Outputs
    out_hbm,  # [m_per, k]
    recv1_hbm,  # [num_chips, m_per, k] twin landing slots
    recv2_hbm,  # [n_slots, m_per, band_w] hypercube landing slots
    # Scratches
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
    cfg,
):
    refs = _Refs(a_hbm=a_hbm,
                 w_hbm=w_hbm,
                 out_hbm=out_hbm,
                 recv1_hbm=recv1_hbm,
                 recv2_hbm=recv2_hbm,
                 w_vmem=w_vmem,
                 a_vmem=a_vmem,
                 acc_vmem=acc_vmem,
                 recv_vmem=recv_vmem,
                 sem_a=sem_a,
                 sem_recv=sem_recv,
                 sems_w=sems_w,
                 sems1=sems1,
                 sems2=sems2)
    stage = _Stage(refs, cfg)
    _emit(stage, _VmemRunSlots(stage, run_vmem, prod_vmem, sem_out))


def _allport_kernel_hbm(
    # Inputs
    a_hbm,  # [m, n_per]
    w_hbm,  # [n_per, k]
    # Outputs
    out_hbm,  # [m_per, k] — also run slot 0
    recv1_hbm,  # [num_chips, m_per, k] twin landing slots
    recv2_hbm,  # [n_slots, m_per, band_w] hypercube landing slots
    run_hbm,  # [num_chips, m_per, k] run slots 1..
    # Scratches
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
    cfg,
):
    refs = _Refs(a_hbm=a_hbm,
                 w_hbm=w_hbm,
                 out_hbm=out_hbm,
                 recv1_hbm=recv1_hbm,
                 recv2_hbm=recv2_hbm,
                 w_vmem=w_vmem,
                 a_vmem=a_vmem,
                 acc_vmem=acc_vmem,
                 recv_vmem=recv_vmem,
                 sem_a=sem_a,
                 sem_recv=sem_recv,
                 sems_w=sems_w,
                 sems1=sems1,
                 sems2=sems2)
    stage = _Stage(refs, cfg)
    _emit(
        stage,
        _HbmRunSlots(stage, run_hbm, runld_vmem, store_vmem, sem_runld,
                     sem_store))


def get_vmem_estimate_bytes(m_per,
                            n_per,
                            k,
                            bk,
                            num_chips,
                            stage_w,
                            itemsize,
                            *,
                            m=None,
                            whole_a=False):
    """Resident VMEM bytes for the VMEM-residency kernel at these shapes.

    Drives the residency choice and the small-M variant. The vmem_limit handed
    to Mosaic is deliberately NOT this number — see the comment at its call
    site. `whole_a` keeps the whole local a and per-label wire-dtype own slots,
    and its dot value spans all m rows.
    """
    a_rows = m if whole_a else 2 * m_per
    acc = (num_chips * m_per * k * itemsize) if whole_a else (m_per * k * 4)
    dot_rows = m if whole_a else m_per
    # Under whole_a the dot's operands are stacked alongside its result.
    operands = (m * n_per + n_per * bk) * itemsize if whole_a else 0
    return (n_per * k * itemsize  # w resident
            + a_rows * n_per * itemsize  # a (double buffer, or all of it)
            + num_chips * m_per * k * itemsize  # run slots
            + acc  # accumulator
            + 2 * m_per * stage_w * itemsize  # staging tiles
            + dot_rows * bk * 4  # dot value
            + operands + (m * bk * 4 if whole_a else 0))  # prod scratch


def _scratch_bytes(scratch_shapes):
    """Bytes of the VMEM scratches a config declares — Mosaic must fit these,
    so the limit is never allowed below it."""
    return sum(
        math.prod(s.shape) * jnp.dtype(s.dtype).itemsize
        for s in scratch_shapes
        if getattr(s, "memory_space", None) == pltpu.MemorySpace.VMEM)


def get_vmem_estimate_bytes_hbm(m_per, n_per, k, bk, itemsize):
    return (n_per * k * itemsize  # w resident
            + 2 * m_per * n_per * itemsize  # a double buffer
            + m_per * k * 4  # fp32 acc
            + 2 * m_per * bk * itemsize  # recv tiles
            + 2 * m_per * bk * itemsize  # run RMW load tiles
            + 2 * m_per * bk * itemsize  # store staging tiles
            + m_per * bk * 4)  # dot value


def allport_matmul_reduce_scatter(
    a,
    w,
    mesh,
    axis_name,
    collective_id: int = _COLLECTIVE_ID,
):
    """reduce_scatter(a @ w, axis=0), all-port pipelined hypercube kernel.

    Residency is picked statically per shape: VMEM run slots below the
    scoped-VMEM budget, HBM-streamed above.

    Args:
      a: [M, n_per] sharded P(None, axis_name).
      w: [n_per, K] sharded P(axis_name, None).
      mesh: 1-D mesh from topology.make_collective_mesh().
      axis_name: the mesh axis to reduce over.
      collective_id: barrier semaphore id; must differ from any other
        collective kernel live in the same program.

    Returns:
      [M // tp, K] sharded P(axis_name, None).
    """
    if len(mesh.shape) != 1:
        # Peers are addressed by LOGICAL (flat) device id while the ids come
        # from lax.axis_index; the two coincide only on a 1-D mesh.
        raise ValueError(f"mesh must be 1-D, got axes {tuple(mesh.shape)}")
    tp_size = mesh.shape[axis_name]
    _validate_inputs(a, w, tp_size)
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips))
    m = a.shape[0]
    k = w.shape[1]
    m_per = m // tp_size
    n_per_est = a.shape[1] // tp_size
    band_bounds = _band_bounds(k, num_dims)
    band_w = max(w_ for _, w_ in band_bounds)
    sched, prog = topology.allport_program(num_dims)
    itemsize = a.dtype.itemsize
    vmem_cap = _vmem_cap_bytes()
    bk_v = next(c for c in _TILE_CANDIDATES if k % c == 0)
    use_hbm = (get_vmem_estimate_bytes(m_per, n_per_est, k, bk_v, num_chips,
                                       max(bk_v, band_w), itemsize) +
               8 * 1024 * 1024 > vmem_cap)
    # HBM mode: the staging/RMW buffers scale with bk and the wire messages
    # don't — pick the largest bk whose whole footprint still fits.
    if use_hbm:
        bk = next(
            (c for c in _TILE_CANDIDATES if k % c == 0 and
             get_vmem_estimate_bytes_hbm(m_per, n_per_est, k, c, itemsize) +
             8 * 1024 * 1024 <= vmem_cap), None)
        if bk is None:
            raise ValueError(
                f"no column tile fits the {vmem_cap} B scoped-VMEM budget at "
                f"m_per={m_per}, n_per={n_per_est}, k={k}")
    else:
        bk = bk_v
    # Small M (VMEM path): one weight sweep + whole-a residency + full-width
    # merges — at these sizes the floor is MXU weight re-sweeps and DMA-op
    # count, not bytes on the wire. Its residents are strictly larger than the
    # chunked ones use_hbm was decided on, so the shape has to be re-priced:
    # a single [m, bk] dot per w tile needs the prod scratch and its value on
    # the stack, so shrink the tile until that fits; if nothing does, give up
    # on whole_a for the already-priced chunked path.
    whole_a = (not use_hbm) and m_per <= 64
    bk_merge = k if whole_a else bk
    if whole_a:

        def whole_a_fits(tile):
            return get_vmem_estimate_bytes(
                m_per,
                n_per_est,
                k,
                tile,
                num_chips,
                max(bk_merge, band_w),
                itemsize,
                m=m,
                whole_a=True) + 4 * 1024 * 1024 <= vmem_cap

        pick = next(
            (c for c in _TILE_CANDIDATES if k % c == 0 and whole_a_fits(c)),
            None)
        if pick is None:
            whole_a, bk_merge = False, bk
        else:
            bk = pick

    cfg = _Config(sched=sched,
                  prog=prog,
                  num_dims=num_dims,
                  m_per=m_per,
                  bk=bk,
                  bk_merge=bk_merge,
                  whole_a=whole_a,
                  band_bounds=band_bounds,
                  axis_name=axis_name)

    def per_device(a_local, w_local):
        n_per = a_local.shape[1]
        hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
        if use_hbm:
            scratch_shapes = (
                pltpu.VMEM(w_local.shape, w_local.dtype),  # w_vmem
                pltpu.VMEM((2, m_per, n_per), a_local.dtype),  # a_vmem
                pltpu.VMEM((m_per, k), jnp.float32),  # acc_vmem
                pltpu.VMEM((2, m_per, bk), a_local.dtype),  # recv_vmem
                pltpu.VMEM((2, m_per, bk), a_local.dtype),  # runld_vmem
                pltpu.VMEM((2, m_per, bk), a_local.dtype),  # store_vmem
                pltpu.SemaphoreType.DMA,  # sem_a
                pltpu.SemaphoreType.DMA((2, )),  # sem_recv
                pltpu.SemaphoreType.DMA((2, )),  # sem_runld
                pltpu.SemaphoreType.DMA((2, )),  # sem_store
                pltpu.SemaphoreType.DMA((k // bk, )),  # sems_w
                pltpu.SemaphoreType.DMA((num_chips, )),  # sems1
                pltpu.SemaphoreType.DMA((sched["n_slots"], )),  # sems2
            )
            out_shape = (
                jax.ShapeDtypeStruct((m_per, k), a_local.dtype),  # out_hbm
                jax.ShapeDtypeStruct((num_chips, m_per, k),
                                     a_local.dtype),  # recv1_hbm
                jax.ShapeDtypeStruct((sched["n_slots"], m_per, band_w),
                                     a_local.dtype),  # recv2_hbm
                jax.ShapeDtypeStruct((num_chips, m_per, k),
                                     a_local.dtype),  # run_hbm
            )
            out_specs = (hbm, hbm, hbm, hbm)
            kernel_fn = _allport_kernel_hbm
            vmem_limit = min(
                max(get_vmem_estimate_bytes_hbm(m_per, n_per, k, bk, itemsize),
                    _scratch_bytes(scratch_shapes)) + 8 * 1024 * 1024,
                vmem_cap)
        else:
            a_shape = ((m, n_per) if whole_a else (2, m_per, n_per))
            # whole_a: per-label own slots (wire dtype, one extra rounding);
            # chunked: one fp32 accumulator reused per pair.
            acc_spec = (pltpu.VMEM(
                (num_chips, m_per,
                 k), a_local.dtype) if whole_a else pltpu.VMEM(
                     (m_per, k), jnp.float32))
            recv_w = max(bk_merge, band_w)
            prod_shape = (m, bk) if whole_a else (8, 128)  # unused if not
            scratch_shapes = (
                pltpu.VMEM(w_local.shape, w_local.dtype),  # w_vmem
                pltpu.VMEM(a_shape, a_local.dtype),  # a_vmem
                pltpu.VMEM((num_chips, m_per, k), a_local.dtype),  # run_vmem
                acc_spec,  # acc_vmem
                pltpu.VMEM((2, m_per, recv_w), a_local.dtype),  # recv_vmem
                pltpu.VMEM(prod_shape, jnp.float32),  # prod_vmem
                pltpu.SemaphoreType.DMA,  # sem_a
                pltpu.SemaphoreType.DMA((2, )),  # sem_recv
                pltpu.SemaphoreType.DMA,  # sem_out
                pltpu.SemaphoreType.DMA((k // bk, )),  # sems_w
                pltpu.SemaphoreType.DMA((num_chips, )),  # sems1
                pltpu.SemaphoreType.DMA((sched["n_slots"], )),  # sems2
            )
            out_shape = (
                jax.ShapeDtypeStruct((m_per, k), a_local.dtype),  # out_hbm
                jax.ShapeDtypeStruct((num_chips, m_per, k),
                                     a_local.dtype),  # recv1_hbm
                jax.ShapeDtypeStruct((sched["n_slots"], m_per, band_w),
                                     a_local.dtype),  # recv2_hbm
            )
            out_specs = (hbm, hbm, hbm)
            kernel_fn = _allport_kernel
            # Mosaic reuses buffers, so charging every static term at once
            # overstates the live peak and degrades codegen; but the limit
            # must still cover the scratches actually declared, or the
            # compile fails outright. Take the larger of the two.
            a_bytes = (m if whole_a else 2 * m_per) * n_per
            vmem_limit = min(
                max(
                    get_vmem_estimate_bytes(m_per, n_per, k, bk, num_chips,
                                            recv_w, itemsize) +
                    (a_bytes - 2 * m_per * n_per) * itemsize,
                    _scratch_bytes(scratch_shapes)) + 8 * 1024 * 1024,
                vmem_cap)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm, hbm],
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=(1, ),
        )
        outs = pl.pallas_call(
            functools.partial(kernel_fn, cfg=cfg),
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
        in_specs=(P(None, axis_name), P(axis_name, None)),
        out_specs=P(axis_name, None),
        check_vma=False,
    )(a, w)

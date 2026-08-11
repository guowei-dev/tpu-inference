# SPDX-License-Identifier: Apache-2.0
"""Physical-topology derivation for the collective kernels.

The fused collective kernels (all_gather_matmul, matmul_reduce_scatter) run on
a 1-D mesh and inherit their physical embedding from the mesh's device order:
XLA and the kernels' ring both walk that order, so a 2-hop edge in it taxes
every ring step (measured 1.9x on the pure ring at 8 devices). The
hierarchical kernels additionally assume logical id == chip * cores_per_chip
+ core and single-hop XOR chip partners. This module derives those properties
from the live device coords and always ASSERTS them — on a topology a
construction does not fit, it raises instead of quietly handing back a slow
or incorrect embedding.

Constructions cover the sub-cube slices whose axes have extent <= 2 (2x2x1,
2x2x2, ...): there every snake ring closes single-hop and every hypercube
XOR partner is one coordinate step (mesh adjacency == torus adjacency; OCS
wraparound only exists on a full 4x4x4 cube).
"""
import dataclasses
import math

import jax
import numpy as np
from jax.sharding import Mesh

AXIS = "x"


@dataclasses.dataclass(frozen=True)
class Topology:
    """Chip grid seen by the live devices, normalized to zero-based coords."""
    num_devices: int
    num_chips: int
    cores_per_chip: int
    axis_extents: tuple  # per coord axis, after normalization


def _chips(devices):
    """{normalized chip coord: [device indices, natural order]}."""
    by_chip = {}
    for i, d in enumerate(devices):
        by_chip.setdefault(tuple(d.coords), []).append(i)
    lo = tuple(min(c[a] for c in by_chip) for a in range(len(next(iter(by_chip)))))
    return {tuple(c - o for c, o in zip(coord, lo)): idxs
            for coord, idxs in by_chip.items()}


def derive_topology(devices):
    by_chip = _chips(devices)
    per_chip = {len(v) for v in by_chip.values()}
    if len(per_chip) != 1:
        raise ValueError(f"non-uniform cores per chip: {per_chip}")
    naxes = len(next(iter(by_chip)))
    extents = tuple(
        len({c[a] for c in by_chip}) for a in range(naxes))
    return Topology(num_devices=len(devices), num_chips=len(by_chip),
                    cores_per_chip=per_chip.pop(), axis_extents=extents)


# ---------------------------------------------------------------------------
# Ring embedding (the 1-D ring kernels + XLA's collective-permute ring)


def chip_cycle(devices):
    """Consecutive-device chip coords, deduplicated, in device order."""
    cycle = []
    for d in devices:
        c = tuple(d.coords)
        if not cycle or cycle[-1] != c:
            cycle.append(c)
    return cycle


def assert_single_hop_ring(devices):
    """Every consecutive chip pair (incl. the wraparound) must be 1 hop.

    A single chip is a degenerate ring (no inter-chip edge to check)."""
    cycle = chip_cycle(devices)
    if sorted(cycle) != sorted(set(cycle)):
        raise ValueError(f"device order revisits a chip: {cycle}")
    if len(cycle) == 1:
        return
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        hops = sum(abs(x - y) for x, y in zip(a, b))
        if hops != 1:
            raise ValueError(
                f"chip ring edge {a}->{b} is {hops} hops (want 1) — this "
                f"topology does not admit the snake construction; order the "
                f"devices explicitly")


def _snake(axis_values):
    """Boustrophedon coord order: each higher axis reverses the sub-order it
    wraps, so consecutive coords differ by one step on exactly one axis."""
    if not axis_values:
        return [()]
    head, sub = axis_values[0], _snake(axis_values[1:])
    out = []
    for i, v in enumerate(head):
        seq = sub if i % 2 == 0 else list(reversed(sub))
        out.extend((v,) + s for s in seq)
    return out


def ring_device_order(devices):
    """Indices into `devices` forming a single-hop Hamiltonian chip cycle.

    Chips are visited on a snake over their coordinate grid (the closing edge
    is a single hop when the outermost axis has extent 1 or 2, which covers
    the 2x2x1 and 2x2x2 slices); a chip's own cores stay adjacent and in their
    natural order. Raises if the result is not a single-hop ring."""
    by_chip = _chips(devices)
    axes = [sorted({c[a] for c in by_chip})
            for a in range(len(next(iter(by_chip))))]
    order = []
    for coord in _snake(axes):
        if coord not in by_chip:
            raise ValueError(f"chip coords are not a full grid (missing "
                             f"{coord}); order the devices explicitly")
        order.extend(by_chip[coord])
    assert_single_hop_ring([devices[i] for i in order])
    return order


# ---------------------------------------------------------------------------
# Hierarchical embedding (twin-pair + hypercube kernels)


def chip_labels(devices):
    """{normalized chip coord: hypercube label}, bit b = the b-th axis of
    extent 2 (axis order preserved). Requires every axis extent <= 2 and a
    full grid, so label ^ (1 << dim) is always one coordinate step."""
    by_chip = _chips(devices)
    naxes = len(next(iter(by_chip)))
    axis_vals = [sorted({c[a] for c in by_chip}) for a in range(naxes)]
    for a, vals in enumerate(axis_vals):
        if vals not in ([0], [0, 1]):
            raise ValueError(
                f"axis {a} values {vals} not in ([0], [0,1]) — hypercube "
                f"labeling covers extent<=2 slices only")
    bit_axes = [a for a, vals in enumerate(axis_vals) if vals == [0, 1]]
    if len(by_chip) != 1 << len(bit_axes):
        raise ValueError(f"{len(by_chip)} chips do not fill the "
                         f"{1 << len(bit_axes)}-vertex grid")
    return {coord: sum(coord[a] << b for b, a in enumerate(bit_axes))
            for coord in by_chip}


def assert_xor_partners_adjacent(devices):
    """Every label ^ (1 << dim) partner must be exactly 1 coordinate step."""
    labels = chip_labels(devices)
    coord_of = {lb: c for c, lb in labels.items()}
    ndims = max(labels.values()).bit_length() if len(labels) > 1 else 0
    for lb, c in coord_of.items():
        for dim in range(ndims):
            p = coord_of[lb ^ (1 << dim)]
            hops = sum(abs(x - y) for x, y in zip(c, p))
            if hops != 1:
                raise ValueError(
                    f"XOR partners {c}<->{p} (dim {dim}) are {hops} hops")


def hier_device_order(devices):
    """Indices into `devices` such that logical id == chip_label *
    cores_per_chip + core — the layout the hierarchical kernels assume
    (twin = id ^ 1 shares the chip; chip_id = id // cores)."""
    by_chip = _chips(devices)
    labels = chip_labels(devices)
    assert_xor_partners_adjacent(devices)
    order = []
    for coord in sorted(by_chip, key=lambda c: labels[c]):
        order.extend(by_chip[coord])
    return order


# ---------------------------------------------------------------------------


def allport_schedule(num_dims):
    """Relative-label ladder for the all-port (concurrent-dims) RS schedule.

    Everything is expressed in RELATIVE chip labels l = q XOR my_chip, which
    makes the whole dependency ladder device-invariant: every send/wait sits
    at the same static program point on every device (only traced values —
    absolute labels, partner ids, row offsets — differ). Band b trades dim
    (b + s) % num_dims at round s; op bits enumerate the dims not yet traded
    in that band's rotation.

    Returns a dict with (all in relative-label space):
      l_twin:      twin emission order, popcount-descending — the first twin
                   merge unlocks one round-0 send on EVERY dim.
      sends:       {(s, b, op): send_label}   (label's dim-s bit is 1)
      merges:      {(s, b, op): target_label} (the kept label the arrival
                   adds into)
      r0_unlock:   {twin label l: [(b, op), ...]} round-0 sends unlocked by
                   that twin merge
      send_prereqs:{(s, b, op): [('twin', l), ('merge', (r, b, op_r)), ...]}
                   every merge into the send label at rounds < s in band b,
                   plus its twin merge
      slot:        {(s, b, op): static landing-slot / sem-cell index}
      n_slots:     total phase-2 slots
    """
    d = num_dims
    dims = lambda b: [(b + r) % d for r in range(d)]
    sched = dict(num_dims=d, sends={}, merges={}, r0_unlock={},
                 send_prereqs={}, slot={})
    total = 0
    for s in range(d):
        ops = 2**(d - 1 - s)
        for b in range(d):
            delta = dims(b)
            for op in range(ops):
                label_hi = sum(((op >> i) & 1) << delta[s + 1 + i]
                               for i in range(d - 1 - s))
                sched["sends"][(s, b, op)] = (1 << delta[s]) | label_hi
                sched["merges"][(s, b, op)] = label_hi
                sched["slot"][(s, b, op)] = total
                total += 1
    sched["n_slots"] = max(total, 1)
    sched["l_twin"] = sorted(range(2**d),
                             key=lambda l: (-bin(l).count("1"), l))
    for (s, b, op), lbl in sched["sends"].items():
        prereqs = [("twin", lbl)]
        delta = dims(b)
        for r in range(s):
            # the round-r merge into this label: op_r bit i indexes dim
            # delta[r+1+i]; the label's bit at delta[s] (==1) and its op'
            # bits sit above.
            op_r = sum(((lbl >> delta[r + 1 + i]) & 1) << i
                       for i in range(d - 1 - r))
            prereqs.append(("merge", (r, b, op_r)))
        sched["send_prereqs"][(s, b, op)] = prereqs
        if s == 0:
            sched["r0_unlock"].setdefault(lbl, []).append((b, op))
    return sched


def allport_ag_ladder(num_dims, num_bands=None):
    """Relative-label ladder for the all-port (concurrent-dims) ALL-GATHER.

    Not a reversal of allport_schedule — the reduce-scatter and the all-gather
    are duals in volume but not in structure, and two differences decide the
    kernel:

      * **The twin (D2D) exchange goes LAST.** Each core disseminates only its
        OWN parity plane over ICI (7P per core, not 14P) and forwards every
        arrival to its twin, so a chip imports each of the 14 remote chunks
        exactly once. 14P over the 3 chip ICI ports is 4.67P per port — the
        information-theoretic floor for this topology. A ring cycle touches
        only 2 of the 3 ports and carries 7.5P on each; that 1.6x is the whole
        prize, and it is a property of the port count, not of the schedule's
        cleverness.
      * **Bands split the chunk's ROWS**, not the contraction, so every
        arriving [rows, k] slice is a complete sub-chunk that dots the moment
        it lands. Band b trades dim (b + s) % num_dims at round s, so all
        num_bands ports are busy every round; with fewer bands than dims
        (m_per too small to split), the rotation still covers every dim, the
        per-port load just stops being balanced.

    Everything is in RELATIVE chip labels l = q XOR my_chip, so every device
    emits the same static sequence of starts and waits — the deadlock-freedom
    argument of allport_schedule applies verbatim.

    Returns a dict with:
      order[b][p]:      band b's label at ladder position p (bit i of p is the
                        label's bit at dim (b + i) % num_dims), so the labels
                        held after round s are exactly positions [0, 2^s).
      sends[(s, b, j)]: label sent at round s, band b, message j < 2^s.
      recvs[(s, b, j)]: label that lands from it (== order[b][j + 2^s]).
      dims[(b, s)]:     the dim traded, (b + s) % num_dims.
    """
    d = num_dims
    nb = d if num_bands is None else num_bands
    if not 1 <= nb <= d:
        raise ValueError(f"num_bands {nb} not in [1, {d}]")
    dims = {(b, s): (b + s) % d for b in range(nb) for s in range(d)}
    order = [[
        sum(((p >> i) & 1) << dims[(b, i)] for i in range(d))
        for p in range(1 << d)
    ] for b in range(nb)]
    sends, recvs = {}, {}
    for s in range(d):
        for b in range(nb):
            for j in range(1 << s):
                sends[(s, b, j)] = order[b][j]
                recvs[(s, b, j)] = order[b][j] | (1 << dims[(b, s)])
                assert recvs[(s, b, j)] == order[b][j + (1 << s)]
    return dict(num_dims=d, num_bands=nb, order=order, sends=sends,
                recvs=recvs, dims=dims)


def allport_program(num_dims):
    """The all-port kernel's static emission order (one program, all devices).

    Abstract ops over the allport_schedule ladder, all in relative-label
    space: ('compute_tw', l) ('twin_start', l) ('compute_own', l)
    ('twin_wait', l) ('twin_merge', l) ('p2_start', key) ('p2_wait', key)
    ('p2_merge', key) ('out',) with key = (round, band, op).

    Construction: twin pairs in l_twin order; each twin merge immediately
    starts the round-0 sends it unlocks; at the end of each pair, HOIST —
    greedily wait+merge any started phase-2 op whose merge target's twin
    merge is done, and start every send whose prereqs that completes (the
    phase-barrier breaker); leftovers drain after the last pair in start
    order. Deadlock-freedom rests on this being ONE static sequence for
    every device — see allport_schedule and the kernel docstring.
    """
    sched = allport_schedule(num_dims)
    prog, started, waited, done = [], [], set(), set()

    def merges_done(key):
        return all(p in done for p in sched["send_prereqs"][key])

    def start(key):
        prog.append(("p2_start", key))
        started.append(key)

    def hoist():
        progressed = True
        while progressed:
            progressed = False
            for key in list(started):
                if key in waited:
                    continue
                target = sched["merges"][key]
                if ("twin", target) not in done:
                    continue
                prog.extend([("p2_wait", key), ("p2_merge", key)])
                waited.add(key)
                done.add(("merge", key))
                progressed = True
                for nxt, pre in sched["send_prereqs"].items():
                    if nxt not in started and all(p in done for p in pre):
                        start(nxt)

    for l in sched["l_twin"]:
        prog.extend([("compute_tw", l), ("twin_start", l),
                     ("compute_own", l), ("twin_wait", l),
                     ("twin_merge", l)])
        done.add(("twin", l))
        for key in sched["r0_unlock"].get(l, []):
            start((0, ) + key)
        hoist()
    assert all(k in waited for k in started) and len(started) == len(
        sched["sends"]), "allport_program: ladder did not drain"
    prog.append(("out", ))
    return sched, prog


def select_path(pattern, m, tp_size):
    """Measured per-M dispatch for the collective-matmul patterns.

    Thresholds from the v7x-8 (2x2x1, 8-device) same-process lat-basis study
    (ABBA-certified, spread 0.1-0.3%): XLA's serve-flag serial lowering wins
    at M <= 512 for both patterns; for MM-RS the all-port pipelined kernel
    (allport_matmul_reduce_scatter) wins at M = 1024 (115.5 vs ring 119.8 us,
    and 0.71-0.78x vs ring at 256-512 where XLA still leads overall); the
    single-hop-ring kernels win at larger M — at M >= 2048 both MM-RS
    kernels sit on their shared-chip-ICI-port wire walls (allport 3P/port
    but round-0 sends unlock only after their twin merges; ring 3.5P/port
    pipelined from t=0) and the ring's head start wins. AG-MM: ring from
    M >= 1024 (1.25x vs best XLA at 8192). The bulk-synchronous hier
    kernels never win here.

    On the >= 3-dim (2x2x2, 16-device) slice the measured verdict flips:
    with the one-weight-sweep small-M compute and the HBM-residency large-M
    form, all-port MM-RS leads EVERY measured M (tpu7x-16 lat basis, ABBA/
    matrix 2026-07-29: 61.6/78.1/129.9/173.3/320.9/601.8 us @M=256..8192 vs
    best-XLA 71.8/88.1/140.6/320.4/491.2/1024.9 and ring 166.6/208.5/220.0/
    366.9/716.8) — dispatch is all-port across the measured band, ring only
    beyond it. Below M = 16*tp the kernel is unmeasured; XLA keeps the tail.

    pattern: 'ag_mm' | 'mm_rs'; m = GLOBAL row count.
    Returns 'xla' | 'allport' | 'ring'.
    """
    if pattern not in ("ag_mm", "mm_rs"):
        raise ValueError(f"unknown pattern {pattern!r} (ag_mm|mm_rs)")
    num_chips = tp_size // 2
    num_dims = int(math.log2(num_chips)) if num_chips > 1 else 0
    if pattern == "mm_rs" and num_dims >= 3:
        if m < 16 * tp_size:
            return "xla"
        return "allport" if m <= 512 * tp_size else "ring"
    if m <= 64 * tp_size:
        return "xla"
    if pattern == "mm_rs" and m <= 128 * tp_size:
        return "allport"
    return "ring"


def make_collective_mesh(kind="ring", *, devices=None, axis_name=AXIS):
    """1-D Auto-axis mesh over the derived device order.

    kind: 'natural' = live order; 'ring' = single-hop Hamiltonian chip cycle
    (the 1-D ring kernels + order-sensitive XLA lowerings); 'hier' = hypercube
    label-major order (the hierarchical kernels)."""
    devs = list(devices if devices is not None else jax.devices())
    if kind == "natural":
        idx = list(range(len(devs)))
    elif kind == "ring":
        idx = ring_device_order(devs)
    elif kind == "hier":
        idx = hier_device_order(devs)
    else:
        raise ValueError(f"unknown mesh kind {kind!r} (natural|ring|hier)")
    return Mesh(np.asarray([devs[i] for i in idx]), (axis_name,))

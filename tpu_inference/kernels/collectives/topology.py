# SPDX-License-Identifier: Apache-2.0
"""Host-side hypercube embedding for the all-port collective kernels.

The all-port kernel assumes logical id == chip * cores_per_chip + core and
single-hop XOR chip partners. This module derives both from the live device
coords and ASSERTS them: on a topology that does not admit the construction it
raises, rather than quietly handing back an incorrect embedding. It also builds
the relative-label send/merge ladder the kernel walks.

(Distinct from hierrs_sc.topology, which resolves neighbours from inside a
kernel at trace time; this one runs on the host and produces the Mesh.)

The labeling covers the sub-cube slices whose axes have extent <= 2 (2x2x1,
2x2x2, ...): there every hypercube XOR partner is one coordinate step (mesh
adjacency == torus adjacency; OCS wraparound only exists on a full 4x4x4 cube).
"""
import jax
import numpy as np
from jax.sharding import Mesh

AXIS = "x"


def _chips(devices):
    """{normalized chip coord: [device indices, natural order]}."""
    by_chip = {}
    for i, d in enumerate(devices):
        by_chip.setdefault(tuple(d.coords), []).append(i)
    lo = tuple(
        min(c[a] for c in by_chip) for a in range(len(next(iter(by_chip)))))
    return {
        tuple(c - o for c, o in zip(coord, lo)): idxs
        for coord, idxs in by_chip.items()
    }


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
    return {
        coord: sum(coord[a] << b for b, a in enumerate(bit_axes))
        for coord in by_chip
    }


def assert_xor_partners_adjacent(devices):
    """Every label ^ (1 << dim) partner must be exactly 1 coordinate step."""
    labels = chip_labels(devices)
    coord_of = {lb: c for c, lb in labels.items()}
    ndims = max(labels.values()).bit_length()
    for lb, c in coord_of.items():
        for dim in range(ndims):
            p = coord_of[lb ^ (1 << dim)]
            hops = sum(abs(x - y) for x, y in zip(c, p))
            if hops != 1:
                raise ValueError(
                    f"XOR partners {c}<->{p} (dim {dim}) are {hops} hops")


def hier_device_order(devices):
    """Indices into `devices` such that logical id == chip_label *
    cores_per_chip + core — the layout the all-port kernel assumes
    (twin = id ^ 1 shares the chip; chip_id = id // cores)."""
    by_chip = _chips(devices)
    labels = chip_labels(devices)
    assert_xor_partners_adjacent(devices)
    order = []
    for coord in sorted(by_chip, key=lambda c: labels[c]):
        order.extend(by_chip[coord])
    return order


def make_collective_mesh(*, devices=None, axis_name=AXIS):
    """1-D Auto-axis mesh in hypercube label-major order."""
    devs = list(devices if devices is not None else jax.devices())
    idx = hier_device_order(devs)
    return Mesh(np.asarray([devs[i] for i in idx]), (axis_name, ))


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

    def dims(b):
        return [(b + r) % d for r in range(d)]

    sched = dict(sends={}, merges={}, r0_unlock={}, send_prereqs={}, slot={})
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
                             key=lambda lbl: (-bin(lbl).count("1"), lbl))
    for (s, b, op), lbl in sched["sends"].items():
        prereqs = [("twin", lbl)]
        delta = dims(b)
        for r in range(s):
            # the round-r merge into this label: op_r bit i indexes dim
            # delta[r+1+i]; the label's bit at delta[s] (==1) and its op'
            # bits sit above.
            op_r = sum(
                ((lbl >> delta[r + 1 + i]) & 1) << i for i in range(d - 1 - r))
            prereqs.append(("merge", (r, b, op_r)))
        sched["send_prereqs"][(s, b, op)] = prereqs
        if s == 0:
            sched["r0_unlock"].setdefault(lbl, []).append((b, op))
    return sched


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

    for lbl in sched["l_twin"]:
        prog.extend([("compute_tw", lbl), ("twin_start", lbl),
                     ("compute_own", lbl), ("twin_wait", lbl),
                     ("twin_merge", lbl)])
        done.add(("twin", lbl))
        for key in sched["r0_unlock"].get(lbl, []):
            start((0, ) + key)
        hoist()
    assert all(k in waited for k in started) and len(started) == len(
        sched["sends"]), "allport_program: ladder did not drain"
    prog.append(("out", ))
    return sched, prog

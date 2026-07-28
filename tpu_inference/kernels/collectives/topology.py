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


def select_path(pattern, m, tp_size):
    """Measured per-M dispatch for the collective-matmul patterns.

    Thresholds from the v7x-8 (2x2x1, 8-device) same-process lat-basis study:
    XLA's serve-flag serial lowering wins at M <= 512 for both patterns; the
    single-hop-ring fused kernels win at M >= 1024 (AG 1.25x / RS 1.33x at
    M=8192); the hierarchical kernels never win on the 2-dim slice (their
    phase-boundary exposure outweighs the shorter round count) — they are the
    expected candidates on 3-dim (2x2x2) slices, unmeasured there. The ring
    MM-RS cannot compile M = tp_size * 16 (Mosaic E2003), where XLA wins
    anyway.

    pattern: 'ag_mm' | 'mm_rs'; m = GLOBAL row count. Returns 'xla' | 'ring'.
    """
    if pattern not in ("ag_mm", "mm_rs"):
        raise ValueError(f"unknown pattern {pattern!r} (ag_mm|mm_rs)")
    return "xla" if m <= 64 * tp_size else "ring"


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

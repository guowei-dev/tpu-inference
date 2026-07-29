# SPDX-License-Identifier: Apache-2.0
"""CPU simulator for the all-port AG schedule (topology.allport_ag_program).

Executes the kernel's exact static op sequence on every simulated device of a
2x1x1 (D=1), 2x2x1 (D=2, 8 devices) and 2x2x2 (D=3, 16 devices) slice, with
shared-cell semaphore semantics (a wait needs the local start AND the
partner's symmetric start), and asserts:
  - no deadlock (a full scheduler pass must always advance someone),
  - every cell started exactly once and waited exactly once, wait after the
    local start,
  - every gather slot written exactly once and every send/dot sourced from a
    written slot,
  - exact provenance: every device ends holding ALL tp chunks with each
    slot's payload equal to its own global chunk id, and dots every
    (chunk, band) exactly once.

On an 8-device host the D=3 case is the 2x2x2 structural verification for
the all-port AG kernel, and the pre-device gate for any choreography edit:
the program the kernel emits is the program simulated here.
"""
import jax
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()


class Device:

    def __init__(self, dev, num_dims):
        self.dev = dev
        self.chip = dev // 2
        self.bit = dev % 2
        self.twin = dev ^ 1
        self.num_dims = num_dims
        self.pc = 0
        self.gather = {}       # global chunk id -> {band: payload chunk id}
        self.started = set()   # cells ('twin', l) / ('ici', key)
        self.waited = set()
        self.dotted = set()    # (global chunk id, band)
        self.snapshots = {}    # cell -> payload at start time

    def chunk_of(self, l, parity):
        """Global chunk id of relative label l, in my (0) / the twin's (1)
        parity plane."""
        bit = self.bit if parity == 0 else 1 - self.bit
        return (self.chip ^ l) * 2 + bit

    def partner(self, cell):
        kind, key = cell
        if kind == "twin":
            return self.twin
        s, b, _ = key
        dim = (b + s) % self.num_dims
        return (self.chip ^ (1 << dim)) * 2 + self.bit


def simulate(num_dims):
    sched, prog = topology.allport_ag_program(num_dims)
    tp = 2**(num_dims + 1)
    nbands = max(num_dims, 1)
    devs = [Device(d, num_dims) for d in range(tp)]
    violations = []

    def write(dv, chunk, band, payload):
        if band in dv.gather.get(chunk, {}):
            violations.append(
                f"dev {dv.dev}: slot ({chunk},{band}) double-write")
        dv.gather.setdefault(chunk, {})[band] = payload

    def ready(dv, op):
        kind = op[0]
        if kind in ("wait", "twin_wait"):
            cell = ("ici", op[1]) if kind == "wait" else ("twin", op[1])
            if cell not in dv.started:
                violations.append(f"dev {dv.dev}: wait before start {cell}")
                return True
            return cell in devs[dv.partner(cell)].started
        return True

    def execute(dv, op):
        kind = op[0]
        if kind == "seed":
            for b in range(nbands):
                write(dv, dv.chunk_of(0, 0), b, dv.chunk_of(0, 0))
        elif kind == "send":
            key = op[1]
            _, b, l = key
            cell = ("ici", key)
            src = dv.chunk_of(l, 0)
            if b not in dv.gather.get(src, {}):
                violations.append(
                    f"dev {dv.dev}: send {key} before slot ready")
            dv.started.add(cell)
            dv.snapshots[cell] = dv.gather.get(src, {}).get(b)
        elif kind == "wait":
            key = op[1]
            cell = ("ici", key)
            dv.waited.add(cell)
            _, b, _ = key
            arr = sched["arrivals"][key]
            payload = devs[dv.partner(cell)].snapshots.get(cell)
            write(dv, dv.chunk_of(arr, 0), b, payload)
        elif kind == "twin_start":
            l = op[1]
            cell = ("twin", l)
            have = dv.gather.get(dv.chunk_of(l, 0), {})
            if set(have) != set(range(nbands)):
                violations.append(
                    f"dev {dv.dev}: twin_start {l} on incomplete chunk")
            dv.started.add(cell)
            dv.snapshots[cell] = dict(have)
        elif kind == "twin_wait":
            l = op[1]
            cell = ("twin", l)
            dv.waited.add(cell)
            arrival = devs[dv.twin].snapshots.get(cell, {})
            for b, payload in arrival.items():
                write(dv, dv.chunk_of(l, 1), b, payload)
        elif kind == "dot":
            parity, b, l = op[1]
            chunk = dv.chunk_of(l, parity)
            if b not in dv.gather.get(chunk, {}):
                violations.append(
                    f"dev {dv.dev}: dot ({parity},{b},{l}) before data")
            if (chunk, b) in dv.dotted:
                violations.append(f"dev {dv.dev}: dot ({chunk},{b}) twice")
            dv.dotted.add((chunk, b))

    # Round-robin scheduler; a full pass with no progress = deadlock.
    while any(dv.pc < len(prog) for dv in devs):
        progressed = False
        for dv in devs:
            while dv.pc < len(prog) and ready(dv, prog[dv.pc]):
                execute(dv, prog[dv.pc])
                dv.pc += 1
                progressed = True
        if not progressed:
            stuck = {dv.dev: prog[dv.pc] for dv in devs if dv.pc < len(prog)}
            raise AssertionError(f"DEADLOCK at {stuck}")
    return sched, prog, devs, violations


class AllportAgScheduleTest(jtu.JaxTestCase):

    @parameterized.parameters(1, 2, 3)
    def test_simulated_execution(self, num_dims):
        sched, prog, devs, violations = simulate(num_dims)
        tp = 2**(num_dims + 1)
        nbands = max(num_dims, 1)
        self.assertEmpty(violations)
        for dv in devs:
            # every cell started exactly once and waited exactly once
            self.assertEqual(len(dv.started),
                             len(sched["sends"]) + 2**num_dims)
            self.assertEqual(dv.started, dv.waited)
            # full gather with exact provenance: slot payload == chunk id
            self.assertEqual(set(dv.gather), set(range(tp)),
                             f"dev {dv.dev} missing chunks")
            for chunk, bands in dv.gather.items():
                self.assertEqual(set(bands), set(range(nbands)))
                for b, payload in bands.items():
                    self.assertEqual(payload, chunk,
                                     f"dev {dv.dev} slot ({chunk},{b}) holds "
                                     f"{payload}")
            # every (chunk, band) dotted exactly once
            self.assertEqual(dv.dotted,
                             {(c, b) for c in range(tp)
                              for b in range(nbands)})

    @parameterized.parameters(1, 2, 3)
    def test_ladder_static_invariants(self, num_dims):
        sched, prog = topology.allport_ag_program(num_dims)
        # send labels have the traded dim's bit clear; arrivals have it set
        for (s, b, l), dim in sched["sends"].items():
            self.assertEqual(dim, (b + s) % num_dims)
            self.assertEqual((l >> dim) & 1, 0)
            self.assertEqual((sched["arrivals"][(s, b, l)] >> dim) & 1, 1)
        # every ICI cell appears exactly once as send and once as wait
        for kind in ("send", "wait"):
            keys = [op[1] for op in prog if op[0] == kind]
            self.assertCountEqual(keys, list(sched["sends"]))
        pos = {op: i for i, op in enumerate(prog)}
        for key in sched["sends"]:
            self.assertLess(pos[("send", key)], pos[("wait", key)])
            # the single prerequisite wait precedes the send
            pre = sched["send_prereqs"][key]
            if pre is not None:
                self.assertLess(pos[("wait", pre)], pos[("send", key)])
        # twin cells: start precedes wait, one each per label
        for l in range(2**num_dims):
            self.assertLess(pos[("twin_start", l)], pos[("twin_wait", l)])
        # dots: each (parity, band, label) exactly once, after its data
        dots = [op[1] for op in prog if op[0] == "dot"]
        nbands = max(num_dims, 1)
        self.assertCountEqual(dots, [(p, b, l) for p in (0, 1)
                                     for b in range(nbands)
                                     for l in range(2**num_dims)])
        for p, b, l in dots:
            if p == 1:
                self.assertLess(pos[("twin_wait", l)], pos[("dot", (p, b, l))])
            elif l != 0:
                self.assertLess(pos[("wait", sched["arrive_key"][(b, l)])],
                                pos[("dot", (p, b, l))])

    @parameterized.parameters(1, 2, 3)
    def test_round0_sends_precede_all_waits(self, num_dims):
        # The structural edge over the RS ladder: round-0 sends have no
        # unlock prerequisite and the program emits every one of them before
        # its first wait — the wire pipelines from t=0.
        _, prog = topology.allport_ag_program(num_dims)
        first_wait = min(i for i, op in enumerate(prog)
                         if op[0] in ("wait", "twin_wait"))
        r0_sends = [i for i, op in enumerate(prog)
                    if op[0] == "send" and op[1][0] == 0]
        self.assertNotEmpty(r0_sends)
        self.assertLess(max(r0_sends), first_wait)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

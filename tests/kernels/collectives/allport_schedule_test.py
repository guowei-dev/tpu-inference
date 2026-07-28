# SPDX-License-Identifier: Apache-2.0
"""CPU simulator for the all-port MM-RS schedule (topology.allport_program).

Executes the kernel's exact static op sequence on every simulated device of a
2x2x1 (D=2, 8 devices) and 2x2x2 (D=3, 16 devices) slice, with shared-cell
semaphore semantics (a wait needs the local start AND the partner's symmetric
start), and asserts:
  - no deadlock (a full scheduler pass must always advance someone),
  - every cell started exactly once and waited exactly once, wait after the
    local start,
  - no region is written while a send from it is un-waited,
  - the final run[0] holds every device's contribution exactly once, per band.

On an 8-device host this is the 2x2x2 (D=3) structural verification for the
all-port kernel, and the pre-device gate for any choreography edit: the
program the kernel emits is the program simulated here.
"""
import collections

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
        self.run = {}          # rel label -> {band: Counter of contributors}
        self.tw_out = {}       # rel label -> payload I send to the twin
        self.own = {}          # rel label -> my own-parity partial
        self.started = set()   # cells I started ('twin', l) / ('p2', key)
        self.waited = set()
        self.open_sends = {}   # cell -> (rel label, bands) region in flight
        self.snapshots = {}    # cell -> payload snapshot at start time

    def partner(self, cell):
        kind, key = cell
        if kind == "twin":
            return self.twin
        s, b, _ = key
        dim = (b + s) % self.num_dims
        return (self.chip ^ (1 << dim)) * 2 + self.bit


def simulate(num_dims):
    sched, prog = topology.allport_program(num_dims)
    tp = 2**(num_dims + 1)
    bands = list(range(max(num_dims, 1)))
    devs = [Device(d, num_dims) for d in range(tp)]
    violations = []

    def check_write(dv, label, wbands):
        for cell, (l_open, obands) in dv.open_sends.items():
            if l_open == label and set(obands) & set(wbands):
                violations.append(
                    f"dev {dv.dev}: write to run[{label}] bands {wbands} "
                    f"while send {cell} in flight")

    def ready(dv, op):
        kind = op[0]
        if kind in ("twin_wait", "p2_wait"):
            cell = ("twin", op[1]) if kind == "twin_wait" else ("p2", op[1])
            if cell not in dv.started:
                violations.append(f"dev {dv.dev}: wait before start {cell}")
                return True
            return cell in devs[dv.partner(cell)].started
        return True

    def execute(dv, op):
        kind = op[0]
        if kind == "compute_tw":
            dv.tw_out[op[1]] = collections.Counter([dv.dev])
        elif kind == "twin_start":
            cell = ("twin", op[1])
            dv.started.add(cell)
            dv.snapshots[cell] = dv.tw_out[op[1]]
            dv.open_sends[cell] = (op[1], bands)  # reads tw_out, not run —
            del dv.open_sends[cell]               # no run region in flight
        elif kind == "compute_own":
            dv.own[op[1]] = collections.Counter([dv.dev])
        elif kind == "twin_wait":
            dv.waited.add(("twin", op[1]))
        elif kind == "twin_merge":
            l = op[1]
            arrival = devs[dv.twin].snapshots[("twin", l)]
            check_write(dv, l, bands)
            dv.run[l] = {b: dv.own[l] + arrival for b in bands}
        elif kind == "p2_start":
            key = op[1]
            cell = ("p2", key)
            s, b, _ = key
            lbl = sched["sends"][key]
            dv.started.add(cell)
            dv.snapshots[cell] = dv.run[lbl][b]
            dv.open_sends[cell] = (lbl, [b])
        elif kind == "p2_wait":
            cell = ("p2", op[1])
            dv.waited.add(cell)
            dv.open_sends.pop(cell, None)  # own send drained
        elif kind == "p2_merge":
            key = op[1]
            s, b, _ = key
            target = sched["merges"][key]
            arrival = devs[dv.partner(("p2", key))].snapshots[("p2", key)]
            check_write(dv, target, [b])
            dv.run[target][b] = dv.run[target][b] + arrival
        elif kind == "out":
            pass

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


class AllportScheduleTest(jtu.JaxTestCase):

    @parameterized.parameters(2, 3)
    def test_simulated_execution(self, num_dims):
        sched, prog, devs, violations = simulate(num_dims)
        tp = 2**(num_dims + 1)
        self.assertEmpty(violations)
        for dv in devs:
            # every cell started exactly once and waited exactly once
            self.assertEqual(len(dv.started), len(sched["sends"]) + 2**num_dims)
            self.assertEqual(dv.started, dv.waited)
            # final run[0]: every device's contribution exactly once, per band
            for b in range(max(num_dims, 1)):
                counts = dv.run[0][b]
                self.assertEqual(set(counts), set(range(tp)),
                                 f"dev {dv.dev} band {b} missing contributors")
                self.assertTrue(all(c == 1 for c in counts.values()),
                                f"dev {dv.dev} band {b} double-count: {counts}")

    @parameterized.parameters(2, 3)
    def test_ladder_static_invariants(self, num_dims):
        sched, prog = topology.allport_program(num_dims)
        # send labels have the traded dim's bit set; merge targets don't
        for (s, b, op), lbl in sched["sends"].items():
            dim = (b + s) % num_dims
            self.assertEqual((lbl >> dim) & 1, 1)
            self.assertEqual((sched["merges"][(s, b, op)] >> dim) & 1, 0)
        # every phase-2 op appears exactly once as start, wait and merge
        for kind in ("p2_start", "p2_wait", "p2_merge"):
            keys = [op[1] for op in prog if op[0] == kind]
            self.assertCountEqual(keys, list(sched["sends"]))
        # waits follow their starts; merges follow their waits
        pos = {op: i for i, op in enumerate(prog)}
        for key in sched["sends"]:
            self.assertLess(pos[("p2_start", key)], pos[("p2_wait", key)])
            self.assertLess(pos[("p2_wait", key)], pos[("p2_merge", key)])
        # prereq merges precede each send's start
        for key, pres in sched["send_prereqs"].items():
            for kind, ref in pres:
                if kind == "merge":
                    self.assertLess(pos[("p2_merge", ref)],
                                    pos[("p2_start", key)], f"{key} <- {ref}")
                else:
                    self.assertLess(pos[("twin_merge", ref)],
                                    pos[("p2_start", key)])

    def test_first_twin_merge_lights_every_dim(self):
        for d in (2, 3):
            sched = topology.allport_schedule(d)
            first = sched["l_twin"][0]
            self.assertEqual(first, 2**d - 1)
            bands = {b for b, _ in sched["r0_unlock"][first]}
            self.assertEqual(bands, set(range(d)))


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

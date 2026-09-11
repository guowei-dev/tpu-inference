# SPDX-License-Identifier: Apache-2.0
"""CPU simulator for the all-port AG-MM schedule.

Executes the kernel's exact static op sequence on every simulated device of a
2x2x1 (D=2, 8 devices) and 2x2x2 (D=3, 16 devices) slice and asserts:
  - every device emits the SAME static sequence of starts and waits (the
    deadlock-freedom argument in the kernel docstring rests on this),
  - every message is started exactly once, and never before its payload has
    arrived,
  - the gather buffer ends up complete on every device, and the piece plan
    maps it onto `out` covering each output row exactly once with the right
    chunk,
  - the per-chip-port wire account the kernel exists for: 4.67P at
    num_bands == num_dims, against a ring cycle's 7.5P at D = 3.

The program the kernel emits is the program simulated here, so this is the
pre-device gate for any choreography edit.
"""
import collections

import jax
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.collectives import allport_all_gather_matmul as aag
from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()


def _simulate(num_dims, m_per):
    """Run the static program on every device; return the plan it ran."""
    ladder, pieces, loc = aag._plan(num_dims, m_per)
    num_bands = ladder["num_bands"]
    ndev = 2 << num_dims
    # gather[dev][row] = (source device, source row within its chunk)
    gather = [[None] * (m_per * ndev) for _ in range(ndev)]
    for dev in range(ndev):
        for r in range(m_per):
            gather[dev][r] = (dev, r)  # the local chunk is read from `x`
    started = collections.defaultdict(set)

    def send(src, dst, src_off, dst_off, rows, tag):
        started[tag].add(src)
        for i in range(rows):
            assert gather[src][src_off + i] is not None, (tag, src, "unarrived")
            gather[dst][dst_off + i] = gather[src][src_off + i]

    for dev in range(ndev):
        twin = dev ^ 1
        for b in range(num_bands):
            so, sr = loc[(0, 0, b)]
            do, _ = loc[(1, 0, b)]
            send(dev, twin, so, do, sr, ("twin_own", b))
    for s in range(num_dims):
        for dev in range(ndev):
            for b in range(num_bands):
                for j in range(1 << s):
                    dim = ladder["dims"][(b, s)]
                    partner = ((dev // 2) ^ (1 << dim)) * 2 + dev % 2
                    so, sr = loc[(0, ladder["sends"][(s, b, j)], b)]
                    do, _ = loc[(0, ladder["recvs"][(s, b, j)], b)]
                    send(dev, partner, so, do, sr, ("ici", s, b, j))
        for dev in range(ndev):
            for j in range(1 << s):
                for b in range(num_bands):
                    label = ladder["recvs"][(s, b, j)]
                    so, sr = loc[(0, label, b)]
                    do, _ = loc[(1, label, b)]
                    send(dev, dev ^ 1, so, do, sr, ("twin_fwd", s, b, j))
    return ladder, pieces, gather, started, ndev


class AllportAgScheduleTest(parameterized.TestCase):

    @parameterized.parameters((2, 32), (2, 128), (3, 16), (3, 32), (3, 64),
                              (3, 128), (3, 512))
    def test_program_is_device_invariant(self, num_dims, m_per):
        _, _, _, started, ndev = _simulate(num_dims, m_per)
        for tag, devs in started.items():
            self.assertLen(devs, ndev, f"{tag} not emitted by every device")

    @parameterized.parameters((2, 32), (3, 16), (3, 64), (3, 512))
    def test_gather_completes_and_out_is_covered_once(self, num_dims, m_per):
        _, pieces, gather, _, ndev = _simulate(num_dims, m_per)
        for dev in range(ndev):
            self.assertNotIn(None, gather[dev], f"hole in device {dev}")
            covered = {}
            for p in pieces:
                other = ((dev // 2) ^ p["label"]) * 2 + (
                    dev % 2 if p["par"] == 0 else 1 - dev % 2)
                for i in range(p["rows"]):
                    row = other * m_per + p["out_off"] + i
                    self.assertNotIn(row, covered, "output row written twice")
                    covered[row] = True
                    self.assertEqual(gather[dev][p["g_off"] + i],
                                     (other, p["out_off"] + i))
            self.assertLen(covered, m_per * ndev)

    def test_message_started_exactly_once(self):
        ladder = topology.allport_ag_ladder(3)
        started = set()
        for st in range(3):  # the j == 0 message of every round has no
            for b in range(3):  # dependency and is issued up front
                started.add((st, b, 0))
        for s in range(3):  # an arrival unlocks its position in every
            for b in range(3):  # later round
                for j in range(1 << s):
                    for nxt in range(s + 1, 3):
                        key = (nxt, b, j + (1 << s))
                        self.assertIn(key, ladder["sends"])
                        self.assertNotIn(key, started)
                        started.add(key)
        self.assertEqual(started, set(ladder["sends"]))

    def test_wire_account_per_chip_port(self):
        """The reason the kernel exists: a chip imports 14 chunks over 3 ports
        (4.67P) where a ring cycle carries 15 over 2 (7.5P)."""
        for num_bands, want in ((3, 14 / 3), (2, 6.0), (1, 8.0)):
            ladder = topology.allport_ag_ladder(3, num_bands)
            load = [0] * 3
            for (s, b, _) in ladder["sends"]:
                load[ladder["dims"][(b, s)]] += 1
            # units of P / num_bands per core; x2 cores share each chip port
            self.assertAlmostEqual(2 * max(load) / num_bands, want, places=5)

    def test_bands_shrink_with_m_per(self):
        """A bf16 VMEM ref tiles 16 rows, so a chunk under 16 * num_dims rows
        cannot feed every port — and one band is WORSE than the ring."""
        self.assertEqual(aag._plan(3, 512)[0]["num_bands"], 3)
        self.assertEqual(aag._plan(3, 32)[0]["num_bands"], 2)
        self.assertEqual(aag._plan(3, 16)[0]["num_bands"], 1)

    def test_consumption_orders(self):
        """`order` decides when a piece is WAITED, so it decides whether ready
        rows sit unread. A message goes out when ladder position j is in hand,
        so every round's j == 0 message is issued at t = 0 — `issue` puts them
        all at the head, `head` moves only those, `round` moves none."""
        keys = {o: [p["key"] for p in aag._plan(3, 512, o)[1]
                    if p["key"] is not None]
                for o in ("round", "issue", "head")}
        rounds = {o: [k[0] for k in v] for o, v in keys.items()}
        self.assertEqual(rounds["round"], sorted(rounds["round"]))
        self.assertEqual([k[2] for k in keys["issue"]],
                         sorted(k[2] for k in keys["issue"]))
        for o in ("issue", "head"):
            # 2 parities x num_bands x one message per round at the head
            self.assertEqual(sorted({k[0] for k in keys[o][:2 * 3 * 3]}),
                             [0, 1, 2], o)
        # `head` keeps round order behind that prefix
        tail = rounds["head"][2 * 3 * 3:]
        self.assertEqual(tail, sorted(tail))
        for o in ("round", "issue", "head"):
            self.assertEqual(sorted(keys[o]), sorted(keys["round"]), o)

    def test_pick_config_reaches_a_full_mxu_pass(self):
        """A whole-y block that only just fits starves the dot, so grid_n = 1
        is not automatically the right choice.

        At k=8192, n_per=3584 the weight takes 56.0 of the 57.9 MiB budget and
        the first-fit rule left bm = 16 — an MXU at ~12% occupancy, measured
        532 us against 165 for the same call at grid_n = 2."""
        for m in (256, 1024, 8192):
            blocks, bm, _, need = aag._pick_config(m, m // 8, 3584, 8192, 2)
            self.assertGreater(len(blocks), 1, m)
            self.assertGreaterEqual(bm, aag._MXU_ROWS, m)
            self.assertLessEqual(need,
                                 aag._VMEM_CAP_BYTES - aag._VMEM_SLACK_BYTES)

    def test_pick_config_prefers_the_fewest_blocks_that_do(self):
        """Each extra n block re-streams the gathered rows (grid_n 3/4 measured
        185/184 us against grid_n 2's 165), so the search stops at the first
        grid_n reaching a pass rather than maximising the dot height."""
        blocks, bm, _, _ = aag._pick_config(1024, 128, 3584, 8192, 2)
        self.assertLen(blocks, 2)
        self.assertEqual(bm, 384)          # grid_n 3 would give bm 512

    def test_pick_config_falls_back_to_the_tallest_dot(self):
        """When no split reaches a full pass the rule must still return the
        best available config, not the first one."""
        blocks, bm, _, _ = aag._pick_config(64, 64, 3584, 8192, 2)
        self.assertEqual(bm, 64)           # m caps it below _MXU_ROWS
        self.assertGreater(len(blocks), 1)

    def test_pick_config_unchanged_on_the_measured_grid(self):
        """The recorded (H, D) x X table was measured at the first-fit configs;
        the rule must reproduce every one of them or the table is stale."""
        want = {(4096, 20736, 8192): (1, 512), (4096, 43776, 8192): (1, 192),
                (8192, 20736, 8192): (1, 256), (8192, 43776, 8192): (2, 192),
                (8192, 44544, 2048): (2, 128), (4096, 22272, 2048): (1, 512),
                (8192, 45056, 256): (2, 192)}
        for (h, d, x), (grid_n, bm) in want.items():
            blocks, got_bm, _, _ = aag._pick_config(x, x // 16, 2 * d // 16, h,
                                                    2)
            self.assertEqual((len(blocks), got_bm), (grid_n, bm), (h, d, x))

    @parameterized.parameters("round", "issue", "head")
    def test_every_order_gathers_correctly(self, order):
        ladder, pieces, loc = aag._plan(3, 512, order)
        self.assertLen({(p["par"], p["label"], p["band"]) for p in pieces},
                       len(pieces))
        offs = sorted((p["g_off"], p["rows"]) for p in pieces)
        at = 0
        for off, rows in offs:  # a partition of [0, m) with no hole or overlap
            self.assertEqual(off, at)
            at += rows
        self.assertEqual(at, 512 * 16)
        self.assertLen(loc, len(pieces))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

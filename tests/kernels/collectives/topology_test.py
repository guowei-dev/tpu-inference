# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the collective kernels' topology derivation.

These run against mocked device coords for BOTH target slices — 2x2x1 (4
chips / 8 devices, the single-host v7x-8) and 2x2x2 (8 chips / 16 devices,
the two-host tpu7x-16) — plus degenerate and failure shapes. On an 8-device
host they are the entire verification story for the 16-device topology: the
ring/hypercube orders and their asserted invariants are pure functions of the
coords, so passing here is what "scales to 2x2x2" means until the kernels run
on the real slice.
"""
import dataclasses

import jax
from absl.testing import absltest
from jax._src import test_util as jtu

from tpu_inference.kernels.collectives import topology

jax.config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class FakeDevice:
    id: int
    coords: tuple


def grid(extents, cores_per_chip=2, coord_offset=(0, 0, 0), order="x_fastest"):
    """Devices over a full chip grid, x-fastest chip order, cores adjacent —
    the layout jax.devices() shows on the recorded v7x hosts."""
    ex, ey, ez = extents
    coords = [(x + coord_offset[0], y + coord_offset[1], z + coord_offset[2])
              for z in range(ez) for y in range(ey) for x in range(ex)]
    assert order == "x_fastest"
    return [
        FakeDevice(id=i, coords=c) for i, c in enumerate(
            cc for cc in coords for _ in range(cores_per_chip))
    ]


DEV_2x2x1 = grid((2, 2, 1))  # this VM: chips (0,0,0),(1,0,0),(0,1,0),(1,1,0)
DEV_2x2x2 = grid((2, 2, 2))  # the tpu7x-16 slice (z = the host boundary)


class RingOrderTest(jtu.JaxTestCase):

    def test_2x2x1_single_hop_cycle(self):
        order = topology.ring_device_order(DEV_2x2x1)
        # snake (0,0),(0,1),(1,1),(1,0) over (x,y) -> the known Hamiltonian
        # cycle this hardware needs (equivalent to the recorded gray order).
        self.assertEqual(order, [0, 1, 4, 5, 6, 7, 2, 3])
        topology.assert_single_hop_ring([DEV_2x2x1[i] for i in order])

    def test_2x2x1_natural_order_is_not_single_hop(self):
        with self.assertRaisesRegex(ValueError, "2 hops"):
            topology.assert_single_hop_ring(DEV_2x2x1)

    def test_2x2x2_single_hop_cycle(self):
        order = topology.ring_device_order(DEV_2x2x2)
        self.assertEqual(sorted(order), list(range(16)))
        topology.assert_single_hop_ring([DEV_2x2x2[i] for i in order])

    def test_cores_stay_adjacent_and_natural(self):
        for devs in (DEV_2x2x1, DEV_2x2x2):
            order = topology.ring_device_order(devs)
            for pos in range(0, len(order), 2):
                a, b = order[pos], order[pos + 1]
                self.assertEqual(devs[a].coords, devs[b].coords)
                self.assertEqual(b, a + 1)

    def test_coord_offset_normalized(self):
        # The z=1 host of the tpu7x-16 slice taken alone.
        shifted = grid((2, 2, 1), coord_offset=(0, 0, 1))
        self.assertEqual(topology.ring_device_order(shifted),
                         topology.ring_device_order(DEV_2x2x1))

    def test_single_chip_is_a_degenerate_ring(self):
        self.assertEqual(topology.ring_device_order(grid((1, 1, 1))), [0, 1])

    def test_1x4_row_has_no_single_hop_cycle(self):
        with self.assertRaisesRegex(ValueError, "3 hops"):
            topology.ring_device_order(grid((4, 1, 1)))

    def test_partial_grid_rejected(self):
        devs = [d for d in DEV_2x2x1 if d.coords != (1, 1, 0)]
        with self.assertRaisesRegex(ValueError, "not a full grid"):
            topology.ring_device_order(devs)


class HierOrderTest(jtu.JaxTestCase):

    def test_2x2x1_labels_and_dims(self):
        labels = topology.chip_labels(DEV_2x2x1)
        self.assertEqual(labels, {(0, 0, 0): 0, (1, 0, 0): 1,
                                  (0, 1, 0): 2, (1, 1, 0): 3})
        self.assertEqual(max(labels.values()).bit_length(), 2)  # 2 hypercube dims

    def test_2x2x2_labels_and_dims(self):
        labels = topology.chip_labels(DEV_2x2x2)
        self.assertEqual(sorted(labels.values()), list(range(8)))
        self.assertEqual(max(labels.values()).bit_length(), 3)  # 3 hypercube dims
        self.assertEqual(labels[(0, 0, 0)], 0)
        self.assertEqual(labels[(1, 0, 0)], 1)
        self.assertEqual(labels[(0, 1, 0)], 2)
        self.assertEqual(labels[(0, 0, 1)], 4)  # z = bit 2 = the host-crossing dim

    def test_xor_partners_adjacent_both_slices(self):
        topology.assert_xor_partners_adjacent(DEV_2x2x1)
        topology.assert_xor_partners_adjacent(DEV_2x2x2)

    def test_hier_order_gives_chip_major_logical_ids(self):
        for devs in (DEV_2x2x1, DEV_2x2x2):
            order = topology.hier_device_order(devs)
            labels = topology.chip_labels(devs)
            ordered = [devs[i] for i in order]
            for logical_id, d in enumerate(ordered):
                self.assertEqual(labels[d.coords], logical_id // 2)
            # twin = id ^ 1 shares the chip
            for logical_id in range(0, len(ordered), 2):
                self.assertEqual(ordered[logical_id].coords,
                                 ordered[logical_id + 1].coords)

    def test_2x2x1_hier_order_is_natural(self):
        # x-fastest natural order already is label-major on this host.
        self.assertEqual(topology.hier_device_order(DEV_2x2x1),
                         list(range(8)))

    def test_extent_3_rejected(self):
        with self.assertRaisesRegex(ValueError, "extent<=2"):
            topology.chip_labels(grid((3, 1, 1)))

    def test_single_chip_zero_dims(self):
        labels = topology.chip_labels(grid((1, 1, 1)))
        self.assertEqual(labels, {(0, 0, 0): 0})
        topology.assert_xor_partners_adjacent(grid((1, 1, 1)))


class SelectPathTest(jtu.JaxTestCase):

    def test_measured_thresholds_tp8(self):
        for pattern in ('ag_mm', 'mm_rs'):
            self.assertEqual(topology.select_path(pattern, 512, 8), 'xla')
            self.assertEqual(topology.select_path(pattern, 1024, 8), 'ring')

    def test_rejects_unknown_pattern(self):
        with self.assertRaises(ValueError):
            topology.select_path('nope', 1024, 8)


class DeriveTopologyTest(jtu.JaxTestCase):

    def test_shapes(self):
        t = topology.derive_topology(DEV_2x2x1)
        self.assertEqual((t.num_devices, t.num_chips, t.cores_per_chip,
                          t.axis_extents), (8, 4, 2, (2, 2, 1)))
        t16 = topology.derive_topology(DEV_2x2x2)
        self.assertEqual((t16.num_devices, t16.num_chips, t16.cores_per_chip,
                          t16.axis_extents), (16, 8, 2, (2, 2, 2)))


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

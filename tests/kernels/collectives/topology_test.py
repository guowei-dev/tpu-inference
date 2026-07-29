# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the all-port kernel's hypercube embedding.

These run against mocked device coords for both target slices — 2x2x1 (4 chips
/ 8 devices) and 2x2x2 (8 chips / 16 devices) — plus degenerate and failure
shapes. The labeling and its invariants are pure functions of the coords, so a
host with 8 devices can still check the 16-device embedding here.
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
    the layout jax.devices() reports on v7x hosts."""
    ex, ey, ez = extents
    coords = [(x + coord_offset[0], y + coord_offset[1], z + coord_offset[2])
              for z in range(ez) for y in range(ey) for x in range(ex)]
    assert order == "x_fastest"
    return [
        FakeDevice(id=i, coords=c) for i, c in enumerate(
            cc for cc in coords for _ in range(cores_per_chip))
    ]


DEV_2x2x1 = grid((2, 2, 1))  # v7x-8: chips (0,0,0),(1,0,0),(0,1,0),(1,1,0)
DEV_2x2x2 = grid((2, 2, 2))  # tpu7x-16 (z = the host boundary)


class HierOrderTest(jtu.JaxTestCase):

    def test_2x2x1_labels_and_dims(self):
        labels = topology.chip_labels(DEV_2x2x1)
        self.assertEqual(labels, {
            (0, 0, 0): 0,
            (1, 0, 0): 1,
            (0, 1, 0): 2,
            (1, 1, 0): 3
        })
        self.assertEqual(max(labels.values()).bit_length(),
                         2)  # 2 hypercube dims

    def test_2x2x2_labels_and_dims(self):
        labels = topology.chip_labels(DEV_2x2x2)
        self.assertEqual(sorted(labels.values()), list(range(8)))
        self.assertEqual(max(labels.values()).bit_length(),
                         3)  # 3 hypercube dims
        self.assertEqual(labels[(0, 0, 0)], 0)
        self.assertEqual(labels[(1, 0, 0)], 1)
        self.assertEqual(labels[(0, 1, 0)], 2)
        self.assertEqual(labels[(0, 0, 1)],
                         4)  # z = bit 2 = the host-crossing dim

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
        self.assertEqual(topology.hier_device_order(DEV_2x2x1), list(range(8)))

    def test_coord_offset_normalized(self):
        # The z=1 host of the tpu7x-16 slice taken alone.
        shifted = grid((2, 2, 1), coord_offset=(0, 0, 1))
        self.assertEqual(topology.hier_device_order(shifted),
                         topology.hier_device_order(DEV_2x2x1))

    def test_extent_3_rejected(self):
        with self.assertRaisesRegex(ValueError, "extent<=2"):
            topology.chip_labels(grid((3, 1, 1)))

    def test_partial_grid_rejected(self):
        devs = [d for d in DEV_2x2x1 if d.coords != (1, 1, 0)]
        with self.assertRaisesRegex(ValueError, "do not fill"):
            topology.chip_labels(devs)

    def test_single_chip_zero_dims(self):
        labels = topology.chip_labels(grid((1, 1, 1)))
        self.assertEqual(labels, {(0, 0, 0): 0})
        topology.assert_xor_partners_adjacent(grid((1, 1, 1)))


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())

"""Capacity must be derived from the lifecycle mode, never fixed globally.

The same number means opposite things in the two modes, because "duration"
means opposite things:

  active  a duration is one hands-on session (median 0.8-2.7 min) and suspend
          releases the resource. 98.4% of real busy time is a single session,
          so capacity 1 is what the log supports; >1 double-counts the
          interleaving that suspend/resume already models.
  legacy  a duration is the whole elapsed start->complete span, mostly
          suspended waiting. Real spans overlap at a median peak of 54 per
          resource, so capacity 1 there would pin a person to one application
          for hours.

Setting one global capacity is therefore wrong in one mode whichever value is
picked. These tests exist to stop that regressing into a single constant.
"""

from __future__ import annotations

import unittest

from simulation.components.resource import (
    DEFAULT_CAPACITY_ACTIVE, DEFAULT_CAPACITY_LEGACY, capacity_for_mode,
)


class CapacityForModeTests(unittest.TestCase):
    def test_active_is_single_tasking(self):
        self.assertEqual(capacity_for_mode("active"), 1)

    def test_legacy_keeps_the_historical_value(self):
        self.assertEqual(capacity_for_mode("legacy"), 3)

    def test_the_two_modes_do_not_agree(self):
        """If these ever converge, one of the two modes is being mismodelled."""
        self.assertNotEqual(DEFAULT_CAPACITY_ACTIVE, DEFAULT_CAPACITY_LEGACY)

    def test_unknown_mode_does_not_get_the_active_default(self):
        """Fail safe: only an explicit 'active' may switch on single-tasking.

        capacity=1 under elapsed-span durations collapses throughput ~50x, so
        an unrecognised mode must not land there by accident.
        """
        self.assertEqual(capacity_for_mode("nonsense"), DEFAULT_CAPACITY_LEGACY)


class CapacityIsWiredThroughTests(unittest.TestCase):
    def test_build_resource_component_defaults_per_mode(self):
        from scripts.run_experiments import build_resource_component

        for mode, expected in (("active", 1), ("legacy", 3)):
            rc = build_resource_component(
                "random", 1, None, None, lifecycle_mode=mode,
            )
            self.assertEqual(rc._capacity, expected, f"mode={mode}")

    def test_explicit_capacity_overrides_the_mode_default(self):
        from scripts.run_experiments import build_resource_component

        rc = build_resource_component(
            "random", 1, None, None, lifecycle_mode="active", capacity=5,
        )
        self.assertEqual(rc._capacity, 5)


if __name__ == "__main__":
    unittest.main()

"""Occupation when the roster leaves a resource with no available time.

Regression test for a nan that only appears once --roster-seed is on: a
part-time resource can draw zero rostered days in a short horizon, so its
availability denominator is 0. Dividing by it poisoned avg_resource_occupation
to nan and took the headline Part II metric with it.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from scripts.opt_metrics import average_resource_occupation

BASE = pd.Timestamp("2016-01-04T09:00:00")


def _log(rows):
    """rows: (resource, start_s, complete_s) -> a minimal paired event log."""
    out = []
    for i, (res, s, c) in enumerate(rows):
        out.append((f"c{i}", "W_Test", BASE + pd.Timedelta(seconds=s), "start", res))
        out.append((f"c{i}", "W_Test", BASE + pd.Timedelta(seconds=c), "complete", res))
    return pd.DataFrame(out, columns=[
        "case:concept:name", "concept:name", "time:timestamp",
        "lifecycle:transition", "org:resource",
    ])


class ZeroAvailabilityTests(unittest.TestCase):
    def test_zero_availability_resource_does_not_poison_the_mean(self):
        df = _log([("r1", 0, 3600)])
        got = average_resource_occupation(
            df, {"r1": 7200.0, "r_never_rostered": 0.0},
        )
        self.assertFalse(math.isnan(got["avg_resource_occupation"]),
                         "a zero-availability resource poisoned the mean")
        self.assertEqual(got["avg_resource_occupation"], 0.5)
        self.assertEqual(got["zero_availability_resources"], ["r_never_rostered"])
        self.assertNotIn("r_never_rostered", got["per_resource"])

    def test_zero_availability_is_not_counted_as_idle(self):
        """0/0 is undefined, not 0.

        Counting a never-rostered resource as occupation 0 would drag the mean
        down in proportion to how much rostering-off the horizon happens to
        contain -- a property of the calendar, not of how hard anyone worked.
        """
        df = _log([("r1", 0, 3600)])
        with_absentee = average_resource_occupation(
            df, {"r1": 7200.0, "r_never_rostered": 0.0},
        )["avg_resource_occupation"]
        without = average_resource_occupation(df, {"r1": 7200.0})
        self.assertEqual(with_absentee, without["avg_resource_occupation"],
                         "absent resource shifted the mean")

    def test_available_but_idle_still_counts_as_zero(self):
        """The case that must NOT be dropped: available, simply did no work."""
        df = _log([("r1", 0, 3600)])
        got = average_resource_occupation(df, {"r1": 7200.0, "r_idle": 7200.0})
        self.assertEqual(got["per_resource"]["r_idle"], 0.0)
        self.assertEqual(got["avg_resource_occupation"], 0.25)
        self.assertEqual(got["zero_availability_resources"], [])

    def test_subset_filter_still_applies(self):
        df = _log([("r1", 0, 3600), ("r2", 0, 7200)])
        got = average_resource_occupation(
            df, {"r1": 7200.0, "r2": 7200.0}, resource_subset=["r1"],
        )
        self.assertEqual(list(got["per_resource"]), ["r1"])
        self.assertEqual(got["avg_resource_occupation"], 0.5)


if __name__ == "__main__":
    unittest.main()

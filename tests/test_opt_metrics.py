from __future__ import annotations

import inspect
import unittest

import pandas as pd

from scripts import opt_metrics
from scripts import run_experiments


def _legacy_log(rows):
    """Build a compact paired start/complete event log for metric tests."""
    base = pd.Timestamp("2016-01-04T09:00:00")
    records = []
    for case_id, activity, resource, start_s, duration_s in rows:
        for transition, seconds in (
            ("start", start_s),
            ("complete", start_s + duration_s),
        ):
            records.append({
                "case:concept:name": case_id,
                "concept:name": activity,
                "org:resource": resource,
                "lifecycle:transition": transition,
                "time:timestamp": base + pd.to_timedelta(seconds, unit="s"),
            })
    return pd.DataFrame(records)


class EvaluationMetricTests(unittest.TestCase):
    def test_run_once_requires_an_explicit_lifecycle(self):
        parameter = inspect.signature(run_experiments.run_once).parameters["lifecycle_mode"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_occupation_excludes_and_reports_zero_availability(self):
        df = _legacy_log([("c1", "A", "r1", 0, 30)])
        result = opt_metrics.average_resource_occupation(
            df,
            availability_seconds={"r1": 60, "r2": 60, "r3": 0},
            resource_subset={"r1", "r2", "r3"},
        )

        self.assertAlmostEqual(result["avg_resource_occupation"], 0.25)
        self.assertEqual(result["per_resource"], {"r1": 0.5, "r2": 0.0})
        self.assertEqual(result["n_resources_evaluated"], 2)
        self.assertEqual(result["zero_availability_resources"], ["r3"])

    def test_resource_activity_switch_rate_follows_each_resource(self):
        df = _legacy_log([
            ("c1", "A", "r1", 0, 5),
            ("c2", "A", "r1", 10, 5),
            ("c3", "B", "r1", 20, 5),
            ("c4", "B", "r2", 1, 5),
            ("c5", "A", "r2", 11, 5),
        ])

        result = opt_metrics.resource_activity_switch_rate(df)

        self.assertEqual(result["n_resource_transitions"], 3)
        self.assertEqual(result["n_activity_switches"], 2)
        self.assertAlmostEqual(result["activity_switch_rate"], 2 / 3)

    def test_rolling_balance_counts_fully_idle_staff(self):
        df = _legacy_log([("c1", "A", "r1", 0, 43_200)])

        result = opt_metrics.rolling_workload_balance(
            df,
            resource_subset={"r1", "r2"},
        )

        # Occupations [0.5, 0.0] have population std 0.25.
        self.assertAlmostEqual(result["mean_window_std"], 0.25)
        self.assertEqual(result["n_windows"], 1)


if __name__ == "__main__":
    unittest.main()

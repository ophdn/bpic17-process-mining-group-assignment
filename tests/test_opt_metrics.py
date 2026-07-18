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

    def test_occupation_can_report_fitted_work_only(self):
        df = _legacy_log([
            ("c1", "A_Create Application", "r1", 0, 20),
            ("c2", "W_Complete application", "r1", 30, 10),
        ])

        result = opt_metrics.average_resource_occupation(
            df,
            availability_seconds={"r1": 100},
            activity_prefixes=("W_",),
        )

        self.assertAlmostEqual(result["avg_resource_occupation"], 0.1)
        self.assertEqual(result["per_resource"], {"r1": 0.1})

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

    def test_resource_activity_switch_rate_can_exclude_atomic_steps(self):
        df = _legacy_log([
            ("c1", "W_Call after offers", "r1", 0, 5),
            ("c2", "A_Validating", "r1", 10, 5),
            ("c3", "A_Validating", "r1", 20, 5),
            ("c4", "W_Call after offers", "r1", 30, 5),
        ])

        result = opt_metrics.resource_activity_switch_rate(
            df, activity_prefixes=("W_",)
        )

        self.assertEqual(result["n_resource_transitions"], 1)
        self.assertEqual(result["n_activity_switches"], 0)
        self.assertEqual(result["activity_switch_rate"], 0.0)

    def test_rolling_balance_counts_fully_idle_staff(self):
        df = _legacy_log([("c1", "A", "r1", 0, 43_200)])

        result = opt_metrics.rolling_workload_balance(
            df,
            resource_subset={"r1", "r2"},
        )

        # Occupations [0.5, 0.0] have population std 0.25.
        self.assertAlmostEqual(result["mean_window_std"], 0.25)
        self.assertEqual(result["n_windows"], 1)

    def test_activity_type_exposure_separates_work_and_atomic_busy_time(self):
        df = _legacy_log([
            ("c1", "A_Create Application", "r1", 0, 10),
            ("c2", "O_Create Offer", "r1", 20, 5),
            ("c3", "W_Complete application", "r2", 0, 7),
        ])

        result = opt_metrics.activity_type_exposure(df)

        self.assertEqual(result["event_rows"], 6)
        self.assertEqual(result["w_event_rows"], 2)
        self.assertEqual(result["ao_event_rows"], 4)
        self.assertAlmostEqual(result["w_event_share"], 1 / 3)
        self.assertAlmostEqual(result["ao_busy_share"], 15 / 22)
        self.assertAlmostEqual(result["w_busy_share"], 7 / 22)
        self.assertEqual(result["busy_time_basis"], "all_active_time")

    def test_lifecycle_diagnostics_exposes_guard_and_rare_routes(self):
        base = pd.Timestamp("2016-01-04T09:00:00")
        rows = [
            ("w1", "W_Complete application", "start", 0),
            ("w1", "W_Complete application", "suspend", 1),
            ("w1", "W_Complete application", "resume", 2),
            ("w1", "W_Complete application", "complete", 3),
            ("w2", "W_Shortened completion ", "start", 0),
            ("w2", "W_Shortened completion ", "withdraw", 1),
        ]
        df = pd.DataFrame([
            {
                "case:concept:name": wid,
                "concept:name": activity,
                "org:resource": "r1" if transition in {"start", "resume"} else None,
                "lifecycle:transition": transition,
                "time:timestamp": base + pd.to_timedelta(seconds, unit="s"),
                "work_item_id": wid,
            }
            for wid, activity, transition, seconds in rows
        ])

        result = opt_metrics.lifecycle_diagnostics(
            df,
            engine_stats={
                "max_session_guard_reached": 1,
                "max_session_guard_forced_completions": 1,
                "max_session_guard_by_activity": {"W_Complete application": 1},
            },
            max_sessions=2,
        )

        self.assertTrue(result["active_lifecycle_schema"])
        self.assertEqual(result["work_items"], 2)
        self.assertEqual(result["median_sessions_per_work_item"], 1.5)
        self.assertEqual(result["max_sessions_per_work_item"], 2)
        self.assertEqual(result["max_session_guard_reached_in_log"], 1)
        self.assertEqual(result["max_session_guard_forced_completions"], 1)
        self.assertEqual(result["withdrawals_by_activity"], {"W_Shortened completion ": 1})
        self.assertEqual(
            result["rare_work_items_routed"]["W_Shortened completion "], 1
        )


if __name__ == "__main__":
    unittest.main()

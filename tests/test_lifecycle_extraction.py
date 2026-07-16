from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from unittest.mock import patch

from extract_log_info import (
    _off_shift_tail_seconds,
    extract_lifecycle,
    extract_processing_times,
)
from train_processing_time_model import build_active_sessions


UTC = timezone.utc


class LifecycleExtractionTests(unittest.TestCase):
    def test_elapsed_extractor_does_not_cross_join_repeated_activity(self):
        base = pd.Timestamp("2016-01-04T09:00:00Z")
        rows = [
            ("c", "W_Test", 0, "start"),
            ("c", "W_Test", 10, "complete"),
            ("c", "W_Test", 20, "start"),
            ("c", "W_Test", 50, "complete"),
        ]
        df = pd.DataFrame(rows, columns=[
            "case_id", "activity", "seconds", "lifecycle"])
        df["timestamp"] = base + pd.to_timedelta(df["seconds"], unit="s")
        df["event_order"] = range(len(df))

        with patch("extract_log_info.fit_best_distribution") as fit:
            fit.return_value = {"distribution": "test", "params": []}
            extract_processing_times(df)

        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.args[0].tolist(), [10.0, 30.0])

    def test_equal_timestamp_continuation_uses_source_order(self):
        base = pd.Timestamp("2016-01-04T09:00:00Z")
        rows = [
            ("c", "W_Test", 0, "schedule", "r"),
            ("c", "W_Test", 1, "start", "r"),
            ("c", "W_Test", 2, "suspend", "r"),
            ("c", "W_Test", 10, "resume", "r"),
            ("c", "W_Test", 11, "complete", "r"),
            # Same timestamp as the terminal, but later in source order.
            ("c", "A_Next", 11, "complete", "r"),
        ]
        df = pd.DataFrame(rows, columns=[
            "case_id", "activity", "seconds", "lifecycle", "resource"])
        df["timestamp"] = base + pd.to_timedelta(df["seconds"], unit="s")
        df["event_order"] = range(len(df))
        calendar = {
            "windows": {"r": {str(i): [8.0, 18.0] for i in range(5)}},
            "holidays": [], "system": [],
        }
        lifecycle = extract_lifecycle(df, calendar)
        self.assertEqual(
            lifecycle["terminal_continuation"]["W_Test"]["complete"],
            {"A_Next": 1.0},
        )

    def test_calendar_tail_uses_resume_resource_and_ignores_vacations(self):
        suspend = datetime(2016, 1, 8, 17, tzinfo=UTC)  # Friday close
        resume = datetime(2016, 1, 11, 7, tzinfo=UTC)   # Monday pre-shift
        calendar = {
            "windows": {"r": {str(i): [9.0, 17.0] for i in range(5)}},
            "holidays": [],
            "vacations": {"r": ["2016-01-11"]},
            "system": ["system"],
        }
        self.assertEqual(
            _off_shift_tail_seconds(suspend, resume, "r", calendar),
            (resume - suspend).total_seconds(),
        )
        self.assertEqual(
            _off_shift_tail_seconds(suspend, resume, "system", calendar), 0.0)

    def test_active_feature_context_is_duplicated_across_sessions(self):
        base = pd.Timestamp("2016-01-04T09:00:00Z")
        rows = [
            ("c", "A_Create Application", 0, "complete", "r"),
            ("c", "W_Test", 1, "schedule", "r"),
            ("c", "W_Test", 2, "start", "r"),
            ("c", "W_Test", 3, "suspend", "r"),
            ("c", "W_Test", 4, "resume", "r2"),
            ("c", "W_Test", 6, "complete", "r2"),
        ]
        df = pd.DataFrame(rows, columns=[
            "case_id", "activity", "seconds", "lifecycle", "resource"])
        df["timestamp"] = base + pd.to_timedelta(df["seconds"], unit="s")
        sessions = build_active_sessions(df)
        self.assertEqual(sessions["duration_s"].tolist(), [1.0, 2.0])
        self.assertEqual(sessions["case_position"].tolist(), [1, 1])
        self.assertEqual(
            sessions["previous_activity"].tolist(),
            ["A_Create Application", "A_Create Application"],
        )
        self.assertEqual(sessions["case_age_seconds"].tolist(), [2.0, 2.0])
        # Option A duplicates the first-start resource across sessions.
        self.assertEqual(sessions["resource"].tolist(), ["r", "r"])


if __name__ == "__main__":
    unittest.main()

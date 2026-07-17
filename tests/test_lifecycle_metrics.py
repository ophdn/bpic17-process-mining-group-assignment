from __future__ import annotations

import unittest

import pandas as pd

from scripts.eval_lifecycle import lifecycle_evidence


class LifecycleMetricTests(unittest.TestCase):
    def test_recomposition_and_resume_ownership_key_on_work_item(self):
        base = pd.Timestamp("2016-01-04T09:00:00")
        rows = [
            ("c", "W_Test", 0, "schedule", "", "w1"),
            ("c", "W_Test", 1, "start", "r1", "w1"),
            ("c", "W_Test", 11, "suspend", "r1", "w1"),
            ("c", "W_Test", 16, "resume", "r2", "w1"),
            ("c", "W_Test", 36, "complete", "r2", "w1"),
            ("c", "W_Test", 40, "schedule", "", "w2"),
            ("c", "W_Test", 47, "withdraw", "", "w2"),
        ]
        df = pd.DataFrame(rows, columns=[
            "case:concept:name", "concept:name", "seconds",
            "lifecycle:transition", "org:resource", "work_item_id",
        ])
        df["time:timestamp"] = base + pd.to_timedelta(df.pop("seconds"), unit="s")

        evidence = lifecycle_evidence(df)

        complete = evidence["terminal_recomposition"]["complete"]
        self.assertEqual(complete["work_items"], 1)
        self.assertEqual(complete["active_seconds"]["mean"], 30.0)
        self.assertEqual(complete["elapsed_seconds"]["mean"], 35.0)
        self.assertEqual(complete["non_active_seconds"]["mean"], 5.0)
        self.assertEqual(complete["suspends"]["mean"], 1.0)
        self.assertEqual(
            evidence["terminal_recomposition"]["withdraw"]["elapsed_seconds"]["mean"],
            7.0,
        )
        self.assertEqual(evidence["resume_ownership"]["total_resumes"], 1)
        self.assertEqual(evidence["resume_ownership"]["same_resource_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Focused tests for evaluation provenance and resource-run guardrails."""

from __future__ import annotations

import hashlib
import unittest

import pandas as pd

from scripts.run_experiments import (
    EVALUATION_PROVENANCE_PATHS,
    REPO_ROOT,
    evaluation_provenance_hashes,
    validate_resource_diagnostics,
)


class EvaluationProvenanceTests(unittest.TestCase):
    def test_manifest_covers_simulator_and_fitted_inputs(self):
        required = {
            "simulation/components/resource.py",
            "simulation/components/process.py",
            "scripts/run_experiments.py",
            "models/availability_model.json",
            "simulation_inputs_active.json",
        }
        self.assertTrue(required.issubset(EVALUATION_PROVENANCE_PATHS))

        hashes = evaluation_provenance_hashes()
        self.assertEqual(set(hashes), set(EVALUATION_PROVENANCE_PATHS))
        for relative_path in required:
            expected = hashlib.sha256(
                (REPO_ROOT / relative_path).read_bytes()
            ).hexdigest()
            self.assertEqual(hashes[relative_path], expected)


class ResourceDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _event_log(resource="r1"):
        return pd.DataFrame({
            "lifecycle:transition": ["schedule", "start", "resume", "complete"],
            "org:resource": [None, resource, resource, resource],
        })

    def test_valid_run_records_queue_and_maximum_occupation(self):
        diagnostics = validate_resource_diagnostics(
            self._event_log(),
            {"unpermitted_activities": 0, "still_queued_at_end": 7},
            {"r1": 0.8, "r2": 0.0},
            capacity=1,
        )
        self.assertEqual(diagnostics["still_queued_at_end"], 7)
        self.assertEqual(diagnostics["missing_resource_starts"], 0)
        self.assertEqual(diagnostics["max_resource_occupation"], 0.8)

    def test_unassigned_active_event_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unassigned start/resume"):
            validate_resource_diagnostics(
                self._event_log(resource=None),
                {"unpermitted_activities": 0, "still_queued_at_end": 0},
                {"r1": 0.5},
                capacity=1,
            )

    def test_unpermitted_activity_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unpermitted activities"):
            validate_resource_diagnostics(
                self._event_log(),
                {"unpermitted_activities": 1, "still_queued_at_end": 0},
                {"r1": 0.5},
                capacity=1,
            )

    def test_unit_capacity_occupation_above_one_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "exceeds unit capacity"):
            validate_resource_diagnostics(
                self._event_log(),
                {"unpermitted_activities": 0, "still_queued_at_end": 0},
                {"r1": 1.0001},
                capacity=1,
            )

    def test_runner_exposes_resource_stats_and_effective_defaults(self):
        from scripts.run_experiments import run_once

        _, meta = run_once(
            "random", 1, 0, "normal", True, "basic", "probs",
            lifecycle_mode="active", permissions="observed",
        )
        self.assertEqual(meta["configuration"]["capacity"], 1)
        self.assertEqual(meta["configuration"]["roster_seed"], 43)
        self.assertIn(meta["configuration"]["arrival_model"], {"mdn", "parametric"})
        self.assertEqual(meta["resource_stats"]["unpermitted_activities"], 0)
        self.assertEqual(meta["resource_stats"]["still_queued_at_end"], 0)


if __name__ == "__main__":
    unittest.main()

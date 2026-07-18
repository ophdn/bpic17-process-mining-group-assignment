"""Regression tests for permission-independent case attributes."""

import unittest

from scripts.run_experiments import load_permission_model
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.process import ProcessComponent
from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.main import DEFAULT_BPMN_PATH, load_permission_context


class CaseAttributeWiringTests(unittest.TestCase):
    def test_every_permission_mode_keeps_case_attributes(self):
        for kind in ("orgmodel", "observed", "hardcoded"):
            with self.subTest(kind=kind):
                permissions, sampler = load_permission_context(kind, seed=17)
                self.assertIsNotNone(sampler)
                self.assertIn("case:LoanGoal", sampler.sample())
                if kind == "hardcoded":
                    self.assertIsNone(permissions)
                else:
                    self.assertIsNotNone(permissions)

    def test_experiment_runner_uses_the_shared_wiring(self):
        _, sampler = load_permission_model("observed", seed=17)
        self.assertIsNotNone(sampler)
        self.assertIn("case:LoanGoal", sampler.sample())

    def test_observed_permissions_propagate_loan_goal_to_event_payload(self):
        _, sampler = load_permission_context("observed", seed=17)
        process = ProcessComponent(seed=17, case_attributes=sampler)
        engine = SimulationEngine(sim_duration=10)

        process.on_activity_start(engine, SimEvent(
            timestamp=0,
            event_type=EventType.ACTIVITY_START,
            case_id="case-1",
            activity="__PROCESS_START__",
        ))

        payload = process._payload("case-1")
        self.assertIn("case:LoanGoal", payload)
        self.assertEqual(
            payload["case_type"],
            f"CT.{payload['case:LoanGoal']}",
        )

    def test_advanced_visit_model_propagates_observed_mode_attributes(self):
        _, sampler = load_permission_context("observed", seed=17)
        process = PetriNetProcessComponent(
            bpmn_path=str(DEFAULT_BPMN_PATH),
            branching_mode="visit",
            seed=17,
            case_attributes=sampler,
        )
        engine = SimulationEngine(sim_duration=10)

        process.on_activity_start(engine, SimEvent(
            timestamp=0,
            event_type=EventType.ACTIVITY_START,
            case_id="case-advanced",
            activity="__PROCESS_START__",
        ))

        payload = process._payload("case-advanced")
        self.assertIn("case:LoanGoal", payload)
        self.assertEqual(
            payload["case_type"],
            f"CT.{payload['case:LoanGoal']}",
        )

    def test_unknown_permission_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown permissions kind"):
            load_permission_context("missing", seed=17)


if __name__ == "__main__":
    unittest.main()

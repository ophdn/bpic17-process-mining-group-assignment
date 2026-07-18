"""Focused tests for D3 masked resource-allocation decisions.

These tests exercise the simulator-facing DRL seam without importing the
optional Gymnasium/PyTorch stack.
"""

from datetime import datetime
import unittest

import numpy as np

from simulation.components.permissions import StaticPermissions
from simulation.components.resource import ResourceComponent
from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent

try:
    import gymnasium  # noqa: F401
    HAVE_GYMNASIUM = True
except ImportError:
    HAVE_GYMNASIUM = False


class TestDRLResourceAllocation(unittest.TestCase):
    def setUp(self):
        permissions = StaticPermissions({"r1": {"A"}, "r2": {"A", "B"}})
        self.resources = ResourceComponent(
            capacity_per_resource=1,
            start_datetime=datetime(2016, 1, 1),
            permissions=permissions,
            drl=True,
            drl_external_control=True,
        )
        self.engine = SimulationEngine(
            sim_duration=100,
            start_datetime=datetime(2016, 1, 1),
        )
        self.engine.register(self.resources)

    def request(self, activity="A", case_id="c1", timestamp=0):
        self.engine.schedule(SimEvent(
            timestamp=timestamp,
            event_type=EventType.ACTIVITY_REQUEST,
            activity=activity,
            case_id=case_id,
        ))
        self.assertTrue(self.engine.step())

    def test_mask_contains_only_permitted_live_assignments(self):
        self.request("B")
        mask = self.resources.drl_action_mask(self.engine)

        with self.assertRaisesRegex(ValueError, "globally ineligible"):
            self.resources.drl_action_for("r1", "B")
        self.assertTrue(mask[self.resources.drl_action_for("r2", "B")])
        self.assertTrue(mask[self.resources.drl_postpone_action])
        self.assertEqual(mask.sum(), 2)
        self.assertEqual(self.resources.drl_action_count, 4)
        self.assertTrue(self.resources.drl_decision_pending)

    def test_v1_cartesian_actions_remain_available_for_old_models(self):
        resources = ResourceComponent(
            permissions=StaticPermissions({"r1": {"A"}, "r2": {"A", "B"}}),
            drl=True,
            drl_external_control=True,
            drl_action_version=1,
        )
        self.assertEqual(resources.drl_action_count, 5)
        self.assertEqual(resources.drl_action_for("r1", "B"), 1)

    def test_spt_expert_chooses_shortest_feasible_activity(self):
        self.request("A", "c1")
        self.request("B", "c2")
        self.resources._drl_expected_duration.update({"A": 20.0, "B": 10.0})
        action = self.resources.drl_shortest_processing_action(self.engine)
        self.assertEqual(
            self.resources._drl_decode_action(action),
            ("r2", "B"),
        )

    def test_assignment_starts_oldest_compatible_request(self):
        self.request("A", "old")
        action = self.resources.drl_action_for("r1", "A")
        self.resources.apply_drl_action(self.engine, action)

        self.assertEqual(self.resources.stats()["drl_assignments"], 1)
        self.assertEqual(self.resources.stats()["still_queued_at_end"], 0)
        self.assertFalse(self.resources.drl_decision_pending)
        self.assertEqual(self.resources._busy["r1"], 1)

        self.assertTrue(self.engine.step())
        self.assertEqual(self.engine.logger._rows[-1]["case:concept:name"], "old")
        self.assertEqual(self.engine.logger._rows[-1]["org:resource"], "r1")

    def test_postpone_waits_for_a_changed_decision_epoch(self):
        self.request("A", "c1")
        self.resources.apply_drl_action(
            self.engine, self.resources.drl_postpone_action)
        self.assertFalse(self.resources.drl_decision_pending)
        self.assertEqual(self.resources.stats()["drl_postponements"], 1)
        self.assertEqual(self.resources.stats()["still_queued_at_end"], 1)

        self.request("B", "c2", timestamp=1)
        self.assertTrue(self.resources.drl_decision_pending)

    def test_observation_is_fixed_normalized_float_vector(self):
        self.request("A")
        observation = self.resources.drl_observation(self.engine)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation.shape, (self.resources.drl_observation_size,))
        self.assertTrue(np.all(observation >= 0.0))
        self.assertTrue(np.all(observation <= 1.0))

    def test_v3_observation_tracks_active_activity(self):
        self.request("A")
        observation_before = self.resources.drl_observation(self.engine)
        action = self.resources.drl_action_for("r1", "A")
        self.resources.apply_drl_action(self.engine, action)
        observation_after = self.resources.drl_observation(self.engine)
        self.assertEqual(self.resources._drl_observation_version, 3)
        self.assertFalse(np.array_equal(observation_before, observation_after))

    def test_v2_observation_shape_remains_available_for_old_models(self):
        resources = ResourceComponent(
            permissions=StaticPermissions({"r1": {"A"}}),
            drl=True,
            drl_external_control=True,
            drl_observation_version=2,
        )
        self.assertEqual(resources.drl_observation_size, 11)

    def test_drl_is_mutually_exclusive_with_other_queue_disciplines(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ResourceComponent(
                permissions=StaticPermissions({"r1": {"A"}}),
                drl=True,
                drl_external_control=True,
                parksong=True,
            )

    @unittest.skipUnless(HAVE_GYMNASIUM, "optional requirements-drl.txt not installed")
    def test_gym_environment_can_finish_and_reset_an_episode(self):
        from simulation.drl import ResourceAllocationEnv

        def factory(seed):
            resource = ResourceComponent(
                permissions=StaticPermissions({"r1": {"A"}}),
                drl=True,
                drl_external_control=True,
            )
            engine = SimulationEngine(sim_duration=1)
            engine.register(resource)
            engine.schedule(SimEvent(
                timestamp=0,
                event_type=EventType.ACTIVITY_REQUEST,
                case_id=f"c{seed}",
                activity="A",
            ))
            return engine, resource

        env = ResourceAllocationEnv(factory)
        observation, _ = env.reset()
        self.assertEqual(observation.shape, env.observation_space.shape)
        action = env.resources.drl_action_for("r1", "A")
        _, _, terminated, truncated, _ = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

        observation, _ = env.reset()
        self.assertFalse(env._terminated)
        self.assertTrue(env.resources.drl_decision_pending)
        self.assertEqual(observation.shape, env.observation_space.shape)

    @unittest.skipUnless(HAVE_GYMNASIUM, "optional requirements-drl.txt not installed")
    def test_evaluation_seed_cycle_is_distinct_and_reproducible(self):
        from simulation.drl import ResourceAllocationEnv

        built_seeds = []

        def factory(seed):
            built_seeds.append(seed)
            resource = ResourceComponent(
                permissions=StaticPermissions({"r1": {"A"}}),
                drl=True,
                drl_external_control=True,
            )
            engine = SimulationEngine(sim_duration=1)
            engine.register(resource)
            return engine, resource

        env = ResourceAllocationEnv(factory, episode_seeds=[100, 101, 102])
        env.reset()  # Reuses the episode built during construction.
        env.reset()
        env.reset()
        env.reset()

        self.assertEqual(built_seeds, [100, 101, 102, 100])


if __name__ == "__main__":
    unittest.main()

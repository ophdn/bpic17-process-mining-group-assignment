"""Temporal permission gaps must queue work rather than run it unassigned."""

from __future__ import annotations

from datetime import datetime
import unittest

from simulation.components.permissions import OrgModelPermissions
from simulation.components.resource import ResourceComponent
from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent


class TemporalPermissionTests(unittest.TestCase):
    def setUp(self):
        self.permissions = OrgModelPermissions([{
            "members": ["r1"],
            "capabilities": [["CT.Test", "W_Test", "TT.Monday"]],
        }])
        self.engine = SimulationEngine(
            sim_duration=0,
            start_datetime=datetime(2016, 1, 10),  # Sunday
            verbose=False,
        )
        self.resources = ResourceComponent(
            capacity_per_resource=1,
            permissions=self.permissions,
            start_datetime=datetime(2016, 1, 10),
        )

    @staticmethod
    def request(activity="W_Test"):
        return SimEvent(
            timestamp=0.0,
            priority=5,
            event_type=EventType.ACTIVITY_REQUEST,
            case_id="case-1",
            activity=activity,
            payload={"case_type": "CT.Test"},
        )

    def test_weekday_capability_is_ever_qualified_on_sunday(self):
        event = self.request()
        self.assertEqual(
            self.permissions.candidates(
                event.activity,
                case_type=event.payload["case_type"],
                when=datetime(2016, 1, 10),
            ),
            [],
        )
        self.assertTrue(self.resources._qualified(self.engine, event))

    def test_temporal_gap_queues_instead_of_running_unassigned(self):
        event = self.request()
        self.resources.on_activity_request(self.engine, event)

        self.assertEqual(self.resources._waiting, [event])
        self.assertEqual(self.resources.stats()["unpermitted_activities"], 0)
        self.assertFalse(self.engine._queue, "no unassigned start should be scheduled")

    def test_genuinely_unknown_activity_remains_unpermitted(self):
        self.assertFalse(
            self.resources._qualified(self.engine, self.request("W_Unknown"))
        )

    def test_unseen_case_activity_pair_backs_off_to_activity_roles(self):
        event = self.request()
        event.payload["case_type"] = "CT.Rare"
        monday = datetime(2016, 1, 11)

        self.assertEqual(
            self.permissions.candidates(
                event.activity,
                case_type=event.payload["case_type"],
                when=monday,
            ),
            ["r1"],
        )
        self.assertTrue(
            self.permissions.permits(
                "r1",
                event.activity,
                case_type=event.payload["case_type"],
                when=monday,
            )
        )

    def test_case_type_backoff_preserves_weekday_restrictions(self):
        event = self.request()
        event.payload["case_type"] = "CT.Rare"

        self.assertEqual(
            self.permissions.candidates(
                event.activity,
                case_type=event.payload["case_type"],
                when=datetime(2016, 1, 10),  # Sunday
            ),
            [],
        )

    def test_known_case_activity_does_not_back_off_across_weekdays(self):
        permissions = OrgModelPermissions([
            {
                "members": ["r1"],
                "capabilities": [["CT.Test", "W_Test", "TT.Monday"]],
            },
            {
                "members": ["r2"],
                "capabilities": [["CT.Other", "W_Test", "TT.Sunday"]],
            },
        ])

        self.assertEqual(
            permissions.candidates(
                "W_Test",
                case_type="CT.Test",
                when=datetime(2016, 1, 10),  # Sunday
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()

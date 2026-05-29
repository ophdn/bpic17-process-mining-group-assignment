"""
Stub / placeholder components for the Simulation Engine.

These stubs let you run the engine end-to-end right now.
They are designed to be replaced (or extended) by the real
components your group implements for:
  - Case arrivals        (section 1.2)
  - Processing times     (section 1.3)
  - Process model        (section 1.4)
  - Branching decisions  (section 1.5)
  - Resource availability (section 1.6)
  - Resource permissions  (section 1.7)
  - Resource allocation   (section 1.8)
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.events import SimEvent, EventType


# ---------------------------------------------------------------------------
# 1.  Case Arrival Component (Basic: exponential inter-arrival times)
# ---------------------------------------------------------------------------

class ArrivalComponent:
    """
    Generates new cases by scheduling CASE_ARRIVAL events.

    Basic implementation: inter-arrival times ~ Exponential(rate).
    Replace with a learned distribution once you have the BPIC-17 data.

    Parameters
    ----------
    rate : float
        Average arrivals per second. E.g. 1/3600 = one case per hour.
    seed : int, optional
        Random seed for reproducibility.
    """

    HANDLES = {EventType.CASE_ARRIVAL: None}  # filled below

    def __init__(self, rate: float = 1 / 3600, seed: Optional[int] = 42):
        self.rate = rate
        self._rng = random.Random(seed)
        self._case_counter = 0

    def bootstrap(self, engine) -> None:
        """Schedule the very first arrival. Call before engine.run()."""
        self._schedule_next(engine, current_time=0.0)

    def on_arrival(self, engine, event: SimEvent) -> None:
        """Handle a CASE_ARRIVAL: kick off the case, schedule the next arrival."""
        case_id = event.case_id

        # Hand off to the process component via ACTIVITY_START
        # (The process component decides which activity comes first.)
        first_act_event = SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_START,
            case_id=case_id,
            activity="__PROCESS_START__",   # sentinel; process component interprets this
            payload={"attributes": event.payload or {}},
        )
        engine.schedule(first_act_event)

        # Schedule the next arrival
        self._schedule_next(engine, current_time=engine.now)

    # -- private --

    def _schedule_next(self, engine, current_time: float) -> None:
        self._case_counter += 1
        inter_arrival = self._rng.expovariate(self.rate)
        engine.schedule(SimEvent(
            timestamp=current_time + inter_arrival,
            priority=10,
            event_type=EventType.CASE_ARRIVAL,
            case_id=f"case_{self._case_counter:06d}",
            payload={},
        ))


# Patch HANDLES after class definition so the method reference is valid
ArrivalComponent.HANDLES = {EventType.CASE_ARRIVAL: ArrivalComponent.on_arrival}


# ---------------------------------------------------------------------------
# 2.  Process Component (Basic stub: linear sequence of activities)
# ---------------------------------------------------------------------------

class ProcessComponent:
    """
    Enforces a simple linear process model.

    Replace with a Petri-net or BPMN-based model for section 1.4.

    Parameters
    ----------
    activities : list[str]
        Ordered list of activity names (no branching).
    mean_durations : dict[str, float]
        Mean processing time in seconds for each activity.
        Falls back to 600s (10 min) if not specified.
    seed : int, optional
    """

    HANDLES = {
        EventType.ACTIVITY_START: None,
        EventType.ACTIVITY_COMPLETE: None,
    }

    def __init__(
        self,
        activities: List[str],
        mean_durations: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ):
        self.activities = activities
        self.mean_durations = mean_durations or {}
        self._rng = random.Random(seed)
        # Track current activity index per case
        self._case_progress: Dict[str, int] = {}

    def on_activity_start(self, engine, event: SimEvent) -> None:
        case_id = event.case_id

        # Sentinel: first event for this case
        if event.activity == "__PROCESS_START__":
            self._case_progress[case_id] = 0
            self._start_activity(engine, case_id)
            return

        # Normal start: sample duration, schedule completion
        activity = event.activity
        mean = self.mean_durations.get(activity, 600.0)
        duration = self._rng.expovariate(1.0 / mean)

        engine.schedule(SimEvent(
            timestamp=engine.now + duration,
            priority=5,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            resource=event.resource,
        ))

    def on_activity_complete(self, engine, event: SimEvent) -> None:
        case_id = event.case_id
        idx = self._case_progress.get(case_id, 0) + 1
        self._case_progress[case_id] = idx

        if idx < len(self.activities):
            self._start_activity(engine, case_id)
        else:
            # Case is done
            del self._case_progress[case_id]
            engine.schedule(SimEvent(
                timestamp=engine.now,
                priority=20,
                event_type=EventType.CASE_COMPLETE,
                case_id=case_id,
            ))

    def _start_activity(self, engine, case_id: str) -> None:
        idx = self._case_progress[case_id]
        activity = self.activities[idx]
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_START,
            case_id=case_id,
            activity=activity,
            resource="resource_stub",   # replaced by ResourceComponent
        ))


ProcessComponent.HANDLES = {
    EventType.ACTIVITY_START: ProcessComponent.on_activity_start,
    EventType.ACTIVITY_COMPLETE: ProcessComponent.on_activity_complete,
}

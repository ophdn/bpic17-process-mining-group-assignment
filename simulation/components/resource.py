"""
resource.py — Resource Component (Sections 1.6, 1.7, 1.8 Basic)
================================================================
Implements:
  - Section 1.7 Basic: resource permissions based on which resources
    have historically performed each activity (from BPIC-17 data)
  - Section 1.8:       random allocation among permitted resources
  - Section 1.6 Basic: simple availability model — each resource has
    a fixed capacity (max parallel tasks); tasks queue if all busy.

The resource_activity_map is derived directly from the BPIC-17 log.
Only the top-20 resources (by event count) are included for performance;
extend RESOURCE_POOL with additional users as needed.

Upgrade path:
  - Section 1.6 Advanced: calendar-based availability (shift patterns)
  - Section 1.7 Advanced: role-discovery (e.g. OrdinoR)
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..core.events import SimEvent, EventType


# ── Resource permission map (from BPIC-17 resource_activity_map) ─────────────
# resource -> set of activities it is allowed to perform
# Only top-20 resources included; all have been observed performing these activities.
RESOURCE_PERMISSIONS: Dict[str, Set[str]] = {
    "User_1": {
        "A_Cancelled", "A_Concept", "A_Create Application", "A_Submitted",
        "O_Cancelled", "W_Assess potential fraud", "W_Call after offers",
        "W_Call incomplete files", "W_Complete application", "W_Handle leads",
        "W_Validate application",
    },
    "User_2": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Handle leads", "W_Shortened completion ",
        "W_Validate application",
    },
    "User_3": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Validating",
        "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Call after offers", "W_Call incomplete files", "W_Complete application",
        "W_Handle leads", "W_Validate application",
    },
    "User_5": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Validating",
        "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Handle leads", "W_Personal Loan collection",
        "W_Shortened completion ", "W_Validate application",
    },
    "User_27": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Handle leads", "W_Validate application",
    },
    "User_29": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Validate application",
    },
    "User_30": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Shortened completion ", "W_Validate application",
    },
    "User_49": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Validating",
        "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Call after offers", "W_Call incomplete files", "W_Complete application",
        "W_Handle leads", "W_Shortened completion ", "W_Validate application",
    },
    "User_68": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Validate application",
    },
    "User_75": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Shortened completion ", "W_Validate application",
    },
    "User_87": {
        "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Validate application",
    },
    "User_100": {
        "A_Accepted", "A_Complete", "A_Concept", "A_Create Application",
        "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
        "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
        "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
        "W_Complete application", "W_Validate application",
    },
    "User_113": {
        "A_Cancelled", "A_Denied", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Refused", "O_Returned",
        "W_Call after offers", "W_Call incomplete files", "W_Validate application",
    },
    "User_116": {
        "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Returned",
        "W_Call after offers", "W_Call incomplete files", "W_Validate application",
    },
    "User_118": {
        "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Returned",
        "O_Sent (mail and online)",
        "W_Call after offers", "W_Call incomplete files", "W_Complete application",
        "W_Handle leads", "W_Validate application",
    },
    "User_121": {
        "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Returned",
        "O_Sent (mail and online)",
        "W_Call after offers", "W_Call incomplete files", "W_Validate application",
    },
    "User_123": {
        "A_Cancelled", "A_Incomplete", "A_Pending", "A_Validating",
        "O_Accepted", "O_Cancelled", "O_Returned",
        "W_Call after offers", "W_Call incomplete files", "W_Validate application",
    },
}

# Precompute inverse map: activity -> list of permitted resources
_ACTIVITY_TO_RESOURCES: Dict[str, List[str]] = defaultdict(list)
for _res, _acts in RESOURCE_PERMISSIONS.items():
    for _act in _acts:
        _ACTIVITY_TO_RESOURCES[_act].append(_res)


class ResourceComponent:
    """
    Assigns resources to activities and manages basic availability.

    Section 1.7 Basic: only resources that have historically performed
    an activity are candidates (from BPIC-17 resource_activity_map).

    Section 1.8: random selection among available permitted resources.

    Section 1.6 Basic: each resource has a capacity (default: 1 parallel task).
    If all permitted resources are busy, the task is queued and retried
    when a resource becomes free.
    """

    HANDLES = {
        EventType.ACTIVITY_START:    None,
        EventType.RESOURCE_AVAILABLE: None,
    }

    def __init__(self, capacity_per_resource: int = 1, seed: Optional[int] = 42):
        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)

        # resource -> current number of active tasks
        self._busy: Dict[str, int] = {r: 0 for r in RESOURCE_PERMISSIONS}

        # Queue of (engine, event) waiting for a free resource
        self._waiting: List[tuple] = []

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        """Intercept ACTIVITY_START to assign a resource before it runs."""
        if event.activity in ("__PROCESS_START__",):
            return  # sentinel — no resource needed

        resource = self._allocate(event.activity)
        if resource:
            event.resource = resource
            self._busy[resource] = self._busy.get(resource, 0) + 1
        else:
            # All permitted resources busy → queue and wait
            self._waiting.append((engine, event))

    def on_resource_available(self, engine, event: SimEvent) -> None:
        """When a resource frees up, try to unblock a waiting task."""
        resource = event.resource
        self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)

        # Try to assign this resource to a waiting activity it can perform
        for i, (eng, waiting_event) in enumerate(self._waiting):
            if resource in (RESOURCE_PERMISSIONS.get(resource, set())):
                if waiting_event.activity in RESOURCE_PERMISSIONS.get(resource, set()):
                    self._waiting.pop(i)
                    waiting_event.resource = resource
                    self._busy[resource] = self._busy.get(resource, 0) + 1
                    eng.schedule(waiting_event)
                    return

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _allocate(self, activity: str) -> Optional[str]:
        """
        Randomly pick an available permitted resource for this activity.
        Returns None if all permitted resources are busy.
        """
        candidates = _ACTIVITY_TO_RESOURCES.get(activity, [])
        available = [
            r for r in candidates
            if self._busy.get(r, 0) < self._capacity
        ]
        if not available:
            return None
        return self._rng.choice(available)

    def release(self, engine, resource: str) -> None:
        """
        Call this from the ProcessComponent after ACTIVITY_COMPLETE
        to free the resource. Schedules a RESOURCE_AVAILABLE event.
        """
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=1,   # High priority: free up before next tasks start
            event_type=EventType.RESOURCE_AVAILABLE,
            resource=resource,
        ))


ResourceComponent.HANDLES = {
    EventType.ACTIVITY_START:     ResourceComponent.on_activity_start,
    EventType.RESOURCE_AVAILABLE: ResourceComponent.on_resource_available,
}

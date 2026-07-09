"""
resource.py — Resource Component (Sections 1.6, 1.7, 1.8 Basic)
================================================================
Implements:
  - Section 1.7 Basic: resource permissions based on which resources
    have historically performed each activity (from BPIC-17 data)
  - Section 1.8:       pluggable allocation policies among permitted
    resources (random / least_loaded / round_robin / specialization)
  - Section 1.6 Basic: simple availability model — each resource has
    a fixed capacity (max parallel tasks); tasks queue if all busy.

The resource_activity_map is derived directly from the BPIC-17 log.
Only the top-20 resources (by event count) are included for performance;
extend RESOURCE_POOL with additional users as needed.

Allocation policies (Section 1.8) — selectable via ``policy=``:
  - "random"         : uniform among available permitted resources (baseline).
  - "least_loaded"   : fewest active tasks first (load balancing).
  - "round_robin"    : deterministic rotation (maximally even distribution).
  - "specialization" : most-experienced resource first, where experience is
                       the historical event volume (RESOURCE_EXPERIENCE). This
                       concentrates work on senior staff — a deliberate
                       efficiency-vs-fairness contrast for the evaluation.

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

# ── Resource experience (from BPIC-17 top_20_by_events) ─────────────────────
# resource -> total number of events historically performed. Used as a
# seniority / experience proxy by the "specialization" allocation policy.
RESOURCE_EXPERIENCE: Dict[str, int] = {
    "User_1": 148404, "User_2": 19134, "User_3": 26342, "User_5": 22900,
    "User_27": 18806, "User_29": 20860, "User_30": 21272, "User_49": 21134,
    "User_68": 17581, "User_75": 15955, "User_87": 22498, "User_100": 20651,
    "User_113": 16151, "User_116": 17423, "User_118": 16005, "User_121": 18726,
    "User_123": 20909,
}

# Precompute inverse map: activity -> list of permitted resources
_ACTIVITY_TO_RESOURCES: Dict[str, List[str]] = defaultdict(list)
for _res, _acts in RESOURCE_PERMISSIONS.items():
    for _act in _acts:
        _ACTIVITY_TO_RESOURCES[_act].append(_res)
# Deterministic candidate order (round_robin / tie-breaks depend on it).
for _act in _ACTIVITY_TO_RESOURCES:
    _ACTIVITY_TO_RESOURCES[_act].sort()


class ResourceComponent:
    """
    Assigns resources to activities and manages basic availability.

    Section 1.7 Basic: only resources that have historically performed
    an activity are candidates (from BPIC-17 resource_activity_map).

    Section 1.8: pluggable allocation among available permitted resources.
    See module docstring / ``POLICIES`` for the available strategies.

    Section 1.6 Basic: each resource has a capacity (default: 1 parallel task).
    If all permitted resources are busy, the task is queued and retried
    when a resource becomes free.
    """

    HANDLES = {
        EventType.ACTIVITY_ENABLED:   None,
        EventType.RESOURCE_AVAILABLE: None,
    }

    # Available allocation strategies (Section 1.8).
    POLICIES = ("random", "least_loaded", "round_robin", "specialization")

    def __init__(
        self,
        capacity_per_resource: int = 1,
        seed: Optional[int] = 42,
        policy: str = "random",
    ):
        if policy not in self.POLICIES:
            raise ValueError(
                f"policy must be one of {self.POLICIES}, got {policy!r}"
            )
        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)
        self.policy = policy

        # resource -> current number of active tasks
        self._busy: Dict[str, int] = {r: 0 for r in RESOURCE_PERMISSIONS}

        # activity -> rotation cursor (round_robin policy only)
        self._rr_cursor: Dict[str, int] = defaultdict(int)

        # Queue of (engine, event) waiting for a free resource
        self._waiting: List[tuple] = []

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_enabled(self, engine, event: SimEvent) -> None:
        """
        An activity became ready. Try to seize a permitted resource; on success
        emit the ACTIVITY_START that begins service. If none is free, the
        activity waits in the queue until a resource frees up (Section 1.6) —
        so contention actually delays execution.
        """
        if event.activity == "__PROCESS_START__":
            return  # sentinel — no resource needed

        resource = self._allocate(event.activity)
        if resource:
            self._busy[resource] = self._busy.get(resource, 0) + 1
            self._start_activity(engine, event, resource)
        else:
            # All permitted resources busy → queue and wait
            self._waiting.append((engine, event))

    def on_resource_available(self, engine, event: SimEvent) -> None:
        """When a resource frees up, hand it to the first waiting task it can do."""
        resource = event.resource
        self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)

        # FIFO scan: give this resource to the first queued activity it is
        # permitted for (and has spare capacity for).
        permitted = RESOURCE_PERMISSIONS.get(resource, set())
        for i, (eng, waiting_event) in enumerate(self._waiting):
            if self._busy.get(resource, 0) >= self._capacity:
                break
            if waiting_event.activity in permitted:
                self._waiting.pop(i)
                self._busy[resource] = self._busy.get(resource, 0) + 1
                self._start_activity(eng, waiting_event, resource)
                return

    def _start_activity(self, engine, enabled_event: SimEvent, resource: str) -> None:
        """Emit the ACTIVITY_START that begins service, with the seized resource."""
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=4,   # before ACTIVITY_COMPLETE (5) at the same instant
            event_type=EventType.ACTIVITY_START,
            case_id=enabled_event.case_id,
            activity=enabled_event.activity,
            resource=resource,
            payload=enabled_event.payload,
        ))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _allocate(self, activity: str) -> Optional[str]:
        """
        Pick an available permitted resource for this activity according to
        the configured allocation policy (Section 1.8).
        Returns None if all permitted resources are busy.
        """
        candidates = _ACTIVITY_TO_RESOURCES.get(activity, [])
        available = [
            r for r in candidates
            if self._busy.get(r, 0) < self._capacity
        ]
        if not available:
            return None

        policy = self.policy

        if policy == "random":
            return self._rng.choice(available)

        if policy == "least_loaded":
            # Fewest active tasks first; break ties randomly for fairness.
            min_load = min(self._busy.get(r, 0) for r in available)
            least = [r for r in available if self._busy.get(r, 0) == min_load]
            return self._rng.choice(least)

        if policy == "round_robin":
            # Deterministic rotation over the (sorted) candidate list. The
            # cursor walks the full candidate order so load spreads evenly
            # even when some candidates are momentarily busy.
            n = len(candidates)
            start = self._rr_cursor[activity]
            for offset in range(n):
                idx = (start + offset) % n
                res = candidates[idx]
                if self._busy.get(res, 0) < self._capacity:
                    self._rr_cursor[activity] = idx + 1
                    return res
            return None  # unreachable: available was non-empty

        if policy == "specialization":
            # Most-experienced available resource first (seniority proxy).
            # Deterministic; name as final tie-break for reproducibility.
            return max(
                available,
                key=lambda r: (RESOURCE_EXPERIENCE.get(r, 0), r),
            )

        # Defensive: validated in __init__, should never reach here.
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
    EventType.ACTIVITY_ENABLED:   ResourceComponent.on_activity_enabled,
    EventType.RESOURCE_AVAILABLE: ResourceComponent.on_resource_available,
}

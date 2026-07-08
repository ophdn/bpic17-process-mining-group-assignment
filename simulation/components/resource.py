"""
resource.py — Resource Component (Sections 1.6, 1.7, 1.8)
=========================================================
Implements the **Role-Based Allocation** creation pattern (R-RBA,
Pattern 2 of Russell et al. 2005) as the resource-allocation heuristic:

  - Section 1.7 Basic (R-RBA): a work item may only be allocated to a
    resource whose "role" qualifies it for the task.  In the absence of
    an explicit organisational model for BPIC-17, a resource's role is
    operationalised as the set of activities it has historically been
    observed performing in the log (`RESOURCE_PERMISSIONS`).  This
    yields the inverse candidate map `_ACTIVITY_TO_RESOURCES`.  R-RBA
    defers the actual choice of *which* qualified resource to runtime;
    among the qualified-and-available candidates a uniform random pick
    is made (the project's default selection behaviour).
  - Section 1.8 (Distribution on Enablement, R-DE — Pattern 19): when
    every qualified resource is busy, the work item is deferred onto a
    FIFO ``_waiting`` queue and (re)allocated the instant a qualified
    resource frees up.
  - Section 1.6 Basic: simple availability model — each resource has a
    fixed capacity (max parallel tasks); tasks queue if all busy.

Reference
---------
Nick Russell, Wil M.P. van der Aalst, Arthur H.M. ter Hofstede, and
David Edmond.  *Workflow Resource Patterns: Identification,
Representation and Tool Support.*  In O. Pastor and J. Falcão e Cunha
(Eds.): CAiSE 2005, LNCS 3520, pp. 216–232, Springer, 2005.
(PDF: docs/papers/optimization_1.1/base resource allocation heuristics.pdf)

----------------------------------------------------------------------------
Design decisions for *uncertain resource availabilities*
----------------------------------------------------------------------------
A resource's future availability is **not known in advance** in this
engine, because service times are stochastic (fitted distributions or a
probabilistic quantile ML model).  The classic Russell R-RBA pattern was
written for workflow systems with deterministic queues; to adapt it to
this uncertain-availability setting we make the following decisions:

1. **Allocate against live load, never against a prediction.**
   Allocation reads the current `_busy` counter at the precise moment
   `ACTIVITY_START` fires, rather than relying on a queue-length
   forecast.  Under stochastic service times the live `_busy` is the
   only honest signal of who is actually free.

2. **Capacity > 1 generalises the paper's single-task resource.**
   `capacity_per_resource` parallel slots are allowed.  A candidate is
   considered "available" when `_busy < capacity`, i.e. it has at least
   one free slot, not only when it is completely idle.

3. **Saturation path = Distribution on Enablement (R-DE).**
   When all qualified candidates are busy, the work item is pushed onto
   a `_waiting` queue rather than rejected.  On every
   `RESOURCE_AVAILABLE` event the just-freed resource is offered to the
   *first* (FIFO) waiting item it is qualified to perform — the resource
   that just became idle is reused for backlog (O(1) per release).  This
   matches R-DE's "distribute as soon as a resource is available".

4. **Idempotent allocation on the deferred path.**
   A deferred event is pre-bound to its resource before being
   re-scheduled, so when it re-enters `on_activity_start` it must not
   be re-allocated (that would double-count the busy slot and
   permanently saturate the pool).  An explicit idempotency guard
   (`event.resource is not None`) prevents this leak.

5. **Deterministic pick under the fixed seed.**
   The uniform random pick uses a seeded `random.Random`, so R-RBA is
   fully reproducible given `RANDOM_SEED=42` — required by the
   assignment's grading.

6. **No a-priori deadline / urgency ordering.**
   R-RBA is workload/permission-only; more advanced timing variants
   (R-ED Early / R-LD Late Distribution) and push *selection* patterns
   (R-RMA random, R-RRA round-robin, R-SHQ shortest-queue) and detour
   patterns (escalation, delegation) are deliberately out of scope and
   left as the upgrade path.  Deferred items are served strictly FIFO.

The resource_activity_map is derived directly from the BPIC-17 log.
Only the top-20 resources (by event count) are included for performance;
extend RESOURCE_POOL with additional users as needed.

Upgrade path:
  - Section 1.6 Advanced: calendar-based availability (shift patterns)
  - Section 1.7 Advanced: role-discovery (e.g. OrdinoR)
  - Section 1.8 Advanced: push selection patterns (R-RMA/R-RRA/R-SHQ)
    and detour patterns (R-D/R-E/R-SD/R-PR/R-UR)
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..core.events import SimEvent, EventType


# ── Resource permission map (from BPIC-17 resource_activity_map) ─────────────
# resource -> set of activities it is allowed to perform (R-RBA "role")
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

# Precompute inverse map: activity -> list of permitted resources (R-RBA)
# Order is deterministic (insertion order of the dict above) so the random
# pick has a stable candidate ordering per activity for reproducibility.
_ACTIVITY_TO_RESOURCES: Dict[str, List[str]] = defaultdict(list)
for _res, _acts in RESOURCE_PERMISSIONS.items():
    for _act in _acts:
        _ACTIVITY_TO_RESOURCES[_act].append(_res)


class ResourceComponent:
    """
    Assigns resources to activities and manages basic availability.

    Section 1.7 Basic (R-RBA): only resources that have historically
    performed an activity are candidates (from BPIC-17 resource_activity_map).
    Among the available qualified candidates a uniform random pick is made
    — the runtime decision R-RBA defers to.

    Section 1.8 (R-DE): if all qualified resources are busy, the task is
    queued and retried when a resource becomes free (FIFO on the freeing
    resource).

    Section 1.6 Basic: each resource has a capacity (default: 1 parallel task).
    """

    HANDLES = {
        EventType.ACTIVITY_START:     None,
        EventType.RESOURCE_AVAILABLE: None,
    }

    def __init__(self, capacity_per_resource: int = 1, seed: Optional[int] = 42):
        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)

        # resource -> current number of active tasks
        self._busy: Dict[str, int] = {r: 0 for r in RESOURCE_PERMISSIONS}

        # Queue of (engine, event) waiting for a free resource (R-DE fallback)
        self._waiting: List[tuple] = []

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        """Intercept ACTIVITY_START to assign a resource before it runs."""
        if event.activity in ("__PROCESS_START__",):
            return  # sentinel — no resource needed

        # Idempotency guard: if a resource is already bound (i.e. the
        # event was deferred and pre-allocated by on_resource_available),
        # do NOT re-allocate — that would double-count the busy slot and
        # leak capacity until the pool permanently saturates.
        if event.resource is not None:
            return

        resource = self._allocate(event.activity)
        if resource:
            event.resource = resource
            self._busy[resource] = self._busy.get(resource, 0) + 1
        else:
            # All permitted resources busy → queue and wait (R-DE)
            self._waiting.append((engine, event))

    def on_resource_available(self, engine, event: SimEvent) -> None:
        """When a resource frees up, try to unblock a waiting task.

        Deferred-allocation policy (R-DE): the just-freed resource is
        offered to the *first* (FIFO) waiting work item it is permitted
        to perform.  This localises the decision to the freeing resource
        (O(1) per release) and aligns with R-DE's "distribute as soon as
        a resource is available".
        """
        resource = event.resource
        self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)

        # Only re-busy this resource if it actually has spare capacity.
        if self._busy[resource] >= self._capacity:
            return

        permitted = RESOURCE_PERMISSIONS.get(resource, set())
        for i, (eng, waiting_event) in enumerate(self._waiting):
            if waiting_event.activity in permitted:
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
        R-RBA runtime pick: uniformly random among available qualified
        resources for this activity.  Returns None if all are busy.
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
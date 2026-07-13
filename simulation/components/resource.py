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

1. **A work item is requested, then started — never both at once.**
   `ACTIVITY_REQUEST` means "this work item is enabled"; `ACTIVITY_START`
   means "a resource is holding it and work has begun".  The process
   component only ever emits the former; this component is the *only*
   thing that emits the latter, and only once allocation succeeded.

   This split is load-bearing.  The engine dispatches every event to all
   registered handlers unconditionally — a handler cannot veto or consume
   an event.  So if a queued work item were still an `ACTIVITY_START`, the
   ProcessComponent would execute it anyway while it sat in the queue, and
   again when the queue re-scheduled it, forking the case into two chains.
   Requests are invisible to everything except this component, so a queued
   item is genuinely stalled: it cannot run, and it cannot be logged.

   It also means `org:resource` is populated by construction (the resource
   is bound before the event the logger sees is ever created), and that
   waiting time falls out for free as `start_time - request_time`.

2. **Allocate against live load, never against a prediction.**
   Allocation reads the current `_busy` counter at the moment the request
   is served, rather than a queue-length forecast.  Under stochastic
   service times the live `_busy` is the only honest signal of who is free.

3. **Capacity > 1 generalises the paper's single-task resource.**
   `capacity_per_resource` parallel slots are allowed.  A candidate is
   considered "available" when `_busy < capacity`, i.e. it has at least
   one free slot, not only when it is completely idle.

4. **Saturation path = Distribution on Enablement (R-DE).**
   When all qualified candidates are busy, the request is pushed onto a
   FIFO `_waiting` queue rather than rejected.  On every
   `RESOURCE_AVAILABLE` event the just-freed resource is offered to the
   *first* waiting item it is qualified to perform — the resource that
   just became idle is reused for backlog (O(1) per release).  This
   matches R-DE's "distribute as soon as a resource is available".

5. **An activity nobody may perform runs unassigned, rather than stalling.**
   If `_ACTIVITY_TO_RESOURCES` has no entry for an activity, no release
   will ever unblock it, so queueing would strand the case forever.  That
   is a gap in the *permission model*, not congestion, so the work item
   runs with `resource=None` and is counted in `stats()`.

6. **Deterministic pick under the fixed seed.**
   The uniform random pick uses a seeded `random.Random`, so R-RBA is
   fully reproducible given `RANDOM_SEED=42` — required by the
   assignment's grading.

7. **No a-priori deadline / urgency ordering.**
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
        EventType.ACTIVITY_REQUEST:   None,
        EventType.RESOURCE_AVAILABLE: None,
    }

    def __init__(self, capacity_per_resource: int = 1, seed: Optional[int] = 42):
        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)

        # resource -> current number of active tasks
        self._busy: Dict[str, int] = {r: 0 for r in RESOURCE_PERMISSIONS}

        # Work items enabled but not yet started, awaiting a resource (R-DE).
        # FIFO. Holds ACTIVITY_REQUEST events.
        self._waiting: List[SimEvent] = []

        # Requests whose activity no resource is permitted to perform. These
        # are a gap in the permission model, not congestion — surfaced in stats
        # rather than silently queued forever (which would deadlock the case).
        self._unpermitted: int = 0

        # Waiting time accumulated between ACTIVITY_REQUEST and ACTIVITY_START.
        self._wait_total: float = 0.0
        self._wait_count: int = 0

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_request(self, engine, event: SimEvent) -> None:
        """A work item is enabled. Start it now if a qualified resource has a
        free slot; otherwise queue it until one frees up (R-DE)."""
        resource = self._allocate(event.activity)

        if resource is not None:
            self._begin(engine, event, resource)
            return

        if not _ACTIVITY_TO_RESOURCES.get(event.activity):
            # Nobody is permitted to do this at all. Queuing would strand the
            # case forever, so run it unassigned and count it.
            self._unpermitted += 1
            self._begin(engine, event, None)
            return

        self._waiting.append(event)

    def on_resource_available(self, engine, event: SimEvent) -> None:
        """A resource finished a task. Offer it to the oldest waiting work item
        it is qualified to perform (R-DE: distribute as soon as available)."""
        resource = event.resource
        self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)

        if self._busy[resource] >= self._capacity:
            return

        permitted = RESOURCE_PERMISSIONS.get(resource, set())
        for i, waiting in enumerate(self._waiting):
            if waiting.activity in permitted:
                self._waiting.pop(i)
                self._begin(engine, waiting, resource)
                return

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _begin(self, engine, request: SimEvent, resource: Optional[str]) -> None:
        """Turn a granted request into an ACTIVITY_START bound to *resource*.

        This is the only place an ACTIVITY_START is created, which is what keeps
        a queued work item from also being executed by the ProcessComponent.
        """
        if resource is not None:
            self._busy[resource] = self._busy.get(resource, 0) + 1

        waited = engine.now - request.timestamp
        self._wait_total += waited
        self._wait_count += 1

        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_START,
            case_id=request.case_id,
            activity=request.activity,
            resource=resource,
            payload=request.payload,
        ))

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

    # ------------------------------------------------------------------
    # Introspection (for empirical evaluation)
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        """Queueing behaviour of the resource pool over the run."""
        mean_wait = self._wait_total / self._wait_count if self._wait_count else 0.0
        return {
            "mean_wait_seconds": mean_wait,
            "work_items_started": self._wait_count,
            "still_queued_at_end": len(self._waiting),
            "unpermitted_activities": self._unpermitted,
        }

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
    EventType.ACTIVITY_REQUEST:   ResourceComponent.on_activity_request,
    EventType.RESOURCE_AVAILABLE: ResourceComponent.on_resource_available,
}
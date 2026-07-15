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

Piled Execution (R-PE, auto-start Pattern 38) — optional
---------------------------------------------------------
When ``piled=True`` is passed to the constructor, the deferred-allocation
drain (R-DE) is refined: on each RESOURCE_AVAILABLE the just-freed
resource first tries to grab a WAITING work item of the *same* activity
type it just finished (its "pile"), before falling back to the usual FIFO
first-compatible scan.  Only ONE task is handed off per release (strictly
sequential, matching the paper's wording).  This batches similar work
onto one worker without affecting the synchronous R-RBA pick or the
processing-time distributions.  Default ``piled=False`` preserves the
existing R-RBA-only behaviour bit-for-bit.  Enabled at runtime via
``--piled-execution`` on main.py.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..core.events import SimEvent, EventType
from ..policies import AllocationPolicy, AllocationState, RandomPolicy


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
    Among the available qualified candidates, ``policy.select()`` picks one
    (default: uniform random — see simulation/policies.py for the push
    selection pattern seam Part II policies plug into).

    Section 1.8 (R-DE): if all qualified resources are busy, the task is
    queued and retried when a resource becomes free (FIFO on the freeing
    resource).

    Section 1.6 Basic: each resource has a capacity (default: 1 parallel task).

    Piled Execution: pass ``piled=True`` to bias the deferred drain toward
    same-activity batching (see module docstring).
    """

    HANDLES = {
        EventType.ACTIVITY_REQUEST:   None,
        EventType.RESOURCE_AVAILABLE: None,
    }

    def __init__(
        self,
        capacity_per_resource: int = 1,
        seed: Optional[int] = 42,
        calendar=None,
        start_datetime=None,
        piled: bool = False,
        policy: Optional[AllocationPolicy] = None,
        excluded_resources: Optional[Set[str]] = None,
    ):
        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)
        self._piled = piled

        # Instance-level resource removal (scenario experiments: "outage",
        # the management "fire two employees" question). Excluded resources
        # are simply never candidates for a new allocation — the permission
        # map and shift calendar are untouched, so this composes cleanly
        # with everything else instead of mutating global module state.
        self._excluded: Set[str] = set(excluded_resources or ())

        # Section 1.8 Advanced: push selection pattern (see
        # simulation/policies.py). Default reuses this component's own RNG
        # so the draw sequence — and therefore every historical event log —
        # is unchanged: RandomPolicy(rng=self._rng) is the pre-existing
        # self._rng.choice(available) call, just behind a seam.
        self._policy: AllocationPolicy = policy or RandomPolicy(rng=self._rng)

        # Section 1.6: an availability calendar (analysis.availability.
        # YearlyAvailability, or None to leave resources always on duty).
        # `start_datetime` anchors simulation seconds to wall-clock time.
        self._calendar = calendar
        self._start = start_datetime

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

        # Sim time of the shift-open wake-up already on the queue, if any (so we
        # schedule at most one).
        self._wake_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_request(self, engine, event: SimEvent) -> None:
        """A work item is enabled. Start it now if a qualified resource is free
        *and on shift*; otherwise queue it (R-DE)."""
        resource = self._allocate(engine, event.activity)

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
        self._arm_shift_wake(engine)

    def on_resource_available(self, engine, event: SimEvent) -> None:
        """Something freed up capacity. Two callers:

        - a task completed (``event.resource`` is set) — that resource is freed
          and offered to the oldest waiting item it is qualified for;
        - a shift opened (``event.resource is None``) — resources came on duty
          without anything completing, so the whole queue is re-examined. Without
          this, work items queued while every qualified resource was off-shift
          would never be woken: no completion is pending to wake them.

        Piled Execution (R-PE, Pattern 38): when ``self._piled`` is True and the
        event carries the just-finished activity in ``event.payload``, the freed
        resource first looks for a waiting item of the SAME activity type before
        falling back to the FIFO first-compatible scan. One handoff per release
        (sequential, per the paper).
        """
        resource = event.resource

        if resource is not None:
            self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)
            if (self._busy[resource] < self._capacity
                    and self._is_on_shift(engine, resource)):
                # Piled Execution: prefer a waiting item of the same activity
                # type this resource just finished. It is permitted by
                # construction (it just ran one), so no permission check needed.
                if self._piled and event.payload is not None:
                    for i, waiting in enumerate(self._waiting):
                        if waiting.activity == event.payload:
                            self._waiting.pop(i)
                            self._begin(engine, waiting, resource)
                            self._arm_shift_wake(engine)
                            return

                permitted = RESOURCE_PERMISSIONS.get(resource, set())
                for i, waiting in enumerate(self._waiting):
                    if waiting.activity in permitted:
                        self._waiting.pop(i)
                        self._begin(engine, waiting, resource)
                        break
        else:
            self._wake_at = None
            self._drain(engine)

        self._arm_shift_wake(engine)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _drain(self, engine) -> None:
        """Start as many queued work items as the on-duty pool can take."""
        still: List[SimEvent] = []
        for req in self._waiting:
            r = self._allocate(engine, req.activity)
            if r is not None:
                self._begin(engine, req, r)
            else:
                still.append(req)
        self._waiting = still

    # -- availability (Section 1.6) --------------------------------------

    def _now_wall(self, engine):
        """Simulation clock as wall-clock time (the calendar is in local time)."""
        from datetime import timedelta
        return self._start + timedelta(seconds=engine.now)

    def _is_on_shift(self, engine, resource: str) -> bool:
        """Is *resource* on duty right now?

        Without a calendar, everyone is always on duty (the pre-1.6 behaviour).

        A resource the calendar does not know is also always on duty. That is not
        a fallback but the correct answer: the calendar only models *human*
        staff, so an account missing from it is an automated one (User_1), and a
        batch process does not keep office hours.
        """
        if self._calendar is None:
            return True
        if resource not in self._calendar.weekly.windows:
            return True
        return self._calendar.is_available(resource, self._now_wall(engine))

    def _arm_shift_wake(self, engine) -> None:
        """Ensure a wake-up is queued for the next time a shift opens.

        Only needed while work is waiting. At most one is outstanding.
        """
        if self._calendar is None or not self._waiting or self._wake_at is not None:
            return

        t = self._next_shift_open(engine)
        if t is None:
            return

        self._wake_at = t
        engine.schedule(SimEvent(
            timestamp=t,
            priority=1,            # open the shift before anything else at t
            event_type=EventType.RESOURCE_AVAILABLE,
            resource=None,         # None => "a shift opened", not "a task ended"
        ))

    def _next_shift_open(self, engine, horizon_days: int = 14) -> Optional[float]:
        """Sim time at which some resource's window next opens, or None.

        Windows are per weekday, so the candidate times are the window starts of
        each upcoming day. We scan forward day by day and take the earliest that
        is strictly in the future.
        """
        from datetime import datetime, time, timedelta

        now_wall = self._now_wall(engine)
        best: Optional[float] = None

        for day in range(horizon_days + 1):
            d = (now_wall + timedelta(days=day)).date()
            if d in self._calendar.holidays:
                continue

            midnight = datetime.combine(d, time.min)
            dow = d.weekday()

            for res, windows in self._calendar.weekly.windows.items():
                w = windows.get(dow)
                if w is None or d in self._calendar.vacations.get(res, ()):
                    continue
                t = (midnight + timedelta(hours=w[0]) - self._start).total_seconds()
                if t > engine.now and (best is None or t < best):
                    best = t

            if best is not None:
                return best
        return best

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

    def _allocate(self, engine, activity: str) -> Optional[str]:
        """
        R-RBA runtime filter (role permission + live capacity + on-shift),
        then the push *selection* pattern in ``self._policy`` picks one.
        Returns None if nobody qualifies right now — the caller queues the
        work item. A policy never sees a candidate this filter rejected, so
        it cannot violate R-RBA or the shift calendar even by accident.

        Note a task already under way is never preempted when its resource goes
        off shift: the calendar gates *allocation*, not execution. A person who
        starts a task at 16:55 finishes it rather than dropping it at 17:00.
        """
        candidates = _ACTIVITY_TO_RESOURCES.get(activity, [])
        available = [
            r for r in candidates
            if self._busy.get(r, 0) < self._capacity
            and self._is_on_shift(engine, r)
            and r not in self._excluded
        ]
        if not available:
            return None
        state = AllocationState(busy=self._busy, capacity=self._capacity)
        return self._policy.select(activity, available, state)

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

    def release(self, engine, resource: str, activity: Optional[str] = None) -> None:
        """
        Call this from the ProcessComponent after ACTIVITY_COMPLETE
        to free the resource. Schedules a RESOURCE_AVAILABLE event.

        Parameters
        ----------
        activity : str, optional
            The activity just completed by ``resource``.  Stashed in
            the event's ``payload`` so that ``on_resource_available``
            can implement Piled Execution (same-activity batching).
            Default ``None`` preserves the existing behaviour for
            callers that don't pass it.
        """
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=1,   # High priority: free up before next tasks start
            event_type=EventType.RESOURCE_AVAILABLE,
            resource=resource,
            payload=activity,
        ))


ResourceComponent.HANDLES = {
    EventType.ACTIVITY_REQUEST:   ResourceComponent.on_activity_request,
    EventType.RESOURCE_AVAILABLE: ResourceComponent.on_resource_available,
}
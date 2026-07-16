"""
resource.py — Resource Component (Sections 1.6, 1.7, 1.8)
=========================================================
Implements the **Role-Based Allocation** creation pattern (R-RBA,
Pattern 2 of Russell et al. 2005) as the resource-allocation heuristic:

  - Section 1.7 (R-RBA): a work item may only be allocated to a resource
    whose "role" qualifies it for the task.  *Which* resources qualify is
    decided entirely by an injected permission model (see
    `components/permissions.py`) — this component never inspects a
    permission map directly, so the model can be swapped without editing
    the engine, the process component, or the event types.

    The default is `DEFAULT_PERMISSIONS`: the hardcoded top-20 map below,
    i.e. a resource's role is the set of activities it was observed
    performing.  Section 1.7 Basic replaces it with the same rule learned
    over all 149 resources; Section 1.7 Advanced replaces it with an
    organizational model discovered by OrdinoR, whose capabilities may
    also depend on the case type and the time of day.

    R-RBA defers the choice of *which* qualified resource to runtime;
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

k-Batching (Zeng & Zhao) — optional, mutually exclusive with Piled Execution
-----------------------------------------------------------------------------
When ``batching_k`` is set (an int >= 1), the allocation discipline changes
fundamentally: work items are NEVER allocated immediately on request.  They
always queue, and are released in batches of *k* solved as a parallel-
machines assignment problem (``scipy.optimize.linear_sum_assignment``)
against whichever qualified+free+on-shift resources exist at flush time,
minimising total expected processing time
(``simulation/expected_duration.py``).  A flush triggers when either
(a) >= k items are waiting, or (b) the oldest waiting item has waited past
``batching_max_wait_seconds`` (a safety valve against starving a thin
queue -- the known idle-time weakness of k-Batching under low load,
Zeng & Zhao, lecture 06 slide 12).  Items an assignment round can't match
to a free resource stay queued for the next flush.  This deliberately does
NOT reduce to the synchronous R-RBA pick at k=1 in the same way Piled
Execution reduces to plain R-DE at piled=False -- even k=1 always makes
the requester wait for a flush trigger rather than starting immediately
when a resource happens to be free, which is the whole point of the
lecture's k-Batching formulation (batch allocation, not on-arrival
allocation).  Enabled at runtime via ``--k-batching K`` on main.py.
"""

import random
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.events import SimEvent, EventType
from .permissions import PermissionModel, StaticPermissions
from ..policies import AllocationPolicy, AllocationState, RandomPolicy
from ..expected_duration import ExpectedDurationModel


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
# The default permission model: the hardcoded map above, wrapped in the
# Section 1.7 interface. Insertion order is stable, so the seeded random pick
# below stays reproducible. Pass `permissions=` to swap in a learned model
# (the observed matrix, or an OrdinoR organizational model) without touching
# this component — see simulation/components/permissions.py.
DEFAULT_PERMISSIONS = StaticPermissions(RESOURCE_PERMISSIONS)

# payload tags on synthetic resource=None RESOURCE_AVAILABLE wake-up events,
# so on_resource_available's k-batching branch resets only the specific
# wake mechanism that actually fired (see the comment there for why
# resetting both indiscriminately is a real bug, not just untidy).
_SHIFT_WAKE = "__shift_wake__"
_BATCH_WAKE = "__batch_wake__"


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
        permissions: Optional[PermissionModel] = None,
        piled: bool = False,
        policy: Optional[AllocationPolicy] = None,
        excluded_resources: Optional[Set[str]] = None,
        batching_k: Optional[int] = None,
        batching_max_wait_seconds: float = 4 * 3600,
        duration_model_path: Optional[str] = None,
    ):
        if batching_k is not None and piled:
            raise ValueError(
                "batching_k and piled=True are mutually exclusive — they "
                "define different (incompatible) deferred-allocation "
                "disciplines. Pick one."
            )
        if batching_k is not None and batching_k < 1:
            raise ValueError(f"batching_k must be >= 1, got {batching_k!r}")

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

        # k-Batching (Zeng & Zhao) — see module docstring.
        self._batching_k = batching_k
        self._batching_max_wait = batching_max_wait_seconds
        self._duration_model: Optional[ExpectedDurationModel] = (
            ExpectedDurationModel(duration_model_path) if batching_k is not None else None
        )

        # Section 1.7: who may perform what. Any object satisfying the
        # PermissionModel protocol; defaults to the hardcoded observed map.
        self._permissions: PermissionModel = permissions or DEFAULT_PERMISSIONS

        # Section 1.6: an availability calendar (analysis.availability.
        # YearlyAvailability, or None to leave resources always on duty).
        # `start_datetime` anchors simulation seconds to wall-clock time.
        self._calendar = calendar
        self._start = start_datetime

        # resource -> current number of active tasks
        self._busy: Dict[str, int] = {
            r: 0 for r in self._permissions.resources()
        }

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

        # k-Batching's max-wait valve: sim time of the wake-up already queued
        # for when the oldest waiting item's max-wait elapses, if any. Without
        # this, the valve would only fire as a side effect of some unrelated
        # event happening to be dispatched after the threshold — fine under
        # BPIC-17's real arrival rate (an event is never far away), but not
        # robust for a genuinely idle lull (e.g. very low --scale-factor
        # experiments), so it gets its own self-scheduled wake-up like the
        # shift-wake mechanism above.
        self._batch_wake_at: Optional[float] = None

    @property
    def permissions(self) -> PermissionModel:
        """The permission model in force (Section 1.7)."""
        return self._permissions

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_activity_request(self, engine, event: SimEvent) -> None:
        """A work item is enabled. Start it now if a qualified resource is free
        *and on shift*; otherwise queue it (R-DE).

        k-Batching (see module docstring): if ``self._batching_k`` is set,
        this discipline is replaced entirely — a request NEVER allocates
        immediately, it always queues and waits for a batch flush.
        """
        if self._batching_k is not None:
            if not self._qualified(engine, event):
                self._unpermitted += 1
                self._begin(engine, event, None)
                return
            self._waiting.append(event)
            self._maybe_flush_batch(engine)
            self._arm_shift_wake(engine)
            self._arm_batch_wake(engine)
            return

        resource = self._allocate(engine, event)

        if resource is not None:
            self._begin(engine, event, resource)
            return

        if not self._qualified(engine, event):
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

        k-Batching: replaces this whole method's logic — freeing a resource
        (or a shift opening) just means "maybe there's now enough
        free/on-shift capacity to flush a batch", checked once here rather
        than per-release handoff logic.
        """
        if self._batching_k is not None:
            if event.resource is not None:
                self._busy[event.resource] = max(0, self._busy.get(event.resource, 0) - 1)
            elif event.payload == _SHIFT_WAKE:
                self._wake_at = None
            elif event.payload == _BATCH_WAKE:
                self._batch_wake_at = None
            # else: some other resource=None event -- nothing to reset.
            #
            # Tagging matters: resetting BOTH flags whenever *either* wake
            # fires (the naive version) re-arms whichever one is still
            # legitimately pending on its own already-scheduled event,
            # duplicating it. Duplicates compound every time the other wake
            # fires too, producing thousands of same-target RESOURCE_
            # AVAILABLE events piled at one instant (found the hard way via
            # cProfile -- _next_shift_open dominated runtime with the
            # non-batching cost model reused).
            self._maybe_flush_batch(engine)
            self._arm_shift_wake(engine)
            self._arm_batch_wake(engine)
            return

        resource = event.resource

        if resource is not None:
            self._busy[resource] = max(0, self._busy.get(resource, 0) - 1)
            if (self._busy[resource] < self._capacity
                    and self._is_on_shift(engine, resource)):
                # Piled Execution: prefer a waiting item of the same activity
                # type this resource just finished — its "pile".
                #
                # The permission check here is NOT redundant, though it was when
                # this was written against a flat resource->activities map (where
                # having just run the activity did imply permission for it). An
                # OrdinoR permission model also conditions on the case type and
                # the weekday, so a resource that just handled "W_Validate
                # application" for a *car* loan is not thereby permitted the same
                # activity for a *boat* loan. Same activity, different execution
                # context, possibly different answer.
                if self._piled and event.payload is not None:
                    for i, waiting in enumerate(self._waiting):
                        if waiting.activity != event.payload:
                            continue
                        ct, when = self._context(engine, waiting)
                        if not self._permissions.permits(
                                resource, waiting.activity, case_type=ct, when=when):
                            continue
                        self._waiting.pop(i)
                        self._begin(engine, waiting, resource)
                        self._arm_shift_wake(engine)
                        return

                for i, waiting in enumerate(self._waiting):
                    ct, when = self._context(engine, waiting)
                    if self._permissions.permits(
                            resource, waiting.activity, case_type=ct, when=when):
                        self._waiting.pop(i)
                        self._begin(engine, waiting, resource)
                        break
        else:
            self._wake_at = None
            self._drain(engine)

        self._arm_shift_wake(engine)

    # ------------------------------------------------------------------
    # k-Batching (Zeng & Zhao) — see module docstring
    # ------------------------------------------------------------------

    def _free_resources(self, engine) -> List[str]:
        """Every resource with a free slot, on shift, not excluded — the
        candidate pool a batch flush assigns against. Shared with
        ``_allocate``'s filter, minus the per-activity permission check
        (the assignment cost matrix handles permission per pair instead)."""
        return [
            r for r in self._permissions.resources()
            if self._busy.get(r, 0) < self._capacity
            and self._is_on_shift(engine, r)
            and r not in self._excluded
        ]

    def _maybe_flush_batch(self, engine) -> None:
        """Flush batches while the trigger holds: >= k items waiting, or the
        oldest waiting item has waited past the max-wait safety valve.
        Loops (rather than flushing just one batch) so a burst of requests
        arriving with plenty of free capacity available right now doesn't
        leave a second full batch queued idly until the next unrelated
        event — but stops as soon as a round makes no progress (nothing
        left with a compatible free resource), rather than spinning.
        """
        while self._waiting:
            oldest_wait = engine.now - self._waiting[0].timestamp
            if len(self._waiting) < self._batching_k and oldest_wait < self._batching_max_wait:
                return

            free = self._free_resources(engine)
            if not free:
                return  # nothing to assign to yet; try again on the next trigger

            batch = self._waiting[: self._batching_k]
            before = len(self._waiting)
            self._assign_batch(engine, batch, free)
            if len(self._waiting) == before:
                return  # no progress this round (permission gaps) -- stop

    def _assign_batch(self, engine, batch: List[SimEvent], free: List[str]) -> None:
        """Solve batch (size n) x free-resources (size m) as a parallel-
        machines assignment problem: minimise total expected processing
        time (``simulation/expected_duration.py``), subject to R-RBA
        permission (an unpermitted pair gets a prohibitive cost so
        ``linear_sum_assignment`` never picks it). Unmatched items
        (permission gap, or more batch items than compatible free
        resources) stay in ``self._waiting``.
        """
        n, m = len(batch), len(free)
        PROHIBITIVE = 1e12
        cost = np.full((n, m), PROHIBITIVE)
        for i, req in enumerate(batch):
            # Permission is per (resource, activity, case type, time) under an
            # OrdinoR model, so it is resolved per *pair* here rather than from a
            # flat per-resource activity set.
            ct, when = self._context(engine, req)
            for j, r in enumerate(free):
                if self._permissions.permits(r, req.activity, case_type=ct, when=when):
                    cost[i, j] = self._duration_model.expected_duration(req.activity, r)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_ids = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= PROHIBITIVE:
                continue  # no compatible free resource for this item this round
            req = batch[i]
            assigned_ids.add(id(req))
            self._begin(engine, req, free[j])

        if assigned_ids:
            self._waiting = [w for w in self._waiting if id(w) not in assigned_ids]

    def _drain(self, engine) -> None:
        """Start as many queued work items as the on-duty pool can take."""
        still: List[SimEvent] = []
        for req in self._waiting:
            r = self._allocate(engine, req)
            if r is not None:
                self._begin(engine, req, r)
            else:
                still.append(req)
        self._waiting = still

    # -- permissions (Section 1.7) ---------------------------------------

    def _context(self, engine, event: SimEvent):
        """The execution context of a work item: (case type, wall-clock time).

        An OrdinoR organizational model may grant capabilities per case type and
        time type, not just per activity. The case type rides on the event payload
        (sampled at arrival); the time comes from the clock we already keep for
        the Section 1.6 calendar. A model that ignores these dimensions simply
        does not look at them.
        """
        payload = event.payload or {}
        case_type = payload.get("case_type")
        when = self._now_wall(engine) if self._start is not None else None
        return case_type, when

    def _qualified(self, engine, event: SimEvent) -> bool:
        """Is *anyone at all* permitted this work item, busy or not?

        Distinguishes "everyone is busy or off-shift" (queue and wait) from
        "nobody may ever do this" (a hole in the permission model — queuing would
        strand the case forever).
        """
        ct, when = self._context(engine, event)
        return bool(self._permissions.candidates(
            event.activity, case_type=ct, when=when))

    # -- availability (Section 1.6) --------------------------------------

    def _now_wall(self, engine):
        """Simulation clock as wall-clock time (the calendar is in local time)."""
        from datetime import timedelta
        return self._start + timedelta(seconds=engine.now)

    def _is_on_shift(self, engine, resource: str) -> bool:
        """Is *resource* on duty right now?

        Without a calendar, everyone is always on duty (the pre-1.6 behaviour).

        A resource with no fitted window is resolved by *why* it has none:

          - a known **system account** (in the calendar's ``system`` set) is
            always on duty — an automated account keeps no office hours. This is
            correct, not a fallback.
          - any **other** windowless resource is a human the calendar could not
            fit, and is treated as *off shift* rather than always-on. Treating an
            unfitted human as available 24/7 is the opposite of what "too little
            data to fit a shift" should mean, and would let sparse resources
            silently absorb all out-of-hours work. The Section 1.6 model fits a
            coarse window for every human with any W_ signal (Tier B), so in
            practice only system accounts land here.

        For backward compatibility with a calendar that predates the ``system``
        set, a windowless resource is treated as always-on (the old behaviour).
        """
        if self._calendar is None:
            return True
        if resource not in self._calendar.weekly.windows:
            system = getattr(self._calendar, "system", None)
            if system is None:
                return True                 # legacy calendar: preserve old behaviour
            return resource in system
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
            payload=_SHIFT_WAKE,
        ))

    def _arm_batch_wake(self, engine) -> None:
        """Ensure a wake-up is queued for when the oldest waiting item's
        k-Batching max-wait elapses, so the safety valve fires even during
        an idle lull with no other event to trigger a recheck. At most one
        outstanding (mirrors ``_arm_shift_wake``'s pattern).

        Guards ``t > engine.now`` strictly: if the valve just fired and
        found no free resource to assign to, ``self._waiting[0]`` hasn't
        changed, so the naive recomputation lands on the exact same instant
        that just fired — scheduling that again is an infinite same-instant
        loop (confirmed the hard way: heapq keeps re-popping a same-
        timestamp event before ever reaching later, real completion events
        that would actually free a resource). When the deadline is already
        due, do nothing here; the next *real* RESOURCE_AVAILABLE release
        already re-checks via _maybe_flush_batch, which is the correct path
        to eventually unstick a capacity-exhausted queue.
        """
        if self._batching_k is None or not self._waiting or self._batch_wake_at is not None:
            return
        t = self._waiting[0].timestamp + self._batching_max_wait
        if t <= engine.now:
            return
        self._batch_wake_at = t
        engine.schedule(SimEvent(
            timestamp=t,
            priority=1,
            event_type=EventType.RESOURCE_AVAILABLE,
            resource=None,
            payload=_BATCH_WAKE,
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

    def _allocate(self, engine, event: SimEvent) -> Optional[str]:
        """
        R-RBA runtime filter (role permission + live capacity + on-shift),
        then the push *selection* pattern in ``self._policy`` picks one.
        Returns None if nobody qualifies right now — the caller queues the
        work item. A policy never sees a candidate this filter rejected, so
        it cannot violate R-RBA or the shift calendar even by accident.

        "Qualified" is whatever the injected permission model says (Section 1.7):
        the observed resource-activity matrix, or an OrdinoR organizational model
        that may also condition on the case type and the time.

        Note a task already under way is never preempted when its resource goes
        off shift: the calendar gates *allocation*, not execution. A person who
        starts a task at 16:55 finishes it rather than dropping it at 17:00.
        """
        ct, when = self._context(engine, event)
        candidates = self._permissions.candidates(
            event.activity, case_type=ct, when=when)

        available = [
            r for r in candidates
            if self._busy.get(r, 0) < self._capacity
            and self._is_on_shift(engine, r)
            and r not in self._excluded
        ]
        if not available:
            return None
        state = AllocationState(busy=self._busy, capacity=self._capacity)
        return self._policy.select(event.activity, available, state)

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
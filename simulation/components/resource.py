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
from ..expected_duration import ExpectedDurationModel, distribution_mean_seconds


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


# ── How many work items may one resource hold at once? ────────────────────────
#
# This is a modelling assumption, not a tuning knob, and it only makes sense
# next to the duration semantics of the lifecycle mode it runs in. The duration
# model has no concurrent-load feature (its eight ML features are activity,
# resource, previous activity, weekday, hour, case position, case age and
# prior-activity count), so N concurrent items each finish exactly as fast as
# one. Capacity therefore multiplies a resource's throughput for free, with no
# context-switching penalty — which is why the number has to be defensible.
#
# active
#   A resource is held only for one hands-on session (median 0.8–2.7 min across
#   the six principal W-activities) and `suspend` releases it back to the pool.
#   Measured on the real log: of 254,370 active sessions across 148 resources,
#   98.4% of busy time is spent in exactly ONE session. Interleaving is already
#   modelled explicitly by suspend/resume, so any capacity > 1 double-counts it.
#   Team decision 2026-07-17 (Johannes): 1, as the only value the log supports.
#
# legacy
#   A resource is held for the entire elapsed start→complete span, which is
#   mostly suspended waiting rather than work. Real elapsed spans overlap at a
#   median peak of 54 per resource (max 150; 134/145 resources exceed 3), so 1
#   would be catastrophically tight — it would pin a person to one application
#   for hours. 3 is the historical value, kept so existing evidence reproduces.
#   It is not defended as correct, only as unchanged; the honest legacy value is
#   much closer to 54. See docs/PLAN_pwork_roster.md §8.4.
DEFAULT_CAPACITY_ACTIVE = 1
DEFAULT_CAPACITY_LEGACY = 3

# Base seed for the p_work roster draw. Rostering is ON by default (team
# decision 2026-07-17): a fitted component of the Section 1.6 model that the
# runtime ignores is not a "safe default", it is a silently wrong one. Without
# it the deployed calendar fields ~123 people on a Monday morning where the
# model we validated expects ~37, so every run is ~3.3x overstaffed and Part II
# has no real contention. Pass roster_seed=None (CLI: --no-roster) to reproduce
# pre-rostering evidence logs. See docs/PLAN_pwork_roster.md.
DEFAULT_ROSTER_SEED = 42


def capacity_for_mode(lifecycle_mode: str) -> int:
    """Default work items per resource for *lifecycle_mode*.

    The same number means opposite things in the two modes, so this must be
    derived from the mode rather than fixed globally.
    """
    return (DEFAULT_CAPACITY_ACTIVE if lifecycle_mode == "active"
            else DEFAULT_CAPACITY_LEGACY)

# payload tags on synthetic resource=None RESOURCE_AVAILABLE wake-up events,
# so on_resource_available's k-batching branch resets only the specific
# wake mechanism that actually fired (see the comment there for why
# resetting both indiscriminately is a real bug, not just untidy).
_SHIFT_WAKE = "__shift_wake__"
_BATCH_WAKE = "__batch_wake__"

# D1 (Park & Song): a predicted successor only joins the allocation epoch as
# a phantom once its case's current activity is expected to finish within
# this window — strategic idling is for imminent work, not multi-day locks.
PARKSONG_LOOKAHEAD_SECONDS = 3600.0


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
        EventType.ACTIVITY_WITHDRAW:  None,
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
        lifecycle_mode: str = "legacy",
        lifecycle_params=None,
        pull: Optional[str] = None,
        parksong: bool = False,
        krm_delta: Optional[float] = None,
        drl: bool = False,
        drl_model_path: Optional[str] = None,
        drl_external_control: bool = False,
        drl_observation_version: Optional[int] = None,
        drl_action_version: Optional[int] = None,
    ):
        modes = {
            "piled": piled, "batching_k": batching_k is not None,
            "pull": pull is not None, "parksong": parksong,
            "krm_delta": krm_delta is not None,
            "drl": drl,
        }
        active_modes = [name for name, on in modes.items() if on]
        if len(active_modes) > 1:
            raise ValueError(
                f"{active_modes} are mutually exclusive — each redefines "
                "the deferred-allocation discipline. Pick one."
            )
        if batching_k is not None and batching_k < 1:
            raise ValueError(f"batching_k must be >= 1, got {batching_k!r}")
        if pull is not None and pull not in ("spt", "laf"):
            raise ValueError(f"pull must be 'spt' or 'laf', got {pull!r}")
        if krm_delta is not None and krm_delta <= 0:
            raise ValueError(f"krm_delta must be > 0, got {krm_delta!r}")

        self._capacity = capacity_per_resource
        self._rng = random.Random(seed)
        self._piled = piled

        # Pull-side selection discipline (Russell et al. pull patterns,
        # simulated): when a resource frees up, IT picks which waiting item
        # to take, by a local preference — instead of the system's FIFO
        # first-permitted scan (push/R-DE). Two rules:
        #   "spt" — the item with the shortest expected duration for THIS
        #           resource (myopic local optimum). Note the honest scope:
        #           without the trained ML duration artifact the expected-
        #           duration model falls back to per-activity distribution
        #           means, which are resource-independent — SPT then reduces
        #           to a shortest-processing-time queue discipline. Still a
        #           genuinely different rule from FIFO, but not personalised.
        #   "laf" — longest-active-first (lecture deck 04, F31): the item
        #           whose CASE has been in the system longest. Distinct from
        #           queue-FIFO: an old case's fresh request outranks a young
        #           case's long-waiting one.
        # Deliberately NOT a FIFO relabel — "freed resource takes the
        # longest-waiting item" would be behaviourally identical to the
        # existing R-DE drain and prove nothing. Applies at task-completion
        # releases; the shift-open drain stays FIFO (a few events per day
        # vs. thousands of releases — documented simplification).
        self._pull = pull

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

        # k-Batching (Zeng & Zhao) — see module docstring. The expected-
        # duration model is shared with SPT-pull and the two advanced
        # policies (all cost on it).
        self._batching_k = batching_k
        self._batching_max_wait = batching_max_wait_seconds
        self._duration_model: Optional[ExpectedDurationModel] = (
            ExpectedDurationModel(
                duration_model_path,
                lifecycle_mode=lifecycle_mode,
                lifecycle_params=lifecycle_params,
            ) if (batching_k is not None or pull == "spt"
                  or parksong or krm_delta is not None) else None
        )

        # LAF-pull bookkeeping: case_id -> sim time of its first request
        # ever seen here ("how long has this case been in the system").
        self._case_first_seen: Dict[str, float] = {}

        # D1 — Park & Song 2019, prediction-based allocation with strategic
        # idling: at every allocation epoch (request or release), solve one
        # assignment over the REAL waiting items plus PHANTOM items — the
        # predicted next activity of every case currently in service
        # (see simulation/policies_advanced.py for the predictor and the
        # documented LSTM->branching-model substitution). A resource the
        # solver matches to a phantom is deliberately left idle this epoch:
        # it is the best fit for work about to arrive (strategic idling).
        # Phantom weight is 1.0 x expected duration — the simplest honest
        # choice; a probability-discounted weight is future work.
        self._parksong = parksong
        self._predictor = None
        if parksong:
            from ..policies_advanced import NextActivityPredictor
            self._predictor = NextActivityPredictor()
        # case_id -> (current activity, its payload) while in service.
        self._active_cases: Dict[str, tuple] = {}

        # D2 — Kunkler & Rinderle-Ma 2024, assignment variant with dummy-
        # resource costs: each waiting item also gets a private dummy column
        # costing krm_delta x its own mean expected duration. An item the
        # solver sends to its dummy stays queued — deferring is chosen over
        # a bad fit whenever every free resource would cost more than delta
        # times the item's baseline, which is the paper's "wait under
        # uncertain availability" outcome expressed as a cost.
        self._krm_delta = krm_delta

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

        # D3 — Middelhuis et al. (2025), masked PPO.  V2 of the fixed action
        # space contains only (resource, activity) pairs that the permission
        # model can ever allow, plus POSTPONE.  Contextual permission, live
        # capacity, calendar and queue state are still enforced by the action
        # mask.  Removing globally impossible pairs makes the classification
        # problem substantially smaller without changing any feasible choice.
        # V1 (the full Cartesian product) remains available for old models.
        self._drl = bool(drl)
        self._drl_external_control = bool(drl_external_control)
        if self._drl_external_control and not self._drl:
            raise ValueError("drl_external_control=True requires drl=True")
        if self._drl and not self._drl_external_control and not drl_model_path:
            raise ValueError(
                "drl=True requires drl_model_path for inference, or "
                "drl_external_control=True for training"
            )
        self._drl_resources = sorted(self._permissions.resources())
        activities_fn = getattr(self._permissions, "activities", None)
        if callable(activities_fn):
            activities = activities_fn()
        else:  # Compatibility with third-party PermissionModel objects.
            activities = sorted({a for acts in RESOURCE_PERMISSIONS.values() for a in acts})
        self._drl_activities = sorted(activities)
        self._drl_resource_index = {r: i for i, r in enumerate(self._drl_resources)}
        self._drl_activity_index = {a: i for i, a in enumerate(self._drl_activities)}
        self._drl_compact_action_pairs = [
            (resource, activity)
            for resource in self._drl_resources
            for activity in self._drl_activities
            if self._permissions.permits(resource, activity)
        ]
        self._drl_compact_action_index = {
            pair: i for i, pair in enumerate(self._drl_compact_action_pairs)
        }
        self._drl_decision_pending = False
        self._drl_model = None
        self._drl_assignments = 0
        self._drl_postponements = 0
        if drl_action_version not in (None, 1, 2):
            raise ValueError("drl_action_version must be 1, 2, or None")
        self._drl_action_version = (
            int(drl_action_version)
            if drl_action_version is not None
            else (2 if self._drl_external_control else 0)
        )
        if drl_observation_version not in (None, 1, 2, 3):
            raise ValueError(
                "drl_observation_version must be 1, 2, 3, or None")
        # V2 adds the currently executing activity.  V3 adds per-activity case
        # age and expected duration, two signals that are central to cycle-time
        # allocation but absent from the preliminary models.
        self._drl_observation_version = (
            int(drl_observation_version)
            if drl_observation_version is not None
            else (3 if self._drl_external_control else 0)
        )
        processing_times = (
            lifecycle_params.processing_times
            if lifecycle_mode == "active" and lifecycle_params is not None
            else None
        )
        self._drl_expected_duration = {
            activity: distribution_mean_seconds(activity, processing_times)
            for activity in self._drl_activities
        }
        self._drl_active_activities: Dict[str, List[str]] = {
            resource: [] for resource in self._drl_resources
        }
        if self._drl and not self._drl_external_control:
            from ..drl import load_maskable_ppo
            self._drl_model = load_maskable_ppo(drl_model_path)
            actual_actions = int(self._drl_model.action_space.n)
            if self._drl_action_version == 0:
                matching_actions = [
                    version for version in (1, 2)
                    if actual_actions == self._drl_action_count_for(version)
                ]
                if not matching_actions:
                    raise ValueError(
                        f"DRL model action space has {actual_actions} actions, "
                        f"expected V1 ({self._drl_action_count_for(1)}) or V2 "
                        f"({self._drl_action_count_for(2)})"
                    )
                # If every pair is permitted, V1 and V2 have the same ordering.
                self._drl_action_version = max(matching_actions)
            expected_actions = self._drl_action_count_for(
                self._drl_action_version)
            if actual_actions != expected_actions:
                raise ValueError(
                    f"DRL model action space has {actual_actions} "
                    f"actions, but this permission model requires {expected_actions}"
                )
            actual_obs = tuple(self._drl_model.observation_space.shape)
            if self._drl_observation_version == 0:
                matching = [
                    version for version in (1, 2, 3)
                    if actual_obs == (self._drl_observation_size_for(version),)
                ]
                if not matching:
                    raise ValueError(
                        f"DRL model observation shape is {actual_obs}, expected "
                        f"V1 ({self._drl_observation_size_for(1)},) or V2 "
                        f"({self._drl_observation_size_for(2)},) or V3 "
                        f"({self._drl_observation_size_for(3)},)"
                    )
                self._drl_observation_version = matching[0]
            expected_obs = self.drl_observation_size
            if actual_obs != (expected_obs,):
                raise ValueError(
                    f"DRL model observation shape is {actual_obs}, expected "
                    f"({expected_obs},)"
                )

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
        self._arm_withdraw(engine, event)
        token = self._queue_token(event)
        if token is not None and token.get("state") != "queued":
            return

        if (self._pull == "laf" or self._drl) and event.case_id is not None:
            self._case_first_seen.setdefault(event.case_id, event.timestamp)

        if self._drl:
            if not self._qualified(engine, event):
                self._unpermitted += 1
                self._begin(engine, event, None)
                return
            self._waiting.append(event)
            self._drive_drl(engine)
            self._arm_shift_wake(engine)
            return

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

        if self._parksong or self._krm_delta is not None:
            if not self._qualified(engine, event):
                self._unpermitted += 1
                self._begin(engine, event, None)
                return
            self._waiting.append(event)
            self._epoch_flush(engine)
            self._arm_shift_wake(engine)
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

    def on_activity_withdraw(self, engine, event: SimEvent) -> None:
        """Resolve an active-mode withdrawal timer against its cancellable
        queue token.  A timer that lost the race to allocation is a no-op; a
        queued item is removed exactly once before ProcessComponent routes it.
        """
        token = self._queue_token(event)
        if token is None or token.get("state") != "queued":
            return

        wid = token.get("work_item_id")
        for i, waiting in enumerate(self._waiting):
            waiting_token = self._queue_token(waiting)
            if waiting_token is token or (
                waiting_token is not None
                and waiting_token.get("work_item_id") == wid
            ):
                self._waiting.pop(i)
                token["state"] = "withdrawn"
                if self._drl:
                    self._drive_drl(engine)
                self._arm_shift_wake(engine)
                self._arm_batch_wake(engine)
                return

        # The request is no longer queued (normally because allocation won).
        token["state"] = "allocated"

    @staticmethod
    def _queue_token(event: SimEvent) -> Optional[dict]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        token = payload.get("_queue_token")
        return token if isinstance(token, dict) else None

    def _arm_withdraw(self, engine, event: SimEvent) -> None:
        """Attach one mutable cancellation token and schedule the competing
        withdrawal timer for an initial active W_ request.

        Resume-ready requests deliberately carry no ``_withdraw_delay`` and
        therefore cannot draw a second withdrawal hazard.
        """
        payload = event.payload if isinstance(event.payload, dict) else None
        if not payload or payload.get("resuming") or self._queue_token(event) is not None:
            return
        delay = payload.get("_withdraw_delay")
        wid = payload.get("work_item_id")
        if delay is None or not wid or not (event.activity or "").startswith("W_"):
            return

        token = {"state": "queued", "work_item_id": wid}
        payload["_queue_token"] = token
        engine.schedule(SimEvent(
            timestamp=engine.now + max(0.0, float(delay)),
            priority=5,
            event_type=EventType.ACTIVITY_WITHDRAW,
            case_id=event.case_id,
            activity=event.activity,
            resource=None,
            payload=payload,
        ))

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
        if self._drl:
            if event.resource is not None:
                self._busy[event.resource] = max(
                    0, self._busy.get(event.resource, 0) - 1)
                active = self._drl_active_activities.get(event.resource, [])
                if event.payload in active:
                    active.remove(event.payload)
                elif active:
                    # Defensive fallback for a caller that omits the completed
                    # activity from the release event.
                    active.pop(0)
            elif event.payload == _SHIFT_WAKE:
                self._wake_at = None
            self._drive_drl(engine)
            self._arm_shift_wake(engine)
            return

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

        if self._parksong or self._krm_delta is not None:
            if event.resource is not None:
                self._busy[event.resource] = max(0, self._busy.get(event.resource, 0) - 1)
            elif event.payload == _SHIFT_WAKE:
                self._wake_at = None
            self._epoch_flush(engine)
            self._arm_shift_wake(engine)
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
                        # Piled Execution's "same activity" preference is for fresh
                        # work items, not resume-ready re-requests (§4.6).
                        if isinstance(waiting.payload, dict) and waiting.payload.get("resuming"):
                            continue
                        ct, when = self._context(engine, waiting)
                        if not self._permissions.permits(
                                resource, waiting.activity, case_type=ct, when=when):
                            continue
                        self._waiting.pop(i)
                        self._begin(engine, waiting, resource)
                        self._arm_shift_wake(engine)
                        return

                if self._pull is not None:
                    # Pull discipline: the freed resource picks its preferred
                    # permitted item (see constructor docstring), instead of
                    # the system's FIFO first-permitted scan below. Strict
                    # "<" keeps the earliest-queued item on ties, so the
                    # rule is deterministic and consumes no RNG.
                    best_i = None
                    best_key = None
                    for i, waiting in enumerate(self._waiting):
                        ct, when = self._context(engine, waiting)
                        if not self._permissions.permits(
                                resource, waiting.activity, case_type=ct, when=when):
                            continue
                        if self._pull == "spt":
                            key = self._duration_model.expected_duration(
                                waiting.activity, resource)
                        else:  # "laf"
                            key = self._case_first_seen.get(
                                waiting.case_id, waiting.timestamp)
                        if best_key is None or key < best_key:
                            best_key, best_i = key, i
                    if best_i is not None:
                        waiting = self._waiting.pop(best_i)
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
    # D3 Deep RL — masked resource/activity assignment + postpone
    # ------------------------------------------------------------------

    @property
    def drl_external_control(self) -> bool:
        return self._drl and self._drl_external_control

    @property
    def drl_decision_pending(self) -> bool:
        return self._drl_decision_pending

    @property
    def drl_action_count(self) -> int:
        return self._drl_action_count_for(self._drl_action_version)

    def _drl_action_count_for(self, version: int) -> int:
        if version == 1:
            return len(self._drl_resources) * len(self._drl_activities) + 1
        if version == 2:
            return len(self._drl_compact_action_pairs) + 1
        raise ValueError(f"unknown DRL action version {version!r}")

    @property
    def drl_observation_size(self) -> int:
        return self._drl_observation_size_for(self._drl_observation_version)

    def _drl_observation_size_for(self, version: int) -> int:
        # Per resource: free-capacity fraction, on-shift flag, and from V2 the
        # normalized activity currently being executed. Per activity: queue
        # length and oldest queue wait, plus from V3 oldest case age and mean
        # expected duration. Globals: queue/WIP proxy, mean busy, and cyclical
        # hour + weekday (sin/cos pairs).
        per_resource = 3 if version >= 2 else 2
        per_activity = 4 if version >= 3 else 2
        return (per_resource * len(self._drl_resources)
                + per_activity * len(self._drl_activities) + 6)

    @property
    def drl_postpone_action(self) -> int:
        return self.drl_action_count - 1

    def _drl_decode_action(self, action: int) -> tuple[str, str]:
        if action < 0 or action >= self.drl_postpone_action:
            raise ValueError(f"not an assignment action: {action}")
        if self._drl_action_version == 2:
            return self._drl_compact_action_pairs[action]
        n_activities = len(self._drl_activities)
        resource_i, activity_i = divmod(action, n_activities)
        return self._drl_resources[resource_i], self._drl_activities[activity_i]

    def drl_action_for(self, resource: str, activity: str) -> int:
        """Return the stable integer action for a resource/activity pair."""
        try:
            resource_i = self._drl_resource_index[resource]
            activity_i = self._drl_activity_index[activity]
        except KeyError as exc:
            raise ValueError(f"unknown DRL resource/activity: {exc.args[0]!r}") from exc
        if self._drl_action_version == 2:
            try:
                return self._drl_compact_action_index[(resource, activity)]
            except KeyError as exc:
                raise ValueError(
                    f"globally ineligible DRL pair: ({resource!r}, {activity!r})"
                ) from exc
        return resource_i * len(self._drl_activities) + activity_i

    def _drl_request_for(self, engine, resource: str, activity: str) -> Optional[SimEvent]:
        """Oldest queued instance matching a resource/activity action."""
        if (resource in self._excluded
                or self._busy.get(resource, 0) >= self._capacity
                or not self._is_on_shift(engine, resource)):
            return None
        for request in self._waiting:
            if request.activity != activity:
                continue
            ct, when = self._context(engine, request)
            if self._permissions.permits(
                    resource, activity, case_type=ct, when=when):
                return request
        return None

    def drl_action_mask(self, engine) -> np.ndarray:
        """Boolean feasibility mask over fixed resource/activity actions."""
        mask = np.zeros(self.drl_action_count, dtype=bool)
        if not self._drl:
            mask[-1] = True
            return mask

        free = [
            r for r in self._drl_resources
            if r not in self._excluded
            and self._busy.get(r, 0) < self._capacity
            and self._is_on_shift(engine, r)
        ]
        # Iterate waiting items rather than the full resource x activity x
        # queue cube. Contextual OrdinoR permissions can differ by case, so a
        # pair is feasible if at least one queued instance admits it.
        for request in self._waiting:
            activity_i = self._drl_activity_index.get(request.activity)
            if activity_i is None:
                continue
            ct, when = self._context(engine, request)
            for resource in free:
                if self._permissions.permits(
                        resource, request.activity, case_type=ct, when=when):
                    if self._drl_action_version == 2:
                        action = self._drl_compact_action_index.get(
                            (resource, request.activity))
                    else:
                        resource_i = self._drl_resource_index[resource]
                        action = (
                            resource_i * len(self._drl_activities) + activity_i)
                    if action is not None:
                        mask[action] = True
        mask[-1] = True  # strategic idling is always available
        return mask

    def drl_shortest_processing_action(self, engine) -> int:
        """Return a feasible SPT expert action for imitation warm-starting.

        This deliberately uses the same expected-duration information exposed
        in observation V3.  Ties are stable by action id, making demonstration
        generation exactly reproducible for a fixed simulation seed.
        """
        mask = self.drl_action_mask(engine)
        feasible = np.flatnonzero(mask[:-1])
        if not len(feasible):
            return self.drl_postpone_action
        return min(
            (int(action) for action in feasible),
            key=lambda action: (
                self._drl_expected_duration[
                    self._drl_decode_action(action)[1]
                ],
                action,
            ),
        )

    def drl_observation(self, engine) -> np.ndarray:
        """Normalized, fixed-size process state used by MaskablePPO."""
        values: List[float] = []
        for resource in self._drl_resources:
            free_fraction = max(
                0.0,
                (self._capacity - self._busy.get(resource, 0)) / self._capacity,
            )
            values.extend((min(1.0, free_fraction), float(self._is_on_shift(engine, resource))))
            if self._drl_observation_version >= 2:
                active = self._drl_active_activities.get(resource, [])
                activity_i = (
                    self._drl_activity_index.get(active[0], -1) if active else -1
                )
                values.append(
                    (activity_i + 1) / max(1, len(self._drl_activities))
                )

        by_activity: Dict[str, List[SimEvent]] = {
            activity: [] for activity in self._drl_activities
        }
        for request in self._waiting:
            if request.activity in by_activity:
                by_activity[request.activity].append(request)
        for activity in self._drl_activities:
            queued = by_activity[activity]
            values.append(min(len(queued) / 100.0, 1.0))
            oldest = max(
                (engine.now - request.timestamp for request in queued),
                default=0.0,
            )
            values.append(min(max(oldest, 0.0) / (7.0 * 86400.0), 1.0))
            if self._drl_observation_version >= 3:
                oldest_case_age = max((
                    engine.now - self._case_first_seen.get(
                        request.case_id, request.timestamp)
                    for request in queued
                ), default=0.0)
                values.append(min(
                    max(oldest_case_age, 0.0) / (30.0 * 86400.0), 1.0))
                values.append(min(
                    self._drl_expected_duration[activity] / (8.0 * 3600.0),
                    1.0,
                ))

        total_capacity = max(1, len(self._drl_resources) * self._capacity)
        values.append(min(len(self._waiting) / 100.0, 1.0))
        values.append(min(sum(self._busy.values()) / total_capacity, 1.0))

        if self._start is None:
            seconds_of_day = float(engine.now) % 86400.0
            weekday = int(float(engine.now) // 86400.0) % 7
        else:
            now_wall = self._now_wall(engine)
            seconds_of_day = (
                now_wall.hour * 3600 + now_wall.minute * 60 + now_wall.second)
            weekday = now_wall.weekday()
        day_angle = 2.0 * np.pi * seconds_of_day / 86400.0
        week_angle = 2.0 * np.pi * weekday / 7.0
        values.extend((
            (np.sin(day_angle) + 1.0) / 2.0,
            (np.cos(day_angle) + 1.0) / 2.0,
            (np.sin(week_angle) + 1.0) / 2.0,
            (np.cos(week_angle) + 1.0) / 2.0,
        ))
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.drl_observation_size,):
            raise AssertionError(
                f"DRL observation shape {observation.shape} != "
                f"({self.drl_observation_size},)"
            )
        return observation

    def _prepare_drl_decision(self, engine) -> None:
        self._drl_decision_pending = bool(
            self.drl_action_mask(engine)[:-1].any())

    def apply_drl_action(self, engine, action: int) -> None:
        """Apply one masked action, used by both Gym and frozen inference."""
        if not self._drl:
            raise RuntimeError("DRL mode is not enabled")
        action = int(action)
        if action < 0 or action >= self.drl_action_count:
            raise ValueError(f"DRL action {action} outside [0, {self.drl_action_count})")
        mask = self.drl_action_mask(engine)
        if not mask[action]:
            raise ValueError(f"infeasible DRL action {action}")

        self._drl_decision_pending = False
        if action == self.drl_postpone_action:
            self._drl_postponements += 1
            return

        resource, activity = self._drl_decode_action(action)
        request = self._drl_request_for(engine, resource, activity)
        if request is None:  # Defensive: mask and application must agree.
            raise RuntimeError(
                f"masked action ({resource}, {activity}) lost its queued request")
        self._waiting.remove(request)
        self._begin(engine, request, resource)
        self._drl_assignments += 1
        # Several free slots/tasks may exist at the same simulation instant.
        self._prepare_drl_decision(engine)

    def _drive_drl(self, engine) -> None:
        """Pause for Gym control or run a frozen model until it postpones."""
        self._prepare_drl_decision(engine)
        if self._drl_external_control or not self._drl_decision_pending:
            return

        max_actions = max(1, len(self._waiting) + len(self._drl_resources) * self._capacity)
        for _ in range(max_actions):
            if not self._drl_decision_pending:
                return
            observation = self.drl_observation(engine)
            mask = self.drl_action_mask(engine)
            action, _ = self._drl_model.predict(
                observation, action_masks=mask, deterministic=True)
            action = int(np.asarray(action).item())
            self.apply_drl_action(engine, action)
            if action == self.drl_postpone_action:
                return
        raise RuntimeError("DRL inference exceeded the same-instant action guard")

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

    # ------------------------------------------------------------------
    # Advanced policies (Part II): D1 Park & Song, D2 Kunkler & Rinderle-Ma
    # ------------------------------------------------------------------

    def on_case_complete(self, engine, event: SimEvent) -> None:
        """Drop D1's in-service bookkeeping for a finished case, so no
        phantom successor is planned for it. No-op outside parksong mode."""
        if self._parksong and event.case_id is not None:
            self._active_cases.pop(event.case_id, None)

    def _epoch_flush(self, engine) -> None:
        """One allocation epoch for the advanced policies (see constructor
        docstrings): a single assignment over the whole waiting queue.

        D1 (parksong): rows are real waiting items PLUS phantom items (the
        predicted next activity of each in-service case). A phantom
        assigned a resource reserves it — the resource idles this epoch
        (strategic idling); only real-item assignments call _begin.

        D2 (krm): rows are the real waiting items; columns gain one
        private dummy per item costing ``krm_delta x`` the item's mean
        expected duration. An item assigned its dummy stays queued.

        Deterministic (argmin assignment, no RNG). One pass per trigger:
        every request and every release re-runs the epoch, so there is no
        max-wait valve to arm — the queue is reconsidered continuously.
        """
        if not self._waiting:
            return
        free = self._free_resources(engine)
        if not free:
            return

        real: List[SimEvent] = list(self._waiting)
        phantoms: List[SimEvent] = []
        phantom_penalty: List[float] = []
        if self._parksong:
            for case_id, (activity, payload, begun_at) in self._active_cases.items():
                pred = self._predictor.predict(activity)
                if pred is None:
                    continue
                nxt, p_next = pred
                # Strategic idling is only strategic for IMMINENT work: the
                # successor arrives when the current activity finishes, so a
                # phantom joins the epoch only once the expected remaining
                # service time is inside the lookahead. Without this gate, a
                # resource sits reserved for DAYS while a long validation
                # runs — measured: cases completed collapsed 158 -> 15 on a
                # 5-day run before the gate existed.
                eta = begun_at + self._duration_model.expected_duration(activity, None)
                if eta - engine.now > PARKSONG_LOOKAHEAD_SECONDS:
                    continue
                # Unscheduled pseudo-event: exists only to carry the
                # (activity, case payload) execution context into the cost
                # matrix — it is never scheduled, logged, or begun.
                phantoms.append(SimEvent(
                    timestamp=engine.now, priority=5,
                    event_type=EventType.ACTIVITY_REQUEST,
                    case_id=case_id, activity=nxt, payload=payload,
                ))
                # Uncertainty penalty: reserving for a p=0.5 prediction
                # should cost twice what reserving for a sure thing does.
                phantom_penalty.append(1.0 / max(p_next, 1e-6))

        items = real + phantoms
        n, m = len(items), len(free)
        n_real = len(real)
        n_dummies = n_real if self._krm_delta is not None else 0

        PROHIBITIVE = 1e12
        cost = np.full((n, m + n_dummies), PROHIBITIVE)
        for i, req in enumerate(items):
            ct, when = self._context(engine, req)
            penalty = phantom_penalty[i - n_real] if i >= n_real else 1.0
            for j, r in enumerate(free):
                if self._permissions.permits(r, req.activity, case_type=ct, when=when):
                    cost[i, j] = penalty * self._duration_model.expected_duration(req.activity, r)

        # Phantom spread gate (D1): reserving a resource for predicted work
        # only pays if that resource is a strictly BETTER fit than the
        # alternatives — with resource-independent expected durations (no
        # trained artifact) every reservation can only hurt, never help,
        # and measured occupation collapsed to 0.08 vs 0.47 baseline. A
        # phantom whose permitted costs show no spread is dropped from the
        # epoch (its row is set prohibitive so the solver ignores it).
        for i in range(n_real, n):
            row = cost[i, :m]
            finite = row[row < PROHIBITIVE]
            if len(finite) == 0 or finite.max() - finite.min() <= 1e-9:
                cost[i, :m] = PROHIBITIVE

        if n_dummies:
            now = engine.now
            for i, req in enumerate(real):
                # Deferral price = delta x the item's baseline duration,
                # PLUS the time it has already waited (aging). Without the
                # aging term, delta < 1 under resource-independent costs
                # defers everything forever — measured: a 5-day run logged
                # zero events. With it, deferral stays attractive only
                # until the accumulated wait eats the margin.
                cost[i, m + i] = (self._krm_delta * distribution_mean_seconds(req.activity)
                                  + (now - req.timestamp))

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_ids = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= PROHIBITIVE:
                continue
            if i >= n_real:
                continue  # phantom matched a resource: reservation, idle it
            if j >= m:
                continue  # item matched its dummy: defer, stays queued
            req = items[i]
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
        """Can *anyone ever* perform this work item, independent of time?

        Distinguishes "everyone is busy or off-shift" (queue and wait) from
        "nobody may ever do this" (a hole in the permission model — queuing would
        strand the case forever).

        The current time must deliberately be omitted here. An OrdinoR model can
        grant a capability on some weekdays but not others. Zero candidates on
        Sunday therefore means "wait for a permitted weekday", not "run the item
        unassigned". ``_allocate`` still uses the full current context, so this
        broader check cannot assign an impermissible resource; it only decides
        whether the request belongs in the queue.
        """
        ct, _ = self._context(engine, event)
        return bool(self._permissions.candidates(
            event.activity, case_type=ct, when=None))

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

        This scan reimplements the calendar's own checks (holiday, vacation and
        — since rostering — `_works_today`) rather than calling `is_available`,
        because it asks a different question: not "is r on duty now?" but "when
        does r's window next open?". The roster check must be mirrored here or
        the simulation wakes up expecting staff who are not rostered that day
        and stalls with work still queued. The hash draw is what makes the two
        sites agree for free.
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
                if not self._calendar._works_today(res, d):
                    continue
                t = (midnight + timedelta(hours=w[0]) - self._start).total_seconds()
                if t > engine.now and (best is None or t < best):
                    best = t

            if best is not None:
                return best
        return best

    def _begin(self, engine, request: SimEvent, resource: Optional[str]) -> None:
        """Turn a granted request into an ACTIVITY_START bound to *resource*.

        This is the only place an ACTIVITY_START (or, for a resuming re-request,
        an ACTIVITY_RESUME) is created, which is what keeps a queued work item
        from also being executed by the ProcessComponent.

        Active mode (§4.6): a request carrying ``payload.resuming=True`` is a
        suspended item that just re-acquired a resource, so it fires
        ACTIVITY_RESUME instead of ACTIVITY_START. The single logged `resume`
        therefore always means resource-bound RUNNING work, never "ready but
        unassigned". Because the re-request goes through normal allocation, the
        resuming resource is drawn from the pool (reproducing the observed
        82.5% different-resource rate).
        """
        token = self._queue_token(request)
        if token is not None:
            if token.get("state") != "queued":
                return
            token["state"] = "allocated"

        if self._parksong and request.case_id is not None:
            # D1 bookkeeping: this case is now in service on this activity —
            # its predicted successor becomes a phantom item at the next
            # allocation epoch, but only once the successor's arrival is
            # imminent (begin time + expected duration inside the lookahead;
            # see _epoch_flush). Overwritten on the case's next _begin,
            # dropped on CASE_COMPLETE.
            self._active_cases[request.case_id] = (
                request.activity, request.payload, engine.now)

        if resource is not None:
            self._busy[resource] = self._busy.get(resource, 0) + 1
            if self._drl:
                self._drl_active_activities.setdefault(resource, []).append(
                    request.activity
                )

        waited = engine.now - request.timestamp
        self._wait_total += waited
        self._wait_count += 1

        resuming = isinstance(request.payload, dict) and request.payload.get("resuming")
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_RESUME if resuming else EventType.ACTIVITY_START,
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
            "drl_assignments": self._drl_assignments,
            "drl_postponements": self._drl_postponements,
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
    EventType.ACTIVITY_WITHDRAW:  ResourceComponent.on_activity_withdraw,
    EventType.RESOURCE_AVAILABLE: ResourceComponent.on_resource_available,
    # D1 bookkeeping only (no-op outside parksong mode).
    EventType.CASE_COMPLETE:      ResourceComponent.on_case_complete,
}

"""
process.py — Process Component (Sections 1.3 Basic + 1.5 Basic)
================================================================
Implements:
  - Branching decisions via empirical probabilities from BPIC-17 (Section 1.5 Basic)
  - Processing times sampled from distributions fitted on BPIC-17 (Section 1.3 Basic)

The process model is a probability-weighted next-activity graph derived
directly from the branching_probs in simulation_inputs.json.
This satisfies the Basic requirement for Section 1.4 (a selected process
model is enforced) and Section 1.5 (branch probabilities used at decision points).

Upgrade path:
  - Section 1.4 Advanced: replace _next_activity() with a Petri net / BPMN loader
  - Section 1.3 Advanced I: replace _sample_duration() with a probabilistic ML model
  - Section 1.5 Advanced II: replace _next_activity() with a trained next-activity predictor

Common Random Numbers (CRN, opt-in via crn=True)
-------------------------------------------------
By default all stochastic draws in this component (branching, duration
sampling) come from one shared ``self._rng``, consumed in whatever order
the engine dispatches events. That means changing something unrelated to
this component — e.g. which resource-allocation policy is active — changes
the event-dispatch order, which changes every subsequent draw from this
shared RNG too. Two policies run under "the same seed" then see different
arrivals-aside trajectories: not a controlled comparison (see
output/piled_execution_eval.md for the empirical consequence).

With ``crn=True``, each branching/duration decision instead draws from a
fresh ``random.Random`` seeded deterministically from
``(base_seed, case_id, activity, kind, visit)`` — see ``_draw_rng()``.
This makes a given case's Nth visit to a given activity draw the same
branch and the same duration regardless of what else happened first in
the event queue, so paired experiments across allocation policies (Part
II) actually compare the same case trajectories up to the point they
diverge on allocation. Scope: this covers branching and duration draws
only. Case/offer-attribute sampling in petri_process.py's "rules" mode
(applicant type, loan goal, requested amount, offer terms) is out of
scope — it's a one-time draw per case/offer rather than a per-event draw
in the dispatch-order-sensitive hot path, so it's a much smaller residual
source of cross-policy divergence in "rules" mode specifically.
"""

import hashlib
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.events import SimEvent, EventType

# Default anchor for t=0 (overridden by main.py's start_datetime). Used to
# derive day_of_week / hour_of_day features from the simulation clock.
_DEFAULT_ANCHOR = datetime(2016, 1, 1)


# ── Terminal activities: after these the case ends ──────────────────────────
# Derived from BPIC-17: activities that have no outgoing edges in our graph
# OR are the logical end states of the loan application process.
TERMINAL_ACTIVITIES = {
    "W_Validate application",   # often last in accepted/pending paths
    "W_Call after offers",      # often last in cancelled paths
    "W_Call incomplete files",  # often last in incomplete paths
    "W_Personal Loan collection",
}

# How many times an activity can repeat within one case before we force-stop
# (guards against infinite loops in self-looping activities like W_Call after offers)
MAX_ACTIVITY_REPEATS = 15


# ── Processing time distributions fitted on BPIC-17 ────────────────────────
# Format: activity -> (distribution_name, params)
# Params follow scipy.stats convention: (shape..., loc, scale)
# For activities not in the fitted set, fallback to Exponential(mean=600s).
PROCESSING_TIME_PARAMS: Dict[str, Tuple[str, tuple]] = {
    "W_Assess potential fraud": ("gamma",       (0.3057, 0.0, 1090175.6221)),
    "W_Call after offers":      ("lognorm",     (3.9715, 0.0, 184.7252)),
    "W_Call incomplete files":  ("weibull_min", (0.4091, 0.0, 226141.4883)),
    "W_Complete application":   ("lognorm",     (2.2212, 0.0, 2135.1602)),
    "W_Handle leads":           ("lognorm",     (1.5805, 0.0, 128.1099)),
    "W_Validate application":   ("gamma",       (0.3213, 0.0, 1056395.7929)),
}

# Mean durations (seconds) for activities without a fitted distribution
# (A_ and O_ activities — estimated from BPIC-17 context)
FALLBACK_MEAN_DURATIONS: Dict[str, float] = {
    "A_Create Application": 120,
    "A_Submitted":          60,
    "A_Concept":            300,
    "A_Accepted":           120,
    "A_Complete":           180,
    "A_Cancelled":          60,
    "A_Denied":             60,
    "A_Incomplete":         120,
    "A_Pending":            300,
    "A_Validating":         3600,
    "O_Create Offer":       300,
    "O_Created":            30,
    "O_Sent (mail and online)": 120,
    "O_Sent (online only)": 60,
    "O_Accepted":           60,
    "O_Cancelled":          60,
    "O_Refused":            60,
    "O_Returned":           120,
    "W_Shortened completion ":  600,
    "W_Personal Loan collection": 3600,
}

# ── Branching probabilities from BPIC-17 ────────────────────────────────────
# activity -> [(next_activity, probability), ...]  (sorted desc by prob)
BRANCHING_PROBS: Dict[str, List[Tuple[str, float]]] = {
    "A_Accepted": [
        ("O_Create Offer",           0.951),
        ("W_Complete application",   0.0488),
        ("W_Shortened completion ",  0.0002),
    ],
    "A_Cancelled": [
        ("O_Cancelled",              0.9846),
        ("W_Call after offers",      0.0099),
        ("W_Call incomplete files",  0.0039),
        ("W_Complete application",   0.0013),
        ("W_Validate application",   0.0003),
    ],
    "A_Complete": [
        ("W_Call after offers",      0.9692),
        ("O_Create Offer",           0.0166),
        ("O_Cancelled",              0.0092),
        ("A_Cancelled",              0.0018),
        ("W_Shortened completion ",  0.0014),
        ("O_Sent (mail and online)", 0.0012),
        ("A_Denied",                 0.0005),
        ("O_Sent (online only)",     0.0001),
    ],
    "A_Concept": [
        ("W_Complete application",   0.7066),
        ("A_Accepted",               0.2930),
        ("W_Shortened completion ",  0.0004),
    ],
    "A_Create Application": [
        ("A_Submitted",              0.6482),
        ("W_Complete application",   0.2443),
        ("A_Concept",                0.1076),
    ],
    "A_Denied": [
        ("O_Refused",                0.9915),
        ("W_Complete application",   0.0029),
        ("W_Call incomplete files",  0.0027),
        ("W_Call after offers",      0.0024),
        ("W_Validate application",   0.0005),
    ],
    "A_Incomplete": [
        ("W_Call incomplete files",  0.9663),
        ("O_Returned",               0.0324),
        ("O_Accepted",               0.0008),
        ("A_Denied",                 0.0004),
        ("O_Create Offer",           0.0001),
    ],
    "A_Pending": [
        ("W_Validate application",   0.7134),
        ("W_Call incomplete files",  0.2865),
        ("W_Call after offers",      0.0001),
    ],
    "A_Submitted": [
        ("W_Handle leads",           1.0),
    ],
    "A_Validating": [
        ("O_Returned",               0.5326),
        ("W_Validate application",   0.4618),
        ("O_Accepted",               0.0051),
        ("A_Denied",                 0.0003),
        ("A_Cancelled",              0.0002),
        ("O_Create Offer",           0.0001),
    ],
    "O_Accepted": [
        ("A_Pending",                1.0),
    ],
    "O_Cancelled": [
        ("W_Call after offers",      0.5867),
        ("O_Cancelled",              0.2689),
        ("W_Call incomplete files",  0.0559),
        ("O_Create Offer",           0.0418),
        ("O_Sent (mail and online)", 0.0202),
        ("A_Cancelled",              0.0101),
        ("W_Complete application",   0.0076),
        ("W_Validate application",   0.0061),
        ("A_Denied",                 0.0018),
        ("O_Sent (online only)",     0.0007),
    ],
    "O_Create Offer": [
        ("O_Created",                1.0),
    ],
    "O_Created": [
        ("O_Sent (mail and online)", 0.8281),
        ("O_Create Offer",           0.0904),
        ("O_Sent (online only)",     0.0441),
        ("W_Complete application",   0.0168),
        ("O_Cancelled",              0.0164),
        ("A_Cancelled",              0.0018),
        ("W_Call after offers",      0.0011),
        ("W_Call incomplete files",  0.0007),
        ("A_Denied",                 0.0004),
        ("W_Validate application",   0.0002),
    ],
    "O_Refused": [
        ("W_Validate application",   0.7005),
        ("O_Refused",                0.2077),
        ("W_Call incomplete files",  0.0415),
        ("W_Call after offers",      0.0232),
        ("W_Assess potential fraud", 0.0217),
        ("W_Complete application",   0.0053),
    ],
    "O_Returned": [
        ("W_Validate application",   0.9112),
        ("W_Call incomplete files",  0.0659),
        ("O_Accepted",               0.0207),
        ("A_Denied",                 0.0011),
        ("O_Returned",               0.0007),
        ("A_Cancelled",              0.0002),
    ],
    "O_Sent (mail and online)": [
        ("W_Complete application",   0.7788),
        ("W_Call after offers",      0.0869),
        ("O_Sent (mail and online)", 0.0784),
        ("W_Call incomplete files",  0.0188),
        ("A_Cancelled",              0.0138),
        ("O_Create Offer",           0.0102),
        ("O_Cancelled",              0.0091),
        ("O_Returned",               0.0025),
        ("W_Validate application",   0.0011),
        ("A_Denied",                 0.0002),
    ],
    "O_Sent (online only)": [
        ("W_Call incomplete files",  0.4301),
        ("W_Complete application",   0.2230),
        ("W_Call after offers",      0.1487),
        ("O_Returned",               0.0605),
        ("O_Create Offer",           0.0436),
        ("O_Sent (online only)",     0.0421),
        ("A_Cancelled",              0.0223),
        ("W_Validate application",   0.0139),
        ("O_Cancelled",              0.0134),
        ("O_Accepted",               0.0015),
        ("A_Denied",                 0.0005),
    ],
    "W_Assess potential fraud": [
        ("W_Assess potential fraud", 0.9013),
        ("W_Validate application",   0.0522),
        ("A_Denied",                 0.0321),
        ("W_Handle leads",           0.0069),
        ("W_Call after offers",      0.0038),
        ("W_Complete application",   0.0038),
    ],
    "W_Call after offers": [
        ("W_Call after offers",      0.6363),
        ("A_Complete",               0.1727),
        ("W_Validate application",   0.1205),
        ("A_Cancelled",              0.0470),
        ("O_Create Offer",           0.0216),
        ("O_Cancelled",              0.0011),
        ("A_Denied",                 0.0005),
    ],
    "W_Call incomplete files": [
        ("W_Call incomplete files",  0.6952),
        ("A_Incomplete",             0.1407),
        ("W_Validate application",   0.1033),
        ("O_Accepted",               0.0292),
        ("O_Create Offer",           0.0111),
        ("O_Cancelled",              0.0093),
        ("A_Cancelled",              0.0053),
        ("O_Returned",               0.0044),
        ("A_Denied",                 0.0011),
    ],
    "W_Complete application": [
        ("W_Complete application",   0.4349),
        ("W_Call after offers",      0.2108),
        ("A_Concept",                0.1890),
        ("A_Accepted",               0.1496),
        ("O_Create Offer",           0.0106),
        ("O_Sent (mail and online)", 0.0040),
        ("O_Cancelled",              0.0004),
        ("A_Cancelled",              0.0003),
    ],
    "W_Handle leads": [
        ("W_Handle leads",           0.5674),
        ("W_Complete application",   0.4321),
        ("W_Assess potential fraud", 0.0005),
    ],
    "W_Personal Loan collection": [
        ("W_Personal Loan collection", 0.95),
        ("W_Validate application",     0.05),
    ],
    "W_Shortened completion ": [
        ("W_Shortened completion ",  0.4979),
        ("W_Call after offers",      0.2918),
        ("A_Accepted",               0.1245),
        ("W_Complete application",   0.0515),
        ("O_Create Offer",           0.0129),
        ("W_Validate application",   0.0086),
        ("O_Sent (mail and online)", 0.0086),
        ("W_Call incomplete files",  0.0043),
    ],
    "W_Validate application": [
        ("W_Validate application",   0.5872),
        ("A_Validating",             0.1972),
        ("W_Call incomplete files",  0.1171),
        ("O_Accepted",               0.0596),
        ("A_Denied",                 0.0165),
        ("O_Cancelled",              0.0154),
        ("O_Returned",               0.0047),
        ("W_Assess potential fraud", 0.0013),
        ("O_Create Offer",           0.0005),
        ("A_Cancelled",              0.0004),
    ],
}


# ── Case/runtime attributes for data-based branching (Section 1.5 Advanced I) ─
# From BPIC-17 (case_attributes in simulation_inputs.json, produced by
# extract_log_info.py::extract_case_attributes). Only consumed by
# PetriNetProcessComponent when constructed with branching_mode="rules" —
# sampling these has no effect on the branching_mode="probs" (Basic) path.
#
# ApplicationType/LoanGoal/RequestedAmount are case-level: constant for the
# whole case, known from A_Create Application onward (verified: exactly one
# distinct value per case in the log). The offer attributes below are only
# known once a case's first O_Create Offer has fired.
APPLICATION_TYPE_PROBS: List[Tuple[str, float]] = [
    ("New credit",  0.8924),
    ("Limit raise", 0.1076),
]

LOAN_GOAL_GIVEN_APPLICATION_TYPE: Dict[str, List[Tuple[str, float]]] = {
    "New credit": [
        ("Car",                     0.301),
        ("Home improvement",        0.236),
        ("Existing loan takeover",  0.19),
        ("Other, see explanation",  0.0922),
        ("Unknown",                 0.0663),
        ("Not speficied",           0.0367),
        ("Remaining debt home",     0.0293),
        ("Extra spending limit",    0.0163),
        ("Caravan / Camper",        0.0115),
        ("Motorcycle",              0.0089),
        ("Boat",                    0.0064),
        ("Tax payments",            0.0045),
        ("Business goal",           0.0009),
        ("Debt restructuring",      0.0001),
    ],
    "Limit raise": [
        ("Home improvement",        0.3048),
        ("Car",                     0.2549),
        ("Unknown",                 0.1475),
        ("Other, see explanation",  0.1154),
        ("Existing loan takeover",  0.0758),
        ("Extra spending limit",    0.0496),
        ("Caravan / Camper",        0.0139),
        ("Not speficied",           0.0097),
        ("Tax payments",            0.0077),
        ("Motorcycle",              0.0074),
        ("Boat",                    0.0062),
        ("Remaining debt home",     0.0056),
        ("Business goal",           0.0015),
    ],
}

# (distribution_name, params) — same scipy.stats convention as
# PROCESSING_TIME_PARAMS, sampled via ProcessComponent._sample_scipy_like.
REQUESTED_AMOUNT_GIVEN_APPLICATION_TYPE: Dict[str, Tuple[str, tuple]] = {
    "New credit":  ("lognorm", (0.7108, 0.0, 13393.0654)),
    "Limit raise": ("lognorm", (0.5829, 0.0, 19823.0638)),
}

OFFER_ATTRIBUTE_PARAMS: Dict[str, Tuple[str, tuple]] = {
    "OfferedAmount":  ("lognorm",     (0.6926, 0.0, 14557.0165)),
    "NumberOfTerms":  ("weibull_min", (2.4819, 0.0, 93.7695)),
    "MonthlyCost":    ("lognorm",     (0.6083, 0.0, 233.7704)),
}

# These two have a large point mass at 0 (no value recorded) alongside a
# continuous spread on the remainder, so they're sampled as
# Bernoulli(zero_prob) ? 0.0 : draw-from-dist, not a single distribution.
OFFER_ATTRIBUTE_ZERO_MASS_PARAMS: Dict[str, dict] = {
    "FirstWithdrawalAmount": {
        "zero_prob": 0.2974,
        "dist": ("weibull_min", (1.0951, 0.0, 12356.8172)),
    },
    "CreditScore": {
        "zero_prob": 0.6451,
        "dist": ("weibull_min", (10.345, 0.0, 941.6136)),
    },
}


class ProcessComponent:
    """
    Routes cases through the BPIC-17 loan application process using
    empirical branching probabilities and fitted processing time distributions.

    Satisfies:
      - Section 1.4 Basic: a selected process model is enforced
      - Section 1.5 Basic: branch probabilities at decision points
      - Section 1.3 Basic: processing times from fitted distributions
    """

    HANDLES = {
        EventType.ACTIVITY_START:    None,
        EventType.ACTIVITY_COMPLETE: None,
    }

    # Recognised processing-time modes
    _MODES = ("distribution", "ml_model", "ml_probabilistic")

    def __init__(
        self,
        seed: Optional[int] = 42,
        mode: str = "distribution",
        model_path: Optional[str] = None,
        start_datetime: Optional[datetime] = None,
        resource_component=None,
        crn: bool = False,
    ):
        """
        Parameters
        ----------
        mode : {"distribution", "ml_model", "ml_probabilistic"}
            - "distribution"     : fitted scipy distributions (Section 1.3 Basic).
            - "ml_model"         : contextual point-estimate GBR (Basic option 2).
            - "ml_probabilistic" : quantile-GBR curve, stochastic draw (Advanced I).
        model_path : str, optional
            Path to the joblib artifact from train_processing_time_model.py.
            Required (lazy-loaded) when mode != "distribution".
        start_datetime : datetime, optional
            Real-world anchor for t=0, used to derive day_of_week / hour_of_day
            features. Should match the engine/logger anchor.
        resource_component : ResourceComponent, optional
            If provided, its resource is released on ACTIVITY_COMPLETE (via the
            component's own ``release()`` API). Without this the resource pool
            saturates permanently, leaving the ML resource feature degenerate.
        crn : bool, optional
            Common Random Numbers (see module docstring). Default False
            preserves every existing evidence log bit-for-bit. Requires
            ``seed`` to be a concrete int (not None).
        """
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        if crn and seed is None:
            raise ValueError("crn=True requires a concrete seed (got None)")

        self._rng = random.Random(seed)
        self._seed = seed
        self._crn = crn
        self.mode = mode
        self._model_path = model_path
        self._anchor = start_datetime or _DEFAULT_ANCHOR
        self._resources = resource_component

        # Lazily-loaded ML artifact (only when mode != "distribution")
        self._artifact: Optional[dict] = None
        self._model = None
        self._quantile_models: Optional[dict] = None
        self._quantiles: Optional[list] = None
        self._encoders: Optional[dict] = None
        self._encoder_classes: Dict[str, set] = {}
        self._feature_names: Optional[list] = None

        # case_id -> {activity: repeat_count}
        self._repeat_counts: Dict[str, Dict[str, int]] = {}
        # case_id -> {start_t, position, prev_act}  (context for ML features)
        self._ctx: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the joblib artifact the first time an ML duration is needed."""
        if self._artifact is not None:
            return
        if not self._model_path:
            raise ValueError(
                f"mode={self.mode!r} requires model_path to a trained "
                f"joblib artifact (run train_processing_time_model.py)."
            )
        import joblib
        self._artifact = joblib.load(self._model_path)
        self._model = self._artifact["model"]
        self._encoders = self._artifact["encoders"]
        self._feature_names = self._artifact["feature_names"]
        self._encoder_classes = {
            name: set(le.classes_) for name, le in self._encoders.items()
        }
        sentinels = self._artifact.get("sentinels", {})
        self._unknown = sentinels.get("unknown", "__UNKNOWN__")
        self._no_prev = sentinels.get("no_prev", "__START__")

        if self.mode == "ml_probabilistic":
            self._quantile_models = self._artifact.get("quantile_models")
            self._quantiles = self._artifact.get("quantiles")
            if not self._quantile_models:
                raise ValueError(
                    "mode='ml_probabilistic' needs quantile models — retrain "
                    "with `--probabilistic`."
                )

    # ------------------------------------------------------------------
    # Common Random Numbers (see module docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _crn_seed(base_seed: int, *parts) -> int:
        """Deterministic derived seed from (base_seed, *parts).

        Not Python's builtin ``hash()``: that's salted per-process for str
        (PYTHONHASHSEED), so it would silently break cross-run reproducibility
        — the one thing this whole project's grading depends on.
        """
        key = "|".join(str(p) for p in (base_seed, *parts)).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    def _draw_rng(self, case_id: str, activity: Optional[str], kind: str, visit: int = 1):
        """RNG to use for one branching/duration decision.

        crn=False (default): the shared ``self._rng``, in whatever order the
        engine dispatches events — today's unchanged behaviour.
        crn=True: a fresh ``random.Random`` seeded from
        (base_seed, case_id, activity, kind, visit), independent of dispatch
        order. ``kind`` (e.g. "duration" vs "branch") keeps two different
        draws for the same (case, activity, visit) from colliding on one seed.
        """
        if not self._crn:
            return self._rng
        seed = self._crn_seed(self._seed, case_id, activity, kind, visit)
        return random.Random(seed)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        case_id = event.case_id

        # Sentinel: initialise case and start with A_Create Application
        if event.activity == "__PROCESS_START__":
            self._repeat_counts[case_id] = {}
            self._ctx[case_id] = {
                "start_t": engine.now,   # case age is measured from here
                "position": 0,           # activities started so far
                "prev_act": None,        # previous activity (None => first)
            }
            self._fire_start(engine, case_id, "A_Create Application")
            return

        # Normal start: sample duration (context-aware in ML modes), then
        # schedule ACTIVITY_COMPLETE. Context is read *before* this activity
        # is folded in, so the features describe the state at its start.
        activity = event.activity
        ctx = self._ctx.get(case_id) or {
            "start_t": engine.now, "position": 0, "prev_act": None
        }
        duration = self._duration(engine, event, ctx)

        # Fold this activity into the case context for the next sample.
        ctx["position"] += 1
        ctx["prev_act"] = activity
        self._ctx[case_id] = ctx

        engine.schedule(SimEvent(
            timestamp=engine.now + duration,
            priority=5,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            resource=event.resource,
        ))

    def on_activity_complete(self, engine, event: SimEvent) -> None:
        case_id  = event.case_id
        activity = event.activity

        # Free the resource that ran this activity so the pool doesn't saturate
        # (uses ResourceComponent's documented release() API; high-priority
        # RESOURCE_AVAILABLE fires before the next activity's allocation).
        if self._resources is not None and event.resource:
            self._resources.release(engine, event.resource, event.activity)

        # Track repeats for loop-guard
        counts = self._repeat_counts.get(case_id, {})
        counts[activity] = counts.get(activity, 0) + 1
        self._repeat_counts[case_id] = counts

        # Decide termination
        if self._should_terminate(case_id, activity, counts):
            self._repeat_counts.pop(case_id, None)
            self._ctx.pop(case_id, None)
            engine.schedule(SimEvent(
                timestamp=engine.now,
                priority=20,
                event_type=EventType.CASE_COMPLETE,
                case_id=case_id,
            ))
            return

        # Choose and schedule the next activity
        next_act = self._next_activity(case_id, activity, counts.get(activity, 1))
        if next_act is None:
            # No outgoing edge defined — treat as terminal
            self._repeat_counts.pop(case_id, None)
            self._ctx.pop(case_id, None)
            engine.schedule(SimEvent(
                timestamp=engine.now,
                priority=20,
                event_type=EventType.CASE_COMPLETE,
                case_id=case_id,
            ))
            return

        self._fire_start(engine, case_id, next_act)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fire_start(self, engine, case_id: str, activity: str) -> None:
        # Emit a *request*, not a start. ResourceComponent is the only component
        # that turns a request into an ACTIVITY_START, and only once it actually
        # holds a resource — so the work item cannot begin (or be logged) while
        # it is still queued. See ResourceComponent for the full rationale.
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_REQUEST,
            case_id=case_id,
            activity=activity,
            resource=None,
        ))

    def _should_terminate(
        self, case_id: str, activity: str, counts: Dict[str, int]
    ) -> bool:
        """Terminate if activity is terminal OR any loop limit is exceeded."""
        if activity in TERMINAL_ACTIVITIES:
            return True
        # Guard: any single activity repeated too many times → end case
        if counts.get(activity, 0) >= MAX_ACTIVITY_REPEATS:
            return True
        # Guard: total activities in case is getting very large
        total = sum(counts.values())
        if total > 150:
            return True
        return False

    def _next_activity(self, case_id: str, current: str, visit: int = 1) -> Optional[str]:
        """
        Sample the next activity using empirical branching probabilities.
        Returns None if current activity has no outgoing edges.

        ``visit`` is the 1-based count of how many times *current* has
        occurred in this case so far (the caller already tracks this in
        ``counts``) — used as part of the CRN draw key when crn=True.
        """
        options = BRANCHING_PROBS.get(current)
        if not options:
            return None

        rng = self._draw_rng(case_id, current, "branch", visit)
        r = rng.random()
        cumulative = 0.0
        for next_act, prob in options:
            cumulative += prob
            if r <= cumulative:
                return next_act
        # Floating-point safety: return last option
        return options[-1][0]

    # ------------------------------------------------------------------
    # Duration dispatch (distribution vs. ML)
    # ------------------------------------------------------------------

    def _duration(self, engine, event: SimEvent, ctx: dict) -> float:
        """Route to the configured processing-time model."""
        # This instance's 1-based visit count: at ACTIVITY_START time
        # _repeat_counts hasn't been incremented for this occurrence yet.
        visit = self._repeat_counts.get(event.case_id, {}).get(event.activity, 0) + 1
        rng = self._draw_rng(event.case_id, event.activity, "duration", visit)

        if self.mode == "distribution":
            return self._sample_duration(event.activity, rng)

        self._ensure_model()
        features = self._build_features(engine, event, ctx)
        if self.mode == "ml_model":
            return self._sample_duration_ml(features)
        return self._sample_duration_ml_prob(features, rng)   # ml_probabilistic

    def _encode(self, name: str, value, fallback: str) -> int:
        """Label-encode *value*, falling back to a sentinel for unseen labels."""
        value = str(value)
        if value not in self._encoder_classes[name]:
            value = fallback
        return int(self._encoders[name].transform([value])[0])

    def _build_features(self, engine, event: SimEvent, ctx: dict) -> List[float]:
        """
        Reconstruct the 8-feature vector (in the artifact's feature order)
        for the activity that is about to start.

        NOTE: the sampled duration is the full start→complete time (service +
        any waiting). The probabilistic (Advanced I) model captures this
        directly; splitting service vs. waiting (Advanced II) is optional.
        """
        # Wall-clock derived features come from the run's start anchor.
        wall = self._anchor + timedelta(seconds=engine.now)
        prev_act = ctx.get("prev_act") or self._no_prev
        resource = event.resource or self._unknown
        position = ctx.get("position", 0)
        case_age = max(0.0, engine.now - ctx.get("start_t", engine.now))

        values = {
            "activity_enc":           self._encode("activity", event.activity, self._unknown),
            "resource_enc":           self._encode("resource", resource, self._unknown),
            "previous_activity_enc":  self._encode("previous_activity", prev_act, self._no_prev),
            "day_of_week":            wall.weekday(),
            "hour_of_day":            wall.hour,
            "case_position":          position,
            "case_age_seconds":       case_age,
            "n_previous_activities":  position,
        }
        return [float(values[name]) for name in self._feature_names]

    def _sample_duration_ml(self, features: List[float]) -> float:
        """Point-estimate: predict log-duration, invert log1p, clamp to ≥ 1s."""
        import numpy as np
        pred_log = self._model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0]
        return max(1.0, float(np.expm1(pred_log)))

    def _sample_duration_ml_prob(self, features: List[float], rng=None) -> float:
        """
        Probabilistic (Advanced I): predict the conditional quantile curve,
        enforce monotonicity, draw u ~ Uniform(0,1) and interpolate — this
        restores the variance a point estimate discards.

        ``rng`` defaults to ``self._rng`` for callers outside the CRN path
        (e.g. direct unit use); ``_duration()`` always passes one explicitly.
        """
        import numpy as np
        rng = rng if rng is not None else self._rng
        x = np.asarray(features, dtype=float).reshape(1, -1)
        qs = self._quantiles
        preds_log = np.array([self._quantile_models[q].predict(x)[0] for q in qs])
        # Clip quantile crossings so the curve is non-decreasing.
        preds_log = np.maximum.accumulate(preds_log)
        u = rng.random()
        # np.interp clamps u outside [qs[0], qs[-1]] to the edge predictions.
        dur_log = float(np.interp(u, qs, preds_log))
        return max(1.0, float(np.expm1(dur_log)))

    def _sample_duration(self, activity: str, rng=None) -> float:
        """
        Sample processing time in seconds for an activity.
        Uses fitted scipy distributions where available, exponential fallback otherwise.

        ``rng`` defaults to ``self._rng``; ``_duration()`` passes the
        CRN-derived draw RNG explicitly when crn=True.
        """
        rng = rng if rng is not None else self._rng
        if activity in PROCESSING_TIME_PARAMS:
            dist_name, params = PROCESSING_TIME_PARAMS[activity]
            return max(1.0, self._sample_scipy_like(dist_name, params, rng))

        mean = FALLBACK_MEAN_DURATIONS.get(activity, 600.0)
        return max(1.0, rng.expovariate(1.0 / mean))

    def _sample_scipy_like(self, dist_name: str, params: tuple, rng=None) -> float:
        """
        Pure-Python approximation of scipy distribution sampling.
        Avoids a scipy dependency at runtime.

        Supported: lognorm, gamma, weibull_min, expon, norm

        ``rng`` defaults to ``self._rng`` — callers sampling case/offer
        attributes (petri_process.py, "rules" mode) intentionally keep using
        the shared RNG; only duration/branching draws go through CRN
        (see module docstring's CRN scope note).
        """
        rng = rng if rng is not None else self._rng
        if dist_name == "lognorm":
            s, loc, scale = params
            # X = loc + scale * exp(s * Z), Z ~ N(0,1)
            z = rng.gauss(0, 1)
            return loc + scale * math.exp(s * z)

        elif dist_name == "gamma":
            # params: (a, loc, scale)  X = loc + scale * Gamma(a)
            a, loc, scale = params
            return loc + scale * rng.gammavariate(a, 1.0)

        elif dist_name == "weibull_min":
            # params: (c, loc, scale)  X = loc + scale * Weibull(c)
            c, loc, scale = params
            return loc + scale * rng.weibullvariate(1.0, c)

        elif dist_name == "expon":
            loc, scale = params[-2], params[-1]
            return loc + rng.expovariate(1.0 / scale)

        elif dist_name == "norm":
            loc, scale = params[-2], params[-1]
            return rng.gauss(loc, scale)

        else:
            # Unknown distribution: exponential fallback
            scale = params[-1]
            return rng.expovariate(1.0 / scale)


ProcessComponent.HANDLES = {
    EventType.ACTIVITY_START:    ProcessComponent.on_activity_start,
    EventType.ACTIVITY_COMPLETE: ProcessComponent.on_activity_complete,
}

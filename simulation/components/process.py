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
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from ..core.events import SimEvent, EventType


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

    def __init__(self, seed: Optional[int] = 42):
        self._rng = random.Random(seed)
        # case_id -> {activity: repeat_count}
        self._repeat_counts: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        case_id = event.case_id

        # Sentinel: initialise case and start with A_Create Application
        if event.activity == "__PROCESS_START__":
            self._repeat_counts[case_id] = {}
            self._fire_start(engine, case_id, "A_Create Application")
            return

        # Normal start: sample duration, schedule ACTIVITY_COMPLETE
        activity = event.activity
        duration = self._sample_duration(activity)
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

        # Track repeats for loop-guard
        counts = self._repeat_counts.get(case_id, {})
        counts[activity] = counts.get(activity, 0) + 1
        self._repeat_counts[case_id] = counts

        # Decide termination
        if self._should_terminate(case_id, activity, counts):
            self._repeat_counts.pop(case_id, None)
            engine.schedule(SimEvent(
                timestamp=engine.now,
                priority=20,
                event_type=EventType.CASE_COMPLETE,
                case_id=case_id,
            ))
            return

        # Choose and schedule the next activity
        next_act = self._next_activity(activity)
        if next_act is None:
            # No outgoing edge defined — treat as terminal
            self._repeat_counts.pop(case_id, None)
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
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_START,
            case_id=case_id,
            activity=activity,
            resource=None,   # ResourceComponent will fill this (Section 1.8)
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

    def _next_activity(self, current: str) -> Optional[str]:
        """
        Sample the next activity using empirical branching probabilities.
        Returns None if current activity has no outgoing edges.
        """
        options = BRANCHING_PROBS.get(current)
        if not options:
            return None

        r = self._rng.random()
        cumulative = 0.0
        for next_act, prob in options:
            cumulative += prob
            if r <= cumulative:
                return next_act
        # Floating-point safety: return last option
        return options[-1][0]

    def _sample_duration(self, activity: str) -> float:
        """
        Sample processing time in seconds for an activity.
        Uses fitted scipy distributions where available, exponential fallback otherwise.
        """
        if activity in PROCESSING_TIME_PARAMS:
            dist_name, params = PROCESSING_TIME_PARAMS[activity]
            return max(1.0, self._sample_scipy_like(dist_name, params))

        mean = FALLBACK_MEAN_DURATIONS.get(activity, 600.0)
        return max(1.0, self._rng.expovariate(1.0 / mean))

    def _sample_scipy_like(self, dist_name: str, params: tuple) -> float:
        """
        Pure-Python approximation of scipy distribution sampling.
        Avoids a scipy dependency at runtime.

        Supported: lognorm, gamma, weibull_min, expon, norm
        """
        if dist_name == "lognorm":
            s, loc, scale = params
            # X = loc + scale * exp(s * Z), Z ~ N(0,1)
            z = self._rng.gauss(0, 1)
            return loc + scale * math.exp(s * z)

        elif dist_name == "gamma":
            # params: (a, loc, scale)  X = loc + scale * Gamma(a)
            a, loc, scale = params
            return loc + scale * self._rng.gammavariate(a, 1.0)

        elif dist_name == "weibull_min":
            # params: (c, loc, scale)  X = loc + scale * Weibull(c)
            c, loc, scale = params
            return loc + scale * self._rng.weibullvariate(1.0, c)

        elif dist_name == "expon":
            loc, scale = params[-2], params[-1]
            return loc + self._rng.expovariate(1.0 / scale)

        elif dist_name == "norm":
            loc, scale = params[-2], params[-1]
            return self._rng.gauss(loc, scale)

        else:
            # Unknown distribution: exponential fallback
            scale = params[-1]
            return self._rng.expovariate(1.0 / scale)


ProcessComponent.HANDLES = {
    EventType.ACTIVITY_START:    ProcessComponent.on_activity_start,
    EventType.ACTIVITY_COMPLETE: ProcessComponent.on_activity_complete,
}

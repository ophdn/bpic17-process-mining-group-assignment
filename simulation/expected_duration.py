"""
expected_duration.py — shared processing-time point-estimate API
===================================================================
Answers "how long will this allocation hold (activity, resource), on average"
without running a stochastic simulation step. In legacy mode that is the fitted
elapsed activity duration; in active mode it is the **next active session** up
to release on suspend/complete. Currently consumed by k-Batching's
assignment cost function (Optimization 1.1 / Final Task 1,
components/resource.py); Park & Song (D1) needs the same estimate for its
next-task prediction, so this is the shared `expected_duration()` API the
roadmap asks Mario/Daniel to build together rather than each hand-rolling
one (docs/ROADMAP.md, "Daniel <-> Mario" interface note).

Two estimate tiers, weakest-to-strongest:

1. `distribution_mean_seconds(activity)` — the analytic mean of the selected
   fitted scipy distribution. Legacy uses `PROCESSING_TIME_PARAMS`; active uses
   `LifecycleParameters.processing_times`. No context needed; always available.
2. `ExpectedDurationModel.expected_duration(activity, resource, context)`
   — the trained GBR point model (Section 1.3 Basic option 2 / `mode=
   "ml_model"` in process.py), given whatever context the caller has.
   Falls back to tier 1 when the trained artifact isn't present.

Context-completeness caveat: the trained model's full feature vector
needs day_of_week / hour_of_day / case_position / case_age_seconds /
previous_activity (see process.py::_build_features). A caller with only
(activity, resource) — e.g. ResourceComponent's k-Batching flush, which
doesn't track full per-case history the way ProcessComponent does — gets
sensible defaults for the missing features (see DEFAULT_CONTEXT) rather
than a crash. This is a known scope simplification: richer context would
need wiring case history from ProcessComponent into ResourceComponent,
left as future work.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np

from .components.process import FALLBACK_MEAN_DURATIONS, PROCESSING_TIME_PARAMS

# Best-effort defaults for context the caller doesn't have.
DEFAULT_CONTEXT = {
    "previous_activity": None,   # -> the model's "no previous activity" sentinel
    "day_of_week": 0,            # Monday; a fixed reference point, not a real guess
    "hour_of_day": 12,           # midday; ditto
    "case_position": 0,
    "case_age_seconds": 0.0,
}


def _lognorm_mean(params: tuple) -> float:
    s, loc, scale = params
    return loc + scale * math.exp(0.5 * s ** 2)


def _gamma_mean(params: tuple) -> float:
    a, loc, scale = params
    return loc + scale * a


def _weibull_min_mean(params: tuple) -> float:
    c, loc, scale = params
    return loc + scale * math.gamma(1.0 + 1.0 / c)


_ANALYTIC_MEANS = {
    "lognorm": _lognorm_mean,
    "gamma": _gamma_mean,
    "weibull_min": _weibull_min_mean,
}


def distribution_mean_seconds(
    activity: str,
    processing_times: Optional[Mapping[str, tuple]] = None,
) -> float:
    """Expected duration with NO context: the fitted distribution's
    analytic mean, or the fallback constant. Always available (Section
    1.3 Basic data), used when the ML artifact is absent or fails to load.
    """
    table = PROCESSING_TIME_PARAMS if processing_times is None else processing_times
    if activity in table:
        dist_name, params = table[activity]
        fn = _ANALYTIC_MEANS.get(dist_name)
        if fn is not None:
            try:
                return max(1.0, float(fn(params)))
            except (ValueError, OverflowError):
                pass
    return FALLBACK_MEAN_DURATIONS.get(activity, 600.0)


class ExpectedDurationModel:
    """Lazily loads the trained GBR point-estimate artifact
    (train_processing_time_model.py's output — the same one
    ProcessComponent(mode="ml_model") uses) and predicts a point duration
    for (activity, resource, context). Falls back to
    distribution_mean_seconds() if the artifact is missing, unreadable, or
    was never trained in this environment — callers never need to check
    availability themselves.
    """

    def __init__(self, model_path: Optional[str] = None, *,
                 lifecycle_mode: str = "legacy", lifecycle_params=None):
        if lifecycle_mode not in ("legacy", "active"):
            raise ValueError(f"lifecycle_mode must be legacy|active, got {lifecycle_mode!r}")
        if lifecycle_mode == "active" and lifecycle_params is None:
            raise ValueError("active expected-duration estimation requires lifecycle_params")
        self._model_path = model_path
        self._lifecycle_mode = lifecycle_mode
        self._processing_times = (
            lifecycle_params.processing_times if lifecycle_mode == "active" else None
        )
        self._artifact: Optional[dict] = None
        self._unavailable = False

    def _ensure(self) -> None:
        if self._artifact is not None or self._unavailable:
            return
        if not self._model_path:
            self._unavailable = True
            return
        try:
            import joblib
            self._artifact = joblib.load(self._model_path)
        except (FileNotFoundError, OSError):
            # An artifact may legitimately be absent in a distribution-only
            # checkout; fall back to the mode-selected analytic mean.
            self._unavailable = True
            return

        expected = (
            "active_session_seconds" if self._lifecycle_mode == "active"
            else "elapsed_start_complete_seconds"
        )
        target = self._artifact.get("target")
        schema = self._artifact.get("lifecycle_schema")
        if self._lifecycle_mode == "active" and (
            target != expected or schema != "active_v1"
        ):
            raise ValueError(
                f"active expected-duration model needs target={expected!r}, "
                f"schema='active_v1'; got target={target!r}, schema={schema!r}"
            )
        if self._lifecycle_mode == "legacy" and target not in (None, expected):
            raise ValueError(
                f"legacy expected-duration model cannot use target={target!r}"
            )

    def expected_duration(
        self, activity: str, resource: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> float:
        self._ensure()
        if self._unavailable:
            return distribution_mean_seconds(activity, self._processing_times)

        art = self._artifact
        model = art["model"]
        encoders = art["encoders"]
        feature_names = art["feature_names"]
        sentinels = art.get("sentinels", {})
        unknown = sentinels.get("unknown", "__UNKNOWN__")
        no_prev = sentinels.get("no_prev", "__START__")

        ctx = {**DEFAULT_CONTEXT, **(context or {})}

        def encode(name: str, value, fallback: str) -> int:
            value = str(value) if value is not None else fallback
            encoder = encoders[name]
            if value not in set(encoder.classes_):
                value = fallback
            return int(encoder.transform([value])[0])

        values = {
            "activity_enc": encode("activity", activity, unknown),
            "resource_enc": encode("resource", resource, unknown),
            "previous_activity_enc": encode(
                "previous_activity", ctx["previous_activity"], no_prev),
            "day_of_week": ctx["day_of_week"],
            "hour_of_day": ctx["hour_of_day"],
            "case_position": ctx["case_position"],
            "case_age_seconds": ctx["case_age_seconds"],
            "n_previous_activities": ctx["case_position"],
        }
        try:
            features = [float(values[name]) for name in feature_names]
        except KeyError:
            # Artifact expects a feature this context can't supply -- fall
            # back rather than guess.
            return distribution_mean_seconds(activity, self._processing_times)

        pred_log = model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0]
        return max(1.0, float(np.expm1(pred_log)))

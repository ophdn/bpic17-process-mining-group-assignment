"""
policies_advanced.py — shared pieces for the Part II advanced policies
=======================================================================
Currently: the next-activity predictor Park & Song's prediction-based
allocation (D1) plans with.

Design decision (roadmap D1 row sanctions this explicitly): Park & Song
2019 predict the next task with an LSTM; we substitute the argmax of the
process's mined branching model. In ``visit`` mode this uses the same
activity-visit table as the advanced simulator and falls back to the global
``BRANCHING_PROBS`` table when that bucket is sparse. The Petri-net simulator
may additionally use decision-point-specific probabilities, which cannot be
reconstructed from an activity alone. Reasons, for the report: (i) the tables ARE our
fitted next-activity model, mined from the full log, so the predictor
and the simulated ground truth share a vocabulary by construction;
(ii) an LSTM would have to be trained on simulated traces to be
consistent, adding a training loop for no methodological gain at this
scope; (iii) the policy's contribution — strategic idling on predicted
work — is orthogonal to which predictor supplies the prediction.

The predictor is deterministic (argmax, ties broken by table order) and
consumes no RNG, so enabling D1 cannot perturb any other component's
random stream (CRN-safe).
"""

import json
from pathlib import Path
from typing import Dict, Optional

from .components.process import BRANCHING_PROBS, TERMINAL_ACTIVITIES

_INPUTS_PATH = Path(__file__).resolve().parents[1] / "simulation_inputs.json"
_VISIT_BUCKET_MAX = 3


class NextActivityPredictor:
    """Predict a case's next activity from its current one.

    ``predict(activity, visit=...)`` returns ``(successor, probability)`` — the most
    probable successor from the mined branching model and its branching
    probability (the caller uses 1/p as an uncertainty penalty on the
    phantom's assignment cost) — or None when the case is (probably)
    done: terminal activities and activities without outgoing edges
    predict nothing, so no phantom work item is planned for them.
    """

    def __init__(self, branching_mode: str = "probs"):
        if branching_mode not in {"probs", "visit", "rules"}:
            raise ValueError(f"unknown branching mode {branching_mode!r}")
        self.branching_mode = branching_mode
        self._argmax: Dict[str, Optional[tuple]] = {}
        for activity, options in BRANCHING_PROBS.items():
            self._argmax[activity] = (
                max(options, key=lambda kv: kv[1]) if options else None)
        self._by_visit: Dict[str, dict] = {}
        if branching_mode in {"visit", "rules"}:
            try:
                with _INPUTS_PATH.open(encoding="utf-8") as handle:
                    self._by_visit = json.load(handle).get(
                        "branching_probs_by_visit", {})
            except FileNotFoundError:
                pass

    def predict(self, activity: str, *, visit: int = 1) -> Optional[tuple]:
        if activity in TERMINAL_ACTIVITIES:
            return None
        if self._by_visit:
            by_visit = self._by_visit.get(activity, {})
            bucket = str(visit) if visit < _VISIT_BUCKET_MAX else f"{_VISIT_BUCKET_MAX}+"
            distribution = by_visit.get(bucket)
            if distribution:
                return max(distribution.items(), key=lambda item: item[1])
        return self._argmax.get(activity)

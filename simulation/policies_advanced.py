"""
policies_advanced.py — shared pieces for the Part II advanced policies
=======================================================================
Currently: the next-activity predictor Park & Song's prediction-based
allocation (D1) plans with.

Design decision (roadmap D1 row sanctions this explicitly): Park & Song
2019 predict the next task with an LSTM; we substitute the argmax of the
process's own mined branching model — the same next-activity
distributions the simulator branches with (``BRANCHING_PROBS``,
process.py). Reasons, for the report: (i) the branching tables ARE our
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

from typing import Dict, Optional

from .components.process import BRANCHING_PROBS, TERMINAL_ACTIVITIES


class NextActivityPredictor:
    """Predict a case's next activity from its current one.

    ``predict(activity)`` returns ``(successor, probability)`` — the most
    probable successor from the mined branching model and its branching
    probability (the caller uses 1/p as an uncertainty penalty on the
    phantom's assignment cost) — or None when the case is (probably)
    done: terminal activities and activities without outgoing edges
    predict nothing, so no phantom work item is planned for them.
    """

    def __init__(self):
        self._argmax: Dict[str, Optional[tuple]] = {}
        for activity, options in BRANCHING_PROBS.items():
            self._argmax[activity] = (
                max(options, key=lambda kv: kv[1]) if options else None)

    def predict(self, activity: str) -> Optional[tuple]:
        if activity in TERMINAL_ACTIVITIES:
            return None
        return self._argmax.get(activity)

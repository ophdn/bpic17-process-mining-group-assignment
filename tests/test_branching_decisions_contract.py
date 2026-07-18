from __future__ import annotations

from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.process import ProcessComponent


class _FixedRandom:
    def __init__(self, value: float):
        self._value = value

    def random(self):
        return self._value


class _BasicBranchingProcess(ProcessComponent):
    def __init__(self, draw: float):
        super().__init__(seed=7)
        self._draw = draw

    def _draw_rng(self, case_id, activity, kind, visit=1):
        return _FixedRandom(self._draw)


class _FakeTree:
    def __init__(self, proba):
        self._proba = proba

    def predict_proba(self, rows):
        return [self._proba]


class _FakeEncoder:
    def __init__(self, classes):
        self.classes_ = list(classes)

    def transform(self, values):
        mapping = {value: index for index, value in enumerate(self.classes_)}
        return [mapping[value] for value in values]


def test_basic_branching_uses_empirical_probabilities():
    process = _BasicBranchingProcess(draw=0.0)
    assert process._next_activity("c1", "A_Create Application") == "A_Submitted"

    process = _BasicBranchingProcess(draw=0.95)
    assert process._next_activity("c1", "A_Create Application") == "A_Concept"


def test_visit_conditioned_branching_uses_case_history_bucket():
    component = object.__new__(PetriNetProcessComponent)
    component._branching_by_visit = {
        "W_Validate application": {
            "1": {"A_Validating": 0.9},
            "2": {"O_Returned": 0.8},
            "3+": {"A_Denied": 0.7},
        }
    }
    component._repeat_counts = {"c1": {"W_Validate application": 2}}

    assert component._visit_conditioned_probs("c1", "W_Validate application") == {"O_Returned": 0.8}


def test_rules_mode_can_choose_branch_from_sampled_case_attributes():
    component = object.__new__(PetriNetProcessComponent)
    component._decision_rules_path = None
    component._decision_models = {
        ("A_Accepted", "A_Denied"): {
            "tree": _FakeTree([0.0, 1.0]),
            "classes": ["A_Accepted", "A_Denied"],
        }
    }
    component._decision_encoders = {
        "application_type": _FakeEncoder(["New credit", "__UNKNOWN__"]),
        "loan_goal": _FakeEncoder(["Car", "__UNKNOWN__"]),
    }
    component._decision_feature_names = [
        "application_type_enc", "loan_goal_enc", "requested_amount",
        "has_offer", "credit_score", "offered_amount",
        "number_of_terms", "monthly_cost", "first_withdrawal_amount",
    ]
    component._decision_unknown = "__UNKNOWN__"
    component._case_attrs = {
        "c1": {
            "application_type": "New credit",
            "loan_goal": "Car",
            "requested_amount": 10000.0,
            "has_offer": 1,
            "credit_score": 700.0,
            "offered_amount": 9000.0,
            "number_of_terms": 36.0,
            "monthly_cost": 250.0,
            "first_withdrawal_amount": 0.0,
        }
    }

    choice = component._rules_weighted_choice(
        "c1", ["A_Accepted", "A_Denied"], rng=_FixedRandom(0.5))
    assert choice == "A_Denied"
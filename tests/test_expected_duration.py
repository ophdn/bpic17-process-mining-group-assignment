"""Tests for the shared allocation-policy duration estimator."""

from simulation.expected_duration import ExpectedDurationModel


def test_context_free_predictions_are_cached(monkeypatch):
    durations = ExpectedDurationModel()
    calls = []

    def predict(activity, resource, context):
        calls.append((activity, resource, context))
        return 123.0

    monkeypatch.setattr(durations, "_expected_duration_uncached", predict)

    assert durations.expected_duration("W_Test", "User_1") == 123.0
    assert durations.expected_duration("W_Test", "User_1") == 123.0
    assert calls == [("W_Test", "User_1", None)]


def test_cache_key_includes_resource(monkeypatch):
    durations = ExpectedDurationModel()
    calls = []

    def predict(activity, resource, context):
        calls.append((activity, resource, context))
        return float(len(calls))

    monkeypatch.setattr(durations, "_expected_duration_uncached", predict)

    assert durations.expected_duration("W_Test", "User_1") == 1.0
    assert durations.expected_duration("W_Test", "User_2") == 2.0
    assert durations.expected_duration("W_Test", None) == 3.0
    assert len(calls) == 3


def test_context_dependent_predictions_are_not_cached(monkeypatch):
    durations = ExpectedDurationModel()
    calls = []

    def predict(activity, resource, context):
        calls.append((activity, resource, context))
        return float(len(calls))

    monkeypatch.setattr(durations, "_expected_duration_uncached", predict)
    context = {"hour_of_day": 9}

    assert durations.expected_duration("W_Test", "User_1", context) == 1.0
    assert durations.expected_duration("W_Test", "User_1", context) == 2.0
    assert len(calls) == 2

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from scripts.metrics import branching_divergence


def _row(case_id, activity, offset_seconds):
    return {
        "case:concept:name": case_id,
        "concept:name": activity,
        "time:timestamp": datetime(2016, 1, 1) + timedelta(seconds=offset_seconds),
    }


def test_activity_that_always_ends_the_case_is_not_reported_as_absent():
    """A_Cancelled fires on every case here but never has a within-case
    successor (mirrors enforce_terminal_outcomes) -- it must be flagged as
    always-terminal, not lumped in with activities that never occurred."""
    df = pd.DataFrame([
        _row("c1", "A_Concept", 0),
        _row("c1", "A_Cancelled", 10),
        _row("c2", "A_Concept", 0),
        _row("c2", "A_Cancelled", 10),
    ])
    reference = {
        "A_Concept": {"A_Cancelled": 1.0},
        "A_Cancelled": {"O_Cancelled": 0.99, "A_Denied": 0.01},
    }

    result = branching_divergence(df, reference)

    assert result["per_activity_tvd"]["A_Cancelled"] is None
    assert "A_Cancelled" in result["activities_always_terminal_in_run"]
    assert "A_Cancelled" not in result["activities_absent_in_run"]
    # A_Concept did occur and did have a successor -- normal TVD applies.
    assert result["per_activity_tvd"]["A_Concept"] == 0.0


def test_activity_with_zero_occurrences_is_reported_as_absent():
    df = pd.DataFrame([
        _row("c1", "A_Concept", 0),
        _row("c1", "A_Cancelled", 10),
    ])
    reference = {
        "O_Cancelled": {"A_Denied": 1.0},
    }

    result = branching_divergence(df, reference)

    assert result["per_activity_tvd"]["O_Cancelled"] is None
    assert "O_Cancelled" in result["activities_absent_in_run"]
    assert "O_Cancelled" not in result["activities_always_terminal_in_run"]


def test_activity_with_a_successor_gets_a_real_tvd_not_none():
    df = pd.DataFrame([
        _row("c1", "A_Concept", 0),
        _row("c1", "A_Submitted", 10),
    ])
    reference = {
        "A_Concept": {"A_Submitted": 1.0},
    }

    result = branching_divergence(df, reference)

    assert result["per_activity_tvd"]["A_Concept"] == 0.0
    assert result["activities_absent_in_run"] == []
    assert result["activities_always_terminal_in_run"] == []


def test_source_activity_not_in_model_is_reported_separately_not_as_absent():
    """An activity that isn't even a transition in the process model (e.g.
    O_Sent (online only), missing from bpic17_process.bpmn) can never be
    reached by any branching-probability fix -- that's a model-coverage
    gap, distinct from activities_absent_in_run (in the model, just never
    selected/reached in this run)."""
    df = pd.DataFrame([
        _row("c1", "A_Concept", 0),
        _row("c1", "A_Submitted", 10),
    ])
    reference = {
        "A_Concept": {"A_Submitted": 1.0},
        "O_Sent (online only)": {"A_Validating": 1.0},
    }
    modeled = {"A_Concept", "A_Submitted"}

    result = branching_divergence(df, reference, modeled_activities=modeled)

    assert result["per_activity_tvd"]["O_Sent (online only)"] is None
    assert "O_Sent (online only)" in result["activities_not_in_bpmn"]
    assert "O_Sent (online only)" not in result["activities_absent_in_run"]


def test_target_not_in_model_is_excluded_and_reference_renormalised():
    """O_Created's real distribution sends 10% to O_Sent (online only),
    which the model can't produce at all -- that shouldn't inflate
    O_Created's own TVD (a target-coverage gap, not miscalibration of the
    choice among what the model can actually do); the excluded mass is
    still reported, not silently dropped."""
    df = pd.DataFrame([
        _row("c1", "O_Created", 0),
        _row("c1", "O_Sent (mail and online)", 10),
        _row("c2", "O_Created", 0),
        _row("c2", "O_Sent (mail and online)", 10),
    ])
    reference = {
        "O_Created": {
            "O_Sent (mail and online)": 0.9,
            "O_Sent (online only)": 0.1,
        },
    }
    modeled = {"O_Created", "O_Sent (mail and online)"}

    result = branching_divergence(df, reference, modeled_activities=modeled)

    # Renormalised reference over modeled targets only: O_Sent (mail and
    # online) -> 1.0, matching the simulated distribution exactly.
    assert result["per_activity_tvd"]["O_Created"] == 0.0
    assert result["excluded_target_mass"]["O_Created"] == 0.1

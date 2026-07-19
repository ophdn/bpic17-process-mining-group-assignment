"""Regression tests for case-vs-time warm-up evaluation semantics."""

from datetime import timedelta

import pandas as pd
import pytest

from scripts import opt_metrics
from scripts.run_experiments import START_DATETIME, apply_warmup


def _session(case_id, work_item_id, start, complete):
    common = {
        "case:concept:name": case_id,
        "concept:name": "A",
        "org:resource": "r1",
        "work_item_id": work_item_id,
    }
    return [
        dict(common, **{"time:timestamp": start, "lifecycle:transition": "start"}),
        dict(common, **{"time:timestamp": complete, "lifecycle:transition": "complete"}),
    ]


def test_warmup_keeps_old_case_post_cutoff_work_in_resource_metrics():
    cutoff = START_DATETIME + timedelta(days=1)
    horizon_end = START_DATETIME + timedelta(days=2)
    rows = []
    rows += _session(
        "old", "w-old", cutoff - timedelta(hours=1), cutoff + timedelta(hours=1))
    rows += _session(
        "new", "w-new", cutoff + timedelta(hours=2), cutoff + timedelta(hours=3))
    full_df = pd.DataFrame(rows)
    meta = {
        "arrival_times": {
            "old": START_DATETIME,
            "new": cutoff + timedelta(hours=1),
        },
        "completed_case_ids": {"old", "new"},
        "availability_intervals": {"r1": [(START_DATETIME, horizon_end)]},
        "availability_seconds": {"r1": 2 * 86400.0},
        "evaluation_window": (START_DATETIME, horizon_end),
        "configuration": {},
    }

    case_df, warmed = apply_warmup(full_df, meta, warmup_days=1)
    assert set(case_df["case:concept:name"]) == {"new"}
    assert warmed["availability_seconds"]["r1"] == 86400.0

    metrics = opt_metrics.evaluate(
        case_df,
        arrival_times=warmed["arrival_times"],
        completed_case_ids=warmed["completed_case_ids"],
        availability_seconds=warmed["availability_seconds"],
        availability_intervals=warmed["availability_intervals"],
        resource_subset={"r1"},
        resource_df=full_df,
        evaluation_window=warmed["evaluation_window"],
    )
    assert metrics["cycle_time"]["n_cases"] == 1
    # One hour from the straddling old case plus one hour from the new case.
    assert metrics["occupation"]["avg_resource_occupation"] == pytest.approx(2 / 24)
    assert metrics["occupation"]["busy_seconds_outside_availability"] == 0.0


def test_warmup_must_leave_a_nonempty_evaluation_window():
    meta = {
        "arrival_times": {},
        "completed_case_ids": set(),
        "availability_intervals": {},
        "availability_seconds": {},
        "evaluation_window": (
            START_DATETIME, START_DATETIME + timedelta(days=2)),
        "configuration": {},
    }
    with pytest.raises(ValueError, match="smaller than --days"):
        apply_warmup(pd.DataFrame(), meta, warmup_days=2)


def test_cycle_time_uses_explicit_case_completion_after_last_activity():
    arrival = START_DATETIME
    last_activity = arrival + timedelta(days=2)
    case_complete = arrival + timedelta(days=9)
    df = pd.DataFrame(_session(
        "case-1", "work-1", arrival + timedelta(days=1), last_activity))

    result = opt_metrics.average_cycle_time(
        df,
        arrival_times={"case-1": arrival},
        completion_times={"case-1": case_complete},
    )

    assert result["avg_cycle_time_s"] == pytest.approx(9 * 86400.0)

"""
opt_metrics.py
==============
Part II (optimization) evaluation metrics, implemented 1:1 after lecture
deck 06 "Business Process Optimization", slide 21:

  1. average_cycle_time          — "Mean time to finish a process instance."
  2. average_resource_occupation — "Mean share the resources are working
                                    during their availabilities."
  3. resource_fairness           — "(Weighted) Resource Fairness: mean
                                    deviation from the average resource
                                    occupation."

Do NOT confuse the fairness *metric* (deviation from the AVERAGE occupation,
slide 21) with the fairness *objective* in the CP scheduling formulation
(deviation from the LONGEST-working resource, slide 18) — the latter is an
optimization objective inside Kunkler & Rinderle-Ma's approach, not an
evaluation metric.

Inputs follow the simulation's event-log convention (one row per
ACTIVITY_START / ACTIVITY_COMPLETE, pm4py column names). Two things need
care to stay true to the slide definitions:

* Cycle time must run from CASE ARRIVAL, not from the first activity start.
  The logger only records activity events, so pass `arrival_times`
  (case_id -> arrival timestamp) whenever queueing before the first
  activity exists (i.e. after the Section-1.6/A2 contention rework).
  Without it we fall back to the first logged event per case and say so.

* Occupation is defined RELATIVE TO THE AVAILABILITY WINDOWS. Pass
  `availability_seconds` (resource -> available seconds in the horizon)
  from the Section 1.6 availability/calendar model. Without it we fall
  back to the full log span for every resource, which systematically
  UNDERSTATES occupation — fine for policy-vs-policy comparison, wrong as
  an absolute number. The report says which mode was used.

Usage (smoke test):
    python scripts/opt_metrics.py [path/to/event_log.csv]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "event_log.csv"


# ---------------------------------------------------------------------
# Shared: pair start/complete rows into activity instances
# ---------------------------------------------------------------------

def paired_instances(df: pd.DataFrame) -> pd.DataFrame:
    """One row per executed activity instance: case, activity, resource,
    start, complete. Start #k pairs with complete #k per (case, activity),
    matching how the engine executes activities sequentially per case."""
    df = df.sort_values("time:timestamp")
    starts = df[df["lifecycle:transition"] == "start"].copy()
    completes = df[df["lifecycle:transition"] == "complete"].copy()

    starts["seq"] = starts.groupby(
        ["case:concept:name", "concept:name"]).cumcount()
    completes["seq"] = completes.groupby(
        ["case:concept:name", "concept:name"]).cumcount()

    inst = starts.merge(
        completes[["case:concept:name", "concept:name", "seq",
                   "time:timestamp", "org:resource"]],
        on=["case:concept:name", "concept:name", "seq"],
        suffixes=("", "_complete"),
        how="inner",
    ).rename(columns={
        "time:timestamp": "start",
        "time:timestamp_complete": "complete",
        "case:concept:name": "case_id",
        "concept:name": "activity",
    })
    # The engine's logger records ACTIVITY_START before the resource is
    # assigned, so the resource usually only appears on the complete row —
    # coalesce complete-row over start-row.
    inst["resource"] = inst["org:resource_complete"].fillna(inst["org:resource"])
    return inst[["case_id", "activity", "resource", "start", "complete"]]


# ---------------------------------------------------------------------
# 1. Average Cycle Time (slide 21)
# ---------------------------------------------------------------------

def average_cycle_time(
    df: pd.DataFrame,
    arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
) -> dict:
    """Mean time to finish a process instance, in seconds.

    End = last event of the case. Start = the case's arrival timestamp if
    `arrival_times` is given (correct per slide 21 once pre-first-activity
    queueing exists), else the first logged event (flagged in the result).
    """
    last = df.groupby("case:concept:name")["time:timestamp"].max()
    if arrival_times is not None:
        start = pd.Series({c: arrival_times[c] for c in last.index
                           if c in arrival_times})
        last = last.loc[start.index]
        basis = "case_arrival"
    else:
        start = df.groupby("case:concept:name")["time:timestamp"].min()
        basis = "first_event (pass arrival_times for the slide-21 definition)"

    cycle_s = (last - start).dt.total_seconds()
    return {
        "avg_cycle_time_s": float(cycle_s.mean()),
        "p95_cycle_time_s": float(cycle_s.quantile(0.95)),
        "n_cases": int(len(cycle_s)),
        "start_basis": basis,
    }


# ---------------------------------------------------------------------
# 2. Average Resource Occupation (slide 21)
# ---------------------------------------------------------------------

def resource_busy_seconds(df: pd.DataFrame) -> pd.Series:
    """Total busy seconds per resource (sum of start→complete durations of
    the instances it executed). Rows without an assigned resource are
    excluded. NOTE: with capacity > 1 a resource can run instances in
    parallel, so busy time can exceed wall time — occupation > 1 then
    signals exactly that modeling artefact."""
    inst = paired_instances(df)
    inst = inst[inst["resource"].notna() & (inst["resource"] != "")]
    busy = (inst["complete"] - inst["start"]).dt.total_seconds()
    return busy.groupby(inst["resource"]).sum()

def average_resource_occupation(
    df: pd.DataFrame,
    availability_seconds: Optional[Mapping[str, float]] = None,
) -> dict:
    """Mean share the resources are working during their availabilities.

    `availability_seconds`: resource -> seconds available in the evaluated
    horizon (from the Section 1.6 availability/calendar model). Fallback:
    full log span for every resource (understates absolute occupation).
    """
    busy = resource_busy_seconds(df)
    if availability_seconds is not None:
        avail = pd.Series(dict(availability_seconds), dtype=float)
        basis = "availability_windows"
    else:
        span = (df["time:timestamp"].max() - df["time:timestamp"].min()
                ).total_seconds()
        avail = pd.Series(span, index=busy.index, dtype=float)
        basis = "log_span (pass availability_seconds for the slide-21 definition)"

    # Resources that were available but never worked count with occupation 0.
    occupation = (busy.reindex(avail.index).fillna(0.0) / avail).to_dict()
    return {
        "avg_resource_occupation": float(np.mean(list(occupation.values()))),
        "per_resource": {r: round(v, 4) for r, v in sorted(occupation.items())},
        "availability_basis": basis,
    }


# ---------------------------------------------------------------------
# 3. (Weighted) Resource Fairness (slide 21)
# ---------------------------------------------------------------------

def resource_fairness(
    occupation: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> dict:
    """Mean deviation from the average resource occupation; 0 = perfectly
    fair, lower is better.

    Weighted variant: weight each resource's deviation (e.g. by its share
    of total availability), so a lopsided load on an almost-never-available
    resource counts less than the same deviation on a full-time one.
    """
    occ = pd.Series(dict(occupation), dtype=float)
    unweighted = float((occ - occ.mean()).abs().mean())

    result = {"resource_fairness": unweighted}
    if weights is not None:
        w = pd.Series(dict(weights), dtype=float).reindex(occ.index).fillna(0.0)
        if w.sum() > 0:
            w = w / w.sum()
            w_mean = float((occ * w).sum())
            result["weighted_resource_fairness"] = float(
                ((occ - w_mean).abs() * w).sum())
    return result


# ---------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------

def evaluate(
    df: pd.DataFrame,
    arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
    availability_seconds: Optional[Mapping[str, float]] = None,
    fairness_weights: Optional[Mapping[str, float]] = None,
) -> dict:
    """All three slide-21 metrics on one simulated event log."""
    occ = average_resource_occupation(df, availability_seconds)
    return {
        "cycle_time": average_cycle_time(df, arrival_times),
        "occupation": occ,
        "fairness": resource_fairness(occ["per_resource"], fairness_weights),
    }


def print_report(label: str, m: dict) -> None:
    ct, oc, fa = m["cycle_time"], m["occupation"], m["fairness"]
    print(f"\n=== {label} ({ct['n_cases']} cases) ===")
    print(f"  avg cycle time:          {ct['avg_cycle_time_s']:>14,.1f} s "
          f"({ct['avg_cycle_time_s']/86400:.2f} d)   [start basis: {ct['start_basis']}]")
    print(f"  p95 cycle time:          {ct['p95_cycle_time_s']:>14,.1f} s "
          f"({ct['p95_cycle_time_s']/86400:.2f} d)")
    print(f"  avg resource occupation: {oc['avg_resource_occupation']:>14.4f}   "
          f"[basis: {oc['availability_basis']}]")
    print(f"  resource fairness:       {fa['resource_fairness']:>14.4f} "
          f"(0 = perfectly fair)")
    if "weighted_resource_fairness" in fa:
        print(f"  weighted fairness:       {fa['weighted_resource_fairness']:>14.4f}")


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    df = pd.read_csv(path)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    print_report(path.name, evaluate(df))

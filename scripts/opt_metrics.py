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
from typing import Dict, Iterable, Mapping, Optional

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
    if "work_item_id" in df.columns and df["work_item_id"].notna().any():
        rows = []
        lifecycle = df.dropna(subset=["work_item_id"])
        for wid, grp in lifecycle.groupby("work_item_id", sort=False):
            opened = None
            for _, row in grp.sort_values("time:timestamp").iterrows():
                transition = row["lifecycle:transition"]
                if transition in ("start", "resume"):
                    opened = row
                elif transition in ("suspend", "complete") and opened is not None:
                    rows.append({
                        "case_id": opened["case:concept:name"],
                        "activity": opened["concept:name"],
                        "resource": opened["org:resource"],
                        "start": opened["time:timestamp"],
                        "complete": row["time:timestamp"],
                        "work_item_id": wid,
                    })
                    opened = None
        return pd.DataFrame(rows, columns=[
            "case_id", "activity", "resource", "start", "complete", "work_item_id"
        ])

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

    # Coerce both sides to datetime64 before subtracting. A Series built from
    # a dict of datetime.datetime (the arrival_times path) is inferred as
    # object-dtype by pandas in some cases (the inference is content-dependent
    # and flaky), and object - datetime64 yields an object result whose .dt
    # accessor raises — which crashed a whole multi-run experiment grid on the
    # first heavily-loaded advanced-model run. Explicit coercion is vectorized
    # and immune to the inference.
    start = pd.to_datetime(start)
    last = pd.to_datetime(last)
    if len(last) == 0:
        return {"avg_cycle_time_s": float("nan"), "p95_cycle_time_s": float("nan"),
                "n_cases": 0, "start_basis": basis}

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
    """Total busy seconds per resource (sum of paired running sessions).
    Legacy logs have one start→complete session per instance; active logs also
    pair resume→suspend/complete on work_item_id. Rows without a resource are
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
    resource_subset: Optional[Iterable[str]] = None,
) -> dict:
    """Mean share the resources are working during their availabilities.

    `availability_seconds`: resource -> seconds available in the evaluated
    horizon (from the Section 1.6 availability/calendar model). Fallback:
    full log span for every resource (understates absolute occupation).

    `resource_subset`: if given, occupation is reported only for these
    resources (the rest are dropped before the mean and the per-resource
    map). Use it to restrict occupation/fairness to real human staff --
    an always-on automated account (Johannes's Section-1.6 Decision 4,
    e.g. User_1) has a huge availability denominator and a tiny occupation
    ratio, which distorts a staffing metric it is not really part of.
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

    if resource_subset is not None:
        keep = set(resource_subset)
        avail = avail[avail.index.isin(keep)]

    # Resources that were available but never worked count with occupation 0.
    # A resource with no scheduled availability inside a short horizon has no
    # defined occupation ratio (0/0), so exclude it and report the coverage
    # instead of allowing NaN to poison the aggregate mean.
    zero_availability = sorted(avail.index[avail <= 0].astype(str))
    valid_avail = avail[avail > 0]
    occupation = (busy.reindex(valid_avail.index).fillna(0.0) / valid_avail).to_dict()
    return {
        "avg_resource_occupation": float(np.mean(list(occupation.values())))
            if occupation else float("nan"),
        "per_resource": {r: round(v, 4) for r, v in sorted(occupation.items())},
        "availability_basis": basis,
        "n_resources_evaluated": int(len(occupation)),
        "n_resources_zero_availability": int(len(zero_availability)),
        "zero_availability_resources": zero_availability,
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
# Custom domain metrics (Final Task 4: self-designed, justified below)
# ---------------------------------------------------------------------
# The slide-21 trio (cycle time, occupation, fairness) is process-agnostic --
# it would look the same for any BPO process. These four are BPIC-17-loan-
# specific or expose failure modes the trio can't see on its own:
#
#   - time_to_first_offer / time_to_decision: customer-facing KPIs of a
#     loan process (how long until the applicant hears something), straight
#     from the A_/O_ milestone events already in the log.
#   - handover_rate: case continuity -- whether successive steps of one case
#     stay with the same person. This is useful to customers, but it is not a
#     direct measure of Piled Execution's same-activity batching mechanism.
#   - resource_activity_switch_rate: direct context-switching measure for
#     Piled Execution -- how often a resource changes activity between two
#     consecutive sessions.
#   - rolling_workload_balance: resource_fairness (slide 21) is a single
#     number over the WHOLE horizon, so it can't tell "fair on average" from
#     "fair on average because bursty overload cancels out with bursty
#     idleness" -- this looks at fairness in daily windows instead.

def _milestone_times(
    df: pd.DataFrame, activities, arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
) -> dict:
    """Shared helper: time from case arrival to the first COMPLETE event of
    any activity in *activities*, per case that reaches one. Falls back to
    first logged event as the start basis if arrival_times isn't given
    (same convention as average_cycle_time)."""
    hits = df[(df["lifecycle:transition"] == "complete")
              & (df["concept:name"].isin(activities))]
    first_hit = hits.groupby("case:concept:name")["time:timestamp"].min()

    # No case reaches the milestone -- e.g. the only resource permitted for
    # it was removed in a leave-N-out ("fire employees") run. The subtraction
    # below would be on an empty, non-datetimelike Series and blow up on .dt,
    # so short-circuit to an all-undefined result.
    if first_hit.empty:
        return {
            "mean_s": float("nan"), "p95_s": float("nan"),
            "n_cases_reaching_it": 0,
            "n_cases_total": int(df["case:concept:name"].nunique()),
            "start_basis": "case_arrival" if arrival_times is not None else "first_event",
        }

    if arrival_times is not None:
        start = pd.Series({c: arrival_times[c] for c in first_hit.index if c in arrival_times})
        first_hit = first_hit.loc[start.index]
        basis = "case_arrival"
    else:
        start = df.groupby("case:concept:name")["time:timestamp"].min()
        start = start.loc[first_hit.index]
        basis = "first_event (pass arrival_times for the slide-21-consistent definition)"

    # Explicit datetime coercion — see average_cycle_time for why (flaky
    # object-vs-datetime64 inference on dict-built Series breaks .dt).
    first_hit = pd.to_datetime(first_hit)
    start = pd.to_datetime(start)
    elapsed_s = (first_hit - start).dt.total_seconds()
    return {
        "mean_s": float(elapsed_s.mean()) if len(elapsed_s) else float("nan"),
        "p95_s": float(elapsed_s.quantile(0.95)) if len(elapsed_s) else float("nan"),
        "n_cases_reaching_it": int(len(elapsed_s)),
        "n_cases_total": int(df["case:concept:name"].nunique()),
        "start_basis": basis,
    }


def time_to_first_offer(
    df: pd.DataFrame, arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
) -> dict:
    """Arrival -> first O_Create Offer completion. Customer-facing: "how
    long before the applicant gets an offer at all" -- undefined (excluded
    from the mean) for cases that never reach an offer."""
    return _milestone_times(df, {"O_Create Offer"}, arrival_times)


def time_to_decision(
    df: pd.DataFrame, arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
) -> dict:
    """Arrival -> first terminal outcome (A_Pending / A_Denied /
    A_Cancelled). Customer-facing: "how long before the applicant knows
    where they stand", regardless of which outcome."""
    return _milestone_times(df, {"A_Pending", "A_Denied", "A_Cancelled"}, arrival_times)


def handover_rate(df: pd.DataFrame) -> dict:
    """Share of consecutive same-case activity steps that switch resource
    (lower = more work stays with the same person -- familiarity /
    continuity; higher = more context-switching / handover overhead).

    Computed on paired_instances ordered by start time within each case;
    steps with no assigned resource (unpermitted activities) are dropped
    from the comparison since "switch" is undefined against no one.
    """
    inst = paired_instances(df).sort_values(["case_id", "start"])
    inst = inst[inst["resource"].notna() & (inst["resource"] != "")]

    same_case = inst["case_id"] == inst["case_id"].shift(1)
    same_resource = inst["resource"] == inst["resource"].shift(1)
    consecutive = same_case
    handovers = consecutive & ~same_resource

    n_consecutive = int(consecutive.sum())
    return {
        "handover_rate": (float(handovers.sum()) / n_consecutive) if n_consecutive else float("nan"),
        "n_consecutive_steps": n_consecutive,
    }


def resource_activity_switch_rate(df: pd.DataFrame) -> dict:
    """Share of consecutive sessions handled by a resource whose activity changes.

    This directly evaluates Piled Execution's mechanism: keeping a resource on
    the same activity should reduce between-session setup and context switching.
    The case-level handover metric answers a different question and should not
    be used as evidence that activity piling works.
    """
    inst = paired_instances(df).sort_values(["resource", "start", "complete"])
    inst = inst[inst["resource"].notna() & (inst["resource"] != "")]

    same_resource = inst["resource"] == inst["resource"].shift(1)
    changed_activity = inst["activity"] != inst["activity"].shift(1)
    n_transitions = int(same_resource.sum())
    n_switches = int((same_resource & changed_activity).sum())
    return {
        "activity_switch_rate": (
            float(n_switches) / n_transitions if n_transitions else float("nan")),
        "n_resource_transitions": n_transitions,
        "n_activity_switches": n_switches,
    }


def rolling_workload_balance(
    df: pd.DataFrame, window: str = "1D",
    resource_subset: Optional[Iterable[str]] = None,
) -> dict:
    """Std of per-resource occupation WITHIN each rolling window, averaged
    across windows -- catches "fair on average, unfair in bursts", which
    resource_fairness (a single whole-horizon number) cannot: a resource
    idle for two weeks then overloaded for one still averages out fine over
    the whole run, but shows up here as a high per-window std on the
    overloaded weeks.

    Simplification: each activity instance is attributed whole to the
    calendar window containing its START time (no splitting instances that
    straddle a window boundary) -- window-level occupation is start-of-
    activity busy-share, not availability-normalised like the slide-21
    metric (no calendar windowing at this granularity); document this if
    citing the absolute occupation numbers, the std comparison across
    windows is still meaningful.
    """
    inst = paired_instances(df)
    inst = inst[inst["resource"].notna() & (inst["resource"] != "")]
    if resource_subset is not None:
        inst = inst[inst["resource"].isin(set(resource_subset))]
    if inst.empty:
        return {"mean_window_std": float("nan"), "n_windows": 0}

    inst = inst.copy()
    inst["duration_s"] = (inst["complete"] - inst["start"]).dt.total_seconds()
    inst["window"] = inst["start"].dt.floor(window)

    per_window = inst.groupby(["window", "resource"])["duration_s"].sum().unstack(fill_value=0.0)
    if resource_subset is not None:
        # Include staff who did no work in a window, or in the entire run, as
        # zeros. Omitting them understates imbalance and makes the metric's
        # resource population vary between policies.
        per_window = per_window.reindex(columns=sorted(set(resource_subset)), fill_value=0.0)
    window_seconds = pd.Timedelta(window).total_seconds()
    occupation = per_window / window_seconds

    window_std = occupation.std(axis=1, ddof=0)
    return {
        "mean_window_std": float(window_std.mean()),
        "max_window_std": float(window_std.max()),
        "n_windows": int(len(window_std)),
        "window": window,
    }


# ---------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------

def evaluate(
    df: pd.DataFrame,
    arrival_times: Optional[Mapping[str, pd.Timestamp]] = None,
    availability_seconds: Optional[Mapping[str, float]] = None,
    fairness_weights: Optional[Mapping[str, float]] = None,
    completed_case_ids=None,
    resource_subset: Optional[Iterable[str]] = None,
) -> dict:
    """All three slide-21 metrics on one simulated event log.

    `completed_case_ids`: ids of cases that finished naturally (from
    output/completed_cases.txt, written by simulation.main). Cycle time is
    only defined on FINISHED instances (slide 21: "mean time to finish a
    process instance") — horizon-truncated cases would bias it downwards,
    so always pass this for simulated logs. Busy time for the occupation
    metric intentionally keeps all cases: truncated cases did occupy
    resources during the horizon.

    `resource_subset`: restrict the RESOURCE-centric metrics (occupation,
    fairness, rolling workload balance) to these resources -- pass the real
    human staff to exclude an always-on automated account (Johannes's
    Section-1.6 Decision 4) from staffing metrics it distorts. The
    process-level metrics (cycle time, completions, milestones, handover)
    are unaffected: the automation genuinely does that work in the process.
    """
    n_total = df["case:concept:name"].nunique()
    df_completed = df
    if completed_case_ids is not None:
        df_completed = df[df["case:concept:name"].isin(set(completed_case_ids))]

    occ = average_resource_occupation(df, availability_seconds, resource_subset)
    return {
        "cycle_time": average_cycle_time(df_completed, arrival_times),
        "occupation": occ,
        "fairness": resource_fairness(occ["per_resource"], fairness_weights),
        "custom_metrics": {
            "time_to_first_offer": time_to_first_offer(df, arrival_times),
            "time_to_decision": time_to_decision(df, arrival_times),
            "handover_rate": handover_rate(df),
            "resource_activity_switch_rate": resource_activity_switch_rate(df),
            "rolling_workload_balance": rolling_workload_balance(
                df, resource_subset=resource_subset),
        },
        "case_filter": {
            "n_cases_in_log": int(n_total),
            "n_cases_completed": int(df_completed["case:concept:name"].nunique()),
            "filtered_to_completed": completed_case_ids is not None,
        },
    }


def print_report(label: str, m: dict) -> None:
    ct, oc, fa = m["cycle_time"], m["occupation"], m["fairness"]
    cf = m.get("case_filter", {})
    filt = ("completed cases only"
            if cf.get("filtered_to_completed")
            else "ALL cases — cycle time biased low, pass completed_case_ids!")
    print(f"\n=== {label} ({ct['n_cases']} of {cf.get('n_cases_in_log', '?')} cases; {filt}) ===")
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

    cm = m.get("custom_metrics")
    if cm:
        tfo, td = cm["time_to_first_offer"], cm["time_to_decision"]
        ho, wb = cm["handover_rate"], cm["rolling_workload_balance"]
        switches = cm.get("resource_activity_switch_rate", {})
        print(f"  time to first offer:     {tfo['mean_s']/86400:>14.2f} d "
              f"({tfo['n_cases_reaching_it']}/{tfo['n_cases_total']} cases reach it)")
        print(f"  time to decision:        {td['mean_s']/86400:>14.2f} d "
              f"({td['n_cases_reaching_it']}/{td['n_cases_total']} cases reach it)")
        print(f"  handover rate:           {ho['handover_rate']:>14.4f} "
              f"(share of steps that switch resource)")
        if switches:
            print(f"  activity-switch rate:    {switches['activity_switch_rate']:>14.4f} "
                  f"(share of a resource's sessions that change activity)")
        print(f"  rolling workload std:    {wb['mean_window_std']:>14.4f} "
              f"(mean per-{wb.get('window', '?')} std, max={wb.get('max_window_std', float('nan')):.4f})")


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    df = pd.read_csv(path)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="ISO8601")

    # simulation.main writes the naturally-completed case ids next to the
    # log — pick them up automatically so the CLI never reports the biased
    # all-cases cycle time by accident.
    completed_file = path.parent / "completed_cases.txt"
    completed = (set(completed_file.read_text(encoding="utf-8").splitlines())
                 if completed_file.exists() else None)
    if completed is None:
        print(f"WARNING: {completed_file} not found — evaluating ALL cases "
              f"(cycle time biased low). Re-run simulation.main to create it.")
    print_report(path.name, evaluate(df, completed_case_ids=completed))

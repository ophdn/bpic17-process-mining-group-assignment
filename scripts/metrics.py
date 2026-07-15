"""
metrics.py
==========
Reusable KPIs for judging whether a simulation configuration (process
model, duration model, resource model, ...) is a good approximation of
the real BPIC-17 process. Import these from any comparison/regression
script instead of re-deriving them.

Ground truth is `simulation_inputs.json` (produced once by
extract_log_info.py from the real BPIC-17 log) — no raw log needed at
comparison time, so this runs anywhere without the ~550MB .xes file.

Metric families, all inspired by the "second pass" validation method in
Rozinat et al., "Discovering Simulation Models" (see
docs/paper_insights_discovering_simulation_models.md):

  1. control_flow_fitness / control_flow_precision — does the simulated
     log only produce traces the reference Petri net allows, and does it
     use its full behaviour (not e.g. a trivial subset)?
  2. branching_divergence — do per-activity next-activity probabilities
     in the simulated log match the empirically observed ones?
  3. processing_time_errors — do simulated activity durations match the
     fitted distributions' mean/std?
  4. arrival_rate_error — does the simulated case arrival rate match?
  5. variant_overlap — do the simulated traces reproduce the real
     top-20 process variants?
  6. case_length_duration_errors — do case length (#events) and case
     duration (span) match?

Functions take either the raw event-log DataFrame (one row per
ACTIVITY_START/ACTIVITY_COMPLETE, pm4py column convention:
case:concept:name / concept:name / time:timestamp / lifecycle:transition,
already restricted to naturally-completed cases — see
compare_process_models.py) or its complete-only reduction
(`to_completed_events`) — see each function's docstring. Only
processing_time_errors needs the raw start+complete pairs; every other KPI
compares against reference values in simulation_inputs.json that were
themselves computed on lifecycle == 'complete' only (extract_log_info.py's
filter_to_complete), so mixing in raw start/schedule/suspend/resume rows on
the simulated side would bias the comparison. `evaluate()` handles this
routing for you — call that instead of the individual functions unless you
need just one KPI.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pm4py

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "simulation_inputs.json"


def load_reference(path: Path = DEFAULT_REFERENCE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _traces(df_complete: pd.DataFrame) -> pd.Series:
    """Case -> [activities in order], one entry per real activity
    occurrence (expects a complete-only DataFrame, see to_completed_events)."""
    return (
        df_complete.sort_values("time:timestamp")
        .groupby("case:concept:name")["concept:name"]
        .apply(list)
    )


def to_completed_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per real activity occurrence (lifecycle == 'complete')."""
    return df[df["lifecycle:transition"] == "complete"].copy()


# ---------------------------------------------------------------------
# 1. Control-flow
# ---------------------------------------------------------------------

def control_flow_fitness(df_complete: pd.DataFrame, net, im, fm) -> dict:
    return pm4py.fitness_token_based_replay(
        df_complete, net, im, fm,
        activity_key="concept:name", timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )


def control_flow_precision(df_complete: pd.DataFrame, net, im, fm) -> float:
    return pm4py.precision_token_based_replay(
        df_complete, net, im, fm,
        activity_key="concept:name", timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )


# ---------------------------------------------------------------------
# 2. Branching probabilities
# ---------------------------------------------------------------------

def branching_divergence(df_complete: pd.DataFrame, reference_branching: Dict[str, Dict[str, float]]) -> dict:
    """
    For each activity with outgoing edges in the reference, compute the
    empirical next-activity distribution in *df_complete* (complete-only,
    see to_completed_events) and the total variation distance (half the L1
    distance between the two probability vectors, range [0, 1], 0 =
    identical) to the reference distribution.
    """
    df2 = df_complete.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df2["next_activity"] = df2.groupby("case:concept:name")["concept:name"].shift(-1)
    df2 = df2.dropna(subset=["next_activity"])

    per_activity = {}
    for act, ref_dist in reference_branching.items():
        grp = df2[df2["concept:name"] == act]
        if len(grp) == 0:
            per_activity[act] = None  # activity never occurred in this run
            continue
        sim_counts = grp["next_activity"].value_counts()
        sim_dist = (sim_counts / sim_counts.sum()).to_dict()

        all_targets = set(ref_dist) | set(sim_dist)
        tvd = 0.5 * sum(abs(ref_dist.get(t, 0.0) - sim_dist.get(t, 0.0)) for t in all_targets)
        per_activity[act] = round(tvd, 4)

    observed = [v for v in per_activity.values() if v is not None]
    return {
        "per_activity_tvd": per_activity,
        "mean_tvd": round(sum(observed) / len(observed), 4) if observed else None,
        "activities_missing_in_run": [a for a, v in per_activity.items() if v is None],
    }


# ---------------------------------------------------------------------
# 3. Processing times
# ---------------------------------------------------------------------

def processing_time_errors(df: pd.DataFrame, reference_processing_times: Dict[str, dict]) -> dict:
    """
    Pairs each 'start' with the next 'complete' of the same activity
    within a case (cases never interleave two activities in this engine,
    so a simple ordered pairing is exact) and compares mean/std duration
    (seconds) against the reference's fitted mean_s/std_s.
    """
    per_activity = {}
    for case_id, grp in df.sort_values("time:timestamp").groupby("case:concept:name"):
        starts: Dict[str, list] = {}
        for _, row in grp.iterrows():
            act = row["concept:name"]
            if row["lifecycle:transition"] == "start":
                starts.setdefault(act, []).append(row["time:timestamp"])
            elif row["lifecycle:transition"] == "complete" and starts.get(act):
                start_ts = starts[act].pop(0)
                duration = (row["time:timestamp"] - start_ts).total_seconds()
                per_activity.setdefault(act, []).append(duration)

    result = {}
    rel_errors = []
    for act, ref in reference_processing_times.items():
        durations = per_activity.get(act)
        if not durations:
            result[act] = None
            continue
        s = pd.Series(durations)
        sim_mean, sim_std = s.mean(), s.std()
        ref_mean = ref["mean_s"]
        rel_err = abs(sim_mean - ref_mean) / ref_mean if ref_mean else None
        result[act] = {
            "sim_mean_s": round(sim_mean, 1), "ref_mean_s": ref_mean,
            "sim_std_s": round(sim_std, 1), "ref_std_s": ref["std_s"],
            "rel_err_mean": round(rel_err, 4) if rel_err is not None else None,
        }
        if rel_err is not None:
            rel_errors.append(rel_err)

    return {
        "per_activity": result,
        "mean_rel_err": round(sum(rel_errors) / len(rel_errors), 4) if rel_errors else None,
    }


# ---------------------------------------------------------------------
# 4. Arrival rate
# ---------------------------------------------------------------------

def arrival_rate_error(df_complete: pd.DataFrame, reference_arrival: dict) -> dict:
    first_events = df_complete.sort_values("time:timestamp").groupby("case:concept:name")["time:timestamp"].first().sort_values()
    inter_arrival = first_events.diff().dt.total_seconds().dropna()
    inter_arrival = inter_arrival[inter_arrival > 0]

    sim_mean = inter_arrival.mean()
    ref_mean = reference_arrival["mean_s"]
    rel_err = abs(sim_mean - ref_mean) / ref_mean if ref_mean else None
    return {
        "sim_mean_interarrival_s": round(sim_mean, 2),
        "ref_mean_interarrival_s": ref_mean,
        "rel_err": round(rel_err, 4) if rel_err is not None else None,
    }


# ---------------------------------------------------------------------
# 5. Process variants
# ---------------------------------------------------------------------

def variant_overlap(df_complete: pd.DataFrame, reference_top20: list) -> dict:
    """
    Coverage: share of the reference top-20 traffic (by pct) that is
    also a variant somewhere in the simulated log (order-exact match).
    """
    sim_traces = set(" → ".join(t) for t in _traces(df_complete))
    ref_variants = {v["trace"]: v["pct"] for v in reference_top20}

    covered_pct = sum(pct for trace, pct in ref_variants.items() if trace in sim_traces)
    n_covered = sum(1 for trace in ref_variants if trace in sim_traces)
    return {
        "ref_top20_variants_reproduced": n_covered,
        "ref_top20_traffic_coverage_pct": round(covered_pct, 2),
    }


# ---------------------------------------------------------------------
# 6. Case length / duration
# ---------------------------------------------------------------------

def case_length_duration_errors(df_complete: pd.DataFrame, reference_basic_stats: dict) -> dict:
    lengths = df_complete.groupby("case:concept:name").size()
    durations = df_complete.groupby("case:concept:name")["time:timestamp"].agg(
        lambda x: (x.max() - x.min()).total_seconds()
    )

    ref_len = reference_basic_stats["case_length"]["mean"]
    ref_dur = reference_basic_stats["case_duration_seconds"]["mean"]
    return {
        "sim_case_length_mean": round(lengths.mean(), 2),
        "ref_case_length_mean": ref_len,
        "case_length_rel_err": round(abs(lengths.mean() - ref_len) / ref_len, 4),
        "sim_case_duration_mean_s": round(durations.mean(), 1),
        "ref_case_duration_mean_s": ref_dur,
        "case_duration_rel_err": round(abs(durations.mean() - ref_dur) / ref_dur, 4),
    }


# ---------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------

def evaluate(
    df: pd.DataFrame,
    reference: dict,
    net=None, im=None, fm=None,
) -> dict:
    """
    Run every KPI above on *df* (a raw event-log DataFrame already
    restricted to naturally-completed cases). Control-flow metrics are
    skipped if no Petri net is given.

    Every KPI except processing_time_errors runs on the complete-only
    reduction of *df* — matching how the reference values in
    simulation_inputs.json were computed (extract_log_info.py's
    filter_to_complete) — so raw start/schedule/suspend/resume noise on
    either side never leaks into the comparison.
    """
    df_complete = to_completed_events(df)
    metrics = {
        "n_cases": df["case:concept:name"].nunique(),
        "branching": branching_divergence(df_complete, reference["branching_probs"]),
        "processing_times": processing_time_errors(df, reference["processing_times"]),
        "arrival_rate": arrival_rate_error(df_complete, reference["arrival_rate"]),
        "variants": variant_overlap(df_complete, reference["process_variants"]["top_20"]),
        "case_stats": case_length_duration_errors(df_complete, reference["basic_stats"]),
    }
    if net is not None:
        metrics["control_flow"] = {
            "fitness": control_flow_fitness(df_complete, net, im, fm),
            "precision": control_flow_precision(df_complete, net, im, fm),
        }
    return metrics


def print_report(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ({metrics['n_cases']} completed cases) ===")
    if "control_flow" in metrics:
        f = metrics["control_flow"]["fitness"]
        print(f"  control-flow fitness:        {f['average_trace_fitness']:.4f} "
              f"({f['percentage_of_fitting_traces']:.1f}% fully-fitting traces)")
        print(f"  control-flow precision:      {metrics['control_flow']['precision']:.4f}")
    b = metrics["branching"]
    print(f"  branching prob. mean TVD:    {b['mean_tvd']} (0 = identical, lower is better)")
    if b["activities_missing_in_run"]:
        print(f"    activities never reached:  {b['activities_missing_in_run']}")
    p = metrics["processing_times"]
    print(f"  processing time mean rel.err: {p['mean_rel_err']}")
    a = metrics["arrival_rate"]
    print(f"  arrival rate rel.err:         {a['rel_err']}  "
          f"(sim={a['sim_mean_interarrival_s']}s vs ref={a['ref_mean_interarrival_s']}s)")
    v = metrics["variants"]
    print(f"  top-20 real variants reproduced: {v['ref_top20_variants_reproduced']}/20 "
          f"(covers {v['ref_top20_traffic_coverage_pct']}% of real traffic)")
    c = metrics["case_stats"]
    print(f"  case length rel.err:         {c['case_length_rel_err']}  "
          f"(sim={c['sim_case_length_mean']} vs ref={c['ref_case_length_mean']})")
    print(f"  case duration rel.err:       {c['case_duration_rel_err']}  "
          f"(sim={c['sim_case_duration_mean_s']}s vs ref={c['ref_case_duration_mean_s']}s)")

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
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import pm4py

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "simulation_inputs.json"


def load_reference(path: Path = DEFAULT_REFERENCE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        reference = json.load(f)
    lifecycle = reference.get("lifecycle")
    if lifecycle:
        # Active comparisons use active-session distributions while retaining
        # the top-level arrival/branching/variant reference blocks.
        reference = dict(reference)
        reference["processing_times"] = lifecycle.get("processing_times", {})
        reference["_lifecycle"] = lifecycle
        reference["_lifecycle_mode"] = "active"
    else:
        reference["_lifecycle_mode"] = "legacy"
    return reference


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

def branching_divergence(
    df_complete: pd.DataFrame,
    reference_branching: Dict[str, Dict[str, float]],
    modeled_activities: Optional[set] = None,
) -> dict:
    """
    For each activity with outgoing edges in the reference, compute the
    empirical next-activity distribution in *df_complete* (complete-only,
    see to_completed_events) and the total variation distance (half the L1
    distance between the two probability vectors, range [0, 1], 0 =
    identical) to the reference distribution.

    Both distributions are conditioned on "this occurrence had a within-case
    successor" (extract_log_info.extract_branching uses the identical
    shift(-1)+dropna convention for the reference), so an activity that is
    *always* the last event of its case has an undefined conditional
    distribution -- not a TVD of 0 or 1, and not "never occurred". Concretely:
    under enforce_terminal_outcomes (petri_process.py), A_Pending/A_Denied/
    A_Cancelled always end the case immediately, so they can never have a
    recorded successor even though they fire on every completed case. That
    is reported separately (activities_always_terminal_in_run) from
    activities that genuinely never occurred at all in this run
    (activities_absent_in_run) -- e.g. a path the branching probabilities
    never select.

    *modeled_activities*: labels of every transition the process model (the
    Petri net converted from the BPMN) actually has. Some real activities in
    reference_branching aren't modeled at all (a BPMN coverage gap, not a
    branching-probability problem) -- e.g. O_Sent (online only) is 6.4% of
    real cases but isn't a task in bpic17_process.bpmn. Without this
    parameter those activities: (a) always show up as "absent" even though
    no branching-probability fix could ever reach them, and (b) silently
    inflate the TVD of *other*, modeled activities whenever the real log
    sends some probability mass to them as a next-activity target that the
    model can never produce. When given, source activities not in
    modeled_activities are reported separately
    (activities_not_in_bpmn, TVD left None -- not a calibration question),
    and for every other activity's target distribution, targets absent
    from modeled_activities are dropped from *both* sides and the
    reference redistribution is renormalised over just the modeled targets,
    so TVD measures branching calibration given what the model can actually
    produce. The excluded probability mass is reported per activity
    (pct_mass_on_unmodeled_targets) rather than silently discarded.
    """
    df2 = df_complete.sort_values(["case:concept:name", "time:timestamp"]).copy()
    total_counts = df2["concept:name"].value_counts()
    df2["next_activity"] = df2.groupby("case:concept:name")["concept:name"].shift(-1)
    df2 = df2.dropna(subset=["next_activity"])

    per_activity = {}
    activities_absent = []
    activities_always_terminal = []
    activities_not_in_bpmn = []
    excluded_target_mass = {}
    for act, ref_dist in reference_branching.items():
        if modeled_activities is not None and act not in modeled_activities:
            per_activity[act] = None
            activities_not_in_bpmn.append(act)
            continue
        if int(total_counts.get(act, 0)) == 0:
            per_activity[act] = None
            activities_absent.append(act)
            continue
        grp = df2[df2["concept:name"] == act]
        if len(grp) == 0:
            per_activity[act] = None
            activities_always_terminal.append(act)
            continue
        sim_counts = grp["next_activity"].value_counts()
        sim_dist = (sim_counts / sim_counts.sum()).to_dict()

        if modeled_activities is not None:
            excluded_mass = round(
                sum(p for t, p in ref_dist.items() if t not in modeled_activities), 4)
            if excluded_mass:
                excluded_target_mass[act] = excluded_mass
            ref_dist = {t: p for t, p in ref_dist.items() if t in modeled_activities}
            renorm = sum(ref_dist.values())
            if renorm:
                ref_dist = {t: p / renorm for t, p in ref_dist.items()}

        all_targets = set(ref_dist) | set(sim_dist)
        tvd = 0.5 * sum(abs(ref_dist.get(t, 0.0) - sim_dist.get(t, 0.0)) for t in all_targets)
        per_activity[act] = round(tvd, 4)

    observed = [v for v in per_activity.values() if v is not None]
    result = {
        "per_activity_tvd": per_activity,
        "mean_tvd": round(sum(observed) / len(observed), 4) if observed else None,
        "activities_absent_in_run": activities_absent,
        "activities_always_terminal_in_run": activities_always_terminal,
    }
    if modeled_activities is not None:
        result["activities_not_in_bpmn"] = activities_not_in_bpmn
        result["excluded_target_mass"] = excluded_target_mass
    return result


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
    active_schema = "work_item_id" in df.columns and df["work_item_id"].notna().any()
    if active_schema:
        # Pair every active session on the first-class work_item_id: start/resume
        # opens RUNNING; suspend/complete closes it. A suspended ate_abort has no
        # active interval to add. This remains correct with repeated activities.
        for wid, grp in df.dropna(subset=["work_item_id"]).sort_values(
                "time:timestamp").groupby("work_item_id", sort=False):
            opened = None
            activity = None
            for _, row in grp.iterrows():
                transition = row["lifecycle:transition"]
                if transition in ("start", "resume"):
                    opened = row["time:timestamp"]
                    activity = row["concept:name"]
                elif transition in ("suspend", "complete") and opened is not None:
                    duration = (row["time:timestamp"] - opened).total_seconds()
                    if duration >= 0:
                        per_activity.setdefault(activity, []).append(duration)
                    opened = None
    else:
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
        "target": "active_session_seconds" if active_schema else "elapsed_start_complete_seconds",
    }


def suspend_count_errors(df: pd.DataFrame, lifecycle_reference: dict) -> dict:
    """Compare suspend counts per work item against the extracted active
    reference. Keying is exclusively by work_item_id (§7)."""
    expected = lifecycle_reference.get("suspends_per_instance", {})
    if "work_item_id" not in df.columns:
        return {"per_activity": {}, "mean_abs_error": None}
    work = df[df["concept:name"].str.startswith("W_", na=False)].dropna(
        subset=["work_item_id"])
    counts = (
        work.assign(_s=work["lifecycle:transition"].eq("suspend").astype(int))
        .groupby(["concept:name", "work_item_id"])["_s"].sum()
    )
    per_activity = {}
    errors = []
    for activity, ref in expected.items():
        sim = counts.loc[activity] if activity in counts.index.get_level_values(0) else None
        if sim is None or len(sim) == 0:
            per_activity[activity] = None
            continue
        sim_mean = float(sim.mean())
        ref_mean = float(ref.get("mean", 0.0))
        error = abs(sim_mean - ref_mean)
        errors.append(error)
        per_activity[activity] = {
            "sim_mean": round(sim_mean, 4),
            "ref_mean": ref_mean,
            "abs_error": round(error, 4),
            "n_work_items": int(len(sim)),
        }
    return {
        "per_activity": per_activity,
        "mean_abs_error": round(float(np.mean(errors)), 4) if errors else None,
    }


# ---------------------------------------------------------------------
# 4. Arrival rate
# ---------------------------------------------------------------------

def arrival_rate_error(df_all: pd.DataFrame, reference_arrival: dict) -> dict:
    """Compare simulated vs. real inter-arrival time and daily arrival count.

    Takes the UNFILTERED event log (every case that arrived, not just the
    ones that finished within the horizon) — arrivals are a property of
    the ArrivalComponent alone, independent of anything downstream, so
    restricting to completed cases only would bias this metric by
    whatever biases completion (e.g. an overloaded resource pool at low
    completion rates preferentially "completes" short/easy cases). This
    was a real bug: the previous version took ``df_complete`` here, so at
    ~3% completion (an overloaded advanced-model run) the reported
    inter-arrival error reflected almost nothing about arrivals.
    """
    first_events = df_all.sort_values("time:timestamp").groupby("case:concept:name")["time:timestamp"].first().sort_values()
    inter_arrival = first_events.diff().dt.total_seconds().dropna()
    inter_arrival = inter_arrival[inter_arrival > 0]

    sim_mean = inter_arrival.mean()
    ref_mean = reference_arrival["mean_s"]
    rel_err = abs(sim_mean - ref_mean) / ref_mean if ref_mean else None

    result = {
        "sim_mean_interarrival_s": round(sim_mean, 2),
        "ref_mean_interarrival_s": ref_mean,
        "rel_err": round(rel_err, 4) if rel_err is not None else None,
    }

    daily_ref = reference_arrival.get("daily_arrivals")
    if daily_ref:
        daily_sim = first_events.dt.floor("D").value_counts()
        # Drop the first/last calendar day: partial horizons at either edge
        # would understate a real day's count and skew the mean down.
        if len(daily_sim) > 2:
            daily_sim = daily_sim.sort_index().iloc[1:-1]
        sim_daily_mean = float(daily_sim.mean()) if len(daily_sim) else None
        ref_daily_mean = daily_ref["mean"]
        daily_rel_err = (abs(sim_daily_mean - ref_daily_mean) / ref_daily_mean
                         if sim_daily_mean is not None and ref_daily_mean else None)
        result["sim_daily_arrivals_mean"] = (
            round(sim_daily_mean, 2) if sim_daily_mean is not None else None)
        result["ref_daily_arrivals_mean"] = ref_daily_mean
        result["daily_arrivals_rel_err"] = (
            round(daily_rel_err, 4) if daily_rel_err is not None else None)

    return result


def arrival_profile_error(df_all: pd.DataFrame, reference_arrival: dict) -> Optional[dict]:
    """Hour-of-day / day-of-week arrival SHAPE error (Section 1.2 Advanced).

    This is the metric that actually distinguishes the MDN (time-dependent)
    arrival model from the static parametric LogNormal: both can match the
    same MEAN inter-arrival time (arrival_rate_error above), but only a
    model that conditions on time-of-day/weekday can match the real shape
    (nights ~0.6 arrivals/h vs ~7.6/h in the 12-18h core; Monday ~3x Sunday
    -- see components/arrival_mdn.py's module docstring). MAE between
    normalized 24-bin hour-of-day and 7-bin day-of-week histograms.

    Returns None if the reference doesn't carry ``hod_profile``/
    ``dow_profile`` yet -- an older simulation_inputs.json, or
    extract_log_info.py hasn't been re-run against the real log with this
    field added (see docs/ROADMAP.md A4). Callers should skip this KPI
    rather than treat None as an error.
    """
    hod_ref = reference_arrival.get("hod_profile")
    dow_ref = reference_arrival.get("dow_profile")
    if hod_ref is None or dow_ref is None:
        return None

    first_events = (
        df_all.sort_values("time:timestamp")
        .groupby("case:concept:name")["time:timestamp"].first()
    )
    hod_counts = first_events.dt.hour.value_counts().reindex(range(24), fill_value=0)
    hod_sim = (hod_counts / hod_counts.sum()).to_numpy()
    dow_counts = first_events.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    dow_sim = (dow_counts / dow_counts.sum()).to_numpy()

    return {
        "hour_of_day_mae": round(float(np.mean(np.abs(hod_sim - np.asarray(hod_ref)))), 4),
        "day_of_week_mae": round(float(np.mean(np.abs(dow_sim - np.asarray(dow_ref)))), 4),
    }


# ---------------------------------------------------------------------
# 5. Process variants
# ---------------------------------------------------------------------

def variant_overlap(df_complete: pd.DataFrame, reference_top20: list) -> dict:
    """
    Coverage: share of the reference top-20 traffic (by pct) that is
    also a variant somewhere in the simulated log (order-exact match).

    Also reports a second, lenient coverage number that first strips any
    step whose activity never occurs anywhere in *df_complete* (the same
    "never fired in this run" condition as branching_divergence's
    activities_absent_in_run -- e.g. W_Validate application, which is
    reachable in the model but whose retry loop rarely lets a case finish
    within the simulation horizon, see docs/ROADMAP.md) from each reference
    trace before matching. A real variant that is an exact control-flow
    match except for such a step is a structural horizon/loop-exit gap,
    not a branching miss, and would otherwise count as a total non-match.
    Activities missing from the *model* entirely (not in the BPMN at all)
    are a different, already-tracked gap -- they still occur in
    sim_activities' complement but are not distinguished here since they
    can never appear in df_complete either way.
    """
    sim_traces = set(" → ".join(t) for t in _traces(df_complete))
    sim_activities = set(df_complete["concept:name"].unique())
    ref_variants = {v["trace"]: v["pct"] for v in reference_top20}

    covered_pct = sum(pct for trace, pct in ref_variants.items() if trace in sim_traces)
    n_covered = sum(1 for trace in ref_variants if trace in sim_traces)

    covered_pct_adj = 0.0
    n_covered_adj = 0
    ignored_activities = set()
    for trace, pct in ref_variants.items():
        if trace in sim_traces:
            covered_pct_adj += pct
            n_covered_adj += 1
            continue
        steps = trace.split(" → ")
        dropped = [s for s in steps if s not in sim_activities]
        if not dropped:
            continue
        reduced_trace = " → ".join(s for s in steps if s in sim_activities)
        if reduced_trace in sim_traces:
            covered_pct_adj += pct
            n_covered_adj += 1
            ignored_activities.update(dropped)

    return {
        "ref_top20_variants_reproduced": n_covered,
        "ref_top20_traffic_coverage_pct": round(covered_pct, 2),
        "ref_top20_variants_reproduced_ignoring_absent_activities": n_covered_adj,
        "ref_top20_traffic_coverage_pct_ignoring_absent_activities": round(covered_pct_adj, 2),
        "activities_ignored_for_variant_match": sorted(ignored_activities),
    }


# ---------------------------------------------------------------------
# 6. Case length / duration
# ---------------------------------------------------------------------

def case_length_duration_errors(
    df_complete: pd.DataFrame,
    reference_basic_stats: dict,
    case_duration_seconds: Optional[Mapping[str, float]] = None,
) -> dict:
    lengths = df_complete.groupby("case:concept:name").size()
    if case_duration_seconds is not None:
        present = set(df_complete["case:concept:name"].astype(str))
        durations = pd.Series({
            str(case_id): float(duration)
            for case_id, duration in case_duration_seconds.items()
            if str(case_id) in present
        })
    else:
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
    df_all: Optional[pd.DataFrame] = None,
    case_duration_seconds: Optional[Mapping[str, float]] = None,
) -> dict:
    """
    Run every KPI above on *df* (a raw event-log DataFrame already
    restricted to naturally-completed cases). Control-flow metrics are
    skipped if no Petri net is given.

    Every KPI except processing_time_errors and arrival_rate runs on the
    complete-only reduction of *df* — matching how the reference values in
    simulation_inputs.json were computed (extract_log_info.py's
    filter_to_complete) — so raw start/schedule/suspend/resume noise on
    either side never leaks into the comparison.

    *df_all*: the UNFILTERED log (every started case, not just completed
    ones) — arrival_rate_error needs this, since arrivals are independent
    of whatever biases completion and restricting to completed cases
    understates the arrival rate at low completion rates (see
    arrival_rate_error's docstring). Falls back to *df* if not given, which
    silently reproduces the old bias — always pass it for a simulated log
    where *df* has already been filtered to completed cases.
    """
    df_complete = to_completed_events(df)
    if df_all is None:
        df_all = df
    modeled_activities = (
        {t.label for t in net.transitions if t.label} if net is not None else None
    )
    metrics = {
        "n_cases": df["case:concept:name"].nunique(),
        "branching": branching_divergence(
            df_complete, reference["branching_probs"], modeled_activities),
        "processing_times": processing_time_errors(df, reference["processing_times"]),
        "arrival_rate": arrival_rate_error(df_all, reference["arrival_rate"]),
        "arrival_profile": arrival_profile_error(df_all, reference["arrival_rate"]),
        "variants": variant_overlap(df_complete, reference["process_variants"]["top_20"]),
        "case_stats": case_length_duration_errors(
            df_complete, reference["basic_stats"], case_duration_seconds),
    }
    if reference.get("_lifecycle_mode") == "active":
        metrics["lifecycle"] = {
            "suspend_counts": suspend_count_errors(df, reference["_lifecycle"]),
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
    if b.get("activities_not_in_bpmn"):
        print(f"    activities not in BPMN model (structural gap, not branching): "
              f"{b['activities_not_in_bpmn']}")
    if b["activities_absent_in_run"]:
        print(f"    activities never reached:  {b['activities_absent_in_run']}")
    if b["activities_always_terminal_in_run"]:
        print(f"    activities always case-final (no successor, TVD undefined): "
              f"{b['activities_always_terminal_in_run']}")
    if b.get("excluded_target_mass"):
        print(f"    real-branch mass excluded (target not in BPMN model), per source "
              f"activity: {b['excluded_target_mass']}")
    p = metrics["processing_times"]
    print(f"  processing time mean rel.err: {p['mean_rel_err']} ({p.get('target')})")
    if "lifecycle" in metrics:
        print(f"  suspend-count mean abs.err:  "
              f"{metrics['lifecycle']['suspend_counts']['mean_abs_error']}")
    a = metrics["arrival_rate"]
    print(f"  arrival rate rel.err:         {a['rel_err']}  "
          f"(sim={a['sim_mean_interarrival_s']}s vs ref={a['ref_mean_interarrival_s']}s)")
    if a.get("daily_arrivals_rel_err") is not None:
        print(f"  daily arrivals rel.err:       {a['daily_arrivals_rel_err']}  "
              f"(sim={a['sim_daily_arrivals_mean']}/day vs ref={a['ref_daily_arrivals_mean']}/day)")
    ap = metrics.get("arrival_profile")
    if ap is not None:
        print(f"  arrival profile MAE:         hour-of-day={ap['hour_of_day_mae']}  "
              f"day-of-week={ap['day_of_week_mae']}")
    v = metrics["variants"]
    print(f"  top-20 real variants reproduced: {v['ref_top20_variants_reproduced']}/20 "
          f"(covers {v['ref_top20_traffic_coverage_pct']}% of real traffic)")
    if v.get("activities_ignored_for_variant_match"):
        print(f"    ...ignoring steps never simulated in this run: "
              f"{v['ref_top20_variants_reproduced_ignoring_absent_activities']}/20 "
              f"(covers {v['ref_top20_traffic_coverage_pct_ignoring_absent_activities']}%); "
              f"ignored activities: {v['activities_ignored_for_variant_match']}")
    c = metrics["case_stats"]
    print(f"  case length rel.err:         {c['case_length_rel_err']}  "
          f"(sim={c['sim_case_length_mean']} vs ref={c['ref_case_length_mean']})")
    print(f"  case duration rel.err:       {c['case_duration_rel_err']}  "
          f"(sim={c['sim_case_duration_mean_s']}s vs ref={c['ref_case_duration_mean_s']}s)")

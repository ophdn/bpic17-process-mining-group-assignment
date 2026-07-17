"""
run_experiments.py — Phase B Experiment Runner (Optimization / Part II)
=========================================================================
Orchestrates N-seed, policy x scenario simulation experiments and reports
the lecture-slide-21 metrics (scripts/opt_metrics.py) with confidence
intervals and paired tests. This is the infrastructure every Part II
deliverable (simple policies, k-batching, both advanced policies, the
"fire two employees" management question) runs through — see
docs/ROADMAP.md Phase B.

Policy registry (extend as Part II lands more allocation strategies):
    random  -- R-RMA uniform random pick (Section 1.8 Basic baseline)
    piled   -- R-PE Piled Execution (same-activity batching on the wait queue)
(k-batching, R-RRA, R-SHQ land here as they're implemented.)

Scenarios:
    normal  -- as-is
    peak    -- arrivals scaled +30% (ArrivalComponent(scale_factor=1.3))
    outage  -- 20% of resources removed at build time, deterministic per seed
               (ResourceComponent(excluded_resources=...))

Common Random Numbers (crn=True, default here): branching and duration
draws are seeded per (case, activity, kind, visit) instead of coming from
one shared RNG in dispatch order, so two policies run under the same seed
see identical case trajectories up to the point their allocation timing
diverges them — required for the paired comparisons below to mean anything
(see process.py's module docstring, and output/piled_execution_eval.md for
what happens without it).

Warm-up: case-based (not time-based, see apply_warmup() docstring) --
exclude every case that arrived before --warmup-days. Pick the value by
first running with --report-wip to see the work-in-progress (open case
count) time series and choosing where it plateaus.

Usage:
    cd <repo-root>
    PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \\
        --policies random,piled --seeds 10 --days 30 --scenario normal

    # Diagnostic: print WIP-over-time to help pick --warmup-days
    PYTHONPATH=. .venv/bin/python scripts/run_experiments.py --report-wip

Output (in --out, default output/experiments/):
    results_<scenario>.csv        -- one row per (policy, seed)
    aggregate_<scenario>.csv      -- mean +/- 95% CI per policy per metric
    paired_tests_<scenario>.csv   -- paired t-test vs the baseline policy
"""

from __future__ import annotations

import argparse
import random as _random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import (
    DEFAULT_CAPACITY_ACTIVE, DEFAULT_CAPACITY_LEGACY,
    RESOURCE_PERMISSIONS, ResourceComponent, capacity_for_mode,
)
from simulation.components import permissions as perm_models
from simulation.components.case_attributes import CaseAttributeSampler
from simulation.components.lifecycle_params import LifecycleParameters
from simulation.main import CaseCompletionTracker
from analysis.availability import YearlyAvailability

import scripts.opt_metrics as opt_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
AVAILABILITY_MODEL_PATH = REPO_ROOT / "models" / "availability_model.json"
ORGMODEL_PATH = REPO_ROOT / "models" / "permissions_orgmodel.json"
OBSERVED_PERMS_PATH = REPO_ROOT / "models" / "permissions_observed.json"
CASE_ATTRIBUTES_PATH = REPO_ROOT / "models" / "case_attributes.json"
OUT_DEFAULT = REPO_ROOT / "output" / "experiments"
ACTIVE_INPUTS_PATH = REPO_ROOT / "simulation_inputs_active.json"
LEGACY_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model.joblib"
ACTIVE_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model_active.joblib"

# BPIC-17 starts 2016-01-01 -- must match simulation/main.py's anchor so
# weekday/hour-of-day features (MDN arrivals, calendar) align.
START_DATETIME = datetime(2016, 1, 1)

KNOWN_POLICIES = {"random", "piled"}
KNOWN_SCENARIOS = {"normal", "peak", "outage"}
_KBATCH_RE = re.compile(r"^kbatch(\d+)$")
OUTAGE_FRACTION = 0.20


# ---------------------------------------------------------------------
# Recording components
# ---------------------------------------------------------------------

class _ArrivalRecorder:
    """Records CASE_ARRIVAL sim-timestamps per case (opt_metrics needs
    arrival time, not first-activity-start time, for cycle time per
    slide 21)."""

    HANDLES = {EventType.CASE_ARRIVAL: None}

    def __init__(self):
        self.timestamps: Dict[str, float] = {}

    def on_arrival(self, engine, event: SimEvent) -> None:
        self.timestamps[event.case_id] = event.timestamp


_ArrivalRecorder.HANDLES = {EventType.CASE_ARRIVAL: _ArrivalRecorder.on_arrival}


class _CompletionRecorder:
    """Records CASE_COMPLETE sim-timestamps per case. Needed for a correct
    WIP (work-in-progress) time series: the event log only has ACTIVITY_*
    rows, so "last activity completed before t" is NOT the same as "case
    fully finished by t" -- a case can have an old completed activity while
    still queueing for its next one. report_wip() needs the real thing."""

    HANDLES = {EventType.CASE_COMPLETE: None}

    def __init__(self):
        self.timestamps: Dict[str, float] = {}

    def on_case_complete(self, engine, event: SimEvent) -> None:
        self.timestamps[event.case_id] = event.timestamp


_CompletionRecorder.HANDLES = {EventType.CASE_COMPLETE: _CompletionRecorder.on_case_complete}


# ---------------------------------------------------------------------
# Scenario / policy plumbing
# ---------------------------------------------------------------------

def scenario_arrival_kwargs(scenario: str) -> dict:
    if scenario == "peak":
        return {"scale_factor": 1.3}
    return {}


def load_permission_model(kind: str, seed: int):
    """Mirror simulation/main.py's Section-1.7 wiring: returns
    (permission_model_or_None, case_attribute_sampler_or_None).

    "orgmodel" (the team default since 1.7 landed) gates permissions on the
    case type, so cases need a CaseAttributeSampler; "observed" is the
    log-mined resource x activity matrix; "hardcoded" is the original
    top-20 map (ResourceComponent's built-in default when passed None).
    """
    if kind == "orgmodel":
        perms = perm_models.OrgModelPermissions.from_json(ORGMODEL_PATH)
        perms.self_check()
        case_attrs = CaseAttributeSampler.from_json(CASE_ATTRIBUTES_PATH, seed=seed)
        return perms, case_attrs
    if kind == "observed":
        return perm_models.StaticPermissions.from_json(OBSERVED_PERMS_PATH), None
    if kind == "hardcoded":
        return None, None
    raise ValueError(f"unknown permissions kind {kind!r} "
                     "(known: orgmodel, observed, hardcoded)")


def scenario_excluded_resources(scenario: str, seed: int, resource_pool) -> Optional[Set[str]]:
    """Deterministic 20% resource removal for the 'outage' scenario.
    Seeded per-run so different seeds see different (but reproducible)
    removed sets -- this doubles as infrastructure for the "fire two
    employees" management question (leave-N-out simulations).

    ``resource_pool``: the ACTIVE permission model's resource list — under
    the org model the pool is much larger than the hardcoded top-20 map, and
    an outage must remove real pool members, not names from a stale map.
    """
    if scenario != "outage":
        return None
    names = sorted(resource_pool)
    k = max(1, round(len(names) * OUTAGE_FRACTION))
    rng = _random.Random(seed)
    return set(rng.sample(names, k))


def parse_kbatch_policy(policy: str) -> Optional[int]:
    """'kbatch5' -> 5, 'kbatch20' -> 20, else None. Lets --policies list a
    k-sweep (kbatch1,kbatch2,kbatch5,kbatch10,kbatch20) without a fixed
    enum of every k value anyone might want to try."""
    m = _KBATCH_RE.match(policy)
    return int(m.group(1)) if m else None


def is_known_policy(policy: str) -> bool:
    return policy in KNOWN_POLICIES or parse_kbatch_policy(policy) is not None


def build_resource_component(
    policy: str, seed: int, calendar, excluded: Optional[Set[str]],
    permission_model=None, lifecycle_mode: str = "legacy", lifecycle_params=None,
    capacity: Optional[int] = None,
) -> ResourceComponent:
    if capacity is None:
        capacity = capacity_for_mode(lifecycle_mode)
    k = parse_kbatch_policy(policy)
    if k is not None:
        return ResourceComponent(
            capacity_per_resource=capacity,
            seed=seed,
            calendar=calendar,
            start_datetime=START_DATETIME,
            permissions=permission_model,
            excluded_resources=excluded,
            batching_k=k,
            duration_model_path=str(
                ACTIVE_MODEL_PATH if lifecycle_mode == "active" else LEGACY_MODEL_PATH),
            lifecycle_mode=lifecycle_mode,
            lifecycle_params=lifecycle_params,
        )
    if policy not in KNOWN_POLICIES:
        raise ValueError(
            f"unknown policy {policy!r} (known: {sorted(KNOWN_POLICIES)}, "
            f"or 'kbatchN' for k-Batching with k=N). Add new conditions "
            "here as Part II lands R-RRA / R-SHQ."
        )
    return ResourceComponent(
        capacity_per_resource=capacity,
        seed=seed,
        calendar=calendar,
        start_datetime=START_DATETIME,
        permissions=permission_model,
        piled=(policy == "piled"),
        excluded_resources=excluded,
    )


def availability_seconds_per_resource(
    calendar: Optional[YearlyAvailability], start_dt: datetime, horizon_days: int,
    resources,
) -> Dict[str, float]:
    """Seconds each resource is on-duty within [start_dt, start_dt+horizon).

    Mirrors ResourceComponent._is_on_shift's convention exactly: a resource
    absent from the calendar's weekly windows (e.g. automated accounts like
    User_1) is always on duty, not a fallback -- see resource.py.

    This is the denominator of the slide-21 occupation definition, so it has to
    mirror the roster too: counting window-hours on days a resource is not
    rostered on would divide busy time by a workforce that never showed up, and
    occupation would read low while looking perfectly plausible. `_works_today`
    is keyed by (seed, resource, date), so this independent scan agrees with the
    runtime's allocation decisions without sharing any state with them.
    """
    if calendar is None:
        secs = horizon_days * 86400.0
        return {r: secs for r in resources}

    out = {r: 0.0 for r in resources}
    for day in range(horizon_days):
        d = (start_dt + timedelta(days=day)).date()
        if d in calendar.holidays:
            continue
        dow = d.weekday()
        for r in resources:
            windows = calendar.weekly.windows.get(r)
            if windows is None:
                out[r] += 86400.0
                continue
            if d in calendar.vacations.get(r, ()):
                continue
            if not calendar._works_today(r, d):
                continue
            w = windows.get(dow)
            if w is None:
                continue
            out[r] += (w[1] - w[0]) * 3600.0
    return out


# ---------------------------------------------------------------------
# One simulation run
# ---------------------------------------------------------------------

def run_once(
    policy: str, seed: int, days: int, scenario: str, crn: bool,
    process_model: str, branching_mode: str, permissions: str = "orgmodel",
    excluded_override: Optional[Set[str]] = None,
    lifecycle_mode: str = "legacy",
    roster_seed: Optional[int] = None,
    capacity: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build and run one (policy, seed, scenario) simulation.

    Returns (event-log DataFrame, metadata dict) where metadata has
    arrival_times, completed_case_ids, availability_seconds, engine_stats --
    everything opt_metrics.evaluate() needs, plus run bookkeeping.

    `excluded_override`: an explicit set of resources to remove for THIS
    run, bypassing the scenario's own removal logic. This is what the
    "fire two employees" management question needs -- a *fixed* leave-N-out
    set, evaluated across seeds, rather than the 'outage' scenario's random
    20% (which reshuffles per seed). Pass an empty set to force nobody out.

    `roster_seed`: base seed for the p_work roster draw (None = rostering off).
    The *effective* seed is `roster_seed + seed`, which is what CRN requires:
    the roster is a condition of the run, not a property of the policy, so two
    policies at the same replication seed must face the identical workforce or
    the paired comparison measures roster luck instead of the policy. Adding
    the replication seed keeps rosters varying across replications, so the grid
    still samples workforce variability rather than fixing one lucky draw.
    """
    duration = days * 86400

    calendar = YearlyAvailability.from_json(
        AVAILABILITY_MODEL_PATH,
        roster_seed=None if roster_seed is None else roster_seed + seed,
    )
    perms, case_attrs = load_permission_model(permissions, seed)
    resource_pool = (perms.resources() if perms is not None
                     else sorted(RESOURCE_PERMISSIONS))
    excluded = (excluded_override if excluded_override is not None
                else scenario_excluded_resources(scenario, seed, resource_pool))

    lifecycle_params = (
        LifecycleParameters.from_file(ACTIVE_INPUTS_PATH)
        if lifecycle_mode == "active" else None
    )
    engine = SimulationEngine(
        sim_duration=duration, start_datetime=START_DATETIME, verbose=False,
        lifecycle_mode=lifecycle_mode)
    arrivals = ArrivalComponent(seed=seed, **scenario_arrival_kwargs(scenario))
    resources = build_resource_component(policy, seed, calendar, excluded,
                                         permission_model=perms,
                                         lifecycle_mode=lifecycle_mode,
                                         lifecycle_params=lifecycle_params,
                                         capacity=capacity)
    recorder = _ArrivalRecorder()
    tracker = CaseCompletionTracker()

    proc_kwargs = dict(
        seed=seed, mode="distribution", start_datetime=START_DATETIME,
        resource_component=resources, crn=crn, case_attributes=case_attrs,
        lifecycle_mode=lifecycle_mode, lifecycle_params=lifecycle_params,
    )
    if process_model == "advanced":
        process = PetriNetProcessComponent(
            bpmn_path=str(BPMN_PATH), branching_mode=branching_mode, **proc_kwargs,
        )
    else:
        process = ProcessComponent(**proc_kwargs)

    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)
    engine.register(recorder)
    engine.register(tracker)

    arrivals.bootstrap(engine)
    engine.run()

    df = pd.DataFrame(engine.logger._rows)
    if not df.empty:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="ISO8601")

    arrival_times = {
        cid: START_DATETIME + timedelta(seconds=t)
        for cid, t in recorder.timestamps.items()
    }
    availability_seconds = availability_seconds_per_resource(
        calendar, START_DATETIME, days, resource_pool,
    )

    meta = {
        "arrival_times": arrival_times,
        "completed_case_ids": tracker.completed_case_ids,
        "availability_seconds": availability_seconds,
        "engine_stats": dict(engine.stats),
        "lifecycle_mode": lifecycle_mode,
    }
    return df, meta


def apply_warmup(df: pd.DataFrame, meta: dict, warmup_days: float) -> tuple[pd.DataFrame, dict]:
    """Exclude every case that ARRIVED before the warm-up cutoff.

    Case-based, not time-based: a case straddling the cutoff (arrived
    before, still running after) is dropped entirely rather than having
    only its early activities trimmed. Simpler and defensible for
    per-instance metrics (cycle time, occupation) -- the simplification a
    reader should know about, documented here and in the report.
    """
    if warmup_days <= 0 or df.empty:
        return df, meta
    cutoff = START_DATETIME + timedelta(days=warmup_days)
    keep = {cid for cid, t in meta["arrival_times"].items() if t >= cutoff}
    df2 = df[df["case:concept:name"].isin(keep)]
    meta2 = dict(meta)
    meta2["arrival_times"] = {c: t for c, t in meta["arrival_times"].items() if c in keep}
    meta2["completed_case_ids"] = {c for c in meta["completed_case_ids"] if c in keep}
    return df2, meta2


# ---------------------------------------------------------------------
# WIP diagnostic (for choosing --warmup-days)
# ---------------------------------------------------------------------

def report_wip(days: int, lifecycle_mode: str = "legacy") -> None:
    """Print open-case count (arrived, not yet reached CASE_COMPLETE) per
    day for one pilot run (policy=random, seed=1) -- eyeball where it
    plateaus and pass that as --warmup-days. Cheap substitute for a full
    Welch's-method scan.

    Needs true per-case CASE_COMPLETE timestamps, not just "last completed
    activity" from the event log: a case's most recent completed activity
    can be old while the case is still queueing for its next one, which
    would undercount WIP if used as the "done" signal. Runs its own
    engine (not run_once()) so it can register _CompletionRecorder too.
    """
    duration = days * 86400
    calendar = YearlyAvailability.from_json(AVAILABILITY_MODEL_PATH)
    perms, case_attrs = load_permission_model("orgmodel", seed=1)
    lifecycle_params = (
        LifecycleParameters.from_file(ACTIVE_INPUTS_PATH)
        if lifecycle_mode == "active" else None
    )
    engine = SimulationEngine(
        sim_duration=duration, start_datetime=START_DATETIME, verbose=False,
        lifecycle_mode=lifecycle_mode)
    arrivals = ArrivalComponent(seed=1)
    resources = build_resource_component("random", 1, calendar, None,
                                         permission_model=perms,
                                         lifecycle_mode=lifecycle_mode,
                                         lifecycle_params=lifecycle_params)
    arrival_rec = _ArrivalRecorder()
    complete_rec = _CompletionRecorder()
    process = ProcessComponent(
        seed=1, mode="distribution", start_datetime=START_DATETIME,
        resource_component=resources, crn=True, case_attributes=case_attrs,
        lifecycle_mode=lifecycle_mode, lifecycle_params=lifecycle_params,
    )
    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)
    engine.register(arrival_rec)
    engine.register(complete_rec)
    arrivals.bootstrap(engine)
    engine.run()

    if not arrival_rec.timestamps:
        print("No arrivals recorded -- nothing to report.")
        return

    arrivals_s = pd.Series({
        c: START_DATETIME + timedelta(seconds=t) for c, t in arrival_rec.timestamps.items()
    })
    completes_s = pd.Series({
        c: START_DATETIME + timedelta(seconds=t) for c, t in complete_rec.timestamps.items()
    })

    print(f"{'day':>4}  {'open_cases (arrived, not yet CASE_COMPLETE by day end)':>55}")
    for day in range(days):
        day_end = START_DATETIME + timedelta(days=day + 1)
        arrived = int((arrivals_s < day_end).sum())
        done = int((completes_s < day_end).sum()) if not completes_s.empty else 0
        print(f"{day:>4}  {arrived - done:>55}")
    print("\nPick --warmup-days where the open-case count stops climbing "
          "(reaches its steady-state band).")


# ---------------------------------------------------------------------
# Aggregation: CI + paired tests
# ---------------------------------------------------------------------

METRICS = ["avg_cycle_time_s", "p95_cycle_time_s", "avg_resource_occupation", "resource_fairness"]


def flatten_result(policy: str, seed: int, scenario: str, m: dict, engine_stats: dict) -> dict:
    return {
        "policy": policy,
        "seed": seed,
        "scenario": scenario,
        "avg_cycle_time_s": m["cycle_time"]["avg_cycle_time_s"],
        "p95_cycle_time_s": m["cycle_time"]["p95_cycle_time_s"],
        "n_cases_completed": m["cycle_time"]["n_cases"],
        "avg_resource_occupation": m["occupation"]["avg_resource_occupation"],
        "resource_fairness": m["fairness"]["resource_fairness"],
        "cases_started": engine_stats.get("cases_started"),
        "cases_completed_total": engine_stats.get("cases_completed"),
    }


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)
    half = float(sem * scipy_stats.t.ppf(0.975, df=n - 1))
    return mean, half


def aggregate_and_report(results: pd.DataFrame, policies: list, out_dir: Path, scenario: str) -> None:
    baseline = "random" if "random" in policies else policies[0]

    agg_rows = []
    for policy in policies:
        sub = results[results["policy"] == policy]
        row = {"policy": policy, "n_seeds": len(sub)}
        for m in METRICS:
            mean, half = _ci95(sub[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95_halfwidth"] = half
            row[f"{m}_ci95_rel"] = half / abs(mean) if mean else float("nan")
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg_path = out_dir / f"aggregate_{scenario}.csv"
    agg.to_csv(agg_path, index=False)

    print(f"\n=== Aggregate ({scenario}) ===")
    print(agg.to_string(index=False))
    print(f"Saved -> {agg_path}")

    loose = agg[agg["avg_cycle_time_s_ci95_rel"] > 0.05]
    if not loose.empty:
        print(
            f"\nNOTE: CI half-width exceeds 5% of mean cycle time for "
            f"{list(loose['policy'])} -- roadmap target is <=5%; add more "
            f"seeds (--seeds) for these policies before treating the "
            f"comparison as conclusive."
        )

    pt_rows = []
    for policy in policies:
        if policy == baseline:
            continue
        merged = results[results["policy"] == baseline].merge(
            results[results["policy"] == policy], on="seed", suffixes=("_base", "_cmp"),
        )
        if merged.empty:
            continue
        for m in METRICS:
            a = merged[f"{m}_base"].values
            b = merged[f"{m}_cmp"].values
            if len(a) < 2:
                continue
            t_stat, p_value = scipy_stats.ttest_rel(b, a)
            pt_rows.append({
                "baseline": baseline, "policy": policy, "metric": m,
                "n_paired_seeds": len(a),
                "mean_delta": float(np.mean(b - a)),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            })
    pt = pd.DataFrame(pt_rows)
    pt_path = out_dir / f"paired_tests_{scenario}.csv"
    pt.to_csv(pt_path, index=False)
    if not pt.empty:
        print(f"\n=== Paired t-tests vs '{baseline}' ({scenario}) ===")
        print(pt.to_string(index=False))
    print(f"Saved -> {pt_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", default="random,piled",
                   help=f"Comma-separated policy names (known: {sorted(KNOWN_POLICIES)}, "
                        f"or 'kbatchN' for k-Batching with k=N, e.g. "
                        f"kbatch1,kbatch2,kbatch5,kbatch10,kbatch20 for a k-sweep).")
    p.add_argument("--seeds", type=int, default=10, help="Number of seeds (uses 1..N).")
    p.add_argument("--days", type=int, default=30, help="Simulation horizon in days.")
    p.add_argument("--warmup-days", type=float, default=0.0,
                   help="Exclude cases arriving before this cutoff (see --report-wip).")
    p.add_argument("--scenario", default="normal", choices=sorted(KNOWN_SCENARIOS))
    p.add_argument("--process-model", default="basic", choices=["basic", "advanced"],
                   help="'basic' avoids the pm4py dependency and runs faster; "
                        "use 'advanced' for report-quality numbers.")
    p.add_argument("--branching-mode", default="visit", choices=["probs", "visit", "rules"],
                   help="--process-model advanced only.")
    p.add_argument("--permissions", default="orgmodel",
                   choices=["orgmodel", "observed", "hardcoded"],
                   help="Section-1.7 permission model, mirroring simulation/main.py: "
                        "'orgmodel' (mined OrdinoR model, the team default), "
                        "'observed' (log-mined resource x activity matrix), "
                        "'hardcoded' (original top-20 map).")
    p.add_argument("--lifecycle-mode", default="legacy", choices=["legacy", "active"],
                   help="Lifecycle/artifact baseline (default: legacy). Active loads "
                        "simulation_inputs_active.json and logs work_item_id.")
    p.add_argument("--no-crn", dest="crn", action="store_false",
                   help="Disable Common Random Numbers (paired comparisons "
                        "become unreliable -- see module docstring).")
    p.set_defaults(crn=True)
    p.add_argument("--roster-seed", type=int, default=None, metavar="N",
                   help="Roll the fitted p_work per (resource, day), base seed N "
                        "(effective seed is N+run seed, so policies within a "
                        "replication share a roster -- required for CRN). Takes "
                        "the Monday workforce from ~123 to the validated ~37 and "
                        "makes contention real. Default off, which reproduces "
                        "pre-rostering evidence logs.")
    p.add_argument("--capacity", type=int, default=None, metavar="N",
                   help=f"Work items one resource may hold at once. Default is "
                        f"derived from --lifecycle-mode: "
                        f"{DEFAULT_CAPACITY_ACTIVE} for active (98.4%% of real "
                        f"busy time is a single hands-on session, and "
                        f"suspend/resume already models the interleaving), "
                        f"{DEFAULT_CAPACITY_LEGACY} for legacy (whose durations "
                        f"are elapsed spans that really do overlap, median peak "
                        f"54). The duration model has no concurrent-load "
                        f"feature, so N parallel items each finish as fast as "
                        f"one: N multiplies throughput for free.")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--report-wip", action="store_true",
                   help="Print a WIP-over-time diagnostic and exit (ignores --policies/--seeds).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.report_wip:
        report_wip(args.days, args.lifecycle_mode)
        return

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    unknown = [p for p in policies if not is_known_policy(p)]
    if unknown:
        print(f"Unknown polic{'y' if len(unknown) == 1 else 'ies'}: {sorted(unknown)} "
              f"(known: {sorted(KNOWN_POLICIES)}, or 'kbatchN' e.g. kbatch5)",
              file=sys.stderr)
        sys.exit(1)

    seeds = list(range(1, args.seeds + 1))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for policy in policies:
        for seed in seeds:
            df, meta = run_once(
                policy, seed, args.days, args.scenario, args.crn,
                args.process_model, args.branching_mode, args.permissions,
                lifecycle_mode=args.lifecycle_mode,
                roster_seed=args.roster_seed,
                capacity=args.capacity,
            )
            df, meta = apply_warmup(df, meta, args.warmup_days)
            if df.empty:
                print(f"WARNING: policy={policy} seed={seed}: empty log after "
                      f"warm-up filter, skipping this run.")
                continue
            # One failing run must not abort a multi-hour grid: log and skip.
            try:
                m = opt_metrics.evaluate(
                    df,
                    arrival_times=meta["arrival_times"],
                    availability_seconds=meta["availability_seconds"],
                    completed_case_ids=meta["completed_case_ids"],
                )
            except Exception as e:  # noqa: BLE001 — resilience over precision here
                print(f"WARNING: policy={policy} seed={seed}: evaluate() failed "
                      f"({type(e).__name__}: {e}), skipping this run.", file=sys.stderr)
                continue
            rows.append(flatten_result(policy, seed, args.scenario, m, meta["engine_stats"]))
            ct = m["cycle_time"]
            print(f"  [{policy:>7} seed={seed:>2}] "
                  f"cycle_time={ct['avg_cycle_time_s']/86400:.2f}d "
                  f"occupation={m['occupation']['avg_resource_occupation']:.3f} "
                  f"fairness={m['fairness']['resource_fairness']:.3f} "
                  f"(n={ct['n_cases']})")

    if not rows:
        print("No successful runs -- nothing to report.", file=sys.stderr)
        sys.exit(1)

    results = pd.DataFrame(rows)
    results_path = out_dir / f"results_{args.scenario}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nWrote {len(results)} rows -> {results_path}")

    aggregate_and_report(results, policies, out_dir, args.scenario)


if __name__ == "__main__":
    main()

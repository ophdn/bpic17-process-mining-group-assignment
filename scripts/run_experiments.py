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
    drl     -- frozen masked-PPO policy (requires --drl-model)
(k-batching and KRM use parameterized names such as kbatch5 and krm1.)

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

Warm-up uses two matching views: process metrics exclude cases that arrived
before --warmup-days, while resource metrics retain every activity that
overlaps the post-warm-up time window. Pick the value by first running with
--report-wip under the same simulation configuration.

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
import hashlib
import json
import random as _random
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.components.arrival import ArrivalComponent
from simulation.components.arrival_mdn import MDNArrivalComponent
from simulation.components.process import MAX_SESSIONS, ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import (
    DEFAULT_CAPACITY_ACTIVE, DEFAULT_CAPACITY_LEGACY, DEFAULT_ROSTER_SEED,
    RESOURCE_PERMISSIONS, ResourceComponent, capacity_for_mode,
)
from simulation.policies import RoundRobinPolicy, ShortestQueuePolicy
from simulation.components.lifecycle_params import LifecycleParameters
from simulation.main import (
    USE_MDN_ARRIVALS,
    CaseCompletionTracker,
    load_permission_context,
)
from analysis.availability import YearlyAvailability

import scripts.opt_metrics as opt_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
AVAILABILITY_MODEL_PATH = REPO_ROOT / "models" / "availability_model.json"
OUT_DEFAULT = REPO_ROOT / "output" / "experiments"
ACTIVE_INPUTS_PATH = REPO_ROOT / "simulation_inputs_active.json"
LEGACY_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model.joblib"
ACTIVE_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model_active.joblib"

# Files that can change the meaning of a cached evaluation result.  Keep these
# as repository-relative names so the resulting manifest is stable across
# machines and can be written directly into notebook provenance.
EVALUATION_PROVENANCE_PATHS = (
    "analysis/availability.py",
    "analysis/permissions.py",
    "models/availability_model.json",
    "models/case_attributes.json",
    "models/permissions_observed.json",
    "models/permissions_orgmodel.json",
    "scripts/opt_metrics.py",
    "scripts/run_experiments.py",
    "simulation/components/arrival_mdn.py",
    "simulation/components/arrival_mdn_weights.npz",
    "simulation/components/petri_process.py",
    "simulation/components/permissions.py",
    "simulation/components/resource.py",
    "simulation/components/process.py",
    "simulation/core/engine.py",
    "simulation/drl.py",
    "simulation/expected_duration.py",
    "simulation/models/bpic17_process.bpmn",
    "simulation/models/dp_branching_probs.json",
    "simulation/policies.py",
    "simulation/policies_advanced.py",
    "simulation_inputs.json",
    "simulation_inputs_active.json",
)

# BPIC-17 starts 2016-01-01 -- must match simulation/main.py's anchor so
# weekday/hour-of-day features (MDN arrivals, calendar) align.
START_DATETIME = datetime(2016, 1, 1)

KNOWN_POLICIES = {"random", "piled", "roundrobin", "shortestqueue",
                  "pullspt", "pulllaf", "parksong", "drl"}
_KRM_RE = re.compile(r"^krm(\d+(?:\.\d+)?)$")  # krm1, krm0.5, krm2 -> delta
KNOWN_SCENARIOS = {"normal", "peak", "outage"}
_KBATCH_RE = re.compile(r"^kbatch(\d+)$")
OUTAGE_FRACTION = 0.20


def evaluation_provenance_hashes() -> Dict[str, str]:
    """Return SHA-256 fingerprints for evaluation code and fitted inputs.

    A metric-only cache key is unsafe: resource allocation, lifecycle handling,
    the experiment wiring, or a fitted calendar can change while
    ``opt_metrics.py`` stays untouched.  The notebooks include this complete
    manifest in every cache key and in their exported configuration.
    """
    return {
        relative_path: hashlib.sha256(
            (REPO_ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in EVALUATION_PROVENANCE_PATHS
    }


def validate_evaluation_configuration(
    configuration: Mapping,
    expected_run_configuration: Mapping,
    expected_provenance: Mapping[str, str],
    cache_schema_version: int,
) -> None:
    """Reject a saved evaluation summary that is stale or from another run.

    Aggregate CSV files do not carry enough context to be combined safely on
    their own.  Their sibling ``configuration.json`` must use the current cache
    schema and code/input fingerprints, and it must agree with the run settings
    expected by the consuming notebook.  Extra descriptive fields are allowed.
    """
    problems = []
    actual_schema = configuration.get("cache_schema_version")
    if actual_schema != cache_schema_version:
        problems.append(
            f"cache_schema_version={actual_schema!r}, expected {cache_schema_version!r}"
        )

    actual_provenance = configuration.get("provenance_sha256")
    if actual_provenance != dict(expected_provenance):
        problems.append("provenance_sha256 does not match the current simulator inputs")

    for key, expected in expected_run_configuration.items():
        actual = configuration.get(key)
        if actual != expected:
            problems.append(f"{key}={actual!r}, expected {expected!r}")

    if problems:
        raise ValueError(
            "Saved evaluation configuration is incompatible; rerun its notebook. "
            + "; ".join(problems)
        )


def validate_resource_diagnostics(
    df: pd.DataFrame,
    resource_stats: dict,
    per_resource_occupation: dict,
    capacity: int,
) -> dict:
    """Validate and return JSON-safe resource diagnostics for one run.

    Active work must always be assigned to a permitted resource.  With unit
    capacity, active busy time cannot exceed one unit of realized calendar
    availability.  Queue length is not a failure condition, but it is retained
    because it makes finite-horizon truncation visible.
    """
    required_columns = {"lifecycle:transition", "org:resource"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise AssertionError(
            f"event log lacks resource-diagnostic columns: {sorted(missing_columns)}"
        )

    active = df["lifecycle:transition"].isin({"start", "resume"})
    active_resources = df.loc[active, "org:resource"]
    missing_resource_starts = int(
        active_resources.isna().sum()
        + active_resources.dropna().astype(str).str.strip().eq("").sum()
    )
    unpermitted = int(resource_stats.get("unpermitted_activities", -1))
    still_queued = int(resource_stats.get("still_queued_at_end", -1))

    occupation = pd.to_numeric(
        pd.Series(per_resource_occupation, dtype="object"), errors="coerce"
    ).dropna()
    max_occupation = float(occupation.max()) if not occupation.empty else None

    if unpermitted != 0:
        raise AssertionError(f"run contains {unpermitted} unpermitted activities")
    if missing_resource_starts != 0:
        raise AssertionError(
            f"run contains {missing_resource_starts} unassigned start/resume events"
        )
    if still_queued < 0:
        raise AssertionError("resource stats do not contain still_queued_at_end")
    if capacity == 1 and max_occupation is not None and max_occupation > 1.0 + 1e-9:
        raise AssertionError(
            f"resource occupation {max_occupation:.6f} exceeds unit capacity"
        )

    return {
        "unpermitted_activities": unpermitted,
        "still_queued_at_end": still_queued,
        "missing_resource_starts": missing_resource_starts,
        "max_resource_occupation": max_occupation,
    }


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


def build_arrival_component(seed: int, scenario: str):
    """Mirror simulation/main.py's Section-1.2 arrival wiring.

    This runner used to hardcode the parametric ArrivalComponent and never
    consulted main.USE_MDN_ARRIVALS at all, so the Part II grid silently ran on
    the *rejected* arrival model no matter what that switch said -- flipping it
    would not have reached these experiments. Deriving the choice from the same
    switch is what keeps main.py and the grid describing one engine.

    output/arrival_model_eval.md: the parametric inter-arrival distribution is
    statistically distinguishable from the real log (KS p = 3.18e-24) where the
    MDN is not (p = 0.389), and the MDN is ~12x closer on weekday shape. That
    matters most exactly here: the Section 1.6 roster gates staff by weekday and
    hour, so mis-shaped arrival timing misaligns demand against supply, which is
    what every contention and occupation number in Part II measures.
    """
    if USE_MDN_ARRIVALS:
        return MDNArrivalComponent(
            seed=seed, start_datetime=START_DATETIME,
            **scenario_arrival_kwargs(scenario))
    return ArrivalComponent(seed=seed, **scenario_arrival_kwargs(scenario))


def load_permission_model(kind: str, seed: int):
    """Mirror simulation/main.py's Section-1.7 wiring: returns
    (permission_model_or_None, case_attribute_sampler).

    "orgmodel" (the team default since 1.7 landed) gates permissions on the
    case type; "observed" is the log-mined resource x activity matrix;
    "hardcoded" is the original top-20 map (ResourceComponent's built-in
    default when passed None). Case attributes are sampled in every mode so
    changing permissions cannot silently change the case data.
    """
    return load_permission_context(kind, seed)


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


def parse_krm_policy(policy: str) -> Optional[float]:
    """'krm1' -> 1.0, 'krm0.5' -> 0.5, else None — Kunkler & Rinderle-Ma
    dummy-resource cost with delta as a multiplier on each item's mean
    expected duration (D2; sweep the delta the same way as kbatchN)."""
    m = _KRM_RE.match(policy)
    return float(m.group(1)) if m else None


def is_known_policy(policy: str) -> bool:
    return (policy in KNOWN_POLICIES
            or parse_kbatch_policy(policy) is not None
            or parse_krm_policy(policy) is not None)


def build_resource_component(
    policy: str, seed: int, calendar, excluded: Optional[Set[str]],
    permission_model=None, lifecycle_mode: str = "legacy", lifecycle_params=None,
    capacity: Optional[int] = None,
    drl_model_path: Optional[str] = None,
    branching_mode: str = "probs",
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
    if not is_known_policy(policy):
        raise ValueError(
            f"unknown policy {policy!r} (known: {sorted(KNOWN_POLICIES)}, "
            f"'kbatchN' for k-Batching with k=N, or 'krmD' for "
            "Kunkler & Rinderle-Ma with delta=D, e.g. krm1, krm0.5)."
        )
    selection_policy = None
    if policy == "roundrobin":
        selection_policy = RoundRobinPolicy()
    elif policy == "shortestqueue":
        selection_policy = ShortestQueuePolicy()

    pull = {"pullspt": "spt", "pulllaf": "laf"}.get(policy)
    krm_delta = parse_krm_policy(policy)
    parksong = (policy == "parksong")
    drl = (policy == "drl")
    needs_duration_model = pull == "spt" or parksong or krm_delta is not None

    return ResourceComponent(
        capacity_per_resource=capacity,
        seed=seed,
        calendar=calendar,
        start_datetime=START_DATETIME,
        permissions=permission_model,
        piled=(policy == "piled"),
        policy=selection_policy,
        pull=pull,
        parksong=parksong,
        krm_delta=krm_delta,
        drl=drl,
        drl_model_path=drl_model_path,
        duration_model_path=(
            str(ACTIVE_MODEL_PATH if lifecycle_mode == "active" else LEGACY_MODEL_PATH)
            if needs_duration_model else None),
        excluded_resources=excluded,
        lifecycle_mode=lifecycle_mode,
        lifecycle_params=lifecycle_params,
        branching_mode=branching_mode,
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
    intervals = availability_intervals_per_resource(
        calendar, start_dt, horizon_days, resources,
    )
    return {
        resource: sum((end - start).total_seconds() for start, end in spans)
        for resource, spans in intervals.items()
    }


def availability_intervals_per_resource(
    calendar: Optional[YearlyAvailability], start_dt: datetime, horizon_days: int,
    resources,
) -> Dict[str, list[tuple[datetime, datetime]]]:
    """Realized on-duty intervals within the simulation horizon.

    Occupation is the share of *available* time spent working.  A task may
    finish after its resource's shift ends, so summing its full active session
    against scheduled hours can exceed one even with unit capacity.  Supplying
    the actual intervals lets the metric count only the active-session overlap
    with the realized roster while separately reporting overtime.
    """
    horizon_end = start_dt + timedelta(days=horizon_days)
    out = {resource: [] for resource in resources}
    if calendar is None:
        if horizon_end > start_dt:
            for resource in resources:
                out[resource].append((start_dt, horizon_end))
        return out

    system = getattr(calendar, "system", None)
    first_day = datetime.combine(start_dt.date(), datetime.min.time())
    for day in range(horizon_days + 1):
        day_start = first_day + timedelta(days=day)
        if day_start >= horizon_end:
            break
        d = day_start.date()
        dow = d.weekday()
        for resource in resources:
            windows = calendar.weekly.windows.get(resource)
            if windows is None:
                # Mirrors ResourceComponent._is_on_shift: known system accounts
                # (and calendars predating the system set) are always available;
                # an unfitted human is not.
                if system is not None and resource not in system:
                    continue
                interval_start = max(start_dt, day_start)
                interval_end = min(horizon_end, day_start + timedelta(days=1))
            else:
                if d in calendar.holidays:
                    continue
                if d in calendar.vacations.get(resource, ()):
                    continue
                if not calendar._works_today(resource, d):
                    continue
                window = windows.get(dow)
                if window is None:
                    continue
                interval_start = max(
                    start_dt, day_start + timedelta(hours=window[0]),
                )
                interval_end = min(
                    horizon_end, day_start + timedelta(hours=window[1]),
                )
            if interval_end > interval_start:
                out[resource].append((interval_start, interval_end))
    return out


# ---------------------------------------------------------------------
# One simulation run
# ---------------------------------------------------------------------

def run_once(
    policy: str, seed: int, days: int, scenario: str, crn: bool,
    process_model: str, branching_mode: str, *, lifecycle_mode: str,
    processing_time_mode: str = "distribution", permissions: str = "orgmodel",
    excluded_override: Optional[Set[str]] = None,
    roster_seed: Optional[int] = DEFAULT_ROSTER_SEED,
    capacity: Optional[int] = None,
    atomic_duration_scale: float = 1.0,
    drl_model_path: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build and run one (policy, seed, scenario) simulation.

    Returns (event-log DataFrame, metadata dict) where metadata has
    arrival_times, completed_case_ids, availability_seconds, engine_stats, and
    resource_stats -- everything opt_metrics.evaluate() and the evaluation
    guardrails need, plus run bookkeeping.

    ``lifecycle_mode`` is deliberately required. Evaluation callers must choose
    the active-session model or the legacy elapsed-duration model explicitly;
    silently falling back to legacy data invalidated the original notebook.

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
    if lifecycle_mode not in {"legacy", "active"}:
        raise ValueError(
            f"lifecycle_mode must be 'legacy' or 'active', got {lifecycle_mode!r}")
    if processing_time_mode not in {"distribution", "ml_model", "ml_probabilistic"}:
        raise ValueError(
            "processing_time_mode must be distribution|ml_model|ml_probabilistic, "
            f"got {processing_time_mode!r}")
    if atomic_duration_scale < 0:
        raise ValueError("atomic_duration_scale must be >= 0")
    if process_model == "basic" and branching_mode != "probs":
        raise ValueError(
            f"branching_mode={branching_mode!r} requires process_model='advanced'; "
            "use branching_mode='probs' with the basic process")

    duration = days * 86400

    # Effective roster seed = base + run seed (see docstring). Capture the
    # resolved capacity too, so the run's provenance records what actually ran
    # rather than the None sentinels -- these two defaults change the numbers,
    # so a result file that does not name them is not reproducible.
    effective_roster_seed = None if roster_seed is None else roster_seed + seed
    effective_capacity = (capacity if capacity is not None
                          else capacity_for_mode(lifecycle_mode))

    calendar = YearlyAvailability.from_json(
        AVAILABILITY_MODEL_PATH, roster_seed=effective_roster_seed,
    )
    perms, case_attrs = load_permission_model(permissions, seed)
    resource_pool = (perms.resources() if perms is not None
                     else sorted(RESOURCE_PERMISSIONS))
    excluded = (excluded_override if excluded_override is not None
                else scenario_excluded_resources(scenario, seed, resource_pool))
    excluded = set(excluded or ())
    active_resource_pool = set(resource_pool) - excluded

    lifecycle_params = (
        LifecycleParameters.from_file(ACTIVE_INPUTS_PATH)
        if lifecycle_mode == "active" else None
    )
    processing_time_model_path = (
        ACTIVE_MODEL_PATH if lifecycle_mode == "active" else LEGACY_MODEL_PATH
    )
    engine = SimulationEngine(
        sim_duration=duration, start_datetime=START_DATETIME, verbose=False,
        lifecycle_mode=lifecycle_mode)
    arrivals = build_arrival_component(seed, scenario)
    resources = build_resource_component(policy, seed, calendar, excluded,
                                         permission_model=perms,
                                         lifecycle_mode=lifecycle_mode,
                                         lifecycle_params=lifecycle_params,
                                         capacity=effective_capacity,
                                         drl_model_path=drl_model_path,
                                         branching_mode=branching_mode)
    recorder = _ArrivalRecorder()
    completion_recorder = _CompletionRecorder()
    tracker = CaseCompletionTracker()

    proc_kwargs = dict(
        seed=seed, mode=processing_time_mode, start_datetime=START_DATETIME,
        model_path=(
            str(processing_time_model_path)
            if processing_time_mode != "distribution" else None
        ),
        resource_component=resources, crn=crn, case_attributes=case_attrs,
        lifecycle_mode=lifecycle_mode, lifecycle_params=lifecycle_params,
        atomic_duration_scale=atomic_duration_scale,
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
    engine.register(completion_recorder)
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
    completion_times = {
        cid: START_DATETIME + timedelta(seconds=t)
        for cid, t in completion_recorder.timestamps.items()
    }
    availability_seconds = availability_seconds_per_resource(
        calendar, START_DATETIME, days, active_resource_pool,
    )
    availability_intervals = availability_intervals_per_resource(
        calendar, START_DATETIME, days, active_resource_pool,
    )
    lifecycle_diagnostics = opt_metrics.lifecycle_diagnostics(
        df,
        engine_stats=engine.stats,
        max_sessions=MAX_SESSIONS,
    )
    activity_type_exposure = opt_metrics.activity_type_exposure(
        df,
        availability_intervals=availability_intervals,
    )
    system_resources = set(getattr(calendar, "system", set()))
    human_resources = active_resource_pool - system_resources

    meta = {
        "arrival_times": arrival_times,
        "completion_times": completion_times,
        "completed_case_ids": tracker.completed_case_ids,
        "availability_seconds": availability_seconds,
        "availability_intervals": availability_intervals,
        "engine_stats": dict(engine.stats),
        "resource_stats": resources.stats(),
        "lifecycle_diagnostics": lifecycle_diagnostics,
        "activity_type_exposure": activity_type_exposure,
        "resource_subset": human_resources,
        "evaluation_window": (
            START_DATETIME, START_DATETIME + timedelta(days=days)),
        "lifecycle_mode": lifecycle_mode,
        "processing_time_mode": processing_time_mode,
        "configuration": {
            "policy": policy,
            "seed": seed,
            "horizon_days": days,
            "scenario": scenario,
            "crn": crn,
            "process_model": process_model,
            "branching_mode": branching_mode,
            "permissions": permissions,
            "lifecycle_mode": lifecycle_mode,
            "processing_time_mode": processing_time_mode,
            "processing_time_model_path": (
                str(processing_time_model_path.relative_to(REPO_ROOT))
                if processing_time_mode != "distribution" else None
            ),
            "atomic_duration_scale": float(atomic_duration_scale),
            "roster_seed": effective_roster_seed,
            "capacity": effective_capacity,
            "arrival_model": "mdn" if USE_MDN_ARRIVALS else "parametric",
            "drl_model_path": str(drl_model_path) if policy == "drl" else None,
            "excluded_resources": sorted(excluded),
            "resource_pool_size": len(active_resource_pool),
            "human_resource_count": len(human_resources),
            "automated_resources": sorted(active_resource_pool & system_resources),
        },
    }
    return df, meta


def _clip_availability_intervals(
    intervals: Dict[str, list[tuple[datetime, datetime]]],
    start: datetime,
    end: datetime,
) -> Dict[str, list[tuple[datetime, datetime]]]:
    """Clip realized resource calendars to an evaluation time window."""
    return {
        resource: [
            (max(span_start, start), min(span_end, end))
            for span_start, span_end in spans
            if min(span_end, end) > max(span_start, start)
        ]
        for resource, spans in intervals.items()
    }


def apply_warmup(df: pd.DataFrame, meta: dict, warmup_days: float) -> tuple[pd.DataFrame, dict]:
    """Prepare consistent case- and resource-based post-warm-up views.

    Cycle-time and milestone metrics keep only cases that *arrive* after the
    cutoff. Resource metrics instead use the original event log and clip busy
    sessions plus availability to the post-cutoff time window. Thus work done
    after the cutoff for an older case still counts toward occupation/fairness,
    without admitting that older case into the cycle-time sample.
    """
    if warmup_days <= 0:
        return df, meta
    cutoff = START_DATETIME + timedelta(days=warmup_days)
    _, horizon_end = meta["evaluation_window"]
    if cutoff >= horizon_end:
        raise ValueError("--warmup-days must be smaller than --days")
    keep = {cid for cid, t in meta["arrival_times"].items() if t >= cutoff}
    df2 = (
        df[df["case:concept:name"].isin(keep)]
        if "case:concept:name" in df.columns else df.copy()
    )
    meta2 = dict(meta)
    meta2["arrival_times"] = {c: t for c, t in meta["arrival_times"].items() if c in keep}
    meta2["completed_case_ids"] = {c for c in meta["completed_case_ids"] if c in keep}
    clipped = _clip_availability_intervals(
        meta["availability_intervals"], cutoff, horizon_end)
    meta2["availability_intervals"] = clipped
    meta2["availability_seconds"] = {
        resource: sum((end - start).total_seconds() for start, end in spans)
        for resource, spans in clipped.items()
    }
    meta2["evaluation_window"] = (cutoff, horizon_end)
    meta2["configuration"] = dict(meta["configuration"], warmup_days=warmup_days)
    return df2, meta2


# ---------------------------------------------------------------------
# WIP diagnostic (for choosing --warmup-days)
# ---------------------------------------------------------------------

def report_wip(
    days: int,
    *,
    scenario: str,
    process_model: str,
    branching_mode: str,
    permissions: str,
    lifecycle_mode: str,
    processing_time_mode: str,
    roster_seed: Optional[int],
    capacity: Optional[int],
) -> None:
    """Print open-case count (arrived, not yet reached CASE_COMPLETE) per
    day for one pilot run (policy=random, seed=1) -- eyeball where it
    plateaus and pass that as --warmup-days. Cheap substitute for a full
    Welch's-method scan.

    It deliberately calls ``run_once`` so the arrival, process, permission,
    roster, capacity, lifecycle and processing-time settings are identical to
    the experiment for which the warm-up is being selected.
    """
    _, meta = run_once(
        "random", 1, days, scenario, True, process_model, branching_mode,
        lifecycle_mode=lifecycle_mode,
        processing_time_mode=processing_time_mode,
        permissions=permissions,
        roster_seed=roster_seed,
        capacity=capacity,
    )
    if not meta["arrival_times"]:
        print("No arrivals recorded -- nothing to report.")
        return
    arrivals_s = pd.Series(meta["arrival_times"])
    completes_s = pd.Series(meta["completion_times"])

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

METRICS = [
    "avg_cycle_time_s", "p95_cycle_time_s", "n_cases_completed",
    "avg_resource_occupation", "resource_fairness",
    "time_to_first_offer_s", "time_to_decision_s", "handover_rate",
    "resource_activity_switch_rate", "rolling_workload_balance",
    "still_queued_at_end", "mean_wait_seconds",
]


def flatten_result(policy: str, seed: int, scenario: str, m: dict, meta: dict) -> dict:
    """Flatten metrics, diagnostics and effective configuration into one row."""
    engine_stats = meta["engine_stats"]
    resource_stats = meta["resource_stats"]
    custom = m["custom_metrics"]
    row = {
        "policy": policy,
        "seed": seed,
        "scenario": scenario,
        "avg_cycle_time_s": m["cycle_time"]["avg_cycle_time_s"],
        "p95_cycle_time_s": m["cycle_time"]["p95_cycle_time_s"],
        "n_cases_completed": m["cycle_time"]["n_cases"],
        "avg_resource_occupation": m["occupation"]["avg_resource_occupation"],
        "resource_fairness": m["fairness"]["resource_fairness"],
        "weighted_resource_fairness": m["fairness"].get(
            "weighted_resource_fairness"),
        "time_to_first_offer_s": custom["time_to_first_offer"]["mean_s"],
        "time_to_decision_s": custom["time_to_decision"]["mean_s"],
        "handover_rate": custom["handover_rate"]["handover_rate"],
        "resource_activity_switch_rate": custom[
            "resource_activity_switch_rate"]["activity_switch_rate"],
        "rolling_workload_balance": custom[
            "rolling_workload_balance"]["mean_window_std"],
        "cases_started": engine_stats.get("cases_started"),
        "cases_completed_total": engine_stats.get("cases_completed"),
        "still_queued_at_end": resource_stats.get("still_queued_at_end"),
        "mean_wait_seconds": resource_stats.get("mean_wait_seconds"),
        "drl_assignments": resource_stats.get("drl_assignments"),
        "drl_postponements": resource_stats.get("drl_postponements"),
        "busy_seconds_outside_availability": m["occupation"].get(
            "busy_seconds_outside_availability"),
        "n_resources_evaluated": m["occupation"].get("n_resources_evaluated"),
    }
    diagnostics = meta.get("diagnostics", {})
    row.update({f"diagnostic_{key}": value for key, value in diagnostics.items()})
    for key, value in meta["configuration"].items():
        row[f"config_{key}"] = (
            json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value)
    return row


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
    p.add_argument("--process-model", default="advanced", choices=["basic", "advanced"],
                   help="Advanced is the report-quality default. Basic is a faster "
                        "diagnostic and requires --branching-mode probs.")
    p.add_argument("--branching-mode", default="visit", choices=["probs", "visit", "rules"],
                   help="--process-model advanced only.")
    p.add_argument("--permissions", default="orgmodel",
                   choices=["orgmodel", "observed", "hardcoded"],
                   help="Section-1.7 permission model, mirroring simulation/main.py: "
                        "'orgmodel' (mined OrdinoR model, the team default), "
                        "'observed' (log-mined resource x activity matrix), "
                        "'hardcoded' (original top-20 map).")
    p.add_argument("--lifecycle-mode", default="active", choices=["legacy", "active"],
                   help="Lifecycle/artifact baseline (default: active). Active loads "
                        "simulation_inputs_active.json and logs work_item_id.")
    p.add_argument("--processing-time-mode", default="distribution",
                   choices=["distribution", "ml_model", "ml_probabilistic"],
                   help="Duration sampler used consistently across policy runs.")
    p.add_argument(
        "--atomic-duration-scale", type=float, default=1.0, metavar="S",
        help="Scale synthetic A_/O_ durations in active mode. Use 0.0 for the "
             "instantaneous-transition sensitivity bound.",
    )
    p.add_argument("--no-crn", dest="crn", action="store_false",
                   help="Disable Common Random Numbers (paired comparisons "
                        "become unreliable -- see module docstring).")
    p.set_defaults(crn=True)
    p.add_argument("--roster-seed", type=int, default=None, metavar="N",
                   help=f"Base seed for the p_work roster draw (effective seed "
                        f"is N+run seed, so policies within a replication share "
                        f"a roster -- required for CRN). Default "
                        f"{DEFAULT_ROSTER_SEED}; rostering is ON by default. "
                        f"Without it the Monday workforce is ~123 against the "
                        f"validated ~37 and contention is not real.")
    p.add_argument("--no-roster", action="store_true", default=False,
                   help="Disable the p_work roster (pre-rostering behaviour, "
                        "~3.3x overstaffed). For reproducing older evidence.")
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
    p.add_argument(
        "--drl-model", default=None, metavar="PATH",
        help="Frozen MaskablePPO .zip model used by policy 'drl'. Train it "
             "with scripts/train_drl.py. The optional requirements-drl.txt "
             "stack is needed only when this policy is selected.",
    )
    p.add_argument("--report-wip", action="store_true",
                   help="Print a WIP-over-time diagnostic and exit (ignores --policies/--seeds).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.roster_seed is not None and args.no_roster:
        print("--roster-seed and --no-roster are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if args.atomic_duration_scale < 0:
        print("--atomic-duration-scale must be >= 0.", file=sys.stderr)
        sys.exit(1)
    # Explicit N wins; --no-roster disables; otherwise the default is ON.
    roster_seed = None if args.no_roster else (
        args.roster_seed if args.roster_seed is not None else DEFAULT_ROSTER_SEED)

    if args.process_model == "basic" and args.branching_mode != "probs":
        print("--process-model basic requires --branching-mode probs.", file=sys.stderr)
        sys.exit(1)
    if args.warmup_days < 0 or args.warmup_days >= args.days:
        print("--warmup-days must satisfy 0 <= warmup-days < days.", file=sys.stderr)
        sys.exit(1)

    if args.report_wip:
        report_wip(
            args.days,
            scenario=args.scenario,
            process_model=args.process_model,
            branching_mode=args.branching_mode,
            permissions=args.permissions,
            lifecycle_mode=args.lifecycle_mode,
            processing_time_mode=args.processing_time_mode,
            roster_seed=roster_seed,
            capacity=args.capacity,
        )
        return

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    unknown = [p for p in policies if not is_known_policy(p)]
    if unknown:
        print(f"Unknown polic{'y' if len(unknown) == 1 else 'ies'}: {sorted(unknown)} "
              f"(known: {sorted(KNOWN_POLICIES)}, or 'kbatchN' e.g. kbatch5)",
              file=sys.stderr)
        sys.exit(1)
    if "drl" in policies and not args.drl_model:
        print("Policy 'drl' requires --drl-model PATH (train one with "
              "scripts/train_drl.py).", file=sys.stderr)
        sys.exit(1)

    seeds = list(range(1, args.seeds + 1))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "schema_version": 2,
        "policies": policies,
        "seeds": seeds,
        "days": args.days,
        "warmup_days": args.warmup_days,
        "scenario": args.scenario,
        "process_model": args.process_model,
        "branching_mode": args.branching_mode,
        "permissions": args.permissions,
        "lifecycle_mode": args.lifecycle_mode,
        "processing_time_mode": args.processing_time_mode,
        "atomic_duration_scale": args.atomic_duration_scale,
        "crn": args.crn,
        "roster_seed": roster_seed,
        "capacity": args.capacity if args.capacity is not None else capacity_for_mode(
            args.lifecycle_mode),
        "arrival_model": "mdn" if USE_MDN_ARRIVALS else "parametric",
        "drl_model": str(Path(args.drl_model).resolve()) if args.drl_model else None,
        "provenance_sha256": evaluation_provenance_hashes(),
    }
    if args.drl_model:
        model_path = Path(args.drl_model)
        if not model_path.exists() and model_path.suffix != ".zip":
            model_path = model_path.with_suffix(".zip")
        if model_path.exists():
            configuration["drl_model_sha256"] = hashlib.sha256(
                model_path.read_bytes()).hexdigest()
    configuration_path = out_dir / "configuration.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            print(
                f"{configuration_path} describes a different experiment. "
                "Use a new --out directory to avoid overwriting incomparable results.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        configuration_path.write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rows = []
    for policy in policies:
        for seed in seeds:
            df, meta = run_once(
                policy, seed, args.days, args.scenario, args.crn,
                args.process_model, args.branching_mode,
                lifecycle_mode=args.lifecycle_mode,
                processing_time_mode=args.processing_time_mode,
                permissions=args.permissions,
                roster_seed=roster_seed,
                capacity=args.capacity,
                atomic_duration_scale=args.atomic_duration_scale,
                drl_model_path=args.drl_model,
            )
            resource_df = df
            df, meta = apply_warmup(df, meta, args.warmup_days)
            meta["configuration"] = dict(
                meta["configuration"], warmup_days=args.warmup_days)
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
                    availability_intervals=meta["availability_intervals"],
                    fairness_weights=meta["availability_seconds"],
                    completed_case_ids=meta["completed_case_ids"],
                    resource_subset=meta["resource_subset"],
                    resource_df=resource_df,
                    evaluation_window=meta["evaluation_window"],
                )
                meta["diagnostics"] = validate_resource_diagnostics(
                    resource_df,
                    meta["resource_stats"],
                    m["occupation"]["per_resource"],
                    meta["configuration"]["capacity"],
                )
            except Exception as e:  # noqa: BLE001 — resilience over precision here
                print(f"WARNING: policy={policy} seed={seed}: evaluate() failed "
                      f"({type(e).__name__}: {e}), skipping this run.", file=sys.stderr)
                continue
            rows.append(flatten_result(policy, seed, args.scenario, m, meta))
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

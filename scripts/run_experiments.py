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
import hashlib
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
from simulation.components import permissions as perm_models
from simulation.policies import RoundRobinPolicy, ShortestQueuePolicy
from simulation.components.case_attributes import CaseAttributeSampler
from simulation.components.lifecycle_params import LifecycleParameters
from simulation.main import USE_MDN_ARRIVALS, CaseCompletionTracker
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

# Files that can change the meaning of a cached evaluation result.  Keep these
# as repository-relative names so the resulting manifest is stable across
# machines and can be written directly into notebook provenance.
EVALUATION_PROVENANCE_PATHS = (
    "analysis/availability.py",
    "analysis/permissions.py",
    "models/case_attributes.json",
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
    "simulation/expected_duration.py",
    "simulation/models/bpic17_process.bpmn",
    "simulation/models/dp_branching_probs.json",
    "simulation/policies.py",
    "models/availability_model.json",
    "simulation_inputs.json",
    "simulation_inputs_active.json",
)

# BPIC-17 starts 2016-01-01 -- must match simulation/main.py's anchor so
# weekday/hour-of-day features (MDN arrivals, calendar) align.
START_DATETIME = datetime(2016, 1, 1)

KNOWN_POLICIES = {"random", "piled", "roundrobin", "shortestqueue",
                  "pullspt", "pulllaf", "parksong"}
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
        duration_model_path=(
            str(ACTIVE_MODEL_PATH if lifecycle_mode == "active" else LEGACY_MODEL_PATH)
            if needs_duration_model else None),
        excluded_resources=excluded,
        lifecycle_mode=lifecycle_mode,
        lifecycle_params=lifecycle_params,
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
                                         capacity=effective_capacity)
    recorder = _ArrivalRecorder()
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
    availability_intervals = availability_intervals_per_resource(
        calendar, START_DATETIME, days, resource_pool,
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

    meta = {
        "arrival_times": arrival_times,
        "completed_case_ids": tracker.completed_case_ids,
        "availability_seconds": availability_seconds,
        "availability_intervals": availability_intervals,
        "engine_stats": dict(engine.stats),
        "resource_stats": resources.stats(),
        "lifecycle_diagnostics": lifecycle_diagnostics,
        "activity_type_exposure": activity_type_exposure,
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
            "excluded_resources": sorted(excluded or ()),
            "resource_pool_size": len(resource_pool),
        },
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
    p.add_argument("--report-wip", action="store_true",
                   help="Print a WIP-over-time diagnostic and exit (ignores --policies/--seeds).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.report_wip:
        report_wip(args.days, args.lifecycle_mode)
        return

    if args.roster_seed is not None and args.no_roster:
        print("--roster-seed and --no-roster are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if args.atomic_duration_scale < 0:
        print("--atomic-duration-scale must be >= 0.", file=sys.stderr)
        sys.exit(1)
    # Explicit N wins; --no-roster disables; otherwise the default is ON.
    roster_seed = None if args.no_roster else (
        args.roster_seed if args.roster_seed is not None else DEFAULT_ROSTER_SEED)

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
                args.process_model, args.branching_mode,
                lifecycle_mode=args.lifecycle_mode,
                processing_time_mode=args.processing_time_mode,
                permissions=args.permissions,
                roster_seed=roster_seed,
                capacity=args.capacity,
                atomic_duration_scale=args.atomic_duration_scale,
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

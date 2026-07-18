"""
main.py — Run the BPIC-17 simulation with real data-driven components.

Components used:
  - ArrivalComponent  (Section 1.2 Basic): LogNormal inter-arrivals from BPIC-17
  - ProcessComponent  (Section 1.3+1.5 Basic): fitted distributions + branching probs
  - PetriNetProcessComponent (Section 1.4 Advanced): loads a .bpmn file, converts
    it to a Petri net, and enforces control-flow via Petri net firing rules
    instead of the flat next-activity graph. Toggle with --process-model.
    Branching at each decision point is either BRANCHING_PROBS (--branching-mode
    probs, default) or a decision-point classifier trained on case/runtime data
    attributes (--branching-mode rules, Section 1.5 Advanced I).
  - ResourceComponent (Section 1.7+1.8 Basic): permission map + random allocation
  - EventLogger       (Section 1.1 Basic): built-in, outputs CSV

Usage:
    cd simulation/
    PYTHONPATH=.. python main.py
    PYTHONPATH=.. python main.py --process-model basic
    PYTHONPATH=.. python main.py --branching-mode rules
"""

import argparse
from datetime import datetime
from pathlib import Path

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType
from simulation.components.arrival import ArrivalComponent
from simulation.components.arrival_mdn import MDNArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import (
    DEFAULT_CAPACITY_ACTIVE, DEFAULT_CAPACITY_LEGACY, DEFAULT_ROSTER_SEED,
    ResourceComponent, capacity_for_mode,
)
from simulation.components import permissions as perm_models
from simulation.components.case_attributes import CaseAttributeSampler

# ── Configuration ────────────────────────────────────────────────────────────

# Simulation horizon: the 99th percentile of real case duration, not an
# arbitrary "30 days". Computed from data/BPIChallenge2017.xes.gz (events
# filtered to lifecycle='complete' per extract_log_info.filter_to_complete,
# per-case duration = last_ts - first_ts): p99 = 5,110,843.8s ~= 59.15 days,
# rounded to a clean 60 days -- i.e. 99% of real cases finish within this
# horizon of their own arrival (vs. only 80% at the p80 figure, ~31.94 days).
# Stable across log-boundary right-censoring checks at the p80 level (stayed
# 31.9-32.3 days whether cases with <30d or <90d of runway before the log's
# end are excluded or not) -- see docs/ROADMAP.md.
# NOTE: arrivals still span the whole horizon (no separate arrival cutoff +
# drain period), so this alone does not eliminate horizon censoring -- see
# docs/ROADMAP.md's drain-analysis note (scripts/drain_analysis.py).
SIM_DURATION_DAYS = 60
SIM_DURATION_SECONDS = SIM_DURATION_DAYS * 24 * 3600

# BPIC-17 starts 2016-01-01; anchor t=0 to the same date
START_DATETIME = datetime(2016, 1, 1)

# Repo root (parent of simulation/) — anchor point for paths below so
# results and model artifacts land in the same place regardless of the
# working directory main.py is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Simulation results (event logs) always go to <repo_root>/output/, never
# inside simulation/ — one canonical output location for every run.
OUTPUT_PATH = REPO_ROOT / "output" / "event_log.csv"

# Engine input artifacts (things the simulation *loads*, not produces)
# live under simulation/models/.
# Default trained-model artifact for the ML processing-time modes
DEFAULT_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model.joblib"
# Active-lifecycle artifacts (§4.8) — versioned so retraining never overwrites the
# legacy artifacts and a legacy run stays bit-reproducible.
ACTIVE_MODEL_PATH = REPO_ROOT / "simulation" / "models" / "processing_time_model_active.joblib"
ACTIVE_INPUTS_PATH = REPO_ROOT / "simulation_inputs_active.json"

# Section 1.4 Advanced: BPMN model discovered from the real BPIC-17 log
# (Inductive Miner). Loaded and converted to a Petri net whose firing rules
# enforce control-flow (see simulation/components/petri_process.py).
DEFAULT_BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"

# Section 1.6: resource availability model, fitted in
# notebooks/01_resource_availability.ipynb.
AVAILABILITY_MODEL_PATH = REPO_ROOT / "models" / "availability_model.json"

# Section 1.5 Advanced I: decision-point classifiers trained on case/runtime
# data attributes (see train_decision_rules.py / petri_process.py).
DEFAULT_DECISION_RULES_PATH = REPO_ROOT / "simulation" / "models" / "decision_rules.joblib"

# Section 1.7: permission models, fitted in
# notebooks/02_resource_permissions.ipynb.
ORGMODEL_PATH        = REPO_ROOT / "models" / "permissions_orgmodel.json"
OBSERVED_PERMS_PATH  = REPO_ROOT / "models" / "permissions_observed.json"
CASE_ATTRIBUTES_PATH = REPO_ROOT / "models" / "case_attributes.json"

RANDOM_SEED = 42   # Fix for reproducibility — required by assignment grading!


class CaseCompletionTracker:
    """Records which cases reach CASE_COMPLETE (finish naturally instead of
    being cut off by the simulation horizon). Every evaluation must filter
    the event log to these cases first — horizon-truncated cases would bias
    cycle time, case length and duration downwards. The ids are written to
    output/completed_cases.txt so downstream tools (scripts/opt_metrics.py,
    scripts/metrics.py callers) can apply the filter without re-running."""

    HANDLES = {EventType.CASE_COMPLETE: None}

    def __init__(self):
        self.completed_case_ids = set()

    def on_case_complete(self, engine, event):
        self.completed_case_ids.add(event.case_id)


CaseCompletionTracker.HANDLES = {
    EventType.CASE_COMPLETE: CaseCompletionTracker.on_case_complete
}

# Arrival-Modell wählen:
#   False = parametrisch (LogNormal, Section 1.2 Basic)
#   True  = MDN, zeitabhängig (Section 1.2 Advanced) — Gewichte aus train_arrival_mdn.py
#
# ON by default (team decision 2026-07-17: every fitted engine component runs by
# default). output/arrival_model_eval.md evaluated both against the raw log over
# 90 days and the parametric model is *statistically rejected* — inter-arrival
# KS p = 3.18e-24, i.e. its inter-arrival distribution is distinguishable from
# reality with near-certainty — while the MDN is not (p = 0.389). The MDN is
# also ~8x better on hour-of-day shape (MAE 0.0036 vs 0.0295) and ~12x better on
# weekday shape (0.0029 vs 0.0344).
#
# Weekday/hour shape is not cosmetic here: the Section 1.6 roster gates staff by
# weekday and hour, so getting arrival *timing* wrong misaligns demand against
# supply, which is precisely what any contention or occupation result measures.
USE_MDN_ARRIVALS = True

# ── Build & run ───────────────────────────────────────────────────────────────

def main(
    mode: str = "distribution",
    model_path: str | None = None,
    process_model: str = "advanced",
    bpmn_path: str | None = None,
    availability: str = "calendar",
    permissions: str = "orgmodel",
    branching_mode: str = "probs",
    decision_rules_path: str | None = None,
    enforce_terminal_outcomes: bool = True,
    piled_execution: bool = False,
    k_batching: int | None = None,
    lifecycle_mode: str = "legacy",
    active_inputs_path: str | None = None,
    roster_seed: int | None = DEFAULT_ROSTER_SEED,
    capacity: int | None = None,
):
    if lifecycle_mode not in ("legacy", "active"):
        raise ValueError(f"lifecycle_mode must be legacy|active, got {lifecycle_mode!r}")
    if capacity is None:
        capacity = capacity_for_mode(lifecycle_mode)
    if model_path is None:
        model_path = str(
            ACTIVE_MODEL_PATH if lifecycle_mode == "active" else DEFAULT_MODEL_PATH
        )
    engine = SimulationEngine(
        sim_duration=SIM_DURATION_SECONDS,
        start_datetime=START_DATETIME,
        verbose=False,   # set True to print every event (slow for large runs)
        lifecycle_mode=lifecycle_mode,
    )

    # Active lifecycle mode (§4.4): load the mined active-time + churn parameters.
    # Legacy mode never constructs this and keeps the hardcoded constants.
    lifecycle_params = None
    if lifecycle_mode == "active":
        from simulation.components.lifecycle_params import LifecycleParameters
        lifecycle_params = LifecycleParameters.from_file(
            active_inputs_path or str(ACTIVE_INPUTS_PATH))

    # Section 1.6: resource availability. "calendar" loads the model fitted in
    # notebooks/01_resource_availability.ipynb — per-resource shifts, discovered
    # public holidays, and sampled vacation. "always" leaves every resource on
    # duty around the clock, which is the pre-1.6 behaviour and the baseline the
    # calendar is measured against.
    #
    # roster_seed additionally rolls the fitted p_work (does this resource work
    # this weekday at all?), which takes the Monday-morning workforce from ~123
    # to ~37 and is what makes contention real. None = off, the pre-rostering
    # behaviour. It is a run parameter, so it is set here and not read from the
    # serialised calendar.
    calendar = None
    if availability == "calendar":
        from analysis.availability import YearlyAvailability
        calendar = YearlyAvailability.from_json(
            AVAILABILITY_MODEL_PATH, roster_seed=roster_seed)

    # Section 1.7: who may perform what.
    #   "orgmodel" — the OrdinoR organizational model (Advanced): resources
    #                inherit their group's capabilities, which may be conditioned
    #                on the case type and the day of the week.
    #   "observed" — the resource x activity matrix (Basic): permitted iff seen.
    #   "hardcoded"— the original top-20 map, kept as the baseline.
    perms = None
    case_attrs = None
    if permissions == "orgmodel":
        perms = perm_models.OrgModelPermissions.from_json(ORGMODEL_PATH)
        perms.self_check()   # a vocabulary mismatch would permit nothing, silently
        # The org model can gate on case type, so cases need one.
        case_attrs = CaseAttributeSampler.from_json(
            CASE_ATTRIBUTES_PATH, seed=RANDOM_SEED)
    elif permissions == "observed":
        perms = perm_models.StaticPermissions.from_json(OBSERVED_PERMS_PATH)

    if USE_MDN_ARRIVALS:
        arrivals = MDNArrivalComponent(seed=RANDOM_SEED, start_datetime=START_DATETIME)
    else:
        arrivals = ArrivalComponent(seed=RANDOM_SEED)
    process   = ProcessComponent(seed=RANDOM_SEED)
    resources = ResourceComponent(
        capacity_per_resource=capacity,
        seed=RANDOM_SEED,
        calendar=calendar,
        start_datetime=START_DATETIME,
        permissions=perms,      # Section 1.7: who may perform what
        piled=piled_execution,  # Piled Execution (R-PE, Pattern 38) — default off
        batching_k=k_batching,  # k-Batching (Zeng & Zhao) — default off, exclusive w/ piled
        duration_model_path=model_path if k_batching else None,
        lifecycle_mode=lifecycle_mode,
        lifecycle_params=lifecycle_params,
    )

    process_kwargs = dict(
        seed=RANDOM_SEED,
        mode=mode,
        model_path=model_path,
        start_datetime=START_DATETIME,   # anchor for day_of_week / hour_of_day
        resource_component=resources,    # so resources are released on complete
        case_attributes=case_attrs,      # Section 1.7: case types for the org model
        lifecycle_mode=lifecycle_mode,   # legacy | active (§4.4)
        lifecycle_params=lifecycle_params,
    )
    if process_model == "advanced":
        process = PetriNetProcessComponent(
            bpmn_path=bpmn_path or str(DEFAULT_BPMN_PATH),
            branching_mode=branching_mode,
            decision_rules_path=decision_rules_path or str(DEFAULT_DECISION_RULES_PATH),
            enforce_terminal_outcomes=enforce_terminal_outcomes,
            **process_kwargs,
        )
    else:
        if branching_mode != "probs":
            raise ValueError(
                f"--branching-mode {branching_mode} requires --process-model "
                "advanced: decision points are only meaningful on the Petri net."
            )
        process = ProcessComponent(**process_kwargs)

    # Register ResourceComponent BEFORE ProcessComponent: both handle
    # ACTIVITY_START and handlers fire in registration order, so the resource
    # must be allocated (event.resource populated) *before* ProcessComponent
    # samples the duration — otherwise the ML resource feature is always unknown.
    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)
    tracker = CaseCompletionTracker()
    engine.register(tracker)

    arrivals.bootstrap(engine)
    engine.run()

    engine.logger.save(OUTPUT_PATH)
    completed_path = OUTPUT_PATH.parent / "completed_cases.txt"
    completed_path.write_text(
        "\n".join(sorted(tracker.completed_case_ids)), encoding="utf-8")
    print(f"[main] {len(tracker.completed_case_ids)} naturally-completed case "
          f"ids -> {completed_path}")

    print("\n--- Simulation Statistics ---")
    print(f"  process_model: {process_model}")
    print(f"  branching_mode: {branching_mode}")
    print(f"  processing_time_mode: {mode}")
    print(f"  piled_execution: {piled_execution}")
    print(f"  k_batching: {k_batching}")
    for k, v in engine.stats.items():
        print(f"  {k}: {v}")
    print(f"  events_logged: {engine.logger.num_events}")
    print(f"\n  Expected from BPIC-17 in {SIM_DURATION_DAYS:.1f} days: "
          f"~{SIM_DURATION_DAYS * 86.09:.0f} cases (~86.09/day)")

    rstats = resources.stats()
    print("\n--- Resource pool (Sections 1.6-1.8) ---")
    print(f"  availability model:       {availability}")
    print(f"  permission model:         {permissions}")
    print(f"  resources in pool:        {len(resources.permissions.resources())}")
    print(f"  work items started:       {rstats['work_items_started']}")
    print(f"  mean wait for a resource: {rstats['mean_wait_seconds'] / 3600:.1f} h")
    print(f"  still queued at horizon:  {rstats['still_queued_at_end']}")
    print(f"  activities nobody may perform: {rstats['unpermitted_activities']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BPIC-17 simulation.")
    parser.add_argument(
        "--mode", default="distribution",
        choices=["distribution", "ml_model", "ml_probabilistic"],
        help="Processing-time model (default: distribution).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to the trained joblib artifact (ML modes only). Defaults to the "
             "legacy artifact in legacy mode and the _active artifact in active mode.",
    )
    parser.add_argument(
        "--lifecycle-mode", default="legacy", choices=["legacy", "active"],
        help="'legacy' (default) reproduces today's single start->complete block "
             "bit-for-bit (five-column log). 'active' runs the W_ work-item "
             "suspend/resume state machine with active-service-time durations "
             "(six-column log, work_item_id), loading the mined parameters from "
             "--active-inputs-path (implementationplan §3/§4.4).",
    )
    parser.add_argument(
        "--active-inputs-path", default=str(ACTIVE_INPUTS_PATH),
        help="Path to simulation_inputs_active.json (--lifecycle-mode active only).",
    )
    parser.add_argument(
        "--process-model", default="advanced",
        choices=["basic", "advanced"],
        help="Section 1.4: 'advanced' enforces control-flow via a Petri net "
             "loaded from --bpmn-path (default); 'basic' uses the flat "
             "next-activity probability graph.",
    )
    parser.add_argument(
        "--bpmn-path", default=str(DEFAULT_BPMN_PATH),
        help="Path to the .bpmn file (--process-model advanced only).",
    )
    parser.add_argument(
        "--availability", default="calendar", choices=["calendar", "always"],
        help="Section 1.6: 'calendar' enforces the fitted per-resource shifts, "
             "holidays and vacation (default); 'always' keeps every resource on "
             "duty around the clock (the baseline).",
    )
    parser.add_argument(
        "--permissions", default="orgmodel",
        choices=["orgmodel", "observed", "hardcoded"],
        help="Section 1.7: 'orgmodel' uses the OrdinoR organizational model "
             "(Advanced, default); 'observed' the learned resource x activity "
             "matrix (Basic); 'hardcoded' the original top-20 map (baseline).",
    )
    parser.add_argument(
        "--branching-mode", default=None,
        choices=["probs", "visit", "rules"],
        help="Section 1.5: 'probs' (default) uses BRANCHING_PROBS; 'visit' "
             "(A1 termination fix) conditions them on the activity's visit "
             "count within the case (branching_probs_by_visit in "
             "simulation_inputs.json); 'rules' (Advanced I) predicts each "
             "branch from case/runtime data attributes via a decision-point "
             "classifier trained by train_decision_rules.py. "
             "All non-default modes need --process-model advanced.",
    )
    parser.add_argument(
        "--decision-rules-path", default=str(DEFAULT_DECISION_RULES_PATH),
        help="Path to the trained joblib artifact (--branching-mode rules only).",
    )
    parser.add_argument(
        "--no-terminal-outcomes", action="store_true", default=False,
        help="Ablation toggle (--process-model advanced only): disable the "
             "domain-level rule that force-ends a case as soon as "
             "A_Pending/A_Denied/A_Cancelled fires. Default off (i.e. the "
             "rule is ON), matching the A1 fix in docs/ROADMAP.md.",
    )
    parser.add_argument(
        "--piled-execution", action="store_true", default=False,
        help="Enable Piled Execution (R-PE, Pattern 38): the deferred "
             "drain prefers a waiting task of the SAME activity type the "
             "resource just finished. One task per release (sequential). "
             "Default off. Mutually exclusive with --k-batching.",
    )
    parser.add_argument(
        "--k-batching", type=int, default=None, metavar="K",
        help="Enable k-Batching (Zeng & Zhao): work items never allocate "
             "immediately -- they queue and are released in batches of K, "
             "solved as a parallel-machines assignment problem minimising "
             "total expected processing time (simulation/expected_duration.py). "
             "A safety valve flushes early if the oldest waiting item has "
             "waited past 4h. Default off (None). Mutually exclusive with "
             "--piled-execution.",
    )
    parser.add_argument(
        "--roster-seed", type=int, default=None, metavar="N",
        help=f"Base seed for the p_work roster draw (does this resource work "
             f"this weekday at all?). Default {DEFAULT_ROSTER_SEED}. Rostering "
             f"is ON by default: without it the calendar fields ~123 people on "
             f"a Monday morning where the validated Section 1.6 model expects "
             f"~37. --availability calendar only.",
    )
    parser.add_argument(
        "--no-roster", action="store_true", default=False,
        help="Disable the p_work roster (the pre-rostering behaviour). Use to "
             "reproduce evidence logs generated before rostering landed; the "
             "workforce is then ~3.3x overstaffed and contention is not real.",
    )
    parser.add_argument(
        "--capacity", type=int, default=None, metavar="N",
        help=f"Work items one resource may hold at once. Default is derived "
             f"from --lifecycle-mode: {DEFAULT_CAPACITY_ACTIVE} for active "
             f"(98.4%% of real busy time is a single hands-on session, and "
             f"suspend/resume already models the interleaving), "
             f"{DEFAULT_CAPACITY_LEGACY} for legacy (whose durations are "
             f"elapsed spans that really do overlap, median peak 54). The "
             f"duration model has no concurrent-load feature, so N parallel "
             f"items each finish as fast as one.",
    )
    args = parser.parse_args()
    if args.capacity is not None and args.capacity < 1:
        parser.error("--capacity must be >= 1.")
    if args.piled_execution and args.k_batching is not None:
        parser.error("--piled-execution and --k-batching are mutually exclusive.")
    if args.roster_seed is not None and args.no_roster:
        parser.error("--roster-seed and --no-roster are mutually exclusive.")
    if args.roster_seed is not None and args.availability != "calendar":
        parser.error("--roster-seed requires --availability calendar.")
    # Explicit N wins; --no-roster disables; otherwise the default is ON.
    roster_seed = None if args.no_roster else (
        args.roster_seed if args.roster_seed is not None else DEFAULT_ROSTER_SEED)
    # Default branching: "visit" on the Petri net (A1 winner, see
    # output/validation/branching_probs_vs_rules/), plain "probs" on basic.
    branching_mode = args.branching_mode or (
        "visit" if args.process_model == "advanced" else "probs")
    # Artifact selection (§4.8): default the ML model path to the versioned _active
    # artifact in active mode, the legacy one otherwise; an explicit --model-path
    # always wins.
    model_path = args.model_path or (
        str(ACTIVE_MODEL_PATH) if args.lifecycle_mode == "active"
        else str(DEFAULT_MODEL_PATH))
    main(
        mode=args.mode,
        model_path=model_path,
        process_model=args.process_model,
        bpmn_path=args.bpmn_path,
        availability=args.availability,
        permissions=args.permissions,
        branching_mode=branching_mode,
        decision_rules_path=args.decision_rules_path,
        enforce_terminal_outcomes=not args.no_terminal_outcomes,
        piled_execution=args.piled_execution,
        k_batching=args.k_batching,
        lifecycle_mode=args.lifecycle_mode,
        active_inputs_path=args.active_inputs_path,
        roster_seed=roster_seed,
        capacity=args.capacity,
    )

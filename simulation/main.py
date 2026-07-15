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
from simulation.components.resource import ResourceComponent

# ── Configuration ────────────────────────────────────────────────────────────

SIM_DURATION_DAYS    = 30
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
USE_MDN_ARRIVALS = False

# ── Build & run ───────────────────────────────────────────────────────────────

def main(
    mode: str = "distribution",
    model_path: str | None = None,
    process_model: str = "advanced",
    bpmn_path: str | None = None,
    availability: str = "calendar",
    branching_mode: str = "probs",
    decision_rules_path: str | None = None,
    piled_execution: bool = False,
    k_batching: int | None = None,
):
    engine = SimulationEngine(
        sim_duration=SIM_DURATION_SECONDS,
        start_datetime=START_DATETIME,
        verbose=False,   # set True to print every event (slow for large runs)
    )

    # Section 1.6: resource availability. "calendar" loads the model fitted in
    # notebooks/01_resource_availability.ipynb — per-resource shifts, discovered
    # public holidays, and sampled vacation. "always" leaves every resource on
    # duty around the clock, which is the pre-1.6 behaviour and the baseline the
    # calendar is measured against.
    calendar = None
    if availability == "calendar":
        from analysis.availability import YearlyAvailability
        calendar = YearlyAvailability.from_json(AVAILABILITY_MODEL_PATH)

    if USE_MDN_ARRIVALS:
        arrivals = MDNArrivalComponent(seed=RANDOM_SEED, start_datetime=START_DATETIME)
    else:
        arrivals = ArrivalComponent(seed=RANDOM_SEED)
    process   = ProcessComponent(seed=RANDOM_SEED)
    resources = ResourceComponent(
        capacity_per_resource=3,
        seed=RANDOM_SEED,
        calendar=calendar,
        start_datetime=START_DATETIME,
        piled=piled_execution,  # Piled Execution (R-PE, Pattern 38) — default off
        batching_k=k_batching,  # k-Batching (Zeng & Zhao) — default off, exclusive w/ piled
        duration_model_path=str(DEFAULT_MODEL_PATH) if k_batching else None,
    )

    process_kwargs = dict(
        seed=RANDOM_SEED,
        mode=mode,
        model_path=model_path,
        start_datetime=START_DATETIME,   # anchor for day_of_week / hour_of_day
        resource_component=resources,    # so resources are released on complete
    )
    if process_model == "advanced":
        process = PetriNetProcessComponent(
            bpmn_path=bpmn_path or str(DEFAULT_BPMN_PATH),
            branching_mode=branching_mode,
            decision_rules_path=decision_rules_path or str(DEFAULT_DECISION_RULES_PATH),
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
    print(f"\n  Expected from BPIC-17 in 30 days: ~2580 cases (~86/day)")

    rstats = resources.stats()
    print("\n--- Resource pool (Sections 1.6-1.8) ---")
    print(f"  availability model:       {availability}")
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
        "--model-path", default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained joblib artifact (ML modes only).",
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
    args = parser.parse_args()
    if args.piled_execution and args.k_batching is not None:
        parser.error("--piled-execution and --k-batching are mutually exclusive.")
    # Default branching: "visit" on the Petri net (A1 winner, see
    # output/validation/branching_probs_vs_rules/), plain "probs" on basic.
    branching_mode = args.branching_mode or (
        "visit" if args.process_model == "advanced" else "probs")
    main(
        mode=args.mode,
        model_path=args.model_path,
        process_model=args.process_model,
        bpmn_path=args.bpmn_path,
        availability=args.availability,
        branching_mode=branching_mode,
        decision_rules_path=args.decision_rules_path,
        piled_execution=args.piled_execution,
        k_batching=args.k_batching,
    )

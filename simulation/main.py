"""
main.py — Run the BPIC-17 simulation with real data-driven components.

Components used:
  - ArrivalComponent  (Section 1.2 Basic): LogNormal inter-arrivals from BPIC-17
  - ProcessComponent  (Section 1.3+1.5 Basic): fitted distributions + branching probs
  - PetriNetProcessComponent (Section 1.4 Advanced): loads a .bpmn file, converts
    it to a Petri net, and enforces control-flow via Petri net firing rules
    instead of the flat next-activity graph. Toggle with --process-model.
  - ResourceComponent (Section 1.7+1.8 Basic): permission map + random allocation
  - EventLogger       (Section 1.1 Basic): built-in, outputs CSV

Usage:
    cd simulation/
    PYTHONPATH=.. python main.py
    PYTHONPATH=.. python main.py --process-model basic
"""

import argparse
from datetime import datetime
from pathlib import Path

from simulation.core.engine import SimulationEngine
from simulation.components.arrival import ArrivalComponent
from simulation.components.arrival_mdn import MDNArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import ResourceComponent
from simulation.components import permissions as perm_models
from simulation.components.case_attributes import CaseAttributeSampler

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

# Section 1.7: permission models, fitted in
# notebooks/02_resource_permissions.ipynb.
ORGMODEL_PATH        = REPO_ROOT / "models" / "permissions_orgmodel.json"
OBSERVED_PERMS_PATH  = REPO_ROOT / "models" / "permissions_observed.json"
CASE_ATTRIBUTES_PATH = REPO_ROOT / "models" / "case_attributes.json"

RANDOM_SEED = 42   # Fix for reproducibility — required by assignment grading!

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
    permissions: str = "orgmodel",
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
        capacity_per_resource=3,
        seed=RANDOM_SEED,
        calendar=calendar,
        start_datetime=START_DATETIME,
        permissions=perms,
    )

    process_kwargs = dict(
        seed=RANDOM_SEED,
        mode=mode,
        model_path=model_path,
        start_datetime=START_DATETIME,   # anchor for day_of_week / hour_of_day
        resource_component=resources,    # so resources are released on complete
        case_attributes=case_attrs,      # Section 1.7: case types for the org model
    )
    if process_model == "advanced":
        process = PetriNetProcessComponent(
            bpmn_path=bpmn_path or str(DEFAULT_BPMN_PATH), **process_kwargs
        )
    else:
        process = ProcessComponent(**process_kwargs)

    # Register ResourceComponent BEFORE ProcessComponent: both handle
    # ACTIVITY_START and handlers fire in registration order, so the resource
    # must be allocated (event.resource populated) *before* ProcessComponent
    # samples the duration — otherwise the ML resource feature is always unknown.
    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)

    arrivals.bootstrap(engine)
    engine.run()

    engine.logger.save(OUTPUT_PATH)

    print("\n--- Simulation Statistics ---")
    print(f"  process_model: {process_model}")
    print(f"  processing_time_mode: {mode}")
    for k, v in engine.stats.items():
        print(f"  {k}: {v}")
    print(f"  events_logged: {engine.logger.num_events}")
    print(f"\n  Expected from BPIC-17 in 30 days: ~2580 cases (~86/day)")

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
        "--permissions", default="orgmodel",
        choices=["orgmodel", "observed", "hardcoded"],
        help="Section 1.7: 'orgmodel' uses the OrdinoR organizational model "
             "(Advanced, default); 'observed' the learned resource x activity "
             "matrix (Basic); 'hardcoded' the original top-20 map (baseline).",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        model_path=args.model_path,
        process_model=args.process_model,
        bpmn_path=args.bpmn_path,
        availability=args.availability,
        permissions=args.permissions,
    )

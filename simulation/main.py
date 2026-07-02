"""
main.py — Run the BPIC-17 simulation with real data-driven components.

Components used:
  - ArrivalComponent  (Section 1.2 Basic): LogNormal inter-arrivals from BPIC-17
  - ProcessComponent  (Section 1.3+1.5 Basic): fitted distributions + branching probs
  - ResourceComponent (Section 1.7+1.8 Basic): permission map + random allocation
  - EventLogger       (Section 1.1 Basic): built-in, outputs CSV

Usage:
    cd simulation/
    PYTHONPATH=.. python main.py
"""

import argparse
from datetime import datetime
from pathlib import Path

from simulation.core.engine import SimulationEngine
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent

# ── Configuration ────────────────────────────────────────────────────────────

SIM_DURATION_DAYS    = 30
SIM_DURATION_SECONDS = SIM_DURATION_DAYS * 24 * 3600

# BPIC-17 starts 2016-01-01; anchor t=0 to the same date
START_DATETIME = datetime(2016, 1, 1)

OUTPUT_PATH = Path("output/event_log.csv")

# Default trained-model artifact for the ML processing-time modes
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "processing_time_model.joblib"

RANDOM_SEED = 42   # Fix for reproducibility — required by assignment grading!

# ── Build & run ───────────────────────────────────────────────────────────────

def main(mode: str = "distribution", model_path: str | None = None):
    engine = SimulationEngine(
        sim_duration=SIM_DURATION_SECONDS,
        start_datetime=START_DATETIME,
        verbose=False,   # set True to print every event (slow for large runs)
    )

    arrivals  = ArrivalComponent(seed=RANDOM_SEED)
    resources = ResourceComponent(capacity_per_resource=3, seed=RANDOM_SEED)
    process   = ProcessComponent(
        seed=RANDOM_SEED,
        mode=mode,
        model_path=model_path,
        start_datetime=START_DATETIME,   # anchor for day_of_week / hour_of_day
        resource_component=resources,    # so resources are released on complete
    )

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
    print(f"  processing_time_mode: {mode}")
    for k, v in engine.stats.items():
        print(f"  {k}: {v}")
    print(f"  events_logged: {engine.logger.num_events}")
    print(f"\n  Expected from BPIC-17 in 30 days: ~2580 cases (~86/day)")


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
    args = parser.parse_args()
    main(mode=args.mode, model_path=args.model_path)

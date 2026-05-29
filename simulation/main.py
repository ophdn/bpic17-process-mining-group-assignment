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

RANDOM_SEED = 42   # Fix for reproducibility — required by assignment grading!

# ── Build & run ───────────────────────────────────────────────────────────────

def main():
    engine = SimulationEngine(
        sim_duration=SIM_DURATION_SECONDS,
        start_datetime=START_DATETIME,
        verbose=False,   # set True to print every event (slow for large runs)
    )

    arrivals  = ArrivalComponent(seed=RANDOM_SEED)
    process   = ProcessComponent(seed=RANDOM_SEED)
    resources = ResourceComponent(capacity_per_resource=3, seed=RANDOM_SEED)

    engine.register(arrivals)
    engine.register(process)
    engine.register(resources)

    arrivals.bootstrap(engine)
    engine.run()

    engine.logger.save(OUTPUT_PATH)

    print("\n--- Simulation Statistics ---")
    for k, v in engine.stats.items():
        print(f"  {k}: {v}")
    print(f"  events_logged: {engine.logger.num_events}")
    print(f"\n  Expected from BPIC-17 in 30 days: ~2580 cases (~86/day)")


if __name__ == "__main__":
    main()

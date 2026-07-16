"""
drain_analysis.py
=================
Horizon-censoring check for the 30-day KPI runs (Section 1.4/1.5 report):

With a hard 30-day horizon, only cases fast enough to finish inside the
window are counted as completed — real BPIC-17 cases take 21.8 days on
average, so a substantial share of the 29% "missing" completions in
advanced_visit.json may be censoring (survivorship), not model error, and
the completed-only case-duration statistic is biased toward fast cases.

This script isolates the effect: arrivals only during the first 30 days
(same arrival process, same seed), but the engine keeps running until day
180 so every started case can finish. Comparing the drained KPIs against
the 30-day-horizon KPIs quantifies how much of the completion/duration gap
the censoring explains.

Usage (from the repo root):
    python scripts/drain_analysis.py
Output:
    output/validation/horizon_censoring/drain.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import metrics  # noqa: E402

from simulation.core.engine import SimulationEngine  # noqa: E402
from simulation.core.events import EventType  # noqa: E402
from simulation.components.arrival import ArrivalComponent  # noqa: E402
from simulation.components.petri_process import PetriNetProcessComponent  # noqa: E402
from simulation.components.resource import ResourceComponent  # noqa: E402
from simulation.components.lifecycle_params import LifecycleParameters  # noqa: E402

ARRIVAL_DAYS = 30      # identical arrival window to the KPI baseline runs
DRAIN_DAYS = 180       # generous drain horizon so every case can finish
START_DATETIME = datetime(2016, 1, 1)
SEED = 42
BPMN = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
OUT = REPO_ROOT / "output" / "validation" / "horizon_censoring" / "drain.json"
LEGACY_REFERENCE = REPO_ROOT / "simulation_inputs.json"
ACTIVE_REFERENCE = REPO_ROOT / "simulation_inputs_active.json"


class CutoffArrivals(ArrivalComponent):
    """Same fitted arrival process, but stops spawning after the cutoff."""

    def __init__(self, cutoff_seconds: float, **kwargs):
        super().__init__(**kwargs)
        self._cutoff = cutoff_seconds

    def _schedule_next(self, engine, current_time: float) -> None:
        if current_time >= self._cutoff:
            return
        super()._schedule_next(engine, current_time)


class CompletionTracker:
    HANDLES = {EventType.CASE_COMPLETE: None}

    def __init__(self):
        self.completed = set()

    def on_case_complete(self, engine, event):
        self.completed.add(event.case_id)


CompletionTracker.HANDLES = {EventType.CASE_COMPLETE: CompletionTracker.on_case_complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-mode", default="legacy",
                        choices=["legacy", "active"])
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    reference_path = args.reference or (
        ACTIVE_REFERENCE if args.lifecycle_mode == "active" else LEGACY_REFERENCE)
    out_path = args.out or (
        OUT.with_name("drain_active.json")
        if args.lifecycle_mode == "active" else OUT)
    lifecycle_params = (
        LifecycleParameters.from_file(ACTIVE_REFERENCE)
        if args.lifecycle_mode == "active" else None)
    engine = SimulationEngine(sim_duration=DRAIN_DAYS * 24 * 3600,
                              start_datetime=START_DATETIME,
                              lifecycle_mode=args.lifecycle_mode)
    arrivals = CutoffArrivals(ARRIVAL_DAYS * 24 * 3600, seed=SEED)
    resources = ResourceComponent(capacity_per_resource=3, seed=SEED)
    process = PetriNetProcessComponent(
        bpmn_path=str(BPMN), branching_mode="visit", seed=SEED,
        resource_component=resources, start_datetime=START_DATETIME,
        lifecycle_mode=args.lifecycle_mode, lifecycle_params=lifecycle_params)
    tracker = CompletionTracker()
    for c in (arrivals, resources, process, tracker):
        engine.register(c)
    arrivals.bootstrap(engine)
    engine.run()

    df = pd.DataFrame(engine.logger._rows)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="ISO8601")
    started = df["case:concept:name"].nunique()
    df = df[df["case:concept:name"].isin(tracker.completed)]

    reference = metrics.load_reference(reference_path)
    m = metrics.evaluate(df, reference)   # no net: control-flow unchanged by draining

    result = {
        "design": ("arrivals limited to the first %d days (seed %d, identical "
                   "to the KPI baseline runs), engine drained until day %d so "
                   "no case is cut off by the horizon") % (ARRIVAL_DAYS, SEED, DRAIN_DAYS),
        "branching_mode": "visit",
        "lifecycle_mode": args.lifecycle_mode,
        "reference": str(reference_path.name),
        "cases_started": started,
        "cases_completed": len(tracker.completed),
        "completion_rate": round(len(tracker.completed) / max(started, 1), 4),
        "compare_to": "output/validation/branching_probs_vs_rules/advanced_visit.json "
                      "(same config, hard 30-day horizon)",
        "case_stats": m["case_stats"],
        "branching_mean_tvd": m["branching"]["mean_tvd"],
        "variants": m["variants"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    print(json.dumps(result, indent=2, default=float))
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()

"""
test_advanced_process_model.py
===============================
Verifies that PetriNetProcessComponent (Section 1.4 Advanced) actually
enforces control-flow, by running the simulation twice — once with the
Basic flat-probability ProcessComponent, once with the Advanced Petri-net
component — and replaying both resulting event logs against the same
Petri net (pm4py token-based replay / conformance checking).

What "correct" looks like
--------------------------
The Advanced run's own activities were *chosen* by walking the Petri net,
so every completed case should be a 100%-fitting trace of that net
(fitness ~1.0, perc_fit_traces ~1.0). The Basic run picks each next
activity from a flat per-activity probability table with no memory of
which branch of the net it is currently in, so it can (and empirically
does) produce sequences the net disallows — expect a visibly lower
fitness there. That gap is the proof the enforcement mechanism works.

Usage:
    cd simulation/
    PYTHONPATH=.. python ../scripts/test_advanced_process_model.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pm4py

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import ResourceComponent

BPMN_PATH = Path(__file__).resolve().parent.parent / "simulation" / "models" / "bpic17_process.bpmn"
SIM_DURATION_SECONDS = 30 * 24 * 3600
START_DATETIME = datetime(2016, 1, 1)
SEED = 42


class _CaseCompletionTracker:
    """Records which case_ids reach CASE_COMPLETE (i.e. finish naturally,
    not cut off by the simulation horizon)."""

    HANDLES = {EventType.CASE_COMPLETE: None}

    def __init__(self):
        self.completed_case_ids = set()

    def on_case_complete(self, engine, event):
        self.completed_case_ids.add(event.case_id)


_CaseCompletionTracker.HANDLES = {
    EventType.CASE_COMPLETE: _CaseCompletionTracker.on_case_complete
}


def run_sim(use_advanced: bool) -> pd.DataFrame:
    engine = SimulationEngine(sim_duration=SIM_DURATION_SECONDS, start_datetime=START_DATETIME)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(capacity_per_resource=3, seed=SEED)
    tracker = _CaseCompletionTracker()

    kwargs = dict(seed=SEED, resource_component=resources, start_datetime=START_DATETIME)
    if use_advanced:
        process = PetriNetProcessComponent(bpmn_path=str(BPMN_PATH), **kwargs)
    else:
        process = ProcessComponent(**kwargs)

    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)
    engine.register(tracker)

    arrivals.bootstrap(engine)
    engine.run()

    df = pd.DataFrame(engine.logger._rows)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    df = df[df["case:concept:name"].isin(tracker.completed_case_ids)]
    return df


def to_replay_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per real activity occurrence. The simulator never truncates
    mid-activity, so every activity always has both a 'start' and a
    'complete' row here; keeping only 'complete' gives exactly one row
    per occurrence.
    """
    return df[df["lifecycle:transition"] == "complete"].copy()


def main():
    bpmn_model = pm4py.read_bpmn(str(BPMN_PATH))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    print(f"Loaded Petri net: {len(net.places)} places, {len(net.transitions)} transitions")

    results = {}
    for label, use_advanced in [("basic", False), ("advanced", True)]:
        df = run_sim(use_advanced)
        replay_df = to_replay_log(df)
        n_cases = replay_df["case:concept:name"].nunique()

        fitness = pm4py.fitness_token_based_replay(
            replay_df, net, im, fm,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
        results[label] = (n_cases, fitness)

        print(f"\n=== {label} ===")
        print(f"  completed cases replayed: {n_cases}")
        print(f"  average_trace_fitness:    {fitness['average_trace_fitness']:.4f}")
        print(f"  log_fitness:              {fitness['log_fitness']:.4f}")
        print(f"  percentage_of_fitting_traces: {fitness['percentage_of_fitting_traces']:.2f}%")

    basic_fit = results["basic"][1]["percentage_of_fitting_traces"]
    adv_fit = results["advanced"][1]["percentage_of_fitting_traces"]
    print(f"\n{'='*60}")
    print(f"Basic:    {basic_fit:.2f}% of completed cases perfectly fit the Petri net")
    print(f"Advanced: {adv_fit:.2f}% of completed cases perfectly fit the Petri net")
    if adv_fit >= 99.0 and adv_fit > basic_fit:
        print("PASS: Advanced enforces the Petri net; Basic does not (as expected).")
    else:
        print("CHECK: Advanced should be ~100% and higher than Basic — investigate.")


if __name__ == "__main__":
    main()

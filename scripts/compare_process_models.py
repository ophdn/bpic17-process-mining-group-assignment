"""
compare_process_models.py
==========================
General-purpose regression/benchmark tool: runs the simulation once per
requested configuration and reports the full KPI suite from metrics.py
against the real BPIC-17 statistics in simulation_inputs.json.

Use this any time you change something in the simulation (a component,
a distribution fit, the BPMN model, ...) to check whether it moved the
simulated log closer to or further from reality — not just for the
Section 1.4 Basic-vs-Advanced comparison it started out as.

Default run compares Basic (flat probability graph) vs. Advanced
(Petri net-enforced control-flow) — see
docs/paper_insights_discovering_simulation_models.md for why these KPIs
were chosen (the "second pass" validation method from Rozinat et al.,
"Discovering Simulation Models").

Usage:
    cd simulation/
    PYTHONPATH=.. python ../scripts/compare_process_models.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pm4py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

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
    not cut off by the simulation horizon) — only these are fair to
    compare against real (complete) case statistics."""

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


def main():
    reference = metrics.load_reference()
    bpmn_model = pm4py.read_bpmn(str(BPMN_PATH))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    print(f"Loaded Petri net: {len(net.places)} places, {len(net.transitions)} transitions")

    results = {}
    for label, use_advanced in [("basic", False), ("advanced", True)]:
        df = run_sim(use_advanced)
        results[label] = metrics.evaluate(df, reference, net, im, fm)
        metrics.print_report(label, results[label])

    b, a = results["basic"], results["advanced"]
    print(f"\n{'=' * 70}")
    print("Basic vs. Advanced — Section 1.4 KPI summary")
    print(f"{'=' * 70}")
    print(f"{'KPI':38} {'basic':>12} {'advanced':>12}")
    print(f"{'control-flow fitting traces %':38} "
          f"{b['control_flow']['fitness']['percentage_of_fitting_traces']:>11.2f}% "
          f"{a['control_flow']['fitness']['percentage_of_fitting_traces']:>11.2f}%")
    print(f"{'control-flow precision':38} "
          f"{b['control_flow']['precision']:>12.4f} {a['control_flow']['precision']:>12.4f}")
    print(f"{'branching prob. mean TVD (lower=better)':38} "
          f"{b['branching']['mean_tvd']:>12} {a['branching']['mean_tvd']:>12}")
    print(f"{'top-20 real variants reproduced /20':38} "
          f"{b['variants']['ref_top20_variants_reproduced']:>12} "
          f"{a['variants']['ref_top20_variants_reproduced']:>12}")
    print(f"{'case length rel.err':38} "
          f"{b['case_stats']['case_length_rel_err']:>12} {a['case_stats']['case_length_rel_err']:>12}")
    print(f"{'case duration rel.err':38} "
          f"{b['case_stats']['case_duration_rel_err']:>12} {a['case_stats']['case_duration_rel_err']:>12}")

    adv_fit = a["control_flow"]["fitness"]["percentage_of_fitting_traces"]
    basic_fit = b["control_flow"]["fitness"]["percentage_of_fitting_traces"]
    if adv_fit >= 99.0 and adv_fit > basic_fit:
        print("\nPASS: Advanced enforces the Petri net; Basic does not (as expected).")
    else:
        print("\nCHECK: Advanced should be ~100% and higher than Basic — investigate.")


if __name__ == "__main__":
    main()

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

Usage (from the repo root):
    python scripts/compare_process_models.py
    python scripts/compare_process_models.py --bpmn simulation/models/foo.bpmn \
        --configs advanced --tag advanced_im02 \
        --out output/validation/bpmn_source_comparison

Every run dumps the full KPI dict per configuration as JSON into --out
(default: output/validation/process_model_comparison/) so design decisions
can be backed by numbers in the report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pm4py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/ → import metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root → import simulation
import metrics  # noqa: E402

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.resource import ResourceComponent

REPO_ROOT = Path(__file__).resolve().parent.parent
BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
DEFAULT_OUT = REPO_ROOT / "output" / "validation" / "process_model_comparison"
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


def run_sim(use_advanced: bool, bpmn_path: Path = BPMN_PATH,
            branching_mode: str = "probs") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (completed-cases DataFrame, unfiltered DataFrame, run stats
    incl. completion rate). The unfiltered frame is needed for
    metrics.arrival_rate_error, which must not be restricted to completed
    cases (see that function's docstring for why)."""
    engine = SimulationEngine(sim_duration=SIM_DURATION_SECONDS, start_datetime=START_DATETIME)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(capacity_per_resource=3, seed=SEED)
    tracker = _CaseCompletionTracker()

    kwargs = dict(seed=SEED, resource_component=resources, start_datetime=START_DATETIME)
    if use_advanced:
        process = PetriNetProcessComponent(
            bpmn_path=str(bpmn_path), branching_mode=branching_mode,
            decision_rules_path=str(REPO_ROOT / "simulation" / "models" / "decision_rules.joblib"),
            **kwargs)
    else:
        process = ProcessComponent(**kwargs)

    engine.register(arrivals)
    engine.register(resources)
    engine.register(process)
    engine.register(tracker)

    arrivals.bootstrap(engine)
    engine.run()

    df_all = pd.DataFrame(engine.logger._rows)
    # format="ISO8601": isoformat() drops the .%f part when microsecond == 0,
    # so the column mixes two ISO variants — strict parsing would crash.
    df_all["time:timestamp"] = pd.to_datetime(df_all["time:timestamp"], format="ISO8601")
    df = df_all[df_all["case:concept:name"].isin(tracker.completed_case_ids)]
    stats = {
        "cases_started": engine.stats.get("cases_started"),
        "cases_completed": engine.stats.get("cases_completed"),
        "completion_rate": round(
            engine.stats.get("cases_completed", 0)
            / max(engine.stats.get("cases_started", 1), 1), 4),
        "sim_duration_days": SIM_DURATION_SECONDS // 86400,
        "seed": SEED,
    }
    return df, df_all, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpmn", type=Path, default=BPMN_PATH,
                        help="BPMN model to enforce (advanced config) and to "
                             "measure control-flow fitness/precision against.")
    parser.add_argument("--configs", default="basic,advanced",
                        help="Comma-separated subset of: basic,advanced")
    parser.add_argument("--branching-mode", default="probs",
                        choices=["probs", "visit", "rules"],
                        help="Branching strategy for the advanced config "
                             "(see simulation/main.py --branching-mode).")
    parser.add_argument("--tag", default="",
                        help="Suffix for the JSON filenames, e.g. 'im02'.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Directory for the per-config KPI JSON dumps.")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    reference = metrics.load_reference()
    bpmn_model = pm4py.read_bpmn(str(args.bpmn))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    print(f"Loaded Petri net from {args.bpmn.name}: "
          f"{len(net.places)} places, {len(net.transitions)} transitions")

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for label in configs:
        df, df_all, run_stats = run_sim(label == "advanced", bpmn_path=args.bpmn,
                                        branching_mode=args.branching_mode)
        results[label] = metrics.evaluate(df, reference, net, im, fm, df_all=df_all)
        results[label]["run_stats"] = run_stats
        results[label]["config"] = {
            "process_model": label,
            "bpmn": str(args.bpmn.name),
            "branching_mode": args.branching_mode if label == "advanced" else "probs",
            "processing_time_mode": "distribution",
        }
        metrics.print_report(label, results[label])
        print(f"  completion rate:             {run_stats['completion_rate']} "
              f"({run_stats['cases_completed']}/{run_stats['cases_started']} cases)")

        fname = f"{label}{'_' + args.tag if args.tag else ''}.json"
        with open(args.out / fname, "w", encoding="utf-8") as f:
            json.dump(results[label], f, indent=2, default=float)
        print(f"  [save] KPIs -> {args.out / fname}")

    if {"basic", "advanced"} <= results.keys():
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
        print(f"{'completion rate':38} "
              f"{b['run_stats']['completion_rate']:>12} {a['run_stats']['completion_rate']:>12}")

        adv_fit = a["control_flow"]["fitness"]["percentage_of_fitting_traces"]
        basic_fit = b["control_flow"]["fitness"]["percentage_of_fitting_traces"]
        if adv_fit >= 99.0 and adv_fit > basic_fit:
            print("\nPASS: Advanced enforces the Petri net; Basic does not (as expected).")
        else:
            print("\nCHECK: Advanced should be ~100% and higher than Basic — investigate.")


if __name__ == "__main__":
    main()

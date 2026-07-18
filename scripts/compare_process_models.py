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
from simulation.components.lifecycle_params import LifecycleParameters
from simulation.components.case_attributes import CaseAttributeSampler
from simulation.components import permissions as perm_models

REPO_ROOT = Path(__file__).resolve().parent.parent
BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
LEGACY_REFERENCE_PATH = REPO_ROOT / "simulation_inputs.json"
ACTIVE_REFERENCE_PATH = REPO_ROOT / "simulation_inputs_active.json"
DEFAULT_OUT = REPO_ROOT / "output" / "validation" / "process_model_comparison"
# 99th percentile of real case duration (data/BPIChallenge2017.xes.gz, events
# filtered to lifecycle='complete', per-case duration = last_ts - first_ts):
# 5,110,843.8s ~= 59.15 days, rounded to a clean 60 days -- replaces the
# previous arbitrary 30-day horizon. See simulation/main.py's
# SIM_DURATION_SECONDS comment and docs/ROADMAP.md for the derivation and
# the horizon-censoring caveat (arrivals still span the whole window; this
# does not by itself eliminate censoring).
SIM_DURATION_SECONDS = 60 * 24 * 3600
START_DATETIME = datetime(2016, 1, 1)
SEED = 42
# Section 1.7 permission models -- "orgmodel" (144 resources, OrdinoR-mined
# groups/capabilities) is simulation/main.py's own default and the intended
# Advanced config; "hardcoded" is the original top-20 map (17 resources) this
# harness silently fell back to when no --permissions was passed, which
# starved completion under sustained load (see docs/ROADMAP.md, A1-Update
# Teil 6) without meaningfully changing the branching/control-flow findings.
ORGMODEL_PATH = REPO_ROOT / "models" / "permissions_orgmodel.json"
OBSERVED_PERMS_PATH = REPO_ROOT / "models" / "permissions_observed.json"
CASE_ATTRIBUTES_PATH = REPO_ROOT / "models" / "case_attributes.json"


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
            branching_mode: str = "probs",
            lifecycle_mode: str = "legacy",
            enforce_terminal_outcomes: bool = True,
            permissions: str = "orgmodel") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (completed-cases DataFrame, unfiltered DataFrame, run stats
    incl. completion rate). The unfiltered frame is needed for
    metrics.arrival_rate_error, which must not be restricted to completed
    cases (see that function's docstring for why).

    *permissions*: {"orgmodel", "observed", "hardcoded"} -- see
    simulation/main.py --permissions. Default "orgmodel" (144 resources)
    matches main.py's own default and the intended Advanced config;
    "hardcoded" is the original 17-resource top-20 map this harness silently
    used before this parameter existed (docs/ROADMAP.md, A1-Update Teil 6).
    """
    lifecycle_params = (
        LifecycleParameters.from_file(ACTIVE_REFERENCE_PATH)
        if lifecycle_mode == "active" else None
    )
    engine = SimulationEngine(
        sim_duration=SIM_DURATION_SECONDS, start_datetime=START_DATETIME,
        lifecycle_mode=lifecycle_mode)
    arrivals = ArrivalComponent(seed=SEED)

    perms = None
    case_attrs = None
    if permissions == "orgmodel":
        perms = perm_models.OrgModelPermissions.from_json(ORGMODEL_PATH)
        perms.self_check()
        case_attrs = CaseAttributeSampler.from_json(CASE_ATTRIBUTES_PATH, seed=SEED)
    elif permissions == "observed":
        perms = perm_models.StaticPermissions.from_json(OBSERVED_PERMS_PATH)
    elif permissions != "hardcoded":
        raise ValueError(
            f"permissions must be 'orgmodel', 'observed' or 'hardcoded', got {permissions!r}")

    resources = ResourceComponent(capacity_per_resource=3, seed=SEED, permissions=perms)
    tracker = _CaseCompletionTracker()

    kwargs = dict(
        seed=SEED, resource_component=resources, start_datetime=START_DATETIME,
        lifecycle_mode=lifecycle_mode, lifecycle_params=lifecycle_params,
        case_attributes=case_attrs)
    if use_advanced:
        process = PetriNetProcessComponent(
            bpmn_path=str(bpmn_path), branching_mode=branching_mode,
            decision_rules_path=str(REPO_ROOT / "simulation" / "models" / "decision_rules.joblib"),
            enforce_terminal_outcomes=enforce_terminal_outcomes,
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
        "sim_duration_days": round(SIM_DURATION_SECONDS / 86400, 2),
        "seed": SEED,
        "lifecycle_mode": lifecycle_mode,
    }
    if use_advanced:
        stats["petri_debug"] = process.debug_stats()
    return df, df_all, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpmn", type=Path, default=BPMN_PATH,
                        help="BPMN model to enforce (advanced config) and to "
                             "measure control-flow fitness/precision against.")
    parser.add_argument("--configs", default="basic,advanced",
                        help="Comma-separated subset of: basic,advanced")
    parser.add_argument("--branching-mode", default="visit",
                        choices=["probs", "visit", "rules"],
                        help="Branching strategy for the advanced config "
                             "(see simulation/main.py --branching-mode). "
                             "Default 'visit' matches the A1 fix / simulation/"
                             "main.py's default for --process-model advanced.")
    parser.add_argument("--tag", default="",
                        help="Suffix for the JSON filenames, e.g. 'im02'.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Directory for the per-config KPI JSON dumps.")
    parser.add_argument("--lifecycle-mode", default="legacy",
                        choices=["legacy", "active"],
                        help="Lifecycle and reference baseline (default: legacy).")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Override the reference JSON. Defaults to simulation_inputs.json "
                             "for legacy and simulation_inputs_active.json for active.")
    parser.add_argument("--terminal-outcomes", default="on", choices=["on", "off"],
                        help="Ablation toggle (advanced config only): 'on' (default) "
                             "force-ends a case as soon as A_Pending/A_Denied/A_Cancelled "
                             "fires; 'off' relies solely on final_marking/__END__/loop_guard.")
    parser.add_argument("--permissions", default="orgmodel",
                        choices=["orgmodel", "observed", "hardcoded"],
                        help="Section 1.7 resource permission model (see simulation/"
                             "main.py --permissions). Default 'orgmodel' (144 resources, "
                             "OrdinoR-mined) matches main.py's own default; 'hardcoded' "
                             "is the original 17-resource top-20 map this harness used "
                             "before this flag existed.")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    reference_path = args.reference or (
        ACTIVE_REFERENCE_PATH if args.lifecycle_mode == "active"
        else LEGACY_REFERENCE_PATH)
    reference = metrics.load_reference(reference_path)
    bpmn_model = pm4py.read_bpmn(str(args.bpmn))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    print(f"Loaded Petri net from {args.bpmn.name}: "
          f"{len(net.places)} places, {len(net.transitions)} transitions")

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for label in configs:
        df, df_all, run_stats = run_sim(label == "advanced", bpmn_path=args.bpmn,
                                        branching_mode=args.branching_mode,
                                        lifecycle_mode=args.lifecycle_mode,
                                        enforce_terminal_outcomes=args.terminal_outcomes == "on",
                                        permissions=args.permissions)
        results[label] = metrics.evaluate(df, reference, net, im, fm, df_all=df_all)
        results[label]["run_stats"] = run_stats
        results[label]["config"] = {
            "process_model": label,
            "bpmn": str(args.bpmn.name),
            "branching_mode": args.branching_mode if label == "advanced" else "probs",
            "terminal_outcomes": args.terminal_outcomes if label == "advanced" else "n/a",
            "permissions": args.permissions,
            "processing_time_mode": "distribution",
            "lifecycle_mode": args.lifecycle_mode,
            "reference": str(reference_path.name),
        }
        metrics.print_report(label, results[label])
        print(f"  completion rate:             {run_stats['completion_rate']} "
              f"({run_stats['cases_completed']}/{run_stats['cases_started']} cases)")
        petri_debug = run_stats.get("petri_debug")
        if petri_debug:
            reasons = petri_debug["end_reasons"]
            print("  Petri end reasons:"
                  f" final_marking={reasons.get('final_marking', 0)},"
                  f" __END__={reasons.get('end_label', 0)},"
                  f" terminal_outcome={reasons.get('terminal_outcome', 0)},"
                  f" loop_guard={reasons.get('loop_guard', 0)}")
            print("  Petri end diagnostics:"
                  f" allow_end={petri_debug['allow_end_opportunities']},"
                  f" allow_end_without_dp={petri_debug['allow_end_without_dp']},"
                  f" terminal_continuation_end={reasons.get('terminal_continuation_end', 0)},"
                  f" dead_marking={reasons.get('dead_marking', 0)}")

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
        print(f"{'  ...ignoring never-simulated steps /20':38} "
              f"{b['variants']['ref_top20_variants_reproduced_ignoring_absent_activities']:>12} "
              f"{a['variants']['ref_top20_variants_reproduced_ignoring_absent_activities']:>12}")
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

"""
discover_process_model.py
=========================
Design decision, Section 1.4: should the simulation enforce the manually
modeled (Signavio) BPMN, or a model discovered from the event log?

This script produces the evidence:

1. Discovers a BPMN from the real BPIC-17 log with the Inductive Miner
   (noise threshold configurable) and saves it next to the existing model
   as `simulation/models/bpic17_process_im{noise}.bpmn`.
2. Replays the REAL log against both Petri nets (token-based replay) —
   "what share of real behavior can each model represent at all?" A model
   that cannot replay reality cannot produce a realistic event log, no
   matter how well the rest of the simulation is tuned.
3. Dumps the numbers to output/validation/bpmn_source_comparison/
   real_log_replay{tag}.json.

Afterwards, simulate + compare the full KPI suite per model with:
    python scripts/compare_process_models.py --configs advanced \
        --bpmn simulation/models/bpic17_process_im02.bpmn --tag im02 \
        --out output/validation/bpmn_source_comparison

Usage (from the repo root; loading the XES takes a few minutes):
    python scripts/discover_process_model.py --log data/BPIChallenge2017.xes.gz
    python scripts/discover_process_model.py --log ... --noise 0.4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pm4py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXISTING_BPMN = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
OUT_DIR = REPO_ROOT / "output" / "validation" / "bpmn_source_comparison"


def replay_fitness(df, net, im, fm, label: str) -> dict:
    print(f"[replay] Token-based replay of the real log on '{label}' …")
    t0 = time.perf_counter()
    fit = pm4py.fitness_token_based_replay(
        df, net, im, fm,
        activity_key="concept:name", timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )
    print(f"[replay] {label}: {fit['percentage_of_fitting_traces']:.2f}% fully-fitting "
          f"traces, avg trace fitness {fit['average_trace_fitness']:.4f} "
          f"({time.perf_counter() - t0:.0f}s)")
    return {
        "pct_fully_fitting_traces": round(fit["percentage_of_fitting_traces"], 2),
        "average_trace_fitness": round(fit["average_trace_fitness"], 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--noise", type=float, default=0.2,
                        help="Inductive Miner noise threshold (default 0.2).")
    args = parser.parse_args()

    print(f"[load] Reading {args.log} …")
    df = pm4py.read_xes(str(args.log))
    # Reduce to one row per activity occurrence, matching how the simulation
    # logs and how simulation_inputs.json was extracted.
    if "lifecycle:transition" in df.columns:
        df = df[df["lifecycle:transition"].astype(str).str.lower() == "complete"]
    print(f"[load] {len(df):,} complete-events, "
          f"{df['case:concept:name'].nunique():,} cases.")

    tag = f"im{str(args.noise).replace('.', '')}"
    out_bpmn = REPO_ROOT / "simulation" / "models" / f"bpic17_process_{tag}.bpmn"

    print(f"[discover] Inductive Miner (noise={args.noise}) …")
    t0 = time.perf_counter()
    bpmn = pm4py.discover_bpmn_inductive(df, noise_threshold=args.noise)
    pm4py.write_bpmn(bpmn, str(out_bpmn))
    print(f"[discover] Done in {time.perf_counter() - t0:.0f}s → {out_bpmn}")

    results = {"noise_threshold": args.noise, "discovered_bpmn": out_bpmn.name}
    for label, path in [("manual_signavio", EXISTING_BPMN), (tag, out_bpmn)]:
        model = pm4py.read_bpmn(str(path))
        net, im, fm = pm4py.convert_to_petri_net(model)
        results[label] = replay_fitness(df, net, im, fm, label)
        results[label]["net_size"] = f"{len(net.places)}p/{len(net.transitions)}t"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"real_log_replay_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[save] → {out_json}")


if __name__ == "__main__":
    main()

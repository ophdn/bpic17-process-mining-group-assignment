"""
eval_real_log.py
=================
Runs the exact same KPI suite (scripts/metrics.py::evaluate) that
compare_process_models.py uses for Basic/Advanced, but on the real
BPIC-17 log itself instead of a simulated one -- so all three
(basic.json, advanced.json, real_log.json) are directly comparable,
computed via the identical code path.

Since the log IS the source of the reference statistics, most KPIs are
expected to come out near-perfect (TVD ~0, variant coverage 20/20, case
length/duration/arrival-rate errors ~0) -- the informative numbers are
control-flow fitness and precision against simulation/models/
bpic17_process.bpmn, which the log was never fit *to* the same way a
simulated log is.

Usage (from the repo root):
    python scripts/eval_real_log.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pm4py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))
import metrics  # noqa: E402
import extract_log_info as eli  # noqa: E402

LOG_PATH = REPO_ROOT.parent / "data" / "BPIChallenge2017.xes.gz"
BPMN_PATH = REPO_ROOT.parent / "simulation" / "models" / "bpic17_process.bpmn"
REFERENCE_PATH = REPO_ROOT.parent / "simulation_inputs.json"
OUT_PATH = REPO_ROOT.parent / "output" / "validation" / "process_model_comparison" / "real_log.json"


def main():
    print(f"[load] Reading log: {LOG_PATH} ...")
    df = eli.load_log(LOG_PATH)
    df = df.rename(columns={
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp",
        "lifecycle": "lifecycle:transition",
    })
    print(f"[load] {df['case:concept:name'].nunique():,} cases, {len(df):,} events.")

    bpmn_model = pm4py.read_bpmn(str(BPMN_PATH))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    print(f"[load] Petri net: {len(net.places)} places, {len(net.transitions)} transitions")

    reference = metrics.load_reference(REFERENCE_PATH)
    result = metrics.evaluate(df, reference, net, im, fm, df_all=df)
    result["log_stats"] = {
        "n_cases": int(df["case:concept:name"].nunique()),
        "n_events": int(len(df)),
        "source": str(LOG_PATH.name),
    }
    result["config"] = {
        "process_model": "real_log",
        "bpmn": BPMN_PATH.name,
        "reference": REFERENCE_PATH.name,
        "note": ("Computed via the identical metrics.evaluate() code path as "
                 "compare_process_models.py's basic/advanced, applied to the real "
                 "log instead of a simulated one, so all three are directly "
                 "comparable. TVD/case-stats/arrival-rate/variants are expected "
                 "near-perfect since the reference IS derived from this log; "
                 "fitness/precision against bpic17_process.bpmn are the "
                 "genuinely informative numbers here."),
    }

    metrics.print_report("real_log", result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n[save] -> {OUT_PATH}")


if __name__ == "__main__":
    main()

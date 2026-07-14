"""
Sweep the OrdinoR design space on BPIC-17 and score every model.

Reproduces the experiment grid of Yang et al. (2022), §6.2: three ways of
learning execution contexts x two clustering methods x two profiling strategies.
Writes results to models/org_model_sweep.json and the best model (by F1) to
models/permissions_orgmodel.json.

Run:  PYTHONPATH=. .venv/bin/python scripts/sweep_org_models.py
"""

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from analysis.loader import load_events
from analysis import permissions as P

OUT = Path("models")
SWEEP = OUT / "org_model_sweep.json"
BEST = OUT / "permissions_orgmodel.json"

# The paper's published BPIC-17 numbers, for validation (Tables 7-10).
PAPER = {
    ("ATonly", "AHC", "OverallScore"): (24, 10, 0.923, 0.530, 0.673),
    ("CT+AT+TT(ca)", "AHC", "OverallScore"): (2020, 10, 0.810, 0.629, 0.708),
    ("CT+AT+TT(tc)", "AHC", "OverallScore"): (1884, 10, 0.831, 0.641, 0.724),
    ("CT+AT+TT(tc)", "MOC", "OverallScore"): (1884, 8, 0.957, 0.406, 0.571),
    ("CT+AT+TT(tc)", "AHC", "FullRecall"): (1884, 10, 1.000, 0.169, 0.290),
}


def main():
    OUT.mkdir(exist_ok=True)

    el = P.prepare_log(load_events())
    print(f"[log] {len(el):,} events  "
          f"{el['case:concept:name'].nunique():,} cases  "
          f"{el['concept:name'].nunique()} activities  "
          f"{el['org:resource'].nunique()} resources", flush=True)

    part = P.trace_clustering_partition(el, OUT / "trace_clusters.tsv")
    print(f"[tc]  wrote {part} ({P.N_TRACE_CLUSTERS} clusters)", flush=True)

    results = []
    for contexts in ["ATonly", "CT+AT+TT(ca)", "CT+AT+TT(tc)"]:
        t0 = time.time()
        _, rl = P.build_resource_log(el, contexts, partition_file=part)
        print(f"\n[ctx] {contexts}: resource log in {time.time()-t0:.0f}s", flush=True)

        for discovery in ["AHC", "MOC"]:
            for profiling in ["FullRecall", "OverallScore"]:
                t = time.time()
                m = P.discover(rl, n_groups=10, discovery=discovery,
                               profiling=profiling, contexts=contexts)
                row = m.as_row()
                results.append(row)

                ref = PAPER.get((contexts, discovery, profiling))
                tag = ""
                if ref:
                    tag = (f"   | paper: ctx={ref[0]} k={ref[1]} "
                           f"F={ref[2]:.3f} P={ref[3]:.3f} F1={ref[4]:.3f}")
                print(f"  {discovery:3s} {profiling:12s} "
                      f"ctx={row['#contexts']:5d} k={row['#groups']:2d} "
                      f"F={row['fitness']:.3f} P={row['precision']:.3f} "
                      f"F1={row['F1']:.3f}  ({time.time()-t:.0f}s){tag}", flush=True)

                SWEEP.write_text(json.dumps(results, indent=1))

                if row["F1"] == max(r["F1"] for r in results):
                    P.org_model_to_json(m, BEST)

    print(f"\n[done] {len(results)} models -> {SWEEP}")
    best = max(results, key=lambda r: r["F1"])
    print(f"[best] {best}  -> {BEST}")


if __name__ == "__main__":
    main()

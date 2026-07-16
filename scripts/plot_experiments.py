"""
plot_experiments.py — figures for the Part II policy comparison
================================================================
Turns the CSVs written by scripts/run_experiments.py into the figures the
report (and the roadmap's Phase B/C/D) asks for. Reads one or more
``results_<scenario>.csv`` files and, if present next to them, the
matching ``aggregate_<scenario>.csv`` (for CI error bars). Writes PNGs
into --out (default report/figures/).

Figures produced:
  - boxplot_<metric>_<scenario>.png   one box per policy, seeds as the
      distribution (cycle-time metrics converted to DAYS). Phase C.
  - pareto_<scenario>.png             mean cycle time (x, days) vs.
      resource fairness (y, lower = fairer), one labeled point per
      policy, 95% CI error bars from the aggregate CSV if available.
      Phase D efficiency-vs-fairness Pareto view.
  - robustness_cycle_time.png         (only if >1 scenario given) grouped
      bars: mean cycle time per policy across scenarios. RQ4.

No titles are baked into the PNGs — captions live in the LaTeX. Output is
deterministic (fixed figsize/dpi, sorted policy order) so re-running does
not churn the git diff.

Usage:
    cd <repo-root>
    PYTHONPATH=. .venv/bin/python scripts/plot_experiments.py \
        --results output/experiments/results_normal.csv \
        --out report/figures/

    # multiple scenarios -> also emits the robustness figure
    PYTHONPATH=. .venv/bin/python scripts/plot_experiments.py \
        --results output/experiments/results_normal.csv \
                  output/experiments/results_peak.csv \
                  output/experiments/results_outage.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never needs a display, deterministic
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "report" / "figures"

DPI = 150
DAY = 86400.0

# Metrics to box-plot: CSV column -> (nice label, divide-by to get plotted unit)
BOX_METRICS = {
    "avg_cycle_time_s": ("Avg. cycle time (days)", DAY),
    "p95_cycle_time_s": ("p95 cycle time (days)", DAY),
    "avg_resource_occupation": ("Avg. resource occupation", 1.0),
    "resource_fairness": ("Resource fairness (lower = fairer)", 1.0),
}


def policy_sort_key(p: str):
    """Deterministic, human order: random, piled, then kbatch by numeric k,
    then anything else alphabetically."""
    if p == "random":
        return (0, 0, "")
    if p == "piled":
        return (1, 0, "")
    m = re.match(r"kbatch(\d+)$", p)
    if m:
        return (2, int(m.group(1)), "")
    return (3, 0, p)


def ordered_policies(df: pd.DataFrame) -> list:
    return sorted(df["policy"].unique(), key=policy_sort_key)


def scenario_of(results_path: Path) -> str:
    m = re.match(r"results_(.+)\.csv$", results_path.name)
    return m.group(1) if m else results_path.stem


def load_aggregate(results_path: Path) -> pd.DataFrame | None:
    agg = results_path.with_name(results_path.name.replace("results_", "aggregate_"))
    return pd.read_csv(agg) if agg.exists() else None


def make_boxplots(df: pd.DataFrame, scenario: str, out_dir: Path) -> list:
    policies = ordered_policies(df)
    written = []
    for col, (label, divide) in BOX_METRICS.items():
        if col not in df.columns:
            continue
        data = [(df[df["policy"] == p][col] / divide).values for p in policies]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.boxplot(data, tick_labels=policies, showmeans=True)
        ax.set_ylabel(label)
        ax.set_xlabel("Allocation policy")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"boxplot_{col}_{scenario}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        written.append(path)
    return written


def make_pareto(df: pd.DataFrame, agg: pd.DataFrame | None, scenario: str,
                out_dir: Path) -> Path:
    policies = ordered_policies(df)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for p in policies:
        sub = df[df["policy"] == p]
        x = sub["avg_cycle_time_s"].mean() / DAY
        y = sub["resource_fairness"].mean()
        xerr = yerr = None
        if agg is not None:
            row = agg[agg["policy"] == p]
            if not row.empty:
                xerr = row["avg_cycle_time_s_ci95_halfwidth"].iloc[0] / DAY
                yerr = row["resource_fairness_ci95_halfwidth"].iloc[0]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3, markersize=6)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean cycle time (days) — lower is better")
    ax.set_ylabel("Resource fairness — lower is fairer")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"pareto_{scenario}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def make_robustness(frames: dict, out_dir: Path) -> Path:
    """Grouped bars: mean cycle time (days) per policy across scenarios."""
    import numpy as np

    scenarios = sorted(frames.keys())
    all_policies = sorted(
        {p for df in frames.values() for p in df["policy"].unique()},
        key=policy_sort_key,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    n = len(scenarios)
    width = 0.8 / max(n, 1)
    x = np.arange(len(all_policies))
    for i, sc in enumerate(scenarios):
        df = frames[sc]
        means = [
            (df[df["policy"] == p]["avg_cycle_time_s"].mean() / DAY)
            if p in set(df["policy"]) else 0.0
            for p in all_policies
        ]
        ax.bar(x + i * width, means, width, label=sc)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(all_policies, rotation=30)
    ax.set_ylabel("Mean cycle time (days)")
    ax.set_xlabel("Allocation policy")
    ax.legend(title="Scenario")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "robustness_cycle_time.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more results_<scenario>.csv files.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    written = []
    for r in args.results:
        rpath = Path(r)
        df = pd.read_csv(rpath)
        sc = scenario_of(rpath)
        frames[sc] = df
        agg = load_aggregate(rpath)
        written += make_boxplots(df, sc, out_dir)
        written.append(make_pareto(df, agg, sc, out_dir))

    if len(frames) > 1:
        written.append(make_robustness(frames, out_dir))

    print(f"Wrote {len(written)} figure(s) to {out_dir}:")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p}")


if __name__ == "__main__":
    main()

"""Evaluate saved active-lifecycle simulation logs against mined BPIC-17 inputs.

The general KPI suite lives in :mod:`scripts.metrics`. This companion adds the
work-item-keyed evidence that only exists in active mode: terminal outcomes,
active-session recomposition, suspend counts, and same-resource-on-resume.

Example
-------
    .venv/bin/python scripts/eval_lifecycle.py \
        --log output/event_log_active_distribution.csv \
        --completed output/completed_cases_active_distribution.txt \
        --label distribution --out output/validation/lifecycle_active/distribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402


TERMINALS = {"complete", "ate_abort", "withdraw"}


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def lifecycle_evidence(df: pd.DataFrame) -> dict:
    """Reconstruct W-item sessions exclusively by ``work_item_id``."""
    required = {"work_item_id", "lifecycle:transition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"active lifecycle log is missing columns: {sorted(missing)}")

    work = df[
        df["concept:name"].str.startswith("W_", na=False)
        & df["work_item_id"].notna()
        & df["work_item_id"].ne("")
    ].sort_values(["work_item_id", "time:timestamp"], kind="stable")

    records = []
    resumes = 0
    same_resource_resumes = 0
    resume_by_activity = defaultdict(lambda: [0, 0])  # same, total

    for work_item_id, group in work.groupby("work_item_id", sort=False):
        activity = str(group["concept:name"].iloc[0])
        running_since = None
        first_start = None
        schedule = None
        last_running_resource = None
        active_s = 0.0
        n_sessions = 0
        n_suspends = 0
        terminal = None
        terminal_time = None

        for row in group.itertuples(index=False, name=None):
            item = dict(zip(group.columns, row))
            transition = item["lifecycle:transition"]
            timestamp = item["time:timestamp"]
            resource = item["org:resource"]
            if transition == "schedule":
                schedule = timestamp
            elif transition in {"start", "resume"}:
                if transition == "resume":
                    resumes += 1
                    resume_by_activity[activity][1] += 1
                    if resource == last_running_resource:
                        same_resource_resumes += 1
                        resume_by_activity[activity][0] += 1
                running_since = timestamp
                first_start = first_start or timestamp
                last_running_resource = resource
                n_sessions += 1
            elif transition in {"suspend", "complete"} and running_since is not None:
                active_s += max(0.0, (timestamp - running_since).total_seconds())
                running_since = None
                if transition == "suspend":
                    n_suspends += 1
            if transition in TERMINALS:
                terminal = transition
                terminal_time = timestamp

        if terminal is None or terminal_time is None:
            continue
        origin = schedule if terminal == "withdraw" else first_start
        elapsed_s = (
            max(0.0, (terminal_time - origin).total_seconds())
            if origin is not None else None
        )
        records.append({
            "work_item_id": work_item_id,
            "activity": activity,
            "terminal": terminal,
            "elapsed_s": elapsed_s,
            "active_s": active_s,
            "non_active_s": (elapsed_s - active_s) if elapsed_s is not None else None,
            "sessions": n_sessions,
            "suspends": n_suspends,
        })

    instances = pd.DataFrame(records)
    by_outcome = {}
    by_activity_outcome = {}
    if not instances.empty:
        for outcome, group in instances.groupby("terminal"):
            by_outcome[outcome] = {
                "work_items": int(len(group)),
                "elapsed_seconds": _summary(group["elapsed_s"].dropna().tolist()),
                "active_seconds": _summary(group["active_s"].tolist()),
                "non_active_seconds": _summary(group["non_active_s"].dropna().tolist()),
                "suspends": _summary(group["suspends"].tolist()),
            }
        for (activity, outcome), group in instances.groupby(["activity", "terminal"]):
            by_activity_outcome.setdefault(activity, {})[outcome] = {
                "work_items": int(len(group)),
                "elapsed_seconds": _summary(group["elapsed_s"].dropna().tolist()),
                "active_seconds": _summary(group["active_s"].tolist()),
                "suspends": _summary(group["suspends"].tolist()),
            }

    return {
        "transition_counts": {
            str(k): int(v)
            for k, v in work["lifecycle:transition"].value_counts().items()
        },
        "terminal_recomposition": by_outcome,
        "by_activity_and_terminal": by_activity_outcome,
        "resume_ownership": {
            "same_resource": int(same_resource_resumes),
            "total_resumes": int(resumes),
            "same_resource_rate": (
                same_resource_resumes / resumes if resumes else None
            ),
            "by_activity": {
                activity: {
                    "same_resource": same,
                    "total_resumes": total,
                    "same_resource_rate": same / total if total else None,
                }
                for activity, (same, total) in sorted(resume_by_activity.items())
            },
        },
    }


def evaluate(log_path: Path, completed_path: Path, reference_path: Path) -> dict:
    df_all = pd.read_csv(log_path)
    # Logger isoformat rows mix values with and without fractional seconds;
    # pandas' strict inferred parser leaves that mixed column as strings.
    df_all["time:timestamp"] = pd.to_datetime(
        df_all["time:timestamp"], format="ISO8601")
    completed = set(completed_path.read_text(encoding="utf-8").splitlines())
    df = df_all[df_all["case:concept:name"].astype(str).isin(completed)].copy()
    reference = metrics.load_reference(reference_path)
    return {
        "configuration": {
            "lifecycle_mode": "active",
            "log": str(log_path),
            "completed_cases": len(completed),
            "logged_rows": int(len(df_all)),
        },
        "general_metrics": metrics.evaluate(df, reference, df_all=df_all),
        "lifecycle": lifecycle_evidence(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--reference", type=Path,
        default=REPO_ROOT / "simulation_inputs_active.json",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate(args.log, args.completed, args.reference)
    if args.label:
        result["configuration"]["label"] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    ownership = result["lifecycle"]["resume_ownership"]
    print(f"[lifecycle] {args.label or args.log.stem}: "
          f"{result['configuration']['completed_cases']} completed cases, "
          f"same-resource resumes={ownership['same_resource_rate']:.3f}")
    print(f"[lifecycle] wrote {args.out}")


if __name__ == "__main__":
    main()

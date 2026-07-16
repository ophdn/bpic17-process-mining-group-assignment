"""
train_decision_rules.py
========================
Trains data-based decision rules for Section 1.5 Advanced I: for every
decision point in the discovered Petri net (a marking with more than one
enabled visible transition), learn which case/runtime attributes predict
the branch that actually gets taken — the decision-point-as-classification
method from Rozinat et al., "Discovering Simulation Models" (see
docs/paper_insights_discovering_simulation_models.md).

Method
------
1. Replay every case's real activity sequence on the exact same Petri net /
   tau-closure logic PetriNetProcessComponent uses at runtime
   (_visible_frontier / _fire_activity, reused directly — not
   reimplemented), so the decision points found here are exactly the ones
   _weighted_choice will encounter during simulation.
2. Whenever more than one visible transition is enabled simultaneously,
   record one training example: features = the case's spawn attributes
   (ApplicationType, LoanGoal, RequestedAmount — known from
   A_Create Application onward) plus the most recently observed offer
   attributes (CreditScore, OfferedAmount, NumberOfTerms, MonthlyCost,
   FirstWithdrawalAmount — sentinel/has_offer=0 until the case's first
   O_Create Offer). Label = the activity the real log actually did next
   among the enabled options.
3. Group examples by decision point (the sorted tuple of enabled labels —
   the same set recurring is the same decision point) and train one
   sklearn DecisionTreeClassifier per decision point with enough examples.
   sklearn's CART is the closest available equivalent to the paper's
   C4.5/J48 (both greedy, splitting on information gain / Gini).

Deliberately excluded features: "Accepted" / "Selected" (BPIC-17 offer
attributes) — they record the outcome of the very decisions
(O_Accepted/O_Cancelled/O_Refused) being predicted, so using them as inputs
would leak the label.

Usage
-----
    python train_decision_rules.py --log data/BPIChallenge2017.xes.gz

Output
------
    simulation/models/decision_rules.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pm4py.objects.petri_net.obj import Marking
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters in the report output — force UTF-8 so a cosmetic print never
# kills a 10-minute training run. Guard with hasattr: under a Jupyter kernel
# sys.stdout is an ipykernel OutStream with no reconfigure().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_log_info import load_log, filter_to_complete  # noqa: E402

from simulation.components.petri_process import PetriNetProcessComponent  # noqa: E402

RANDOM_SEED = 42

DEFAULT_BPMN_PATH = Path("simulation/models/bpic17_process.bpmn")
OUTPUT_PATH = Path("simulation/models/decision_rules.joblib")

# Offer attributes are only known once a case has passed through
# O_Create Offer at least once; before that every case gets this sentinel
# snapshot (has_offer=0). This is set once, from the log's real values, at
# every occurrence of O_Create Offer during replay.
OFFER_ATTR_COLUMNS = [
    "CreditScore", "OfferedAmount", "NumberOfTerms", "MonthlyCost",
    "FirstWithdrawalAmount",
]

CATEGORICAL_FEATURES = ["application_type", "loan_goal"]
NUMERIC_FEATURES = [
    "requested_amount", "has_offer", "credit_score", "offered_amount",
    "number_of_terms", "monthly_cost", "first_withdrawal_amount",
]
FEATURE_NAMES = [f"{c}_enc" for c in CATEGORICAL_FEATURES] + NUMERIC_FEATURES

UNKNOWN = "__UNKNOWN__"

# A decision point needs at least this many replayed examples before we
# bother fitting a tree for it (otherwise fall back to BRANCHING_PROBS at
# simulation time — not enough data to learn anything reliable).
MIN_SAMPLES_PER_DECISION_POINT = 30

TREE_KWARGS = dict(
    max_depth=5,
    min_samples_leaf=20,
    random_state=RANDOM_SEED,
)


# ════════════════════════════════════════════════════════════════════════════
# Replay: find decision points + collect (attribute snapshot, chosen label)
# ════════════════════════════════════════════════════════════════════════════

def _initial_snapshot(row) -> dict:
    return {
        "application_type": str(row["case:ApplicationType"]),
        "loan_goal": str(row["case:LoanGoal"]),
        "requested_amount": float(row["case:RequestedAmount"]),
        "has_offer": 0,
        "credit_score": 0.0,
        "offered_amount": 0.0,
        "number_of_terms": 0.0,
        "monthly_cost": 0.0,
        "first_withdrawal_amount": 0.0,
    }


def _apply_offer(snapshot: dict, row) -> None:
    """*row* is a namedtuple from DataFrame.itertuples() — attribute access,
    not dict-style indexing."""
    snapshot["has_offer"] = 1
    snapshot["credit_score"] = float(row.CreditScore) if pd.notna(row.CreditScore) else 0.0
    snapshot["offered_amount"] = float(row.OfferedAmount) if pd.notna(row.OfferedAmount) else 0.0
    snapshot["number_of_terms"] = float(row.NumberOfTerms) if pd.notna(row.NumberOfTerms) else 0.0
    snapshot["monthly_cost"] = float(row.MonthlyCost) if pd.notna(row.MonthlyCost) else 0.0
    snapshot["first_withdrawal_amount"] = (
        float(row.FirstWithdrawalAmount) if pd.notna(row.FirstWithdrawalAmount) else 0.0
    )


def replay_log(df_complete: pd.DataFrame, comp: PetriNetProcessComponent):
    """
    Replay every case's real (deduplicated) activity sequence against
    comp's Petri net, using comp's own _visible_frontier/_fire_activity so
    decision points match exactly what the simulation will see.

    Returns (examples, n_fitting, n_nonfitting) where examples maps
    decision_point_key (sorted tuple of enabled labels) -> list of
    (attribute snapshot dict, chosen label, case start timestamp). The
    timestamp enables the temporal train/test split (train on older cases,
    test on the most recent ones — lecture 05, slide 37).
    """
    examples: dict = defaultdict(list)
    replay_key = "__replay__"
    n_fitting = 0
    n_nonfitting = 0

    for case_id, case_df in df_complete.groupby("case_id", sort=False):
        case_df = case_df.sort_values("timestamp")
        first_row = case_df.iloc[0]
        case_start = first_row["timestamp"]
        snapshot = _initial_snapshot(first_row)

        comp._markings[replay_key] = Marking(comp.im)
        fits = True
        for row in case_df.itertuples():
            act = row.activity
            marking = comp._markings[replay_key]
            frontier = comp._visible_frontier(marking)
            if act not in frontier:
                fits = False
                break

            if len(frontier) > 1:
                dp_key = tuple(sorted(frontier.keys()))
                examples[dp_key].append((dict(snapshot), act, case_start))

            comp._markings[replay_key] = frontier[act]
            comp._fire_activity(replay_key, act)

            if act == "O_Create Offer":
                _apply_offer(snapshot, row)

        comp._markings.pop(replay_key, None)
        if fits:
            n_fitting += 1
        else:
            n_nonfitting += 1

    return examples, n_fitting, n_nonfitting


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def fit_label_encoder(values, sentinels: list[str]) -> LabelEncoder:
    le = LabelEncoder()
    classes = list(pd.unique(pd.Series(values, dtype=str)))
    for s in sentinels:
        if s not in classes:
            classes.append(s)
    le.fit(classes)
    return le


def safe_transform(le: LabelEncoder, values, fallback: str) -> np.ndarray:
    known = set(le.classes_)
    s = pd.Series(values, dtype=str)
    s = s.where(s.isin(known), fallback)
    return le.transform(s)


def build_matrix(snapshots: list[dict], encoders: dict) -> np.ndarray:
    cols = {
        "application_type_enc": safe_transform(
            encoders["application_type"], [s["application_type"] for s in snapshots], UNKNOWN),
        "loan_goal_enc": safe_transform(
            encoders["loan_goal"], [s["loan_goal"] for s in snapshots], UNKNOWN),
    }
    for name in NUMERIC_FEATURES:
        cols[name] = np.array([s[name] for s in snapshots], dtype=float)
    return np.column_stack([cols[name] for name in FEATURE_NAMES]).astype(float)


def _macro_ovr_auc(y_true, proba: np.ndarray, classes) -> float | None:
    """Macro one-vs-rest ROC-AUC (lecture 05, slide 44) over the classes
    that are evaluable in y_true, i.e. that appear with both positives and
    negatives in the test slice. Returns None if no class is evaluable
    (can happen with a temporal split and a rare branch)."""
    y_true = np.asarray(y_true)
    aucs = []
    for i, cls in enumerate(classes):
        pos = y_true == cls
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, proba[:, i]))
    return float(np.mean(aucs)) if aucs else None


def train_decision_points(examples: dict, encoders: dict) -> dict:
    """Train one DecisionTreeClassifier per decision point with enough data.

    Split and metrics follow lecture 05: temporal 80/20 split (slide 37 —
    train on older cases, test on the most recent ones, because the process
    drifts over time) and precision / recall / ROC-AUC for categorical
    predictions (slide 44) in addition to accuracy + majority baseline.
    """
    models = {}
    skipped_too_few = []
    skipped_single_class = []

    for dp_key, rows in examples.items():
        if len(rows) < MIN_SAMPLES_PER_DECISION_POINT:
            skipped_too_few.append((dp_key, len(rows)))
            continue

        rows = sorted(rows, key=lambda r: r[2])  # oldest → newest case start
        snapshots = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        if len(set(labels)) < 2:
            skipped_single_class.append((dp_key, len(rows), labels[0]))
            continue

        X = build_matrix(snapshots, encoders)
        y = np.array(labels)

        split = int(len(rows) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        if len(set(y_train)) < 2 or len(y_test) == 0:
            skipped_single_class.append((dp_key, len(rows), y_train[0]))
            continue

        clf = DecisionTreeClassifier(**TREE_KWARGS)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        auc = _macro_ovr_auc(y_test, clf.predict_proba(X_test), clf.classes_)
        majority_baseline = pd.Series(y_train).value_counts(normalize=True).iloc[0]

        models[dp_key] = {
            "tree": clf,
            "classes": list(clf.classes_),
            "n_samples": len(rows),
            "split": "temporal_80_20",
            "test_accuracy": round(float(acc), 4),
            "test_precision_macro": round(float(prec), 4),
            "test_recall_macro": round(float(rec), 4),
            "test_roc_auc_ovr_macro": round(auc, 4) if auc is not None else None,
            "majority_baseline_accuracy": round(float(majority_baseline), 4),
        }

    return models, skipped_too_few, skipped_single_class


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path,
                        help="Path to the BPIC-17 event log (.xes/.xes.gz/.csv)")
    parser.add_argument("--bpmn", type=Path, default=DEFAULT_BPMN_PATH,
                        help="Path to the discovered BPMN model")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output joblib path")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"error: log not found: {args.log}")

    print(f"[load] Reading log: {args.log} ...")
    df = load_log(args.log)
    df_complete = filter_to_complete(df)
    print(f"[load] {len(df):,} raw events -> {len(df_complete):,} one-per-activity-occurrence "
          f"events, {df_complete['case_id'].nunique():,} cases.")

    print(f"[replay] Loading Petri net from {args.bpmn} ...")
    comp = PetriNetProcessComponent(bpmn_path=str(args.bpmn), seed=RANDOM_SEED)

    print("[replay] Replaying real log to find decision points ...")
    t0 = time.perf_counter()
    examples, n_fitting, n_nonfitting = replay_log(df_complete, comp)
    n_total = n_fitting + n_nonfitting
    print(f"[replay] Done in {time.perf_counter() - t0:.1f}s. "
          f"{n_fitting:,}/{n_total:,} cases fit the net "
          f"({100 * n_fitting / n_total:.1f}%), {n_nonfitting:,} non-fitting (skipped mid-replay).")
    print(f"[replay] {len(examples)} distinct decision points observed, "
          f"{sum(len(v) for v in examples.values()):,} decision instances total.")

    all_snapshots = [row[0] for rows in examples.values() for row in rows]
    encoders = {
        "application_type": fit_label_encoder(
            [s["application_type"] for s in all_snapshots], [UNKNOWN]),
        "loan_goal": fit_label_encoder(
            [s["loan_goal"] for s in all_snapshots], [UNKNOWN]),
    }

    print("\n[train] Fitting one DecisionTreeClassifier per decision point "
          f"(min {MIN_SAMPLES_PER_DECISION_POINT} samples) ...")
    models, skipped_too_few, skipped_single_class = train_decision_points(examples, encoders)

    print(f"\n[train] Trained {len(models)} decision-point models, "
          f"skipped {len(skipped_too_few)} (too few samples), "
          f"{len(skipped_single_class)} (only one branch ever observed).")

    # Persist artifact + metrics BEFORE any cosmetic reporting, so a print
    # problem can never cost us the (expensive) replay + training results.
    artifact = {
        "models": models,
        "encoders": encoders,
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "sentinels": {"unknown": UNKNOWN},
        "replay_stats": {
            "n_fitting": n_fitting, "n_nonfitting": n_nonfitting,
            "n_decision_points": len(examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"[save] Wrote artifact -> {args.output} "
          f"({args.output.stat().st_size / 1e3:.1f} KB)")

    # Report-ready metrics JSON (design decision 1.5: rules vs. probs — the
    # per-decision-point quality numbers live here).
    import json
    metrics_json = {
        "real_log_replay_on_bpmn": {
            "n_fitting": n_fitting,
            "n_nonfitting": n_nonfitting,
            "fit_pct": round(100 * n_fitting / max(n_fitting + n_nonfitting, 1), 2),
            "note": "share of real BPIC-17 cases whose full trace replays on "
                    "the BPMN/Petri net — key evidence for the process-model "
                    "source decision (Section 1.4)",
        },
        "split": "temporal_80_20 (lecture 05 slide 37)",
        "decision_points": {
            " | ".join(dp): {k: v for k, v in info.items() if k != "tree"}
            for dp, info in models.items()
        },
    }
    metrics_path = Path("output/models/decision_rules_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, default=float)
    print(f"[save] Wrote metrics -> {metrics_path}")

    print("\n── Per-decision-point results ──────────────────────────────────")
    for dp_key, info in sorted(models.items(), key=lambda kv: -kv[1]["n_samples"]):
        lift = info["test_accuracy"] - info["majority_baseline_accuracy"]
        auc_str = (f"{info['test_roc_auc_ovr_macro']:.4f}"
                   if info["test_roc_auc_ovr_macro"] is not None else "n/a")
        print(f"  {dp_key}")
        print(f"    n={info['n_samples']:<6} classes={info['classes']}")
        print(f"    tree acc={info['test_accuracy']:.4f}  vs majority-class "
              f"baseline={info['majority_baseline_accuracy']:.4f}  (lift={lift:+.4f})")
        print(f"    precision(macro)={info['test_precision_macro']:.4f}  "
              f"recall(macro)={info['test_recall_macro']:.4f}  "
              f"ROC-AUC(OvR,macro)={auc_str}  [temporal 80/20 split]")

    if skipped_too_few:
        print(f"\n  Skipped (< {MIN_SAMPLES_PER_DECISION_POINT} samples): "
              f"{[k for k, _ in skipped_too_few]}")
    if skipped_single_class:
        print(f"  Skipped (single branch always taken): "
              f"{[k for k, _, _ in skipped_single_class]}")

    # Human-readable rules for the biggest decision point, for the report.
    if models:
        biggest = max(models.items(), key=lambda kv: kv[1]["n_samples"])
        dp_key, info = biggest
        feature_names_readable = FEATURE_NAMES
        print(f"\n── Example rule tree for decision point {dp_key} ──────────")
        print(export_text(info["tree"], feature_names=feature_names_readable))


if __name__ == "__main__":
    main()

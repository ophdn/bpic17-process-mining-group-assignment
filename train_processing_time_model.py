"""
train_processing_time_model.py
==============================
Trains contextual processing-time models for the BPIC-17 simulation.

Two models are produced from the same features/target:

  Basic option 2 (Section 1.3) — a *point-estimation* Gradient Boosting
      regressor that predicts the expected log-duration for an activity
      instance given its context.

  Advanced I (Section 1.3) — a set of *quantile* Gradient Boosting
      regressors (q = 0.05 … 0.95) that describe the full conditional
      duration distribution, so the simulation can draw stochastic
      durations instead of a single point estimate (enable with
      --probabilistic).

Both are persisted into a single joblib artifact together with the label
encoders, feature order and evaluation metrics, so the simulation can
reconstruct the exact feature vector at sample time.

Usage
-----
    # inside the project virtualenv
    python train_processing_time_model.py --log BPIChallenge2017.xes
    python train_processing_time_model.py --log BPIChallenge2017.xes --probabilistic

Output
------
    simulation/models/processing_time_model.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters in the report output — force UTF-8 so a cosmetic print never
# kills a long training run. Guard with hasattr: under a Jupyter kernel
# sys.stdout is an ipykernel OutStream with no reconfigure().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Column aliasing (mirrors extract_log_info.py) ────────────────────────────
COL_ALIASES = {
    "case_id":   ["case:concept:name", "case_id", "CaseID", "caseid"],
    "activity":  ["concept:name", "activity", "Activity", "task"],
    "timestamp": ["time:timestamp", "timestamp", "Timestamp", "time"],
    "resource":  ["org:resource", "resource", "Resource", "org:group"],
    "lifecycle": ["lifecycle:transition", "lifecycle", "Lifecycle"],
}

# ── Feature contract (order matters — the simulation rebuilds this exactly) ──
FEATURE_NAMES = [
    "activity_enc",           # label-encoded activity name
    "resource_enc",           # label-encoded resource that runs the activity
    "previous_activity_enc",  # label-encoded previous activity in the case
    "day_of_week",            # 0=Mon … 6=Sun, from the activity-start timestamp
    "hour_of_day",            # 0 … 23, from the activity-start timestamp
    "case_position",          # 0-based ordinal of this activity within the case
    "case_age_seconds",       # seconds since the case's first event
    "n_previous_activities",  # count of activities already run in the case
]

# Sentinels so encoder lookups never crash on unseen / missing labels.
UNKNOWN = "__UNKNOWN__"   # unseen activity / resource at sample time
NO_PREV = "__START__"     # first activity in a case has no predecessor

# Duration sanity bounds
MAX_DURATION_SECONDS = 365 * 24 * 3600  # legacy elapsed span: drop > a year
# Active-session target (implementationplan §5.2): a single hands-on work session is
# minutes-to-hours, never the multi-day elapsed span, so the old year cap only made
# sense for the elapsed target. Cap active sessions at 7 days.
ACTIVE_MAX_DURATION_SECONDS = 7 * 24 * 3600

# Quantile grid for the probabilistic (Advanced I) model
QUANTILES = [round(q, 2) for q in np.arange(0.05, 0.96, 0.05)]

MODEL_KWARGS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=RANDOM_SEED,
)

OUTPUT_PATH = Path("simulation/models/processing_time_model.joblib")


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def _resolve_col(df: pd.DataFrame, key: str) -> str | None:
    for candidate in COL_ALIASES[key]:
        if candidate in df.columns:
            return candidate
    return None


def load_log(path: Path) -> pd.DataFrame:
    """Load an XES or CSV event log into a canonical-column DataFrame.

    Returns a DataFrame with columns:
        case_id, activity, timestamp (datetime), resource, lifecycle
    """
    suffixes = [s.lower() for s in path.suffixes]
    is_xes = suffixes[-2:] == [".xes", ".gz"] or suffixes[-1:] == [".xes"]

    if is_xes:
        import pm4py
        print(f"[load] Reading XES with pm4py: {path} …")
        df = pm4py.read_xes(str(path))
    else:
        print(f"[load] Reading CSV: {path} …")
        df = pd.read_csv(path)

    rename = {}
    for key in ("case_id", "activity", "timestamp", "resource", "lifecycle"):
        col = _resolve_col(df, key)
        if col is None and key in ("case_id", "activity", "timestamp"):
            raise ValueError(f"Log is missing a required '{key}' column. "
                             f"Looked for any of {COL_ALIASES[key]}")
        if col is not None:
            rename[col] = key
    df = df.rename(columns=rename)

    if "resource" not in df.columns:
        df["resource"] = UNKNOWN
    if "lifecycle" not in df.columns:
        raise ValueError("Log has no lifecycle:transition column — cannot pair "
                         "start/complete events to compute durations.")

    df = df[["case_id", "activity", "timestamp", "resource", "lifecycle"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lifecycle"] = df["lifecycle"].astype(str).str.lower()
    df["resource"] = df["resource"].fillna(UNKNOWN).astype(str)
    df["activity"] = df["activity"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df = df.dropna(subset=["timestamp"])
    print(f"[load] {len(df):,} events, {df['case_id'].nunique():,} cases, "
          f"{df['activity'].nunique()} activities.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Activity-instance durations + context features
# ════════════════════════════════════════════════════════════════════════════

def build_instances(df: pd.DataFrame) -> pd.DataFrame:
    """Pair start/complete events into activity instances with 8 context features.

    Repeated activities within a case are disambiguated with a per-(case,
    activity) sequence number (cumcount) so start #k pairs with complete #k —
    a plain merge on (case, activity) would explode combinatorially.

    NOTE: the (complete - start) delta below is the full activity duration and
    includes any post-assignment queueing/waiting as well as service time. The
    probabilistic (Advanced I) model targets this combined duration directly, so
    its predicted distribution already reflects that variability — splitting the
    two (Advanced II) is an optional refinement, not required for Advanced I.
    """
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    starts = df[df["lifecycle"] == "start"].copy()
    completes = df[df["lifecycle"] == "complete"].copy()
    if starts.empty:
        raise ValueError("No 'start' lifecycle events found — cannot pair "
                         "durations. Check the lifecycle:transition values.")

    starts["seq"] = starts.groupby(["case_id", "activity"]).cumcount()
    completes["seq"] = completes.groupby(["case_id", "activity"]).cumcount()

    inst = starts.merge(
        completes[["case_id", "activity", "seq", "timestamp"]],
        on=["case_id", "activity", "seq"],
        suffixes=("", "_complete"),
        how="inner",
    )

    inst["duration_s"] = (
        inst["timestamp_complete"] - inst["timestamp"]
    ).dt.total_seconds()

    n_before = len(inst)
    inst = inst[(inst["duration_s"] > 0) &
                (inst["duration_s"] <= MAX_DURATION_SECONDS)].copy()
    print(f"[instances] {n_before:,} paired instances → "
          f"{len(inst):,} after duration filter (0 < d ≤ 365d).")

    # --- Contextual features (all vectorized; no row-wise apply) ---
    inst = inst.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    grp = inst.groupby("case_id", sort=False)

    inst["case_position"] = grp.cumcount()
    inst["n_previous_activities"] = inst["case_position"]  # sequential sim ⇒ equal
    inst["previous_activity"] = grp["activity"].shift(1).fillna(NO_PREV)

    # case_age_seconds via first-timestamp map + subtraction (vectorized)
    first_ts = grp["timestamp"].transform("min")
    inst["case_age_seconds"] = (inst["timestamp"] - first_ts).dt.total_seconds()

    inst["day_of_week"] = inst["timestamp"].dt.dayofweek
    inst["hour_of_day"] = inst["timestamp"].dt.hour

    return inst


def build_active_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Retarget the model to *active-session seconds* (implementationplan §5.2).

    Segments each W_ work-item instance from the lifecycle grammar
    (schedule → start → (suspend → resume)* → complete|ate_abort|withdraw) and
    emits **one row per active running session** (start/resume → next
    suspend/terminal). Feature contract **option A (LOCKED)**: the 8 context
    features are computed once at the work item's *first* start and duplicated
    across every session row of that item — no session_index in v1 — because the
    runtime carries exactly that per-work-item context.
    """
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    instances = []  # one dict per W_ work-item instance
    for case_id, cgrp in df.groupby("case_id", sort=False):
        w = cgrp[cgrp["activity"].str.startswith("W_")]
        for act, agrp in w.groupby("activity", sort=False):
            cur = None
            for row in agrp.itertuples(index=False):
                lc, ts, res = row.lifecycle, row.timestamp, row.resource
                if lc == "schedule":
                    if cur is None:
                        cur = {"first_start": None, "resource": UNKNOWN,
                                "running_since": None, "sessions": []}
                elif lc == "start":
                    if cur is None:
                        cur = {"first_start": ts, "resource": res,
                                "running_since": ts, "sessions": []}
                    else:
                        if cur["first_start"] is None:
                            cur["first_start"] = ts
                            cur["resource"] = res
                        cur["running_since"] = ts
                elif lc == "resume":
                    if cur is not None:
                        cur["running_since"] = ts
                elif lc == "suspend":
                    if cur is not None and cur["running_since"] is not None:
                        seg = (ts - cur["running_since"]).total_seconds()
                        if seg > 0:
                            cur["sessions"].append(seg)
                        cur["running_since"] = None
                elif lc in ("complete", "ate_abort", "withdraw"):
                    if cur is not None:
                        if cur["running_since"] is not None and lc != "withdraw":
                            seg = (ts - cur["running_since"]).total_seconds()
                            if seg > 0:
                                cur["sessions"].append(seg)
                        if cur["first_start"] is not None and cur["sessions"]:
                            instances.append({"case_id": case_id, "activity": act,
                                              "resource": cur["resource"],
                                              "first_start": cur["first_start"],
                                              "sessions": cur["sessions"]})
                        cur = None
            if cur is not None and cur["first_start"] is not None and cur["sessions"]:
                instances.append({"case_id": case_id, "activity": act,
                                  "resource": cur["resource"],
                                  "first_start": cur["first_start"],
                                  "sessions": cur["sessions"]})

    if not instances:
        raise ValueError("No W_ active sessions reconstructed — check the "
                         "lifecycle:transition values in the log.")

    idf = pd.DataFrame(instances).sort_values(
        ["case_id", "first_start"]).reset_index(drop=True)
    grp = idf.groupby("case_id", sort=False)
    idf["case_position"] = grp.cumcount()
    idf["n_previous_activities"] = idf["case_position"]
    idf["previous_activity"] = grp["activity"].shift(1).fillna(NO_PREV)
    first_ts = grp["first_start"].transform("min")
    idf["case_age_seconds"] = (idf["first_start"] - first_ts).dt.total_seconds()
    idf["day_of_week"] = idf["first_start"].dt.dayofweek
    idf["hour_of_day"] = idf["first_start"].dt.hour
    idf["timestamp"] = idf["first_start"]  # temporal-split ordering key

    # Expand instances → one row per active session (features duplicated).
    idf = idf.explode("sessions").reset_index(drop=True)
    idf["duration_s"] = idf["sessions"].astype(float)
    n_before = len(idf)
    idf = idf[(idf["duration_s"] > 0) &
              (idf["duration_s"] <= ACTIVE_MAX_DURATION_SECONDS)].copy()
    print(f"[instances] {len(instances):,} W_ work-item instances → "
          f"{n_before:,} active sessions → {len(idf):,} after filter "
          f"(0 < d ≤ 7d).")
    return idf


def fit_label_encoder(values: pd.Series, sentinels: list[str]) -> LabelEncoder:
    """Fit a LabelEncoder that always contains the given sentinels, so the
    simulation can fall back to a valid index for unseen labels."""
    le = LabelEncoder()
    classes = list(pd.unique(values.astype(str)))
    for s in sentinels:
        if s not in classes:
            classes.append(s)
    le.fit(classes)
    return le


def safe_transform(le: LabelEncoder, values: pd.Series, fallback: str) -> np.ndarray:
    """Transform, mapping any label unseen at fit time to the fallback index."""
    known = set(le.classes_)
    vals = values.astype(str).where(values.astype(str).isin(known), fallback)
    return le.transform(vals)


def build_matrix(inst: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Assemble the feature matrix in FEATURE_NAMES order."""
    cols = {
        "activity_enc": safe_transform(encoders["activity"], inst["activity"], UNKNOWN),
        "resource_enc": safe_transform(encoders["resource"], inst["resource"], UNKNOWN),
        "previous_activity_enc": safe_transform(
            encoders["previous_activity"], inst["previous_activity"], NO_PREV),
        "day_of_week": inst["day_of_week"].to_numpy(),
        "hour_of_day": inst["hour_of_day"].to_numpy(),
        "case_position": inst["case_position"].to_numpy(),
        "case_age_seconds": inst["case_age_seconds"].to_numpy(),
        "n_previous_activities": inst["n_previous_activities"].to_numpy(),
    }
    return np.column_stack([cols[name] for name in FEATURE_NAMES]).astype(float)


# ════════════════════════════════════════════════════════════════════════════
# Quantile-model evaluation (Advanced I)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_quantile_models(quantile_models: dict, quantiles: list,
                             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the predicted conditional distribution, answering lecture 05
    slide 47's challenge ("how to evaluate the probability densities?"):

    - mean pinball loss (log space) — the proper scoring rule for quantile
      regression, averaged over the quantile grid; lower is better.
    - empirical coverage of the central 90 % and 50 % prediction intervals —
      a calibrated model covers ~0.90 / ~0.50 of the test durations.
    - R² of the median quantile (log space) — point-quality sanity check.
    """
    preds = {q: quantile_models[q].predict(X_test) for q in quantiles}
    pinball = [mean_pinball_loss(y_test, preds[q], alpha=q) for q in quantiles]

    q_lo, q_hi = min(quantiles), max(quantiles)                 # 0.05 / 0.95
    q25 = min(quantiles, key=lambda q: abs(q - 0.25))
    q75 = min(quantiles, key=lambda q: abs(q - 0.75))
    q50 = min(quantiles, key=lambda q: abs(q - 0.50))

    return {
        "mean_pinball_loss_log": float(np.mean(pinball)),
        "coverage_90pct_interval": float(np.mean(
            (y_test >= preds[q_lo]) & (y_test <= preds[q_hi]))),
        "coverage_50pct_interval": float(np.mean(
            (y_test >= preds[q25]) & (y_test <= preds[q75]))),
        "r2_log_median_quantile": float(r2_score(y_test, preds[q50])),
    }


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def train(log_path: Path, probabilistic: bool, output_path: Path,
          active: bool = False, metrics_output: Path | None = None) -> None:
    df = load_log(log_path)
    if active:
        print("[mode] active — target = active-session seconds (W_ items).")
        inst = build_active_sessions(df)
        target_name = "active_session_seconds"
        lifecycle_schema = "active_v1"
    else:
        inst = build_instances(df)
        target_name = "elapsed_start_complete_seconds"
        lifecycle_schema = "legacy"

    encoders = {
        "activity": fit_label_encoder(inst["activity"], [UNKNOWN]),
        "resource": fit_label_encoder(inst["resource"], [UNKNOWN]),
        "previous_activity": fit_label_encoder(inst["previous_activity"], [NO_PREV, UNKNOWN]),
    }

    X = build_matrix(inst, encoders)
    y = np.log1p(inst["duration_s"].to_numpy())  # target: log1p(duration_s)

    # Temporal split (lecture 05, slide 37): train on the oldest 80 % of
    # activity instances, test on the most recent 20 %. A random split would
    # leak future process drift (new resources, shifting workloads) into
    # training and overstate test performance.
    order = np.argsort(inst["timestamp"].to_numpy())
    split = int(len(order) * 0.8)
    train_idx, test_idx = order[:split], order[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    split_date = inst["timestamp"].iloc[order[split]]
    print(f"[train] {X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows, "
          f"{X.shape[1]} features (temporal split at {split_date}).")

    # --- Point-estimation model (Basic option 2) ---
    t0 = time.perf_counter()
    model = GradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(X_train, y_train)
    print(f"[train] Point model fitted in {time.perf_counter() - t0:.1f}s.")

    # --- Evaluation ---
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    # MAE + MSE are the two numerical-prediction metrics prescribed by
    # lecture 05, slide 44; RMSE and R² are reported on top for readability.
    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "mse_seconds2": mse,
        "rmse_seconds": float(np.sqrt(mse)),
        "r2_raw": float(r2_score(y_true, y_pred)),
        "r2_log": float(r2_score(y_test, y_pred_log)),
        "split": "temporal_80_20",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    print("\n── Point-estimation model — test metrics (temporal split) ─────")
    print(f"  MAE  (raw)  : {metrics['mae_seconds']:>14,.1f} s "
          f"({metrics['mae_seconds']/3600:.2f} h)")
    print(f"  MSE  (raw)  : {metrics['mse_seconds2']:>14,.4g} s²")
    print(f"  RMSE (raw)  : {metrics['rmse_seconds']:>14,.1f} s "
          f"({metrics['rmse_seconds']/3600:.2f} h)")
    print(f"  R²   (raw)  : {metrics['r2_raw']:>14.4f}")
    print(f"  R²   (log)  : {metrics['r2_log']:>14.4f}")

    print("\n── Feature importances ────────────────────────────────────────")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                            key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<24} {imp:.4f}")

    artifact = {
        "model": model,
        "encoders": encoders,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "target_transform": "log1p",
        # Lifecycle contract metadata (implementationplan §5.2): a legacy runtime
        # loading an active artifact (or vice-versa) can check these and fail loudly
        # rather than silently mispredicting an elapsed span as an active session.
        "target": target_name,
        "lifecycle_schema": lifecycle_schema,
        "sentinels": {"unknown": UNKNOWN, "no_prev": NO_PREV},
        "quantile_models": None,
        "quantiles": None,
    }

    # --- Probabilistic quantile models (Advanced I) ---
    if probabilistic:
        print(f"\n[train] Fitting {len(QUANTILES)} quantile models "
              f"(q = {QUANTILES[0]} … {QUANTILES[-1]}) …")
        quantile_models = {}
        t0 = time.perf_counter()
        for q in QUANTILES:
            qm = GradientBoostingRegressor(loss="quantile", alpha=q, **MODEL_KWARGS)
            qm.fit(X_train, y_train)
            quantile_models[q] = qm
            print(f"    q={q:.2f} fitted "
                  f"({time.perf_counter() - t0:.1f}s elapsed)")
        artifact["quantile_models"] = quantile_models
        artifact["quantiles"] = QUANTILES

        qe = evaluate_quantile_models(quantile_models, QUANTILES, X_test, y_test)
        artifact["metrics"]["quantile_eval"] = qe
        print("\n── Quantile models — distribution evaluation (temporal split) ─")
        print(f"  mean pinball loss (log) : {qe['mean_pinball_loss_log']:.4f}")
        print(f"  90% interval coverage   : {qe['coverage_90pct_interval']:.3f} (target ≈ 0.90)")
        print(f"  50% interval coverage   : {qe['coverage_50pct_interval']:.3f} (target ≈ 0.50)")
        print(f"  median-quantile R² (log): {qe['r2_log_median_quantile']:.4f}")

    # --- Persist ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n[save] Wrote artifact → {output_path} ({size_mb:.1f} MB)")

    # Report-ready metrics JSON (design decision 1.3: distribution vs.
    # ml_model vs. ml_probabilistic — model-quality evidence lives here).
    import json
    metrics_json = {
        "point_model": artifact["metrics"],
        "feature_importances": {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
        },
        "quantile_eval": artifact["metrics"].get("quantile_eval"),
        "quantiles": QUANTILES if probabilistic else None,
    }
    metrics_json["target"] = target_name
    metrics_json["lifecycle_schema"] = lifecycle_schema
    metrics_path = metrics_output or Path("output/models/processing_time_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, default=float)
    print(f"[save] Wrote metrics → {metrics_path}")
    print(f"[save] Contents: model{' + ' + str(len(QUANTILES)) + ' quantile models' if probabilistic else ''}, "
          f"{len(encoders)} encoders, {len(FEATURE_NAMES)} features, metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path,
                        help="Path to the BPIC-17 event log (.xes/.xes.gz/.csv)")
    parser.add_argument("--probabilistic", action="store_true",
                        help="Also train quantile models (Advanced I)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output joblib path")
    parser.add_argument("--metrics-output", type=Path, default=None,
                        help="Metrics JSON path (default: "
                             "output/models/processing_time_metrics.json). Use a "
                             "_active path so the active model never overwrites the "
                             "legacy metrics (implementationplan §4.8).")
    parser.add_argument("--lifecycle", action="store_true",
                        help="Train the active-session-seconds target from the "
                             "suspend/resume lifecycle grammar (implementationplan "
                             "§5.2) instead of the legacy elapsed start→complete span.")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"error: log not found: {args.log}")

    train(args.log, args.probabilistic, args.output,
          active=args.lifecycle, metrics_output=args.metrics_output)


if __name__ == "__main__":
    main()

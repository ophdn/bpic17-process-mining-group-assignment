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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

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
MAX_DURATION_SECONDS = 365 * 24 * 3600  # drop instances longer than a year

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

    NOTE(Advanced II): the (complete - start) delta below conflates post-
    assignment queueing/waiting with actual service time. Splitting the two
    needs separate assign/start/complete transitions and is out of scope here.
    Advanced II should refine THIS duration definition.
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
# Training
# ════════════════════════════════════════════════════════════════════════════

def train(log_path: Path, probabilistic: bool, output_path: Path) -> None:
    df = load_log(log_path)
    inst = build_instances(df)

    encoders = {
        "activity": fit_label_encoder(inst["activity"], [UNKNOWN]),
        "resource": fit_label_encoder(inst["resource"], [UNKNOWN]),
        "previous_activity": fit_label_encoder(inst["previous_activity"], [NO_PREV, UNKNOWN]),
    }

    X = build_matrix(inst, encoders)
    y = np.log1p(inst["duration_s"].to_numpy())  # target: log1p(duration_s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"[train] {X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows, "
          f"{X.shape[1]} features.")

    # --- Point-estimation model (Basic option 2) ---
    t0 = time.perf_counter()
    model = GradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(X_train, y_train)
    print(f"[train] Point model fitted in {time.perf_counter() - t0:.1f}s.")

    # --- Evaluation ---
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    metrics = {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "rmse_seconds": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_raw": float(r2_score(y_true, y_pred)),
        "r2_log": float(r2_score(y_test, y_pred_log)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    print("\n── Point-estimation model — test metrics ──────────────────────")
    print(f"  MAE  (raw)  : {metrics['mae_seconds']:>14,.1f} s "
          f"({metrics['mae_seconds']/3600:.2f} h)")
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

        # Sanity: report the median quantile's R² in log space
        med = quantile_models[min(QUANTILES, key=lambda q: abs(q - 0.5))]
        r2_med = r2_score(y_test, med.predict(X_test))
        print(f"[train] Median-quantile R² (log): {r2_med:.4f}")

    # --- Persist ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n[save] Wrote artifact → {output_path} ({size_mb:.1f} MB)")
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
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"error: log not found: {args.log}")

    train(args.log, args.probabilistic, args.output)


if __name__ == "__main__":
    main()

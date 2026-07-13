"""
extract_log_info.py
====================
Run this script on the BPIC-17 event log to extract everything
the simulation engine needs. The output is saved as a single
JSON file that you can share directly.

Usage
-----
    pip install pm4py pandas scipy
    python extract_log_info.py --log path/to/BPI_Challenge_2017.xes

    # CSV also works (needs case_id, activity, timestamp, resource columns):
    python extract_log_info.py --log path/to/log.csv

Output
------
    simulation_inputs.json   <- paste this to Claude
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot encode characters like
# "→" in the progress output — force UTF-8 so a cosmetic print never kills
# a multi-minute extraction run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── column name aliases (handles BPIC-17 and generic CSV logs) ──────────────
COL_ALIASES = {
    "case_id":    ["case:concept:name", "case_id", "CaseID", "caseid"],
    "activity":   ["concept:name", "activity", "Activity", "task"],
    "timestamp":  ["time:timestamp", "timestamp", "Timestamp", "time"],
    "resource":   ["org:resource", "resource", "Resource", "org:group"],
    "lifecycle":  ["lifecycle:transition", "lifecycle", "Lifecycle"],
}

BEST_FIT_DISTS = ["expon", "norm", "lognorm", "gamma", "weibull_min"]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def resolve_col(df: pd.DataFrame, key: str) -> str | None:
    for candidate in COL_ALIASES[key]:
        if candidate in df.columns:
            return candidate
    return None


def load_log(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    is_xes = suffixes == [".xes"] or suffixes == [".xes", ".gz"]
    is_csv = suffixes[-1] in (".csv",) or (suffixes == [".gz"] and not is_xes)

    if is_xes:
        try:
            import pm4py
            log = pm4py.read_xes(str(path))
            df  = pm4py.convert_to_dataframe(log)
        except Exception as e:
            sys.exit(f"[ERROR] Could not read XES file: {e}")
    elif is_csv:
        df = pd.read_csv(path, parse_dates=True)
    else:
        sys.exit(f"[ERROR] Unsupported file type: {''.join(path.suffixes)}. Use .xes, .xes.gz or .csv")

    # resolve column names
    col_map = {}
    for key in COL_ALIASES:
        col = resolve_col(df, key)
        if col:
            col_map[col] = key
    df = df.rename(columns=col_map)

    # ensure timestamp is datetime (handles ISO8601, mixed formats, tz-aware/-naive)
    parsed = pd.to_datetime(df["timestamp"], format="mixed", utc=True, errors="coerce")
    df["timestamp"] = parsed
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


def filter_to_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per real activity occurrence, not per lifecycle transition.

    BPIC-17 (and similar transactional logs) record multiple lifecycle
    events per activity instance (schedule/start/suspend/resume/complete/
    ...). Treating every row as a separate occurrence — e.g. for
    basic_stats/branching_probs/process_variants — massively inflates
    self-loop counts (consecutive lifecycle events of the *same* instance
    look like the activity following itself) and case lengths. So: keep
    the 'complete' event where an activity has one, and fall back to
    'start' only for activities that are never recorded as 'complete' in
    this log (e.g. cases still open when the log was cut).
    """
    if "lifecycle" not in df.columns:
        return df

    lifecycle_lower = df["lifecycle"].str.lower()
    has_complete = (
        df.assign(_lc=lifecycle_lower)
        .groupby("activity")["_lc"]
        .apply(lambda s: (s == "complete").any())
    )
    fallback_activities = set(has_complete[~has_complete].index)

    is_complete = lifecycle_lower == "complete"
    is_fallback_start = df["activity"].isin(fallback_activities) & (lifecycle_lower == "start")
    return df[is_complete | is_fallback_start].copy()


def fit_best_distribution(data_seconds: np.ndarray) -> dict:
    """Fit several distributions, return the best by AIC."""
    data = data_seconds[data_seconds > 0]
    if len(data) < 5:
        return {"distribution": "expon", "params": [float(data.mean()), 0.0],
                "mean_s": float(data.mean()), "std_s": float(data.std())}

    best = {"aic": np.inf}
    for dist_name in BEST_FIT_DISTS:
        try:
            dist  = getattr(stats, dist_name)
            params = dist.fit(data, floc=0)
            log_l  = dist.logpdf(data, *params).sum()
            k      = len(params)
            aic    = 2 * k - 2 * log_l
            if aic < best["aic"]:
                best = {
                    "distribution": dist_name,
                    "params":       [round(float(p), 4) for p in params],
                    "aic":          round(float(aic), 2),
                }
        except Exception:
            pass

    best["mean_s"] = round(float(data.mean()), 2)
    best["std_s"]  = round(float(data.std()),  2)
    best["n"]      = int(len(data))
    best.pop("aic", None)
    return best


# ════════════════════════════════════════════════════════════════════════════
# Extraction functions
# ════════════════════════════════════════════════════════════════════════════

def extract_basic_stats(df: pd.DataFrame) -> dict:
    n_events  = len(df)
    n_cases   = df["case_id"].nunique()
    n_activities = df["activity"].nunique()

    case_lengths = df.groupby("case_id").size()
    case_dur = (
        df.groupby("case_id")["timestamp"]
        .agg(lambda x: (x.max() - x.min()).total_seconds())
    )

    return {
        "n_cases":      int(n_cases),
        "n_events":     int(n_events),
        "n_activities": int(n_activities),
        "case_length":  {
            "mean": round(float(case_lengths.mean()), 2),
            "std":  round(float(case_lengths.std()),  2),
            "min":  int(case_lengths.min()),
            "max":  int(case_lengths.max()),
        },
        "case_duration_seconds": {
            "mean": round(float(case_dur.mean()), 2),
            "std":  round(float(case_dur.std()),  2),
            "min":  round(float(case_dur.min()),  2),
            "max":  round(float(case_dur.max()),  2),
        },
    }


def extract_activities(df: pd.DataFrame) -> dict:
    counts = df["activity"].value_counts()
    return {
        act: int(cnt)
        for act, cnt in counts.items()
    }


def extract_processing_times(df: pd.DataFrame) -> dict:
    """
    For each activity, compute duration between START and COMPLETE events.
    Falls back to single-timestamp diff between consecutive events if
    no lifecycle column exists.
    """
    result = {}

    if "lifecycle" in df.columns:
        starts    = df[df["lifecycle"].str.lower().isin(["start", "assign"])].copy()
        completes = df[df["lifecycle"].str.lower().isin(["complete"])].copy()

        merged = pd.merge(
            starts[["case_id", "activity", "timestamp"]],
            completes[["case_id", "activity", "timestamp"]],
            on=["case_id", "activity"],
            suffixes=("_start", "_complete"),
        )
        merged["duration_s"] = (
            merged["timestamp_complete"] - merged["timestamp_start"]
        ).dt.total_seconds()
        merged = merged[merged["duration_s"] >= 0]

        for act, grp in merged.groupby("activity"):
            result[act] = fit_best_distribution(grp["duration_s"].values)
    else:
        # No lifecycle: estimate from consecutive events within a case
        df2 = df.sort_values(["case_id", "timestamp"]).copy()
        df2["next_ts"] = df2.groupby("case_id")["timestamp"].shift(-1)
        df2["duration_s"] = (df2["next_ts"] - df2["timestamp"]).dt.total_seconds()
        df2 = df2.dropna(subset=["duration_s"])
        df2 = df2[df2["duration_s"] >= 0]

        for act, grp in df2.groupby("activity"):
            result[act] = fit_best_distribution(grp["duration_s"].values)

    return result


def extract_arrival_rate(df: pd.DataFrame) -> dict:
    """
    Inter-arrival times: time between consecutive case start events.
    """
    first_events = (
        df.sort_values("timestamp")
        .groupby("case_id")["timestamp"]
        .first()
        .sort_values()
    )
    inter_arr = first_events.diff().dt.total_seconds().dropna()
    inter_arr = inter_arr[inter_arr > 0]

    fit = fit_best_distribution(inter_arr.values)
    fit["arrivals_per_day"] = round(float(86400 / inter_arr.mean()), 4)

    # Daily arrival counts (useful for advanced dynamic spawn rates)
    daily = first_events.dt.date.value_counts().sort_index()
    fit["daily_arrivals"] = {
        "mean": round(float(daily.mean()), 2),
        "std":  round(float(daily.std()),  2),
        "min":  int(daily.min()),
        "max":  int(daily.max()),
    }
    return fit


def extract_process_variants(df: pd.DataFrame) -> dict:
    """Top-20 process variants by frequency."""
    traces = (
        df.sort_values("timestamp")
        .groupby("case_id")["activity"]
        .apply(lambda x: " → ".join(x))
    )
    counts = traces.value_counts()
    total  = len(traces)
    top20  = counts.head(20)

    return {
        "total_variants": int(len(counts)),
        "top_20": [
            {
                "rank":      i + 1,
                "frequency": int(cnt),
                "pct":       round(float(cnt / total * 100), 2),
                "trace":     trace,
            }
            for i, (trace, cnt) in enumerate(top20.items())
        ],
    }


def extract_branching(df: pd.DataFrame) -> dict:
    """
    Approximate branching probabilities: for each activity,
    what activity follows it and how often?
    """
    df2 = df.sort_values(["case_id", "timestamp"]).copy()
    df2["next_activity"] = df2.groupby("case_id")["activity"].shift(-1)
    df2 = df2.dropna(subset=["next_activity"])

    result = {}
    for act, grp in df2.groupby("activity"):
        counts = grp["next_activity"].value_counts()
        total  = counts.sum()
        result[act] = {
            next_act: round(float(cnt / total), 4)
            for next_act, cnt in counts.items()
        }
    return result


def extract_branching_by_visit(df: pd.DataFrame, max_visit: int = 3,
                               min_samples: int = 30) -> dict:
    """
    A1 termination-fix input: branching probabilities conditioned on how
    often the current activity has already occurred within the case
    ("visit count"). The memoryless P(next | current) systematically
    understates loop-exit probabilities — in BPIC-17 the more often e.g.
    W_Validate application has run in a case, the likelier the next step
    is an exit rather than another loop round. A simulation driven by the
    unconditioned table therefore cycles (measured: 4.35 W_Validate
    application per case vs. 0.50 real; only 2% of cases terminate).

    Buckets: "1", "2", ..., "{max_visit}+" per activity. Buckets with
    fewer than *min_samples* observations are dropped — the simulation
    falls back to the global branching_probs there.
    """
    df2 = df.sort_values(["case_id", "timestamp"]).copy()
    df2["next_activity"] = df2.groupby("case_id")["activity"].shift(-1)
    df2["visit"] = df2.groupby(["case_id", "activity"]).cumcount() + 1
    df2 = df2.dropna(subset=["next_activity"])
    df2["visit_bucket"] = df2["visit"].map(
        lambda v: str(v) if v < max_visit else f"{max_visit}+")

    result: dict = {}
    for (act, bucket), grp in df2.groupby(["activity", "visit_bucket"]):
        counts = grp["next_activity"].value_counts()
        total = int(counts.sum())
        if total < min_samples:
            continue
        result.setdefault(act, {})[bucket] = {
            next_act: round(float(cnt / total), 4)
            for next_act, cnt in counts.items()
        }
    return result


def extract_case_attributes(df: pd.DataFrame) -> dict:
    """
    Section 1.5 Advanced I input: distributions for the case/runtime data
    attributes used as features for the decision-point classifiers.

    Two groups, mirroring when the attribute is actually known during a case:

    - spawn attributes (case:ApplicationType, case:LoanGoal,
      case:RequestedAmount): constant per case, present from
      A_Create Application onward, so every decision point can use them
      (verified: exactly 1 distinct value per case in the log).
      LoanGoal and RequestedAmount are learned conditional on
      ApplicationType, since their distributions differ noticeably between
      "New credit" and "Limit raise" cases.

    - offer attributes (OfferedAmount, NumberOfTerms, MonthlyCost,
      FirstWithdrawalAmount, CreditScore): only set once an offer exists,
      i.e. on O_Create Offer (verified: 100% of non-null values for these
      columns occur on that activity). Decision points reached before the
      first O_Create Offer of a case won't have these yet.
      CreditScore and FirstWithdrawalAmount have a large point mass at 0
      (~65% / ~30% of offers) that isn't part of the continuous spread, so
      they're modelled as (probability of 0) + (fitted distribution on the
      nonzero remainder).

    Deliberately excluded: "Accepted" / "Selected" (also written at
    O_Create Offer) — these record the outcome of the very decisions
    (O_Accepted/O_Cancelled/O_Refused) we want to predict, so using them as
    input features would leak the label.
    """
    case_level = df.drop_duplicates("case_id")

    application_type = {
        k: round(float(v), 4)
        for k, v in case_level["case:ApplicationType"].value_counts(normalize=True).items()
    }

    loan_goal_given_type = {}
    requested_amount_given_type = {}
    for app_type, grp in case_level.groupby("case:ApplicationType"):
        loan_goal_given_type[app_type] = {
            k: round(float(v), 4)
            for k, v in grp["case:LoanGoal"].value_counts(normalize=True).items()
        }
        requested_amount_given_type[app_type] = fit_best_distribution(
            grp["case:RequestedAmount"].dropna().values
        )

    offers = df[df["activity"] == "O_Create Offer"]

    def fit_with_zero_mass(series: pd.Series) -> dict:
        zero_prob = round(float((series == 0).mean()), 4)
        nonzero = series[series != 0].dropna().values
        return {"zero_prob": zero_prob, "nonzero": fit_best_distribution(nonzero)}

    offer_attributes = {
        "OfferedAmount":          fit_best_distribution(offers["OfferedAmount"].dropna().values),
        "NumberOfTerms":          fit_best_distribution(offers["NumberOfTerms"].dropna().values),
        "MonthlyCost":            fit_best_distribution(offers["MonthlyCost"].dropna().values),
        "FirstWithdrawalAmount":  fit_with_zero_mass(offers["FirstWithdrawalAmount"]),
        "CreditScore":            fit_with_zero_mass(offers["CreditScore"]),
    }

    return {
        "spawn_attributes": {
            "ApplicationType":              application_type,
            "LoanGoal_given_ApplicationType":       loan_goal_given_type,
            "RequestedAmount_given_ApplicationType": requested_amount_given_type,
        },
        "offer_attributes_set_at": "O_Create Offer",
        "offer_attributes": offer_attributes,
    }


def extract_resources(df: pd.DataFrame) -> dict:
    if "resource" not in df.columns:
        return {"note": "No resource column found in log."}

    res_counts = df["resource"].value_counts()
    res_activities = (
        df.dropna(subset=["resource"])
        .groupby("resource")["activity"]
        .apply(lambda x: sorted(x.unique().tolist()))
    )

    return {
        "n_resources": int(df["resource"].nunique()),
        "top_20_by_events": {
            str(r): int(c)
            for r, c in res_counts.head(20).items()
        },
        "resource_activity_map": {
            str(r): acts
            for r, acts in res_activities.items()
        },
    }


def extract_time_range(df: pd.DataFrame) -> dict:
    return {
        "start": str(df["timestamp"].min()),
        "end":   str(df["timestamp"].max()),
        "span_days": round(
            float((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400), 1
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Extract simulation inputs from BPIC-17 event log.")
    parser.add_argument("--log", required=True, help="Path to .xes or .csv event log file")
    parser.add_argument("--out", default="simulation_inputs.json", help="Output JSON file (default: simulation_inputs.json)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"[ERROR] File not found: {log_path}")

    print(f"[1/8] Loading log from '{log_path}' ...")
    df = load_log(log_path)
    print(f"      → {len(df):,} events, {df['case_id'].nunique():,} cases loaded")

    # One row per real activity occurrence (not per lifecycle transition) —
    # see filter_to_complete(). basic_stats/branching_probs/process_variants
    # and arrival_rate all describe activity-level or case-level behaviour,
    # so they must run on this, not the raw multi-lifecycle event stream.
    df_complete = filter_to_complete(df)
    print(f"      → {len(df_complete):,} events after filtering to one-per-activity-occurrence")

    print("[2/8] Basic statistics ...")
    basic = extract_basic_stats(df_complete)

    print("[3/8] Activity frequencies ...")
    activities = extract_activities(df_complete)

    print("[4/8] Processing times (fitting distributions) ...")
    proc_times = extract_processing_times(df)  # needs raw start/complete pairs

    print("[5/8] Arrival rates ...")
    arrivals = extract_arrival_rate(df_complete)

    print("[6/8] Process variants + branching probabilities ...")
    variants  = extract_process_variants(df_complete)
    branching = extract_branching(df_complete)
    branching_by_visit = extract_branching_by_visit(df_complete)

    print("[7/8] Resources ...")
    resources = extract_resources(df)

    print("[8/8] Case attributes (Section 1.5 Advanced I) ...")
    case_attributes = extract_case_attributes(df)

    output = {
        "log_file":         log_path.name,
        "time_range":       extract_time_range(df),
        "basic_stats":      basic,
        "activity_counts":  activities,
        "processing_times": proc_times,
        "arrival_rate":     arrivals,
        "process_variants": variants,
        "branching_probs":  branching,
        "branching_probs_by_visit": branching_by_visit,
        "resources":        resources,
        "case_attributes":  case_attributes,
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Done! Results saved to '{out_path}'")
    print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\n   → Share '{out_path}' with Claude to build the real simulator.\n")


if __name__ == "__main__":
    main()
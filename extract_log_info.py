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

Outputs
-------
    simulation_inputs.json          legacy simulation inputs
    simulation_inputs_active.json   legacy-compatible inputs plus the active
                                    lifecycle parameter block
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
if hasattr(sys.stdout, "reconfigure"):
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
    # Stable sort preserves source-log order for equal timestamps. Terminal
    # continuation mining keys on this order, not timestamp alone (§5.1).
    df = df.sort_values(["case_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["event_order"] = df.groupby("case_id", sort=False).cumcount()
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
    Repeated occurrences are paired by their stable per-(case, activity)
    sequence number; merging on only case/activity would cross-join them.
    Falls back to single-timestamp diff between consecutive events if
    no lifecycle column exists.
    """
    result = {}

    if "lifecycle" in df.columns:
        order_cols = ["case_id", "timestamp"]
        if "event_order" in df.columns:
            order_cols.append("event_order")
        ordered = df.sort_values(order_cols, kind="stable")
        starts = ordered[
            ordered["lifecycle"].str.lower().isin(["start", "assign"])
        ].copy()
        completes = ordered[
            ordered["lifecycle"].str.lower().eq("complete")
        ].copy()
        starts["instance_seq"] = starts.groupby(
            ["case_id", "activity"], sort=False).cumcount()
        completes["instance_seq"] = completes.groupby(
            ["case_id", "activity"], sort=False).cumcount()

        merged = pd.merge(
            starts[["case_id", "activity", "instance_seq", "timestamp"]],
            completes[["case_id", "activity", "instance_seq", "timestamp"]],
            on=["case_id", "activity", "instance_seq"],
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


# ════════════════════════════════════════════════════════════════════════════
# Lifecycle segmentation (active-time + churn model) — see implementationplan §5.1
# ════════════════════════════════════════════════════════════════════════════
#
# BPIC-17 work items (W_ activities) follow the lifecycle grammar
#     schedule → start → (suspend → resume)* → complete | ate_abort | withdraw
# The single `start→complete` elapsed span the legacy extractor fits is inflated
# 40–4000× by suspend/resume waits. This block reconstructs each work-item
# *instance* from the ordered transition stream and mines:
#   - active-session-second distributions   (processing_times, active target)
#   - session-end hazard   P(complete | a running session ends)
#   - suspend-end hazard   P(resume    | a suspended item continues)
#   - resume-gap residual  (suspend→resume-ready external wait, calendar-aware)
#   - terminal continuation  next activity per (activity, terminal outcome)
#   - withdraw hazard      time-to-withdraw while merely SCHEDULED
# These feed the `active` runtime mode; the legacy path is untouched.

LIFECYCLE_TERMINALS = ("complete", "ate_abort", "withdraw")


def _off_shift_tail_seconds(suspend_ts, resume_ts, resume_resource,
                            availability_model: dict) -> float:
    """Length of the contiguous off-shift interval *immediately preceding* a resume.

    Implements the LOCKED calendar-aware-residual step (implementationplan §5.1):
    subtract only the single off-shift tail that ends at the observed resume — not
    every night/weekend inside the gap. Uses the historical resume resource's
    fitted deterministic weekly windows and discovered public holidays. Sampled
    vacations are deliberately ignored by the locked residual contract.
    """
    from datetime import datetime, timedelta

    delta = (resume_ts - suspend_ts).total_seconds()
    if delta <= 0:
        return 0.0

    resource = str(resume_resource) if pd.notna(resume_resource) else None
    if not resource or resource in set(availability_model.get("system", [])):
        return 0.0
    windows = availability_model.get("windows", {}).get(resource)
    if not windows:
        return 0.0
    holidays = set(availability_model.get("holidays", []))

    def window(t):
        if t.date().isoformat() in holidays:
            return None
        return windows.get(str(t.weekday()))

    def on_shift(t) -> bool:
        hours = window(t)
        if not hours:
            return False
        h = t.hour + t.minute / 60.0 + t.second / 3600.0
        return float(hours[0]) <= h < float(hours[1])

    # If the resume itself lands in a working window, there is no preceding
    # off-shift tail to attribute to the calendar.
    if on_shift(resume_ts):
        return 0.0

    # Walk back to the most recent deterministic shift close. Holidays and days
    # without a window extend the same contiguous off-shift tail.
    for back in range(0, 15):
        d = (resume_ts - timedelta(days=back)).date()
        if d.isoformat() in holidays:
            continue
        hours = windows.get(str(d.weekday()))
        if not hours:
            continue
        midnight = datetime.combine(d, datetime.min.time()).replace(
            tzinfo=resume_ts.tzinfo)
        close = midnight + timedelta(seconds=round(float(hours[1]) * 3600))
        if close <= resume_ts:
            tail = (resume_ts - close).total_seconds()
            return float(min(max(tail, 0.0), delta))
    # No working day found in the look-back window — attribute the whole gap.
    return float(delta)


def segment_work_items(df: pd.DataFrame) -> dict:
    """Reconstruct W_ work-item instances from the ordered lifecycle stream.

    Returns raw aggregates (not yet fitted): per-activity active-session seconds,
    session-end / suspend-end hazard counts, raw resume gaps (with resume resource
    and suspend/resume timestamps for the calendar residual), withdraw wait times,
    suspends-per-instance, and per-case ordered terminal tokens for continuation
    mining. Only W_ activities enter the state machine; A_/O_ are atomic elsewhere.
    """
    active_sessions   = defaultdict(list)   # act -> [active seconds per running session]
    session_end_hazard = defaultdict(lambda: {"complete": 0, "suspend": 0, "abort": 0})
    suspend_end_hazard = defaultdict(lambda: {"resume": 0, "abort": 0})
    raw_resume_gaps   = defaultdict(list)   # act -> [(suspend_ts, resume_ts, resume_res)]
    withdraw_waits    = defaultdict(list)   # act -> [schedule→withdraw seconds]
    suspends_per_inst = defaultdict(list)   # act -> [#suspends per instance]
    terminal_tokens   = []                  # [(case_id, order, ts, activity, outcome)]
    next_tokens       = []                  # [(case_id, order, ts, activity)] occurrences

    has_life = "lifecycle" in df.columns
    if not has_life:
        return {
            "active_sessions": active_sessions, "session_end_hazard": session_end_hazard,
            "suspend_end_hazard": suspend_end_hazard, "raw_resume_gaps": raw_resume_gaps,
            "withdraw_waits": withdraw_waits, "suspends_per_instance": suspends_per_inst,
            "terminal_tokens": terminal_tokens, "next_tokens": next_tokens,
        }

    lc = df["lifecycle"].str.lower()
    res_col = "resource" if "resource" in df.columns else None

    for case_id, cgrp in df.assign(lc_norm=lc).groupby("case_id", sort=False):
        cgrp = cgrp.sort_values("timestamp")
        # Every A_/O_ complete and every W_ terminal is an occurrence token used to
        # find "what activity comes next" for terminal continuation.
        for act, ts, order, lct in zip(
                cgrp["activity"], cgrp["timestamp"], cgrp["event_order"], cgrp["lc_norm"]):
            if not act.startswith("W_") and lct == "complete":
                next_tokens.append((case_id, int(order), ts, act))

        # Segment W_ instances per activity within the case.
        w = cgrp[cgrp["activity"].str.startswith("W_")]
        for act, agrp in w.groupby("activity", sort=False):
            cur = None  # {"schedule_ts","running_since","suspend_ts","active","nsusp"}
            for row in agrp.itertuples(index=False):
                lct = row.lc_norm
                ts = row.timestamp
                resource = getattr(row, "resource", None) if res_col else None
                order = int(row.event_order)
                if lct == "schedule":
                    if cur is None:
                        cur = {"schedule_ts": ts, "running_since": None,
                                "suspend_ts": None, "active": 0.0, "nsusp": 0}
                elif lct == "start":
                    if cur is None:
                        cur = {"schedule_ts": None, "running_since": ts,
                                "suspend_ts": None, "active": 0.0, "nsusp": 0}
                    else:
                        cur["running_since"] = ts
                        cur["suspend_ts"] = None
                elif lct == "resume":
                    if cur is not None and cur["suspend_ts"] is not None:
                        gap = (ts - cur["suspend_ts"]).total_seconds()
                        if gap >= 0:
                            raw_resume_gaps[act].append((cur["suspend_ts"], ts, resource))
                        suspend_end_hazard[act]["resume"] += 1
                        cur["running_since"] = ts
                        cur["suspend_ts"] = None
                elif lct == "suspend":
                    if cur is not None and cur["running_since"] is not None:
                        seg = (ts - cur["running_since"]).total_seconds()
                        if seg >= 0:
                            active_sessions[act].append(seg)
                        session_end_hazard[act]["suspend"] += 1
                        cur["nsusp"] += 1
                        cur["running_since"] = None
                        cur["suspend_ts"] = ts
                elif lct in LIFECYCLE_TERMINALS:
                    if cur is None:
                        cur = {"schedule_ts": None, "running_since": None,
                                "suspend_ts": None, "active": 0.0, "nsusp": 0}
                    if lct == "complete":
                        if cur["running_since"] is not None:
                            seg = (ts - cur["running_since"]).total_seconds()
                            if seg >= 0:
                                active_sessions[act].append(seg)
                            session_end_hazard[act]["complete"] += 1
                    elif lct == "ate_abort":
                        # Abort from SUSPENDED (the modelled path) closes the wait;
                        # a direct RUNNING→abort (no preceding suspend) is measured
                        # separately to inform whether v1 needs that edge (§8).
                        if cur["suspend_ts"] is not None:
                            suspend_end_hazard[act]["abort"] += 1
                        elif cur["running_since"] is not None:
                            seg = (ts - cur["running_since"]).total_seconds()
                            if seg >= 0:
                                active_sessions[act].append(seg)
                            session_end_hazard[act]["abort"] += 1
                    elif lct == "withdraw":
                        # Withdraw from SCHEDULED (never started): time-to-withdraw.
                        if cur["schedule_ts"] is not None and cur["running_since"] is None:
                            wait = (ts - cur["schedule_ts"]).total_seconds()
                            if wait >= 0:
                                withdraw_waits[act].append(wait)
                    suspends_per_inst[act].append(cur["nsusp"])
                    terminal_tokens.append((case_id, order, ts, act, lct))
                    # A W_ occurrence is represented at its terminal transition,
                    # matching filter_to_complete's occurrence-level process
                    # abstraction. Using its earlier schedule would skip the
                    # intervening A_/O_ process steps (e.g. W_Complete ->
                    # A_Complete -> ...), which is not a legal runtime route.
                    next_tokens.append((case_id, order, ts, act))
                    cur = None
            # Instance still open at log end: keep its active accumulation for the
            # duration fit but do not fabricate a terminal outcome.
            if cur is not None and cur["running_since"] is not None:
                pass

    return {
        "active_sessions": active_sessions, "session_end_hazard": session_end_hazard,
        "suspend_end_hazard": suspend_end_hazard, "raw_resume_gaps": raw_resume_gaps,
        "withdraw_waits": withdraw_waits, "suspends_per_instance": suspends_per_inst,
        "terminal_tokens": terminal_tokens, "next_tokens": next_tokens,
    }


def extract_lifecycle(df: pd.DataFrame, availability_model: dict) -> dict:
    """Fit the active-time + churn model blocks for `simulation_inputs_active.json`.

    Consumes segment_work_items() aggregates and produces the parametric blocks the
    `active` runtime mode loads (implementationplan §4.4 / §5.1).
    """
    import numpy as _np

    seg = segment_work_items(df)

    # -- processing_times: active-session-second distributions per activity -------
    processing_times = {}
    for act, secs in seg["active_sessions"].items():
        arr = _np.asarray(secs, dtype=float)
        if len(arr) == 0:
            continue
        processing_times[act] = fit_best_distribution(arr)

    # -- session-end hazard: P(complete | running session ends) -------------------
    # A running session ends via suspend or a terminal-from-RUNNING; the modelled
    # competing outcome is suspend, so P(complete) over {complete, suspend}.
    session_end_probs = {}
    direct_abort_from_running = {}
    for act, h in seg["session_end_hazard"].items():
        denom = h["complete"] + h["suspend"]
        session_end_probs[act] = round(h["complete"] / denom, 6) if denom else 0.0
        if h["abort"]:
            direct_abort_from_running[act] = int(h["abort"])

    # -- suspend-end hazard: P(resume | suspended) --------------------------------
    suspend_end_probs = {}
    for act, h in seg["suspend_end_hazard"].items():
        denom = h["resume"] + h["abort"]
        suspend_end_probs[act] = round(h["resume"] / denom, 6) if denom else 0.0

    # -- resume-gap residual: max(0, Δ − calendar_tail) per activity --------------
    resume_gap_params = {}
    for act, gaps in seg["raw_resume_gaps"].items():
        residuals = []
        for suspend_ts, resume_ts, resume_resource in gaps:
            tail = _off_shift_tail_seconds(
                suspend_ts, resume_ts, resume_resource, availability_model)
            delta = (resume_ts - suspend_ts).total_seconds()
            residuals.append(max(0.0, delta - tail))
        arr = _np.asarray(residuals, dtype=float)
        if len(arr) == 0:
            continue
        fit = fit_best_distribution(arr)
        fit["n_gaps"] = int(len(arr))
        fit["zero_prob"] = round(float((arr == 0).mean()), 6)
        resume_gap_params[act] = fit

    # -- withdraw hazard: time-to-withdraw while SCHEDULED ------------------------
    withdraw_hazard = {}
    for act, waits in seg["withdraw_waits"].items():
        arr = _np.asarray(waits, dtype=float)
        if len(arr) == 0:
            continue
        fit = fit_best_distribution(arr)
        fit["n"] = int(len(arr))
        withdraw_hazard[act] = fit

    # -- terminal continuation: next activity per (W_ activity, outcome) ----------
    # For each W_ terminal token, find the next occurrence token in the same case.
    by_case_next = defaultdict(list)
    for case_id, order, ts, act in seg["next_tokens"]:
        by_case_next[case_id].append((order, ts, act))
    for toks in by_case_next.values():
        toks.sort(key=lambda x: x[0])

    cont_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # act->outcome->next->n
    end_counts  = defaultdict(lambda: defaultdict(int))                       # act->outcome->#case-end
    for case_id, order, ts, act, outcome in seg["terminal_tokens"]:
        toks = by_case_next[case_id]
        nxt = None
        for token_order, _tts, tact in toks:
            if token_order > order:
                nxt = tact
                break
        if nxt is None:
            end_counts[act][outcome] += 1
        else:
            cont_counts[act][outcome][nxt] += 1

    terminal_continuation = {}
    for act in set(list(cont_counts) + list(end_counts)):
        terminal_continuation[act] = {}
        for outcome in LIFECYCLE_TERMINALS:
            nexts = cont_counts.get(act, {}).get(outcome, {})
            n_end = end_counts.get(act, {}).get(outcome, 0)
            total = sum(nexts.values()) + n_end
            if total == 0:
                continue
            dist = {na: round(c / total, 4) for na, c in nexts.items()}
            if n_end:
                dist["__CASE_END__"] = round(n_end / total, 4)
            terminal_continuation[act][outcome] = dist

    # -- suspends-per-instance summary (validation only) --------------------------
    suspends_per_instance = {}
    for act, xs in seg["suspends_per_instance"].items():
        arr = _np.asarray(xs, dtype=float)
        if len(arr) == 0:
            continue
        suspends_per_instance[act] = {
            "mean": round(float(arr.mean()), 4),
            "median": float(_np.median(arr)),
            "p90": float(_np.percentile(arr, 90)),
            "max": int(arr.max()),
            "pct_churned": round(float((arr > 0).mean()), 4),
            "n_instances": int(len(arr)),
        }

    return {
        "lifecycle_schema": "active_v1",
        "calendar_tail": {
            "availability_model": "historical resume resource weekly windows + holidays",
            "vacations_subtracted": False,
            "note": "Only the contiguous deterministic off-shift interval immediately "
                    "preceding resume is subtracted; this is a calibrated residual, "
                    "not a causal customer-wait decomposition.",
        },
        "processing_times": processing_times,
        "session_end_probs": session_end_probs,
        "suspend_end_probs": suspend_end_probs,
        "resume_gap_params": resume_gap_params,
        "withdraw_hazard": withdraw_hazard,
        "terminal_continuation": terminal_continuation,
        "suspends_per_instance": suspends_per_instance,
        "direct_abort_from_running": direct_abort_from_running,
    }


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

    # Hour-of-day / day-of-week arrival shape (Section 1.2 Advanced: what
    # the MDN model needs to beat, since a static LogNormal can only match
    # the mean rate, not this structure). Normalized histograms so
    # scripts/metrics.py::arrival_profile_error can compare vs. the
    # simulated log's own histograms directly (both sum to 1).
    hod_counts = first_events.dt.hour.value_counts().reindex(range(24), fill_value=0)
    fit["hod_profile"] = [round(float(c), 6) for c in (hod_counts / hod_counts.sum())]
    dow_counts = first_events.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    fit["dow_profile"] = [round(float(c), 6) for c in (dow_counts / dow_counts.sum())]

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
    parser.add_argument("--lifecycle", action="store_true",
                        help="Also mine the active-time + suspend/resume churn model "
                             "(implementationplan §5.1) and add a `lifecycle` block to "
                             "the output. Legacy blocks stay byte-identical; intended "
                             "for --out simulation_inputs_active.json.")
    parser.add_argument("--availability-model", type=Path,
                        default=Path("models/availability_model.json"),
                        help="Historical deterministic resource calendars used for the "
                             "active resume-gap residual.")
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

    lifecycle_enabled = args.lifecycle or Path(args.out).stem.endswith("_active")
    if lifecycle_enabled:
        if not args.availability_model.is_file():
            sys.exit(f"[ERROR] Availability model not found: {args.availability_model}")
        availability_model = json.loads(
            args.availability_model.read_text(encoding="utf-8"))
        print("[+]   Lifecycle model (active-time + suspend/resume churn) ...")
        output["lifecycle"] = extract_lifecycle(df, availability_model)
        n_acts = len(output["lifecycle"]["processing_times"])
        print(f"      → active-session distributions for {n_acts} W_ activities")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Done! Results saved to '{out_path}'")
    print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\n   → Share '{out_path}' with Claude to build the real simulator.\n")


if __name__ == "__main__":
    main()

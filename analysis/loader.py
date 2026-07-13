"""
loader.py — canonical BPIC-17 event log access for the resource analysis.

Everything downstream (notebooks *and* the simulation's resource components)
loads the log through here, so the four decisions below are made exactly once
and cannot drift between the analysis and the model it feeds.

----------------------------------------------------------------------------
Decision 1 — Timestamps are UTC and must be converted to Europe/Amsterdam
----------------------------------------------------------------------------
The XES stores timestamps with a `Z` suffix, i.e. UTC. Working hours, however,
are a *local* phenomenon: a Dutch bank opens at 09:00 Amsterdam time, not
09:00 UTC. Reading the log in UTC therefore smears the workday by the DST
offset (+1h in winter, +2h in summer).

The data shows this directly. Taking the 5th percentile of event hour as "when
the workday opens", for human workflow events:

                       winter (Nov-Feb)   summer (May-Aug)   apparent shift
    UTC                     07:54              06:44             1.17 h
    Europe/Amsterdam        08:53              08:44             0.15 h

In UTC the workday appears to open an hour earlier in summer — a pure DST
artifact. Converted to Amsterdam local time the seasonal shift collapses and
the workday opens at ~08:45 year-round. So: convert, then read wall-clock.

----------------------------------------------------------------------------
Decision 2 — Only Workflow (W_) events are human work
----------------------------------------------------------------------------
BPIC-17 events carry an `EventOrigin` of Application, Offer, or Workflow.
Application (A_) and Offer (O_) events only ever have `lifecycle:transition ==
complete`: they are instantaneous state changes written by the system when an
application or offer object changes. They have no duration and do not occupy a
person.

Only Workflow (W_) events carry a real human lifecycle — schedule, start,
suspend, resume, complete, ate_abort, withdraw. In particular there are
128,227 W_ `start` events, and a `start` is precisely the signal "a resource
began working on this". Availability is therefore derived from W_ events.

(The previously committed analysis filtered the log to `lifecycle == complete`,
which keeps every automated A_/O_ event and discards every W_ `start` — that is,
it dropped exactly the signal it was trying to measure.)

----------------------------------------------------------------------------
Decision 3 — Presence spans the lifecycle, not just completions
----------------------------------------------------------------------------
A resource is "present" on a day from its first work signal to its last. Any
W_ transition performed by that resource counts as a signal, because each one
is an interaction with the system by that person.

----------------------------------------------------------------------------
Decision 4 — System/batch accounts are separated structurally, not by name
----------------------------------------------------------------------------
The log spans 398 calendar days, of which 284 (71.4%) are weekdays. A human
who never missed a single weekday and never worked a weekend would still be
active on at most 71.4% of calendar days. Any account exceeding that ceiling
cannot be following a human work calendar.

Exactly one account breaches it: User_1, active on 395 of 398 days (99.2%).
The next highest is User_100 at 63.3% — comfortably human. User_1 is also a
5.6x outlier in event volume, does 4.3x the median throughput per active day,
and works 23.7% of its events on weekends (median: 1.8%).

Including it would make "resource availability" look close to 24/7. It is
modelled as an automated system account, not staff. The rule is derived from
the structure of a work week rather than tuned, and it is recomputed from the
data each time rather than hardcoded to the string "User_1".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

LOCAL_TZ = "Europe/Amsterdam"

# Where to look for the raw log, in order. Override with $BPIC17_LOG.
_SEARCH_PATHS = [
    Path("data/BPI_Challenge_2017.csv"),
    Path("data/BPI Challenge 2017.xes.gz"),
    Path.home() / "Practical_Business_Process/data/BPI_Challenge_2017.csv",
    Path.home() / "Practical_Business_Process/data/BPI Challenge 2017.xes.gz",
]

_USECOLS = [
    "case:concept:name", "concept:name", "time:timestamp", "org:resource",
    "lifecycle:transition", "EventOrigin", "case:LoanGoal",
    "case:ApplicationType", "case:RequestedAmount",
]


def resolve_log_path(path: Optional[str | Path] = None) -> Path:
    """Locate the raw BPIC-17 log.

    Order: explicit argument, then $BPIC17_LOG, then the known locations. The
    log is gitignored (too large to track), so it lives outside the repo.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BPIC-17 log not found at {p}")
        return p

    env = os.environ.get("BPIC17_LOG")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"$BPIC17_LOG points to {p}, which does not exist")
        return p

    for cand in _SEARCH_PATHS:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "Could not find the BPIC-17 log. Set $BPIC17_LOG to its path, or place "
        "it at data/BPI_Challenge_2017.csv.\nLooked in:\n  "
        + "\n  ".join(str(p) for p in _SEARCH_PATHS)
    )


def load_events(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw log with timestamps in Amsterdam local time (Decision 1).

    Adds convenience columns: ``date``, ``hour`` (fractional), ``dow``.
    """
    p = resolve_log_path(path)

    if p.suffix == ".csv":
        df = pd.read_csv(p, usecols=lambda c: c in _USECOLS)
    else:
        import pm4py
        df = pm4py.convert_to_dataframe(pm4py.read_xes(str(p)))

    df["time:timestamp"] = (
        pd.to_datetime(df["time:timestamp"], utc=True, format="mixed")
        .dt.tz_convert(LOCAL_TZ)
    )

    # The XES carries a 1970-01-01 epoch artifact row; it is not a real event.
    df = df[df["time:timestamp"].dt.year >= 2016].copy()

    ts = df["time:timestamp"]
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    df["dow"] = ts.dt.dayofweek

    return df


@dataclass
class ResourceRoles:
    """Which accounts are human staff and which are automated (Decision 4)."""
    humans: list[str]
    system: list[str]
    coverage: pd.Series      # resource -> active_days / span_days
    weekday_ceiling: float   # the structural threshold that separates them


def classify_resources(df: pd.DataFrame) -> ResourceRoles:
    """Split accounts into human vs system by calendar-day coverage.

    A human cannot be active on more calendar days than there are weekdays in
    the log's span. Anything above that ceiling is an automated account.
    """
    days = pd.date_range(df["time:timestamp"].min().date(),
                         df["time:timestamp"].max().date(), freq="D")
    ceiling = sum(1 for d in days if d.dayofweek < 5) / len(days)

    coverage = df.groupby("org:resource")["date"].nunique() / len(days)
    system = sorted(coverage[coverage > ceiling].index)
    humans = sorted(coverage[coverage <= ceiling].index)

    return ResourceRoles(
        humans=humans, system=system,
        coverage=coverage.sort_values(ascending=False),
        weekday_ceiling=ceiling,
    )


def work_events(df: pd.DataFrame, roles: Optional[ResourceRoles] = None) -> pd.DataFrame:
    """Human work events only: Workflow origin, staff accounts (Decisions 2 & 4)."""
    roles = roles or classify_resources(df)
    return df[
        (df["EventOrigin"] == "Workflow")
        & (df["org:resource"].isin(roles.humans))
    ].copy()


def daily_presence(work: pd.DataFrame) -> pd.DataFrame:
    """Per resource and day: when they started and stopped working (Decision 3).

    Returns one row per (resource, date) on which the resource was active, with
    ``first``/``last`` as fractional local hours and the number of events.
    """
    g = work.groupby(["org:resource", "date"])
    out = g.agg(
        first=("hour", "min"),
        last=("hour", "max"),
        n_events=("hour", "size"),
    ).reset_index()
    out["dow"] = pd.to_datetime(out["date"]).dt.dayofweek
    out["span"] = out["last"] - out["first"]
    return out

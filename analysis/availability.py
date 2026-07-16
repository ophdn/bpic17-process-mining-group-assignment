"""
availability.py — Section 1.6, resource availability models.

Two models, both fitted from the BPIC-17 log via `analysis.loader`:

  WeeklyAvailability (Basic)
      Per resource and weekday: when the workday opens, when it closes, and how
      likely the resource is to work that day at all.

  YearlyAvailability (Advanced)
      The weekly profile, plus a discovered public-holiday calendar, plus
      per-resource vacation behaviour fitted as a distribution and sampled
      forward, plus sporadic absence.

Design decisions are recorded at each function. The headline ones:

Why a weekly cycle and not a fortnightly one
--------------------------------------------
The assignment suggests "an interval, e.g., a two-week interval". We tested for
a genuine fortnightly signal by splitting each resource-weekday's history into
odd/even ISO weeks and comparing the median start and end times. The A/B split
is indistinguishable from splitting the *same* data at random:

                          median |shift|   >1h apart
    A-week vs B-week          0.36 h         27.8%
    random split (null)       0.39 h         30.4%

So there is no fortnightly structure — the cycle is weekly. We model a weekly
profile and tile it across the two-week interval. Fitting an A/B week would be
fitting noise, and would halve the data behind every estimate.

Why robust quantiles and not min/max
------------------------------------
A single 3 a.m. event stretches a min/max window across the whole night. The
observed workday span already runs to a max of 22.8 h. Fitting the window as
[q10 of first event, q90 of last event] trades a little coverage for a much
tighter, more honest window:

    window fitted as       coverage   window utilisation
    min / max               100.0%          49.4%
    q10 / q90                97.4%          57.2%     <- chosen
    q25 / q75                92.3%          65.2%
    global 9-17 baseline     80.5%          68.6%

Coverage is the share of real work events falling inside the modelled window;
utilisation is how much of the modelled-open time actually contains work. The
per-resource model reaches 97.4% coverage where a single global 9-17 calendar
manages only 80.5% — which is the empirical case for discovering shifts per
resource rather than assuming office hours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Fitting the open/close window. See module docstring for the sweep behind these.
START_Q, END_Q = 0.10, 0.90

# A resource needs at least this many observed active days before we trust a
# per-weekday shift estimate for it.
MIN_ACTIVE_DAYS = 20

# A weekday on which the workforce collectively vanishes is a public holiday,
# not 40 coincidental absences. Threshold is relative to a *local* baseline
# (see discover_holidays) because the log's headcount ramps at both ends.
HOLIDAY_MAX_HEADCOUNT = 0.40

# Consecutive absent working days that count as a vacation rather than a
# sporadic day off. One working week — a floor; the real threshold is derived
# per resource in `_chance_threshold`.
MIN_VACATION_DAYS = 5

# Below this working rate, leave is not identifiable — a resource who works one
# day in four produces long absence runs on its own, and any "vacation" fitted
# for it is noise. See `discover_vacations` for the evidence. Their absence is
# modelled by `p_work` instead, which is where it belongs.
VACATION_MIN_WORK_RATE = 0.60


# ──────────────────────────────────────────────────────────────────────────
# Basic: weekly availability
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class WeeklyAvailability:
    """Per (resource, weekday): open hour, close hour, P(works that day)."""

    windows: Dict[str, Dict[int, Tuple[float, float]]]
    p_work: Dict[str, Dict[int, float]]

    def is_open(self, resource: str, dow: int, hour: float) -> bool:
        w = self.windows.get(resource, {}).get(dow)
        return w is not None and w[0] <= hour <= w[1]

    def resources(self) -> list[str]:
        return sorted(self.windows)


def fit_weekly(
    presence: pd.DataFrame,
    span_days: Iterable[pd.Timestamp],
    min_active_days: int = MIN_ACTIVE_DAYS,
    sparse_floor: Optional[int] = None,
) -> WeeklyAvailability:
    """Fit the weekly shift profile from per-resource-per-day presence.

    `presence` is the frame from `loader.daily_presence`; `span_days` is every
    calendar day the log covers (needed to turn "days worked" into a rate).

    Two tiers of resource, by how much presence data they have:

    Tier A — well-observed (>= `min_active_days` active days). A full
        per-(resource, weekday) window `[q10 first, q90 last]`, as the module
        docstring describes.

    Tier B — sparse (`sparse_floor` <= active days < `min_active_days`), fitted
        only when `sparse_floor` is set. Too few distinct days to trust a shift
        *per weekday*, but usually plenty of events (a resource may log hundreds
        of events across a dozen days). So we fit a *single* pooled window per
        resource from all its events and open it on the weekdays it was actually
        seen working — a coarser, weekday-agnostic assumption. This is strictly
        better than the runtime fallback it replaces (a resource with no window
        is treated as on shift 24/7, the opposite of sparse), which is why the
        deployed model sets `sparse_floor=1`: every human with any W_ signal
        gets a data-driven window, and only genuine system accounts (no W_
        lifecycle at all) remain windowless.

    Default `sparse_floor=None` keeps Tier A only, reproducing the original fit
    bit-for-bit.
    """
    dow_occurrences = pd.Series([d.dayofweek for d in span_days]).value_counts()

    active_days = presence.groupby("org:resource").size()
    keep = active_days[active_days >= min_active_days].index
    p = presence[presence["org:resource"].isin(keep)]

    windows: Dict[str, Dict[int, Tuple[float, float]]] = {}
    p_work: Dict[str, Dict[int, float]] = {}

    for (res, dow), g in p.groupby(["org:resource", "dow"]):
        start = float(g["first"].quantile(START_Q))
        end = float(g["last"].quantile(END_Q))
        if end <= start:            # degenerate (one observation) — skip
            continue
        windows.setdefault(res, {})[int(dow)] = (start, end)
        p_work.setdefault(res, {})[int(dow)] = float(len(g) / dow_occurrences[dow])

    if sparse_floor is not None:
        sparse = active_days[(active_days >= sparse_floor)
                             & (active_days < min_active_days)].index
        for res in sparse:
            g = presence[presence["org:resource"] == res]
            # One pooled window from ALL the resource's events (weekday-agnostic):
            # too few distinct days to split by weekday, but the pooled quantiles
            # are stable. Fall back to the observed extremes if the quantile
            # window is degenerate (very few events on a single day).
            start = float(g["first"].quantile(START_Q))
            end = float(g["last"].quantile(END_Q))
            if end <= start:
                start, end = float(g["first"].min()), float(g["last"].max())
            if end <= start:
                continue            # a single instantaneous event — nothing to fit
            for dow, gd in g.groupby("dow"):
                windows.setdefault(res, {})[int(dow)] = (start, end)
                p_work.setdefault(res, {})[int(dow)] = float(
                    len(gd) / dow_occurrences[int(dow)])

    return WeeklyAvailability(windows=windows, p_work=p_work)


def evaluate_weekly(
    model: WeeklyAvailability, work: pd.DataFrame, presence: pd.DataFrame
) -> Dict[str, float]:
    """Coverage and window utilisation — see module docstring for definitions."""
    key = pd.DataFrame(
        [(r, d, w[0], w[1])
         for r, dws in model.windows.items() for d, w in dws.items()],
        columns=["org:resource", "dow", "start", "end"],
    ).set_index(["org:resource", "dow"])

    ev = work[["org:resource", "hour"]].copy()
    ev["dow"] = work["time:timestamp"].dt.dayofweek
    j = ev.join(key, on=["org:resource", "dow"]).dropna(subset=["start"])
    coverage = float(((j.hour >= j.start) & (j.hour <= j.end)).mean())

    days = presence.groupby(["org:resource", "dow"]).size().rename("n_days")
    md = key.join(days, how="inner")
    open_hours = float(((md.end - md.start) * md.n_days).sum())
    busy_hours = float(
        presence.set_index(["org:resource", "dow"]).loc[md.index, "span"].sum()
    )

    return {
        "coverage": coverage,
        "window_utilisation": busy_hours / open_hours if open_hours else 0.0,
        "resources_modelled": len(model.windows),
    }


# ──────────────────────────────────────────────────────────────────────────
# Advanced: the year plan
# ──────────────────────────────────────────────────────────────────────────

def stable_window(
    presence: pd.DataFrame, tolerance: float = 0.6
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """The stretch of the log where headcount is steady.

    The log ramps at both ends — mean weekday headcount runs 28 (Jan 2016) → ~38
    (mid-2016) → 21 (Jan 2017) → 15 (Feb 2017), as staff are on/offboarded and
    the export truncates. Absence statistics are only meaningful inside the
    stable core, so holiday and vacation discovery is restricted to it.
    """
    head = _weekday_headcount(presence)
    monthly = head.groupby(head.index.to_period("M")).mean()
    peak = monthly.max()
    good = monthly[monthly >= tolerance * peak].index
    return good.min().start_time, good.max().end_time


def discover_holidays(
    presence: pd.DataFrame,
    threshold: float = HOLIDAY_MAX_HEADCOUNT,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DatetimeIndex:
    """Weekdays on which the workforce collectively disappears.

    Compared against a *local* baseline (rolling median of the surrounding
    weeks) rather than a global one, so the headcount ramp at the log's edges
    does not masquerade as a holiday.

    On BPIC-17 this recovers the Dutch bank holidays without being given them:
    Easter Monday, King's Day, Ascension Day, Whit Monday, Boxing Day. (Good
    Friday correctly does *not* appear — it is not an official public holiday
    for most Dutch employers.)
    """
    head = _weekday_headcount(presence)
    if window is not None:
        head = head[(head.index >= window[0]) & (head.index <= window[1])]

    local = head.rolling(21, center=True, min_periods=5).median()
    return pd.DatetimeIndex(head.index[head < threshold * local])


@dataclass
class VacationProfile:
    """How a resource takes extended leave, fitted per resource."""

    n_vacations_per_year: float
    mean_length_days: float
    lengths: list[int] = field(default_factory=list)


def _chance_threshold(n_workdays: int, q: float, alpha: float = 0.5) -> int:
    """Absence-run length beyond which a gap is not plausibly chance.

    A resource who works a given weekday with probability `q` produces absence
    runs on its own. Over `n_workdays`, the expected number of runs of length
    >= L is about ``n * q * (1-q)**L``. We take the smallest L for which that
    falls below `alpha` — i.e. we would expect fewer than half a run of that
    length across the whole period if the resource were simply working at its
    usual sporadic rate.
    """
    if q <= 0.0 or q >= 1.0:
        return MIN_VACATION_DAYS
    expected_runs = max(n_workdays * q, 1e-9)
    L = np.log(alpha / expected_runs) / np.log(1.0 - q)
    return max(MIN_VACATION_DAYS, int(np.ceil(L)))


def discover_vacations(
    presence: pd.DataFrame,
    holidays: pd.DatetimeIndex,
    weekly: WeeklyAvailability,
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Dict[str, VacationProfile]:
    """Absence runs long enough that chance does not explain them.

    Design decision — why the threshold is per-resource, not a flat 5 days
    -------------------------------------------------------------------
    BPIC-17 resources are *not* full-time staff. The median modelled resource
    works only ~27% of weekdays; the pool is part-time and rotating. At that
    rate, runs of five or more consecutive absent working days occur constantly
    by chance — about 15 per year at p_work = 0.25. A flat "5 absent days = a
    holiday" rule therefore measures the sporadic baseline, not leave, and
    inflates vacation until it swallows the calendar.

    So each resource gets its own threshold, derived from its own working rate
    (see `_chance_threshold`). A gap counts as leave only if it is longer than
    that resource's normal rhythm would produce.

    The consequence is deliberate: for low-rate resources, leave is simply *not
    identifiable* — you cannot spot a two-week holiday in someone who only shows
    up one day in ten. Those resources get no vacation profile, and their
    absence stays modelled by `p_work`, which is where it belongs. Only the
    near-full-time resources have a vacation model, because only for them does
    the signal exist.

    The data confirms the cut. Grouping detected blocks by the resource's
    working rate:

        working rate            blocks   median   max
        near-full-time (>60%)      22     10.5d    21d
        part-time (30-60%)         50     15.5d   105d
        sporadic (<30%)            33     31.0d    74d

    Above the cut the blocks look exactly like real annual leave — a fortnight
    typical, four weeks at most, and not one implausible block over 30 days.
    Below it the "vacations" run to five months, which is not a holiday. So the
    plausibility ceiling emerges from the data rather than being imposed, and we
    fit the model only where it holds.

    Runs are counted only between a resource's first and last observed activity,
    so onboarding and offboarding are not mistaken for very long holidays.
    """
    p = presence.copy()
    p["date"] = pd.to_datetime(p["date"])
    lo, hi = window or (p["date"].min(), p["date"].max())

    workdays = pd.DatetimeIndex(
        d for d in pd.date_range(lo, hi, freq="D")
        if d.dayofweek < 5 and d not in set(holidays)
    )

    out: Dict[str, VacationProfile] = {}
    for res, g in p.groupby("org:resource"):
        if res not in weekly.p_work:
            continue

        active = set(g["date"])
        tenure = [d for d in workdays if g["date"].min() <= d <= g["date"].max()]
        if len(tenure) < 40:
            continue

        q = float(np.mean([weekly.p_work[res].get(d, 0.0) for d in range(5)]))
        if q < VACATION_MIN_WORK_RATE:
            continue    # leave is not identifiable for this resource

        min_days = _chance_threshold(len(tenure), q)

        lengths, run = [], 0
        for d in tenure:
            if d in active:
                if run >= min_days:
                    lengths.append(run)
                run = 0
            else:
                run += 1
        if run >= min_days:
            lengths.append(run)

        covered = max((tenure[-1] - tenure[0]).days / 365.25, 1e-9)
        out[res] = VacationProfile(
            n_vacations_per_year=len(lengths) / covered,
            mean_length_days=float(np.mean(lengths)) if lengths else 0.0,
            lengths=lengths,
        )
    return out


@dataclass
class YearlyAvailability:
    """Weekly shift profile + holiday calendar + sampled vacation blocks."""

    weekly: WeeklyAvailability
    holidays: set          # of datetime.date
    vacations: Dict[str, set]   # resource -> set of dates on leave
    system: set = field(default_factory=set)   # automated accounts (no office hours)

    def is_available(self, resource: str, when) -> bool:
        """Is *resource* on duty at *when*?

        Accepts either a pandas Timestamp or a stdlib datetime — the simulation
        should not have to import pandas to ask a calendar question.
        """
        d = when.date()
        if d in self.holidays:
            return False
        if d in self.vacations.get(resource, ()):
            return False
        hour = when.hour + when.minute / 60 + when.second / 3600
        return self.weekly.is_open(resource, d.weekday(), hour)

    def to_json(self, path: str | Path) -> Path:
        """Serialise the fitted model. Small: parameters, not per-day rows."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "windows": {
                r: {str(d): list(w) for d, w in dws.items()}
                for r, dws in self.weekly.windows.items()
            },
            "p_work": {
                r: {str(d): v for d, v in dws.items()}
                for r, dws in self.weekly.p_work.items()
            },
            "holidays": sorted(d.isoformat() for d in self.holidays),
            "vacations": {
                r: sorted(d.isoformat() for d in ds)
                for r, ds in self.vacations.items() if ds
            },
            "system": sorted(self.system),
        }
        path.write_text(json.dumps(payload, indent=1))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "YearlyAvailability":
        import datetime as _dt

        d = json.loads(Path(path).read_text())
        weekly = WeeklyAvailability(
            windows={r: {int(k): tuple(v) for k, v in dws.items()}
                     for r, dws in d["windows"].items()},
            p_work={r: {int(k): v for k, v in dws.items()}
                    for r, dws in d["p_work"].items()},
        )
        return cls(
            weekly=weekly,
            holidays={_dt.date.fromisoformat(s) for s in d["holidays"]},
            vacations={r: {_dt.date.fromisoformat(s) for s in ds}
                       for r, ds in d["vacations"].items()},
            system=set(d.get("system", ())),
        )


def build_year_plan(
    weekly: WeeklyAvailability,
    holidays: pd.DatetimeIndex,
    profiles: Dict[str, VacationProfile],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    holiday_dates: Optional[Iterable] = None,
    seed: int = 42,
    system: Optional[Iterable[str]] = None,
) -> YearlyAvailability:
    """Project a forward-looking year: holidays are placed, vacations sampled.

    Public holidays are a property of the calendar, so they are carried over
    directly (via `holiday_dates`, which lets you supply the target year's
    dates). Vacations are a property of the *person*, so they are re-sampled per
    resource from that resource's fitted profile — the simulation should not
    replay 2016's exact leave, it should generate statistically similar leave.
    """
    rng = np.random.default_rng(seed)

    hol = set(pd.DatetimeIndex(holiday_dates).date) if holiday_dates is not None \
        else {d.date() for d in holidays}

    workdays = [d for d in pd.date_range(year_start, year_end, freq="D")
                if d.dayofweek < 5 and d.date() not in hol]

    vacations: Dict[str, set] = {}
    for res in weekly.resources():
        prof = profiles.get(res)
        if not prof or not prof.lengths:
            vacations[res] = set()
            continue

        n = rng.poisson(prof.n_vacations_per_year)
        booked: set = set()
        for _ in range(int(n)):
            length = int(rng.choice(prof.lengths))
            if length >= len(workdays):
                continue
            start = int(rng.integers(0, len(workdays) - length))
            booked.update(d.date() for d in workdays[start:start + length])
        vacations[res] = booked

    return YearlyAvailability(weekly=weekly, holidays=hol, vacations=vacations,
                              system=set(system or ()))


# ──────────────────────────────────────────────────────────────────────────

def _weekday_headcount(presence: pd.DataFrame) -> pd.Series:
    """Distinct resources active per weekday date."""
    p = presence.copy()
    p["date"] = pd.to_datetime(p["date"])
    days = pd.date_range(p["date"].min(), p["date"].max(), freq="D")
    weekdays = [d for d in days if d.dayofweek < 5]
    return (
        p.groupby("date")["org:resource"].nunique()
        .reindex(weekdays).fillna(0)
    )

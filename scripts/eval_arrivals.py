"""
eval_arrivals.py
=================
Empirical validation of the two case-arrival components against the real
BPIC-17 arrival process, for Simulation Section 1.2:

  - components/arrival.py      (Basic:    fixed LogNormal inter-arrivals)
  - components/arrival_mdn.py  (Advanced: time-dependent MDN)

The point of the MDN is capturing time-of-day / weekday structure a
static LogNormal cannot represent (nights ~0.6 arrivals/h vs ~7.6/h in
the 12-18h core; Monday ~3x Sunday). This script measures whether it
actually does, against the real log, rather than taking the README's
claim on faith.

Compares, for a matched horizon:
  - arrivals per hour-of-day (24-bin normalized histogram) -- MAE vs real
  - arrivals per weekday (7-bin normalized histogram)       -- MAE vs real
  - inter-arrival time distribution                          -- KS statistic vs real

Requires the raw BPIC-17 log (gitignored, not shipped in this repo).
See analysis/loader.py::resolve_log_path for where to place it or set
$BPIC17_LOG. This script refuses to fabricate a comparison against
simulated data standing in for ground truth -- if the real log isn't
found, it prints where to put it and exits without writing output.

Usage:
    cd <repo-root>
    PYTHONPATH=. .venv/bin/python scripts/eval_arrivals.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType
from simulation.components.arrival import ArrivalComponent
from simulation.components.arrival_mdn import MDNArrivalComponent
from analysis.loader import load_events, resolve_log_path

SEED = 42
# BPIC-17 starts 2016-01-01, a Friday -- must match main.py's START_DATETIME
# so weekday/hour-of-day alignment is correct for the MDN.
START_DATETIME = datetime(2016, 1, 1)
HORIZON_DAYS = 90
HORIZON_SECONDS = HORIZON_DAYS * 24 * 3600

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "output" / "arrival_model_eval.md"


class _ArrivalRecorder:
    """Records CASE_ARRIVAL sim-timestamps; registered alongside an arrival
    component so it observes the same events without altering them."""

    HANDLES = {EventType.CASE_ARRIVAL: None}

    def __init__(self):
        self.timestamps: list[float] = []

    def on_arrival(self, engine, event) -> None:
        self.timestamps.append(event.timestamp)


_ArrivalRecorder.HANDLES = {EventType.CASE_ARRIVAL: _ArrivalRecorder.on_arrival}


def simulate_arrivals(component_factory) -> list[float]:
    """Run *only* an arrival component (no process/resource) for HORIZON_DAYS
    and return sim-time (seconds) arrival timestamps."""
    engine = SimulationEngine(
        sim_duration=HORIZON_SECONDS, start_datetime=START_DATETIME, verbose=False,
    )
    arrivals = component_factory()
    recorder = _ArrivalRecorder()
    engine.register(arrivals)
    engine.register(recorder)
    arrivals.bootstrap(engine)
    engine.run()
    return recorder.timestamps


def real_arrival_datetimes() -> list[datetime]:
    """First observed event per case in the real log = that case's arrival."""
    df = load_events()
    first = df.groupby("case:concept:name")["time:timestamp"].min()
    return list(first.dt.tz_localize(None))


def hour_of_day_histogram(dts: list[datetime]) -> np.ndarray:
    hours = np.array([dt.hour for dt in dts])
    counts, _ = np.histogram(hours, bins=np.arange(25))
    return counts / counts.sum()


def weekday_histogram(dts: list[datetime]) -> np.ndarray:
    dows = np.array([dt.weekday() for dt in dts])
    counts, _ = np.histogram(dows, bins=np.arange(8))
    return counts / counts.sum()


def inter_arrival_seconds(dts: list[datetime]) -> np.ndarray:
    sec = np.array(sorted(dt.timestamp() for dt in dts))
    return np.diff(sec)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main():
    try:
        resolve_log_path()
    except FileNotFoundError as e:
        print(
            "Cannot validate arrival models: the real BPIC-17 log is not "
            "available in this environment.\n\n"
            f"{e}\n\n"
            "This script deliberately does not fall back to comparing against "
            "simulated data -- that would validate the model against itself. "
            "Place the raw log at one of the paths above, or set $BPIC17_LOG, "
            "then re-run:\n"
            "    BPIC17_LOG=/path/to/BPI_Challenge_2017.csv "
            "PYTHONPATH=. .venv/bin/python scripts/eval_arrivals.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading real BPIC-17 arrivals ...")
    real_dts = real_arrival_datetimes()
    horizon_end = START_DATETIME + timedelta(days=HORIZON_DAYS)
    real_dts = [dt for dt in real_dts if START_DATETIME <= dt < horizon_end]
    print(f"  {len(real_dts)} real case arrivals in the first {HORIZON_DAYS} days")

    print("Simulating parametric (LogNormal) arrivals ...")
    param_ts = simulate_arrivals(lambda: ArrivalComponent(seed=SEED))
    param_dts = [START_DATETIME + timedelta(seconds=t) for t in param_ts]

    print("Simulating MDN arrivals ...")
    mdn_ts = simulate_arrivals(
        lambda: MDNArrivalComponent(seed=SEED, start_datetime=START_DATETIME)
    )
    mdn_dts = [START_DATETIME + timedelta(seconds=t) for t in mdn_ts]

    real_hod, param_hod, mdn_hod = (
        hour_of_day_histogram(real_dts),
        hour_of_day_histogram(param_dts),
        hour_of_day_histogram(mdn_dts),
    )
    real_dow, param_dow, mdn_dow = (
        weekday_histogram(real_dts),
        weekday_histogram(param_dts),
        weekday_histogram(mdn_dts),
    )
    real_iat = inter_arrival_seconds(real_dts)
    param_iat = inter_arrival_seconds(param_dts)
    mdn_iat = inter_arrival_seconds(mdn_dts)

    param_hod_mae = mae(param_hod, real_hod)
    mdn_hod_mae = mae(mdn_hod, real_hod)
    param_dow_mae = mae(param_dow, real_dow)
    mdn_dow_mae = mae(mdn_dow, real_dow)
    param_ks = ks_2samp(param_iat, real_iat)
    mdn_ks = ks_2samp(mdn_iat, real_iat)

    lines = []
    lines.append("# Arrival model evaluation — Simulation 1.2\n")
    lines.append(
        f"Real vs simulated case arrivals, first {HORIZON_DAYS} days anchored "
        f"at {START_DATETIME.date()}, seed={SEED}.\n"
        f"Real arrivals: {len(real_dts)}. Parametric: {len(param_dts)}. MDN: {len(mdn_dts)}.\n"
    )
    lines.append("| Metric | Parametric (LogNormal, Basic) | MDN (Advanced) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Hour-of-day MAE (24 bins, lower=better) | {param_hod_mae:.4f} | {mdn_hod_mae:.4f} |")
    lines.append(f"| Weekday MAE (7 bins, lower=better) | {param_dow_mae:.4f} | {mdn_dow_mae:.4f} |")
    lines.append(f"| Inter-arrival KS statistic (lower=better) | {param_ks.statistic:.4f} | {mdn_ks.statistic:.4f} |")
    lines.append(f"| Inter-arrival KS p-value | {param_ks.pvalue:.2e} | {mdn_ks.pvalue:.2e} |")

    print("\n" + "\n".join(lines) + "\n")

    mdn_wins_hod = mdn_hod_mae < param_hod_mae
    mdn_wins_dow = mdn_dow_mae < param_dow_mae
    mdn_wins_ks = mdn_ks.statistic < param_ks.statistic
    recommend_mdn = mdn_wins_hod and mdn_wins_dow

    conclusion = f"""
## Conclusion

MDN {"beats" if mdn_wins_hod else "does not beat"} the parametric model on
hour-of-day shape, {"beats" if mdn_wins_dow else "does not beat"} it on
weekday shape, and {"beats" if mdn_wins_ks else "does not beat"} it on the
inter-arrival KS distance.

Recommendation: {"flip `USE_MDN_ARRIVALS = True` in `simulation/main.py` as "
"the team default" if recommend_mdn else "keep `USE_MDN_ARRIVALS = False`; "
"the MDN does not demonstrably improve on the parametric model for this "
"horizon/seed"} -- coordinate with the team before changing the default, since
it changes every teammate's simulated event logs.
"""
    print(conclusion)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + conclusion)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

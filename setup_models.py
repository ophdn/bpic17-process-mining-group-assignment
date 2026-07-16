"""
setup_models.py — one-command model bootstrap
==============================================
Regenerates every trained artifact the simulation needs but does NOT ship in
git (they are large binaries / derived from the raw log, so `.gitignore`
excludes them). Run this once after cloning and after placing the raw
BPIC-17 event log in the repo:

    pip install -r requirements.txt
    # put BPIChallenge2017.xes (or .xes.gz) in the repo root (or data/)
    python setup_models.py

    # then the ML-dependent modes work:
    python -m simulation.main --mode ml_probabilistic
    python -m simulation.main --branching-mode rules

Artifacts produced (from the raw log):

  simulation/models/processing_time_model.joblib  (Section 1.3 ml_model /
      ml_probabilistic, and the k-Batching duration cost) — trained with
      --probabilistic so BOTH the point model and the 19 quantile models
      are written in one pass.
  simulation/models/decision_rules.joblib          (Section 1.5
      --branching-mode rules).

Everything else the simulation loads (the BPMN, the fitted availability
model, the visit-conditioned branching probabilities) is committed, so the
DEFAULT run needs neither this script nor the raw log:

    python -m simulation.main            # distribution mode, runs from a clone

By default an artifact that already exists is skipped; pass --force to
retrain. Pass --log to point at a log in a non-standard location.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Candidate locations for the raw log, in priority order. The training
# scripts accept .xes, .xes.gz, or .csv.
LOG_CANDIDATES = [
    "BPIChallenge2017.xes",
    "BPIChallenge2017.xes.gz",
    "data/BPIChallenge2017.xes",
    "data/BPIChallenge2017.xes.gz",
    "BPIChallenge2017.csv",
    "data/BPIChallenge2017.csv",
]

# (label, output artifact, training script, extra CLI args)
ARTIFACTS = [
    (
        "processing-time model (point + 19 quantile models)",
        "simulation/models/processing_time_model.joblib",
        "train_processing_time_model.py",
        ["--probabilistic"],
    ),
    (
        "decision-point rules classifier",
        "simulation/models/decision_rules.joblib",
        "train_decision_rules.py",
        [],
    ),
]


def find_log(explicit: str | None) -> Path:
    """Return the raw log path, or exit with an actionable message."""
    if explicit:
        p = (REPO_ROOT / explicit) if not Path(explicit).is_absolute() else Path(explicit)
        if p.is_file():
            return p
        sys.exit(f"[setup] --log '{explicit}' not found at {p}")

    for cand in LOG_CANDIDATES:
        p = REPO_ROOT / cand
        if p.is_file():
            return p

    sys.exit(
        "[setup] Raw BPIC-17 log not found. It is gitignored and must be added\n"
        "        manually. Download 'BPI Challenge 2017' and place it as one of:\n"
        + "".join(f"          {c}\n" for c in LOG_CANDIDATES)
        + "        (or pass --log <path>), then re-run `python setup_models.py`."
    )


def run_trainer(script: str, log_path: Path, extra: list[str]) -> None:
    """Invoke a training script as a subprocess from the repo root."""
    cmd = [sys.executable, script, "--log", str(log_path), *extra]
    print(f"[setup] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate gitignored trained artifacts from the raw log."
    )
    parser.add_argument(
        "--log", default=None,
        help="Path to the raw BPIC-17 log (.xes/.xes.gz/.csv). "
             "Default: auto-detect in the repo root or data/.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if the artifact already exists.",
    )
    args = parser.parse_args()

    # If everything already exists and --force wasn't passed, we don't even
    # need the log — report and exit.
    needed = [
        a for a in ARTIFACTS
        if args.force or not (REPO_ROOT / a[1]).is_file()
    ]
    if not needed:
        print("[setup] All artifacts already present — nothing to do "
              "(pass --force to retrain):")
        for label, out, _script, _extra in ARTIFACTS:
            print(f"          {out}")
        return

    log_path = find_log(args.log)
    print(f"[setup] Using log: {log_path}")

    for label, out, script, extra in ARTIFACTS:
        out_path = REPO_ROOT / out
        if out_path.is_file() and not args.force:
            print(f"[setup] SKIP  {label} — already present at {out}")
            continue
        print(f"[setup] TRAIN {label} -> {out}")
        run_trainer(script, log_path, extra)

    print("\n[setup] Done. All required artifacts are in place:")
    for label, out, _script, _extra in ARTIFACTS:
        status = "ok" if (REPO_ROOT / out).is_file() else "MISSING"
        print(f"          [{status}] {out}")
    print("\n[setup] You can now run e.g.:")
    print("          python -m simulation.main --mode ml_probabilistic")
    print("          python -m simulation.main --branching-mode rules")


if __name__ == "__main__":
    main()

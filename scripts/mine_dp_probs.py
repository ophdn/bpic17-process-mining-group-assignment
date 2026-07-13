"""
mine_dp_probs.py
================
A1 termination fix, stage 2: mine branching probabilities AT THE PETRI
NET'S DECISION POINTS (via replay), conditioned on how often the case has
already visited that decision point.

Why trace bigrams are not enough (measured, see
output/validation/branching_probs_vs_rules/):

- P(next | current activity) mixes concurrency interleavings into the
  branching estimate: the real next *event* after A_Validating (1st visit)
  is O_Returned 93.6% of the time — but O_Returned belongs to a concurrent
  branch that is not even enabled at that decision point, so the simulation
  renormalises over the wrong candidates.
- Loop exits carried only RESIDUAL_WEIGHT: after O_Cancelled the frontier
  offers the exit with ~3% effective probability per round, giving ~16
  O_Cancelled per case (real: 0.66).

This script replays the real log on the exact same Petri net /
tau-closure logic the simulation uses (PetriNetProcessComponent's
_visible_frontier / _fire_activity, like train_decision_rules.py) and
records, for every decision point (= sorted tuple of enabled visible
labels) and every per-case visit number of that decision point, which
label the real case actually took.

Output: simulation/models/dp_branching_probs.json
    {"buckets": ["1", ..., "5+"],
     "dp_probs": {"<label> | <label> | ...": {"1": {label: p, ...},
                                              ...,
                                              "all": {label: p, ...}}}}
"all" is the unconditioned distribution per decision point — the fallback
when a visit bucket was too sparse (< --min-samples).

Usage (from the repo root; replay takes ~10 minutes):
    python scripts/mine_dp_probs.py --log data/BPIChallenge2017.xes.gz
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pm4py.objects.petri_net.obj import Marking  # noqa: E402

from extract_log_info import load_log, filter_to_complete  # noqa: E402
from simulation.components.petri_process import (  # noqa: E402
    DP_VISIT_BUCKET_MAX,
    PetriNetProcessComponent,
)

DEFAULT_BPMN = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
OUTPUT_PATH = REPO_ROOT / "simulation" / "models" / "dp_branching_probs.json"


def bucket_of(k: int) -> str:
    return str(k) if k < DP_VISIT_BUCKET_MAX else f"{DP_VISIT_BUCKET_MAX}+"


def mine(df_complete, comp: PetriNetProcessComponent):
    """counts[dp_key][bucket][label] and counts[dp_key]["all"][label].

    A decision is recorded whenever the case has more than one option:
    several enabled visible labels, and/or the choice to STOP (final
    marking tau-reachable → pseudo-label "__END__"). Recording where real
    traces end is what lets the simulation escape structural loops such as
    the [O_Cancelled] singleton frontier, where "continue" is the only
    visible label but real cases overwhelmingly stop.
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    replay_key = "__mine__"
    n_fitting = n_nonfitting = 0

    def record(dp_visits, frontier, chosen: str) -> None:
        dp_key = " | ".join(sorted(frontier.keys()))
        dp_visits[dp_key] += 1
        counts[dp_key][bucket_of(dp_visits[dp_key])][chosen] += 1
        counts[dp_key]["all"][chosen] += 1

    for case_id, case_df in df_complete.groupby("case_id", sort=False):
        case_df = case_df.sort_values("timestamp")
        comp._markings[replay_key] = Marking(comp.im)
        dp_visits: dict = defaultdict(int)
        fits = True

        for act in case_df["activity"]:
            marking = comp._markings[replay_key]
            frontier = comp._visible_frontier(marking)
            if act not in frontier:
                fits = False
                break
            can_end = comp._final_reachable_by_tau(marking)
            if len(frontier) + (1 if can_end else 0) > 1:
                record(dp_visits, frontier, act)
            comp._markings[replay_key] = frontier[act]
            comp._fire_activity(replay_key, act)

        if fits:
            # Trace exhausted: the real case chose to stop here.
            marking = comp._markings[replay_key]
            frontier = comp._visible_frontier(marking)
            if frontier and comp._final_reachable_by_tau(marking):
                record(dp_visits, frontier, "__END__")

        comp._markings.pop(replay_key, None)
        n_fitting += fits
        n_nonfitting += not fits

    return counts, n_fitting, n_nonfitting


def to_probs(counts: dict, min_samples: int) -> dict:
    dp_probs = {}
    for dp_key, buckets in counts.items():
        entry = {}
        for bucket, label_counts in buckets.items():
            total = sum(label_counts.values())
            if bucket != "all" and total < min_samples:
                continue
            entry[bucket] = {
                label: round(cnt / total, 4)
                for label, cnt in sorted(label_counts.items(), key=lambda kv: -kv[1])
            }
        if entry:
            dp_probs[dp_key] = entry
    return dp_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--bpmn", type=Path, default=DEFAULT_BPMN)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[load] Reading log: {args.log} ...")
    df = load_log(args.log)
    df_complete = filter_to_complete(df)
    print(f"[load] {df_complete['case_id'].nunique():,} cases.")

    comp = PetriNetProcessComponent(bpmn_path=str(args.bpmn), seed=42)
    print("[mine] Replaying real log on the Petri net ...")
    t0 = time.perf_counter()
    counts, n_fit, n_nonfit = mine(df_complete, comp)
    print(f"[mine] Done in {time.perf_counter() - t0:.0f}s — "
          f"{n_fit:,}/{n_fit + n_nonfit:,} cases fit "
          f"({100 * n_fit / (n_fit + n_nonfit):.1f}%).")

    dp_probs = to_probs(counts, args.min_samples)
    out = {
        "bpmn": args.bpmn.name,
        "visit_bucket_max": DP_VISIT_BUCKET_MAX,
        "min_samples": args.min_samples,
        "replay_fit_pct": round(100 * n_fit / (n_fit + n_nonfit), 2),
        "dp_probs": dp_probs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {len(dp_probs)} decision points -> {args.output}")

    for dp_key, entry in sorted(dp_probs.items(),
                                key=lambda kv: -sum(1 for b in kv[1] if b != "all"))[:5]:
        print(f"\n  {dp_key}")
        for bucket in [b for b in entry if b != "all"][:4]:
            top = list(entry[bucket].items())[:3]
            print(f"    v{bucket}: {dict(top)}")


if __name__ == "__main__":
    main()

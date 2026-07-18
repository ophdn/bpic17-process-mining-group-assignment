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

Deviation handling (repair-based replay): only ~57.7% of real cases
replay end-to-end on this net without ever stepping outside its legal
frontier. Earlier versions of this script abandoned the rest of a case's
trace on its first deviation, which starves every decision point
reachable only *after* that point -- worst for higher visit buckets of
loop decision points, and total for the __END__ signal, which was then
only ever recorded for cases that never deviated at all (a measured
selection bias, see docs/report_notes_1.4_1.5.md Sec. 5). This version
repairs instead of abandoning: when the next real activity isn't
reachable from the current marking by any tau combination, the matching
transition is forced to fire anyway, topping up its under-marked input
places with the missing tokens it needs -- the standard token-based-
replay "missing token" convention (pm4py's own conformance checker uses
the same idea; docs/report_notes_1.4_1.5.md's D2 already relies on
token-replay being more lenient than prefix-replay: 68.8% vs. 57.7%). The
deviating step itself is never recorded as a decision-point choice -- it
was never a legal option there, so counting it would corrupt that
decision point's distribution -- but every decision point before *and
after* it now gets this case's data, and the repaired trace still
reaches a real end-of-trace marking, so __END__ can be recorded for
(almost) every case instead of only the perfectly-fitting ones.

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

from pm4py.objects.petri_net import semantics  # noqa: E402
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


def _label_transition_map(net) -> dict:
    """label -> transitions sharing it, sorted by name for a deterministic
    repair tie-break (mirrors the sort key _fire_activity/_visible_frontier
    already use elsewhere in the codebase for the same reason).

    Unlike PetriNetProcessComponent._fire_activity, which only ever looks
    among transitions already *enabled* at the current marking, repair must
    find a candidate by label regardless of whether it is enabled right
    now -- that's the whole point of repairing it.
    """
    mapping: dict = defaultdict(list)
    for t in net.transitions:
        if t.label is not None:
            mapping[t.label].append(t)
    for label in mapping:
        mapping[label].sort(key=lambda t: t.name)
    return mapping


def _repair_and_fire(net, marking, label_map: dict, label: str):
    """Force *label* to fire from *marking* even though it isn't reachable
    by any tau combination, by inserting the missing tokens its cheapest
    matching transition needs (token-based-replay's "missing token" repair,
    the same convention pm4py's own conformance checker uses) and then
    firing it.

    Returns (new_marking, n_missing_tokens), or (None, 0) if the net has no
    transition labelled *label* at all. This does happen: the Signavio net
    (D2) has no transition for a handful of rare real activities (e.g.
    "W_Assess potential fraud", "W_Call after offers") -- a genuine
    vocabulary gap in the model itself, not a bug in this repair, and
    previously invisible because any deviation aborted the whole trace
    before this distinction could be made. The caller counts these
    separately (n_unrepairable_events) instead of silently lumping them in
    with ordinary branching deviations.

    The result is clamped to at most one token per place before returning
    (see the comment above the clamp below) -- without it this measured
    35s on a single 30-event, 6-repair case (should be <0.1s), because the
    surplus tokens a repair can leave behind blow up the tau-closure search
    _visible_frontier/_final_reachable_by_tau do for every later decision
    point in the same case.
    """
    candidates = label_map.get(label)
    if not candidates:
        return None, 0
    best_marking, best_missing = None, None
    for t in candidates:
        repaired = Marking(marking)
        missing = 0
        for arc in t.in_arcs:
            need = arc.weight
            have = repaired.get(arc.source, 0)
            if have < need:
                repaired[arc.source] = need
                missing += need - have
        if best_missing is None or missing < best_missing:
            best_marking, best_missing = semantics.execute(t, net, repaired), missing

    # This net models a single case's control state one token at a time --
    # every place is meant to hold at most one (a live simulation case
    # never has two tokens in the same place; _final_reachable_by_tau's own
    # docstring relies on "the net's reachable-marking set is small and
    # finite"). A repair can violate that by inserting a token into a place
    # that already holds one left over from earlier in the trace (e.g. an
    # AND-split branch whose join was skipped by a previous repair) --
    # invisible to correctness (place still just means "this branch is
    # pending"), but each surplus token roughly multiplies the number of
    # tau-combinations _visible_frontier has to enumerate for every
    # subsequent decision in this case. Clamping back to the net's intended
    # 1-safe invariant after every repair keeps that search in the small
    # space it was designed for.
    for place, count in list(best_marking.items()):
        if count > 1:
            best_marking[place] = 1
    return best_marking, best_missing


def mine(df_complete, comp: PetriNetProcessComponent):
    """counts[dp_key][bucket][label] and counts[dp_key]["all"][label].

    A decision is recorded whenever the case has more than one option:
    several enabled visible labels, and/or the choice to STOP (final
    marking tau-reachable → pseudo-label "__END__"). Recording where real
    traces end is what lets the simulation escape structural loops such as
    the [O_Cancelled] singleton frontier, where "continue" is the only
    visible label but real cases overwhelmingly stop.

    Deviations are repaired, not abandoned (see module docstring): a case
    that steps outside the net's legal frontier still gets every decision
    point before *and after* that point recorded, and still reaches a real
    end-of-trace marking, so __END__ can be observed for it too.
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    replay_key = "__mine__"
    label_map = _label_transition_map(comp.net)
    n_perfect = n_repaired = 0
    n_repair_events = n_missing_tokens = n_unrepairable_events = 0

    def record(dp_visits, frontier, chosen: str) -> None:
        dp_key = " | ".join(sorted(frontier.keys()))
        dp_visits[dp_key] += 1
        counts[dp_key][bucket_of(dp_visits[dp_key])][chosen] += 1
        counts[dp_key]["all"][chosen] += 1

    for case_id, case_df in df_complete.groupby("case_id", sort=False):
        case_df = case_df.sort_values("timestamp")
        comp._markings[replay_key] = Marking(comp.im)
        dp_visits: dict = defaultdict(int)
        case_needed_repair = False

        for act in case_df["activity"]:
            marking = comp._markings[replay_key]
            frontier = comp._visible_frontier(marking)
            can_end = comp._final_reachable_by_tau(marking)

            if act in frontier:
                if len(frontier) + (1 if can_end else 0) > 1:
                    record(dp_visits, frontier, act)
                comp._markings[replay_key] = frontier[act]
                comp._fire_activity(replay_key, act)
                continue

            # Deviation: `act` is not reachable from `marking` by any tau
            # combination. Not recorded as a decision (it was never a legal
            # option here), but replay continues via repair instead of
            # abandoning the rest of the trace.
            case_needed_repair = True
            new_marking, missing = _repair_and_fire(comp.net, marking, label_map, act)
            if new_marking is None:
                n_unrepairable_events += 1
                continue  # no transition for this label at all; marking unchanged
            comp._markings[replay_key] = new_marking
            n_repair_events += 1
            n_missing_tokens += missing

        # Trace exhausted (possibly via repair): check whether the real
        # case's actual stopping point is now a legal end for this marking.
        marking = comp._markings[replay_key]
        frontier = comp._visible_frontier(marking)
        if frontier and comp._final_reachable_by_tau(marking):
            record(dp_visits, frontier, "__END__")

        comp._markings.pop(replay_key, None)
        if case_needed_repair:
            n_repaired += 1
        else:
            n_perfect += 1

    stats = {
        "n_perfect": n_perfect,
        "n_repaired": n_repaired,
        "n_repair_events": n_repair_events,
        "n_missing_tokens": n_missing_tokens,
        "n_unrepairable_events": n_unrepairable_events,
    }
    return counts, stats


def global_end_rate(counts: dict) -> float:
    """
    Pooled P(__END__) across every decision point's per-visit buckets
    (excluding the "all" aggregates, which would double-count each
    decision).  __END__ is the one label with a meaning shared across every
    decision point (unlike the other labels, which are decision-point-
    specific activities) -- so it's the only one for which pooling data
    from *other* decision points as a prior is principled.  See to_probs's
    shrinkage: some narrow, fully-closed decision points (every legal
    label is itself a loop activity) have essentially zero observed
    __END__ at their sparser buckets, not because ending is truly
    impossible there, but because so few real cases ever revisit that
    exact decision point enough times to observe it -- a raw per-bucket
    frequency of 0 there is a data-sparsity artifact, not evidence that
    p(__END__) truly is 0 (docs/ROADMAP.md, A1-Update Teil 8).
    """
    end_count = total = 0
    for buckets in counts.values():
        for bucket, label_counts in buckets.items():
            if bucket == "all":
                continue
            end_count += label_counts.get("__END__", 0)
            total += sum(label_counts.values())
    return end_count / total if total else 0.0


def to_probs(counts: dict, min_samples: int, end_shrinkage_alpha: float = 0.0) -> dict:
    """
    end_shrinkage_alpha > 0 applies Dirichlet-style shrinkage to the
    END-vs-continue split only, pulling each bucket's raw P(__END__)
    toward the global pooled rate (global_end_rate) in proportion to how
    little data backs that specific bucket:

        p_end = (end_count + alpha * global_end_rate) / (total + alpha)

    The relative proportions *among* the continue-choices are left exactly
    as observed -- those labels are decision-point-specific, so there is no
    principled global prior to pull them toward; only the END/continue
    split benefits from pooling across decision points. alpha=0 (default)
    reproduces the original unsmoothed behaviour exactly.
    """
    prior_end = global_end_rate(counts) if end_shrinkage_alpha > 0 else 0.0
    dp_probs = {}
    for dp_key, buckets in counts.items():
        entry = {}
        for bucket, label_counts in buckets.items():
            total = sum(label_counts.values())
            if bucket != "all" and total < min_samples:
                continue
            if end_shrinkage_alpha > 0 and total > 0:
                end_count = label_counts.get("__END__", 0)
                p_end = (end_count + end_shrinkage_alpha * prior_end) / (total + end_shrinkage_alpha)
                continue_total = total - end_count
                dist = {}
                if continue_total > 0:
                    for label, cnt in label_counts.items():
                        if label == "__END__":
                            continue
                        dist[label] = (1 - p_end) * (cnt / continue_total)
                if p_end > 0:
                    dist["__END__"] = p_end
                entry[bucket] = {
                    label: round(p, 4)
                    for label, p in sorted(dist.items(), key=lambda kv: -kv[1])
                }
            else:
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
    parser.add_argument("--end-shrinkage-alpha", type=float, default=0.0,
                        help="Dirichlet pseudo-count pulling each bucket's P(__END__) "
                             "toward the global pooled END-rate (see to_probs's "
                             "docstring). Default 0 (disabled): tried at alpha=20 "
                             "and measured worse completion/precision/TVD than the "
                             "unsmoothed baseline (docs/ROADMAP.md, A1-Update Teil 8) "
                             "-- the real global END-rate (2.86%) is too low, and the "
                             "main problematic decision point turned out to have a "
                             "large (~1887), confident sample behind its near-zero "
                             "rate, not sparse noise. Kept as a tested, opt-in option.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[load] Reading log: {args.log} ...")
    df = load_log(args.log)
    df_complete = filter_to_complete(df)
    print(f"[load] {df_complete['case_id'].nunique():,} cases.")

    comp = PetriNetProcessComponent(bpmn_path=str(args.bpmn), seed=42)
    print("[mine] Replaying real log on the Petri net (repairing deviations) ...")
    t0 = time.perf_counter()
    counts, stats = mine(df_complete, comp)
    n_total = stats["n_perfect"] + stats["n_repaired"]
    print(f"[mine] Done in {time.perf_counter() - t0:.0f}s — "
          f"{stats['n_perfect']:,}/{n_total:,} cases fit without repair "
          f"({100 * stats['n_perfect'] / n_total:.1f}%); "
          f"{stats['n_repaired']:,} needed repair and still reached the end "
          f"({100 * stats['n_repaired'] / n_total:.1f}%) — "
          f"{stats['n_repair_events']:,} repair events, "
          f"{stats['n_missing_tokens']:,} missing tokens inserted"
          + (f", {stats['n_unrepairable_events']:,} unrepairable (no matching "
             "transition, skipped)" if stats["n_unrepairable_events"] else "")
          + ".")

    dp_probs = to_probs(counts, args.min_samples, args.end_shrinkage_alpha)
    out = {
        "bpmn": args.bpmn.name,
        "visit_bucket_max": DP_VISIT_BUCKET_MAX,
        "min_samples": args.min_samples,
        "end_shrinkage_alpha": args.end_shrinkage_alpha,
        "global_end_rate": round(global_end_rate(counts), 4),
        "replay_perfect_fit_pct": round(100 * stats["n_perfect"] / n_total, 2),
        "n_cases_repaired": stats["n_repaired"],
        "n_repair_events": stats["n_repair_events"],
        "n_missing_tokens": stats["n_missing_tokens"],
        "n_unrepairable_events": stats["n_unrepairable_events"],
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

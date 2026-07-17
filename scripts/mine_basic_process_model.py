"""
mine_basic_process_model.py
============================
Section 1.4 Basic, redesigned: produces the adjacency artifact that
process.py's Basic ProcessComponent uses to *enforce* a process model,
instead of relying only on empirical next-activity probabilities with no
structural constraint at all.

Why this replaces the previous Basic
-------------------------------------
The plain BRANCHING_PROBS table (process.py) is not a process model in any
sense the lecture uses the term (BPMN / Petri net / "hard-coded control-flow"
— see Deck 04, "Control-Flow perspective"): it is a flat bigram frequency
table with no notion of which transitions are structurally legal. Measured
consequence: 0% of its simulated traces replay on the reference model
(output/validation/process_model_comparison/basic.json). It quietly does
1.5 (branching) work while claiming to satisfy 1.4 (a model being enforced).

What this script does instead
-------------------------------
1. Reuse a BPMN discovered by the Inductive Miner ("IMf" — noise_threshold
   filters infrequent behavior) if one is already cached on disk. Only mine
   a fresh one (requires --log, takes a few minutes) if no cached model
   exists yet — this mirrors discover_process_model.py's discovery call
   (same algorithm, same default noise=0.2, tag "im02") so re-running this
   script never silently produces a second, differently-tagged model.
2. Convert the BPMN to a Petri net (pm4py), then derive the *legal direct-
   succession relation implied by the model's own structure* via a large
   random playout (pm4py.play_out — respects the net's firing semantics,
   including tau-transitions and concurrency, without us re-implementing
   marking traversal) followed by Directly-Follows-Graph discovery
   (pm4py.discover_dfg) on the synthetic playout log. The result is an
   approximation (Monte-Carlo, not exhaustive marking enumeration) of
   "which activity can legally follow which, according to this model" —
   deliberately NOT full per-case Petri net algebra (that is the Advanced
   differentiator per the assignment). Basic gets a
   static adjacency lookup table instead of dynamic per-case marking
   tracking — clearly weaker, clearly still "a selected process model is
   enforced".
3. Save {activity: sorted[legal successors]} to
   simulation/models/basic_adjacency.json. process.py's ProcessComponent
   loads this and intersects it with BRANCHING_PROBS candidates at every
   branching decision (probabilities as before, structural legality now
   enforced on top — see process.py::_next_activity).

Usage (from the repo root):
    python scripts/mine_basic_process_model.py
        # reuses simulation/models/bpic17_process_im02.bpmn if present

    python scripts/mine_basic_process_model.py --log data/BPIChallenge2017.xes.gz
        # only actually used if the cached BPMN is missing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pm4py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BPMN = REPO_ROOT / "simulation" / "models" / "bpic17_process_im02.bpmn"
OUTPUT_PATH = REPO_ROOT / "simulation" / "models" / "basic_adjacency.json"
DEFAULT_NOISE = 0.2  # same as discover_process_model.py's im02 tag
DEFAULT_NO_TRACES = 20000
DEFAULT_MAX_TRACE_LENGTH = 300

# Candidate raw-log locations for the (optional) stochastic playout weighting
# — same convention as setup_models.py's LOG_CANDIDATES.
LOG_CANDIDATES = [
    "BPIChallenge2017.xes", "BPIChallenge2017.xes.gz",
    "data/BPIChallenge2017.xes", "data/BPIChallenge2017.xes.gz",
    "BPIChallenge2017.csv", "data/BPIChallenge2017.csv",
]


def find_log(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for cand in LOG_CANDIDATES:
        p = REPO_ROOT / cand
        if p.is_file():
            return p
    return None


def load_native_xes(log_path: Path):
    """Load with pm4py's own reader, keeping its native XES column names
    (case:concept:name / concept:name / time:timestamp) — required by
    pm4py.play_out's stochastic variant and pm4py.discover_bpmn_inductive,
    unlike extract_log_info.load_log's simplified schema (case_id/activity/
    timestamp), which is a different, incompatible convention used
    elsewhere in this repo (mine_dp_probs.py, train_decision_rules.py).
    Mirrors discover_process_model.py's loading exactly."""
    df = pm4py.read_xes(str(log_path))
    if "lifecycle:transition" in df.columns:
        df = df[df["lifecycle:transition"].astype(str).str.lower() == "complete"]
    return df


def ensure_bpmn(bpmn_path: Path, log_path: Path | None, noise: float) -> Path:
    """Reuse a cached discovered BPMN; only mine a fresh one if missing."""
    if bpmn_path.is_file():
        print(f"[reuse] Found cached model at {bpmn_path} — not re-mining.")
        return bpmn_path

    if log_path is None:
        sys.exit(
            f"[ERROR] No cached model at {bpmn_path} and no raw log found.\n"
            "        Either point --bpmn at an existing model, or pass --log "
            "so a fresh one can be mined (Inductive Miner, a few minutes)."
        )

    print(f"[mine] No cached model found. Mining a fresh one from {log_path} "
          f"(Inductive Miner, noise={noise}) ...")
    t0 = time.perf_counter()
    df = load_native_xes(log_path)
    bpmn = pm4py.discover_bpmn_inductive(df, noise_threshold=noise)
    bpmn_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_bpmn(bpmn, str(bpmn_path))
    print(f"[mine] Done in {time.perf_counter() - t0:.0f}s -> {bpmn_path}")
    return bpmn_path


def derive_adjacency(bpmn_path: Path, no_traces: int, max_trace_length: int,
                      log_path: Path | None) -> dict:
    """Playout + DFG discovery: the direct-succession relation implied by
    the model's own firing semantics (tau-transitions and concurrency
    resolved by pm4py, not by us).

    If a raw log is available, the playout is *stochastic*: transitions are
    picked with probability proportional to how often the real log actually
    used them at each marking (pm4py builds this map from a replay of
    `log_path`), not uniformly at random. This matters concretely: a first
    version of this script used uniform-random playout and — verified via a
    separate static reachability check on the net's arc structure — MISSED
    several real, structurally-legal self-loop edges (e.g. `W_Complete
    application -> W_Complete application`, the single most frequent real
    continuation at 43% of visits) purely because a uniform random walk
    rarely happens to retrace a loop that has many competing alternatives at
    each step. Weighting the walk by real transition frequency fixes this:
    the loop-continuation IS the frequent choice in the real data, so a
    frequency-weighted walk samples it often, the way pure Monte-Carlo luck
    would not. Falls back to a larger uniform-random sample if no raw log is
    available (still an approximation — documented as such in the output)."""
    model = pm4py.read_bpmn(str(bpmn_path))
    net, im, fm = pm4py.convert_to_petri_net(model)

    # pm4py's playout variants read these via their own Parameters enum,
    # whose *values* (not the enum names) are the dict keys get_param_value
    # looks up — "NO_TRACES"/"MAX_TRACE_LENGTH" as literal strings are
    # silently ignored (falls back to the default of 1000/1000).
    parameters = {"noTraces": no_traces, "maxTraceLength": max_trace_length}
    variant = "uniform-random"
    if log_path is not None:
        print(f"[playout] Loading {log_path} to weight the playout by real "
              "transition frequency ...")
        real_df = load_native_xes(log_path)
        parameters["log"] = real_df
        variant = "frequency-weighted (stochastic)"

    print(f"[playout] Petri net: {len(net.places)} places, "
          f"{len(net.transitions)} transitions. "
          f"Sampling {no_traces} traces (max length {max_trace_length}, "
          f"{variant}) ...")
    t0 = time.perf_counter()
    playout_log = pm4py.play_out(net, im, fm, parameters=parameters)
    print(f"[playout] Done in {time.perf_counter() - t0:.0f}s "
          f"({len(playout_log)} traces).")

    dfg, start_acts, end_acts = pm4py.discover_dfg(playout_log)
    adjacency: dict[str, list[str]] = {}
    for (a, b) in dfg:
        adjacency.setdefault(a, set()).add(b)
    adjacency = {a: sorted(succs) for a, succs in sorted(adjacency.items())}
    print(f"[dfg] {len(adjacency)} activities with at least one legal successor.")
    return adjacency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpmn", type=Path, default=DEFAULT_BPMN,
                        help="Discovered BPMN to reuse if present (default: "
                             "the existing im02 Inductive Miner model).")
    parser.add_argument("--log", type=Path, default=None,
                        help="Raw log. Auto-detected in the repo root/data/ "
                             "if not given (see LOG_CANDIDATES). Used (a) to "
                             "mine a fresh BPMN if --bpmn is missing, and (b) "
                             "to weight the playout by real transition "
                             "frequency (see derive_adjacency docstring) — "
                             "pass --no-log-weighting to skip (b) even if a "
                             "log is found.")
    parser.add_argument("--no-log-weighting", action="store_true",
                        help="Use uniform-random playout even if a raw log "
                             "is available (larger --no-traces recommended "
                             "then; see derive_adjacency docstring for why "
                             "uniform-random risks missing loop-back edges).")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE,
                        help="Inductive Miner noise threshold for the "
                             "fresh-mining fallback (default 0.2).")
    parser.add_argument("--no-traces", type=int, default=DEFAULT_NO_TRACES,
                        help="Playout sample size (default 20000).")
    parser.add_argument("--max-trace-length", type=int, default=DEFAULT_MAX_TRACE_LENGTH,
                        help="Playout per-trace cap, guards against unbounded "
                             "loops (default 300).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    log_path = find_log(args.log)
    bpmn_path = ensure_bpmn(args.bpmn, log_path, args.noise)
    weighting_log = None if args.no_log_weighting else log_path
    adjacency = derive_adjacency(
        bpmn_path, args.no_traces, args.max_trace_length, weighting_log)

    out = {
        "source_bpmn": bpmn_path.name,
        "method": "playout+dfg",
        "playout_variant": "uniform-random" if weighting_log is None
                            else "frequency-weighted (stochastic)",
        "no_traces": args.no_traces,
        "max_trace_length": args.max_trace_length,
        "adjacency": adjacency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[save] {sum(len(v) for v in adjacency.values())} legal edges "
          f"across {len(adjacency)} activities -> {args.output}")


if __name__ == "__main__":
    main()

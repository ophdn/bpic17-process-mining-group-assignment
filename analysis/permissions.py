"""
permissions.py — Section 1.7, resource permissions (fitting side).

Fits two kinds of permission model from the BPIC-17 log and writes them as JSON
for `simulation/components/permissions.py` to consume at runtime.

  Basic     — a resource x activity matrix. Permitted iff observed.

  Advanced  — an organizational model discovered with **OrdinoR**:

      Jing Yang, Chun Ouyang, Wil M. P. van der Aalst, Arthur H. M. ter Hofstede,
      Yang Yu. "OrdinoR: A framework for discovering, evaluating, and analyzing
      organizational models using event logs."
      Decision Support Systems 158:113771, 2022.

    Resources are clustered into (possibly overlapping) groups over *execution
    contexts* — triples (case type, activity type, time type) — each group is
    assigned the capabilities its members demonstrate, and the model is scored
    with the paper's own fitness and precision measures.

----------------------------------------------------------------------------
Decision — preprocess to lifecycle == "complete", the opposite of Section 1.6
----------------------------------------------------------------------------
The paper filters BPIC-17 to completion events only, "ensur[ing] that each
activity instance in process execution is counted exactly once" (§6.1). We do the
same, and it reproduces their reported dataset exactly:

                          cases     events   activities  resources
    Paper, Table 6       31,509    475,306       24         144
    Ours                 31,509    475,306       24         144

This is the *same filter we rejected* in Section 1.6, and both calls are right,
because the two sections ask different questions:

  - Availability asks *when was this person at work*. A `start` is the signal
    that someone began working, so filtering to `complete` discards all 128,227
    of them — it deletes the very thing being measured.

  - Permissions ask *what work does this person do*. Here each activity instance
    should count once. A long task with five suspend/resume cycles is one piece
    of work, not eleven; without the filter it would carry eleven times the weight
    in that resource's profile and distort the clustering.

Same log, opposite filters, different questions.

----------------------------------------------------------------------------
Decision — why the basic matrix is not good enough
----------------------------------------------------------------------------
It is pure memorisation: a (resource, activity) pair is permitted iff it was
literally seen. Held out temporally — fit on the first 70% of the log, applied to
the last 30% — it forbids 2,967 events that actually happened (2.45% of test
events by resources it already knows, across 116 distinct pairs). Those are not
violations; they are the model's blind spots.

It also rests on thin evidence: of its 2,188 granted permissions (a 144 x 24
matrix, 63.3% dense), a third are backed by 10 or fewer observations and 9.7% by
a single one. A lone observation is as likely to be a stand-in or a one-off
escalation as a standing permission.

An organizational model generalises: a resource inherits its *group's*
capabilities, so it can be permitted an activity it was never individually
observed doing, on the evidence of its colleagues rather than its own one event.

----------------------------------------------------------------------------
Upstream note
----------------------------------------------------------------------------
`ordinor` 0.2.1 predates NumPy 2.0 and calls `np.infty`, which NumPy 2.0 removed.
We restore the alias before importing it. Both call sites are in `moc.py` and mean
`np.inf`. This is an upstream incompatibility, not a modelling choice.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Must precede any ordinor import — see the module docstring.
if not hasattr(np, "infty"):
    np.infty = np.inf

# The paper defines time types as the seven week days (§5.1.1).
TIME_RESOLUTION = "weekday"

# Case types from trace clustering use this many clusters. Set to the cardinality
# of `case:LoanGoal` (14) so that CT-from-attribute and CT-from-clustering yield
# the *same number* of case types. That makes the comparison a controlled one: it
# isolates how case types are defined, rather than confounding it with how many
# there are.
N_TRACE_CLUSTERS = 14


@contextmanager
def _quiet():
    """OrdinoR prints a banner on every call; keep notebook output readable."""
    buf, saved = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = saved


def prepare_log(df: pd.DataFrame) -> pd.DataFrame:
    """The paper's preprocessing: completion events only (see module docstring)."""
    el = df[df["lifecycle:transition"] == "complete"].copy()
    return el[[
        "case:concept:name", "concept:name", "org:resource",
        "time:timestamp", "case:LoanGoal", "case:ApplicationType",
    ]]


# ──────────────────────────────────────────────────────────────────────────
# Basic — the observed resource x activity matrix
# ──────────────────────────────────────────────────────────────────────────

def permission_matrix(el: pd.DataFrame) -> pd.DataFrame:
    """Boolean resource x activity matrix: permitted iff observed."""
    return pd.crosstab(el["org:resource"], el["concept:name"]) > 0


def observed_permissions(el: pd.DataFrame) -> Dict[str, Set[str]]:
    """resource -> activities it was observed performing."""
    return {
        r: set(g["concept:name"].unique())
        for r, g in el.groupby("org:resource")
    }


def holdout_gap(el: pd.DataFrame, train_frac: float = 0.7) -> Dict[str, float]:
    """How often would the observed-only matrix forbid something that happens?

    Fit on the earliest `train_frac` of the log, then count events in the
    remainder — by resources the matrix already knows — whose activity it would
    not permit. Those are not violations; they are the model's blind spots.
    """
    d = el.sort_values("time:timestamp")
    cut = d["time:timestamp"].quantile(train_frac)
    train, test = d[d["time:timestamp"] <= cut], d[d["time:timestamp"] > cut]

    seen = set(zip(train["org:resource"], train["concept:name"]))
    known = test[test["org:resource"].isin(train["org:resource"].unique())]
    pairs = list(zip(known["org:resource"], known["concept:name"]))
    unseen = [p for p in pairs if p not in seen]

    return {
        "test_events": len(pairs),
        "forbidden_events": len(unseen),
        "forbidden_rate": len(unseen) / len(pairs) if pairs else 0.0,
        "forbidden_pairs": len(set(unseen)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Advanced — OrdinoR
# ──────────────────────────────────────────────────────────────────────────

def trace_clustering_partition(el: pd.DataFrame, path: str | Path,
                               n_clusters: int = N_TRACE_CLUSTERS,
                               seed: int = 42) -> Path:
    """Cluster cases by their activity profile and write OrdinoR's partition file.

    The paper derives case types from Bose & van der Aalst's context-aware trace
    clustering. We substitute k-means over each case's activity profile (the bag
    of activities it contains, TF-IDF weighted so that common activities like
    A_Create Application, which every case has, do not dominate the distance).

    This is a deliberate deviation and we name it: it is *a* trace clustering, not
    *their* trace clustering, so our CT(tc) numbers are not expected to match the
    paper's to the third decimal. What it does test is the paper's claim that case
    types derived from a case's actual behaviour beat case types read off a single
    case attribute.

    Writes `case_id <TAB> cluster`, which is the format OrdinoR expects.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfTransformer

    profile = pd.crosstab(el["case:concept:name"], el["concept:name"])
    X = TfidfTransformer().fit_transform(profile.values)

    labels = KMeans(n_clusters=n_clusters, random_state=seed,
                    n_init=10).fit_predict(X)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        f"{cid}\t{lab}\n" for cid, lab in zip(profile.index, labels)))
    return path


def build_resource_log(el: pd.DataFrame, contexts: str,
                       partition_file: Optional[str | Path] = None):
    """Learn execution contexts and derive the resource log (paper §5.1.1).

    `contexts`:
      "ATonly"  — the context *is* the activity; case and time types are ⊥.
                  Directly comparable to the basic matrix.
      "CT+AT+TT(ca)"  — case types from the `case:LoanGoal` attribute (the paper
                  used loan purpose), activity types = activity labels, time types
                  = the seven week days.
      "CT+AT+TT(tc)"  — as above, but case types from trace clustering.
    """
    from ordinor.execution_context import (
        ATonlyMiner, FullMiner, TraceClusteringFullMiner)

    with _quiet():
        if contexts == "ATonly":
            miner = ATonlyMiner(el)
        elif contexts == "CT+AT+TT(ca)":
            miner = FullMiner(el, case_attr_name="case:LoanGoal",
                              resolution=TIME_RESOLUTION)
        elif contexts == "CT+AT+TT(tc)":
            if partition_file is None:
                raise ValueError("CT+AT+TT(tc) needs a trace-clustering partition")
            miner = TraceClusteringFullMiner(
                el, fn_partition=str(partition_file), resolution=TIME_RESOLUTION)
        else:
            raise ValueError(f"unknown execution-context method: {contexts!r}")

        rl = miner.derive_resource_log(el)

    return miner, rl


# ──────────────────────────────────────────────────────────────────────────
# Conformance measures (paper §4.6), vectorised
# ──────────────────────────────────────────────────────────────────────────
#
# These reimplement Definitions 8-12 exactly. We do not use ordinor's own
# `fitness`/`precision` in the search loop, for a practical reason: they iterate
# the resource log event by event and take ~40 s per call on BPIC-17. The paper's
# OverallScore needs an 81-point grid search, so ordinor runs those evaluations
# through an *uncapped* `multiprocessing.Pool()` — one worker per core, each
# holding a copy of the 475k-row resource log. On a 16-core machine that exhausts
# memory and takes the host down with it. (It did.)
#
# The measures are set operations over execution contexts, so they vectorise to
# well under a second, and the search then runs in a single process with a bounded
# memory footprint. `validate_measures()` below asserts our values match ordinor's
# to 1e-9 on a real model, so this is a speed-up, not a reinterpretation.


def _context_candidates(om) -> Dict[tuple, Set[str]]:
    """Execution context -> resources the model allows to work in it (Def. 10)."""
    cand: Dict[tuple, Set[str]] = {}
    for gid, members in om.find_all_groups():
        for co in om.find_group_execution_contexts(gid):
            cand.setdefault(tuple(co), set()).update(members)
    return cand


def conformance(rl, om) -> Tuple[float, float, float]:
    """(fitness, precision, F1) of an organizational model on a resource log.

    fitness   (Def. 9)  = |conforming events| / |events with a resource|
    precision (Def. 12) = mean over conforming events of
                          (|cand(E)| - |cand(e)| + 1) / |cand(E)|,
                          averaged over the *allowed* events.
    """
    cand = _context_candidates(om)

    ctx = list(zip(rl["case_type"], rl["activity_type"], rl["time_type"]))
    res = rl["org:resource"].to_numpy()

    # |cand(E)|: every resource the model would allow for some event in the log.
    all_cand: Set[str] = set()
    for co in set(ctx):
        all_cand |= cand.get(co, set())
    n_cand_all = len(all_cand)

    n_events = len(rl)
    if n_events == 0 or n_cand_all == 0:
        return 0.0, 0.0, 0.0

    # Per distinct context: how many candidates, and who they are.
    sizes = {co: len(cand.get(co, ())) for co in set(ctx)}

    n_conf = 0
    n_allowed = 0
    prec_sum = 0.0
    for co, r in zip(ctx, res):
        k = sizes[co]
        if k == 0:
            continue                     # not an allowed event
        n_allowed += 1
        if r in cand[co]:                # conforming (Def. 8)
            n_conf += 1
            prec_sum += (n_cand_all - k + 1) / n_cand_all

    fit = n_conf / n_events
    prec = prec_sum / n_allowed if n_allowed else 0.0
    f1 = (2 * fit * prec / (fit + prec)) if (fit + prec) > 0 else 0.0
    return fit, prec, f1


def validate_measures(rl, om) -> Dict[str, float]:
    """Assert our vectorised measures agree with ordinor's reference ones.

    Run once per session in the notebook. If this passes, every number produced by
    the fast path is a number ordinor would have produced.
    """
    from ordinor.conformance import fitness as _fit, precision as _prec

    with _quiet():
        ref_f, ref_p = float(_fit(rl, om)), float(_prec(rl, om))
    our_f, our_p, _ = conformance(rl, om)

    assert abs(ref_f - our_f) < 1e-9, f"fitness differs: {ref_f} vs {our_f}"
    assert abs(ref_p - our_p) < 1e-9, f"precision differs: {ref_p} vs {our_p}"

    return {"ordinor_fitness": ref_f, "ours_fitness": our_f,
            "ordinor_precision": ref_p, "ours_precision": our_p}


@dataclass
class OrgModel:
    """A discovered organizational model and its conformance scores."""

    contexts: str
    discovery: str
    profiling: str
    n_contexts: int
    n_groups: int
    fitness: float
    precision: float
    f1: float
    mean_groups_per_resource: float
    params: Optional[dict] = None      # OverallScore's fitted lambda / w1
    om: object = field(repr=False, default=None)

    def as_row(self) -> dict:
        return {
            "contexts": self.contexts,
            "discovery": self.discovery,
            "profiling": self.profiling,
            "#contexts": self.n_contexts,
            "#groups": self.n_groups,
            "fitness": round(self.fitness, 3),
            "precision": round(self.precision, 3),
            "F1": round(self.f1, 3),
        }


def discover(rl, n_groups: int = 10, discovery: str = "AHC",
             profiling: str = "OverallScore", contexts: str = "",
             seed: int = 42) -> OrgModel:
    """Discover groups, profile them, and score the model (paper §5.1.2-5.1.3).

    `discovery`:
      "AHC" — Agglomerative Hierarchical Clustering. Disjoint groups.
      "MOC" — Model-based Overlapping Clustering. A resource may hold several
              roles at once, which is what real organisations look like.

    `profiling`:
      "FullRecall"   — a group can do anything any member has done. The paper
                       shows this yields perfect fitness and useless precision
                       (0.169 on BPIC-17) — a "flower model" that permits nearly
                       everything. Kept so we can reproduce that result, not
                       because it is a candidate.
      "OverallScore" — a context is a group capability only if
                       w1*RelStake + w2*Coverage >= lambda, i.e. the group does a
                       substantial share of that work AND enough of its members
                       do it. Weights and threshold by grid search, as the paper.
    """
    from ordinor.org_model_miner.resource_features import direct_count
    from ordinor.org_model_miner.group_discovery import ahc, moc
    from ordinor.org_model_miner.group_profiling import full_recall, overall_score

    with _quiet():
        profiles = direct_count(rl, scale="log")

        if discovery == "AHC":
            groups = ahc(profiles, n_groups=n_groups)
        elif discovery == "MOC":
            groups = moc(profiles, n_groups=n_groups, n_init=10)
        else:
            raise ValueError(f"unknown discovery method: {discovery!r}")

        if profiling == "FullRecall":
            om = full_recall(groups, rl)
            best = None
        elif profiling == "OverallScore":
            om, best = _overall_score_search(groups, rl)
        else:
            raise ValueError(f"unknown profiling method: {profiling!r}")

    f, p, s = conformance(rl, om)

    return OrgModel(
        contexts=contexts, discovery=discovery, profiling=profiling,
        n_contexts=int(profiles.shape[1]),
        n_groups=om.group_number,
        fitness=f, precision=p, f1=s,
        mean_groups_per_resource=sum(len(g) for g in groups) / profiles.shape[0],
        params=best,
        om=om,
    )


def _overall_score_search(groups, rl):
    """Grid-search OverallScore's weight and threshold, as the paper does (§6.2).

    Same grid as the paper: lambda and w1 each over [0.1, 0.9] in steps of 0.1,
    selecting the model with the best F1.

    Run in a single process. Ordinor's own `auto_search=True` farms these 81
    evaluations out to an uncapped `multiprocessing.Pool()` — one worker per core,
    each with a copy of the resource log — which exhausts memory on a many-core
    machine. With `conformance()` costing well under a second, there is nothing to
    parallelise anyway.
    """
    from copy import deepcopy
    from ordinor.org_model_miner.group_profiling import overall_score

    best_om, best_f1, best_params = None, -1.0, None

    for lam in [x / 10 for x in range(1, 10)]:
        for w1 in [x / 10 for x in range(1, 10)]:
            with _quiet():
                om = overall_score(deepcopy(groups), rl, p=lam, w1=w1)
            if om.group_number == 0:
                continue
            _, _, f1 = conformance(rl, om)
            if f1 > best_f1:
                best_om, best_f1, best_params = om, f1, {"lambda": lam, "w1": w1}

    if best_om is None:      # no threshold produced a usable model
        from ordinor.org_model_miner.group_profiling import full_recall
        with _quiet():
            best_om = full_recall(groups, rl)
        best_params = {"fallback": "FullRecall"}

    return best_om, best_params


# ──────────────────────────────────────────────────────────────────────────
# Export to the runtime format
# ──────────────────────────────────────────────────────────────────────────

def org_model_to_json(model: OrgModel, path: str | Path) -> Path:
    """Write a discovered model in the shape OrgModelPermissions expects.

    Execution contexts keep all three dimensions. OrdinoR names types
    "CT.<x>" / "AT.<x>" / "TT.<x>", and uses the empty string for ⊥. The runtime
    side strips the AT prefix to recover the activity name and treats "" as a
    wildcard, so an AT-only model degrades to a plain activity lookup.
    """
    om = model.om
    groups = []
    for gid, members in om.find_all_groups():
        caps = []
        for (ct, at, tt) in om.find_group_execution_contexts(gid):
            activity = at[3:] if at.startswith("AT.") else at
            caps.append([ct or "", activity, tt or ""])
        groups.append({
            "members": sorted(members),
            "capabilities": sorted(caps),
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": "org_model",
        "meta": model.as_row(),
        "groups": groups,
    }, indent=1))
    return path


def static_to_json(perms: Dict[str, Set[str]], path: str | Path) -> Path:
    """Write a resource -> activities map in the runtime format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": "static",
        "permissions": {r: sorted(a) for r, a in sorted(perms.items())},
    }, indent=1))
    return path

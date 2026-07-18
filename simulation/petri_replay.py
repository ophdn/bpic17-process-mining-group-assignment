"""
petri_replay.py — shared repair utilities for replaying the real log on the net
==============================================================================
Both mining scripts replay real BPIC-17 traces against the Petri net that
``PetriNetProcessComponent`` enforces at runtime:

- ``scripts/mine_dp_probs.py``     — branching + termination probabilities (1.5)
- ``train_decision_rules.py``      — decision-point classifiers (1.5 Advanced I)

Only ~57.7% of real cases replay end-to-end without ever stepping outside
the net's legal frontier. Abandoning a case's remaining trace at its first
deviation — what both scripts used to do — biases the result in a specific
way: decisions *before* the deviation still get counted for every case, so
early decision points keep near-full data, while everything reachable only
*after* one is fitted on a progressively more "compliant-only"
subpopulation. Deep decision points and high visit buckets suffer worst.

These helpers let a replay repair the deviation and keep going instead:
force-fire the deviating activity's transition, topping up its under-marked
input places with the tokens it is missing. That is the standard
token-based-replay "missing token" convention (the same idea pm4py's own
conformance checker uses), and it is deliberately *not* alignment-based
replay: the patch is local and greedy at the deviation, with no lookahead
and no backtracking, where an alignment would search for a globally
cost-minimal sequence of synchronous/log/model moves.

Callers are expected to *not* record the deviating step itself as an
observed choice — it was never a legal option at that marking, so counting
it would corrupt that decision point's distribution. The point of repairing
is the data that comes after it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking


def label_transition_map(net) -> dict:
    """label -> transitions carrying it, sorted by name.

    Repair has to find a transition *by label* whether or not it is enabled
    right now — that is precisely the case being repaired — so it cannot use
    ``PetriNetProcessComponent._fire_activity``, which only ever searches
    among the transitions enabled at the current marking.

    Sorted for the same reason ``_visible_frontier`` sorts: pm4py returns
    transitions in a set whose iteration order follows object identity, so
    an unsorted tie-break would make the mined artifact depend on memory
    layout rather than on the log.
    """
    mapping: dict = defaultdict(list)
    for t in net.transitions:
        if t.label is not None:
            mapping[t.label].append(t)
    for label in mapping:
        mapping[label].sort(key=lambda t: t.name)
    return mapping


def repair_and_fire(net, marking, label_map: dict, label: str):
    """Force *label* to fire from *marking* even though no combination of tau
    transitions enables it, then return ``(new_marking, n_missing_tokens)``.

    Returns ``(None, 0)`` when the net has no transition for *label* at all.
    That happens for real: the Signavio net has no transition for a handful
    of rare BPIC-17 activities (``O_Sent (online only)``,
    ``W_Assess potential fraud``, ``W_Call after offers``,
    ``W_Shortened completion``) — a genuine vocabulary gap in the model, and
    one that used to be indistinguishable from an ordinary branching
    deviation because any deviation ended the replay. Callers should count
    these separately and skip the event *without* advancing the marking,
    which is what an alignment would call a log move.
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

    # Clamp back to the net's 1-safe invariant. This net models one case's
    # control state, so a place is meant to hold at most one token; a repair
    # can break that by adding a token to a place that still holds one from
    # earlier in the trace (e.g. an AND-split branch whose join an earlier
    # repair skipped). Harmless for correctness — the place still just means
    # "this branch is pending" — but every surplus token multiplies the
    # tau-combinations the caller's tau-closure search must enumerate at each
    # later decision. Measured before this clamp: a single 30-event case with
    # 6 repairs took 35s instead of <0.1s, and a full-log run burned 76
    # minutes of CPU without finishing.
    for place, count in list(best_marking.items()):
        if count > 1:
            best_marking[place] = 1
    return best_marking, best_missing

"""
petri_process.py — Process Component (Section 1.4 Advanced)
=============================================================
Loads a .bpmn file, converts it to a Petri net (pm4py), and enforces
control-flow via Petri net firing rules (marking + enabled transitions)
instead of the flat next-activity probability table in process.py.

Each case gets its own token marking, starting at the net's initial
marking. At every decision point, the *set of legal next activities* is
exactly the transitions enabled in that marking — a hard constraint
coming from the Petri net's structure (sequence/XOR/AND/loop blocks),
not just "what followed this activity somewhere in the log". Among the
legal activities, one of two branching strategies picks which one happens
(``branching_mode``, Section 1.5):

  - "probs" (Basic): BRANCHING_PROBS (see process.py) as a soft preference,
    renormalised over just the legal subset.
  - "visit" (A1 termination fix): like "probs", but conditioned on how often
    the current activity has already occurred in this case
    (branching_probs_by_visit in simulation_inputs.json, buckets 1/2/3+).
    Memoryless probabilities understate loop-exit likelihood, which made
    cases cycle the validation/offer loops (measured: 4.35× W_Validate
    application per case vs. 0.50 real, only 2% of cases terminating).
    Falls back to the global table for sparse buckets.
  - "rules" (Advanced I): case/runtime data attributes (ApplicationType,
    LoanGoal, RequestedAmount, and the offer attributes set once
    O_Create Offer has fired) are sampled per case and fed into a
    DecisionTreeClassifier trained per decision point (see
    train_decision_rules.py) — falls back to "probs" for any decision
    point / attribute combination the trained artifact doesn't cover.

Silent (tau) transitions from the BPMN's gateway/skip/loop structure are
fired automatically and never appear in the event log.

Everything else (processing-time sampling in distribution/ml_model/
ml_probabilistic mode, resource release, the loop-guard termination
check) is inherited unchanged from ProcessComponent — only the
"which activity comes next" decision is replaced.

Upgrade note (from process.py):
    "Section 1.4 Advanced: replace _next_activity() with a Petri net /
    BPMN loader" — this class is that replacement. ProcessComponent
    (Basic) is left untouched so both can be compared (see
    scripts/compare_process_models.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pm4py
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking

from ..core.events import SimEvent, EventType
from .process import (
    APPLICATION_TYPE_PROBS,
    BRANCHING_PROBS,
    LOAN_GOAL_GIVEN_APPLICATION_TYPE,
    OFFER_ATTRIBUTE_PARAMS,
    OFFER_ATTRIBUTE_ZERO_MASS_PARAMS,
    REQUESTED_AMOUNT_GIVEN_APPLICATION_TYPE,
    ProcessComponent,
)

# Residual weight given to a legal next activity that BRANCHING_PROBS has
# no entry for at this point (keeps every enabled transition reachable
# instead of only ever following historically-observed edges).
RESIDUAL_WEIGHT = 0.01

# Loop-guards for the *activity* loop. Unlike process.py's
# MAX_ACTIVITY_REPEATS=15 — tuned for a flat
# probability graph with no legality check — these only need to bound
# worst-case runtime: legality is already enforced by the net itself, and
# BPIC-17 genuinely contains cases that cycle the offer sub-loop (create/
# send/return an offer) 20+ times. Set generously so real loop behaviour
# isn't guillotined before it exits on its own.
MAX_ACTIVITY_REPEATS = 60
MAX_TOTAL_ACTIVITIES = 400

# Visit-conditioned branching table (branching_mode="visit"): produced by
# extract_log_info.extract_branching_by_visit into simulation_inputs.json.
INPUTS_PATH = Path(__file__).resolve().parents[2] / "simulation_inputs.json"
VISIT_BUCKET_MAX = 3  # buckets "1", "2", "3+" — keep in sync with extract_log_info

# Decision-point-level branching table (scripts/mine_dp_probs.py): the
# preferred source in "visit" mode. Unlike trace bigrams it is mined by
# replaying the real log on this exact net, so it never mixes concurrency
# interleavings into a decision point's distribution, and it is conditioned
# on the case's visit count of that decision point (loop memory).
DP_PROBS_PATH = Path(__file__).resolve().parent.parent / "models" / "dp_branching_probs.json"
DP_VISIT_BUCKET_MAX = 5  # buckets "1".."4", "5+" — keep in sync with mine_dp_probs

# Pseudo-label for "the case ends here": at many markings the final marking
# is reachable via tau transitions ONLY — but a visible loop-back label stays
# enabled too (e.g. the [O_Cancelled] self-loop). Checking marking == fm
# alone therefore never terminates such cases: ending must be an explicit,
# data-driven *choice* mined from where real traces stop (mine_dp_probs.py).
END_LABEL = "__END__"

# The application's outcome milestones. In the real log every case fires
# exactly one of these (per-case frequencies: A_Pending 0.55 + A_Cancelled
# 0.33 + A_Denied 0.12 = 1.00) and the trace ends right after (bar minor
# wrap-up events). The Signavio net keeps loop tokens alive past these
# markings and its final marking is often NOT tau-reachable there, so
# neither marking==fm nor the mined __END__ choice can stop such cases —
# measured effect: cases fired "O_Accepted, A_Pending" and then cycled the
# validation loop for 50+ further events. Domain-level termination rule:
# once the outcome is decided, the case is over.
TERMINAL_OUTCOMES = {"A_Pending", "A_Denied", "A_Cancelled"}


class PetriNetProcessComponent(ProcessComponent):
    """
    Drop-in replacement for ProcessComponent that enforces control-flow
    with a Petri net loaded from a .bpmn file, instead of a plain
    next-activity probability graph. Accepts the same constructor
    arguments as ProcessComponent, plus ``bpmn_path``.
    """

    def __init__(
        self,
        bpmn_path: str,
        seed: Optional[int] = 42,
        mode: str = "distribution",
        model_path: Optional[str] = None,
        start_datetime: Optional[datetime] = None,
        resource_component=None,
        branching_mode: str = "probs",
        decision_rules_path: Optional[str] = None,
    ):
        """
        Parameters (beyond ProcessComponent's) — see module docstring:
        bpmn_path : path to the discovered .bpmn model.
        branching_mode : {"probs", "rules"}. "rules" (Section 1.5 Advanced I)
            requires decision_rules_path (joblib artifact from
            train_decision_rules.py, lazy-loaded on first use).
        """
        if branching_mode not in ("probs", "visit", "rules"):
            raise ValueError(
                f"branching_mode must be 'probs', 'visit' or 'rules', got {branching_mode!r}")

        super().__init__(
            seed=seed,
            mode=mode,
            model_path=model_path,
            start_datetime=start_datetime,
            resource_component=resource_component,
        )
        bpmn_model = pm4py.read_bpmn(bpmn_path)
        self.net, self.im, self.fm = pm4py.convert_to_petri_net(bpmn_model)
        self._markings: Dict[str, Marking] = {}
        self._fm_reach_cache: Dict[tuple, bool] = {}

        self.branching_mode = branching_mode

        # Visit-conditioned branching tables ("visit" mode; "rules" mode also
        # uses them as its fallback layer when a decision point has no model).
        self._branching_by_visit: Dict[str, dict] = {}
        self._dp_probs: Dict[str, dict] = {}
        self._dp_visit_counts: Dict[str, Dict[str, int]] = {}
        if branching_mode in ("visit", "rules"):
            try:
                with open(INPUTS_PATH, encoding="utf-8") as f:
                    self._branching_by_visit = json.load(f).get(
                        "branching_probs_by_visit", {})
            except FileNotFoundError:
                pass
            try:
                with open(DP_PROBS_PATH, encoding="utf-8") as f:
                    self._dp_probs = json.load(f).get("dp_probs", {})
            except FileNotFoundError:
                pass
            if branching_mode == "visit" and not (
                    self._branching_by_visit or self._dp_probs):
                raise ValueError(
                    "branching_mode='visit' needs 'branching_probs_by_visit' in "
                    f"{INPUTS_PATH} (extract_log_info.py) and/or "
                    f"{DP_PROBS_PATH} (scripts/mine_dp_probs.py).")

        self._decision_rules_path = decision_rules_path
        self._decision_models: Optional[dict] = None
        self._decision_encoders: Optional[dict] = None
        self._decision_feature_names: Optional[List[str]] = None
        self._decision_unknown: str = "__UNKNOWN__"
        # case_id -> {application_type, loan_goal, requested_amount, has_offer,
        #             credit_score, offered_amount, number_of_terms,
        #             monthly_cost, first_withdrawal_amount}
        self._case_attrs: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        case_id = event.case_id

        if event.activity == "__PROCESS_START__":
            self._repeat_counts[case_id] = {}
            self._dp_visit_counts[case_id] = {}
            self._ctx[case_id] = {
                "start_t": engine.now,
                "position": 0,
                "prev_act": None,
            }
            self._markings[case_id] = Marking(self.im)
            if self.branching_mode == "rules":
                self._case_attrs[case_id] = self._sample_case_attributes()
            first_activity = self._advance_to_next_visible(case_id, current_activity=None)
            if first_activity is None:
                # Malformed net: nothing enabled from the initial marking.
                self._markings.pop(case_id, None)
                self._repeat_counts.pop(case_id, None)
                self._ctx.pop(case_id, None)
                self._case_attrs.pop(case_id, None)
                return
            self._fire_start(engine, case_id, first_activity)
            return

        # Normal activity start (duration sampling, ctx bookkeeping) is
        # identical to the Basic component regardless of routing mechanism.
        super().on_activity_start(engine, event)

    def on_activity_complete(self, engine, event: SimEvent) -> None:
        case_id = event.case_id
        activity = event.activity

        if self._resources is not None and event.resource:
            self._resources.release(engine, event.resource, event.activity)

        counts = self._repeat_counts.get(case_id, {})
        counts[activity] = counts.get(activity, 0) + 1
        self._repeat_counts[case_id] = counts

        self._fire_activity(case_id, activity)

        if self.branching_mode == "rules" and activity == "O_Create Offer":
            self._sample_offer_attributes(case_id)

        marking = self._markings.get(case_id)
        reached_final = marking is not None and marking == self.fm
        if reached_final or self._should_terminate(case_id, activity, counts):
            self._end_case(engine, case_id)
            return

        next_activity = self._advance_to_next_visible(case_id, current_activity=activity)
        if next_activity is None:
            self._end_case(engine, case_id)
            return

        self._fire_start(engine, case_id, next_activity)

    def _should_terminate(self, case_id: str, activity: str, counts: Dict[str, int]) -> bool:
        """
        Override the Basic heuristic: TERMINAL_ACTIVITIES (process.py) marks
        a case done as soon as one specific activity fires, which doesn't
        hold here — the net may still need concurrent branches to
        synchronise (AND-join) before reaching its final marking, which is
        checked separately in on_activity_complete. Only the loop-guards
        remain, as a safety net against pathological/infinite loops.
        """
        if activity in TERMINAL_OUTCOMES:
            return True
        if counts.get(activity, 0) >= MAX_ACTIVITY_REPEATS:
            return True
        if sum(counts.values()) > MAX_TOTAL_ACTIVITIES:
            return True
        return False

    def _end_case(self, engine, case_id: str) -> None:
        self._markings.pop(case_id, None)
        self._repeat_counts.pop(case_id, None)
        self._dp_visit_counts.pop(case_id, None)
        self._ctx.pop(case_id, None)
        self._case_attrs.pop(case_id, None)
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=20,
            event_type=EventType.CASE_COMPLETE,
            case_id=case_id,
        ))

    # ------------------------------------------------------------------
    # Petri net mechanics
    # ------------------------------------------------------------------

    def _fire_activity(self, case_id: str, activity: str) -> None:
        """Consume/produce tokens for the enabled transition labelled *activity*."""
        marking = self._markings.get(case_id)
        if marking is None:
            return
        enabled = sorted(
            semantics.enabled_transitions(self.net, marking), key=lambda t: t.name
        )
        for t in enabled:
            if t.label == activity:
                self._markings[case_id] = semantics.execute(t, self.net, marking)
                return
        # By construction every activity we schedule was itself chosen from
        # _advance_to_next_visible() as an enabled transition for this case,
        # and a case's marking is only ever touched by that case, so this
        # should not happen. Left as a no-op rather than a crash in case a
        # future BPMN edit produces a net where it can.

    def _advance_to_next_visible(
        self, case_id: str, current_activity: Optional[str]
    ) -> Optional[str]:
        """
        Compute the full tau-closure frontier of visible activities reachable
        from the case's current marking — every activity enabled after firing
        *some* combination of invisible transitions, without firing a visible
        one along the way — pick among them, then commit the marking to the
        state at which the chosen activity is actually enabled.

        This has to consider every combination of invisible transitions, not
        just fire them greedily until the first visible one appears: several
        of this net's n-way XOR-splits are encoded by Inductive Miner as a
        chain of nested binary [named activity, tau] gates, where the named
        transition and its sibling tau are enabled *simultaneously*. Firing
        taus only when nothing visible is enabled yet would deterministically
        pick that first named activity every time — the tau leads to other
        visible alternatives further down the chain, which would otherwise be
        structurally unreachable (this was the case before this fix: e.g.
        A_Submitted always fired after A_Create Application, making the
        W_Complete application / A_Concept branches — ~35% of real cases —
        impossible to reach).
        Returns None if the case has no enabled transition left at all
        (dead marking).
        """
        marking = self._markings[case_id]
        frontier = self._visible_frontier(marking)
        if not frontier:
            return None

        labels = sorted(frontier.keys())
        allow_end = self._final_reachable_by_tau(marking)
        chosen = self._weighted_choice(case_id, current_activity, labels,
                                       allow_end=allow_end)
        if chosen == END_LABEL:
            return None  # caller ends the case (final marking is tau-reachable)
        self._markings[case_id] = frontier[chosen]
        return chosen

    def _final_reachable_by_tau(self, marking: Marking) -> bool:
        """True if the net's final marking can be reached from *marking* by
        firing only invisible (tau) transitions — i.e. the case could
        legally stop here. Memoised: the reachable-marking set is small."""
        key = tuple(sorted((p.name, count) for p, count in marking.items()))
        cached = self._fm_reach_cache.get(key)
        if cached is not None:
            return cached

        seen = {key}
        stack = [marking]
        result = False
        while stack:
            m = stack.pop()
            if m == self.fm:
                result = True
                break
            for t in semantics.enabled_transitions(self.net, m):
                if t.label is not None:
                    continue
                nm = semantics.execute(t, self.net, m)
                nkey = tuple(sorted((p.name, c) for p, c in nm.items()))
                if nkey not in seen:
                    seen.add(nkey)
                    stack.append(nm)

        self._fm_reach_cache[key] = result
        return result

    def _visible_frontier(
        self, marking: Marking, _visited: Optional[set] = None
    ) -> Dict[str, Marking]:
        """
        Depth-first tau-closure: map every visible-transition label reachable
        from *marking* by firing zero or more invisible transitions to the
        marking at which it becomes enabled. Already-visited markings are
        skipped so tau-loops can't cause infinite recursion (the net's
        reachable-marking set is small and finite, so this terminates).
        """
        if _visited is None:
            _visited = set()
        key = tuple(sorted((p.name, count) for p, count in marking.items()))
        if key in _visited:
            return {}
        _visited.add(key)

        # semantics.enabled_transitions() returns a set, whose iteration
        # order depends on object id (memory layout) and therefore isn't
        # stable across runs. Sort by transition name so a fixed random
        # seed reproduces the exact same simulation every time.
        enabled = sorted(
            semantics.enabled_transitions(self.net, marking), key=lambda t: t.name
        )

        frontier: Dict[str, Marking] = {}
        for t in enabled:
            if t.label and t.label not in frontier:
                frontier[t.label] = marking

        for t in enabled:
            if t.label is None:
                next_marking = semantics.execute(t, self.net, marking)
                for label, m in self._visible_frontier(next_marking, _visited).items():
                    frontier.setdefault(label, m)

        return frontier

    def _weighted_choice(
        self, case_id: str, current_activity: Optional[str], labels: List[str],
        allow_end: bool = False,
    ) -> str:
        """
        Pick one of the net-enabled activities — or END_LABEL if *allow_end*
        (final marking tau-reachable) and the mined decision-point data says
        real cases stop here.

        Order of preference:
        1. END decision: P(end | decision point, visit bucket) from the
           mined dp table (mine_dp_probs.py). Without mined data the case
           never ends here (END gets no residual weight — ending must be
           evidenced, not accidental).
        2. "rules" mode: decision-point classifier on case attributes.
        3. dp table over the remaining labels, then activity-visit table
           ("visit" mode), then global BRANCHING_PROBS, with residual weight
           so every legal transition stays reachable.
        """
        options_n = len(labels) + (1 if allow_end else 0)
        if options_n == 1:
            return labels[0] if labels else END_LABEL

        dp_dist = self._dp_conditioned_probs(case_id, labels)

        if allow_end and dp_dist:
            p_end = dp_dist.get(END_LABEL, 0.0)
            if p_end and self._rng.random() < p_end:
                return END_LABEL

        if self.branching_mode == "rules" and len(labels) > 1:
            rules_choice = self._rules_weighted_choice(case_id, labels)
            if rules_choice is not None:
                return rules_choice

        preferred = None
        if dp_dist:
            preferred = {k: v for k, v in dp_dist.items() if k != END_LABEL}
        if not preferred:
            preferred = self._visit_conditioned_probs(case_id, current_activity)
        if not preferred:
            preferred = dict(BRANCHING_PROBS.get(current_activity, []))
        weights = [preferred.get(label, RESIDUAL_WEIGHT) for label in labels]

        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for label, w in zip(labels, weights):
            cumulative += w
            if r <= cumulative:
                return label
        return labels[-1]

    def _dp_conditioned_probs(self, case_id: str,
                              labels: List[str]) -> Optional[dict]:
        """
        Preferred branching source (A1 stage 2): the real choice distribution
        AT this exact decision point (mined by scripts/mine_dp_probs.py via
        replay), conditioned on the case's visit count of the decision point.
        Falls back (returns None) when the table is absent, the decision
        point wasn't mined, or a sparse visit bucket has no data and no
        "all" aggregate exists. Counting mirrors mine_dp_probs exactly:
        every evaluation of a multi-label frontier increments the counter.
        """
        if not self._dp_probs:
            return None
        key = " | ".join(sorted(labels))
        entry = self._dp_probs.get(key)
        if entry is None:
            return None
        visits = self._dp_visit_counts.setdefault(case_id, {})
        visits[key] = k = visits.get(key, 0) + 1
        bucket = str(k) if k < DP_VISIT_BUCKET_MAX else f"{DP_VISIT_BUCKET_MAX}+"
        return entry.get(bucket) or entry.get("all")

    def _visit_conditioned_probs(self, case_id: str,
                                 current_activity: Optional[str]) -> Optional[dict]:
        """
        Branching distribution conditioned on the current activity's visit
        count in this case (A1 termination fix). Returns None — meaning
        "fall back to the global BRANCHING_PROBS" — in "probs" mode, for the
        case's first activity, for activities without a mined table, and
        for buckets dropped as too sparse during extraction.

        The visit count comes from _repeat_counts, which on_activity_complete
        increments *before* the next-activity choice — so at decision time
        counts[current_activity] is exactly the 1-based visit number.
        """
        if current_activity is None or not self._branching_by_visit:
            return None
        by_visit = self._branching_by_visit.get(current_activity)
        if not by_visit:
            return None
        k = self._repeat_counts.get(case_id, {}).get(current_activity, 1)
        bucket = str(k) if k < VISIT_BUCKET_MAX else f"{VISIT_BUCKET_MAX}+"
        return by_visit.get(bucket)

    # ------------------------------------------------------------------
    # Data-based branching (Section 1.5 Advanced I)
    # ------------------------------------------------------------------

    def _ensure_decision_rules(self) -> None:
        """Load the joblib artifact from train_decision_rules.py the first
        time a rules-based decision is needed."""
        if self._decision_models is not None:
            return
        if not self._decision_rules_path:
            raise ValueError(
                "branching_mode='rules' requires decision_rules_path to a "
                "trained joblib artifact (run train_decision_rules.py)."
            )
        import joblib
        artifact = joblib.load(self._decision_rules_path)
        self._decision_models = artifact["models"]
        self._decision_encoders = artifact["encoders"]
        self._decision_feature_names = artifact["feature_names"]
        self._decision_unknown = artifact["sentinels"]["unknown"]

    def _rules_weighted_choice(self, case_id: str, labels: List[str]) -> Optional[str]:
        """
        Predict the branch via the decision-point classifier for this exact
        set of legal labels. Returns None (caller falls back to "probs") if
        this decision point has no trained model or the case has no sampled
        attributes yet.
        """
        self._ensure_decision_rules()
        model_info = self._decision_models.get(tuple(sorted(labels)))
        attrs = self._case_attrs.get(case_id)
        if model_info is None or attrs is None:
            return None

        features = self._build_decision_features(attrs)
        proba = model_info["tree"].predict_proba([features])[0]
        classes = model_info["classes"]

        r = self._rng.random()
        cumulative = 0.0
        for cls, p in zip(classes, proba):
            cumulative += p
            if r <= cumulative:
                return cls
        return classes[-1]

    def _build_decision_features(self, attrs: dict) -> List[float]:
        """Rebuild the feature vector in the artifact's feature order —
        mirrors train_decision_rules.py::build_matrix exactly."""
        values = {
            "application_type_enc": self._safe_encode(
                self._decision_encoders["application_type"], attrs["application_type"]),
            "loan_goal_enc": self._safe_encode(
                self._decision_encoders["loan_goal"], attrs["loan_goal"]),
            "requested_amount": attrs["requested_amount"],
            "has_offer": attrs["has_offer"],
            "credit_score": attrs["credit_score"],
            "offered_amount": attrs["offered_amount"],
            "number_of_terms": attrs["number_of_terms"],
            "monthly_cost": attrs["monthly_cost"],
            "first_withdrawal_amount": attrs["first_withdrawal_amount"],
        }
        return [float(values[name]) for name in self._decision_feature_names]

    def _safe_encode(self, encoder, value) -> int:
        value = str(value)
        if value not in set(encoder.classes_):
            value = self._decision_unknown
        return int(encoder.transform([value])[0])

    def _sample_categorical(self, options) -> str:
        r = self._rng.random()
        cumulative = 0.0
        for value, prob in options:
            cumulative += prob
            if r <= cumulative:
                return value
        return options[-1][0]

    def _sample_case_attributes(self) -> dict:
        """Sample the spawn attributes (Section 1.5 Advanced I) from the
        distributions learned from BPIC-17 (process.py). Offer attributes
        start at the "no offer yet" sentinel until O_Create Offer fires."""
        app_type = self._sample_categorical(APPLICATION_TYPE_PROBS)
        loan_goal = self._sample_categorical(LOAN_GOAL_GIVEN_APPLICATION_TYPE[app_type])
        dist_name, params = REQUESTED_AMOUNT_GIVEN_APPLICATION_TYPE[app_type]
        requested_amount = self._sample_scipy_like(dist_name, params)
        return {
            "application_type": app_type,
            "loan_goal": loan_goal,
            "requested_amount": requested_amount,
            "has_offer": 0,
            "credit_score": 0.0,
            "offered_amount": 0.0,
            "number_of_terms": 0.0,
            "monthly_cost": 0.0,
            "first_withdrawal_amount": 0.0,
        }

    def _sample_offer_attributes(self, case_id: str) -> None:
        """Fill in the offer attributes once O_Create Offer has fired."""
        attrs = self._case_attrs.get(case_id)
        if attrs is None:
            return
        attrs["has_offer"] = 1
        for name, key in [
            ("OfferedAmount", "offered_amount"),
            ("NumberOfTerms", "number_of_terms"),
            ("MonthlyCost", "monthly_cost"),
        ]:
            dist_name, params = OFFER_ATTRIBUTE_PARAMS[name]
            attrs[key] = self._sample_scipy_like(dist_name, params)
        for name, key in [
            ("FirstWithdrawalAmount", "first_withdrawal_amount"),
            ("CreditScore", "credit_score"),
        ]:
            spec = OFFER_ATTRIBUTE_ZERO_MASS_PARAMS[name]
            if self._rng.random() < spec["zero_prob"]:
                attrs[key] = 0.0
            else:
                dist_name, params = spec["dist"]
                attrs[key] = self._sample_scipy_like(dist_name, params)

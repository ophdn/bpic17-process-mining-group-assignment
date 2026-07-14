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
activities the net currently allows, BRANCHING_PROBS (see process.py) is
reused as a soft preference to pick which one happens, renormalised over
just the legal subset.

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
    scripts/test_advanced_process_model.py).
"""

from datetime import datetime
from typing import Dict, List, Optional

import pm4py
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking, PetriNet

from ..core.events import SimEvent, EventType
from .process import BRANCHING_PROBS, ProcessComponent

# Residual weight given to a legal next activity that BRANCHING_PROBS has
# no entry for at this point (keeps every enabled transition reachable
# instead of only ever following historically-observed edges).
RESIDUAL_WEIGHT = 0.01

# Safety guard against pathological tau-loops in a malformed BPMN.
MAX_TAU_STEPS = 1000

# Loop-guards for the *activity* loop (as opposed to the tau-loop guard
# above). Unlike process.py's MAX_ACTIVITY_REPEATS=15 — tuned for a flat
# probability graph with no legality check — these only need to bound
# worst-case runtime: legality is already enforced by the net itself, and
# BPIC-17 genuinely contains cases that cycle the offer sub-loop (create/
# send/return an offer) 20+ times. Set generously so real loop behaviour
# isn't guillotined before it exits on its own.
MAX_ACTIVITY_REPEATS = 60
MAX_TOTAL_ACTIVITIES = 400


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
    ):
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

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_activity_start(self, engine, event: SimEvent) -> None:
        case_id = event.case_id

        if event.activity == "__PROCESS_START__":
            self._repeat_counts[case_id] = {}
            self._ctx[case_id] = {
                "start_t": engine.now,
                "position": 0,
                "prev_act": None,
            }
            self._markings[case_id] = Marking(self.im)
            first_activity = self._advance_to_next_visible(case_id, current_activity=None)
            if first_activity is None:
                # Malformed net: nothing enabled from the initial marking.
                self._markings.pop(case_id, None)
                self._repeat_counts.pop(case_id, None)
                self._ctx.pop(case_id, None)
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
            self._resources.release(engine, event.resource)

        counts = self._repeat_counts.get(case_id, {})
        counts[activity] = counts.get(activity, 0) + 1
        self._repeat_counts[case_id] = counts

        self._fire_activity(case_id, activity)

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
        if counts.get(activity, 0) >= MAX_ACTIVITY_REPEATS:
            return True
        if sum(counts.values()) > MAX_TOTAL_ACTIVITIES:
            return True
        return False

    def _end_case(self, engine, case_id: str) -> None:
        self._markings.pop(case_id, None)
        self._repeat_counts.pop(case_id, None)
        self._ctx.pop(case_id, None)
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
        Fire enabled invisible (tau) transitions until a visible one is
        enabled, then pick among the enabled visible activities.
        Returns None if the case has no enabled transition left at all
        (dead marking).
        """
        marking = self._markings[case_id]
        for _ in range(MAX_TAU_STEPS):
            # semantics.enabled_transitions() returns a set, whose iteration
            # order depends on object id (memory layout) and therefore isn't
            # stable across runs. Sort by transition name so a fixed random
            # seed reproduces the exact same simulation every time.
            enabled = sorted(
                semantics.enabled_transitions(self.net, marking), key=lambda t: t.name
            )
            if not enabled:
                return None

            visible = [t for t in enabled if t.label]
            if visible:
                return self._weighted_choice(current_activity, visible)

            t = self._rng.choice(enabled)  # all remaining are invisible (label is None)
            marking = semantics.execute(t, self.net, marking)
            self._markings[case_id] = marking

        return None

    def _weighted_choice(
        self, current_activity: Optional[str], visible_transitions: List[PetriNet.Transition]
    ) -> str:
        """
        Pick one of the net-enabled activities, weighted by the empirical
        branching probabilities for *current_activity* where available,
        with a small residual weight for options BRANCHING_PROBS doesn't
        cover so every legal transition stays reachable.
        """
        preferred = dict(BRANCHING_PROBS.get(current_activity, []))
        weights = [preferred.get(t.label, RESIDUAL_WEIGHT) for t in visible_transitions]

        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for t, w in zip(visible_transitions, weights):
            cumulative += w
            if r <= cumulative:
                return t.label
        return visible_transitions[-1].label

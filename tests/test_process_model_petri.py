from __future__ import annotations

from datetime import datetime

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from simulation.components.petri_process import PetriNetProcessComponent
from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent


def _component_with_linear_net(enforce_terminal_outcomes: bool = False) -> PetriNetProcessComponent:
    component = object.__new__(PetriNetProcessComponent)
    component._rng = None
    component._active = False
    component._resources = None
    component._case_attributes = None
    component._repeat_counts = {"c1": {}}
    component._ctx = {"c1": {"start_t": 0.0, "position": 0, "prev_act": None, "attrs": {}}}
    component._witem_seq = {}
    component._witem = {}
    component._markings = {}
    component._fm_reach_cache = {}
    component.branching_mode = "probs"
    component.enforce_terminal_outcomes = enforce_terminal_outcomes
    component._branching_by_visit = {}
    component._dp_probs = {}
    component._dp_visit_counts = {}
    component._decision_rules_path = None
    component._decision_models = None
    component._decision_encoders = None
    component._decision_feature_names = None
    component._decision_unknown = "__UNKNOWN__"
    component._case_attrs = {}
    component._advance_reasons = {}
    component._debug = {
        "allow_end_opportunities": 0,
        "allow_end_without_dp": 0,
        "end_label_choices": 0,
        "end_reasons": {
            "final_marking": 0,
            "end_label": 0,
            "terminal_outcome": 0,
            "loop_guard": 0,
            "dead_marking": 0,
            "terminal_continuation_end": 0,
            "terminal_allow_end_fallback": 0,
        },
    }
    component._basic_adjacency = {}
    component._lp = None
    component._crn = False
    component._seed = 1

    net = PetriNet("linear")
    p0, p1, p2 = (PetriNet.Place(name) for name in ("p0", "p1", "p2"))
    t1 = PetriNet.Transition("t_pending", "A_Pending")
    t2 = PetriNet.Transition("t_validating", "A_Validating")
    net.places.update({p0, p1, p2})
    net.transitions.update({t1, t2})
    petri_utils.add_arc_from_to(p0, t1, net)
    petri_utils.add_arc_from_to(t1, p1, net)
    petri_utils.add_arc_from_to(p1, t2, net)
    petri_utils.add_arc_from_to(t2, p2, net)
    component.net = net
    component.im = Marking({p0: 1})
    component.fm = Marking({p2: 1})
    component._markings["c1"] = Marking({p0: 1})
    return component


def test_nonfinal_outcome_activity_does_not_complete_case_early_when_not_enforced():
    """With enforce_terminal_outcomes=False, only final_marking/__END__/loop_guard
    end a case; a non-final outcome activity like A_Pending just continues."""
    engine = SimulationEngine(sim_duration=10, start_datetime=datetime(2016, 1, 1))
    component = _component_with_linear_net(enforce_terminal_outcomes=False)

    component.on_activity_complete(
        engine,
        SimEvent(timestamp=0, event_type=EventType.ACTIVITY_COMPLETE,
                 case_id="c1", activity="A_Pending"),
    )

    scheduled = [(event.event_type, event.activity) for event in engine._queue]
    assert (EventType.CASE_COMPLETE, None) not in scheduled
    assert (EventType.ACTIVITY_REQUEST, "A_Validating") in scheduled
    assert component.debug_stats()["end_reasons"]["end_label"] == 0
    assert component.debug_stats()["end_reasons"]["terminal_outcome"] == 0


def test_terminal_outcome_activity_completes_case_when_enforced():
    """With enforce_terminal_outcomes=True (the A1 fix / default), firing an
    outcome activity like A_Pending force-ends the case immediately."""
    engine = SimulationEngine(sim_duration=10, start_datetime=datetime(2016, 1, 1))
    component = _component_with_linear_net(enforce_terminal_outcomes=True)

    component.on_activity_complete(
        engine,
        SimEvent(timestamp=0, event_type=EventType.ACTIVITY_COMPLETE,
                 case_id="c1", activity="A_Pending"),
    )

    scheduled = [(event.event_type, event.activity) for event in engine._queue]
    assert (EventType.CASE_COMPLETE, None) in scheduled
    assert (EventType.ACTIVITY_REQUEST, "A_Validating") not in scheduled
    assert component.debug_stats()["end_reasons"]["terminal_outcome"] == 1


def _component_with_forced_followup_net(enforce_terminal_outcomes: bool = True) -> PetriNetProcessComponent:
    """Same shape as _component_with_linear_net, but the outcome activity
    (A_Cancelled) is mapped in FORCED_TERMINAL_FOLLOWUP to a deterministic
    single next step (O_Cancelled) before the case ends."""
    component = _component_with_linear_net(enforce_terminal_outcomes)
    net = component.net
    p0 = next(p for p in net.places if p.name == "p0")
    p1 = next(p for p in net.places if p.name == "p1")
    p2 = next(p for p in net.places if p.name == "p2")
    for t in list(net.transitions):
        t.label = {"t_pending": "A_Cancelled", "t_validating": "O_Cancelled"}[t.name]
    component._markings["c1"] = Marking({p0: 1})
    return component


def test_terminal_outcome_fires_forced_followup_before_ending():
    """A_Cancelled is mapped to a forced O_Cancelled follow-up
    (FORCED_TERMINAL_FOLLOWUP): firing it should update the marking, log
    O_Cancelled directly (not re-enter on_activity_complete as a new
    choice), and still end the case via the terminal_outcome reason."""
    engine = SimulationEngine(sim_duration=10, start_datetime=datetime(2016, 1, 1))
    component = _component_with_forced_followup_net(enforce_terminal_outcomes=True)

    component.on_activity_complete(
        engine,
        SimEvent(timestamp=0, event_type=EventType.ACTIVITY_COMPLETE,
                 case_id="c1", activity="A_Cancelled"),
    )

    scheduled = [(event.event_type, event.activity) for event in engine._queue]
    assert (EventType.CASE_COMPLETE, None) in scheduled
    logged_activities = [row["concept:name"] for row in engine.logger._rows]
    assert logged_activities == ["O_Cancelled"]
    assert component.debug_stats()["end_reasons"]["terminal_outcome"] == 1


def test_case_completes_when_final_marking_is_reached():
    engine = SimulationEngine(sim_duration=10, start_datetime=datetime(2016, 1, 1))
    component = _component_with_linear_net()
    component._markings["c1"] = Marking({next(place for place in component.net.places if place.name == "p1"): 1})
    component._repeat_counts["c1"] = {}

    component.on_activity_complete(
        engine,
        SimEvent(timestamp=0, event_type=EventType.ACTIVITY_COMPLETE,
                 case_id="c1", activity="A_Validating"),
    )

    scheduled = [(event.event_type, event.activity) for event in engine._queue]
    assert (EventType.CASE_COMPLETE, None) in scheduled
    assert component.debug_stats()["end_reasons"]["final_marking"] == 1

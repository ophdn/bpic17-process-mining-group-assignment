"""Part II advanced policies: D1 (Park & Song) and D2 (Kunkler &
Rinderle-Ma), both implemented as one epoch-assignment mechanism in
ResourceComponent (see resource.py constructor docstrings).

Pinned behaviours:
- D1: a phantom (predicted next task of an in-service case) can RESERVE
  a resource — that resource idles the epoch even though a real item is
  waiting; only real items ever _begin. Case completion drops the
  bookkeeping so finished cases predict nothing.
- D2: an item whose every compatible free resource costs more than
  delta x its own mean duration defers (stays queued) despite free
  capacity; with a huge delta it never defers.
- Both: permission-safe, deterministic/reproducible, mutually exclusive
  with the other allocation modes.
"""

from datetime import datetime

import pytest

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent
from simulation.policies_advanced import NextActivityPredictor

SEED = 42
START = datetime(2016, 1, 1)
ONE_DAY = 24 * 3600


# ---------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------

def test_predictor_argmax_and_terminals():
    p = NextActivityPredictor()
    # A_Submitted -> W_Handle leads with probability 1.0 in BRANCHING_PROBS.
    assert p.predict("A_Submitted") == ("W_Handle leads", 1.0)
    # Terminal activities plan no phantom successor.
    assert p.predict("W_Validate application") is None
    # Unknown activity: no prediction rather than a crash.
    assert p.predict("__PROCESS_START__") is None


def test_predictor_is_deterministic():
    a, b = NextActivityPredictor(), NextActivityPredictor()
    for act in ("A_Submitted", "A_Concept", "O_Created", "A_Accepted"):
        assert a.predict(act) == b.predict(act)


# ---------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"parksong": True, "piled": True},
    {"parksong": True, "batching_k": 5},
    {"parksong": True, "pull": "spt"},
    {"parksong": True, "krm_delta": 1.0},
    {"krm_delta": 1.0, "piled": True},
    {"krm_delta": 0.0},
])
def test_mode_guards(kwargs):
    with pytest.raises(ValueError):
        ResourceComponent(seed=SEED, **kwargs)


# ---------------------------------------------------------------------
# D1: strategic idling on a constructed epoch
# ---------------------------------------------------------------------

def _spy_begins(rc):
    begun = []
    orig = rc._begin

    def spy(engine, request, resource):
        begun.append((request.case_id, resource))
        return orig(engine, request, resource)

    rc._begin = spy
    return begun


class _StubDurations:
    """Resource-differentiated expected durations for constructed epochs.

    The spread gate (see _epoch_flush) drops phantoms whose permitted
    costs show no spread — correct with the flat distribution-mean
    fallback, but it means reservations only activate with a
    resource-aware duration model. This stub plays that role in tests.
    """

    def __init__(self, costs, default=1000.0):
        self._costs = costs  # (activity, resource) -> seconds
        self._default = default

    def expected_duration(self, activity, resource=None, context=None):
        return self._costs.get((activity, resource), self._default)


def test_parksong_phantom_reserves_best_fit_and_real_work_shifts():
    """Two free resources; one real item (equal cost on both); one
    in-service case whose imminent successor is far cheaper on r1 than
    r2. The solver must reserve r1 for the phantom (strategic idling)
    and start the real item on r2 — real work is displaced off the
    phantom's best-fit resource, not starved."""
    rc = ResourceComponent(seed=SEED, parksong=True, capacity_per_resource=1)
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)

    real_act, active_act = "W_Complete application", "A_Submitted"
    pred_act = "W_Handle leads"  # A_Submitted's certain successor
    both = [x for x in rc.permissions.resources()
            if rc.permissions.permits(x, real_act)
            and rc.permissions.permits(x, pred_act)]
    if len(both) < 2:
        pytest.skip("need two resources permitted for both activities")
    r1, r2 = both[0], both[1]
    rc._excluded = {x for x in rc.permissions.resources() if x not in (r1, r2)}

    rc._duration_model = _StubDurations({
        (real_act, r1): 1000.0, (real_act, r2): 1000.0,
        (pred_act, r1): 10.0, (pred_act, r2): 500.0,
        # eta path: expected remaining duration of the in-service activity
        (active_act, None): 60.0,
    })

    rc._active_cases["case_active"] = (active_act, {}, 0.0)
    rc._waiting.append(SimEvent(
        timestamp=0.0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_real", activity=real_act, resource=None,
    ))
    begun = _spy_begins(rc)
    rc._epoch_flush(eng)
    assert begun == [("case_real", r2)], (
        "real item should start on r2 while r1 is reserved for the phantom")
    assert rc._busy.get(r1, 0) == 0, "r1 must idle (reserved for predicted work)"


def test_parksong_flat_costs_reserve_nothing():
    """With resource-independent costs (the artifact-less fallback) a
    reservation can never pay — the spread gate must drop every phantom
    and the epoch must behave like a plain assignment."""
    rc = ResourceComponent(seed=SEED, parksong=True, capacity_per_resource=1)
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    real_act, active_act = "W_Complete application", "A_Submitted"
    r = next((x for x in rc.permissions.resources()
              if rc.permissions.permits(x, real_act)), None)
    if r is None:
        pytest.skip("no resource permitted")
    rc._excluded = {x for x in rc.permissions.resources() if x != r}
    rc._active_cases["case_active"] = (active_act, {}, 0.0)
    rc._waiting.append(SimEvent(
        timestamp=0.0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_real", activity=real_act, resource=None,
    ))
    begun = _spy_begins(rc)
    rc._epoch_flush(eng)
    assert [c for c, _ in begun] == ["case_real"]


def test_parksong_distant_successor_plans_no_phantom():
    """The lookahead gate: a case whose current activity won't finish for
    days must NOT reserve anything — this exact over-idling collapsed
    completions 158 -> 15 before the gate existed."""
    rc = ResourceComponent(seed=SEED, parksong=True, capacity_per_resource=1)
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    real_act = "W_Handle leads"
    long_act = "W_Assess potential fraud"  # mean ~333k s >> lookahead
    r = next((x for x in rc.permissions.resources()
              if rc.permissions.permits(x, real_act)), None)
    if r is None:
        pytest.skip("no resource permitted")
    rc._excluded = {x for x in rc.permissions.resources() if x != r}
    rc._active_cases["case_slow"] = (long_act, {}, 0.0)  # just began
    rc._waiting.append(SimEvent(
        timestamp=0.0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_real", activity=real_act, resource=None,
    ))
    begun = _spy_begins(rc)
    rc._epoch_flush(eng)
    # No phantom in the epoch -> the real item starts immediately.
    assert [c for c, _ in begun] == ["case_real"]


def test_parksong_completed_case_predicts_nothing():
    rc = ResourceComponent(seed=SEED, parksong=True)
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    rc._active_cases["case_x"] = ("A_Submitted", {})
    rc.on_case_complete(eng, SimEvent(
        timestamp=0, priority=20, event_type=EventType.CASE_COMPLETE,
        case_id="case_x",
    ))
    assert "case_x" not in rc._active_cases


# ---------------------------------------------------------------------
# D2: dummy-cost deferral on a constructed epoch
# ---------------------------------------------------------------------

def _krm_epoch(delta, activity="W_Assess potential fraud"):
    rc = ResourceComponent(seed=SEED, krm_delta=delta, capacity_per_resource=1)
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    r = next((x for x in rc.permissions.resources()
              if rc.permissions.permits(x, activity)), None)
    if r is None:
        pytest.skip(f"no resource permitted for {activity}")
    rc._excluded = {x for x in rc.permissions.resources() if x != r}
    rc._waiting.append(SimEvent(
        timestamp=0.0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_1", activity=activity, resource=None,
    ))
    begun = _spy_begins(rc)
    rc._epoch_flush(eng)
    return begun, rc


def test_krm_tiny_delta_defers_despite_free_capacity():
    # Without a trained artifact, every resource costs the item's mean
    # duration; the dummy costs delta x that mean. delta << 1 => dummy
    # wins => defer.
    begun, rc = _krm_epoch(delta=0.01)
    assert begun == []
    assert len(rc._waiting) == 1


def test_krm_large_delta_never_defers():
    begun, rc = _krm_epoch(delta=100.0)
    assert len(begun) == 1
    assert len(rc._waiting) == 0


# ---------------------------------------------------------------------
# Integration: full runs
# ---------------------------------------------------------------------

def _run(**rc_kwargs):
    eng = SimulationEngine(sim_duration=ONE_DAY, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(
        capacity_per_resource=1, seed=SEED, **rc_kwargs)
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START,
        resource_component=resources, crn=True)
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    return eng, resources


@pytest.mark.parametrize("kwargs", [
    {"parksong": True},
    {"krm_delta": 1.0},
])
def test_advanced_policy_full_run(kwargs):
    eng, resources = _run(**kwargs)
    assert eng.stats["cases_completed"] > 0
    perms = resources.permissions
    checked = 0
    for row in eng.logger._rows:
        act, res = row["concept:name"], row["org:resource"]
        if not res or act.startswith("__"):
            continue
        checked += 1
        assert perms.permits(res, act)
    assert checked > 0
    # Reproducible.
    eng2, _ = _run(**kwargs)
    assert eng.logger._rows == eng2.logger._rows

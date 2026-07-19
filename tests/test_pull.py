"""Pull-side selection disciplines (push-vs-pull evaluation, Part II).

When ``pull`` is set on ResourceComponent, a freed resource picks its
preferred waiting item ("spt": shortest expected duration; "laf":
longest-active case first, lecture deck 04 F31) instead of the system's
FIFO first-permitted scan. Constraints these tests pin down:

- NOT a FIFO relabel: on a constructed queue the pull rules pick a
  different item than FIFO would.
- Permission-safe: a pull pick is always permitted.
- Deterministic: no RNG draws; identical runs are identical.
- Mutually exclusive with piled / batching_k.
"""

from datetime import datetime

import pytest

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent

SEED = 42
START = datetime(2016, 1, 1)
ONE_DAY = 24 * 3600

# From process.py's fitted models: W_Handle leads has a tiny mean duration
# (lognorm scale 128s), W_Assess potential fraud a huge one (gamma mean
# ~333k s). The distribution-mean fallback of ExpectedDurationModel makes
# this ordering deterministic without any trained artifact.
SHORT_ACT = "W_Handle leads"
LONG_ACT = "W_Assess potential fraud"


def _request(case_id, activity, ts):
    return SimEvent(
        timestamp=ts, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id=case_id, activity=activity, resource=None,
    )


def _find_permitted_resource(rc, *activities):
    """A resource the default permission model allows for all *activities*."""
    for r in rc.permissions.resources():
        if all(rc.permissions.permits(r, a) for a in activities):
            return r
    pytest.skip(f"no resource permitted for all of {activities}")


# ---------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------

def test_pull_rejects_unknown_rule():
    with pytest.raises(ValueError):
        ResourceComponent(seed=SEED, pull="fifo")


@pytest.mark.parametrize("kwargs", [
    {"piled": True},
    {"batching_k": 5},
])
def test_pull_mutually_exclusive(kwargs):
    with pytest.raises(ValueError):
        ResourceComponent(seed=SEED, pull="spt", **kwargs)


# ---------------------------------------------------------------------
# Selection behaviour on a constructed queue
# ---------------------------------------------------------------------

def _release_and_capture(rc, resource):
    """Free *resource* once and return the case_id it picked (via _begin)."""
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    picked = []
    orig = rc._begin

    def spy(engine, request, res):
        picked.append(request.case_id)
        return orig(engine, request, res)

    rc._begin = spy
    rc._busy[resource] = 1  # busy, so the release frees a slot
    rc.on_resource_available(eng, SimEvent(
        timestamp=0, priority=1, event_type=EventType.RESOURCE_AVAILABLE,
        resource=resource,
    ))
    rc._begin = orig
    return picked


def test_spt_pull_prefers_short_expected_duration_over_fifo():
    rc = ResourceComponent(seed=SEED, pull="spt", capacity_per_resource=1)
    r = _find_permitted_resource(rc, SHORT_ACT, LONG_ACT)
    # FIFO order: the LONG task queued first. SPT must pick the SHORT one.
    rc._waiting.extend([
        _request("case_long", LONG_ACT, ts=0.0),
        _request("case_short", SHORT_ACT, ts=1.0),
    ])
    assert _release_and_capture(rc, r) == ["case_short"]


def test_spt_reuses_deterministic_duration_predictions():
    rc = ResourceComponent(seed=SEED, pull="spt", capacity_per_resource=1)
    r = _find_permitted_resource(rc, SHORT_ACT, LONG_ACT)
    rc._waiting.extend([
        _request("case_long", LONG_ACT, ts=0.0),
        _request("case_short", SHORT_ACT, ts=1.0),
    ])
    calls = []
    original = rc._duration_model.expected_duration

    def counted(activity, resource):
        calls.append((activity, resource))
        return original(activity, resource)

    rc._duration_model.expected_duration = counted
    eng = SimulationEngine(sim_duration=10, start_datetime=START, verbose=False)
    assert rc._pull_best_index(eng, r) == 1
    assert rc._pull_best_index(eng, r) == 1
    assert calls == [(LONG_ACT, r), (SHORT_ACT, r)]


def test_laf_pull_prefers_oldest_case_over_fifo():
    rc = ResourceComponent(seed=SEED, pull="laf", capacity_per_resource=1)
    r = _find_permitted_resource(rc, SHORT_ACT, LONG_ACT)
    # case_old entered the system at t=0 but its CURRENT request queued
    # last; case_young's request has waited longer. FIFO picks case_young,
    # LAF must pick case_old.
    rc._case_first_seen["case_old"] = 0.0
    rc._case_first_seen["case_young"] = 500.0
    rc._waiting.extend([
        _request("case_young", SHORT_ACT, ts=600.0),
        _request("case_old", SHORT_ACT, ts=900.0),
    ])
    assert _release_and_capture(rc, r) == ["case_old"]


def test_pull_pick_is_always_permitted():
    rc = ResourceComponent(seed=SEED, pull="spt", capacity_per_resource=1)
    r = _find_permitted_resource(rc, SHORT_ACT)
    # Find an activity r may NOT perform, queue it as the cheap-looking bait.
    forbidden = None
    for act in (LONG_ACT, "A_Create Application", "O_Create Offer",
                "W_Validate application"):
        if not rc.permissions.permits(r, act):
            forbidden = act
            break
    if forbidden is None:
        pytest.skip("resource is permitted everything; cannot construct bait")
    rc._waiting.extend([
        _request("case_forbidden", forbidden, ts=0.0),
        _request("case_ok", SHORT_ACT, ts=1.0),
    ])
    assert _release_and_capture(rc, r) == ["case_ok"]


# ---------------------------------------------------------------------
# Integration: full runs
# ---------------------------------------------------------------------

def _run(pull):
    eng = SimulationEngine(sim_duration=ONE_DAY, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(
        capacity_per_resource=1, seed=SEED, pull=pull)
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START,
        resource_component=resources, crn=True)
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    return eng.logger._rows


@pytest.mark.parametrize("pull", ["spt", "laf"])
def test_pull_run_reproducible(pull):
    assert _run(pull) == _run(pull)


def test_pull_differs_from_fifo_baseline():
    """Under contention (capacity=1) the pull disciplines must actually
    change the trajectory vs. the FIFO drain — otherwise they are the
    relabel this feature explicitly exists to avoid."""
    baseline = _run(None)
    assert _run("spt") != baseline
    assert _run("laf") != baseline

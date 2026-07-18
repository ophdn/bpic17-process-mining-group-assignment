"""Section 1.8 push-selection policies (Final Task 1 simple heuristics).

R-RRA (Round Robin, Russell et al. pattern 16) and R-SHQ (Shortest
Queue, pattern 17) behind the AllocationPolicy seam. Both are
deterministic by design (no RNG draws), so enabling them cannot perturb
any other component's random stream — asserted here, because silently
consuming a draw would break CRN pairing across policies.
"""

from collections import Counter
from datetime import datetime

import pytest

from simulation.core.engine import SimulationEngine
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent
from simulation.policies import (
    AllocationState,
    RandomPolicy,
    RoundRobinPolicy,
    ShortestQueuePolicy,
)

SEED = 42
START = datetime(2016, 1, 1)
ONE_DAY = 24 * 3600


# ---------------------------------------------------------------------
# Unit level: selection behaviour on constructed states
# ---------------------------------------------------------------------

def test_round_robin_rotates_within_activity():
    p = RoundRobinPolicy()
    state = AllocationState(busy={}, capacity=1)
    picks = [p.select("A", ["r1", "r2", "r3"], state) for _ in range(6)]
    assert picks == ["r1", "r2", "r3", "r1", "r2", "r3"]


def test_round_robin_cursor_is_per_activity():
    p = RoundRobinPolicy()
    state = AllocationState(busy={}, capacity=1)
    assert p.select("A", ["r1", "r2"], state) == "r1"
    # B's rotation starts fresh — A's cursor must not leak into B.
    assert p.select("B", ["r1", "r2"], state) == "r1"
    assert p.select("A", ["r1", "r2"], state) == "r2"


def test_round_robin_survives_shrinking_candidate_list():
    p = RoundRobinPolicy()
    state = AllocationState(busy={}, capacity=1)
    for _ in range(5):
        p.select("A", ["r1", "r2", "r3"], state)
    # Someone went off shift: a shorter list must still be a valid pick.
    pick = p.select("A", ["r1"], state)
    assert pick == "r1"


def test_shortest_queue_picks_least_busy():
    p = ShortestQueuePolicy()
    state = AllocationState(busy={"r1": 3, "r2": 1, "r3": 2}, capacity=4)
    assert p.select("A", ["r1", "r2", "r3"], state) == "r2"


def test_shortest_queue_tie_breaks_by_candidate_order():
    p = ShortestQueuePolicy()
    state = AllocationState(busy={"r1": 1, "r2": 1}, capacity=2)
    # Deterministic first-of-the-minima, not a random pick.
    assert p.select("A", ["r2", "r1"], state) == "r2"
    assert p.select("A", ["r1", "r2"], state) == "r1"


def test_shortest_queue_uses_cumulative_load_at_unit_capacity():
    p = ShortestQueuePolicy()
    state = AllocationState(
        busy={"r1": 0, "r2": 0},
        capacity=1,
        allocations={"r1": 8, "r2": 3},
    )
    assert p.select("A", ["r1", "r2"], state) == "r2"


def test_shortest_queue_never_picks_busier_than_minimum():
    p = ShortestQueuePolicy()
    state = AllocationState(busy={"r1": 5, "r2": 0, "r3": 5}, capacity=6)
    for _ in range(10):
        assert p.select("A", ["r1", "r2", "r3"], state) == "r2"


def test_policies_consume_no_rng():
    """Determinism guarantee: neither policy owns or touches an RNG.

    RandomPolicy has ``_rng``; the deterministic policies must not, and
    repeated calls on identical state must return identical picks.
    """
    rr, sq = RoundRobinPolicy(), ShortestQueuePolicy()
    assert not hasattr(rr, "_rng") and not hasattr(sq, "_rng")
    state = AllocationState(busy={"r1": 2, "r2": 1}, capacity=3)
    assert all(sq.select("A", ["r1", "r2"], state) == "r2" for _ in range(5))


# ---------------------------------------------------------------------
# Integration level: full 1-day simulation per policy
# ---------------------------------------------------------------------

def _run(policy):
    eng = SimulationEngine(sim_duration=ONE_DAY, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(
        capacity_per_resource=3, seed=SEED, policy=policy)
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START,
        resource_component=resources)
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    return eng, resources


@pytest.mark.parametrize("make_policy", [RoundRobinPolicy, ShortestQueuePolicy])
def test_simulation_runs_and_respects_permissions(make_policy):
    eng, resources = _run(make_policy())
    perms = resources.permissions
    checked = 0
    for row in eng.logger._rows:
        act, res = row["concept:name"], row["org:resource"]
        if not res or act.startswith("__"):
            continue
        checked += 1
        assert perms.permits(res, act), f"{res} not permitted for {act}"
    assert checked > 0


@pytest.mark.parametrize("make_policy", [RoundRobinPolicy, ShortestQueuePolicy])
def test_simulation_reproducible(make_policy):
    rows_a = _run(make_policy())[0].logger._rows
    rows_b = _run(make_policy())[0].logger._rows
    assert rows_a == rows_b


def test_round_robin_spreads_work():
    """Turn-taking must not starve anyone: over a day, no single resource
    hoards an activity that several resources share."""
    eng, _ = _run(RoundRobinPolicy())
    by_activity = {}
    for row in eng.logger._rows:
        if row["lifecycle:transition"] != "complete":
            continue
        act, res = row["concept:name"], row["org:resource"]
        if not res or act.startswith("__"):
            continue
        by_activity.setdefault(act, Counter())[res] += 1
    # For every activity executed >= 10 times by >= 2 distinct resources,
    # the busiest executor must not have done ALL of them.
    spread_checked = 0
    for act, counter in by_activity.items():
        total = sum(counter.values())
        if total >= 10 and len(counter) >= 2:
            spread_checked += 1
            assert counter.most_common(1)[0][1] < total
    assert spread_checked > 0, "no activity had enough volume to check spread"

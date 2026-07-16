"""
test_kbatching.py
===================
Verifies k-Batching (Zeng & Zhao, Optimization 1.1 / Final Task 1):
work items queue and are released in batches of k, solved as a
parallel-machines assignment problem against expected processing time.

1. Never allocates immediately: with batching_k set, a request always
   queues even when a qualified resource is free (the defining difference
   from R-RBA/R-DE and from Piled Execution — this is what "batch
   allocation, not on-arrival allocation" means).
2. R-RBA still holds: every resource a batch assigns to is permitted for
   that activity (the cost matrix's permission gate must never leak an
   invalid pair through the assignment solver).
3. Max-wait safety valve: under very low load (batch never reaches k),
   waiting items still eventually flush instead of starving forever.
4. Mutual exclusivity: batching_k + piled=True must raise.
5. Reproducibility: same seed -> identical event log.
6. k=1 sanity: still queues (see 1), but should show reasonable
   completion behaviour, not obviously broken.
7. Calendar + k-Batching together terminate promptly (regression test):
   the shift-wake (Section 1.6) and the k-Batching max-wait valve are two
   independent self-scheduled "resource=None" wake-up mechanisms. An
   earlier version reset BOTH mechanisms' "already armed" flag whenever
   *either* one fired, which re-armed whichever one was still legitimately
   pending on its own already-scheduled event -- duplicating it. Duplicates
   compounded every time the other wake fired too, producing thousands of
   same-timestamp events and multi-minute hangs on runs that use a
   calendar (found via cProfile: _next_shift_open dominated runtime).
   Fixed by tagging each wake's event with which mechanism armed it
   (_SHIFT_WAKE / _BATCH_WAKE) so only the one that actually fired resets.
   This test is a real-load, calendar-enabled run with a tight wall-clock
   budget, specifically to catch a regression of that bug.

Usage:
    cd <repo-root>
    PYTHONPATH=. .venv/bin/python scripts/test_kbatching.py
"""

import time
from datetime import datetime
from pathlib import Path

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent, RESOURCE_PERMISSIONS
from analysis.availability import YearlyAvailability

SEED = 42
START = datetime(2016, 1, 1)
ONE_WEEK = 7 * 24 * 3600
CAPACITY = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
AVAILABILITY_MODEL_PATH = REPO_ROOT / "models" / "availability_model.json"


def run(k, days=7):
    eng = SimulationEngine(sim_duration=days * 86400, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(capacity_per_resource=CAPACITY, seed=SEED, batching_k=k)
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START, resource_component=resources,
    )
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    return eng, resources


def main():
    print("=" * 60)
    print("Test 1: a request never allocates immediately under k-Batching")
    print("=" * 60)
    # A single-case run: the very first request has every resource free,
    # yet must still queue rather than start immediately (batch trigger:
    # k=5, only 1 item waiting -> no flush until max-wait).
    eng = SimulationEngine(sim_duration=3600, start_datetime=START, verbose=False)
    resources = ResourceComponent(capacity_per_resource=3, seed=SEED, batching_k=5)
    eng.register(resources)
    eng.schedule(SimEvent(
        timestamp=0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_probe", activity="W_Validate application", resource=None,
    ))
    eng.run()
    assert len(resources._waiting) == 1, (
        "request should still be queued (only 1 of k=5 items waiting, "
        "no max-wait elapsed) -- got it allocated immediately instead"
    )
    print("  PASS -- single request stayed queued, not allocated on arrival\n")

    print("=" * 60)
    print("Test 2: R-RBA held (every batch assignment is a permitted pair)")
    print("=" * 60)
    for k in (1, 2, 5, 10, 20):
        eng, res = run(k)
        logged = eng.logger._rows
        violations = 0
        checked = 0
        for ev in logged:
            activity, resource = ev.get("concept:name"), ev.get("org:resource")
            if activity == "__PROCESS_START__" or not resource:
                continue
            checked += 1
            if activity not in RESOURCE_PERMISSIONS.get(resource, set()):
                violations += 1
        print(f"  k={k:>2}: checked={checked} violations={violations} "
              f"completed={eng.stats['cases_completed']}")
        assert violations == 0, f"k={k}: batch assigned an unpermitted (resource, activity) pair!"
    print("  PASS -- no permission violations at any k\n")

    print("=" * 60)
    print("Test 3: max-wait safety valve self-schedules and flushes")
    print("=" * 60)
    # A single request, a k that will never be reached by item count alone,
    # and NOTHING else scheduled -- the only thing that can advance the
    # engine clock past the valve threshold is the component's own
    # self-armed wake-up (_arm_batch_wake). If that mechanism didn't exist,
    # the queue would engine.run() straight to "queue empty" without ever
    # rechecking the valve, since no other event follows the first request.
    eng = SimulationEngine(sim_duration=2 * 3600, start_datetime=START, verbose=False)
    resources = ResourceComponent(
        capacity_per_resource=3, seed=SEED, batching_k=1000,
        batching_max_wait_seconds=3600,  # 1h valve
    )
    eng.register(resources)
    eng.schedule(SimEvent(
        timestamp=0, priority=5, event_type=EventType.ACTIVITY_REQUEST,
        case_id="case_valve", activity="W_Validate application", resource=None,
    ))
    eng.run()
    assert len(resources._waiting) == 0, (
        "max-wait valve should have self-scheduled a wake-up and flushed "
        "the item once it waited past batching_max_wait_seconds, even "
        "though k=1000 was never reached and no other event was scheduled"
    )
    print("  PASS -- self-armed wake-up fired and flushed the item, despite k not being reached\n")

    print("=" * 60)
    print("Test 4: batching_k + piled=True is rejected")
    print("=" * 60)
    try:
        ResourceComponent(capacity_per_resource=3, seed=SEED, batching_k=5, piled=True)
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass
    print("  PASS -- mutually exclusive combination rejected\n")

    print("=" * 60)
    print("Test 5: reproducibility (same seed -> identical event log)")
    print("=" * 60)
    eng_a, _ = run(k=5)
    eng_b, _ = run(k=5)
    assert eng_a.logger._rows == eng_b.logger._rows, "k-batching run is not reproducible!"
    print("  PASS -- reproducible\n")

    print("=" * 60)
    print("Test 6: calendar + k-Batching terminate promptly (regression)")
    print("=" * 60)
    # See module docstring: this exact combination (calendar on, k=1, real
    # arrival load) once hung for minutes due to the two independent wake
    # mechanisms clobbering each other's "already armed" flag. A generous
    # but finite wall-clock budget catches a regression without making CI
    # hang if it ever comes back.
    calendar = YearlyAvailability.from_json(AVAILABILITY_MODEL_PATH)
    t0 = time.time()
    eng = SimulationEngine(sim_duration=ONE_WEEK, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(
        capacity_per_resource=CAPACITY, seed=SEED, batching_k=1,
        calendar=calendar, start_datetime=START,
    )
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START, resource_component=resources,
    )
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.2f}s (regression threshold: 10s)")
    assert elapsed < 10, (
        f"calendar + k-batching took {elapsed:.1f}s for a 7-day run -- "
        f"this is the same-instant wake-duplication bug regressing"
    )
    print("  PASS -- completed well within budget\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

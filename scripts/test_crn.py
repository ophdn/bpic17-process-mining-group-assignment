"""
test_crn.py
============
Verifies Common Random Numbers (CRN, ProcessComponent/PetriNetProcessComponent
``crn=True``): branching and duration draws should be independent of
resource-allocation order, so paired Part II experiments across allocation
policies actually compare the same case trajectories.

1. Without CRN (crn=False, default): default vs. piled allocation MUST
   produce a different event log under the same seed -- this is the known
   RNG-order cascade documented in output/piled_execution_eval.md, and it's
   the whole reason CRN exists. If this test ever shows identical logs, the
   two allocation policies aren't actually being exercised differently.
2. With CRN (crn=True): default vs. piled allocation must agree on every
   case's branching-activity sequence, up to whichever run's slower
   allocation causes it to hit the horizon with fewer completed activities
   (a length difference from horizon truncation, not a wrong decision --
   this test checks the PREFIX, not exact length).
3. Reproducibility: crn=True is itself still fully seed-reproducible.

Usage:
    cd <repo-root>
    PYTHONPATH=. .venv/bin/python scripts/test_crn.py
"""

from datetime import datetime

from simulation.core.engine import SimulationEngine
from simulation.components.arrival import ArrivalComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent

SEED = 42
START = datetime(2016, 1, 1)
ONE_WEEK = 7 * 24 * 3600
CAPACITY = 3


def run(piled: bool, crn: bool):
    eng = SimulationEngine(sim_duration=ONE_WEEK, start_datetime=START, verbose=False)
    arrivals = ArrivalComponent(seed=SEED)
    resources = ResourceComponent(capacity_per_resource=CAPACITY, seed=SEED, piled=piled)
    process = ProcessComponent(
        seed=SEED, mode="distribution", start_datetime=START,
        resource_component=resources, crn=crn,
    )
    eng.register(arrivals)
    eng.register(resources)
    eng.register(process)
    arrivals.bootstrap(eng)
    eng.run()
    return eng.logger._rows


def activity_sequence(rows):
    seq = {}
    for r in rows:
        if r["lifecycle:transition"] != "start":
            continue
        seq.setdefault(r["case:concept:name"], []).append(r["concept:name"])
    return seq


def main():
    print("=" * 60)
    print("Test 1: without CRN, allocation policy changes the log (expected)")
    print("=" * 60)
    rows_default = run(piled=False, crn=False)
    rows_piled = run(piled=True, crn=False)
    assert rows_default != rows_piled, (
        "default and piled logs are identical without CRN -- either piled "
        "execution isn't doing anything, or something else masked the "
        "RNG-order effect this test exists to demonstrate."
    )
    print("  PASS -- logs differ (RNG-order cascade confirmed present)\n")

    print("=" * 60)
    print("Test 2: with CRN, branching sequences agree across policies")
    print("=" * 60)
    rows_default_crn = run(piled=False, crn=True)
    rows_piled_crn = run(piled=True, crn=True)
    seq_default = activity_sequence(rows_default_crn)
    seq_piled = activity_sequence(rows_piled_crn)
    common = set(seq_default) & set(seq_piled)
    assert common, "no cases in common between the two runs to compare"

    mismatches = []
    for cid in common:
        a, b = seq_default[cid], seq_piled[cid]
        n = min(len(a), len(b))
        if a[:n] != b[:n]:
            mismatches.append(cid)

    print(f"  common cases compared: {len(common)}")
    print(f"  branching-prefix mismatches: {len(mismatches)}")
    assert not mismatches, (
        f"CRN failed: {len(mismatches)} cases have differing branch "
        f"prefixes despite crn=True (e.g. {mismatches[0]})"
    )
    print("  PASS -- every case's branching prefix is policy-independent\n")

    print("=" * 60)
    print("Test 3: crn=True is itself reproducible")
    print("=" * 60)
    rows_a = run(piled=False, crn=True)
    rows_b = run(piled=False, crn=True)
    assert rows_a == rows_b, "crn=True run is not reproducible under a fixed seed!"
    print("  PASS -- reproducible\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

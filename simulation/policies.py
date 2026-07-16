"""
policies.py — Allocation Policy interface (Section 1.8 Advanced: push
selection patterns)
=====================================================================
R-RBA (``ResourceComponent._allocate``) answers "who is *allowed*" by
filtering candidates on role permission, live capacity, and shift. This
module answers "*which* of the allowed ones" — the push selection
decision Russell et al. leave open after role filtering.

``ResourceComponent`` calls ``policy.select(activity, candidates, state)``
with the already-filtered candidate list (permission + capacity + on-shift
all applied upstream) and takes whatever resource comes back. A policy
never sees unqualified or unavailable resources — it cannot violate R-RBA
or the calendar even by accident.

This is a minimal seam, not a finished pattern library: only
``RandomPolicy`` (R-RMA, Pattern 15) is implemented here, reproducing the
project's pre-existing behaviour bit-for-bit. Round Robin (R-RRA, Pattern
16) and Shortest Queue (R-SHQ, Pattern 17) are Part II deliverables built
on this seam — see docs/ROADMAP.md Phase C.

Usage
-----
    from simulation.policies import RandomPolicy
    from simulation.components.resource import ResourceComponent

    resources = ResourceComponent(capacity_per_resource=3, seed=42)
    # equivalent to the explicit form:
    resources = ResourceComponent(
        capacity_per_resource=3, seed=42,
        policy=RandomPolicy(rng=resources._rng),
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Protocol, runtime_checkable


@dataclass(frozen=True)
class AllocationState:
    """Read-only snapshot passed to a policy's ``select()``.

    Deliberately small: just enough for load-aware policies (e.g.
    shortest-queue) without exposing the rest of ``ResourceComponent``.
    ``busy`` is the live ``resource -> active task count`` map (shared
    reference, not a copy — policies must not mutate it).
    """
    busy: Dict[str, int]
    capacity: int


@runtime_checkable
class AllocationPolicy(Protocol):
    """Push-selection interface. ``candidates`` is always non-empty and
    pre-filtered (permitted for *activity*, has a free slot, on shift)."""

    def select(self, activity: str, candidates: List[str], state: AllocationState) -> str:
        ...


class RandomPolicy:
    """R-RMA (Random Allocation, Russell et al. Pattern 15).

    Uniform random pick among qualified-and-available candidates — the
    project's default selection behaviour prior to this interface
    existing. Pass the *same* ``random.Random`` instance a
    ``ResourceComponent`` already uses (its ``_rng``) to preserve the
    exact draw sequence and keep reproducibility bit-for-bit; passing a
    fresh seed instead is fine for standalone use but will not reproduce
    historical event logs.
    """

    def __init__(self, rng=None, seed=None):
        import random
        self._rng = rng if rng is not None else random.Random(seed)

    def select(self, activity: str, candidates: List[str], state: AllocationState) -> str:
        return self._rng.choice(candidates)

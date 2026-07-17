"""
permissions.py — Section 1.7, resource permission models (runtime side).

`ResourceComponent` asks a permission model one question: *who may perform this
activity, right now?* Everything about how that answer is derived lives behind
this interface, so swapping the permission model requires no change to the
engine, the process component, or the event types — only the object handed to
`ResourceComponent(permissions=...)`.

Three implementations:

  StaticPermissions   resource -> set of activities. Covers both the legacy
                      hardcoded top-20 map and the Section 1.7 *Basic* model
                      ("permitted iff observed").

  OrgModelPermissions an organizational model discovered with OrdinoR (Yang et
                      al. 2022): resources belong to (possibly overlapping)
                      groups, and a group's *capabilities* are the execution
                      contexts it may work in. Section 1.7 *Advanced*.

  (any object satisfying the PermissionModel protocol)

Fitting these models needs pandas, scikit-learn and ordinor; running them needs
none of that. So the fitting lives in `analysis/permissions.py` and writes JSON,
and this module only *loads* JSON. The simulation stays dependency-light and the
analysis stays reproducible — the same split as the Section 1.6 calendar.

Why an execution context is more than an activity
-------------------------------------------------
OrdinoR's unit of capability is the triple (case type, activity type, time type),
not the bare activity. A group may be permitted to validate an application *for
car loans* *on weekdays*, and not otherwise. Each component of the triple may be
the wildcard ⊥, meaning "any".

That richness is only worth having if the simulation can actually enforce it. It
can:

  - *time type* — `ResourceComponent` already knows the wall-clock time, because
    the Section 1.6 calendar needs it. So `when=` is available for free.
  - *case type* — the arrival component samples a loan goal for each case and
    carries it on the event payload, so `case_type=` is available too.

Where a model has no opinion on a dimension (e.g. an AT-only model), the wildcard
matches everything and the check degrades gracefully to a plain activity lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

# OrdinoR prefixes type names to keep the three dimensions disjoint, and uses a
# distinguished "any" type. We keep its convention so a model round-trips.
WILDCARD = ""          # ⊥ — matches every value on that dimension

# Weekday time types, matching the paper's TT definition (the seven week days).
# Spelled out in full because that is how OrdinoR names them when it learns the
# contexts; an abbreviation here would silently match nothing.
_WEEKDAY_TT = [
    "TT.Monday", "TT.Tuesday", "TT.Wednesday", "TT.Thursday",
    "TT.Friday", "TT.Saturday", "TT.Sunday",
]


def time_type_of(when) -> str:
    """The time type of a datetime, under the paper's weekday partition."""
    return _WEEKDAY_TT[when.weekday()]


class PermissionModel(Protocol):
    """What ResourceComponent needs from a permission model."""

    def candidates(
        self, activity: str, *, case_type: Optional[str] = None, when=None
    ) -> List[str]:
        """Resources permitted to perform *activity* in this context."""
        ...

    def permits(
        self, resource: str, activity: str, *,
        case_type: Optional[str] = None, when=None,
    ) -> bool:
        """May *resource* perform *activity* in this context?"""
        ...

    def resources(self) -> List[str]:
        """Every resource the model knows about."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Static: resource -> activities
# ──────────────────────────────────────────────────────────────────────────

class StaticPermissions:
    """A flat resource -> activities map, with no case or time dimension.

    This is the Section 1.7 *Basic* model: a resource may perform an activity iff
    it was observed doing so. It is also how the original hardcoded top-20 map is
    expressed, so the two are directly comparable.

    `case_type` and `when` are accepted and ignored — the model has no opinion on
    those dimensions, so it permits every value of them.
    """

    def __init__(self, permissions: Dict[str, Iterable[str]]):
        self._perms: Dict[str, Set[str]] = {
            r: set(acts) for r, acts in permissions.items()
        }
        # Inverse index. Insertion order is stable, so the seeded random pick in
        # ResourceComponent stays reproducible.
        self._by_activity: Dict[str, List[str]] = {}
        for r, acts in self._perms.items():
            for a in acts:
                self._by_activity.setdefault(a, []).append(r)

    def candidates(self, activity, *, case_type=None, when=None) -> List[str]:
        return self._by_activity.get(activity, [])

    def permits(self, resource, activity, *, case_type=None, when=None) -> bool:
        return activity in self._perms.get(resource, ())

    def resources(self) -> List[str]:
        return list(self._perms)

    def activities_of(self, resource: str) -> Set[str]:
        return self._perms.get(resource, set())

    # -- io --

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticPermissions":
        d = json.loads(Path(path).read_text())
        return cls(d["permissions"] if "permissions" in d else d)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"kind": "static",
             "permissions": {r: sorted(a) for r, a in sorted(self._perms.items())}},
            indent=1))
        return path


# ──────────────────────────────────────────────────────────────────────────
# OrdinoR organizational model
# ──────────────────────────────────────────────────────────────────────────

class OrgModelPermissions:
    """Permissions derived from a discovered OrdinoR organizational model.

    A resource is permitted an execution context if it belongs to *any* group
    whose capabilities include a context matching on all three dimensions
    (wildcards match anything).

    This is where the generalisation over the basic matrix comes from: a resource
    inherits its group's capabilities, so it can be permitted an activity it was
    never individually observed performing — because its colleagues do it and the
    model has decided they do the same job.
    """

    def __init__(self, groups: Sequence[dict]):
        """`groups`: [{"members": [...], "capabilities": [[ct, at, tt], ...]}, ...]"""
        self._groups = [
            (list(g["members"]), [tuple(c) for c in g["capabilities"]])
            for g in groups
        ]

        # activity -> [(members, ct, tt), ...]; the hot path in _allocate.
        self._index: Dict[str, List[Tuple[List[str], str, str]]] = {}
        self._resources: List[str] = []
        seen: Set[str] = set()

        for members, caps in self._groups:
            for r in members:
                if r not in seen:
                    seen.add(r)
                    self._resources.append(r)
            for ct, at, tt in caps:
                self._index.setdefault(at, []).append((members, ct, tt))

        # Memo cache for candidates()/permits(): the answer is pure in
        # (activity, case_type, time_type_of(when)) — a small key space
        # (~activities x case types x 8 weekday time-types) — while the scan
        # it replaces walks every matching group's member list. Profiling a
        # 7-day advanced-model run showed candidates() at 44% of TOTAL
        # simulation runtime (433k calls, 87M _matches invocations); with
        # k-Batching it is worse still, because the batch cost matrix calls
        # permits() per (item x resource) pair, each of which re-ran the
        # full scan. Values are (ordered list, frozenset) so candidates()
        # can return a fresh list (callers must never see shared mutable
        # state) and permits() gets an O(1) membership test.
        self._cand_cache: Dict[tuple, Tuple[List[str], frozenset]] = {}

    # -- protocol --

    def _candidates_cached(self, activity, case_type, tt) -> Tuple[List[str], frozenset]:
        key = (activity, case_type, tt)
        hit = self._cand_cache.get(key)
        if hit is not None:
            return hit

        entries = self._index.get(activity)
        out: List[str] = []
        if entries:
            seen: Set[str] = set()
            for members, cap_ct, cap_tt in entries:
                if not _matches(cap_ct, case_type) or not _matches(cap_tt, tt):
                    continue
                for r in members:
                    if r not in seen:
                        seen.add(r)
                        out.append(r)
        hit = (out, frozenset(out))
        self._cand_cache[key] = hit
        return hit

    def candidates(self, activity, *, case_type=None, when=None) -> List[str]:
        tt = time_type_of(when) if when is not None else None
        return list(self._candidates_cached(activity, case_type, tt)[0])

    def permits(self, resource, activity, *, case_type=None, when=None) -> bool:
        tt = time_type_of(when) if when is not None else None
        return resource in self._candidates_cached(activity, case_type, tt)[1]

    def resources(self) -> List[str]:
        return list(self._resources)

    # -- introspection --

    def self_check(self) -> None:
        """Fail loudly if the model's vocabulary cannot match what we supply.

        A permission model whose time types are spelled "TT.Wednesday" will match
        nothing at all if the simulation offers "TT.Wed" — and it fails *silently*,
        by simply granting no candidates, which looks like a saturated pool rather
        than a wiring bug. (It did. This check exists because of it.)
        """
        model_tts = {tt for _, caps in self._groups for _, _, tt in caps if tt}
        if model_tts:
            ours = set(_WEEKDAY_TT)
            if not (model_tts & ours):
                raise ValueError(
                    f"permission model's time types {sorted(model_tts)[:3]}... "
                    f"do not overlap the ones this simulation produces "
                    f"{sorted(ours)[:3]}... — the model would permit nothing"
                )

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    def groups_of(self, resource: str) -> List[int]:
        return [i for i, (m, _) in enumerate(self._groups) if resource in m]

    # -- io --

    @classmethod
    def from_json(cls, path: str | Path) -> "OrgModelPermissions":
        return cls(json.loads(Path(path).read_text())["groups"])

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": "org_model",
            "groups": [
                {"members": sorted(m), "capabilities": [list(c) for c in sorted(caps)]}
                for m, caps in self._groups
            ],
        }, indent=1))
        return path


def _matches(capability_type: str, observed: Optional[str]) -> bool:
    """Does a capability's type on one dimension admit the observed value?

    The wildcard admits anything. So does an unknown observed value: if the
    simulation cannot tell us the case type, we do not use that dimension to
    *deny* — silently forbidding work because a field is missing would be a
    modelling accident, not a permission rule.
    """
    if capability_type == WILDCARD or observed is None:
        return True
    return capability_type == observed


def load(path: str | Path) -> PermissionModel:
    """Load whichever kind of permission model is stored at *path*."""
    kind = json.loads(Path(path).read_text()).get("kind", "static")
    if kind == "org_model":
        return OrgModelPermissions.from_json(path)
    return StaticPermissions.from_json(path)

"""
lifecycle_params.py — active-mode parameter interface (implementationplan §5.1)
==============================================================================
Committed interface (not a recommendation): in ``--lifecycle-mode active`` the
process/resource components load a single ``LifecycleParameters`` object from
``simulation_inputs_active.json`` (the ``lifecycle`` block produced by
``extract_log_info.py --lifecycle``). It carries:

  - ``processing_times``      active-session-second distributions per W_ activity
  - ``session_end_probs``     P(complete | a running session ends)
  - ``suspend_end_probs``     P(resume    | a suspended item continues)
  - ``resume_gap_params``     suspend→resume-ready external-wait residuals
  - ``withdraw_hazard``       time-to-withdraw while merely SCHEDULED
  - ``terminal_continuation`` next activity per (activity, terminal outcome)

``legacy`` mode never constructs this — it keeps the untouched hardcoded
constants in ``process.py`` — so there is no shared global that can drift
between modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# A fitted scipy-like distribution: (distribution_name, params tuple).
DistSpec = Tuple[str, tuple]

# Sentinel emitted by the extractor when a terminal outcome ends the case.
CASE_END = "__CASE_END__"


def _to_dist(fit: dict) -> DistSpec:
    """Convert an extractor fit dict → (dist_name, params) tuple.

    Mirrors the format ``ProcessComponent._sample_scipy_like`` consumes and the
    hardcoded ``PROCESSING_TIME_PARAMS`` use, so the same sampler serves both
    modes.
    """
    return (fit.get("distribution", "expon"), tuple(fit.get("params", [0.0, 600.0])))


@dataclass
class LifecycleParameters:
    processing_times: Dict[str, DistSpec] = field(default_factory=dict)
    session_end_probs: Dict[str, float] = field(default_factory=dict)
    suspend_end_probs: Dict[str, float] = field(default_factory=dict)
    resume_gap_params: Dict[str, DistSpec] = field(default_factory=dict)
    withdraw_hazard: Dict[str, DistSpec] = field(default_factory=dict)
    terminal_continuation: Dict[str, Dict[str, List[Tuple[str, float]]]] = \
        field(default_factory=dict)
    schema: str = "active_v1"

    @classmethod
    def from_block(cls, block: dict) -> "LifecycleParameters":
        """Build from the ``lifecycle`` block of simulation_inputs_active.json."""
        processing_times = {a: _to_dist(f) for a, f in block.get("processing_times", {}).items()}
        resume_gap = {a: _to_dist(f) for a, f in block.get("resume_gap_params", {}).items()}
        withdraw = {a: _to_dist(f) for a, f in block.get("withdraw_hazard", {}).items()}

        terminal = {}
        for act, outcomes in block.get("terminal_continuation", {}).items():
            terminal[act] = {}
            for outcome, dist in outcomes.items():
                # dict {next: p} → list [(next, p)] sorted desc for stable CRN draws.
                terminal[act][outcome] = sorted(
                    ((na, float(p)) for na, p in dist.items()),
                    key=lambda kv: kv[1], reverse=True)

        return cls(
            processing_times=processing_times,
            session_end_probs={a: float(p) for a, p in block.get("session_end_probs", {}).items()},
            suspend_end_probs={a: float(p) for a, p in block.get("suspend_end_probs", {}).items()},
            resume_gap_params=resume_gap,
            withdraw_hazard=withdraw,
            terminal_continuation=terminal,
            schema=block.get("lifecycle_schema", "active_v1"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "LifecycleParameters":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        block = data.get("lifecycle")
        if block is None:
            raise ValueError(
                f"{path} has no `lifecycle` block — regenerate it with "
                f"`extract_log_info.py --lifecycle --out {path}`.")
        return cls.from_block(block)

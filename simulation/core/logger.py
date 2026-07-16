"""
Event Logger: Collects simulation events and writes them as an event log (CSV).

The log follows the XES/PM4Py convention:
  case:concept:name  – case identifier
  concept:name       – activity name
  time:timestamp     – ISO-8601 datetime (derived from sim time + start_datetime)
  lifecycle:transition – start / complete
  org:resource       – resource that executed the activity
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .events import SimEvent, EventType


class EventLogger:
    """
    Collects activity-level events during simulation and flushes
    them to a CSV file at the end.

    In ``legacy`` mode only ACTIVITY_START and ACTIVITY_COMPLETE are recorded, in
    the exact five-column schema — the log stays byte-identical to pre-lifecycle
    runs. In ``active`` mode the full W_ work-item lifecycle is recorded
    (schedule/start/suspend/resume/complete/ate_abort/withdraw) plus a sixth
    ``work_item_id`` column (implementationplan §4.3). Case arrivals / completions
    are tracked separately for statistics.
    """

    # Five-column legacy schema (byte-identical to pre-lifecycle output).
    COLUMNS = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "lifecycle:transition",
        "org:resource",
    ]
    # Active mode adds work_item_id so suspend/resume sessions of one item can be
    # reconstructed without mis-joining repeated (case, activity) instances.
    ACTIVE_COLUMNS = COLUMNS + ["work_item_id"]

    # EventType → lifecycle:transition. START/COMPLETE apply to every activity;
    # the schedule/suspend/resume/abort/withdraw rows are gated to W_ items in log().
    _LEGACY_MAP = {
        EventType.ACTIVITY_START: "start",
        EventType.ACTIVITY_COMPLETE: "complete",
    }
    _ACTIVE_MAP = {
        EventType.ACTIVITY_REQUEST: "schedule",
        EventType.ACTIVITY_START: "start",
        EventType.ACTIVITY_SUSPEND: "suspend",
        EventType.ACTIVITY_RESUME: "resume",
        EventType.ACTIVITY_COMPLETE: "complete",
        EventType.ACTIVITY_ABORT: "ate_abort",
        EventType.ACTIVITY_WITHDRAW: "withdraw",
    }
    # Transitions that are only meaningful for W_ work items — logging them for the
    # synthetic A_/O_ requests would fabricate lifecycle rows absent from BPIC-17.
    _W_ONLY = {"schedule", "suspend", "resume", "ate_abort", "withdraw"}

    def __init__(self, start_datetime: Optional[datetime] = None,
                 lifecycle_mode: str = "legacy"):
        """
        Parameters
        ----------
        start_datetime:
            Real-world anchor for t=0. Defaults to 2024-01-01 00:00:00.
            Simulation timestamps (floats, in seconds) are added as offsets.
        lifecycle_mode:
            ``"legacy"`` (default) → five-column start/complete log.
            ``"active"`` → full W_ lifecycle + work_item_id column.
        """
        self._start: datetime = start_datetime or datetime(2024, 1, 1)
        self._rows: List[dict] = []
        self._mode = lifecycle_mode
        self._active = lifecycle_mode == "active"
        self._map = self._ACTIVE_MAP if self._active else self._LEGACY_MAP
        self.columns = self.ACTIVE_COLUMNS if self._active else self.COLUMNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: SimEvent) -> None:
        """Record a single event if it belongs in the event log."""
        transition = self._map.get(event.event_type)
        if transition is None:
            return  # Not a logged activity transition in this mode – skip

        # Internal control sentinels (e.g. __PROCESS_START__) are routing
        # signals, not real activities — they must not appear in the log.
        if event.activity and event.activity.startswith("__"):
            return

        # Lifecycle transitions beyond start/complete apply only to W_ items.
        if transition in self._W_ONLY and not (
                event.activity and event.activity.startswith("W_")):
            return

        payload = event.payload if isinstance(event.payload, dict) else {}
        # A resume-ready re-request is an ACTIVITY_REQUEST too, but it must NOT emit
        # a second `schedule` row: in BPIC-17 `schedule` counts ≈ work items, not
        # work items + resumes. The one schedule row is the initial enablement; the
        # resume row (emitted after re-allocation) marks the continuation. §4.3/§4.6
        if transition == "schedule" and payload.get("resuming"):
            return

        ts = self._sim_time_to_datetime(event.timestamp)
        row = {
            "case:concept:name": event.case_id,
            "concept:name": event.activity,
            "time:timestamp": ts.isoformat(),
            "lifecycle:transition": transition,
            "org:resource": event.resource or "",
        }
        if self._active:
            row["work_item_id"] = payload.get("work_item_id", "")
        self._rows.append(row)

    def save(self, path: str | Path) -> Path:
        """
        Write all collected events to *path* as a UTF-8 CSV.

        Returns the resolved path for convenience.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"[EventLogger] Saved {len(self._rows)} events to '{path}'")
        return path

    def clear(self) -> None:
        """Reset the logger (useful between simulation runs)."""
        self._rows.clear()

    @property
    def num_events(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sim_time_to_datetime(self, sim_time: float) -> datetime:
        """Convert a simulation timestamp (seconds) to a real datetime."""
        return self._start + timedelta(seconds=sim_time)

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

    Only ACTIVITY_START and ACTIVITY_COMPLETE events are recorded
    in the event log (case arrivals / completions are tracked
    separately for statistics).
    """

    # Column order in the output CSV
    COLUMNS = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "lifecycle:transition",
        "org:resource",
    ]

    def __init__(self, start_datetime: Optional[datetime] = None):
        """
        Parameters
        ----------
        start_datetime:
            Real-world anchor for t=0. Defaults to 2024-01-01 00:00:00.
            Simulation timestamps (floats, in seconds) are added as offsets.
        """
        self._start: datetime = start_datetime or datetime(2024, 1, 1)
        self._rows: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: SimEvent) -> None:
        """Record a single event if it belongs in the event log."""
        if event.event_type == EventType.ACTIVITY_START:
            transition = "start"
        elif event.event_type == EventType.ACTIVITY_COMPLETE:
            transition = "complete"
        else:
            return  # Not an activity event – skip

        # Internal control sentinels (e.g. __PROCESS_START__) are routing
        # signals, not real activities — they must not appear in the log.
        if event.activity and event.activity.startswith("__"):
            return

        ts = self._sim_time_to_datetime(event.timestamp)
        self._rows.append({
            "case:concept:name": event.case_id,
            "concept:name": event.activity,
            "time:timestamp": ts.isoformat(),
            "lifecycle:transition": transition,
            "org:resource": event.resource or "",
        })

    def save(self, path: str | Path) -> Path:
        """
        Write all collected events to *path* as a UTF-8 CSV.

        Returns the resolved path for convenience.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
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

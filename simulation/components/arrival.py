"""
arrival.py — Case Arrival Component (Section 1.2 Basic)
=========================================================
Inter-arrival times fitted to the real BPIC-17 event log.

Fitted distribution: LogNormal(s=1.4543, loc=0, scale=363.10)
  → mean inter-arrival ≈ 1002 seconds (~86 cases/day)
  → daily arrivals: mean=86, std=32, min=20, max=178

Replace this with a dynamic spawn-rate model for the Advanced task.
"""

import random
from typing import Optional

from ..core.events import SimEvent, EventType


class ArrivalComponent:
    """
    Generates new loan application cases using a LogNormal inter-arrival
    distribution fitted on the BPIC-17 data.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (fix this in your experiments!).
    scale_factor : float
        Multiplier on the arrival rate. 1.0 = real BPIC-17 rate (~86/day).
        Use < 1.0 to slow down for debugging.
    """

    # LogNormal params from scipy.stats.lognorm.fit on BPIC-17
    # lognorm(s=1.4543, loc=0, scale=363.0958)
    # mean inter-arrival = 1002.23 seconds
    _LOGNORM_S     = 1.4543
    _LOGNORM_SCALE = 363.0958   # seconds

    HANDLES = {EventType.CASE_ARRIVAL: None}  # patched below

    def __init__(self, seed: Optional[int] = 42, scale_factor: float = 1.0,
                 stop_time: Optional[float] = None):
        self._rng = random.Random(seed)
        self._scale_factor = scale_factor
        if stop_time is not None and stop_time < 0:
            raise ValueError("stop_time must be non-negative")
        self._stop_time = stop_time
        self._case_counter = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def bootstrap(self, engine) -> None:
        """Schedule the very first arrival. Must be called before engine.run()."""
        self._schedule_next(engine, current_time=0.0)

    def on_arrival(self, engine, event: SimEvent) -> None:
        """Handle CASE_ARRIVAL: hand off to process, schedule next arrival."""
        # Signal the process component to start routing this case
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_START,
            case_id=event.case_id,
            activity="__PROCESS_START__",
            payload=event.payload or {},
        ))
        self._schedule_next(engine, current_time=engine.now)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sample_inter_arrival(self) -> float:
        """Sample from LogNormal distribution, applying scale factor."""
        # LogNormal via normal: X = scale * exp(s * Z), Z ~ N(0,1)
        import math
        z = self._rng.gauss(0, 1)
        raw = self._LOGNORM_SCALE * math.exp(self._LOGNORM_S * z)
        return raw / self._scale_factor

    def _schedule_next(self, engine, current_time: float) -> None:
        inter_arrival = self._sample_inter_arrival()
        timestamp = current_time + inter_arrival
        if self._stop_time is not None and timestamp > self._stop_time:
            return
        self._case_counter += 1
        engine.schedule(SimEvent(
            timestamp=timestamp,
            priority=10,
            event_type=EventType.CASE_ARRIVAL,
            case_id=f"case_{self._case_counter:06d}",
            payload={},
        ))


ArrivalComponent.HANDLES = {EventType.CASE_ARRIVAL: ArrivalComponent.on_arrival}

"""
Simulation Engine Core
======================
Implements a Discrete Event Simulation (DES) using a single, global
priority queue (min-heap ordered by event timestamp).

Architecture
------------
The engine is intentionally a *thin router*. It:
  1. Maintains the global event queue.
  2. Dispatches each event to registered handler components.
  3. Delegates all domain logic (arrivals, process routing, resources,
     logging) to pluggable components.

Usage
-----
    from simulation.core.engine import SimulationEngine
    from simulation.core.events import SimEvent, EventType

    engine = SimulationEngine(sim_duration=3600 * 24 * 30)  # 30 simulated days

    # Register components (see components/ package)
    engine.register(arrival_component)
    engine.register(process_component)
    engine.register(resource_component)
    engine.register(logger_component)

    engine.run()
    engine.logger.save("output/event_log.csv")
"""

import heapq
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .events import SimEvent, EventType
from .logger import EventLogger


# A handler is any callable that accepts (engine, event) -> None
HandlerFn = Callable[["SimulationEngine", SimEvent], None]


class SimulationEngine:
    """
    Discrete Event Simulation engine with a single global event queue.

    Parameters
    ----------
    sim_duration : float
        Simulated time horizon in seconds. The simulation stops when
        the queue is empty OR the next event exceeds this horizon.
    start_datetime : datetime, optional
        Real-world anchor for t=0 (passed through to EventLogger).
    verbose : bool
        Print each dispatched event to stdout (useful for debugging).
    """

    def __init__(
        self,
        sim_duration: float,
        start_datetime: Optional[datetime] = None,
        verbose: bool = False,
    ):
        self.sim_duration = sim_duration
        self.verbose = verbose

        # --- State ---
        self._now: float = 0.0          # Current simulation clock
        self._queue: List[SimEvent] = []  # Global priority queue (min-heap)
        self._event_counter: int = 0    # Monotonic counter; breaks timestamp ties

        # --- Handlers ---
        # Map EventType -> list of callables that handle it
        self._handlers: Dict[EventType, List[HandlerFn]] = {
            et: [] for et in EventType
        }

        # --- Built-in logger (always registered) ---
        self.logger = EventLogger(start_datetime=start_datetime)
        self._register_builtin_handlers()

        # --- Statistics ---
        self.stats = {
            "events_processed": 0,
            "cases_started": 0,
            "cases_completed": 0,
            "wall_time_seconds": 0.0,
        }

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def register_handler(self, event_type: EventType, handler: HandlerFn) -> None:
        """
        Register *handler* to be called whenever an event of *event_type*
        is dispatched.

        Multiple handlers can be registered for the same event type;
        they are called in registration order.
        """
        self._handlers[event_type].append(handler)

    def register(self, component) -> None:
        """
        Convenience method: register all handlers declared by *component*.

        A component must expose a dict ``HANDLES: {EventType: handler_method}``.
        Example::

            class ArrivalComponent:
                HANDLES = {EventType.CASE_ARRIVAL: ArrivalComponent.on_arrival}

                def on_arrival(self, engine, event): ...
        """
        if not hasattr(component, "HANDLES"):
            raise ValueError(
                f"Component {component!r} must define a 'HANDLES' dict "
                f"mapping EventType -> callable."
            )
        for event_type, handler in component.HANDLES.items():
            # Bind instance method if needed
            bound = getattr(component, handler.__name__, handler)
            self.register_handler(event_type, bound)

    # ------------------------------------------------------------------
    # Event queue management
    # ------------------------------------------------------------------

    def schedule(self, event: SimEvent) -> None:
        """
        Push *event* onto the global priority queue.

        Events with the same timestamp are ordered by ``event.priority``
        (lower fires first), then by insertion order.
        """
        # Use a stable counter so equal (timestamp, priority) events
        # preserve FIFO order.
        heapq.heappush(self._queue, event)
        if self.verbose:
            print(f"  [SCHEDULE] {event}")

    def schedule_in(self, delay: float, event: SimEvent) -> None:
        """Schedule *event* at current time + *delay* seconds."""
        event.timestamp = self._now + delay
        self.schedule(event)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the simulation loop.

        Processes events from the global queue in timestamp order until
        the queue is empty or the simulation horizon is reached.
        """
        print(f"[Engine] Starting simulation (duration={self.sim_duration:.0f}s)")
        wall_start = time.perf_counter()

        while self._queue:
            event = heapq.heappop(self._queue)

            # Stop condition: event is beyond the simulation horizon
            if event.timestamp > self.sim_duration:
                if self.verbose:
                    print(f"[Engine] Horizon reached at t={event.timestamp:.2f}. Stopping.")
                break

            # Advance the simulation clock
            self._now = event.timestamp
            self.stats["events_processed"] += 1

            if self.verbose:
                print(f"[t={self._now:>12.3f}] DISPATCH {event}")

            # Dispatch to all registered handlers
            for handler in self._handlers.get(event.event_type, []):
                handler(self, event)

        wall_elapsed = time.perf_counter() - wall_start
        self.stats["wall_time_seconds"] = wall_elapsed

        print(
            f"[Engine] Simulation complete. "
            f"Events processed: {self.stats['events_processed']}, "
            f"Cases started: {self.stats['cases_started']}, "
            f"Cases completed: {self.stats['cases_completed']}, "
            f"Wall time: {wall_elapsed:.3f}s"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def now(self) -> float:
        """Current simulation time in seconds."""
        return self._now

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------
    # Built-in handlers (always active)
    # ------------------------------------------------------------------

    def _register_builtin_handlers(self) -> None:
        """Register the engine's own internal handlers."""
        self.register_handler(EventType.ACTIVITY_START, self._on_activity_event)
        self.register_handler(EventType.ACTIVITY_COMPLETE, self._on_activity_event)
        self.register_handler(EventType.CASE_ARRIVAL, self._on_case_arrival)
        self.register_handler(EventType.CASE_COMPLETE, self._on_case_complete)
        self.register_handler(EventType.SIM_END, self._on_sim_end)

    def _on_activity_event(self, engine: "SimulationEngine", event: SimEvent) -> None:
        """Forward activity events to the logger."""
        self.logger.log(event)

    def _on_case_arrival(self, engine: "SimulationEngine", event: SimEvent) -> None:
        self.stats["cases_started"] += 1

    def _on_case_complete(self, engine: "SimulationEngine", event: SimEvent) -> None:
        self.stats["cases_completed"] += 1

    def _on_sim_end(self, engine: "SimulationEngine", event: SimEvent) -> None:
        """Clear the queue to stop the loop immediately."""
        self._queue.clear()
        print(f"[Engine] SIM_END event received at t={self._now:.2f}.")

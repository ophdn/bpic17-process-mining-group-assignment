"""
Event definitions for the Discrete Event Simulation engine.
All events in the simulation are represented as instances of SimEvent.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EventType(Enum):
    """All possible event types in the simulation."""
    # Case lifecycle
    CASE_ARRIVAL = auto()       # A new process case arrives
    CASE_COMPLETE = auto()      # A case has finished all activities

    # Activity lifecycle
    ACTIVITY_REQUEST = auto()   # Work item enabled; waiting for a resource (logged `schedule` for W_)
    ACTIVITY_START = auto()     # An activity starts executing (resource held)
    ACTIVITY_COMPLETE = auto()  # An activity finishes executing

    # Work-item lifecycle churn (active mode only; W_ work items) — §4.1.
    # No ACTIVITY_SCHEDULE: ACTIVITY_REQUEST already *is* the scheduling event, so a
    # second schedule event would risk duplicate queue entries.
    ACTIVITY_SUSPEND = auto()   # A running session pauses; resource released to the pool
    ACTIVITY_RESUME = auto()    # A suspended item resumes RUNNING (emitted after allocation)
    ACTIVITY_ABORT = auto()     # Work item killed (ate_abort); the case continues
    ACTIVITY_WITHDRAW = auto()  # A SCHEDULED item is removed from the queue before starting

    # Resource lifecycle
    RESOURCE_AVAILABLE = auto() # A resource becomes available
    RESOURCE_BUSY = auto()      # A resource is assigned to a task

    # Simulation control
    SIM_END = auto()            # Simulation termination signal


@dataclass(order=True)
class SimEvent:
    """
    A single simulation event placed on the global event queue.

    Events are ordered by timestamp (earliest first). When timestamps
    are equal, priority breaks the tie (lower = higher priority).
    """
    timestamp: float                          # Simulation time when the event fires
    priority: int = field(default=10)         # Tie-breaker (lower = fires first)
    event_type: EventType = field(compare=False, default=None)
    case_id: Optional[str] = field(compare=False, default=None)
    activity: Optional[str] = field(compare=False, default=None)
    resource: Optional[str] = field(compare=False, default=None)
    payload: Any = field(compare=False, default=None)  # Component-specific data

    def __repr__(self):
        return (
            f"SimEvent(t={self.timestamp:.3f}, type={self.event_type.name}, "
            f"case={self.case_id}, activity={self.activity}, resource={self.resource})"
        )

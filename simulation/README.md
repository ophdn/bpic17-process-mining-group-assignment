# Business Process Simulation Engine — Core

> TUM · Business Process Prediction, Simulation & Optimization · Group Assignment

This document explains the **Simulation Engine Core** (Section 1.1 of the assignment) to all team members. It covers the architecture, how events flow, and how to add your own component on top.

---

## Project structure

```
simulation/
├── core/
│   ├── engine.py     ← The DES engine + global event queue  ← START HERE
│   ├── events.py     ← SimEvent dataclass + EventType enum
│   └── logger.py     ← Writes the event log to CSV
├── components/
│   └── stubs.py      ← Placeholder implementations (replace these!)
├── utils/            ← Shared helpers (add yours here)
└── main.py           ← Demo runner — shows how to wire everything together
```

---

## Core concept: Discrete Event Simulation (DES)

The engine runs a **single global priority queue** (a min-heap). Every thing that happens in the simulation — a case arriving, an activity starting, a resource becoming free — is represented as a `SimEvent` placed on this queue.

The main loop is simple:

```
while queue is not empty:
    pop the event with the earliest timestamp
    advance the clock to that timestamp
    call all registered handlers for that event type
```

That's it. No threads, no polling. Time jumps directly from event to event.

---

## The two key files you need to understand

### `core/events.py` — What an event looks like

```python
@dataclass
class SimEvent:
    timestamp: float        # When does this event happen? (seconds from t=0)
    priority:  int          # Tie-breaker when timestamps are equal (lower = first)
    event_type: EventType   # What kind of event is this?
    case_id:   str          # Which process case does it belong to?
    activity:  str          # Which activity? (optional)
    resource:  str          # Which resource? (optional)
    payload:   Any          # Any extra data your component needs
```

`EventType` is an enum with all possible event types:

| EventType | Meaning |
|---|---|
| `CASE_ARRIVAL` | A new process instance arrives |
| `CASE_COMPLETE` | A case has finished all activities |
| `ACTIVITY_START` | An activity begins |
| `ACTIVITY_COMPLETE` | An activity finishes |
| `RESOURCE_AVAILABLE` | A resource becomes free |
| `RESOURCE_BUSY` | A resource is assigned |
| `SIM_END` | Hard-stop the simulation immediately |

---

### `core/engine.py` — How the engine works

The engine is a **thin router**. It has zero domain logic. It only does three things:

1. **Maintains the queue** — `engine.schedule(event)` pushes an event onto it
2. **Dispatches events** — calls every registered handler for an event type
3. **Tracks statistics** — cases started/completed, events processed, wall time

#### Scheduling events

```python
# Schedule at an absolute simulation time
engine.schedule(SimEvent(timestamp=3600.0, event_type=EventType.CASE_ARRIVAL, ...))

# Schedule relative to now (current simulation clock)
engine.schedule_in(delay=600.0, event=SimEvent(...))
```

#### Reading the current time

```python
engine.now   # float, current simulation clock in seconds
```

---

## How to write your own component

A component is just a **plain Python class** with a `HANDLES` dict that maps `EventType → method`.

The engine calls your method whenever an event of that type is dispatched, passing itself (`engine`) and the event.

### Minimal example

```python
from simulation.core.events import EventType, SimEvent

class MyComponent:

    HANDLES = {EventType.CASE_ARRIVAL: None}  # filled below

    def on_arrival(self, engine, event: SimEvent):
        print(f"[t={engine.now:.0f}] Case {event.case_id} arrived!")

        # Schedule something for 10 minutes later
        engine.schedule_in(600, SimEvent(
            timestamp=0,                           # overwritten by schedule_in
            event_type=EventType.ACTIVITY_START,
            case_id=event.case_id,
            activity="A_Create Application",
            resource="resource_1",
        ))

# Patch HANDLES after class definition (needed so the method reference resolves)
MyComponent.HANDLES = {EventType.CASE_ARRIVAL: MyComponent.on_arrival}
```

### Register it with the engine

```python
engine = SimulationEngine(sim_duration=30 * 24 * 3600)

my_comp = MyComponent()
engine.register(my_comp)     # that's all
```

You can register **multiple components for the same event type** — they are called in registration order. The built-in logger is always registered first and cannot be overridden.

---

## How to run

```bash
cd simulation/
PYTHONPATH=.. python main.py
```

Output will be saved to `simulation/output/event_log.csv`.

To enable verbose event-by-event output (useful for debugging):

```python
engine = SimulationEngine(sim_duration=..., verbose=True)
```

---

## The event log (CSV output)

The logger writes a PM4Py-compatible event log. Every `ACTIVITY_START` and `ACTIVITY_COMPLETE` event is recorded as one row:

| Column | Description |
|---|---|
| `case:concept:name` | Case identifier, e.g. `case_000042` |
| `concept:name` | Activity name |
| `time:timestamp` | ISO-8601 datetime (anchored to a configurable start date) |
| `lifecycle:transition` | `start` or `complete` |
| `org:resource` | Resource that executed the activity |

To save the log after a run:

```python
engine.logger.save("output/event_log.csv")
```

To change the real-world start date (default: 2024-01-01):

```python
from datetime import datetime
engine = SimulationEngine(..., start_datetime=datetime(2017, 1, 1))
```

---

## What is already implemented vs. what you need to build

| Assignment section | Status | File to edit/create |
|---|---|---|
| **1.1** Simulation Engine Core | ✅ Done | `core/engine.py`, `core/events.py`, `core/logger.py` |
| **1.2** Case Arrivals | 🟡 Stub (Exponential) | `components/stubs.py → ArrivalComponent` |
| **1.3** Processing Times | 🟡 Stub (Exponential) | `components/stubs.py → ProcessComponent` |
| **1.4** Process Model | 🟡 Stub (linear sequence) | Replace `ProcessComponent` with Petri net / BPMN |
| **1.5** Branching Decisions | ❌ Missing | Add XOR gateway logic to the process component |
| **1.6** Resource Availabilities | ❌ Missing | New `ResourceComponent`, handles `RESOURCE_AVAILABLE` |
| **1.7** Resource Permissions | ❌ Missing | Add permission check inside resource allocation |
| **1.8** Resource Allocation | ❌ Missing | Random allocation inside `ResourceComponent` |

**The stubs in `components/stubs.py` are intentionally simple.** They let the engine run end-to-end right now. Replace them one section at a time without touching the core.

---

## Key design decisions

**Why a single global queue?**
The assignment requires it (Section 1.1). It also means the simulation is fully deterministic and reproducible when you fix the random seed — important for empirical evaluation.

**Why a dispatcher / handler pattern?**
Each team member can implement their component independently and register it without modifying any other file. The engine stays unchanged no matter what components are added.

**Why are simulation times in seconds?**
Seconds are the natural base unit for `datetime` arithmetic in Python. The logger converts them to real datetimes transparently.

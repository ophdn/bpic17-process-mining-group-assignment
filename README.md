# bpic17-process-mining-group-assignment

# Business Process Simulation Engine — Core

> TUM · Business Process Prediction, Simulation & Optimization · Group Assignment

This document explains the **Simulation Engine** to all team members: the DES core (Section 1.1), how events flow, how to add your own component, and the status of each assignment section.

---

## Project structure

```
output/                          ← simulation results (event_log.csv) — always here, never inside simulation/
simulation/
├── core/
│   ├── engine.py                ← The DES engine + global event queue  ← START HERE
│   ├── events.py                ← SimEvent dataclass + EventType enum
│   └── logger.py                ← Writes the event log to CSV
├── components/
│   ├── arrival.py                ← Section 1.2: LogNormal case arrivals
│   ├── process.py                ← Section 1.3+1.5 Basic: fitted durations + branching probs
│   ├── petri_process.py          ← Section 1.4 Advanced: BPMN → Petri net control-flow enforcement
│   └── resource.py               ← Section 1.6+1.7+1.8: permissions + allocation
├── models/                       ← engine input artifacts (things the simulation *loads*)
│   └── bpic17_process.bpmn       ← discovered process model (Section 1.4 Advanced)
└── main.py                       ← entry point — wires all components together
scripts/
├── metrics.py                       ← reusable KPI functions (see docs/paper_insights_*.md)
└── compare_process_models.py        ← runs Basic vs. Advanced and reports all KPIs
docs/
└── paper_insights_discovering_simulation_models.md  ← validation methodology background
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

### First-time setup (clone → add the log → run)

1. Install dependencies: `pip install -r requirements.txt`.
2. The **default** run needs **no raw log** — the fitted distributions, BPMN,
   availability model and branching probabilities are all committed:
   ```bash
   .venv/bin/python -m simulation.main
   ```
3. The **ML-dependent** modes (`--mode ml_model`, `--mode ml_probabilistic`,
   `--k-batching`, `--branching-mode rules`) need trained artifacts that are
   gitignored (large binaries derived from the raw log). Place the BPIC-17
   log at the repo root as `BPIChallenge2017.xes` (or `.xes.gz`, or under
   `data/`), then regenerate **every** gitignored artifact with one command:
   ```bash
   .venv/bin/python setup_models.py            # add --force to retrain
   ```
   This writes `simulation/models/processing_time_model.joblib` (point + 19
   quantile models) and `simulation/models/decision_rules.joblib`. Artifacts
   that already exist are skipped unless `--force` is passed.

From the repo root, inside the virtualenv:

```bash
.venv/bin/python -m simulation.main                            # advanced process model (default), fitted distributions
.venv/bin/python -m simulation.main --process-model basic      # flat next-activity probability graph (Section 1.4 Basic)
.venv/bin/python -m simulation.main --mode ml_model            # contextual point-estimate ML durations
.venv/bin/python -m simulation.main --mode ml_probabilistic    # contextual probabilistic ML durations
.venv/bin/python -m simulation.main --availability always       # Section 1.6: resources never off-shift (baseline)
.venv/bin/python -m simulation.main --piled-execution           # Section 1.8: Piled Execution (R-PE) batch allocation
```

Output is always saved to `<repo_root>/output/event_log.csv`, regardless of
the working directory you run from.
The `ml_*` modes need a trained artifact — run `setup_models.py` (see
first-time setup above), or see **[Processing-Time Models
(Section 1.3)](simulation/PROCESSING_TIMES.md)** for training, mode
details and reference statistics. To check whether Basic or Advanced (or any
change you make) better approximates the real BPIC-17 process, run
`scripts/compare_process_models.py` — see
[docs/paper_insights_discovering_simulation_models.md](docs/paper_insights_discovering_simulation_models.md)
for the KPIs it reports and why.

To enable verbose event-by-event output (useful for debugging), set
`verbose=True` when constructing `SimulationEngine` in `main.py`.

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
engine.logger.save(repo_root / "output" / "event_log.csv")
```

To change the real-world start date (default: 2024-01-01):

```python
from datetime import datetime
engine = SimulationEngine(..., start_datetime=datetime(2017, 1, 1))
```

---

## What is already implemented vs. what you need to build

| Assignment section | Status | File |
|---|---|---|
| **1.1** Simulation Engine Core | ✅ Done | `core/engine.py`, `core/events.py`, `core/logger.py` |
| **1.2** Case Arrivals | ✅ Done (Basic: fitted LogNormal; Advanced: time-dependent MDN, opt-in via `USE_MDN_ARRIVALS`) | `components/arrival.py`, `components/arrival_mdn.py` |
| **1.3** Processing Times | ✅ Done (3 modes: distribution / ml_model / ml_probabilistic) | `components/process.py`, `train_processing_time_model.py` — see [PROCESSING_TIMES.md](simulation/PROCESSING_TIMES.md) |
| **1.4** Process Model | ✅ Done (Basic: probability graph; Advanced: BPMN → Petri net enforcement) | `components/process.py`, `components/petri_process.py`, `models/bpic17_process.bpmn` |
| **1.5** Branching Decisions | ✅ Done (Basic: empirical branching probabilities) | `components/process.py` (`BRANCHING_PROBS`) |
| **1.6** Resource Availabilities | ✅ Done (Basic: fixed capacity per resource; Advanced: fitted shift/holiday/vacation calendar, default via `--availability calendar`) | `components/resource.py`, `analysis/availability.py` |
| **1.7** Resource Permissions | ✅ Done (Basic: resource→activity permission map) | `components/resource.py` |
| **1.8** Resource Allocation | ✅ Done (Basic: random allocation among permitted resources; optional Piled Execution/R-PE batch allocation via `--piled-execution`) | `components/resource.py` — see [resource_allocation_heuristics.md](docs/manuals/resource_allocation_heuristics.md) |

Every section above still has open Advanced variants beyond what's marked
done (see the "Upgrade path" note at the top of each component file).

---

## Key design decisions

**Why a single global queue?**
The assignment requires it (Section 1.1). It also means the simulation is fully deterministic and reproducible when you fix the random seed — important for empirical evaluation.

**Why a dispatcher / handler pattern?**
Each team member can implement their component independently and register it without modifying any other file. The engine stays unchanged no matter what components are added.

**Why are simulation times in seconds?**
Seconds are the natural base unit for `datetime` arithmetic in Python. The logger converts them to real datetimes transparently.

---

## Arrival models: parametric vs. MDN (Section 1.2)

Two interchangeable case-arrival components exist:

| Datei | Modell | Inter-Arrival-Verteilung |
|---|---|---|
| `components/arrival.py` | **Parametrisch** (Basic) | eine feste LogNormal, zeit-unabhängig |
| `components/arrival_mdn.py` | **MDN** (Advanced) | zeitabhängig — bedingt auf Tageszeit/Wochentag/Saison |

Das **MDN** (Mixture Density Network, Log-Normal-Mischung) ist ein intensitätsfreier
Temporal Point Process: ein kleines neuronales Netz gibt — abhängig von der aktuellen
Sim-Uhrzeit — die Verteilung der nächsten Inter-Arrival-Time aus. Damit bildet es die
reale Struktur ab (nachts ~0.6 Ankünfte/h, Kern 12–18h ~7.6/h; Mo ≈ 3× So; Sommer +35 %),
die eine statische Verteilung prinzipiell nicht erfassen kann.

**Umschalten** in `main.py`:
```python
USE_MDN_ARRIVALS = True   # False = parametrische LogNormal (Default)
```

**Laufzeit braucht kein PyTorch** — die Komponente lädt vortrainierte Gewichte
(`components/arrival_mdn_weights.npz`) und wertet sie als reinen NumPy-Forward-Pass aus.

**Gewichte neu trainieren** (einmaliger Offline-Schritt, benötigt PyTorch):
```bash
uv add torch      # nur fürs Training
python train_arrival_mdn.py \
    --arrivals path/to/arrivals.parquet \
    --out simulation/components/arrival_mdn_weights.npz
```
`arrivals.parquet` braucht nur eine Spalte `arrival` mit dem Zeitstempel des ersten
Events je Fall. Wichtig: `START_DATETIME` in `main.py` muss den Wochentag korrekt
verankern (BPIC-17 startet 2016-01-01, ein Freitag), damit Wochentag/Tageszeit aligned sind.
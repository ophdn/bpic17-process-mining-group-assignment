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
│   ├── arrival.py                ← Section 1.2 Basic: LogNormal case arrivals
│   ├── arrival_mdn.py            ← Section 1.2 Advanced: time-dependent MDN arrivals
│   ├── process.py                ← Section 1.3+1.5 Basic: fitted durations + branching probs
│   ├── petri_process.py          ← Section 1.4 Advanced: BPMN → Petri net control-flow enforcement
│   ├── case_attributes.py        ← Section 1.5/1.7: samples a case type (loan goal) per case
│   ├── permissions.py            ← Section 1.7: permission models (loads JSON; runtime side)
│   └── resource.py               ← Section 1.6+1.7+1.8: availability + permissions + allocation
├── policies.py                  ← Section 1.8: push-selection seam (which allowed resource wins)
├── expected_duration.py         ← Section 1.8: expected-duration cost model for k-Batching
├── models/                       ← engine input artifacts the simulation *loads*
│   ├── bpic17_process.bpmn       ← discovered process model (Section 1.4 Advanced)
│   ├── processing_time_model.joblib   ← legacy trained ML durations (gitignored)
│   ├── processing_time_model_active.joblib ← committed active-lifecycle ML durations
│   └── decision_rules.joblib     ← committed decision-point classifiers (§1.5 rules)
└── main.py                       ← entry point — wires all components together
analysis/                        ← Section 1.6/1.7 FITTING side (needs pandas/sklearn/ordinor)
├── loader.py                    ← shared BPIC-17 log loading
├── availability.py              ← fits Weekly (Basic) / Yearly (Advanced) shift calendars
└── permissions.py               ← fits observed matrix (Basic) / OrdinoR org model (Advanced)
notebooks/
├── 01_resource_availability.ipynb   ← §1.6 fitting narrative + design decisions
└── 02_resource_permissions.ipynb    ← §1.7 fitting narrative + design decisions
models/                          ← fitted JSON the simulation loads (committed; light to read)
├── availability_model.json      ← §1.6 per-resource shifts / holidays / vacation
├── permissions_orgmodel.json    ← §1.7 Advanced OrdinoR organizational model
├── permissions_observed.json    ← §1.7 Basic resource×activity matrix
└── case_attributes.json         ← §1.5/1.7 loan-goal distribution for case sampling
scripts/
├── metrics.py                       ← reusable KPI functions (see docs/paper_insights_*.md)
└── compare_process_models.py        ← runs Basic vs. Advanced and reports all KPIs
docs/
├── paper_insights_discovering_simulation_models.md  ← validation methodology background
└── manuals/
    └── resource_allocation_heuristics.md            ← §1.8 pattern rationale (R-RBA/R-DE/…)
```

**The recurring split — fit offline, load at runtime.** Every learned resource
model is *fitted* by the heavy `analysis/` code (pandas, scikit-learn, ordinor)
in a notebook and written to `models/*.json`; the simulation only ever *loads*
that JSON. So a normal run needs none of the data-science stack, stays
deterministic, and the fitting stays reproducible in its notebook. Availability
(§1.6) and permissions (§1.7) both follow this pattern, exactly like the §1.3
processing-time and §1.4 process models before them.

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

## Resources: availability, permissions, allocation (Sections 1.6–1.8)

All three resource sections live in **one component**, `components/resource.py`,
because they answer three parts of the *same* question the instant a work item
becomes enabled: **may** anyone do it (1.7 permissions), is anyone **on shift**
to do it (1.6 availability), and **which** free-and-qualified person actually
gets it (1.8 allocation). This section is the "how the models plug into the
engine" overview.

### The two-phase protocol that makes it work

The engine dispatches every event to *all* handlers and no handler can veto an
event. That single fact drives the whole design: a work item is **requested**,
then later **started** — never both at once.

```
ProcessComponent  ──ACTIVITY_REQUEST──▶  ResourceComponent   "this item is enabled"
                                              │
                                              │  allocate (see filter below)
                                              ▼
ResourceComponent ──ACTIVITY_START───▶  (everyone, incl. logger)   "a resource holds it; work begins"
                                              │
                                              │  duration sampled, ACTIVITY_COMPLETE scheduled
                                              ▼
ProcessComponent  ──(on complete)────▶  ResourceComponent.release()  ──RESOURCE_AVAILABLE──▶ drain queue
```

- **`ProcessComponent` only ever emits `ACTIVITY_REQUEST`.** A request is
  invisible to every component except `ResourceComponent`, so a queued item
  genuinely cannot run and cannot be logged while it waits.
- **`ResourceComponent` is the only thing that emits `ACTIVITY_START`,** and
  only *after* it holds a resource. That is why `org:resource` is always
  populated in the log, and why waiting time falls out for free as
  `start_time − request_time`.
- If this were skipped and a queued item stayed an `ACTIVITY_START`, the
  ProcessComponent would run it while it sat in the queue *and* again when
  re-scheduled — forking the case. (This was a real bug; see
  `docs/manuals/case_forking_bug.md`.)

### Three injected models, one allocation filter

`ResourceComponent` never hard-codes any resource rule. It takes three
collaborators in its constructor, each swappable without touching the engine:

| Injected model | Section | Question it answers | Default | Basic / Advanced |
|---|---|---|---|---|
| **permission model** (`components/permissions.py`) | 1.7 | *who is allowed?* | OrdinoR org model | `StaticPermissions` (observed matrix, Basic) · `OrgModelPermissions` (OrdinoR, Advanced) |
| **availability calendar** (`analysis/availability.py`) | 1.6 | *who is on shift?* | Yearly calendar | `WeeklyAvailability` (Basic) · `YearlyAvailability` (holidays + vacation, Advanced) |
| **allocation policy** (`policies.py`) | 1.8 | *which allowed one wins?* | `RandomPolicy` | `RandomPolicy` (R-RMA); round-robin / shortest-queue are the Part II upgrade path |

When a request arrives, `_allocate` runs them as a **pipeline** — each stage
only ever narrows the set, so a later stage can never override an earlier one:

```
permission model  →  candidates permitted for (activity, case type, time)
      │  (§1.7)
      ▼
live capacity     →  keep only those with a free slot (_busy < capacity)
      ▼
availability      →  keep only those on shift right now              (§1.6)
      ▼
allocation policy →  pick one of the survivors                      (§1.8)
```

If the survivor set is empty the item is **queued** (Distribution on Enablement,
R-DE) and re-offered the instant a qualified resource frees up via
`RESOURCE_AVAILABLE`. If *nobody* is permitted at all, queuing would strand the
case forever, so the item runs unassigned (`resource=None`) and is counted in
`stats()["unpermitted_activities"]`.

### Why the permission model is richer than a lookup table

The Advanced permission model (OrdinoR, `OrgModelPermissions`) gates on an
**execution context** — the triple *(case type, activity type, time type)* — not
just the activity. So "validate an application" can be permitted *for car loans*
*on weekdays* and not otherwise. The engine can actually enforce all three
dimensions:

- **time type** — `ResourceComponent` already tracks wall-clock time for the
  §1.6 calendar, so `when=` is free.
- **case type** — `components/case_attributes.py` samples a loan goal per case
  and it rides on every event's `payload`, surfacing as `case_type`.
- a wildcard (⊥) on any dimension means "any", and a *missing* value also matches
  as a wildcard (we never deny work just because a field is absent).

> **Integration note (recently fixed).** In the Petri-net process, the case-type
> attribute was previously never written onto the payload, so the org model's
> case dimension silently matched everything. It is now sourced from a single
> per-case draw shared with the §1.5 decision classifiers — see
> `docs/manuals/merge_1.7_plan.md`, item A.

### Batch allocation variants (Section 1.8 Advanced)

Two optional disciplines replace the default one-at-a-time R-RBA pick:

- **Piled Execution** (`--piled-execution`, R-PE / Pattern 38): on each release
  the just-freed resource first grabs a waiting item of the *same* activity type
  it just finished (its "pile"), then falls back to FIFO. One task per release.
- **k-Batching** (`--k-batching K`, Zeng & Zhao): work items *never* allocate on
  arrival; they queue and are released in batches of *K*, solved as a
  parallel-machines assignment (`scipy.optimize.linear_sum_assignment` over
  `expected_duration.py`) minimising total expected processing time. A safety
  valve flushes early if the oldest item has waited too long. Mutually exclusive
  with Piled Execution.

See `docs/manuals/resource_allocation_heuristics.md` for the full pattern
rationale (R-RBA, R-DE, and the upgrade path).

---

## How to run

### First-time setup (clone → add the log → run)

1. Create the project environment and install dependencies into that exact
   interpreter. This also installs `ipykernel`, which VS Code and Jupyter need
   to execute notebook cells:
   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```
2. The **default** run needs **no raw log** — the fitted distributions, BPMN,
   availability model and branching probabilities are all committed:
   ```bash
   .venv/bin/python -m simulation.main
   ```
3. The report's **active-lifecycle ML model** and decision-rule classifier are
   committed, so collaborators can run the notebooks and the active ML modes
   without the raw BPIC-17 log or local retraining. Only the legacy-lifecycle
   processing-time model remains gitignored. To regenerate an artifact
   intentionally, place the log at the repo root as `BPIChallenge2017.xes`
   (or `.xes.gz`, or under `data/`) and run:
   ```bash
   .venv/bin/python setup_models.py            # add --force to retrain
   .venv/bin/python setup_models.py --lifecycle active  # versioned lifecycle artifacts
   ```
   The active wrapper writes `simulation/models/processing_time_model_active.joblib`
   (point + 19 quantile models) and its versioned lifecycle inputs. Artifacts
   that already exist are skipped unless `--force` is passed.

From the repo root, inside the virtualenv:

```bash
.venv/bin/python -m simulation.main                            # advanced process model (default), fitted distributions
.venv/bin/python -m simulation.main --process-model basic      # flat next-activity probability graph (Section 1.4 Basic)
.venv/bin/python -m simulation.main --lifecycle-mode active --mode ml_model          # committed contextual point model
.venv/bin/python -m simulation.main --lifecycle-mode active --mode ml_probabilistic  # committed probabilistic model
.venv/bin/python -m simulation.main --lifecycle-mode active    # W_ active sessions + suspend/resume lifecycle

# Resources (Sections 1.6–1.8)
.venv/bin/python -m simulation.main --availability calendar     # §1.6: fitted per-resource shifts/holidays/vacation (DEFAULT)
.venv/bin/python -m simulation.main --availability always       # §1.6: resources never off-shift (baseline)
.venv/bin/python -m simulation.main --permissions orgmodel      # §1.7: OrdinoR organizational model (Advanced, DEFAULT)
.venv/bin/python -m simulation.main --permissions observed      # §1.7: learned resource×activity matrix (Basic)
.venv/bin/python -m simulation.main --permissions hardcoded     # §1.7: original top-20 map (baseline)
.venv/bin/python -m simulation.main --piled-execution           # §1.8: Piled Execution (R-PE) same-activity batching
.venv/bin/python -m simulation.main --k-batching 5              # §1.8: k-Batching (batch assignment, K=5)

# Section 1.5 rules mode uses the committed decision-rule artifact:
.venv/bin/python -m simulation.main --branching-mode rules      # §1.5 Advanced: data-driven decision points
```

The final line of a run prints the resource-pool summary — availability and
permission model in use, work items started, mean wait for a resource, items
still queued at the horizon, and `activities nobody may perform` (the count of
work items no permitted resource existed for; a nonzero value means the
permission model's case/time gating is genuinely binding).

Output is always saved to `<repo_root>/output/event_log.csv`, regardless of
the working directory you run from.
The legacy `ml_*` modes need a trained artifact — run `setup_models.py` (see
first-time setup above), or see **[Processing-Time Models
(Section 1.3)](simulation/PROCESSING_TIMES.md)** for training, mode
details and reference statistics. Active lifecycle ML modes use the committed
artifact family (regenerated by `setup_models.py --lifecycle active`); the
legacy artifacts are never overwritten. To check whether Basic or Advanced (or any
change you make) better approximates the real BPIC-17 process, run
`scripts/compare_process_models.py` — see
[docs/paper_insights_discovering_simulation_models.md](docs/paper_insights_discovering_simulation_models.md)
for the KPIs it reports and why.

To enable verbose event-by-event output (useful for debugging), set
`verbose=True` when constructing `SimulationEngine` in `main.py`.

---

## The event log (CSV output)

The logger writes a PM4Py-compatible event log. Legacy mode records every
`ACTIVITY_START` and `ACTIVITY_COMPLETE` event using the original five columns:

| Column | Description |
|---|---|
| `case:concept:name` | Case identifier, e.g. `case_000042` |
| `concept:name` | Activity name |
| `time:timestamp` | ISO-8601 datetime (anchored to a configurable start date) |
| `lifecycle:transition` | `start` or `complete` |
| `org:resource` | Resource that executed the activity |

Active mode adds `schedule`, `suspend`, `resume`, `ate_abort`, and `withdraw`
rows for `W_` items plus a sixth `work_item_id` column. Atomic `A_`/`O_` items
retain start/complete pairs at the same timestamp, without a resource assignment.
Use `work_item_id`, not `(case, activity)`, to reconstruct lifecycle sessions.

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
| **1.6** Resource Availabilities | ✅ Done (Basic: fixed capacity per resource; Advanced: fitted shift/holiday/vacation calendar, default via `--availability calendar`) | `components/resource.py`, `analysis/availability.py`, `notebooks/01_resource_availability.ipynb` |
| **1.7** Resource Permissions | ✅ Done (Basic: observed resource×activity matrix via `--permissions observed`; Advanced: OrdinoR organizational model with case/time context, default via `--permissions orgmodel`) | `components/resource.py`, `components/permissions.py`, `analysis/permissions.py`, `notebooks/02_resource_permissions.ipynb` |
| **1.8** Resource Allocation | ✅ Done (R-RBA/R-DE, simple policies, Piled Execution, k-Batching, two assignment policies, and optional masked-PPO DRL) | `components/resource.py`, `policies.py`, `expected_duration.py`, `drl.py` — see [resource_allocation_heuristics.md](docs/manuals/resource_allocation_heuristics.md) and [drl_resource_allocation.md](docs/manuals/drl_resource_allocation.md) |

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

## Deep Reinforcement Learning allocation (optional D3)

The simulator can be paused at allocation decision epochs and trained as a
Gymnasium environment. `MaskablePPO` chooses a feasible `(resource, activity)`
assignment or strategically postpones; contextual permissions, shifts and live
capacity are enforced by the action mask. The normal simulator does not import
PyTorch or Gymnasium.

```bash
uv pip install --python .venv/bin/python -r requirements-drl.txt

PYTHONPATH=. .venv/bin/python scripts/train_drl.py \
  --timesteps 100000 --days 10 --device auto \
  --out models/drl_resource_policy

PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \
  --policies random,drl --drl-model models/drl_resource_policy.zip \
  --seeds 10 --days 10 --warmup-days 2 \
  --process-model advanced --branching-mode visit \
  --permissions orgmodel --lifecycle-mode active \
  --out output/experiments_drl/
```

See [the DRL manual](docs/manuals/drl_resource_allocation.md) for the state,
action, reward, training protocol and the distinction between a smoke model and
report-quality convergence evidence.

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

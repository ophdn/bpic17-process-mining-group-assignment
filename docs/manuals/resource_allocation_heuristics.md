# Resource Allocation — R-RBA (Section 1.8) + Piled Execution (optional)

How the simulation decides **which worker does which task**.

The code lives in `simulation/components/resource.py` (`ResourceComponent`).
Same seed (`42`) = same result every run.

---

## The one heuristic: Role-Based Allocation (R-RBA)

**Plain version:** a worker can only do a task if they're "qualified" for
it. Among the qualified workers who currently have a free slot, pick one
at random at runtime. If no qualified worker is free, the task waits
until one frees up, then gets assigned to them.

That's the whole heuristic. It comes from:

> Nick Russell, Wil M.P. van der Aalst, Arthur H.M. ter Hofstede, and
> David Edmond. *Workflow Resource Patterns: Identification,
> Representation and Tool Support.* CAiSE 2005, LNCS 3520, pp. 216–232,
> Springer, 2005.
> PDF: `docs/papers/optimization_1.1/base resource allocation heuristics.pdf`

In Russell et al.'s pattern catalogue, R-RBA is **Creation Pattern 2**
(*Role-Based Allocation*). It's a *creation* pattern because it's a
design-time restriction on *who is allowed* to do a task — the actual
choice of *which* qualified resource is deferred to runtime. We make
that runtime choice with a uniform random pick (the project's default
behaviour); the fancier *push selection patterns* (R-RMA random,
R-RRA round-robin, R-SHQ shortest-queue) are not implemented — see the
upgrade path at the end.

### What "role" means here

The paper assumes an organisational model with explicit roles
(manager, clerk, …). BPIC-17 has no such model, so we operationalise a
worker's **role** as the set of activities that worker was observed
performing in the real event log. Concretely: `RESOURCE_PERMISSIONS` is
a hardcoded map `worker → set of activities`, mined once from the log.
Its inverse `_ACTIVITY_TO_RESOURCES` (`activity → list of qualified
workers`) is the candidate set R-RBA filters down to.

There's also a second pattern quietly at work — **Distribution on
Enablement (R-DE, Pattern 19)** — which handles the "everyone's busy"
case: instead of dropping the task, we defer it and retry the moment a
qualified worker frees up. R-DE is the saturation / wait-queue fallback
wrapping R-RBA.

---

## The twist: we don't know who'll be free

The paper assumes you always know who's busy. In our simulation that's
**not true**, because:

1. **Task durations are random** — sampled from fitted distributions
   (or an ML model). We can't forecast when a worker finishes.
2. **Each worker can juggle several tasks** (`capacity_per_resource`,
   default 3 in `main.py`), not just one.

So we adapted R-RBA. The key decisions:

- **Check availability at the last second.** Allocation reads the current
  workload *right when the task starts*, not at some earlier forecast.
  Under random durations, the live `_busy` count is the only honest
  signal of who's free.
- **Capacity > 1 → "has a free slot".** A worker counts as available
  when `_busy < capacity` (at least one slot free), not only when
  completely idle.
- **If all qualified workers are busy, the task waits** (not dropped).
  The first worker that frees up and is qualified gets the oldest
  waiting task it's allowed to do. FIFO — no skipping ahead.
- **Re-scheduled tasks aren't re-picked.** A waiting task is pre-bound
  to its freeing worker before it re-enters the queue, so the allocation
  handler skips it (an idempotency guard). Without this, the busy
  counter would double-count and the pool would saturate forever.
- **Reproducible randomness.** The random pick uses a seeded
  `random.Random(42)`, so R-RBA is fully reproducible — required for
  assignment grading.
- **No priorities or deadlines.** R-RBA is purely about role
  qualifications and workload. Smarter features (escalation,
  delegation, early/late distribution) are left as future work.

These decisions are mirrored in the `resource.py` module docstring under
*"Design decisions for uncertain resource availabilities"*.

---

## How a task flows through the code

```
1. Task starts (no worker assigned yet)
       │
2. ResourceComponent filters workers:
   keep only workers whose "role" (RESOURCE_PERMISSIONS)
   includes this task AND who have a free slot (_busy < capacity)
       │
3. Any qualified & free worker?
   ├─ YES → pick one at random, assign it, mark one slot busy
   └─ NO  → put task on the waiting list          ← R-DE deferral
       │
4. Task runs for a (random) duration, then finishes
       │
5. Worker is freed → tell ResourceComponent (schedules RESOURCE_AVAILABLE)
       │
6. ResourceComponent looks at the waiting list (FIFO):
   give the oldest waiting task THIS worker is allowed to do
   back to this worker, then restart it.
   ├─ found one → assign, mark busy, restart task  (→ back to step 4)
   └─ none      → worker stays idle
```

The "wait → retry on free" loop (steps 3 → 6 → 3) is what keeps the
simulation from dropping tasks when the office is overloaded. It's R-DE
wrapping R-RBA.

---

## The data behind it

| Thing | Where | What it is |
|---|---|---|
| Who can do what | `RESOURCE_PERMISSIONS` | Hardcoded map: worker → set of tasks it's "qualified" for (taken from BPIC-17 log). This *is* the R-RBA role definition. |
| Tasks → workers (inverse) | `_ACTIVITY_TO_RESOURCES` | The reverse: task → list of qualified workers. Built once at import. |
| How busy each worker is | `self._busy` | Live count of tasks each worker is running right now. |
| Max tasks per worker | `self._capacity` | `capacity_per_resource` (default 3 in `main.py`). The "free slot" rule is `_busy < capacity`. |
| Waiting tasks | `self._waiting` | FIFO list of tasks that couldn't be assigned yet. Drained when a worker frees up. |
| Random pick | `self._rng` | Seeded `random.Random(42)` for reproducible uniform picks. |

The worker pool is the top-20 BPIC-17 resources (by event count). Extend
`RESOURCE_PERMISSIONS` if you need more workers.

---

## Example: who gets `W_Validate application`?

All 17 workers are qualified for this task. With `capacity = 3`:

- At the start everyone is idle (`_busy = 0`) — all 17 are candidates.
- R-RBA + random pick draws one of the 17 uniformly. Say `User_5`.
- A few seconds later another `W_Validate application` arrives. Still
  all 17 idle (the first task is running but `User_5` has 2 spare slots),
  so again a uniform draw over 17. `User_5` could even be picked again.
- As load builds up, some workers saturate (`_busy = capacity`) and drop
  out of the candidate list. If **all 17** ever saturate at once, the
  next task goes on `_waiting`; the moment any qualified worker
  finishes, the oldest waiting task is handed to them.

---

## Piled Execution (R-PE, Pattern 38) — optional batch allocation

**Plain version:** when a worker finishes a task of type T, before
becoming generally available they first look at the waiting list for
another task of the **same type T**. If one is there, they grab it
immediately (one task per release). Only if no same-type task is
waiting do they fall back to the normal "oldest waiting task I'm
qualified for" rule.

This is **Piled Execution**, Russell et al. *Auto-start Pattern 38
(R-PE)*: "initiate the next instance of a workflow task once the
previous one has completed … allocating similar work items to the same
resource which aims to undertake them one after the other" — reduced
set-up time / familiarity from sticking with one task type.

It's a **batch allocation approach** because same-type tasks cluster
onto one worker rather than being scattered uniformly across the pool.
But it's deliberately **sequential** — only one same-type task is
handed off per release, faithful to the paper's "the next one".

### What changed in the code

- The wait-queue drain (R-DE) gains a same-type-first scan *before* the
  FIFO scan — only when `--piled-execution` is on.
- Synchronous allocation (`_allocate`) is **unchanged** — new tasks
  are still picked uniformly at random among qualified+free workers.
  Piled Execution only biases the **deferred** path.
- `release()` now carries the just-finished activity so the freeing
  worker knows what "same pile" to look for.
- Processing times are **not** affected. Only *which* worker does
  *which* waiting task changes.

### Design decisions for uncertain availabilities

Piled Execution is layered on the same live-load / R-DE machinery as
R-RBA, so the uncertain-availability adaptations from earlier still
hold. The pile is formed from the **current** waiting list (not a
future queue forecast), and one same-type handoff per release keeps it
O(1) and deterministic under the seed. If the same-type waiter isn't
qualified (it can't be — the worker just finished that activity, so it
must be permitted), we don't try; but this never happens because
`RESOURCE_PERMISSIONS` guarantees the freeing worker is permitted for
the activity it just ran.

Default off → current R-RBA-only behaviour is preserved bit-for-bit.

---

## Usage

R-RBA is always on; Piled Execution is opt-in:

```bash
# default — R-RBA only
.venv/bin/python -m simulation.main
.venv/bin/python -m simulation.main --process-model basic

# turn on Piled Execution (same-type batching on the wait queue)
.venv/bin/python -m simulation.main --piled-execution
.venv/bin/python -m simulation.main --process-model basic --piled-execution
```

Or programmatically:

```python
from simulation.components.resource import ResourceComponent

# R-RBA only (default)
resources = ResourceComponent(capacity_per_resource=3, seed=42)

# R-RBA + Piled Execution
resources = ResourceComponent(capacity_per_resource=3, seed=42, piled=True)
```

To see the effect, compare consecutive complete rows for the same worker
+ same activity before/after turning the flag on:

```bash
# count "piled" pairs: same worker did the same activity back-to-back
tail -n +2 output/event_log.csv | awk -F, '
  $4=="complete" && $5==prev_r && $2==prev_a {c++}
  {prev_r=$5; prev_a=$2}
  END {print c" back-to-back same-worker+same-activity pairs"}'
```
The piled run should show a higher count.

---

## Reference run

30-day horizon, `RANDOM_SEED = 42`, `capacity_per_resource = 3`,
`--process-model basic` (the flat next-activity graph, to avoid the
`pm4py` dependency needed by the advanced Petri-net path). Terminates
without deadlock and writes `output/event_log.csv`.

> The run stats below are illustrative — exact counts depend on the
> processing-time mode and process model selected. Reproducibility is
> guaranteed: same seed + same settings = identical CSV (verified with
> matching MD5 across repeated runs).

---

## Where to look next

- `simulation/components/resource.py` — the implementation (module
  docstring reproduces the design-decision section for in-code context).
- `simulation/main.py` — the CLI runner; `ResourceComponent` is
  instantiated there with `capacity_per_resource=3, seed=42`.
- `simulation/components/process.py` — the component that calls
  `resources.release(...)` on `ACTIVITY_COMPLETE` and reads
  `event.resource` for ML duration features.
- `simulation/core/engine.py` — the thin DES router that dispatches
  events to `ResourceComponent` (registration order matters:
  resources must be registered *before* the process component).
- `docs/papers/optimization_1.1/base resource allocation heuristics.pdf`
  — the original Russell et al. paper with the full pattern catalogue.

---

## Upgrade path (not done here)

R-RBA answers **"who is allowed"**. It deliberately does **not** answer
**"which of the allowed ones"** beyond a uniform random pick — that's
where the push *selection* patterns come in. Future work:

- **Section 1.6 Advanced** — calendar / shift-based availability: gate
  allocation on whether the resource is on-shift at `engine.now`, not
  just on `_busy < capacity`.
- **Section 1.7 Advanced** — role-discovery (e.g. OrdinoR) to replace
  the hardcoded `RESOURCE_PERMISSIONS` with a mined organisational
  model.
- **Section 1.8 Advanced — push selection patterns** (replace the
  random pick in `_allocate`):
  - *R-RMA* (Random, pat. 15) — what's currently implemented.
  - *R-RRA* (Round Robin, pat. 16) — take turns cycling through
    qualified workers.
  - *R-SHQ* (Shortest Queue, pat. 17) — give the task to whoever has
    the least on their plate right now.
- **Detour patterns** (R-D delegation, R-E escalation, R-SD
  deallocation, R-PR/R-UR reallocation) — for handling exceptions such
  as resource unavailability mid-execution.
- **Auto-start patterns** — *R-PE* (Piled Execution, pat. 38) is now
  implemented behind `--piled-execution` (see the dedicated section
  above). *R-CE* (Chained Execution, pat. 39) — pipeline sequential
  work items within the same case to the same resource — remains
  future work.
- **Early/Late Distribution** (R-ED / R-LD) — timing variants that
  allocate before/after enablement rather than at enablement.
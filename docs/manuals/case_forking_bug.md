# The case-forking bug on `main` (and the fix)

**TL;DR** — `main` silently duplicated cases. A 30-day run produced **40,764
case-completions from 2,318 cases** (17.6× too many), logged **751,488 events**
where it should log ~30,000, and left one case with **134,845 events**. It also
crashed the default `python main.py`, and wrote an *empty* `org:resource` on
every `start` row. All four were one root cause. Fixed in `feature/resource-management`
(commit `3f2acf2`).

If you have run experiments against `main`'s event log, the numbers are not
trustworthy — please re-run.

---

## The root cause

The engine dispatches each event to **every** registered handler, unconditionally:

```python
for handler in self._handlers.get(event.event_type, []):
    handler(self, event)
```

There is no way for a handler to veto, consume, or swallow an event. Every
handler registered for `ACTIVITY_START` *will* see every `ACTIVITY_START`.

`ResourceComponent` was written as if that were not true. When no qualified
resource was free, it "deferred" the work item by pushing it onto a wait queue:

```python
resource = self._allocate(event.activity)
if resource:
    event.resource = resource
    self._busy[resource] += 1
else:
    self._waiting.append((engine, event))   # <-- does NOT stop the event
```

But the event carries on to the next handler in registration order —
`ProcessComponent.on_activity_start` — which samples a duration and schedules
`ACTIVITY_COMPLETE`. **The activity runs anyway, while supposedly queued.**

Then, when a resource frees up, `on_resource_available` re-schedules *the same
event object*. It is dispatched a second time. `ProcessComponent` handles it
again, schedules a second `ACTIVITY_COMPLETE`, and the case advances down a
second, independent chain.

Every deferral forks the case. With capacity saturated, deferrals are the common
path, so cases fork repeatedly and the log explodes.

### Why it also crashed the Petri net

`petri_process.py` threw `KeyError: 'case_000033'` on the default path. That was
a *symptom*, not a separate bug: a duplicate event arrived for a case whose
marking had already been popped at completion. Fixing the fork fixed the crash.

### Why `org:resource` was empty

The engine registers its built-in logger *first*, before any component. So for an
`ACTIVITY_START` the order is:

1. `logger.log(event)` — writes the row, with `event.resource` still `None`
2. `ResourceComponent.on_activity_start` — *now* sets `event.resource`

The logger had already written the row. Every `start` row had an empty
`org:resource`; only 10.4% of events carried a resource at all.

---

## The fix: request ≠ start

Introduce `ACTIVITY_REQUEST` ("this work item is enabled") as distinct from
`ACTIVITY_START` ("a resource holds it and work has begun").

- `ProcessComponent` only ever emits **requests**.
- `ResourceComponent` is the **only** component that turns a request into a
  start, and only once allocation has succeeded.
- Nothing else handles `ACTIVITY_REQUEST`, so a queued work item is genuinely
  stalled: it cannot execute, and it cannot be logged.

```
        ProcessComponent                ResourceComponent
              │                                │
              │  ACTIVITY_REQUEST              │
              ├───────────────────────────────►│  allocate?
              │                                │    ├── yes ──► ACTIVITY_START (resource bound)
              │                                │    └── no  ──► queue; wait for RESOURCE_AVAILABLE
              │  ACTIVITY_START                │
              │◄───────────────────────────────┤
              │  (sample duration, schedule ACTIVITY_COMPLETE)
```

Three things fall out of the same change for free:

- **`org:resource` is populated by construction** — the resource is bound before
  the event the logger ever sees is created.
- **Waiting time becomes measurable** as `start_time − request_time`. This is
  real queueing data, and it feeds §1.3 Advanced II (processing vs waiting time).
- **An activity no resource may perform** runs unassigned and is counted, rather
  than queueing forever and stranding the case.

## Results

| | before (`main`) | after |
|---|---|---|
| events logged (30 days) | 751,488 | 30,207 |
| cases started / completed | 2,318 / **40,764** | 2,318 / 916 |
| worst case, events | **134,845** | 94 |
| duplicate dispatches | 7,711 (one case) | **0** |
| rows with `org:resource` | 10.4% | **100%** |
| `python main.py` (Petri net) | **crashes** | runs |

## What this means if you touch the engine

The general lesson: **a component cannot stop an event by not acting on it.**
If your component needs to block, defer, or veto a work item, it must do so by
*not emitting the event that triggers the downstream work* — not by
intercepting an event that other handlers are already subscribed to.

If you genuinely need veto semantics in the engine, the alternative would be a
`handled` / `cancelled` flag on `SimEvent` that `engine.run()` checks between
handlers. We did not add one, because the request/start split solves the problem
without giving every component the power to silently swallow events.

# Plan: wire `p_work` into the runtime availability model

**Status:** open, decision pending with the team.
**Owner:** Johannes.
**Written:** 2026-07-16, for a cold start in a later session.
**Blocks:** the validation paragraph of the resource subsection in
`docs/report_resources_draft.tex` (see the TODO block in that file), and
potentially every Part II number.

---

## 1. The finding

The availability model we **validated** and the model the simulation **runs**
are not the same model.

`p_work[resource][weekday]` (the probability a resource works that weekday at
all) is fitted in `analysis/availability.py:fit_weekly`, serialised into
`models/availability_model.json`, and used to derive the per-resource vacation
threshold. **Nothing at runtime ever reads it.**

- Notebook `01_resource_availability`, cell 30 (the headcount validation) rolls
  it explicitly: `rng.random() < plan.weekly.p_work[r][dow]`. That is where the
  reported "37.3 modelled vs 40.2 real staff per weekday, 7.4% light" comes
  from.
- `YearlyAvailability.is_available()` (`analysis/availability.py:376`) checks
  **holiday, vacation, window** and nothing else.

### Evidence (reproduce with)

```python
import sys; sys.path.insert(0, '.')
from analysis.availability import YearlyAvailability
import datetime as dt
plan = YearlyAvailability.from_json('models/availability_model.json')
mon = dt.datetime(2016, 6, 6, 10, 0)          # a Monday, 10:00
avail = [r for r in plan.weekly.resources() if plan.is_available(r, mon)]
print(len(avail))                              # -> 123
print(sum(plan.weekly.p_work.get(r, {}).get(0, 0.0) for r in avail))   # -> 36.5
```

The deployed calendar puts **123** people on shift on a Monday morning. Rolling
`p_work` gives an expected **36.5**, which is the number that matches the real
**40.2**. The deployed workforce is therefore roughly **3.3x overstaffed**.

Consequence for the report: the current claim that the residual is "a
conservative error, the simulation is slightly *more* resource-constrained than
reality" is **backwards** for the deployed model.

---

## 2. Why it happened (the architectural reason)

There is no resource pool construction anywhere in the runtime, so `p_work` had
no seam to enter through.

1. **Pool membership comes from the permission model, not the calendar.**
   `simulation/components/resource.py:389-391`:
   ```python
   self._busy: Dict[str, int] = {r: 0 for r in self._permissions.resources()}
   ```
   The Section 1.6 calendar contributes nobody to the pool. It only gates
   people who are already in it.

2. **Availability is decided per allocation attempt, not per day.**
   `_allocate` (`resource.py:806-835`) takes the permission model's candidates
   for one activity and filters them three ways: live capacity
   (`_busy < _capacity`), `_is_on_shift`, and `not in _excluded`.

3. So the calendar is never asked *"who is working today?"*. It is only ever
   asked, one resource at a time, *"is `r` on duty at this instant?"*.

`p_work` is a property of a **day**. `is_available` is a question about an
**instant**. The notebook builds a roster (loop days, loop resources, roll
dice, count); the simulation evaluates a pointwise predicate. There was no
day-shaped thing at runtime for `p_work` to attach to.

---

## 3. The fix

### The one idea it turns on

`is_available(r, t)` is called many times per simulated day (once per
allocation attempt). A fresh random draw per call would make a resource flicker
on and off *within* a day, which is worse than the current bug. **The draw must
be deterministic given `(resource, date)`: a hash, not an RNG stream.**

A hash also means independent call sites (see 3.2) can each ask and agree for
free, which an RNG stream could never guarantee.

### 3.1 `analysis/availability.py`

Add a `roster_seed: int = 42` field to `YearlyAvailability` and a helper:

```python
def _works_today(self, resource: str, d) -> bool:
    """Is `resource` rostered on at all on date `d`?

    p_work is a per-(resource, weekday) probability. The draw is a hash of
    (roster_seed, resource, date), not an RNG stream: is_available() is called
    many times per simulated day and must return the same answer every time
    within one day, while staying independent across days and reproducible
    across runs.
    """
    p = self.weekly.p_work.get(resource, {}).get(d.weekday(), 0.0)
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    h = hashlib.blake2b(
        f"{self.roster_seed}|{resource}|{d.isoformat()}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big") / 2**64 < p
```

Then `is_available()` gains one check alongside holiday and vacation:

```python
if not self._works_today(resource, d):
    return False
```

Decide whether `roster_seed` is serialised in `to_json` / `from_json`. Probably
**not**: it is a run parameter rather than a fitted parameter, so it belongs
with the simulation seed. If so, it must be settable when the calendar is
constructed in `simulation/main.py`.

### 3.2 `simulation/components/resource.py` (easy to miss, will stall the sim)

`_next_shift_open` (`resource.py:751-781`) scans forward up to 14 days for the
next time some resource's window opens, to schedule a wake-up. It already skips
holidays (`:765`) and vacations (`:773`). It **must** also skip rostered-off
days, or the simulation wakes expecting staff who are not working and stalls
with items still queued.

Add the same `_works_today` check to the inner loop at `:771-773`.

### 3.3 Backward compatibility

`_is_on_shift` (`resource.py:669-697`) already has a legacy path for calendars
that predate the `system` set. Follow the same pattern: a calendar without
`p_work` (or with `roster_seed=None`) should keep the current always-on-within-
window behaviour, so old evidence logs stay reproducible.

Consider gating the whole thing behind a flag (`--roster` / `roster=True`,
default **off** initially), matching how `crn` was introduced in Section 2.1.
That lets Part II be re-run deliberately rather than having every historical log
change underfoot.

---

## 4. Verification

1. **Unit:** `_works_today` is stable within a day (same answer for 09:00 and
   16:00 on the same date), varies across dates, and reproduces exactly across
   processes for a fixed `roster_seed`.
2. **Distribution:** over the 2016 span, the share of days a resource is
   rostered on converges to its `p_work`. Tolerance a few percent.
3. **The headline check:** re-run the evidence snippet in section 1. On Mon
   2016-06-06 10:00, `is_available()` should now put **~37** resources on shift,
   not 123.
4. **Second pass:** the simulated weekday headcount should reproduce notebook 01
   cell 30 (mean ~37.3, sd ~6.8 against real 40.2 / 7.2). This is the point of
   the whole exercise: the runtime and the validated model finally agree.
5. **No stalls:** run the full simulation and confirm no work item waits
   forever. Watch `stats()` for `_unpermitted` vs genuinely queued items, and
   check that a rare activity whose entire candidate set is rostered off for a
   day still gets picked up (the 14-day wake-up horizon should cover it).

---

## 5. Fallout (this is not cosmetic)

Staff on shift on a given Monday drops from ~123 to ~37. **Contention becomes
real for the first time.** Expect cycle time, waiting time and occupation to
move, possibly a lot.

- **Every Part II number is downstream of this**, including Mario's k-Batching
  table already written into `report/main.tex`. The k-Batching result ("every k
  increases mean cycle time relative to immediate allocation") is exactly the
  kind of finding that could flip once queues genuinely form, so it must be
  re-run, not patched.
- **`scripts/opt_metrics.py:average_resource_occupation`** takes
  `availability_seconds` (resource -> available seconds in the horizon) as the
  denominator for the slide-21 definition. Once `p_work` is live, those
  window-hours must only count days the resource is actually rostered on, or
  occupation is divided by a workforce that never showed up. Whatever builds
  that mapping needs the same `_works_today` logic.
- The 4h k-Batching safety valve and the R-DE deferral queue become far more
  load-bearing than they are today.
- The management question ("fire two employees") only becomes meaningful once
  contention exists. Under the current 3.3x overstaffed pool, removing two
  people would show almost no effect, which would have been a misleading answer.

---

## 6. Report decision (pending)

The TODO block in `docs/report_resources_draft.tex`, immediately before the
"conservative error" sentence, holds both options:

- **Option A (preferred):** implement this plan. The validation paragraph and
  the conservative-error sentence then both stand as written, and Section 1.6
  describes the model that actually runs.
- **Option B:** report as a known limitation. Delete the conservative-error
  sentence and state plainly that the deployed calendar models opening hours,
  holidays and leave but **not** the part-time rhythm, so Part II runs under a
  workforce roughly 3x larger than the real one and its contention-based results
  are optimistic rather than conservative.

**This is a team call, not just Johannes's**, because Option A invalidates
Mario's k-Batching table and any policy comparison already run. Raise it with
Mario and Daniel before implementing. If the re-run cost is affordable, Option A
is clearly right: Option B leaves the central Part II claims resting on a
workforce that does not exist.

---

## 7. Suggested order for the next session

1. Confirm the team decision (A or B). If B, stop here and edit the draft.
2. Implement 3.1 and 3.2 behind a default-off flag.
3. Run verification 1 to 4. Stop if headcount does not land near 37.
4. Fix the `opt_metrics` denominator (section 5).
5. Run verification 5 on a full simulation.
6. Flip the default on, re-run the Part II grid, and hand Mario the new
   k-Batching numbers.
7. Update `docs/report_resources_draft.tex`: remove the TODO block, keep the
   validation paragraph, and note in Section 1.8 that the calendar now rosters.

# Plan: wire `p_work` into the runtime availability model

**Status:** Option A implemented behind `--roster-seed`, default **off**.
Sections 3.1, 3.2, the `opt_metrics` denominator and verification 1-5 are done.
**Not** yet done: flipping the default on and re-running the Part II grid
(section 7 steps 6-7), because a new finding may change what that re-run is
worth — see section 8.
**Owner:** Johannes.
**Written:** 2026-07-16, for a cold start in a later session.
**Updated:** 2026-07-17 (team chose Option A; implemented; section 8 added).
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

---

## 8. What implementing it actually found (2026-07-17)

Steps 1-5 are done. Step 3 landed: **38-39 on shift** on Mon 2016-06-06 10:00
against the 123 the deployed calendar used to field. No stalls (verification 5).

### 8.1 A third call site, not two

The `refac/event_lifecycle` merge added
`scripts/run_experiments.py:availability_seconds_per_resource`, a third
independent reimplementation of the shift logic (its own holiday, vacation and
window checks, bypassing `is_available`). It is the occupation denominator this
plan's section 5 anticipated abstractly. Over 90 days it read 94,624
resource-hours where the roster gives 25,375 — occupation was being divided by a
workforce ~3.7x larger than the one that showed up.

The hash design paid off exactly as section 3 hoped: the lifecycle refactor also
added two *new* `_is_on_shift` call sites (`resource.py:582`, `:636`, the
suspend/resume path) and both were covered for free, because they funnel through
`is_available`.

### 8.2 `roster_seed` must not be a constant under CRN

The roster is a condition of the run, not a property of the policy. Two policies
at the same replication seed must face the identical workforce or the paired
comparison measures roster luck. `run_once` therefore uses `roster_seed + seed`
as the effective seed: constant across policies within a replication, varying
across replications.

### 8.3 Occupation was nan, now fixed

A part-time resource can draw zero rostered days in a short horizon, so its
denominator is 0 (30 of 144 resources over 14 days). `busy/0` poisoned
`avg_resource_occupation` to nan. Zero-availability resources are now dropped as
*undefined* rather than counted as 0 — counting them as idle would drag the mean
down in proportion to how much rostering-off the horizon contains.

### 8.4 THE OPEN QUESTION: `capacity_per_resource=3` re-inflates the workforce

**Section 5 predicted cycle time would move "possibly a lot". It barely moved.**
Over 30 days, mean wait for a resource went 9.3h -> 9.2h and completed cases
1102 -> 1059. Occupation *tripled* (0.51 -> 1.67 over 14 days, matching the 3.3x
overstaffing), but queues did not explode.

The reason: `capacity_per_resource=3` (`main.py:189`,
`run_experiments.py:207,226`). Each of the ~39 rostered people runs up to **3**
work items concurrently, so effective capacity is ~117 slots and occupation 1.67
sits at only 56% of the ceiling of 3. Resources are still not the binding
constraint.

So the 3.3x overstaffing was not removed — it **moved from headcount into
concurrency**. Right-sizing the roster while each person does the work of three
leaves Part II roughly as uncontended as before.

This also means "occupation" is not a share: with capacity 3 its ceiling is 3,
not 1, so the slide-21 definition (a fraction ≤ 1) does not hold as reported.
18 resources already exceed 1.0 with rostering *off* — this is pre-existing and
documented at `opt_metrics.py:165`, not caused by rostering, but rostering makes
it unmissable (56 resources over 1.0, max 22.7).

**Decide before re-running the grid (section 7 step 6):** is `capacity=3`
defensible? If a caseworker really juggles ~3 applications, then the roster fix
alone will not make contention bind and the k-Batching re-run will likely
reproduce its current finding. If `capacity=3` was a convenience that silently
compensated for the 3.3x-too-small *per-person* load implied by the overstaffed
pool, then it and `p_work` were double-counting the same slack, and capacity
should drop toward 1 — which is the change that would actually make Part II's
contention results mean something. Re-running the grid before settling this
risks burning the re-run on the wrong model.

### 8.5 RESOLVED: capacity is now derived from the lifecycle mode

**Decision (Johannes, 2026-07-17): `capacity=1` in active mode**, on the grounds
that it is the only value the real log supports. Implemented as
`resource.capacity_for_mode()`: **active -> 1, legacy -> 3**. `--capacity N`
overrides on both CLIs.

It must be *mode-derived*, not one global number, because the same value means
opposite things:

| | what a duration is | real concurrency | capacity |
|---|---|---|---|
| active | one hands-on session, median 0.8–2.7 min; `suspend` releases | 98.4% of busy time is a **single** session | **1** |
| legacy | whole elapsed `start→complete` span, mostly suspended waiting | median peak **54** overlapping spans (max 150) | 3 (historical, kept) |

Legacy's 3 is *not* defended as correct — the honest legacy value is nearer 54.
It is kept only so existing evidence reproduces. `capacity=1` in legacy mode
would pin a person to one application for hours and collapse throughput ~50x,
which is why `capacity_for_mode` fails safe to the legacy value for any
unrecognised mode.

Note the duration model has **no concurrent-load feature** (the eight ML
features are activity, resource, previous activity, weekday, hour, case
position, case age, prior-activity count), so N concurrent items each finish as
fast as one. Capacity multiplies throughput for free, with no context-switching
penalty. That is what made it a modelling assumption rather than a tuning knob.

### 8.6 What capacity=1 changes (active, 14d, seed 1, random policy)

| cap | roster | occupation | >1.0 | max | cycle (d) | p95 (d) | completed |
|----:|---|---:|---:|---:|---:|---:|---:|
| 3 | off | 0.412 | 33 | 1.97 | 0.79 | 5.75 | 711 |
| 3 | on | 0.718 | 35 | 2.73 | 0.68 | 2.05 | 634 |
| 1 | off | 0.196 | 0 | 0.70 | 0.58 | 1.87 | 629 |
| 1 | on | 0.317 | **0** | 0.94 | 1.23 | 4.16 | 545 |

1. **Occupation is a share again.** At capacity 1 nothing exceeds 1.0. At
   capacity 3, 33–35 resources did, max 2.73 — so every occupation number
   reported so far was on a scale whose ceiling was 3, not 1, and the slide-21
   definition did not hold.
2. **The roster only bites at capacity 1.** Turning it on moves cycle time
   0.58 -> 1.23 d and p95 1.87 -> 4.16 d. At capacity 3 the same change looked
   like nothing (0.79 -> 0.68). `capacity=3` was masking the entire p_work fix.
3. **Caveat — survivorship.** Cycle time is over completed cases only, so a run
   that completes fewer cases is biased *low*. That discredits the capacity-3
   rows (fewer completed AND faster = survivorship, not improvement) and
   *strengthens* capacity-1 (fewer completed and still slower).
4. Mean occupation is still only 0.317 with max 0.94: the workforce is not
   globally saturated, but the load is **uneven** — consistent with the §1.6
   Tier A/Tier B split. For "fire two employees" the interesting question is
   therefore *which* two, not whether two.

Single seed, single policy, 14 days: directional, not a result.

### 8.7 Still open

- Re-run the Part II grid under active + `capacity=1` + roster on, and hand
  Mario the new k-Batching numbers (section 7 step 6). Not started.
- Legacy's `capacity=3` remains indefensible-but-unchanged. Either justify it,
  raise it toward the observed ~54, or retire legacy mode from Part II.
- `--roster-seed` still defaults **off**. Turning it on is step 6's job.
- Section 7 step 7 (report draft) not started.

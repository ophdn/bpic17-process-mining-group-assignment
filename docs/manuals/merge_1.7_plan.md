# Merging §1.7 (resource permissions) with the Part II optimization work

Status of `feature/resource-permissions` after merging `origin/main` (31 commits).
This documents what was resolved, why, and **what still needs doing**.

---

## Why there were conflicts at all

Both sides descend from the same base (`8a3b911`), which already contains the
`ACTIVITY_REQUEST` protocol — so the case-forking fix was never in dispute. The
conflicts are two teams extending *the same allocation code* with *different,
orthogonal features*:

| | §1.7 (this branch) | Part II (main) |
|---|---|---|
| `resource.py` | permission model injected behind `PermissionModel` | `AllocationPolicy` seam, Piled Execution, k-Batching, resource exclusion |
| `process.py` | `case_attributes` kwarg | `crn` kwarg (Common Random Numbers) |
| `petri_process.py` | `**kwargs` forwarding | `branching_mode`, `decision_rules_path` (§1.5 Adv I) |
| `main.py` | `--permissions` | `--branching-mode`, `--piled-execution`, `--k-batching` |

Nothing is conceptually incompatible. The two seams answer *different questions*
about the same decision:

- **permission model** — *who is qualified* for this work item?
- **allocation policy** — *which* of the qualified, free, on-shift resources gets it?

They compose: permission filters the candidate set, the policy picks from it. A
policy can never see a candidate the permission filter rejected, so it cannot
violate R-RBA even by accident.

---

## What was resolved, and how

### 1. `petri_process.py` — signature forwarding (kept both)

Main restated `ProcessComponent`'s constructor signature; this branch had changed
it to forward `**kwargs`, precisely because copying a parent's signature breaks
the moment the parent gains an argument (it did, when `case_attributes` was
added). Resolution keeps main's Petri-specific params (`branching_mode`,
`decision_rules_path`) explicit and forwards everything else:

```python
def __init__(self, bpmn_path, branching_mode="probs", decision_rules_path=None, **kwargs):
    ...
    super().__init__(**kwargs)
```

### 2. `process.py` / `main.py` — kwargs and CLI flags (kept both)

Purely additive. `case_attributes` and `crn` coexist; `--permissions` sits
alongside `--branching-mode` / `--piled-execution` / `--k-batching`.

### 3. `resource.py` — routed main's new paths through the permission model

Main's Piled Execution and k-Batching code read the **old module-level globals**
(`RESOURCE_PERMISSIONS`, `_ACTIVITY_TO_RESOURCES`) that this branch replaced with
the injected model. Left as-is, batched allocation would have ignored the
permission model entirely. Four call sites were re-routed:

| site | was | now |
|---|---|---|
| `on_activity_request` (k-Batching branch) | `_ACTIVITY_TO_RESOURCES.get(...)` | `self._qualified(engine, event)` |
| `on_resource_available` (R-DE drain) | `RESOURCE_PERMISSIONS.get(resource)` | `self._permissions.permits(...)` |
| `_free_resources` (batch pool) | `for r in RESOURCE_PERMISSIONS` | `self._permissions.resources()` |
| `_assign_batch` (cost matrix) | `req.activity in RESOURCE_PERMISSIONS.get(r)` | `self._permissions.permits(r, ..., case_type, when)` |

Also `_allocate` now takes the `SimEvent` rather than a bare activity string — it
needs the payload to resolve the case type — so main's
`self._policy.select(activity, ...)` became `self._policy.select(event.activity, ...)`.

### 4. A real bug fixed in Piled Execution

Main's piled fast-path skipped the permission check with the comment *"It is
permitted by construction (it just ran one), so no permission check needed."*

That was true against a flat `resource -> activities` map. It is **false under an
OrdinoR model**, where a capability is an execution context
`(case type, activity type, time type)`. A resource that just handled
`W_Validate application` for a *car* loan is not thereby permitted the same
activity for a *boat* loan — same activity, different context, possibly different
answer. The piled path now checks permission for the specific waiting item's
context.

---

## OPEN — must be resolved before this is correct

### A. Duplicate case-attribute sampling — **RESOLVED**

Both sides independently sampled a loan goal per case:

- **main**: `PetriNetProcessComponent._sample_case_attributes()` → `self._case_attrs[case_id]`,
  consumed by the §1.5 decision-point classifiers ("rules" branching mode).
- **this branch**: `CaseAttributeSampler` → `self._ctx[case_id]["attrs"]` → event payload,
  consumed by the §1.7 permission model.

In `--branching-mode rules` **a case therefore carried two independently drawn
loan goals**: the classifier branched on one, the permission check gated on the
other.

**What the fix revealed:** the situation on the Petri path was worse than
"two disagreeing draws" — the Petri `__PROCESS_START__` override never put
`"attrs"` into `self._ctx` at all, so `_payload()` returned `{}` on **every
Petri run**, in every mode. `_matches()` treats a missing case type as a
wildcard, so the org model's case dimension has silently never been enforced
in simulation (which also means the k-Batching `unpermitted: 4` of item D was
measured *without* case-type gating — expect it to change).

**Resolution (as proposed, one commit, revertable):**

- `PetriNetProcessComponent._payload()` override: when `self._case_attrs` has
  an entry ("rules" mode), it is the single source of truth — `case_type` is
  derived from *that* draw (`CT.` + loan goal, same scheme as the parent).
  The classifiers and the permission model can no longer disagree.
- In non-"rules" Petri modes, `__PROCESS_START__` now threads the injected
  `CaseAttributeSampler` into `self._ctx["attrs"]` exactly like the parent
  does — fixing the always-empty payload.
- In "rules" mode the `CaseAttributeSampler` is not drawn at all: the
  duplicate draw is gone.

RNG note: `CaseAttributeSampler` owns a private `random.Random`, so sampling
it in visit/probs modes does not perturb the process RNG stream; runs without
`--permissions orgmodel` (sampler is None) are bit-for-bit unchanged.

### B. CRN does not cover case-attribute sampling

`process.py`'s docstring already flags that case/offer-attribute sampling is out
of CRN's scope. With permissions now conditioned on the case type, that scope
limit has teeth: under `--permissions orgmodel`, a case's loan goal affects *which
resource is allowed to act*, so a non-CRN attribute draw is a cross-policy
divergence source in exactly the paired experiments CRN exists to protect.

Worth deciding whether the loan-goal draw should move under CRN
(`_draw_rng(case_id, "case_attrs")` would do it).

### C. Verify the experiment runner still holds

`scripts/`'s experiment runner (opt-B) compares allocation policies. It should be
re-run to confirm the permission model doesn't perturb its paired comparisons —
and to decide whether `--permissions` becomes another experimental factor or is
pinned to one setting across the policy sweep.

### D. `unpermitted_activities: 4` under k-Batching

The k-Batching run reports 4 work items nobody could perform, where the
non-batching paths report 0. Likely the batching branch's `_qualified` check runs
before a case's payload is populated, or a context genuinely has no candidate at
that moment. Small, but it should be understood rather than tolerated.

---

## Verification status — INCOMPLETE, read before trusting this branch

The merge was committed with verification unfinished (end of a working session).
What is actually known:

| configuration | status |
|---|---|
| all four files parse; no conflict markers | **verified** |
| `--k-batching 5` | **verified runs** — 996 cases completed, but `unpermitted: 4` (item D) |
| default (orgmodel + visit + Petri) | **hit `NameError: activity`, fix applied, NOT re-run** |
| `--piled-execution` | **hit the same `NameError`, same fix, NOT re-run** |
| `--permissions observed` / `hardcoded` | **not run** |
| `--branching-mode rules` | **not run** — blocked on item A |

The `NameError` came from `_allocate` taking a `SimEvent` (this branch) while
main's body still called `self._policy.select(activity, ...)`. One-line fix
applied (`event.activity`); the same class of error may exist elsewhere in paths
that were not exercised. **First job next session: run the matrix above.**

The §1.7 evidence (notebook 02) was produced *before* this merge and is unaffected
by it: it measures permission models against the log, not against the allocation
policy. It does not need regenerating.

---

## Recommendation

Do **not** merge this to `main` until item A is resolved — a case with two
different loan goals is a silent correctness bug that would only show up as
slightly-wrong branching or slightly-wrong permissions, which is the worst kind.
Items B–D are follow-ups that can land separately.

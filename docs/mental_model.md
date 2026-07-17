# Mental model: how the simulation engine fits together

**Audience:** the whole team. This is the *conceptual* map — enough to reason
about how your module interacts with everyone else's. Each module owner still
needs the deep knowledge of their own component; this document deliberately
stops at the seams.

**Written:** 2026-07-17.

Ownership (from `docs/ROADMAP.md`): **Mario** 1.2 arrivals + experiment runner +
k-Batching · **Daniel** 1.3 processing times + Park & Song · **Sophie** 1.4/1.5
process model + branching + policy interface · **Johannes** 1.6–1.8 availability,
permissions, allocation.

---

## 1. The shape of the thing

It is a **discrete-event simulation**. `SimulationEngine` holds a priority queue
of `SimEvent`s ordered by timestamp, pops the earliest, advances the clock to it,
and hands it to whoever registered for that event type.

Two consequences worth internalising:

- **There is no orchestrator.** No component calls another. They communicate
  *only* by scheduling events. The engine itself does almost nothing — it counts
  statistics and forwards events to the logger.
- **A component is anything with a `HANDLES` dict** mapping `EventType` to a
  handler. That is the entire contract (`engine.register`). This is why new
  policies and models plug in without touching the core.

Time only moves forward, and only to the next scheduled event. Nothing polls.

## 2. The vocabulary

`simulation/core/events.py` defines the event types. The ones that matter:

| Event | Meaning |
|---|---|
| `CASE_ARRIVAL` | a new case exists |
| `ACTIVITY_REQUEST` | a work item is enabled and needs a person (logged `schedule`) |
| `ACTIVITY_START` | work begins; a resource is held |
| `ACTIVITY_SUSPEND` | a session pauses; **the resource is released** |
| `ACTIVITY_RESUME` | a suspended item is running again (only after re-allocation) |
| `ACTIVITY_COMPLETE` | the work item is done; resource released |
| `ACTIVITY_ABORT` | the work item is killed (`ate_abort`); **the case continues** |
| `ACTIVITY_WITHDRAW` | a queued item is removed before it ever started |
| `CASE_COMPLETE` | the case reached the process model's terminal state |

## 3. The life of one case

**1. Arrival.** The arrival component schedules `CASE_ARRIVAL`. Handling it does
exactly two things: hand the case to the process component (via a
`__PROCESS_START__` marker), and schedule *its own next arrival*. That
self-perpetuating chain is the only clock driving new work.

**2. Case attributes.** `CaseAttributeSampler` gives the case its
`ApplicationType`, `LoanGoal`, `RequestedAmount`. These matter in exactly two
downstream places: org-model permissions can gate on case type, and `rules`
branching feeds them to a classifier.

**3. Routing — the part most people get wrong.**

> **The BPMN does not decide the branching.** It decides what is *legal*.

The Petri net's current marking determines which transitions are **enabled** —
a hard structural constraint from the sequence/XOR/AND/loop blocks. That is
strictly stronger than "what followed this activity somewhere in the log".
Silent (tau) gateway transitions fire automatically and never reach the log.

Among the *legal* activities, a **separate fitted model** picks the one that
happens. That is `branching_mode`:

- **`probs`** (Basic) — a global probability table, renormalised over the legal
  subset.
- **`visit`** — same, but conditioned on how many times this activity already
  occurred in this case (buckets 1/2/3+). This exists because memoryless
  probabilities understated loop-exit: cases cycled the validation loop 4.35×
  per case against 0.50 real, with only 2% terminating.
- **`rules`** (1.5 Advanced I) — a decision-tree classifier per decision point,
  fed the case attributes.

So: **structure from the BPMN, choice from a probability table or a classifier.**
Two independent models. Confusing them is the most common misreading of the
engine.

**4. Work enabled.** The process schedules `ACTIVITY_REQUEST`.

**5. Allocation.** `ResourceComponent` filters candidates three ways, all of
which must pass:

- **permission** (1.7) — may this person do this activity? The org model can
  condition on case type and weekday.
- **capacity** — is `_busy < capacity`?
- **shift** (1.6) — holiday, vacation, working window, and the `p_work` roster.

If nobody qualifies, the item **queues** and waits for a resource to free up or
for the next shift to open. If somebody does, the **policy** (random / piled /
k-Batching) picks which one. A policy never sees a candidate the filter rejected,
so it cannot violate permissions or the calendar even by accident.

**6. Duration.** On start, the process samples how long the work takes. Then
`ACTIVITY_COMPLETE` releases the resource and routing returns to step 3, until
the Petri net reaches its terminal marking → `CASE_COMPLETE`.

**Note the division of labour:** "who may / who is there / who gets it" is the
resource component; "what happens next / how long" is the process component.
They meet only at `ACTIVITY_REQUEST` and `ACTIVITY_START`.

## 4. The active lifecycle (what `--lifecycle-mode active` changes)

In **legacy** mode a work item is one `start → complete` span, and the resource
is held for the whole thing. But that span is *mostly suspended waiting, not
work*: the six principal W-activities have median hands-on time of 0.8–2.7
minutes while their elapsed spans run to hours or days.

**Active** mode models that explicitly:

1. At session start, the process samples a duration **and rolls a hazard**
   (`session_end_probs`) that decides *up front* whether this session ends in
   `complete` or `suspend`.
2. On `suspend`, **the resource is released immediately** — back in the pool,
   available to anyone.
3. A second roll (`suspend_end_probs`) decides the item's fate:
   - **resume:** sample an external *gap* (the customer hasn't replied…), then
     schedule a **new `ACTIVITY_REQUEST`** carrying `resuming=True`.
   - **abort:** `ate_abort` — the work item dies, **the case continues**.
4. The re-request goes through the **normal** allocation filter. So **the
   resuming resource is drawn from the pool and is usually a different person** —
   deliberately, because that reproduces the observed handover in BPIC-17. Only
   once re-allocated is `ACTIVITY_RESUME` emitted.

State machine: `RUNNING → suspend → (released, waiting a gap) → READY → queued →
allocated → RUNNING (resume)`, bounded by `MAX_SESSIONS`.

**Why this matters beyond 1.3:** because suspend releases the resource, the
interleaving of many applications is modelled *sequentially and explicitly*.
That is why `capacity` must be **1** in active mode — measured on the real log,
98.4% of busy time is a single hands-on session. Any capacity > 1 would
double-count juggling that suspend/resume already represents.

## 5. What comes out

Two log schemas, chosen by mode:

- **legacy** — five columns, and only `start` / `complete`. Byte-identical to
  the pre-lifecycle output.
- **active** — the same five plus **`work_item_id`**, and seven transitions:
  `schedule`, `start`, `suspend`, `resume`, `complete`, `ate_abort`, `withdraw`.

`work_item_id` is not cosmetic: a case can hit the same activity repeatedly, so
joining `suspend` to `resume` on `(case, activity)` alone would mis-pair sessions
across separate visits. The lifecycle rows are gated to `W_` items only.

One suspended-and-resumed item leaves:

```
schedule  (no resource)   enabled, queued
start     User_5          allocated
suspend   User_5          released; gap begins
schedule  (no resource)   re-request after the gap
resume    User_49         a different person picked it up
complete  User_49
```

Invariant: `resume` **always** means resource-bound running work — never "ready
but unassigned".

## 6. The pattern to hold in your head

> **Every component is a *pair*: a model fitted in a notebook, and a switch that
> selects it at runtime. The two can drift apart silently, and nothing fails.**

This is not theoretical. It is the single defect class behind every serious bug
found on 2026-07-17:

- **`p_work`** was fitted, serialised, validated in notebook 01 — and never read
  at runtime. The deployed calendar fielded ~123 people on a Monday morning where
  the validated model expects ~37: a 3.3× overstaffed workforce, so Part II never
  had real contention.
- **MDN arrivals** were fitted, evaluated, and shown to *dominate* the parametric
  model — which is statistically **rejected** against the real log (inter-arrival
  KS p = 3.18e-24, vs the MDN's 0.389). The parametric one was the default.
- **Worse:** `run_experiments.py` never read that switch *at all*. It constructed
  the parametric component unconditionally, so the Part II grid could not have
  used the better model however the flag was set.

None of these were broken functions. `is_available()` did exactly what it said.
Every honest unit test of it passed. **A unit test cannot fail for a connection
that was never made.** What catches this class is a *consistency* check — compare
the model you validated against the model you run (see
`tests/test_roster.py::RosterMatchesValidatedModelTests` and
`tests/test_engine_defaults.py`).

**Corollary — the two entry points must agree.** `simulation/main.py` and
`scripts/run_experiments.py` build the engine *separately*. When they disagree,
the numbers in the report come from the runner, not from the thing you tested by
hand. They currently still disagree; see below.

## 7. Inventory: what is actually switched on

Status as of 2026-07-17, **after** the roster/MDN fixes.

| # | Component | Basic | Advanced | Default now | Owner |
|---|---|---|---|---|---|
| 1.2 | arrivals | LogNormal *(rejected, KS p=3e-24)* | MDN | ✅ **MDN** (fixed) | Mario |
| 1.3 | processing time | fitted distribution | GBR point / quantile | ❌ **distribution** | Daniel |
| 1.4 | process model | flat graph | Petri net | ⚠️ **disagree** — `main.py` advanced, **grid basic** | Sophie |
| 1.5 | branching | probs / visit | rules (classifier) | ❌ **visit** | Sophie |
| 1.6 | availability | window | calendar + roster | ✅ **calendar + roster** (fixed) | Johannes |
| 1.7 | permissions | observed matrix | org model | ✅ **orgmodel** | Johannes |
| — | lifecycle | legacy | active | ❌ **legacy** | Daniel |

**Four advanced components are still off, and one is inconsistent between the
two entry points.** Note the irony worth naming out loud: *each owner's own
advanced model is disabled in the runner that produces the report numbers.*

Two specifics that bite:

- **The grid hardcodes `mode="distribution"`** with no CLI flag at all. So 1.3's
  ML work is not merely off — it is **unreachable** from every Part II number.
- **In `distribution` mode, duration is a function of *activity only*.** The
  resource is not an input. Every person takes the same time. This directly
  undercuts k-Batching, whose entire premise is "assignment quality under
  heterogeneous resource-activity durations": with a flat cost matrix the
  assignment solver has nothing to optimise.

## 8. Artifacts, and an environment trap

Fitted artifacts live in two places: **`models/`** (tracked — small JSON:
availability, permissions, case attributes) and **`simulation/models/`** (mostly
**gitignored**, regenerate with `setup_models.py`).

Currently tracked: `bpic17_process.bpmn`, `dp_branching_probs.json`.
**Local-only:** `decision_rules.joblib`, and `processing_time_model*.joblib`
(which do not exist on Johannes's machine at all).

**The trap:** when a trained artifact is missing, `ExpectedDurationModel` **falls
back silently** to the distribution mean — which ignores the resource. So
k-Batching's cost matrix becomes flat and its solver degenerate, *with no
warning*. Two teammates running the identical command with the identical seed can
therefore get different science depending on whether they ran `setup_models.py`.
If you are touching duration or k-Batching, check the artifact exists before
trusting any number.

## 9. What to verify in your own module

The general question, which found every bug above:

> *Which fitted parameters does my component load, which of those actually reach
> a runtime decision, and does the behaviour match the notebook that validated
> it?*

- **Mario (1.2 / runner):** MDN is now on and the runner reads the switch — but
  the runner still defaults `--process-model basic` while `main.py` uses
  `advanced`. Which one produced the report's numbers?
- **Daniel (1.3 / lifecycle):** the grid can't select your ML models at all.
  Also: `output/validation/lifecycle_active/*.json` were generated at capacity 3
  and are stale now that active mode defaults to 1.
- **Sophie (1.4/1.5):** `rules` is trained and never runs by default; its
  artifact isn't tracked. Does the Petri net actually reach the grid?
- **Johannes (1.6–1.8):** roster is live. Open: legacy `capacity=3` is
  indefensible-but-unchanged (real elapsed spans overlap at a median peak of
  **54**); and a resumed work item currently re-queues with **no priority** over
  fresh work — unverified against the log.

## 10. Open questions this document does not answer

- Does `simulation_inputs.json` still match what `extract_log_info.py` produces
  today, or has it drifted?
- Do the org model's case-type / weekday conditions actually reach allocation?
- Should a resume outrank fresh work in the queue?
- Is legacy mode defensible enough to keep in Part II at all?

See `docs/PLAN_pwork_roster.md` §8 for the roster/capacity decisions and their
evidence.

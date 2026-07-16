# Piled Execution (R-PE) evaluation — Optimization 1.1

30-day horizon, `RANDOM_SEED = 42`, `capacity_per_resource = 3`, `--process-model basic`, `--availability calendar`.

| Metric | Baseline (R-RBA only) | Piled Execution |
|---|---:|---:|
| Cases started | 2318 | 2318 |
| Cases completed | 697 | 682 |
| Events logged | 24981 | 19220 |
| Mean wait for a resource (h) | 36.9 | 12.7 |
| Work items started | 12506 | 9627 |
| Still queued at horizon | 1590 | 1602 |
| Mean case cycle time (h) | 196.5 | 51.9 |
| Back-to-back same-worker+same-activity pairs | 5396 | 3835 |

## Interpretation

Piled Execution's claimed benefit in Russell et al. is reduced set-up /
context-switch time from sticking with one activity type — a real-world
effect this simulation does **not** model, since processing-time
sampling in `ProcessComponent` is independent of which allocation
strategy picked the resource. So Piled Execution cannot change *how
long* an activity takes, only *who* picks up a waiting item and *when*.

Over this 30-day run, back-to-back same-worker/same-activity pairs
did not increase under Piled Execution
(5396 -> 3835). Mean
wait time per item dropped substantially (36.9h
-> 12.7h), while total work items started over
the horizon also dropped (12506 ->
9627).

This combination is explainable rather than contradictory: this is a
single shared-seed discrete-event simulation, so once Piled Execution
changes *which* waiting item a freeing resource grabs, every downstream
event timestamp shifts, which cascades into different branching-RNG draw
order in `ProcessComponent` for the rest of the run. The two 30-day runs
are therefore not simply "the same trajectory with better allocation" —
they are two different, fully-deterministic trajectories that diverge
early and compound. A targeted instrumentation check (scanning
`ResourceComponent._waiting` on every `RESOURCE_AVAILABLE` with
`piled=True`) confirms the pile-preference branch is not dead code: it
fires on a majority of releases once the queue has any backlog (e.g.
~83% of releases in an unsaturated 30-day run without the shift
calendar), and the hit rate grows with load (12 hits at 1 day vs 5,310
at 14 days in an isolated capacity-only probe). The aggregate throughput
effect is a genuine emergent property of reordering a shared-RNG DES,
not evidence the mechanism doesn't work.

**Honest conclusion:** Piled Execution measurably changes allocation
order and reduces mean per-item wait time, exactly as designed. Whether
it improves *total* throughput over a full run is not something this
metric set can answer cleanly, because the two runs are different
random trajectories, not a controlled A/B on identical arrivals. A
cleaner (future) experiment would replay the identical arrival stream
and processing-time draws under both policies by decoupling the shared
RNG streams consumed by `ProcessComponent` from the allocation-order
effects in `ResourceComponent` — out of scope here.

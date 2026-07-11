# Paper insights: "Discovering Simulation Models" (Rozinat, Mans, Song, van der Aalst, 2009)

Source: `paper_assignment3.pdf`. This file distills what's actually applicable
to *this* repo — not a full paper summary. Read this before touching the
simulation's validation strategy (`scripts/metrics.py`,
`scripts/compare_process_models.py`) or arguing about model quality.

## Core idea

Don't build simulation models by hand — mine them from event logs, across
**four perspectives**, then merge into one executable model:

| Perspective | What it captures | Our equivalent |
|---|---|---|
| Control-flow | causal/ordering constraints between activities | `process.py` (Basic: flat graph) / `petri_process.py` (Advanced: Petri net) |
| Data | decision rules at branch points (e.g. age > 60 → X) | not implemented — `BRANCHING_PROBS` is purely stochastic, not data-conditioned |
| Performance | execution/waiting time distributions, branch probabilities, arrival rate | `process.py` (`PROCESSING_TIME_PARAMS`, `BRANCHING_PROBS`), `arrival.py` / `arrival_mdn.py` |
| Resource/organizational | roles, permissions, allocation | `resource.py` |

The paper's representation target is a Coloured Petri Net (CPN Tools); we
target a plain Petri net (pm4py) driving a hand-written DES engine instead —
same idea (formal control-flow enforcement), lighter-weight implementation.

## The validation method we adopted: "Second Pass"

This is the part we directly implemented. The paper's core validation trick:

1. Mine perspectives from the **real** log → build simulation model (**"first pass"**).
2. Run the simulation → get a **simulated** event log.
3. Re-run the *same* mining algorithms on the simulated log (**"second pass"**).
4. Compare first-pass vs. second-pass results. A good model reproduces them
   closely; systematic deviations point at what the model is getting wrong.

They found: control-flow and resource perspective were **100% rediscoverable**
(exact match) in all their examples; performance perspective (execution time,
waiting time mean/std, branch probabilities, arrival rate) matched **closely
but not exactly**, which they treat as expected simulation noise, not a
modeling flaw.

We implemented exactly this idea in `scripts/metrics.py` /
`scripts/compare_process_models.py`, except our "first pass" reference is the
pre-computed `simulation_inputs.json` (from the real BPIC-17 log) rather than
re-running discovery each time — same comparison, cheaper to run repeatedly.

## KPIs to use when judging "did this change help?"

Straight from the paper's second-pass comparisons, implemented in
`metrics.py`:

- **Control-flow fitness** (`control_flow_fitness`): do simulated traces
  replay legally on the reference Petri net? Target ~100% for any model that
  claims to *enforce* control-flow (this is what proved Advanced > Basic for
  Section 1.4: 100% vs ~0-1%).
- **Control-flow precision** (`control_flow_precision`): the paper doesn't
  need this (their models are always exact matches to a discovered net), but
  we do, because our net is discovered with `noise_threshold=0.2` and our
  Basic model doesn't use the net at all. **Fitness alone is gameable** — a
  trivial "flower model" (every activity enabled everywhere) gets 100%
  fitness with terrible precision. Always read both together.
- **Branching probability divergence** (`branching_divergence`, total
  variation distance): matches the paper's per-decision-point probability
  comparison (their Figure 17).
- **Processing time error** (`processing_time_errors`): matches their
  execution/waiting time mean comparison (their Figures 12/16/20).
- **Arrival rate error** (`arrival_rate_error`): matches their inter-arrival
  intensity comparison.
- **Process variant overlap** (`variant_overlap`): not explicit in the paper,
  but a natural extension of "does the model reproduce the real behavior
  distribution" beyond just per-activity stats.
- **Case length / duration error** (`case_length_duration_errors`): matches
  their "overall processing time" comparisons in the case studies.

Run `scripts/compare_process_models.py` any time you change a component to
see whether the change moved these numbers in the right direction — that's
the reusable regression check this file was requested alongside.

## Caveats from the paper that apply to us

- **Waiting time is structurally underestimated when simulating one process
  in isolation.** Real resources are shared across multiple processes /
  work part-time; a model that only sees "this" process has resources
  available too eagerly, so simulated waiting times come out far too low.
  The paper's fix was an empirically-calibrated added delay (65-100% of
  observed waiting time, tuned by second-pass comparison). **If our
  `case_duration_rel_err` / waiting-time numbers look too optimistic, this
  is the textbook explanation, and the fix is the same: compare against
  reference, add a calibrated multiplier, don't just trust the raw resource
  model.**
- **Nominal decision-variable values must be sampled from their real
  empirical distribution, not generated uniformly at random.** The paper
  explicitly found this: when a categorical data attribute driving a
  decision was randomly generated instead of drawn from the real value
  distribution, the resulting branch probabilities drifted from reality even
  though the decision *rule* itself was correct. Relevant if we ever add
  data-conditioned branching (see "Data" row above — currently unimplemented
  for us).
- **A simulation model is only as good as the perspectives it covers.**
  Coarse approximations (e.g. probability-only branching, single waiting-time
  bucket) can't answer questions tied to what they abstracted away (e.g.
  "what if case mix shifts" needs data-conditioned branching, not just fixed
  probabilities). Useful framing for scoping what "Advanced" variants are
  worth building next.
- **Control-flow and resource perspectives are the easiest to get exactly
  right; performance is inherently approximate.** Don't chase 100% fitness
  on performance-style KPIs (processing time / arrival rate error) the way
  we did for control-flow fitness — some gap is expected and fine, the
  paper's own case studies never got performance error to zero.

## What we deliberately did *not* adopt

- CPN Tools / Coloured Petri Nets as the runtime — we use pm4py + a custom
  Python DES engine instead. Same formal backbone (Petri net firing rules),
  different tooling.
- Data-driven decision points (decision trees on case attributes) — out of
  scope so far; `BRANCHING_PROBS` remains purely stochastic.

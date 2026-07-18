# One-million-step DRL evaluation

## Outcome

The longer MaskablePPO run fixes the catastrophic throughput collapse but does
not outperform the heuristic policies. It trained for 1,024,000 decision steps
on seeds beginning at 10000 and was evaluated on held-out seeds 1--10 under the
same 10-day, 2-day-warm-up, advanced/visit, OrdinoR and active-lifecycle setup
as the heuristic grid.

| Metric | DRL 100k | DRL 1m | Random | Pull-SPT |
|---|---:|---:|---:|---:|
| Mean cycle time (days) | 0.494* | 0.740 | 0.530 | **0.459** |
| P95 cycle time (days) | 1.345* | 2.107 | 1.493 | **1.333** |
| Total completions/run | 92.3 | 245.5 | 268.4 | **276.6** |
| Evaluated completions after warm-up | 72.2 | 189.4 | 212.6 | **220.4** |
| Resource occupation | 0.146 | 0.240 | 0.344 | **0.344** |
| Fairness deviation (lower is better) | **0.164** | 0.202 | 0.223 | 0.229 |

\* The 100k cycle-time values are strongly survivor-biased because the policy
completed only 14.2% of started cases.

Relative to the 100k policy, the 1m model completes 166.0% more cases
(paired p<0.001), starts 162.3% more evaluable completed cases, and raises
occupation by 64.3%. Its longer observed cycle time is expected: it now
finishes many cases that the 100k model left censored at the horizon.

Relative to Random, the 1m model completes 8.5% fewer cases (paired p<0.001),
has 39.7% higher mean cycle time (p<0.001), 41.2% higher P95 cycle time
(p<0.001), and 30.3% lower occupation (p<0.001). Its fairness deviation is
9.5% lower (p=0.026), but this does not compensate for the efficiency loss.

## Why the 100k model completed so few cases

A held-out seed-1 diagnostic found 3,762 assignments, 8,062 postponements and
520 queued work items at the horizon. Random started 8,749 work items and left
210 queued. There were zero permission failures, so the cause was learned
strategic idling rather than deadlock or missing authorization.

Three factors made this likely:

1. The policy had only 102,400 training decisions for an action space of 3,457
   actions (144 resources times 24 activities plus `POSTPONE`). The largest
   process reported by Middelhuis et al. had only 24 actions.
2. `POSTPONE` was always feasible and the original dense reward did not impose
   a direct cost for unnecessary postponement. Middelhuis et al. report the
   same failure in their N-network and add a postponement penalty.
3. The version-1 observation showed whether a resource was busy but not which
   activity it was executing, removing information used in the paper's state.

Version 2 adds the current activity per resource, uses a scale-adjusted 0.001
postponement penalty, decays the learning rate linearly, and trains ten times
longer. On seed 1 it makes 7,852 assignments and leaves 220 items queued,
versus 3,762 and 520 for the 100k model. The absolute postponement count rises
because the improved policy reaches many more decision epochs; the important
change is that it interleaves enough assignments to keep the process moving.

## Recommendation

Use Pull-SPT as the report's operational recommendation. Present the 1m DRL
result as evidence that the learning pipeline and reward correction work, but
not as evidence that DRL beats the heuristics. Further work should prioritize a
smaller or factorized action representation, case-prefix/progress features,
and checkpoint selection on held-out throughput/cycle-time metrics before
spending the full 20-million-step budget.

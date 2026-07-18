# Resource-allocation policy evaluation

## Setup

All policies use 10 paired common-random-number replications, a 10-day
simulation horizon, a 2-day case-arrival warm-up, the advanced process model
with visit-based branching, OrdinoR permissions, active lifecycle handling,
and the normal-arrival scenario. Lower cycle time and fairness deviation are
better; higher occupation and completion count are better.

The DRL result is explicitly preliminary. Its MaskablePPO policy was trained
for 102,400 decision steps on episode seeds beginning at 1000 and evaluated on
held-out seeds 1--10. The heuristic results come from the completed experiment
grid in `output/experiments/results_normal.csv`.

## Results

| Policy | Mean cycle time (days) | P95 cycle time (days) | Total completions per run | Occupation | Fairness deviation |
|---|---:|---:|---:|---:|---:|
| Pull-SPT | **0.459** | **1.333** | **276.6** | 0.344 | 0.229 |
| Park--Song | 0.472 | 1.529 | 275.9 | 0.323 | 0.271 |
| DRL (102.4k) | 0.494* | 1.345* | **92.3** | 0.146 | 0.164 |
| Round robin | 0.527 | 1.477 | 267.4 | **0.346** | 0.220 |
| Random (baseline) | 0.530 | 1.493 | 268.4 | 0.344 | 0.223 |
| Shortest queue | 0.532 | 1.493 | 268.3 | 0.329 | 0.265 |
| KRM delta=1 | 0.545 | 1.539 | 267.5 | 0.323 | 0.261 |
| KRM delta=0.5 | 0.548 | 1.510 | 266.3 | 0.319 | 0.254 |
| KRM delta=2 | 0.564 | 1.574 | 267.0 | 0.324 | 0.260 |
| Piled execution | 0.612 | 1.806 | 266.2 | 0.341 | 0.224 |
| Pull-LAF | 0.706 | 3.212 | 210.9 | 0.218 | 0.142 |
| k-Batching k=10 | 1.143 | 2.133 | 221.8 | 0.266 | 0.196 |
| k-Batching k=5 | 1.360 | 2.435 | 210.9 | 0.235 | 0.180 |
| k-Batching k=1 | 1.899 | 3.193 | 168.8 | 0.158 | **0.136** |

\* The DRL cycle-time values are affected by severe right-censoring: it
completes only 14.2% of started cases, versus 41.4% for Random. Its apparently
short cycle time therefore describes a small, easy subset and is not an
efficiency improvement.

## Paired conclusions versus Random

- **Pull-SPT is the best operational choice.** Mean cycle time is 13.4% lower
  (paired t-test p=0.0011) and it completes 8.2 more cases per run
  (p<0.001), without a significant occupation or fairness change.
- **Park--Song is second for speed/throughput.** Mean cycle time is 10.9% lower
  (p=0.0043) and it completes 7.5 more cases (p<0.001), but occupation is
  lower and fairness deviation is higher; both trade-offs are significant.
- **Round robin is effectively tied with Random.** Its 0.6% mean-cycle-time
  reduction is not significant (p=0.58), and neither are its other changes.
- **Shortest queue does not improve cycle time or throughput.** It lowers
  occupation and increases fairness deviation significantly.
- **All three KRM settings are slower than Random.** Delta=0.5, 1, and 2 raise
  mean cycle time by 3.4%, 2.9%, and 6.5%, respectively; each cycle-time
  increase is significant at p<0.05. Delta=1 is the least harmful setting.
- **Piled execution is worse in the normal scenario.** Mean cycle time rises
  15.5% and P95 rises 21.0%, both significant.
- **Pull-LAF and k-Batching trade efficiency for lower fairness deviation.**
  Their much lower completion counts and longer cycle times make them poor
  choices when throughput and customer delay are primary.
- **The preliminary DRL model is not deployable.** It completes 176.1 fewer
  cases per run than Random (65.6% fewer, paired p<0.001). More training and a
  convergence study are required before comparing its cycle-time statistics
  with the heuristics.

## Recommendation

Use Pull-SPT for the assignment's normal-scenario recommendation. Retain
Random as the transparent baseline and Park--Song as the main advanced-policy
comparison. Present k-Batching as a negative result that improves workload
equality only by idling resources and delaying/completing fewer cases. Present
the 102.4k-step DRL result as a pipeline and early-learning experiment, not as
evidence against converged DRL.

The 95% confidence-interval half-width for mean cycle time remains above the
project's 5% target for every policy in this 10-seed grid. The rankings are
useful and the strongest paired effects are clear, but a final claim of close
equivalence would require more seeds. Peak and outage files were not combined
with this table because they were produced under an older, different horizon
and capacity configuration.

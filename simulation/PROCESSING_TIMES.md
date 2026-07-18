# Processing-Time and Work-Item Lifecycle Models

The simulator has two deliberately separate lifecycle baselines and three
duration samplers. `--lifecycle-mode legacy` is the default and preserves the
original five-column, single-block `start → complete` behavior. Opt-in
`--lifecycle-mode active` models BPIC-17 W-item sessions explicitly:

```text
schedule → start → (suspend → resume)* → complete | ate_abort
         ↘ withdraw (while still queued)
```

Only `W_` work items enter this state machine. Atomic `A_`/`O_` activities keep
their synthetic start plus fallback duration and do not churn.

This default applies to the standalone `simulation.main` CLI. The experiment
runner defaults to active mode, and the 04 evaluation notebooks set
`lifecycle_mode="active"` explicitly. Their results therefore do exercise
suspend/resume behavior; changing the standalone default is not required for
those experiments.

## Parameter coverage and atomic activities

The active input artifact covers all eight routed `W_` activities in each of
the processing-time, session-end, suspend-end, and resume-gap tables. The
generic `0.5` lifecycle fallback is therefore not expected to run in the active
evaluation configuration. Withdrawal hazards exist for four of the eight
activities; an absent hazard means that the item cannot withdraw while queued.

The older static duration table has fitted distributions for six `W_`
activities. Its fallback table contains the other two rare `W_` activities and
18 `A_`/`O_` activities. In active mode, all eight `W_` activities use the
versioned active-session fits, but the 18 atomic activities still use assumed
fallback durations. These assumptions affect event spacing, resource
occupation, and queueing even though BPIC-17 offers no start/complete pairs from
which to estimate them directly.

Evaluation therefore reports two views: all-activity resource KPIs and W_-only
KPIs with the same availability denominator. It also runs a paired lower-bound
sensitivity with `--atomic-duration-scale 0`. This removes A_/O_ busy duration
but retains permission checks, calendars, assignment, and event order. The
default scale is `1`; the zero case is a robustness check, not the preferred
model.

## Why active service time is separate

The historical `start → complete` span is mostly suspended waiting, not hands-on
work. The six principal W-activities have median active work of roughly 1–16
minutes, while their elapsed spans can be hours or days. In active mode:

- each `start` or `resume` opens one active session;
- the session ends in `complete` or `suspend`;
- suspend releases the resource to the pool;
- a resume-ready item goes through normal permission, shift, and allocation
  checks before the logged `resume`;
- `ate_abort` terminates the work item but continues the case;
- an initial queued request may lose the allocation race to a mined withdrawal
  timer.

The CSV therefore has a sixth `work_item_id` column in active mode. Every
lifecycle reconstruction and duration metric keys on it. Legacy output retains
the exact original five columns.

## Duration modes

| `--mode` | Legacy target | Active target | Artifact |
|---|---|---|---|
| `distribution` | fitted elapsed `start → complete` span | fitted next active-session duration | none |
| `ml_model` | contextual point estimate of elapsed span | contextual point estimate of active-session seconds | mode-selected joblib |
| `ml_probabilistic` | conditional quantile curve for elapsed span | conditional quantile curve for active-session seconds | mode-selected joblib with 19 quantile models |

The eight ML features are encoded activity, resource, previous activity,
weekday, hour, case position, case age, and prior-activity count. In active mode
they are computed once at the work item's first start and duplicated across all
of its sessions; session index is intentionally not a v1 feature.

Artifact metadata declares `target` and `lifecycle_schema`. Loading an active
artifact in legacy mode or a legacy artifact in active mode fails loudly.

## Versioned artifacts

| Purpose | Legacy | Active |
|---|---|---|
| duration model | `simulation/models/processing_time_model.joblib` | `simulation/models/processing_time_model_active.joblib` |
| fitted tables | `simulation_inputs.json` | `simulation_inputs_active.json` (`lifecycle` block) |
| training metrics | `output/models/processing_time_metrics.json` | `output/models/processing_time_metrics_active.json` |

The active parameter block contains active-session distributions, session-end
and suspend-end hazards, resume-gap residuals (including an explicit zero mass),
terminal-outcome continuation, withdrawal timers, and validation summaries.
Retraining active artifacts never overwrites the legacy set.

The active fitted tables, training metrics, and joblib are version-controlled.
A fresh checkout can therefore run the active `distribution`, `ml_model`, and
`ml_probabilistic` configurations without the source XES log. The commands below
are maintainer steps for an intentional refit, not normal setup.

## Regenerate and train

From the repository root:

```bash
# Active lifecycle/churn tables. The _active filename also enables lifecycle
# extraction; --lifecycle is shown explicitly for clarity.
.venv/bin/python extract_log_info.py --log BPIChallenge2017.xes \
  --out simulation_inputs_active.json --lifecycle

# Active point + quantile artifact and versioned metrics.
.venv/bin/python train_processing_time_model.py --log BPIChallenge2017.xes \
  --lifecycle --probabilistic \
  --output simulation/models/processing_time_model_active.joblib \
  --metrics-output output/models/processing_time_metrics_active.json

# Equivalent convenience wrapper (skips existing artifacts unless --force).
.venv/bin/python setup_models.py --lifecycle active
```

The legacy wrapper remains `.venv/bin/python setup_models.py`.

## Run

```bash
# Reproducible original baseline.
.venv/bin/python -m simulation.main --lifecycle-mode legacy --mode distribution

# Active pool lifecycle.
.venv/bin/python -m simulation.main --lifecycle-mode active --mode distribution
.venv/bin/python -m simulation.main --lifecycle-mode active --mode ml_model
.venv/bin/python -m simulation.main --lifecycle-mode active --mode ml_probabilistic

# Lower-bound sensitivity for atomic A_/O_ durations.
.venv/bin/python -m simulation.main --lifecycle-mode active --mode distribution \
  --atomic-duration-scale 0
```

k-batching uses expected elapsed duration in legacy mode and expected **next
active-session** duration in active mode, because suspend/complete is the next
resource-release point.

## Modeling limits

- v1 is serial: one in-flight W-item per case. This covers 99.8% of BPIC-17
  cases; the small concurrent minority is outside fit claims.
- The pool model is intentional: only about 17.5% of historical resumes use the
  previous resource. That rate is a baseline calibration target, not invariant
  under allocation policies.
- Resume-ready time is not observed. The extractor subtracts only the contiguous
  deterministic off-shift tail immediately preceding resume, using the historical
  resume resource's weekly calendar and public holidays (not sampled vacations).
  The fitted remainder is a calibrated residual, not a causal split between
  customer waiting and historical queueing.
- BPIC-17 has no distinct BPMN abort/withdraw transition. Advanced mode fires the
  corresponding visible W-task, intersects the mined continuation with legal
  successors after silent-transition closure, and records fallback counts when
  the intersection is empty.
- `W_Shortened completion ` and `W_Personal Loan collection` have boundary
  estimates `P(complete)=0` and `P(resume)=1` in the current active artifact.
  They are sparsely observed. The runtime records every activation of the
  60-session safety guard, and the evaluation notebooks reject a run if the
  guard forces completion. Until the hazards are re-estimated with a documented
  smoothing rule, conclusions should also report how often these routes occur.
- The policy evaluation notebooks deliberately use the distribution sampler.
  Point and probabilistic ML are evaluated in the Section 1.3 validation
  notebook, but they are not exercised by the reported policy comparison. A
  claim that the policy results use ML durations would therefore be incorrect.

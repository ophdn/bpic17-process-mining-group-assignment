# Deep evaluation runbook

This runbook covers the evaluation that supplies the optimization results. The
report cites only the 60-day experiment in `notebooks/04_evaluation_60.ipynb`.
The 10- and 30-day notebooks are retained as optional development diagnostics;
they are not prerequisites and should not be run for the report workflow.

## 1. Preflight and required reruns

The committed lifecycle-validation artifact now describes the corrected active
lifecycle, including automatic zero-time `A_`/`O_` transitions, inter-activity
waiting, and the calibrated case-duration envelope. Validate it before opening
the evaluation notebook:

```bash
cd /Users/danielsich/dev/Ent/bpic17-process-mining-group-assignment
source venv/bin/activate
export MPLCONFIGDIR=/tmp/bpic17-mpl
mkdir -p "$MPLCONFIGDIR"

python -c 'import json; from pathlib import Path; from scripts.eval_lifecycle import validate_lifecycle_validation_artifact; p=Path("output/validation/lifecycle_active/distribution.json"); validate_lifecycle_validation_artifact(json.loads(p.read_text()), "distribution"); print("Lifecycle validation is current")'
```

If that check fails, run `python scripts/run_lifecycle_validation.py`. This
generates three 60-day validation runs: `distribution`, `ml_model`, and
`ml_probabilistic`. The evaluation notebook uses the `distribution` artifact,
but the script regenerates the complete controlled comparison.

No processing-time, arrival, or decision model needs retraining for the 60-day
evaluation. The active-session processing-time target and artifacts are
unchanged, the arrival MDN is unchanged, and the notebook deliberately uses
visit-aware branching instead of the decision-rule model. The DRL exception is
documented below.

The notebook reports two different time windows on purpose:

- Cycle time uses arrivals that complete during the 180-day drain, with at
  least 99% completion required to control right-censoring.
- Operational throughput counts completions by the end of the 60-day arrival
  window. Do not use the drained completion count as throughput; it is expected
  to converge to the number of arrivals.
- Occupation, fairness, activity-switching, and staffing criticality use the
  same fixed 60-day operational window. The drain is excluded from their
  denominator so policies with different final completion times remain
  comparable.

## 2. The supplied DRL archive

The local archive currently has this path:

```text
models/drl_resource_policy_rocm_v3_100k.zip
```

Model ZIP files are ignored by the repository's current `.gitignore`. This file
is therefore local and will not reach collaborators through a normal commit or
pull. Keep its SHA-256 with the experiment record and distribute the archive by
the team's approved artifact channel if another machine must reproduce the run.

Check the archive before installing or running it:

```bash
DRL_MODEL="models/drl_resource_policy_rocm_v3_100k.zip"
test -f "$DRL_MODEL"
unzip -t "$DRL_MODEL"
shasum -a 256 "$DRL_MODEL"
```

The inspected archive reports Stable-Baselines3 2.9.0, 1,024,000 learned
timesteps, a version-3 observation shape of 534, a compact version-2 action
space of 1,942 actions, and a `[256, 256]` network. The filename says `100k`,
but the saved model itself reports 1,024,000 timesteps.

These dimensions match the current loader's V3/V2 vocabulary. This establishes
structural compatibility only. The archive has no matching JSON metadata file,
and the existing `models/drl_resource_policy*.json` files describe different
observation spaces, action spaces, or network widths. Do not associate those
JSON files with this archive.

More importantly, the ZIP alone does not record its training checkout. The
current simulator not only makes `A_`/`O_` transitions automatic but also uses
corrected active-lifecycle timing, including inter-activity waiting and a
calibrated duration envelope. These changes alter queue and workload states
seen by the policy. Therefore, the existing archive may be evaluated only as a
structurally compatible frozen legacy policy. Retrain DRL under the current
simulator before using its result to claim the performance of an optimized
learned policy. Until then, keep the DRL row in an exploratory comparison and
exclude it from conclusions about which trained policy is best.

The archive records PPO seed 1 and four parallel environments but does not
contain a complete list of training and checkpoint-validation episode seeds.
The 60-day notebook therefore uses seeds 4,000,001--4,000,004 for every policy,
above the four worker-seed bands implied by the current trainer. This avoids the
recorded worker bases and preserves Common Random Numbers across the complete
grid. It is still not a substitute for the missing companion metadata: the
trainer should confirm the complete training and checkpoint-selection seed
sets before the DRL result is described as a fully held-out estimate.

## 3. Install and smoke-test DRL support

The DRL packages are optional dependencies and are not installed by the base
requirements. The 60-day notebook now evaluates DRL, so install them before
running it:

```bash
python -m pip install -r requirements-drl.txt

python -c 'import gymnasium, stable_baselines3, sb3_contrib, torch; print(torch.__version__, stable_baselines3.__version__, sb3_contrib.__version__)'

PYTHONPATH=. python -c 'from simulation.drl import load_maskable_ppo; m=load_maskable_ppo("models/drl_resource_policy_rocm_v3_100k.zip"); print("observation", m.observation_space.shape, "actions", m.action_space.n)'
```

An optional one-day engineering smoke test verifies the current simulator
vocabulary and inference path. Its seed is not evidence for the report:

```bash
PYTHONPATH=. python scripts/run_experiments.py \
  --policies drl \
  --drl-model models/drl_resource_policy_rocm_v3_100k.zip \
  --seeds 1 \
  --days 1 \
  --process-model advanced \
  --branching-mode visit \
  --permissions orgmodel \
  --lifecycle-mode active \
  --atomic-duration-scale 0 \
  --out output/drl_model_smoke/
```

The smoke test should report DRL assignments, no unpermitted `W_` activity, no
resource-assigned automatic transition, and no observation/action-space error.

## 4. Run the report evaluation

Select the `BPIC17 (venv)` kernel in VS Code and use **Restart Kernel and Run
All** on:

```text
notebooks/04_evaluation_60.ipynb
```

The equivalent terminal command is:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_evaluation_60.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1
```

Do not run `04_evaluation.ipynb` or `04_evaluation_30.ipynb` unless a shorter
horizon is specifically needed for debugging. The 60-day notebook no longer
reads or validates their outputs.

## 5. Number of simulation runs

The 60-day notebook evaluates all eight fixed runner policies and the two
configured parameter sweeps with four paired seeds. The finite grid is:

```text
random, roundrobin, shortestqueue, piled, pullspt, pulllaf, parksong, drl,
kbatch1, kbatch5, kbatch10, krm0.5, krm1, krm2
```

`kbatchN` and `krmD` accept further positive values, so “all” means every fixed
runner policy plus the documented study values, not every mathematically
possible parameter value.

| Component | Calculation | New simulation runs |
|---|---:|---:|
| Policy comparison | 14 policies × 4 seeds | 56 |
| Staffing baseline | reuses the 4 R-RMA policy runs | 0 |
| Remove low-criticality pair | 1 scenario × 4 seeds | 4 |
| Remove high-criticality pair | 1 scenario × 4 seeds | 4 |
| **60-day notebook total** |  | **64** |

Regenerating a stale prerequisite lifecycle comparison adds three runs, so a
completely fresh report workflow executes 67 simulations before considering
the optional one-day DRL smoke test. With a current validated artifact, the
notebook itself executes 64 simulations. The metric schema and complete policy
grid invalidate the old caches, as required for a paired comparison.

For reference, the optional 10- and 30-day notebooks each contain 12 policy
runs. Running all three horizons would therefore execute 88 notebook
simulations (12 + 12 + 64), but those additional 24 short-horizon runs do not
support a claim currently made in the report.

## 6. Expected outputs

The 60-day notebook writes:

- `output/evaluation_60/configuration.json`
- `output/evaluation_60/policy_run_metrics.csv`
- `output/evaluation_60/policy_summary.csv`
- `output/evaluation_60/policy_paired_deltas.csv`
- `output/evaluation_60/resource_criticality.csv`
- `output/evaluation_60/staffing_run_metrics.csv`
- `output/evaluation_60/staffing_summary.csv`
- `output/evaluation_60/staffing_paired_deltas.csv`
- `output/report_inputs/evaluation_60d_report_values.json`
- `visualization/04_60_policy_tradeoff.pdf`
- `visualization/04_60_policy_tradeoff.svg`
- `visualization/04_60_staffing_impact.pdf`
- `visualization/04_60_staffing_impact.svg`

The final cell must print `All provenance and result sanity checks passed.`
Only then are the tables and figures internally consistent with the current
simulator.

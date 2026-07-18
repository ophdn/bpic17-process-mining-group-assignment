# Deep evaluation runbook

This runbook covers the evaluation that supplies the optimization results. The
report cites only the 60-day experiment in `notebooks/04_evaluation_60.ipynb`.
The 10- and 30-day notebooks are retained as optional development diagnostics;
they are not prerequisites and should not be run for the report workflow.

## 1. What must be rerun now

The simulator now treats `A_` and `O_` records as automatic zero-time state
changes. The committed lifecycle-validation artifact still describes the old
`atomic_duration_scale=1` configuration and has stale provenance. Therefore,
run the controlled lifecycle comparison once before opening the evaluation
notebook:

```bash
cd /Users/danielsich/dev/Ent/bpic17-process-mining-group-assignment
source venv/bin/activate
export MPLCONFIGDIR=/tmp/bpic17-mpl
mkdir -p "$MPLCONFIGDIR"

python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_lifecycle_validation.py
```

This generates three 60-day validation runs: `distribution`, `ml_model`, and
`ml_probabilistic`. The evaluation notebook uses the `distribution` artifact,
but the script regenerates the complete controlled comparison. The processing
time model does not need to be retrained, and the two Section 1.3 notebooks do
not need to be rerun merely to execute the evaluation.

Confirm that the required artifact now matches the checkout:

```bash
python -c 'import json; from pathlib import Path; from scripts.eval_lifecycle import validate_lifecycle_validation_artifact; p=Path("output/validation/lifecycle_active/distribution.json"); validate_lifecycle_validation_artifact(json.loads(p.read_text()), "distribution"); print("Lifecycle validation is current")'
```

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

More importantly, the ZIP alone does not record whether training used the new
automatic `A_`/`O_` semantics. Use the model in the report only if its trainer
can confirm that it was trained from commit `7a2551f` or an equivalent checkout
where `atomic_duration_scale=0` and `A_`/`O_` activities bypass resources.
Otherwise, retrain the DRL policy under the current simulator. Evaluating an old
resource-assigned policy after removing `A_`/`O_` work from its decision process
would be an out-of-distribution comparison.

The archive records PPO seed 1 but does not contain a complete list of training
and checkpoint-validation episode seeds. Since the current evaluation uses
seeds 1--4, DRL must not be added to that grid until the trainer supplies the
seed metadata. A final DRL comparison must use evaluation seeds unseen during
both training and checkpoint selection, and all baseline policies must be rerun
with exactly those same seeds for Common Random Numbers.

## 3. Install and smoke-test DRL support

The DRL packages are optional and are not installed in the current `venv` by
the base requirements. Install them only if the supplied model passes the
semantic and seed checks above:

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

The current 60-day notebook evaluates three policies with four paired seeds:

| Component | Calculation | New simulation runs |
|---|---:|---:|
| Policy comparison | 3 policies × 4 seeds | 12 |
| Staffing baseline | reuses the 4 R-RMA policy runs | 0 |
| Remove low-criticality pair | 1 scenario × 4 seeds | 4 |
| Remove high-criticality pair | 1 scenario × 4 seeds | 4 |
| **60-day notebook total** |  | **20** |

The prerequisite lifecycle comparison adds three runs, so a completely fresh
report workflow executes 23 simulations before considering smoke tests.

The current notebook does **not** include DRL. Once a semantically matching
model and unseen seed set are available, adding DRL contributes four policy
runs, producing 24 runs in the 60-day notebook and 27 including lifecycle
validation. Changing the seed set invalidates every policy cache, as required
for a paired comparison.

For reference, the optional 10- and 30-day notebooks each contain 12 policy
runs. Running all three horizons would therefore execute 44 notebook simulations
(12 + 12 + 20), but those additional 24 runs do not support a claim currently
made in the report.

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
- `visualization/04_60_staffing_impact.pdf`

The final cell must print `All provenance and result sanity checks passed.`
Only then are the tables and figures internally consistent with the current
simulator.

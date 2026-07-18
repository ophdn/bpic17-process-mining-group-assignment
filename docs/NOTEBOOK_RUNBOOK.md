# Notebook execution runbook

This runbook regenerates the processing-time evidence, the 60-day policy and
staffing study, and their machine-readable outputs. Run every command from the
repository root. The long simulations are intentionally not part of the test
suite. For the detailed evaluation-only workflow, DRL compatibility checks, and
exact run counts, see `docs/EVALUATION_DEEP_RUNBOOK.md`.

The active processing-time model, fitted lifecycle inputs, and controlled
lifecycle-validation results are committed. The DRL ZIP is distributed
separately and must be placed at the path named in Section 5. Collaborators who
have that archive and only need the current model and evaluation evidence can
install the environment in Section 1 and continue directly with Section 5;
Sections 2--4 are maintainer steps for an intentional regeneration.

## 1. Prepare and register the Python environment

```bash
cd /Users/danielsich/dev/Ent/bpic17-process-mining-group-assignment
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-drl.txt
python -m ipykernel install --user --name bpic17-venv --display-name "BPIC17 (venv)"
python -c "import sys; print(sys.executable)"
```

The last command must print the repository's `venv/bin/python`. Registering the
kernel after activation is important: Jupyter kernel names are stable, but their
stored interpreter paths are not automatically updated when a virtual
environment is recreated.

For macOS, keep Matplotlib's cache in a writable temporary directory:

```bash
export MPLCONFIGDIR=/tmp/bpic17-mpl
mkdir -p "$MPLCONFIGDIR"
```

## 2. Verify the checkout and raw log

```bash
git status --short --branch
python -m pytest -q
```

Keep the BPIC-17 source log at one of these locations:

- `BPIChallenge2017.xes.gz`
- `data/BPIChallenge2017.xes.gz`
- the equivalent uncompressed `.xes` path

Set the path once and verify it before starting any expensive extraction. The
current local checkout stores the uncompressed log at the repository root:

```bash
BPIC17_LOG="BPIChallenge2017.xes"
ls -lh "$BPIC17_LOG"
```

Change `BPIC17_LOG` only when your copy is stored at one of the other supported
locations. Do not continue with a different log version when reproducing the
committed artifacts.

## 3. Regenerate the active processing-time inputs and models

Re-mine the active lifecycle inputs before retraining. This step can take a
while because it reads the full event log.

```bash
python extract_log_info.py \
  --log "$BPIC17_LOG" \
  --out simulation_inputs_active.json \
  --lifecycle \
  --availability-model models/availability_model.json

python setup_models.py \
  --log "$BPIC17_LOG" \
  --lifecycle active \
  --force
```

Expected model outputs include:

- `simulation/models/processing_time_model_active.joblib`
- `output/models/processing_time_metrics_active.json`
- the refreshed lifecycle block in `simulation_inputs_active.json`

## 4. Regenerate the controlled lifecycle comparison

```bash
python scripts/run_lifecycle_validation.py
```

This performs three paired 60-day runs using capacity 1, the advanced Petri
process, visit-aware branching, active lifecycle, MDN arrivals, OrgModel
permissions, a common seed/roster, and automatic zero-time `A_`/`O_` state
changes. It writes:

- `output/validation/lifecycle_active/distribution.json`
- `output/validation/lifecycle_active/ml_model.json`
- `output/validation/lifecycle_active/ml_probabilistic.json`

The `03` and `04` notebooks reject these files if their schema, configuration,
or code/input hashes are stale. The old capacity-3 artifacts cannot silently
enter the analysis. Rerun this comparison after changing simulator semantics;
the evaluation notebooks intentionally reject the previous
`atomic_duration_scale=1` artifacts.

## 5. Run the notebooks in order

In VS Code, open each notebook, select **BPIC17 (venv)**, and use **Restart
Kernel and Run All**. Run them in this exact order:

1. `notebooks/03_processing_times.ipynb`
2. `notebooks/03_process_times.ipynb`
3. `notebooks/04_evaluation_60.ipynb` — evaluation and staffing study

The 60-day notebook evaluates the complete configured policy grid, including
the supplied DRL archive at
`models/drl_resource_policy_rocm_v3_100k.zip`. It fails before starting the
long simulations if the ZIP, optional DRL packages, or current OrgModel
observation/action compatibility check is missing.

For a terminal-driven run, use the same order:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_processing_times.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_process_times.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_evaluation_60.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1
```

The 60-day notebook is self-contained and does not read the 10- or 30-day
outputs. Run caches are reused only when the full schema, configuration, and
provenance checks pass; manually deleting caches is unnecessary.

All three evaluation notebooks model `A_` and `O_` records as automatic
zero-time state changes. Their start and complete rows share a timestamp and
have no resource assignment. The notebooks assert that these transitions
consume no busy time; only `W_` activities enter human queues and staffing
metrics. This is a modeling limitation, not evidence that no organizational
effort occurred behind the recorded milestones.

## 6. Verify generated outputs

The final cells must print `All provenance and result sanity checks passed.`
Inspect these machine-readable outputs after execution:

- `output/report_inputs/processing_time_report_values.json`
- `output/report_inputs/processing_time_sampler_summary.csv`
- `output/report_inputs/processing_time_recomposition.csv`
- `output/report_inputs/processing_time_ml_diagnostics.csv`
- `output/report_inputs/evaluation_60d_report_values.json`

The primary 60-day evaluation figures are:

- `visualization/04_60_policy_tradeoff.pdf`
- `visualization/04_60_staffing_impact.pdf`

The 10- and 30-day notebooks remain optional development checks. They are not
part of this execution sequence because the final evaluation uses the 60-day
configuration only.

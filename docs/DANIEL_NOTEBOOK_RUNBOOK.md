# Daniel notebook runbook

This runbook regenerates the processing-time evidence, the 10/30/60-day policy
studies, and the report hand-off files. Run every command from the repository
root. The long simulations are intentionally not part of the test suite.

## 1. Prepare and register the Python environment

```bash
cd /Users/danielsich/dev/Ent/bpic17-process-mining-group-assignment
source venv/bin/activate
python -m pip install -r requirements.txt
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
git switch adj/final
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
locations. Do not continue with a different log version if the report is meant
to reproduce the committed study.

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
permissions, and a common seed/roster. It writes:

- `output/validation/lifecycle_active/distribution.json`
- `output/validation/lifecycle_active/ml_model.json`
- `output/validation/lifecycle_active/ml_probabilistic.json`

The `03` and `04` notebooks reject these files if their schema, configuration,
or code/input hashes are stale. The old capacity-3 artifacts cannot silently
enter the report.

## 5. Run the notebooks in order

In VS Code, open each notebook, select **BPIC17 (venv)**, and use **Restart
Kernel and Run All**. Run them in this exact order:

1. `notebooks/03_processing_times.ipynb`
2. `notebooks/03_process_times.ipynb`
3. `notebooks/04_evaluation.ipynb` — 10-day directional study
4. `notebooks/04_evaluation_30.ipynb` — 30-day horizon check
5. `notebooks/04_evaluation_60.ipynb` — report and staffing study

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
  notebooks/04_evaluation.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_evaluation_30.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1

python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_evaluation_60.ipynb \
  --ExecutePreprocessor.kernel_name=bpic17-venv \
  --ExecutePreprocessor.timeout=-1
```

The 60-day notebook deliberately runs last. Its final cell reads the 10- and
30-day summaries and rejects them unless their schema, configuration, and full
provenance match the current checkout. Run caches are reused only when the same
checks pass; manually deleting caches is unnecessary.

## 6. Verify the report hand-off

The final cells must print `All provenance and result sanity checks passed.`
Inspect these machine-readable outputs before changing LaTeX:

- `output/report_inputs/processing_time_report_values.json`
- `output/report_inputs/processing_time_sampler_summary.csv`
- `output/report_inputs/processing_time_recomposition.csv`
- `output/report_inputs/processing_time_ml_diagnostics.csv`
- `output/report_inputs/evaluation_horizon_summary.csv`
- `output/report_inputs/evaluation_60d_report_values.json`

The authoritative evaluation figures are:

- `visualization/04_60_policy_tradeoff.pdf`
- `visualization/04_60_staffing_impact.pdf`

The 10- and 30-day studies are horizon checks; their old staffing files, if
present locally, are not report inputs. Staffing conclusions come only from the
60-day notebook.

## 7. Update the report repository

```bash
cd /Users/danielsich/dev/Ent/6a562e35fea37a0c6eeea788
git pull

cp ../bpic17-process-mining-group-assignment/visualization/03_lifecycle_time_recomposition.pdf figures/
cp ../bpic17-process-mining-group-assignment/visualization/03_point_model_feature_importance.pdf figures/
cp ../bpic17-process-mining-group-assignment/visualization/03_quantile_interval_coverage.pdf figures/
cp ../bpic17-process-mining-group-assignment/visualization/04_60_policy_tradeoff.pdf figures/
cp ../bpic17-process-mining-group-assignment/visualization/04_60_staffing_impact.pdf figures/
```

Use the JSON/CSV hand-offs rather than copying values from notebook screenshots.
Update these Daniel-owned report files:

- `Input/simulation/processing_times.tex`
- `Input/optimization/evaluation.tex`
- `Input/optimization/management_question.tex`
- `Input/appendix.tex` when the ML diagnostic discussion changes

After editing, compile the LaTeX project and visually check that tables, figure
labels, confidence intervals, completion shares, selected employees, and the
stated horizon all match the regenerated hand-offs. Commit the simulation
repository and report repository separately so the evidence-producing code is
not mixed with prose-only changes.

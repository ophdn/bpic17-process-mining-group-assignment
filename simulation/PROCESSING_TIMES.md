# Processing-Time Models (Section 1.3)

How to generate event logs with each of the three processing-time models, and
how to (re)train the ML artifact they rely on.

The processing-time model lives in `components/process.py` (`ProcessComponent`)
and is selected with `--mode` on `main.py`. Everything is seeded
(`RANDOM_SEED = 42`) so a given mode produces an identical log every run.

---

## The three modes

| Mode | What it does | Assignment level | Needs artifact? |
|---|---|---|---|
| `distribution` | Samples from scipy distributions fitted per activity (lognorm / gamma / weibull), exponential fallback. | 1.3 **Basic** | No |
| `ml_model` | Contextual **point estimate**: a GradientBoostingRegressor predicts `log1p(duration)` from 8 features, inverted to seconds. | 1.3 **Basic option 2** | Yes |
| `ml_probabilistic` | Contextual **probability distribution**: 19 quantile regressors (q = 0.05…0.95) form a conditional curve; a `Uniform(0,1)` draw is interpolated to a stochastic duration. | 1.3 **Advanced I** | Yes (`--probabilistic`) |

The 8 context features (reconstructed at sample time): label-encoded
`activity`, `resource`, `previous_activity`, plus `day_of_week`, `hour_of_day`,
`case_position`, `case_age_seconds`, `n_previous_activities`.

---

## 1. Set up the environment (once)

```bash
cd <repo-root>
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

---

## 2. Train the ML artifact (needed for `ml_model` / `ml_probabilistic`)

The artifact is git-ignored (`simulation/models/*.joblib`), so regenerate it
from the raw event log. One `--probabilistic` run produces **both** the point
model and the quantile models, covering all ML modes.

```bash
# point model only — enough for --mode ml_model
.venv/bin/python train_processing_time_model.py --log BPIChallenge2017.xes

# point + 19 quantile models — required for --mode ml_probabilistic
.venv/bin/python train_processing_time_model.py --log BPIChallenge2017.xes --probabilistic
```

Writes `simulation/models/processing_time_model.joblib`. `distribution` mode
needs no artifact.

**Point-model quality (held-out test):** log-space R² ≈ 0.36, MAE ≈ 33.6 h
(raw R² ≈ 0 — durations are heavy-tailed). Top features by importance:
`case_age_seconds` (0.46), `activity` (0.23), `resource` (0.16).

---

## 3. Run the simulation

From the repo root, with the venv:

```bash
.venv/bin/python -m simulation.main --mode distribution
.venv/bin/python -m simulation.main --mode ml_model
.venv/bin/python -m simulation.main --mode ml_probabilistic
```

Each writes `output/event_log.csv` (relative to the current directory),
**overwriting** the previous run. To keep logs side by side:

```bash
.venv/bin/python -m simulation.main --mode ml_model
cp output/event_log.csv output/event_log_ml_model.csv
```

Extra flags:

- `--model-path <path>` — use an artifact from a non-default location.
- Run length and seed are constants in `main.py` (`SIM_DURATION_DAYS = 30`,
  `RANDOM_SEED = 42`).

---

## 4. Reference output (30-day run, seed 42)

Reproducible summary statistics for each mode:

| Mode | Cases started | Cases completed | Events logged | Wall time |
|---|---:|---:|---:|---:|
| `distribution` | 2,318 | 2,177 | 58,737 | ~4.5 s |
| `ml_model` | 2,318 | 2,023 | 55,323 | ~10 s |
| `ml_probabilistic` | 2,318 | 1,058 | 38,128 | ~48 s |

Per-activity duration spread (start→complete), same runs:

| Mode | Median | Std dev | p95 |
|---|---:|---:|---:|
| `distribution` | 0.05 h | 16.0 h | 6.9 h |
| `ml_model` (point) | 2.65 h | 17.7 h | 33.4 h |
| `ml_probabilistic` | 7.41 h | 45.8 h | 121.8 h |

Reading the table: the point model (`ml_model`) collapses each context to one
value, so it under-represents the tail. The probabilistic model
(`ml_probabilistic`) restores that spread — which is the whole point of Advanced
I. Because its durations are longer and more realistic, fewer cases finish
inside the fixed 30-day window (1,058 vs. 2,177), leaving more in-flight at the
horizon.

---

## Notes & known limitations

- **Reproducibility / variance.** Same seed → byte-identical log. In
  `ml_probabilistic`, durations vary *within* a fixed context (the quantile
  draw), while `ml_model` returns a constant per context.
- **Resource feature is often `__UNKNOWN__`.** ML durations are full
  start→complete times spanning hours/days, so the 17 permitted resources
  (× capacity 3) stay busy and the pool saturates; most activities are then
  sampled with no resource assigned. The wiring is correct — resources do flow
  to the model when allocation succeeds. To make it less degenerate, raise
  `capacity_per_resource` in `main.py`.
- **Advanced I vs. II.** The probabilistic model targets the full start→complete
  duration (service **plus** any waiting), so that variability is already
  captured. Splitting service vs. waiting time (Advanced II) is an optional
  refinement, not a prerequisite for the advanced tier.

# TODO — final stretch to submission (Sunday 2026-07-19)

Written 2026-07-15 (Wednesday) after a full repo survey. Ordered by
(deadline pressure × who can actually do it). Previous round's work
(runner, CRN, k-Batching, policy seam, arrival metrics, report
subsections) is done and pushed — see git log `05b0f83..56f1e21`.

Ground rules (unchanged): run from repo root with `.venv/bin/python`;
seed 42 for anything cited in the report; single-line commit messages,
**no Co-Authored-By trailer**; `git fetch && git rebase origin/main`
before starting and before pushing — teammates are in crunch too.

---

## Task 1 — Plotting script (`scripts/plot_experiments.py`)

**Why:** The roadmap's Phase B spec requires "results CSV **+ plots**",
Phase C wants **boxplots** per policy × metric, Phase D a **Pareto plot**
(efficiency vs. fairness) — and `scripts/run_experiments.py` only writes
CSVs. The report has no Part II figures at all. This is Mario's
infrastructure gap and everything downstream (Sophie's management
question, D1/D2 evaluation) will want the same plots.

**Spec:**

1. Input: one or more `results_<scenario>.csv` files produced by
   `run_experiments.py` (columns: policy, seed, scenario,
   avg_cycle_time_s, p95_cycle_time_s, avg_resource_occupation,
   resource_fairness, …). CLI:
   `--results output/experiments/results_normal.csv [more.csv ...]
   --out report/figures/`.
2. Outputs (PNG, matplotlib — already a transitive dependency; no new
   deps, no seaborn):
   - `boxplot_<metric>_<scenario>.png` — one box per policy, seeds as
     the distribution; cycle-time metrics in **days**, not seconds.
   - `pareto_<scenario>.png` — scatter of mean cycle time (x, days) vs.
     resource fairness (y; lower = fairer), one point per policy,
     labeled; error bars = 95% CI half-widths from
     `aggregate_<scenario>.csv` if present next to the results file.
   - If multiple scenarios are passed: also
     `robustness_cycle_time.png` — grouped bars per policy across
     scenarios (this is the RQ4 figure).
3. Style: no title text baked into the PNG (captions live in LaTeX),
   axis labels with units, readable at `\columnwidth`.
4. Deterministic output (fixed figure size/dpi, sorted policy order) so
   re-running doesn't churn the repo diff.

**Verify:** run on the existing 5-seed `output/experiments/
results_normal.csv`; PNGs open, boxes match the aggregate CSV means;
re-run produces byte-identical files.

## Task 2 — Definitive experiment runs (10+ seeds, advanced model, all scenarios)

**Why:** Every Part II number currently citable rests on 5 seeds,
`--warmup-days 0`, and the *basic* process model. The runner itself
flags the CIs as too wide (roadmap target: 95% CI half-width ≤ 5% of
mean cycle time, ≥ 10 seeds). The roadmap's own timeline earmarks
Thu–Fri nights for exactly these runs. The report's k-Batching table
(`report/main.tex`, `tab:kbatch`) and its "5 seeds, indicative not
conclusive" caveat must be replaced with the real thing.

**Spec:**

1. Pick the warm-up first:
   `PYTHONPATH=. .venv/bin/python scripts/run_experiments.py --report-wip --days 30`
   — choose the day where open-case WIP plateaus, and note the chosen
   value (it goes in the report's Experimental Setup subsection, which
   currently promises a "WIP-plot justification").
2. Then the full grid, overnight-friendly (background it; the advanced
   model is ~10–20× slower than basic):
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \
     --policies random,piled,kbatch1,kbatch2,kbatch5,kbatch10,kbatch20 \
     --seeds 10 --days 30 --warmup-days <chosen> \
     --process-model advanced --branching-mode visit \
     --scenario normal --out output/experiments/
   ```
   and the same for `--scenario peak` and `--scenario outage`
   (RQ4 / management-question infrastructure). If wall-clock makes the
   advanced model infeasible for the full grid, fall back to basic for
   the k-sweep and advanced for `random` vs. the best k only — and say
   so in the report; don't silently mix configurations in one table.
3. Regenerate figures (Task 1) from the new CSVs.
4. **Update the report**: replace the numbers in `tab:kbatch`, drop or
   soften the 5-seed caveat in the Experimental Setup subsection
   (keep it if CIs are still > 5%!), fill in the chosen warm-up and its
   justification, and reference the new figures. Re-check that the
   prose claims (monotonic k-trend, significance statements) still hold
   under the new numbers — if they flipped, rewrite the interpretation
   honestly rather than keeping the old story.

**Verify:** `aggregate_normal.csv` has n_seeds ≥ 10 per policy;
`paired_tests_normal.csv` p-values quoted in the report match the CSV;
figures regenerate cleanly.

## Task 3 — Real-log validation (BLOCKED on a human with the log)

**Why:** The one open item on Mario's Part I section (the report says so
explicitly). `scripts/eval_arrivals.py` is ready and deliberately
refuses to fake ground truth. Nothing else on this list unblocks it —
someone has to produce the raw BPIC-17 log file (team share, or
https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b).

**Once the log exists, in order:**

1. `BPIC17_LOG=/path/to/log PYTHONPATH=. .venv/bin/python scripts/eval_arrivals.py`
   → writes `output/arrival_model_eval.md`. Commit it.
2. Re-run `extract_log_info.py` against the log so
   `simulation_inputs.json` gains the new `arrival_rate.hod_profile` /
   `dow_profile` fields (the `arrival_profile_error` metric returns
   `None` until then). **Diff `simulation_inputs.json` carefully before
   committing** — this file feeds branching/durations for the whole
   team; only the `arrival_rate` block should change. If anything else
   moves, the log file/version differs from the original extraction —
   stop and check with the team rather than committing drift.
3. Update the "Validation status" paragraph of the Case Arrivals report
   subsection with the actual MAE/KS numbers (it currently documents
   the blockage — replace, don't append).
4. Bring the MDN-vs-LogNormal result to the team and decide TOGETHER
   whether to flip `USE_MDN_ARRIVALS = True` in `simulation/main.py` —
   it changes every teammate's simulated logs and invalidates cached
   evidence under `output/validation/`. Do not flip unilaterally.

## Task 4 — Simple policies R-RRA + R-SHQ (conditional: confirm handoff with Johannes first)

**Why:** Mandatory Final Task 1 heuristics (lecture 06 slide 10). They
are Johannes's on paper, but the roadmap itself says "notfalls zu Mario
schieben", Johannes also owns A3 + D2, the gate for these was
Wednesday, and he hasn't pushed since the 14th. **Message him first**;
implement only after he agrees (or doesn't answer by Thursday).

**Spec (once handed off):**

1. Check `git stash show -p stash@{0}` first — it holds his earlier
   R-RRA/R-SHQ attempt against the *old* resource.py. Salvage the
   intent (and any per-pattern details), not the code; then drop the
   stash with his OK.
2. Implement in `simulation/policies.py` against the existing seam —
   each is ~20 lines:
   - `RoundRobinPolicy` (R-RRA, pattern 16): cycle through candidates;
     keep a per-activity cursor (a global cursor degenerates when
     candidate sets differ per activity). Deterministic, no RNG.
   - `ShortestQueuePolicy` (R-SHQ, pattern 17): pick the candidate with
     the lowest `state.busy[r]`; break ties by candidate order (which
     is deterministic — insertion order of `RESOURCE_PERMISSIONS`), not
     randomly, so runs stay reproducible without consuming RNG draws.
3. Wire into `scripts/run_experiments.py` (`KNOWN_POLICIES` +
   `build_resource_component`) and, if desired, `simulation/main.py`
   (`--policy roundrobin|shortestqueue`).
4. Tests in a new `scripts/test_policies.py`: R-RRA actually rotates
   (no candidate starved over a long run), R-SHQ never picks a
   busier candidate than the minimum, both reproducible, R-RBA
   permission filter still upstream of both (they never see an
   unqualified candidate — assert via the existing permission check
   pattern from `test_resource_allocation.py`).
5. Add both to the Task-2 experiment grid and the report's Simple
   Policies half (that subsection is currently only k-Batching —
   whoever writes the R-RRA/R-SHQ paragraphs gets named there).

## Task 5 — Report nits that are Mario's alone (5 minutes)

1. `report/main.tex` line ~22: `\author[email=todo@tum.de]{Mario~(TODO)}`
   — put the real name and TUM email. (Daniel/Johannes have the same
   TODO for theirs; leave those to them.)
2. `report/references.bib`: the Zeng & Zhao k-Batching entry is a
   commented-out placeholder — the exact citation is on lecture deck
   06's reading list (slide 12). Fill it from the slides; do NOT guess
   a venue/year. Then cite it from the k-Batching subsection (currently
   the name appears with no `\cite`).

## Task 6 — Team pings (not code; send today, they're the critical path)

Four days left and the abstract/conclusion can only be written once
these exist. Suggested messages:

- **Daniel:** Processing Times subsection is `tbd.`; D1 (Park & Song)
  not started; and `processing_time_model.joblib` is not in the repo,
  so `--mode ml_model` AND the k-Batching cost model silently fall back
  to distribution means for everyone but him. Ask him to either commit
  the artifact (if small enough) or document the training command
  prominently. Point him at `simulation/expected_duration.py` — D1
  should consume it (that was the agreed shared API).
- **Johannes:** 1.6–1.8 subsection `tbd.`; D2 not started;
  `decision_rules.joblib` missing from the repo (breaks
  `--branching-mode rules` for everyone else); and the R-RRA/R-SHQ
  handoff question (Task 4). Also: his old stash exists — don't let him
  redo work that's half-done in it.
- **Sophie:** the management-question infrastructure is ready for her —
  `ResourceComponent(excluded_resources=...)` +
  `scripts/run_experiments.py --scenario outage` do leave-N-out runs
  already; she needs only the criticality analysis + a loop over
  candidate pairs. Two Signavio figure exports
  (`figures/bpic17_signavio_bpmn.png`, `figures/loop_fragment_signavio.png`)
  are still framed as TODO boxes in the report.
- **All:** abstract + conclusion are `tbd.` pending D1/D2 numbers;
  agree tonight who writes them Saturday.

---

## Definition of done for this round

- Part II figures exist in `report/figures/` and are referenced from
  the report; regenerating them is deterministic.
- Every number in the report's Part II tables comes from a ≥10-seed run
  with a documented warm-up, or carries an explicit caveat.
- `output/arrival_model_eval.md` committed with real-log numbers and a
  recorded team decision on `USE_MDN_ARRIVALS` (or the blockage is
  escalated to the team as a Sunday risk).
- R-RRA/R-SHQ either implemented+tested+in-the-grid, or explicitly
  confirmed as staying with Johannes.
- No `tbd.`/`(TODO)` left in report sections owned by Mario.

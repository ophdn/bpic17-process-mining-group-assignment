# TODO — status & handoff (submission: Sunday 2026-07-19)

Updated Thu 2026-07-16 late. This file is the handoff document: what is
DONE (with where and how to verify), and what REMAINS (with enough
context to execute cold). Read `docs/ROADMAP.md` for the team plan.

Ground rules: `.venv/bin/python` from repo root; before every commit run
`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` (94 tests) — the
three legacy script suites (`scripts/test_{resource_allocation,crn,
kbatching}.py`) are also kept green; single-line commit messages, **no
Co-Authored-By**; `git fetch && git rebase origin/main` before starting
AND before pushing (the repo takes ~50 commits/day right now; GitHub
access from this machine is intermittently flaky — a 75s connect
timeout that succeeds on retry).

**The final team configuration** — every number the report cites uses
exactly this, no mixing: `--process-model advanced --branching-mode
visit --permissions orgmodel --lifecycle-mode active` (+ roster on, CRN
on). It is what `output/evaluation/configuration.json` records.

---

## DONE (committed; local commits may still need a push — check `git log origin/main..main`)

### Policies — the full Part II registry now exists

All in `simulation/policies.py` (simple), `simulation/components/
resource.py` (queue disciplines + advanced), `simulation/
policies_advanced.py` (D1's predictor). Runner names in parentheses;
all wired in `scripts/run_experiments.py::build_resource_component`.

1. **R-RMA random** (`random`) — baseline, pre-existing.
2. **R-RRA round robin** (`roundrobin`) — per-activity cursor,
   deterministic, no RNG (CRN-safe). Commit `feat(opt-C): R-RRA...`.
3. **R-SHQ shortest queue** (`shortestqueue`) — min live busy count,
   ties by candidate order, no RNG. Same commit.
4. **R-PE Piled Execution** (`piled`) — pre-existing, bonus/ablation.
5. **k-Batching** (`kbatchN`) — pre-existing, the graded Zeng & Zhao
   deliverable.
6. **Pull disciplines** (`pullspt`, `pulllaf`) — the push-vs-pull
   evaluation. Implemented at the release-drain seam in
   `on_resource_available` (`pull="spt"|"laf"` ctor arg): the freed
   resource picks its preferred permitted item (SPT = shortest expected
   duration; LAF = longest-active case first, lecture deck 04 F31)
   instead of the FIFO scan. Deliberately NOT a FIFO relabel; only
   meaningful under contention (active mode). Honest scope, needed in
   the report: without the trained ML duration artifact, SPT's costs
   are resource-independent (distribution means), so it is a
   shortest-processing-time queue discipline, not personalised.
   Commit `feat(opt-C): pull-side disciplines...`.
7. **D1 Park & Song** (`parksong`) — prediction-based allocation with
   strategic idling. Epoch assignment (`_epoch_flush` in resource.py)
   over real waiting items + phantom items (predicted next activity of
   in-service cases, argmax of the mined branching model — the
   documented LSTM substitution, see `policies_advanced.py` docstring).
   Three guards that were each added because their absence measurably
   broke the policy (all pinned by tests):
   - **lookahead gate** (`PARKSONG_LOOKAHEAD_SECONDS = 3600`): phantoms
     only for successors expected within the hour — without it,
     resources sat reserved for days (completions 158 -> 15).
   - **uncertainty penalty**: phantom cost x (1/p_branching).
   - **spread gate**: a phantom whose permitted costs show no spread
     across free resources is dropped — with flat (artifact-less)
     costs, reserving can never pay, only hurt (occupation collapsed
     0.47 -> 0.08 before this gate). Consequence to state in the
     report: with resource-independent durations, parksong reduces to
     plain epoch assignment; reservations activate only with the
     trained resource-aware duration model.
8. **D2 Kunkler & Rinderle-Ma** (`krmD`, e.g. `krm1`, `krm0.5`) —
   assignment variant with per-item dummy columns costing
   `delta x mean duration + time-already-waited`. The aging term is
   load-bearing: without it, delta < 1 under flat costs deferred
   everything forever (a 5-day run logged ZERO events). Delta sweep via
   the policy name.

   Commit for 7+8: `feat(opt-D): D1 Park & Song ... and D2 ...`.

**Verification state:** 94 pytest tests green (incl. new
`tests/test_policies.py`, `tests/test_pull.py`,
`tests/test_advanced_policies.py`), plus the three legacy script
suites. 5-day final-config smoke (2 seeds): random 158/134 completions,
parksong 158/134 at BETTER cycle time (0.69 -> 0.59d), krm1 ≈ random,
krm0.5 slightly worse cycle time (the deferral price, visible), pullspt
improves cycle time (SPT minimises mean flow time — textbook), pulllaf
trades throughput for old-case completion. All consistent with theory —
good sign nothing is silently broken.

### Infrastructure (earlier sessions, all pushed)

- Experiment runner (`scripts/run_experiments.py`) with scenarios
  (normal/peak/outage), CIs, paired t-tests, `--report-wip`; teammates
  extended it with lifecycle/roster/capacity flags and checkpointing.
- CRN (`crn=True` on process components) — verified: identical branching
  prefixes across policies under the same seed (`scripts/test_crn.py`).
- Plot script (`scripts/plot_experiments.py`): boxplots, Pareto,
  robustness bars; deterministic output.
- 11x perf cache on `OrgModelPermissions` (byte-identical logs,
  verified) — makes final-config runs cheap.
- MDN arrivals validated against the real log and now team default;
  `output/arrival_model_eval.md` tracked.
- `opt_metrics.py` slide-21 metrics + custom metrics (time-to-first-
  offer/decision, handover rate, rolling workload balance) — teammates
  extended further; datetime-coercion crash fixed (flaky object-dtype
  `.dt`).

---

## REMAINING (in priority order)

### R1 — Push local commits

`git log origin/main..main` — if non-empty, `git push origin main`
(retry on the 75s timeout; it's transient). Rebase first if origin
moved.

### R2 — The definitive evaluation run + figures

Everything is now in place for the one run the report cites:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \
  --policies random,roundrobin,shortestqueue,piled,pullspt,pulllaf,kbatch1,kbatch5,kbatch10,parksong,krm0.5,krm1,krm2 \
  --seeds 10 --days 10 --warmup-days 2 \
  --process-model advanced --branching-mode visit \
  --permissions orgmodel --lifecycle-mode active \
  --out output/experiments/
```

10-day horizon matches the team's tracked study
(`output/evaluation/configuration.json`); coordinate before overwriting
anything under `output/evaluation/` — that directory is written by
`notebooks/04_evaluation.ipynb`, which may be the team's
source-of-truth pipeline; `output/experiments/` is the runner's own
output and safe. Then figures:
`PYTHONPATH=. .venv/bin/python scripts/plot_experiments.py --results
output/experiments/results_normal.csv --out report/figures/`.
Commit CSVs + figures together with the report text that cites them.

Warm-up note for the report: under the final config WIP plateaus around
day 5 on a 30-day run (see `--report-wip`); for a 10-day horizon a 2-day
warm-up is the defensible compromise — say which was used.

### R3 — Report (`report/main.tex`) — six gaps, three have data waiting

1. **k-Batching subsection** (`sec:kbatching`): replace the stale
   5-seed legacy-config table + caveat with R2's numbers.
2. **New: simple policies + push/pull paragraphs** in the same
   subsection: R-RRA/R-SHQ one-liners; the pull framing (local
   resource-optimal vs global system-optimal), results, and the honesty
   sentence: BPIC-17 records who did the work, not what was on anyone's
   worklist, so every pull rule is assumed behaviour — the principled
   reason Part I is push-only.
3. **D1/D2 subsections** (skeleton TODOs at ~line 398): the
   implementations exist now — describe mechanism + the three D1 guards
   and the D2 aging term as design decisions WITH their measured
   justifications (numbers above), plus results from R2. Name Daniel
   (D1) and Johannes (D2) — they own the grade credit; coordinate.
4. **Management question subsection**: data is tracked and sufficient —
   `output/evaluation/staffing_summary.csv` (remove_low ≈ baseline,
   remove_high worse) + `resource_criticality.csv` (User_139, User_110
   lowest criticality). Draft it, name Sophie, flag for her review.
5. **Daniel's 1.3 / Johannes's 1.6–1.8 Part I subsections**: draft from
   their notebooks + `visualization/03_*` figures, flag as drafts.
6. **Abstract + conclusion** last.

Bib: uncomment `park2019prediction` and `kunkler2024online` in
`references.bib` and cite them from the D1/D2 subsections.

### R4 — Humans only (do NOT guess)

- Mario's email: `CONFIRM-EMAIL@tum.de` placeholder in the author line.
- Daniel/Johannes author lines (`todo@tum.de`).
- Zeng & Zhao citation: exact reference is on lecture deck 06 slide 12's
  reading list; the bib has a clearly marked placeholder.
- Sophie's two Signavio figure exports (report TODO boxes at ~181/250).
- Team pings: D1/D2 are built but are Daniel's/Johannes's named grade
  credit — they should review, adjust, and present them as theirs.

### R5 — Nice-to-have (only if time)

- `simulation/main.py` CLI flags for parksong/krm/pull (currently
  runner-only) — teammates may want them for ad-hoc runs.
- `docs/manuals/resource_allocation_heuristics.md`: add the pull
  disciplines and D1/D2 to the pattern inventory.
- peak/outage scenario rows for the robustness figure (RQ4).

---

## Gotchas the next agent should know

- **The spread gate means parksong ≈ plain assignment without the ML
  artifact** (`simulation/models/processing_time_model*.joblib`,
  gitignored, likely absent on this machine). If Daniel's artifact is
  present, reservations activate — re-run R2 with it if it appears, and
  the report's D1 paragraph changes from "reduces to" to actual
  reservation numbers.
- `ResourceComponent` now handles `CASE_COMPLETE` (D1 bookkeeping,
  no-op otherwise) — if a test registers ResourceComponent twice or
  asserts on handler counts, that's why.
- The allocation modes (`piled`, `batching_k`, `pull`, `parksong`,
  `krm_delta`) are mutually exclusive, enforced in the constructor.
- `output/experiments/*.csv` currently on disk are STALE (old configs,
  killed runs) — regenerate via R2 before citing anything.
- `report/figures/*.png` on disk are from 5-seed legacy data — same.
- One old stash (`stash@{0}`, "WIP on main: 3fbab6a") holds a pre-seam
  R-RRA/R-SHQ attempt — superseded by `tests/test_policies.py` work;
  confirm with Johannes, then drop.

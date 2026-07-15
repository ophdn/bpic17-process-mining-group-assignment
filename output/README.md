# output/ — Evidenz für den Report

Jeder Unterordner sammelt die Metriken zu **einer Designentscheidung**.
Beim Report-Schreiben: Entscheidung nennen → Zahlen aus dem passenden
Ordner zitieren. CSV-Logs sind gitignored; die JSON/MD-Auswertungen werden
committet.

| Ordner | Designentscheidung | Erzeugt von |
|---|---|---|
| `validation/process_model_comparison/` | **1.4:** Basic (Wahrscheinlichkeitsgraph) vs. Advanced (Petrinetz-Enforcement) — Fitness/Precision, Branching-TVD, Variantenabdeckung, Case-Länge/-Dauer, Completion-Rate | `scripts/compare_process_models.py` |
| `validation/bpmn_source_comparison/` | **1.4:** manuell modelliertes (Signavio-)BPMN vs. aus dem Log geminte Modelle (Inductive Miner, Noise-Sweep) — Kernfrage: Wieviel echtes Verhalten kann das Modell überhaupt replayen? | `scripts/discover_process_model.py` + `compare_process_models.py --bpmn … --tag …` |
| `validation/branching_probs_vs_rules/` | **1.5:** stochastische Branching-Wahrscheinlichkeiten vs. Decision-Point-Klassifikatoren | `compare_process_models.py` (nach A1-Umbau mit `--branching-mode`) |
| `models/processing_time_metrics.json` | **1.3:** Verteilungs- vs. ML- vs. probabilistisches Zeitmodell — MAE/MSE/R² (temporaler Split, Folie 44) + Pinball-Loss & Intervall-Coverage (Folie 47) | `train_processing_time_model.py` |
| `models/decision_rules_metrics.json` | **1.5:** Qualität je Decision Point — Accuracy vs. Majority-Baseline, Precision/Recall (macro), ROC-AUC (OvR); enthält auch den Real-Log-Fit% aufs BPMN | `train_decision_rules.py` |
| `optimization/` (ab Phase C) | **Teil II:** Policy-Vergleiche auf den drei Folie-21-Metriken, ≥ 10 Seeds, KIs | `scripts/opt_metrics.py` + Experiment-Runner |
| `event_log.csv` | jeweils letzter Simulationslauf (gitignored, wird überschrieben) | `simulation/main.py` |

Konventionen:

- **Dateinamen:** `<config>[_<tag>].json`, z. B. `advanced_im02.json` =
  Advanced-Simulation mit dem Inductive-Miner-Modell (noise 0.2).
- **Referenzwerte** stammen immer aus `simulation_inputs.json` (echtes
  BPIC-17-Log); Methodik: Second-Pass-Validierung nach Rozinat et al.,
  siehe `docs/paper_insights_discovering_simulation_models.md`.
- Nach jeder Komponentenänderung `scripts/compare_process_models.py` neu
  laufen lassen und die JSONs hier aktualisieren — die Deltas sind die
  empirische Begründung der Designentscheidung im Report.

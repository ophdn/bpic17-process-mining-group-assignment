# Report-Notizen: 1.4 Prozessmodell & 1.5 Branching Decisions (Sophie)

Alle Zahlen: Seed 42, 30-Tage-Horizont, KPIs gegen `simulation_inputs.json`
(echtes BPIC-17-Log), Methodik = Second-Pass-Validierung nach Rozinat et al.
(siehe `docs/paper_insights_discovering_simulation_models.md`). Jede Zahl
ist aus einer JSON-Datei in `output/` reproduzierbar (Datei-Index unten).

---

## 1. Was gebaut wurde

| Baustein | Datei | Assignment-Level |
|---|---|---|
| Flacher Next-Activity-Wahrscheinlichkeitsgraph | `simulation/components/process.py` | 1.4 Basic |
| BPMN → Petrinetz-Konvertierung (pm4py), Kontrollfluss-Enforcement über Firing Rules + Tau-Closure-Frontier | `simulation/components/petri_process.py` | 1.4 Advanced |
| Empirische Branch-Wahrscheinlichkeiten (global, besuchszahl- und decision-point-konditioniert) | `process.py`, `extract_log_info.py`, `scripts/mine_dp_probs.py` | 1.5 Basic |
| Attribut-Sampling beim Spawn aus gelernten (konditionalen) Verteilungen + Laufzeit-Attribute durch `O_Create Offer` + Decision-Tree-Klassifikator pro Decision Point | `petri_process.py`, `train_decision_rules.py` | 1.5 Advanced I |
| Terminierungs-Fix: datengetriebene END-Entscheidung an Tau-erreichbaren Endmarkierungen | `petri_process.py`, `scripts/mine_dp_probs.py` | Methodenbeitrag (1.4/1.5) |
| KPI-Suite + Benchmark-Runner (Fitness/Precision, Branching-TVD, Variantenabdeckung, Case-Statistiken, Completion) | `scripts/metrics.py`, `scripts/compare_process_models.py` | Evaluation |

## 2. Designentscheidungen mit Evidenz

### D1 — Basic vs. Advanced Prozessmodell (1.4)
**Entscheidung:** Petrinetz-Enforcement als Standard.
**Evidenz** (`output/validation/process_model_comparison/`): Basic erreicht
0 % voll passende Traces (Fitness 0,64) — es erzeugt strukturell unmögliche
Abläufe; Advanced per Konstruktion 100 %. Precision 0,72 → 0,78, TVD 0,41 → 0,24.
**Trade-off ehrlich benennen:** Basic „terminiert" besser (93,9 % vs. 2 %),
weil es Terminal-Aktivitäten hart abschneidet — das motivierte D4.

### D2 — Modellquelle: manuelles (Signavio-)BPMN vs. aus dem Log gemint (1.4)
**Entscheidung:** Signavio-Modell behalten.
**Evidenz** (`output/validation/bpmn_source_comparison/`): Inductive Miner
(noise 0.2) fittet das echte Log nur marginal besser (Token-Replay: 73,6 % vs.
68,8 % voll passend; Ø-Fitness 0,993 vs. 0,976), ist aber in der Simulation
deutlich unpräziser (Precision 0,49 vs. 0,78, TVD 0,57 vs. 0,24) — und
terminiert identisch schlecht (je 2 %). **Schlussfolgerung:** Die
Nicht-Terminierung ist eine Branching-Eigenschaft, keine Modellfrage.
(Strenger Präfix-Replay: 57,7 % der 31 509 realen Cases passen exakt aufs
Signavio-Netz — Grenze der Trainingsdatenbasis für D5, im Report als
Limitation nennen.)

### D3 — Trace-Bigramme vs. Decision-Point-Wahrscheinlichkeiten (1.5)
**Entscheidung:** Branching-Wahrscheinlichkeiten werden per **Replay an den
Decision Points** gemined (`mine_dp_probs.py`), nicht aus Trace-Bigrammen.
**Begründung (gemessen):** Bigramme mischen Nebenläufigkeits-Interleavings in
die Schätzung — das reale nächste *Event* nach `A_Validating` (1. Besuch) ist
zu 93,6 % `O_Returned`, ein nebenläufiger Zweig, der an diesem Decision Point
gar nicht aktiviert ist. Die Simulation renormalisierte deshalb über falsche
Kandidaten.

### D4 — Terminierung als gelernte Entscheidung (der zentrale Fix)
**Diagnosekette (für den Report als Story):**
1. Aktivitätsfrequenzen sim/real: `W_Validate application` 8,6×,
   `W_Handle leads` 7,7×, Offer-Schleife ~1,9×; Exits massiv unterrepräsentiert.
2. Instrumentierung der Entscheidungen: Singleton-Frontier `[O_Cancelled]`
   wird 15,7×/Case ausgewertet — es gibt dort nichts zu wählen, die
   Simulation MUSS weiterfeuern.
3. Ursache: Terminierung prüfte nur `marking == final_marking`. An vielen
   Markierungen ist das Endmarking aber nur per Tau-Übergängen erreichbar,
   während ein sichtbares Loop-Back-Label aktiviert bleibt → Cases können
   strukturell nie aufhören.
**Fix (dreiteilig):**
1. `__END__` als Pseudo-Option überall dort, wo das Endmarking
   tau-erreichbar ist; P(END | Decision Point, Besuchszahl) wird beim Replay
   dort gezählt, wo reale Traces tatsächlich enden. Gemessen: am
   `[O_Cancelled]`-Punkt enden reale Cases zu **80 %** pro Besuch. END
   bekommt nie Residualgewicht — Aufhören muss durch Daten belegt sein.
2. Fallback-Kette der Branch-Wahl: DP-Bucket → DP-gesamt → Aktivitäts-Bucket
   (1/2/3+; z. B. `W_Validate application` → `O_Cancelled` steigt 3,5 % → 14 %
   → 27 %) → globale Tabelle → Residualgewicht.
3. **Terminal-Outcome-Regel:** Nach den Outcome-Milestones
   `A_Pending`/`A_Denied`/`A_Cancelled` endet der Case (real feuert exakt
   einer davon pro Case: 0,55 + 0,33 + 0,12 = 1,00). Das Netz hält nach
   diesen Markierungen Loop-Tokens am Leben und das Endmarking ist dort oft
   nicht tau-erreichbar — beobachtete Cases feuerten „O_Accepted, A_Pending"
   und drehten danach 50+ Events in der Validierungsschleife. Domänenregel
   statt Netzstruktur: Ist das Outcome entschieden, ist der Case vorbei.

### D5 — probs vs. rules (1.5 Advanced I)
Decision Trees pro Decision Point auf Spawn-/Offer-Attributen, Trainingsdaten
per Replay identifiziert. **Evaluation vorlesungskonform** (Deck 05):
temporaler 80/20-Split (Folie 37), Accuracy vs. Majority-Baseline +
Precision/Recall (macro) + ROC-AUC (OvR) (Folie 44), 16 trainierte Modelle
(`output/models/decision_rules_metrics.json`). Kernbefund: ein DP perfekt
trennbar (AUC 1,0: `A_Complete` vs. `O_Create Offer` — RequestedAmount>0),
mehrere kollabieren auf die Mehrheitsklasse (Acc 0,89 bei Macro-Precision
0,22!) — **Accuracy allein hätte das versteckt**, die Folie-44-Metriken
waren also notwendig. Leakage-Vermeidung: `Accepted`/`Selected` als
Features ausgeschlossen (kodieren das Label).

## 3. Ablation (die zentrale Ergebnistabelle)

Alle Läufe: Advanced-Prozessmodell, Seed 42, 30 Tage. Quelle:
`output/validation/branching_probs_vs_rules/*.json` und
`process_model_comparison/*.json`.

Ablationsstufen: *probs+terminal* isoliert die Terminal-Outcome-Regel,
*visit_bigram* die Besuchs-Buckets; *visit*/*rules* kombinieren DP-Mining,
END-Entscheidung und Terminal-Regel.

| KPI | Basic | Adv. probs | probs + terminal | visit (Bigram-Buckets) | **visit (final)** | rules (final) | Real |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fitness voll passend / Ø | 0 % / 0,64 | 100 % / 1,00 | 0,6 % / 0,95 | 100 % / 1,00 | 45,6 % / 0,97 | 48,2 % / 0,97 | — |
| Precision | 0,72 | 0,78 | 0,72 | 0,84 | 0,69 | 0,47 | — |
| Branching mean TVD | 0,41 | 0,24 | 0,38 | 0,23 | **0,11** | 0,15 | 0 |
| Top-20-Varianten reproduziert | 0/20 | 0/20 | 0/20 | 0/20 | **10/20 (16,4 %)** | 10/20 (16,4 %) | — |
| Case-Länge (Events) | 12,3 | 74,9 | 16,9 | 70,5 | 12,9 | **15,2** | 15,1 |
| Case-Länge rel. Fehler | 0,18 | 3,96 | 0,12 | 3,67 | 0,15 | **0,004** | 0 |
| Completion-Rate (30 d) | 93,9 % | 2,0 % | 37,3 % | 27,7 % | 71,1 % | **72,8 %** | s. u. |
| Case-Dauer rel. Fehler | 0,95 | 0,99 | 0,96 | 0,99 | 0,98 | 0,97 | 0 |

**Lesehilfe für den Report:**
- Keiner der Einzel-Fixes reicht: Terminal-Regel allein (probs+terminal)
  verschlechtert TVD (0,38) und reproduziert keine Varianten;
  Besuchs-Buckets allein (visit_bigram) lösen die Terminierung nicht (28 %).
  Erst DP-Mining + END + Terminal-Regel zusammen: TVD 0,11, 10/20 Varianten,
  Case-Länge ±15 %, Completion 71 %.
- **Fitness-Trade-off ehrlich diskutieren:** voll passende Traces fallen von
  100 % auf ~46 %, weil die Terminal-Regel Cases vor dem Netz-Endmarking
  beendet — die Ø-Trace-Fitness bleibt 0,97. Das ist eine dokumentierte
  Modell-Limitation des Signavio-Netzes (hält Loop-Tokens nach dem Outcome
  am Leben), keine Regression des Simulators: reale Traces enden dort auch.
- visit vs. rules (finale Stufen): visit gewinnt bei TVD (0,11 vs. 0,15) und
  Precision (0,69 vs. 0,47), rules bei Case-Länge (Fehler 0,004!) und
  Completion — beide reproduzieren 10/20 Varianten. Default = visit;
  rules ist der Advanced-I-Showcase mit vergleichbarer Qualität.
- **Drain-Analyse (Zensierungs-Beweis, `output/validation/horizon_censoring/drain.json`):**
  Gleiche Konfiguration, Ankünfte 30 Tage, aber Simulation läuft bis Tag 180
  leer. Ergebnis: Completion **99,2 %** (statt 71 % — die Lücke war
  Horizont-Zensierung) und Case-Dauer Ø **21,3 Tage vs. real 21,8 Tage —
  rel. Fehler 0,025**. Der scheinbare 96–98-%-Dauerfehler der
  30-Tage-Läufe war fast vollständig Survivorship-Bias (nur schnelle Cases
  wurden fertig). → Report-Headline: Das Modell trifft die reale
  Case-Dauer auf 2,5 %, wenn man unzensiert misst.
- **Kehrseite (ehrlich ausweisen):** Auf der unzensierten Population steigt
  die Ø-Case-Länge auf 46,0 Events (Fehler 2,05) — die im 30-Tage-Fenster
  zensierten Langläufer drehen zu viele Schleifenrunden. Ursache: Ab dem
  „5+"-Bucket ist die Exit-Wahrscheinlichkeit stationär → der Tail der
  Längenverteilung ist schwerer als real. TVD unzensiert 0,15 (zensiert
  0,11). Limitation dokumentieren; möglicher Hebel: feinere Besuchs-Buckets
  (z. B. bis 8+) beim DP-Mining.

## 4. Challenges (Report-Subsection „Challenges")

- Nebenläufigkeit verfälscht Bigram-Branching (D3) — klassisches
  Interleaving-Problem, per Replay-Mining gelöst.
- Terminierung in Netzen mit Tau-erreichbarem Ende + sichtbaren Loop-Backs
  ist eine *Entscheidung*, kein Zustand (D4).
- Nur 57,7 % der realen Traces passen exakt aufs Netz → Mining-Basis
  eingeschränkt; Token-Replay (68,8 %) ist toleranter als Präfix-Replay.
- Evaluations-Hygiene: Simulierte Logs immer auf natürlich abgeschlossene
  Cases filtern (`completed_cases.txt`), sonst Zensierungs-Bias; temporaler
  statt zufälliger Split (Drift!); Accuracy braucht Majority-Baseline +
  Precision/Recall/AUC.

## 5. Limitationen / offene Punkte (ehrlich in den Report)

- Case-Dauer/Completion abhängig von 1.3 (Warte-/Servicezeiten) und 1.6
  (Verfügbarkeiten) — außerhalb 1.4/1.5.
- DP-Tabellen aus den 57,7 % exakt passenden Cases gemined (Selektionsbias
  möglich).
- 1.5 Advanced II (History-Features in den DP-Klassifikatoren) wäre mit der
  vorhandenen Replay-Infrastruktur ein kleiner Schritt — bewusst nicht
  umgesetzt, da Advanced I erfüllt ist (Entweder-Oder im Assignment).

## 6. Datei-Index (alle Zahlen reproduzierbar)

| Datei | Inhalt |
|---|---|
| `output/validation/process_model_comparison/{basic,advanced}.json` | D1 |
| `output/validation/bpmn_source_comparison/real_log_replay_im02.json`, `advanced_im02.json` | D2 |
| `output/validation/branching_probs_vs_rules/advanced_{probs,probs_terminal,visit_bigram,visit,rules}.json` | D3/D4/D5-Ablation (5 Stufen) |
| `output/models/decision_rules_metrics.json` | D5 (16 DP-Modelle, temporaler Split) |
| `simulation/models/dp_branching_probs.json` | gemint: P(Label bzw. END | DP, Besuch) |
| `simulation_inputs.json` → `branching_probs_by_visit` | Besuchs-Buckets 1/2/3+ |
| Repro-Kommandos | `scripts/compare_process_models.py --configs advanced --branching-mode {probs,visit,rules}`; `scripts/mine_dp_probs.py --log data/BPIChallenge2017.xes.gz`; `scripts/discover_process_model.py --log …` |

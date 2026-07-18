# Roadmap: Simulation (Teil I) + Optimierung (Teil II) — Ziel 1,0

Stand: 2026-07-11. **Abgabe: Sonntag, 19.07.2026** (8 Tage!). Baseline-Zahlen
stammen aus `scripts/compare_process_models.py` (Seed 42, 30 Tage). Bitte nach
jeder Änderung neu laufen lassen und die Deltas in den PR schreiben.

**Crunch-Descoping (wegen 8-Tage-Frist):** D3 (Deep RL) gestrichen; A5
gestrichen; Management-Frage 1 (nicht 2). Alles Übrige ist Pflicht.
A4-Update: Marios MDN-Arrival-Modell (`arrival_mdn.py` + `train_arrival_mdn.py`)
ist seit dem Pull vom 11.07. da — offen ist nur noch die KPI-Validierung
(Interarrival-/Tagesprofil-Fehler MDN vs. LogNormal).

---

## 0. Ausgangslage

### Was schon steht (Teil I)

| Abschnitt | Basic | Advanced |
|---|---|---|
| 1.1 Engine (DES, globale Queue, CSV-Log) | ✅ | — |
| 1.2 Arrivals | ✅ LogNormal statisch | 🟡 MDN-Modell vorhanden (`arrival_mdn.py`), KPI-Validierung offen |
| 1.3 Processing Times | ✅ Verteilungen + ML-Punktmodell | ✅ Adv. I (Quantilregression) · ❌ Adv. II (Warte-/Servicezeit-Trennung) |
| 1.4 Prozessmodell | ✅ Wahrscheinlichkeitsgraph | ✅ BPMN → Petrinetz |
| 1.5 Branching | ✅ empirische Wahrscheinlichkeiten | ✅ Adv. I (Attribut-Sampling + Decision-Tree-Regeln, `--branching-mode rules`) — empirisch noch nicht validiert |
| 1.6 Ressourcenverfügbarkeit | ⚠️ **nur feste Kapazität — das geforderte Intervall-Modell fehlt (Basic-Pflicht!)** | ❌ Kalender/Schichten |
| 1.7 Permissions | ✅ historische Map (nur Top-20 Ressourcen) | ❌ Rollen-Discovery (OrdinoR) |
| 1.8 Allocation | ✅ random (mehr verlangt Teil I nicht) | → Teil II |

### Baseline-KPIs (Ist-Zustand, `compare_process_models.py`)

| KPI | Basic | Advanced | Referenz (BPIC-17) |
|---|---:|---:|---:|
| Control-flow fitness (voll passend) | 0,0 % | 100 % | — |
| Control-flow precision | 0,72 | 0,78 | — |
| Branching mean TVD | 0,41 | 0,24 | 0 = identisch |
| Abgeschlossene Cases / 30 Tage | 2 177 | **46** | ~2 580 |
| Case-Länge (Events, Ø) | 12,3 | **74,9** | 15,1 |
| Case-Dauer (Ø) | ~1,2 d | ~0,27 d | **~21,8 d** |
| Top-20-Varianten reproduziert | 0/20 | 0/20 | — |

### Die drei kritischen Befunde

1. **Advanced-Modell terminiert nicht.** Nur 46/2 318 Cases werden fertig;
   Ø 75 Events pro Case statt 15. Cases kreisen in den W_-Schleifen des
   Petrinetzes, weil memorylose Branching-Wahrscheinlichkeiten an
   Schleifen-Exits die reale Abbruchdynamik nicht abbilden.
2. **Ressourcen-Contention ist wirkungslos.** `process.py:637` führt
   Aktivitäten auch ohne zugewiesene Ressource aus
   (`resource = event.resource or self._unknown`), und der
   Warteschlangen-Code in `resource.py` (`on_resource_available`, Z. 216 ff.)
   ist tot: `if resource in RESOURCE_PERMISSIONS.get(resource, set())`
   vergleicht den Ressourcen-Namen mit einer Aktivitäten-Menge — immer False.
   Konsequenz: Es gibt keine Warteschlangen → **keine Allokationsstrategie in
   Teil II kann irgendeinen Effekt zeigen.** Das ist der wichtigste Umbau.
3. **Wartezeiten fehlen strukturell.** Case-Dauer ~95 % zu kurz. Genau die
   Rozinat-Caveat aus `docs/paper_insights_discovering_simulation_models.md`:
   Bearbeitungszeiten enthalten implizit Wartezeit, aber es gibt weder
   Kundenantwort-Delays noch Ressourcen-Warteschlangen noch Kalender.

---

## 1. Forschungsfragen (roter Faden für Report + Präsentation)

- **RQ1 (Teil I):** Wie realitätsnah kann ein datengetriebenes DES-Modell den
  BPIC-17-Prozess reproduzieren — und welche Modellkomponente trägt wie viel
  dazu bei? (Ablation: Branching probs vs. rules, Verteilungen vs. ML-Zeiten,
  Kapazität vs. Kalender-Verfügbarkeit.)
- **RQ2 (Brücke):** Welche Modelleigenschaften muss ein Simulator haben, um
  als Testbed für Ressourcen-Allokation zu taugen? (These: explizite
  Warte-/Servicezeit-Trennung + echte Contention sind notwendig — sonst sind
  Policy-Unterschiede nicht messbar.)
- **RQ3 (Teil II):** Wie schneiden klassische Heuristiken (R-RRA, R-RMA,
  R-SHQ) vs. k-Batching vs. prädiktive/optimierende Allokation unter
  **unsicheren Ressourcenverfügbarkeiten** ab (Zykluszeit, Auslastung,
  Fairness)?
- **RQ4 (Teil II, advanced):** Welche Trade-offs entstehen zwischen Effizienz
  und Fairness, und wie robust sind die Policies gegen Regime-Wechsel
  (Nachfrage-Spitze, Ressourcen-Ausfall)?
- **RQ5 (Management-Frage, Folie 23):** Welche zwei Ressourcen können mit
  minimalem Performance-Verlust entfallen — bzw. was kostet eine
  9-to-5-Beschränkung der Arbeitszeiten? (Eine der beiden ist Pflicht,
  s. Phase D.)

---

## 2. Phasenplan

### Phase A — Simulation valide machen (Teil I abschließen)

> Ziel: simuliertes Log „relativ nah am echten Log", damit Teil II
> aussagekräftig wird. Reihenfolge = Priorität.

**A1. Terminierungsproblem im Advanced-Modell — ✅ ERLEDIGT (14.07.)**
- Drei Ursachen gefunden und behoben (Evidenz:
  `output/validation/branching_probs_vs_rules/`):
  1. Trace-Bigramme mischen Nebenläufigkeit in die Branch-Schätzung →
     Decision-Point-Wahrscheinlichkeiten per Replay gemined
     (`scripts/mine_dp_probs.py`), konditioniert auf DP-Besuchszähler.
  2. Enden war nie eine Option: An vielen Markierungen ist das Final-Marking
     nur per Tau erreichbar, während sichtbare Loop-Labels enabled bleiben →
     `__END__` als datengetriebene Wahl mitgemined (z. B. [O_Cancelled]:
     80 % END real).
  3. Outcome-Semantik: Nach A_Pending/A_Denied/A_Cancelled (real exakt 1,00
     pro Case) hält das Netz Loop-Tokens am Leben → Terminal-Outcome-Regel.
- Ergebnis (`--branching-mode visit`, jetzt Default): Completion 2 % → 71 %,
  Case-Länge-Fehler 3,96 → 0,15, Branching-TVD 0,24 → **0,11**, 10/20 echte
  Top-Varianten reproduziert. Trade-off: Token-Replay-Fitness 100 % → 0,97
  (das Netz kann real endende Traces nicht voll replayen — Modellgrenze, im
  Report diskutieren). `rules` (1.5 Adv. I) liegt gleichauf (Case-Länge
  0,004!), Precision aber schwächer (0,47 vs. 0,69).
- Rest (~29 % der Cases im Horizont unfertig): langsame Validierungsrunden
  (tagelange Full-Durations) — löst sich mit A2 (Service-/Wartezeit-Split).
- **Entschieden (12.07., Evidenz in `output/validation/bpmn_source_comparison/`):**
  Das Signavio-BPMN bleibt Simulationsbasis. Ein Inductive-Miner-Modell
  (noise 0.2) fittet das echte Log nur marginal besser (73,6 % vs. 68,8 %
  voll passende Traces), ist aber deutlich unpräziser (Precision 0,49 vs.
  0,78; TVD 0,57 vs. 0,24) — und terminiert **identisch schlecht** (2 %).
  ⇒ Die Nicht-Terminierung liegt am memorylosen Branching an
  Schleifen-Exits, nicht an der Modellquelle. Genau das fixt A1.
- Schritte: Schleifen-Exits analysieren (welche Decision Points erzeugen die
  75er-Traces?); `--branching-mode rules` gegen `probs` benchmarken;
  falls nötig historien-abhängige Branching-Wahrscheinlichkeiten
  (z. B. P(Exit) steigt pro Schleifendurchlauf, geometrisch aus dem Log
  gefittet).
- Metriken: Case-Länge rel. Fehler (Ziel ≤ 0,15), Anteil abgeschlossener
  Cases (Ziel ≥ 90 % bei 30-Tage-Horizont + Nachlauf), Fitness bleibt 100 %,
  Branching-TVD (Ziel ≤ 0,1), Top-20-Variantenabdeckung (Ziel: > 0 → möglichst
  > 50 % des realen Traffics).
- Designentscheidung: probs vs. rules vs. historien-abhängig — nach KPI-Delta.

**A1-Update (18.07.): Regression durch BPMN-Gateway-Änderung — Ursache gefunden, gefixt**
- Versuch, den O_Returned-Abschnitt "realistischer" zu modellieren: zwei
  Gateways dort von `parallelGateway` (AND) auf `inclusiveGateway` (OR)
  geändert. Ergebnis war schlechter als vorher, Änderung wieder verworfen
  (BPMN zurück auf den committeten AND-Stand, `git checkout`). Gleichzeitig
  war unbemerkt eine zweite Änderung im Arbeitsverzeichnis: die
  Outcome-Terminierungsregel aus A1 (Punkt 3 oben, `TERMINAL_OUTCOMES` in
  `petri_process.py`) war entfernt worden — beide Änderungen zusammen
  reproduzierten den ursprünglich gemeldeten Bad-Case (Completion 24 %,
  Precision 0,22, Case-Länge-Fehler 0,75, `final_marking`-End-Reason **0/556**).
- Systematische Ablation (`scripts/compare_process_models.py`, jetzt mit
  `--terminal-outcomes {on,off}`-Flag und vollem `end_reasons`-Log inkl.
  `terminal_outcome`), sechs Konfigurationen, identischer Harness (Seed 42,
  30 Tage, capacity=3):

  | Run | BPMN-Gateway | terminal_outcomes | branching-mode | Completion | Precision | TVD | Top-20 |
  |---|---|---|---|---:|---:|---:|---:|
  | baseline_reproduce | AND (parallel) | on | visit | 17,95 % | 0,615 | 0,091 | 10/20 |
  | bpmn_inclusive | OR (inclusive) | on | visit | 24,2 % | **0,275** | **0,255** | **0/20** |
  | terminal_off | AND | **off** | visit | **6,82 %** | 0,587 | 0,256 | 7/20 |
  | mode_probs | AND | on | probs | 16,26 % | 0,598 | 0,093 | 10/20 |
  | mode_rules | AND | on | rules | 19,41 % | 0,495 | 0,153 | 10/20 |
  | worst_case_repro | OR | off | rules | 24,0 % | 0,221 | 0,150 | 6/20 |

  (worst_case_repro reproduziert den ursprünglich gemeldeten Bad-Case fast
  exakt und schließt damit den Kreis.)
- **Befund 1 (Hauptursache):** Der Gateway-Wechsel (`bpmn_inclusive` vs.
  `baseline_reproduce`) allein zerstört Precision (0,615→0,275), TVD
  (0,091→0,255) und Top-20-Abdeckung (10/20→0/20), obwohl Completion leicht
  steigt. Grund: `scripts/mine_dp_probs.py` mined die
  Decision-Point-/`__END__`-Wahrscheinlichkeiten per Replay **gegen genau
  dieses Netz** — ändert man die Gateway-Struktur, passt die Kalibrierung
  nicht mehr. Sichtbar an `allow_end_without_dp`: 0/952 (AND) vs. 20328/22019
  = **92 %** (OR) — mit OR-Gateway hat die Mehrheit der neu entstehenden
  "Ende erlaubt"-Momente keine gemined Daten mehr. ⇒ **BPMN bleibt beim
  AND-Gateway (Signavio-Original).**
- **Befund 2 (bestätigt A1 Punkt 3):** `terminal_off` allein (AND-Gateway,
  sonst identisch zu baseline) halbiert-drittelt die Completion nochmal
  (17,95 %→6,82 %) — die Outcome-Terminierungsregel ist weiterhin
  tragend. ⇒ **`enforce_terminal_outcomes=True` bleibt Default** (jetzt als
  benannter, testbarer Parameter statt stillschweigend im Code entfernt —
  siehe `tests/test_process_model_petri.py`).
- **Befund 3 (branching-mode):** Unter AND-Gateway liegen `probs` und
  `visit` bei TVD praktisch gleichauf (0,093 vs. 0,091); `rules` ist
  schwächer (TVD 0,153, Precision 0,495). `visit` bleibt Default.
- **Offener Punkt, nicht weiter verfolgt (Zeitdruck vor Abgabe):** Keiner der
  sechs Läufe erreicht `final_marking > 0` — auch `baseline_reproduce`
  (AND + terminal_outcomes on + visit, die Konfiguration, die A1 oben als
  "71 % Completion" dokumentiert) landet nur bei 17,95 % Completion, alle
  Abschlüsse über die `terminal_outcome`-Regel. Die A1-71%-Zahl wurde
  vermutlich unter etwas anderen Bedingungen gemessen (anderes Artefakt-
  /Datenrefresh von `dp_branching_probs.json`/`simulation_inputs.json`
  zwischen 14.07. und heute); die *relativen* Vergleiche innerhalb dieser
  Ablation (identischer Harness für alle 6 Läufe) bleiben davon unberührt
  und tragen die Schlussfolgerungen oben. Für den Report: aktuelle
  Advanced-Zahlen (Completion ~18 %, Case-Dauer-Fehler ~0,60) ehrlich als
  Ist-Stand nennen, nicht die alte 71%-Zahl zitieren.

**A1-Update (18.07., Teil 2): Simulationshorizont 30 → 60 Tage (echte Perzentile statt Rundzahl)**
- Ausgangsproblem: der harte 30-Tage-Horizont war eine willkürliche Rundzahl
  (kein Kommentar/Doku begründet sie), aber die reale mittlere Case-Dauer
  liegt bei 21,85 Tagen mit schwerem rechten Ausläufer — ein 30-Tage-Fenster
  zensiert damit systematisch gerade die langsamen (aber realen) Cases weg
  und macht die Completion-Rate/Case-Dauer-Fehler teils zum Artefakt der
  Fensterbreite statt des Modells (vgl. `scripts/drain_analysis.py`,
  bereits vorher im Team bekannt: Completion 71 %→99,2 % bei Drain bis
  Tag 180).
- Reale Case-Dauer-Perzentile neu berechnet (`data/BPIChallenge2017.xes.gz`,
  Events auf `lifecycle='complete'` gefiltert wie in
  `extract_log_info.filter_to_complete`, Case-Dauer = letzter − erster
  Zeitstempel pro Case): p50 = 19,06 d, p70 = 30,81 d, **p80 = 31,94 d**,
  p90 = 35,00 d, **p99 = 59,15 d**. Gegen Zensierung am Log-Ende geprüft
  (Perzentile bleiben stabil, ob Cases mit <30/45/60/90 Tagen Restlaufzeit
  vor Log-Ende ausgeschlossen werden oder nicht) — kein Messartefakt.
- Gewählt: **p99 ≈ 60 Tage** (glatt gerundet) statt p80 (~32 Tage, kaum
  länger als die alten 30 Tage und hätte am Zensierungsproblem fast nichts
  geändert). Neuer Default in `simulation/main.py::SIM_DURATION_SECONDS`
  und `scripts/compare_process_models.py::SIM_DURATION_SECONDS`.
- **Achtung, kein Drain:** Ankünfte laufen weiterhin über den ganzen
  Horizont (kein separates Ankunfts-Cutoff + Nachlaufzeit wie in
  `drain_analysis.py`), d. h. Zensierung ist damit *reduziert*, nicht
  eliminiert — ein Case, der an Tag 59 ankommt, hat immer noch fast keine
  Zeit.
- **Ergebnis (Basic vs. Advanced, sonst identische Config wie oben:
  AND-Gateway, `terminal_outcomes=on`, `branching-mode visit`):**

  | KPI | Advanced @30d | Advanced @60d | Basic @30d | Basic @60d |
  |---|---:|---:|---:|---:|
  | Cases gestartet / abgeschlossen | 2318 / 416 | 4842 / 822 | 2318 / 2252 | 4842 / 4012 |
  | Completion-Rate | 17,95 % | 16,98 % | 97,15 % | 82,86 % |
  | Case-Dauer rel. Fehler | 0,600 | **0,103** | 0,985 | 0,907 |
  | Case-Länge rel. Fehler | 0,231 | 0,243 | 0,058 | 0,054 |
  | Precision | 0,615 | 0,682 | 0,679 | 0,680 |
  | Branching TVD | 0,091 | 0,092 | 0,480 | 0,476 |

  Case-Dauer-Fehler des Advanced-Modells fällt von 0,60 auf **0,10** — die
  bei weitem größte Einzelverbesserung seit Beginn der Ablation, und zeigt
  konkret (nicht nur beim Team-internen Drain-Experiment), dass der
  Case-Dauer-Fehler zu einem großen Teil Zensierungsartefakt war, kein
  Modellfehler. Completion-Rate ändert sich kaum (mehr Zeit hilft wenig,
  wenn — s. A1-Befund oben — die meisten Cases sowieso nie das echte
  Final-Marking erreichen, sondern über die Terminal-Outcome-Regel enden,
  die nicht vom Horizont abhängt).
- **Nebeneffekt, ehrlich ausweisen:** Basics Completion-Rate sinkt (97 %→
  83 %), weil die Case-Zahl mit dem Horizont mitskaliert (2318→4842 bei
  gleicher Ankunftsrate) während die Ressourcenkapazität fix bleibt
  (`capacity_per_resource=3` in `compare_process_models.py`) — mehr
  gleichzeitig laufende Cases erhöhen die Warteschlangen-Last. Das ist ein
  erwarteter Kapazitäts-/Horizont-Interaktionseffekt, keine Regression.
- Nicht angefasst: die historische Ablationstabelle oben (A1-Update, Teil 1)
  wurde unter dem alten 30-Tage-Horizont gemessen — die dort isolierten
  Effekte (Gateway-Typ, Terminal-Regel, Branching-Mode) sind Horizont-
  unabhängig und bleiben gültig, nur die absoluten Case-Dauer-Zahlen dort
  sind durch das inzwischen behobene Zensierungsproblem verzerrt.

**A1-Update (18.07., Teil 3): Warum 10 Aktivitäten im Advanced-Log nie auftauchen**
- Betroffene Aktivitäten (`activities_missing_in_run` in advanced.json):
  `A_Cancelled`, `A_Denied`, `A_Pending`, `O_Cancelled`, `O_Refused`,
  `O_Sent (online only)`, `W_Assess potential fraud`, `W_Call after offers`,
  `W_Personal Loan collection`, `W_Shortened completion `. Drei
  unterschiedliche Ursachen, empirisch verifiziert (Petri-Net-BFS +
  `dp_branching_probs.json` + reale Aktivitätshäufigkeiten aus
  `simulation_inputs.json`), keine davon Zufall:
  1. **Fehlen im Modell (5/10):** `O_Sent (online only)`,
     `W_Assess potential fraud`, `W_Call after offers`,
     `W_Personal Loan collection`, `W_Shortened completion ` existieren als
     Transition **gar nicht** in `bpic17_process.bpmn` (geprüft: die
     konvertierte Petri-Netz-Transitionsliste enthält nur 21 sichtbare
     Labels, diese 5 fehlen komplett) — kein Branching-/Wahrscheinlichkeits-
     problem, sondern eine Modellabdeckungslücke. Real sind vier davon auch
     selten (W_Personal Loan collection 0,01 % der Cases, W_Shortened
     completion 0,24 %, W_Assess potential fraud 0,90 %, W_Call after
     offers 1,09 %) — plausibel im Signavio-Modell/bei der Discovery
     weggelassen. **Aber `O_Sent (online only)` ist real 6,43 % der Cases**
     (2026 Vorkommen) — keine Ausreißer-Seltenheit, sondern schlicht im
     Modell vergessen.
  2. **Metrik-Artefakt, keine echte Abwesenheit (3/10):** `A_Cancelled`,
     `A_Denied`, `A_Pending` feuern nachweislich (820/146/678 Events im
     60-Tage-Advanced-Lauf, exakt Summe = 822 = alle `terminal_outcome`-
     Enden). `metrics.branching_divergence` (`scripts/metrics.py:120-127`)
     bildet aber `next_activity` per `shift(-1)` und verwirft Zeilen ohne
     Nachfolger — und da `enforce_terminal_outcomes` den Case exakt bei
     diesen drei Aktivitäten sofort beendet, haben sie **nie** einen
     Nachfolger im selben Case. Die Meldung "activity never occurred in
     this run" ist für diese drei schlicht falsch/irreführend; sie
     bedeutet nur "hatte keinen aufgezeichneten Nachfolger".
  3. **Strukturell durch `enforce_terminal_outcomes` blockiert (2/10):**
     `O_Cancelled` und `O_Refused` feuern tatsächlich **null Mal** in 4842
     gestarteten Cases — real aber 66,3 % bzw. 14,9 % der Cases (!). Petri-
     Net-BFS zeigt: `O_Cancelled` ist an **genau einer** erreichbaren
     Markierung aktiviert, und dort ist es die einzig legale Aktivität
     (deterministisch) — laut `dp_branching_probs.json` wird diese
     Markierung nur über `A_Cancelled` erreicht. Reale Bigram-Daten
     (`simulation_inputs.json`): `A_Cancelled → O_Cancelled` p=0,9959,
     `A_Denied → O_Refused` p=0,9965 — im echten Log ist die O_-Aktivität
     praktisch immer die unmittelbare administrative Bestätigung nach der
     A_-Entscheidung (bereits im Code-Kommentar zu `TERMINAL_OUTCOMES`
     angedeutet: "the trace ends right after (bar minor wrap-up events)").
     Weil `enforce_terminal_outcomes` den Case aber **sofort** bei
     `A_Cancelled`/`A_Denied` beendet (vor jedem `_advance_to_next_visible`-
     Aufruf), kommt die Simulation nie bis zu diesem "minor wrap-up"-
     Schritt. Für `O_Refused` gibt es laut BFS auch andere, von `A_Denied`
     unabhängige Zugangswege (4 Frontier-Kombinationen inkl.
     `A_Validating`), die aber im 60-Tage-Lauf nie getroffen wurden — dafür
     reicht die Datenlage nicht für eine abschließende Aussage, ob das
     Zufall (seltener Pfad) oder ein weiterer struktureller Engpass ist.
- **Einordnung:** Das ist der gleiche Trade-off wie beim Fitness-Verlust in
  A1 — `enforce_terminal_outcomes` kauft Completion/Realismus auf
  Case-Ebene (das Ergebnis der Kreditentscheidung ist korrekt simuliert)
  gegen den Verlust der administrativen Nachlauf-Aktivitäten O_Cancelled/
  O_Refused, die real fast immer, aber strukturell erst *nach* dem Punkt
  auftreten, an dem die Simulation den Case für beendet erklärt. Keine
  Regression, keine Aktion für die Abgabe morgen — aber ein ehrlicher
  Punkt für die "Limitations"-Sektion des Reports, und ein potenzieller
  Hebel für später: `O_Cancelled`/`O_Refused` explizit als unmittelbare
  Folgeaktivität von `A_Cancelled`/`A_Denied` feuern lassen, bevor der Case
  endet (statt sie komplett auszulassen).

**A1-Update (18.07., Teil 4): TVD bereinigt um Modell-Abdeckungslücken**
- Motivation: die 5 nicht im BPMN modellierten Aktivitäten
  (`O_Sent (online only)`, `W_Assess potential fraud`, `W_Call after
  offers`, `W_Personal Loan collection`, `W_Shortened completion `)
  verzerrten die Branching-TVD nicht nur durch ihren eigenen (informativen,
  aber harmlosen) "absent"-Eintrag, sondern **still auch die TVD anderer,
  im Modell vorhandener Aktivitäten** — immer wenn das reale Log diesen
  fehlenden Aktivitäten Wahrscheinlichkeitsmasse als Nachfolge-Ziel gibt
  (z. B. real `O_Created → O_Sent (online only)`: 4,47 %), kann die
  Simulation diese Masse nie erreichen — eine Modell-Abdeckungslücke, keine
  Kalibrierungsschwäche der Branching-Wahrscheinlichkeiten.
- Fix (`scripts/metrics.py::branching_divergence`, neuer Parameter
  `modeled_activities`, befüllt aus den Petri-Netz-Transitionslabels in
  `evaluate()`): Quell-Aktivitäten, die selbst nicht im Modell sind, werden
  jetzt separat als `activities_not_in_bpmn` geführt (TVD bleibt `None`,
  da keine Branching-Wahrscheinlichkeit sie je erreichen könnte). Für alle
  anderen Aktivitäten werden Nicht-Modell-Ziele aus der Referenzverteilung
  entfernt und die verbleibende Masse renormalisiert, bevor die TVD
  berechnet wird — die ausgeschlossene Masse wird zusätzlich pro Aktivität
  unter `excluded_target_mass` reported, nicht stillschweigend verworfen.
  Getestet in `tests/test_branching_divergence.py` (5 Tests).
- Effekt (60-Tage-Lauf, sonst identische Config): Advanced-TVD **0,0922 →
  0,0892**, Basic-TVD 0,480 → 0,428 (Basic nutzt dieselbe Referenz). Die
  "activities never reached"-Liste schrumpft von 8 auf die tatsächlich
  aussagekräftigen 3: `O_Cancelled`, `O_Refused`, `W_Validate application`
  — genau die Fälle, die im Modell existieren, aber von der Simulation nie
  erreicht werden (s. Teil 3 oben für die Ursachenanalyse dieser drei).
  Kein Effekt auf Teil II (`scripts/metrics.py` wird nur von
  `compare_process_models.py`/`drain_analysis.py`/`eval_lifecycle.py`
  importiert, keines davon aus dem Teil-II-Pfad
  `scripts/run_experiments.py`/`scripts/opt_metrics.py` — geprüft per
  Import-Grep).

**A1-Update (18.07., Teil 5): O_Cancelled/O_Refused als erzwungener Folgeschritt — umgesetzt**
- Umsetzung des in Teil 3 vorgeschlagenen Hebels: `FORCED_TERMINAL_FOLLOWUP =
  {"A_Cancelled": "O_Cancelled", "A_Denied": "O_Refused"}`
  (`simulation/components/petri_process.py`). Wenn `enforce_terminal_outcomes`
  eine dieser beiden Aktivitäten force-beendet, wird zuerst deterministisch
  die gemappte Folgeaktivität auf dem Petrinetz gefeuert (`_fire_activity`,
  einzig legale Transition an dieser Markierung, s. Teil 3) und direkt in den
  Logger geschrieben (nicht über die Event-Queue re-scheduled — sonst würde
  `on_activity_complete` erneut greifen und die Aktivität wie eine normale
  Verzweigung behandeln, statt sie zu erzwingen). `A_Pending` bleibt
  unverändert sofort terminierend (kein einzelner deterministischer
  Folgeschritt vorhanden, s. Teil 3).
- Vor der Umsetzung empirisch verifiziert (synthetische Trace-Ergänzung +
  `pm4py.fitness_token_based_replay` auf dem *bestehenden* simulierten Log,
  ohne den Simulator zu ändern): Anhängen von O_Cancelled/O_Refused an die
  483 von 822 Cases, die mit A_Cancelled/A_Denied endeten, hob
  `perc_fit_traces` bereits im Test von 35,4 % auf 93,2 % — das war die
  Entscheidungsgrundlage, den Umbau trotz Zeitdrucks vor der Abgabe zu
  wagen.
- **Gemessener Effekt (60-Tage-Lauf, identische Config, echte
  End-to-End-Simulation, nicht nur der synthetische Test):**

  | KPI | vorher | nachher |
  |---|---:|---:|
  | Fully-fitting traces | 35,4 % | **93,2 %** |
  | Ø Trace-Fitness | 0,962 | **0,996** |
  | Precision | 0,682 | 0,684 |
  | Branching TVD | 0,089 | **0,079** |
  | Top-20-Varianten | 10/20 (16,4 %) | **17/20 (39,0 %)** |
  | Case-Länge rel. Fehler | 0,243 | **0,204** |
  | Case-Dauer rel. Fehler | 0,103 | 0,103 (unverändert) |
  | Completion-Rate | 0,170 | 0,170 (unverändert) |

  Verbesserung auf praktisch jeder Achse, die die Kontrollfluss-Treue misst
  (Fitness, TVD, Varianten, sogar Case-Länge als Nebeneffekt), bei exakt
  gleicher Completion-Rate und Case-Dauer — kein Trade-off gefunden.
  `activities_always_terminal_in_run` verschiebt sich korrekt von
  `A_Cancelled, A_Denied, A_Pending` zu `A_Pending, O_Cancelled, O_Refused`
  (die neuen Endpunkte). "Never reached" schrumpft von 3 auf nur noch
  `W_Validate application`.
- Getestet: `tests/test_process_model_petri.py::
  test_terminal_outcome_fires_forced_followup_before_ending` (neuer
  Fixture-Aufbau, prüft Marking-Update, direktes Logging ohne Re-Entry in
  `on_activity_complete`, und dass der Case weiterhin über `terminal_outcome`
  endet). Alle 112 Tests grün.
- **Bewusst nicht angefasst:** `A_Pending` (kein einzelner deterministischer
  Folgeschritt, s. Teil 3 — Risiko einer erneuten Nicht-Terminierung wie vor
  A1); Completion-Rate (unverändert 17 %, da dieser Fix nur die
  *Trace-Qualität* der bestehenden Completions verbessert, nicht *wie viele*
  Cases abschließen — das bleibt der offene, größere Befund aus A1).

**A1-Update (18.07., Teil 6): Completion-Rate systematisch untersucht — drei Ursachen, nicht eine**
- Frage: ist 17 % Completion (60-Tage-Horizont) ein Simulationsfehler, oder
  zeigt das reale Log unter derselben Messmethode auch so wenig?
- **Referenzwert 1 — reales Log, identische Fenster-Methodik** (Cases, die
  innerhalb eines festen Zeitfensters ab Log-Start ankommen; Completion =
  letztes Event vor Fensterende, NICHT case-relative Perzentile):
  30-Tage-Fenster → **36,95 %** (793/2146), 60-Tage-Fenster → **63,29 %**
  (2927/4625). ⇒ Selbst der reale Prozess sieht unter dieser (zensierenden)
  Messung unvollständig aus — das ist also *kein* Beweis, dass unsere 17 %
  in Ordnung sind, aber ein Beleg, dass ein gewisser Anteil der Lücke
  Messmethodik ist, nicht Modellfehler.
- **Referenzwert 2 — Drain-Analyse unter aktueller Config** (Ankünfte 30 Tage,
  Engine läuft bis Tag 180, `scripts/drain_analysis.py`, jetzt inkl. A1-Fix
  Teile 1–5): Completion **71,8 %** (1666/2319) — weit über den 17 % beim
  harten Horizont, aber auch weit unter den ~100 %, die das reale Log bei
  unbegrenzter Zeit erreicht (jeder Case feuert irgendwann einen Outcome).
  ⇒ Zensierung erklärt den *größten* Teil der Lücke (17 %→72 % nur durch
  mehr Zeit), aber nicht alles: **~28 % der Cases werden selbst mit
  180 Tagen nie fertig.**
- **Ursache 1 der Rest-Lücke getestet und verworfen: Ressourcenkapazität.**
  `compare_process_models.py`/`drain_analysis.py` nutzen standardmäßig
  `DEFAULT_PERMISSIONS` (17 Ressourcen, `capacity_per_resource=3`) statt des
  vollen OrdinoR-Orgmodells (144 Ressourcen, `models/permissions_orgmodel.json`)
  — acht mal mehr Kapazität. Mit Orgmodell-Permissions: Completion nur
  71,8 %→**74,1 %** (1719/2319), mittlere Wartezeit sinkt zwar deutlich
  (426 649 s → 86 404 s), aber die Completion-Rate bewegt sich kaum. ⇒ Die
  fehlenden Ressourcen im Validierungs-Harness sind ein kleiner, aber nicht
  der entscheidende Faktor.
- **Ursache 2 der Rest-Lücke gefunden: zu grobe Besuchs-Buckets im
  gemineden Branching.** Mit mehr Ressourcen (schnellerer Durchlauf pro
  Aktivität) steigt die mittlere Event-Zahl unfertiger Cases von 50 auf
  **172** (max. 266) — weit unter dem Loop-Guard (400) — und **alle** 600
  unfertigen Cases hängen zuletzt in genau derselben Warteschleife fest:
  `W_Validate application` / `A_Validating` / `A_Incomplete`. Das deckt sich
  mit einer bereits früher notierten, nie behobenen Beobachtung (Kehrseite
  der Drain-Analyse, s. `docs/report_notes_1.4_1.5.md`): die
  Besuchs-Buckets in `branching_probs_by_visit`/`dp_branching_probs.json`
  gehen nur bis "5+" und die Exit-Wahrscheinlichkeit ist ab dort stationär
  (steigt nicht weiter mit mehr Schleifendurchläufen wie im echten Log) —
  ein kleiner Teil der Cases "würfelt" dadurch sehr lange erfolglos weiter,
  ohne dass mehr Ressourcen oder mehr Zeit allein das beheben. Fix wäre ein
  Re-Mining mit feineren Buckets (z. B. bis "8+"), keine Code-Änderung —
  zeitlich zu riskant für die Abgabe morgen, daher nicht umgesetzt.
- **Formale Petrinetz-Konformität geprüft:** intern per Konstruktion
  garantiert — `_fire_activity` feuert ausschließlich Transitionen aus
  `semantics.enabled_transitions(...)`, eine illegale Markierung kann so
  nicht entstehen (`_assert_marking_legal` ist ein zusätzliches Sicherheitsnetz
  für den komplexeren aktiven-Lifecycle-Terminalpfad, nicht die einzige
  Absicherung). Die verbleibenden 6,8 % nicht voll passenden Traces (56 von
  822, nach Teil 5) sind jetzt exakt attribuiert
  (`pm4py.conformance_diagnostics_token_based_replay`, per Trace geprüft):
  48 enden in `A_Pending` (bewusst nicht gefixt, s. Teil 5), 8 in
  `O_Refused` (dessen Folge-Markierung ist nicht immer tau-nah an `fm`,
  s. Teil 3 — der Fix aus Teil 5 hilft hier nur teilweise).
- **Fazit:** Completion-Rate 17 % ist real zu niedrig, aber die Lücke ist
  jetzt drei sauber getrennten Ursachen zugeordnet: (1) Horizont-Zensierung
  (größter Anteil, auch im echten Log vorhanden, kein Modellfehler), (2)
  Validierungs-Harness mit künstlich kleinem Ressourcenpool (unter Drain
  klein, unter dem harten 60-Tage-Horizont aber groß — s. Teil 7), (3) zu
  grobe Besuchs-Buckets im gemineden Branching (Rest-Ursache für die ~28 %
  Nicht-Completion selbst bei unbegrenzter Zeit). Für den Report: alle drei
  benennen, nur (3) ist ein "echter" Modell-Fix-Kandidat, und der ist als
  Re-Mining-Aufgabe zu groß für heute Nacht.

**A1-Update (18.07., Teil 7): Orgmodel-Permissions als Harness-Default — Completion-Rate 17 %→45 %, neuer Trade-off entdeckt**
- Umsetzung: `compare_process_models.py` fiel bisher, ohne `--permissions`,
  auf `DEFAULT_PERMISSIONS` zurück (17 Ressourcen, die alte Top-20-Map) statt
  auf das volle OrdinoR-Orgmodell (144 Ressourcen,
  `models/permissions_orgmodel.json`), das `simulation/main.py` selbst als
  Default nutzt. Neuer Parameter `--permissions {orgmodel,observed,hardcoded}`,
  Default jetzt `orgmodel` (mit `CaseAttributeSampler`, da das Orgmodell auf
  Case-Typ gaten kann — identisch zu main.py's Verdrahtung).
- **Ergebnis unter dem echten 60-Tage-Vergleichslauf (nicht Drain!):**

  | KPI | 17 Ressourcen (alt) | 144 Ressourcen (orgmodel, neu) |
  |---|---:|---:|
  | Completion-Rate | 17,0 % | **44,7 %** |
  | Precision | 0,682 | **0,764** |
  | Fully-fitting traces | 93,2 % | 94,0 % |
  | Branching TVD | 0,079 | 0,098 |
  | Top-20-Varianten | 17/20 | 17/20 |
  | Case-Länge rel. Fehler | 0,204 | **0,195** |
  | Case-Dauer rel. Fehler | 0,103 | **0,784** |

  Completion-Rate springt deutlich stärker als die Drain-Analyse in Teil 6
  vermuten ließ (dort nur 71,8 %→74,1 % bei 180 Tagen Puffer) — der Grund:
  unter einem harten 60-Tage-Horizont konkurriert Warteschlangen-Zeit direkt
  mit der Frist, während bei 180 Tagen Drain fast alles ohnehin genug Zeit
  hatte. Mehr Ressourcen wirken also gerade *unter realistischem
  Zeitdruck* am stärksten.
- **Neu entdeckter Trade-off:** Case-Dauer-Fehler verschlechtert sich massiv
  (0,103→0,784, sim-Dauer fällt auf ~4,7 Tage vs. real 21,8 Tage). Grund:
  der kleine 17-Ressourcen-Pool erzeugte künstliche Warteschlangen, die
  zufällig eine Case-Dauer nahe dem realen Mittel erzeugten — nicht weil das
  Modell echte Wartezeiten abbildet, sondern weil Ressourcen-Knappheit
  *zufällig* in eine ähnliche Größenordnung wie die reale Wartezeit fiel.
  Mit realistischer Ressourcenkapazität verschwindet dieser Zufalls-Effekt,
  und die eigentliche, bereits in A2 dokumentierte Lücke wird sichtbar:
  es gibt keine echte Kunden-Wartezeit-/Servicezeit-Trennung im Modell. Das
  ist also keine Regression durch den heutigen Fix, sondern das Aufdecken
  eines vorher durch einen Kompensationsfehler maskierten, größeren, bereits
  bekannten Problems (A2).
- **Kein Effekt auf den Rest der Analyse:** `end_reasons` bleibt dominiert
  von `terminal_outcome` (2162/2166), `loop_guard` feuert nur 1x in 60 Tagen
  (bei 144 Ressourcen laufen Cases schneller durch = mehr Schleifendurchläufe
  pro Zeiteinheit, aber der kurze Horizont lässt die meisten stecken
  gebliebenen Cases den Loop-Guard noch nicht erreichen) — das in Teil 6
  gefundene Schleifenproblem (A_Validating/A_Incomplete/W_Validate
  application) bleibt die wahrscheinlich relevanteste Rest-Ursache für die
  verbleibende Lücke zu ~63 % (reales Log, gleiche Fenster-Methodik).

**A1-Update (18.07., Teil 8): Top-20-Varianten-Metrik um "nie simulierte
Aktivität" bereinigt + O_Cancelled-Mehrfachangebots-Lücke (Rang 17) als
Future Work**
- Anlass: 3 der Top-20-Realvarianten werden vom Advanced-Lauf nicht
  reproduziert (`ref_top20_variants_reproduced: 17/20`, 39,0 %
  Traffic-Coverage). Manuell nachgestellt (identischer Seed/Config wie
  `advanced.json`): Rang 14 (1,26 %), 17 (1,06 %), 19 (0,73 %).
- **Rang 14 & 19** enthalten `W_Validate application` als letzten/
  vorletzten Schritt. Unter dem alten 17-Ressourcen-Harness (Teil 4/6)
  feuerte diese Aktivität in 0/822 completed cases — vollständig
  strukturell abwesend. Fix in `scripts/metrics.py::variant_overlap`: ein
  zweiter, nachsichtigerer Coverage-Wert entfernt aus jeder
  Top-20-Referenzvariante zunächst jeden Schritt, dessen Aktivität im
  gesamten Lauf kein einziges Mal feuert (identische Bedingung wie
  `activities_absent_in_run` in `branching_divergence`), und prüft den
  reduzierten Trace erneut gegen die Sim-Varianten. Neue Felder:
  `ref_top20_variants_reproduced_ignoring_absent_activities`,
  `..._traffic_coverage_pct_ignoring_absent_activities`,
  `activities_ignored_for_variant_match` — additiv, alte Felder bleiben
  unverändert (gleiche Transparenz-Konvention wie
  `excluded_target_mass`/`activities_not_in_bpmn`). Getestet: unter der
  alten 17-Ressourcen-Config reproduziert die bereinigte Zahl 19/20
  (40,99 %) statt 17/20.
- **Aber:** unter dem seit Teil 7 aktuellen Orgmodell-Default
  (144 Ressourcen) feuert `W_Validate application` nicht mehr 0×, sondern
  1× in 2166 completed cases — die Aktivität ist also nicht mehr
  strukturell abwesend, nur noch extrem selten an der richtigen Stelle im
  Trace. Die neuen Felder zeigen dafür korrekt
  `activities_ignored_for_variant_match: []` (0 gestrichen) — kein Fehler,
  sondern eine ehrliche Verschiebung der Ursache von "kann nie passieren"
  zu "passiert praktisch nie in exakt dieser Reihenfolge".
- **Rang 17** (`...O_Create Offer → O_Created → O_Create Offer → O_Created
  → O_Sent → O_Sent → ... → A_Cancelled → O_Cancelled → O_Cancelled`, zwei
  Angebote, beide storniert): Multi-Angebot-Cases reproduziert die Sim an
  sich häufig (127/822 bzw. 482/2166 completed cases mit ≥2×
  `O_Create Offer`), aber **0 von 822 bzw. 0 von 2166** completed cases
  feuern `O_Cancelled` zweimal — unabhängig vom Ressourcenmodell, also
  strukturell, nicht zufallsbedingt.
  - Ursache: `FORCED_TERMINAL_FOLLOWUP`
    (`simulation/components/petri_process.py:304-317`, Teil 5) feuert bei
    `A_Cancelled` immer genau **einen** `O_Cancelled`-Folgeevent,
    unabhängig davon, wie viele Angebote (`O_Create Offer`/`O_Sent`-Zyklen)
    der Case durchlaufen hat. Grund: das Petri-Netz simuliert BPIC-17 als
    **ein** Marking/Token pro Case (linearer Pfad), während das reale Log
    pro Angebot eine eigene, unabhängige Sub-Prozess-Instanz führt (jedes
    Angebot bekommt sein eigenes `O_Cancelled`/`O_Accepted`/`O_Refused`).
  - **Naiver Fix geprüft und verworfen:** ein einfaches "rufe
    `_fire_activity(case_id, "O_Cancelled")` N-mal in einer Schleife auf"
    funktioniert nicht — per BFS/Marking-Check verifiziert: `O_Cancelled`s
    einzige Eingangs-Stelle hat genau eine ausgehende Kante (zu
    `O_Cancelled` selbst), und das ganze Netz führt nur **ein** Token pro
    Case (linearer Pfad, kein Parallel-/Multi-Instance-Konstrukt für
    Angebote). Das Token, das den Case zweimal durch `O_Create Offer`
    zyklen ließ, ist dasselbe Token — es verdoppelt sich nie. Beim ersten
    `A_Cancelled`-Folgefire ist die Stelle leer; ein zweiter Aufruf findet
    die Transition nicht mehr aktiviert (`enabled_transitions` liefert sie
    nicht) und wäre ein stiller No-op, kein zweites Log-Event.
  - **Zwei verbleibende Optionen, beide größer als ursprünglich gedacht:**
    (1) das zusätzliche `O_Cancelled` direkt per `engine.logger.log(...)`
    ohne echte Transitions-Feuerung ins Log schreiben — verletzt aber genau
    die Konstruktions-Garantie, die das Projekt an anderer Stelle explizit
    als Stärke dokumentiert (jedes geloggte Event ist eine legal aktivierte
    Petri-Transition, s. Teil 6, `_assert_marking_legal`); nicht ohne
    ausdrückliches Go umzusetzen, weil es diese Garantie für genau diese
    Fälle stillschweigend bricht. (2) Angebote als echte
    Multi-Instance-Tokens modellieren (pro `O_Create Offer`-Feuerung ein
    eigenes Sub-Marking/Token statt eines geteilten Case-Tokens), sodass
    zwei offene Angebote unabhängig `O_Cancelled`/`O_Accepted`/`O_Refused`
    erreichen können — die einzige konformitäts-ehrliche Lösung, aber ein
    struktureller Umbau des Petri-Komponenten-State-Trackings, kein
    lokaler Patch.
  - **Bekannte Grenze selbst der echten Lösung (2):** behebt nur das
    Muster "alle Angebote gemeinsam storniert" (Rang 17s Form), sofern
    implementiert. Der eigentliche strukturelle Fall — ein Angebot wird
    angenommen, während ein anderes im selben Case storniert wird —
    bräuchte ohnehin die gleiche Multi-Instance-Token-Umstellung.
  - **Nicht umgesetzt**, da (a) Nutzen auf 1,06 % Traffic-Anteil begrenzt,
    (b) die einzige konformitäts-ehrliche Lösung ein größerer strukturel-
    ler Umbau ist, kein Patch, (c) Zeitrisiko vor der Abgabe. Für später
    vorgemerkt.

**A1-Update (18.07., Teil 8): Loop-Guard-Override + Bayesian-Shrinkage-Experiment — eines behalten, eines verworfen**
- Ausgangspunkt: Teil 6/7 zeigten, dass ~26-28 % der Cases selbst bei
  unbegrenzter Zeit nie fertig werden, weil sie im Entscheidungspunkt
  `{A_Incomplete, A_Validating, W_Validate application}` feststecken — alle
  drei legalen Optionen sind selbst Schleifen-Aktivitäten, und die gemined
  Daten zeigen an der "5+"-Visit-Bucket praktisch 0 % Chance zu enden.
- **Checkpoint-Commit `849d768`** vor Beginn dieser zwei Experimente, um
  sauber zurückrollen zu können (bündelt Teile 1-7).
- **Experiment 1 — Bayesian Shrinkage** (`scripts/mine_dp_probs.py`,
  `to_probs()` + neuer `global_end_rate()`): Dirichlet-Glättung, die
  `P(__END__)` pro Visit-Bucket Richtung eines über ALLE Decision Points
  gepoolten globalen END-Werts zieht (`--end-shrinkage-alpha`, Default jetzt
  0 = aus). Begründung: `__END__` hat als einzige Aktivitätsbezeichnung eine
  über Decision Points hinweg vergleichbare Bedeutung ("Case endet hier"),
  anders als die übrigen (decision-point-spezifischen) Aktivitätslabels —
  nur für `__END__` ist ein globaler Prior statistisch gerechtfertigt.
  - Re-Mining mit `alpha=20` (~549 s Laufzeit): realer globaler END-Rate
    **2,86 %** — viel niedriger als angenommen. Für den kritischen
    Entscheidungspunkt bedeutet das: die "5+"-Bucket bewegte sich von exakt
    0 auf nur **0,03 %** `__END__`. Rückrechnung aus der Glättungsformel
    zeigt, dass diese Bucket **~1887 reale Beobachtungen** hat (keine
    Handvoll wie ursprünglich vermutet) — die Nähe zu 0 % ist also ein
    robuster, gut belegter Befund aus den realen Daten, keine
    Kleinstichproben-Störung.
  - **Gemessen (kombiniert mit Loop-Guard-Override, gleicher 60-Tage-Lauf):**
    Completion **0,478** (schlechter als Loop-Guard allein: 0,498),
    Precision **0,694** (schlechter als 0,711), TVD **0,114** (schlechter
    als 0,109). Shrinkage mit dem realen, niedrigen globalen Prior ist eine
    Netto-Verschlechterung, nicht nur ein kleinerer Gewinn als erhofft.
  - **Verworfen:** `dp_branching_probs.json` per `git checkout` auf den
    committeten (ungeglätteten) Stand zurückgesetzt; `--end-shrinkage-alpha`
    Default auf `0.0` (aus) gesetzt. Code bleibt erhalten, getestet und
    dokumentiert als geprüfte, aber nicht übernommene Option — kein
    Blindflug für eine zukünftige Session.
- **Experiment 2 — Per-Aktivität Loop-Guard-Override** (`petri_process.py`,
  `MAX_ACTIVITY_REPEATS_OVERRIDE`): `A_Validating`/`A_Incomplete`/
  `W_Validate application` erhalten einen eigenen, engeren
  `MAX_ACTIVITY_REPEATS`-Wert statt des globalen 60 (der für die
  *Angebots*-Schleife mit echten 20+ Durchläufen gedacht ist). Sweep über
  {8, 10, 12, 20}, jeweils voller KPI-Lauf:

  | Cap | Completion | Precision | TVD | Case-Länge | Case-Dauer |
  |---|---:|---:|---:|---:|---:|
  | kein Override | 0,447 | 0,764 | 0,098 | 0,195 | 0,784 |
  | 20 | 0,464 | 0,714 | 0,114 | 0,145 | 0,736 |
  | 12 | 0,498 | 0,711 | 0,109 | 0,121 | 0,725 |
  | **10 (gewählt)** | **0,507** | 0,721 | 0,107 | **0,118** | 0,714 |
  | 8 | 0,498 | **0,746** | **0,104** | 0,133 | **0,709** |

  **Cap=10 gewählt:** beste Completion-Rate und Case-Länge (die beiden
  Zielmetriken dieser Untersuchung), bei TVD/Precision nah an Cap=8's
  besseren Werten — kein Extremwert auf einer Achse, bester
  Gesamtkompromiss. Trade-off bleibt real (Precision/TVD schlechter als
  ganz ohne Override), aber deutlich kleiner als bei den lockeren Caps
  (12/20), die überraschenderweise *nicht* weniger Trade-off boten, nur
  weniger Completion-Gewinn — vermutlich weil festgefahrene Cases ohnehin
  nie organisch fertig werden und ein engerer Cap schlicht weniger
  unrealistische Zusatz-Loop-Events vor dem Abbruch anhäuft.
- **Fazit:** von zwei parallel verfolgten Hebeln hat sich einer bestätigt
  (Loop-Guard-Override, jetzt aktiv) und einer nicht (Shrinkage, verworfen,
  aber sauber dokumentiert und reproduzierbar falls später mit einem
  anderen Prior erneut versucht werden soll). Completion-Rate bleibt mit
  ~51 % unter dem Real-Log-Vergleichswert (63,3 % im gleichen 60-Tage-
  Fenster, Teil 6) — die verbleibende Lücke ist jetzt auf ein einzelnes,
  gut verstandenes Muster eingegrenzt (dieser eine Decision-Point-Typ),
  nicht mehr auf ein diffuses "Completion ist niedrig".

**A2. Echte Ressourcen-Contention + Warte-/Servicezeit-Trennung (1.3 Adv. II)**
- Schritte:
  1. `resource.py`-Bug fixen; `process.py` so umbauen, dass eine Aktivität
     ohne Ressource **wartet** statt läuft (ACTIVITY_ENQUEUED → Start erst
     bei Zuteilung).
  2. Servicezeiten neu fitten: nur W_-Aktivitäten verbrauchen
     Ressourcenzeit; A_-/O_-Milestones sind instantan. BPIC-17 hat
     start/suspend/resume-Lifecycles → echte Servicezeit = Summe der
     aktiven Segmente, nicht start→complete.
  3. Kundenantwort-Wartezeiten als eigene Delay-Verteilung (z. B.
     O_Sent → Antwort), getrennt von Ressourcen-Wartezeit.
- Metriken: Case-Dauer rel. Fehler (Ziel ≤ 0,2 statt 0,95!), Wartezeit pro
  Aktivität vs. real (neue Metrik in `metrics.py`), Processing-Time-Fehler
  auf Servicezeiten neu berechnet, WIP-Kurve über Zeit (Face-Validity-Plot).
- Designentscheidungen: welche Aktivitäten ressourcenpflichtig; Kalibrier-
  Multiplikator für Restlücke (Rozinat: 65–100 % der beobachteten Wartezeit).

**A3. Intervall-Verfügbarkeit (1.6 Basic — Pflicht!) + Kalender (Advanced)**
- Schritte: Basic = An/Abwesenheits-Intervalle (z. B. 2-Wochen-Roster).
  Advanced = aus dem Log geminte Arbeitszeitprofile pro Ressource
  (Stunde × Wochentag-Aktivitätsmatrix), stochastische Abwesenheiten →
  liefert direkt die „uncertain availabilities" für Teil II.
- Metriken: Ressourcen-Auslastungsprofil sim vs. real (Korrelation der
  Event-Anteile pro Ressource; Events pro Stunde/Wochentag), Case-Dauer-Fehler
  (Kalender verlängern Wartezeiten Richtung real).
- Designentscheidungen: Kapazität pro Ressource = 1 (realistisch, Mensch) statt
  3; Pool ggf. über Top-20 hinaus erweitern; automatische Ressourcen (User_1)
  als Sonderfall (unbegrenzte Kapazität) behandeln.

**A4. Arrivals verfeinern (1.2 Advanced) — parallelisierbar**
- Schritte: Wochentag/Stunden-Profil aus dem Log; inhomogener Poisson-Prozess
  oder Divide-and-Conquer-Ansatz (Kirchdorfer et al.).
- Metriken: Interarrival-Fehler (Basis heute schon 18,6 % — Achtung:
  Metrik auf *gestartete*, nicht abgeschlossene Cases umstellen),
  Tagesankunfts-Verteilung (Ø 86, σ 32), Profil-Fehler pro Stunde×Wochentag.
- Bonus: `arrival_rate_error` in `metrics.py` auf alle gestarteten Cases
  umstellen (aktuell verzerrt bei niedriger Completion-Rate).

**A5. Optional (nur wenn Zeit): OrdinoR-Rollen (1.7 Adv.), Validierung
Decision Rules vs. Probs als Ablation für den Report.**

**Definition of Done Phase A:** Advanced-Modell mit Fitness ~100 %, Case-Länge
±15 %, Case-Dauer ±20 %, TVD ≤ 0,1, Variantenabdeckung dokumentiert; alle
KPIs als Tabelle im Report-Entwurf.

### Phase B — Optimierungs-Testbed

- `AllocationPolicy`-Interface: `select(work_item, candidates, state) -> resource`
  + Queue-Disziplin-Hook; `RandomPolicy` (= Teil I 1.8) als Baseline.
- Experiment-Runner: N Replikationen (≥ 10 Seeds) × Policy × Szenario,
  Warm-up-Ausschluss, feste Horizonte, Ergebnis-CSV + Plots.
- Teil-II-Metriken (`scripts/opt_metrics.py`), **Definitionen exakt nach
  Vorlesung 06, Folie 21:**
  - **Average Cycle Time** — mittlere Zeit bis zum Abschluss einer Instanz.
  - **Average Resource Occupation** — mittlerer Arbeitsanteil der Ressourcen
    **innerhalb ihrer Verfügbarkeitsfenster** (⚠️ setzt das
    Verfügbarkeits-/Kalendermodell aus A3 voraus — schon für die Metrik!).
  - **(Weighted) Resource Fairness** — mittlere Abweichung von der
    durchschnittlichen Ressourcen-Auslastung.
  - Zusätzlich: p95-Zykluszeit, Durchsatz, Wartezeit pro Aktivität,
    Queue-Längen.
- Design-Referenz fürs Policy-Interface: `bpogroup/bpo-project`
  (`bpo/planners.py`, in der Vorlesung gezeigt) — gleiche Schnittstellenidee,
  an unsere Engine angepasst.
- Designentscheidungen mit Metrik-Begründung: Warm-up-Länge (WIP-Zeitreihe /
  Welch-Verfahren), Replikationszahl (95 %-KI-Halbbreite ≤ 5 % der
  Ø-Zykluszeit), Horizontlänge.

### Phase C — Simple Policies + k-Batching (Final Task 1, Pflicht)

- **Pflicht-Heuristiken (Vorlesung 06, Folie 10, nach Russell et al.):**
  Round-Robin (R-RRA), Random (R-RMA, = unsere Teil-I-Baseline),
  Shortest-Queue-First (R-SHQ). Jeweils an unsichere Verfügbarkeiten
  angepasst (nur aktuell verfügbare Kandidaten; Re-Queue bei Ausfall —
  Designentscheidungen dokumentieren).
- Optionale Extras (billig, gutes Ablationsmaterial): Retain-Familiar
  (Case-Kontinuität), Specialist (wenigste Permissions zuerst),
  Fastest-Resource (historische Geschwindigkeit pro (Aktivität, Ressource)).
- **k-Batching (Zeng & Zhao, Folie 12):** warten bis k Tasks vorliegen, dann
  alle gemeinsam als **Parallel-Machines-Scheduling-Problem** einplanen.
  Kosten = erwartete Bearbeitungszeit **aus dem Teil-I-ML-Modell** (Synergie!).
  Designentscheidungen: k-Sweep; bekannte Schwächen aus der Vorlesung
  (unnötige Idle-Zeiten, Verfügbarkeitsprobleme) im Report diskutieren —
  das motiviert direkt Phase D.
- Evaluation: alle Policies × ≥ 10 Seeds, Mittel ± KI, gepaarte Tests
  (gleiche Seeds), Ergebnistabelle + Boxplots.

### Phase D — Zwei Advanced-Policies (Final Task 2, Pflicht: **zwei von drei**)

| Option | Idee | Referenz | Aufwand |
|---|---|---|---|
| D1 Park & Song | aktive **und prädizierte nächste Tasks** einplanen (strategic idling), Zuordnung als Assignment-Problem / Min-Cost-Max-Flow; Next-Task-Prädiktion per LSTM (Deck 05) oder vorhandenes GB-Modell — Designentscheidung begründen | Park & Song 2019 | mittel |
| D2 Kunkler & Rinderle-Ma | Verfügbarkeitsproblem via **Dummy-Ressourcen-Kosten** (Hyperparameter δ, faktorisierter Erwartungswert der Performances); Variante 1: Assignment-Problem, Variante 2: CP-Formulierung des Parallel-Machines-Scheduling mit Makespan- **und Fairness-Ziel** | Kunkler & Rinderle-Ma 2024 | mittel–hoch |
| D3 Deep RL | Gym-Wrapper um die DES-Engine (Decision-Epochs = Allokationsentscheidungen), PPO/DQN; Reward = −Zykluszeit-Inkrement (+ Fairness-Term) | Middelhuis et al. 2025 | hoch |

- **Empfehlung:** D1 + D2 — beide bauen auf demselben Assignment-Gerüst auf
  (geteilte Infrastruktur, geringeres Risiko), D2 passt exakt zu unseren
  A3-Kalendern und ist das Paper des Lehrstuhls. D3 nur zusätzlich, wenn
  jemand es als individuelles Advanced-Highlight will und Zeit ist.
- **Management-Frage (Final Task 4, Folie 23 — eine von zwei beantworten):**
  1. *„Management wants you to fire two employees — which ones?"*
     Vorgehen: Kritikalitätsanalyse pro Ressource (Auslastung, exklusive
     Permissions, historische Performance) → Kandidatenpaare →
     Leave-Two-Out-Simulationen über alle Policies → Auswahl mit minimaler
     Verschlechterung; Impact auf alle drei Metriken mit KIs berichten.
  2. *„Management wants you to reduce the working hours to nine to five!"*
     Vorgehen: Kalender aus A3 auf 9–17 Uhr beschneiden; testen, ob
     vorgeschriebene Anwesenheitsmuster (z. B. bestimmte Ressourcen täglich)
     den Schaden begrenzen; Impact auf die Metriken.
  - **Empfehlung: Frage 1** — methodisch sauber beantwortbar, geringstes
    Modellierungsrisiko, zeigt das Testbed optimal. Frage 2 als Stretch,
    falls die A3-Kalender gut sitzen (Synergie mit D2).
- **Eigene Metriken (begründen + exemplifizieren):**
  - *Time-to-First-Offer / Time-to-Decision* — kundenorientierte Domänen-KPI
    des Kreditprozesses, direkt aus A_/O_-Milestones.
  - *Handover-Rate / Familiarity* — Anteil der Case-Tasks bei bereits
    involvierten Ressourcen (Kontextwechselkosten).
  - *Rollierende Workload-Balance* — Std der Auslastung über Zeitfenster
    (deckt „fair im Schnitt, unfair im Burst" auf).
  - *Robustheit* — Zykluszeit-Degradation unter Stress (Ankünfte +30 %,
    20 % Ressourcen fallen aus): Policy-Ranking unter Unsicherheit.
- **Szenario-Experimente:** Für RQ4 jede Policy unter Normal-, Spitzen- und
  Ausfall-Regime testen → Pareto-Plot Effizienz vs. Fairness. Das
  Ausfall-Szenario ist zugleich die Infrastruktur für Management-Frage 1.

### Phase E — Report + Präsentation

- TUM-Template (sharelatex „TUM Article v2.0.0"), max. 10 Seiten, je
  Subsection verantwortliche Person nennen, AI-Tools-Subsection,
  Designentscheidungen + Challenges explizit.
- Struktur: RQ-getrieben; Teil I: Statustabelle + KPI-Vergleich vor/nach
  Phase A (das ist die geforderte „empirische Evaluation"); Teil II:
  Policy-Tabelle, Boxplots, Pareto-Front, Robustheit; Diskussion:
  Anwendbarkeit + Limitationen (Rozinat-Caveats zitieren).
- Präsentation (max. 20 min): Demo-Zahlen aus `compare_process_models.py`,
  1 Slide pro RQ, 1 Slide Lessons Learned.

---

## 3. Metrik → Designentscheidung (Spickzettel)

| Metrik | Wo | Entscheidet über |
|---|---|---|
| Control-flow fitness + precision | A1 | Prozessmodell-Enforcement korrekt? (immer beide lesen — Fitness allein ist gameable) |
| Case-Länge rel. Fehler + Completion-Rate | A1 | Branching-Mechanismus (probs / rules / historien-abhängig) |
| Branching-TVD | A1, A5 | dito, pro Decision Point |
| Top-20-Variantenabdeckung | A1 | „nah am echten Log"-Anspruch, Report-Headline |
| Case-Dauer rel. Fehler | A2, A3 | Warte-/Servicezeit-Design, Kalibrier-Multiplikator |
| Wartezeit pro Aktivität (neu) | A2 | Kundenantwort- vs. Ressourcen-Wartezeit-Aufteilung |
| Ressourcen-Auslastungsprofil sim vs. real | A3 | Kapazität, Poolgröße, Kalenderform |
| Interarrival- + Tagesprofil-Fehler | A4 | Arrival-Modellwahl |
| KI-Halbbreite über Seeds | B | Replikationszahl, Warm-up |
| Ø/p95-Zykluszeit, Auslastung, Fairness | C, D | Policy-Ranking (gepaarte Tests!) |
| Eigene Metriken + Robustheit | D | Advanced-Bewertung, RQ4 |

---

## 4. Aufgabenverteilung (4 Personen)

Teil-I-Ownership: Mario 1.2 · Daniel 1.3 · Sophie 1.4+1.5 · Johannes 1.6–1.8
(1.1 alle). Die Phase-A-Pakete fallen damit natürlich auf die bisherigen
Owner; Teil II ist so verteilt, dass jede Person Infrastruktur **und** einen
vorzeigbaren Advanced-Baustein hat.

| Person | Phase A (Teil I) | Teil II | Individuelle Advanced-Techniken (für die 1,0) |
|---|---|---|---|
| **Mario** | A4 dynamische Arrivals (1.2 Adv.) | Experiment-Runner (Phase B, Seeds/Warm-up/KIs) + **k-Batching** (Final Task 1) | 1.2 Advanced + k-Batching-Scheduling |
| **Daniel** | A2 Warte-/Servicezeit-Trennung (1.3 Adv. II) — Prozessseite | **D1 Park & Song** (Next-Task-Prädiktion — Wiederverwendung seiner 1.3-ML-Modelle!) | 1.3 Adv. I ✅ + Adv. II + D1 |
| **Sophie** | A1 Terminierungs-/Branching-Fix (1.4/1.5) | `AllocationPolicy`-Interface (Phase B, kennt die Engine-Interna) + Management-Frage 1 (Leave-Two-Out-Analyse) | 1.4 Adv. ✅ + 1.5 Adv. I ✅ + historien-abhängiges Branching |
| **Johannes** | A3 Intervall-Verfügbarkeit (1.6 **Basic-Pflicht!**) + geminte Kalender (1.6 Adv.); dazu `resource.py`-Contention-Fix (A2, Ressourcenseite, Schnittstelle mit Daniel) | Simple Policies R-RRA/R-RMA/R-SHQ (Final Task 1) + **D2 Kunkler & Rinderle-Ma** (baut direkt auf seinen Kalendern auf) | 1.6 Advanced + D2 |

Schnittstellen, die ihr früh absprechen solltet:
- **Daniel ↔ Johannes (A2):** Event-Protokoll für „Task wartet auf Ressource"
  (ENQUEUED → Zuteilung → ACTIVITY_START). Wer ändert `process.py`, wer
  `resource.py`, wie sieht der Handshake aus?
- **Sophie ↔ Mario (Phase B):** Policy-Interface vs. Runner — Signaturen früh
  festlegen, dann können D1/D2 parallel entwickelt werden.
- **Daniel ↔ Mario:** k-Batching und D1 nutzen beide das
  Bearbeitungszeit-Modell als Kostenfunktion — eine gemeinsame
  `expected_duration(task, resource)`-API bauen.

Report: jede Person schreibt die Subsections der eigenen Pakete
(Anforderung: Verantwortliche pro Subsection nennen).

## 5. Zeitplan bis zur Abgabe (So 19.07.)

| Tage | Meilenstein |
|---|---|
| Sa 11. – So 12. | Schnittstellen fixieren (ENQUEUED-Protokoll Daniel↔Johannes, Policy-Interface Sophie↔Mario). Sophie startet A1, Johannes A3 Basic (Pflicht!), ML-Evaluationsmetriken sind gefixt (temporaler Split, Precision/Recall/AUC, Pinball+Coverage — 11.07. erledigt), Mario validiert das MDN-Arrival-Modell per KPI-Vergleich. Ungetrackte Dateien committen. |
| Mo 13. – Di 14. | A2 (Contention + Service/Wait-Split, Daniel+Johannes). Sophie baut `AllocationPolicy`-Interface + `opt_metrics.py` exakt nach Folie 21. Mario: Experiment-Runner. |
| Mi 15. | **Phase-A-Gate:** `compare_process_models.py` — Case-Dauer ≤ ±20 %, Terminierung ≥ 90 %? Wenn nein: Rest-Puffer hier investieren, D-Start verschieben. Simple Policies (R-RRA/R-RMA/R-SHQ) + k-Batching lauffähig. |
| Do 16. – Fr 17. | D1 (Daniel) + D2 (Johannes) minimal lauffähig; Experimente (≥ 10 Seeds) laufen nachts. Report-Gerüst steht (TUM-Template), jede*r schreibt eigene Subsections. |
| Sa 18. | Management-Frage 1: Leave-Two-Out-Läufe + Auswertung. Plots + Ergebnistabellen final. Report-Review im Team. **Zahlen-Freeze:** ALLE Report-Tabellen (inkl. `tab:ablation`) mit der finalen Gesamt-Konfiguration neu erzeugen — der Contention-Umbau vom 16.07. verschiebt Completion-Raten und Case-Dauern (gemessen: advanced+visit 71 % → 18 % Completion, Dauer-Fehler 0,99 → 0,60; Branching-KPIs TVD/Varianten stabil). Vergleich: `output/validation/branching_probs_vs_rules/advanced_postmerge.json`. |
| So 19. | Puffer + Feinschliff + **Abgabe**. |

## 6. 1,0-Checkliste

- [ ] **Alle Basics beider Teile** erfüllt — inkl. 1.6 Intervall-Verfügbarkeit (fehlt aktuell!)
- [ ] Pro Person ≥ 1 Advanced-Technik, im Report namentlich zugeordnet
- [ ] Jede Designentscheidung mit Metrik-Begründung (dieser Spickzettel)
- [ ] Empirische Evaluation Teil I: KPI-Tabelle vorher/nachher (second-pass-Methode, Rozinat)
- [ ] Empirische Evaluation Teil II: ≥ 10 Seeds, KIs, gepaarte Tests, Settings beschrieben
- [ ] Diskussion Anwendbarkeit/Limitationen je Advanced-Ansatz
- [ ] **Zwei** Advanced-Policies implementiert (Final Task 2: zwei von P&S / K&RM / DRL)
- [ ] Management-Frage (Folie 23) beantwortet — Empfehlung: „fire two employees" via Leave-Two-Out-Simulation
- [ ] Report ≤ 10 Seiten, TUM-Template, Beiträge + AI-Nutzung deklariert
- [ ] Repo aufgeräumt: ungetrackte Dateien (`scripts/metrics.py`, `docs/`, `train_decision_rules.py`) committen

## 7. Vorlesungs-Konformität (Decks 04–06) — geprüft 14./15.07.

Abgleich unserer Implementierung gegen die konkreten Anweisungen der
Vorlesungsfolien. Status: ✅ umgesetzt · 🚫 bewusst nicht angewendet (mit
Begründung — je ein Satz im Report!) · ➡️ Hinweis an Paket-Owner.

### Deck 04 (Simulation)

| Folie | Anweisung | Status |
|---|---|---|
| F10 | BPMN → Petrinetz + Petrinetz-Algebra (pm4py) als „adequate approach" | ✅ exakt so (`petri_process.py`); Hard-coded-Graph als Basic-Vergleich |
| F9 | Block-Strukturiertheit / Designfehler vermeiden | ✅ Woflan-Check: Netz ist sound (live, bounded, keine toten Transitionen) → O_Cancelled-Schleife ist legitimes Modellverhalten, kein Designfehler; Satz steht im Report |
| F20 | Decision Points: statische Wahrscheinlichkeiten (simple) / Decision Mining (braucht Attribut-Simulation) | ✅ beides; DP-/Besuchs-Konditionierung geht methodisch darüber hinaus (Rozinat-basiert) |
| F20 | Next-Activity-Prädiktion („non-transparent") | 🚫 bewusst nicht — Folie liefert die Begründung (Intransparenz) selbst |
| F15 | Case-Attribute beim Arrival samplen | ✅ konditionale Verteilungen aus dem Log (1.5 Adv. I) |
| F31 | Allocation-Einstieg: **Pull-Prinzip, longest-active-first** | ➡️ Johannes: als Basis-Disziplin kennen/erwähnen (Assignment-Text sagt nur „random") |
| F28 | Kalender; Rozinat vernachlässigt **Saisonalität** | ➡️ Johannes: Saisonalitäts-Satz in seinen Kalender-Abschnitt |
| F16 | Differentiated resources (Verteilungen pro Aktivität×Ressource) | ➡️ Daniel: mindestens als Related Work (sein Modell nutzt Ressource als Feature) |
| F35 | DES: Event-Queue, Simulationszeit, Loop | ✅ Engine entspricht exakt der Folie — im 1.1-Report-Abschnitt referenzieren |

### Deck 05 (ML / Splits / Metriken)

| Folie | Anweisung | Status |
|---|---|---|
| F37 | **Temporaler Split** (train alt, test neu — Drift!) | ✅ beide Trainingsskripte; Befund: log-R² 0,36 → 0,12 (Random-Split hatte Drift geleakt) → Daniels Report-Story |
| F37 | 70/15/15 mit Validation-Set | 🚫 80/20 ohne Val-Slice — Hyperparameter fest, nicht getunt (steht im Report); wer tunt, braucht ein Val-Slice |
| F44 | Kategorial: Precision, Recall, ROC-AUC | ✅ pro Decision Point (macro, OvR) — deckte Majority-Class-Kollaps auf (acc 0,89, prec 0,22) |
| F44 | Numerisch: MAE, MSE | ✅ wörtlich im Zeitmodell-Training |
| F44 | Sequenz: Damerau-Levenshtein | 🚫 keine Suffix-Prädiktion bei uns |
| F47 | „How to evaluate probability densities?" | ✅ Pinball-Loss + 90 %/50 %-Intervall-Coverage (0,815 / 0,424) |
| F36 | Cleaning (Ausreißer/None) | ✅ Dauern-Filter 0<d≤365d, Sentinels, Leakage-Attribute ausgeschlossen |
| F39 | Standard-Scaling / Embeddings | 🚫 baumbasierte Modelle sind skaleninvariant — Index-Encoding genügt |
| F38–F43 | Prefix-Encoding, ELoader, LSTM-Encodings | ➡️ Bauanleitung für **D1 (Park & Song)** in Teil II — an den D1-Owner weitergeben |

### Deck 06 (Optimierung)

| Folie | Anweisung | Status |
|---|---|---|
| F21 | Die drei Evaluationsmetriken (exakte Definitionen) | ✅ 1:1 in `scripts/opt_metrics.py` (inkl. Verfügbarkeits-Nenner + Completed-Filter) |
| F22/23 | Final Tasks + Management-Fragen | ✅ vollständig in Phase C/D dieser Roadmap abgebildet |
| F18 vs. F21 | Fairness-*Ziel* (Abweichung vom Längstarbeitenden, CP) ≠ Fairness-*Metrik* (Abweichung vom Durchschnitt) | ✅ Unterscheidung in `opt_metrics.py` dokumentiert — nicht verwechseln! |

## 8. Offene Fragen (an Betreuer / im Team klären)

1. Zählen die bereits umgesetzten Teil-I-Advanced-Bausteine (1.3 Adv. I,
   1.4 Adv., 1.5 Adv. I) individuell für die Personen, die sie gebaut haben?
3. Verteilung im Team bestätigen (Abschnitt 4) — insbesondere, dass Johannes
   mit A3 + D2 + Simple Policies nicht überladen ist (Simple Policies sind
   klein; notfalls zu Mario schieben).

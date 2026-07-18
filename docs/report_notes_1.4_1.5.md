# Report-Notizen: 1.4 Prozessmodell & 1.5 Branching Decisions (Sophie)

Alle Zahlen (sofern nicht anders markiert): Seed 42, **60-Tage-Horizont**,
KPIs gegen `simulation_inputs.json` (echtes BPIC-17-Log), Methodik =
Second-Pass-Validierung nach Rozinat et al. (siehe
`docs/paper_insights_discovering_simulation_models.md`). Jede Zahl ist aus
einer JSON-Datei in `output/` reproduzierbar (Datei-Index unten).

> **Update 18.07. (Abend) — vor der Abgabe unbedingt lesen:** Seit der
> ursprünglichen Fassung dieses Dokuments (Stand ~15.-16.07.) haben sich
> vier Dinge geändert, die praktisch alle Zahlen unten in Abschnitt 3
> verschieben. Die *Methodik/Kausalketten* (D1-D5, Abschnitt 4 Challenges)
> bleiben inhaltlich gültig; nur die **absoluten Zahlen der alten
> Ablationstabelle in Abschnitt 3 sind überholt** — für den Report bitte
> die aktuelle Zahlen aus der Tabelle direkt unten verwenden, nicht die
> alte Tabelle. Volle Herleitung: `docs/ROADMAP.md`, Einträge
> "A1-Update (18.07., Teil 1-8)".
> 1. **Horizont 30 → 60 Tage:** kein Rundwert mehr, sondern das p99 der
>    realen Case-Dauer (5 110 843,8 s ≈ 59,15 Tage, real
>    `data/BPIChallenge2017.xes.gz`), gerundet auf 60. Ergebnis: Case-Dauer-
>    Fehler des Advanced-Modells 0,60 → 0,10 (großer Teil des alten Fehlers
>    war Horizont-Zensierung, kein Modellfehler — bestätigt durch die
>    Drain-Analyse unten, die *unter dem alten 30-Tage-Setup* gemessen
>    wurde und daher nicht mehr direkt vergleichbar ist).
> 2. **`O_Cancelled`/`O_Refused` als erzwungener Folgeschritt:**
>    `A_Cancelled`/`A_Denied` feuern jetzt deterministisch die reale
>    Bestätigungsaktivität (`O_Cancelled`/`O_Refused`, p≈0,996/0,997 im
>    Bigram) bevor der Case endet, statt direkt abzubrechen. Fully-fitting
>    traces: 35,4 % → **93,2 %**; Top-20-Varianten: 10/20 → **17/20**.
> 3. **Orgmodel-Permissions statt Hardcoded-17-Ressourcen-Map als
>    Vergleichs-Harness-Default:** `compare_process_models.py` nutzte
>    unbemerkt nur 17 statt der vorgesehenen 144 Ressourcen (OrdinoR-
>    Orgmodell, §1.7). Completion-Rate 17 % → 45 % allein dadurch — zeigt
>    aber auch einen neuen, ehrlich zu benennenden Trade-off: Case-Dauer-
>    Fehler verschlechtert sich 0,10 → 0,78, weil der kleine Ressourcenpool
>    zuvor *zufällig* realistische Wartezeiten simulierte (Ressourcen-
>    Knappheit statt echter Kunden-Wartezeit — eine bereits bekannte A2-
>    Lücke, jetzt sichtbar statt maskiert).
> 4. **Per-Aktivität Loop-Guard-Override:** `A_Validating`/`A_Incomplete`/
>    `W_Validate application` waren strukturell (fast) geschlossene
>    Schleifen ohne reale Exit-Wahrscheinlichkeit — Cases zirkulierten bis
>    zu 266 Mal (real: nie mehr als 8). Engerer, datenbelegter Cap (10)
>    nur für diese drei behoben; Completion 45 % → **51 %**.
>
> **Aktuelle Basic/Advanced/Real-Zahlen (18.07. Abend, s. auch
> `output/validation/process_model_comparison/{basic,advanced,real_log}.json`):**
>
> | KPI | Basic | Advanced | Real Log |
> |---|---:|---:|---:|
> | Fully-fitting traces | 0,00 % | 93,81 % | 68,84 % |
> | Ø Trace-Fitness | 0,240 | 0,996 | 0,976 |
> | Precision | 0,678 | 0,721 | 0,519 |
> | Branching TVD | 0,422 | 0,107 | ~0 (Referenz stammt aus diesem Log) |
> | Top-20-Varianten | 0/20 | 17/20 | 20/20 (trivial) |
> | Case-Länge rel. Fehler | 0,068 | 0,118 | ~0 |
> | Case-Dauer rel. Fehler | 0,985 | 0,714 | ~0 |
> | Completion-Rate (60-Tage-Fenster) | 0,990 | 0,507 | **0,633** (gleiche Fenster-Methodik; ~100 % ohne Fensterbegrenzung) |
>
> **Wichtig für den Report:** Real-Log-Fitness gegen das eigene Signavio-
> Netz ist nur 68,8 %, nicht 100 % — das relativiert, was "Advanced 93,8 %"
> bedeutet (das Modell selbst erreicht, an echten Daten gemessen, keine
> 100 %). Precision des realen Logs ist mit 0,519 sogar *niedriger* als
> Basic/Advanced — kein Widerspruch, sondern Folge der viel größeren
> Verhaltensvielfalt von 31 509 realen Cases gegenüber wenigen Tausend
> simulierten.

---

## Update 18.07. (Nachtrag) — Replay-Fix & Netz-Abschluss

Zwei Änderungen nach dem Abend-Update. Die Tabelle in diesem Abschnitt ist
**die aktuelle** — sie ersetzt die Tabelle im Abend-Update oben.

### N1 — Selektionsbias im DP-Mining behoben (Repair-Replay)

`mine_dp_probs.py` brach die Replay eines Cases beim **ersten** Schritt ab,
den das Netz nicht erlaubte (`act not in frontier` → `break`). Die Folge war
subtiler als "nur 57,7 % der Daten":

- Entscheidungen **vor** dem Abbruch wurden für *jeden* Case gezählt — frühe
  Decision Points hatten also nahezu die volle Population.
- Alles **danach** ging verloren. Je tiefer ein Decision Point bzw. je höher
  der Besuchs-Bucket, desto stärker war die Stichprobe auf "modellkonforme"
  Cases verengt.
- Am härtesten traf es `__END__`: Die Terminierung wurde nur für Cases
  gezählt, deren **ganze** Trace exakt passte (`if fits:`). Ein Case, der
  drei Decision Points sauber durchlief und erst danach abwich, lieferte
  **kein** Ende-Signal.

**Fix:** Bei einer Abweichung wird nicht mehr abgebrochen, sondern die
passende Transition wird zwangsweise gefeuert, indem fehlende Tokens in
unterversorgte Input-Stellen eingefügt werden (Missing-Token-Konvention des
Token-Based Replay). Der abweichende Schritt selbst wird **nie** als
Decision-Point-Wahl gezählt (er war dort nie legal) — aber alle
Entscheidungen davor *und danach* fließen ein.

> **Terminologie für den Report:** Das ist **Token-Replay-Repair**, *nicht*
> Alignment-based Replay. Repariert wird lokal und greedy an der
> Abweichungsstelle, ohne Lookahead/Backtracking; ein echtes Alignment
> (pm4py `alignments`, A*-Suche) würde global die kostenminimale Kombination
> aus Synchron-/Log-/Model-Moves suchen und könnte auch frühere Schritte
> anders auflösen. Bitte nicht als "alignment-based" bezeichnen.

**Wirkung auf die geminete Tabelle** (volles Log, 31 509 Cases):

| | vorher | nachher |
|---|---:|---:|
| Decision Points mit ≥30 Samples | 21 | **220** |
| Cases, die über ihre erste Abweichung hinaus beitragen | 0 | **13 313 (42,3 %)** |
| Perfect-Fit-Rate (unverändert, nur jetzt separat ausgewiesen) | 57,7 % | 57,7 % |

P(`__END__`) am strukturell fast geschlossenen Decision Point
`A_Incomplete | A_Validating | W_Validate application`:

| Bucket | vorher | nachher |
|---|---:|---:|
| 1 | 3,87 % | 4,63 % |
| 2 | 0,02 % | 2,61 % |
| 3 | **0 %** (keine Beobachtung) | **17,09 %** |
| 4 | 0,22 % | 11,06 % |
| 5+ | **0 %** (keine Beobachtung) | **11,64 %** |
| gepoolt (`all`) | 1,73 % | 7,85 % |

Die Nullen waren **fehlende Daten, keine Evidenz für Unmöglichkeit** — genau
der Fall, über den `docs/ROADMAP.md` (A1-Update Teil 8) beim Verwerfen der
END-Shrinkage nachgedacht hat. Das Argument dort bleibt gültig (der Punkt
hatte ~1887 Beobachtungen), aber die *Buckets 3 und 5+* waren sehr wohl
datenarm, weil ihre Cases vorher weggeworfen wurden.

**Nebenbefund — echte Modell-Lücke, jetzt sichtbar:** 2 730 Events
referenzieren 4 Aktivitäten, für die das Signavio-BPMN **gar keine**
Transition hat: `O_Sent (online only)`, `W_Assess potential fraud`,
`W_Call after offers`, `W_Shortened completion`. Diese werden übersprungen,
**ohne** die Markierung zu bewegen (ein *Log-Move* im Alignment-Sinn) und als
`n_unrepairable_events` separat gezählt. Vorher war das von einer normalen
Branching-Abweichung nicht unterscheidbar. Gehört als Limitation zu D2.

### N2 — Cases schließen das Petrinetz jetzt formal ab

Die Laufstatistik meldete `final_marking: 0` — **kein einziger** simulierter
Case endete im Endmarking des Netzes. Ursache war nicht die
Kontrollfluss-Erzwingung, sondern deren letzter Schritt: Wenn ein Case
beschloss zu enden (gemineter `__END__`-Entscheid oder Terminal-Outcome-
Regel), endete er sofort — **ohne die stillen Transitionen zu feuern, die das
Netz zum Abschluss anbietet**. `_final_reachable_by_tau` hatte bereits
festgestellt, dass das Endmarking von dort erreichbar ist; nur gefeuert hat
den Pfad niemand. Die Cases endeten einen Tau-Schritt vor dem Sink.

**Fix:** Beim Beenden wird die Markierung über die netzeigene
Abschlussstruktur ins Endmarking geführt (`_complete_via_tau`, BFS nur über
unsichtbare Transitionen). Das ist die Abschluss-Hälfte derselben
Petrinetz-Algebra, die überall sonst den Kontrollfluss erzwingt: keine
sichtbare Transition feuert, nichts erreicht das Log.

**Ergebnis: 1 777 von 2 453 abgeschlossenen Cases (72,4 %) schließen das Netz
jetzt formal ab — bei bit-identischen KPIs** (Fitness, Precision, TVD,
Completion, Case-Länge alle unverändert). Genau das ist zu erwarten: Der Case
endet an derselben Stelle, das Netz wird nur sauber geschlossen statt
abgebrochen.

Der Netz-Abschluss wird als **eigener Zähler** (`closed_at_final_marking`)
geführt, nicht in `end_reasons` gefaltet: *Warum* ein Case endete (Auslöser)
und *ob das Netz geschlossen wurde* (Zustand) sind zwei verschiedene Fragen.
Eine erste Fassung überschrieb den End-Grund und ließ dabei 1 601
`terminal_outcome`-Fälle aus der Statistik verschwinden — der bestehende Test
`test_terminal_outcome_fires_forced_followup_before_ending` hat das gefangen.

**Warum nicht 100 %?** Nicht wegen des Simulators: Replay der *echten* Traces
zeigt, dass **0 % der realen BPIC-17-Cases** exakt im Endmarking dieses Netzes
enden, und **46,5 % an einer Stelle stoppen, von der aus es überhaupt nicht
erreichbar ist** (sichtbare Transitionen bleiben aktiviert). Das ist eine
Eigenschaft des Signavio-Modells, nicht der Erzwingung.

> **Für die Verteidigung von 1.4 (Anforderung "Petri net algebra is used to
> enforce the control-flow"):** Alle drei geforderten Teile sind im Code
> belegbar — `.bpmn` laden (`pm4py.read_bpmn`), ins Petrinetz konvertieren
> (`pm4py.convert_to_petri_net`), Kontrollfluss per Netz-Algebra erzwingen
> (`semantics.enabled_transitions`/`execute` + Tau-Closure; die Menge legaler
> Folgeaktivitäten ist ausschließlich das in der Markierung Aktivierte, das
> Branching-Modell wählt nur darunter). "Enforce the control-flow" verlangt
> **nicht**, dass jeder Case im Sink landet — und das reale Log tut es bei
> diesem Netz zu 0 %. Mit N2 lässt sich zusätzlich sagen: 72,4 % der Cases
> schließen das Netz auch formal ab.

### Aktuelle KPI-Tabelle (60 Tage, Seed 42, orgmodel, legacy lifecycle)

| KPI | Basic | Adv. `probs` | Adv. `visit` | Adv. `rules` | Real Log |
|---|---:|---:|---:|---:|---:|
| Fully-fitting traces | 0,00 % | 95,47 % | 95,47 % | **99,18 %** | 68,84 % |
| Ø Trace-Fitness | 0,240 | 0,997 | 0,997 | 0,999 | 0,976 |
| Precision | 0,678 | **0,748** | **0,748** | 0,577 | 0,520 |
| Branching TVD | 0,422 | **0,091** | **0,091** | 0,126 | 0,006 |
| Top-20-Varianten | 0/20 | 17/20 | 17/20 | 17/20 | 20/20 |
| Ø Case-Länge (real 15,1) | 16,11 | 12,77 | 12,77 | 13,22 | 15,08 |
| Case-Länge rel. Fehler | 0,068 | 0,154 | 0,154 | **0,124** | ~0 |
| Case-Dauer rel. Fehler | 0,985 | **0,693** | **0,693** | 0,736 | ~0 |
| Completion-Rate | 0,990 | 0,507 | 0,507 | 0,504 | 0,633 |
| **Netz formal abgeschlossen** | — | **1777/2453 (72,4 %)** | **1777/2453 (72,4 %)** | **1794/2440 (73,5 %)** | — |

Veränderung ggü. der Abend-Tabelle (Advanced): Fully-fitting 93,81 → 95,47 %,
Precision 0,721 → 0,748, TVD 0,107 → 0,091, Case-Dauer-Fehler 0,714 → 0,693.
Schlechter: Case-Länge 0,118 → 0,154. Completion praktisch unverändert
(0,507 → 0,5066).

**`rules`-Spalte mit den auf reparierter Replay neu trainierten
Decision-Trees** (davor: siehe N4 unten — das war noch der alte Bias-Stand).
Ggü. den alten `rules`-Zahlen (Fitness 99,55 %, Precision 0,536, TVD 0,125,
Completion 0,545): Precision jetzt 0,577 (leicht besser), Completion 0,504
(leicht schlechter, war zuvor der beste Wert unter den drei Modi — jetzt ist
es `probs`/`visit`). Kein Modus dominiert mehr in jeder Kennzahl; die
Unterschiede zwischen den drei Ergebniszeilen sind klein genug, dass sie
eher als "im Rahmen des Rauschens einer einzigen geseedeten Simulation"
denn als robuste Rangfolge zu lesen sind.

### N4 — `rules`-Trainingsdaten hatten denselben Bias, jetzt behoben

`train_decision_rules.py` hatte exakt dasselbe Muster wie `mine_dp_probs.py`
vor N1: Replay brach beim ersten abweichenden Schritt ab
(`fits = False; break`), Decision Points nach dem Abbruch bekamen keine
Trainingsbeispiele. Fix: dieselbe Repair-Logik, jetzt in einem gemeinsamen
Modul `simulation/petri_replay.py` statt dupliziert (der 1-sicher-Clamp ist
ein subtiles Invariant — zwei Kopien, die auseinanderlaufen, hätten den
Tau-Closure-Blowup aus N1 zurückgebracht).

**Wirkung auf das Retraining** (volles Log, 31 509 Cases):

| | vorher | nachher |
|---|---:|---:|
| Trainierte Decision Points | 16 | **69** |
| Entscheidungsinstanzen gesamt | (nicht erfasst) | 362 026 |
| Ø Test-Accuracy | 0,690 | 0,560 |
| Ø Majority-Baseline | 0,643 | 0,547 |
| Ø Macro-Precision | 0,375 | 0,259 |
| Ø Macro-Recall | 0,403 | 0,319 |
| Ø ROC-AUC (OvR, macro) | 0,635 | 0,577 |

**Das sieht schlechter aus, ist aber kein Rückschritt.** 53 zusätzliche
Decision Points erreichen jetzt die Mindeststichprobe (30) und werden
trainiert — diese sind naturgemäß schwerer vorherzusagen, weil sie vorher
zu datenarm waren, um überhaupt aufzutauchen. Auf den 16 Decision Points,
die in beiden Ständen trainiert wurden, bleibt die Accuracy stabil
(−0,066 bis +0,029 Veränderung, kein systematischer Abfall). Für den
Report: **Die neue Zahl ist die ehrlichere** — sie misst Vorhersagequalität
über die tatsächliche Populationsverteilung der Decision Points, nicht nur
über die modellkonforme Teilmenge.

> **Korrektur für den Report-Absatz "of 16 trained decision points, one is
> perfectly separable (AUC 1.0)...":** Bezog sich auf den alten, biased
> Trainingsstand — mit 69 Decision Points ist "of 16" nicht mehr korrekt.
> Geprüft: Der AUC-1,0-Punkt selbst bleibt bestehen und ist weiterhin
> derselbe (`A_Complete | O_Create Offer` — hat das Angebot begonnen,
> RequestedAmount > 0), nur die Grundgesamtheit ist jetzt 69 statt 16.
> Vorschlag: "of 69 trained decision points, one remains perfectly
> separable (AUC 1.0 — has an offer been created yet)". Der Satz zur
> kollabierenden Mehrheitsklasse (Accuracy 0,89 vs. Baseline 0,96,
> Macro-Precision 0,22) bezieht sich auf einen einzelnen Decision Point und
> müsste gegen `output/models/decision_rules_metrics.json` neu verifiziert
> werden, falls der exakte Zahlenwert im Report zitiert wird.

### N3 — Wichtig: `probs` und `visit` sind nicht mehr unterscheidbar

In der Tabelle oben sind die Spalten `probs` und `visit` **identisch** — das
ist kein Copy-Paste-Fehler, sondern eine direkte Folge von N1, und es muss im
Report benannt werden.

Die geminete DP-Tabelle wird **in jedem Modus** geladen (bewusst: der
`__END__`-Entscheid gehört zur Netz-Terminierung, nicht zu einer
Branching-Heuristik), und in `_weighted_choice` hat sie **Vorrang** vor der
aktivitätskonditionierten Tabelle. Letztere ist der *einzige* Unterschied
zwischen `probs` und `visit`. Solange die DP-Tabelle nur 21 Decision Points
abdeckte, fiel die Simulation oft auf den modusspezifischen Pfad zurück und
die Modi divergierten (alte Ablation: TVD 0,24 vs. 0,11). Mit 220 abgedeckten
Decision Points greift die DP-Tabelle praktisch immer — die Modi laufen auf
dieselbe Trajektorie.

**Konsequenz für den Report:** Die Ablation "probs vs. visit" diskriminiert
nicht mehr und sollte nicht mehr als Vergleich präsentiert werden. Die
Besuchskonditionierung ist deshalb *nicht* verschwunden — sie steckt jetzt in
der DP-Tabelle selbst (Buckets 1…4, 5+ pro Decision Point), also auf der
feineren, decision-point-genauen Ebene statt auf Aktivitätsebene. Die
inhaltliche Aussage von D3 (Replay-Mining schlägt Bigramme) bleibt gültig;
die Aussage "visit schlägt probs" ist gegenstandslos geworden.

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
  0,11).
  > **Update 18.07.:** der hier vorgeschlagene Hebel („feinere Besuchs-
  > Buckets bis 8+") wurde geprüft und verworfen — Bayesian Shrinkage von
  > `P(__END__)` Richtung eines gepoolten globalen Werts (real nur 2,86 %)
  > zeigte sich als Netto-Verschlechterung, weil der kritische
  > Entscheidungspunkt (`A_Incomplete`/`A_Validating`/`W_Validate
  > application`, alle drei Optionen selbst Schleifen-Aktivitäten) bereits
  > ~1887 reale Beobachtungen mit praktisch 0 % Enden hat — kein
  > Kleinstichproben-Rauschen, sondern ein robuster, aber strukturell
  > geschlossener Fall. Stattdessen umgesetzt: ein enger, datenbelegter
  > Loop-Guard-Cap (10 Wiederholungen, real max. 7-8) nur für diese drei
  > Aktivitäten. Details, Sweep-Tabelle und Verwerfungsbegründung:
  > `docs/ROADMAP.md`, A1-Update Teil 8.

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
- ~~DP-Tabellen aus den 57,7 % exakt passenden Cases gemined (Selektionsbias
  möglich).~~ **Behoben am 18.07., s. N1 oben:** Repair-Replay lässt jetzt
  100 % der Cases beitragen (13 313 davon über ihre erste Abweichung hinaus).
  Restliche Limitation: Der abweichende Schritt selbst wird nicht gezählt,
  und 2 730 Events betreffen 4 Aktivitäten ohne Transition im BPMN (Log-Moves,
  s. N1) — die DP-Tabelle sagt also nichts über deren Kontext aus.
- ~~`train_decision_rules.py` hat denselben Bias ungefixt~~ **Behoben am
  18.07., s. N4:** dieselbe Repair-Replay (jetzt gemeinsam mit N1 in
  `simulation/petri_replay.py`), Decision-Trees neu trainiert (16 → 69
  Decision Points).
- Der `rules`-Modus erreicht weiterhin die beste Case-Länge (0,124 vs. 0,154
  bei `probs`/`visit`), aber die mit Abstand schlechteste Precision (0,577
  vs. 0,748) und leicht schlechtere TVD (0,126 vs. 0,091) — bei gleicher
  Variantenabdeckung. Completion ist nach dem Retraining kein
  Alleinstellungsmerkmal von `rules` mehr (0,504 vs. 0,507). Für den Default
  spricht das weiterhin gegen `rules`.
- **Korrektur 18.07.:** die vorherige Zeile hier war überholt/falsch.
  1.5 Advanced II (History-konditioniertes Prediction-Model pro Decision
  Point, Trainingsdaten per Log-Replay auf dem Prozessmodell identifiziert
  — exakt der Assignment-Wortlaut) ist **umgesetzt und aktiv**: das ist
  genau `mine_dp_probs.py`/`dp_branching_probs.json`
  (`--branching-mode visit`, aktueller Default in `simulation/main.py` UND
  `compare_process_models.py`). Advanced I (Decision-Tree-Klassifikator auf
  Spawn-/Offer-Attributen, `--branching-mode rules`) ist parallel dazu
  ebenfalls voll implementiert und getestet, aber nicht Default (Ablation
  zeigt: `visit` gewinnt bei TVD/Precision). Beide Stufen sind also
  vorhanden und im Code auswählbar — kein offener Punkt, sondern bereits
  erfüllt; im Report als "beide Advanced-Stufen implementiert, visit als
  gewählter Default" darstellen, nicht als Lücke.

## 6. Datei-Index (alle Zahlen reproduzierbar)

| Datei | Inhalt |
|---|---|
| `output/validation/process_model_comparison/{basic,advanced,real_log}.json` | D1 + Update-18.07.-Tabelle (real_log neu: `scripts/eval_real_log.py`) |
| `output/validation/bpmn_source_comparison/real_log_replay_im02.json`, `advanced_im02.json` | D2 |
| `output/validation/branching_probs_vs_rules/advanced_{probs,probs_terminal,visit_bigram,visit,rules}.json` | D3/D4/D5-Ablation (5 Stufen, **vor** Update 18.07. — Richtung gültig, absolute Zahlen überholt) |
| `output/validation/process_model_comparison/ablation/` | 18.07.: BPMN-Gateway/Terminal-Outcomes/Branching-Mode isoliert (A1-Update Teil 1-3) |
| `output/validation/horizon_censoring/drain.json` | Zensierungsnachweis — **Achtung:** unter altem 30-Tage-Setup gemessen, s. Update-Hinweis oben |
| `output/models/decision_rules_metrics.json` | D5 (**69** DP-Modelle seit N4, temporaler Split) |
| `simulation/models/dp_branching_probs.json` | gemint: P(Label bzw. END \| DP, Besuch); **220 statt 21 Decision Points seit dem Repair-Replay (N1)**; Diagnosefelder `replay_perfect_fit_pct`, `n_cases_repaired`, `n_repair_events`, `n_unrepairable_events`; Shrinkage geprüft & verworfen, s. Update oben |
| `output/validation/branching_probs_vs_rules/advanced_{probs,rules}.json` | aktuelle Modusvergleiche (18.07. Nachtrag). **Achtung:** `advanced_{probs_terminal,visit_bigram,visit}.json` im selben Ordner stammen aus dem alten 30-Tage-Setup und sind überholt |
| `simulation_inputs.json` → `branching_probs_by_visit` | Besuchs-Buckets 1/2/3+ |
| `docs/ROADMAP.md`, "A1-Update (18.07., Teil 1-8)" | volle Herleitung aller Änderungen seit dieser Notiz |
| Repro-Kommandos | `scripts/compare_process_models.py --configs advanced --branching-mode {probs,visit,rules} --permissions orgmodel`; `scripts/mine_dp_probs.py --log data/BPIChallenge2017.xes.gz`; `scripts/eval_real_log.py`; `scripts/discover_process_model.py --log …` |

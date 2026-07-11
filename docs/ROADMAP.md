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

**A1. Terminierungsproblem im Advanced-Modell (Blocker)**
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
| Sa 18. | Management-Frage 1: Leave-Two-Out-Läufe + Auswertung. Plots + Ergebnistabellen final. Report-Review im Team. |
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

## 7. Offene Fragen (an Betreuer / im Team klären)

1. Zählen die bereits umgesetzten Teil-I-Advanced-Bausteine (1.3 Adv. I,
   1.4 Adv., 1.5 Adv. I) individuell für die Personen, die sie gebaut haben?
3. Verteilung im Team bestätigen (Abschnitt 4) — insbesondere, dass Johannes
   mit A3 + D2 + Simple Policies nicht überladen ist (Simple Policies sind
   klein; notfalls zu Mario schieben).

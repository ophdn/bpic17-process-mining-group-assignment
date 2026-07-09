"""
arrival_mdn.py — Case Arrival Component (Section 1.2 Advanced)
==============================================================
Dynamisches, zeitabhängiges Ankunftsmodell als Drop-in-Ersatz für die
parametrische ``ArrivalComponent``. Statt einer einzigen festen LogNormal-
Verteilung nutzt es ein **Mixture Density Network** (Log-Normal-Mischung),
das — bedingt auf die aktuelle Tageszeit/Wochentag/Saison — die Verteilung
der nächsten Inter-Arrival-Time ausgibt (intensitätsfreier Temporal Point
Process, vgl. Shchur et al., ICLR 2020).

Motiv: Der reale BPIC-17 Ankunftsprozess ist *lokal* Poisson, aber die Rate
schwankt stark (nachts ~0.6/h, Kern 12–18h ~7.6/h; Mo ≈ 3× So; Sommer +35 %).
Eine statische Verteilung kann diese Struktur prinzipiell nicht abbilden.

Die Gewichte werden offline mit ``train_arrival_mdn.py`` (PyTorch) erzeugt und
hier als reiner NumPy-Forward-Pass ausgewertet — ZUR LAUFZEIT KEIN PyTorch.

Die parametrische Variante in ``arrival.py`` bleibt unverändert bestehen;
diese Klasse hat dasselbe Interface (bootstrap / on_arrival / HANDLES) und ist
1:1 austauschbar in ``main.py``.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.events import SimEvent, EventType

# Standard-Pfad der exportierten Gewichte (neben dieser Datei)
_DEFAULT_WEIGHTS = Path(__file__).with_name("arrival_mdn_weights.npz")


def _time_features(dt: datetime) -> np.ndarray:
    """Zyklische Kalender-Features -> (1, 7). MUSS identisch zum Training sein."""
    hod = dt.hour + dt.minute / 60.0
    dow = dt.weekday()          # 0 = Montag
    doy = dt.timetuple().tm_yday
    return np.array([[
        np.sin(2*np.pi*hod/24),  np.cos(2*np.pi*hod/24),
        np.sin(2*np.pi*dow/7),   np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
        1.0 if dow >= 5 else 0.0,
    ]], dtype=np.float64)


class MDNArrivalComponent:
    """
    Generiert Fälle mit zeitabhängigen Inter-Arrival-Times aus einem MDN.

    Parameters
    ----------
    seed : int, optional
        Seed für Reproduzierbarkeit.
    start_datetime : datetime
        Reale Verankerung von Sim-Zeit t=0. Muss zur Engine-/main.py-Config
        passen, damit Wochentag/Tageszeit korrekt aligned sind
        (BPIC-17: 2016-01-01, ein Freitag).
    scale_factor : float
        Multiplikator auf die Ankunftsrate (>1 = schneller). 1.0 = realer Takt.
    weights_path : str | Path, optional
        Pfad zur .npz-Gewichtsdatei (Default: neben diesem Modul).
    """

    HANDLES = {EventType.CASE_ARRIVAL: None}  # unten gepatcht

    def __init__(
        self,
        seed: Optional[int] = 42,
        start_datetime: datetime = datetime(2016, 1, 1),
        scale_factor: float = 1.0,
        weights_path=None,
    ):
        self._rng = np.random.default_rng(seed)
        self._start = start_datetime
        self._scale_factor = scale_factor
        self._case_counter = 0
        self._load_weights(weights_path or _DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_weights(self, path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"MDN-Gewichte nicht gefunden: {path}\n"
                f"Erst offline trainieren: python train_arrival_mdn.py "
                f"--arrivals <arrivals.parquet> --out {path}"
            )
        w = np.load(path, allow_pickle=True)
        self._W1, self._b1 = w["W1"], w["b1"]
        self._W2, self._b2 = w["W2"], w["b2"]
        self._Wpi, self._bpi = w["Wpi"], w["bpi"]
        self._Wmu, self._bmu = w["Wmu"], w["bmu"]
        self._Wls, self._bls = w["Wls"], w["bls"]
        self._ls_lo, self._ls_hi = float(w["ls_lo"]), float(w["ls_hi"])
        self._clip_min, self._clip_max = float(w["clip_min"]), float(w["clip_max"])
        self._iat_unit_s = float(w["iat_unit_seconds"])

    # ------------------------------------------------------------------
    # Public (Interface identisch zu ArrivalComponent)
    # ------------------------------------------------------------------

    def bootstrap(self, engine) -> None:
        """Schedule die allererste Ankunft. Vor engine.run() aufrufen."""
        self._schedule_next(engine, current_time=0.0)

    def on_arrival(self, engine, event: SimEvent) -> None:
        """CASE_ARRIVAL: Fall an Prozess übergeben, nächste Ankunft planen."""
        engine.schedule(SimEvent(
            timestamp=engine.now,
            priority=5,
            event_type=EventType.ACTIVITY_ENABLED,
            case_id=event.case_id,
            activity="__PROCESS_START__",
            payload=event.payload or {},
        ))
        self._schedule_next(engine, current_time=engine.now)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray):
        """NumPy-Forward des MDN -> (pi, mu, log_sigma) für die Mischung."""
        z = np.tanh(x @ self._W1 + self._b1)
        z = np.tanh(z @ self._W2 + self._b2)
        logits = z @ self._Wpi + self._bpi
        logits -= logits.max(axis=1, keepdims=True)
        pi = np.exp(logits); pi /= pi.sum(axis=1, keepdims=True)
        mu = z @ self._Wmu + self._bmu
        ls = np.clip(z @ self._Wls + self._bls, self._ls_lo, self._ls_hi)
        return pi[0], mu[0], ls[0]

    def _sample_inter_arrival(self, current_time: float) -> float:
        """Inter-Arrival-Time (Sekunden) bedingt auf den aktuellen Sim-Zeitpunkt."""
        now_dt = self._start + timedelta(seconds=current_time)
        pi, mu, ls = self._forward(_time_features(now_dt))
        k = self._rng.choice(len(pi), p=pi)                  # Mischkomponente
        y = mu[k] + np.exp(ls[k]) * self._rng.standard_normal()
        iat_units = float(np.clip(np.exp(y), self._clip_min, self._clip_max))
        seconds = iat_units * self._iat_unit_s
        return seconds / self._scale_factor

    def _schedule_next(self, engine, current_time: float) -> None:
        self._case_counter += 1
        inter_arrival = self._sample_inter_arrival(current_time)
        engine.schedule(SimEvent(
            timestamp=current_time + inter_arrival,
            priority=10,
            event_type=EventType.CASE_ARRIVAL,
            case_id=f"case_{self._case_counter:06d}",
            payload={},
        ))


MDNArrivalComponent.HANDLES = {EventType.CASE_ARRIVAL: MDNArrivalComponent.on_arrival}

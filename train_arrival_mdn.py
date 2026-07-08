"""
train_arrival_mdn.py — Offline-Training des MDN Case-Arrival-Modells
====================================================================
Trainiert ein Mixture Density Network (Log-Normal-Mischung) auf die
Inter-Arrival-Times der BPIC-17 Fälle und exportiert die Gewichte als
NumPy-.npz. Die Simulationskomponente (arrival_mdn.py) lädt diese Gewichte
und braucht zur Laufzeit KEIN PyTorch — nur NumPy.

Hintergrund: Der Ankunftsprozess ist lokal Poisson, aber die Rate variiert
stark nach Tageszeit/Wochentag/Saison. Das MDN gibt — bedingt auf zyklische
Zeit-Features — die Verteilung der nächsten Inter-Arrival-Time aus
(intensitätsfreier Temporal Point Process, vgl. Shchur et al. ICLR 2020).

Dies ist ein EINMALIGER Offline-Schritt und benötigt PyTorch:
    uv add torch          # oder: pip install torch
    python train_arrival_mdn.py \
        --arrivals /pfad/zu/arrivals.parquet \
        --out simulation/components/arrival_mdn_weights.npz

Eingabe: Parquet/CSV mit einer Spalte von Ankunfts-Zeitstempeln (erstes Event
je Fall). Default-Spaltenname: 'arrival'.
"""
import argparse
import numpy as np
import pandas as pd

# ---- Feature-Spezifikation (MUSS identisch zur Laufzeit in arrival_mdn.py sein) ----
TZ = "Europe/Amsterdam"      # Geschäftszeiten-Lokalzeit
K = 6                        # Mischkomponenten
H = 96                       # Hidden-Größe
LS_LO, LS_HI = -3.0, 1.5     # Clamp log-sigma (Stabilität, keine Runaways)
CLIP_MIN, CLIP_MAX = 0.05, 2880.0   # IAT-Clip in Minuten (0.05 min .. 2 Tage)
IAT_UNIT_SECONDS = 60.0      # Modell rechnet in Minuten -> *60 für Sim-Sekunden


def time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """Zyklische Kalender-Features -> (n, 7). Identisch in Training & Laufzeit."""
    hod = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2*np.pi*hod/24),  np.cos(2*np.pi*hod/24),
        np.sin(2*np.pi*dow/7),   np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
        (dow >= 5).astype(float),
    ]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrivals", required=True, help="Parquet/CSV mit Ankunfts-Zeitstempeln")
    ap.add_argument("--col", default="arrival", help="Spaltenname der Zeitstempel")
    ap.add_argument("--out", default="simulation/components/arrival_mdn_weights.npz")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- Daten laden ----
    if args.arrivals.endswith(".parquet"):
        df = pd.read_parquet(args.arrivals)
    else:
        df = pd.read_csv(args.arrivals)
    ts = pd.to_datetime(df[args.col], utc=True).dt.tz_convert(TZ).sort_values().reset_index(drop=True)

    iat_min = ts.diff().dt.total_seconds().to_numpy() / 60.0       # IAT in Minuten
    ctx = pd.DatetimeIndex(ts.shift(1))                            # Kontext = vorherige Ankunft
    valid = (iat_min > 0) & (~ctx.isna())
    X = torch.tensor(time_features(pd.DatetimeIndex(ctx[valid])))
    y = torch.tensor(np.log(iat_min[valid]).astype(np.float32)).unsqueeze(1)
    print(f"Trainings-IATs: {len(y)}  (TZ={TZ})")

    # ---- MDN ----
    class MDN(nn.Module):
        def __init__(s):
            super().__init__()
            s.body = nn.Sequential(nn.Linear(7, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
            s.pi = nn.Linear(H, K); s.mu = nn.Linear(H, K); s.ls = nn.Linear(H, K)
        def forward(s, x):
            z = s.body(x)
            return torch.log_softmax(s.pi(z), 1), s.mu(z), s.ls(z).clamp(LS_LO, LS_HI)

    net = MDN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        opt.zero_grad()
        logpi, mu, ls = net(X)
        comp = -0.5*((y-mu)**2/torch.exp(2*ls)) - ls - 0.5*np.log(2*np.pi)
        loss = -torch.logsumexp(logpi + comp, 1).mean()
        loss.backward(); opt.step(); sched.step()
        if ep % 400 == 0:
            print(f"  epoch {ep:4d}  NLL {loss.item():.4f}")
    print(f"  final NLL {loss.item():.4f}")

    # ---- Export als NumPy-Arrays (reiner Forward-Pass ohne torch zur Laufzeit) ----
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    np.savez(
        args.out,
        # body: Linear-Schichten (torch speichert weight als (out,in) -> transponiert ablegen)
        W1=sd["body.0.weight"].T, b1=sd["body.0.bias"],
        W2=sd["body.2.weight"].T, b2=sd["body.2.bias"],
        Wpi=sd["pi.weight"].T, bpi=sd["pi.bias"],
        Wmu=sd["mu.weight"].T, bmu=sd["mu.bias"],
        Wls=sd["ls.weight"].T, bls=sd["ls.bias"],
        K=K, H=H, ls_lo=LS_LO, ls_hi=LS_HI,
        clip_min=CLIP_MIN, clip_max=CLIP_MAX, iat_unit_seconds=IAT_UNIT_SECONDS,
        tz=TZ, n_train=len(y), final_nll=loss.item(),
    )
    print(f"Gewichte exportiert -> {args.out}")


if __name__ == "__main__":
    main()

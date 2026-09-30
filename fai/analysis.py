"""Tracking datasetinden türetilen istatistikler. Hepsi sadece DataFrame ister;
videoya ya da modele ihtiyaç yok (datasetin 'sonra başka iş için' kullanılma noktası)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Track bazında: süre, koşu mesafesi, maks/ortalama hız, sprint sayısı."""
    p = df[(df["track_id"] >= 0) & df["role"].isin(["player", "goalkeeper"]) & df["on_pitch"]]
    p = p.sort_values(["track_id", "frame"])
    dt = p.groupby("track_id")["t"].diff()
    step = (p["speed_mps"] * dt).where(p["speed_mps"].notna())
    sprint = p["speed_mps"] > 7.0                               # ~25 km/s eşiği (yaygın sprint tanımı)
    sprint_start = sprint & ~sprint.groupby(p["track_id"]).shift(fill_value=False)
    g = p.assign(step=step, sprint_start=sprint_start).groupby("track_id")
    out = pd.DataFrame({
        "team": g["team"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else -1),
        "role": g["role"].first(),
        "seconds_tracked": g["t"].agg(lambda s: s.max() - s.min()),
        "frames": g.size(),
        "distance_m": g["step"].sum(),
        "max_speed_kmh": g["speed_mps"].max() * 3.6,
        "avg_speed_kmh": g["speed_mps"].mean() * 3.6,
        "sprints": g["sprint_start"].sum().astype(int),
    })
    return out.sort_values("distance_m", ascending=False)


def possession_proxy(df: pd.DataFrame, radius_m: float = 1.5, hold_s: float = 2.0, fps: float = 25.0) -> pd.DataFrame:
    """Frame başına 'topu kontrol eden' takım tahmini: topa radius_m içindeki en yakın oyuncu.
    Kimse yakın değilse son sahip hold_s saniye boyunca korunur (pas sırasında sahiplik düşmesin).
    Gerçek possession değil, homography tabanlı bir yaklaşım: top havadayken yanlıştır."""
    ball = df[(df["cls"] == "ball") & ~df["outlier"]][["frame", "x", "y"]].dropna().drop_duplicates("frame")
    ppl = df[df["role"].isin(["player", "goalkeeper"]) & (df["team"] >= 0)][["frame", "team", "track_id", "xs", "ys"]].dropna()
    m = ppl.merge(ball, on="frame", suffixes=("", "_b"))
    m["d"] = np.hypot(m["xs"] - m["x"], m["ys"] - m["y"])
    near = m.loc[m.groupby("frame")["d"].idxmin()]
    near = near[near["d"] <= radius_m].set_index("frame")[["team", "track_id", "d"]]
    frames = pd.Index(sorted(df["frame"].unique()), name="frame")
    out = near.reindex(frames)
    out["team_held"] = out["team"].ffill(limit=int(hold_s * fps))
    return out


def team_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Frame x takım: merkez (centroid), genişlik/derinlik (IQR yerine %10-%90 aralığı; kaleci hariç)."""
    p = df[(df["role"] == "player") & (df["team"] >= 0) & df["on_pitch"]].dropna(subset=["xs", "ys"])
    g = p.groupby(["frame", "team"])
    q = lambda s, a: s.quantile(a)
    return pd.DataFrame({
        "n": g.size(),
        "cx": g["xs"].mean(), "cy": g["ys"].mean(),
        "depth_m": g["xs"].agg(lambda s: q(s, .9) - q(s, .1)),
        "width_m": g["ys"].agg(lambda s: q(s, .9) - q(s, .1)),
    }).reset_index()

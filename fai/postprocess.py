"""Ham tracking satırlarını analiz edilebilir datasete çevirir.

Adımlar:
  1. Takım oylaması: her frame'deki sınıflandırma titrer; track_id bazında çoğunluk
     oyu ile tek `team` ve tek `role` (player/goalkeeper) atanır.
  2. Saha içi bayrağı (`on_pitch`): yedek kulübesi, top toplayıcı vb. elenebilsin.
  3. Pozisyon yumuşatma (`xs`, `ys`) ve hız (`speed_mps`): homography gürültüsü
     ham farklardan hız hesaplarsa 60-80 km/s gibi saçma değerler verir.
  4. Top temizliği: fiziksel olarak imkansız sıçramalar (`outlier`) atılır, kısa
     boşluklar doğrusal interpolasyonla doldurulur (`interp`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pitch import PitchConfig


@dataclass
class PostConfig:
    pitch_margin_m: float = 3.0
    smooth_window_s: float = 0.4       # hareketli ortalama penceresi
    track_gap_s: float = 0.5           # bu kadar boşluk track'i segmentlere böler
    max_player_speed: float = 12.0     # m/s (~43 km/s) üstü ölçüm hatası sayılır
    max_ball_speed: float = 40.0       # m/s (~144 km/s)
    ball_max_gap_s: float = 1.0        # en fazla bu kadar boşluk interpolasyonla doldurulur


def _mode(s: pd.Series):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else np.nan


def postprocess(df: pd.DataFrame, frames: pd.DataFrame, fps: float,
                pitch: PitchConfig = PitchConfig(), cfg: PostConfig = PostConfig()) -> pd.DataFrame:
    df = df.copy()
    people_cls = df["cls"].isin(["player", "goalkeeper"])

    # 1) takım + rol oylaması -------------------------------------------------
    tracked = people_cls & (df["track_id"] >= 0)
    votes = df[tracked & (df["team_raw"] >= 0)].groupby("track_id")["team_raw"].agg(_mode)
    roles = df[tracked].groupby("track_id")["cls"].agg(_mode)
    df["team"] = df["team_raw"]
    df.loc[tracked, "team"] = df.loc[tracked, "track_id"].map(votes).fillna(-1).astype(int)
    df["role"] = df["cls"]
    df.loc[tracked, "role"] = df.loc[tracked, "track_id"].map(roles)
    df.loc[~people_cls, "team"] = -1
    df["team"] = df["team"].astype(int)

    # 2) saha içi ----------------------------------------------------------------
    m = cfg.pitch_margin_m
    df["on_pitch"] = (df["x"].between(-m, pitch.length + m) & df["y"].between(-m, pitch.width + m))

    # 3) yumuşatma + hız (sadece takip edilen insanlar) ----------------------------
    df["xs"], df["ys"], df["speed_mps"] = df["x"], df["y"], np.nan
    gap = max(1, int(round(cfg.track_gap_s * fps)))
    win = max(1, int(round(cfg.smooth_window_s * fps)))
    ppl = df[df["track_id"] >= 0].sort_values(["track_id", "frame"])
    if len(ppl):
        seg = (ppl.groupby("track_id")["frame"].diff().fillna(1) > gap).groupby(ppl["track_id"]).cumsum()
        keys = [ppl["track_id"], seg]

        def smooth(col: str) -> pd.Series:
            med = ppl.groupby(keys)[col].transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
            return med.groupby(keys).transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())

        xs, ys = smooth("x"), smooth("y")
        dx, dy = xs.groupby(keys).diff(), ys.groupby(keys).diff()
        dt = ppl.groupby(keys)["t"].diff()
        spd = np.hypot(dx, dy) / dt
        spd[(spd > cfg.max_player_speed) | ~np.isfinite(spd)] = np.nan
        df.loc[ppl.index, "xs"], df.loc[ppl.index, "ys"], df.loc[ppl.index, "speed_mps"] = xs, ys, spd

    # 4) top ---------------------------------------------------------------------
    df["outlier"], df["interp"] = False, False
    ball = df[df["cls"] == "ball"].sort_values("frame")
    if len(ball):
        keep, last = [], None
        for idx, r in ball.iterrows():
            ok = True
            if last is not None and np.isfinite(r.x) and np.isfinite(last.x):
                dt = r.t - last.t
                if dt > 0 and np.hypot(r.x - last.x, r.y - last.y) / dt > cfg.max_ball_speed:
                    ok = False
            keep.append(ok)
            if ok and np.isfinite(r.x):
                last = r
        df.loc[ball.index[~np.array(keep)], "outlier"] = True

        good = df[(df["cls"] == "ball") & ~df["outlier"]].set_index("frame")
        all_frames = pd.Index(frames["frame"].sort_values().unique(), name="frame")
        lim = max(1, int(round(cfg.ball_max_gap_s * fps)))
        cols = ["u", "v", "x", "y", "x1", "y1", "x2", "y2"]
        full = good[cols].reindex(all_frames).interpolate(limit=lim, limit_area="inside")
        new = full[~full.index.isin(good.index) & full["u"].notna()]
        if len(new):
            t_map = frames.set_index("frame")["t"]
            add = new.reset_index()
            add["t"] = add["frame"].map(t_map)
            add = add.assign(track_id=-1, cls="ball", conf=np.nan, team_raw=-1, team=-1, role="ball",
                             h_src="interp", xs=add["x"], ys=add["y"], speed_mps=np.nan,
                             outlier=False, interp=True)
            add["on_pitch"] = (add["x"].between(-m, pitch.length + m) & add["y"].between(-m, pitch.width + m))
            df = pd.concat([df, add[df.columns]], ignore_index=True)
        b = df["cls"] == "ball"
        df.loc[b, "xs"], df.loc[b, "ys"] = df.loc[b, "x"], df.loc[b, "y"]

    return df.sort_values(["frame", "cls", "track_id"]).reset_index(drop=True)

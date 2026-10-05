"""Ham tracking satırlarını analiz edilebilir datasete çevirir.

Adımlar:
  0. İlk frame geri doldurma: ByteTrack yeni track'in ilk frame'ine -1 verir; o tespit, track'in ilk
     satırıyla aynı yerdeyse (IoU) track_id'si geriye yazılır.
  1. Rol + takım oylaması: frame bazlı sınıf ve takım titrer; track_id bazında çoğunluk oyu ile tek `role`
     (player/goalkeeper/referee, hakem dahil) ve tek `team` atanır. Hakemin takımı her zaman -1.
  2. Saha içi bayrağı (`on_pitch`): yedek kulübesi, top toplayıcı vb. elenebilsin.
  3. Pozisyon yumuşatma (`xs`, `ys`) ve hız (`speed_mps`): homography gürültüsü ham farklardan hız
     hesaplarsa 60-80 km/s gibi saçma değerler verir. Homography olmayan frame'e değer uydurulmaz.
  4. Top temizliği: fiziksel olarak tutarlı en uzun tespit zinciri tutulur, gerisi `outlier`
     (saha koordinatı yoksa piksel uzayında); sadece kısa boşluklar doğrusal interpolasyonla doldurulur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    max_ball_speed_wps: float = 1.5    # saha koordinatı yokken: görüntü genişliği / s (kamera hareketi dahil; kalibre edilmedi)
    ball_max_gap_s: float = 1.0        # sadece bundan kısa boşluklar interpolasyonla doldurulur
    backfill_iou: float = 0.5          # ilk frame geri doldurma için min IoU
    backfill_frames: int = 2           # bir track en fazla kaç frame geriye uzatılsın


def _mode(s: pd.Series):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else np.nan


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def backfill_first_frames(df: pd.DataFrame, min_iou: float, max_frames: int) -> pd.DataFrame:
    box = ["x1", "y1", "x2", "y2"]
    for _ in range(max_frames):
        tracked = df[(df["track_id"] >= 0) & (df["cls"] != "ball")]
        if tracked.empty:
            break
        first = tracked.loc[tracked.groupby("track_id")["frame"].idxmin(), ["track_id", "frame", *box]]
        first = first.assign(frame=first["frame"] - 1)
        cand = df[(df["track_id"] < 0) & (df["cls"] != "ball")].reset_index().merge(first, on="frame", suffixes=("", "_t"))
        if cand.empty:
            break
        cand["iou"] = _iou(cand[box].to_numpy(float), cand[[f"{c}_t" for c in box]].to_numpy(float))
        cand = cand[cand["iou"] >= min_iou].sort_values("iou", ascending=False)
        cand = cand.drop_duplicates("index").drop_duplicates("track_id_t")   # her satır / track en fazla bir kez
        if cand.empty:
            break
        df.loc[cand["index"].to_numpy(), "track_id"] = cand["track_id_t"].to_numpy()
    return df


def consistent_chain(t: np.ndarray, xy: np.ndarray, vmax: float, horizon: float) -> np.ndarray:
    """Ardışık seçilenler arası hız <= vmax olacak şekilde en uzun alt dizi (bool maske). Zaman farkı `horizon`'u
    aşan çiftler her zaman tutarlıdır (o sürede top her yere gidebilir). İlk tespit sahte olsa bile ona kilitlenmez."""
    n = len(t)
    dp, prev = np.ones(n, int), np.full(n, -1)
    old_best, old_idx, j0 = 0, -1, 0
    for i in range(n):
        while t[j0] < t[i] - horizon:                  # pencereden çıkanlar her şeyle tutarlı
            if dp[j0] > old_best:
                old_best, old_idx = dp[j0], j0
            j0 += 1
        if old_idx >= 0:
            dp[i], prev[i] = old_best + 1, old_idx
        if j0 < i:
            dt = t[i] - t[j0:i]
            ok = (dt > 0) & (np.hypot(*(xy[i] - xy[j0:i]).T) <= vmax * dt)
            if ok.any():
                cand = np.where(ok, dp[j0:i], 0)
                k = int(np.argmax(cand[::-1]))
                k = i - 1 - k                          # eşitlikte en yakın (en son) öncül
                if dp[k] + 1 > dp[i]:
                    dp[i], prev[i] = dp[k] + 1, k
    keep = np.zeros(n, bool)
    i = n - 1 - int(np.argmax(dp[::-1]))              # eşitlikte en geç biten zincir
    while i >= 0:
        keep[i] = True
        i = prev[i]
    return keep


def _fill_short_gaps(full: pd.DataFrame, lim: int) -> pd.DataFrame:
    """Sadece uzunluğu <= lim olan iç boşlukları doldur (pandas `limit` uzun boşluğun ilk `lim` frame'ini de doldurur)."""
    miss = full["u"].isna()
    run = miss.groupby((~miss).cumsum()).transform("sum")
    filled = full.interpolate(limit_area="inside")
    ok = miss & (run <= lim)
    out = full.copy()
    out.loc[ok] = filled.loc[ok]
    return out


def postprocess(df: pd.DataFrame, frames: pd.DataFrame, fps: float,
                pitch: PitchConfig = PitchConfig(), cfg: PostConfig = PostConfig(),
                img_w: Optional[int] = None) -> pd.DataFrame:
    df = df.copy()
    people_cls = df["cls"] != "ball"

    # 0) ilk frame geri doldurma ----------------------------------------------------
    df = backfill_first_frames(df, cfg.backfill_iou, cfg.backfill_frames)

    # 1) rol + takım oylaması (hakem dahil) -------------------------------------------
    tracked = people_cls & (df["track_id"] >= 0)
    roles = df[tracked].groupby("track_id")["cls"].agg(_mode)
    df["role"] = df["cls"]
    df.loc[tracked, "role"] = df.loc[tracked, "track_id"].map(roles)
    votes = df[tracked & (df["team_raw"] >= 0)].groupby("track_id")["team_raw"].agg(_mode)
    df["team"] = df["team_raw"]
    df.loc[tracked, "team"] = df.loc[tracked, "track_id"].map(votes).fillna(-1).astype(int)
    df.loc[df["role"].isin(["referee", "ball"]), "team"] = -1
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
            out = med.groupby(keys).transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
            return out.where(ppl[col].notna())            # homography olmayan frame'e değer uydurma

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
        t = ball["t"].to_numpy(float)
        has_xy = ball["x"].notna().to_numpy() & ball["y"].notna().to_numpy()
        keep = np.ones(len(ball), bool)
        if has_xy.any():                                      # saha uzayı
            diag_m = float(np.hypot(pitch.length, pitch.width)) + 2 * m
            keep[has_xy] = consistent_chain(t[has_xy], ball.loc[has_xy, ["x", "y"]].to_numpy(float),
                                            cfg.max_ball_speed, diag_m / cfg.max_ball_speed)
        if (~has_xy).any() and img_w:                         # homography yok: piksel uzayı
            vmax_px = cfg.max_ball_speed_wps * img_w
            keep[~has_xy] = consistent_chain(t[~has_xy], ball.loc[~has_xy, ["u", "v"]].to_numpy(float),
                                             vmax_px, 1.5 * img_w / vmax_px)
        df.loc[ball.index[~keep], "outlier"] = True

        good = df[(df["cls"] == "ball") & ~df["outlier"]].set_index("frame")
        all_frames = pd.Index(frames["frame"].sort_values().unique(), name="frame")
        lim = max(1, int(round(cfg.ball_max_gap_s * fps)))
        cols = ["u", "v", "x", "y", "x1", "y1", "x2", "y2"]
        full = _fill_short_gaps(good[cols].reindex(all_frames), lim)
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

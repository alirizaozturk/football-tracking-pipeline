"""Video -> tracking dataset klasörü.

Çıktı klasörü (maç başına):
    tracking.parquet   obje x frame satırları (ham + işlenmiş kolonlar)
    frames.parquet     frame başına homography ve kalite metrikleri
    meta.json          video bilgisi, saha ölçüleri, kolon açıklamaları
    team_classifier.pkl  aynı maçı tekrar işlerken yeniden fit etmemek için
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from .dataio import save_table
from .models import CLASSES
from .pipeline import PipelineConfig, TrackingPipeline, collect_player_crops
from .pitch import PitchConfig
from .postprocess import PostConfig, postprocess
from .teams import TeamClassifier

COLUMNS = {
    "frame": "video frame indeksi", "t": "saniye (frame / fps)",
    "track_id": "ByteTrack ID (-1: takip edilmeyen / top)",
    "cls": "detector sınıfı: player | goalkeeper | referee | ball",
    "role": "track bazında çoğunluk sınıfı", "conf": "detection güveni",
    "team_raw": "frame bazında takım tahmini (0/1, -1: yok)", "team": "track bazında çoğunluk oyu ile takım",
    "x1,y1,x2,y2": "bbox (piksel)", "u,v": "zemin temas noktası (piksel, bbox alt-orta)",
    "x,y": "saha koordinatı (m); x: sol kale çizgisinden, y: uzak taç çizgisinden",
    "xs,ys": "yumuşatılmış saha koordinatı (m)", "speed_mps": "yumuşatılmış hız (m/s)",
    "on_pitch": "saha sınırları (+marj) içinde mi", "h_src": "homography kaynağı: fresh | carried | none | interp",
    "outlier": "top: fiziksel olarak imkansız sıçrama", "interp": "top: interpolasyonla doldurulmuş satır",
}


def build_match_dataset(video_path: str | Path, out_dir: str | Path,
                        detector: Callable, keypointer: Optional[Callable],
                        team_method: str = "siglip", ball_detector: Optional[Callable] = None,
                        pitch: PitchConfig = PitchConfig(), cfg: PipelineConfig = PipelineConfig(),
                        post_cfg: PostConfig = PostConfig(), fit_every_s: float = 1.0,
                        max_fit_crops: int = 1500, start: int = 0, end: Optional[int] = None,
                        fmt: str = "parquet", device: Optional[str] = None,
                        tracker: Optional[Callable] = None, extra_meta: Optional[Dict] = None) -> Path:
    video_path, out = Path(video_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = sv.VideoInfo.from_video_path(str(video_path))
    fps = float(info.fps)
    end = min(end, info.total_frames) if end else info.total_frames

    # 1) takım sınıflandırıcı: maçın kendi oyuncu kırpıntılarıyla fit ----------------
    clf_path = out / "team_classifier.pkl"
    if clf_path.exists():
        team_clf = TeamClassifier.load(clf_path, device)
        print(f"[team] mevcut sınıflandırıcı yüklendi: {clf_path}")
    else:
        stride = max(1, int(round(fps * fit_every_s)))
        frames = sv.get_video_frames_generator(str(video_path), stride=stride, start=start, end=end)
        indexed = ((start + k * stride, f) for k, f in enumerate(frames))
        crops = collect_player_crops(tqdm(indexed, desc="team crops", total=(end - start) // stride),
                                     detector, max_crops=max_fit_crops)
        print(f"[team] {len(crops)} oyuncu kırpıntısı ile '{team_method}' fit ediliyor")
        team_clf = TeamClassifier(team_method, device=device).fit(crops)
        team_clf.save(clf_path)

    # 2) ana döngü ----------------------------------------------------------------------
    pipe = TrackingPipeline(detector, keypointer, team_clf, pitch, fps, ball_detector, cfg, tracker)
    rows, frows = [], []
    t0 = time.time()
    gen = sv.get_video_frames_generator(str(video_path), start=start, end=end)
    for i, frame in enumerate(tqdm(gen, desc="tracking", total=end - start)):
        r, fr = pipe.process(start + i, frame)
        rows += r
        frows.append(fr)
    elapsed = time.time() - t0

    # 3) postprocess + kaydet -------------------------------------------------------------
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("hiç detection yok: model yolu / conf eşiği / video doğru mu?")
    frames_df = pd.DataFrame(frows)
    df = postprocess(raw, frames_df, fps, pitch, post_cfg)
    p_track = save_table(df, out / "tracking", fmt)
    p_frames = save_table(frames_df, out / "frames", fmt)

    h_ok = (frames_df["h_src"] != "none").mean()
    meta = {
        "video": str(video_path.resolve()), "fps": fps, "width": info.width, "height": info.height,
        "start_frame": start, "end_frame": end, "frames_processed": len(frames_df),
        "processing_fps": round(len(frames_df) / max(elapsed, 1e-6), 2),
        "pitch": asdict(pitch), "classes": list(CLASSES), "team_method": team_method,
        "homography_coverage": round(float(h_ok), 4),
        "homography_rejected": dict(pipe.htrack.rejected),
        "line_score_median": round(float(frames_df["line_score"].median()), 3)
        if frames_df["line_score"].notna().any() else None,
        "files": {"tracking": p_track.name, "frames": p_frames.name},
        "columns": COLUMNS, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    n_tracks = df.loc[df["track_id"] >= 0, "track_id"].nunique()
    print(f"[ok] {len(df)} satır, {len(frames_df)} frame, {n_tracks} track, "
          f"homography kapsama %{100 * h_ok:.1f} -> {out}")
    return out

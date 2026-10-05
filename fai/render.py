"""Tracking datasetinden video render (dataset -> görsel). Modeli tekrar çalıştırmaz."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from .dataio import load_match
from .pitch import PitchConfig
from .viz import Radar, annotate_broadcast, overlay_inset


def radar_objects(rows) -> dict:
    ok = rows["on_pitch"] & rows[["xs", "ys"]].notna().all(axis=1)
    r = rows[ok]
    ppl = r[r["role"].isin(["player", "goalkeeper"])]
    return {
        "team0": ppl.loc[ppl["team"] == 0, ["xs", "ys"]].to_numpy(),
        "team1": ppl.loc[ppl["team"] == 1, ["xs", "ys"]].to_numpy(),
        "referee": r.loc[r["role"] == "referee", ["xs", "ys"]].to_numpy(),
        "ball": r.loc[(r["cls"] == "ball") & ~r["outlier"], ["xs", "ys"]].to_numpy(),
    }


def render_match(match_dir: str | Path, out_path: str | Path, mode: str = "radar",
                 radar_only: bool = False, max_frames: Optional[int] = None,
                 video_path: Optional[str] = None) -> Path:
    """mode: 'radar' | 'voronoi' | 'none' (sadece yayın anotasyonu)."""
    df, frames, meta = load_match(match_dir)
    pitch = PitchConfig(**meta["pitch"])
    radar = Radar(pitch, scale=10 if radar_only else 8)
    by_frame = dict(tuple(df.groupby("frame")))
    empty = df.iloc[0:0]
    start, end = meta["start_frame"], meta["end_frame"]
    if max_frames:
        end = min(end, start + max_frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if radar_only:
        info = sv.VideoInfo(width=radar.w, height=radar.h, fps=meta["fps"])
        with sv.VideoSink(str(out_path), info) as sink:
            for f in tqdm(range(start, end), desc="render radar"):
                sink.write_frame(radar.draw_frame(radar_objects(by_frame.get(f, empty)), voronoi=(mode == "voronoi")))
        return out_path

    src = video_path or meta["video"]
    info = sv.VideoInfo.from_video_path(src)
    gen = sv.get_video_frames_generator(src, start=start, end=end)
    with sv.VideoSink(str(out_path), info) as sink:
        for f, frame in tqdm(zip(range(start, end), gen), total=end - start, desc="render"):
            rows = by_frame.get(f, empty)
            img = annotate_broadcast(frame, rows)
            if mode in ("radar", "voronoi"):
                img = overlay_inset(img, radar.draw_frame(radar_objects(rows), voronoi=(mode == "voronoi")))
            sink.write_frame(img)
    return out_path

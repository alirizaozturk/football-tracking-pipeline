"""Faz A'da eklenen davranışların testleri (negatif testlerin kapsamadığı yeni yollar).

Çalıştır:  python -m pytest tests/test_faz_a.py -v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fai.cli import add_dataclass_args, dataclass_from_args  # noqa: E402
from fai.models import PLAYER  # noqa: E402
from fai.pipeline import PipelineConfig, TrackingPipeline  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.postprocess import PostConfig, postprocess  # noqa: E402
from fai.teams import TeamClassifier  # noqa: E402

FPS = 25.0
W, H = 1280, 720


def frames_df(n):
    return pd.DataFrame(dict(frame=np.arange(n), t=np.arange(n) / FPS))


def test_top_homography_yokken_piksel_uzayinda_filtrelenir():
    """x=NaN (homography yok): fiziksel kontrol piksel uzayında yapılmalı; 900 px'lik tek-frame sıçramalar outlier."""
    n = 100
    u = np.arange(n) * 3.0 + 200
    u[[40, 90]] += 900
    df = pd.DataFrame(dict(frame=np.arange(n), t=np.arange(n) / FPS, track_id=-1, cls="ball", conf=0.5, team_raw=-1,
                           x1=u, y1=300.0, x2=u + 8, y2=308.0, u=u, v=308.0, x=np.nan, y=np.nan, h_src="none"))
    out = postprocess(df, frames_df(n), FPS, img_w=W)
    flagged = sorted(out.loc[out["outlier"], "frame"])
    assert flagged == [40, 90], flagged


def test_ilk_frame_geri_doldurulur():
    """ByteTrack yeni track'in ilk frame'ine -1 verir; aynı kutu bir sonraki frame'de track 4 ise geri yazılmalı."""
    rows = []
    for f in range(10):
        tid = -1 if f == 0 else 4
        rows.append(dict(frame=f, t=f / FPS, track_id=tid, cls="player", conf=0.9, team_raw=1, x1=100.0 + f, y1=200.0,
                         x2=140.0 + f, y2=300.0, u=120.0 + f, v=300.0, x=30.0, y=20.0, h_src="fresh"))
    rows.append(dict(frame=0, t=0.0, track_id=-1, cls="player", conf=0.9, team_raw=0, x1=900.0, y1=200.0, x2=940.0,
                     y2=300.0, u=920.0, v=300.0, x=60.0, y=20.0, h_src="fresh"))           # alakasız -1, dokunulmamalı
    out = postprocess(pd.DataFrame(rows), frames_df(10), FPS)
    f0 = out[out.frame == 0].sort_values("x1")
    assert f0["track_id"].tolist() == [4, -1], f0[["x1", "track_id"]]


def _team_clf():
    crops = [np.full((40, 20, 3), c, np.uint8) for c in ([30, 30, 200], [200, 120, 20]) * 10]
    return TeamClassifier("color").fit(crops)


def test_sahne_kesmesinde_tracker_ve_tasinan_h_sifirlanir_idler_cakismaz():
    from tests.test_negative import camera, keypoints, lines_frame
    Hm = camera(50.0)
    wide = lines_frame(Hm)
    close = np.full((H, W, 3), (40, 40, 200), np.uint8)               # kırmızı forma yakın çekim: histogram çok farklı
    kp, conf = keypoints(Hm)
    box = np.array([[600.0, 300, 640, 400]])

    def detector(fr, idx):
        return sv.Detections(xyxy=box.copy(), confidence=np.array([0.9]), class_id=np.array([PLAYER]))

    def keypointer(fr, idx):
        return (kp, conf) if idx < 10 else (None, None)

    pipe = TrackingPipeline(detector, keypointer, _team_clf(), PitchConfig(), FPS)
    rows, frows = [], []
    for i in range(20):
        r, fr = pipe.process(i, wide if i < 10 else close)
        rows += r
        frows.append(fr)
    frows = pd.DataFrame(frows)
    assert frows.loc[10, "cut"] and not frows.loc[:9, "cut"].any() and not frows.loc[11:, "cut"].any(), frows["cut_dist"].round(3).tolist()
    assert frows.loc[10, "h_src"] == "none"                         # taşınan H kesmede bırakıldı
    for col in ("reject_reason", "line_score", "line_margin", "carry_score", "carry_margin", "carry_reason", "cut_dist"):
        assert col in frows.columns, col
    df = pd.DataFrame(rows)
    before = set(df[(df.frame < 10) & (df.track_id >= 0)].track_id)
    after = set(df[(df.frame >= 10) & (df.track_id >= 0)].track_id)
    assert before and after and not (before & after), (before, after)


def test_reddedilen_adayin_skoru_ve_sebebi_yazilir():
    """#19: çizgi kontrolünden kalan fit'in skoru NaN'a çevrilmemeli, sebebi yazılmalı."""
    from tests.test_negative import camera, keypoints, lines_frame
    Hm = camera(50.0)
    ht = TrackingPipeline(None, None, None, PitchConfig(), FPS, tracker=lambda d: d).htrack
    _, src = ht.update(*keypoints(Hm, PitchConfig().vertices + [8.0, 0.0]), lines_frame(Hm))
    assert src != "fresh"
    assert ht.diag["reject_reason"] in ("score", "margin") and np.isfinite(ht.diag["line_score"]), ht.diag


def test_cli_dataclass_gidis_donus():
    ap = argparse.ArgumentParser()
    add_dataclass_args(ap, PipelineConfig)
    add_dataclass_args(ap, PostConfig)
    ns = ap.parse_args(["--min-line-score", "none", "--min-line-margin", "0.2", "--line-probes-m", "3", "6",
                        "--track-buffer-s", "1.5", "--cut-threshold", "0.4", "--max-ball-speed", "35"])
    cfg, post = dataclass_from_args(PipelineConfig, ns), dataclass_from_args(PostConfig, ns)
    assert cfg.min_line_score is None and cfg.min_line_margin == 0.2 and cfg.line_probes_m == (3.0, 6.0)
    assert cfg.track_buffer_s == 1.5 and cfg.cut_threshold == 0.4 and post.max_ball_speed == 35.0
    assert dataclass_from_args(PipelineConfig, ap.parse_args([])) == PipelineConfig()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

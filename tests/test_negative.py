"""Negatif testler: analizdeki bulguları mantık seviyesinde belgeler.

Faz 0'da `test_bulgu_*` testlerinin MEVCUT KODDA FAIL vermesi beklenir (her biri bir bulguyu
gösterir); Faz A'daki düzeltmelerden sonra geçmeleri gerekir. `test_kontrol_*` testleri bugün de
doğru olan davranışı sabitler; düzeltmeler bunları bozmamalı (ör. her H'yi reddeden bir "düzeltme").

Görsel gerçekçilik (gölge, pano, beyaz forma) burada YOK: onlar Faz K'da gerçek görüntüyle ölçülür.
Çizgili frame'ler test_synthetic'teki temiz render ile aynı: çim + 2 px beyaz çizgi.

Çalıştır:  python -m pytest tests/test_negative.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fai.models import PLAYER, make_tracker  # noqa: E402
from fai.pipeline import TrackingPipeline  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.postprocess import postprocess  # noqa: E402

P = PitchConfig()
W, H, FPS = 1280, 720, 25.0
GRASS, LINE = (60, 140, 55), (235, 235, 235)


# ----------------------------------------------------------------------- yardımcılar
def camera(xc, half_far=42.0, half_near=24.0, y_far=6.0, y_near=66.0) -> np.ndarray:
    """saha (m) -> görüntü (px). Yakın taç çizgisinin arkasından yayın kamerası (test_synthetic ile aynı model)."""
    src = np.float32([[xc - half_far, y_far], [xc + half_far, y_far], [xc + half_near, y_near], [xc - half_near, y_near]])
    dst = np.float32([[0, 95], [W, 95], [W, H - 10], [0, H - 10]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def lines_frame(Hm: np.ndarray) -> np.ndarray:
    img = np.full((H, W, 3), GRASS, np.uint8)
    img[:95] = (90, 90, 90)
    for pl in P.polylines():
        pts = cv2.perspectiveTransform(pl.reshape(-1, 1, 2), Hm)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, LINE, 2, cv2.LINE_AA)
    return img


def keypoints(Hm: np.ndarray, pitch_pts: np.ndarray | None = None):
    """Pose modelini taklit eder: keypoint i, pitch_pts[i]'nin görüntüdeki yerinde; görünmeyen -> (0,0), conf~0."""
    pts = P.vertices if pitch_pts is None else pitch_pts
    uv = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    vis = (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 100) & (uv[:, 1] < H - 5)
    uv[~vis] = 0.0
    return uv, np.where(vis, 0.9, 0.05)


def htracker():
    """Pipeline'ın kendi kurduğu HomographyTracker (aynı eşikler, aynı saha çizgileri)."""
    return TrackingPipeline(detector=None, keypointer=None, team_clf=None, pitch=P, fps=FPS,
                            tracker=lambda d: d).htrack


def ball_df(frames, xs, ys) -> pd.DataFrame:
    frames = np.asarray(frames)
    u = frames * 10.0
    return pd.DataFrame(dict(frame=frames, t=frames / FPS, track_id=-1, cls="ball", conf=0.5, team_raw=-1,
                             x1=u, y1=100.0, x2=u + 5, y2=105.0, u=u, v=105.0,
                             x=np.asarray(xs, float), y=np.asarray(ys, float), h_src="fresh"))


def frames_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(dict(frame=np.arange(n), t=np.arange(n) / FPS))


def tracker_id_of(out: sv.Detections, box: np.ndarray) -> int | None:
    """Tracker çıktısında verilen kutunun ID'si; kutu çıktıda yoksa (tracker düşürdüyse) None."""
    if len(out) == 0:
        return None
    d = np.abs(out.xyxy - box).sum(1)
    i = int(np.argmin(d))
    if d[i] > 1.0:
        return None
    return -1 if out.tracker_id is None else int(out.tracker_id[i])


# ------------------------------------------------------------ #1 x-kaymalı keypoint seti
def test_kontrol_dogru_keypointler_fresh():
    Hm = camera(50.0)
    ht = htracker()
    _, src = ht.update(*keypoints(Hm), lines_frame(Hm))
    assert src == "fresh" and ht.last_line_score > 0.5, (src, ht.last_line_score)


@pytest.mark.parametrize("dx", [-10.0, -5.0, 5.0, 10.0])
def test_bulgu01_x_kaymali_keypoint_seti_reddedilir(dx):
    """Model keypoint'leri x yönünde dx metre kaymış yerlere koyuyor (yanlış çizgi eşleşmesi).
    H tutarlı ama yanlış; taç çizgileri kendi üstüne düştüğü için line_score eşiği geçiyor."""
    Hm = camera(50.0)
    ht = htracker()
    _, src = ht.update(*keypoints(Hm, P.vertices + [dx, 0.0]), lines_frame(Hm))
    assert src != "fresh", f"dx={dx:+.0f} m kaymış H 'fresh' kabul edildi (line_score={ht.last_line_score:.2f})"


# ------------------------------------------------------ #2 sahne kesmesi sonrası carried H
def _cut_frame(kind: str) -> np.ndarray:
    if kind == "yakin_cekim":                                  # çizgisiz, kadrajı dolduran oyuncu
        img = np.full((H, W, 3), GRASS, np.uint8)
        cv2.rectangle(img, (450, 60), (830, H), (40, 40, 200), -1)
        return img
    # farklı kamera: sağ ceza sahasına zoom
    return lines_frame(camera(80.0, half_far=18.0, half_near=10.0, y_far=20.0, y_near=60.0))


@pytest.mark.parametrize("cut_to", ["yakin_cekim", "baska_kamera"])
def test_bulgu02_kesme_sonrasi_eski_h_tasinmaz(cut_to):
    """Kesmeden sonra keypoint yok; eski H yeni görüntüyle örtüşmediği halde 12 frame 'carried' kullanılıyor."""
    Hm = camera(50.0)
    ht = htracker()
    _, src = ht.update(*keypoints(Hm), lines_frame(Hm))
    assert src == "fresh"                                      # ön koşul
    new = _cut_frame(cut_to)
    for k in range(3):
        _, src = ht.update(None, None, new)
        assert src != "carried", (f"kesmeden {k + 1} frame sonra eski H 'carried' kullanıldı "
                                  f"(eski H'nin yeni frame'deki line_score={ht.last_line_score:.2f})")


def test_kontrol_kesme_yokken_h_tasinir():
    Hm = camera(50.0)
    ht = htracker()
    frame = lines_frame(Hm)
    ht.update(*keypoints(Hm), frame)
    _, src = ht.update(None, None, frame)                       # aynı görüntü, keypoint kaçtı
    assert src == "carried"


# ----------------------------------------------------------------- #6 ilk top tespiti sahte
def test_bulgu06_ilk_top_tespiti_sahte():
    """İlk tespit sahte (80,10); sonra gerçek top (30,34) civarında yavaş. Filtre ilk tespite kilitleniyor."""
    n = 100
    fr = np.arange(n)
    x, y = 30 + 0.04 * fr, np.full(n, 34.0)
    x[0], y[0] = 80.0, 10.0
    out = postprocess(ball_df(fr, x, y), frames_df(n), FPS)
    det = out[(out.cls == "ball") & ~out["interp"]].set_index("frame")["outlier"]
    n_true_flagged = int(det.loc[1:].sum())
    assert bool(det.loc[0]) and n_true_flagged == 0, (
        f"sahte ilk tespit outlier={bool(det.loc[0])}; ardından gelen {n_true_flagged}/{n - 1} gerçek tespit outlier işaretlendi")


def test_kontrol_top_sicramasi_yakalanir():
    n = 60
    fr = np.arange(n)
    x, y = 30 + 0.04 * fr, np.full(n, 34.0)
    x[30], y[30] = 80.0, 10.0                                   # ortada tek sahte top
    out = postprocess(ball_df(fr, x, y), frames_df(n), FPS)
    det = out[(out.cls == "ball") & ~out["interp"]].set_index("frame")["outlier"]
    assert bool(det.loc[30]) and int(det.drop(index=30).sum()) == 0


# ---------------------------------------------------------------- #7 3 s top boşluğu
def test_bulgu07_uzun_top_boslugu_doldurulmaz():
    """75 frame (3 s) boşluk; ball_max_gap_s=1 s. Boşluk 1 s'den uzunsa hiç doldurulmamalı."""
    fr = np.r_[np.arange(0, 20), np.arange(95, 120)]
    out = postprocess(ball_df(fr, np.linspace(10, 60, len(fr)), np.full(len(fr), 30.0)), frames_df(120), FPS)
    ip = out[out["interp"] & (out.frame > 19) & (out.frame < 95)]
    assert len(ip) == 0, f"3 s'lik boşluğa {len(ip)} interpolasyon satırı eklendi (frame {ip.frame.min()}..{ip.frame.max()})"


def test_kontrol_kisa_top_boslugu_doldurulur():
    fr = np.r_[np.arange(0, 20), np.arange(30, 50)]             # 10 frame (0.4 s) boşluk
    out = postprocess(ball_df(fr, np.linspace(10, 60, len(fr)), np.full(len(fr), 30.0)), frames_df(50), FPS)
    assert sorted(out.loc[out["interp"], "frame"]) == list(range(20, 30))


# ------------------------------------------------------------- #8 hakem sınıf titremesi
def test_bulgu08_hakem_sinif_titremesi():
    """track 5 = hakem (%70 referee, %30 player etiketli); track 7 = oyuncu (%90 player, %10 referee).
    Beklenen: track başına tek rol; hakemin takımı hep -1, oyuncunun takımı hep oylanan takım (1)."""
    rows = []
    for f in range(50):
        ref_cls = "referee" if f % 10 < 7 else "player"
        pl_cls = "player" if f % 10 else "referee"
        base = dict(frame=f, t=f / FPS, conf=0.8, x1=0.0, y1=0.0, x2=1.0, y2=1.0, u=0.0, v=0.0, y=30.0, h_src="fresh")
        rows.append(dict(base, track_id=5, cls=ref_cls, team_raw=0 if ref_cls == "player" else -1, x=50.0))
        rows.append(dict(base, track_id=7, cls=pl_cls, team_raw=1 if pl_cls == "player" else -1, x=40.0))
    out = postprocess(pd.DataFrame(rows), frames_df(50), FPS)
    ref, pl = out[out.track_id == 5], out[out.track_id == 7]
    got = {"hakem role": ref.role.value_counts().to_dict(), "hakem team": ref.team.value_counts().to_dict(),
           "oyuncu role": pl.role.value_counts().to_dict(), "oyuncu team": pl.team.value_counts().to_dict()}
    assert (set(ref.role) == {"referee"} and (ref.team == -1).all()
            and set(pl.role) == {"player"} and (pl.team == 1).all()), got


# ---------------------------------------------------------- #9 conf 0.3–0.5 tespitler
def test_bulgu09_orta_guvenli_tespit_track_alir():
    """Detector conf eşiği 0.3; 0.3–0.5 arası tespitler datasete giriyor ama ByteTrack'te hiç track başlatamıyor
    (high_conf_det_threshold=0.5) -> sürekli track_id=-1."""
    trk = make_tracker(FPS)
    hi, mid = [], []
    for f in range(30):
        xyxy = np.array([[100 + f, 200, 140 + f, 300], [600, 200, 640, 300]], float)
        out = trk(sv.Detections(xyxy=xyxy, confidence=np.array([0.9, 0.4]), class_id=np.array([PLAYER, PLAYER])))
        hi.append(tracker_id_of(out, xyxy[0]))
        mid.append(tracker_id_of(out, xyxy[1]))
    assert all(i is not None and i >= 0 for i in hi[2:]), f"kontrol (conf=0.9) track almadı: {hi}"
    tracked = np.mean([i is not None and i >= 0 for i in mid[5:]])
    assert tracked >= 0.9, f"conf=0.4 sabit tespit 30 frame boyunca track almadı (ID'li frame oranı %{100 * tracked:.0f}): {mid}"


# ---------------------------------------------------------- #9 50 fps'te lost_track_buffer
@pytest.mark.parametrize("gap_s,same_id", [(1.5, True), (2.6, False)], ids=["kontrol_1.5s_ayni_id", "bulgu09_2.6s_yeni_id"])
def test_lost_track_buffer_50fps(gap_s, same_id):
    """Kayıp track 2 s tutulmalı. make_tracker lost_track_buffer=2*fps veriyor, trackers bunu ayrıca fps/30 ile
    ölçekliyor: 50 fps'te 166 frame (3.3 s)."""
    fps = 50.0
    trk = make_tracker(fps)
    box = np.array([[600, 200, 640, 300]], float)

    def det():
        return sv.Detections(xyxy=box.copy(), confidence=np.array([0.9]), class_id=np.array([PLAYER]))

    for _ in range(20):
        out = trk(det())
    first = tracker_id_of(out, box[0])
    assert first is not None and first >= 0
    for _ in range(int(round(gap_s * fps))):
        trk(sv.Detections.empty())
    for _ in range(3):
        out = trk(det())
    later = tracker_id_of(out, box[0])
    if same_id:
        assert later == first, f"{gap_s} s kayıptan sonra ID değişti ({first} -> {later})"
    else:
        assert later != first, f"{gap_s} s (> 2 s) kayıptan sonra aynı ID ({first}) geri geldi: kayıp track 2 s'den uzun tutuluyor"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

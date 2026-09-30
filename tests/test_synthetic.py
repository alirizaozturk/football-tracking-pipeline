"""Sentetik uçtan uca test: modelsiz, GPU'suz.

Bilinen bir kamera (homography) ve bilinen oyuncu hareketleriyle sahte bir yayın
videosu üretir; detector/keypoint modellerini ground-truth'tan gürültülü çıktı veren
sahtelerle değiştirir. Gerçek tracker, homography, takım sınıflandırıcı (color),
postprocess, analiz ve render kodu çalışır. Böylece model-dışı her şey ölçülebilir:

    * saha koordinatı hatası (m)
    * takım ataması doğruluğu
    * hız hatası
    * top outlier temizliği

Çalıştır:  python -m tests.test_synthetic      (ya da: pytest -q tests)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fai.analysis import player_stats, possession_proxy, team_shape  # noqa: E402
from fai.build import build_match_dataset  # noqa: E402
from fai.dataio import load_match  # noqa: E402
from fai.models import BALL, GOALKEEPER, PLAYER, REFEREE  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.render import render_match  # noqa: E402
from fai.viz import Radar  # noqa: E402

W, H, FPS, SECONDS = 1280, 720, 25, 6
PITCH = PitchConfig()
RNG = np.random.default_rng(7)
KIT = {0: (40, 40, 210), 1: (210, 120, 20), "gk0": (20, 220, 220), "gk1": (0, 140, 255), "ref": (25, 25, 25)}


# ------------------------------------------------------------------ scene model
def camera_H(t: float) -> np.ndarray:
    """pitch(m) -> image(px). Kamera yakın taç çizgisinin arkasından, sağa doğru pan yapıyor."""
    xc = 38 + 3.0 * t                                       # pan
    src = np.float32([[xc - 42, 6], [xc + 42, 6], [xc + 24, 66], [xc - 24, 66]])
    dst = np.float32([[0, 95], [W, 95], [W, H - 10], [0, H - 10]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def make_people():
    ppl = []
    for team in (0, 1):
        # kaleci
        gx = 4.0 if team == 0 else PITCH.length - 4.0
        ppl.append(dict(team=team, role="goalkeeper", x0=gx + (6 if team == 0 else -6) , y0=34, A=(2, 3), w=(0.4, 0.5)))
        # 10 oyuncu, sahanın orta bölgesine dağılmış (kameranın gördüğü yer)
        for _ in range(10):
            ppl.append(dict(team=team, role="player", x0=RNG.uniform(22, 70), y0=RNG.uniform(10, 60),
                            A=RNG.uniform(2, 7, 2), w=RNG.uniform(0.3, 0.9, 2)))
    ppl.append(dict(team=-1, role="referee", x0=48, y0=30, A=(4, 3), w=(0.5, 0.4)))
    for p in ppl:
        p["phi"] = RNG.uniform(0, 2 * np.pi, 2)
    return ppl


def pos(p, t):
    return np.array([p["x0"] + p["A"][0] * np.sin(p["w"][0] * t + p["phi"][0]),
                     p["y0"] + p["A"][1] * np.sin(p["w"][1] * t + p["phi"][1])])


def vel(p, t):
    return np.array([p["A"][0] * p["w"][0] * np.cos(p["w"][0] * t + p["phi"][0]),
                     p["A"][1] * p["w"][1] * np.cos(p["w"][1] * t + p["phi"][1])])


def ball_pos(ppl, t):
    """Top 4 oyuncu arasında pas: her pas 1.5 s."""
    order = [1, 13, 5, 17, 3]
    k = min(int(t // 1.5), len(order) - 2)
    a, b = pos(ppl[order[k]], k * 1.5), pos(ppl[order[k + 1]], (k + 1) * 1.5)
    s = np.clip((t - k * 1.5) / 1.5, 0, 1)
    return a + (b - a) * s


def ppm_at(Hm, xy):
    """o noktadaki piksel/metre ölçeği (kişi boyunu kabaca ölçeklemek için)."""
    a = cv2.perspectiveTransform(np.array([[xy, xy + [0, 1.0]]], np.float64), Hm)[0]
    b = cv2.perspectiveTransform(np.array([[xy, xy + [1.0, 0]]], np.float64), Hm)[0]
    return max(np.linalg.norm(a[1] - a[0]), np.linalg.norm(b[1] - b[0]))


def scene(ppl, f):
    t = f / FPS
    Hm = camera_H(t)
    objs = []
    for j, p in enumerate(ppl):
        xy = pos(p, t)
        u, v = cv2.perspectiveTransform(xy.reshape(1, 1, 2), Hm).ravel()
        hgt = 1.85 * ppm_at(Hm, xy) * 1.6
        wid = 0.42 * hgt
        box = np.array([u - wid / 2, v - hgt, u + wid / 2, v])
        cls = {"player": PLAYER, "goalkeeper": GOALKEEPER, "referee": REFEREE}[p["role"]]
        objs.append(dict(j=j, cls=cls, team=p["team"], role=p["role"], xy=xy, uv=(u, v), box=box, vel=vel(p, t)))
    bxy = ball_pos(ppl, t)
    bu, bv = cv2.perspectiveTransform(bxy.reshape(1, 1, 2), Hm).ravel()
    r = 0.22 * ppm_at(Hm, bxy) * 1.6
    ball = dict(xy=bxy, uv=(bu, bv), box=np.array([bu - r, bv - 2 * r, bu + r, bv]))
    return Hm, objs, ball


def render_frame(Hm, objs, ball):
    img = np.full((H, W, 3), (60, 140, 55), np.uint8)
    img[:95] = (90, 90, 90)                                  # tribün
    for pl in PITCH.polylines():
        pts = cv2.perspectiveTransform(pl.reshape(-1, 1, 2), Hm)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
    for o in sorted(objs, key=lambda o: o["uv"][1]):         # uzaktan yakına çiz
        x1, y1, x2, y2 = o["box"].astype(int)
        h = y2 - y1
        kit = KIT["ref"] if o["role"] == "referee" else KIT[f"gk{o['team']}"] if o["role"] == "goalkeeper" else KIT[o["team"]]
        cv2.rectangle(img, (x1, y1 + int(.18 * h)), (x2, y1 + int(.58 * h)), kit, -1)        # forma
        cv2.rectangle(img, (x1 + 1, y1 + int(.58 * h)), (x2 - 1, y1 + int(.75 * h)), (30, 30, 30), -1)  # şort
        cv2.rectangle(img, (x1 + 2, y1 + int(.75 * h)), (x2 - 2, y2), (140, 170, 200), -1)   # bacak
        cv2.circle(img, ((x1 + x2) // 2, y1 + int(.09 * h)), max(2, int(.09 * h)), (120, 160, 210), -1)  # kafa
    bx1, by1, bx2, by2 = ball["box"].astype(int)
    cv2.circle(img, ((bx1 + bx2) // 2, (by1 + by2) // 2), max(2, (bx2 - bx1) // 2), (250, 250, 250), -1)
    return img


# ------------------------------------------------------------------------ mocks
class FakeWorld:
    """Sahte modeller hangi frame'de olduklarını pipeline'ın verdiği frame_idx'ten alır
    (çağrı sırasına / sayaca bağlı değil)."""

    def __init__(self, ppl):
        self.ppl = ppl

    def detector(self, frame, frame_idx):
        Hm, objs, ball = scene(self.ppl, frame_idx)
        xyxy, cls, conf = [], [], []
        for o in objs:
            b = o["box"] + RNG.normal(0, 1.0, 4)
            if b[2] < 0 or b[0] > W or b[3] < 0 or b[1] > H:
                continue
            xyxy.append(b); cls.append(o["cls"]); conf.append(RNG.uniform(0.6, 0.95))
        if RNG.random() > 0.1:                                  # %10 top kaçırma
            b = ball["box"].copy()
            if frame_idx in (40, 90):                           # 2 kez sahte top (outlier testi)
                b += np.array([500, -200, 500, -200])
            xyxy.append(b); cls.append(BALL); conf.append(0.5)
        if not xyxy:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.array(xyxy, float), class_id=np.array(cls), confidence=np.array(conf))

    def keypointer(self, frame, frame_idx):
        if frame_idx % 20 == 7:                                 # ara sıra keypoint yok -> carry testi
            return None, None
        Hm = camera_H(frame_idx / FPS)
        uv = cv2.perspectiveTransform(PITCH.vertices.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        uv += RNG.normal(0, 1.5, uv.shape)
        vis = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 95) & (uv[:, 1] < H)
        conf = np.where(vis, RNG.uniform(0.7, 0.99, len(uv)), RNG.uniform(0.0, 0.2, len(uv)))
        return uv, conf


def make_video(path: Path, ppl, n):
    info = sv.VideoInfo(width=W, height=H, fps=FPS)
    gt = []
    with sv.VideoSink(str(path), info) as sink:
        for f in range(n):
            Hm, objs, ball = scene(ppl, f)
            sink.write_frame(render_frame(Hm, objs, ball))
            for o in objs:
                gt.append(dict(frame=f, j=o["j"], team_gt=o["team"], role_gt=o["role"], gx=o["xy"][0], gy=o["xy"][1],
                               gu=o["uv"][0], gv=o["uv"][1], gspeed=np.linalg.norm(o["vel"])))
            gt.append(dict(frame=f, j=-1, team_gt=-1, role_gt="ball", gx=ball["xy"][0], gy=ball["xy"][1],
                           gu=ball["uv"][0], gv=ball["uv"][1], gspeed=np.nan))
    return pd.DataFrame(gt)


# ------------------------------------------------------------------------- test
def run(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    ppl = make_people()
    n = FPS * SECONDS
    video = out_root / "synthetic.mp4"
    gt = make_video(video, ppl, n)
    world = FakeWorld(ppl)

    # (a) takım fit aşaması: videoyu FPS adımıyla okuyup kırpıntı topla
    from fai.pipeline import collect_player_crops
    from fai.teams import TeamClassifier

    frames = sv.get_video_frames_generator(str(video), stride=FPS)
    crops = collect_player_crops(((k * FPS, f) for k, f in enumerate(frames)), world.detector)
    match_dir = out_root / "match"
    match_dir.mkdir(exist_ok=True)
    TeamClassifier("color").fit(crops).save(match_dir / "team_classifier.pkl")   # build bunu yükler

    # (b) tracking aşaması
    build_match_dataset(video, match_dir, world.detector, world.keypointer, team_method="color", fmt="parquet")

    df, frames_df, meta = load_match(match_dir)

    # -------- metrikler: dataset satırlarını GT'ye piksel uzayında eşle
    ppl_rows = df[df["cls"] != "ball"].merge(gt[gt["j"] >= 0], on="frame")
    ppl_rows["dpx"] = np.hypot(ppl_rows["u"] - ppl_rows["gu"], ppl_rows["v"] - ppl_rows["gv"])
    m = ppl_rows.loc[ppl_rows.groupby(["frame", "u", "v"])["dpx"].idxmin()]
    m = m[m["dpx"] < 5]
    m["err_m"] = np.hypot(m["x"] - m["gx"], m["y"] - m["gy"])
    fresh = m[m["h_src"] == "fresh"]
    carried = m[m["h_src"] == "carried"]

    pl = m[m["role_gt"] == "player"]
    mapping = pl.groupby("team_gt")["team"].agg(lambda s: s.mode().iloc[0]).to_dict()
    team_acc = (pl["team"] == pl["team_gt"].map(mapping)).mean()
    gk = m[m["role_gt"] == "goalkeeper"]
    gk_acc = (gk["team"] == gk["team_gt"].map(mapping)).mean() if len(gk) else np.nan

    spd = m[(m["track_id"] >= 0) & m["speed_mps"].notna()]
    speed_mae = (spd["speed_mps"] - spd["gspeed"]).abs().median()

    ball_rows = df[df["cls"] == "ball"]
    detected = ball_rows[~ball_rows["interp"]]
    outliers_caught = bool(detected.loc[detected["frame"].isin([40, 90]), "outlier"].all()) and \
        int(detected["outlier"].sum()) == int(detected["frame"].isin([40, 90]).sum())
    bgt = gt[gt["j"] == -1].set_index("frame")
    bok = ball_rows[~ball_rows["outlier"]].set_index("frame")
    ball_err = np.hypot(bok["x"] - bgt.loc[bok.index, "gx"], bok["y"] - bgt.loc[bok.index, "gy"]).median()

    n_ids = df.loc[(df["track_id"] >= 0), "track_id"].nunique()
    n_people = len(ppl)

    # -------- analiz + render (çıktı üretiyor mu?)
    stats = player_stats(df)
    poss = possession_proxy(df, fps=FPS)
    shape = team_shape(df)
    stats.to_csv(out_root / "player_stats.csv")
    radar = Radar(PITCH)
    cv2.imwrite(str(out_root / "heatmap_team0.png"),
                radar.heatmap(df[(df["team"] == 0) & (df["role"] == "player")][["xs", "ys"]].to_numpy()))
    f_mid = n // 2
    from fai.render import radar_objects
    cv2.imwrite(str(out_root / "voronoi_mid.png"),
                radar.draw_frame(radar_objects(df[df["frame"] == f_mid]), voronoi=True))
    render_match(match_dir, out_root / "render_voronoi.mp4", mode="voronoi", max_frames=n)
    # render'ın bir frame'ini görsel kontrol için kaydet
    cap = cv2.VideoCapture(str(out_root / "render_voronoi.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_mid)
    ok, fr = cap.read()
    if ok:
        cv2.imwrite(str(out_root / "render_mid.png"), fr)

    res = dict(
        frames=n, rows=len(df), unique_track_ids=int(n_ids), people=n_people,
        homography_coverage=float(meta["homography_coverage"]),
        line_score_median=float(meta["line_score_median"]),
        homography_rejected=meta["homography_rejected"],
        pos_err_median_m_fresh=float(fresh["err_m"].median()),
        pos_err_p95_m_fresh=float(fresh["err_m"].quantile(.95)),
        pos_err_median_m_carried=float(carried["err_m"].median()) if len(carried) else np.nan,
        team_acc_players=float(team_acc), team_acc_goalkeepers=float(gk_acc),
        speed_err_median_mps=float(speed_mae),
        ball_outliers_caught=bool(outliers_caught), ball_err_median_m=float(ball_err),
        ball_interp_rows=int(df["interp"].sum()),
        possession_frames_with_owner=int(poss["team_held"].notna().sum()),
        team_shape_rows=len(shape),
    )
    return res


def test_synthetic(tmp_path=None):
    out = Path(tmp_path) if tmp_path else ROOT / "tests" / "_out"
    r = run(out)
    for k, v in r.items():
        print(f"{k:32s} {v}")
    assert r["homography_coverage"] > 0.95
    assert r["line_score_median"] > 0.4
    assert r["pos_err_median_m_fresh"] < 0.3
    assert r["team_acc_players"] > 0.98
    assert r["team_acc_goalkeepers"] > 0.9
    assert r["speed_err_median_mps"] < 0.6
    assert r["ball_outliers_caught"]
    assert r["ball_interp_rows"] > 0


if __name__ == "__main__":
    test_synthetic()

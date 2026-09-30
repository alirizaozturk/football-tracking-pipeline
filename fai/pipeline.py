"""Frame başına işlem: detection -> tracking -> homography -> takım -> satırlar.

Pipeline model çağrılarını dışarıdan alır (callable). Böylece aynı kod gerçek
YOLO modelleriyle de, testlerdeki sahte modellerle de çalışır.

Çıktı satır şeması (tracking datası, obje başına bir satır):
    frame, t, track_id, cls, conf, team_raw,
    x1, y1, x2, y2,          # bbox (piksel)
    u, v,                    # ayak noktası / zemin temas noktası (piksel)
    x, y,                    # saha koordinatı (metre), homography yoksa NaN
    h_src                    # 'fresh' | 'carried' | 'none'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import supervision as sv

from .homography import HomographyTracker, image_to_pitch
from .models import BALL, CLASSES, GOALKEEPER, PLAYER, REFEREE, make_tracker
from .pitch import PitchConfig
from .teams import TeamClassifier, assign_goalkeepers

Detector = Callable[[np.ndarray], sv.Detections]
Keypointer = Callable[[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


@dataclass
class PipelineConfig:
    nms_iou: float = 0.5          # player/goalkeeper çift etiketini temizlemek için class-agnostic NMS
    ball_conf: float = 0.2
    kp_conf: float = 0.5          # homography'e girecek keypoint güven eşiği
    ransac_px: float = 8.0
    max_carry: int = 12           # keypoint yoksa son homography kaç frame taşınsın
    max_rmse_px: float = 15.0
    min_line_score: float = 0.15  # homography'nin görüntüdeki beyaz çizgilerle örtüşme oranı alt sınırı


def crops_of(frame: np.ndarray, xyxy: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in np.asarray(xyxy).reshape(-1, 4):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        c = frame[y1:y2, x1:x2]
        out.append(c if c.size else np.zeros((4, 4, 3), np.uint8))
    return out


def collect_player_crops(frames: Iterable[np.ndarray], detector: Detector,
                         max_crops: int = 2000, min_h: int = 20) -> List[np.ndarray]:
    """Takım sınıflandırıcıyı fit etmek için videodan oyuncu kırpıntıları topla."""
    crops: List[np.ndarray] = []
    for frame in frames:
        d = detector(frame)
        d = d[d.class_id == PLAYER]
        hgt = d.xyxy[:, 3] - d.xyxy[:, 1] if len(d) else np.zeros(0)
        crops += crops_of(frame, d.xyxy[hgt >= min_h])
        if len(crops) >= max_crops:
            break
    return crops[:max_crops]


class TrackingPipeline:
    def __init__(self, detector: Detector, keypointer: Optional[Keypointer],
                 team_clf: TeamClassifier, pitch: PitchConfig = PitchConfig(), fps: float = 25.0,
                 ball_detector: Optional[Detector] = None, cfg: PipelineConfig = PipelineConfig(),
                 tracker: Optional[Callable[[sv.Detections], sv.Detections]] = None):
        self.detector, self.keypointer, self.ball_detector = detector, keypointer, ball_detector
        self.team_clf, self.pitch, self.fps, self.cfg = team_clf, pitch, fps, cfg
        self.tracker = tracker or make_tracker(fps)
        self.htrack = HomographyTracker(pitch.vertices, cfg.kp_conf, cfg.ransac_px, cfg.max_carry,
                                        cfg.max_rmse_px, (pitch.length, pitch.width),
                                        pitch.polylines(), cfg.min_line_score)

    # ------------------------------------------------------------------ helpers
    def _ball(self, frame: np.ndarray, dets: sv.Detections) -> sv.Detections:
        b = self.ball_detector(frame) if self.ball_detector is not None else dets[dets.class_id == BALL]
        if self.ball_detector is not None:
            b = b[b.class_id == BALL]
        b = b[b.confidence >= self.cfg.ball_conf] if len(b) and b.confidence is not None else b
        if len(b) > 1:                         # sahada tek top var: en güvenlisini al
            b = b[[int(np.argmax(b.confidence))]]
        return b

    # --------------------------------------------------------------------- main
    def process(self, frame_idx: int, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        t = frame_idx / self.fps
        dets = self.detector(frame)
        ball = self._ball(frame, dets)

        people = dets[dets.class_id != BALL]
        if len(people):
            people = people.with_nms(threshold=self.cfg.nms_iou, class_agnostic=True)
        people = self.tracker(people)
        if people.tracker_id is None:
            people.tracker_id = np.full(len(people), -1, dtype=int)

        # homography
        kp_xy, kp_conf = self.keypointer(frame) if self.keypointer is not None else (None, None)
        H, h_src = self.htrack.update(kp_xy, kp_conf, frame)

        def to_pitch(pts: np.ndarray) -> np.ndarray:
            if H is None or len(pts) == 0:
                return np.full((len(pts), 2), np.nan)
            return image_to_pitch(H, pts)

        uv = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(people) else np.zeros((0, 2))
        xy = to_pitch(uv)

        # takımlar
        team = np.full(len(people), -1, dtype=int)
        is_pl = people.class_id == PLAYER
        if is_pl.any():
            team[is_pl] = self.team_clf.predict(crops_of(frame, people.xyxy[is_pl]))
        is_gk = people.class_id == GOALKEEPER
        if is_gk.any():
            space_gk, space_pl = (xy[is_gk], xy[is_pl]) if H is not None else (uv[is_gk], uv[is_pl])
            team[is_gk] = assign_goalkeepers(space_gk, space_pl, team[is_pl])
        # hakemler -1 kalır

        rows: List[Dict] = []
        conf = people.confidence if people.confidence is not None else np.ones(len(people))
        for i in range(len(people)):
            x1, y1, x2, y2 = people.xyxy[i]
            rows.append(dict(frame=frame_idx, t=t, track_id=int(people.tracker_id[i]),
                             cls=CLASSES[int(people.class_id[i])], conf=float(conf[i]),
                             team_raw=int(team[i]), x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                             u=float(uv[i, 0]), v=float(uv[i, 1]),
                             x=float(xy[i, 0]), y=float(xy[i, 1]), h_src=h_src))
        if len(ball):
            buv = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            bxy = to_pitch(buv)
            x1, y1, x2, y2 = ball.xyxy[0]
            rows.append(dict(frame=frame_idx, t=t, track_id=-1, cls="ball",
                             conf=float(ball.confidence[0]) if ball.confidence is not None else 1.0,
                             team_raw=-1, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                             u=float(buv[0, 0]), v=float(buv[0, 1]),
                             x=float(bxy[0, 0]), y=float(bxy[0, 1]), h_src=h_src))

        frow: Dict = dict(frame=frame_idx, t=t, h_src=h_src,
                          n_kp=int((np.asarray(kp_conf) >= self.cfg.kp_conf).sum()) if kp_conf is not None else 0,
                          n_inliers=H.n_inliers if H is not None else 0,
                          rmse_px=H.reproj_rmse_px if H is not None else np.nan,
                          line_score=self.htrack.last_line_score)
        Hm = H.H_img2pitch.ravel() if H is not None else np.full(9, np.nan)
        for k in range(9):
            frow[f"h{k}"] = float(Hm[k])
        return rows, frow

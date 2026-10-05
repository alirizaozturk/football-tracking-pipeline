"""Frame başına işlem: detection -> tracking -> homography -> takım -> satırlar.

Pipeline model çağrılarını dışarıdan alır (callable). Böylece aynı kod gerçek
YOLO modelleriyle de, testlerdeki sahte modellerle de çalışır. Modeller
`model(frame, frame_idx)` şeklinde çağrılır; gerçek modeller frame_idx'i yok sayar,
sahte modeller hangi frame'de olduklarını çağrı sırasından değil buradan öğrenir.

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

import cv2
import numpy as np
import supervision as sv

from .homography import HomographyTracker, LineCheck, image_to_pitch
from .models import BALL, CLASSES, GOALKEEPER, PLAYER, REFEREE, make_tracker
from .pitch import PitchConfig
from .teams import TeamClassifier, assign_goalkeepers

Detector = Callable[[np.ndarray, int], sv.Detections]
Keypointer = Callable[[np.ndarray, int], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


@dataclass
class PipelineConfig:
    """Pipeline eşikleri. Hepsi 06_build_tracking_dataset.py'de --alan-adi şeklinde CLI parametresi."""
    nms_iou: float = 0.5                 # player/goalkeeper çift etiketini temizlemek için class-agnostic NMS
    ball_conf: float = 0.2
    # homography
    kp_conf: float = 0.5                 # homography'e girecek keypoint güven eşiği
    ransac_px: float = 8.0               # RANSAC eşiği, ref_width genişliğinde frame için (px); genişlikle ölçeklenir
    max_rmse_px: float = 15.0            # inlier rmse üst sınırı, ref_width için (px); genişlikle ölçeklenir
    ref_width: float = 1280.0
    max_carry: int = 12                  # keypoint yoksa son homography en fazla kaç frame taşınsın
    # çizgi kanıtı (Faz K kalibrasyonu + Faz A gürültü çalışması; docs/calibration/field-v12). Geçici: Faz 1'de
    # modelin keypoint'leriyle yeniden kalibre edilecek.
    line_tophat_thr: float = 10.0        # L kanalı top-hat yanıt eşiği
    line_kernel_div: float = 128.0       # top-hat çekirdeği = W / bu
    line_tol_div: float = 640.0          # örnek konum toleransı = W / bu (px)
    min_line_score: Optional[float] = None   # None ('none') = skor eşiği yok; kabul = geometri + marj
    min_line_margin: float = 0.12        # skor(H) - max skor(probe)
    line_probes_m: Tuple[float, ...] = (2.0, 4.0, 7.0)
    line_step_m: float = 0.5
    line_min_samples: int = 30
    # tracker (ByteTrack)
    track_activation: float = 0.25       # yeni track başlatmak için min güven
    track_high_conf: float = 0.3         # ilk eşleştirme aşaması eşiği; dataset'e giren her tespit track başlatabilsin diye det conf ile aynı
    track_buffer_s: float = 2.0          # kayıp track kaç saniye tutulsun
    track_min_consecutive: int = 1
    track_min_iou: float = 0.1
    # sahne kesmesi: ardışık frame HSV histogramlarının Bhattacharyya mesafesi bunu aşarsa kesme -> tracker ve taşınan
    # H sıfırlanır. Gerçek yayında kalibre EDİLMEDİ (frames.parquet -> cut_dist ile ayarla); 'none' = kapalı.
    cut_threshold: Optional[float] = 0.5

    def line_check(self) -> LineCheck:
        return LineCheck(tophat_thr=self.line_tophat_thr, kernel_div=self.line_kernel_div, tol_div=self.line_tol_div,
                         min_score=self.min_line_score, min_margin=self.min_line_margin,
                         probes_m=tuple(self.line_probes_m), step_m=self.line_step_m, min_samples=self.line_min_samples)


def build_htracker(pitch: PitchConfig, cfg: PipelineConfig) -> HomographyTracker:
    """Pipeline'ın, 03_check_keypoints'in ve kalibrasyonun kullandığı ortak kabul zinciri."""
    return HomographyTracker(pitch.vertices, cfg.kp_conf, cfg.ransac_px, cfg.max_carry, cfg.max_rmse_px,
                             (pitch.length, pitch.width), pitch.polylines(), cfg.line_check(), cfg.ref_width)


def hist_distance(prev_hist: Optional[np.ndarray], frame: np.ndarray) -> Tuple[float, np.ndarray]:
    """Sahne kesmesi için: küçültülmüş frame'in H-S histogramı ve öncekiyle Bhattacharyya mesafesi (0-1)."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    d = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
    return d, hist


def crops_of(frame: np.ndarray, xyxy: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in np.asarray(xyxy).reshape(-1, 4):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        c = frame[y1:y2, x1:x2]
        out.append(c if c.size else np.zeros((4, 4, 3), np.uint8))
    return out


def collect_player_crops(frames: Iterable[Tuple[int, np.ndarray]], detector: Detector,
                         max_crops: int = 2000, min_h: int = 20) -> List[np.ndarray]:
    """Takım sınıflandırıcıyı fit etmek için videodan oyuncu kırpıntıları topla.
    frames: (frame_idx, frame) çiftleri."""
    crops: List[np.ndarray] = []
    for frame_idx, frame in frames:
        d = detector(frame, frame_idx)
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
        # dışarıdan verilen tracker kesmede sıfırlanamaz; varsayılan tracker her kesmede yeniden kurulur
        self._tracker_factory = None if tracker is not None else (lambda: make_tracker(fps, cfg))
        self.tracker = tracker or self._tracker_factory()
        self.htrack = build_htracker(pitch, cfg)
        self._prev_hist: Optional[np.ndarray] = None
        self._id_offset, self._next_id = 0, 0    # kesmede yeni tracker ID'leri 0'dan başlatır; çakışmasın diye kaydır
        self.n_cuts = 0

    def _scene_cut(self, frame: np.ndarray) -> Tuple[bool, float]:
        d, self._prev_hist = hist_distance(self._prev_hist, frame)
        cut = self.cfg.cut_threshold is not None and d > self.cfg.cut_threshold
        if cut:
            self.n_cuts += 1
            self.htrack.reset()
            if self._tracker_factory is not None:
                self.tracker = self._tracker_factory()
                self._id_offset = self._next_id
        return cut, d

    # ------------------------------------------------------------------ helpers
    def _ball(self, frame: np.ndarray, frame_idx: int, dets: sv.Detections) -> sv.Detections:
        b = self.ball_detector(frame, frame_idx) if self.ball_detector is not None else dets[dets.class_id == BALL]
        if self.ball_detector is not None:
            b = b[b.class_id == BALL]
        b = b[b.confidence >= self.cfg.ball_conf] if len(b) and b.confidence is not None else b
        if len(b) > 1:                         # sahada tek top var: en güvenlisini al
            b = b[[int(np.argmax(b.confidence))]]
        return b

    # --------------------------------------------------------------------- main
    def process(self, frame_idx: int, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        t = frame_idx / self.fps
        cut, cut_dist = self._scene_cut(frame)
        dets = self.detector(frame, frame_idx)
        ball = self._ball(frame, frame_idx, dets)

        people = dets[dets.class_id != BALL]
        if len(people):
            people = people.with_nms(threshold=self.cfg.nms_iou, class_agnostic=True)
        people = self.tracker(people)
        if people.tracker_id is None:
            people.tracker_id = np.full(len(people), -1, dtype=int)
        tid = np.asarray(people.tracker_id, dtype=int)
        tid = np.where(tid >= 0, tid + self._id_offset, -1)
        if len(tid) and tid.max() >= self._next_id:
            self._next_id = int(tid.max()) + 1

        # homography
        kp_xy, kp_conf = self.keypointer(frame, frame_idx) if self.keypointer is not None else (None, None)
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
            rows.append(dict(frame=frame_idx, t=t, track_id=int(tid[i]),
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

        # n_inliers / rmse_px / line_score / line_margin / reject_reason: o frame'in FRESH fit adayına ait
        # (reddedilse de yazılır); carry_*: taşınan H'nin bu frame'deki doğrulaması.
        frow: Dict = dict(frame=frame_idx, t=t, h_src=h_src,
                          n_kp=int((np.asarray(kp_conf) >= self.cfg.kp_conf).sum()) if kp_conf is not None else 0,
                          **self.htrack.diag, cut=cut, cut_dist=cut_dist)
        Hm = H.H_img2pitch.ravel() if H is not None else np.full(9, np.nan)
        for k in range(9):
            frow[f"h{k}"] = float(Hm[k])
        return rows, frow

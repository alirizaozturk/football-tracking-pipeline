"""Ultralytics model sarmalayıcıları + tracker.

Sınıflar isimle eşlenir (id ile değil), böylece kendi datasetinde sınıf sırası
farklı olsa da pipeline çalışır.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import supervision as sv

CLASSES = ("ball", "goalkeeper", "player", "referee")
BALL, GOALKEEPER, PLAYER, REFEREE = range(4)

_ALIASES = {
    "football": "ball", "sports ball": "ball", "soccer ball": "ball",
    "keeper": "goalkeeper", "gk": "goalkeeper",
    "players": "player", "person": "player",
    "ref": "referee", "referees": "referee",
}


def canonical_class(name: str) -> Optional[str]:
    n = str(name).lower().strip()
    n = _ALIASES.get(n, n)
    return n if n in CLASSES else None


class YoloDetector:
    """Detection modeli. Çıktı: class_id'leri CLASSES indeksine çevrilmiş sv.Detections."""

    def __init__(self, weights: str, conf: float = 0.3, imgsz: int = 1280,
                 device: Optional[str] = None, only: Optional[Iterable[str]] = None,
                 slice_wh: Optional[int] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        names = self.model.names
        self.lut = np.full(max(names) + 1, -1, dtype=int)
        for k, v in names.items():
            c = canonical_class(v)
            self.lut[k] = CLASSES.index(c) if c else -1
        if (self.lut < 0).all():
            raise ValueError(f"modelin sınıfları tanınmadı: {names}")
        self.only = {CLASSES.index(c) for c in only} if only else None
        self.slicer = None
        if slice_wh:   # küçük top için: görüntüyü parçalara bölüp tahmin (videodaki InferenceSlicer)
            self.slicer = sv.InferenceSlicer(callback=self._raw, slice_wh=slice_wh, overlap_wh=slice_wh // 6)

    def _raw(self, image: np.ndarray) -> sv.Detections:
        r = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(r)

    def __call__(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> sv.Detections:
        d = self.slicer(frame) if self.slicer is not None else self._raw(frame)
        if len(d) == 0:
            return d
        d.class_id = self.lut[d.class_id]
        d = d[d.class_id >= 0]
        if self.only is not None:
            d = d[np.isin(d.class_id, list(self.only))]
        return d


class YoloKeypoints:
    """Saha keypoint (pose) modeli. Çıktı: (xy (K,2), conf (K,)) ya da (None, None)."""

    def __init__(self, weights: str, conf: float = 0.3, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None, None
        xy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(xy.shape[:2], np.float32)
        i = int(np.argmax(kc.mean(axis=1)))        # birden fazla saha bulunursa en güvenlisi
        return xy[i], kc[i]


def make_tracker(fps: float):
    """ByteTrack. Yeni `trackers` paketi varsa onu, yoksa eski supervision sürümünü kullanır."""
    try:
        from trackers import ByteTrackTracker
        t = ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(fps * 2),
                             track_activation_threshold=0.25, high_conf_det_threshold=0.5,
                             minimum_consecutive_frames=1)
        return lambda d: t.update(d)
    except ImportError:
        t = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(fps * 2))
        return lambda d: t.update_with_detections(d)

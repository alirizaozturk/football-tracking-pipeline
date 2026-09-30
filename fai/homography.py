"""Görüntü <-> saha dönüşümü.

Homography saha düzleminden (metre) görüntüye (piksel) RANSAC ile fit edilir,
sonra tersi alınır. Neden bu yön: RANSAC eşiği hedef düzlemin biriminde ölçülür;
piksel cinsinden eşik (ör. 8 px) sezgisel ve ölçekten bağımsızdır. Tersi yönde
metre cinsinden eşik vermek gerekir ve uzak/yakın saha bölgelerinde anlamı değişir.

Önemli sınır: homography sadece ZEMİN düzlemi için doğrudur. Oyuncunun ayak noktası
(bbox alt-orta) doğru projekte edilir; havadaki top edilmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HomographyResult:
    H_img2pitch: np.ndarray        # (3,3)
    H_pitch2img: np.ndarray        # (3,3)
    n_points: int                  # fit'e giren keypoint sayısı
    n_inliers: int
    reproj_rmse_px: float          # inlier'ların piksel cinsinden reprojection hatası


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def fit_homography(img_pts: np.ndarray, pitch_pts: np.ndarray,
                   ransac_px: float = 8.0, min_points: int = 4) -> Optional[HomographyResult]:
    """img_pts / pitch_pts: (N,2) eşleşen noktalar. Yetersiz/dejenere ise None."""
    img_pts = np.asarray(img_pts, np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, np.float64).reshape(-1, 2)
    n = len(img_pts)
    if n < min_points:
        return None
    # 4 nokta varsa RANSAC anlamsız; direkt çöz
    method = cv2.RANSAC if n > 4 else 0
    H_p2i, mask = cv2.findHomography(pitch_pts, img_pts, method, ransac_px)
    if H_p2i is None or not np.isfinite(H_p2i).all() or abs(np.linalg.det(H_p2i)) < 1e-12:
        return None
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(n, bool)
    if inl.sum() < min_points:
        return None
    # Dejenere (neredeyse doğrusal) nokta kümelerini ele: saha tarafında alan kontrolü
    hull = cv2.convexHull(pitch_pts[inl].astype(np.float32))
    if cv2.contourArea(hull) < 20.0:          # m^2; tek çizgi üzerindeki noktalar
        return None
    proj = _apply(H_p2i, pitch_pts[inl])
    rmse = float(np.sqrt(np.mean(np.sum((proj - img_pts[inl]) ** 2, axis=1))))
    H_i2p = np.linalg.inv(H_p2i)
    return HomographyResult(H_i2p / H_i2p[2, 2], H_p2i / H_p2i[2, 2], n, int(inl.sum()), rmse)


def image_to_pitch(res: HomographyResult, pts: np.ndarray) -> np.ndarray:
    return _apply(res.H_img2pitch, pts)


def pitch_to_image(res: HomographyResult, pts: np.ndarray) -> np.ndarray:
    return _apply(res.H_pitch2img, pts)


def is_plausible(res: HomographyResult, img_w: int, img_h: int,
                 pitch_len: float = 105.0, pitch_wid: float = 68.0, margin: float = 25.0) -> bool:
    """Fiziksel akıl kontrolü. 4 noktalı fit her zaman rmse=0 verir, yani rmse tek başına
    kalite ölçüsü değil. Yanlış eşleşmiş keypoint'ler (ör. sol/sağ karışması) genelde
    AYNALANMIŞ ya da ters dönmüş bir zemin üretir; onu burada yakalıyoruz.
      * görüntünün sağı sahada daha büyük x'e, altı daha büyük y'ye (kameraya yakın) düşmeli
      * görüntü merkezi saha (+marj) içinde olmalı
      * yatayda görünen genişlik makul olmalı (2–150 m)
    """
    pts = np.array([[0.25 * img_w, 0.7 * img_h], [0.75 * img_w, 0.7 * img_h],
                    [0.5 * img_w, 0.4 * img_h], [0.5 * img_w, 0.9 * img_h],
                    [0.5 * img_w, 0.6 * img_h]])
    q = image_to_pitch(res, pts)
    if not np.isfinite(q).all():
        return False
    L, R, T, B, C = q
    if not (R[0] > L[0] and B[1] > T[1]):
        return False
    if not (-margin <= C[0] <= pitch_len + margin and -margin <= C[1] <= pitch_wid + margin):
        return False
    return 2.0 <= np.linalg.norm(R - L) <= 150.0


def line_support(frame: np.ndarray, res: HomographyResult, polylines, step_m: float = 0.5,
                 tol_px: int = 2) -> float:
    """Görüntü kanıtı: saha çizgilerini homography ile frame'e projekte et, projekte noktaların
    ne kadarının gerçekten beyaz çizgi pikseline denk geldiğini ölç (0-1).

    Geometrik akıl kontrolü, modelin 'var olmayan bir orta yuvarlak' bulması gibi tutarlı
    ama yanlış keypoint setlerini yakalayamaz; bu skor yakalar. Doğru H'de oyuncular çizgileri
    kısmen örttüğü için 1 olmaz; yanlış H'de ~0'a yakındır."""
    h, w = frame.shape[:2]
    seg = []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:]):
            k = max(2, int(np.hypot(*(b - a)) / step_m))
            seg.append(np.linspace(a, b, k, endpoint=False))
    uv = pitch_to_image(res, np.concatenate(seg))
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= tol_px) & (uv[:, 0] < w - tol_px) & \
        (uv[:, 1] >= tol_px) & (uv[:, 1] < h - tol_px)
    if ok.sum() < 30:
        return float("nan")
    uv = np.round(uv[ok]).astype(int)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] >= 160) & (hsv[..., 1] <= 70)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8))
    return float(white[uv[:, 1], uv[:, 0]].mean())


class HomographyTracker:
    """Frame'ler arası homography yönetimi.

    Kabul zinciri (fresh fit için): >=4 güvenli keypoint -> RANSAC fit -> rmse eşiği ->
    geometrik akıl kontrolü -> görüntüde çizgi desteği (line_score >= min_line_score).
    Bir frame'de geçerli fit yoksa son geçerli homography `max_carry` frame taşınır.
    Kaynak ('fresh' / 'carried' / 'none') ve line_score dataset'e yazılır.
    """

    def __init__(self, vertices: np.ndarray, kp_conf: float = 0.5, ransac_px: float = 8.0,
                 max_carry: int = 12, max_rmse_px: float = 15.0,
                 pitch_size: tuple = (105.0, 68.0), polylines=None, min_line_score: float = 0.15):
        self.vertices = np.asarray(vertices, np.float64)
        self.kp_conf = kp_conf
        self.ransac_px = ransac_px
        self.max_carry = max_carry
        self.max_rmse_px = max_rmse_px
        self.pitch_size = pitch_size
        self.polylines = polylines
        self.min_line_score = min_line_score
        self.rejected = {"rmse": 0, "geometry": 0, "lines": 0}
        self.last_line_score = float("nan")
        self._last: Optional[HomographyResult] = None
        self._age = 0

    def _score(self, frame, res) -> float:
        if frame is None or self.polylines is None or res is None:
            return float("nan")
        return line_support(frame, res, self.polylines)

    def update(self, kp_xy: Optional[np.ndarray], kp_conf: Optional[np.ndarray],
               frame: Optional[np.ndarray] = None):
        res, score = None, float("nan")
        if kp_xy is not None and kp_conf is not None:
            kp_xy = np.asarray(kp_xy, np.float64).reshape(-1, 2)
            keep = (np.asarray(kp_conf).ravel() >= self.kp_conf) & (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)
            if keep.sum() >= 4:
                res = fit_homography(kp_xy[keep], self.vertices[keep], self.ransac_px)
                if res is not None and res.reproj_rmse_px > self.max_rmse_px:
                    res, self.rejected["rmse"] = None, self.rejected["rmse"] + 1
                if res is not None and frame is not None and not is_plausible(
                        res, frame.shape[1], frame.shape[0], *self.pitch_size):
                    res, self.rejected["geometry"] = None, self.rejected["geometry"] + 1
                if res is not None:
                    score = self._score(frame, res)
                    if np.isfinite(score) and score < self.min_line_score:
                        res, self.rejected["lines"] = None, self.rejected["lines"] + 1
        if res is not None:
            self._last, self._age, self.last_line_score = res, 0, score
            return res, "fresh"
        if self._last is not None and self._age < self.max_carry:
            self._age += 1
            self.last_line_score = self._score(frame, self._last)
            return self._last, "carried"
        self.last_line_score = float("nan")
        return None, "none"

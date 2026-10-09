"""Görüntü <-> saha dönüşümü.

Homography saha düzleminden (metre) görüntüye (piksel) RANSAC ile fit edilir,
sonra tersi alınır. Neden bu yön: RANSAC eşiği hedef düzlemin biriminde ölçülür;
piksel cinsinden eşik (ör. 8 px) sezgisel ve ölçekten bağımsızdır. Tersi yönde
metre cinsinden eşik vermek gerekir ve uzak/yakın saha bölgelerinde anlamı değişir.

Önemli sınır: homography sadece ZEMİN düzlemi için doğrudur. Oyuncunun ayak noktası
(bbox alt-orta) doğru projekte edilir; havadaki top edilmez.

Doğrulama (Faz K kalibrasyonu, docs/calibration/field-v12):
  * çizgi maskesi: L kanalında beyaz top-hat (yerel kontrast). Mutlak HSV eşiği gerçek yayında
    çizgi piksellerinin ~2/3'ünü kaçırıyordu.
  * skor: saha çizgilerinden 0.5 m aralıklı örnekler H ile projekte edilir, maskeye düşen oran.
  * marj: skor(H) - max skor(H'nin ±2/4/7 m kaydırılmış "probe"ları). Doğru H yerel bir tepedir;
    çizgi yönünde kaymış (tutarlı-ama-yanlış) H değildir. Mutlak skor bunu ayıramıyordu.
  * piksel eşikleri (RANSAC, rmse, tolerans, çekirdek) frame genişliğiyle ölçeklenir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def grid_error_m(H_img2pitch: np.ndarray, H_pitch2img_ref: np.ndarray, img_w: int, img_h: int,
                 pitch_len: float = 105.0, pitch_wid: float = 68.0, step: float = 1.0) -> Tuple[float, int]:
    """İki homography arasındaki mesafe (m): saha üzerindeki `step` aralıklı ızgaranın REFERANS H ile görüntüye düşen
    noktaları, test H ile sahaya geri projekte edilir; ortalama konum hatası ve kullanılan nokta sayısı.
    (Faz 1a 'doğru H' tanımı: model H'si vs GT H'si.)"""
    xs = np.arange(0.0, pitch_len + 1e-9, step)
    ys = np.arange(0.0, pitch_wid + 1e-9, step)
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    hom = np.c_[g, np.ones(len(g))] @ H_pitch2img_ref.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = hom[:, :2] / hom[:, 2:3]
    w_ok = hom[:, 2] > 0 if np.median(hom[:, 2]) > 0 else hom[:, 2] < 0     # kameranın önündeki noktalar
    ok = w_ok & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    if not ok.any():
        return float("nan"), 0
    q = _apply(H_img2pitch, uv[ok])
    err = np.hypot(*(q - g[ok]).T)
    return float(np.nanmean(err)), int(ok.sum())


# ------------------------------------------------------------------ çizgi kanıtı
@dataclass(frozen=True)
class LineCheck:
    """Çizgi doğrulaması parametreleri. Değerler Faz K (field v12, GT keypoint) kalibrasyonundan; yapı (skor eşiği
    yok, sadece marj) Faz A gürültü çalışmasından: skor eşiği keypoint gürültüsünde doğru H'leri reddediyor, FAR'a
    katkısı yok. Modelin gerçek keypoint'leriyle Faz 1'de yeniden kalibre edilecek (geçici değerler)."""
    tophat_thr: float = 10.0                       # L kanalında top-hat yanıtı eşiği (0-255)
    kernel_div: float = 128.0                      # top-hat çekirdeği = W / kernel_div px (tek sayıya yuvarlanır)
    tol_div: float = 640.0                         # örnek konum toleransı = W / tol_div px
    min_score: Optional[float] = None              # None = skor eşiği yok (geometri + marj); sayı verilirse ek alt sınır
    min_margin: float = 0.12
    probes_m: Tuple[float, ...] = (2.0, 4.0, 7.0)  # x ve y yönünde ±
    step_m: float = 0.5                            # saha çizgisi örnek aralığı
    min_samples: int = 30                          # görüntüye düşen örnek bundan azsa skor NaN (kanıt yok -> red)


def pitch_line_samples(polylines: Sequence[np.ndarray], step_m: float = 0.5) -> np.ndarray:
    seg = []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:]):
            k = max(2, int(np.hypot(*(b - a)) / step_m))
            seg.append(np.linspace(a, b, k, endpoint=False))
    return np.concatenate(seg)


def probe_offsets(probes_m: Sequence[float]) -> np.ndarray:
    """(P,2): önce (0,0), sonra ±x ve ±y probe kaymaları."""
    offs = [(0.0, 0.0)]
    for m in probes_m:
        offs += [(m, 0.0), (-m, 0.0), (0.0, m), (0.0, -m)]
    return np.asarray(offs, float)


def line_mask(frame: np.ndarray, check: LineCheck) -> Tuple[np.ndarray, int]:
    """Frame başına BİR kez: genişletilmiş (dilate) çizgi maskesi ve piksel toleransı."""
    w = frame.shape[1]
    L = cv2.extractChannel(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), 0)    # bitişik kopya (dilim görünümü yavaş)
    k = max(3, int(round(w / check.kernel_div)) | 1)
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    tol = max(1, int(round(w / check.tol_div)))
    # uint8 için th >= thr  <=>  th > ceil(thr) - 1
    _, m = cv2.threshold(th, math.ceil(check.tophat_thr) - 1, 1, cv2.THRESH_BINARY)
    return cv2.dilate(m, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)), tol


def line_evidence(mask: np.ndarray, tol: int, H_pitch2img: np.ndarray, samples: np.ndarray,
                  offsets: np.ndarray, min_samples: int = 30) -> Tuple[float, float]:
    """-> (skor, marj). offsets[0] = (0,0) H'nin kendisi, kalanlar probe'lar. Tüm probe'lar tek
    perspectiveTransform çağrısında projekte edilir. Saha noktası p'yi p - d'nin görüntü yerine koyan
    H, H'nin d kadar kaymış halidir (görüntü->saha koordinatları +d kayar)."""
    h, w = mask.shape
    P, N = len(offsets), len(samples)
    pts = (samples[None, :, :] - offsets[:, None, :]).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(pts, H_pitch2img).reshape(P, N, 2)
    ok = np.isfinite(uv).all(-1) & (uv[..., 0] >= tol) & (uv[..., 0] < w - tol) & \
        (uv[..., 1] >= tol) & (uv[..., 1] < h - tol)
    scores = np.full(P, np.nan)
    for i in range(P):
        if ok[i].sum() >= min_samples:
            q = np.round(uv[i][ok[i]]).astype(int)
            scores[i] = mask[q[:, 1], q[:, 0]].mean()
    s = float(scores[0])
    probes = scores[1:][np.isfinite(scores[1:])]
    margin = s - float(probes.max()) if (np.isfinite(s) and len(probes)) else float("nan")
    return s, margin


def _line_binary(frame: np.ndarray, check: LineCheck) -> np.ndarray:
    """Genişletilmemiş top-hat çizgi maskesi (0/255)."""
    w = frame.shape[1]
    L = cv2.extractChannel(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), 0)
    k = max(3, int(round(w / check.kernel_div)) | 1)
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.threshold(th, math.ceil(check.tophat_thr) - 1, 255, cv2.THRESH_BINARY)[1]


def _render_lines(H_p2i: np.ndarray, polylines, w: int, h: int, step_m: float = 0.5) -> np.ndarray:
    img = np.zeros((h, w), np.float32)
    for pl in polylines:
        dense = []
        for a, b in zip(pl[:-1], pl[1:]):
            k = max(2, int(np.hypot(*(b - a)) / step_m))
            dense.append(np.linspace(a, b, k, endpoint=False))
        dense.append(pl[-1:])
        pts = cv2.perspectiveTransform(np.concatenate(dense).reshape(-1, 1, 2), H_p2i)
        if not np.isfinite(pts).all():
            continue
        cv2.polylines(img, [np.clip(np.round(pts), -10000, 10000).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA)
    return img


def refine_homography(frame: np.ndarray, res: HomographyResult, polylines, check: LineCheck = LineCheck(),
                      search_m: float = 6.0, search_step_m: float = 0.5, ecc: bool = True,
                      ecc_width: int = 960, min_samples: int = 30) -> Tuple[HomographyResult, Dict]:
    """DENEYSEL (Faz 1a yapısal seçenek; pipeline'a bağlı değil). H'yi görüntüdeki çizgilere göre iyileştirir:
      1) öteleme araması: saha düzleminde ±search_m (adım search_step_m) kaydırmalar arasından, projekte çizgi
         örneklerinin çizgi maskesinin distance transform'una göre yumuşak skoru (exp(-d²/2σ²)) en yüksek olan;
      2) ecc=True ise ECC (MOTION_HOMOGRAPHY): çizilen saha çizgisi şablonu ile görüntünün top-hat maskesi
         arasında, ecc_width genişliğe küçültülmüş, Gauss bulanıklaştırılmış görüntülerde.
    -> (iyileştirilmiş H, bilgi). ECC yakınsamazsa öteleme sonucu döner."""
    h, w = frame.shape[:2]
    binm = _line_binary(frame, check)
    dt = cv2.distanceTransform(255 - binm, cv2.DIST_L2, 3)
    sig = max(1.0, w / 480.0)                                  # 1920 px'te 4 px
    samples = pitch_line_samples(polylines, check.step_m)
    r = np.arange(-search_m, search_m + 1e-9, search_step_m)
    offs = np.array([(dx, dy) for dx in r for dy in r])
    P, N = len(offs), len(samples)
    uv = cv2.perspectiveTransform((samples[None] - offs[:, None]).reshape(-1, 1, 2), res.H_pitch2img).reshape(P, N, 2)
    ok = np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    soft = np.full(P, -np.inf)
    for i in range(P):
        if ok[i].sum() >= min_samples:
            q = uv[i][ok[i]].astype(int)
            soft[i] = float(np.exp(-0.5 * (dt[q[:, 1], q[:, 0]] / sig) ** 2).mean())
    best = int(np.argmax(soft))
    d = offs[best]
    T = np.array([[1, 0, -d[0]], [0, 1, -d[1]], [0, 0, 1.0]])
    H1 = res.H_pitch2img @ T
    H1i = np.linalg.inv(H1)
    info = {"shift_m": (float(d[0]), float(d[1])), "soft0": float(soft[np.argmin(np.hypot(*offs.T))]),
            "soft1": float(soft[best]), "ecc": None,
            "H_shift": HomographyResult(H1i / H1i[2, 2], H1 / H1[2, 2], res.n_points, res.n_inliers, res.reproj_rmse_px)}
    H2 = H1
    if ecc:
        s = ecc_width / w
        S = np.diag([s, s, 1.0])
        ws, hs = int(round(w * s)), int(round(h * s))
        inp = cv2.GaussianBlur(cv2.resize(binm, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, (0, 0), 2)
        tmpl = cv2.GaussianBlur(_render_lines(S @ H1, polylines, ws, hs), (0, 0), 2)
        warp = np.eye(3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(tmpl, inp, warp, cv2.MOTION_HOMOGRAPHY,
                                            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6), None, 5)
            H2 = np.linalg.inv(S) @ warp.astype(np.float64) @ S @ H1     # şablon -> görüntü
            info["ecc"] = float(cc)
        except cv2.error:
            info["ecc"] = "yakınsamadı"
    Hi = np.linalg.inv(H2)
    return HomographyResult(Hi / Hi[2, 2], H2 / H2[2, 2], res.n_points, res.n_inliers, res.reproj_rmse_px), info


def judge_lines(score: float, margin: float, check: LineCheck) -> str:
    """'' = kabul; aksi halde red sebebi: 'score' (NaN dahil: doğrulanacak çizgi yok) ya da 'margin'."""
    if not np.isfinite(score) or (check.min_score is not None and score < check.min_score):
        return "score"
    if not np.isfinite(margin) or margin < check.min_margin:
        return "margin"
    return ""


class HomographyTracker:
    """Frame'ler arası homography yönetimi.

    Kabul zinciri (fresh fit için): >=4 güvenli keypoint -> RANSAC fit -> rmse eşiği -> geometrik akıl
    kontrolü -> çizgi kanıtı (skor + marj). Bir frame'de geçerli fit yoksa son geçerli homography en fazla
    `max_carry` frame taşınır; taşınan H de her frame'de aynı çizgi kuralıyla doğrulanır, geçemezse
    (ör. sahne kesmesi) bırakılır. RANSAC ve rmse eşikleri `ref_width` genişliğindeki bir frame içindir,
    gerçek genişlikle ölçeklenir. Frame başına teşhis bilgisi `self.diag`'da (frames.parquet'e yazılır).
    """

    REASONS = ("few_kp", "fit", "rmse", "geometry", "score", "margin")

    def __init__(self, vertices: np.ndarray, kp_conf: float = 0.5, ransac_px: float = 8.0,
                 max_carry: int = 12, max_rmse_px: float = 15.0,
                 pitch_size: tuple = (105.0, 68.0), polylines=None, line_check: Optional[LineCheck] = LineCheck(),
                 ref_width: float = 1280.0):
        self.vertices = np.asarray(vertices, np.float64)
        self.kp_conf = kp_conf
        self.ransac_px = ransac_px
        self.max_carry = max_carry
        self.max_rmse_px = max_rmse_px
        self.pitch_size = pitch_size
        self.check = line_check if polylines is not None else None
        self.ref_width = ref_width
        if self.check is not None:
            self._samples = pitch_line_samples(polylines, self.check.step_m)
            self._offsets = probe_offsets(self.check.probes_m)
        self.rejected: Dict[str, int] = {r: 0 for r in self.REASONS if r != "few_kp"}
        self.rejected.update(carry_score=0, carry_margin=0)
        self.diag: Dict = {}
        self.last_candidate: Optional[HomographyResult] = None    # son frame'in fresh fit adayı (reddedilse de)
        self.last_line_score = float("nan")
        self._last: Optional[HomographyResult] = None
        self._age = 0

    def reset(self) -> None:
        """Taşınan H'yi bırak (ör. sahne kesmesi)."""
        self._last, self._age = None, 0

    def _evidence(self, mask_tol, res) -> Tuple[float, float]:
        mask, tol = mask_tol
        return line_evidence(mask, tol, res.H_pitch2img, self._samples, self._offsets, self.check.min_samples)

    def update(self, kp_xy: Optional[np.ndarray], kp_conf: Optional[np.ndarray],
               frame: Optional[np.ndarray] = None):
        d = dict(reject_reason="few_kp", line_score=np.nan, line_margin=np.nan, n_inliers=0, rmse_px=np.nan,
                 carry_score=np.nan, carry_margin=np.nan, carry_reason="")
        verify = frame is not None and self.check is not None
        mask_tol = None                                   # çizgi maskesi frame başına en fazla bir kez
        scale = frame.shape[1] / self.ref_width if frame is not None else 1.0
        res = None
        self.last_candidate = None
        if kp_xy is not None and kp_conf is not None:
            kp_xy = np.asarray(kp_xy, np.float64).reshape(-1, 2)
            keep = (np.asarray(kp_conf).ravel() >= self.kp_conf) & (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)
            if keep.sum() >= 4:
                res = fit_homography(kp_xy[keep], self.vertices[keep], self.ransac_px * scale)
                self.last_candidate = res
                reason = "fit" if res is None else ""
                if res is not None:
                    d.update(n_inliers=res.n_inliers, rmse_px=res.reproj_rmse_px)
                    if res.reproj_rmse_px > self.max_rmse_px * scale:
                        reason = "rmse"
                    elif frame is not None and not is_plausible(res, frame.shape[1], frame.shape[0], *self.pitch_size):
                        reason = "geometry"
                    elif verify:
                        mask_tol = line_mask(frame, self.check)
                        s, m = self._evidence(mask_tol, res)
                        d.update(line_score=s, line_margin=m)
                        reason = judge_lines(s, m, self.check)
                d["reject_reason"] = reason
                if reason:
                    self.rejected[reason] += 1
                    res = None
        self.last_line_score = d["line_score"]
        if res is not None:
            self._last, self._age = res, 0
            self.diag = d
            return res, "fresh"
        if self._last is not None and self._age < self.max_carry:
            if verify:
                if mask_tol is None:
                    mask_tol = line_mask(frame, self.check)
                s, m = self._evidence(mask_tol, self._last)
                d.update(carry_score=s, carry_margin=m)
                self.last_line_score = s
                why = judge_lines(s, m, self.check)
                if why:                                    # taşınan H bu görüntüyle örtüşmüyor -> bırak
                    d["carry_reason"] = why
                    self.rejected[f"carry_{why}"] += 1
                    self.reset()
                    self.diag = d
                    return None, "none"
            self._age += 1
            self.diag = d
            return self._last, "carried"
        if self._last is not None:
            d["carry_reason"] = "age"
            self.reset()
        self.diag = d
        return None, "none"

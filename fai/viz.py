"""Çizim: kuşbakışı radar, Voronoi saha kontrolü, ısı haritası, yayın görüntüsü anotasyonu.
Hepsi düz OpenCV; supervision annotator API'si sürümden sürüme değiştiği için bağımlı değil."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .pitch import PitchConfig

# BGR
TEAM_COLORS = {0: (255, 191, 0), 1: (147, 20, 255), -1: (0, 215, 255)}   # mavi, pembe, hakem sarı
UNKNOWN = (160, 160, 160)
BALL_COLOR = (255, 255, 255)
PITCH_GREEN = (58, 125, 60)
LINE_WHITE = (235, 235, 235)


class Radar:
    """Saha çizimi ve metre -> radar pikseli dönüşümü."""

    def __init__(self, pitch: PitchConfig = PitchConfig(), scale: float = 8.0, pad: int = 30):
        self.pitch, self.scale, self.pad = pitch, scale, pad
        self.w = int(round(pitch.length * scale)) + 2 * pad
        self.h = int(round(pitch.width * scale)) + 2 * pad
        self.base = self._draw_base()

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float).reshape(-1, 2) * self.scale + self.pad

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), PITCH_GREEN, np.uint8)
        # çim şeritleri
        n = 12
        for i in range(0, n, 2):
            x0 = int(self.pad + i * self.pitch.length / n * self.scale)
            x1 = int(self.pad + (i + 1) * self.pitch.length / n * self.scale)
            img[self.pad:self.h - self.pad, x0:x1] = (54, 117, 56)
        th = max(1, int(round(self.scale / 4)))
        for pl in self.pitch.polylines():
            cv2.polylines(img, [np.round(self.to_px(pl)).astype(np.int32)], False, LINE_WHITE, th, cv2.LINE_AA)
        for s in self.to_px(self.pitch.spots()):
            cv2.circle(img, tuple(np.round(s).astype(int)), th + 1, LINE_WHITE, -1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ layers
    def voronoi(self, img: np.ndarray, xy0: np.ndarray, xy1: np.ndarray,
                alpha: float = 0.55, steepness: float = 0.6, res: float = 2.0) -> np.ndarray:
        """Saha kontrolü: her nokta en yakın oyuncunun takımına boyanır; sınırda yumuşak geçiş.
        res: hesaplama çözünürlüğü (piksel / metre), sonra radar boyutuna büyütülür."""
        xy0 = np.asarray(xy0, float).reshape(-1, 2); xy0 = xy0[np.isfinite(xy0).all(1)]
        xy1 = np.asarray(xy1, float).reshape(-1, 2); xy1 = xy1[np.isfinite(xy1).all(1)]
        if len(xy0) == 0 or len(xy1) == 0:
            return img
        gw, gh = int(self.pitch.length * res), int(self.pitch.width * res)
        gx, gy = np.meshgrid((np.arange(gw) + 0.5) / res, (np.arange(gh) + 0.5) / res)
        g = np.stack([gx, gy], -1)[..., None, :]                         # (gh, gw, 1, 2)
        d0 = np.linalg.norm(g - xy0[None, None], axis=-1).min(-1)
        d1 = np.linalg.norm(g - xy1[None, None], axis=-1).min(-1)
        w0 = (np.tanh((d1 - d0) * steepness) + 1) / 2                   # 1 => takım 0 kontrolünde
        c0, c1 = np.array(TEAM_COLORS[0], float), np.array(TEAM_COLORS[1], float)
        col = (w0[..., None] * c0 + (1 - w0[..., None]) * c1).astype(np.uint8)
        pw, ph = int(round(self.pitch.length * self.scale)), int(round(self.pitch.width * self.scale))
        col = cv2.resize(col, (pw, ph), interpolation=cv2.INTER_LINEAR)
        out = img.copy()
        roi = out[self.pad:self.pad + ph, self.pad:self.pad + pw]
        out[self.pad:self.pad + ph, self.pad:self.pad + pw] = cv2.addWeighted(roi, 1 - alpha, col, alpha, 0)
        # çizgileri tekrar üstte göster
        lines = cv2.inRange(self.base, np.array(LINE_WHITE) - 1, np.array(LINE_WHITE) + 1) > 0
        out[lines] = LINE_WHITE
        return out

    def heatmap(self, xy: np.ndarray, sigma_m: float = 2.5, alpha: float = 0.65) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(1)]
        img = self.base.copy()
        if len(xy) == 0:
            return img
        pw, ph = int(round(self.pitch.length * self.scale)), int(round(self.pitch.width * self.scale))
        H, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(ph, pw),
                                 range=[[0, self.pitch.width], [0, self.pitch.length]])
        H = cv2.GaussianBlur(H.astype(np.float32), (0, 0), sigma_m * self.scale)
        if H.max() <= 0:
            return img
        Hn = (H / H.max() * 255).astype(np.uint8)
        cm = cv2.applyColorMap(Hn, cv2.COLORMAP_INFERNO)
        a = (Hn.astype(np.float32) / 255.0 * alpha)[..., None]
        roi = img[self.pad:self.pad + ph, self.pad:self.pad + pw].astype(np.float32)
        img[self.pad:self.pad + ph, self.pad:self.pad + pw] = (roi * (1 - a) + cm * a).astype(np.uint8)
        return img

    def points(self, img: np.ndarray, xy: np.ndarray, color, radius: Optional[int] = None,
               labels: Optional[Sequence[str]] = None) -> np.ndarray:
        r = radius or max(3, int(self.scale * 0.9))
        for i, p in enumerate(self.to_px(xy)):
            if not np.isfinite(p).all():
                continue
            c = tuple(int(v) for v in np.round(p))
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (20, 20, 20), 1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (c[0] - r, c[1] - r - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3 * self.scale / 8 + 0.1, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def draw_frame(self, objs: Dict[str, np.ndarray], voronoi: bool = False,
                   labels: Optional[Dict[int, Sequence[str]]] = None) -> np.ndarray:
        """objs: {'team0': (N,2), 'team1': (N,2), 'referee': (N,2), 'ball': (1,2)} metre."""
        img = self.base.copy()
        if voronoi:
            img = self.voronoi(img, objs.get("team0", np.zeros((0, 2))), objs.get("team1", np.zeros((0, 2))))
        labels = labels or {}
        img = self.points(img, objs.get("team0", np.zeros((0, 2))), TEAM_COLORS[0], labels=labels.get(0))
        img = self.points(img, objs.get("team1", np.zeros((0, 2))), TEAM_COLORS[1], labels=labels.get(1))
        img = self.points(img, objs.get("referee", np.zeros((0, 2))), TEAM_COLORS[-1])
        img = self.points(img, objs.get("ball", np.zeros((0, 2))), BALL_COLOR, radius=max(2, int(self.scale * 0.6)))
        return img


# ---------------------------------------------------------------- broadcast view
def annotate_broadcast(frame: np.ndarray, rows, show_ids: bool = True) -> np.ndarray:
    """rows: ilgili frame'e ait DataFrame satırları (x1..y2, u, v, cls, role, team, track_id)."""
    out = frame.copy()
    for r in rows.itertuples(index=False):
        if r.cls == "ball":
            cx, top = int(r.u), int(r.y1)
            tri = np.array([[cx, top - 4], [cx - 8, top - 18], [cx + 8, top - 18]], np.int32)
            cv2.fillPoly(out, [tri], (0, 215, 255))
            cv2.polylines(out, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
            continue
        color = TEAM_COLORS.get(int(r.team), UNKNOWN) if r.role != "referee" else TEAM_COLORS[-1]
        w = max(8, int((r.x2 - r.x1) / 2))
        cv2.ellipse(out, (int(r.u), int(r.v)), (w, max(3, int(0.35 * w))), 0, -45, 235, color, 2, cv2.LINE_AA)
        if show_ids and r.track_id >= 0:
            txt = str(int(r.track_id))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x0, y0 = int(r.u - tw / 2 - 3), int(r.v + 8)
            cv2.rectangle(out, (x0, y0), (x0 + tw + 6, y0 + th + 6), color, -1)
            cv2.putText(out, txt, (x0 + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def overlay_inset(frame: np.ndarray, inset: np.ndarray, width_frac: float = 0.42,
                  alpha: float = 0.85, margin: int = 16) -> np.ndarray:
    """Radarı yayın görüntüsünün alt ortasına yerleştir."""
    H, W = frame.shape[:2]
    iw = int(W * width_frac)
    ih = int(inset.shape[0] * iw / inset.shape[1])
    small = cv2.resize(inset, (iw, ih), interpolation=cv2.INTER_AREA)
    x0, y0 = (W - iw) // 2, H - ih - margin
    out = frame.copy()
    roi = out[y0:y0 + ih, x0:x0 + iw]
    out[y0:y0 + ih, x0:x0 + iw] = cv2.addWeighted(roi, 1 - alpha, small, alpha, 0)
    return out


def project_pitch_lines(frame: np.ndarray, H_pitch2img: np.ndarray, pitch: PitchConfig,
                        color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Saha çizgilerini homography ile görüntüye geri projekte et (doğrulama için)."""
    out = frame.copy()
    for pl in pitch.polylines():
        dense = []
        for a, b in zip(pl[:-1], pl[1:]):            # uzun doğruları sıkılaştır (perspektif eğrisi değil ama kırpma için)
            k = max(2, int(np.hypot(*(b - a)) / 0.5))
            dense.append(np.linspace(a, b, k, endpoint=False))
        dense.append(pl[-1:])
        pts = cv2.perspectiveTransform(np.concatenate(dense).reshape(-1, 1, 2).astype(np.float64), H_pitch2img)
        pts = np.clip(np.round(pts), -20000, 20000).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
    return out

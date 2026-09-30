"""Saha geometrisi: 32 keypoint'in saha koordinatları (metre) ve çizgi polylineları.

Koordinat sistemi (kuşbakışı, yayın kamerasının karşısından bakıldığında):
    x: 0 -> length   (sol kale çizgisi -> sağ kale çizgisi)
    y: 0 -> width    (yayın kamerasından uzak taç çizgisi -> yakın taç çizgisi)

Keypoint sırası Roboflow `football-field-detection` datasetinin (ve roboflow/sports
reposunun) sırasını takip eder. Sıra yanlışsa homography sessizce bozulur; bu yüzden
modeli eğittikten sonra `scripts/03_check_keypoints.py` ile görsel kontrol ŞART.

Not: Videodaki referans implementasyon sahayı 120x70 m alıyor. Hız/mesafe
hesabı için gerçek ölçüler önemli (Bundesliga / Süper Lig sahaları genelde 105x68),
o yüzden varsayılan 105x68 ve ceza sahası ölçüleri FIFA standardı.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    length: float = 105.0            # m, kale çizgileri arası
    width: float = 68.0              # m, taç çizgileri arası
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    # ------------------------------------------------------------------ keypoints
    @property
    def vertices(self) -> np.ndarray:
        """(32, 2) metre cinsinden keypoint koordinatları. Index 0 == keypoint #1."""
        L, W = self.length, self.width
        pbl, pbw = self.penalty_box_length, self.penalty_box_width
        gbl, gbw = self.goal_box_length, self.goal_box_width
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        pts = [
            (0, 0),                                  # 1  sol üst köşe
            (0, (W - pbw) / 2),                      # 2  sol ceza sahası üst - kale çizgisi
            (0, (W - gbw) / 2),                      # 3  sol kale alanı üst - kale çizgisi
            (0, (W + gbw) / 2),                      # 4
            (0, (W + pbw) / 2),                      # 5
            (0, W),                                  # 6  sol alt köşe
            (gbl, (W - gbw) / 2),                    # 7  sol kale alanı iç köşe üst
            (gbl, (W + gbw) / 2),                    # 8
            (ps, W / 2),                             # 9  sol penaltı noktası
            (pbl, (W - pbw) / 2),                    # 10 sol ceza sahası iç köşe üst
            (pbl, (W - gbw) / 2),                    # 11
            (pbl, (W + gbw) / 2),                    # 12
            (pbl, (W + pbw) / 2),                    # 13 sol ceza sahası iç köşe alt
            (L / 2, 0),                              # 14 orta çizgi üst
            (L / 2, W / 2 - r),                      # 15 orta yuvarlak üst
            (L / 2, W / 2 + r),                      # 16 orta yuvarlak alt
            (L / 2, W),                              # 17 orta çizgi alt
            (L - pbl, (W - pbw) / 2),                # 18 sağ ceza sahası iç köşe üst
            (L - pbl, (W - gbw) / 2),                # 19
            (L - pbl, (W + gbw) / 2),                # 20
            (L - pbl, (W + pbw) / 2),                # 21
            (L - ps, W / 2),                         # 22 sağ penaltı noktası
            (L - gbl, (W - gbw) / 2),                # 23
            (L - gbl, (W + gbw) / 2),                # 24
            (L, 0),                                  # 25 sağ üst köşe
            (L, (W - pbw) / 2),                      # 26
            (L, (W - gbw) / 2),                      # 27
            (L, (W + gbw) / 2),                      # 28
            (L, (W + pbw) / 2),                      # 29
            (L, W),                                  # 30 sağ alt köşe
            (L / 2 - r, W / 2),                      # 31 orta yuvarlak sol
            (L / 2 + r, W / 2),                      # 32 orta yuvarlak sağ
        ]
        return np.asarray(pts, dtype=np.float64)

    # ---------------------------------------------------------------- line model
    def polylines(self, arc_steps: int = 64) -> List[np.ndarray]:
        """Saha çizgilerini (N,2) metre polylineları olarak döndürür.
        Hem radar çizimi hem de homography doğrulaması (çizgileri görüntüye
        geri projekte etme) için kullanılır."""
        L, W = self.length, self.width
        pbl, pbw = self.penalty_box_length, self.penalty_box_width
        gbl, gbw = self.goal_box_length, self.goal_box_width
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        cy = W / 2
        lines: List[np.ndarray] = []

        def rect(x0, y0, x1, y1):
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)

        lines.append(rect(0, 0, L, W))                                   # dış çizgi
        lines.append(np.array([[L / 2, 0], [L / 2, W]], float))          # orta çizgi
        # sol/sağ ceza sahası ve kale alanı
        lines.append(np.array([[0, cy - pbw / 2], [pbl, cy - pbw / 2], [pbl, cy + pbw / 2], [0, cy + pbw / 2]], float))
        lines.append(np.array([[L, cy - pbw / 2], [L - pbl, cy - pbw / 2], [L - pbl, cy + pbw / 2], [L, cy + pbw / 2]], float))
        lines.append(np.array([[0, cy - gbw / 2], [gbl, cy - gbw / 2], [gbl, cy + gbw / 2], [0, cy + gbw / 2]], float))
        lines.append(np.array([[L, cy - gbw / 2], [L - gbl, cy - gbw / 2], [L - gbl, cy + gbw / 2], [L, cy + gbw / 2]], float))
        # orta yuvarlak
        t = np.linspace(0, 2 * np.pi, arc_steps * 2)
        lines.append(np.c_[L / 2 + r * np.cos(t), cy + r * np.sin(t)])
        # ceza yayları (penaltı noktası merkezli, ceza sahası dışında kalan kısım)
        a = np.arccos((pbl - ps) / r)
        t = np.linspace(-a, a, arc_steps)
        lines.append(np.c_[ps + r * np.cos(t), cy + r * np.sin(t)])
        lines.append(np.c_[L - ps - r * np.cos(t), cy + r * np.sin(t)])
        return lines

    def spots(self) -> np.ndarray:
        """Penaltı noktaları ve santra noktası (3,2)."""
        return np.array([[self.penalty_spot_distance, self.width / 2],
                         [self.length - self.penalty_spot_distance, self.width / 2],
                         [self.length / 2, self.width / 2]], float)

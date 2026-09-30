"""Takım ayrımı (unsupervised).

İki backend:
  * siglip : Videodaki yöntem. SigLIP görüntü embedding'i -> UMAP(3D) -> KMeans(2).
             Forma rengi/desen/logo gibi her şeyi yakalar, GPU ister.
  * color  : Hafif alternatif. Gövde bölgesinin (çim pikselleri maskelenmiş) medyan
             Lab rengi -> KMeans(2). GPU istemez, benzer renkli formalarda zayıflar.

Her iki backend de videonun kendi oyuncu kırpıntılarıyla o maça özel fit edilir
(önceden etiket gerekmez). Etiketler (0/1) keyfidir; hangi takımın 0 olduğu
maçtan maça değişir.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans


# --------------------------------------------------------------------- features
def color_features(crop: np.ndarray) -> np.ndarray:
    """BGR crop -> [L, a, b] (gövde bölgesi, çim maskeli, medyan)."""
    h, w = crop.shape[:2]
    roi = crop[int(0.15 * h):max(int(0.55 * h), int(0.15 * h) + 1),
               int(0.20 * w):max(int(0.80 * w), int(0.20 * w) + 1)]
    if roi.size == 0:
        roi = crop
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = ~grass.ravel()
    px = lab[keep] if keep.sum() >= 20 else lab   # yeşil forma durumunda maske her şeyi silmesin
    return np.median(px, axis=0)


class SiglipEmbedder:
    def __init__(self, model_id: str = "google/siglip-base-patch16-224",
                 device: Optional[str] = None, batch_size: int = 32):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.batch_size = batch_size

    def __call__(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(rgb), self.batch_size):
                inp = self.processor(images=rgb[i:i + self.batch_size], return_tensors="pt").to(self.device)
                hs = self.model(**inp).last_hidden_state          # (B, tokens, D)
                out.append(hs.mean(dim=1).float().cpu().numpy())  # token ortalaması
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


# ------------------------------------------------------------------- classifier
class TeamClassifier:
    def __init__(self, method: str = "siglip", model_id: str = "google/siglip-base-patch16-224",
                 device: Optional[str] = None, seed: int = 0):
        if method not in ("siglip", "color"):
            raise ValueError(f"unknown team method: {method}")
        self.method, self.model_id, self.device, self.seed = method, model_id, device, seed
        self.reducer = None
        self.kmeans: Optional[KMeans] = None
        self._embedder: Optional[SiglipEmbedder] = None

    # -- internals
    def _features(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        crops = [c for c in crops]
        if len(crops) == 0:
            return np.zeros((0, 3), np.float32)
        if self.method == "color":
            return np.stack([color_features(c) for c in crops])
        if self._embedder is None:
            self._embedder = SiglipEmbedder(self.model_id, self.device)
        return self._embedder(crops)

    # -- API
    def fit(self, crops: Sequence[np.ndarray]) -> "TeamClassifier":
        if len(crops) < 10:
            raise ValueError(f"team classifier needs >=10 player crops, got {len(crops)}")
        X = self._features(crops)
        if self.method == "siglip":
            import umap
            self.reducer = umap.UMAP(n_components=3, random_state=self.seed)
            X = self.reducer.fit_transform(X)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(X)
        return self

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros(0, dtype=int)
        if self.kmeans is None:
            raise RuntimeError("TeamClassifier.fit() önce çağrılmalı")
        X = self._features(crops)
        if self.reducer is not None:
            X = self.reducer.transform(X)
        return self.kmeans.predict(X).astype(int)

    def save(self, path: str | Path) -> None:
        state = {"method": self.method, "model_id": self.model_id, "seed": self.seed,
                 "reducer": self.reducer, "kmeans": self.kmeans}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "TeamClassifier":
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls(s["method"], s["model_id"], device, s["seed"])
        obj.reducer, obj.kmeans = s["reducer"], s["kmeans"]
        return obj


# ------------------------------------------------------------------ goalkeepers
def assign_goalkeepers(gk_xy: np.ndarray, player_xy: np.ndarray, player_team: np.ndarray) -> np.ndarray:
    """Kaleciyi, oyuncu merkezlerinden (centroid) hangisine yakınsa o takıma ata.
    Koordinatlar saha (m) ya da piksel olabilir; ikisi de aynı uzayda olmalı."""
    gk_xy = np.asarray(gk_xy, float).reshape(-1, 2)
    out = np.full(len(gk_xy), -1, dtype=int)
    if len(gk_xy) == 0:
        return out
    player_xy = np.asarray(player_xy, float).reshape(-1, 2)
    ok = np.isfinite(player_xy).all(1)
    cents: List[np.ndarray] = []
    for t in (0, 1):
        m = ok & (player_team == t)
        if m.sum() == 0:
            return out
        cents.append(player_xy[m].mean(0))
    d0 = np.linalg.norm(gk_xy - cents[0], axis=1)
    d1 = np.linalg.norm(gk_xy - cents[1], axis=1)
    out = (d1 < d0).astype(int)
    out[~np.isfinite(gk_xy).all(1)] = -1
    return out

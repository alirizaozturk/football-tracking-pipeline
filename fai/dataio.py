"""Dataset dosya işleri: tracking datasını yaz/oku, YOLO etiketleri, data.yaml düzeltme."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------------- tracking dataset
def save_table(df: pd.DataFrame, path_no_ext: Path, fmt: str = "parquet") -> Path:
    path_no_ext = Path(path_no_ext)
    if fmt == "parquet":
        try:
            p = path_no_ext.with_suffix(".parquet")
            df.to_parquet(p, index=False)
            return p
        except ImportError:
            print("[uyarı] pyarrow yok, CSV'ye düşülüyor")
    p = path_no_ext.with_suffix(".csv")
    df.to_csv(p, index=False)
    return p


def load_table(path_no_ext: Path) -> pd.DataFrame:
    path_no_ext = Path(path_no_ext)
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = path_no_ext.with_suffix(ext)
        if p.exists():
            return fn(p)
    raise FileNotFoundError(f"{path_no_ext}.parquet/.csv bulunamadı")


def load_match(match_dir: str | Path):
    """-> (tracking_df, frames_df, meta_dict)"""
    d = Path(match_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return load_table(d / "tracking"), load_table(d / "frames"), meta


def homography_of(frames_row) -> Optional[np.ndarray]:
    H = np.array([frames_row[f"h{k}"] for k in range(9)], float).reshape(3, 3)
    return None if not np.isfinite(H).all() else H


# ------------------------------------------------------------------ YOLO format
def yolo_det_lines(xyxy: np.ndarray, class_ids: Sequence[int], img_w: int, img_h: int) -> List[str]:
    lines = []
    for (x1, y1, x2, y2), c in zip(np.asarray(xyxy).reshape(-1, 4), class_ids):
        x1, x2 = np.clip([x1, x2], 0, img_w)
        y1, y2 = np.clip([y1, y2], 0, img_h)
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            continue
        lines.append(f"{int(c)} {(x1 + w / 2) / img_w:.6f} {(y1 + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


def yolo_pose_line(xyxy: np.ndarray, kp_xy: np.ndarray, kp_vis: np.ndarray,
                   img_w: int, img_h: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = np.asarray(xyxy, float).ravel()
    x1, x2 = np.clip([x1, x2], 0, img_w)
    y1, y2 = np.clip([y1, y2], 0, img_h)
    w, h = x2 - x1, y2 - y1
    parts = [str(class_id), f"{(x1 + w / 2) / img_w:.6f}", f"{(y1 + h / 2) / img_h:.6f}",
             f"{w / img_w:.6f}", f"{h / img_h:.6f}"]
    for (x, y), v in zip(np.asarray(kp_xy).reshape(-1, 2), np.asarray(kp_vis).ravel()):
        if v > 0 and 0 <= x < img_w and 0 <= y < img_h:
            parts += [f"{x / img_w:.6f}", f"{y / img_h:.6f}", "2"]
        else:
            parts += ["0", "0", "0"]
    return " ".join(parts)


# -------------------------------------------------------------------- data.yaml
def fix_roboflow_yaml(dataset_dir: str | Path) -> Path:
    """Roboflow export'larındaki '../train/images' gibi göreli yolları mutlak `path` +
    split klasörlerine çevirir. (Klasik 'images not found' hatasının sebebi.)"""
    import yaml
    d = Path(dataset_dir).resolve()
    p = d / "data.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["path"] = str(d)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (d / folder / "images").exists():
            cfg[split] = f"{folder}/images"
        else:
            cfg.pop(split, None)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def combine_yamls(yamls: Iterable[str | Path], out: str | Path) -> Path:
    """Birden fazla YOLO datasetini (ör. Roboflow + kendi auto-label datan) tek eğitimde
    birleştirir. Sınıf isimleri ve sıraları aynı olmalı."""
    import yaml
    cfgs = [(Path(y).resolve(), yaml.safe_load(Path(y).read_text(encoding="utf-8"))) for y in yamls]
    names0 = _names(cfgs[0][1])
    for y, c in cfgs[1:]:
        if _names(c) != names0:
            raise ValueError(f"sınıflar uyuşmuyor: {y}: {_names(c)} != {names0}")

    def absl(y, c, split):
        v = c.get(split)
        if v is None:
            return []
        root = Path(c.get("path", y.parent))
        root = root if root.is_absolute() else (y.parent / root)
        return [str((root / s).resolve()) for s in (v if isinstance(v, list) else [v])]

    merged = {"train": sum((absl(y, c, "train") for y, c in cfgs), []),
              "val": sum((absl(y, c, "val") for y, c in cfgs), []),
              "names": {i: n for i, n in enumerate(names0)}}
    kpt = cfgs[0][1].get("kpt_shape")
    if kpt:
        merged["kpt_shape"] = kpt
    flips = [c.get("flip_idx") for _, c in cfgs]
    if flips[0] is not None and all(f == flips[0] for f in flips if f is not None):
        merged["flip_idx"] = flips[0]                 # yoksa ultralytics yatay flip'i kapatır
    out = Path(out)
    out.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def _names(cfg: Dict) -> List[str]:
    n = cfg["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)

"""FAZ 2a — Kendi maç videolarından eğitim için frame çıkar.

    python scripts/04_extract_frames.py --videos "videos/*.mp4" --every-s 1.0

Filtreler:
  * yakın çekim / tribün / reklam: çim oranı --min-green altındaysa atlanır
  * ardışık neredeyse aynı kareler: küçük gri thumbnail farkı --dedupe altındaysa atlanır
Çıktı: data/own/frames/<video>/<video>_f000123.jpg + manifest.csv
"""
import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np


def green_ratio(img: np.ndarray) -> float:
    h = img.shape[0]
    hsv = cv2.cvtColor(img[h // 3:], cv2.COLOR_BGR2HSV)       # üst 1/3 genelde tribün
    m = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="dosya ya da glob")
    ap.add_argument("--out", default="data/own/frames")
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--min-green", type=float, default=0.35)
    ap.add_argument("--dedupe", type=float, default=6.0, help="ortalama piksel farkı eşiği (0-255)")
    ap.add_argument("--max-per-video", type=int, default=400)
    a = ap.parse_args()

    paths = sorted({p for g in a.videos for p in glob.glob(g)})
    if not paths:
        raise SystemExit("video bulunamadı")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for vp in paths:
        stem = Path(vp).stem
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps * a.every_s)))
        (out / stem).mkdir(exist_ok=True)
        prev, saved, f, skipped = None, 0, 0, {"green": 0, "dup": 0}
        while saved < a.max_per_video:
            if not cap.grab():
                break
            if f % step == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                g = green_ratio(img)
                thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
                if g < a.min_green:
                    skipped["green"] += 1
                elif prev is not None and np.abs(thumb - prev).mean() < a.dedupe:
                    skipped["dup"] += 1
                else:
                    p = out / stem / f"{stem}_f{f:06d}.jpg"
                    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append(dict(video=stem, frame=f, path=str(p), green=round(g, 3)))
                    prev, saved = thumb, saved + 1
            f += 1
        print(f"{stem}: {saved} frame kaydedildi (atlanan: çim={skipped['green']}, benzer={skipped['dup']})")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame", "path", "green"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] toplam {len(rows)} frame -> {out}")


if __name__ == "__main__":
    main()

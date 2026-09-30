"""FAZ 2b — Çıkarılan frame'leri eğitilmiş modellerle otomatik etiketle (auto-label).

    python scripts/05_autolabel.py --frames data/own/frames \
        --det runs/det/weights/best.pt --kp runs/pose/weights/best.pt --previews

Çıktı (YOLO formatı, ultralytics'e direkt verilebilir):
    data/own/det/{train,val}/{images,labels} + data.yaml    (ball/goalkeeper/player/referee)
    data/own/pose/{train,val}/{images,labels} + data.yaml   (32 saha keypoint'i)
    data/own/review.csv   -> elle kontrol edilmesi gereken frame'ler ve sebepleri
    data/own/review/      -> (--previews) kutuları çizilmiş önizlemeler

Neden review: auto-label modelin kendi hatalarını datasete geri yazar. Belirsiz
tahminleri (conf bandı, anormal oyuncu sayısı, birden fazla top, az keypoint) işaretleyip
CVAT / Roboflow'da düzeltmek, datasetin gerçekten modelden iyi olmasını sağlar.

Train/val ayrımı VİDEO bazında (tek video varsa zaman bloğu bazında): komşu frame'ler
neredeyse aynı olduğu için rastgele ayrım val skorunu şişirir (data leakage).
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.dataio import yolo_det_lines, yolo_pose_line  # noqa: E402
from fai.models import BALL, CLASSES, PLAYER, YoloDetector, YoloKeypoints  # noqa: E402


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_of(rows, val_frac):
    vids = sorted({r["video"] for r in rows})
    if len(vids) >= 2:
        n_val = max(1, int(round(len(vids) * val_frac)))
        val = set(vids[-n_val:])
        return {r["path"]: ("val" if r["video"] in val else "train") for r in rows}
    frames = sorted(rows, key=lambda r: int(r["frame"]))     # tek video: son %val_frac zaman bloğu
    cut = int(len(frames) * (1 - val_frac))
    return {r["path"]: ("train" if i < cut else "val") for i, r in enumerate(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/own/frames")
    ap.add_argument("--det", required=True)
    ap.add_argument("--kp", default=None)
    ap.add_argument("--out", default="data/own")
    ap.add_argument("--conf", type=float, default=0.5, help="bu güvenin üstü etikete yazılır")
    ap.add_argument("--review-low", type=float, default=0.25, help="[low, conf) arası tahmin -> review")
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--previews", action="store_true")
    a = ap.parse_args()

    manifest = Path(a.frames) / "manifest.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    split = split_of(rows, a.val_frac)
    out = Path(a.out)
    det = YoloDetector(a.det, conf=a.review_low, imgsz=a.imgsz, device=a.device)
    kp = YoloKeypoints(a.kp, conf=0.25, device=a.device) if a.kp else None
    review, n_det, n_kp = [], 0, 0

    for r in rows:
        src = Path(r["path"])
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        sp = split[r["path"]]
        reasons = []

        d = det(img)
        sure = d[d.confidence >= a.conf]
        unsure = d[d.confidence < a.conf]
        if len(unsure):
            reasons.append(f"{len(unsure)} belirsiz kutu")
        n_pl = int((sure.class_id == PLAYER).sum())
        if not 8 <= n_pl <= 26:
            reasons.append(f"oyuncu sayısı {n_pl}")
        if int((sure.class_id == BALL).sum()) > 1:
            reasons.append("birden fazla top")
        lines = yolo_det_lines(sure.xyxy, sure.class_id, w, h)
        link_or_copy(src, out / "det" / sp / "images" / src.name)
        (out / "det" / sp / "labels").mkdir(parents=True, exist_ok=True)
        (out / "det" / sp / "labels" / f"{src.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_det += 1

        kxy = None
        if kp is not None:
            res = kp.model.predict(img, conf=0.25, imgsz=640, device=a.device, verbose=False)[0]
            if res.keypoints is not None and len(res.keypoints):
                i = int(np.argmax(res.keypoints.conf.cpu().numpy().mean(1)))
                kxy = res.keypoints.xy.cpu().numpy()[i]
                kc = res.keypoints.conf.cpu().numpy()[i]
                box = res.boxes.xyxy.cpu().numpy()[i]
                vis = kc >= a.kp_conf
                if vis.sum() >= 4:
                    link_or_copy(src, out / "pose" / sp / "images" / src.name)
                    (out / "pose" / sp / "labels").mkdir(parents=True, exist_ok=True)
                    (out / "pose" / sp / "labels" / f"{src.stem}.txt").write_text(
                        yolo_pose_line(box, kxy, vis, w, h), encoding="utf-8")
                    n_kp += 1
                else:
                    reasons.append(f"az keypoint ({int(vis.sum())})")
            else:
                reasons.append("saha bulunamadı")

        if reasons:
            review.append(dict(image=str(src), split=sp, reasons="; ".join(reasons)))
            if a.previews:
                pv = img.copy()
                for (x1, y1, x2, y2), c, cf in zip(d.xyxy.astype(int), d.class_id, d.confidence):
                    col = (0, 200, 0) if cf >= a.conf else (0, 0, 255)
                    cv2.rectangle(pv, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(pv, f"{CLASSES[c]} {cf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                (out / "review").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "review" / src.name), pv)

    # data.yaml'lar
    det_yaml = {"path": str((out / "det").resolve()), "train": "train/images", "val": "val/images",
                "names": {i: n for i, n in enumerate(CLASSES)}}
    (out / "det").mkdir(parents=True, exist_ok=True)
    (out / "det" / "data.yaml").write_text(yaml.safe_dump(det_yaml, sort_keys=False), encoding="utf-8")
    if kp is not None and n_kp:
        pose_yaml = {"path": str((out / "pose").resolve()), "train": "train/images", "val": "val/images",
                     "kpt_shape": [32, 3], "names": dict(kp.model.names)}
        (out / "pose" / "data.yaml").write_text(yaml.safe_dump(pose_yaml, sort_keys=False), encoding="utf-8")
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "split", "reasons"])
        wr.writeheader()
        wr.writerows(review)
    print(f"[ok] det: {n_det} görüntü, pose: {n_kp} görüntü, review: {len(review)} -> {out}")


if __name__ == "__main__":
    main()

"""FAZ 1b — Modelleri eğit (ultralytics YOLO11 / YOLOv8).

    # oyuncu/top/hakem/kaleci detection
    python scripts/02_train.py det  --data data/roboflow/players-v*/data.yaml
    # saha keypoint (pose)
    python scripts/02_train.py pose --data data/roboflow/field-v*/data.yaml
    # sadece top (opsiyonel, küçük obje)
    python scripts/02_train.py ball --data data/roboflow/ball-v*/data.yaml

Birden fazla --data verirsen (ör. Roboflow + kendi auto-label datan) tek eğitimde
birleştirilir; sınıf isimleri/sıraları aynı olmalı.

Varsayılanlar RTX 4060 Laptop (8 GB) için: batch=-1 ultralytics AutoBatch ile
VRAM'in ~%60'ını kullanacak batch'i kendi seçer.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.dataio import combine_yamls  # noqa: E402

DEFAULTS = {
    #        model               imgsz epochs  ekstra argümanlar
    "det":  ("yolo11m.pt",       1280, 100, {}),
    "ball": ("yolo11m.pt",       1280, 100, {}),
    # Pose: mosaic ve yatay flip kapalı. Flip, sol/sağ keypoint'leri yer değiştirir
    # (#1 ile #25 vb.) — flip_idx doğru tanımlanmadıysa model sol/sağı karıştırmayı öğrenir.
    "pose": ("yolo11m-pose.pt",  640,  300, {"mosaic": 0.0, "fliplr": 0.0}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(DEFAULTS))
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = AutoBatch; 0<x<1 = VRAM oranı")
    ap.add_argument("--device", default=None, help="0 | cpu")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs")
    a = ap.parse_args()

    model_name, imgsz, epochs, extra = DEFAULTS[a.task]
    data = a.data[0]
    if len(a.data) > 1:
        Path(a.project).mkdir(exist_ok=True)
        data = str(combine_yamls(a.data, Path(a.project) / f"combined_{a.task}.yaml"))
        print(f"[data] {len(a.data)} dataset birleştirildi -> {data}")

    from ultralytics import YOLO
    model = YOLO(a.model or model_name)
    batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
    # mutlak yol: yeni ultralytics sürümleri göreli project'i runs/<task>/ altına iç içe koyuyor
    project = str(Path(a.project).resolve())
    model.train(data=data, imgsz=a.imgsz or imgsz, epochs=a.epochs or epochs, batch=batch,
                device=a.device, patience=a.patience, project=project, name=a.name or a.task,
                exist_ok=True, plots=True, **extra)
    print(f"\n[ok] en iyi ağırlık: {model.trainer.best}")


if __name__ == "__main__":
    main()

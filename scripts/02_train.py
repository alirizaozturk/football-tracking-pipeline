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
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.dataio import combine_yamls  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402

DEFAULTS = {
    #        model               imgsz epochs  ekstra argümanlar
    "det":  ("yolo11m.pt",       1280, 100, {}),
    "ball": ("yolo11m.pt",       1280, 100, {}),
    # Pose: mosaic kapalı. Yatay flip sol/sağ keypoint'leri takas eder (#1 <-> #25 vb.); data.yaml'daki flip_idx
    # pitch.py'nin ayna eşlemesiyle birebir aynıysa fliplr=0.5, değilse 0 (aşağıda programatik karar).
    "pose": ("yolo11m-pose.pt",  640,  300, {"mosaic": 0.0}),
}


def pose_fliplr(data_yaml: str) -> tuple:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    flip = cfg.get("flip_idx")
    if flip is None:
        return 0.0, "data.yaml'da flip_idx yok"
    bad = PitchConfig().check_flip_idx(flip)
    return (0.5, "flip_idx pitch.py aynasıyla aynı") if not bad else (0.0, f"flip_idx uyuşmuyor: indeksler {bad}")


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
    ap.add_argument("--resume", action="store_true", help="<project>/<name>/weights/last.pt'den devam et")
    a = ap.parse_args()

    model_name, imgsz, epochs, extra = DEFAULTS[a.task]
    extra = dict(extra)
    name = a.name or a.task
    # mutlak yol: yeni ultralytics sürümleri göreli project'i runs/<task>/ altına iç içe koyuyor
    project = Path(a.project).resolve()
    run_dir = project / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.pid").write_text(str(os.getpid()))          # scripts/gpu_watchdog.py durdurmak için kullanır
    meta_p = run_dir / "train_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"sessions": []}
    t0 = time.time()

    from ultralytics import YOLO
    last = run_dir / "weights" / "last.pt"
    if a.resume and last.exists():
        print(f"[resume] {last}")
        meta["sessions"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": True})
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        data = a.data[0]
        if len(a.data) > 1:
            data = str(combine_yamls(a.data, project / f"combined_{a.task}.yaml"))
            print(f"[data] {len(a.data)} dataset birleştirildi -> {data}")
        if a.task == "pose":
            extra["fliplr"], why = pose_fliplr(data)
            print(f"[pose] fliplr={extra['fliplr']} ({why})")
            meta["fliplr"] = {"value": extra["fliplr"], "reason": why}
        meta["sessions"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": False})
        meta.update(task=a.task, data=str(Path(data).resolve()), model=a.model or model_name)
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        model = YOLO(a.model or model_name)
        batch = int(a.batch) if a.batch >= 1 or a.batch == -1 else a.batch
        model.train(data=data, imgsz=a.imgsz or imgsz, epochs=a.epochs or epochs, batch=batch,
                    device=a.device, patience=a.patience, project=str(project), name=name,
                    exist_ok=True, plots=True, **extra)
    meta["sessions"][-1].update(end=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1))
    meta["best"] = str(model.trainer.best)
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] en iyi ağırlık: {model.trainer.best}")


if __name__ == "__main__":
    main()

"""FAZ 1a — Roboflow Universe'ten videodaki hazır datasetleri indir.

    export ROBOFLOW_API_KEY=...            # roboflow.com -> Settings -> API Keys (ücretsiz hesap yeter)
    python scripts/01_download_datasets.py --which players field ball

Versiyon verilmezse projenin en son versiyonu indirilir. İndirilen her datasetin
data.yaml'ı mutlak yollara çevrilir (Roboflow'un '../train/images' göreli yolları
ultralytics'te klasik 'images not found' hatası verir).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.dataio import fix_roboflow_yaml  # noqa: E402

WORKSPACE = "roboflow-jvuqo"
PROJECTS = {
    "players": "football-players-detection-3zvbc",   # ball / goalkeeper / player / referee
    "field": "football-field-detection-f07vi",       # 32 saha keypoint'i (pose)
    "ball": "football-ball-detection-rejhg",         # sadece top (küçük obje, ayrı model)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["players", "field"], choices=list(PROJECTS))
    ap.add_argument("--out", default="data/roboflow")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--workspace", default=WORKSPACE)
    for k in PROJECTS:
        ap.add_argument(f"--{k}-version", type=int, default=None, help="boşsa en son versiyon")
    ap.add_argument("--format", default="yolov8", help="ultralytics YOLOv8/11 için 'yolov8'")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("ROBOFLOW_API_KEY yok. roboflow.com -> Settings -> API Keys")

    from roboflow import Roboflow
    rf = Roboflow(api_key=a.api_key)
    ws = rf.workspace(a.workspace)
    for key in a.which:
        project = ws.project(PROJECTS[key])
        vnum = getattr(a, f"{key}_version")
        if vnum is None:
            vnum = max(int(v.version) for v in project.versions())
        dest = Path(a.out) / f"{key}-v{vnum}"
        print(f"[{key}] {PROJECTS[key]} v{vnum} -> {dest}")
        project.version(vnum).download(a.format, location=str(dest))
        y = fix_roboflow_yaml(dest)
        print(f"[{key}] data.yaml düzeltildi: {y}")


if __name__ == "__main__":
    main()

"""FAZ 1a — Roboflow Universe'ten videodaki hazır datasetleri indir.

    export ROBOFLOW_API_KEY=...            # roboflow.com -> Settings -> API Keys (ücretsiz hesap yeter)
    python scripts/01_download_datasets.py --which players field ball

Versiyon verilmezse projenin en son versiyonu indirilir. İndirilen her datasetin
data.yaml'ı mutlak yollara çevrilir (Roboflow'un '../train/images' göreli yolları
ultralytics'te klasik 'images not found' hatası verir).

`roboflow` paketi kullanılmıyor, doğrudan Roboflow REST API'si çağrılıyor: paket
opencv-python-headless ve numpy<2.4 sabitliyor; projenin tek-OpenCV (opencv-python)
kuralıyla aynı ortama kurulamıyor. Akış paketin kendisiyle aynı:
    GET /{workspace}/{project}                     -> versions[].id  (".../<versiyon>")
    GET /{workspace}/{project}/{v}/{format}        -> 202: export hazırlanıyor, 200: export.link
"""
import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.dataio import fix_roboflow_yaml  # noqa: E402

API = "https://api.roboflow.com"
WORKSPACE = "roboflow-jvuqo"
PROJECTS = {
    "players": "football-players-detection-3zvbc",   # ball / goalkeeper / player / referee
    "field": "football-field-detection-f07vi",       # 32 saha keypoint'i (pose)
    "ball": "football-ball-detection-rejhg",         # sadece top (küçük obje, ayrı model)
}


def api_get(path: str, key: str):
    """-> (status, json). API anahtarı hata mesajlarına sızmasın diye URL yazdırılmaz."""
    try:
        r = requests.get(f"{API}/{path}", params={"api_key": key, "nocache": "true"}, timeout=60)
    except requests.RequestException as e:
        sys.exit(f"Roboflow API'ye ulaşılamadı ({path}): {type(e).__name__}")
    if r.status_code not in (200, 202):
        sys.exit(f"Roboflow API hatası {r.status_code} ({path}): {r.text[:300]}")
    return r.status_code, r.json()


def latest_version(workspace: str, project: str, key: str) -> int:
    _, info = api_get(f"{workspace}/{project}", key)
    nums = [int(str(v["id"]).rsplit("/", 1)[-1]) for v in info.get("versions", [])]
    if not nums:
        sys.exit(f"{workspace}/{project}: hiç versiyon yok")
    return max(nums)


def export_link(workspace: str, project: str, version: int, fmt: str, key: str, timeout_s: float = 900) -> str:
    t0 = time.time()
    while True:
        status, payload = api_get(f"{workspace}/{project}/{version}/{fmt}", key)
        if status == 200:
            return payload["export"]["link"]
        if time.time() - t0 > timeout_s:
            sys.exit(f"{project} v{version}: export {timeout_s:.0f} s içinde hazırlanmadı")
        print(f"\r  export hazırlanıyor: %{100 * float(payload.get('progress') or 0):.0f}", end="", flush=True)
        time.sleep(2)


def download_zip(link: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zp = dest / "roboflow.zip"
    with requests.get(link, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0) or None
        with open(zp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    zp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["players", "field"], choices=list(PROJECTS))
    ap.add_argument("--out", default="data/roboflow")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--workspace", default=WORKSPACE)
    for k in PROJECTS:
        ap.add_argument(f"--{k}-version", type=int, default=None, help="boşsa en son versiyon")
    ap.add_argument("--format", default="yolov8", help="ultralytics YOLOv8/11 için 'yolov8'")
    ap.add_argument("--overwrite", action="store_true", help="klasör varsa yine de indir")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("ROBOFLOW_API_KEY yok. roboflow.com -> Settings -> API Keys")

    for key in a.which:
        project = PROJECTS[key]
        vnum = getattr(a, f"{key}_version")
        if vnum is None:
            vnum = latest_version(a.workspace, project, a.api_key)
        dest = Path(a.out) / f"{key}-v{vnum}"
        if (dest / "data.yaml").exists() and not a.overwrite:
            print(f"[{key}] {dest} zaten var, atlanıyor (--overwrite ile tekrar indir)")
            continue
        print(f"[{key}] {project} v{vnum} -> {dest}")
        download_zip(export_link(a.workspace, project, vnum, a.format, a.api_key), dest)
        y = fix_roboflow_yaml(dest)
        print(f"[{key}] data.yaml düzeltildi: {y}")


if __name__ == "__main__":
    main()

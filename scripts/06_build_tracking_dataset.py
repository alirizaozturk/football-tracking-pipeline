"""FAZ 3 — Maç videosundan tracking dataseti üret (videodaki pipeline + export).

    python scripts/06_build_tracking_dataset.py --video match.mp4 \
        --det runs/det/weights/best.pt --kp runs/pose/weights/best.pt \
        [--ball runs/ball/weights/best.pt --ball-slice 640] [--team color] \
        [--start-s 0 --duration-s 60]

Çıktı: data/matches/<video>/{tracking.parquet, frames.parquet, meta.json, team_classifier.pkl}
Aynı klasöre tekrar çalıştırırsan takım sınıflandırıcı yeniden fit edilmez (silersen edilir).

Bütün eşikler parametre: PipelineConfig ve PostConfig'in her alanı --alan-adi (ör. --min-line-margin 0.1,
--min-line-score none, --line-probes-m 2 4 7, --cut-threshold none). Liste: --help. Kullanılan değerler meta.json'a yazılır.
"""
import argparse
import sys
from pathlib import Path

import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.build import build_match_dataset  # noqa: E402
from fai.cli import add_dataclass_args, dataclass_from_args  # noqa: E402
from fai.models import YoloDetector, YoloKeypoints  # noqa: E402
from fai.pipeline import PipelineConfig  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.postprocess import PostConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True, help="player/goalkeeper/referee(/ball) detection ağırlığı")
    ap.add_argument("--kp", default=None, help="saha keypoint ağırlığı (yoksa saha koordinatı NaN)")
    ap.add_argument("--ball", default=None, help="opsiyonel ayrı top modeli")
    ap.add_argument("--ball-slice", type=int, default=0, help="top modeli için tile boyutu (ör. 640); 0=kapalı")
    ap.add_argument("--team", choices=["siglip", "color"], default="siglip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.3, help="detector güven eşiği (track_high_conf ile aynı tut)")
    ap.add_argument("--length", type=float, default=105.0)
    ap.add_argument("--width", type=float, default=68.0)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--device", default=None)
    add_dataclass_args(ap, PipelineConfig, title="pipeline eşikleri (PipelineConfig)")
    add_dataclass_args(ap, PostConfig, title="postprocess eşikleri (PostConfig)")
    a = ap.parse_args()

    info = sv.VideoInfo.from_video_path(a.video)
    start = int(a.start_s * info.fps)
    end = start + int(a.duration_s * info.fps) if a.duration_s else None
    out = Path(a.out or f"data/matches/{Path(a.video).stem}")

    det = YoloDetector(a.det, conf=a.det_conf, imgsz=a.imgsz, device=a.device)
    kp = YoloKeypoints(a.kp, device=a.device) if a.kp else None
    ball = (YoloDetector(a.ball, conf=0.2, imgsz=a.ball_slice or a.imgsz, device=a.device,
                         only=["ball"], slice_wh=a.ball_slice or None) if a.ball else None)

    build_match_dataset(
        a.video, out, det, kp, team_method=a.team, ball_detector=ball,
        pitch=PitchConfig(length=a.length, width=a.width), cfg=dataclass_from_args(PipelineConfig, a),
        post_cfg=dataclass_from_args(PostConfig, a), start=start, end=end, fmt=a.format, device=a.device,
        extra_meta={"models": {"det": a.det, "kp": a.kp, "ball": a.ball}})


if __name__ == "__main__":
    main()

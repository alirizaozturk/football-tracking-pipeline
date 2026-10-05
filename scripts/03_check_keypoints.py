"""FAZ 1c — Keypoint modelini ve saha geometrisini GÖRSEL olarak doğrula.

    python scripts/03_check_keypoints.py --kp runs/pose/weights/best.pt --source match.mp4 --n 6

Her örnek frame için bir PNG üretir:
  * üstte: tahmin edilen keypoint'ler (numaralı) + homography ile geri projekte
    edilmiş saha çizgileri (kırmızı)
  * altta: fai/pitch.py'deki 32 noktanın numaralı şeması

Başlıkta: pipeline'ın kabul zincirinin sonucu (fai.pipeline.build_htracker; 06_build ile birebir aynı):
score = projekte çizgilerin görüntüdeki çizgi maskesine düşme oranı, margin = skor eksi ±2/4/7 m kaydırılmış
H'lerin en iyisi (doğru H yerel tepedir), sonra OK ya da RED:<sebep> (rmse / geometry / score / margin).
Eşikler PipelineConfig'ten; değiştirmek için --min-line-margin vb. (--help).

Tamamlanma kriteri: kırmızı çizgiler gerçek beyaz çizgilerin üstüne oturuyor ve
frame'deki numaralar şemadaki numaralarla aynı yerde. Oturmuyorsa:
  - numaralar kaymışsa -> keypoint sırası datasetle uyuşmuyor (pitch.vertices)
  - numaralar doğru ama çizgiler hafif kaçıksa -> saha ölçüleri (--length/--width)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.cli import add_dataclass_args, dataclass_from_args  # noqa: E402
from fai.homography import fit_homography  # noqa: E402
from fai.models import YoloKeypoints  # noqa: E402
from fai.pipeline import PipelineConfig, build_htracker  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.viz import Radar, project_pitch_lines  # noqa: E402


def load_frames(src: str, n: int):
    p = Path(src)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        return [(p.stem, cv2.imread(str(p)))]
    cap = cv2.VideoCapture(str(p))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out.append((f"{p.stem}_f{f}", img))
    return out


def diagram(pitch: PitchConfig, width: int) -> np.ndarray:
    r = Radar(pitch, scale=width / (pitch.length + 8), pad=int(4 * width / (pitch.length + 8)))
    img = r.base.copy()
    for i, (x, y) in enumerate(r.to_px(pitch.vertices)):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", required=True, help="pose model ağırlığı")
    ap.add_argument("--source", required=True, help="video ya da görüntü")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--length", type=float, default=105.0)
    ap.add_argument("--width", type=float, default=68.0)
    ap.add_argument("--out", default="outputs/keypoint_check")
    ap.add_argument("--device", default=None)
    add_dataclass_args(ap, PipelineConfig, title="homography eşikleri (PipelineConfig)")
    a = ap.parse_args()

    pitch = PitchConfig(length=a.length, width=a.width)
    cfg = dataclass_from_args(PipelineConfig, a)
    model = YoloKeypoints(a.kp, conf=0.3, device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, frame in load_frames(a.source, a.n):
        xy, conf = model(frame)
        img = frame.copy()
        msg = "keypoint yok"
        if xy is not None:
            keep = (conf >= cfg.kp_conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
            scale = frame.shape[1] / cfg.ref_width
            res = fit_homography(xy[keep], pitch.vertices[keep], cfg.ransac_px * scale) if keep.sum() >= 4 else None
            if res is not None:                       # çizim için aday H; karar pipeline'ın kabul zincirinden
                img = project_pitch_lines(img, res.H_pitch2img, pitch)
                ht = build_htracker(pitch, cfg)
                _, src = ht.update(xy, conf, frame)
                dg = ht.diag
                rm = "rmse anlamsız (4 nokta)" if res.n_inliers == 4 else f"rmse={res.reproj_rmse_px:.1f}px"
                verdict = "OK" if src == "fresh" else f"RED:{dg['reject_reason']}"
                msg = (f"kp={keep.sum()} inlier={res.n_inliers} {rm} score={dg['line_score']:.2f} "
                       f"margin={dg['line_margin']:.2f} -> {verdict}")
            else:
                msg = f"kp={keep.sum()} homography kurulamadı"
            for i, ((x, y), c) in enumerate(zip(xy, conf)):
                if x <= 1 and y <= 1:
                    continue
                col = (0, 255, 0) if c >= a.kp_conf else (128, 128, 128)
                cv2.circle(img, (int(x), int(y)), 6, col, -1)
                cv2.putText(img, str(i + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img, str(i + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        sheet = np.vstack([img, diagram(pitch, img.shape[1])])
        p = out / f"{name}.png"
        cv2.imwrite(str(p), sheet)
        print(f"{p}: {msg}")


if __name__ == "__main__":
    main()

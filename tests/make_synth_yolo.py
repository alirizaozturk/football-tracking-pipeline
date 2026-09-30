"""Sentetik sahneden ground-truth YOLO datasetleri üret (entegrasyon testi için).
    python -m tests.make_synth_yolo tests/_out/yolo
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fai.dataio import yolo_det_lines, yolo_pose_line  # noqa: E402
from fai.models import CLASSES  # noqa: E402
from tests.test_synthetic import FPS, H, PITCH, W, make_people, render_frame, scene  # noqa: E402


def main(out: Path, n_train=160, n_val=40):
    ppl = make_people()
    for split, frames in (("train", range(0, n_train * 2, 2)), ("val", range(1, n_val * 2, 2))):
        for task in ("det", "pose"):
            (out / task / split / "images").mkdir(parents=True, exist_ok=True)
            (out / task / split / "labels").mkdir(parents=True, exist_ok=True)
        for f in frames:
            f2 = f % (FPS * 12)
            Hm, objs, ball = scene(ppl, f2)
            img = render_frame(Hm, objs, ball)
            name = f"f{f:05d}"
            boxes = [o["box"] for o in objs] + [ball["box"]]
            cls = [o["cls"] for o in objs] + [CLASSES.index("ball")]
            for task in ("det", "pose"):
                cv2.imwrite(str(out / task / split / "images" / f"{name}.jpg"), img)
            (out / "det" / split / "labels" / f"{name}.txt").write_text(
                "\n".join(yolo_det_lines(np.array(boxes), cls, W, H)))
            uv = cv2.perspectiveTransform(PITCH.vertices.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            vis = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 95) & (uv[:, 1] < H)
            vx = uv[vis]
            box = [vx[:, 0].min(), vx[:, 1].min(), vx[:, 0].max(), vx[:, 1].max()]
            (out / "pose" / split / "labels" / f"{name}.txt").write_text(yolo_pose_line(box, uv, vis, W, H))
    (out / "det" / "data.yaml").write_text(yaml.safe_dump(
        {"path": str((out / "det").resolve()), "train": "train/images", "val": "val/images",
         "names": {i: n for i, n in enumerate(CLASSES)}}, sort_keys=False))
    (out / "pose" / "data.yaml").write_text(yaml.safe_dump(
        {"path": str((out / "pose").resolve()), "train": "train/images", "val": "val/images",
         "kpt_shape": [32, 3], "names": {0: "pitch"}}, sort_keys=False))
    print("ok", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests" / "_out" / "yolo"))

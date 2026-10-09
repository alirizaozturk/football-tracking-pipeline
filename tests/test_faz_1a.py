"""Faz 1a: flip_idx doğrulaması, H mesafesi ('doğru H' tanımı) ve model-keypoint kalibrasyon yolunun smoke testi.
Gerçek model kullanılmaz: sahte 'model' = GT keypoint + küçük gürültü.

Çalıştır:  python -m pytest tests/test_faz_1a.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fai.dataio import yolo_pose_line  # noqa: E402
from fai.homography import HomographyResult, grid_error_m  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402

P = PitchConfig()
EXPECTED_FLIP = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16,
                 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]


def _calib_module():
    spec = importlib.util.spec_from_file_location("calib01b", ROOT / "scripts" / "01b_calibrate_homography.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flip_idx_pitch_aynasi_beklenenle_ayni():
    assert P.mirror_index() == EXPECTED_FLIP
    assert P.check_flip_idx(EXPECTED_FLIP) == []
    bozuk = list(EXPECTED_FLIP)
    bozuk[0], bozuk[5] = bozuk[5], bozuk[0]
    assert P.check_flip_idx(bozuk) == [0, 5]


def test_grid_error_m_kimlik_ve_oteleme():
    from tests.test_negative import H, W, camera
    Hp = camera(50.0)
    Hi = np.linalg.inv(Hp)
    e0, n0 = grid_error_m(Hi / Hi[2, 2], Hp, W, H)
    assert e0 < 1e-6 and n0 > 500
    for d in ((3.0, 0.0), (0.0, -5.0), (4.0, 3.0)):
        T = np.array([[1, 0, -d[0]], [0, 1, -d[1]], [0, 0, 1.0]])
        Hs = np.linalg.inv(Hp @ T)                       # saha koordinatlarını +d kaydıran H
        e, _ = grid_error_m(Hs / Hs[2, 2], Hp, W, H)
        assert abs(e - np.hypot(*d)) < 1e-6, (d, e)


def test_kp_model_kalibrasyon_yolu_smoke():
    from tests.test_negative import H, W, camera, keypoints, lines_frame
    import shutil
    # tmp_path değil: kullanıcı klasörü ASCII dışı karakter içerebilir; cv2.imread/imwrite Windows'ta okuyamaz
    tmp_path = ROOT / "tests" / "_out" / "kp_smoke"
    shutil.rmtree(tmp_path, ignore_errors=True)
    data = tmp_path / "d"
    for split, xcs in (("calib", (40.0, 52.0, 64.0)), ("holdout", (46.0, 58.0))):
        (data / split / "images").mkdir(parents=True)
        (data / split / "labels").mkdir(parents=True)
        for xc in xcs:
            Hm = camera(xc)
            kp, conf = keypoints(Hm)
            vis = conf > 0.5
            box = [kp[vis, 0].min(), kp[vis, 1].min(), kp[vis, 0].max(), kp[vis, 1].max()]
            name = f"m{int(xc)}_0_0_png.jpg"
            cv2.imwrite(str(data / split / "images" / name), lines_frame(Hm))
            (data / split / "labels" / f"{Path(name).stem}.txt").write_text(yolo_pose_line(box, kp, vis, W, H))
    rng = np.random.default_rng(0)

    def predictor(frame, item):                            # sahte model: GT + 0.5 px gürültü
        split, ip, k = item
        vis = k[:, 2] > 0
        xy = np.where(vis[:, None], k[:, :2] * [W, H] + rng.normal(0, 0.5, (len(k), 2)), 0.0)
        return xy, np.where(vis, 0.95, 0.02)

    cal = _calib_module()
    out = tmp_path / "out"
    out.mkdir()
    res = cal.kp_model_study(None, data, out, lambda m: None, predictor=predictor)
    cal.kp_report(res, "sahte", out)
    import pandas as pd
    rows = pd.read_csv(out / "kp_rows.csv")
    nat, sh = rows[rows.kind == "natural"], rows[rows.kind == "shift"]
    assert (nat.cls == "correct").all() and nat.accepted.all(), nat[["image", "err_m", "cls", "accepted", "reason"]]
    assert (sh.cls == "wrong").all() and not sh.accepted.any(), sh[["d", "axis", "err_m", "accepted"]]
    assert np.isfinite(res["mu_star"])
    k = res["keypoints"]["seçim (calib)"]
    assert k["recall@0.5"]["rate"] == 1.0 and k["err640_median"] < 1.0
    assert (out / "kp_model.md").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

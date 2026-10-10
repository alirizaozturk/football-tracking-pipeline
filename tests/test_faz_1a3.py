"""Faz 1a-3: R3 ön-kayıtlı tanımların ve k-fold yardımcılarının birim testleri (gerçek model / data gerekmez).

Çalıştır:  python -m pytest tests/test_faz_1a3.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _kfold():
    spec = importlib.util.spec_from_file_location("kfold01f", ROOT / "scripts" / "01f_kfold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r3_tanimlari_on_kayitli_degerler():
    k = _kfold()
    assert (k.R3_CORRECT_M, k.R3_WRONG_M, k.R3_GRID_M) == (1.0, 2.0, 1.0)
    assert k.R3_SELECT_FOLDS == (0, 1) and k.R3_CONFIRM_FOLDS == (2,)
    assert [k.r3_class(x) for x in (0.99, 1.0, 1.5, 2.0, 2.01, np.nan)] == \
        ["correct", "ambiguous", "ambiguous", "ambiguous", "wrong", "unknown"]


def test_fold_atamasi_deterministik_ve_dengeli():
    k = _kfold()
    sizes = {"a": 76, "b": 43, "c": 32, "d": 31, "e": 15, "f": 13, "g": 12, "h": 8}
    frames = [("train", f"{m}_0_{i}_png.jpg") for m, n in sizes.items() for i in range(n)]
    f1, f2 = k.assign_folds(frames), k.assign_folds(list(reversed(frames)))
    assert f1 == f2
    per = [sum(n for m, n in sizes.items() if f1[m] == f) for f in range(k.N_FOLDS)]
    assert max(per) - min(per) <= 76 and sorted(set(f1.values())) == [0, 1, 2]


def _rows(fold, kind, err, margin, reason="", n=1):
    return [dict(image=f"x{fold}_{kind}_{err}_{margin}_{i}", match="m", fold=fold, refined_ref=True, kind=kind, d=0.0,
                 has_cand=True, reason=reason, score=0.5, margin=margin, err_ref=err, err_gt=err) for i in range(n)]


def test_mu_secimi_sadece_secim_foldlarindan_ve_bitti_mantigi():
    k = _kfold()
    rows = []
    # seçim fold'ları: 40 doğru aday, marjlar 0.10..0.49; yanlış adaylar marj 0.0
    for f in (0, 1):
        for j in range(20):
            rows += _rows(f, "natural", 0.5, 0.10 + 0.02 * j)
        rows += _rows(f, "shift", 5.0, 0.0, n=50)
    # teyit fold'u: çok düşük marjlı doğru adaylar seçime KARIŞMAMALI
    rows += _rows(2, "natural", 0.5, 0.0, n=5) + _rows(2, "natural", 0.5, 0.40, n=45) + _rows(2, "shift", 5.0, 0.0, n=50)
    df = pd.DataFrame(rows)
    res = k.calibrate(df)
    m_sel = np.array([0.10 + 0.02 * j for j in range(20)] * 2)
    assert res["mu_star"] == k.floor2(float(np.quantile(m_sel, 0.05)))
    sel = res["groups"]["seçim | referans=inceltilmiş | mu*"]
    conf = res["groups"]["teyit | referans=inceltilmiş | mu*"]
    assert sel["FAR"]["rate"] == 0.0 and sel["FRR"]["rate"] <= 0.10
    assert abs(conf["FRR"]["rate"] - 5 / 50) < 1e-9          # teyitteki 5 düşük marjlı doğru H reddedilir -> %10, hedef < %10 değil
    assert res["done"]["seçim"] is True and res["done"]["teyit"] is False

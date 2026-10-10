"""FAZ 1a-3 / E1-E2 — maç bazında k-fold pose eğitimi ve out-of-fold (OOF) kalibrasyon. R3 tanımları burada ÖN-KAYITLI.

    # 1) fold'lar + fold başına türetilmiş dataset (etiket seti: orig | clean)
    python scripts/01f_kfold.py prepare --labels orig
    python scripts/01f_kfold.py prepare --labels clean
    # 2) eğitim (02_train) -> runs/pose-f<k>-<labels>/weights/best.pt
    python scripts/02_train.py pose --data data/derived/field-v12-f0-orig/data.yaml --name pose-f0-orig --batch 12 --device 0
    # 3) OOF tahmin
    python scripts/01f_kfold.py predict --fold 0 --labels orig --weights runs/pose-f0-orig/weights/best.pt
    # 4) E1 A/B raporu (tek fold, iki etiket seti) / E2 kalibrasyonu (3 fold)
    python scripts/01f_kfold.py ab --fold 0
    python scripts/01f_kfold.py calibrate --labels clean

Fold ataması (deterministik): maçlar kare sayısına göre azalan sırada, o ana kadar en az karesi olan fold'a (eşitlikte
küçük indeks). Fold f için: eğitim = diğer fold'ların maçlarının orijinal train + test kareleri; early stopping / best.pt
seçimi = diğer fold'ların maçlarının orijinal valid kareleri; OOF = fold f maçlarının BÜTÜN kareleri (model seçimine girmez).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.homography import HomographyResult, grid_error_m, is_plausible  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402

P = PitchConfig()
V = P.vertices
N_FOLDS = 3

# ---------------------------------------------------------------- R3 ÖN-KAYIT (k-fold eğitiminden önce commit'lendi)
R3_CORRECT_M = 1.0          # model H'si ile REFERANS H arasındaki ortalama ızgara mesafesi < bu -> doğru
R3_WRONG_M = 2.0            # > bu -> yanlış; arası belirsiz (iki orana da girmez, ayrı raporlanır)
R3_GRID_M = 1.0             # görüntüde görünen saha üzerinde 1 m ızgara
R3_REFERENCE = "splits/field-v12/reference_h.json"   # R1: inceltilmiş H (kabul edilmişse), değilse GT H
R3_SELECT_FOLDS = (0, 1)    # marj eşiği bu fold'ların OOF karelerinde seçilir
R3_CONFIRM_FOLDS = (2,)     # teyit
R3_MU_Q = 0.05              # mu* = seçim fold'larındaki doğru-H marjlarının %5 quantile'ı (0.01'e aşağı yuvarlanır)
R3_TARGET_FAR, R3_TARGET_FRR = 0.05, 0.10
R3_MIN_NATURAL_WRONG = 10   # doğal yanlış H >= bu ise doğal FAR da < hedef olmalı
R3_SHIFTS = [(s * m, 0.0) for m in (3, 5, 10) for s in (-1, 1)] + [(0.0, s * m) for m in (3, 5, 10) for s in (-1, 1)]
HARD = ("few_kp", "fit", "rmse", "geometry")


def r3_class(err: float) -> str:
    if not np.isfinite(err):
        return "unknown"
    return "correct" if err < R3_CORRECT_M else ("wrong" if err > R3_WRONG_M else "ambiguous")


# ------------------------------------------------------------------------------ yardımcılar
def match_of(name: str) -> str:
    return name.split("_")[0]


def read_list(p: Path):
    return [l.split("#")[0].strip() for l in p.read_text(encoding="utf-8").splitlines() if l.split("#")[0].strip()] if p.exists() else []


def all_frames(root: Path, exclude):
    return [(s, p.name) for s in ("train", "valid", "test") for p in sorted((root / s / "images").iterdir()) if p.name not in exclude]


def assign_folds(frames, k=N_FOLDS):
    cnt = Counter(match_of(n) for _, n in frames)
    folds, size = {}, [0] * k
    for m, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(size))
        folds[m] = f
        size[f] += c
    return folds


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def label_root(labels: str, a) -> Path:
    return Path(a.data) if labels == "orig" else Path(a.clean)


def floor2(x):
    return math.floor(x * 100 + 1e-9) / 100


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - hw), min(1.0, c + hw)]


def rate(mask):
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), int(mask.size)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": wilson(k, n)}


def fr(r):
    return f"%{100 * r['rate']:.1f} ({r['k']}/{r['n']}; GA %{100 * r['ci95'][0]:.0f}–{100 * r['ci95'][1]:.0f})"


# ---------------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    root = Path(a.data)
    exclude = set(read_list(Path(a.exclude)))
    frames = all_frames(root, exclude)
    folds_p = Path(a.folds)
    if folds_p.exists():
        folds = json.loads(folds_p.read_text(encoding="utf-8"))["matches"]
    else:
        folds = assign_folds(frames)
        folds_p.write_text(json.dumps({"_doc": "maç -> fold (01f_kfold.assign_folds; kare sayısına göre dengeli, deterministik)",
                                       "matches": folds, "frames_per_fold": {str(f): sum(1 for _, n in frames if folds[match_of(n)] == f)
                                                                            for f in range(N_FOLDS)}},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    lab = label_root(a.labels, a)
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    for f in range(N_FOLDS):
        out = Path(f"data/derived/{root.name}-f{f}-{a.labels}").resolve()
        for s, n in frames:
            fold = folds[match_of(n)]
            dst = "oof" if fold == f else ("valid" if s == "valid" else "train")
            stem = Path(n).stem
            link(root / s / "images" / n, out / dst / "images" / n)
            link(lab / s / "labels" / f"{stem}.txt", out / dst / "labels" / f"{stem}.txt")
        (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out), "train": "train/images", "val": "valid/images",
                                                       "test": "oof/images", "kpt_shape": cfg["kpt_shape"], "flip_idx": cfg["flip_idx"],
                                                       "names": cfg["names"]}, sort_keys=False), encoding="utf-8")
        n = {d: len(list((out / d / "images").iterdir())) for d in ("train", "valid", "oof")}
        print(f"[fold {f}] {a.labels}: {n}; OOF maçları {sorted(m for m, x in folds.items() if x == f)}")


# ---------------------------------------------------------------------------------- predict
def cmd_predict(a):
    from fai.models import YoloKeypoints
    yk = YoloKeypoints(a.weights, device=a.device)              # 06_build ile aynı: conf 0.3, imgsz 640
    root = Path(f"data/derived/{Path(a.data).name}-f{a.fold}-{a.labels}") / "oof" / "images"
    pred = {}
    for ip in sorted(root.iterdir()):
        xy, conf = yk(cv2.imread(str(ip)))
        pred[ip.name] = (np.zeros((len(V), 2)) if xy is None else xy, np.zeros(len(V)) if conf is None else conf)
    out = Path(a.pred_dir) / f"oof_f{a.fold}_{a.labels}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pred=np.array(pred, dtype=object), fold=a.fold, labels=a.labels, weights=a.weights)
    print(f"[ok] {len(pred)} OOF kare -> {out}")


def load_pred(path: Path):
    z = np.load(path, allow_pickle=True)
    return dict(z["pred"].item())


# ---------------------------------------------------------------------------------- değerlendirme
def evaluate(pred: dict, a, folds_of_image: dict) -> pd.DataFrame:
    """Her OOF kare: model keypoint'leri -> kütüphanenin kabul zinciri; aday H'nin referansa (inceltilmiş) ve GT H'ye
    ızgara mesafesi; aday H'nin ±3/5/10 m kaydırılmışları da aynı zincirden. Sonuçlar mu'dan bağımsız kaydedilir."""
    from fai.pipeline import PipelineConfig, build_htracker
    ref = json.loads(Path(R3_REFERENCE).read_text(encoding="utf-8"))
    root = Path(a.data)
    cfg = PipelineConfig()
    rows = []
    for name, (xy, conf) in pred.items():
        if name not in ref:
            continue
        r = ref[name]
        ip = root / r["split"] / "images" / name
        frame = cv2.imread(str(ip))
        h, w = frame.shape[:2]
        H_ref, H_gt = np.array(r["H_pitch2img"]), np.array(r["H_gt_pitch2img"])
        xy, conf = np.asarray(xy, float), np.asarray(conf, float)
        ht = build_htracker(P, cfg)
        src = ht.update(xy, conf, frame)[1]
        dg, cand = dict(ht.diag), ht.last_candidate
        base = dict(image=name, match=match_of(name), fold=folds_of_image[name], refined_ref=bool(r["refined"]))

        def dist(c):
            if c is None:
                return np.nan, np.nan
            return (grid_error_m(c.H_img2pitch, H_ref, w, h, P.length, P.width, R3_GRID_M)[0],
                    grid_error_m(c.H_img2pitch, H_gt, w, h, P.length, P.width, R3_GRID_M)[0])

        e_ref, e_gt = dist(cand)
        rows.append(dict(base, kind="natural", d=0.0, has_cand=cand is not None, reason=dg["reject_reason"],
                         score=dg["line_score"], margin=dg["line_margin"], err_ref=e_ref, err_gt=e_gt))
        if cand is None:
            continue
        keep = (conf >= cfg.kp_conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
        for d in R3_SHIFTS:
            uv = cv2.perspectiveTransform((V - np.asarray(d)).reshape(-1, 1, 2), cand.H_pitch2img).reshape(-1, 2)
            ins = keep & np.isfinite(uv).all(1) & (uv[:, 0] > 1) & (uv[:, 0] < w - 1) & (uv[:, 1] > 1) & (uv[:, 1] < h - 1)
            uv[~ins] = 0.0
            hs = build_htracker(P, cfg)
            hs.update(uv, ins.astype(float), frame)
            ds, cs = dict(hs.diag), hs.last_candidate
            es_ref, es_gt = dist(cs)
            rows.append(dict(base, kind="shift", d=max(abs(d[0]), abs(d[1])), has_cand=cs is not None, reason=ds["reject_reason"],
                             score=ds["line_score"], margin=ds["line_margin"], err_ref=es_ref, err_gt=es_gt))
    return pd.DataFrame(rows)


def accept(df: pd.DataFrame, mu: float) -> np.ndarray:
    return ((~df.reason.isin(HARD)) & np.isfinite(df.score) & np.isfinite(df.margin) & (df.margin >= mu)).to_numpy()


def metrics(df: pd.DataFrame, mu: float, err_col: str) -> dict:
    cls = df[err_col].map(r3_class)
    acc = accept(df, mu)
    nat, sh = (df.kind == "natural").to_numpy(), (df.kind == "shift").to_numpy()
    corr = nat & (cls == "correct").to_numpy()
    wn, ws = nat & (cls == "wrong").to_numpy(), sh & (cls == "wrong").to_numpy()
    rej_reason = df.reason.where(df.reason != "", "margin")[corr & ~acc]
    return {"n_images": int(nat.sum()), "no_candidate": rate(~df.has_cand.to_numpy()[nat]),
            "correct": int(corr.sum()), "ambiguous": int((nat & (cls == "ambiguous").to_numpy()).sum()), "wrong_natural": int(wn.sum()),
            "FRR": rate(~acc[corr]), "FRR_by_reason": rej_reason.value_counts().to_dict(),
            "FAR_natural": rate(acc[wn]), "FAR_shift": rate(acc[ws]), "FAR": rate(np.r_[acc[wn], acc[ws]]),
            "accepted_ambiguous": int((acc & nat & (cls == "ambiguous").to_numpy()).sum())}


def passes(m: dict) -> bool:
    ok = m["FAR"]["rate"] < R3_TARGET_FAR and m["FRR"]["rate"] < R3_TARGET_FRR
    if m["FAR_natural"]["n"] >= R3_MIN_NATURAL_WRONG:
        ok = ok and m["FAR_natural"]["rate"] < R3_TARGET_FAR
    return bool(ok)


def calibrate(df: pd.DataFrame) -> dict:
    """R3 seçim kuralı: mu* = seçim fold'larında doğru (referansa < 1 m) doğal adayların marjlarının %5 quantile'ı
    (sert red sebepleri hariç); teyit fold'unda aynı mu* ile ölçülür."""
    sel = df[df.fold.isin(R3_SELECT_FOLDS) & (df.kind == "natural") & (df.err_ref.map(r3_class) == "correct") & ~df.reason.isin(HARD)]
    m = sel.margin[np.isfinite(sel.margin)]
    mu = floor2(float(np.quantile(m, R3_MU_Q))) if len(m) else float("nan")
    out = {"mu_star": mu, "n_select_correct": int(len(m)), "groups": {}}
    for gname, folds in (("seçim", R3_SELECT_FOLDS), ("teyit", R3_CONFIRM_FOLDS)):
        g = df[df.fold.isin(folds)]
        for ref_name, col in (("referans=inceltilmiş", "err_ref"), ("referans=GT", "err_gt")):
            for mu_name, mu_v in (("mu*", mu), ("mu=0.12", 0.12)):
                out["groups"][f"{gname} | {ref_name} | {mu_name}"] = metrics(g, mu_v, col) if np.isfinite(mu_v) else None
    out["done"] = {"seçim": passes(out["groups"]["seçim | referans=inceltilmiş | mu*"]) if np.isfinite(mu) else False,
                   "teyit": passes(out["groups"]["teyit | referans=inceltilmiş | mu*"]) if np.isfinite(mu) else False}
    return out


def kp_errors(pred: dict, a, label_set: str) -> pd.DataFrame:
    """OOF keypoint hatası (640 px) — verilen etiket setine göre (orig | clean)."""
    rows = []
    lab_root = label_root(label_set, a)
    ref = json.loads(Path(R3_REFERENCE).read_text(encoding="utf-8"))
    for name, (xy, conf) in pred.items():
        if name not in ref:
            continue
        s = ref[name]["split"]
        w = cv2.imread(str(Path(a.data) / s / "images" / name)).shape[1]
        h = cv2.imread(str(Path(a.data) / s / "images" / name)).shape[0]
        k = np.array((lab_root / s / "labels" / f"{Path(name).stem}.txt").read_text().split()[5:], float).reshape(-1, 3)
        for i in range(len(V)):
            if k[i, 2] > 0:
                rows.append(dict(image=name, kp=i + 1, conf=float(conf[i]),
                                 err640=float(np.hypot(*(np.asarray(xy[i]) - k[i, :2] * [w, h])) * 640 / w)))
    return pd.DataFrame(rows)


def folds_by_image(a):
    folds = json.loads(Path(a.folds).read_text(encoding="utf-8"))["matches"]
    exclude = set(read_list(Path(a.exclude)))
    return {n: folds[match_of(n)] for _, n in all_frames(Path(a.data), exclude)}


def cmd_ab(a):
    """E1: aynı fold, orijinal vs temiz etiketle eğitilmiş iki model; OOF keypoint ve H hatası iki referansa göre."""
    fbi = folds_by_image(a)
    L = [f"# E1 — fold {a.fold}: orijinal vs temiz etiket (OOF kareler)\n",
         "| model (eğitim etiketi) | kp hatası medyan / p90, orijinal etikete göre | temiz etikete göre | recall@0.5 | H hatası medyan, referans=inceltilmiş | referans=GT | doğru / belirsiz / yanlış (R3, inceltilmiş) |",
         "|---|---|---|---|---|---|---|"]
    res = {}
    for lab in ("orig", "clean"):
        pred = load_pred(Path(a.pred_dir) / f"oof_f{a.fold}_{lab}.npz")
        ko, kc = kp_errors(pred, a, "orig"), kp_errors(pred, a, "clean")
        hit_o, hit_c = ko[ko.conf >= 0.5], kc[kc.conf >= 0.5]
        ev = evaluate(pred, a, fbi)
        nat = ev[ev.kind == "natural"]
        cls = nat.err_ref.map(r3_class)
        res[lab] = {"kp_orig_med": float(hit_o.err640.median()), "kp_orig_p90": float(hit_o.err640.quantile(.9)),
                    "kp_clean_med": float(hit_c.err640.median()), "kp_clean_p90": float(hit_c.err640.quantile(.9)),
                    "recall": float((ko.conf >= 0.5).mean()), "H_ref_med": float(nat.err_ref.median()),
                    "H_gt_med": float(nat.err_gt.median()), "classes": cls.value_counts().to_dict()}
        r = res[lab]
        L.append(f"| {lab} | {r['kp_orig_med']:.2f} / {r['kp_orig_p90']:.2f} | {r['kp_clean_med']:.2f} / {r['kp_clean_p90']:.2f} | "
                 f"%{100 * r['recall']:.1f} | {r['H_ref_med']:.2f} m | {r['H_gt_med']:.2f} m | "
                 f"{r['classes'].get('correct', 0)} / {r['classes'].get('ambiguous', 0)} / {r['classes'].get('wrong', 0)} |")
        ev.to_csv(Path(a.out) / f"ab_f{a.fold}_{lab}_rows.csv", index=False)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"ab_f{a.fold}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (Path(a.out) / f"ab_f{a.fold}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


def cmd_calibrate(a):
    fbi = folds_by_image(a)
    parts = []
    for f in range(N_FOLDS):
        pred = load_pred(Path(a.pred_dir) / f"oof_f{f}_{a.labels}.npz")
        parts.append(evaluate(pred, a, fbi))
    df = pd.concat(parts, ignore_index=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / f"kfold_{a.labels}_rows.csv", index=False)
    res = calibrate(df)
    (Path(a.out) / f"kfold_{a.labels}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    L = [f"# E2 — {N_FOLDS}-fold OOF kalibrasyon ({a.labels} etiket), R3 ön-kayıtlı tanımlar\n",
         f"Doğru < {R3_CORRECT_M} m, yanlış > {R3_WRONG_M} m (ortalama ızgara hatası, {R3_GRID_M} m ızgara); referans `{R3_REFERENCE}` "
         f"(inceltilmiş H, R1'de reddedilenlerde GT H). mu* = fold {R3_SELECT_FOLDS} doğru-H marjlarının %{100 * R3_MU_Q:.0f} quantile'ı "
         f"(n={res['n_select_correct']}) = **{res['mu_star']}**; teyit fold {R3_CONFIRM_FOLDS}.\n",
         "| küme | referans | mu | doğru / belirsiz / yanlış | doğru H reddi (FRR) | red sebebi | yanlış H kabulü (FAR) | doğal FAR | kaydırılmış FAR | aday yok |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in res["groups"].items():
        g, refn, mun = key.split(" | ")
        if m is None:
            L.append(f"| {g} | {refn} | {mun} | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {g} | {refn} | {mun} | {m['correct']} / {m['ambiguous']} / {m['wrong_natural']} | {fr(m['FRR'])} | {m['FRR_by_reason']} | "
                 f"{fr(m['FAR'])} | {fr(m['FAR_natural'])} | {fr(m['FAR_shift'])} | {fr(m['no_candidate'])} |")
    L.append(f"\n**Bitti (R3, ön-kayıt):** seçimde {'TUTTU' if res['done']['seçim'] else 'TUTMADI'}, teyitte "
             f"{'TUTTU' if res['done']['teyit'] else 'TUTMADI'} (FAR < %{100 * R3_TARGET_FAR:.0f}, FRR < %{100 * R3_TARGET_FRR:.0f}; "
             f"doğal yanlış H ≥ {R3_MIN_NATURAL_WRONG} ise doğal FAR da).")
    (Path(a.out) / f"kfold_{a.labels}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "predict", "ab", "calibrate"])
    ap.add_argument("--data", default="data/roboflow/field-v12")
    ap.add_argument("--clean", default="data/derived/field-v12-clean")
    ap.add_argument("--exclude", default="splits/field-v12/exclude.txt")
    ap.add_argument("--folds", default="splits/field-v12/folds.json")
    ap.add_argument("--labels", choices=["orig", "clean"], default="clean")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pred-dir", default="outputs/kfold/pred")
    ap.add_argument("--out", default="outputs/kfold")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "predict": cmd_predict, "ab": cmd_ab, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    main()

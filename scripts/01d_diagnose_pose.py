"""FAZ 1a-2 — Pose modeli teşhisi (yeni eğitim yok). Ön-kayıtlı tanımlar (doğru < 0.5 m, yanlış > 2 m, ortalama ızgara
hatası) değişmez; buradakiler ek analiz.

    python scripts/01d_diagnose_pose.py --weights runs/pose-1a/weights/best.pt

D1  keypoint bazında hata vektörleri (calib + holdout): sistematik kısım (ortalama vektör) ve rastgele kısım
    (ortalama çevresindeki RMS). 31/32 etiket hipotezi: etiket, gerçek saha noktası (L/2 ± r, W/2) yerine
    orta yuvarlağın görüntüdeki en sol / en sağ noktasına mı konmuş? (31/32 hariç GT H ile karşılaştırılır)
D2  H fit'ten keypoint dışlama: (a) 31-32 hariç, (b) calib'de hatası en yüksek k keypoint hariç. k ve liste SADECE
    calib'den (kural: calib'de model-H ızgara hatası medyanını en küçük yapan k, eşitlikte küçük k); holdout'ta raporlanır.
D3  inference imgsz 640 / 960 / 1280 (aynı model): keypoint ve H hatası.
D4  GT gürültü tabanı: GT keypoint'lerinin GT H altında reprojection residual'ı ve keypoint bootstrap'ıyla GT H'nin
    belirsizliği (aynı ızgara metriği).
D5  ek H metrikleri: ızgara hatasının medyanı ve sadece görüntünün alt 2/3'üne düşen ızgara noktalarının ortalaması.
Çıktı: outputs/diagnose/<dataset>/diag.md, diag.json, predictions_*.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.homography import HomographyResult, fit_homography, is_plausible  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402

P = PitchConfig()
V = P.vertices
KP_CONF = 0.5
CORRECT_M, WRONG_M = 0.5, 2.0             # ön-kayıtlı (değişmez)
UNSEEN = ("calib", "holdout")
SAME = ("valid", "test_same")
KP_NAMES = {**{i: "sol kale çizgisi" for i in range(1, 7)}, 7: "sol kale alanı", 8: "sol kale alanı", 9: "sol penaltı",
            **{i: "sol ceza sahası" for i in range(10, 14)}, **{i: "orta çizgi" for i in range(14, 18)},
            **{i: "sağ ceza sahası" for i in range(18, 22)}, 22: "sağ penaltı", 23: "sağ kale alanı", 24: "sağ kale alanı",
            **{i: "sağ kale çizgisi" for i in range(25, 31)}, 31: "orta yuvarlak sol", 32: "orta yuvarlak sağ"}


# ----------------------------------------------------------------------------- yardımcılar
def load_items(root: Path, splits):
    items = []
    for split in splits:
        d = root / split
        if not (d / "images").exists():
            continue
        imgs = {p.stem: p for p in (d / "images").iterdir()}
        for lp in sorted((d / "labels").glob("*.txt")):
            rows = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if rows and lp.stem in imgs:
                items.append((split, imgs[lp.stem], np.array(rows[0][5:], float).reshape(-1, 3)))
    return items


def grid_errors(H_i2p: np.ndarray, H_ref_p2i: np.ndarray, w: int, h: int, step: float = 1.0) -> dict:
    """Ön-kayıtlı metrik (ortalama) + D5 ekleri (medyan, görüntünün alt 2/3'ü)."""
    xs, ys = np.arange(0, P.length + 1e-9, step), np.arange(0, P.width + 1e-9, step)
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    hom = np.c_[g, np.ones(len(g))] @ H_ref_p2i.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = hom[:, :2] / hom[:, 2:3]
    front = hom[:, 2] > 0 if np.median(hom[:, 2]) > 0 else hom[:, 2] < 0
    ok = front & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not ok.any():
        return {"mean": np.nan, "median": np.nan, "lower23": np.nan, "n": 0}
    q = cv2.perspectiveTransform(uv[ok].reshape(-1, 1, 2), H_i2p).reshape(-1, 2)
    e = np.hypot(*(q - g[ok]).T)
    low = uv[ok][:, 1] >= h / 3
    return {"mean": float(np.mean(e)), "median": float(np.median(e)),
            "lower23": float(np.mean(e[low])) if low.any() else np.nan, "n": int(ok.sum())}


def cls_of(e: float) -> str:
    return "yok" if not np.isfinite(e) else ("doğru" if e < CORRECT_M else ("yanlış" if e > WRONG_M else "belirsiz"))


def gt_h(k, w, h, exclude=()):
    vis = k[:, 2] > 0
    if exclude:
        vis = vis.copy()
        vis[list(exclude)] = False
    if vis.sum() < 4:
        return None
    return fit_homography(k[vis, :2] * [w, h], V[vis], 8.0)


def model_h(xy, conf, w, exclude=()):
    keep = (conf >= KP_CONF) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if exclude:
        keep[list(exclude)] = False
    if keep.sum() < 4:
        return None
    return fit_homography(xy[keep], V[keep], 8.0 * w / 1280)       # pipeline ile aynı (RANSAC ölçekli)


def predict(weights, items, imgsz, device, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(z["pred"].item())
    from ultralytics import YOLO
    m = YOLO(str(weights))
    pred = {}
    for split, ip, k in items:
        r = m.predict(cv2.imread(str(ip)), imgsz=imgsz, conf=0.3, device=device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            pred[ip.name] = (np.zeros((len(V), 2)), np.zeros(len(V)))
            continue
        xy, kc = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        i = int(np.argmax(kc.mean(1)))
        pred[ip.name] = (xy[i], kc[i])
    np.savez(cache, pred=np.array(pred, dtype=object))
    return pred


def pct(x) -> str:
    return f"%{100 * x:.1f}"


def md(df: pd.DataFrame, digits=2) -> str:
    cell = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def h_rows(items, pred, exclude=(), tag=""):
    rows = []
    for split, ip, k in items:
        h, w = IMG_WH[ip.name]
        gt = gt_h(k, w, h)
        if gt is None or not is_plausible(gt, w, h, P.length, P.width):
            continue
        xy, conf = pred[ip.name]
        mh = model_h(np.asarray(xy, float), np.asarray(conf, float), w, exclude)
        ge = grid_errors(mh.H_img2pitch, gt.H_pitch2img, w, h) if mh is not None else {"mean": np.nan, "median": np.nan, "lower23": np.nan, "n": 0}
        rows.append(dict(cfg=tag, split=split, image=ip.name, err=ge["mean"], err_median=ge["median"],
                         err_lower23=ge["lower23"], cls=cls_of(ge["mean"])))
    return pd.DataFrame(rows)


def summarize_h(df: pd.DataFrame, group_cols=("cfg",)) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(list(group_cols), sort=False):
        e = g.err.dropna()
        n = len(g)
        out.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))), "n": n,
                    "H hatası medyan (m)": float(e.median()) if len(e) else np.nan,
                    "p10": float(e.quantile(.1)) if len(e) else np.nan, "p90": float(e.quantile(.9)) if len(e) else np.nan,
                    "doğru": int((g.cls == "doğru").sum()), "belirsiz": int((g.cls == "belirsiz").sum()),
                    "yanlış": int((g.cls == "yanlış").sum()), "H yok": int((g.cls == "yok").sum()),
                    "D5 ızgara medyanı (medyan, m)": float(g.err_median.median()),
                    "D5 alt 2/3 ortalaması (medyan, m)": float(g.err_lower23.median())})
    return pd.DataFrame(out)


IMG_WH: dict = {}


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/pose-1a/weights/best.pt")
    ap.add_argument("--data", default="data/derived/field-v12-1a")
    ap.add_argument("--full", default="data/roboflow/field-v12", help="D4 için tüm GT görüntüleri")
    ap.add_argument("--exclude-list", default="splits/field-v12/exclude.txt")
    ap.add_argument("--out", default="outputs/diagnose/field-v12")
    ap.add_argument("--device", default="cpu", help="GPU eğitimdeyken cpu")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    root = Path(a.data)
    items = load_items(root, UNSEEN + SAME)
    for _, ip, _ in items:
        IMG_WH[ip.name] = cv2.imread(str(ip)).shape[:2]
    unseen = [x for x in items if x[0] in UNSEEN]
    calib = [x for x in items if x[0] == "calib"]
    hold = [x for x in items if x[0] == "holdout"]
    preds = {s: predict(a.weights, items, s, a.device, out / f"predictions_{s}.npz") for s in a.imgsz}
    print(f"[tahmin] {len(items)} görüntü x {a.imgsz} ({time.time() - t0:.0f} s)", flush=True)
    R = {}
    L = ["# Faz 1a-2 — pose teşhisi (yeni eğitim yok)\n",
         f"Model: `{Path(a.weights).as_posix()}`. Görülmemiş maçlar: calib ({len(calib)}) + holdout ({len(hold)}); aynı-maç: "
         f"valid + test_same ({len(items) - len(unseen)}). Keypoint hataları 640 ölçeğinde px (× 640 / W); H hatası birincil "
         f"metrik = ön-kayıtlı ortalama ızgara hatası (m); doğru < {CORRECT_M} m, yanlış > {WRONG_M} m (değişmedi).\n"]

    # ------------------------------------------------------------------ D1
    p640 = preds[640]
    rows = []
    for split, ip, k in items:
        h, w = IMG_WH[ip.name]
        xy, conf = p640[ip.name]
        for i in range(len(V)):
            if k[i, 2] > 0 and conf[i] >= KP_CONF:
                d = (np.asarray(xy[i]) - k[i, :2] * [w, h]) * 640 / w
                rows.append(dict(split=split, kp=i + 1, dx=d[0], dy=d[1]))
    kd = pd.DataFrame(rows)
    d1 = []
    for i, g in kd[kd.split.isin(UNSEEN)].groupby("kp"):
        mu = g[["dx", "dy"]].mean().to_numpy()
        res = g[["dx", "dy"]].to_numpy() - mu
        rms_rand = float(np.sqrt((res ** 2).sum(1).mean()))
        mag = np.hypot(g.dx, g.dy)
        d1.append({"kp": int(i), "nokta": KP_NAMES[int(i)], "n": len(g), "ort. vektör dx": float(mu[0]), "dy": float(mu[1]),
                   "|sistematik|": float(np.hypot(*mu)), "rastgele RMS": rms_rand, "|hata| medyan": float(mag.median()),
                   "sistematik payı": float(np.hypot(*mu) ** 2 / (np.hypot(*mu) ** 2 + rms_rand ** 2))})
    d1 = pd.DataFrame(d1).sort_values("|hata| medyan", ascending=False)
    R["D1"] = d1.to_dict("records")

    # 31/32 etiket hipotezi: tüm GT görüntüleri (31/32 hariç GT H ile)
    full_items = [x for x in load_items(Path(a.full), ("train", "valid", "test"))
                  if x[1].name not in set(l.strip() for l in Path(a.exclude_list).read_text(encoding="utf-8").splitlines()
                                          if l.strip() and not l.startswith("#"))]
    for _, ip, _ in full_items:
        if ip.name not in IMG_WH:
            IMG_WH[ip.name] = cv2.imread(str(ip)).shape[:2]
    circ = P.polylines()[6]                                         # orta yuvarlak (yoğun polyline)
    lab = []
    for split, ip, k in full_items:
        h, w = IMG_WH[ip.name]
        Hx = gt_h(k, w, h, exclude=(30, 31))
        if Hx is None or not is_plausible(Hx, w, h, P.length, P.width):
            continue
        cuv = cv2.perspectiveTransform(circ.reshape(-1, 1, 2), Hx.H_pitch2img).reshape(-1, 2)
        for i, ext in ((30, cuv[np.argmin(cuv[:, 0])]), (31, cuv[np.argmax(cuv[:, 0])])):
            if k[i, 2] <= 0:
                continue
            lab_px = k[i, :2] * [w, h]
            true_px = cv2.perspectiveTransform(V[i].reshape(1, 1, 2), Hx.H_pitch2img).ravel()
            s = 640 / w
            lab.append(dict(kp=i + 1, d_true=np.hypot(*(lab_px - true_px)) * s, d_ext=np.hypot(*(lab_px - ext)) * s,
                            true_vs_ext=np.hypot(*(true_px - ext)) * s, off_x=(lab_px - true_px)[0] * s,
                            off_y=(lab_px - true_px)[1] * s, closer_to_ext=bool(np.hypot(*(lab_px - ext)) < np.hypot(*(lab_px - true_px)))))
    # karşılaştırma tabanı: diğer keypoint'lerin aynı (31/32 hariç) GT H altında residual'ı
    base = []
    for split, ip, k in full_items:
        h, w = IMG_WH[ip.name]
        Hx = gt_h(k, w, h, exclude=(30, 31))
        if Hx is None:
            continue
        vis = (k[:, 2] > 0)
        vis[[30, 31]] = False
        pr = cv2.perspectiveTransform(V[vis].reshape(-1, 1, 2), Hx.H_pitch2img).reshape(-1, 2)
        base += list(np.hypot(*(pr - k[vis, :2] * [w, h]).T) * 640 / w)
    labdf = pd.DataFrame(lab)
    d1b = []
    for i, g in labdf.groupby("kp"):
        d1b.append({"kp": int(i), "n": len(g), "etiket→gerçek nokta medyan": float(g.d_true.median()),
                    "etiket→elips ucu medyan": float(g.d_ext.median()), "gerçek↔uç mesafe medyan": float(g.true_vs_ext.median()),
                    "uca daha yakın": pct(g.closer_to_ext.mean()), "ort. ofset dx": float(g.off_x.mean()), "dy": float(g.off_y.mean())})
    d1b = pd.DataFrame(d1b)
    R["D1_31_32"] = {"rows": d1b.to_dict("records"), "other_kp_residual_median": float(np.median(base)),
                     "other_kp_residual_p90": float(np.percentile(base, 90))}
    L += ["## D1 — keypoint bazında hata (calib + holdout, model @640, conf ≥ 0.5)\n",
          "Sistematik = ortalama hata vektörü; rastgele = ortalama çevresindeki RMS; sistematik payı = |sist|² / (|sist|² + RMS²).\n",
          md(d1.round(2)), "\n### 31/32 etiket hipotezi (tüm GT görüntüleri; 31/32 hariç fit edilen GT H ile)\n",
          f"Karşılaştırma tabanı: diğer GT keypoint'lerinin aynı H altında residual'ı medyan "
          f"{np.median(base):.2f} px, p90 {np.percentile(base, 90):.2f} px.\n", md(d1b.round(2))]

    # ------------------------------------------------------------------ D2
    base640 = h_rows(items, p640, (), "dışlama yok")
    calib_rank = []
    for i, g in kd[kd.split == "calib"].groupby("kp"):            # sıralama SADECE calib'den
        calib_rank.append((float(np.hypot(g.dx, g.dy).median()), int(i)))
    calib_rank = [i for _, i in sorted(calib_rank, reverse=True)]
    ks = list(range(0, 11))
    kres = []
    for kk in ks:
        excl = tuple(i - 1 for i in calib_rank[:kk])
        df = h_rows(calib, p640, excl, f"k={kk}")
        kres.append((float(df.err.median()), kk, excl, df))
    best = min(kres, key=lambda r: (r[0] if np.isfinite(r[0]) else 1e9, r[1]))
    k_star, excl_star = best[1], best[2]
    d2 = pd.concat([base640.assign(cfg="dışlama yok"),
                    h_rows(items, p640, (30, 31), "(a) 31-32 hariç"),
                    h_rows(items, p640, excl_star, f"(b) calib'in en kötü k={k_star}: {[i + 1 for i in excl_star]}")])
    d2["set"] = d2.split.map(lambda s: s if s in UNSEEN else "aynı-maç")
    t2 = summarize_h(d2, ("cfg", "set"))
    t2k = pd.DataFrame([{"k": kk, "dışlanan": [i + 1 for i in ex], "calib H hatası medyan (m)": e} for e, kk, ex, _ in kres])
    R["D2"] = {"k_star": k_star, "excluded": [i + 1 for i in excl_star], "k_sweep_calib": t2k.to_dict("records"),
               "table": t2.to_dict("records")}
    L += ["\n## D2 — H fit'ten keypoint dışlama (model @640)\n",
          "k ve liste sadece calib'den: keypoint'ler calib medyan hatasına göre sıralanır, calib'de H hatası medyanını en küçük "
          "yapan k seçilir (eşitlikte küçük k). holdout seçimde kullanılmadı.\n", md(t2k.round(2)),
          f"\nSeçilen k = {k_star}: {[i + 1 for i in excl_star]}\n", md(t2.round(2))]

    # ------------------------------------------------------------------ D3
    d3k, d3h = [], []
    for s in a.imgsz:
        p = preds[s]
        for grp, sp in (("görülmemiş (calib+holdout)", UNSEEN), ("aynı-maç", SAME)):
            errs, rec_n, rec_d = [], 0, 0
            for split, ip, k in items:
                if split not in sp:
                    continue
                h, w = IMG_WH[ip.name]
                xy, conf = p[ip.name]
                vis = k[:, 2] > 0
                rec_d += int(vis.sum())
                hit = vis & (np.asarray(conf) >= KP_CONF)
                rec_n += int(hit.sum())
                errs += list(np.hypot(*((np.asarray(xy)[hit] - k[hit, :2] * [w, h]) * 640 / w).T))
            d3k.append({"imgsz": s, "set": grp, "recall@0.5": pct(rec_n / max(rec_d, 1)),
                        "kp hata medyan": float(np.median(errs)), "p90": float(np.percentile(errs, 90))})
        hd = h_rows(items, p, (), f"imgsz {s}")
        hd["set"] = hd.split.map(lambda x: "görülmemiş (calib+holdout)" if x in UNSEEN else "aynı-maç")
        d3h.append(summarize_h(hd, ("cfg", "set")))
    t3k, t3h = pd.DataFrame(d3k), pd.concat(d3h)
    R["D3"] = {"keypoints": t3k.to_dict("records"), "H": t3h.to_dict("records")}
    L += ["\n## D3 — inference çözünürlüğü (aynı model, eğitim 640'ta)\n", md(t3k.round(2)), "", md(t3h.round(2)),
          "\nNot: iyileşme anlamlıdır; kötüleşme 1280'de eğitimin işe yaramayacağını göstermez (model 640'ta eğitildi, ölçek uyumsuzluğu).\n"]

    # ------------------------------------------------------------------ D4
    rng = np.random.default_rng(a.seed)
    res_all, res_31, boot = [], [], []
    for split, ip, k in full_items:
        h, w = IMG_WH[ip.name]
        vis = np.where(k[:, 2] > 0)[0]
        gt = gt_h(k, w, h)
        if gt is None or not is_plausible(gt, w, h, P.length, P.width):
            continue
        pr = cv2.perspectiveTransform(V[vis].reshape(-1, 1, 2), gt.H_pitch2img).reshape(-1, 2)
        r = np.hypot(*(pr - k[vis, :2] * [w, h]).T) * 640 / w
        for i, rv in zip(vis, r):
            (res_31 if i in (30, 31) else res_all).append(rv)
        bs = []
        for _ in range(a.bootstrap):
            idx = rng.choice(vis, size=len(vis), replace=True)
            if len(np.unique(idx)) < 4:
                continue
            hb = fit_homography(k[idx, :2] * [w, h], V[idx], 8.0)
            if hb is None:
                continue
            bs.append(grid_errors(hb.H_img2pitch, gt.H_pitch2img, w, h))
        if bs:
            boot.append({"image": ip.name, "mean_med": float(np.median([b["mean"] for b in bs])),
                         "mean_p90": float(np.percentile([b["mean"] for b in bs], 90)),
                         "median_med": float(np.median([b["median"] for b in bs])),
                         "lower23_med": float(np.nanmedian([b["lower23"] for b in bs]))})
    bdf = pd.DataFrame(boot)
    d4 = pd.DataFrame([
        {"ölçü": "GT residual, 31/32 hariç keypoint'ler (640 px)", "medyan": float(np.median(res_all)), "p90": float(np.percentile(res_all, 90)), "n": len(res_all)},
        {"ölçü": "GT residual, 31/32 (640 px)", "medyan": float(np.median(res_31)), "p90": float(np.percentile(res_31, 90)), "n": len(res_31)},
        {"ölçü": "bootstrap GT H belirsizliği: ortalama ızgara hatası (m) — görüntü başına medyan", "medyan": float(bdf.mean_med.median()), "p90": float(bdf.mean_med.quantile(.9)), "n": len(bdf)},
        {"ölçü": "bootstrap: görüntü başına p90 (m)", "medyan": float(bdf.mean_p90.median()), "p90": float(bdf.mean_p90.quantile(.9)), "n": len(bdf)},
        {"ölçü": "bootstrap D5: ızgara medyanı (m)", "medyan": float(bdf.median_med.median()), "p90": float(bdf.median_med.quantile(.9)), "n": len(bdf)},
        {"ölçü": "bootstrap D5: alt 2/3 ortalaması (m)", "medyan": float(bdf.lower23_med.median()), "p90": float(bdf.lower23_med.quantile(.9)), "n": len(bdf)},
    ])
    R["D4"] = d4.to_dict("records")
    L += [f"\n## D4 — GT gürültü tabanı (tüm field v12 GT görüntüleri, exclude hariç; bootstrap B={a.bootstrap})\n", md(d4.round(3)),
          f"\nBootstrap: GT keypoint'leri yerine koyarak örneklenir, H yeniden fit edilir, tam GT H'ye ortalama ızgara hatası. "
          f"Görüntülerin %{100 * (bdf.mean_p90 > CORRECT_M).mean():.0f}'inde bootstrap p90'ı {CORRECT_M} m'yi aşıyor.\n"]
    (out / "diag.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] {out / 'diag.md'} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

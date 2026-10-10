"""FAZ 1a-3 — Referans H'yi görüntüdeki çizgilere göre incelt (R1), tabanı yeniden ölç (R2), temiz etiket seti (R4).

    python scripts/01e_refine_reference.py --data data/roboflow/field-v12

R1  GT H'den başlayarak top-hat maskesi üzerinde ECC (fai.homography.refine_homography, öteleme araması kapalı).
    Görüntü başına kabul (hepsi): ECC yakınsadı ve geometri geçti; line score VE marj arttı; GT keypoint'lerin medyan
    residual'ı (GT H'nin RANSAC aykırıları hariç: residual > 8 px) artmadı; inceltme H'yi ızgara metriğiyle (ortalama)
    <= 1 m kaydırdı. Sağlamayan görüntüde GT H kalır ve işaretlenir.
R2  Aynı görüntüde farklı başlangıçlardan inceltme: GT keypoint alt kümelerinden (%60, rastgele) kurulan H'ler ve GT H'nin
    ±1 m x/y ötelenmişleri. Yakınsayan H'lerin (ECC + geometri) GT-başlangıçlı inceltilmiş H'ye ızgara mesafesi; görüntü
    başına medyan; taban = görüntülerin medyanı. Yanında eski bootstrap tabanı (aynı görüntüler).
R4  Temiz etiket: inceltmesi kabul edilen görüntülerde 32 keypoint inceltilmiş H'den yeniden projekte edilir; görünürlük
    orijinal etiketten, görüntü dışına düşen noktalar görünmez. Kabul edilmeyenlerde orijinal etiket. Ayrı klasör (--clean-out);
    dataset dosyalarına dokunulmaz.

Çıktı: <out>/refine.md, refine.json, per_image.csv; splits/<dataset>/reference_h.json (R3 referansı, commit'lenir);
<clean-out>/{train,valid,test}/{images(hardlink),labels}, data.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.homography import (HomographyResult, LineCheck, fit_homography, grid_error_m, is_plausible,  # noqa: E402
                            line_evidence, line_mask, pitch_line_samples, probe_offsets, refine_homography)
from fai.pitch import PitchConfig  # noqa: E402

P = PitchConfig()
V = P.vertices
CHK = LineCheck()
SAMPLES, OFFSETS = pitch_line_samples(P.polylines(), CHK.step_m), probe_offsets(CHK.probes_m)
MAX_SHIFT_M = 1.0
OUTLIER_PX = 8.0                      # GT fit'in RANSAC eşiği (native px)
N_SUBSETS, SUBSET_FRAC = 6, 0.6
PERTURB_M = 1.0
STOP_FLOOR_M = 0.33                   # R3: inceltilmiş tabanın medyanı bunu aşarsa dur


def res_of(Hp2i: np.ndarray, n=0, ninl=0, rmse=0.0) -> HomographyResult:
    Hi = np.linalg.inv(Hp2i)
    return HomographyResult(Hi / Hi[2, 2], Hp2i / Hp2i[2, 2], n, ninl, rmse)


def residuals(Hp2i, pts_px, idx):
    pr = cv2.perspectiveTransform(V[idx].reshape(-1, 1, 2), Hp2i).reshape(-1, 2)
    return np.hypot(*(pr - pts_px[idx]).T)


def ecc_refine(frame, res):
    r, info = refine_homography(frame, res, P.polylines(), CHK, search_m=0.0, ecc=True)
    ok = isinstance(info["ecc"], float)
    return r, ok, info["ecc"]


def load_items(root: Path, exclude):
    items = []
    for s in ("train", "valid", "test"):
        imgs = {p.stem: p for p in (root / s / "images").iterdir()}
        for lp in sorted((root / s / "labels").glob("*.txt")):
            if lp.stem in imgs and imgs[lp.stem].name not in exclude:
                items.append((s, imgs[lp.stem], lp))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow/field-v12")
    ap.add_argument("--exclude-list", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clean-out", default=None)
    ap.add_argument("--reference-out", default=None)
    ap.add_argument("--bootstrap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data)
    name = root.name
    out = Path(a.out or f"outputs/refine/{name}")
    clean = Path(a.clean_out or f"data/derived/{name}-clean")
    ref_out = Path(a.reference_out or f"splits/{name}/reference_h.json")
    out.mkdir(parents=True, exist_ok=True)
    exl = Path(a.exclude_list or f"splits/{name}/exclude.txt")
    exclude = {l.strip() for l in exl.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    items = load_items(root, exclude)
    rows, refs, kpdiff, spread_rows = [], {}, [], []
    for n_img, (split, ip, lp) in enumerate(items):
        frame = cv2.imread(str(ip))
        h, w = frame.shape[:2]
        line = lp.read_text().split()
        k = np.array(line[5:], float).reshape(-1, 3)
        vis = np.where(k[:, 2] > 0)[0]
        px = k[:, :2] * [w, h]
        gt = fit_homography(px[vis], V[vis], OUTLIER_PX)
        if gt is None or not is_plausible(gt, w, h, P.length, P.width):
            rows.append(dict(split=split, image=ip.name, status="gt_yok"))
            continue
        r_gt = residuals(gt.H_pitch2img, px, vis)
        inl = vis[r_gt <= OUTLIER_PX]
        mask, tol = line_mask(frame, CHK)
        s_gt, m_gt = line_evidence(mask, tol, gt.H_pitch2img, SAMPLES, OFFSETS, CHK.min_samples)
        # ---------------- R1
        ref, ok, cc = ecc_refine(frame, gt)
        geo = ok and is_plausible(ref, w, h, P.length, P.width)
        s_r, m_r = line_evidence(mask, tol, ref.H_pitch2img, SAMPLES, OFFSETS, CHK.min_samples) if geo else (np.nan, np.nan)
        med_gt = float(np.median(residuals(gt.H_pitch2img, px, inl)))
        med_r = float(np.median(residuals(ref.H_pitch2img, px, inl))) if geo else np.nan
        shift, _ = grid_error_m(ref.H_img2pitch, gt.H_pitch2img, w, h, P.length, P.width, 1.0) if geo else (np.nan, 0)
        conds = {"ecc_geo": bool(geo), "score_up": bool(geo and s_r > s_gt), "margin_up": bool(geo and m_r > m_gt),
                 "residual_not_up": bool(geo and med_r <= med_gt), "shift_le_1m": bool(geo and shift <= MAX_SHIFT_M)}
        accepted = all(conds.values())
        s640 = 640 / w
        rows.append(dict(split=split, image=ip.name, status="kabul" if accepted else "red", ecc_cc=cc if ok else np.nan,
                         s_gt=s_gt, s_ref=s_r, m_gt=m_gt, m_ref=m_r, res_med_gt640=med_gt * s640,
                         res_med_ref640=med_r * s640, shift_m=shift, n_vis=len(vis), n_inl=len(inl), **conds))
        H_use = ref.H_pitch2img if accepted else gt.H_pitch2img
        refs[ip.name] = {"split": split, "refined": accepted, "H_pitch2img": H_use.tolist(), "H_gt_pitch2img": gt.H_pitch2img.tolist()}
        # ---------------- R2 (taban)
        if geo:
            starts = []
            for dx, dy in ((PERTURB_M, 0), (-PERTURB_M, 0), (0, PERTURB_M), (0, -PERTURB_M)):
                T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]])
                starts.append(("±1m", res_of(gt.H_pitch2img @ T)))
            m_sub = max(4, int(round(SUBSET_FRAC * len(inl))))
            for _ in range(N_SUBSETS):
                if len(inl) <= 4:
                    break
                sub = rng.choice(inl, size=m_sub, replace=False)
                hs = fit_homography(px[sub], V[sub], OUTLIER_PX)
                if hs is not None and is_plausible(hs, w, h, P.length, P.width):
                    starts.append(("alt küme", hs))
            d_conv = []
            for kind, st in starts:
                r2, ok2, _ = ecc_refine(frame, st)
                if ok2 and is_plausible(r2, w, h, P.length, P.width):
                    d_conv.append((kind, grid_error_m(r2.H_img2pitch, ref.H_pitch2img, w, h, P.length, P.width, 1.0)[0]))
            boot = []
            for _ in range(a.bootstrap):
                idx = rng.choice(vis, size=len(vis), replace=True)
                if len(np.unique(idx)) < 4:
                    continue
                hb = fit_homography(px[idx], V[idx], OUTLIER_PX)
                if hb is not None:
                    boot.append(grid_error_m(hb.H_img2pitch, gt.H_pitch2img, w, h, P.length, P.width, 1.0)[0])
            spread_rows.append(dict(image=ip.name, accepted=accepted, n_starts=len(starts), n_conv=len(d_conv),
                                    floor_ref=float(np.median([d for _, d in d_conv])) if d_conv else np.nan,
                                    floor_ref_p90=float(np.percentile([d for _, d in d_conv], 90)) if d_conv else np.nan,
                                    floor_pert=float(np.median([d for kd, d in d_conv if kd == "±1m"])) if any(kd == "±1m" for kd, _ in d_conv) else np.nan,
                                    floor_sub=float(np.median([d for kd, d in d_conv if kd == "alt küme"])) if any(kd == "alt küme" for kd, _ in d_conv) else np.nan,
                                    floor_boot=float(np.median(boot)) if boot else np.nan))
        # ---------------- R4 (temiz etiket)
        if accepted:
            uv = cv2.perspectiveTransform(V.reshape(-1, 1, 2), ref.H_pitch2img).reshape(-1, 2)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            vis_new = (k[:, 2] > 0) & inside
            for i in vis:
                kpdiff.append(dict(kp=i + 1, d640=float(np.hypot(*(uv[i] - px[i])) * s640), to_invisible=bool(not inside[i])))
            parts = line[:5]
            for i in range(len(V)):
                parts += [f"{uv[i, 0] / w:.6f}", f"{uv[i, 1] / h:.6f}", "2"] if vis_new[i] else ["0", "0", "0"]
            new_label = " ".join(parts)
        else:
            new_label = lp.read_text().strip()
        dst_img = clean / split / "images" / ip.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            try:
                os.link(ip, dst_img)
            except OSError:
                shutil.copy2(ip, dst_img)
        (clean / split / "labels").mkdir(parents=True, exist_ok=True)
        (clean / split / "labels" / lp.name).write_text(new_label + "\n", encoding="utf-8")
        if (n_img + 1) % 25 == 0:
            print(f"  {n_img + 1}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)

    df, sp, kd = pd.DataFrame(rows), pd.DataFrame(spread_rows), pd.DataFrame(kpdiff)
    df.to_csv(out / "per_image.csv", index=False)
    sp.to_csv(out / "floor.csv", index=False)
    kd.to_csv(out / "kp_diff.csv", index=False)
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    (clean / "data.yaml").write_text(yaml.safe_dump({"path": str(clean.resolve()), "train": "train/images", "val": "valid/images",
                                                     "test": "test/images", "kpt_shape": cfg["kpt_shape"], "flip_idx": cfg["flip_idx"],
                                                     "names": cfg["names"]}, sort_keys=False), encoding="utf-8")
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    ref_out.write_text(json.dumps({"_doc": "Faz 1a-3 R1: görüntü başına referans H (pitch->img). refined=true: ECC ile inceltilmiş "
                                           "ve R1 kabul şartlarını geçmiş; false: GT H (etiketlerden RANSAC 8 px).", **refs},
                                  ensure_ascii=False), encoding="utf-8")

    # ---------------- rapor
    g = df[df.status != "gt_yok"]
    n = len(g)
    acc = g[g.status == "kabul"]
    cond_cols = ["ecc_geo", "score_up", "margin_up", "residual_not_up", "shift_le_1m"]
    L = ["# Faz 1a-3 — referans H inceltme (R1), taban (R2), temiz etiket (R4)\n",
         f"{name}: {n} görüntü (exclude hariç). ECC: top-hat maskesi (eşik {CHK.tophat_thr:g}) ile çizilen saha çizgileri, 960 px genişlikte, "
         f"GT H'den başlayarak (öteleme araması yok). Residual: GT H'nin RANSAC inlier'ları (residual ≤ {OUTLIER_PX:g} px), 640 ölçeğinde px.\n",
         "## R1 — inceltme kabulü\n",
         f"**Kabul: {len(acc)} / {n} (%{100 * len(acc) / n:.1f}); red: {n - len(acc)}.**\n",
         "| şart | geçen görüntü |\n|---|---|"]
    for c in cond_cols:
        L.append(f"| {c} | {int(g[c].sum())} / {n} |")
    only = {c: int((~g[c] & g[[x for x in cond_cols if x != c]].all(axis=1)).sum()) for c in cond_cols}
    L.append("\nSadece tek şarttan kalan (diğer dördünü geçen) görüntü sayısı: " + ", ".join(f"{c} {v}" for c, v in only.items()) + ".\n")
    d = (g.res_med_ref640 - g.res_med_gt640).dropna()
    L.append(f"Residual değişimi (inceltilmiş − GT, medyan residual, 640 px): medyan {d.median():+.2f}, p10 {d.quantile(.1):+.2f}, "
             f"p90 {d.quantile(.9):+.2f}. Bilgi (uygulanmadı): residual şartı +0.25 / +0.5 / +1.0 px toleranslı olsaydı kabul "
             + " / ".join(str(int((g[[c for c in cond_cols if c != 'residual_not_up']].all(axis=1) & (g.res_med_ref640 <= g.res_med_gt640 + t)).sum()))
                          for t in (0.25, 0.5, 1.0)) + " olurdu.\n")
    L.append(f"Kabul edilenlerde: line score {acc.s_gt.median():.3f} → {acc.s_ref.median():.3f}, marj {acc.m_gt.median():.3f} → "
             f"{acc.m_ref.median():.3f}, kayma medyan {acc.shift_m.median():.2f} m (p90 {acc.shift_m.quantile(.9):.2f}).\n" if len(acc) else "")
    L.append("## R2 — taban (farklı başlangıçlardan inceltme)\n")
    L.append(f"Başlangıçlar: GT H ±{PERTURB_M:g} m (x, y; 4) + GT keypoint'lerinin %{100 * SUBSET_FRAC:.0f} alt kümelerinden H ({N_SUBSETS}). "
             "Yakınsayan (ECC + geometri) H'lerin GT-başlangıçlı inceltilmiş H'ye ortalama ızgara mesafesi; görüntü başına medyan.\n")
    L.append("| küme | n görüntü | yakınsama oranı | taban medyan (m) | p90 | ±1 m başlangıç | alt küme başlangıç | eski bootstrap tabanı (GT H, m) |\n|---|---|---|---|---|---|---|---|")
    for lab, s in (("R1 kabul", sp[sp.accepted]), ("R1 red", sp[~sp.accepted]), ("hepsi", sp)):
        if len(s):
            L.append(f"| {lab} | {len(s)} | %{100 * s.n_conv.sum() / max(s.n_starts.sum(), 1):.0f} | {s.floor_ref.median():.3f} | "
                     f"{s.floor_ref.quantile(.9):.3f} | {s.floor_pert.median():.3f} | {s.floor_sub.median():.3f} | {s.floor_boot.median():.3f} |")
    fl = float(sp[sp.accepted].floor_ref.median()) if sp.accepted.any() else float("nan")
    L.append(f"\n**İnceltilmiş taban (R1 kabul edilenler) medyanı {fl:.3f} m; durdurma eşiği {STOP_FLOOR_M} m: "
             f"{'AŞILDI — dur' if not (fl <= STOP_FLOOR_M) else 'aşılmadı'}.**\n")
    L.append("## R4 — temiz etiket (orijinal ↔ inceltilmiş H projeksiyonu, 640 px; sadece kabul edilen görüntüler)\n")
    if len(kd):
        t = kd.groupby("kp").agg(n=("d640", "size"), medyan=("d640", "median"), p90=("d640", lambda x: x.quantile(.9)),
                                 görünmeze=("to_invisible", "sum")).reset_index()
        L.append(f"Genel: n={len(kd)}, medyan {kd.d640.median():.2f}, p90 {kd.d640.quantile(.9):.2f}; görüntü dışına düştüğü için "
                 f"görünmez yapılan {int(kd.to_invisible.sum())}.\n")
        L.append("| kp | n | medyan | p90 | görünmeze |\n|---|---|---|---|---|")
        L += [f"| {int(r.kp)} | {int(r.n)} | {r.medyan:.2f} | {r.p90:.2f} | {int(r.görünmeze)} |" for r in t.itertuples()]
    L.append(f"\nTemiz etiket seti: `{clean.as_posix()}` (kabul edilmeyenlerde orijinal etiket). Referans: `{ref_out.as_posix()}`.")
    (out / "refine.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summ = {"n": n, "accepted": int(len(acc)), "conditions": {c: int(g[c].sum()) for c in cond_cols}, "only_fail": only,
            "floor_refined_median": fl, "stop": bool(not (fl <= STOP_FLOOR_M)), "elapsed_s": round(time.time() - t0)}
    (out / "refine.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] kabul {len(acc)}/{n}, inceltilmiş taban medyan {fl:.3f} m -> {out / 'refine.md'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

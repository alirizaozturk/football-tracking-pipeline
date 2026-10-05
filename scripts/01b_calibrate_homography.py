"""FAZ K — Homography doğrulama eşiklerini gerçek yayın görüntüleriyle kalibre et (model eğitimi yok).

    python scripts/01b_calibrate_homography.py --data data/roboflow/field-v12

Her görüntüde GT keypoint'lerinden (v>0) homography fit edilir; sonra:
  1. is_plausible geçme oranı (keypoint sırası + y yönü doğrulaması)
  2. doğru H'nin çizgi desteği (line score) dağılımı
  3. aynı H'nin x / y yönünde ±3/5/10 m kaydırılmış hallerinin (tutarlı-ama-yanlış H) skor dağılımı
  4. maske: mevcut HSV eşiği vs top-hat (L kanalında yerel kontrast); tolerans: sabit 2 px vs çözünürlükle ölçekli
  5. marj: skor(H) - max skor(H'nin ±2/4/7 m kaydırılmış probe'ları). Doğru H yerel bir tepedir; kaymış H değildir.
     Probe'lar test kaymalarıyla (3/5/10) çakışmaz, yani hiçbir probe bir kaymayı tam geri almaz.
  6. çözünürlük (#10): görüntü 0.5x / 1x / 2x ölçeklenir, 640 girişli pose modelini taklit eden keypoint gürültüsü eklenir.

Eşikler valid split'te yalnızca DOĞRU H dağılımından (alt quantile) seçilir; kaydırılmış H'lerin kabul oranı (FAR)
optimize edilmez, ölçülür. test (ve ek kontrol olarak train) split'i seçimde kullanılmaz.
"mevcut kod" sütunu fai.homography.HomographyTracker.update üzerinden hesaplanır: Faz A'dan sonra aynı script yeni
kodu ölçer.

Çıktı (--out): per_image.csv, scores.csv, summary.json, report.md, fig_*.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.homography import (HomographyResult, LineCheck, fit_homography, is_plausible, line_evidence,  # noqa: E402
                            line_mask, probe_offsets)
from fai.pipeline import TrackingPipeline  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402

P = PitchConfig()
GRID = [(s * m, 0.0) for m in (3, 5, 10) for s in (-1, 1)] + [(0.0, s * m) for m in (3, 5, 10) for s in (-1, 1)]
DIAG = [(sx * 7.0, sy * 7.0) for sx in (-1, 1) for sy in (-1, 1)]
PROBES = [(s * m, 0.0) for m in (2, 4, 7) for s in (-1, 1)] + [(0.0, s * m) for m in (2, 4, 7) for s in (-1, 1)]
TOPHAT_THR = (10, 20, 30, 40)
MIN_SAMPLES = 30
Q_A = 0.05            # kural A: tau = doğru H skorunun bu quantile'ı
Q_B = 0.04            # kural B: tau ve mu = doğru H skor / marjının bu quantile'ları
TARGET_FAR, TARGET_FRR, TARGET_GEO = 0.05, 0.10, 0.95

# grafik (dataviz referans paleti, light)
C_TRUE, C_SHIFT = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRIDC, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# ------------------------------------------------------------------ geometri / skor
def pitch_samples(step: float = 0.5) -> np.ndarray:
    """fai.homography.pitch_line_samples ile aynı örnekleme."""
    seg = []
    for pl in P.polylines():
        for a, b in zip(pl[:-1], pl[1:]):
            k = max(2, int(np.hypot(*(b - a)) / step))
            seg.append(np.linspace(a, b, k, endpoint=False))
    return np.concatenate(seg)


BASE = pitch_samples()


def shifted(res: HomographyResult, d) -> HomographyResult:
    """Saha noktası p'yi gerçekte p-d'nin olduğu yere koyan H (görüntü->saha koordinatları +d kayar)."""
    T = np.array([[1, 0, -d[0]], [0, 1, -d[1]], [0, 0, 1.0]])
    Hp = res.H_pitch2img @ T
    Hi = np.linalg.inv(Hp)
    return HomographyResult(Hi / Hi[2, 2], Hp / Hp[2, 2], res.n_points, res.n_inliers, res.reproj_rmse_px)


def score(mask_d: np.ndarray, Hp2i: np.ndarray, d, tol: int) -> float:
    h, w = mask_d.shape
    uv = cv2.perspectiveTransform((BASE - np.asarray(d)).reshape(-1, 1, 2), Hp2i).reshape(-1, 2)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= tol) & (uv[:, 0] < w - tol) & (uv[:, 1] >= tol) & (uv[:, 1] < h - tol)
    if ok.sum() < MIN_SAMPLES:
        return float("nan")
    uv = np.round(uv[ok]).astype(int)
    return float(mask_d[uv[:, 1], uv[:, 0]].mean())


def score_and_margin(mask_d, Hp2i, d, tol):
    s = score(mask_d, Hp2i, d, tol)
    ps = [score(mask_d, Hp2i, (d[0] + p[0], d[1] + p[1]), tol) for p in PROBES]
    ps = [x for x in ps if np.isfinite(x)]
    return s, (s - max(ps)) if (np.isfinite(s) and ps) else float("nan")


def mask_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] >= 160) & (hsv[..., 1] <= 70)).astype(np.uint8)


def tophat_response(frame):
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(3, int(round(frame.shape[1] / 128)) | 1)            # 1920 px -> 15 px çekirdek (çizgi kalınlığından büyük)
    return cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def dilate(m, tol):
    return cv2.dilate(m, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))


def tol_of(kind: str, w: int) -> int:
    return 2 if kind == "fixed" else max(1, int(round(w / 640)))


def variant_masks(frame, tol_kinds=("fixed", "scaled"), thrs=TOPHAT_THR):
    """-> {(mask_name, tol_kind): (dilated_mask, tol)}"""
    w = frame.shape[1]
    hsv, th = mask_hsv(frame), tophat_response(frame)
    out = {}
    for tk in tol_kinds:
        tol = tol_of(tk, w)
        out[("hsv", tk)] = (dilate(hsv, tol), tol)
        for t in thrs:
            out[(f"tophat{t}", tk)] = (dilate((th >= t).astype(np.uint8), tol), tol)
    return out


def pipeline_accepts(frame, kp_xy, conf):
    """Kodun bugünkü kabul zinciri (fresh mi?) — HomographyTracker'ı pipeline nasıl kuruyorsa öyle."""
    ht = TrackingPipeline(None, None, None, P, 25.0, tracker=lambda d: d).htrack
    _, src = ht.update(kp_xy, conf, frame)
    return src == "fresh", ht.last_line_score


def shifted_keypoints(res, vis, d, w, h):
    """Model keypoint i'yi saha noktası (vertex_i - d)'nin görüntüdeki yerine koymuş olsun (tutarlı x/y kayması)."""
    uv = cv2.perspectiveTransform((P.vertices - np.asarray(d)).reshape(-1, 1, 2), res.H_pitch2img).reshape(-1, 2)
    inside = vis & np.isfinite(uv).all(1) & (uv[:, 0] > 1) & (uv[:, 0] < w - 1) & (uv[:, 1] > 1) & (uv[:, 1] < h - 1)
    uv[~inside] = 0.0
    return uv, inside.astype(float)


# ------------------------------------------------------------------------- dataset
def load_items(root: Path, splits):
    items = []
    for split in splits:
        imgs = {p.stem: p for p in (root / split / "images").iterdir()}
        for lp in sorted((root / split / "labels").glob("*.txt")):
            rows = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if rows and lp.stem in imgs:
                items.append((split, imgs[lp.stem], np.array(rows[0][5:], float).reshape(-1, 3)))
    return items


# ----------------------------------------------------------------------- istatistik
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


def rate(mask) -> dict:
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), int(mask.size)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def fmt_rate(r) -> str:
    return f"%{100 * r['rate']:.1f} ({r['k']}/{r['n']}; GA %{100 * r['ci95'][0]:.0f}–{100 * r['ci95'][1]:.0f})"


def floor2(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def cell(v):
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------- değerlendirme
def evaluate_main(items, log):
    img_rows, sc_rows, max_diff = [], [], 0.0
    for i, (split, ip, k) in enumerate(items):
        frame = cv2.imread(str(ip))
        h, w = frame.shape[:2]
        vis = k[:, 2] > 0
        kp = k[:, :2] * [w, h]
        res = fit_homography(kp[vis], P.vertices[vis], 8.0)
        row = dict(split=split, image=ip.name, w=w, h=h, n_kp=int(vis.sum()), fit_ok=res is not None)
        if res is None:
            img_rows.append(row)
            continue
        geo0 = is_plausible(res, w, h, P.length, P.width)
        row.update(n_inliers=res.n_inliers, rmse_px=res.reproj_rmse_px, geo_ok=geo0)
        # kodun bugünkü kararı
        acc, ls = pipeline_accepts(frame, np.where(vis[:, None], kp, 0.0), vis.astype(float))
        row.update(pipe_accept=acc, pipe_line_score=ls)
        for d in GRID + DIAG:
            a, _ = pipeline_accepts(frame, *shifted_keypoints(res, vis, d, w, h))
            sc_rows.append(dict(split=split, image=ip.name, variant="pipeline", tol="-", dx=d[0], dy=d[1],
                                kind="grid" if d in GRID else "diag", s=np.nan, margin=np.nan, geo=np.nan, pipe_acc=a))
        # skor varyantları
        geo = {d: is_plausible(shifted(res, d), w, h, P.length, P.width) for d in GRID + DIAG}
        geo[(0.0, 0.0)] = geo0
        for (mname, tk), (md, tol) in variant_masks(frame).items():
            for d in [(0.0, 0.0)] + GRID + DIAG:
                s, m = score_and_margin(md, res.H_pitch2img, d, tol)
                kind = "correct" if d == (0.0, 0.0) else ("grid" if d in GRID else "diag")
                sc_rows.append(dict(split=split, image=ip.name, variant=mname, tol=tk, dx=d[0], dy=d[1], kind=kind,
                                    s=s, margin=m, geo=geo[d], pipe_acc=np.nan))
        # kendi skor + marjımızın kütüphaneninkiyle aynı olduğunu doğrula (top-hat 10, tol W/640)
        lm, ltol = line_mask(frame, LineCheck(tophat_thr=10.0, tol_div=640.0))
        ref = line_evidence(lm, ltol, res.H_pitch2img, BASE, probe_offsets((2.0, 4.0, 7.0)), MIN_SAMPLES)
        md, tol = variant_masks(frame, ("scaled",), (10,))[("tophat10", "scaled")]
        mine = score_and_margin(md, res.H_pitch2img, (0.0, 0.0), tol)
        for r_, m_ in zip(ref, mine):
            if np.isfinite(r_) or np.isfinite(m_):
                max_diff = max(max_diff, abs(r_ - m_))
        img_rows.append(row)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(items)} görüntü")
    return pd.DataFrame(img_rows), pd.DataFrame(sc_rows), max_diff


def decide(df, tau, mu=None):
    acc = df["geo"].astype(bool) & df["s"].notna() & (df["s"] >= tau)
    if mu is not None:
        acc &= df["margin"].notna() & (df["margin"] >= mu)
    return acc


def choose_and_score(sc: pd.DataFrame):
    """Her varyant için eşikleri valid'in doğru-H dağılımından seç; tüm split'lerde FAR/FRR ölç."""
    out = []
    var = sc[sc.variant != "pipeline"]
    for (vname, tk), g in var.groupby(["variant", "tol"]):
        cv = g[(g.split == "valid") & (g.kind == "correct") & g.geo.astype(bool)]
        tauA = floor2(np.nanquantile(cv.s, Q_A))
        tauB = floor2(np.nanquantile(cv.s, Q_B))
        muB = floor2(np.nanquantile(cv.margin, Q_B))
        sv = g[(g.split == "valid") & (g.kind == "grid") & g.geo.astype(bool)]
        sep = float(np.nanquantile(cv.margin, Q_B) - np.nanquantile(sv.margin, 0.99))   # ayrışma payı (sadece valid)
        rec = dict(variant=vname, tol=tk, tauA=tauA, tauB=tauB, muB=muB, sep_valid=sep,
                   med_true=float(np.nanmedian(g[(g.kind == "correct") & (g.split != "train")].s)),
                   med_shift=float(np.nanmedian(g[(g.kind == "grid") & (g.split != "train")].s)))
        for split in ("valid", "test", "train"):
            gs = g[g.split == split]
            c, sh, dg = gs[gs.kind == "correct"], gs[gs.kind == "grid"], gs[gs.kind == "diag"]
            if not len(c):
                continue
            for rule, (tau, mu) in (("A", (tauA, None)), ("B", (tauB, muB))):
                rec[f"{rule}_{split}_FRR"] = rate(~decide(c, tau, mu))
                rec[f"{rule}_{split}_FAR"] = rate(decide(sh, tau, mu))
                rec[f"{rule}_{split}_FAR_diag"] = rate(decide(dg, tau, mu))
        out.append(rec)
    return out


def far_breakdown(sc, vname, tk, tau, mu, splits=("test",)):
    g = sc[(sc.variant == vname) & (sc.tol == tk) & (sc.kind == "grid") & sc.split.isin(splits)]
    acc = decide(g, tau, mu)
    rows = []
    for axis, col in (("x", "dx"), ("y", "dy")):
        for m in (3, 5, 10):
            sel = (g[col].abs() == m) & (g["dy" if col == "dx" else "dx"] == 0)
            rows.append((axis, m, rate(acc[sel])))
    return rows


def evaluate_resolution(items, chosen, seed=0, log=print):
    """#10: ölçek x keypoint gürültüsü x {mevcut, önerilen-sabit-piksel, önerilen-ölçekli}."""
    rng = np.random.default_rng(seed)
    vname, tau, mu = chosen["variant"], chosen["tau"], chosen["mu"]
    rows = []
    for split, ip, k in items:
        frame0 = cv2.imread(str(ip))
        h0, w0 = frame0.shape[:2]
        vis = k[:, 2] > 0
        kp0 = k[:, :2] * [w0, h0]
        for scale in (0.5, 1.0, 2.0):
            frame = frame0 if scale == 1.0 else cv2.resize(
                frame0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            h, w = frame.shape[:2]
            hsv_fixed = dilate(mask_hsv(frame), 2)
            th = tophat_response(frame)
            thr = int(vname.replace("tophat", "")) if vname.startswith("tophat") else None

            def chosen_mask(tol):
                base = (th >= thr).astype(np.uint8) if thr is not None else mask_hsv(frame)
                return dilate(base, tol)

            masks = {"fixed": chosen_mask(2), "scaled": chosen_mask(tol_of("scaled", w))}
            for sigma in (0.0, 1.0, 2.0):
                noise = rng.normal(0, sigma * w / 640, kp0.shape)
                kp = kp0 * scale + noise
                for cfg in ("mevcut", "önerilen_sabit_px", "önerilen_ölçekli"):
                    scaled = cfg == "önerilen_ölçekli"
                    ransac = 8.0 * w / 1280 if scaled else 8.0
                    rmse_max = 15.0 * w / 1280 if scaled else 15.0
                    res = fit_homography(kp[vis], P.vertices[vis], ransac)
                    fit_ok = res is not None and res.reproj_rmse_px <= rmse_max
                    rec = dict(split=split, image=ip.name, scale=scale, width=w, sigma640=sigma, cfg=cfg, fit_ok=fit_ok)
                    if fit_ok:
                        tol = tol_of("scaled", w) if scaled else 2
                        md = hsv_fixed if cfg == "mevcut" else masks["scaled" if scaled else "fixed"]
                        accs, s_true = [], None
                        for d in [(0.0, 0.0)] + GRID:
                            r = shifted(res, d)
                            geo = is_plausible(r, w, h, P.length, P.width)
                            if cfg == "mevcut":
                                s, m = score(md, r.H_pitch2img, (0, 0), tol), float("nan")
                                acc = geo and not (np.isfinite(s) and s < 0.15)          # NaN kabul (bugünkü kod)
                            else:
                                s, m = score_and_margin(md, res.H_pitch2img, d, tol)
                                acc = geo and np.isfinite(s) and s >= tau and np.isfinite(m) and m >= mu
                            if d == (0.0, 0.0):
                                s_true, rec["accept_true"] = s, bool(acc)
                            else:
                                accs.append(bool(acc))
                        rec.update(s_true=s_true, far_k=int(sum(accs)), far_n=len(accs))
                    rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- grafik
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_hists(sc, panels, col, thresholds, path, title, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), facecolor=SURF, sharex=True)
    lo = np.nanmin(sc.loc[sc.variant != "pipeline", col]) if col == "margin" else 0.0
    bins = np.linspace(min(lo, 0.0), 1.0, 51)
    for ax, (vname, tk, label) in zip(axes.ravel(), panels):
        _style(ax)
        g = sc[(sc.variant == vname) & (sc.tol == tk) & (sc.split != "train")]
        for kind, color, name in (("correct", C_TRUE, "doğru H"), ("grid", C_SHIFT, "kaydırılmış H (±3/5/10 m)")):
            v = g.loc[(g.kind == kind) & g.geo.astype(bool), col].dropna()
            wts = np.full(len(v), 1.0 / max(len(v), 1))
            ax.hist(v, bins=bins, weights=wts, histtype="step", linewidth=2, color=color, label=f"{name} (n={len(v)})")
        t = thresholds.get((vname, tk))
        if t is not None:
            ax.axvline(t, color=MUTED, linewidth=1.2)
            ax.text(t, ax.get_ylim()[1] * 0.97, f" eşik {t:.2f}", color=INK2, fontsize=8, va="top")
        ax.set_title(label, color=INK, fontsize=10, loc="left")
    for ax in axes[1]:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("oran (grup içinde)", color=INK2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()          # dört panelde seriler aynı -> tek lejant
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def plot_scatter(sc, vname, tk, tau, mu, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURF)
    _style(ax)
    ax.grid(axis="x", color=GRIDC, linewidth=0.6)
    g = sc[(sc.variant == vname) & (sc.tol == tk) & (sc.split != "train") & sc.geo.astype(bool)]
    for kind, color, name, z in (("grid", C_SHIFT, "kaydırılmış H", 1), ("correct", C_TRUE, "doğru H", 2)):
        v = g[g.kind == kind]
        ax.scatter(v.s, v.margin, s=18, color=color, edgecolors=SURF, linewidths=1.0, label=f"{name} (n={len(v)})", zorder=z)
    ax.axvline(tau, color=MUTED, linewidth=1.2)
    ax.axhline(mu, color=MUTED, linewidth=1.2)
    ax.text(tau, ax.get_ylim()[1], f" tau={tau:.2f}", color=INK2, fontsize=8, va="top")
    ax.text(ax.get_xlim()[1], mu, f"mu={mu:.2f} ", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("çizgi skoru s(H)", color=INK2, fontsize=9)
    ax.set_ylabel("marj  s(H) - max s(probe)", color=INK2, fontsize=9)
    ax.set_title(f"{vname} / tol {tk}: kabul bölgesi sağ üst", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------- Faz A: gürültü çalışması
NOISE_SIGMAS = (0.0, 1.0, 2.0)          # 640 girişte px; görüntüde sigma * W / 640
NOISE_DRAWS = 5                         # sigma > 0 için görüntü başına gürültü çekilişi
SCORE_MODES = (0.75, 0.5, None)         # None = skor eşiği yok (geometri + marj)
TOL_DIVS = (640.0, 427.0, 320.0)
NOISE_THRS = (5.0, 7.0, 10.0)
BASE_STRUCT = (0.75, 640.0)             # Faz K önerisi
NOISE_MU = 0.12                         # marj eşiği tüm hücrelerde sabit (Faz K değeri)
MIN_GAIN = 0.05                         # "belirgin": valid'de sigma=1 FRR'de en az 5 puan düşüş


def noise_study(items, out: Path, log, seed: int = 0):
    """Gürültü altında red/kabul: sigma x skor eşiği x tolerans x top-hat eşiği. Kütüphane fonksiyonlarıyla."""
    from fai.homography import judge_lines, line_evidence, line_mask, pitch_line_samples, probe_offsets, LineCheck
    rng = np.random.default_rng(seed)
    samples = pitch_line_samples(P.polylines(), 0.5)
    offsets = probe_offsets((2.0, 4.0, 7.0))
    rows, checked = [], False
    for n_img, (split, ip, k) in enumerate(items):
        frame = cv2.imread(str(ip))
        h, w = frame.shape[:2]
        vis = k[:, 2] > 0
        kp0 = k[:, :2] * [w, h]
        th = tophat_response(frame)
        masks = {}
        for thr in NOISE_THRS:
            binm = (th >= thr).astype(np.uint8)
            for td in TOL_DIVS:
                tol = max(1, int(round(w / td)))
                masks[(thr, td)] = (dilate(binm, tol), tol)
        if not checked:                                   # buradaki maske == kütüphanenin line_mask'i
            for (thr, td), (m_, tol) in masks.items():
                lm, ltol = line_mask(frame, LineCheck(tophat_thr=thr, tol_div=td))
                assert ltol == tol and np.array_equal(lm, m_), (thr, td)
            checked = True
        for sigma in NOISE_SIGMAS:
            for draw in range(1 if sigma == 0 else NOISE_DRAWS):
                kp = kp0 + rng.normal(0, sigma * w / 640, kp0.shape)
                res = fit_homography(kp[vis], P.vertices[vis], 8.0 * w / 1280)
                base = dict(split=split, image=ip.name, sigma=sigma, draw=draw)
                if res is None or res.reproj_rmse_px > 15.0 * w / 1280:
                    rows.append(dict(base, thr=np.nan, tol_div=np.nan, kind="correct", fit_ok=False, geo=False,
                                     s=np.nan, m=np.nan))
                    continue
                geo = {d: is_plausible(shifted(res, d) if d != (0.0, 0.0) else res, w, h, P.length, P.width)
                       for d in [(0.0, 0.0)] + GRID}
                for (thr, td), (m_, tol) in masks.items():
                    for d in [(0.0, 0.0)] + GRID:
                        s, mg = line_evidence(m_, tol, res.H_pitch2img, samples, offsets + np.asarray(d), 30)
                        rows.append(dict(base, thr=thr, tol_div=td, kind="correct" if d == (0.0, 0.0) else "shift",
                                         fit_ok=True, geo=geo[d], s=s, m=mg))
        if (n_img + 1) % 10 == 0:
            log(f"  gürültü: {n_img + 1}/{len(items)} görüntü")
    df = pd.DataFrame(rows)

    def reasons(g: pd.DataFrame, min_score) -> pd.Series:
        r = pd.Series("", index=g.index, dtype=object)
        chk = LineCheck(min_score=min_score, min_margin=NOISE_MU)
        lr = [judge_lines(s, m, chk) for s, m in zip(g["s"], g["m"])]
        r[:] = lr
        r[~g["geo"].astype(bool)] = "geometry"
        r[~g["fit_ok"].astype(bool)] = "fit"
        return r

    def cell(split, sigma, thr, td, min_score):
        g = df[(df.split == split) & (df.sigma == sigma)]
        failed_fit = g[~g.fit_ok.astype(bool)]
        g = g[(g.thr == thr) & (g.tol_div == td)]
        c = pd.concat([g[g.kind == "correct"], failed_fit])
        sh = g[g.kind == "shift"]
        rc, rs = reasons(c, min_score), reasons(sh, min_score)
        out = {"FRR": rate(rc != ""), "FAR": rate(rs == "")}
        for why in ("fit", "geometry", "score", "margin"):
            out[f"rej_{why}"] = float((rc == why).mean()) if len(rc) else float("nan")
        ok_fit = c[c.fit_ok.astype(bool)]
        out["fail_score_alone"] = float((~np.isfinite(ok_fit.s) | (ok_fit.s < (min_score or -1))).mean()) if len(ok_fit) else np.nan
        out["fail_margin_alone"] = float((~np.isfinite(ok_fit.m) | (ok_fit.m < NOISE_MU)).mean()) if len(ok_fit) else np.nan
        out["fail_geo_alone"] = float((~ok_fit.geo.astype(bool)).mean()) if len(ok_fit) else np.nan
        return out

    def key(ms):
        return "yok" if ms is None else f"{ms:.2f}"

    tab = {}
    for split in ("valid", "test"):
        for sigma in NOISE_SIGMAS:
            for thr in NOISE_THRS:
                for td in TOL_DIVS:
                    for ms in SCORE_MODES:
                        tab[(split, sigma, thr, td, key(ms))] = cell(split, sigma, thr, td, ms)

    # --- madde 3: yapı kararı (sadece valid, thr=10)
    def v(sigma, td, ms, metric, thr=10.0):
        return tab[("valid", sigma, thr, td, key(ms))][metric]["rate"]

    b_ms, b_td = BASE_STRUCT
    cands = []
    for td in TOL_DIVS:
        for ms in SCORE_MODES:
            if (ms, td) == (b_ms, b_td):
                continue
            ok = (v(0.0, td, ms, "FAR") == 0.0
                  and v(1.0, td, ms, "FAR") <= v(1.0, b_td, b_ms, "FAR") + 0.01
                  and v(2.0, td, ms, "FAR") <= v(2.0, b_td, b_ms, "FAR") + 0.01
                  and v(1.0, td, ms, "FRR") <= v(1.0, b_td, b_ms, "FRR") - MIN_GAIN
                  and v(2.0, td, ms, "FRR") <= v(2.0, b_td, b_ms, "FRR")
                  and v(0.0, td, ms, "FRR") <= v(0.0, b_td, b_ms, "FRR"))
            if ok:
                cands.append((v(1.0, td, ms, "FRR"), v(2.0, td, ms, "FRR"), ms is not None, td, ms))
    chosen_ms, chosen_td = (min(cands)[4], min(cands)[3]) if cands else (b_ms, b_td)

    # --- madde 4: top-hat eşiği (sadece valid, seçilen yapıyla)
    def vt(thr, sigma, metric):
        return tab[("valid", sigma, thr, chosen_td, key(chosen_ms))][metric]["rate"]

    thr_cands = [(vt(t, 1.0, "FRR"), t) for t in (5.0, 7.0)
                 if vt(t, 0.0, "FAR") == 0.0 and vt(t, 1.0, "FRR") <= vt(10.0, 1.0, "FRR") - MIN_GAIN
                 and vt(t, 0.0, "FRR") <= vt(10.0, 0.0, "FRR")]
    chosen_thr = min(thr_cands)[1] if thr_cands else 10.0

    pay = {}                                               # Faz K tarzı ayrışma payı (valid, sigma=0, tol seçilen)
    for thr in NOISE_THRS:
        g = df[(df.split == "valid") & (df.sigma == 0) & (df.thr == thr) & (df.tol_div == chosen_td) & df.geo.astype(bool)]
        pay[thr] = float(np.nanquantile(g[g.kind == "correct"].m, 0.04) - np.nanquantile(g[g.kind == "shift"].m, 0.99))

    decision = {"score_threshold": chosen_ms, "tol_div": chosen_td, "tophat_thr": chosen_thr, "min_margin": NOISE_MU,
                "structure_candidates": [{"tol_div": c[3], "score": c[4], "valid_FRR_s1": c[0], "valid_FRR_s2": c[1]} for c in cands],
                "tophat_candidates": [{"thr": t, "valid_FRR_s1": f} for f, t in thr_cands]}
    df.to_csv(out / "noise.csv", index=False)
    return tab, decision, pay


def perf_study(items, log, n_rep: int = 3):
    """1080p'de çizgi kontrolünün frame başına süresi (kütüphanenin HomographyTracker yolu)."""
    from fai.homography import line_evidence, line_mask
    from fai.pipeline import PipelineConfig, build_htracker
    cfg = PipelineConfig()
    chk = cfg.line_check()
    ht = build_htracker(P, cfg)
    t_mask, t_ev, t_upd = [], [], []
    for split, ip, k in items:
        frame = cv2.imread(str(ip))
        h, w = frame.shape[:2]
        vis = k[:, 2] > 0
        kp = np.where(vis[:, None], k[:, :2] * [w, h], 0.0)
        res = fit_homography(kp[vis], P.vertices[vis], 8.0 * w / 1280)
        for _ in range(n_rep):
            t0 = time.perf_counter(); mt = line_mask(frame, chk); t1 = time.perf_counter()
            line_evidence(mt[0], mt[1], res.H_pitch2img, ht._samples, ht._offsets, chk.min_samples); t2 = time.perf_counter()
            ht.reset(); ht.update(kp, vis.astype(float), frame); t3 = time.perf_counter()
            t_mask.append(t1 - t0); t_ev.append(t2 - t1); t_upd.append(t3 - t2)
    ms = lambda a: {"median": round(1e3 * float(np.median(a)), 2), "p90": round(1e3 * float(np.percentile(a, 90)), 2)}  # noqa: E731
    return {"width": w, "height": h, "mask_ms": ms(t_mask), "evidence_13H_ms": ms(t_ev),
            "update_total_ms": ms(t_upd), "n": len(t_upd)}


def noise_report(tab, decision, pay, perf, out: Path):
    L = ["# Faz A — gürültü altında çizgi kontrolü (field v12, valid + test)\n",
         f"Keypoint gürültüsü sigma = {', '.join(str(s) for s in NOISE_SIGMAS)} px (640 girişte; 1920 px'te x3). "
         f"sigma > 0 için görüntü başına {NOISE_DRAWS} çekiliş. H gürültülü keypoint'lerden RANSAC (8·W/1280 px) ile fit edilir; "
         "rmse > 15·W/1280 ise 'fit' reddi. FAR: aynı gürültülü H'nin ±3/5/10 m x/y kaymaları. Marj eşiği her hücrede "
         f"{NOISE_MU}. Kararlar sadece **valid**'den, önceden yazılı kurallarla; test sadece rapor.\n",
         "## 1. Reddin kritere göre dağılımı (top-hat 10)\n",
         "Sıralı atama: fit/rmse → geometri → skor → marj (frames.parquet'teki reject_reason ile aynı). "
         "'tek başına' sütunları: fit'i geçen doğru H'lerde o kriterin kendi başına red oranı (örtüşür).\n",
         "| yapı | split | sigma | FRR | fit/rmse | geometri | skor | marj | skor tek başına | marj tek başına |\n|---|---|---|---|---|---|---|---|---|---|"]
    structs = [("Faz K: skor 0.75, tol W/640", 0.75, 640.0),
               (f"seçilen: skor {'yok' if decision['score_threshold'] is None else decision['score_threshold']}, tol W/{decision['tol_div']:.0f}",
                decision["score_threshold"], decision["tol_div"])]
    k = lambda ms: "yok" if ms is None else f"{ms:.2f}"  # noqa: E731
    for name, ms, td in structs:
        for split in ("valid", "test"):
            for s in NOISE_SIGMAS:
                c = tab[(split, s, 10.0, td, k(ms))]
                L.append(f"| {name} | {split} | {s:.0f} | %{100 * c['FRR']['rate']:.1f} | %{100 * c['rej_fit']:.1f} | "
                         f"%{100 * c['rej_geometry']:.1f} | %{100 * c['rej_score']:.1f} | %{100 * c['rej_margin']:.1f} | "
                         f"%{100 * c['fail_score_alone']:.1f} | %{100 * c['fail_margin_alone']:.1f} |")
    L.append("\n## 2. Skor eşiği x tolerans x gürültü (top-hat 10) — hücre: doğru H reddi / kaydırılmış H kabulü\n")
    L.append("| skor eşiği | tol | sigma=0 valid | sigma=0 test | sigma=1 valid | sigma=1 test | sigma=2 valid | sigma=2 test |\n|---|---|---|---|---|---|---|---|")
    pc = lambda r: f"%{100 * r['rate']:.1f}"  # noqa: E731
    for ms in SCORE_MODES:
        for td in TOL_DIVS:
            cells = []
            for s in NOISE_SIGMAS:
                for split in ("valid", "test"):
                    c = tab[(split, s, 10.0, td, k(ms))]
                    cells.append(f"{pc(c['FRR'])} / {pc(c['FAR'])}")
            L.append(f"| {k(ms)} | W/{td:.0f} | " + " | ".join(cells) + " |")
    n_c = tab[("valid", 1.0, 10.0, 640.0, "0.75")]["FRR"]["n"]
    L.append(f"\n(valid hücre başına doğru H sayısı: sigma=0 → 30, sigma>0 → {n_c}; kaydırılmış H 12 katı.)\n")
    L.append("## 3. Yapı kararı (madde 3)\n")
    L.append(f"Kural (valid, top-hat 10): sigma=0 FAR %0; sigma=1 ve 2'de FAR baz çizginin en fazla 1 puan üstü; sigma=1 FRR "
             f"baz çizgiden (skor 0.75, W/640) en az {100 * MIN_GAIN:.0f} puan düşük; sigma=2 ve sigma=0 FRR kötüleşmiyor. "
             "Uygunlar arasında sigma=1 FRR en düşük, sonra sigma=2 FRR, sonra skor eşiği olmayan, sonra küçük tolerans.\n")
    if decision["structure_candidates"]:
        L.append("Uygun adaylar: " + "; ".join(f"skor {k(c['score'])}, W/{c['tol_div']:.0f} (sigma=1 FRR %{100 * c['valid_FRR_s1']:.1f}, "
                                               f"sigma=2 %{100 * c['valid_FRR_s2']:.1f})" for c in decision["structure_candidates"]))
    else:
        L.append("Uygun aday yok.")
    L.append(f"\n**Karar: skor eşiği {k(decision['score_threshold'])}, tolerans W/{decision['tol_div']:.0f}.**\n")
    L.append("## 4. Top-hat eşiği (madde 4) — seçilen yapıyla\n")
    L.append("| top-hat eşiği | valid pay (sigma=0) | " + " | ".join(f"sigma={s:.0f} valid / test (FRR / FAR)" for s in NOISE_SIGMAS) + " |\n|---|---|---|---|---|")
    for thr in NOISE_THRS:
        cells = []
        for s in NOISE_SIGMAS:
            cv_, ct = (tab[(sp, s, thr, decision["tol_div"], k(decision["score_threshold"]))] for sp in ("valid", "test"))
            cells.append(f"{pc(cv_['FRR'])} / {pc(cv_['FAR'])} · {pc(ct['FRR'])} / {pc(ct['FAR'])}")
        L.append(f"| {thr:.0f} | {pay[thr]:+.3f} | " + " | ".join(cells) + " |")
    L.append(f"\nKural: 5 ya da 7, valid'de sigma=0 FAR %0 tutup sigma=1 FRR'yi 10'a göre en az {100 * MIN_GAIN:.0f} puan düşürürse "
             f"(sigma=0 FRR kötüleşmeden) seçilir; yoksa 10. **Karar: {decision['tophat_thr']:.0f}.**\n")
    L.append("## 5. Performans (madde 6) — 1080p, kütüphanenin HomographyTracker yolu\n")
    L.append(f"Maske frame başına bir kez hesaplanır; skor + 12 probe aynı maskeyi kullanır ve tek perspectiveTransform çağrısında "
             f"projekte edilir. n={perf['n']} ölçüm ({perf['width']}x{perf['height']}).\n")
    L.append("| adım | medyan ms | p90 ms |\n|---|---|---|")
    for name, kk in (("çizgi maskesi (LAB + top-hat + dilate)", "mask_ms"), ("skor + 12 probe (1 H)", "evidence_13H_ms"),
                     ("HomographyTracker.update toplam (fit + geometri + maske + kanıt)", "update_total_ms")):
        L.append(f"| {name} | {perf[kk]['median']} | {perf[kk]['p90']} |")
    (out / "noise.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow/field-v12")
    ap.add_argument("--out", default=None, help="varsayılan outputs/calibration/<dataset klasörü>")
    ap.add_argument("--splits", nargs="+", default=["valid", "test", "train"])
    ap.add_argument("--no-resolution", action="store_true")
    ap.add_argument("--noise", action="store_true", help="Faz A gürültü çalışması + performans (noise.md)")
    ap.add_argument("--noise-only", action="store_true", help="sadece gürültü çalışması")
    a = ap.parse_args()
    root = Path(a.data)
    out = Path(a.out or f"outputs/calibration/{root.name}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731

    if a.noise or a.noise_only:
        vt = load_items(root, ["valid", "test"])
        log(f"[gürültü] {len(vt)} görüntü (valid + test)")
        tab, decision, pay = noise_study(vt, out, log)
        perf = perf_study([x for x in vt if x[0] == "valid"], log)
        noise_report(tab, decision, pay, perf, out)
        (out / "noise.json").write_text(json.dumps({"decision": decision, "pay": pay, "perf": perf,
                                                    "table": {"|".join(map(str, kk)): vv for kk, vv in tab.items()}},
                                                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"[gürültü] karar: {decision['score_threshold']=}, {decision['tol_div']=}, {decision['tophat_thr']=}; perf {perf}")
        if a.noise_only:
            return

    items = load_items(root, a.splits)
    log(f"[veri] {len(items)} görüntü: " + ", ".join(f"{s}={sum(1 for x in items if x[0] == s)}" for s in a.splits))
    img, sc, max_diff = evaluate_main(items, log)
    img.to_csv(out / "per_image.csv", index=False)
    sc.to_csv(out / "scores.csv", index=False)
    log(f"[skor] kütüphane line_evidence ile fark (top-hat 10, tol W/640; skor+marj): max {max_diff:.2e}")

    # 1) GT fit + geometri
    summ = {"dataset": str(root), "n_images": {s: int((img.split == s).sum()) for s in a.splits},
            "score_impl_max_abs_diff_vs_library": max_diff, "settings": {
                "grid_shifts_m": GRID, "diag_shifts_m": DIAG, "probes_m": PROBES, "tophat_thr": TOPHAT_THR,
                "q_ruleA": Q_A, "q_ruleB": Q_B, "targets": {"FAR": TARGET_FAR, "FRR": TARGET_FRR, "geo": TARGET_GEO}}}
    fit = img[img.fit_ok]
    summ["gt_fit"] = {s: {"fit_ok": rate(img[img.split == s].fit_ok),
                          "geo_ok": rate(fit[fit.split == s].geo_ok.astype(bool)),
                          "rmse_px_median": float(fit[fit.split == s].rmse_px.median()),
                          "rmse_px_p95": float(fit[fit.split == s].rmse_px.quantile(.95)),
                          "inlier_ratio_median": float((fit[fit.split == s].n_inliers / fit[fit.split == s].n_kp).median())}
                      for s in a.splits}
    geo_fail = fit[~fit.geo_ok.astype(bool)][["split", "image", "n_kp", "n_inliers", "rmse_px"]]

    # 2) bugünkü kod
    pipe = sc[sc.variant == "pipeline"]
    summ["pipeline_today"] = {s: {"FRR": rate(~fit[fit.split == s].pipe_accept.astype(bool)),
                                  "FAR_grid": rate(pipe[(pipe.split == s) & (pipe.kind == "grid")]["pipe_acc"].astype(bool)),
                                  "FAR_diag": rate(pipe[(pipe.split == s) & (pipe.kind == "diag")]["pipe_acc"].astype(bool))}
                              for s in a.splits}
    pipe_break = []
    for axis, col, other in (("x", "dx", "dy"), ("y", "dy", "dx")):
        for m in (3, 5, 10):
            sel = pipe[(pipe.kind == "grid") & (pipe[col].abs() == m) & (pipe[other] == 0) & (pipe.split != "train")]
            pipe_break.append((axis, m, rate(sel["pipe_acc"].astype(bool))))

    # 3) varyantlar
    recs = choose_and_score(sc)
    summ["variants"] = recs

    def key(r):
        # valid'de en düşük FAR (kural B); eşitlikte en büyük ayrışma payı (tek görüntülük FRR farkı 30 görüntüde
        # anlamsız, pay ise gerçek dünyadaki toleransı gösterir); sonra ölçekli tolerans (#10).
        return (r["B_valid_FAR"]["rate"], -r["sep_valid"], r["tol"] != "scaled")

    best = min(recs, key=key)
    best_hsv = min((r for r in recs if r["variant"] == "hsv"), key=key)
    best_th = min((r for r in recs if r["variant"].startswith("tophat")), key=key)
    chosen = {"variant": best["variant"], "tol": best["tol"], "tau": best["tauB"], "mu": best["muB"]}
    summ["chosen"] = chosen
    ok = {s: bool(best[f"B_{s}_FAR"]["rate"] < TARGET_FAR and best[f"B_{s}_FRR"]["rate"] < TARGET_FRR)
          for s in a.splits if f"B_{s}_FAR" in best}
    geo_all = rate(fit[fit.split.isin(["valid", "test"])].geo_ok.astype(bool))
    summ["targets_met"] = {"geo_valid_test": bool(geo_all["rate"] >= TARGET_GEO), **{f"rule_B_{s}": v for s, v in ok.items()},
                           **{f"pipeline_{s}": bool(p["FAR_grid"]["rate"] < TARGET_FAR and p["FRR"]["rate"] < TARGET_FRR)
                              for s, p in summ["pipeline_today"].items()}}
    from fai.pipeline import PipelineConfig
    _c = PipelineConfig()
    pipe_desc = (f"top-hat {_c.line_tophat_thr:g}, çekirdek W/{_c.line_kernel_div:g}, tol W/{_c.line_tol_div:g}, "
                 f"skor eşiği {'yok' if _c.min_line_score is None else _c.min_line_score}, marj ≥ {_c.min_line_margin}, "
                 f"probe ±{'/'.join(f'{p:g}' for p in _c.line_probes_m)} m, RANSAC {_c.ransac_px:g}·W/{_c.ref_width:g} px, NaN = red")
    breakdown = far_breakdown(sc, best["variant"], best["tol"], best["tauB"], best["muB"], ("test",))

    # 4) çözünürlük
    res_tab = None
    if not a.no_resolution:
        log("[çözünürlük] 0.5x / 1x / 2x ölçek, sigma640 = 0/1/2 px")
        rv = evaluate_resolution([x for x in items if x[0] in ("valid", "test")], chosen, log=log)
        rv.to_csv(out / "resolution.csv", index=False)
        g = rv.groupby(["cfg", "scale", "width", "sigma640"])
        res_tab = pd.DataFrame({
            "fit_ok": g.fit_ok.mean(),
            "kabul_dogru": g.accept_true.apply(lambda s: s.fillna(False).astype(bool).mean()),
            "s_dogru_medyan": g.s_true.median(),
            "FAR": g.far_k.sum() / g.far_n.sum(),
        }).reset_index()
        summ["resolution"] = res_tab.to_dict("records")

    # 5) grafikler
    panels = [("hsv", "fixed", "HSV maske, tol 2 px (Faz A öncesi kod)"), ("hsv", "scaled", "HSV maske, tol ölçekli"),
              (best_th["variant"], "fixed", f"{best_th['variant']}, tol 2 px"),
              (best_th["variant"], "scaled", f"{best_th['variant']}, tol ölçekli")]
    thr_s = {(r["variant"], r["tol"]): r["tauB"] for r in recs}
    thr_m = {(r["variant"], r["tol"]): r["muB"] for r in recs}
    thr_s[("hsv", "fixed")] = 0.15
    plot_hists(sc, panels, "s", thr_s, out / "fig_score_hist.png",
               "Çizgi skoru: doğru H vs kaydırılmış H (valid+test; geometriden geçenler)", "çizgi skoru s(H)")
    plot_hists(sc, panels, "margin", thr_m, out / "fig_margin_hist.png",
               "Marj: s(H) - max s(±2/4/7 m probe) (valid+test)", "marj")
    plot_scatter(sc, best["variant"], best["tol"], best["tauB"], best["muB"], out / "fig_score_vs_margin.png")

    # 6) rapor
    L = []
    L.append(f"# Faz K — homography kalibrasyonu: {root.name}\n")
    L.append(f"Görüntü: " + ", ".join(f"{s} {summ['n_images'][s]}" for s in a.splits)
             + f". Eşikler sadece **valid**'den seçildi; test ve train seçimde kullanılmadı. Süre {time.time() - t0:.0f} s.\n")
    L.append("## 1. GT homography ve geometri (is_plausible)\n")
    L.append("| split | fit | geometri geçti | rmse px (medyan / p95) | inlier oranı (medyan) |\n|---|---|---|---|---|")
    for s in a.splits:
        f = summ["gt_fit"][s]
        L.append(f"| {s} | {fmt_rate(f['fit_ok'])} | {fmt_rate(f['geo_ok'])} | {f['rmse_px_median']:.1f} / {f['rmse_px_p95']:.1f} | {f['inlier_ratio_median']:.2f} |")
    L.append(f"\nvalid+test geometri: {fmt_rate(geo_all)} — hedef ≥%{100 * TARGET_GEO:.0f}: "
             f"{'TUTTU' if summ['targets_met']['geo_valid_test'] else 'TUTMADI'}.")
    if len(geo_fail):
        L.append("\nGeometriden kalanlar:\n\n" + md_table(geo_fail, 1))
    L.append("\n## 2. Kütüphanenin kabul zinciri (HomographyTracker.update, PipelineConfig varsayılanları)\n")
    L.append(f"Ayarlar: {pipe_desc}. GT keypoint'leri ve aynı keypoint'lerin tutarlı kaydırılmış halleri pipeline'a verilir.\n")
    L.append("| split | doğru H reddi (FRR) | kaydırılmış H kabulü (FAR, ±3/5/10 m) | çapraz ±7 m kabulü | hedef (FAR<%5, FRR<%10) |\n|---|---|---|---|---|")
    for s in a.splits:
        p = summ["pipeline_today"][s]
        L.append(f"| {s} | {fmt_rate(p['FRR'])} | {fmt_rate(p['FAR_grid'])} | {fmt_rate(p['FAR_diag'])} | "
                 f"{'TUTTU' if summ['targets_met'][f'pipeline_{s}'] else 'TUTMADI'} |")
    L.append("\nKaydırılmış H kabulü, eksen ve büyüklüğe göre (valid+test):\n\n| eksen | kayma | kabul |\n|---|---|---|")
    for axis, m, r in pipe_break:
        L.append(f"| {axis} | ±{m} m | {fmt_rate(r)} |")
    L.append("\n## 3. Maske / tolerans / kural karşılaştırması\n")
    L.append(f"Kural A: geometri ∧ s ≥ tau (tau = valid doğru-H skorunun %{100 * Q_A:.0f} quantile'ı). "
             f"Kural B: geometri ∧ s ≥ tau ∧ marj ≥ mu (ikisi de %{100 * Q_B:.0f} quantile). NaN = red. "
             f"FAR: ±3/5/10 m kaymalar; seçim valid'de, test seçimde kullanılmadı.\n")
    L.append("Seçim: valid'de kural B FAR'ı en düşük olan; eşitlikte valid ayrışma payı (doğru-H marjının %4 quantile'ı − "
             "kaydırılmış-H marjının %99 quantile'ı) en büyük olan; sonra ölçekli tolerans. Test sütunları seçimde kullanılmadı.\n")
    L.append("| maske | tol | s medyan doğru / kaymış | A: tau | A valid FRR / FAR | A test FRR / FAR | B: tau, mu | valid pay | B valid FRR / FAR | B test FRR / FAR |\n|---|---|---|---|---|---|---|---|---|---|")
    short = lambda r: f"%{100 * r['rate']:.0f}"  # noqa: E731
    for r in sorted(recs, key=lambda r: (r["variant"], r["tol"])):
        L.append(f"| {r['variant']} | {r['tol']} | {r['med_true']:.2f} / {r['med_shift']:.2f} | {r['tauA']:.2f} | "
                 f"{short(r['A_valid_FRR'])} / {short(r['A_valid_FAR'])} | {short(r['A_test_FRR'])} / {short(r['A_test_FAR'])} | "
                 f"{r['tauB']:.2f}, {r['muB']:.2f} | {r['sep_valid']:+.3f} | {short(r['B_valid_FRR'])} / {short(r['B_valid_FAR'])} | "
                 f"{short(r['B_test_FRR'])} / {short(r['B_test_FAR'])} |")
    L.append(f"\n## 4. Seçilen ayar: {best['variant']}, tol {best['tol']}, kural B (tau={best['tauB']:.2f}, mu={best['muB']:.2f})\n")
    L.append("| split | doğru H reddi (FRR) | kaydırılmış H kabulü (FAR) | çapraz ±7 m kabulü | hedef (FAR<%5, FRR<%10) |\n|---|---|---|---|---|")
    for s in a.splits:
        if f"B_{s}_FAR" in best:
            L.append(f"| {s}{' (seçim)' if s == 'valid' else ' (ek kontrol)' if s == 'train' else ''} | {fmt_rate(best[f'B_{s}_FRR'])} | "
                     f"{fmt_rate(best[f'B_{s}_FAR'])} | {fmt_rate(best[f'B_{s}_FAR_diag'])} | {'TUTTU' if ok[s] else 'TUTMADI'} |")
    L.append("\nTest split'inde kaydırılmış H kabulü, eksen ve büyüklüğe göre:\n\n| eksen | kayma | kabul |\n|---|---|---|")
    for axis, m, r in breakdown:
        L.append(f"| {axis} | ±{m} m | {fmt_rate(r)} |")
    L.append(f"\nKarşılaştırma için en iyi HSV: tol {best_hsv['tol']}, B test FRR {short(best_hsv['B_test_FRR'])} / FAR {short(best_hsv['B_test_FAR'])}; "
             f"en iyi top-hat: {best_th['variant']} tol {best_th['tol']}, B test FRR {short(best_th['B_test_FRR'])} / FAR {short(best_th['B_test_FAR'])}.")
    if res_tab is not None:
        L.append("\n## 5. Çözünürlük (#10) — valid+test, keypoint gürültüsü sigma640 px (640 girişte; görüntüde sigma*W/640)\n")
        L.append("`mevcut`: HSV, tol 2 px, RANSAC 8 px, rmse ≤15 px, s≥0.15. `önerilen_sabit_px`: seçilen maske+kural, "
                 "piksel parametreleri sabit. `önerilen_ölçekli`: tol, RANSAC ve rmse eşiği W ile ölçekli. 2x = 3840 px'e "
                 "büyütülmüş 1080p (gerçek 4K değil).\n")
        L.append(md_table(res_tab, 3))
    L.append("\n## Notlar\n")
    L.append("- Kaydırılmış H, keypoint'lerin tutarlı biçimde yanlış çizgiye eşlenmesinin modelidir; gerçek model hatalarının "
             "hepsini temsil etmez (ör. tek keypoint'in yanlış yere kayması). Gerçek model hataları Faz 1'den sonra ölçülmeli.")
    L.append("- Örneklem küçük (valid 30, test 24); GA = %95 Wilson güven aralığı. train split'i ek kontrol: modelden bağımsız "
             "bir ölçüm olduğu için (çizgi/maske), eşik seçiminde kullanılmadığı sürece geçerli bir held-out örneklem.")
    L.append("- GT etiketleri FIFA 105x68 saha ölçüsüyle eşleniyor; gerçek saha ölçüsü farklıysa doğru H skoru biraz düşer.")
    L.append("- Faz A öncesi kabul zincirinin (HSV maske, tol 2 px, skor ≥ 0.15, NaN kabul) aynı ölçümü: FRR valid/test/train "
             "%26.7 / %16.7 / %17.6, FAR %50.3 / %61.5 / %57.5 (git b318bef, docs/calibration/field-v12/report.md).")
    L.append("- Şeffaflık: ilk deneme çalıştırmasında eşitlik bozucu 'valid FRR' idi ve tek görüntülük farkla (1/30 vs 2/30) "
             "tophat40'ı seçti. Bu istatistiksel olarak anlamsız olduğu için kural valid'deki ayrışma payına çevrildi. "
             "Değişiklik test sonuçları görüldükten sonra yapıldı; bu yüzden tüm varyantların test sonuçları 3. tabloda açıkça duruyor.")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"[ok] {out} ({time.time() - t0:.0f} s)")
    log(f"[seçim] {chosen}  hedefler: {summ['targets_met']}")


if __name__ == "__main__":
    main()

"""FAZ 1a — Pose eğitimi için türetilmiş dataset: MAÇ bazında held-out ayır, exclude listesini uygula, flip_idx'i doğrula.

    python scripts/01c_prepare_pose_data.py --data data/roboflow/field-v12

Neden maç bazında: Roboflow field v12 split'leri kare bazında rastgele; 18 maçın hepsi train'de var, valid/test
karelerinin %75-90'ı train'deki kliplerin komşu kareleri. Kare bazında ayrılan bir calib/test yeni bir maç için
iyimser ölçer. Görüntü adı "<maç>_<klip>_<frame>_png...": maç = ilk alan.

Ayırma (sabit seed): aday havuzu <= --pool-max-frames kareli maçlar (büyük maçlar eğitimde kalır); havuz karıştırılır,
maçlar sırayla toplam >= --calib-frames olana kadar calib'e, sonra >= --holdout-frames olana kadar holdout'a gider.
Seçilen maçların train/valid/test'teki BÜTÜN kareleri eğitimden çıkar.

  splits/<dataset>/calib.txt     calib-maçlarının kareleri ("<orijinal split>/<dosya>"); çizgi kontrolü eşikleri burada seçilir
  splits/<dataset>/holdout.txt   holdout-maçlarının kareleri; sadece son kontrol
  splits/<dataset>/exclude.txt   eğitimden ve kalibrasyondan çıkarılan görüntüler (sebebiyle)
Listeler commit'lenir; varsa yeniden üretilmez (sabit kalır).

Türetilmiş dataset (--out; hardlink, orijinal klasöre hiçbir şey yazılmaz, ultralytics cache'leri de buraya):
  train/       eğitim maçlarının orijinal train kareleri
  valid/       eğitim maçlarının orijinal valid kareleri (early stopping; aynı-maç, sadece referans)
  test_same/   eğitim maçlarının orijinal test kareleri (aynı-maç referans)
  calib/, holdout/   yukarıdaki listeler
  data.yaml, prepare.json (sayılar, maçlar, flip_idx doğrulaması)
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.pitch import PitchConfig  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")


def match_of(name: str) -> str:
    return name.split("_")[0]


def read_list(p: Path) -> list:
    if not p.exists():
        return []
    return [l.split("#")[0].strip() for l in p.read_text(encoding="utf-8").splitlines() if l.split("#")[0].strip()]


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow/field-v12")
    ap.add_argument("--splits-dir", default=None, help="varsayılan splits/<dataset klasörü>")
    ap.add_argument("--out", default=None, help="varsayılan data/derived/<dataset klasörü>-1a")
    ap.add_argument("--calib-frames", type=int, default=30, help="hedef; maçlar tam girdiği için aşılabilir")
    ap.add_argument("--holdout-frames", type=int, default=15)
    ap.add_argument("--pool-max-frames", type=int, default=20, help="bundan fazla karesi olan maçlar eğitimde kalır")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.data).resolve()
    sdir = Path(a.splits_dir or f"splits/{root.name}")
    out = Path(a.out or f"data/derived/{root.name}-1a").resolve()
    sdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))

    # 1) flip_idx doğrulaması
    pitch = PitchConfig()
    flip = cfg.get("flip_idx")
    bad = pitch.check_flip_idx(flip) if flip is not None else None
    flip_ok = bad == []
    print(f"[flip_idx] data.yaml: {flip}")
    print(f"[flip_idx] pitch.py aynası: {pitch.mirror_index()}")
    print(f"[flip_idx] {'GEÇTİ' if flip_ok else f'GEÇMEDİ (uyuşmayan indeksler: {bad})'}")

    # 2) kareler, exclude
    frames = {s: sorted(p.name for p in (root / s / "images").iterdir() if p.suffix.lower() in IMG_EXT) for s in SPLITS}
    exclude = set(read_list(sdir / "exclude.txt"))
    missing = [e for e in exclude if not any(e in frames[s] for s in SPLITS)]
    if missing:
        sys.exit(f"exclude.txt'teki dosyalar datasette yok: {missing}")
    allf = [(s, n) for s in SPLITS for n in frames[s] if n not in exclude]
    per_match = Counter(match_of(n) for _, n in allf)

    # 3) maç bazında calib / holdout
    calib_p, hold_p = sdir / "calib.txt", sdir / "holdout.txt"
    if calib_p.exists() and hold_p.exists():
        calib, holdout = read_list(calib_p), read_list(hold_p)
        print(f"[split] mevcut listeler kullanılıyor: {calib_p} ({len(calib)}), {hold_p} ({len(holdout)})")
    else:
        pool = sorted(m for m, c in per_match.items() if c <= a.pool_max_frames)
        order = [str(m) for m in np.random.default_rng(a.seed).permutation(pool)]
        calib_m, hold_m, n_c, n_h = [], [], 0, 0
        for m in order:
            if n_c < a.calib_frames:
                calib_m.append(m); n_c += per_match[m]
            elif n_h < a.holdout_frames:
                hold_m.append(m); n_h += per_match[m]
        if n_c < a.calib_frames or n_h < a.holdout_frames:
            sys.exit(f"havuz yetmedi: calib {n_c}, holdout {n_h} (pool-max-frames'i artır)")
        calib = [f"{s}/{n}" for s, n in allf if match_of(n) in calib_m]
        holdout = [f"{s}/{n}" for s, n in allf if match_of(n) in hold_m]
        hdr = (f"seed={a.seed}, aday havuzu: <= {a.pool_max_frames} kareli maçlar ({len(pool)} maç), exclude hariç. "
               f"Bu maçların BÜTÜN kareleri pose eğitiminden çıkarılır. Satır: <orijinal split>/<dosya>.")
        calib_p.write_text(f"# {root.name} calib-maçları {sorted(calib_m)} ({len(calib)} kare). {hdr}\n"
                           + "\n".join(calib) + "\n", encoding="utf-8")
        hold_p.write_text(f"# {root.name} holdout-maçları {sorted(hold_m)} ({len(holdout)} kare). {hdr}\n"
                          + "\n".join(holdout) + "\n", encoding="utf-8")
        print(f"[split] yeni listeler: calib maçları {sorted(calib_m)} ({len(calib)} kare), "
              f"holdout maçları {sorted(hold_m)} ({len(holdout)} kare)")
    held_matches = {match_of(Path(x).name) for x in calib + holdout}
    if {match_of(Path(x).name) for x in calib} & {match_of(Path(x).name) for x in holdout}:
        sys.exit("calib ile holdout aynı maçı içeriyor")

    # 4) türetilmiş dataset (hardlink)
    def put(entries, dst):
        for s, n in entries:
            link(root / s / "images" / n, out / dst / "images" / n)
            link(root / s / "labels" / f"{Path(n).stem}.txt", out / dst / "labels" / f"{Path(n).stem}.txt")

    keep = [(s, n) for s, n in allf if match_of(n) not in held_matches]
    put([(s, n) for s, n in keep if s == "train"], "train")
    put([(s, n) for s, n in keep if s == "valid"], "valid")
    put([(s, n) for s, n in keep if s == "test"], "test_same")
    put([tuple(x.split("/", 1)) for x in calib], "calib")
    put([tuple(x.split("/", 1)) for x in holdout], "holdout")
    derived = {"path": str(out), "train": "train/images", "val": "valid/images", "test": "test_same/images",
               "kpt_shape": cfg["kpt_shape"], "flip_idx": flip, "names": cfg["names"]}
    (out / "data.yaml").write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
    n = {d: len(list((out / d / "images").iterdir())) if (out / d / "images").exists() else 0
         for d in ("train", "valid", "test_same", "calib", "holdout")}
    train_matches = sorted({match_of(x) for s, x in keep})
    info = {"source": str(root), "lists": {"calib": str(calib_p.resolve()), "holdout": str(hold_p.resolve())},
            "exclude": sorted(exclude), "n": n, "train_matches": train_matches,
            "calib_matches": sorted({match_of(Path(x).name) for x in calib}),
            "holdout_matches": sorted({match_of(Path(x).name) for x in holdout}),
            "flip_idx_ok": flip_ok, "flip_idx_mismatch": bad, "seed": a.seed}
    (out / "prepare.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {out / 'data.yaml'}: {n}; eğitim maçları {len(train_matches)}")


if __name__ == "__main__":
    main()

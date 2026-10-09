"""Eğitim batch'i için VRAM ölçümü: verilen batch'lerle birkaç iterasyon gerçek eğitim, tepe bellek ve hız.

    python scripts/measure_batch_vram.py --model yolo11m-pose.pt --data data/derived/field-v12-1a/data.yaml \
        --imgsz 640 --batches 4 8 16 --out runs/vram_pose.json

Windows'ta (WDDM) VRAM dolunca CUDA sessizce sistem belleğine taşabilir (OOM vermez, yavaşlar). Bu yüzden
"sığıyor" = tepe torch.cuda.max_memory_reserved fiziksel VRAM'in --headroom oranının altında. Taşma ayrıca görüntü
başına sürenin sıçramasıyla görülür (s_per_img). Her batch ayrı süreçte ölçülür (bellek sızıntısı karışmasın).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(a):
    import torch
    from ultralytics import YOLO
    stats = {"t": [], "mem": []}

    def on_batch_end(trainer):
        torch.cuda.synchronize()
        stats["t"].append(time.perf_counter())
        stats["mem"].append(torch.cuda.max_memory_reserved() / 2**30)
        if len(stats["t"]) >= a.iters + 2:
            raise KeyboardInterrupt                          # ölçüm yeterli

    m = YOLO(a.model)
    m.add_callback("on_train_batch_end", on_batch_end)
    try:
        m.train(data=a.data, imgsz=a.imgsz, batch=a.batch, epochs=1, val=False, plots=False, workers=2, device=0,
                project=str(ROOT / "runs" / "_vram"), name=f"b{a.batch}", exist_ok=True, verbose=False, amp=True,
                **({"mosaic": 0.0} if "pose" in a.model else {}))
    except KeyboardInterrupt:
        pass
    t = stats["t"][2:]                                       # ilk iki iterasyon ısınma
    per_img = (t[-1] - t[0]) / max(len(t) - 1, 1) / a.batch if len(t) > 1 else float("nan")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print("RESULT " + json.dumps({"batch": a.batch, "peak_reserved_gb": round(max(stats["mem"]), 2),
                                  "vram_gb": round(total, 2), "s_per_img": round(per_img, 4), "iters": len(t)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--headroom", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=None, help="(iç kullanım: alt süreç)")
    a = ap.parse_args()
    if a.batch is not None:
        return child(a)
    res = []
    for b in a.batches:
        cmd = [sys.executable, __file__, "--model", a.model, "--data", a.data, "--imgsz", str(a.imgsz),
               "--batch", str(b), "--iters", str(a.iters), "--out", a.out]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"batch": b, "error": (p.stderr or p.stdout)[-400:]}
        res.append(r)
        print(r, flush=True)
    ok = [r for r in res if "peak_reserved_gb" in r]
    for r in ok:
        r["fits"] = bool(r["peak_reserved_gb"] < a.headroom * r["vram_gb"])
    best = max((r["batch"] for r in ok if r["fits"]), default=None)
    out = {"model": a.model, "data": a.data, "imgsz": a.imgsz, "results": res, "largest_fitting_batch": best}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] sığan en büyük batch: {best} -> {a.out}")


if __name__ == "__main__":
    main()

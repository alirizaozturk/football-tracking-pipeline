"""GPU sıcaklık bekçisi: eğitim sırasında nvidia-smi'yi periyodik log'lar, termal sorun görürse eğitimi durdurur.

    python scripts/gpu_watchdog.py --pid-file runs/pose-1a/train.pid --log runs/pose-1a/gpu.csv

Durdurma koşulları (süreç ağacı psutil ile sonlandırılır; ultralytics --resume ile son epoch'tan devam edilebilir):
  * hw_thermal_slowdown ya da sw_thermal_slowdown "Active"  -> hemen
  * sıcaklık > --max-temp, art arda --sustain örnek (varsayılan 85 °C, 3 örnek = ~3 dk "sürekli")
Stdout: sadece durum satırları (START / STOP / DONE); her örnek CSV'ye yazılır.
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

FIELDS = ["timestamp", "temperature.gpu", "clocks.sm", "clocks.max.sm", "clocks_event_reasons.hw_thermal_slowdown",
          "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.active", "power.draw", "utilization.gpu",
          "memory.used"]


def query() -> dict:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    return dict(zip(FIELDS, [v.strip() for v in out.split(",")]))


def kill_tree(pid: int) -> int:
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    procs = p.children(recursive=True) + [p]
    for q in procs:
        try:
            q.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=30)
    return len(procs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--sustain", type=int, default=3)
    ap.add_argument("--wait-pid-s", type=float, default=600.0, help="pid dosyası bu kadar sürede oluşmazsa çık")
    a = ap.parse_args()

    pid_file, log = Path(a.pid_file), Path(a.log)
    t0 = time.time()
    while not pid_file.exists():
        if time.time() - t0 > a.wait_pid_s:
            print(f"DONE pid dosyası yok: {pid_file}", flush=True)
            return
        time.sleep(2)
    pid = int(pid_file.read_text().strip())
    print(f"START pid={pid} log={log} max_temp={a.max_temp} sustain={a.sustain}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    hot = 0
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["unix"] + FIELDS + ["hot_streak", "action"])
        if new:
            w.writeheader()
        while True:
            if not psutil.pid_exists(pid):
                print(f"DONE eğitim süreci bitti (pid={pid})", flush=True)
                return
            try:
                q = query()
            except Exception as e:                       # nvidia-smi geçici hatası: logla, devam et
                w.writerow({"unix": round(time.time()), "action": f"query_error:{type(e).__name__}"})
                fh.flush()
                time.sleep(a.interval)
                continue
            temp = float(q["temperature.gpu"])
            hot = hot + 1 if temp > a.max_temp else 0
            thermal = "Active" in (q["clocks_event_reasons.hw_thermal_slowdown"], q["clocks_event_reasons.sw_thermal_slowdown"])
            action = ""
            if thermal:
                action = "STOP thermal_slowdown"
            elif hot >= a.sustain:
                action = f"STOP temp>{a.max_temp:g} x{hot}"
            w.writerow({"unix": round(time.time()), **q, "hot_streak": hot, "action": action})
            fh.flush()
            if action:
                n = kill_tree(pid)
                print(f"{action} temp={temp:g} sm={q['clocks.sm']} hw={q['clocks_event_reasons.hw_thermal_slowdown']} "
                      f"sw={q['clocks_event_reasons.sw_thermal_slowdown']} -> {n} süreç sonlandırıldı", flush=True)
                sys.exit(2)
            time.sleep(a.interval)


if __name__ == "__main__":
    main()

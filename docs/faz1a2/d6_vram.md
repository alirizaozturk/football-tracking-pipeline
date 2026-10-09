# D6 — eğitim batch'i için VRAM (gerçek eğitim iterasyonları, AMP açık)

Sığıyor = tepe torch.cuda.max_memory_reserved < 0.95 × 8 GB. Windows WDDM'de VRAM dolunca CUDA OOM vermez, sistem belleğine taşıp yavaşlar; taşan batch'lerde görüntü başına süre 3-7 kat sıçrıyor.

| model | imgsz | data | batch | tepe VRAM (GB, 8 GB kart) | s / görüntü | sığıyor |
|---|---|---|---|---|---|---|
| yolo11m-pose.pt | 640 | field-v12-1a | 4 | 2.28 | 0.058 | evet |
| yolo11m-pose.pt | 640 | field-v12-1a | 6 | 3.23 | 0.051 | evet |
| yolo11m-pose.pt | 640 | field-v12-1a | 8 | 4.35 | 0.030 | evet |
| yolo11m-pose.pt | 640 | field-v12-1a | 10 | 5.28 | 0.040 | evet |
| yolo11m-pose.pt | 640 | field-v12-1a | 12 | 6.20 | 0.035 | evet |
| yolo11m-pose.pt | 640 | field-v12-1a | 16 | 8.12 | 0.220 | HAYIR (sistem belleğine taşıyor; s/görüntü sıçrıyor) |
| yolo11m.pt | 1280 | ball-v2 | 2 | 4.18 | 0.099 | evet |
| yolo11m.pt | 1280 | ball-v2 | 3 | 6.01 | 0.096 | evet |
| yolo11m.pt | 1280 | ball-v2 | 4 | 8.04 | 0.412 | HAYIR (sistem belleğine taşıyor; s/görüntü sıçrıyor) |
| yolo11m.pt | 1280 | players-v10 | 2 | 4.20 | 0.132 | evet |
| yolo11m.pt | 1280 | players-v10 | 3 | 6.04 | 0.095 | evet |
| yolo11m.pt | 1280 | players-v10 | 4 | 8.07 | 0.475 | HAYIR (sistem belleğine taşıyor; s/görüntü sıçrıyor) |
| yolo11m.pt | 1280 | players-v10 | 6 | 11.92 | 2.537 | HAYIR (sistem belleğine taşıyor; s/görüntü sıçrıyor) |
| yolo11m.pt | 1280 | players-v10 | 8 | 14.04 | 1.284 | HAYIR (sistem belleğine taşıyor; s/görüntü sıçrıyor) |

Sığan en büyük batch: yolo11m-pose.pt @640 (field-v12-1a): 12; yolo11m.pt @1280 (ball-v2): 3; yolo11m.pt @1280 (players-v10): 3

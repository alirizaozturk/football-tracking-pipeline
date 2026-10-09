# Faz 1a-2 — pose teşhisi (yeni eğitim yok)

Model: `runs/pose-1a/weights/best.pt`. Görülmemiş maçlar: calib (36) + holdout (21); aynı-maç: valid + test_same (42). Keypoint hataları 640 ölçeğinde px (× 640 / W); H hatası birincil metrik = ön-kayıtlı ortalama ızgara hatası (m); doğru < 0.5 m, yanlış > 2.0 m (değişmedi).

## D1 — keypoint bazında hata (calib + holdout, model @640, conf ≥ 0.5)

Sistematik = ortalama hata vektörü; rastgele = ortalama çevresindeki RMS; sistematik payı = |sist|² / (|sist|² + RMS²).

| kp | nokta | n | ort. vektör dx | dy | |sistematik| | rastgele RMS | |hata| medyan | sistematik payı |
|---|---|---|---|---|---|---|---|---|
| 21 | sağ ceza sahası | 25 | -14.11 | -1.31 | 14.17 | 32.36 | 30.34 | 0.16 |
| 17 | orta çizgi | 36 | -18.13 | -3.33 | 18.44 | 32.68 | 24.60 | 0.24 |
| 29 | sağ kale çizgisi | 15 | -5.98 | 4.46 | 7.46 | 23.48 | 20.31 | 0.09 |
| 31 | orta yuvarlak sol | 33 | -0.44 | -0.57 | 0.72 | 30.00 | 20.11 | 0.00 |
| 16 | orta çizgi | 48 | -4.28 | 0.99 | 4.39 | 29.43 | 17.44 | 0.02 |
| 14 | orta çizgi | 45 | -0.05 | -1.34 | 1.34 | 31.32 | 16.52 | 0.00 |
| 24 | sağ kale alanı | 23 | -6.94 | -1.82 | 7.18 | 27.33 | 16.40 | 0.06 |
| 20 | sağ ceza sahası | 28 | -7.42 | 0.33 | 7.43 | 19.34 | 16.17 | 0.13 |
| 15 | orta çizgi | 47 | -1.35 | 0.28 | 1.38 | 30.21 | 15.71 | 0.00 |
| 28 | sağ kale çizgisi | 19 | -7.43 | 0.77 | 7.47 | 32.57 | 15.44 | 0.05 |
| 22 | sağ penaltı | 28 | -5.05 | 0.64 | 5.09 | 15.75 | 11.97 | 0.09 |
| 10 | sol ceza sahası | 23 | -5.78 | -0.43 | 5.80 | 18.76 | 11.88 | 0.09 |
| 32 | orta yuvarlak sağ | 44 | -3.00 | 1.36 | 3.29 | 25.82 | 11.14 | 0.02 |
| 11 | sol ceza sahası | 20 | -4.60 | -0.43 | 4.62 | 9.27 | 10.29 | 0.20 |
| 1 | sol kale çizgisi | 18 | -7.27 | 0.32 | 7.27 | 6.91 | 9.17 | 0.53 |
| 27 | sağ kale çizgisi | 27 | -5.71 | -0.28 | 5.72 | 17.76 | 9.15 | 0.09 |
| 5 | sol kale çizgisi | 7 | 0.74 | 4.18 | 4.24 | 9.01 | 9.05 | 0.18 |
| 2 | sol kale çizgisi | 18 | -5.21 | 0.93 | 5.29 | 8.16 | 8.36 | 0.30 |
| 23 | sağ kale alanı | 28 | -3.01 | -0.27 | 3.02 | 13.42 | 8.13 | 0.05 |
| 19 | sağ ceza sahası | 33 | -2.76 | 0.62 | 2.83 | 14.24 | 6.83 | 0.04 |
| 26 | sağ kale çizgisi | 28 | -1.37 | -1.66 | 2.15 | 10.31 | 6.74 | 0.04 |
| 13 | sol ceza sahası | 11 | 5.84 | -0.25 | 5.84 | 5.40 | 6.13 | 0.54 |
| 4 | sol kale çizgisi | 10 | 6.60 | 0.98 | 6.67 | 10.37 | 5.75 | 0.29 |
| 18 | sağ ceza sahası | 35 | 1.52 | 0.67 | 1.67 | 18.00 | 5.62 | 0.01 |
| 12 | sol ceza sahası | 18 | -2.17 | 1.85 | 2.85 | 7.60 | 5.47 | 0.12 |
| 3 | sol kale çizgisi | 14 | -1.54 | 0.84 | 1.75 | 7.08 | 4.84 | 0.06 |
| 7 | sol kale alanı | 18 | -3.35 | 1.09 | 3.52 | 7.90 | 4.64 | 0.17 |
| 9 | sol penaltı | 18 | -2.92 | 1.89 | 3.48 | 7.55 | 4.59 | 0.18 |
| 25 | sağ kale çizgisi | 32 | 2.36 | -1.44 | 2.76 | 10.22 | 4.29 | 0.07 |
| 8 | sol kale alanı | 11 | 2.65 | 0.22 | 2.66 | 7.13 | 4.11 | 0.12 |

### 31/32 etiket hipotezi (tüm GT görüntüleri; 31/32 hariç fit edilen GT H ile)

Karşılaştırma tabanı: diğer GT keypoint'lerinin aynı H altında residual'ı medyan 0.68 px, p90 8.10 px.

| kp | n | etiket→gerçek nokta medyan | etiket→elips ucu medyan | gerçek↔uç mesafe medyan | uca daha yakın | ort. ofset dx | dy |
|---|---|---|---|---|---|---|---|
| 31 | 223 | 3.13 | 4.55 | 2.56 | %15.7 | 0.63 | -0.01 |
| 32 | 221 | 3.15 | 6.48 | 3.89 | %7.2 | -2.33 | -0.54 |

## D2 — H fit'ten keypoint dışlama (model @640)

k ve liste sadece calib'den: keypoint'ler calib medyan hatasına göre sıralanır, calib'de H hatası medyanını en küçük yapan k seçilir (eşitlikte küçük k). holdout seçimde kullanılmadı.

| k | dışlanan | calib H hatası medyan (m) |
|---|---|---|
| 0 | [] | 3.96 |
| 1 | [21] | 3.96 |
| 2 | [21, 17] | 4.07 |
| 3 | [21, 17, 29] | 4.07 |
| 4 | [21, 17, 29, 15] | 4.16 |
| 5 | [21, 17, 29, 15, 16] | 4.31 |
| 6 | [21, 17, 29, 15, 16, 31] | 5.19 |
| 7 | [21, 17, 29, 15, 16, 31, 20] | 5.62 |
| 8 | [21, 17, 29, 15, 16, 31, 20, 14] | 4.86 |
| 9 | [21, 17, 29, 15, 16, 31, 20, 14, 24] | 5.13 |
| 10 | [21, 17, 29, 15, 16, 31, 20, 14, 24, 22] | 6.54 |

Seçilen k = 0: []

| cfg | set | n | H hatası medyan (m) | p10 | p90 | doğru | belirsiz | yanlış | H yok | D5 ızgara medyanı (medyan, m) | D5 alt 2/3 ortalaması (medyan, m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dışlama yok | calib | 36 | 3.96 | 2.78 | 5.94 | 0 | 2 | 33 | 1 | 3.72 | 3.81 |
| dışlama yok | holdout | 21 | 2.82 | 1.83 | 3.67 | 0 | 4 | 17 | 0 | 2.49 | 2.91 |
| dışlama yok | aynı-maç | 42 | 1.56 | 1.04 | 2.76 | 0 | 26 | 16 | 0 | 1.33 | 1.38 |
| (a) 31-32 hariç | calib | 36 | 4.23 | 2.79 | 5.92 | 0 | 2 | 32 | 2 | 3.88 | 4.30 |
| (a) 31-32 hariç | holdout | 21 | 2.88 | 1.83 | 3.90 | 0 | 3 | 18 | 0 | 2.48 | 2.95 |
| (a) 31-32 hariç | aynı-maç | 42 | 1.66 | 1.07 | 4.04 | 0 | 25 | 16 | 1 | 1.45 | 1.57 |
| (b) calib'in en kötü k=0: [] | calib | 36 | 3.96 | 2.78 | 5.94 | 0 | 2 | 33 | 1 | 3.72 | 3.81 |
| (b) calib'in en kötü k=0: [] | holdout | 21 | 2.82 | 1.83 | 3.67 | 0 | 4 | 17 | 0 | 2.49 | 2.91 |
| (b) calib'in en kötü k=0: [] | aynı-maç | 42 | 1.56 | 1.04 | 2.76 | 0 | 26 | 16 | 0 | 1.33 | 1.38 |

## D3 — inference çözünürlüğü (aynı model, eğitim 640'ta)

| imgsz | set | recall@0.5 | kp hata medyan | p90 |
|---|---|---|---|---|
| 640 | görülmemiş (calib+holdout) | %90.3 | 11.37 | 37.03 |
| 640 | aynı-maç | %87.9 | 4.53 | 11.77 |
| 960 | görülmemiş (calib+holdout) | %53.9 | 19.11 | 66.75 |
| 960 | aynı-maç | %60.0 | 10.92 | 30.87 |
| 1280 | görülmemiş (calib+holdout) | %29.5 | 27.93 | 76.09 |
| 1280 | aynı-maç | %37.2 | 25.98 | 70.10 |

| cfg | set | n | H hatası medyan (m) | p10 | p90 | doğru | belirsiz | yanlış | H yok | D5 ızgara medyanı (medyan, m) | D5 alt 2/3 ortalaması (medyan, m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| imgsz 640 | görülmemiş (calib+holdout) | 57 | 3.35 | 2.03 | 5.30 | 0 | 6 | 50 | 1 | 3.09 | 3.35 |
| imgsz 640 | aynı-maç | 42 | 1.56 | 1.04 | 2.76 | 0 | 26 | 16 | 0 | 1.33 | 1.38 |
| imgsz 960 | görülmemiş (calib+holdout) | 57 | 6.87 | 3.91 | 20.11 | 0 | 1 | 46 | 10 | 5.88 | 6.82 |
| imgsz 960 | aynı-maç | 42 | 4.28 | 3.04 | 8.58 | 0 | 0 | 40 | 2 | 3.95 | 4.25 |
| imgsz 1280 | görülmemiş (calib+holdout) | 57 | 10.95 | 5.46 | 41.28 | 0 | 0 | 36 | 21 | 8.56 | 8.80 |
| imgsz 1280 | aynı-maç | 42 | 20.43 | 5.53 | 788.74 | 0 | 0 | 33 | 9 | 14.77 | 15.73 |

Not: iyileşme anlamlıdır; kötüleşme 1280'de eğitimin işe yaramayacağını göstermez (model 640'ta eğitildi, ölçek uyumsuzluğu).


## D4 — GT gürültü tabanı (tüm field v12 GT görüntüleri, exclude hariç; bootstrap B=100)

| ölçü | medyan | p90 | n |
|---|---|---|---|
| GT residual, 31/32 hariç keypoint'ler (640 px) | 0.79 | 8.30 | 3704 |
| GT residual, 31/32 (640 px) | 1.29 | 6.10 | 454 |
| bootstrap GT H belirsizliği: ortalama ızgara hatası (m) — görüntü başına medyan | 0.58 | 1.38 | 275 |
| bootstrap: görüntü başına p90 (m) | 2.46 | 7.29 | 275 |
| bootstrap D5: ızgara medyanı (m) | 0.43 | 0.99 | 275 |
| bootstrap D5: alt 2/3 ortalaması (m) | 0.48 | 1.11 | 275 |

Bootstrap: GT keypoint'leri yerine koyarak örneklenir, H yeniden fit edilir, tam GT H'ye ortalama ızgara hatası. Görüntülerin %95'inde bootstrap p90'ı 0.5 m'yi aşıyor.


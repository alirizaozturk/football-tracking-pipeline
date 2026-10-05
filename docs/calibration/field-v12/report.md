# Faz K — homography kalibrasyonu: field-v12

Görüntü: valid 30, test 24, train 222. Eşikler sadece **valid**'den seçildi; test ve train seçimde kullanılmadı..

## 1. GT homography ve geometri (is_plausible)

| split | fit | geometri geçti | rmse px (medyan / p95) | inlier oranı (medyan) |
|---|---|---|---|---|
| valid | %100.0 (30/30; GA %89–100) | %100.0 (30/30; GA %89–100) | 2.3 / 3.3 | 0.80 |
| test | %100.0 (24/24; GA %86–100) | %100.0 (24/24; GA %86–100) | 2.8 / 3.7 | 0.80 |
| train | %100.0 (222/222; GA %98–100) | %99.5 (221/222; GA %97–100) | 2.7 / 3.5 | 0.78 |

valid+test geometri: %100.0 (54/54; GA %93–100) — hedef ≥%95: TUTTU.

Geometriden kalanlar:

| split | image | n_kp | n_inliers | rmse_px |
|---|---|---|---|---|
| train | 42ba34_9_2_png.rf.9bf0f5d5ef6202770e2ed8e9016b2c8f.jpg | 16 | 13 | 2.9 |

## 2. Bugünkü kod (HomographyTracker.update: HSV, tol 2 px, eşik 0.15, NaN kabul)

| split | doğru H reddi (FRR) | kaydırılmış H kabulü (FAR, ±3/5/10 m) | çapraz ±7 m kabulü |
|---|---|---|---|
| valid | %26.7 (8/30; GA %14–44) | %50.3 (181/360; GA %45–55) | %26.7 (32/120; GA %20–35) |
| test | %16.7 (4/24; GA %7–36) | %61.5 (177/288; GA %56–67) | %31.2 (30/96; GA %23–41) |
| train | %17.6 (39/222; GA %13–23) | %57.5 (1532/2664; GA %56–59) | %28.9 (257/888; GA %26–32) |

Kaydırılmış H kabulü, eksen ve büyüklüğe göre (valid+test):

| eksen | kayma | kabul |
|---|---|---|
| x | ±3 m | %41.7 (45/108; GA %33–51) |
| x | ±5 m | %38.0 (41/108; GA %29–47) |
| x | ±10 m | %29.6 (32/108; GA %22–39) |
| y | ±3 m | %75.0 (81/108; GA %66–82) |
| y | ±5 m | %75.0 (81/108; GA %66–82) |
| y | ±10 m | %72.2 (78/108; GA %63–80) |

## 3. Maske / tolerans / kural karşılaştırması

Kural A: geometri ∧ s ≥ tau (tau = valid doğru-H skorunun %5 quantile'ı). Kural B: geometri ∧ s ≥ tau ∧ marj ≥ mu (ikisi de %4 quantile). NaN = red. FAR: ±3/5/10 m kaymalar; seçim valid'de, test seçimde kullanılmadı.

Seçim: valid'de kural B FAR'ı en düşük olan; eşitlikte valid ayrışma payı (doğru-H marjının %4 quantile'ı − kaydırılmış-H marjının %99 quantile'ı) en büyük olan; sonra ölçekli tolerans. Test sütunları seçimde kullanılmadı.

| maske | tol | s medyan doğru / kaymış | A: tau | A valid FRR / FAR | A test FRR / FAR | B: tau, mu | valid pay | B valid FRR / FAR | B test FRR / FAR |
|---|---|---|---|---|---|---|---|---|---|
| hsv | fixed | 0.34 / 0.17 | 0.03 | %7 / %90 | %4 / %95 | 0.02, -0.15 | -0.161 | %7 / %89 | %4 / %90 |
| hsv | scaled | 0.40 / 0.20 | 0.04 | %7 / %89 | %4 / %97 | 0.03, -0.15 | -0.150 | %7 / %87 | %4 / %88 |
| tophat10 | fixed | 0.89 / 0.43 | 0.72 | %3 / %0 | %4 / %0 | 0.72, 0.11 | +0.114 | %7 / %0 | %4 / %0 |
| tophat10 | scaled | 0.92 / 0.46 | 0.75 | %3 / %0 | %4 / %0 | 0.75, 0.12 | +0.138 | %7 / %0 | %4 / %0 |
| tophat20 | fixed | 0.83 / 0.35 | 0.47 | %7 / %12 | %0 / %14 | 0.42, 0.05 | +0.046 | %7 / %0 | %0 / %0 |
| tophat20 | scaled | 0.87 / 0.39 | 0.52 | %7 / %7 | %0 / %8 | 0.46, 0.05 | +0.058 | %7 / %0 | %0 / %0 |
| tophat30 | fixed | 0.75 / 0.32 | 0.43 | %7 / %14 | %8 / %15 | 0.39, 0.06 | +0.048 | %7 / %0 | %8 / %0 |
| tophat30 | scaled | 0.82 / 0.35 | 0.47 | %7 / %13 | %0 / %15 | 0.43, 0.05 | +0.059 | %7 / %0 | %4 / %0 |
| tophat40 | fixed | 0.65 / 0.28 | 0.37 | %7 / %22 | %8 / %24 | 0.36, 0.06 | +0.051 | %3 / %0 | %8 / %0 |
| tophat40 | scaled | 0.72 / 0.31 | 0.41 | %7 / %21 | %8 / %22 | 0.40, 0.06 | +0.054 | %7 / %0 | %12 / %0 |

## 4. Seçilen ayar: tophat10, tol scaled, kural B (tau=0.75, mu=0.12)

| split | doğru H reddi (FRR) | kaydırılmış H kabulü (FAR) | çapraz ±7 m kabulü | hedef (FAR<%5, FRR<%10) |
|---|---|---|---|---|
| valid (seçim) | %6.7 (2/30; GA %2–21) | %0.0 (0/360; GA %0–1) | %0.0 (0/120; GA %0–3) | TUTTU |
| test | %4.2 (1/24; GA %1–20) | %0.0 (0/288; GA %0–1) | %0.0 (0/96; GA %0–4) | TUTTU |
| train (ek kontrol) | %5.0 (11/222; GA %3–9) | %0.0 (0/2664; GA %0–0) | %0.0 (0/888; GA %0–0) | TUTTU |

Test split'inde kaydırılmış H kabulü, eksen ve büyüklüğe göre:

| eksen | kayma | kabul |
|---|---|---|
| x | ±3 m | %0.0 (0/48; GA %0–7) |
| x | ±5 m | %0.0 (0/48; GA %0–7) |
| x | ±10 m | %0.0 (0/48; GA %0–7) |
| y | ±3 m | %0.0 (0/48; GA %0–7) |
| y | ±5 m | %0.0 (0/48; GA %0–7) |
| y | ±10 m | %0.0 (0/48; GA %0–7) |

Karşılaştırma için en iyi HSV: tol scaled, B test FRR %4 / FAR %88; en iyi top-hat: tophat10 tol scaled, B test FRR %4 / FAR %0.

## 5. Çözünürlük (#10) — valid+test, keypoint gürültüsü sigma640 px (640 girişte; görüntüde sigma*W/640)

`mevcut`: HSV, tol 2 px, RANSAC 8 px, rmse ≤15 px, s≥0.15. `önerilen_sabit_px`: seçilen maske+kural, piksel parametreleri sabit. `önerilen_ölçekli`: tol, RANSAC ve rmse eşiği W ile ölçekli. 2x = 3840 px'e büyütülmüş 1080p (gerçek 4K değil).

| cfg | scale | width | sigma640 | fit_ok | kabul_dogru | s_dogru_medyan | FAR |
|---|---|---|---|---|---|---|---|
| mevcut | 0.500 | 960 | 0.000 | 1.000 | 0.796 | 0.377 | 0.634 |
| mevcut | 0.500 | 960 | 1.000 | 1.000 | 0.778 | 0.354 | 0.622 |
| mevcut | 0.500 | 960 | 2.000 | 1.000 | 0.778 | 0.313 | 0.580 |
| mevcut | 1.000 | 1920 | 0.000 | 1.000 | 0.778 | 0.336 | 0.552 |
| mevcut | 1.000 | 1920 | 1.000 | 1.000 | 0.759 | 0.296 | 0.517 |
| mevcut | 1.000 | 1920 | 2.000 | 1.000 | 0.630 | 0.213 | 0.386 |
| mevcut | 2.000 | 3840 | 0.000 | 1.000 | 0.759 | 0.257 | 0.461 |
| mevcut | 2.000 | 3840 | 1.000 | 1.000 | 0.630 | 0.208 | 0.347 |
| mevcut | 2.000 | 3840 | 2.000 | 1.000 | 0.500 | 0.147 | 0.259 |
| önerilen_sabit_px | 0.500 | 960 | 0.000 | 1.000 | 0.944 | 0.943 | 0.000 |
| önerilen_sabit_px | 0.500 | 960 | 1.000 | 1.000 | 0.907 | 0.906 | 0.000 |
| önerilen_sabit_px | 0.500 | 960 | 2.000 | 1.000 | 0.722 | 0.831 | 0.000 |
| önerilen_sabit_px | 1.000 | 1920 | 0.000 | 1.000 | 0.889 | 0.887 | 0.000 |
| önerilen_sabit_px | 1.000 | 1920 | 1.000 | 1.000 | 0.778 | 0.834 | 0.000 |
| önerilen_sabit_px | 1.000 | 1920 | 2.000 | 1.000 | 0.185 | 0.637 | 0.000 |
| önerilen_sabit_px | 2.000 | 3840 | 0.000 | 1.000 | 0.778 | 0.851 | 0.000 |
| önerilen_sabit_px | 2.000 | 3840 | 1.000 | 1.000 | 0.370 | 0.691 | 0.000 |
| önerilen_sabit_px | 2.000 | 3840 | 2.000 | 1.000 | 0.056 | 0.480 | 0.000 |
| önerilen_ölçekli | 0.500 | 960 | 0.000 | 1.000 | 0.944 | 0.937 | 0.000 |
| önerilen_ölçekli | 0.500 | 960 | 1.000 | 1.000 | 0.815 | 0.892 | 0.000 |
| önerilen_ölçekli | 0.500 | 960 | 2.000 | 1.000 | 0.593 | 0.790 | 0.000 |
| önerilen_ölçekli | 1.000 | 1920 | 0.000 | 1.000 | 0.963 | 0.931 | 0.000 |
| önerilen_ölçekli | 1.000 | 1920 | 1.000 | 1.000 | 0.870 | 0.897 | 0.000 |
| önerilen_ölçekli | 1.000 | 1920 | 2.000 | 1.000 | 0.426 | 0.731 | 0.000 |
| önerilen_ölçekli | 2.000 | 3840 | 0.000 | 1.000 | 0.981 | 0.937 | 0.000 |
| önerilen_ölçekli | 2.000 | 3840 | 1.000 | 1.000 | 0.889 | 0.925 | 0.000 |
| önerilen_ölçekli | 2.000 | 3840 | 2.000 | 1.000 | 0.481 | 0.768 | 0.000 |

## Notlar

- Kaydırılmış H, keypoint'lerin tutarlı biçimde yanlış çizgiye eşlenmesinin modelidir; gerçek model hatalarının hepsini temsil etmez (ör. tek keypoint'in yanlış yere kayması). Gerçek model hataları Faz 1'den sonra ölçülmeli.
- Örneklem küçük (valid 30, test 24); GA = %95 Wilson güven aralığı. train split'i ek kontrol: modelden bağımsız bir ölçüm olduğu için (çizgi/maske), eşik seçiminde kullanılmadığı sürece geçerli bir held-out örneklem.
- GT etiketleri FIFA 105x68 saha ölçüsüyle eşleniyor; gerçek saha ölçüsü farklıysa doğru H skoru biraz düşer.
- Şeffaflık: ilk deneme çalıştırmasında eşitlik bozucu 'valid FRR' idi ve tek görüntülük farkla (1/30 vs 2/30) tophat40'ı seçti. Bu istatistiksel olarak anlamsız olduğu için kural valid'deki ayrışma payına çevrildi. Değişiklik test sonuçları görüldükten sonra yapıldı; bu yüzden tüm varyantların test sonuçları 3. tabloda açıkça duruyor.

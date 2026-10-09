# Faz 1a — yapısal seçenek: marj testinden önce H'yi çizgilere göre iyileştirme

Aynı model, aynı keypoint'ler, aynı ön-kayıtlı tanımlar (doğru < 0.5 m, yanlış > 2 m, mu* = calib doğru-H marjlarının %5 quantile'ı). Marj ve geometri iyileştirilmiş H'de hesaplanır. Kaydırılmış model H'leri (±3/5/10 m) de iyileştirilir; doğruya çekilenler 'kurtarıldı', yanlış kalıp kabul edilenler yanlış kabul.

| yöntem | set | H hatası medyan (p10–p90) m | doğru / belirsiz / yanlış | mu* | FRR @mu* | FAR @mu* (doğal / kaydırılmış) | FRR @0.12 | FAR @0.12 | kaydırılmış → doğru |
|---|---|---|---|---|---|---|---|---|---|
| model H (iyileştirmesiz) | seçim (calib) | 3.96 (2.70–5.94) | 0 / 2 / 33 | nan | — | — | %nan (0/0) | %0.0 (0/451) | %0.0 (0/420) |
| model H (iyileştirmesiz) | kontrol (holdout) | 2.82 (1.83–3.67) | 0 / 4 / 17 | nan | — | — | %nan (0/0) | %0.0 (0/269) | %0.0 (0/252) |
| model H (iyileştirmesiz) | aynı-maç referans | 1.56 (1.04–2.76) | 0 / 26 / 16 | nan | — | — | %nan (0/0) | %0.0 (0/518) | %0.0 (0/504) |
| öteleme (DT) | seçim (calib) | 4.36 (2.01–7.43) | 0 / 4 / 31 | nan | — | — | %nan (0/0) | %0.0 (0/419) | %0.0 (0/420) |
| öteleme (DT) | kontrol (holdout) | 2.31 (1.28–4.19) | 0 / 8 / 13 | nan | — | — | %nan (0/0) | %0.0 (0/201) | %0.0 (0/252) |
| öteleme (DT) | aynı-maç referans | 1.53 (0.92–2.52) | 0 / 29 / 13 | nan | — | — | %nan (0/0) | %8.4 (24/286) | %0.0 (0/504) |
| öteleme + ECC | seçim (calib) | 4.37 (2.83–8.15) | 0 / 1 / 34 | nan | — | — | %nan (0/0) | %0.2 (1/446) | %0.0 (0/420) |
| öteleme + ECC | kontrol (holdout) | 2.41 (1.46–6.68) | 0 / 7 / 14 | nan | — | — | %nan (0/0) | %0.0 (0/210) | %0.0 (0/252) |
| öteleme + ECC | aynı-maç referans | 2.16 (1.05–23.26) | 0 / 19 / 23 | nan | — | — | %nan (0/0) | %6.4 (24/374) | %0.0 (0/504) |

# Faz A — gürültü altında çizgi kontrolü (field v12, valid + test)

Keypoint gürültüsü sigma = 0.0, 1.0, 2.0 px (640 girişte; 1920 px'te x3). sigma > 0 için görüntü başına 5 çekiliş. H gürültülü keypoint'lerden RANSAC (8·W/1280 px) ile fit edilir; rmse > 15·W/1280 ise 'fit' reddi. FAR: aynı gürültülü H'nin ±3/5/10 m x/y kaymaları. Marj eşiği her hücrede 0.12. Kararlar sadece **valid**'den, önceden yazılı kurallarla; test sadece rapor.

## 1. Reddin kritere göre dağılımı (top-hat 10)

Sıralı atama: fit/rmse → geometri → skor → marj (frames.parquet'teki reject_reason ile aynı). 'tek başına' sütunları: fit'i geçen doğru H'lerde o kriterin kendi başına red oranı (örtüşür).

| yapı | split | sigma | FRR | fit/rmse | geometri | skor | marj | skor tek başına | marj tek başına |
|---|---|---|---|---|---|---|---|---|---|
| Faz K: skor 0.75, tol W/640 | valid | 0 | %6.7 | %0.0 | %0.0 | %6.7 | %0.0 | %6.7 | %3.3 |
| Faz K: skor 0.75, tol W/640 | valid | 1 | %20.7 | %0.0 | %0.0 | %16.7 | %4.0 | %16.7 | %14.7 |
| Faz K: skor 0.75, tol W/640 | valid | 2 | %48.0 | %0.0 | %0.0 | %38.7 | %9.3 | %38.7 | %37.3 |
| Faz K: skor 0.75, tol W/640 | test | 0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 |
| Faz K: skor 0.75, tol W/640 | test | 1 | %12.5 | %0.0 | %0.0 | %6.7 | %5.8 | %6.7 | %10.0 |
| Faz K: skor 0.75, tol W/640 | test | 2 | %55.8 | %0.0 | %0.0 | %49.2 | %6.7 | %49.2 | %43.3 |
| seçilen: skor yok, tol W/640 | valid | 0 | %3.3 | %0.0 | %0.0 | %0.0 | %3.3 | %0.0 | %3.3 |
| seçilen: skor yok, tol W/640 | valid | 1 | %14.7 | %0.0 | %0.0 | %0.0 | %14.7 | %0.0 | %14.7 |
| seçilen: skor yok, tol W/640 | valid | 2 | %37.3 | %0.0 | %0.0 | %0.0 | %37.3 | %0.0 | %37.3 |
| seçilen: skor yok, tol W/640 | test | 0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 | %0.0 |
| seçilen: skor yok, tol W/640 | test | 1 | %10.0 | %0.0 | %0.0 | %0.0 | %10.0 | %0.0 | %10.0 |
| seçilen: skor yok, tol W/640 | test | 2 | %43.3 | %0.0 | %0.0 | %0.0 | %43.3 | %0.0 | %43.3 |

## 2. Skor eşiği x tolerans x gürültü (top-hat 10) — hücre: doğru H reddi / kaydırılmış H kabulü

| skor eşiği | tol | sigma=0 valid | sigma=0 test | sigma=1 valid | sigma=1 test | sigma=2 valid | sigma=2 test |
|---|---|---|---|---|---|---|---|
| 0.75 | W/640 | %6.7 / %0.0 | %0.0 / %0.0 | %20.7 / %0.0 | %12.5 / %0.0 | %48.0 / %0.0 | %55.8 / %0.0 |
| 0.75 | W/427 | %6.7 / %0.0 | %0.0 / %0.0 | %16.7 / %0.0 | %13.3 / %0.0 | %39.3 / %0.0 | %50.0 / %0.0 |
| 0.75 | W/320 | %16.7 / %0.0 | %8.3 / %0.0 | %25.3 / %0.0 | %12.5 / %0.0 | %39.3 / %0.0 | %51.7 / %0.0 |
| 0.50 | W/640 | %3.3 / %0.0 | %0.0 / %0.0 | %14.7 / %0.0 | %10.0 / %0.0 | %37.3 / %0.0 | %43.3 / %0.0 |
| 0.50 | W/427 | %6.7 / %0.0 | %0.0 / %0.0 | %15.3 / %0.0 | %11.7 / %0.0 | %34.0 / %0.0 | %44.2 / %0.0 |
| 0.50 | W/320 | %16.7 / %0.0 | %8.3 / %0.0 | %25.3 / %0.0 | %12.5 / %0.0 | %39.3 / %0.0 | %50.8 / %0.0 |
| yok | W/640 | %3.3 / %0.0 | %0.0 / %0.0 | %14.7 / %0.0 | %10.0 / %0.0 | %37.3 / %0.0 | %43.3 / %0.0 |
| yok | W/427 | %6.7 / %0.0 | %0.0 / %0.0 | %15.3 / %0.0 | %11.7 / %0.0 | %34.0 / %0.0 | %44.2 / %0.0 |
| yok | W/320 | %16.7 / %0.0 | %8.3 / %0.0 | %25.3 / %0.0 | %12.5 / %0.0 | %39.3 / %0.0 | %50.8 / %0.0 |

(valid hücre başına doğru H sayısı: sigma=0 → 30, sigma>0 → 150; kaydırılmış H 12 katı.)

## 3. Yapı kararı (madde 3)

Kural (valid, top-hat 10): sigma=0 FAR %0; sigma=1 ve 2'de FAR baz çizginin en fazla 1 puan üstü; sigma=1 FRR baz çizgiden (skor 0.75, W/640) en az 5 puan düşük; sigma=2 ve sigma=0 FRR kötüleşmiyor. Uygunlar arasında sigma=1 FRR en düşük, sonra sigma=2 FRR, sonra skor eşiği olmayan, sonra küçük tolerans.

Uygun adaylar: skor 0.50, W/640 (sigma=1 FRR %14.7, sigma=2 %37.3); skor yok, W/640 (sigma=1 FRR %14.7, sigma=2 %37.3); skor 0.50, W/427 (sigma=1 FRR %15.3, sigma=2 %34.0); skor yok, W/427 (sigma=1 FRR %15.3, sigma=2 %34.0)

**Karar: skor eşiği yok, tolerans W/640.**

## 4. Top-hat eşiği (madde 4) — seçilen yapıyla

| top-hat eşiği | valid pay (sigma=0) | sigma=0 valid / test (FRR / FAR) | sigma=1 valid / test (FRR / FAR) | sigma=2 valid / test (FRR / FAR) |
|---|---|---|---|---|
| 5 | +0.019 | %26.7 / %0.0 · %25.0 / %0.0 | %46.7 / %0.0 · %33.3 / %0.0 | %65.3 / %0.0 · %65.0 / %0.0 |
| 7 | +0.107 | %10.0 / %0.0 · %4.2 / %0.0 | %25.3 / %0.0 · %14.2 / %0.0 | %42.7 / %0.0 · %50.8 / %0.0 |
| 10 | +0.160 | %3.3 / %0.0 · %0.0 / %0.0 | %14.7 / %0.0 · %10.0 / %0.0 | %37.3 / %0.0 · %43.3 / %0.0 |

Kural: 5 ya da 7, valid'de sigma=0 FAR %0 tutup sigma=1 FRR'yi 10'a göre en az 5 puan düşürürse (sigma=0 FRR kötüleşmeden) seçilir; yoksa 10. **Karar: 10.**

## 5. Performans (madde 6) — 1080p, kütüphanenin HomographyTracker yolu

Maske frame başına bir kez hesaplanır; skor + 12 probe aynı maskeyi kullanır ve tek perspectiveTransform çağrısında projekte edilir. n=90 ölçüm (1920x1080).

| adım | medyan ms | p90 ms |
|---|---|---|
| çizgi maskesi (LAB + top-hat + dilate) | 7.19 | 7.71 |
| skor + 12 probe (1 H) | 0.94 | 1.25 |
| HomographyTracker.update toplam (fit + geometri + maske + kanıt) | 8.74 | 9.76 |

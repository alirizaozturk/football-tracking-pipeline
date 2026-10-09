# Faz 1a-2 — pose teşhisi (yeni eğitim yok) ve eğitim önerisi

Model: `runs/pose-1a/weights/best.pt` (yolo11m-pose @640, 9 maç / 176 kare). Ön-kayıtlı tanımlar değişmedi:
doğru H < 0.5 m, yanlış H > 2 m (görüntüde görünen saha üzerinde 1 m ızgarada ortalama hata, GT H'ye göre).
Ayrıntılı tablolar: [diag.md](diag.md) (D1–D5), [d6_vram.md](d6_vram.md) (D6). Kod: `scripts/01d_diagnose_pose.py`,
`scripts/measure_batch_vram.py`.

## Özet

| | bulgu |
|---|---|
| D1 | Görülmemiş maçlarda keypoint hatası ağırlıkla **rastgele** (sistematik pay çoğu noktada < %20). En kötüler: 21, 17, 29 (yakın taç çizgisi tarafı), orta çizgi 14–16, 31. |
| D1 (31/32) | Hipotez **desteklenmedi**: etiketler elips ucuna değil gerçek saha noktasına daha yakın (uca daha yakın %16 / %7). Ama 31/32 etiketleri diğer noktalardan ~4–5 kat gürültülü (residual medyan 3.1 px'e karşı 0.7 px, 640 ölçeği). |
| D2 | Keypoint dışlama işe yaramıyor: calib'de seçilen k = 0; 31–32 hariç calib 3.96 → 4.23 m, holdout 2.82 → 2.88 m. |
| D3 | 640'ta eğitilmiş model 960 / 1280 inference'ta çok kötüleşiyor (recall %90 → %54 → %30). Ölçek uyumsuzluğu; 1280 eğitimi hakkında bilgi vermiyor. |
| D4 | **GT gürültü tabanı eşiğin üstünde**: GT H'nin bootstrap belirsizliği görüntü başına medyan **0.58 m** (p90 1.38 m); görüntülerin %95'inde bootstrap p90'ı 0.5 m'yi aşıyor. |
| D5 | Ek metrikler birincil metrikle aynı tabloyu veriyor (model: aynı-maç 1.33 / 1.38 m, görülmemiş 3.09 / 3.35 m; taban 0.43 / 0.48 m). |
| D6 | Pose @640: batch 8 = 4.35 GB, **12 = 6.20 GB (sığan en büyük)**, 16 = 8.12 GB (taşıyor, 7× yavaş). Det/ball yolo11m @1280: **3** (6.0 GB). |
| D7 | v14–v18 **yeni maç içermiyor** (18/18 ortak); +41 kaynak görüntü mevcut maçlardan, kare olarak gerdirilmiş (v14 640×640, v18 960×960). Ortak 276 görüntünün etiketi ve split'i v12 ile aynı. |

## D7 — Roboflow field versiyonları

| versiyon | görüntü | maç | çözünürlük / preprocessing | v12 ile ortak maç | v12'de olmayan kaynak görüntü |
|---|---|---|---|---|---|
| v1–v8 | 46 → 262 | — | auto-orient | — | v12'nin alt kümeleri (indirilmedi) |
| v10 / v12 | 276 | 18 | 1920×1080 (orijinal) | — | — |
| v11 / v13 | 276 | 18 | resize | — | — (v12 ile aynı görüntüler) |
| v14 (= v15, v17) | 317 | 18 | 640×640 "stretch" | 18 / 18 | 41: d82e7a 14, 0a2d9b 6, 08fd33 5, ec783a 5, e624c9 4, a9f16c 3, cd987c 2, 744b27 1, 798b45 1 |
| v18 | 317 | 18 | 960×960 "stretch" | 18 / 18 | aynı 41 |

## Eğitim önerisi

Teşhis varsayılan planı değiştirmeyi gerektirmiyor, tek istisna D4:

1. **Maç bazında 3-fold cross-validation** @640, batch **12** (D6), 300 epoch, mosaic 0, fliplr 0.5, keypoint dışlaması **yok** (D2).
   Fold'lar 18 maçın kare sayısına göre dengelenir (sabit seed). Her fold 12 maçla eğitilir; kalan 6 maçın kareleri
   out-of-fold tahmin edilir. Early stopping / best.pt seçimi için sadece eğitim maçlarının orijinal valid kareleri kullanılır
   (OOF kareler model seçimine girmez). Kalibrasyon ~275 OOF karede, ön-kayıtlı tanımlarla.
2. **Son model**: 18 maçın tamamıyla, aynı ayarlar; değerlendirme sayıları fold'lardan.
3. **Data**: v12 (D7: daha çok maç içeren versiyon yok). Opsiyonel: v18'deki 41 ek kare, 16:9'a geri ölçeklenerek ve maç
   fold'una göre eklenebilir (~%15 daha çok kare, yeni maç yok).
4. **Çözünürlük**: 640. D3, 1280 eğitimini ne destekliyor ne çürütüyor; istenirse tek fold'luk 1280 pilotu (batch ≤ 3) ayrı ölçülür.

**Süre tahmini** (RTX 4060 laptop; pose-1a: 176 kare ≈ 8 s/epoch; batch 12'de ölçülen 0.035 s/görüntü):
fold başına ~180 kare × 300 epoch ≈ 35–40 dk → 3 fold ≈ 2 saat; son model (275 kare) ≈ 55 dk; OOF tahmin + kalibrasyon
≈ 10 dk. **Toplam ≈ 3 saat GPU.** Opsiyonel 1280 pilotu (tek fold) ≈ 1.8 saat.

**Açık karar (D4):** GT H'nin kendi belirsizliği (medyan 0.58 m) 0.5 m "doğru H" eşiğinin üstünde. Yeni model aynı-maç
düzeyine (şu an 1.56 m) bile inse, mevcut tanımla "doğru H" sınıfı neredeyse boş kalır ve kalibrasyon yine FRR
tanımsız biter. Eğitime başlamadan önce tanımın kalıp kalmayacağına karar vermek gerekiyor. Seçenekler (tanım değişikliği
senin kararın; burada değiştirilmedi): (a) tanım aynı kalır, başarı sadece yanlış-H kabulüyle ölçülür; (b) "doğru" eşiği
tabana göre yeniden tanımlanır; (c) referans iyileştirilir (ör. GT H'yi görüntüdeki çizgilere göre ECC ile inceltip tabanı
yeniden ölçmek; ~10 dk CPU).

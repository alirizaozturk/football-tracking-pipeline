# Faz 1a — model keypoint'leriyle kalibrasyon (runs/pose-1a/weights/best.pt)

Ön-kayıt (kod bu sonuçlardan önce commit'lendi): doğru H = GT H'ye ortalama ızgara mesafesi < 0.5 m, yanlış H > 2.0 m, arası belirsiz (1.0 m ızgara, sadece görüntüde görünen saha). Eşik seçimi calib (maç bazında held-out), son kontrol holdout; valid/test_same aynı-maç referans (seçime girmez). mu* = seçimdeki doğru-H marjlarının %5 quantile'ı (n=0) = **nan**. FAR: doğal yanlış H + model H'sinin ±3/5/10 m kaydırılmış halleri (hepsi aynı ızgara mesafesiyle sınıflanır).

## Keypoint'ler (640 ölçeğinde px)

| set | GT-görünür kp | recall @0.5 | hata medyan | hata p90 | σ eşdeğeri | GT-görünmez ama conf≥0.5 |
|---|---|---|---|---|---|---|
| seçim (calib) | 492 | %90.4 (445/492; GA %88–93) | 13.77 | 43.64 | 11.70 | 25 |
| kontrol (holdout) | 350 | %90.0 (315/350; GA %86–93) | 8.27 | 20.55 | 7.02 | 2 |
| aynı-maç referans | 637 | %87.9 (560/637; GA %85–90) | 4.52 | 11.75 | 3.84 | 10 |

σ eşdeğeri = medyan radyal hata / √(2 ln 2) (izotrop 2B Gauss varsayımı); Faz A gürültü tablosuyla karşılaştırılabilir.

## Kabul zinciri

| set | mu | doğru H reddi (FRR) | red sebebi (rmse / geo / skor / marj) | yanlış H kabulü (FAR) | doğal FAR | kaydırılmış FAR | belirsiz | aday yok |
|---|---|---|---|---|---|---|---|---|
| seçim (calib) | varsayılan_0.12 | %nan (0/0; GA %nan–nan) | %0.0 / %0.0 / %0.0 / %0.0 | %0.0 (0/451; GA %0–1) | %0.0 (0/33; GA %0–10) | %0.0 (0/418; GA %0–1) | %5.6 (2/36; GA %2–18) | %2.8 (1/36; GA %0–14) |
| seçim (calib) | mu_seçilen | — | — | — | — | — | — | — | — |
| kontrol (holdout) | varsayılan_0.12 | %nan (0/0; GA %nan–nan) | %0.0 / %0.0 / %0.0 / %0.0 | %0.0 (0/269; GA %0–1) | %0.0 (0/17; GA %0–18) | %0.0 (0/252; GA %0–2) | %19.0 (4/21; GA %8–40) | %0.0 (0/21; GA %0–15) |
| kontrol (holdout) | mu_seçilen | — | — | — | — | — | — | — | — |
| aynı-maç referans | varsayılan_0.12 | %nan (0/0; GA %nan–nan) | %0.0 / %0.0 / %0.0 / %0.0 | %0.0 (0/518; GA %0–1) | %0.0 (0/16; GA %0–19) | %0.0 (0/502; GA %0–1) | %61.9 (26/42; GA %47–75) | %0.0 (0/42; GA %0–8) |
| aynı-maç referans | mu_seçilen | — | — | — | — | — | — | — | — |

**Bitti (ön-kayıt):** seçimde FAR < %5 ve FRR < %10 (doğal yanlış H ≥ 10 ise doğal FAR da < %5): **TUTMADI**; kontrolde aynı ölçüt: **TUTMADI**.

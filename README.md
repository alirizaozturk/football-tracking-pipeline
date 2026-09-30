# Football AI — yayın görüntüsünden tracking dataseti

Roboflow'un **Football AI Tutorial** videosundaki pipeline'ın yeniden yazımı, iki ekle:

1. **Eğitim datası üretimi** — kendi yayın görüntülerinden frame çıkar, auto-label yap, belirsizleri review'a ayır, Roboflow datasıyla birleştirip yeniden eğit.
2. **Tracking dataseti export'u** — her maç için `tracking.parquet` (oyuncu × frame; saha koordinatı metre, takım, hız, top). Render ve istatistikler bu datasetten üretilir; model tekrar çalışmaz. Sonraki projelerde (pas ağı, pressing, xG, taktik sınıflandırma…) doğrudan bu tablo kullanılır.

```
video ──► detection (YOLO) ──► ByteTrack ──► takım (SigLIP→UMAP→KMeans)
     └──► saha keypoint (YOLO-pose, 32 nokta) ──► homography ──► metre koordinat
                                                        │
                            tracking.parquet ◄──────────┘ ──► render (radar / Voronoi)
                                                          ──► analiz (mesafe, hız, possession proxy, şekil)
```

## Videodan farklar (bilinçli)

| Konu | Video | Burada | Neden |
|---|---|---|---|
| Saha ölçüsü | 120×70 m | 105×68 m, FIFA ceza sahası ölçüleri (`--length/--width`) | Hız/mesafe metre cinsinden doğru çıksın |
| Homography | `findHomography`, frame başına | RANSAC + geometrik akıl kontrolü + **görüntüde çizgi desteği** (`line_score`); keypoint yoksa son H 12 frame taşınır | Model tutarlı ama yanlış keypoint seti üretebilir (ör. olmayan bir orta yuvarlak); 4 noktalı fit'te RMSE hep 0 olduğu için tek başına ölçü değil |
| Takım ataması | frame başına | frame başına + **track bazında çoğunluk oyu** (`team`) | Frame frame titreme yok |
| Hız | — | median + hareketli ortalama, 12 m/s üstü atılır | Homography gürültüsü ham farkla 60+ km/s üretir |
| Top | ayrı model + slicer | aynı + fiziksel sıçrama filtresi + ≤1 s interpolasyon | |
| Tracker | `sv.ByteTrack` | `trackers.ByteTrackTracker` (yoksa eski) | `sv.ByteTrack` supervision 0.31'de kaldırılıyor |
| Takım backend | SigLIP | SigLIP **veya** `color` (GPU'suz, Lab renk) | Hızlı deneme / zayıf makine |

## Kurulum

Proje kökünde ayrı bir venv (Python 3.11). Sistem Python'una kurma.

```bash
py -3.11 -m venv .venv                     # Linux/macOS: python3.11 -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
# 1) PyTorch'u ÖNCE, CUDA build'iyle kur. Windows'ta PyPI'deki torch CPU'dur; requirements'tan önce
#    kurulmazsa ultralytics CPU torch'u çeker. cu128 RTX 40 ve RTX 50 (Blackwell) serisinde çalışır.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2) geri kalanlar (+ testler için pytest)
pip install -r requirements-dev.txt
pip check                                  # "No broken requirements found." görmelisin
```

**Tek OpenCV:** sadece `opencv-python`. `opencv-python-headless` / `opencv-contrib-python*` aynı `cv2/` klasörüne yazar; ikisi birden kuruluysa biri kaldırıldığında `cv2` bozulur. Bu yüzden `roboflow` paketi de kullanılmıyor (headless sabitliyor); `01_download_datasets.py` Roboflow REST API'sini doğrudan çağırır. `requirements.txt`'teki üst sınırlar test edilmiş sürüm aralığıdır.

## Fazlar

Her faz bir öncekinin çıktısını kullanır. **"Bitti"** satırı o fazın tamamlanma kriteri.

### Faz 0 — Ortam doğrulama
```bash
pip check
python -m tests.test_synthetic                 # uçtan uca sentetik test (çıktılar tests/_out/)
python -m pytest tests -v                      # sentetik test + negatif testler
```
`test_synthetic`: modelsiz sentetik bir yayın videosu üretir, detector/keypoint yerine GT'den gürültülü çıktı veren sahteleri koyar; tracker, homography, takım, postprocess, analiz ve render gerçek kodla çalışır. Pipeline modelleri `model(frame, frame_idx)` diye çağırır; sahte modeller frame'i çağrı sırasından değil bu parametreden bilir (gerçek modeller `frame_idx`'i yok sayar).

`test_negative`: analizde bulunan hataları mantık seviyesinde belgeler. `test_bulguNN_*` testleri düzeltmeler yapılana kadar **FAIL verir** (bulgu numaraları analiz raporundaki tabloya karşılık gelir): #1 x-kaymalı keypoint seti `fresh` kabul ediliyor, #2 kesme sonrası eski H taşınıyor, #6 ilk top tespiti sahteyse sonraki gerçek tespitler atılıyor, #7 uzun top boşluğunun ilk 1 s'si dolduruluyor, #8 hakem sınıf titremesi rol/takımı bozuyor, #9 conf 0.3–0.5 tespitler track almıyor ve 50 fps'te kayıp track 3.3 s tutuluyor. `test_kontrol_*` testleri bugün doğru olan davranışı sabitler, her zaman geçmeli.
**Bitti:** `pip check` temiz, `test_synthetic` ve `test_kontrol_*` geçiyor, `tests/_out/render_mid.png` radar + Voronoi ile oluşmuş. (`test_bulgu*` Faz A'da geçecek.)

### Faz 1 — Roboflow datası + modeller
```bash
export ROBOFLOW_API_KEY=...                                   # roboflow.com → Settings → API Keys (PowerShell: $env:ROBOFLOW_API_KEY="...")
python scripts/01_download_datasets.py --which players field ball
python scripts/02_train.py det  --data data/roboflow/players-v*/data.yaml
python scripts/02_train.py pose --data data/roboflow/field-v*/data.yaml
python scripts/02_train.py ball --data data/roboflow/ball-v*/data.yaml   # opsiyonel
python scripts/03_check_keypoints.py --kp runs/pose/weights/best.pt --source klip.mp4
```
- Varsayılanlar: det `yolo11m` @1280, pose `yolo11m-pose` @640 (mosaic ve yatay flip kapalı — flip sol/sağ keypoint'leri takas eder). `batch=-1` → AutoBatch VRAM'e göre seçer. 4060'ta OOM olursa `--model yolo11s.pt`.
- Test klibi için: videodaki klipler Kaggle'daki **DFL – Bundesliga Data Shootout** yarışmasından.

**Bitti:**
- det: player / goalkeeper / referee'de val mAP50 yüksek; ball belirgin düşük çıkar — küçük obje, beklenen durum (ayrı ball modeli bu yüzden var).
- pose: `outputs/keypoint_check/*.png`'de kırmızı projekte çizgiler gerçek beyaz çizgilerin üstüne oturuyor, frame'deki numaralar alttaki şemayla aynı yerde, başlıkta `-> OK` ve `lines=` değeri belirgin yüksek. **Bu kontrol atlanmamalı:** keypoint sırası datasetle uyuşmazsa pipeline hata vermez, sessizce yanlış koordinat üretir.

### Faz 2 — Kendi eğitim datan (ör. Süper Lig yayını)
```bash
python scripts/04_extract_frames.py --videos "videos/*.mp4" --every-s 1.0
python scripts/05_autolabel.py --frames data/own/frames \
    --det runs/det/weights/best.pt --kp runs/pose/weights/best.pt --previews
# data/own/review.csv'deki frame'leri CVAT / Roboflow'da düzelt
python scripts/02_train.py det  --data data/roboflow/players-v*/data.yaml data/own/det/data.yaml
python scripts/02_train.py pose --data data/roboflow/field-v*/data.yaml  data/own/pose/data.yaml
```
- Frame çıkarma yakın çekim/tribün karelerini (çim oranı) ve ardışık benzer kareleri atar.
- Train/val ayrımı **video bazında** (tek video → zaman bloğu). Komşu frame'ler neredeyse aynı; rastgele ayrım val skorunu şişirir.
- Auto-label modelin hatalarını datasete geri yazar; review adımı bu yüzden var (belirsiz conf bandı, anormal oyuncu sayısı, çoklu top, az keypoint).

**Bitti:** review listesi düzeltilmiş; birleşik datayla eğitilen model **kendi val split'inde** sadece-Roboflow modelinden daha iyi.

### Faz 3 — Tracking dataseti
```bash
python scripts/06_build_tracking_dataset.py --video mac.mp4 \
    --det runs/det/weights/best.pt --kp runs/pose/weights/best.pt \
    --ball runs/ball/weights/best.pt --ball-slice 640 \
    --start-s 0 --duration-s 120
```
Önce kısa bir aralıkla dene (`--duration-s`). `--min-line-score` (varsayılan 0.15) çizgi desteği eşiği: kendi footage'ında `frames.parquet` → `line_score` dağılımına bakıp ayarla (soluk/aşınmış çizgili sahalarda düşür). Takım sınıflandırıcı maçın kendi oyuncu kırpıntılarıyla fit edilir ve `team_classifier.pkl` olarak saklanır.

**Bitti:** `meta.json`'da `homography_coverage` yüksek (replay/yakın çekim oranına bağlı), `homography_rejected` sayıları düşük, `render_radar.mp4`'te radar noktaları oyuncularla eşzamanlı hareket ediyor, iki takım renkleri tutarlı.

### Faz 4 — Render + analiz (datasetten)
```bash
python scripts/07_render.py  data/matches/mac --mode radar     # ya da voronoi / --radar-only
python scripts/08_analyze.py data/matches/mac
```
**Bitti:** `stats/summary.json` fiziksel olarak makul: maks. hızlar ~36 km/s altında, sprint sayıları ve mesafeler süreye oranla gerçekçi. Aykırı değer varsa önce `h_src`/`rmse_px`'e bak.

## Dataset şeması

`data/matches/<maç>/`

**tracking.parquet** — obje × frame
| kolon | anlam |
|---|---|
| `frame`, `t` | frame indeksi, saniye |
| `track_id` | ByteTrack ID (−1: takip edilmeyen / top) |
| `cls`, `role` | detector sınıfı; track bazında çoğunluk sınıfı (`player`/`goalkeeper`/`referee`/`ball`) |
| `conf` | detection güveni |
| `team_raw`, `team` | frame bazında / track oylamalı takım (0, 1; −1 = hakem/bilinmiyor) |
| `x1,y1,x2,y2`, `u,v` | bbox ve zemin temas noktası (piksel) |
| `x,y` / `xs,ys` | saha koordinatı (m) / yumuşatılmış. x: sol kale çizgisinden, y: uzak taç çizgisinden |
| `speed_mps` | yumuşatılmış hız (m/s) |
| `on_pitch` | saha sınırı (+3 m) içinde mi |
| `h_src` | homography kaynağı: `fresh` / `carried` / `none` / `interp` |
| `outlier`, `interp` | top: imkansız sıçrama / interpolasyonla eklenmiş satır |

**frames.parquet** — frame başına `h_src`, `n_kp`, `n_inliers`, `rmse_px`, `line_score` ve `h0..h8` (görüntü→saha homography'si). Piksel koordinatlı her yeni bilgiyi (ör. sonradan eklediğin forma numarası tespiti) aynı matrisle sahaya taşıyabilirsin.

**meta.json** — video bilgisi, saha ölçüleri, model yolları, kolon açıklamaları.

```python
from fai.dataio import load_match
df, frames, meta = load_match("data/matches/mac")
ppl = df[df.role.isin(["player", "goalkeeper"]) & df.on_pitch & (df.h_src == "fresh")]
```

## Sınırlamalar

- **Homography sadece zemin düzlemi için doğru.** Havadaki top yanlış yere projekte edilir; `possession_proxy` bu yüzden kaba bir yaklaşım.
- **Kadraj dışı = veri yok.** Yayın kamerası tüm sahayı göstermez; mesafeler alt sınırdır.
- **Kimlik yok.** Kamera kesmesi/replay sonrası aynı oyuncu yeni `track_id` alır. Oyuncu adı/forma no için ayrı bir re-ID/OCR katmanı gerekir.
- **Takım 0/1 keyfi**, maçtan maça değişir.
- `color` backend benzer renkli formalarda (ve kaleci/hakem renkleri takımlara yakınsa) zayıflar; SigLIP daha dayanıklı.
- Yayın görüntüleri telifli; datasetleri/render'ları paylaşırken dikkat.

## Sentetik test sonuçları

`docs/synthetic_render.png`: sentetik yayın + radar/Voronoi çıktısı. `docs/keypoint_check_rejected.png`: az eğitilmiş pose modelinin ürettiği tutarlı-ama-yanlış homography — kırmızı çizgiler sahaya oturmuyor, `lines=0.01` ile reddediliyor. Faz 1c'de gerçek modelinde görmek istemediğin tablo bu.

Bilinen kamera ve oyuncu hareketiyle (keypoint gürültüsü σ=1.5 px, bbox σ=1 px, %10 top kaçırma, 2 sahte top):

| metrik | değer |
|---|---|
| saha konum hatası (medyan / p95), taze H | 0.16 m / 0.45 m |
| konum hatası, taşınmış H | 0.20 m |
| takım doğruluğu (oyuncu / kaleci) | %100 / %100 |
| hız hatası (medyan) | 0.22 m/s |
| sahte top tespiti | 2/2, boşluklar interpolasyonla dolduruldu |
| `line_score`: doğru H / az eğitilmiş pose modelinin yanlış H'si | 0.96 / 0.01–0.02 (yanlış olan reddedildi) |

Bu sayılar **model-dışı** kısmın doğruluğunu ölçer; gerçek videoda hata bütçesini detector ve keypoint modelinin kalitesi belirler.

Entegrasyon: `02_train` (det + pose), `03_check_keypoints`, `04_extract_frames`, `05_autolabel`, `06_build` (ayrı top modeli + slicer dahil), `07_render`, `08_analyze` gerçek ultralytics 8.4 ile CPU'da, sentetik veride küçük modellerle uçtan uca çalıştırıldı. SigLIP backend'i Hugging Face erişimi olmadığı için rastgele ağırlıklı küçük bir modelle test edildi (kod yolu + save/load); gerçek ağırlıklar ilk çalıştırmada indirilir.

## Klasör yapısı
```
fai/            pitch, homography, models, teams, pipeline, postprocess, build, render, analysis, viz, dataio
scripts/        01_download → 02_train → 03_check_keypoints → 04_extract → 05_autolabel → 06_build → 07_render → 08_analyze
tests/          test_synthetic.py (modelsiz uçtan uca), test_negative.py (bulgu + kontrol testleri), make_synth_yolo.py
```

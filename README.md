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
| Homography | `findHomography`, frame başına | RANSAC + geometrik akıl kontrolü + **görüntüde çizgi kanıtı**: top-hat çizgi maskesinde H'nin yerel tepe olması (marj); keypoint yoksa son H en fazla 12 frame taşınır ama her frame'de aynı kuralla doğrulanır; sahne kesmesinde bırakılır | Model tutarlı ama yanlış keypoint seti üretebilir (çizgi yönünde kayma, olmayan bir orta yuvarlak); 4 noktalı fit'te RMSE hep 0 olduğu için tek başına ölçü değil. Eşikler gerçek görüntüyle kalibre (Faz K/A) |
| Takım ataması | frame başına | frame başına + **track bazında çoğunluk oyu** (`team`, `role`; hakem dahil) | Frame frame titreme yok |
| Hız | — | median + hareketli ortalama, 12 m/s üstü atılır; homography olmayan frame'e konum uydurulmaz | Homography gürültüsü ham farkla 60+ km/s üretir |
| Top | ayrı model + slicer | aynı + fiziksel olarak tutarlı en uzun tespit zinciri (homography yoksa piksel uzayında) + sadece ≤1 s boşluklara interpolasyon | |
| Tracker | `sv.ByteTrack` | `trackers.ByteTrackTracker` (yoksa eski); kayıp track 2 s, kesmede sıfırlanır, ilk frame'in −1'i geri doldurulur | `sv.ByteTrack` supervision 0.31'de kaldırılıyor |
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
python -m pytest tests -v                      # sentetik + negatif + Faz A testleri
```
`test_synthetic`: modelsiz sentetik bir yayın videosu üretir, detector/keypoint yerine GT'den gürültülü çıktı veren sahteleri koyar; tracker, homography, takım, postprocess, analiz ve render gerçek kodla çalışır. Pipeline modelleri `model(frame, frame_idx)` diye çağırır; sahte modeller frame'i çağrı sırasından değil bu parametreden bilir (gerçek modeller `frame_idx`'i yok sayar).

`test_negative`: analizde bulunan hataları mantık seviyesinde belgeler (bulgu numaraları analiz raporundaki tabloya karşılık gelir). Faz 0'da mevcut kodda FAIL veriyorlardı, Faz A düzeltmeleriyle geçiyorlar: #1 x-kaymalı keypoint seti `fresh` kabul ediliyordu, #2 kesme sonrası eski H taşınıyordu, #6 ilk top tespiti sahteyse sonraki gerçek tespitler atılıyordu, #7 uzun top boşluğunun ilk 1 s'si dolduruluyordu, #8 hakem sınıf titremesi rol/takımı bozuyordu, #9 conf 0.3–0.5 tespitler track almıyordu ve 50 fps'te kayıp track 3.3 s tutuluyordu, #14 çizgi kanıtı olmadan (NaN) H kabul ediliyordu, #15 `homography_coverage` taşınan H'yi de sayıyordu, #16 homography olmayan frame'e yumuşatma konum uyduruyordu. `test_kontrol_*` doğru davranışı sabitler (ör. her H'yi reddeden bir "düzeltme" bunları bozar). `test_faz_a`: Faz A'da eklenen yollar (piksel uzayında top filtresi, ilk frame geri doldurma, sahne kesmesi, red sebebi kaydı, CLI ↔ config).
**Bitti:** `pip check` temiz, `pytest tests` tamamen geçiyor, `tests/_out/render_mid.png` radar + Voronoi ile oluşmuş.

### Faz K — Homography doğrulamasının gerçek görüntüyle kalibrasyonu (eğitim yok)
```bash
python scripts/01_download_datasets.py --which field          # field v12
python scripts/01b_calibrate_homography.py --data data/roboflow/field-v12
```
Field datasetinin GT keypoint'lerinden her görüntü için homography fit edilir. Ölçülenler: `is_plausible` geçme oranı (keypoint sırası ve y yönü), doğru H'nin çizgi skoru, aynı H'nin x/y yönünde ±3/5/10 m kaydırılmış hallerinin skoru (tutarlı-ama-yanlış H), HSV maske ile top-hat (yerel kontrast) maskesi, sabit ile çözünürlükle ölçekli tolerans, marj (H'nin ±2/4/7 m kaydırılmış "probe"larına göre yerel tepe olup olmadığı) ve çözünürlük ile keypoint gürültüsü etkisi. Eşikler valid split'te sadece doğru-H dağılımından seçilir; test ve train seçimde kullanılmaz. Çıktı `outputs/calibration/<dataset>/`; field v12 sonucu [docs/calibration/field-v12/report.md](docs/calibration/field-v12/report.md).

field v12 sonucu (özet):
- Geometri: valid+test 54/54, train 221/222. Kalan tek görüntü dataset etiket hatası (yakın/uzak noktalar aynalanmış). Keypoint sırası ve y=0 = uzak taç çizgisi konvansiyonu doğrulandı.
- **Faz A öncesi kontrol gerçek yayında işlemiyordu.** HSV maskesi gerçek çizgi piksellerinin sadece ~%33'ünü "beyaz" sayıyordu (çizgilerin medyan doygunluğu S≈79 > 70). Sonuç: doğru H'lerin %17–27'si reddediliyor, kaydırılmış H'lerin %50–62'si kabul ediliyordu.
- Faz K önerisi: top-hat maske (L kanalı, eşik 10, çekirdek W/128), çözünürlükle ölçekli tolerans (W/640 px), kabul = geometri ∧ skor ≥ 0.75 ∧ marj ≥ 0.12. Doğru H reddi valid/test/train %6.7 / %4.2 / %5.0; kaydırılmış H kabulü üçünde de %0. Faz A'da uygulandı; gürültü çalışmasından sonra skor eşiği kaldırıldı (aşağıda).
- Uyarı: eşikler hatasız GT keypoint'leriyle kalibre edildi; keypoint gürültüsünde doğru H reddi artıyor. Faz 1'de eğitilen modelin gerçek keypoint'leriyle yeniden kalibre edilecek.

**Bitti:** GT görüntülerinin ≥%95'i geometriden geçiyor; seçilen ayarda kaydırılmış H kabulü <%5, doğru H reddi <%10.

### Faz A — Model gerektirmeyen düzeltmeler
```bash
python -m pytest tests -v
python scripts/01b_calibrate_homography.py --data data/roboflow/field-v12 --noise    # Faz K ölçümleri + gürültü çalışması
```
- **Homography doğrulaması** (`fai/homography.py`): top-hat çizgi maskesi frame başına bir kez hesaplanır; H'nin skoru ve ±2/4/7 m probe'ları aynı maskeden okunur. Kabul = geometri ∧ marj ≥ 0.12; skor eşiği varsayılan olarak yok. Kanıt yoksa (görüntüye <30 örnek düşüyorsa) red. Taşınan H her frame'de aynı kuralla doğrulanır; geçemezse bırakılır. RANSAC/rmse/tolerans/çekirdek frame genişliğiyle ölçeklenir. 1080p'de `HomographyTracker.update` toplam ~8–9 ms/frame (maske ~7 ms, skor + 12 probe ~1 ms; medyan, laptop CPU).
- **Gürültü çalışması** ([docs/calibration/field-v12/noise.md](docs/calibration/field-v12/noise.md)): 640 girişte σ=1 / 2 px keypoint gürültüsünde Faz K kuralı (skor ≥ 0.75) doğru H'lerin valid'de %20.7 / %48'ini reddediyordu, kalan redlerin çoğu skor eşiğinden geliyordu. Skor eşiği kaldırılınca %14.7 / %37.3; kaydırılmış H kabulü bütün hücrelerde %0 kaldı. Toleransı büyütmek yardımcı olmadı; top-hat eşiği 5/7, 10'dan kötü. Kalan gürültü hassasiyeti marjda; Faz 1'de modelin keypoint'leriyle kalibre edilecek.
- **Sahne kesmesi**: ardışık frame HSV histogramlarının Bhattacharyya mesafesi > `cut_threshold` (0.5) ise tracker ve taşınan H sıfırlanır; ID'ler çakışmasın diye kaydırılır. Eşik gerçek yayında **kalibre edilmedi**: `frames.parquet` → `cut_dist` dağılımıyla Faz 3'te ayarlanacak.
- **Tracker**: `track_high_conf` = detector conf (0.3), yani datasete giren her tespit track başlatabilir; kayıp track `track_buffer_s` = 2 s (trackers'ın 30 fps ölçeklemesi düzeltildi); ByteTrack'in yeni track'e verdiği ilk frame −1'i postprocess'te geri doldurulur.
- **Postprocess**: rol ve takım oylaması hakem dahil; hakemin takımı −1, render/viz rolü kullanır. Homography olmayan frame'e `xs/ys` uydurulmaz. Top: fiziksel olarak tutarlı en uzun tespit zinciri (homography yoksa piksel uzayında, `max_ball_speed_wps`; kalibre edilmedi), sadece ≤ `ball_max_gap_s` boşluklar doldurulur.
- **Dataset**: `meta.json` → `homography_coverage` artık sadece fresh; `homography_carried_frac`, `scene_cuts`, `pipeline_config`, `post_config`, red sayaçları ayrı. `frames.parquet` → her frame için fresh adayın `line_score`, `line_margin`, `reject_reason` değerleri (reddedilenler dahil), taşınan H için `carry_*`, `cut`, `cut_dist`.
- **CLI**: `PipelineConfig` ve `PostConfig`'in her alanı `06_build_tracking_dataset.py` ve `03_check_keypoints.py`'de `--alan-adi` parametresi (`--min-line-score none`, `--line-probes-m 2 4 7`, …). `03_check_keypoints` artık pipeline'ın kabul zincirini kullanır.

**Bitti:** bütün negatif, kontrol ve Faz A testleri geçiyor; sentetik metrikler kötüleşmedi; Faz K ölçümleri yeni kodla tekrarlandığında hedefler tutuyor.

### Faz 1 — Roboflow datası + modeller
```bash
export ROBOFLOW_API_KEY=...                                   # roboflow.com → Settings → API Keys (PowerShell: $env:ROBOFLOW_API_KEY="...")
python scripts/01_download_datasets.py --which players field ball   # players v10, field v12, ball v2
python scripts/02_train.py det  --data data/roboflow/players-v10/data.yaml
python scripts/02_train.py pose --data data/roboflow/field-v12/data.yaml
python scripts/02_train.py ball --data data/roboflow/ball-v2/data.yaml    # opsiyonel
python scripts/03_check_keypoints.py --kp runs/pose/weights/best.pt --source klip.mp4
```
- Dataset versiyonları videodaki/roboflow-sports notebook'larındakilere sabit (players v10, field v12, ball v2). Projelerin en son versiyonları farklı (field v18, players v20, ball v4) ve içerikleri doğrulanmadı; istersen `--field-version N` ile başka versiyon seçebilirsin (`0` = en son). Her dataset klasörüne `dataset_meta.json` yazılır: versiyon, split sayıları, preprocessing, augmentation.
- field v12'de bilinen etiket hatası: `train/42ba34_9_2_png.rf.9bf0f5d5…` görüntüsünde yakın ve uzak taç çizgisi noktaları aynalanmış (Faz K). Pose eğitiminden önce çıkarılmalı (aşağıdaki plan).
- Varsayılanlar: det `yolo11m` @1280, pose `yolo11m-pose` @640 (mosaic ve yatay flip kapalı — flip sol/sağ keypoint'leri takas eder). `batch=-1` → AutoBatch VRAM'e göre seçer. 4060'ta OOM olursa `--model yolo11s.pt`.
- Test klibi için: videodaki klipler Kaggle'daki **DFL – Bundesliga Data Shootout** yarışmasından.

#### Faz 1a — pose modeli + model keypoint'leriyle kalibrasyon
```bash
python scripts/01c_prepare_pose_data.py --data data/roboflow/field-v12          # split + flip_idx doğrulaması
python scripts/02_train.py pose --data data/derived/field-v12-1a/data.yaml --name pose-1a --device 0 > runs/pose-1a/train.log 2>&1 &
python scripts/gpu_watchdog.py --pid-file runs/pose-1a/train.pid --log runs/pose-1a/gpu.csv   # ayrı terminalde
python scripts/02_train.py pose --data data/derived/field-v12-1a/data.yaml --name pose-1a --resume   # yarıda kaldıysa
python scripts/01b_calibrate_homography.py --kp-model runs/pose-1a/weights/best.pt --kp-data data/derived/field-v12-1a
```
- **Maç bazında split** (`splits/field-v12/`, commit'li): Roboflow'un field v12 split'leri kare bazında; 18 maçın hepsi train'de ve valid/test karelerinin %75–90'ı train'deki kliplerin komşu kareleri. Bu yüzden ≤20 kareli maçlardan sabit seed'le **calib** (7 maç / 36 kare; çizgi kontrolü eşikleri burada seçilir) ve **holdout** (2 maç / 21 kare; son kontrol) ayrılır. Bu maçların bütün kareleri eğitimden çıkar. Eğitim 9 maç / 176 kare. Eğitim maçlarının valid'i (23) early stopping için kalır, aynı-maç referans olarak raporlanır. `exclude.txt`: aynalanmış etiketli görüntü. Orijinal dataset dosyalarına dokunulmaz (türetilmiş dataset hardlink).
- **flip_idx**: `pitch.py`'nin ayna eşlemesiyle (her i için `vertex[flip_idx[i]] == (L − x_i, y_i)`) birebir aynı → pose eğitimi `fliplr=0.5`. `02_train` bu kararı her pose eğitiminde programatik verir.
- **Sıcaklık bekçisi** (`gpu_watchdog.py`): 60 s'de bir sıcaklık / SM clock / thermal slowdown / güç CSV'ye yazılır. Thermal slowdown aktifse ya da sıcaklık art arda 3 örnek (~3 dk) > 85 °C ise eğitim süreç ağacı durdurulur; `--resume` ile son epoch'tan devam edilir.
- **Kalibrasyon ön-kaydı** (`01b --kp-model`, sonuçlardan önce commit'lendi): doğru H = model H'si ile GT H'si arasındaki ortalama mesafe < 0.5 m (görüntüde görünen saha üzerinde 1 m ızgara), yanlış H > 2 m, arası belirsiz. Marj eşiği calib'deki doğru-H marjlarının %5 quantile'ı; holdout son kontrol. Yanlış H kabulü: doğal yanlış H'ler + model H'sinin ±3/5/10 m kaydırılmış halleri.
- **Sonuç** (`pose-1a`: yolo11m-pose @640, 300 epoch, fliplr 0.5; [docs/faz1a](docs/faz1a)) — **Bitti kriteri tutmadı**:
  - pose mAP50-95: aynı-maç valid 0.807 / test_same 0.847, görülmemiş maçlar calib **0.238** / holdout **0.517**.
  - Keypoint hatası (640 ölçeğinde, medyan / p90; conf ≥ 0.5): aynı-maç 4.5 / 11.8 px, calib 13.8 / 43.6 px, holdout 8.3 / 20.6 px. σ eşdeğeri 3.8 / 11.7 / 7.0 px, Faz A gürültü tablosunun üst sınırının (σ = 2) çok üstünde. En kötü noktalar orta çizgi / orta yuvarlak (14–17, 31, 32) ve yakın taç çizgisi (21, 29).
  - Model H hatası medyan: calib 3.96 m, holdout 2.82 m, aynı-maç 1.56 m. Hiçbir sette < 0.5 m "doğru H" yok, bu yüzden marj eşiği seçilemedi (FRR tanımsız). Kabul zinciri bütün yanlış H'leri ve kaydırılmış H'leri reddediyor (yanlış kabul %0); yani görülmemiş maçlarda bu modelle homography kapsaması ~0 olur.
  - Yapısal seçenek (marjdan önce H'yi çizgilere göre iyileştirme: distance transform üzerinde öteleme araması, + ECC; `01b --kp-refine`, `fai.homography.refine_homography`, deneysel): doğru H üretmedi, kaydırılmış H'leri doğruya çekmedi, aynı-maç sette yanlış kabulü %0'dan %6–8'e çıkardı. Pipeline'a bağlanmadı.
  - Eğitim 40 dk; GPU en fazla 69 °C, thermal slowdown yok, bekçi durdurma yapmadı.
  - Açık karar: pose doğruluğu artırılmadan (ör. pose @1280, daha büyük model, daha çok maç / Faz 2 datası) kalibrasyon tekrarlanamaz.

#### Faz 1b — det + ball + klip kontrolü (plan, henüz başlanmadı)
- det (players v10) ve ball (v2) eğitimi; sınıf bazında val metrikleri Roboflow sports notebook'larındaki değerlerle karşılaştırılır.
- 2–3 DFL klibinde `03_check_keypoints` (Kaggle **DFL – Bundesliga Data Shootout**, `clips/` klasörü). Field v12'de olmayan bir maçtan en az bir klip gerekli: field v12'nin 18 maçı `08fd33, 0a2d9b, 0bfacc, 121364, 2e57b9, 42ba34, 4b770a, 538438, 54745b, 55c9d1, 744b27, 798b45, a9f16c, c01561, cd987c, d82e7a, e624c9, ec783a`. Dikkat: players v10 ve ball v2 iki maç daha içeriyor (`40cd38`, `573e61`); yani `573e61_0.mp4` pose için görülmemiş ama det / ball için eğitim maçı. Det ve pose'u birlikte görülmemiş bir maçta denemek için yukarıdaki 18 maç + `40cd38`, `573e61` dışında bir klip gerekli. Öneri: bu 20 maç dışında 1–2 klip + `573e61_0.mp4` (pose için görülmemiş) + aynı-maç referansı olarak `2e57b9_0.mp4` (videodaki klip).

**Bitti:**
- det: player / goalkeeper / referee'de val mAP50 yüksek; ball belirgin düşük çıkar — küçük obje, beklenen durum (ayrı ball modeli bu yüzden var).
- pose: `outputs/keypoint_check/*.png`'de kırmızı projekte çizgiler gerçek beyaz çizgilerin üstüne oturuyor, frame'deki numaralar alttaki şemayla aynı yerde, başlıkta `-> OK` (skor ve marj yazılı). **Bu kontrol atlanmamalı:** keypoint sırası datasetle uyuşmazsa pipeline hata vermez, sessizce yanlış koordinat üretir.

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
Önce kısa bir aralıkla dene (`--duration-s`). Bütün eşikler parametre (`--help`): çizgi kontrolü `--min-line-margin`, `--line-tol-div`, `--line-tophat-thr`, `--min-line-score`; sahne kesmesi `--cut-threshold`; tracker `--track-*`; top `--max-ball-speed*`, `--ball-max-gap-s`. Kullanılan değerler `meta.json` → `pipeline_config` / `post_config`'e yazılır. Ayar yaparken `frames.parquet`'e bak: `reject_reason` hangi kriterin reddettiğini, `line_score` / `line_margin` reddedilenler dahil dağılımı, `cut_dist` kesme eşiğinin nereye konacağını gösterir. Takım sınıflandırıcı maçın kendi oyuncu kırpıntılarıyla fit edilir ve `team_classifier.pkl` olarak saklanır.

**Faz 3 notu (plan):** #9 düzeltmesinde `track_high_conf` = detector conf olduğu için ByteTrack'in düşük güvenli ikinci eşleştirme aşaması hiç çalışmıyor. Gerçek klipte değerlendirilecek: detector conf'u düşür (ör. `--det-conf 0.1`), `--track-high-conf` ~0.3–0.4'te tut, eşleşmeyen düşük güvenli tespitleri datasetten at; etkisi ID kopması sayısıyla birlikte raporlanacak.

**Bitti:** `meta.json`'da `homography_coverage` (sadece doğrulanmış fresh H) yüksek (replay/yakın çekim oranına bağlı), `homography_rejected` sayıları düşük, `scene_cuts` videodaki kesme sayısıyla uyumlu, `render_radar.mp4`'te radar noktaları oyuncularla eşzamanlı hareket ediyor, iki takım renkleri tutarlı.

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
| `track_id` | ByteTrack ID (−1: takip edilmeyen / top); sahne kesmesinden sonra yeni ID'ler öncekilerle çakışmaz |
| `cls`, `role` | detector sınıfı (frame bazında); track bazında çoğunluk sınıfı, hakem dahil (`player`/`goalkeeper`/`referee`/`ball`). Filtrelerde `role` kullan |
| `conf` | detection güveni |
| `team_raw`, `team` | frame bazında / track oylamalı takım (0, 1; −1 = hakem/bilinmiyor; `role == referee` ise her zaman −1) |
| `x1,y1,x2,y2`, `u,v` | bbox ve zemin temas noktası (piksel) |
| `x,y` / `xs,ys` | saha koordinatı (m) / yumuşatılmış (homography olmayan frame'de ikisi de NaN). x: sol kale çizgisinden, y: uzak taç çizgisinden |
| `speed_mps` | yumuşatılmış hız (m/s) |
| `on_pitch` | saha sınırı (+3 m) içinde mi |
| `h_src` | homography kaynağı: `fresh` / `carried` / `none` / `interp` |
| `outlier`, `interp` | top: fiziksel olarak tutarlı en uzun tespit zincirinin dışında / interpolasyonla eklenmiş satır (sadece ≤1 s boşluklar) |

**frames.parquet** — frame başına: `h_src`, `n_kp`; fresh fit adayının `reject_reason` (`''` kabul, `few_kp` / `fit` / `rmse` / `geometry` / `score` / `margin`), `line_score`, `line_margin`, `n_inliers`, `rmse_px` (aday reddedilse de yazılır); taşınan H'nin bu frame'deki `carry_score`, `carry_margin`, `carry_reason`; `cut`, `cut_dist` (sahne kesmesi); `h0..h8` (kullanılan görüntü→saha homography'si). Piksel koordinatlı her yeni bilgiyi (ör. sonradan eklediğin forma numarası tespiti) aynı matrisle sahaya taşıyabilirsin.

**meta.json** — video bilgisi, saha ölçüleri, model yolları, `homography_coverage` (fresh), `homography_carried_frac`, `homography_rejected`, `scene_cuts`, kullanılan `pipeline_config` / `post_config`, kolon açıklamaları.

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
| fresh homography / taşınan (150 frame'in 8'inde keypoint bilerek yok) | %94.7 / %5.3 |

Bu sayılar **model-dışı** kısmın doğruluğunu ölçer; gerçek videoda hata bütçesini detector ve keypoint modelinin kalitesi belirler. Sentetik görüntü (düz yeşil üstünde saf beyaz çizgi) çizgi kontrolü için kolay bir durum; gerçek yayın görüntüsündeki davranış için Faz K / Faz A raporlarına bak. (Faz A'dan önceki HSV skoru burada doğru H için 0.96, yanlış H için 0.01–0.02 veriyordu; gerçek görüntüde ise ikisini ayıramıyordu.)

Entegrasyon: `02_train` (det + pose), `03_check_keypoints`, `04_extract_frames`, `05_autolabel`, `06_build` (ayrı top modeli + slicer dahil), `07_render`, `08_analyze` gerçek ultralytics 8.4 ile CPU'da, sentetik veride küçük modellerle uçtan uca çalıştırıldı. SigLIP backend'i Hugging Face erişimi olmadığı için rastgele ağırlıklı küçük bir modelle test edildi (kod yolu + save/load); gerçek ağırlıklar ilk çalıştırmada indirilir.

## Klasör yapısı
```
fai/            pitch, homography, models, teams, pipeline, postprocess, build, render, analysis, viz, dataio, cli
scripts/        01_download → 01b_calibrate_homography → 01c_prepare_pose_data → 02_train (+ gpu_watchdog) → 03_check_keypoints → 04_extract → 05_autolabel → 06_build → 07_render → 08_analyze
splits/         commit'li split listeleri (field-v12: calib, holdout, exclude)
tests/          test_synthetic.py (modelsiz uçtan uca), test_negative.py (bulgu + kontrol testleri), test_faz_a.py, test_faz_1a.py, make_synth_yolo.py
```

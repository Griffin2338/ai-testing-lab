# od-testkit

Object detection modelleri için pytest tabanlı kabul ve dayanıklılık testleri. Modeli bir YAML dosyasıyla tanımlıyorsunuz, `od-testkit run` testleri koşuyor ve tek dosyalık bir HTML rapor üretiyor.

Trafik kamerasında araç ve yaya sayan bir uygulamayı test ederken mAP değeri tek başına pek bir şey söylemez. Uygulamanın kullandığı eşikte kaç aracı ya da yayayı kaçırdığını, görüntü karardığında ne olduğunu, ONNX'e çevrilen modelin orijinaliyle aynı kutuları verip vermediğini ayrı ayrı ölçmeniz gerekir. Bu kit o ölçümleri tekrar edilebilir hale getiriyor.

*English summary at the bottom.*

## Testler

| ID | Ne ölçer | Yeni etiket gerekir mi |
|---|---|---|
| TC-01 | Operasyon eşiğinde sınıf bazlı precision/recall, bootstrap güven aralığı, AP50 ve AP50-95 | evet |
| TC-02 | Parlaklık ±%40 | hayır, mevcut etiketler |
| TC-03 | 10 bozulma × 5 şiddet: gürültü, bulanıklık, JPEG, sis, düşük ışık... | hayır, mevcut etiketler |
| TC-04 | Yatay çevirme sonrası aynı tespitler | hayır |
| TC-05 | 0.75× / 1.25× ölçek ve kenar dolgusu | hayır, mevcut etiketler |
| TC-06 | Görüntüye nesne yapıştırma: yeni nesne bulunmalı, eskiler değişmemeli | hayır, mevcut etiketler |
| TC-07 | Hedef nesne içermeyen görüntülerde yanlış pozitif | hayır |
| TC-08 | Küçük / orta / büyük nesnelerde recall | evet |
| TC-10 | p50 / p95 / p99 gecikme, FPS | hayır |
| TC-11 | Export paritesi, örneğin ONNX ve PyTorch | evet |
| TC-12 | Modelin emin olup etiketle çeliştiği tespitlerin listesi | evet |
| TC-14 | Güven skoru kalibrasyonu (ECE) | evet |

TC-09 (video zamansal tutarlılık) henüz yok. TC-13 (basılı adversarial patch) fiziksel bir deney olduğu için elle yapılıyor.

Birkaç kural:

- Kapılar `operating_point.conf` değerindeki precision/recall'a bakar. AP raporda sadece bilgi için durur.
- `min_instances` altında kalan sınıf veya dilim "Insufficient data" olarak raporlanır ve pytest'te gerekçesiyle `skipped` görünür. Geçti sayılmaz.
- Güven aralıkları görüntü bazında bootstrap ile hesaplanır, çünkü aynı görüntüdeki kutular birbirinden bağımsız değil.
- Yanlış pozitifler TIDE'daki gibi türlerine ayrılır: tekrar, lokalizasyon, sınıf karışıklığı, arka plan.
- `predict(image) -> Detections` yazabildiğiniz her modeli test edebilirsiniz: YOLO, RT-DETR, ONNX, TensorRT ya da bir REST servisi.

## Kurulum

```bash
pip install -e ".[yolo]"   # Ultralytics adaptörüyle
pip install -e .           # sadece çekirdek
```

## Sentetik demo

Demo dedektörü kırmızı ve mavi şekilleri renk eşiğiyle buluyor ve içinde bilerek bırakılmış üç hata var. Karanlık pikselleri atıyor, küçük nesneleri eliyor ve turuncuyu kırmızı sanıyor. TC-01 geçiyor; hatalar TC-02, TC-03, TC-07 ve TC-08'de çıkıyor.

```bash
od-testkit synth examples/synthetic/data
od-testkit run examples/synthetic/config.yaml --out reports/synthetic
```

Derin öğrenme kütüphanesi gerekmez, koşum bir dakika sürer.

## YOLO11n (ONNX) ve COCO128

```bash
pip install -e ".[yolo]" onnx onnxruntime onnxslim
python examples/coco128/prepare.py
od-testkit run examples/coco128/config.yaml --out reports/coco128
```

`prepare.py`, COCO128'i (~7 MB) ve `yolo11n.pt`'yi indirir, sonra modeli ONNX'e çevirir. COCO128'de 128 görüntü olduğu için `person` dışındaki sınıflar 200 nesne eşiğinin altında kalır.

Lisans notu: Ultralytics paketi ve YOLO11 ağırlıkları AGPL-3.0 lisanslı. Bu repo onları içermiyor, `[yolo]` ekiyle ayrıca kuruluyorlar. Kendi ürününüzde kullanacaksanız Ultralytics lisansını ayrıca okuyun. COCO veri seti CC BY 4.0 lisanslı.

Bir CPU koşumunda (`person`, conf 0.25) çıkan sonuçlar:

- TC-01: AP50 0.78, operasyon eşiğinde recall 0.67, %95 GA [0.58, 0.76]. Üç kişiden biri kaçıyor.
- TC-03: Hareket bulanıklığının en hafif seviyesinde recall 0.38'e iniyor. JPEG, sis ve düşük ışıkta düşüş küçük.
- TC-04: Çevirme sonrası tespitlerin %76'sı yerinde kalıyor. Skoru 0.55 üstündekilerde oran %97.5. Kararsız kutular eşiğin hemen üstünde.
- TC-11: ONNX ile `.pt` aynı mAP'i veriyor (fark +0.2 puan). Ama tespitlerin %8'i farklı kutuyla çıkıyor. ONNX sabit 640×640 giriş alıyor, PyTorch ise dikdörtgen letterbox kullanıyor.
- TC-12: COCO etiketlerinde 7 şüpheli yer listeleniyor. Örneğin etiketlenmemiş bir şişe.

Bu eşikler örnek. Amaç YOLO11n'i değerlendirmek değil, kitin neleri yakaladığını göstermek.

## Kendi modelinizi test etmek

Test setini YOLO formatında hazırlayın: `images/` ve `labels/` klasörleri, her satırda `class cx cy w h`. Sonra `examples/coco128/config.yaml` dosyasını kopyalayıp düzenleyin. En çok şu üç alan önemli:

- `operating_point.conf`: uygulamanın gerçekte kullandığı eşik
- `classes_of_interest`: iş açısından kritik sınıflar
- `min_instances`: bir sınıf hakkında karar vermek için gereken nesne sayısı

Model bir servisin arkasındaysa `examples/http_detector.py` dosyasını kendi API'nize göre değiştirin:

```yaml
detector:
  type: python
  target: examples/http_detector.py:HttpDetector
  kwargs: {url: "http://localhost:8000/detect", class_names: [car, truck, bus, pedestrian]}
```

```bash
od-testkit run my_config.yaml --out reports/rc-1.4 --junit reports/rc-1.4/junit.xml
od-testkit run my_config.yaml -k "tc01 or tc04"
```

Bir kapı başarısız olursa çıkış kodu 1 olur, CI'da release'i durdurmak için bunu kullanabilirsiniz.

## Örneklem büyüklüğü

```bash
od-testkit stats sample-size --p 0.90 --margin 0.03   # 385
od-testkit stats ci --k 873 --n 900                   # Wilson aralığı ve tek taraflı alt sınır
od-testkit stats zero-failure --rate 0.01             # 299
```

## Dosyalar

```
od_testkit/
  structures.py   Detections, Sample
  geometry.py     IoU
  matching.py     eşleştirme ve hata türleri
  metrics.py      operasyon noktası metrikleri, bootstrap, AP, boyut dilimleri, ECE
  stats.py        Wilson, Clopper-Pearson, örneklem büyüklüğü, McNemar
  transforms.py   parlaklık, bozulmalar, flip / ölçek / dolgu, nesne yapıştırma
  detectors.py    Ultralytics, demo dedektörü, python: adaptörü
  checks.py       testlerin kendisi
  suite/          pytest suite'i
  report.py       report.json ve report.html
tests/            kitin kendi testleri (pytest)
examples/         sentetik demo, COCO128, HTTP adaptörü
```

## English summary

od-testkit runs acceptance and robustness tests for object detection models from a YAML config:

- operating-point precision/recall with bootstrap intervals;
- label-free metamorphic tests (flip, scale, padding, object insertion);
- graded corruptions;
- false positives on negative images, size-bucketed recall;
- latency, calibration, export parity, and a label-audit queue.

Classes with too few objects are reported as insufficient data. Every run writes HTML and JSON reports and exits with code 1 when a gate fails. Try it with `od-testkit synth examples/synthetic/data && od-testkit run examples/synthetic/config.yaml`.

MIT license.

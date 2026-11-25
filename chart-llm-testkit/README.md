# chart-llm-testkit

Bir tabloyu ve "sektörlere göre toplam işlem hacmini göster" gibi bir isteği alıp LLM ile grafik çizen uygulamalar için test kiti. Demo verisi uydurma hisse senedi işlemlerinden oluşuyor (`chart_testkit/datasets.py`).

Bu tür uygulamalarda oracle'ı verinin kendisi sağlar. Grafiğin göstermesi gereken sayıları pandas ile hesaplayabilir, çizilen grafikten geri okuduğunuz sayılarla karşılaştırabilirsiniz. Kit bunu yapıyor. Üretilen kodu ayrı bir süreçte çalıştırıyor, matplotlib veya plotly figürünü okuyup bir "chart spec"e çeviriyor, sonra veriyle ve tasarım kurallarıyla kıyaslıyor. Her istek birden çok kez koşuyor, çünkü LLM çıktısı her seferinde aynı olmuyor.

*English summary at the bottom.*

## Bir koşum ne yapar

Her üretim için sırayla şunlar olur:

1. Uygulama kodu üretir ya da netleştirme sorusu sorar.
2. Statik politika kontrolü: izinli importlar, `eval`/`open`/`savefig`/`read_csv` yasağı, dunder erişimi, uydurma veri işaretleri (`np.random`, sayı listesi literalleri, olmayan sütunlar).
3. Sandbox'ta çalıştırma: ayrı Python süreci, API anahtarı ve proxy değişkenleri silinmiş ortam, zaman aşımı, POSIX'te bellek sınırı. Python audit hook'u ağ bağlantısını, alt süreç açmayı ve çalışma klasörü dışına dosya yazmayı engeller.
4. Figürden grafik türü, kategoriler, değerler, eksen aralıkları, etiketler, legend ve tick etiketlerinin çakışması okunur.
5. Case dosyasındaki `expect` kontrolleri çalışır.

Audit hook bir test düzeneğidir, güvenlik sınırı değildir. Üretimde kodu ağı kapalı bir container içinde çalıştırın.

## Case'ler

`cases/default.yaml` içinde 17 case var:

| ID | Ne test eder |
|---|---|
| TC-01 | Toplama doğruluğu: bar yükseklikleri `groupby().sum()` ile ±%0.5 içinde mi (NaN ve boş kategori dahil) |
| TC-02 | Bar grafiğin değer ekseni sıfırdan başlıyor mu |
| TC-03 | "Dosyaya kaydet", "internetten kur çek" gibi isteklerde kod politikası |
| TC-04 | Sandbox: sistem dosyası okuma, ortam değişkeni sızıntısı, dışarı yazma, ağ, kabuk, alt süreç, sonsuz döngü |
| TC-05 | Sütun adına gizlenmiş prompt injection |
| TC-06 | Olmayan sütun istendiğinde soru sorma ya da türetip etikette belirtme, uydurmama |
| TC-07 | Top-N ve sıralama |
| TC-08 | Soruya uygun grafik türü (trend, ilişki, pay) |
| TC-09 | Yığılmış barlarda toplamlar, legend sayısı, birim (USD) |
| TC-10 | 40 uzun kategori adında etiket çakışması ve taşma |
| TC-11 | Metamorfik: satır karıştırma, ×1000 ölçekleme, kategori yeniden adlandırma |
| TC-12 | 20 tekrarın hepsi doğru mu (pass^k) |
| TC-13 | matplotlib, plotly ve seaborn aynı sayıları veriyor mu |
| TC-14 | `traded_value_prev` gibi benzer adlı ve alakasız sütunları kullanmama |
| TC-15 | Birden çok tarih sütunu varken hangisinin kullanıldığını yazma ya da sorma |
| TC-16 | İş kuralı: iptal edilen işlemler net hacimden düşülüyor mu |
| TC-17 | Türkçe istek |

`category: safety` olan case'lerde her koşu geçmek zorunda. Diğerlerinde varsayılan eşik %80. Bunu `min_pass_rate` veya `require_all` ile değiştirebilirsiniz.

Yeni bir case eklemek için YAML'a bir giriş yazmanız yeterli:

```yaml
- id: TC-20
  title: Ortalama işlem tutarı
  dataset: {builtin: trades, seed: 20}
  request: Average trade value by sector.
  expect:
    chart_type: [bar, barh]
    data: {groupby: sector, value: traded_value, agg: mean}
    zero_baseline: true
    labels: true
```

Kullanılabilecek `expect` anahtarları `chart_testkit/oracle.py` içindeki `CHECKS` sözlüğünde. Yanlış yazılmış bir anahtar hata verir, sessizce geçmez.

## Kurulum

```bash
pip install -e ".[seaborn]"          # sahte LLM ile demo
pip install -e ".[claude,seaborn]"   # Claude ile
```

## Demo: sahte LLM

`FakeLLM` sadece bu case'lerdeki istekleri anlayan, kural tabanlı bir LLM taklidi. API anahtarı gerektirmez. İki profili var:

- `careful` doğru kod yazar. Bütün case'ler geçmeli; kiti değiştirdikten sonra kontrol için kullanılır.
- `buggy` gerçek LLM çıktısında görülen hataları yapar: ekseni sıfırdan başlatmamak, veri uydurmak, sütun adındaki talimata uymak, dosyayı ev klasörüne kaydetmek, internetten kur çekmek, Top-N'i ters sıralamak, `traded_value_prev` sütununu seçmek, seaborn'un varsayılan ortalamasını kullanmak, uzun etiketleri döndürmemek, "Date" gibi belirsiz eksen adı yazmak, Türkçe isteği anlamamak ve arada bir toplam yerine ortalama almak.

```bash
chart-testkit run examples/fake.yaml --out reports/fake
chart-testkit run examples/fake-careful.yaml --out reports/fake-careful
```

Bir Windows bilgisayarda buggy profilde 17 case'ten 13'ü kaldı, careful profilde hepsi geçti. Bir koşum dört işçiyle 3 dakika kadar sürdü.

## Claude ile

```bash
export ANTHROPIC_API_KEY=...        # ya da: ant auth login
chart-testkit run examples/claude.yaml --out reports/claude
```

Varsayılan model `claude-opus-5-5`, `effort: medium`. Model bir isteği güvenlik gerekçesiyle reddederse, sunucu tarafında yedek bir modele geçiliyor (`fallbacks: default`); istemiyorsanız config'te `fallbacks: false` yapın. Varsayılan case'lerle bir koşum 145 üretim isteği gönderir; kodu hata veren her koşu için bir onarım isteği daha eklenir.

Tek bir isteği denemek için:

```bash
chart-testkit try examples/claude.yaml "Top 5 tickers by traded value, highest first." --png out.png
```

## Kendi uygulamanızı test etmek

Uygulamanız kod üretip döndürüyorsa `examples/http_app.py` adaptörünü API'nize göre uyarlayın:

```yaml
app:
  type: python
  target: examples/http_app.py:HttpChartApp
  kwargs: {url: "http://localhost:8000/chart-code"}
```

Kit, dönen kodu kendi sandbox'ında çalıştırır. Uygulamanız kodu kendisi çalıştırıp sadece resim döndürüyorsa, test modunda kodu da döndürmesini isteyin. Resimden sayı okumak da mümkün (DePlot, VLM), ama okuma hatası test sonucuna karışır.

## Dosyalar

```
chart_testkit/
  sandbox.py, _child.py   ayrı süreçte çalıştırma ve audit hook
  extract.py              matplotlib / plotly figürü -> chart spec
  policy.py               statik kod kontrolleri
  oracle.py               expect kontrolleri
  datasets.py             hazır test tabloları
  llm.py                  ClaudeLLM, FakeLLM
  app.py                  test edilen demo uygulama ve adaptör arayüzü
  runner.py               tekrarlar, varyantlar, metamorfik çiftler, toplama
  report.py               report.json, report.html (her case için örnek kod ve grafik)
  suite/                  pytest suite'i
cases/default.yaml
examples/                 fake.yaml, fake-careful.yaml, claude.yaml, http_app.py
tests/                    kitin kendi testleri
```

## English summary

chart-llm-testkit tests apps that turn a table and a natural-language request into a chart with an LLM:

- It runs the generated code in a separate process. API keys are stripped from the environment, and an audit hook blocks network, subprocess and out-of-tree writes.
- It reads the matplotlib or plotly figure back into numbers and compares them with pandas ground truth.
- It applies design rules (zero baseline, label overlap, slice count).
- It repeats every request and reports pass rates with Wilson intervals; safety cases must pass every run.

It ships 17 cases, a scripted fake LLM with known bugs for CI, and a Claude backend (`claude-opus-5-5`).

MIT license.

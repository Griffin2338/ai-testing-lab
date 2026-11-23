# ai-testing-lab

[![od-testkit](https://github.com/Griffin2338/ai-testing-lab/actions/workflows/od-testkit.yml/badge.svg)](https://github.com/Griffin2338/ai-testing-lab/actions/workflows/od-testkit.yml)
[![chart-llm-testkit](https://github.com/Griffin2338/ai-testing-lab/actions/workflows/chart-llm-testkit.yml/badge.svg)](https://github.com/Griffin2338/ai-testing-lab/actions/workflows/chart-llm-testkit.yml)

Yapay zekâ bileşeni olan uygulamaları test etmek için küçük, çalıştırılabilir kitler.

| Klasör | İçerik | Durum |
|---|---|---|
| [od-testkit](od-testkit/) | Object detection modelleri için kabul, dayanıklılık ve metamorfik testler | çalışıyor |
| [chart-llm-testkit](chart-llm-testkit/) | LLM ile veriden grafik üreten uygulamaların testi | çalışıyor |
| llm-app-evals | Chatbot / RAG değerlendirme ve güvenlik testleri | planlandı |

## Hızlı deneme

İki demo da GPU ya da API anahtarı istemiyor. Python 3.10 veya üstü yeterli.

```bash
cd od-testkit
pip install -e .
od-testkit synth examples/synthetic/data
od-testkit run examples/synthetic/config.yaml --out reports/synthetic
```

```bash
cd chart-llm-testkit
pip install -e .
chart-testkit run examples/fake.yaml --out reports/fake
```

İki demoda da hatalı bir model bilerek kullanılıyor, bu yüzden bazı testlerin kalması bekleniyor. Raporlar `reports/` altında HTML olarak çıkıyor. Her kitin ayrıntılı kullanımı kendi README dosyasında.

## English

Small, runnable test kits for applications with an AI component:

- **od-testkit:** operating-point precision/recall with confidence intervals, label-free metamorphic tests, corruptions, negatives, latency, export parity and calibration for object detectors.
- **chart-llm-testkit:** runs LLM-generated plotting code in a sandbox, reads the figure back into numbers and checks them against pandas, plus code policy, design rules and repeated runs.

Both demos run on a CPU without API keys. Each kit's README has the details (Turkish, with an English summary at the bottom).

MIT license.

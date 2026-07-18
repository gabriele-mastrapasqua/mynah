<p align="center">
  <img src="assets/mynah-logo3-generic.png" alt="Mynah ASR" width="320">
</p>

# Mynah ASR

**A lightweight native C runtime for multilingual speech recognition & translation.**

Run NVIDIA NeMo speech models anywhere — streaming e offline, senza Python.

```
$ mynah transcribe -m models/parakeet-tdt-0.6b-v3 -i audio.wav --timestamps
[5.2s audio | load 0.04s | inferenza 0.48s | RTF 0.092 | lang=auto]
  0.00   0.56  Ciao,
  0.64   0.72  questo
  ...

$ mynah transcribe -m models/canary-1b-v2 -i italiano.wav --lang it --target-lang en
The satellite, in space, receives the signal and then sends it back almost instantly.

$ rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah stream -m models/nemotron-3.5-...
(trascrizione live cache-aware, latenza 80 ms – 1.12 s a scelta)
```

## Cos'è

Mynah non è "another speech recognizer": è un **runtime nativo per i modelli ASR
moderni**, nello stile di `llama.cpp` / `whisper.cpp`. Encoder FastConformer condiviso,
decoder intercambiabili dietro una vtable (`engine.h`), streaming come cittadino
di prima classe.

- **C11 puro**, zero dipendenze runtime (solo BLAS: Accelerate/OpenBLAS)
- **10 modelli su 4 famiglie di decoder** — RNNT, TDT, CTC e AED (Canary)
- **Speech translation**: Canary traduce il parlato (25 lingue EU ↔ inglese con
  canary-1b-v2, en↔de/es/fr coi flash) — CLI `--target-lang`, server
  `/v1/audio/translations`
- **CPU-first** — RTF 0.015–0.06 offline warm su Apple Silicon (audio lungo);
  backend **Metal** (−25…45%) e CUDA (da validare); int8/int4 con kernel
  SDOT/VNNI nativi
- Streaming **cache-aware** (Nemotron): latenza 80 ms–1.12 s selezionabile a
  runtime, testo mai ritrattato, **byte-identico al percorso offline**
- **40 lingue** con language detection (Nemotron), 25 EU con PnC+ITN
  (Parakeet v3, canary-1b-v2)
- **Timestamp per parola**, segmentazione automatica dei file lunghi sul
  silenzio (model-aware), batch weight-stationary, server REST+WebSocket
  OpenAI-compatible
- **Bindings Python** (ctypes, zero dipendenze) su `libmynah`
- Qualità misurata su **audio reale** ([samples/](samples/README.md) FLEURS,
  CC-BY): CER 0.00–0.07 in 11 lingue, traduzioni valutate contro riferimenti
  paralleli — `make test-samples`
- Python solo come tooling offline (conversione pesi, oracolo di riferimento, eval)

## Stato

**v0.4-dev, feature-complete verso la v1** — 10 modelli funzionanti:

| Modello | Cosa fa | Stato |
|---|---|---|
| [nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | **streaming** cache-aware + offline, 40 lingue, LID | ✅ |
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | offline TDT, 25 lingue EU, PnC+ITN | ✅ |
| [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | TDT+CTC 114M EN — il più veloce | ✅ |
| [parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) / [ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | offline EN | ✅ |
| [parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) / [ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | offline EN, 42 layer | ✅ |
| [canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash) / [1b-flash](https://huggingface.co/nvidia/canary-1b-flash) | ASR en/de/es/fr + **traduzione** + word-ts | ✅ |
| [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | ASR **25 lingue EU** + traduzione en↔24, ITN | ✅ |

RTF su M-series, audio ~65 s warm (matrice completa e metodologia in
[docs/benchmarks.md](docs/benchmarks.md)):

| modello | f32 CPU | Metal | int8 |
|---|---|---|---|
| parakeet-tdt_ctc-110m | 0.015 | **0.010** | 0.015 |
| canary-180m-flash (AED, +traduzione) | 0.060 | 0.054 | **0.030** |
| parakeet-tdt-0.6b-v3 | 0.047 | **0.030** | 0.046 |
| nemotron-3.5-asr-streaming-0.6b | 0.055 | **0.040** | 0.054 |
| parakeet-rnnt/ctc-0.6b | 0.05/0.04 | 0.028/0.022 | ≈f32 |
| parakeet-rnnt/ctc-1.1b | 0.07/0.06 | 0.041/0.033 | ≈f32 |
| canary-1b-flash (AED, +traduzione) | 0.143 | **0.081** | ~0.07 |

RAM: 110m 0.44 GB · 180m 0.71 · 0.6B ~2.4 · 1b-flash 3.3 · 1.1b 4.0 GB
(int8: ~⅓). Streaming Nemotron: ~26 ms/chunk da 80 ms (9 ms int4).

Ogni stadio numerico è validato contro un oracolo numpy di riferimento
(`make test`: mel bit-esatto, encoder a tolleranza f32, streaming ≡ offline).
Roadmap completa: [PLAN.md](PLAN.md) · [TODO.md](TODO.md).

## Quickstart

```sh
# 1. build (macOS: Accelerate; Linux: apt install libopenblas-dev)
make

# 2. scarica il modello (~2.6 GB) e convertilo in-place
scripts/download_model.sh
cd tools && uv sync && uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b && cd ..

# 3. (opzionale) checkpoint quantizzato: 0.79 GB invece di 2.55, load istantaneo
./mynah quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int8

# 4. trascrivi
./mynah transcribe -m models/nemotron-3.5-asr-streaming-0.6b -i file.wav --lang auto  # --quant int8

# 5. streaming da microfono/pipe (raw s16le 16 kHz mono su stdin)
ffmpeg -v quiet -i qualsiasi.mp3 -f s16le -ar 16000 -ac 1 - | \
  ./mynah stream -m models/nemotron-3.5-asr-streaming-0.6b --lookahead 3
```

WAV a sample rate diversi da 16 kHz vengono ricampionati automaticamente
(per mp3/m4a: `ffmpeg -i file.mp3 -ar 16000 -ac 1 out.wav`).
Lingue supportate e preset di latenza: [docs/nemotron-languages.md](docs/nemotron-languages.md)
· [docs/streaming.md](docs/streaming.md).

## CLI

```
mynah transcribe -m <model_dir> -i <file.wav>
    --lang <tag|auto>        lingua sorgente (it-IT, en, auto per la detection)
    --target-lang <xx>       AED/Canary: lingua di USCITA ≠ sorgente = traduzione
    --timestamps             una parola per riga: t0 t1 parola
    --decoder default|ctc    head CTC dei modelli hybrid (più veloce)
    --lookahead N            streaming-preset Nemotron (0|1|3|6|13)
    --segment-sec S          limite per segmento (default model-aware: 30s/300s)
    --quant int8|int4        checkpoint quantizzato (o quantizzazione al load)
    --backend cpu|metal|cuda backend GEMM (fallback CPU)
    --caps auto|scalar|avx2|vnni   livello SIMD x86 (default: cpuid)

mynah stream -m <model_dir> [--lang auto] [--lookahead N] [--quant int8|int4]
    trascrizione live da stdin (raw s16le 16 kHz mono), testo mai ritrattato

mynah quantize -m <model_dir> --quant int8|int4
    salva il checkpoint pre-quantizzato (⅓ della RAM, load zero-copy)

mynah --version
```

## Server (REST + WebSocket, OpenAI-compatible)

```sh
./mynah-server -m models/canary-1b-v2 -p 8090 --threads 4 --batch 8

curl -F file=@audio.wav -F language=it http://localhost:8090/v1/audio/transcriptions
curl -F file=@audio_de.wav -F language=de http://localhost:8090/v1/audio/translations
# WebSocket streaming: GET /v1/audio/stream (PCM in, delta JSON out)
```

`verbose_json` include i timestamp per parola; `--batch N` attiva il micro-batching
weight-stationary tra richieste concorrenti. Dettagli: [docs/server.md](docs/server.md).

## Bindings Python

```python
# make shared && python3 ...
from mynah import Mynah
m = Mynah("models/parakeet-tdt-0.6b-v3")
text, words = m.transcribe("audio.wav", timestamps=True)
Mynah("models/canary-1b-v2").transcribe("it.wav", lang="it>en")   # traduzione
```

ctypes puro, zero dipendenze: [bindings/python/](bindings/python/mynah.py).

## API C (libmynah)

```c
#include "mynah.h"

mynah_model *m = mynah_load("models/parakeet-tdt-0.6b-v3");
char lang[16];
mynah_word *words; int n_words;
char *text = mynah_transcribe_ts(m, samples, n_samples, "auto", -1, lang,
                                 &words, &n_words);   /* o mynah_transcribe */
printf("[%s] %s\n", lang, text);
for (int i = 0; i < n_words; i++)
    printf("%.2f-%.2f %s\n", words[i].t0, words[i].t1, words[i].word);
```

Esempio completo compilabile: [`examples/minimal.c`](examples/minimal.c).
Reference: [docs/api.md](docs/api.md) · streaming: [docs/streaming.md](docs/streaming.md).
`make lib` produce `libmynah.a`.

## Layout

```
src/        runtime C (libmynah) — engine dietro vtable (engine.h)
cli/        CLI `mynah`
server/     `mynah-server` REST + WebSocket
bindings/   Python (ctypes su libmynah)
tools/      tooling Python (uv): convertitore pesi, oracolo numpy, eval
tests/      parità per-stadio vs oracolo + e2e (make test; skip senza modello)
samples/    audio reale CC-BY (FLEURS, 11 lingue) per i test di qualità
docs/       architettura, modelli, lingue, streaming, benchmark
reference/  config/tokenizer estratti dai checkpoint (per sviluppo)
```

## Build & test

```sh
make              # CLI + server (build/ separata, versione da git)
make lib          # libmynah.a        make shared   # .dylib/.so per i bindings
make install      # PREFIX=/usr/local: bin + lib + include
make test         # parità vs oracolo + e2e (exit 77 = skip senza modello)
make test-samples # qualità su audio reale: CER ASR + traduzioni, cpu+metal
make test-server  # REST + concorrenza + WebSocket + translations
make bench        # RTF sui fixture   make leaks / make ubsan / make asan
make golden-dump  # rigenera i dump di riferimento (richiede tools/ + modello)
```

## Licenza

MIT. I pesi dei modelli hanno le rispettive licenze (Nemotron 3.5: OpenMDW-1.1).

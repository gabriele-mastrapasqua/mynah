# Mynah ASR

**A lightweight native C runtime for multilingual streaming and offline speech recognition.**

Run NeMo speech models anywhere. Fast multilingual ASR, without Python.

```
$ mynah transcribe -m models/parakeet-tdt-0.6b-v3 -i audio.wav --timestamps
[5.2s audio | load 0.04s | inferenza 0.48s | RTF 0.092 | lang=auto]
  0.00   0.56  Ciao,
  0.64   0.72  questo
  ...

$ rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah stream -m models/nemotron-3.5-... 
(trascrizione live cache-aware, latenza 80 ms – 1.12 s a scelta)
```

## Cos'è

Mynah non è "another speech recognizer": è un **runtime nativo per i modelli ASR
moderni**, nello stile di `llama.cpp` / `whisper.cpp`. Encoder FastConformer condiviso,
decoder intercambiabili, streaming come cittadino di prima classe.

- **C11 puro**, zero dipendenze runtime (solo BLAS: Accelerate/OpenBLAS)
- **CPU-first** — RTF ~0.09 offline (0.6B, Apple Silicon), ~0.02 col 110M;
  backend Metal opzionale; streaming realtime con ampio margine
- Streaming **cache-aware**: latenza 80 ms–1.12 s selezionabile a runtime,
  testo mai ritrattato, **byte-identico al percorso offline** (verificato nei test)
- **40 lingue** con language detection (Nemotron), 25 EU con PnC+ITN (Parakeet v3)
- **Timestamp per parola** (`--timestamps`), decoder RNNT/TDT/CTC, quantizzazione
  int8/int4, segmentazione automatica dei file lunghi sul silenzio
- Python solo come tooling offline (conversione pesi, oracolo di riferimento, eval)

## Stato

**v0.4-dev** — runtime end-to-end (offline + streaming) con tre famiglie di decoder:

| Modello | Stato |
|---|---|
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — streaming, 40 lingue | ✅ funzionante |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — offline TDT, 25 lingue EU, PnC+ITN | ✅ funzionante |
| [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) — offline TDT+CTC 114M EN (da `.nemo`) | ✅ funzionante |
| [nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) / [ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) — offline EN | ✅ funzionanti |
| famiglia Canary (ASR + traduzione) | pianificato (v0.8+) |

`make bench` su M-series (CPU, warm — audio fixture 4–5 s):

| modello | RTF | RAM |
|---|---|---|
| parakeet-tdt_ctc-110m | **0.021** | 0.44 GB |
| parakeet-tdt-0.6b-v3 | 0.089 | 2.4 GB |
| parakeet-rnnt/ctc-0.6b | 0.09 | 2.3 GB |
| nemotron-3.5 (offline-chunked) | 0.108 | 2.4 GB |

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

WAV a sample rate diversi da 16 kHz vengono ricampionati automaticamente.
Lingue supportate e preset di latenza: [docs/nemotron-languages.md](docs/nemotron-languages.md)
· [docs/streaming.md](docs/streaming.md).

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
src/        runtime C (libmynah)
cli/        CLI `mynah`
tools/      tooling Python (uv): convertitore pesi, oracolo numpy, eval
tests/      parità per-stadio vs oracolo + e2e (make test; skip senza modello)
docs/       architettura, modelli, lingue, streaming
reference/  config/tokenizer estratti dai checkpoint (per sviluppo)
```

## Build & test

```sh
make            # CLI
make lib        # libmynah.a
make test       # parità vs oracolo + e2e (exit 77 = skip senza modello)
make golden-dump  # rigenera i dump di riferimento (richiede tools/ + modello)
make bench      # RTF sui fixture
make asan       # build con AddressSanitizer/UBSan
```

## Licenza

MIT. I pesi dei modelli hanno le rispettive licenze (Nemotron 3.5: OpenMDW-1.1).

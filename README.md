# Mynah ASR

**A lightweight native C runtime for multilingual streaming and offline speech recognition.**

Run NeMo speech models anywhere. Fast multilingual ASR, without Python.

```
$ mynah transcribe -m models/nemotron-3.5-asr-streaming-0.6b -i audio.wav --lang auto
[5.2s audio | load 0.04s | inferenza 0.91s | RTF 0.17 | lang=it-IT]
Ciao, questo è un test di riconoscimento vocale in italiano.

$ rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah stream -m models/... --lang auto
(trascrizione live, latenza 80 ms – 1.12 s a scelta)
```

## Cos'è

Mynah non è "another speech recognizer": è un **runtime nativo per i modelli ASR
moderni**, nello stile di `llama.cpp` / `whisper.cpp`. Encoder FastConformer condiviso,
decoder intercambiabili, streaming come cittadino di prima classe.

- **C11 puro**, zero dipendenze runtime (solo BLAS: Accelerate/OpenBLAS)
- **CPU-first** — RTF ~0.15 offline su Apple Silicon; streaming realtime con margine ~3×
- Streaming **cache-aware**: latenza 80 ms–1.12 s selezionabile a runtime,
  testo mai ritrattato, **byte-identico al percorso offline** (verificato nei test)
- **40 lingue** con language detection automatica (Nemotron 3.5), punteggiatura nativa
- Python solo come tooling offline (conversione pesi, oracolo di riferimento, eval)

## Stato

**v0.4-dev** — il runtime funziona end-to-end (offline + streaming) con due engine:

| Modello | Stato |
|---|---|
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — streaming, 40 lingue | ✅ funzionante |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — offline TDT, 25 lingue EU, PnC+ITN | ✅ funzionante |
| [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) — offline TDT 114M EN (da `.nemo`) | ✅ funzionante |
| resto famiglia Parakeet (RNNT/CTC) | pianificato (v0.5) |
| famiglia Canary (ASR + traduzione) | pianificato (v0.8+) |

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

mynah_model *m = mynah_load("models/nemotron-3.5-asr-streaming-0.6b");
char lang[16];
char *text = mynah_transcribe(m, samples, n_samples, "auto", -1, lang);
printf("[%s] %s\n", lang, text);
free(text);
mynah_free(m);
```

Streaming: vedi [docs/streaming.md](docs/streaming.md). `make lib` produce `libmynah.a`.

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

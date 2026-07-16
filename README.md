# Mynah ASR

**A lightweight native C runtime for multilingual streaming and offline speech recognition.**

Run NeMo speech models anywhere. Fast multilingual ASR, without Python.

> ⚠️ Stato: **pre-alpha, in sviluppo**. Niente funziona ancora — siamo in fase di
> fondamenta (M0). Vedi [PLAN.md](PLAN.md) e [TODO.md](TODO.md).

## Cos'è

Mynah non è "another speech recognizer": è un **runtime nativo per i modelli ASR moderni**,
nello stile di `llama.cpp` / `whisper.cpp`. Encoder FastConformer condiviso, decoder
intercambiabili (RNNT, TDT, CTC), streaming come cittadino di prima classe.

- **C puro**, zero dipendenze runtime (solo BLAS)
- **CPU-first**, backend Metal/CUDA opzionali (in roadmap)
- Streaming **cache-aware** a bassa latenza (80 ms – 1.12 s, configurabile)
- Python solo come tooling offline (conversione pesi, oracolo, eval)

## Modelli target

| Versione | Modello | Stato |
|---|---|---|
| v1 🎯 | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — streaming, 40 lingue | in sviluppo |
| v0.4+ | famiglia Parakeet (TDT/RNNT/CTC) | pianificato |
| v0.8+ | famiglia Canary (ASR + speech translation) | pianificato |

Catalogo completo e architetture: [docs/models.md](docs/models.md) ·
[docs/architecture-notes.md](docs/architecture-notes.md) ·
[docs/nemotron-arch.md](docs/nemotron-arch.md)

## Layout

```
src/        runtime C (libmynah)
cli/        CLI `mynah`
tools/      tooling Python (uv): convertitore pesi, oracolo, eval
tests/      unit test C + golden test
docs/       documentazione tecnica
reference/  config/tokenizer estratti dai checkpoint (per sviluppo)
scripts/    utilità (download modelli, ...)
```

## Build (per ora: solo lo stub CLI)

```sh
make        # produce ./mynah (stub: stampa versione e usage)
make clean
```

## Download del modello

```sh
scripts/download_model.sh    # → models/nemotron-3.5-asr-streaming-0.6b/
```

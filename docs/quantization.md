# Mynah — Quantizzazione INT8 / INT4

## TL;DR

```sh
# una volta: salva i checkpoint pre-quantizzati (tool integrato, zero dipendenze)
./mynah quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int8   # 0.79 GB
./mynah quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int4   # 0.57 GB

# poi: load ISTANTANEO (mmap zero-copy, il f32 da 2.55 GB non serve più a runtime)
./mynah transcribe -m <dir> -i audio.wav --quant int8
./mynah-server -m <dir> --quant int8
```

## Numeri (Apple Silicon, fixture 5.2 s, cache calda)

| | f32 | int8 prequant | int4 prequant |
|---|---|---|---|
| file pesi | 2.55 GB | **0.79 GB** | **0.57 GB** |
| load | ~0.04 s (mmap) | **0.01 s** | **0.01 s** |
| offline RTF | 0.17 | **0.10** | 0.15 |
| streaming (5.2 s, wall) | 9.2 s | **~2.1 s** | **~1.9 s** |
| testo fixture | riferimento | **identico** | quasi identico (una maiuscola, un "it"/"this") |

I dot small-T (streaming/decode) usano **int8×int8 nativo**: quantizzazione delle
attivazioni per-riga (ricetta qwen-tts) + **SDOT** su ARMv8.2 (`vdotq_s32`, q8 e q4 —
nibble deinterleavati con `vld2q_s8`), **VNNI** (`dpbusd`) o **AVX2** (trick sign/abs
llama.cpp) su x86 per q8; fallback NEON-f32/scalare altrove. Kernel validati da
`tests/test_qmat` (self-test senza modello, gira anche in CI x86).
Streaming int4: RTF ~0.37 (~0.11 s/chunk su budget 0.32) — il formato più leggero è
anche il più veloce, come dev'essere.

## Cosa viene quantizzato

Solo i grandi linear consumati dalle GEMM (~92% dei parametri, ricetta prior-art):
FFN linear1/2, attention q/k/v/o, pointwise conv1/2, joint head. Restano f32:
`relative_k_proj` (T grande a ogni chunk), conv2d subsampling, depthwise conv, LSTM,
embedding, norm, bias.

Schemi: **INT8** per-riga simmetrica (scale [n]); **INT4** per-gruppo da 32 con nibble
packed (scale [n, k/32], stile Q4_0).

## Formato su disco

`model.int8.safetensors` / `model.int4.safetensors`, **autosufficienti**: per ogni
tensore quantizzato `<nome>.q8` (I8 [n,k]) o `<nome>.q4` (U8 [n,k/2]) + `<nome>.scales`
(F32); tutti gli altri tensori copiati f32 verbatim. Il runtime li mappa zero-copy.

Fallback: senza checkpoint pre-quantizzato, `--quant int8|int4` quantizza al load dal
f32 (~9 s, e su macOS le pagine f32 toccate restano contate in RSS — su Linux vengono
rilasciate con madvise).

## Kernel (src/qmat.c)

- `T <= 16` (chunk streaming, decode): dot quantizzato diretto — legge 4×/8× meno byte
  dove il matvec è bandwidth-bound.
- `T > 16` (offline/batch): dequant per-chiamata in scratch + GEMM BLAS.

## Test

`make test` include e2e con `--quant int8` e `--quant int4` (skippati se i checkpoint
non sono stati generati). Leak check: 0 byte anche sul percorso quantizzato.

## Regression CER multilingua (2026-07-17)

Suite completa `test-nemo-langs` (34 locali + 4 varianti, 102 sample Tatoeba/FLEURS,
lingua esplicita, soglia CER 0.3) con `uv run python -m eval.test_langs --quant int8|int4`:

|            | f32   | int8  | int4  |
|------------|-------|-------|-------|
| CER medio  | 0.133 | 0.145 | 0.227 |
| lingue OK  | 37/37 | 36/37 | 35/37 |
| ΔCER>0.05  | —     | 2/38  | 19/38 |

- **INT8 ≈ trasparente**: CER identico al f32 su ~90% delle lingue. Peggiorano solo
  ro-RO (+0.15, già borderline in f32: 0.47) e ja-JP (+0.33 su 3 sample: un sample
  degenerato — rumore statistico da verificare con più dati).
- **INT4 costa sulla suite reale** (i fixture `say` non lo mostravano): metà delle
  lingue perde >0.05 di CER, es-ES e ro-RO scendono sotto soglia. Resta valido per
  lo streaming dove la banda domina, ma per qualità multilingua usare int8.

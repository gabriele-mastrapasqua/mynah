# Benchmarks — measured RTF and RAM

> Measurements from **2026-07-18** on M-series (16 GB), models on local disk,
> **warm** runs (2 warm-ups before measuring — see the methodology note at the
> bottom). Reproducible with `make bench` (short fixtures) + manual runs on the
> long file. RTF = inference time / audio duration (lower = faster; 0.05 ≈ 20×
> faster than realtime).

## Short fixtures (4–5 s, `make bench`) — f32 CPU

| model | RTF | RAM |
|---|---|---|
| parakeet-tdt_ctc-110m | 0.022 | 0.44 GB |
| canary-180m-flash | 0.127 | 0.71 GB |
| parakeet-tdt-0.6b-v3 | 0.097 | 2.35 GB |
| nemotron-3.5-asr-streaming-0.6b | 0.113 | 2.38 GB |
| parakeet-rnnt-1.1b / ctc-1.1b | 0.19 | 4.0 GB |
| canary-1b-flash | 0.37 | 3.3 GB |

On short clips the fixed costs dominate (prompt/decoder warm-up): the real
picture is below, on long audio.

## Long audio (~65 s) — CPU vs Metal vs int8

| model | f32 CPU | Metal | int8 CPU |
|---|---|---|---|
| parakeet-tdt_ctc-110m | 0.015 | 0.010 | 0.015 |
| canary-180m-flash | 0.060 | 0.054 | **0.030** |
| parakeet-tdt-0.6b-v3 | 0.047 | **0.030** | 0.046 |
| nemotron-3.5-asr-streaming-0.6b | 0.055 | **0.040** | 0.054 |
| parakeet-rnnt-0.6b¹ | 0.050 | 0.028 | — |
| parakeet-ctc-0.6b¹ | 0.042 | 0.022 | — |
| parakeet-rnnt-1.1b¹ | 0.068 | 0.041 | — |
| parakeet-ctc-1.1b¹ | 0.062 | 0.033 | — |
| canary-1b-flash¹ | 0.143 | 0.081 | 0.133² |
| canary-1b-v2 | to be measured³ | — | — |

¹ measured before the move to the NAS (same day, same protocol).
² 4 s fixture (long run not re-measured); the AED int8/f32 ratio (~2×) applies
here as well.
³ encoder = 1b-flash (32L) + 8L decoder (2× the flash's): expected ~1.3-1.5× the
1b-flash. Measure with weights on local disk (the NAS numbers don't count).

Key takeaways:
- **Metal** wins on the encoder of every model (−25…45%); the gain shrinks
  where the decoder dominates (canary: the AED decode stays on CPU).
- **int8** is ~neutral on the offline encoder (dequant+sgemm on AMX matches f32)
  but **halves Canary's AED decode** (SDOT dot kernel on the T=1 path) and
  triples Nemotron streaming speed (weight bandwidth).
- int8 checkpoints on disk: 110m 0.15 GB · canary-180m 0.22 GB · 0.6b ~0.8 GB ·
  canary-1b 0.98 GB · 1.1b ~1.2 GB — zero-copy load ~instant.

## Streaming (Nemotron, cache-aware)

~26 ms of compute per 80 ms chunk (f32), ~9 ms with int4+SDOT — realtime with
ample headroom at every latency preset (0/1/3/6/13 lookahead chunks).

## Methodology and known pitfalls

- **Always 2+ warm-ups**: the first run pays the mmap weight page-in (seconds).
- On machines with little RAM (16 GB) models ≥3 GB can pay swap on the first
  runs, especially with Metal (+~half the weights in f16 GPU buffers): trust
  only the steady-state runs.
- **Weights over the network (SMB/NAS) don't hold the page cache** like local
  files: RTF up to 15× worse. Benchmark ONLY with models on local disk.
- NEVER ±INFINITY in code with `-ffast-math` (cost 6.5× in RTF — see
  architecture-notes §6).

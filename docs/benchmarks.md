# Benchmarks — RTF e RAM misurati

> Misure del **2026-07-18** su M-series (16 GB), modelli su disco locale, run
> **warm** (2 warm-up prima della misura — vedi nota metodologia in fondo).
> Riproducibili con `make bench` (fixture corte) + run manuali sul file lungo.
> RTF = tempo di inferenza / durata audio (più basso = più veloce; 0.05 ≈ 20×
> più veloce del realtime).

## Fixture corte (4–5 s, `make bench`) — f32 CPU

| modello | RTF | RAM |
|---|---|---|
| parakeet-tdt_ctc-110m | 0.022 | 0.44 GB |
| canary-180m-flash | 0.127 | 0.71 GB |
| parakeet-tdt-0.6b-v3 | 0.097 | 2.35 GB |
| nemotron-3.5-asr-streaming-0.6b | 0.113 | 2.38 GB |
| parakeet-rnnt-1.1b / ctc-1.1b | 0.19 | 4.0 GB |
| canary-1b-flash | 0.37 | 3.3 GB |

Sui clip corti pesano i costi fissi (prompt/decoder warm-up): il quadro vero è
sotto, sull'audio lungo.

## Audio lungo (~65 s) — CPU vs Metal vs int8

| modello | f32 CPU | Metal | int8 CPU |
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
| canary-1b-v2 | da misurare³ | — | — |

¹ misurati prima dello spostamento su NAS (stessa giornata, stesso protocollo).
² fixture 4 s (long non rimisurato); il rapporto int8/f32 dell'AED (~2×) vale
anche qui.
³ encoder = 1b-flash (32L) + decoder 8L (2× dei flash): atteso ~1.3-1.5× il
1b-flash. Misurare con pesi su disco locale (i numeri dal NAS non valgono).

Letture chiave:
- **Metal** vince sull'encoder di tutti i modelli (−25…45%); il guadagno cala
  dove domina il decoder (canary: l'AED decode resta su CPU).
- **int8** è ~neutro sull'encoder offline (il dequant+sgemm su AMX va come l'f32)
  ma **dimezza l'AED decode** di Canary (kernel dot SDOT sul percorso T=1) e
  triplica la velocità dello streaming Nemotron (banda pesi).
- Checkpoint int8 su disco: 110m 0.15 GB · canary-180m 0.22 GB · 0.6b ~0.8 GB ·
  canary-1b 0.98 GB · 1.1b ~1.2 GB — load zero-copy ~istantaneo.

## Streaming (Nemotron, cache-aware)

~26 ms di calcolo per chunk da 80 ms (f32), ~9 ms con int4+SDOT — realtime con
ampio margine a ogni preset di latenza (0/1/3/6/13 chunk di lookahead).

## Metodologia e trappole note

- **Sempre 2+ warm-up**: il primo run paga il page-in dei pesi mmap (secondi).
- Su macchine con poca RAM (16 GB) i modelli ≥3 GB possono pagare swap ai primi
  run, specie con Metal (+~metà del peso in buffer f16 GPU): fidarsi solo dei
  run a regime.
- **I pesi via rete (SMB/NAS) non tengono la page cache** come i file locali:
  RTF fino a 15× peggiori. Benchmarkare SOLO con modelli su disco locale.
- MAI ±INFINITY nel codice con `-ffast-math` (costò 6.5× di RTF — vedi
  architecture-notes §6).

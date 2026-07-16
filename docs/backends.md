# Mynah — Backend di calcolo (CPU / Metal / CUDA)

Selezione a runtime: `--backend cpu|metal|cuda` (CLI e server). Pattern qwen-tts:
richiesta → `resolve()` → nota su stderr → **fallback graceful a CPU** se il backend
non è disponibile o non compilato. Le GEMM sotto 24 righe restano sempre su CPU
(il round-trip GPU non paga sui chunk streaming).

## CPU (default)

BLAS: Accelerate (macOS) / OpenBLAS (Linux) per le GEMM; kernel propri SDOT/VNNI/AVX2
per i dot quantizzati (vedi [quantization.md](quantization.md)). Su Apple Silicon è
il backend più veloce oggi (AMX): offline RTF 0.10 (int8).

## Metal (macOS, compilato di default)

`src/metal_mps.m`: MPSMatrixMultiplication con **pesi residenti** (MTLBuffer cacheato
per pointer — pattern weight-cache di qwen-tts) e buffer I/O riusabili.

**Stato onesto** (misurato su M-series, 63 s di audio): RTF 0.122 vs **0.102 CPU** —
Accelerate/AMX vince ancora: il commit+wait sincrono per singola GEMM domina.
Perché Metal superi la CPU servono (in TODO): encoding asincrono per layer/modello
(un solo wait), fp16, e batch più grandi. Il backend resta utile come infrastruttura
e su Mac con CPU deboli/occupate.

## CUDA (Linux, `make cuda`)

`src/cuda_gemm.cu`: cuBLAS sgemm con weight-cache residente per-pointer, buffer device
riusabili, stream+handle propri (pattern qwen_tts_cuda.c).

> ⚠️ **Cross-compiled, NON ancora validato su hardware** (stesso approccio usato in
> qwen-tts per il VNNI): su una macchina Linux+CUDA eseguire `make cuda && make test`
> prima di fidarsi. Fallback CPU automatico su ogni errore CUDA.

Evoluzioni previste (TODO, pattern qwen-tts): pesi bf16 residenti + `cublasGemmEx`,
decode fuso residente, CUDA Graphs sul loop streaming.

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

**v2 (fp16 + fusioni)**: pesi residenti convertiti fp16 una volta; FFN fusa
(GEMM→SiLU shader→GEMM in un command buffer, intermedio [T,4096] mai sceso dalla GPU)
e q/k/v in un solo sync. Misurato su M-series, 63 s di audio:
**RTF 0.066 vs 0.077 CPU** (−15%) — e il testo resta identico (fp16 innocuo qui).
Sotto 24 righe (chunk streaming) si resta su CPU. Margini ulteriori: attention e
conv su GPU (oggi CPU tra i sync), command buffer per-layer completo.

## CUDA (Linux, `make cuda`)

`src/cuda_gemm.cu`: cuBLAS sgemm con weight-cache residente per-pointer, buffer device
riusabili, stream+handle propri (pattern qwen_tts_cuda.c).

> ⚠️ **Cross-compiled, NON ancora validato su hardware** (stesso approccio usato in
> qwen-tts per il VNNI): su una macchina Linux+CUDA eseguire `make cuda && make test`
> prima di fidarsi. Fallback CPU automatico su ogni errore CUDA.

Evoluzioni previste (TODO, pattern qwen-tts): pesi bf16 residenti + `cublasGemmEx`,
decode fuso residente, CUDA Graphs sul loop streaming.

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

**v2 (fp16 + fusioni)**: pesi residenti fp16; FFN fusa (GEMM→SiLU shader→GEMM,
un sync) e q/k/v insieme → −15% vs CPU.

**v3 (blocchi interi su GPU)**: attention completa in UN command buffer — qkv,
bias_u/v (shader), relk, GEMM per-head su **viste MPS strided** (zero permute),
softmax+rel_shift+finestra chunked in shader (accumulo float), ctx, o_proj — e conv
module intero (pw1→GLU→dwconv9→LayerNorm→SiLU→pw2, tutti shader propri).
**4 sync per layer** (erano ~10 in v1). Misurato back-to-back su 63 s:
**−21% vs CPU** (RTF 0.085 vs 0.107), testo identico su tutte le lingue provate,
0 leak. Sotto 24 righe (chunk streaming) si resta su CPU.
Margini v4: residual/LN tra i blocchi su GPU (command buffer per-layer completo,
~1 sync/layer), conversioni f32↔f16 in shader, batch server su Metal.

## CUDA (Linux, `make cuda`)

`src/cuda_gemm.cu`: cuBLAS sgemm con weight-cache residente per-pointer, buffer device
riusabili, stream+handle propri (pattern qwen_tts_cuda.c).

> ⚠️ **Cross-compiled, NON ancora validato su hardware** (stesso approccio usato in
> qwen-tts per il VNNI): su una macchina Linux+CUDA eseguire `make cuda && make test`
> prima di fidarsi. Fallback CPU automatico su ogni errore CUDA.

Evoluzioni previste (TODO, pattern qwen-tts): pesi bf16 residenti + `cublasGemmEx`,
decode fuso residente, CUDA Graphs sul loop streaming.

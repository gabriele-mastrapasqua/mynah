# Mynah — Backend di calcolo (CPU / Metal / CUDA)

Selezione a runtime: `--backend cpu|metal|cuda` (CLI e server). Pattern qwen-tts:
richiesta → `resolve()` → nota su stderr → **fallback graceful a CPU** se il backend
non è disponibile o non compilato. Le GEMM sotto 24 righe restano sempre su CPU
(il round-trip GPU non paga sui chunk streaming).

## CPU (default)

BLAS: Accelerate (macOS) / OpenBLAS (Linux) per le GEMM; kernel propri SDOT/VNNI/AVX2
per i dot quantizzati (vedi [quantization.md](quantization.md)). Su Apple Silicon è
il backend più veloce oggi (AMX): offline RTF 0.10 (int8).

**Dispatch ISA a runtime (x86)**: i kernel VNNI e AVX2 sono compilati sempre con
target-attribute (nessun `-march` richiesto: un solo binario release multi-target)
e selezionati via cpuid+xgetbv alla prima chiamata. Override con `--caps
auto|scalar|avx2|vnni` (CLI e server) o env `MYNAH_CAPS` — pattern `--caps` di
qwen-tts; un livello superiore a quello della CPU viene declassato con nota.
Su ARM NEON/SDOT restano compile-time (Apple Silicon ha sempre dotprod).
⚠️ Come per CUDA: i percorsi AVX2/VNNI sono validati da `tests/test_qmat` in CI
Linux x86, non su questo Mac (Rosetta qui non espone AVX2 → livello scalar, testato).

## Metal (macOS, compilato di default)

`src/metal_mps.m`: MPSMatrixMultiplication con **pesi residenti** (MTLBuffer cacheato
per pointer — pattern weight-cache di qwen-tts) e buffer I/O riusabili.

**v2 (fp16 + fusioni)**: pesi residenti fp16; FFN fusa (GEMM→SiLU shader→GEMM,
un sync) e q/k/v insieme → −15% vs CPU.

**v3 (blocchi interi su GPU)**: attention completa in UN command buffer — qkv,
bias_u/v (shader), relk, GEMM per-head su **viste MPS strided** (zero permute),
softmax+rel_shift+finestra chunked in shader (accumulo float), ctx, o_proj — e conv
module intero (pw1→GLU→dwconv9→LayerNorm→SiLU→pw2, tutti shader propri).
4 sync per layer (erano ~10 in v1).

**v4 (encoder intero su GPU, un sync per forward)**: anche lo stream residuo resta
su GPU — **f32 per fedeltà numerica** (LN tra i blocchi e residual add accumulano
in f32 come la CPU; i blocchi restano f16 come in v3). Conversioni f32↔f16 negli
shader; la CPU fa solo due memcpy per forward. Un command buffer per layer,
commit senza wait (la GPU esegue il layer i mentre la CPU encoda i+1), **wait solo
sull'ultimo**. Le layernorm sono threadgroup-per-riga con riduzione `simd_sum`
(letture coalescenti: un thread per riga con stride d costava ~15 ms/layer ed
era il collo di bottiglia della prima v4). Conversione pesi al load via vImage.
Il **batch server** passa da Metal: ogni segmento fa l'encoder intero su GPU
(pesi residenti = weight-stationary comunque), parity 4/4 vs B=1.

Misurato **in-process warm** (scenario server, 63 s, best-of-N nello stesso
minuto — la misura per-processo è dominata da page-in e conversione pesi):
encoder 0.70 s vs 2.1 s della v3; totale **RTF 0.051 vs 0.068 CPU (−25%)**
(v3: 0.072). Testo identico IT/EN/DE/FR/ES, 0 leak. Sotto 24 righe (chunk
streaming) si resta su CPU. `MYNAH_METAL_PROF=1` stampa encode/wait/GPU time.

Con la pipeline CPU ottimizzata (greedy a blocchi, subsampling im2col, mel
sparso — vedi TODO M5 2026-07-17), i totali warm sul 63 s scendono a
**Metal RTF 0.042, CPU 0.060**. Nota: il decode resta SEMPRE su BLAS CPU
anche col backend Metal (determinismo tra backend: stessi logits, stesso testo).

## CUDA (Linux, `make cuda`)

`src/cuda_gemm.cu`: cuBLAS sgemm con weight-cache residente per-pointer, buffer device
riusabili, stream+handle propri (pattern qwen_tts_cuda.c).

> ⚠️ **Cross-compiled, NON ancora validato su hardware** (stesso approccio usato in
> qwen-tts per il VNNI): su una macchina Linux+CUDA eseguire `make cuda && make test`
> prima di fidarsi. Fallback CPU automatico su ogni errore CUDA.

Evoluzioni previste (TODO, pattern qwen-tts): pesi bf16 residenti + `cublasGemmEx`,
decode fuso residente, CUDA Graphs sul loop streaming.

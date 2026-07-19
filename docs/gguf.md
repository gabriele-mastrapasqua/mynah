# Mynah — GGUF weight container

## TL;DR

```sh
# export a converted model to GGUF (weights only — mynah.json/tokens stay as-is)
cd tools && uv run python export_gguf.py ../models/parakeet-tdt_ctc-110m --dtype q8_0

# the runtime picks model.gguf up automatically when model.safetensors is absent
./mynah transcribe -m models/parakeet-tdt_ctc-110m -i audio.wav
```

**safetensors stays the default.** GGUF is an *alternative container* for the
same weights: at load everything is normalized into the same internal tensor
table (F32 zero-copy from mmap; F16/BF16/Q8_0/Q4_0 dequantized to f32), so the
encoder/decoder/Metal/CUDA code paths are identical for both containers — one
code path, per repo rule.

## What is supported

| | |
|---|---|
| GGUF versions | v2 and v3 (v1 has a different, u32-based layout: rejected) |
| ggml tensor types | F32, F16, BF16, Q8_0, Q4_0 — anything else fails with a clear error |
| model config | **always from `mynah.json`** (config-driven rule): the GGUF KV metadata is not trusted for hyperparameters |
| lookup order | `model.int8/int4.safetensors` (with `--quant`) → `model.safetensors` → `model.gguf` |

`export_gguf.py --dtype q8_0|q4_0` quantizes only ≥2-D matrices whose last dim
is a multiple of 32; 1-D tensors (bias, norms, BatchNorm running stats) stay
f32 — quantizing `running_var` corrupts `1/sqrt(var)` (found the hard way:
the model output degenerated to garbage until 1-D tensors were exempted).

File sizes for parakeet-tdt_ctc-110m: f32 459 MB · q8_0 164 MB · q4_0 114 MB.
Note: GGUF quantized types are dequantized to f32 **at load** (RAM = f32 size);
for lowest RAM and the native SDOT/VNNI kernels use mynah's own
`mynah quantize` int8/int4 checkpoints ([quantization.md](quantization.md)).

## Validation

- `tests/test_gguf` (model-free, runs in CI): synthetic GGUF files — dims
  reversal (ggml `ne[0]` = fastest), F32 bit-exact zero-copy, per-type dequant
  tolerances, and a malformed-file battery (bad magic, v1, truncation,
  out-of-file tensor offsets, absurd string lengths, non-power-of-2 alignment,
  unsupported types) that must all fail cleanly.
- `tests/test_gguf.sh` (part of `make test`, skips without model): exports the
  110m to f32 and q8_0 GGUF and checks the f32 transcription is **identical**
  to the safetensors load, q8_0 matches the expected text.

## Not (yet) supported

Q4_K / Q5_K / Q6_K and reading third-party GGUFs (e.g. parakeet.cpp's, which
carry their own metadata/naming) — planned as a separate interop milestone.
The parser is hardened against hostile input (overflow-checked arithmetic,
bounded strings/arrays/recursion, mmap range validation; origin: the keyra
project's parser, reviewed and extended).

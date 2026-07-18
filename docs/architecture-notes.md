# Mynah — Architecture notes (papers + NeMo code)

> Sources verified on **2026-07-16** (fetched from arxiv and the actual sources on GitHub).
> Goal: everything needed to reimplement FastConformer + CTC/RNNT/TDT in C
> with cache-aware streaming. Complementary to [models.md](models.md).

## 1. Foundational papers (in reading order)

| # | Paper | URL | Why we need it |
|---|---|---|---|
| 1 | **Conformer** (Gulati et al. 2020) | https://arxiv.org/abs/2005.08100 | structure of the base block |
| 2 | **FastConformer** (Rekesh et al. 2023) | https://arxiv.org/abs/2305.05084 | 8× dw-separable subsampling, kernel 9 — the encoder of ALL our models |
| 3 | **Cache-aware streaming Conformer** (Noroozi et al., ICASSP 2024) | https://arxiv.org/abs/2312.17279 | THE paper on Nemotron's streaming mechanism — v1 |
| 4 | **RNN-T** (Graves 2012) | https://arxiv.org/abs/1211.3711 | prediction network + joint + blank — v1 decoder |
| 5 | **TDT** (Xu et al., ICML 2023) | https://arxiv.org/abs/2304.06795 | duration prediction, frame skipping — v0.4 decoder |
| 6 | Parakeet-v3 / Canary-v2 tech report | https://arxiv.org/abs/2509.14128 | training/eval of the target models |
| 7 | Stateless prediction network (Ghodsi et al. 2020) | https://storage.googleapis.com/gweb-research2023-media/pubtools/5775.pdf | LSTM-free alternative (in NeMo: `StatelessTransducerDecoder`) |
| 8 | Multi-blank Transducers (TDT precursor) | https://arxiv.org/abs/2211.03541 | historical context |

Key points for the implementation:
- **Conformer block (macaron, pre-norm)**: ½·FFN → rel-pos MHSA → Conv module → ½·FFN →
  final LayerNorm. The FFNs enter the residual with factor **0.5** (`fc_factor`).
- **Conv module**: pointwise Conv1d (d→2d) → GLU → causal depthwise (kernel **9** in
  FastConformer) → norm (batch_norm by default, **layer_norm in streaming models**) → Swish → pointwise (d→d).
- **Rel-pos attention (Transformer-XL)**: `scores = (matrix_ac + matrix_bd)/sqrt(d_k)`,
  with `matrix_ac = (q+pos_bias_u)·kᵀ`, `matrix_bd = rel_shift((q+pos_bias_v)·pᵀ)`,
  `p = linear_pos(pos_emb)`. Learned biases `pos_bias_u/v`.
- **FastConformer vs Conformer**: **8×** subsampling (instead of 4×) with 256-channel
  depthwise-separable conv → 80 ms encoder frame; depthwise kernel 31→**9**.
- **TDT**: the joint produces `vocab+1+num_durations` logits; **two separate softmax/argmax**
  (token and duration); inference: `t += duration`; **blank with duration 0 → forced to 1**
  to guarantee progress; duration 0 with a token ⇒ multiple emissions on the same frame.
- **Stateless pred net**: only embeddings of the last `context_size` tokens, no LSTM —
  loses very little WER with wordpieces. Interesting option to simplify the C code.

## 2. Complete inference pipeline (operational summary)

```
PCM 16 kHz mono
  → preemphasis 0.97
  → STFT: hann window 400 (25 ms), hop 160 (10 ms), n_fft 512     [classic Parakeet: win 320/0.02s]
  → |·|² → mel filterbank (80 or 128 bands, from the config)
  → log(x + 2^-24)                                                 [log_zero_guard add]
  → normalize per_feature (per-band mean/var over the utterance;
    in streaming: online_normalization or none — from the config!)
  → ConvSubsampling dw_striding 8× (3 stride-2 stages; causal for streaming: left_pad 2, right_pad 1)
  → N × ConformerLayer (pre-norm macaron, rel-pos MHSA, dw-conv k=9)
  → decoder:
     CTC:  Linear d_model→vocab+1 → argmax → collapse + dedup blank
     RNNT: pred = Embedding(vocab+1, 640, blank_as_pad: blank ⇒ zero emb = SOS) + LSTM
           joint: enc_proj(enc) ⊕ pred_proj(pred) → ReLU → Linear → vocab+1
           greedy: per frame t, inner loop until blank or max_symbols_per_step
     TDT:  like RNNT but logits [vocab+1 | durations]; double argmax; t += duration
```

`AudioToMelSpectrogramPreprocessor` defaults (NeMo `features.py`): dither 1e-5 (0 in inference),
pad_to 16, mag_power 2.0. **Every value must be read from the model config, never assumed.**

## 3. Cache-aware streaming (the mechanism for v1)

From paper 2312.17279 + `conformer_encoder.py`:

- Training with constrained left context and lookahead → at inference **each frame is computed
  exactly once**, zero recomputation (unlike buffered streaming).
- **Three caches**:
  - `cache_last_channel` `(n_layers, B, last_channel_cache_size, d_model)` — past attention
    K/V per layer (size = left context, e.g. 56 for Nemotron 3.5, 70 for the EN model)
  - `cache_last_time` `(n_layers, B, d_model, kernel−1)` — state of the causal depthwise conv
  - pre-encode cache: `subsampling_factor+1` mel frames for subsampling continuity
- **Attention update**: `key = value = concat([cache, new], dim=time)`; after the computation the
  cache retains the last `last_channel_cache_size` frames, discarding `cache_drop_size`.
- **Multi-lookahead**: a single model, multiple latencies; `att_context_size=[left, right]`
  selectable at runtime (`set_default_att_context_size()`), e.g. Nemotron `[56, 0|1|3|6|13]`
  = 80 ms → 1.12 s.
- Reference NeMo API: `encoder.get_initial_cache_state(B)` +
  `conformer_stream_step(feat, len, cache_ch, cache_time, cache_len, keep_all_outputs,
  drop_extra_pre_encoded, return_transcription)` — returns partials + updated caches +
  RNNT hypothesis to pass to the next chunk.
- Real streaming ⇒ `online_normalization` for the mel (classic per-feature requires
  the entire utterance).

## 4. NeMo code map (verified paths, NVIDIA/NeMo repo, main)

| Component | Path |
|---|---|
| Cache-aware streaming script | `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` |
| Offline transcribe script | `examples/asr/transcribe_speech.py` |
| Encoder | `nemo/collections/asr/modules/conformer_encoder.py` (`ConformerEncoder`, `get_initial_cache_state`, `setup_streaming_params`) |
| Block + conv module | `nemo/collections/asr/parts/submodules/conformer_modules.py` (`ConformerLayer`, `ConformerConvolution`) |
| Rel-pos attention | `nemo/collections/asr/parts/submodules/multi_head_attention.py` (`RelPositionMultiHeadAttention`, `update_cache`) |
| Subsampling | `nemo/collections/asr/parts/submodules/subsampling.py` (`ConvSubsampling`, mode `dw_striding`) |
| Mel preprocessing | `nemo/collections/asr/modules/audio_preprocessing.py` + `parts/preprocessing/features.py` (`FilterbankFeatures`) |
| Prediction + joint | `nemo/collections/asr/modules/rnnt.py` (`RNNTDecoder`, `RNNTJoint`, `StatelessTransducerDecoder`) |
| Greedy RNNT | `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` (`GreedyRNNTInfer._greedy_decode`) |
| Greedy TDT (label-looping) | `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py` |
| CTC decoding | `nemo/collections/asr/parts/submodules/ctc_greedy_decoding.py`, `ctc_decoding.py` |
| .nemo format | `nemo/core/connectors/save_restore_connector.py` |
| Tokenizer | `nemo/collections/common/tokenizers/sentencepiece_tokenizer.py`, `aggregate_tokenizer.py` |

Note: the streaming script also exists in the new reorganized repo **NVIDIA-NeMo/Speech**
(https://github.com/NVIDIA-NeMo/Speech) — same path under `examples/asr/`.

**`.nemo` format**: tar (uncompressed since NeMo ≥1.7) with `model_config.yaml` (full config:
preprocessor/encoder/decoder/joint/decoding/tokenizer), `model_weights.ckpt` (PyTorch
state_dict) and UUID-prefixed artifacts (SentencePiece `*.model`, `vocab.txt`).
`tar xf model.nemo` extracts everything.

NVIDIA docs:
- ASR intro: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
- Models (FastConformer + cache-aware streaming section): https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html

## 5. Existing native ports — to study BEFORE writing code

| Project | URL | What it teaches |
|---|---|---|
| **mudler/parakeet.cpp** ⭐ | https://github.com/mudler/parakeet.cpp | The closest to Mynah: C++17 on ggml, GGUF, quant f16→q4_k, CTC+RNNT+TDT+hybrid (including multilingual v3), **cache-aware streaming with EOU**, "WER 0 vs NeMo", C-API + CLI + server |
| **istupakov/onnx-asr** ⭐ | https://github.com/istupakov/onnx-asr | Pure Python+numpy (no torch): preprocessing and greedy TDT readable in a few hundred lines — excellent oracle/reference |
| k2-fsa/sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | NeMo transducer decoding outside PyTorch, production-grade C++ runtime |
| Frikallo/parakeet.cpp | https://github.com/Frikallo/parakeet.cpp | port on a custom tensor lib with Metal |
| handy-computer/transcribe.cpp | https://github.com/handy-computer/transcribe.cpp | mel→FastConformer→TDT pipeline with timestamps (only partially verified) |
| jason-ni/parakeet.cpp | https://github.com/jason-ni/parakeet.cpp | another port (not examined in depth) |

**Positioning of Mynah relative to parakeet.cpp** (closest competitor): Mynah aims for
(a) pure C with no ggml dependency, qwen-asr/qwen-tts style with its own kernels, (b) **multilingual
streaming Nemotron 3.5 as a first-class citizen** (parakeet.cpp mostly covers Parakeet),
(c) integrated OpenAI-compatible server. To evaluate in M0 whether the GGUF/ggml format is
worthwhile anyway — or whether safetensors mmap (as in qwen-*) remains the choice, given that
HF-native repos already provide them.

## 6. Critical implementation details (checklist for the C code)

- [ ] `fc_factor = 0.5` on the two FFNs of the macaron block (forgetting it = silently wrong output)
- [ ] `rel_shift` in the rel-pos attention (easy to get the indexing wrong)
- [ ] RNNT blank = **last index** of the vocab (Nemotron: 13087; Parakeet: 1024; tdt-v3: 8192)
- [ ] `blank_as_pad`: the blank embedding is zero and serves as SOS in the prediction network
- [ ] TDT: blank with duration 0 forced to 1 (otherwise infinite loop)
- [ ] `max_symbols_per_step` (10 for Nemotron) limits the RNNT inner loop per frame
- [ ] causal subsampling: asymmetric padding (left 2, right 1)
- [ ] conv module norm: batch_norm in offline models, layer_norm in streaming ones — from the config
- [ ] mel normalization: per_feature offline vs online in streaming — from the config
- [ ] `calc_length` for the post-subsampling length: `floor((L+pad−k)/stride)+1` per stage
- [ ] dither: active only in training (1e-5), 0 in inference — do not add noise in the runtime

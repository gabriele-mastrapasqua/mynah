# Parakeet TDT 0.6B V3 — definitive architecture (verified)

> Primary sources, extracted on 2026-07-17 and saved in
> [`reference/parakeet-tdt-0.6b-v3/`](../reference/parakeet-tdt-0.6b-v3/):
> `config.json` + `processor_config.json` + `tokenizer.json` (HF-native port),
> `model_config.yaml` (extracted from the `.nemo` via range request, without downloading the 2.5 GB),
> `safetensors_header.json` (shapes of all 723 tensors, 627M params F32).
> NeMo class: `EncDecRNNTBPEModel` with TDT decoding (nemo target `RNNTDecoder` +
> `RNNTJoint num_extra_outputs: 5`); HF class: `ParakeetForTDT`.

**Target M3/v0.4** — 25 EU languages (it included), automatic PnC, word/segment/char timestamps,
offline (chunked streaming only, not cache-aware). License CC-BY-4.0.

## Differences vs Nemotron 3.5 (what the runtime must add)

| Aspect | Nemotron 3.5 (already supported) | Parakeet TDT v3 | Work in C |
|---|---|---|---|
| Mel normalization | `NA` (none) | **`per_feature`** (per-bin mean/std over the utterance's valid frames) | new config-driven branch in features.c |
| Subsampling | **causal** dw_striding (asymmetric padding) | **non-causal** dw_striding (`causal_downsampling: false`, symmetric padding) | symmetric-padding branch |
| Attention | `chunked_limited` [56,r] | **full** (`att_context_size: [-1,-1]`, `regular`) | simpler case of the existing chunked loop |
| Conv module | **causal** depthwise, norm **layer_norm** | **symmetric** depthwise (pad 4/4), norm **batch_norm** (running stats) | fold BN→scale+shift at load/convert; symmetric padding |
| Decoder | pure RNNT | **TDT**: head 8198 = 8193 tokens + **5 durations [0,1,2,3,4]** | `decoder_tdt.c`: greedy with `duration`-frame skips |
| Language prompt | one-hot 128 POST-encoder + prompt_projector | **no prompt** (LID implicit in the vocab) | prompt path skipped via config |
| Vocab / blank | 13087 + blank **13087** | 8192 + blank **8192** (HF vocab_size 8193) | already config-driven |
| Mel bins / d_model / layers / heads | 128 / 1024 / 24 / 8 | **identical** | — |

Also identical: `use_bias: false` on FFN/attn, ff_expansion 4 (FFN 4096) with macaron
fc_factor 0.5, SiLU, k9 conv, rel_pos with `relative_k_proj` + per-layer `bias_u/bias_v`
(`untie_biases: true`), `xscaling: false`, prednet LSTM 2L·640 with `blank_as_pad: true`,
joint `ReLU(enc_proj + pred_proj) → head`, max_symbols_per_step 10, dither 1e-5 training-only.

## Complete numeric pipeline

### Feature extractor (`AudioToMelSpectrogramPreprocessor`)
- 16 kHz mono, window 0.025 s = 400 samples (hann), stride 0.01 s = 160, n_fft 512
- 128 mel, log=true, preemphasis 0.97, `pad_to: 0`
- **`normalize: per_feature`**: for each mel bin, subtract the mean and divide by the std
  (ddof=1 in NeMo) computed over the utterance's valid frames. ⇒ The mel depends
  on the ENTIRE utterance: no streaming≡offline identity possible (consistent:
  the model is offline-only).

### Encoder (offline `ConformerEncoder`, 24 layers, d_model 1024)
- **Non-causal** `dw_striding` subsampling 8×, 256 channels (same tensors as Nemotron):
  - `subsampling.layers.0`: Conv2d 1→256, k3, s2 (+bias), **symmetric** padding (1,1)
  - 2 stages: depthwise [256,1,3,3] s2 + pointwise [256,256,1,1] (+bias), padding (1,1)
  - freq flatten: **linear [1024, 4096]** (4096 = 256 ch × **16** bins: 128/8 exactly,
    vs 17 for Nemotron — Nemotron's causal padding leaves one extra bin)
- Conformer block: identical to Nemotron EXCEPT the conv module:
  - pointwise_conv1 [2048,1024,1] → GLU → depthwise [1024,1,9] **symmetric** (pad 4/4) →
    **`conv.norm` = BatchNorm1d** ([1024] weight+bias+running_mean+running_var;
    `num_batches_tracked` to be ignored) → SiLU → pointwise_conv2 — no bias on the convs
    (`convolution_bias: false`), the BN biases yes
  - BN in inference = per-channel affine: `y = (x−μ)/√(σ²+eps) · γ + β` →
    **fold into scale+shift at convert time** (eps 1e-5, verify in the HF code)
- **Full** attention (`att_context_size: [-1,-1]`): no mask, the whole sequence.
  `pos_emb_max_len: 5000` (= 50 min at 8×10 ms... no: 5000 frames × 80 ms = 400 s;
  beyond that, local attention is needed — out of scope for v0.4, long audio via segmentation)

### Projections toward the joint
- `encoder_projector`: Linear 1024→640 (+bias)
- `decoder.decoder_projector`: Linear 640→640 (+bias) on the LSTM output

### TDT decoder (`RNNTDecoder` + `RNNTJoint num_extra_outputs=5`)
- `decoder.embedding` [8193, 640], blank id **8192** = zero row = SOS (`blank_as_pad`)
- LSTM 2L·640 (weight_ih/hh [2560,640] = 4 gates × 640), state advances only on non-blank
- Joint: `ReLU(enc_proj[t] + pred_proj) → head [8198, 640]` (+bias)
- **Logits [8198] = [8193 tokens (incl. blank 8192) | 5 durations]**. In HF
  `generation_config.suppress_tokens: [8193..8197]` = the duration positions excluded
  from the token argmax.
- **Greedy TDT** (difference from greedy RNNT):
  1. `tok = argmax(logits[0:8193])`, `dur = argmax(logits[8193:8198])` → duration ∈ {0,1,2,3,4}
  2. if `tok != blank`: emit, advance the LSTM
  3. `t += dur`; if `dur == 0` and `tok == blank`: `t += 1` (avoids stalling — verify
     the exact rule in the NeMo/HF greedy: `_greedy_decode_blank_as_pad` TDT)
  4. max_symbols_per_step 10 per frame, as in RNNT
  - Note: the blocking over blank runs from the Nemotron decode must be generalized:
    with TDT, `dur>1` skips already reduce the steps (~high RTFx), the GEMM blocking
    remains useful but the grid of visited frames is no longer contiguous.
- Timestamps nearly free: emission frame × 80 ms (+ subsampling offset)

### Tokenizer
- HF `tokenizer.json` (BPE) / SPE `tokenizer.model` in the .nemo — vocab **8192** + blank
- Control tokens in the vocab (ids 0–9): `<unk>`, `<|nospeech|>`, `<pad>`, `<|endoftext|>`,
  `<|startoftranscript|>`, `<|pnc|>`, `<|nopnc|>`, `<|startofcontext|>`, `<|itn|>`,
  `<|noitn|>` — legacy of the multitask training; in greedy decode they must be **stripped**,
  like Nemotron's language tags (verify whether the model actually emits them)
- 25 languages: bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk —
  auto-LID, **no language token expected in the output** (unlike Nemotron)

## aux_ctc in the YAML
As with Nemotron: auxiliary training-only CTC head (interctc), **not present in the exported
HF weights** (verified: no `ctc*` tensor in the header) — the runtime ignores it.

## Points verified during implementation (2026-07-17)
Reference vendored in `reference/transformers-parakeet/`; all closed:
1. `per_feature` (from `ParakeetFeatureExtractor`): per-bin mean over the valid frames,
   variance **ddof=1**, `x = (x-μ)/(σ+1e-5)`; valid frames = S//hop as with Nemotron.
2. BatchNorm eps **1e-5** (default `nn.BatchNorm1d`), folded into scale+shift at load.
3. Greedy TDT (from `ParakeetTDTGenerationMixin`): at EVERY step the frame advances by the
   predicted duration (`durations[argmax(logits[V:])]`), even on non-blank;
   blank with dur 0 → forced to 1; non-blank with dur 0 → re-emits on the same frame
   (per-frame max_symbols guard, NeMo semantics). No RNNT inner/outer loop.
4. Symmetric padding confirmed: `Conv2d(padding=1)` in time AND freq → 16 freq bins.
C per-stage parity vs the oracle and identical transcriptions on it/en/de/fr/es in `make test`.

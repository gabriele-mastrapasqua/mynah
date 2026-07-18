# Nemotron 3.5 ASR Streaming 0.6B — definitive architecture (verified)

> Primary sources, extracted on 2026-07-16 and saved in
> [`reference/nemotron-3.5-asr-streaming-0.6b/`](../reference/nemotron-3.5-asr-streaming-0.6b/):
> `config.json` + `processor_config.json` (HF port), `model_config.yaml` + `tokenizer.model`
> + `vocab.txt` (extracted from the `.nemo` via range request, without downloading the 2.4 GB),
> `safetensors_header.json` (shapes of all 655 tensors, via range request on the header).
> **All open items from M0.1 are closed.** NeMo class:
> `EncDecRNNTBPEModelWithPrompt` (nemo 2.8.0rc0); HF class: `Nemotron3_5AsrForRNNT`.

## Answers to the previously open questions

| Question | Verified answer |
|---|---|
| Prediction network type | **LSTM, 2 layers, hidden 640** (`decoder.lstm.*`: weight_ih/hh `[2560,640]` = 4 gates × 640) |
| Joint dim | **640** (`joint_hidden: 640`, activation **ReLU**) |
| Feature normalization | **`normalize: "NA"` — NO mel normalization!** Great news for streaming: no per-utterance statistics to manage |
| Tokenizer | **SentencePiece BPE** (`type: bpe`), "universal merged tokenizer" 40 languages, vocab **13087** + blank = 13088 outputs |
| TDT decoder? | No: **pure RNNT** (`durations: []`, NeMo target `RNNTDecoder`) |
| Conv module norm | **layer_norm** (`conv_norm_type: layer_norm`; confirmed by the weights: only weight+bias, no running stats) |
| aux_ctc in the YAML | Auxiliary CTC head **training-only** (loss weight 0.1) — not exported in the HF weights: the runtime ignores it |

## Complete numeric pipeline

### Feature extractor (`AudioToMelSpectrogramPreprocessor`)
- 16 kHz mono, **window 0.025 s = 400 samples** (hann), **stride 0.01 s = 160**, **n_fft 512**
- **128 mel**, log=true, preemphasis 0.97, dither 1e-5 (training; 0 in inference)
- `pad_to: 0`, `normalize: "NA"` → the log-mel goes into the model as-is

### Encoder (cache-aware `ConformerEncoder`, 24 layers, d_model 1024)
- `use_bias: false` on FFN/attention (confirmed: no bias in the linear weights)
- **Causal** `dw_striding` subsampling 8×, 256 channels:
  - `conv_in`: Conv2d 1→256, k3, s2 (+bias)
  - 2 stages: depthwise Conv2d 256 k3 s2 + pointwise 256→256 k1 (+bias)
  - freq flatten: **linear [1024, 4352]** (4352 = 256 ch × 17 residual freq bins from 128 mel /8)
- **Language prompt — POST-encoder** (verified in `reference/transformers-nemotron3_5_asr/`
  `modular_nemotron3_5_asr.py:384-403`): the encoder runs WITHOUT the prompt; the one-hot 128
  (id from `prompt_dictionary`, `auto`=101) is concatenated to the **encoder output**
  (1024+128=1152) → `prompt_projector`: Linear 1152→2048 → ReLU → Linear 2048→1024 →
  `encoder_projector` 1024→640. Dictionary of 105 locales in the processor_config.
  (Constant over time ⇒ in streaming it is applied per chunk on the valid frames.)
- Conformer block (pre-norm macaron, per-layer tensors):
  - `norm_feed_forward1` (LN) → FFN1: linear1 [4096,1024] → SiLU → linear2 [1024,4096], residual ×0.5
  - `norm_self_att` (LN) → rel-pos MHSA: q/k/v/o_proj [1024,1024] without bias,
    `relative_k_proj` [1024,1024] (= linear_pos), `bias_u`/`bias_v` [8,128] per-head
    (untie_biases: true → distinct biases per layer)
  - `norm_conv` (LN) → conv module: pointwise_conv1 [2048,1024,1] → GLU → depthwise [1024,1,9]
    **causal** → `conv.norm` (**LayerNorm** [1024]) → SiLU → pointwise_conv2 [1024,1024,1] — no bias
  - `norm_feed_forward2` → FFN2 (same as FFN1), residual ×0.5
  - `norm_out` (final LN of the layer)
- **`chunked_limited`** attention: `att_context_size` = [[56,3],[56,0],[56,6],[56,13]]
  (default [56,3] = 320 ms; right = lookahead in 80 ms frames). `xscaling: false`
  (no √d scaling on the input embedding), pos_emb_max_len 5000.

### Projections toward the decoder
- `encoder_projector`: Linear 1024→640 (+bias) — maps the encoder into the joint space

### RNNT decoder
- `decoder.embedding` [13088, 640] with `blank_as_pad: true` (blank id **13087** ⇒ zero row,
  used as SOS)
- `decoder.lstm`: 2 layers, hidden 640
- `decoder.decoder_projector`: Linear 640→640 (+bias)
- Joint: `ReLU(enc_proj(enc) + dec_proj(pred))` → `joint.head`: Linear 640→**13088** (+bias)
- Greedy: `max_symbols_per_step: 10`; NeMo default strategy `greedy_batch`

### Parameter count: 637,997,088 (all F32 in the safetensors, 2.55 GB)

| Module | Tensors |
|---|---|
| encoder.layers (24×26) | 624 |
| encoder.subsampling | 12 |
| encoder_projector | 2 |
| prompt_projector | 4 |
| decoder (embedding + lstm + projector) | 11 |
| joint.head | 2 |
| **total** | **655** |

## Details from the transformers reference (verified against the code, 2026-07-16)

Downloaded into `reference/transformers-nemotron_asr_streaming/` (implementation) and
`reference/transformers-nemotron3_5_asr/` (3.5 variant with the prompt). Key points:

- **Rel pos encoding (Transformer-XL)**: positions from `L−1` down to `−(L−1)` (L = chunk +
  cached_frames), `inv_freq = 1/10000^(2i/d)`, **interleaved** layout `[sin,cos,sin,cos,…]`,
  computation forced to float32. `input_scale = 1.0` (`scale_input: false`).
- **Attention**: `matrix_bd = (q+bias_v)·rel_k(pos)ᵀ` → `rel_shift` (pad 1 on the left,
  view [.., L+1?, q] skip first row) → slice to the first `total_key_length` → ×1/√dk →
  the mask (chunked_limited + validity) is applied with `-inf` **on matrix_bd**, which then
  acts as an additive bias to `softmax((q+bias_u)·kᵀ/√dk + matrix_bd)`.
- **chunked_limited**: `chunk = right+1`, `left_chunks = left//chunk`, visible iff
  `0 ≤ q_chunk − kv_chunk ≤ left_chunks`.
- **Conv module**: depthwise **always causal** (pad left k−1=8, right 0) even offline;
  order: pw1 → GLU → zero the padding frames → causal dw → LayerNorm(B,T,C) → SiLU → pw2.
- **Causal subsampling**: freq axis pad (2,1) always in the forward; time axis pad (2,1)
  offline; in streaming the Conv2d cache keeps `left_pad = k−stride = 1` frame (+1 extra
  zero `init_pad` only on the first chunk). Per-stage length: `floor(len/2)+1`.
  ReLU after each stage; flatten `[B,T,C·F']` channel-major; then Linear 4352→1024.
- **Feature extractor**: `torch.stft(center=True, pad_mode="constant")`, Hann
  `periodic=False` 400; valid frames = `floor(L/hop)`; frames beyond the valid ones **zeroed**;
  never any normalization. Filterbank: `librosa.filters.mel(norm="slaney")`. Streaming:
  subsequent chunks with `center=False` on `audio[hop·frame − n_fft/2 :]` reproduce
  the offline pass frame-by-frame.
- **Streaming mel chunks** (for lookahead `r`): first chunk = `1 + 8r` mel frames,
  subsequent ones = `8(r+1)` (e.g. r=6: 49 then 56). The last chunk must be padded to the exact size.
- **Decoder**: SOS = **blank** (first `decoder_input_ids` = blank_id); **all-blank fast
  path**: if the last token is blank, the LSTM does not run and the cached output is
  reused (⇒ state advances only on non-blank emission, like NeMo). `nn.LSTM batch_first`,
  `decoder_projector` on the output.
- **Joint**: `head(ReLU(enc_640 + dec_640))` → 13088. Greedy loop: frame idx advances on
  blank or after `max_symbols_per_step`; stop when the encoder is exhausted; in streaming,
  when the decoder consumes all frames, the next chunk is encoded and appended.

## Implications for the Mynah runtime

1. **No mel normalization** → the streaming features pipeline is simpler than expected
   (the online-normalization problem flagged in the plan goes away).
2. **Tiny joint** (640→13088 once per step): the dominant cost is the encoder;
   the greedy RNNT decode is nearly free on CPU.
3. **`use_bias: false`** almost everywhere in the encoder → matmul kernels without bias-add; bias only
   in subsampling, projector, LSTM and head.
4. **SiLU everywhere** (FFN and conv module) — not a learned Swish-β, it is the standard SiLU.
5. The **language prompt** enters AFTER the encoder (on the output, before the encoder_projector):
   in streaming it is applied to the valid frames of each chunk; with `auto` (101) it performs language
   detection and emits the language tag in the output.
6. **BPE vocab 13087 + blank**: the tokenizer decode is the easy part (id → piece → text,
   `▁` handling); `vocab.txt` is in "##suffix" format (WordPiece-style, for display — the
   real reference is the SentencePiece `tokenizer.model`).
7. The HF port (`model.safetensors`, F32) has clean tensor names (`encoder.layers.N.*`) →
   **the converter starts from there**, not from the pickle `.ckpt` inside the `.nemo`.
8. Watch out for the two lookahead files: HF `config.json` uses `sliding_window: 57` (56+1) and
   `supported_num_lookahead_tokens: [3,0,6,13]`; the NeMo YAML `att_context_size [[56,3],...]`.
   Same semantics, different nomenclature — the Mynah `config.json` normalizes to `[left, right]`.

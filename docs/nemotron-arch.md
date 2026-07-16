# Nemotron 3.5 ASR Streaming 0.6B — architettura definitiva (verificata)

> Fonti primarie, estratte il 2026-07-16 e salvate in
> [`reference/nemotron-3.5-asr-streaming-0.6b/`](../reference/nemotron-3.5-asr-streaming-0.6b/):
> `config.json` + `processor_config.json` (porting HF), `model_config.yaml` + `tokenizer.model`
> + `vocab.txt` (estratti dal `.nemo` via range request, senza scaricare i 2.4 GB),
> `safetensors_header.json` (shape di tutti i 655 tensori, via range request sull'header).
> **Tutti i punti aperti di M0.1 sono chiusi.** Classe NeMo:
> `EncDecRNNTBPEModelWithPrompt` (nemo 2.8.0rc0); classe HF: `Nemotron3_5AsrForRNNT`.

## Risposte ai punti che erano aperti

| Domanda | Risposta verificata |
|---|---|
| Tipo prediction network | **LSTM, 2 layer, hidden 640** (`decoder.lstm.*`: weight_ih/hh `[2560,640]` = 4 gate × 640) |
| Dim joint | **640** (`joint_hidden: 640`, activation **ReLU**) |
| Normalizzazione feature | **`normalize: "NA"` — NESSUNA normalizzazione mel!** Ottima notizia per lo streaming: niente statistiche per-utterance da gestire |
| Tokenizer | **SentencePiece BPE** (`type: bpe`), "universal merged tokenizer" 40 lingue, vocab **13087** + blank = 13088 output |
| Decoder TDT? | No: **RNNT puro** (`durations: []`, target NeMo `RNNTDecoder`) |
| Norm del conv module | **layer_norm** (`conv_norm_type: layer_norm`; confermato dai pesi: solo weight+bias, niente running stats) |
| aux_ctc nel YAML | Head CTC ausiliaria **solo di training** (loss weight 0.1) — non esportata nei pesi HF: il runtime la ignora |

## Pipeline numerica completa

### Feature extractor (`AudioToMelSpectrogramPreprocessor`)
- 16 kHz mono, **window 0.025 s = 400 campioni** (hann), **stride 0.01 s = 160**, **n_fft 512**
- **128 mel**, log=true, preemphasis 0.97, dither 1e-5 (training; 0 in inference)
- `pad_to: 0`, `normalize: "NA"` → il log-mel va nel modello così com'è

### Encoder (`ConformerEncoder` cache-aware, 24 layer, d_model 1024)
- `use_bias: false` su FFN/attention (confermato: nessun bias nei pesi dei linear)
- Subsampling `dw_striding` **causale** 8×, canali 256:
  - `conv_in`: Conv2d 1→256, k3, s2 (+bias)
  - 2 stadi: depthwise Conv2d 256 k3 s2 + pointwise 256→256 k1 (+bias)
  - flatten freq: **linear [1024, 4352]** (4352 = 256 ch × 17 bin freq residui da 128 mel /8)
- **Language prompt — POST-encoder** (verificato in `reference/transformers-nemotron3_5_asr/`
  `modular_nemotron3_5_asr.py:384-403`): l'encoder gira SENZA prompt; l'one-hot 128
  (id da `prompt_dictionary`, `auto`=101) si concatena all'**output dell'encoder**
  (1024+128=1152) → `prompt_projector`: Linear 1152→2048 → ReLU → Linear 2048→1024 →
  `encoder_projector` 1024→640. Dizionario di 105 locale nel processor_config.
  (Costante nel tempo ⇒ nello streaming si applica per chunk sui frame validi.)
- Blocco Conformer (pre-norm macaron, tensori per layer):
  - `norm_feed_forward1` (LN) → FFN1: linear1 [4096,1024] → SiLU → linear2 [1024,4096], residual ×0.5
  - `norm_self_att` (LN) → MHSA rel-pos: q/k/v/o_proj [1024,1024] senza bias,
    `relative_k_proj` [1024,1024] (= linear_pos), `bias_u`/`bias_v` [8,128] per-head
    (untie_biases: true → bias distinti per layer)
  - `norm_conv` (LN) → conv module: pointwise_conv1 [2048,1024,1] → GLU → depthwise [1024,1,9]
    **causale** → `conv.norm` (**LayerNorm** [1024]) → SiLU → pointwise_conv2 [1024,1024,1] — no bias
  - `norm_feed_forward2` → FFN2 (come FFN1), residual ×0.5
  - `norm_out` (LN finale del layer)
- Attention **`chunked_limited`**: `att_context_size` = [[56,3],[56,0],[56,6],[56,13]]
  (default [56,3] = 320 ms; right = lookahead in frame da 80 ms). `xscaling: false`
  (niente scala √d sull'embedding input), pos_emb_max_len 5000.

### Proiezioni verso il decoder
- `encoder_projector`: Linear 1024→640 (+bias) — porta l'encoder nello spazio joint

### Decoder RNNT
- `decoder.embedding` [13088, 640] con `blank_as_pad: true` (blank id **13087** ⇒ riga zero,
  usata come SOS)
- `decoder.lstm`: 2 layer, hidden 640
- `decoder.decoder_projector`: Linear 640→640 (+bias)
- Joint: `ReLU(enc_proj(enc) + dec_proj(pred))` → `joint.head`: Linear 640→**13088** (+bias)
- Greedy: `max_symbols_per_step: 10`; strategia NeMo di default `greedy_batch`

### Conteggio parametri: 637,997,088 (tutti F32 nel safetensors, 2.55 GB)

| Modulo | Tensori |
|---|---|
| encoder.layers (24×26) | 624 |
| encoder.subsampling | 12 |
| encoder_projector | 2 |
| prompt_projector | 4 |
| decoder (embedding + lstm + projector) | 11 |
| joint.head | 2 |
| **totale** | **655** |

## Dettagli dalla reference transformers (verificati sul codice, 2026-07-16)

Scaricata in `reference/transformers-nemotron_asr_streaming/` (implementazione) e
`reference/transformers-nemotron3_5_asr/` (variante 3.5 col prompt). Punti chiave:

- **Pos encoding rel (Transformer-XL)**: posizioni da `L−1` a `−(L−1)` (L = chunk +
  cached_frames), `inv_freq = 1/10000^(2i/d)`, layout **interleaved** `[sin,cos,sin,cos,…]`,
  calcolo forzato in float32. `input_scale = 1.0` (`scale_input: false`).
- **Attention**: `matrix_bd = (q+bias_v)·rel_k(pos)ᵀ` → `rel_shift` (pad 1 a sinistra,
  view [.., L+1?, q] skip prima riga) → slice ai primi `total_key_length` → ×1/√dk →
  la mask (chunked_limited + validity) si applica con `-inf` **su matrix_bd**, che poi fa
  da bias additivo a `softmax((q+bias_u)·kᵀ/√dk + matrix_bd)`.
- **chunked_limited**: `chunk = right+1`, `left_chunks = left//chunk`, visibile sse
  `0 ≤ q_chunk − kv_chunk ≤ left_chunks`.
- **Conv module**: depthwise **sempre causale** (pad left k−1=8, right 0) anche offline;
  ordine: pw1 → GLU → zero dei frame padding → dw causale → LayerNorm(B,T,C) → SiLU → pw2.
- **Subsampling causale**: asse freq pad (2,1) sempre nel forward; asse tempo pad (2,1)
  offline; in streaming la cache Conv2d tiene `left_pad = k−stride = 1` frame (+1 zero
  extra `init_pad` solo sul primo chunk). Lunghezza per stadio: `floor(len/2)+1`.
  ReLU dopo ogni stadio; flatten `[B,T,C·F']` channel-major; poi Linear 4352→1024.
- **Feature extractor**: `torch.stft(center=True, pad_mode="constant")`, Hann
  `periodic=False` 400; valid frames = `floor(L/hop)`; frame oltre il valid **azzerati**;
  mai normalizzazione. Filterbank: `librosa.filters.mel(norm="slaney")`. Streaming:
  chunk successivi con `center=False` su `audio[hop·frame − n_fft/2 :]` riproducono
  frame-per-frame il passaggio offline.
- **Chunk mel streaming** (per lookahead `r`): primo chunk = `1 + 8r` frame mel,
  successivi = `8(r+1)` (es. r=6: 49 poi 56). L'ultimo chunk va paddato alla size esatta.
- **Decoder**: SOS = **blank** (primo `decoder_input_ids` = blank_id); **all-blank fast
  path**: se l'ultimo token è blank, la LSTM non gira e si riusa l'output cacheato
  (⇒ stato avanza solo su emissione non-blank, come NeMo). `nn.LSTM batch_first`,
  `decoder_projector` sull'output.
- **Joint**: `head(ReLU(enc_640 + dec_640))` → 13088. Loop greedy: frame idx avanza su
  blank o dopo `max_symbols_per_step`; stop quando l'encoder è esaurito; in streaming
  quando il decoder consuma tutti i frame si encoda il chunk successivo e si appende.

## Implicazioni per il runtime Mynah

1. **Niente normalizzazione mel** → la pipeline features streaming è più semplice del previsto
   (cade il problema online-normalization segnato nel piano).
2. **Joint minuscolo** (640→13088 una volta per step): il costo dominante è l'encoder;
   il decode RNNT greedy è quasi gratis su CPU.
3. **`use_bias: false`** quasi ovunque nell'encoder → kernel matmul senza bias-add; bias solo
   in subsampling, projector, LSTM e head.
4. **SiLU ovunque** (FFN e conv module) — non Swish-β appresa, è la SiLU standard.
5. Il **language prompt** entra DOPO l'encoder (sull'output, prima dell'encoder_projector):
   nello streaming si applica ai frame validi di ogni chunk; con `auto` (101) fa language
   detection ed emette il tag lingua nell'output.
6. **Vocab BPE 13087 + blank**: il decode del tokenizer è la parte facile (id → pezzo → testo,
   gestione `▁`); `vocab.txt` è in formato "##suffix" (stile WordPiece per il display — il
   riferimento vero è `tokenizer.model` SentencePiece).
7. Il porting HF (`model.safetensors`, F32) ha nomi tensori puliti (`encoder.layers.N.*`) →
   **il convertitore parte da lì**, non dal `.ckpt` pickle dentro il `.nemo`.
8. Attenzione ai due lookahead file: HF `config.json` usa `sliding_window: 57` (56+1) e
   `supported_num_lookahead_tokens: [3,0,6,13]`; il YAML NeMo `att_context_size [[56,3],...]`.
   Stessa semantica, nomenclature diverse — il `config.json` Mynah normalizza su `[left, right]`.

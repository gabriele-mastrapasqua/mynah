# Parakeet TDT 0.6B V3 — architettura definitiva (verificata)

> Fonti primarie, estratte il 2026-07-17 e salvate in
> [`reference/parakeet-tdt-0.6b-v3/`](../reference/parakeet-tdt-0.6b-v3/):
> `config.json` + `processor_config.json` + `tokenizer.json` (porting HF-native),
> `model_config.yaml` (estratto dal `.nemo` via range request, senza scaricare i 2.5 GB),
> `safetensors_header.json` (shape di tutti i 723 tensori, 627M param F32).
> Classe NeMo: `EncDecRNNTBPEModel` con decoding TDT (nemo target `RNNTDecoder` +
> `RNNTJoint num_extra_outputs: 5`); classe HF: `ParakeetForTDT`.

**Target M3/v0.4** — 25 lingue EU (it incluso), PnC automatiche, timestamp word/segment/char,
offline (streaming solo chunked, non cache-aware). Licenza CC-BY-4.0.

## Differenze vs Nemotron 3.5 (ciò che il runtime deve aggiungere)

| Aspetto | Nemotron 3.5 (già supportato) | Parakeet TDT v3 | Lavoro nel C |
|---|---|---|---|
| Normalizzazione mel | `NA` (nessuna) | **`per_feature`** (media/std per bin sui frame validi dell'utterance) | nuovo ramo config-driven in features.c |
| Subsampling | dw_striding **causale** (padding asimmetrico) | dw_striding **non-causale** (`causal_downsampling: false`, padding simmetrico) | ramo padding simmetrico |
| Attention | `chunked_limited` [56,r] | **full** (`att_context_size: [-1,-1]`, `regular`) | caso più semplice del loop chunked esistente |
| Conv module | depthwise **causale**, norm **layer_norm** | depthwise **simmetrico** (pad 4/4), norm **batch_norm** (running stats) | fold BN→scale+shift al load/convert; padding simmetrico |
| Decoder | RNNT puro | **TDT**: head 8198 = 8193 token + **5 durations [0,1,2,3,4]** | `decoder_tdt.c`: greedy con salto di `duration` frame |
| Language prompt | one-hot 128 POST-encoder + prompt_projector | **niente prompt** (LID implicita nel vocab) | percorso prompt saltato via config |
| Vocab / blank | 13087 + blank **13087** | 8192 + blank **8192** (vocab_size HF 8193) | già config-driven |
| Mel bins / d_model / layer / head | 128 / 1024 / 24 / 8 | **identici** | — |

Identici anche: `use_bias: false` su FFN/attn, ff_expansion 4 (FFN 4096) con fc_factor 0.5
macaron, SiLU, k9 conv, rel_pos con `relative_k_proj` + `bias_u/bias_v` per-layer
(`untie_biases: true`), `xscaling: false`, prednet LSTM 2L·640 con `blank_as_pad: true`,
joint `ReLU(enc_proj + pred_proj) → head`, max_symbols_per_step 10, dither 1e-5 solo training.

## Pipeline numerica completa

### Feature extractor (`AudioToMelSpectrogramPreprocessor`)
- 16 kHz mono, window 0.025 s = 400 campioni (hann), stride 0.01 s = 160, n_fft 512
- 128 mel, log=true, preemphasis 0.97, `pad_to: 0`
- **`normalize: per_feature`**: per ogni bin mel, sottrai media e dividi per std
  (ddof=1 in NeMo) calcolate sui frame validi dell'utterance. ⇒ Il mel dipende
  dall'INTERA utterance: nessuna identità streaming≡offline possibile (coerente:
  il modello è offline-only).

### Encoder (`ConformerEncoder` offline, 24 layer, d_model 1024)
- Subsampling `dw_striding` 8× **non-causale**, canali 256 (stessi tensori di Nemotron):
  - `subsampling.layers.0`: Conv2d 1→256, k3, s2 (+bias), padding **simmetrico** (1,1)
  - 2 stadi: depthwise [256,1,3,3] s2 + pointwise [256,256,1,1] (+bias), padding (1,1)
  - flatten freq: **linear [1024, 4096]** (4096 = 256 ch × **16** bin: 128/8 esatto,
    vs 17 di Nemotron — il padding causale di Nemotron lascia un bin in più)
- Blocco Conformer: identico a Nemotron TRANNE il conv module:
  - pointwise_conv1 [2048,1024,1] → GLU → depthwise [1024,1,9] **simmetrico** (pad 4/4) →
    **`conv.norm` = BatchNorm1d** ([1024] weight+bias+running_mean+running_var;
    `num_batches_tracked` da ignorare) → SiLU → pointwise_conv2 — no bias sui conv
    (`convolution_bias: false`), i bias di BN sì
  - BN in inference = affine per-canale: `y = (x−μ)/√(σ²+eps) · γ + β` →
    **foldare in scale+shift al convert** (eps 1e-5, verificare nel codice HF)
- Attention **full** (`att_context_size: [-1,-1]`): niente maschera, tutta la sequenza.
  `pos_emb_max_len: 5000` (= 50 min a 8×10 ms... no: 5000 frame × 80 ms = 400 s;
  oltre serve local attention — fuori scope v0.4, long-audio via segmentazione)

### Proiezioni verso il joint
- `encoder_projector`: Linear 1024→640 (+bias)
- `decoder.decoder_projector`: Linear 640→640 (+bias) sull'uscita LSTM

### Decoder TDT (`RNNTDecoder` + `RNNTJoint num_extra_outputs=5`)
- `decoder.embedding` [8193, 640], blank id **8192** = riga zero = SOS (`blank_as_pad`)
- LSTM 2L·640 (weight_ih/hh [2560,640] = 4 gate × 640), stato avanza solo su non-blank
- Joint: `ReLU(enc_proj[t] + pred_proj) → head [8198, 640]` (+bias)
- **Logits [8198] = [8193 token (incl. blank 8192) | 5 durations]**. In HF
  `generation_config.suppress_tokens: [8193..8197]` = le posizioni duration escluse
  dall'argmax token.
- **Greedy TDT** (differenza dal greedy RNNT):
  1. `tok = argmax(logits[0:8193])`, `dur = argmax(logits[8193:8198])` → durata ∈ {0,1,2,3,4}
  2. se `tok != blank`: emetti, avanza LSTM
  3. `t += dur`; se `dur == 0` e `tok == blank`: `t += 1` (evita stallo — verificare
     la regola esatta nel greedy NeMo/HF: `_greedy_decode_blank_as_pad` TDT)
  4. max_symbols_per_step 10 per frame come RNNT
  - Nota: il blocking sulle run di blank del decode Nemotron va generalizzato:
    con TDT i salti `dur>1` riducono già i passi (~RTFx alto), il blocco GEMM
    resta utile ma la griglia dei frame visitati non è più contigua.
- Timestamp quasi gratis: frame di emissione × 80 ms (+ offset subsampling)

### Tokenizer
- HF `tokenizer.json` (BPE) / SPE `tokenizer.model` nel .nemo — vocab **8192** + blank
- Token di controllo nel vocab (id 0–9): `<unk>`, `<|nospeech|>`, `<pad>`, `<|endoftext|>`,
  `<|startoftranscript|>`, `<|pnc|>`, `<|nopnc|>`, `<|startofcontext|>`, `<|itn|>`,
  `<|noitn|>` — retaggio del training multitask; in decode greedy vanno **strippati**
  come i tag lingua di Nemotron (verificare se il modello li emette davvero)
- 25 lingue: bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk —
  auto-LID, **nessun token lingua in output atteso** (a differenza di Nemotron)

## aux_ctc nel YAML
Come per Nemotron: head CTC ausiliaria di training (interctc), **non presente nei pesi
HF esportati** (verificato: nessun tensore `ctc*` nell'header) — il runtime la ignora.

## Punti da verificare nell'oracolo (prima del C)
1. Semantica esatta `per_feature` (ddof, frame validi vs padding) vs HF
   `ParakeetFeatureExtractor` — parità numerica sul mel.
2. eps del BatchNorm e fold corretto (parità su un layer encoder).
3. Regola di avanzamento del greedy TDT su `dur=0` (blank e non-blank) e interazione
   con max_symbols — riferimento: NeMo `rnnt_greedy_decoding.py` TDT / HF ParakeetForTDT.
4. Padding simmetrico subsampling: shape attese dei bin freq (16, non 17).

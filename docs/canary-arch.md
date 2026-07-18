# Canary Flash — architettura verificata (AED)

Fonte: `model_config.yaml` + `model_weights.ckpt` dei `.nemo` ufficiali (2026-07-18)
e sorgenti NeMo (`canary2.py`, `transformer_modules.py`, `transformer_decoders.py`,
`aed_multitask_models.py`). Target M6: `nvidia/canary-180m-flash` (de-risking) e
`nvidia/canary-1b-flash`. Modello NeMo: `EncDecMultiTaskModel` (AED multitask:
ASR + speech translation en↔de/es/fr).

NOTA: l'export HF-native di canary-1b-flash (config.json + model.safetensors) è
SOLO-encoder (1292 tensori `encoder.*`, `nemo_decoder_type: none`): per Canary si
passa SEMPRE dal `.nemo` (estrattore già in convert_nemo.py, esteso per l'AED).

## 180m-flash in numeri

| blocco | valore |
|---|---|
| mel | 128 bin, per_feature, n_fft 512, win 400, hop 160, dither 0 (inference) |
| encoder | FastConformer **17L·512·8h**, ff 4×, k9 **batch_norm**, dw_striding 8× **non-causale**, att **full [-1,-1]**, rel_pos, xscaling false, use_bias true (default) |
| proiezione | `encoder_decoder_proj`: Linear 512→1024 (+bias). Identity se enc==dec hidden |
| transf_encoder | 0 layer (assente nel ckpt: non usato) |
| transf_decoder | Transformer **4L·1024·8h**, inner 4096, **ReLU**, **pre-LN** + final_layer_norm, max_seq 1024 |
| head | `log_softmax.mlp.layer0`: Linear 1024→5248 (softmax non serve per greedy) |
| vocab | 5248 = spl_tokens 1152 + en/de/es/fr 1024 ciascuno (aggregate, offset cumulativi) |

1b-flash: encoder 32L·1024 (proj = Identity, quindi assente dal ckpt), decoder
4L·1024, vocab 16384? (verificare dallo yaml alla conversione). Stessa struttura.

L'encoder è ESATTAMENTE la variante Parakeet non-causale già implementata
(BN foldata, attention full, bias, subsampling simmetrico) — riuso totale,
Metal incluso. Il lavoro nuovo è solo proj + decoder + tokenizer + prompt.

## Decoder Transformer (nomi ckpt → canonici)

```
encoder_decoder_proj.{weight,bias}                    → enc_dec_proj.*   [1024,512]
transf_decoder._embedding.token_embedding.weight      → aed.embedding.weight [5248,1024]
transf_decoder._embedding.position_embedding.pos_enc  → aed.pos_enc [1024,1024] (buffer!)
transf_decoder._embedding.layer_norm.{weight,bias}    → aed.emb_norm.*
transf_decoder._decoder.layers.N.layer_norm_1.*       → aed.layers.N.ln_self.*
  .first_sub_layer.{query,key,value}_net, out_projection → aed.layers.N.self_attn.{q,k,v,o}_proj
transf_decoder._decoder.layers.N.layer_norm_2.*       → aed.layers.N.ln_cross.*
  .second_sub_layer....                               → aed.layers.N.cross_attn.{q,k,v,o}_proj
transf_decoder._decoder.layers.N.layer_norm_3.*       → aed.layers.N.ln_ffn.*
  .third_sub_layer.dense_in/dense_out                 → aed.layers.N.ffn.linear1/linear2
transf_decoder._decoder.final_layer_norm.*            → aed.final_norm.*
log_softmax.mlp.layer0.{weight,bias}                  → aed.head.*
```

Semantiche verificate dai sorgenti NeMo:

- **Embedding**: `emb = LayerNorm(token_emb[id] + pos_enc[pos])` poi dropout (0 in
  inference). NIENTE scala sqrt(d) sul token embedding; la scala è già DENTRO
  `pos_enc` (buffer nel ckpt: sin/cos **divisi per sqrt(hidden)** — prenderlo
  dal ckpt, bit-esatto, come le mel filterbank). Posizioni da 0 (start_pos per
  la generazione incrementale).
- **Blocco pre-LN** (`forward_preln`): `x += SelfAttn(LN1(x))` (mask causale) →
  `x += CrossAttn(LN2(x), enc)` → `x += FFN(LN3(x))`; dopo l'ultimo layer
  `final_layer_norm`.
- **Attention** (self e cross): q e k divisi CIASCUNO per `dk^(1/4)`
  (equivale allo scaling 1/sqrt(dk) standard); softmax pieno sulle posizioni
  valide (self: causale; cross: tutti i frame encoder). Proiezioni con bias.
- **FFN**: `linear2(ReLU(linear1(x)))`, bias presenti.
- **Flusso**: `enc_out [T,512] → enc_dec_proj → [T,1024] → cross-K/V per tutti
  i layer → greedy loop` (beam_size 1 di default nello yaml). La head produce
  logits 5248; argmax; stop a `<|endoftext|>` (id 3) o max_generation_delta
  (50) + len prompt... usare max_seq 1024 come guardia.

## Tokenizer aggregato (CanaryTokenizer, type "agg")

Ordine (= ordine `langs` nello yaml) e offset GLOBALI dei sub-tokenizer SPE BPE:

| sub | size | offset |
|---|---|---|
| spl_tokens | 1152 | 0 |
| en | 1024 | 1152 |
| de | 1024 | 2176 |
| es | 1024 | 3200 |
| fr | 1024 | 4224 |

Conversione: tokens.json = lista piatta dei 5248 pezzi in ordine di id globale.
Detok identica agli altri modelli (▁ → spazio); gli id < 1152 (speciali) NON si
stampano. Speciali chiave (id globali): `<unk>` 0, `<|nospeech|>` 1, `<pad>` 2,
`<|endoftext|>` 3 (EOS), `<|startoftranscript|>` 4, `<|pnc|>` 5 / `<|nopnc|>` 6,
`<|startofcontext|>` 7, `<|itn|>` 8 / `<|noitn|>` 9, `<|timestamp|>` 10 /
`<|notimestamp|>` 11, `<|diarize|>` 12 / `<|nodiarize|>` 13,
`<|emo:undefined|>` 16, lingue: `<|en|>` 62, `<|fr|>` 69, `<|de|>` 76, `<|es|>` 169.

## Prompt format `canary2`

Template user (da `Canary2PromptFormatter`, slot in quest'ordine):

```
<|startofcontext|> [decodercontext] <|startoftranscript|> |emotion| |source_lang| |target_lang| |pnc| |itn| |timestamp| |diarize|
```

Default dallo yaml: decodercontext vuoto (0 token), emotion `<|emo:undefined|>`,
pnc `<|pnc|>`, itn `<|noitn|>`, timestamp `<|notimestamp|>`, diarize `<|nodiarize|>`.
Prompt ASR EN (9 token): `[7, 4, 16, 62, 62, 5, 9, 11, 13]`.
**Traduzione** = target_lang ≠ source_lang (es. en→de: `[7, 4, 16, 62, 76, 5, 9, 11, 13]`).
La risposta del modello segue il prompt e termina con `<|endoftext|>`.

Con `<|timestamp|>` il modello emette token `<|N|>` (frame 80 ms) attorno alle
parole — supporto rimandato (v1 engine: notimestamp).

## Piano di validazione

1. Oracolo numpy: riuso encoder parakeet + `aed_decode()` nuovo; confronto
   trascrizioni sui WAV fixture (en + it non supportato → en/de/es/fr) e
   traduzione en→de/es/fr a vista.
2. C: parità per-stadio vs oracolo (enc_proj, emb, layer 0/N, logits primo step)
   poi e2e testo identico.
3. Golden in `make test` col 180m (scaricabile in CI? 735 MB — valutare).

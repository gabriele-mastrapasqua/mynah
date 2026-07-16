# Mynah — Note di architettura (paper + codice NeMo)

> Fonti verificate il **2026-07-16** (fetch di arxiv e dei sorgenti reali su GitHub).
> Obiettivo: tutto ciò che serve per reimplementare in C FastConformer + CTC/RNNT/TDT
> con streaming cache-aware. Complementare a [models.md](models.md).

## 1. Paper fondamentali (in ordine di lettura)

| # | Paper | URL | Perché ci serve |
|---|---|---|---|
| 1 | **Conformer** (Gulati et al. 2020) | https://arxiv.org/abs/2005.08100 | struttura del blocco base |
| 2 | **FastConformer** (Rekesh et al. 2023) | https://arxiv.org/abs/2305.05084 | subsampling 8× dw-separable, kernel 9 — l'encoder di TUTTI i nostri modelli |
| 3 | **Cache-aware streaming Conformer** (Noroozi et al., ICASSP 2024) | https://arxiv.org/abs/2312.17279 | IL paper del meccanismo streaming di Nemotron — v1 |
| 4 | **RNN-T** (Graves 2012) | https://arxiv.org/abs/1211.3711 | prediction network + joint + blank — decoder v1 |
| 5 | **TDT** (Xu et al., ICML 2023) | https://arxiv.org/abs/2304.06795 | duration prediction, skip dei frame — decoder v0.4 |
| 6 | Tech report Parakeet-v3 / Canary-v2 | https://arxiv.org/abs/2509.14128 | training/eval dei modelli target |
| 7 | Stateless prediction network (Ghodsi et al. 2020) | https://storage.googleapis.com/gweb-research2023-media/pubtools/5775.pdf | alternativa senza LSTM (in NeMo: `StatelessTransducerDecoder`) |
| 8 | Multi-blank Transducers (precursore TDT) | https://arxiv.org/abs/2211.03541 | contesto storico |

Punti chiave per l'implementazione:
- **Blocco Conformer (macaron, pre-norm)**: ½·FFN → MHSA rel-pos → Conv module → ½·FFN →
  LayerNorm finale. Le FFN entrano nel residual con fattore **0.5** (`fc_factor`).
- **Conv module**: pointwise Conv1d (d→2d) → GLU → depthwise causale (kernel **9** in
  FastConformer) → norm (batch_norm default, **layer_norm nei modelli streaming**) → Swish → pointwise (d→d).
- **Attention rel-pos (Transformer-XL)**: `scores = (matrix_ac + matrix_bd)/sqrt(d_k)`,
  con `matrix_ac = (q+pos_bias_u)·kᵀ`, `matrix_bd = rel_shift((q+pos_bias_v)·pᵀ)`,
  `p = linear_pos(pos_emb)`. Bias appresi `pos_bias_u/v`.
- **FastConformer vs Conformer**: subsampling **8×** (invece di 4×) con conv
  depthwise-separable a 256 canali → frame encoder da 80 ms; kernel depthwise 31→**9**.
- **TDT**: il joint produce `vocab+1+num_durations` logits; **due softmax/argmax separati**
  (token e duration); inferenza: `t += duration`; **blank con duration 0 → forzata a 1**
  per garantire progresso; duration 0 con token ⇒ emissione multipla sullo stesso frame.
- **Stateless pred net**: solo embedding degli ultimi `context_size` token, niente LSTM —
  perde pochissimo WER con wordpieces. Opzione interessante per semplificare il C.

## 2. Pipeline di inferenza completa (sintesi operativa)

```
PCM 16 kHz mono
  → preemphasis 0.97
  → STFT: window hann 400 (25 ms), hop 160 (10 ms), n_fft 512     [Parakeet classici: win 320/0.02s]
  → |·|² → mel filterbank (80 o 128 bande, dal config)
  → log(x + 2^-24)                                                 [log_zero_guard add]
  → normalize per_feature (media/var per banda sull'utterance;
    in streaming: online_normalization o nessuna — dal config!)
  → ConvSubsampling dw_striding 8× (3 stadi stride-2; causale per streaming: left_pad 2, right_pad 1)
  → N × ConformerLayer (pre-norm macaron, MHSA rel-pos, dw-conv k=9)
  → decoder:
     CTC:  Linear d_model→vocab+1 → argmax → collapse + dedup blank
     RNNT: pred = Embedding(vocab+1, 640, blank_as_pad: blank ⇒ emb zero = SOS) + LSTM
           joint: enc_proj(enc) ⊕ pred_proj(pred) → ReLU → Linear → vocab+1
           greedy: per frame t, inner loop fino a blank o max_symbols_per_step
     TDT:  come RNNT ma logits [vocab+1 | durations]; doppio argmax; t += duration
```

Default `AudioToMelSpectrogramPreprocessor` (NeMo `features.py`): dither 1e-5 (0 in inference),
pad_to 16, mag_power 2.0. **Ogni valore va letto dal config del modello, mai assunto.**

## 3. Streaming cache-aware (meccanismo per la v1)

Dal paper 2312.17279 + `conformer_encoder.py`:

- Training con left context e lookahead vincolati → in inferenza **ogni frame è calcolato
  una volta sola**, zero ricalcolo (a differenza del buffered streaming).
- **Tre cache**:
  - `cache_last_channel` `(n_layers, B, last_channel_cache_size, d_model)` — K/V passati
    dell'attention per layer (size = left context, es. 56 per Nemotron 3.5, 70 per l'EN)
  - `cache_last_time` `(n_layers, B, d_model, kernel−1)` — stato della depthwise conv causale
  - pre-encode cache: `subsampling_factor+1` frame mel per la continuità del subsampling
- **Update attention**: `key = value = concat([cache, nuovo], dim=time)`; dopo il calcolo la
  cache trattiene gli ultimi `last_channel_cache_size` frame scartando `cache_drop_size`.
- **Multi-lookahead**: un solo modello, più latenze; `att_context_size=[left, right]`
  selezionabile a runtime (`set_default_att_context_size()`), es. Nemotron `[56, 0|1|3|6|13]`
  = 80 ms → 1.12 s.
- API NeMo di riferimento: `encoder.get_initial_cache_state(B)` +
  `conformer_stream_step(feat, len, cache_ch, cache_time, cache_len, keep_all_outputs,
  drop_extra_pre_encoded, return_transcription)` — ritorna parziali + cache aggiornate +
  ipotesi RNNT da passare al chunk successivo.
- Streaming reale ⇒ `online_normalization` per il mel (la per-feature classica richiede
  l'utterance intera).

## 4. Mappa del codice NeMo (path verificati, repo NVIDIA/NeMo main)

| Componente | Path |
|---|---|
| Script streaming cache-aware | `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` |
| Script transcribe offline | `examples/asr/transcribe_speech.py` |
| Encoder | `nemo/collections/asr/modules/conformer_encoder.py` (`ConformerEncoder`, `get_initial_cache_state`, `setup_streaming_params`) |
| Blocco + conv module | `nemo/collections/asr/parts/submodules/conformer_modules.py` (`ConformerLayer`, `ConformerConvolution`) |
| Attention rel-pos | `nemo/collections/asr/parts/submodules/multi_head_attention.py` (`RelPositionMultiHeadAttention`, `update_cache`) |
| Subsampling | `nemo/collections/asr/parts/submodules/subsampling.py` (`ConvSubsampling`, mode `dw_striding`) |
| Preprocessing mel | `nemo/collections/asr/modules/audio_preprocessing.py` + `parts/preprocessing/features.py` (`FilterbankFeatures`) |
| Prediction + joint | `nemo/collections/asr/modules/rnnt.py` (`RNNTDecoder`, `RNNTJoint`, `StatelessTransducerDecoder`) |
| Greedy RNNT | `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` (`GreedyRNNTInfer._greedy_decode`) |
| Greedy TDT (label-looping) | `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py` |
| CTC decoding | `nemo/collections/asr/parts/submodules/ctc_greedy_decoding.py`, `ctc_decoding.py` |
| Formato .nemo | `nemo/core/connectors/save_restore_connector.py` |
| Tokenizer | `nemo/collections/common/tokenizers/sentencepiece_tokenizer.py`, `aggregate_tokenizer.py` |

Nota: lo script streaming esiste anche nel nuovo repo riorganizzato **NVIDIA-NeMo/Speech**
(https://github.com/NVIDIA-NeMo/Speech) — stesso path sotto `examples/asr/`.

**Formato `.nemo`**: tar (non compresso da NeMo ≥1.7) con `model_config.yaml` (config completa:
preprocessor/encoder/decoder/joint/decoding/tokenizer), `model_weights.ckpt` (state_dict
PyTorch) e artifact con prefisso UUID (SentencePiece `*.model`, `vocab.txt`).
`tar xf model.nemo` e si estrae tutto.

Docs NVIDIA:
- ASR intro: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
- Models (FastConformer + sezione cache-aware streaming): https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html

## 5. Porting nativi esistenti — da studiare PRIMA di scrivere codice

| Progetto | URL | Cosa insegna |
|---|---|---|
| **mudler/parakeet.cpp** ⭐ | https://github.com/mudler/parakeet.cpp | Il più vicino a Mynah: C++17 su ggml, GGUF, quant f16→q4_k, CTC+RNNT+TDT+hybrid (anche v3 multilingua), **cache-aware streaming con EOU**, "WER 0 vs NeMo", C-API + CLI + server |
| **istupakov/onnx-asr** ⭐ | https://github.com/istupakov/onnx-asr | Python+numpy puro (no torch): preprocessing e greedy TDT leggibili in poche centinaia di righe — ottimo oracolo/riferimento |
| k2-fsa/sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | decoding transducer NeMo fuori da PyTorch, runtime C++ production-grade |
| Frikallo/parakeet.cpp | https://github.com/Frikallo/parakeet.cpp | porting su tensor lib custom con Metal |
| handy-computer/transcribe.cpp | https://github.com/handy-computer/transcribe.cpp | pipeline mel→FastConformer→TDT con timestamp (verificato solo parzialmente) |
| jason-ni/parakeet.cpp | https://github.com/jason-ni/parakeet.cpp | altro porting (non approfondito) |

**Posizionamento di Mynah rispetto a parakeet.cpp** (concorrente più vicino): Mynah punta a
(a) C puro senza dipendenza ggml, stile qwen-asr/qwen-tts con kernel propri, (b) **Nemotron
3.5 multilingua streaming come primo cittadino** (parakeet.cpp copre soprattutto Parakeet),
(c) server OpenAI-compatible integrato. Da valutare in M0 se il formato GGUF/ggml convenga
comunque — o se safetensors mmap (come qwen-*) resti la scelta, dato che i repo HF-native
li forniscono già.

## 6. Dettagli implementativi critici (checklist per il C)

- [ ] `fc_factor = 0.5` sulle due FFN del blocco macaron (dimenticarlo = output sbagliato silenzioso)
- [ ] `rel_shift` nell'attention rel-pos (facile sbagliare l'indexing)
- [ ] blank RNNT = **ultimo indice** del vocab (Nemotron: 13087; Parakeet: 1024; tdt-v3: 8192)
- [ ] `blank_as_pad`: l'embedding del blank è zero e funge da SOS nella prediction network
- [ ] TDT: blank con duration 0 forzata a 1 (altrimenti loop infinito)
- [ ] `max_symbols_per_step` (10 per Nemotron) limita l'inner loop RNNT per frame
- [ ] subsampling causale: padding asimmetrico (left 2, right 1)
- [ ] norm del conv module: batch_norm nei modelli offline, layer_norm negli streaming — dal config
- [ ] normalizzazione mel: per_feature offline vs online in streaming — dal config
- [ ] `calc_length` per la lunghezza post-subsampling: `floor((L+pad−k)/stride)+1` per stadio
- [ ] dither: attivo solo in training (1e-5), 0 in inference — non aggiungere rumore nel runtime

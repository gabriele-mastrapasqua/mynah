# Mynah — Catalogo modelli target (NVIDIA NeMo speech)

> Riferimenti verificati su HuggingFace il **2026-07-16** (fetch delle model card, dei
> `config.json` e dei file tree reali). Tenere aggiornato quando NVIDIA rilascia varianti.

Tratti comuni a quasi tutta la famiglia (dai `config.json` verificati):
- **Encoder FastConformer**: subsampling conv depthwise-separable **8×**, d_model **1024**,
  **8 attention heads**, conv module kernel **9**. 24 layer = varianti "XL" 0.6B,
  42 layer = "XXL" 1.1B, 17 layer = small.
- **Feature**: 16 kHz mono, log-mel, hop **160** (10 ms), window **400** (25 ms), n_fft **512**,
  preemphasis **0.97**. **80 mel** per i Parakeet classici, **128 mel** per tdt-0.6b-v3 e Nemotron.
- **Tokenizer**: SentencePiece (Unigram 1024 per gli EN classici; unificato 8192 per tdt-v3;
  unificato 13088 per Nemotron 3.5; 16384 per Canary v2).
- Distribuzione: `.nemo` (tar: pesi + `model_config.yaml` + tokenizer); i modelli 2025+
  hanno spesso anche il **porting HF-native** (`model.safetensors` + `config.json` + `tokenizer.json`).

---

## 🎯 Target v1 — Nemotron Speech Streaming

### nvidia/nemotron-3.5-asr-streaming-0.6b ← **IL modello v1**
**https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b** (rilascio 2026-06-04)

- **Architettura** (da `config.json` verificato, classe HF `Nemotron3_5AsrForRNNT`):
  - Encoder cache-aware streaming FastConformer: **24 layer, d_model 1024, 8 heads
    (8 KV), FFN 4096, conv kernel 9, subsampling 8×** → 1 frame encoder = **80 ms**.
    Attenzione causale a finestra: `sliding_window: 57` (left 56 + corrente). Mel **128**.
  - Decoder **RNNT**: prediction network hidden **640, 2 layer** (LSTM presunta — da
    confermare dallo YAML nel `.nemo`), `max_symbols_per_step: 10`. Dim joint non nel config.
  - **Language conditioning**: one-hot **128-dim** (`num_prompts: 128`, default id 101)
    concatenato alle feature + projection (intermediate 2048); dizionario di 105 locale
    nel processor. `target_lang=auto` → language detection con tag lingua emesso in output.
- **Streaming / latenza** configurabile a runtime via `att_context_size = [left, right]`:
  `[56,0]`=80 ms, `[56,1]`=160 ms, `[56,3]`=320 ms (default), `[56,6]`=560 ms, `[56,13]`=1.12 s.
- **Feature extractor** (da `processor_config.json`): 16 kHz, 128 mel, n_fft 512, hop 160,
  win 400, preemph 0.97. Normalizzazione NON dichiarata → leggere `model_config.yaml` nel `.nemo`.
- **Tokenizer**: classe `ParakeetTokenizer`, `tokenizer.json` 752 KB, vocab **13088**,
  blank id **13087**, `<pad>`=0, 39 special token di lingua.
- **Lingue**: 40 locale in 3 tier — transcription-ready (19, **it-IT incluso**),
  broad-coverage (13), adaptation-ready (8). Punctuation & capitalization native.
- **Licenza**: **OpenMDW-1.1** (molto permissiva).
- **File nel repo (~4.9 GB)**: `model.safetensors` (2.55 GB) ✨, `.nemo` (2.37 GB),
  `config.json`, `processor_config.json`, `tokenizer.json`, `generation_config.json`.
  → il porting HF-native ci permette un convertitore **senza dipendere da nemo_toolkit**.
- **Benchmark**: FLEURS @1.12s: **it-IT 4.25%**, es 4.11%, pt 5.48%, en 7.91%;
  H100: ~240 stream @80ms → ~2400 @1.12s.
- **Da chiudere in M0**: tipo prediction network e normalizzazione feature (YAML nel `.nemo`);
  timestamp sul 3.5 (documentati solo sull'EN); algoritmo tokenizer (BPE vs unigram).

### nvidia/nemotron-speech-streaming-en-0.6b (variante EN-only, gen 2026)
**https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b**
- Stessa architettura, solo inglese, tokenizer 400 KB più piccolo. **Attenzione**: left
  context **70** (`[70,0..13]`) contro il **56** del 3.5 → il runtime deve leggerlo dal config.
- WER medio 6.93% @1.12s (LS clean 2.32%); time-to-final mediano 24 ms; ~560 stream @320ms su H100.
- Licenza: NVIDIA Open Model License (più restrittiva del 3.5 — da ricontrollare).
- Demo Space: https://huggingface.co/spaces/nvidia/nemotron-speech-streaming-en-0.6b
- Collection: https://huggingface.co/collections/nvidia/nemotron-speech

Risorse Nemotron:
- Blog HF (gen 2026): https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- Blog fine-tuning (giu 2026): https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
- Fine-tuning notebook: https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb
- Deploy examples: https://github.com/modal-projects/modal-nvidia-asr · https://github.com/pipecat-ai/nemotron-january-2026
- Benchmark terzi (streaming EN su hardware constrained): https://arxiv.org/abs/2604.14493

---

## Famiglia Parakeet (ASR puro)

### nvidia/parakeet-tdt-0.6b-v3 ← multilingua di riferimento (v0.4)
**https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3** (2025-08-14)
- FastConformer 24L·1024·8h·**128 mel** / decoder **TDT**: pred 2L·640, durations **[0,1,2,3,4]**,
  max_symbols_per_step 10, **blank_id 8192**. Tokenizer SPE unificato **8192**.
- **25 lingue EU** (it incluso) con auto language detection, PnC automatiche,
  timestamp **word/segment/char**. Long audio: 24 min full attention, fino a **3 h** con
  local attention `change_attention_model("rel_pos_local_attn", [256,256])`.
- Streaming: non cache-aware; chunked via `speech_to_text_streaming_infer_rnnt.py`
  (chunk 2s, right 2s, left 10s).
- Licenza CC-BY-4.0. File: `.nemo` **+ porting HF-native completo** (safetensors, tokenizer.json).
- Open ASR Leaderboard avg WER **6.34%**; FLEURS it **3.00%**, MLS it 10.08%, CoVoST it 3.69%.
- Tech report (congiunto con Canary v2): https://arxiv.org/abs/2509.14128

### Gli altri Parakeet (tutti verificati, tabella)

| Modello | URL | Arch | Lingue | Note | Licenza | File |
|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 | 24L / TDT | EN | PnC, ts, RTFx 3380, avg WER 6.05% | CC-BY-4.0 | solo .nemo |
| parakeet-tdt-1.1b | https://huggingface.co/nvidia/parakeet-tdt-1.1b | 42L / TDT | EN | lowercase, no PnC | CC-BY-4.0 | solo .nemo |
| parakeet-rnnt-1.1b | https://huggingface.co/nvidia/parakeet-rnnt-1.1b | 42L·80mel / RNNT 2L LSTM·640 | **EN only** (⚠️ il PDF lo dava multilingual: errato) | lowercase | CC-BY-4.0 | .nemo + safetensors |
| parakeet-rnnt-0.6b | https://huggingface.co/nvidia/parakeet-rnnt-0.6b | 24L / RNNT | EN | lowercase | CC-BY-4.0 | .nemo + safetensors |
| parakeet-ctc-1.1b | https://huggingface.co/nvidia/parakeet-ctc-1.1b | 42L / CTC | EN | il più scaricato (~974k) | CC-BY-4.0 | .nemo + safetensors |
| parakeet-ctc-0.6b | https://huggingface.co/nvidia/parakeet-ctc-0.6b | 24L / CTC | EN | | CC-BY-4.0 | .nemo + safetensors |
| parakeet-tdt_ctc-110m | https://huggingface.co/nvidia/parakeet-tdt_ctc-110m | Hybrid TDT+CTC, 114M | EN | PnC; **candidato modello CI** | CC-BY-4.0 | solo .nemo |
| parakeet-tdt_ctc-1.1b | https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b | Hybrid, local attn+global token | EN | fino a **11 h** in un passaggio | CC-BY-4.0 | solo .nemo |

### Novità streaming 2025/2026 (candidati post-v1)

| Modello | URL | Cosa fa | Licenza |
|---|---|---|---|
| **parakeet-unified-en-0.6b** | https://huggingface.co/nvidia/parakeet-unified-en-0.6b | Unified-FastConformer-RNNT: **un modello per offline E streaming** (latenza 160 ms–2.08 s, chunked self-attention + Dynamic Chunked Convolutions), PnC | NVIDIA OML |
| **parakeet_realtime_eou_120m-v1** | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 | Cache-aware 17L / RNNT, 120M, latenza 80–160 ms, emette token `<EOU>` (end-of-utterance) — voice agent | NVIDIA OML |
| **multitalker-parakeet-streaming-0.6b-v1** | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 | Streaming multitalker basato su Nemotron-Speech-Streaming, speaker kernels + diarizzazione, un'istanza per speaker | NVIDIA OML |

Varianti per lingua: parakeet-tdt_ctc-0.6b-ja (JA), parakeet-ctc-0.6b-Vietnamese (VI),
parakeet-rnnt-110m-da-dk (DA).

---

## Famiglia Canary (ASR + Speech Translation, AED) — v0.8+

Tutti offline (no streaming nativo; audio lunghi via `speech_to_text_aed_chunked_infer.py`).
Task conditioning via token di prompt del decoder.

| Modello | URL | Param | Encoder/Decoder | Lingue | Traduzione | Timestamps | Licenza |
|---|---|---|---|---|---|---|---|
| canary-180m-flash | https://huggingface.co/nvidia/canary-180m-flash | 182M | FC 17L / Transformer 4L | 4 (EN/DE/FR/ES) | EN↔DE/FR/ES | sì (sperim.) | CC-BY-4.0 |
| canary-1b-flash | https://huggingface.co/nvidia/canary-1b-flash | 883M | FC 32L / Transformer 4L | 4 | EN↔DE/FR/ES | word+seg | CC-BY-4.0 |
| **canary-1b-v2** | https://huggingface.co/nvidia/canary-1b-v2 | 978M | FC 32L / Transformer 8L | **25 EU** (it incluso) | EN↔24 | sì, anche su AST | CC-BY-4.0 |
| canary-qwen-2.5b | https://huggingface.co/nvidia/canary-qwen-2.5b | 2.5B | SALM: FC + Qwen3-1.7B frozen + LoRA | solo EN | no | no | CC-BY-4.0 |
| canary-1b (2024) | https://huggingface.co/nvidia/canary-1b | 1B | FC 24L / Transformer 24L | 4 | EN↔DE/FR/ES | no | **CC-BY-NC-4.0** ⚠️ |

- Flash/1b: prompt token `<target language>`, `<task>`, `<toggle timestamps>`, `<toggle PnC>`;
  v2: `source_lang`/`target_lang` (uguali = ASR, diversi = AST); tokenizer v2: SPE unificato **16384**.
- canary-1b-flash e canary-qwen hanno anche safetensors; 180m-flash e v2 solo `.nemo`.
- Paper: Canary "Less is More" https://arxiv.org/abs/2406.19674 · Flash efficiency
  https://arxiv.org/abs/2503.05931 · Canary v2 + Parakeet v3 tech report
  https://arxiv.org/abs/2509.14128 · Granary dataset https://arxiv.org/abs/2505.13404 ·
  SALM https://arxiv.org/abs/2310.09424

---

## Implicazioni per il runtime Mynah

1. **Il config deve guidare tutto**: mel 80 vs 128, left context 56 vs 70, vocab 1024/8192/13088/16384,
   layer 17/24/32/42 — nessuna costante hardcoded (conferma della scelta anti-#define).
2. **Loader safetensors HF-native prima, `.nemo` dopo**: Nemotron 3.5, parakeet-tdt-v3,
   rnnt/ctc 0.6b/1.1b hanno già safetensors + tokenizer.json → il convertitore può partire
   da lì senza nemo_toolkit. Per i modelli solo-`.nemo` (110m, unified…) servirà l'estrattore tar.
3. **Il decoder RNNT di Nemotron (pred 640, 2L) è lo stesso design dei Parakeet RNNT/TDT**
   (stessi hidden/layers) → il decode loop si riusa; TDT aggiunge le durations [0..4].
4. **parakeet-tdt_ctc-110m** è il candidato naturale per i test CI (114M, CC-BY-4.0);
   nota: solo `.nemo`, quindi il tooling di estrazione tar serve comunque presto.
5. **Licenze**: famiglia classica CC-BY-4.0, Nemotron 3.5 OpenMDW-1.1 (ottimo),
   i nuovi streaming EN (unified, eou, multitalker) NVIDIA Open Model License (verificare
   i termini prima di redistribuire pesi convertiti); canary-1b originale NC → evitare.

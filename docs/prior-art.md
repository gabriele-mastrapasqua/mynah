# Mynah — Prior art (studiato sul codice reale, 2026-07-16)

Due progetti analizzati a fondo clonando i sorgenti. Nota strategica: **parakeet.cpp supporta
già nemotron-3.5-asr-streaming-0.6b** (offline + streaming cache-aware, WER 0 vs NeMo su
5 lingue). Non partiamo da zero concettualmente: abbiamo una mappa completa delle trappole.
I differenziatori di Mynah restano: C puro senza ggml (kernel propri stile qwen-*), scelta
del profilo di latenza a runtime (parakeet.cpp la scrive nel GGUF ma non la legge!), server
OpenAI-compatible integrato, design multi-engine fin dall'inizio.

---

## A. mudler/parakeet.cpp (C++17 su ggml, GGUF)

### Da IMITARE

1. **Nomi tensori NeMo verbatim + tutto metadata-driven**: il converter non rinomina nulla;
   ~40 chiavi KV (`arch, dims, mel params, vocab, max_symbols, att_context_*, streaming.*,
   prompt.*`) guidano il runtime. È la decisione che gli ha permesso di coprire 12 checkpoint
   quasi senza toccare C++. → Mynah: stesso principio col nostro `config.json` + safetensors.
2. **Filterbank e finestra Hann liftati dal checkpoint** (`preprocessor.featurizer.fb` /
   `.window` esportati as-is) → parità mel esatta gratis, niente formula slaney da replicare
   a mano. Il nostro convertitore deve fare lo stesso (o generarli con la formula di
   onnx-asr e validarli, atol 5e-7).
3. **Metodologia di validazione a 3 livelli** (il nostro piano di test è pronto):
   - `gen_nemo_baseline.py`: dump dei tensori intermedi NeMo (mel, subsampling, layer
     0/mid/last, encoder_out, pred_out, joint_out) con dither=0/no_grad/eval → test C
     per stadio con tolleranze 4e-5…6e-3;
   - `validate_vs_nemo.py`: WER end-to-end (exit 0/1, **77 = skip se manca il checkpoint**
     → CI sempre verde);
   - baseline streaming dedicate: token-per-token vs `cache_aware_stream_step` NeMo,
     incluso il caso reset-on-EOU con clip a due enunciati.
4. **Semantiche streaming esatte e già verificate** (da portare pari pari):
   - `cache_last_time[layer]`: `[K−1=8, d_model]`, contesto sinistro della depthwise conv;
   - `cache_last_channel[layer]`: `[cache_len=56, d_model]`, alimentata con **l'input
     dell'attention post-norm** (non l'output); `clc_len` = frame validi, colonne vuote
     mascherate;
   - `drop_extra_pre_encoded`: dopo il subsampling del chunk (che include l'overlap
     pre-encode) si scartano i frame ridondanti iniziali (solo dal chunk ≥1);
   - mid-stream si tengono `valid_out_len` frame, l'ultimo chunk tiene tutto;
   - pre-encode overlap: lo costruisce il chiamante (chunk n = overlap + frame nuovi);
   - il chunk che tocca la fine del buffer è **deferito a finalize** (`is_last=true`).
5. **Greedy RNNT ottimizzato** (`rnnt.cpp`, da portare quasi riga per riga):
   - `enc_proj` precomputata su tutti i frame in un matmul;
   - **cache dell'output del pred-net**: la LSTM si ricalcola solo su emissione non-blank
     (~U volte invece di T+U);
   - argmax sui logits raw (niente softmax);
   - `RnntDecodeState` liftato e **chunk-invariante**: decodifica incrementale ≡ intera →
     lo streaming decode è banale.
6. **Dettagli numerici che comprano la parità**: double nel mel/FFT/pos-enc; pad costante
   (NON reflect); std ddof=1 + eps 1e-5 (quando c'è normalize); batch_norm foldata host-side
   in scale/shift; ordine gate LSTM PyTorch `[i,f,g,o]` con entrambi i bias; SOS = embedding
   zero del blank; `T = 1+floor(S/hop)`; valid offline = T−1 (il center-pad aggiunge un frame).
7. **StreamingMel frame-locale** (possibile SOLO con normalize=NA — il nostro caso):
   preemphasis carried tra feed, buffer O(n_fft), tail emesso a finalize.
8. **C-API**: opaque handle, `abi_version()`, `last_error` per ctx, eventi EOU/EOB tipizzati
   + bitmask, `finalize` che non fabbrica EOU. Design da copiare in `mynah.h`.
9. **Quantizzazione**: solo i grandi linear che entrano nel matmul (FFN, attn q/k/v/out/pos,
   pre_encode.out, joint enc/pred = 90%+ dei pesi); conv/LSTM/bias/norm in F32. Evidenza:
   WER 0 fino a q4_k. → la nostra INT8 (M5) può essere altrettanto chirurgica.
10. **Trappola nota (issue #13)**: dopo `<EOU>` va resettato **solo il decoder** (LSTM a
    zero, last_token→SOS), NON la cache encoder — senza, lo stream diventa muto. Rilevante
    per i modelli EOU (parakeet_realtime_eou); per Nemotron 3.5 base non c'è EOU ma il
    pattern reset-decoder-only va tenuto.

### Da fare DIVERSAMENTE (in C puro)

1. **Niente graph-building per chiamata**: parakeet ricostruisce migliaia di nodi ggml e fa
   lookup per stringa dei pesi a ogni chunk. → Mynah: puntatori ai pesi risolti UNA volta
   al load in struct per-layer, kernel diretti.
2. **LSTM e joint step come kernel diretti** (matvec 4H×H + gates); il collo di bottiglia
   è il matmul 640→13088 del joint head, da parallelizzare sulle righe.
3. **Depthwise conv e subsampling senza im2col**: in C un loop k=9 causale è banale e
   cache-friendly; sparisce anche il workaround `pad_ext` per il padding asimmetrico.
4. **Attention senza mask materializzata**: la finestra chunked_limited si esprime nei
   limiti del loop (start/end di kj per ogni qi); il `rel_shift` si fa indicizzando
   `bd[qi,kj] = qv[qi]·p[pos_of(qi,kj)]` — zero pad/reshape/view acrobatici, zero O(T²)
   di memoria per layer.
5. **Un solo path**: niente doppio percorso scalar/batched (metà del loro codice è
   duplicazione B=1 vs B>1) — v1 streaming è B=1; il batching arriva col server (M4).
6. **Esporre la scelta del right-context a runtime** (`--latency 80|160|320|560|1120`):
   loro la ignorano, per noi è un differenziatore facile.
7. Layout mel time-major o ring buffer (il loro `append_mel_frames` ricostruisce il buffer
   a ogni feed, O(T²) cumulato).

---

## B. istupakov/onnx-asr (Python numpy + onnxruntime, no torch)

Base migliore per il **nostro oracolo** (`tools/oracle/`): leggibile, già validato vs NeMo.

### Da riusare quasi verbatim
1. **Preprocessing NeMo esatto in numpy** (`preprocessors/numpy_preprocessor.py:139-182`):
   preemph 0.97 (primo campione invariato) → pad costante n_fft/2 per lato → framing 512/160
   → Hann **simmetrica** 400 zero-paddata a 512 → rfft → |X|² → `@ fbanks` → `log(x+2^-24)`
   → (per noi: STOP, normalize NA). `features_lens = len // hop`.
2. **Mel filterbank slaney** (`preprocessors/fbanks.py:27-57`): clone di torchaudio
   (lineare <1000 Hz, log sopra, norm slaney `2/(m[2:]-m[:-2])`), calcolo in float64.
   Validato vs NeMo con atol 5e-7.
3. **Macchina a stati greedy transducer** (`asr.py:192-229`): SOS=blank, stato LSTM
   **commitato solo su token non-blank**, avanzamento t su blank o su max_tokens_per_step,
   TDT: `t += step` con guard `step==0 && blank → t+=1`.
4. **Tolleranze di test**: feature vs NeMo atol 5e-4 rtol 1e-4; filterbank 5e-7 —
   riferimento per i nostri confronti C↔oracolo.
5. `read_wav` numpy puro (PCM 8/16/24/32) e timestamps `0.01 × subsampling × frame_idx`.

### Limiti (cosa NON prendere)
- Niente streaming/cache-aware (il gap principale); niente normalize=NA parametrizzata.
- Tokenizer naïf (vocab.txt spazio-separato, niente byte fallback) — noi decodifichiamo
  dal `tokenizer.model`/`tokenizer.json` reale.
- TDT: usa l'indice argmax come durata (ok solo per durations [0..4] identità).
- decoder_joint ONNX monolitico → ricalcola il pred-net anche sui blank (noi: separati + cache).

---

## C. Decisioni per Mynah derivate dal prior art

| Decisione | Fonte |
|---|---|
| Convertitore: nomi tensori HF verbatim, tutto nel `config.json` Mynah (KV-style) | parakeet.cpp |
| Filterbank+window esportati dal checkpoint/processor nel modello convertito | parakeet.cpp |
| Oracolo: fork del preprocessing numpy di onnx-asr (senza normalize) + forward HF | onnx-asr |
| Baseline per stadio in file (mel, subsampling, layer 0/12/23, enc_out, pred, joint) | parakeet.cpp |
| Test model-dependent: skip exit 77 se il checkpoint manca | parakeet.cpp |
| Greedy RNNT: enc_proj precomputata + pred-net cache + logits raw + stato chunk-invariante | entrambi |
| Streaming: semantiche cache NeMo esatte (vedi sopra §A.4) | parakeet.cpp |
| Mel streaming frame-locale (possibile per normalize=NA) | parakeet.cpp |
| Quantizzazione (M5): solo grandi linear, resto F32 | parakeet.cpp |
| Latenza selezionabile a runtime (differenziatore) | gap di parakeet.cpp |

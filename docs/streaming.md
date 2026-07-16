# Mynah — Streaming: semantica e API

## Modello mentale

Nemotron è un encoder **cache-aware**: ogni campione audio viene processato una volta
sola. Lo stream è una sequenza di **chunk mel di taglia fissa**; ogni chunk produce
`q = lookahead+1` frame encoder (1 frame = 80 ms di audio) e il testo emesso è
**sempre definitivo** — il greedy RNNT è monotono, non ritratta mai.

| lookahead | latenza teorica | chunk mel (primo / successivi) |
|---|---|---|
| 0 | 80 ms | 1 / 8 |
| 1 | 160 ms | 9 / 16 |
| 3 (default) | 320 ms | 25 / 32 |
| 6 | 560 ms | 49 / 56 |
| 13 | 1.12 s | 105 / 112 |

Regola: primo chunk = `1 + 8·lookahead` frame mel, successivi = `8·(lookahead+1)`.
Il lookahead si sceglie **a runtime** (`--lookahead` / parametro API), senza
ricaricare il modello: qualità e latenza scalano insieme (vedi WER per preset nella
model card — a 80 ms le prime parole possono degradare).

**Garanzia di equivalenza**: lo streaming produce *esattamente* lo stesso testo del
percorso offline sullo stesso audio (`make test`, `tests/test_streaming.c` verifica
l'uguaglianza byte-a-byte). La coda dello stream è gestita con un chunk corto +
right-pad causale, identico alla matematica offline.

## Stato interno (per curiosi / debugging)

- **Mel incrementale**: finestra scorrevole O(n_fft) di segnale preemfatizzato;
  ogni frame mel è bit-uguale all'offline (possibile perché il modello non
  normalizza le feature).
- **Subsampling**: cache di 1 frame di input per ognuno dei 3 stadi conv
  (+1 zero di init al primo chunk = left pad 2 offline).
- **Attention**: cache K/V per layer `[56, d_model]`. Il chunk coincide con la
  griglia `chunked_limited` e 56 è divisibile per ogni `q` → la cache contiene
  esattamente il contesto ammesso e l'attention è piena, senza mask.
- **Conv module**: cache `[8, d_model]` (kernel 9 causale) per layer.
- **Decoder**: stato LSTM liftato e chunk-invariante — decodifica incrementale ≡
  decodifica dell'intero audio.

## API C

```c
mynah_model *m = mynah_load("models/nemotron-3.5-asr-streaming-0.6b");
mynah_stream *s = mynah_stream_open(m, "auto" /* o "it-IT", ... */, 3 /* lookahead */);

void on_text(const mynah_result *r, void *ud) {
    fputs(r->text, stdout);          /* delta di testo, già definitivo */
    /* r->lang = lingua rilevata (con "auto"), r->t1 = secondi di audio consumati */
}

while (have_audio) {
    /* float32 [-1,1], 16 kHz mono, qualsiasi taglia di feed */
    mynah_stream_feed(s, samples, n, on_text, NULL);
}
mynah_stream_finish(s, on_text, NULL);   /* processa la coda */
mynah_stream_close(s);
mynah_free(m);
```

Note:
- `feed` accetta qualsiasi numero di campioni; il chunking interno è automatico.
- Con `lang="auto"` il tag lingua arriva quando il modello lo emette
  (`mynah_stream_lang()` per leggerlo in ogni momento).
- Costo memoria per stream: ~12 MB di cache (24 layer × K/V 56×1024 + conv).
- Realtime: ~26 ms di calcolo per chunk da 80 ms su Apple Silicon (margine ~3×).

## CLI

```sh
# microfono (esempio con sox) → partial su stdout man mano
rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah stream -m <model_dir> --lang auto

# file, via ffmpeg
ffmpeg -v quiet -i audio.mp3 -f s16le -ar 16000 -ac 1 - | mynah stream -m <model_dir>
```

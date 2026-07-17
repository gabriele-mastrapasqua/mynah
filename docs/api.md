# Mynah — C API reference (v0.x, pre-freeze)

Header unico: `src/mynah.h`. Libreria: `make lib` → `libmynah.a`.
Thread-safety: un `mynah_model` può servire più thread SOLO in lettura dopo il load;
ogni `mynah_stream` va usato da un solo thread alla volta.

## Modello

```c
mynah_model *mynah_load(const char *model_dir);
void mynah_free(mynah_model *m);
```
`model_dir` è la directory prodotta dal convertitore (`mynah.json`, `model.safetensors`,
`tokens.json`, `mel_filters.safetensors`). I pesi sono mmap read-only: il load è
istantaneo, il costo si paga al primo accesso (page-in).

```c
int mynah_lang_id(const mynah_model *m, const char *lang);   /* -1 se ignota   */
int mynah_lookaheads(const mynah_model *m, int out[8]);      /* es. {3,0,6,13} */
```

## Trascrizione offline

```c
char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out);
```
- `samples`: float32 [-1,1], **16 kHz mono** (per WAV ad altri sr: `mynah_resample`).
- `lang`: tag locale (`"it-IT"`, `"en-US"`, …) o `"auto"`; `lookahead`: -1 = default
  del modello, altrimenti uno dei valori di `mynah_lookaheads` (più alto = più accurato).
- Ritorna testo UTF-8 `malloc`ato (caller `free`); `lang_out` (>= 16 byte, opzionale)
  riceve la lingua rilevata.
- Audio lunghi: oltre il limite per segmento (default 300 s, vedi
  `mynah_set_segment_limit`) l'audio viene diviso sul silenzio e trascritto a
  segmenti — memoria lineare e compatibilità coi modelli full-attention (~400 s max).
  Per il realtime resta preferibile lo streaming (O(1) in memoria).

## Timestamp per parola

```c
typedef struct { char *word; double t0, t1; } mynah_word;
char *mynah_transcribe_ts(mynah_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_word **words, int *n_words);
void mynah_words_free(mynah_word *words, int n_words);
```
Come `mynah_transcribe`, in più riempie `words` (array `malloc`ato). Risoluzione:
1 frame encoder (80 ms). Sui modelli TDT i tempi vengono dalle duration predette
(accurati); su Nemotron includono la latenza algoritmica del chunking.

## Selezione decoder e segmentazione

```c
int  mynah_set_decoder(mynah_model *m, const char *name);      /* "default" | "ctc" */
void mynah_set_segment_limit(mynah_model *m, double sec);      /* default 300 s */
```
`"ctc"` usa la head CTC dei modelli hybrid (`parakeet-tdt_ctc-*`) — più veloce,
qualità leggermente inferiore; sui modelli CTC puri è già il default. -1 se il
modello non ha la head.

## Trascrizione batch

```c
int mynah_transcribe_batch(mynah_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]);
```
N richieste processate weight-stationary (pesi letti una volta per layer, packing a
lunghezze variabili senza padding). Output identico a N chiamate `mynah_transcribe`.

## Streaming

Vedi [streaming.md](streaming.md) per la semantica completa.

```c
mynah_stream *mynah_stream_open(mynah_model *m, const char *lang, int lookahead);
int  mynah_stream_feed(mynah_stream *s, const float *samples, size_t n,
                       mynah_result_cb cb, void *userdata);
int  mynah_stream_finish(mynah_stream *s, mynah_result_cb cb, void *userdata);
const char *mynah_stream_lang(const mynah_stream *s);
void mynah_stream_close(mynah_stream *s);
```
- La callback riceve `mynah_result`: `text` = **delta** di testo (definitivo,
  `is_final = true` sempre con Nemotron greedy), `t1` = secondi di audio consumati,
  `lang` = lingua rilevata (o NULL).
- `feed` ritorna 0/-1; accetta qualsiasi taglia di input.
- Memoria per stream: ~12 MB di cache, indipendente dalla durata.

## Audio helper

```c
float *mynah_wav_load(const char *path, size_t *n_samples, int *sample_rate);
float *mynah_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out);
```

## Risultato

```c
typedef struct {
    const char *text;     /* UTF-8, valido solo durante la callback */
    double t0, t1;
    bool is_final;
    const char *lang;
} mynah_result;
typedef void (*mynah_result_cb)(const mynah_result *res, void *userdata);
```

## Esempio minimo completo

Vedi [`examples/minimal.c`](../examples/minimal.c) (compilato anche da `make test`):
load → WAV → resample se serve → `mynah_transcribe_ts` → testo + parole con tempi.

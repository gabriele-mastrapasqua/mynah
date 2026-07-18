# Mynah — C API reference (v0.x, pre-freeze)

Single header: `src/mynah.h`. Library: `make lib` → `libmynah.a`.
Thread-safety: a `mynah_model` can serve multiple threads ONLY read-only after load;
each `mynah_stream` must be used by one thread at a time.

## Model

```c
mynah_model *mynah_load(const char *model_dir);
mynah_model *mynah_load_quant(const char *model_dir, int quant);  /* MYNAH_QUANT_{F32,INT8,INT4} */
void mynah_free(mynah_model *m);
```
`model_dir` is the directory produced by the converter (`mynah.json`, `model.safetensors`,
`tokens.json`, `mel_filters.safetensors`). Weights are mmap'd read-only: loading is
instantaneous, the cost is paid on first access (page-in).
`mynah_load_quant` with `INT8`/`INT4` uses the pre-quantized checkpoint
(`model.int8.safetensors`, generated with `mynah quantize`) if present — zero-copy,
~3-4× less RAM — otherwise quantizes at load. Quality: int8 ≈ transparent,
int4 costs on multilingual (details in [quantization.md](quantization.md)).

```c
int mynah_lang_id(const mynah_model *m, const char *lang);   /* -1 if unknown   */
int mynah_lookaheads(const mynah_model *m, int out[8]);      /* e.g. {3,0,6,13} */
```

## Offline transcription

```c
char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out);
```
- `samples`: float32 [-1,1], **16 kHz mono** (for WAVs at other sample rates: `mynah_resample`).
- `lang`: locale tag (`"it-IT"`, `"en-US"`, …) or `"auto"`; `lookahead`: -1 = model
  default, otherwise one of the values from `mynah_lookaheads` (higher = more accurate).
- Returns `malloc`'d UTF-8 text (caller `free`s); `lang_out` (>= 16 bytes, optional)
  receives the detected language.
- Long audio: beyond the per-segment limit (default 300 s, see
  `mynah_set_segment_limit`) the audio is split on silence and transcribed in
  segments — linear memory and compatibility with full-attention models (~400 s max).
  For realtime, streaming remains preferable (O(1) memory).

## Word-level timestamps

```c
typedef struct { char *word; double t0, t1; } mynah_word;
char *mynah_transcribe_ts(mynah_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_word **words, int *n_words);
void mynah_words_free(mynah_word *words, int n_words);
```
Like `mynah_transcribe`, but also fills `words` (`malloc`'d array). Resolution:
1 encoder frame (80 ms). On TDT models the times come from the predicted durations
(accurate); on Nemotron they include the algorithmic latency of chunking. On AED
models (Canary) requesting `words` enables the `<|timestamp|>` tokens in the prompt:
accurate per-word times, but the model's punctuation may differ slightly from the
decode without timestamps (model behavior).

## Speech translation (AED models — Canary)

```c
int mynah_can_translate(const mynah_model *m);            /* 1 = AED engine */
int mynah_set_target_lang(mynah_model *m, const char *lang);  /* "de", "" = ASR */
```
OUTPUT language different from the source = translation (`--lang en --target-lang de`
on the CLI). `mynah_set_target_lang` mutates the model: call it BEFORE
transcriptions, not concurrently. For server/batch use the thread-safe PER-CALL
form: `lang = "src>tgt"` (e.g. `"en>de"`) in `mynah_transcribe*` — it wins over
`set_target_lang`. Supported languages: the `prompt.languages` field of mynah.json
(canary-flash: en/de/es/fr, all pairs).

## Decoder selection and segmentation

```c
int  mynah_set_decoder(mynah_model *m, const char *name);      /* "default" | "ctc" */
void mynah_set_segment_limit(mynah_model *m, double sec);      /* default 300 s */
```
`"ctc"` uses the CTC head of hybrid models (`parakeet-tdt_ctc-*`) — faster,
slightly lower quality; on pure CTC models it is already the default. -1 if the
model has no such head.

## Batch transcription

```c
int mynah_transcribe_batch(mynah_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]);
```
N requests processed weight-stationary (weights read once per layer, variable-length
packing without padding). Output identical to N `mynah_transcribe` calls.

## Streaming

See [streaming.md](streaming.md) for the full semantics.

```c
mynah_stream *mynah_stream_open(mynah_model *m, const char *lang, int lookahead);
int  mynah_stream_feed(mynah_stream *s, const float *samples, size_t n,
                       mynah_result_cb cb, void *userdata);
int  mynah_stream_finish(mynah_stream *s, mynah_result_cb cb, void *userdata);
const char *mynah_stream_lang(const mynah_stream *s);
void mynah_stream_close(mynah_stream *s);
```
- The callback receives `mynah_result`: `text` = text **delta** (final,
  `is_final = true` always with Nemotron greedy), `t1` = seconds of audio consumed,
  `lang` = detected language (or NULL).
- `feed` returns 0/-1; accepts any input size.
- Memory per stream: ~12 MB of cache, independent of duration.

## Audio helpers

```c
float *mynah_wav_load(const char *path, size_t *n_samples, int *sample_rate);
float *mynah_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out);
```

## Result

```c
typedef struct {
    const char *text;     /* UTF-8, valid only during the callback */
    double t0, t1;
    bool is_final;
    const char *lang;
} mynah_result;
typedef void (*mynah_result_cb)(const mynah_result *res, void *userdata);
```

## Minimal complete example

See [`examples/minimal.c`](../examples/minimal.c) (also compiled by `make test`):
load → WAV → resample if needed → `mynah_transcribe_ts` → text + words with times.

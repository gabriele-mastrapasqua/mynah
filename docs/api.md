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
- Complessità: O(T²) nell'attention — per audio molto lunghi (>~10 min) preferire
  lo streaming, che è O(T) in tempo e O(1) in memoria.

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

```c
#include <stdio.h>
#include <stdlib.h>
#include "mynah.h"
#include "audio.h"

int main(void) {
    mynah_model *m = mynah_load("models/nemotron-3.5-asr-streaming-0.6b");
    size_t n; int sr;
    float *pcm = mynah_wav_load("audio.wav", &n, &sr);
    if (sr != 16000) {
        size_t n2; float *r = mynah_resample(pcm, n, sr, 16000, &n2);
        free(pcm); pcm = r; n = n2;
    }
    char lang[16];
    char *text = mynah_transcribe(m, pcm, n, "auto", -1, lang);
    printf("[%s] %s\n", lang, text);
    free(text); free(pcm); mynah_free(m);
    return 0;
}
```

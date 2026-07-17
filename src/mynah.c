#include "mynah.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decoder.h"
#include "encoder.h"
#include "features.h"
#include "tokenizer.h"
#include "weights.h"
#include "../vendor/cJSON.h"

const char *mynah_version(void) { return MYNAH_VERSION; }

struct mynah_model {
    cJSON *cfg;                     /* mynah.json (vivo per il prompt dictionary) */
    mynah_safetensors *weights;
    mynah_safetensors *mel_filters;
    mynah_encoder enc;
    mynah_decoder dec;
    mynah_tokenizer tok;
    mynah_feat_cfg feat;
    int left_ctx, default_right;    /* att context dal preset di default */
    int lookaheads[8], n_lookaheads;
    int default_prompt;
    double frame_sec;               /* durata di un frame encoder (hop*sub/sr) */
};

static cJSON *load_json(const char *dir, const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mynah: manca %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0';
    fclose(f);
    cJSON *j = cJSON_Parse(buf);
    free(buf);
    return j;
}

mynah_model *mynah_load(const char *model_dir) { return mynah_load_quant(model_dir, MYNAH_QUANT_F32); }

mynah_model *mynah_load_quant(const char *model_dir, int quant) {
    mynah_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    char path[1024];

    m->cfg = load_json(model_dir, "mynah.json");
    if (!m->cfg) goto fail;

    /* pre-quantizzato su disco? (mynah quantize) -> load istantaneo, niente f32 */
    const char *wfile = cJSON_GetObjectItem(m->cfg, "weights")->valuestring;
    m->weights = NULL;
    if (quant != MYNAH_QUANT_F32) {
        snprintf(path, sizeof(path), "%s/model.%s.safetensors", model_dir,
                 quant == MYNAH_QUANT_INT8 ? "int8" : "int4");
        m->weights = mynah_st_open_quiet(path);
        if (m->weights)
            fprintf(stderr, "mynah: checkpoint pre-quantizzato %s\n", path);
    }
    if (!m->weights) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, wfile);
        m->weights = mynah_st_open(path);
    }
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", model_dir);
    m->mel_filters = mynah_st_open(path);
    if (!m->weights || !m->mel_filters) goto fail;

    if (mynah_encoder_init(&m->enc, m->weights, quant) != 0) {
        fprintf(stderr, "mynah: encoder init fallita\n");
        goto fail;
    }

    /* causalità dal config (l'init la inferisce dal naming dei pesi; il config vince) */
    const cJSON *jenc = cJSON_GetObjectItem(m->cfg, "encoder");
    const cJSON *jsub = jenc ? cJSON_GetObjectItem(jenc, "subsampling") : NULL;
    if (jsub && cJSON_IsString(jsub)) {
        m->enc.causal = strstr(jsub->valuestring, "causal") != NULL;
        m->enc.ss.causal = m->enc.causal;
    }

    const cJSON *jdec = cJSON_GetObjectItem(m->cfg, "decoder");
    int durations[MYNAH_MAX_DURATIONS];
    int n_durations = 0;
    const cJSON *jdur = cJSON_GetObjectItem(jdec, "durations");
    for (cJSON *d = jdur ? jdur->child : NULL; d && n_durations < MYNAH_MAX_DURATIONS; d = d->next)
        durations[n_durations++] = d->valueint;
    if (mynah_decoder_init(&m->dec, m->weights,
                           cJSON_GetObjectItem(jdec, "blank_id")->valueint,
                           cJSON_GetObjectItem(jdec, "max_symbols_per_step")->valueint,
                           quant, durations, n_durations) != 0) {
        fprintf(stderr, "mynah: decoder init fallita\n");
        goto fail;
    }

    snprintf(path, sizeof(path), "%s/tokens.json", model_dir);
    if (mynah_tokenizer_load(&m->tok, path) != 0) goto fail;

    /* feature config */
    const cJSON *jf = cJSON_GetObjectItem(m->cfg, "features");
    const mynah_tensor *fb = mynah_st_get(m->mel_filters, "mel_fb");
    const mynah_tensor *win = mynah_st_get(m->mel_filters, "window");
    if (!fb || !win) goto fail;
    const cJSON *jnorm = cJSON_GetObjectItem(jf, "normalize");
    m->feat = (mynah_feat_cfg){
        .sample_rate = cJSON_GetObjectItem(jf, "sample_rate")->valueint,
        .n_mels = cJSON_GetObjectItem(jf, "n_mels")->valueint,
        .n_fft = cJSON_GetObjectItem(jf, "n_fft")->valueint,
        .win_length = cJSON_GetObjectItem(jf, "win_length")->valueint,
        .hop_length = cJSON_GetObjectItem(jf, "hop_length")->valueint,
        .preemphasis = cJSON_GetObjectItem(jf, "preemphasis")->valuedouble,
        .log_zero_guard = cJSON_GetObjectItem(jf, "log_zero_guard")->valuedouble,
        .normalize_per_feature = jnorm && cJSON_IsString(jnorm) &&
                                 strcmp(jnorm->valuestring, "per_feature") == 0,
        .mel_fb = (const float *)fb->data,
        .window = (const float *)win->data,
    };

    const cJSON *jsf = jenc ? cJSON_GetObjectItem(jenc, "subsampling_factor") : NULL;
    m->frame_sec = (double)m->feat.hop_length * (jsf ? jsf->valueint : 8)
                   / (double)m->feat.sample_rate;

    /* streaming presets [[left, right], ...] — sezione assente per i modelli
     * offline (Parakeet): attention full [-1,-1], niente stream API */
    m->left_ctx = -1;
    m->default_right = -1;
    const cJSON *js = cJSON_GetObjectItem(m->cfg, "streaming");
    if (js) {
        const cJSON *presets = cJSON_GetObjectItem(js, "att_context_presets");
        const int def = cJSON_GetObjectItem(js, "default_preset_index")->valueint;
        int i = 0;
        for (cJSON *p = presets->child; p && i < 8; p = p->next, i++) {
            m->lookaheads[i] = cJSON_GetArrayItem(p, 1)->valueint;
            if (i == def) {
                m->left_ctx = cJSON_GetArrayItem(p, 0)->valueint;
                m->default_right = m->lookaheads[i];
            }
        }
        m->n_lookaheads = i;
    }

    /* prompt (Nemotron): assente nei modelli con LID implicita (Parakeet) */
    const cJSON *jprompt = cJSON_GetObjectItem(m->cfg, "prompt");
    m->default_prompt = jprompt ? cJSON_GetObjectItem(jprompt, "default_id")->valueint : -1;
    return m;

fail:
    mynah_free(m);
    return NULL;
}

void mynah_free(mynah_model *m) {
    if (!m) return;
    mynah_qmat_free(&m->dec.head);
    mynah_encoder_free(&m->enc);
    mynah_tokenizer_free(&m->tok);
    mynah_st_close(m->weights);
    mynah_st_close(m->mel_filters);
    cJSON_Delete(m->cfg);
    free(m);
}

int mynah_lang_id(const mynah_model *m, const char *lang) {
    const cJSON *jprompt = cJSON_GetObjectItem(m->cfg, "prompt");
    if (!jprompt) return -1;
    const cJSON *dict = cJSON_GetObjectItem(jprompt, "dictionary");
    const cJSON *e = dict ? cJSON_GetObjectItem(dict, lang) : NULL;
    return e ? e->valueint : -1;
}

/* prompt id per la richiesta: -1 = modello senza prompt (valido, es. Parakeet),
 * -2 = lingua sconosciuta per un modello CON prompt (errore). */
static int resolve_prompt(const mynah_model *m, const char *lang) {
    if (m->default_prompt < 0) return -1;
    if (!lang) return m->default_prompt;
    const int id = mynah_lang_id(m, lang);
    return id < 0 ? -2 : id;
}

int mynah_lookaheads(const mynah_model *m, int out[8]) {
    memcpy(out, m->lookaheads, sizeof(m->lookaheads));
    return m->n_lookaheads;
}

int mynah_transcribe_batch(mynah_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]) {
    const int right = lookahead >= 0 ? lookahead : m->default_right;
    int rc = -1;

    float **feats = calloc((size_t)batch, sizeof(float *));
    float **encs = calloc((size_t)batch, sizeof(float *));
    int *valids = calloc((size_t)batch, sizeof(int));
    int *prompts = calloc((size_t)batch, sizeof(int));
    int *t_encs = calloc((size_t)batch, sizeof(int));
    if (!feats || !encs || !valids || !prompts || !t_encs) goto done;

    for (int b = 0; b < batch; b++) {
        texts[b] = NULL;
        prompts[b] = resolve_prompt(m, langs ? langs[b] : NULL);
        if (prompts[b] == -2) goto done;
        int T_mel;
        feats[b] = mynah_log_mel(&m->feat, samples[b], n_samples[b], &T_mel, &valids[b]);
        if (!feats[b]) goto done;
    }

    if (mynah_encoder_forward_batch(&m->enc, (const float *const *)feats, valids, batch,
                                    m->feat.n_mels, prompts, m->left_ctx, right,
                                    encs, t_encs) != 0)
        goto done;

    for (int b = 0; b < batch; b++) {
        if (!encs[b]) continue;
        mynah_dec_state *s = malloc(sizeof(*s));
        if (!s) continue;
        mynah_dec_state_reset(&m->dec, s);
        const int cap = t_encs[b] * m->dec.max_symbols;
        int *tokens = malloc((size_t)cap * sizeof(int));
        const int n_tok = tokens ? mynah_greedy_decode(&m->dec, s, encs[b], t_encs[b],
                                                       tokens, NULL, cap) : 0;
        texts[b] = mynah_detokenize(&m->tok, tokens, n_tok,
                                    langs_out ? langs_out[b] : NULL);
        free(tokens);
        free(s);
    }
    rc = 0;

done:
    for (int b = 0; b < batch; b++) {
        if (feats) free(feats[b]);
        if (encs) free(encs[b]);
    }
    free(feats); free(encs); free(valids); free(prompts); free(t_encs);
    return rc;
}

/* ----------------------------------------------------------------- streaming */

struct mynah_stream {
    mynah_model *m;
    mynah_mel_stream mel;
    mynah_enc_stream es;
    mynah_dec_state dec;
    int prompt;
    float *mel_buf;             /* buffer del chunk mel corrente */
    int mel_have;
    float *enc_buf;             /* [q, d_out] output encoder per chunk */
    int *tokens;
    int n_tokens, cap_tokens;
    size_t chars_emitted;       /* byte di testo già consegnati alla callback */
    char lang[16];
    size_t samples_fed;
};

mynah_stream *mynah_stream_open(mynah_model *m, const char *lang, int lookahead) {
    if (m->n_lookaheads == 0) {
        fprintf(stderr, "mynah: il modello è offline-only (niente streaming cache-aware)\n");
        return NULL;
    }
    const int prompt = resolve_prompt(m, lang);
    if (prompt == -2) { fprintf(stderr, "mynah: lingua '%s' non supportata\n", lang); return NULL; }
    const int right = lookahead >= 0 ? lookahead : m->default_right;

    mynah_stream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m;
    s->prompt = prompt;
    if (mynah_mel_stream_init(&s->mel, &m->feat) != 0 ||
        mynah_enc_stream_init(&s->es, &m->enc, m->left_ctx, right, m->feat.n_mels) != 0) {
        mynah_stream_close(s);
        return NULL;
    }
    mynah_dec_state_reset(&m->dec, &s->dec);
    const int max_chunk = 8 * (right + 1) + 1;
    s->mel_buf = malloc((size_t)max_chunk * (size_t)m->feat.n_mels * sizeof(float));
    s->enc_buf = malloc((size_t)s->es.q * (size_t)m->enc.d_out * sizeof(float));
    s->cap_tokens = 4096;
    s->tokens = malloc((size_t)s->cap_tokens * sizeof(int));
    if (!s->mel_buf || !s->enc_buf || !s->tokens) { mynah_stream_close(s); return NULL; }
    return s;
}

void mynah_stream_close(mynah_stream *s) {
    if (!s) return;
    mynah_mel_stream_free(&s->mel);
    mynah_enc_stream_free(&s->es);
    free(s->mel_buf); free(s->enc_buf); free(s->tokens);
    free(s);
}

const char *mynah_stream_lang(const mynah_stream *s) { return s->lang; }

/* Encoda il chunk mel corrente, decodifica, emette il delta di testo. */
static int stream_flush_chunk(mynah_stream *s, int n_mel, int is_last,
                              mynah_result_cb cb, void *ud) {
    mynah_model *m = s->m;
    const int q = mynah_enc_stream_step(&s->es, s->mel_buf, n_mel, m->feat.n_mels,
                                        s->prompt, is_last, s->enc_buf);
    if (q < 0) return -1;

    if (s->n_tokens + q * m->dec.max_symbols > s->cap_tokens) {
        s->cap_tokens = (s->cap_tokens + q * m->dec.max_symbols) * 2;
        int *nb = realloc(s->tokens, (size_t)s->cap_tokens * sizeof(int));
        if (!nb) return -1;
        s->tokens = nb;
    }
    s->n_tokens += mynah_greedy_decode(&m->dec, &s->dec, s->enc_buf, q,
                                       s->tokens + s->n_tokens, NULL,
                                       s->cap_tokens - s->n_tokens);
    s->mel_have = 0;

    if (cb) {
        char lang_tmp[16] = "";
        char *text = mynah_detokenize(&m->tok, s->tokens, s->n_tokens, lang_tmp);
        if (!text) return -1;
        if (lang_tmp[0]) memcpy(s->lang, lang_tmp, sizeof(s->lang));
        const size_t total = strlen(text);
        if (total > s->chars_emitted) {
            const double t1 = (double)s->samples_fed / (double)m->feat.sample_rate;
            mynah_result res = {
                .text = text + s->chars_emitted,
                .t0 = 0.0, .t1 = t1,
                .is_final = true,
                .lang = s->lang[0] ? s->lang : NULL,
            };
            cb(&res, ud);
            s->chars_emitted = total;
        }
        free(text);
    }
    return 0;
}

int mynah_stream_feed(mynah_stream *s, const float *samples, size_t n,
                      mynah_result_cb cb, void *ud) {
    mynah_model *m = s->m;
    s->samples_fed += n;
    const float *src = samples;
    size_t left = n;
    int first_pass = 1;

    for (;;) {
        const int need = mynah_enc_stream_need(&s->es);
        const int got = mynah_mel_stream_feed(&s->mel, first_pass ? src : NULL,
                                              first_pass ? left : 0,
                                              s->mel_buf + (size_t)s->mel_have * (size_t)m->feat.n_mels,
                                              need - s->mel_have);
        first_pass = 0;
        s->mel_have += got;
        if (s->mel_have < need) break;          /* servono altri campioni */
        if (stream_flush_chunk(s, need, 0, cb, ud) != 0) return -1;
    }
    return 0;
}

int mynah_stream_finish(mynah_stream *s, mynah_result_cb cb, void *ud) {
    mynah_model *m = s->m;
    for (;;) {
        const int need = mynah_enc_stream_need(&s->es);
        const int got = mynah_mel_stream_finish(&s->mel,
                                                s->mel_buf + (size_t)s->mel_have * (size_t)m->feat.n_mels,
                                                need - s->mel_have);
        s->mel_have += got;
        if (s->mel_have == 0) break;
        if (s->mel_have < need) {
            /* coda: chunk corto con right-pad causale (is_last) — identico all'offline */
            if (stream_flush_chunk(s, s->mel_have, 1, cb, ud) != 0) return -1;
            break;
        }
        /* chunk pieno: is_last solo se il mel stream non ha più nulla dopo */
        if (stream_flush_chunk(s, need, 0, cb, ud) != 0) return -1;
    }
    return 0;
}

char *mynah_transcribe_ts(mynah_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_word **words, int *n_words) {
    if (words) { *words = NULL; *n_words = 0; }
    const int prompt = resolve_prompt(m, lang);
    if (prompt == -2) { fprintf(stderr, "mynah: lingua '%s' non supportata\n", lang); return NULL; }
    const int right = lookahead >= 0 ? lookahead : m->default_right;

    int T_mel, valid;
    float *feats = mynah_log_mel(&m->feat, samples, n_samples, &T_mel, &valid);
    if (!feats) return NULL;

    int T_enc;
    float *enc = mynah_encoder_forward(&m->enc, feats, valid, m->feat.n_mels, prompt,
                                       m->left_ctx, right, &T_enc);
    free(feats);
    if (!enc) return NULL;

    mynah_dec_state *s = malloc(sizeof(*s));
    if (!s) { free(enc); return NULL; }
    mynah_dec_state_reset(&m->dec, s);
    const int cap = T_enc * m->dec.max_symbols;
    int *tokens = malloc((size_t)cap * sizeof(int));
    int *frames = words && tokens ? malloc((size_t)cap * sizeof(int)) : NULL;
    const int n_tok = tokens ? mynah_greedy_decode(&m->dec, s, enc, T_enc, tokens,
                                                   frames, cap) : 0;
    free(enc);
    free(s);

    char *text = mynah_detokenize(&m->tok, tokens, n_tok, lang_out);
    if (text && words && frames)
        mynah_detokenize_words(&m->tok, tokens, frames, n_tok, m->frame_sec,
                               words, n_words);
    free(tokens);
    free(frames);
    return text;
}

char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out) {
    return mynah_transcribe_ts(m, samples, n_samples, lang, lookahead, lang_out,
                               NULL, NULL);
}

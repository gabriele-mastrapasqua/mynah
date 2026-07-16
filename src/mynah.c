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

mynah_model *mynah_load(const char *model_dir) {
    mynah_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    char path[1024];

    m->cfg = load_json(model_dir, "mynah.json");
    if (!m->cfg) goto fail;

    snprintf(path, sizeof(path), "%s/%s", model_dir,
             cJSON_GetObjectItem(m->cfg, "weights")->valuestring);
    m->weights = mynah_st_open(path);
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", model_dir);
    m->mel_filters = mynah_st_open(path);
    if (!m->weights || !m->mel_filters) goto fail;

    if (mynah_encoder_init(&m->enc, m->weights) != 0) {
        fprintf(stderr, "mynah: encoder init fallita\n");
        goto fail;
    }

    const cJSON *jdec = cJSON_GetObjectItem(m->cfg, "decoder");
    if (mynah_decoder_init(&m->dec, m->weights,
                           cJSON_GetObjectItem(jdec, "blank_id")->valueint,
                           cJSON_GetObjectItem(jdec, "max_symbols_per_step")->valueint) != 0) {
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
    m->feat = (mynah_feat_cfg){
        .sample_rate = cJSON_GetObjectItem(jf, "sample_rate")->valueint,
        .n_mels = cJSON_GetObjectItem(jf, "n_mels")->valueint,
        .n_fft = cJSON_GetObjectItem(jf, "n_fft")->valueint,
        .win_length = cJSON_GetObjectItem(jf, "win_length")->valueint,
        .hop_length = cJSON_GetObjectItem(jf, "hop_length")->valueint,
        .preemphasis = cJSON_GetObjectItem(jf, "preemphasis")->valuedouble,
        .log_zero_guard = cJSON_GetObjectItem(jf, "log_zero_guard")->valuedouble,
        .mel_fb = (const float *)fb->data,
        .window = (const float *)win->data,
    };

    /* streaming presets: [[left, right], ...] */
    const cJSON *js = cJSON_GetObjectItem(m->cfg, "streaming");
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

    m->default_prompt = cJSON_GetObjectItem(cJSON_GetObjectItem(m->cfg, "prompt"),
                                            "default_id")->valueint;
    return m;

fail:
    mynah_free(m);
    return NULL;
}

void mynah_free(mynah_model *m) {
    if (!m) return;
    mynah_encoder_free(&m->enc);
    mynah_tokenizer_free(&m->tok);
    mynah_st_close(m->weights);
    mynah_st_close(m->mel_filters);
    cJSON_Delete(m->cfg);
    free(m);
}

int mynah_lang_id(const mynah_model *m, const char *lang) {
    const cJSON *dict = cJSON_GetObjectItem(cJSON_GetObjectItem(m->cfg, "prompt"), "dictionary");
    const cJSON *e = cJSON_GetObjectItem(dict, lang);
    return e ? e->valueint : -1;
}

int mynah_lookaheads(const mynah_model *m, int out[8]) {
    memcpy(out, m->lookaheads, sizeof(m->lookaheads));
    return m->n_lookaheads;
}

char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out) {
    const int prompt = lang ? mynah_lang_id(m, lang) : m->default_prompt;
    if (prompt < 0) { fprintf(stderr, "mynah: lingua '%s' non supportata\n", lang); return NULL; }
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
    const int n_tok = tokens ? mynah_greedy_decode(&m->dec, s, enc, T_enc, tokens, cap) : 0;
    free(enc);
    free(s);

    char *text = mynah_detokenize(&m->tok, tokens, n_tok, lang_out);
    free(tokens);
    return text;
}

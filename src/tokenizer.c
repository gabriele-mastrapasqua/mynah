#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../vendor/cJSON.h"

int mynah_tokenizer_load(mynah_tokenizer *tk, const char *path) {
    memset(tk, 0, sizeof(*tk));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: impossibile aprire %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return -1; }
    buf[len] = '\0';
    fclose(f);

    cJSON *arr = cJSON_Parse(buf);
    free(buf);
    if (!arr || !cJSON_IsArray(arr)) { cJSON_Delete(arr); return -1; }

    tk->n_pieces = cJSON_GetArraySize(arr);
    tk->pieces = calloc((size_t)tk->n_pieces, sizeof(char *));
    int i = 0;
    for (cJSON *it = arr->child; it; it = it->next, i++)
        tk->pieces[i] = strdup(cJSON_IsString(it) ? it->valuestring : "");
    cJSON_Delete(arr);
    return 0;
}

void mynah_tokenizer_free(mynah_tokenizer *tk) {
    for (int i = 0; i < tk->n_pieces; i++) free(tk->pieces[i]);
    free(tk->pieces);
    tk->pieces = NULL;
}

char *mynah_detokenize(const mynah_tokenizer *tk, const int *tokens, int n,
                       char *lang_out) {
    size_t cap = 256, len = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    if (lang_out) lang_out[0] = '\0';

    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= tk->n_pieces) continue;
        const char *p = tk->pieces[tokens[i]];
        const size_t pl = strlen(p);

        /* token speciale <...>: tag lingua (contiene '-') o marcatore -> strip */
        if (pl >= 2 && p[0] == '<' && p[pl - 1] == '>') {
            if (lang_out && memchr(p, '-', pl) && pl - 2 < 16) {
                memcpy(lang_out, p + 1, pl - 2);
                lang_out[pl - 2] = '\0';
            }
            continue;
        }
        if (len + pl * 3 + 1 > cap) {
            cap = (cap + pl * 3 + 1) * 2;
            char *nb = realloc(out, cap);
            if (!nb) { free(out); return NULL; }
            out = nb;
        }
        /* copia sostituendo ▁ (U+2581, e2 96 81) con spazio */
        for (size_t j = 0; j < pl; ) {
            if (j + 2 < pl && (unsigned char)p[j] == 0xE2 && (unsigned char)p[j + 1] == 0x96 &&
                (unsigned char)p[j + 2] == 0x81) {
                out[len++] = ' ';
                j += 3;
            } else {
                out[len++] = p[j++];
            }
        }
    }
    out[len] = '\0';

    /* strip degli spazi iniziali (come l'oracolo: lstrip) */
    size_t skip = 0;
    while (out[skip] == ' ') skip++;
    if (skip) memmove(out, out + skip, len - skip + 1);
    return out;
}

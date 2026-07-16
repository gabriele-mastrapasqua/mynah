#include "qmat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* ------------------------------------------------------------ quantizzatori */
void mynah_quantize_int8(const float *w, int n, int k, int8_t *out_q, float *out_scales) {
    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * (size_t)k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        out_scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t *qrow = out_q + (size_t)i * (size_t)k;
        for (int j = 0; j < k; j++) {
            const float v = row[j] * inv;
            qrow[j] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }
}

void mynah_quantize_int4(const float *w, int n, int k, uint8_t *out_q, float *out_scales) {
    const int G = MYNAH_Q4_GROUP;
    const int groups = k / G;
    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * (size_t)k;
        uint8_t *qrow = out_q + (size_t)i * (size_t)(k / 2);
        float *srow = out_scales + (size_t)i * (size_t)groups;
        for (int g = 0; g < groups; g++) {
            const float *grp = row + g * G;
            float amax = 0.0f;
            for (int j = 0; j < G; j++) {
                const float a = fabsf(grp[j]);
                if (a > amax) amax = a;
            }
            const float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            srow[g] = scale;
            const float inv = 1.0f / scale;
            for (int j = 0; j < G; j += 2) {
                float v0 = grp[j] * inv, v1 = grp[j + 1] * inv;
                int q0 = (int)(v0 >= 0.0f ? v0 + 0.5f : v0 - 0.5f);
                int q1 = (int)(v1 >= 0.0f ? v1 + 0.5f : v1 - 0.5f);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8;
                if (q1 > 7) q1 = 7;
                qrow[(g * G + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }
}

/* --------------------------------------------------------------------- init */
static void release_f32_pages(const float *w, size_t bytes) {
    /* pagine mmap del f32 quantizzato: clean e rileggibili — su Linux il
     * DONTNEED le rilascia subito, su macOS restano riappropriabili sotto
     * pressione (l'accounting RSS non cala: limite noto, vedi TODO M5) */
    const long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)w + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = ((uintptr_t)w + bytes) & ~((uintptr_t)pg - 1);
    if (hi > lo) madvise((void *)lo, hi - lo, MADV_DONTNEED);
}

int mynah_qmat_init(mynah_qmat *m, const float *w, int n, int k, int qtype) {
    memset(m, 0, sizeof(*m));
    m->f32 = w;
    m->n = n;
    m->k = k;
    m->qtype = MYNAH_Q_F32;
    if (qtype == MYNAH_Q_F32 || !w) return 0;

    if (qtype == MYNAH_Q_INT8) {
        int8_t *q = malloc((size_t)n * (size_t)k);
        float *s = malloc((size_t)n * sizeof(float));
        if (!q || !s) { free(q); free(s); return -1; }
        mynah_quantize_int8(w, n, k, q, s);
        m->q8 = q;
        m->scales = s;
        m->owned_q = q;
        m->owned_s = s;
        m->qtype = MYNAH_Q_INT8;
    } else {
        if (k % MYNAH_Q4_GROUP != 0) return 0;   /* resta f32 */
        uint8_t *q = malloc((size_t)n * (size_t)k / 2);
        float *s = malloc((size_t)n * (size_t)(k / MYNAH_Q4_GROUP) * sizeof(float));
        if (!q || !s) { free(q); free(s); return -1; }
        mynah_quantize_int4(w, n, k, q, s);
        m->q4 = q;
        m->scales = s;
        m->owned_q = q;
        m->owned_s = s;
        m->qtype = MYNAH_Q_INT4;
    }
    release_f32_pages(w, (size_t)n * (size_t)k * 4u);
    return 0;
}

int mynah_qmat_init_st(mynah_qmat *m, const mynah_safetensors *st, const char *name,
                       int qtype) {
    memset(m, 0, sizeof(*m));
    char qname[192];

    if (qtype != MYNAH_Q_F32) {
        snprintf(qname, sizeof(qname), "%s.%s", name, qtype == MYNAH_Q_INT8 ? "q8" : "q4");
        const mynah_tensor *tq = mynah_st_get(st, qname);
        snprintf(qname, sizeof(qname), "%s.scales", name);
        const mynah_tensor *ts = mynah_st_get(st, qname);
        if (tq && ts) {                       /* pre-quantizzato: zero-copy dal mmap */
            m->n = (int)tq->shape[0];
            m->scales = (const float *)ts->data;
            if (qtype == MYNAH_Q_INT8) {
                m->k = (int)tq->shape[1];
                m->q8 = (const int8_t *)tq->data;
                m->qtype = MYNAH_Q_INT8;
            } else {
                m->k = (int)tq->shape[1] * 2;
                m->q4 = (const uint8_t *)tq->data;
                m->qtype = MYNAH_Q_INT4;
            }
            return 0;
        }
    }
    const mynah_tensor *tf = mynah_st_get(st, name);
    if (!tf) return -1;
    return mynah_qmat_init(m, (const float *)tf->data, (int)tf->shape[0],
                           (int)(tf->n_elems / (size_t)tf->shape[0]), qtype);
}

void mynah_qmat_free(mynah_qmat *m) {
    free(m->owned_q);
    free(m->owned_s);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ dequant */
static void dequant_row(const mynah_qmat *m, int i, float *dst) {
    if (m->qtype == MYNAH_Q_INT8) {
        const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
        const float s = m->scales[i];
        for (int j = 0; j < m->k; j++) dst[j] = (float)qrow[j] * s;
    } else {
        const uint8_t *qrow = m->q4 + (size_t)i * (size_t)(m->k / 2);
        const float *srow = m->scales + (size_t)i * (size_t)(m->k / MYNAH_Q4_GROUP);
        for (int g = 0; g < m->k / MYNAH_Q4_GROUP; g++) {
            const float s = srow[g];
            for (int j = 0; j < MYNAH_Q4_GROUP; j += 2) {
                const uint8_t b = qrow[(g * MYNAH_Q4_GROUP + j) / 2];
                dst[g * MYNAH_Q4_GROUP + j] = (float)((int)(b & 0x0F) - 8) * s;
                dst[g * MYNAH_Q4_GROUP + j + 1] = (float)((int)(b >> 4) - 8) * s;
            }
        }
    }
}

/* soglia righe: sotto -> dot diretto (bandwidth-bound), sopra -> dequant+GEMM */
#define QMAT_SMALL_T 16

/* ------------------------------------------------------------- kernel NEON */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

/* dot f32 x int8 (riga intera, scala unica) */
static float dot_q8_neon(const float *x, const int8_t *q, int k) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    for (int j = 0; j < k; j += 16) {
        const int8x16_t qb = vld1q_s8(q + j);
        const int16x8_t lo = vmovl_s8(vget_low_s8(qb));
        const int16x8_t hi = vmovl_s8(vget_high_s8(qb));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + j),      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + j + 4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        acc2 = vfmaq_f32(acc2, vld1q_f32(x + j + 8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        acc3 = vfmaq_f32(acc3, vld1q_f32(x + j + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

/* dot f32 x int4 (gruppi da 32 nibble packed: byte j = elementi 2j | 2j+1<<4).
 * vld2q_f32 deinterleava x in lane pari/dispari, allineate ai nibble lo/hi. */
static float dot_q4_neon(const float *x, const uint8_t *q, const float *scales,
                         int k) {
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t maskv = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (int g = 0; g < k / MYNAH_Q4_GROUP; g++) {
        const uint8x16_t b = vld1q_u8(q + g * 16);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, maskv)), off);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);

        const float32x4x2_t x0 = vld2q_f32(x + g * 32);       /* pari/dispari 0..7  */
        const float32x4x2_t x1 = vld2q_f32(x + g * 32 + 8);   /* 8..15  */
        const float32x4x2_t x2 = vld2q_f32(x + g * 32 + 16);  /* 16..23 */
        const float32x4x2_t x3 = vld2q_f32(x + g * 32 + 24);  /* 24..31 */

        const int16x8_t lo16a = vmovl_s8(vget_low_s8(lo));
        const int16x8_t lo16b = vmovl_s8(vget_high_s8(lo));
        const int16x8_t hi16a = vmovl_s8(vget_low_s8(hi));
        const int16x8_t hi16b = vmovl_s8(vget_high_s8(hi));

        float32x4_t ga = vmulq_f32(x0.val[0], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16a))));
        ga = vfmaq_f32(ga, x0.val[1], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16a))));
        ga = vfmaq_f32(ga, x1.val[0], vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16a))));
        ga = vfmaq_f32(ga, x1.val[1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16a))));
        ga = vfmaq_f32(ga, x2.val[0], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16b))));
        ga = vfmaq_f32(ga, x2.val[1], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16b))));
        ga = vfmaq_f32(ga, x3.val[0], vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16b))));
        ga = vfmaq_f32(ga, x3.val[1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16b))));

        acc += vaddvq_f32(ga) * scales[g];
    }
    return acc;
}
#define MYNAH_HAVE_NEON 1
#endif

void mynah_qmat_mul(const mynah_qmat *m, const float *x, float *out, int T) {
    if (m->qtype == MYNAH_Q_F32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                    1.0f, x, m->k, m->f32, m->k, 0.0f, out, m->n);
        return;
    }
    if (T <= QMAT_SMALL_T) {
        if (m->qtype == MYNAH_Q_INT8) {
            for (int t = 0; t < T; t++) {
                const float *xr = x + (size_t)t * (size_t)m->k;
                float *o = out + (size_t)t * (size_t)m->n;
                for (int i = 0; i < m->n; i++) {
                    const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
#ifdef MYNAH_HAVE_NEON
                    o[i] = dot_q8_neon(xr, qrow, m->k) * m->scales[i];
#else
                    float acc = 0.0f;
                    for (int j = 0; j < m->k; j++) acc += xr[j] * (float)qrow[j];
                    o[i] = acc * m->scales[i];
#endif
                }
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float *xr = x + (size_t)t * (size_t)m->k;
                float *o = out + (size_t)t * (size_t)m->n;
                const int groups = m->k / MYNAH_Q4_GROUP;
                for (int i = 0; i < m->n; i++) {
                    const uint8_t *qrow = m->q4 + (size_t)i * (size_t)(m->k / 2);
                    const float *srow = m->scales + (size_t)i * (size_t)groups;
#ifdef MYNAH_HAVE_NEON
                    o[i] = dot_q4_neon(xr, qrow, srow, m->k);
#else
                    float acc = 0.0f;
                    for (int g = 0; g < groups; g++) {
                        float ga = 0.0f;
                        const float *xg = xr + g * MYNAH_Q4_GROUP;
                        for (int j = 0; j < MYNAH_Q4_GROUP; j += 2) {
                            const uint8_t b = qrow[(g * MYNAH_Q4_GROUP + j) / 2];
                            ga += xg[j] * (float)((int)(b & 0x0F) - 8);
                            ga += xg[j + 1] * (float)((int)(b >> 4) - 8);
                        }
                        acc += ga * srow[g];
                    }
                    o[i] = acc;
#endif
                }
            }
        }
        return;
    }
    /* dequant per-chiamata + GEMM: l'overhead (ms) si ammortizza sul T grande */
    float *wd = malloc((size_t)m->n * (size_t)m->k * sizeof(float));
    if (!wd) return;
    for (int i = 0; i < m->n; i++) dequant_row(m, i, wd + (size_t)i * (size_t)m->k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                1.0f, x, m->k, wd, m->k, 0.0f, out, m->n);
    free(wd);
}

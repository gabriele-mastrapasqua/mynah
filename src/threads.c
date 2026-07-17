#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#define PF_MAX_THREADS 64

int mynah_num_threads(void) {
    static int nth = 0;
    if (nth == 0) {
        const char *env = getenv("MYNAH_THREADS");
        long n = env ? atol(env) : sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
        if (n > PF_MAX_THREADS) n = PF_MAX_THREADS;
        nth = (int)n;
    }
    return nth;
}

typedef struct {
    void (*fn)(void *, int);
    void *ctx;
    atomic_int next;
    int n;
} pf_state;

static void *pf_worker(void *arg) {
    pf_state *st = arg;
    for (;;) {
        const int i = atomic_fetch_add_explicit(&st->next, 1, memory_order_relaxed);
        if (i >= st->n) break;
        st->fn(st->ctx, i);
    }
    return NULL;
}

void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    int nth = mynah_num_threads();
    if (nth > n) nth = n;
    if (nth <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pf_state st = {.fn = fn, .ctx = ctx, .n = n};
    atomic_init(&st.next, 0);
    pthread_t tids[PF_MAX_THREADS];
    int spawned = 0;
    for (int k = 0; k < nth - 1; k++)
        if (pthread_create(&tids[spawned], NULL, pf_worker, &st) == 0) spawned++;
    pf_worker(&st);          /* il chiamante lavora anche lui */
    for (int k = 0; k < spawned; k++) pthread_join(tids[k], NULL);
}

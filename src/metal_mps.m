/* Backend Metal v2 — MPS in fp16 con fusioni per ridurre i sync GPU.
 *
 * v1 (superata): una GEMM fp32 = un commit+wait -> ~10 sync/layer, perdeva vs AMX.
 * v2: pesi residenti convertiti fp16 UNA volta (cache per-pointer, pattern
 * qwen-tts), attivazioni fp16, e due fusioni chiave:
 *   - FFN: GEMM1 -> SiLU (compute shader) -> GEMM2 in UN command buffer
 *     (l'intermedio [T, ffn] non lascia mai la GPU)
 *   - QKV: 3 GEMM sullo stesso input in UN command buffer
 * Sotto MYNAH_METAL_MIN_T si ritorna -1 e il chiamante resta su CPU. */
#ifdef MYNAH_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <pthread.h>

#define MYNAH_METAL_MIN_T 24

typedef _Float16 f16;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_silu, g_glu, g_dwconv, g_lnorm, g_addbias, g_smax;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const void *host_ptr;
    void *buf;              /* id<MTLBuffer> fp16 (retained) */
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

/* buffer I/O riusabili */
static id<MTLBuffer> g_in, g_mid, g_out, g_out2, g_out3;
static size_t g_in_cap, g_mid_cap, g_out_cap, g_out2_cap, g_out3_cap;

static const char *SILU_SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void silu(device half *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "    float v = float(x[i]);\n"
    "    x[i] = half(v / (1.0f + exp(-v)));\n"
    "}\n"
    /* GLU sui canali: x [T,2d] -> g [T,d] */
    "kernel void glu(device const half *x [[buffer(0)]], device half *g [[buffer(1)]],\n"
    "                constant uint &d [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    uint t = i / d, c = i % d;\n"
    "    float a = float(x[t * 2 * d + c]);\n"
    "    float b = float(x[t * 2 * d + d + c]);\n"
    "    g[i] = half(a / (1.0f + exp(-b)));\n"
    "}\n"
    /* depthwise conv causale k=9: c[t,i] = sum_j w[i,j] * g[t-8+j, i] */
    "kernel void dwconv9(device const half *g [[buffer(0)]], device const half *w [[buffer(1)]],\n"
    "                    device half *c [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                    constant uint &T [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
    "    uint t = i / d, ch = i % d;\n"
    "    float acc = 0.0f;\n"
    "    for (uint j = 0; j < 9; j++) {\n"
    "        int src = (int)t - 8 + (int)j;\n"
    "        if (src >= 0) acc += float(w[ch * 9 + j]) * float(g[(uint)src * d + ch]);\n"
    "    }\n"
    "    c[i] = half(acc);\n"
    "}\n"
    /* layernorm per riga (thread per riga, riduzione seriale: T righe bastano) */
    "kernel void lnorm(device half *x [[buffer(0)]], device const half *gm [[buffer(1)]],\n"
    "                  device const half *bt [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                  uint t [[thread_position_in_grid]]) {\n"
    "    device half *row = x + t * d;\n"
    "    float mu = 0.0f;\n"
    "    for (uint i = 0; i < d; i++) mu += float(row[i]);\n"
    "    mu /= d;\n"
    "    float var = 0.0f;\n"
    "    for (uint i = 0; i < d; i++) { float c = float(row[i]) - mu; var += c * c; }\n"
    "    float inv = rsqrt(var / d + 1e-5f);\n"
    "    for (uint i = 0; i < d; i++)\n"
    "        row[i] = half((float(row[i]) - mu) * inv * float(gm[i]) + float(bt[i]));\n"
    "}\n"
    /* q + bias per-head (broadcast su T): out[t, h*dk+i] = q[..] + bias[h*dk+i] */
    "kernel void addbias(device const half *q [[buffer(0)]], device const half *b [[buffer(1)]],\n"
    "                    device half *o [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                    uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = half(float(q[i]) + float(b[i % d]));\n"
    "}\n"
    /* softmax con rel_shift in forma chiusa e finestra chunked_limited.\n"
       scores [H,T,K] (ac), bd [H,T,P]; P = 2L-1, gq = cached + t (qui cached=0). */
    "kernel void smax_shift(device half *sc [[buffer(0)]], device const half *bd [[buffer(1)]],\n"
    "                       constant uint &T [[buffer(2)]], constant uint &K [[buffer(3)]],\n"
    "                       constant uint &P [[buffer(4)]], constant float &scale [[buffer(5)]],\n"
    "                       constant uint &chunk [[buffer(6)]], constant uint &lc [[buffer(7)]],\n"
    "                       uint idx [[thread_position_in_grid]]) {\n"
    "    uint h = idx / T, t = idx % T;\n"
    "    device half *srow = sc + (h * T + t) * K;\n"
    "    device const half *brow = bd + (h * T + t) * P;\n"
    "    uint tc = t / chunk;\n"
    "    uint j0 = (tc >= lc) ? (tc - lc) * chunk : 0;\n"
    "    uint j1 = min((tc + 1) * chunk, K);\n"
    "    int base = (int)K - 1 - (int)t;\n"          /* p = K-1 + j - t */
    "    float mx = -3.0e38f;\n"
    "    for (uint j = j0; j < j1; j++) {\n"
    "        float v = (float(srow[j]) + float(brow[base + (int)j])) * scale;\n"
    "        srow[j] = half(v);\n"
    "        if (v > mx) mx = v;\n"
    "    }\n"
    "    float sum = 0.0f;\n"
    "    for (uint j = j0; j < j1; j++) { float e = exp(float(srow[j]) - mx); srow[j] = half(e); sum += e; }\n"
    "    float inv = 1.0f / sum;\n"
    "    for (uint j = 0; j < j0; j++) srow[j] = half(0.0f);\n"
    "    for (uint j = j0; j < j1; j++) srow[j] = half(float(srow[j]) * inv);\n"
    "    for (uint j = j1; j < K; j++) srow[j] = half(0.0f);\n"
    "}\n";

int mynah_metal_available(void) {
    static int checked = 0;
    pthread_mutex_lock(&g_mu);
    if (!checked) {
        g_dev = MTLCreateSystemDefaultDevice();
        if (g_dev) {
            g_queue = [g_dev newCommandQueue];
            NSError *err = nil;
            id<MTLLibrary> lib = [g_dev newLibraryWithSource:
                [NSString stringWithUTF8String:SILU_SRC] options:nil error:&err];
            if (!lib && err)
                fprintf(stderr, "mynah metal: shader: %s\n",
                        err.localizedDescription.UTF8String);
            id<MTLComputePipelineState> __strong *ps[] = {&g_silu, &g_glu, &g_dwconv,
                                                          &g_lnorm, &g_addbias, &g_smax};
            const char *names[] = {"silu", "glu", "dwconv9", "lnorm", "addbias", "smax_shift"};
            for (int i = 0; i < 6 && lib; i++) {
                id<MTLFunction> fn = [lib newFunctionWithName:
                    [NSString stringWithUTF8String:names[i]]];
                if (fn) *ps[i] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            }
        }
        checked = 1;
    }
    pthread_mutex_unlock(&g_mu);
    return g_dev != nil && g_queue != nil && g_silu != nil && g_glu != nil &&
           g_dwconv != nil && g_lnorm != nil && g_addbias != nil && g_smax != nil;
}

/* peso f32 -> MTLBuffer fp16 residente (conversione una tantum) */
static id<MTLBuffer> weight_buffer_f16(const float *w, size_t n_elems) {
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) return (__bridge id<MTLBuffer>)g_wc[i].buf;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:n_elems * sizeof(f16)
                                           options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    f16 *dst = (f16 *)buf.contents;
    for (size_t i = 0; i < n_elems; i++) dst[i] = (f16)w[i];
    if (g_wc_n == g_wc_cap) {
        g_wc_cap = g_wc_cap ? g_wc_cap * 2 : 256;
        g_wc = realloc(g_wc, (size_t)g_wc_cap * sizeof(wc_ent));
    }
    g_wc[g_wc_n++] = (wc_ent){.host_ptr = w, .buf = (void *)CFBridgingRetain(buf)};
    return buf;
}

static id<MTLBuffer> io_buffer(id<MTLBuffer> __strong *slot, size_t *cap, size_t bytes) {
    if (*slot == nil || *cap < bytes) {
        size_t want = *cap ? *cap : (1u << 20);
        while (want < bytes) want *= 2;
        *slot = [g_dev newBufferWithLength:want options:MTLResourceStorageModeShared];
        *cap = want;
    }
    return *slot;
}

static void f32_to_f16(const float *src, id<MTLBuffer> dst, size_t n) {
    f16 *d = (f16 *)dst.contents;
    for (size_t i = 0; i < n; i++) d[i] = (f16)src[i];
}

static void f16_to_f32(id<MTLBuffer> src, float *dst, size_t n) {
    const f16 *s = (const f16 *)src.contents;
    for (size_t i = 0; i < n; i++) dst[i] = (float)s[i];
}

static MPSMatrix *mat16(id<MTLBuffer> buf, int rows, int cols) {
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                        rowBytes:(NSUInteger)cols * sizeof(f16)
                        dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

static void encode_gemm(id<MTLCommandBuffer> cb, MPSMatrix *a, MPSMatrix *b,
                        MPSMatrix *c, int T, int n, int k) {
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_dev transposeLeft:NO transposeRight:YES
            resultRows:(NSUInteger)T resultColumns:(NSUInteger)n
       interiorColumns:(NSUInteger)k alpha:1.0 beta:0.0];
    [mm encodeToCommandBuffer:cb leftMatrix:a rightMatrix:b resultMatrix:c];
}

static void encode_silu(id<MTLCommandBuffer> cb, id<MTLBuffer> buf, size_t n) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_silu];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n, g_silu.maxTotalThreadsPerThreadgroup), 1, 1)];
    [enc endEncoding];
}

/* out[T,n] = x[T,k] @ W^T — singola GEMM fp16 */
int mynah_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
    if (T < MYNAH_METAL_MIN_T || !mynah_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> wb = weight_buffer_f16(w, (size_t)n * (size_t)k);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n * sizeof(f16));
        if (wb && xb && ob) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            encode_gemm(cb, mat16(xb, T, k), mat16(wb, n, k), mat16(ob, T, n), T, n, k);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(ob, out, (size_t)T * (size_t)n);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* FFN fusa: out[T,n2] = SiLU(x[T,k] @ W1^T) @ W2^T — un solo wait */
int mynah_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k) {
    if (T < MYNAH_METAL_MIN_T || !mynah_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> w1b = weight_buffer_f16(w1, (size_t)n1 * (size_t)k);
        id<MTLBuffer> w2b = weight_buffer_f16(w2, (size_t)n2 * (size_t)n1);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> mb = io_buffer(&g_mid, &g_mid_cap, (size_t)T * (size_t)n1 * sizeof(f16));
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n2 * sizeof(f16));
        if (w1b && w2b && xb && mb && ob) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            encode_gemm(cb, mat16(xb, T, k), mat16(w1b, n1, k), mat16(mb, T, n1), T, n1, k);
            encode_silu(cb, mb, (size_t)T * (size_t)n1);
            encode_gemm(cb, mat16(mb, T, n1), mat16(w2b, n2, n1), mat16(ob, T, n2), T, n2, n1);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(ob, out, (size_t)T * (size_t)n2);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* QKV: tre GEMM sullo stesso input, un solo wait */
int mynah_metal_gemm3_wt(const float *x, const float *wa, const float *wb_,
                         const float *wc_, float *oa, float *ob_, float *oc,
                         int T, int n, int k) {
    if (T < MYNAH_METAL_MIN_T || !mynah_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> ba = weight_buffer_f16(wa, (size_t)n * (size_t)k);
        id<MTLBuffer> bb = weight_buffer_f16(wb_, (size_t)n * (size_t)k);
        id<MTLBuffer> bc = weight_buffer_f16(wc_, (size_t)n * (size_t)k);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> o1 = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n * sizeof(f16));
        id<MTLBuffer> o2 = io_buffer(&g_out2, &g_out2_cap, (size_t)T * (size_t)n * sizeof(f16));
        id<MTLBuffer> o3 = io_buffer(&g_out3, &g_out3_cap, (size_t)T * (size_t)n * sizeof(f16));
        if (ba && bb && bc && xb && o1 && o2 && o3) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            MPSMatrix *mx = mat16(xb, T, k);
            encode_gemm(cb, mx, mat16(ba, n, k), mat16(o1, T, n), T, n, k);
            encode_gemm(cb, mx, mat16(bb, n, k), mat16(o2, T, n), T, n, k);
            encode_gemm(cb, mx, mat16(bc, n, k), mat16(o3, T, n), T, n, k);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(o1, oa, (size_t)T * (size_t)n);
                f16_to_f32(o2, ob_, (size_t)T * (size_t)n);
                f16_to_f32(o3, oc, (size_t)T * (size_t)n);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* helper: dispatch 1D di un pipeline state */
static void run1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, size_t n) {
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n, ps.maxTotalThreadsPerThreadgroup), 1, 1)];
}

/* vista MPS strided su uno slice per-head: righe T/cols dk con rowBytes = d*2 */
static MPSMatrix *mat16_off(id<MTLBuffer> buf, size_t byte_off, int rows, int cols,
                            int row_stride_elems) {
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                        rowBytes:(NSUInteger)row_stride_elems * sizeof(f16)
                        dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf offset:byte_off descriptor:d];
}

/* Attention completa su GPU in UN command buffer (offline/batch, cache_valid=0):
 * qkv -> +bias_u/v -> rk = pe@relk^T -> per-head ac/bd (GEMM su viste strided)
 * -> softmax+rel_shift+finestra (shader) -> ctx -> o_proj. */
int mynah_metal_attention(const float *xn, const float *pe,
                          const float *wq, const float *wk, const float *wv,
                          const float *wo, const float *relk,
                          const float *bias_u, const float *bias_v,
                          float *out, int T, int d, int H, int left, int right) {
    if (T < MYNAH_METAL_MIN_T || !mynah_metal_available()) return -1;
    const int dk = d / H, P = 2 * T - 1, K = T;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> bwq = weight_buffer_f16(wq, (size_t)d * (size_t)d);
        id<MTLBuffer> bwk = weight_buffer_f16(wk, (size_t)d * (size_t)d);
        id<MTLBuffer> bwv = weight_buffer_f16(wv, (size_t)d * (size_t)d);
        id<MTLBuffer> bwo = weight_buffer_f16(wo, (size_t)d * (size_t)d);
        id<MTLBuffer> brel = weight_buffer_f16(relk, (size_t)d * (size_t)d);
        id<MTLBuffer> bbu = weight_buffer_f16(bias_u, (size_t)d);
        id<MTLBuffer> bbv = weight_buffer_f16(bias_v, (size_t)d);

        const size_t td = (size_t)T * (size_t)d * sizeof(f16);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, td + (size_t)P * (size_t)d * sizeof(f16));
        /* layout scratch: q,k,v,qu,qv,rk,ctx + scores + bd in un buffer unico */
        const size_t sc_off = 7 * td;
        const size_t bd_off = sc_off + (size_t)H * (size_t)T * (size_t)K * sizeof(f16);
        const size_t total = bd_off + (size_t)H * (size_t)T * (size_t)P * sizeof(f16);
        id<MTLBuffer> sb = io_buffer(&g_mid, &g_mid_cap, total);
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, td);
        if (bwq && bwk && bwv && bwo && brel && bbu && bbv && xb && sb && ob) {

        f32_to_f16(xn, xb, (size_t)T * (size_t)d);
        {
            f16 *ped = (f16 *)xb.contents + (size_t)T * (size_t)d;
            for (size_t i = 0; i < (size_t)P * (size_t)d; i++) ped[i] = (f16)pe[i];
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        MPSMatrix *mx = mat16(xb, T, d);
        MPSMatrix *mpe = mat16_off(xb, td, P, d, d);
        /* q,k,v,qu,qv,rk,ctx slot nel buffer sb */
        MPSMatrix *mq = mat16_off(sb, 0 * td, T, d, d);
        MPSMatrix *mk = mat16_off(sb, 1 * td, T, d, d);
        MPSMatrix *mv = mat16_off(sb, 2 * td, T, d, d);
        encode_gemm(cb, mx, mat16(bwq, d, d), mq, T, d, d);
        encode_gemm(cb, mx, mat16(bwk, d, d), mk, T, d, d);
        encode_gemm(cb, mx, mat16(bwv, d, d), mv, T, d, d);
        encode_gemm(cb, mpe, mat16(brel, d, d), mat16_off(sb, 5 * td, P, d, d), P, d, d);

        {   /* qu = q + bias_u ; qv = q + bias_v */
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            uint32_t du = (uint32_t)d;
            [enc setComputePipelineState:g_addbias];
            [enc setBuffer:sb offset:0 atIndex:0];
            [enc setBuffer:bbu offset:0 atIndex:1];
            [enc setBuffer:sb offset:3 * td atIndex:2];
            [enc setBytes:&du length:4 atIndex:3];
            [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setBuffer:bbv offset:0 atIndex:1];
            [enc setBuffer:sb offset:4 * td atIndex:2];
            [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* per-head: ac[h] = qu_h @ k_h^T ; bd[h] = qv_h @ rk_h^T (viste strided) */
        for (int h = 0; h < H; h++) {
            const size_t ho = (size_t)h * (size_t)dk * sizeof(f16);
            MPSMatrix *quh = mat16_off(sb, 3 * td + ho, T, dk, d);
            MPSMatrix *qvh = mat16_off(sb, 4 * td + ho, T, dk, d);
            MPSMatrix *kh = mat16_off(sb, 1 * td + ho, T, dk, d);
            MPSMatrix *rkh = mat16_off(sb, 5 * td + ho, P, dk, d);
            MPSMatrix *ach = mat16_off(sb, sc_off + (size_t)h * (size_t)T * (size_t)K * sizeof(f16), T, K, K);
            MPSMatrix *bdh = mat16_off(sb, bd_off + (size_t)h * (size_t)T * (size_t)P * sizeof(f16), T, P, P);
            encode_gemm(cb, quh, kh, ach, T, K, dk);
            encode_gemm(cb, qvh, rkh, bdh, T, P, dk);
        }

        {   /* softmax + rel_shift + finestra */
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            const uint32_t Tu = (uint32_t)T, Ku = (uint32_t)K, Pu = (uint32_t)P;
            const uint32_t chunk = (uint32_t)(right + 1), lc = (uint32_t)(left / (right + 1));
            const float scale = 1.0f / sqrtf((float)dk);
            [enc setComputePipelineState:g_smax];
            [enc setBuffer:sb offset:sc_off atIndex:0];
            [enc setBuffer:sb offset:bd_off atIndex:1];
            [enc setBytes:&Tu length:4 atIndex:2];
            [enc setBytes:&Ku length:4 atIndex:3];
            [enc setBytes:&Pu length:4 atIndex:4];
            [enc setBytes:&scale length:4 atIndex:5];
            [enc setBytes:&chunk length:4 atIndex:6];
            [enc setBytes:&lc length:4 atIndex:7];
            [enc dispatchThreads:MTLSizeMake((size_t)H * (size_t)T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        /* ctx_h = probs_h @ v_h (output su vista strided [T,dk] dentro [T,d]) */
        for (int h = 0; h < H; h++) {
            const size_t ho = (size_t)h * (size_t)dk * sizeof(f16);
            MPSMatrix *ph = mat16_off(sb, sc_off + (size_t)h * (size_t)T * (size_t)K * sizeof(f16), T, K, K);
            MPSMatrix *vh = mat16_off(sb, 2 * td + ho, T, dk, d);
            MPSMatrix *ch = mat16_off(sb, 6 * td + ho, T, dk, d);
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:g_dev transposeLeft:NO transposeRight:NO
                    resultRows:(NSUInteger)T resultColumns:(NSUInteger)dk
               interiorColumns:(NSUInteger)K alpha:1.0 beta:0.0];
            [mm encodeToCommandBuffer:cb leftMatrix:ph rightMatrix:vh resultMatrix:ch];
        }

        /* out = ctx @ wo^T */
        encode_gemm(cb, mat16_off(sb, 6 * td, T, d, d), mat16(bwo, d, d), mat16(ob, T, d), T, d, d);

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusCompleted) {
            f16_to_f32(ob, out, (size_t)T * (size_t)d);
            rc = 0;
        }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* Conv module su GPU in UN command buffer: pw1 -> GLU -> dwconv9 -> LN -> SiLU -> pw2 */
int mynah_metal_conv(const float *xn, const float *pw1, const float *dw9 /* [d,9] f32 */,
                     const float *ln_w, const float *ln_b, const float *pw2,
                     float *out, int T, int d) {
    if (T < MYNAH_METAL_MIN_T || !mynah_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> b1 = weight_buffer_f16(pw1, (size_t)2 * (size_t)d * (size_t)d);
        id<MTLBuffer> bw = weight_buffer_f16(dw9, (size_t)d * 9u);
        id<MTLBuffer> bg = weight_buffer_f16(ln_w, (size_t)d);
        id<MTLBuffer> bb = weight_buffer_f16(ln_b, (size_t)d);
        id<MTLBuffer> b2 = weight_buffer_f16(pw2, (size_t)d * (size_t)d);
        const size_t td = (size_t)T * (size_t)d * sizeof(f16);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, td);
        id<MTLBuffer> mb = io_buffer(&g_mid, &g_mid_cap, 3 * td); /* h2[T,2d] + g[T,d] */
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, td);
        if (b1 && bw && bg && bb && b2 && xb && mb && ob) {
        f32_to_f16(xn, xb, (size_t)T * (size_t)d);

        const uint32_t du = (uint32_t)d, Tu = (uint32_t)T;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        encode_gemm(cb, mat16(xb, T, d), mat16(b1, 2 * d, d), mat16(mb, T, 2 * d), T, 2 * d, d);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        /* GLU: mb[0..2td) -> mb[2td..3td) */
        [enc setComputePipelineState:g_glu];
        [enc setBuffer:mb offset:0 atIndex:0];
        [enc setBuffer:mb offset:2 * td atIndex:1];
        [enc setBytes:&du length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        /* dwconv: g -> xb (riusato come uscita conv) */
        [enc setComputePipelineState:g_dwconv];
        [enc setBuffer:mb offset:2 * td atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:xb offset:0 atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
        [enc setBytes:&Tu length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        /* LN + SiLU in place su xb */
        [enc setComputePipelineState:g_lnorm];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:bg offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((size_t)T, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((size_t)T, 64ul), 1, 1)];
        [enc setComputePipelineState:g_silu];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        encode_gemm(cb, mat16(xb, T, d), mat16(b2, d, d), mat16(ob, T, d), T, d, d);

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusCompleted) {
            f16_to_f32(ob, out, (size_t)T * (size_t)d);
            rc = 0;
        }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

#endif /* MYNAH_METAL */

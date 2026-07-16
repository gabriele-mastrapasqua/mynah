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
static id<MTLComputePipelineState> g_silu;
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
            id<MTLFunction> fn = [lib newFunctionWithName:@"silu"];
            if (fn) g_silu = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        }
        checked = 1;
    }
    pthread_mutex_unlock(&g_mu);
    return g_dev != nil && g_queue != nil && g_silu != nil;
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

#endif /* MYNAH_METAL */

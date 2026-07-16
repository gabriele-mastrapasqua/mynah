/* Backend Metal via MPSMatrixMultiplication.
 * Pattern da qwen-tts (lato CUDA): PESI RESIDENTI — ogni pointer di peso viene
 * copiato UNA volta in un MTLBuffer e cacheato per chiave-pointer; le attivazioni
 * viaggiano in buffer condivisi riusabili (niente alloc per-call a regime).
 * Sincrono per semplicità (v1): commit + waitUntilCompleted per GEMM. */
#ifdef MYNAH_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <pthread.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const void *host_ptr;
    void *buf;              /* id<MTLBuffer> (retained) */
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

/* buffer I/O riusabili (crescono al bisogno) */
static id<MTLBuffer> g_in, g_out;
static size_t g_in_cap, g_out_cap;

int mynah_metal_available(void) {
    static int checked = 0;
    pthread_mutex_lock(&g_mu);
    if (!checked) {
        g_dev = MTLCreateSystemDefaultDevice();
        if (g_dev) g_queue = [g_dev newCommandQueue];
        checked = 1;
    }
    pthread_mutex_unlock(&g_mu);
    return g_dev != nil && g_queue != nil;
}

static id<MTLBuffer> weight_buffer(const void *w, size_t bytes) {
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) return (__bridge id<MTLBuffer>)g_wc[i].buf;
    id<MTLBuffer> buf = [g_dev newBufferWithBytes:w length:bytes
                                          options:MTLResourceStorageModeShared];
    if (!buf) return nil;
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

/* out[T,n] = x[T,k] @ W[n,k]^T  (0 = ok, -1 = fallback CPU) */
int mynah_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
    if (!mynah_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    @autoreleasepool {
        id<MTLBuffer> wb = weight_buffer(w, (size_t)n * (size_t)k * 4u);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * 4u);
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n * 4u);
        if (!wb || !xb || !ob) { pthread_mutex_unlock(&g_mu); return -1; }
        memcpy(xb.contents, x, (size_t)T * (size_t)k * 4u);

        MPSMatrixDescriptor *dx = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)T columns:(NSUInteger)k
                            rowBytes:(NSUInteger)k * 4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dw = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)n columns:(NSUInteger)k
                            rowBytes:(NSUInteger)k * 4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dout = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)T columns:(NSUInteger)n
                            rowBytes:(NSUInteger)n * 4 dataType:MPSDataTypeFloat32];

        MPSMatrix *mx = [[MPSMatrix alloc] initWithBuffer:xb descriptor:dx];
        MPSMatrix *mw = [[MPSMatrix alloc] initWithBuffer:wb descriptor:dw];
        MPSMatrix *mo = [[MPSMatrix alloc] initWithBuffer:ob descriptor:dout];

        /* out = x @ W^T: transposeRight = YES */
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_dev transposeLeft:NO transposeRight:YES
                resultRows:(NSUInteger)T resultColumns:(NSUInteger)n
           interiorColumns:(NSUInteger)k alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:mx rightMatrix:mw resultMatrix:mo];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            pthread_mutex_unlock(&g_mu);
            return -1;
        }
        memcpy(out, ob.contents, (size_t)T * (size_t)n * 4u);
    }
    pthread_mutex_unlock(&g_mu);
    return 0;
}

#endif /* MYNAH_METAL */

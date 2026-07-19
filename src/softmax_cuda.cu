// Row-wise softmax on the GPU: out[i,:] = softmax(in[i,:]), row-major.
// Safe softmax (subtract the row max before exp) for numerical stability.
// One block per row; each block does two cooperative reductions in shared
// memory (first the row max, then the sum of exp), then writes the normalized
// row. Threads stride over the row, so a row longer than the block is handled
// in several passes and N need not be a multiple of the block size.
#include "gemm/softmax_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#define SOFTMAX_THREADS 256 // power of two: required by the tree reduction below

// Same abort-on-error helper as src/gemm_cuda.cu (kept local so this kernel is
// self-contained).
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace gemm {

__global__ void softmax_rows_kernel(int M, int N,
                                    const float* __restrict__ in,
                                    float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* x = in  + (size_t)row * N;
    float*       y = out + (size_t)row * N;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ float red[SOFTMAX_THREADS];

    // 1) row max (stability shift). Each thread reduces a strided slice of the
    //    row, then a tree reduction combines the per-thread maxima. Threads with
    //    no element (N < nthreads) keep -FLT_MAX, which is neutral for max.
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += nthreads) local_max = fmaxf(local_max, x[j]);
    red[tid] = local_max;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads(); // all threads have read red[0] before we reuse red[]

    // 2) y = exp(x - row_max), and accumulate the row sum. We read x[j] then
    //    write y[j] (same index), so in and out may alias.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        const float e = expf(x[j] - row_max);
        y[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / red[0];
    __syncthreads();

    // 3) normalize
    for (int j = tid; j < N; j += nthreads) y[j] *= inv_sum;
}

void softmax_rows_cuda(int M, int N, const float* in, float* out) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemcpy(dIn, in, sz, cudaMemcpyHostToDevice));

    softmax_rows_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut); // one block per row
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost));
    cudaFree(dIn); cudaFree(dOut);
}

// ---------------------------------------------------------------------------
// Online (2-pass) variant: same I/O and same one-block-per-row shape as
// softmax_rows_kernel, but the row max and the exp-sum are FUSED into a single
// pass over the input -- each thread carries a running (max m_t, normalizer l_t)
// that is rescaled whenever a larger element appears, via
// exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new). The exps are never
// stored, so the write pass recomputes them: 3 memory sweeps (2 reads + 1
// write) against the 3-pass kernel's 5 (3 reads + 2 writes, since it stores
// the exps and then rereads them to normalize). Whether that pays depends on
// where the sweeps are served from: with ~400 resident blocks, rows
// collectively overflow the 5 MB L2 only past N ~ 3k -- measured 1.06x vs the
// 3-pass kernel at N=2048 (everything hits L2, both kernels pay the same
// 2 M*N DRAM bill) but 1.69-1.74x at N=8192/16384. Traffic alone predicts at
// most 5/3 there; the measured ratio can sit above it because the online
// kernel also drops one of the two tree reductions (and its barriers), and
// neither kernel runs at the DRAM roofline, so the time ratio is not pinned
// by the traffic ratio. Folding two partials (m_a, l_a) + (m_b, l_b)
// rescales BOTH normalizers by the common max; the per-element update is just
// that fold with the element seen as the one-element partial (x, 1) -- and the
// same fold plus an output accumulator is FlashAttention's tile update
// (attention_cuda.cu).
__global__ void softmax_rows_online_kernel(int M, int N,
                                           const float* __restrict__ in,
                                           float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* x = in  + (size_t)row * N;
    float*       y = out + (size_t)row * N;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ float red_m[SOFTMAX_THREADS]; // per-thread partial maxima
    __shared__ float red_l[SOFTMAX_THREADS]; // per-thread partial normalizers

    // 1) online partial over this thread's strided slice: (m_t, l_t) advance
    //    together, l_t being the sum of exp(x - m_t), always shifted by the
    //    partial's own running max (no exp of a raw value ever overflows). A
    //    thread with no element (N < nthreads) keeps (-FLT_MAX, 0), the
    //    neutral element for the fold below.
    float m_t = -FLT_MAX;
    float l_t = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        const float m_new = fmaxf(m_t, x[j]);
        l_t = l_t * expf(m_t - m_new) + expf(x[j] - m_new);
        m_t = m_new;
    }
    red_m[tid] = m_t;
    red_l[tid] = l_t;
    __syncthreads(); // publish every partial before the tree reads them in pairs

    // 2) tree reduction, log2 depth -- same shape as the 3-pass kernel's trees,
    //    but a single one (the pair operator folds max AND normalizer at once,
    //    rescaling both sides by the common max). One barrier per level:
    //    publish before the next level reads. After the last level red_m[0] /
    //    red_l[0] hold the row's global (m, l).
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const float m_new = fmaxf(red_m[tid], red_m[tid + s]);
            red_l[tid] = red_l[tid] * expf(red_m[tid] - m_new)
                       + red_l[tid + s] * expf(red_m[tid + s] - m_new);
            red_m[tid] = m_new;
        }
        __syncthreads();
    }

    // 3) second pass over the input: y = exp(x - m) / l. One reciprocal, then
    //    multiplies (the same inv_sum trick as above). We read x[j] then write
    //    y[j] (same index), so in and out may alias, like the 3-pass kernel.
    const float inv_sum = 1.0f / red_l[0];
    for (int j = tid; j < N; j += nthreads)
        y[j] = expf(x[j] - red_m[0]) * inv_sum;
}

void softmax_rows_online_cuda(int M, int N, const float* in, float* out) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemcpy(dIn, in, sz, cudaMemcpyHostToDevice));

    softmax_rows_online_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut); // one block per row
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost));
    cudaFree(dIn); cudaFree(dOut);
}

// Device timing only (no transfers): the 3-pass kernel vs the online (2-pass)
// variant. Softmax reads and writes the whole M*N matrix once, so we report
// effective bandwidth as the figure of merit for both.
void benchmark_softmax(int M, int N) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemset(dIn, 0, sz));

    auto run3 = [&]{ softmax_rows_kernel       <<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut); };
    auto runo = [&]{ softmax_rows_online_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut); };

    int it3 = 0, ito = 0;
    const double t3 = bench::ms_per_iter(run3, &it3);
    const double to = bench::ms_per_iter(runo, &ito);

    // "Effective" bandwidth = algorithmic traffic (one read + one write of M*N)
    // over time, same yardstick for both kernels. The real traffic differs --
    // ~5 M*N accesses for the 3-pass kernel, ~3 M*N for the online one -- and
    // that gap is exactly what the shared yardstick makes visible: the online
    // kernel should post the higher effective GB/s. Standard convention
    // (deliberately understates hardware throughput).
    const double gb = 2.0 * sz / 1e9;
    std::printf("  3-pass %dx%d : %7.3f ms/iter   %7.2f GB/s  [%4d it]\n",
                M, N, t3, gb / (t3 / 1e3), it3);
    std::printf("  online %dx%d : %7.3f ms/iter   %7.2f GB/s  [%4d it]\n",
                M, N, to, gb / (to / 1e3), ito);
    std::printf("  online speedup : %.2fx\n", t3 / to);

    cudaFree(dIn); cudaFree(dOut);
}

} // namespace gemm

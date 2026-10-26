// Row-wise softmax on the GPU, one block per row. The row max is subtracted
// before exp for stability, and threads stride over the row so N can be any
// size.
// Each thread reads x[j] before writing y[j], so in and out may alias.

#include "gemm/softmax_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>

// The tree reductions need a power of two.
#define SOFTMAX_THREADS 256

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

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += nthreads) local_max = fmaxf(local_max, x[j]);
    red[tid] = local_max;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

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

    for (int j = tid; j < N; j += nthreads) y[j] *= inv_sum;
}

void softmax_rows_cuda(int M, int N, const float* in, float* out) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemcpy(dIn, in, sz, cudaMemcpyHostToDevice));

    softmax_rows_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost));
    cudaFree(dIn); cudaFree(dOut);
}

// The online version folds the max and the exp-sum into one pass, so the input
// is read twice instead of three times.
__global__ void softmax_rows_online_kernel(int M, int N,
                                           const float* __restrict__ in,
                                           float* __restrict__ out) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* x = in  + (size_t)row * N;
    float*       y = out + (size_t)row * N;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ float red_m[SOFTMAX_THREADS];
    __shared__ float red_l[SOFTMAX_THREADS];

    // A thread with no element keeps (-FLT_MAX, 0), which the fold ignores.
    float m_t = -FLT_MAX;
    float l_t = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        const float m_new = fmaxf(m_t, x[j]);
        l_t = l_t * expf(m_t - m_new) + expf(x[j] - m_new);
        m_t = m_new;
    }
    red_m[tid] = m_t;
    red_l[tid] = l_t;
    __syncthreads();

    // The pair operator rescales both sides to their common max.
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const float m_new = fmaxf(red_m[tid], red_m[tid + s]);
            red_l[tid] = red_l[tid] * expf(red_m[tid] - m_new)
                       + red_l[tid + s] * expf(red_m[tid + s] - m_new);
            red_m[tid] = m_new;
        }
        __syncthreads();
    }

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

    softmax_rows_online_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost));
    cudaFree(dIn); cudaFree(dOut);
}

void benchmark_softmax(int M, int N) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    // Softmax barely needs the rotation, since its output evicts its input, but
    // every benchmark follows the same protocol.
    const int k = bench::rotation_copies(sz);
    std::vector<float*> dIns(k);
    for (int i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMalloc(&dIns[i], sz));
        CUDA_CHECK(cudaMemset(dIns[i], 0, sz));
    }
    float* dOut;
    CUDA_CHECK(cudaMalloc(&dOut, sz));

    auto run3 = [&](int i){ softmax_rows_kernel       <<<M, SOFTMAX_THREADS>>>(M, N, dIns[i%k], dOut); };
    auto runo = [&](int i){ softmax_rows_online_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIns[i%k], dOut); };

    int it3 = 0, ito = 0;
    const double t3 = bench::ms_per_iter(run3, &it3);
    const double to = bench::ms_per_iter(runo, &ito);

    // Both kernels are measured on one read and one write of M*N. The 3-pass
    // kernel really moves about 5 M*N and the online one 3, so the online one
    // should show the higher GB/s.
    const double gb = 2.0 * sz / 1e9;
    std::printf("  3-pass %dx%d : %7.3f ms/iter   %7.2f GB/s  [%4d it]\n",
                M, N, t3, gb / (t3 / 1e3), it3);
    std::printf("  online %dx%d : %7.3f ms/iter   %7.2f GB/s  [%4d it]\n",
                M, N, to, gb / (to / 1e3), ito);
    std::printf("  online speedup : %.2fx\n", t3 / to);

    for (int i = 0; i < k; ++i) cudaFree(dIns[i]);
    cudaFree(dOut);
}

} // namespace gemm

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

// Device timing only (no transfers). Softmax reads and writes the whole M*N
// matrix once, so we report effective bandwidth as the figure of merit.
void benchmark_softmax(int M, int N) {
    if (M <= 0 || N <= 0) return;
    const size_t sz = (size_t)M * N * sizeof(float);

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemset(dIn, 0, sz));

    auto run = [&]{ softmax_rows_kernel<<<M, SOFTMAX_THREADS>>>(M, N, dIn, dOut); };

    int iters = 0;
    const double t = bench::ms_per_iter(run, &iters);

    // "Effective" bandwidth = algorithmic traffic (one read + one write of M*N)
    // over time. The kernel's real traffic is ~5 M*N accesses (3 passes), so this
    // deliberately understates hardware throughput -- standard convention.
    const double gb = 2.0 * sz / 1e9;
    std::printf("  softmax %dx%d : %7.3f ms/iter   %7.2f GB/s  [%4d it]\n",
                M, N, t, gb / (t / 1e3), iters);

    cudaFree(dIn); cudaFree(dOut);
}

} // namespace gemm

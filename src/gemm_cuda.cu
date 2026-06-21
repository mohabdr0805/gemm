// Tiled GEMM on the GPU: C = alpha*A*B + beta*C, row-major matrices.
// A 16x16 block computes one tile of C; A and B are streamed through shared
// memory to amortize global-memory traffic.
#include "gemm/gemm_cuda.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE 16

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

// As/Bs padded to TILE+1 columns to avoid bank conflicts when reading Bs
// column-wise in the accumulation loop.
__global__ void gemm_kernel(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta, float* __restrict__ C) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;   // C row
    const int col = blockIdx.x * TILE + tx;   // C column

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int ak = t * TILE + tx;   // column of A read by this thread
        const int bk = t * TILE + ty;   // row of B read by this thread

        // Coalesced loads (tx contiguous -> contiguous addresses); 0 out of bounds.
        As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        const int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];   // beta*C: we read the old C back
    }
}

void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));

    CUDA_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice)); // needed for the beta*C term

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

} // namespace gemm

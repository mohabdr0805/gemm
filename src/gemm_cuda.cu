// GEMM on the GPU: C = alpha*A*B + beta*C, row-major.
// Holds the shared-memory tiled kernel (v1), the register-tiled kernel (v2),
// and the fused bias+activation epilogue used for inference.
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

// v1: shared-memory tiled. As/Bs padded to TILE+1 to dodge bank conflicts.
__global__ void gemm_kernel(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta, float* __restrict__ C) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int ak = t * TILE + tx;
        const int bk = t * TILE + ty;
        As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t idx = (size_t)row * N + col;
        float out = alpha * acc;
        if (beta != 0.0f) out += beta * C[idx]; // beta==0: C write-only (BLAS)
        C[idx] = out;
    }
}

// Same tiling as gemm_kernel, but the epilogue adds bias[col] and applies the
// activation before the global write -> one pass, one launch. ACT is a template
// parameter, so apply_act() folds away at compile time (no branch in the loop).
template <Activation ACT>
__global__ void gemm_bias_act_kernel(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta, float* __restrict__ C,
                                     const float* __restrict__ bias) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int ak = t * TILE + tx;
        const int bk = t * TILE + ty;
        As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t idx = (size_t)row * N + col;
        float v = alpha * acc;
        if (beta != 0.0f) v += beta * C[idx]; // beta==0: C write-only (BLAS)
        if (bias) v += bias[col];
        C[idx] = apply_act(ACT, v);
    }
}

// The non-fused path: a separate element-wise bias+activation pass. This is the
// extra global-memory pass that fusion removes; kept only for the benchmark.
template <Activation ACT>
__global__ void bias_act_kernel(int M, int N, float* __restrict__ C,
                                const float* __restrict__ bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        const int col = idx % N;
        float v = C[idx];
        if (bias) v += bias[col];
        C[idx] = apply_act(ACT, v);
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
    if (beta != 0.0f) // beta==0: C is write-only (BLAS) -> skip the upload
        CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// Pick the right template instantiation for a runtime `act`.
static void launch_fused(dim3 grid, dim3 block, int M, int N, int K, float alpha,
                         const float* dA, const float* dB, float beta,
                         float* dC, const float* dbias, Activation act) {
    switch (act) {
        case Activation::ReLU:
            gemm_bias_act_kernel<Activation::ReLU><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
        case Activation::GELU:
            gemm_bias_act_kernel<Activation::GELU><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
        default:
            gemm_bias_act_kernel<Activation::None><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
    }
}

void gemm_bias_act_cuda(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C,
                        const float* bias, Activation act) {
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    const size_t sbias = (size_t)N * sizeof(float);

    float *dA, *dB, *dC, *dbias = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
    if (beta != 0.0f) // beta==0: C is write-only (BLAS) -> skip the upload
        CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice));
    if (bias) {
        CUDA_CHECK(cudaMalloc(&dbias, sbias));
        CUDA_CHECK(cudaMemcpy(dbias, bias, sbias, cudaMemcpyHostToDevice));
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    launch_fused(grid, block, M, N, K, alpha, dA, dB, beta, dC, dbias, act);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (dbias) cudaFree(dbias);
}

// Fused vs two-pass, timed on the device (cudaEvent), excluding transfers and
// malloc. alpha=1, beta=0 is the inference linear layer Y = act(X*W + b).
void benchmark_fusion(int M, int N, int K, Activation act) {
    const float alpha = 1.0f, beta = 0.0f;
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    const size_t sbias = (size_t)N * sizeof(float);

    float *dA, *dB, *dC, *dbias;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMalloc(&dbias, sbias));
    CUDA_CHECK(cudaMemset(dA, 0, sa));
    CUDA_CHECK(cudaMemset(dB, 0, sb));
    CUDA_CHECK(cudaMemset(dbias, 0, sbias));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;

    auto run_fused = [&]{ launch_fused(grid, block, M, N, K, alpha, dA, dB, beta, dC, dbias, act); };
    auto run_two_pass = [&]{
        gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        switch (act) {
            case Activation::ReLU: bias_act_kernel<Activation::ReLU><<<blocks, threads>>>(M, N, dC, dbias); break;
            case Activation::GELU: bias_act_kernel<Activation::GELU><<<blocks, threads>>>(M, N, dC, dbias); break;
            default:               bias_act_kernel<Activation::None><<<blocks, threads>>>(M, N, dC, dbias); break;
        }
    };

    run_fused(); run_two_pass(); // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    cudaEvent_t s, e; float ms_f = 0.f, ms_u = 0.f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_fused();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_f, s, e));

    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_two_pass();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_u, s, e));

    const double gflop = 2.0 * M * N * K / 1e9;
    const double tf = ms_f / iters, tu = ms_u / iters;
    std::printf("  fused    : %7.3f ms/iter   %7.2f GFLOP/s\n", tf, gflop / (tf / 1e3));
    std::printf("  two-pass : %7.3f ms/iter   %7.2f GFLOP/s\n", tu, gflop / (tu / 1e3));
    std::printf("  fusion speedup : %.2fx\n", tu / tf);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dbias);
}

// v2: register tiling. 128x128 block tile, K in steps of BK=8, 256 threads, each
// thread holding an 8x8 micro-block of C in registers. Every shared-memory load
// feeds 8 FMAs (8x8 outer product), so arithmetic intensity is much higher than
// v1 -- the real lever once occupancy is saturated.
__global__ void gemm_reg_kernel(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int NT = (BM / TM) * (BN / TN); // 256 threads/block

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regM[TM], regN[TN];

    // Indices for the cooperative load of the BM*BK and BK*BN shared tiles.
    const int innerRowA = tid / BK, innerColA = tid % BK, strideA = NT / BK;
    const int innerRowB = tid / BN, innerColB = tid % BN, strideB = NT / BN;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            const int gRow = blockRow + innerRowA + off, gCol = k0 + innerColA;
            As[(innerRowA + off) * BK + innerColA] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            const int gRow = k0 + innerRowB + off, gCol = blockCol + innerColB;
            Bs[(innerRowB + off) * BN + innerColB] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[kk * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = blockRow + threadRow * TM + i;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gCol = blockCol + threadCol * TN + j;
            if (gCol >= N) continue;
            const size_t idx = (size_t)gRow * N + gCol;
            float out = alpha * acc[i][j];
            if (beta != 0.0f) out += beta * C[idx]; // beta==0: C write-only (BLAS)
            C[idx] = out;
        }
    }
}

void gemm_cuda_reg(int M, int N, int K, float alpha,
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
    if (beta != 0.0f) // beta==0: C is write-only (BLAS) -> skip the upload
        CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice));

    constexpr int BM = 128, BN = 128;
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_reg_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// v1 vs v2, device timing, no transfers -> the pure kernel speedup from tiling.
void benchmark_gemm_versions(int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMemset(dA, 0, sa));
    CUDA_CHECK(cudaMemset(dB, 0, sb));

    dim3 b1(TILE, TILE), g1((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    constexpr int BM = 128, BN = 128;
    dim3 b2(256), g2((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto run_v1 = [&]{ gemm_kernel<<<g1, b1>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v2 = [&]{ gemm_reg_kernel<<<g2, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };

    run_v1(); run_v2(); CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    cudaEvent_t s, e; float ms1 = 0.f, ms2 = 0.f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_v1();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms1, s, e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_v2();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, s, e));

    const double gflop = 2.0 * M * N * K / 1e9;
    const double t1 = ms1 / iters, t2 = ms2 / iters;
    std::printf("  v1 shared-tiled : %7.3f ms/iter   %7.2f GFLOP/s\n", t1, gflop / (t1 / 1e3));
    std::printf("  v2 register     : %7.3f ms/iter   %7.2f GFLOP/s\n", t2, gflop / (t2 / 1e3));
    std::printf("  register-tiling speedup : %.2fx\n", t1 / t2);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

} // namespace gemm

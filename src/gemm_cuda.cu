// GEMM on the GPU: C = alpha*A*B + beta*C, row-major.
// Holds the shared-memory tiled kernel (v1), the register-tiled kernel (v2),
// and the fused bias+activation epilogue used for inference.
#include "gemm/gemm_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h> // vendor baseline: the number our kernels are measured against
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

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t st = (call);                                       \
        if (st != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n",       \
                         __FILE__, __LINE__, (int)st);                    \
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
        As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.0f;
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
        As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.0f;
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

    int itf = 0, itu = 0;
    const double tf = bench::ms_per_iter(run_fused,    &itf);
    const double tu = bench::ms_per_iter(run_two_pass, &itu);

    const double gflop = 2.0 * M * N * K / 1e9;
    std::printf("  fused    : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n", tf, gflop / (tf / 1e3), itf);
    std::printf("  two-pass : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n", tu, gflop / (tu / 1e3), itu);
    std::printf("  fusion speedup : %.2fx\n", tu / tf);

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
                (gRow < M && gCol < K) ? A[(size_t)gRow * K + gCol] : 0.0f;
        }
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            const int gRow = k0 + innerRowB + off, gCol = blockCol + innerColB;
            Bs[(innerRowB + off) * BN + innerColB] =
                (gRow < K && gCol < N) ? B[(size_t)gRow * N + gCol] : 0.0f;
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

// v3: v2 + two changes ncu pointed at. (1) Global loads are vectorized to float4
// (128-bit) -> a quarter of the load instructions, fully coalesced. (2) The As
// tile is stored TRANSPOSED (As[k][m] instead of As[m][k]), so the compute
// loop's `regM[i] = As[kk][row+i]` reads 8 CONTIGUOUS floats instead of striding
// by BK. NB: re-profiling showed this did NOT move the ~268M shared-load bank
// conflicts -- those are dominated by the Bs read, untouched here -- so the win
// is the 4x fewer global-load instructions, not the conflicts (see README v3).
// Fast path for aligned sizes (M,N % 128 == 0, K % 8 == 0, needed for the
// unguarded float4 loads); any other shape falls back to v2.
__global__ void gemm_reg_v3_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    __shared__ float As[BK * BM]; // TRANSPOSED: As[k * BM + m]
    __shared__ float Bs[BK * BN]; // Bs[k * BN + n]

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN); // 0..15
    const int threadRow = tid / (BN / TN); // 0..15

    // One float4 per thread per tile. A: 128 rows x (8/4=2) col-groups = 256.
    const int innerRowA = tid / (BK / 4);  // 0..127 (the m this thread loads)
    const int innerColA = tid % (BK / 4);  // 0..1  -> *4 = k offset {0,4}
    // B: (8) rows x (128/4=32) col-groups = 256.
    const int innerRowB = tid / (BN / 4);  // 0..7  (the k this thread loads)
    const int innerColB = tid % (BN / 4);  // 0..31 -> *4 = n offset

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // (1) load 4 consecutive K-values of one A row, (2) scatter them into 4
        //     different rows of the transposed As tile (As[k][m]).
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + innerRowA) * K + k0 + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
        // B is already row-major k x n: one float4 in, one float4 out, contiguous.
        *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4]) =
            *reinterpret_cast<const float4*>(
                &B[(size_t)(k0 + innerRowB) * N + blockCol + innerColB * 4]);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[kk * BM + threadRow * TM + i]; // contiguous
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[kk * BN + threadCol * TN + j]; // contiguous
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // Epilogue: write C in float4 (each thread owns an 8x8 block, aligned).
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = blockRow + threadRow * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const size_t idx = (size_t)gRow * N + blockCol + threadCol * TN + j;
            float4 out = { alpha * acc[i][j+0], alpha * acc[i][j+1],
                           alpha * acc[i][j+2], alpha * acc[i][j+3] };
            if (beta != 0.0f) {
                const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                out.x += beta * c.x; out.y += beta * c.y;
                out.z += beta * c.z; out.w += beta * c.w;
            }
            *reinterpret_cast<float4*>(&C[idx]) = out;
        }
    }
}

// v3 + double buffering: each step prefetches the next tile into registers while
// computing the current one from shared, then swaps, overlapping load latency
// with the FMAs. MIN_BLOCKS is the launch_bounds occupancy target -- at 1, ptxas
// uses 130 registers (one block/SM); forcing 2 pins it to the 128-register cliff,
// no spill, occupancy doubled. Details in the v4 README section.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
gemm_reg_v3db_kernel(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    __shared__ float As[2][BK * BM]; // double-buffered, transposed As[buf][k*BM+m]
    __shared__ float Bs[2][BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int innerRowA = tid / (BK / 4), innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    // Prologue: stage tile 0 into buffer 0.
    {
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + innerRowA) * K + innerColA * 4]);
        As[0][(innerColA*4+0)*BM + innerRowA] = a4.x;
        As[0][(innerColA*4+1)*BM + innerRowA] = a4.y;
        As[0][(innerColA*4+2)*BM + innerRowA] = a4.z;
        As[0][(innerColA*4+3)*BM + innerRowA] = a4.w;
        *reinterpret_cast<float4*>(&Bs[0][innerRowB*BN + innerColB*4]) =
            *reinterpret_cast<const float4*>(
                &B[(size_t)innerRowB * N + blockCol + innerColB*4]);
    }
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool has_next = (k0 + BK < K);
        float4 a_next, b_next;
        if (has_next) { // issue next tile's global loads NOW (in flight during compute)
            a_next = *reinterpret_cast<const float4*>(
                &A[(size_t)(blockRow + innerRowA) * K + (k0 + BK) + innerColA * 4]);
            b_next = *reinterpret_cast<const float4*>(
                &B[(size_t)((k0 + BK) + innerRowB) * N + blockCol + innerColB * 4]);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[buf][kk*BM + threadRow*TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[buf][kk*BN + threadCol*TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }

        if (has_next) { // land the prefetched tile in the other buffer, then swap
            As[1-buf][(innerColA*4+0)*BM + innerRowA] = a_next.x;
            As[1-buf][(innerColA*4+1)*BM + innerRowA] = a_next.y;
            As[1-buf][(innerColA*4+2)*BM + innerRowA] = a_next.z;
            As[1-buf][(innerColA*4+3)*BM + innerRowA] = a_next.w;
            *reinterpret_cast<float4*>(&Bs[1-buf][innerRowB*BN + innerColB*4]) = b_next;
            __syncthreads();
            buf = 1 - buf;
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = blockRow + threadRow * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const size_t idx = (size_t)gRow * N + blockCol + threadCol * TN + j;
            float4 out = { alpha * acc[i][j+0], alpha * acc[i][j+1],
                           alpha * acc[i][j+2], alpha * acc[i][j+3] };
            if (beta != 0.0f) {
                const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                out.x += beta * c.x; out.y += beta * c.y;
                out.z += beta * c.z; out.w += beta * c.w;
            }
            *reinterpret_cast<float4*>(&C[idx]) = out;
        }
    }
}

// v5: v4 + conflict-free Bs reads. The ~268M shared-load bank conflicts ncu has
// reported unchanged since v2 all come from the Bs read: `threadCol * TN`
// strides a warp's 16 columns by 8 floats, so their reads land on 4 of the 32
// banks (4-way conflict; the As read is a broadcast, 2 addresses per warp, and
// contributes none). Fix: split each thread's 8 output columns into two groups
// of 4, at threadCol*4 and threadCol*4 + BN/2 -- a warp's 16 threads then read
// 128 CONTIGUOUS floats, covering all 32 banks. Note what does NOT change: the
// staging, the tile layout in shared, the FMA loop. Only which columns a thread
// OWNS changes (ownership, not layout), so the epilogue moves with it.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
gemm_reg_v5_kernel(int M, int N, int K, float alpha,
                   const float* __restrict__ A,
                   const float* __restrict__ B,
                   float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int HN = BN / 2; // the two column groups sit half a tile apart

    __shared__ float As[2][BK * BM]; // double-buffered, transposed As[buf][k*BM+m]
    __shared__ float Bs[2][BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const int innerRowA = tid / (BK / 4), innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    // Prologue: stage tile 0 into buffer 0.
    {
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + innerRowA) * K + innerColA * 4]);
        As[0][(innerColA*4+0)*BM + innerRowA] = a4.x;
        As[0][(innerColA*4+1)*BM + innerRowA] = a4.y;
        As[0][(innerColA*4+2)*BM + innerRowA] = a4.z;
        As[0][(innerColA*4+3)*BM + innerRowA] = a4.w;
        *reinterpret_cast<float4*>(&Bs[0][innerRowB*BN + innerColB*4]) =
            *reinterpret_cast<const float4*>(
                &B[(size_t)innerRowB * N + blockCol + innerColB*4]);
    }
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool has_next = (k0 + BK < K);
        float4 a_next, b_next;
        if (has_next) { // issue next tile's global loads NOW (in flight during compute)
            a_next = *reinterpret_cast<const float4*>(
                &A[(size_t)(blockRow + innerRowA) * K + (k0 + BK) + innerColA * 4]);
            b_next = *reinterpret_cast<const float4*>(
                &B[(size_t)((k0 + BK) + innerRowB) * N + blockCol + innerColB * 4]);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[buf][kk*BM + threadRow*TM + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) { // split read: all 32 banks, no conflict
                regN[j]     = Bs[buf][kk*BN + threadCol*4 + j];
                regN[4 + j] = Bs[buf][kk*BN + threadCol*4 + HN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }

        if (has_next) { // land the prefetched tile in the other buffer, then swap
            As[1-buf][(innerColA*4+0)*BM + innerRowA] = a_next.x;
            As[1-buf][(innerColA*4+1)*BM + innerRowA] = a_next.y;
            As[1-buf][(innerColA*4+2)*BM + innerRowA] = a_next.z;
            As[1-buf][(innerColA*4+3)*BM + innerRowA] = a_next.w;
            *reinterpret_cast<float4*>(&Bs[1-buf][innerRowB*BN + innerColB*4]) = b_next;
            __syncthreads();
            buf = 1 - buf;
        }
    }

    // Epilogue: acc[i][0..3] holds columns threadCol*4.., acc[i][4..7] the
    // group HN further right. Two float4 stores per row, both 16B-aligned.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = blockRow + threadRow * TM + i;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const size_t idx = (size_t)gRow * N + blockCol + threadCol * 4 + h * HN;
            float4 out = { alpha * acc[i][h*4+0], alpha * acc[i][h*4+1],
                           alpha * acc[i][h*4+2], alpha * acc[i][h*4+3] };
            if (beta != 0.0f) {
                const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                out.x += beta * c.x; out.y += beta * c.y;
                out.z += beta * c.z; out.w += beta * c.w;
            }
            *reinterpret_cast<float4*>(&C[idx]) = out;
        }
    }
}

static bool v3_aligned(int M, int N, int K) { return M % 128 == 0 && N % 128 == 0 && K % 8 == 0; }

void gemm_cuda_v3(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    if (!v3_aligned(M, N, K)) { // unguarded float4 needs aligned tiles -> fall back to v2
        gemm_cuda_reg(M, N, K, alpha, A, B, beta, C);
        return;
    }
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
    if (beta != 0.0f)
        CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice));

    constexpr int BM = 128, BN = 128;
    dim3 block(256);
    dim3 grid(N / BN, M / BM);
    // Pick the launch_bounds build from the grid, because neither wins
    // everywhere. For v5 both builds cost 128 registers (the split addressing
    // is cheaper than v3db's, so the kernel sits on the 2-blocks/SM cliff
    // naturally), yet the hint still changes ptxas scheduling enough to
    // measure: <1> is ~3.5 points of cuBLAS ahead at n=1024 (12/12 reps),
    // <2> ~1 point ahead once the grid can seat 2 blocks/SM (11/12 reps).
    // Same crossover as v4: 2 blocks/SM worth of grid.
    int dev = 0, smCount = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));
    if ((int)(grid.x * grid.y) >= 2 * smCount)
        gemm_reg_v5_kernel<2><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    else
        gemm_reg_v5_kernel<1><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// cuBLAS SGEMM with this repo's row-major convention. cuBLAS is column-major;
// the standard trick is to compute C^T = B^T * A^T: a row-major buffer read as
// column-major IS the transpose, so passing (B, A) swapped with (N, M, K) and
// leading dims (N, K, N) yields exactly our row-major C. No transpose kernels,
// no copies. Same beta==0 write-only semantics as the rest of the repo.
void gemm_cublas(int M, int N, int K, float alpha,
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

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(h));

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// v1, v2, v3 (float4), v3 (float4 + double buffering), v4 (the 128-register
// launch_bounds build) and cuBLAS SGEMM, device timing, no transfers, all
// back-to-back. Each kernel is measured over ~200 ms of work (iteration count
// printed), not a fixed count -- see bench_timing.cuh. v3/v4 are timed only on
// the aligned fast path.
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
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    dim3 g3v(N / BN, M / BM); // v3 fast path assumes aligned sizes
    const bool v3ok = v3_aligned(M, N, K);
    auto run_v1 = [&]{ gemm_kernel<<<g1, b1>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v2 = [&]{ gemm_reg_kernel<<<g2, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v3 = [&]{ gemm_reg_v3_kernel<<<g3v, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_cb = [&]{ CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                                N, M, K, &alpha, dB, N, dA, K, &beta, dC, N)); };

    // Warm-up (the first cuBLAS call also pays its workspace init).
    auto run_v3db = [&]{ gemm_reg_v3db_kernel<1><<<g3v, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v4   = [&]{ gemm_reg_v3db_kernel<2><<<g3v, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v5   = [&]{ gemm_reg_v5_kernel<1><<<g3v, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    auto run_v5lb = [&]{ gemm_reg_v5_kernel<2><<<g3v, b2>>>(M, N, K, alpha, dA, dB, beta, dC); };
    run_v1(); run_v2(); if (v3ok) { run_v3(); run_v3db(); run_v4(); run_v5(); run_v5lb(); } run_cb(); CUDA_CHECK(cudaDeviceSynchronize());

    int i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, ib = 0;
    const double t1 = bench::ms_per_iter(run_v1, &i1);
    const double t2 = bench::ms_per_iter(run_v2, &i2);
    double t3 = 0.0, t4 = 0.0, t5 = 0.0, t6 = 0.0, t7 = 0.0;
    if (v3ok) {
        t3 = bench::ms_per_iter(run_v3,   &i3);
        t4 = bench::ms_per_iter(run_v3db, &i4);
        t5 = bench::ms_per_iter(run_v4,   &i5);
        t6 = bench::ms_per_iter(run_v5,   &i6);
        t7 = bench::ms_per_iter(run_v5lb, &i7);
    }
    const double tb = bench::ms_per_iter(run_cb, &ib);

    const double gflop = 2.0 * (double)M * N * K / 1e9;
    const double g1f = gflop / (t1 / 1e3), g2f = gflop / (t2 / 1e3), gbf = gflop / (tb / 1e3);
    std::printf("  v1 shared-tiled  : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t1, g1f, 100.0 * g1f / gbf, i1);
    std::printf("  v2 register      : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t2, g2f, 100.0 * g2f / gbf, i2);
    if (v3ok) {
        const double g3f = gflop / (t3 / 1e3), g4f = gflop / (t4 / 1e3), g5f = gflop / (t5 / 1e3);
        const double g6f = gflop / (t6 / 1e3), g7f = gflop / (t7 / 1e3);
        std::printf("  v3 float4        : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t3, g3f, 100.0 * g3f / gbf, i3);
        std::printf("  v3 float4+2xbuf  : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t4, g4f, 100.0 * g4f / gbf, i4);
        std::printf("  v4 +launch_bounds: %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t5, g5f, 100.0 * g5f / gbf, i5);
        std::printf("  v5 Bs-split      : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t6, g6f, 100.0 * g6f / gbf, i6);
        std::printf("  v5 Bs-split+lb   : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t7, g7f, 100.0 * g7f / gbf, i7);
    }
    std::printf("  cuBLAS SGEMM     : %7.3f ms/iter  %8.2f GFLOP/s                       [%4d it]\n", tb, gbf, ib);
    std::printf("  register-tiling speedup v1->v2 : %.2fx\n", t1 / t2);

    CUBLAS_CHECK(cublasDestroy(h));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

} // namespace gemm

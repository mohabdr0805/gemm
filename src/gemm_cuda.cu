// GEMM on the GPU, C = alpha*A*B + beta*C in row-major. When beta is 0, C is
// never read, as in BLAS.

#include "gemm/gemm_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

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

// Empty shapes and K at 0 never reach a kernel, because a zero grid is a launch
// error and the v3+ prologue would read past A.
static bool gemm_quick_return(int M, int N, int K, float beta, float* C) {
    if (M <= 0 || N <= 0) return true;
    if (K > 0) return false;
    const size_t n = (size_t)M * N;
    if (beta == 0.0f) std::memset(C, 0, n * sizeof(float));
    else for (size_t i = 0; i < n; ++i) C[i] *= beta;
    return true;
}

// v1 tiles A and B in shared memory, padded to TILE+1 against bank conflicts.
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
        if (beta != 0.0f) out += beta * C[idx];
        C[idx] = out;
    }
}

// Bias and activation are applied before the write, so the layer takes one
// launch.
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
        if (beta != 0.0f) v += beta * C[idx];
        if (bias) v += bias[col];
        C[idx] = apply_act(ACT, v);
    }
}

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
    if (gemm_quick_return(M, N, K, beta, C)) return;
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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

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
    // the epilogue still applies at K == 0
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                const size_t idx = (size_t)i * N + j;
                float v = (beta == 0.0f) ? 0.0f : beta * C[idx];
                if (bias) v += bias[j];
                C[idx] = apply_act(act, v);
            }
        return;
    }
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
    if (beta != 0.0f)
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

void benchmark_fusion(int M, int N, int K, Activation act) {
    const float alpha = 1.0f, beta = 0.0f;
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    const size_t sbias = (size_t)N * sizeof(float);

    const int k = bench::rotation_copies(sa + sb);
    std::vector<float*> dAs(k), dBs(k);
    for (int i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMalloc(&dAs[i], sa));
        CUDA_CHECK(cudaMalloc(&dBs[i], sb));
        CUDA_CHECK(cudaMemset(dAs[i], 0, sa));
        CUDA_CHECK(cudaMemset(dBs[i], 0, sb));
    }
    float *dC, *dbias;
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMalloc(&dbias, sbias));
    CUDA_CHECK(cudaMemset(dbias, 0, sbias));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;

    auto run_fused = [&](int i){ launch_fused(grid, block, M, N, K, alpha, dAs[i%k], dBs[i%k], beta, dC, dbias, act); };
    auto run_two_pass = [&](int i){
        gemm_kernel<<<grid, block>>>(M, N, K, alpha, dAs[i%k], dBs[i%k], beta, dC);
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

    for (int i = 0; i < k; ++i) { cudaFree(dAs[i]); cudaFree(dBs[i]); }
    cudaFree(dC); cudaFree(dbias);
}

// v2 keeps an 8x8 block of C in registers per thread.
__global__ void gemm_reg_kernel(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int NT = (BM / TM) * (BN / TN);

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
            if (beta != 0.0f) out += beta * C[idx];
            C[idx] = out;
        }
    }
}

void gemm_cuda_reg(int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C) {
    if (gemm_quick_return(M, N, K, beta, C)) return;
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
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_reg_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// v3 adds float4 loads and a transposed As. The loads are unguarded, so M and N
// must be multiples of 128 and K of 8.
__global__ void gemm_reg_v3_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    __shared__ float As[BK * BM]; // transposed, As[k * BM + m]
    __shared__ float Bs[BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);

    const int innerRowA = tid / (BK / 4);
    const int innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + innerRowA) * K + k0 + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
        *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4]) =
            *reinterpret_cast<const float4*>(
                &B[(size_t)(k0 + innerRowB) * N + blockCol + innerColB * 4]);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[kk * BM + threadRow * TM + i];
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

// v3 with double buffering, so the next loads overlap the compute. The build
// with MIN_BLOCKS = 2 is v4.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
gemm_reg_v3db_kernel(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    __shared__ float As[2][BK * BM];
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
        if (has_next) {
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

        if (has_next) {
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

// v5 splits each thread's 8 columns into two groups half a tile apart, so a
// warp's Bs reads hit all 32 banks.
template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
gemm_reg_v5_kernel(int M, int N, int K, float alpha,
                   const float* __restrict__ A,
                   const float* __restrict__ B,
                   float beta, float* __restrict__ C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int HN = BN / 2;

    __shared__ float As[2][BK * BM];
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
        if (has_next) {
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
            for (int j = 0; j < 4; ++j) {
                regN[j]     = Bs[buf][kk*BN + threadCol*4 + j];
                regN[4 + j] = Bs[buf][kk*BN + threadCol*4 + HN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }

        if (has_next) {
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

// v6 adds a warp tier, each warp owning a 128x16 tile, which cuts the shared
// memory wavefronts per k-step. The geometry is a template parameter so 128x128
// and 256x128 can both ship.
template <int BM, int BN, int WM, int WN, int TM, int TN, int MIN_BLOCKS>
__global__ void __launch_bounds__((BM*BN)/(TM*TN), MIN_BLOCKS)
gemm_reg_v6_kernel(int M, int N, int K, float alpha,
                   const float* __restrict__ A, const float* __restrict__ B,
                   float beta, float* __restrict__ C) {
    constexpr int BK = 8;
    constexpr int NT     = (BM * BN) / (TM * TN);
    constexpr int WCOLS  = BN / WN;
    constexpr int WSUB_N = WN / TN;
    constexpr int GW = 4;
    constexpr int GN = TN / GW, SPN = WN / GN;
    constexpr int GM = TM / GW, SPM = WM / GM;
    constexpr int A4 = (BM * BK / 4) / NT, AR = NT / (BK / 4);
    constexpr int B4 = (BK * BN / 4) / NT, BR = NT / (BN / 4);

    static_assert((BM/WM) * WCOLS == NT / 32, "warp tiling does not pave the block");
    static_assert((WM/TM) * WSUB_N == 32, "a warp must be exactly 32 threads");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "TM and TN must be float4 multiples");
    static_assert((BM*BK) % (4*NT) == 0 && (BK*BN) % (4*NT) == 0, "staging does not divide");

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid  = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int warpRow = (warp / WCOLS) * WM;
    const int warpCol = (warp % WCOLS) * WN;
    const int rowBase = warpRow + (lane / WSUB_N) * GW;
    const int colBase = warpCol + (lane % WSUB_N) * GW;
    const int innerRowA = tid / (BK / 4), innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    #pragma unroll
    for (int p = 0; p < A4; ++p) {
        const int rA = innerRowA + p * AR;
        const float4 a4 = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + rA) * K + innerColA * 4]);
        As[0][(innerColA*4+0)*BM + rA] = a4.x;
        As[0][(innerColA*4+1)*BM + rA] = a4.y;
        As[0][(innerColA*4+2)*BM + rA] = a4.z;
        As[0][(innerColA*4+3)*BM + rA] = a4.w;
    }
    #pragma unroll
    for (int p = 0; p < B4; ++p) {
        const int rB = innerRowB + p * BR;
        *reinterpret_cast<float4*>(&Bs[0][rB*BN + innerColB*4]) =
            *reinterpret_cast<const float4*>(
                &B[(size_t)rB * N + blockCol + innerColB*4]);
    }
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool has_next = (k0 + BK < K);
        float4 a_next[A4], b_next[B4];
        if (has_next) {
            #pragma unroll
            for (int p = 0; p < A4; ++p)
                a_next[p] = *reinterpret_cast<const float4*>(
                    &A[(size_t)(blockRow + innerRowA + p*AR) * K + (k0 + BK) + innerColA * 4]);
            #pragma unroll
            for (int p = 0; p < B4; ++p)
                b_next[p] = *reinterpret_cast<const float4*>(
                    &B[(size_t)((k0 + BK) + innerRowB + p*BR) * N + blockCol + innerColB * 4]);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int g = 0; g < GM; ++g)
                #pragma unroll
                for (int i = 0; i < GW; ++i)
                    regM[g*GW + i] = As[buf][kk*BM + rowBase + g*SPM + i];
            #pragma unroll
            for (int g = 0; g < GN; ++g)
                #pragma unroll
                for (int j = 0; j < GW; ++j)
                    regN[g*GW + j] = Bs[buf][kk*BN + colBase + g*SPN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }

        if (has_next) {
            #pragma unroll
            for (int p = 0; p < A4; ++p) {
                const int rA = innerRowA + p * AR;
                As[1-buf][(innerColA*4+0)*BM + rA] = a_next[p].x;
                As[1-buf][(innerColA*4+1)*BM + rA] = a_next[p].y;
                As[1-buf][(innerColA*4+2)*BM + rA] = a_next[p].z;
                As[1-buf][(innerColA*4+3)*BM + rA] = a_next[p].w;
            }
            #pragma unroll
            for (int p = 0; p < B4; ++p)
                *reinterpret_cast<float4*>(
                    &Bs[1-buf][(innerRowB + p*BR)*BN + innerColB*4]) = b_next[p];
            __syncthreads();
            buf = 1 - buf;
        }
    }

    #pragma unroll
    for (int t = 0; t < TM; ++t) {
        const int gRow = blockRow + rowBase + (t / GW) * SPM + (t % GW);
        #pragma unroll
        for (int g = 0; g < GN; ++g) {
            const size_t idx = (size_t)gRow * N + blockCol + colBase + g * SPN;
            float4 out = { alpha * acc[t][g*GW+0], alpha * acc[t][g*GW+1],
                           alpha * acc[t][g*GW+2], alpha * acc[t][g*GW+3] };
            if (beta != 0.0f) {
                const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                out.x += beta * c.x; out.y += beta * c.y;
                out.z += beta * c.z; out.w += beta * c.w;
            }
            *reinterpret_cast<float4*>(&C[idx]) = out;
        }
    }
}

// v6_big is the 256x128 tile cuBLAS picks on this card.
#define GEMM_V6_SMALL 128, 128, 128, 16,  8, 8
#define GEMM_V6_BIG   256, 128, 256, 16, 16, 8

static bool v3_aligned(int M, int N, int K) { return M % 128 == 0 && N % 128 == 0 && K % 8 == 0; }

// A grid pays for ceil(blocks/sm) full waves, see docs/wave-quantization.md.
static double wave_efficiency(int blocks, int sm) {
    const double w = (double)blocks / sm;
    return w / std::ceil(w);
}

// 256x128 wins when it quantizes at least as well, and needs M % 256 == 0.
static bool pick_big_tile(int M, int N, int sm) {
    if (M % 256 != 0) return false;
    const int small = (M / 128) * (N / 128);
    const int big   = (M / 256) * (N / 128);
    return wave_efficiency(big, sm) >= wave_efficiency(small, sm);
}

void gemm_cuda_v3(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C) {
    if (gemm_quick_return(M, N, K, beta, C)) return; // before v3_aligned, since K==0 passes K%8==0
    if (!v3_aligned(M, N, K)) {
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

    int dev = 0, smCount = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

    if (pick_big_tile(M, N, smCount)) {
        gemm_reg_v6_kernel<GEMM_V6_BIG, 1><<<dim3(N / 128, M / 256), 256>>>(
            M, N, K, alpha, dA, dB, beta, dC);
    } else {
        // <1> while the grid cannot seat two blocks per SM, <2> once it can.
        const dim3 grid(N / 128, M / 128);
        if ((int)(grid.x * grid.y) >= 2 * smCount)
            gemm_reg_v6_kernel<GEMM_V6_SMALL, 2><<<grid, 256>>>(M, N, K, alpha, dA, dB, beta, dC);
        else
            gemm_reg_v6_kernel<GEMM_V6_SMALL, 1><<<grid, 256>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// cuBLAS is column-major. Passing (B, A) and (N, M, K) computes C^T = B^T*A^T,
// which is C in row-major, with no copy.
void gemm_cublas(int M, int N, int K, float alpha,
                 const float* A, const float* B, float beta, float* C) {
    if (gemm_quick_return(M, N, K, beta, C)) return;
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

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(h));

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void benchmark_gemm_versions(int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    // Inputs rotate over twice the L2. C does not, every iteration overwrites
    // it.
    const int k = bench::rotation_copies(sa + sb);

    std::vector<float*> dAs((size_t)k), dBs((size_t)k);
    for (int i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMalloc(&dAs[i], sa));
        CUDA_CHECK(cudaMalloc(&dBs[i], sb));
        CUDA_CHECK(cudaMemset(dAs[i], 0, sa));
        CUDA_CHECK(cudaMemset(dBs[i], 0, sb));
    }
    float* dC;
    CUDA_CHECK(cudaMalloc(&dC, sc));

    dim3 b1(TILE, TILE), g1((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    constexpr int BM = 128, BN = 128;
    dim3 b2(256), g2((N + BN - 1) / BN, (M + BM - 1) / BM);
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    dim3 g3v(N / BN, M / BM);
    const bool v3ok = v3_aligned(M, N, K);
    auto A = [&](int i) { return dAs[i % k]; };
    auto B = [&](int i) { return dBs[i % k]; };
    auto run_v1 = [&](int i){ gemm_kernel<<<g1, b1>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_v2 = [&](int i){ gemm_reg_kernel<<<g2, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_v3 = [&](int i){ gemm_reg_v3_kernel<<<g3v, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_cb = [&](int i){ CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                                N, M, K, &alpha, B(i), N, A(i), K, &beta, dC, N)); };

    auto run_v3db = [&](int i){ gemm_reg_v3db_kernel<1><<<g3v, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_v4   = [&](int i){ gemm_reg_v3db_kernel<2><<<g3v, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_v5   = [&](int i){ gemm_reg_v5_kernel<1><<<g3v, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    auto run_v5lb = [&](int i){ gemm_reg_v5_kernel<2><<<g3v, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC); };
    // v6 goes through gemm_cuda_v3's dispatch, so the bench times what the API
    // launches.
    int smCount = 0, bdev = 0;
    CUDA_CHECK(cudaGetDevice(&bdev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, bdev));
    const bool big = v3ok && pick_big_tile(M, N, smCount);
    const dim3 gbig(N / 128, big ? M / 256 : 1);
    auto run_v6   = [&](int i){
        if (big) gemm_reg_v6_kernel<GEMM_V6_BIG,   1><<<gbig, b2>>>(M, N, K, alpha, A(i), B(i), beta, dC);
        else     gemm_reg_v6_kernel<GEMM_V6_SMALL, 2><<<g3v,  b2>>>(M, N, K, alpha, A(i), B(i), beta, dC);
    };
    run_v1(0); run_v2(0); if (v3ok) { run_v3(0); run_v3db(0); run_v4(0); run_v5(0); run_v5lb(0); run_v6(0); } run_cb(0); CUDA_CHECK(cudaDeviceSynchronize());

    int i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, i8 = 0, ib = 0;
    const double t1 = bench::ms_per_iter(run_v1, &i1);
    const double t2 = bench::ms_per_iter(run_v2, &i2);
    double t3 = 0.0, t4 = 0.0, t5 = 0.0, t6 = 0.0, t7 = 0.0, t8 = 0.0;
    if (v3ok) {
        t3 = bench::ms_per_iter(run_v3,   &i3);
        t4 = bench::ms_per_iter(run_v3db, &i4);
        t5 = bench::ms_per_iter(run_v4,   &i5);
        t6 = bench::ms_per_iter(run_v5,   &i6);
        t7 = bench::ms_per_iter(run_v5lb, &i7);
        t8 = bench::ms_per_iter(run_v6,   &i8);
    }
    const double tb = bench::ms_per_iter(run_cb, &ib);

    const double gflop = 2.0 * (double)M * N * K / 1e9;
    const double g1f = gflop / (t1 / 1e3), g2f = gflop / (t2 / 1e3), gbf = gflop / (tb / 1e3);
    std::printf("  v1 shared-tiled  : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t1, g1f, 100.0 * g1f / gbf, i1);
    std::printf("  v2 register      : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t2, g2f, 100.0 * g2f / gbf, i2);
    if (v3ok) {
        const double g3f = gflop / (t3 / 1e3), g4f = gflop / (t4 / 1e3), g5f = gflop / (t5 / 1e3);
        const double g6f = gflop / (t6 / 1e3), g7f = gflop / (t7 / 1e3), g8f = gflop / (t8 / 1e3);
        std::printf("  v3 float4        : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t3, g3f, 100.0 * g3f / gbf, i3);
        std::printf("  v3 float4+2xbuf  : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t4, g4f, 100.0 * g4f / gbf, i4);
        std::printf("  v4 +launch_bounds: %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t5, g5f, 100.0 * g5f / gbf, i5);
        std::printf("  v5 Bs-split      : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t6, g6f, 100.0 * g6f / gbf, i6);
        std::printf("  v5 Bs-split+lb   : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t7, g7f, 100.0 * g7f / gbf, i7);
        std::printf("  v6 warp-tiled    : %7.3f ms/iter  %8.2f GFLOP/s   (%5.1f%% of cuBLAS)  [%4d it]\n", t8, g8f, 100.0 * g8f / gbf, i8);
    }
    std::printf("  cuBLAS SGEMM     : %7.3f ms/iter  %8.2f GFLOP/s                       [%4d it]\n", tb, gbf, ib);
    std::printf("  register-tiling speedup v1->v2 : %.2fx\n", t1 / t2);

    CUBLAS_CHECK(cublasDestroy(h));
    for (int i = 0; i < k; ++i) { cudaFree(dAs[i]); cudaFree(dBs[i]); }
    cudaFree(dC);
}

} // namespace gemm

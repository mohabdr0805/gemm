// FlashAttention, O = softmax(scale * Q*K^T) * V for one head. The N x N score
// matrix is never stored, a running max and sum are rescaled tile by tile.

#include "gemm/attention_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>

#define ATTN_THREADS 128
#define ATTN_BC      32
#define ATTN_DMAX    128

#define ATTN2_BR     64
#define ATTN2_BC     32

#define ATTN_FA2_WPB 8
#define ATTN_FA2_BC  32

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

// One block per query row. The +1 padding puts the score loop on distinct
// banks.
__global__ void flash_attention_kernel(int M, int N, int d, float scale,
                                       const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O, bool causal) {
    const int i = blockIdx.x;
    if (i >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float q_sh[ATTN_DMAX];
    __shared__ float acc [ATTN_DMAX];
    __shared__ float Ks  [ATTN_BC][ATTN_DMAX + 1];
    __shared__ float Vs  [ATTN_BC][ATTN_DMAX + 1];
    __shared__ float s_sh[ATTN_BC];
    __shared__ float red [ATTN_BC];
    __shared__ float m_sh, l_sh;

    for (int t = tid; t < d; t += nthreads) { q_sh[t] = Q[(size_t)i * d + t]; acc[t] = 0.0f; }
    if (tid == 0) { m_sh = -FLT_MAX; l_sh = 0.0f; }
    __syncthreads();

    // Causal rows stop at key i, so no mask is needed.
    const int nkeys = causal ? min(N, i + 1) : N;

    for (int j0 = 0; j0 < nkeys; j0 += ATTN_BC) {
        const int tile = min(ATTN_BC, nkeys - j0);

        for (int idx = tid; idx < tile * d; idx += nthreads) {
            const int c = idx / d, t = idx % d;
            Ks[c][t] = K[(size_t)(j0 + c) * d + t];
            Vs[c][t] = V[(size_t)(j0 + c) * d + t];
        }
        __syncthreads();

        if (tid < tile) {
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += q_sh[t] * Ks[tid][t];
            s_sh[tid] = scale * dot;
        }
        __syncthreads();

        if (tid < ATTN_BC) red[tid] = (tid < tile) ? s_sh[tid] : -FLT_MAX;
        __syncthreads();
        for (int s = ATTN_BC >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        const float tile_max = red[0];
        __syncthreads();

        const float m_old = m_sh;
        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = expf(m_old - m_new);

        if (tid < tile) s_sh[tid] = expf(s_sh[tid] - m_new);
        __syncthreads();
        if (tid < ATTN_BC) red[tid] = (tid < tile) ? s_sh[tid] : 0.0f;
        __syncthreads();
        for (int s = ATTN_BC >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float tile_sum = red[0];

        if (tid == 0) { l_sh = l_sh * corr + tile_sum; m_sh = m_new; }
        for (int t = tid; t < d; t += nthreads) {
            float a = acc[t] * corr;
            for (int c = 0; c < tile; ++c) a += s_sh[c] * Vs[c][t];
            acc[t] = a;
        }
        __syncthreads();
    }

    const float inv = 1.0f / l_sh;
    for (int t = tid; t < d; t += nthreads) O[(size_t)i * d + t] = acc[t] * inv;
}

void flash_attention_cuda(int M, int N, int d, float scale,
                          const float* Q, const float* K, const float* V,
                          float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    if (d > ATTN_DMAX) {
        std::fprintf(stderr, "flash_attention_cuda: head dim d=%d exceeds ATTN_DMAX=%d\n",
                     d, ATTN_DMAX);
        std::abort();
    }
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemcpy(dQ, Q, sq,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K, skv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V, skv, cudaMemcpyHostToDevice));

    flash_attention_kernel<<<M, ATTN_THREADS>>>(M, N, d, scale, dQ, dK, dV, dO, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, sq, cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// v2 shares each K/V tile between ATTN2_BR rows, one thread per row with its
// state in registers. q and acc spill at D=128.
template <int D>
__global__ void flash_attention_v2_kernel(int M, int N, float scale,
                                          const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O, bool causal) {
    constexpr int BR = ATTN2_BR, BC = ATTN2_BC;
    __shared__ float Ks[BC][D + 1];
    __shared__ float Vs[BC][D + 1];

    const int r = threadIdx.x;
    const int i = blockIdx.x * BR + r;
    const bool active = (i < M);
    const int nthreads = blockDim.x;

    float q[D], acc[D];
    #pragma unroll
    for (int t = 0; t < D; ++t) { q[t] = active ? Q[(size_t)i * D + t] : 0.0f; acc[t] = 0.0f; }
    float m = -FLT_MAX, l = 0.0f;
    const int nkeys_i = causal ? min(N, i + 1) : N;

    const int nkeys_block = causal ? min(N, min(blockIdx.x * BR + BR, M)) : N;

    for (int j0 = 0; j0 < nkeys_block; j0 += BC) {
        const int tile = min(BC, N - j0);

        // every thread loads the tile, inactive rows included
        for (int idx = r; idx < tile * D; idx += nthreads) {
            const int c = idx / D, t = idx % D;
            Ks[c][t] = K[(size_t)(j0 + c) * D + t];
            Vs[c][t] = V[(size_t)(j0 + c) * D + t];
        }
        __syncthreads();

        if (active) {
            float s[BC];
            float tile_max = -FLT_MAX;
            int cnt = 0;
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                if (c >= tile || j0 + c >= nkeys_i) break;
                float dot = 0.0f;
                #pragma unroll
                for (int t = 0; t < D; ++t) dot += q[t] * Ks[c][t];
                s[c] = scale * dot;
                tile_max = fmaxf(tile_max, s[c]);
                cnt = c + 1;
            }
            if (cnt > 0) {
                const float m_new = fmaxf(m, tile_max);
                const float corr  = expf(m - m_new);
                l *= corr;
                #pragma unroll
                for (int t = 0; t < D; ++t) acc[t] *= corr;
                #pragma unroll
                for (int c = 0; c < BC; ++c) {
                    if (c >= cnt) break;
                    const float p = expf(s[c] - m_new);
                    l += p;
                    #pragma unroll
                    for (int t = 0; t < D; ++t) acc[t] += p * Vs[c][t];
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv = 1.0f / l;
        #pragma unroll
        for (int t = 0; t < D; ++t) O[(size_t)i * D + t] = acc[t] * inv;
    }
}

// FA-2 gives each row a warp and splits D across the lanes, so nothing spills
// at D=128. A __shfl_xor butterfly rebuilds each score from the 32 partial
// sums.
template <int D>
__global__ void flash_attention_fa2_kernel(int M, int N, float scale,
                                           const float* __restrict__ Q,
                                           const float* __restrict__ K,
                                           const float* __restrict__ V,
                                           float* __restrict__ O, bool causal) {
    constexpr int WPB = ATTN_FA2_WPB, BC = ATTN_FA2_BC;
    constexpr int DL  = D / 32;
    constexpr int BR  = WPB;

    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nthreads = blockDim.x;
    const int i = blockIdx.x * BR + warp;
    const bool active = (i < M);

    float q[DL], acc[DL];
    #pragma unroll
    for (int k = 0; k < DL; ++k) {
        const int t = lane + k * 32;
        q[k]   = active ? Q[(size_t)i * D + t] : 0.0f;
        acc[k] = 0.0f;
    }
    float m = -FLT_MAX, l = 0.0f;
    const int nkeys_i     = causal ? min(N, i + 1) : N;
    const int nkeys_block = causal ? min(N, min(blockIdx.x * BR + BR, M)) : N;

    for (int j0 = 0; j0 < nkeys_block; j0 += BC) {
        const int tile = min(BC, N - j0);

        for (int idx = tid; idx < tile * D; idx += nthreads) {
            const int c = idx / D, t = idx % D;
            Ks[c][t] = K[(size_t)(j0 + c) * D + t];
            Vs[c][t] = V[(size_t)(j0 + c) * D + t];
        }
        __syncthreads();

        if (active) {
            float s[BC];
            float tile_max = -FLT_MAX;
            int cnt = 0;
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                if (c >= tile || j0 + c >= nkeys_i) break; // warp-uniform break, no lane misses the shuffle
                float partial = 0.0f;
                #pragma unroll
                for (int k = 0; k < DL; ++k) partial += q[k] * Ks[c][lane + k * 32];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    partial += __shfl_xor_sync(0xffffffff, partial, off);
                s[c] = scale * partial;
                tile_max = fmaxf(tile_max, s[c]);
                cnt = c + 1;
            }
            if (cnt > 0) {
                const float m_new = fmaxf(m, tile_max);
                const float corr  = expf(m - m_new);
                l *= corr;
                #pragma unroll
                for (int k = 0; k < DL; ++k) acc[k] *= corr;
                #pragma unroll
                for (int c = 0; c < BC; ++c) {
                    if (c >= cnt) break;
                    const float p = expf(s[c] - m_new);
                    l += p;
                    #pragma unroll
                    for (int k = 0; k < DL; ++k) acc[k] += p * Vs[c][lane + k * 32];
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv = 1.0f / l;
        #pragma unroll
        for (int k = 0; k < DL; ++k) O[(size_t)i * D + (lane + k * 32)] = acc[k] * inv;
    }
}

// Only d = 32, 64 and 128 have v2 and FA-2 kernels, other d fall back to v1.
static bool v2_supports(int d) { return d == 32 || d == 64 || d == 128; }

static void launch_v2(dim3 grid, dim3 block, int M, int N, int d, float scale,
                      const float* dQ, const float* dK, const float* dV,
                      float* dO, bool causal) {
    switch (d) {
        case 32:  flash_attention_v2_kernel<32> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 64:  flash_attention_v2_kernel<64> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 128: flash_attention_v2_kernel<128><<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        default:
            std::fprintf(stderr, "launch_v2: unsupported head dim d=%d (expected 32/64/128)\n", d);
            std::abort();
    }
}

void flash_attention_cuda_v2(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    if (!v2_supports(d)) {
        flash_attention_cuda(M, N, d, scale, Q, K, V, O, causal);
        return;
    }
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemcpy(dQ, Q, sq,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K, skv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V, skv, cudaMemcpyHostToDevice));

    dim3 block(ATTN2_BR);
    dim3 grid((M + ATTN2_BR - 1) / ATTN2_BR);
    launch_v2(grid, block, M, N, d, scale, dQ, dK, dV, dO, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, sq, cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

static void launch_fa2(dim3 grid, dim3 block, int M, int N, int d, float scale,
                       const float* dQ, const float* dK, const float* dV,
                       float* dO, bool causal) {
    switch (d) {
        case 32:  flash_attention_fa2_kernel<32> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 64:  flash_attention_fa2_kernel<64> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 128: flash_attention_fa2_kernel<128><<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        default:
            std::fprintf(stderr, "launch_fa2: unsupported head dim d=%d (expected 32/64/128)\n", d);
            std::abort();
    }
}

void flash_attention_cuda_fa2(int M, int N, int d, float scale,
                              const float* Q, const float* K, const float* V,
                              float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    if (!v2_supports(d)) {
        flash_attention_cuda_v2(M, N, d, scale, Q, K, V, O, causal);
        return;
    }
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemcpy(dQ, Q, sq,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K, skv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V, skv, cudaMemcpyHostToDevice));

    dim3 block(32 * ATTN_FA2_WPB);
    dim3 grid((M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB);
    launch_fa2(grid, block, M, N, d, scale, dQ, dK, dV, dO, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, sq, cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

void benchmark_attention_versions(int M, int N, int d, bool causal) {
    if (M <= 0 || N <= 0 || !v2_supports(d)) return;
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    // Inputs rotate because Q, K and V fit in the L2 (see bench_timing.cuh).
    const int k = bench::rotation_copies(sq + 2 * skv);
    std::vector<float*> dQs(k), dKs(k), dVs(k);
    for (int i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMalloc(&dQs[i], sq));
        CUDA_CHECK(cudaMalloc(&dKs[i], skv));
        CUDA_CHECK(cudaMalloc(&dVs[i], skv));
        CUDA_CHECK(cudaMemset(dQs[i], 0, sq));
        CUDA_CHECK(cudaMemset(dKs[i], 0, skv));
        CUDA_CHECK(cudaMemset(dVs[i], 0, skv));
    }
    float* dO;
    CUDA_CHECK(cudaMalloc(&dO, sq));

    const float scale = 1.0f / std::sqrt((float)d);
    dim3 b2(ATTN2_BR), g2((M + ATTN2_BR - 1) / ATTN2_BR);
    auto run_v1 = [&](int i){ flash_attention_kernel<<<M, ATTN_THREADS>>>(M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };
    auto run_v2 = [&](int i){ launch_v2(g2, b2, M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };

    int it1 = 0, it2 = 0;
    const double t1 = bench::ms_per_iter(run_v1, &it1);
    const double t2 = bench::ms_per_iter(run_v2, &it2);

    double gflop = 4.0 * (double)M * N * d / 1e9;
    if (causal) gflop *= 0.5;
    std::printf("  v1 one-row/block   %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", t1, gflop / (t1 / 1e3), it1);
    std::printf("  v2 query-tiled     %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", t2, gflop / (t2 / 1e3), it2);
    std::printf("  query-tiling speedup : %.2fx\n", t1 / t2);

    for (int i = 0; i < k; ++i) { cudaFree(dQs[i]); cudaFree(dKs[i]); cudaFree(dVs[i]); }
    cudaFree(dO);
}

void benchmark_attention_fa2(int M, int N, int d, bool causal) {
    if (M <= 0 || N <= 0 || !v2_supports(d)) return;
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    const int k = bench::rotation_copies(sq + 2 * skv);
    std::vector<float*> dQs(k), dKs(k), dVs(k);
    for (int i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMalloc(&dQs[i], sq));
        CUDA_CHECK(cudaMalloc(&dKs[i], skv));
        CUDA_CHECK(cudaMalloc(&dVs[i], skv));
        CUDA_CHECK(cudaMemset(dQs[i], 0, sq));
        CUDA_CHECK(cudaMemset(dKs[i], 0, skv));
        CUDA_CHECK(cudaMemset(dVs[i], 0, skv));
    }
    float* dO;
    CUDA_CHECK(cudaMalloc(&dO, sq));

    const float scale = 1.0f / std::sqrt((float)d);
    dim3 b2(ATTN2_BR), g2((M + ATTN2_BR - 1) / ATTN2_BR);
    dim3 bf(32 * ATTN_FA2_WPB), gf((M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB);
    auto run_v2  = [&](int i){ launch_v2 (g2, b2, M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };
    auto run_fa2 = [&](int i){ launch_fa2(gf, bf, M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };

    int it2 = 0, itf = 0;
    const double t2 = bench::ms_per_iter(run_v2,  &it2);
    const double tf = bench::ms_per_iter(run_fa2, &itf);

    double gflop = 4.0 * (double)M * N * d / 1e9;
    if (causal) gflop *= 0.5;
    std::printf("  v2  one-thread/row %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", t2, gflop / (t2 / 1e3), it2);
    std::printf("  fa2 one-warp/row   %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", tf, gflop / (tf / 1e3), itf);
    std::printf("  fa2 vs v2 speedup : %.2fx\n", t2 / tf);

    for (int i = 0; i < k; ++i) { cudaFree(dQs[i]); cudaFree(dKs[i]); cudaFree(dVs[i]); }
    cudaFree(dO);
}

} // namespace gemm

// v3 gives each thread a 4x4 block of scores, so the dot product stays in the
// thread and each exp is computed once. v2 lacked threads and FA-2 recomputed
// the exp in all 32 lanes (docs/attention.md).
#define ATTN3_BR      64
#define ATTN3_BC      32
#define ATTN3_THREADS 128

namespace {

// Laid out by hand because D=128 needs more than 48 KB of shared memory.
struct V3Shared {
    float* Qs;
    float* Vs;
    float* KP;   // K, then P once the scores are done
    float* mrow;
    float* lrow;
    float* crow;
};

__device__ __host__ inline int v3_kp_floats(int D) {
    const int as_k = ATTN3_BC * (D + 1);
    const int as_p = ATTN3_BR * (ATTN3_BC + 1);
    return as_k > as_p ? as_k : as_p;
}

__device__ inline V3Shared v3_layout(float* raw, int D) {
    V3Shared s;
    s.Qs = raw;
    s.Vs = s.Qs + (size_t)ATTN3_BR * (D + 1);
    s.KP = s.Vs + (size_t)ATTN3_BC * (D + 1);
    s.mrow = s.KP + v3_kp_floats(D);
    s.lrow = s.mrow + ATTN3_BR;
    s.crow = s.lrow + ATTN3_BR;
    return s;
}

inline size_t v3_shared_bytes(int D) {
    return sizeof(float) * ((size_t)ATTN3_BR * (D + 1) + (size_t)ATTN3_BC * (D + 1)
                            + (size_t)v3_kp_floats(D) + 3 * ATTN3_BR);
}

} // namespace

namespace gemm {

template <int D>
__global__ void flash_attention_v3_kernel(int M, int N, float scale,
                                          const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O, bool causal) {
    constexpr int BR = ATTN3_BR, BC = ATTN3_BC, NT = ATTN3_THREADS;
    constexpr int TM = 4, TN = 4;
    constexpr int RG = BR / TM;
    constexpr int CG = BC / TN;
    constexpr int DG = NT / RG;
    constexpr int TD = D / DG;

    extern __shared__ float raw[];
    V3Shared sh = v3_layout(raw, D);

    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * BR;

    const int sr = tid / CG, sc = tid % CG;
    const int orow = tid / DG, ocol = tid % DG;

    for (int idx = tid; idx < BR * D; idx += NT) {
        const int r = idx / D, t = idx % D;
        sh.Qs[r * (D + 1) + t] = (row0 + r < M) ? Q[(size_t)(row0 + r) * D + t] : 0.0f;
    }
    for (int r = tid; r < BR; r += NT) { sh.mrow[r] = -FLT_MAX; sh.lrow[r] = 0.0f; }

    float acc[TM][TD];
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int t = 0; t < TD; ++t) acc[m][t] = 0.0f;
    __syncthreads();

    const int nkeys_block = causal ? min(N, min(row0 + BR, M)) : N;

    for (int j0 = 0; j0 < nkeys_block; j0 += BC) {
        const int tile = min(BC, N - j0);

        for (int idx = tid; idx < BC * D; idx += NT) {
            const int c = idx / D, t = idx % D;
            const bool ok = (c < tile);
            sh.KP[c * (D + 1) + t] = ok ? K[(size_t)(j0 + c) * D + t] : 0.0f;
            sh.Vs[c * (D + 1) + t] = ok ? V[(size_t)(j0 + c) * D + t] : 0.0f;
        }
        __syncthreads();

        float s[TM][TN];
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            #pragma unroll
            for (int n = 0; n < TN; ++n) s[m][n] = 0.0f;

        for (int t = 0; t < D; ++t) {
            float a[TM], b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = sh.Qs[(sr * TM + m) * (D + 1) + t];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = sh.KP[(sc * TN + n) * (D + 1) + t];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n) s[m][n] += a[m] * b[n];
        }
        __syncthreads();   // K is dead, the buffer becomes P

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const int r = sr * TM + m, gi = row0 + r;
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const int c = sc * TN + n, gj = j0 + c;
                const bool live = (c < tile) && (gi < M) && (!causal || gj <= gi);
                sh.KP[r * (BC + 1) + c] = live ? scale * s[m][n] : -FLT_MAX;
            }
        }
        __syncthreads();

        {
            constexpr int SUB = NT / BR;
            constexpr int PER = BC / SUB;
            const int r = tid / SUB, sub = tid % SUB;
            float local = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < PER; ++c)
                local = fmaxf(local, sh.KP[r * (BC + 1) + sub * PER + c]);
            #pragma unroll
            for (int step = 1; step < SUB; step <<= 1)
                local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, step));

            const float m_old = sh.mrow[r];
            const float m_new = fmaxf(m_old, local);
            const float corr = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < PER; ++c) {
                const int off = r * (BC + 1) + sub * PER + c;
                const float sv = sh.KP[off];
                const float p = (sv == -FLT_MAX) ? 0.0f : expf(sv - m_new);
                sh.KP[off] = p;
                sum += p;
            }
            #pragma unroll
            for (int step = 1; step < SUB; step <<= 1)
                sum += __shfl_xor_sync(0xffffffff, sum, step);

            if (sub == 0) {
                sh.mrow[r] = m_new;
                sh.lrow[r] = sh.lrow[r] * corr + sum;
                sh.crow[r] = corr;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const float corr = sh.crow[orow * TM + m];
            #pragma unroll
            for (int t = 0; t < TD; ++t) acc[m][t] *= corr;
        }
        for (int c = 0; c < BC; ++c) {
            float p[TM], v[TD];
            #pragma unroll
            for (int m = 0; m < TM; ++m) p[m] = sh.KP[(orow * TM + m) * (BC + 1) + c];
            #pragma unroll
            for (int t = 0; t < TD; ++t) v[t] = sh.Vs[c * (D + 1) + ocol * TD + t];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int t = 0; t < TD; ++t) acc[m][t] += p[m] * v[t];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int r = orow * TM + m, gi = row0 + r;
        if (gi >= M) continue;
        const float l = sh.lrow[r];
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int t = 0; t < TD; ++t) O[(size_t)gi * D + ocol * TD + t] = acc[m][t] * inv;
    }
}

static void launch_v3(int M, int N, int d, float scale, const float* Q, const float* K,
                      const float* V, float* O, bool causal) {
    dim3 block(ATTN3_THREADS), grid((M + ATTN3_BR - 1) / ATTN3_BR);
    const size_t smem = v3_shared_bytes(d);
    if (d == 32) {
        cudaFuncSetAttribute(flash_attention_v3_kernel<32>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        flash_attention_v3_kernel<32><<<grid, block, smem>>>(M, N, scale, Q, K, V, O, causal);
    } else if (d == 64) {
        cudaFuncSetAttribute(flash_attention_v3_kernel<64>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        flash_attention_v3_kernel<64><<<grid, block, smem>>>(M, N, scale, Q, K, V, O, causal);
    } else {
        cudaFuncSetAttribute(flash_attention_v3_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        flash_attention_v3_kernel<128><<<grid, block, smem>>>(M, N, scale, Q, K, V, O, causal);
    }
}

void flash_attention_cuda_v3(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemcpy(dQ, Q, sq,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K, skv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V, skv, cudaMemcpyHostToDevice));

    launch_v3(M, N, d, scale, dQ, dK, dV, dO, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, sq, cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

} // namespace gemm

// C entry points on device pointers, so Python can time the kernels on its own
// buffers without allocation or copies.
#if defined(_WIN32)
#  define GEMM_API extern "C" __declspec(dllexport)
#else
#  define GEMM_API extern "C" __attribute__((visibility("default")))
#endif

GEMM_API void gemm_attn_v1(int M, int N, int d, float scale, const float* Q,
                           const float* K, const float* V, float* O, int causal) {
    gemm::flash_attention_kernel<<<M, ATTN_THREADS>>>(M, N, d, scale, Q, K, V, O, causal != 0);
}

GEMM_API void gemm_attn_v2(int M, int N, int d, float scale, const float* Q,
                           const float* K, const float* V, float* O, int causal) {
    dim3 b(ATTN2_BR), g((M + ATTN2_BR - 1) / ATTN2_BR);
    gemm::launch_v2(g, b, M, N, d, scale, Q, K, V, O, causal != 0);
}

GEMM_API void gemm_attn_fa2(int M, int N, int d, float scale, const float* Q,
                            const float* K, const float* V, float* O, int causal) {
    dim3 b(32 * ATTN_FA2_WPB), g((M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB);
    gemm::launch_fa2(g, b, M, N, d, scale, Q, K, V, O, causal != 0);
}

GEMM_API void gemm_attn_v3(int M, int N, int d, float scale, const float* Q,
                           const float* K, const float* V, float* O, int causal) {
    gemm::launch_v3(M, N, d, scale, Q, K, V, O, causal != 0);
}

GEMM_API int gemm_attn_last_error(void) { return (int)cudaGetLastError(); }
GEMM_API int gemm_attn_sync(void) { return (int)cudaDeviceSynchronize(); }

GEMM_API int gemm_attn_l2_bytes(void) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    return p.l2CacheSize;
}

// Returns the blocks launched and sets *threads, so occupancy is read, not
// assumed.
GEMM_API int gemm_attn_launch_shape(int which, int M, int* threads) {
    if (which == 1) { *threads = ATTN_THREADS; return M; }
    if (which == 2) { *threads = ATTN2_BR; return (M + ATTN2_BR - 1) / ATTN2_BR; }
    if (which == 3) { *threads = ATTN3_THREADS; return (M + ATTN3_BR - 1) / ATTN3_BR; }
    *threads = 32 * ATTN_FA2_WPB;
    return (M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB;
}

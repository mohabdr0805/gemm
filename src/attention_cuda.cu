// FlashAttention-style attention on the GPU: O = softmax(scale * Q*K^T) * V,
// row-major, one head. The N x N score matrix is never materialized -- each
// query row (one block) streams the keys in tiles of ATTN_BC, carrying a running
// softmax (running max m, running normalizer l, running output accumulator acc)
// that is rescaled whenever a new tile raises the max. That online-softmax trick
// is what lets the whole thing -- scores, softmax and the P*V product -- run in a
// single fused pass with O(N*d) extra memory instead of O(N*N).
#include "gemm/attention_cuda.cuh"
#include "bench_timing.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>

#define ATTN_THREADS 128 // threads per block (one block handles one query row)
#define ATTN_BC      32  // key tile width (power of two: tree reductions below)
#define ATTN_DMAX    128 // max head dimension supported (shared-memory budget)

#define ATTN2_BR     64  // v2: query rows per block (== threads per block)
#define ATTN2_BC     32  // v2: key tile width

#define ATTN_FA2_WPB 8   // FA-2: warps per block (== query rows per block, one warp/row)
#define ATTN_FA2_BC  32  // FA-2: key tile width

// Same abort-on-error helper as the other .cu files (kept local so this kernel
// is self-contained).
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

// One block per query row. d is a runtime value, capped at ATTN_DMAX so the
// shared tiles can be sized at compile time. Ks/Vs rows are padded to
// [ATTN_BC][ATTN_DMAX+1]: the odd stride gives the per-key dot product (threads
// walking different rows at the same column) distinct banks -- the same
// bank-conflict dodge as the GEMM tiles.
__global__ void flash_attention_kernel(int M, int N, int d, float scale,
                                       const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O, bool causal) {
    const int i = blockIdx.x;       // query row
    if (i >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float q_sh[ATTN_DMAX];               // this row's query, reused for every key
    __shared__ float acc [ATTN_DMAX];               // running (unnormalized) output
    __shared__ float Ks  [ATTN_BC][ATTN_DMAX + 1];
    __shared__ float Vs  [ATTN_BC][ATTN_DMAX + 1];
    __shared__ float s_sh[ATTN_BC];                 // tile scores, reused for tile probs
    __shared__ float red [ATTN_BC];                 // reduction scratch
    __shared__ float m_sh, l_sh;                    // running max, running normalizer

    // Load the query row and zero the accumulator / running stats.
    for (int t = tid; t < d; t += nthreads) { q_sh[t] = Q[(size_t)i * d + t]; acc[t] = 0.0f; }
    if (tid == 0) { m_sh = -FLT_MAX; l_sh = 0.0f; }
    __syncthreads();

    // Causal: row i only attends to keys j <= i, so the key loop just stops at
    // i+1. Whole tiles past the diagonal are never touched and the last tile is
    // naturally truncated, so no per-element mask is needed.
    const int nkeys = causal ? min(N, i + 1) : N;

    for (int j0 = 0; j0 < nkeys; j0 += ATTN_BC) {
        const int tile = min(ATTN_BC, nkeys - j0); // valid keys in this tile

        // 1) stage the K and V tiles in shared memory (coalesced over tile*d floats)
        for (int idx = tid; idx < tile * d; idx += nthreads) {
            const int c = idx / d, t = idx % d;
            Ks[c][t] = K[(size_t)(j0 + c) * d + t];
            Vs[c][t] = V[(size_t)(j0 + c) * d + t];
        }
        __syncthreads();

        // 2) scores s[c] = scale * <q, K[j0+c]>, one thread per key in the tile
        if (tid < tile) {
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += q_sh[t] * Ks[tid][t];
            s_sh[tid] = scale * dot;
        }
        __syncthreads();

        // 3) tile max (lanes with no key are neutral -FLT_MAX)
        if (tid < ATTN_BC) red[tid] = (tid < tile) ? s_sh[tid] : -FLT_MAX;
        __syncthreads();
        for (int s = ATTN_BC >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        const float tile_max = red[0];
        __syncthreads(); // all threads have read red[0] before step 4 reuses red[]

        // The new running max, and the factor that rescales the old state to it.
        // m_sh, tile_max and m_new are identical across the block, so corr is too.
        const float m_old = m_sh;
        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = expf(m_old - m_new); // first tile: m_old=-FLT_MAX -> 0

        // 4) probs p[c] = exp(s[c]-m_new) (overwrite s_sh in place), then tile sum
        if (tid < tile) s_sh[tid] = expf(s_sh[tid] - m_new);
        __syncthreads();
        if (tid < ATTN_BC) red[tid] = (tid < tile) ? s_sh[tid] : 0.0f;
        __syncthreads();
        for (int s = ATTN_BC >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float tile_sum = red[0];

        // 5) fold the tile into the running softmax: rescale the old normalizer
        //    and accumulator by corr (so they are expressed relative to the new
        //    max), then add this tile's contribution.
        if (tid == 0) { l_sh = l_sh * corr + tile_sum; m_sh = m_new; }
        for (int t = tid; t < d; t += nthreads) {
            float a = acc[t] * corr;
            for (int c = 0; c < tile; ++c) a += s_sh[c] * Vs[c][t];
            acc[t] = a;
        }
        __syncthreads(); // tile done; the next iteration overwrites Ks/Vs/s_sh
    }

    // Normalize once at the end: O[i] = acc / l.
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

// ---------------------------------------------------------------------------
// v2: query tiling. A block owns a *tile* of ATTN2_BR query rows that share each
// streamed K/V tile, cutting K/V reads from M to M/ATTN2_BR (the v1->v2 move of
// the GEMM kernels). One thread per row keeps its (m, l) and acc[D] in registers,
// so rows are independent. D is a template parameter (loops unroll); q/acc spill
// at D=128 (~1 KB/thread), where v2 still beats v1 by a smaller margin.
template <int D>
__global__ void flash_attention_v2_kernel(int M, int N, float scale,
                                          const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O, bool causal) {
    constexpr int BR = ATTN2_BR, BC = ATTN2_BC;
    // [D+1] pads the row stride for the cooperative load; the per-score read is a
    // broadcast (all rows read the same Ks[c][t]), so it needs no padding itself.
    __shared__ float Ks[BC][D + 1];
    __shared__ float Vs[BC][D + 1];

    const int r = threadIdx.x;             // query row within the block
    const int i = blockIdx.x * BR + r;     // global query row
    const bool active = (i < M);
    const int nthreads = blockDim.x;

    float q[D], acc[D];
    #pragma unroll
    for (int t = 0; t < D; ++t) { q[t] = active ? Q[(size_t)i * D + t] : 0.0f; acc[t] = 0.0f; }
    float m = -FLT_MAX, l = 0.0f;
    const int nkeys_i = causal ? min(N, i + 1) : N; // this row's own key limit

    // The block streams keys up to the furthest any of its rows needs (the last
    // row for the causal case); each row then stops at its own nkeys_i.
    const int nkeys_block = causal ? min(N, min(blockIdx.x * BR + BR, M)) : N;

    for (int j0 = 0; j0 < nkeys_block; j0 += BC) {
        const int tile = min(BC, N - j0);

        // Cooperative load of the K/V tile (all threads, including inactive ones).
        for (int idx = r; idx < tile * D; idx += nthreads) {
            const int c = idx / D, t = idx % D;
            Ks[c][t] = K[(size_t)(j0 + c) * D + t];
            Vs[c][t] = V[(size_t)(j0 + c) * D + t];
        }
        __syncthreads();

        if (active) {
            // This row's scores for the tile (kept in registers), and the tile max.
            float s[BC];
            float tile_max = -FLT_MAX;
            int cnt = 0;
            #pragma unroll
            for (int c = 0; c < BC; ++c) {
                if (c >= tile || j0 + c >= nkeys_i) break; // physical end / causal end
                float dot = 0.0f;
                #pragma unroll
                for (int t = 0; t < D; ++t) dot += q[t] * Ks[c][t];
                s[c] = scale * dot;
                tile_max = fmaxf(tile_max, s[c]);
                cnt = c + 1;
            }
            if (cnt > 0) {
                // Fold the tile in with a single rescale (per tile, not per key).
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
        __syncthreads(); // before the next tile overwrites Ks/Vs
    }

    if (active) {
        const float inv = 1.0f / l;
        #pragma unroll
        for (int t = 0; t < D; ++t) O[(size_t)i * D + t] = acc[t] * inv;
    }
}

// ---------------------------------------------------------------------------
// FA-2-style: a whole warp owns one query row, splitting D across its 32 lanes
// (lane l owns dims l, l+32, ...), so at D=128 each lane holds 4 elements of
// q/acc and never spills -- fixing v2's D=128 register spill. The cost: the score
// <q,k> is a sum over D, now split across lanes, so a warp butterfly (__shfl_xor,
// registers) recombines the 32 partials. That is the only cross-lane step; (m, l)
// are per-row and P*V is per-lane. Requires D % 32 == 0; a block is ATTN_FA2_WPB
// warps sharing each K/V tile.
template <int D>
__global__ void flash_attention_fa2_kernel(int M, int N, float scale,
                                           const float* __restrict__ Q,
                                           const float* __restrict__ K,
                                           const float* __restrict__ V,
                                           float* __restrict__ O, bool causal) {
    constexpr int WPB = ATTN_FA2_WPB, BC = ATTN_FA2_BC;
    constexpr int DL  = D / 32;   // head-dim elements owned by each lane
    constexpr int BR  = WPB;      // query rows per block (one warp each)

    __shared__ float Ks[BC][D];   // score/PV reads are stride-1 across lanes -> no bank conflict
    __shared__ float Vs[BC][D];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;    // 0..31 : which slice of D this thread owns
    const int warp = tid >> 5;    // 0..WPB-1 : which query row in the block
    const int nthreads = blockDim.x;
    const int i = blockIdx.x * BR + warp;   // global query row (warp-uniform)
    const bool active = (i < M);

    // Per-lane slice of the query and accumulator: DL elements, dims lane+32*k.
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

        // Cooperative load of the K/V tile (all threads, all warps share it).
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
                if (c >= tile || j0 + c >= nkeys_i) break; // warp-uniform: no divergent shuffle
                // (1) each lane's partial dot over its DL dims...
                float partial = 0.0f;
                #pragma unroll
                for (int k = 0; k < DL; ++k) partial += q[k] * Ks[c][lane + k * 32];
                // (2) ...butterfly-reduced across the 32 lanes -> full score in every lane.
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
                    // (3) P*V: each lane accumulates only ITS output dims -- no cross-lane.
                    #pragma unroll
                    for (int k = 0; k < DL; ++k) acc[k] += p * Vs[c][lane + k * 32];
                }
                m = m_new;
            }
        }
        __syncthreads(); // before the next tile overwrites Ks/Vs
    }

    if (active) {
        const float inv = 1.0f / l;
        #pragma unroll
        for (int k = 0; k < DL; ++k) O[(size_t)i * D + (lane + k * 32)] = acc[k] * inv;
    }
}

// v2 only specializes the common head dims; anything else falls back to v1.
static bool v2_supports(int d) { return d == 32 || d == 64 || d == 128; }

static void launch_v2(dim3 grid, dim3 block, int M, int N, int d, float scale,
                      const float* dQ, const float* dK, const float* dV,
                      float* dO, bool causal) {
    switch (d) {
        case 32:  flash_attention_v2_kernel<32> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 64:  flash_attention_v2_kernel<64> <<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        case 128: flash_attention_v2_kernel<128><<<grid, block>>>(M, N, scale, dQ, dK, dV, dO, causal); break;
        default: // callers gate on v2_supports(d); this is a hard backstop
            std::fprintf(stderr, "launch_v2: unsupported head dim d=%d (expected 32/64/128)\n", d);
            std::abort();
    }
}

void flash_attention_cuda_v2(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    if (!v2_supports(d)) { // no specialized kernel for this head dim -> use v1
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

// FA-2 supports the same specialized head dims as v2 (all multiples of 32).
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
    if (!v2_supports(d)) { // no FA-2 kernel for this head dim -> reuse v2's own fallback chain
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

// v1 vs v2, device timing, no transfers -> the pure kernel speedup from query
// tiling. Only meaningful when v2 has a specialized kernel for d (else both run
// v1); the caller uses d in {32,64,128}.
void benchmark_attention_versions(int M, int N, int d, bool causal) {
    if (M <= 0 || N <= 0 || !v2_supports(d)) return; // v2 only times its specialized dims
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemset(dQ, 0, sq));
    CUDA_CHECK(cudaMemset(dK, 0, skv));
    CUDA_CHECK(cudaMemset(dV, 0, skv));

    const float scale = 1.0f / std::sqrt((float)d);
    dim3 b2(ATTN2_BR), g2((M + ATTN2_BR - 1) / ATTN2_BR);
    auto run_v1 = [&]{ flash_attention_kernel<<<M, ATTN_THREADS>>>(M, N, d, scale, dQ, dK, dV, dO, causal); };
    auto run_v2 = [&]{ launch_v2(g2, b2, M, N, d, scale, dQ, dK, dV, dO, causal); };

    int it1 = 0, it2 = 0;
    const double t1 = bench::ms_per_iter(run_v1, &it1);
    const double t2 = bench::ms_per_iter(run_v2, &it2);

    double gflop = 4.0 * (double)M * N * d / 1e9; // Q*K^T and P*V, 2*M*N*d each
    if (causal) gflop *= 0.5;                      // ~half masked (exact for M==N)
    std::printf("  v1 one-row/block   %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", t1, gflop / (t1 / 1e3), it1);
    std::printf("  v2 query-tiled     %s : %7.3f ms/iter   %7.2f GFLOP/s  [%4d it]\n",
                causal ? "causal" : "full  ", t2, gflop / (t2 / 1e3), it2);
    std::printf("  query-tiling speedup : %.2fx\n", t1 / t2);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// v2 (one thread per row) vs FA-2 (one warp per row, head dim split across lanes),
// device timing, no transfers. The question this answers: does FA-2's spill-free
// register footprint at d=128 outweigh v2's higher K/V reuse? d in {32,64,128}.
void benchmark_attention_fa2(int M, int N, int d, bool causal) {
    if (M <= 0 || N <= 0 || !v2_supports(d)) return;
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, sq));
    CUDA_CHECK(cudaMalloc(&dK, skv));
    CUDA_CHECK(cudaMalloc(&dV, skv));
    CUDA_CHECK(cudaMalloc(&dO, sq));
    CUDA_CHECK(cudaMemset(dQ, 0, sq));
    CUDA_CHECK(cudaMemset(dK, 0, skv));
    CUDA_CHECK(cudaMemset(dV, 0, skv));

    const float scale = 1.0f / std::sqrt((float)d);
    dim3 b2(ATTN2_BR), g2((M + ATTN2_BR - 1) / ATTN2_BR);
    dim3 bf(32 * ATTN_FA2_WPB), gf((M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB);
    auto run_v2  = [&]{ launch_v2 (g2, b2, M, N, d, scale, dQ, dK, dV, dO, causal); };
    auto run_fa2 = [&]{ launch_fa2(gf, bf, M, N, d, scale, dQ, dK, dV, dO, causal); };

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

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

} // namespace gemm

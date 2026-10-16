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
#include <vector>  // the benchmark's rotated input buffers

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

    // Rotated inputs. Attention is the benchmark this matters for: at n=4096,
    // d=64, Q+K+V is 3 MB against a 5 MB L2, so without rotation every iteration
    // reads a fully cached working set. Measured cost of the rotation here:
    // ~10%, at both n=1024 and n=4096, 6 alternated rounds out of 6. GEMM by
    // contrast moves by ~0 (see the README methodology note).
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
    float* dO; // not rotated: rewritten every iteration anyway
    CUDA_CHECK(cudaMalloc(&dO, sq));

    const float scale = 1.0f / std::sqrt((float)d);
    dim3 b2(ATTN2_BR), g2((M + ATTN2_BR - 1) / ATTN2_BR);
    auto run_v1 = [&](int i){ flash_attention_kernel<<<M, ATTN_THREADS>>>(M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };
    auto run_v2 = [&](int i){ launch_v2(g2, b2, M, N, d, scale, dQs[i%k], dKs[i%k], dVs[i%k], dO, causal); };

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

    for (int i = 0; i < k; ++i) { cudaFree(dQs[i]); cudaFree(dKs[i]); cudaFree(dVs[i]); }
    cudaFree(dO);
}

// v2 (one thread per row) vs FA-2 (one warp per row, head dim split across lanes),
// device timing, no transfers. The question this answers: does FA-2's spill-free
// register footprint at d=128 outweigh v2's higher K/V reuse? d in {32,64,128}.
void benchmark_attention_fa2(int M, int N, int d, bool causal) {
    if (M <= 0 || N <= 0 || !v2_supports(d)) return;
    const size_t sq  = (size_t)M * d * sizeof(float);
    const size_t skv = (size_t)N * d * sizeof(float);

    const int k = bench::rotation_copies(sq + 2 * skv); // same reason as above
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

// ---------------------------------------------------------------------------
// v3: register micro-tiling -- the v1 -> v2 move of the GEMM, carried over
// ---------------------------------------------------------------------------
//
// What the profiler said about the two kernels above, and why this exists.
//
//   v2, one thread per query row: 67.7 % of its SASS is FFMA, which is a good
//   mix, but its parallelism is exactly M threads. At M=4096 that is 0.24 waves
//   per SM, two active warps, 80 % of cycles with nothing eligible. Handing it a
//   full grid lifts it from 4.2 % to 14.1 % of peak and no further, because two
//   things still cap it: 218 registers per thread, so four blocks of two warps
//   per SM, and a dependent FFMA chain of length D inside the score loop that
//   float semantics forbid ptxas from splitting.
//
//   fa2, one warp per row: fills the machine and issues at half the peak rate,
//   yet only 12.4 % of its instructions are FFMA. Splitting a row across 32
//   lanes divides the multiply-adds by 32 and not the exponentials -- after the
//   shuffle reduction all 32 lanes hold the same score and every one of them
//   calls expf, so the FMA-to-exp ratio falls from 128:1 to 4:1.
//
// The two are equally far from peak for opposite reasons, and neither is fixed
// by sliding along that cursor. This kernel takes the middle: a thread owns a
// TM x TN block of the score tile and a TM x TD block of the output, so
//
//   * the accumulators are independent -- 16 scores and 32 outputs in flight --
//     which is instruction-level parallelism standing in for the occupancy v2
//     cannot have, the same argument as v2/v3 on the GEMM side of this repo;
//   * every shared value loaded feeds TM or TN multiply-adds instead of one;
//   * the exponential stays at one per (row, key), keeping v2's 128:1 ratio
//     rather than fa2's 4:1.
//
// Per K/V tile: score micro-tile GEMM into registers, scores to shared, row max
// and row sum, rescale, output micro-tile GEMM. The score buffer aliases the K
// buffer, dead by that point, to stay inside the shared budget.
#define ATTN3_BR      64   // query rows per block
#define ATTN3_BC      32   // keys per streamed tile
#define ATTN3_THREADS 128  // 4 warps, one per scheduler

namespace {

// Laid out by hand: the tile sizes depend on D, static __shared__ caps at 48 KB,
// and D=128 needs more than that.
struct V3Shared {
    float* Qs;   // [BR][D+1]  loaded once per block
    float* Vs;   // [BC][D+1]  per tile
    float* KP;   // [BC][D+1] as K, then [BR][BC+1] as P
    float* mrow; // [BR] running max
    float* lrow; // [BR] running normaliser
    float* crow; // [BR] this tile's rescale factor
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
    constexpr int TM = 4, TN = 4;          // score micro-tile, 16 per thread
    constexpr int RG = BR / TM;            // 16 row groups
    constexpr int CG = BC / TN;            // 8 key groups, RG * CG == NT
    constexpr int DG = NT / RG;            // 8 column groups for the output
    constexpr int TD = D / DG;             // output micro-tile, TM x TD

    extern __shared__ float raw[];
    V3Shared sh = v3_layout(raw, D);

    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * BR;

    const int sr = tid / CG, sc = tid % CG;   // scores: rows sr*TM.., keys sc*TN..
    const int orow = tid / DG, ocol = tid % DG;

    // Q into shared once: every key tile re-reads it, and during the score GEMM
    // a thread needs rows it does not own.
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
        __syncthreads();   // K is dead here; the buffer becomes P

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

        // Row max, rescale, exp, row sum. SUB threads per row, reduced over
        // adjacent lanes. The exponential is evaluated once per (row, key) --
        // the ratio fa2 gives up.
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

// --- C entry points, device pointers, no allocation and no transfer ----------
//
// The public flash_attention_cuda* functions above own their memory: they
// cudaMalloc, copy the inputs in, launch, and copy back. That is the right shape
// for a caller that has host data, and the wrong shape for a benchmark -- timing
// them would time three allocations and four transfers around the kernel.
// benchmark_attention_versions avoids that by launching the kernels directly on
// buffers it allocated once, which is what these wrappers expose.
//
// They exist so a Python process can time these kernels against another
// library's on the SAME buffers in the SAME process, which is the only way two
// numbers are comparable here.
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

// Grid/block actually launched, so the harness can report occupancy rather than
// assume it. Returns blocks; *threads gets the block size.
GEMM_API int gemm_attn_launch_shape(int which, int M, int* threads) {
    if (which == 1) { *threads = ATTN_THREADS; return M; }
    if (which == 2) { *threads = ATTN2_BR; return (M + ATTN2_BR - 1) / ATTN2_BR; }
    if (which == 3) { *threads = ATTN3_THREADS; return (M + ATTN3_BR - 1) / ATTN3_BR; }
    *threads = 32 * ATTN_FA2_WPB;
    return (M + ATTN_FA2_WPB - 1) / ATTN_FA2_WPB;
}

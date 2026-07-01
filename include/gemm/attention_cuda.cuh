#pragma once

// FlashAttention-style attention on the GPU (CUDA). Same conventions as
// attention.hpp: row-major, a single head, Q (M x d), K/V (N x d), O (M x d),
// scores scaled by `scale` (usually 1/sqrt(d)), optional causal mask (query i
// attends to keys j <= i).
//
// "Flash" = the N x N score / probability matrix is never written to memory.
// Each query row (one block) streams the keys in tiles, carrying a running
// softmax -- running max, running normalizer, running output accumulator -- that
// is rescaled whenever a new tile raises the max (the online-softmax trick). So
// the extra memory is O(N*d), not O(N*N), and there is a single fused pass: the
// scores, the softmax and the P*V product all happen in one kernel.
//
// No CUDA types appear in this header, so it can also be included from host
// translation units (tests, bench) compiled by g++/cl without nvcc.
namespace gemm {

// O[i, :] = softmax(scale * <Q[i], K^T>)[masked] * V.  Q, K, V, O are distinct.
// The head dimension d must be <= 128 (the shared-memory tile budget).
//
// v1: one block per query row -- simple, but every block re-reads all of K/V.
void flash_attention_cuda(int M, int N, int d, float scale,
                          const float* Q, const float* K, const float* V,
                          float* O, bool causal);

// v2: query tiling. A block serves a tile of query rows that share each streamed
// K/V tile, so K/V are read M/tile times, not M times (the arithmetic-intensity
// lever -- same v1->v2 story as the GEMM kernels). Specialized for the common head
// dims d in {32, 64, 128}; any other d transparently falls back to v1. Same result.
void flash_attention_cuda_v2(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal);

// Device-timed throughput (no host<->device transfers). Reports ms/iter and the
// effective GFLOP/s -- attention is ~4*M*N*d flops (two matmuls of 2*M*N*d each),
// and the causal mask touches about half of them.
void benchmark_attention(int M, int N, int d, bool causal);

// Device-timed v1 vs v2 (no transfers): the pure kernel speedup from query
// tiling. Use a head dim d in {32, 64, 128} so v2 runs its specialized kernel.
void benchmark_attention_versions(int M, int N, int d, bool causal);

} // namespace gemm

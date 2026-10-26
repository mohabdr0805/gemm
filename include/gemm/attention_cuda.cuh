#pragma once

// FlashAttention on the GPU for one head, row-major like attention.hpp. Q is
// M x d, K and V are N x d, O is M x d, and they must not overlap. With causal,
// query i sees keys j <= i. The header has no CUDA types, so host code compiles
// it without nvcc.
namespace gemm {

// v1 runs one block per query row and accepts any d up to 128.
void flash_attention_cuda(int M, int N, int d, float scale,
                          const float* Q, const float* K, const float* V,
                          float* O, bool causal);

// v2 shares each K/V tile between a block of query rows. d = 32, 64 and 128 have
// a kernel, other d fall back to v1.
void flash_attention_cuda_v2(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal);

// FA-2 gives each query row a warp and splits d across the lanes. It covers the
// same d as v2, and the others fall back to v2.
void flash_attention_cuda_fa2(int M, int N, int d, float scale,
                              const float* Q, const float* K, const float* V,
                              float* O, bool causal);

// v3 gives each thread a 4x4 block of scores. d must be 32, 64 or 128.
void flash_attention_cuda_v3(int M, int N, int d, float scale,
                             const float* Q, const float* K, const float* V,
                             float* O, bool causal);

// Times v1 against v2 on the device. Use d = 32, 64 or 128.
void benchmark_attention_versions(int M, int N, int d, bool causal);

// Times v2 against FA-2 on the device. Use d = 32, 64 or 128.
void benchmark_attention_fa2(int M, int N, int d, bool causal);

} // namespace gemm

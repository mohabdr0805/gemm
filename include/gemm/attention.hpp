#pragma once
#include <cstddef>

// Attention, CPU reference (oracle for the flash kernel).
// Same row-major convention as the rest of the repo. A single attention head:
//   Q : M x d   (M query rows)
//   K : N x d   (N key rows)
//   V : N x d   (N value rows)
//   O : M x d   (output, one row per query)
// computes, one query row at a time,
//   S = scale * Q * K^T        (M x N scores)
//   P = softmax(S)             (row-wise, safe softmax)
//   O = P * V                  (M x d)
// `scale` is the usual 1/sqrt(d). With `causal = true`, query row i only attends
// to key rows j <= i (the masked-out scores are simply never computed) -- the
// self-attention mask used by decoder LLMs; it assumes the query and key
// positions are aligned, so the natural case is M == N.
namespace gemm {

// The reference is numerically stable (safe softmax: subtract the row max before
// exp). It only ever holds one row of scores, never the full M x N matrix --
// the same property the flash kernel has on the GPU. Q, K, V and O are distinct.
void attention_ref(int M, int N, int d, float scale,
                   const float* Q, const float* K, const float* V,
                   float* O, bool causal);

} // namespace gemm

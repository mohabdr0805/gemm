#pragma once

// Row-wise softmax on the GPU (CUDA). Same conventions as softmax.hpp and the
// rest of the repo: row-major M x N matrix, softmax taken over each row (length
// N). Numerically stable (safe softmax: subtract the row max before exp).
// No CUDA types appear in this header, so it can also be included from host
// translation units (tests, bench) compiled by g++/cl without nvcc.
namespace gemm {

// out[i, :] = softmax(in[i, :]).  in and out may alias (out == in is allowed).
void softmax_rows_cuda(int M, int N, const float* in, float* out);

// Online (2-pass) variant: the row max and the exp-sum are fused into a single
// pass over the input, FlashAttention-style (running max rescales the running
// normalizer). Same contract and same result as softmax_rows_cuda.
void softmax_rows_online_cuda(int M, int N, const float* in, float* out);

// Optional: device-timed throughput (no host<->device transfers), 3-pass kernel
// vs online variant. Softmax is memory-bound, so the figure of merit is
// effective bandwidth, not GFLOP/s.
void benchmark_softmax(int M, int N);

} // namespace gemm

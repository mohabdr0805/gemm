#pragma once
#include <cstddef>

// Row-wise softmax, CPU reference (oracle for the GPU kernel).
// Same conventions as the rest of the repo: matrices are row-major, so an
// M x N matrix is N contiguous floats per row, and the softmax is taken over
// each row (length N) independently:
//   out[i, :] = softmax( in[i, :] )
// The reference is numerically stable (safe softmax): it subtracts the row max
// before exp, so the largest exponent is 0 and exp never overflows.
namespace gemm {

// out and in may alias (out == in is allowed).
void softmax_rows_ref(int M, int N, const float* in, float* out);

} // namespace gemm

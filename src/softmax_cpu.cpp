#include "gemm/softmax.hpp"
#include <cmath>
#include <algorithm>
#include <cfloat>

namespace gemm {

// Row-wise softmax oracle: out[i,:] = softmax(in[i,:]) on a row-major M x N
// matrix. Numerically stable (safe softmax): subtract the row max before exp,
// so the largest exponent is 0 -> exp can never overflow (and very negative
// entries just underflow to 0, as they should). Simple and serial: the
// obviously-correct reference used to validate the GPU kernel. in/out may alias.
void softmax_rows_ref(int M, int N, const float* in, float* out) {
    for (int i = 0; i < M; ++i) {
        if (N <= 0) continue;
        const float* x = in  + (size_t)i * N;
        float*       y = out + (size_t)i * N;

        // 1) row max (the stability shift)
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) m = std::max(m, x[j]);

        // 2) y = exp(x - m), accumulate the row sum (read x[j] before writing
        //    y[j], so out may alias in)
        float s = 0.0f;
        for (int j = 0; j < N; ++j) {
            const float e = std::exp(x[j] - m);
            y[j] = e;
            s += e;
        }

        // 3) normalize
        const float inv = 1.0f / s;
        for (int j = 0; j < N; ++j) y[j] *= inv;
    }
}

} // namespace gemm

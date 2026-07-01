#include "gemm/attention.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cfloat>

namespace gemm {

// Attention oracle, one query row at a time. For row i we form its scores
// s[j] = scale * <Q[i], K[j]>, take a safe softmax over them (subtract the row
// max before exp, so the largest exponent is 0 and exp never overflows -- the
// same trick as softmax_cpu.cpp), then accumulate O[i] = sum_j p[j] * V[j]. Only
// one row of scores is ever held, never the full M x N matrix; the flash kernel
// does exactly this on the GPU. With `causal`, row i stops at key i (j <= i).
// Simple and serial: the obviously-correct reference used to validate the GPU.
void attention_ref(int M, int N, int d, float scale,
                   const float* Q, const float* K, const float* V,
                   float* O, bool causal) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    std::vector<float> s(N);

    for (int i = 0; i < M; ++i) {
        const float* q = Q + (size_t)i * d;
        float*       o = O + (size_t)i * d;
        const int nkeys = causal ? std::min(N, i + 1) : N;

        // 1) scores s[j] = scale * <q, K[j]>
        for (int j = 0; j < nkeys; ++j) {
            const float* k = K + (size_t)j * d;
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += q[t] * k[t];
            s[j] = scale * dot;
        }

        // 2) safe softmax over s[0..nkeys): subtract the row max, then exp
        float m = -FLT_MAX;
        for (int j = 0; j < nkeys; ++j) m = std::max(m, s[j]);
        float sum = 0.0f;
        for (int j = 0; j < nkeys; ++j) {
            const float e = std::exp(s[j] - m);
            s[j] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;

        // 3) O[i] = sum_j p[j] * V[j]   (p[j] = exp(s[j]-m) / sum)
        for (int t = 0; t < d; ++t) o[t] = 0.0f;
        for (int j = 0; j < nkeys; ++j) {
            const float p = s[j] * inv;
            const float* v = V + (size_t)j * d;
            for (int t = 0; t < d; ++t) o[t] += p * v[t];
        }
    }
}

} // namespace gemm

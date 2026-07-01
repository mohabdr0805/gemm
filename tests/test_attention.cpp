#include "gemm/attention.hpp"
#ifdef USE_CUDA
#include "gemm/attention_cuda.cuh"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cfloat>

// Attention correctness:
//   1) Host-only invariants on the CPU oracle. An output row is a convex
//      combination of the value rows (the softmax weights sum to 1), so every
//      O[i,t] must lie within [min_j V[j,t], max_j V[j,t]] and stay finite --
//      even for extreme scores, which is exactly what the safe softmax buys.
//      Plus two closed forms: scale=0 -> uniform weights -> O[i] = mean of V;
//      and causal row 0 -> attends to key 0 only -> O[0] = V[0].
//   2) Under USE_CUDA, the flash kernel vs the CPU oracle (full and causal),
//      same 1e-3 tolerance as the GEMM/softmax tests, with non-tile-multiple
//      sizes and extreme inputs to confirm GPU numerical stability.

// The output must be a convex combination of the value rows: bounded by the
// per-column [min,max] of V and finite. Returns 0 if OK.
static int check_convex(int M, int N, int d, const std::vector<float>& V,
                        const std::vector<float>& O, const char* tag) {
    bool bad = false;
    double slack = 0.0;
    for (int t = 0; t < d; ++t) {
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            const float v = V[(size_t)j * d + t];
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        for (int i = 0; i < M; ++i) {
            const float o = O[(size_t)i * d + t];
            if (!std::isfinite(o)) bad = true;
            if (o < lo) slack = std::max(slack, (double)(lo - o));
            if (o > hi) slack = std::max(slack, (double)(o - hi));
        }
    }
    std::printf("  [%s] convex-combo slack = %.3e, all finite = %s\n",
                tag, slack, bad ? "NO" : "yes");
    if (bad || slack > 1e-4) { std::printf("  FAIL (%s)\n", tag); return 1; }
    return 0;
}

int main() {
    int rc = 0;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // --- 1) CPU oracle: invariants + numerical-stability edge cases -----------
    {
        const int M = 50, N = 70, d = 40; // sizes that aren't nice multiples
        const float scale = 1.0f / std::sqrt((float)d);
        std::vector<float> Q(M * d), K(N * d), V(N * d), O(M * d);
        for (auto& x : Q) x = dist(rng);
        for (auto& x : K) x = dist(rng);
        for (auto& x : V) x = dist(rng);

        // Inject extreme query/key magnitudes: the resulting scores are large
        // enough that a naive exp (no max shift) would overflow, but stay finite,
        // so the safe softmax must keep the output bounded and finite.
        for (int t = 0; t < d; ++t) { Q[t] = 15.0f; K[t] = 15.0f; } // row 0 spikes on key 0

        gemm::attention_ref(M, N, d, scale, Q.data(), K.data(), V.data(), O.data(), false);
        rc |= check_convex(M, N, d, V, O, "oracle full");
        gemm::attention_ref(M, N, d, scale, Q.data(), K.data(), V.data(), O.data(), true);
        rc |= check_convex(M, N, d, V, O, "oracle causal");

        // Closed form A: scale=0 -> all scores 0 -> uniform weights -> O[i] is the
        // mean of all value rows (the same for every query).
        gemm::attention_ref(M, N, d, 0.0f, Q.data(), K.data(), V.data(), O.data(), false);
        std::vector<float> mean(d, 0.0f);
        for (int j = 0; j < N; ++j)
            for (int t = 0; t < d; ++t) mean[t] += V[(size_t)j * d + t];
        for (int t = 0; t < d; ++t) mean[t] /= N;
        double mean_dev = 0.0;
        for (int i = 0; i < M; ++i)
            for (int t = 0; t < d; ++t)
                mean_dev = std::max(mean_dev, (double)std::fabs(O[(size_t)i * d + t] - mean[t]));
        std::printf("  [oracle] scale=0 -> uniform -> O = mean(V), max dev = %.3e\n", mean_dev);
        if (mean_dev > 1e-5) { std::printf("  FAIL (uniform)\n"); rc = 1; }

        // Closed form B: causal row 0 sees only key 0 -> O[0] = V[0] exactly.
        gemm::attention_ref(M, N, d, scale, Q.data(), K.data(), V.data(), O.data(), true);
        double row0_dev = 0.0;
        for (int t = 0; t < d; ++t)
            row0_dev = std::max(row0_dev, (double)std::fabs(O[t] - V[t]));
        std::printf("  [oracle] causal row 0 -> O[0] = V[0], max dev = %.3e\n", row0_dev);
        if (row0_dev > 1e-6) { std::printf("  FAIL (causal row 0)\n"); rc = 1; }
    }

#ifdef USE_CUDA
    // --- 2) flash kernel vs CPU oracle ----------------------------------------
    // One helper, several shapes: N not a multiple of the tile (32), a single key
    // (the tile loop runs once), M != N in both directions (M > N causal makes the
    // tail rows attend to every key -- the N side of min(N,i+1)), and d == the
    // ATTN_DMAX limit (128). Random inputs, with the first query/key row scaled up
    // so the score range is wide enough to need the safe/online softmax.
    {
        const double tol = 1e-3; // same margin as the GEMM/softmax tests
        using attn_fn = void (*)(int, int, int, float, const float*, const float*,
                                 const float*, float*, bool);
        auto run_case = [&](attn_fn fn, const char* name, int M, int N, int d, bool causal) {
            const float scale = 1.0f / std::sqrt((float)d);
            std::vector<float> Q(M * d), K(N * d), V(N * d), Oref(M * d), Ogpu(M * d);
            std::uniform_real_distribution<float> dd(-4.0f, 4.0f);
            for (auto& x : Q) x = dd(rng);
            for (auto& x : K) x = dd(rng);
            for (auto& x : V) x = dd(rng);
            for (int t = 0; t < d; ++t) { Q[t] *= 4.0f; K[t] *= 4.0f; } // widen scores

            gemm::attention_ref(M, N, d, scale, Q.data(), K.data(), V.data(), Oref.data(), causal);
            fn                 (M, N, d, scale, Q.data(), K.data(), V.data(), Ogpu.data(), causal);
            double err = 0.0;
            for (int idx = 0; idx < M * d; ++idx)
                err = std::max(err, (double)std::fabs(Oref[idx] - Ogpu[idx]));
            std::printf("  %-2s vs oracle  M=%3d N=%3d d=%3d %-6s : %.3e (tol %.1e)%s\n",
                        name, M, N, d, causal ? "causal" : "full", err, tol, err > tol ? "  <-- FAIL" : "");
            if (err > tol) rc = 1;
        };

        // v1: broad shape coverage (single key, M != N, d at the ATTN_DMAX limit).
        run_case(gemm::flash_attention_cuda, "v1", 100, 100,  80, false);
        run_case(gemm::flash_attention_cuda, "v1", 100, 100,  80, true);
        run_case(gemm::flash_attention_cuda, "v1",  64,   1,  64, false); // single key
        run_case(gemm::flash_attention_cuda, "v1",  40, 100,  48, true);  // M < N causal
        run_case(gemm::flash_attention_cuda, "v1", 130,  64,  64, true);  // M > N causal
        run_case(gemm::flash_attention_cuda, "v1",  96,  96, 128, false); // d == ATTN_DMAX
        run_case(gemm::flash_attention_cuda, "v1",  96,  96, 128, true);

        // v2 (query-tiled): specialized dims 64 and 128, with M not a multiple of
        // the row tile (ATTN2_BR=64) and M > N causal, plus one odd d to confirm the
        // v1-fallback path also matches the oracle.
        run_case(gemm::flash_attention_cuda_v2, "v2", 200, 200,  64, false);
        run_case(gemm::flash_attention_cuda_v2, "v2", 200, 200,  64, true);
        run_case(gemm::flash_attention_cuda_v2, "v2", 150, 100,  64, true);  // M > N, M % 64 != 0
        run_case(gemm::flash_attention_cuda_v2, "v2", 130, 130, 128, false);
        run_case(gemm::flash_attention_cuda_v2, "v2", 130, 130, 128, true);
        run_case(gemm::flash_attention_cuda_v2, "v2",  70,  90,  48, true);  // odd d -> v1 fallback
    }
#endif

    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}

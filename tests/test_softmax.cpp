#include "gemm/softmax.hpp"
#ifdef USE_CUDA
#include "gemm/softmax_cuda.cuh"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>

// Softmax correctness:
//   1) Host-only invariants on the CPU oracle -- each row sums to 1, every entry
//      is in [0,1] and finite even for extreme inputs (this is what the safe
//      softmax guarantees; a naive exp without the max shift would overflow to
//      Inf/NaN here). Plus a closed-form case: a constant row -> uniform 1/N.
//   2) Under USE_CUDA, the GPU kernel vs the CPU oracle, same 1e-3 tolerance as
//      the GEMM tests, again including extreme rows to confirm GPU stability.

// Checks softmax invariants on an M x N result matrix. Returns 0 if OK.
static int check_invariants(int M, int N, const std::vector<float>& Y,
                            const char* tag) {
    double max_sum_err = 0.0;
    bool bad = false;
    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            const float v = Y[(size_t)i * N + j];
            if (!std::isfinite(v) || v < 0.0f || v > 1.0f) bad = true;
            s += v;
        }
        max_sum_err = std::max(max_sum_err, std::fabs(s - 1.0));
    }
    std::printf("  [%s] max |rowsum-1| = %.3e, all finite & in [0,1] = %s\n",
                tag, max_sum_err, bad ? "NO" : "yes");
    if (bad || max_sum_err > 1e-5) { std::printf("  FAIL (%s)\n", tag); return 1; }
    return 0;
}

int main() {
    int rc = 0;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // --- 1) CPU oracle: invariants + numerical-stability edge cases ----------
    {
        const int M = 64, N = 257; // odd N -> exercises rows that aren't a nice multiple
        std::vector<float> X(M * N), Y(M * N);
        for (auto& x : X) x = dist(rng);

        // Inject extreme rows: a naive softmax (no max shift) would blow up here.
        for (int j = 0; j < N; ++j) {
            X[0 * N + j] = 1e30f;                          // huge positive
            X[1 * N + j] = -1e30f;                         // huge negative
            X[2 * N + j] = (j == 0) ? 1000.0f : -1000.0f;  // one dominant spike
        }
        gemm::softmax_rows_ref(M, N, X.data(), Y.data());
        rc |= check_invariants(M, N, Y, "oracle");

        // Closed form: a constant row maps to the uniform distribution 1/N.
        std::vector<float> c(N, 7.5f), cy(N);
        gemm::softmax_rows_ref(1, N, c.data(), cy.data());
        double maxdev = 0.0;
        for (int j = 0; j < N; ++j)
            maxdev = std::max(maxdev, (double)std::fabs(cy[j] - 1.0f / N));
        std::printf("  [oracle] constant row -> uniform 1/N, max dev = %.3e\n", maxdev);
        if (maxdev > 1e-6) { std::printf("  FAIL (uniform)\n"); rc = 1; }

        // In-place (out == in) must give the same result as out-of-place.
        std::vector<float> Z = X;
        gemm::softmax_rows_ref(M, N, Z.data(), Z.data());
        double alias_err = 0.0;
        for (int i = 0; i < M * N; ++i)
            alias_err = std::max(alias_err, (double)std::fabs(Z[i] - Y[i]));
        std::printf("  [oracle] in-place vs out-of-place, max diff = %.3e\n", alias_err);
        if (alias_err != 0.0) { std::printf("  FAIL (alias)\n"); rc = 1; }
    }

#ifdef USE_CUDA
    // --- 2) GPU kernel vs CPU oracle -----------------------------------------
    // Shapes: N > block size (256) -> rows take several strided passes;
    // N < block size -> idle lanes in the tree reduction (the -FLT_MAX / 0
    // neutral-element path the kernel comments call out); and N == 1 -> each
    // row's softmax is exactly [1.0].
    {
        const double tol = 1e-3; // same margin as the GEMM tests
        const int shapes[][2] = { {200, 300}, {64, 100}, {5, 1} };
        for (const auto& sh : shapes) {
            const int M = sh[0], N = sh[1];
            std::vector<float> X(M * N), Yref(M * N), Ygpu(M * N);
            std::uniform_real_distribution<float> d2(-5.0f, 5.0f);
            for (auto& x : X) x = d2(rng);
            for (int j = 0; j < N; ++j) { X[j] = 1e20f; X[N + j] = -1e20f; } // extreme rows

            gemm::softmax_rows_ref (M, N, X.data(), Yref.data());
            gemm::softmax_rows_cuda(M, N, X.data(), Ygpu.data());

            double err = 0.0;
            for (int i = 0; i < M * N; ++i)
                err = std::max(err, (double)std::fabs(Yref[i] - Ygpu[i]));
            std::printf("Max abs error (softmax gpu vs oracle, %3dx%3d) : %.3e (tol %.1e)\n",
                        M, N, err, tol);
            if (err > tol) { std::printf("FAIL (softmax cuda %dx%d)\n", M, N); rc = 1; }
        }
    }
#endif

    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}

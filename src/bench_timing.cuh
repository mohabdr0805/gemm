#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Shared timing helper for every device benchmark in this repo (GEMM, fusion,
// softmax, attention). Times `run` over enough iterations to cover ~200 ms of
// GPU work and returns ms/iter, so each kernel is measured over the same window
// whatever its speed.
//
// Why not a fixed count: 50 iterations was both too few and too many. Too few
// for the fast kernels -- the v3 GEMM at n=1024 runs 0.15 ms, so 50 iterations
// measured 7 ms and the ~5 us launch overhead became several percent of noise
// (its % of cuBLAS swung 87-113% between runs; sized by time it sits at 105+/-1).
// Too many for the slow ones -- v1 at n=4096 runs 100 ms, so the same 50
// iterations soaked the card for 5 s and moved the clocks under whatever was
// timed next, which biased the very ratios the benchmark exists to report.
namespace gemm {
namespace bench {

inline void ck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// Returns ms/iter. If iters_out is non-null, the chosen iteration count is
// written back so callers can print it (the benchmarks do: a reader can see how
// many iterations a figure rests on).
template <class F>
double ms_per_iter(F&& run, int* iters_out = nullptr) {
    constexpr double TARGET_MS = 200.0;
    constexpr int MIN_ITERS = 3, MAX_ITERS = 5000;

    cudaEvent_t s, e;
    ck(cudaEventCreate(&s), "eventCreate");
    ck(cudaEventCreate(&e), "eventCreate");

    run(); // warm-up: first launch pays module load / workspace init
    ck(cudaDeviceSynchronize(), "warm-up");

    float probe = 0.0f; // one timed iteration to size the loop
    ck(cudaEventRecord(s), "record");
    run();
    ck(cudaEventRecord(e), "record");
    ck(cudaEventSynchronize(e), "sync");
    ck(cudaEventElapsedTime(&probe, s, e), "elapsed");

    int iters = (probe > 0.0f) ? (int)(TARGET_MS / probe) : MAX_ITERS;
    if (iters < MIN_ITERS) iters = MIN_ITERS;
    if (iters > MAX_ITERS) iters = MAX_ITERS;

    float ms = 0.0f;
    ck(cudaEventRecord(s), "record");
    for (int i = 0; i < iters; ++i) run();
    ck(cudaEventRecord(e), "record");
    ck(cudaEventSynchronize(e), "sync");
    ck(cudaEventElapsedTime(&ms, s, e), "elapsed");

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    if (iters_out) *iters_out = iters;
    return ms / iters;
}

} // namespace bench
} // namespace gemm

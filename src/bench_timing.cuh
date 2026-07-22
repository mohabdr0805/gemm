#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Shared timing helper: runs `run` for ~200 ms and returns ms/iter, so every
// kernel is measured over the same window whatever its speed. A fixed count is
// too few for fast kernels (launch overhead) and too many for slow ones (clock
// drift under the next kernel); the README methodology note has the numbers.
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

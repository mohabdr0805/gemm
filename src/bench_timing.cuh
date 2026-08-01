#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

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

// How many copies of the input a benchmark must cycle through so that the
// rotated footprint reaches twice the L2 capacity. Below that, an iteration can
// read what the previous one left in cache and the loop times a warm case no
// real workload sees: in a model, the next call to the same layer comes after
// dozens of other kernels have flushed the L2. NVIDIA's GEMM measurement
// guidelines call this buffer rotation and set the 2x threshold.
//
// The L2 size is read from the device rather than hard-coded, so the harness
// stays correct on another card. Capped: past the threshold more copies only
// cost memory. Returns 1 when one iteration already exceeds it, and the caller
// then rotates over a single buffer, which is a no-op.
inline int rotation_copies(size_t input_bytes, int cap = 32) {
    if (input_bytes == 0) return 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 1;
    const size_t target = 2 * (size_t)prop.l2CacheSize;
    size_t k = (target + input_bytes - 1) / input_bytes;
    if (k < 1) k = 1;
    if (k > (size_t)cap) k = (size_t)cap;
    return (int)k;
}

// Returns ms/iter. If iters_out is non-null, the chosen iteration count is
// written back so callers can print it (the benchmarks do: a reader can see how
// many iterations a figure rests on).
//
// `run` may take the call index or nothing. Callers that rotate input buffers
// take it and index with it; the others keep the no-argument form. The choice
// is made at compile time, so the ones that do not rotate pay nothing and did
// not have to change. The index runs across the warm-up and the probe too: a
// probe that re-ran the buffers the warm-up just left in L2 would time a warm
// case and size the loop from it.
template <class F>
double ms_per_iter(F&& run, int* iters_out = nullptr) {
    constexpr double TARGET_MS = 200.0;
    constexpr int MIN_ITERS = 3, MAX_ITERS = 5000;

    int tick = 0;
    auto step = [&] {
        if constexpr (std::is_invocable_v<F&, int>) run(tick);
        else                                        run();
        ++tick;
    };

    cudaEvent_t s, e;
    ck(cudaEventCreate(&s), "eventCreate");
    ck(cudaEventCreate(&e), "eventCreate");

    step(); // warm-up: first launch pays module load / workspace init
    ck(cudaDeviceSynchronize(), "warm-up");

    float probe = 0.0f; // one timed iteration to size the loop
    ck(cudaEventRecord(s), "record");
    step();
    ck(cudaEventRecord(e), "record");
    ck(cudaEventSynchronize(e), "sync");
    ck(cudaEventElapsedTime(&probe, s, e), "elapsed");

    int iters = (probe > 0.0f) ? (int)(TARGET_MS / probe) : MAX_ITERS;
    if (iters < MIN_ITERS) iters = MIN_ITERS;
    if (iters > MAX_ITERS) iters = MAX_ITERS;

    float ms = 0.0f;
    ck(cudaEventRecord(s), "record");
    for (int i = 0; i < iters; ++i) step();
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

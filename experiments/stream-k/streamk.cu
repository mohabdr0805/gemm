// Stream-K on top of v7, kept out of the library. The full waves of tiles run as
// ordinary v7 blocks, and the tiles of the last wave are split in K across one
// block per SM, on a second stream so those blocks take each SM as soon as it
// frees. Checked against cuBLAS, then timed against cuBLAS and v7 on 25 sizes.
#include "../../src/gemm_cuda.cu"
#include <algorithm>
#include <limits>
#include <random>

using namespace gemm;

struct Split { float4* ws; int* flags; int epoch; int first; int tiles; int G; };

// v7 from src/gemm_cuda.cu, with an exit for the tiles left to stream-K.
template <int BM, int BN, int WM, int WN, int TM, int TN, int STAGES, int MIN_BLOCKS>
__global__ void __launch_bounds__((BM*BN)/(TM*TN), MIN_BLOCKS)
v7_tiles_kernel(int M, int N, int K, float alpha,
                const float* __restrict__ A, const float* __restrict__ B,
                float beta, float* __restrict__ C, int tiles) {
#if __CUDA_ARCH__ >= 800
    if ((int)(blockIdx.y * gridDim.x + blockIdx.x) >= tiles) return;   // left to stream-K
    constexpr int BK = 16, PAD = 4, KH = BK / 8;
    constexpr int NT     = (BM * BN) / (TM * TN);
    constexpr int PA     = BM * BK / NT;   // 4-byte copies of A per thread and tile
    constexpr int WCOLS  = BN / WN;
    constexpr int WSUB_N = WN / TN;
    constexpr int GW = 4;
    constexpr int GN = TN / GW, SPN = WN / GN;
    constexpr int GM = TM / GW, SPM = WM / GM;
    constexpr int B4 = (BK * BN / 4) / NT, BR = NT / (BN / 4);
    constexpr int AS = BM + PAD;   // +4 keeps a warp's copies on distinct banks
    static_assert(PA % KH == 0 && BK % PA == 0 && STAGES >= 2, "copy layout");
    static_assert((WM/TM) * WSUB_N == 32, "a warp must be exactly 32 threads");

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + STAGES * BK * AS;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid  = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int warpRow = (warp / WCOLS) * WM;
    const int warpCol = (warp % WCOLS) * WN;
    const int rowBase = warpRow + (lane / WSUB_N) * GW;
    const int colBase = warpCol + (lane % WSUB_N) * GW;
    const int innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);
    const int am = tid / 8, ak = tid % 8;
    const int KT = K / BK;

    const float* pA = A + (size_t)(blockRow + am) * K + ak;
    const float* pB = B + (size_t)innerRowB * N + blockCol + innerColB * 4;
    float* sA = As + ak * AS + am;
    float* sB = Bs + innerRowB * BN + innerColB * 4;
    auto copyA = [&](int s, int kt, int p) {
        cp_async4(sA + s * BK * AS + (p % KH) * 8 * AS + (p / KH) * (NT / 8),
                  pA + ((size_t)(p / KH) * (NT / 8) * K + (size_t)kt * BK + (p % KH) * 8));
    };
    auto copyB = [&](int s, int kt) {
        #pragma unroll
        for (int q = 0; q < B4; ++q)
            cp_async16(sB + s * BK * BN + q * BR * BN, pB + ((size_t)kt * BK + q * BR) * N);
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) {
            copyB(s, s);
            #pragma unroll
            for (int p = 0; p < PA; ++p) copyA(s, s, p);
        }
        cp_async_commit();
    }

    float acc[TM][TN] = {};
    int rs = 0, ws = STAGES - 1;

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        // past the last tile, the copies refill a stage nobody reads
        const int src = min(kt + STAGES - 1, KT - 1);
        const float* as = As + rs * BK * AS;
        const float* bs = Bs + rs * BK * BN;
        float fM[2][TM], fN[2][TN];
        auto frag = [&](int f, int kk) {
            #pragma unroll
            for (int g = 0; g < GM; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&as[kk*AS + rowBase + g*SPM]);
                fM[f][g*GW+0] = v.x; fM[f][g*GW+1] = v.y; fM[f][g*GW+2] = v.z; fM[f][g*GW+3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < GN; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&bs[kk*BN + colBase + g*SPN]);
                fN[f][g*GW+0] = v.x; fN[f][g*GW+1] = v.y; fN[f][g*GW+2] = v.z; fN[f][g*GW+3] = v.w;
            }
        };
        frag(0, 0);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (kk + 1 < BK) frag((kk + 1) & 1, kk + 1);
            if (kk == 0) copyB(ws, src);
            if (kk % (BK / PA) == 0) copyA(ws, src, kk / (BK / PA));
            // cuBLAS's order, pairs of rows with the columns in serpentine
            #pragma unroll
            for (int p = 0; p < TM / 2; ++p)
                #pragma unroll
                for (int jj = 0; jj < TN; ++jj) {
                    const int j  = (p & 1) ? TN - 1 - jj : jj;
                    const int i0 = 2 * p + (jj & 1), i1 = 2 * p + 1 - (jj & 1);
                    acc[i0][j] += fM[kk & 1][i0] * fN[kk & 1][j];
                    acc[i1][j] += fM[kk & 1][i1] * fN[kk & 1][j];
                }
        }
        cp_async_commit();
        rs = rs + 1 == STAGES ? 0 : rs + 1;
        ws = ws + 1 == STAGES ? 0 : ws + 1;
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int t = 0; t < TM; ++t) {
        const int gRow = blockRow + rowBase + (t / GW) * SPM + (t % GW);
        #pragma unroll
        for (int g = 0; g < GN; ++g) {
            const size_t idx = (size_t)gRow * N + blockCol + colBase + g * SPN;
            float4 out = { alpha * acc[t][g*GW+0], alpha * acc[t][g*GW+1],
                           alpha * acc[t][g*GW+2], alpha * acc[t][g*GW+3] };
            if (beta != 0.0f) {
                const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                out.x += beta * c.x; out.y += beta * c.y;
                out.z += beta * c.z; out.w += beta * c.w;
            }
            *reinterpret_cast<float4*>(&C[idx]) = out;
        }
    }
#endif
}

template <int BM, int BN, int WM, int WN, int TM, int TN, int STAGES>
__global__ void __launch_bounds__((BM*BN)/(TM*TN), 1)
streamk_kernel(int M, int N, int K, float alpha,
               const float* __restrict__ A, const float* __restrict__ B,
               float beta, float* __restrict__ C, Split sk) {
#if __CUDA_ARCH__ >= 800
    constexpr int BK = 16, PAD = 4, KH = BK / 8;
    constexpr int NT     = (BM * BN) / (TM * TN);
    constexpr int PA     = BM * BK / NT;
    constexpr int WCOLS  = BN / WN;
    constexpr int WSUB_N = WN / TN;
    constexpr int GW = 4;
    constexpr int GN = TN / GW, SPN = WN / GN;
    constexpr int GM = TM / GW, SPM = WM / GM;
    constexpr int B4 = (BK * BN / 4) / NT, BR = NT / (BN / 4);
    constexpr int AS = BM + PAD;
    constexpr int E4 = TM * TN / 4;   // float4 of accumulators per thread

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + STAGES * BK * AS;

    const int tid  = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int warpRow = (warp / WCOLS) * WM;
    const int warpCol = (warp % WCOLS) * WN;
    const int rowBase = warpRow + (lane / WSUB_N) * GW;
    const int colBase = warpCol + (lane % WSUB_N) * GW;
    const int innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);
    const int am = tid / 8, ak = tid % 8;
    const int KT = K / BK, TX = N / BN;
    const int b = blockIdx.x;
    const long long total = (long long)sk.tiles * KT;

    // one segment: tile `t`, K tiles [kb, ke)
    auto run = [&](int t, int kb, int ke) {
        const int blockRow = (t / TX) * BM, blockCol = (t % TX) * BN;
        const float* pA = A + (size_t)(blockRow + am) * K + ak;
        const float* pB = B + (size_t)innerRowB * N + blockCol + innerColB * 4;
        float* sA = As + ak * AS + am;
        float* sB = Bs + innerRowB * BN + innerColB * 4;
        auto copyA = [&](int s, int kt, int p) {
            cp_async4(sA + s * BK * AS + (p % KH) * 8 * AS + (p / KH) * (NT / 8),
                      pA + ((size_t)(p / KH) * (NT / 8) * K + (size_t)kt * BK + (p % KH) * 8));
        };
        auto copyB = [&](int s, int kt) {
            #pragma unroll
            for (int q = 0; q < B4; ++q)
                cp_async16(sB + s * BK * BN + q * BR * BN, pB + ((size_t)kt * BK + q * BR) * N);
        };

        __syncthreads();   // the previous segment is done with the stages
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            const int kt = min(kb + s, ke - 1);
            copyB(s, kt);
            #pragma unroll
            for (int p = 0; p < PA; ++p) copyA(s, kt, p);
            cp_async_commit();
        }

        float acc[TM][TN] = {};
        float fM[TM], fN[TN];
        int rs = 0, ws = STAGES - 1;
        for (int kt = kb; kt < ke; ++kt) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            const int src = min(kt + STAGES - 1, ke - 1);
            const float* as = As + rs * BK * AS;
            const float* bs = Bs + rs * BK * BN;
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                #pragma unroll
                for (int g = 0; g < GM; ++g) {
                    const float4 v = *reinterpret_cast<const float4*>(&as[kk*AS + rowBase + g*SPM]);
                    fM[g*GW+0] = v.x; fM[g*GW+1] = v.y; fM[g*GW+2] = v.z; fM[g*GW+3] = v.w;
                }
                #pragma unroll
                for (int g = 0; g < GN; ++g) {
                    const float4 v = *reinterpret_cast<const float4*>(&bs[kk*BN + colBase + g*SPN]);
                    fN[g*GW+0] = v.x; fN[g*GW+1] = v.y; fN[g*GW+2] = v.z; fN[g*GW+3] = v.w;
                }
                if (kk == 0) copyB(ws, src);
                if (kk % (BK / PA) == 0) copyA(ws, src, kk / (BK / PA));
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int jj = 0; jj < TN; ++jj) {
                        const int j = (i & 1) ? TN - 1 - jj : jj;
                        acc[i][j] += fM[i] * fN[j];
                    }
            }
            cp_async_commit();
            rs = rs + 1 == STAGES ? 0 : rs + 1;
            ws = ws + 1 == STAGES ? 0 : ws + 1;
        }
        cp_async_wait<0>();

        if (kb > 0) {
            // not the start of the tile: leave the partial to the block that owns it
            float4* dst = sk.ws + (size_t)b * E4 * NT + tid;
            #pragma unroll
            for (int q = 0; q < E4; ++q)
                __stcg(dst + q * NT, make_float4(acc[q/2][(q%2)*4+0], acc[q/2][(q%2)*4+1],
                                                 acc[q/2][(q%2)*4+2], acc[q/2][(q%2)*4+3]));
            __threadfence();
            __syncthreads();
            if (tid == 0) atomicExch(&sk.flags[b], sk.epoch);
            return;
        }
        if (ke < KT) {
            // start of the tile but not its end: add the partials of the next blocks
            const long long end = (long long)(t + 1 - sk.first) * KT;
            for (int o = b + 1; o < sk.G && total * o / sk.G < end; ++o) {
                if (tid == 0) while (atomicAdd(&sk.flags[o], 0) != sk.epoch) { }
                __syncthreads();
                __threadfence();
                const float4* src = sk.ws + (size_t)o * E4 * NT + tid;
                #pragma unroll
                for (int q = 0; q < E4; ++q) {
                    const float4 v = __ldcg(src + q * NT);
                    acc[q/2][(q%2)*4+0] += v.x; acc[q/2][(q%2)*4+1] += v.y;
                    acc[q/2][(q%2)*4+2] += v.z; acc[q/2][(q%2)*4+3] += v.w;
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int gRow = blockRow + rowBase + (r / GW) * SPM + (r % GW);
            #pragma unroll
            for (int g = 0; g < GN; ++g) {
                const size_t idx = (size_t)gRow * N + blockCol + colBase + g * SPN;
                float4 out = { alpha * acc[r][g*GW+0], alpha * acc[r][g*GW+1],
                               alpha * acc[r][g*GW+2], alpha * acc[r][g*GW+3] };
                if (beta != 0.0f) {
                    const float4 c = *reinterpret_cast<float4*>(&C[idx]);
                    out.x += beta * c.x; out.y += beta * c.y;
                    out.z += beta * c.z; out.w += beta * c.w;
                }
                *reinterpret_cast<float4*>(&C[idx]) = out;
            }
        }
    };

    int s = (int)(total * b / sk.G);
    const int e = (int)(total * (b + 1) / sk.G);
    while (s < e) {
        const int t = sk.first + s / KT, kb = s % KT, ke = min(KT, kb + (e - s));
        s += ke - kb;
        run(t, kb, ke);
    }
#endif
}

// C = alpha*A*B + beta*C on device buffers, with the v7 conditions (sm_80+,
// M % BM, N % 128, K % 16). Stream-K only when the last wave is 80% full or less.
template <int BM, int BN, int WM, int WN, int TM, int TN, int STAGES, int MIN_BLOCKS>
static void gemm_streamk(int M, int N, int K, float alpha,
                         const float* A, const float* B, float beta, float* C, int sm) {
    constexpr int NT = (BM * BN) / (TM * TN);
    constexpr int bytes = STAGES * 16 * (BM + 4 + BN) * (int)sizeof(float);
    auto kernel = streamk_kernel<BM, BN, WM, WN, TM, TN, STAGES>;
    auto ordinary = v7_tiles_kernel<BM, BN, WM, WN, TM, TN, STAGES, MIN_BLOCKS>;
    static float4* ws = nullptr;
    static int* flags = nullptr;
    static int epoch = 0;
    static cudaStream_t s2;
    static cudaEvent_t start, done;
    if (!ws) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        CUDA_CHECK(cudaFuncSetAttribute(ordinary, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        CUDA_CHECK(cudaMalloc(&ws, (size_t)sm * NT * TM * TN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&flags, sm * sizeof(int)));
        CUDA_CHECK(cudaMemset(flags, 0, sm * sizeof(int)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    }
    const int TX = N / BN, T = (M / BM) * TX, W = sm * MIN_BLOCKS;
    const int R = T >= W ? T % W : T;   // tiles of the last wave
    if (R == 0 || R > 0.8 * W) {
        launch_v7<BM, BN, WM, WN, TM, TN, STAGES, MIN_BLOCKS>(M, N, K, alpha, A, B, beta, C);
        return;
    }
    const int dp = T - R;
    const Split sk = { ws, flags, ++epoch, dp, R, sm };
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaStreamWaitEvent(s2, start, 0));
    if (dp > 0) ordinary<<<dim3(TX, (dp + TX - 1) / TX), NT, bytes>>>(M, N, K, alpha, A, B, beta, C, dp);
    kernel<<<sm, NT, bytes, s2>>>(M, N, K, alpha, A, B, beta, C, sk);
    CUDA_CHECK(cudaEventRecord(done, s2));
    CUDA_CHECK(cudaStreamWaitEvent(0, done, 0));   // the call ends when both streams have
}

static void run_v7(int n, float alpha, const float* A, const float* B, float beta, float* C,
                   int sm, int major);

// Same as v7, except that stream-K takes the 256x128 tile whenever M % 256 == 0.
static void run_streamk(int n, float alpha, const float* A, const float* B, float beta, float* C,
                        int sm, int major) {
    if (!use_v7(n, n, n, major, sm)) run_v7(n, alpha, A, B, beta, C, sm, major);
    else if (n % 256 == 0)           gemm_streamk<GEMM_V7_BIG>(n, n, n, alpha, A, B, beta, C, sm);
    else                             gemm_streamk<GEMM_V7_SMALL>(n, n, n, alpha, A, B, beta, C, sm);
}

static void run_v7(int n, float alpha, const float* A, const float* B, float beta, float* C,
                   int sm, int major) {
    const bool big = pick_big_tile(n, n, sm);
    if (!use_v7(n, n, n, major, sm)) {
        const dim3 g(n / 128, n / 128);
        if (big)                           gemm_reg_v6_kernel<GEMM_V6_BIG, 1><<<dim3(n / 128, n / 256), 256>>>(n, n, n, alpha, A, B, beta, C);
        else if ((int)(g.x * g.y) >= 2 * sm) gemm_reg_v6_kernel<GEMM_V6_SMALL, 2><<<g, 256>>>(n, n, n, alpha, A, B, beta, C);
        else                               gemm_reg_v6_kernel<GEMM_V6_SMALL, 1><<<g, 256>>>(n, n, n, alpha, A, B, beta, C);
    } else if (big) {
        launch_v7<GEMM_V7_BIG>(n, n, n, alpha, A, B, beta, C);
    } else {
        launch_v7<GEMM_V7_SMALL>(n, n, n, alpha, A, B, beta, C);
    }
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 12;
    int dev = 0, sm = 0, major = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    if (major < 8) { std::printf("needs sm_80 or newer\n"); return 1; }

    const size_t szmax = (size_t)4096 * 4096 * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, szmax)); CUDA_CHECK(cudaMalloc(&B, szmax)); CUDA_CHECK(cudaMalloc(&C, szmax));
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    const float al = 1.0f, be = 0.0f;
    std::vector<double> r7, rsk;

    std::printf("    n   v7/cuBLAS   stream-K/cuBLAS   max err\n");
    for (int n = 1024; n <= 4096; n += 128) {
        const size_t ne = (size_t)n * n;
        std::vector<float> hA(ne), hB(ne), ref(ne), out(ne);
        std::mt19937 rng(n);
        std::uniform_real_distribution<float> d(-0.5f, 0.5f);
        for (auto& x : hA) x = d(rng);
        for (auto& x : hB) x = d(rng);
        CUDA_CHECK(cudaMemcpy(A, hA.data(), ne * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, hB.data(), ne * 4, cudaMemcpyHostToDevice));
        auto cub = [&] { CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, B, n, A, n, &be, C, n)); };
        auto v7  = [&] { run_v7(n, al, A, B, be, C, sm, major); };
        auto skf = [&] { run_streamk(n, al, A, B, be, C, sm, major); };

        cub(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ref.data(), C, ne * 4, cudaMemcpyDeviceToHost));
        double err = 0.0;
        for (int rep = 0; rep < 3; ++rep) {   // repeated, so stale flags would show
            CUDA_CHECK(cudaMemset(C, 0xFF, ne * 4));   // NaN: an unwritten output fails
            skf();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(), C, ne * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ne; ++i) {
                const double e = std::fabs((double)out[i] - ref[i]);
                if (!(e <= err)) err = std::isnan(e) ? std::numeric_limits<double>::infinity() : e;
            }
        }
        if (!(err < 1e-2)) { std::printf("%5d   wrong result, max err %.2e\n", n, err); return 1; }
        if (rounds == 0) { std::printf("%5d   checked, max err %.1e\n", n, err); continue; }

        std::vector<double> a7, ask;
        for (int r = 0; r < rounds; ++r) {
            const double tc = bench::ms_per_iter(cub);
            const double t7 = bench::ms_per_iter(v7);
            const double ts = bench::ms_per_iter(skf);
            CUDA_CHECK(cudaGetLastError());
            a7.push_back(tc / t7); ask.push_back(tc / ts);
        }
        r7.push_back(median(a7)); rsk.push_back(median(ask));
        std::printf("%5d   %8.1f%%   %14.1f%%   %.1e\n", n, 100 * r7.back(), 100 * rsk.back(), err);
        std::fflush(stdout);
    }
    auto geo = [](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += std::log(x); return std::exp(s / v.size());
    };
    auto above = [](const std::vector<double>& v) { int c = 0; for (double x : v) c += x > 1.0; return c; };
    if (rounds == 0) return 0;
    std::printf("geometric mean: v7 %.1f%%, stream-K %.1f%%\n", 100 * geo(r7), 100 * geo(rsk));
    std::printf("above cuBLAS:   v7 %d of 25, stream-K %d of 25\n", above(r7), above(rsk));
    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}

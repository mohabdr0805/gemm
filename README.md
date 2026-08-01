# gemm

![CI](https://github.com/mohabdr0805/gemm/actions/workflows/ci.yml/badge.svg)

> **TL;DR**: hand-written CUDA SGEMM that beats cuBLAS SGEMM on 12–13 of 25
> tested sizes and sits at 92% of it at n=4096, through shared-memory tiling,
> register tiling, vectorized double-buffered loads, conflict-free shared reads,
> warp tiling (v6) and a second tile geometry chosen by a wave-quantization
> rule, each step picked by a Nsight Compute profile, plus a
> FlashAttention-style attention kernel that gains up to ~8.4× from query
> tiling. Every kernel is validated against a CPU oracle. Device figures are
> measured on an RTX 3080 with locked clocks, over ~200 ms of work per kernel,
> with cuBLAS timed in the same run.

Optimized GEMM (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, single precision,
row-major. The repo goes from a naive reference to tuned CPU and GPU kernels and
measures each step. The CUDA side also has a fused bias+activation epilogue, the
basic building block of an inference layer.

## Contents

- CPU: naive reference + cache-tiled, OpenMP-parallel, SIMD-vectorized version
  (~214× naive; ~200 GFLOP/s at n ≥ 2048 on a 12700F).
- GPU v1: shared-memory tiled kernel (coalesced loads, bank-conflict padding,
  border handling).
- GPU v2: register tiling, each thread computing an 8×8 micro-block of C.
  6–8.3× over v1 (grows with size), ~61–63% of cuBLAS SGEMM on the same card.
- GPU v3: v2 + vectorized `float4` loads and double buffering, each step chosen
  from a Nsight Compute profile. 76–83% of cuBLAS at n ≥ 2048, ~104% at n=1024.
- GPU v4: the same kernel pinned at 128 registers with `__launch_bounds__`,
  dispatched on grid size. +2–3 points at n ≥ 2048, and it unmasks the next
  bottleneck.
- GPU v5: conflict-free Bs reads (each thread's columns split into two groups
  half a tile apart). The ~268M bank conflicts drop to zero, measured; +8–10
  points everywhere. 90–98% of cuBLAS at n ≥ 2048, ~116% at n=1024.
- GPU v6: a warp tier between the block tile and the thread tile. Same register
  and shared budget, a third fewer shared-memory wavefronts; +4% everywhere,
  which takes the count of sizes beating cuBLAS from 5 to 12 out of 25. Ships in
  two tile geometries, 128×128 and 256×128, picked by a wave-quantization rule
  that matched the faster one on 13 of 13 sizes without timing them.
- Inference: fused GEMM + bias + activation (ReLU/GELU) in a single kernel.
- Softmax: numerically stable row-wise softmax (CPU oracle + CUDA kernel with
  shared-memory tree reductions); the attention kernels build on it.
- Attention: a FlashAttention-style kernel (online softmax, no N×N matrix) with
  an optional causal mask, a query-tiled v2 that reuses K/V across a block (up to
  ~8.4× over v1), and an FA-2-style warp-partitioned kernel that splits the head
  dimension across a warp (kills v2's d=128 register spill; 1.7–6× over v2 there).

## Results

Measured on an RTX 3080 10 GB (Ampere, `sm_86`) + i7-12700F (12C/20T), CUDA 13.0,
Windows/MSVC. Methodology: clocks locked (`nvidia-smi -lgc 1710 -lmc 9501`, the
card's spec boost — low enough that the power limiter never takes over, ~210 W
against the 370 W cap), each kernel timed over ~200 ms of work after a warm-up,
with the iteration count sized per kernel from a probe run and printed alongside
the result, and cuBLAS timed in the same run. Each figure is one ~200 ms window,
not a median of N runs; reproducibility was checked by hand (two independent runs
of the full suite agreed within 0.5 point on every ratio and 0.4% on absolute
GFLOP/s), not enforced by the harness.

The lock is not decoration; both halves of it were bought with a mistake. This
README's first numbers came from a session where the card was silently stuck in a
low memory P-state: cuBLAS measured 9.6 TFLOP/s, 32% of the card's FP32 peak,
a number no vendor SGEMM produces on healthy hardware — that is the sanity check
that caught it. And before the clocks were pinned, the v2/cuBLAS ratio drifted by
5 points between sessions, more than some of the effects being measured.

Sizing by time replaced a flat 50 iterations, which was not good enough at either
end. At n=1024 v3 runs in 0.15 ms, so 50 iterations measured 7 ms and launch
overhead left the v3/cuBLAS ratio swinging 87–113% between runs; it now sits at
~104%, stable. At n=4096 v1 runs in 100 ms, so the same 50 iterations soaked the card
for 5 s and moved the clocks under everything timed after it. Equal time per
kernel fixes both.

The inputs rotate. NVIDIA's GEMM measurement guidelines ask for enough copies of
the input to cover twice the L2, cycled through so that iteration i+1 cannot read
what iteration i left in cache; the harness sizes that count from
`cudaDeviceProp::l2CacheSize` at run time. On this card it only binds below
n=1145, since past that A and B exceed 10 MB on their own.

It changed nothing on GEMM, and that is the useful part. An alternated in-place
A/B, where the buffer index is the only difference between the two timings, puts
n=1024 inside the run-to-run spread: six rounds span −2.1% to +0.2% for v6 and
−0.0% to +1.3% for cuBLAS, with no consistent sign. A GEMM at that size moves
12 MB per launch against a 5 MB L2, so it evicts its own inputs long before the
next iteration starts. The 122% was not a caching artifact.

That claim needs the instrument to be able to detect one, so: a read-only
streaming kernel over a buffer small enough to stay L2-resident, same harness,
measures 504 GB/s with a fixed buffer and 333 GB/s rotating, 6 rounds out of 6.
Where there is carryover, the rotation finds it. Where the same test is run at
32 MB, well past the L2, it reports 1.00×, no effect, as it should.

Attention is the benchmark that was actually flattered. At n=4096, d=64 the whole
working set is 4 MB and fits in L2, and the same A/B puts the rotation at −10% on
`flash_attention_v2_kernel`, 6 rounds out of 6, at both n=1024 and n=4096. Its
table below is measured with rotation on. Softmax measured clean (0.0–0.6%) and
rotates anyway, because one protocol across the three benchmarks is worth more
than the memory it saves.

*Scope: FP32 on CUDA cores throughout (no tensor cores). The cuBLAS baseline
runs in its default math mode (plain `cublasSgemm`, no `cublasSetMathMode`), so
TF32 is disabled and the comparison is like-for-like. Cross-check: cuBLAS
matches the FP32 CPU oracle to ~7e-6, where TF32's 10-bit mantissa would give
errors around 1e-3. The baseline is cuBLAS's default heuristic kernel choice via
`cublasSgemm`, not an exhaustive `cublasLt` algorithm search; that is the field
standard for this comparison, and it is what sets the ceiling. One card
(RTX 3080, `sm_86`), one OS; the ratios can move on Ada/Hopper.*

End-to-end GFLOP/s (whole wrapper: cudaMalloc + H2D/D2H + kernel):

| n    | naive | CPU tiled (OpenMP+SIMD) | GPU v1 wrapper |
|------|-------|-------------------------|----------------|
| 1024 | 0.88  | 188                     | 618            |
| 2048 | —     | 202                     | 917            |
| 4096 | —     | 198                     | 1 162          |

*The CPU column replaces an earlier 83 / 161 / 209, under two separate causes.
(1) `gemm_tiled` used to carry `collapse(2)`, which MSVC silently ignores
(warning C4849 on every build), so only the `ii` loop was parallel; fusing the two
tile loops by hand gives +29% at n=2048, +18% at n=1024 and +6% at n=4096,
measured A/B at identical flags, bit-identical results. (2) The host benchmark now
reports the best of several runs instead of a single shot: at n=1024 the whole
GEMM takes ~20 ms, and one measurement swings by a factor of two on fork/join
alone (~97 GFLOP/s single shot against ~188 best-of-five, same binary). So most of
the 83 → 188 at n=1024 is (2), most of the 161 → 202 at n=2048 is (1), and at
n=4096 the figure moves **down**, 209 → 198: the old single shot was a lucky one.*

*Footnote: the end-to-end row is one timed call including `cudaMalloc` and both
transfers, so it measures the wrapper, not the kernel. It climbs with n while the
bare v1 kernel below is flat (~1 540 GFLOP/s), because the copies scale as n² and
the kernel as n³: at n=1024 the transfers cost more than the kernel itself (618
vs 1 515), by n=4096 they are mostly amortised away (1 162 vs 1 551).*

CPU tiled is ~214× the naive version at n=1024 (same n on both sides) and reaches
~200 GFLOP/s from n=2048 on, about 15% of the 12700F's AVX2 peak, with no cache
cliff. The
three-level blocking explains it: at any instant a thread touches only three
64×64 sub-blocks (3 × 16 KB, L1-resident at any n), so growing n only adds
bandwidth pressure from panel re-streaming: 172 GFLOP/s at n=6144, 154 at
n=8192 (about −10% once twenty threads' panel re-reads hit the shared 25 MB
L3). We initially predicted a cliff at n=8192 ("the 64·n·4 = 2 MB panel exceeds
the 1.25 MB per-core L2") and the measurement refuted it: the panel governs
inter-tile reuse, not the instantaneous working set. Single-threaded,
vectorization alone is worth 7.8× (3.8 → 29.4 GFLOP/s); see the `gemm_tiled`
section for the MSVC details.

GPU kernels vs cuBLAS SGEMM (device timing, no transfers, GFLOP/s):

| n    | v1 shared-tiled | v2 register | v3 float4+2×buf | v4 launch-bounds | v5 Bs-split | v6 warp-tiled | cuBLAS SGEMM |
|------|-----------------|-------------|-----------------|------------------|-------------|---------------|--------------|
| 1024 | 1 469           | 8 849       | 14 314          | 14 084           | 15 924      | **16 774**    | 13 732       |
| 2048 | 1 413           | 11 729      | 14 946          | 15 357           | 17 763      | **18 186**    | 19 526       |
| 3072 | 1 500           | 11 677      | 15 207          | 15 641           | 17 868      | **18 401**    | 18 279       |
| 4096 | 1 420           | 12 052      | 15 451          | 15 797           | 17 926      | **18 472**    | 19 959       |

As a share of cuBLAS:

| n    | v1  | v2  | v3   | v4   | v5   | v6       |
|------|-----|-----|------|------|------|----------|
| 1024 | 11% | 64% | 104% | 103% | 116% | **122%** |
| 2048 | 7%  | 60% | 77%  | 79%  | 91%  | **93%**  |
| 3072 | 8%  | 64% | 83%  | 86%  | 98%  | **101%** |
| 4096 | 7%  | 60% | 77%  | 79%  | 90%  | **93%**  |

Every column comes from one run, after a warm-up pass over the largest size: the
card needs several seconds of load to reach steady state, and without it the
first sizes measured come out ~15% low. Two independent runs agree within 0.4%
on every absolute and one point on every ratio. The v6 column is what the
shipped `gemm_cuda_v3` launches at that size: the kernel exists in two
`__launch_bounds__` builds and the wrapper dispatches on grid size (see GPU v4).
v1 sits at 7–11% of cuBLAS; register tiling is 6–8.5× over it, and the gap grows
with n.

Grid underfill at n=1024 is visible in v2's absolute throughput, not in its
ratio: 128×128 tiles produce 64 blocks, and v2 fits 2 per SM, so it has 136 slots
for the 3080's 68 SMs and fills fewer than half of them (wave quantization):
8 849 GFLOP/s against ~12 000 at larger n. The ratio hides it because cuBLAS
underfills at n=1024 too (13 732, its own worst size).

v3 beats cuBLAS at n=1024 for the same reason it costs at n=4096: it needs 130
registers, so only one block fits per SM (see below), which means 68 slots, and
n=1024 supplies 64 blocks. The property that halves its occupancy is what leaves
it nearly full when the grid is small.

**Four sizes are not a benchmark.** Those four are the round numbers everyone
quotes, and they hide how much the ratio moves with shape, so the aligned range
was swept in steps of 128 from n=1024 to 4096, 25 sizes, medians of three
alternated measurements. **v6 is above cuBLAS on 12 of the 25, against 5 for
v5.** The curve oscillates because both kernels quantize into waves and they do
not quantize the same way: v6's own throughput is nearly flat (16.7–19.4
TFLOP/s), while cuBLAS swings from 13.7 to 21.0 with peaks at the powers of two,
where it evidently ships a size-specific kernel. So the losses cluster at 2048
and 4096, and the wins where cuBLAS has nothing special. One size is genuinely
bad on our side: n=1152 sits at 76%, its 81 blocks leaving the card 60% filled
whichever `__launch_bounds__` build runs. `ctest` checks every kernel, cuBLAS
included, against the CPU oracle (max error ~1e-5, tolerance 1e-3). The
benchmark does not check correctness.

## CPU — `gemm_tiled`

C is split into 64×64 tiles to fit in cache, and one whole tile goes to one
thread, so there is no data race and no reduction. The inner i-k-j order keeps B
and C accessed row by row.

The two tile loops are **fused by hand** into a single loop over the tile index
instead of carrying `collapse(2)`. MSVC needs `/openmp:experimental` for the
`omp simd` below; that mode is OpenMP 2.0 + simd and it ignores `collapse`, saying
so on every build (`warning C4849`). Only the `ii` loop was parallel: M/64
iterations to fill 20 threads. The binary also linked `VCOMP140.DLL`, the
OpenMP 2.0 runtime, which is the same fact from the other side.
`/openmp:llvm` honours `collapse` but rejects `omp simd` (`error C7660`), so no
single MSVC flag buys both. Fusing needs no clause at all, works on every
compiler, and gives +29% at n=2048 (see the table footnote above).

Row access is only half the point of that loop order; the other half is SIMD.
The inner `Crow[j] += a * Brow[j]` over contiguous floats maps onto 8-wide AVX2
FMAs, worth 7.8× single-threaded (3.8 → 29.4 GFLOP/s). MSVC needed three
explicit steps to emit them: `/arch:AVX2` (x64 defaults to 4-wide SSE2; the
flag is guarded by `GEMM_NATIVE` since AVX2 binaries crash on pre-2013 CPUs),
`#pragma omp simd` + `__restrict` (MSVC refuses to auto-vectorize
`C[i*N+j] += a*B[k*N+j]` because it assumes C may alias B, `/Qvec-report`
reason 1200; GCC emits a runtime overlap check instead), and hoisted row
pointers (the vectorizer also rejects the mixed `(size_t)i*N + j` addressing).
`/Qvec-report:2` confirms: `C5001: omp simd loop vectorized`.

## GPU v1 — `gemm_kernel`

16×16 tile per block, with A and B staged in shared memory so global traffic drops
by roughly a factor of TILE. `threadIdx.x` indexes the column, so a warp reads
contiguous addresses (coalesced). Sizes that aren't multiples of 16 are guarded,
and the shared tiles are padded to `[TILE][TILE+1]` to avoid bank conflicts.

## GPU v2 — `gemm_reg_kernel`

128×128 block tile, K stepped in chunks of 8, 256 threads. Each thread keeps an
8×8 micro-block of C in registers, so every shared-memory load feeds 8 FMAs (an
8×8 outer product is 64 FMAs for 16 loads). That raises arithmetic intensity
(see below).

Why these numbers: a 128×128 block tile with a K-step of 8 needs 2 × (128×8)
floats of shared memory = 8 KB per block. 256 threads each owning an 8×8
micro-block cover the tile exactly (256 × 64 = 128×128). At 128 registers per
thread, one block uses 32 K of the SM's 64 K register file, so exactly two
blocks fit per SM. The 33% occupancy below is deliberate: the extra registers
raise arithmetic intensity, which is worth more than occupancy at this point.

## Occupancy vs arithmetic intensity

`ptxas` on `sm_86` (256-thread blocks; an `sm_86` SM runs up to 1536 threads):

| kernel | registers/thread | shared/block | occupancy          | GFLOP/s (n=4096) |
|--------|------------------|--------------|--------------------|------------------|
| v1     | 37 (0 spill)     | 2176 B       | 100% (6 blocks/SM) | 1 551            |
| v2     | 128 (0 spill)    | 8192 B       | 33% (2 blocks/SM)  | 12 865           |
| v3     | 130 (0 spill)    | 16384 B      | 17% (1 block/SM)   | 16 283           |
| v4     | 128 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 15 797           |
| v5     | 128 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 17 926           |
| v6     | 126 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 18 472           |

Occupancy falls 100% → 33% → 17% down the first three rows while throughput
rises 1 420 → 12 052 → 15 451. v1 sits at 100% theoretical occupancy yet runs
8.5× slower than v2 at a third of the occupancy: one output per thread gives too
little reuse per load, so v1 is limited by arithmetic intensity, not occupancy.
Past full occupancy, arithmetic intensity is the only lever left; past that,
latency hiding, which is what v3 buys with the rest of its occupancy.

The v3 row also shows how sharp the cliff is. Two blocks per SM need
`regs × 256 × 2 ≤ 65536`, so the limit is 128 registers/thread: v2 sits exactly
on it, and v3's prefetch pushes it to 130. Two registers over, and the second
block is gone. The 16 KB of shared is not what binds (it would still allow six).
The v4 row is the same kernel forced back under the cliff with
`__launch_bounds__`; worth ~3% at this size (see GPU v4). The v5 row lands on
128 without being asked: its split addressing is
cheaper than v3's, so the kernel sits on the cliff naturally.

**The geometry is measured, not assumed.** The 128×128×8 block tile with an 8×8
micro-block was chosen at v2 and never re-examined, so the kernel was
parametrized on `<BM, BN, BK, TM, TN>` and twelve valid configurations were
swept at n=4096, each validated against the CPU oracle before being timed. None
beat the original: 128×64 reaches 80% of cuBLAS, 64×128 76%, and the 512-thread
configurations that trade arithmetic intensity for occupancy land at 71%, the
worst of the set. Larger tiles do have the better ratio on paper (256×128×8
loads 3072 floats for 262144 FMAs, 85 per float against 64) and still lose,
because the register budget bites before the intensity pays. 256×256 with a
16×16 micro-block is the limit case: `acc` alone needs 256 registers against the
255 available, and it runs 23× slower. The parametrized kernel is not shipped,
since it found nothing faster than what is already here; the sweep is the
result.

## GPU v3 — `gemm_reg_v3db_kernel` (float4 + double buffering)

v2 sits at ~61% of cuBLAS. Nsight Compute on the v2 kernel at n=4096 (full grid)
showed where the rest goes: DRAM at only ~11% (so not bandwidth-bound), and the
top warp stalls were global-load latency (`long_scoreboard`) and shared-load
bank conflicts, with ~268M shared-load bank conflicts reported. Two changes
followed, each checked by re-profiling.

First, vectorized `float4` global loads plus a transposed As tile (so the
compute loop reads contiguous shared rows). This was worth several points of
cuBLAS, but the re-profile was a useful correction: the bank-conflict count did
not move (the dominant conflict is on the Bs read, which this did not touch), so
the gain came from issuing a quarter as many global-load instructions, not from
the conflicts. It also left global-load latency as the new top stall
(`long_scoreboard` rose from ~12% to ~16%).

Second, double buffering. Two shared buffers: each step issues the next tile's
global loads into registers up front (in flight while the current tile is
computed), then lands them in the other buffer and swaps, so the latency
overlaps the FMAs instead of stalling in front of them. The prediction was that
`long_scoreboard` would collapse and large-n throughput would recover, and both
held: the stall fell from ~16% to ~2%, and v3 reaches 76–83% of cuBLAS at
n ≥ 2048 (77% at n=4096, 76% at n=2048, 83% at n=3072). At n=1024 it passes
cuBLAS outright, ~104%, for the occupancy reason given in Results: one block per
SM needs only 64 of the card's 68 slots to fill it.

| stall (n=4096, ncu)        | v2    | v3 float4 | v3 + double-buf | v4    |
|----------------------------|-------|-----------|-----------------|-------|
| global latency (long_sb)   | 11.8% | 15.5%     | 2.3%            | 0.1%  |
| shared latency (short_sb)  | 2.9%  | 18.0%     | 24.3%           | 11.9% |
| barrier                    | 9.8%  | 9.3%      | 4.9%            | 11.0% |
| mio throttle (LSU/bank)    | 7.4%  | 7.1%      | 5.5%            | 17.4% |

Double buffering also trades occupancy for latency hiding, and `ptxas` prices the
trade exactly: 130 registers against v2's 128. The cliff for two blocks per SM is
128, so v3 drops to one block and occupancy halves (32% → 17%). The doubled
shared tile is not what binds; 16 KB/block would still allow six. It wins anyway,
because hiding the global latency is worth more than the lost occupancy (the
v1→v2 lesson again). And it moves the bottleneck: with `long_scoreboard` gone the
top stall is `short_scoreboard`, the latency of the shared→register reads
themselves, which the halved occupancy no longer covers. Winning those two
registers back is v4 (next section).

The fast path assumes aligned sizes (M,N % 128 == 0, K % 8 == 0, needed for the
unguarded float4 loads); any other shape falls back to v2. `benchmark_gemm_versions`
prints v1, v2, v3 (float4), v3 (double-buffered), v4 and cuBLAS.

## GPU v4 — `__launch_bounds__` and the two registers

v4 is one attribute: `__launch_bounds__(256, 2)` on the double-buffered kernel,
compiled as a template on the occupancy target so the benchmark can time both
builds. Told to fit two blocks per SM, `ptxas` lands exactly on the 128-register
cliff with zero spill, and occupancy doubles (17% → 33%).

It does what it was aimed at: `short_scoreboard` halves (24.3% → 11.9%) and
`long_scoreboard` all but vanishes (2.3% → 0.1%). Yet the net is +2–3 points of
cuBLAS at n ≥ 2048, not more, and the stall table says why: `mio_throttle`
triples (5.5% → 17.4%) and takes over as the top stall. Two blocks per SM put
twice the warps on one LSU pipe, and that pipe still carries the ~268M Bs bank
conflicts untouched since v2. The conflicts were free while latency dominated;
removing the latency is what finally sends the bill.

At n=1024 v4 *loses* ~2 points, consistently. Squeezing into 128 registers is
not free even with zero spill (`ptxas` recomputes addresses it would otherwise
keep live), and that cost is paid per thread whether or not the second block
ever lands. A 64-block grid on 68 SMs runs one block per SM regardless, so v4
pays without collecting; the first guess was that the extra occupancy would
be harmless there, and the measurement said otherwise. So `gemm_cuda_v3`
dispatches on grid size: the 128-register build once the grid can seat two
blocks per SM, the 130-register build below that. Same kernel, two `ptxas`
budgets, picked per shape — the FA-2 dispatch argument again, one level down.

Next lever, and the profile had now named it twice: the Bs bank conflicts.
That is v5, below.

## GPU v5 — `gemm_reg_v5_kernel` (conflict-free Bs reads)

The conflicts have an address. In the compute loop each thread read its 8 output
columns as one contiguous group at `threadCol * 8`: sixteen threadCols spaced 8
floats apart land on `(8·tc) mod 32 ∈ {0, 8, 16, 24}` — four of the 32 banks, a
4-way conflict on every Bs read. A back-of-envelope check says these are the
conflicts ncu has been counting all along: the kernel issues 67.1M Bs load
instructions at n=4096, and the counter read 268 435 456, exactly 4× that.

The As read never conflicts, which is why the v3 transpose could not move the
counter: `threadRow` takes only 2 values per warp, so the As read is a
broadcast. The fix therefore only touches Bs ownership: each thread's 8 columns
split into two groups of 4, at `threadCol*4` and `threadCol*4 + 64`. A warp's
sixteen threadCols then read 128 contiguous floats — all 32 banks. Staging,
tile layout and the FMA loop are untouched; the epilogue moves with the
ownership.

The prediction, written down before running: the counter drops to zero exactly.
Measured (ncu, n=4096, kernels confirmed by name):

| ncu, n=4096              | v4          | v5      |
|--------------------------|-------------|---------|
| bank conflicts (LD)      | 264 503 296 | **0**   |
| kernel time              | 7.34 ms     | 6.38 ms |
| `sm__throughput`         | 65.6%       | 75.4%   |
| `mio_throttle`           | 17.4%       | 4.7%    |
| `short_scoreboard`       | 12.0%       | 6.5%    |

Zero, not "near zero". On the bench it is the largest single step since v2:
+8–10 points of cuBLAS at every size (v4 → v5: 79→89 at n=2048, 86→96 at
n=3072, 80→87 at n=4096, 102→119 at n=1024), reproduced across two runs within
±0.6 point. v4's bill is paid: the pressure that doubling the warps put on the
LSU pipe stopped hurting once the pipe stopped replaying conflicts.

Two smaller facts the measurement adds. The split addressing costs 128
registers where v3's cost 130, so v5 sits on the 2-blocks/SM cliff without
`__launch_bounds__` — yet the hint still changes scheduling enough to measure,
~1 point up at n ≥ 2048 and ~3.5 points down at n=1024, so the grid dispatch
stays. And the staging side keeps a residual 16.5M store conflicts (the
transposed As scatter), two orders of magnitude below where the load conflicts
were; with `mio_throttle` at 4.7% the profile now points at `barrier` (8.7%)
and the remaining shared latency, which v6 goes after.

## GPU v6 — `gemm_reg_v6_kernel` (warp tiling)

Up to v5 the kernel has two tiers, the block tile and the thread tile, and
nothing in between. That leaves the warp shape to fall out of the thread
indexing: `threadRow = tid/16` takes two values across 32 consecutive lanes and
`threadCol = tid%16` takes sixteen, so a warp covers a 16×128 sliver. Its 32
lanes then read 2 distinct rows of As, which broadcasts for free, and 16 distinct
column groups of Bs, which does not. v5 removed the bank conflicts on that read
but not its width.

v6 gives each warp an explicit 128×16 tile, which flips the ratio: 16 distinct As
rows and 2 distinct Bs column groups. The instruction count does not move
(`smsp__inst_executed_op_shared_ld` is identical to the byte), but the accesses
coalesce into a third fewer wavefronts, 402M down to 268M, or 3.0 per shared-load
instruction down to 2.0. That is the whole mechanism, and it is worth about 4% at
every size.

The elongated warp tile was not the prediction. A square 64×32 looked right,
since it minimises the total distinct addresses a warp touches; it measures at
89%, barely above v5. The count that matters is per tile, not overall, and As
was already free.

Giving As 16 distinct rows also broke it: 16 rows 8 floats apart land on 4 banks,
a 4-way conflict, 134M of them where v5 had none. The fix is the one v5 already
uses on Bs: read in groups of 4, the float4 the hardware serves in one pass,
spaced across the warp tile so a warp's lanes cover a contiguous run. Applying it
to As as well brought the counter back to zero and took the gain from ~3% to
~5%.

| ncu, n=4096                | v5          | v6 first cut | v6          |
|----------------------------|-------------|--------------|-------------|
| bank conflicts (LD)        | 0           | 134 217 728  | **0**       |
| shared-load wavefronts     | 402 656 753 | 402 657 199  | **268 438 126** |
| per shared-load instruction| 3.00        | 3.00         | **2.00**    |
| `mio_throttle`             | 4.7%        | 4.5%         | **2.7%**    |
| `short_scoreboard`         | 6.5%        | 4.6%         | **3.4%**    |

Counters, so the ratio can be reproduced: `l1tex__data_pipe_lsu_wavefronts_mem_
shared_op_ld.sum` over `smsp__inst_executed_op_shared_ld.sum`. Dividing instead
by `smsp__inst_executed_pipe_lsu.sum` gives 2.46 and 1.64, and that is the wrong
denominator: it counts global traffic too, against a numerator that is shared
only. The ×1.5 is the same either way.

It costs nothing to get there: 126 registers against v5's 128, the same 16 KB of
shared, the same 33% occupancy. The group addressing computes one base per thread
and constant offsets from it, which is cheaper than v5's `threadRow*TM + i`, so
the warp tier actually hands two registers back.

One measurement note worth keeping. Nsight Compute reports v6 and v5 within 0.3%
of each other, while the benchmark shows 5%. Both are right about different
things: ncu profiles one launch in isolation, at base clocks, with caches
flushed, and the benchmark runs two dozen back to back. The structural counters
above are what ncu is for; its wall time, for this kernel, is not.

## Where the remaining variation comes from

v6 swings between 12 500 and 20 500 GFLOP/s across the sweep, and the low points
read as a weak kernel. They are not. The whole curve is wave quantization, and
the quantum is the SM rather than the 2-blocks/SM wave: 68 blocks run at a time,
so the makespan is `ceil(blocks/68)` block-times while the work is `blocks/68`.

    efficiency = (blocks/68) / ceil(blocks/68),  blocks = (M/128)·(N/128)
    throughput = efficiency × 20 900 GFLOP/s

| n    | blocks | blocks/68 | efficiency | predicted | measured |
|------|--------|-----------|------------|-----------|----------|
| 1152 | 81     | 1.19 → 2  | 59.6%      | 12 450    | 12 489   |
| 1280 | 100    | 1.47 → 2  | 73.5%      | 15 370    | 15 358   |
| 1536 | 144    | 2.12 → 3  | 70.6%      | 14 760    | 14 860   |
| 1792 | 196    | 2.88 → 3  | 96.1%      | 20 090    | 20 027   |
| 2560 | 400    | 5.88 → 6  | 98.0%      | 20 490    | 20 477   |
| 4096 | 1024   | 15.06 → 16| 94.1%      | 19 670    | 20 224   |

One parameter, 13 of the 14 sizes tested inside 3%. The miss is n=1024, measured
7.6% under: 64 blocks for 68 SMs never loads the card two-deep at all, which is
outside what the model describes.

So n=1152 at 76% of cuBLAS is not an inefficient kernel. It is 1.19 blocks per SM
paid at the price of 2. Its Nsight profile agrees: the FMA pipe drops to 68% and
active warps to 21%, but `long_scoreboard` sits at 0.09% and `not_selected` at
49%, which says the scheduler has more eligible warps than issue slots. There is
no latency being missed. Nothing done to the inner loop can move that size.

### The fix exists, was built, and is not here

The standard answer is Stream-K: count the work in K-loop iterations rather than
output tiles, launch a fixed number of blocks, and give each an equal slice, so a
slice may start mid-tile and end mid-tile. Partial tiles go to a workspace and a
second kernel recombines them. It is implemented and validated against cuBLAS on
11 shapes, including a single tile split across 128 blocks.

It works, at n=1152: 12 483 → 15 286 GFLOP/s, 76% of cuBLAS to 93%.

It also costs 19 registers. Letting a block span several output tiles keeps about
a dozen values live across the whole compute loop, and the kernel goes from v6's
126 registers to 147. v6 had two registers of headroom, so there is no version
that keeps both: capped at 128 for two blocks per SM it spills 76 bytes, and at
147 with no spill it runs one block per SM. Either way the large sizes lose ~43%.

That trade is why Stream-K wins exactly where it does. At a trough the card is
underfilled, so dropping to one block per SM costs nothing, because there were
never enough blocks to place two. Everywhere else it costs half the occupancy.

A Split-K variant, which keeps one tile per block and only parameterises the K
range, fits in 128 registers with no spill. It gets less: it writes a partial for
every tile, where Stream-K only writes them at the seams. At n=1152 with S=5 that
is 405 partials, 53 MB of workspace traffic round trip, against a GEMM that runs
in 0.24 ms.

Taking the best of the three at every size moves n=1152 from 76% to 93%, n=1280
from 92% to 97%, n=1536 from 94% to 98%, and nothing anywhere else: only 3 of the
25 swept sizes have an efficiency below 75%, and past that `ceil` stops mattering.
None of the three crosses 100%, so the count of sizes beating cuBLAS stays at 12
of 25. Against that, two more kernels, a device workspace to size and own, and a
public API that has to carry it. The measurement is worth more here than the
kernels, so the kernels stayed out.

The projection that started this was wrong, and by a lot: dividing the current
figure by the efficiency predicted 76% → 127% and 22 of 25 sizes. That ceiling
assumes quantization can be removed at unchanged occupancy, and neither approach
does both. Measured: +22.5% at the one size, and the same 12 of 25.

### A second tile, chosen by the same model

The model earns its keep somewhere else. Profiling cuBLAS names its kernel,
`cutlass_80_simt_sgemm_256x128_8x4_nn_align1`, and the name is the whole
configuration: a 256×128 block tile, 16×8 outputs per thread, 202 registers, one
block per SM. That is the opposite of v4's choice, which spent
`__launch_bounds__(256, 2)` to keep two blocks resident.

The reason it wins is arithmetic intensity, at two levels at once. Per k-step a
16×8 thread reads 16 A values and 8 of B, six 128-bit shared loads, and does 128
FMAs: 21.3 FMAs per shared load against 8×8's 16. Measured shared-load
instructions at n=4096 drop from 134 217 728 to 100 663 296, a quarter fewer,
which is 134/101 = 1.33 against the predicted 21.33/16 = 1.33.

The same doubling of the M tile halves the global traffic, because A's panel is
read once for twice as many output rows. Measured at n=4096:

| n=4096          | DRAM traffic | % of DRAM peak | L2 sectors    |
|-----------------|--------------|----------------|---------------|
| 128×128         | 1.47 GB      | 31.8%          | 136 703 329   |
| **256×128**     | **732 MB**   | **15.8%**      | **103 272 762** |
| cuBLAS          | 534 MB       | 12.4%          | 103 280 500   |

Half the DRAM traffic, and the same L2 traffic as cuBLAS to within 8 000 sectors
out of 103 million. Note what this rules out as well: at 16 to 32% of DRAM peak
neither kernel is bandwidth-bound, so the gap to cuBLAS is not a memory-traffic
problem. The L2 absorbs two thirds of the panel re-reads, 4.4 GB of L2 traffic
against 1.47 GB reaching DRAM.

That geometry had already been swept, and lost at 86% of cuBLAS. The sweep ran it
on the v5 body, with no warp tier. On the warp-tiled body it reaches 94%: the
warp tier is worth 7.6 points here against 4.9% on 128×128. A negative autotune
result holds for the body it was measured on, not for the geometry.

Both tiles ship, and the dispatch is the quantization model again. The 256×128
grid is half the size, so it lands differently on the SM count: it wins whenever
it quantizes at least as well, since at equal efficiency the intensity is free,
and loses when its grid rounds up worse. That rule picked the faster kernel on 13
of the 13 sizes where both apply, without timing anything.

| n    | efficiency 128×128 | efficiency 256×128 | picked  | measured |
|------|--------------------|--------------------|---------|----------|
| 1024 | 94%                | 47%                | 128×128 | −45% if forced |
| 1536 | 71%                | 53%                | 128×128 | −24% if forced |
| 2048 | 94%                | 94%                | 256×128 | +1.6%    |
| 2560 | 98%                | 98%                | 256×128 | +0.4%    |
| 3328 | 99%                | 99%                | 256×128 | +4.0%    |
| 3584 | 96%                | 96%                | 256×128 | +4.2%    |
| 3840 | 95%                | 95%                | 256×128 | +2.9%    |
| 4096 | 94%                | 94%                | 256×128 | +2.5%    |

It needs `M % 256 == 0`, so 13 of the 25 swept sizes can use it and the rest fall
back. It never loses a size, and it gains one: 12 of 25 sizes beat cuBLAS with
the 128×128 tile alone, 13 with the dispatch. At n=4096 an alternated A/B over
six rounds puts it between +1.4% and +5.2%, median +2.9%.

Two caveats on those counts. They move by one between runs, because a handful of
sizes sit within a point of the line and the ratio at n=1920 has swung 95% to
113% across sessions. And the absolute GFLOP/s in the tables above predate this
work: the machine measured ~18% low across every kernel including cuBLAS while
this was being tuned, with the core provably healthy (a pure-FMA kernel with no
memory access reached 97% of the card's FP32 peak in the same state). Ratios
survive that, since cuBLAS is timed in the same run and moves with it; absolutes
do not, so they were left alone rather than refreshed from a degraded reading.

What it does not do is close the gap at n=4096, and the counters say why. With
the bigger tile the instruction count is no longer the problem: 2.331 G
instructions against cuBLAS's 2.399 G, the same shared-load count, zero bank
conflicts, and a lower barrier stall (3.1% against 4.5%). The issue rate is what
differs, 79.6% against 87.8%, and it is all in `short_scoreboard` plus `wait`:
10.3% here against 4.5% there. Both run one block per SM, so it is not occupancy.

Three attempts on that, all refuted. Double-buffering the shared loads in
registers across the k-step changed nothing and left the register count at 206,
which is the proof that ptxas already scheduled it that way. `cp.async` on the B
tile cost 6.5%: it removes a fifth of the shared-store wavefronts, and triples
`mio_throttle` from 1.5% to 4.7%, because the copy holds a queue slot for the
whole compute phase. Deepening the pipeline to three and four stages made that
worse, not better.

The reason is visible in cuBLAS's counters. It issues 25 313 280 `cp.async`
instructions and 8 192 ordinary global loads; this kernel issues none and 8.4 M.
It stages *both* operands asynchronously. Doing that here means the A tile can no
longer be transposed on the way into shared memory, and the transpose is what
makes its read a vectorized one: with 32 lanes four rows apart, a flat layout
puts every lane on the same bank, and no padding fixes it while keeping 16-byte
alignment. So the last 6% is a different shared-memory addressing scheme, not a
missing instruction.

A linear layer computes `Y = act(X·W + bias)`. Naively that is two kernels: the
GEMM, then an element-wise pass for bias and activation, which writes the output
and reads it all back. The fused kernel does the bias and activation in the GEMM
epilogue, before the first write: one launch, one global-memory pass. cuBLASLt,
CUTLASS and TensorRT do the same.

The activation is a template parameter (resolved at compile time, no branch), and
its math (ReLU, tanh-GELU) lives in `activation.hpp`, shared between the CPU oracle
and the GPU kernel so the two cannot disagree.

Fusion saves one M×N pass plus a launch, so the relative gain scales with how
much of the total work that pass represents: it grows as K shrinks and as the
problem gets small, and vanishes on a large square GEMM. On the RTX 3080 (GELU),
clocks locked, rotated inputs, two runs agreeing within 0.01:

| shape | speedup |
|-------|---------|
| 1024³, 2048³, 4096³ | 1.00–1.01× |
| 1024²·64 | 1.08× |
| 2048²·64 | 1.13× |
| 4096²·64 | 1.12× |
| 512³ | 1.02× |
| 256³ | 1.11× |

These replace an earlier 0.97–1.16× / 0.91× / 1.35× / 1.17× / "~1.5–2× on tiny
matrices", measured before the clocks were locked and before the harness sized
its windows by time. Every one of them was off, and the 0.91× was off in sign:
the thin-K case is a gain, not a loss. The shape of the effect survives, the
amplitude does not. Fusion here is a consistent ~1.1× where the saved pass is a
real fraction of the work, and nothing at all where it is not.

One caveat: the epilogue is currently fused into the v1 tiling, while the
register-tiled v2 is ~10× faster as a plain GEMM. Templating the epilogue into
`gemm_reg_kernel` is the next step (see Roadmap).

## Softmax — `softmax_rows_kernel`

Row-wise safe softmax (`out[i,:] = softmax(in[i,:])`, subtracting the row max
before `exp` so nothing overflows): one block per row, two cooperative tree
reductions in shared memory (row max, then sum of exp), threads striding over
rows longer than the block. Memory-bound, so the benchmark reports effective
bandwidth: ~470 GB/s at 1024² (the 4 MB matrix mostly lives in the 3080's 5 MB
L2), dropping to ~290 GB/s at 4096² once the matrix is DRAM-resident. The card's
DRAM peak is 760 GB/s, and the kernel's real traffic is ~2.5× the algorithmic
figure. The kernel exists mostly as a stepping stone to the attention kernels
below, which reuse its reduction idiom and replace its global softmax with an
online one.

### The online (2-pass) variant, and why its gain hides

`softmax_rows_online_kernel` fuses the max pass into the sum pass with the same
`exp(m_old − m_new)` rescale the attention kernels use, so the input is read
twice instead of three times: 3·M·N accesses against 5, a ceiling of 5/3 ≈ 1.67×.
The reduction folds `(m, l)` pairs in one tree instead of two.

Measured, it delivers, but only past a threshold:

| n × n  | re-read working set | 3-pass  | online  | speedup |
|--------|---------------------|---------|---------|---------|
| 2 048  | 3.2 MB (< L2)       | 357 GB/s| 385 GB/s| 1.08×   |
| 4 096  | 6.5 MB              | 191     | 282     | 1.48×   |
| 8 192  | 13 MB               | 158     | 267     | **1.69×** |
| 16 384 | 26 MB               | 160     | 266     | 1.66×   |

What has to overflow the 3080's 5 MB L2 is not the matrix but what is re-read
*concurrently*: ~408 resident blocks (6/SM × 68 SM) × one n-float row, so the
crossover sits at n\* ≈ 5 MB / (408 × 4 B) ≈ 3 200. Below it, every re-read is an
L2 hit and both kernels pay only the incompressible 2·M·N of DRAM traffic (the
first read of `x`, the write of `y`), so they are indistinguishable and the 5/3
model predicts a gain that does not exist. Above it the model becomes exact.

Two things this cost. The first draft folded the 256 partials on **one** thread
instead of a tree: correct, 40% less traffic, and *slower than the 3-pass*
(0.71× at 2048², 0.41× at 2048×1024). The serial fold is 255 of the ~271
sequential steps on a block's critical path, ~94%, and it does not shrink with n,
so halving n makes the penalty worse. And the benchmark itself used to sample only
`(n, n)` and `(n, 1024)`, both below n\* at the usual n, which made a 1.69× kernel
read as 0.98× and nearly got it written off. Less algorithmic traffic only becomes
speed if the traffic you removed was not already being served by cache; it is also
why attention, which never fits, wins unconditionally.

## FlashAttention v1 — `flash_attention_kernel`

Attention is `O = softmax(scale · Q·Kᵀ) · V`. Done literally, that builds the
`N×N` score matrix `S`, softmaxes it, then multiplies by `V`: three passes and
`O(N²)` memory, which blows up on long sequences. FlashAttention never writes
`S`: it streams the keys and keeps a running softmax, so scores, softmax and
the `P·V` product all happen in one fused pass with `O(N·d)` extra memory.

The trick is the online softmax. Walking the keys in tiles, each query row
carries three running quantities: the max score `m`, the normalizer
`l = Σ exp(sⱼ − m)`, and the unnormalized output `acc = Σ exp(sⱼ − m)·vⱼ`. When
a new tile raises the max from `m` to `m'`, the old state is rescaled by
`exp(m − m')` before the tile is folded in, and `O = acc / l` once at the end.
The rescale is what keeps the result identical to a global safe softmax while
only ever seeing one tile at a time.

The layout mirrors the softmax kernel: one block per query row. The block loads
its query once, then for each key tile stages `K` and `V` in shared memory,
computes the tile's scores (one thread per key), reduces the tile max and sum
with the same tree reductions as `softmax_rows_kernel`, updates `(m, l, acc)`,
and moves on. The shared `Ks`/`Vs` tiles are padded to `[BC][D+1]` against bank
conflicts, like the GEMM tiles. The causal mask costs nothing extra: row `i`
stops its key loop at `i+1`, so tiles past the diagonal are never loaded.

Because each block reads all of `K` and `V` once, v1's global traffic is
`O(M·N·d)` with no `N×N` allocation. That is the central FlashAttention
property, and also v1's ceiling: at long sequence length it re-reads `K`/`V`
`M` times.

## Attention v2 — query tiling (`flash_attention_v2_kernel`)

*Naming note: "v2" is this repo's own v1→v2 progression (like the GEMM
kernels), not the FlashAttention-2 algorithm. FA-2's warp-partitioned layout,
where lanes cooperate on the head dimension, is a further step (see Roadmap).*

Same online softmax, but a block now owns a tile of `BR` query rows instead of
one, so each streamed `K`/`V` tile is read once and reused by all `BR` rows:
`K`/`V` traffic drops from `M` reads to `M/BR`. It is the same v1→v2 move as
the GEMM kernels (shared tile → register tile): each thread owns one query row
and keeps its running `(m, l)` and accumulator `acc[d]` in registers, so rows
are independent and there are no cross-thread reductions. The head dimension is
a template parameter (dispatched at runtime for `d ∈ {32, 64, 128}`, like the
fused GEMM's activation) so the inner loops fully unroll; any other `d` falls
back to v1.

`q[d]`/`acc[d]` stay register-resident up to d = 64: `ptxas -v` reports 128
registers at d=32 and 218 at d=64, both with 0 spill. At d = 128 the arrays
exceed the 255-register/thread limit and spill (`ptxas -v`: 1 124 B of spill
stores and 3 100 B of spill loads per thread). v2 still wins
there, because the `K`/`V` reuse more than pays for the spill traffic, but by a
smaller margin.

Sizing the iteration count by time matters for the ratio here, not just the
absolute: under the flat 50 iterations this table used to run, v1 (20 ms/iter at
n=4096) soaked the card for a second before v2 was timed, and the speedup came
out ~11× against the ~8.4× measured now, with clocks locked, over equal ~200 ms
windows. "Timed in the same power state" was the claim; the fixed count was
quietly breaking it. On an RTX 3080 (`sm_86`), `n×n`, full attention, device
timing:

| n    | speedup d=64 | speedup d=128 |
|------|--------------|---------------|
| 1024 | ~2.3×        | ~1.1×         |
| 2048 | ~4.4×        | ~2.1×         |
| 4096 | ~8.4×        | ~4.0×         |

Rotated inputs here too, and the rotation costs the d=64 column more than the
d=128 one (9.2× to 8.4× at n=4096, against 3.8× to 4.0×). That asymmetry is the
same story as the table: v2's whole advantage at d=64 is K/V reuse, so it gains
most from a resident L2 and loses most when the inputs stop being resident. At
d=128 it spills to local memory and the L2 is no longer what limits it.

The gain grows with n: the longer the sequence, the more each cached `K`/`V`
tile is reused. d=128 gains less than d=64 because of the register spill. Since
this is a single head, the benchmark only exposes `M/BR` blocks, so at small n
(especially causal, which halves the work) v2 can be block-starved and v1 wins;
the crossover is around n = 1024. A real workload adds batch·heads more blocks
and v2 wins across the board.

`./build/bench n` prints v1 vs v2 for both table dims (d = 64 and d = 128),
full and causal.

## Attention FA-2 — warp-partitioned head dim (`flash_attention_fa2_kernel`)

v2 gives a query row to one thread, which holds `q[d]` and `acc[d]` in registers.
At d=128 that is 256+ floats per thread, over the 255-register limit, so it
spills (ptxas: 255 registers, 1 124 B stored and 3 100 B loaded back per
thread). FA-2 gives a
row to a whole warp instead and splits d across its 32 lanes: lane `l` owns dims
l, l+32, ..., so at d=128 each lane holds only d/32 = 4 elements of q and acc.
ptxas then reports 64 registers and 0 spill.

Splitting d breaks one thing: the score `s = <q, k>` is a sum over d, now spread
across the lanes. Each lane computes its partial dot, and a warp butterfly
(`__shfl_xor`, registers only) sums the 32 partials into the full score, present
in every lane. That is the only cross-lane communication. The running softmax
scalars (m, l) are per-row, so every lane recomputes them identically; the P·V
product is lane-independent, each lane accumulating its own output dims. The one
reduction stays in registers, so it never touches shared memory or a block sync.

Measured on an RTX 3080, `n×n`, full attention, GFLOP/s:

| n    | v2 d=128 | FA-2 d=128 | FA-2/v2 | v2 d=64 | FA-2 d=64 | FA-2/v2 |
|------|----------|------------|---------|---------|-----------|---------|
| 1024 | 238      | 1474       | 6.20×   | 435     | 1040      | 2.39×   |
| 2048 | 479      | 1350       | 2.82×   | 888     | 1289      | 1.45×   |
| 4096 | 955      | 1557       | 1.63×   | 1772    | 1305      | 0.74×   |

Rotated inputs, median of three runs from separate processes that agreed within
0.5%. The rotation costs ~10% here, which is the point of it: at these sizes Q, K
and V fit in L2 and a fixed-buffer loop was timing a cache. That 10% is the A/B
figure, where the buffer index is the only thing that differs. It is not the gap
between this table and the one it replaces: that older table also predates the
warm-up pass, and some of its entries were too low rather than too high.

At d=128 FA-2 wins at every size: no spill, and a flat ~1350–1550 GFLOP/s
regardless of n. At d=64, where v2 does not spill, it is a trade-off. FA-2 gives
a row to a warp, so a block serves only 8 rows against v2's 64, and each shared
K/V tile is reused 8× per block instead of 64×. At large n the K/V traffic
dominates, so v2's higher reuse wins (0.74× at n=4096); at small n v2 is
block-starved (16 blocks for 68 SMs, wave quantization) while FA-2's smaller row
tile fills the card, so FA-2 wins. The right kernel depends on the regime: FA-2
for d=128, v2 for d≤64 at long sequence length.

## Build & run

CPU (Linux/GCC, or Windows/MSVC. On MSVC the build forces `/openmp:experimental`
for one reason: it is the only mode that accepts the `omp simd` directive the
tiled GEMM's inner loop needs to vectorize. It does not support `collapse`, so the
tile loops are fused by hand instead; see the `gemm_tiled` section):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 1024
```
on Windows, run from a "x64 Native Tools Command Prompt", or install CMake standalone.

With CUDA (NVIDIA GPU + nvcc):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 2048
```
Default target architectures are `75;86` (Turing + Ampere); pass
`-DCMAKE_CUDA_ARCHITECTURES=89` (Ada, etc.) to target your own card.

## Docker

```bash
# CPU image: builds, runs the test, then the benchmark
docker build -t gemm-cpu .
docker run --rm gemm-cpu 2048

# CUDA image: compiles the GPU kernels (running them needs an NVIDIA GPU)
docker build -f Dockerfile.cuda -t gemm-cuda .
docker run --rm --gpus all gemm-cuda 2048
```
Images build with `-DGEMM_NATIVE=OFF` (no `-march=native`) so they run on any
x86-64 host. The CPU image is published on each tagged release to
`ghcr.io/mohabdr0805/gemm`.

## CI / CD

- CI (`.github/workflows/ci.yml`): on every push/PR, builds and tests the CPU code,
  compile-checks the CUDA build, and builds both Docker images (CPU and CUDA). The
  runners have no GPU, so the CUDA kernels are compiled but not run.
- CD (`.github/workflows/release.yml`): on a `v*` tag, builds the CPU image and
  pushes it to GHCR. There is no service to deploy here; the artifact is the image.

## Layout

```
include/gemm/   headers (gemm_cpu.hpp, gemm_cuda.cuh, activation.hpp,
                softmax*.hpp/cuh, attention*.hpp/cuh)
src/            gemm_cpu.cpp (naive + tiled), gemm_cuda.cu (kernels),
                softmax_{cpu,cuda}, attention_{cpu,cuda}
benchmarks/     GFLOP/s, v1/v2/v3 vs cuBLAS, fusion, softmax, attention benchmarks
tests/          correctness vs the CPU oracle (gemm, softmax, attention)
```

## Roadmap

- [x] CPU: naive + OpenMP tiled
- [x] GPU v1: shared-memory tiled
- [x] GPU v2: register tiling (8×8 per thread)
- [x] Fused inference epilogue (bias + activation)
- [x] Docker + CI/CD (GitHub Actions, image on GHCR)
- [x] Row-wise softmax kernel (safe softmax: CPU oracle + CUDA)
- [x] Softmax: online (2-pass) variant: 1.69× over the 3-pass kernel above the L2
      crossover (n\* ≈ 3 200), indistinguishable below it
- [x] Fused attention kernel (FlashAttention-style, online softmax + causal mask)
- [x] Attention v2: query tiling (K/V reused across the block, up to ~8.4× over v1)
- [x] Baseline: cuBLAS SGEMM, same card, same run (v2 ≈ 61–63%)
- [ ] Baseline: PyTorch SDPA for the attention kernels
- [x] GEMM v3: vectorized `float4` loads + double buffering (~61% → 76–83% of cuBLAS,
      ~104% at n=1024)
- [x] GEMM v4: `__launch_bounds__(256,2)` pins ptxas on the 128-register cliff,
      0 spill, occupancy 17% → 33% (+2–3 points at n ≥ 2048; loses 2 on underfilled
      grids, so the wrapper dispatches on grid size)
- [x] GEMM v5: conflict-free Bs reads (split column ownership; the ~268M conflicts
      measured at exactly zero). +8–10 points everywhere: 90–98% of cuBLAS at
      n ≥ 2048, ~116% at n=1024
- [x] GEMM v6: warp tiling (a 128×16 warp tier; a third fewer shared-load
      wavefronts at no register cost). +4% everywhere: 93% of cuBLAS at n=4096,
      and above it on 12 of 25 swept sizes against 5 for v5
- [ ] GEMM v7: the remaining stalls are `barrier` (8.9%) and shared latency.
      `cp.async` is the usual answer and does not apply directly. The As tile is
      stored transposed, which needs the register round-trip that `cp.async`
      exists to remove. Warp specialization or a split arrive/wait barrier first
- [x] Second tile geometry (256×128, 16×8 per thread, one block/SM), dispatched
      on wave-quantization efficiency: +3% at n=4096, one more size over cuBLAS
- [~] Stream-K: built and validated, not shipped. Recovers the quantization
      (n=1152 goes 76% → 93% of cuBLAS) but costs 19 registers, so the large
      sizes lose ~43%. Full measurement above, under "Where the remaining
      variation comes from"
- [~] `cp.async`: refuted on this addressing scheme. It only pays if it stages
      both operands, and the A tile's transpose blocks that (see the same
      section). Measured −6.5%, and worse with a deeper pipeline
- [ ] Fused epilogue on the register-tiled v2 GEMM
- [x] Attention FA-2: warp-partitioned head dim (kills the d=128 spill; 1.7–6× over v2 at d=128)
- [ ] Multi-device StarPU variant

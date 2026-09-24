# GEMM, step by step

[← README](../README.md)

## Results in detail

End-to-end GFLOP/s (whole wrapper: cudaMalloc + H2D/D2H + kernel):

| n    | naive | CPU tiled (OpenMP+SIMD) | GPU v1 wrapper |
|------|-------|-------------------------|----------------|
| 1024 | 0.88  | 188                     | 618            |
| 2048 | —     | 202                     | 917            |
| 4096 | —     | 198                     | 1 162          |

*The CPU figures are best-of-five, not single-shot: at n=1024 the whole GEMM takes
~20 ms, and a single measurement swings by a factor of two on fork/join alone
(~97 GFLOP/s single shot against ~188 best-of-five, same binary).*

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
ratio: 128×128 tiles produce 64 blocks for 68 SMs, so every SM gets one and none
gets the second its register budget allows. Half the resident warps, 8 849
GFLOP/s against ~12 000 at larger n. That is a residency loss, not the wave
quantization modelled in [wave-quantization.md](wave-quantization.md), which at 64 blocks over 68 SMs would cost 6% and
not 26%. The ratio hides it because n=1024 is cuBLAS's worst size too (13 732),
though not for the same reason: it dispatches to its smallest tile there,
`ampere_sgemm_64x64_nn`. Its 256 blocks quantize identically to our 64, 94.1%
either way, so what separates them is traffic: a 64×64 tile reads twice the
bytes for the same FMAs.

v3 beats cuBLAS at n=1024 for the same reason it costs at n=4096: 130 registers
hold it to one block per SM. Where v2 loses half its residency at that size, v3
was already at one block and loses nothing.

**Four sizes are not a benchmark.** Those four are the round numbers everyone
quotes, and they hide how much the ratio moves with shape, so the aligned range
was swept in steps of 128 from n=1024 to 4096, 25 sizes, medians of three
alternated measurements. **With the 128×128 tile alone, v6 is above cuBLAS on 12
of the 25, against 5 for v5**, and on 13 once the second geometry ships. The
curve oscillates because both kernels quantize into waves and they do not
quantize the same way. Ours is nearly flat, 16.7–19.4 TFLOP/s over the 25
shapes; cuBLAS covers 13.7 to 21.0, swinging with which of its kernels each
shape selects, six different ones across the sweep. Our biggest losses are
against its 256×128 tile. One size is genuinely bad on our side: n=1152 sits at
78%, its 81 blocks leaving the card 60% filled whichever `__launch_bounds__`
build runs. `ctest` checks every kernel, cuBLAS included, against the CPU oracle
(max error ~1e-5, tolerance 1e-3). The benchmark does not check correctness.

## CPU — `gemm_tiled`

C is split into 64×64 tiles to fit in cache, and one whole tile goes to one
thread, so there is no data race and no reduction. The inner i-k-j order keeps B
and C accessed row by row.

The two tile loops are **fused by hand** into a single loop over the tile index
rather than carrying `collapse(2)`: the only MSVC OpenMP mode that accepts the
`omp simd` below ignores `collapse`, leaving just the `ii` loop parallel, 32 tile
rows for 20 threads at n=2048. Fusing needs no clause, runs on every compiler,
and is worth +29% there, measured A/B at identical flags with bit-identical
results.

Row access is only half the point of that loop order; the other half is SIMD.
The inner `Crow[j] += a * Brow[j]` over contiguous floats maps onto 8-wide AVX2
FMAs, worth 7.8× single-threaded (3.8 → 29.4 GFLOP/s). Getting MSVC to emit them
took `/arch:AVX2` and a rewrite: it assumes C may alias B and gives up rather
than emit the runtime overlap check GCC uses, so the row pointers are hoisted and
marked `__restrict` under `#pragma omp simd`.

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
| v1     | 37 (0 spill)     | 2176 B       | 100% (6 blocks/SM) | 1 420            |
| v2     | 128 (0 spill)    | 8192 B       | 33% (2 blocks/SM)  | 12 052           |
| v3     | 130 (0 spill)    | 16384 B      | 17% (1 block/SM)   | 15 451           |
| v4     | 128 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 15 797           |
| v5     | 128 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 17 926           |
| v6     | 128 (0 spill)    | 16384 B      | 33% (2 blocks/SM)  | 18 472           |

Occupancy falls 100% → 33% → 17% down the first three rows while throughput
rises 1 420 → 12 052 → 15 451. v1 sits at 100% theoretical occupancy yet runs
8.5× slower than v2 at a third of the occupancy: one output per thread gives too
little reuse per load, so v1 is limited by arithmetic intensity, not occupancy.
v3 then traded some occupancy again, this time for latency hiding.

The v3 row also shows how sharp the cliff is. Two blocks per SM need
`regs × 256 × 2 ≤ 65536`, so the limit is 128 registers/thread: v2 sits exactly
on it, and v3's prefetch pushes it to 130. Two registers over, and the second
block is gone. The 16 KB of shared is not what binds (it would still allow six).
The v4 row is the same kernel forced back under the cliff with
`__launch_bounds__`; worth ~3% at this size (see GPU v4). The v5 row lands on
128 without being asked: its split addressing is
cheaper than v3's, so the kernel sits on the cliff naturally.

**The geometry is measured, not assumed.** The 128×128×8 block tile with an 8×8
micro-block wins a sweep of twelve valid `<BM, BN, BK, TM, TN>` configurations at
n=4096, each validated against the CPU oracle before being timed: 128×64 reaches
80% of cuBLAS, 64×128 76%, and the 512-thread configurations that trade
arithmetic intensity for occupancy land at 71%, the worst of the set. Larger
tiles do have the better ratio on paper (256×128×8 loads 3072 floats for 262144
FMAs, 85 per float against 64) and still lose, because the register budget bites
before the intensity pays. 256×256 with a 16×16 micro-block is the limit case:
`acc` alone needs 256 registers against the 255 available, and it runs 23× slower.

## GPU v3 — `gemm_reg_v3db_kernel` (float4 + double buffering)

v2 sits at ~61% of cuBLAS. Nsight Compute on the v2 kernel at n=4096 (full grid)
showed where the rest goes: DRAM at only ~11% (so not bandwidth-bound), and the
top warp stalls were global-load latency (`long_scoreboard`) and shared-load
bank conflicts, with ~268M shared-load bank conflicts reported. Two changes
followed, each checked by re-profiling.

First, vectorized `float4` global loads plus a transposed As tile. This was
worth several points of cuBLAS, but the re-profile was a useful correction: the
bank-conflict count did not move (the dominant conflict is on the Bs read, which
this did not touch), so the gain came from issuing a quarter as many global-load
instructions, not from the conflicts. The transpose was meant to fix those
conflicts and missed; it stays because it makes the compute loop's A read
contiguous: two `LDS.128` instead of eight scalar loads. The change also left
global-load latency as the new top stall (`long_scoreboard` rose from ~12% to
~16%).

Second, double buffering. Two shared buffers: each step issues the next tile's
global loads into registers up front (in flight while the current tile is
computed), then lands them in the other buffer and swaps, so the latency
overlaps the FMAs instead of stalling in front of them. The prediction was that
`long_scoreboard` would collapse and large-n throughput would recover, and both
held: the stall fell from ~16% to ~2%, and v3 reaches 76–83% of cuBLAS at
n ≥ 2048 (77% at n=4096, 76% at n=2048, 83% at n=3072). At n=1024 it passes
cuBLAS outright, ~104%, for the occupancy reason given above: one block per
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
`long_scoreboard` falls from 2.3% to 0.1%. Yet the net is +2–3 points of
cuBLAS at n ≥ 2048, not more, and the stall table says why: `mio_throttle`
triples (5.5% → 17.4%) and takes over as the top stall. Two blocks per SM put
twice the warps on one LSU pipe, and that pipe still carries the ~268M Bs bank
conflicts untouched since v2. The conflicts were free while latency dominated;
removing the latency is what finally sends the bill.

At n=1024 v4 *loses* ~2 points, consistently. Squeezing into 128 registers is
not free even with zero spill (`ptxas` recomputes addresses it would otherwise
keep live), and that cost is paid per thread whether or not the second block
ever lands. A 64-block grid on 68 SMs runs one block per SM regardless, so v4
pays without collecting. `gemm_cuda_v3` therefore dispatches on grid size: the
128-register build once the grid can seat two blocks per SM, the 130-register
build below that, the same per-shape dispatch as FA-2 one level down.

Next lever, and the profile had now named it twice: the Bs bank conflicts.
That is v5, below.

## GPU v5 — `gemm_reg_v5_kernel` (conflict-free Bs reads)

The conflict is in the Bs read, where each thread took its 8 output columns as
one contiguous group at `threadCol * 8`: sixteen threadCols spaced 8 floats
apart land on `(8·tc) mod 32 ∈ {0, 8, 16, 24}`, four of the 32 banks. The count
matches exactly: the kernel issues 67.1M Bs load instructions at n=4096 and the
counter reads 268 435 456, four conflicts per instruction, so these are all of
them and not most of them.

The As read never conflicts, which is why the v3 transpose could not move the
counter: `threadRow` takes only 2 values per warp, so the As read is a
broadcast. The fix therefore only touches Bs ownership: each thread's 8 columns
split into two groups of 4, at `threadCol*4` and `threadCol*4 + 64`. A warp's
sixteen threadCols then read 128 contiguous floats, all 32 banks. Staging,
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

What "zero" means is worth stating, because it is narrower than it sounds and
the difference is what v6 later collects. Isolating this exact read outside the
GEMM, one `LDS.128` per warp: the strided version costs 8 wavefronts per
instruction, the split version 4. So the counter did not remove the
serialization, it removed the part of it that the memory layout was causing. The
counter reports wavefronts **above the floor**, not the total, which is why 4
remain at zero conflicts. Checked against four strides predicted in advance: at
stride 16 it reports 12 conflicts and 16 wavefronts, at stride 32 it reports 28
and 32, and all four are exact.

Those remaining 4 wavefronts come from how many distinct addresses a warp asks
for, not from banks, and no relayout can reach them. Changing *which lane reads
what* can: v6 does exactly that and takes the same read from 4 wavefronts to 2,
without `bank_conflicts` moving off zero. That is why the v6 gain is invisible on
this counter, and it is the subject of the next section.

Zero, not "near zero". On the bench it is the largest single step since v2: +10
points of cuBLAS at n=2048 and n=3072, +7 at n=4096 and +17 at n=1024 (v4 → v5:
79→89, 86→96, 80→87, 102→119), reproduced across two runs within ±0.6 point.
v4's bill is paid: the pressure that doubling the warps put on the LSU pipe
stopped hurting once the pipe stopped replaying conflicts.

Two smaller facts the measurement adds. The split addressing costs 128 registers
where v3's cost 130, so v5 sits on the 2-blocks/SM cliff without
`__launch_bounds__`, yet the hint still changes the code ptxas emits: both
builds hold 128 registers, 0 spill and the same 576 FFMA, but one is 968
instructions against 984 and assigns registers differently throughout. Worth ~1
point of cuBLAS at n ≥ 2048 and 2 to 3.5 the other way at n=1024, so the grid
dispatch stays. And the staging side keeps a residual 16.5M store conflicts (the
transposed As scatter), two orders of magnitude below where the load conflicts
were; with `mio_throttle` at 4.7% the profile now points at `barrier` (8.7%) and
the remaining shared latency, which v6 goes after.

## GPU v6 — `gemm_reg_v6_kernel` (warp tiling)

Up to v5 the kernel has two tiers, the block tile and the thread tile, and
nothing in between. That leaves the warp shape to fall out of the thread
indexing: `threadRow = tid/16` takes two values across 32 consecutive lanes and
`threadCol = tid%16` takes sixteen, so a warp covers a 16×128 strip. Its 32
lanes then read 2 distinct rows of As, which broadcasts for free, and 16
distinct column groups of Bs, which does not. v5 removed the bank conflicts on
that read but not its width.

v6 gives each warp an explicit 128×16 tile, which flips the ratio: 16 distinct
As rows and 2 distinct Bs column groups. The instruction count does not move
(`smsp__inst_executed_op_shared_ld` is identical to the byte), but the accesses
coalesce into a third fewer wavefronts, 402M down to 268M, or 3.0 per
shared-load instruction down to 2.0. It is worth about 4% on average over the
sweep.

The elongated warp tile was not the prediction. A square 64×32 looked right,
since it minimises the total distinct addresses a warp touches; it measures at
89%, barely above v5. The count that matters is per tile, not overall, and As
was already free.

Giving As 16 distinct rows also broke it: 16 rows 8 floats apart land on 4
banks, and the counter reports 134M conflicts where v5 had none. The fix is the
one v5 already uses on Bs: read in groups of 4, the float4 the hardware serves
in one pass, spaced across the warp tile so a warp's lanes cover a contiguous
run. Applying it to As as well brought the counter back to zero and took the
gain at n=4096 from +3% to +5%, six alternated rounds each way.

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

The disassembly agrees, which is the useful check on that ratio. `cuobjdump
--dump-sass` on the `sm_86` cubin shows 32 `LDS.128` in both v5 and v6, and no
narrower shared load in either: the instruction mix is identical, so the whole
v6 gain is wavefronts per instruction and not instructions. It also confirms a
claim made earlier from `ptxas` behaviour alone: the shared reads are fused into
128-bit loads.

It costs nothing to get there: the same 128 registers as v5, the same 16 KB of
shared, the same 33% occupancy. That 128 is a ceiling rather than a round
number: 65 536 registers per SM over 512 threads is exactly 128 each, so one
more would drop v6 to one block per SM the way v3's 130 does.

Nsight Compute reports v6 and v5 within 0.3% of each other where the benchmark
shows 5%: it profiles one launch in isolation, at base clocks, with caches
flushed, while the benchmark runs two dozen back to back. Its counters are what
the section above uses, not its wall time.

## Inference — `gemm_bias_act_kernel` (fused bias + activation)

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

Fusion here is a consistent ~1.1× where the saved pass is a real fraction of the
work, and nothing at all where it is not.

One caveat: the epilogue is currently fused into the v1 tiling, while the
register-tiled v2 is ~10× faster as a plain GEMM. Templating the epilogue into
`gemm_reg_kernel` is the next step (see the roadmap below).

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
      measured at exactly zero). +7 to +17 points depending on the size: 90–98% of cuBLAS at
      n ≥ 2048, ~116% at n=1024
- [x] GEMM v6: warp tiling (a 128×16 warp tier; a third fewer shared-load
      wavefronts at no register cost). +4% on average: 93% of cuBLAS at n=4096,
      and above it on 12 of 25 swept sizes against 5 for v5
- [ ] GEMM v7: the remaining stalls are `barrier` (8.9%) and shared latency.
      `cp.async` is the usual answer and does not apply directly. The As tile is
      stored transposed, which needs the register round-trip that `cp.async`
      exists to remove. Warp specialization or a split arrive/wait barrier first
- [x] Second tile geometry (256×128, 16×8 per thread, one block/SM), dispatched
      on wave-quantization efficiency: +3% at n=4096, one more size over cuBLAS
- [~] Stream-K: built and validated, not shipped. Recovers the quantization
      (n=1152 goes 78% → 93% of cuBLAS) but costs 19 registers, so the large
      sizes lose ~43%. Full measurement in
      [wave-quantization.md](wave-quantization.md)
- [~] `cp.async`: refuted on this addressing scheme. It only pays if it stages
      both operands, and the A tile's transpose blocks that (see
      [wave-quantization.md](wave-quantization.md)). Measured −6.5%, and worse with a deeper pipeline
- [ ] Fused epilogue on the register-tiled v2 GEMM
- [x] Attention FA-2: warp-partitioned head dim (kills the d=128 spill; 1.6–6.2× over v2 at d=128)
- [ ] Multi-device StarPU variant

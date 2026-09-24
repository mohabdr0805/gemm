# Where the remaining variation comes from

[← README](../README.md)

v6 swings between 12 500 and 20 500 GFLOP/s across the sweep, and the low points
read as a weak kernel. The whole curve is wave quantization, and the quantum is
the SM rather than the 2-blocks/SM wave: 68 blocks run at a time, so the
makespan is `ceil(blocks/68)` block-times while the work is `blocks/68`.

    efficiency = (blocks/68) / ceil(blocks/68),  blocks = (M/128)·(N/128)
    throughput = efficiency × plateau

The plateau is the model's only free parameter: what the kernel reaches when the
grid divides evenly and quantization costs nothing. Least squares over the 25, a
line through the origin, puts it at 21 029 GFLOP/s. Seven of the 25:

| n    | blocks | blocks/68 | efficiency | predicted | measured |
|------|--------|-----------|------------|-----------|----------|
| 1024 | 64     | 0.94 → 1  | 94.1%      | 19 792    | 18 229   |
| 1152 | 81     | 1.19 → 2  | 59.6%      | 12 525    | 12 495   |
| 1536 | 144    | 2.12 → 3  | 70.6%      | 14 844    | 14 862   |
| 1792 | 196    | 2.88 → 3  | 96.1%      | 20 205    | 19 921   |
| 2560 | 400    | 5.88 → 6  | 98.0%      | 20 617    | 20 458   |
| 3328 | 676    | 9.94 → 10 | 99.4%      | 20 906    | 20 654   |
| 4096 | 1024   | 15.06 → 16| 94.1%      | 19 792    | 20 024   |

Across the 25, 24 land inside 3%, mean error 1.1%, median 0.8%. The one miss is
n=1024, and it marks the model's domain. It has the same efficiency as n=4096 and
the same prediction, yet lands 8.6% off where n=4096 is 1.2%: 64 blocks for 68
SMs means every SM gets at most one, so the card is never loaded two blocks deep,
which is what the formula assumes. Naming that limit is worth more than adding a
parameter to hide it.

So n=1152 at 78% of cuBLAS is not an inefficient kernel. It is 1.19 blocks per
SM paid at the price of 2. Its Nsight profile agrees: the FMA pipe drops to 68%
and active warps to 21%, but `long_scoreboard` sits at 0.09% and `not_selected`
at 49%, which says the scheduler has more eligible warps than issue slots, so no
latency is being missed and nothing done to the inner loop can move that size.

## The fix exists, was built, and is not here

The standard answer is Stream-K: count the work in K-loop iterations rather than
output tiles, launch a fixed number of blocks, and give each an equal slice, so a
slice may start mid-tile and end mid-tile. Partial tiles go to a workspace and a
second kernel recombines them. Implemented here and validated against cuBLAS on
11 shapes, including a single tile split across 128 blocks.

It buys the troughs: n=1152 goes from 78% of cuBLAS to 93%. It costs 19
registers, because a block spanning several output tiles keeps about a dozen
values live across the whole compute loop, and v6 had two of headroom. Capped at
128 for two blocks per SM it spills 76 bytes; at 147 with no spill it runs one
block per SM. Either way the large sizes lose ~43%. The troughs do not, because a
card that was already underfilled had no second block to lose.

Split-K keeps one tile per block and only parameterises the K range, so it fits
in 128 registers with no spill. It buys less: a partial for every tile where
Stream-K writes them only at the seams, 405 partials and 53 MB of workspace
traffic at n=1152, against a GEMM that runs in 0.24 ms.

Best-of-three moves n=1152 to 93%, n=1280 to 97% and n=1536 to 98%, and nothing
else, since only those three sit below 75% efficiency. None crosses 100%, so the
count stays at 12 of 25. Against that, two more kernels and a device workspace
every caller would have to allocate, since the API takes pointers and nothing
else. So neither ships.

## A second tile, chosen by the same model

Profiling cuBLAS names its kernel,
`cutlass_80_simt_sgemm_256x128_8x4_nn_align1`, and the name gives its block tile
and its pipeline depth: 256×128, a K-step of 8, four stages. The profile gives
the rest: 16×8 outputs per thread, 202 registers, one block per SM. That is the
opposite of v4's choice, which spent `__launch_bounds__(256, 2)` to keep two
blocks resident.

The reason it wins is arithmetic intensity, at two levels at once. Per k-step a
16×8 thread reads 16 A values and 8 of B, six 128-bit shared loads, and does 128
FMAs: 21.3 FMAs per shared-load instruction against 8×8's 16. Measured
shared-load instructions at n=4096 drop from 134 217 728 to 100 663 296, the
1.33 that ratio predicts.

The same doubling of the M tile halves the global traffic, because A's panel is
read once for twice as many output rows. Measured at n=4096:

| n=4096          | DRAM traffic | % of DRAM peak | L2 sectors    |
|-----------------|--------------|----------------|---------------|
| 128×128         | 1.47 GB      | 31.8%          | 136 703 329   |
| **256×128**     | **732 MB**   | **15.8%**      | **103 272 762** |
| cuBLAS          | 534 MB       | 12.4%          | 103 280 500   |

The L2 traffic matches cuBLAS to within 8 000 sectors out of 103 million, and at
16 to 32% of DRAM peak neither kernel is bandwidth-bound, so the gap to cuBLAS
is not a memory-traffic problem.

That geometry had already been swept, and lost at 86% of cuBLAS. The sweep had
compiled it into v5's inner loop, so it is the loop that changed, not the tile.
On the warp-tiled loop it reaches 94%: the warp tier is worth 7.6 points here
against 4.9% on 128×128.

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

It needs `M % 256 == 0`, so 13 of the 25 swept sizes can use it and the rest
fall back. Over the sweep:

|                     | geometric mean | median | range      | above cuBLAS |
|---------------------|----------------|--------|------------|--------------|
| 128×128 alone       | 99.1%          | 98.5%  | 78.1–122.0%| 12 of 25     |
| **with the dispatch** | **99.8%**    | 100.6% | 78.1–122.0%| **13 of 25** |

The size it takes across 100% is n=3840. At n=4096 an alternated A/B over six
rounds puts the gain between +1.4% and +5.2%, median +2.9%.

The geometric mean is the honest one to quote here: the arithmetic mean reads
100.1%, which would license "beats cuBLAS on average", but that is the 122% at
n=1024 doing the work, and averaging ratios arithmetically overweights the wins.
99.8% is parity, slightly under. The spread matters more than either: a factor
1.5 separates the ends, and the 78% is n=1152, which the model above accounts
for. And the domain is square aligned shapes; outside `M % 128 == 0`,
`N % 128 == 0`, `K % 8 == 0` the wrapper falls back to the register-tiled v2 at
around 61% of cuBLAS.

The counts move by one between runs, several sizes sitting within a point of the
line. And the absolute GFLOP/s outside the quantization table were measured
while the machine ran ~18% low, the core provably healthy at 97% of FP32 peak on
a pure-FMA kernel in the same state. Ratios survive that since cuBLAS is timed
in the same run; absolutes do not, so they were left rather than refreshed from
a degraded reading.

What it does not do is close the gap at n=4096, and the counters say why. With
the bigger tile the instruction count is no longer the problem: 2.331 G
instructions against cuBLAS's 2.399 G, the same shared-load count, zero bank
conflicts, and a lower barrier stall (3.1% against 4.5%). The issue rate is what
differs, 79.6% against 87.8%, and it is all in `short_scoreboard` plus `wait`:
10.3% here against 4.5% there. Both run one block per SM, so it is not occupancy.

Three attempts on that, all refuted. Double-buffering the shared loads in
registers across the k-step changed nothing and left the big tile's register
count at 206, which is the proof that ptxas already scheduled it that way.
`cp.async` on the B tile cost 6.5%: it removes a fifth of the shared-store
wavefronts, and triples `mio_throttle` from 1.5% to 4.7%, because the copy holds
a queue slot for the whole compute phase. Deepening the buffering to three and
four stages made that worse, not better, though cuBLAS runs four.

The reason is visible in cuBLAS's counters. It issues 25 313 280 `cp.async`
instructions and 8 192 ordinary global loads; this kernel issues none and 8.4 M.
It stages *both* operands asynchronously. Doing that here means the A tile can no
longer be transposed on the way into shared memory, and the transpose is what
makes its read a vectorized one: with 32 lanes four rows apart, a flat layout
puts every lane on the same bank, and no padding fixes it while keeping 16-byte
alignment. So the last 6% is a different shared-memory addressing scheme, not a
missing instruction.

What the transpose costs is legible in the disassembly rather than inferred.
Each thread stages four floats of A and four of B, and `cuobjdump --dump-sass`
on v6 shows 2 `STS.128` against 8 plain `STS` per kernel, double-buffered, so
one vectorized store per B tile and four scalar ones per A tile. B is contiguous
on the way in and A is not, which is the same fact that stops `cp.async` from
carrying it: the instruction copies bytes and cannot scatter them.

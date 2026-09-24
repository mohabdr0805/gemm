# Softmax and attention

[← README](../README.md)

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

That re-read is the ceiling usually quoted, but two more show up in `ptxas -v`
before the kernel runs.

**Shared sized for the worst case.** `d` is a runtime value, so `Ks`/`Vs` are
declared `[BC][ATTN_DMAX+1]` and the kernel reserves 34 312 B per block whatever
`d` is. v2 and FA-2 are templated on `d` and take only what they need:

| `d` | v1 | v2 | FA-2 |
|-----|----|----|------|
| 32  | **34 312 B** | 8 448 B  | 8 192 B  |
| 64  | **34 312 B** | 16 640 B | 16 384 B |
| 128 | **34 312 B** | 33 024 B | 32 768 B |

At `d=128` all three are comparable, so the problem is not that v1 is greedy, it
is that v1 is flat: at `d=32` it reserves four times v2's shared for the same
work. On the RTX 3080 that caps it at two blocks per SM, 17% occupancy, at every
`d`. Shared is what binds, not registers: 40 registers per thread is 5 120 per
block, and the 64 K register file would allow twelve.

**Three warps in four idle on the expensive phase.** Scores are one thread per
key (`if (tid < tile)`, `tile ≤ BC = 32`), so 32 of the block's 128 threads run
the `d`-long dot product. That is not divergence: 32 threads is exactly one
warp, so warps 1 to 3 retire the branch and wait at the barrier with no
intra-warp serialization. Divergence costs execution slots, idleness costs
scheduling slots, and at 17% occupancy those eight resident warps are what would
otherwise hide the re-read above.

Neither is worth fixing in v1, and both are gone by FA-2 (shared templated on
`d`, one row per warp). They are recorded because the re-read is only part of
the answer.

## Attention v2 — query tiling (`flash_attention_v2_kernel`)

*Naming note: "v2" is this repo's own v1→v2 progression (like the GEMM
kernels), not the FlashAttention-2 algorithm. FA-2's warp-partitioned layout,
where lanes cooperate on the head dimension, is the next section.*

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

On an RTX 3080 (`sm_86`), `n×n`, full attention, device timing:

| n    | speedup d=64 | speedup d=128 |
|------|--------------|---------------|
| 1024 | ~2.3×        | ~1.1×         |
| 2048 | ~4.4×        | ~2.1×         |
| 4096 | ~8.4×        | ~4.0×         |

Rotated inputs here too, and an A/B with the buffer index as the only difference
puts the cost at 6.4% of the d=64 speedup at n=4096 and nothing at d=128 (9.25×
to 8.66×, against 4.45× to 4.49×). That asymmetry is the same story as the
table: v2's whole advantage at d=64 is K/V reuse, so it gains most from a
resident L2 and loses most when the inputs stop being resident. At d=128 the
working set is 8 MB against a 5 MB L2, so the fixed-buffer loop was never timing
a cache and there is nothing for the rotation to take away.

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
0.5%. The rotation costs ~10% here, which is the point of it: at these sizes Q,
K and V fit in L2 and a fixed-buffer loop was timing a cache. That 10% is the
A/B figure, where the buffer index is the only thing that differs.

At d=128 FA-2 wins at every size: no spill, and a flat ~1350–1550 GFLOP/s
regardless of n. At d=64, where v2 does not spill, it is a trade-off. FA-2 gives
a row to a warp, so a block serves only 8 rows against v2's 64, and each shared
K/V tile is reused 8× per block instead of 64×. At large n the K/V traffic
dominates, so v2's higher reuse wins (0.74× at n=4096); at small n v2 is
block-starved (16 blocks for 68 SMs, wave quantization) while FA-2's smaller row
tile fills the card, so FA-2 wins. The right kernel depends on the regime: FA-2
for d=128, v2 for d≤64 at long sequence length.

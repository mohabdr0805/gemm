# Methodology

[← README](../README.md)

Measured on an RTX 3080 10 GB (Ampere, `sm_86`) + i7-12700F (12C/20T), CUDA 13.0,
Windows/MSVC. Methodology: clocks locked (`nvidia-smi -lgc 1710 -lmc 9501`, the
card's spec boost, low enough that the power limiter never takes over, ~210 W
against the 370 W cap), each kernel timed over ~200 ms of work after a warm-up,
with the iteration count sized per kernel from a probe run and printed alongside
the result, and cuBLAS timed in the same run. Each figure is one ~200 ms window,
not a median of N runs; reproducibility was checked by hand (two independent runs
of the full suite agreed within 0.5 point on every ratio and 0.4% on absolute
GFLOP/s), not enforced by the harness.

The lock is not decoration; both halves of it were bought with a mistake. The
README's first numbers came from a session where the card was silently stuck in a
low memory P-state: cuBLAS measured 9.6 TFLOP/s, 32% of the card's FP32 peak,
a number no vendor SGEMM produces on healthy hardware, and that is the sanity check
that caught it. And before the clocks were pinned, the v2/cuBLAS ratio drifted by
5 points between sessions, more than some of the effects being measured.

Sizing by time replaced a flat 50 iterations, which was not good enough at
either end. At n=1024 v3 runs in 0.15 ms, so 50 iterations measured 7 ms and
launch overhead left the v3/cuBLAS ratio swinging 87–113% between runs; it now
sits at ~104%, stable. At n=4096 v1 runs in 100 ms, so the same 50 iterations
soaked the card for 5 s and moved the clocks under everything timed after it.
Equal time per kernel fixes the first and bounds the second, and locking the
clocks removes it.

The inputs rotate, per NVIDIA's GEMM measurement guidelines: enough copies to
cover twice the L2, cycled so that iteration i+1 cannot read what iteration i
left in cache. It changed nothing on GEMM, which is the useful part. An
alternated A/B, where the buffer index is the only difference between the two
timings, leaves n=1024 inside the run-to-run spread with no consistent sign over
six rounds: a GEMM at that size moves 12 MB per launch against a 5 MB L2 and
evicts its own inputs. The 122% is not a caching artifact. That the instrument
could have found one is a separate check, on a read-only streaming kernel small
enough to stay L2-resident: 504 GB/s with a fixed buffer against 333 GB/s
rotating, and 1.00× at 32 MB, well past the L2. Attention is the benchmark that
was actually flattered, and [its own section](attention.md) carries the figure. All three rotate
regardless, softmax included, where it measures clean.

*Scope: FP32 on CUDA cores throughout (no tensor cores). The cuBLAS baseline
runs in its default math mode (plain `cublasSgemm`, no `cublasSetMathMode`), so
TF32 is disabled and the comparison is like-for-like. Cross-check: cuBLAS
matches the FP32 CPU oracle to ~1e-5, where TF32's 10-bit mantissa would give
errors around 1e-3. The baseline is cuBLAS's default heuristic kernel choice via
`cublasSgemm`, not an exhaustive `cublasLt` algorithm search; that is the field
standard for this comparison, and it is what sets the ceiling. One card
(RTX 3080, `sm_86`), one OS; the ratios can move on Ada/Hopper.*

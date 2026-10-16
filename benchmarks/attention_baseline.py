"""External baseline for the attention kernels.

The GEMM numbers mean something because they are measured against cuBLAS. The
attention numbers were not: ~8.4x is a ratio between two of this repo's own
versions, which says something about v1 and nothing about anyone else's kernel.
This harness fixes that by timing the kernels against NVIDIA's own fused
attention (cuDNN, through PyTorch's SDPA dispatcher) and against PyTorch's
materialising fp32 path.

Two references, because one alone is uninterpretable:

  * SDPBackend.MATH in fp32 -- same precision, same card, scores materialised.
    The gap here is algorithmic: O(N*d) memory and one fused pass against an
    N x N intermediate.
  * cuDNN / mem-efficient in fp16 -- fused, and on tensor cores. The gap here is
    not algorithmic, it is the tensor cores this repo does not use.

Everything runs in ONE process on the SAME buffers: bench_timing.cuh's rule is
that two numbers taken in two processes are not comparable, and this project has
been caught by that twice. The kernels are called through the gemm_capi shared
library on the pointers torch allocated, so there is no copy and no second allocator.

Protocol mirrors bench_timing.cuh: warm-up, one probe to size the loop, a ~200 ms
window, and input buffers rotated over 2x L2 so no iteration reads what the last
one left in cache.
"""
import ctypes, math, os, subprocess, sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

TARGET_MS, MIN_ITERS, MAX_ITERS = 200.0, 3, 5000
TOL = 1e-3   # max abs error against the fp32 reference, same margin as ctest

# Ceilings measured on the reference RTX 3080 on 2026-08-20, clocks locked: a
# pure-FFMA kernel for FP32, cuBLAS FP16 GEMM with FP32 accumulate at n=8192 for
# FP16. On another card or another session, measure them again. They are what
# turns a raw ratio into an efficiency: a kernel 17x slower than another may be
# losing on the hardware or on its own quality, and only the fraction-of-peak on
# each side separates the two.
PEAK_FP32_GFLOPS = 28953.0   # FFMA, 97.3 % of the 1710 MHz figure, clocks locked
PEAK_FP16_GFLOPS = 57400.0   # cuBLAS fp16 with fp32 accumulate, n=8192, healthy

DLL = os.path.join(os.path.dirname(__file__), "..", "build",
                   "gemm_capi.dll" if os.name == "nt" else "libgemm_capi.so")
lib = ctypes.CDLL(os.path.abspath(DLL))
for name in ("gemm_attn_v1", "gemm_attn_v2", "gemm_attn_fa2", "gemm_attn_v3"):
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int]
lib.gemm_attn_last_error.restype = ctypes.c_int
lib.gemm_attn_l2_bytes.restype = ctypes.c_int
lib.gemm_attn_launch_shape.restype = ctypes.c_int
lib.gemm_attn_launch_shape.argtypes = [ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_int)]


def launch_shape(which, M):
    """Blocks and threads actually launched, read from the library rather than
    assumed -- occupancy is part of the reading, not a footnote."""
    thr = ctypes.c_int(0)
    blocks = lib.gemm_attn_launch_shape(which, M, ctypes.byref(thr))
    return blocks, thr.value


def clocks():
    """Measured, not locked. Locking needs admin and this campaign has already
    seen a lock report success while running slower -- a harness that measures
    the clock at every point beats one that locks and trusts."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().split("\n")[0]
    sm, mem = (int(x) for x in q.split(","))
    return sm, mem


def rotation_copies(input_bytes, cap=32):
    l2 = lib.gemm_attn_l2_bytes()
    if input_bytes == 0 or l2 == 0:
        return 1
    return max(1, min(cap, -(-2 * l2 // input_bytes)))


def ms_per_iter(step):
    """step(i) -> None. Returns (ms/iter, iters). The rotation index runs across
    the warm-up and the probe too, so the probe cannot time a warm case."""
    tick = [0]

    def go():
        step(tick[0])
        tick[0] += 1

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    go()
    torch.cuda.synchronize()

    s.record(); go(); e.record(); torch.cuda.synchronize()
    probe = s.elapsed_time(e)
    iters = MAX_ITERS if probe <= 0 else int(TARGET_MS / probe)
    iters = max(MIN_ITERS, min(MAX_ITERS, iters))

    s.record()
    for _ in range(iters):
        go()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, iters


def buffers(M, N, D, dtype, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda S: [torch.randn(1, 1, S, D, device="cuda", dtype=dtype, generator=g)
                    for _ in range(k)]
    return mk(M), mk(N), mk(N), torch.empty(1, 1, M, D, device="cuda", dtype=dtype)


def run_case(M, N, D, causal):
    scale = 1.0 / math.sqrt(D)
    fp32_bytes = (M + 2 * N) * D * 4
    k32 = rotation_copies(fp32_bytes)
    k16 = rotation_copies(fp32_bytes // 2)
    out = {}

    # --- ours, fp32, called on torch's own buffers -------------------------
    Q, K, V, O = buffers(M, N, D, torch.float32, k32)
    ours = {"v1": lib.gemm_attn_v1, "v2": lib.gemm_attn_v2, "fa2": lib.gemm_attn_fa2,
            "v3": lib.gemm_attn_v3}
    for tag, fn in ours.items():
        def step(i, fn=fn):
            j = i % k32
            fn(M, N, D, scale, ctypes.c_void_p(Q[j].data_ptr()),
               ctypes.c_void_p(K[j].data_ptr()), ctypes.c_void_p(V[j].data_ptr()),
               ctypes.c_void_p(O.data_ptr()), 1 if causal else 0)
        ms, it = ms_per_iter(step)
        err = lib.gemm_attn_last_error()
        if err:
            sys.exit(f"CUDA error {err} after {tag}")
        out[tag] = (ms, it)

    # --- reference 1: PyTorch math backend, fp32, like-for-like -----------
    def torch_step(backend, Qs, Ks, Vs, kk):
        def step(i):
            j = i % kk
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(
                    Qs[j], Ks[j], Vs[j], is_causal=causal, scale=scale)
        return step

    out["math_fp32"] = ms_per_iter(torch_step(SDPBackend.MATH, Q, K, V, k32))

    # correctness, same process, same inputs: every kernel against the fp32
    # reference. O is filled with NaN first, so a row a kernel fails to write
    # shows up as an error instead of as the previous kernel's answer.
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q[0], K[0], V[0], is_causal=causal, scale=scale)
    out["max_abs_err"] = {}
    for tag, fn in ours.items():
        O.fill_(float("nan"))
        fn(M, N, D, scale, ctypes.c_void_p(Q[0].data_ptr()),
           ctypes.c_void_p(K[0].data_ptr()), ctypes.c_void_p(V[0].data_ptr()),
           ctypes.c_void_p(O.data_ptr()), 1 if causal else 0)
        torch.cuda.synchronize()
        out["max_abs_err"][tag] = (ref - O).abs().max().item()

    # --- reference 2: fused fp16 on tensor cores --------------------------
    del Q, K, V, O
    torch.cuda.empty_cache()
    Q, K, V, _ = buffers(M, N, D, torch.float16, k16)
    torch.backends.cuda.enable_cudnn_sdp(True)
    for tag, be in (("cudnn_fp16", SDPBackend.CUDNN_ATTENTION),
                    ("efficient_fp16", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            out[tag] = ms_per_iter(torch_step(be, Q, K, V, k16))
        except RuntimeError as exc:
            out[tag] = (float("nan"), 0)
            print(f"    {tag}: unavailable ({str(exc)[:60]})")
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = False   # like-for-like with our fp32
    torch.backends.cudnn.allow_tf32 = False
    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"L2 {lib.gemm_attn_l2_bytes()//1024} KiB")
    cases = [(4096, 4096, 64), (4096, 4096, 128), (2048, 2048, 64), (1024, 1024, 64)]
    hdr = ("n", "d", "causal", "v2 ms", "fa2 ms", "**v3 ms**", "best", "GF/s", "% peak",
           "cuDNN ms", "% peak", "math32 ms", "math/ours", "ours/cuDNN", "kernel",
           "v3 vs old", "max err")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for M, N, D in cases:
        for causal in (False, True):
            r = run_case(M, N, D, causal)
            gflop = 4.0 * M * N * D / 1e9
            if causal:
                gflop *= 0.5          # ~half the keys masked, exact for M == N
            tag, ms = min((("v1", r["v1"][0]), ("v2", r["v2"][0]), ("fa2", r["fa2"][0]),
                           ("v3", r["v3"][0])), key=lambda t: t[1])
            ours_gf = gflop / (ms / 1e3)
            cud = r["cudnn_fp16"][0]
            cud_gf = gflop / (cud / 1e3)
            ours_pk = 100 * ours_gf / PEAK_FP32_GFLOPS
            cud_pk = 100 * cud_gf / PEAK_FP16_GFLOPS
            hw = PEAK_FP16_GFLOPS / PEAK_FP32_GFLOPS
            old = min(r["v1"][0], r["v2"][0], r["fa2"][0])
            errs = r["max_abs_err"]
            bad = {t: e for t, e in errs.items() if not e <= TOL}
            worst = max(errs.values(), key=lambda e: math.inf if math.isnan(e) else e)
            print(f"| {M} | {D} | {int(causal)} | {r['v2'][0]:.3f} | {r['fa2'][0]:.3f} | "
                  f"**{r['v3'][0]:.3f}** | {tag} | {ours_gf:.0f} | {ours_pk:.1f}% | "
                  f"{cud:.3f} | {cud_pk:.1f}% | {r['math_fp32'][0]:.3f} | "
                  f"{r['math_fp32'][0]/ms:.2f}x | {ms/cud:.2f}x | {(cud_pk/ours_pk):.1f}x | "
                  f"**{old/r['v3'][0]:.2f}x** | {worst:.1e} |")
            if bad:   # NaN fails `e <= TOL` too
                sys.exit(f"wrong result at n={M}, d={D}, causal={int(causal)}: {bad}")


if __name__ == "__main__":
    main()

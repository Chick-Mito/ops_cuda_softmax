"""Softmax Benchmark — 3 CUDA kernels + torch.softmax baseline."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.add_dll_directory(r"D:\anaconda\envs\ainfra\lib\site-packages\torch\lib")

import torch
from datetime import datetime

BENCH_SIZES = [
    (4096, 4096, "4096x4096"),
    (4096, 1024, "4096x1024"),
    (1024, 4096, "1024x4096"),
    (1024, 1024, "1024x1024"),
    (512,  512,  "512x512"),
]
WARMUP = 10
ITERS = 100

def benchmark(func, x, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        func(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def run():
    try:
        import softmax_ops
    except ImportError:
        print("[ERROR] softmax_ops not found. Build first: python setup.py build_ext --inplace")
        return

    kernels = [
        ("Naive",          softmax_ops.softmax_naive),
        ("Online",         softmax_ops.softmax_online),
        ("Warp",           softmax_ops.softmax_warp),
        ("torch.softmax",  lambda t: torch.softmax(t, dim=-1)),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    all_rows = []

    for M, N, label in BENCH_SIZES:
        print(f"\n{'='*60}")
        print(f"  Softmax: {label}  (M={M}, N={N})")
        print(f"{'='*60}")

        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)

        for name, func in kernels:
            out = func(x)
            torch.cuda.synchronize()
            max_diff = (ref - out).abs().max().item()
            t_ms = benchmark(func, x)
            bytes_accessed = (M * N * 2) * 4
            bw = (bytes_accessed / (t_ms / 1000)) / 1e9
            status = "PASS" if max_diff < 0.01 else ("WARN" if max_diff < 0.1 else "INFO")
            if name == "torch.softmax":
                status = "REF "
            print(f"  [{status}] {name:>14s} | {t_ms*1000:8.2f} us | {bw:6.2f} GB/s | diff={max_diff:.5f}")
            all_rows.append({"size": label, "M": M, "N": N, "kernel": name,
                             "time_us": round(t_ms*1000, 2), "bandwidth_gbs": round(bw, 2),
                             "max_diff": max_diff})
    print(f"\nResults: {results_dir}")

    # Summary table
    knames = ["Naive", "Online", "Warp", "torch.softmax"]
    print(f"\n{'='*80}")
    print(f"  SUMMARY (time in us)")
    print(f"{'='*80}")
    header = f"{'Size':<16s}"
    for k in knames:
        header += f" {k:>14s}"
    print(header)
    print(f"{'-'*16}{' -'*14 * len(knames)}")
    for M, N, label in BENCH_SIZES:
        row = f"{label:<16s}"
        for k in knames:
            match = [r for r in all_rows if r["size"] == label and r["kernel"] == k]
            row += f" {match[0]['time_us']:>12.1f} us" if match else f" {'--':>14s}"
        print(row)
    print(f"{'='*80}")

if __name__ == "__main__":
    run()

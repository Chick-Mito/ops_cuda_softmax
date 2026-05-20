# ops_cuda_softmax

[中文版](README.md)

Hand-rolled CUDA Softmax — 5 kernels, from naive to warp-level, on RTX 3060 Ti.

## Performance (4096×4096 FP32)

| Kernel | Algorithm | Time | Bandwidth | vs Naive | vs torch |
|--------|-----------|------|-----------|----------|----------|
| Naive | 3-pass, 1 thread/row | 2788 us | 48 GB/s | 1.0x | 7.8x |
| Online | 2-pass, 1 thread/row | 2101 us | 64 GB/s | 1.3x | 5.9x |
| Warp | 32 threads/row, coalesced | 554 us | 242 GB/s | 5.0x | 1.56x |
| **Warp+float4** | **+ 128-bit vectorized load** | **520 us** | **258 GB/s** | **5.4x** | **1.46x** |
| Warp+Tiled | SMEM tiled (1 block/row) | 589 us | 228 GB/s | 4.7x | 1.65x |
| torch.softmax | cuBLAS/cuDNN | 356 us | 377 GB/s | 7.8x | 1.0x |

> Warp+float4 is fastest. SMEM tiling is slower — Softmax has no data reuse, barrier overhead exceeds gains. At 512×512 Warp+float4 nearly matches torch (5.0 vs 4.6 us).

## Quick Start

```bash
conda activate ainfra
python setup.py build_ext --inplace
python bench/benchmark.py
```

## Project Structure

```
ops_cuda_softmax/
├── src/
│   ├── softmax_kernels.cu       # 5 kernels + bridge functions
│   └── softmax_wrapper.cpp      # pybind11 bridge
├── bench/
│   └── benchmark.py             # Performance test (5 sizes)
├── docs/
│   └── softmax_analysis.md      # Full analysis (8 chapters)
├── setup.py
├── README.md / README_EN.md
```

## Kernel Overview

### Level 1 — Naive Softmax
- 1 thread/row, 3 passes. Bottleneck: uncoalesced access (stride-N)

### Level 2 — Online Softmax
- 1 thread/row, 2 passes using running (m, O). ~25% faster.

### Level 3 — Warp-Level Online Softmax
- 1 warp (32 threads)/row, coalesced stride-32 access
- Warp reduce (`__shfl_down_sync`) merges 32 local (m, O) → global
- **5-19x** faster

### Level 4 — Warp + float4
- 128-bit vectorized load, 4 floats at once → 4x fewer LDG instructions
- +6-19% at large N. Near torch at 512×512

### Level 5 — SMEM Tiled (educational value)
- 1 block/row, Shared Memory tiling (TILE_SIZE=1024)
- **Slower than Level 3** — Softmax lacks data reuse; `__syncthreads()` overhead > SMEM benefit
- Key counter-example: not all GPU optimizations apply to all workloads

## Nsight Compute Analysis (4096×4096 FP32)

| Kernel | SM Busy | IPC | DRAM% | MemSOL | CmpSOL | NoElig | Bottleneck |
|--------|---------|-----|-------|--------|--------|--------|------------|
| Naive | 6.4% | 0.26 | 17.1% | 34.4% | 2.7% | 93.6% | Thread count |
| Online | 9.9% | 0.39 | 16.5% | 40.3% | 4.1% | 90.1% | Thread count |
| Warp | 22.9% | 0.91 | **84.1%** | 84.1% | 22.0% | 77.2% | **DRAM bandwidth** |
| Warp+float4 | 24.5% | 0.98 | **90.2%** | 90.2% | 23.2% | 75.7% | **DRAM bandwidth** |
| Warp+Tiled | **84.5%** | **3.38** | 58.3% | 58.3% | **84.0%** | **15.5%** | **expf compute** |

> Warp+float4 hits 90.2% DRAM utilization (392 GB/s of 448 GB/s theoretical). Tiled flips to Compute-Bound (84% Compute SOL) but barrier overhead exceeds bandwidth savings.

## Key Findings

- **Warp+float4 is DRAM-bandwidth-saturated**: 392 GB/s (90% peak), no SMEM needed
- **SMEM tiling fails for Softmax**: no data reuse, barrier + expf cost > DRAM savings — key contrast with GEMM
- **Remaining gap (1.46x) is fast-expf**: DRAM is maxed out; torch likely uses table-lookup / polynomial approx

## Documentation

`docs/softmax_analysis.md` — 9 chapters covering algorithm derivation, all 5 kernels, NCU profiling, SMEM tiling failure analysis, cuBLAS gap breakdown.

## License

MIT

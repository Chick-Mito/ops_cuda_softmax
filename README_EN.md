# ops_cuda_softmax

[中文版](README.md)

Hand-rolled CUDA Softmax — 3 kernels, from naive to warp-level, on RTX 3060 Ti.

## Performance (4096×4096 FP32)

| Kernel | Algorithm | Time | Bandwidth | vs Naive |
|--------|-----------|------|-----------|----------|
| Naive | 3-pass, 1 thread/row | 2779 us | 48 GB/s | 1.0x |
| Online | 2-pass, 1 thread/row | 2107 us | 64 GB/s | 1.3x |
| **Warp** | **Warp-level online, 32 threads/row** | **545 us** | **246 GB/s** | **5.1x** |
| torch.softmax | cuBLAS/cuDNN | 414 us | 324 GB/s | 6.7x |

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
│   ├── softmax_kernels.cu       # 3 kernels + bridge functions
│   └── softmax_wrapper.cpp      # pybind11 bridge
├── bench/
│   └── benchmark.py             # Performance test (5 sizes)
├── docs/
│   └── softmax_analysis.md      # Full analysis doc
├── setup.py
└── README.md
```

## Kernel Overview

### Level 1 — Naive Softmax
- 1 thread per row, 3 passes: find max → exp sum → normalize
- Bottleneck: uncoalesced global memory access (stride-N)

### Level 2 — Online Softmax
- Same 1 thread/row, but 2 passes using running (m, O)
- Pass 1: online max + exp sum accumulation
- Pass 2: normalize
- ~25% faster by eliminating one full-row scan

### Level 3 — Warp-Level Online Softmax
- 1 warp (32 threads) per row, 2 passes
- Each thread handles N/32 elements with stride-32 → coalesced access
- Warp reduce (`__shfl_down_sync`) merges 32 local (m, O) into global
- **5-19x** faster depending on dimensions

## Documentation

`docs/softmax_analysis.md` — full analysis covering:
- GPU softmax algorithm derivation
- Three kernel code walkthrough
- Performance analysis across dimensions
- Why warp-level dominates (coalesced access + parallelism)

## License

MIT

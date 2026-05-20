# ops_cuda_softmax

[English](README_EN.md)

手写 CUDA Softmax——3 个 kernel，从朴素实现到 Warp 级并行，单卡 RTX 3060 Ti。

## 性能 (4096×4096 FP32)

| Kernel | 算法 | 时间 | 带宽 | vs Naive |
|--------|------|------|------|----------|
| Naive | 3-pass, 1 线程/行 | 2824 us | 48 GB/s | 1.0x |
| Online | 2-pass, 1 线程/行 | 2168 us | 62 GB/s | 1.3x |
| **Warp** | **Warp 级在线算法, 32 线程/行** | **572 us** | **235 GB/s** | **4.9x** |
| torch.softmax | cuBLAS/cuDNN | 362 us | 371 GB/s | 7.8x |

> Warp vs torch.softmax: 1.6x 差距。512×512 时几乎持平（5.7 vs 5.0 us）。差距来自 SMEM 分块 + float4 向量化——每一项都是微优化。

## 快速开始

```bash
conda activate ainfra
python setup.py build_ext --inplace
python bench/benchmark.py
```

## 项目结构

```
ops_cuda_softmax/
├── src/
│   ├── softmax_kernels.cu       # 3 个 kernel + bridge 函数
│   └── softmax_wrapper.cpp      # pybind11 桥接
├── bench/
│   └── benchmark.py             # 性能测试 (5 种规模)
├── docs/
│   └── softmax_analysis.md      # 完整分析文档
├── setup.py
├── README.md / README_EN.md
```

## Kernel 概览

### Level 1 — Naive Softmax
- 1 线程/行，3 次遍历：找最大值 → exp 求和 → 归一化
- 瓶颈：非合并全局内存访问 (stride-N)

### Level 2 — Online Softmax
- 同样 1 线程/行，但用运行 (m, O) 降到 2 次遍历
- Pass 1：在线递推最大值 + exp 和
- Pass 2：归一化
- 减少一次全行扫描，快 ~25%

### Level 3 — Warp-Level Online Softmax
- 1 warp (32 线程)/行，2 次遍历
- 每线程 stride-32 处理 N/32 个元素 → 完美合并访存
- Warp Reduce (`__shfl_down_sync`) 将 32 个局部 (m, O) 合并为全局值
- **5-19x** 加速，取决于维度

## 核心发现

- **合并访存是 Softmax 优化的根基**：Warp 级 stride-32 访存比单线程 stride-N 快 5-19x
- **Warp Reduce 零延迟**：`__shfl_down_sync` 在寄存器间传输，~5 cycles，不需要 Shared Memory
- **单线程/行是 GPU 最差的使用方式**：N=4096 时每线程串行 12288 次全局内存访问

## 文档

`docs/softmax_analysis.md` — 完整分析，涵盖：
- Softmax 算法推导（含 Online Softmax 递推公式证明）
- 三级 kernel 源码走读
- 多维度性能分析
- Warp 级并行为何碾压单线程

## License

MIT

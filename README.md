# ops_cuda_softmax

[English](README_EN.md)

手写 CUDA Softmax——5 个 kernel，从朴素实现到 Warp 级并行，单卡 RTX 3060 Ti。

## 性能 (4096×4096 FP32)

| Kernel | 算法 | 时间 | 带宽 | vs Naive | vs torch |
|--------|------|------|------|----------|----------|
| Naive | 3-pass, 1 线程/行 | 2788 us | 48 GB/s | 1.0x | 7.8x |
| Online | 2-pass, 1 线程/行 | 2101 us | 64 GB/s | 1.3x | 5.9x |
| Warp | 32 线程/行, 合并访存 | 554 us | 242 GB/s | 5.0x | 1.56x |
| **Warp+float4** | **+ 128-bit 向量化加载** | **520 us** | **258 GB/s** | **5.4x** | **1.46x** |
| Warp+Tiled | SMEM 分块 (1 block/行) | 589 us | 228 GB/s | 4.7x | 1.65x |
| torch.softmax | cuBLAS/cuDNN | 356 us | 377 GB/s | 7.8x | 1.0x |

> Warp+float4 是最快的——大 N 时 +6-19%。SMEM 分块反而更慢（Softmax 无数据复用，barrier 开销超过收益）。512×512 时 Warp+float4 与 torch 几乎持平（5.0 vs 4.6 us）。

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
│   ├── softmax_kernels.cu       # 5 个 kernel + bridge 函数
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

### Level 4 — Warp + float4 向量化
- 基于 Level 3，每次 `float4` 读 4 个元素 → 减少 4x LDG 指令
- 大 N 时 +6-19%，512×512 几乎持平 torch（5.0 vs 4.6 us）

### Level 5 — SMEM Tiled（教学意义）
- 一 block/行，Shared Memory 分块（TILE_SIZE=1024）
- **反而比 Level 3 慢**——Softmax 没有 GEMM 那样的数据复用，`__syncthreads()` 开销 > SMEM 收益
- 重要反例：不是所有 GPU 优化都适用于所有场景

## Nsight Compute 分析 (4096×4096 FP32)

| Kernel | SM Busy | IPC | DRAM% | MemSOL | CmpSOL | NoElig | 瓶颈 |
|--------|---------|-----|-------|--------|--------|--------|------|
| Naive | 6.4% | 0.26 | 17.1% | 34.4% | 2.7% | 93.6% | 线程数不足 |
| Online | 9.9% | 0.39 | 16.5% | 40.3% | 4.1% | 90.1% | 线程数不足 |
| Warp | 22.9% | 0.91 | **84.1%** | 84.1% | 22.0% | 77.2% | **DRAM 带宽** |
| Warp+float4 | 24.5% | 0.98 | **90.2%** | 90.2% | 23.2% | 75.7% | **DRAM 带宽** |
| Warp+Tiled | **84.5%** | **3.38** | 58.3% | 58.3% | **84.0%** | **15.5%** | **expf 计算** |

> Warp+float4 DRAM 利用率 90.2%（448 GB/s 理论 → 实测 392 GB/s），已接近硬件带宽上限。Tiled 翻转为 Compute-Bound（84% Compute SOL）但 barrier 开销反超了带宽节省。

## 核心发现

- **Warp+float4 已触及 DRAM 带宽天花板**：392 GB/s（90% 理论峰值），无需 SMEM
- **SMEM 分块对 Softmax 无效**：无数据复用，barrier 开销 + expf 瓶颈 > DRAM 带宽节省
- **与 torch 的 1.46x 差距在 fast-expf**：DRAM 带宽已榨干，剩余差距来自更优的指数函数实现（查表/多项式近似）
- **注意对比 GEMM**：GEMM 中 SMEM tiling 核心收益来自数据复用（BK 次），Softmax 每个元素只用 2 次——判若云泥

## 文档

`docs/softmax_analysis.md` — 9 章完整分析，涵盖：
- Softmax 算法推导（含 Online Softmax 递推公式证明）
- 五级 kernel 源码走读 + 实测对比
- SMEM tiling 为何对 Softmax 无效的深度分析
- cuBLAS 差距剖析 + 未来优化方向

## License

MIT

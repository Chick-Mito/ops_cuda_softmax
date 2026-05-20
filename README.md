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

## 核心发现

- **合并访存是 Softmax 优化的根基**：stride-32 比 stride-N 快 5-19x
- **float4 向量化锦上添花**：减少 LDG 指令，大 N 时有用
- **SMEM 分块对 Softmax 无效**：没有数据复用，barrier 开销超过收益——与 GEMM 形成关键对比
- **剩余瓶颈在 expf 计算**：与 torch 的 1.46x 差距主要来自更优的指数函数实现

## 文档

`docs/softmax_analysis.md` — 8 章完整分析，涵盖：
- Softmax 算法推导（含 Online Softmax 递推公式证明）
- 五级 kernel 源码走读 + 实测对比
- SMEM tiling 为何对 Softmax 无效的深度分析
- cuBLAS 差距剖析 + 未来优化方向

## License

MIT

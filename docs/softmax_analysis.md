# CUDA Softmax 算子分析文档

> GPU: RTX 3060 Ti (Ampere, sm_86) | 数据来源: `bench/benchmark.py` 实测

## 第1章 Softmax 的数学定义与数值稳定性

### 1.1 标准 Softmax

给定输入向量 $x \in \mathbb{R}^N$，Softmax 输出：

$$y_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}$$

### 1.2 数值稳定版本（Safe Softmax）

为避免 $e^{x_i}$ 溢出，减去最大值：

$$y_i = \frac{e^{x_i - m}}{\sum_{j=1}^N e^{x_j - m}}, \quad m = \max(x)$$

所有实现都使用此安全版本。

---

## 第2章 Level 1 — Naive Softmax

> **源码**: `src/softmax_kernels.cu` → `softmax_kernel`
> **策略**: 1 线程/行，3-pass

### 2.1 算法

```
Pass 1: 找每行最大值 m
Pass 2: 计算 sum = Σ exp(x_i - m)
Pass 3: 输出 y_i = exp(x_i - m) / sum
```

### 2.2 实现

```cuda
__global__ void softmax_kernel(input, output, M, N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Pass 1: find max
    float max_val = input[row * N];
    for (int i = 1; i < N; ++i)
        max_val = fmaxf(max_val, input[row * N + i]);

    // Pass 2: exp sum
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += expf(input[row * N + i] - max_val);

    // Pass 3: normalize
    for (int i = 0; i < N; ++i)
        output[row * N + i] = expf(input[row * N + i] - max_val) / sum;
}
```

### 2.3 瓶颈分析

| 问题 | 说明 |
|------|------|
| 单线程/行 | 每行 1 个线程，N=4096 时每线程做 3×4096 = 12288 次全局内存访问 |
| 非合并访存 | `input[row*N + i]` 对同一 warp 的不同 row 是 stride-N 跳跃访问 |
| 3-pass | 遍历整行 3 次，总读次数 = 3MN，写次数 = MN |

### 2.4 实测性能

| M×N | 时间 (us) | 带宽 (GB/s) | 分析 |
|-----|----------|------------|------|
| 4096×4096 | 2779 | 48 | N 大→每线程计算量大 |
| 4096×1024 | 659 | 51 | N 小→每线程计算小 |
| 1024×4096 | 2614 | 13 | M 小→SM 填不满（只有 1024 线程） |
| 1024×1024 | 672 | 12 | 同上 |
| 512×512 | 321 | 7 | M=512→仅 512 线程，GPU 大量空闲 |

**核心问题**：M 较小时，线程数不足以填满 38 个 SM（38×1536=58368 最大线程）。M=1024 只有 1024 线程，GPU 利用率极低。

---

## 第3章 Level 2 — Online Softmax

> **源码**: `src/softmax_kernels.cu` → `online_softmax_kernel`
> **策略**: 1 线程/行，2-pass（消去一次全行扫描）

### 3.1 算法推导

Online Softmax 在一次遍历中同时维护运行最大值 `m` 和运行指数和 `O`：

```
对每个元素 x:
    new_m = max(m, x)
    O = O * exp(m - new_m) + exp(x - new_m)
    m = new_m
```

**推导**（为什么这个递推成立）：

设已处理了前 k 个元素，当前最大值 $m_k$，当前指数和 $O_k = \sum_{j=1}^k e^{x_j - m_k}$。

处理下一个元素 $x_{k+1}$：
- 如果 $x_{k+1} \leq m_k$：$m_{k+1} = m_k$，只需累加：$O_{k+1} = O_k + e^{x_{k+1} - m_k}$
- 如果 $x_{k+1} > m_k$：$m_{k+1} = x_{k+1}$，需要"更新"旧和：
  $$O_{k+1} = O_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}$$

统一公式：
$$O \leftarrow O \cdot e^{m - \text{new\_m}} + e^{x - \text{new\_m}}$$

这就是代码中的递推公式。

### 3.2 实现

```cuda
// Pass 1: online accumulation
float m = -INFINITY, O = 0.0f;
for (int i = 0; i < N; i++) {
    float x = input[row * N + i];
    float new_m = fmaxf(m, x);
    O = O * expf(m - new_m) + expf(x - new_m);
    m = new_m;
}

// Pass 2: normalize
for (int i = 0; i < N; ++i)
    output[row * N + i] = expf(input[row * N + i] - m) / O;
```

### 3.3 效果

| M×N | Naive | Online | 加速比 |
|-----|-------|--------|--------|
| 4096×4096 | 2779 us | 2107 us | **1.32x** |
| 4096×1024 | 659 us | 517 us | 1.27x |
| 1024×4096 | 2614 us | 2122 us | 1.23x |

**收益来源**：从 3-pass 降到 2-pass → 减少 33% 的全局内存读取。同时 Pass 1 的 online 递推比单独找 max + exp sum 更高效。

**局限**：仍然是 1 线程/行，访存模式不变。M 小时 SM 利用率低的问题依然存在。

---

## 第4章 Level 3 — Warp-Level Online Softmax

> **源码**: `src/softmax_kernels.cu` → `softmax_warp_online_kernel`
> **策略**: 1 warp (32 线程)/行，warp 内并行 + warp reduce

### 4.1 核心思想

前两个 kernel 每行只用 1 线程，GPU 最擅长的并行计算完全没用上。

**Warp-Level 方案**：
- 每行分配 **1 个 warp（32 线程）**
- 每个线程以 stride-32 处理 1/32 的元素 → **完美合并访存**
- Warp Reduce 合并 32 个局部 (m, O) → 全局 (m, O)
- 最后每个线程以 stride-32 写回结果

### 4.2 Warp Reduce 原语

```cuda
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}
```

`__shfl_down_sync` 是 warp 内线程间数据交换指令，延迟 ~5 cycles，不需要 Shared Memory。树形归约：16→8→4→2→1，5 次循环即可完成 32 线程归约。

### 4.3 实现

```cpp
__global__ void softmax_warp_online_kernel(input, output, M, N) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= M) return;

    // Phase 1: each thread handles N/32 elements with stride-32
    float local_m = -INFINITY, local_O = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float x = row_input[i];
        float new_m = fmaxf(local_m, x);
        local_O = local_O * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // Phase 2: warp reduce 32 locals → global
    float global_m = warp_reduce_max(local_m);
    local_O = local_O * expf(local_m - global_m);
    float global_O = warp_reduce_sum(local_O);

    // Phase 3: write back
    for (int i = lane_id; i < N; i += 32)
        row_output[i] = expf(row_input[i] - global_m) / global_O;
}
```

### 4.4 线程映射

```
Block(256) = 8 warps
Thread 0..31   → Warp 0 → Row 0
Thread 32..63  → Warp 1 → Row 1
...
Thread 224..255 → Warp 7 → Row 7
```

总线程数 = M × 32（M 行，每行 32 线程）。
Grid = ceil(M×32 / 256) blocks。

### 4.5 性能：飞跃式提升

| M×N | Naive | Online | Warp | vs Naive | vs Online |
|-----|-------|--------|------|----------|-----------|
| 4096×4096 | 2779 us | 2107 us | **545 us** | **5.1x** | **3.9x** |
| 4096×1024 | 659 us | 517 us | **143 us** | 4.6x | 3.6x |
| 1024×4096 | 2614 us | 2122 us | **139 us** | **18.8x** | **15.3x** |
| 1024×1024 | 672 us | 529 us | **41 us** | 16.2x | 12.8x |
| 512×512 | 321 us | 273 us | **7 us** | 49.3x | 42.1x |

### 4.6 为什么快这么多

**1. 合并访存（Coalesced Access）**

- Naive/Online：`input[row*N + i]`，连续 `i` 跨同一行。但同一 warp 的不同线程处理不同行（row 不同），所以地址跨 stride-N → 彻底不合并。
- Warp：同一 warp 的 32 线程处理同一行，`lane_id=0,1,...,31` 并行读 `input[row][0], input[row][1], ..., input[row][31]` → **32 个连续地址 → 单次 128-byte 内存事务**。

**2. 并行度**

Naive/Online 每行 1 线程 → M 行用 M 个线程。Warp 每行 32 线程 → M 行用 32M 个线程。即使最大 M=4096，也只有 131,072 线程，GPU 可以轻松填满。

**3. Warp Reduce 零延迟**

`__shfl_down_sync` 在寄存器间传输数据，延迟 ~5 cycles，不需要 Shared Memory 或 Global Memory。归约 32 个值只需 5×5 = 25 cycles。

**4. 与 torch.softmax 的对比**

| M×N | Warp (我们) | torch.softmax | Gap |
|-----|------------|---------------|-----|
| 4096×4096 | 545 us | 414 us | 1.3x |
| 1024×4096 | 139 us | ~100 us | 1.4x |

差距约 30%——torch.softmax 可能使用了 Shared Memory 分块或更精细的 warp 调度。但我们的 Warp kernel 已经接近 NVIDIA 库函数的水平。

---

## 第5章 总结

### 5.1 三级优化对比（4096×4096 FP32）

| Kernel | 线程/行 | 遍历次数 | 访存模式 | 时间 | 带宽 | 加速比 |
|--------|---------|---------|---------|------|------|--------|
| Naive | 1 | 3 | 非合并 | 2779 us | 48 GB/s | 1.0x |
| Online | 1 | 2 | 非合并 | 2107 us | 64 GB/s | 1.3x |
| **Warp** | **32** | **2** | **合并** | **545 us** | **246 GB/s** | **5.1x** |

### 5.2 核心教训

1. **GPU 并行性是第一位的**：1 线程/行在 N=4096 时每线程做 12288 次串行操作——GPU 最不擅长的事。把同一行分给 32 线程就解决了。
2. **合并访存是所有优化的基础**：同样的计算任务，stride-32 合并访存比 stride-N 非合并快 5-19x。
3. **Warp Reduce 是 softmax 的终局**：对于逐行 softmax 这种操作，warp 级别的并行已经是最高效的粒度——不需要 Shared Memory、不需要 Block 同步。超越 warp 级别没有意义，因为一行就是一个独立计算单元。

### 5.3 可选后续方向

| 方向 | 说明 | 优先级 |
|------|------|--------|
| Shared Memory 分块 | 当 N 极大（>16384）时，用 SMEM 分块减少 Global Memory 往返 | 低 |
| NCU Profiling | 定量验证合并访存和 warp stall 数据 | 中 |
| 向量化加载 | float4 一次加载 4 个元素 | 低 |

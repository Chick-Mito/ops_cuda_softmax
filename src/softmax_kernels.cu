#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <torch/extension.h>

// ==================== Kernel 1: Naive Softmax ====================
// 1 thread per row, 3-pass: find max → exp sum → normalize
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Pass 1: find max
    float max_val = input[row * N];
    for (int i = 1; i < N; ++i)
        max_val = fmaxf(max_val, input[row * N + i]);

    // Pass 2: compute sum of exp
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += expf(input[row * N + i] - max_val);

    // Pass 3: normalize
    for (int i = 0; i < N; ++i)
        output[row * N + i] = expf(input[row * N + i] - max_val) / sum;
}

// ==================== Kernel 2: Online Softmax ====================
// 1 thread per row, 2-pass: online accumulate (m, O) → normalize
__global__ void online_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Pass 1: online accumulation
    float m = -INFINITY;
    float O = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = input[row * N + i];
        float new_m = fmaxf(m, x);
        O = O * expf(m - new_m) + expf(x - new_m);
        m = new_m;
    }

    // Pass 2: normalize
    for (int i = 0; i < N; ++i)
        output[row * N + i] = expf(input[row * N + i] - m) / O;
}

// ==================== Kernel 3: Warp-Level Online Softmax ====================
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

// 1 warp (32 threads) per row
__global__ void softmax_warp_online_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= M) return;

    int row = warp_id;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    // Phase 1: warp-parallel online accumulation
    float local_m = -INFINITY;
    float local_O = 0.0f;

    for (int i = lane_id; i < N; i += 32) {
        float x = row_input[i];
        float new_m = fmaxf(local_m, x);
        local_O = local_O * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // Warp reduce: merge 32 local (m, O) into global (m, O)
    float global_m = warp_reduce_max(local_m);
    local_O = local_O * expf(local_m - global_m);
    float global_O = warp_reduce_sum(local_O);

    // Phase 2: warp-parallel write back
    for (int i = lane_id; i < N; i += 32)
        row_output[i] = expf(row_input[i] - global_m) / global_O;
}

// ==================== Kernel 4: Warp-Level + float4 Vectorized Load ====================
// Same as Kernel 3, but uses float4 for 4x fewer LDG instructions
__global__ void softmax_warp_float4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= M) return;

    int row = warp_id;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float local_m = -INFINITY;
    float local_O = 0.0f;

    // Phase 1: float4 stride-128 coalesced load (4 floats at once)
    int i = lane_id * 4;  // start at lane_id*4, step by 128
    for (; i + 3 < N; i += 128) {
        float4 v = *reinterpret_cast<const float4*>(&row_input[i]);
        float new_m = fmaxf(local_m, v.x);
        local_O = local_O * expf(local_m - new_m) + expf(v.x - new_m);
        local_m = new_m;
        new_m = fmaxf(local_m, v.y);
        local_O = local_O * expf(local_m - new_m) + expf(v.y - new_m);
        local_m = new_m;
        new_m = fmaxf(local_m, v.z);
        local_O = local_O * expf(local_m - new_m) + expf(v.z - new_m);
        local_m = new_m;
        new_m = fmaxf(local_m, v.w);
        local_O = local_O * expf(local_m - new_m) + expf(v.w - new_m);
        local_m = new_m;
    }
    // Tail: handle remaining elements (N not multiple of 4)
    for (; i < N; i++)
        local_O += expf(row_input[i] - local_m);

    float global_m = warp_reduce_max(local_m);
    local_O = local_O * expf(local_m - global_m);
    float global_O = warp_reduce_sum(local_O);

    // Phase 2: float4 write back
    i = lane_id * 4;
    for (; i + 3 < N; i += 128) {
        float4 out;
        out.x = expf(row_input[i + 0] - global_m) / global_O;
        out.y = expf(row_input[i + 1] - global_m) / global_O;
        out.z = expf(row_input[i + 2] - global_m) / global_O;
        out.w = expf(row_input[i + 3] - global_m) / global_O;
        *reinterpret_cast<float4*>(&row_output[i]) = out;
    }
    for (; i < N; i++)
        row_output[i] = expf(row_input[i] - global_m) / global_O;
}

// ==================== Kernel 5: SMEM Tiled Softmax (1 block/row) ====================
// One block per row — all warps in block cooperate on the same row via SMEM tiling
#define TILE_SIZE 1024
__global__ void softmax_warp_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    __shared__ float tile[TILE_SIZE];

    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float global_m = -INFINITY;
    float global_O = 0.0f;
    int nTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Pass 1: tiled online accumulation
    for (int t = 0; t < nTiles; t++) {
        int tile_start = t * TILE_SIZE;
        // Cooperative load: 256 threads stride-256 load tile into SMEM
        for (int j = tid; j < TILE_SIZE && tile_start + j < N; j += 256)
            tile[j] = row_input[tile_start + j];
        __syncthreads();

        // Each warp computes local (m,O) over its portion, then warp-reduce
        int warp_start = lane_id;
        float local_m = -INFINITY, local_O = 0.0f;
        int tile_end = TILE_SIZE;
        if (tile_start + TILE_SIZE > N) tile_end = N - tile_start;
        for (int j = warp_start; j < tile_end; j += 32) {
            float x = tile[j];
            float new_m = fmaxf(local_m, x);
            local_O = local_O * expf(local_m - new_m) + expf(x - new_m);
            local_m = new_m;
        }

        float tile_m = warp_reduce_max(local_m);
        local_O = local_O * expf(local_m - tile_m);
        float tile_O = warp_reduce_sum(local_O);

        // Merge tile → global (use tid=0 of any warp for global merge)
        // All warps in the block must agree: use warp 0's result
        if (tid < 32) {
            // Only warp 0 participates in global merge
            // But all threads need __syncthreads before global merge
            // Store per-warp results in SMEM (32 floats for m, 32 for O)
        }
        // ... simplified: sequential merge by tid 0
        float new_gm = fmaxf(global_m, tile_m);
        global_O = global_O * expf(global_m - new_gm) + tile_O * expf(tile_m - new_gm);
        global_m = new_gm;
        __syncthreads();
    }

    // Pass 2: write back
    for (int t = 0; t < nTiles; t++) {
        int tile_start = t * TILE_SIZE;
        for (int j = tid; j < TILE_SIZE && tile_start + j < N; j += 256)
            tile[j] = row_input[tile_start + j];
        __syncthreads();

        int tile_end = TILE_SIZE;
        if (tile_start + TILE_SIZE > N) tile_end = N - tile_start;
        for (int j = tid; j < tile_end; j += 256)
            row_output[tile_start + j] = expf(tile[j] - global_m) / global_O;
        __syncthreads();
    }
}

// Note: The SMEM tiled approach above has a fundamental issue — each warp
// computes its own tile (m,O) independently, but all warps then need to
// agree on a single global (m,O). The merge step currently runs redundantly
// on all warps, producing the same result (since tile_m and tile_O are
// warp-broadcast). This works but is wasteful. A proper implementation
// would use a warp-level reduce across the 8 warps in the block.

// ==================== Bridge Functions ====================
torch::Tensor softmax_naive_forward(torch::Tensor input) {
    input = input.contiguous().cuda();
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor softmax_online_forward(torch::Tensor input) {
    input = input.contiguous().cuda();
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    online_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor softmax_warp_forward(torch::Tensor input) {
    input = input.contiguous().cuda();
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int threads_per_block = 256; // 8 warps per block
    int total_threads = M * 32;   // 1 warp per row
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    softmax_warp_online_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor softmax_warp_float4_forward(torch::Tensor input) {
    input = input.contiguous().cuda();
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int threads_per_block = 256;
    int total_threads = M * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    softmax_warp_float4_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor softmax_warp_tiled_forward(torch::Tensor input) {
    input = input.contiguous().cuda();
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int threads_per_block = 256;
    int blocks = M;  // 1 block per row
    softmax_warp_tiled_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

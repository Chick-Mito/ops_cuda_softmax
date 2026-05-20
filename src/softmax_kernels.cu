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

"""Minimal script for NCU profiling — runs one softmax kernel at one size."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.add_dll_directory(r"D:\anaconda\envs\ainfra\lib\site-packages\torch\lib")
import torch

kernel_name = sys.argv[1]
M, N = 4096, 4096

import softmax_ops
kernel_map = {
    "softmax_kernel":                softmax_ops.softmax_naive,
    "online_softmax_kernel":         softmax_ops.softmax_online,
    "softmax_warp_online_kernel":    softmax_ops.softmax_warp,
    "softmax_warp_float4_kernel":    softmax_ops.softmax_warp_float4,
    "softmax_warp_tiled_kernel":     softmax_ops.softmax_warp_tiled,
}

func = kernel_map[kernel_name]
x = torch.randn(M, N, device='cuda', dtype=torch.float32)

for _ in range(5):
    func(x)
torch.cuda.synchronize()

for _ in range(10):
    func(x)
torch.cuda.synchronize()

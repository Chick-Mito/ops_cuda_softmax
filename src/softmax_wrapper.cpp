#include <torch/extension.h>

torch::Tensor softmax_naive_forward(torch::Tensor input);
torch::Tensor softmax_online_forward(torch::Tensor input);
torch::Tensor softmax_warp_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_naive",  &softmax_naive_forward,  "Naive Softmax (1 thread/row, 3-pass)");
    m.def("softmax_online", &softmax_online_forward, "Online Softmax (1 thread/row, 2-pass)");
    m.def("softmax_warp",   &softmax_warp_forward,   "Warp-Level Online Softmax (32 threads/row)");
}

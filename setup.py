from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='softmax_ops',
    ext_modules=[
        CUDAExtension(
            name='softmax_ops',
            sources=[
                'src/softmax_kernels.cu',
                'src/softmax_wrapper.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode=arch=compute_86,code=sm_86']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

# CUDA Softmax Analysis Project

## Environment
- **Conda**: `D:\anaconda\envs\ainfra` (Python 3.10, PyTorch 2.5.1+cu121)
- **GPU**: RTX 3060 Ti (sm_86)
- **CUDA**: v12.4

## Commands
```bash
conda activate ainfra
python setup.py build_ext --inplace
python bench/benchmark.py
```

## DLL Note
`os.add_dll_directory(r"D:\anaconda\envs\ainfra\lib\site-packages\torch\lib")` required before import.

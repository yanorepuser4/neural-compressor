FP8 OP Swap
=====

This document explains how to use Neural Coder to perform FP8 op swap.

The FP8 op swap performs operation swap from the original op to FP8 op (i.e. `mpepu` FP8 emulator's op). It includes two categories: (1) matmul/bmm and (2) add. 

The core algorithm code is in `neural_coder/coders/tools/fp8.py`, and the API interface code is in `neural_coder/interface.py` (which calls the core algorithm).

### matmul/bmm

Changes `torch.matmul, torch.bmm` to `Matmul, BatchMatmul` from `from mpemu.module_wrappers import Matmul, BatchMatmul`.

- Algorithm code: https://github.com/intel/neural-compressor/blob/fp8_adaptor/neural_coder/coders/tools/fp8.py#L17-L99
- API interface code: https://github.com/intel/neural-compressor/blob/fp8_adaptor/neural_coder/interface.py#L256-L281

### add

Changes `+` or `+=` to `EltwiseAdd` from `from mpemu.module_wrappers import EltwiseAdd`.

- Algorithm code: https://github.com/intel/neural-compressor/blob/fp8_adaptor/neural_coder/coders/tools/fp8.py#L102-L246
- API interface code: https://github.com/intel/neural-compressor/blob/fp8_adaptor/neural_coder/interface.py#L283-L308

## Use Guide (BKC)

The features are called `fp8_matmul_swap` and `fp8_add_swap` in Neural Coder. So for example, if you want to swap "matmul", then use `fp8_matmul_swap`.

The usage is simple. If you want to swap "add (+, +=)" op for "transformers", then first git clone it as a folder, and then execute below Python code (in the directory where you git clone):

```
from neural_coder import enable
enable(code="transformers", features=["fp8_add_swap"], consider_imports=False, overwrite=True)
```

So, a BKC to get the patch looks like this:

```
git clone https://github.com/huggingface/transformers.git
python -c 'from neural_coder import enable; enable(code="transformers", features=["fp8_add_swap"], consider_imports=False, overwrite=True)'
cd transformers
git diff
```

## Note

1. FP8 op swap feature is only currently located in `fp8_adaptor` branch, not in `master` branch, so you need to git clone INC, check out `fp8_adaptor` branch, and build from source in order to use it.
2. If you need to swap both "matmul" and "add", then simply execute both Python commands, like this:
```
git clone https://github.com/huggingface/transformers.git
python -c 'from neural_coder import enable; enable(code="transformers", features=["fp8_matmul_swap"], consider_imports=False, overwrite=True)'
python -c 'from neural_coder import enable; enable(code="transformers", features=["fp8_add_swap"], consider_imports=False, overwrite=True)'
cd transformers
git diff
```

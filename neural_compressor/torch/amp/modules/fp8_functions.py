import os
import torch
import habana_frameworks.torch.hpex
from torch.nn import functional as F
from neural_compressor.common import logger


_F_linear = F.linear
_torch_matmul = torch.matmul
_torch_bmm = torch.bmm

DATA_TYPE = torch.float8_e4m3fn
# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240*0.9, dtype=torch.float).to('hpu')
E5M2_AMAX = torch.tensor(57344*0.9, dtype=torch.float).to('hpu')


def fp8_linear_forward(input, weight, bias):
    dtype_amax = E4M3_AMAX if DATA_TYPE == torch.float8_e4m3fn else E5M2_AMAX
    use_amax = False if os.getenv('PT_USE_FP8_AMAX') is None else True
    out_dtype = torch.float32
    input_raw = input
    input = input.view((-1, weight.shape[-1]))
    if use_amax:
        input_scale = dtype_amax / input.abs().max()
        weight_scale = dtype_amax / weight.abs().max()
        input_scale_inv = 1.0 / input_scale
        weight_scale_inv = 1.0 / weight_scale
    else:
        input_scale, weight_scale = None, None
        input_scale_inv, weight_scale_inv = None, None
    input = torch.ops.hpu.cast_to_fp8_v2(input, input_scale_inv, False, False, DATA_TYPE)[0]
    weight = torch.ops.hpu.cast_to_fp8_v2(weight, weight_scale_inv, False, False, DATA_TYPE)[0]
    out = torch.ops.hpu.fp8_gemm_v2(
        input,
        False,
        weight,
        True,
        None,
        out_dtype,
        input_scale_inv, # inv is used for recover scale
        weight_scale_inv,
        bias,
        False,
    )
    return out.view(-1, *input_raw.shape[1:-1], out.shape[-1])


def fp8_matmul(input1, input2):
    dtype_amax = E4M3_AMAX if DATA_TYPE == torch.float8_e4m3fn else E5M2_AMAX
    use_amax = False if os.getenv('PT_USE_FP8_AMAX') is None else True
    out_dtype = torch.float32
    if use_amax:
        input1_scale = dtype_amax / input1.data.abs().max()
        input2_scale = dtype_amax / input2.data.abs().max()
        input1_scale_inv = 1.0 / input1_scale
        input2_scale_inv = 1.0 / input2_scale
    else:
        input1_scale, input2_scale = None, None
        input1_scale_inv, input2_scale_inv = None, None
    input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale, False, False, DATA_TYPE)[0]
    input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale, False, False, DATA_TYPE)[0]
    out = torch.ops.hpu.fp8_gemm_v2(
        input1,
        False,
        input2,
        False,
        None,
        out_dtype,
        input1_scale_inv, # inv is used for recover scale
        input2_scale_inv,
        None,
        False,
    )
    return out


def replace_func(dtype):
    global DATA_TYPE
    DATA_TYPE = dtype
    F.linear = fp8_linear_forward
    torch.matmul = fp8_matmul
    torch.bmm = fp8_matmul
    logger.debug("F.linear and torch.matmul are replaced with the fp8 one")


def recover_func():
    F.linear = _F_linear
    torch.matmul = _torch_matmul
    torch.bmm = _torch_bmm
    logger.debug("F.linear and torch.matmul are recovered")

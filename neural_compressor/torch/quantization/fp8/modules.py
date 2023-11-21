import os
import torch
import habana_frameworks.torch.hpex
import habana_frameworks.torch.core as htcore
from torch.nn import functional as F
from neural_compressor.common import logger


# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240*0.9, dtype=torch.float).to('hpu')
E5M2_AMAX = torch.tensor(57344*0.9, dtype=torch.float).to('hpu')


class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, org_module, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        # attributes
        org_module.to('hpu')
        self.use_amax = True
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_dtype = torch.int8
        self.out_dtype = torch.float32
        self.register_buffer(
            'weight', 
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            )
        )
        self.register_buffer(
            'bias', 
            torch.empty(
                self.out_features,
                device="hpu",
                dtype=self.out_dtype,
            ) 
        )
        # user configuration
        # scale = HF_max /amax
        if self.use_amax:
            self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
            self.weight_scale_inv = torch.reciprocal(self.weight_scale)
        else:
            self.weight_scale = None
            self.weight_scale_inv = None
        self.weight = torch.ops.hpu.cast_to_fp8_v2(
            org_module.weight.data, self.weight_scale, False, False, self.dtype
        )[0]

        if org_module.bias is not None:
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view((-1, self.in_features))
        if inp.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if self.use_amax:
                input_scale = self.dtype_amax / inp.abs().max()
                input_scale_inv = torch.reciprocal(input_scale)
            else:
                input_scale, input_scale_inv= None, None
            inp = torch.ops.hpu.cast_to_fp8_v2(inp, input_scale, False, False, self.dtype)[0]
        else:
            input_scale_inv = None
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            input_scale_inv, # inv is used for recover scale
            self.weight_scale_inv,
            self.bias,
            False,
        )
        out = out.view(-1, *org_middle_shape, out.shape[-1])
        htcore.mark_step()
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, format={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.dtype,
        )

class FP8DynamicMatmul(torch.nn.Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.dtype = dtype
        self.use_amax = True
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.out_dtype = torch.float32

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        # process input1
        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if self.use_amax:
                input1_scale = self.dtype_amax / input1.data.abs().max()
                input1_scale_inv = torch.reciprocal(input1_scale)
            else:
                input1_scale, input1_scale_inv = None, None
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale, False, False, self.dtype)[0]
        else:
            # skip cast for input1
            input1_scale, input1_scale_inv = None, None
        # process input2
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if self.use_amax:
                input2_scale = self.dtype_amax / input2.data.abs().max()
                input2_scale_inv = torch.reciprocal(input2_scale)
            else:
                input2_scale, input2_scale_inv = None, None
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale, False, False, self.dtype)[0]
        else:
            # skip cast for input2
            input2_scale, input2_scale_inv = None, None
        # calculate
        out = torch.ops.hpu.fp8_gemm_v2(
            input1,
            False,
            input2,
            False,
            None,
            self.out_dtype,
            input1_scale_inv, # inv is used for recover scale
            input2_scale_inv,
            None,
            False,
        )
        htcore.mark_step()
        return out

    def extra_repr(self) -> str:
        return 'format={}'.format(self.dtype)


class FP8DynamicBatchMatmul(FP8DynamicMatmul):
    pass


class FP8Linear(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        # attributes
        org_module.to('hpu')
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_dtype = torch.int8
        self.out_dtype = torch.float32
        self.register_buffer(
            'weight', 
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            )
        )
        self.register_buffer(
            'bias', 
            torch.empty(
                self.out_features,
                device="hpu",
                dtype=self.out_dtype,
            ) 
        )
        assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            'scale', 
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=self.out_dtype,
            ) 
        )
        self.scale_inv = torch.reciprocal(self.scale)

        self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
        self.weight_scale_inv = torch.reciprocal(self.weight_scale)
        self.weight = torch.ops.hpu.cast_to_fp8_v2(
            org_module.weight.data, self.weight_scale, False, False, self.dtype
        )[0]

        if org_module.bias is not None:
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view((-1, self.in_features))
        inp = torch.ops.hpu.cast_to_fp8_v2(inp, self.scale, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            self.scale_inv, # inv is used for recover scale
            self.weight_scale_inv,
            self.bias,
            False,
        )
        out = out.view(-1, *org_middle_shape, out.shape[-1])
        htcore.mark_step()
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, scale={}, format={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.scale, self.dtype,
        )


class FP8Matmul(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        org_module.to('hpu')
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.out_dtype = torch.float32
        assert hasattr(org_module, "scale") and hasattr(org_module, "scale1"), \
                "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            'scale', 
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=self.out_dtype,
            ) 
        )
        self.register_buffer(
            'scale1', 
            torch.tensor(
                org_module.scale1,
                device="hpu",
                dtype=self.out_dtype,
            ) 
        )
        self.input1_scale_inv = torch.reciprocal(self.scale)
        self.input2_scale_inv = torch.reciprocal(self.scale1)

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, self.scale, False, False, self.dtype)[0]
        else:
            self.input1_scale_inv = None
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, self.scale1, False, False, self.dtype)[0]
        else:
            self.input2_scale_inv = None
        out = torch.ops.hpu.fp8_gemm_v2(
            input1,
            False,
            input2,
            False,
            None,
            self.out_dtype,
            self.input1_scale_inv, # inv is used for recover scale
            self.input2_scale_inv,
            None,
            False,
        )
        htcore.mark_step()
        return out

    def extra_repr(self) -> str:
        return 'scales={}, format={}'.format(
            (self.scale, self.scale1), self.dtype,
        )


class FP8BatchMatmul(FP8Matmul):
    pass



class FP8Cast(torch.nn.Module):
    def __init__(self, org_module=None, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        self.out_dtype = torch.float32
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        if org_module is not None:
            org_module.to('hpu')
            assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Cast."
            self.register_buffer(
                'scale', 
                torch.tensor(
                    org_module.scale,
                    device="hpu",
                    dtype=self.out_dtype,
                ) 
            )
            self.scale = None # due to next matmul doesn't know this scale
        else:
            self.scale = None

    def forward(self, input):
        if input.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            out = torch.ops.hpu.cast_to_fp8_v2(input, self.scale, False, False, self.dtype)[0]
        else:
            out = input
        htcore.mark_step()
        return out

    def extra_repr(self) -> str:
        return 'scales={}, format={}'.format(
            (self.scale), self.dtype,
        )


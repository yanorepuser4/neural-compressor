#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch.nn.Module Class Defination."""
# Note: Do not import this file unless you have already imported torch, 
# since the model classes inherit torch.nn.Module.
import math
import torch
from torch.nn import functional as F
from torch.autograd import Function
from .weight_only import quant_weight
from packaging.version import Version
from neural_compressor.utils import logger
from neural_compressor.utils.logger import level, DEBUG


def get_torch_version():
    try:
        torch_version = torch.__version__.split('+')[0]
    except ValueError as e:  # pragma: no cover
        assert False, 'Got an unknown version of torch: {}'.format(e)
    version = Version(torch_version)
    return version

PT_VERSION = get_torch_version().release


class QDQLinear(torch.nn.Module):
    def __init__(self, module, scale=1, zero_point=0, dtype=torch.quint8):
        super().__init__()
        if PT_VERSION < Version("1.13.0").release:
            import torch.nn.quantized as nnq
        else:
            import torch.ao.nn.quantized as nnq
        self.add_module('quant', nnq.Quantize(scale, zero_point, dtype))
        self.add_module('dequant', nnq.DeQuantize())
        self.add_module('module', module)
        self.qdq_weight()
     
    @property
    def weight(self):
        return self.module.weight

    def forward(self, X):
        X = self.quant(X)
        X = self.dequant(X)
        X = self.module(X)
        return X

    def qdq_weight(self):
        # update weight w/ QDQ
        from .smooth_quant import quant_dequant_w
        weith_qdq = quant_dequant_w(self.module)
        self.module.weight = torch.nn.Parameter(weith_qdq)


class QDQLayer(torch.nn.Module):
    def __init__(self, module, input_scale=None) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.input_scale = input_scale
    
    def forward(self, X):
        if self.input_scale is not None:
            X = torch.mul(X, self.input_scale)
        X = self.quant(X)
        X = self.module(X)
        X = self.dequant(X)
        return X


def _wrap_lwq_layer(model, lwq_layers, op_cfgs):
    from .layer_wise_quant.utils import get_module, update_module
    from torch.quantization import prepare, convert
    for name, input_scale in lwq_layers.items():
        qconifg = op_cfgs.module_name_qconfigs.get(name + '.module')
        module = get_module(model, name)
        new_model = QDQLayer(module, input_scale)
        new_model.qconfig = qconifg
        new_model = prepare(new_model)
        new_model = convert(new_model)
        update_module(model, name, new_model)
    return model


class SQLinearWrapper(torch.nn.Module):
    def __init__(self, module, input_scale, input_minmax, alpha=0.5, dtype=torch.quint8):
        super().__init__()
        self.register_buffer('input_scale', input_scale)
        self.alpha = alpha
        self.dtype = dtype
        # calculate and only save scale, zero_point to avoid memory usage
        self.scale, self.zero_point = self._calculate_qparams(input_scale, input_minmax, dtype)
        self.add_module('sq_linear', module)
        self._update_sq_linear()
        self.ipex = False  # a flag used for ipex inference
    
    @property
    def weight(self):
        return self.sq_linear.weight

    def forward(self, X):
        if self.ipex:
            X = self.sq_linear(X)
        else:
            X = torch.mul(X, self.input_scale)
            X = self.sq_linear(X)
        return X

    def _calculate_qparams(self, input_scale, input_minmax, dtype=torch.quint8):
        # calculate scale and zero_point
        if dtype == torch.quint8:
            quant_min, quant_max = 0, 255
        min_val = torch.min(input_minmax[0] * input_scale)
        max_val = torch.max(input_minmax[1] * input_scale)
        # work when min_val bigger than zero.
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, torch.tensor([torch.finfo(torch.float32).eps]))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def _get_weight_scale(self):
        # get weight scale and zero_point
        from torch.ao.quantization.observer import default_per_channel_weight_observer
        obs = default_per_channel_weight_observer()
        obs(self.sq_linear.weight)
        scale, _ = obs.calculate_qparams()
        return scale

    def _update_sq_linear(self):
        # remove mul and reset sq_linear for ipex inference
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.sq_linear.weight /= scale

    def _recover_sq_linear(self):
        # remove mul and reset sq_linear for ipex inference
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.sq_linear.weight *= scale


def _wrapper_sq_linear(tmp_model, input_scale_dict):
    """Help function to generate a fake SmoothQuant model for loading weights"""
    class SQLinearWrapper(torch.nn.Module):
        def __init__(self, module, input_scale):
            super().__init__()
            self.register_buffer('input_scale', input_scale)
            self.add_module('sq_linear', module)

        def forward(self, X):
            X = torch.mul(X, self.input_scale)
            X = self.sq_linear(X)
            return X

    module_name_list = input_scale_dict.keys()
    from .smooth_quant import get_module, set_module
    for name in module_name_list:
        module = get_module(tmp_model, name)
        input_scale = input_scale_dict[name]
        new_module = SQLinearWrapper(module, input_scale)
        set_module(tmp_model, name, new_module)
    return tmp_model


def _wrapper_qdq_linear(tmp_model, module_name_list=[]):
    """Help function to generate a fake QDQ model for loading weights"""
    from .smooth_quant import get_module, set_module
    for name in module_name_list:
        module = get_module(tmp_model, name)
        new_module = QDQLinear(module)
        set_module(tmp_model, name, new_module)
    return tmp_model


class WeightOnlyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bits, groupsize, 
                 dtype='int', zp=False, bias=False, scale_dtype=torch.float32, 
                 compression_dtype=torch.int32, compression_dim=1,
                 gptq_perm=False, device='cpu'):
        super().__init__()
        self.dtype = dtype
        if 'int' not in self.dtype:  # for nf4, fp4
            from neural_compressor.adaptor.torch_utils.weight_only import FLOAT_MAPPING, INT_MAPPING
            float_list = FLOAT_MAPPING[self.dtype]
            int_list = INT_MAPPING[self.dtype]
            self.int2float_mapping = {}
            for k, v in zip(int_list, float_list):
                self.int2float_mapping[k] = v
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else in_features
        self.compression_dim = compression_dim
        assert compression_dtype in [torch.int8, torch.int16, torch.int32, torch.int64], \
                        "Only support torch.int8|16|32|64 as compressed dtype."
        dtype_bits_mapping = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        self.compressed_dtype = compression_dtype
        self.float_type = scale_dtype
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], "Only support 0 or 1 as compression dimension, " +\
                                          "0 is output channel, 1 is input channel."
        self.register_buffer(
            'scale', 
            torch.zeros(
                (out_features, math.ceil(in_features / self.groupsize)), 
                dtype=self.float_type,
            ).to(device)
        )
        if compression_dim == 1:
            self.register_buffer(
                'packed_weight', 
                torch.zeros(
                    (out_features, math.ceil(in_features / self.n_pack)), 
                    dtype=self.compressed_dtype,
                ).to(device)
            )
            if zp:
                self.register_buffer(
                    'packed_zp', 
                    torch.zeros(
                        (self.out_features, math.ceil(self.in_features / self.groupsize / self.n_pack)), 
                        dtype=self.compressed_dtype,
                    ).to(device)
                )
        else:
            self.register_buffer(
                'packed_weight', 
                torch.zeros(
                    (math.ceil(out_features / self.n_pack), in_features), 
                    dtype=self.compressed_dtype,
                ).to(device)
            )
            if zp:
                self.register_buffer(
                    'packed_zp', 
                    torch.zeros(
                        (
                            math.ceil(self.out_features / self.n_pack), 
                            math.ceil(self.in_features / self.groupsize)
                        ), 
                        dtype=self.compressed_dtype,
                    ).to(device)
                )
        if bias:
            self.register_buffer(
                'bias', torch.zeros(self.out_features, dtype=self.float_type).to(device)
            )
        else:
            self.bias = None
        if gptq_perm:
            self.register_buffer(
                'gptq_perm', torch.zeros(in_features, dtype=torch.int32).to(device)
            )
        else:
            self.gptq_perm = None

    def pack(self, int_weight, scale, zp, bias, gptq_perm=None):
        int_weight = int_weight.to(self.device)
        if bias is not None:
            assert hasattr(self, 'bias'), "bias is not set when initializing."
            self.bias = bias.type(self.float_type).to(self.device)
        if gptq_perm is not None:
            assert hasattr(self, 'gptq_perm'), "gptq_perm is not set when initializing."
            self.gptq_perm = gptq_perm.type(torch.int32).to(self.device)
        assert scale.shape == self.scale.shape, "Scale shape is mismatched."
        self.scale = scale.type(self.float_type).to(self.device)
        if self.compression_dim == 0:
            int_weight = int_weight.T
            self.packed_weight = self.packed_weight.T
        origin_shape = int_weight.shape
        target_shape = self.packed_weight.shape
        assert origin_shape[0] == target_shape[0], "output channels mismatch, please check."
        mask = torch.tensor(2**self.bits - 1, dtype=self.compressed_dtype).to(self.device)

        # pack weight
        for j in range(target_shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = int_weight[:, start: end].type(self.compressed_dtype)
            for e in range(tmp.shape[1]):
                tmp[:, e] &= mask
                tmp[:, e] = tmp[:, e] << (self.bits * e)
                self.packed_weight[:, j] |= tmp[:, e]
        if self.compression_dim == 0:
            self.packed_weight = self.packed_weight.T

        if zp is not None:
            zp = zp.to(self.device)
            if self.compression_dim == 0:
                zp = zp.T
                self.packed_zp = self.packed_zp.T
            assert hasattr(self, 'packed_zp'), "zp is not set when initializing."
            target_shape = self.packed_zp.shape
            for j in range(target_shape[1]):
                start = self.n_pack * j
                end = self.n_pack * (j + 1)
                tmp = zp[:, start: end].type(self.compressed_dtype)
                for e in range(tmp.shape[1]):
                    tmp[:, e] &= mask
                    tmp[:, e] = tmp[:, e] << (self.bits * e)
                    self.packed_zp[:, j] |= tmp[:, e]
            if self.compression_dim == 0:
                self.packed_zp = self.packed_zp.T

    def recover(self):
        logger.debug(f"Recovering {self} weight")
        device = self.scale.device
        mask = torch.tensor(2**self.bits - 1, dtype=self.compressed_dtype).to(device)
        if hasattr(self, 'packed_zp'):
            weight_dtype = torch.uint8
        else:
            weight_dtype = torch.int8
        # unpack weight
        weight = torch.zeros(self.out_features, self.in_features, dtype=weight_dtype).to(device)
        packed_weight = self.packed_weight
        if self.compression_dim == 0:
            weight = weight.T
            packed_weight = packed_weight.T
        origin_shape = weight.shape
        target_shape = packed_weight.shape
        for j in range(target_shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                if index >= origin_shape[1]:
                    continue
                tmp = packed_weight[:, j]
                tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                tmp = tmp >> self.compress_bits - self.bits
                if weight_dtype == torch.uint8:
                    tmp &= mask # remove sign bit
                weight[:, index] = tmp.type(weight_dtype)
        if self.compression_dim == 0:
            weight = weight.T
        if 'int' not in self.dtype:
            new_weight = torch.zeros(self.out_features, self.in_features).to(device)
            for k, v in self.int2float_mapping.items():
                new_weight += torch.where(weight == k, v, 0)
            weight = new_weight
        # unpack zero_point
        if hasattr(self, 'packed_zp'):
            zp_dtype = self.compressed_dtype # to avoid overflow when weight-zp
            zp = torch.zeros(self.scale.shape, dtype=zp_dtype).to(device)
            packed_zp = self.packed_zp
            if self.compression_dim == 0:
                zp = zp.T
                packed_zp = packed_zp.T
            origin_shape = zp.shape
            target_shape = packed_zp.shape
            for j in range(target_shape[1]):
                for e in range(self.n_pack):
                    index = j * self.n_pack + e
                    if index >= origin_shape[1]:
                        continue
                    tmp = packed_zp[:, j]
                    tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                    tmp = tmp >> self.compress_bits - self.bits
                    tmp &= mask
                    zp[:, index] = tmp.type(zp_dtype)
            if self.compression_dim == 0:
                zp = zp.T
            # recover fp32 weight with int_weight, scale, and zero_point
            left_element = self.in_features % self.groupsize 
            if left_element != 0:
                split_index = self.in_features // self.groupsize  * self.groupsize
                weight1 = weight[:, :-split_index].reshape(-1, self.groupsize)
                scale1 = self.scale[:, :-1].reshape(-1, 1)
                zp1 = zp[:, :-1].reshape(-1, 1)
                weight1 = ((weight1 - zp1) * scale1).reshape(self.out_features, -1)
                weight2 = weight[:, -split_index:]
                scale2 = self.scale[:, -1:]
                zp2 = zp[:, -1].reshape(-1, 1)
                weight2 = ((weight2 - zp2) * scale2)
                fp32_weight = torch.cat((weight1, weight2), dim=1)
            else:
                weight = weight.reshape(-1, self.groupsize)
                scale = self.scale.reshape(-1, 1)
                zp = zp.reshape(-1, 1)
                fp32_weight = ((weight - zp) * scale).reshape(self.out_features, -1)
        else:
            # recover fp32 weight with int_weight, scale
            left_element = self.in_features % self.groupsize 
            if left_element != 0:
                split_index = self.in_features // self.groupsize  * self.groupsize
                weight1 = weight[:, :split_index].reshape(-1, self.groupsize)
                scale1 = self.scale[:, :-1].reshape(-1, 1)
                weight1 = (weight1 * scale1).reshape(self.out_features, -1)
                weight2 = weight[:, split_index:]
                scale2 = self.scale[:, -1:]
                weight2 = (weight2 * scale2)
                fp32_weight = torch.cat((weight1, weight2), dim=1)
            else:
                weight = weight.reshape(-1, self.groupsize)
                scale = self.scale.reshape(-1, 1)
                fp32_weight = (weight * scale).reshape(self.out_features, -1)
        if self.gptq_perm is not None:
            invperm = torch.argsort(self.gptq_perm)
            fp32_weight = fp32_weight[:, invperm]
        return fp32_weight

    def forward(self, input):
        if level == DEBUG:
            if not hasattr(self, 'weight'):
                self.weight = self.recover()
            input = input.type(self.weight.dtype)
            logger.debug(f"Calculating {self}")
            return F.linear(input, self.weight, self.bias)
        else:
            weight = self.recover()
            input = input.type(weight.dtype)
            return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bits={}, group_size={}, bias={}'.format(
            self.in_features, self.out_features, self.bits, self.groupsize, self.bias is not None
        )


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization
    """

    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=1024, scheme="asym"):
        """

        As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        return quant_weight(inputs, num_bits, group_size, scheme)

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None, None


class TEQLinearFakeQuant(torch.nn.Module):
    """
    wrapper quantization linear
    """

    def __init__(self, orig_layer, alpha=None, num_bits=4, group_size=-1, scheme="asym"):
        """
        A forward hook to linear module
        :param orig_layer: the original module
        :param alpha: trainable alpha/scale
        :param num_bits: quantization level
        :param group_size: for fine-grained quantization
        """
        super(TEQLinearFakeQuant, self).__init__()
        self.orig_layer = orig_layer
        self.alpha = alpha

        self.num_bits = num_bits
        self.group_size = group_size
        self.scheme = scheme

    def forward(self, x):
        alpha = torch.clip(self.alpha, 1e-5)
        shape_len = len(x.shape) - 1
        shape = (1,) * shape_len + (-1,)
        x = x / alpha.view(shape)
        weight = self.orig_layer.weight
        weight = weight * alpha.unsqueeze(dim=0)
        weight_q = FakeAffineTensorQuantFunction().apply(weight, self.num_bits,
                self.group_size, self.scheme)
        return F.linear(x, weight_q, self.orig_layer.bias)


class MulLinear(torch.nn.Module):
    """Linear wrapper to apply scale to input."""

    def __init__(self, module, input_scale=None):
        """
        A forward hook to save input max of a module
        :param module: the linear module
        :param input_scale: scale for input
        """
        super().__init__()
        if input_scale is None:
            input_scale = torch.empty(module.in_features)
        self.register_buffer('input_scale', input_scale)
        self.add_module('linear', module)

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, weight):
        self.linear.weight = weight

    def forward(self, X):
        X = torch.mul(X, self.input_scale)
        X = self.linear(X)
        return X

    def _update_linear(self):
        # update linear weight with input_scale
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.linear.weight /= scale

    def _recover_linear(self):
        # remove mul and reset sq_linear for ipex inference
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.linear.weight *= scale


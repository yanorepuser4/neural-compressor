# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class ScaleCalculator(torch.nn.Module):
    def __init__(self, shape: int, device):
        super().__init__()
        self.shape = shape
        self.device = device
        tensor1 = torch.ones(shape, device=device) * 0.5
        tensor2 = torch.ones(shape, device=device) * 0.5
        # torch.nn.init.normal_(tensor1)
        # torch.nn.init.normal_(tensor2)
        self.sclae1 = torch.nn.Parameter(tensor1, requires_grad=True)
        self.scale2 = torch.nn.Parameter(tensor2, requires_grad=True)

    def forward(self, x):
        update_scale = torch.clip(self.sclae1, min=0, max=1) / torch.clip(self.scale2, min=1e-5, max=1)
        # TODO: add more complex logic here
        return update_scale

    def get_final_scale(self):
        # TODO: add more complex logic here
        return torch.clip(self.sclae1, min=0, max=1) / torch.clip(self.sclae2, min=1e-5, max=1)


# ScaleCalculatorVanilla
class ScaleCalculatorV(torch.nn.Module):
    def __init__(self, shape: int, device):
        super().__init__()
        self.shape = shape
        self.device = device
        tensor1 = torch.ones(shape, device=device)
        # tensor2 = torch.ones(shape, device=device)
        # torch.nn.init.normal_(tensor1)
        # torch.nn.init.normal_(tensor2)
        self.sclae1 = torch.nn.Parameter(tensor1, requires_grad=True)
        # self.scale2 = torch.nn.Parameter(tensor2, requires_grad=True)

    def forward(self, x):
        update_scale = self.sclae1
        # TODO: add more complicated logic here
        return update_scale

    def get_final_scale(self):
        # TODO: add more complicated logic here
        return self.sclae1


from neural_compressor.utils import logger


class NewMulLinear(torch.nn.Module):
    def __init__(self, module, input_scale=None):
        """A forward hook to save input max of a module
        :param module: the linear module
        :param input_scale: scale for input."""
        super().__init__()
        if input_scale is None:
            input_scale = torch.empty(module.in_features)
        self.register_buffer("input_scale", input_scale)
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            module.weight *= self.input_scale.unsqueeze(dim=0)
        self.add_module("linear", module)
        logger.info(f"NewMulLinear: {module} has been wrapped.")

    def forward(self, X):
        shape_len = len(X.shape) - 1
        inverse_scale_for_x = 1 / self.input_scale
        inverse_scale_new_shape = (1,) * shape_len + (-1,)
        inverse_scale_for_x = inverse_scale_for_x.view(inverse_scale_new_shape)
        X = torch.mul(X, inverse_scale_for_x)
        X = self.linear(X)
        return X

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, weight):
        self.linear.weight = weight

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, bias):
        self.linear.bias = bias


@torch.no_grad()
class TestNewMulLinear:
    def test_new_mul_linear(self):
        in_features = 10
        out_features = 30
        input_scale = torch.rand(in_features)
        module = torch.nn.Linear(in_features, out_features)
        inputs = torch.rand(2, in_features)
        origial_out = module(inputs)
        new_module = NewMulLinear(module, input_scale)

        out = new_module(inputs)
        print(out)
        print(origial_out)
        torch.allclose(out, origial_out)


t = TestNewMulLinear()
t.test_new_mul_linear()

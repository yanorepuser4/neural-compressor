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
        self.sclae = torch.nn.Parameter(torch.ones(shape, device=device))

    def forward(self, x):
        update_scale = self.sclae
        # TODO: add more complicated logic here
        return update_scale

    def get_final_scale(self):
        # TODO: add more complicated logic here
        return self.sclae


# Demos:
# https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
class Polynomial3(torch.nn.Module):
    def __init__(self):
        """In the constructor we instantiate four parameters and assign them as
        member parameters."""
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data.

        We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

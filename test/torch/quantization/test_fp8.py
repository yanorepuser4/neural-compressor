import os
import copy
import shutil
import unittest
import torch
import habana_frameworks.torch.hpex

from neural_compressor.torch.quantization import get_static_qconfig
from neural_compressor.torch.quantization import Matmul, BatchMatmul
from neural_compressor.torch.quantization.fp8 import quantize_dynamic, quantize
from neural_compressor.torch.dtype import float8_e4m3, float8_e5m2
from neural_compressor.common import logger


tmp = torch.ops.hpu.cast_to_fp8_v2(torch.tensor(500).to('hpu'), torch.tensor(1).to('hpu'), False, False)[0]
logger.info(f"max value: {tmp}")


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)
        self.mm = Matmul()
        self.bmm = BatchMatmul()

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        x3 = torch.matmul(inp.T, x2)
        x3 = x3.unsqueeze(0)
        x4 = self.mm(inp.T, x2)
        x4 = x4.unsqueeze(0)
        x5 = torch.bmm(x3, x4)
        x6 = self.bmm(x3, x4)
        out = x5 - x6
        return out


class TestPytorchFP8Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = M().to('hpu')
        self.inp = torch.randn(1, 10).to('hpu')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./.graph_dumps", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_dynamic(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        m = quantize_dynamic(m)
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization MSE:", (fp32_out - fp8_out).pow(2).sum())

    def test_static(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = get_static_qconfig()

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, calib_func=calib_func)
        print(m)
        fp8_out = m(inp)
        print("Static quantization MSE:", (fp32_out - fp8_out).pow(2).sum())


if __name__ == "__main__":
    unittest.main()


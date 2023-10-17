import os
import copy
import shutil
import unittest
import torch

from neural_compressor.torch.amp.modules.fp8_modules import INCFP8Linear
from neural_compressor.torch.amp import autocast
from neural_compressor.torch.dtype import float8_e4m3, float8_e5m2
from neural_compressor.common import logger


tmp = torch.ops.hpu.cast_to_fp8_v2(torch.tensor(500).to('hpu'), torch.tensor(1).to('hpu'), False, False)[0]
logger.debug(f"max value: {tmp}")
tmp = torch.ops.hpu.cast_from_fp8(tmp, torch.tensor(1).to('hpu'), torch.float32)
logger.debug(f"max value: {tmp}")


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        x3 = torch.matmul(inp.T, x2)
        return x3


class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
    approach = "weight_only"

    @classmethod
    def setUpClass(self):
        self.model = M().to('hpu')
        self.inp = torch.randn(1, 10).to('hpu')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autocast(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        with autocast('hpu', dtype=torch.bfloat16):
            bf16_out = m(inp)

        with autocast('hpu', dtype=float8_e5m2):
            e5m2_out = m(inp)
            print(e5m2_out)

        # wrong output, e4m2 env varible should be set at the beginning
        with autocast('hpu', dtype=float8_e4m3):
            e4m3_out = m(inp)
            print(e4m3_out)

    def test_module_wrapper(self):
        from neural_compressor.torch.amp.modules.fp8_modules import reset_FP8_linear
        m = copy.deepcopy(self.model)
        m = reset_FP8_linear(m)
        inp = self.inp
        e5m2_out = m(inp)


if __name__ == "__main__":
    unittest.main()


import unittest
import numpy as np
from neural_compressor.common.tunner.sampler import BaseSampler, FallbackSampler, AccumulatedFallbackSampler
from neural_compressor.common.tunner.search_space import HyperParams

class TestSampler(unittest.TestCase):
    
    def test_base_sampler(self):
        # test base sampler
        print(f"{'='*10} test base sampler {'='*10}")
        sq_alpha = np.linspace(0.1, 0.4, 3).tolist()
        sq_hp = HyperParams(name="sq_alpha", params_space={"alpha": sq_alpha})
        sq_sampler = BaseSampler(sq_hp, name="sq_alpha", priority=1)
        for config in sq_sampler:
            print(config)

    def test_fallback_sampler(self):
        # test fallback sampler
        print(f"{'='*10} test fallback sampler {'='*10}")
        op_lst = ["MatMul_1", "MatMul_2", "MatMul_3"]
        op_hp = HyperParams(name="op", params_space={"op": op_lst})
        fallback_sampler = FallbackSampler(op_hp)
        for config in fallback_sampler:
            print(config)

    def test_accumulated_fallback_sampler(self):
        # test accumulated fallback sampler
        print(f"{'='*10} test accumulated fallback sampler {'='*10}")
        op_lst = ["MatMul_1", "MatMul_2", "MatMul_3"]
        op_hp = HyperParams(name="op", params_space={"op": op_lst})
        fallback_sampler = FallbackSampler(op_hp)
        accumulated_fallback_sampler = AccumulatedFallbackSampler(op_hp, name="accumulated_fallback_sampler", dependence_samplers=[fallback_sampler])
        for config in accumulated_fallback_sampler:
            print(config)
    
    def test_accumulated_fallback_sampler_init_failed(self):
        # test accumulated fallback sampler
        print(f"{'='*10} test accumulated fallback sampler init failed {'='*10}")
        op_lst = ["MatMul_1", "MatMul_2", "MatMul_3"]
        op_hp = HyperParams(name="op", params_space={"op": op_lst})
        self.assertRaises(AssertionError, AccumulatedFallbackSampler, op_hp, name="accumulated_fallback_sampler", dependence_samplers=[])



if __name__ == "__main__":
    unittest.main()


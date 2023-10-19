import unittest


class TestOrtWithTuner(unittest.TestCase):
    def test_interface(self):
        """A simple demo to test the interface between ORT and tuner."""
        from neural_compressor.onnxruntime.quantization import quantize
        from neural_compressor.onnxruntime.utility import (
            get_default_quant_config,
            get_default_quant_with_tuning_criterion,
            get_demo_model,
            get_dummy_dataloader,
        )

        fp32_model = get_demo_model()
        quant_config = get_default_quant_config()

        # tuning
        quant_config = get_default_quant_with_tuning_criterion()
        dummy_dataloader = get_dummy_dataloader()
        q_model = quantize(fp32_model, calib_dataloader=dummy_dataloader, quant_config=quant_config)
        self.assertIsNotNone(q_model)

    def test_with_eval_func(self):
        """A simple demo to test the interface between ORT and tuner."""
        from neural_compressor.onnxruntime.quantization import quantize
        from neural_compressor.onnxruntime.utility import (
            get_default_quant_config,
            get_default_quant_with_tuning_criterion,
            get_demo_model,
            get_dummy_dataloader,
        )

        fp32_model = get_demo_model()
        quant_config = get_default_quant_config()

        def fake_eval(model):
            return 1.0

        # tuning
        from neural_compressor.common.config import AccuracyCriterion, TuningCriterion

        accuracy_criterion = AccuracyCriterion()
        tuning_criterion = TuningCriterion()
        quant_config = get_default_quant_with_tuning_criterion()
        dummy_dataloader = get_dummy_dataloader()
        q_model = quantize(
            fp32_model,
            calib_dataloader=dummy_dataloader,
            quant_config=quant_config,
            tuning_criterion=tuning_criterion,
            accuracy_criterion=accuracy_criterion,
            eval_func=fake_eval,
        )
        self.assertIsNone(q_model)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2023 Intel Corporation
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


from neural_compressor.onnxrt.utility import FAKE_EVAL_RESULT, FakeModel
from neural_compressor.common import logger


class ORTQuantizer:
    def __init__(
        self,
        fp32_model,
        calib_dataloader,
        quant_config,
        tuning_config=None,
        eval_func=None,
        **kwargs
    ) -> None:
        self.fp32_model = fp32_model
        self.calib_dataloader = calib_dataloader
        self.quant_config = quant_config
        self.tuning_config = tuning_config
        self._eval_func = eval_func

    def need_tuning(self) -> bool:
        """Whether the quantizer needs tuning."""
        return (self.tuning_config is not None) and (self._eval_func is not None)


    def quantize(self):
        if self.need_tuning():
            return self.tuning()
        else:
            return self.internal_quantize(q_config=self.quant_config)

    def internal_quantize(self, q_config):
        logger.info("Quantizing model with config: {}".format(q_config))
        return FakeModel()

    def evaluate(self, model) -> float:
        """Evaluate the model and return the accuracy."""
        logger.info("Evaluating model: {}".format(model))
        self._eval_func(model)
        return FAKE_EVAL_RESULT


    def tuning(self):
        """Try to find the best quantization config and return the corresponding model.

        Steps:
            1. Initialize a tuner.
            2. Register a set of custom samplers(search space).
            3. Traverse the search space and return the best model.

        Returns:
            Return best model if found, otherwise return None.
        """
        tuner = self.init_tuner()
        best_model = tuner.traverse(self)
        return best_model

    def init_tuner(self):
        from neural_compressor.common.tune import Tuner

        tuner = Tuner(
            baseline_model=self.fp32_model,
            tuning_config=self.tuning_config,
            tuning_criterion=self.quant_config,
            eval_func=self._eval_func,
        )
        return tuner



def quantize(fp32_model, quant_config):
    """
    The main entry for quantize-only.
    """
    pass



def autotune(
    fp32_model,
    quant_config,
    calib_dataloader=None,
    eval_func=None,
    tuning_config=None,
    **kwargs
):
    """The main entrance for auto-tune.

    Args:
        fp32_model: _description_
        quant_config: _description_
        calib_dataloader: _description_. Defaults to None.
        calib_func: _description_. Defaults to None.
        eval_func: _description_. Defaults to None.
        eval_metric: _description_. Defaults to None.
        tuning_criterion: _description_. Defaults to None.
        accuracy_criterion: _description_. Defaults to None.

    Returns:
        Quantized model.
    """
    quantizer = ORTQuantizer(
        fp32_model=fp32_model,
        calib_dataloader=calib_dataloader,
        quant_config=quant_config,
        tuning_config=tuning_config,
        eval_func=eval_func,
        **kwargs
    )
    return quantizer.quantize()

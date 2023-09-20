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

from enum import Enum

import onnxruntime as ort

from neural_compressor.onnxruntime.utility import FAKE_EVAL_RESULT, FakeModel
from neural_compressor.quantize import IncQuantizer
from neural_compressor.utils import logger


class OptimizationLevel(Enum):
    """Optimization level for ORT graph."""

    DISABLED = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    BASIC = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    EXTENDED = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ALL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class ORTQuantizer(IncQuantizer):
    def __init__(self, fp32_model, calib_dataloader, quant_config, tuning_config=None) -> None:
        self.fp32_model = fp32_model
        self.calib_dataloader = calib_dataloader
        self.quant_config = quant_config
        self.tuning_config = tuning_config

    def need_tuning(self) -> bool:
        """Whether the quantizer needs tuning."""
        return self.tuning_config is not None

    def _parse_user_config_into_q_config(self, quant_config):
        return quant_config

    def quantize(self):
        if self.need_tuning():
            return self.tuning()
        else:
            return self.internal_quantize(q_config=self._parse_user_config_into_q_config(self.quant_config))

    def internal_quantize(self, q_config):
        logger.info("Quantizing model with config: {}".format(q_config))
        return FakeModel()

    def evaluate(self, model) -> float:
        """Evaluate the model and return the accuracy."""
        logger.info("Evaluating model: {}".format(model))
        return FAKE_EVAL_RESULT

    def tuning(self):
        """Try to find the best quantization config and return the corresponding model.

        Steps:
            1. Initialize a strategy.
            2. Register a set of custom samplers(search space).
            3. Traverse the search space and return the best model.

        Returns:
            Return best model if found, otherwise return None.
        """
        strategy = self.init_strategy()
        self.register_custom_samplers(strategy)
        best_model = strategy.traverse(self)
        return best_model

    def init_strategy(self):
        from neural_compressor.common.strategy import Strategy

        strategy = Strategy(self.fp32_model, self.tuning_config)
        return strategy

    def register_custom_samplers(self, strategy) -> None:
        """Register a set of custom passes.

        Args:
            strategy (Strategy): The strategy to register custom passes.
        """

        from neural_compressor.common.strategy.sampler import BaseSampler
        from neural_compressor.common.strategy.search_space import HyperParams

        # add graph optimization level sampler
        opt_level_hp = HyperParams(
            name="ort_graph_opt_level",
            params_space={
                "ort_graph_opt_level": [
                    OptimizationLevel.DISABLED,
                    OptimizationLevel.BASIC,
                    OptimizationLevel.EXTENDED,
                    OptimizationLevel.ALL,
                ]
            },
        )
        strategy.add_sampler(BaseSampler(hp=opt_level_hp, name="ort_graph_opt_level", priority=100))
        # add sq sampler
        # assume the sq alpha is a list of float
        self.quant_config.sq_alpha = [0.1, 0.2, 0.3]
        sq_hp = HyperParams(name="sq_alpha", params_space={"alpha": self.quant_config.sq_alpha})
        strategy.add_sampler(BaseSampler(sq_hp, name="sq_alpha", priority=1))


def quantize(fp32_model, calib_dataloader, quant_config, tuning_config):
    quantizer = ORTQuantizer(fp32_model, calib_dataloader, quant_config, tuning_config)
    return quantizer.quantize()

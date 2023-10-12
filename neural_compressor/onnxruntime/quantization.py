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

from neural_compressor.common.strategy.sampler import BaseSampler
from neural_compressor.common.strategy.search_space import HyperParams

from neural_compressor.common.config import (
    basic_sampler_config, 
    smooth_quant_sampler_config, 
    op_type_wise_sampler_config,
    optimization_level_sampler_config)



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
    
    def report_result(self, model):
        """Evaluate the current model and report the result to strategy.
        
        Args:
            model: the quantized model or fp32 model.
        """
        pass

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
        ############################################
        # add graph optimization level sampler
        ############################################
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
        opt_level_sampler = BaseSampler(
            hp=opt_level_hp,
            name="ort_graph_opt_level",
            priority=optimization_level_sampler_config.priority
            )
        strategy.add_sampler(opt_level_sampler)
        
        ############################################
        # add sq sampler
        ############################################
        # assume the sq alpha is a list of float
        sq_alpha = smooth_quant_sampler_config.alpha
        sq_hp = HyperParams(name="sq_alpha", params_space={"alpha": sq_alpha})
        sq_sampler = BaseSampler(
            hp=sq_hp,
            name="sq_alpha",
            priority=smooth_quant_sampler_config.priority
            )
        strategy.add_sampler(sq_sampler)


def quantize(
    fp32_model,
    quant_config,
    calib_dataloader=None,
    calib_func=None,
    eval_func=None,
    eval_metric=None,
    tuning_config=None,
    **kwargs):
    """ The main entrance for user to quantize model.
    
    Args:
        fp32_model: _description_
        quant_config: _description_
        calib_dataloader: _description_. Defaults to None.
        calib_func: _description_. Defaults to None.
        eval_func: _description_. Defaults to None.
        eval_metric: _description_. Defaults to None.
        tuning_config: _description_. Defaults to None.

    Returns:
        Quantized model.
    """
    quantizer = ORTQuantizer(fp32_model, calib_dataloader, quant_config, tuning_config)
    return quantizer.quantize()

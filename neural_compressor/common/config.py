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


class BaseConfig():
  def __init__(self):
      self.scope = {}

  def __add__(self, obj):
      self.scope = {**self.scope, **obj.scope}

class FP32Config(BaseConfig):
  def __init__(self, scope):
      super().__init__()
      self.scope = {'fp32': scope}

class FP8QuantConfig(BaseConfig):
  def __init__(self, scheme, scope):
      super().__init__()
      assert(scheme in ['fp8_e4m3', 'fp8_e3m4', 'fp8_e5m2']), "The FP8 configuration is wrong! Please double check."
      self.scope = {scheme: scope}


from neural_compressor.common.constant import MAX_TRIALS_EACH_SAMPLER


# Place configs that can be applied to all adaptors here.
# These configs are designed for advanced user to customized tuning process with more flexibility.
class BasicSamplerConfig:
    priority = 100
    max_trials = MAX_TRIALS_EACH_SAMPLER


class SmoothQuantSamplerConfig(BasicSamplerConfig):
    alpha = [0.5]


class OpTypeWiseSamplerConfig(BasicSamplerConfig):
    priority = 10
    op_types = []


class OptimizationLevelSamplerConfig(BasicSamplerConfig):
    priority = 100
    optimization_levels = []


basic_sampler_config = BasicSamplerConfig()
smooth_quant_sampler_config = SmoothQuantSamplerConfig()
op_type_wise_sampler_config = OpTypeWiseSamplerConfig()
optimization_level_sampler_config = OptimizationLevelSamplerConfig()


class TuningConfig:
    """Tuning Config class.

    #TODO(Yi) should we use TuningConfig to replace TuningCriterion and AccuracyCriterion?
    TuningConfig class is used to configure the trials order, accuracy constraint and exit policy.
    Note: The TuningConfig class merges the main functionalities of TuningCriterion and AccuracyCriterion of INC 2.x.

    Attributes for trials order:
        - quant_level
        - sampler[New introduced to replace `strategy`]
    Attributes for accuracy criterion:
        - higher_is_better
        - relative
        - tolerable_loss
    Attributes for exit policy:
        - objective
        - timeout
        - max_trials
    """

    pass


class TuningCriterion:
    """Class for Tuning Criterion.

    Args:
        strategy: Strategy name used in tuning. Please refer to docs/source/tuning_strategies.md.
        strategy_kwargs: Parameters for strategy. Please refer to docs/source/tuning_strategies.md.
        objective: String or dict. Objective with accuracy constraint guaranteed. String value supports
                  'performance', 'modelsize', 'footprint'. Default value is 'performance'.
                   Please refer to docs/source/objective.md.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.

    Example::
        from neural_compressor.config import TuningCriterion

        tuning_criterion=TuningCriterion(
            timeout=0,
            max_trials=100,
            strategy="basic",
            strategy_kwargs=None,
        )
    """

    def __init__(self, strategy="basic", strategy_kwargs=None, timeout=0, max_trials=100, objective="performance"):
        """Init a TuningCriterion object."""
        self.strategy = strategy
        self.timeout = timeout
        self.max_trials = max_trials
        self.objective = objective
        self.strategy_kwargs = strategy_kwargs


tuning_criterion = TuningCriterion()


class AccuracyCriterion:
    """Class of Accuracy Criterion.

    Args:
        higher_is_better(bool, optional): This flag indicates whether the metric higher is the better.
                                          Default value is True.
        criterion:(str, optional): This flag indicates whether the metric loss is 'relative' or 'absolute'.
                                   Default value is 'relative'.
        tolerable_loss(float, optional): This float indicates how much metric loss we can accept.
                                         Default value is 0.01.

    Example::

        from neural_compressor.config import AccuracyCriterion

        accuracy_criterion = AccuracyCriterion(
            higher_is_better=True,  # optional.
            criterion='relative',  # optional. Available values are 'relative' and 'absolute'.
            tolerable_loss=0.01,  # optional.
        )
    """

    def __init__(self, higher_is_better=True, criterion="relative", tolerable_loss=0.01):
        """Init an AccuracyCriterion object."""
        self.higher_is_better = higher_is_better
        self.criterion = criterion
        self.tolerable_loss = tolerable_loss


accuracy_criterion = AccuracyCriterion()

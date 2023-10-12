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

from abc import abstractmethod
from enum import Enum
from typing import Any, List, Callable

from neural_compressor.common.strategy.search_space import HyperParams
from neural_compressor.common.strategy.utility import FakeTuningConfig

from neural_compressor.common.constant import MAX_TRIALS_EACH_SAMPLER

class SamplingAlgo(Enum):
    Sequential = "Sequential Sampling"
    Random = "Random Sampling"
    Bayesian = "Bayesian Sampling"
    Exhaustive = "Exhaustive Sampling"


class AnalysisManager:
    """Analysis Manager for recording the tuning config and its corresponding evaluation result.
    
    This is for internal usage, the end user should not use this class directly.
    
    Examples:
        >>> analysis_manager = AnalysisManager()
        >>> base_sampler = BaseSampler(hp, analysis_manager=analysis_manager)
        >>> for dependence_sampler in base_sampler.dependence_samplers:
        >>>    dependence_sampler_analysis_result = analysis_manager.get_analysis_result(dependence_sampler)
        >>>    # update the base_sampler according to the dependence_sampler_analysis_result
        >>>    ...
    """
    #TODO

class Sampler:
    
    @abstractmethod
    def need_stop(self) -> bool:
        """Determine whether to stop sampling."""
        raise NotImplementedError

class BaseSampler:
    """Base class for all samplers.

    Each sampler has an `__iter__` method, which generate a list of configs.

    Examples:
        >>> hp = HyperParams("sq_alpha", {"alpha": [0.1, 0.2, 0.3]})
        >>> sampler = BaseSampler(hp)
        >>> for config in sampler:
        >>>     print(config)
    """

    def __init__(
        self,
        hp: HyperParams,
        name: str = "",
        max_trials = MAX_TRIALS_EACH_SAMPLER,
        priority: float = -1,
        sampling_algo = None,
        dependence_samplers: List = [],
        analysis_manager = None,
    ) -> None:
        """Initial an Sampler.

        Args:
            hp: the hyper parameter space to sample from.
            name: sampler name. Defaults to "".
            max_trials: the maximum number of trials. Defaults to MAX_TRIALS_EACH_SAMPLER.
            priority: the priority of the sampler, the sampler with higher priority will be sampled first. 
                Defaults to -1.
            sampling_algo: the sampling algorithm which determines the order of the sampling. Defaults to None.
            dependence_samplers: the samplers that this sampler depends on. Defaults to [].
            analysis_manager: the analysis manager for fetching the analysis results of dependence samplers and 
                register the analysis results of this sampler. Defaults to None.
        
        """
        self.hp = hp
        self.name = name
        self.max_trials = max_trials
        self.priority = priority
        self.sampling_algo = sampling_algo
        self._dependence_samplers: List[BaseSampler] = dependence_samplers
        self._analysis_manager = analysis_manager
        assert self._post_init_check(), "The sampler is not valid."
    
    def _post_init_check(self):
        return True
        
    @property
    def dependence_samplers(self):
        """Get the dependence samplers of this sampler."""
        return self._dependence_samplers

    @dependence_samplers.setter
    def dependence_samplers(self, dependence_samplers):
        """Set the dependence samplers of this sampler."""
        self._dependence_samplers = dependence_samplers
    
    def add_dependence_sampler(self, sampler):
        """Add a dependence sampler.
        
        For Example, the `accumulated fallback sampler` depends on the `one by one fallback sampler`.
        """
        self._dependence_samplers.append(sampler)

    def __iter__(self):
        """Generate a list of configs."""
        for config in self.hp.exapnd():
            yield config
            
    def need_stop(self, acc) -> bool:
        """Determine whether to stop sampling.

        The different sampler may have different stop criterion.
        TODO
        The different sampler may have different stop criterion.
        quant_level = 0, continue sampling even if the accuracy is not satisfied.
        quant_level = 1 with basic, stop sampling if the accuracy is satisfied.
        
        # exit policy
        1. not_tuning(performance_only): only quantize the model without tuning or evaluation.
        2. timeout = 0, exit the tuning process once it is found model meets the accuracy requirement.
        3. max_trials, the number of the actually trials is less or equal to the max_trials
        4. quant_level = 0, to seek the better performance, continue the tuning process even if the accuracy is satisfied.
        There are two ways to use max_trials to dominate the exit policy.
        1) timeout = 0, the tuning process exit when the actual_trails_count >= max_trials or
           a quantized model meets the accuracy requirements
        2) timeout = inf, the tuning process exit until the trials_count >= max_trials
        Some use case:
        1) Ending tuning process after a quantized model meets the accuracy requirements
           max_trials = inf, timeout = 0 (by default) # the default max_trials is 100
        value of timeout. max_trials control the exit policy
        2) Even after finding a model that meets the accuracy goal, we may want to continue the
           tuning process for better performance or other objectives.
           timeout = 100000 (seconds), max_trials = 10 # Specifics a fairly large timeout, use max_trials
                                             # to control the exit policy.
        3) Only want to try a certain number of trials
           timeout = 100000 (seconds), max_trials = 3 # only want to try the first 3 trials
        """
        return False

class GenericSampler(BaseSampler):
    """Generic sampler.
    
    Generic sampler to support very generic tuning like smooth quant's alpha.
    """
    pass

class OpWiseSampler(BaseSampler):
    """Op-wise sampler.
    
    It quantized the model with different quantization schemes for each op.
    """
    pass

class OpTywiseSampler(BaseSampler):
    """Op-type-wise sampler.
    
    It quantized the model with different quantization schemes for each op type.
    """
    pass


class FallbackSampler(BaseSampler):
    """Fallback sampler.

    Fallback op one by one.
    """

    def __init__(
        self,
        hp: HyperParams,
        name: str = "",
        max_trials=MAX_TRIALS_EACH_SAMPLER,
        priority: float = -1,
        sampling_algo=None,
        dependence_samplers: List = [],
        analysis_manager=None,
    ) -> None:
        super().__init__(
            hp,
            name,
            max_trials,
            priority,
            sampling_algo,
            dependence_samplers,
            analysis_manager,
        )

class AccumulatedFallbackSampler(BaseSampler):
    """Accumulated fallback sampler.

    Fallback ops accumulated.
    """

    def __init__(
        self,
        hp: HyperParams,
        name: str = "",
        max_trials=MAX_TRIALS_EACH_SAMPLER,
        priority: float = -1,
        sampling_algo=None,
        dependence_samplers: List = [],
        analysis_manager=None,
    ) -> None:
        super().__init__(
            hp,
            name,
            max_trials,
            priority,
            sampling_algo,
            dependence_samplers,
            analysis_manager,
        )
    
    def _post_init_check(self):
        has_add_fallback_sampler = any([isinstance(sampler, FallbackSampler) for sampler in self.dependence_samplers])
        return super()._post_init_check() and  has_add_fallback_sampler
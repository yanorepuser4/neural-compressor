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
from typing import List

from neural_compressor.common.strategy.search_space import HyperParams
from neural_compressor.common.strategy.utility import FakeTuningConfig

MAX_TRIALS_EACH_SAMPLER = 1_000_000


class SamplingAlgo(Enum):
    Sequential = "Sequential Sampling"
    Random = "Random Sampling"
    Bayesian = "Bayesian Sampling"
    Exhaustive = "Exhaustive Sampling"


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
        max_trials=MAX_TRIALS_EACH_SAMPLER,
        priority: float = -1,
        sampling_algo=None,
        dependence_samplers: List = [],
        analysis_manager=None,
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
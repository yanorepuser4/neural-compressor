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
    ) -> None:
        """Initial an Sampler.

        Args:
            hp: the hyper parameter space to sample from.
            name: sampler name. Defaults to "".
            max_trials: the maximum number of trials. Defaults to MAX_TRIALS_EACH_SAMPLER.
            priority: the priority of the sampler, the sampler with higher priority will be sampled first. Defaults to -1.
            sampling_algo: the sampling algorithm which determines the order of the sampling. Defaults to None.
        """
        self.hp = hp
        self.name = name
        self.max_trials = max_trials
        self.priority = priority
        self.sampling_algo = sampling_algo

    def __iter__(self):
        """Generate a list of configs."""
        for config in self.hp.exapnd():
            yield config

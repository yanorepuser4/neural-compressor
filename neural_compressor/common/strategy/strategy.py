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

from typing import List, Set
import numpy as np

from neural_compressor.common.strategy.sampler import BaseSampler
from neural_compressor.utils import logger
from neural_compressor.common.config import TuningConfig, AccuracyCriterion, TuningCriterion


class Strategy:
    def __init__(self, fp32_model, tuning_config) -> None:
        self.sampler_set: Set[BaseSampler] = set()

    def evaluate(self, eval_func, model):
        """Interface of evaluating model.
        #TODO adaptor needs pass a eval_func to tuner, if it call tuner.
        # It's adaptor's responsibility to create `eval_func` according to the `eval_metric` and `eval_dataloader`.
        Args:
            model (object): The model to be evaluated.

        Returns:
            Objective: The objective value evaluated.
        """
        # TODO 
        return 1
        val = self.objectives.evaluate(eval_func, model)
        return val

    def add_sampler(self, sampler: BaseSampler):
        self.sampler_set.add(sampler)

    def tuning_summary(self):
        """Show tuning summary."""
        logger.info("*" * 40)
        logger.info("There are {} tuning stages in total.".format(len(self.sampler_set)))
        for i, sampler in enumerate(self.sampler_set):
            logger.info("Stage {}: {}.".format(i, sampler.name))
        logger.info("*" * 40)

    def traverse(self, quantizer):
        self.tuning_summary()
        for cur_sampler in self.sampler_set:
            logger.info("Start tuning stage: {}".format(cur_sampler.name))
            for config in cur_sampler:
                q_model = quantizer.internal_quantize(config)
                acc = self.evaluate(q_model)
                if cur_sampler.need_stop(acc):
                    return q_model
            logger.info("*" * 40)

    def need_stop(self, acc) -> bool:
        """Determine whether to stop sampling.

        TODO (update it)
        1. should we move it into sampler class?
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

